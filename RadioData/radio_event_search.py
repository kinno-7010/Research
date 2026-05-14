#!/usr/bin/env python3
"""
radio_event_search.py

Integrate dynamic spectra from five radio instruments into one event-search plot.

Frequency assignment:
    Wind/WAVES RAD2 :   1--14 MHz https://spdf.gsfc.nasa.gov/pub/data/wind/waves/rad2_l2/
    Iitate HF       :  14--42 MHz http://adrastea.gp.tohoku.ac.jp/~jupiter/data/cdf-j/?C=N;O=D
    Australia-ASSA  :  42--160 MHz https://soleil.i4ds.ch/solarradio/data/2002-20yy_Callisto/
    IPRT            : 160--470 MHz http://radio.gp.tohoku.ac.jp/db/IPRT-SUN/DATA2/
    Yamagawa        : 470--1000 MHz https://solobs.nict.go.jp/radio/cgi-bin/MainDisplay.pl

Yamagawa data is manually downloaded from the download page due to security reasons of website.
YAMAGAWA_<YYYYMMDDHH>I.fits is the expected file name.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from astropy.io import fits
from cdflib import CDF, cdfepoch
import gzip
import shutil
import urllib.error
import urllib.request

# =============================================================================
# User-editable default configuration
# =============================================================================

WIND_RAW_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/Wind/Rawdata")
HF_RAW_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/HF_plot/Rawdata")
ASSA_RAW_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/e-Callisto/Rawdata")
IPRT_RAW_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/IPRT/Rawdata")
YAMAGAWA_RAW_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/Yamagawa/Rawdata")

DEFAULT_OUTPUT_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/combine")


# =============================================================================
# Data containers
# =============================================================================

@dataclass(frozen=True)
class InstrumentSpec:
    """Definition of one frequency block in the stitched spectrum."""

    name: str
    fmin_mhz: float
    fmax_mhz: float
    fallback_nfreq: int


@dataclass
class FileCheckResult:
    """File availability report for one instrument."""

    instrument: str
    existing_files: list[Path]
    missing_expected_files: list[Path]
    notes: list[str]


INSTRUMENTS: tuple[InstrumentSpec, ...] = (
    InstrumentSpec("Wind/RAD2", 1.0, 14.0, 64),
    InstrumentSpec("HF antenna", 14.0, 40.0, 96),
    InstrumentSpec("Australia-ASSA", 40.0, 150.0, 128),
    InstrumentSpec("IPRT", 150.0, 470.0, 256),
    InstrumentSpec("Yamagawa", 470.0, 1000.0, 256),
)


# =============================================================================
# General utilities
# =============================================================================

def normalize_path(path_like: str | Path) -> Path:
    """
    Convert Windows-style and mixed-separator paths to a Path usable under WSL.

    Examples
    --------
    D:\\wsl\\home\\... -> /mnt/d/wsl/home/...
    /mnt/d/wsl\\home\\... -> /mnt/d/wsl/home/...
    """
    text = str(path_like).replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", text)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2).lstrip("/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(text)

# =============================================================================
# Download utilities
# =============================================================================

DOWNLOAD_TIMEOUT_SECONDS = 90
ASSA_REMOTE_ROOT = "https://soleil.i4ds.ch/solarradio/data/2002-20yy_Callisto"


def download_url_if_missing(
    url: str,
    output_path: Path,
    timeout: int = DOWNLOAD_TIMEOUT_SECONDS,
) -> bool:
    """Download a URL to output_path only when output_path does not exist.

    Returns True when a new file was downloaded, and False when the file already
    existed or the download failed. A temporary .part file is used so that a
    failed download does not leave a corrupt final file.
    """
    output_path = normalize_path(output_path)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[download skip] exists: {output_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".part")

    print(f"[download] {url}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with open(temporary_path, "wb") as file_obj:
                shutil.copyfileobj(response, file_obj)
        temporary_path.replace(output_path)
        print(f"[download saved] {output_path}")
        return True
    except urllib.error.HTTPError as exc:
        print(f"[download missing] {url} ({exc.code})")
    except urllib.error.URLError as exc:
        print(f"[download failed] {url} ({exc})")
    except TimeoutError as exc:
        print(f"[download timeout] {url} ({exc})")
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return False


def download_gzip_fit_if_missing(
    url: str,
    output_fit_path: Path,
    timeout: int = DOWNLOAD_TIMEOUT_SECONDS,
    keep_gzip: bool = False,
) -> bool:
    """Download *.fit.gz and extract it to *.fit if the FITS file is absent."""
    output_fit_path = normalize_path(output_fit_path)
    if output_fit_path.exists() and output_fit_path.stat().st_size > 0:
        print(f"[download skip] exists: {output_fit_path}")
        return False

    output_fit_path.parent.mkdir(parents=True, exist_ok=True)
    gzip_path = output_fit_path.with_name(output_fit_path.name + ".gz")

    if not gzip_path.exists() or gzip_path.stat().st_size == 0:
        ok = download_url_if_missing(url, gzip_path, timeout=timeout)
        if not ok and (not gzip_path.exists() or gzip_path.stat().st_size == 0):
            return False

    temporary_fit_path = output_fit_path.with_name(output_fit_path.name + ".part")
    try:
        with gzip.open(gzip_path, "rb") as gz_obj:
            with open(temporary_fit_path, "wb") as fit_obj:
                shutil.copyfileobj(gz_obj, fit_obj)
        temporary_fit_path.replace(output_fit_path)
        print(f"[gzip extracted] {output_fit_path}")
        if not keep_gzip and gzip_path.exists():
            gzip_path.unlink()
        return True
    except OSError as exc:
        print(f"[gzip failed] {gzip_path} ({exc})")
    finally:
        if temporary_fit_path.exists():
            temporary_fit_path.unlink()

    return False


def read_remote_text(url: str, timeout: int = DOWNLOAD_TIMEOUT_SECONDS) -> str | None:
    """Read a remote directory listing as text."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        print(f"[remote listing failed] {url} ({exc})")
        return None


def assa_remote_directory_url(day: pd.Timestamp) -> str:
    """Return the e-CALLISTO remote directory for one UT date."""
    return (
        f"{ASSA_REMOTE_ROOT}/"
        f"{day.strftime('%Y')}/{day.strftime('%m')}/{day.strftime('%d')}/"
    )


def assa_filename_time_range(filename: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Infer an approximate time interval from an ASSA filename."""
    match = re.match(r"^Australia-ASSA_(\d{8})_(\d{6})_\d{2}\.fit(?:\.gz)?$", filename)
    if not match:
        return None
    date_code, time_code = match.groups()
    start = pd.Timestamp(f"{date_code} {time_code}")
    end = start + pd.Timedelta(minutes=20)
    return start, end


def list_remote_assa_filenames(
    day: pd.Timestamp,
    focuscodes: Sequence[str],
) -> list[str]:
    """List remote Australia-ASSA *.fit.gz files for one date and focuscode set."""
    directory_url = assa_remote_directory_url(day)
    html = read_remote_text(directory_url)
    if html is None:
        return []

    date_code = day.strftime("%Y%m%d")
    code_pattern = "|".join(re.escape(str(code).zfill(2)) for code in focuscodes)
    pattern = re.compile(
        rf"Australia-ASSA_{date_code}_[0-9]{{6}}_(?:{code_pattern})\.fit\.gz"
    )
    return sorted(set(pattern.findall(html)))


def download_missing_wind_files(start_time: pd.Timestamp, end_time: pd.Timestamp) -> None:
    """Download missing Wind/WAVES RAD2 daily CDF files."""
    for path in expected_wind_paths(start_time, end_time):
        if path.exists() and path.stat().st_size > 0:
            continue
        match = re.search(r"(\d{8})", path.name)
        if match is None:
            continue
        date_code = match.group(1)
        year = date_code[:4]
        url = f"https://spdf.gsfc.nasa.gov/pub/data/wind/waves/rad2_l2/{year}/{path.name}"
        download_url_if_missing(url, path)


def download_missing_hf_files(start_time: pd.Timestamp, end_time: pd.Timestamp) -> None:
    """Download missing Iitate HF daily CDF files."""
    for path in expected_hf_paths(start_time, end_time):
        if path.exists() and path.stat().st_size > 0:
            continue
        url = f"http://adrastea.gp.tohoku.ac.jp/~jupiter/data/cdf-j/{path.name}"
        download_url_if_missing(url, path)


def download_missing_iprt_files(start_time: pd.Timestamp, end_time: pd.Timestamp) -> None:
    """Download missing IPRT daily FITS files."""
    for path in expected_iprt_paths(start_time, end_time):
        if path.exists() and path.stat().st_size > 0:
            continue
        match = re.search(r"(\d{8})", path.name)
        if match is None:
            continue
        date_code = match.group(1)
        year = date_code[:4]
        url = f"http://radio.gp.tohoku.ac.jp/db/IPRT-SUN/DATA2/{year}/{path.name}"
        download_url_if_missing(url, path)


def download_missing_assa_files(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    focuscodes: Sequence[str],
) -> None:
    """Download and extract missing Australia-ASSA e-CALLISTO FITS files."""
    for day in iter_dates(start_time, end_time):
        directory_url = assa_remote_directory_url(day)
        filenames = list_remote_assa_filenames(day, focuscodes)
        if not filenames:
            print(f"[ASSA] no remote listing match for {day.strftime('%Y-%m-%d')} focuscodes={tuple(focuscodes)}")
            continue

        for filename_gz in filenames:
            approx_range = assa_filename_time_range(filename_gz)
            if approx_range is None:
                continue
            file_start, file_end = approx_range
            if not overlaps(file_start, file_end, start_time, end_time):
                continue

            filename_fit = filename_gz[:-3]
            output_fit_path = ASSA_RAW_DIR / filename_fit
            if output_fit_path.exists() and output_fit_path.stat().st_size > 0:
                continue

            url = directory_url + filename_gz
            download_gzip_fit_if_missing(url, output_fit_path)


def download_missing_input_files(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    assa_focuscodes: Sequence[str],
    download_wind: bool = True,
    download_hf: bool = True,
    download_assa: bool = True,
    download_iprt: bool = True,
) -> None:
    """Download missing input files for all supported automatic sources.

    Yamagawa is intentionally excluded because its download page requires values
    calculated interactively on the website. Keep downloading that dataset
    manually and place files under YAMAGAWA_RAW_DIR.
    """
    print("\n================ missing-file download ================")
    if download_wind:
        download_missing_wind_files(start_time, end_time)
    if download_hf:
        download_missing_hf_files(start_time, end_time)
    if download_assa:
        download_missing_assa_files(start_time, end_time, assa_focuscodes)
    if download_iprt:
        download_missing_iprt_files(start_time, end_time)
    print("[Yamagawa] automatic download is skipped; download manually if needed.")
    print("=======================================================\n")

def iter_dates(start_time: pd.Timestamp, end_time: pd.Timestamp) -> list[pd.Timestamp]:
    """Return calendar dates touched by [start_time, end_time]."""
    start_date = pd.Timestamp(start_time).normalize()
    end_date = pd.Timestamp(end_time).normalize()
    return list(pd.date_range(start=start_date, end=end_date, freq="D"))


def iter_hours(start_time: pd.Timestamp, end_time: pd.Timestamp) -> list[pd.Timestamp]:
    """Return hour stamps touched by [start_time, end_time]."""
    start_hour = pd.Timestamp(start_time).floor("h")
    end_hour = pd.Timestamp(end_time).floor("h")
    return list(pd.date_range(start=start_hour, end=end_hour, freq="h"))


def finite_time_range(
    times: pd.DatetimeIndex,
    default_start: pd.Timestamp | None = None,
    default_end: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return a safe time range for a time axis."""
    if len(times) == 0:
        if default_start is None or default_end is None:
            raise ValueError("Cannot infer time range from an empty time axis.")
        return default_start, default_end
    return pd.Timestamp(times[0]), pd.Timestamp(times[-1])


def overlaps(
    file_start: pd.Timestamp,
    file_end: pd.Timestamp,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> bool:
    """True if two closed time intervals overlap."""
    return file_end >= requested_start and file_start <= requested_end


def frequency_to_mhz(freq: np.ndarray) -> np.ndarray:
    """
    Convert a frequency array to MHz if it appears to be in Hz.

    Many CDF files store Hz, while several FITS products in this workflow store MHz.
    The heuristic is intentionally conservative: values above 1e5 are treated as Hz.
    """
    freq = np.asarray(freq, dtype=float)
    finite = freq[np.isfinite(freq)]
    if finite.size and np.nanmedian(np.abs(finite)) > 1.0e5:
        return freq / 1.0e6
    return freq


def collapse_duplicate_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """Average columns that share the same frequency label."""
    if df.empty:
        return df
    collapsed = df.T.groupby(level=0).mean().T
    collapsed.columns.name = "frequency_mhz"
    return collapsed


def dataframe_from_arrays(
    times: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    freqs_mhz: Sequence[float] | np.ndarray,
    values: np.ndarray,
) -> pd.DataFrame:
    """
    Build a DataFrame with index=time and columns=frequency.

    The function accepts either (time, frequency) or (frequency, time) arrays and
    transposes automatically when the dimensions identify the orientation.
    """
    time_index = pd.DatetimeIndex(pd.to_datetime(times))
    freq_array = np.asarray(freqs_mhz, dtype=float)
    data = np.asarray(values, dtype=float)

    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D array after polarization selection, got shape={data.shape}")

    if data.shape == (len(time_index), len(freq_array)):
        data_tf = data
    elif data.shape == (len(freq_array), len(time_index)):
        data_tf = data.T
    else:
        raise ValueError(
            "Cannot identify data orientation: "
            f"data shape={data.shape}, n_time={len(time_index)}, n_freq={len(freq_array)}"
        )

    freq_index = pd.Index(np.round(freq_array.astype(float), 6), name="frequency_mhz")
    df = pd.DataFrame(data_tf, index=time_index, columns=freq_index)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_index()
    return collapse_duplicate_frequencies(df)


def combine_time_rows(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate time blocks and average duplicate timestamps."""
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame()
    combined = pd.concat(usable, axis=0).sort_index()
    combined = combined.groupby(combined.index).mean()
    return collapse_duplicate_frequencies(combined)


def clip_frequency(df: pd.DataFrame, fmin_mhz: float, fmax_mhz: float) -> pd.DataFrame:
    """Restrict columns to [fmin_mhz, fmax_mhz]."""
    if df.empty:
        return df
    freqs = df.columns.astype(float)
    mask = (freqs >= fmin_mhz) & (freqs <= fmax_mhz)
    return df.loc[:, mask]


def trim_time(df: pd.DataFrame, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
    """Restrict rows to [start_time, end_time]."""
    if df.empty:
        return df
    return df.loc[(df.index >= start_time) & (df.index <= end_time)]


def record_bounds_for_time_range(
    times: pd.DatetimeIndex,
    start_time: pd.Timestamp | None,
    end_time: pd.Timestamp | None,
) -> tuple[int, int] | None:
    """Return inclusive record bounds overlapping the requested time range."""
    if len(times) == 0:
        return None
    time_index = pd.DatetimeIndex(times).sort_values()
    start = pd.Timestamp(start_time) if start_time is not None else time_index[0]
    end = pd.Timestamp(end_time) if end_time is not None else time_index[-1]
    left = int(np.searchsorted(time_index.values, np.datetime64(start), side="left"))
    right = int(np.searchsorted(time_index.values, np.datetime64(end), side="right")) - 1
    if left >= len(time_index) or right < 0 or left > right:
        return None
    return max(left, 0), min(right, len(time_index) - 1)

def get_cdf_variable_names(cdf_obj: CDF) -> list[str]:
    """Return all variable names contained in a CDF object."""
    info = cdf_obj.cdf_info()
    return list(getattr(info, "zVariables", [])) + list(getattr(info, "rVariables", []))


def choose_hf_intensity_variable(cdf_obj: CDF, requested: str = "RH") -> str:
    """Choose the actual HF intensity variable present in the CDF file.

    Older/local Iitate HF files may contain variables named RH/LH, whereas the
    2026 file distributed from the archive contains CH1/CH2. This resolver
    keeps the public interface as RH/LH but falls back to CH1/CH2 when needed.
    """
    variables = get_cdf_variable_names(cdf_obj)
    variable_set = set(variables)
    requested_upper = str(requested).upper()

    aliases = {
        "RH": ["RH", "R", "CH1"],
        "LH": ["LH", "L", "CH2"],
        "CH1": ["CH1", "RH", "R"],
        "CH2": ["CH2", "LH", "L"],
    }

    for candidate in aliases.get(requested_upper, [requested_upper]):
        if candidate in variable_set:
            if candidate != requested_upper:
                warnings.warn(
                    f"HF variable '{requested_upper}' was not found; "
                    f"using '{candidate}' instead. Available variables: {variables}"
                )
            return candidate

    data_candidates = [name for name in variables if re.fullmatch(r"CH\d+", name)]
    if data_candidates:
        selected = sorted(data_candidates)[0]
        warnings.warn(
            f"HF variable '{requested_upper}' was not found; using '{selected}' instead. "
            f"Available variables: {variables}"
        )
        return selected

    raise KeyError(
        f"No usable HF intensity variable found. Requested '{requested_upper}', "
        f"available variables are {variables}"
    )


def apply_cdf_fill_and_valid_range(
    cdf_obj: CDF,
    variable_name: str,
    values: np.ndarray,
) -> np.ndarray:
    """Replace CDF fill values and values outside declared valid range with NaN."""
    data = np.asarray(values, dtype=float)

    try:
        attrs = cdf_obj.varattsget(variable_name)
    except Exception:
        return data

    fill_value = attrs.get("FILLVAL")
    if fill_value is not None:
        fill_float = float(np.asarray(fill_value).ravel()[0])
        data[np.isclose(data, fill_float)] = np.nan

    valid_min = attrs.get("VALIDMIN")
    valid_max = attrs.get("VALIDMAX")

    if valid_min is not None:
        vmin = float(np.asarray(valid_min).ravel()[0])
        data[data < vmin] = np.nan

    if valid_max is not None:
        vmax = float(np.asarray(valid_max).ravel()[0])

        # For uint8 HF products, the archive occasionally contains values above
        # the nominal VALIDMAX during strong/contaminated intervals. Do not
        # discard them; only the explicit FILLVAL is treated as missing.
        if variable_name.upper() not in {"CH1", "CH2"}:
            data[data > vmax] = np.nan

    return data


def report_time_gaps(
    instrument_name: str,
    times: pd.DatetimeIndex,
    native_dt: pd.Timedelta | None,
    max_gap_factor: float = 2.5,
    max_segments: int = 8,
) -> None:
    """Print large native-time gaps that will remain white/NaN in the plot."""
    if native_dt is None or len(times) < 2:
        return

    sorted_times = pd.DatetimeIndex(times).sort_values().unique()
    diffs = sorted_times[1:] - sorted_times[:-1]
    threshold = native_dt * max_gap_factor
    large = np.where(diffs > threshold)[0]

    if large.size == 0:
        return

    print(
        f"{instrument_name}: {large.size} native time gap(s) larger than "
        f"{format_resolution(threshold, 's')} will remain NaN/white."
    )

    for idx in large[:max_segments]:
        print(
            f"  gap: {pd.Timestamp(sorted_times[idx])} -> "
            f"{pd.Timestamp(sorted_times[idx + 1])} "
            f"({pd.Timedelta(diffs[idx]).total_seconds():.3f} s)"
        )

    if large.size > max_segments:
        print(f"  ... {large.size - max_segments} more gap(s)")

def resample_to_common_grid(
    df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    cadence: str,
    native_dt: pd.Timedelta | None = None,
    interpolate_short_gaps: bool = True,
    max_gap_factor: float = 2.5,
) -> pd.DataFrame:
    """Resample data to the common time grid and preserve original NaNs.

    Short empty bins produced only by upsampling from a coarser native cadence
    can be interpolated for display continuity. However, bins that actually
    contain an original sample whose value is NaN are forced back to NaN after
    interpolation. Therefore, original bad/missing data remain white in the
    final plot.
    """
    if df.empty:
        return df.reindex(target_index)

    resampled = df.resample(cadence, origin=target_index[0]).mean()
    aligned = resampled.reindex(target_index)

    # Presence is independent of whether the physical value is finite.
    # This distinguishes "there was an original record but it was NaN"
    # from "there was no record because the native cadence is coarser
    # than the common grid".
    sample_presence = pd.DataFrame(
        1.0,
        index=df.index,
        columns=df.columns,
    )
    presence_resampled = sample_presence.resample(cadence, origin=target_index[0]).max()
    presence_aligned = presence_resampled.reindex(target_index).fillna(0.0) > 0.0
    original_nan_mask = presence_aligned & aligned.isna()

    if not interpolate_short_gaps:
        return aligned.mask(original_nan_mask)

    if native_dt is None:
        native_dt = estimate_time_resolution(df.index)
    if native_dt is None or pd.isna(native_dt) or native_dt <= pd.Timedelta(0):
        return aligned.mask(original_nan_mask)

    grid_dt = pd.to_timedelta(cadence)
    if grid_dt <= pd.Timedelta(0):
        return aligned.mask(original_nan_mask)

    max_gap = native_dt * max_gap_factor
    limit_bins = int(np.ceil(max_gap / grid_dt))
    if limit_bins <= 0:
        return aligned.mask(original_nan_mask)

    interpolated = aligned.interpolate(
        method="time",
        limit=limit_bins,
        limit_area="inside",
    )

    return interpolated.mask(original_nan_mask)

def normalize_by_frequency_median(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each frequency bin by its median over the selected interval."""
    if df.empty:
        return df
    medians = df.median(axis=0, skipna=True)
    medians = medians.replace(0.0, np.nan)
    return df.divide(medians, axis=1)


def timedelta_to_seconds(delta: pd.Timedelta | None) -> float | None:
    """Convert a pandas Timedelta-like value to seconds."""
    if delta is None or pd.isna(delta):
        return None
    return float(pd.Timedelta(delta).total_seconds())


def estimate_time_resolution(times: pd.DatetimeIndex) -> pd.Timedelta | None:
    """Estimate the native time cadence from a DatetimeIndex."""
    if len(times) < 2:
        return None
    series = pd.Series(pd.DatetimeIndex(times).sort_values().unique())
    diffs = series.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None
    return pd.Timedelta(diffs.median())


def estimate_frequency_resolution(freqs_mhz: Sequence[float] | np.ndarray) -> float | None:
    """Estimate the native frequency spacing in MHz."""
    freqs = np.asarray(freqs_mhz, dtype=float)
    freqs = np.unique(np.round(freqs[np.isfinite(freqs)], 6))
    if freqs.size < 2:
        return None
    diffs = np.diff(np.sort(freqs))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return float(np.nanmedian(diffs))


def format_resolution(value: float | pd.Timedelta | None, unit: str) -> str:
    """Format a time or frequency resolution for diagnostic output."""
    if value is None:
        return "unknown"
    if isinstance(value, pd.Timedelta):
        seconds = value.total_seconds()
        return f"{seconds:.6g} s"
    return f"{float(value):.6g} {unit}"


def build_completed_frequency_axis(
    actual_freqs_mhz: Sequence[float] | np.ndarray,
    fmin_mhz: float,
    fmax_mhz: float,
    fallback_nfreq: int,
) -> np.ndarray:
    """
    Build a frequency axis covering the full assigned instrumental band.

    The axis preserves the native observed frequencies and extends uncovered
    portions of the assigned band with NaN-ready columns. This prevents a
    plotting artifact in which pcolormesh visually stretches the last observed
    channel across a large unobserved frequency gap.
    """
    actual = np.asarray(actual_freqs_mhz, dtype=float)
    actual = actual[np.isfinite(actual)]
    actual = actual[(actual >= fmin_mhz) & (actual <= fmax_mhz)]
    actual = np.unique(np.round(actual, 6))
    actual.sort()

    if actual.size == 0:
        return np.round(np.geomspace(fmin_mhz, fmax_mhz, fallback_nfreq), 6)

    step = estimate_frequency_resolution(actual)
    if step is None or not np.isfinite(step) or step <= 0:
        step = (fmax_mhz - fmin_mhz) / max(fallback_nfreq - 1, 1)

    lower: list[float] = []
    value = actual[0] - step
    while value >= fmin_mhz:
        lower.append(value)
        value -= step

    upper: list[float] = []
    value = actual[-1] + step
    while value <= fmax_mhz:
        upper.append(value)
        value += step

    # Add exact boundaries only when a sizable uncovered edge would otherwise
    # remain. Small half-bin differences are left untouched to avoid creating
    # artificial ultra-narrow cells at the band boundary.
    if actual[0] - fmin_mhz > 0.51 * step:
        lower.append(fmin_mhz)
    if fmax_mhz - actual[-1] > 0.51 * step:
        upper.append(fmax_mhz)

    completed = np.concatenate([
        np.asarray(lower, dtype=float),
        actual,
        np.asarray(upper, dtype=float),
    ])
    completed = np.unique(np.round(completed, 6))
    completed = completed[(completed >= fmin_mhz) & (completed <= fmax_mhz)]
    completed.sort()
    return completed


def complete_frequency_coverage(df: pd.DataFrame, spec: InstrumentSpec) -> pd.DataFrame:
    """
    Reindex an instrument frame onto a full native-resolution assigned band.

    Frequencies not actually observed by the file remain NaN. For example, the
    attached Australia-ASSA file covers about 15--86.94 MHz, so its assigned
    42--160 MHz panel keeps 42--86.94 MHz data and fills roughly 87--160 MHz
    with NaN columns.
    """
    if df.empty:
        return df

    full_freqs = build_completed_frequency_axis(
        actual_freqs_mhz=df.columns.astype(float).to_numpy(),
        fmin_mhz=spec.fmin_mhz,
        fmax_mhz=spec.fmax_mhz,
        fallback_nfreq=spec.fallback_nfreq,
    )
    full_columns = pd.Index(full_freqs, name="frequency_mhz")
    return df.reindex(columns=full_columns)


def nan_frame(
    target_index: pd.DatetimeIndex,
    fmin_mhz: float,
    fmax_mhz: float,
    nfreq: int,
) -> pd.DataFrame:
    """Create a NaN-only frame for a missing instrument."""
    freqs = np.geomspace(fmin_mhz, fmax_mhz, nfreq)
    freqs = np.round(freqs, 6)
    return pd.DataFrame(np.nan, index=target_index, columns=pd.Index(freqs, name="frequency_mhz"))


# =============================================================================
# File discovery
# =============================================================================

def expected_wind_paths(start_time: pd.Timestamp, end_time: pd.Timestamp) -> list[Path]:
    """Expected Wind/RAD2 daily CDF files."""
    return [
        WIND_RAW_DIR / f"wi_l2_wav_rad2_{day.strftime('%Y%m%d')}_v01.cdf"
        for day in iter_dates(start_time, end_time)
    ]


def expected_hf_paths(start_time: pd.Timestamp, end_time: pd.Timestamp) -> list[Path]:
    """Expected Iitate HF daily CDF files."""
    return [
        HF_RAW_DIR / f"it_h1_hf_{day.strftime('%Y%m%d')}_v01.cdf"
        for day in iter_dates(start_time, end_time)
    ]


def expected_iprt_paths(start_time: pd.Timestamp, end_time: pd.Timestamp) -> list[Path]:
    """Expected IPRT daily FITS files."""
    return [
        IPRT_RAW_DIR / f"{day.strftime('%Y%m%d')}_IPRT.fits"
        for day in iter_dates(start_time, end_time)
    ]


def expected_yamagawa_paths(start_time: pd.Timestamp, end_time: pd.Timestamp) -> list[Path]:
    """Expected hourly Yamagawa FITS files."""
    return [
        YAMAGAWA_RAW_DIR / f"YAMAGAWA_{hour.strftime('%Y%m%d%H')}I.fits"
        for hour in iter_hours(start_time, end_time)
    ]


def iprt_timerange(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Infer the time range of one IPRT FITS file."""
    with fits.open(path) as hdul:
        header = hdul[0].header
        start = pd.Timestamp(f"{header['DATE-OBS']} {header['TIME-OBS']}")
        n_time = int(header["NAXIS1"])

        if "DATE-END" in header and "TIME-END" in header:
            end = pd.Timestamp(f"{header['DATE-END']} {header['TIME-END']}")
        else:
            dt_seconds = float(header.get("CDELT1", 1.0))
            end = start + pd.to_timedelta(dt_seconds * max(n_time - 1, 0), unit="s")

    return start, end


def callisto_timerange(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Infer the time range of one e-Callisto FITS file."""
    with fits.open(path) as hdul:
        primary = hdul[0]
        calibration = hdul[1].data[0]
        start = pd.Timestamp(f"{primary.header['DATE-OBS']} {primary.header['TIME-OBS']}")
        offsets = np.asarray(calibration[0], dtype=float)
        if offsets.size == 0:
            return start, start
        end = start + pd.to_timedelta(float(np.nanmax(offsets)), unit="s")
    return start, end


def yamagawa_timerange(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Infer the time range of one Yamagawa FITS file."""
    with fits.open(path) as hdul:
        header = hdul[0].header
        start = pd.Timestamp(f"{header['DATE-OBS']} {header['TIME-OBS']}")
        n_time = int(header["NAXIS1"])
        dt_seconds = float(header.get("CDELT1", 1.0))
        end = start + pd.to_timedelta(dt_seconds * max(n_time - 1, 0), unit="s")
    return start, end


def find_assa_paths(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    focuscodes: Sequence[str] = ("62", "56"),
) -> list[Path]:
    """Find Australia-ASSA e-Callisto files overlapping the requested interval.

    The suffix after the final underscore is the e-Callisto focuscode.  For
    Australia-ASSA, the useful choices for this combined plot are typically:

        62/63 : Phase 2.5 low-frequency system, about 15--88 MHz
        56/57 : Phase 3 high-frequency system, about 108--370 MHz

    The default uses one representative low-frequency and one representative
    high-frequency channel, avoiding automatic averaging of the two polarization
    partners.  Set focuscodes=("62", "63", "56", "57") in main() if you want
    all available files to be combined.
    """
    requested_codes = {str(code).zfill(2) for code in focuscodes}
    pattern = re.compile(r"^Australia-ASSA_(\d{8})_(\d{6})_(\d{2})\.fit$")
    paths: list[Path] = []

    if not ASSA_RAW_DIR.exists():
        return paths

    valid_dates = {day.strftime("%Y%m%d") for day in iter_dates(start_time, end_time)}
    for path in sorted(ASSA_RAW_DIR.iterdir()):
        match = pattern.match(path.name)
        if not match:
            continue

        date_code, time_code, focuscode = match.groups()
        if date_code not in valid_dates or focuscode not in requested_codes:
            continue

        try:
            file_start, file_end = callisto_timerange(path)
        except Exception as exc:
            # e-Callisto files are normally 15-minute blocks. Use the filename
            # timestamp as a conservative fallback if the FITS calibration table
            # cannot be read.
            file_start = pd.Timestamp(f"{date_code} {time_code}")
            file_end = file_start + pd.Timedelta(minutes=20)
            warnings.warn(f"Could not read ASSA header for {path.name}: {exc}")

        if overlaps(file_start, file_end, start_time, end_time):
            paths.append(path)

    return paths


def find_iprt_paths(start_time: pd.Timestamp, end_time: pd.Timestamp) -> list[Path]:
    """Find IPRT FITS files overlapping the requested interval."""
    pattern = re.compile(r"^\d{8}_IPRT\.fits$")
    paths: list[Path] = []

    if not IPRT_RAW_DIR.exists():
        return paths

    for path in sorted(IPRT_RAW_DIR.iterdir()):
        if not pattern.match(path.name):
            continue
        try:
            file_start, file_end = iprt_timerange(path)
        except Exception as exc:
            warnings.warn(f"Could not read IPRT header for {path.name}: {exc}")
            continue
        if overlaps(file_start, file_end, start_time, end_time):
            paths.append(path)

    return paths


def find_yamagawa_paths(start_time: pd.Timestamp, end_time: pd.Timestamp) -> list[Path]:
    """Find Yamagawa FITS files overlapping the requested interval."""
    # The official files are YAMAGAWA_YYYYMMDDHHI.fits.  The relaxed suffix is
    # intentional so that copied files such as YAMAGAWA_YYYYMMDDHHI(1).fits are
    # still found during local tests.
    pattern = re.compile(r"^YAMAGAWA_(\d{10})I.*\.fits$")
    paths: list[Path] = []

    if not YAMAGAWA_RAW_DIR.exists():
        return paths

    candidate_hours = {hour.strftime("%Y%m%d%H") for hour in iter_hours(start_time, end_time)}
    for path in sorted(YAMAGAWA_RAW_DIR.iterdir()):
        match = pattern.match(path.name)
        if not match or match.group(1) not in candidate_hours:
            continue
        try:
            file_start, file_end = yamagawa_timerange(path)
        except Exception as exc:
            warnings.warn(f"Could not read Yamagawa header for {path.name}: {exc}")
            continue
        if overlaps(file_start, file_end, start_time, end_time):
            paths.append(path)

    return paths


def check_daily_files(
    instrument: str,
    expected_paths: Sequence[Path],
) -> FileCheckResult:
    """Check existence of expected daily/hourly files."""
    existing = [path for path in expected_paths if path.exists()]
    missing = [path for path in expected_paths if not path.exists()]
    notes = []
    if missing:
        notes.append(f"{len(missing)} expected file(s) missing.")
    else:
        notes.append("All expected file(s) exist.")
    return FileCheckResult(instrument, existing, missing, notes)


def check_assa_files(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    focuscodes: Sequence[str] = ("62", "56"),
) -> FileCheckResult:
    """Check overlapping Australia-ASSA FITS files for selected focuscodes."""
    existing = find_assa_paths(start_time, end_time, focuscodes=focuscodes)
    missing: list[Path] = []
    notes = []
    code_text = ",".join(str(code).zfill(2) for code in focuscodes)
    if existing:
        notes.append(f"{len(existing)} overlapping file(s) found for focuscode(s) {code_text}.")
    else:
        notes.append(f"No overlapping Australia-ASSA file found for focuscode(s) {code_text}.")
    return FileCheckResult("Australia-ASSA", existing, missing, notes)


def check_iprt_files(start_time: pd.Timestamp, end_time: pd.Timestamp) -> FileCheckResult:
    """Check overlapping IPRT FITS files."""
    existing = find_iprt_paths(start_time, end_time)
    expected = expected_iprt_paths(start_time, end_time)
    expected_missing = [path for path in expected if not path.exists()]
    notes = []
    if existing:
        notes.append(f"{len(existing)} overlapping file(s) found.")
    else:
        notes.append("No overlapping IPRT file found.")
    return FileCheckResult("IPRT", existing, expected_missing, notes)


def check_yamagawa_files(start_time: pd.Timestamp, end_time: pd.Timestamp) -> FileCheckResult:
    """Check overlapping Yamagawa FITS files."""
    existing = find_yamagawa_paths(start_time, end_time)
    expected = expected_yamagawa_paths(start_time, end_time)
    expected_missing = [path for path in expected if not path.exists()]
    notes = []
    if existing:
        notes.append(f"{len(existing)} overlapping file(s) found.")
    else:
        notes.append("No overlapping Yamagawa file found.")
    return FileCheckResult("Yamagawa", existing, expected_missing, notes)


def check_all_files(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    assa_focuscodes: Sequence[str] = ("62", "56"),
) -> dict[str, FileCheckResult]:
    """Check input files for all instruments."""
    checks = {
        "Wind/RAD2": check_daily_files("Wind/RAD2", expected_wind_paths(start_time, end_time)),
        "HF antenna": check_daily_files("HF antenna", expected_hf_paths(start_time, end_time)),
        "Australia-ASSA": check_assa_files(start_time, end_time, focuscodes=assa_focuscodes),
        "IPRT": check_iprt_files(start_time, end_time),
        "Yamagawa": check_yamagawa_files(start_time, end_time),
    }
    return checks


def print_file_report(checks: dict[str, FileCheckResult]) -> None:
    """Print a concise availability report."""
    print("\n================ file availability ================")
    for name, result in checks.items():
        print(f"[{name}]")
        for note in result.notes:
            print(f"  - {note}")
        if result.existing_files:
            for path in result.existing_files:
                print(f"  + {path}")
        if result.missing_expected_files:
            for path in result.missing_expected_files:
                print(f"  - MISSING: {path}")
    print("===================================================\n")


# =============================================================================
# Instrument-specific readers
# =============================================================================

def load_wind_rad2_file(
    path: Path,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read one Wind/WAVES RAD2 CDF file."""
    cdf_obj = CDF(str(path))
    try:
        epoch_all = cdf_obj.varget("Epoch")
        all_times = pd.to_datetime(cdfepoch.to_datetime(epoch_all))
        bounds = record_bounds_for_time_range(pd.DatetimeIndex(all_times), start_time, end_time)
        if bounds is None:
            return pd.DataFrame()
        startrec, endrec = bounds

        epoch = cdf_obj.varget("Epoch", startrec=startrec, endrec=endrec)
        frequency_hz = np.asarray(cdf_obj.varget("FREQUENCY"), dtype=float)
        psd = np.asarray(cdf_obj.varget("PSD_V2_S", startrec=startrec, endrec=endrec), dtype=float)
        background = np.asarray(cdf_obj.varget("BACKGROUND_S"), dtype=float)
    finally:
        try:
            cdf_obj.close()
        except Exception:
            pass

    freq_mhz = frequency_hz / 1.0e6
    times = pd.to_datetime(cdfepoch.to_datetime(epoch))

    with np.errstate(divide="ignore", invalid="ignore"):
        intensity = np.divide(
            psd,
            background,
            out=np.full_like(psd, np.nan, dtype=float),
            where=background > 0,
        )

    return dataframe_from_arrays(times, freq_mhz, intensity)


def load_hf_file(
    path: Path,
    polarization: str = "RH",
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read one Iitate HF CDF file.

    The HF archive is not completely uniform in variable naming. Some files use
    RH/LH, while the 2026-04-24 file uses CH1/CH2. The requested polarization is
    therefore resolved to the available CDF variable before reading.
    """
    cdf_obj = CDF(str(path))

    try:
        epoch_all = cdf_obj.varget("Epoch")
        all_times = pd.to_datetime(cdfepoch.to_datetime(epoch_all))
        bounds = record_bounds_for_time_range(pd.DatetimeIndex(all_times), start_time, end_time)

        if bounds is None:
            return pd.DataFrame()

        startrec, endrec = bounds

        intensity_variable = choose_hf_intensity_variable(cdf_obj, polarization)

        epoch = cdf_obj.varget("Epoch", startrec=startrec, endrec=endrec)
        frequency_hz = np.asarray(cdf_obj.varget("Frequency"), dtype=float)
        intensity = np.asarray(
            cdf_obj.varget(intensity_variable, startrec=startrec, endrec=endrec),
            dtype=float,
        )
        intensity = apply_cdf_fill_and_valid_range(cdf_obj, intensity_variable, intensity)

    finally:
        try:
            cdf_obj.close()
        except Exception:
            pass

    times = pd.to_datetime(cdfepoch.to_datetime(epoch))
    freq_mhz = frequency_hz / 1.0e6

    return dataframe_from_arrays(times, freq_mhz, intensity)

def load_assa_file(
    path: Path,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read one Australia-ASSA/e-Callisto FITS file.

    The calibration HDU stores the actual sampled time offsets and frequency
    list.  The primary image is usually ordered as (frequency, time), with the
    frequency axis descending in the file; it is reversed here so that all
    instruments use ascending MHz columns.
    """
    with fits.open(path) as hdul:
        primary = hdul[0]
        calibration = hdul[1].data[0]

        file_start = pd.Timestamp(f"{primary.header['DATE-OBS']} {primary.header['TIME-OBS']}")
        time_offsets = np.asarray(calibration[0], dtype=float)
        frequencies = np.asarray(calibration[1], dtype=float)

        freq_mhz = frequency_to_mhz(frequencies[::-1])
        times = file_start + pd.to_timedelta(time_offsets, unit="s")

        # Original primary data shape is normally (frequency, time).
        data = np.asarray(primary.data, dtype=float)[::-1, :].T

    requested_start = pd.Timestamp(start_time) if start_time is not None else pd.Timestamp(times[0])
    requested_end = pd.Timestamp(end_time) if end_time is not None else pd.Timestamp(times[-1])
    time_mask = (times >= requested_start) & (times <= requested_end)
    if not np.any(time_mask):
        return pd.DataFrame()

    return dataframe_from_arrays(pd.DatetimeIndex(times[time_mask]), freq_mhz, data[time_mask, :])


def load_iprt_file(
    path: Path,
    polarization_index: int = 0,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read one IPRT FITS file."""
    with fits.open(path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

        start = pd.Timestamp(f"{header['DATE-OBS']} {header['TIME-OBS']}")
        n_time = int(header["NAXIS1"])

        if "DATE-END" in header and "TIME-END" in header:
            end = pd.Timestamp(f"{header['DATE-END']} {header['TIME-END']}")
            times = pd.date_range(start=start, end=end, periods=n_time)
        else:
            dt_seconds = float(header.get("CDELT1", 1.0))
            times = start + pd.to_timedelta(np.arange(n_time) * dt_seconds, unit="s")

        bounds = record_bounds_for_time_range(pd.DatetimeIndex(times), start_time, end_time)
        if bounds is None:
            return pd.DataFrame()
        startrec, endrec = bounds
        times = times[startrec:endrec + 1]

        n_freq = int(header["NAXIS2"])
        crval2 = float(header["CRVAL2"])
        cdelt2 = float(header["CDELT2"])
        freqs = crval2 + np.arange(n_freq) * cdelt2
        freqs_mhz = frequency_to_mhz(freqs)
        freq_mask = (freqs_mhz >= 150.0) & (freqs_mhz <= 470.0)

        if data.ndim == 3:
            if not (0 <= polarization_index < data.shape[0]):
                raise ValueError(f"polarization_index={polarization_index} is outside data shape={data.shape}")
            data_2d = np.asarray(data[polarization_index, freq_mask, startrec:endrec + 1], dtype=float)
        elif data.ndim == 2:
            data_2d = np.asarray(data[freq_mask, startrec:endrec + 1], dtype=float)
        else:
            raise ValueError(f"Unexpected IPRT data shape={data.shape}")

    return dataframe_from_arrays(times, freqs_mhz[freq_mask], data_2d)


def load_yamagawa_file(
    path: Path,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read one Yamagawa FITS file."""
    with fits.open(path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

        start = pd.Timestamp(f"{header['DATE-OBS']} {header['TIME-OBS']}")
        n_time = int(header["NAXIS1"])
        dt_seconds = float(header.get("CDELT1", 1.0))
        times = start + pd.to_timedelta(np.arange(n_time) * dt_seconds, unit="s")

        bounds = record_bounds_for_time_range(pd.DatetimeIndex(times), start_time, end_time)
        if bounds is None:
            return pd.DataFrame()
        startrec, endrec = bounds
        times = times[startrec:endrec + 1]

        n_freq = int(header.get("NAXIS2", data.shape[0]))
        if "CRVAL2" in header and "CDELT2" in header:
            freqs = float(header["CRVAL2"]) + np.arange(n_freq) * float(header["CDELT2"])
        else:
            # Fallback copied from the original Yamagawa plotting script.
            freqs = np.linspace(70.0, 9000.0, n_freq)

        freqs_mhz = frequency_to_mhz(freqs)
        freq_mask = (freqs_mhz >= 470.0) & (freqs_mhz <= 1000.0)
        data_2d = np.asarray(data[freq_mask, startrec:endrec + 1], dtype=float)

        nodata = header.get("NODATA")
        if nodata is not None:
            data_2d[np.isclose(data_2d, float(nodata))] = np.nan

    return dataframe_from_arrays(times, freqs_mhz[freq_mask], data_2d)


# =============================================================================
# Integration workflow
# =============================================================================

def load_many(
    paths: Sequence[Path],
    loader: Callable[[Path, pd.Timestamp | None, pd.Timestamp | None], pd.DataFrame],
    instrument_name: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    """Read multiple files and concatenate them along the time axis.

    The requested time window is passed into each reader. This is important for
    large daily CDF/FITS products such as Iitate HF and Yamagawa, because it
    avoids loading a full day or full hour when only a short event interval is
    needed.
    """
    frames: list[pd.DataFrame] = []

    for path in paths:
        if not path.exists():
            continue
        try:
            frame = loader(path, start_time, end_time)
        except Exception as exc:
            warnings.warn(f"{instrument_name}: failed to read {path}: {exc}")
            continue
        frames.append(frame)

    return combine_time_rows(frames)


def prepare_instrument_frame(
    spec: InstrumentSpec,
    paths: Sequence[Path],
    loader: Callable[[Path], pd.DataFrame],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    target_index: pd.DatetimeIndex,
    cadence: str,
    interpolate_short_gaps: bool = True,
    max_gap_factor: float = 2.5,
) -> pd.DataFrame:
    """Load, clip, resample, short-gap-fill, and median-normalize one instrument."""
    raw = load_many(paths, loader, spec.name, start_time, end_time)

    if raw.empty:
        print(f"{spec.name}: no readable data -> NaN block")
        return nan_frame(target_index, spec.fmin_mhz, spec.fmax_mhz, spec.fallback_nfreq)

    raw = trim_time(raw, start_time, end_time)
    raw = clip_frequency(raw, spec.fmin_mhz, spec.fmax_mhz)

    if raw.empty or raw.shape[1] == 0:
        print(f"{spec.name}: no data in assigned band {spec.fmin_mhz:g}-{spec.fmax_mhz:g} MHz -> NaN block")
        return nan_frame(target_index, spec.fmin_mhz, spec.fmax_mhz, spec.fallback_nfreq)

    native_dt = estimate_time_resolution(raw.index)
    native_df = estimate_frequency_resolution(raw.columns.astype(float).to_numpy())

    report_time_gaps(
        spec.name,
        raw.index,
        native_dt,
        max_gap_factor=max_gap_factor,
    )

    raw = complete_frequency_coverage(raw, spec)

    gridded = resample_to_common_grid(
        raw,
        target_index,
        cadence,
        native_dt=native_dt,
        interpolate_short_gaps=interpolate_short_gaps,
        max_gap_factor=max_gap_factor,
    )
    normalized = normalize_by_frequency_median(gridded)

    filled_time_bins = int(np.sum(np.any(np.isfinite(normalized.to_numpy()), axis=1)))
    finite_freq_bins = int(np.sum(np.any(np.isfinite(normalized.to_numpy()), axis=0)))

    print(
        f"{spec.name}: native dt~{format_resolution(native_dt, 's')}, "
        f"native df~{format_resolution(native_df, 'MHz')}, "
        f"assigned grid={raw.shape[1]} frequency bins, "
        f"finite frequencies={finite_freq_bins}, "
        f"filled times={filled_time_bins}/{len(target_index)} after gridding"
    )

    return normalized
def combine_instruments(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate all instruments along frequency and sort by frequency."""
    combined = pd.concat(frames, axis=1)
    combined = collapse_duplicate_frequencies(combined)
    combined = combined.sort_index(axis=1)
    return combined


def build_combined_spectrum(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    cadence: str = "1s",
    hf_polarization: str = "RH",
    iprt_polarization_index: int = 0,
    assa_focuscodes: Sequence[str] = ("62", "56"),
    interpolate_short_gaps: bool = True,
    max_gap_factor: float = 2.5,
    print_report: bool = True,
) -> tuple[pd.DataFrame, dict[str, FileCheckResult]]:
    """Build the combined dynamic spectrum on a common time grid.

    The common grid is kept at 1 s by default.  Short gaps caused only by
    upsampling from a coarser native cadence are filled by time interpolation;
    longer data gaps and truly unobserved frequency ranges remain NaN.
    """
    if start_time >= end_time:
        raise ValueError("start_time must be earlier than end_time.")

    target_index = pd.date_range(start=start_time, end=end_time, freq=cadence)
    if len(target_index) == 0:
        raise ValueError("Empty target time grid. Check start/end/cadence.")

    checks = check_all_files(start_time, end_time, assa_focuscodes=assa_focuscodes)
    if print_report:
        print_file_report(checks)

    spec_by_name = {spec.name: spec for spec in INSTRUMENTS}

    wind_frame = prepare_instrument_frame(
        spec_by_name["Wind/RAD2"],
        checks["Wind/RAD2"].existing_files,
        load_wind_rad2_file,
        start_time,
        end_time,
        target_index,
        cadence,
        interpolate_short_gaps=interpolate_short_gaps,
        max_gap_factor=max_gap_factor,
    )

    hf_frame = prepare_instrument_frame(
        spec_by_name["HF antenna"],
        checks["HF antenna"].existing_files,
        lambda path, st, et: load_hf_file(path, polarization=hf_polarization, start_time=st, end_time=et),
        start_time,
        end_time,
        target_index,
        cadence,
        interpolate_short_gaps=interpolate_short_gaps,
        max_gap_factor=max_gap_factor,
    )

    assa_frame = prepare_instrument_frame(
        spec_by_name["Australia-ASSA"],
        checks["Australia-ASSA"].existing_files,
        load_assa_file,
        start_time,
        end_time,
        target_index,
        cadence,
        interpolate_short_gaps=interpolate_short_gaps,
        max_gap_factor=max_gap_factor,
    )

    iprt_frame = prepare_instrument_frame(
        spec_by_name["IPRT"],
        checks["IPRT"].existing_files,
        lambda path, st, et: load_iprt_file(path, polarization_index=iprt_polarization_index, start_time=st, end_time=et),
        start_time,
        end_time,
        target_index,
        cadence,
        interpolate_short_gaps=interpolate_short_gaps,
        max_gap_factor=max_gap_factor,
    )

    yamagawa_frame = prepare_instrument_frame(
        spec_by_name["Yamagawa"],
        checks["Yamagawa"].existing_files,
        load_yamagawa_file,
        start_time,
        end_time,
        target_index,
        cadence,
        interpolate_short_gaps=interpolate_short_gaps,
        max_gap_factor=max_gap_factor,
    )

    combined = combine_instruments([wind_frame, hf_frame, assa_frame, iprt_frame, yamagawa_frame])
    return combined, checks


# =============================================================================
# Plotting
# =============================================================================

def auto_color_limits(values: np.ndarray) -> tuple[float, float]:
    """Color limits tuned for median-normalized dynamic spectra.

    After per-frequency median normalization, the quiet level is near 1.  A wide
    percentile range over all instruments can be dominated by Yamagawa/IPRT
    bright features and makes weak low-frequency bursts disappear.  This helper
    therefore keeps the lower bound close to the quiet level and chooses a
    conservative upper bound for event-search visibility.
    """
    finite = values[np.isfinite(values)]
    finite = finite[finite > 0]
    if finite.size == 0:
        return 1.0, 1.2

    quiet = float(np.nanmedian(finite))
    upper = float(np.nanpercentile(finite, 99.0))

    if not np.isfinite(quiet):
        quiet = 1.0
    if not np.isfinite(upper):
        upper = 1.2

    vmin = min(1.0, max(0.0, quiet * 0.98))
    vmax = max(1.15, min(upper, 1.6))

    if vmax <= vmin:
        vmax = vmin + 0.2

    return float(vmin), float(vmax)


def geometric_midpoint(a: float, b: float) -> float:
    """Geometric midpoint for placing labels on a logarithmic y-axis."""
    return float(np.sqrt(a * b))


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    """Convert monotonically increasing bin centers to plotting edges.

    This makes pcolormesh respect the actual frequency/time resolution better
    than passing center coordinates directly. For irregular grids, internal
    edges are midpoints between adjacent centers.
    """
    centers = np.asarray(values, dtype=float)
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("centers_to_edges expects a non-empty 1-D array.")
    if centers.size == 1:
        width = 1.0
        return np.array([centers[0] - width / 2.0, centers[0] + width / 2.0])

    internal = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - 0.5 * (centers[1] - centers[0])
    last = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate([[first], internal, [last]])


def datetime_centers_to_edges(times: pd.DatetimeIndex) -> np.ndarray:
    """Convert datetime bin centers to Matplotlib date-number edges."""
    centers = mdates.date2num(pd.DatetimeIndex(times).to_pydatetime())
    return centers_to_edges(centers)

def apply_display_gain_by_instrument(
    values: np.ndarray,
    freq_centers: np.ndarray,
    gain_by_instrument: dict[str, float] | None,
) -> np.ndarray:
    """Apply display-only contrast gain to median-normalized values.

    The scientific data array is still I / median(frequency).  This function is
    used only just before plotting so that weak bands such as Australia-ASSA can
    be seen in the same single-panel figure as much brighter bands.

    For each instrument band, the plotted value is

        display = 1 + gain * (I / median - 1)

    gain=1 leaves the median-normalized intensity unchanged.  Values that are
    NaN remain NaN.
    """
    plot_values = np.array(values, dtype=float, copy=True)
    if not gain_by_instrument:
        return plot_values

    freqs = np.asarray(freq_centers, dtype=float)
    for spec in INSTRUMENTS:
        gain = float(gain_by_instrument.get(spec.name, 1.0))
        if np.isclose(gain, 1.0):
            continue

        freq_mask = (freqs >= spec.fmin_mhz) & (freqs <= spec.fmax_mhz)
        if not np.any(freq_mask):
            continue

        band = plot_values[freq_mask, :]
        with np.errstate(invalid="ignore"):
            plot_values[freq_mask, :] = 1.0 + gain * (band - 1.0)

    return plot_values


def format_display_gain_label(gain_by_instrument: dict[str, float] | None) -> str:
    """Return a compact label describing display-only contrast gains."""
    if not gain_by_instrument:
        return "Intensity / median at each frequency bin"

    active = []
    for spec in INSTRUMENTS:
        gain = float(gain_by_instrument.get(spec.name, 1.0))
        if not np.isclose(gain, 1.0):
            active.append(f"{spec.name}×{gain:g}")

    if not active:
        return "Intensity / median at each frequency bin"

    return "Display-enhanced I/median; gain: " + ", ".join(active)


def add_instrument_boundary_labels(
    ax,
    min_frequency: float,
    max_frequency: float,
    label_time: pd.Timestamp,
) -> None:
    """Draw instrument boundary lines and explicitly label visible frequencies.

    Only boundaries included in the plotted frequency range are drawn. This is
    important when main() is called with a restricted band, for example
    min_frequency=25 and max_frequency=38.
    """
    boundary_specs = [
        (14.0, "14 MHz"),
        (40.0, "40 MHz"),
        (150.0, "150 MHz"),
        (470.0, "470 MHz"),
    ]
    label_x = mdates.date2num(label_time)

    for boundary, label in boundary_specs:
        if not (min_frequency < boundary < max_frequency):
            continue

        ax.axhline(boundary, color="black", linestyle="--", linewidth=1.0, alpha=0.85)
        ax.text(
            label_x,
            boundary * 1.025,
            f" {label}",
            ha="right",
            va="bottom",
            fontsize=10,
            color="black",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
        )
        
        
def plot_combined_spectrum(
    spectrum: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    min_frequency: float,
    max_frequency: float,
    output_path: Path | None = None,
    show: bool = False,
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
    log_frequency: bool = True,
    title: str | None = None,
    display_gain_by_instrument: dict[str, float] | None = None,
) -> None:
    """Plot the combined dynamic spectrum in the requested frequency range.

    The input ``spectrum`` is already normalized as I / median for each
    frequency bin. ``display_gain_by_instrument`` is an optional display-only
    contrast stretch; it does not change the saved/returned spectrum values.

    NaN cells are drawn in white. Therefore, original NaN data, absent files,
    and unobserved frequency ranges are visually distinguished from weak but
    valid data.
    """
    if spectrum.empty:
        raise ValueError("The combined spectrum is empty.")

    min_frequency = float(min_frequency)
    max_frequency = float(max_frequency)
    if min_frequency <= 0:
        raise ValueError("min_frequency must be positive when using a logarithmic frequency axis.")
    if min_frequency >= max_frequency:
        raise ValueError("min_frequency must be smaller than max_frequency.")

    plot_spectrum = clip_frequency(spectrum, min_frequency, max_frequency)
    if plot_spectrum.empty or plot_spectrum.shape[1] == 0:
        raise ValueError(
            f"No frequency bins remain in the requested range: "
            f"{min_frequency:g}-{max_frequency:g} MHz"
        )

    time_edges = datetime_centers_to_edges(plot_spectrum.index)
    freq_centers = plot_spectrum.columns.astype(float).to_numpy()
    freq_edges = centers_to_edges(freq_centers)
    raw_values = plot_spectrum.to_numpy().T
    values = apply_display_gain_by_instrument(
        raw_values,
        freq_centers,
        display_gain_by_instrument,
    )

    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = auto_color_limits(values)
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_facecolor("white")

    mesh = ax.pcolormesh(
        time_edges,
        freq_edges,
        np.ma.masked_invalid(values),
        shading="flat",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
    )

    print(f"Color scale: vmin={vmin:g}, vmax={vmax:g}")
    print(f"Plot frequency range: {min_frequency:g}-{max_frequency:g} MHz")

    ax.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
    ax.set_ylim(min_frequency, max_frequency)
    if log_frequency:
        ax.set_yscale("log")

    ax.set_xlabel("Time [UT]", fontsize=14)
    ax.set_ylabel("Frequency [MHz]", fontsize=14)
    ax.set_title(
        title
        or "Integrated radio dynamic spectrum: Wind + HF + Australia-ASSA + IPRT + Yamagawa",
        fontsize=15,
    )

    label_time = start_time + (end_time - start_time) * 0.985
    add_instrument_boundary_labels(ax, min_frequency, max_frequency, label_time)

    label_x = mdates.date2num(end_time)
    label_specs = [
        ("Wind/RAD2", 1.0, 14.0),
        ("HF antenna", 14.0, 40.0),
        ("Australia-ASSA", 40.0, 150.0),
        ("IPRT", 150.0, 470.0),
        ("Yamagawa", 470.0, 1000.0),
    ]
    for label, y0, y1 in label_specs:
        visible_y0 = max(y0, min_frequency)
        visible_y1 = min(y1, max_frequency)
        if visible_y0 >= visible_y1:
            continue

        ax.text(
            label_x,
            geometric_midpoint(visible_y0, visible_y1),
            f" {label}",
            ha="right",
            va="center",
            fontsize=11,
            color="black",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2),
        )

    from matplotlib.ticker import FixedLocator
    candidate_yticks = np.array(
        [1, 2, 5, 10, 14, 20, 40, 50, 100, 150, 300, 470, 700, 1000],
        dtype=float,
    )
    yticks = candidate_yticks[
        (candidate_yticks >= min_frequency) & (candidate_yticks <= max_frequency)
    ]
    if yticks.size == 0:
        yticks = np.array([min_frequency, max_frequency], dtype=float)

    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    ax.tick_params(axis="both", labelsize=12)

    # cbar = fig.colorbar(mesh, ax=ax, pad=0.015, shrink=0.92)
    # cbar.set_label(format_display_gain_label(display_gain_by_instrument), fontsize=12)
    # cbar.ax.tick_params(labelsize=11)

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
        
def plot_combined_spectrum_on_axis(
    ax,
    spectrum: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    min_frequency: float,
    max_frequency: float,
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
    log_frequency: bool = True,
    title: str | None = None,
    display_gain_by_instrument: dict[str, float] | None = None,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    show_instrument_labels: bool = True,
):
    """Plot one time segment of the combined spectrum on an existing axis."""
    if spectrum.empty:
        raise ValueError("The combined spectrum is empty.")

    min_frequency = float(min_frequency)
    max_frequency = float(max_frequency)
    if min_frequency <= 0:
        raise ValueError("min_frequency must be positive when using a logarithmic frequency axis.")
    if min_frequency >= max_frequency:
        raise ValueError("min_frequency must be smaller than max_frequency.")

    panel_spectrum = trim_time(spectrum, start_time, end_time)
    panel_spectrum = clip_frequency(panel_spectrum, min_frequency, max_frequency)
    if panel_spectrum.empty or panel_spectrum.shape[1] == 0:
        raise ValueError(
            f"No data remain in the requested panel range: "
            f"time={start_time}--{end_time}, freq={min_frequency:g}--{max_frequency:g} MHz"
        )

    time_edges = datetime_centers_to_edges(panel_spectrum.index)
    freq_centers = panel_spectrum.columns.astype(float).to_numpy()
    freq_edges = centers_to_edges(freq_centers)
    raw_values = panel_spectrum.to_numpy().T
    values = apply_display_gain_by_instrument(
        raw_values,
        freq_centers,
        display_gain_by_instrument,
    )

    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = auto_color_limits(values)
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")
    ax.set_facecolor("white")

    mesh = ax.pcolormesh(
        time_edges,
        freq_edges,
        np.ma.masked_invalid(values),
        shading="flat",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
    ax.set_ylim(min_frequency, max_frequency)
    if log_frequency:
        ax.set_yscale("log")

    if show_xlabel:
        ax.set_xlabel("Time [UT]", fontsize=11)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Frequency [MHz]", fontsize=11)
    else:
        ax.set_ylabel("")

    if title is not None:
        ax.set_title(title, fontsize=11)

    label_time = start_time + (end_time - start_time) * 0.985
    add_instrument_boundary_labels(ax, min_frequency, max_frequency, label_time)

    if show_instrument_labels:
        label_x = mdates.date2num(end_time)
        label_specs = [
            ("Wind/RAD2", 1.0, 14.0),
            ("HF antenna", 14.0, 42.0),
            ("Australia-ASSA", 42.0, 150.0),
            ("IPRT", 150.0, 470.0),
            ("Yamagawa", 470.0, 1000.0),
        ]
        for label, y0, y1 in label_specs:
            visible_y0 = max(y0, min_frequency)
            visible_y1 = min(y1, max_frequency)
            if visible_y0 >= visible_y1:
                continue

            ax.text(
                label_x,
                geometric_midpoint(visible_y0, visible_y1),
                f" {label}",
                ha="right",
                va="center",
                fontsize=9,
                color="black",
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.5),
            )

    from matplotlib.ticker import FixedLocator
    candidate_yticks = np.array(
        [1, 2, 5, 10, 14, 20, 42, 50, 100, 150, 300, 470, 700, 1000],
        dtype=float,
    )
    yticks = candidate_yticks[
        (candidate_yticks >= min_frequency) & (candidate_yticks <= max_frequency)
    ]
    if yticks.size == 0:
        yticks = np.array([min_frequency, max_frequency], dtype=float)

    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.tick_params(axis="both", labelsize=10)

    return mesh
        
def export_spectrum_csv(spectrum: pd.DataFrame, output_path: Path) -> None:
    """Save the combined spectrum as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spectrum.to_csv(output_path, index_label="time")
    print(f"CSV saved: {output_path}")

def plot_daily_multi_panel(
    spectrum: pd.DataFrame,
    day_start: pd.Timestamp,
    min_frequency: float,
    max_frequency: float,
    output_path: Path | None = None,
    show: bool = False,
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
    log_frequency: bool = True,
    display_gain_by_instrument: dict[str, float] | None = None,
) -> None:
    """Plot one full day as 4 rows × 2 columns of 3-hour panels."""
    if spectrum.empty:
        raise ValueError("The combined spectrum is empty.")

    min_frequency = float(min_frequency)
    max_frequency = float(max_frequency)
    if min_frequency <= 0:
        raise ValueError("min_frequency must be positive when using a logarithmic frequency axis.")
    if min_frequency >= max_frequency:
        raise ValueError("min_frequency must be smaller than max_frequency.")

    full_spectrum = trim_time(spectrum, day_start, day_start + pd.Timedelta(days=1))
    full_spectrum = clip_frequency(full_spectrum, min_frequency, max_frequency)
    if full_spectrum.empty or full_spectrum.shape[1] == 0:
        raise ValueError(
            f"No frequency bins remain in the requested range: "
            f"{min_frequency:g}-{max_frequency:g} MHz"
        )

    freq_centers = full_spectrum.columns.astype(float).to_numpy()
    raw_values = full_spectrum.to_numpy().T
    display_values = apply_display_gain_by_instrument(
        raw_values,
        freq_centers,
        display_gain_by_instrument,
    )

    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = auto_color_limits(display_values)
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    print(f"Color scale: vmin={vmin:g}, vmax={vmax:g}")
    print(f"Plot frequency range: {min_frequency:g}-{max_frequency:g} MHz")

    fig, axes = plt.subplots(4, 2, figsize=(18, 18), sharey=True)
    panel_mesh = None

    for panel_index in range(8):
        row = panel_index // 2
        col = panel_index % 2
        ax = axes[row, col]

        panel_start = day_start + pd.Timedelta(hours=3 * panel_index)
        panel_end = panel_start + pd.Timedelta(hours=3)
        if panel_index == 7:
            panel_end = day_start + pd.Timedelta(days=1)

        panel_title = f"{panel_start.strftime('%Y-%m-%d %H:%M:%S')} -- {panel_end.strftime('%H:%M:%S')}"
        panel_mesh = plot_combined_spectrum_on_axis(
            ax=ax,
            spectrum=full_spectrum,
            start_time=panel_start,
            end_time=panel_end,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            log_frequency=log_frequency,
            title=panel_title,
            display_gain_by_instrument=display_gain_by_instrument,
            show_ylabel=(col == 0),
            show_xlabel=(row == 3),
            show_instrument_labels=True,
        )

    fig.suptitle(
        f"Integrated radio dynamic spectrum: {day_start.strftime('%Y-%m-%d')} (3-hour panels)",
        fontsize=18,
        y=0.995,
    )

    # cbar = fig.colorbar(panel_mesh, ax=axes.ravel().tolist(), pad=0.01, shrink=0.98)
    # cbar.set_label(format_display_gain_label(display_gain_by_instrument), fontsize=12)
    # cbar.ax.tick_params(labelsize=11)

    fig.tight_layout(rect=[0.03, 0.03, 0.96, 0.98])

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

# =============================================================================
# Main execution settings
# =============================================================================

def main(
    year: int,
    month: int,
    day: int,
    min_frequency: float = 1.0,
    max_frequency: float = 1000.0,
    download_missing_files: bool = True,
) -> None:
    """Run the daily 3-hour-panel plotting workflow for one UT date.

    The specified date is split into eight panels on a 4 × 2 layout:
    00--03, 03--06, 06--09, ..., 21--24 UT.
    """
    # ---------------- user settings ----------------
    day_start = pd.Timestamp(year=int(year), month=int(month), day=int(day))
    day_end = day_start + pd.Timedelta(days=1)
    min_frequency = float(min_frequency)
    max_frequency = float(max_frequency)

    common_cadence = "1s"
    hf_polarization = "CH1"
    iprt_polarization_index = 0

    # Australia-ASSA focuscodes:
    #   "62"/"63" : low-frequency Phase 2.5, about 15--88 MHz
    #   "56"/"57" : high-frequency Phase 3, about 108--370 MHz
    assa_focuscodes = ("62", "60", "56")

    # If True, missing Wind/HF/ASSA/IPRT files are downloaded before plotting.
    # Yamagawa remains manual because the download page requires calculated values.
    download_missing_files = bool(download_missing_files)

    output_dir = Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/combine/event_search")
    output_prefix = "radio_event_search"
    export_csv = False
    show_plot = True

    # Median-normalized dynamic spectra are usually near 1.0. A narrow color
    # scale makes weak Type-II-like structures easier to see.
    cmap = "jet"
    vmin = 1.00
    vmax = 1.2
    log_frequency = True

    # Display-only contrast boost. Set all gains to 1.0 for a strictly raw
    # I/median color plot. NaN cells are still plotted in white.
    display_gain_by_instrument = {
        "Wind/RAD2": 1.0,
        "HF antenna": 1.0,
        "Australia-ASSA": 4.0,
        "IPRT": 1.0,
        "Yamagawa": 0.1,
    }
    # ------------------------------------------------

    global WIND_RAW_DIR, HF_RAW_DIR, ASSA_RAW_DIR, IPRT_RAW_DIR, YAMAGAWA_RAW_DIR
    WIND_RAW_DIR = normalize_path(WIND_RAW_DIR)
    HF_RAW_DIR = normalize_path(HF_RAW_DIR)
    ASSA_RAW_DIR = normalize_path(ASSA_RAW_DIR)
    IPRT_RAW_DIR = normalize_path(IPRT_RAW_DIR)
    YAMAGAWA_RAW_DIR = normalize_path(YAMAGAWA_RAW_DIR)

    if download_missing_files:
        download_missing_input_files(
            start_time=day_start,
            end_time=day_end,
            assa_focuscodes=assa_focuscodes,
        )

    output_dir = normalize_path(output_dir)
    date_tag = day_start.strftime("%Y%m%d")
    freq_tag = f"{min_frequency:g}-{max_frequency:g}MHz".replace(".", "p")
    output_base = output_dir / f"{output_prefix}_{date_tag}_daily_3hgrid_{freq_tag}"

    spectrum, _ = build_combined_spectrum(
        start_time=day_start,
        end_time=day_end,
        cadence=common_cadence,
        hf_polarization=hf_polarization.upper(),
        iprt_polarization_index=iprt_polarization_index,
        assa_focuscodes=assa_focuscodes,
        print_report=True,
    )

    if export_csv:
        export_spectrum_csv(spectrum, output_base.with_suffix(".csv"))

    plot_daily_multi_panel(
        spectrum=spectrum,
        day_start=day_start,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        output_path=output_base.with_suffix(".png"),
        show=show_plot,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        log_frequency=log_frequency,
        display_gain_by_instrument=display_gain_by_instrument,
    )    
# if __name__ == "__main__":
#     start_time_str = "2026-04-24 00:00:00"
#     end_time_str = "2026-04-24 03:00:00"
    
#     min_frequency = 1.0
#     max_frequency = 1000.0

#     download_missing_files = True

#     main(start_time_str, end_time_str, min_frequency, max_frequency, download_missing_files)


if __name__ == "__main__":
    year = 2026
    month = 4
    day = 23

    min_frequency = 1.0
    max_frequency = 1000.0

    download_missing_files = True

    main(year, month, day, min_frequency, max_frequency, download_missing_files)