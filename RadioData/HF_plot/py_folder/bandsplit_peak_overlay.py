from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import SecondLocator
from matplotlib.ticker import FuncFormatter, MultipleLocator

try:
    from .peak_analysis import (
        calculate_fit_with_error,
        calculate_peak_time_and_freq,
        plot_removed_dynamic_spectrum_with_peak,
        plot_removed_dynamic_spectrum_with_peak_2,
    )
    from .frequency_conversion import density_from_frequency, frequency_from_density
    from .utils import _initialize_data_parameters
except ImportError:
    from peak_analysis import (
        calculate_fit_with_error,
        calculate_peak_time_and_freq,
        plot_removed_dynamic_spectrum_with_peak,
        plot_removed_dynamic_spectrum_with_peak_2,
    )
    from frequency_conversion import density_from_frequency, frequency_from_density
    from utils import _initialize_data_parameters


RESEARCH_ROOT = Path(__file__).resolve().parents[3]
COMBINE_DIR = RESEARCH_ROOT / "RadioData" / "combine"
if str(COMBINE_DIR) not in sys.path:
    sys.path.append(str(COMBINE_DIR))

_RADIO_DATA_DIR = Path(__file__).resolve().parents[2]
if str(_RADIO_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_RADIO_DATA_DIR))

import radio_event_search as _radio_es

from predict_type2_const_speed import f_model_from_r, invert_r_from_f
from wind_hf_assa_dynamic_spectrum import (
    combine_spectra,
    create_dataframe,
    load_callisto,
    load_hf,
    load_wind_rad2,
    normalize_by_median,
    resample_to_grid,
)

WIND_RAW_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/Wind/Rawdata")
HF_RAW_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/HF_plot/Rawdata")
ASSA_RAW_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/e-Callisto/Rawdata")

DEFAULT_PEAK_START = [
    "2022-06-13T03:25:00",
    "2022-06-13T03:25:30",
    "2022-06-13T03:26:30",
    "2022-06-13T03:28:45",
    "2022-06-13T03:29:30",
    "2022-06-13T03:30:10",
    "2022-06-13T03:30:40",
]
DEFAULT_PEAK_END = [
    "2022-06-13T03:33:00",
    "2022-06-13T03:26:30",
    "2022-06-13T03:28:45",
    "2022-06-13T03:29:30",
    "2022-06-13T03:30:10",
    "2022-06-13T03:30:40",
    "2022-06-13T03:31:20",
]
DEFAULT_PEAK_FREQ_MIN = [24, 32, 30, 29.64, 28.2, 27.5, 26]
DEFAULT_PEAK_FREQ_MAX = [38, 37, 37, 34, 34, 34, 33]


def _patch_radio_event_search_directories() -> None:
    """Align radio_event_search raw-directory globals with this module (see radio_event_search.py)."""
    _radio_es.WIND_RAW_DIR = WIND_RAW_DIR
    _radio_es.HF_RAW_DIR = HF_RAW_DIR
    _radio_es.ASSA_RAW_DIR = ASSA_RAW_DIR


def _resolve_instrument_paths(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    assa_focuscodes: Sequence[str],
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Resolve Wind/HF daily CDF paths from date stamps in filenames, and ASSA FITS paths
    from filename times plus FITS header/calibration coverage (find_assa_paths).
    """
    _patch_radio_event_search_directories()
    wind_expected = _radio_es.expected_wind_paths(start_time, end_time)
    hf_expected = _radio_es.expected_hf_paths(start_time, end_time)
    wind_paths = [p for p in wind_expected if p.is_file() and p.stat().st_size > 0]
    hf_paths = [p for p in hf_expected if p.is_file() and p.stat().st_size > 0]
    assa_paths = _radio_es.find_assa_paths(start_time, end_time, focuscodes=assa_focuscodes)

    if not wind_paths:
        raise FileNotFoundError(
            "No Wind/RAD2 CDF found for the requested interval under "
            f"{WIND_RAW_DIR}. Expected filenames like "
            f"wi_l2_wav_rad2_{start_time.strftime('%Y%m%d')}_v01.cdf "
            f"(see expected_wind_paths in radio_event_search.py)."
        )
    if not hf_paths:
        raise FileNotFoundError(
            "No Iitate HF CDF found for the requested interval under "
            f"{HF_RAW_DIR}. Expected filenames like "
            f"it_h1_hf_{start_time.strftime('%Y%m%d')}_v01.cdf "
            f"(see expected_hf_paths in radio_event_search.py)."
        )
    if not assa_paths:
        raise FileNotFoundError(
            "No Australia-ASSA FITS overlapping the requested interval under "
            f"{ASSA_RAW_DIR} for focuscodes={tuple(assa_focuscodes)} "
            f"(see find_assa_paths in radio_event_search.py)."
        )
    return wind_paths, hf_paths, assa_paths


def _load_wind_rad2_stacked(paths: Sequence[Path]) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Load one or more daily Wind RAD2 CDFs and concatenate along time."""
    ordered = sorted(paths, key=lambda p: p.name)
    times_parts: list[pd.DatetimeIndex] = []
    values_parts: list[np.ndarray] = []
    freq_mhz: np.ndarray | None = None
    for path in ordered:
        t, f, v = load_wind_rad2(path)
        if freq_mhz is None:
            freq_mhz = f
        elif not np.allclose(freq_mhz, f):
            raise ValueError(f"Wind/RAD2 frequency grid mismatch between files (check {path.name}).")
        times_parts.append(t)
        values_parts.append(v)
    assert freq_mhz is not None
    combined_times = pd.DatetimeIndex(
        np.concatenate([np.asarray(ti, dtype="datetime64[ns]") for ti in times_parts])
    )
    combined_values = np.vstack(values_parts)
    order = np.argsort(combined_times)
    return combined_times[order], freq_mhz, combined_values[order]


def _load_hf_stacked(paths: Sequence[Path], polarization: str) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Load one or more daily HF CDFs and concatenate along time."""
    ordered = sorted(paths, key=lambda p: p.name)
    times_parts: list[pd.DatetimeIndex] = []
    values_parts: list[np.ndarray] = []
    freq_mhz: np.ndarray | None = None
    for path in ordered:
        t, f, v = load_hf(path, polarization)
        if freq_mhz is None:
            freq_mhz = f
        elif not np.allclose(freq_mhz, f):
            raise ValueError(f"HF frequency grid mismatch between files (check {path.name}).")
        times_parts.append(t)
        values_parts.append(v)
    assert freq_mhz is not None
    combined_times = pd.DatetimeIndex(
        np.concatenate([np.asarray(ti, dtype="datetime64[ns]") for ti in times_parts])
    )
    combined_values = np.vstack(values_parts)
    order = np.argsort(combined_times)
    return combined_times[order], freq_mhz, combined_values[order]


def _freq_to_r(f_mhz: np.ndarray | float, branch: str = "F", factor: float = 1.0) -> np.ndarray | float:
    f_arr = np.asarray(f_mhz, dtype=float)
    vec = np.vectorize(lambda v: invert_r_from_f(float(v), branch=branch, factor=factor))
    return vec(f_arr)


def _r_to_freq(r_rs: np.ndarray | float, branch: str = "F", factor: float = 1.0) -> np.ndarray | float:
    r_arr = np.asarray(r_rs, dtype=float)
    vec = np.vectorize(lambda v: f_model_from_r(float(v), branch=branch, factor=factor))
    return vec(r_arr)


def _lane_to_series(lane: Sequence[Tuple[str, float]], name: str) -> pd.Series:
    t = pd.to_datetime([ts for ts, _ in lane])
    y = np.asarray([v for _, v in lane], dtype=float)
    s = pd.Series(y, index=t, name=name)
    return s.sort_index()


def _merge_asof_on_time(
    left: pd.Series,
    right: pd.Series,
    left_name: str,
    right_name: str,
    tolerance: str = "2s",
) -> pd.DataFrame:
    ldf = left.rename(left_name).sort_index().reset_index().rename(columns={"index": "time"})
    rdf = right.rename(right_name).sort_index().reset_index().rename(columns={"index": "time"})
    out = pd.merge_asof(
        ldf,
        rdf,
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    out = out.dropna(subset=[left_name, right_name])
    return out


def _build_combined_spectrum(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    min_frequency: float,
    max_frequency: float,
    cadence: str = "0.5s",
    polarization: str = "RH",
) -> pd.DataFrame:
    wind_times, wind_freqs, wind_values = load_wind_rad2(WIND_CDF_PATH)
    hf_times, hf_freqs, hf_values = load_hf(HF_CDF_PATH, polarization)
    assa_times, assa_freqs, assa_values = load_callisto(ASSA_FITS_PATHS)

    target_index = pd.date_range(start=start_time, end=end_time, freq=cadence)
    if len(target_index) == 0:
        raise ValueError("Target index is empty. Check cadence and time range.")

    time_margin = pd.to_timedelta(cadence)
    extended_range = (start_time - time_margin, end_time + time_margin)

    wind_df = create_dataframe(wind_times, wind_freqs, wind_values, extended_range)
    wind_df = resample_to_grid(wind_df, target_index, cadence)
    wind_df = normalize_by_median(wind_df)

    hf_df = create_dataframe(hf_times, hf_freqs, hf_values, extended_range)
    hf_df = resample_to_grid(hf_df, target_index, cadence)
    hf_df = normalize_by_median(hf_df)

    assa_df = create_dataframe(assa_times, assa_freqs, assa_values, extended_range)
    assa_df = resample_to_grid(assa_df, target_index, cadence)
    assa_df = normalize_by_median(assa_df)

    combined = combine_spectra([wind_df, hf_df, assa_df])
    frequency_index = combined.columns.astype(float)
    freq_mask = (frequency_index >= min_frequency) & (frequency_index <= max_frequency)
    combined = combined.loc[:, freq_mask]
    if combined.empty:
        raise ValueError("No data remains after applying the frequency bounds.")
    return combined


def _remove_new_images(ax: plt.Axes, image_ids_before: set[int]) -> None:
    for image in list(ax.images):
        if id(image) not in image_ids_before:
            image.remove()


def _overlay_peak_points_same_as_main_peak(
    fig: plt.Figure,
    ax: plt.Axes,
    peak_start_time: Sequence[str],
    peak_end_time: Sequence[str],
    peak_freq_min: Sequence[float],
    peak_freq_max: Sequence[float],
    time_tick_sec: int = 60,
    freq_tick_mhz: float = 1.0,
    med_filter_size: tuple[int, int] = (1, 1),
    vmin: float = 80,
    vmax: float = 95,
    title: str = "Second Harmonic Lane Analysis",
    hf_time: np.ndarray | None = None,
    hf_frequency_mhz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outlier_z = 3.0
    if hf_time is None or hf_frequency_mhz is None:
        _, hf_time, hf_frequency_mhz, _ = _initialize_data_parameters()

    times_red, freqs_red = [], []
    times_blue, freqs_blue = [], []

    for i in range(len(peak_start_time)):
        if i == 0:
            continue

        if i in (1, 2):
            before_image_ids = {id(image) for image in ax.images}
            masked_data = plot_removed_dynamic_spectrum_with_peak(
                fig,
                ax,
                start_time=peak_start_time[i],
                end_time=peak_end_time[i],
                freq_min=peak_freq_min[i],
                freq_max=peak_freq_max[i],
                time_tick_sec=time_tick_sec,
                freq_tick_mhz=freq_tick_mhz,
                med_filter_size=med_filter_size,
                vmin=vmin,
                vmax=vmax,
                title=title,
                scatter_color="red",
                outlier_z=outlier_z,
            )
            _remove_new_images(ax, before_image_ids)

            t, f = calculate_peak_time_and_freq(
                hf_time,
                hf_frequency_mhz,
                masked_data,
                peak_start_time[i],
                peak_end_time[i],
                peak_freq_min[i],
                peak_freq_max[i],
                med_filter_size,
                outlier_z,
            )
            times_red.append(t)
            freqs_red.append(f)
        else:
            if i == len(peak_start_time) - 2:
                before_image_ids = {id(image) for image in ax.images}
                masked_data = plot_removed_dynamic_spectrum_with_peak(
                    fig,
                    ax,
                    start_time=peak_start_time[i],
                    end_time=peak_end_time[i],
                    freq_min=peak_freq_min[i],
                    freq_max=peak_freq_max[i],
                    time_tick_sec=time_tick_sec,
                    freq_tick_mhz=freq_tick_mhz,
                    med_filter_size=med_filter_size,
                    vmin=vmin,
                    vmax=vmax,
                    title=title,
                    scatter_color="blue",
                    outlier_z=outlier_z,
                )
                _remove_new_images(ax, before_image_ids)

                t, f = calculate_peak_time_and_freq(
                    hf_time,
                    hf_frequency_mhz,
                    masked_data,
                    peak_start_time[i],
                    peak_end_time[i],
                    peak_freq_min[i],
                    peak_freq_max[i],
                    med_filter_size,
                    outlier_z,
                )
                times_blue.append(t)
                freqs_blue.append(f)
            else:
                before_image_ids = {id(image) for image in ax.images}
                masked_data_2 = plot_removed_dynamic_spectrum_with_peak_2(
                    fig,
                    ax,
                    start_time=peak_start_time[i],
                    end_time=peak_end_time[i],
                    freq_min=peak_freq_min[i],
                    freq_max=peak_freq_max[i],
                    time_tick_sec=time_tick_sec,
                    freq_tick_mhz=freq_tick_mhz,
                    med_filter_size=med_filter_size,
                    vmin=vmin,
                    vmax=vmax,
                    title=title,
                    scatter_color="blue",
                    threshold_high=91,
                    outlier_z=outlier_z,
                )
                _remove_new_images(ax, before_image_ids)

                t, f = calculate_peak_time_and_freq(
                    hf_time,
                    hf_frequency_mhz,
                    masked_data_2,
                    peak_start_time[i],
                    peak_end_time[i],
                    peak_freq_min[i],
                    peak_freq_max[i],
                    med_filter_size,
                    outlier_z,
                )
                times_blue.append(t)
                freqs_blue.append(f)

    if times_red:
        times_red = np.concatenate(times_red)
        freqs_red = np.concatenate(freqs_red)
    else:
        times_red = np.array([])
        freqs_red = np.array([])

    if times_blue:
        times_blue = np.concatenate(times_blue)
        freqs_blue = np.concatenate(freqs_blue)
    else:
        times_blue = np.array([])
        freqs_blue = np.array([])

    return times_red, freqs_red, times_blue, freqs_blue


def plot_bandsplit_fig1_with_main_peak_points(
    start_time: str = "2022-06-13T03:25:00",
    end_time: str = "2022-06-13T03:33:00",
    min_frequency: float = 25,
    max_frequency: float = 47,
    model_factor: float = 2.8,
    cadence: str = "0.5s",
    polarization: str = "RH",
    cmap: str = "viridis",
    vmin: float | None = 1.0,
    vmax: float | None = 1.1,
    log_scale: bool = True,
    show: bool = True,
    use_removed_background: bool = False,
    output_path: Path | None = None,
    peak_start_time: Sequence[str] = DEFAULT_PEAK_START,
    peak_end_time: Sequence[str] = DEFAULT_PEAK_END,
    peak_freq_min: Sequence[float] = DEFAULT_PEAK_FREQ_MIN,
    peak_freq_max: Sequence[float] = DEFAULT_PEAK_FREQ_MAX,
) -> tuple[plt.Figure, plt.Axes]:
    start_time_ts = pd.Timestamp(start_time)
    end_time_ts = pd.Timestamp(end_time)
    if start_time_ts >= end_time_ts:
        raise ValueError("Start time must be earlier than end time.")

    n_segments = len(peak_start_time)
    if not (len(peak_end_time) == n_segments == len(peak_freq_min) == len(peak_freq_max)):
        raise ValueError("Peak segment arrays must have the same length.")

    spectrum = _build_combined_spectrum(
        start_time=start_time_ts,
        end_time=end_time_ts,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        cadence=cadence,
        polarization=polarization,
    )

    time_axis = spectrum.index.to_pydatetime()
    freq_axis = spectrum.columns.to_numpy()
    values = spectrum.to_numpy().T
    if use_removed_background:
        # mask_band = (freq_axis >= 35) & (freq_axis <= 40)
        # flat = values[mask_band, :].ravel()
        # mu, sigma = flat.mean(), flat.std()
        # clean = flat[(flat > mu - 3 * sigma) & (flat < mu + 3 * sigma)]
        # threshold = clean.mean() * 1.05
        # values = np.where(values > threshold, values, np.nan)
        freq_median = np.nanmedian(values, axis=1, keepdims=True)
        normalized_values = np.divide(
            values,
            freq_median,
            out=np.full_like(values, np.nan, dtype=float),
            where=freq_median != 0,
        )
        values = np.where(normalized_values >= 1.01, normalized_values, np.nan)

    fig, ax = plt.subplots(figsize=(16.5, 8.3))
    ax.pcolormesh(
        mdates.date2num(time_axis),
        freq_axis,
        values,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    upper_lane = [("2022-06-13T03:25:30", 35.5),
        ("2022-06-13T03:25:40", 35.7), ("2022-06-13T03:25:50", 35.9), ("2022-06-13T03:26:00", 36.1), ("2022-06-13T03:26:10", 36.2),
        ("2022-06-13T03:26:20", 36.3), ("2022-06-13T03:26:30", 36.4), ("2022-06-13T03:26:40", 36.5),
        ("2022-06-13T03:26:50", 36.3), ("2022-06-13T03:27:00", 36.0), ("2022-06-13T03:27:10", 35.7), ("2022-06-13T03:27:20", 35.4),
        ("2022-06-13T03:27:30", 35.1), ("2022-06-13T03:27:40", 34.8), ("2022-06-13T03:27:50", 34.5), ("2022-06-13T03:28:00", 34.2),
        ("2022-06-13T03:28:10", 33.9), ("2022-06-13T03:28:20", 33.6), ("2022-06-13T03:28:30", 33.3), ("2022-06-13T03:28:40", 33.0),
        ("2022-06-13T03:28:50", 32.7), ("2022-06-13T03:29:00", 32.4), ("2022-06-13T03:29:10", 32.1), ("2022-06-13T03:29:20", 31.8),
        ("2022-06-13T03:29:30", 31.5), ("2022-06-13T03:29:40", 31.4), ("2022-06-13T03:29:50", 31.3),
        ("2022-06-13T03:30:00", 31.3), ("2022-06-13T03:30:10", 31.1), ("2022-06-13T03:30:20", 31.0), ("2022-06-13T03:30:30", 30.9),
        ("2022-06-13T03:30:40", 30.8), ("2022-06-13T03:30:50", 30.8), ("2022-06-13T03:31:00", 30.7), ("2022-06-13T03:31:10", 30.6),
        ("2022-06-13T03:31:20", 30.3), ("2022-06-13T03:31:30", 30.2), ("2022-06-13T03:31:40", 29.7),
        ("2022-06-13T03:31:50", 29.6), ("2022-06-13T03:32:00", 29.5), ("2022-06-13T03:32:10", 29.4), ("2022-06-13T03:32:20", 29.3),
        ("2022-06-13T03:32:30", 29.2), ("2022-06-13T03:32:40", 29.2),
        ("2022-06-13T03:32:50", 29.0), ("2022-06-13T03:33:00", 28.9),
    ]

    lower_lane = [("2022-06-13T03:25:30", 33.8),
        ("2022-06-13T03:25:40", 33.2), ("2022-06-13T03:25:50", 32.8), ("2022-06-13T03:26:00", 32.4), ("2022-06-13T03:26:10", 31.9),
        ("2022-06-13T03:26:20", 31.7), ("2022-06-13T03:26:30", 31.6), ("2022-06-13T03:26:40", 31.5), ("2022-06-13T03:26:50", 31.0),
        ("2022-06-13T03:27:00", 30.9), ("2022-06-13T03:27:10", 30.8), ("2022-06-13T03:27:20", 30.6), ("2022-06-13T03:27:30", 30.5),
        ("2022-06-13T03:27:40", 30.4), ("2022-06-13T03:27:50", 30.2), ("2022-06-13T03:28:00", 30.0), ("2022-06-13T03:28:10", 29.8),
        ("2022-06-13T03:28:20", 29.7),
        ("2022-06-13T03:28:30", 29.8), ("2022-06-13T03:28:40", 30.0), ("2022-06-13T03:28:50", 30.2),
        ("2022-06-13T03:29:00", 29.7), ("2022-06-13T03:29:10", 29.1),
        ("2022-06-13T03:29:20", 29.1), ("2022-06-13T03:29:30", 28.8), ("2022-06-13T03:29:40", 28.5), ("2022-06-13T03:29:50", 28.2),
        ("2022-06-13T03:30:00", 27.9), ("2022-06-13T03:30:10", 27.6), ("2022-06-13T03:30:20", 27.3), ("2022-06-13T03:31:00", 26.9),
        ("2022-06-13T03:31:10", 26.4), ("2022-06-13T03:31:20", 26.0),
        ("2022-06-13T03:31:30", 25.9), ("2022-06-13T03:31:40", 25.8), ("2022-06-13T03:31:50", 25.7), ("2022-06-13T03:32:00", 25.6),
        ("2022-06-13T03:32:10", 25.5), ("2022-06-13T03:32:20", 25.4), ("2022-06-13T03:32:30", 25.3), ("2022-06-13T03:32:40", 25.2),
    ]

    def _to_mdates_xy(lane: Sequence[Tuple[str, float]]) -> tuple[np.ndarray, list[float]]:
        t = [dt.datetime.fromisoformat(ts) for ts, _ in lane]
        y = [v for _, v in lane]
        return mdates.date2num(t), y

    upper_x, upper_y = _to_mdates_xy(upper_lane)
    lower_x, lower_y = _to_mdates_xy(lower_lane)

    def _drift_rate_stats(x_mdates: np.ndarray, y_mhz: Sequence[float]) -> tuple[float | None, float | None]:
        if len(x_mdates) < 2:
            return None, None
        dt_sec = np.diff(x_mdates) * 86400.0
        df_mhz = np.diff(y_mhz)
        rates = df_mhz / dt_sec
        return float(np.mean(rates)), float(np.std(rates))

    upper_mean, upper_std = _drift_rate_stats(upper_x, upper_y)
    lower_mean, lower_std = _drift_rate_stats(lower_x, lower_y)

    ax.scatter(upper_x, upper_y, color="orange", marker="+", s=50, zorder=12)
    ax.scatter(lower_x, lower_y, color="purple", marker="+", s=50, zorder=12)
    ax.plot(upper_x, upper_y, color="orange", linestyle="--", linewidth=2)
    ax.plot(lower_x, lower_y, color="purple", linestyle="--", linewidth=2)

    intermittent_time = pd.Timestamp("2022-06-13T03:28:45")
    ax.axvline(x=mdates.date2num(intermittent_time), color="black", linestyle="--", linewidth=2)
    ax.text(mdates.date2num(intermittent_time), 47, intermittent_time.strftime("%H:%M:%S"), color="black", fontsize=16, ha="left", va="top")

    shift_lower_frequency, shift_upper_frequency = 4.7, 8.3
    upper_x_np, upper_y_np = np.array(upper_x), np.array(upper_y)
    shifted_upper_y_np = upper_y_np + shift_upper_frequency
    lower_x_np, lower_y_np = np.array(lower_x), np.array(lower_y)
    shifted_lower_y_np = lower_y_np + shift_lower_frequency
    cutoff = mdates.date2num(pd.Timestamp("2022-06-13T03:28:45"))
    upper_mask = upper_x_np < cutoff
    ax.plot(
        upper_x_np[upper_mask],
        shifted_upper_y_np[upper_mask],
        color="orange",
        linestyle="--",
        linewidth=2,
    )
    lower_mask = lower_x_np < cutoff
    # ax.plot(lower_x_np[lower_mask], shifted_lower_y_np[lower_mask], color="purple", linestyle="--", linewidth=2)

    split_lower_lane = [("2022-06-13T03:25:30", 40.4),
        ("2022-06-13T03:25:40", 40.2), ("2022-06-13T03:25:50", 39.8), ("2022-06-13T03:26:00", 39.4), ("2022-06-13T03:26:10", 38.9),
        ("2022-06-13T03:26:20", 38.5),
        ("2022-06-13T03:26:30", 38), ("2022-06-13T03:26:40", 38), ("2022-06-13T03:26:50", 38.0),
        ("2022-06-13T03:27:00", 38.5), ("2022-06-13T03:27:10", 38.5), ("2022-06-13T03:27:20", 38.5),
        ("2022-06-13T03:27:30", 38.5), ("2022-06-13T03:27:40", 38.5), ("2022-06-13T03:27:50", 38.5), ("2022-06-13T03:28:00", 39.0),
        ("2022-06-13T03:28:10", 39), ("2022-06-13T03:28:20", 39.5),
        ("2022-06-13T03:28:30", 39.5), ("2022-06-13T03:28:40", 39.5), ("2022-06-13T03:28:50", 39.5),
    ]
    split_lower_x, split_lower_y = _to_mdates_xy(split_lower_lane)
    ax.plot(split_lower_x, split_lower_y, color="purple", linestyle="--", linewidth=2)


    main_max_ser = _lane_to_series(upper_lane, name="main_max")
    main_min_ser = _lane_to_series(lower_lane, name="main_min")

    split_max_times = pd.to_datetime([ts for ts, _ in upper_lane])
    split_max_freq = np.asarray([v for _, v in upper_lane], dtype=float) + float(shift_upper_frequency)
    split_max_ser = pd.Series(split_max_freq, index=split_max_times, name="split_max").sort_index()
    split_max_ser = split_max_ser[split_max_ser.index < intermittent_time]

    split_min_ser = _lane_to_series(split_lower_lane, name="split_min")
    split_min_ser = split_min_ser[split_min_ser.index < intermittent_time]

    center_split_df = _merge_asof_on_time(split_max_ser, split_min_ser, "split_max", "split_min")
    center_split_df["center_split"] = 0.5 * (center_split_df["split_max"] + center_split_df["split_min"])
    center_main_df = _merge_asof_on_time(main_max_ser, main_min_ser, "main_max", "main_min")
    center_main_df["center_main"] = 0.5 * (center_main_df["main_max"] + center_main_df["main_min"])

    center_split_x = mdates.date2num(pd.to_datetime(center_split_df["time"]))
    center_split_y = center_split_df["center_split"].to_numpy(dtype=float)
    center_main_x = mdates.date2num(pd.to_datetime(center_main_df["time"]))
    center_main_y = center_main_df["center_main"].to_numpy(dtype=float)

    center_split_mean, center_split_std = _drift_rate_stats(center_split_x, center_split_y)
    center_main_mean, center_main_std = _drift_rate_stats(center_main_x, center_main_y)

    ax.plot(center_split_x, center_split_y, color="black", linestyle="--", linewidth=3)
    ax.plot(center_main_x, center_main_y, color="black", linestyle="--", linewidth=3)

    _, hf_time, hf_frequency_mhz, _ = _initialize_data_parameters()
    times_red, freqs_red, times_blue, freqs_blue = _overlay_peak_points_same_as_main_peak(
        fig=fig,
        ax=ax,
        peak_start_time=peak_start_time,
        peak_end_time=peak_end_time,
        peak_freq_min=peak_freq_min,
        peak_freq_max=peak_freq_max,
        time_tick_sec=60,
        freq_tick_mhz=1,
        med_filter_size=(1, 1),
        vmin=80,
        vmax=95,
        title="Second Harmonic Lane Analysis",
        hf_time=hf_time,
        hf_frequency_mhz=hf_frequency_mhz,
    )

    if len(times_red) > 0 and len(times_blue) > 0:
        xnum_red = mdates.date2num(times_red)
        xnum_blue = mdates.date2num(times_blue)
        xnum_total = np.concatenate([xnum_red, xnum_blue])
        t_sec_total = (xnum_total - xnum_total[0]) * 86400.0
        t_sec_red = (xnum_red - xnum_red[0]) * 86400.0
        t_sec_blue = (xnum_blue - xnum_blue[0]) * 86400.0

        slope_red, intercept_red, stderr_red = calculate_fit_with_error(ax, times_red, freqs_red)
        slope_blue, intercept_blue, stderr_blue = calculate_fit_with_error(ax, times_blue, freqs_blue)
        fit_red = slope_red * t_sec_red + intercept_red
        fit_blue = slope_blue * t_sec_blue + intercept_blue

        dens_red_start = density_from_frequency(fit_red[0] / 2)
        dens_red_end = density_from_frequency(fit_red[-1] / 2)
        freq_dens_red_start = frequency_from_density(dens_red_start) * 2
        freq_dens_red_end = frequency_from_density(dens_red_end) * 2
        dens_blue_start = density_from_frequency(fit_blue[0] / 2)
        dens_blue_end = density_from_frequency(fit_blue[-1] / 2)
        freq_dens_blue_start = frequency_from_density(dens_blue_start) * 2
        freq_dens_blue_end = frequency_from_density(dens_blue_end) * 2

        freq_red_mid = fit_red[np.argmin(np.abs(t_sec_total - t_sec_red[-1]))]
        if len(times_red) >= 2:
            ax.plot(
                xnum_red,
                fit_red,
                linestyle="--",
                linewidth=3,
                color="#ff00ff",
                zorder=12,
                label=f"Red FDR (2nd harmonic): {slope_red/2:.3e}±{stderr_red/2:.3e} MHz/s",
            )
            ax.scatter(
                xnum_red[0],
                freq_dens_red_start,
                color="#ff00ff",
                marker="x",
                s=100,
                zorder=12,
                label=f"{freq_dens_red_start:.2f}[MHz] @ {_freq_to_r(freq_dens_red_start, branch='H', factor=model_factor):.3f}[$R_\\odot$] $\\rightarrow$ {freq_red_mid:.3f}[MHz] @ {_freq_to_r(freq_red_mid, branch='H', factor=model_factor):.3f}[$R_\\odot$]",
            )
            ax.scatter(
                xnum_red[np.argmin(np.abs(t_sec_total - t_sec_red[-1]))],
                freq_red_mid,
                color="#ff00ff",
                marker="x",
                s=100,
                zorder=12,
            )

        if len(times_blue) >= 2:
            ax.plot(
                xnum_blue,
                fit_blue,
                linestyle="--",
                linewidth=3,
                color="#00ffff",
                zorder=12,
                label=f"Blue FDR (2nd harmonic): {slope_blue/2:.3e}±{stderr_blue/2:.3e} MHz/s",
            )
            ax.scatter(xnum_blue[-1], freq_dens_blue_end, color="#00ffff", marker="x", s=100, zorder=12)
            ax.scatter(
                xnum_blue[0],
                freq_dens_blue_start,
                color="#00ffff",
                marker="x",
                s=100,
                zorder=12,
                label=f"{freq_dens_blue_start:.2f}[MHz] @ {_freq_to_r(freq_dens_blue_start, branch='H', factor=model_factor):.3f}[$R_\\odot$] $\\rightarrow$ {freq_dens_blue_end:.2f}[MHz] @ {_freq_to_r(freq_dens_blue_end, branch='H', factor=model_factor):.3f}[$R_\\odot$]",
            )

    t_line_red = dt.datetime.fromisoformat("2022-06-13T03:25:30")
    xnum_red_line = mdates.date2num(t_line_red)
    ax.axvline(xnum_red_line, color="red", linestyle="--", linewidth=2)
    ax.text(xnum_red_line, max_frequency, "03:25:30", color="red", va="top", ha="left", fontsize=16)

    t_line_blue = dt.datetime.fromisoformat("2022-06-13T03:31:20")
    xnum_blue_line = mdates.date2num(t_line_blue)
    ax.axvline(xnum_blue_line, color="blue", linestyle="--", linewidth=2)
    ax.text(xnum_blue_line, max_frequency, "03:31:20", color="blue", va="top", ha="left", fontsize=16)

    ax.set_ylabel("Frequency [MHz]", fontsize=16)
    if log_scale is not False:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
    ax.set_xlabel("Time [UT]", fontsize=16)
    ax.set_title(
        "Dynamic Spectrum; HF antenna (14-40 MHz) + Australia-ASSA (40-85 MHz)",
        fontsize=14,
    )
    ax.set_xlim(mdates.date2num(start_time_ts), mdates.date2num(end_time_ts))
    ax.set_ylim(min_frequency, max_frequency)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(SecondLocator(interval=60*2))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.0f}"))
    ax.tick_params(axis="x", rotation=0, labelrotation=0, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    secax = ax.secondary_yaxis(
        "right",
        functions=(
            lambda f_mhz: _freq_to_r(f_mhz, branch="H", factor=model_factor),
            lambda r_rs: _r_to_freq(r_rs, branch="H", factor=model_factor),
        ),
    )
    secax.set_ylabel(f"Radial distance (Harmonic) [R$_\\odot$] ({model_factor}× Saito1977)", fontsize=14)
    secax.tick_params(axis="y", labelsize=12)
    secax.yaxis.set_major_locator(MultipleLocator(0.1))
    secax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.1f}"))

    ax.legend(fontsize=12, loc="upper right")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def main() -> None:
    start_time = "2022-06-13T03:25:00"
    end_time = "2022-06-13T03:37:00"
    min_frequency = 23
    max_frequency = 47
    use_removed_background = True
    output_path = RESEARCH_ROOT / "RadioData" / "HF_plot" / "output" / "bandsplit_fig1_with_main_peak_points.png"

    plot_bandsplit_fig1_with_main_peak_points(
        start_time=start_time,
        end_time=end_time,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        use_removed_background=use_removed_background,
        output_path=output_path,
        show=True,
    )


if __name__ == "__main__":
    main()
