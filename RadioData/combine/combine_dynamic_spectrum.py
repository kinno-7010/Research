#!/usr/bin/env python3
"""
Utility to merge Wind/RAD2 (1–14 MHz), Iitate HF (15–40 MHz), and
e-Callisto/ASSA (40–85 MHz) dynamic spectra into a single frequency-stitched
spectrogram.

Paths and time span are provided via configurable constants. The script creates a
common time grid (default 1 s cadence), normalizes each instrument by its
per-frequency median, joins the spectra along the frequency axis, and saves a
figure. Combined data can optionally be exported as a CSV file for further
analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np
import pandas as pd
from astropy.io import fits
from cdflib import CDF, cdfepoch
from matplotlib.dates import SecondLocator, MinuteLocator
# Fixed data locations
WIND_CDF_PATH = Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/Wind/Rawdata/wi_l2_wav_rad2_20220613_v01.cdf")
HF_CDF_PATH = Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/HF_plot/Rawdata/it_h1_hf_20220613_v01.cdf")
ASSA_FITS_PATHS = [
    Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_031500_62.fit"),
    Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_033000_62.fit"),
]
DEFAULT_OUTPUT_FIGURE = Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/combine/wind_hf_assa_dynamic_spectrum.png")
DEFAULT_OUTPUT_CSV = Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/combine/wind_hf_assa_dynamic_spectrum.csv")


def load_wind_rad2(path: Path) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Load Wind/WAVES RAD2 dynamic spectrum."""
    cdf_obj = CDF(str(path))
    epoch = cdf_obj.varget("Epoch")
    frequency_hz = cdf_obj.varget("FREQUENCY").astype(float)
    psd = cdf_obj.varget("PSD_V2_S").astype(float)
    background = cdf_obj.varget("BACKGROUND_S").astype(float)

    freq_mhz = frequency_hz / 1e6
    times = pd.to_datetime(cdfepoch.to_datetime(epoch))

    with np.errstate(divide="ignore", invalid="ignore"):
        intensity = np.divide(
            psd,
            background,
            out=np.full_like(psd, np.nan, dtype=float),
            where=background > 0,
        )

    return pd.DatetimeIndex(times), freq_mhz, intensity


def load_hf(path: Path, polarization: str) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Load HF spectrograph data (Iitate observatory)."""
    cdf_obj = CDF(str(path))
    epoch = cdf_obj.varget("Epoch")
    frequency_hz = cdf_obj.varget("Frequency").astype(float)
    intensity = cdf_obj.varget(polarization.upper()).astype(float)

    times = pd.to_datetime(cdfepoch.to_datetime(epoch))
    freq_mhz = frequency_hz / 1e6

    return pd.DatetimeIndex(times), freq_mhz, intensity


def read_callisto_file(path: Path) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Read a single e-Callisto FITS file and return time, frequency, data arrays."""
    with fits.open(path) as hdul:
        primary = hdul[0]
        calibration = hdul[1].data[0]

        start_time = pd.Timestamp(f"{primary.header['DATE-OBS']} {primary.header['TIME-OBS']}")
        time_offsets = calibration[0].astype(float)
        frequencies = calibration[1].astype(float)

        # Convert to ascending MHz order for consistency across instruments.
        freq_mhz = frequencies[::-1]
        times = start_time + pd.to_timedelta(time_offsets, unit="s")

        # Primary data shape: (frequency_index, time_index)
        data = primary.data.astype(float)[::-1, :].T

    return pd.DatetimeIndex(times), freq_mhz, data


def load_callisto(paths: Sequence[Path]) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Load and stitch multiple Callisto FITS files along the time axis."""
    times_list: List[pd.DatetimeIndex] = []
    data_list: List[np.ndarray] = []

    freq_reference: np.ndarray | None = None

    for path in sorted(paths):
        times, freqs, data = read_callisto_file(path)
        times_list.append(times)
        data_list.append(data)

        if freq_reference is None:
            freq_reference = freqs
        else:
            if not np.allclose(freq_reference, freqs):
                raise ValueError(f"Frequency axis mismatch in {path}")

    if freq_reference is None:
        raise ValueError("No Callisto files supplied.")

    combined_times = times_list[0]
    for additional in times_list[1:]:
        combined_times = combined_times.append(additional)
    combined_data = np.vstack(data_list)

    return combined_times, freq_reference, combined_data


def create_dataframe(
    times: pd.DatetimeIndex,
    freqs_mhz: np.ndarray,
    values: np.ndarray,
    time_range: Tuple[pd.Timestamp, pd.Timestamp] | None,
) -> pd.DataFrame:
    """Convert arrays to a (time, frequency) DataFrame with optional time trimming."""
    freq_index = pd.Index(np.round(freqs_mhz.astype(float), 6)).astype(float)
    df = pd.DataFrame(values, index=times, columns=freq_index)
    df = collapse_duplicate_frequencies(df)

    if time_range is not None:
        start, end = time_range
        df = df.loc[(df.index >= start) & (df.index <= end)]

    return df


def collapse_duplicate_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """Average columns that share the same frequency label."""
    collapsed = df.T.groupby(level=0).mean().T
    collapsed.columns.name = "frequency_mhz"
    return collapsed


def resample_to_grid(
    df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    rule: str,
) -> pd.DataFrame:
    """
    Resample to the target cadence and interpolate missing samples.

    The resample operation averages within each bin (downsampling) or inserts NaNs
    for empty bins (upsampling). Linear interpolation in time fills interior gaps
    while leaving leading/trailing gaps as NaNs.
    """
    if df.empty:
        return df.reindex(target_index)

    resampled = df.resample(rule).mean()
    aligned = resampled.reindex(target_index)
    interpolated = aligned.interpolate(method="time", limit_direction="forward")
    return interpolated


def normalize_by_median(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each frequency channel by its median value."""
    medians = df.median(axis=0, skipna=True).replace(0, np.nan)
    normalized = df.divide(medians)
    return normalized


def combine_spectra(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate spectra along the frequency axis and sort columns."""
    merged = pd.concat(frames, axis=1)
    merged = merged.sort_index(axis=1)
    return merged


def plot_dynamic_spectrum(
    spectrum: pd.DataFrame,
    output_path: Path | None,
    title: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    log_scale: bool,
    show: bool,
    start_time, 
    end_time,
) -> None:
    """Generate and optionally save the combined dynamic spectrum figure."""
    print("----------------figure export----------------")
    if spectrum.empty:
        raise ValueError("Combined spectrum is empty. Check the time range and input files.")

    time_axis = spectrum.index.to_pydatetime()
    freq_axis = spectrum.columns.to_numpy()
    values = spectrum.to_numpy().T  # shape -> (freq, time)

    fig, ax = plt.subplots(figsize=(18, 8))
    mesh = ax.pcolormesh(
        mdates.date2num(time_axis),
        freq_axis,
        values,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    
    ax.axhline(14.0, color="white", linestyle="--", linewidth=1)
    ax.text(mdates.date2num(end_time), 14.0, "Wind/RAD2", color="white", fontsize=14, ha="right", va="top", fontweight="bold")
    ax.axhline(40.0, color="white", linestyle="--", linewidth=1)
    ax.text(mdates.date2num(end_time), 40.0, "HF antenna", color="white", fontsize=14, ha="right", va="top", fontweight="bold")
    ax.text(mdates.date2num(end_time), 85.0, "Australia-ASSA", color="white", fontsize=14, ha="right", va="top", fontweight="bold")
    
    ax.set_ylabel("Frequency [MHz]", fontsize=16)
    ax.set_yscale("log")
    ax.set_xlabel("Time [UT]", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(SecondLocator(interval=10*60))
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.0f}"))
    ax.tick_params(axis="x", rotation=0, labelrotation=0, labelsize=14)
    ax.tick_params(axis="y", labelsize=12)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.5)
    cbar.set_label(
        "Intensity normalized to per-frequency median"
        + (" (log10)" if log_scale else ""),
        fontsize=14,
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {output_path} \n")

    if show:
        plt.show()
    else:
        plt.close(fig)


def export_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Export combined spectrum to CSV."""
    print("----------------csv export----------------")
    df.to_csv(output_path, index_label="time")
    print(f"Data exported to {output_path} \n")

def main(start_time, end_time, min_frequency, max_frequency) -> None:
    # ----- Configuration section (edit as needed) -----
    cadence = "0.5s"
    polarization = "RH"  # or "LH"
    show_plot = True
    log_scale = False
    cmap = "viridis"
    vmin = None
    vmax = None
    # -------------------------------------------------

    if start_time >= end_time:
        raise ValueError("Start time must be earlier than end time.")

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

    title = "Dynamic Spectrum; Wind/RAD2 (1-14 MHz) + HF antenna (14-40 MHz) + Australia-ASSA (40-85 MHz)"
    if combined.empty:
        raise ValueError("No data remains after applying the frequency bounds.")

    data_output_path = Path(f"/mnt/d/wsl/home/kinno-7010/Research/RadioData/combine/wind_hf_assa_dynamic_spectrum_{start_time.strftime('%Y-%m-%d_%H%M%S')}_{end_time.strftime('%H%M%S')}.csv")
    figure_path = Path(f"/mnt/d/wsl/home/kinno-7010/Research/RadioData/combine/wind_hf_assa_dynamic_spectrum_{start_time.strftime('%Y-%m-%d_%H%M%S')}_{end_time.strftime('%H%M%S')}.png")
    data_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    export_dataframe(combined, data_output_path)
    
    plot_dynamic_spectrum(
        combined,
        figure_path,
        title=title,
        cmap=cmap,
        vmin=1.0,
        vmax=1.1,
        log_scale=log_scale,
        show=show_plot,
        start_time=start_time,
        end_time=end_time,
    )


if __name__ == "__main__":
    main(
        start_time=pd.Timestamp("2022-06-13 03:00:00"),
        end_time=pd.Timestamp("2022-06-13 05:00:00"),
        min_frequency=1.0,
        max_frequency=90.0,
    )
