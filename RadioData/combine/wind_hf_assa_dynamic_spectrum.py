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

import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np
import pandas as pd
from astropy.io import fits
from cdflib import CDF, cdfepoch
from matplotlib.dates import SecondLocator, MinuteLocator
# Fixed data locations
WIND_CDF_PATH = Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/Wind/Rawdata/wi_l2_wav_rad2_20220613_v01.cdf")
HF_CDF_PATH = Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/HF_plot/Rawdata/it_h1_hf_20220613_v01.cdf")
ASSA_FITS_PATHS = [
    Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_031500_62.fit"),
    Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_033000_62.fit"),
]
DEFAULT_OUTPUT_FIGURE = Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/combine/wind_hf_assa_dynamic_spectrum.png")
DEFAULT_OUTPUT_CSV = Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/combine/wind_hf_assa_dynamic_spectrum.csv")


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


def draw_line_between_points(
    ax,
    point_start_time,
    point_end_time,
    point_start_frequency,
    point_end_frequency,
    color,
    start_time, end_time, min_frequency, max_frequency,  # retained for compatibility
    **plot_kwargs
) -> tuple[plt.Line2D | None, float | None]:
    """
    Draw a line defined in *linear data coordinates* (time vs frequency),
    clipped to the data-rectangle [start_time,end_time]×[min_frequency,max_frequency].
    This guarantees: 
      - 線形軸なら真っ直ぐ 
      - ログ軸では（期待通り）曲線
    and it passes exactly through the two specified points if they are inside bounds.
    """
    import numpy as np
    import matplotlib.dates as mdates
    import math
    import pandas as pd

    if (point_start_time is None or point_end_time is None or
        point_start_frequency is None or point_end_frequency is None):
        return None, None

    # ---- convert to linear data coords ----
    t1 = pd.Timestamp(point_start_time)
    t2 = pd.Timestamp(point_end_time)
    x1 = mdates.date2num(t1)           # days
    x2 = mdates.date2num(t2)
    y1 = float(point_start_frequency)  # MHz
    y2 = float(point_end_frequency)

    if math.isclose(x1, x2) and math.isclose(y1, y2):
        return None, None

    # slope in MHz/s (info用)
    delta_seconds = (t2 - t1).total_seconds()
    slope_mhz_per_sec = math.inf if math.isclose(delta_seconds, 0.0) else (y2 - y1) / delta_seconds

    # data-rect bounds (linear)
    x_min = mdates.date2num(pd.Timestamp(start_time))
    x_max = mdates.date2num(pd.Timestamp(end_time))
    y_min = float(min_frequency)
    y_max = float(max_frequency)

    dx = x2 - x1
    dy = y2 - y1

    # ---- find intersections with the data rectangle in *data coords* ----
    cand = []  # list of (x,y)
    eps = 1e-12

    def add_if_inside(x, y):
        if (x_min - eps) <= x <= (x_max + eps) and (y_min - eps) <= y <= (y_max + eps):
            cand.append((x, y))

    # x = const edges
    if not math.isclose(dx, 0.0):
        y_at_xmin = y1 + dy * (x_min - x1) / dx
        y_at_xmax = y1 + dy * (x_max - x1) / dx
        add_if_inside(x_min, y_at_xmin)
        add_if_inside(x_max, y_at_xmax)

    # y = const edges
    if not math.isclose(dy, 0.0):
        x_at_ymin = x1 + dx * (y_min - y1) / dy
        x_at_ymax = x1 + dx * (y_max - y1) / dy
        add_if_inside(x_at_ymin, y_min)
        add_if_inside(x_at_ymax, y_max)

    # remove duplicates (corner hits etc.)
    uniq = []
    for p in cand:
        if not any(abs(p[0]-q[0]) < 1e-9 and abs(p[1]-q[1]) < 1e-9 for q in uniq):
            uniq.append(p)

    # Fallback: if numerical issues leave <2 intersections, use the two user points
    if len(uniq) < 2:
        xs = [x1, x2]
        ys = [y1, y2]
    else:
        # two endpoints of the clipped infinite line
        # (choose the most separated two by projection onto direction vector)
        import numpy as np
        v = np.array([dx, dy])
        dots = [np.dot(np.array(p) - np.array([x1, y1]), v) for p in uniq]
        A = uniq[int(np.argmin(dots))]
        B = uniq[int(np.argmax(dots))]

        # ---- sample densely in *data coords* along AB ----
        # ensure the two user-specified points are included if they lie inside bounds
        anchors = [A, B]
        if (x_min <= x1 <= x_max) and (y_min <= y1 <= y_max):
            anchors.append((x1, y1))
        if (x_min <= x2 <= x_max) and (y_min <= y2 <= y_max):
            anchors.append((x2, y2))

        # param t along direction v using x or y for robustness
        def param_t(x, y):
            if abs(dx) >= abs(dy) and not math.isclose(dx, 0.0):
                return (x - x1) / dx
            else:
                return (y - y1) / dy if not math.isclose(dy, 0.0) else 0.0

        ts = sorted({param_t(px, py) for (px, py) in anchors})
        t_min, t_max = ts[0], ts[-1]

        # dense sampling (includes anchors)
        t_dense = np.linspace(t_min, t_max, 400)
        t_all = np.unique(np.concatenate([t_dense, np.array(ts)]))
        xs = x1 + t_all * dx
        ys = y1 + t_all * dy

    # ---- draw (Matplotlib will transform to log if必要) ----
    line = ax.plot(
        xs, ys,
        **({"color": color, "linestyle": "--", "linewidth": 1.2, "alpha": 0.7} | plot_kwargs),
    )[0]

    return line, slope_mhz_per_sec


def plot_dynamic_spectrum(
    spectrum: pd.DataFrame,
    output_path: Path | None,
    title: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    log_scale: bool,
    show: bool,
    draw_lines: bool,
    start_time, 
    end_time,
    min_frequency,
    max_frequency,
    point_start_time_1=None,
    point_end_time_1=None,
    point_start_frequency_1=None,
    point_end_frequency_1=None,
    point_start_time_2=None,
    point_end_time_2=None,
    point_start_frequency_2=None,
    point_end_frequency_2=None,
) -> None:
    """Generate and optionally save the combined dynamic spectrum figure."""
    print("----------------figure export----------------")
    if spectrum.empty:
        raise ValueError("Combined spectrum is empty. Check the time range and input files.")

    time_axis = spectrum.index.to_pydatetime()
    freq_axis = spectrum.columns.to_numpy()
    values = spectrum.to_numpy().T  # shape -> (freq, time)

    # fig, ax = plt.subplots(figsize=(18, 8))
    fig, ax = plt.subplots(figsize=(12, 6))
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
    ax.text(mdates.date2num(end_time), 40.0, "Iitate HF antenna", color="white", fontsize=14, ha="right", va="top", fontweight="bold")
    ax.text(mdates.date2num(end_time), 85.0, "Australia-ASSA", color="white", fontsize=14, ha="right", va="top", fontweight="bold")
    
    CME_time = mdates.date2num(pd.Timestamp("2022-06-13 03:12:00"))
    flare_time = mdates.date2num(pd.Timestamp("2022-06-13 04:07:00"))
    SRB_start_time = mdates.date2num(pd.Timestamp("2022-06-13 03:25:40"))
    SRB_end_time = mdates.date2num(pd.Timestamp("2022-06-13 03:31:20"))
    SRB_end_time_harmonic = mdates.date2num(pd.Timestamp("2022-06-13 03:50:00"))
    # ax.axvline(CME_time, color="white", linestyle="--", linewidth=1)
    # ax.text(CME_time, 5.7, " Main CME\n erupted\n 03:12 UT", color="white", fontsize=14, ha="left", va="bottom", fontweight="bold")
    # ax.axvline(flare_time, color="white", linestyle="--", linewidth=1)
    # # ax.text(flare_time, 85, " M3.4 flare peaked\n 04:07 UT", color="white", fontsize=14, ha="left", va="top", fontweight="bold")
    # ax.axvline(SRB_start_time, color="red", linestyle="--", linewidth=1)
    # # ax.text(SRB_start_time, 85, " SRB II start\n 03:25:40 UT", color="red", fontsize=14, ha="left", va="top", fontweight="bold")
    # ax.axvline(SRB_end_time, color="red", linestyle="--", linewidth=1)
    # # ax.text(SRB_end_time, 85, " SRB II end\n 03:31:20 UT", color="red", fontsize=14, ha="left", va="top", fontweight="bold")
    # ax.axvline(SRB_end_time_harmonic, color="red", linestyle="--", linewidth=1)
    # ax.text(SRB_end_time_harmonic, 85, " SRB II (Harmonic) end\n 03:50:00 UT", color="red", fontsize=14, ha="left", va="top", fontweight="bold")
    
    # ax.text(mdates.date2num(pd.Timestamp("2022-06-13T05:00:00")), 8, "Second Harmonic ", fontsize=14, ha="right", va="bottom", fontweight="bold", color="pink")
    # ax.text(mdates.date2num(pd.Timestamp("2022-06-13T05:00:00")), 4, "Fundamental ", fontsize=14, ha="right", va="top", fontweight="bold", color="#A0E4FF")
    
    ax.scatter(mdates.date2num(pd.Timestamp("2022-06-13T03:25:40")), 34.8, color="magenta", marker="o", s=100, label="03:25:40 UT (34.8 MHz)")
    ax.scatter(mdates.date2num(pd.Timestamp("2022-06-13T03:28:45")), 31.5, color="#00FF00", marker="o", s=100, label="03:28:45 UT (31.5 MHz)")
    ax.scatter(mdates.date2num(pd.Timestamp("2022-06-13T03:31:20")), 28.0, color="cyan", marker="o", s=100, label="03:31:20 UT (28.0 MHz)")

    
    ax.set_ylabel("Frequency [MHz]", fontsize=16)
    if log_scale is not False:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
    ax.set_xlabel("Time [UT]", fontsize=16)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
    ax.set_ylim(min_frequency, max_frequency)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(SecondLocator(interval=60))
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.0f}"))
    ax.tick_params(axis="x", rotation=0, labelrotation=0, labelsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="lower right", fontsize=14)

    if draw_lines:
        line1, slope1 = draw_line_between_points(
            ax,
            point_start_time=point_start_time_1,
            point_end_time=point_end_time_1,
            point_start_frequency=point_start_frequency_1,
            point_end_frequency=point_end_frequency_1,
            start_time=start_time,
            end_time=end_time,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            color="red",
        )
        line2, slope2 = draw_line_between_points(
            ax,
            point_start_time=point_start_time_2,
            point_end_time=point_end_time_2,
            point_start_frequency=point_start_frequency_2,
            point_end_frequency=point_end_frequency_2,
            start_time=start_time,
            end_time=end_time,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            color="cyan",
        )

        legend_handles: list[plt.Line2D] = []
        legend_labels: list[str] = []

        def format_slope(tag: str, slope: float | None) -> str | None:
            if slope is None:
                return None
            if math.isinf(slope):
                return f"{tag}: slope = ∞ MHz/s"
            return f"{tag}: slope = {slope:+.5f} MHz/s"

        label1 = format_slope("Line 1", slope1)
        if line1 is not None and label1 is not None:
            legend_handles.append(line1)
            legend_labels.append(label1)

        label2 = format_slope("Line 2", slope2)
        if line2 is not None and label2 is not None:
            legend_handles.append(line2)
            legend_labels.append(label2)

        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc="lower right")



    # cbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.5)
    # cbar.set_label(
    #     "Intensity normalized to per-frequency median"
    #     + (" (log10)" if log_scale else ""),
    #     fontsize=14,
    # )

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
    
    
def Saito1977(rho):  # 2.5-5.5Rs
    C1 = [1.36e6, 5.27e6, 3.15e6]
    C2 = [1.68e8, 3.54e6, 1.60e6]
    d1 = [2.14, 3.30, 4.71]
    d2 = [6.13, 5.80, 3.01]

    # Guard against rho <= 0 to avoid zero being raised to negative powers
    rho_safe = np.clip(np.asarray(rho, dtype=float), 1e-6, None)

    background = C1[0]*rho_safe**(-d1[0])+C2[0]*rho_safe**(-d2[0])
    eq_hole = C1[1]*rho_safe**(-d1[1])+C2[1]*rho_safe**(-d2[1])
    pole_hole = C1[2]*rho_safe**(-d1[2])+C2[2]*rho_safe**(-d2[2])
    return background #, eq_hole, pole_hole    
    


def main(
    start_time,
    end_time,
    min_frequency,
    max_frequency,
    draw_lines: bool = True,
    point_start_time=None,
    point_end_time=None,
    point_start_frequency=None,
    point_end_frequency=None,
    point_start_time_2=None,
    point_end_time_2=None,
    point_start_frequency_2=None,
    point_end_frequency_2=None,
) -> None:
    # ----- Configuration section (edit as needed) -----
    cadence = "0.5s"
    polarization = "RH"  # or "LH"
    show_plot = True
    log_scale = True
    cmap = "viridis"
    vmin = None
    vmax = None
    point_start_time = pd.Timestamp(point_start_time) if point_start_time is not None else None
    point_end_time = pd.Timestamp(point_end_time) if point_end_time is not None else None
    point_start_frequency = float(point_start_frequency) if point_start_frequency is not None else None
    point_end_frequency = float(point_end_frequency) if point_end_frequency is not None else None
    point_start_time_2 = pd.Timestamp(point_start_time_2) if point_start_time_2 is not None else None
    point_end_time_2 = pd.Timestamp(point_end_time_2) if point_end_time_2 is not None else None
    point_start_frequency_2 = float(point_start_frequency_2) if point_start_frequency_2 is not None else None
    point_end_frequency_2 = float(point_end_frequency_2) if point_end_frequency_2 is not None else None

    point_start_time_1 = point_start_time
    point_end_time_1 = point_end_time
    point_start_frequency_1 = point_start_frequency
    point_end_frequency_1 = point_end_frequency
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

    data_output_path = Path(f"/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/combine/wind_hf_assa_dynamic_spectrum_{start_time.strftime('%Y-%m-%d_%H%M%S')}_{end_time.strftime('%H%M%S')}.csv")
    figure_path = Path(f"/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/combine/wind_hf_assa_dynamic_spectrum_{start_time.strftime('%Y-%m-%d_%H%M%S')}_{end_time.strftime('%H%M%S')}.png")
    data_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    # export_dataframe(combined, data_output_path)
    
    plot_dynamic_spectrum(
        combined,
        figure_path,
        title=title,
        cmap=cmap,
        vmin=1.0,
        vmax=1.1,
        log_scale=log_scale,
        show=show_plot,
        draw_lines=draw_lines,
        start_time=start_time,
        end_time=end_time,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        point_start_time_1=point_start_time_1,
        point_end_time_1=point_end_time_1,
        point_start_frequency_1=point_start_frequency_1,
        point_end_frequency_1=point_end_frequency_1,
        point_start_time_2=point_start_time_2,
        point_end_time_2=point_end_time_2,
        point_start_frequency_2=point_start_frequency_2,
        point_end_frequency_2=point_end_frequency_2,
    )


if __name__ == "__main__":
    # start_time = pd.Timestamp("2022-06-13 03:00:00")
    # end_time = pd.Timestamp("2022-06-13 05:00:00")
    # min_frequency = 1.0
    # max_frequency = 90.0
    # draw_lines = False
    
    start_time = pd.Timestamp("2022-06-13 03:25:00")
    end_time = pd.Timestamp("2022-06-13 03:34:00")
    min_frequency = 25.0
    max_frequency = 38.0
    draw_lines = False
    
    # harmonic
    point_start_time_1 = pd.Timestamp("2022-06-13 03:26:00")
    point_end_time_1 = pd.Timestamp("2022-06-13 03:32:00")
    point_start_frequency_1 = 34.5
    point_end_frequency_1 = 27.5
    
    # Fundamental
    point_start_time_2 = pd.Timestamp("2022-06-13 03:26:00")
    point_end_time_2 = pd.Timestamp("2022-06-13 03:32:00")
    point_start_frequency_2 = 17.25
    point_end_frequency_2 = 13.75
    
    main(
        start_time=start_time,
        end_time=end_time,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        draw_lines=draw_lines,
        point_start_time=None,
        point_end_time=None,
        point_start_frequency=None,
        point_end_frequency=None,
    )
