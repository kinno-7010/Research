import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, LogLocator, FuncFormatter
from matplotlib.dates import SecondLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import median_filter
from scipy.stats import zscore
from .utils import _to_datetime, _slice_data

def calculate_dynamic_spectrum_with_peak(
    fig, ax, time_array, freq_mhz, data,
    start_time, end_time,
    freq_min, freq_max,
    time_tick_sec, freq_tick_mhz,
    med_filter_size,
    vmin, vmax,
    title, scatter_color,
    outlier_z
):
    """
    Plot dynamic spectrum and mark peak frequencies over time.
    Filters out z-score outliers among detected peaks.
    Returns nothing.
    """
    # Convert inputs
    start_dt = _to_datetime(start_time)
    end_dt = _to_datetime(end_time)

    # Slice data
    t_sel, f_sel, d_sel = _slice_data(
        time_array, data, start_dt, end_dt,
        freq_mhz, freq_min, freq_max
    )

    # Median filter to reduce noise
    d_filt = median_filter(d_sel.astype(float), size=med_filter_size)

    # ── Detect peak frequency for each timestep ─────────────
    peak_times, peak_freqs = [], []
    for idx, row in enumerate(d_filt):
        if np.all(np.isnan(row)):
            continue
        max_val = np.nanmax(row)
        candidates = np.where(row == max_val)[0]
        # 複数候補があれば、その周波数の平均を取る
        freqs = f_sel[candidates]
        peak_freq = freqs.mean()
        peak_times.append(t_sel[idx])
        peak_freqs.append(peak_freq)

    peak_times = np.array(peak_times)
    peak_freqs = np.array(peak_freqs, float)

    # ── Outlier removal by z-score ─────────────────────────
    if peak_freqs.size:
        mask_inlier = np.abs(zscore(peak_freqs)) <= outlier_z
        peak_times = peak_times[mask_inlier]
        peak_freqs = peak_freqs[mask_inlier]

    # ── Plot the spectrum ──────────────────────────────────
    extent = [
        mdates.date2num(t_sel[0]), mdates.date2num(t_sel[-1]),
        f_sel[0], f_sel[-1]
    ]
    im = ax.imshow(
        d_filt.T, origin='lower', aspect='auto',
        extent=extent, cmap='viridis', vmin=vmin, vmax=vmax
    )

    # ── Mark peak points ───────────────────────────────────
    if peak_times.size > 0:
        ax.scatter(
            mdates.date2num(peak_times), peak_freqs,
            c=scatter_color, s=1.5, alpha=0.7
        )

    # ── Plot formatting ────────────────────────────────────
    ax.set_title(title, fontsize=18)
    ax.set_ylabel('Frequency (MHz)', fontsize=16)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.xaxis.set_major_locator(SecondLocator(interval=time_tick_sec))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.yaxis.set_major_locator(MultipleLocator(freq_tick_mhz))
    ax.tick_params(axis='both', which='major', labelsize=14)

    # ── Colorbar ───────────────────────────────────────────
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Intensity (dB)', fontsize=16)


def calculate_peak_time_and_freq(
    time_array, freq_mhz, data,
    start_time, end_time,
    freq_min, freq_max,
    med_filter_size,
    outlier_z,
):
    """
    指定区間・周波数帯のピーク時刻・周波数を抽出し、
    周波数の z-score 外れ値除去まで行う。
    Returns:
      times: np.ndarray of datetime
      freqs: np.ndarray of float (MHz)
    """
    # 1) 時刻変換
    t0 = _to_datetime(start_time)
    t1 = _to_datetime(end_time)

    # 2) スライス
    mask_t = (time_array >= t0) & (time_array <= t1)
    t_sel = time_array[mask_t]
    d_sel = data[mask_t, :]
    mask_f = (freq_mhz >= freq_min) & (freq_mhz <= freq_max)
    f_sel = freq_mhz[mask_f]
    d_sel = d_sel[:, mask_f]

    # 3) メディアンフィルタ
    d_filt = median_filter(d_sel.astype(float), size=med_filter_size)

    # 4) 各時刻ピーク抽出
    peak_times = []
    peak_freqs = []
    for i, row in enumerate(d_filt):
        if np.all(np.isnan(row)):
            continue
        m = np.nanmax(row)
        idxs = np.where(row == m)[0]
        freq_peak = f_sel[idxs].mean()
        peak_times.append(t_sel[i])
        peak_freqs.append(freq_peak)

    # 5) np.array化
    times = np.array(peak_times)
    freqs = np.array(peak_freqs, float)

    # 6) 周波数の外れ値除去
    if len(freqs) >= 2:
        mask = np.abs(zscore(freqs)) <= outlier_z
        times = times[mask]
        freqs = freqs[mask]

    return times, freqs


def plot_removed_dynamic_spectrum_with_peak(
    fig, ax, time_array, freq_mhz, data,
    start_time, end_time,
    freq_min, freq_max,
    time_tick_sec, freq_tick_mhz,
    med_filter_size, vmin, vmax, title, scatter_color,
    outlier_z
):
    """
    Apply 3σ cleaning in 35–40 MHz, then plot dynamic spectrum with peaks.
    """
    # 3σ cleaning in specified band
    mask_band = (freq_mhz >= 35) & (freq_mhz <= 40)
    band_data = data[:, mask_band]
    flat = band_data.flatten()
    mu, sigma = np.mean(flat), np.std(flat)
    clean = flat[(flat < mu + 3*sigma) & (flat > mu - 3*sigma)]
    threshold = np.mean(clean) * 1.05

    # Mask data below threshold
    masked_data = np.where(data > threshold, data, np.nan)

    # Delegate to calculate_dynamic_spectrum_with_peak
    calculate_dynamic_spectrum_with_peak(
        fig, ax, time_array, freq_mhz, masked_data,
        start_time, end_time,
        freq_min, freq_max,
        time_tick_sec, freq_tick_mhz,
        med_filter_size,
        vmin, vmax,
        title, scatter_color,
        outlier_z
    )