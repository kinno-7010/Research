import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, LogLocator, FuncFormatter
from matplotlib.dates import SecondLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import median_filter
from scipy.stats import zscore
from scipy.stats import linregress

try:
    from .utils import _to_datetime, _slice_data
    from .spectrum_plot import time, frequency_mhz, data
except ImportError:
    from utils import _to_datetime, _slice_data
    from spectrum_plot import time, frequency_mhz, data

def calculate_fit_with_error(ax, times, freqs):
    """
    times: np.ndarray of datetime
    freqs: np.ndarray of float (MHz)
    Returns (slope, stderr)
    """
    # datetime → プロット用数字
    xnum = mdates.date2num(times)
    # 秒に変換
    t_sec = (xnum - xnum[0]) * 86400.0

    # 線形回帰
    res = linregress(t_sec, freqs)
    slope, intercept, stderr = res.slope, res.intercept, res.stderr

    return slope, intercept, stderr

def calculate_dynamic_spectrum_with_peak(
    fig, ax,
    start_time, end_time,
    freq_min, freq_max,
    time_tick_sec, freq_tick_mhz,
    med_filter_size,
    vmin, vmax,
    title, scatter_color,
    outlier_z=3.0,
    time_array=None, freq_array=None, data_array=None
):
    """
    Plot dynamic spectrum and mark peak frequencies over time.
    Filters out z-score outliers among detected peaks.
    Returns nothing.
    """
    # Convert inputs
    start_dt = _to_datetime(start_time)
    end_dt = _to_datetime(end_time)

    # Use provided arrays when指定, otherwise fall back to globals
    t_arr = time if time_array is None else time_array
    f_arr = frequency_mhz if freq_array is None else freq_array
    d_arr = data if data_array is None else data_array

    # Slice data
    t_sel, f_sel, d_sel = _slice_data(
        t_arr, d_arr, start_dt, end_dt,
        f_arr, freq_min, freq_max
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
            s=30, facecolors=scatter_color,
            edgecolors='black', linewidth=0.8,
            alpha=0.9, zorder=5
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
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="1%", pad=0.1)
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.ax.tick_params(labelsize=14)
    # cbar.set_label('Intensity (dB)', fontsize=16)

    return peak_times, peak_freqs


def calculate_peak_time_and_freq(
    time_array, freq_mhz, data_array,
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
    t_sel, f_sel, d_sel = _slice_data(
        time_array, data_array, t0, t1,
        freq_mhz, freq_min, freq_max
    )

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
    fig, ax,
    start_time, end_time,
    freq_min, freq_max,
    time_tick_sec, freq_tick_mhz,
    med_filter_size, vmin, vmax, title, scatter_color,
    outlier_z=3.0
):
    """
    Apply 3σ cleaning in 35–40 MHz, then plot dynamic spectrum with peaks.
    """
    # 3σ cleaning in specified band
    freq_array = np.asarray(frequency_mhz, dtype=float)
    mask_band = (freq_array >= 35) & (freq_array <= 40)
    band_data = data[:, mask_band]
    flat = band_data.flatten()
    mu, sigma = np.mean(flat), np.std(flat)
    clean = flat[(flat < mu + 3*sigma) & (flat > mu - 3*sigma)]
    threshold = np.mean(clean) * 1.05

    # Mask data below threshold
    masked_data = np.where(data > threshold, data, np.nan)

    # Delegate to calculate_dynamic_spectrum_with_peak
    calculate_dynamic_spectrum_with_peak(
        fig, ax,
        start_time, end_time,
        freq_min, freq_max,
        time_tick_sec, freq_tick_mhz,
        med_filter_size,
        vmin, vmax,
        title, scatter_color,
        outlier_z=outlier_z,
        time_array=time,
        freq_array=frequency_mhz,
        data_array=masked_data
    )
    
    return masked_data

def plot_removed_dynamic_spectrum_with_peak_2(
    fig, ax,
    start_time, end_time,
    freq_min, freq_max,
    time_tick_sec, freq_tick_mhz,
    med_filter_size, vmin, vmax,
    title,
    scatter_color='blue',
    threshold_high=93,
    outlier_z=3.0
):
    """
    Apply 3σ cleaning in 35–40 MHz, then plot dynamic spectrum with peaks.
    Lower threshold = mean of 3σ-cleaned band * 1.05
    Upper threshold = threshold_high
    """
    # 1) 35–40 MHz帯だけを取り出し、flatten
    mask_band = (frequency_mhz >= 35) & (frequency_mhz <= 40)
    band_data = data[:, mask_band]
    flat = band_data.flatten()

    # 2) 3σクリーニングして下限閾値を決定
    mu, sigma = np.mean(flat), np.std(flat)
    clean = flat[(flat < mu + 3*sigma) & (flat > mu - 3*sigma)]
    threshold_low = np.mean(clean) * 1.05

    # 3) 元の data に対して閾値マスク
    mask = (data > threshold_low) & (data < threshold_high)
    masked_data = np.where(mask, data, np.nan)

    # 4) クリーニング後スペクトルをピークプロット関数へ委譲
    peak_times, peak_freqs = calculate_dynamic_spectrum_with_peak(
        fig, ax,
        start_time, end_time, freq_min, freq_max,
        time_tick_sec, freq_tick_mhz, med_filter_size,
        vmin, vmax, title, scatter_color,
        outlier_z=outlier_z,
        time_array=time,
        freq_array=frequency_mhz,
        data_array=masked_data
    )
    
    ax.scatter(
        mdates.date2num(peak_times),
        peak_freqs,
        s=30, facecolors=scatter_color,
        edgecolors='black', linewidth=0.8,
        alpha=0.9, zorder=5
    )
    
    return masked_data


if __name__ == "__main__":
    import os

    print("=" * 60)
    print("HF Radio Peak Analysis Tool")
    print("=" * 60)
    print()
    
    # Load data
    if not (time.size and frequency_mhz.size and data.size):
        print("Failed to load data. Exiting...")
        exit(1)
    
    # --- Define the output directory path ---
    output_dir = '../output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nPlotting cleaned dynamic spectrum with peak detection...")
    try:
        from .utils import _initialize_plot_parameters
    except ImportError:
        from utils import _initialize_plot_parameters
    
    start_time, end_time, freq_min, freq_max, time_tick_sec, freq_tick_mhz, med_filter_size, vmin, vmax, title = _initialize_plot_parameters()
    
    # Peak analysis parameters
    scatter_color = 'red'
    outlier_z = 3.0
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    plot_removed_dynamic_spectrum_with_peak(
        fig, ax,
        start_time, end_time,
        freq_min, freq_max,
        time_tick_sec, freq_tick_mhz,
        med_filter_size, vmin, vmax, title, scatter_color
    )
    ax.set_xlabel('Time (UTC)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/removed_HF_peak_analysis_{start_time}-{end_time}_{freq_min}-{freq_max}MHz.png')
    print(f'Saved: {output_dir}/removed_HF_peak_analysis_{start_time}-{end_time}_{freq_min}-{freq_max}MHz.png')
    plt.show()