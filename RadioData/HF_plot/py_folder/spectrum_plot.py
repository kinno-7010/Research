import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, LogLocator, FuncFormatter
from matplotlib.dates import SecondLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import median_filter
from .utils import _to_datetime, _slice_data

def plot_dynamic_spectrum(
    fig, ax, time_array, freq_mhz, data,
    start_time, end_time,
    freq_min, freq_max,
    time_tick_sec, freq_tick_mhz,
    med_filter_size,
    vmin, vmax,
    title
):
    """
    Plot a dynamic spectrum between start_time and end_time.
    """
    start_dt = _to_datetime(start_time)
    end_dt = _to_datetime(end_time)
    t_sel, f_sel, d_sel = _slice_data(
        time_array, data, start_dt, end_dt,
        freq_mhz, freq_min, freq_max
    )

    # Noise reduction
    d_filt = median_filter(d_sel.astype(float), size=med_filter_size)

    # Render image
    extent = [
        mdates.date2num(t_sel[0]), mdates.date2num(t_sel[-1]),
        f_sel[0], f_sel[-1]
    ]
    im=ax.imshow(
        d_filt.T, origin='lower', aspect='auto',
        extent=extent, cmap='viridis', vmin=vmin, vmax=vmax
    )
    # カラーバーの追加
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Intensity (dB)', fontsize=16)

    # Labels and formatting
    ax.set_title(title, fontsize=18)
    ax.set_ylabel('Frequency (MHz)', fontsize=16)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.xaxis.set_major_locator(SecondLocator(interval=time_tick_sec))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.yaxis.set_major_locator(MultipleLocator(freq_tick_mhz))
    ax.tick_params(axis='both', which='major', labelsize=14)


def plot_removed_dynamic_spectrum(
    fig, ax, time_array, freq_mhz, data,
    start_time, end_time,
    freq_min, freq_max,
    time_tick_sec, freq_tick_mhz,
    med_filter_size, vmin, vmax
):
    """
    Remove 3σ outliers in 35–40 MHz band and plot the cleaned dynamic spectrum.
    """
    # 35–40 MHz band extraction
    mask_band = (freq_mhz >= 35) & (freq_mhz <= 40)
    band_data = data[:, mask_band]
    flat = band_data.flatten()
    mu, sigma = np.mean(flat), np.std(flat)
    lower, upper = mu - 3*sigma, mu + 3*sigma
    clean = flat[(flat > lower) & (flat < upper)]
    clean_mean = np.mean(clean)

    # Mask below threshold
    threshold = clean_mean*1.05
    masked = np.where(data > threshold, data, np.nan)

    # Delegate to plot_dynamic_spectrum
    plot_dynamic_spectrum(
        fig, ax, time_array, freq_mhz, masked,
        start_time, end_time,
        freq_min, freq_max,
        time_tick_sec, freq_tick_mhz,
        med_filter_size,
        vmin, vmax,
        title='Second Harmonic Dynamic Spectrum (removed)'
    )


def plot_drift_line(
    ax, t0, t1, freq_start, freq_end,
    fmt='--', color='red', lw=2
):
    """
    Plot a drift line and mark endpoints on the given Axes.
    """
    t0_dt = _to_datetime(t0)
    t1_dt = _to_datetime(t1)

    delta_t = (t1_dt - t0_dt).total_seconds()
    drift_rate = (freq_end - freq_start) / delta_t / 2  # MHz/s

    ax.plot(
        [mdates.date2num(t0_dt), mdates.date2num(t1_dt)],
        [freq_start, freq_end],
        fmt, color=color, linewidth=lw, alpha=0.7
    )
    ax.plot(
        mdates.date2num(t0_dt), freq_start,
        'o', color=color, markersize=8, alpha=0.7
    )
    ax.plot(
        mdates.date2num(t1_dt), freq_end,
        's', color=color, markersize=8, alpha=0.7
    )

    return drift_rate