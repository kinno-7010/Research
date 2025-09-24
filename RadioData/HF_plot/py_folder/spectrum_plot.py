import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, LogLocator, FuncFormatter
from matplotlib.dates import SecondLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import median_filter
import cdflib
import datetime as dt
from astropy.time import Time

try:
    from .utils import _to_datetime, _slice_data, _initialize_plot_parameters, _initialize_data_parameters
except ImportError:
    from utils import _to_datetime, _slice_data, _initialize_plot_parameters, _initialize_data_parameters

file_path, time, frequency_mhz, data = _initialize_data_parameters()

def plot_dynamic_spectrum(
    fig, ax,
    start_time, end_time,
    freq_min, freq_max,
    time_tick_sec, freq_tick_mhz,
    med_filter_size,
    vmin, vmax,
    title,
    background_method
):
    """
    Plot a dynamic spectrum between start_time and end_time.
    """
    start_dt = _to_datetime(start_time)
    end_dt = _to_datetime(end_time)
    t_sel, f_sel, d_sel = _slice_data(
        time, data, start_dt, end_dt,
        frequency_mhz, freq_min, freq_max
    )

    # Background removal processing
    if background_method == 'median_time':
        print("  - 各周波数での時間中央値を減算")
        background = np.median(d_sel, axis=0, keepdims=True)
        d_sel = d_sel - background
    else:
        print("  - 背景除去処理をスキップします")

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
    fig, ax,
    start_time, end_time,
    freq_min, freq_max,
    time_tick_sec, freq_tick_mhz,
    med_filter_size, vmin, vmax,
    background_method
):
    """
    Remove 3σ outliers in 35–40 MHz band and plot the cleaned dynamic spectrum.
    """
    global data
    
    # Apply background removal if median_time is selected
    if background_method == 'median_time':
        print("  - median_time処理後のデータからclean_meanを計算")
        # Apply median_time background removal to get processed data
        background_removed_data = data - np.median(data, axis=0, keepdims=True)
        # 35–40 MHz band extraction from processed data
        mask_band = (frequency_mhz >= 35) & (frequency_mhz <= 40)
        band_data = background_removed_data[:, mask_band]
        
        flat = band_data.flatten()
        mu, sigma = np.mean(flat), np.std(flat)
        lower, upper = mu - 3*sigma, mu + 3*sigma
        clean = flat[(flat > lower) & (flat < upper)]
        clean_mean = np.mean(clean)
        
        # Mask below threshold using background_removed_data
        threshold = 2
        masked = np.where(background_removed_data >= threshold, background_removed_data, np.nan)
        
        # Temporarily save original data and use masked data
        original_data = data
        data = masked
        
        # Delegate to plot_dynamic_spectrum
        plot_dynamic_spectrum(
            fig, ax,
            start_time, end_time,
            freq_min, freq_max,
            time_tick_sec, freq_tick_mhz,
            med_filter_size,
            vmin, vmax,
            title='Second Harmonic Dynamic Spectrum (removed) (median_time)',
            background_method='none'
        )
    else:
        print("  - 元データからclean_meanを計算")
        # 35–40 MHz band extraction from original data
        mask_band = (frequency_mhz >= 35) & (frequency_mhz <= 40)
        band_data = data[:, mask_band]
        
        flat = band_data.flatten()
        mu, sigma = np.mean(flat), np.std(flat)
        lower, upper = mu - 3*sigma, mu + 3*sigma
        clean = flat[(flat > lower) & (flat < upper)]
        clean_mean = np.mean(clean)
        
        # Mask below threshold using original data
        threshold = clean_mean * 1.05
        masked = np.where(data > threshold, data, np.nan)

        # Temporarily save original data and use masked data
        original_data = data
        data = masked
        
        # Delegate to plot_dynamic_spectrum
        plot_dynamic_spectrum(
            fig, ax,
            start_time, end_time,
            freq_min, freq_max,
            time_tick_sec, freq_tick_mhz,
            med_filter_size,
            vmin, vmax,
            title='Second Harmonic Dynamic Spectrum (removed)',
            background_method=background_method
        )
    
    # Restore original data
    data = original_data


# In spectrum_plot.py

def plot_manual_drift_line(
    fig, ax, segment,
    time_tick_sec, freq_tick_mhz,
    med_filter_size, vmin, vmax,
    line_color='red'  # Added parameter for color
):
    """
    Plot a drift line and mark endpoints on the given Axes.
    """
    t0, t1, freq_start, freq_end = segment
    
    t0_dt = _to_datetime(t0)
    t1_dt = _to_datetime(t1)

    delta_t = (t1_dt - t0_dt).total_seconds()
    if delta_t == 0:
        return 0
    drift_rate = (freq_end - freq_start) / delta_t # MHz/s

    # The background spectrum is now plotted only once outside the loop.
    # So we remove this call from here to avoid re-drawing it every time.
    # plot_removed_dynamic_spectrum(...)

    # Draw the line segments on the plot
    ax.plot(
        [mdates.date2num(t0_dt), mdates.date2num(t1_dt)],
        [freq_start, freq_end],
        '--', color=line_color, linewidth=2, alpha=0.7
    )
    ax.plot(
        mdates.date2num(t0_dt), freq_start,
        'o', color=line_color, markersize=8, alpha=0.7
    )
    ax.plot(
        mdates.date2num(t1_dt), freq_end,
        'o', color=line_color, markersize=8, alpha=0.7
    )
    return drift_rate

if __name__ == "__main__":
    import os # <-- 1. Import the 'os' module at the top of this block

    print("=" * 60)
    print("HF Radio Dynamic Spectrum Analysis Tool")
    print("=" * 60)
    print()
    
    # Load data
    if not (file_path, time, frequency_mhz, data):
        print("Failed to load data. Exiting...")
        exit(1)
    
    # --- Define the output directory path ---
    output_dir = '../output' # <-- 2. Define the correct output path
    os.makedirs(output_dir, exist_ok=True) # <-- 3. Create the directory if it doesn't exist
    
    print()
    print("Available analysis options:")
    print("1. plot_dynamic_spectrum: Basic Dynamic Spectrum - Full time range view")
    print("2. plot_removed_dynamic_spectrum: Cleaned Dynamic Spectrum - Background noise removed (3σ method)")
    print("3. plot_drift_line: Dynamic Spectrum with Drift Lines - Type II burst drift analysis")
    print("0. Exit")
    print("-" * 60)
    

    
    while True:
        try:
            choice = input("Enter your choice (0-3): ").strip()
            
            if choice == '1':
                print("\nPlotting basic dynamic spectrum...")
                start_time, end_time, freq_min, freq_max, time_tick_sec, freq_tick_mhz, med_filter_size, vmin, vmax, title = _initialize_plot_parameters()
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                background_method_value = input("Background method (0: none, 1: median_time): ")
                if background_method_value == '1':
                    background_method = 'median_time'
                    vmin = 0
                    vmax = 10
                    # titleにbackground_methodを追記
                    title = f"{title} ({background_method})"
                else:
                    background_method = 'none'
                
                plot_dynamic_spectrum(
                    fig, ax,
                    start_time, end_time,
                    freq_min, freq_max,
                    time_tick_sec, freq_tick_mhz,
                    med_filter_size,
                    vmin, vmax,
                    title,
                    background_method
                )
                ax.set_xlabel('Time (UTC)', fontsize=16)
                plt.tight_layout()
                # 4. Use the new path variable to save the file
                # start_time, end_timeの":"を消去し空白を詰める
                start_time = start_time.replace(":", "")
                end_time = end_time.replace(":", "")
                if background_method == 'median_time':
                    plt.savefig(f'{output_dir}/HF_dynamic_spectrum_{start_time}-{end_time}_{freq_min}-{freq_max}MHz_{background_method}.png')
                else:
                    plt.savefig(f'{output_dir}/HF_dynamic_spectrum_{start_time}-{end_time}_{freq_min}-{freq_max}MHz_none.png')
                print(f'{output_dir}/HF_dynamic_spectrum_{start_time}-{end_time}_{freq_min}-{freq_max}MHz_{background_method}.png')
                plt.show()
                
            elif choice == '2':
                print("\nPlotting removed dynamic spectrum...")
                start_time, end_time, freq_min, freq_max, time_tick_sec, freq_tick_mhz, med_filter_size, vmin, vmax, title = _initialize_plot_parameters()
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                background_method_value = input("Background method (0: none, 1: median_time): ")
                if background_method_value == '1':
                    background_method = 'median_time'
                    vmin = 3
                    vmax = 10
                    # titleにbackground_methodを追記
                    title = f"{title} (removed) ({background_method})"
                else:
                    background_method = 'none'
                
                plot_removed_dynamic_spectrum(
                    fig, ax,
                    start_time, end_time,
                    freq_min, freq_max,
                    time_tick_sec, freq_tick_mhz,
                    med_filter_size, vmin, vmax,
                    background_method
                )
                ax.set_xlabel('Time (UTC)', fontsize=16)
                plt.tight_layout()
                # 4. Use the new path variable here as well
                # start_time, end_timeの":"を消去し空白を詰める
                start_time = start_time.replace(":", "")
                end_time = end_time.replace(":", "")
                if background_method == 'median_time':
                    plt.savefig(f'{output_dir}/removed_HF_dynamic_spectrum_{start_time}-{end_time}_{freq_min}-{freq_max}MHz_{background_method}.png')
                    print(f'{output_dir}/removed_HF_dynamic_spectrum_{start_time}-{end_time}_{freq_min}-{freq_max}MHz_{background_method}.png')
                else:
                    plt.savefig(f'{output_dir}/removed_HF_dynamic_spectrum_{start_time}-{end_time}_{freq_min}-{freq_max}MHz_none.png')
                    print(f'{output_dir}/removed_HF_dynamic_spectrum_{start_time}-{end_time}_{freq_min}-{freq_max}MHz_none.png')
                plt.show()
                
                                
            elif choice == '3':
                print("\nPlotting dynamic spectrum with drift lines...")
                start_time, end_time, freq_min, freq_max, time_tick_sec, freq_tick_mhz, med_filter_size, vmin, vmax, title = _initialize_plot_parameters()
                
                # Note: The original code creates two subplots (ax0, ax1) but only displays ax0
                # when calling plot_manual_drift_line. This might be a logic error.
                # Assuming the goal is to draw on ax0 and then show ax1 below it.
                fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
                ax0, ax1 = axes
                
                background_method_value = input("Background method (0: none, 1: median_time): ")
                if background_method_value == '1':
                    background_method = 'median_time'
                    vmin = 0
                    vmax = 10
                    # titleにbackground_methodを追記
                    title = f"{title} (removed) ({background_method})"
                else:
                    background_method = 'none'
                
                # --------------------- ax[0] ---------------------
                # Plot the base dynamic spectrum just once on ax0
                plot_removed_dynamic_spectrum(
                    fig, ax0,
                    start_time, end_time,
                    freq_min, freq_max,
                    time_tick_sec, freq_tick_mhz,
                    med_filter_size,
                    vmin, vmax,
                    background_method
                )

                # Define segments
                upper_segments = [
                    ("2022-06-13T03:25:40","2022-06-13T03:26:40",35.7,36.5),
                    ("2022-06-13T03:26:40","2022-06-13T03:29:55",36.5,31),
                    ("2022-06-13T03:29:55","2022-06-13T03:31:15",31,30.7),
                    ("2022-06-13T03:31:15","2022-06-13T03:31:35",30.7,29.7),
                    ("2022-06-13T03:31:35","2022-06-13T03:32:40",29.7,29.2),
                ]
                lower_segments = [
                    ("2022-06-13T03:25:40","2022-06-13T03:26:10",33,31.8),
                    ("2022-06-13T03:26:10","2022-06-13T03:28:15",31.8,29.7),
                    ("2022-06-13T03:28:15","2022-06-13T03:28:48",29.7,30.2),
                    ("2022-06-13T03:28:45","2022-06-13T03:29:05",30.2,29.1),
                    ("2022-06-13T03:29:05","2022-06-13T03:31:00",29.1,26.9),
                    ("2022-06-13T03:31:00","2022-06-13T03:31:20",26.9,26),
                ]
                
                upper_drift_rates, lower_drift_rates = [], []
                
                # --- FIX 1: Process upper_segments and lower_segments in separate loops ---
                # Process upper segments
                for segment in upper_segments:
                    rate = plot_manual_drift_line(
                        fig, ax0, segment,
                        time_tick_sec, freq_tick_mhz, med_filter_size, vmin, vmax,
                        line_color='red' # Pass color for drawing
                    )
                    upper_drift_rates.append(rate)

                # Process lower segments
                for segment in lower_segments:
                    rate = plot_manual_drift_line(
                        fig, ax0, segment,
                        time_tick_sec, freq_tick_mhz, med_filter_size, vmin, vmax,
                        line_color='blue' # Pass color for drawing
                    )
                    lower_drift_rates.append(rate)

                ax0.set_xlabel('Time (UTC)', fontsize=16)
                # --- FIX 2: Remove the unnecessary legend call ---
                # ax0.legend()

                # --------------------- ax[1] ---------------------
                # Plot drift rates over time
                for (t0, t1, *_), rate in zip(upper_segments, upper_drift_rates):
                    ax1.hlines(rate, _to_datetime(t0), _to_datetime(t1), color='red', lw=2)
                    
                for (t0, t1, *_), rate in zip(lower_segments, lower_drift_rates):
                    ax1.hlines(rate, _to_datetime(t0), _to_datetime(t1), color='blue', lw=2)
                    
                # Connect segments with dotted lines
                for i in range(len(upper_segments) - 1):
                    t1, r1 = _to_datetime(upper_segments[i][1]), upper_drift_rates[i]
                    t2, r2 = _to_datetime(upper_segments[i+1][0]), upper_drift_rates[i+1]
                    ax1.plot([t1, t2], [r1, r2], linestyle='dotted', color='red', linewidth=1)

                for i in range(len(lower_segments) - 1):
                    t1, r1 = _to_datetime(lower_segments[i][1]), lower_drift_rates[i]
                    t2, r2 = _to_datetime(lower_segments[i+1][0]), lower_drift_rates[i+1]
                    ax1.plot([t1, t2], [r1, r2], linestyle='dotted', color='blue', linewidth=1)

                # Plot average lines
                upper_avg, lower_avg = np.mean(upper_drift_rates), np.mean(lower_drift_rates)
                upper_std, lower_std = np.std(upper_drift_rates), np.std(lower_drift_rates)
                ax1.axhline(upper_avg, color='red', ls='--', lw=2, label=f'Upper avg: {upper_avg:.3e} ± {upper_std:.3e}')
                ax1.axhline(lower_avg, color='blue', ls='--', lw=2, label=f'Lower avg: {lower_avg:.3e} ± {lower_std:.3e}')

                ax1.set_xlabel('Time (UT)', fontsize=16)
                ax1.set_ylabel('Drift Rate (MHz/s)', fontsize=16)
                ax1.tick_params(axis='both', which='major', labelsize=14)
                ax1.legend(loc='upper right', fontsize=14)
                ax1.grid(True)
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))


                plt.tight_layout()
                # start_time, end_timeの":"を消去し空白を詰める
                start_time = start_time.replace(":", "")
                end_time = end_time.replace(":", "")
                if background_method == 'median_time':
                    plt.savefig(f'{output_dir}/manual_drift_line_HF_dynamic_spectrum_{start_time}-{end_time}_{freq_min}-{freq_max}MHz_{background_method}.png')
                else:
                    plt.savefig(f'{output_dir}/manual_drift_line_HF_dynamic_spectrum_{start_time}-{end_time}_{freq_min}-{freq_max}MHz_none.png')
                print(f'{output_dir}/manual_drift_line_HF_dynamic_spectrum_{start_time}-{end_time}_{freq_min}-{freq_max}MHz_{background_method}.png')
                plt.show()
                
                                
            elif choice == '0':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please enter 0-3.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
        print("\n" + "=" * 60)