import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from matplotlib.ticker import MultipleLocator, LogLocator, FuncFormatter
from matplotlib.dates import SecondLocator
from scipy.ndimage import median_filter
from scipy.stats import zscore
from peak_analysis import plot_removed_dynamic_spectrum_with_peak, plot_removed_dynamic_spectrum_with_peak_2, calculate_peak_time_and_freq, calculate_fit_with_error
from frequency_conversion import density_from_frequency, frequency_from_density
from utils import _initialize_data_parameters

sys.path.append("/mnt/d/wsl/home/kinno-7010/Research/RadioData/combine")
from predict_type2_const_speed import f_model_from_r, invert_r_from_f


RS_KM = 6.957e5  # solar radius in km
R_SCALE = 2.0


def _freq_to_r(f_mhz: np.ndarray | float, branch: str = "F", factor: float = 1.0) -> np.ndarray | float:
    f_arr = np.asarray(f_mhz, dtype=float)
    vec = np.vectorize(lambda v: invert_r_from_f(float(v), branch=branch, factor=factor))
    return vec(f_arr)

def _r_to_freq(r_rs: np.ndarray | float, branch: str = "F", factor: float = 1.0) -> np.ndarray | float:
    r_arr = np.asarray(r_rs, dtype=float)
    vec = np.vectorize(lambda v: f_model_from_r(float(v), branch=branch, factor=factor))
    return vec(r_arr)

file_path, time, frequency_mhz, data = _initialize_data_parameters()

def plot_removed_dynamic_spectrum_with_distance_axis(fig, ax, time_array: np.ndarray, freq_mhz: np.ndarray, data: np.ndarray, start_time: str, end_time: str, freq_min: float, freq_max: float, time_tick_sec: int, freq_tick_mhz: float, med_filter_size: tuple[int, int], vmin: float, vmax: float, title: str, model_factor: float = 1.0):
    """
    Remove 3σ outliers in 35–40 MHz band and plot dynamic spectrum.
    Left y-axis: Frequency [MHz] (log)
    Right y-axis: Density [cm^-3] (log), converted from Frequency
    """
    # --- outlier mask ---
    mask_band = (freq_mhz >= 35) & (freq_mhz <= 40)
    flat = data[:, mask_band].ravel()
    mu, sigma = flat.mean(), flat.std()
    clean = flat[(flat>mu-3*sigma)&(flat<mu+3*sigma)]
    thresh = clean.mean() * 1.05
    masked = np.where(data > thresh, data, np.nan)

    # --- slice in time/freq ---
    t0 = dt.datetime.fromisoformat(start_time)
    t1 = dt.datetime.fromisoformat(end_time)
    mask_t  = (time_array >= t0) & (time_array <= t1)
    t_sel   = time_array[mask_t]
    d_t     = masked[mask_t,:]
    mask_f  = (freq_mhz >= freq_min) & (freq_mhz <= freq_max)
    f_sel   = freq_mhz[mask_f]
    d_sel   = d_t[:, mask_f]

    # --- noise reduction ---
    d_filt = median_filter(d_sel.astype(float), size=med_filter_size)

    # --- main image ---
    extent = [
        mdates.date2num(t_sel[0]), mdates.date2num(t_sel[-1]),
        f_sel[0], f_sel[-1]
    ]
    ax.imshow(
        d_filt.T, origin='lower', aspect='auto',
        extent=extent, cmap='viridis', vmin=vmin, vmax=vmax
    )

    # --- 左軸: Frequency ---
    ax.set_title(title, fontsize=18)
    ax.set_ylabel('Frequency (MHz)', fontsize=16)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.xaxis.set_major_locator(SecondLocator(interval=time_tick_sec))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.yaxis.set_major_locator(MultipleLocator(freq_tick_mhz))
    ax.tick_params(axis='both', which='major', labelsize=14)

    # --- 右軸: Radial distance (2nd harmonic corrected) ---
    secax = ax.secondary_yaxis(
        "right",
        functions=(
            lambda f_mhz: _freq_to_r(f_mhz, branch="H", factor=model_factor),
            lambda r_rs: _r_to_freq(r_rs, branch="H", factor=model_factor),
        ),
    )
    secax.set_ylabel(
        f"Radial distance (Harmonic) [R$_\\odot$] ({model_factor}× $160^\\circ$ line)",
        fontsize=14,
    )
    secax.tick_params(axis="y", labelsize=12)
    secax.yaxis.set_major_locator(MultipleLocator(0.1))
    secax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.1f}"))


def plot_peak_dynamic_spectrum(start_time: str, end_time: str, freq_min: float, freq_max: float, time_tick_sec: int, freq_tick_mhz: float, med_filter_size: tuple[int, int], vmin: float, vmax: float, title: str, model_factor: float = 1.0):
    
    outlier_z = 3.0
    fig, ax = plt.subplots(figsize=(12, 6))

    # 1) リスト初期化
    times_red, freqs_red = [], []
    times_blue, freqs_blue = [], []

    for i in range(len(start_time)):
        # ダイナミックスペクトルのプロット部分（省略可）
        if i == 0:
            plot_removed_dynamic_spectrum_with_distance_axis(
                fig, ax, time_array=time, freq_mhz=frequency_mhz, data=data,
                start_time=start_time[i], end_time=end_time[i],
                freq_min=freq_min[i], freq_max=freq_max[i],
                time_tick_sec=time_tick_sec, freq_tick_mhz=freq_tick_mhz,
                med_filter_size=med_filter_size,
                vmin=vmin, vmax=vmax,
                title=title,
                model_factor=model_factor
            )
            continue

        # --- 赤点区間 ---
        if i in (1, 2):
            masked_data = plot_removed_dynamic_spectrum_with_peak(
                fig, ax,
                start_time=start_time[i], end_time=end_time[i],
                freq_min=freq_min[i], freq_max=freq_max[i],
                time_tick_sec=time_tick_sec, freq_tick_mhz=freq_tick_mhz,
                med_filter_size=med_filter_size,
                vmin=vmin, vmax=vmax,
                title=title, scatter_color='red',
                outlier_z=outlier_z
            )
            # ← ここで時刻・周波数を抽出
            t, f = calculate_peak_time_and_freq(
                time, frequency_mhz, masked_data,
                start_time[i], end_time[i],
                freq_min[i], freq_max[i],
                med_filter_size, outlier_z
            )
            times_red .append(t)
            freqs_red .append(f)

        # --- 青点区間 ---
        else:
            # 最後から2番目だけ別関数でもいいですが同様に取得できます
            if i == len(start_time) - 2:
                masked_data = plot_removed_dynamic_spectrum_with_peak(
                    fig, ax,
                    start_time=start_time[i], end_time=end_time[i],
                    freq_min=freq_min[i], freq_max=freq_max[i],
                    time_tick_sec=time_tick_sec, freq_tick_mhz=freq_tick_mhz,
                    med_filter_size=med_filter_size,
                    vmin=vmin, vmax=vmax,
                    title=title, scatter_color='blue',
                    outlier_z=outlier_z
                )
                # ← ここでも抽出
                t, f = calculate_peak_time_and_freq(
                    time, frequency_mhz, masked_data,
                    start_time[i], end_time[i],
                    freq_min[i], freq_max[i],
                    med_filter_size, outlier_z
                )
                times_blue.append(t)
                freqs_blue.append(f)
            else:
                masked_data_2 = plot_removed_dynamic_spectrum_with_peak_2(
                    fig, ax,
                    start_time=start_time[i], end_time=end_time[i],
                    freq_min=freq_min[i], freq_max=freq_max[i],
                    time_tick_sec=time_tick_sec, freq_tick_mhz=freq_tick_mhz,
                    med_filter_size=med_filter_size,
                    vmin=vmin, vmax=vmax,
                    title=title, scatter_color='blue',
                    threshold_high=91, outlier_z=outlier_z
                )
                # ← ここでも抽出
                t, f = calculate_peak_time_and_freq(
                    time, frequency_mhz, masked_data_2,
                    start_time[i], end_time[i],
                    freq_min[i], freq_max[i],
                    med_filter_size, outlier_z
                )
                times_blue.append(t)
                freqs_blue.append(f)
            

    # 2) フラット化
    if times_red:
        times_red  = np.concatenate(times_red)
        freqs_red  = np.concatenate(freqs_red)
    else:
        times_red = np.array([]); freqs_red = np.array([])

    if times_blue:
        times_blue = np.concatenate(times_blue)
        freqs_blue = np.concatenate(freqs_blue)
    else:
        times_blue = np.array([]); freqs_blue = np.array([])

    # 点が取得できなかった場合は後続のインデックス参照を避けて終了
    if len(times_red) == 0 or len(times_blue) == 0:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(fontsize=12)
        plt.show()
        return

    #--------------
    xnum_red, xnum_blue = mdates.date2num(times_red), mdates.date2num(times_blue)
    xnum_total = np.concatenate([xnum_red, xnum_blue])

    t_sec_total, t_sec_red, t_sec_blue = (xnum_total - xnum_total[0]) * 86400.0, (xnum_red - xnum_red[0]) * 86400.0, (xnum_blue - xnum_blue[0]) * 86400.0

    # red lineのフィット関数を求める
    slope_red, intercept_red, stderr_red = calculate_fit_with_error(ax, times_red, freqs_red)
    slope_blue, intercept_blue, stderr_blue = calculate_fit_with_error(ax, times_blue, freqs_blue)

    # フィット直線
    fit_red, fit_blue = slope_red * t_sec_red + intercept_red, slope_blue * t_sec_blue + intercept_blue

    # red line fit の終わりの密度
    dens_red_start, dens_red_end = density_from_frequency(fit_red[0]/2), density_from_frequency(fit_red[-1]/2)
    freq_dens_red_start, freq_dens_red_end = frequency_from_density(dens_red_start)*2, frequency_from_density(dens_red_end)*2
    dens_blue_start, dens_blue_end = density_from_frequency(fit_blue[0]/2), density_from_frequency(fit_blue[-1]/2)
    freq_dens_blue_start, freq_dens_blue_end = frequency_from_density(dens_blue_start)*2, frequency_from_density(dens_blue_end)*2
    
    # t_sec_red[-1]のFrequencyを計算
    freq_red_mid = fit_red[np.argmin(np.abs(t_sec_total - t_sec_red[-1]))]
    dens_red_mid = density_from_frequency(freq_red_mid/2)

    # 3) フィット
    if len(times_red) >= 2:
        ax.plot(
            xnum_red, fit_red,
            linestyle='--', linewidth=3,
            color='#ff00ff', zorder=12,
            label=f"Red FDR (2nd harmonic): {slope_red/2:.3e}±{stderr_red/2:.3e} MHz/s"
        )
# ax.vlines(xnum_blue[-1], freq_min[0], freq_max[0], color='black', linestyle='--', linewidth=3)
        ax.scatter(xnum_red[0], freq_dens_red_start, color='#ff00ff', marker='x', s=100, zorder=12, label=f'{freq_dens_red_start:.2f}[MHz] @ {_freq_to_r(freq_dens_red_start, branch="H", factor=model_factor):.3f}[$R_\\odot$] $\\rightarrow$ {freq_red_mid:.3f}[MHz] @ {_freq_to_r(freq_red_mid, branch="H", factor=model_factor):.3f}[$R_\\odot$]') # $\\rightarrow$ {freq_dens_red_end:.3f}[MHz] @ {_freq_to_r(freq_dens_red_end, branch="H"):.3f}[$R_\\odot$]')
        ax.scatter(xnum_red[np.argmin(np.abs(t_sec_total - t_sec_red[-1]))], freq_red_mid, color='#ff00ff', marker='x', s=100, zorder=12)
        # ax.scatter(xnum_blue[-1], freq_dens_red_end, color='#ff00ff', marker='x', s=100, zorder=12)
        
    if len(times_blue) >= 2:

        
        # print(f'dens_red: {dens_red:.4e}[/cc]')
        
        ax.plot(
            xnum_blue, fit_blue,
            linestyle='--', linewidth=3,
            color='#00ffff', zorder=12,
            label=f"Blue FDR (2nd harmonic): {slope_blue/2:.3e}±{stderr_blue/2:.3e} MHz/s"
        )
        
        ax.scatter(xnum_blue[-1], freq_dens_blue_end, color='#00ffff', marker='x', s=100, zorder=12)
        ax.scatter(xnum_blue[0], freq_dens_blue_start, color='#00ffff', marker='x', s=100, zorder=12, label=f'{freq_dens_blue_start:.2f}[MHz] @ {_freq_to_r(freq_dens_blue_start, branch="H", factor=model_factor):.3f}[$R_\\odot$] $\\rightarrow$ {freq_dens_blue_end:.2f}[MHz] @ {_freq_to_r(freq_dens_blue_end, branch="H", factor=model_factor):.3f}[$R_\\odot$]')


    # 4) 縦線
    t_line = dt.datetime.fromisoformat('2022-06-13T03:28:45')
    xnum   = mdates.date2num(t_line)
    ax.axvline(x=xnum, color='black', linestyle='--', linewidth=2)
    ax.text(xnum, freq_max[0], '03:28:45', color='black',
            fontsize=16, ha='left', va='top')

    ax.axvline(xnum_red[0], color='red', linestyle='--', linewidth=2)
    ax.text(xnum_red[0], freq_max[0], f'{times_red[0].strftime('%H:%M:%S')}',
            color='red', va='top', ha='left', fontsize=16)
    ax.axvline(xnum_blue[-1], color='blue', linestyle='--', linewidth=2)
    ax.text(xnum_blue[-1], freq_max[0], f'{times_blue[-1].strftime('%H:%M:%S')}',
            color='blue', va='top', ha='left', fontsize=16)
    # ax.axhline(y=29.75, color='black', linestyle='--', linewidth=2)
    
    # 5) upper lane/lower lane
    upper_lane = [("2022-06-13T03:25:40", 35.7), ("2022-06-13T03:25:50", 35.9), ("2022-06-13T03:26:00", 36.1), ("2022-06-13T03:26:10", 36.2),
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
                  
                  ("2022-06-13T03:32:50", 29.0), ("2022-06-13T03:33:00", 28.9)
                  ]
    
    lower_lane = [("2022-06-13T03:25:40", 33.2), ("2022-06-13T03:25:50", 32.8), ("2022-06-13T03:26:00", 32.4), ("2022-06-13T03:26:10", 31.9),
                  
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
                  ("2022-06-13T03:32:10", 25.5), ("2022-06-13T03:32:20", 25.4), ("2022-06-13T03:32:30", 25.3), ("2022-06-13T03:32:40", 25.2)
                  ]

    # 文字列→datetime→mdates数値に変換してから描画（直接文字列を渡さない）
    def _to_mdates_xy(lane):
        t = [dt.datetime.fromisoformat(ts) for ts, _ in lane]
        y = [v for _, v in lane]
        return mdates.date2num(t), y

    upper_x, upper_y = _to_mdates_xy(upper_lane)
    lower_x, lower_y = _to_mdates_xy(lower_lane)

    # ドリフトレート（MHz/s）の平均・標準偏差を算出
    def _drift_rate_stats(x_mdates, y_mhz):
        if len(x_mdates) < 2:
            return None, None
        dt_sec = np.diff(x_mdates) * 86400.0
        df_mhz = np.diff(y_mhz)
        rates = df_mhz / dt_sec
        return float(np.mean(rates)), float(np.std(rates))

    upper_mean, upper_std = _drift_rate_stats(upper_x, upper_y)
    lower_mean, lower_std = _drift_rate_stats(lower_x, lower_y)

    # プロットは線のみ（scatterはコメントアウトのまま）
    ax.scatter(upper_x, upper_y, color='orange', marker='+', s=50, zorder=12)
    ax.scatter(lower_x, lower_y, color='purple', marker='+', s=50, zorder=12)
    upper_label = 'Upper Lane'
    lower_label = 'Lower Lane'
    # この周波数は 2nd harmonic のため、Fundamental 換算として 1/2 を用いる
    if upper_mean is not None:
        upper_label += f" (FDR={upper_mean/2:.3e}±{upper_std/2:.3e} MHz/s)"
    if lower_mean is not None:
        lower_label += f" (FDR={lower_mean/2:.3e}±{lower_std/2:.3e} MHz/s)"
    ax.plot(upper_x, upper_y, color='orange', linestyle='--', linewidth=2, label=upper_label)
    ax.plot(lower_x, lower_y, color='purple', linestyle='--', linewidth=2, label=lower_label)

    ax.legend(fontsize=12)
    output_path = '/mnt/d/wsl/home/kinno-7010/Research/RadioData/HF_plot/output/peak_dynamic_spectrum_upper_lower_160degree.png'
    plt.savefig(output_path)
    print(f'Saved: {output_path}')
    plt.show()
    
    
if __name__ == "__main__":
    plot_peak_dynamic_spectrum(
        start_time=[
            "2022-06-13T03:25:00",
            "2022-06-13T03:25:30",
            "2022-06-13T03:26:30",
            "2022-06-13T03:28:45",
            "2022-06-13T03:29:30",
            "2022-06-13T03:30:10",
            "2022-06-13T03:30:40"
        ],
        end_time=[
            "2022-06-13T03:33:00",
            "2022-06-13T03:26:30",
            "2022-06-13T03:28:45",
            "2022-06-13T03:29:30",
            "2022-06-13T03:30:10",
            "2022-06-13T03:30:40",
            "2022-06-13T03:31:20"
        ],
        freq_min=[24, 32, 30, 29.64, 28.2, 27.5, 26],
        freq_max=[38, 37, 37, 34, 34, 34, 33],
        time_tick_sec=60,
        freq_tick_mhz=1,
        med_filter_size=(1, 1),
        vmin=80,
        vmax=95,
        title="Second Harmonic Lane Analysis",
        model_factor=1
    )