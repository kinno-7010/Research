import numpy as np
import datetime as dt
import matplotlib.dates as mdates
import cdflib

def _to_datetime(dt_input):
    """
    Convert a datetime or ISO-format string to a datetime object.
    """
    return dt.datetime.fromisoformat(dt_input) if isinstance(dt_input, str) else dt_input

def _to_seconds(times):
    """
    matplotlib の日付数 → 経過秒数配列に変換
    """
    xnum = mdates.date2num(times)
    return (xnum - xnum[0]) * 86400.0, xnum

def _slice_data(time_array, data, start_dt, end_dt, freq_mhz, freq_min, freq_max):
    """
    Slice time and frequency ranges from a 2D data array.
    Returns: t_sel, f_sel, d_sel
    """
    # Time slice
    mask_t = (time_array >= start_dt) & (time_array <= end_dt)
    t_sel = time_array[mask_t]
    d_sel = data[mask_t, :]

    # Frequency slice
    mask_f = (freq_mhz >= freq_min) & (freq_mhz <= freq_max)
    f_sel = freq_mhz[mask_f]
    d_sel = d_sel[:, mask_f]
    return t_sel, f_sel, d_sel

def _initialize_data_parameters():
        # CDFファイルのパス
    file_path = "/mnt/d/wsl/home/kinno-7010/Research/RadioData/HF_plot/Rawdata/it_h1_hf_20220613_v01.cdf"

    # CDFファイルを読み込み
    cdf_file = cdflib.CDF(file_path)

    # データの読み込み
    epoch = cdf_file.varget('Epoch')        # 時間 (ms)
    frequency = cdf_file.varget('Frequency')# 周波数 (Hz)
    data = cdf_file.varget('RH')         # パワーフラックス密度 (dB)

    # 周波数を MHz に変換
    frequency_mhz = frequency / 1e6

    # 基準時間の定義 (UT)
    base_time = dt.datetime(2022, 6, 13, 0, 0, 0)

    # 時間配列の作成: Epoch の差分をミリ秒とみなし datetime に変換
    offset_ms = epoch - epoch[0]
    time = np.array([
        base_time + dt.timedelta(milliseconds=float(ms))
        for ms in offset_ms
    ])
    return file_path, time, frequency_mhz, data

def _initialize_plot_parameters(time_tick_sec=None, freq_tick_mhz=None, med_filter_size=None, vmin=None, vmax=None, title=None):
    
    start_time = input("Enter the start time: [default: 2022-06-13T03:25:00]")
    if start_time == '':
        start_time = "2022-06-13T03:25:00"
    end_time = input("Enter the end time: [default: 2022-06-13T03:34:00]")
    if end_time == '':
        end_time = "2022-06-13T03:34:00"
    freq_min = input("Enter the frequency minimum: [default: 25 MHz]")
    if freq_min == '':
        freq_min = 25
    freq_max = input("Enter the frequency maximum: [default: 38 MHz]")
    if freq_max == '':
        freq_max = 38
    
    if time_tick_sec is None:
        time_tick_sec = 60
    if freq_tick_mhz is None:
        freq_tick_mhz = 5
    if med_filter_size is None:
        med_filter_size = (1, 1)
    if vmin is None:
        vmin = 80
    if vmax is None:
        vmax = 95
    if title is None:
        title = 'HF Radio Dynamic Spectrum'
    
    return start_time, end_time, freq_min, freq_max, time_tick_sec, freq_tick_mhz, med_filter_size, vmin, vmax, title