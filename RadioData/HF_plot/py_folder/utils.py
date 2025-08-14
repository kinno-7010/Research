import numpy as np
import datetime as dt
import matplotlib.dates as mdates

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