import numpy as np
import matplotlib.dates as mdates
from scipy.stats import zscore, linregress

try:
    from .spectrum_plot import time, frequency_mhz, data
except ImportError:
    from spectrum_plot import time, frequency_mhz, data

def compute_drift_stats(times, freqs, outlier_z):
    """
    times: np.ndarray of datetime
    freqs: np.ndarray of float (MHz)
    Returns:
      avg_drift: float or None
      std_drift: float or None
      drift_array: np.ndarray or None
    """
    # 時系列点が 2 点未満なら計算不可
    if len(freqs) < 2:
        return None, None, None

    dt_list = []
    df_list = []
    for i in range(len(times) - 1):
        dt_sec = (times[i+1] - times[i]).total_seconds()
        if dt_sec > 0:
            dt_list.append(dt_sec)
            df_list.append(freqs[i+1] - freqs[i])

    if not dt_list:
        return None, None, None

    dt_arr = np.array(dt_list)
    df_arr = np.array(df_list)
    drift_arr = df_arr / dt_arr

    # z-score外れ値除去
    if len(drift_arr) >= 2 and outlier_z is not None:
        mask = np.abs(zscore(drift_arr)) <= outlier_z
        drift_arr = drift_arr[mask]

    if len(drift_arr) == 0:
        return None, None, None

    avg_drift = np.mean(drift_arr)
    std_drift = np.std(drift_arr) if len(drift_arr) > 1 else 0.0

    return avg_drift, std_drift, drift_arr


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

    # フィット直線
    fit = slope * t_sec + intercept

    return slope, intercept, stderr


def plot_fit_with_error(ax, times, freqs, color, label):
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

    # フィット直線
    fit = slope * t_sec + intercept
    
    ax.plot(
        xnum, fit,
        linestyle='--', linewidth=3,
        color=color,
        label=f"{label} fit (2nd harmonic): {slope/2:.3e}±{stderr/2:.3e} MHz/s"
    )

    return slope, stderr