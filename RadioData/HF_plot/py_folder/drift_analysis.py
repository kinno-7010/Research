import numpy as np
import matplotlib.dates as mdates

try:
    from .utils import _to_datetime
    from .spectrum_plot import plot_drift_line, time, frequency_mhz, data
except ImportError:
    from utils import _to_datetime
    from spectrum_plot import plot_drift_line, time, frequency_mhz, data

def time_weighted_average(segments):
    """
    Compute weighted average of rates over segments.
    """
    total_time = 0.0
    weighted_sum = 0.0
    for t0, t1, rate in segments:
        t0_dt = _to_datetime(t0)
        t1_dt = _to_datetime(t1)
        dt_sec = (t1_dt - t0_dt).total_seconds()
        weighted_sum += rate * dt_sec
        total_time += dt_sec
    return weighted_sum / total_time


def compute_drift_rates_and_durations(segments, divide_by_two=True):
    """
    Calculate drift rates and durations for given segments.
    """
    rates, dts = [], []
    for t0, t1, f0, f1 in segments:
        t0_dt = _to_datetime(t0)
        t1_dt = _to_datetime(t1)
        dt_sec = (t1_dt - t0_dt).total_seconds()
        rate = (f1 - f0) / dt_sec
        if divide_by_two:
            rate /= 2
        rates.append(rate)
        dts.append(dt_sec)
    return rates, dts


def plot_segments(ax, segments, color, label=None, divide_by_two=True):
    """
    Plot multiple drift segments and compute average rate.
    """
    rates, dts = compute_drift_rates_and_durations(segments, divide_by_two)
    for (t0, t1, f0, f1), rate in zip(segments, rates):
        plot_drift_line(ax, t0, t1, f0, f1, fmt='-', color=color, lw=2)
        print(f'Drift {f0}→{f1} MHz: {rate:.3e} MHz/s')

    if label and segments:
        last_t0, last_t1, _, last_f1 = segments[-1]
        t0_dt = _to_datetime(last_t0)
        t1_dt = _to_datetime(last_t1)
        ax.plot(
            [mdates.date2num(t0_dt), mdates.date2num(t1_dt)],
            [last_f1, last_f1],
            '--', color=color, linewidth=1, label=label
        )
    time_avg = np.average(rates, weights=dts)
    return rates, dts, time_avg