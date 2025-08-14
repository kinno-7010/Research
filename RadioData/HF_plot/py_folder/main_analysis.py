#!/usr/bin/env python3

import numpy as np
import cdflib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, MaxNLocator, FormatStrFormatter, FuncFormatter, LogLocator, FixedLocator
from matplotlib.dates import SecondLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import median_filter
from scipy.stats import zscore
from scipy.optimize import fsolve
from matplotlib.colors import Normalize
import datetime as dt
import matplotlib.ticker as mticker
from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar

# Import our custom modules
from utils import _to_datetime, _to_seconds, _slice_data
from spectrum_plot import plot_dynamic_spectrum, plot_removed_dynamic_spectrum, plot_drift_line
from peak_analysis import calculate_dynamic_spectrum_with_peak, calculate_peak_time_and_freq, plot_removed_dynamic_spectrum_with_peak
from drift_analysis import time_weighted_average, compute_drift_rates_and_durations, plot_segments
from frequency_conversion import density_from_frequency, frequency_from_density, density_from_frequency_harmonic, frequency_from_density_harmonic
from statistics import compute_drift_stats, calculate_fit_with_error, plot_fit_with_error
from solar_models import Saito1970, Saito1977, find_rho_for_value

def load_hf_data(file_path):
    """
    Load HF radio data from CDF file
    
    Parameters:
    file_path: str - Path to the CDF file
    
    Returns:
    time: np.ndarray - Array of datetime objects
    frequency_mhz: np.ndarray - Frequency array in MHz
    data: np.ndarray - Power flux density data in dB
    """
    # Load CDF file
    cdf_file = cdflib.CDF(file_path)
    
    # Extract variables
    epoch = cdf_file.varget('Epoch')        # 時間 (ms)
    frequency = cdf_file.varget('Frequency')# 周波数 (Hz)
    data = cdf_file.varget('RH')            # パワーフラックス密度 (dB)
    
    # Convert frequency to MHz
    frequency_mhz = frequency / 1e6
    
    # Physical constants
    _e = 1.60217662e-19    # C
    _m_e = 9.10938356e-31   # kg
    _eps0 = 8.854187817e-12  # F/m
    
    # Create time array
    base_time = dt.datetime(2022, 6, 13, 0, 0, 0)
    
    # 時間配列の作成: Epoch の差分をミリ秒とみなし datetime に変換
    offset_ms = epoch - epoch[0]
    time = np.array([
        base_time + dt.timedelta(milliseconds=float(ms))
        for ms in offset_ms
    ])
    
    cdf_file.close()
    
    return time, frequency_mhz, data

def main():
    """
    Main analysis function
    """
    # Set data file path
    file_path = "/mnt/d/wsl/home/kinno-7010/Research/RadioData/HF_plot/Rawdata/it_h1_hf_20220613_v01.cdf"
    
    # Load data
    print("Loading HF radio data...")
    time, frequency_mhz, data = load_hf_data(file_path)
    
    print(f"Data loaded successfully!")
    print(f"Time range: {time[0]} to {time[-1]}")
    print(f"Frequency range: {frequency_mhz[0]:.2f} to {frequency_mhz[-1]:.2f} MHz")
    print(f"Data shape: {data.shape}")
    
    # Example analysis parameters
    start_time = "2022-06-13T03:00:00"
    end_time = "2022-06-13T04:00:00"
    freq_min = 20.0
    freq_max = 80.0
    med_filter_size = (3, 3)
    vmin = 40
    vmax = 80
    outlier_z = 2.0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot dynamic spectrum with peaks
    calculate_dynamic_spectrum_with_peak(
        fig, ax, time, frequency_mhz, data,
        start_time, end_time,
        freq_min, freq_max,
        120, 10,  # time_tick_sec, freq_tick_mhz
        med_filter_size,
        vmin, vmax,
        "HF Radio Dynamic Spectrum with Peaks", 'red',
        outlier_z
    )
    
    ax.set_xlabel('Time (UTC)', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("Analysis completed!")

if __name__ == "__main__":
    main()