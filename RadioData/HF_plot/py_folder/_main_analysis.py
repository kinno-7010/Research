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
try:
    from .utils import _to_datetime, _to_seconds, _slice_data
    from .spectrum_plot import plot_dynamic_spectrum, plot_removed_dynamic_spectrum, plot_drift_line, load_hf_data, time, frequency_mhz, data
    from .peak_analysis import calculate_dynamic_spectrum_with_peak, calculate_peak_time_and_freq, plot_removed_dynamic_spectrum_with_peak
    from .drift_analysis import time_weighted_average, compute_drift_rates_and_durations, plot_segments
    from .frequency_conversion import density_from_frequency, frequency_from_density, density_from_frequency_harmonic, frequency_from_density_harmonic
    from .statistics import compute_drift_stats, calculate_fit_with_error, plot_fit_with_error
    from .solar_models import Saito1970, Saito1977, find_rho_for_value
except ImportError:
    from utils import _to_datetime, _to_seconds, _slice_data
    from spectrum_plot import plot_dynamic_spectrum, plot_removed_dynamic_spectrum, plot_drift_line, load_hf_data, time, frequency_mhz, data
    from peak_analysis import calculate_dynamic_spectrum_with_peak, calculate_peak_time_and_freq, plot_removed_dynamic_spectrum_with_peak
    from drift_analysis import time_weighted_average, compute_drift_rates_and_durations, plot_segments
    from frequency_conversion import density_from_frequency, frequency_from_density, density_from_frequency_harmonic, frequency_from_density_harmonic
    from statistics import compute_drift_stats, calculate_fit_with_error, plot_fit_with_error
    from solar_models import Saito1970, Saito1977, find_rho_for_value

# load_hf_data function is now imported from spectrum_plot.py

def main():
    """
    Main analysis function
    """
    # Load data using global variables
    print("Loading HF radio data...")
    if not load_hf_data():
        print("Failed to load data. Exiting...")
        return
    
    print(f"Data loaded successfully!")
    print(f"Time range: {time[0]} to {time[-1]}")
    print(f"Frequency range: {frequency_mhz[0]:.2f} to {frequency_mhz[-1]:.2f} MHz")
    print(f"Data shape: {data.shape}")
    
    # Example analysis parameters
    start_time = "2022-06-13T03:25:00"
    end_time = "2022-06-13T03:34:00"
    freq_min = 25.0
    freq_max = 38.0
    med_filter_size = (1, 1)
    vmin = 80
    vmax = 95
    outlier_z = 3.0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot dynamic spectrum with peaks
    calculate_dynamic_spectrum_with_peak(
        fig, ax,
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