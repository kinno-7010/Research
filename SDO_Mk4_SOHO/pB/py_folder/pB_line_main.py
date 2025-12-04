# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from io_and_processing import (
    load_and_prepare_instrument_data, combine_corona_data, extract_pB_profile
)
from plotting_utils import (
    plot_combined_image, generate_ne_profile_plot
)
from constants_vdh import (
    invert_ablation, triple_power, density_from_frequency
)

# --- File paths (unchanged) ---
filename_mk4 = r'/mnt/d/wsl/home/kinno-7010/Research/MK4_coronagraph/MK4_coronagraph_KCOR/pB/Rawdata/20220613_025810_kcor_l2.fts'
filename_lasco = r'/mnt/d/wsl/home/kinno-7010/Research/SOHO/pB/C2-PB-20220613_0258.fts'

def main():
        # 1) Load data
    data_mk4, params_mk4 = load_and_prepare_instrument_data(filename_mk4, "Mk4")
    data_lasco, params_lasco = load_and_prepare_instrument_data(filename_lasco, "SOHO/LASCO", is_lasco=True)

    final_image, r_map_lasco, theta_to_plot = None, None, 150.0
    r_ranges = {'mk4_inner': 1.1, 'mk4_outer_lasco_inner': 2.2, 'lasco_outer': 7.0}

    # 2) Build grid & merge
    if params_lasco and data_lasco is not None :
        _y, _x = np.indices((params_lasco['ny'], params_lasco['nx']))
        r_map_lasco = np.hypot((_x - params_lasco['cx']) / params_lasco['px_per_rsun'],
                                (_y - params_lasco['cy']) / params_lasco['px_per_rsun'])
        if params_mk4 and data_mk4 is not None:
            final_image = combine_corona_data(data_lasco, params_lasco, data_mk4, params_mk4, r_map_lasco, r_ranges)
        else:
            print("Warning: MK4 data/params failed to load. Using LASCO data only for 'final_image'.")
            final_image = data_lasco.copy()
            final_image[r_map_lasco > r_ranges['lasco_outer']] = np.nan
            lasco_effective_inner = r_ranges.get('mk4_outer_lasco_inner', 2.0)
            final_image[r_map_lasco < lasco_effective_inner ] = np.nan

        if final_image is not None:
            plot_combined_image(final_image, r_map_lasco, params_lasco, r_ranges, theta_deg_overlay=theta_to_plot, xlim_pix=(-150, 0), ylim_pix=(-100, 150))
    else:
        print("Critical: LASCO data/params failed to load. Cannot proceed with image processing.")
        raise SystemExit

    # 3) Extract pB profile and invert
    bin_width = 0.01
    current_plot_r_min = r_ranges.get('mk4_inner',1.1)
    current_plot_r_max = r_ranges.get('lasco_outer',7.0)
    edges = np.arange(current_plot_r_min, current_plot_r_max + bin_width, bin_width)
    r_mid = (edges[:-1] + edges[1:]) / 2 if len(edges) > 1 else np.array([])
    n_bins = len(r_mid)

    pB_line = np.full_like(r_mid, np.nan)
    if final_image is not None and r_map_lasco is not None and params_lasco and n_bins > 0 :
        pB_line = extract_pB_profile(final_image, r_map_lasco, theta_to_plot, edges,
                                        params_lasco['cy'], params_lasco['cx'],
                                        params_lasco['ny'], params_lasco['nx'])

    Ne_line = np.full_like(pB_line, np.nan)
    if not np.all(np.isnan(pB_line)) and n_bins > 0:
        Ne_line = invert_ablation(pB_line, r_mid, edges, n_bins)

    valid_ne_indices = ~np.isnan(Ne_line) & (Ne_line > 1e-9)
    r_all_valid_ne = r_mid[valid_ne_indices]
    Ne_all_valid = Ne_line[valid_ne_indices]

    fit_r_min = current_plot_r_min
    fit_r_max_limit_inclusive = current_plot_r_max + 1.0

    fitting_mask = (r_all_valid_ne >= fit_r_min) & (r_all_valid_ne <= fit_r_max_limit_inclusive)
    r_for_fitting = r_all_valid_ne[fitting_mask]
    Ne_for_fitting = Ne_all_valid[fitting_mask]

    # 4) Density bounds for highlighting (14–42 MHz)
    ne_14MHz_limit = density_from_frequency(14)
    ne_42MHz_limit = density_from_frequency(42)
    density_lower_highlight = np.nanmin([ne_14MHz_limit, ne_42MHz_limit])
    density_upper_highlight = np.nanmax([ne_14MHz_limit, ne_42MHz_limit])

    # 5) Fit
    from scipy.optimize import curve_fit
    initial_guess = [1e8, -2, 1e7, -4, 1e6, -6, 1e5, -8, 1e4, -10]
    fit_params = tuple(initial_guess)
    try:
        if len(r_for_fitting) >= len(initial_guess):
            popt, _ = curve_fit(triple_power, r_for_fitting, Ne_for_fitting, p0=initial_guess, maxfev=300000)
            fit_params = tuple(popt)
        elif len(r_for_fitting) > 0:
            print(f"Warning: Not enough data for robust fitting ({len(r_for_fitting)} points, need {len(initial_guess)}). Fit may be unreliable or based on initial guess.")
    except Exception as e:
        print(f"Fitting error: {e}. Using initial guess.")

    # 6) Axes & plot limits
    fig_ne, ax_ne = plt.subplots(figsize=(10, 7))
    density_plot_min_val, density_plot_max_val = 1e3, 1e8
    r_curve_for_plot_limits = np.linspace(current_plot_r_min, current_plot_r_max, 200) if current_plot_r_min < current_plot_r_max else np.array([current_plot_r_min])
    ne_on_curve_for_plot_limits = np.full_like(r_curve_for_plot_limits, np.nan)
    if len(fit_params) == 10 and len(r_curve_for_plot_limits) > 0: 
        ne_on_curve_for_plot_limits = triple_power(r_curve_for_plot_limits, *fit_params)
    y_values_to_consider = []
    if len(Ne_for_fitting) > 0: y_values_to_consider.extend(Ne_for_fitting[Ne_for_fitting > 0])
    if np.any(np.isfinite(ne_on_curve_for_plot_limits)):
        y_values_to_consider.extend(ne_on_curve_for_plot_limits[ (ne_on_curve_for_plot_limits > 0) & np.isfinite(ne_on_curve_for_plot_limits)])
    if len(y_values_to_consider) > 0:
        y_finite = [y for y in y_values_to_consider if np.isfinite(y) and y > 1e-9]
        if len(y_finite) > 0:
            density_plot_min_val = np.min(y_finite) * 0.05
            density_plot_max_val = np.max(y_finite) * 20
    for highlight_val in (density_lower_highlight, density_upper_highlight):
        if np.isfinite(highlight_val):
            density_plot_max_val = max(density_plot_max_val, highlight_val * 1.2)
            if highlight_val > 0:
                density_plot_min_val = min(density_plot_min_val, highlight_val * 0.8)
    if density_plot_min_val <= 0: density_plot_min_val = 1e1
    if density_plot_max_val <= density_plot_min_val: density_plot_max_val = density_plot_min_val * 1000

    # 7) Required call + show (plot settings unchanged)
    # generate_ne_profile_plot(
    #     ax_ne,                            # to ax
    #     r_for_fitting,                    # to r_fit_data_points
    #     Ne_for_fitting,                   # to Ne_fit_data_points
    #     fit_params,                       # to fit_params_tuple
    #     current_plot_r_min,               # to plot_r_min
    #     current_plot_r_max,               # to plot_r_max
    #     theta_to_plot,                    # to theta_deg_val
    #     (density_plot_min_val, density_plot_max_val), # to density_plot_limits
    #     {'Newkirk_C': 1.8, 'Saito1970_C': 6.0, 'Saito1977_C': 6.0}, # to model_multipliers_dict
    #     density_lower_highlight,          # to density_lower_highlight_bound
    #     density_upper_highlight           # to density_upper_highlight_bound
    # )
    plt.show()

if __name__ == "__main__":
    main()
