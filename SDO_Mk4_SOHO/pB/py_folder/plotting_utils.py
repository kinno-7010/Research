# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from constants_vdh import (
    triple_power, density_from_frequency, frequency_from_density,
    Newkirk1961, Saito1970, Saito1977, find_rho_for_value
)

def plot_ne_scatter_and_fit(ax,
                            r_fit_points, Ne_fit_points,
                            r_curve_for_line, fit_params_tuple, theta_deg,
                            density_lower_highlight, density_upper_highlight):
    if len(fit_params_tuple) == 10:
        A1, p1, A2, p2, A3, p3, A4, p4, A5, p5 = fit_params_tuple
        fit_label_str = (f'Fit: {A1:.2e}r$^{{{p1:.2f}}}$+'
                         f'{A2:.2e}r$^{{{p2:.2f}}}$+'
                         f'{A3:.2e}r$^{{{p3:.2f}}}$+'
                         f'{A4:.2e}r$^{{{p4:.2f}}}$+'
                         f'{A5:.2e}r$^{{{p5:.2f}}}$')
    else:
        fit_label_str = f"Fit (params: {len(fit_params_tuple)})"

    # if len(r_curve_for_line) > 0 and len(fit_params_tuple) == 10:
    #     Ne_on_curve = triple_power(r_curve_for_line, *fit_params_tuple)
    #     ax.plot(r_curve_for_line, Ne_on_curve, '-', label=fit_label_str, linewidth=3, color='blue', zorder=1)

    if len(r_fit_points) > 0 and len(Ne_fit_points) > 0:
        is_lower_valid = not np.isnan(density_lower_highlight)
        is_upper_valid = not np.isnan(density_upper_highlight)

        if is_lower_valid and is_upper_valid:
            mask_in_highlight_range = (Ne_fit_points >= density_lower_highlight) & \
                                      (Ne_fit_points <= density_upper_highlight)
        elif is_lower_valid:
            mask_in_highlight_range = (Ne_fit_points >= density_lower_highlight)
        elif is_upper_valid:
            mask_in_highlight_range = (Ne_fit_points <= density_upper_highlight)
        else:
            mask_in_highlight_range = np.zeros_like(Ne_fit_points, dtype=bool)

        if np.any(mask_in_highlight_range):
            ax.scatter(r_fit_points[mask_in_highlight_range], Ne_fit_points[mask_in_highlight_range],
                       s=60, c='red', marker='+', label=f'Fit Data (14-42 MHz, θ={theta_deg:.0f}°)')
        if np.any(~mask_in_highlight_range):
            ax.scatter(r_fit_points[~mask_in_highlight_range], Ne_fit_points[~mask_in_highlight_range],
                       s=40, c='cyan', marker='x', label=f'Fit Data (Other, θ={theta_deg:.0f}°)')
    elif len(r_fit_points) > 0:
        ax.scatter(r_fit_points, np.full_like(r_fit_points, np.nan), s=40, c='gray', marker='o',
                   label="Fit Data (No Ne values)")

def _fallback_r_from_model(target_density, r_search_range, model_callable):
    if model_callable is None:
        return None
    try:
        return find_rho_for_value(
            model_callable,
            target_density,
            rho_min=r_search_range[0],
            rho_max=r_search_range[1]
        )
    except Exception:
        return None


def find_r_from_density_local(target_density, r_search_range, fit_params, fallback_model=None):
    r_values_search = np.linspace(r_search_range[0], r_search_range[1], 2000)
    densities_on_curve = triple_power(r_values_search, *fit_params)
    valid_mask = np.isfinite(densities_on_curve) & (densities_on_curve > 0)
    if not np.any(valid_mask):
        fallback_r = _fallback_r_from_model(target_density, r_search_range, fallback_model)
        if fallback_r is None:
            print(f"Warning: Could not determine r for target_density {target_density:.2e}")
        return fallback_r

    r_valid = r_values_search[valid_mask]
    dens_valid = densities_on_curve[valid_mask]
    idx = np.argmin(np.abs(dens_valid - target_density))
    best_r = r_valid[idx]
    rel_err = np.inf if target_density == 0 else abs(dens_valid[idx] - target_density) / target_density
    within_range = (target_density >= dens_valid.min()) and (target_density <= dens_valid.max())
    if within_range and rel_err <= 0.5:
        return best_r

    fallback_r = _fallback_r_from_model(target_density, r_search_range, fallback_model)
    if fallback_r is not None:
        return fallback_r

    print(f"Warning: Could not find a good match for target_density {target_density:.2e}")
    return best_r

def plot_frequency_radii(ax, r_plot_max, fit_params_tuple, density_plot_min):
    ne_14MHz = density_from_frequency(14)
    ne_42MHz = density_from_frequency(42)
    fallback_model = lambda rho: Saito1970(rho, phi=0)
    r_14MHz = find_r_from_density_local(ne_14MHz, (1.0, r_plot_max), fit_params_tuple, fallback_model=fallback_model)
    r_42MHz = find_r_from_density_local(ne_42MHz, (1.0, r_plot_max), fit_params_tuple, fallback_model=fallback_model)

    if np.isfinite(r_14MHz):
        print(f"14MHz (Ne={ne_14MHz:.2e} cm^-3) に対応する r = {r_14MHz:.2f} R_sun")
    else:
        print(f"14MHz (Ne={ne_14MHz:.2e} cm^-3) に対応する r を決定できませんでした")

    if np.isfinite(r_42MHz):
        print(f"42MHz (Ne={ne_42MHz:.2e} cm^-3) に対応する r = {r_42MHz:.2f} R_sun")
    else:
        print(f"42MHz (Ne={ne_42MHz:.2e} cm^-3) に対応する r を決定できませんでした")

def plot_reference_density_models(ax, r_curve, model_multipliers):
    nc = model_multipliers.get('Newkirk_C', 1)
    s70c = model_multipliers.get('Saito1970_C', 5.3)
    s77c = model_multipliers.get('Saito1977_C', 4.9)
    ax.plot(r_curve, nc * Newkirk1961(r_curve), 
            label=f'{nc} fold Newkirk 1961', linestyle='--', linewidth=2)
    ax.plot(r_curve, s70c * Saito1970(r_curve, phi=0),
            label=f'{s70c} fold Saito 1970 (eq.)', linestyle='--', linewidth=2)
    ax.plot(r_curve, s77c * Saito1977(r_curve), 
            label=f'{s77c} fold Saito+ 1977', linestyle='--', linewidth=2)

def plot_hf_antenna_band(ax, r_plot_max):
    ne_14MHz = density_from_frequency(14)
    ne_42MHz = density_from_frequency(42)
    ax.axhspan(ne_14MHz, ne_42MHz, color='gray', alpha=0.15,
               label='HF antenna (14–42 MHz)')
    ax.text(r_plot_max, ne_14MHz, '14 MHz', va='bottom', ha='right', fontsize=12, color='dimgray')
    ax.text(r_plot_max, ne_42MHz, '42 MHz', va='top', ha='right', fontsize=12, color='dimgray')

def setup_ne_plot_axes_and_legend(ax, x_limits, y_limits, title):
    ax.set_yscale('log')
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_xlabel('r [R$_\\odot$]', fontsize=16)
    ax.set_ylabel('N$_e$ [cm$^{-3}$]', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(which='both', ls='--', alpha=0.7)
    ax.set_title(title, fontsize=18)
    ax.legend(fontsize=11, loc='upper right')

def add_frequency_secondary_axis(ax):
    secax = ax.secondary_yaxis(
        'right',
        functions=(frequency_from_density, density_from_frequency)
    )
    secax.set_ylabel('Plasma Frequency [MHz]', fontsize=16)
    secax.set_yscale('log')
    secax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    secax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}" if x >=1 else f"{x:.1f}"))
    secax.tick_params(axis='y', which='major', labelsize=14)

def generate_ne_profile_plot(ax,
                              r_fit_data_points, Ne_fit_data_points,
                              fit_params_tuple,
                              plot_r_min, plot_r_max,
                              theta_deg_val,
                              density_plot_limits,
                              model_multipliers_dict,
                              density_lower_highlight_bound, density_upper_highlight_bound):
    if plot_r_min is None or plot_r_max is None or plot_r_min >= plot_r_max:
        r_curve_for_line_drawing = np.array([plot_r_min]) if plot_r_min is not None else np.array([])
    else:
        r_curve_for_line_drawing = np.linspace(plot_r_min, plot_r_max, 500)

    plot_ne_scatter_and_fit(ax, r_fit_data_points, Ne_fit_data_points,
                            r_curve_for_line_drawing, fit_params_tuple, theta_deg_val,
                            density_lower_highlight_bound, density_upper_highlight_bound)
    
    density_min_for_plot, _ = density_plot_limits
    if len(fit_params_tuple) == 10:
        plot_frequency_radii(ax, plot_r_max, fit_params_tuple, density_min_for_plot)
        plot_reference_density_models(ax, r_curve_for_line_drawing, model_multipliers_dict)

    plot_hf_antenna_band(ax, r_plot_max=plot_r_max)
    title = f'Electron Density Profile (θ={theta_deg_val:.0f}°)'
    setup_ne_plot_axes_and_legend(ax, (plot_r_min, plot_r_max), density_plot_limits, title)
    add_frequency_secondary_axis(ax)

def plot_combined_image(image_data, r_map_plot, params_lasco, r_ranges, theta_deg_overlay=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    extent_pixels = [-params_lasco['cx'], params_lasco['nx'] - params_lasco['cx'],
                     -params_lasco['cy'], params_lasco['ny'] - params_lasco['cy']]
    valid_data = image_data[~np.isnan(image_data) & (image_data > 0)]
    plot_vmin = np.min(valid_data) if len(valid_data) > 0 else 1e-11
    plot_vmax = np.max(valid_data) if len(valid_data) > 0 else 1e-6
    if plot_vmin <= 0: plot_vmin = 1e-12
    if plot_vmax <= plot_vmin: plot_vmax = plot_vmin * 1000
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color='lightgray')
    im = ax.imshow(image_data, origin='lower', cmap=cmap,
                   norm=LogNorm(vmin=plot_vmin, vmax=plot_vmax),
                   extent=extent_pixels, aspect='equal')
    int_levels = np.arange(1, int(np.floor(r_ranges['lasco_outer'])) + 1)
    ax.contour(r_map_plot, levels=int_levels, colors='white', linewidths=1,
               linestyles='--', extent=extent_pixels, alpha=0.7)
    boundary_lines_for_legend = [] 
    for level_val, (label_text, color) in [(r_ranges['mk4_inner'], (f"{r_ranges['mk4_inner']:.1f} $R_\\odot$ (Mk4 inner)", 'magenta')),
                                           (r_ranges['mk4_outer_lasco_inner'], (f"{r_ranges['mk4_outer_lasco_inner']:.1f} $R_\\odot$ (Mk4/LASCO)", 'green')),
                                           (r_ranges['lasco_outer'], (f"{r_ranges['lasco_outer']:.1f} $R_\\odot$ (LASCO outer)", 'blue'))]:
        if level_val <= np.nanmax(r_map_plot) and level_val >= np.nanmin(r_map_plot):
            ax.contour(r_map_plot, levels=[level_val], colors=[color], linewidths=1.2,
                       linestyles='-.', extent=extent_pixels)
            proxy_line = plt.Line2D([0], [0], linestyle='-.', color=color, linewidth=1.2, label=label_text)
            boundary_lines_for_legend.append(proxy_line)
    ax.plot(0, 0, '+', color='black', markersize=12, markeredgewidth=1.5)
    if theta_deg_overlay is not None:
        theta_rad_overlay = np.radians(theta_deg_overlay)
        r_line_min_rsun = 0
        r_line_max_rsun = r_ranges['lasco_outer']
        r_coords_rsun = np.array([r_line_min_rsun, r_line_max_rsun])
        x_overlay_pix = r_coords_rsun * params_lasco['px_per_rsun'] * np.cos(theta_rad_overlay)
        y_overlay_pix = r_coords_rsun * params_lasco['px_per_rsun'] * np.sin(theta_rad_overlay)
        line_artist_theta, = ax.plot(x_overlay_pix, y_overlay_pix, 
                                      color='cyan', linestyle='-', linewidth=2, 
                                      label=f'θ={theta_deg_overlay:.0f}°')
        boundary_lines_for_legend.append(line_artist_theta)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)
    cb = plt.colorbar(im, cax=cax, label='pB [B$_\\odot$]')
    cb.ax.tick_params(labelsize=12)
    ax.set_title(f"Combined Mk4 & LASCO pB ({r_ranges['mk4_inner']:.1f}–{r_ranges['lasco_outer']:.1f} $R_\\odot$)", fontsize=18)
    if boundary_lines_for_legend:
        ax.legend(handles=boundary_lines_for_legend, loc='upper right', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
