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
    invert_ablation, triple_power, density_from_frequency, set_u_from_instrument, set_u
)

# --- File paths (unchanged) ---
filename_kcor = r'/mnt/d/wsl/home/kinno-7010/Research/MK4_coronagraph/MK4_coronagraph_KCOR/pB/Rawdata/20220613_025810_kcor_l2_pb.fts'
filename_lasco = r'/mnt/d/wsl/home/kinno-7010/Research/SOHO/pB/C2-PB-20220613_0258.fts'

def main(fit_r_min, fit_r_max):
    # instrument_mk4 = "Mk4"
    instrument_kcor = "K-Cor"
    instrument_lasco = "SOHO/LASCO"
    instrument_for_u = instrument_lasco

    # 1) Load data
    data_kcor, params_kcor = load_and_prepare_instrument_data(filename_kcor, instrument_kcor)
    data_lasco, params_lasco = load_and_prepare_instrument_data(filename_lasco, instrument_lasco, is_lasco=True)

    if not (params_lasco and data_lasco is not None) and (params_kcor and data_kcor is not None):
        instrument_for_u = instrument_kcor

    set_u_from_instrument(instrument_for_u)
    print("[INFO] Density inversion assumes axisymmetry (line-of-sight symmetric shell).")

    final_image, r_map_lasco, theta_to_plot = None, None, 150.0
    r_ranges = {'kcor_inner': 1.1, 'kcor_outer_lasco_inner': 2.2, 'lasco_outer': 7.0}

    # 2) Build grid & merge
    if params_lasco and data_lasco is not None :
        _y, _x = np.indices((params_lasco['ny'], params_lasco['nx']))
        r_map_lasco = np.hypot((_x - params_lasco['cx']) / params_lasco['px_per_rsun'],
                                (_y - params_lasco['cy']) / params_lasco['px_per_rsun'])
        if params_kcor and data_kcor is not None:
            final_image = combine_corona_data(data_lasco, params_lasco, data_kcor, params_kcor, r_map_lasco, r_ranges)
        else:
            print("Warning: K-Cor data/params failed to load. Using LASCO data only for 'final_image'.")
            final_image = data_lasco.copy()
            final_image[r_map_lasco > r_ranges['lasco_outer']] = np.nan
            lasco_effective_inner = r_ranges.get('kcor_outer_lasco_inner', 2.0)
            final_image[r_map_lasco < lasco_effective_inner ] = np.nan

        if final_image is not None:
            plot_combined_image(final_image, r_map_lasco, params_lasco, r_ranges, theta_deg_overlay=theta_to_plot, xlim_pix=(-200, 0), ylim_pix=(-100, 150))
    else:
        print("Critical: LASCO data/params failed to load. Cannot proceed with image processing.")
        raise SystemExit

    # 3) Extract pB profile and invert
    bin_width = 0.01
    current_plot_r_min = r_ranges.get('kcor_inner',1.1)
    current_plot_r_max = r_ranges.get('lasco_outer',4.0)
    edges = np.arange(current_plot_r_min, current_plot_r_max + bin_width, bin_width)
    r_mid = (edges[:-1] + edges[1:]) / 2 if len(edges) > 1 else np.array([])
    n_bins = len(r_mid)

    pB_line = np.full_like(r_mid, np.nan)
    if final_image is not None and r_map_lasco is not None and params_lasco and n_bins > 0 :
        pB_line = extract_pB_profile(final_image, r_map_lasco, theta_to_plot, edges,
                                        params_lasco['cy'], params_lasco['cx'],
                                        params_lasco['ny'], params_lasco['nx'])

    # --- u を半径域で切替えて反転する（K-Cor: ～2.2Rs, LASCO: >2.2Rs） ---
    Ne_line = np.full_like(pB_line, np.nan)
    if not np.all(np.isnan(pB_line)) and n_bins > 0:
        r_boundary = r_ranges.get('kcor_outer_lasco_inner', 2.2)
        # K-Cor 区間（内側）
        mask_kcor = (r_mid <= r_boundary)
        if np.any(mask_kcor):
            set_u_from_instrument(instrument_kcor)
            last_k = np.where(mask_kcor)[0][-1]
            try:
                Ne_k = invert_ablation(
                    pB_line[:last_k+1],
                    r_mid[:last_k+1],
                    edges[:last_k+2],
                    last_k+1
                )
                Ne_line[:last_k+1] = Ne_k
            except Exception as e:
                print(f"[WARN] K-Cor segment inversion failed: {e}")
        # LASCO 区間（外側）
        mask_lasco = (r_mid > r_boundary)
        if np.any(mask_lasco):
            set_u_from_instrument(instrument_lasco)
            first_l = np.where(mask_lasco)[0][0]
            try:
                Ne_l = invert_ablation(
                    pB_line[first_l:],
                    r_mid[first_l:],
                    edges[first_l:],
                    len(r_mid) - first_l
                )
                Ne_line[first_l:] = Ne_l
            except Exception as e:
                print(f"[WARN] LASCO segment inversion failed: {e}")

    valid_ne_indices = ~np.isnan(Ne_line) & (Ne_line > 1e-9)
    r_all_valid_ne = r_mid[valid_ne_indices]
    Ne_all_valid = Ne_line[valid_ne_indices]

    # --- 3σ outlier cut in log10(Ne) for plotting/fit ---
    if Ne_all_valid.size > 0:
        mask_pos = Ne_all_valid > 0
        logNe = np.log10(Ne_all_valid[mask_pos])
        if logNe.size >= 3:
            mu, sigma = np.nanmean(logNe), np.nanstd(logNe)
            keep = np.full_like(Ne_all_valid, False, dtype=bool)
            keep_indices = np.where(mask_pos)[0][np.abs(logNe - mu) <= 3 * sigma]
            keep[keep_indices] = True
            r_all_valid_ne = r_all_valid_ne[keep]
            Ne_all_valid = Ne_all_valid[keep]

    fitting_mask = (r_all_valid_ne >= fit_r_min) & (r_all_valid_ne <= fit_r_max)
    r_for_fitting = r_all_valid_ne[fitting_mask]
    Ne_for_fitting = Ne_all_valid[fitting_mask]
    if len(r_for_fitting) == 0:
        print(f"[WARN] No valid Ne points in fit range {fit_r_min}–{fit_r_max} Rsun.")
    else:
        print(f"[INFO] Fitting points: {len(r_for_fitting)} in {fit_r_min}–{fit_r_max} Rsun "
              f"(total valid Ne: {len(r_all_valid_ne)})")

    # 4) Density bounds for highlighting (14–42 MHz)
    ne_14MHz_limit = density_from_frequency(14)
    ne_42MHz_limit = density_from_frequency(42)
    density_lower_highlight = np.nanmin([ne_14MHz_limit, ne_42MHz_limit])
    density_upper_highlight = np.nanmax([ne_14MHz_limit, ne_42MHz_limit])

    # 5) Fit
    from scipy.optimize import curve_fit
    # 0次 + 5項（計6項、11パラメータ）モデル
    def triple_power_const(r, A0, A1, p1, A2, p2, A3, p3, A4, p4, A5, p5):
        return A0 + A1*r**p1 + A2*r**p2 + A3*r**p3 + A4*r**p4 + A5*r**p5

    initial_guess = [1e3, 1e8, -2, 1e7, -4, 1e6, -6, 1e5, -8, 1e4, -10]
    # r をスケールして桁を抑制
    r_scale = 2.0
    r_fit_scaled = r_for_fitting / r_scale
    # 係数と指数に現実的な制約を設けて発散を防ぐ
    A_lower, A_upper = 1e2, 1e12
    p_lower, p_upper = -12, -1
    lower_bounds = [0.0,
                    A_lower, p_lower,
                    A_lower, p_lower,
                    A_lower, p_lower,
                    A_lower, p_lower,
                    A_lower, p_lower]
    upper_bounds = [1e9,
                    A_upper, p_upper,
                    A_upper, p_upper,
                    A_upper, p_upper,
                    A_upper, p_upper,
                    A_upper, p_upper]
    fit_params = tuple(initial_guess)
    try:
        if len(r_for_fitting) >= len(initial_guess):
            popt, _ = curve_fit(
                triple_power_const,
                r_fit_scaled,
                Ne_for_fitting,
                p0=initial_guess,
                bounds=(lower_bounds, upper_bounds),
                maxfev=300000
            )
            fit_params = tuple(popt)
        elif len(r_for_fitting) > 0:
            print(f"Warning: Not enough data for robust fitting ({len(r_for_fitting)} points, need {len(initial_guess)}). Fit may be unreliable or based on initial guess.")
    except Exception as e:
        print(f"Fitting error: {e}. Using initial guess.")

    # 6) Axes & plot limits
    fig_ne, ax_ne = plt.subplots(figsize=(14, 6))
    density_plot_min_val, density_plot_max_val = 1e5, 1e10
    r_curve_for_plot_limits = np.linspace(current_plot_r_min, current_plot_r_max, 200) if current_plot_r_min < current_plot_r_max else np.array([current_plot_r_min])
    ne_on_curve_for_plot_limits = np.full_like(r_curve_for_plot_limits, np.nan)
    if len(fit_params) == 11 and len(r_curve_for_plot_limits) > 0: 
        ne_on_curve_for_plot_limits = triple_power_const(r_curve_for_plot_limits / r_scale, *fit_params)
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
    
    # フィッティングに使うデータ範囲は 2.0–3.0 Rsun（fit_r_min/max）。散布表示は 1.2–4.0 Rsun。
    r_min_for_plot, r_max_for_plot = 1.1, 6.0

    # 7) Required call + show (plot settings unchanged)
    generate_ne_profile_plot(
        ax_ne,                            # to ax
        r_for_fitting,                    # to r_fit_data_points
        Ne_for_fitting,                   # to Ne_fit_data_points
        fit_params,                       # to fit_params_tuple
        r_min_for_plot, r_max_for_plot,  # to plot_r_min, plot_r_max
        fit_r_min, fit_r_max,            # to fit_r_min, fit_r_max
        theta_to_plot,                    # to theta_deg_val
        (density_plot_min_val, density_plot_max_val), # to density_plot_limits
        {
            'BaumbachAllen_C': 8.0,
            'Newkirk_C': 1.8,
            'Saito1977_C': 6.0,
        }, # to model_multipliers_dict
        density_lower_highlight,          # to density_lower_highlight_bound
        density_upper_highlight           # to density_upper_highlight_bound
    )


    # フィット曲線をFigure 2上に描画（プロット範囲全体に延ばす）
    if len(fit_params) == 11:
        r_plot_line = np.linspace(r_min_for_plot, r_max_for_plot, 400)
        ne_fit_line = triple_power_const(r_plot_line / r_scale, *fit_params)
        A0, A1, p1, A2, p2, A3, p3, A4, p4, A5, p5 = fit_params
        lbl = (f"Fit: {A0:.2e} + {A1:.2e} r^{p1:.2f} + {A2:.2e} r^{p2:.2f} + "
               f"{A3:.2e} r^{p3:.2f} + {A4:.2e} r^{p4:.2f} + {A5:.2e} r^{p5:.2f}")
        ax_ne.plot(r_plot_line, ne_fit_line, color='red', linestyle='-', label=lbl, linewidth=3, alpha=0.8)
        print("Fit parameters: ", fit_params)
        ax_ne.legend()

    # 散布プロットを 1.2–4.0 Rsun で表示（fit に使わない領域も見せる）
    mask_plot_points = (r_all_valid_ne >= r_min_for_plot) & (r_all_valid_ne <= r_max_for_plot)
    r_plot_points = r_all_valid_ne[mask_plot_points]
    Ne_plot_points = Ne_all_valid[mask_plot_points]
    if len(r_plot_points) > 0:
        ax_ne.scatter(r_plot_points, Ne_plot_points, s=12, color='black', alpha=0.6, label=f'Data along the cyan line ({r_min_for_plot:.1f}--{r_max_for_plot:.1f} Rs)')
        ax_ne.legend()
    
    ax_ne.axvline(x=2.2, color='gray', linestyle='--', linewidth=1)
    ax_ne.text(2.2, 5e4, ' K-Cor/LASCO boundary\n (2.2 Rs)', color='black', fontsize=12, ha='left', va='bottom')

    ax_ne.set_xlim(r_min_for_plot, r_max_for_plot)
    ax_ne.set_ylim(5e4, density_plot_max_val)
    pB_line_output_path = f"/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/pB/pB_line_main_{theta_to_plot:.0f}deg_fit{fit_r_min:.1f}-{fit_r_max:.1f}.png"
    plt.savefig(pB_line_output_path, dpi=300, bbox_inches="tight")
    print(f"✓ pB line plot saved: {pB_line_output_path}")
    plt.show()

if __name__ == "__main__":
    main(fit_r_min=2.2, fit_r_max=5.0)
