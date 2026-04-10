# -*- coding: utf-8 -*-
# pB_line_main.py を基に、複数の位置角でサンプリングとプロットを行うスクリプト。

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from io_and_processing import (
    load_and_prepare_instrument_data,
    combine_corona_data,
    extract_pB_profile,
)
from plotting_utils import add_frequency_secondary_axis
from constants_vdh import invert_ablation, set_u_from_instrument, density_from_frequency

# pB_line_main.py の設定・モデルを再利用（本ファイルでは pB_line_main.py を変更しない）
from pB_line_main import filename_kcor, filename_lasco

def triple_power_const(r, A0, A1, p1, A2, p2, A3, p3, A4, p4, A5, p5):
    return A0 + A1*r**p1 + A2*r**p2 + A3*r**p3 + A4*r**p4 + A5*r**p5


def compute_ne_profile_for_angle(
    theta_deg,
    final_image,
    r_map_lasco,
    params_lasco,
    params_kcor,
    r_ranges,
    edges,
    r_mid,
    fit_r_min,
    fit_r_max,
    instrument_kcor,
    instrument_lasco,
):
    """単一の位置角で pB 抽出・反転・フィットを実施し、プロット用情報を返す。"""
    result = {
        "theta": theta_deg,
        "r_all_valid_ne": np.array([]),
        "Ne_all_valid": np.array([]),
        "r_for_fitting": np.array([]),
        "Ne_for_fitting": np.array([]),
        "fit_params": (),
        "r_plot_points": np.array([]),
        "Ne_plot_points": np.array([]),
        "r_plot_line": np.array([]),
        "ne_fit_line": np.array([]),
    }

    def _fill_nan_1d(arr: np.ndarray) -> np.ndarray:
        """1D配列の NaN を最近傍線形補間で埋め、端は最近傍値で補う。"""
        if arr.size == 0:
            return arr
        filled = arr.copy()
        idx = np.arange(filled.size)
        finite_mask = np.isfinite(filled)
        if not np.any(finite_mask):
            return filled
        finite_idx = idx[finite_mask]
        finite_val = filled[finite_mask]
        # 端を最近傍値で埋める
        if finite_idx[0] > 0:
            filled[: finite_idx[0]] = finite_val[0]
        if finite_idx[-1] < filled.size - 1:
            filled[finite_idx[-1] + 1 :] = finite_val[-1]
        # 内部を線形補間
        nan_mask = ~finite_mask
        if np.any(nan_mask):
            filled[nan_mask] = np.interp(idx[nan_mask], finite_idx, finite_val)
        return filled

    bin_count = len(r_mid)
    pB_line = np.full_like(r_mid, np.nan)
    if (
        final_image is not None
        and r_map_lasco is not None
        and params_lasco
        and bin_count > 0
    ):
        pB_line = extract_pB_profile(
            final_image,
            r_map_lasco,
            theta_deg,
            edges,
            params_lasco["cy"],
            params_lasco["cx"],
            params_lasco["ny"],
            params_lasco["nx"],
            angle_halfwidth_deg=10.0,  # 扇幅を広げて外側の欠損を減らす
        )

    # --- u を半径域で切り替えて反転（K-Cor: ～2.2Rs, LASCO: >2.2Rs） ---
    Ne_line = np.full_like(pB_line, np.nan)
    if not np.all(np.isnan(pB_line)) and bin_count > 0:
        r_boundary = r_ranges.get("kcor_outer_lasco_inner", 2.2)
        # K-Cor 区間（内側）
        mask_kcor = r_mid <= r_boundary
        if np.any(mask_kcor):
            set_u_from_instrument(instrument_kcor)
            last_k = np.where(mask_kcor)[0][-1]
            try:
                pb_seg = _fill_nan_1d(pB_line[: last_k + 1])
                Ne_k = invert_ablation(
                    pb_seg,
                    r_mid[: last_k + 1],
                    edges[: last_k + 2],
                    last_k + 1,
                )
                Ne_line[: last_k + 1] = Ne_k
            except Exception as e:
                print(f"[WARN] K-Cor segment inversion failed at θ={theta_deg:.1f}°: {e}")
        # LASCO 区間（外側）
        mask_lasco = r_mid > r_boundary
        if np.any(mask_lasco):
            set_u_from_instrument(instrument_lasco)
            first_l = np.where(mask_lasco)[0][0]
            try:
                pb_seg = _fill_nan_1d(pB_line[first_l:])
                Ne_l = invert_ablation(
                    pb_seg,
                    r_mid[first_l:],
                    edges[first_l:],
                    len(r_mid) - first_l,
                )
                Ne_line[first_l:] = Ne_l
            except Exception as e:
                print(f"[WARN] LASCO segment inversion failed at θ={theta_deg:.1f}°: {e}")

    # 反転結果のうち有限値を保持（符号は問わずプロット用に残す）
    valid_ne_indices = np.isfinite(Ne_line)
    r_all_valid_ne = r_mid[valid_ne_indices]
    Ne_all_valid = Ne_line[valid_ne_indices]

    # --- 3σ 外れ値除去（log10(Ne)） ---
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

    # フィットは正の値に限定
    fitting_mask = (r_all_valid_ne >= fit_r_min) & (r_all_valid_ne <= fit_r_max) & (Ne_all_valid > 0)
    r_for_fitting = r_all_valid_ne[fitting_mask]
    Ne_for_fitting = Ne_all_valid[fitting_mask]

    # --- フィット ---
    initial_guess = [1e3, 1e8, -2, 1e7, -4, 1e6, -6, 1e5, -8, 1e4, -10]
    r_scale = 2.0
    r_fit_scaled = r_for_fitting / r_scale
    A_lower, A_upper = 1e2, 1e12
    p_lower, p_upper = -12, -1
    lower_bounds = [
        0.0,
        A_lower,
        p_lower,
        A_lower,
        p_lower,
        A_lower,
        p_lower,
        A_lower,
        p_lower,
        A_lower,
        p_lower,
    ]
    upper_bounds = [
        1e9,
        A_upper,
        p_upper,
        A_upper,
        p_upper,
        A_upper,
        p_upper,
        A_upper,
        p_upper,
        A_upper,
        p_upper,
    ]
    fit_params = tuple(initial_guess)
    try:
        from scipy.optimize import curve_fit

        if len(r_for_fitting) >= len(initial_guess):
            popt, _ = curve_fit(
                triple_power_const,
                r_fit_scaled,
                Ne_for_fitting,
                p0=initial_guess,
                bounds=(lower_bounds, upper_bounds),
                maxfev=300000,
            )
            fit_params = tuple(popt)
        elif len(r_for_fitting) > 0:
            print(
                f"[WARN] θ={theta_deg:.0f}°: insufficient points for robust fit "
                f"({len(r_for_fitting)} vs {len(initial_guess)}). Using initial guess."
            )
    except Exception as e:
        print(f"[WARN] θ={theta_deg:.0f}° fitting error: {e}. Using initial guess.")

    # プロット用範囲
    r_min_for_plot, r_max_for_plot = 1.1, 6.0
    mask_plot_points = (r_all_valid_ne >= r_min_for_plot) & (r_all_valid_ne <= r_max_for_plot)
    r_plot_points = r_all_valid_ne[mask_plot_points]
    # プロットでは非正の値も可視化したいので、極小正値にクリップしてログ軸で表示可能にする
    Ne_plot_points_raw = Ne_all_valid[mask_plot_points]
    Ne_plot_points = np.where(np.isfinite(Ne_plot_points_raw), np.maximum(Ne_plot_points_raw, 1e-12), np.nan)

    r_plot_line = np.linspace(r_min_for_plot, r_max_for_plot, 400)
    ne_fit_line = (
        triple_power_const(r_plot_line / r_scale, *fit_params)
        if len(fit_params) == 11
        else np.array([])
    )

    # フィットパラメータを標準出力に表示（Legendには載せない）
    if len(fit_params) == 11:
        A0, A1, p1, A2, p2, A3, p3, A4, p4, A5, p5 = fit_params
        print(
            f"[FIT θ={theta_deg:.0f}°] "
            f"A0={A0}, "
            f"A1={A1}, p1={p1}, "
            f"A2={A2}, p2={p2}, "
            f"A3={A3}, p3={p3}, "
            f"A4={A4}, p4={p4}, "
            f"A5={A5}, p5={p5}"
        )

    result.update(
        {
            "r_all_valid_ne": r_all_valid_ne,
            "Ne_all_valid": Ne_all_valid,
            "r_for_fitting": r_for_fitting,
            "Ne_for_fitting": Ne_for_fitting,
            "fit_params": fit_params,
            "r_plot_points": r_plot_points,
            "Ne_plot_points": Ne_plot_points,
            "r_plot_line": r_plot_line,
            "ne_fit_line": ne_fit_line,
        }
    )
    return result


def plot_combined_image_with_lines(
    final_image,
    r_map_lasco,
    params_lasco,
    r_ranges,
    angles_deg,
    colors_for_angles,
    xlim_pix=None,
    ylim_pix=None,
):
    """結合画像に複数の位置角ガイドラインを色付きで重ねる。"""
    fig, ax = plt.subplots(figsize=(10, 10))

    extent_pixels = [
        -params_lasco["cx"],
        params_lasco["nx"] - params_lasco["cx"],
        -params_lasco["cy"],
        params_lasco["ny"] - params_lasco["cy"],
    ]

    valid_data = final_image[~np.isnan(final_image) & (final_image > 0)]
    plot_vmin = np.min(valid_data) if len(valid_data) > 0 else 1e-11
    plot_vmax = np.max(valid_data) if len(valid_data) > 0 else 1e-6
    if plot_vmin <= 0:
        plot_vmin = 1e-12
    if plot_vmax <= plot_vmin:
        plot_vmax = plot_vmin * 1000

    cmap_img = plt.cm.plasma.copy()
    cmap_img.set_bad(color="lightgray")

    im = ax.imshow(
        final_image,
        origin="lower",
        cmap=cmap_img,
        norm=colors.LogNorm(vmin=plot_vmin, vmax=plot_vmax),
        extent=extent_pixels,
        aspect="equal",
    )

    int_levels = np.arange(1, int(np.floor(r_ranges["lasco_outer"])) + 1)
    ax.contour(
        r_map_lasco,
        levels=int_levels,
        colors="white",
        linewidths=1,
        linestyles="--",
        extent=extent_pixels,
        alpha=0.7,
    )

    boundary_lines = []
    for level_val, (label_text, color_val) in [
        (
            r_ranges["kcor_inner"],
            (f"{r_ranges['kcor_inner']:.1f} $R_\\odot$ (K-Cor inner)", "magenta"),
        ),
        (
            r_ranges["kcor_outer_lasco_inner"],
            (f"{r_ranges['kcor_outer_lasco_inner']:.1f} $R_\\odot$ (K-Cor/LASCO)", "green"),
        ),
        (
            r_ranges["lasco_outer"],
            (f"{r_ranges['lasco_outer']:.1f} $R_\\odot$ (LASCO outer)", "blue"),
        ),
    ]:
        if level_val <= np.nanmax(r_map_lasco) and level_val >= np.nanmin(r_map_lasco):
            ax.contour(
                r_map_lasco,
                levels=[level_val],
                colors=[color_val],
                linewidths=1.2,
                linestyles="-.",
                extent=extent_pixels,
            )
            boundary_lines.append(
                plt.Line2D(
                    [0], [0], linestyle="-.", color=color_val, linewidth=1.2, label=label_text
                )
            )

    ax.plot(0, 0, "+", color="black", markersize=12, markeredgewidth=1.5)

    line_handles = []
    for th, col in zip(angles_deg, colors_for_angles):
        theta_rad = np.radians(th)
        r_coords_rsun = np.array([0.0, r_ranges["lasco_outer"]])
        x_overlay_pix = r_coords_rsun * params_lasco["px_per_rsun"] * np.cos(theta_rad)
        y_overlay_pix = r_coords_rsun * params_lasco["px_per_rsun"] * np.sin(theta_rad)
        lh, = ax.plot(
            x_overlay_pix,
            y_overlay_pix,
            color=col,
            linestyle="-",
            linewidth=2,
            label=f"θ={th:.0f}°",
        )
        line_handles.append(lh)

    if xlim_pix is not None:
        ax.set_xlim(xlim_pix)
    if ylim_pix is not None:
        ax.set_ylim(ylim_pix)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("pB [B$_\\odot$]", fontsize=14)
    cb.ax.tick_params(labelsize=12)

    ax.set_title(
        f"pB with multiple PAs ({angles_deg[0]:.0f}–{angles_deg[-1]:.0f}°)",
        fontsize=18,
    )

    handles_for_legend = boundary_lines + line_handles
    if handles_for_legend:
        ax.legend(handles=handles_for_legend, loc="upper right", fontsize=10)

    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout()
    return fig, ax


def main(fit_r_min=2.2, fit_r_max=5.0, min_angle_deg=140.0, max_angle_deg=201.0, angle_step_deg=10.0):
    instrument_kcor = "K-Cor"
    instrument_lasco = "SOHO/LASCO"
    instrument_for_u = instrument_lasco

    angles_deg = list(np.arange(min_angle_deg, max_angle_deg, angle_step_deg))

    # 1) Load data
    data_kcor, params_kcor = load_and_prepare_instrument_data(filename_kcor, instrument_kcor)
    data_lasco, params_lasco = load_and_prepare_instrument_data(
        filename_lasco, instrument_lasco, is_lasco=True
    )

    if not (params_lasco and data_lasco is not None) and (params_kcor and data_kcor is not None):
        instrument_for_u = instrument_kcor

    set_u_from_instrument(instrument_for_u)
    print("[INFO] Density inversion assumes axisymmetry (line-of-sight symmetric shell).")

    final_image, r_map_lasco = None, None
    r_ranges = {"kcor_inner": 1.1, "kcor_outer_lasco_inner": 2.2, "lasco_outer": 7.0}

    # 2) Build grid & merge
    if params_lasco and data_lasco is not None:
        _y, _x = np.indices((params_lasco["ny"], params_lasco["nx"]))
        r_map_lasco = np.hypot(
            (_x - params_lasco["cx"]) / params_lasco["px_per_rsun"],
            (_y - params_lasco["cy"]) / params_lasco["px_per_rsun"],
        )
        if params_kcor and data_kcor is not None:
            final_image = combine_corona_data(
                data_lasco, params_lasco, data_kcor, params_kcor, r_map_lasco, r_ranges
            )
        else:
            print("Warning: K-Cor data/params failed to load. Using LASCO data only for 'final_image'.")
            final_image = data_lasco.copy()
            final_image[r_map_lasco > r_ranges["lasco_outer"]] = np.nan
            lasco_effective_inner = r_ranges.get("kcor_outer_lasco_inner", 2.0)
            final_image[r_map_lasco < lasco_effective_inner] = np.nan

    else:
        print("Critical: LASCO data/params failed to load. Cannot proceed with image processing.")
        raise SystemExit

    # 3) 共通の半径ビン設定
    bin_width = 0.01
    current_plot_r_min = r_ranges.get("kcor_inner", 1.1)
    current_plot_r_max = r_ranges.get("lasco_outer", 4.0)
    edges = np.arange(current_plot_r_min, current_plot_r_max + bin_width, bin_width)
    r_mid = (edges[:-1] + edges[1:]) / 2 if len(edges) > 1 else np.array([])

    # 4) カラーマップ準備（角度に応じて段階色）
    cmap = cm.get_cmap("viridis")
    norm = colors.Normalize(vmin=min(angles_deg), vmax=max(angles_deg))
    colors_for_angles = [cmap(norm(th)) for th in angles_deg]

    # 5) 画像に複数ラインを描画
    fig_img, ax_img = plot_combined_image_with_lines(
        final_image,
        r_map_lasco,
        params_lasco,
        r_ranges,
        angles_deg,
        colors_for_angles,
        xlim_pix=(-200, 0),
        ylim_pix=(-100, 150),
    )
    img_output = "/mnt/d/wsl/home/kinno-7010/Research_data/SDO_Mk4_SOHO/pB/pB_multi_lines.png"
    fig_img.savefig(img_output, dpi=300, bbox_inches="tight")
    print(f"✓ pB multi-line image saved: {img_output}")

    # 6) 各角度で Ne プロファイル計算
    results = []
    for th, col in zip(angles_deg, colors_for_angles):
        res = compute_ne_profile_for_angle(
            th,
            final_image,
            r_map_lasco,
            params_lasco,
            params_kcor,
            r_ranges,
            edges,
            r_mid,
            fit_r_min,
            fit_r_max,
            instrument_kcor,
            instrument_lasco,
        )
        res["color"] = col
        results.append(res)

    # 7) fig_ne をまとめて描画（scatter とフィットを同色に）
    fig_ne, ax_ne = plt.subplots(figsize=(14, 6))
    ax_ne.set_yscale("log")
    r_min_for_plot, r_max_for_plot = 1.1, 6.0
    ax_ne.set_xlim(r_min_for_plot, r_max_for_plot)
    ax_ne.axvline(x=2.2, color="gray", linestyle="--", linewidth=1)
    ax_ne.text(2.2, 1e4, " K-Cor/LASCO boundary\n (2.2 Rs)", color="black", fontsize=12, ha="left", va="bottom")

    # HF バンド
    ne_14MHz_limit = density_from_frequency(14)
    ne_42MHz_limit = density_from_frequency(42)
    density_lower_highlight = np.nanmin([ne_14MHz_limit, ne_42MHz_limit])
    density_upper_highlight = np.nanmax([ne_14MHz_limit, ne_42MHz_limit])
    ax_ne.axhspan(density_lower_highlight, density_upper_highlight, color="gray", alpha=0.15, label="HF antenna (14–42 MHz)")

    # 軸範囲を決めるために全ラインの値を収集
    y_values_for_limits = []
    for res in results:
        if len(res["Ne_for_fitting"]) > 0:
            y_values_for_limits.extend(res["Ne_for_fitting"][res["Ne_for_fitting"] > 0])
        if len(res["ne_fit_line"]) > 0:
            y_values_for_limits.extend(res["ne_fit_line"][res["ne_fit_line"] > 0])
        if len(res["Ne_plot_points"]) > 0:
            y_values_for_limits.extend(res["Ne_plot_points"][res["Ne_plot_points"] > 0])

    density_plot_min_val, density_plot_max_val = 1e4, 1e10
    # if len(y_values_for_limits) > 0:
    #     y_finite = [y for y in y_values_for_limits if np.isfinite(y) and y > 1e-9]
    #     if len(y_finite) > 0:
    #         density_plot_min_val = max(np.min(y_finite) * 0.05, 1e1)
    #         density_plot_max_val = np.max(y_finite) * 20
    # for highlight_val in (density_lower_highlight, density_upper_highlight):
    #     if np.isfinite(highlight_val):
    #         density_plot_max_val = max(density_plot_max_val, highlight_val * 1.2)
    #         if highlight_val > 0:
    #             density_plot_min_val = min(density_plot_min_val, highlight_val * 0.8)
    # if density_plot_max_val <= density_plot_min_val:
    #     density_plot_max_val = density_plot_min_val * 1000

    ax_ne.set_ylim(density_plot_min_val, density_plot_max_val)

    # プロット
    for res in results:
        col = res["color"]
        th = res["theta"]
        if len(res["r_plot_points"]) > 0:
            ax_ne.scatter(
                res["r_plot_points"],
                res["Ne_plot_points"],
                s=14,
                color=col,
                alpha=0.7,
                label=f"θ={th:.0f}° data",
            )
        if len(res["ne_fit_line"]) > 0:
            ax_ne.plot(
                res["r_plot_line"],
                res["ne_fit_line"],
                color=col,
                linestyle="-",
                linewidth=2,
                alpha=0.9,
                label=f"θ={th:.0f}° fit",
            )

    ax_ne.set_xlabel("r [R$_\\odot$]", fontsize=16)
    ax_ne.set_ylabel("N$_e$ [cm$^{-3}$]", fontsize=16)
    ax_ne.tick_params(axis="both", which="major", labelsize=14)
    ax_ne.grid(which="both", ls="--", alpha=0.7)
    ax_ne.set_title(
        f"Electron Density Profiles (θ={angles_deg[0]:.0f}–{angles_deg[-1]:.0f}°, fit: {fit_r_min:.1f}–{fit_r_max:.1f} $R_\\odot$)",
        fontsize=16,
    )
    # 右軸に周波数換算を追加
    add_frequency_secondary_axis(ax_ne)

    # 凡例が多くなるので重複ラベルを除外
    handles, labels = ax_ne.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_ne.legend(unique.values(), unique.keys(), fontsize=10, loc="upper right", ncol=2)

    ne_output = f"/mnt/d/wsl/home/kinno-7010/Research_data/SDO_Mk4_SOHO/pB/pB_multi_line_profile_{int(angles_deg[0])}-{int(angles_deg[-1])}deg.png"
    fig_ne.savefig(ne_output, dpi=300, bbox_inches="tight")
    print(f"✓ pB multi-line Ne profile saved: {ne_output}")
    plt.show()


if __name__ == "__main__":
    main(fit_r_min=2.2, fit_r_max=5.0, min_angle_deg=190.0, max_angle_deg=191.0, angle_step_deg=10.0)