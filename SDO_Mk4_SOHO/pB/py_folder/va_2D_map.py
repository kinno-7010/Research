#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D_alfven_speed_map.py

Combine the 2D density map (from 2D_density_map.py) and
the 2D magnetic-field map (from 2D_magnetic_field_map.py)
using their exported CSV files, and compute the Alfvén speed
at each pixel:

    v_A = B / sqrt(4π ρ),   ρ = μ m_p n_e

where:
  - B   [G]      : magnetic-field strength (PFSS, POS)
  - n_e [cm^-3]  : electron density (from pB inversion)
  - m_p [g]      : proton mass
  - μ (~1.2)     : mean molecular weight for coronal plasma

Both maps are combined on a common (y_pix, x_pix) grid.
If either B or n_e is NaN (or n_e <= 0), v_A is set to NaN.

Expected CSV formats:
  - Density CSV from export_density_csv():
      y_pix,x_pix,x_Rsun,y_Rsun,r_Rsun,theta_deg,Ne_cm^-3

  - B-field CSV from export_B_csv():
      x_pix,y_pix,r_Rsun,PA_deg,B_G
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
sys.path.append(r'D:\wsl\home\kinno-7010\Research\SDO_Mk4_SOHO\pB')
sys.path.append(r'D:\wsl\home\kinno-7010\Research\PFSS')
sys.path.append(r'D:\wsl\home\kinno-7010\Research\PFSS\py_folder')



# --- Physical constants (cgs) ---
M_PROTON = 1.6726219e-24  # g
FOUR_PI  = 4.0 * np.pi


# ---------------------------------------
# CSV loaders
# ---------------------------------------
def load_density_csv(csv_path: str):
    """
    Load density CSV exported by export_density_csv().

    Expected columns (in this exact order):
      0: y_pix    (int)
      1: x_pix    (int)
      2: x_Rsun   (float)
      3: y_Rsun   (float)
      4: r_Rsun   (float)
      5: theta_deg(float)
      6: Ne_cm^-3 (float)
    """
    arr = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    y_pix     = arr[:, 0].astype(int)
    x_pix     = arr[:, 1].astype(int)
    x_Rsun    = arr[:, 2]
    y_Rsun    = arr[:, 3]
    r_Rsun    = arr[:, 4]
    theta_deg = arr[:, 5]
    Ne        = arr[:, 6]

    return y_pix, x_pix, x_Rsun, y_Rsun, r_Rsun, theta_deg, Ne


def load_B_csv(csv_path: str):
    """
    Load B CSV exported by export_B_csv().

    Expected columns:
      0: x_pix  (int)
      1: y_pix  (int)
      2: r_Rsun (float)
      3: PA_deg (float)
      4: B_G    (float)
    """
    arr = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    x_pix  = arr[:, 0].astype(int)
    y_pix  = arr[:, 1].astype(int)
    r_Rsun = arr[:, 2]
    PA_deg = arr[:, 3]
    B_G    = arr[:, 4]

    return x_pix, y_pix, r_Rsun, PA_deg, B_G


# ---------------------------------------
# Build 2D maps on a common grid
# ---------------------------------------
def build_2d_maps_from_csv(dens_csv_path: str, B_csv_path: str):
    """
    Construct 2D arrays for Ne, B, r, theta, x_Rsun, y_Rsun
    on a common (ny, nx) grid using the CSVs.
    """
    (
        y_dens, x_dens,
        x_Rsun_d, y_Rsun_d,
        r_Rsun_d, theta_deg_d,
        Ne_d,
    ) = load_density_csv(dens_csv_path)

    x_B, y_B, r_Rsun_B, PA_deg_B, B_G = load_B_csv(B_csv_path)

    # Determine common grid size
    nx = int(max(x_dens.max(), x_B.max())) + 1
    ny = int(max(y_dens.max(), y_B.max())) + 1
    shape = (ny, nx)

    # Initialise maps with NaN
    Ne_map        = np.full(shape, np.nan, dtype=float)
    B_map         = np.full(shape, np.nan, dtype=float)
    r_map         = np.full(shape, np.nan, dtype=float)
    theta_map     = np.full(shape, np.nan, dtype=float)
    x_Rsun_map    = np.full(shape, np.nan, dtype=float)
    y_Rsun_map    = np.full(shape, np.nan, dtype=float)

    # Fill density-related maps
    Ne_map[y_dens, x_dens]      = Ne_d
    r_map[y_dens, x_dens]       = r_Rsun_d
    theta_map[y_dens, x_dens]   = theta_deg_d
    x_Rsun_map[y_dens, x_dens]  = x_Rsun_d
    y_Rsun_map[y_dens, x_dens]  = y_Rsun_d

    # Fill B map
    B_map[y_B, x_B] = B_G

    return Ne_map, B_map, r_map, theta_map, x_Rsun_map, y_Rsun_map


# ---------------------------------------
# Compute Alfvén speed
# ---------------------------------------
def compute_alfven_speed(B_map: np.ndarray,
                         Ne_map: np.ndarray,
                         mu: float = 1.2) -> np.ndarray:
    """
    Compute Alfvén speed v_A [km/s] on the same 2D grid:

        v_A = B / sqrt(4π ρ),  ρ = μ m_p n_e

    Parameters
    ----------
    B_map : 2D array [G]
    Ne_map: 2D array [cm^-3]
    mu    : mean molecular weight (~1.2 for typical corona)

    Returns
    -------
    vA_kms : 2D array [km/s], NaN where B or Ne is invalid / non-positive.
    """
    B = B_map.astype(float)
    Ne = Ne_map.astype(float)

    rho = mu * M_PROTON * Ne  # [g/cm^3]

    vA = np.full_like(B, np.nan, dtype=float)

    mask = np.isfinite(B) & np.isfinite(Ne) & (Ne > 0.0)
    # Babs を想定しているが、万が一負の B が入っていたら絶対値を取る
    B_eff = np.abs(B[mask])

    vA[mask] = B_eff / np.sqrt(FOUR_PI * rho[mask])  # [cm/s]
    vA_kms = vA / 1.0e5  # -> [km/s]

    return vA_kms


# ---------------------------------------
# Plotting
# ---------------------------------------
# ---------------------------------------
# Plotting
# ---------------------------------------

def _infer_pixel_extent_from_rsun_maps(
    x_Rsun_map: np.ndarray,
    y_Rsun_map: np.ndarray,
) -> tuple[list[float], float, float]:
    """
    x_Rsun_map, y_Rsun_map と配列インデックス (xx, yy) の線形関係
        x_pix ≃ a_x * x_Rsun + b_x
        y_pix ≃ a_y * y_Rsun + b_y
    を最小二乗で求めて，太陽中心 (cx, cy) と
    「太陽中心からのピクセル座標」の extent を返す。
    """
    ny, nx = x_Rsun_map.shape
    yy, xx = np.indices((ny, nx))

    # --- x方向: j = a_x * x_Rsun + b_x
    mask_x = np.isfinite(x_Rsun_map)
    if np.any(mask_x):
        xvals = x_Rsun_map[mask_x].ravel()
        jvals = xx[mask_x].ravel().astype(float)
        A_x = np.vstack([xvals, np.ones_like(xvals)]).T
        a_x, b_x = np.linalg.lstsq(A_x, jvals, rcond=None)[0]
        cx = b_x          # 太陽中心の x ピクセル座標
        px_per_rsun_x = a_x
    else:
        cx = nx / 2.0
        px_per_rsun_x = np.nan

    # --- y方向: i = a_y * y_Rsun + b_y
    mask_y = np.isfinite(y_Rsun_map)
    if np.any(mask_y):
        yvals = y_Rsun_map[mask_y].ravel()
        ivals = yy[mask_y].ravel().astype(float)
        A_y = np.vstack([yvals, np.ones_like(yvals)]).T
        a_y, b_y = np.linalg.lstsq(A_y, ivals, rcond=None)[0]
        cy = b_y          # 太陽中心の y ピクセル座標
        px_per_rsun_y = a_y
    else:
        cy = ny / 2.0
        px_per_rsun_y = np.nan

    # （必要なら）平均の px_per_rsun も求められるが，ここでは使わない
    # px_per_rsun = 0.5 * (px_per_rsun_x + px_per_rsun_y)

    # 太陽中心 (cx, cy) を原点とするピクセル座標系の extent
    extent_pixels = [-cx, nx - cx, -cy, ny - cy]

    return extent_pixels, cx, cy


def plot_alfven_map(
    vA_map: np.ndarray,
    r_map: np.ndarray,
    x_Rsun_map: np.ndarray,
    y_Rsun_map: np.ndarray,
    r_ranges: dict,
    title: str = "2D Alfvén Speed Map",
    out_png: str | None = None,
):
    """
    2D Alfvén speed map を「太陽中心からのピクセル」座標でプロット。

    - 画像座標: X, Y [pixels from Sun center]
    - r_map は R⊙ 単位のままで，等半径のコンターや境界リングに使用する。
    """

    # --- extent を Rsun ではなく「ピクセル（太陽中心から）」に変換
    extent_pixels, cx, cy = _infer_pixel_extent_from_rsun_maps(
        x_Rsun_map, y_Rsun_map
    )

    # --- カラースケールの範囲（有効値から自動設定）
    finite_vals = vA_map[np.isfinite(vA_map) & (vA_map > 0)]
    if finite_vals.size == 0:
        vmin, vmax = 10.0, 10000.0
    else:
        vmin = np.nanpercentile(finite_vals, 5)    # 下位5%
        vmax = np.nanpercentile(finite_vals, 95)   # 上位5%
        vmin = max(vmin, 10.0)       # [km/s]
        vmax = max(vmax, vmin * 1.5)

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color="lightgray")

    # --- 本体画像（vA）
    im = ax.imshow(
        vA_map,
        origin="lower",
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=extent_pixels,   # ★ ピクセル座標
        aspect="equal",
    )

    # --- 等半径 (整数 R⊙) コンター
    if np.isfinite(r_map).any():
        max_r = np.nanmax(r_map)
        int_levels = np.arange(1, int(np.floor(max_r)) + 1)
        ax.contour(
            r_map,
            levels=int_levels,
            colors="white",
            linewidths=1.0,
            linestyles="--",
            alpha=0.7,
            extent=extent_pixels,  # ★ 同じ extent
        )

    # --- 境界リング (Mk4, Mk4/LASCO, LASCO outer)
    boundary_lines_for_legend = []
    for level_val, (label_text, color) in [
        (r_ranges["mk4_inner"], (f"{r_ranges['mk4_inner']:.1f} $R_\\odot$ (Mk4 inner)", "magenta")),
        (r_ranges["mk4_outer_lasco_inner"], (f"{r_ranges['mk4_outer_lasco_inner']:.1f} $R_\\odot$ (Mk4/LASCO)", "green")),
        (r_ranges["lasco_outer"], (f"{r_ranges['lasco_outer']:.1f} $R_\\odot$ (LASCO outer)", "blue")),
    ]:
        if level_val <= np.nanmax(r_map) and level_val >= np.nanmin(r_map):
            ax.contour(
                r_map,
                levels=[level_val],
                colors=[color],
                linewidths=1.2,
                linestyles="-.",
                extent=extent_pixels,  # ★ 同じ extent
            )
            proxy = plt.Line2D(
                [0],
                [0],
                linestyle="-.",
                color=color,
                linewidth=1.2,
                label=label_text,
            )
            boundary_lines_for_legend.append(proxy)

    # --- 太陽中心 (0, 0) [pixels] にマーカー
    ax.plot(0.0, 0.0, "+", color="black", markersize=10, markeredgewidth=1.5)

    # --- カラーバー
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("Alfvén speed $v_A$ [km s$^{-1}$]", fontsize=14)

    # --- 軸ラベル・タイトル
    ax.set_xlabel("X [pixels from Sun center]")
    ax.set_ylabel("Y [pixels from Sun center]")
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # --- 表示範囲（ここで指定する値は本当に pixel 単位になる）
    xlim_pix = (-150, 0)
    ylim_pix = (-100, 150)
    ax.set_xlim(xlim_pix)
    ax.set_ylim(ylim_pix)

    if boundary_lines_for_legend:
        ax.legend(handles=boundary_lines_for_legend, loc="upper right", fontsize=10)

    plt.tight_layout()
    if out_png is not None:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"✓ Saved Alfvén map PNG to: {out_png}")
    plt.show()

    return fig, ax


# ---------------------------------------
# CSV export for v_A
# ---------------------------------------
def export_alfven_csv(
    out_csv: str,
    vA_map: np.ndarray,
    Ne_map: np.ndarray,
    B_map: np.ndarray,
    r_map: np.ndarray,
    theta_map: np.ndarray,
    x_Rsun_map: np.ndarray,
    y_Rsun_map: np.ndarray,
):
    """
    Export per-pixel quantities where v_A is finite:

      y_pix,x_pix,x_Rsun,y_Rsun,r_Rsun,theta_deg,Ne_cm^-3,B_G,vA_km_s
    """
    ny, nx = vA_map.shape
    yy, xx = np.indices((ny, nx))

    mask = np.isfinite(vA_map)  # 有効な Alfvén 速度のみ出力

    y_pix_col    = yy[mask].ravel().astype(int)
    x_pix_col    = xx[mask].ravel().astype(int)
    x_Rsun_col   = x_Rsun_map[mask].ravel()
    y_Rsun_col   = y_Rsun_map[mask].ravel()
    r_Rsun_col   = r_map[mask].ravel()
    theta_col    = theta_map[mask].ravel()
    Ne_col       = Ne_map[mask].ravel()
    B_col        = B_map[mask].ravel()
    vA_kms_col   = vA_map[mask].ravel()

    header = "y_pix,x_pix,x_Rsun,y_Rsun,r_Rsun,theta_deg,Ne_cm^-3,B_G,vA_km_s"
    data = np.column_stack(
        [
            y_pix_col,
            x_pix_col,
            x_Rsun_col,
            y_Rsun_col,
            r_Rsun_col,
            theta_col,
            Ne_col,
            B_col,
            vA_kms_col,
        ]
    )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    np.savetxt(out_csv, data, delimiter=",", header=header, comments="")
    print(f"✓ Saved Alfvén CSV to: {out_csv} ({data.shape[0]} rows)")


# ---------------------------------------
# Main
# ---------------------------------------
def main():
    # === PFSS source-surface radius (Rss) をここで設定 ===
    #    例: 2.0, 2.5, 3.0 など（単位は R⊙、太陽中心からの半径）
    rss = 2.5

    # ---- 入力 CSV のパスをあなたの環境に合わせて設定 ----

    # 2D_density_map.py で出力した CSV（軸対称 or 球対称）
    # density_csv = r"/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/pB/2D_density_map_axi.csv"
    density_csv = r"/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/pB/2D_density_map_axi_20220613_0300.csv"

    # 2D_magnetic_field_map.py で出力した CSV
    # ここでは Bmap のファイル名に rss を含める想定にしています。
    # （PFSS 側のコードの出力名を例えば "Bmap_2D_POS_rss2.5.csv" のように揃えてください）
    out_dir_pfss = r"/mnt/d/wsl/home/kinno-7010/Research/PFSS"
    B_csv = os.path.join(out_dir_pfss, f"magnetic_field_2D_map_rss={rss:.1f}_20220613_0300.csv")
    # B_csv = os.path.join(out_dir_pfss, f"magnetic_field_2D_map_rss={rss:.1f}_20220613_0900.csv")

    # ---- 出力ファイル ----
    out_dir  = r"/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/pB/"
    out_png  = os.path.join(out_dir, f"2D_alfven_speed_map_rss={rss:.1f}_20220613_0300.png")
    out_csv  = os.path.join(out_dir, f"2D_alfven_speed_map_rss={rss:.1f}_20220613_0300.csv")

    # out_png  = os.path.join(out_dir, f"2D_alfven_speed_map_rss={rss:.1f}_20110920.png")
    # out_csv  = os.path.join(out_dir, f"2D_alfven_speed_map_rss={rss:.1f}_20110920.csv")

    # Mk4/LASCO の半径範囲（他のコードと揃える）
    r_ranges = {
        "mk4_inner": 1.0,
        "mk4_outer_lasco_inner": 2.2,
        "lasco_outer": 7.0,
    }

    # ---- CSV から 2D マップを構築 ----
    (
        Ne_map,
        B_map,
        r_map,
        theta_map,
        x_Rsun_map,
        y_Rsun_map,
    ) = build_2d_maps_from_csv(density_csv, B_csv)

    # ---- Alfvén 速度の計算 ----
    vA_map = compute_alfven_speed(B_map, Ne_map, mu=1.2)

    # ---- プロット ----
    title = (
        "2D Alfvén Speed from PFSS (B) + pB Inversion (Ne)\n2022-06-13 03:00:00 UT"
        # "2D Alfvén Speed from PFSS (B) + pB Inversion (Ne)\n2011-09-22 09:00:00 UT"
        f"Rss = {rss:.1f} R$_\\odot$"
    )
    plot_alfven_map(
        vA_map=vA_map,
        r_map=r_map,
        x_Rsun_map=x_Rsun_map,
        y_Rsun_map=y_Rsun_map,
        r_ranges=r_ranges,
        title=title,
        out_png=out_png,
    )

    # ---- CSV 保存 ----
    export_alfven_csv(
        out_csv=out_csv,
        vA_map=vA_map,
        Ne_map=Ne_map,
        B_map=B_map,
        r_map=r_map,
        theta_map=theta_map,
        x_Rsun_map=x_Rsun_map,
        y_Rsun_map=y_Rsun_map,
    )


if __name__ == "__main__":
    main()
