#!/usr/bin/env python3
"""
Compute a 2D magnetic-field-strength map (Zucca+2014 style) using pfsspy
from an SDO/HMI radial synoptic map plus an HMI LOS magnetogram.

Workflow
--------
1. Load the HMI radial synoptic map (hmi.synoptic_mr_polfil_720s.*.Mr_polfil.fits)
   and optionally resample it to (ny, nx).
2. Run a PFSS extrapolation with a user-specified source surface height Rss.
3. Use the PFSS solution to construct an |B|(r, latitude) map by averaging
   over ±10° in Carrington longitude around a central longitude and its
   antipode (lon0+180°), following Zucca et al. (2014).
4. Plot the 2D |B| map with the 'plasma' colormap.

References
----------
- Zucca et al., 2014, A&A, 564, A47, doi:10.1051/0004-6361/201322650
- pfsspy: Stansby et al., 2020, JOSS, 5, 2732, doi:10.21105/joss.02732
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.units as u

import sunpy.map

import pfsspy
import pfsspy.utils
from io_and_processing import load_and_prepare_instrument_data, combine_corona_data
from plotting_utils import add_radial_guides_on_ax

def make_lasco_grid_and_final_image(
    filename_lasco: str,
    filename_mk4: str | None,
    r_ranges: dict,
):
    """
    Replicates the exact grid used by the 2D density map:
    - uses LASCO header as reference grid
    - optionally stitches Mk4/K-Cor into the inner annulus
    Returns:
        final_image        : 2D pB (stitched) on LASCO grid (for plotting extent/masks only)
        r_map_lasco        : 2D heliocentric distance in Rsun at each pixel (LASCO grid)
        theta_map_lasco    : 2D position angle [rad], measured from +X to +Y (same as your density code)
        params_lasco       : dict with LASCO geometry
    """
    # Load LASCO
    data_lasco, params_lasco = load_and_prepare_instrument_data(filename_lasco, "SOHO/LASCO", is_lasco=True)

    # Build r, theta on LASCO grid
    y_idx, x_idx = np.indices((params_lasco["ny"], params_lasco["nx"]))
    x_norm = (x_idx - params_lasco["cx"]) / params_lasco["px_per_rsun"]
    y_norm = (y_idx - params_lasco["cy"]) / params_lasco["px_per_rsun"]
    r_map_lasco = np.hypot(x_norm, y_norm)              # [Rsun]
    theta_map_lasco = np.arctan2(y_norm, x_norm)        # [rad], 0 along +X, CCW

    # Optionally stitch Mk4/K-Cor to replicate density-grid content (ensures identical masks)
    if filename_mk4 is not None and os.path.exists(filename_mk4):
        data_mk4, params_mk4 = load_and_prepare_instrument_data(filename_mk4, "Mk4/K-Cor", is_lasco=False)
        final_image = combine_corona_data(
            data_lasco, params_lasco, data_mk4, params_mk4, r_map_lasco, r_ranges
        )
    else:
        final_image = data_lasco.copy()
        final_image[r_map_lasco < r_ranges.get("mk4_outer_lasco_inner", 2.2)] = np.nan
        final_image[r_map_lasco > r_ranges.get("lasco_outer", 7.0)] = np.nan

    return final_image, r_map_lasco, theta_map_lasco, params_lasco



# ------------------------------------------------------------
# 1. HMI を読み込み
#    - LOS マップ: CRLN_OBS を取得するため
#    - Radial synoptic map: PFSS 境界条件 Br(φ, θ)
# ------------------------------------------------------------

def load_hmi_los_map(hmi_fits_path: str) -> sunpy.map.GenericMap:
    """
    Load an SDO/HMI line-of-sight magnetogram as a SunPy map.

    Parameters
    ----------
    hmi_fits_path : str
        Path to the HMI FITS file (e.g. hmi.M_720s.*.fits).

    Returns
    -------
    hmi_map : sunpy.map.GenericMap
        Helioprojective LOS magnetogram in units of Gauss.
    """
    hmi_map = sunpy.map.Map(hmi_fits_path)
    # Ensure the data are float64 for subsequent operations
    data = hmi_map.data.astype(np.float64)
    hmi_map = sunpy.map.Map(data, hmi_map.meta)
    return hmi_map


def load_hmi_synoptic_br_map(
    synoptic_fits_path: str,
    ny: int = 180,
    nx: int = 360,
) -> sunpy.map.GenericMap:
    """
    Load an HMI radial synoptic map (hmi.synoptic_mr_polfil_720s.*.Mr_polfil.fits)
    and optionally resample it to (ny, nx).

    This provides Br(longitude, latitude) on a regular Carrington
    CEA grid suitable for pfsspy.Input.
    """
    # 1) ファイルを SunPy Map として読み込み
    syn_map = sunpy.map.Map(synoptic_fits_path)

    # 2) データを float64 にしておく
    data = syn_map.data.astype(np.float64)
    syn_map = sunpy.map.Map(data, syn_map.meta)

    # 3) pfsspy のヘッダ修正 (★ 戻り値を代入しない ★)
    fix_hmi_meta = getattr(pfsspy.utils, "fix_hmi_meta", None)
    if callable(fix_hmi_meta):
        # 以前は: syn_map = fix_hmi_meta(syn_map) となっていたはず
        fix_hmi_meta(syn_map)  # in-place で meta を修正する

    # 4) 必要なら (ny, nx) にリサンプル
    ny_cur, nx_cur = syn_map.data.shape
    if (ny_cur, nx_cur) != (ny, nx):
        # new_dimensions は [nx, ny] * u.pix を受け取り、(ny, nx) を返す
        syn_map = syn_map.resample([nx, ny] * u.pix)

    # 5) 単位を Gauss と見なす（Mx/cm^2 ≒ G）
    syn_map.meta["BUNIT"] = "G"

    # 6) pfsspy で full-sun CEA マップとして認識されるかチェック（任意）
    try:
        pfsspy.utils.is_full_sun_synoptic_map(syn_map, error=True)
    except Exception as e:
        print(
            "WARNING: map may not be recognized as a full-sun synoptic CEA map by pfsspy:",
            repr(e),
        )

    return syn_map


# ------------------------------------------------------------
# 2. PFSS 解の計算
# ------------------------------------------------------------

def compute_pfss_solution(
    br_map: sunpy.map.GenericMap,
    rss: float = 2.5,
    nr: int = 60,
) -> pfsspy.Output:
    """
    Compute a PFSS solution using pfsspy from a Br CEA map.

    Parameters
    ----------
    br_map : sunpy.map.GenericMap
        Radial magnetic field at the photosphere (Carrington CEA).
    rss : float, optional
        Source surface radius (Rss) in units of R_sun.
    nr : int, optional
        Number of radial grid cells between 1 R_sun and Rss.

    Returns
    -------
    pfss_out : pfsspy.Output
        PFSS solution object.
    """
    # pfsspy.Input expects a CEA, full-sun, regular grid in (phi, s=cos(theta)).
    pfss_input = pfsspy.Input(br_map, nr, rss)
    pfss_out = pfsspy.pfss(pfss_input)
    return pfss_out


# ------------------------------------------------------------
# 3. Zucca+2014 型 2D |B| マップの構成
# ------------------------------------------------------------

def compute_B_pos_map(
    pfss_out: pfsspy.Output,
    lon_center_deg: float,
    lon_half_width_deg: float = 10.0,
):
    """
    Construct a full-longitude equatorial |B|(r, phi) map from a PFSS solution,
    and use it for a plane-view (x, y) plot with the Sun center at (0, 0).

    NOTE
    ----
    - In contrast to the previous implementation, we no longer average over
      ±lon_half_width_deg around a central longitude.
    - Instead, we take an equatorial slice (theta ≈ 90 deg) and keep ALL
      Carrington longitudes (0–360 deg).
    - The parameters lon_center_deg and lon_half_width_deg are kept only for
      API compatibility with the existing main().

    Parameters
    ----------
    pfss_out : pfsspy.Output
        PFSS solution from pfsspy.pfss.
    lon_center_deg : float
        Unused (kept for API compatibility).
    lon_half_width_deg : float
        Unused (kept for API compatibility).

    Returns
    -------
    r_centers : ndarray, shape (nr,)
        Radial coordinates (r / R_sun) on the PFSS grid (cell centers).
    phi_centers_deg : ndarray, shape (nphi,)
        Carrington longitudes (deg) on the PFSS grid (cell centers, 0–360).
    B_eq_center : ndarray, shape (nphi, nr)
        Equatorial |B|(phi, r) [Gauss] at cell centers.
    """
    # 1. Vector field from PFSS: bg has shape (nphi+1, ns+1, nr+1, 3)
    bg = pfss_out.bg

    # 2. Compute |B| on the 3D grid
    #    BUNIT="Gauss" is not a FITS-standard unit, so pfsspy often treats it
    #    as dimensionless. Numerically, 1 Mx/cm^2 ≈ 1 G, so we use the values
    #    as Gauss when bg.unit is dimensionless.
    if bg.unit == u.dimensionless_unscaled:
        B_mag = np.linalg.norm(bg.value, axis=-1)  # (nphi+1, ns+1, nr+1)
    else:
        B_mag = np.linalg.norm(bg.to(u.G).value, axis=-1)

    # Drop the last phi index (periodic)
    B_mag_phi = B_mag[:-1, :, :]  # (nphi, ns+1, nr+1)
    nphi, ns_plus, nr_plus = B_mag_phi.shape

    # 3. PFSS grid geometry
    grid = pfss_out.grid
    # radial cell centers
    r_centers = np.exp(grid.rc)  # (nr,)
    nr = r_centers.size

    # s = cos(theta) at cell centers, length ns
    s_centers = grid.sc  # (ns,)
    # equator: theta = 90 deg → s = cos(theta) = 0
    idx_eq = int(np.argmin(np.abs(s_centers)))

    if idx_eq < 0 or idx_eq >= (ns_plus - 1):
        raise ValueError(
            f"Equatorial index out of range: idx_eq={idx_eq}, ns_plus={ns_plus}"
        )

    # 4. Take an equatorial slice by averaging the two s-edges that bracket s≈0
    #    Result: |B|(phi, r_edge) with shape (nphi, nr+1)
    B_eq_edges = 0.5 * (
        B_mag_phi[:, idx_eq, :] + B_mag_phi[:, idx_eq + 1, :]
    )  # (nphi, nr_plus)

    # 5. Convert r-edge values to r-centered values: (nphi, nr)
    if nr_plus != (nr + 1):
        raise ValueError(
            f"Inconsistent radial sizes: nr_plus={nr_plus}, nr_centers={nr}"
        )
    B_eq_center = 0.5 * (B_eq_edges[:, :-1] + B_eq_edges[:, 1:])  # (nphi, nr)

    # 6. Longitude cell centers (0–360 deg)
    phi_centers_deg = np.linspace(0.0, 360.0, nphi, endpoint=False)

    return r_centers, phi_centers_deg, B_eq_center


# ------------------------------------------------------------
# 4. プロット部分
# ------------------------------------------------------------

def plot_B_2d_map(
    r_centers: np.ndarray,
    lat_centers: np.ndarray,
    B_pos_center: np.ndarray,
    rss: float,
    lon_center_deg: float,
    px_per_rsun: float = 100.0,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "plasma",
):
    """
    2D |B| マップを (x, y) 平面にプロットする。
    太陽中心を (0,0)、軸の単位を pixel とし、aspect='equal' で表示する。

    Parameters
    ----------
    r_centers : ndarray, shape (nr,)
        Radial coordinates r / R_sun at cell centers (from PFSS grid).
    lat_centers : ndarray, shape (ns,)
        「角度」座標 [deg]。現在は heliographic latitude を使っているが、
        一般には極座標の角度として扱う。
    B_pos_center : ndarray, shape (ns, nr)
        |B| [G] averaged over the POS-like longitude windows.
    rss : float
        Source surface radius Rss [R_sun], used only for annotation.
    lon_center_deg : float
        Central Carrington longitude used for the averaging (for annotation).
    px_per_rsun : float, optional
        1 R_sun あたりの pixel 数（スケール因子）。
    vmin, vmax : float, optional
        Color scale limits in Gauss.
    cmap : str, optional
        Matplotlib colormap name (default: 'plasma').
    """

    # lat_centers を極座標の角度として扱う
    angle_centers_deg = lat_centers

    # 1. pcolormesh 用に r と角度のエッジを作る
    nr = len(r_centers)
    ns = len(angle_centers_deg)

    # Radial edges
    r_edges = np.empty(nr + 1, dtype=float)
    r_edges[1:-1] = 0.5 * (r_centers[:-1] + r_centers[1:])
    r_edges[0] = r_centers[0] - 0.5 * (r_centers[1] - r_centers[0])
    r_edges[-1] = r_centers[-1] + 0.5 * (r_centers[-1] - r_centers[-2])

    # Angle edges (deg)
    angle_edges = np.empty(ns + 1, dtype=float)
    angle_edges[1:-1] = 0.5 * (angle_centers_deg[:-1] + angle_centers_deg[1:])
    angle_edges[0] = angle_centers_deg[0] - 0.5 * (angle_centers_deg[1] - angle_centers_deg[0])
    angle_edges[-1] = angle_centers_deg[-1] + 0.5 * (angle_centers_deg[-1] - angle_centers_deg[-2])

    # rad に変換（平面上の極座標の角度として使う）
    angle_edges_rad = np.deg2rad(angle_edges)

    # 2. 極座標グリッド → x,y（まず R_sun 単位）
    R_edges, ANG_edges = np.meshgrid(r_edges, angle_edges_rad)
    X_edges_rs = R_edges * np.cos(ANG_edges)
    Y_edges_rs = R_edges * np.sin(ANG_edges)

    # 3. pixel 単位に変換
    X_edges = X_edges_rs * px_per_rsun
    Y_edges = Y_edges_rs * px_per_rsun

    # 4. (x, y) 平面にプロット：太陽中心が (0,0)
    fig, ax = plt.subplots(figsize=(8, 8))

    pcm = ax.pcolormesh(
        X_edges,
        Y_edges,
        B_pos_center,
        shading="auto",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap=cmap,
    )
    # カラーバーの幅を細くする
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.01)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label(r"$|B|$ [G]", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.set_xlabel(r"x [pixel]", fontsize=14)
    ax.set_ylabel(r"y [pixel]", fontsize=14)

    ax.set_title(
        rf"PFSS |B| map (R$_{{ss}}$={rss:.2f} R$_\odot$)"
        # rf"lon$_{{center}}$={lon_center_deg:.1f}$^\circ$)"
        # rf"1 R$_\odot$={px_per_rsun:.1f} px)"
    , fontsize=16)

    # 5. 原点まわりで対称にし，aspect を equal に
    ax.set_aspect("equal", "box")
    r_max_pix = r_edges[-1] * px_per_rsun
    ax.set_xlim(-r_max_pix, r_max_pix)
    ax.set_ylim(-r_max_pix, r_max_pix)

    fig.tight_layout()
    return fig, ax

def export_B_csv(
    out_csv: str,
    B_map: np.ndarray,
    r_map: np.ndarray,
    theta_map: np.ndarray,
    params_lasco: dict,
):
    """
    Save (x_pix, y_pix, r[Rsun], PA[deg], B[G]) per valid pixel.
    """
    ny, nx = B_map.shape
    y_idx, x_idx = np.indices((ny, nx))
    mask = np.isfinite(B_map)

    x_pix = x_idx[mask]
    y_pix = y_idx[mask]
    r_val = r_map[mask]
    pa_deg = np.degrees(theta_map[mask])
    B_val = B_map[mask]

    arr = np.column_stack([x_pix, y_pix, r_val, pa_deg, B_val])
    header = "x_pix,y_pix,r_Rsun,PA_deg,B_G"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    np.savetxt(out_csv, arr, delimiter=",", header=header, comments="")
    print(f"✓ CSV exported: {out_csv}  ({arr.shape[0]} points)")

# ------------------------------------------------------------
# 5. main: Rss やファイルパスをここで指定
# ------------------------------------------------------------

# def main():
#     filename_mk4 = r"/mnt/d/wsl/home/kinno-7010/Research_data/MK4_coronagraph/MK4_coronagraph_KCOR/pB/20220613_025810_kcor_l2.fts"
#     filename_lasco = r"/mnt/d/wsl/home/kinno-7010/Research_data/SOHO/pB/C2-PB-20220613_0258.fts"
    
#     r_ranges = {"mk4_inner": 1.0, "mk4_outer_lasco_inner": 2.2, "lasco_outer": 7.0}
    
    
#     final_image, r_map, theta_map, params_lasco = make_lasco_grid_and_final_image(
#         filename_lasco, filename_mk4, r_ranges
#     )
    
    
#     parser = argparse.ArgumentParser(
#         description=(
#             "Compute a 2D magnetic field strength map from an HMI radial synoptic map "
#             "using pfsspy (Zucca et al. 2014 style)."
#         )
#     )

#     parser.add_argument(
#         "--synoptic",
#         type=str,
#         default=(
#             "/mnt/d/wsl/home/kinno-7010/Research_data/SDO/HMI/Rawdata/"
#             # "hmi.synoptic_mr_polfil_720s.2258.Mr_polfil.fits"
#             "hmi.synoptic_mr_polfil_720s.2115.Mr_polfil.fits"
#         ),
#         help="Path to HMI hmi.synoptic_mr_polfil_720s.*.Mr_polfil.fits file.",
#     )
#     parser.add_argument(
#         "--hmi",
#         type=str,
#         default=(
#             "/mnt/d/wsl/home/kinno-7010/Research_data/SDO/HMI/Rawdata/"
#             # "hmi.M_720s.20220613_030000_TAI.fits"
#             "hmi.M_720s.20110922_090000_TAI.fits"
#         ),
#         help="Path to HMI hmi.M_720s.*.fits file (used to get CRLN_OBS).",
#     )
#     parser.add_argument(
#         "--rss",
#         type=float,
#         default=2.5,
#         help="Source surface radius Rss in units of R_sun (e.g., 2.0–3.5).",
#     )
#     parser.add_argument(
#         "--nr",
#         type=int,
#         default=60,
#         help="Number of radial grid cells between 1 R_sun and Rss.",
#     )
#     parser.add_argument(
#         "--ny",
#         type=int,
#         default=180,
#         help="Number of latitude pixels in the synoptic CEA Carrington map.",
#     )
#     parser.add_argument(
#         "--nx",
#         type=int,
#         default=360,
#         help="Number of longitude pixels in the synoptic CEA Carrington map.",
#     )
#     parser.add_argument(
#         "--lon0",
#         type=float,
#         default=None,
#         help=(
#             "Central Carrington longitude (deg) for ±10° POS averaging. "
#             "Default: use CRLN_OBS from the HMI LOS map."
#         ),
#     )
#     parser.add_argument(
#         "--px-per-rsun",
#         type=float,
#         default=100.0,
#         help="1 R_sun あたりの pixel 数（平面図マップのスケール）.",
#     )
#     parser.add_argument(
#         "--csv",
#         type=str,
#         # default="/mnt/d/wsl/home/kinno-7010/Research_data/PFSS/magnetic_field_2D_map_rss=2.5_20220613_0300.csv",
#         default="/mnt/d/wsl/home/kinno-7010/Research_data/PFSS/magnetic_field_2D_map_rss=2.5_20110922_0900.csv",
#         help="2D |B| マップを出力する CSV ファイルパス.",
#     )
#     parser.add_argument(
#         "--vmin",
#         type=float,
#         default=0.1,
#         help="Minimum of color scale in Gauss (optional).",
#     )
#     parser.add_argument(
#         "--vmax",
#         type=float,
#         default=10,
#         help="Maximum of color scale in Gauss (optional).",
#     )

#     args = parser.parse_args()

#     # 1. HMI LOS magnetogram（CRLN_OBS を取得するだけ）
#     hmi_map = load_hmi_los_map(args.hmi)

#     # 2. HMI radial synoptic Br map
#     br_map = load_hmi_synoptic_br_map(args.synoptic, ny=args.ny, nx=args.nx)

#     # 3. POS 平均の中心経度を決定
#     if args.lon0 is None:
#         if "CRLN_OBS" in hmi_map.meta:
#             lon_center_deg = float(hmi_map.meta["CRLN_OBS"])
#         else:
#             lon_center_deg = 0.0
#             print("WARNING: CRLN_OBS not found in HMI LOS header; using lon0=0 deg.")
#     else:
#         lon_center_deg = args.lon0

#     # 4. PFSS 解
#     pfss_out = compute_pfss_solution(br_map, rss=args.rss, nr=args.nr)

#     # 5. Zucca-style POS-averaged |B| map
#     r_centers, lat_centers, B_pos_center = compute_B_pos_map(
#         pfss_out,
#         lon_center_deg=lon_center_deg,
#         lon_half_width_deg=10.0,
#     )

#     # 6. CSV 出力
#     export_mf_2d_csv(
#         r_centers=r_centers,
#         lat_centers=lat_centers,
#         B_pos_center=B_pos_center,
#         px_per_rsun=args.px_per_rsun,
#         csv_path=args.csv,
#     )

#     # 7. プロット（plasma カラーマップ、軸は pixel 単位）
#     fig, ax = plot_B_2d_map(
#         r_centers,
#         lat_centers,
#         B_pos_center,
#         rss=args.rss,
#         lon_center_deg=lon_center_deg,
#         px_per_rsun=args.px_per_rsun,
#         vmin=args.vmin,
#         vmax=args.vmax,
#         cmap="plasma",
#     )

#     output_path = (
#         f"/mnt/d/wsl/home/kinno-7010/Research_data/PFSS/"
#         # f"magnetic_field_2D_map_rss={args.rss}_20220613_0300.png"
#         f"magnetic_field_2D_map_rss={args.rss}_20110922_0900.png"
#     )
#     fig.savefig(output_path, dpi=300, bbox_inches="tight")
#     print(f"Saved PNG: {output_path}")

#     plt.show()

def main():
    # ---- Paths (edit to your environment) ----
    # pB / density-map side (same as you used for 2D density)
    filename_mk4 = r"/mnt/d/wsl/home/kinno-7010/Research_data/MK4_coronagraph/MK4_coronagraph_KCOR/pB/20220613_025810_kcor_l2.fts"
    filename_lasco = r"/mnt/d/wsl/home/kinno-7010/Research_data/SOHO/pB/C2-PB-20220613_0258.fts"

    # PFSS / HMI input
    hmi_file = r"/mnt/d/wsl/home/kinno-7010/Research_data/SDO/HMI/Rawdata/hmi.M_720s.20110922_090000_TAI.fits"

    # Same annulus as density figure
    r_ranges = {"mk4_inner": 1.0, "mk4_outer_lasco_inner": 2.2, "lasco_outer": 7.0}

    # ---- Step 1: replicate LASCO grid (same as density) ----
    final_image, r_map, theta_map, params_lasco = make_lasco_grid_and_final_image(
        filename_lasco, filename_mk4, r_ranges
    )

    # ---- Step 2: PFSS solution from HMI ----
    # Use your helper to prepare HMI and compute PFSS
    # ---- Step 2: PFSS solution from HMI ----
    # Use your helper to prepare HMI and compute PFSS
    hmi_data = prepare_hmi_for_pfss(hmi_file)

    # HMI フルディスク SunPy Map（PFSS の下端境界）
    hmi_map = hmi_data["full_map"]          # ← 追加

    # PFSS グリッド設定（Zucca+2014 相当：rss ~ 2.5 R⊙、nrho は 50 程度）
    nrho = 50                               # ← 追加
    rss = 2.5

    # pfsspy.Output を直接受け取る（辞書アクセスしない）
    pfss_solution = compute_pfss_solution(hmi_map, nrho=nrho, rss=rss)

    r_ax, _, _ = resolve_pfss_axes(pfss_solution, fallback={"rss": rss, "nrho": nrho})
    print(f"PFSS ready: rss={rss} Rs, nrho={nrho}, grid r∈[{np.nanmin(r_ax):.2f},{np.nanmax(r_ax):.2f}] R⊙")

    # ---- Step 3: sample PFSS on the POS grid (same pixels) ----
    # For a limb event use pos_longitude_deg ~ 90. For disk center, ~0 deg.
    B_map = sample_pfss_on_pos_grid(
        pfss_solution,
        r_map_rsun=r_map,
        theta_map=theta_map,
        pos_longitude_deg=90.0,   # East/West limb POS slice
        component="Babs",         # "Br" or "Babs"
        b0_deg=float(hmi_data["full_map"].meta.get("CRLT_OBS", 0.0)),  # solar B0 tilt
        pfss_fallback={"rss": rss, "nrho": nrho},
    )

    # Mask outside the valid annulus (exactly same as density map used)
    B_map = np.where((r_map >= r_ranges["mk4_inner"]) & (r_map <= r_ranges["lasco_outer"]), B_map, np.nan)
    
        # Output directory (adjust if needed)
    out_dir = r"/mnt/d/wsl/home/kinno-7010/Research_data/PFSS"
    # out_png = os.path.join(out_dir, f"2D_magnetic_field_map_rss={rss}.png")
    # out_csv = os.path.join(out_dir, f"2D_magnetic_field_map_rss={rss}.csv")
    out_png = os.path.join(out_dir, f"2D_magnetic_field_map_rss={rss}_20110922_0900.png")
    out_csv = os.path.join(out_dir, f"2D_magnetic_field_map_rss={rss}_20110922_0900.csv")

    # ---- Step 4: plot ----
    plot_B_2d_map(
        B_map,
        r_map,
        params_lasco,
        r_ranges,
        # title="2D Magnetic Field Map from PFSS (POS, $R_{\\mathrm{ss}}$="+f"{rss}"+" $R_\\odot$)",
        title="2D Magnetic Field Map from PFSS (POS, $R_{\\mathrm{ss}}$="+f"{rss}"+" $R_\\odot$)\n2011-09-22 09:00:00 UT",
        out_png=out_png,
        # xlim_pix=(-150, 0),
        # ylim_pix=(-100, 150),
        vmin=0.01,
        vmax=10.0,
    )

    # ---- Step 5: optional CSV ----
    export_B_csv(out_csv, B_map, r_map, theta_map, params_lasco)


if __name__ == "__main__":
    main()
