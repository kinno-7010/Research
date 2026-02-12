#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_overlay_tomography_gcs_from_fits.py

Option A: Run tomography directly from pB FITS each time (no NPZ caching),
then overlay tomography isodensity (iso-frequency) surfaces with a GCS shell.

Key points
----------
- Tomography uses the user's regularized solver implemented in:
    main_regularized_tomography.py
  which reconstructs ne(r,theta,phi) in a Carrington Cartesian frame.

- GCS geometry is generated using the user's gcs_overlay package:
    gcs_overlay/gcs_geometry.py  (PyThea-based wireframe curves)
  and is then transformed from Heliographic Stonyhurst -> Heliographic Carrington
  before overlaying, so that the coordinate frames match.

- No argparse is used. Edit the __main__ block to set:
    * PB_FITS list
    * tomography grid/solver parameters
    * GCS 6 parameters
    * ISO frequencies (MHz) and harmonic (1 or 2)

Dependencies (your environment)
------------------------------
numpy, pyvista
astropy + sunpy (needed for coordinate transforms and FITS handling by tomography + GCS)

Notes
-----
This script intentionally recomputes tomography every run, as requested.
For exploration, consider reducing NR/NTH/NPH and OUT_N to speed up.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import pyvista as pv
except Exception as e:
    raise SystemExit("pyvista is required: pip install pyvista") from e


# -----------------------------------------------------------------------------
# Import user's tomography + GCS helpers
# -----------------------------------------------------------------------------
def _ensure_on_path(p: Path) -> None:
    p = p.resolve()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _import_user_modules(tomo_dir: Path, gcs_pkg_dir: Path):
    """
    Ensure tomo_dir and gcs_pkg_dir parent are on sys.path, then import.
    gcs_pkg_dir must be the directory containing __init__.py (package root).
    """
    _ensure_on_path(tomo_dir)
    _ensure_on_path(gcs_pkg_dir.parent)

    # Tomography
    from main_regularized_tomography import (  # type: ignore
        SphericalGrid,
        RegularizedTomography,
        build_observation,
        build_rays_for_observation,
        ybk_profile_fft,
        ne_cm3_from_fp_mhz,
    )

    # GCS package (user-provided)
    from gcs_overlay.gcs_overlay import GCSParams  # type: ignore
    from gcs_overlay.gcs_geometry import sample_gcs_wireframe_points  # type: ignore

    return (
        SphericalGrid,
        RegularizedTomography,
        build_observation,
        build_rays_for_observation,
        ybk_profile_fft,
        ne_cm3_from_fp_mhz,
        GCSParams,
        sample_gcs_wireframe_points,
    )
    
def sun_to_observer_unit_vector(obs0) -> np.ndarray:
    """
    Return unit vector pointing from Sun center to the observer (Sun->Earth) in the
    same Carrington Cartesian basis as the tomography volume.

    Priority:
      1) obs0.lonlat_deg  (Carrington lon/lat of observer)
      2) -obs0.cam_z      (since obs0.cam_z is observer->Sun in your implementation)
    """
    v = None
    if hasattr(obs0, "lonlat_deg") and obs0.lonlat_deg is not None:
        lon_deg, lat_deg = obs0.lonlat_deg
        lon = np.deg2rad(float(lon_deg))
        lat = np.deg2rad(float(lat_deg))
        v = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=float)

    if v is None and hasattr(obs0, "cam_z"):
        v = -np.asarray(obs0.cam_z, dtype=float)

    if v is None:
        raise ValueError("Cannot determine Sun->observer direction (need obs0.lonlat_deg or obs0.cam_z).")

    n = np.linalg.norm(v)
    if (not np.isfinite(n)) or n == 0:
        raise ValueError("Invalid Sun->observer direction vector.")
    return v / n


# -----------------------------------------------------------------------------
# Coordinate transform: HGS (Stonyhurst) -> HGC (Carrington)
# -----------------------------------------------------------------------------
def transform_points_hgs_to_hgc(
    pts_xyz_rsun: np.ndarray,
    obstime_iso: str,
    observer: str = "earth",
) -> np.ndarray:
    """
    Convert Cartesian points from Heliographic Stonyhurst (HGS) to Heliographic Carrington (HGC).

    Tomography grid: Carrington-rotating Cartesian basis (phi = Carrington longitude).
    GCS wireframe: typically produced/interpreted in HGS.
    => Must convert HGS -> HGC before overlay, otherwise the GCS appears rotated.

    SunPy version note:
      - HeliographicStonyhurst does NOT accept 'observer' in recent SunPy (your error).
      - HeliographicCarrington may accept/require observer depending on version.
        We try with observer and fall back if not supported.
    """
    pts = np.asarray(pts_xyz_rsun, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("pts_xyz_rsun must have shape (N, 3) in units of Rsun.")

    import astropy.units as u
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, CartesianRepresentation
    from sunpy.coordinates import frames

    t = Time(obstime_iso)

    rep = CartesianRepresentation(
        x=pts[:, 0] * u.R_sun,
        y=pts[:, 1] * u.R_sun,
        z=pts[:, 2] * u.R_sun,
    )

    # HGS: no observer keyword in your SunPy
    c_hgs = SkyCoord(rep, frame=frames.HeliographicStonyhurst(obstime=t))

    # HGC: observer may/may not be accepted depending on SunPy version
    try:
        target = frames.HeliographicCarrington(obstime=t, observer=observer)
    except TypeError:
        target = frames.HeliographicCarrington(obstime=t)

    c_hgc = c_hgs.transform_to(target)
    xyz = c_hgc.cartesian.xyz.to_value(u.R_sun).T  # (N,3)
    return xyz


# -----------------------------------------------------------------------------
# Tomography (from pB FITS) -> ne volume
# -----------------------------------------------------------------------------
def reconstruct_ne_from_pb_fits(
    pb_fits_list: List[str],
    *,
    # Grid
    r_min: float,
    r_max: float,
    nr: int,
    nth: int,
    nph: int,
    # Ray sampling
    ds_rsun: float,
    # Preprocess / mask
    out_n: int,
    r_use_min: float,
    r_use_max: float,
    limb_u: float,
    pb_floor: float | str,
    apply_spatial_despike: bool,
    despike_nsig: float,
    despike_med: int,
    filt: bool,
    dpa_deg: float,
    hm: int,
    width_pix: int,
    q_low: float,
    # Inversion
    lam: float,
    maxiter: int,
    tol: float,
    positivity: bool = True,
    # Optional radial regularization weight (same as tomography main)
    use_wt_nr: bool = False,
    # Observer lon/lat override (rare; normally use FITS header)
    lonlat_default: Optional[Tuple[float, float]] = None,
    lonlat_override_map: Optional[dict] = None,  # {fits_path: (lon,lat)}
):
    """
    Run the same pipeline as main_regularized_tomography.py but from a list of FITS paths,
    returning (grid, ne_scaled, obs_list0, info).
    """
    # Import user's tomography objects at runtime (after sys.path set)
    # (Imported by caller via _import_user_modules.)
    raise RuntimeError("This function is patched-in at runtime by main().")


# -----------------------------------------------------------------------------
# PyVista helpers
# -----------------------------------------------------------------------------
def _polyline_from_points(points: np.ndarray) -> pv.PolyData:
    """Create a single polyline connecting points in order."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 2:
        return pv.PolyData(points)
    # VTK polyline cell encoding: [n, 0,1,2,...,n-1]
    cells = np.empty(n + 1, dtype=np.int64)
    cells[0] = n
    cells[1:] = np.arange(n, dtype=np.int64)
    poly = pv.PolyData(points)
    poly.lines = cells
    return poly


def _grid_shape(grid):
    """Return (nr, nth, nph) for a SphericalGrid across version differences."""
    if hasattr(grid, "n"):
        try:
            n = getattr(grid, "n")
            if isinstance(n, (tuple, list)) and len(n) == 3:
                return int(n[0]), int(n[1]), int(n[2])
        except Exception:
            pass
    # Common attribute names in your SphericalGrid implementation
    if all(hasattr(grid, a) for a in ("nr", "nth", "nph")):
        return int(grid.nr), int(grid.nth), int(grid.nph)
    # Deduce from voxel centers as a robust fallback
    rr, tt, pp = grid.voxel_centers_sph()
    return int(rr.shape[0]), int(rr.shape[1]), int(rr.shape[2])


def build_tomography_structured_grid(grid, ne_1d: np.ndarray) -> pv.StructuredGrid:
    """
    Build a PyVista StructuredGrid from tomography spherical grid + ne vector,
    matching the ordering used in visualize_isosurface() in main_regularized_tomography.py.
    """
    nr, nth, nph = _grid_shape(grid)
    rr, tt, pp = grid.voxel_centers_sph()  # type: ignore[attr-defined]

    ne3 = ne_1d.reshape((nr, nth, nph), order="C")

    # Close periodic boundary in phi to reduce seam artifacts
    pp2 = np.concatenate([pp, pp[:, :, :1] + 2.0 * np.pi], axis=2)
    rr2 = np.concatenate([rr, rr[:, :, :1]], axis=2)
    tt2 = np.concatenate([tt, tt[:, :, :1]], axis=2)
    ne2 = np.concatenate([ne3, ne3[:, :, :1]], axis=2)

    xx = rr2 * np.sin(tt2) * np.cos(pp2)
    yy = rr2 * np.sin(tt2) * np.sin(pp2)
    zz = rr2 * np.cos(tt2)

    sg = pv.StructuredGrid(xx, yy, zz)
    sg["ne"] = ne2.ravel(order="F")
    return sg


def add_isosurfaces(plotter, sg, iso_freqs_mhz, harmonic=2, opacity=0.35, colors=None):
    """
    StructuredGrid sg（'ne' スカラー）に対し、iso_freqs_mhz（MHz）の等周波数面を重ねる。
    main_regularized_tomography.py の表示ロジックに寄せて、
    - 再構成 f-range を計算して表示
    - 範囲外周波数はクリップ
    - 何も出なければ明示メッセージ
    を行う。
    """
    import numpy as np
    import pyvista as pv

    if colors is None:
        colors = ["red", "cyan", "gold", "magenta", "orange"]

    # --- helpers (依存を増やさず式で実装) ---
    def ne_cm3_from_fp_mhz_local(f_mhz, H):
        # f_emit = H * 8980 * sqrt(ne) [Hz]
        return (float(f_mhz) * 1e6 / (8980.0 * float(H))) ** 2

    def fp_mhz_from_ne_cm3_local(ne_cm3, H):
        ne_cm3 = np.asarray(ne_cm3, dtype=float)
        return (float(H) * 8980.0 * np.sqrt(ne_cm3)) / 1e6

    if "ne" not in sg.array_names:
        raise ValueError("StructuredGrid sg must have scalar 'ne'.")

    sg.set_active_scalars("ne")
    ne_all = np.asarray(sg["ne"], dtype=float)
    ne_pos = ne_all[np.isfinite(ne_all) & (ne_all > 0)]

    if ne_pos.size == 0:
        plotter.add_text("No positive density in reconstruction.", position="upper_left", font_size=12, color="black")
        print("No isosurface rendered.\nCheck tomography reconstruction values (all non-positive).")
        return

    fmin = float(fp_mhz_from_ne_cm3_local(ne_pos.min(), harmonic))
    fmax = float(fp_mhz_from_ne_cm3_local(ne_pos.max(), harmonic))

    harm_label = "Second Harmonic" if harmonic == 2 else "Fundamental"
    # plotter.add_text(
    #     f"Reconstructed f-range: {fmin:.2f} .. {fmax:.2f} MHz ({harm_label})",
    #     position="lower_left",
    #     font_size=12,
    #     color="black",
    # )

    rendered = 0
    for i, f_req in enumerate(iso_freqs_mhz):
        f_req = float(f_req)

        # 端ぴったりは等値面が消えることがあるので、ほんの少し内側へ寄せる
        f_use = f_req
        if f_use < fmin:
            f_use = fmin * 1.001
        if f_use > fmax:
            f_use = fmax * 0.999

        if abs(f_use - f_req) / max(f_req, 1e-9) > 1e-6:
            print(f"[WARN] Requested {f_req:.2f} MHz is outside reconstruction range; using {f_use:.2f} MHz instead.")

        ne_iso = ne_cm3_from_fp_mhz_local(f_use, harmonic)
        surf = sg.contour(isosurfaces=[ne_iso], scalars="ne")

        if surf.n_points == 0:
            continue

        plotter.add_mesh(
            surf,
            color=colors[i % len(colors)],
            opacity=float(opacity),
            smooth_shading=True,
        )
        rendered += 1

    if rendered == 0:
        print("No isosurface rendered.\nCheck iso_freqs_mhz or tomography reconstruction range.")
        plotter.add_text(
            "No isosurface rendered.\nCheck iso_freqs_mhz or tomography reconstruction range.",
            position="upper_left",
            font_size=14,
            color="black",
        )

def add_sun_earth_line(
    plotter,
    obs0,
    *,
    length_rsun: float = 15.0,
    start_rsun: float = 1.0,     # ★太陽表面から開始（Rsun=1）
    color: str = "orange",
    line_width: int = 5,
):
    """
    Sun–Earth line drawn from the solar surface (not from the center).
    """
    import pyvista as pv

    x_hat = sun_to_observer_unit_vector(obs0)  # Sun->Earth
    p0 = x_hat * float(start_rsun)
    p1 = x_hat * (float(start_rsun) + float(length_rsun))
    plotter.add_mesh(pv.Line(p0, p1), color=color, line_width=int(line_width))

def add_physical_axes_triad(
    plotter,
    obs0,
    *,
    origin_rsun: float = 1.0,   # sub-Earth point on the surface
    axis_len: float = 2.0,
    line_width: int = 6,
    color_x: str = "crimson",
    color_y: str = "seagreen",
    color_z: str = "royalblue",
):
    """
    Draw a labeled triad:
      X: Sun->Earth line
      Z: Solar north (+Z of the tomography Cartesian)
      Y: 'East/right' direction defined as Z x X (right direction when up=North)

    Note:
      If you want the *physical* heliographic East instead of "right-on-screen",
      flip Y by multiplying by -1.
    """
    import numpy as np
    import pyvista as pv

    x_hat = sun_to_observer_unit_vector(obs0)
    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)

    y_hat = np.cross(z_hat, x_hat)
    yn = np.linalg.norm(y_hat)
    if (not np.isfinite(yn)) or yn == 0:
        # pathological case: x parallel z
        y_hat = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        y_hat = y_hat / yn

    o = x_hat * float(origin_rsun)

    def _arrow(dir_hat, color):
        # Arrow from o to o + dir_hat*axis_len
        arr = pv.Arrow(start=o, direction=dir_hat, scale=float(axis_len))
        plotter.add_mesh(arr, color=color)

    _arrow(x_hat, color_x)
    _arrow(y_hat, color_y)
    _arrow(z_hat, color_z)

    # Labels near tips
    tips = np.vstack([o + x_hat * axis_len, o + y_hat * axis_len, o + z_hat * axis_len])
    labels = ["X (Sun–Earth)", "Y (East / right)", "Z (North)"]
    plotter.add_point_labels(
        tips,
        labels,
        point_size=0,
        font_size=14,
        text_color="black",
        always_visible=True,
        shape=None,
    )

def add_solar_latlon_grid(
    plotter,
    *,
    radius: float = 1.002,     # slightly above photosphere for visibility
    dlon_deg: float = 30.0,
    dlat_deg: float = 30.0,
    n_lon_samples: int = 361,
    n_lat_samples: int = 181,
    color: str = "lightgray",
    line_width: int = 2,
    opacity: float = 0.6,
):
    """
    Draw Carrington-style lat/lon grid lines on the solar sphere in the same
    Cartesian basis as the tomography volume.
    """
    import numpy as np

    # Latitude lines: lat fixed, lon sweeps 0..360
    lons = np.deg2rad(np.linspace(0.0, 360.0, n_lon_samples))
    for lat_deg in np.arange(-90.0 + dlat_deg, 90.0, dlat_deg):
        lat = np.deg2rad(lat_deg)
        # colatitude theta = 90 - lat
        th = (np.pi / 2.0) - lat
        rr = float(radius)

        x = rr * np.sin(th) * np.cos(lons)
        y = rr * np.sin(th) * np.sin(lons)
        z = rr * np.cos(th) * np.ones_like(lons)

        pts = np.column_stack([x, y, z])
        plotter.add_mesh(_polyline_from_points(pts), color=color, line_width=int(line_width), opacity=float(opacity))

    # Longitude lines: lon fixed, lat sweeps -90..+90
    lats = np.deg2rad(np.linspace(-90.0, 90.0, n_lat_samples))
    for lon_deg in np.arange(0.0, 360.0, dlon_deg):
        lon = np.deg2rad(lon_deg)
        th = (np.pi / 2.0) - lats
        rr = float(radius)

        x = rr * np.sin(th) * np.cos(lon)
        y = rr * np.sin(th) * np.sin(lon)
        z = rr * np.cos(th)

        pts = np.column_stack([x, y, z])
        plotter.add_mesh(_polyline_from_points(pts), color=color, line_width=int(line_width), opacity=float(opacity))

def add_runinfo_legend(
    plotter,
    *,
    obstime_iso: str,
    iso_freqs_mhz: List[float],
    harmonic: int,
    gcs_params: dict,
    position: str = "upper_right",
    font_size: int = 12,
):
    """
    Add a compact run-information legend inside the figure.
    """
    freqs = ", ".join([f"{float(f):.1f}" for f in iso_freqs_mhz])
    txt = (
        f"Time: {obstime_iso}\n"
        f"Iso-freq: [{freqs}] MHz (H={int(harmonic)})\n"
        f"GCS: h={gcs_params['h_apex']:.2f} Rsun, kappa={gcs_params['kappa']:.3f}\n"
        f"     alpha={gcs_params['alpha_deg']:.1f} deg, tilt={gcs_params['tilt_deg']:.1f} deg\n"
        f"     lon={gcs_params['lon_deg']:.1f} deg, lat={gcs_params['lat_deg']:.1f} deg"
    )
    plotter.add_text(txt, position=position, font_size=int(font_size), color="black")


def build_gcs_surface_from_parallels(parallels: List[np.ndarray]) -> Optional[pv.PolyData]:
    """
    Create an approximate GCS shell surface by lofting between 'parallels'
    (curves that share the same number of nodes along the skeleton).
    """
    if not parallels:
        return None
    # Ensure consistent lengths
    n0 = parallels[0].shape[0]
    if n0 < 4:
        return None
    for arr in parallels:
        if arr.shape[0] != n0:
            # Cannot loft if lengths differ
            return None

    # Close in phi by duplicating the first ring at the end
    rings = [np.asarray(p, dtype=np.float64) for p in parallels]
    rings.append(rings[0].copy())

    # Stack: (n_nodes, n_rings, 3)
    arr = np.stack(rings, axis=1)
    X = arr[:, :, 0]
    Y = arr[:, :, 1]
    Z = arr[:, :, 2]
    sgrid = pv.StructuredGrid(X, Y, Z)
    return sgrid.extract_surface().triangulate()


def set_camera_from_observation(
    plotter: pv.Plotter,
    obs0,
    distance_rsun: float = 15.0,
):
    """
    Match the camera convention in main_regularized_tomography.visualize_isosurface().

    IMPORTANT:
      In your tomography code, obs.cam_z was constructed as cam_z = -obs_vec,
      where obs_vec points Sun->observer.
      => cam_z points observer->Sun.
      Using cam_z directly as a camera position can flip the viewpoint.

    Therefore we use obs0.lonlat_deg (Carrington lon/lat of the observer) to place the camera
    on the Sun->observer line, same as visualize_isosurface().
    """
    import numpy as np

    if hasattr(obs0, "lonlat_deg") and obs0.lonlat_deg is not None:
        lon_deg, lat_deg = obs0.lonlat_deg
    else:
        # Fallback: Earth-view approximation
        lon_deg, lat_deg = 0.0, 0.0

    lon = np.deg2rad(float(lon_deg))
    lat = np.deg2rad(float(lat_deg))

    cam_dir = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    cam_pos = cam_dir * float(distance_rsun)

    plotter.camera_position = [cam_pos, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

def apply_brightness_scale_like_main(tomo, ne_raw, y_obs):
    """
    main_regularized_tomography.py と同じ考え方で、ne_raw に最小二乗スケールを掛けて
    観測 y_obs の絶対値（pB）に整合させる。

    Parameters
    ----------
    tomo : RegularizedTomography
        A_times(x) と W（連結済み重み）を持つ想定
    ne_raw : (Nvox,) ndarray
        正則化付き反転の生解（スケールが縮んでいることがある）
    y_obs : array-like or list[array-like]
        観測ベクトル（各観測の連結、または list で与えてもよい）

    Returns
    -------
    ne_scaled : (Nvox,) ndarray
    scale : float
    """
    import numpy as np

    if isinstance(y_obs, (list, tuple)):
        y_obs_vec = np.concatenate([np.asarray(v).ravel() for v in y_obs])
    else:
        y_obs_vec = np.asarray(y_obs).ravel()

    y_pred = np.asarray(tomo.A_times(ne_raw)).ravel()
    w = np.asarray(tomo.W).ravel()

    m = np.isfinite(y_obs_vec) & np.isfinite(y_pred) & np.isfinite(w)
    if not np.any(m):
        return ne_raw, 1.0

    w2 = w[m] ** 2
    denom = np.sum(w2 * y_pred[m] ** 2)
    if denom <= 0 or (not np.isfinite(denom)):
        return ne_raw, 1.0

    scale = np.sum(w2 * y_pred[m] * y_obs_vec[m]) / denom

    # 物理的に負スケールは不適切なのでフォールバック
    if (not np.isfinite(scale)) or (scale <= 0):
        scale = 1.0

    return ne_raw * scale, float(scale)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    # ---- Import modules ----
    TOMO_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research/Tomography/py_folder")
    GCS_PKG_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research/GCS/gcs_overlay")
    
    PB_FITS_LIST = [
        "/mnt/d/wsl/home/kinno-7010/Research/Tomography/Rawdata/pB_Kcor_LASCO_axi_20220613_0300.fits",
        "/mnt/d/wsl/home/kinno-7010/Research/Tomography/Rawdata/COR1A_pb_pre_20220613_030100.fits"
    ]

    # 2) Time used for GCS orientation and frame transform
    OBSTIME_ISO = "2022-06-13T03:25:29"  # example; set to your fitted time

    # 3) Tomography grid + inversion parameters
    R_MIN, R_MAX = 1.5, 5.0
    NR, NTH, NPH = 80, 120, 180       # increase for fidelity; decrease for speed
    DS = 0.02                    # LOS step (Rsun)
    LAM = 1e-2                        # Tikhonov weight
    MAXITER, TOL = 60, 1e-4
    WT_R = None                 # True to enable radial weighting like in main

    # 4) Preprocess / mask parameters (same meaning as tomography main)
    OUT_N = 128                       # rebin to OUT_N x OUT_N for inversion
    R_USE_MIN, R_USE_MAX = 1.6, 5.5   # use pixels in this projected r-range (Rsun)
    LIMB_U = 0.63                     # limb darkening parameter for Thomson kernel
    PB_FLOOR = "auto"                 # or float; e.g. 1e-13
    APPLY_SPATIAL_DESPIKE = True
    DESPIKE_NSIG, DESPIKE_MED = 6.0, 3
    FILT = False
    DPA_DEG = 3.0
    HM = 3
    WIDTH_PIX = 0
    Q_LOW = 10.0
    

    # 5) Iso-frequency settings (MHz) and harmonic (1=fundamental, 2=harmonic)
    ISO_FREQ_MHZ = [34.0]
    HARMONIC = 2

    # 6) GCS parameters (6 parameters)
    #    (Units follow your gcs_geometry: h_apex [Rsun], kappa [0-1), alpha/tilt/lon/lat [deg])
    GCS_H_APEX = 3.39
    GCS_KAPPA = 0.12
    GCS_ALPHA_DEG = 20.0
    GCS_TILT_DEG = -85.0
    GCS_LON_DEG = -44.0
    GCS_LAT_DEG = 10.0
    LONLAT_OVERRIDE_MAP = None
    LONLAT_DEFAULT = None
    GCS_OBSERVER = "earth"

    # 7) Rendering
    SHOW_SUN = True
    SHOW_GUI = True
    SAVE_PNG = True
    PNG_PATH = Path(f"/mnt/d/wsl/home/kinno-7010/Research/Tomography/output/overlay_tomo_gcs_{''.join(str(f) for f in ISO_FREQ_MHZ)}MHz.png")

    (
        SphericalGrid,
        RegularizedTomography,
        build_observation,
        build_rays_for_observation,
        ybk_profile_fft,
        ne_cm3_from_fp_mhz,
        GCSParams,
        sample_gcs_wireframe_points,
    ) = _import_user_modules(TOMO_DIR, GCS_PKG_DIR)

    # ---- Build grid ----
    r_edges = np.linspace(R_MIN, R_MAX, NR + 1)
    th_edges = np.linspace(0.0, np.pi, NTH + 1)
    ph_edges = np.linspace(0.0, 2.0 * np.pi, NPH + 1)
    grid = SphericalGrid(r_edges=r_edges, th_edges=th_edges, ph_edges=ph_edges)

    # ---- Load observations ----
    obs_list = []
    rays_list = []
    y_list = []

    for fp in PB_FITS_LIST:
        print(f"Reading {fp}...")
        obs = build_observation(
            Path(fp),
            out_n=OUT_N,
            r_use_min=R_USE_MIN,
            r_use_max=R_USE_MAX,
            limb_u=LIMB_U,
            apply_spatial_despike=APPLY_SPATIAL_DESPIKE,
            filt=FILT,
            despike_nsig=DESPIKE_NSIG,
            despike_med=DESPIKE_MED,
            pb_floor=PB_FLOOR,
            dpa_deg=DPA_DEG,
            hm=HM,
            width_pix=WIDTH_PIX,
            q_low=Q_LOW,
            lonlat_override=(LONLAT_OVERRIDE_MAP.get(Path(fp).name) if LONLAT_OVERRIDE_MAP else None),
            lonlat_default=LONLAT_DEFAULT,
        )

        # main_regularized_tomography.py と同じ：観測ベクトルは「pbそのもの」
        y_vec = obs.pb.ravel()[obs.idx_map].astype(np.float64)
        y_list.append(y_vec)

        # rays
        rays = build_rays_for_observation(
            obs=obs,
            grid=grid,
            ds_rsun=DS,
            r_min=R_MIN,
            r_max=R_MAX,
            limb_u=LIMB_U,
        )
        rays_list.append(rays)

        obs_list.append(obs)

    y_obs = np.concatenate(y_list) if y_list else np.array([], dtype=np.float64)


    # ---- Solve tomography ----
    tomo = RegularizedTomography(grid, obs_list, rays_list, lam=LAM, wt_r=WT_R)

    # ★ ここが修正点：solve は y_obs だけを取る
    ne_raw, info = tomo.solve(y_obs, maxiter=MAXITER, tol=TOL, positivity=True)

    # 後段スケール（あなたが追加したやつ）：y_obs を渡す
    ne_raw, scale = apply_brightness_scale_like_main(tomo, ne_raw, y_obs)
    print(f"[INFO] brightness scale applied: {scale:.3e}")

    if info != 0:
        print(f"[WARN] Tomography CG did not fully converge (info={info}).")

    ne = ne_raw.reshape((NR, NTH, NPH))




    if info != 0:
        print(f"[WARN] Tomography CG did not fully converge (info={info}). Consider stronger regularization or more images.")

    ne = ne_raw.reshape((NR, NTH, NPH))

    # ---- Iso-density surfaces (from iso-frequency) ----
    ne_iso_list = [float(ne_cm3_from_fp_mhz(f, harmonic=HARMONIC)) for f in ISO_FREQ_MHZ]

    # ---- Build PyVista grid ----
    sg = build_tomography_structured_grid(grid, ne)

    # ---- Plot ----
    p = pv.Plotter(off_screen=(not SHOW_GUI))
    p.set_background("white")

    obs0 = obs_list[0]  # ★先に定義

    if SHOW_SUN:
        p.add_mesh(
            pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60),
            opacity=0.15,
            color="grey",
        )

    # isosurface は「周波数リスト」を渡す（あなたの add_isosurfaces の定義に一致）
    add_isosurfaces(p, sg, ISO_FREQ_MHZ, harmonic=HARMONIC, opacity=0.35)
    
    # 太陽の緯度経度グリッド
    add_solar_latlon_grid(p, radius=1.002, dlon_deg=30.0, dlat_deg=30.0, line_width=2, opacity=0.6)

    # Sun–Earth line（太陽表面から）
    add_sun_earth_line(p, obs0, length_rsun=12.0, start_rsun=1.0, color="orange", line_width=5)

    # 物理方向の座標軸トライアド（X=Sun–Earth, Y=右方向, Z=北）
    add_physical_axes_triad(p, obs0, origin_rsun=1.0, axis_len=2.5)

    # 図中Legend（指定周波数・GCS・時刻）
    add_runinfo_legend(
        p,
        obstime_iso=OBSTIME_ISO,
        iso_freqs_mhz=ISO_FREQ_MHZ,
        harmonic=HARMONIC,
        gcs_params=dict(
            h_apex=float(GCS_H_APEX),
            kappa=float(GCS_KAPPA),
            alpha_deg=float(GCS_ALPHA_DEG),
            tilt_deg=float(GCS_TILT_DEG),
            lon_deg=float(GCS_LON_DEG),
            lat_deg=float(GCS_LAT_DEG),
        ),
        position="upper_right",
        font_size=12,
    )


    # ---- GCS wireframe ----
    params = GCSParams(
        h_apex=float(GCS_H_APEX),
        kappa=float(GCS_KAPPA),
        alpha_deg=float(GCS_ALPHA_DEG),
        lon_deg=float(GCS_LON_DEG),
        lat_deg=float(GCS_LAT_DEG),
        tilt_deg=float(GCS_TILT_DEG),
    )

    curves = sample_gcs_wireframe_points(params, obstime=OBSTIME_ISO)

    for key in ("parallels", "meridians", "legs"):
        for pts in curves.get(key, []):
            pts_hgc = transform_points_hgs_to_hgc(
                np.asarray(pts),
                obstime_iso=OBSTIME_ISO,
                observer=GCS_OBSERVER,
            )
            p.add_mesh(_polyline_from_points(pts_hgc), color="lime", line_width=2)

    # カメラ
    set_camera_from_observation(p, obs0, distance_rsun=15.0)

    # 出力
    if SAVE_PNG:
        PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
        p.show(screenshot=str(PNG_PATH))
        print(f"[OK] Saved: {PNG_PATH}")
    else:
        p.show()


if __name__ == "__main__":
    main()
