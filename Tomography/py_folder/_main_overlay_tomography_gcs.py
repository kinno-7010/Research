#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay tomography iso-frequency surfaces with a GCS model in 3D (PyVista).

What this script does
---------------------
1) Loads your tomography output (.npz) containing a spherical grid and n_e.
2) Converts an observed radio frequency f to the corresponding electron density
   (assuming plasma emission at fundamental or harmonic).
3) Extracts an iso-density surface (iso-frequency surface) from the 3D tomography cube.
4) Overlays a GCS model (wireframe, and optionally a simple surface patch from the GCS "parallels")
   in the same coordinate frame.

Important implementation notes
------------------------------
- The "spiky" artifacts you saw are a classic symptom of **scalar ordering mismatch** and/or an
  **open periodic seam** in phi. This script reproduces the same reshape+flatten logic used in
  your tomography-only plot:
      ne3 = ne.reshape((nr,nth,nph), order="C")
      scalars_for_pyvista = ne3.ravel(order="F")
  and optionally closes the phi seam by duplicating the first phi slice at 2π.

- The "GCS looks like it points in a strange direction" was primarily because the camera was not
  set to an Earth-view geometry. This script can set an Earth-view camera automatically.

References (physics)
--------------------
- Plasma frequency relation used for the iso-frequency surface:
  Cunha-Silva et al., 2015, doi: 10.1051/0004-6361/201425388
- Thomson-scattering basis of pB inversions (context for your tomography inputs):
  Hayes et al., 2001, doi: 10.1086/319029
- GCS forward model:
  Thernisien et al., 2006, doi: 10.1086/508254
  Thernisien et al., 2009, doi: 10.1007/s11207-009-9346-5
  Thernisien, 2011, doi: 10.1088/0067-0049/194/2/33

(No argparse as requested: edit parameters in __main__.)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pyvista as pv


# -----------------------------------------------------------------------------
# sys.path helpers
# -----------------------------------------------------------------------------
def _add_sys_path(p: Path) -> None:
    p = Path(p).expanduser().resolve()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def import_user_modules(
    tomography_dir: Path,
    gcs_overlay_dir: Path,
    pythea_root: Optional[Path] = None,
):
    """
    Add your project directories to sys.path and import required user modules.
    """
    _add_sys_path(tomography_dir)
    _add_sys_path(gcs_overlay_dir)
    if pythea_root is not None:
        _add_sys_path(pythea_root)

    import main_regularized_tomography as tomo  # type: ignore
    from gcs_geometry import GCSParams, sample_gcs_wireframe_points  # type: ignore

    return {"tomo": tomo, "GCSParams": GCSParams, "sample_gcs_wireframe_points": sample_gcs_wireframe_points}


# -----------------------------------------------------------------------------
# Camera utilities
# -----------------------------------------------------------------------------
def set_camera_from_lonlat(plotter: pv.Plotter, lon_deg: float, lat_deg: float, distance_rsun: float = 15.0) -> None:
    """
    Place the camera at (lon,lat) with the given distance (in R_sun) in the same frame as plotted data.
    """
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    cam_dir = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=float)
    cam_pos = tuple((distance_rsun * cam_dir).tolist())
    plotter.camera_position = [cam_pos, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]


def default_earth_camera_lonlat(obstime: Optional[str], tomo_frame: str) -> Optional[Tuple[float, float]]:
    """
    Return an Earth-view (lon,lat) in the target frame.

    - Stonyhurst: Earth view is (0,0) by definition.
    - Carrington: Earth view is approximately (L0,B0) at obstime (requires sunpy).
    """
    tomo_frame = tomo_frame.lower().strip()
    if tomo_frame.startswith("stony"):
        return (0.0, 0.0)

    if not tomo_frame.startswith("carr") or obstime is None:
        return None

    try:
        import astropy.time
        import sunpy.coordinates.sun as sun
        t = astropy.time.Time(obstime)
        L0 = float(sun.L0(t).to_value("deg"))
        B0 = float(sun.B0(t).to_value("deg"))
        return (L0, B0)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Tomography grid -> PyVista StructuredGrid (ordering-critical)
# -----------------------------------------------------------------------------
def build_pyvista_structured_grid_from_tomography(grid, ne: np.ndarray, close_phi: bool = True) -> pv.StructuredGrid:
    """
    Convert your tomography spherical voxel-center grid + n_e into a PyVista StructuredGrid.

    This reproduces the same ordering you used in the tomography-only plot, and optionally closes
    the phi periodic seam to avoid needle-like artifacts on extracted iso-surfaces.
    """
    rr, tt, pp = grid.voxel_centers_sph()  # shapes (nr,nth,nph)
    nr, nth, nph = rr.shape

    ne_arr = np.asarray(ne)
    if ne_arr.ndim == 1:
        if ne_arr.size != nr * nth * nph:
            raise ValueError(f"ne.size={ne_arr.size} but grid expects {nr*nth*nph} points.")
        ne3 = ne_arr.reshape((nr, nth, nph), order="C")
    elif ne_arr.ndim == 3:
        if ne_arr.shape != (nr, nth, nph):
            raise ValueError(f"ne.shape={ne_arr.shape} but grid expects {(nr,nth,nph)}.")
        ne3 = ne_arr
    else:
        raise ValueError(f"Unsupported ne shape: {ne_arr.shape} (expect 1D or 3D).")

    if close_phi:
        rr2 = np.concatenate([rr, rr[..., :1]], axis=2)
        tt2 = np.concatenate([tt, tt[..., :1]], axis=2)
        pp2 = np.concatenate([pp, pp[..., :1] + 2.0 * np.pi], axis=2)
        ne2 = np.concatenate([ne3, ne3[..., :1]], axis=2)
    else:
        rr2, tt2, pp2, ne2 = rr, tt, pp, ne3

    # spherical -> cartesian (same convention as your tomography code)
    x = rr2 * np.cos(tt2) * np.cos(pp2)
    y = rr2 * np.cos(tt2) * np.sin(pp2)
    z = rr2 * np.sin(tt2)

    sg = pv.StructuredGrid(x, y, z)
    sg["ne_cm3"] = ne2.ravel(order="F")  # critical
    return sg


# -----------------------------------------------------------------------------
# Frame transform helper (SunPy)
# -----------------------------------------------------------------------------
def transform_points_sunpy(xyz_rsun: np.ndarray, obstime: str, src_frame: str, dst_frame: str) -> np.ndarray:
    """
    Transform Sun-centered cartesian points between heliographic frames using sunpy+astropy.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    import sunpy.coordinates.frames as frames

    src_frame = src_frame.lower().strip()
    dst_frame = dst_frame.lower().strip()

    if src_frame.startswith("stony"):
        f_src = frames.HeliographicStonyhurst(obstime=obstime)
    elif src_frame.startswith("carr"):
        f_src = frames.HeliographicCarrington(obstime=obstime)
    else:
        raise ValueError(f"Unsupported src_frame: {src_frame}")

    if dst_frame.startswith("stony"):
        f_dst = frames.HeliographicStonyhurst(obstime=obstime)
    elif dst_frame.startswith("carr"):
        f_dst = frames.HeliographicCarrington(obstime=obstime)
    else:
        raise ValueError(f"Unsupported dst_frame: {dst_frame}")

    c = SkyCoord(
        x=xyz_rsun[:, 0] * u.R_sun,
        y=xyz_rsun[:, 1] * u.R_sun,
        z=xyz_rsun[:, 2] * u.R_sun,
        frame=f_src,
        representation_type="cartesian",
    ).transform_to(f_dst)

    return np.column_stack([c.cartesian.x.to_value(u.R_sun), c.cartesian.y.to_value(u.R_sun), c.cartesian.z.to_value(u.R_sun)])


# -----------------------------------------------------------------------------
# GCS curves -> wireframe + optional surface patch
# -----------------------------------------------------------------------------
def gcs_curves(
    sample_gcs_wireframe_points,
    params,
    obstime: str,
    observer_hgs: str,
    n_parallels: int = 24,
    n_meridians: int = 32,
    n_leg_points: int = 220,
    n_front_points: int = 520,
) -> Dict[str, list]:
    """
    Sample GCS curves (parallels/meridians/legs) using your gcs_geometry.py helper.

    The returned dict contains lists of curves; each curve is an (N,3) array in R_sun.
    """
    return sample_gcs_wireframe_points(
        params,
        obstime=obstime,
        observer=observer_hgs,
        n_parallels=n_parallels,
        n_meridians=n_meridians,
        n_leg_points=n_leg_points,
        n_front_points=n_front_points,
    )


def curves_to_wireframe_multiblock(
    curves: Dict[str, list],
    obstime: str,
    src_frame: str,
    dst_frame: str,
) -> pv.MultiBlock:
    """
    Convert sampled curves into a MultiBlock of PolyData polylines.
    """
    blocks = pv.MultiBlock()
    bi = 0
    for group, group_curves in curves.items():
        for c in group_curves:
            pts = np.asarray(c, dtype=float)
            if dst_frame.lower().strip() != src_frame.lower().strip():
                pts = transform_points_sunpy(pts, obstime=obstime, src_frame=src_frame, dst_frame=dst_frame)

            poly = pv.PolyData(pts)
            n = pts.shape[0]
            poly.lines = np.hstack(([n], np.arange(n, dtype=np.int64)))
            blocks[bi] = poly
            blocks.set_block_name(bi, f"{group}_{bi}")
            bi += 1
    return blocks


def parallels_to_surface_polydata(
    parallels: list,
    obstime: str,
    src_frame: str,
    dst_frame: str,
    close_loop: bool = True,
) -> Optional[pv.PolyData]:
    """
    Build a simple triangulated surface patch by "lofting" between adjacent GCS parallels.

    This is a pragmatic way to render a *surface* without depending on any particular PyThea
    mesh API. It typically captures the front (toroidal) part of the GCS shell well.

    Limitations:
    - This does not explicitly reconstruct the legs as a watertight surface.
    - Surface quality depends on how the sampling function defines the parallels.

    Returns None if the input does not look like a consistent parametric grid.
    """
    if len(parallels) < 2:
        return None

    # Ensure consistent point counts
    m = np.asarray(parallels[0]).shape[0]
    if any(np.asarray(p).shape[0] != m for p in parallels):
        return None

    # Stack points into a (K, M, 3) array
    P = np.stack([np.asarray(p, dtype=float) for p in parallels], axis=0)  # (K,M,3)
    if dst_frame.lower().strip() != src_frame.lower().strip():
        # transform each parallel
        P2 = []
        for k in range(P.shape[0]):
            P2.append(transform_points_sunpy(P[k], obstime=obstime, src_frame=src_frame, dst_frame=dst_frame))
        P = np.stack(P2, axis=0)

    K, M, _ = P.shape
    pts = P.reshape((-1, 3))

    # Build triangle faces
    faces = []
    def vid(i, j):
        return i * M + j

    j_max = M if close_loop else (M - 1)
    for i in range(K - 1):
        for j in range(j_max):
            j2 = (j + 1) % M
            if (not close_loop) and (j == M - 1):
                continue

            a = vid(i, j)
            b = vid(i, j2)
            c = vid(i + 1, j2)
            d = vid(i + 1, j)

            # two triangles: (a,b,c) and (a,c,d)
            faces.append([3, a, b, c])
            faces.append([3, a, c, d])

    if not faces:
        return None

    faces_arr = np.array(faces, dtype=np.int64).ravel()
    surf = pv.PolyData(pts, faces_arr)
    surf.clean(inplace=True)
    return surf


# -----------------------------------------------------------------------------
# Main overlay routine
# -----------------------------------------------------------------------------
def overlay_isosurface_and_gcs(
    tomography_npz: Path,
    freq_mhz: float,
    harmonic: int,
    gcs_params,
    obstime: str,
    observer_hgs: str = "earth",
    tomo_frame: str = "carrington",
    gcs_src_frame: str = "stonyhurst",
    camera_lonlat_deg: Optional[Tuple[float, float]] = None,
    build_gcs_surface: bool = True,
    close_phi: bool = True,
    out_png: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Overlay one iso-frequency surface from tomography with a GCS wireframe (+ optional surface patch).
    """
    tomography_npz = Path(tomography_npz).expanduser().resolve()
    dat = np.load(tomography_npz, allow_pickle=True)

    r_edges = dat["r_edges"]
    theta_edges = dat["theta_edges"]
    phi_edges = dat["phi_edges"]
    ne = dat["ne"]

    # Plasma emission: f_obs = s * f_pe, where s=1 (F) or 2 (H).
    # f_pe [MHz] = 8.98e-3 * sqrt(n_e[cm^-3]) (Cunha-Silva et al., 2015, doi: 10.1051/0004-6361/201425388)
    f_pe_mhz = float(freq_mhz) / float(harmonic)
    ne_iso = (f_pe_mhz / 8.98e-3) ** 2

    # Use your SphericalGrid class to match voxel-center conventions
    from main_regularized_tomography import SphericalGrid  # type: ignore
    grid = SphericalGrid(r_edges=r_edges, theta_edges=theta_edges, phi_edges=phi_edges)

    pv_grid = build_pyvista_structured_grid_from_tomography(grid, ne=ne, close_phi=close_phi)
    pv_grid["ne_cm3"] = np.nan_to_num(pv_grid["ne_cm3"], nan=0.0, posinf=0.0, neginf=0.0)
    iso = pv_grid.contour(isosurfaces=[float(ne_iso)], scalars="ne_cm3")

    # GCS curves from your helper
    from gcs_geometry import sample_gcs_wireframe_points  # type: ignore
    curves = gcs_curves(sample_gcs_wireframe_points, gcs_params, obstime=obstime, observer_hgs=observer_hgs)

    gcs_wire = curves_to_wireframe_multiblock(curves, obstime=obstime, src_frame=gcs_src_frame, dst_frame=tomo_frame)

    gcs_surf = None
    if build_gcs_surface and ("parallels" in curves):
        gcs_surf = parallels_to_surface_polydata(
            curves["parallels"],
            obstime=obstime,
            src_frame=gcs_src_frame,
            dst_frame=tomo_frame,
            close_loop=True,
        )

    # Plot
    plotter = pv.Plotter(off_screen=not show, window_size=(1400, 1000))
    plotter.set_background("black")

    # Sun sphere
    plotter.add_mesh(pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=64), opacity=0.25, smooth_shading=True)

    # Iso-surface
    plotter.add_mesh(iso, opacity=0.35, smooth_shading=True)

    # GCS surface patch (optional) + wireframe
    if gcs_surf is not None:
        plotter.add_mesh(gcs_surf, opacity=0.15, smooth_shading=True)
    for blk in gcs_wire:
        plotter.add_mesh(blk, line_width=2.0)

    # Camera: default Earth view if not specified
    if camera_lonlat_deg is None:
        camera_lonlat_deg = default_earth_camera_lonlat(obstime=obstime, tomo_frame=tomo_frame)

    if camera_lonlat_deg is not None:
        set_camera_from_lonlat(plotter, lon_deg=camera_lonlat_deg[0], lat_deg=camera_lonlat_deg[1], distance_rsun=15.0)
    else:
        plotter.view_isometric()

    plotter.add_text(f"iso: {freq_mhz:.2f} MHz (harm={harmonic})", position="upper_left", font_size=12, color="white")

    if out_png is not None:
        out_png = Path(out_png).expanduser().resolve()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(out_png), auto_close=True)
    else:
        plotter.show(auto_close=True)


# -----------------------------------------------------------------------------
# __main__ (edit here; no argparse)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Directories (EDIT as needed)
    TOMOGRAPHY_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research/Tomography/")
    GCS_OVERLAY_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research/GCS/gcs_overlay")

    # If PyThea is *not* installed as a pip package but is a local repo, set the parent directory
    # that contains the "PyThea/" folder.
    PYTHEA_ROOT = None  # e.g., Path("/mnt/d/wsl/home/kinno-7010/Research")

    # 2) Inputs
    TOMO_NPZ = TOMOGRAPHY_DIR /"Rawdata/ne3d_solution.npz"  # <-- change to your file

    # 3) Iso-surface selection
    ISO_FREQ_MHZ = 37.0
    HARMONIC = 2

    # 4) GCS parameters (6)
    # lon, lat, tilt, height, half_angle(alpha), kappa
    GCS_HEIGHT_RSUN = 3.380
    GCS_KAPPA = 0.12
    GCS_HALF_ANGLE_DEG = 20.0
    GCS_TILT_DEG = -85.0
    GCS_LON_DEG = -44.0
    GCS_LAT_DEG = 10.0

    # 5) Time / frames / camera
    OBSTIME = "2022-06-13T03:24:58"

    # Interpret tomography cube coordinates in:
    TOMO_FRAME = "stonyhurst"  # or "stonyhurst"

    # GCS fit params are usually given in:
    GCS_SRC_FRAME = "stonyhurst"

    # For strict Earth view, set explicitly:
    # CAMERA_LONLAT_DEG = (0.0, 0.0) for Stonyhurst,
    # or (L0,B0) for Carrington (auto if sunpy available).
    CAMERA_LONLAT_DEG = None

    # 6) Rendering controls
    BUILD_GCS_SURFACE = True
    CLOSE_PHI_SEAM = True

    OUT_PNG = TOMOGRAPHY_DIR / f"overlay_tomo_gcs_{OBSTIME}_{ISO_FREQ_MHZ:.1f}MHz.png"
    SHOW = True

    # Imports (adds your directories to sys.path)
    mods = import_user_modules(TOMOGRAPHY_DIR, GCS_OVERLAY_DIR, pythea_root=PYTHEA_ROOT)
    GCSParams = mods["GCSParams"]

    gcs_params = GCSParams(
        h_apex=GCS_HEIGHT_RSUN,
        kappa=GCS_KAPPA,
        alpha_deg=GCS_HALF_ANGLE_DEG,
        tilt_deg=GCS_TILT_DEG,
        lon_deg=GCS_LON_DEG,
        lat_deg=GCS_LAT_DEG,
    )

    overlay_isosurface_and_gcs(
        tomography_npz=TOMO_NPZ,
        freq_mhz=ISO_FREQ_MHZ,
        harmonic=HARMONIC,
        gcs_params=gcs_params,
        obstime=OBSTIME,
        observer_hgs="earth",
        tomo_frame=TOMO_FRAME,
        gcs_src_frame=GCS_SRC_FRAME,
        camera_lonlat_deg=CAMERA_LONLAT_DEG,
        build_gcs_surface=BUILD_GCS_SURFACE,
        close_phi=CLOSE_PHI_SEAM,
        out_png=OUT_PNG,
        show=SHOW,
    )
