#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
overlay_tomography_gcs.py
========================

Purpose
-------
Overlay (i) an iso-frequency surface derived from a 3-D electron-density
tomography solution and (ii) a Graduated Cylindrical Shell (GCS) model shell.

This script is designed to *reuse your existing code* by importing:
- `SphericalGrid`, `ne_cm3_from_fp_mhz`, `fp_mhz_from_ne_cm3` from
  `main_regularized_tomography.py`
- `GCSParams`, `sample_gcs_wireframe_points` from your `gcs_overlay` package

It uses PyVista for 3-D rendering.

Key idea
--------
Type-II/related radio emission is often near the plasma frequency f_pe or its
harmonic, and f_pe is directly tied to electron density:

    f_pe = (1/2π) sqrt(n_e e^2 / (ε0 m_e))
         ≈ 8980 sqrt(n_e[cm^-3])  [Hz]

Thus an iso-frequency surface in the corona corresponds to an iso-density
surface. If the observed emission is harmonic (2 f_pe), then the iso-density is
computed from f/2.

Notes on frames (important)
---------------------------
Your tomography code builds rays in what it calls a “Carrington frame” (it
explicitly bins by (r, θ, φ) where φ is derived from arctan2(y, x) and
interpretation depends on how you define φ=0). Your GCS helper functions produce
3-D points in Heliographic Stonyhurst (HGS) Cartesian coordinates (SunPy).

To make the overlay meaningful, the GCS shell and tomography volume must be in
the *same* heliocentric frame. This script supports two modes:
- `tomo_frame = "stonyhurst"`: assume the tomography (x,y,z) is HGS-like
- `tomo_frame = "carrington"`: transform GCS points from HGS -> HGC (Carrington)

The HGS->HGC transform is performed with SunPy coordinate transforms if
available. (SunPy defines L0 and Carrington transforms; see docs.)
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


# -----------------------------------------------------------------------------
# Path setup: edit these if needed (NO argparse by request)
# -----------------------------------------------------------------------------
def _maybe_add_to_syspath(p: Union[str, Path]) -> None:
    p = Path(p).expanduser().resolve()
    if p.exists():
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def setup_import_paths(
    tomo_dir: Union[str, Path],
    gcs_parent_dir: Union[str, Path],
) -> None:
    """
    Add your tomography and GCS parent directories to sys.path.

    - `tomo_dir` should contain `main_regularized_tomography.py`
    - `gcs_parent_dir` should contain the `gcs_overlay/` package directory
      (i.e., `gcs_parent_dir / "gcs_overlay" / "__init__.py"` exists)
    """
    _maybe_add_to_syspath(tomo_dir)
    _maybe_add_to_syspath(gcs_parent_dir)


# -----------------------------------------------------------------------------
# Import your existing functions (with fallbacks)
# -----------------------------------------------------------------------------
def import_user_modules():
    """
    Import symbols from the user's existing scripts/packages.

    Returns
    -------
    SphericalGrid, ne_cm3_from_fp_mhz, fp_mhz_from_ne_cm3, GCSParams, sample_gcs_wireframe_points
    """
    # Tomography utilities
    try:
        from main_regularized_tomography import (  # type: ignore
            SphericalGrid,
            ne_cm3_from_fp_mhz,
            fp_mhz_from_ne_cm3,
        )
    except Exception as e:
        raise ImportError(
            "Failed to import tomography utilities from main_regularized_tomography.py. "
            "Confirm `tomo_dir` points to the folder containing that file."
        ) from e

    # GCS utilities (preferred as a package: gcs_overlay)
    try:
        from gcs_overlay import GCSParams, sample_gcs_wireframe_points  # type: ignore
    except Exception:
        # fallback: direct module import (if your files are not in a package)
        try:
            from gcs_geometry import GCSParams, sample_gcs_wireframe_points  # type: ignore
        except Exception as e:
            raise ImportError(
                "Failed to import GCS utilities. Expected either:\n"
                "  - a package `gcs_overlay` on PYTHONPATH, or\n"
                "  - a module `gcs_geometry.py` on PYTHONPATH.\n"
                "Confirm `gcs_parent_dir` is correct."
            ) from e

    return SphericalGrid, ne_cm3_from_fp_mhz, fp_mhz_from_ne_cm3, GCSParams, sample_gcs_wireframe_points


# -----------------------------------------------------------------------------
# Coordinate utilities
# -----------------------------------------------------------------------------
def _xyz_to_lonlatr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian (Rsun units) -> (lon[rad], lat[rad], r[Rsun]).
    """
    r = np.sqrt(x * x + y * y + z * z)
    lon = np.arctan2(y, x)
    lat = np.arcsin(np.where(r > 0, z / r, 0.0))
    return lon, lat, r


def _lonlatr_to_xyz(lon: np.ndarray, lat: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert (lon[rad], lat[rad], r[Rsun]) -> Cartesian (Rsun units).
    """
    clat = np.cos(lat)
    x = r * clat * np.cos(lon)
    y = r * clat * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z


def transform_points_hgs_to_hgc(
    pts_xyz: np.ndarray,
    obstime_str: str,
    observer: str = "earth",
) -> np.ndarray:
    """
    Transform Nx3 Cartesian points from HGS -> Heliographic Carrington (HGC).

    Prefer SunPy transforms; if unavailable, fall back to a simple z-rotation by L0.
    """
    pts_xyz = np.asarray(pts_xyz, dtype=float)
    if pts_xyz.ndim != 2 or pts_xyz.shape[1] != 3:
        raise ValueError("pts_xyz must be (N,3) array")

    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astropy.time import Time
        from sunpy.coordinates import frames
        # Build HGS SkyCoord from Cartesian
        X, Y, Z = pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2]
        lon, lat, r = _xyz_to_lonlatr(X, Y, Z)
        hgs = SkyCoord(
            lon=lon * u.rad,
            lat=lat * u.rad,
            radius=r * u.R_sun,
            frame=frames.HeliographicStonyhurst,
            obstime=Time(obstime_str),
        )
        # Transform to Carrington (requires observer)
        hgc = hgs.transform_to(frames.HeliographicCarrington(observer=observer, obstime=hgs.obstime))
        c = hgc.cartesian
        return np.vstack([c.x.to_value(u.R_sun), c.y.to_value(u.R_sun), c.z.to_value(u.R_sun)]).T
    except Exception:
        # Fallback: use L0 (Carrington longitude of disk center as seen from Earth)
        try:
            from astropy import units as u
            from astropy.time import Time
            from sunpy.coordinates.sun import L0  # apparent Carrington longitude of disk center
            L0_deg = L0(Time(obstime_str)).to_value(u.deg)
        except Exception as e:
            raise ImportError(
                "SunPy is required for HGS->HGC transform (or at least for L0 fallback). "
                "Install sunpy (and astropy)."
            ) from e

        ang = np.deg2rad(L0_deg)
        ca, sa = np.cos(ang), np.sin(ang)
        Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        return (Rz @ pts_xyz.T).T


def transform_surface_hgs_to_hgc(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, obstime_str: str, observer: str = "earth"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply HGS->HGC transform to a 2-D surface grid (X,Y,Z) in Rsun units.
    """
    shp = X.shape
    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    pts2 = transform_points_hgs_to_hgc(pts, obstime_str=obstime_str, observer=observer)
    X2 = pts2[:, 0].reshape(shp)
    Y2 = pts2[:, 1].reshape(shp)
    Z2 = pts2[:, 2].reshape(shp)
    return X2, Y2, Z2


# -----------------------------------------------------------------------------
# PyVista construction helpers
# -----------------------------------------------------------------------------
def build_pyvista_structured_grid_from_tomography(grid, ne_3d: np.ndarray):
    """
    Convert tomography (spherical grid + ne[r,th,ph]) into a PyVista StructuredGrid.
    """
    import pyvista as pv

    rr, tt, pp = grid.voxel_centers_sph()  # (nr, nth, nph), radians for angles
    x = rr * np.sin(tt) * np.cos(pp)
    y = rr * np.sin(tt) * np.sin(pp)
    z = rr * np.cos(tt)

    g = pv.StructuredGrid(x, y, z)
    g["ne_cm3"] = ne_3d.ravel(order="F")  # keep consistent with your tomography visualizer
    return g


def build_gcs_shell_surface(
    gcs_params,
    n_theta: int = 80,
    n_phi: int = 160,
    tomo_frame: str = "stonyhurst",
    obstime_str: Optional[str] = None,
    observer_for_hgc: str = "earth",
):
    """
    Build a PyVista surface mesh for the GCS shell.

    Returns a PyVista PolyData representing the shell surface.
    """
    import pyvista as pv

    try:
        from astropy import units as u
        from pythea.geometrical_models.gcs import GCS
    except Exception as e:
        raise ImportError(
            "To build a GCS *surface*, this script requires `pythea` and `astropy`. "
            "If you only want a wireframe, set `build_surface=False` in main."
        ) from e

    shell = GCS(
        height=gcs_params.height * u.R_sun,
        longitude=gcs_params.longitude * u.deg,
        latitude=gcs_params.latitude * u.deg,
        kappa=gcs_params.kappa,
        half_angle=gcs_params.half_angle * u.deg,
        tilt=gcs_params.tilt * u.deg,
        nbverts=max(128, n_phi),  # polyline resolution for internal use
    )

    theta = np.linspace(0.0, np.pi, int(n_theta))
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi))
    TT, PP = np.meshgrid(theta, phi, indexing="xy")  # (n_phi, n_theta)

    X, Y, Z = shell.rotate(*shell.shell(TT, PP))  # HGS Cartesian in Rsun

    if tomo_frame.lower().startswith("carr"):
        if obstime_str is None:
            raise ValueError("obstime_str is required to transform GCS HGS->HGC when tomo_frame='carrington'.")
        X, Y, Z = transform_surface_hgs_to_hgc(X, Y, Z, obstime_str=obstime_str, observer=observer_for_hgc)

    g = pv.StructuredGrid(X, Y, Z)
    surf = g.extract_surface().triangulate()
    return surf


def add_gcs_wireframe_to_plotter(
    plotter,
    wire: Dict[str, List[np.ndarray]],
    color: str = "white",
    line_width: float = 2.0,
    opacity: float = 1.0,
):
    """
    Add GCS wireframe polylines (from sample_gcs_wireframe_points) to a PyVista plotter.
    """
    import pyvista as pv

    def _add_polyline(pts: np.ndarray):
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return
        poly = pv.lines_from_points(pts, close=False)
        plotter.add_mesh(poly, color=color, line_width=float(line_width), opacity=float(opacity))

    for key in ("parallels", "meridians", "legs"):
        for pts in wire.get(key, []):
            _add_polyline(pts)


# -----------------------------------------------------------------------------
# Main visualization routine
# -----------------------------------------------------------------------------
def overlay_isosurface_and_gcs(
    tomo_npz: Union[str, Path],
    iso_freqs_mhz: Union[float, Iterable[float]],
    harmonic: int,
    gcs_params,
    obstime_str: str,
    tomo_frame: str = "stonyhurst",
    build_surface: bool = True,
    show_sun: bool = True,
    iso_opacity: float = 0.5,
    gcs_opacity: float = 0.12,
    gcs_color: str = "white",
    wire_color: str = "white",
    wire_width: float = 2.0,
    camera_lonlat_deg: Optional[Tuple[float, float]] = None,
    show_axes: bool = True,
    show_gui: bool = True,
    save_png: bool = False,
    png_path: Optional[Union[str, Path]] = None,
):
    """
    Load tomography -> build iso-frequency surfaces -> overlay GCS shell/wireframe.
    """
    import pyvista as pv
    from astropy.time import Time

    SphericalGrid, ne_cm3_from_fp_mhz, fp_mhz_from_ne_cm3, GCSParams, sample_gcs_wireframe_points = import_user_modules()

    tomo_npz = Path(tomo_npz).expanduser().resolve()
    if not tomo_npz.exists():
        raise FileNotFoundError(f"Tomography solution not found: {tomo_npz}")

    # --------------------
    # Load tomography
    # --------------------
    dat = np.load(tomo_npz)
    ne = dat["ne"]
    r_edges = dat["r_edges"]
    th_edges = dat["th_edges"]
    ph_edges = dat["ph_edges"]
    grid = SphericalGrid(r_edges, th_edges, ph_edges)

    pv_grid = build_pyvista_structured_grid_from_tomography(grid, ne)

    # --------------------
    # Iso-frequency surfaces (converted to iso-density)
    # --------------------
    if np.isscalar(iso_freqs_mhz):
        freq_list = [float(iso_freqs_mhz)]
    else:
        freq_list = [float(x) for x in iso_freqs_mhz]

    # f_obs = harmonic * f_pe  =>  f_pe = f_obs / harmonic
    iso_ne_vals = [float(ne_cm3_from_fp_mhz(f / float(harmonic))) for f in freq_list]

    # Rendering
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=not bool(show_gui))
    plotter.set_background("black")

    if show_axes:
        plotter.add_axes(line_width=1.5)

    # Sun sphere (for context only; no background coronagraph per your request)
    if show_sun:
        sun = pv.Sphere(radius=1.0, center=(0, 0, 0), theta_resolution=90, phi_resolution=90)
        plotter.add_mesh(sun, color="gray", opacity=0.25)

    # Add iso-surfaces
    iso_colors = ["tomato", "deepskyblue", "gold", "limegreen", "violet", "orange"]
    if len(iso_colors) < len(freq_list):
        k = (len(freq_list) + len(iso_colors) - 1) // len(iso_colors)
        iso_colors = (iso_colors * k)[: len(freq_list)]

    for f_mhz, ne_val, col in zip(freq_list, iso_ne_vals, iso_colors):
        surf = pv_grid.contour([ne_val], scalars="ne_cm3")
        # If the contour is empty, still keep the run going
        if surf.n_points > 0:
            plotter.add_mesh(surf, color=col, opacity=float(iso_opacity), name=f"iso_{f_mhz:.2f}MHz")
        # Add a text legend entry
        plotter.add_text(f"iso: {f_mhz:.2f} MHz (harm={harmonic})", position="upper_left", font_size=10, color="white")

    # --------------------
    # Add GCS: surface + wireframe
    # --------------------
    obstime = Time(obstime_str)

    # Wireframe (always)
    wire = sample_gcs_wireframe_points(
        gcs_params,
        obstime=obstime,
        n_parallels=10,
        n_meridians=14,
        include_legs=True,
    )

    # Transform wireframe if tomography frame is Carrington
    if tomo_frame.lower().startswith("carr"):
        wire2: Dict[str, List[np.ndarray]] = {"parallels": [], "meridians": [], "legs": []}
        for key in wire2.keys():
            for pts in wire.get(key, []):
                wire2[key].append(transform_points_hgs_to_hgc(pts, obstime_str=obstime_str, observer="earth"))
        wire = wire2

    add_gcs_wireframe_to_plotter(plotter, wire, color=wire_color, line_width=wire_width, opacity=1.0)

    if build_surface:
        gcs_surf = build_gcs_shell_surface(
            gcs_params,
            n_theta=90,
            n_phi=180,
            tomo_frame=tomo_frame,
            obstime_str=obstime_str,
            observer_for_hgc="earth",
        )
        plotter.add_mesh(gcs_surf, color=gcs_color, opacity=float(gcs_opacity))

    # --------------------
    # Camera
    # --------------------
    plotter.view_isometric()
    plotter.camera.zoom(1.2)

    if camera_lonlat_deg is not None:
        # This is a simple *view* helper: set camera position on a sphere.
        lon_deg, lat_deg = camera_lonlat_deg
        lon = np.deg2rad(lon_deg)
        lat = np.deg2rad(lat_deg)
        cam_r = 10.0  # Rsun, arbitrary distance
        cx = cam_r * np.cos(lat) * np.cos(lon)
        cy = cam_r * np.cos(lat) * np.sin(lon)
        cz = cam_r * np.sin(lat)
        plotter.camera_position = [(cx, cy, cz), (0, 0, 0), (0, 0, 1)]

    # --------------------
    # Show / save
    # --------------------
    if save_png:
        if png_path is None:
            png_path = tomo_npz.with_suffix("").as_posix() + "_overlay.png"
        png_path = Path(png_path).expanduser().resolve()
        png_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(png_path), auto_close=True)
        return str(png_path)

    plotter.show(auto_close=True)
    return None


# -----------------------------------------------------------------------------
# Entry point (edit ONLY here for parameters; no argparse by request)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Set your local code locations
    TOMO_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research/Tomography/py_folder")
    GCS_PARENT_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research/GCS")  # contains gcs_overlay/

    setup_import_paths(TOMO_DIR, GCS_PARENT_DIR)

    # 2) Tomography NPZ (your saved 3D solution)
    TOMO_NPZ = Path("/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/pB/Rawdata/ne3d_solution.npz")

    # 3) Iso-frequency surface (MHz) and harmonic
    #    Example: harmonic=2 means you provide observed frequency ~ 2 f_pe
    ISO_FREQS_MHZ = [25.0]   # can be float or list[float]
    HARMONIC = 2

    # 4) GCS 6 parameters (Thernisien-style)
    #    height [Rsun], lon/lat/tilt [deg], kappa [-], half_angle [deg]
    #    Replace the numbers below with your fitted values.
    from gcs_overlay import GCSParams  # type: ignore

    GCS_PARAMS = GCSParams(
        height=3.5,
        longitude=10.0,
        latitude=5.0,
        tilt=20.0,
        kappa=0.35,
        half_angle=30.0,
        label="GCS",
    )

    # 5) Time (needed if you request Carrington transforms)
    OBSTIME_STR = "2022-06-13T03:00:00"

    # 6) Frame choice: set to "carrington" if your tomography longitudes are Carrington
    TOMO_FRAME = "carrington"   # "stonyhurst" or "carrington"

    # 7) Output control
    SHOW_GUI = True
    SAVE_PNG = True
    PNG_PATH = Path("/mnt/d/wsl/home/kinno-7010/Research/overlay_tomo_gcs.png")

    overlay_isosurface_and_gcs(
        tomo_npz=TOMO_NPZ,
        iso_freqs_mhz=ISO_FREQS_MHZ,
        harmonic=HARMONIC,
        gcs_params=GCS_PARAMS,
        obstime_str=OBSTIME_STR,
        tomo_frame=TOMO_FRAME,
        build_surface=True,
        show_sun=True,
        iso_opacity=0.55,
        gcs_opacity=0.10,
        gcs_color="white",
        wire_color="white",
        wire_width=2.0,
        camera_lonlat_deg=None,
        show_axes=True,
        show_gui=SHOW_GUI,
        save_png=SAVE_PNG,
        png_path=PNG_PATH,
    )
