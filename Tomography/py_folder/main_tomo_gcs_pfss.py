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

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
import sunpy.map
import numpy as np

try:
    import pyvista as pv
except Exception as e:
    raise SystemExit("pyvista is required: pip install pyvista") from e
sys.path.append("/home/kinno-7010/Research_code/GCS/gcs_overlay")
# sys.path.append("/home/kinno-7010/Research_code/GCS/gcs_overlay")  # 共用PC 用
from main_regularized_tomography import (  # type: ignore
    SphericalGrid,
    RegularizedTomography,
    build_observation,
    build_rays_for_observation,
    ybk_profile_fft,
    ne_cm3_from_fp_mhz,
)
from gcs_geometry import GCSParams, sample_gcs_wireframe_points
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, CartesianRepresentation
from sunpy.coordinates import frames
# -----------------------------------------------------------------------------
# Import user's tomography + GCS helpers
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Observer-geometry helpers (Carrington consistency)
# -----------------------------------------------------------------------------

def _wrap180(deg: float) -> float:
    """Wrap angle to [-180, 180) degrees."""
    return (deg + 180.0) % 360.0 - 180.0


def _read_fits_header_minimal(fp: Path):
    """
    Read FITS primary header with minimal overhead.
    Uses astropy.io.fits (preferred). Raises on I/O errors.
    """
    from astropy.io import fits  # local import to keep module import light
    return fits.getheader(str(fp), 0)


def _wrap360(deg: float) -> float:
    """Wrap angle to [0, 360) degrees."""
    return deg % 360.0


def _build_lonlat_override_map_carrington(pb_fits_list: List[Union[str, Dict]]) -> Dict[str, Tuple[float, float]]:
    """
    Build {basename: (CRLN_OBS, CRLT_OBS)} override map to force
    Carrington observer long/lat in the tomography geometry.

    Background
    ----------
    In this pipeline, the tomography forward model and the reconstructed volume are
    defined in a Carrington-rotating Cartesian basis (see camera_basis_from_lonlat
    and SphericalGrid in main_regularized_tomography.py).

    Many solar FITS headers include both:
      - CRLN_OBS / CRLT_OBS : Carrington lon/lat of the observer
      - HGLN_OBS / HGLT_OBS : Stonyhurst lon/lat of the observer

    If a file lacks CRLN_OBS/CRLT_OBS, falling back to HGLN_OBS/HGLT_OBS *as if they
    were Carrington* introduces a longitude offset of ~L0 (Earth Carrington longitude),
    which can place reconstructed density on the wrong hemisphere and also misalign
    GCS/PFSS overlays.

    Strategy
    --------
    1) Prefer CRLN_OBS/CRLT_OBS when present.
    2) Otherwise, convert Stonyhurst → Carrington using an offset L0 derived from any
       file that has BOTH CRLN_OBS and HGLN_OBS:
           L0 ≈ wrap360(CRLN_OBS - HGLN_OBS)
       because (to good approximation)  HGLN_OBS ≈ CRLN_OBS - L0 (wrapped to [-180,180)).
    """
    # Collect unique filepaths
    paths: List[Path] = []
    for item in pb_fits_list:
        if isinstance(item, dict):
            paths.append(Path(str(item["path"])))
        else:
            paths.append(Path(str(item)))

    # Derive L0 (Earth Carrington longitude) from any header that has both CRLN and HGLN.
    L0: Optional[float] = None
    for fp in paths:
        try:
            hdr = _read_fits_header_minimal(fp)
        except Exception:
            continue
        crln = hdr.get("CRLN_OBS")
        hgln = hdr.get("HGLN_OBS")
        if (crln is None) or (hgln is None):
            continue
        try:
            L0 = _wrap360(float(crln) - float(hgln))
            break
        except Exception:
            continue

    override: Dict[str, Tuple[float, float]] = {}
    for fp in paths:
        hdr = _read_fits_header_minimal(fp)
        bn = fp.name

        crln = hdr.get("CRLN_OBS")
        crlt = hdr.get("CRLT_OBS")
        if (crln is not None) and (crlt is not None):
            override[bn] = (_wrap360(float(crln)), float(crlt))
            continue

        # Fallback: use Stonyhurst lon/lat and convert to Carrington using L0.
        hgln = hdr.get("HGLN_OBS")
        hglt = hdr.get("HGLT_OBS")
        if (hgln is None) or (hglt is None) or (L0 is None):
            raise KeyError(
                f"Cannot determine Carrington observer lon/lat for: {fp}. "
                f"Need CRLN_OBS/CRLT_OBS, or HGLN_OBS/HGLT_OBS plus at least one file "
                f"that provides both CRLN_OBS and HGLN_OBS to derive L0."
            )

        crln_est = _wrap360(float(hgln) + float(L0))
        override[bn] = (crln_est, float(hglt))

    return override
def _safe_import_pfsspy():
    """
    Import pfsspy in a way that is more stable on WSL / OpenMP / numba environments.

    Returns
    -------
    pfsspy, utils, tracing
    """
    import builtins
    import sys

    # If pfsspy is already imported, we still prefer a clean import under a numba-block,
    # because some environments crash/behave inconsistently depending on import order.
    for key in list(sys.modules):
        if key == "pfsspy" or key.startswith("pfsspy."):
            sys.modules.pop(key, None)

    # Temporarily remove numba modules if they are already loaded.
    numba_module = sys.modules.pop("numba", None)
    numba_submodules = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("numba.")}

    class _BlockNumbaImport:
        def __enter__(self):
            self._orig_import = builtins.__import__

            def _import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "numba" or name.startswith("numba."):
                    raise ModuleNotFoundError("numba blocked for pfsspy import stability")
                return self._orig_import(name, globals, locals, fromlist, level)

            builtins.__import__ = _import

        def __exit__(self, exc_type, exc, tb):
            builtins.__import__ = self._orig_import

    try:
        with _BlockNumbaImport():
            import pfsspy  # noqa
            from pfsspy import utils, tracing  # noqa
    finally:
        # Restore numba modules
        if numba_module is not None:
            sys.modules["numba"] = numba_module
        sys.modules.update(numba_submodules)

    return pfsspy, utils, tracing


def _resample_polyline_xyz(
    pts: np.ndarray,
    n: int,
    *,
    closed: bool = True,
    close_tol: float = 1e-6,
) -> np.ndarray:
    """
    Resample a 3D polyline to n points (or n+1 if closed=True and we close the loop).

    Parameters
    ----------
    pts : (N,3) array
        Input polyline points.
    n : int
        Number of samples along the curve (excluding the duplicated closing point).
    closed : bool
        If True, treat curve as periodic (last connects to first) and return (n+1,3)
        with last point equal to first, so surface seams are closed.
    close_tol : float
        Tolerance to consider first/last identical.

    Returns
    -------
    out : (n,3) or (n+1,3) array
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("pts must have shape (N,3).")
    if pts.shape[0] < 2:
        return pts.copy()

    # Remove NaN rows
    m = np.all(np.isfinite(pts), axis=1)
    pts = pts[m]
    if pts.shape[0] < 2:
        return pts.copy()

    # If closed and already has duplicated endpoint, drop the last
    if closed:
        d = np.linalg.norm(pts[-1] - pts[0])
        if d <= close_tol:
            pts = pts[:-1]
            if pts.shape[0] < 2:
                pts = np.vstack([pts, pts[0]])

        # Build periodic polyline by appending first point
        pts_ext = np.vstack([pts, pts[0]])
    else:
        pts_ext = pts

    # Arc-length parameter
    seg = np.diff(pts_ext, axis=0)
    ds = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = s[-1]
    if not np.isfinite(total) or total <= 0:
        out = np.repeat(pts_ext[:1], n + (1 if closed else 0), axis=0)
        return out

    if closed:
        # sample on [0,total) with n points, then close explicitly by repeating first
        s_new = np.linspace(0.0, total, int(n), endpoint=False)
        x = np.interp(s_new, s, pts_ext[:, 0])
        y = np.interp(s_new, s, pts_ext[:, 1])
        z = np.interp(s_new, s, pts_ext[:, 2])
        out = np.column_stack([x, y, z])
        out = np.vstack([out, out[0]])
        return out

    # open curve: sample with endpoint True
    s_new = np.linspace(0.0, total, int(n), endpoint=True)
    x = np.interp(s_new, s, pts_ext[:, 0])
    y = np.interp(s_new, s, pts_ext[:, 1])
    z = np.interp(s_new, s, pts_ext[:, 2])
    out = np.column_stack([x, y, z])
    return out


def build_gcs_shell_surface_from_parallels(
    parallels_hgc: List[np.ndarray],
    *,
    n_u: int = 120,
    closed_u: bool = True,
    triangulate: bool = True,
) -> Optional[pv.PolyData]:
    """
    Build an approximate GCS shell surface (membrane) by lofting between parallel rings.

    Parameters
    ----------
    parallels_hgc : list of (Ni,3) arrays
        Parallel curves in *HGC Cartesian* (Rsun units). Each curve is a ring around the shell.
    n_u : int
        Number of samples along each ring (u-direction).
    closed_u : bool
        If True, close each ring seam (recommended).
    triangulate : bool
        If True, triangulate the extracted surface.

    Returns
    -------
    surf : pv.PolyData or None
    """
    if parallels_hgc is None or len(parallels_hgc) < 2:
        return None

    rings = []
    for pts in parallels_hgc:
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 4:
            continue
        pts_rs = _resample_polyline_xyz(pts, int(n_u), closed=bool(closed_u))
        if pts_rs.shape[0] < 4:
            continue
        rings.append(pts_rs)

    if len(rings) < 2:
        return None

    # Ensure same length
    nu0 = rings[0].shape[0]
    for r in rings:
        if r.shape[0] != nu0:
            return None

    # Stack into StructuredGrid: (nu, nv)
    arr = np.stack(rings, axis=1)  # (nu, nv, 3)
    X = arr[:, :, 0]
    Y = arr[:, :, 1]
    Z = arr[:, :, 2]

    sgrid = pv.StructuredGrid(X, Y, Z)

    surf = sgrid.extract_surface()
    if triangulate:
        surf = surf.triangulate()

    return surf


def add_gcs_shell_surface(
    plotter,
    params,
    *,
    obstime_iso: str,
    observer: str = "earth",
    color: str = "lime",
    opacity: float = 0.18,
    smooth_shading: bool = True,
    n_u: int = 140,
    closed_u: bool = True,
    show_wireframe: bool = False,
    wire_color: str = "lime",
    wire_width: int = 2,
    return_surface: bool = False,   # ★追加
):
    """
    Render a GCS *surface* ("membrane") by lofting between the parallel rings.

    return_surface=True のとき、戻り値 dict に "surface": pv.PolyData を含める。
    """
    import numpy as np

    curves = sample_gcs_wireframe_points(params, obstime=obstime_iso)

    parallels = curves.get("parallels", [])
    if parallels is None or len(parallels) < 2:
        print("[WARN] GCS surface skipped: curves['parallels'] is empty or too short.")
        return None

    parallels_hgc = []
    for pts in parallels:
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 4:
            continue
        pts_hgc = transform_points_hgs_to_hgc(
            pts,
            obstime_iso=obstime_iso,
            observer=observer,
        )
        parallels_hgc.append(pts_hgc)

    surf = build_gcs_shell_surface_from_parallels(
        parallels_hgc,
        n_u=int(n_u),
        closed_u=bool(closed_u),
        triangulate=True,
    )
    if surf is None or surf.n_points == 0:
        print("[WARN] GCS surface build failed (no surface points).")
        return None

    plotter.add_mesh(
        surf,
        color=color,
        opacity=float(opacity),
        smooth_shading=bool(smooth_shading),
        lighting=True,
        pickable=False,
    )

    if show_wireframe:
        for key in ("parallels", "meridians", "legs"):
            for pts in curves.get(key, []):
                pts = np.asarray(pts, dtype=np.float64)
                if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
                    continue
                pts_hgc = transform_points_hgs_to_hgc(
                    pts,
                    obstime_iso=obstime_iso,
                    observer=observer,
                )
                plotter.add_mesh(
                    _polyline_from_points(pts_hgc),
                    color=wire_color,
                    line_width=int(wire_width),
                    lighting=False,
                )

    info = {"n_points": int(surf.n_points), "n_cells": int(surf.n_cells)}
    if return_surface:
        info["surface"] = surf
    print(f"[INFO] GCS surface rendered: n_points={surf.n_points}, n_cells={surf.n_cells}")
    return info

def add_gcs_surface_at_radius(
    plotter,
    gcs_surface,
    *,
    r0: float,
    dr: float = 0.05,
    mode: str = "band",          # "band" or "scatter"
    color: str = "yellow",
    opacity: float = 0.95,
    point_size: int = 12,
    max_points: int = 5000,
    rng_seed: int = 0,
):
    """
    太陽中心から r=r0±dr (Rsun) の範囲にある GCS 表面部分をハイライトする。

    変更点:
      - 「別の add_text を増やす」方式をやめ、run-info(TextActor)の中身を更新する。
      - 位置は add_runinfo_legend で決めた position（左上）を維持する。
    """
    import numpy as np
    import pyvista as pv

    def _set_runinfo_text(new_txt: str):
        # run-info actor を「置き換え」る（SetInput が効かない環境でも確実）
        pos = getattr(plotter, "_runinfo_text_position", "upper_left")
        fs = getattr(plotter, "_runinfo_text_font_size", 12)
        col = getattr(plotter, "_runinfo_text_color", "black")

        old = getattr(plotter, "_runinfo_text_actor", None)
        if old is not None:
            try:
                plotter.remove_actor(old)
            except Exception:
                pass

        actor = plotter.add_text(new_txt, position=pos, font_size=int(fs), color=col)
        plotter._runinfo_text_actor = actor
        plotter._runinfo_text_base = new_txt
        return actor

    def _append_runinfo_line(line: str):
        base = getattr(plotter, "_runinfo_text_base", None)
        if base is None:
            # run-info がまだ無いなら pending へ
            pending = getattr(plotter, "_pending_runinfo_lines", [])
            if line not in pending:
                pending.append(line)
            plotter._pending_runinfo_lines = pending
            return

        extras = getattr(plotter, "_runinfo_text_extra_lines", [])
        if line not in extras:
            extras.append(line)
        plotter._runinfo_text_extra_lines = extras

        new_txt = base.split("\n")
        # base に既に含まれていなければ追加
        if line not in new_txt:
            new_txt.append(line)
        # extras も一応反映（重複回避）
        for e in extras:
            if e not in new_txt:
                new_txt.append(e)

        _set_runinfo_text("\n".join(new_txt))

        try:
            plotter.render()
        except Exception:
            pass

    if gcs_surface is None or getattr(gcs_surface, "n_points", 0) == 0:
        print("[WARN] add_gcs_surface_at_radius: empty gcs_surface")
        return {"n_selected": 0, "n_total": 0}

    r0 = float(r0)
    dr = float(abs(dr))

    surf = gcs_surface.copy(deep=True)
    pts = np.asarray(surf.points, dtype=float)
    rr = np.sqrt(np.sum(pts * pts, axis=1))
    surf["r_rsun"] = rr

    lo, hi = (r0 - dr), (r0 + dr)

    if mode.lower() == "band":
        band = surf.threshold(value=(lo, hi), scalars="r_rsun")
        nsel = int(getattr(band, "n_points", 0))
        if nsel == 0:
            print(f"[INFO] No GCS surface cells within r={r0:g}±{dr:g} Rsun")
            return {"n_selected": 0, "n_total": int(surf.n_points)}

        plotter.add_mesh(
            band,
            color=color,
            opacity=float(opacity),
            smooth_shading=True,
            lighting=True,
            pickable=False,
        )

        band_line = f"Band: r = {r0:.2f} ± {dr:.2f}"+ " R⊙"
        _append_runinfo_line(band_line)

        print(f"[INFO] GCS radial band rendered: r={r0:g}±{dr:g} Rsun, n_points={nsel}")
        return {"n_selected": nsel, "n_total": int(surf.n_points)}

    elif mode.lower() == "scatter":
        m = np.isfinite(rr) & (rr >= lo) & (rr <= hi)
        idx = np.where(m)[0]
        if idx.size == 0:
            print(f"[INFO] No GCS vertices within r={r0:g}±{dr:g} Rsun")
            return {"n_selected": 0, "n_total": int(surf.n_points)}

        if idx.size > int(max_points):
            rng = np.random.default_rng(int(rng_seed))
            idx = rng.choice(idx, size=int(max_points), replace=False)

        pts_sel = pts[idx]
        plotter.add_mesh(
            pv.PolyData(pts_sel),
            color=color,
            point_size=int(point_size),
            render_points_as_spheres=True,
            opacity=float(opacity),
            pickable=False,
        )

        band_line = f"Band: r = {r0:.2f} ± {dr:.2f} R⊙"
        _append_runinfo_line(band_line)

        print(f"[INFO] GCS radial scatter rendered: r={r0:g}±{dr:g} Rsun, n_points={pts_sel.shape[0]}")
        return {"n_selected": int(pts_sel.shape[0]), "n_total": int(surf.n_points)}

    else:
        raise ValueError("mode must be 'band' or 'scatter'")

    
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


def add_isosurfaces(
    plotter,
    sg,
    iso_freqs_mhz,
    harmonic=2,
    opacity=0.35,
    colors=None,
    *,
    return_surfaces: bool = False,
    range_text_mode: str = "runinfo",  # "runinfo" (推奨), "plot", "none"
):
    """
    StructuredGrid sg（'ne' スカラー）に対し、iso_freqs_mhz（MHz）の等周波数面を重ねる。

    range_text_mode:
      - "runinfo": 旧 lower_left のテキストを描かず、plotter に保持して後で run-info に統合
      - "plot":    従来どおり lower_left に表示
      - "none":    表示しない
    """
    if colors is None:
        colors = "red"
        colors = "cyan"
        colors = "gold"

    def ne_cm3_from_fp_mhz_local(f_mhz, H):
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
        return [] if return_surfaces else None

    fmin = float(fp_mhz_from_ne_cm3_local(ne_pos.min(), harmonic))
    fmax = float(fp_mhz_from_ne_cm3_local(ne_pos.max(), harmonic))

    harm_label = "Second Harmonic" if harmonic == 2 else "Fundamental"
    range_line = f"Reconstructed f-range: {fmin:.2f} .. {fmax:.2f} MHz ({harm_label})"

    mode = str(range_text_mode).lower()
    if mode == "plot":
        # plotter.add_text(range_line, position="lower_left", font_size=12, color="black")
        pass
    elif mode == "runinfo":
        # run-info がまだ無い段階でも保持できるように "pending" として保存
        pending = getattr(plotter, "_pending_runinfo_lines", [])
        # if range_line not in pending:
        #     # pending.append(range_line)
        #     pass
        # plotter._pending_runinfo_lines = pending
    elif mode == "none":
        pass
    else:
        raise ValueError("range_text_mode must be 'runinfo', 'plot', or 'none'.")

    rendered = 0
    surfaces = []

    for i, f_req in enumerate(iso_freqs_mhz):
        f_req = float(f_req)

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
        surfaces.append(surf)
        rendered += 1

    if rendered == 0:
        print("No isosurface rendered.\nCheck iso_freqs_mhz or tomography reconstruction range.")
        plotter.add_text(
            "No isosurface rendered.\nCheck iso_freqs_mhz or tomography reconstruction range.",
            position="upper_left",
            font_size=14,
            color="black",
        )
    else:
        print(f"[INFO] Isosurfaces rendered: {rendered}")

    return surfaces if return_surfaces else None

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
    x_hat = sun_to_observer_unit_vector(obs0)  # Sun->Earth
    p0 = x_hat * float(start_rsun)
    p1 = x_hat * (float(start_rsun) + float(length_rsun))
    plotter.add_mesh(pv.Line(p0, p1), color=color, line_width=int(line_width))

def add_physical_axes_triad(
    plotter,
    obs0,
    *,
    origin_rsun: float = 1.0,     # sub-Earth point on the surface
    axis_len: float = 1.0,        # ★短く
    shaft_radius: float = 0.02,   # ★細く
    tip_radius: float = 0.04,     # ★細く
    tip_length: float = 0.15,
    color_x: str = "crimson",
    color_y: str = "seagreen",
    color_z: str = "royalblue",
    label_font_size: int = 12,
):
    """
    Draw a labeled triad:
      X: Sun->Earth line
      Z: Solar north (+Z of the tomography Cartesian)
      Y: 'West/right' direction defined as Z x X (right direction when up=North)
    """
    x_hat = sun_to_observer_unit_vector(obs0)
    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)

    y_hat = np.cross(z_hat, x_hat)
    yn = np.linalg.norm(y_hat)
    if (not np.isfinite(yn)) or yn == 0:
        y_hat = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        y_hat = y_hat / yn

    o = x_hat * float(origin_rsun)

    def _arrow(dir_hat, color):
        arr = pv.Arrow(
            start=o,
            direction=dir_hat,
            scale=float(axis_len),
            tip_length=float(tip_length),
            tip_radius=float(tip_radius),
            shaft_radius=float(shaft_radius),
        )
        plotter.add_mesh(arr, color=color)

    _arrow(x_hat, color_x)
    _arrow(y_hat, color_y)
    _arrow(z_hat, color_z)

    tips = np.vstack([o + x_hat * axis_len, o + y_hat * axis_len, o + z_hat * axis_len])
    labels = ["X (Sun–Earth)", "Y (West / right)", "Z (North)"]
    plotter.add_point_labels(
        tips,
        labels,
        point_size=0,
        font_size=int(label_font_size),
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
    pfss_params: Optional[dict] = None,
    position: str = "upper_right",
    font_size: int = 12,
):
    """
    右上に出ていた旧Legendを廃止し、左上に run-info を 1 つだけ出す。
    add_isosurfaces() が保存した _pending_runinfo_lines もここで統合する。
    """
    freqs = ", ".join([f"{float(f):.1f}" for f in iso_freqs_mhz])

    base_lines = [
        f"Time: {obstime_iso}",
        f"Iso-freq: [{freqs}] MHz (H={int(harmonic)})",
        f"GCS: h={gcs_params['h_apex']:.2f} Rsun, kappa={gcs_params['kappa']:.3f}",
        f"     alpha={gcs_params['alpha_deg']:.1f} deg, tilt={gcs_params['tilt_deg']:.1f} deg",
        f"     lon={gcs_params['lon_deg']:.1f} deg, lat={gcs_params['lat_deg']:.1f} deg",
    ]

    if pfss_params:
        rss = pfss_params.get("rss", None)
        pfss_line = "PFSS"
        if rss is not None:
            pfss_line += f": Rss={float(rss):.2f} Rsun"
        base_lines.append(pfss_line)

    # add_isosurfaces() などが先に保持した pending を取り込む
    pending = getattr(plotter, "_pending_runinfo_lines", [])
    pending = [str(s) for s in pending] if pending else []
    for s in pending:
        if s not in base_lines:
            base_lines.append(s)

    txt = "\n".join(base_lines)

    # 既存 actor があれば削除（＝右上などの旧Legendを確実に消す）
    old_actor = getattr(plotter, "_runinfo_text_actor", None)
    if old_actor is not None:
        try:
            plotter.remove_actor(old_actor)
        except Exception:
            pass

    # ★左上に固定（呼び出し側が upper_right を渡してもここで統一したい場合は下行を有効化）
    # position = "upper_left"

    actor = plotter.add_text(txt, position=position, font_size=int(font_size), color="black")

    # 後から追記更新できるよう保持
    plotter._runinfo_text_base = txt
    plotter._runinfo_text_actor = actor
    plotter._runinfo_text_position = position
    plotter._runinfo_text_font_size = int(font_size)
    plotter._runinfo_text_color = "black"
    plotter._runinfo_text_extra_lines = []  # add_gcs_surface_at_radius 等が追記

    return actor


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
    distance_rsun: float = 5.0,
):
    """
    Robust camera placement for PyVista in this workflow.

    Fixes “blank (text-only) PNG” cases by:
      - using sun_to_observer_unit_vector(obs0) (less fragile than lon/lat metadata)
      - auto-increasing camera distance if scene bounds are larger than expected
      - forcing a sane clipping range + resetting clipping based on actors

    Also:
      - reset VTK interactor style to TrackballCamera
      - remove VTK's "Press R to toggle selection tool" overlay if it exists
    """
    cam_dir = sun_to_observer_unit_vector(obs0)

    # If PFSS/GCS accidentally introduces large-scale coordinates, keep them visible.
    b = plotter.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
    b = np.asarray(b, dtype=float)
    if np.all(np.isfinite(b)):
        max_extent = float(np.max(np.abs(b)))
        # Put camera sufficiently far to see the whole scene.
        min_dist = max(20.0, 3.0 * max_extent)
        dist = max(float(distance_rsun), min_dist)
    else:
        dist = float(distance_rsun)

    cam_pos = cam_dir * dist

    # Debug: log computed camera info (harmless in headless/offscreen runs)
    try:
        print(f"[DEBUG] set_camera_from_observation: cam_dir={cam_dir}, dist={dist}, cam_pos={cam_pos}, bounds={b}")
    except Exception:
        pass

    # Set camera explicitly (PyVista high-level API)
    try:
        plotter.camera_position = [cam_pos.tolist(), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    except Exception:
        pass

    # Robust fallback: directly set VTK camera properties in case the high-level API
    # does not take effect due to backend/version differences.
    try:
        cam = getattr(plotter, "camera", None)
        if cam is not None:
            cam.SetPosition(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
            cam.SetFocalPoint(0.0, 0.0, 0.0)
            cam.SetViewUp(0.0, 0.0, 1.0)
            # Ensure the plotter reflects the change immediately
            try:
                plotter.render()
            except Exception:
                pass
    except Exception:
        pass

    # Force clipping range (then refine based on scene actors)
    try:
        plotter.camera.clipping_range = (0.01, max(1000.0, 4.0 * dist))
    except Exception:
        pass

    # Very important: update clipping from current actors (prevents “all clipped out”)
    try:
        plotter.reset_camera_clipping_range()
    except Exception:
        pass

    # ---- Fix: disable RubberBandPick-style interaction and remove its on-screen hint ----
    # 1) Prefer PyVista API if available
    try:
        plotter.enable_trackball_style()
    except Exception:
        pass

    # 2) Fallback: force VTK interactor style directly (if an interactor exists)
    try:
        import vtk  # noqa
        if getattr(plotter, "iren", None) is not None and getattr(plotter.iren, "interactor", None) is not None:
            plotter.iren.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    except Exception:
        pass

    # 3) Remove the specific overlay actor if it was already added to the renderer
    try:
        import vtk  # noqa

        ren = getattr(plotter, "renderer", None)
        if ren is not None:
            props = ren.GetViewProps()
            props.InitTraversal()

            to_remove = []
            n = props.GetNumberOfItems()
            for _ in range(n):
                prop = props.GetNextProp()
                txt = None

                if isinstance(prop, vtk.vtkCornerAnnotation):
                    # 0..3 corners; check all to be safe
                    try:
                        txt = "\n".join([(prop.GetText(i) or "") for i in range(4)])
                    except Exception:
                        txt = None
                elif isinstance(prop, vtk.vtkTextActor):
                    try:
                        txt = prop.GetInput()
                    except Exception:
                        txt = None

                if txt and ("toggle selection tool" in txt):
                    to_remove.append(prop)

            for prop in to_remove:
                try:
                    ren.RemoveViewProp(prop)
                except Exception:
                    pass
    except Exception:
        pass


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

def compute_gcs_meridian_parallel_crosspoints_hgc(
    curves: dict,
    *,
    obstime_iso: str,
    observer: str = "earth",
    n_resample: int = 200,
    cross_tol_rsun: float = 0.05,
    merge_tol_rsun: float = 0.03,
) -> np.ndarray:
    """
    GCS wireframe curves（HGS座標）から、meridian と parallel の交点候補を抽出し、
    HGC（tomography座標系）で返す。

    実装方針:
      - 各 curve を HGS->HGC 変換
      - resample して点数を揃える
      - meridian×parallel の最近接点ペアを探し、距離が cross_tol 以下なら交点候補として採用
      - 近い点は merge_tol で重複統合
    """
    parallels = curves.get("parallels", []) or []
    meridians = curves.get("meridians", []) or []
    if len(parallels) == 0 or len(meridians) == 0:
        return np.zeros((0, 3), dtype=float)

    # --- HGS -> HGC and resample ---
    par_list = []
    for pts in parallels:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 4:
            continue
        pts_hgc = transform_points_hgs_to_hgc(pts, obstime_iso=obstime_iso, observer=observer)
        pts_rs = _resample_polyline_xyz(pts_hgc, int(n_resample), closed=True)
        if pts_rs.shape[0] >= 4:
            par_list.append(pts_rs)

    mer_list = []
    for pts in meridians:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 4:
            continue
        pts_hgc = transform_points_hgs_to_hgc(pts, obstime_iso=obstime_iso, observer=observer)
        pts_rs = _resample_polyline_xyz(pts_hgc, int(n_resample), closed=False)
        if pts_rs.shape[0] >= 4:
            mer_list.append(pts_rs)

    if len(par_list) == 0 or len(mer_list) == 0:
        return np.zeros((0, 3), dtype=float)

    # --- Find near-intersections by closest approach ---
    cross_pts = []
    ct2 = float(cross_tol_rsun) ** 2

    for m in mer_list:
        # (Nm,1,3) - (1,Np,3) -> (Nm,Np,3)
        for p in par_list:
            d = m[:, None, :] - p[None, :, :]
            d2 = np.sum(d * d, axis=2)
            ij = np.unravel_index(int(np.argmin(d2)), d2.shape)
            if float(d2[ij]) <= ct2:
                pm = m[ij[0]]
                pp = p[ij[1]]
                cross_pts.append(0.5 * (pm + pp))

    if len(cross_pts) == 0:
        return np.zeros((0, 3), dtype=float)

    pts = np.asarray(cross_pts, dtype=float)

    # --- Merge duplicates within merge_tol ---
    merged = []
    mt2 = float(merge_tol_rsun) ** 2
    for q in pts:
        keep = True
        for r in merged:
            if float(np.sum((q - r) ** 2)) <= mt2:
                keep = False
                break
        if keep:
            merged.append(q)

    return np.asarray(merged, dtype=float)
def _point_to_surface_distance_rsun(points_xyz: np.ndarray, surf: pv.PolyData) -> np.ndarray:
    """
    Return |distance| from each point to the surface (Rsun units).

    Uses PyVista compute_implicit_distance if available; falls back to VTK otherwise.
    """
    points_xyz = np.asarray(points_xyz, dtype=float)
    if points_xyz.size == 0:
        return np.zeros((0,), dtype=float)

    try:
        pd = pv.PolyData(points_xyz)
        out = pd.compute_implicit_distance(surf)
        d = np.asarray(out["implicit_distance"], dtype=float)
        return np.abs(d)
    except Exception:
        import vtk
        ipd = vtk.vtkImplicitPolyDataDistance()
        ipd.SetInput(surf)
        d = np.empty((points_xyz.shape[0],), dtype=float)
        for i, p in enumerate(points_xyz):
            d[i] = abs(float(ipd.EvaluateFunction(p.tolist())))
        return d


def add_gcs_tomography_overlap_points(
    plotter,
    *,
    gcs_cross_points_hgc: np.ndarray,
    tomo_isosurfaces: list,
    tol_rsun: float = 0.05,
    color: str = "black",
    point_size: int = 14,
):
    """
    GCS (meridian×parallel) 交点候補のうち、tomography 等値面に近い点を抽出してプロットする。

    Parameters
    ----------
    gcs_cross_points_hgc : (N,3)
        HGC座標の交点候補
    tomo_isosurfaces : list[pv.PolyData]
        add_isosurfaces(..., return_surfaces=True) が返した等値面
    tol_rsun : float
        「重なり」の許容距離（Rsun）
    """
    pts = np.asarray(gcs_cross_points_hgc, dtype=float)
    if pts.size == 0 or tomo_isosurfaces is None or len(tomo_isosurfaces) == 0:
        print("[INFO] No overlap points (no GCS cross points or no tomo isosurfaces).")
        return {"n_candidates": int(pts.shape[0]), "n_overlap": 0}

    tol = float(tol_rsun)
    hit = np.zeros((pts.shape[0],), dtype=bool)

    for surf in tomo_isosurfaces:
        if surf is None or getattr(surf, "n_points", 0) == 0:
            continue
        d = _point_to_surface_distance_rsun(pts, surf)
        hit |= (d <= tol)

    pts_hit = pts[hit]
    if pts_hit.shape[0] == 0:
        print("[INFO] No GCS×tomo overlap points found within tol.")
        return {"n_candidates": int(pts.shape[0]), "n_overlap": 0}

    plotter.add_mesh(
        pv.PolyData(pts_hit),
        color=color,
        point_size=int(point_size),
        render_points_as_spheres=True,
        pickable=False,
    )
    print(f"[INFO] GCS×tomo overlap points: {pts_hit.shape[0]} (tol={tol:g} Rsun)")
    return {"n_candidates": int(pts.shape[0]), "n_overlap": int(pts_hit.shape[0])}


# -----------------------------------------------------------------------------
# PFSS helpers (HMI -> PFSS -> 3D field-line overlay)
# -----------------------------------------------------------------------------
def compute_pfss_output_from_hmi(
    hmi_fits: str,
    *,
    nrho: int = 50,
    rss: float = 2.5,
    helio_shape: Tuple[int, int] = (180, 360),
    fill_nan: float = 0.0,
):
    """
    Compute PFSS solution from a single full-disk HMI magnetogram.

    Key stability points:
      - Limit threads (WSL/OpenMPの不安定化回避)
      - Import pfsspy via _safe_import_pfsspy() (numbaブロックを一貫させる)
    """
    # Limit threading to reduce segfault risk on some WSL/OpenMP stacks.
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    pfsspy, utils, _ = _safe_import_pfsspy()

    hmi = sunpy.map.Map(hmi_fits)

    nlat, nlon = int(helio_shape[0]), int(helio_shape[1])
    if nlat < 2 or nlon < 2:
        raise ValueError(f"helio_shape must be >= (2,2); got {helio_shape}")

    # --- Build a CAR header that spans exactly 360 deg in longitude and 180 deg in latitude ---
    cdelt1 = 360.0 / float(nlon)
    cdelt2 = 180.0 / float(nlat)

    car_header = {
        "NAXIS": 2,
        "NAXIS1": nlon,
        "NAXIS2": nlat,
        "CTYPE1": "CRLN-CAR",
        "CTYPE2": "CRLT-CAR",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CDELT1": cdelt1,
        "CDELT2": cdelt2,
        "CRPIX1": (nlon / 2.0) + 0.5,
        "CRPIX2": (nlat / 2.0) + 0.5,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "LONPOLE": 180.0,
        "DATE-OBS": getattr(hmi.date, "isot", str(hmi.date)),
    }

    # Propagate useful metadata if present.
    for key in (
        "HGLN_OBS", "HGLT_OBS", "DSUN_OBS", "RSUN_REF", "RSUN_OBS",
        "CRLN_OBS", "CRLT_OBS", "SOLAR_B0", "SOLAR_P0"
    ):
        if key in hmi.meta:
            car_header[key] = hmi.meta[key]

    if "RSUN_REF" not in car_header and "rsun_ref" in hmi.meta:
        car_header["RSUN_REF"] = hmi.meta["rsun_ref"]
    if "DSUN_OBS" not in car_header and "dsun_obs" in hmi.meta:
        car_header["DSUN_OBS"] = hmi.meta["dsun_obs"]

    # ---- Reproject to CAR ----
    bmap_car = hmi.reproject_to(car_header)

    car_data = np.array(bmap_car.data, dtype=float)
    car_data[~np.isfinite(car_data)] = float(fill_nan)
    bmap_car = sunpy.map.Map(car_data, bmap_car.meta)

    # ---- Convert CAR -> CEA (pfsspy requirement) ----
    cea_out = utils.car_to_cea(bmap_car)
    if isinstance(cea_out, tuple) and len(cea_out) == 2:
        cea_data, cea_meta = cea_out
        bmap_cea = sunpy.map.Map(cea_data, cea_meta)
    else:
        bmap_cea = cea_out

    cea_data = np.array(bmap_cea.data, dtype=float)
    cea_data[~np.isfinite(cea_data)] = float(fill_nan)
    bmap_cea = sunpy.map.Map(cea_data, bmap_cea.meta)

    # ---- PFSS solve ----
    rss_val = float(rss)
    try:
        pfss_input = pfsspy.Input(bmap_cea, nrho=nrho, rss=rss_val)
    except TypeError as e:
        if "unexpected keyword argument" in str(e) and "nrho" in str(e):
            try:
                pfss_input = pfsspy.Input(bmap_cea, nrho, rss=rss_val)
            except TypeError:
                pfss_input = pfsspy.Input(bmap_cea, nrho, rss_val)
        else:
            raise

    pfss_output = pfsspy.pfss(pfss_input)
    return pfss_output, bmap_cea


def build_visible_hemisphere_seeds(
    pfss_output,
    obs0,
    *,
    n_lon: int = 30,
    n_lat: int = 15,
    lat_max_deg: float = 80.0,
    r_seed_rsun: float = 1.001,   # ★球面(=1.0)からわずかに浮かせて z-fighting 回避
):
    """
    Build seed points on (slightly above) the solar surface.

    Important fix:
      - r_seed_rsun=1.001 to avoid z-fighting with the drawn photosphere sphere (radius=1.0).
      - We interleave front/back hemispheres so that even if the observer vector sign is
        inconsistent in obs0, some field lines will still be visible.
    """
    x_hat = sun_to_observer_unit_vector(obs0)

    lons = np.linspace(0.0, 360.0, int(n_lon), endpoint=False)
    lats = np.linspace(-float(lat_max_deg), float(lat_max_deg), int(n_lat))
    Lon, Lat = np.meshgrid(lons, lats, indexing="xy")

    lon_rad = np.deg2rad(Lon.ravel())
    lat_rad = np.deg2rad(Lat.ravel())

    ux = np.cos(lat_rad) * np.cos(lon_rad)
    uy = np.cos(lat_rad) * np.sin(lon_rad)
    uz = np.sin(lat_rad)
    dot = ux * x_hat[0] + uy * x_hat[1] + uz * x_hat[2]

    # --- Interleave front/back seeds to be robust to sign conventions ---
    idx_pos = np.where(dot >= 0.0)[0]
    idx_neg = np.where(dot < 0.0)[0]

    # Sort: front-most first, back-most first
    idx_pos = idx_pos[np.argsort(dot[idx_pos])[::-1]]
    idx_neg = idx_neg[np.argsort(dot[idx_neg])]  # more negative first

    # Interleave
    order = []
    i = j = 0
    while i < idx_pos.size or j < idx_neg.size:
        if i < idx_pos.size:
            order.append(idx_pos[i])
            i += 1
        if j < idx_neg.size:
            order.append(idx_neg[j])
            j += 1
    order = np.asarray(order, dtype=int)

    seed_lon = Lon.ravel()[order] * u.deg
    seed_lat = Lat.ravel()[order] * u.deg
    seed_r = float(r_seed_rsun) * u.R_sun

    try:
        frame = pfss_output.coordinate_frame
        seeds = SkyCoord(seed_lon, seed_lat, seed_r, frame=frame)
    except Exception:
        t = pfss_output.map.date if hasattr(pfss_output, "map") else None
        try:
            frame = frames.HeliographicCarrington(obstime=t, observer="earth")
        except TypeError:
            frame = frames.HeliographicCarrington(obstime=t)
        seeds = SkyCoord(seed_lon, seed_lat, seed_r, frame=frame)

    return seeds
def hmi_roi_pixels_like_2d_script(hmi_map):
    """
    2Dコード（plot_hmi_pfss_overlay.py の prepare_hmi_for_pfss）と同じ ROI を返す。
    具体的には:
      x: [center_x-512, center_x+0]
      y: [center_y-100, center_y+512]

    Returns
    -------
    (x_lims_pix, y_lims_pix) : tuple(tuple(int,int), tuple(int,int))
    """
    data = hmi_map.data
    ny, nx = data.shape
    center_x, center_y = nx // 2, ny // 2

    x_min_pix, x_max_pix = center_x - 512, center_x + 0
    y_min_pix, y_max_pix = center_y - 100, center_y + 512

    # 画像境界にクリップ
    x_min_pix = int(np.clip(x_min_pix, 0, nx - 1))
    x_max_pix = int(np.clip(x_max_pix, 1, nx))
    y_min_pix = int(np.clip(y_min_pix, 0, ny - 1))
    y_max_pix = int(np.clip(y_max_pix, 1, ny))

    # 念のため順序保証
    if x_max_pix <= x_min_pix:
        x_min_pix, x_max_pix = 0, nx
    if y_max_pix <= y_min_pix:
        y_min_pix, y_max_pix = 0, ny

    return (x_min_pix, x_max_pix), (y_min_pix, y_max_pix)

def build_pfss_seeds_from_hmi_roi(
    hmi_map,
    pfss_output,
    *,
    x_lims_pix,
    y_lims_pix,
    n_seeds_x: int = 20,
    n_seeds_y: int = 20,
    use_strong_field: bool = True,
    field_threshold: float = 200.0,
    margin_pix: int = 50,
    r_seed_rsun: float = 1.001,
    rng_seed: int = 42,
):
    """
    2Dコードの define_field_line_seeds() と同等の seed 生成を、3D PFSS 用に提供する。

    - ROI (x_lims_pix, y_lims_pix) 内に限定
    - use_strong_field=True の場合、|B| > field_threshold の画素からランダムサンプル
    - use_strong_field=False の場合、ROI内に均等グリッド
    - pixel_to_world → pfss_output.coordinate_frame へ変換
    - 半径は r_seed_rsun * R_sun に固定（z-fighting回避）

    Returns
    -------
    seeds : astropy.coordinates.SkyCoord  (frame = pfss_output.coordinate_frame)
    """
    xmin, xmax = map(int, x_lims_pix)
    ymin, ymax = map(int, y_lims_pix)

    if use_strong_field:
        roi = np.array(hmi_map.data[ymin:ymax, xmin:xmax], dtype=float)
        abs_roi = np.abs(roi)
        mask = np.isfinite(abs_roi) & (abs_roi > float(field_threshold))
        yy, xx = np.where(mask)

        if xx.size == 0:
            # 強磁場が無ければグリッドへフォールバック
            use_strong_field = False
        else:
            n_target = int(min(n_seeds_x * n_seeds_y, xx.size))
            rng = np.random.default_rng(int(rng_seed))
            if xx.size > n_target:
                pick = rng.choice(xx.size, size=n_target, replace=False)
                xx = xx[pick]
                yy = yy[pick]

            x_pixels = xx + xmin
            y_pixels = yy + ymin

    if not use_strong_field:
        # ROI内均等グリッド（端は margin_pix だけ避ける：2Dコードと同じ意図）
        x0 = xmin + int(margin_pix)
        x1 = xmax - int(margin_pix)
        y0 = ymin + int(margin_pix)
        y1 = ymax - int(margin_pix)

        # margin が効きすぎた場合は ROI 全体を使う
        if x1 <= x0:
            x0, x1 = xmin, xmax
        if y1 <= y0:
            y0, y1 = ymin, ymax

        x_1d = np.linspace(x0, x1, int(n_seeds_x))
        y_1d = np.linspace(y0, y1, int(n_seeds_y))
        X, Y = np.meshgrid(x_1d, y_1d, indexing="xy")
        x_pixels = X.ravel()
        y_pixels = Y.ravel()

    # pixel -> world (HPC) -> PFSS frame
    seeds_hpc = hmi_map.pixel_to_world(x_pixels * u.pixel, y_pixels * u.pixel)
    if not isinstance(seeds_hpc, SkyCoord):
        seeds_hpc = SkyCoord(seeds_hpc)

    # pfss_output の座標系へ
    try:
        frame = pfss_output.coordinate_frame
        seeds_pfss = seeds_hpc.transform_to(frame)
    except Exception:
        # 最終手段：変換できない点がある場合、まずSkyCoord化してからtransform
        seeds_pfss = SkyCoord(seeds_hpc).transform_to(pfss_output.coordinate_frame)

    # off-disk などで lon/lat が NaN になる点を除外し、半径を固定
    lon = getattr(seeds_pfss, "lon", None)
    lat = getattr(seeds_pfss, "lat", None)
    if lon is None or lat is None:
        raise RuntimeError("Seeds could not be represented with lon/lat in the PFSS coordinate frame.")

    lon = lon.to(u.deg)
    lat = lat.to(u.deg)
    good = np.isfinite(lon.value) & np.isfinite(lat.value)

    if np.count_nonzero(good) == 0:
        raise RuntimeError("All ROI seeds became invalid after transforming to PFSS frame (off-disk / WCS issue).")

    seeds_out = SkyCoord(
        lon[good],
        lat[good],
        float(r_seed_rsun) * u.R_sun,
        frame=pfss_output.coordinate_frame,
    )

    return seeds_out

def add_pfss_from_hmi_3d_roi_seeds(
    plotter,
    obs0,
    *,
    hmi_fits: str,
    rss: float = 2.5,
    nrho: int = 50,
    helio_shape: Tuple[int, int] = (180, 360),
    x_lims_pix=None,
    y_lims_pix=None,
    n_seeds_x: int = 30,
    n_seeds_y: int = 30,
    use_strong_field: bool = True,
    field_threshold: float = 200.0,
    max_lines: int = 300,
    tracer_step_size: float = 0.01,
    tracer_max_steps: int = 20000,
    line_width: int = 2,
    opacity: float = 1.0,
    open_color: str = "red",
    closed_color: str = "black",
    prefer_fortran: bool = True,
):
    """
    2DのPFSS（HMI ROI + 強磁場 seed）と同条件の seed を用いて、3D（PyVista）へ PFSS を描画する。

    - seed は build_pfss_seeds_from_hmi_roi() で生成
    - open/closed は fline.open を優先し、無ければ終端半径でフォールバック
    - 色は open=open_color, closed=closed_color
    """
    # PFSS解
    _, _, tracing = _safe_import_pfsspy()
    pfss_output, _ = compute_pfss_output_from_hmi(
        hmi_fits,
        nrho=nrho,
        rss=rss,
        helio_shape=helio_shape,
    )

    # HMI map（seed作成用）
    hmi_map = sunpy.map.Map(hmi_fits)

    # ROIが指定されなければ 2Dコードと同じ ROI を採用
    if x_lims_pix is None or y_lims_pix is None:
        x_lims_pix, y_lims_pix = hmi_roi_pixels_like_2d_script(hmi_map)

    # seeds（2D同様）
    seeds = build_pfss_seeds_from_hmi_roi(
        hmi_map,
        pfss_output,
        x_lims_pix=x_lims_pix,
        y_lims_pix=y_lims_pix,
        n_seeds_x=int(n_seeds_x),
        n_seeds_y=int(n_seeds_y),
        use_strong_field=bool(use_strong_field),
        field_threshold=float(field_threshold),
        r_seed_rsun=1.001,
        rng_seed=42,
    )

    # tracer選択（2Dに合わせるなら FortranTracer 推奨）
    tracer = None
    tracer_used = None

    def _try_fortran_tracer():
        if not hasattr(tracing, "FortranTracer"):
            return None, None
        safe_step = float(tracer_step_size)
        if safe_step < 0.25:
            safe_step = 1.0
        try:
            t = tracing.FortranTracer(max_steps="auto", step_size=safe_step)
            return t, f"FortranTracer(max_steps='auto', step_size={safe_step:g})"
        except TypeError:
            safe_max_steps = int(min(int(tracer_max_steps), 2000))
            t = tracing.FortranTracer(max_steps=safe_max_steps, step_size=safe_step)
            return t, f"FortranTracer(max_steps={safe_max_steps}, step_size={safe_step:g})"
        except Exception:
            return None, None

    def _try_python_tracer():
        if not hasattr(tracing, "PythonTracer"):
            return None, None
        try:
            t = tracing.PythonTracer(atol=1e-4, rtol=1e-4)
            return t, "PythonTracer"
        except Exception:
            return None, None

    if prefer_fortran:
        tracer, tracer_used = _try_fortran_tracer()
        if tracer is None:
            tracer, tracer_used = _try_python_tracer()
    else:
        tracer, tracer_used = _try_python_tracer()
        if tracer is None:
            tracer, tracer_used = _try_fortran_tracer()

    if tracer is None:
        raise RuntimeError("No available PFSS tracer backend (neither PythonTracer nor FortranTracer).")

    print(f"[INFO] PFSS tracer (ROI-seeds): {tracer_used}")
    print(f"[INFO] ROI pixels: x={x_lims_pix}, y={y_lims_pix}, strong={use_strong_field}, thr={field_threshold:g} G, seeds={len(seeds)}")

    n_open = 0
    n_closed = 0
    n_lines = 0
    lw = max(2, int(line_width))

    for seed in seeds:
        if n_lines >= int(max_lines):
            break

        try:
            fl_container = tracer.trace(seed, pfss_output)
        except Exception:
            continue
        if fl_container is None:
            continue

        flines = getattr(fl_container, "field_lines", None)
        if flines is None:
            flines = [fl_container]

        for fline in flines:
            if n_lines >= int(max_lines):
                break

            # open/closed 判定
            is_open = None
            if hasattr(fline, "open"):
                is_open = bool(fline.open)
            elif hasattr(fline, "is_open"):
                is_open = bool(fline.is_open)

            # 座標
            try:
                sc = getattr(fline, "coords", getattr(fline, "coordinates", None))
                if sc is None:
                    continue
                xyz = sc.cartesian.xyz.to_value(u.R_sun).T
            except Exception:
                continue

            if xyz.ndim != 2 or xyz.shape[0] < 2 or xyz.shape[1] != 3:
                continue
            if not np.all(np.isfinite(xyz)):
                continue
            if np.nanmax(np.abs(xyz)) > 1.0e3:
                continue

            # open 判定が無ければ終端半径でフォールバック
            if is_open is None:
                try:
                    end_r = sc.radius[-1].to_value(u.R_sun)
                    is_open = bool(end_r > 2.0)
                except Exception:
                    is_open = False

            if is_open:
                n_open += 1
                color = open_color
            else:
                n_closed += 1
                color = closed_color

            plotter.add_mesh(
                _polyline_from_points(xyz),
                color=color,
                line_width=lw,
                opacity=float(opacity),
                lighting=False,
                render_lines_as_tubes=True,
                pickable=False,
            )
            n_lines += 1

    print(f"[INFO] PFSS(ROI-seeds) lines: total={n_lines}, open={n_open}, closed={n_closed}")

    return {
        "rss": float(rss),
        "nrho": int(nrho),
        "n_lines": int(n_lines),
        "n_open": int(n_open),
        "n_closed": int(n_closed),
        "tracer": tracer_used,
        "seed_mode": "HMI_ROI_strong_field" if use_strong_field else "HMI_ROI_grid",
        "field_threshold_G": float(field_threshold),
        "x_lims_pix": tuple(map(int, x_lims_pix)),
        "y_lims_pix": tuple(map(int, y_lims_pix)),
    }


def add_pfss_from_hmi_3d(
    plotter,
    obs0,
    *,
    hmi_fits: str,
    rss: float = 2.5,
    nrho: int = 50,
    helio_shape: Tuple[int, int] = (180, 360),
    seed_n_lon: int = 20,     # ★増やす
    seed_n_lat: int = 20,     # ★増やす
    max_lines: int = 300,     # ★増やす
    tracer_step_size: float = 0.01,
    tracer_max_steps: int = 20000,
    line_width: int = 2,
    opacity: float = 1.0,
    prefer_fortran: bool = False,
):
    """
    Compute PFSS from HMI and overlay field lines in PyVista.

    Fixes / conventions:
      - Render lines as tubes (render_lines_as_tubes=True) to avoid point-like artifacts.
      - Use slightly-off-surface seeds (handled in build_visible_hemisphere_seeds).
      - Color convention: open=red, closed=black.
    """
    _, _, tracing = _safe_import_pfsspy()

    pfss_output, _ = compute_pfss_output_from_hmi(
        hmi_fits,
        nrho=nrho,
        rss=rss,
        helio_shape=helio_shape,
    )

    seeds = build_visible_hemisphere_seeds(
        pfss_output,
        obs0,
        n_lon=int(seed_n_lon),
        n_lat=int(seed_n_lat),
        lat_max_deg=80.0,
        r_seed_rsun=1.001,
    )
    if seeds is None or len(seeds) == 0:
        raise RuntimeError("No PFSS seeds were generated.")

    # --- tracer selection ---
    tracer = None
    tracer_used = None

    def _try_fortran_tracer():
        if not hasattr(tracing, "FortranTracer"):
            return None, None
        safe_step = float(tracer_step_size)
        if safe_step < 0.25:
            safe_step = 1.0
        try:
            t = tracing.FortranTracer(max_steps="auto", step_size=safe_step)
            return t, f"FortranTracer(max_steps='auto', step_size={safe_step:g})"
        except TypeError:
            safe_max_steps = int(min(int(tracer_max_steps), 2000))
            t = tracing.FortranTracer(max_steps=safe_max_steps, step_size=safe_step)
            return t, f"FortranTracer(max_steps={safe_max_steps}, step_size={safe_step:g})"
        except Exception:
            return None, None

    def _try_python_tracer():
        if not hasattr(tracing, "PythonTracer"):
            return None, None
        try:
            t = tracing.PythonTracer(atol=1e-4, rtol=1e-4)
            return t, "PythonTracer"
        except Exception:
            return None, None

    if prefer_fortran:
        tracer, tracer_used = _try_fortran_tracer()
        if tracer is None:
            tracer, tracer_used = _try_python_tracer()
    else:
        tracer, tracer_used = _try_python_tracer()
        if tracer is None:
            tracer, tracer_used = _try_fortran_tracer()

    if tracer is None:
        raise RuntimeError("No available PFSS tracer backend (neither PythonTracer nor FortranTracer).")

    print(f"[INFO] PFSS tracer: {tracer_used}")

    n_open = 0
    n_closed = 0
    n_lines = 0

    lw = max(2, int(line_width))
    a_open = float(opacity)
    a_closed = float(opacity)

    for seed in seeds:
        if n_lines >= int(max_lines):
            break

        try:
            fl_container = tracer.trace(seed, pfss_output)
        except Exception:
            continue
        if fl_container is None:
            continue

        flines = getattr(fl_container, "field_lines", None)
        if flines is None:
            flines = [fl_container]

        for fline in flines:
            if n_lines >= int(max_lines):
                break

            is_open = bool(getattr(fline, "open", getattr(fline, "is_open", False)))

            try:
                sc = getattr(fline, "coords", getattr(fline, "coordinates", None))
                if sc is None:
                    continue
                xyz = sc.cartesian.xyz.to_value(u.R_sun).T  # (N,3)
            except Exception:
                continue

            if xyz.ndim != 2 or xyz.shape[0] < 2 or xyz.shape[1] != 3:
                continue
            if not np.all(np.isfinite(xyz)):
                continue
            if np.nanmax(np.abs(xyz)) > 1.0e3:
                continue

            color = "red" if is_open else "black"  # ★要望どおり
            if is_open:
                n_open += 1
                alpha = a_open
            else:
                n_closed += 1
                alpha = a_closed

            poly = _polyline_from_points(xyz)

            plotter.add_mesh(
                poly,
                color=color,
                line_width=lw,
                opacity=alpha,
                lighting=False,
                render_lines_as_tubes=True,
                pickable=False,
            )
            n_lines += 1

    if n_lines == 0:
        print("[WARN] PFSS produced zero field lines (check seeds and tracer output).")
    else:
        print(f"[INFO] PFSS lines plotted: total={n_lines}, open={n_open}, closed={n_closed}")

    return {
        "rss": float(rss),
        "nrho": int(nrho),
        "n_lines": int(n_lines),
        "n_open": int(n_open),
        "n_closed": int(n_closed),
        "tracer": tracer_used,
    }


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main(Frequency_MHz: List[float], time_iso: str, h_apex: float, kappa: float, alpha_deg: float, tilt_deg: float, lon_deg: float, lat_deg: float, rss: float, r_scatter: float):
    # ---- Import modules ----
    TOMO_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/py_folder")
    GCS_PKG_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/GCS/gcs_overlay")
    # TOMO_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/Tomography\py_folder")
    # GCS_PKG_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/GCS\gcs_overlay")

    OBSTIME_ISO = time_iso

    R_MIN, R_MAX = 1.5, 4
    NR, NTH, NPH = 250, 180, 360
    DS = 0.01
    LAM = 1e-2
    MAXITER, TOL = 100, 1e-6
    WT_R = None

    OUT_N = 512

    R_USE_MIN, R_USE_MAX = 1.5, 4.0
    U_KCOR  = 0.4530
    U_LASCO = 0.6135
    U_COR1A = 0.56
    LIMB_U_DEFAULT = 0.6135
    R_SWITCH_U = 2.2

    PB_FITS_LIST = [
        {
            "path": "/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/pB_Kcor_LASCO_axi_20220613_0300.fits",
            # "path": "/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/pB_Kcor_LASCO_axi_20220613_0300.fits",
            "limb_u": U_KCOR,
            "r_use_min": R_USE_MIN,
            "r_use_max": R_SWITCH_U,
            "tag": "KCOR-part",
        },
        {
            "path": "/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/pB_Kcor_LASCO_axi_20220613_0300.fits",
            # "path": "/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/pB_Kcor_LASCO_axi_20220613_0300.fits",
            "limb_u": U_LASCO,
            "r_use_min": R_SWITCH_U,
            "r_use_max": R_USE_MAX,
            "tag": "LASCO-part",
        },
        {
            "path": "/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/pB_Kcor_LASCO_axi_20220613_0300.fits",
            # "path": "/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/COR1A_pb_pre_20220613_030100.fits",
            "limb_u": U_COR1A,
            "r_use_min": R_USE_MIN,
            "r_use_max": R_USE_MAX,
            "tag": "COR1A",
        },
    ]

    PB_FLOOR = "auto"
    APPLY_SPATIAL_DESPIKE = True
    DESPIKE_NSIG, DESPIKE_MED = 6.0, 3
    FILT = False
    DPA_DEG = 1.0
    HM = 3
    WIDTH_PIX = 0
    Q_LOW = 10.0

    ISO_FREQ_MHZ = Frequency_MHz
    HARMONIC = 2

    GCS_H_APEX = h_apex
    GCS_KAPPA = kappa
    GCS_ALPHA_DEG = alpha_deg
    GCS_TILT_DEG = tilt_deg
    GCS_LON_DEG = lon_deg
    GCS_LAT_DEG = lat_deg
    # Force Carrington observer long/lat for tomography geometry (avoid accidental Stonyhurst fallback)

    LONLAT_OVERRIDE_MAP = _build_lonlat_override_map_carrington(PB_FITS_LIST)
    if LONLAT_OVERRIDE_MAP:
        print("[INFO] lonlat_override map (Carrington: CRLN_OBS/CRLT_OBS in deg):")
        for _k, (_lon, _lat) in LONLAT_OVERRIDE_MAP.items():
            print(f"  - {_k}: lon={_lon:.6f} deg, lat={_lat:.6f} deg")
    LONLAT_DEFAULT = None
    GCS_OBSERVER = "earth"

    DO_PFSS = True
    HMI_FITS = "/mnt/d/wsl/home/kinno-7010/Research_data/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    # HMI_FITS = "/mnt/d/wsl/home/kinno-7010/Research_data/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    PFSS_RSS = rss
    PFSS_NRHO = 40

    # ★PFSS本数を増やす（seed密度＆描画上限）
    PFSS_SEED_N_LON = 15
    PFSS_SEED_N_LAT = 15
    PFSS_MAX_LINES = 150
    PFSS_FIELD_THRESHOLD = 150.0

    PFSS_TRACER_STEP = 0.01
    PFSS_TRACER_MAX_STEPS = 20000

    SHOW_SUN = True
    SHOW_GUI = True
    SAVE_PNG = True
    PNG_PATH = Path(f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/output/overlay_tomo_gcs_{''.join(str(f) for f in ISO_FREQ_MHZ)}MHz_highres.png")
    # PNG_PATH = Path(f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography\output\overlay_tomo_gcs_{''.join(str(f) for f in ISO_FREQ_MHZ)}MHz_highres.png")


    # ---- Build grid ----
    r_edges = np.linspace(R_MIN, R_MAX, NR + 1)
    th_edges = np.linspace(0.0, np.pi, NTH + 1)
    ph_edges = np.linspace(0.0, 2.0 * np.pi, NPH + 1)
    grid = SphericalGrid(r_edges=r_edges, th_edges=th_edges, ph_edges=ph_edges)

    # ---- Load observations ----
    obs_list = []
    rays_list = []
    y_list = []

    for item in PB_FITS_LIST:
        if isinstance(item, dict):
            fp = str(item.get("path"))
            limb_u = float(item.get("limb_u", LIMB_U_DEFAULT))
            r_use_min_i = float(item.get("r_use_min", R_USE_MIN))
            r_use_max_i = float(item.get("r_use_max", R_USE_MAX))
            tag = str(item.get("tag", ""))
        else:
            fp = str(item)
            limb_u = float(LIMB_U_DEFAULT)
            r_use_min_i = float(R_USE_MIN)
            r_use_max_i = float(R_USE_MAX)
            tag = ""

        tag_txt = f" [{tag}]" if tag else ""
        print(f"Reading {fp}{tag_txt} (r_use={r_use_min_i:.2f}-{r_use_max_i:.2f} Rs, limb_u={limb_u:.3f})...")

        obs = build_observation(
            Path(fp),
            out_n=OUT_N,
            r_use_min=r_use_min_i,
            r_use_max=r_use_max_i,
            limb_u=limb_u,
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

        y_vec = obs.pb.ravel()[obs.idx_map].astype(np.float64)
        y_list.append(y_vec)

        rays = build_rays_for_observation(
            obs=obs,
            grid=grid,
            ds_rsun=DS,
            r_min=R_MIN,
            r_max=R_MAX,
            limb_u=limb_u,
        )
        rays_list.append(rays)
        obs_list.append(obs)

    y_obs = np.concatenate(y_list) if y_list else np.array([], dtype=np.float64)

    # ---- Solve tomography ----
    tomo = RegularizedTomography(grid, obs_list, rays_list, lam=LAM, wt_r=WT_R)
    ne_raw, info = tomo.solve(y_obs, maxiter=MAXITER, tol=TOL, positivity=True)

    ne_raw, scale = apply_brightness_scale_like_main(tomo, ne_raw, y_obs)
    print(f"[INFO] brightness scale applied: {scale:.3e}")
    if info != 0:
        print(f"[WARN] Tomography CG did not fully converge (info={info}). Consider stronger regularization or more images.")

    ne_vals = np.asarray(ne_raw, dtype=float)
    m = np.isfinite(ne_vals)
    if np.any(m):
        print(f"[INFO] ne range: min={ne_vals[m].min():.3e}, max={ne_vals[m].max():.3e}")
    else:
        print("[WARN] ne contains no finite values.")

    print(f"[INFO] ISO_FREQ_MHZ={ISO_FREQ_MHZ}, harmonic={HARMONIC}")

    # ---- Build PyVista grid ----
    sg = build_tomography_structured_grid(grid, ne_raw)
    print(f"[INFO] StructuredGrid bounds: {sg.bounds}")

    # ---- Plot ----
    off_screen = (not SHOW_GUI)
    if SHOW_GUI and not os.environ.get("DISPLAY"):
        print("[WARN] DISPLAY not set; forcing off-screen rendering.")
        off_screen = True
        try:
            pv.start_xvfb()
        except Exception as e:
            print(f"[WARN] pv.start_xvfb failed: {e}")

    p = pv.Plotter(off_screen=off_screen)
    p.set_background("white")

    try:
        p.enable_depth_peeling()
    except Exception:
        pass
    try:
        p.enable_anti_aliasing("ssaa")
    except Exception:
        pass

    obs0 = obs_list[0]

    if SHOW_SUN:
        p.add_mesh(
            pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60),
            opacity=0.2,
            color="grey",
        )
    
    if ISO_FREQ_MHZ == [33.8]:
        colors = ["red"]
    elif ISO_FREQ_MHZ == [31.5]:
        colors = ["gold"]
    elif ISO_FREQ_MHZ == [28.0]:
        colors = ["cyan"]

    tomo_surfs = add_isosurfaces(p, sg, ISO_FREQ_MHZ, harmonic=HARMONIC, opacity=0.2, colors=colors,
                             return_surfaces=True, range_text_mode="runinfo")


    add_solar_latlon_grid(p, radius=1.002, dlon_deg=30.0, dlat_deg=30.0, line_width=2, opacity=0.6)
    add_sun_earth_line(p, obs0, length_rsun=5.0, start_rsun=1.0, color="orange", line_width=5)

    # ★軸を短く・細く
    add_physical_axes_triad(
        p, obs0,
        origin_rsun=1.0,
        axis_len=1.6,
        shaft_radius=0.04,
        tip_radius=0.08,
        tip_length=0.25,
        label_font_size=12,
    )

    pfss_info = None
    if DO_PFSS:
        try:
            pfss_info = add_pfss_from_hmi_3d_roi_seeds(
                p,
                obs0,
                hmi_fits=HMI_FITS,
                rss=PFSS_RSS,
                nrho=PFSS_NRHO,
                n_seeds_x=PFSS_SEED_N_LON,
                n_seeds_y=PFSS_SEED_N_LAT,
                use_strong_field=True,
                field_threshold=PFSS_FIELD_THRESHOLD,
                max_lines=PFSS_MAX_LINES,
                line_width=2,
                opacity=1.0,
                open_color="red",
                closed_color="black",
                prefer_fortran=True,
            )

        except Exception as e:
            print(f"[WARN] PFSS overlay skipped due to error: {e}")
            pfss_info = None

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
        pfss_params=pfss_info,
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
    gcs_cross = compute_gcs_meridian_parallel_crosspoints_hgc(
        curves,
        obstime_iso=OBSTIME_ISO,
        observer=GCS_OBSERVER,
        n_resample=200,
        cross_tol_rsun=0.05,
        merge_tol_rsun=0.03,
    )

    add_gcs_tomography_overlap_points(
        p,
        gcs_cross_points_hgc=gcs_cross,
        tomo_isosurfaces=tomo_surfs,
        tol_rsun=0.10,
        color="black",
        point_size=14,
    )   


    # for key in ("parallels", "meridians", "legs"):
    #     for pts in curves.get(key, []):
    #         pts_hgc = transform_points_hgs_to_hgc(
    #             np.asarray(pts),
    #             obstime_iso=OBSTIME_ISO,
    #             observer=GCS_OBSERVER,
    #         )
    #         p.add_mesh(_polyline_from_points(pts_hgc), color="lime", line_width=2)
    gcs_info = add_gcs_shell_surface(
        p,
        params,
        obstime_iso=OBSTIME_ISO,
        observer=GCS_OBSERVER,
        color="lime",
        opacity=0.3,
        n_u=120,
        closed_u=True,
        show_wireframe=True,
        wire_color="lime",
        wire_width=2,
        return_surface=True,   # ★追加
    )

    if gcs_info and "surface" in gcs_info:
        add_gcs_surface_at_radius(
            p,
            gcs_info["surface"],
            r0=r_scatter,      # ★ここを任意に変更
            dr=0.05,     # ★帯の厚み（まずは 0.03〜0.10 Rs 程度）
            mode="band", # "band" or "scatter"
            color="yellow",
            opacity=0.8,
        )

    if gcs_info is not None:
        print(f"[INFO] GCS surface rendered: n_points={gcs_info['n_points']}, n_cells={gcs_info['n_cells']}")
    else:
        print("[WARN] GCS surface not rendered.")


    # ---- Camera (robust) ----
    set_camera_from_observation(p, obs0, distance_rsun=4.0)

    try:
        p.reset_camera_clipping_range()
        p.render()
    except Exception:
        pass

    # ---- Output ----
    if SAVE_PNG:
        PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # レンダリングせずscreenshotのみ保存（plt.showせずにpngを保存）
        p.off_screen = True
        p.show(screenshot=str(PNG_PATH), auto_close=False)
        print(f"[OK] Saved: {PNG_PATH}")
        # 画面表示も行う
        p.off_screen = False
        p.show()
    else:
        p.show()


if __name__ == "__main__":
    import gc
    
    time_iso = "2022-06-13T03:25:29"
    Frequency_MHz = [33.8]
    h_apex = 3.39
    kappa = 0.10
    alpha_deg = 20.0
    tilt_deg = 87.0
    lon_deg = -44.0
    lat_deg = 10.0
    rss = 2.5
    r_scatter = 2.95
    main(Frequency_MHz=Frequency_MHz, time_iso=time_iso, h_apex=h_apex, kappa=kappa, alpha_deg=alpha_deg, tilt_deg=tilt_deg, lon_deg=lon_deg, lat_deg=lat_deg, rss=rss, r_scatter=r_scatter)
    
    # メモリ解放
    # gc.collect()
        
    # time_iso = "2022-06-13T03:28:46"
    # Frequency_MHz = [31.5]
    # h_apex = 3.63
    # kappa = 0.10
    # alpha_deg = 22.0
    # tilt_deg = 87.0
    # lon_deg = -44.0
    # lat_deg = 10.0
    # rss = 2.5
    # r_scatter = 3.08
    # main(Frequency_MHz=Frequency_MHz, time_iso=time_iso, h_apex=h_apex, kappa=kappa, alpha_deg=alpha_deg, tilt_deg=tilt_deg, lon_deg=lon_deg, lat_deg=lat_deg, rss=rss, r_scatter=r_scatter)
    # # メモリ解放
    # gc.collect()
    
    
    # time_iso = "2022-06-13T03:31:17"
    # Frequency_MHz = [28.0]
    # h_apex = 3.81
    # kappa = 0.10
    # alpha_deg = 23.0
    # tilt_deg = 87.0
    # lon_deg = -44.0
    # lat_deg = 10.0
    # rss = 2.5
    # r_scatter = 3.30
    # main(Frequency_MHz=Frequency_MHz, time_iso=time_iso, h_apex=h_apex, kappa=kappa, alpha_deg=alpha_deg, tilt_deg=tilt_deg, lon_deg=lon_deg, lat_deg=lat_deg, rss=rss, r_scatter=r_scatter)
