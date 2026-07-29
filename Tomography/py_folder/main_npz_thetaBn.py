#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_overlay_tomography_spheroid_pfss_from_npz.py

Load an already-computed tomography solution from an NPZ file, then overlay
tomography isodensity (iso-frequency) surfaces with Spheroid/PFSS geometry.

Key points
----------
- Tomography inversion is NOT recomputed here. The 3-D electron density and
  spherical grid are loaded from the precomputed NPZ file.

- Spheroid geometry is generated in Heliographic Stonyhurst coordinates and
  transformed to Heliographic Carrington before overlaying, so that it matches
  the tomography volume.

- No argparse is used. Edit the __main__ block to set:
    * Spheroid parameters
    * ISO frequencies (MHz) and harmonic (1 or 2)

  NPZ-loading settings are kept in load_tomography_from_npz().

Dependencies (your environment)
------------------------------
numpy, pyvista
astropy + sunpy (needed for coordinate transforms and FITS/PFSS handling)

Notes
-----
This script intentionally skips the expensive tomography inversion and reuses
the stored ne(r, theta, phi) solution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os, sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict
import sunpy.map
import numpy as np

try:
    import pyvista as pv
except Exception as e:
    raise SystemExit("pyvista is required: pip install pyvista") from e
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, CartesianRepresentation
from sunpy.coordinates import frames


@dataclass
class SphericalGrid:
    """Minimal spherical grid container needed for NPZ-based tomography rendering."""
    r_edges: np.ndarray
    th_edges: np.ndarray
    ph_edges: np.ndarray

    @property
    def nr(self) -> int:
        return int(self.r_edges.size - 1)

    @property
    def nth(self) -> int:
        return int(self.th_edges.size - 1)

    @property
    def nph(self) -> int:
        return int(self.ph_edges.size - 1)

    @property
    def nvox(self) -> int:
        return int(self.nr * self.nth * self.nph)

    def voxel_centers_sph(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])
        th = 0.5 * (self.th_edges[:-1] + self.th_edges[1:])
        ph = 0.5 * (self.ph_edges[:-1] + self.ph_edges[1:])
        rr, tt, pp = np.meshgrid(r, th, ph, indexing="ij")
        return rr, tt, pp


@dataclass
class SimpleObserver:
    """Minimal observer object used by the existing overlay/camera helpers."""
    lonlat_deg: Tuple[float, float]
    cam_z: Optional[np.ndarray] = None


def parse_target_datetime(value: str | datetime) -> datetime:
    """Parse the target time used to construct the NPZ file name."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)

    text = str(value).strip()
    formats = (
        "%Y%m%d_%H%M",
        "%Y%m%d_%H%M%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H:%M:%S",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse target_time={value!r}.")


# -----------------------------------------------------------------------------
# Observer-geometry helpers (Carrington consistency)
# -----------------------------------------------------------------------------

def _safe_import_pfsspy():
    """
    Import pfsspy in a way that is more stable on WSL / OpenMP / numba environments.

    Returns
    -------
    pfsspy, utils, tracing
    """
    import builtins
    
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


def add_surface_at_radius(
    plotter,
    surface,
    *,
    r0: float,
    dr: float = 0.05,
    mode: str = "band",
    color: str = "#0011ff",
    opacity: float = 0.5,
    point_size: int = 12,
    max_points: int = 5000,
    rng_seed: int = 0,
    label: str = "Surface",
):
    """
    Highlight an arbitrary PyVista surface within r=r0±dr Rsun.

    Highlight a radial band on the Spheroid surface.
    """
    import numpy as np
    import pyvista as pv

    def _set_runinfo_text(new_txt: str):
        # run-info actor を置き換え、SetInput が効かない環境でも表示を更新する。
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
        if line not in new_txt:
            new_txt.append(line)
        for e in extras:
            if e not in new_txt:
                new_txt.append(e)

        _set_runinfo_text("\n".join(new_txt))

        try:
            plotter.render()
        except Exception:
            pass

    if surface is None or getattr(surface, "n_points", 0) == 0:
        print(f"[WARN] add_surface_at_radius: empty {label} surface")
        return {"n_selected": 0, "n_total": 0}

    r0 = float(r0)
    dr = float(abs(dr))
    surf = surface.copy(deep=True)
    pts = np.asarray(surf.points, dtype=float)
    rr = np.sqrt(np.sum(pts * pts, axis=1))
    surf["r_rsun"] = rr

    lo, hi = r0 - dr, r0 + dr

    if mode.lower() == "band":
        band = surf.threshold(value=(lo, hi), scalars="r_rsun")
        nsel = int(getattr(band, "n_points", 0))
        if nsel == 0:
            print(f"[INFO] No {label} surface cells within r={r0:g}±{dr:g} Rsun")
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

        print(f"[INFO] {label} radial band rendered: r={r0:g}±{dr:g} Rsun, n_points={nsel}")
        return {"n_selected": nsel, "n_total": int(surf.n_points)}

    if mode.lower() == "scatter":
        m = np.isfinite(rr) & (rr >= lo) & (rr <= hi)
        idx = np.where(m)[0]
        if idx.size == 0:
            print(f"[INFO] No {label} vertices within r={r0:g}±{dr:g} Rsun")
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

        print(f"[INFO] {label} radial scatter rendered: r={r0:g}±{dr:g} Rsun, n_points={pts_sel.shape[0]}")
        return {"n_selected": int(pts_sel.shape[0]), "n_total": int(surf.n_points)}

    raise ValueError("mode must be 'band' or 'scatter'")



def add_spheroid_tomography_overlap_points(
    plotter,
    *,
    spheroid_cross_points_hgc: np.ndarray,
    tomo_isosurfaces: list,
    tol_rsun: float = 0.10,
    color: str = "black",
    colors=None,
    frequencies_mhz=None,
    point_size: int = 14,
    label: str = "Spheroid",
):
    """
    Plot Spheroid meridian-parallel crosspoints close to tomography isosurfaces.

    The candidate points are not all vertices of the Spheroid surface. They are
    the discrete intersections of the Spheroid wireframe meridians and
    parallels, matching the point-selection strategy used by the GCS overlay.

    A candidate within ``tol_rsun`` of one or more tomography isosurfaces is
    assigned to the nearest isosurface and rendered with that isosurface's
    color. A point close to multiple surfaces is therefore plotted only once.

    Parameters
    ----------
    spheroid_cross_points_hgc : (N, 3) ndarray
        Meridian-parallel intersection candidates in HGC Cartesian coordinates
        and units of Rsun.
    colors : sequence of str or str, optional
        Colors corresponding to ``tomo_isosurfaces``. If omitted, ``color`` is
        used for every surface.
    frequencies_mhz : sequence of float, optional
        Frequencies corresponding to ``tomo_isosurfaces``; used for log
        messages.
    """
    import numpy as np
    import pyvista as pv

    pts = np.asarray(spheroid_cross_points_hgc, dtype=float)
    if pts.ndim != 2 or pts.shape[1:] != (3,) or pts.shape[0] == 0:
        print(f"[INFO] No {label}×tomo overlap points (no valid wireframe crosspoints).")
        return {"n_candidates": 0, "n_overlap": 0, "n_overlap_by_surface": {}}

    if tomo_isosurfaces is None or len(tomo_isosurfaces) == 0:
        print(f"[INFO] No {label}×tomo overlap points (no tomography isosurfaces).")
        return {
            "n_candidates": int(pts.shape[0]),
            "n_overlap": 0,
            "n_overlap_by_surface": {},
        }

    n_surfaces = len(tomo_isosurfaces)
    if colors is None:
        surface_colors = [str(color)] * n_surfaces
    elif isinstance(colors, str):
        surface_colors = [colors] * n_surfaces
    else:
        surface_colors = list(colors)
        if len(surface_colors) == 0:
            surface_colors = [str(color)] * n_surfaces

    if frequencies_mhz is None:
        surface_frequencies = [None] * n_surfaces
    else:
        surface_frequencies = list(frequencies_mhz)

    tol = float(abs(tol_rsun))
    nearest_distance = np.full(pts.shape[0], np.inf, dtype=float)
    nearest_surface_index = np.full(pts.shape[0], -1, dtype=int)

    for i, surf in enumerate(tomo_isosurfaces):
        if surf is None or getattr(surf, "n_points", 0) == 0:
            continue

        distance = _point_to_surface_distance_rsun(pts, surf)
        assign = (
            np.isfinite(distance)
            & (distance <= tol)
            & (distance < nearest_distance)
        )
        nearest_distance[assign] = distance[assign]
        nearest_surface_index[assign] = i

    overlap_idx = np.where(nearest_surface_index >= 0)[0]
    if overlap_idx.size == 0:
        print(f"[INFO] No {label}×tomo overlap points found within tol={tol:g} Rsun.")
        return {
            "n_candidates": int(pts.shape[0]),
            "n_overlap": 0,
            "n_overlap_by_surface": {},
        }

    overlap_counts = {}
    for i in range(n_surfaces):
        idx_i = overlap_idx[nearest_surface_index[overlap_idx] == i]
        if idx_i.size == 0:
            continue

        surface_color = surface_colors[i % len(surface_colors)]
        pts_hit = pts[idx_i]
        plotter.add_mesh(
            pv.PolyData(pts_hit),
            color=surface_color,
            point_size=int(point_size),
            render_points_as_spheres=True,
            pickable=False,
        )

        if i < len(surface_frequencies) and surface_frequencies[i] is not None:
            surface_name = f"{float(surface_frequencies[i]):g} MHz"
        else:
            surface_name = f"surface {i + 1}"

        overlap_counts[surface_name] = int(pts_hit.shape[0])
        print(
            f"[INFO] {label}×tomo overlap crosspoints for {surface_name}: "
            f"{pts_hit.shape[0]} (color={surface_color}, tol={tol:g} Rsun)"
        )

    return {
        "n_candidates": int(pts.shape[0]),
        "n_overlap": int(overlap_idx.size),
        "n_overlap_by_surface": overlap_counts,
    }


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
    Spheroid geometry is generated in HGS.
    => Must convert HGS -> HGC before overlay.

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
# Spheroid dome helpers
# -----------------------------------------------------------------------------

@dataclass
class SpheroidDome3DParams:
    """Prolate/oblate spheroid dome in HGS, using the same parameterization as plot_spheroid_C2.py.

    Semi-axes are derived from the apex height h = r_apex - 1:
        b = kappa * h
        a = b / sqrt(1 - epsilon^2)  for epsilon < 0
        a = b * sqrt(1 - epsilon^2)  for epsilon > 0

    The two-point axis mode is the main mode used in plot_spheroid_C2.py:
    anchor_lon/lat define the surface intersection, and apex_lon/lat/apex_r define
    the 3-D apex direction and radius.
    """
    kappa: float
    epsilon: float

    anchor_lon_deg: Optional[float] = None
    anchor_lat_deg: Optional[float] = None
    apex_lon_deg: Optional[float] = None
    apex_lat_deg: Optional[float] = None
    apex_r_rsun: Optional[float] = None
    apex_height_rsun: Optional[float] = None

    center_lon_deg: Optional[float] = None
    center_lat_deg: Optional[float] = None
    center_r_rsun: Optional[float] = None

    n_meridians: int = 14
    n_parallels: int = 8
    n_line_pts: int = 240

    only_above_surface: bool = True
    only_visible: bool = True

    @property
    def b_rsun(self) -> float:
        return float(self._b_rsun)

    @property
    def a_rsun(self) -> float:
        return float(self._a_rsun)

    def _has_two_point_axis(self) -> bool:
        return (
            self.anchor_lon_deg is not None
            and self.anchor_lat_deg is not None
            and self.apex_lon_deg is not None
            and self.apex_lat_deg is not None
            and ((self.apex_r_rsun is not None) or (self.apex_height_rsun is not None))
        )

    def _has_center_axis(self) -> bool:
        return (
            self.center_lon_deg is not None
            and self.center_lat_deg is not None
            and (
                (self.apex_r_rsun is not None)
                or (self.apex_height_rsun is not None)
                or (self.center_r_rsun is not None)
            )
        )

    def __post_init__(self) -> None:
        if self.apex_r_rsun is None and self.apex_height_rsun is not None:
            self.apex_r_rsun = float(1.0 + self.apex_height_rsun)
        elif self.apex_r_rsun is not None and self.apex_height_rsun is None:
            self.apex_height_rsun = float(self.apex_r_rsun - 1.0)

        if self.apex_height_rsun is None:
            raise ValueError("SpheroidDome3DParams requires apex_r_rsun or apex_height_rsun.")

        self.kappa = float(self.kappa)
        self.epsilon = float(self.epsilon)
        if self.kappa < 0.0:
            raise ValueError("kappa must be non-negative.")
        if not (-1.0 < self.epsilon < 1.0):
            raise ValueError("signed epsilon must satisfy -1 < epsilon < 1.")

        h = float(self.apex_height_rsun)
        b0 = float(self.kappa * h)
        eps_abs = abs(float(self.epsilon))
        denom = np.sqrt(max(1.0e-12, 1.0 - eps_abs**2))

        # Same signed-epsilon convention as plot_spheroid_C2.py.
        self._b_rsun = float(b0)
        if self.epsilon < 0.0:
            self._a_rsun = float(b0 / denom)
        else:
            self._a_rsun = float(b0 * denom)

        if self._has_two_point_axis():
            anchor = _cart_rsun_from_lonlat(float(self.anchor_lon_deg), float(self.anchor_lat_deg), 1.0)
            apex = _cart_rsun_from_lonlat(float(self.apex_lon_deg), float(self.apex_lat_deg), float(self.apex_r_rsun))
            axis_u = _unit_vec(apex - anchor)
            center = apex - self.a_rsun * axis_u
            center_lon_deg, center_lat_deg, center_r_rsun = _lonlat_from_cart_rsun(center)

            if self.center_lon_deg is None:
                self.center_lon_deg = float(center_lon_deg)
            if self.center_lat_deg is None:
                self.center_lat_deg = float(center_lat_deg)
            if self.center_r_rsun is None:
                self.center_r_rsun = float(center_r_rsun)
            return

        if not self._has_center_axis():
            raise ValueError(
                "SpheroidDome3DParams: provide either center_* + apex_r/height or "
                "anchor_* + apex_* + apex_r/height."
            )

        if self.anchor_lon_deg is None:
            self.anchor_lon_deg = float(self.center_lon_deg)  # type: ignore[arg-type]
        if self.anchor_lat_deg is None:
            self.anchor_lat_deg = float(self.center_lat_deg)  # type: ignore[arg-type]
        if self.apex_lon_deg is None:
            self.apex_lon_deg = float(self.center_lon_deg)  # type: ignore[arg-type]
        if self.apex_lat_deg is None:
            self.apex_lat_deg = float(self.center_lat_deg)  # type: ignore[arg-type]
        if self.center_r_rsun is None and self.apex_r_rsun is not None:
            self.center_r_rsun = float(self.apex_r_rsun - self.a_rsun)

    def runinfo_label(self) -> str:
        if self._has_two_point_axis():
            return "\n".join(
                [
                    f"Spheroid: h={self.apex_height_rsun:.2f} Rsun, kappa={self.kappa:.3f}",
                    f"     epsilon={self.epsilon:.3f}, apex=({float(self.apex_lon_deg):.1f},{float(self.apex_lat_deg):.1f}) deg",
                ]
            )
        return "\n".join(
            [
                f"Spheroid: h={self.apex_height_rsun:.2f} Rsun, kappa={self.kappa:.3f}",
                f"     epsilon={self.epsilon:.3f}, center=({float(self.center_lon_deg):.1f},{float(self.center_lat_deg):.1f}) deg",
            ]
        )


def _hgs_unit_vectors(lon_deg: float, lat_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return radial, increasing-longitude, and increasing-latitude unit vectors in HGS."""
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    cosl, sinl = np.cos(lon), np.sin(lon)
    cosb, sinb = np.cos(lat), np.sin(lat)

    r_hat = np.array([cosb * cosl, cosb * sinl, sinb], dtype=float)
    e_lon = np.array([-sinl, cosl, 0.0], dtype=float)
    e_lat = np.array([-sinb * cosl, -sinb * sinl, cosb], dtype=float)

    r_hat /= np.linalg.norm(r_hat)
    e_lon /= np.linalg.norm(e_lon)
    e_lat /= np.linalg.norm(e_lat)
    return r_hat, e_lon, e_lat


def _cart_rsun_from_lonlat(lon_deg: float, lat_deg: float, r_rsun: float) -> np.ndarray:
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    return np.array(
        [
            r_rsun * np.cos(lat) * np.cos(lon),
            r_rsun * np.cos(lat) * np.sin(lon),
            r_rsun * np.sin(lat),
        ],
        dtype=float,
    )


def _lonlat_from_cart_rsun(cart_rsun: np.ndarray) -> Tuple[float, float, float]:
    arr = np.asarray(cart_rsun, dtype=float).reshape(3)
    x, y, z = float(arr[0]), float(arr[1]), float(arr[2])
    r = float(np.sqrt(x * x + y * y + z * z))
    if r <= 0.0:
        raise ValueError("zero-radius cartesian point cannot be converted to lon/lat.")
    lon_deg = float(np.rad2deg(np.arctan2(y, x)))
    lat_deg = float(np.rad2deg(np.arcsin(np.clip(z / r, -1.0, 1.0))))
    return lon_deg, lat_deg, r


def _unit_vec(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr))
        if n == 0.0:
            raise ValueError("zero norm vector")
        return arr / n
    if arr.ndim == 2:
        n = np.linalg.norm(arr, axis=0, keepdims=True)
        n = np.where(n == 0.0, 1.0, n)
        return arr / n
    raise ValueError("unsupported ndarray ndim")


def _orthonormal_basis_from_axis(axis_u: np.ndarray, ref_vec: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    axis_u = _unit_vec(axis_u)
    if ref_vec is None:
        ref_vec = np.array([0.0, 0.0, 1.0], dtype=float)
    ref = np.asarray(ref_vec, dtype=float)
    ref = ref - np.dot(ref, axis_u) * axis_u
    if np.linalg.norm(ref) < 1.0e-10:
        alt = np.array([1.0, 0.0, 0.0], dtype=float) if abs(axis_u[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
        ref = alt - np.dot(alt, axis_u) * axis_u
    e1 = _unit_vec(ref)
    e2 = _unit_vec(np.cross(axis_u, e1))
    return e1, e2


def _spheroid_axis_geometry_rsun(params: SpheroidDome3DParams) -> Dict[str, np.ndarray]:
    if params._has_two_point_axis():
        anchor = _cart_rsun_from_lonlat(float(params.anchor_lon_deg), float(params.anchor_lat_deg), 1.0)
        apex = _cart_rsun_from_lonlat(float(params.apex_lon_deg), float(params.apex_lat_deg), float(params.apex_r_rsun))
        axis_u = _unit_vec(apex - anchor)
        center = apex - float(params.a_rsun) * axis_u
        e1, e2 = _orthonormal_basis_from_axis(axis_u, ref_vec=anchor)
        return {"center": center, "axis_u": axis_u, "e1": e1, "e2": e2, "anchor": anchor, "apex": apex}

    r_hat, e_lon, e_lat = _hgs_unit_vectors(float(params.center_lon_deg), float(params.center_lat_deg))
    center = float(params.center_r_rsun) * r_hat
    anchor = 1.0 * r_hat
    apex = center + float(params.a_rsun) * r_hat
    return {"center": center, "axis_u": r_hat, "e1": e_lon, "e2": e_lat, "anchor": anchor, "apex": apex}


def _split_points_by_mask(points: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
    """Split a 3-D polyline into contiguous finite/visible segments."""
    points = np.asarray(points, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    cuts = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[0], cuts + 1])
    ends = np.concatenate([cuts + 1, [idx.size]])
    out: List[np.ndarray] = []
    for s0, s1 in zip(starts, ends):
        seg_idx = idx[s0:s1]
        if seg_idx.size >= 2:
            out.append(points[seg_idx])
    return out


def _spheroid_visibility_mask_hgc(points_hgc: np.ndarray, obs0, only_visible: bool) -> np.ndarray:
    if not only_visible or obs0 is None:
        return np.ones(points_hgc.shape[0], dtype=bool)
    try:
        obs_hat = sun_to_observer_unit_vector(obs0)
        return np.sum(points_hgc * obs_hat[None, :], axis=1) >= 0.0
    except Exception:
        return np.ones(points_hgc.shape[0], dtype=bool)



def _resample_polyline_xyz(
    pts: np.ndarray,
    n: int,
    *,
    closed: bool = False,
    close_tol: float = 1.0e-6,
) -> np.ndarray:
    """
    Resample a 3-D polyline at approximately uniform arc-length intervals.

    When ``closed=True``, the first point is appended after resampling so the
    returned polyline contains an explicit closing segment.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("pts must have shape (N, 3).")
    if pts.shape[0] < 2:
        return pts.copy()

    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if pts.shape[0] < 2:
        return pts.copy()

    if closed:
        if np.linalg.norm(pts[-1] - pts[0]) <= float(close_tol):
            pts = pts[:-1]
        if pts.shape[0] < 2:
            return pts.copy()
        pts_ext = np.vstack([pts, pts[0]])
    else:
        pts_ext = pts

    segment_length = np.linalg.norm(np.diff(pts_ext, axis=0), axis=1)
    arc_length = np.concatenate([[0.0], np.cumsum(segment_length)])
    total_length = float(arc_length[-1])
    if not np.isfinite(total_length) or total_length <= 0.0:
        n_out = int(n) + (1 if closed else 0)
        return np.repeat(pts_ext[:1], n_out, axis=0)

    if closed:
        sample_arc = np.linspace(0.0, total_length, int(n), endpoint=False)
    else:
        sample_arc = np.linspace(0.0, total_length, int(n), endpoint=True)

    out = np.column_stack(
        [
            np.interp(sample_arc, arc_length, pts_ext[:, axis])
            for axis in range(3)
        ]
    )
    if closed:
        out = np.vstack([out, out[0]])
    return out


def sample_spheroid_dome_wireframe_components_hgs(
    params: SpheroidDome3DParams,
) -> Dict[str, List[np.ndarray]]:
    """
    Sample Spheroid meridians and parallels separately in HGS Cartesian coordinates.

    Curves below the photosphere are removed when ``only_above_surface=True``.
    The separate curve groups are used both for rendering and for constructing
    meridian-parallel intersection candidates.
    """
    geom = _spheroid_axis_geometry_rsun(params)
    center = geom["center"][:, None]
    axis_u = geom["axis_u"][:, None]
    e1 = geom["e1"][:, None]
    e2 = geom["e2"][:, None]

    meridians: List[np.ndarray] = []
    parallels: List[np.ndarray] = []

    def _above_surface_mask(cart: np.ndarray) -> np.ndarray:
        if not params.only_above_surface:
            return np.ones(cart.shape[1], dtype=bool)
        rr = np.sqrt(np.sum(cart * cart, axis=0))
        return rr >= 1.0

    alphas = np.linspace(0.0, np.pi, int(params.n_line_pts), endpoint=True)
    ca = np.cos(alphas)[None, :]
    sa = np.sin(alphas)[None, :]

    for beta in np.linspace(
        0.0,
        2.0 * np.pi,
        int(params.n_meridians),
        endpoint=False,
    ):
        dir_perp = np.cos(beta) * e1 + np.sin(beta) * e2
        cart = (
            center
            + float(params.a_rsun) * ca * axis_u
            + float(params.b_rsun) * sa * dir_perp
        )
        mask = _above_surface_mask(cart)
        meridians.extend(_split_points_by_mask(cart.T, mask))

    alpha_list = np.linspace(
        0.0,
        np.pi,
        int(params.n_parallels) + 2,
    )[1:-1]
    betas_line = np.linspace(
        0.0,
        2.0 * np.pi,
        int(params.n_line_pts),
        endpoint=True,
    )
    cb = np.cos(betas_line)[None, :]
    sb = np.sin(betas_line)[None, :]

    for alpha0 in alpha_list:
        ca0 = float(np.cos(alpha0))
        sa0 = float(np.sin(alpha0))
        dir_perp = cb * e1 + sb * e2
        cart = (
            center
            + float(params.a_rsun) * ca0 * axis_u
            + float(params.b_rsun) * sa0 * dir_perp
        )
        mask = _above_surface_mask(cart)
        parallels.extend(_split_points_by_mask(cart.T, mask))

    return {
        "meridians": meridians,
        "parallels": parallels,
    }


def sample_spheroid_dome_wireframe_hgs(
    params: SpheroidDome3DParams,
) -> List[np.ndarray]:
    """Sample all Spheroid wireframe curves in HGS Cartesian coordinates [Rsun]."""
    components = sample_spheroid_dome_wireframe_components_hgs(params)
    return components["meridians"] + components["parallels"]


def compute_spheroid_meridian_parallel_crosspoints_hgc(
    params: SpheroidDome3DParams,
    *,
    obstime_iso: str,
    observer: str = "earth",
    n_resample: int = 200,
    cross_tol_rsun: float = 0.05,
    merge_tol_rsun: float = 0.03,
) -> np.ndarray:
    """
    Extract Spheroid meridian-parallel intersection candidates in HGC coordinates.

    This follows the GCS overlap-point procedure:
      1. Generate meridian and parallel curves separately.
      2. Transform every curve from HGS to HGC.
      3. Resample each curve along arc length.
      4. Find the closest point pair for every meridian-parallel combination.
      5. Accept the midpoint when the closest separation is within
         ``cross_tol_rsun``.
      6. Merge duplicate candidates within ``merge_tol_rsun``.
    """
    components = sample_spheroid_dome_wireframe_components_hgs(params)
    meridians_hgs = components["meridians"]
    parallels_hgs = components["parallels"]

    if len(meridians_hgs) == 0 or len(parallels_hgs) == 0:
        return np.zeros((0, 3), dtype=float)

    meridians_hgc: List[np.ndarray] = []
    for pts_hgs in meridians_hgs:
        pts_hgc = transform_points_hgs_to_hgc(
            np.asarray(pts_hgs, dtype=float),
            obstime_iso=obstime_iso,
            observer=observer,
        )
        pts_resampled = _resample_polyline_xyz(
            pts_hgc,
            int(n_resample),
            closed=False,
        )
        if pts_resampled.shape[0] >= 4:
            meridians_hgc.append(pts_resampled)

    parallels_hgc: List[np.ndarray] = []
    for pts_hgs in parallels_hgs:
        pts_hgc = transform_points_hgs_to_hgc(
            np.asarray(pts_hgs, dtype=float),
            obstime_iso=obstime_iso,
            observer=observer,
        )
        is_closed = np.linalg.norm(pts_hgc[-1] - pts_hgc[0]) <= 1.0e-4
        pts_resampled = _resample_polyline_xyz(
            pts_hgc,
            int(n_resample),
            closed=bool(is_closed),
        )
        if pts_resampled.shape[0] >= 4:
            parallels_hgc.append(pts_resampled)

    if len(meridians_hgc) == 0 or len(parallels_hgc) == 0:
        return np.zeros((0, 3), dtype=float)

    cross_points: List[np.ndarray] = []
    cross_tol_squared = float(abs(cross_tol_rsun)) ** 2

    for meridian in meridians_hgc:
        for parallel in parallels_hgc:
            delta = meridian[:, None, :] - parallel[None, :, :]
            distance_squared = np.sum(delta * delta, axis=2)
            i_meridian, i_parallel = np.unravel_index(
                int(np.argmin(distance_squared)),
                distance_squared.shape,
            )
            if float(distance_squared[i_meridian, i_parallel]) <= cross_tol_squared:
                cross_points.append(
                    0.5 * (
                        meridian[i_meridian]
                        + parallel[i_parallel]
                    )
                )

    if len(cross_points) == 0:
        return np.zeros((0, 3), dtype=float)

    merged: List[np.ndarray] = []
    merge_tol_squared = float(abs(merge_tol_rsun)) ** 2
    for candidate in np.asarray(cross_points, dtype=float):
        if any(
            float(np.sum((candidate - accepted) ** 2)) <= merge_tol_squared
            for accepted in merged
        ):
            continue
        merged.append(candidate)

    out = np.asarray(merged, dtype=float)
    print(
        f"[INFO] Spheroid wireframe crosspoints: {out.shape[0]} "
        f"(meridian_segments={len(meridians_hgc)}, "
        f"parallel_segments={len(parallels_hgc)}, "
        f"cross_tol={float(abs(cross_tol_rsun)):g} Rsun, "
        f"merge_tol={float(abs(merge_tol_rsun)):g} Rsun)"
    )
    return out


def _sample_spheroid_footprint_cart_rsun(params: SpheroidDome3DParams, n_beta: Optional[int] = None) -> np.ndarray:
    geom = _spheroid_axis_geometry_rsun(params)
    center = geom["center"]
    axis_u = geom["axis_u"]
    e1 = geom["e1"]
    e2 = geom["e2"]

    if n_beta is None:
        n_beta = int(params.n_line_pts)

    betas = np.linspace(0.0, 2.0 * np.pi, int(n_beta), endpoint=True)
    alpha_grid = np.linspace(0.0, np.pi, 1201)
    pts: List[np.ndarray] = []

    for beta in betas:
        dir_perp = np.cos(beta) * e1 + np.sin(beta) * e2
        ca = np.cos(alpha_grid)
        sa = np.sin(alpha_grid)
        cart = (
            center[:, None]
            + float(params.a_rsun) * ca[None, :] * axis_u[:, None]
            + float(params.b_rsun) * sa[None, :] * dir_perp[:, None]
        )
        rr = np.linalg.norm(cart, axis=0)
        f = rr - 1.0

        roots: List[float] = []
        hit = np.where(np.isclose(f, 0.0, atol=1.0e-5))[0]
        for idx in hit:
            roots.append(float(alpha_grid[idx]))

        for i in range(len(alpha_grid) - 1):
            if f[i] == 0.0 or f[i + 1] == 0.0:
                continue
            if f[i] * f[i + 1] < 0.0:
                a0 = float(alpha_grid[i] + (0.0 - f[i]) * (alpha_grid[i + 1] - alpha_grid[i]) / (f[i + 1] - f[i]))
                roots.append(a0)

        if not roots:
            continue
        alpha0 = max(roots, key=lambda a: np.sin(a))
        p = center + float(params.a_rsun) * np.cos(alpha0) * axis_u + float(params.b_rsun) * np.sin(alpha0) * dir_perp
        if abs(np.linalg.norm(p) - 1.0) < 5.0e-3:
            pts.append(p)

    if not pts:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(pts, dtype=float)


def sample_spheroid_footprint_hgs(params: SpheroidDome3DParams) -> List[np.ndarray]:
    """Sample the photospheric intersection of the spheroid dome in HGS Cartesian coordinates."""
    if params._has_two_point_axis():
        pts = _sample_spheroid_footprint_cart_rsun(params)
        return [pts] if pts.shape[0] >= 2 else []

    r_hat, e_lon, e_lat = _hgs_unit_vectors(float(params.center_lon_deg), float(params.center_lat_deg))
    center_r = float(params.center_r_rsun)
    center = center_r * r_hat
    a = float(params.a_rsun)
    b = float(params.b_rsun)
    cr = center_r

    A = a * a - b * b
    B = 2.0 * cr * a
    C = cr * cr + b * b - 1.0

    if abs(A) < 1.0e-12:
        if abs(B) < 1.0e-12:
            return []
        candidates = [(-C / B)]
    else:
        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            return []
        sdisc = float(np.sqrt(disc))
        candidates = [(-B + sdisc) / (2.0 * A), (-B - sdisc) / (2.0 * A)]

    valid = [c for c in candidates if np.isfinite(c) and (-1.0 <= c <= 1.0)]
    if not valid:
        return []

    best_c = max(valid, key=lambda cc: np.sqrt(max(0.0, 1.0 - cc * cc)))
    alpha0 = float(np.arccos(best_c))
    sin_a0 = float(np.sin(alpha0))
    cos_a0 = float(np.cos(alpha0))

    bet = np.linspace(0.0, 2.0 * np.pi, int(params.n_line_pts), endpoint=True)
    dir_latlon = np.cos(bet)[None, :] * e_lon[:, None] + np.sin(bet)[None, :] * e_lat[:, None]
    cart = center[:, None] + a * cos_a0 * r_hat[:, None] + b * sin_a0 * dir_latlon
    return [cart.T]


def build_spheroid_surface_hgc(
    params: SpheroidDome3DParams,
    *,
    obstime_iso: str,
    observer: str = "earth",
    n_alpha: int = 96,
    n_beta: int = 160,
) -> Optional[pv.PolyData]:
    """Build a clipped spheroid surface as PyVista PolyData in HGC Cartesian coordinates."""
    geom = _spheroid_axis_geometry_rsun(params)
    center = geom["center"]
    axis_u = geom["axis_u"]
    e1 = geom["e1"]
    e2 = geom["e2"]

    alphas = np.linspace(0.0, np.pi, int(n_alpha), endpoint=True)
    betas = np.linspace(0.0, 2.0 * np.pi, int(n_beta), endpoint=False)
    aa, bb = np.meshgrid(alphas, betas, indexing="ij")

    ca = np.cos(aa)
    sa = np.sin(aa)
    cb = np.cos(bb)
    sb = np.sin(bb)

    pts_hgs = (
        center[None, None, :]
        + float(params.a_rsun) * ca[:, :, None] * axis_u[None, None, :]
        + float(params.b_rsun) * sa[:, :, None] * (cb[:, :, None] * e1[None, None, :] + sb[:, :, None] * e2[None, None, :])
    )
    pts_flat_hgs = pts_hgs.reshape((-1, 3))
    pts_flat_hgc = transform_points_hgs_to_hgc(
        pts_flat_hgs,
        obstime_iso=obstime_iso,
        observer=observer,
    )

    # Analytic normal of the local spheroid implicit surface:
    #   ((q.axis_u)/a)^2 + ((q.e1)/b)^2 + ((q.e2)/b)^2 = 1.
    # Its gradient gives the geometric shock-normal direction. The HGS normal
    # is transformed to HGC using a small endpoint displacement, which is robust
    # across SunPy versions without relying on differential transformations.
    q_hgs = pts_flat_hgs - center[None, :]
    a2 = max(float(params.a_rsun) ** 2, 1.0e-20)
    b2 = max(float(params.b_rsun) ** 2, 1.0e-20)
    q_axis = q_hgs @ axis_u
    q_e1 = q_hgs @ e1
    q_e2 = q_hgs @ e2
    normal_hgs = (
        (q_axis / a2)[:, None] * axis_u[None, :]
        + (q_e1 / b2)[:, None] * e1[None, :]
        + (q_e2 / b2)[:, None] * e2[None, :]
    )
    normal_hgs_norm = np.linalg.norm(normal_hgs, axis=1)
    good_normal_hgs = np.isfinite(normal_hgs_norm) & (normal_hgs_norm > 0.0)
    normal_hgs[good_normal_hgs] /= normal_hgs_norm[good_normal_hgs, None]
    normal_hgs[~good_normal_hgs] = np.nan

    normal_step_rsun = 1.0e-5
    normal_tip_hgc = transform_points_hgs_to_hgc(
        pts_flat_hgs + normal_step_rsun * normal_hgs,
        obstime_iso=obstime_iso,
        observer=observer,
    )
    normal_hgc = normal_tip_hgc - pts_flat_hgc
    normal_hgc_norm = np.linalg.norm(normal_hgc, axis=1)
    good_normal_hgc = np.isfinite(normal_hgc_norm) & (normal_hgc_norm > 0.0)
    normal_hgc[good_normal_hgc] /= normal_hgc_norm[good_normal_hgc, None]
    normal_hgc[~good_normal_hgc] = np.nan

    rr = np.sqrt(np.sum(pts_flat_hgs * pts_flat_hgs, axis=1)).reshape((int(n_alpha), int(n_beta)))
    valid = np.ones((int(n_alpha), int(n_beta)), dtype=bool)
    if params.only_above_surface:
        valid &= rr >= 1.0

    faces: List[int] = []
    nb = int(n_beta)
    for ia in range(int(n_alpha) - 1):
        for ib in range(nb):
            ib2 = (ib + 1) % nb
            ids = [ia * nb + ib, ia * nb + ib2, (ia + 1) * nb + ib2, (ia + 1) * nb + ib]
            if valid[ia, ib] and valid[ia, ib2] and valid[ia + 1, ib2] and valid[ia + 1, ib]:
                faces.extend([4] + ids)

    if not faces:
        return None
    surf = pv.PolyData(
        pts_flat_hgc.astype(np.float64),
        np.asarray(faces, dtype=np.int64),
    )
    surf.point_data["spheroid_normal_hgc"] = normal_hgc.astype(np.float64)
    return surf.triangulate().clean()


def add_spheroid_dome_3d(
    plotter,
    params: SpheroidDome3DParams,
    *,
    obstime_iso: str,
    observer: str = "earth",
    obs0=None,
    color: str = "magenta",
    surface_opacity: float = 0.16,
    wire_opacity: float = 0.95,
    footprint_opacity: float = 1.0,
    line_width: int = 3,
    footprint_width: int = 4,
    marker_radius: float = 0.045,
    show_surface: bool = True,
    show_wireframe: bool = True,
    show_footprint: bool = True,
    show_markers: bool = True,
    return_surface: bool = False,
) -> Dict[str, object]:
    """Overlay the plot_spheroid_C2.py spheroid model in the same HGC 3-D space as tomography/PFSS.

    If return_surface=True, the returned dictionary includes ``surface`` as a
    PyVista PolyData. This allows radial-band and tomography-overlap diagnostics
    to be applied to the Spheroid surface.
    """
    info: Dict[str, object] = {"n_surface_points": 0, "n_wire_lines": 0, "n_footprint_lines": 0}
    spheroid_surface = None

    if show_surface:
        try:
            surf = build_spheroid_surface_hgc(params, obstime_iso=obstime_iso, observer=observer)
            if surf is not None and surf.n_points > 0:
                plotter.add_mesh(
                    surf,
                    color=color,
                    opacity=float(surface_opacity),
                    smooth_shading=True,
                    lighting=True,
                    pickable=False,
                )
                info["n_surface_points"] = int(surf.n_points)
                spheroid_surface = surf
        except Exception as exc:
            print(f"[WARN] Spheroid surface skipped: {exc}")

    wire_params = params
    wire_lines_hgs = sample_spheroid_dome_wireframe_hgs(wire_params)
    if show_wireframe and not wire_lines_hgs and params.only_visible:
        wire_params = replace(params, only_visible=False)
        wire_lines_hgs = sample_spheroid_dome_wireframe_hgs(wire_params)

    if show_wireframe:
        for pts_hgs in wire_lines_hgs:
            pts_hgc = transform_points_hgs_to_hgc(pts_hgs, obstime_iso=obstime_iso, observer=observer)
            visible = _spheroid_visibility_mask_hgc(pts_hgc, obs0, only_visible=wire_params.only_visible)
            finite = np.all(np.isfinite(pts_hgc), axis=1)
            for seg in _split_points_by_mask(pts_hgc, finite & visible):
                plotter.add_mesh(
                    _polyline_from_points(seg),
                    color=color,
                    line_width=int(line_width),
                    opacity=float(wire_opacity),
                    lighting=False,
                    render_lines_as_tubes=True,
                    pickable=False,
                )
                info["n_wire_lines"] += 1

    if show_footprint:
        for pts_hgs in sample_spheroid_footprint_hgs(params):
            pts_hgc = transform_points_hgs_to_hgc(pts_hgs, obstime_iso=obstime_iso, observer=observer)
            visible = _spheroid_visibility_mask_hgc(pts_hgc, obs0, only_visible=params.only_visible)
            finite = np.all(np.isfinite(pts_hgc), axis=1)
            for seg in _split_points_by_mask(pts_hgc, finite & visible):
                plotter.add_mesh(
                    _polyline_from_points(seg),
                    color=color,
                    line_width=int(footprint_width),
                    opacity=float(footprint_opacity),
                    lighting=False,
                    render_lines_as_tubes=True,
                    pickable=False,
                )
                info["n_footprint_lines"] += 1

    if show_markers:
        try:
            geom = _spheroid_axis_geometry_rsun(params)
            anchor_hgc = transform_points_hgs_to_hgc(geom["anchor"][None, :], obstime_iso=obstime_iso, observer=observer)[0]
            apex_hgc = transform_points_hgs_to_hgc(geom["apex"][None, :], obstime_iso=obstime_iso, observer=observer)[0]
            plotter.add_mesh(pv.Sphere(radius=float(marker_radius), center=tuple(anchor_hgc)), color="yellow", opacity=1.0, pickable=False)
            plotter.add_mesh(pv.Sphere(radius=float(marker_radius), center=tuple(apex_hgc)), color="orange", opacity=1.0, pickable=False)
        except Exception as exc:
            print(f"[WARN] Spheroid markers skipped: {exc}")

    if return_surface and spheroid_surface is not None and getattr(spheroid_surface, "n_points", 0) > 0:
        info["surface"] = spheroid_surface

    print(
        f"[INFO] Spheroid rendered: surface_points={info['n_surface_points']}, "
        f"wire_lines={info['n_wire_lines']}, footprint_lines={info['n_footprint_lines']}"
    )
    return info


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
    matching the ordering used in visualize_isosurface() in main_multi_tomo.py.
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
    opacity=0.3,
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
            # Keep the list aligned with iso_freqs_mhz/colors even when one
            # requested contour cannot be constructed.
            surfaces.append(None)
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
    spheroid_params: Optional[SpheroidDome3DParams] = None,
    pfss_params: Optional[dict] = None,
    position: str = "upper_right",
    font_size: int = 12,
):
    """
    Draw a compact run-info text block.
    add_isosurfaces() が保存した _pending_runinfo_lines もここで統合する。
    """
    freqs = ", ".join([f"{float(f):.1f}" for f in iso_freqs_mhz])

    base_lines = [
        f"Time: {obstime_iso}",
        f"Iso-freq: [{freqs}] MHz (H={int(harmonic)})",
    ]

    if spheroid_params is not None:
        base_lines.append(spheroid_params.runinfo_label())

    if pfss_params:
        rss = pfss_params.get("rss", None)
        pfss_line = "PFSS"
        if rss is not None:
            pfss_line += f": Rss={float(rss):.2f} Rsun"
        base_lines.append(pfss_line)

    pending = getattr(plotter, "_pending_runinfo_lines", [])
    pending = [str(s) for s in pending] if pending else []
    for s in pending:
        if s not in base_lines:
            base_lines.append(s)

    txt = "\n".join(base_lines)

    old_actor = getattr(plotter, "_runinfo_text_actor", None)
    if old_actor is not None:
        try:
            plotter.remove_actor(old_actor)
        except Exception:
            pass

    actor = plotter.add_text(txt, position=position, font_size=int(font_size), color="black")

    plotter._runinfo_text_base = txt
    plotter._runinfo_text_actor = actor
    plotter._runinfo_text_position = position
    plotter._runinfo_text_font_size = int(font_size)
    plotter._runinfo_text_color = "black"
    plotter._runinfo_text_extra_lines = []

    return actor

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

    # If PFSS/Spheroid introduces large-scale coordinates, keep them visible.
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



# -----------------------------------------------------------------------------
# Shock-normal angle theta_Bn on the Spheroid surface
# -----------------------------------------------------------------------------

def _hgc_skycoord_from_cartesian_points(
    points_hgc_rsun: np.ndarray,
    *,
    obstime_iso: str,
    observer: str = "earth",
) -> SkyCoord:
    """Create Heliographic Carrington coordinates from HGC Cartesian points."""
    points = np.asarray(points_hgc_rsun, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_hgc_rsun must have shape (N, 3).")

    rep = CartesianRepresentation(
        x=points[:, 0] * u.R_sun,
        y=points[:, 1] * u.R_sun,
        z=points[:, 2] * u.R_sun,
    )
    t = Time(obstime_iso)
    try:
        hgc_frame = frames.HeliographicCarrington(obstime=t, observer=observer)
    except TypeError:
        hgc_frame = frames.HeliographicCarrington(obstime=t)
    return SkyCoord(rep, frame=hgc_frame)


def interpolate_pfss_b_cartesian_on_hgc_points(
    pfss_output,
    points_hgc_rsun: np.ndarray,
    *,
    obstime_iso: str,
    observer: str = "earth",
    radial_epsilon_rsun: float = 1.0e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate the PFSS magnetic field at arbitrary HGC Cartesian points.

    Returns
    -------
    b_xyz : (N, 3) ndarray
        Cartesian magnetic-field vectors in the Carrington basis. Invalid or
        out-of-domain points are NaN.
    valid : (N,) ndarray of bool
        True where the PFSS field was successfully evaluated.

    Notes
    -----
    pfsspy defines the model only in 1 < r/Rsun < Rss. Points outside that
    shell are deliberately left invalid rather than extrapolated.
    """
    points = np.asarray(points_hgc_rsun, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_hgc_rsun must have shape (N, 3).")

    n_points = int(points.shape[0])
    b_xyz = np.full((n_points, 3), np.nan, dtype=float)
    radius = np.linalg.norm(points, axis=1)

    rss = float(getattr(getattr(pfss_output, "grid", None), "rss", np.nan))
    if not np.isfinite(rss):
        raise ValueError("Could not determine PFSS source-surface radius from pfss_output.grid.rss.")

    eps = float(abs(radial_epsilon_rsun))
    domain = (
        np.all(np.isfinite(points), axis=1)
        & np.isfinite(radius)
        & (radius > 1.0 + eps)
        & (radius < rss - eps)
    )
    idx = np.where(domain)[0]
    if idx.size == 0:
        return b_xyz, np.zeros(n_points, dtype=bool)

    coords_hgc = _hgc_skycoord_from_cartesian_points(
        points[idx],
        obstime_iso=obstime_iso,
        observer=observer,
    )
    coords_pfss = coords_hgc.transform_to(pfss_output.coordinate_frame)

    try:
        if hasattr(pfss_output, "get_bvec"):
            b_quantity = pfss_output.get_bvec(coords_pfss, out_type="cartesian")
            b_values = np.asarray(getattr(b_quantity, "value", b_quantity), dtype=float)
        else:
            # Compatibility fallback for older pfsspy versions. _brgi returns
            # Cartesian components in the PFSS/Carrington basis.
            coords_pfss.representation_type = "spherical"
            query = np.column_stack(
                [
                    coords_pfss.lon.to_value(u.rad),
                    np.sin(coords_pfss.lat.to_value(u.rad)),
                    np.log(coords_pfss.radius.to_value(u.R_sun)),
                ]
            )
            b_values = np.asarray(pfss_output._brgi(query), dtype=float)
    except Exception as exc:
        raise RuntimeError(f"PFSS magnetic-field interpolation failed: {exc}") from exc

    if b_values.ndim == 1:
        b_values = b_values.reshape(1, 3)
    if b_values.shape != (idx.size, 3):
        raise RuntimeError(
            "Unexpected PFSS vector shape: "
            f"got {b_values.shape}, expected {(idx.size, 3)}."
        )

    good_local = np.all(np.isfinite(b_values), axis=1)
    b_norm = np.linalg.norm(b_values, axis=1)
    good_local &= np.isfinite(b_norm) & (b_norm > 0.0)
    b_xyz[idx[good_local]] = b_values[good_local]

    valid = np.all(np.isfinite(b_xyz), axis=1)
    return b_xyz, valid


def compute_spheroid_theta_bn_surface(
    spheroid_surface_hgc: pv.PolyData,
    pfss_output,
    *,
    obstime_iso: str,
    observer: str = "earth",
    radial_epsilon_rsun: float = 1.0e-4,
    quasi_perpendicular_threshold_deg: float = 70.0,
) -> Tuple[pv.PolyData, Dict[str, float]]:
    r"""
    Compute the acute shock-normal angle theta_Bn on a Spheroid surface.

    The angle is defined as

        theta_Bn = arccos(|B dot n| / (|B| |n|)),

    so the result lies in 0--90 degrees and is independent of the arbitrary
    inward/outward orientation of the triangulated surface normal.
    """
    if spheroid_surface_hgc is None or spheroid_surface_hgc.n_points == 0:
        raise ValueError("spheroid_surface_hgc is empty.")

    surface = spheroid_surface_hgc.triangulate().clean()
    if "spheroid_normal_hgc" in surface.point_data:
        normals = np.asarray(
            surface.point_data["spheroid_normal_hgc"],
            dtype=float,
        )
        normal_source = "analytic spheroid gradient"
    else:
        # Compatibility fallback for surfaces created by an older version of
        # build_spheroid_surface_hgc().
        surface = surface.compute_normals(
            point_normals=True,
            cell_normals=False,
            split_vertices=False,
            consistent_normals=True,
            auto_orient_normals=True,
            inplace=False,
        )
        normals = np.asarray(surface.point_data["Normals"], dtype=float)
        normal_source = "PyVista point normals"

    points = np.asarray(surface.points, dtype=float)
    b_xyz, valid_b = interpolate_pfss_b_cartesian_on_hgc_points(
        pfss_output,
        points,
        obstime_iso=obstime_iso,
        observer=observer,
        radial_epsilon_rsun=radial_epsilon_rsun,
    )

    n_norm = np.linalg.norm(normals, axis=1)
    b_norm = np.linalg.norm(b_xyz, axis=1)
    valid = (
        valid_b
        & np.all(np.isfinite(normals), axis=1)
        & np.isfinite(n_norm)
        & (n_norm > 0.0)
        & np.isfinite(b_norm)
        & (b_norm > 0.0)
    )

    theta_bn_deg = np.full(surface.n_points, np.nan, dtype=float)
    cos_theta = np.full(surface.n_points, np.nan, dtype=float)
    cos_theta[valid] = np.abs(np.einsum("ij,ij->i", b_xyz[valid], normals[valid])) / (
        b_norm[valid] * n_norm[valid]
    )
    cos_theta[valid] = np.clip(cos_theta[valid], 0.0, 1.0)
    theta_bn_deg[valid] = np.rad2deg(np.arccos(cos_theta[valid]))

    surface.point_data["theta_Bn_deg"] = theta_bn_deg
    surface.point_data["B_pfss_magnitude"] = b_norm
    surface.point_data["pfss_valid"] = valid.astype(np.uint8)

    n_valid = int(np.count_nonzero(valid))
    n_total = int(surface.n_points)
    threshold = float(quasi_perpendicular_threshold_deg)
    if n_valid > 0:
        values = theta_bn_deg[valid]
        summary = {
            "n_total": float(n_total),
            "n_valid": float(n_valid),
            "valid_fraction": float(n_valid / n_total),
            "theta_min_deg": float(np.nanmin(values)),
            "theta_median_deg": float(np.nanmedian(values)),
            "theta_mean_deg": float(np.nanmean(values)),
            "theta_max_deg": float(np.nanmax(values)),
            "quasi_perpendicular_threshold_deg": threshold,
            "quasi_perpendicular_fraction": float(np.mean(values >= threshold)),
        }
    else:
        summary = {
            "n_total": float(n_total),
            "n_valid": 0.0,
            "valid_fraction": 0.0,
            "theta_min_deg": np.nan,
            "theta_median_deg": np.nan,
            "theta_mean_deg": np.nan,
            "theta_max_deg": np.nan,
            "quasi_perpendicular_threshold_deg": threshold,
            "quasi_perpendicular_fraction": np.nan,
        }

    print(
        "[INFO] theta_Bn on Spheroid: "
        f"normal={normal_source}, "
        f"valid={n_valid}/{n_total} ({100.0 * summary['valid_fraction']:.1f}%), "
        f"min/median/mean/max="
        f"{summary['theta_min_deg']:.2f}/"
        f"{summary['theta_median_deg']:.2f}/"
        f"{summary['theta_mean_deg']:.2f}/"
        f"{summary['theta_max_deg']:.2f} deg, "
        f"fraction(theta_Bn>={threshold:g} deg)="
        f"{100.0 * summary['quasi_perpendicular_fraction']:.1f}%"
    )
    return surface, summary


def add_spheroid_theta_bn_colormap(
    plotter,
    spheroid_surface_hgc: pv.PolyData,
    pfss_output,
    *,
    obstime_iso: str,
    observer: str = "earth",
    cmap: str = "turbo",
    opacity: float = 0.88,
    outside_domain_color: str = "lightgray",
    outside_domain_opacity: float = 0.10,
    quasi_perpendicular_threshold_deg: float = 70.0,
    scalar_bar_title: str = "theta_Bn [deg]",
) -> Dict[str, object]:
    """Color the PFSS-valid portion of the Spheroid by theta_Bn."""
    theta_surface, summary = compute_spheroid_theta_bn_surface(
        spheroid_surface_hgc,
        pfss_output,
        obstime_iso=obstime_iso,
        observer=observer,
        quasi_perpendicular_threshold_deg=quasi_perpendicular_threshold_deg,
    )

    # Draw the complete Spheroid faintly so the PFSS outer boundary is visually
    # distinguishable from a missing Spheroid surface.
    plotter.add_mesh(
        theta_surface,
        color=outside_domain_color,
        opacity=float(outside_domain_opacity),
        smooth_shading=True,
        lighting=True,
        pickable=False,
    )

    valid_mask = np.isfinite(
        np.asarray(
            theta_surface["theta_Bn_deg"],
            dtype=float,
        )
    )

    if not np.any(valid_mask):
        print(
            "[WARN] No Spheroid vertices lie inside "
            "the valid PFSS domain."
        )
        return {
            "surface": theta_surface,
            "valid_surface": None,
            "summary": summary,
        }

    # Keep only cells whose vertices all have valid theta_Bn values.
    # This avoids interpolating colors across the PFSS
    # source-surface boundary.
    try:
        valid_surface = theta_surface.threshold(
            value=(float(quasi_perpendicular_threshold_deg), 90.0),
            scalars="theta_Bn_deg",
            preference="point",
            all_scalars=True,
        )
    except TypeError:
        valid_surface = theta_surface.threshold(
            value=(float(quasi_perpendicular_threshold_deg), 90.0),
            scalars="theta_Bn_deg",
            preference="point",
        )

    if valid_surface.n_points == 0:
        print(
            "[WARN] theta_Bn values were computed, "
            "but no all-valid surface cells remained."
        )
        return {
            "surface": theta_surface,
            "valid_surface": valid_surface,
            "summary": summary,
        }

    scalar_bar_args = {
        "title": scalar_bar_title,
        "vertical": True,
        "position_x": 0.04,
        "position_y": 0.18,
        "width": 0.08,
        "height": 0.60,
        "title_font_size": 14,
        "label_font_size": 12,
        "fmt": "%.0f",
        "color": "black",
    }

    plotter.add_mesh(
        valid_surface,
        scalars="theta_Bn_deg",
        cmap=cmap,
        clim=(
            float(quasi_perpendicular_threshold_deg),
            90.0,
        ),
        opacity=float(opacity),
        smooth_shading=True,
        lighting=True,
        scalar_bar_args=scalar_bar_args,
        pickable=False,
    )

    return {
        "surface": theta_surface,
        "valid_surface": valid_surface,
        "summary": summary,
    }

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

    x_min_pix, x_max_pix = center_x - 512, center_x + 100
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
    line_width: int = 5,
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
        "pfss_output": pfss_output,
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
    line_width: int = 5,
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

def _npz_first_existing_path(candidates: List[Path]) -> Path:
    """Return the first existing path, or raise a useful FileNotFoundError."""
    for path in candidates:
        if path.exists():
            return path

    msg = "Precomputed tomography NPZ was not found. Tried:\n"
    msg += "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(msg)


def _npz_get_first(npz, keys: Tuple[str, ...]):
    """Return the first array whose key exists in an np.load(...)."""
    for key in keys:
        if key in npz.files:
            return npz[key], key
    raise KeyError(f"None of the required keys were found in NPZ: {keys}. Available keys: {npz.files}")


def _scalar_from_npz(npz, keys: Tuple[str, ...], default=None):
    """Return a scalar value from an NPZ file if any of the keys exists."""
    for key in keys:
        if key in npz.files:
            arr = np.asarray(npz[key])
            if arr.size == 0:
                continue
            val = arr.reshape(-1)[0]
            if isinstance(val, bytes):
                return val.decode()
            return val.item() if hasattr(val, "item") else val
    return default


def _simple_observer_from_npz_or_time(npz, observer_time_iso: str) -> SimpleObserver:
    """
    Build the minimal observer object required by the overlay routines.

    Priority:
      1) Use forced Earth-view rendering camera lon/lat saved in the NPZ.
      2) Use observer Carrington lon/lat saved in the NPZ if present.
      3) Otherwise compute Earth's Carrington lon/lat at observer_time_iso.
      4) If SunPy ephemeris conversion fails, fall back to (0, 0) with a warning.
    """
    render_lon = _scalar_from_npz(
        npz,
        ("render_camera_lon_deg",),
        default=None,
    )
    render_lat = _scalar_from_npz(
        npz,
        ("render_camera_lat_deg",),
        default=None,
    )
    render_is_earth = _scalar_from_npz(
        npz,
        ("render_camera_is_earth_view",),
        default=False,
    )

    if render_lon is not None and render_lat is not None and bool(render_is_earth):
        lon_f = float(render_lon) % 360.0
        lat_f = float(render_lat)
        print(
            "[INFO] Observer Carrington lon/lat loaded from NPZ Earth-view render camera: "
            f"({lon_f:.3f}, {lat_f:.3f}) deg"
        )
        return SimpleObserver(lonlat_deg=(lon_f, lat_f))

    lon = _scalar_from_npz(
        npz,
        ("obs_lon_deg", "observer_lon_deg", "crln_obs", "CRLN_OBS", "lon_obs_deg"),
        default=None,
    )
    lat = _scalar_from_npz(
        npz,
        ("obs_lat_deg", "observer_lat_deg", "crlt_obs", "CRLT_OBS", "lat_obs_deg"),
        default=None,
    )

    if lon is None or lat is None:
        for key in ("obs_lonlat_deg", "observer_lonlat_deg", "lonlat_deg"):
            if key in npz.files:
                arr = np.asarray(npz[key], dtype=float).reshape(-1)
                if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                    lon, lat = float(arr[0]), float(arr[1])
                    break

    if lon is not None and lat is not None:
        lon_f = float(lon) % 360.0
        lat_f = float(lat)
        print(f"[INFO] Observer Carrington lon/lat loaded from NPZ: ({lon_f:.3f}, {lat_f:.3f}) deg")
        return SimpleObserver(lonlat_deg=(lon_f, lat_f))

    try:
        from sunpy.coordinates import get_body_heliographic_stonyhurst

        t = Time(observer_time_iso)
        earth_hgs = get_body_heliographic_stonyhurst("earth", t)
        try:
            hgc_frame = frames.HeliographicCarrington(obstime=t, observer="earth")
        except TypeError:
            hgc_frame = frames.HeliographicCarrington(obstime=t)
        earth_hgc = earth_hgs.transform_to(hgc_frame)
        lon_f = float(earth_hgc.lon.to_value(u.deg)) % 360.0
        lat_f = float(earth_hgc.lat.to_value(u.deg))
        print(
            f"[INFO] Observer Carrington lon/lat computed from time "
            f"{observer_time_iso}: ({lon_f:.3f}, {lat_f:.3f}) deg"
        )
        return SimpleObserver(lonlat_deg=(lon_f, lat_f))
    except Exception as exc:
        print(
            "[WARN] Could not derive observer Carrington lon/lat from NPZ or SunPy "
            f"({exc}). Falling back to lon/lat=(0, 0) deg."
        )
        return SimpleObserver(lonlat_deg=(0.0, 0.0))

def _build_npz_path_candidates(
    *,
    npz_dir: Path,
    target_tag: str,
    window_tag: str,
    frequency_mhz: List[float],
    harmonic: int,
    other_tag: str = None
) -> List[Path]:
    """Build robust NPZ filename candidates around the requested filename pattern."""
    freq_tags = []
    variants = [
        "-".join(str(float(f)) for f in frequency_mhz),
        "-".join(f"{float(f):g}" for f in frequency_mhz),
        "-".join(str(f) for f in frequency_mhz),
    ]
    for tag in variants:
        if tag not in freq_tags:
            freq_tags.append(tag)

    out: List[Path] = []
    for freq_tag in freq_tags:
        if other_tag is None:
            # Requested naming convention.
            out.append(
                # npz_dir / f"ne3d_{target_tag}_{window_tag}_{freq_tag}MHz_h{int(harmonic)}.npz"
                npz_dir / f"ne3d_{target_tag}_{window_tag}_{freq_tag}MHz_h{int(harmonic)}_backup.npz"
            )
            # Backward-compatible fallback for files produced by the previous overlay code.
            out.append(
                npz_dir / f"ne3d_solution_{target_tag}_{window_tag}_{freq_tag}MHz.npz"
            )
            # out.append(npz_dir / f"time_window_{window_tag}_earth_cor1a_ne3d_solution.npz")
            # out.append(npz_dir / f"time_window_{window_tag}_earth_only_ne3d_solution.npz")
            # out.append(npz_dir / f"time_window_{window_tag}_cor1a_only_ne3d_solution.npz")
        elif other_tag is not None:
            out.append(
                npz_dir / f"ne3d_solution_{target_tag}_{window_tag}_{freq_tag}MHz_{other_tag}.npz"
            )
            out.append(
                npz_dir.parent / f"ne3d_solution_{target_tag}_{window_tag}_{freq_tag}MHz_{other_tag}.npz"
            )
            
    if freq_tags is not None and len(freq_tags) >= 2:
        out.append(
            npz_dir / f"ne3d_solution_{target_tag}_{window_tag}_{freq_tags[0]}-{freq_tags[-1]}MHz.npz"
        )
    return out


def load_tomography_from_npz(Frequency_MHz: List[float], other_tag: str = None, TARGET_TIME: str = "20220613_030000", SEARCH_WINDOW_DAYS: float = 7.0, HARMONIC: int = 2) -> Dict[str, Any]:
    """
    Load the precomputed tomography solution from NPZ and return objects needed
    by the Spheroid/PFSS overlay.

    Expected primary filename pattern:
      /mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/ne_npz/
      ne3d_{TARGET_TAG}_{WINDOW_TAG}_{FREQ_TAG}MHz_h{int(HARMONIC)}.npz

    Required NPZ arrays:
      - ne
      - r_edges
      - th_edges
      - ph_edges

    Optional NPZ arrays/metadata:
      - harmonic
      - target_time or observer_time_iso
      - obs_lon_deg/obs_lat_deg or obs_lonlat_deg
    """
    # TARGET_TIME = "20220613_030000"
    # SEARCH_WINDOW_DAYS = 7.0
    # HARMONIC = 2
    NPZ_DIR = Path(
        "/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/ne_npz",
        # f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/step1_timewindow_viewpoint_9cases_20220613_030000_33MHz/time_window_{int(SEARCH_WINDOW_DAYS)}d_cor1a_only/"
    )
    OTHER_TAG = other_tag

    target_dt = parse_target_datetime(TARGET_TIME)
    TARGET_TAG = target_dt.strftime("%Y%m%d_%H%M%S")
    WINDOW_TAG = f"pm{int(SEARCH_WINDOW_DAYS)}d"

    npz_candidates = _build_npz_path_candidates(
        npz_dir=NPZ_DIR,
        target_tag=TARGET_TAG,
        window_tag=WINDOW_TAG,
        frequency_mhz=Frequency_MHz,
        harmonic=HARMONIC,
        other_tag=OTHER_TAG
    )
    npz_path = _npz_first_existing_path(npz_candidates)

    print("[INFO] Loading precomputed tomography NPZ:")
    print(f"       {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        ne_arr, ne_key = _npz_get_first(data, ("ne", "ne_scaled", "electron_density", "density", "ne_cm3"))
        r_edges, _ = _npz_get_first(data, ("r_edges",))
        th_edges, _ = _npz_get_first(data, ("th_edges", "theta_edges"))
        ph_edges, _ = _npz_get_first(data, ("ph_edges", "phi_edges"))

        r_edges = np.asarray(r_edges, dtype=np.float64).reshape(-1)
        th_edges = np.asarray(th_edges, dtype=np.float64).reshape(-1)
        ph_edges = np.asarray(ph_edges, dtype=np.float64).reshape(-1)
        grid = SphericalGrid(r_edges=r_edges, th_edges=th_edges, ph_edges=ph_edges)

        ne_arr = np.asarray(ne_arr, dtype=np.float64)
        if ne_arr.size != grid.nvox:
            raise ValueError(
                f"NPZ density array '{ne_key}' has size {ne_arr.size}, but grid.nvox={grid.nvox}. "
                f"shape={ne_arr.shape}, grid=({grid.nr}, {grid.nth}, {grid.nph})."
            )
        ne = ne_arr.reshape(-1, order="C")

        harmonic_saved = _scalar_from_npz(data, ("harmonic", "HARMONIC"), default=HARMONIC)
        harmonic = int(harmonic_saved)

        observer_time = _scalar_from_npz(
            data,
            ("observer_time_iso", "target_time_iso", "target_time", "TARGET_TIME"),
            default=target_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        observer_time_iso = str(observer_time)
        try:
            observer_time_iso = parse_target_datetime(observer_time_iso).strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            pass

        obs_ref = _simple_observer_from_npz_or_time(data, observer_time_iso)

    pos = np.isfinite(ne) & (ne > 0)
    if np.any(pos):
        print(
            f"[INFO] Loaded density key='{ne_key}', grid=({grid.nr}, {grid.nth}, {grid.nph}), "
            f"ne range={np.nanmin(ne[pos]):.3e}..{np.nanmax(ne[pos]):.3e} cm^-3"
        )
    else:
        print(f"[WARN] Loaded density key='{ne_key}', but no positive finite density values were found.")

    return {
        "grid": grid,
        "ne": ne,
        "obs_ref": obs_ref,
        "target_time": TARGET_TIME,
        "harmonic": harmonic,
        "save_ne_npz": npz_path,
    }


def _colors_for_iso_freqs(iso_freq_mhz: List[float]) -> List[str]:
    """Return stable colors for one or more iso-frequency surfaces."""
    if iso_freq_mhz == [33]:
        return ["gold"]
    if iso_freq_mhz == [31.5]:
        return ["cyan"]
    if iso_freq_mhz == [28.0]:
        return ["tomato"]
    base = ["gold", "cyan", "tomato", "deepskyblue", "limegreen", "violet", "orange"]
    n = len(iso_freq_mhz)
    return (base * ((n + len(base) - 1) // len(base)))[:n]


def main(
    Frequency_MHz: List[float],
    time_iso: str,
    time_tomography: float,
    rss: float,
    r_scatter: float,
    r_scatter_dr: float,
    spheroid_params: Optional[SpheroidDome3DParams] = None,
    other_tag: str = None,
    time_window: float = 7.0,
    harmonic: int = 2,
    THETA_BN_QUASI_PERP_DEG: float = 30.0
):
    OBSTIME_ISO = time_iso

    # ---- Runtime timers ----
    t_run_start = time.perf_counter()

    def _log_elapsed(label: str, t0: float, *, since_start: bool = False) -> float:
        now = time.perf_counter()
        if since_start:
            print(f"[TIME] {label}: {now - t_run_start:.2f} s since script start")
        else:
            print(f"[TIME] {label}: {now - t0:.2f} s")
        return now

    print(f"[TIME] main() started for {OBSTIME_ISO}")

    # ---- Tomography: load precomputed NPZ instead of recomputing inversion ----
    t_tomo_start = time.perf_counter()
    tomo_result = load_tomography_from_npz(
        Frequency_MHz,
        other_tag,
        TARGET_TIME=time_tomography,
        SEARCH_WINDOW_DAYS=time_window,
        HARMONIC=harmonic,
    )
    grid = tomo_result["grid"]
    ne = tomo_result["ne"]
    obs0 = tomo_result["obs_ref"]
    _log_elapsed("Tomography NPZ load", t_tomo_start, since_start=False)
    _log_elapsed("Elapsed after tomography NPZ load", t_run_start, since_start=True)

    ISO_FREQ_MHZ = Frequency_MHz
    HARMONIC = int(tomo_result["harmonic"])

    # ---- PFSS/Spheroid parameters for overlay ----
    SPHEROID_OBSERVER = "earth"

    DO_PFSS = True
    DO_THETA_BN = True
    HMI_FITS = "/mnt/d/wsl/home/kinno-7010/Research_data/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    PFSS_RSS = rss
    PFSS_NRHO = 80

    PFSS_SEED_N_LON = 50
    PFSS_SEED_N_LAT = 50
    PFSS_MAX_LINES = 300
    PFSS_FIELD_THRESHOLD = 50.0

    THETA_BN_CMAP = "turbo"
    THETA_BN_OPACITY = 1.0
    # THETA_BN_QUASI_PERP_DEG = 30.0

    SHOW_SUN = True
    SHOW_GUI = True
    SAVE_PNG = True
    SHOW_RADIAL_BAND = True
    OTHER_TAG = "" if other_tag is None else f"_{other_tag}"
    PNG_PATH = Path(
        f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/output/multi-tomo/thetaBn/"
        f"tomo_sphe_thetaBn{int(THETA_BN_QUASI_PERP_DEG)}_{'-'.join(str(f) for f in ISO_FREQ_MHZ)}MHz_"
        f"{time_tomography.replace(':', '')}_pm{time_window}d{OTHER_TAG}.png"
    )

    print(f"[INFO] ISO_FREQ_MHZ={ISO_FREQ_MHZ}, harmonic={HARMONIC}")

    # ---- Build PyVista grid ----
    t_grid_start = time.perf_counter()
    sg = build_tomography_structured_grid(grid, ne)
    print(f"[INFO] StructuredGrid bounds: {sg.bounds}")
    _log_elapsed("PyVista StructuredGrid build", t_grid_start)

    # ---- Plot ----
    t_scene_start = time.perf_counter()
    off_screen = not SHOW_GUI
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

    if SHOW_SUN:
        p.add_mesh(
            pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60),
            opacity=0.2,
            color="grey",
        )

    colors = _colors_for_iso_freqs(ISO_FREQ_MHZ)
    tomo_surfs = add_isosurfaces(
        p,
        sg,
        ISO_FREQ_MHZ,
        harmonic=HARMONIC,
        opacity=0.2,
        colors=colors,
        return_surfaces=True,
        range_text_mode="runinfo",
    )

    add_solar_latlon_grid(
        p,
        radius=1.002,
        dlon_deg=30.0,
        dlat_deg=30.0,
        line_width=2,
        opacity=0.6,
    )
    add_sun_earth_line(
        p,
        obs0,
        length_rsun=5.0,
        start_rsun=1.0,
        color="orange",
        line_width=5,
    )

    add_physical_axes_triad(
        p,
        obs0,
        origin_rsun=1.0,
        axis_len=1.6,
        shaft_radius=0.04,
        tip_radius=0.08,
        tip_length=0.25,
        label_font_size=12,
    )

    pfss_info = None
    pfss_output = None
    if DO_PFSS:
        t_pfss_start = time.perf_counter()
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
                line_width=5,
                opacity=1.0,
                open_color="red",
                closed_color="black",
                prefer_fortran=True,
            )
            pfss_output = pfss_info.get("pfss_output")
        except Exception as e:
            print(f"[WARN] PFSS overlay skipped due to error: {e}")
            pfss_info = None
            pfss_output = None
        finally:
            _log_elapsed("PFSS calculation/overlay", t_pfss_start)

    if spheroid_params is not None:
        t_spheroid_start = time.perf_counter()
        spheroid_cross = compute_spheroid_meridian_parallel_crosspoints_hgc(
            spheroid_params,
            obstime_iso=OBSTIME_ISO,
            observer=SPHEROID_OBSERVER,
            n_resample=200,
            cross_tol_rsun=0.05,
            merge_tol_rsun=0.03,
        )

        # Build the surface once. The wireframe/footprint function is called with
        # show_surface=False so the theta_Bn colormap is not obscured by magenta.
        spheroid_surface = build_spheroid_surface_hgc(
            spheroid_params,
            obstime_iso=OBSTIME_ISO,
            observer=SPHEROID_OBSERVER,
        )

        add_spheroid_dome_3d(
            p,
            spheroid_params,
            obstime_iso=OBSTIME_ISO,
            observer=SPHEROID_OBSERVER,
            obs0=obs0,
            color="magenta",
            surface_opacity=0.14,
            wire_opacity=0.7,
            footprint_opacity=1.0,
            line_width=1,
            footprint_width=4,
            marker_radius=0.045,
            show_surface=False,
            show_wireframe=True,
            show_footprint=True,
            show_markers=True,
            return_surface=False,
        )

        theta_bn_result = None
        if spheroid_surface is not None and spheroid_surface.n_points > 0:
            if DO_THETA_BN and pfss_output is not None:
                try:
                    theta_bn_result = add_spheroid_theta_bn_colormap(
                        p,
                        spheroid_surface,
                        pfss_output,
                        obstime_iso=OBSTIME_ISO,
                        observer=SPHEROID_OBSERVER,
                        cmap=THETA_BN_CMAP,
                        opacity=THETA_BN_OPACITY,
                        outside_domain_color="lightgray",
                        outside_domain_opacity=0.10,
                        quasi_perpendicular_threshold_deg=THETA_BN_QUASI_PERP_DEG,
                        scalar_bar_title="theta_Bn [deg]",
                    )
                    summary = theta_bn_result["summary"]
                    pending = getattr(p, "_pending_runinfo_lines", [])
                    theta_line = (
                        f"theta_Bn: median={summary['theta_median_deg']:.1f} deg, "
                        f">={THETA_BN_QUASI_PERP_DEG:g} deg: "
                        f"{100.0 * summary['quasi_perpendicular_fraction']:.1f}%"
                    )
                    if theta_line not in pending:
                        pending.append(theta_line)
                    p._pending_runinfo_lines = pending
                except Exception as exc:
                    print(f"[WARN] theta_Bn colormap skipped: {exc}")
                    p.add_mesh(
                        spheroid_surface,
                        color="magenta",
                        opacity=0.14,
                        smooth_shading=True,
                        lighting=True,
                        pickable=False,
                    )
            else:
                if DO_THETA_BN and pfss_output is None:
                    print("[WARN] theta_Bn requires a valid PFSS output; drawing the Spheroid without theta_Bn colors.")
                p.add_mesh(
                    spheroid_surface,
                    color="magenta",
                    opacity=0.14,
                    smooth_shading=True,
                    lighting=True,
                    pickable=False,
                )

            # if SHOW_RADIAL_BAND:
            #     add_surface_at_radius(
            #         p,
            #         spheroid_surface,
            #         r0=r_scatter,
            #         dr=r_scatter_dr,
            #         mode="band",
            #         color="#0011ff",
            #         opacity=0.35,
            #         label="Spheroid",
            #     )
        else:
            print("[WARN] Spheroid surface could not be built.")

        add_spheroid_tomography_overlap_points(
            p,
            spheroid_cross_points_hgc=spheroid_cross,
            tomo_isosurfaces=tomo_surfs,
            tol_rsun=0.10,
            colors=colors,
            frequencies_mhz=ISO_FREQ_MHZ,
            point_size=14,
            label="Spheroid",
        )

        _log_elapsed("Spheroid overlay + theta_Bn", t_spheroid_start)

    # Add run information after theta_Bn/radial-band diagnostics so all pending
    # lines are included in a single text actor.
    add_runinfo_legend(
        p,
        obstime_iso=OBSTIME_ISO,
        iso_freqs_mhz=ISO_FREQ_MHZ,
        harmonic=HARMONIC,
        spheroid_params=spheroid_params,
        pfss_params=pfss_info,
        position="upper_right",
        font_size=12,
    )

    t_camera_start = time.perf_counter()
    set_camera_from_observation(p, obs0, distance_rsun=4.0)

    try:
        p.reset_camera_clipping_range()
        p.render()
    except Exception:
        pass
    t_render_ready = _log_elapsed("Camera setup + first render", t_camera_start)
    _log_elapsed("Total PyVista scene construction", t_scene_start)
    print(
        "[TIME] Total time from execution start to PyVista-display-ready: "
        f"{t_render_ready - t_run_start:.2f} s"
    )

    if SAVE_PNG:
        PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if off_screen:
            t_png_start = time.perf_counter()
            p.show(screenshot=str(PNG_PATH), auto_close=True)
            _log_elapsed("PNG screenshot save", t_png_start)
            print(f"[OK] Saved: {PNG_PATH}")
        else:
            print(
                f"[TIME] Calling PyVista interactive display at "
                f"{time.perf_counter() - t_run_start:.2f} s since script start"
            )
            print("[INFO] Close the PyVista window to save the final camera view as PNG.")
            t_show_start = time.perf_counter()
            p.show(auto_close=False)
            _log_elapsed("PyVista interactive window duration", t_show_start)

            t_png_start = time.perf_counter()
            try:
                p.screenshot(str(PNG_PATH))
                _log_elapsed("PNG screenshot save after interactive view", t_png_start)
                print(f"[OK] Saved final PyVista view: {PNG_PATH}")
            finally:
                try:
                    p.close()
                except Exception:
                    pass
    else:
        print(
            f"[TIME] Calling PyVista display at "
            f"{time.perf_counter() - t_run_start:.2f} s since script start"
        )
        t_show_start = time.perf_counter()
        p.show()
        _log_elapsed("PyVista window duration", t_show_start)


if __name__ == "__main__":
    time_iso = "2022-06-13T03:26:29"
    
    TIME_TOMOGRAPHY = "2022-06-13T03:00:00"
    Frequency_MHz = [33, 43]
    TIME_WINDOW_DAYS = 5.0
    HARMONIC = 2
    
    OTHER_TAG = None
    # OTHER_TAG = "no-weight"
    THETA_BN_QUASI_PERP_DEG = 60
    
    rss = 2.5

    # ---- Spheroid parameters----
    spheroid_anchor_lon_deg = -30.0
    spheroid_anchor_lat_deg = +19.0
    spheroid_apex_lon_deg = -55.0
    spheroid_apex_lat_deg = +5.0
    spheroid_apex_rsun = 3.27
    spheroid_kappa = 0.50
    spheroid_epsilon = -0.45
    r_scatter = 1.8 # blue belt
    r_scatter_dr = 0.1
    
    
    spheroid = SpheroidDome3DParams(
        kappa=float(spheroid_kappa),
        epsilon=float(spheroid_epsilon),
        anchor_lon_deg=float(spheroid_anchor_lon_deg),
        anchor_lat_deg=float(spheroid_anchor_lat_deg),
        apex_lon_deg=float(spheroid_apex_lon_deg),
        apex_lat_deg=float(spheroid_apex_lat_deg),
        apex_r_rsun=float(spheroid_apex_rsun),
        n_meridians=72,
        n_parallels=72,
        n_line_pts=360,
        only_above_surface=False,
        only_visible=True,
    )

    main(
        Frequency_MHz=Frequency_MHz,
        time_iso=time_iso,
        time_tomography = TIME_TOMOGRAPHY,
        rss=rss,
        r_scatter=r_scatter,
        r_scatter_dr=r_scatter_dr,
        spheroid_params=spheroid,
        other_tag=OTHER_TAG,
        time_window=TIME_WINDOW_DAYS,
        harmonic=HARMONIC,
        THETA_BN_QUASI_PERP_DEG=THETA_BN_QUASI_PERP_DEG
    )
    
        
