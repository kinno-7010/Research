"""plot_spheroid_C2_clean.py

K-COR + LASCO-C2 (and AIA193 background) composite difference image produced by
`integrated_analysis.create_single_diff_image()` with a 3D spheroid dome overlay.

Design notes (aligned with `aia_spheroid_plot.py`):
- Spheroid is a center-of-symmetry prolate spheroid (a,b,b).
- Axis geometry can be specified by center-based radial mode or
  two-point mode (anchor + apex lon/lat + apex height/radius).
- Optional clipping:
    * only_above_surface: keep r >= 1
    * only_visible: keep points on the visible hemisphere (approx. by observer vector dot-product)
- If wireframe becomes empty with only_visible=True, automatically retry with only_visible=False.

This script intentionally contains NO GCS code.
"""

from __future__ import annotations

import os, sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, CartesianRepresentation

import sunpy.map
from sunpy.coordinates import frames as sunpy_frames

# --- integrated_analysis import (prefer local) ---
sys.path.append("/home/kinno-7010/Research_code/SDO_Mk4_SOHO/py_folder")
try:
    from integrated_analysis import create_single_diff_image, create_single_diff_from_time_image
except Exception as exc:
    raise ImportError(
        "Failed to import create_single_diff_image from integrated_analysis.py. "
        "Place integrated_analysis.py next to this script or make it importable via PYTHONPATH."
    ) from exc
from plot_cor_csv import create_single_diff_from_csv_image

# ==========================================================
# Spheroid parameters (same model as in aia_gcs_plot.py)
# ==========================================================

@dataclass
class SpheroidDome3DParams:
    """Prolate spheroid dome in HGS parameterized by (kappa, epsilon).

    The semi-axes are derived from the apex height h = r_apex - 1 as
        b = kappa * h
        a = b / sqrt(1 - epsilon^2)
    where ``a`` is the radial semi-axis and ``b`` is the transverse semi-axis.

    Axis geometry can be specified in two ways:
    (A) center-based radial mode:
        center_lon_deg, center_lat_deg, and apex_r_rsun or apex_height_rsun
    (B) two-point axis mode:
        anchor_lon_deg, anchor_lat_deg, apex_lon_deg, apex_lat_deg,
        and apex_r_rsun or apex_height_rsun
    """

    kappa: float
    epsilon: float

    # (B) Two-point axis specification
    anchor_lon_deg: float | None = None
    anchor_lat_deg: float | None = None
    apex_lon_deg: float | None = None
    apex_lat_deg: float | None = None
    apex_r_rsun: float | None = None
    apex_height_rsun: float | None = None

    # (A) Center-based specification (legacy radial axis)
    center_lon_deg: float | None = None
    center_lat_deg: float | None = None
    center_r_rsun: float | None = None

    n_meridians: int = 12
    n_parallels: int = 7
    n_line_pts: int = 240

    only_above_surface: bool = True
    only_visible: bool = True

    def _has_two_point_axis(self) -> bool:
        return (
            (self.anchor_lon_deg is not None)
            and (self.anchor_lat_deg is not None)
            and (self.apex_lon_deg is not None)
            and (self.apex_lat_deg is not None)
            and ((self.apex_r_rsun is not None) or (self.apex_height_rsun is not None))
        )

    def _has_center_axis(self) -> bool:
        return (
            (self.center_lon_deg is not None)
            and (self.center_lat_deg is not None)
            and ((self.apex_r_rsun is not None) or (self.apex_height_rsun is not None) or (self.center_r_rsun is not None))
        )

    @property
    def b_rsun(self) -> float:
        return float(self._b_rsun)

    @property
    def a_rsun(self) -> float:
        return float(self._a_rsun)

    def __post_init__(self) -> None:
        if self.apex_r_rsun is None and self.apex_height_rsun is not None:
            self.apex_r_rsun = float(1.0 + self.apex_height_rsun)
        elif self.apex_r_rsun is not None and self.apex_height_rsun is None:
            self.apex_height_rsun = float(self.apex_r_rsun - 1.0)

        if self.apex_height_rsun is None:
            raise ValueError(
                "SpheroidDome3DParams: apex_r_rsun or apex_height_rsun is required when using kappa and epsilon."
            )

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

        # Signed-epsilon convention:
        #   epsilon < 0 : prolate along the symmetry axis (a > b)
        #   epsilon > 0 : oblate  along the symmetry axis (a < b)
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

    def legend_label(self) -> str:
        if self._has_two_point_axis():
            return (
                f"Spheroid: $\\kappa$={self.kappa:.3f}, "
                f"$\\epsilon$={self.epsilon:.3f}, "
                f"apex(lon,lat)=({float(self.apex_lon_deg):.1f},{float(self.apex_lat_deg):.1f})$^\\circ$, "
                f"r_apex={float(self.apex_r_rsun):.2f} R$_\\odot$"
            )
        return (
            f"Spheroid: $\\kappa$={self.kappa:.3f}, "
            f"$\\epsilon$={self.epsilon:.3f}, "
            f"center r={self.center_r_rsun:.2f} R$_\\odot$, "
            f"(lon,lat)=({self.center_lon_deg:.1f},{self.center_lat_deg:.1f})$^\\circ$"
        )

# ==========================================================
# Geometry helpers
# ==========================================================

def _hgs_unit_vectors(lon_deg: float, lat_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vectors at (lon,lat) in HGS cartesian basis.

    Returns
    -------
    r_hat : radial unit vector
    e_lon : unit vector in increasing longitude direction
    e_lat : unit vector in increasing latitude direction
    """
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


def _split_skycoord_by_mask(coords: SkyCoord, mask: np.ndarray) -> list[SkyCoord]:
    """Split SkyCoord into contiguous True segments to avoid line bridging."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []

    cuts = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[0], cuts + 1])
    ends = np.concatenate([cuts + 1, [idx.size]])

    segs: list[SkyCoord] = []
    for s, e in zip(starts, ends):
        seg_idx = idx[s:e]
        if seg_idx.size >= 2:
            segs.append(coords[seg_idx])
    return segs


def _cart_rsun_from_lonlat(lon_deg: float, lat_deg: float, r_rsun: float) -> np.ndarray:
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    cosl, sinl = np.cos(lon), np.sin(lon)
    cosb, sinb = np.cos(lat), np.sin(lat)
    return np.array([r_rsun * cosb * cosl, r_rsun * cosb * sinl, r_rsun * sinb], dtype=float)

def _lonlat_from_cart_rsun(cart_rsun: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(cart_rsun, dtype=float).reshape(3)
    x, y, z = float(arr[0]), float(arr[1]), float(arr[2])
    r = float(np.sqrt(x * x + y * y + z * z))
    if r <= 0.0:
        raise ValueError("zero-radius cartesian point cannot be converted to Stonyhurst lon/lat.")
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


def _orthonormal_basis_from_axis(axis_u: np.ndarray, ref_vec: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    axis_u = _unit_vec(axis_u)
    if ref_vec is None:
        ref_vec = np.array([0.0, 0.0, 1.0], dtype=float)
    ref = np.asarray(ref_vec, dtype=float)
    ref = ref - np.dot(ref, axis_u) * axis_u
    if np.linalg.norm(ref) < 1e-10:
        alt = np.array([1.0, 0.0, 0.0], dtype=float) if abs(axis_u[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
        ref = alt - np.dot(alt, axis_u) * axis_u
    e1 = _unit_vec(ref)
    e2 = _unit_vec(np.cross(axis_u, e1))
    return e1, e2


def _spheroid_axis_geometry_rsun(params: SpheroidDome3DParams) -> dict[str, np.ndarray]:
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


def _sample_footprint_cart_rsun(params: SpheroidDome3DParams, n_beta: int | None = None) -> np.ndarray:
    geom = _spheroid_axis_geometry_rsun(params)
    center = geom["center"]
    axis_u = geom["axis_u"]
    e1 = geom["e1"]
    e2 = geom["e2"]

    if n_beta is None:
        n_beta = int(params.n_line_pts)

    betas = np.linspace(0.0, 2.0 * np.pi, n_beta, endpoint=False)
    alpha_grid = np.linspace(0.0, np.pi, 1201)
    pts: list[np.ndarray] = []

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

        roots: list[float] = []
        hit = np.where(np.isclose(f, 0.0, atol=1e-5))[0]
        for idx in hit:
            roots.append(float(alpha_grid[idx]))

        s = np.sign(f)
        for i in range(len(alpha_grid) - 1):
            if s[i] == 0 or s[i + 1] == 0:
                continue
            if f[i] * f[i + 1] < 0:
                a0 = float(alpha_grid[i] + (0.0 - f[i]) * (alpha_grid[i + 1] - alpha_grid[i]) / (f[i + 1] - f[i]))
                roots.append(a0)

        if not roots:
            continue

        alpha0 = max(roots, key=lambda a: np.sin(a))
        p = center + float(params.a_rsun) * np.cos(alpha0) * axis_u + float(params.b_rsun) * np.sin(alpha0) * dir_perp
        if abs(np.linalg.norm(p) - 1.0) < 5e-3:
            pts.append(p)

    if not pts:
        return np.zeros((3, 0), dtype=float)
    return np.stack(pts, axis=1)


# ==========================================================
# Spheroid -> HPC sampling (same logic as aia_spheroid_plot.py)
# ==========================================================

def _visible_mask(
    coords_hgs: SkyCoord,
    reference_map: sunpy.map.Map | None = None,
    *,
    only_visible: bool = True,
) -> np.ndarray:
    if not only_visible or reference_map is None:
        return np.ones(coords_hgs.shape, dtype=bool)
    try:
        obs_vec = reference_map.observer_coordinate.cartesian.xyz.to_value(u.R_sun)
        pt_vec = coords_hgs.cartesian.xyz.to_value(u.R_sun)
        s = np.sign(obs_vec[0]) if obs_vec[0] != 0 else 1.0
        dot_sum = np.sum(obs_vec[:, None] * pt_vec, axis=0)
        return (dot_sum * s > 0)
    except Exception as exc:
        print(f"[WARN] visibility mask disabled (visible check failed): {exc}")
        return np.ones(coords_hgs.shape, dtype=bool)


def spheroid_dome_apex_hpc(params: SpheroidDome3DParams, reference_map: sunpy.map.Map) -> SkyCoord:
    geom = _spheroid_axis_geometry_rsun(params)
    cart_rsun = geom["apex"]
    rep = CartesianRepresentation(cart_rsun[0] * u.R_sun, cart_rsun[1] * u.R_sun, cart_rsun[2] * u.R_sun)
    apex_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)
    return apex_hgs.transform_to(reference_map.coordinate_frame)


def spheroid_axis_footpoint_hpc(params: SpheroidDome3DParams, reference_map: sunpy.map.Map) -> SkyCoord:
    geom = _spheroid_axis_geometry_rsun(params)
    cart_rsun = geom["anchor"]
    rep = CartesianRepresentation(cart_rsun[0] * u.R_sun, cart_rsun[1] * u.R_sun, cart_rsun[2] * u.R_sun)
    fp_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)
    return fp_hgs.transform_to(reference_map.coordinate_frame)


def spheroid_footprint_angular_radius_deg(params: SpheroidDome3DParams) -> float | None:
    if params._has_two_point_axis():
        pts = _sample_footprint_cart_rsun(params)
        if pts.shape[1] < 8:
            return None
        anchor_u = _unit_vec(_spheroid_axis_geometry_rsun(params)["anchor"])
        vv = _unit_vec(pts)
        cosang = np.clip(anchor_u @ vv, -1.0, 1.0)
        ang = np.arccos(cosang)
        return float(np.rad2deg(np.median(ang)))

    a = float(params.a_rsun)
    b = float(params.b_rsun)
    cr = float(params.center_r_rsun)

    A = (a * a - b * b)
    B = 2.0 * cr * a
    C = (cr * cr + b * b - 1.0)

    candidates: list[float] = []
    if abs(A) < 1e-12:
        if abs(B) < 1e-12:
            return None
        candidates = [(-C / B)]
    else:
        disc = B * B - 4.0 * A * C
        if disc < 0:
            return None
        sdisc = float(np.sqrt(disc))
        candidates = [(-B + sdisc) / (2.0 * A), (-B - sdisc) / (2.0 * A)]

    valid = [c for c in candidates if np.isfinite(c) and (-1.0 <= c <= 1.0)]
    if not valid:
        return None

    best_c = max(valid, key=lambda cc: np.sqrt(max(0.0, 1.0 - cc * cc)))
    cos_psi = cr + a * best_c
    cos_psi = float(np.clip(cos_psi, -1.0, 1.0))
    psi = float(np.arccos(cos_psi))
    return float(np.rad2deg(psi))


def sample_spheroid_footprint_hpc(params: SpheroidDome3DParams, reference_map: sunpy.map.Map) -> list[SkyCoord]:
    if params._has_two_point_axis():
        pts = _sample_footprint_cart_rsun(params)
        if pts.shape[1] == 0:
            return []
        rep = CartesianRepresentation(pts[0] * u.R_sun, pts[1] * u.R_sun, pts[2] * u.R_sun)
        coords_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)
        coords_hpc = coords_hgs.transform_to(reference_map.coordinate_frame)
        mask = _visible_mask(coords_hgs, reference_map, only_visible=params.only_visible)
        return _split_skycoord_by_mask(coords_hpc, mask)

    r_hat, e_lon, e_lat = _hgs_unit_vectors(float(params.center_lon_deg), float(params.center_lat_deg))
    center_r = float(params.center_r_rsun)
    center = center_r * r_hat
    a = float(params.a_rsun)
    b = float(params.b_rsun)
    cr = center_r

    A = (a * a - b * b)
    B = 2.0 * cr * a
    C = (cr * cr + b * b - 1.0)

    candidates: list[float] = []
    if abs(A) < 1e-12:
        if abs(B) < 1e-12:
            return []
        candidates = [(-C / B)]
    else:
        disc = B * B - 4.0 * A * C
        if disc < 0:
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

    bet = np.linspace(0.0, 2.0 * np.pi, params.n_line_pts, endpoint=True)
    sin_b = np.sin(bet)
    cos_b = np.cos(bet)
    dir_latlon = (cos_b[None, :] * e_lon[:, None] + sin_b[None, :] * e_lat[:, None])

    cart = center[:, None] + a * cos_a0 * r_hat[:, None] + b * sin_a0 * dir_latlon
    rep = CartesianRepresentation(cart[0] * u.R_sun, cart[1] * u.R_sun, cart[2] * u.R_sun)
    coords_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)
    mask = _visible_mask(coords_hgs, reference_map, only_visible=params.only_visible)
    coords_hpc = coords_hgs.transform_to(reference_map.coordinate_frame)
    return _split_skycoord_by_mask(coords_hpc, mask)


def sample_spheroid_dome_wireframe_hpc(params: SpheroidDome3DParams, reference_map: sunpy.map.Map) -> list[SkyCoord]:
    geom = _spheroid_axis_geometry_rsun(params)
    center = geom["center"][:, None]
    axis_u = geom["axis_u"][:, None]
    e1 = geom["e1"][:, None]
    e2 = geom["e2"][:, None]

    lines_hpc: list[SkyCoord] = []

    def _above_surface_mask_cart(cart_rsun: np.ndarray) -> np.ndarray:
        if not params.only_above_surface:
            return np.ones(cart_rsun.shape[1], dtype=bool)
        rr = np.sqrt(np.sum(cart_rsun ** 2, axis=0))
        return rr >= 1.0

    alphas = np.linspace(0.0, np.pi, params.n_line_pts, endpoint=True)
    ca = np.cos(alphas)[None, :]
    sa = np.sin(alphas)[None, :]

    betas = np.linspace(0.0, 2.0 * np.pi, params.n_meridians, endpoint=False)
    for beta in betas:
        dir_perp = (np.cos(beta) * e1 + np.sin(beta) * e2)
        cart = center + float(params.a_rsun) * ca * axis_u + float(params.b_rsun) * sa * dir_perp

        rep = CartesianRepresentation(cart[0, :] * u.R_sun, cart[1, :] * u.R_sun, cart[2, :] * u.R_sun)
        coords_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)
        mask = _above_surface_mask_cart(cart) & _visible_mask(coords_hgs, reference_map, only_visible=params.only_visible)
        coords_hpc = coords_hgs.transform_to(reference_map.coordinate_frame)
        lines_hpc.extend(_split_skycoord_by_mask(coords_hpc, mask))

    alpha_list = np.linspace(0.0, np.pi, params.n_parallels + 2)[1:-1]
    betas_line = np.linspace(0.0, 2.0 * np.pi, params.n_line_pts, endpoint=True)
    cb = np.cos(betas_line)[None, :]
    sb = np.sin(betas_line)[None, :]

    for alpha0 in alpha_list:
        ca0 = float(np.cos(alpha0))
        sa0 = float(np.sin(alpha0))
        dir_perp = cb * e1 + sb * e2
        cart = center + float(params.a_rsun) * ca0 * axis_u + float(params.b_rsun) * sa0 * dir_perp

        rep = CartesianRepresentation(cart[0, :] * u.R_sun, cart[1, :] * u.R_sun, cart[2, :] * u.R_sun)
        coords_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)
        mask = _above_surface_mask_cart(cart) & _visible_mask(coords_hgs, reference_map, only_visible=params.only_visible)
        coords_hpc = coords_hgs.transform_to(reference_map.coordinate_frame)
        lines_hpc.extend(_split_skycoord_by_mask(coords_hpc, mask))

    return lines_hpc


# ==========================================================
# Plot helpers (HPC -> relative pixels in composite axes)
# ==========================================================

def _hpc_to_rel_pix(coords_hpc: SkyCoord, rsun_arcsec: float, px_per_rsun: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert HPC Tx/Ty [arcsec] into composite plot coordinates [pixels]."""
    x_arcsec = np.asarray(coords_hpc.Tx.to_value(u.arcsec), dtype=float)
    y_arcsec = np.asarray(coords_hpc.Ty.to_value(u.arcsec), dtype=float)
    x_px = (x_arcsec / float(rsun_arcsec)) * float(px_per_rsun)
    y_px = (y_arcsec / float(rsun_arcsec)) * float(px_per_rsun)
    return x_px, y_px


def overlay_spheroid_on_coronagraph_axes(
    ax: "plt.Axes",
    reference_map: sunpy.map.Map,
    params_lasco: dict,
    spheroid_params: SpheroidDome3DParams,
    *,
    color: str = "#00FF00",
    lw_wire: float = 1.0,
    lw_footprint: float = 2.2,
    alpha_wire: float = 0.85,
    alpha_footprint: float = 0.95,
    zorder_wire: int = 6,
    zorder_markers: int = 8,
    verbose: bool = True,
) -> SpheroidDome3DParams:
    """Overlay spheroid dome (wireframe + footprint + markers) on the composite axes."""

    rsun_arcsec = float(reference_map.rsun_obs.to_value(u.arcsec))
    px_per_rsun = float(params_lasco["px_per_rsun"])

    if verbose:
        psi_deg = spheroid_footprint_angular_radius_deg(spheroid_params)
        if psi_deg is None:
            print("[INFO] footprint ψ = N/A (no photospheric intersection)")
        else:
            print(f"[INFO] footprint ψ = {psi_deg:.2f} deg")

    wire_lines_hpc = sample_spheroid_dome_wireframe_hpc(spheroid_params, reference_map)
    if (len(wire_lines_hpc) == 0) and spheroid_params.only_visible:
        if verbose:
            print("[WARN] spheroid wireframe is empty; retry with only_visible=False")
        spheroid_params = replace(spheroid_params, only_visible=False)
        wire_lines_hpc = sample_spheroid_dome_wireframe_hpc(spheroid_params, reference_map)

    for ln in wire_lines_hpc:
        x_px, y_px = _hpc_to_rel_pix(ln, rsun_arcsec, px_per_rsun)
        ax.plot(x_px, y_px, color=color, linewidth=lw_wire, alpha=alpha_wire, zorder=zorder_wire)

    footprint_lines_hpc = sample_spheroid_footprint_hpc(spheroid_params, reference_map)
    for fp in footprint_lines_hpc:
        x_px, y_px = _hpc_to_rel_pix(fp, rsun_arcsec, px_per_rsun)
        ax.plot(x_px, y_px, color=color, linewidth=lw_footprint, alpha=alpha_footprint, zorder=zorder_wire + 1)

    # axis-footpoint marker (Anchor in HGS / Stonyhurst)
    try:
        anchor_hpc = spheroid_axis_footpoint_hpc(spheroid_params, reference_map)
        x0, y0 = _hpc_to_rel_pix(anchor_hpc, rsun_arcsec, px_per_rsun)
        ax.plot(
            [float(np.atleast_1d(x0)[0])],
            [float(np.atleast_1d(y0)[0])],
            marker="*",
            linestyle="None",
            markerfacecolor="yellow",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=20.0,
            zorder=zorder_markers,
            label=(
                f"axis surface intersection (HGS lon,lat)=({float(spheroid_params.anchor_lon_deg):.1f},"
                f"{float(spheroid_params.anchor_lat_deg):.1f})°"
            ),
        )
    except Exception as exc:
        if verbose:
            print(f"[WARN] axis-footpoint marker skipped: {exc}")

    # apex marker
    try:
        apex_hpc = spheroid_dome_apex_hpc(spheroid_params, reference_map)
        x1, y1 = _hpc_to_rel_pix(apex_hpc, rsun_arcsec, px_per_rsun)
        apex_label = (
            f"3D spheroid apex ($\\kappa$={spheroid_params.kappa:.3f}, "
            f"$\\epsilon$={spheroid_params.epsilon:.3f}, "
            f"r={spheroid_params.apex_r_rsun:.3f} $R_\\odot$)"
        )
        ax.plot(
            [float(np.atleast_1d(x1)[0])],
            [float(np.atleast_1d(y1)[0])],
            marker="o",
            linestyle="None",
            markerfacecolor="orange",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=10.0,
            zorder=zorder_markers,
            label=apex_label,
        )
    except Exception as exc:
        if verbose:
            print(f"[WARN] apex marker skipped: {exc}")

    # dummy handle for legend
    ax.plot([], [], color=color, lw=3, alpha=0.7, label=spheroid_params.legend_label())
    ax.plot([], [], color=color, lw=3, alpha=0.7)

    return spheroid_params
# ==========================================================
# Main
# ==========================================================

def main(
    target_time_str: str,
    spheroid_params: SpheroidDome3DParams,
    *,
    out_png: str | Path | None = None,
    spheroid_color: str = "#00FF00",
    delta_time_min: int = 10,
    mk4_vmin: float = -4.0,
    mk4_vmax: float = 4.0,
    lasco_vmin: float = -10.0,
    lasco_vmax: float = 10.0,
    aia_vmin: float | None = None,
    aia_vmax: float | None = None,
):
    backend_initial = plt.get_backend().lower()
    if "agg" in backend_initial:
        try:
            plt.switch_backend("TkAgg")
            print("[INFO] Switched matplotlib backend to TkAgg.")
        except Exception as exc:
            print(f"[INFO] TkAgg backend unavailable ({exc}); continue with {backend_initial}.")

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    print(f"[INFO] Building K-COR+LASCO composite for {target_time_str}")
    # res = create_single_diff_from_csv_image(
    #     ax,
    #     target_time_str,
    #     delta_time_min,
    #     mk4_inner=1.4,
    #     mk4_outer_lasco_inner=3.0,
    #     lasco_outer=6.0,
    #     xlim_min=-250,
    #     xlim_max=0,
    #     ylim_min=-100,
    #     ylim_max=200,
    #     mk4_vmin=mk4_vmin,
    #     mk4_vmax=mk4_vmax,
    #     lasco_vmin=lasco_vmin,
    #     lasco_vmax=lasco_vmax,
    #     aia_vmin=aia_vmin,
    #     aia_vmax=aia_vmax,
    # )
    
    res = create_single_diff_from_time_image(
        ax,
        target_time_str,
        delta_time_min,
        mk4_inner=1.4,
        mk4_outer_lasco_inner=3.0,
        lasco_outer=6.0,
        xlim_min=-250,
        xlim_max=0,
        ylim_min=-100,
        ylim_max=200
    )

    params_lasco = res["params_lasco"]
    lasco_map = res["lasco_map"]

    overlay_spheroid_on_coronagraph_axes(
        ax,
        lasco_map,
        params_lasco,
        spheroid_params,
        color=spheroid_color,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("X [pixels]")
    ax.set_ylabel("Y [pixels]")
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_png}")

    backend = plt.get_backend().lower()
    if backend == "agg":
        print(f"[INFO] Non-interactive backend ({backend}); skip plt.show().")
        plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    # Example: parameterization consistent with aia_spheroid_plot.py
    target_time = "2022-06-13T03:36:18"

    # Axis surface intersection (anchor)
    anchor_lon_deg = -30.0
    anchor_lat_deg = +19.0

    # Apex direction (can differ from anchor direction)
    apex_lon_deg = -45.0
    apex_lat_deg = +17.0

    # Apex radius and spheroid shape
    apex_rsun = 5.13
    kappa = 0.40
    epsilon = 0.62

    spheroid = SpheroidDome3DParams(
        kappa=float(kappa),
        epsilon=float(epsilon),
        anchor_lon_deg=float(anchor_lon_deg),
        anchor_lat_deg=float(anchor_lat_deg),
        apex_lon_deg=float(apex_lon_deg),
        apex_lat_deg=float(apex_lat_deg),
        apex_r_rsun=float(apex_rsun),
        n_meridians=12,
        n_parallels=7,
        n_line_pts=240,
        only_above_surface=True,
        only_visible=True,
    )

    output_path = f"/mnt/d/wsl/home/kinno-7010/Research/GCS/output/C2_spheroid_{target_time.replace(':','')}.png"
    main(
        target_time_str=target_time,
        spheroid_params=spheroid,
        out_png=output_path,
        mk4_vmin=-4.0,
        mk4_vmax=4.0,
        lasco_vmin=-10.0,
        lasco_vmax=10.0,
        aia_vmin=None,
        aia_vmax=None,
    )