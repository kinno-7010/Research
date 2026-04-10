from __future__ import annotations

"""
plot_spheroid_multiview.py

Earth-view K-COR + LASCO-C2 composite on ax[0] and STEREO-A/SECCHI/EUVI+COR1
integrated image on ax[1], with the SAME 3D spheroid projected into both views.

Design policy
-------------
- Reuse the existing coronagraph plotting settings by calling:
    * integrated_analysis.create_single_diff_from_time_image()
    * STEREO_integrated_plot.create_integrated_stereo_image()
- Do NOT hand-rotate or hand-shift the spheroid between viewpoints.
  Instead, define the spheroid once in 3D HGS (Stonyhurst) and project it into
  each observer frame with SunPy coordinate transforms.
- Earth-view composite uses the existing relative-pixel convention from
  plot_spheroid_C2.py.
- STEREO-A view is a WCSAxes, so the projected spheroid is converted with
  world_to_pixel() on the same common reference map used by the integrated plot.
"""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy.coordinates import CartesianRepresentation, SkyCoord
import sunpy.map
from sunpy.coordinates import frames as sunpy_frames

# -----------------------------------------------------------------------------
# Existing plotting code (kept as the plotting backend)
# -----------------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

sys.path.append("/home/kinno-7010/Research_code/SDO_Mk4_SOHO/py_folder")
sys.path.append("/home/kinno-7010/Research_code/STEREO-A/SECCHI")

try:
    from integrated_analysis import create_single_diff_from_time_image
except Exception as exc:
    raise ImportError(
        "Failed to import create_single_diff_from_time_image from integrated_analysis.py"
    ) from exc

try:
    from STEREO_integrated_plot import (
        build_common_reference_map,
        create_integrated_stereo_image,
    )
except Exception as exc:
    raise ImportError(
        "Failed to import create_integrated_stereo_image/build_common_reference_map "
        "from STEREO_integrated_plot.py"
    ) from exc


# =============================================================================
# Spheroid model (copied/adapted from plot_spheroid_C2.py)
# =============================================================================

@dataclass
class SpheroidDome3DParams:
    """3D spheroid dome in HGS parameterized by (kappa, epsilon)."""

    kappa: float
    epsilon: float

    # Two-point axis specification
    anchor_lon_deg: float | None = None
    anchor_lat_deg: float | None = None
    apex_lon_deg: float | None = None
    apex_lat_deg: float | None = None
    apex_r_rsun: float | None = None
    apex_height_rsun: float | None = None

    # Legacy center-based specification
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
            self.anchor_lon_deg is not None
            and self.anchor_lat_deg is not None
            and self.apex_lon_deg is not None
            and self.apex_lat_deg is not None
            and (self.apex_r_rsun is not None or self.apex_height_rsun is not None)
        )

    def _has_center_axis(self) -> bool:
        return (
            self.center_lon_deg is not None
            and self.center_lat_deg is not None
            and (
                self.apex_r_rsun is not None
                or self.apex_height_rsun is not None
                or self.center_r_rsun is not None
            )
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
            raise ValueError("apex_r_rsun or apex_height_rsun is required.")

        self.kappa = float(self.kappa)
        self.epsilon = float(self.epsilon)
        if self.kappa < 0.0:
            raise ValueError("kappa must be non-negative.")
        if not (-1.0 < self.epsilon < 1.0):
            raise ValueError("epsilon must satisfy -1 < epsilon < 1.")

        h = float(self.apex_height_rsun)
        b0 = float(self.kappa * h)
        eps_abs = abs(float(self.epsilon))
        denom = np.sqrt(max(1.0e-12, 1.0 - eps_abs**2))

        self._b_rsun = float(b0)
        if self.epsilon < 0.0:
            self._a_rsun = float(b0 / denom)
        else:
            self._a_rsun = float(b0 * denom)

        if self._has_two_point_axis():
            anchor = _cart_rsun_from_lonlat(
                float(self.anchor_lon_deg),
                float(self.anchor_lat_deg),
                1.0,
            )
            apex = _cart_rsun_from_lonlat(
                float(self.apex_lon_deg),
                float(self.apex_lat_deg),
                float(self.apex_r_rsun),
            )
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
                "Provide either center_* + apex_r/height or anchor_* + apex_* + apex_r/height."
            )

        if self.anchor_lon_deg is None:
            self.anchor_lon_deg = float(self.center_lon_deg)
        if self.anchor_lat_deg is None:
            self.anchor_lat_deg = float(self.center_lat_deg)
        if self.apex_lon_deg is None:
            self.apex_lon_deg = float(self.center_lon_deg)
        if self.apex_lat_deg is None:
            self.apex_lat_deg = float(self.center_lat_deg)
        if self.center_r_rsun is None and self.apex_r_rsun is not None:
            self.center_r_rsun = float(self.apex_r_rsun - self.a_rsun)

    def legend_label(self) -> str:
        return (
            f"Spheroid: κ={self.kappa:.3f}, ε={self.epsilon:.3f}, "
            f"apex(lon,lat)=({float(self.apex_lon_deg):.1f},{float(self.apex_lat_deg):.1f})°, "
            f"r_apex={float(self.apex_r_rsun):.2f} R☉"
        )

def _extract_earth_display_calibration(ax: plt.Axes) -> dict[str, float] | None:
    """Infer the Earth-panel display scale from the already drawn reference circles.

    The guide-circle labels in integrated_analysis.py are written like
    ``1.4 $R_\odot$`` and ``3.0 $R_\odot$``. Therefore the parser must accept
    the optional LaTeX dollar sign between the numeric value and the letter R.
    """
    import re

    candidates: list[tuple[float, float, float, str]] = []

    for ln in ax.lines:
        label = str(ln.get_label())
        if 'R' not in label:
            continue

        m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*\$?\s*R', label)
        if m is None:
            continue

        radius_rsun = float(m.group(1))
        if not np.isfinite(radius_rsun) or radius_rsun <= 0.0:
            continue

        x = np.asarray(ln.get_xdata(), dtype=float)
        y = np.asarray(ln.get_ydata(), dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]
        if x.size < 20:
            continue

        rr = np.sqrt(x * x + y * y)
        r_disp = float(np.nanmedian(rr))
        scatter = float(np.nanmedian(np.abs(rr - r_disp)))
        if not np.isfinite(r_disp) or r_disp <= 0.0:
            continue

        px_per_rsun_disp = r_disp / radius_rsun
        candidates.append((scatter, radius_rsun, px_per_rsun_disp, label))

    if not candidates:
        return None

    candidates.sort(key=lambda t: (t[0], t[1]))
    scatter, radius_rsun, px_per_rsun_disp, label = candidates[0]
    return {
        'cx_disp': 0.0,
        'cy_disp': 0.0,
        'px_per_rsun_disp': float(px_per_rsun_disp),
        'radius_rsun_label': float(radius_rsun),
        'label': label,
        'scatter': float(scatter),
    }

    
def _hgs_unit_vectors(lon_deg: float, lat_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _fit_circle_kasa(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Simple algebraic circle fit.

    Returns
    -------
    cx, cy, r : float
        Circle center and radius in display-axis units.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        raise ValueError('Need at least 3 finite points for circle fit.')

    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = coef
    cx = -0.5 * D
    cy = -0.5 * E
    r2 = cx * cx + cy * cy - F
    if not np.isfinite(r2) or r2 <= 0.0:
        raise ValueError('Circle fit returned non-positive radius squared.')
    r = float(np.sqrt(r2))
    return float(cx), float(cy), r


def _split_skycoord_by_mask(coords: SkyCoord, mask: np.ndarray) -> list[SkyCoord]:
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



def _cart_rsun_to_hpc_arcsec(
    cart_rsun: np.ndarray,
    reference_map: sunpy.map.GenericMap,
) -> tuple[np.ndarray, np.ndarray]:
    """Project HGS Cartesian coordinates to helioprojective arcsec.

    This avoids relying on map-to-map WCS transforms for the spheroid points.
    The projection uses the observer longitude/latitude and distance taken from
    the actual data map for each viewpoint.
    """
    arr = np.asarray(cart_rsun, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(3, 1)

    xg, yg, zg = arr[0], arr[1], arr[2]
    r = np.sqrt(xg * xg + yg * yg + zg * zg)
    lon = np.arctan2(yg, xg)
    lat = np.arcsin(np.clip(zg / np.where(r == 0.0, 1.0, r), -1.0, 1.0))

    obs_lon_deg, obs_lat_deg, d_rsun = _observer_hgs_from_map(reference_map)
    lon0 = np.deg2rad(obs_lon_deg)
    lat0 = np.deg2rad(obs_lat_deg)

    dlon = np.arctan2(np.sin(lon - lon0), np.cos(lon - lon0))

    x = r * np.cos(lat) * np.sin(dlon)
    y = r * (np.sin(lat) * np.cos(lat0) - np.cos(lat) * np.cos(dlon) * np.sin(lat0))
    z = d_rsun - r * (
        np.sin(lat) * np.sin(lat0) + np.cos(lat) * np.cos(dlon) * np.cos(lat0)
    )

    tx_arcsec = np.rad2deg(np.arctan2(x, z)) * 3600.0
    ty_arcsec = np.rad2deg(np.arctan2(y, np.sqrt(x * x + z * z))) * 3600.0
    return np.asarray(tx_arcsec, dtype=float), np.asarray(ty_arcsec, dtype=float)



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
        raise ValueError("zero-radius cartesian point cannot be converted to lon/lat.")
    lon_deg = float(np.rad2deg(np.arctan2(y, x)))
    lat_deg = float(np.rad2deg(np.arcsin(np.clip(z / r, -1.0, 1.0))))
    return lon_deg, lat_deg, r


def _map_rsun_pixel(reference_map: sunpy.map.GenericMap) -> float:
    """Solar radius in displayed pixel units for the map/WCS being plotted."""
    try:
        scale_x = abs(float(reference_map.scale.axis1.to_value(u.arcsec / u.pix)))
    except Exception:
        scale_x = abs(float(reference_map.meta["CDELT1"]))
    rsun_arcsec = _map_rsun_arcsec(reference_map)
    return float(rsun_arcsec / scale_x)

def _map_center_pixel(reference_map: sunpy.map.GenericMap) -> tuple[float, float]:
    """Solar-center pixel coordinates in the plotted map.

    For the common STEREO reference map used by create_integrated_stereo_image(),
    CRVAL=(0,0) and the map is explicitly rebuilt north-up, so CRPIX gives the
    displayed center directly.  We still try world_to_pixel first for robustness.
    """
    try:
        sun_center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=reference_map.coordinate_frame)
        sun_center_pix = reference_map.world_to_pixel(sun_center)
        return float(sun_center_pix.x.value), float(sun_center_pix.y.value)
    except Exception:
        meta = getattr(reference_map, "meta", {})
        return float(meta["CRPIX1"]) - 1.0, float(meta["CRPIX2"]) - 1.0


def _map_rsun_arcsec(reference_map: sunpy.map.GenericMap) -> float:
    """Solar angular radius [arcsec] from the map metadata."""
    try:
        return float(reference_map.rsun_obs.to_value(u.arcsec))
    except Exception:
        meta = getattr(reference_map, "meta", {})
        if "RSUN_OBS" in meta:
            return float(meta["RSUN_OBS"])
        if "RSUN" in meta:
            return float(meta["RSUN"])
        raise

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


def _orthonormal_basis_from_axis(
    axis_u: np.ndarray,
    ref_vec: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    axis_u = _unit_vec(axis_u)
    if ref_vec is None:
        ref_vec = np.array([0.0, 0.0, 1.0], dtype=float)
    ref = np.asarray(ref_vec, dtype=float)
    ref = ref - np.dot(ref, axis_u) * axis_u
    if np.linalg.norm(ref) < 1e-10:
        alt = (
            np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(axis_u[0]) < 0.9
            else np.array([0.0, 1.0, 0.0], dtype=float)
        )
        ref = alt - np.dot(alt, axis_u) * axis_u
    e1 = _unit_vec(ref)
    e2 = _unit_vec(np.cross(axis_u, e1))
    return e1, e2

def _observer_hgs_from_map(reference_map: sunpy.map.GenericMap) -> tuple[float, float, float]:
    """Observer Stonyhurst lon [deg], lat [deg], distance [Rsun]."""
    try:
        obs_hgs = reference_map.observer_coordinate.transform_to(
            sunpy_frames.HeliographicStonyhurst(obstime=reference_map.date)
        )
        lon_deg = float(obs_hgs.lon.to_value(u.deg))
        lat_deg = float(obs_hgs.lat.to_value(u.deg))
        d_rsun = float(obs_hgs.radius.to_value(u.R_sun))
        return lon_deg, lat_deg, d_rsun
    except Exception:
        meta = getattr(reference_map, "meta", {})
        lon_deg = float(meta.get("HGLN_OBS", 0.0))
        lat_deg = float(meta.get("HGLT_OBS", meta.get("CRLT_OBS", 0.0)))
        dsun_obs = float(meta["DSUN_OBS"])
        rsun_ref = float(meta.get("RSUN_REF", 6.957e8))
        d_rsun = float(dsun_obs / rsun_ref)
        return lon_deg, lat_deg, d_rsun

def _spheroid_axis_geometry_rsun(params: SpheroidDome3DParams) -> dict[str, np.ndarray]:
    if params._has_two_point_axis():
        anchor = _cart_rsun_from_lonlat(
            float(params.anchor_lon_deg), float(params.anchor_lat_deg), 1.0
        )
        apex = _cart_rsun_from_lonlat(
            float(params.apex_lon_deg),
            float(params.apex_lat_deg),
            float(params.apex_r_rsun),
        )
        axis_u = _unit_vec(apex - anchor)
        center = apex - float(params.a_rsun) * axis_u
        e1, e2 = _orthonormal_basis_from_axis(axis_u, ref_vec=anchor)
        return {
            "center": center,
            "axis_u": axis_u,
            "e1": e1,
            "e2": e2,
            "anchor": anchor,
            "apex": apex,
        }

    r_hat, e_lon, e_lat = _hgs_unit_vectors(
        float(params.center_lon_deg),
        float(params.center_lat_deg),
    )
    center = float(params.center_r_rsun) * r_hat
    anchor = 1.0 * r_hat
    apex = center + float(params.a_rsun) * r_hat
    return {
        "center": center,
        "axis_u": r_hat,
        "e1": e_lon,
        "e2": e_lat,
        "anchor": anchor,
        "apex": apex,
    }

def _split_cart_by_mask(cart: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Split 3xN Cartesian arrays into contiguous visible segments."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []

    cuts = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[0], cuts + 1])
    ends = np.concatenate([cuts + 1, [idx.size]])

    segs: list[np.ndarray] = []
    for s, e in zip(starts, ends):
        seg_idx = idx[s:e]
        if seg_idx.size >= 2:
            segs.append(cart[:, seg_idx])
    return segs



def _sample_footprint_cart_rsun(
    params: SpheroidDome3DParams,
    n_beta: int | None = None,
) -> np.ndarray:
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
                a0 = float(
                    alpha_grid[i]
                    + (0.0 - f[i])
                    * (alpha_grid[i + 1] - alpha_grid[i])
                    / (f[i + 1] - f[i])
                )
                roots.append(a0)

        if not roots:
            continue

        alpha0 = max(roots, key=lambda a: np.sin(a))
        p = (
            center
            + float(params.a_rsun) * np.cos(alpha0) * axis_u
            + float(params.b_rsun) * np.sin(alpha0) * dir_perp
        )
        if abs(np.linalg.norm(p) - 1.0) < 5e-3:
            pts.append(p)

    if not pts:
        return np.zeros((3, 0), dtype=float)
    return np.stack(pts, axis=1)


# =============================================================================
# Coordinate sampling and projection
# =============================================================================

def _visible_mask(
    coords_hgs: SkyCoord,
    reference_map: sunpy.map.GenericMap | None = None,
    *,
    only_visible: bool = True,
) -> np.ndarray:
    """
    Approximate front-side visibility using the observer direction.

    This uses the physically relevant criterion pt_hat · obs_hat > 0.
    No ad-hoc sign flip is applied, so the criterion remains consistent for
    off-Earth observers such as STEREO-A.
    """
    if not only_visible or reference_map is None:
        return np.ones(coords_hgs.shape, dtype=bool)

    try:
        obs_vec = reference_map.observer_coordinate.cartesian.xyz.to_value(u.R_sun)
        obs_hat = _unit_vec(np.asarray(obs_vec, dtype=float).reshape(3))

        pt_vec = coords_hgs.cartesian.xyz.to_value(u.R_sun)
        pt_hat = _unit_vec(np.asarray(pt_vec, dtype=float))

        dot_sum = np.sum(obs_hat[:, None] * pt_hat, axis=0)
        return dot_sum > 0.0
    except Exception as exc:
        print(f"[WARN] visibility mask disabled ({exc})")
        return np.ones(coords_hgs.shape, dtype=bool)



def _visible_mask_cart(
    cart_rsun: np.ndarray,
    reference_map: sunpy.map.GenericMap | None = None,
    *,
    only_visible: bool = True,
) -> np.ndarray:
    """Front-side visibility mask in HGS Cartesian space."""
    arr = np.asarray(cart_rsun, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(3, 1)

    if not only_visible or reference_map is None:
        return np.ones(arr.shape[1], dtype=bool)

    try:
        obs_lon_deg, obs_lat_deg, _ = _observer_hgs_from_map(reference_map)
        obs_hat = _cart_rsun_from_lonlat(obs_lon_deg, obs_lat_deg, 1.0)
        obs_hat = _unit_vec(obs_hat)
        pt_hat = _unit_vec(arr)
        dot_sum = np.sum(obs_hat[:, None] * pt_hat, axis=0)
        return dot_sum > 0.0
    except Exception as exc:
        print(f"[WARN] visibility mask disabled ({exc})")
        return np.ones(arr.shape[1], dtype=bool)

def spheroid_dome_apex_arcsec(
    params: SpheroidDome3DParams,
    reference_map: sunpy.map.GenericMap,
) -> tuple[float, float]:
    geom = _spheroid_axis_geometry_rsun(params)
    tx, ty = _cart_rsun_to_hpc_arcsec(geom["apex"], reference_map)
    return float(np.atleast_1d(tx)[0]), float(np.atleast_1d(ty)[0])

def spheroid_dome_apex_hpc(
    params: SpheroidDome3DParams,
    reference_map: sunpy.map.GenericMap,
) -> SkyCoord:
    geom = _spheroid_axis_geometry_rsun(params)
    cart_rsun = geom["apex"]
    rep = CartesianRepresentation(
        cart_rsun[0] * u.R_sun,
        cart_rsun[1] * u.R_sun,
        cart_rsun[2] * u.R_sun,
    )
    apex_hgs = SkyCoord(
        rep,
        frame=sunpy_frames.HeliographicStonyhurst,
        obstime=reference_map.date,
    )
    return apex_hgs.transform_to(reference_map.coordinate_frame)

def spheroid_axis_footpoint_arcsec(
    params: SpheroidDome3DParams,
    reference_map: sunpy.map.GenericMap,
) -> tuple[float, float]:
    geom = _spheroid_axis_geometry_rsun(params)
    tx, ty = _cart_rsun_to_hpc_arcsec(geom["anchor"], reference_map)
    return float(np.atleast_1d(tx)[0]), float(np.atleast_1d(ty)[0])

def spheroid_axis_footpoint_hpc(
    params: SpheroidDome3DParams,
    reference_map: sunpy.map.GenericMap,
) -> SkyCoord:
    geom = _spheroid_axis_geometry_rsun(params)
    cart_rsun = geom["anchor"]
    rep = CartesianRepresentation(
        cart_rsun[0] * u.R_sun,
        cart_rsun[1] * u.R_sun,
        cart_rsun[2] * u.R_sun,
    )
    fp_hgs = SkyCoord(
        rep,
        frame=sunpy_frames.HeliographicStonyhurst,
        obstime=reference_map.date,
    )
    return fp_hgs.transform_to(reference_map.coordinate_frame)


def sample_spheroid_footprint_cart(
    params: SpheroidDome3DParams,
    reference_map: sunpy.map.GenericMap,
) -> list[np.ndarray]:
    if params._has_two_point_axis():
        pts = _sample_footprint_cart_rsun(params)
        if pts.shape[1] == 0:
            return []
        mask = _visible_mask_cart(pts, reference_map, only_visible=params.only_visible)
        return _split_cart_by_mask(pts, mask)

    r_hat, e_lon, e_lat = _hgs_unit_vectors(
        float(params.center_lon_deg),
        float(params.center_lat_deg),
    )
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
    mask = _visible_mask_cart(cart, reference_map, only_visible=params.only_visible)
    return _split_cart_by_mask(cart, mask)




def sample_spheroid_footprint_hpc(
    params: SpheroidDome3DParams,
    reference_map: sunpy.map.GenericMap,
) -> list[SkyCoord]:
    if params._has_two_point_axis():
        pts = _sample_footprint_cart_rsun(params)
        if pts.shape[1] == 0:
            return []
        rep = CartesianRepresentation(
            pts[0] * u.R_sun,
            pts[1] * u.R_sun,
            pts[2] * u.R_sun,
        )
        coords_hgs = SkyCoord(
            rep,
            frame=sunpy_frames.HeliographicStonyhurst,
            obstime=reference_map.date,
        )
        coords_hpc = coords_hgs.transform_to(reference_map.coordinate_frame)
        mask = _visible_mask(coords_hgs, reference_map, only_visible=params.only_visible)
        return _split_skycoord_by_mask(coords_hpc, mask)

    r_hat, e_lon, e_lat = _hgs_unit_vectors(
        float(params.center_lon_deg),
        float(params.center_lat_deg),
    )
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
    coords_hgs = SkyCoord(
        rep,
        frame=sunpy_frames.HeliographicStonyhurst,
        obstime=reference_map.date,
    )
    mask = _visible_mask(coords_hgs, reference_map, only_visible=params.only_visible)
    coords_hpc = coords_hgs.transform_to(reference_map.coordinate_frame)
    return _split_skycoord_by_mask(coords_hpc, mask)

def sample_spheroid_dome_wireframe_cart(
    params: SpheroidDome3DParams,
    reference_map: sunpy.map.GenericMap,
) -> list[np.ndarray]:
    geom = _spheroid_axis_geometry_rsun(params)
    center = geom["center"][:, None]
    axis_u = geom["axis_u"][:, None]
    e1 = geom["e1"][:, None]
    e2 = geom["e2"][:, None]

    lines_cart: list[np.ndarray] = []

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
        dir_perp = np.cos(beta) * e1 + np.sin(beta) * e2
        cart = center + float(params.a_rsun) * ca * axis_u + float(params.b_rsun) * sa * dir_perp
        mask = _above_surface_mask_cart(cart) & _visible_mask_cart(
            cart,
            reference_map,
            only_visible=params.only_visible,
        )
        lines_cart.extend(_split_cart_by_mask(cart, mask))

    alpha_list = np.linspace(0.0, np.pi, params.n_parallels + 2)[1:-1]
    betas_line = np.linspace(0.0, 2.0 * np.pi, params.n_line_pts, endpoint=True)
    cb = np.cos(betas_line)[None, :]
    sb = np.sin(betas_line)[None, :]

    for alpha0 in alpha_list:
        ca0 = float(np.cos(alpha0))
        sa0 = float(np.sin(alpha0))
        dir_perp = cb * e1 + sb * e2
        cart = center + float(params.a_rsun) * ca0 * axis_u + float(params.b_rsun) * sa0 * dir_perp
        mask = _above_surface_mask_cart(cart) & _visible_mask_cart(
            cart,
            reference_map,
            only_visible=params.only_visible,
        )
        lines_cart.extend(_split_cart_by_mask(cart, mask))

    return lines_cart



def sample_spheroid_dome_wireframe_hpc(
    params: SpheroidDome3DParams,
    reference_map: sunpy.map.GenericMap,
) -> list[SkyCoord]:
    geom = _spheroid_axis_geometry_rsun(params)
    center = geom["center"][:, None]
    axis_u = geom["axis_u"][:, None]
    e1 = geom["e1"][:, None]
    e2 = geom["e2"][:, None]

    lines_hpc: list[SkyCoord] = []

    def _above_surface_mask_cart(cart_rsun: np.ndarray) -> np.ndarray:
        if not params.only_above_surface:
            return np.ones(cart_rsun.shape[1], dtype=bool)
        rr = np.sqrt(np.sum(cart_rsun**2, axis=0))
        return rr >= 1.0

    alphas = np.linspace(0.0, np.pi, params.n_line_pts, endpoint=True)
    ca = np.cos(alphas)[None, :]
    sa = np.sin(alphas)[None, :]

    betas = np.linspace(0.0, 2.0 * np.pi, params.n_meridians, endpoint=False)
    for beta in betas:
        dir_perp = np.cos(beta) * e1 + np.sin(beta) * e2
        cart = center + float(params.a_rsun) * ca * axis_u + float(params.b_rsun) * sa * dir_perp
        rep = CartesianRepresentation(cart[0, :] * u.R_sun, cart[1, :] * u.R_sun, cart[2, :] * u.R_sun)
        coords_hgs = SkyCoord(
            rep,
            frame=sunpy_frames.HeliographicStonyhurst,
            obstime=reference_map.date,
        )
        mask = _above_surface_mask_cart(cart) & _visible_mask(
            coords_hgs,
            reference_map,
            only_visible=params.only_visible,
        )
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
        coords_hgs = SkyCoord(
            rep,
            frame=sunpy_frames.HeliographicStonyhurst,
            obstime=reference_map.date,
        )
        mask = _above_surface_mask_cart(cart) & _visible_mask(
            coords_hgs,
            reference_map,
            only_visible=params.only_visible,
        )
        coords_hpc = coords_hgs.transform_to(reference_map.coordinate_frame)
        lines_hpc.extend(_split_skycoord_by_mask(coords_hpc, mask))

    return lines_hpc


# =============================================================================
# Plot helpers
# =============================================================================

def _hpc_to_rel_pix(
    coords_hpc: SkyCoord,
    rsun_arcsec: float,
    px_per_rsun: float,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert HPC arcsec to the Earth composite display coordinates.

    Parameters
    ----------
    coords_hpc : SkyCoord
        Helioprojective coordinates in the observer frame of ``reference_map``.
    rsun_arcsec : float
        Apparent solar radius of the reference map in arcsec.
    px_per_rsun : float
        Solar radius measured in *display-axis* pixels/units, not raw LASCO pixels.
    x0, y0 : float
        Display-space solar-center coordinates.
    """
    x_arcsec = np.asarray(coords_hpc.Tx.to_value(u.arcsec), dtype=float)
    y_arcsec = np.asarray(coords_hpc.Ty.to_value(u.arcsec), dtype=float)
    x_px = x0 + (x_arcsec / rsun_arcsec) * px_per_rsun
    y_px = y0 + (y_arcsec / rsun_arcsec) * px_per_rsun
    return x_px, y_px

def overlay_spheroid_on_coronagraph_axes(
    ax: plt.Axes,
    reference_map: sunpy.map.GenericMap,
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
    add_legend_handles: bool = True,
    verbose: bool = True,
) -> SpheroidDome3DParams:
    """Overlay on the Earth-view panel without changing the existing panel limits."""
    rsun_arcsec = float(reference_map.rsun_obs.to_value(u.arcsec))

    display_cal = _extract_earth_display_calibration(ax)
    if display_cal is not None:
        px_per_rsun = float(display_cal["px_per_rsun_disp"])
        x_center_disp = float(display_cal["cx_disp"])
        y_center_disp = float(display_cal["cy_disp"])
        if verbose:
            print(
                f"[INFO] Earth-view display calibration from '{display_cal['label']}': "
                f"px_per_rsun={px_per_rsun:.3f}, center=({x_center_disp:.3f},{y_center_disp:.3f}), "
                f"scatter={display_cal.get('scatter', np.nan):.3f}"
            )
    else:
        px_per_rsun = float(params_lasco["px_per_rsun"])
        x_center_disp = 0.0
        y_center_disp = 0.0
        if verbose:
            print(
                "[WARN] Earth-view display calibration could not be inferred from the drawn "
                "reference circles; falling back to params_lasco['px_per_rsun']."
            )

    xlim0 = ax.get_xlim()
    ylim0 = ax.get_ylim()

    wire_lines_hpc = sample_spheroid_dome_wireframe_hpc(spheroid_params, reference_map)
    if (len(wire_lines_hpc) == 0) and spheroid_params.only_visible:
        if verbose:
            print("[WARN] Earth-view wireframe is empty; retry with only_visible=False")
        spheroid_params = replace(spheroid_params, only_visible=False)
        wire_lines_hpc = sample_spheroid_dome_wireframe_hpc(spheroid_params, reference_map)

    for ln in wire_lines_hpc:
        x_px, y_px = _hpc_to_rel_pix(
            ln, rsun_arcsec, px_per_rsun, x0=x_center_disp, y0=y_center_disp
        )
        ax.plot(x_px, y_px, color=color, linewidth=lw_wire, alpha=alpha_wire, zorder=zorder_wire)

    footprint_lines_hpc = sample_spheroid_footprint_hpc(spheroid_params, reference_map)
    for fp in footprint_lines_hpc:
        x_px, y_px = _hpc_to_rel_pix(
            fp, rsun_arcsec, px_per_rsun, x0=x_center_disp, y0=y_center_disp
        )
        ax.plot(
            x_px,
            y_px,
            color=color,
            linewidth=lw_footprint,
            alpha=alpha_footprint,
            zorder=zorder_wire + 1,
        )

    try:
        anchor_hpc = spheroid_axis_footpoint_hpc(spheroid_params, reference_map)
        x0, y0 = _hpc_to_rel_pix(
            anchor_hpc, rsun_arcsec, px_per_rsun, x0=x_center_disp, y0=y_center_disp
        )
        ax.plot(
            [float(np.atleast_1d(x0)[0])],
            [float(np.atleast_1d(y0)[0])],
            marker="*",
            linestyle="None",
            markerfacecolor="yellow",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=18.0,
            zorder=zorder_markers,
            label="Anchor" if add_legend_handles else None,
        )
    except Exception as exc:
        if verbose:
            print(f"[WARN] Earth-view anchor marker skipped: {exc}")

    try:
        apex_hpc = spheroid_dome_apex_hpc(spheroid_params, reference_map)
        x1, y1 = _hpc_to_rel_pix(
            apex_hpc, rsun_arcsec, px_per_rsun, x0=x_center_disp, y0=y_center_disp
        )
        ax.plot(
            [float(np.atleast_1d(x1)[0])],
            [float(np.atleast_1d(y1)[0])],
            marker="o",
            linestyle="None",
            markerfacecolor="orange",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=9.0,
            zorder=zorder_markers,
            label="Apex" if add_legend_handles else None,
        )
    except Exception as exc:
        if verbose:
            print(f"[WARN] Earth-view apex marker skipped: {exc}")

    if add_legend_handles:
        ax.plot([], [], color=color, lw=3, alpha=0.8, label=spheroid_params.legend_label())

    ax.set_xlim(xlim0)
    ax.set_ylim(ylim0)

    if verbose:
        print(
            f"[INFO] Earth-view display scale used for overlay: "
            f"rsun_arcsec={rsun_arcsec:.3f}, px_per_rsun={px_per_rsun:.3f}"
        )
    return spheroid_params

def overlay_spheroid_on_wcs_axes(
    ax: plt.Axes,
    reference_map: sunpy.map.GenericMap,
    spheroid_params: SpheroidDome3DParams,
    *,
    color: str = "#00FF00",
    lw_wire: float = 1.0,
    lw_footprint: float = 2.2,
    alpha_wire: float = 0.85,
    alpha_footprint: float = 0.95,
    zorder_wire: int = 6,
    zorder_markers: int = 8,
    add_legend_handles: bool = False,
) -> SpheroidDome3DParams:
    """Overlay on the STEREO-A panel using the displayed common-map data scale."""
    rsun_arcsec = _map_rsun_arcsec(reference_map)
    px_per_rsun = _map_rsun_pixel(reference_map)
    x_center_pix, y_center_pix = _map_center_pixel(reference_map)

    xlim0 = ax.get_xlim()
    ylim0 = ax.get_ylim()

    wire_lines_cart = sample_spheroid_dome_wireframe_cart(spheroid_params, reference_map)
    if len(wire_lines_cart) == 0 and spheroid_params.only_visible:
        print("[WARN] STEREO-view wireframe is empty; retry with only_visible=False")
        spheroid_params = replace(spheroid_params, only_visible=False)
        wire_lines_cart = sample_spheroid_dome_wireframe_cart(spheroid_params, reference_map)

    for cart in wire_lines_cart:
        tx_arcsec, ty_arcsec = _cart_rsun_to_hpc_arcsec(cart, reference_map)
        x_pix = x_center_pix + (tx_arcsec / rsun_arcsec) * px_per_rsun
        y_pix = y_center_pix + (ty_arcsec / rsun_arcsec) * px_per_rsun
        ax.plot(x_pix, y_pix, color=color, linewidth=lw_wire, alpha=alpha_wire, zorder=zorder_wire)

    footprint_lines_cart = sample_spheroid_footprint_cart(spheroid_params, reference_map)
    for cart in footprint_lines_cart:
        tx_arcsec, ty_arcsec = _cart_rsun_to_hpc_arcsec(cart, reference_map)
        x_pix = x_center_pix + (tx_arcsec / rsun_arcsec) * px_per_rsun
        y_pix = y_center_pix + (ty_arcsec / rsun_arcsec) * px_per_rsun
        ax.plot(
            x_pix,
            y_pix,
            color=color,
            linewidth=lw_footprint,
            alpha=alpha_footprint,
            zorder=zorder_wire + 1,
        )

    try:
        tx0, ty0 = spheroid_axis_footpoint_arcsec(spheroid_params, reference_map)
        x0 = x_center_pix + (tx0 / rsun_arcsec) * px_per_rsun
        y0 = y_center_pix + (ty0 / rsun_arcsec) * px_per_rsun
        ax.plot(
            [float(x0)],
            [float(y0)],
            marker="*",
            linestyle="None",
            markerfacecolor="yellow",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=18.0,
            zorder=zorder_markers,
            label="Anchor" if add_legend_handles else None,
        )
    except Exception as exc:
        print(f"[WARN] STEREO-view anchor marker skipped: {exc}")

    try:
        tx1, ty1 = spheroid_dome_apex_arcsec(spheroid_params, reference_map)
        x1 = x_center_pix + (tx1 / rsun_arcsec) * px_per_rsun
        y1 = y_center_pix + (ty1 / rsun_arcsec) * px_per_rsun
        ax.plot(
            [float(x1)],
            [float(y1)],
            marker="o",
            linestyle="None",
            markerfacecolor="orange",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=9.0,
            zorder=zorder_markers,
            label="Apex" if add_legend_handles else None,
        )
    except Exception as exc:
        print(f"[WARN] STEREO-view apex marker skipped: {exc}")

    if add_legend_handles:
        ax.plot([], [], color=color, lw=3, alpha=0.8, label=spheroid_params.legend_label())

    ax.set_xlim(xlim0)
    ax.set_ylim(ylim0)

    print(
        f"[INFO] STEREO-view data scale: rsun_arcsec={rsun_arcsec:.3f}, "
        f"display_px_per_rsun={px_per_rsun:.3f}, center=({x_center_pix:.2f},{y_center_pix:.2f})"
    )
    return spheroid_params
# =============================================================================
# Main multiview plot
# =============================================================================

def plot_multiview_spheroid(
    target_time_str: str,
    spheroid_params: SpheroidDome3DParams,
    *,
    out_png: str | Path | None = None,
    spheroid_color: str = "#00FF00",
    delta_time_min: int = 10,
    euvi_outer_rsun: float = 1.30,
    cor1_outer_rsun: float = 4.0,
):
    backend_initial = plt.get_backend().lower()
    if "agg" in backend_initial:
        try:
            plt.switch_backend("TkAgg")
            print("[INFO] Switched matplotlib backend to TkAgg.")
        except Exception as exc:
            print(f"[INFO] TkAgg backend unavailable ({exc}); continue with {backend_initial}.")

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=300)
    ax_earth = axes[0]
    ax_stereo_placeholder = axes[1]

    # ------------------------------------------------------------------
    # Earth view: keep the existing plotting settings
    # ------------------------------------------------------------------
    print(f"[INFO] Building Earth-view composite for {target_time_str}")
    earth_res = create_single_diff_from_time_image(
        ax_earth,
        target_time_str,
        delta_time_min,
        mk4_inner=1.4,
        mk4_outer_lasco_inner=3.0,
        lasco_outer=6.0,
        xlim_min=-250,
        xlim_max=0,
        ylim_min=-100,
        ylim_max=200,
    )
    lasco_map = earth_res["lasco_map"]
    params_lasco = earth_res["params_lasco"]

    overlay_spheroid_on_coronagraph_axes(
        ax_earth,
        lasco_map,
        params_lasco,
        spheroid_params,
        color=spheroid_color,
        add_legend_handles=True,
    )
    ax_earth.set_title(f"Earth view: K-COR + LASCO-C2\n{target_time_str}")
    ax_earth.set_aspect("equal")
    ax_earth.set_xlabel("X [pixels]")
    ax_earth.set_ylabel("Y [pixels]")
    ax_earth.legend(loc="upper left", fontsize=8)

    # ------------------------------------------------------------------
    # STEREO-A view: keep the existing integrated plot settings
    # ------------------------------------------------------------------
    print(f"[INFO] Building STEREO-A integrated plot for {target_time_str}")
    ax_stereo, cor1_diff_map, euvi_diff_map, _, _, _ = create_integrated_stereo_image(
        ax_stereo_placeholder,
        target_time=target_time_str,
        cor1_base_minutes_before=delta_time_min,
        euvi_dt_minutes=delta_time_min,
        euvi_outer_rsun=euvi_outer_rsun,
        cor1_outer_rsun=cor1_outer_rsun,
    )

    # Rebuild the same common WCS used by create_integrated_stereo_image().
    stereo_common_map = build_common_reference_map(
        euvi_diff_map,
        outer_rsun=cor1_outer_rsun,
    )

    overlay_spheroid_on_wcs_axes(
        ax_stereo,
        stereo_common_map,
        spheroid_params,
        color=spheroid_color,
        add_legend_handles=False,
    )

    # fig.suptitle(
    #     "Same 3D Spheroid projected to Earth view and STEREO-A view",
    #     fontsize=15,
    #     y=0.98,
    # )
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

    return {
        "fig": fig,
        "ax_earth": ax_earth,
        "ax_stereo": ax_stereo,
        "lasco_map": lasco_map,
        "cor1_map": cor1_diff_map,
        "euvi_map": euvi_diff_map,
        "stereo_common_map": stereo_common_map,
    }

if __name__ == "__main__":
    target_time = "2022-06-13T03:36:18"

    spheroid = SpheroidDome3DParams(
        kappa=0.40,
        epsilon=0.62,
        anchor_lon_deg=-30.0,
        anchor_lat_deg=19.0,
        apex_lon_deg=-45.0,
        apex_lat_deg=17.0,
        apex_r_rsun=5.13,
        n_meridians=12,
        n_parallels=7,
        n_line_pts=240,
        only_above_surface=True,
        only_visible=True,
    )

    out_png = (
        "/mnt/d/wsl/home/kinno-7010/Research_data/GCS/output/"
        f"multiview_spheroid_{target_time.replace(':', '')}.png"
    )

    plot_multiview_spheroid(
        target_time_str=target_time,
        spheroid_params=spheroid,
        out_png=out_png,
        delta_time_min=10,
        euvi_outer_rsun=1.30,
        cor1_outer_rsun=4.0,
    )
