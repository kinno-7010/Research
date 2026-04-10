from __future__ import annotations

import json
import math
import importlib.util
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import os
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib

# -----------------------------------------------------------------------------
# User-code imports
# -----------------------------------------------------------------------------
# Adjust these if your actual filenames differ.
sys.path.append("/home/kinno-7010/Research_code/GCS")
sys.path.append("/home/kinno-7010/Research_code/SDO_Mk4_SOHO/py_folder")
sys.path.append("/home/kinno-7010/Research_code/STEREO-A/SECCHI/COR1/py_folder")
sys.path.append("/home/kinno-7010/Research_code/STEREO-A/SECCHI")
sys.path.append("/home/kinno-7010/Research_code/GCS/astrolibpy")
import mpfit

from plot_spheroid_C2 import (
    SpheroidDome3DParams,
    overlay_spheroid_on_coronagraph_axes,
    sample_spheroid_dome_wireframe_hpc,
    sample_spheroid_footprint_hpc,
    spheroid_axis_footpoint_hpc,
    spheroid_dome_apex_hpc,
)
from integrated_analysis import create_single_diff_from_time_image
from STEREO_integrated_plot import (
    create_integrated_stereo_image,
    build_common_reference_map,
    get_params as get_stereo_params,
)
from dataclasses import asdict, dataclass, replace

import plot_spheroid_C2 as sphmod
# -----------------------------------------------------------------------------
# Small containers
# -----------------------------------------------------------------------------
@dataclass
class ViewObservation:
    name: str
    obs_xy: np.ndarray
    obs_pa_deg: np.ndarray
    obs_r_rsun: np.ndarray
    reference_map: Any
    display_mode: str          # "earth_relpix" or "stereo_arcsec"
    weight_scale: float
    sigma_coord: float
    params_display: dict | None = None

# -----------------------------------------------------------------------------
# MPFIT loader
# -----------------------------------------------------------------------------
def import_mpfit_callable(mpfit_module_path: str | Path | None = None):
    """Return a callable MPFIT entry point.

    This function is robust against the following cases:
    1. `import mpfit` gives a module that directly defines a callable `mpfit`
    2. `from mpfit import mpfit` returns the callable itself
    3. `from mpfit import mpfit` returns a submodule, which then contains a callable `mpfit`
    4. a user-supplied mpfit.py path is given
    """
    def _resolve_callable(obj, where: str):
        # Case A: already callable (function or class)
        if callable(obj):
            return obj

        # Case B: module/object with attribute `mpfit` that is callable
        inner = getattr(obj, "mpfit", None)
        if callable(inner):
            return inner

        # Case C: module/object with nested attribute `mpfit.mpfit`
        if inner is not None:
            inner2 = getattr(inner, "mpfit", None)
            if callable(inner2):
                return inner2

        raise TypeError(f"Resolved MPFIT object from {where} is not callable: {type(obj)}")

    # --------------------------------------------------------------
    # 1) Try the already-imported module first
    # --------------------------------------------------------------
    try:
        return _resolve_callable(mpfit, "global import mpfit")
    except Exception:
        pass

    # --------------------------------------------------------------
    # 2) Try standard import patterns
    # --------------------------------------------------------------
    try:
        from mpfit import mpfit as imported_mpfit  # type: ignore
        return _resolve_callable(imported_mpfit, "from mpfit import mpfit")
    except Exception:
        pass

    try:
        import importlib
        imported_module = importlib.import_module("mpfit")
        return _resolve_callable(imported_module, 'importlib.import_module("mpfit")')
    except Exception:
        pass

    # --------------------------------------------------------------
    # 3) Fall back to a user-supplied mpfit.py path
    # --------------------------------------------------------------
    if mpfit_module_path is None:
        raise ImportError(
            "mpfit could not be imported as a callable. "
            "Set mpfit_module_path to a Python-3-compatible mpfit.py."
        )

    mpfit_module_path = Path(mpfit_module_path)
    if not mpfit_module_path.exists():
        raise FileNotFoundError(f"mpfit module not found: {mpfit_module_path}")

    spec = importlib.util.spec_from_file_location("mpfit_local", str(mpfit_module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for: {mpfit_module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return _resolve_callable(module, f"file path {mpfit_module_path}")

# -----------------------------------------------------------------------------
# Geometry / coordinate helpers
# -----------------------------------------------------------------------------
def _pa_from_xy(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Position angle in degrees, north=0, counterclockwise positive."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def _rho_rsun_from_xy(dx: np.ndarray, dy: np.ndarray, px_per_rsun: float) -> np.ndarray:
    return np.sqrt(dx * dx + dy * dy) / float(px_per_rsun)

def prepare_clicked_front_points(pa_deg: np.ndarray, rho_rsun: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort clicked leading-edge points by position angle without coarse binning.

    The previous implementation rebinned observations into 5-degree bins before the fit,
    which discarded many manually selected constraints and made it easier for MPFIT to drift
    toward boundary solutions. Here we keep all finite clicked points and only sort them by PA.
    """
    pa_deg = np.asarray(pa_deg, dtype=float) % 360.0
    rho_rsun = np.asarray(rho_rsun, dtype=float)

    good = np.isfinite(pa_deg) & np.isfinite(rho_rsun)
    pa_deg = pa_deg[good]
    rho_rsun = rho_rsun[good]

    if pa_deg.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(pa_deg)
    return pa_deg[order], rho_rsun[order]

def ensure_interactive_backend(preferred_backends: tuple[str, ...] = ("TkAgg", "QtAgg")) -> str:
    """Ensure that matplotlib is running with an interactive backend.

    integrated_analysis.py imports matplotlib with the Agg backend for batch-safe
    plotting. That is fine for file output, but this GUI fitter needs mouse/keyboard
    events. Therefore we explicitly switch away from Agg before creating figures.
    """
    backend0 = matplotlib.get_backend()
    backend0_l = str(backend0).lower()

    interactive_markers = ("tkagg", "qtagg", "qt5agg", "gtk3agg", "wxagg", "macosx")
    if any(m in backend0_l for m in interactive_markers):
        plt.ion()
        return str(matplotlib.get_backend())

    if not _has_gui_display():
        raise RuntimeError(
            "No GUI display was detected, but this script is waiting for interactive clicks. "
            "Run it in a GUI-enabled session (for example WSLg/X11) or switch to a file-based point input workflow."
        )

    tried: list[str] = []
    for backend in preferred_backends:
        try:
            plt.switch_backend(backend)
            plt.ion()
            backend_now = str(matplotlib.get_backend())
            print(f"[INFO] Switched matplotlib backend from {backend0} to {backend_now}.")
            return backend_now
        except Exception as exc:
            tried.append(f"{backend}: {exc}")

    raise RuntimeError(
        "Failed to switch matplotlib to an interactive backend. "
        f"Current backend={backend0}. Tried: {'; '.join(tried)}"
    )
    
def _pa_r_to_xy_rsun(pa_deg: np.ndarray, rho_rsun: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert (PA, rho/Rsun) to image-plane Cartesian coordinates in Rsun units."""
    pa_rad = np.deg2rad(np.asarray(pa_deg, dtype=float))
    rho = np.asarray(rho_rsun, dtype=float)
    x = rho * np.sin(pa_rad)
    y = rho * np.cos(pa_rad)
    return x, y
    
def pa_rsun_to_arcsec_xy(
    pa_deg: np.ndarray,
    rho_rsun: np.ndarray,
    rsun_arcsec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (PA, rho/Rsun) to arcsec coordinates around Sun center."""
    pa_rad = np.deg2rad(np.asarray(pa_deg, dtype=float))
    rho_arcsec = np.asarray(rho_rsun, dtype=float) * float(rsun_arcsec)
    x = rho_arcsec * np.sin(pa_rad)
    y = rho_arcsec * np.cos(pa_rad)
    return x, y
    
def centered_pa_rsun_to_xy(pa_deg: np.ndarray, rho_rsun: np.ndarray, px_per_rsun: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert (PA, rho/Rsun) to centered image coordinates used by the Earth-view panel."""
    pa_rad = np.deg2rad(np.asarray(pa_deg, dtype=float))
    rho_pix = np.asarray(rho_rsun, dtype=float) * float(px_per_rsun)
    x = rho_pix * np.sin(pa_rad)
    y = rho_pix * np.cos(pa_rad)
    return x, y
    
def centered_pixel_points_to_pa_rsun(points_xy: np.ndarray, px_per_rsun: float) -> tuple[np.ndarray, np.ndarray]:
    """For Earth-view panel from create_single_diff_from_time_image.

    That panel uses centered pixel-offset coordinates, so the Sun center is (0, 0).
    """
    dx = np.asarray(points_xy[:, 0], dtype=float)
    dy = np.asarray(points_xy[:, 1], dtype=float)
    pa = _pa_from_xy(dx, dy)
    rho = _rho_rsun_from_xy(dx, dy, px_per_rsun)
    return pa, rho

def map_pa_rsun_to_pixel_xy(
    pa_deg: np.ndarray,
    rho_rsun: np.ndarray,
    reference_map,
    px_per_rsun: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (PA, rho/Rsun) to map pixel coordinates used by the WCS panel."""
    sun_center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=reference_map.coordinate_frame)
    sun_center_pix = reference_map.world_to_pixel(sun_center)
    x0 = float(sun_center_pix.x.value)
    y0 = float(sun_center_pix.y.value)

    pa_rad = np.deg2rad(np.asarray(pa_deg, dtype=float))
    rho_pix = np.asarray(rho_rsun, dtype=float) * float(px_per_rsun)
    x = x0 + rho_pix * np.sin(pa_rad)
    y = y0 + rho_pix * np.cos(pa_rad)
    return x, y

def map_pixel_points_to_pa_rsun(
    points_xy: np.ndarray,
    reference_map,
    px_per_rsun: float,
) -> tuple[np.ndarray, np.ndarray]:
    """For WCS panel from create_integrated_stereo_image.

    That panel uses map pixel coordinates, so we must subtract the actual solar-center pixel.
    """
    sun_center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=reference_map.coordinate_frame)
    sun_center_pix = reference_map.world_to_pixel(sun_center)
    x0 = float(sun_center_pix.x.value)
    y0 = float(sun_center_pix.y.value)

    dx = np.asarray(points_xy[:, 0], dtype=float) - x0
    dy = np.asarray(points_xy[:, 1], dtype=float) - y0
    pa = _pa_from_xy(dx, dy)
    rho = _rho_rsun_from_xy(dx, dy, px_per_rsun)
    return pa, rho

def _hpc_to_rel_pix(coords_hpc: SkyCoord, rsun_arcsec: float, px_per_rsun: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert HPC Tx/Ty [arcsec] to Earth-composite relative pixel coordinates."""
    x_arcsec = np.asarray(coords_hpc.Tx.to_value(u.arcsec), dtype=float)
    y_arcsec = np.asarray(coords_hpc.Ty.to_value(u.arcsec), dtype=float)
    x_px = (x_arcsec / float(rsun_arcsec)) * float(px_per_rsun)
    y_px = (y_arcsec / float(rsun_arcsec)) * float(px_per_rsun)
    return x_px, y_px

def _hpc_to_arcsec_xy(coords_hpc: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
    """Convert HPC coordinates directly to arcsec for the STEREO-A panel."""
    x_arcsec = np.asarray(coords_hpc.Tx.to_value(u.arcsec), dtype=float)
    y_arcsec = np.asarray(coords_hpc.Ty.to_value(u.arcsec), dtype=float)
    return x_arcsec, y_arcsec

def _map_scale_arcsec_per_pix(reference_map) -> float:
    u0 = reference_map.scale[0].unit
    s0 = abs(float(reference_map.scale[0].to_value(u0)))
    s1 = abs(float(reference_map.scale[1].to_value(u0)))
    return 0.5 * (s0 + s1)


def arcsec_points_to_pa_rsun(
    points_xy: np.ndarray,
    rsun_arcsec: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(points_xy[:, 0], dtype=float)
    y = np.asarray(points_xy[:, 1], dtype=float)
    pa = _pa_from_xy(x, y)
    rho = np.sqrt(x * x + y * y) / float(rsun_arcsec)
    return pa, rho


def _nearest_distances_xy(obs_xy: np.ndarray, model_xy: np.ndarray) -> np.ndarray:
    if obs_xy.size == 0 or model_xy.size == 0:
        return np.full((max(len(obs_xy), 1),), 1.0e3, dtype=float)
    d2 = np.sum((obs_xy[:, None, :] - model_xy[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.min(d2, axis=1))


def _resample_polyline_by_arclength(xy: np.ndarray, n_out: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[0] == 0 or xy.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    if xy.shape[0] == 1 or n_out <= 1:
        return xy[[0]].copy()

    seg = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0.0:
        idx = np.linspace(0, xy.shape[0] - 1, max(2, n_out)).round().astype(int)
        return xy[np.clip(idx, 0, xy.shape[0] - 1)]

    s_new = np.linspace(0.0, total, max(2, n_out))
    x_new = np.interp(s_new, s, xy[:, 0])
    y_new = np.interp(s_new, s, xy[:, 1])
    return np.column_stack([x_new, y_new])


def _symmetric_front_distance_residuals(
    obs_xy: np.ndarray,
    model_xy: np.ndarray,
    *,
    sigma_coord: float,
    reverse_weight: float = 1.0,
    reverse_min_points: int = 32,
) -> np.ndarray:
    obs_xy = np.asarray(obs_xy, dtype=float)
    model_xy = np.asarray(model_xy, dtype=float)

    if obs_xy.ndim != 2 or model_xy.ndim != 2 or obs_xy.shape[0] == 0 or model_xy.shape[0] == 0:
        n = max(int(obs_xy.shape[0]) if obs_xy.ndim == 2 else 0, 1)
        return np.full(n, 1.0e3 / max(float(sigma_coord), 1.0), dtype=float)

    d_obs = _nearest_distances_xy(obs_xy, model_xy) / max(float(sigma_coord), 1.0)
    d_obs = d_obs / max(np.sqrt(obs_xy.shape[0]), 1.0)

    n_rev = max(int(reverse_min_points), int(obs_xy.shape[0]))
    model_xy_rs = _resample_polyline_by_arclength(model_xy, n_rev)
    d_mod = _nearest_distances_xy(model_xy_rs, obs_xy) / max(float(sigma_coord), 1.0)
    d_mod = d_mod / max(np.sqrt(model_xy_rs.shape[0]), 1.0)

    return np.concatenate([d_obs.astype(float), float(reverse_weight) * d_mod.astype(float)])


def unwrap_angles_with_largest_gap(
    angles_deg: np.ndarray,
    boundary: float | None = None,
) -> tuple[np.ndarray, float]:
    ang = np.mod(np.asarray(angles_deg, dtype=float), 360.0)
    if ang.size == 0:
        return ang.copy(), 0.0 if boundary is None else float(boundary)

    if boundary is None:
        s = np.sort(ang)
        diffs = np.diff(np.r_[s, s[0] + 360.0])
        igap = int(np.argmax(diffs))
        boundary = float(s[(igap + 1) % len(s)])

    out = (ang - boundary) % 360.0 + boundary
    return out, float(boundary)


def envelope_from_pa_rho(
    pa_deg: np.ndarray,
    rho_rsun: np.ndarray,
    *,
    bin_deg: float,
    boundary: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    pa_deg = np.asarray(pa_deg, dtype=float)
    rho_rsun = np.asarray(rho_rsun, dtype=float)

    good = np.isfinite(pa_deg) & np.isfinite(rho_rsun)
    pa_deg = pa_deg[good]
    rho_rsun = rho_rsun[good]
    if pa_deg.size < 2:
        return np.array([]), np.array([]), 0.0 if boundary is None else float(boundary)

    pa_u, boundary = unwrap_angles_with_largest_gap(pa_deg, boundary=boundary)
    pa_min = float(np.nanmin(pa_u))
    pa_max = float(np.nanmax(pa_u))

    if not np.isfinite(pa_min) or not np.isfinite(pa_max) or (pa_max - pa_min) <= 0:
        return np.array([]), np.array([]), boundary

    edges = np.arange(math.floor(pa_min), math.ceil(pa_max) + bin_deg, bin_deg, dtype=float)
    if edges.size < 2:
        edges = np.array([pa_min, pa_max + bin_deg], dtype=float)

    ibin = np.digitize(pa_u, edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    rho_env = np.full(centers.shape, np.nan, dtype=float)

    for i in range(len(centers)):
        m = ibin == i
        if np.any(m):
            rho_env[i] = np.nanmax(rho_rsun[m])

    valid = np.isfinite(rho_env)
    return centers[valid], rho_env[valid], boundary


def sample_spheroid_surface_hpc(
    reference_map,
    spheroid_params,
    *,
    dense_for_fit: bool = False,
):
    params_use = spheroid_params
    if dense_for_fit:
        params_use = replace(
            spheroid_params,
            n_meridians=max(int(getattr(spheroid_params, "n_meridians", 12)), 72),
            n_parallels=max(int(getattr(spheroid_params, "n_parallels", 7)), 36),
            n_line_pts=max(int(getattr(spheroid_params, "n_line_pts", 240)), 720),
        )

    geom = sphmod._spheroid_axis_geometry_rsun(params_use)
    center = np.asarray(geom["center"], dtype=float).reshape(3, 1, 1)
    axis_u = np.asarray(geom["axis_u"], dtype=float).reshape(3, 1, 1)
    e1 = np.asarray(geom["e1"], dtype=float).reshape(3, 1, 1)
    e2 = np.asarray(geom["e2"], dtype=float).reshape(3, 1, 1)
    a = float(params_use.a_rsun)
    b = float(params_use.b_rsun)

    n_alpha = 91 if dense_for_fit else 61
    n_beta = 361 if dense_for_fit else 181

    alphas = np.linspace(0.0, np.pi, n_alpha, endpoint=True)
    betas = np.linspace(0.0, 2.0 * np.pi, n_beta, endpoint=False)
    ca = np.cos(alphas)[None, :, None]
    sa = np.sin(alphas)[None, :, None]
    cb = np.cos(betas)[None, None, :]
    sb = np.sin(betas)[None, None, :]

    cart = center + a * ca * axis_u + b * sa * (cb * e1 + sb * e2)
    cart = cart.reshape(3, -1)

    rr = np.sqrt(np.sum(cart ** 2, axis=0))
    if params_use.only_above_surface:
        mask_surface = rr >= 1.0
    else:
        mask_surface = np.ones(rr.shape, dtype=bool)

    rep = sphmod.CartesianRepresentation(
        cart[0] * sphmod.u.R_sun,
        cart[1] * sphmod.u.R_sun,
        cart[2] * sphmod.u.R_sun,
    )
    coords_hgs = sphmod.SkyCoord(
        rep,
        frame=sphmod.sunpy_frames.HeliographicStonyhurst,
        obstime=reference_map.date,
    )

    mask_vis = sphmod._visible_mask(coords_hgs, reference_map, only_visible=params_use.only_visible)
    mask = mask_surface & mask_vis

    if not np.any(mask) and params_use.only_visible:
        mask = mask_surface
    if not np.any(mask):
        return sphmod.SkyCoord([], [], unit=sphmod.u.arcsec, frame=reference_map.coordinate_frame)

    coords_hpc = coords_hgs.transform_to(reference_map.coordinate_frame)
    return coords_hpc[mask]

def _nearest_front_distance_rsun(
    obs_pa_deg: np.ndarray,
    obs_r_rsun: np.ndarray,
    model_pa_deg: np.ndarray,
    model_r_rsun: np.ndarray,
) -> np.ndarray:
    """Return nearest image-plane distance [Rsun] from each observed point to the model front.

    This compares the clicked leading-edge points with the projected model front
    in Cartesian image-plane coordinates, instead of dropping points whose PA is
    outside the model coverage.
    """
    if model_pa_deg.size < 2:
        return np.full(obs_pa_deg.shape, 1.0e6, dtype=float)

    ox, oy = _pa_r_to_xy_rsun(obs_pa_deg, obs_r_rsun)
    mx, my = _pa_r_to_xy_rsun(model_pa_deg, model_r_rsun)

    obs_xy = np.column_stack([ox, oy]).astype(float)
    mdl_xy = np.column_stack([mx, my]).astype(float)

    # Pairwise Euclidean distances in the image plane [Rsun]
    d2 = np.sum((obs_xy[:, None, :] - mdl_xy[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.min(d2, axis=1))

def _collect_hpc_points_as_pa_rsun(lines_hpc: list[SkyCoord], reference_map) -> tuple[np.ndarray, np.ndarray]:
    """Flatten projected spheroid curves and convert to (PA, rho/Rsun)."""
    rsun_arcsec = float(reference_map.rsun_obs.to_value(u.arcsec))

    pa_all: list[np.ndarray] = []
    rho_all: list[np.ndarray] = []
    for line in lines_hpc:
        x_arcsec = np.asarray(line.Tx.to_value(u.arcsec), dtype=float)
        y_arcsec = np.asarray(line.Ty.to_value(u.arcsec), dtype=float)
        pa_all.append(_pa_from_xy(x_arcsec, y_arcsec))
        rho_all.append(np.sqrt(x_arcsec * x_arcsec + y_arcsec * y_arcsec) / rsun_arcsec)

    if not pa_all:
        return np.array([], dtype=float), np.array([], dtype=float)

    return np.concatenate(pa_all), np.concatenate(rho_all)

def _has_gui_display() -> bool:
    """Return True when a GUI display is likely available."""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True

    display_vars = ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET")
    return any(bool(os.environ.get(v)) for v in display_vars)


def backend_is_interactive() -> bool:
    """Return True if matplotlib backend supports GUI interaction."""
    backend = str(matplotlib.get_backend()).lower()
    interactive_markers = ("tkagg", "qtagg", "qt5agg", "gtk3agg", "wxagg", "macosx")
    return any(m in backend for m in interactive_markers)

# -----------------------------------------------------------------------------
# Front extraction on the image plane
# -----------------------------------------------------------------------------
def bin_outermost_front_by_pa(
    pa_deg: np.ndarray,
    rho_rsun: np.ndarray,
    pa_bin_deg: float = 5.0,
    r_min_rsun: float = 1.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce arbitrary image-plane points to one outermost point per PA bin."""
    pa_deg = np.asarray(pa_deg, dtype=float) % 360.0
    rho_rsun = np.asarray(rho_rsun, dtype=float)

    good = np.isfinite(pa_deg) & np.isfinite(rho_rsun) & (rho_rsun >= float(r_min_rsun))
    pa_deg = pa_deg[good]
    rho_rsun = rho_rsun[good]

    if pa_deg.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    nbins = int(round(360.0 / float(pa_bin_deg)))
    edges = np.linspace(0.0, 360.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    idx = np.digitize(pa_deg, edges) - 1
    idx[idx == nbins] = 0

    out_pa = []
    out_r = []
    for ib in range(nbins):
        m = idx == ib
        if not np.any(m):
            continue
        out_pa.append(float(centers[ib]))
        out_r.append(float(np.nanmax(rho_rsun[m])))

    return np.asarray(out_pa, dtype=float), np.asarray(out_r, dtype=float)


def sample_model_front_by_pa(
    params: SpheroidDome3DParams,
    reference_map,
    pa_bin_deg: float = 5.0,
    r_min_rsun: float = 1.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the 3D spheroid and extract the outermost front on the image plane.

    If the visible-side clipping removes all curves, retry with only_visible=False
    so that the plotting and the front sampling remain consistent.
    """
    params_use = params
    wire_lines_hpc = sample_spheroid_dome_wireframe_hpc(params_use, reference_map)

    if (len(wire_lines_hpc) == 0) and bool(params.only_visible):
        params_use = replace(params, only_visible=False)
        wire_lines_hpc = sample_spheroid_dome_wireframe_hpc(params_use, reference_map)

    pa_all, rho_all = _collect_hpc_points_as_pa_rsun(wire_lines_hpc, reference_map)
    return bin_outermost_front_by_pa(pa_all, rho_all, pa_bin_deg=pa_bin_deg, r_min_rsun=r_min_rsun)

def interpolate_model_r_at_obs_pa(
    obs_pa_deg: np.ndarray,
    model_pa_deg: np.ndarray,
    model_r_rsun: np.ndarray,
) -> np.ndarray:
    """Periodic linear interpolation in PA."""
    if model_pa_deg.size < 2:
        return np.full_like(obs_pa_deg, np.nan, dtype=float)

    srt = np.argsort(model_pa_deg)
    xp = np.asarray(model_pa_deg[srt], dtype=float)
    fp = np.asarray(model_r_rsun[srt], dtype=float)

    xp_ext = np.concatenate([xp - 360.0, xp, xp + 360.0])
    fp_ext = np.concatenate([fp, fp, fp])
    return np.interp(obs_pa_deg, xp_ext, fp_ext, left=np.nan, right=np.nan)


# -----------------------------------------------------------------------------
# Plot helpers for the STEREO / WCS panel
# -----------------------------------------------------------------------------
def _world_to_pixel_xy(reference_map, coords_hpc: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
    xp, yp = reference_map.world_to_pixel(coords_hpc)
    return np.asarray(xp.value, dtype=float), np.asarray(yp.value, dtype=float)


def overlay_spheroid_on_wcs_axes(
    ax,
    reference_map,
    spheroid_params: SpheroidDome3DParams,
    *,
    color: str = "#00FF00",
    lw_wire: float = 1.0,
    lw_footprint: float = 2.0,
    alpha_wire: float = 0.85,
    alpha_footprint: float = 0.95,
    zorder_wire: int = 6,
) -> SpheroidDome3DParams:
    """Overlay the same spheroid on the STEREO-A panel in arcsec coordinates.

    Important:
    The STEREO integrated panel is displayed in Solar X/Y [arcsec], not map pixels.
    Therefore we must draw HPC Tx/Ty directly in arcsec instead of using world_to_pixel().
    """
    params_use = spheroid_params
    wire_lines_hpc = sample_spheroid_dome_wireframe_hpc(params_use, reference_map)

    if (len(wire_lines_hpc) == 0) and bool(params_use.only_visible):
        params_use = replace(params_use, only_visible=False)
        wire_lines_hpc = sample_spheroid_dome_wireframe_hpc(params_use, reference_map)

    for ln in wire_lines_hpc:
        x_arcsec, y_arcsec = _hpc_to_arcsec_xy(ln)
        ax.plot(x_arcsec, y_arcsec, color=color, linewidth=lw_wire, alpha=alpha_wire, zorder=zorder_wire)

    for fp in sample_spheroid_footprint_hpc(params_use, reference_map):
        x_arcsec, y_arcsec = _hpc_to_arcsec_xy(fp)
        ax.plot(x_arcsec, y_arcsec, color=color, linewidth=lw_footprint, alpha=alpha_footprint, zorder=zorder_wire + 1)

    return params_use

def overlay_spheroid_on_earth_axes(
    ax,
    reference_map,
    params_lasco: dict,
    spheroid_params: SpheroidDome3DParams,
    *,
    color: str = "#00FF00",
    lw_wire: float = 1.0,
    lw_footprint: float = 2.2,
    alpha_wire: float = 0.85,
    alpha_footprint: float = 0.95,
    zorder_wire: int = 6,
) -> SpheroidDome3DParams:
    """Overlay spheroid on the Earth-view composite axes without extra anchor/apex markers.

    This follows plot_spheroid_C2.py coordinate conventions: relative pixel coordinates
    centered on the Sun.
    """
    rsun_arcsec = float(reference_map.rsun_obs.to_value(u.arcsec))
    px_per_rsun = float(params_lasco["px_per_rsun"])

    params_use = spheroid_params
    wire_lines_hpc = sample_spheroid_dome_wireframe_hpc(params_use, reference_map)
    if (len(wire_lines_hpc) == 0) and bool(params_use.only_visible):
        params_use = replace(params_use, only_visible=False)
        wire_lines_hpc = sample_spheroid_dome_wireframe_hpc(params_use, reference_map)

    for ln in wire_lines_hpc:
        x_px, y_px = _hpc_to_rel_pix(ln, rsun_arcsec, px_per_rsun)
        ax.plot(x_px, y_px, color=color, linewidth=lw_wire, alpha=alpha_wire, zorder=zorder_wire)

    footprint_lines_hpc = sample_spheroid_footprint_hpc(params_use, reference_map)
    for fp in footprint_lines_hpc:
        x_px, y_px = _hpc_to_rel_pix(fp, rsun_arcsec, px_per_rsun)
        ax.plot(x_px, y_px, color=color, linewidth=lw_footprint, alpha=alpha_footprint, zorder=zorder_wire + 1)

    return params_use


# -----------------------------------------------------------------------------
# Interactive point picking
# -----------------------------------------------------------------------------
def collect_clicked_points(ax, fig, prompt: str, color: str = "cyan") -> np.ndarray:
    """Left click = add, backspace = remove last, enter = finish.

    Clicked points are displayed as circles so they remain visually distinct from
    the fitted model markers that are drawn later as stars.
    """
    if not backend_is_interactive():
        raise RuntimeError(
            "Matplotlib is running with a non-interactive backend "
            f"({matplotlib.get_backend()}). GUI point picking cannot start."
        )

    print(prompt)
    print("  left click: add point | backspace: remove last point | enter: finish")

    points: list[tuple[float, float]] = []
    artists: list[Any] = []
    done = {"value": False}

    def redraw():
        fig.canvas.draw_idle()
        try:
            fig.canvas.flush_events()
        except Exception:
            pass

    def on_click(event):
        if done["value"]:
            return
        if event.inaxes is not ax:
            return
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        points.append((float(event.xdata), float(event.ydata)))
        art = ax.plot(
            event.xdata,
            event.ydata,
            marker="o",
            linestyle="None",
            markerfacecolor="green",
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=5,
            zorder=50,
        )[0]
        artists.append(art)
        redraw()

    def on_key(event):
        if event.key in ("enter", "return"):
            done["value"] = True
        elif event.key == "backspace":
            if points:
                points.pop()
                art = artists.pop()
                art.remove()
                redraw()

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
    cid_key = fig.canvas.mpl_connect("key_press_event", on_key)

    try:
        try:
            fig.show()
        except Exception:
            pass
        try:
            plt.show(block=False)
        except Exception:
            pass
        plt.pause(0.1)

        while not done["value"]:
            plt.pause(0.05)
    finally:
        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_key)

    return np.asarray(points, dtype=float)
# -----------------------------------------------------------------------------
# MPFIT state construction
# -----------------------------------------------------------------------------

def make_two_point_spheroid_params(
    *,
    anchor_lon_deg: float,
    anchor_lat_deg: float,
    apex_lon_deg: float,
    apex_lat_deg: float,
    apex_r_rsun: float,
    kappa: float,
    epsilon: float,
    n_meridians: int,
    n_parallels: int,
    n_line_pts: int,
) -> SpheroidDome3DParams:
    return SpheroidDome3DParams(
        kappa=float(kappa),
        epsilon=float(epsilon),
        anchor_lon_deg=float(anchor_lon_deg),
        anchor_lat_deg=float(anchor_lat_deg),
        apex_lon_deg=float(apex_lon_deg),
        apex_lat_deg=float(apex_lat_deg),
        apex_r_rsun=float(apex_r_rsun),
        n_meridians=int(n_meridians),
        n_parallels=int(n_parallels),
        n_line_pts=int(n_line_pts),
        only_above_surface=True,
        only_visible=True,
    )


def view_model_front_xy_pa_rho(
    view: ViewObservation,
    spheroid_params: SpheroidDome3DParams,
    *,
    front_bin_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords_hpc = sample_spheroid_surface_hpc(
        view.reference_map,
        spheroid_params,
        dense_for_fit=True,
    )
    if len(coords_hpc) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    if view.display_mode == "earth_relpix":
        rsun_arcsec = float(view.reference_map.rsun_obs.to_value(u.arcsec))
        px_per_rsun = float(view.params_display["px_per_rsun"])
        x_mod, y_mod = sphmod._hpc_to_rel_pix(coords_hpc, rsun_arcsec, px_per_rsun)
        x_mod = np.asarray(x_mod, dtype=float)
        y_mod = np.asarray(y_mod, dtype=float)
        rho_mod = np.sqrt(x_mod * x_mod + y_mod * y_mod) / px_per_rsun
        pa_mod = _pa_from_xy(x_mod, y_mod)

        pa_front, rho_front, _ = envelope_from_pa_rho(pa_mod, rho_mod, bin_deg=front_bin_deg)
        if len(pa_front) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        x_front, y_front = centered_pa_rsun_to_xy(pa_front, rho_front, px_per_rsun)
        return x_front, y_front, pa_front, rho_front

    if view.display_mode == "stereo_arcsec":
        x_mod = np.asarray(coords_hpc.Tx.to_value(u.arcsec), dtype=float)
        y_mod = np.asarray(coords_hpc.Ty.to_value(u.arcsec), dtype=float)
        rsun_arcsec = float(view.reference_map.rsun_obs.to_value(u.arcsec))
        rho_mod = np.sqrt(x_mod * x_mod + y_mod * y_mod) / rsun_arcsec
        pa_mod = _pa_from_xy(x_mod, y_mod)

        pa_front, rho_front, _ = envelope_from_pa_rho(pa_mod, rho_mod, bin_deg=front_bin_deg)
        if len(pa_front) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        x_front, y_front = pa_rsun_to_arcsec_xy(pa_front, rho_front, rsun_arcsec)
        return x_front, y_front, pa_front, rho_front

    raise ValueError(f"Unknown display_mode: {view.display_mode}")


def make_centered_spheroid_params(
    lon_deg: float,
    lat_deg: float,
    apex_r_rsun: float,
    kappa: float,
    epsilon_fixed: float,
    *,
    n_meridians: int,
    n_parallels: int,
    n_line_pts: int,
) -> SpheroidDome3DParams:
    """Create the Earth-frame spheroid used for fitting and plotting.

    We fit the axis direction (lon, lat), height, and kappa.
    Epsilon is held fixed by default.
    """
    return SpheroidDome3DParams(
        kappa=float(kappa),
        epsilon=float(epsilon_fixed),
        center_lon_deg=float(lon_deg),
        center_lat_deg=float(lat_deg),
        apex_r_rsun=float(apex_r_rsun),
        n_meridians=int(n_meridians),
        n_parallels=int(n_parallels),
        n_line_pts=int(n_line_pts),
        only_above_surface=True,
        only_visible=True,
    )


def build_view_observations(
    earth_click_xy: np.ndarray,
    stereo_click_xy: np.ndarray,
    *,
    p_lasco: dict,
    lasco_map,
    stereo_common_map,
    equalize_view_weights: bool,
    click_sigma_px: float,
) -> list[ViewObservation]:
    obs_list: list[ViewObservation] = []

    earth_pa, earth_r = centered_pixel_points_to_pa_rsun(earth_click_xy, p_lasco["px_per_rsun"])
    earth_pa, earth_r = prepare_clicked_front_points(earth_pa, earth_r)
    if earth_pa.size == 0:
        raise ValueError("No valid Earth-view leading-edge points were selected.")
    earth_scale = 1.0 / math.sqrt(max(1, earth_pa.size)) if equalize_view_weights else 1.0
    obs_list.append(
        ViewObservation(
            name="Earth(K-COR+LASCO)",
            obs_xy=np.asarray(earth_click_xy, dtype=float),
            obs_pa_deg=earth_pa,
            obs_r_rsun=earth_r,
            reference_map=lasco_map,
            display_mode="earth_relpix",
            weight_scale=float(earth_scale),
            sigma_coord=float(click_sigma_px),
            params_display=p_lasco,
        )
    )

    # Right panel is plotted in arcsec, not map pixels.
    rsun_arcsec = float(stereo_common_map.rsun_obs.to_value(u.arcsec))
    stereo_pa, stereo_r = arcsec_points_to_pa_rsun(stereo_click_xy, rsun_arcsec)
    stereo_pa, stereo_r = prepare_clicked_front_points(stereo_pa, stereo_r)
    if stereo_pa.size == 0:
        raise ValueError("No valid STEREO-A/COR1 leading-edge points were selected.")
    stereo_scale = 1.0 / math.sqrt(max(1, stereo_pa.size)) if equalize_view_weights else 1.0
    sigma_arcsec = float(click_sigma_px) * _map_scale_arcsec_per_pix(stereo_common_map)
    obs_list.append(
        ViewObservation(
            name="STEREO-A(COR1)",
            obs_xy=np.asarray(stereo_click_xy, dtype=float),
            obs_pa_deg=stereo_pa,
            obs_r_rsun=stereo_r,
            reference_map=stereo_common_map,
            display_mode="stereo_arcsec",
            weight_scale=float(stereo_scale),
            sigma_coord=float(sigma_arcsec),
            params_display=None,
        )
    )

    return obs_list
# -----------------------------------------------------------------------------
# Residual function passed to MPFIT
# -----------------------------------------------------------------------------
def spheroid_mpfit_residual(
    p,
    fjac=None,
    fit_state: dict[str, Any] | None = None,
):
    if fit_state is None:
        return [-1, np.array([1.0e30], dtype=float)]

    try:
        epsilon = float(p[4]) if bool(fit_state["fit_epsilon"]) else float(fit_state["epsilon_fixed"])
        spheroid = make_two_point_spheroid_params(
            anchor_lon_deg=float(fit_state["anchor_lon_deg"]),
            anchor_lat_deg=float(fit_state["anchor_lat_deg"]),
            apex_lon_deg=float(p[0]),
            apex_lat_deg=float(p[1]),
            apex_r_rsun=float(p[2]),
            kappa=float(p[3]),
            epsilon=float(epsilon),
            n_meridians=int(fit_state["n_meridians"]),
            n_parallels=int(fit_state["n_parallels"]),
            n_line_pts=int(fit_state["n_line_pts"]),
        )
    except Exception:
        return [-1, np.full(64, 200.0, dtype=float)]

    residual_chunks: list[np.ndarray] = []

    for view in fit_state["views"]:
        obs_xy = np.asarray(view.obs_xy, dtype=float)
        if obs_xy.shape[0] < 3:
            continue

        mod_x, mod_y, pa_mod, rho_mod = view_model_front_xy_pa_rho(
            view,
            spheroid,
            front_bin_deg=float(fit_state["envelope_bin_deg"]),
        )

        if len(mod_x) < 5:
            nrev = max(int(fit_state["reverse_front_min_points"]), int(obs_xy.shape[0]))
            residual_chunks.append(np.full(obs_xy.shape[0] + nrev + 4, 120.0, dtype=float))
            continue

        mod_xy = np.column_stack([mod_x, mod_y])

        d_sym = _symmetric_front_distance_residuals(
            obs_xy,
            mod_xy,
            sigma_coord=float(view.sigma_coord),
            reverse_weight=float(fit_state["model_to_obs_weight"]),
            reverse_min_points=int(fit_state["reverse_front_min_points"]),
        )

        pa_obs = np.asarray(view.obs_pa_deg, dtype=float)
        rho_obs = np.asarray(view.obs_r_rsun, dtype=float)

        span = np.full(4, 60.0, dtype=float)
        if len(pa_obs) >= 2 and len(pa_mod) >= 5:
            pa_obs_env, rho_obs_env, boundary = envelope_from_pa_rho(
                pa_obs,
                rho_obs,
                bin_deg=float(fit_state["envelope_bin_deg"]),
                boundary=None,
            )
            pa_mod_env, rho_mod_env, _ = envelope_from_pa_rho(
                pa_mod,
                rho_mod,
                bin_deg=float(fit_state["envelope_bin_deg"]),
                boundary=boundary,
            )

            if len(pa_obs_env) >= 2 and len(pa_mod_env) >= 2:
                obs_min = float(np.min(pa_obs_env))
                obs_max = float(np.max(pa_obs_env))
                mod_min = float(np.min(pa_mod_env))
                mod_max = float(np.max(pa_mod_env))
                obs_ctr = 0.5 * (obs_min + obs_max)
                mod_ctr = 0.5 * (mod_min + mod_max)
                obs_wid = obs_max - obs_min
                mod_wid = mod_max - mod_min
                pa_sigma = max(float(fit_state["pa_edge_sigma_deg"]), 1.0)

                span = np.array([
                    (mod_min - obs_min) / pa_sigma,
                    (mod_max - obs_max) / pa_sigma,
                    (mod_ctr - obs_ctr) / pa_sigma,
                    (mod_wid - obs_wid) / pa_sigma,
                ], dtype=float)

        residual_chunks.append(float(view.weight_scale) * np.concatenate([d_sym.astype(float), span.astype(float)]))

    if not residual_chunks:
        return [0, np.full(64, 200.0, dtype=float)]

    return [0, np.concatenate(residual_chunks).astype(float)]
# -----------------------------------------------------------------------------
# Multistart MPFIT wrapper
# -----------------------------------------------------------------------------
def build_parinfo(
    p0: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    *,
    fit_epsilon: bool,
) -> list[dict[str, Any]]:
    names = ["apex_lon_deg", "apex_lat_deg", "apex_r_rsun", "kappa"]
    if fit_epsilon:
        names.append("epsilon")

    parinfo: list[dict[str, Any]] = []
    for value, name in zip(p0, names):
        lo, hi = bounds[name]
        parinfo.append(
            {
                "value": float(value),
                "fixed": 0,
                "limited": [1, 1],
                "limits": [float(lo), float(hi)],
                "parname": name,
            }
        )
    return parinfo

def generate_multistart_guesses(
    initial_guess: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    n_starts: int,
    rng_seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(int(rng_seed))
    starts = [np.asarray(initial_guess, dtype=float).copy()]

    keys = ["lon_deg", "lat_deg", "apex_r_rsun", "kappa"]
    for _ in range(max(0, int(n_starts) - 1)):
        p = np.empty(4, dtype=float)
        for i, key in enumerate(keys):
            lo, hi = bounds[key]
            p[i] = rng.uniform(lo, hi)
        starts.append(p)
    return starts


def fit_spheroid_with_mpfit(
    views: list[ViewObservation],
    *,
    anchor_lon_deg: float,
    anchor_lat_deg: float,
    initial_guess: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    fit_epsilon: bool,
    epsilon_fixed: float,
    envelope_bin_deg: float,
    click_sigma_px: float,
    pa_edge_sigma_deg: float,
    model_to_obs_weight: float,
    reverse_front_min_points: int,
    n_meridians: int,
    n_parallels: int,
    n_line_pts: int,
    n_starts: int,
    rng_seed: int,
    mpfit_module_path: str | Path | None,
    maxiter: int = 200,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    gtol: float = 1e-10,
    quiet: int = 1,
) -> dict[str, Any]:
    mpfit_callable = import_mpfit_callable(mpfit_module_path)

    fit_state = {
        "views": views,
        "anchor_lon_deg": float(anchor_lon_deg),
        "anchor_lat_deg": float(anchor_lat_deg),
        "fit_epsilon": bool(fit_epsilon),
        "epsilon_fixed": float(epsilon_fixed),
        "envelope_bin_deg": float(envelope_bin_deg),
        "click_sigma_px": float(click_sigma_px),
        "pa_edge_sigma_deg": float(pa_edge_sigma_deg),
        "model_to_obs_weight": float(model_to_obs_weight),
        "reverse_front_min_points": int(reverse_front_min_points),
        "n_meridians": int(n_meridians),
        "n_parallels": int(n_parallels),
        "n_line_pts": int(n_line_pts),
    }

    names = ["apex_lon_deg", "apex_lat_deg", "apex_r_rsun", "kappa"]
    if fit_epsilon:
        names.append("epsilon")

    p0 = np.array([float(initial_guess[name]) for name in names], dtype=float)

    rng = np.random.default_rng(int(rng_seed))
    starts = [p0.copy()]
    for _ in range(max(0, int(n_starts) - 1)):
        trial = np.empty_like(p0)
        for i, name in enumerate(names):
            lo, hi = bounds[name]
            trial[i] = rng.uniform(lo, hi)
        starts.append(trial)

    best_result = None
    best_fnorm = np.inf

    print("Fit start")

    for i, start in enumerate(starts, start=1):
        parinfo = build_parinfo(start, bounds=bounds, fit_epsilon=fit_epsilon)

        result = mpfit_callable(
            spheroid_mpfit_residual,
            start,
            functkw={"fit_state": fit_state},
            parinfo=parinfo,
            maxiter=int(maxiter),
            ftol=float(ftol),
            xtol=float(xtol),
            gtol=float(gtol),
            quiet=int(quiet),
        )

        status = int(getattr(result, "status", -999))
        fnorm = float(getattr(result, "fnorm", np.inf))
        success = status > 0 and np.isfinite(fnorm)
        print(f"[INFO] multistart {i}/{len(starts)} status={status} fnorm={fnorm:.6g} success={success}")

        if success and fnorm < best_fnorm:
            best_fnorm = fnorm
            best_result = result

    if best_result is None:
        raise RuntimeError("MPFIT failed for all multistart seeds.")

    params = np.asarray(best_result.params, dtype=float)
    perror = getattr(best_result, "perror", None)
    perror = np.asarray(perror, dtype=float) if perror is not None else np.full_like(params, np.nan)

    dof = int(sum(len(v.obs_xy) for v in views) - len(params))
    if dof > 0 and np.isfinite(best_result.fnorm):
        perror_scaled = perror * np.sqrt(float(best_result.fnorm) / float(dof))
    else:
        perror_scaled = np.full_like(perror, np.nan)

    best_param_dict = {
        "anchor_lon_deg": float(anchor_lon_deg),
        "anchor_lat_deg": float(anchor_lat_deg),
        "apex_lon_deg": float(params[0]),
        "apex_lat_deg": float(params[1]),
        "apex_r_rsun": float(params[2]),
        "kappa": float(params[3]),
        "epsilon": float(params[4]) if fit_epsilon else float(epsilon_fixed),
    }

    perror_dict = {
        "anchor_lon_deg": np.nan,
        "anchor_lat_deg": np.nan,
        "apex_lon_deg": float(perror[0]),
        "apex_lat_deg": float(perror[1]),
        "apex_r_rsun": float(perror[2]),
        "kappa": float(perror[3]),
        "epsilon": float(perror[4]) if fit_epsilon else np.nan,
    }
    perror_scaled_dict = {
        "anchor_lon_deg": np.nan,
        "anchor_lat_deg": np.nan,
        "apex_lon_deg": float(perror_scaled[0]),
        "apex_lat_deg": float(perror_scaled[1]),
        "apex_r_rsun": float(perror_scaled[2]),
        "kappa": float(perror_scaled[3]),
        "epsilon": float(perror_scaled[4]) if fit_epsilon else np.nan,
    }

    best_spheroid = make_two_point_spheroid_params(
        anchor_lon_deg=best_param_dict["anchor_lon_deg"],
        anchor_lat_deg=best_param_dict["anchor_lat_deg"],
        apex_lon_deg=best_param_dict["apex_lon_deg"],
        apex_lat_deg=best_param_dict["apex_lat_deg"],
        apex_r_rsun=best_param_dict["apex_r_rsun"],
        kappa=best_param_dict["kappa"],
        epsilon=best_param_dict["epsilon"],
        n_meridians=int(n_meridians),
        n_parallels=int(n_parallels),
        n_line_pts=int(n_line_pts),
    )

    return {
        "mpfit_result": best_result,
        "best_param_dict": best_param_dict,
        "perror_dict": perror_dict,
        "perror_scaled_dict": perror_scaled_dict,
        "best_spheroid": best_spheroid,
        "dof": dof,
    }
# -----------------------------------------------------------------------------
# End-to-end workflow
# -----------------------------------------------------------------------------
def run_spheroid_mpfit_gui(
    *,
    target_time_str: str,
    earth_delta_time_min: int,
    cor1_base_minutes_before: int,
    euvi_dt_minutes: int,
    earth_xlim_min: float,
    earth_xlim_max: float,
    earth_ylim_min: float,
    earth_ylim_max: float,
    mk4_inner: float,
    mk4_outer_lasco_inner: float,
    lasco_outer: float,
    euvi_outer_rsun: float,
    cor1_outer_rsun: float,
    anchor_lon_deg: float,
    anchor_lat_deg: float,
    initial_guess: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    fit_epsilon: bool,
    epsilon_fixed: float,
    envelope_bin_deg: float,
    click_sigma_px: float,
    pa_edge_sigma_deg: float,
    model_to_obs_weight: float,
    reverse_front_min_points: int,
    equalize_view_weights: bool,
    n_meridians: int,
    n_parallels: int,
    n_line_pts: int,
    n_starts: int,
    rng_seed: int,
    mpfit_module_path: str | Path | None,
    out_dir: str | Path,
):
    backend_now = ensure_interactive_backend()
    print(f"[INFO] Using matplotlib backend: {backend_now}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 8), dpi=180)
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)

    try:
        fig.show()
    except Exception:
        pass
    try:
        plt.show(block=False)
    except Exception:
        pass
    plt.pause(0.1)

    earth_res = create_single_diff_from_time_image(
        ax0,
        target_time_str,
        earth_delta_time_min,
        mk4_inner=mk4_inner,
        mk4_outer_lasco_inner=mk4_outer_lasco_inner,
        lasco_outer=lasco_outer,
        xlim_min=earth_xlim_min,
        xlim_max=earth_xlim_max,
        ylim_min=earth_ylim_min,
        ylim_max=earth_ylim_max,
    )
    p_lasco = earth_res["params_lasco"]
    lasco_map = earth_res["lasco_map"]

    stereo_ret = create_integrated_stereo_image(
        ax1,
        target_time=target_time_str,
        cor1_base_minutes_before=cor1_base_minutes_before,
        euvi_dt_minutes=euvi_dt_minutes,
        euvi_outer_rsun=euvi_outer_rsun,
        cor1_outer_rsun=cor1_outer_rsun,
    )
    ax1 = stereo_ret[0]
    euvi_diff_map = stereo_ret[2]

    stereo_common_map = build_common_reference_map(euvi_diff_map, outer_rsun=cor1_outer_rsun)

    fig.canvas.draw_idle()

    earth_click_xy = collect_clicked_points(
        ax0,
        fig,
        prompt="[Earth view] click the CME leading edge on K-COR + LASCO-C2.",
        color="magenta",
    )
    stereo_click_xy = collect_clicked_points(
        ax1,
        fig,
        prompt="[STEREO-A view] click the CME leading edge on EUVI + COR1.",
        color="cyan",
    )

    if earth_click_xy.shape[0] < 3 or stereo_click_xy.shape[0] < 3:
        raise RuntimeError("At least three points are required in each panel.")

    views = build_view_observations(
        earth_click_xy,
        stereo_click_xy,
        p_lasco=p_lasco,
        lasco_map=lasco_map,
        stereo_common_map=stereo_common_map,
        equalize_view_weights=equalize_view_weights,
        click_sigma_px=click_sigma_px,
    )

    fit_out = fit_spheroid_with_mpfit(
        views,
        anchor_lon_deg=anchor_lon_deg,
        anchor_lat_deg=anchor_lat_deg,
        initial_guess=initial_guess,
        bounds=bounds,
        fit_epsilon=fit_epsilon,
        epsilon_fixed=epsilon_fixed,
        envelope_bin_deg=envelope_bin_deg,
        click_sigma_px=click_sigma_px,
        pa_edge_sigma_deg=pa_edge_sigma_deg,
        model_to_obs_weight=model_to_obs_weight,
        reverse_front_min_points=reverse_front_min_points,
        n_meridians=n_meridians,
        n_parallels=n_parallels,
        n_line_pts=n_line_pts,
        n_starts=n_starts,
        rng_seed=rng_seed,
        mpfit_module_path=mpfit_module_path,
    )

    best = fit_out["best_param_dict"]
    perror = fit_out["perror_dict"]
    perror_scaled = fit_out["perror_scaled_dict"]
    best_spheroid = fit_out["best_spheroid"]
    mpfit_result = fit_out["mpfit_result"]

    print("\n[RESULT] best-fit parameters")
    print(f"  anchor_lon_deg = {best['anchor_lon_deg']:.4f}")
    print(f"  anchor_lat_deg = {best['anchor_lat_deg']:.4f}")
    print(f"  apex_lon_deg   = {best['apex_lon_deg']:.4f} +/- {perror['apex_lon_deg']:.4f} (formal), {perror_scaled['apex_lon_deg']:.4f} (scaled)")
    print(f"  apex_lat_deg   = {best['apex_lat_deg']:.4f} +/- {perror['apex_lat_deg']:.4f} (formal), {perror_scaled['apex_lat_deg']:.4f} (scaled)")
    print(f"  apex_r_rsun    = {best['apex_r_rsun']:.4f} +/- {perror['apex_r_rsun']:.4f} (formal), {perror_scaled['apex_r_rsun']:.4f} (scaled)")
    print(f"  kappa          = {best['kappa']:.4f} +/- {perror['kappa']:.4f} (formal), {perror_scaled['kappa']:.4f} (scaled)")
    print(f"  epsilon        = {best['epsilon']:.4f}" + (f" +/- {perror['epsilon']:.4f} (formal), {perror_scaled['epsilon']:.4f} (scaled)" if fit_epsilon else " (fixed)"))
    print(f"  status         = {getattr(mpfit_result, 'status', None)}")
    print(f"  fnorm          = {getattr(mpfit_result, 'fnorm', None)}")
    print(f"  niter          = {getattr(mpfit_result, 'niter', None)}")

    overlay_spheroid_on_earth_axes(
        ax0,
        lasco_map,
        p_lasco,
        best_spheroid,
        color="#00FF00",
    )
    overlay_spheroid_on_wcs_axes(
        ax1,
        stereo_common_map,
        best_spheroid,
        color="#00FF00",
    )

    earth_view = next(v for v in views if v.name == "Earth(K-COR+LASCO)")
    stereo_view = next(v for v in views if v.name == "STEREO-A(COR1)")

    ex, ey, _, _ = view_model_front_xy_pa_rho(
        earth_view,
        best_spheroid,
        front_bin_deg=envelope_bin_deg,
    )
    if len(ex) > 0:
        ax0.plot(
            ex,
            ey,
            linestyle="None",
            marker="*",
            markerfacecolor="green",
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=5,
            zorder=70,
        )

    sx, sy, _, _ = view_model_front_xy_pa_rho(
        stereo_view,
        best_spheroid,
        front_bin_deg=envelope_bin_deg,
    )
    if len(sx) > 0:
        ax1.plot(
            sx,
            sy,
            linestyle="None",
            marker="*",
            markerfacecolor="green",
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=5,
            zorder=70,
        )

    ax0.plot(
        earth_click_xy[:, 0],
        earth_click_xy[:, 1],
        linestyle="None",
        marker="o",
        markerfacecolor="green",
        markeredgecolor="black",
        markeredgewidth=1,
        markersize=5,
        zorder=60,
    )
    ax1.plot(
        stereo_click_xy[:, 0],
        stereo_click_xy[:, 1],
        linestyle="None",
        marker="o",
        markerfacecolor="green",
        markeredgecolor="black",
        markeredgewidth=1,
        markersize=5,
        zorder=60,
    )

    ax0.set_title(ax0.get_title() + "\nMPFIT best-fit spheroid")
    ax1.set_title(ax1.get_title() + "\nMPFIT best-fit spheroid")

    plt.tight_layout()

    safe_time = target_time_str.replace(":", "").replace("-", "").replace(" ", "_").replace("T", "_")
    out_png = out_dir / f"spheroid_mpfit_fit_{safe_time}.png"
    out_json = out_dir / f"spheroid_mpfit_fit_{safe_time}.json"

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"[DONE] saved figure: {out_png}")

    payload = {
        "target_time_str": target_time_str,
        "earth_click_xy": earth_click_xy.tolist(),
        "stereo_click_xy": stereo_click_xy.tolist(),
        "best_params": best,
        "formal_1sigma": perror,
        "scaled_1sigma": perror_scaled,
        "mpfit_status": int(getattr(mpfit_result, "status", -999)),
        "mpfit_fnorm": float(getattr(mpfit_result, "fnorm", np.nan)),
        "mpfit_niter": int(getattr(mpfit_result, "niter", -1)),
        "best_spheroid_dataclass": asdict(best_spheroid),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] saved parameters: {out_json}")

    return {
        "figure_path": out_png,
        "json_path": out_json,
        "best_spheroid": best_spheroid,
        "payload": payload,
    }
        
if __name__ == "__main__":
    mpfit_module_path = "/home/kinno-7010/Research_code/GCS/astrolibpy/mpfit/mpfit.py"

    target_time_str = "2022-06-13T03:36:18"

    earth_delta_time_min = 10
    cor1_base_minutes_before = 10
    euvi_dt_minutes = 10

    mk4_inner = 1.4
    mk4_outer_lasco_inner = 3.0
    lasco_outer = 6.0
    euvi_outer_rsun = 1.30
    cor1_outer_rsun = 4.0

    earth_xlim_min = -512
    earth_xlim_max = 512
    earth_ylim_min = -512
    earth_ylim_max = 512

    # Two-point axis geometry
    anchor_lon_deg = -30.0
    anchor_lat_deg = 19.0

    # Nikou 2025 に厳密に寄せるなら fit_epsilon=False, epsilon_fixed=0.0
    fit_epsilon = False
    epsilon_fixed = 0.0

    initial_guess = {
        "apex_lon_deg": -45.0,
        "apex_lat_deg": 17.0,
        "apex_r_rsun": 5.13,
        "kappa": 0.40,
        "epsilon": -0.30,
    }

    bounds = {
        "apex_lon_deg": (-140.0, 20.0),
        "apex_lat_deg": (-80.0, 80.0),
        "apex_r_rsun": (1.2, 10.0),
        "kappa": (0.08, 0.80),
        "epsilon": (-0.95, 0.95),
    }

    envelope_bin_deg = 1.0
    click_sigma_px = 6.0
    pa_edge_sigma_deg = 4.0
    model_to_obs_weight = 1.0
    reverse_front_min_points = 32
    equalize_view_weights = True

    n_meridians = 12
    n_parallels = 7
    n_line_pts = 240

    n_starts = 24
    rng_seed = 42

    out_dir = "/mnt/d/wsl/home/kinno-7010/Research_data/GCS/output"

    run_spheroid_mpfit_gui(
        target_time_str=target_time_str,
        earth_delta_time_min=earth_delta_time_min,
        cor1_base_minutes_before=cor1_base_minutes_before,
        euvi_dt_minutes=euvi_dt_minutes,
        earth_xlim_min=earth_xlim_min,
        earth_xlim_max=earth_xlim_max,
        earth_ylim_min=earth_ylim_min,
        earth_ylim_max=earth_ylim_max,
        mk4_inner=mk4_inner,
        mk4_outer_lasco_inner=mk4_outer_lasco_inner,
        lasco_outer=lasco_outer,
        euvi_outer_rsun=euvi_outer_rsun,
        cor1_outer_rsun=cor1_outer_rsun,
        anchor_lon_deg=anchor_lon_deg,
        anchor_lat_deg=anchor_lat_deg,
        initial_guess=initial_guess,
        bounds=bounds,
        fit_epsilon=fit_epsilon,
        epsilon_fixed=epsilon_fixed,
        envelope_bin_deg=envelope_bin_deg,
        click_sigma_px=click_sigma_px,
        pa_edge_sigma_deg=pa_edge_sigma_deg,
        model_to_obs_weight=model_to_obs_weight,
        reverse_front_min_points=reverse_front_min_points,
        equalize_view_weights=equalize_view_weights,
        n_meridians=n_meridians,
        n_parallels=n_parallels,
        n_line_pts=n_line_pts,
        n_starts=n_starts,
        rng_seed=rng_seed,
        mpfit_module_path=mpfit_module_path,
        out_dir=out_dir,
    )