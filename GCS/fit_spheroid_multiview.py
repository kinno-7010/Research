#!/usr/bin/env python3
"""
Interactive multiview spheroid fitting for CME/shock fronts.

This script is designed to *reuse* the user's existing plotting code:
- integrated_analysis.create_single_diff_from_time_image  (Earth-view K-COR + LASCO-C2)
- cor1_diff_plot.create_cor1_difference_plot              (STEREO-A / SECCHI COR1)
- plot_spheroid_C2 geometry / projection helpers          (3D spheroid model)

Main idea
---------
1. Build a synchronized 2-panel GUI (Earth-view composite + COR1).
2. Click leading-edge points on both panels.
3. Fit a 3D spheroid with an MPFIT-style residual formulation.
4. Reproject the best-fit spheroid onto both panels and save PNG + JSON.

Why this is better than two separate sequential GUIs
----------------------------------------------------
Keeping the two viewpoints visible in one window reduces operator drift and
makes it much easier to notice inconsistent front picking across viewpoints.

Optimizer backend
-----------------
- backend='mpfit' : uses the historical Python port of MPFIT if available.
- backend='scipy' : uses scipy.optimize.least_squares.
- backend='auto'  : tries MPFIT first, then falls back to SciPy.

Important modeling choice
-------------------------
The existing spheroid implementation in plot_spheroid_C2.py is expressed in
Heliographic Stonyhurst (HGS) and *then* transformed to the observer frame of
reference_map. Therefore, the correct way to obtain the COR1 view is **not** to
manually subtract the STEREO-A separation angle from Earth-view parameters.
Instead, keep one intrinsic 3D parameter set in HGS and project it through the
COR1 map's observer geometry.

Default fit strategy
--------------------
By default, the axis-footpoint (anchor_lon, anchor_lat) is kept fixed and only
5 parameters are fitted:
    apex_lon, apex_lat, apex_r, kappa, epsilon
This is usually more stable than fitting the anchor as well, because the front
alone often underconstrains the photospheric footpoint.

Usage example
-------------
python fit_spheroid_multiview_gui.py \
    --target-time 2022-06-13T03:46:36 \
    --anchor-lon -30 --anchor-lat 19 \
    --apex-lon -45 --apex-lat 17 --apex-r 5.13 \
    --kappa 0.40 --epsilon 0.62 \
    --backend auto

Mouse / keyboard
----------------
- Left click   : add a point on the active panel
- Right click  : undo the last point on the active panel
- c            : clear points on the active panel
- d            : run fit
- s            : save current figure + JSON
- q            : quit
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("/home/kinno-7010/Research_code/GCS/astrolibpy")
import mpfit

# -----------------------------------------------------------------------------
# Path configuration
# -----------------------------------------------------------------------------
DEFAULT_INTEGRATED_ANALYSIS_PATHS = [
    "/home/kinno-7010/Research_code/SDO_Mk4_SOHO/py_folder/integrated_analysis.py",
    "/mnt/data/integrated_analysis.py",
]
DEFAULT_COR1_DIFF_PLOT_PATHS = [
    "/home/kinno-7010/Research_code/STEREO-A/SECCHI/STEREO_integrated_plot.py",
    "/mnt/data/STEREO_integrated_plot.py",
]
DEFAULT_PLOT_SPHEROID_C2_PATHS = [
    "/home/kinno-7010/Research_code/GCS/plot_spheroid_C2.py",
    "/mnt/data/plot_spheroid_C2.py",
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _find_first_existing(paths: list[str]) -> Path:
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    raise FileNotFoundError("None of the candidate paths exists:\n  " + "\n  ".join(paths))


def _load_module_from_file(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _safe_float(x: Any) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    
def _map_params(m) -> dict:
    px_per_rsun = m.rsun_obs.to_value(mods_global.sph.u.arcsec) / abs(m.scale.axis1.to_value(mods_global.sph.u.arcsec / mods_global.sph.u.pix))
    sun_center = mods_global.sph.SkyCoord(0 * mods_global.sph.u.arcsec, 0 * mods_global.sph.u.arcsec, frame=m.coordinate_frame)
    sun_center_pix = m.world_to_pixel(sun_center)
    return {
        "nx": m.data.shape[1],
        "ny": m.data.shape[0],
        "cx": float(sun_center_pix.x.value),
        "cy": float(sun_center_pix.y.value),
        "px_per_rsun": float(px_per_rsun),
    }

def _map_pixels_between_param_grids(x: np.ndarray, y: np.ndarray, src_params: dict, dst_params: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_norm = (x - float(src_params["cx"])) / float(src_params["px_per_rsun"])
    y_norm = (y - float(src_params["cy"])) / float(src_params["px_per_rsun"])
    x_dst = x_norm * float(dst_params["px_per_rsun"]) + float(dst_params["cx"])
    y_dst = y_norm * float(dst_params["px_per_rsun"]) + float(dst_params["cy"])
    return np.asarray(x_dst, dtype=float), np.asarray(y_dst, dtype=float)

def _xy_to_pa_rho_with_params(x: np.ndarray, y: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x - float(params["cx"])
    dy = y - float(params["cy"])
    rho = np.sqrt(dx * dx + dy * dy) / float(params["px_per_rsun"])
    pa = pa_from_xy(dx, dy)
    return np.asarray(pa, dtype=float), np.asarray(rho, dtype=float)

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
    sigma_px: float,
    reverse_weight: float = 1.0,
    reverse_min_points: int = 32,
) -> np.ndarray:
    obs_xy = np.asarray(obs_xy, dtype=float)
    model_xy = np.asarray(model_xy, dtype=float)
    if obs_xy.ndim != 2 or model_xy.ndim != 2 or obs_xy.shape[0] == 0 or model_xy.shape[0] == 0:
        n = max(int(obs_xy.shape[0]) if obs_xy.ndim == 2 else 0, 1)
        return np.full(n, 1.0e3 / max(float(sigma_px), 1.0), dtype=float)

    d_obs = _nearest_distances_xy(obs_xy, model_xy) / max(float(sigma_px), 1.0)
    d_obs = d_obs / max(np.sqrt(obs_xy.shape[0]), 1.0)

    n_rev = max(int(reverse_min_points), int(obs_xy.shape[0]))
    model_xy_rs = _resample_polyline_by_arclength(model_xy, n_rev)
    d_mod = _nearest_distances_xy(model_xy_rs, obs_xy) / max(float(sigma_px), 1.0)
    d_mod = d_mod / max(np.sqrt(model_xy_rs.shape[0]), 1.0)

    return np.concatenate([d_obs.astype(float), float(reverse_weight) * d_mod.astype(float)])

def pa_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Position angle in degrees, measured CCW from solar north."""
    pa = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return np.asarray(pa, dtype=float)


def angular_distance_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((np.asarray(a) - np.asarray(b) + 180.0) % 360.0) - 180.0


def unwrap_angles_with_largest_gap(angles_deg: np.ndarray, boundary: float | None = None) -> tuple[np.ndarray, float]:
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
    """Construct an outer-envelope rho(PA) using max(rho) in PA bins."""
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


def interp_on_common_grid(
    pa_a: np.ndarray,
    rho_a: np.ndarray,
    pa_b: np.ndarray,
    rho_b: np.ndarray,
    *,
    grid_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(pa_a) < 2 or len(pa_b) < 2:
        return np.array([]), np.array([]), np.array([])

    lo = max(float(np.min(pa_a)), float(np.min(pa_b)))
    hi = min(float(np.max(pa_a)), float(np.max(pa_b)))
    if hi - lo < grid_deg:
        return np.array([]), np.array([]), np.array([])

    grid = np.arange(lo, hi + 0.5 * grid_deg, grid_deg, dtype=float)
    if grid.size < 2:
        return np.array([]), np.array([]), np.array([])

    rho_a_i = np.interp(grid, pa_a, rho_a)
    rho_b_i = np.interp(grid, pa_b, rho_b)
    return grid, rho_a_i, rho_b_i




def ensure_interactive_backend() -> None:
    backend = plt.get_backend().lower()
    if backend != "agg":
        return

    # integrated_analysis.py forces Agg for batch processing, which is fine for
    # movie generation but not for point-click GUIs. Try to recover an
    # interactive backend here.
    for cand in ("QtAgg", "TkAgg", "Qt5Agg"):
        try:
            plt.switch_backend(cand)
            print(f"[INFO] Switched matplotlib backend from Agg to {cand} for GUI use.")
            return
        except Exception:
            continue

    print("[WARN] matplotlib backend is still Agg. GUI clicking will not work until an interactive backend is available.")

def sample_spheroid_surface_hpc(
    mods: ModuleBundle,
    reference_map,
    spheroid_params,
    *,
    dense_for_fit: bool = False,
):
    params_use = spheroid_params
    if dense_for_fit:
        n_alpha = max(int(getattr(mods_global_config, "surface_n_alpha", 91)), 91) if 'mods_global_config' in globals() else 91
        n_beta = max(int(getattr(mods_global_config, "surface_n_beta", 361)), 361) if 'mods_global_config' in globals() else 361
    else:
        n_alpha = 61
        n_beta = 181

    geom = mods.sph._spheroid_axis_geometry_rsun(params_use)
    center = np.asarray(geom["center"], dtype=float).reshape(3, 1, 1)
    axis_u = np.asarray(geom["axis_u"], dtype=float).reshape(3, 1, 1)
    e1 = np.asarray(geom["e1"], dtype=float).reshape(3, 1, 1)
    e2 = np.asarray(geom["e2"], dtype=float).reshape(3, 1, 1)
    a = float(params_use.a_rsun)
    b = float(params_use.b_rsun)

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

    rep = mods.sph.CartesianRepresentation(cart[0] * mods.sph.u.R_sun, cart[1] * mods.sph.u.R_sun, cart[2] * mods.sph.u.R_sun)
    coords_hgs = mods.sph.SkyCoord(rep, frame=mods.sph.sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)
    mask_vis = mods.sph._visible_mask(coords_hgs, reference_map, only_visible=params_use.only_visible)
    mask = mask_surface & mask_vis
    if not np.any(mask) and params_use.only_visible:
        mask = mask_surface
    if not np.any(mask):
        return mods.sph.SkyCoord([], [], unit=mods.sph.u.arcsec, frame=reference_map.coordinate_frame)

    coords_hpc = coords_hgs.transform_to(reference_map.coordinate_frame)
    return coords_hpc[mask]

def covariance_from_jacobian(jac: np.ndarray, cost: float, n_resid: int, n_free: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    if jac is None:
        return None, None
    jac = np.asarray(jac, dtype=float)
    if jac.ndim != 2 or jac.shape[0] <= jac.shape[1]:
        return None, None

    dof = n_resid - n_free
    if dof <= 0:
        return None, None

    _, s, vt = np.linalg.svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    good = s > threshold
    if not np.any(good):
        return None, None

    v = vt.T
    cov = (v[:, good] / (s[good] ** 2)) @ v[:, good].T
    s_sq = 2.0 * float(cost) / dof
    cov *= s_sq
    perr = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return cov, perr


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class ModuleBundle:
    ia: ModuleType
    cor1: ModuleType
    sph: ModuleType


@dataclass
class FitParameterSpec:
    name: str
    value: float
    lower: float
    upper: float
    fixed: bool = False


@dataclass
class FitResult:
    backend: str
    success: bool
    status: int
    message: str
    params: dict[str, float]
    errors: dict[str, float | None]
    chi2: float | None
    dof: int | None
    nfev: int | None
    covar: list[list[float]] | None


@dataclass
class GUIConfig:
    target_time: str
    earth_base_minutes: int = 2
    cor1_base_minutes: int = 10
    cor1_plot_radius_rsun: float = 4.0
    pa_bin_deg: float = 5.0
    envelope_bin_deg: float = 1.0
    max_multistarts: int = 12
    global_search_samples: int = 40
    global_search_keep: int = 6
    random_seed: int = 0
    backend: str = "auto"
    fit_anchor: bool = False
    fit_epsilon: bool = False
    click_sigma_px: float = 6.0
    pa_edge_sigma_deg: float = 4.0
    model_to_obs_weight: float = 1.0
    reverse_front_min_points: int = 32
    surface_n_alpha: int = 91
    surface_n_beta: int = 361
    output_dir: str = "/mnt/d/wsl/home/kinno-7010/Research/GCS/output"

    anchor_lon_deg: float = -30.0
    anchor_lat_deg: float = 19.0
    apex_lon_deg: float = -45.0
    apex_lat_deg: float = 17.0
    apex_r_rsun: float = 5.13
    kappa: float = 0.40
    epsilon: float = -0.30

    only_above_surface: bool = True
    only_visible: bool = True
    n_meridians: int = 12
    n_parallels: int = 7
    n_line_pts: int = 240
    

# -----------------------------------------------------------------------------
# Module loading / geometry access
# -----------------------------------------------------------------------------

def load_modules(
    integrated_analysis_path: Path | None = None,
    cor1_diff_plot_path: Path | None = None,
    plot_spheroid_c2_path: Path | None = None,
) -> ModuleBundle:
    integrated_analysis_path = integrated_analysis_path or _find_first_existing(DEFAULT_INTEGRATED_ANALYSIS_PATHS)
    cor1_diff_plot_path = cor1_diff_plot_path or _find_first_existing(DEFAULT_COR1_DIFF_PLOT_PATHS)
    plot_spheroid_c2_path = plot_spheroid_c2_path or _find_first_existing(DEFAULT_PLOT_SPHEROID_C2_PATHS)

    # Make sure the original module directories are importable.
    sys.path.insert(0, str(integrated_analysis_path.parent))
    sys.path.insert(0, str(cor1_diff_plot_path.parent))
    sys.path.insert(0, str(plot_spheroid_c2_path.parent))

    ia = _load_module_from_file("user_integrated_analysis", integrated_analysis_path)
    cor1 = _load_module_from_file("user_cor1_diff_plot", cor1_diff_plot_path)
    sph = _load_module_from_file("user_plot_spheroid_C2", plot_spheroid_c2_path)
    return ModuleBundle(ia=ia, cor1=cor1, sph=sph)


def build_spheroid_params(mods: ModuleBundle, config: GUIConfig, fitted: dict[str, float] | None = None):
    p = {
        "anchor_lon_deg": config.anchor_lon_deg,
        "anchor_lat_deg": config.anchor_lat_deg,
        "apex_lon_deg": config.apex_lon_deg,
        "apex_lat_deg": config.apex_lat_deg,
        "apex_r_rsun": config.apex_r_rsun,
        "kappa": config.kappa,
        "epsilon": config.epsilon,
    }
    if fitted is not None:
        p.update(fitted)

    return mods.sph.SpheroidDome3DParams(
        kappa=float(p["kappa"]),
        epsilon=float(p["epsilon"]),
        anchor_lon_deg=float(p["anchor_lon_deg"]),
        anchor_lat_deg=float(p["anchor_lat_deg"]),
        apex_lon_deg=float(p["apex_lon_deg"]),
        apex_lat_deg=float(p["apex_lat_deg"]),
        apex_r_rsun=float(p["apex_r_rsun"]),
        n_meridians=int(config.n_meridians),
        n_parallels=int(config.n_parallels),
        n_line_pts=int(config.n_line_pts),
        only_above_surface=bool(config.only_above_surface),
        only_visible=bool(config.only_visible),
    )


# -----------------------------------------------------------------------------
# Panel contexts
# -----------------------------------------------------------------------------

class BasePanelContext:
    def __init__(self, name: str, ax):
        self.name = name
        self.ax = ax
        self.click_xy: list[tuple[float, float]] = []
        self.click_artist = None
        self.fit_artists: list[Any] = []
        self.ax.format_coord = self._safe_format_coord

    @staticmethod
    def _safe_format_coord(x: float, y: float) -> str:
        if not np.isfinite(x) or not np.isfinite(y):
            return "x=nan, y=nan"
        return f"x={x:.2f}, y={y:.2f}"

    def add_point(self, x: float, y: float) -> None:
        self.click_xy.append((float(x), float(y)))
        self.redraw_clicks()

    def undo_last(self) -> None:
        if self.click_xy:
            self.click_xy.pop()
            self.redraw_clicks()

    def clear_points(self) -> None:
        self.click_xy = []
        self.redraw_clicks()

    def clicked_xy_array(self) -> np.ndarray:
        if not self.click_xy:
            return np.empty((0, 2), dtype=float)
        return np.asarray(self.click_xy, dtype=float)

    def redraw_clicks(self) -> None:
        if self.click_artist is not None:
            try:
                self.click_artist.remove()
            except Exception:
                pass
            self.click_artist = None

        if not self.click_xy:
            self.ax.figure.canvas.draw_idle()
            return

        xy = self.clicked_xy_array()
        self.click_artist = self.ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=28,
            c="magenta",
            edgecolors="black",
            linewidths=0.6,
            zorder=40,
            label=f"clicked front ({self.name})",
        )
        self.ax.figure.canvas.draw_idle()

    def clear_fit_artists(self) -> None:
        for art in self.fit_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.fit_artists = []

    def radial_sigma_rsun(self) -> float:
        raise NotImplementedError

    def clicked_pa_rho(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def model_pa_rho(self, mods: ModuleBundle, spheroid_params) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def model_front_plot_xy(
        self,
        mods: ModuleBundle,
        spheroid_params,
        *,
        front_bin_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def overlay_fit(self, mods: ModuleBundle, spheroid_params, config: GUIConfig) -> None:
        raise NotImplementedError

    def serialize_clicks(self) -> dict[str, Any]:
        pa, rho = self.clicked_pa_rho()
        return {
            "panel": self.name,
            "n_points": len(self.click_xy),
            "click_xy": self.click_xy,
            "click_pa_deg": pa.tolist(),
            "click_rho_rsun": rho.tolist(),
        }

class EarthCompositePanel(BasePanelContext):
    def __init__(self, ax, params_lasco: dict, lasco_map):
        super().__init__("earth", ax)
        self.params_lasco = params_lasco
        self.lasco_map = lasco_map
        self.px_per_rsun = float(params_lasco["px_per_rsun"])

    def radial_sigma_rsun(self) -> float:
        return max(1.0 / self.px_per_rsun, 0.02)

    def clicked_pa_rho(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.click_xy:
            return np.array([]), np.array([])
        xy = self.clicked_xy_array()
        x = xy[:, 0]
        y = xy[:, 1]
        rho = np.sqrt(x * x + y * y) / self.px_per_rsun
        pa = pa_from_xy(x, y)
        return pa, rho

    def _surface_xy_and_pa_rho(self, mods: ModuleBundle, spheroid_params, *, dense_for_fit: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        coords_hpc = sample_spheroid_surface_hpc(
            mods,
            self.lasco_map,
            spheroid_params,
            dense_for_fit=dense_for_fit,
        )
        if len(coords_hpc) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        x_px, y_px = mods.sph._hpc_to_rel_pix(
            coords_hpc,
            float(self.lasco_map.rsun_obs.to_value(mods.sph.u.arcsec)),
            self.px_per_rsun,
        )
        x_px = np.asarray(x_px, dtype=float)
        y_px = np.asarray(y_px, dtype=float)
        rho = np.sqrt(x_px * x_px + y_px * y_px) / self.px_per_rsun
        pa = pa_from_xy(x_px, y_px)
        return x_px, y_px, pa, rho

    def model_pa_rho(self, mods: ModuleBundle, spheroid_params) -> tuple[np.ndarray, np.ndarray]:
        _, _, pa, rho = self._surface_xy_and_pa_rho(mods, spheroid_params, dense_for_fit=True)
        return pa, rho

    def model_front_plot_xy(
        self,
        mods: ModuleBundle,
        spheroid_params,
        *,
        front_bin_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        _, _, pa_mod, rho_mod = self._surface_xy_and_pa_rho(mods, spheroid_params, dense_for_fit=True)
        if len(pa_mod) < 2:
            return np.array([]), np.array([])

        pa_front, rho_front, _ = envelope_from_pa_rho(
            pa_mod,
            rho_mod,
            bin_deg=front_bin_deg,
            boundary=None,
        )
        if len(pa_front) == 0:
            return np.array([]), np.array([])

        pa_rad = np.deg2rad(pa_front)
        x = rho_front * self.px_per_rsun * np.sin(pa_rad)
        y = rho_front * self.px_per_rsun * np.cos(pa_rad)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def overlay_fit(self, mods: ModuleBundle, spheroid_params, config: GUIConfig) -> None:
        self.clear_fit_artists()
        mods.sph.overlay_spheroid_on_coronagraph_axes(
            self.ax,
            self.lasco_map,
            self.params_lasco,
            spheroid_params,
            color="#00FF00",
            verbose=False,
        )

        x_front, y_front = self.model_front_plot_xy(
            mods,
            spheroid_params,
            front_bin_deg=float(config.pa_bin_deg),
        )
        if len(x_front) > 0:
            art = self.ax.scatter(
                x_front,
                y_front,
                s=30,
                marker="*",
                c="blue",
                edgecolors="white",
                linewidths=0.6,
                zorder=41,
                label="fitted front points",
            )
            self.fit_artists.append(art)

        self.ax.figure.canvas.draw_idle()
        
        
class COR1Panel(BasePanelContext):
    def __init__(self, ax, display_map, raw_cor1_map=None):
        super().__init__("cor1", ax)
        self.display_map = display_map
        self.raw_cor1_map = raw_cor1_map
        self.source_map = raw_cor1_map if raw_cor1_map is not None else display_map
        self.u = mods_global.sph.u if 'mods_global' in globals() else None
        self.display_params = _map_params(self.display_map)
        self.source_params = _map_params(self.source_map)

    def _map_scale_arcsec_per_pix(self) -> float:
        u = self.display_map.scale[0].unit
        s0 = abs(_safe_float(self.display_map.scale[0].to_value(u)))
        s1 = abs(_safe_float(self.display_map.scale[1].to_value(u)))
        return 0.5 * (s0 + s1)
    
    def _rsun_arcsec(self) -> float:
        return float(self.display_map.rsun_obs.to_value(mods_global.sph.u.arcsec))

    def _rsun_pix(self) -> float:
        return float(self.display_params["px_per_rsun"])

    def radial_sigma_rsun(self) -> float:
        return max(1.0 / self._rsun_pix(), 0.02)

    def clicked_pa_rho(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.click_xy:
            return np.array([]), np.array([])
        xy = self.clicked_xy_array()
        return _xy_to_pa_rho_with_params(xy[:, 0], xy[:, 1], self.display_params)

    def _surface_xy_and_pa_rho(self, mods: ModuleBundle, spheroid_params, *, dense_for_fit: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        coords_hpc = sample_spheroid_surface_hpc(
            mods,
            self.source_map,
            spheroid_params,
            dense_for_fit=dense_for_fit,
        )
        if len(coords_hpc) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        pix = self.source_map.world_to_pixel(coords_hpc)
        x_src = np.asarray(pix.x.value, dtype=float)
        y_src = np.asarray(pix.y.value, dtype=float)
        x, y = _map_pixels_between_param_grids(x_src, y_src, self.source_params, self.display_params)
        pa, rho = _xy_to_pa_rho_with_params(x, y, self.display_params)
        return x, y, pa, rho
    
    def model_pa_rho(self, mods: ModuleBundle, spheroid_params) -> tuple[np.ndarray, np.ndarray]:
        _, _, pa, rho = self._surface_xy_and_pa_rho(mods, spheroid_params, dense_for_fit=True)
        return pa, rho

    def model_front_plot_xy(
        self,
        mods: ModuleBundle,
        spheroid_params,
        *,
        front_bin_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_mod, y_mod, pa_mod, rho_mod = self._surface_xy_and_pa_rho(mods, spheroid_params, dense_for_fit=True)
        if len(pa_mod) < 2:
            return np.array([]), np.array([])

        pa_front, rho_front, boundary = envelope_from_pa_rho(
            pa_mod,
            rho_mod,
            bin_deg=front_bin_deg,
            boundary=None,
        )
        if len(pa_front) == 0:
            return np.array([]), np.array([])

        # keep displayed front on the same synthetic canvas used by the integrated image
        pa_u, _ = unwrap_angles_with_largest_gap(pa_mod, boundary=boundary)
        order = np.argsort(pa_u)
        xy_sorted = np.column_stack([x_mod[order], y_mod[order]])
        pa_sorted = pa_u[order]
        x_front = np.interp(pa_front, pa_sorted, xy_sorted[:, 0])
        y_front = np.interp(pa_front, pa_sorted, xy_sorted[:, 1])
        return np.asarray(x_front, dtype=float), np.asarray(y_front, dtype=float)
    
    def overlay_fit(self, mods: ModuleBundle, spheroid_params, config: GUIConfig) -> None:
        self.clear_fit_artists()
        self.fit_artists.extend(
            overlay_spheroid_on_cor1_axes(
                mods,
                self.ax,
                self.display_map,
                spheroid_params,
                color="#00FF00",
                source_map=self.source_map,
                source_params=self.source_params,
                display_params=self.display_params,
            )
        )

        x_front, y_front = self.model_front_plot_xy(
            mods,
            spheroid_params,
            front_bin_deg=float(config.pa_bin_deg),
        )
        if len(x_front) > 0:
            art = self.ax.scatter(
                x_front,
                y_front,
                s=80,
                marker="*",
                c="blue",
                edgecolors="white",
                linewidths=0.6,
                zorder=41,
                label="fitted front points",
            )
            self.fit_artists.append(art)

        self.ax.figure.canvas.draw_idle()# -----------------------------------------------------------------------------
# Projection helpers
# -----------------------------------------------------------------------------

def projected_model_points_hpc(
    mods: ModuleBundle,
    reference_map,
    spheroid_params,
    *,
    include_footprint: bool = True,
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

    lines = mods.sph.sample_spheroid_dome_wireframe_hpc(params_use, reference_map)
    if len(lines) == 0 and getattr(params_use, "only_visible", False):
        params_use = replace(params_use, only_visible=False)
        lines = mods.sph.sample_spheroid_dome_wireframe_hpc(params_use, reference_map)

    coords = []
    for seg in list(lines):
        if len(seg) > 0:
            coords.append(seg)

    if include_footprint:
        fps = mods.sph.sample_spheroid_footprint_hpc(params_use, reference_map)
        for seg in list(fps):
            if len(seg) > 0:
                coords.append(seg)

    return coords

# -----------------------------------------------------------------------------
# COR1 overlay helper using existing spheroid geometry samplers
# -----------------------------------------------------------------------------

def overlay_spheroid_on_cor1_axes(
    mods: ModuleBundle,
    ax,
    diff_map,
    spheroid_params,
    *,
    color="#00FF00",
    source_map=None,
    source_params: dict | None = None,
    display_params: dict | None = None,
) -> list[Any]:
    artists: list[Any] = []

    if source_map is None:
        source_map = diff_map
    if source_params is None:
        source_params = _map_params(source_map)
    if display_params is None:
        display_params = _map_params(diff_map)

    wire_lines_hpc = mods.sph.sample_spheroid_dome_wireframe_hpc(spheroid_params, source_map)
    if (len(wire_lines_hpc) == 0) and getattr(spheroid_params, "only_visible", False):
        tmp = replace(spheroid_params, only_visible=False)
        wire_lines_hpc = mods.sph.sample_spheroid_dome_wireframe_hpc(tmp, source_map)

    for ln in wire_lines_hpc:
        pix = source_map.world_to_pixel(ln)
        x_src = np.asarray(pix.x.value, dtype=float)
        y_src = np.asarray(pix.y.value, dtype=float)
        x_disp, y_disp = _map_pixels_between_param_grids(x_src, y_src, source_params, display_params)
        art = ax.plot(
            x_disp,
            y_disp,
            color=color,
            linewidth=1.0,
            alpha=0.85,
            zorder=25,
        )
        artists.extend(art)

    footprint_lines_hpc = mods.sph.sample_spheroid_footprint_hpc(spheroid_params, source_map)
    for fp in footprint_lines_hpc:
        pix = source_map.world_to_pixel(fp)
        x_src = np.asarray(pix.x.value, dtype=float)
        y_src = np.asarray(pix.y.value, dtype=float)
        x_disp, y_disp = _map_pixels_between_param_grids(x_src, y_src, source_params, display_params)
        art = ax.plot(
            x_disp,
            y_disp,
            color=color,
            linewidth=2.2,
            alpha=0.95,
            zorder=26,
        )
        artists.extend(art)

    try:
        apex_hpc = mods.sph.spheroid_dome_apex_hpc(spheroid_params, source_map)
        apex_pix = source_map.world_to_pixel(apex_hpc)
        x_disp, y_disp = _map_pixels_between_param_grids(
            np.asarray([_safe_float(apex_pix.x.value)], dtype=float),
            np.asarray([_safe_float(apex_pix.y.value)], dtype=float),
            source_params,
            display_params,
        )
        art = ax.plot(
            [float(x_disp[0])],
            [float(y_disp[0])],
            marker="o",
            linestyle="None",
            markerfacecolor="orange",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=10,
            zorder=30,
            label=(
                f"COR1 projected spheroid apex (κ={spheroid_params.kappa:.3f}, "
                f"ε={spheroid_params.epsilon:.3f}, r={spheroid_params.apex_r_rsun:.3f} R⊙)"
            ),
        )
        artists.extend(art)
    except Exception as exc:
        print(f"[WARN] COR1 apex marker skipped: {exc}")

    try:
        anchor_hpc = mods.sph.spheroid_axis_footpoint_hpc(spheroid_params, source_map)
        anchor_pix = source_map.world_to_pixel(anchor_hpc)
        x_disp, y_disp = _map_pixels_between_param_grids(
            np.asarray([_safe_float(anchor_pix.x.value)], dtype=float),
            np.asarray([_safe_float(anchor_pix.y.value)], dtype=float),
            source_params,
            display_params,
        )
        art = ax.plot(
            [float(x_disp[0])],
            [float(y_disp[0])],
            marker="*",
            linestyle="None",
            markerfacecolor="yellow",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=18,
            zorder=31,
            label=(
                f"axis surface intersection (lon,lat)=({float(spheroid_params.anchor_lon_deg):.1f},"
                f"{float(spheroid_params.anchor_lat_deg):.1f})°"
            ),
        )
        artists.extend(art)
    except Exception as exc:
        print(f"[WARN] COR1 anchor marker skipped: {exc}")

    return artists

# -----------------------------------------------------------------------------
# Residual construction
# -----------------------------------------------------------------------------

def make_parameter_specs(
    config: GUIConfig,
    earth_panel: EarthCompositePanel | None = None,
    cor1_panel: COR1Panel | None = None,
) -> list[FitParameterSpec]:
    def _window_bounds(center: float, half_width: float, hard_lo: float, hard_hi: float) -> tuple[float, float]:
        lo = max(hard_lo, float(center) - float(half_width))
        hi = min(hard_hi, float(center) + float(half_width))
        if lo >= hi:
            lo, hi = hard_lo, hard_hi
        return lo, hi

    rho_click = []
    for panel in (earth_panel, cor1_panel):
        if panel is None:
            continue
        _, rho = panel.clicked_pa_rho()
        if len(rho) > 0:
            rho_click.append(np.asarray(rho, dtype=float))

    if rho_click:
        rho_all = np.concatenate(rho_click)
        rho_max = float(np.nanmax(rho_all))
        rho_med = float(np.nanmedian(rho_all))
        r_lo = max(1.10, min(config.apex_r_rsun, rho_med) - 1.2)
        r_hi = min(8.0, max(config.apex_r_rsun + 0.6, rho_max + 1.0))
        if r_lo >= r_hi:
            r_lo, r_hi = 1.10, min(8.0, max(config.apex_r_rsun + 1.0, 6.0))
    else:
        r_lo, r_hi = max(1.10, config.apex_r_rsun - 1.0), min(8.0, config.apex_r_rsun + 1.5)

    apex_lon_lo, apex_lon_hi = _window_bounds(config.apex_lon_deg, 35.0, -180.0, 180.0)
    apex_lat_lo, apex_lat_hi = _window_bounds(config.apex_lat_deg, 25.0, -80.0, 80.0)
    kappa_lo, kappa_hi = _window_bounds(config.kappa, 0.25, 0.08, 1.0)

    specs: list[FitParameterSpec] = []
    if config.fit_anchor:
        a_lon_lo, a_lon_hi = _window_bounds(config.anchor_lon_deg, 10.0, -180.0, 180.0)
        a_lat_lo, a_lat_hi = _window_bounds(config.anchor_lat_deg, 8.0, -80.0, 80.0)
        specs.extend([
            FitParameterSpec("anchor_lon_deg", config.anchor_lon_deg, a_lon_lo, a_lon_hi, fixed=False),
            FitParameterSpec("anchor_lat_deg", config.anchor_lat_deg, a_lat_lo, a_lat_hi, fixed=False),
        ])
    else:
        specs.extend([
            FitParameterSpec("anchor_lon_deg", config.anchor_lon_deg, config.anchor_lon_deg, config.anchor_lon_deg, fixed=True),
            FitParameterSpec("anchor_lat_deg", config.anchor_lat_deg, config.anchor_lat_deg, config.anchor_lat_deg, fixed=True),
        ])

    specs.extend([
        FitParameterSpec("apex_lon_deg", config.apex_lon_deg, apex_lon_lo, apex_lon_hi, fixed=False),
        FitParameterSpec("apex_lat_deg", config.apex_lat_deg, apex_lat_lo, apex_lat_hi, fixed=False),
        FitParameterSpec("apex_r_rsun", config.apex_r_rsun, r_lo, r_hi, fixed=False),
        FitParameterSpec("kappa", config.kappa, kappa_lo, kappa_hi, fixed=False),
    ])

    if config.fit_epsilon:
        eps_lo, eps_hi = _window_bounds(config.epsilon, 0.25, -0.95, 0.95)
        specs.append(FitParameterSpec("epsilon", config.epsilon, eps_lo, eps_hi, fixed=False))
    else:
        specs.append(FitParameterSpec("epsilon", config.epsilon, config.epsilon, config.epsilon, fixed=True))

    return specs

def vector_from_specs(specs: list[FitParameterSpec]) -> np.ndarray:
    return np.array([sp.value for sp in specs if not sp.fixed], dtype=float)


def apply_free_vector_to_specs(specs: list[FitParameterSpec], free_values: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    j = 0
    for sp in specs:
        if sp.fixed:
            out[sp.name] = float(sp.value)
        else:
            out[sp.name] = float(free_values[j])
            j += 1
    return out


def bounds_from_specs(specs: list[FitParameterSpec]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([sp.lower for sp in specs if not sp.fixed], dtype=float)
    hi = np.array([sp.upper for sp in specs if not sp.fixed], dtype=float)
    return lo, hi


def residual_vector(
    free_values: np.ndarray,
    *,
    specs: list[FitParameterSpec],
    mods: ModuleBundle,
    config: GUIConfig,
    earth_panel: EarthCompositePanel,
    cor1_panel: COR1Panel,
) -> np.ndarray:
    param_dict = apply_free_vector_to_specs(specs, free_values)

    try:
        spheroid = build_spheroid_params(mods, config, fitted=param_dict)
    except Exception:
        return np.full(64, 200.0, dtype=float)

    panel_residuals: list[np.ndarray] = []

    for panel in (earth_panel, cor1_panel):
        obs_xy = panel.clicked_xy_array()
        if obs_xy.shape[0] < 3:
            continue

        mod_x, mod_y = panel.model_front_plot_xy(
            mods,
            spheroid,
            front_bin_deg=float(config.envelope_bin_deg),
        )
        if len(mod_x) < 5:
            nrev = max(int(getattr(config, "reverse_front_min_points", 32)), int(obs_xy.shape[0]))
            panel_residuals.append(np.full(obs_xy.shape[0] + nrev + 4, 120.0, dtype=float))
            continue

        mod_xy = np.column_stack([mod_x, mod_y])
        sigma_px = max(float(config.click_sigma_px), 1.0)
        d_sym = _symmetric_front_distance_residuals(
            obs_xy,
            mod_xy,
            sigma_px=sigma_px,
            reverse_weight=float(getattr(config, "model_to_obs_weight", 1.0)),
            reverse_min_points=int(getattr(config, "reverse_front_min_points", 32)),
        )

        pa_obs, rho_obs = panel.clicked_pa_rho()
        pa_mod, rho_mod = panel.model_pa_rho(mods, spheroid)
        span = np.full(4, 60.0, dtype=float)
        if len(pa_obs) >= 2 and len(pa_mod) >= 5:
            pa_obs_env, rho_obs_env, boundary = envelope_from_pa_rho(
                pa_obs,
                rho_obs,
                bin_deg=float(config.envelope_bin_deg),
                boundary=None,
            )
            pa_mod_env, rho_mod_env, _ = envelope_from_pa_rho(
                pa_mod,
                rho_mod,
                bin_deg=float(config.envelope_bin_deg),
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
                pa_sigma = max(float(config.pa_edge_sigma_deg), 1.0)
                span = np.array([
                    (mod_min - obs_min) / pa_sigma,
                    (mod_max - obs_max) / pa_sigma,
                    (mod_ctr - obs_ctr) / pa_sigma,
                    (mod_wid - obs_wid) / pa_sigma,
                ], dtype=float)

        panel_residuals.append(np.concatenate([d_sym.astype(float), span.astype(float)]))

    if not panel_residuals:
        return np.full(64, 200.0, dtype=float)

    return np.concatenate(panel_residuals)


# -----------------------------------------------------------------------------
# Optimizer backends
# -----------------------------------------------------------------------------

def try_mpfit_import():
    def _extract_mpfit_callable(obj):
        if obj is None:
            return None
        cand = getattr(obj, "mpfit", None)
        if callable(cand):
            return cand
        return None

    try:
        import importlib
        mod = importlib.import_module("mpfit")
        ctor = _extract_mpfit_callable(mod)
        if ctor is not None:
            return ctor
    except Exception:
        pass

    try:
        import importlib
        submod = importlib.import_module("mpfit.mpfit")
        ctor = _extract_mpfit_callable(submod)
        if ctor is not None:
            return ctor
    except Exception:
        pass

    candidate_dirs: list[Path] = []

    # 1) script directory / current working directory
    try:
        candidate_dirs.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    candidate_dirs.append(Path.cwd())

    # 2) directories of the user modules already loaded by this script
    for attr in ("ia", "cor1", "sph"):
        try:
            mod = getattr(mods_global, attr)  # type: ignore[name-defined]
            mod_file = getattr(mod, "__file__", None)
            if mod_file:
                candidate_dirs.append(Path(mod_file).resolve().parent)
        except Exception:
            pass

    # 3) all directories currently visible from sys.path
    for p in sys.path:
        try:
            if p:
                candidate_dirs.append(Path(p).resolve())
        except Exception:
            continue

    seen: set[str] = set()
    ordered_dirs: list[Path] = []
    for d in candidate_dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            ordered_dirs.append(d)

    candidate_files: list[Path] = []
    for d in ordered_dirs:
        candidate_files.append(d / "mpfit.py")
        candidate_files.append(d / "mpfit" / "mpfit.py")
        candidate_files.append(d / "mpfit" / "__init__.py")

    load_index = 0
    for candidate in candidate_files:
        if not candidate.exists():
            continue
        try:
            module = _load_module_from_file(f"mpfit_local_{load_index}", candidate)
            load_index += 1
            ctor = _extract_mpfit_callable(module)
            if ctor is not None:
                return ctor
        except Exception:
            continue

    return None

def fit_with_mpfit(
    *,
    specs: list[FitParameterSpec],
    mods: ModuleBundle,
    config: GUIConfig,
    earth_panel: EarthCompositePanel,
    cor1_panel: COR1Panel,
) -> FitResult:
    mpfit_ctor = try_mpfit_import()
    if mpfit_ctor is None:
        raise ImportError(
            "Python MPFIT callable could not be imported. "
            "Place astrolibpy's mpfit.py in the script directory, as mpfit/mpfit.py on PYTHONPATH, or as an importable mpfit module."
        )

    p0 = vector_from_specs(specs)
    lo, hi = bounds_from_specs(specs)

    def mpfit_func(p, fjac=None, **kwargs):
        resid = residual_vector(
            np.asarray(p, dtype=float),
            specs=kwargs["specs"],
            mods=kwargs["mods"],
            config=kwargs["config"],
            earth_panel=kwargs["earth_panel"],
            cor1_panel=kwargs["cor1_panel"],
        )
        return [0, resid]

    def _step_for_param(name: str) -> float:
        return {
            "anchor_lon_deg": 0.3,
            "anchor_lat_deg": 0.3,
            "apex_lon_deg": 0.3,
            "apex_lat_deg": 0.3,
            "apex_r_rsun": 0.03,
            "kappa": 0.01,
            "epsilon": 0.02,
        }.get(name, 0.0)

    def build_parinfo(start: np.ndarray) -> list[dict[str, Any]]:
        parinfo: list[dict[str, Any]] = []
        j = 0
        for sp in specs:
            if sp.fixed:
                continue
            parinfo.append({
                "parname": sp.name,
                "value": float(start[j]),
                "fixed": 0,
                "limited": [1, 1],
                "limits": [float(sp.lower), float(sp.upper)],
                "step": float(_step_for_param(sp.name)),
                "mpside": 2,
            })
            j += 1
        return parinfo

    rng = np.random.default_rng(int(config.random_seed))
    starts: list[np.ndarray] = []
    if p0.size > 0:
        starts.append(np.asarray(p0, dtype=float))
        n_global = max(int(config.global_search_samples), 0)
        if n_global > 0:
            candidates = []
            candidates.append((float(np.sum(residual_vector(p0, specs=specs, mods=mods, config=config, earth_panel=earth_panel, cor1_panel=cor1_panel) ** 2)), np.asarray(p0, dtype=float)))
            for _ in range(n_global):
                trial = rng.uniform(lo, hi)
                cost = float(np.sum(residual_vector(trial, specs=specs, mods=mods, config=config, earth_panel=earth_panel, cor1_panel=cor1_panel) ** 2))
                candidates.append((cost, trial))
            candidates.sort(key=lambda t: t[0])
            for _, vec in candidates[:max(int(config.global_search_keep), 1)]:
                starts.append(np.asarray(vec, dtype=float))

        width = np.maximum(hi - lo, 1.0e-6)
        while len(starts) < max(int(config.max_multistarts), 1):
            base = starts[min(len(starts) - 1, max(int(config.global_search_keep), 1) - 1)] if len(starts) > 1 else p0
            jitter = rng.normal(0.0, 0.08, size=p0.size) * width
            starts.append(np.clip(base + jitter, lo, hi))

    unique_starts: list[np.ndarray] = []
    for st in starts[:max(int(config.max_multistarts), 1)]:
        if not any(np.allclose(st, uu, atol=1.0e-10, rtol=0.0) for uu in unique_starts):
            unique_starts.append(st)
    starts = unique_starts

    best_result = None
    best_cost = np.inf
    best_success = False

    for k, start in enumerate(starts):
        result = mpfit_ctor(
            mpfit_func,
            np.asarray(start, dtype=float),
            parinfo=build_parinfo(np.asarray(start, dtype=float)),
            functkw={
                "specs": specs,
                "mods": mods,
                "config": config,
                "earth_panel": earth_panel,
                "cor1_panel": cor1_panel,
            },
            quiet=1,
            nprint=0,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            maxiter=400,
            autoderivative=1,
        )

        if getattr(result, "params", None) is None:
            print(
                f"[INFO] multistart {k+1}/{len(starts)} status={getattr(result, 'status', None)} "
                f"cost=nan success=False"
            )
            continue

        status = int(getattr(result, "status", -999))
        success = bool(status > 0)
        cost = float(getattr(result, "fnorm", np.inf))
        print(f"[INFO] multistart {k+1}/{len(starts)} status={status} cost={cost:.6g} success={success}")

        if (best_result is None) or ((success and not best_success) or (success == best_success and cost < best_cost)):
            best_result = result
            best_cost = cost
            best_success = success

    if best_result is None or getattr(best_result, "params", None) is None:
        raise RuntimeError("MPFIT did not return any valid parameter vector in any multistart run.")

    best = apply_free_vector_to_specs(specs, np.asarray(best_result.params, dtype=float))
    errors = {name: None for name in best.keys()}
    if getattr(best_result, "perror", None) is not None:
        j = 0
        for sp in specs:
            if sp.fixed:
                errors[sp.name] = None
            else:
                perr_j = best_result.perror[j]
                errors[sp.name] = float(perr_j) if perr_j is not None else None
                j += 1

    chi2 = None
    if getattr(best_result, "fnorm", None) is not None:
        chi2 = float(best_result.fnorm)

    covar = None
    if getattr(best_result, "covar", None) is not None:
        try:
            covar = np.asarray(best_result.covar, dtype=float).tolist()
        except Exception:
            covar = None

    dof = None
    if getattr(best_result, "dof", None) is not None:
        dof = int(best_result.dof)

    return FitResult(
        backend="mpfit",
        success=bool(getattr(best_result, "status", 0) > 0),
        status=int(getattr(best_result, "status", -999)),
        message=str(getattr(best_result, "errmsg", "")),
        params=best,
        errors=errors,
        chi2=chi2,
        dof=dof,
        nfev=int(getattr(best_result, "nfev", 0)) if getattr(best_result, "nfev", None) is not None else None,
        covar=covar,
    )
    
def fit_with_scipy(
    *,
    specs: list[FitParameterSpec],
    mods: ModuleBundle,
    config: GUIConfig,
    earth_panel: EarthCompositePanel,
    cor1_panel: COR1Panel,
) -> FitResult:
    from scipy.optimize import least_squares

    p0 = vector_from_specs(specs)
    lo, hi = bounds_from_specs(specs)

    rng = np.random.default_rng(int(config.random_seed))
    starts = [np.asarray(p0, dtype=float)]
    width = hi - lo
    for _ in range(max(int(config.max_multistarts) - 1, 0)):
        jitter = rng.normal(0.0, 0.10, size=p0.size) * width
        starts.append(np.clip(p0 + jitter, lo, hi))

    best_res = None
    best_cost = np.inf

    for k, start in enumerate(starts):
        res = least_squares(
            residual_vector,
            x0=np.asarray(start, dtype=float),
            bounds=(lo, hi),
            method="trf",
            x_scale="jac",
            loss="linear",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=1000,
            kwargs={
                "specs": specs,
                "mods": mods,
                "config": config,
                "earth_panel": earth_panel,
                "cor1_panel": cor1_panel,
            },
        )
        if res.cost < best_cost:
            best_cost = float(res.cost)
            best_res = res
        print(f"[INFO] multistart {k+1}/{len(starts)} cost={res.cost:.6g} success={res.success}")

    if best_res is None:
        raise RuntimeError("SciPy least_squares did not return any result.")

    best = apply_free_vector_to_specs(specs, np.asarray(best_res.x, dtype=float))
    cov, perr = covariance_from_jacobian(best_res.jac, float(best_res.cost), best_res.fun.size, best_res.x.size)

    errors = {name: None for name in best.keys()}
    if perr is not None:
        j = 0
        for sp in specs:
            if sp.fixed:
                errors[sp.name] = None
            else:
                errors[sp.name] = float(perr[j])
                j += 1

    return FitResult(
        backend="scipy",
        success=bool(best_res.success),
        status=int(best_res.status),
        message=str(best_res.message),
        params=best,
        errors=errors,
        chi2=float(2.0 * best_res.cost),
        dof=int(best_res.fun.size - best_res.x.size),
        nfev=int(best_res.nfev),
        covar=None if cov is None else np.asarray(cov, dtype=float).tolist(),
    )

def run_fit(
    *,
    specs: list[FitParameterSpec],
    mods: ModuleBundle,
    config: GUIConfig,
    earth_panel: EarthCompositePanel,
    cor1_panel: COR1Panel,
) -> FitResult:
    backend = config.backend.lower()
    if backend == "mpfit":
        return fit_with_mpfit(
            specs=specs,
            mods=mods,
            config=config,
            earth_panel=earth_panel,
            cor1_panel=cor1_panel,
        )
    if backend == "scipy":
        return fit_with_scipy(
            specs=specs,
            mods=mods,
            config=config,
            earth_panel=earth_panel,
            cor1_panel=cor1_panel,
        )
    if backend == "auto":
        try:
            return fit_with_mpfit(
                specs=specs,
                mods=mods,
                config=config,
                earth_panel=earth_panel,
                cor1_panel=cor1_panel,
            )
        except Exception as exc:
            print(f"[WARN] MPFIT backend failed, falling back to SciPy: {exc}")
            return fit_with_scipy(
                specs=specs,
                mods=mods,
                config=config,
                earth_panel=earth_panel,
                cor1_panel=cor1_panel,
            )
    raise ValueError("backend must be 'auto', 'mpfit', or 'scipy'.")
# -----------------------------------------------------------------------------
# GUI driver
# -----------------------------------------------------------------------------

class MultiViewSpheroidGUI:
    def __init__(self, mods: ModuleBundle, config: GUIConfig):
        self.mods = mods
        self.config = config
        self.last_fit: FitResult | None = None

        fig = plt.figure(figsize=(16, 8), dpi=300)
        self.fig = fig
        ax0 = fig.add_subplot(1, 2, 1)
        ax1 = fig.add_subplot(1, 2, 2)

        print(f"[INFO] Building Earth-view panel for {config.target_time}")
        earth_res = mods.ia.create_single_diff_from_time_image(
            ax0,
            config.target_time,
            config.earth_base_minutes,
            mk4_inner=1.4,
            mk4_outer_lasco_inner=2.2,
            lasco_outer=7.0,
            xlim_min=-512,
            xlim_max=512,
            ylim_min=-512,
            ylim_max=512,
        )
        self.earth_panel = EarthCompositePanel(ax0, earth_res["params_lasco"], earth_res["lasco_map"])

        print(f"[INFO] Building COR1 panel for {config.target_time}")
        cor1_ax, raw_cor1_map, euvi_diff_map, *_ = mods.cor1.create_integrated_stereo_image(
            ax1,
            config.target_time,
            cor1_base_minutes_before=config.cor1_base_minutes,
            cor1_outer_rsun=config.cor1_plot_radius_rsun,
        )
        display_common_map = mods.cor1.build_common_reference_map(
            euvi_diff_map,
            outer_rsun=config.cor1_plot_radius_rsun,
        )
        self.cor1_panel = COR1Panel(cor1_ax, display_common_map, raw_cor1_map)
        self._disable_axes_artist_mouseover(self.earth_panel.ax)
        self._disable_axes_artist_mouseover(self.cor1_panel.ax)

        self.active_panel: BasePanelContext = self.earth_panel
        self._connect_events()
        fig.tight_layout()
        print("- 左クリック: 点追加")
        print("- 右クリック: 1点戻す")
        print("- `c`: アクティブ panel の点を全消去")
        print("- `d`: フィット実行")
        print("- `s`: 現在の図と JSON 保存")
        print("- `q`: 終了")
        
    @staticmethod
    def _disable_axes_artist_mouseover(ax) -> None:
        for art in ax.get_children():
            try:
                art.set_mouseover(False)
            except Exception:
                pass

    def _connect_events(self) -> None:
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _highlight_active_panel(self) -> None:
        for panel in (self.earth_panel, self.cor1_panel):
            for sp in panel.ax.spines.values():
                sp.set_linewidth(1.2)
                sp.set_color("black")
        for sp in self.active_panel.ax.spines.values():
            sp.set_linewidth(3.0)
            sp.set_color("magenta")
        self.fig.canvas.draw_idle()

    def panel_from_axes(self, ax) -> BasePanelContext | None:
        if ax is self.earth_panel.ax:
            return self.earth_panel
        if ax is self.cor1_panel.ax:
            return self.cor1_panel
        return None

    def on_click(self, event) -> None:
        panel = self.panel_from_axes(event.inaxes)
        if panel is None or event.xdata is None or event.ydata is None:
            return

        self.active_panel = panel
        self._highlight_active_panel()

        if event.button == 1:
            panel.add_point(event.xdata, event.ydata)
            print(f"[INFO] Added point to {panel.name}: ({event.xdata:.2f}, {event.ydata:.2f})")
        elif event.button == 3:
            panel.undo_last()
            print(f"[INFO] Undo last point on {panel.name}")

    def on_key(self, event) -> None:
        if event.key == "c":
            self.active_panel.clear_points()
            print(f"[INFO] Cleared points on {self.active_panel.name}")
        elif event.key == "d":
            print("\n=================\nFit start... \n")
            self.fit_and_overlay()
        elif event.key == "s":
            self.save_outputs()
        elif event.key == "q":
            plt.close(self.fig)

    def fit_and_overlay(self) -> None:
        if len(self.earth_panel.click_xy) < 3 or len(self.cor1_panel.click_xy) < 3:
            print("[WARN] At least 3 points are required on each panel before fitting.")
            return

        specs = make_parameter_specs(self.config, self.earth_panel, self.cor1_panel)
        result = run_fit(
            specs=specs,
            mods=self.mods,
            config=self.config,
            earth_panel=self.earth_panel,
            cor1_panel=self.cor1_panel,
        )
        self.last_fit = result

        print("\n=== FIT RESULT ===")
        print(f"backend = {result.backend}")
        print(f"success = {result.success}")
        print(f"status  = {result.status}")
        print(f"message = {result.message}")
        if result.chi2 is not None and result.dof is not None and result.dof > 0:
            print(f"chi2/dof = {result.chi2:.6g} / {result.dof} = {result.chi2/result.dof:.6g}")
        for k, v in result.params.items():
            err = result.errors.get(k, None)
            if err is None:
                print(f"  {k:16s} = {v:12.6f}")
            else:
                print(f"  {k:16s} = {v:12.6f} ± {err:.6f}")
        print("==================\n")

        spheroid = build_spheroid_params(self.mods, self.config, fitted=result.params)
        self.earth_panel.overlay_fit(self.mods, spheroid, self.config)
        self.cor1_panel.overlay_fit(self.mods, spheroid, self.config)

        self.save_outputs(auto=True)

    def _output_stem(self) -> str:
        t = self.config.target_time.replace(":", "").replace("-", "").replace("T", "_")
        return f"spheroid_multiview_fit_{t}"

    def save_outputs(self, auto: bool = False) -> None:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = self._output_stem()
        png_path = out_dir / f"{stem}.png"
        json_path = out_dir / f"{stem}.json"

        self.fig.savefig(png_path, dpi=300, bbox_inches="tight")

        payload = {
            "config": asdict(self.config),
            "earth_clicks": self.earth_panel.serialize_clicks(),
            "cor1_clicks": self.cor1_panel.serialize_clicks(),
            "fit_result": None if self.last_fit is None else asdict(self.last_fit),
        }
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

        prefix = "[AUTO-SAVE]" if auto else "[SAVE]"
        print(f"{prefix} Figure: {png_path}")
        print(f"{prefix} JSON  : {json_path}")
        
        
# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(
    config: GUIConfig,
    *,
    integrated_analysis_path: str | Path | None = None,
    cor1_diff_plot_path: str | Path | None = None,
    plot_spheroid_c2_path: str | Path | None = None,
) -> None:
    global mods_global, mods_global_config

    mods_global = load_modules(
        integrated_analysis_path=None if integrated_analysis_path is None else Path(integrated_analysis_path),
        cor1_diff_plot_path=None if cor1_diff_plot_path is None else Path(cor1_diff_plot_path),
        plot_spheroid_c2_path=None if plot_spheroid_c2_path is None else Path(plot_spheroid_c2_path),
    )

    mods_global_config = config
    ensure_interactive_backend()
    gui = MultiViewSpheroidGUI(mods_global, config)
    plt.show()    
    
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # User-configurable parameters
    # -------------------------------------------------------------------------
    # ------------------------------------------
    # 固定値
    earth_base_minutes = 10
    cor1_base_minutes = 10
    cor1_plot_radius_rsun = 4.0

    pa_bin_deg = 5.0
    envelope_bin_deg = 2.0

    backend = "mpfit"          # "auto", "mpfit", or "scipy"
    multistarts = 10
    seed = 0
    fit_anchor = False
    
    model_to_obs_weight=0.7,
    reverse_front_min_points=24,
    
    integrated_analysis_path = None
    cor1_diff_plot_path = None
    plot_spheroid_c2_path = None

    output_dir = "/mnt/d/wsl/home/kinno-7010/Research_data/GCS/output"
    # ------------------------------------------
    ###########################################
    # 変数
    target_time = "2022-06-13T03:48:36"
    
    anchor_lon_deg = -30.0
    anchor_lat_deg = 19.0
    apex_lon_deg = -54.0394
    apex_lat_deg = 4.9501
    apex_r_rsun = 4.6322
    kappa = 0.5304
    epsilon = -0.3125

    '''
    - 左クリック: 点追加
    - 右クリック: 1点戻す
    - `c`: アクティブ panel の点を全消去
    - `d`: フィット実行
    - `s`: 現在の図と JSON 保存
    - `q`: 終了
    '''
    ###########################################

    config = GUIConfig(
        target_time=target_time,
        earth_base_minutes=earth_base_minutes,
        cor1_base_minutes=cor1_base_minutes,
        cor1_plot_radius_rsun=cor1_plot_radius_rsun,
        pa_bin_deg=pa_bin_deg,
        envelope_bin_deg=envelope_bin_deg,
        max_multistarts=multistarts,
        random_seed=seed,
        backend=backend,
        fit_anchor=fit_anchor,
        output_dir=output_dir,
        anchor_lon_deg=anchor_lon_deg,
        anchor_lat_deg=anchor_lat_deg,
        apex_lon_deg=apex_lon_deg,
        apex_lat_deg=apex_lat_deg,
        apex_r_rsun=apex_r_rsun,
        kappa=kappa,
        epsilon=epsilon,
    )

    main(
        config,
        integrated_analysis_path=integrated_analysis_path,
        cor1_diff_plot_path=cor1_diff_plot_path,
        plot_spheroid_c2_path=plot_spheroid_c2_path,
    )
