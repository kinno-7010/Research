#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_extended_multi_tomo.py

STEP 1 extension for main_multi_tomo.py.

This file does not modify main_multi_tomo.py.  Instead, it imports the functions/classes
from main_multi_tomo.py and adds diagnostic / validation features around the existing
tomography pipeline.

STEP 1 contents
---------------
1. Lambda scan diagnostics
   - solve the same tomography problem for multiple Tikhonov lambda values
   - save data misfit, regularization norm, density/frequency range, and target-frequency metrics
   - estimate a simple L-curve corner candidate

2. Per-image and per-instrument residual diagnostics
   - weighted residual metrics for each input pB image
   - group-level residual summaries
   - optional residual-map saving as .npy and .png

3. Optional true leave-one-image-out validation
   - disabled by default because it is expensive
   - when enabled, one image is held out, the solution is estimated from the remaining images,
     and the held-out image is predicted by forward projection

4. Final run
   - save an NPZ solution and PNG using either the original lambda or the L-curve candidate
   - the default is to keep the original lambda so that STEP 1 remains diagnostic-first

Important
---------
Put this file in the same directory as main_multi_tomo.py, or set BASE_MODULE_PATH below.
The original main_multi_tomo.py is treated as a read-only source module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import hashlib
import importlib.util
import json
import math
import pickle
import sys
import time

# from SOHO.pB.tomo_example import f
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy import ndimage
except Exception:
    ndimage = None


# ============================================================
# Import original tomography code without modifying it
# ============================================================

BASE_MODULE_PATH = Path(__file__).with_name("main_multi_tomo.py")


def import_base_tomo_module(path: Path = BASE_MODULE_PATH):
    """Import main_multi_tomo.py as a module from an explicit file path."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Base tomography file not found: {path}\n"
            "Place main_extended_multi_tomo.py in the same directory as main_multi_tomo.py, "
            "or edit BASE_MODULE_PATH."
        )

    spec = importlib.util.spec_from_file_location("main_multi_tomo_base", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["main_multi_tomo_base"] = module
    spec.loader.exec_module(module)
    return module


base = import_base_tomo_module()


# ============================================================
# Small containers
# ============================================================

@dataclass
class PreparedProblem:
    args: SimpleNamespace
    pb_paths: List[Path]
    grid: object
    obs_list: List[object]
    rays: List[object]
    y_obs: np.ndarray
    wt_r: Optional[np.ndarray]
    density_basis: Optional[np.ndarray]
    obs_r_bounds: List[Tuple[float, float]]
    ybk_list: List[Tuple[np.ndarray, np.ndarray]]


@dataclass
class LambdaResult:
    lam: float
    info: int
    solve_seconds: float
    data_misfit_norm: float
    data_misfit_rms: float
    weighted_rms_rel: float
    regularization_norm: float
    suggested_brightness_scale: float
    ne_min: float
    ne_max: float
    f_min_mhz: float
    f_max_mhz: float
    target_metrics: Dict[str, float]
    solution_raw: Optional[np.ndarray] = None
    ne_raw: Optional[np.ndarray] = None


# ============================================================
# General utilities
# ============================================================

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path | str) -> Path:
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def finite_or_nan(value) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def write_rows_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    # Preserve first-row order, then append any later keys.
    keys: List[str] = list(rows[0].keys())
    for row in rows[1:]:
        for k in row.keys():
            if k not in keys:
                keys.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# Ray cache utilities
# ============================================================

RAY_CACHE_VERSION = "ray_cache_v1"
_RAY_MEMORY_CACHE: Dict[str, object] = {}


def _hash_update_value(h: "hashlib._Hash", value: object) -> None:
    h.update(repr(value).encode("utf-8"))
    h.update(b"\0")


def _hash_update_array(h: "hashlib._Hash", arr: np.ndarray) -> None:
    a = np.ascontiguousarray(arr)
    _hash_update_value(h, str(a.dtype))
    _hash_update_value(h, a.shape)
    h.update(a.view(np.uint8))
    h.update(b"\0")


def ray_cache_key(
    obs,
    pb_path: Path,
    grid,
    ds_rsun: float,
    r_min: float,
    r_max: float,
    limb_u: float,
) -> str:
    """Return a stable key for one observation/grid/ray-construction setup."""
    h = hashlib.sha256()
    _hash_update_value(h, RAY_CACHE_VERSION)

    base_path = Path(BASE_MODULE_PATH).expanduser().resolve()
    if base_path.exists():
        st_base = base_path.stat()
        _hash_update_value(h, str(base_path))
        _hash_update_value(h, int(st_base.st_mtime_ns))
        _hash_update_value(h, int(st_base.st_size))

    p = Path(pb_path).expanduser().resolve()
    _hash_update_value(h, str(p))
    if p.exists():
        st = p.stat()
        _hash_update_value(h, int(st.st_mtime_ns))
        _hash_update_value(h, int(st.st_size))

    _hash_update_value(h, float(ds_rsun))
    _hash_update_value(h, float(r_min))
    _hash_update_value(h, float(r_max))
    _hash_update_value(h, float(limb_u))
    _hash_update_value(h, getattr(obs, "lonlat_deg", None))
    _hash_update_array(h, np.asarray(grid.r_edges, dtype=np.float64))
    _hash_update_array(h, np.asarray(grid.th_edges, dtype=np.float64))
    _hash_update_array(h, np.asarray(grid.ph_edges, dtype=np.float64))
    _hash_update_array(h, np.asarray(obs.idx_map, dtype=np.int64))
    _hash_update_array(h, np.asarray(obs.x, dtype=np.float64))
    _hash_update_array(h, np.asarray(obs.y, dtype=np.float64))
    return h.hexdigest()


def ray_cache_path(cache_dir: Path | str, key: str) -> Path:
    return Path(cache_dir).expanduser() / f"ray_{key}.pkl"


def load_cached_ray(key: str, cache_dir: Path | str = "") -> Optional[object]:
    """Load one cached ray object from memory first, then from disk if available."""
    if key in _RAY_MEMORY_CACHE:
        return _RAY_MEMORY_CACHE[key]

    if not cache_dir:
        return None

    fp = ray_cache_path(cache_dir, key)
    if not fp.exists():
        return None

    try:
        with fp.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or payload.get("version") != RAY_CACHE_VERSION:
            return None
        ray = payload.get("ray")
        if ray is None:
            return None
        _RAY_MEMORY_CACHE[key] = ray
        return ray
    except Exception as exc:
        print(f"[WARN] Failed to load cached rays from {fp}: {exc}")
        return None


def save_cached_ray(key: str, ray: object, cache_dir: Path | str = "") -> None:
    """Save one ray object to memory and, optionally, to disk."""
    _RAY_MEMORY_CACHE[key] = ray

    if not cache_dir:
        return

    fp = ray_cache_path(cache_dir, key)
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = fp.with_suffix(fp.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump({"version": RAY_CACHE_VERSION, "ray": ray}, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(fp)
    except Exception as exc:
        print(f"[WARN] Failed to save cached rays to {fp}: {exc}")


def weighted_stats(y_obs: np.ndarray, y_pred: np.ndarray, W: np.ndarray) -> Dict[str, float]:
    """Return robust residual statistics for one vector of measurements."""
    y_obs = np.asarray(y_obs, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    W = np.asarray(W, dtype=np.float64).ravel()

    m = np.isfinite(y_obs) & np.isfinite(y_pred) & np.isfinite(W) & (W > 0)
    if not np.any(m):
        return {
            "n": 0,
            "misfit_norm": float("nan"),
            "misfit_rms": float("nan"),
            "weighted_rms_rel": float("nan"),
            "median_obs_over_pred": float("nan"),
            "median_obs": float("nan"),
            "median_pred": float("nan"),
            "median_residual": float("nan"),
            "mad_residual": float("nan"),
        }

    yo = y_obs[m]
    yp = y_pred[m]
    ww = W[m]
    residual = yp - yo
    wres = ww * residual

    denom = np.maximum(np.abs(yo), 1e-30)
    rel = (yp - yo) / denom

    good_ratio = np.isfinite(yo) & np.isfinite(yp) & (np.abs(yp) > 0)
    med_ratio = float(np.nanmedian(yo[good_ratio] / yp[good_ratio])) if np.any(good_ratio) else float("nan")

    med_res = float(np.nanmedian(residual))
    mad_res = float(1.4826 * np.nanmedian(np.abs(residual - med_res)))

    return {
        "n": int(np.count_nonzero(m)),
        "misfit_norm": float(np.sqrt(np.sum(wres * wres))),
        "misfit_rms": float(np.sqrt(np.mean(wres * wres))),
        "weighted_rms_rel": float(np.sqrt(np.mean((ww * rel) ** 2))),
        "median_obs_over_pred": med_ratio,
        "median_obs": float(np.nanmedian(yo)),
        "median_pred": float(np.nanmedian(yp)),
        "median_residual": med_res,
        "mad_residual": mad_res,
    }


def normalized_lcurve_corner(
    lams: Sequence[float],
    misfits: Sequence[float],
    reg_norms: Sequence[float],
) -> Optional[float]:
    """
    Pick a simple L-curve corner candidate.

    Method:
      - work in log10(misfit)-log10(reg_norm) space
      - normalize both axes to [0, 1]
      - choose the point with the largest distance from the line connecting endpoints

    This is intentionally a diagnostic, not a proof of optimality.
    """
    lams = np.asarray(lams, dtype=np.float64)
    mis = np.asarray(misfits, dtype=np.float64)
    reg = np.asarray(reg_norms, dtype=np.float64)

    good = (
        np.isfinite(lams) & (lams > 0)
        & np.isfinite(mis) & (mis > 0)
        & np.isfinite(reg) & (reg > 0)
    )
    if np.count_nonzero(good) < 3:
        return None

    lams = lams[good]
    x = np.log10(mis[good])
    y = np.log10(reg[good])

    # Sort by lambda to make endpoints meaningful.
    order = np.argsort(lams)
    lams = lams[order]
    x = x[order]
    y = y[order]

    def _norm(v):
        vmin = np.nanmin(v)
        vmax = np.nanmax(v)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return np.zeros_like(v)
        return (v - vmin) / (vmax - vmin)

    xn = _norm(x)
    yn = _norm(y)
    p0 = np.array([xn[0], yn[0]], dtype=np.float64)
    p1 = np.array([xn[-1], yn[-1]], dtype=np.float64)
    line = p1 - p0
    line_norm = np.linalg.norm(line)
    if line_norm <= 0:
        return None

    distances = []
    for xi, yi in zip(xn, yn):
        p = np.array([xi, yi], dtype=np.float64)
        dist = abs(np.cross(line, p - p0)) / line_norm
        distances.append(dist)

    k = int(np.nanargmax(distances))
    return float(lams[k])


# ============================================================
# Problem construction using functions from main_multi_tomo.py
# ============================================================

def apply_defaults(args: SimpleNamespace) -> SimpleNamespace:
    """Add missing arguments with the same defaults as the base script."""
    defaults = dict(
        pb_fits=[],
        out_n=128,

        default_lonlat="",
        lonlat_file="",

        r_min=1.5,
        r_max=4.0,
        nr=40,
        nth=60,
        nph=120,

        ds=0.02,
        limb_u=base.DEFAULT_LIMB_U,

        filt=1,
        despike_nsig=6.0,
        despike_med=5,
        pb_floor="",
        dpa_deg=1.0,
        r_use_min=1.5,
        r_use_max=4.0,
        r_use_min_by_group={},
        r_use_max_by_group={},
        pb_scale_by_group={},
        hm=6,
        wt_nr=1,

        lam=1.0,
        q_low=0.0,
        width_pix=2.0,
        maxiter=10000,
        tol=1e-3,
        apply_brightness_scale=False,
        density_prior_model="none",
        density_prior_scale=1.0,
        calibration_reference_group="earth_merged",

        data_dir="",
        cor1a_data_dir="",
        target_time="",
        search_window_days=7.0,
        auto_find_pb_fits=False,
        include_kcor_lasco=True,
        include_cor1a=True,
        include_lasco_only=True,
        deduplicate_pb_fits=True,
        use_temporal_despike=False,
        ne3dtomo_global_ybk=False,
        show_ray_progress=True,
        use_ray_cache=True,
        ray_cache_dir="",

        save_prepped_dir="",
        save_ne_npz="",

        show_gui=False,
        freq_mhz=25.0,
        freq_mhz_list=None,
        harmonic=1,
        iso_colors=None,
        save_png=False,
        png_path="",

        step1_output_dir="step1_diagnostics",
        step1_run_lambda_scan=True,
        step1_lambda_values=[],
        step1_final_lam_mode="default",
        step1_save_residual_npy=True,
        step1_save_residual_png=False,
        step1_run_leave_one_image_out=False,
        step1_loo_final_lambda_only=True,
        step1_loo_max_holdouts=None,
        step1_run_comparison_suite=False,
        step1_comparison_output_dir="",
        step1_comparison_scenarios=[],
    )
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


def prepare_tomography_problem(args: SimpleNamespace) -> PreparedProblem:
    """
    Prepare observations, grid, y_obs, rays, and regularization weights.

    This is intentionally a transparent wrapper around the original functions in
    main_multi_tomo.py.  The original file is not modified.
    """
    args = apply_defaults(args)

    if bool(args.auto_find_pb_fits):
        if not args.data_dir:
            raise ValueError("data_dir is required when auto_find_pb_fits=True.")
        if not args.target_time:
            raise ValueError("target_time is required when auto_find_pb_fits=True.")

        found = base.find_pb_fits_in_time_window(
            data_dir=Path(args.data_dir),
            target_time=args.target_time,
            window_days=float(args.search_window_days),
            include_kcor_lasco=bool(args.include_kcor_lasco),
            include_cor1a=bool(args.include_cor1a),
            include_lasco_only=bool(args.include_lasco_only),
            cor1a_data_dir=(args.cor1a_data_dir if getattr(args, "cor1a_data_dir", "") else None),
            match_cor1a_to_earth=True,
            max_cor1a_match_minutes=90.0,
        )
        if bool(args.deduplicate_pb_fits):
            found = base.deduplicate_tomography_pb_paths(found, verbose=True)
        args.pb_fits = [str(p) for p in found]
        print(f"[INFO] Tomography-ready pB files selected: {len(args.pb_fits)}")
        for path in args.pb_fits:
            print(f"       {path}")

    if not args.pb_fits:
        raise ValueError("pb_fits is empty. Set PB_FITS or use AUTO_FIND_PB_FITS=True.")

    default_lonlat = None
    if args.default_lonlat:
        a, b = args.default_lonlat.split(",")
        default_lonlat = (float(a), float(b))

    lonlat_map = {}
    if args.lonlat_file:
        fp = Path(args.lonlat_file)
        if not fp.exists():
            raise FileNotFoundError(fp)
        with fp.open("r", newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row or row[0].strip().startswith("#") or len(row) < 3:
                    continue
                lonlat_map[row[0].strip()] = (float(row[1]), float(row[2]))

    pb_paths = [Path(p) for p in args.pb_fits]
    for p in pb_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    r_use_min_by_group = base.normalize_group_float_map(args.r_use_min_by_group, "r_use_min_by_group")
    r_use_max_by_group = base.normalize_group_float_map(args.r_use_max_by_group, "r_use_max_by_group")
    pb_scale_by_group = base.normalize_group_float_map(args.pb_scale_by_group, "pb_scale_by_group")

    pb_overrides = {}
    if args.filt and len(pb_paths) >= 2 and bool(args.use_temporal_despike):
        pb_overrides = base.build_ne3dtomo_temporal_despike_overrides(
            pb_paths=pb_paths,
            out_n=int(args.out_n),
            nsig=float(args.despike_nsig),
        )
        if not pb_overrides:
            print("[INFO] Temporal despike requested, but no homogeneous group was usable; applying spatial despike per image only.")
    elif args.filt and len(pb_paths) >= 2:
        print("[INFO] Global temporal despike disabled; applying spatial despike per image only.")

    r_edges = np.linspace(float(args.r_min), float(args.r_max), int(args.nr) + 1)
    th_edges = np.linspace(0.0, np.pi, int(args.nth) + 1)
    ph_edges = np.linspace(0.0, 2.0 * np.pi, int(args.nph) + 1)
    grid = base.SphericalGrid(r_edges=r_edges, th_edges=th_edges, ph_edges=ph_edges)

    density_basis = base.density_basis_from_grid(
        grid,
        model=str(args.density_prior_model),
        scale=float(args.density_prior_scale),
    )
    if density_basis is not None:
        print(
            "[INFO] Density prior enabled: "
            f"model={str(args.density_prior_model)!r}, scale={float(args.density_prior_scale):.6g}, "
            "solving ne = prior * q."
        )
    else:
        print("[INFO] Density prior disabled: solving absolute electron density ne.")

    save_prepped_dir = Path(args.save_prepped_dir) if args.save_prepped_dir else None

    obs_list: List[object] = []
    local_ybk_list: List[Tuple[np.ndarray, np.ndarray]] = []
    obs_r_bounds: List[Tuple[float, float]] = []

    for p in pb_paths:
        group_key = base.tomography_observation_group_key(p)
        obs_r_use_min = float(r_use_min_by_group.get(group_key, args.r_use_min))
        obs_r_use_max = float(r_use_max_by_group.get(group_key, args.r_use_max))

        if obs_r_use_min >= obs_r_use_max:
            raise ValueError(f"Invalid r_use bounds for {p.name}: {obs_r_use_min} >= {obs_r_use_max} Rsun.")
        if obs_r_use_min < float(args.r_min) - 1e-8:
            raise ValueError(
                f"{p.name} uses r_use_min={obs_r_use_min} Rsun, smaller than reconstruction r_min={args.r_min} Rsun."
            )
        if obs_r_use_max > float(args.r_max) + 1e-8:
            raise ValueError(
                f"{p.name} uses r_use_max={obs_r_use_max} Rsun, larger than reconstruction r_max={args.r_max} Rsun."
            )

        obs = base.build_observation(
            pb_fits=p,
            out_n=int(args.out_n),
            pb_override=pb_overrides.get(p),
            apply_spatial_despike=(p not in pb_overrides),
            r_use_min=obs_r_use_min,
            r_use_max=obs_r_use_max,
            limb_u=float(args.limb_u),
            filt=bool(args.filt),
            despike_nsig=float(args.despike_nsig),
            despike_med=int(args.despike_med),
            pb_floor=args.pb_floor,
            dpa_deg=float(args.dpa_deg),
            hm=int(args.hm),
            width_pix=float(args.width_pix),
            q_low=float(args.q_low),
            lonlat_override=lonlat_map.get(p.name) or lonlat_map.get(str(p)) or lonlat_map.get(p.stem),
            lonlat_default=default_lonlat,
            save_prepped_dir=save_prepped_dir,
        )

        pb_scale = float(pb_scale_by_group.get(group_key, 1.0))
        if abs(pb_scale - 1.0) > 1e-12:
            obs = base.scale_observation_pb(obs, pb_scale)
            print(
                f"[INFO] Applied explicit pB calibration scale to {p.name}: "
                f"group={group_key}, scale={pb_scale:.6g}"
            )

        obs_list.append(obs)
        obs_r_bounds.append((obs_r_use_min, obs_r_use_max))

        rho = np.hypot(obs.x, obs.y)
        print(
            f"[GEOM] {p.name}: group={group_key}, lonlat={obs.lonlat_deg}, "
            f"used_pixels={obs.idx_map.size}, r_use={obs_r_use_min:.3f}..{obs_r_use_max:.3f} Rs, "
            f"pb_scale={pb_scale:.6g}, rho={np.nanmin(rho):.3f}..{np.nanmax(rho):.3f} Rs"
        )

        rgrid, ybk, _ = base.ybk_profile_fft(
            pb=obs.pb,
            hdr=obs.hdr,
            rmin=obs_r_use_min,
            rmax=obs_r_use_max,
            dpa_deg=float(args.dpa_deg),
            nr=240,
            hm=int(args.hm),
            width_pix=float(args.width_pix),
            q_low=float(args.q_low),
        )
        local_ybk_list.append((rgrid, ybk))

    if bool(getattr(args, "ne3dtomo_global_ybk", False)):
        grouped_indices: Dict[str, List[int]] = {}
        for i, p in enumerate(pb_paths):
            grouped_indices.setdefault(base.tomography_observation_group_key(p), []).append(i)

        ybk_list: List[Tuple[np.ndarray, np.ndarray]] = [local_ybk_list[i] for i in range(len(obs_list))]
        for key, indices in grouped_indices.items():
            group_obs = [obs_list[i] for i in indices]
            group_bounds = [obs_r_bounds[i] for i in indices]
            group_rmin = group_bounds[0][0]
            group_rmax = group_bounds[0][1]

            if any((abs(b0 - group_rmin) > 1e-8 or abs(b1 - group_rmax) > 1e-8) for b0, b1 in group_bounds):
                raise ValueError(
                    f"Group {key!r} has mixed r_use bounds; global ybk requires one radial range per group."
                )

            rgrid_g, ybk_g, pb_noise_g = base.ybk_profile_fft_stack(
                observations=group_obs,
                rmin=group_rmin,
                rmax=group_rmax,
                dpa_deg=float(args.dpa_deg),
                nr=240,
                hm=int(args.hm),
                width_pix=float(args.width_pix),
                q_low=float(args.q_low),
            )
            for i in indices:
                obs_list[i] = base.update_observation_weights_from_ybk(
                    obs=obs_list[i],
                    rgrid=rgrid_g,
                    ybk=ybk_g,
                    pb_noise=pb_noise_g,
                    pb_floor=args.pb_floor,
                )
                ybk_list[i] = (rgrid_g, ybk_g)
            print(f"[INFO] Ne3dTomo-style global ybk(r) applied for group {key!r} (n={len(indices)}).")
    else:
        ybk_list = local_ybk_list

    y_list: List[np.ndarray] = []
    for p, obs in zip(pb_paths, obs_list):
        y_vec = obs.pb.ravel()[obs.idx_map]
        y_list.append(y_vec)
        vv = y_vec[np.isfinite(y_vec)]
        if vv.size:
            print(
                f"[INFO] {p.name}: pB (used pixels) min/med/max = "
                f"{np.min(vv):.3e} / {np.median(vv):.3e} / {np.max(vv):.3e}"
            )

    y_obs = np.concatenate(y_list) if y_list else np.array([], dtype=np.float64)
    if y_obs.size == 0 or not np.any(np.isfinite(y_obs)):
        raise ValueError("y_obs is empty or all-NaN. Check masks and preprocessing.")

    rays: List[object] = []
    n_obs = len(obs_list)
    use_ray_cache = bool(getattr(args, "use_ray_cache", True))
    ray_cache_dir = str(getattr(args, "ray_cache_dir", "") or "")
    if use_ray_cache and ray_cache_dir:
        ensure_dir(ray_cache_dir)
        print(f"[INFO] Ray cache enabled: {ray_cache_dir}")
    elif use_ray_cache:
        print("[INFO] Ray cache enabled: in-memory only.")

    for i, (obs, p) in enumerate(zip(obs_list, pb_paths), start=1):
        cache_key = None
        ray = None
        if use_ray_cache:
            cache_key = ray_cache_key(
                obs=obs,
                pb_path=p,
                grid=grid,
                ds_rsun=float(args.ds),
                r_min=float(args.r_min),
                r_max=float(args.r_max),
                limb_u=float(args.limb_u),
            )
            ray = load_cached_ray(cache_key, ray_cache_dir)

        if ray is not None:
            if bool(getattr(args, "show_ray_progress", True)):
                nonempty = sum(1 for idx in ray.vox_idx if idx.size > 0)
                print(
                    f"[INFO] Reusing cached rays {i}/{n_obs}: {Path(p).name} "
                    f"(non-empty rays={nonempty}/{len(ray.vox_idx)})",
                    flush=True,
                )
        else:
            if bool(getattr(args, "show_ray_progress", True)):
                print(f"[INFO] Building rays {i}/{n_obs}: {Path(p).name} (used pixels={obs.idx_map.size})", flush=True)

            ray = base.build_rays_for_observation(
                obs=obs,
                grid=grid,
                ds_rsun=float(args.ds),
                r_min=float(args.r_min),
                r_max=float(args.r_max),
                limb_u=float(args.limb_u),
            )

            if use_ray_cache and cache_key is not None:
                save_cached_ray(cache_key, ray, ray_cache_dir)

            if bool(getattr(args, "show_ray_progress", True)):
                nonempty = sum(1 for idx in ray.vox_idx if idx.size > 0)
                print(
                    f"[INFO] Finished rays {i}/{n_obs}: {Path(p).name} "
                    f"(non-empty rays={nonempty}/{len(ray.vox_idx)})",
                    flush=True,
                )

        rays.append(ray)

    wt_r = None
    if int(args.wt_nr):
        r_cent = 0.5 * (r_edges[:-1] + r_edges[1:])
        ybks = [np.interp(r_cent, rgi, ybki) for (rgi, ybki) in ybk_list]
        ybk_mean = np.nanmean(np.stack(ybks, axis=0), axis=0)

        good = np.isfinite(ybk_mean) & (ybk_mean > 0)
        if np.count_nonzero(good) < 3:
            print("[WARN] wt_nr requested, but ybk_mean is not usable. Disabling radial weighting.")
            wt_r = None
        else:
            ybk_clean = ybk_mean.copy()
            if not np.all(good):
                ybk_clean[~good] = np.interp(r_cent[~good], r_cent[good], ybk_mean[good])
            ymax = float(np.nanmax(ybk_clean[good]))
            if not np.isfinite(ymax) or ymax <= 0:
                ymax = 1.0
            wt_r = ybk_clean / ymax
            wt_r = np.where(np.isfinite(wt_r) & (wt_r > 0), wt_r, 1.0)

    return PreparedProblem(
        args=args,
        pb_paths=pb_paths,
        grid=grid,
        obs_list=obs_list,
        rays=rays,
        y_obs=y_obs,
        wt_r=wt_r,
        density_basis=density_basis,
        obs_r_bounds=obs_r_bounds,
        ybk_list=ybk_list,
    )


# ============================================================
# Diagnostics
# ============================================================

def approximate_voxel_volumes_rsun3(grid) -> np.ndarray:
    """Exact spherical-shell voxel volumes in Rsun^3."""
    dr3 = (grid.r_edges[1:] ** 3 - grid.r_edges[:-1] ** 3) / 3.0
    dcosth = np.cos(grid.th_edges[:-1]) - np.cos(grid.th_edges[1:])
    dphi = grid.ph_edges[1:] - grid.ph_edges[:-1]
    vol = dr3[:, None, None] * dcosth[None, :, None] * dphi[None, None, :]
    return vol.astype(np.float64)


def target_frequency_metrics(grid, ne: np.ndarray, freq_mhz: float, harmonic: int) -> Dict[str, float]:
    """
    Compute simple, grid-based diagnostics for the region above target density.

    This is not an exact marching-cubes surface measurement.  It is a stable
    voxel-domain diagnostic for comparing lambda values:
      - number of voxels above threshold
      - volume above threshold
      - volume-weighted centroid of above-threshold region
      - number of connected components
      - largest component volume fraction
    """
    nr, nth, nph = grid.nr, grid.nth, grid.nph
    ne3 = np.asarray(ne, dtype=np.float64).reshape((nr, nth, nph), order="C")
    ne_target = base.ne_cm3_from_fp_mhz(float(freq_mhz), harmonic=int(harmonic))

    good = np.isfinite(ne3) & (ne3 >= ne_target)
    n_vox = int(np.count_nonzero(good))
    out = {
        "freq_mhz": float(freq_mhz),
        "target_ne_cm3": float(ne_target),
        "n_vox_ge_target": n_vox,
        "volume_ge_target_rsun3": 0.0,
        "centroid_x_rsun": float("nan"),
        "centroid_y_rsun": float("nan"),
        "centroid_z_rsun": float("nan"),
        "centroid_r_rsun": float("nan"),
        "n_components": 0,
        "largest_component_fraction": float("nan"),
    }
    if n_vox == 0:
        return out

    vol = approximate_voxel_volumes_rsun3(grid)
    vsel = vol[good]
    vtot = float(np.sum(vsel))
    out["volume_ge_target_rsun3"] = vtot

    xx, yy, zz = grid.voxel_centers_xyz()
    cx = float(np.sum(xx[good] * vsel) / max(vtot, 1e-300))
    cy = float(np.sum(yy[good] * vsel) / max(vtot, 1e-300))
    cz = float(np.sum(zz[good] * vsel) / max(vtot, 1e-300))
    out["centroid_x_rsun"] = cx
    out["centroid_y_rsun"] = cy
    out["centroid_z_rsun"] = cz
    out["centroid_r_rsun"] = float(np.sqrt(cx * cx + cy * cy + cz * cz))

    if ndimage is not None:
        structure = np.zeros((3, 3, 3), dtype=np.int8)
        structure[1, 1, :] = 1
        structure[1, :, 1] = 1
        structure[:, 1, 1] = 1
        labeled, ncomp = ndimage.label(good, structure=structure)
        out["n_components"] = int(ncomp)
        if ncomp > 0:
            comp_vol = np.bincount(labeled.ravel(), weights=vol.ravel())
            # comp_vol[0] is background.
            largest = float(np.max(comp_vol[1:])) if comp_vol.size > 1 else 0.0
            out["largest_component_fraction"] = largest / max(vtot, 1e-300)

    return out


def regularization_norm(solution_raw: np.ndarray, grid, wt_r: Optional[np.ndarray]) -> float:
    """Return sqrt(x^T L^T L x)."""
    x = np.asarray(solution_raw, dtype=np.float64).ravel()
    ltlx = base.apply_LTL(x, grid, wt_r=wt_r)
    val = float(np.dot(x, ltlx))
    if not np.isfinite(val) or val < 0:
        return float("nan")
    return float(np.sqrt(val))


def solve_one_lambda(
    tomo,
    prepared: PreparedProblem,
    lam: float,
    keep_solution: bool,
) -> LambdaResult:
    """Solve one lambda value and collect global diagnostics."""
    tomo.lam = float(lam)
    t0 = time.time()
    solution_raw, info = tomo.solve(
        prepared.y_obs,
        maxiter=int(prepared.args.maxiter),
        tol=float(prepared.args.tol),
        positivity=True,
    )
    elapsed = time.time() - t0

    ne_raw = tomo.solution_to_density(solution_raw)
    y_pred = tomo.A_times(solution_raw)
    stats = weighted_stats(prepared.y_obs, y_pred, tomo.W)
    suggested_scale = base.weighted_projection_scale(prepared.y_obs, y_pred, tomo.W, min_count=100)
    reg_norm = regularization_norm(solution_raw, prepared.grid, prepared.wt_r)

    fr = base.frequency_range_mhz_from_ne(ne_raw, harmonic=int(prepared.args.harmonic))
    if fr is None:
        ne_min = ne_max = f_min = f_max = float("nan")
    else:
        ne_min, ne_max, f_min, f_max = [float(v) for v in fr]

    target_metrics: Dict[str, float] = {}
    freq_list = list(prepared.args.freq_mhz_list) if prepared.args.freq_mhz_list is not None else [float(prepared.args.freq_mhz)]
    for f in freq_list:
        m = target_frequency_metrics(prepared.grid, ne_raw, float(f), int(prepared.args.harmonic))
        prefix = f"f{float(f):.3f}MHz_".replace(".", "p")
        for k, v in m.items():
            target_metrics[prefix + k] = finite_or_nan(v)

    return LambdaResult(
        lam=float(lam),
        info=int(info),
        solve_seconds=float(elapsed),
        data_misfit_norm=finite_or_nan(stats["misfit_norm"]),
        data_misfit_rms=finite_or_nan(stats["misfit_rms"]),
        weighted_rms_rel=finite_or_nan(stats["weighted_rms_rel"]),
        regularization_norm=finite_or_nan(reg_norm),
        suggested_brightness_scale=finite_or_nan(suggested_scale),
        ne_min=finite_or_nan(ne_min),
        ne_max=finite_or_nan(ne_max),
        f_min_mhz=finite_or_nan(f_min),
        f_max_mhz=finite_or_nan(f_max),
        target_metrics=target_metrics,
        solution_raw=solution_raw if keep_solution else None,
        ne_raw=ne_raw if keep_solution else None,
    )


def lambda_result_to_row(result: LambdaResult) -> Dict[str, object]:
    row: Dict[str, object] = {
        "lambda": result.lam,
        "cg_info": result.info,
        "solve_seconds": result.solve_seconds,
        "data_misfit_norm": result.data_misfit_norm,
        "data_misfit_rms": result.data_misfit_rms,
        "weighted_rms_rel": result.weighted_rms_rel,
        "regularization_norm": result.regularization_norm,
        "suggested_brightness_scale": result.suggested_brightness_scale,
        "ne_min_cm3": result.ne_min,
        "ne_max_cm3": result.ne_max,
        "f_min_mhz": result.f_min_mhz,
        "f_max_mhz": result.f_max_mhz,
    }
    row.update(result.target_metrics)
    return row


def run_lambda_scan(prepared: PreparedProblem, lambdas: Sequence[float], output_dir: Path) -> Tuple[List[LambdaResult], Optional[float]]:
    """Run lambda scan with one shared forward matrix."""
    output_dir = ensure_dir(output_dir)

    print("[STEP1] Building RegularizedTomography object once for lambda scan...")
    tomo = base.RegularizedTomography(
        prepared.grid,
        prepared.obs_list,
        prepared.rays,
        lam=float(lambdas[0]),
        wt_r=prepared.wt_r,
        density_basis=prepared.density_basis,
    )

    results: List[LambdaResult] = []
    for lam in lambdas:
        print(f"[STEP1] Solving lambda={float(lam):.6g} ...")
        res = solve_one_lambda(tomo, prepared, float(lam), keep_solution=False)
        results.append(res)
        print(
            f"[STEP1] lambda={res.lam:.6g}: "
            f"misfit_rms={res.data_misfit_rms:.4e}, "
            f"reg_norm={res.regularization_norm:.4e}, "
            f"f_range={res.f_min_mhz:.3f}..{res.f_max_mhz:.3f} MHz, "
            f"scale_hint={res.suggested_brightness_scale:.6g}, "
            f"cg_info={res.info}, time={res.solve_seconds:.1f}s"
        )

    rows = [lambda_result_to_row(r) for r in results]
    write_rows_csv(output_dir / "step1_lambda_scan.csv", rows)

    lam_corner = normalized_lcurve_corner(
        [r.lam for r in results],
        [r.data_misfit_norm for r in results],
        [r.regularization_norm for r in results],
    )

    summary = {
        "lambda_values": [float(r.lam) for r in results],
        "lcurve_lambda_candidate": lam_corner,
        "note": (
            "The L-curve candidate is only a diagnostic. "
            "Do not adopt it automatically without checking residual maps and target-frequency stability."
        ),
    }
    (output_dir / "step1_lambda_scan_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[STEP1] Saved lambda scan CSV: {output_dir / 'step1_lambda_scan.csv'}")
    if lam_corner is not None:
        print(f"[STEP1] L-curve corner candidate: lambda={lam_corner:.6g}")
    else:
        print("[STEP1] L-curve corner candidate could not be determined.")

    return results, lam_corner


def per_image_residual_rows(
    prepared: PreparedProblem,
    tomo,
    solution_raw: np.ndarray,
    y_pred: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for i, (path, obs, sl) in enumerate(zip(prepared.pb_paths, prepared.obs_list, tomo.slices)):
        yo = prepared.y_obs[sl]
        yp = y_pred[sl]
        ww = tomo.W[sl]
        st = weighted_stats(yo, yp, ww)
        rows.append({
            "obs_index": i,
            "pb_name": path.name,
            "pb_path": str(path),
            "group": base.tomography_observation_group_key(path),
            "obs_datetime": str(base.parse_pb_filename_datetime(path)),
            "lon_deg": obs.lonlat_deg[0] if obs.lonlat_deg is not None else np.nan,
            "lat_deg": obs.lonlat_deg[1] if obs.lonlat_deg is not None else np.nan,
            "used_pixels": int(obs.idx_map.size),
            **st,
        })
    return rows


def group_residual_rows(image_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = {}
    for row in image_rows:
        groups.setdefault(str(row["group"]), []).append(row)

    out: List[Dict[str, object]] = []
    for key, rows in groups.items():
        n_total = int(sum(int(r.get("n", 0)) for r in rows))
        out.append({
            "group": key,
            "n_images": len(rows),
            "n_points": n_total,
            "median_weighted_rms_rel_by_image": float(np.nanmedian([finite_or_nan(r.get("weighted_rms_rel")) for r in rows])),
            "median_misfit_rms_by_image": float(np.nanmedian([finite_or_nan(r.get("misfit_rms")) for r in rows])),
            "median_obs_over_pred_by_image": float(np.nanmedian([finite_or_nan(r.get("median_obs_over_pred")) for r in rows])),
            "max_weighted_rms_rel_image": float(np.nanmax([finite_or_nan(r.get("weighted_rms_rel")) for r in rows])),
        })
    return out


def save_residual_maps(
    prepared: PreparedProblem,
    tomo,
    y_pred: np.ndarray,
    output_dir: Path,
    save_npy: bool = True,
    save_png: bool = False,
) -> None:
    if not save_npy and not save_png:
        return

    out_dir = ensure_dir(output_dir)
    for i, (path, obs, sl) in enumerate(zip(prepared.pb_paths, prepared.obs_list, tomo.slices)):
        yo = prepared.y_obs[sl]
        yp = y_pred[sl]
        residual = yp - yo
        rel = residual / np.maximum(np.abs(yo), 1e-30)

        shape = obs.pb.shape
        res_map = np.full(shape[0] * shape[1], np.nan, dtype=np.float64)
        rel_map = np.full(shape[0] * shape[1], np.nan, dtype=np.float64)
        res_map[obs.idx_map] = residual
        rel_map[obs.idx_map] = rel
        res_map = res_map.reshape(shape)
        rel_map = rel_map.reshape(shape)

        stem = f"{i:03d}_{path.stem}"
        if save_npy:
            np.save(out_dir / f"{stem}_residual.npy", res_map.astype(np.float32))
            np.save(out_dir / f"{stem}_relative_residual.npy", rel_map.astype(np.float32))

        if save_png and plt is not None:
            # Let matplotlib choose colormap defaults; robust symmetric limits.
            finite = rel_map[np.isfinite(rel_map)]
            if finite.size:
                vmax = float(np.nanpercentile(np.abs(finite), 98.0))
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = 1.0
            else:
                vmax = 1.0

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(rel_map, origin="lower", vmin=-vmax, vmax=vmax)
            ax.set_title(f"Relative residual: {path.name}", fontsize=8)
            ax.set_xlabel("x pixel")
            ax.set_ylabel("y pixel")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(out_dir / f"{stem}_relative_residual.png", dpi=150)
            plt.close(fig)


def run_final_diagnostics(
    prepared: PreparedProblem,
    lam: float,
    output_dir: Path,
    save_residual_npy: bool,
    save_residual_png: bool,
) -> Tuple[object, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Solve once at final lambda and save residual diagnostics.
    Returns (tomo, solution_raw, ne, y_pred, suggested_scale).
    """
    output_dir = ensure_dir(output_dir)

    print(f"[STEP1] Final diagnostic solve with lambda={float(lam):.6g}")
    tomo = base.RegularizedTomography(
        prepared.grid,
        prepared.obs_list,
        prepared.rays,
        lam=float(lam),
        wt_r=prepared.wt_r,
        density_basis=prepared.density_basis,
    )

    solution_raw, info = tomo.solve(
        prepared.y_obs,
        maxiter=int(prepared.args.maxiter),
        tol=float(prepared.args.tol),
        positivity=True,
    )
    if info != 0:
        print(f"[WARN] Final CG did not fully converge (info={info}).")

    ne_raw = tomo.solution_to_density(solution_raw)
    y_pred = tomo.A_times(solution_raw)
    W = tomo.W
    suggested_scale = base.weighted_projection_scale(prepared.y_obs, y_pred, W, min_count=100)

    print("[STEP1] Final global projection diagnostic:")
    base.print_projection_fit_diagnostic("step1_final/global", prepared.y_obs, y_pred, W)
    base.print_projection_fit_diagnostics_by_group("step1_final", prepared.pb_paths, tomo, prepared.y_obs, y_pred, W)
    base.print_group_calibration_hints(
        "step1_final",
        prepared.pb_paths,
        tomo,
        prepared.y_obs,
        y_pred,
        W,
        reference_group=str(prepared.args.calibration_reference_group),
    )

    image_rows = per_image_residual_rows(prepared, tomo, solution_raw, y_pred)
    write_rows_csv(output_dir / "step1_per_image_residuals.csv", image_rows)
    write_rows_csv(output_dir / "step1_group_residuals.csv", group_residual_rows(image_rows))
    print(f"[STEP1] Saved per-image residual CSV: {output_dir / 'step1_per_image_residuals.csv'}")
    print(f"[STEP1] Saved group residual CSV: {output_dir / 'step1_group_residuals.csv'}")

    save_residual_maps(
        prepared,
        tomo,
        y_pred,
        output_dir=output_dir / "residual_maps",
        save_npy=bool(save_residual_npy),
        save_png=bool(save_residual_png),
    )

    scale = suggested_scale if bool(prepared.args.apply_brightness_scale) else 1.0
    ne = ne_raw * scale

    return tomo, solution_raw, ne, y_pred, suggested_scale


def run_leave_one_image_out(
    prepared: PreparedProblem,
    lambdas: Sequence[float],
    output_dir: Path,
    max_holdouts: Optional[int] = None,
) -> None:
    """
    Expensive true leave-one-image-out validation.

    For each held-out image and lambda:
      - solve using all other images
      - project the solution into the held-out geometry
      - save held-out residual metrics

    This is disabled by default in __main__.
    """
    output_dir = ensure_dir(output_dir)

    n = len(prepared.obs_list)
    holdout_indices = list(range(n))
    if max_holdouts is not None and max_holdouts > 0:
        holdout_indices = holdout_indices[: int(max_holdouts)]

    rows: List[Dict[str, object]] = []
    for ihold in holdout_indices:
        print(f"[STEP1-LOO] Holdout {ihold + 1}/{n}: {prepared.pb_paths[ihold].name}")
        train_obs = [obs for i, obs in enumerate(prepared.obs_list) if i != ihold]
        train_rays = [ray for i, ray in enumerate(prepared.rays) if i != ihold]
        train_y_parts = []
        for i, obs in enumerate(prepared.obs_list):
            if i == ihold:
                continue
            train_y_parts.append(obs.pb.ravel()[obs.idx_map])
        y_train = np.concatenate(train_y_parts)

        held_obs = prepared.obs_list[ihold]
        held_ray = prepared.rays[ihold]
        y_held = held_obs.pb.ravel()[held_obs.idx_map]

        train_tomo = base.RegularizedTomography(
            prepared.grid,
            train_obs,
            train_rays,
            lam=float(lambdas[0]),
            wt_r=prepared.wt_r,
            density_basis=prepared.density_basis,
        )
        held_tomo = base.RegularizedTomography(
            prepared.grid,
            [held_obs],
            [held_ray],
            lam=float(lambdas[0]),
            wt_r=prepared.wt_r,
            density_basis=prepared.density_basis,
        )

        for lam in lambdas:
            train_tomo.lam = float(lam)
            t0 = time.time()
            sol, info = train_tomo.solve(
                y_train,
                maxiter=int(prepared.args.maxiter),
                tol=float(prepared.args.tol),
                positivity=True,
            )
            elapsed = time.time() - t0
            y_pred_held = held_tomo.A_times(sol)
            st = weighted_stats(y_held, y_pred_held, held_tomo.W)
            rows.append({
                "holdout_index": ihold,
                "holdout_name": prepared.pb_paths[ihold].name,
                "holdout_group": base.tomography_observation_group_key(prepared.pb_paths[ihold]),
                "lambda": float(lam),
                "cg_info": int(info),
                "solve_seconds": float(elapsed),
                **st,
            })
            print(
                f"[STEP1-LOO] holdout={ihold}, lambda={float(lam):.6g}, "
                f"heldout_misfit_rms={st['misfit_rms']:.4e}, "
                f"heldout_weighted_rms_rel={st['weighted_rms_rel']:.4e}, info={info}"
            )

    write_rows_csv(output_dir / "step1_leave_one_image_out.csv", rows)
    print(f"[STEP1-LOO] Saved: {output_dir / 'step1_leave_one_image_out.csv'}")


# ============================================================
# Save final solution in the same spirit as base.main()
# ============================================================

def save_final_npz(
    prepared: PreparedProblem,
    tomo,
    solution_raw: np.ndarray,
    ne: np.ndarray,
    y_pred: np.ndarray,
    suggested_scale: float,
    final_lambda: float,
    output_path: Path,
) -> None:
    """Save a solution NPZ with extra STEP1 diagnostic metadata."""
    args = prepared.args
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ne_raw = tomo.solution_to_density(solution_raw)
    scale = suggested_scale if bool(args.apply_brightness_scale) else 1.0

    pb_scale_by_group = base.normalize_group_float_map(args.pb_scale_by_group, "pb_scale_by_group")
    r_use_min_by_group = base.normalize_group_float_map(args.r_use_min_by_group, "r_use_min_by_group")
    r_use_max_by_group = base.normalize_group_float_map(args.r_use_max_by_group, "r_use_max_by_group")

    freq_list = list(args.freq_mhz_list) if args.freq_mhz_list is not None else [float(args.freq_mhz)]

    obs_lonlat_deg = np.array(
        [(obs.lonlat_deg if obs.lonlat_deg is not None else (np.nan, np.nan)) for obs in prepared.obs_list],
        dtype=np.float64,
    )
    obs_group_keys = np.array([base.tomography_observation_group_key(p) for p in prepared.pb_paths], dtype="U64")
    obs_r_use_min = np.array([b[0] for b in prepared.obs_r_bounds], dtype=np.float64)
    obs_r_use_max = np.array([b[1] for b in prepared.obs_r_bounds], dtype=np.float64)
    obs_used_pixels = np.array([obs.idx_map.size for obs in prepared.obs_list], dtype=np.int64)
    
    render_camera_time = base.parse_target_datetime(args.target_time)
    render_camera_lonlat = earth_view_camera_lonlat_from_target_time(args.target_time)

    np.savez_compressed(
        output_path,
        ne=ne.astype(np.float32),
        ne_raw=ne_raw.astype(np.float32),
        solution_raw=solution_raw.astype(np.float32),
        density_basis=(
            prepared.density_basis.astype(np.float32)
            if prepared.density_basis is not None
            else np.ones_like(ne_raw, dtype=np.float32)
        ),
        scale_brightness=float(scale),
        suggested_scale_brightness=float(suggested_scale),
        apply_brightness_scale=bool(args.apply_brightness_scale),
        final_lambda=float(final_lambda),
        step1_generated_by="main_extended_multi_tomo.py",
        density_prior_model=str(args.density_prior_model),
        density_prior_scale=float(args.density_prior_scale),
        pb_scale_group_keys=np.array(list(pb_scale_by_group.keys()), dtype="U64"),
        pb_scale_group_values=np.array(list(pb_scale_by_group.values()), dtype=np.float64),
        r_use_min_group_keys=np.array(list(r_use_min_by_group.keys()), dtype="U64"),
        r_use_min_group_values=np.array(list(r_use_min_by_group.values()), dtype=np.float64),
        r_use_max_group_keys=np.array(list(r_use_max_by_group.keys()), dtype="U64"),
        r_use_max_group_values=np.array(list(r_use_max_by_group.values()), dtype=np.float64),
        pb_paths=np.array([str(p) for p in prepared.pb_paths], dtype="U2048"),
        pb_names=np.array([p.name for p in prepared.pb_paths], dtype="U256"),
        obs_group_keys=obs_group_keys,
        obs_lonlat_deg=obs_lonlat_deg,
        obs_r_use_min=obs_r_use_min.astype(np.float32),
        obs_r_use_max=obs_r_use_max.astype(np.float32),
        obs_used_pixels=obs_used_pixels,
        data_dir=str(args.data_dir),
        cor1a_data_dir=str(args.cor1a_data_dir),
        target_time=str(args.target_time),
        render_camera_is_earth_view=bool(True),
        render_camera_mode=str("forced_target_time_sub_earth"),
        render_camera_source=str("sunpy.coordinates.sun.L0_B0"),
        render_camera_time_utc=str(render_camera_time.strftime("%Y-%m-%dT%H:%M:%S")),
        render_camera_lon_deg=float(render_camera_lonlat[0]),
        render_camera_lat_deg=float(render_camera_lonlat[1]),
        render_camera_lonlat_frame=str("Carrington sub-Earth L0 / B0"),
        search_window_days=float(args.search_window_days),
        include_kcor_lasco=bool(args.include_kcor_lasco),
        include_cor1a=bool(args.include_cor1a),
        include_lasco_only=bool(args.include_lasco_only),
        deduplicate_pb_fits=bool(args.deduplicate_pb_fits),
        out_n=int(args.out_n),
        r_min=float(args.r_min),
        r_max=float(args.r_max),
        nr=int(args.nr),
        nth=int(args.nth),
        nph=int(args.nph),
        ds=float(args.ds),
        limb_u=float(args.limb_u),
        filt=bool(args.filt),
        despike_nsig=float(args.despike_nsig),
        despike_med=int(args.despike_med),
        pb_floor=str(args.pb_floor),
        dpa_deg=float(args.dpa_deg),
        r_use_min=float(args.r_use_min),
        r_use_max=float(args.r_use_max),
        hm=int(args.hm),
        wt_nr=bool(args.wt_nr),
        lam=float(final_lambda),
        q_low=float(args.q_low),
        width_pix=float(args.width_pix),
        maxiter=int(args.maxiter),
        tol=float(args.tol),
        use_temporal_despike=bool(args.use_temporal_despike),
        ne3dtomo_global_ybk=bool(args.ne3dtomo_global_ybk),
        calibration_reference_group=str(args.calibration_reference_group),
        harmonic=int(args.harmonic),
        freq_mhz_list=np.array(freq_list, dtype=np.float64),
        r_edges=prepared.grid.r_edges.astype(np.float32),
        th_edges=prepared.grid.th_edges.astype(np.float32),
        ph_edges=prepared.grid.ph_edges.astype(np.float32),
    )
    print(f"[OK] Saved STEP1 final solution NPZ: {output_path}")


def save_final_png(prepared: PreparedProblem, ne: np.ndarray, png_path: Path) -> None:
    args = prepared.args
    freq_list = list(args.freq_mhz_list) if args.freq_mhz_list is not None else [float(args.freq_mhz)]

    cam_ll = earth_view_camera_lonlat_from_target_time(args.target_time)
    target_dt = base.parse_target_datetime(args.target_time)

    print(
        "[INFO] STEP1 Earth-view rendering camera forced from target_time: "
        f"time_utc={target_dt:%Y-%m-%d %H:%M:%S}, "
        f"Carrington_L0={cam_ll[0]:.6f} deg, "
        f"B0={cam_ll[1]:.6f} deg"
    )

    base.visualize_isosurface(
        grid=prepared.grid,
        ne=ne,
        iso_freqs_mhz=freq_list,
        harmonic=int(args.harmonic),
        show_sun=True,
        opacity=0.5,
        camera_lonlat=cam_ll,
        show_gui=bool(args.show_gui),
        save_png=bool(args.save_png),
        png_path=Path(png_path),
        colors=getattr(args, "iso_colors", None),
    )

def earth_view_camera_lonlat_from_target_time(
    target_time: str,
) -> Tuple[float, float]:
    """
    Return the Earth-view rendering camera lon/lat at target_time.

    The returned longitude is the apparent Carrington longitude of disk center
    as seen from Earth, i.e. sub-Earth Carrington longitude L0.  The returned
    latitude is the apparent heliographic latitude of disk center B0.

    This is used only for final rendering metadata and PNG camera placement.
    It does not change the tomography inversion or the LOS geometry of individual
    input observations.
    """
    if target_time is None or str(target_time).strip() == "":
        raise ValueError(
            "target_time is required to force the final rendering camera to Earth view."
        )

    if hasattr(base, "earth_view_camera_lonlat_from_target_time"):
        return base.earth_view_camera_lonlat_from_target_time(target_time)

    try:
        import astropy.units as u
        from astropy.time import Time
        from sunpy.coordinates import sun
    except Exception as exc:
        raise ImportError(
            "For forced Earth-view rendering, SunPy is required to compute "
            "the sub-Earth Carrington longitude/latitude from target_time. "
            "Install it with e.g. `pip install sunpy`."
        ) from exc

    target_dt = base.parse_target_datetime(target_time)
    obstime = Time(target_dt, scale="utc")

    lon_deg = float(sun.L0(obstime).to_value(u.deg)) % 360.0
    lat_deg = float(sun.B0(obstime).to_value(u.deg))

    if not (np.isfinite(lon_deg) and np.isfinite(lat_deg)):
        raise ValueError(
            f"Invalid Earth-view Carrington lon/lat computed for target_time={target_time!r}: "
            f"lon={lon_deg}, lat={lat_deg}"
        )

    return lon_deg, lat_deg

def group_projection_fit_scales(
    prepared: PreparedProblem,
    tomo,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute weighted projection fit scales for each observing group.

    For each group, this returns the scalar s that minimizes
    || W * (s * y_pred - y_obs) ||.  It is useful for diagnosing relative
    photometric scale offsets, especially COR1A versus Earth-view merged pB.
    """
    groups: Dict[str, List[int]] = {}
    for i, path in enumerate(prepared.pb_paths):
        groups.setdefault(base.tomography_observation_group_key(path), []).append(i)

    fit_scales: Dict[str, float] = {}
    for group_key, indices in groups.items():
        y_obs_parts = []
        y_pred_parts = []
        w_parts = []
        for i in indices:
            sl = tomo.slices[i]
            y_obs_parts.append(np.asarray(prepared.y_obs)[sl])
            y_pred_parts.append(np.asarray(y_pred)[sl])
            w_parts.append(np.asarray(tomo.W)[sl])

        if y_obs_parts:
            fit_scales[group_key] = base.weighted_projection_scale(
                np.concatenate(y_obs_parts),
                np.concatenate(y_pred_parts),
                np.concatenate(w_parts),
                min_count=100,
            )

    return fit_scales


def clone_args_for_scenario(
    base_args: SimpleNamespace,
    scenario: Dict[str, object],
    scenario_dir: Path,
) -> SimpleNamespace:
    """
    Clone the base argument namespace and apply one comparison-scenario override.

    Only the scenario-specific parameters and output paths are changed.  This keeps
    each scenario isolated without modifying the original main_multi_tomo.py.
    """
    cloned = SimpleNamespace(**vars(base_args))

    for key, value in scenario.items():
        if key in ("name", "comparison_axis", "note"):
            continue
        setattr(cloned, key, value)

    scenario_name = str(scenario.get("name", "scenario"))

    # Keep scenario output isolated.  These are path changes only;
    # physical/inversion parameters remain those of the scenario.
    cloned.step1_output_dir = str(scenario_dir)

    if getattr(base_args, "save_prepped_dir", ""):
        cloned.save_prepped_dir = str(scenario_dir / "prepped")

    if getattr(base_args, "save_ne_npz", ""):
        cloned.save_ne_npz = str(scenario_dir / f"{scenario_name}_ne3d_solution.npz")

    if bool(getattr(base_args, "save_png", False)):
        cloned.png_path = str(scenario_dir / f"{scenario_name}_isosurface.png")

    return apply_defaults(cloned)


def write_comparison_key_metrics(output_root: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write a lightweight comparison CSV with the most important scalar metrics."""
    key_rows: List[Dict[str, object]] = []
    wanted = [
        "scenario",
        "comparison_axis",
        "lambda",
        "search_window_days",
        "use_density_prior",
        "density_prior_model",
        "density_prior_scale",
        "wt_nr",
        "ne3dtomo_global_ybk",
        "misfit",
        "fmax_mhz",
        "volume_rsun3",
        "components",
        "centroid_r_rsun",
        "largest_component_fraction",
        "fit_scale_earth_merged",
        "fit_scale_cor1a",
        "cor1a_over_earth_fit_scale",
        "regularization_norm",
        "median_obs_over_pred",
        "output_dir",
    ]

    for row in rows:
        key_rows.append({key: row.get(key, np.nan) for key in wanted})

    write_rows_csv(Path(output_root) / "step1_comparison_key_metrics.csv", key_rows)

def make_final_summary_row(
    prepared: PreparedProblem,
    tomo,
    solution_raw: np.ndarray,
    ne: np.ndarray,
    y_pred: np.ndarray,
    suggested_scale: float,
    final_lambda: float,
    output_dir: Path,
    scenario_name: str = "default",
    comparison_axis: str = "single",
    scan_results: Optional[Sequence[LambdaResult]] = None,
    lcurve_candidate: Optional[float] = None,
) -> Dict[str, object]:
    """
    Build one compact summary row for a final STEP1 solution.

    The row is intended for:
      - per-scenario step1_final_summary.csv
      - global step1_comparison_suite_summary.csv

    Required physical/comparison quantities included:
      - misfit
      - fmax_mhz
      - volume_rsun3
      - components
      - centroid_r_rsun
      - COR1A/Earth fit scale
    """
    output_dir = Path(output_dir)

    stats = weighted_stats(prepared.y_obs, y_pred, tomo.W)
    reg_norm = regularization_norm(solution_raw, prepared.grid, prepared.wt_r)

    fr = base.frequency_range_mhz_from_ne(ne, harmonic=int(prepared.args.harmonic))
    if fr is None:
        ne_min = ne_max = f_min = f_max = float("nan")
    else:
        ne_min, ne_max, f_min, f_max = [float(v) for v in fr]

    fit_scales = group_projection_fit_scales(prepared, tomo, y_pred)

    fit_scale_earth = finite_or_nan(fit_scales.get("earth_merged", np.nan))
    fit_scale_cor1a = finite_or_nan(fit_scales.get("cor1a", np.nan))
    if np.isfinite(fit_scale_earth) and fit_scale_earth > 0 and np.isfinite(fit_scale_cor1a):
        cor1a_over_earth = float(fit_scale_cor1a / fit_scale_earth)
    else:
        cor1a_over_earth = float("nan")

    # ------------------------------------------------------------
    # Target-frequency metrics.
    # For this project this is usually 33.8 MHz, harmonic=2.
    # ------------------------------------------------------------
    freq_list = (
        list(prepared.args.freq_mhz_list)
        if prepared.args.freq_mhz_list is not None
        else [float(prepared.args.freq_mhz)]
    )

    target_rows: Dict[str, float] = {}
    first_target = None
    for freq in freq_list:
        m = target_frequency_metrics(
            prepared.grid,
            ne,
            float(freq),
            int(prepared.args.harmonic),
        )
        if first_target is None:
            first_target = m

        prefix = f"f{float(freq):.3f}MHz_".replace(".", "p")
        for key, value in m.items():
            target_rows[prefix + key] = finite_or_nan(value)

    if first_target is None:
        first_target = {
            "volume_ge_target_rsun3": float("nan"),
            "n_components": float("nan"),
            "centroid_r_rsun": float("nan"),
            "largest_component_fraction": float("nan"),
        }

    # ------------------------------------------------------------
    # Lambda-scan summary, if available.
    # ------------------------------------------------------------
    scan_lambdas = []
    scan_best_misfit_lambda = float("nan")
    if scan_results:
        scan_lambdas = [float(r.lam) for r in scan_results]
        finite_scan = [r for r in scan_results if np.isfinite(r.data_misfit_norm)]
        if finite_scan:
            scan_best_misfit_lambda = float(min(finite_scan, key=lambda r: r.data_misfit_norm).lam)
            
    render_camera_time = base.parse_target_datetime(prepared.args.target_time)
    render_camera_lonlat = earth_view_camera_lonlat_from_target_time(prepared.args.target_time)

    row: Dict[str, object] = {
        "scenario": str(scenario_name),
        "comparison_axis": str(comparison_axis),
        "output_dir": str(output_dir),

        "target_time": str(getattr(prepared.args, "target_time", "")),
        "render_camera_is_earth_view": True,
        "render_camera_mode": "forced_target_time_sub_earth",
        "render_camera_source": "sunpy.coordinates.sun.L0_B0",
        "render_camera_time_utc": render_camera_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "render_camera_lon_deg": float(render_camera_lonlat[0]),
        "render_camera_lat_deg": float(render_camera_lonlat[1]),
        "render_camera_lonlat_frame": "Carrington sub-Earth L0 / B0",
        "render_camera_note": (
            "Final PNG camera is forced to Earth view at target_time. "
            "The reconstructed density grid remains a Sun-centered 3-D Carrington grid."
        ),
        "search_window_days": finite_or_nan(getattr(prepared.args, "search_window_days", np.nan)),
        "n_observations": int(len(prepared.pb_paths)),
        "n_measurements": int(np.asarray(prepared.y_obs).size),

        "out_n": int(getattr(prepared.args, "out_n", -1)),
        "r_min": finite_or_nan(getattr(prepared.args, "r_min", np.nan)),
        "r_max": finite_or_nan(getattr(prepared.args, "r_max", np.nan)),
        "nr": int(getattr(prepared.args, "nr", -1)),
        "nth": int(getattr(prepared.args, "nth", -1)),
        "nph": int(getattr(prepared.args, "nph", -1)),

        "lambda": float(final_lambda),
        "lcurve_lambda_candidate": finite_or_nan(lcurve_candidate),
        "scan_lambdas": json.dumps(scan_lambdas, ensure_ascii=False),
        "scan_best_misfit_lambda": finite_or_nan(scan_best_misfit_lambda),

        "density_prior_model": str(getattr(prepared.args, "density_prior_model", "")),
        "density_prior_scale": finite_or_nan(getattr(prepared.args, "density_prior_scale", np.nan)),
        "use_density_prior": str(getattr(prepared.args, "density_prior_model", "")).lower() not in ("", "none", "off", "false", "0"),

        "wt_nr": int(getattr(prepared.args, "wt_nr", 0)),
        "ne3dtomo_global_ybk": bool(getattr(prepared.args, "ne3dtomo_global_ybk", False)),
        "r_use_min": finite_or_nan(getattr(prepared.args, "r_use_min", np.nan)),
        "r_use_max": finite_or_nan(getattr(prepared.args, "r_use_max", np.nan)),
        "r_use_min_by_group": json.dumps(getattr(prepared.args, "r_use_min_by_group", {}), ensure_ascii=False),
        "pb_scale_by_group": json.dumps(getattr(prepared.args, "pb_scale_by_group", {}), ensure_ascii=False),

        # Required comparison quantities.
        "misfit": finite_or_nan(stats.get("misfit_rms")),
        "misfit_norm": finite_or_nan(stats.get("misfit_norm")),
        "weighted_rms_rel": finite_or_nan(stats.get("weighted_rms_rel")),
        "regularization_norm": finite_or_nan(reg_norm),
        "median_obs_over_pred": finite_or_nan(stats.get("median_obs_over_pred")),

        "ne_min_cm3": finite_or_nan(ne_min),
        "ne_max_cm3": finite_or_nan(ne_max),
        "f_min_mhz": finite_or_nan(f_min),
        "fmax_mhz": finite_or_nan(f_max),
        "f_max_mhz": finite_or_nan(f_max),

        "volume_rsun3": finite_or_nan(first_target.get("volume_ge_target_rsun3")),
        "components": int(first_target.get("n_components", 0)),
        "centroid_r_rsun": finite_or_nan(first_target.get("centroid_r_rsun")),
        "largest_component_fraction": finite_or_nan(first_target.get("largest_component_fraction")),

        "fit_scale_earth_merged": fit_scale_earth,
        "fit_scale_cor1a": fit_scale_cor1a,
        "cor1a_over_earth_fit_scale": finite_or_nan(cor1a_over_earth),

        "suggested_brightness_scale": finite_or_nan(suggested_scale),
        "apply_brightness_scale": bool(getattr(prepared.args, "apply_brightness_scale", False)),
    }

    row.update(target_rows)
    return row


def run_comparison_suite(args: SimpleNamespace) -> None:
    """
    Run a set of STEP1 comparison scenarios and save summary CSV files.

    Expected args fields:
      - step1_comparison_scenarios: list[dict]
      - step1_comparison_output_dir: optional output root
      - step1_output_dir: fallback output root for single-run diagnostics

    Each scenario may override any args attribute, for example:
      {"name": "lambda_10", "comparison_axis": "lambda", "lam": 10.0}
      {"name": "time_pm3d", "comparison_axis": "time_window", "search_window_days": 3.0}
      {"name": "density_prior_off_lambda_scan", "density_prior_model": "none", ...}
    """
    args = apply_defaults(args)

    scenarios = list(getattr(args, "step1_comparison_scenarios", []))
    if not scenarios:
        raise ValueError("step1_comparison_scenarios is empty. Define scenarios before calling run_comparison_suite().")

    # ------------------------------------------------------------
    # Determine comparison-suite output root.
    # ------------------------------------------------------------
    if getattr(args, "step1_comparison_output_dir", ""):
        output_root = ensure_dir(getattr(args, "step1_comparison_output_dir"))
    else:
        try:
            target_tag = base.parse_target_datetime(getattr(args, "target_time", "")).strftime("%Y%m%d_%H%M%S")
        except Exception:
            target_tag = now_tag()

        freq_list = (
            list(args.freq_mhz_list)
            if getattr(args, "freq_mhz_list", None) is not None
            else [float(getattr(args, "freq_mhz", 0.0))]
        )
        freq_tag = "-".join(str(float(f)).rstrip("0").rstrip(".") for f in freq_list)

        if getattr(args, "data_dir", ""):
            output_root = ensure_dir(
                Path(args.data_dir)
                / f"step1_comparison_suite_{target_tag}_{freq_tag}MHz"
            )
        else:
            output_root = ensure_dir(
                Path(getattr(args, "step1_output_dir", "step1_diagnostics")).parent
                / f"step1_comparison_suite_{target_tag}_{freq_tag}MHz"
            )

    print(f"[STEP1-COMP] Comparison suite output root: {output_root}")

    summary_rows: List[Dict[str, object]] = []

    # ------------------------------------------------------------
    # Main scenario loop.
    # ------------------------------------------------------------
    for index, scenario in enumerate(scenarios, start=1):
        scenario_name = str(scenario.get("name", f"scenario_{index:03d}"))
        comparison_axis = str(scenario.get("comparison_axis", "unspecified"))
        scenario_dir = ensure_dir(output_root / scenario_name)

        print("=" * 80)
        print(f"[STEP1-COMP] Scenario {index}/{len(scenarios)}: {scenario_name}")
        print(f"[STEP1-COMP] comparison_axis={comparison_axis}")
        print(f"[STEP1-COMP] scenario_dir={scenario_dir}")
        print("=" * 80)

        scenario_args = clone_args_for_scenario(args, scenario, scenario_dir)

        lambda_values = [float(v) for v in getattr(scenario_args, "step1_lambda_values", [scenario_args.lam])]
        if not lambda_values:
            lambda_values = [float(scenario_args.lam)]

        prepared = prepare_tomography_problem(scenario_args)

        # Save selected file list for each scenario.
        file_rows = []
        for i, path in enumerate(prepared.pb_paths):
            file_rows.append({
                "index": i,
                "name": path.name,
                "path": str(path),
                "group": base.tomography_observation_group_key(path),
                "datetime": str(base.parse_pb_filename_datetime(path)),
                "used_pixels": int(prepared.obs_list[i].idx_map.size),
                "r_use_min": prepared.obs_r_bounds[i][0],
                "r_use_max": prepared.obs_r_bounds[i][1],
            })
        write_rows_csv(scenario_dir / "step1_selected_observations.csv", file_rows)

        scan_results: List[LambdaResult] = []
        lcurve_candidate: Optional[float] = None

        if bool(getattr(scenario_args, "step1_run_lambda_scan", True)):
            scan_results, lcurve_candidate = run_lambda_scan(prepared, lambda_values, scenario_dir)
        else:
            print("[STEP1-COMP] Lambda scan disabled for this scenario.")

        final_lam_mode = str(getattr(scenario_args, "step1_final_lam_mode", "default")).strip().lower()
        if final_lam_mode in ("lcurve", "corner") and lcurve_candidate is not None:
            final_lam = float(lcurve_candidate)
        elif final_lam_mode in ("min_misfit", "min-data-misfit") and scan_results:
            final_lam = float(min(scan_results, key=lambda r: r.data_misfit_norm).lam)
        else:
            final_lam = float(scenario_args.lam)

        print(f"[STEP1-COMP] Final lambda mode={final_lam_mode!r}; final lambda={final_lam:.6g}")

        tomo, solution_raw, ne, y_pred, suggested_scale = run_final_diagnostics(
            prepared=prepared,
            lam=final_lam,
            output_dir=scenario_dir,
            save_residual_npy=bool(getattr(scenario_args, "step1_save_residual_npy", False)),
            save_residual_png=bool(getattr(scenario_args, "step1_save_residual_png", False)),
        )

        summary_row = make_final_summary_row(
            prepared=prepared,
            tomo=tomo,
            solution_raw=solution_raw,
            ne=ne,
            y_pred=y_pred,
            suggested_scale=suggested_scale,
            final_lambda=final_lam,
            output_dir=scenario_dir,
            scenario_name=scenario_name,
            comparison_axis=comparison_axis,
            scan_results=scan_results,
            lcurve_candidate=lcurve_candidate,
        )

        write_rows_csv(scenario_dir / "step1_final_summary.csv", [summary_row])
        print(f"[STEP1-COMP] Saved scenario summary: {scenario_dir / 'step1_final_summary.csv'}")

        if getattr(scenario_args, "save_ne_npz", ""):
            save_final_npz(
                prepared=prepared,
                tomo=tomo,
                solution_raw=solution_raw,
                ne=ne,
                y_pred=y_pred,
                suggested_scale=suggested_scale,
                final_lambda=final_lam,
                output_path=Path(scenario_args.save_ne_npz),
            )

        if bool(getattr(scenario_args, "save_png", False)):
            save_final_png(prepared, ne, Path(scenario_args.png_path))

        summary_rows.append(summary_row)

        # Save after every scenario to preserve partial progress if a later run fails.
        write_rows_csv(output_root / "step1_comparison_suite_summary.csv", summary_rows)
        write_comparison_key_metrics(output_root, summary_rows)
        print(f"[STEP1-COMP] Updated: {output_root / 'step1_comparison_suite_summary.csv'}")
        print(f"[STEP1-COMP] Updated: {output_root / 'step1_comparison_key_metrics.csv'}")

    print("[STEP1-COMP] Completed comparison suite.")
    print(f"[STEP1-COMP] Summary CSV: {output_root / 'step1_comparison_suite_summary.csv'}")
    print(f"[STEP1-COMP] Key metrics CSV: {output_root / 'step1_comparison_key_metrics.csv'}")

# ============================================================
# Main STEP1 workflow
# ============================================================

def main(args: SimpleNamespace) -> None:
    args = apply_defaults(args)

    step1_output_dir = ensure_dir(getattr(args, "step1_output_dir", "step1_diagnostics"))
    lambda_values = [float(v) for v in getattr(args, "step1_lambda_values", [args.lam])]
    if not lambda_values:
        lambda_values = [float(args.lam)]

    print("[STEP1] Preparing tomography problem using functions from main_multi_tomo.py...")
    prepared = prepare_tomography_problem(args)

    # Save selected file list for reproducibility.
    file_rows = []
    for i, p in enumerate(prepared.pb_paths):
        file_rows.append({
            "index": i,
            "name": p.name,
            "path": str(p),
            "group": base.tomography_observation_group_key(p),
            "datetime": str(base.parse_pb_filename_datetime(p)),
            "used_pixels": int(prepared.obs_list[i].idx_map.size),
            "r_use_min": prepared.obs_r_bounds[i][0],
            "r_use_max": prepared.obs_r_bounds[i][1],
        })
    write_rows_csv(step1_output_dir / "step1_selected_observations.csv", file_rows)

    scan_results: List[LambdaResult] = []
    lcurve_candidate: Optional[float] = None

    if bool(getattr(args, "step1_run_lambda_scan", True)):
        scan_results, lcurve_candidate = run_lambda_scan(prepared, lambda_values, step1_output_dir)
    else:
        print("[STEP1] Lambda scan disabled.")

    final_lam_mode = str(getattr(args, "step1_final_lam_mode", "default")).strip().lower()
    if final_lam_mode in ("lcurve", "corner") and lcurve_candidate is not None:
        final_lam = float(lcurve_candidate)
    elif final_lam_mode in ("min_misfit", "min-data-misfit") and scan_results:
        final_lam = float(min(scan_results, key=lambda r: r.data_misfit_norm).lam)
    else:
        final_lam = float(args.lam)

    print(f"[STEP1] Final lambda mode={final_lam_mode!r}; final lambda={final_lam:.6g}")

    tomo, solution_raw, ne, y_pred, suggested_scale = run_final_diagnostics(
        prepared=prepared,
        lam=final_lam,
        output_dir=step1_output_dir,
        save_residual_npy=bool(getattr(args, "step1_save_residual_npy", True)),
        save_residual_png=bool(getattr(args, "step1_save_residual_png", False)),
    )

    if bool(getattr(args, "step1_run_leave_one_image_out", False)):
        run_leave_one_image_out(
            prepared,
            lambdas=[final_lam] if bool(getattr(args, "step1_loo_final_lambda_only", True)) else lambda_values,
            output_dir=step1_output_dir,
            max_holdouts=getattr(args, "step1_loo_max_holdouts", None),
        )

    if args.save_ne_npz:
        save_final_npz(
            prepared=prepared,
            tomo=tomo,
            solution_raw=solution_raw,
            ne=ne,
            y_pred=y_pred,
            suggested_scale=suggested_scale,
            final_lambda=final_lam,
            output_path=Path(args.save_ne_npz),
        )

    if bool(args.save_png):
        if args.png_path:
            png_path = Path(args.png_path)
        else:
            base_name = Path(args.save_ne_npz).with_suffix("") if args.save_ne_npz else step1_output_dir / "step1_ne3d_solution"
            freq_list = list(args.freq_mhz_list) if args.freq_mhz_list is not None else [float(args.freq_mhz)]
            tag = "_".join([f"{float(f):.2f}" for f in freq_list])
            png_path = base_name.parent / f"{base_name.name}_iso_{tag}MHz_h{int(args.harmonic)}.png"
        print("[STEP1] Save png to", png_path)
        save_final_png(prepared, ne, png_path)

    print("[STEP1] Completed.")
    print(f"[STEP1] Diagnostics directory: {step1_output_dir}")


# ============================================================
# Editable settings
# ============================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Original tomography settings mirrored from the attached main_multi_tomo.py.
    # Edit here, not in main_multi_tomo.py.
    # ------------------------------------------------------------------
    PB_FITS = []

    DATA_DIR = "/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata"
    COR1A_DATA_DIR = "/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/pB/Rawdata"
    TARGET_TIME = "20220613_030000"
    SEARCH_WINDOW_DAYS = 7.0

    AUTO_FIND_PB_FITS = True
    INCLUDE_KCOR_LASCO = True
    INCLUDE_COR1A = True
    INCLUDE_LASCO_ONLY = False
    DEDUPLICATE_PB_FITS = True

    DEFAULT_LONLAT = ""
    LONLAT_FILE = ""

    OUT_N = 128

    R_MIN, R_MAX = 1.5, 4.0
    NR, NTH, NPH = 48, 48, 96

    DS = 0.01

    HM = 5

    WT_NR = 1
    LAM = 10.0
    Q_LOW = 0.0
    WIDTH_PIX = 1.0

    MAXITER = 10000
    TOL = 1e-4

    DESPIKE_NSIG = 5.0
    DESPIKE_MED = 5

    LIMB_U = base.DEFAULT_LIMB_U
    FILT = 1
    PB_FLOOR = ""

    DPA_DEG = 1.0
    R_USE_MIN, R_USE_MAX = 1.5, 4.0
    R_USE_MIN_BY_GROUP = {"cor1a": 1.5}
    R_USE_MAX_BY_GROUP = {}
    PB_SCALE_BY_GROUP = {}

    APPLY_BRIGHTNESS_SCALE = False
    DENSITY_PRIOR_MODEL = "saito_equatorial"
    DENSITY_PRIOR_SCALE = 2.8
    CALIBRATION_REFERENCE_GROUP = "earth_merged"

    USE_TEMPORAL_DESPIKE = False
    NE3DTOMO_GLOBAL_YBK = False
    SHOW_RAY_PROGRESS = True
    USE_RAY_CACHE = True

    SHOW_GUI = False
    HARMONIC = 2
    FREQ_MHZ_LIST = [33.8]
    ISO_COLORS = ["yellow"]

    TARGET_TAG = base.parse_target_datetime(TARGET_TIME).strftime("%Y%m%d_%H%M%S")
    WINDOW_TAG = f"pm{int(SEARCH_WINDOW_DAYS)}d"
    FREQ_TAG = "-".join(str(float(f)).rstrip("0").rstrip(".") for f in FREQ_MHZ_LIST)
    RAY_CACHE_DIR = (
        f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/"
        f"ray_cache_{TARGET_TAG}"
    )

    # ------------------------------------------------------------------
    # STEP 1 diagnostic settings
    # ------------------------------------------------------------------
    STEP1_OUTPUT_DIR = (
        f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/"
        f"step1_compare_{TARGET_TAG}_{WINDOW_TAG}_{FREQ_TAG}MHz"
    )

    STEP1_RUN_LAMBDA_SCAN = True

    # Scan around the original LAM=40.0.  Keep this list short at first because
    # each lambda requires one full CG solve.
    STEP1_LAMBDA_VALUES = [5, 10.0, 20.0]

    # "default" keeps the original LAM.  Use "lcurve" only after checking diagnostics.
    STEP1_FINAL_LAM_MODE = "default"

    # Save residual maps.  NPY is light and useful; PNG can create many files.
    STEP1_SAVE_RESIDUAL_NPY = False
    STEP1_SAVE_RESIDUAL_PNG = False

    # True leave-one-image-out is expensive.  Keep it False until the basic scan works.
    STEP1_RUN_LEAVE_ONE_IMAGE_OUT = False
    STEP1_LOO_FINAL_LAMBDA_ONLY = True
    STEP1_LOO_MAX_HOLDOUTS = None  # e.g., 3 for quick testing

    # Comparison suite.  When True, the script runs the scenario list below instead
    # of the single STEP1 diagnostic workflow.
    STEP1_RUN_COMPARISON_SUITE = True
    STEP1_COMPARISON_OUTPUT_DIR = (
        f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/"
        f"step1_comparison_suite_{TARGET_TAG}_{FREQ_TAG}MHz"
    )
    STEP1_COMPARISON_SCENARIOS = [
        # 1) Time-window comparison: +/-3 days vs +/-7 days.
        # {
        #     "name": "time_window_3d",
        #     "comparison_axis": "time_window",
        #     "search_window_days": 3.0,
        #     "step1_run_lambda_scan": False,
        # },
        # {
        #     "name": "time_window_7d",
        #     "comparison_axis": "time_window",
        #     "search_window_days": 7.0,
        #     "step1_run_lambda_scan": False,
        # },
        # 2) Fixed-lambda comparison.
        *[
            {
                "name": f"lambda_{lambda_para}",
                "comparison_axis": "lambda",
                "lam": float(lambda_para),
                "step1_lambda_values": [float(lambda_para)],
                "step1_run_lambda_scan": False,
            }
            for lambda_para in np.arange(1, 20)
        ],
        # {
        #     "name": "lambda_5",
        #     "comparison_axis": "lambda",
        #     "lam": 5.0,
        #     "step1_lambda_values": [5.0],
        #     "step1_run_lambda_scan": False,
        # },
        # {
        #     "name": "lambda_10",
        #     "comparison_axis": "lambda",
        #     "lam": 10.0,
        #     "step1_lambda_values": [10.0],
        #     "step1_run_lambda_scan": False,
        # },
        # {
        #     "name": "lambda_20",
        #     "comparison_axis": "lambda",
        #     "lam": 20.0,
        #     "step1_lambda_values": [20.0],
        #     "step1_run_lambda_scan": False,
        # },

        # 3) Density-prior on/off comparison.  The off case also runs a small lambda scan.
        # {
        #     "name": "density_prior_on",
        #     "comparison_axis": "density_prior",
        #     "density_prior_model": "saito_equatorial",
        #     "density_prior_scale": 2.8,
        #     "step1_run_lambda_scan": False,
        # },
        # {
        #     "name": "density_prior_off_lambda_scan",
        #     "comparison_axis": "density_prior",
        #     "density_prior_model": "none",
        #     "density_prior_scale": 1.0,
        #     "step1_lambda_values": [5.0, 10.0, 20.0],
        #     "step1_run_lambda_scan": True,
        #     "step1_final_lam_mode": "default",
        # },

        # 4) Regularization radial-weighting comparison.
        # {
        #     "name": "weighting_on",
        #     "comparison_axis": "weighting",
        #     "wt_nr": 1,
        #     "step1_run_lambda_scan": False,
        # },
        # {
        #     "name": "weighting_off",
        #     "comparison_axis": "weighting",
        #     "wt_nr": 0,
        #     "step1_run_lambda_scan": False,
        # },

        # 5) Density-prior scale comparison.
        # {
        #     "name": "density_prior_scale_1p0",
        #     "comparison_axis": "density_prior_scale",
        #     "density_prior_model": "saito_equatorial",
        #     "density_prior_scale": 1.0,
        #     "step1_run_lambda_scan": False,
        # },
        {
            "name": "density_prior_scale_2p0",
            "comparison_axis": "density_prior_scale",
            "density_prior_model": "saito_equatorial",
            "density_prior_scale": 2.0,
            "step1_run_lambda_scan": False,
        },
        # {
        #     "name": "density_prior_scale_2p8",
        #     "comparison_axis": "density_prior_scale",
        #     "density_prior_model": "saito_equatorial",
        #     "density_prior_scale": 2.8,
        #     "step1_run_lambda_scan": False,
        # },
        {
            "name": "density_prior_scale_3p0",
            "comparison_axis": "density_prior_scale",
            "density_prior_model": "saito_equatorial",
            "density_prior_scale": 3.0,
            "step1_run_lambda_scan": False,
        },
        # {
        #     "name": "density_prior_scale_4p0",
        #     "comparison_axis": "density_prior_scale",
        #     "density_prior_model": "saito_equatorial",
        #     "density_prior_scale": 4.0,
        #     "step1_run_lambda_scan": False,
        # },
    ]

    SAVE_PREPPED_DIR = (
        f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/"
        f"tomo_prepped_{TARGET_TAG}_{WINDOW_TAG}"
    )
    SAVE_NE_NPZ = (
        f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/ne_npz/"
        f"ne3d_solution_{TARGET_TAG}_{WINDOW_TAG}_{FREQ_TAG}MHz_step1.npz"
    )
    SAVE_PNG_PATH = (
        f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/output/multi-tomo/"
        f"tomo_{TARGET_TAG}_{WINDOW_TAG}_{FREQ_TAG}MHz_step1.png"
    )

    args = SimpleNamespace(
        pb_fits=PB_FITS,
        out_n=OUT_N,
        data_dir=DATA_DIR,
        cor1a_data_dir=COR1A_DATA_DIR,
        target_time=TARGET_TIME,
        search_window_days=SEARCH_WINDOW_DAYS,
        auto_find_pb_fits=AUTO_FIND_PB_FITS,
        include_kcor_lasco=INCLUDE_KCOR_LASCO,
        include_cor1a=INCLUDE_COR1A,
        include_lasco_only=INCLUDE_LASCO_ONLY,
        deduplicate_pb_fits=DEDUPLICATE_PB_FITS,

        default_lonlat=DEFAULT_LONLAT,
        lonlat_file=LONLAT_FILE,

        r_min=R_MIN,
        r_max=R_MAX,
        nr=NR,
        nth=NTH,
        nph=NPH,
        ds=DS,
        limb_u=LIMB_U,

        filt=FILT,
        despike_nsig=DESPIKE_NSIG,
        despike_med=DESPIKE_MED,
        pb_floor=PB_FLOOR,
        dpa_deg=DPA_DEG,
        r_use_min=R_USE_MIN,
        r_use_max=R_USE_MAX,
        r_use_min_by_group=R_USE_MIN_BY_GROUP,
        r_use_max_by_group=R_USE_MAX_BY_GROUP,
        pb_scale_by_group=PB_SCALE_BY_GROUP,
        hm=HM,
        wt_nr=WT_NR,

        lam=LAM,
        q_low=Q_LOW,
        width_pix=WIDTH_PIX,
        maxiter=MAXITER,
        tol=TOL,
        apply_brightness_scale=APPLY_BRIGHTNESS_SCALE,
        density_prior_model=DENSITY_PRIOR_MODEL,
        density_prior_scale=DENSITY_PRIOR_SCALE,
        calibration_reference_group=CALIBRATION_REFERENCE_GROUP,

        use_temporal_despike=USE_TEMPORAL_DESPIKE,
        ne3dtomo_global_ybk=NE3DTOMO_GLOBAL_YBK,
        show_ray_progress=SHOW_RAY_PROGRESS,
        use_ray_cache=USE_RAY_CACHE,
        ray_cache_dir=RAY_CACHE_DIR,

        save_prepped_dir=SAVE_PREPPED_DIR,
        save_ne_npz=SAVE_NE_NPZ,

        show_gui=SHOW_GUI,
        freq_mhz_list=FREQ_MHZ_LIST,
        harmonic=HARMONIC,
        iso_colors=ISO_COLORS,

        save_png=True,
        png_path=SAVE_PNG_PATH,

        step1_output_dir=STEP1_OUTPUT_DIR,
        step1_run_lambda_scan=STEP1_RUN_LAMBDA_SCAN,
        step1_lambda_values=STEP1_LAMBDA_VALUES,
        step1_final_lam_mode=STEP1_FINAL_LAM_MODE,
        step1_save_residual_npy=STEP1_SAVE_RESIDUAL_NPY,
        step1_save_residual_png=STEP1_SAVE_RESIDUAL_PNG,
        step1_run_leave_one_image_out=STEP1_RUN_LEAVE_ONE_IMAGE_OUT,
        step1_loo_final_lambda_only=STEP1_LOO_FINAL_LAMBDA_ONLY,
        step1_loo_max_holdouts=STEP1_LOO_MAX_HOLDOUTS,
        step1_run_comparison_suite=STEP1_RUN_COMPARISON_SUITE,
        step1_comparison_output_dir=STEP1_COMPARISON_OUTPUT_DIR,
        step1_comparison_scenarios=STEP1_COMPARISON_SCENARIOS,
    )

    if STEP1_RUN_COMPARISON_SUITE:
        run_comparison_suite(args)
    else:
        main(args)
