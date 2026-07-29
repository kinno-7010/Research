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

5. Current main_multi_tomo.py compatibility
   - use the same Thomson-kernel normalization and ray-cache key
   - use the same density-prior and positivity settings
   - apply the same Earth-reference / COR1A relative radiometric calibration
   - support joint Earth-view + COR1A, Earth-view-only, and COR1A-only scenarios
   - select each single-view data set as a subset of the corresponding joint time-window selection

Important
---------
Put this file in the same directory as main_multi_tomo.py, or set BASE_MODULE_PATH below.
The original main_multi_tomo.py is treated as a read-only source module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import copy
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
    ray_cache_hits: int = 0
    ray_cache_memory_hits: int = 0
    ray_cache_disk_hits: int = 0
    ray_cache_misses: int = 0
    ray_cache_load_failures: int = 0
    ray_cache_disabled: bool = False
    ray_cache_keys: List[str] = field(default_factory=list)



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
    normal_equation_relative_residual: float
    data_objective: float
    regularization_objective: float
    total_objective: float
    solver_iterations: int = 0
    solver_used_preconditioner: bool = False
    group_forward_gains: Dict[str, float] = field(default_factory=dict)
    group_data_corrections: Dict[str, float] = field(default_factory=dict)
    cross_calibration_iterations: int = 0
    cross_calibration_converged: bool = False
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

RAY_CACHE_VERSION = getattr(base, "RAY_CACHE_VERSION", "ray_cache_v2")
_RAY_MEMORY_CACHE: Dict[str, object] = {}
_FORWARD_MATRIX_MEMORY_CACHE: "OrderedDict[str, object]" = OrderedDict()


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
    thomson_normalize_msb: bool = True,
    thomson_kernel_scale: float = 1.0,
) -> str:
    """Delegate to the current main_multi_tomo.py cache-key implementation."""
    return base.ray_cache_key(
        obs=obs,
        pb_path=pb_path,
        grid=grid,
        ds_rsun=ds_rsun,
        r_min=r_min,
        r_max=r_max,
        limb_u=limb_u,
        thomson_normalize_msb=thomson_normalize_msb,
        thomson_kernel_scale=thomson_kernel_scale,
    )




def ray_cache_path(cache_dir: Path | str, key: str) -> Path:
    return base.ray_cache_path(cache_dir, key)




def load_cached_ray(key: str, cache_dir: Path | str = "") -> Optional[object]:
    """Delegate to the current main_multi_tomo.py ray-cache loader."""
    return base.load_cached_ray(key, cache_dir)




def save_cached_ray(key: str, ray: object, cache_dir: Path | str = "") -> None:
    """Delegate to the current main_multi_tomo.py ray-cache writer."""
    base.save_cached_ray(key, ray, cache_dir)



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


def _default_group_forward_gains(prepared: PreparedProblem) -> Dict[str, float]:
    return {
        key: 1.0
        for key in sorted(
            {base.tomography_observation_group_key(Path(path)) for path in prepared.pb_paths}
        )
    }


def _resolve_cross_calibration_radial_range(
    prepared: PreparedProblem,
) -> Tuple[Optional[float], Optional[float]]:
    """Resolve the explicit or common radial range used for group-gain fitting."""
    r_min = base._optional_positive_float(
        getattr(prepared.args, "cross_calibration_r_min", "")
    )
    r_max = base._optional_positive_float(
        getattr(prepared.args, "cross_calibration_r_max", "")
    )
    if r_min is None:
        r_min = max(float(bounds[0]) for bounds in prepared.obs_r_bounds)
    if r_max is None:
        r_max = min(float(bounds[1]) for bounds in prepared.obs_r_bounds)
    if r_min >= r_max:
        raise ValueError(
            f"Cross-calibration has no common radial range: {r_min} >= {r_max} Rsun"
        )
    return float(r_min), float(r_max)



def configure_group_cross_calibration(
    prepared: PreparedProblem,
    tomo,
    initial_group_forward_gains=None,
) -> Tuple[Dict[str, float], List[Dict[str, object]], Optional[np.ndarray]]:
    """
    Apply the Earth-reference/COR1A gain model used by main_multi_tomo.py.

    When automatic calibration is disabled, ``fixed_group_forward_gains`` may
    be supplied to compare a fixed relative calibration against both automatic
    calibration and the uncalibrated gain=1 case.  Earth-view and STEREO-A
    observations remain in the inversion in all three cases.
    """
    args = prepared.args
    gains = _default_group_forward_gains(prepared)
    history: List[Dict[str, object]] = []
    seed: Optional[np.ndarray] = None

    if bool(getattr(args, "auto_cross_calibrate_groups", False)):
        cal_r_min, cal_r_max = _resolve_cross_calibration_radial_range(prepared)
        initial = (
            initial_group_forward_gains
            if initial_group_forward_gains is not None
            else getattr(args, "cross_calibration_initial_gain_by_group", {})
        )
        print(
            "[CAL] Calibration solver settings: "
            f"maxiter={int(getattr(args, 'cross_calibration_solver_maxiter', args.maxiter))}, "
            f"tol={float(getattr(args, 'cross_calibration_solver_tol', args.tol)):.3g}, "
            f"preconditioner={bool(getattr(args, 'solver_use_preconditioner', True))}"
        )
        gains, history, seed = base.run_group_cross_calibration(
            tomo=tomo,
            pb_paths=prepared.pb_paths,
            y_obs=prepared.y_obs,
            reference_group=str(args.calibration_reference_group),
            initial_group_forward_gains=initial,
            max_iterations=int(args.cross_calibration_max_iterations),
            convergence_tol=float(args.cross_calibration_tolerance),
            damping=float(args.cross_calibration_damping),
            gain_min=float(args.cross_calibration_gain_min),
            gain_max=float(args.cross_calibration_gain_max),
            r_min=cal_r_min,
            r_max=cal_r_max,
            min_count=int(args.cross_calibration_min_count),
            clip_sigma=float(args.cross_calibration_clip_sigma),
            solve_maxiter=int(getattr(args, "cross_calibration_solver_maxiter", args.maxiter)),
            solve_tol=float(getattr(args, "cross_calibration_solver_tol", args.tol)),
            positivity_method=str(args.positivity_method),
            solve_use_preconditioner=bool(getattr(args, "solver_use_preconditioner", True)),
            solve_preconditioner_floor=float(getattr(args, "solver_preconditioner_floor", 1e-12)),
        )
        calibration_mode = "automatic"
    else:
        fixed = base.normalize_group_float_map(
            getattr(args, "fixed_group_forward_gains", {}),
            "fixed_group_forward_gains",
        )
        gains.update(fixed)
        gains[str(args.calibration_reference_group)] = 1.0
        for key, value in gains.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid fixed forward gain for {key!r}: {value}")
        tomo.set_measurement_scale(
            base.build_group_measurement_scale_vector(prepared.pb_paths, tomo, gains)
        )
        calibration_mode = "fixed" if fixed else "none"
        if fixed:
            print(f"[CAL] Automatic calibration disabled; fixed forward gains applied: {gains}")
        else:
            print("[CAL] Automatic calibration disabled; all group forward gains remain 1.0.")

    corrections = {key: 1.0 / value for key, value in gains.items()}
    converged = bool(
        history
        and finite_or_nan(history[-1].get("max_abs_log_gain_change"))
        <= float(args.cross_calibration_tolerance)
    )
    if calibration_mode in ("fixed", "none"):
        converged = True

    tomo.cross_calibration_mode = calibration_mode
    tomo.cross_calibration_forward_gains = dict(gains)
    tomo.cross_calibration_data_corrections = dict(corrections)
    tomo.cross_calibration_history = list(history)
    tomo.cross_calibration_converged = converged
    tomo.cross_calibration_seed = seed

    args.cross_calibration_final_forward_gains = dict(gains)
    args.cross_calibration_final_data_corrections = dict(corrections)
    args.cross_calibration_history = list(history)

    print(f"[CAL] Calibration mode: {calibration_mode}")
    print(f"[CAL] Final forward gains (model -> observed pB): {gains}")
    print(
        "[CAL] Equivalent data corrections (observed pB -> reference scale): "
        f"{corrections}"
    )
    return gains, history, seed






def apply_defaults(args: SimpleNamespace) -> SimpleNamespace:
    """Add missing arguments with defaults synchronized to main_multi_tomo.py."""
    defaults = dict(
        pb_fits=[],
        out_n=128,

        default_lonlat="",
        lonlat_file="",

        r_min=1.5,
        r_max=4.0,
        nr=0,
        nth=60,
        nph=120,
        auto_grid_max_m_over_n=1.5,

        ds=0.02,
        limb_u=base.DEFAULT_LIMB_U,
        limb_u_mode="instrument_bandpass",
        limb_u_use_allen=False,
        limb_u_bandpass_nm_by_instrument=dict(base.DEFAULT_LIMB_BANDPASS_NM),
        limb_u_override_by_instrument={},
        limb_u_bandpass_samples=401,
        limb_u_weight_hdu_names=base.DEFAULT_LIMB_WEIGHT_HDU_NAMES,

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
        lambda_scan_values=[],
        lambda_select_mode="fixed",
        q_low=0.0,
        width_pix=2.0,
        maxiter=10000,
        tol=1e-3,
        solver_use_preconditioner=True,
        solver_preconditioner_floor=1e-12,
        positivity_method="clip",
        apply_brightness_scale=False,
        use_density_prior=True,
        density_prior_model="none",
        density_prior_scale=1.0,
        density_prior_file="",
        thomson_normalize_msb=True,
        thomson_kernel_scale=1.0,
        run_pb_unit_diagnostics=False,
        pb_diagnostic_paths=[],
        calibration_reference_group="earth_merged",
        auto_cross_calibrate_groups=False,
        fixed_group_forward_gains={},
        cross_calibration_initial_gain_by_group={},
        cross_calibration_max_iterations=3,
        cross_calibration_tolerance=0.01,
        cross_calibration_solver_maxiter=5000,
        cross_calibration_solver_tol=3e-3,
        cross_calibration_damping=0.7,
        cross_calibration_gain_min=0.25,
        cross_calibration_gain_max=4.0,
        cross_calibration_r_min="",
        cross_calibration_r_max="",
        cross_calibration_min_count=100,
        cross_calibration_clip_sigma=4.0,
        cross_calibration_recalibrate_after_lambda_selection=True,

        data_dir="",
        cor1a_data_dir="",
        target_time="",
        search_window_days=7.0,
        auto_find_pb_fits=False,
        exclude_earth_times=[],
        keep_cor1a_for_excluded_earth_times=True,
        observation_groups=[],
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
        save_summary_csv=False,
        summary_csv_path="",

        show_gui=True,
        freq_mhz=25.0,
        freq_mhz_list=None,
        harmonic=1,
        iso_colors=None,
        save_png=True,
        png_path="",

        step1_output_dir="step1_diagnostics",
        step1_run_lambda_scan=True,
        step1_lambda_values=[],
        step1_final_lam_mode="default",
        step1_lambda_scan_descending=True,
        step1_lambda_scan_warm_start=True,
        step1_fast_lambda_suite=True,
        step1_forward_matrix_cache_max_entries=2,
        step1_skip_completed_scenarios=True,
        step1_skip_completed_unconverged=False,
        step1_save_residual_npy=True,
        step1_save_residual_png=False,
        step1_radial_residual_bins=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        step1_save_calibration_diagnostics=True,
        step1_calibration_radial_bins=[1.7, 2.0, 2.5, 3.0, 3.5],
        step1_compute_coverage=False,
        step1_save_coverage_npz=False,
        step1_coverage_low_relative_threshold=1e-3,
        step1_save_target_mask_npz=False,
        step1_run_leave_one_image_out=False,
        step1_loo_final_lambda_only=True,
        step1_loo_max_holdouts=None,
        step1_run_block_holdout=False,
        step1_block_holdout_days=1.0,
        step1_block_holdout_max_blocks=6,
        step1_run_comparison_suite=False,
        step1_comparison_output_dir="",
        step1_comparison_scenarios=[],
        step1_overlap_reference_scenario="lambda_5",
        step1_write_overlap_metrics=True,
    )
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, copy.deepcopy(v))

    if not bool(getattr(args, "use_density_prior", True)):
        args.density_prior_model = "none"

    return args





def prepare_tomography_problem(args: SimpleNamespace) -> PreparedProblem:
    """Prepare observations, grid, measurements, cached rays, and regularization weights."""
    args = apply_defaults(args)

    if bool(getattr(args, "run_pb_unit_diagnostics", False)):
        diag_paths = [Path(p) for p in getattr(args, "pb_diagnostic_paths", [])]
        base.print_pb_calibration_report(diag_paths)

    if bool(getattr(args, "thomson_normalize_msb", True)):
        print(
            "[INFO] Thomson kernel MSB normalization enabled. "
            f"limb_u_mode={str(getattr(args, 'limb_u_mode', 'fixed'))!r}; "
            "instrument-bandpass kernels are averaged in A/B coefficient space."
        )
    else:
        print(
            "[INFO] Thomson kernel MSB normalization disabled; "
            f"limb_u_mode={str(getattr(args, 'limb_u_mode', 'fixed'))!r}."
        )
    if abs(float(getattr(args, "thomson_kernel_scale", 1.0)) - 1.0) > 1e-12:
        print(f"[INFO] Explicit Thomson kernel scale applied: {float(args.thomson_kernel_scale):.6g}")

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
            exclude_earth_times=getattr(args, "exclude_earth_times", []),
            keep_cor1a_for_excluded_earth_times=bool(
                getattr(args, "keep_cor1a_for_excluded_earth_times", True)
            ),
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

    observation_groups = getattr(args, "observation_groups", [])
    if isinstance(observation_groups, str):
        observation_groups = [
            item.strip()
            for item in observation_groups.split(",")
            if item.strip()
        ]
    selected_group_keys = {
        str(item).strip()
        for item in (observation_groups or [])
        if str(item).strip()
    }
    if selected_group_keys:
        before_count = len(pb_paths)
        pb_paths = [
            path
            for path in pb_paths
            if base.tomography_observation_group_key(path) in selected_group_keys
        ]
        args.pb_fits = [str(path) for path in pb_paths]
        print(
            "[INFO] Observation-group filter applied: "
            f"groups={sorted(selected_group_keys)}, "
            f"selected={len(pb_paths)}/{before_count}"
        )
        if not pb_paths:
            raise ValueError(
                "No pB FITS files remain after observation_groups filtering: "
                f"{sorted(selected_group_keys)}"
            )

    r_use_min_by_group = base.normalize_group_float_map(args.r_use_min_by_group, "r_use_min_by_group")
    r_use_max_by_group = base.normalize_group_float_map(args.r_use_max_by_group, "r_use_max_by_group")
    pb_scale_by_group = base.normalize_group_float_map(args.pb_scale_by_group, "pb_scale_by_group")

    pb_overrides = {}
    if args.filt and len(pb_paths) >= 2 and bool(args.use_temporal_despike):
        pb_overrides = base.build_ne3dtomo_temporal_despike_overrides(
            pb_paths=pb_paths, out_n=int(args.out_n), nsig=float(args.despike_nsig)
        )
        if not pb_overrides:
            print("[INFO] Temporal despike requested, but no homogeneous group was usable; applying spatial despike per image only.")
    elif args.filt and len(pb_paths) >= 2:
        print("[INFO] Global temporal despike disabled; applying spatial despike per image only.")

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
            raise ValueError(f"{p.name} uses r_use_min={obs_r_use_min} Rsun, smaller than reconstruction r_min={args.r_min} Rsun.")
        if obs_r_use_max > float(args.r_max) + 1e-8:
            raise ValueError(f"{p.name} uses r_use_max={obs_r_use_max} Rsun, larger than reconstruction r_max={args.r_max} Rsun.")

        obs = base.build_observation(
            pb_fits=p,
            out_n=int(args.out_n),
            pb_override=pb_overrides.get(p),
            apply_spatial_despike=(p not in pb_overrides),
            r_use_min=obs_r_use_min,
            r_use_max=obs_r_use_max,
            limb_u=float(args.limb_u),
            limb_u_mode=str(args.limb_u_mode),
            limb_u_use_allen=bool(args.limb_u_use_allen),
            limb_u_bandpass_nm_by_instrument=args.limb_u_bandpass_nm_by_instrument,
            limb_u_override_by_instrument=args.limb_u_override_by_instrument,
            limb_u_bandpass_samples=int(args.limb_u_bandpass_samples),
            limb_u_weight_hdu_names=args.limb_u_weight_hdu_names,
            thomson_normalize_msb=bool(args.thomson_normalize_msb),
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
            print(f"[INFO] Applied explicit pB calibration scale to {p.name}: group={group_key}, scale={pb_scale:.6g}")

        obs_list.append(obs)
        obs_r_bounds.append((obs_r_use_min, obs_r_use_max))
        rho = np.hypot(obs.x, obs.y)
        print(
            f"[GEOM] {p.name}: group={group_key}, lonlat={obs.lonlat_deg}, "
            f"used_pixels={obs.idx_map.size}, r_use={obs_r_use_min:.3f}..{obs_r_use_max:.3f} Rs, "
            f"pb_scale={pb_scale:.6g}, rho={np.nanmin(rho):.3f}..{np.nanmax(rho):.3f} Rs, "
            f"limb_model={obs.limb_u_model}, components={obs.limb_component_names}"
        )
        rgrid, ybk, _ = base.ybk_profile_fft(
            pb=obs.pb, hdr=obs.hdr, rmin=obs_r_use_min, rmax=obs_r_use_max,
            dpa_deg=float(args.dpa_deg), nr=240, hm=int(args.hm),
            width_pix=float(args.width_pix), q_low=float(args.q_low),
        )
        local_ybk_list.append((rgrid, ybk))

    if bool(getattr(args, "ne3dtomo_global_ybk", False)):
        grouped_indices: Dict[str, List[int]] = {}
        for i, p in enumerate(pb_paths):
            grouped_indices.setdefault(base.tomography_observation_group_key(p), []).append(i)
        ybk_list: List[Tuple[np.ndarray, np.ndarray]] = [local_ybk_list[i] for i in range(len(obs_list))]
        for key, indices_group in grouped_indices.items():
            group_obs = [obs_list[i] for i in indices_group]
            group_bounds = [obs_r_bounds[i] for i in indices_group]
            group_rmin, group_rmax = group_bounds[0]
            if any((abs(b0 - group_rmin) > 1e-8 or abs(b1 - group_rmax) > 1e-8) for b0, b1 in group_bounds):
                raise ValueError(f"Group {key!r} has mixed r_use bounds; global ybk requires one radial range per group.")
            rgrid_g, ybk_g, pb_noise_g = base.ybk_profile_fft_stack(
                observations=group_obs, rmin=group_rmin, rmax=group_rmax,
                dpa_deg=float(args.dpa_deg), nr=240, hm=int(args.hm),
                width_pix=float(args.width_pix), q_low=float(args.q_low),
            )
            for i in indices_group:
                obs_list[i] = base.update_observation_weights_from_ybk(
                    obs=obs_list[i], rgrid=rgrid_g, ybk=ybk_g,
                    pb_noise=pb_noise_g, pb_floor=args.pb_floor,
                )
                ybk_list[i] = (rgrid_g, ybk_g)
            print(f"[INFO] Ne3dTomo-style global ybk(r) applied for group {key!r} (n={len(indices_group)}).")
    else:
        ybk_list = local_ybk_list

    y_list: List[np.ndarray] = []
    for p, obs in zip(pb_paths, obs_list):
        y_vec = obs.pb.ravel()[obs.idx_map]
        y_list.append(y_vec)
        vv = y_vec[np.isfinite(y_vec)]
        if vv.size:
            print(f"[INFO] {p.name}: pB (used pixels) min/med/max = {np.min(vv):.3e} / {np.median(vv):.3e} / {np.max(vv):.3e}")
    y_obs = np.concatenate(y_list) if y_list else np.array([], dtype=np.float64)
    if y_obs.size == 0 or not np.any(np.isfinite(y_obs)):
        raise ValueError("y_obs is empty or all-NaN. Check masks and preprocessing.")

    args.nr, args.nth, args.nph, achieved_m_over_n = (
        base.choose_nr_for_measurement_to_voxel_ratio(
            n_measurements=int(y_obs.size),
            max_m_over_n=float(args.auto_grid_max_m_over_n),
        )
    )
    n_voxels = int(args.nr * args.nth * args.nph)
    print(
        "[AUTO-GRID] "
        f"M={int(y_obs.size)}, grid={args.nr}x{args.nth}x{args.nph}, "
        f"N={n_voxels}, M/N={achieved_m_over_n:.6f}, "
        f"constraint=M/N<={float(args.auto_grid_max_m_over_n):.6g}"
    )

    r_edges = np.linspace(float(args.r_min), float(args.r_max), int(args.nr) + 1)
    th_edges = np.linspace(0.0, np.pi, int(args.nth) + 1)
    ph_edges = np.linspace(0.0, 2.0 * np.pi, int(args.nph) + 1)
    grid = base.SphericalGrid(r_edges=r_edges, th_edges=th_edges, ph_edges=ph_edges)

    density_basis = base.density_basis_from_grid(
        grid,
        model=str(args.density_prior_model),
        scale=float(args.density_prior_scale),
        prior_file=str(getattr(args, "density_prior_file", "")),
    )
    if density_basis is not None:
        prior_source = str(getattr(args, "density_prior_file", "") or "")
        source_text = f", file={prior_source!r}" if prior_source else ""
        print(
            "[INFO] Density prior enabled: "
            f"model={str(args.density_prior_model)!r}, "
            f"scale={float(args.density_prior_scale):.6g}{source_text}, "
            "solving ne = prior * q."
        )
    else:
        print("[INFO] Density prior disabled: solving absolute electron density ne.")

    rays: List[object] = []
    n_obs = len(obs_list)
    use_ray_cache = bool(getattr(args, "use_ray_cache", True))
    ray_cache_dir = str(getattr(args, "ray_cache_dir", "") or "")
    cache_hits = cache_memory_hits = cache_disk_hits = cache_misses = cache_load_failures = 0
    ray_cache_keys: List[str] = []
    if use_ray_cache and ray_cache_dir:
        ensure_dir(ray_cache_dir)
        print(f"[CACHE] Ray cache enabled: disk+memory, directory={ray_cache_dir}")
    elif use_ray_cache:
        print("[CACHE] Ray cache enabled: memory only; cache will not survive process restart.")
    else:
        print("[CACHE-DISABLED] Ray cache is disabled; every observation will rebuild its rays.")
    print("[CACHE] Note: ray cache reuses LOS/Thomson ray bundles only; every lambda still requires a solver run.")

    memory_cache = getattr(base, "_RAY_MEMORY_CACHE", {})
    for i, (obs, p) in enumerate(zip(obs_list, pb_paths), start=1):
        cache_key = None
        ray = None
        source = None
        disk_file = None
        disk_existed = False
        memory_existed = False
        cache_key = base.ray_cache_key(
            obs=obs, pb_path=p, grid=grid, ds_rsun=float(args.ds),
            r_min=float(args.r_min), r_max=float(args.r_max), limb_u=float(args.limb_u),
            thomson_normalize_msb=bool(args.thomson_normalize_msb),
            thomson_kernel_scale=float(args.thomson_kernel_scale),
        )
        ray_cache_keys.append(cache_key)
        if use_ray_cache:
            memory_existed = cache_key in memory_cache
            if ray_cache_dir:
                disk_file = base.ray_cache_path(ray_cache_dir, cache_key)
                disk_existed = Path(disk_file).exists()
            ray = base.load_cached_ray(cache_key, ray_cache_dir)
            if ray is not None:
                cache_hits += 1
                if memory_existed:
                    cache_memory_hits += 1
                    source = "memory"
                else:
                    cache_disk_hits += 1
                    source = "disk"
                nonempty = sum(1 for idx in ray.vox_idx if idx.size > 0)
                print(f"[CACHE-HIT:{source.upper()}] {i}/{n_obs}: {p.name} (non-empty rays={nonempty}/{len(ray.vox_idx)})", flush=True)
            else:
                cache_misses += 1
                if disk_existed:
                    cache_load_failures += 1
                    print(f"[CACHE-MISS] {i}/{n_obs}: {p.name}; cache file exists but could not be loaded (version/corruption mismatch): {disk_file}", flush=True)
                else:
                    location = str(disk_file) if disk_file is not None else "memory cache"
                    print(f"[CACHE-MISS] {i}/{n_obs}: {p.name}; no matching cache entry at {location}", flush=True)

        if ray is None:
            if bool(getattr(args, "show_ray_progress", True)):
                print(f"[INFO] Building rays {i}/{n_obs}: {p.name} (used pixels={obs.idx_map.size})", flush=True)
            ray = base.build_rays_for_observation(
                obs=obs, grid=grid, ds_rsun=float(args.ds),
                r_min=float(args.r_min), r_max=float(args.r_max), limb_u=float(args.limb_u),
                thomson_normalize_msb=bool(args.thomson_normalize_msb),
                thomson_kernel_scale=float(args.thomson_kernel_scale),
            )
            if use_ray_cache and cache_key is not None:
                base.save_cached_ray(cache_key, ray, ray_cache_dir)
                if ray_cache_dir:
                    print(f"[CACHE-WRITE] {p.name}: {base.ray_cache_path(ray_cache_dir, cache_key)}", flush=True)
            if bool(getattr(args, "show_ray_progress", True)):
                nonempty = sum(1 for idx in ray.vox_idx if idx.size > 0)
                print(f"[INFO] Finished rays {i}/{n_obs}: {p.name} (non-empty rays={nonempty}/{len(ray.vox_idx)})", flush=True)
        rays.append(ray)

    if use_ray_cache:
        print(
            f"[CACHE-SUMMARY] total={n_obs}, hits={cache_hits} "
            f"(memory={cache_memory_hits}, disk={cache_disk_hits}), misses={cache_misses}, "
            f"load_failures={cache_load_failures}"
        )
        if cache_hits == 0:
            print("[WARN][CACHE-NOT-USED] No ray bundle was reused in this scenario; all rays were rebuilt.")
        elif cache_misses > 0:
            print("[WARN][CACHE-PARTIAL] Ray cache was used only partially; unmatched observations were rebuilt.")
        else:
            print("[CACHE-OK] All ray bundles were reused from cache.")

    wt_r = None
    if int(args.wt_nr):
        r_cent = 0.5 * (r_edges[:-1] + r_edges[1:])
        ybks = [np.interp(r_cent, rgi, ybki) for (rgi, ybki) in ybk_list]
        ybk_mean = np.nanmean(np.stack(ybks, axis=0), axis=0)
        good = np.isfinite(ybk_mean) & (ybk_mean > 0)
        if np.count_nonzero(good) < 3:
            print("[WARN] wt_nr requested, but ybk_mean is not usable. Disabling radial weighting.")
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
        args=args, pb_paths=pb_paths, grid=grid, obs_list=obs_list, rays=rays,
        y_obs=y_obs, wt_r=wt_r, density_basis=density_basis,
        obs_r_bounds=obs_r_bounds, ybk_list=ybk_list,
        ray_cache_hits=cache_hits,
        ray_cache_memory_hits=cache_memory_hits,
        ray_cache_disk_hits=cache_disk_hits,
        ray_cache_misses=cache_misses,
        ray_cache_load_failures=cache_load_failures,
        ray_cache_disabled=not use_ray_cache,
        ray_cache_keys=ray_cache_keys,
    )



def _forward_matrix_cache_key(prepared: PreparedProblem) -> str:
    """Return a key for the physical sparse ray operator A.

    The key excludes lambda, data weights, density prior, and instrument gains,
    because none of those changes the physical LOS/Thomson matrix stored in A.
    """
    h = hashlib.sha256()
    _hash_update_value(h, "forward_matrix_v1")
    _hash_update_value(h, prepared.grid.nvox)
    _hash_update_value(h, np.asarray(prepared.y_obs).size)
    for key in prepared.ray_cache_keys:
        _hash_update_value(h, key)
    return h.hexdigest()


def build_regularized_tomography(
    prepared: PreparedProblem,
    lam: float,
):
    """Construct the solver while reusing an identical in-memory CSR matrix."""
    key = _forward_matrix_cache_key(prepared)
    max_entries = max(0, int(getattr(prepared.args, "step1_forward_matrix_cache_max_entries", 2)))
    cached = _FORWARD_MATRIX_MEMORY_CACHE.get(key)
    if cached is not None:
        _FORWARD_MATRIX_MEMORY_CACHE.move_to_end(key)
        print(f"[MATRIX-CACHE-HIT] key={key[:12]}..., entries={len(_FORWARD_MATRIX_MEMORY_CACHE)}")
    else:
        print(f"[MATRIX-CACHE-MISS] key={key[:12]}...; building CSR matrix once.")

    tomo = base.RegularizedTomography(
        prepared.grid,
        prepared.obs_list,
        prepared.rays,
        lam=float(lam),
        wt_r=prepared.wt_r,
        density_basis=prepared.density_basis,
        forward_matrix=cached,
        use_preconditioner=bool(getattr(prepared.args, "solver_use_preconditioner", True)),
        preconditioner_floor=float(getattr(prepared.args, "solver_preconditioner_floor", 1e-12)),
    )

    if cached is None and max_entries > 0:
        _FORWARD_MATRIX_MEMORY_CACHE[key] = tomo.A_csr
        _FORWARD_MATRIX_MEMORY_CACHE.move_to_end(key)
        while len(_FORWARD_MATRIX_MEMORY_CACHE) > max_entries:
            old_key, _ = _FORWARD_MATRIX_MEMORY_CACHE.popitem(last=False)
            print(f"[MATRIX-CACHE-EVICT] key={old_key[:12]}...")
        print(
            f"[MATRIX-CACHE-WRITE] key={key[:12]}..., "
            f"entries={len(_FORWARD_MATRIX_MEMORY_CACHE)}"
        )
    elif max_entries <= 0:
        print("[MATRIX-CACHE-DISABLED] In-memory CSR reuse is disabled.")
    return tomo


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



def target_frequency_mask(grid, ne: np.ndarray, freq_mhz: float, harmonic: int) -> np.ndarray:
    """Return the 3-D boolean region whose density reaches the target plasma frequency."""
    ne3 = np.asarray(ne, dtype=np.float64).reshape((grid.nr, grid.nth, grid.nph), order="C")
    ne_target = base.ne_cm3_from_fp_mhz(float(freq_mhz), harmonic=int(harmonic))
    return np.isfinite(ne3) & (ne3 >= ne_target)


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentiles: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    use = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(use):
        return np.full(len(percentiles), np.nan, dtype=np.float64)
    values = values[use]
    weights = weights[use]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(np.asarray(percentiles, dtype=np.float64) / 100.0, cdf, values)


def _minimal_circular_longitude_extent_deg(phi_rad: np.ndarray) -> Tuple[float, float, float]:
    phi = np.mod(np.asarray(phi_rad, dtype=np.float64).ravel(), 2.0 * np.pi)
    phi = phi[np.isfinite(phi)]
    if phi.size == 0:
        return float("nan"), float("nan"), float("nan")
    if phi.size == 1:
        deg = float(np.rad2deg(phi[0]))
        return deg, deg, 0.0
    s = np.sort(phi)
    extended = np.concatenate([s, s[:1] + 2.0 * np.pi])
    gaps = np.diff(extended)
    k = int(np.argmax(gaps))
    start = extended[k + 1] % (2.0 * np.pi)
    end = extended[k] % (2.0 * np.pi)
    extent = 2.0 * np.pi - gaps[k]
    return float(np.rad2deg(start)), float(np.rad2deg(end)), float(np.rad2deg(extent))


def _target_region_surface_area_rsun2(grid, mask: np.ndarray) -> float:
    """Approximate exposed voxel-face area of a target region in spherical coordinates."""
    m = np.asarray(mask, dtype=bool)
    if m.shape != (grid.nr, grid.nth, grid.nph) or not np.any(m):
        return 0.0

    r0 = grid.r_edges[:-1]
    r1 = grid.r_edges[1:]
    th0 = grid.th_edges[:-1]
    th1 = grid.th_edges[1:]
    dphi = grid.ph_edges[1:] - grid.ph_edges[:-1]
    dr2 = 0.5 * (r1 * r1 - r0 * r0)
    dcosth = np.cos(th0) - np.cos(th1)
    dtheta = th1 - th0

    area = 0.0

    # Inner/outer radial faces.
    inner_exposed = m & np.concatenate([np.ones((1, grid.nth, grid.nph), bool), ~m[:-1]], axis=0)
    outer_exposed = m & np.concatenate([~m[1:], np.ones((1, grid.nth, grid.nph), bool)], axis=0)
    area += float(np.sum(inner_exposed * (r0[:, None, None] ** 2) * dcosth[None, :, None] * dphi[None, None, :]))
    area += float(np.sum(outer_exposed * (r1[:, None, None] ** 2) * dcosth[None, :, None] * dphi[None, None, :]))

    # Theta faces. sin(theta)=0 naturally suppresses polar face area.
    low_theta_exposed = m & np.concatenate([np.ones((grid.nr, 1, grid.nph), bool), ~m[:, :-1]], axis=1)
    high_theta_exposed = m & np.concatenate([~m[:, 1:], np.ones((grid.nr, 1, grid.nph), bool)], axis=1)
    area += float(np.sum(low_theta_exposed * dr2[:, None, None] * np.sin(th0)[None, :, None] * dphi[None, None, :]))
    area += float(np.sum(high_theta_exposed * dr2[:, None, None] * np.sin(th1)[None, :, None] * dphi[None, None, :]))

    # Periodic phi faces.
    prev_phi = np.roll(m, 1, axis=2)
    next_phi = np.roll(m, -1, axis=2)
    low_phi_exposed = m & ~prev_phi
    high_phi_exposed = m & ~next_phi
    phi_face_area = dr2[:, None, None] * dtheta[None, :, None]
    area += float(np.sum(low_phi_exposed * phi_face_area))
    area += float(np.sum(high_phi_exposed * phi_face_area))
    return area


def solver_objective_diagnostics(tomo, prepared: PreparedProblem, solution_raw: np.ndarray) -> Dict[str, float]:
    """Evaluate objective terms and the relative residual of the normal equation."""
    x = np.asarray(solution_raw, dtype=np.float64).ravel()
    residual = tomo.A_times(x) - prepared.y_obs
    wres = tomo.W * residual
    ltlx = base.apply_LTL(x, prepared.grid, wt_r=prepared.wt_r)
    reg_quad = float(np.dot(x, ltlx))
    if not np.isfinite(reg_quad) or reg_quad < 0:
        reg_quad = float("nan")
    data_obj = 0.5 * float(np.dot(wres, wres))
    reg_obj = 0.5 * float(tomo.lam) * reg_quad if np.isfinite(reg_quad) else float("nan")
    total_obj = data_obj + reg_obj if np.isfinite(reg_obj) else float("nan")

    b = tomo.AT_times((tomo.W * tomo.W) * prepared.y_obs)
    hx = tomo.AT_times((tomo.W * tomo.W) * tomo.A_times(x)) + float(tomo.lam) * ltlx
    den = float(np.linalg.norm(b))
    normal_rel = float(np.linalg.norm(hx - b) / max(den, 1e-300))
    return {
        "normal_equation_relative_residual": normal_rel,
        "data_objective": data_obj,
        "regularization_objective": reg_obj,
        "total_objective": total_obj,
    }


def radial_residual_rows(
    prepared: PreparedProblem,
    tomo,
    y_pred: np.ndarray,
    radial_bins: Sequence[float],
) -> List[Dict[str, object]]:
    """Return group-resolved projection diagnostics in helioprojective-radius bins."""
    edges = np.asarray(radial_bins, dtype=np.float64)
    if edges.size < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError(f"Invalid radial residual bins: {radial_bins}")
    rows: List[Dict[str, object]] = []
    for path, obs, sl in zip(prepared.pb_paths, prepared.obs_list, tomo.slices):
        rho = np.hypot(obs.x.ravel()[obs.idx_map], obs.y.ravel()[obs.idx_map])
        yo = prepared.y_obs[sl]
        yp = np.asarray(y_pred)[sl]
        ww = tomo.W[sl]
        group = base.tomography_observation_group_key(path)
        for lo, hi in zip(edges[:-1], edges[1:]):
            use = np.isfinite(rho) & (rho >= lo) & (rho < hi)
            if not np.any(use):
                continue
            st = weighted_stats(yo[use], yp[use], ww[use])
            rows.append({
                "pb_name": path.name,
                "group": group,
                "r_bin_min_rsun": float(lo),
                "r_bin_max_rsun": float(hi),
                **st,
            })

    # Add aggregate rows by group and radius bin.
    groups = sorted({str(r["group"]) for r in rows})
    for group in groups:
        group_indices = [i for i, p in enumerate(prepared.pb_paths) if base.tomography_observation_group_key(p) == group]
        for lo, hi in zip(edges[:-1], edges[1:]):
            yo_parts, yp_parts, w_parts = [], [], []
            for i in group_indices:
                obs = prepared.obs_list[i]
                sl = tomo.slices[i]
                rho = np.hypot(obs.x.ravel()[obs.idx_map], obs.y.ravel()[obs.idx_map])
                use = np.isfinite(rho) & (rho >= lo) & (rho < hi)
                if np.any(use):
                    yo_parts.append(prepared.y_obs[sl][use])
                    yp_parts.append(np.asarray(y_pred)[sl][use])
                    w_parts.append(tomo.W[sl][use])
            if yo_parts:
                st = weighted_stats(np.concatenate(yo_parts), np.concatenate(yp_parts), np.concatenate(w_parts))
                rows.append({
                    "pb_name": "__GROUP_TOTAL__",
                    "group": group,
                    "r_bin_min_rsun": float(lo),
                    "r_bin_max_rsun": float(hi),
                    **st,
                })
    return rows


def per_image_calibration_rows(prepared: PreparedProblem, tomo, solution_raw: np.ndarray) -> List[Dict[str, object]]:
    """Save robust per-image gain estimates and their relative group normalization."""
    base_pred = tomo.A_times_unscaled(solution_raw)
    cal_r_min, cal_r_max = _resolve_cross_calibration_radial_range(prepared)
    rows: List[Dict[str, object]] = []
    group_values: Dict[str, List[float]] = {}
    for path, obs, sl in zip(prepared.pb_paths, prepared.obs_list, tomo.slices):
        rho = np.hypot(obs.x.ravel()[obs.idx_map], obs.y.ravel()[obs.idx_map])
        use = np.isfinite(rho) & (rho >= cal_r_min) & (rho <= cal_r_max)
        gain, n_used = base.robust_weighted_projection_scale(
            prepared.y_obs[sl][use],
            base_pred[sl][use],
            tomo.W[sl][use],
            min_count=int(prepared.args.cross_calibration_min_count),
            clip_sigma=float(prepared.args.cross_calibration_clip_sigma),
            max_clip_iter=2,
        )
        group = base.tomography_observation_group_key(path)
        if np.isfinite(gain) and gain > 0:
            group_values.setdefault(group, []).append(float(gain))
        rows.append({
            "pb_name": path.name,
            "pb_path": str(path),
            "group": group,
            "obs_datetime": str(base.parse_pb_filename_datetime(path)),
            "r_min_rsun": cal_r_min,
            "r_max_rsun": cal_r_max,
            "fit_gain_model_to_observed": finite_or_nan(gain),
            "used_pixels": int(n_used),
        })

    medians = {k: float(np.nanmedian(v)) for k, v in group_values.items() if v}
    ref = medians.get(str(prepared.args.calibration_reference_group), np.nan)
    for row in rows:
        gmed = medians.get(str(row["group"]), np.nan)
        row["group_fit_gain_median"] = finite_or_nan(gmed)
        row["relative_to_reference_group"] = finite_or_nan(gmed / ref) if np.isfinite(ref) and ref > 0 else np.nan
    return rows


def calibration_radial_gain_rows(
    prepared: PreparedProblem,
    tomo,
    solution_raw: np.ndarray,
    radial_bins: Sequence[float],
) -> List[Dict[str, object]]:
    """Estimate effective instrument gain as a function of projected radius."""
    edges = np.asarray(radial_bins, dtype=np.float64)
    if edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError(f"Invalid calibration radial bins: {radial_bins}")
    base_pred = tomo.A_times_unscaled(solution_raw)
    rows: List[Dict[str, object]] = []
    groups = sorted({base.tomography_observation_group_key(p) for p in prepared.pb_paths})
    for group in groups:
        indices = [i for i, p in enumerate(prepared.pb_paths) if base.tomography_observation_group_key(p) == group]
        for lo, hi in zip(edges[:-1], edges[1:]):
            yo_parts, yp_parts, w_parts = [], [], []
            for i in indices:
                obs = prepared.obs_list[i]
                sl = tomo.slices[i]
                rho = np.hypot(obs.x.ravel()[obs.idx_map], obs.y.ravel()[obs.idx_map])
                use = np.isfinite(rho) & (rho >= lo) & (rho < hi)
                if np.any(use):
                    yo_parts.append(prepared.y_obs[sl][use])
                    yp_parts.append(base_pred[sl][use])
                    w_parts.append(tomo.W[sl][use])
            if not yo_parts:
                continue
            gain, n_used = base.robust_weighted_projection_scale(
                np.concatenate(yo_parts), np.concatenate(yp_parts), np.concatenate(w_parts),
                min_count=int(prepared.args.cross_calibration_min_count),
                clip_sigma=float(prepared.args.cross_calibration_clip_sigma),
                max_clip_iter=2,
            )
            rows.append({
                "group": group,
                "r_bin_min_rsun": float(lo),
                "r_bin_max_rsun": float(hi),
                "fit_gain_model_to_observed": finite_or_nan(gain),
                "equivalent_data_correction": finite_or_nan(1.0 / gain) if np.isfinite(gain) and gain > 0 else np.nan,
                "used_pixels": int(n_used),
            })
    return rows


def compute_coverage_diagnostics(
    prepared: PreparedProblem,
    tomo,
    output_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Compute diag(A^T C^T W^2 C A) without materializing A squared.

    The calculation is chunked over CSR rows to keep peak memory bounded.  The
    saved ``coverage_density`` array refers to physical-density space; when a
    density basis is active, ``coverage_solver`` additionally includes basis^2.
    """
    coverage = np.zeros(tomo.grid.nvox, dtype=np.float64)
    row_weight = (np.asarray(tomo.W) * np.asarray(tomo.measurement_scale)) ** 2
    indptr = tomo.A_csr.indptr
    indices = tomo.A_csr.indices
    data = tomo.A_csr.data
    chunk_rows = 4096
    for r0 in range(0, tomo.n_meas, chunk_rows):
        r1 = min(tomo.n_meas, r0 + chunk_rows)
        start = int(indptr[r0])
        stop = int(indptr[r1])
        if stop <= start:
            continue
        counts = np.diff(indptr[r0:r1 + 1])
        rw = np.repeat(row_weight[r0:r1], counts)
        np.add.at(coverage, indices[start:stop], data[start:stop] * data[start:stop] * rw)

    coverage_solver = coverage.copy()
    if tomo.density_basis is not None:
        coverage_solver *= np.asarray(tomo.density_basis, dtype=np.float64) ** 2

    positive = coverage[np.isfinite(coverage) & (coverage > 0)]
    max_cov = float(np.nanmax(positive)) if positive.size else np.nan
    threshold_rel = float(getattr(prepared.args, "step1_coverage_low_relative_threshold", 1e-3))
    normalized = coverage / max_cov if np.isfinite(max_cov) and max_cov > 0 else np.zeros_like(coverage)
    stats = {
        "coverage_min_positive": float(np.nanmin(positive)) if positive.size else np.nan,
        "coverage_p16": float(np.nanpercentile(positive, 16)) if positive.size else np.nan,
        "coverage_median": float(np.nanmedian(positive)) if positive.size else np.nan,
        "coverage_p84": float(np.nanpercentile(positive, 84)) if positive.size else np.nan,
        "coverage_max": max_cov,
        "coverage_zero_fraction": float(np.mean(~np.isfinite(coverage) | (coverage <= 0))),
        "coverage_low_relative_fraction": float(np.mean(normalized < threshold_rel)),
        "coverage_low_relative_threshold": threshold_rel,
    }

    freq_list = list(prepared.args.freq_mhz_list) if prepared.args.freq_mhz_list is not None else [float(prepared.args.freq_mhz)]
    # The target-region coverage is added later in run_final_diagnostics, after ne is known.
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            coverage_density=coverage.astype(np.float32).reshape((prepared.grid.nr, prepared.grid.nth, prepared.grid.nph)),
            coverage_solver=coverage_solver.astype(np.float32).reshape((prepared.grid.nr, prepared.grid.nth, prepared.grid.nph)),
            coverage_relative=normalized.astype(np.float32).reshape((prepared.grid.nr, prepared.grid.nth, prepared.grid.nph)),
            r_edges=prepared.grid.r_edges.astype(np.float32),
            th_edges=prepared.grid.th_edges.astype(np.float32),
            ph_edges=prepared.grid.ph_edges.astype(np.float32),
            frequency_list_mhz=np.asarray(freq_list, dtype=np.float64),
        )
        print(f"[STEP1] Saved coverage map: {output_path}")
    tomo.coverage_density = coverage
    tomo.coverage_solver = coverage_solver
    tomo.coverage_relative = normalized
    return stats


def solution_positivity_diagnostics(solution_raw: np.ndarray) -> Dict[str, float]:
    x = np.asarray(solution_raw, dtype=np.float64).ravel()
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {
            "solution_negative_voxel_fraction": np.nan,
            "solution_zero_voxel_fraction": np.nan,
            "solution_minimum": np.nan,
            "solution_maximum": np.nan,
        }
    return {
        "solution_negative_voxel_fraction": float(np.mean(finite < 0)),
        "solution_zero_voxel_fraction": float(np.mean(finite == 0)),
        "solution_minimum": float(np.min(finite)),
        "solution_maximum": float(np.max(finite)),
    }


def resample_mask_to_grid(mask: np.ndarray, source_grid, target_grid) -> np.ndarray:
    """Nearest-cell resampling between regular spherical tomography grids."""
    src = np.asarray(mask, dtype=bool).reshape((source_grid.nr, source_grid.nth, source_grid.nph))
    rt = 0.5 * (target_grid.r_edges[:-1] + target_grid.r_edges[1:])
    tt = 0.5 * (target_grid.th_edges[:-1] + target_grid.th_edges[1:])
    pt = np.mod(0.5 * (target_grid.ph_edges[:-1] + target_grid.ph_edges[1:]), 2.0 * np.pi)
    ir = np.clip(np.searchsorted(source_grid.r_edges, rt, side="right") - 1, 0, source_grid.nr - 1)
    it = np.clip(np.searchsorted(source_grid.th_edges, tt, side="right") - 1, 0, source_grid.nth - 1)
    ip = np.clip(np.searchsorted(source_grid.ph_edges, pt, side="right") - 1, 0, source_grid.nph - 1)
    return src[np.ix_(ir, it, ip)]


def mask_overlap_metrics(mask_a: np.ndarray, mask_b: np.ndarray, grid) -> Dict[str, float]:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Mask shape mismatch: {a.shape} != {b.shape}")
    inter = a & b
    union = a | b
    na = int(np.count_nonzero(a))
    nb = int(np.count_nonzero(b))
    ni = int(np.count_nonzero(inter))
    nu = int(np.count_nonzero(union))
    vol = approximate_voxel_volumes_rsun3(grid)
    return {
        "jaccard": float(ni / nu) if nu else np.nan,
        "dice": float(2 * ni / (na + nb)) if (na + nb) else np.nan,
        "intersection_volume_rsun3": float(np.sum(vol[inter])),
        "union_volume_rsun3": float(np.sum(vol[union])),
        "reference_volume_rsun3": float(np.sum(vol[a])),
        "scenario_volume_rsun3": float(np.sum(vol[b])),
    }


def target_frequency_metrics(grid, ne: np.ndarray, freq_mhz: float, harmonic: int) -> Dict[str, float]:
    """Compute volumetric, positional, component, and extent diagnostics above a target density."""
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
        "surface_area_rsun2": 0.0,
        "centroid_x_rsun": float("nan"),
        "centroid_y_rsun": float("nan"),
        "centroid_z_rsun": float("nan"),
        "centroid_r_rsun": float("nan"),
        "r_min_rsun": float("nan"),
        "r_p16_rsun": float("nan"),
        "r_median_rsun": float("nan"),
        "r_p84_rsun": float("nan"),
        "r_max_rsun": float("nan"),
        "latitude_min_deg": float("nan"),
        "latitude_max_deg": float("nan"),
        "latitude_extent_deg": float("nan"),
        "longitude_arc_start_deg": float("nan"),
        "longitude_arc_end_deg": float("nan"),
        "longitude_extent_deg": float("nan"),
        "n_components": 0,
        "largest_component_fraction": float("nan"),
        "second_largest_component_fraction": float("nan"),
    }
    if n_vox == 0:
        return out

    vol = approximate_voxel_volumes_rsun3(grid)
    vsel = vol[good]
    vtot = float(np.sum(vsel))
    out["volume_ge_target_rsun3"] = vtot
    out["surface_area_rsun2"] = _target_region_surface_area_rsun2(grid, good)

    rr, tt, pp = grid.voxel_centers_sph()
    xx, yy, zz = grid.voxel_centers_xyz()
    cx = float(np.sum(xx[good] * vsel) / max(vtot, 1e-300))
    cy = float(np.sum(yy[good] * vsel) / max(vtot, 1e-300))
    cz = float(np.sum(zz[good] * vsel) / max(vtot, 1e-300))
    out["centroid_x_rsun"] = cx
    out["centroid_y_rsun"] = cy
    out["centroid_z_rsun"] = cz
    out["centroid_r_rsun"] = float(np.sqrt(cx * cx + cy * cy + cz * cz))

    rsel = rr[good]
    out["r_min_rsun"] = float(np.min(rsel))
    out["r_max_rsun"] = float(np.max(rsel))
    rp = _weighted_percentile(rsel, vsel, [16.0, 50.0, 84.0])
    out["r_p16_rsun"], out["r_median_rsun"], out["r_p84_rsun"] = [float(v) for v in rp]

    lat = 90.0 - np.rad2deg(tt[good])
    out["latitude_min_deg"] = float(np.min(lat))
    out["latitude_max_deg"] = float(np.max(lat))
    out["latitude_extent_deg"] = float(np.max(lat) - np.min(lat))
    lon_start, lon_end, lon_extent = _minimal_circular_longitude_extent_deg(pp[good])
    out["longitude_arc_start_deg"] = lon_start
    out["longitude_arc_end_deg"] = lon_end
    out["longitude_extent_deg"] = lon_extent

    if ndimage is not None:
        structure = np.zeros((3, 3, 3), dtype=np.int8)
        structure[1, 1, :] = 1
        structure[1, :, 1] = 1
        structure[:, 1, 1] = 1
        labeled, ncomp = ndimage.label(good, structure=structure)
        # Merge labels touching the periodic phi boundary before measuring volumes.
        if ncomp > 0:
            parent = np.arange(ncomp + 1, dtype=np.int64)
            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            def union(a, b):
                if a and b:
                    ra, rb = find(int(a)), find(int(b))
                    if ra != rb:
                        parent[rb] = ra
            left = labeled[:, :, 0]
            right = labeled[:, :, -1]
            for a, b in zip(left.ravel(), right.ravel()):
                union(a, b)
            if np.any(parent[1:] != np.arange(1, ncomp + 1)):
                remap = np.arange(ncomp + 1)
                for k in range(1, ncomp + 1):
                    remap[k] = find(k)
                labeled = remap[labeled]
                unique = np.unique(labeled[labeled > 0])
                compact = {old: new for new, old in enumerate(unique, start=1)}
                for old, new in compact.items():
                    labeled[labeled == old] = new
                ncomp = len(unique)
        out["n_components"] = int(ncomp)
        if ncomp > 0:
            comp_vol = np.bincount(labeled.ravel(), weights=vol.ravel())
            ranked = np.sort(comp_vol[1:])[::-1] if comp_vol.size > 1 else np.array([], dtype=float)
            if ranked.size:
                out["largest_component_fraction"] = float(ranked[0] / max(vtot, 1e-300))
            if ranked.size > 1:
                out["second_largest_component_fraction"] = float(ranked[1] / max(vtot, 1e-300))
            else:
                out["second_largest_component_fraction"] = 0.0

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
    x0: Optional[np.ndarray] = None,
) -> LambdaResult:
    """Solve one lambda using fixed instrument gains and an optional warm start."""
    tomo.lam = float(lam)
    if x0 is None:
        x0 = getattr(tomo, "cross_calibration_seed", None)
    t0 = time.time()
    solution_raw, info = tomo.solve(
        prepared.y_obs,
        maxiter=int(prepared.args.maxiter),
        tol=float(prepared.args.tol),
        positivity=True,
        positivity_method=str(prepared.args.positivity_method),
        x0=x0,
        use_preconditioner=bool(getattr(prepared.args, "solver_use_preconditioner", True)),
        preconditioner_floor=float(getattr(prepared.args, "solver_preconditioner_floor", 1e-12)),
    )
    elapsed = time.time() - t0
    tomo._last_lambda_solution = np.asarray(solution_raw, dtype=np.float64)

    ne_raw = tomo.solution_to_density(solution_raw)
    y_pred = tomo.A_times(solution_raw)
    stats = weighted_stats(prepared.y_obs, y_pred, tomo.W)
    suggested_scale = base.weighted_projection_scale(prepared.y_obs, y_pred, tomo.W, min_count=100)
    reg_norm = regularization_norm(solution_raw, prepared.grid, prepared.wt_r)
    solver_diag = solver_objective_diagnostics(tomo, prepared, solution_raw)

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

    gains = dict(getattr(tomo, "cross_calibration_forward_gains", {}))
    corrections = dict(getattr(tomo, "cross_calibration_data_corrections", {}))
    history = list(getattr(tomo, "cross_calibration_history", []))

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
        normal_equation_relative_residual=finite_or_nan(solver_diag["normal_equation_relative_residual"]),
        data_objective=finite_or_nan(solver_diag["data_objective"]),
        regularization_objective=finite_or_nan(solver_diag["regularization_objective"]),
        total_objective=finite_or_nan(solver_diag["total_objective"]),
        solver_iterations=int(getattr(tomo, "last_solver_iterations", 0)),
        solver_used_preconditioner=bool(getattr(tomo, "last_solver_used_preconditioner", False)),
        group_forward_gains=gains,
        group_data_corrections=corrections,
        cross_calibration_iterations=len(history),
        cross_calibration_converged=bool(getattr(tomo, "cross_calibration_converged", False)),
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
        "normal_equation_relative_residual": result.normal_equation_relative_residual,
        "data_objective": result.data_objective,
        "regularization_objective": result.regularization_objective,
        "total_objective": result.total_objective,
        "solver_iterations": int(result.solver_iterations),
        "solver_used_preconditioner": bool(result.solver_used_preconditioner),
        "suggested_brightness_scale": result.suggested_brightness_scale,
        "ne_min_cm3": result.ne_min,
        "ne_max_cm3": result.ne_max,
        "f_min_mhz": result.f_min_mhz,
        "f_max_mhz": result.f_max_mhz,
        "cross_calibration_forward_gains": json.dumps(result.group_forward_gains, ensure_ascii=False),
        "cross_calibration_data_corrections": json.dumps(result.group_data_corrections, ensure_ascii=False),
        "cross_calibration_iterations": int(result.cross_calibration_iterations),
        "cross_calibration_converged": bool(result.cross_calibration_converged),
        "cor1a_forward_gain": finite_or_nan(result.group_forward_gains.get("cor1a", np.nan)),
        "cor1a_data_correction": finite_or_nan(result.group_data_corrections.get("cor1a", np.nan)),
    }
    row.update(result.target_metrics)
    return row





def run_lambda_scan(
    prepared: PreparedProblem,
    lambdas: Sequence[float],
    output_dir: Path,
    keep_solutions: bool = False,
) -> Tuple[List[LambdaResult], Optional[float], object]:
    """Run a continuation lambda scan with one CSR matrix and one calibration.

    The instrument gains are calibrated once at ``prepared.args.lam`` and then
    held fixed while only lambda changes.  By default the solve order is from
    large to small lambda and each converged solution warm-starts the next one.
    This preserves the inverse problem for every lambda while avoiding repeated
    preprocessing, CSR construction, and gain calibration.
    """
    output_dir = ensure_dir(output_dir)

    values = sorted({float(v) for v in lambdas})
    if not values:
        raise ValueError("Lambda scan received no values.")
    descending = bool(getattr(prepared.args, "step1_lambda_scan_descending", True))
    solve_order = sorted(values, reverse=descending)
    warm_start = bool(getattr(prepared.args, "step1_lambda_scan_warm_start", True))

    print("[STEP1] Building/reusing one RegularizedTomography object for lambda scan...")
    tomo = build_regularized_tomography(prepared, float(prepared.args.lam))
    configure_group_cross_calibration(prepared, tomo)
    print(
        f"[LAMBDA-CONTINUATION] solve_order={solve_order}, "
        f"warm_start={warm_start}, gains_fixed_after_calibration=True"
    )

    results_by_lambda: Dict[float, LambdaResult] = {}
    x_seed = getattr(tomo, "cross_calibration_seed", None)
    for lam in solve_order:
        print(f"[STEP1] Solving lambda={float(lam):.6g} ...")
        res = solve_one_lambda(
            tomo,
            prepared,
            float(lam),
            keep_solution=bool(keep_solutions),
            x0=x_seed,
        )
        results_by_lambda[float(lam)] = res
        if warm_start:
            x_seed = getattr(tomo, "_last_lambda_solution", None)
        print(
            f"[STEP1] lambda={res.lam:.6g}: "
            f"misfit_rms={res.data_misfit_rms:.4e}, "
            f"reg_norm={res.regularization_norm:.4e}, "
            f"f_range={res.f_min_mhz:.3f}..{res.f_max_mhz:.3f} MHz, "
            f"cor1a_correction={res.group_data_corrections.get('cor1a', np.nan):.6g}, "
            f"iterations={res.solver_iterations}, cg_info={res.info}, "
            f"time={res.solve_seconds:.1f}s"
        )

    # Preserve ascending lambda in CSV and later plotting tables.
    results = [results_by_lambda[v] for v in sorted(results_by_lambda)]
    rows = [lambda_result_to_row(r) for r in results]
    write_rows_csv(output_dir / "step1_lambda_scan.csv", rows)

    lam_corner = normalized_lcurve_corner(
        [r.lam for r in results],
        [r.data_misfit_norm for r in results],
        [r.regularization_norm for r in results],
    )

    summary = {
        "lambda_values": [float(r.lam) for r in results],
        "lambda_solve_order": solve_order,
        "lambda_warm_start": warm_start,
        "lcurve_lambda_candidate": lam_corner,
        "calibration_lambda": float(prepared.args.lam),
        "cross_calibration_forward_gains": getattr(tomo, "cross_calibration_forward_gains", {}),
        "cross_calibration_data_corrections": getattr(tomo, "cross_calibration_data_corrections", {}),
        "note": (
            "One physical CSR matrix is shared. Instrument gains are calibrated "
            "once at the baseline lambda and held fixed across the lambda-only scan."
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

    return results, lam_corner, tomo



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
    initial_group_forward_gains: Optional[Dict[str, float]] = None,
    recalibrate_cross_calibration: bool = True,
    existing_tomo=None,
    precomputed_solution_raw: Optional[np.ndarray] = None,
    precomputed_info: Optional[int] = None,
    precomputed_iterations: Optional[int] = None,
) -> Tuple[object, np.ndarray, np.ndarray, np.ndarray, float]:
    """Solve once at the final lambda and save extended validation diagnostics."""
    output_dir = ensure_dir(output_dir)
    print(f"[STEP1] Final diagnostic evaluation with lambda={float(lam):.6g}")
    tomo = existing_tomo if existing_tomo is not None else build_regularized_tomography(prepared, float(lam))
    tomo.lam = float(lam)

    if precomputed_solution_raw is None:
        calibration_seed = None
        reuse_gain = (
            bool(getattr(prepared.args, "auto_cross_calibrate_groups", False))
            and not bool(recalibrate_cross_calibration)
            and bool(initial_group_forward_gains)
        )
        if reuse_gain:
            gains = _default_group_forward_gains(prepared)
            gains.update(base.normalize_group_float_map(initial_group_forward_gains, "initial_group_forward_gains"))
            gains[str(prepared.args.calibration_reference_group)] = 1.0
            tomo.set_measurement_scale(base.build_group_measurement_scale_vector(prepared.pb_paths, tomo, gains))
            corrections = {key: 1.0 / value for key, value in gains.items()}
            tomo.cross_calibration_mode = "reused"
            tomo.cross_calibration_forward_gains = dict(gains)
            tomo.cross_calibration_data_corrections = dict(corrections)
            tomo.cross_calibration_history = []
            tomo.cross_calibration_converged = True
            tomo.cross_calibration_seed = None
            print(f"[CAL] Reusing lambda-scan forward gains: {gains}")
        else:
            _, _, calibration_seed = configure_group_cross_calibration(
                prepared, tomo, initial_group_forward_gains=initial_group_forward_gains
            )

        solution_raw, info = tomo.solve(
            prepared.y_obs,
            maxiter=int(prepared.args.maxiter),
            tol=float(prepared.args.tol),
            positivity=True,
            positivity_method=str(prepared.args.positivity_method),
            x0=calibration_seed,
            use_preconditioner=bool(getattr(prepared.args, "solver_use_preconditioner", True)),
            preconditioner_floor=float(getattr(prepared.args, "solver_preconditioner_floor", 1e-12)),
        )
    else:
        solution_raw = np.asarray(precomputed_solution_raw, dtype=np.float64).ravel()
        if solution_raw.size != prepared.grid.nvox:
            raise ValueError(
                f"precomputed_solution_raw has size {solution_raw.size}, expected {prepared.grid.nvox}"
            )
        info = int(precomputed_info if precomputed_info is not None else 0)
        if precomputed_iterations is not None:
            tomo.last_solver_iterations = int(precomputed_iterations)
        print(
            "[FAST-REUSE] Reusing the already solved lambda-continuation solution; "
            "no additional calibration or CG solve is performed."
        )

    tomo.final_solver_info = int(info)
    if info != 0:
        print(f"[WARN] Final solver did not fully converge (info={info}).")

    ne_raw = tomo.solution_to_density(solution_raw)
    y_pred = tomo.A_times(solution_raw)
    W = tomo.W
    suggested_scale = base.weighted_projection_scale(prepared.y_obs, y_pred, W, min_count=100)

    solver_diag = solver_objective_diagnostics(tomo, prepared, solution_raw)
    positivity_diag = solution_positivity_diagnostics(solution_raw)
    tomo.final_solver_diagnostics = dict(solver_diag)
    tomo.final_positivity_diagnostics = dict(positivity_diag)
    print(
        "[SOLVER] "
        f"normal_rel_residual={solver_diag['normal_equation_relative_residual']:.4e}, "
        f"data_objective={solver_diag['data_objective']:.4e}, "
        f"regularization_objective={solver_diag['regularization_objective']:.4e}, "
        f"total_objective={solver_diag['total_objective']:.4e}"
    )
    print(
        "[POSITIVITY] "
        f"method={prepared.args.positivity_method}, negative_fraction={positivity_diag['solution_negative_voxel_fraction']:.4e}, "
        f"zero_fraction={positivity_diag['solution_zero_voxel_fraction']:.4e}, "
        f"min={positivity_diag['solution_minimum']:.4e}, max={positivity_diag['solution_maximum']:.4e}"
    )

    print("[STEP1] Final global projection diagnostic:")
    base.print_projection_fit_diagnostic("step1_final/global", prepared.y_obs, y_pred, W)
    base.print_projection_fit_diagnostics_by_group("step1_final", prepared.pb_paths, tomo, prepared.y_obs, y_pred, W)
    base.print_group_calibration_hints(
        "step1_final", prepared.pb_paths, tomo, prepared.y_obs, y_pred, W,
        reference_group=str(prepared.args.calibration_reference_group),
    )

    image_rows = per_image_residual_rows(prepared, tomo, solution_raw, y_pred)
    write_rows_csv(output_dir / "step1_per_image_residuals.csv", image_rows)
    write_rows_csv(output_dir / "step1_group_residuals.csv", group_residual_rows(image_rows))
    radial_rows = radial_residual_rows(
        prepared, tomo, y_pred,
        radial_bins=getattr(prepared.args, "step1_radial_residual_bins", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
    )
    write_rows_csv(output_dir / "step1_radial_residuals.csv", radial_rows)
    print(f"[STEP1] Saved per-image residual CSV: {output_dir / 'step1_per_image_residuals.csv'}")
    print(f"[STEP1] Saved group residual CSV: {output_dir / 'step1_group_residuals.csv'}")
    print(f"[STEP1] Saved radial residual CSV: {output_dir / 'step1_radial_residuals.csv'}")

    if bool(getattr(prepared.args, "step1_save_calibration_diagnostics", True)):
        calibration_rows = per_image_calibration_rows(prepared, tomo, solution_raw)
        calibration_radial_rows = calibration_radial_gain_rows(
            prepared, tomo, solution_raw,
            radial_bins=getattr(prepared.args, "step1_calibration_radial_bins", [1.7, 2.0, 2.5, 3.0, 3.5]),
        )
        write_rows_csv(output_dir / "step1_per_image_calibration.csv", calibration_rows)
        write_rows_csv(output_dir / "step1_calibration_by_radius.csv", calibration_radial_rows)
        tomo.per_image_calibration_rows = calibration_rows
        tomo.calibration_radial_rows = calibration_radial_rows
        print(f"[STEP1] Saved calibration diagnostics: {output_dir / 'step1_per_image_calibration.csv'}")
        print(f"[STEP1] Saved radial calibration diagnostics: {output_dir / 'step1_calibration_by_radius.csv'}")
    else:
        tomo.per_image_calibration_rows = []
        tomo.calibration_radial_rows = []

    coverage_stats: Dict[str, float] = {}
    if bool(getattr(prepared.args, "step1_compute_coverage", False)):
        coverage_path = (
            output_dir / "step1_coverage_map.npz"
            if bool(getattr(prepared.args, "step1_save_coverage_npz", False))
            else None
        )
        coverage_stats = compute_coverage_diagnostics(prepared, tomo, coverage_path)
        for freq in (list(prepared.args.freq_mhz_list) if prepared.args.freq_mhz_list is not None else [float(prepared.args.freq_mhz)]):
            mask = target_frequency_mask(prepared.grid, ne_raw, float(freq), int(prepared.args.harmonic)).ravel()
            cov = np.asarray(tomo.coverage_relative).ravel()
            prefix = f"coverage_target_{float(freq):g}MHz_"
            coverage_stats[prefix + "median_relative"] = float(np.nanmedian(cov[mask])) if np.any(mask) else np.nan
            coverage_stats[prefix + "p16_relative"] = float(np.nanpercentile(cov[mask], 16)) if np.any(mask) else np.nan
            coverage_stats[prefix + "low_fraction"] = float(np.mean(cov[mask] < float(prepared.args.step1_coverage_low_relative_threshold))) if np.any(mask) else np.nan
        print(f"[COVERAGE] {coverage_stats}")
    tomo.coverage_diagnostics = coverage_stats

    scale = suggested_scale if bool(prepared.args.apply_brightness_scale) else 1.0
    ne = ne_raw * scale

    if bool(getattr(prepared.args, "step1_save_target_mask_npz", False)):
        masks = {}
        freq_list = list(prepared.args.freq_mhz_list) if prepared.args.freq_mhz_list is not None else [float(prepared.args.freq_mhz)]
        for freq in freq_list:
            masks[f"mask_{float(freq):g}MHz"] = target_frequency_mask(prepared.grid, ne, float(freq), int(prepared.args.harmonic)).astype(np.uint8)
        np.savez_compressed(
            output_dir / "step1_target_frequency_masks.npz",
            **masks,
            r_edges=prepared.grid.r_edges.astype(np.float32),
            th_edges=prepared.grid.th_edges.astype(np.float32),
            ph_edges=prepared.grid.ph_edges.astype(np.float32),
        )
        print(f"[STEP1] Saved target-frequency masks: {output_dir / 'step1_target_frequency_masks.npz'}")

    save_residual_maps(
        prepared, tomo, y_pred, output_dir=output_dir / "residual_maps",
        save_npy=bool(save_residual_npy), save_png=bool(save_residual_png),
    )
    return tomo, solution_raw, ne, y_pred, suggested_scale






def run_leave_one_image_out(
    prepared: PreparedProblem,
    lambdas: Sequence[float],
    output_dir: Path,
    max_holdouts: Optional[int] = None,
) -> None:
    """
    Expensive true leave-one-image-out validation using the current solver,
    positivity setting, and relative instrument-gain model.
    """
    output_dir = ensure_dir(output_dir)

    n = len(prepared.obs_list)
    holdout_indices = list(range(n))
    if max_holdouts is not None and max_holdouts > 0:
        holdout_indices = holdout_indices[: int(max_holdouts)]

    rows: List[Dict[str, object]] = []
    for ihold in holdout_indices:
        print(f"[STEP1-LOO] Holdout {ihold + 1}/{n}: {prepared.pb_paths[ihold].name}")
        train_indices = [i for i in range(n) if i != ihold]
        train_obs = [prepared.obs_list[i] for i in train_indices]
        train_rays = [prepared.rays[i] for i in train_indices]
        train_paths = [prepared.pb_paths[i] for i in train_indices]
        train_y_parts = [prepared.obs_list[i].pb.ravel()[prepared.obs_list[i].idx_map] for i in train_indices]
        y_train = np.concatenate(train_y_parts)

        held_obs = prepared.obs_list[ihold]
        held_ray = prepared.rays[ihold]
        held_path = prepared.pb_paths[ihold]
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

        train_prepared = PreparedProblem(
            args=prepared.args,
            pb_paths=train_paths,
            grid=prepared.grid,
            obs_list=train_obs,
            rays=train_rays,
            y_obs=y_train,
            wt_r=prepared.wt_r,
            density_basis=prepared.density_basis,
            obs_r_bounds=[prepared.obs_r_bounds[i] for i in train_indices],
            ybk_list=[prepared.ybk_list[i] for i in train_indices],
        )
        gains, history, calibration_seed = configure_group_cross_calibration(
            train_prepared, train_tomo
        )
        held_tomo.set_measurement_scale(
            base.build_group_measurement_scale_vector([held_path], held_tomo, gains)
        )

        for lam in lambdas:
            train_tomo.lam = float(lam)
            held_tomo.lam = float(lam)
            t0 = time.time()
            sol, info = train_tomo.solve(
                y_train,
                maxiter=int(prepared.args.maxiter),
                tol=float(prepared.args.tol),
                positivity=True,
                positivity_method=str(prepared.args.positivity_method),
                x0=calibration_seed,
            )
            elapsed = time.time() - t0
            y_pred_held = held_tomo.A_times(sol)
            st = weighted_stats(y_held, y_pred_held, held_tomo.W)
            rows.append({
                "holdout_index": ihold,
                "holdout_name": held_path.name,
                "holdout_group": base.tomography_observation_group_key(held_path),
                "lambda": float(lam),
                "cg_info": int(info),
                "solve_seconds": float(elapsed),
                "cross_calibration_forward_gains": json.dumps(gains, ensure_ascii=False),
                "cross_calibration_iterations": int(len(history)),
                **st,
            })
            print(
                f"[STEP1-LOO] holdout={ihold}, lambda={float(lam):.6g}, "
                f"heldout_misfit_rms={st['misfit_rms']:.4e}, "
                f"heldout_weighted_rms_rel={st['weighted_rms_rel']:.4e}, info={info}"
            )

    write_rows_csv(output_dir / "step1_leave_one_image_out.csv", rows)
    print(f"[STEP1-LOO] Saved: {output_dir / 'step1_leave_one_image_out.csv'}")




def run_time_block_holdout(
    prepared: PreparedProblem,
    lam: float,
    output_dir: Path,
    block_days: float = 1.0,
    max_blocks: Optional[int] = 6,
) -> None:
    """Hold out contiguous time blocks while reusing already prepared ray bundles."""
    output_dir = ensure_dir(output_dir)
    dated = [(i, base.parse_pb_filename_datetime(p)) for i, p in enumerate(prepared.pb_paths)]
    dated = [(i, dt) for i, dt in dated if dt is not None]
    if not dated:
        print("[STEP1-BLOCK] No parseable observation times; block holdout skipped.")
        return
    block_seconds = float(block_days) * 86400.0
    tmin = min(dt for _, dt in dated)
    tmax = max(dt for _, dt in dated)
    starts = []
    t = tmin
    while t <= tmax:
        starts.append(t)
        t = t + base.timedelta(seconds=block_seconds) if hasattr(base, 'timedelta') else t + __import__('datetime').timedelta(seconds=block_seconds)
    if max_blocks is not None and int(max_blocks) > 0 and len(starts) > int(max_blocks):
        picks = np.linspace(0, len(starts) - 1, int(max_blocks)).round().astype(int)
        starts = [starts[i] for i in sorted(set(picks.tolist()))]

    rows: List[Dict[str, object]] = []
    from datetime import timedelta as _timedelta
    for iblock, start in enumerate(starts):
        end = start + _timedelta(days=float(block_days))
        held = [i for i, dt in dated if start <= dt < end]
        if not held:
            continue
        train = [i for i in range(len(prepared.obs_list)) if i not in held]
        train_groups = {base.tomography_observation_group_key(prepared.pb_paths[i]) for i in train}
        if str(prepared.args.calibration_reference_group) not in train_groups or len(train_groups) < 2:
            print(f"[STEP1-BLOCK] Skip block {start}..{end}: training set lacks both instrument groups.")
            continue

        train_obs = [prepared.obs_list[i] for i in train]
        train_rays = [prepared.rays[i] for i in train]
        train_paths = [prepared.pb_paths[i] for i in train]
        y_train = np.concatenate([prepared.obs_list[i].pb.ravel()[prepared.obs_list[i].idx_map] for i in train])
        train_prepared = PreparedProblem(
            args=prepared.args, pb_paths=train_paths, grid=prepared.grid,
            obs_list=train_obs, rays=train_rays, y_obs=y_train,
            wt_r=prepared.wt_r, density_basis=prepared.density_basis,
            obs_r_bounds=[prepared.obs_r_bounds[i] for i in train],
            ybk_list=[prepared.ybk_list[i] for i in train],
        )
        train_tomo = base.RegularizedTomography(
            prepared.grid, train_obs, train_rays, lam=float(lam),
            wt_r=prepared.wt_r, density_basis=prepared.density_basis,
        )
        gains, history, seed = configure_group_cross_calibration(train_prepared, train_tomo)
        sol, info = train_tomo.solve(
            y_train, maxiter=int(prepared.args.maxiter), tol=float(prepared.args.tol),
            positivity=True, positivity_method=str(prepared.args.positivity_method), x0=seed,
        )

        for i in held:
            held_tomo = base.RegularizedTomography(
                prepared.grid, [prepared.obs_list[i]], [prepared.rays[i]], lam=float(lam),
                wt_r=prepared.wt_r, density_basis=prepared.density_basis,
            )
            held_tomo.set_measurement_scale(
                base.build_group_measurement_scale_vector([prepared.pb_paths[i]], held_tomo, gains)
            )
            y_held = prepared.obs_list[i].pb.ravel()[prepared.obs_list[i].idx_map]
            pred = held_tomo.A_times(sol)
            st = weighted_stats(y_held, pred, held_tomo.W)
            rows.append({
                "block_index": iblock,
                "block_start": start.isoformat(),
                "block_end": end.isoformat(),
                "holdout_index": i,
                "holdout_name": prepared.pb_paths[i].name,
                "holdout_group": base.tomography_observation_group_key(prepared.pb_paths[i]),
                "lambda": float(lam),
                "solver_info": int(info),
                "cross_calibration_iterations": len(history),
                "cross_calibration_forward_gains": json.dumps(gains, ensure_ascii=False),
                **st,
            })
        print(f"[STEP1-BLOCK] Completed block {iblock + 1}/{len(starts)}: {start}..{end}, held={len(held)}")

    write_rows_csv(output_dir / "step1_time_block_holdout.csv", rows)
    print(f"[STEP1-BLOCK] Saved: {output_dir / 'step1_time_block_holdout.csv'}")


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
    """Save a solution NPZ with extended diagnostic metadata."""
    args = prepared.args
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ne_raw = tomo.solution_to_density(solution_raw)
    scale = suggested_scale if bool(args.apply_brightness_scale) else 1.0
    pb_scale_by_group = base.normalize_group_float_map(args.pb_scale_by_group, "pb_scale_by_group")
    r_use_min_by_group = base.normalize_group_float_map(args.r_use_min_by_group, "r_use_min_by_group")
    r_use_max_by_group = base.normalize_group_float_map(args.r_use_max_by_group, "r_use_max_by_group")
    freq_list = list(args.freq_mhz_list) if args.freq_mhz_list is not None else [float(args.freq_mhz)]
    obs_lonlat_deg = np.array([(obs.lonlat_deg if obs.lonlat_deg is not None else (np.nan, np.nan)) for obs in prepared.obs_list], dtype=np.float64)
    obs_group_keys = np.array([base.tomography_observation_group_key(p) for p in prepared.pb_paths], dtype="U64")
    obs_r_use_min = np.array([b[0] for b in prepared.obs_r_bounds], dtype=np.float64)
    obs_r_use_max = np.array([b[1] for b in prepared.obs_r_bounds], dtype=np.float64)
    obs_used_pixels = np.array([obs.idx_map.size for obs in prepared.obs_list], dtype=np.int64)
    obs_limb_models = np.array([obs.limb_u_model for obs in prepared.obs_list], dtype="U128")
    obs_limb_component_names_json = np.array(
        [json.dumps(list(obs.limb_component_names)) for obs in prepared.obs_list], dtype="U512"
    )
    obs_limb_u_values_json = np.array(
        [json.dumps([float(v) for v in obs.limb_u_values]) for obs in prepared.obs_list], dtype="U512"
    )
    obs_limb_coeff_a_json = np.array(
        [json.dumps([float(v) for v in obs.limb_coeff_a]) for obs in prepared.obs_list], dtype="U512"
    )
    obs_limb_coeff_b_json = np.array(
        [json.dumps([float(v) for v in obs.limb_coeff_b]) for obs in prepared.obs_list], dtype="U512"
    )
    render_camera_time = base.parse_target_datetime(args.target_time)
    render_camera_lonlat = earth_view_camera_lonlat_from_target_time(args.target_time)
    solver_diag = getattr(tomo, "final_solver_diagnostics", {})
    positivity_diag = getattr(tomo, "final_positivity_diagnostics", {})
    coverage_diag = getattr(tomo, "coverage_diagnostics", {})

    np.savez_compressed(
        output_path,
        ne=ne.astype(np.float32), ne_raw=ne_raw.astype(np.float32), solution_raw=solution_raw.astype(np.float32),
        density_basis=(prepared.density_basis.astype(np.float32) if prepared.density_basis is not None else np.ones_like(ne_raw, dtype=np.float32)),
        scale_brightness=float(scale), suggested_scale_brightness=float(suggested_scale),
        apply_brightness_scale=bool(args.apply_brightness_scale), final_lambda=float(final_lambda),
        limb_u=float(args.limb_u),
        limb_u_mode=str(getattr(args, "limb_u_mode", "fixed")),
        limb_u_use_allen=bool(getattr(args, "limb_u_use_allen", False)),
        limb_u_bandpass_nm_by_instrument_json=json.dumps(
            getattr(args, "limb_u_bandpass_nm_by_instrument", {}), ensure_ascii=False
        ),
        limb_u_override_by_instrument_json=json.dumps(
            getattr(args, "limb_u_override_by_instrument", {}), ensure_ascii=False
        ),
        obs_limb_models=obs_limb_models,
        obs_limb_component_names_json=obs_limb_component_names_json,
        obs_limb_u_values_json=obs_limb_u_values_json,
        obs_limb_coeff_a_json=obs_limb_coeff_a_json,
        obs_limb_coeff_b_json=obs_limb_coeff_b_json,
        step1_generated_by="main_extended_multi_tomo.py",
        density_prior_model=str(args.density_prior_model), density_prior_scale=float(args.density_prior_scale),
        density_prior_file=str(getattr(args, "density_prior_file", "")),
        pb_scale_group_keys=np.array(list(pb_scale_by_group.keys()), dtype="U64"),
        pb_scale_group_values=np.array(list(pb_scale_by_group.values()), dtype=np.float64),
        r_use_min_group_keys=np.array(list(r_use_min_by_group.keys()), dtype="U64"),
        r_use_min_group_values=np.array(list(r_use_min_by_group.values()), dtype=np.float64),
        r_use_max_group_keys=np.array(list(r_use_max_by_group.keys()), dtype="U64"),
        r_use_max_group_values=np.array(list(r_use_max_by_group.values()), dtype=np.float64),
        pb_paths=np.array([str(p) for p in prepared.pb_paths], dtype="U2048"),
        pb_names=np.array([p.name for p in prepared.pb_paths], dtype="U256"),
        obs_group_keys=obs_group_keys, obs_lonlat_deg=obs_lonlat_deg,
        obs_r_use_min=obs_r_use_min.astype(np.float32), obs_r_use_max=obs_r_use_max.astype(np.float32),
        obs_used_pixels=obs_used_pixels, data_dir=str(args.data_dir), cor1a_data_dir=str(args.cor1a_data_dir),
        target_time=str(args.target_time), render_camera_is_earth_view=True,
        render_camera_mode="forced_target_time_sub_earth", render_camera_source="sunpy.coordinates.sun.L0_B0",
        render_camera_time_utc=render_camera_time.strftime("%Y-%m-%dT%H:%M:%S"),
        render_camera_lon_deg=float(render_camera_lonlat[0]), render_camera_lat_deg=float(render_camera_lonlat[1]),
        render_camera_lonlat_frame="Carrington sub-Earth L0 / B0",
        search_window_days=float(args.search_window_days), include_kcor_lasco=bool(args.include_kcor_lasco),
        include_cor1a=bool(args.include_cor1a), include_lasco_only=bool(args.include_lasco_only),
        deduplicate_pb_fits=bool(args.deduplicate_pb_fits), out_n=int(args.out_n),
        r_min=float(args.r_min), r_max=float(args.r_max), nr=int(args.nr), nth=int(args.nth), nph=int(args.nph),
        ds=float(args.ds), filt=bool(args.filt),
        despike_nsig=float(args.despike_nsig), despike_med=int(args.despike_med), pb_floor=str(args.pb_floor),
        dpa_deg=float(args.dpa_deg), r_use_min=float(args.r_use_min), r_use_max=float(args.r_use_max),
        hm=int(args.hm), wt_nr=bool(args.wt_nr), lam=float(final_lambda), q_low=float(args.q_low),
        width_pix=float(args.width_pix), maxiter=int(args.maxiter), tol=float(args.tol),
        positivity_method=str(args.positivity_method),
        use_density_prior=(bool(args.use_density_prior) and str(args.density_prior_model).strip().lower() not in ("", "none", "off", "false", "0")),
        thomson_normalize_msb=bool(args.thomson_normalize_msb), thomson_kernel_scale=float(args.thomson_kernel_scale),
        use_temporal_despike=bool(args.use_temporal_despike), ne3dtomo_global_ybk=bool(args.ne3dtomo_global_ybk),
        calibration_reference_group=str(args.calibration_reference_group),
        calibration_mode=str(getattr(tomo, "cross_calibration_mode", "unknown")),
        auto_cross_calibrate_groups=bool(args.auto_cross_calibrate_groups),
        fixed_group_forward_gains_json=json.dumps(getattr(args, "fixed_group_forward_gains", {}), ensure_ascii=False),
        cross_calibration_group_keys=np.array(list(getattr(tomo, "cross_calibration_forward_gains", {}).keys()), dtype="U64"),
        cross_calibration_forward_gains=np.array(list(getattr(tomo, "cross_calibration_forward_gains", {}).values()), dtype=np.float64),
        cross_calibration_data_corrections=np.array([getattr(tomo, "cross_calibration_data_corrections", {}).get(k, np.nan) for k in getattr(tomo, "cross_calibration_forward_gains", {})], dtype=np.float64),
        cross_calibration_iterations=int(len(getattr(tomo, "cross_calibration_history", []))),
        cross_calibration_converged=bool(getattr(tomo, "cross_calibration_converged", False)),
        cross_calibration_history_json=json.dumps(getattr(tomo, "cross_calibration_history", []), ensure_ascii=False),
        cross_calibration_tolerance=float(args.cross_calibration_tolerance), cross_calibration_damping=float(args.cross_calibration_damping),
        cross_calibration_gain_min=float(args.cross_calibration_gain_min), cross_calibration_gain_max=float(args.cross_calibration_gain_max),
        cross_calibration_r_min=str(args.cross_calibration_r_min), cross_calibration_r_max=str(args.cross_calibration_r_max),
        cross_calibration_solver_maxiter=int(getattr(args, "cross_calibration_solver_maxiter", args.maxiter)),
        cross_calibration_solver_tol=float(getattr(args, "cross_calibration_solver_tol", args.tol)),
        solver_use_preconditioner=bool(getattr(args, "solver_use_preconditioner", True)),
        solver_preconditioner_floor=float(getattr(args, "solver_preconditioner_floor", 1e-12)),
        solver_iterations=int(getattr(tomo, "last_solver_iterations", 0)),
        final_solver_info=int(getattr(tomo, "final_solver_info", -1)),
        solver_diagnostics_json=json.dumps(solver_diag, ensure_ascii=False),
        positivity_diagnostics_json=json.dumps(positivity_diag, ensure_ascii=False),
        coverage_diagnostics_json=json.dumps(coverage_diag, ensure_ascii=False),
        use_ray_cache=bool(args.use_ray_cache), ray_cache_dir=str(args.ray_cache_dir),
        ray_cache_hits=int(prepared.ray_cache_hits), ray_cache_memory_hits=int(prepared.ray_cache_memory_hits),
        ray_cache_disk_hits=int(prepared.ray_cache_disk_hits), ray_cache_misses=int(prepared.ray_cache_misses),
        ray_cache_load_failures=int(prepared.ray_cache_load_failures),
        harmonic=int(args.harmonic), freq_mhz_list=np.array(freq_list, dtype=np.float64),
        r_edges=prepared.grid.r_edges.astype(np.float32), th_edges=prepared.grid.th_edges.astype(np.float32),
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
    """Clone all current base settings and apply only scenario-specific overrides."""
    cloned = SimpleNamespace(**copy.deepcopy(vars(base_args)))

    for key, value in scenario.items():
        if key in ("name", "comparison_axis", "note"):
            continue
        setattr(cloned, key, copy.deepcopy(value))

    scenario_name = str(scenario.get("name", "scenario"))
    cloned.step1_output_dir = str(scenario_dir)

    if getattr(base_args, "save_prepped_dir", ""):
        cloned.save_prepped_dir = str(scenario_dir / "prepped")

    if getattr(base_args, "save_ne_npz", ""):
        cloned.save_ne_npz = str(scenario_dir / f"{scenario_name}_ne3d_solution.npz")

    if bool(getattr(base_args, "save_png", False)):
        cloned.png_path = str(scenario_dir / f"{scenario_name}_isosurface.png")

    cloned = apply_defaults(cloned)
    # Preserve the user-requested scenario settings before auto-grid selection
    # mutates nr/nth/nph during problem preparation.  This makes resume/skip
    # reliable for long multi-scenario runs.
    cloned.step1_requested_scenario_config_hash = scenario_configuration_hash(cloned)
    return cloned




def write_comparison_key_metrics(output_root: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write a lightweight comparison CSV while retaining lambda/parameter plotting columns."""
    wanted = [
        "scenario", "comparison_axis", "parameter_name", "parameter_value", "lambda",
        "search_window_days", "observation_groups", "n_observations",
        "n_earth_view_observations", "n_cor1a_observations", "n_measurements",
        "n_voxels", "unknown_to_measurement_ratio", "out_n", "nr", "nth", "nph",
        "ds", "hm", "width_pix",
        "use_density_prior", "density_prior_model", "density_prior_scale", "density_prior_file",
        "wt_nr", "ne3dtomo_global_ybk", "calibration_mode",
        "misfit", "weighted_rms_rel", "normal_equation_relative_residual",
        "data_objective", "regularization_objective", "total_objective",
        "fmax_mhz", "volume_rsun3", "surface_area_rsun2", "components",
        "centroid_r_rsun", "r_p16_rsun", "r_median_rsun", "r_p84_rsun",
        "longitude_extent_deg", "latitude_extent_deg",
        "largest_component_fraction", "second_largest_component_fraction",
        "fit_scale_earth_merged", "fit_scale_cor1a", "cor1a_over_earth_fit_scale",
        "cor1a_forward_gain", "cor1a_data_correction", "cor1a_image_gain_p16",
        "cor1a_image_gain_median", "cor1a_image_gain_p84",
        "cross_calibration_iterations", "cross_calibration_converged", "final_solver_info",
        "solver_iterations", "solver_used_preconditioner",
        "regularization_norm", "median_obs_over_pred",
        "ray_cache_hits", "ray_cache_misses", "ray_cache_all_reused",
        "target_jaccard_vs_reference", "target_dice_vs_reference", "overlap_reference_scenario",
        "output_dir",
    ]
    key_rows = [{key: row.get(key, np.nan) for key in wanted} for row in rows]
    write_rows_csv(Path(output_root) / "step1_comparison_key_metrics.csv", key_rows)
    # Long-form-friendly copy: parameter_name/value and lambda are kept explicitly
    # for later plotting with lambda on x and the selected metric/parameter on y.
    write_rows_csv(Path(output_root) / "step1_comparison_long_form.csv", key_rows)



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
    scenario_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build one scalar-rich summary row for scenario comparison and later plotting."""
    output_dir = Path(output_dir)
    scenario_metadata = dict(scenario_metadata or {})
    stats = weighted_stats(prepared.y_obs, y_pred, tomo.W)
    reg_norm = regularization_norm(solution_raw, prepared.grid, prepared.wt_r)
    solver_diag = getattr(tomo, "final_solver_diagnostics", solver_objective_diagnostics(tomo, prepared, solution_raw))
    positivity_diag = getattr(tomo, "final_positivity_diagnostics", solution_positivity_diagnostics(solution_raw))
    coverage_diag = getattr(tomo, "coverage_diagnostics", {})

    fr = base.frequency_range_mhz_from_ne(ne, harmonic=int(prepared.args.harmonic))
    if fr is None:
        ne_min = ne_max = f_min = f_max = float("nan")
    else:
        ne_min, ne_max, f_min, f_max = [float(v) for v in fr]

    fit_scales = group_projection_fit_scales(prepared, tomo, y_pred)
    fit_scale_earth = finite_or_nan(fit_scales.get("earth_merged", np.nan))
    fit_scale_cor1a = finite_or_nan(fit_scales.get("cor1a", np.nan))
    cor1a_over_earth = float(fit_scale_cor1a / fit_scale_earth) if np.isfinite(fit_scale_earth) and fit_scale_earth > 0 and np.isfinite(fit_scale_cor1a) else np.nan

    freq_list = list(prepared.args.freq_mhz_list) if prepared.args.freq_mhz_list is not None else [float(prepared.args.freq_mhz)]
    target_rows: Dict[str, float] = {}
    first_target = None
    for freq in freq_list:
        m = target_frequency_metrics(prepared.grid, ne, float(freq), int(prepared.args.harmonic))
        if first_target is None:
            first_target = m
        prefix = f"f{float(freq):.3f}MHz_".replace(".", "p")
        for key, value in m.items():
            target_rows[prefix + key] = finite_or_nan(value)
    first_target = first_target or {}

    scan_lambdas = [float(r.lam) for r in scan_results] if scan_results else []
    finite_scan = [r for r in (scan_results or []) if np.isfinite(r.data_misfit_norm)]
    scan_best_misfit_lambda = float(min(finite_scan, key=lambda r: r.data_misfit_norm).lam) if finite_scan else np.nan
    render_camera_time = base.parse_target_datetime(prepared.args.target_time)
    render_camera_lonlat = earth_view_camera_lonlat_from_target_time(prepared.args.target_time)

    calibration_rows = list(getattr(tomo, "per_image_calibration_rows", []))
    cor_rows = [r for r in calibration_rows if r.get("group") == "cor1a" and np.isfinite(finite_or_nan(r.get("fit_gain_model_to_observed")))]
    cor_vals = np.asarray([float(r["fit_gain_model_to_observed"]) for r in cor_rows], dtype=float) if cor_rows else np.array([], dtype=float)
    if cor_vals.size:
        cor_p16, cor_med, cor_p84 = np.nanpercentile(cor_vals, [16, 50, 84])
    else:
        cor_p16 = cor_med = cor_p84 = np.nan

    parameter_name = str(scenario_metadata.get("parameter_name", comparison_axis))
    parameter_value = scenario_metadata.get("parameter_value", scenario_metadata.get(parameter_name, np.nan))
    observation_group_keys = [
        base.tomography_observation_group_key(path)
        for path in prepared.pb_paths
    ]
    n_earth_view_observations = sum(key.startswith("earth_") for key in observation_group_keys)
    n_cor1a_observations = sum(key == "cor1a" for key in observation_group_keys)
    row: Dict[str, object] = {
        "scenario": str(scenario_name), "comparison_axis": str(comparison_axis),
        "scenario_config_hash": str(getattr(
            prepared.args,
            "step1_requested_scenario_config_hash",
            scenario_configuration_hash(prepared.args),
        )),
        "parameter_name": parameter_name, "parameter_value": parameter_value,
        "scenario_note": str(scenario_metadata.get("note", "")), "output_dir": str(output_dir),
        "target_time": str(getattr(prepared.args, "target_time", "")),
        "render_camera_is_earth_view": True, "render_camera_mode": "forced_target_time_sub_earth",
        "render_camera_source": "sunpy.coordinates.sun.L0_B0",
        "render_camera_time_utc": render_camera_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "render_camera_lon_deg": float(render_camera_lonlat[0]), "render_camera_lat_deg": float(render_camera_lonlat[1]),
        "render_camera_lonlat_frame": "Carrington sub-Earth L0 / B0",
        "search_window_days": finite_or_nan(getattr(prepared.args, "search_window_days", np.nan)),
        "observation_groups": json.dumps(sorted(set(observation_group_keys)), ensure_ascii=False),
        "n_earth_view_observations": int(n_earth_view_observations),
        "n_cor1a_observations": int(n_cor1a_observations),
        "n_observations": int(len(prepared.pb_paths)), "n_measurements": int(np.asarray(prepared.y_obs).size),
        "n_voxels": int(prepared.grid.nvox),
        "unknown_to_measurement_ratio": float(prepared.grid.nvox / max(1, np.asarray(prepared.y_obs).size)),
        "out_n": int(getattr(prepared.args, "out_n", -1)),
        "r_min": finite_or_nan(getattr(prepared.args, "r_min", np.nan)), "r_max": finite_or_nan(getattr(prepared.args, "r_max", np.nan)),
        "nr": int(getattr(prepared.args, "nr", -1)), "nth": int(getattr(prepared.args, "nth", -1)), "nph": int(getattr(prepared.args, "nph", -1)),
        "ds": finite_or_nan(getattr(prepared.args, "ds", np.nan)), "hm": int(getattr(prepared.args, "hm", -1)),
        "width_pix": finite_or_nan(getattr(prepared.args, "width_pix", np.nan)),
        "despike_nsig": finite_or_nan(getattr(prepared.args, "despike_nsig", np.nan)),
        "lambda": float(final_lambda), "lcurve_lambda_candidate": finite_or_nan(lcurve_candidate),
        "scan_lambdas": json.dumps(scan_lambdas, ensure_ascii=False), "scan_best_misfit_lambda": finite_or_nan(scan_best_misfit_lambda),
        "density_prior_model": str(getattr(prepared.args, "density_prior_model", "")),
        "density_prior_scale": finite_or_nan(getattr(prepared.args, "density_prior_scale", np.nan)),
        "density_prior_file": str(getattr(prepared.args, "density_prior_file", "")),
        "use_density_prior": bool(getattr(prepared.args, "use_density_prior", True)) and str(getattr(prepared.args, "density_prior_model", "none")).strip().lower() not in ("", "none", "off", "false", "0"),
        "positivity_method": str(getattr(prepared.args, "positivity_method", "clip")),
        "thomson_normalize_msb": bool(getattr(prepared.args, "thomson_normalize_msb", True)),
        "thomson_kernel_scale": finite_or_nan(getattr(prepared.args, "thomson_kernel_scale", 1.0)),
        "wt_nr": int(getattr(prepared.args, "wt_nr", 0)),
        "ne3dtomo_global_ybk": bool(getattr(prepared.args, "ne3dtomo_global_ybk", False)),
        "r_use_min": finite_or_nan(getattr(prepared.args, "r_use_min", np.nan)), "r_use_max": finite_or_nan(getattr(prepared.args, "r_use_max", np.nan)),
        "r_use_min_by_group": json.dumps(getattr(prepared.args, "r_use_min_by_group", {}), ensure_ascii=False),
        "pb_scale_by_group": json.dumps(getattr(prepared.args, "pb_scale_by_group", {}), ensure_ascii=False),
        "misfit": finite_or_nan(stats.get("misfit_rms")), "misfit_norm": finite_or_nan(stats.get("misfit_norm")),
        "weighted_rms_rel": finite_or_nan(stats.get("weighted_rms_rel")), "regularization_norm": finite_or_nan(reg_norm),
        "median_obs_over_pred": finite_or_nan(stats.get("median_obs_over_pred")),
        "normal_equation_relative_residual": finite_or_nan(solver_diag.get("normal_equation_relative_residual")),
        "data_objective": finite_or_nan(solver_diag.get("data_objective")),
        "regularization_objective": finite_or_nan(solver_diag.get("regularization_objective")),
        "total_objective": finite_or_nan(solver_diag.get("total_objective")),
        **{k: finite_or_nan(v) for k, v in positivity_diag.items()},
        **{k: finite_or_nan(v) for k, v in coverage_diag.items()},
        "ne_min_cm3": finite_or_nan(ne_min), "ne_max_cm3": finite_or_nan(ne_max),
        "f_min_mhz": finite_or_nan(f_min), "fmax_mhz": finite_or_nan(f_max), "f_max_mhz": finite_or_nan(f_max),
        "volume_rsun3": finite_or_nan(first_target.get("volume_ge_target_rsun3")),
        "surface_area_rsun2": finite_or_nan(first_target.get("surface_area_rsun2")),
        "components": int(first_target.get("n_components", 0)),
        "centroid_r_rsun": finite_or_nan(first_target.get("centroid_r_rsun")),
        "r_min_target_rsun": finite_or_nan(first_target.get("r_min_rsun")),
        "r_p16_rsun": finite_or_nan(first_target.get("r_p16_rsun")),
        "r_median_rsun": finite_or_nan(first_target.get("r_median_rsun")),
        "r_p84_rsun": finite_or_nan(first_target.get("r_p84_rsun")),
        "r_max_target_rsun": finite_or_nan(first_target.get("r_max_rsun")),
        "longitude_extent_deg": finite_or_nan(first_target.get("longitude_extent_deg")),
        "latitude_extent_deg": finite_or_nan(first_target.get("latitude_extent_deg")),
        "largest_component_fraction": finite_or_nan(first_target.get("largest_component_fraction")),
        "second_largest_component_fraction": finite_or_nan(first_target.get("second_largest_component_fraction")),
        "fit_scale_earth_merged": fit_scale_earth, "fit_scale_cor1a": fit_scale_cor1a,
        "cor1a_over_earth_fit_scale": finite_or_nan(cor1a_over_earth),
        "limb_u_mode": str(getattr(prepared.args, "limb_u_mode", "fixed")),
        "limb_u_use_allen": bool(getattr(prepared.args, "limb_u_use_allen", False)),
        "limb_u_bandpass_nm_by_instrument": json.dumps(
            getattr(prepared.args, "limb_u_bandpass_nm_by_instrument", {}), ensure_ascii=False
        ),
        "limb_u_override_by_instrument": json.dumps(
            getattr(prepared.args, "limb_u_override_by_instrument", {}), ensure_ascii=False
        ),
        "observation_limb_models": json.dumps(
            [obs.limb_u_model for obs in prepared.obs_list], ensure_ascii=False
        ),
        "observation_limb_components": json.dumps(
            [list(obs.limb_component_names) for obs in prepared.obs_list], ensure_ascii=False
        ),
        "observation_limb_u_eff": json.dumps(
            [list(obs.limb_u_values) for obs in prepared.obs_list], ensure_ascii=False
        ),
        "calibration_mode": str(getattr(tomo, "cross_calibration_mode", "unknown")),
        "auto_cross_calibrate_groups": bool(getattr(prepared.args, "auto_cross_calibrate_groups", False)),
        "fixed_group_forward_gains": json.dumps(getattr(prepared.args, "fixed_group_forward_gains", {}), ensure_ascii=False),
        "cross_calibration_reference_group": str(getattr(prepared.args, "calibration_reference_group", "earth_merged")),
        "cross_calibration_forward_gains": json.dumps(getattr(tomo, "cross_calibration_forward_gains", {}), ensure_ascii=False),
        "cross_calibration_data_corrections": json.dumps(getattr(tomo, "cross_calibration_data_corrections", {}), ensure_ascii=False),
        "cross_calibration_iterations": int(len(getattr(tomo, "cross_calibration_history", []))),
        "cross_calibration_converged": bool(getattr(tomo, "cross_calibration_converged", False)),
        "cor1a_forward_gain": finite_or_nan(getattr(tomo, "cross_calibration_forward_gains", {}).get("cor1a", np.nan)),
        "cor1a_data_correction": finite_or_nan(getattr(tomo, "cross_calibration_data_corrections", {}).get("cor1a", np.nan)),
        "cor1a_image_gain_p16": finite_or_nan(cor_p16), "cor1a_image_gain_median": finite_or_nan(cor_med), "cor1a_image_gain_p84": finite_or_nan(cor_p84),
        "final_solver_info": int(getattr(tomo, "final_solver_info", -1)),
        "solver_iterations": int(getattr(tomo, "last_solver_iterations", 0)),
        "solver_used_preconditioner": bool(getattr(tomo, "last_solver_used_preconditioner", False)),
        "solver_preconditioner_floor": finite_or_nan(getattr(prepared.args, "solver_preconditioner_floor", np.nan)),
        "cross_calibration_solver_maxiter": int(getattr(prepared.args, "cross_calibration_solver_maxiter", prepared.args.maxiter)),
        "cross_calibration_solver_tol": finite_or_nan(getattr(prepared.args, "cross_calibration_solver_tol", prepared.args.tol)),
        "suggested_brightness_scale": finite_or_nan(suggested_scale),
        "apply_brightness_scale": bool(getattr(prepared.args, "apply_brightness_scale", False)),
        "ray_cache_enabled": not prepared.ray_cache_disabled,
        "ray_cache_hits": int(prepared.ray_cache_hits), "ray_cache_memory_hits": int(prepared.ray_cache_memory_hits),
        "ray_cache_disk_hits": int(prepared.ray_cache_disk_hits), "ray_cache_misses": int(prepared.ray_cache_misses),
        "ray_cache_load_failures": int(prepared.ray_cache_load_failures),
        "ray_cache_all_reused": bool((not prepared.ray_cache_disabled) and prepared.ray_cache_misses == 0 and prepared.ray_cache_hits == len(prepared.pb_paths)),
    }
    row.update(target_rows)
    return row




def scenario_parameter_metadata(scenario: Dict[str, object], scenario_args: SimpleNamespace) -> Dict[str, object]:
    axis = str(scenario.get("comparison_axis", "unspecified"))
    mapping = {
        "lambda": ("lambda", getattr(scenario_args, "lam", np.nan)),
        "time_window": ("search_window_days", getattr(scenario_args, "search_window_days", np.nan)),
        "time_window_observation_set": (
            "time_window_observation_set",
            scenario.get(
                "parameter_value",
                f"{getattr(scenario_args, 'search_window_days', np.nan)}d / "
                f"{getattr(scenario_args, 'observation_groups', [])}",
            ),
        ),
        "grid_resolution": ("grid", f"{scenario_args.nr}x{scenario_args.nth}x{scenario_args.nph}"),
        "density_prior": ("density_prior_model", getattr(scenario_args, "density_prior_model", "")),
        "density_prior_scale": ("density_prior_scale", getattr(scenario_args, "density_prior_scale", np.nan)),
        "weighting": ("wt_nr", getattr(scenario_args, "wt_nr", np.nan)),
        "global_ybk": ("ne3dtomo_global_ybk", getattr(scenario_args, "ne3dtomo_global_ybk", False)),
        "cross_calibration": ("calibration_mode", "automatic" if getattr(scenario_args, "auto_cross_calibrate_groups", False) else ("fixed" if getattr(scenario_args, "fixed_group_forward_gains", {}) else "none")),
        "calibration_initial_gain": ("cor1a_initial_gain", getattr(scenario_args, "cross_calibration_initial_gain_by_group", {}).get("cor1a", np.nan)),
        "calibration_radial_range": ("calibration_radial_range", f"{scenario_args.cross_calibration_r_min}-{scenario_args.cross_calibration_r_max}"),
        "hm": ("hm", getattr(scenario_args, "hm", np.nan)),
        "width_pix": ("width_pix", getattr(scenario_args, "width_pix", np.nan)),
        "ds": ("ds", getattr(scenario_args, "ds", np.nan)),
        "positivity": ("positivity_method", getattr(scenario_args, "positivity_method", "")),
        "validation": ("validation", scenario.get("name", "")),
    }
    name, value = mapping.get(axis, (axis, scenario.get("parameter_value", np.nan)))
    return {"parameter_name": name, "parameter_value": value, "note": scenario.get("note", "")}



def build_time_window_observation_set_scenarios(
    time_windows_days: Sequence[float] = (3.0, 5.0, 7.0),
) -> List[Dict[str, object]]:
    """Build the 3 time windows x 3 observation sets comparison matrix.

    For every time window, the automatic file search is first run with both
    Earth-view and COR1A enabled.  ``observation_groups`` is then applied inside
    ``prepare_tomography_problem``.  Consequently, the Earth-only and COR1A-only
    inputs are exact subsets of the corresponding joint selection.
    """
    scenarios: List[Dict[str, object]] = []
    for window_days in time_windows_days:
        window = float(window_days)
        if not np.isfinite(window) or window <= 0:
            raise ValueError(f"Invalid comparison time window: {window_days!r}")
        window_tag = f"{window:g}d".replace(".", "p")

        scenarios.extend([
            {
                "name": f"time_window_{window_tag}_earth_cor1a",
                "comparison_axis": "time_window_observation_set",
                "parameter_value": f"{window:g} days / Earth-view+COR1A",
                "search_window_days": window,
                "include_kcor_lasco": True,
                "include_cor1a": True,
                "observation_groups": ["earth_merged", "earth_lasco_only", "cor1a"],
                "auto_cross_calibrate_groups": True,
                "fixed_group_forward_gains": {},
                "calibration_reference_group": "earth_merged",
                "step1_run_lambda_scan": False,
                "note": (
                    "Joint reconstruction. Single-view cases for the same window "
                    "are filtered from this joint file selection."
                ),
            },
            {
                "name": f"time_window_{window_tag}_earth_only",
                "comparison_axis": "time_window_observation_set",
                "parameter_value": f"{window:g} days / Earth-view only",
                "search_window_days": window,
                "include_kcor_lasco": True,
                "include_cor1a": True,
                "observation_groups": ["earth_merged", "earth_lasco_only"],
                "auto_cross_calibrate_groups": False,
                "fixed_group_forward_gains": {},
                "cross_calibration_initial_gain_by_group": {},
                "calibration_reference_group": "earth_merged",
                "step1_run_lambda_scan": False,
                "note": "Single-view reconstruction using only the Earth-view subset.",
            },
            {
                "name": f"time_window_{window_tag}_cor1a_only",
                "comparison_axis": "time_window_observation_set",
                "parameter_value": f"{window:g} days / COR1A only",
                "search_window_days": window,
                "include_kcor_lasco": True,
                "include_cor1a": True,
                "observation_groups": ["cor1a"],
                "auto_cross_calibrate_groups": False,
                "fixed_group_forward_gains": {},
                "cross_calibration_initial_gain_by_group": {},
                "calibration_reference_group": "cor1a",
                "step1_run_lambda_scan": False,
                "note": "Single-view reconstruction using only the COR1A subset.",
            },
        ])
    return scenarios



def scenario_configuration_hash(args: SimpleNamespace) -> str:
    """Hash the scientific and numerical settings that define one final solution."""
    fields = [
        "target_time", "search_window_days", "auto_find_pb_fits",
        "exclude_earth_times", "keep_cor1a_for_excluded_earth_times",
        "observation_groups", "include_kcor_lasco", "include_cor1a", "include_lasco_only",
        "deduplicate_pb_fits", "out_n", "r_min", "r_max", "nr", "nth", "nph",
        "ds", "limb_u", "filt", "despike_nsig", "despike_med", "pb_floor",
        "dpa_deg", "r_use_min", "r_use_max", "r_use_min_by_group",
        "r_use_max_by_group", "pb_scale_by_group", "hm", "wt_nr", "lam",
        "q_low", "width_pix", "maxiter", "tol", "positivity_method",
        "use_density_prior", "density_prior_model", "density_prior_scale",
        "density_prior_file", "limb_u", "limb_u_mode", "limb_u_use_allen",
        "limb_u_bandpass_nm_by_instrument", "limb_u_override_by_instrument",
        "limb_u_bandpass_samples", "limb_u_weight_hdu_names",
        "thomson_normalize_msb", "thomson_kernel_scale", "use_temporal_despike",
        "ne3dtomo_global_ybk", "calibration_reference_group",
        "auto_cross_calibrate_groups", "fixed_group_forward_gains",
        "cross_calibration_initial_gain_by_group", "cross_calibration_max_iterations",
        "cross_calibration_tolerance", "cross_calibration_damping",
        "cross_calibration_gain_min", "cross_calibration_gain_max",
        "cross_calibration_r_min", "cross_calibration_r_max",
        "cross_calibration_min_count", "cross_calibration_clip_sigma",
        "cross_calibration_solver_maxiter", "cross_calibration_solver_tol",
        "solver_use_preconditioner", "solver_preconditioner_floor",
        "harmonic", "freq_mhz_list", "auto_grid_max_m_over_n",
    ]
    payload = {key: copy.deepcopy(getattr(args, key, None)) for key in fields}
    payload["nr"] = "auto_from_measurements"
    payload["nth"] = 60
    payload["nph"] = 120
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _selected_observation_rows(prepared: PreparedProblem) -> List[Dict[str, object]]:
    return [{
        "index": i,
        "name": path.name,
        "path": str(path),
        "group": base.tomography_observation_group_key(path),
        "datetime": str(base.parse_pb_filename_datetime(path)),
        "used_pixels": int(prepared.obs_list[i].idx_map.size),
        "r_use_min": prepared.obs_r_bounds[i][0],
        "r_use_max": prepared.obs_r_bounds[i][1],
    } for i, path in enumerate(prepared.pb_paths)]


def _try_load_completed_scenario(
    scenario_args: SimpleNamespace,
    scenario_dir: Path,
) -> Optional[Tuple[Dict[str, object], Tuple[object, np.ndarray]]]:
    """Load a previously completed, matching, converged scenario for safe resume."""
    if not bool(getattr(scenario_args, "step1_skip_completed_scenarios", True)):
        return None
    summary_path = Path(scenario_dir) / "step1_final_summary.csv"
    if not summary_path.exists():
        return None
    try:
        with summary_path.open("r", newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))
    except Exception as exc:
        print(f"[RESUME] Could not read {summary_path}: {exc}; scenario will be recomputed.")
        return None

    expected_hash = str(getattr(
        scenario_args,
        "step1_requested_scenario_config_hash",
        scenario_configuration_hash(scenario_args),
    ))
    if str(row.get("scenario_config_hash", "")) != expected_hash:
        print("[RESUME] Existing summary has different/legacy settings; scenario will be recomputed.")
        return None

    try:
        solver_info = int(float(row.get("final_solver_info", "-1")))
    except Exception:
        solver_info = -1
    allow_unconverged = bool(getattr(scenario_args, "step1_skip_completed_unconverged", False))
    if solver_info != 0 and not allow_unconverged:
        print(
            f"[RESUME] Existing scenario is not converged (final_solver_info={solver_info}); "
            "it will be recomputed with the accelerated solver."
        )
        return None

    npz_path = Path(getattr(scenario_args, "save_ne_npz", ""))
    if not npz_path.exists():
        print(f"[RESUME] Summary exists but NPZ is missing: {npz_path}; recomputing.")
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            ne = np.asarray(z["ne"], dtype=np.float64).ravel()
            grid = base.SphericalGrid(
                r_edges=np.asarray(z["r_edges"], dtype=np.float64),
                th_edges=np.asarray(z["th_edges"], dtype=np.float64),
                ph_edges=np.asarray(z["ph_edges"], dtype=np.float64),
            )
        freq_list = list(scenario_args.freq_mhz_list) if scenario_args.freq_mhz_list is not None else [float(scenario_args.freq_mhz)]
        mask = target_frequency_mask(grid, ne, float(freq_list[0]), int(scenario_args.harmonic))
    except Exception as exc:
        print(f"[RESUME] Could not restore target mask from {npz_path}: {exc}; recomputing.")
        return None

    print(f"[RESUME-SKIP] Reusing completed scenario: {scenario_dir.name}")
    return row, (grid, mask)


def _run_fast_lambda_comparison_batch(
    args: SimpleNamespace,
    lambda_scenarios: Sequence[Dict[str, object]],
    output_root: Path,
    summary_rows: List[Dict[str, object]],
    target_records: Dict[str, Tuple[object, np.ndarray]],
) -> None:
    """Evaluate lambda=1..19 with one preprocessing pass, CSR matrix, and gain fit."""
    if not lambda_scenarios:
        return
    values = sorted({float(s["lam"]) for s in lambda_scenarios})
    shared_dir = ensure_dir(Path(output_root) / "_lambda_shared")
    first = dict(lambda_scenarios[0])
    first["name"] = "_lambda_shared"
    first["lam"] = float(getattr(args, "lam", 5.0))
    first["step1_run_lambda_scan"] = True
    first["step1_lambda_values"] = values
    first["step1_compute_coverage"] = False
    first["step1_save_coverage_npz"] = False
    first["step1_save_target_mask_npz"] = False
    shared_args = clone_args_for_scenario(args, first, shared_dir)

    # Skip the entire batch only when every lambda has a matching reusable result.
    loaded = []
    for scenario in lambda_scenarios:
        scenario_name = str(scenario["name"])
        scenario_dir = ensure_dir(Path(output_root) / scenario_name)
        scenario_args = clone_args_for_scenario(args, scenario, scenario_dir)
        restored = _try_load_completed_scenario(scenario_args, scenario_dir)
        if restored is None:
            loaded = []
            break
        loaded.append((scenario_name, restored))
    if loaded and len(loaded) == len(lambda_scenarios):
        for scenario_name, (row, record) in loaded:
            summary_rows.append(row)
            target_records[scenario_name] = record
        write_rows_csv(Path(output_root) / "step1_comparison_suite_summary.csv", summary_rows)
        write_comparison_key_metrics(Path(output_root), summary_rows)
        print("[FAST-LAMBDA] All lambda scenarios were restored from converged outputs.")
        return

    print("=" * 80)
    print(
        f"[FAST-LAMBDA] Shared lambda batch: values={values}; "
        f"calibration_lambda={shared_args.lam}"
    )
    print("[FAST-LAMBDA] One gain fit is held fixed so that only lambda changes.")
    print("=" * 80)
    prepared = prepare_tomography_problem(shared_args)
    selected_rows = _selected_observation_rows(prepared)
    write_rows_csv(shared_dir / "step1_selected_observations.csv", selected_rows)
    scan_results, lcurve_candidate, tomo = run_lambda_scan(
        prepared,
        values,
        shared_dir,
        keep_solutions=True,
    )
    by_lambda = {float(r.lam): r for r in scan_results}

    for scenario in lambda_scenarios:
        scenario_name = str(scenario["name"])
        comparison_axis = str(scenario.get("comparison_axis", "lambda"))
        lam = float(scenario["lam"])
        result = by_lambda[lam]
        scenario_dir = ensure_dir(Path(output_root) / scenario_name)
        scenario_args = clone_args_for_scenario(args, scenario, scenario_dir)
        prepared_i = copy.copy(prepared)
        prepared_i.args = scenario_args
        write_rows_csv(scenario_dir / "step1_selected_observations.csv", selected_rows)

        tomo_i, solution_raw, ne, y_pred, suggested_scale = run_final_diagnostics(
            prepared=prepared_i,
            lam=lam,
            output_dir=scenario_dir,
            save_residual_npy=bool(getattr(scenario_args, "step1_save_residual_npy", False)),
            save_residual_png=bool(getattr(scenario_args, "step1_save_residual_png", False)),
            existing_tomo=tomo,
            precomputed_solution_raw=result.solution_raw,
            precomputed_info=result.info,
            precomputed_iterations=result.solver_iterations,
            recalibrate_cross_calibration=False,
        )
        metadata = scenario_parameter_metadata(scenario, scenario_args)
        summary_row = make_final_summary_row(
            prepared=prepared_i,
            tomo=tomo_i,
            solution_raw=solution_raw,
            ne=ne,
            y_pred=y_pred,
            suggested_scale=suggested_scale,
            final_lambda=lam,
            output_dir=scenario_dir,
            scenario_name=scenario_name,
            comparison_axis=comparison_axis,
            scan_results=scan_results,
            lcurve_candidate=lcurve_candidate,
            scenario_metadata=metadata,
        )
        write_rows_csv(scenario_dir / "step1_final_summary.csv", [summary_row])
        if getattr(scenario_args, "save_ne_npz", ""):
            save_final_npz(
                prepared_i, tomo_i, solution_raw, ne, y_pred,
                suggested_scale, lam, Path(scenario_args.save_ne_npz),
            )
        if bool(getattr(scenario_args, "save_png", False)):
            save_final_png(prepared_i, ne, Path(scenario_args.png_path))

        freq_list = list(scenario_args.freq_mhz_list) if scenario_args.freq_mhz_list is not None else [float(scenario_args.freq_mhz)]
        target_records[scenario_name] = (
            prepared_i.grid,
            target_frequency_mask(prepared_i.grid, ne, float(freq_list[0]), int(scenario_args.harmonic)),
        )
        summary_rows.append(summary_row)
        write_rows_csv(Path(output_root) / "step1_comparison_suite_summary.csv", summary_rows)
        write_comparison_key_metrics(Path(output_root), summary_rows)
        print(
            f"[FAST-LAMBDA] Completed {scenario_name}: iterations={result.solver_iterations}, "
            f"cg_info={result.info}"
        )


def run_comparison_suite(args: SimpleNamespace) -> None:
    """Run comparison scenarios, preserve lambda-oriented tables, and compute overlap diagnostics."""
    args = apply_defaults(args)
    scenarios = list(getattr(args, "step1_comparison_scenarios", []))
    if not scenarios:
        raise ValueError("step1_comparison_scenarios is empty. Define scenarios before calling run_comparison_suite().")

    if getattr(args, "step1_comparison_output_dir", ""):
        output_root = ensure_dir(getattr(args, "step1_comparison_output_dir"))
    else:
        try:
            target_tag = base.parse_target_datetime(getattr(args, "target_time", "")).strftime("%Y%m%d_%H%M%S")
        except Exception:
            target_tag = now_tag()
        freq_list = list(args.freq_mhz_list) if getattr(args, "freq_mhz_list", None) is not None else [float(getattr(args, "freq_mhz", 0.0))]
        freq_tag = "-".join(str(float(f)).rstrip("0").rstrip(".") for f in freq_list)
        parent = Path(args.data_dir) if getattr(args, "data_dir", "") else Path(getattr(args, "step1_output_dir", "step1_diagnostics")).parent
        output_root = ensure_dir(parent / f"step1_comparison_suite_{target_tag}_{freq_tag}MHz")

    print(f"[STEP1-COMP] Comparison suite output root: {output_root}")
    summary_rows: List[Dict[str, object]] = []
    target_records: Dict[str, Tuple[object, np.ndarray]] = {}
    lambda_batch_processed = False
    lambda_scenarios = [
        scenario for scenario in scenarios
        if str(scenario.get("comparison_axis", "")) == "lambda"
    ]

    for index, scenario in enumerate(scenarios, start=1):
        scenario_name = str(scenario.get("name", f"scenario_{index:03d}"))
        comparison_axis = str(scenario.get("comparison_axis", "unspecified"))
        if (
            comparison_axis == "lambda"
            and bool(getattr(args, "step1_fast_lambda_suite", True))
        ):
            if not lambda_batch_processed:
                _run_fast_lambda_comparison_batch(
                    args,
                    lambda_scenarios,
                    output_root,
                    summary_rows,
                    target_records,
                )
                lambda_batch_processed = True
            continue

        scenario_dir = ensure_dir(output_root / scenario_name)
        print("=" * 80)
        print(f"[STEP1-COMP] Scenario {index}/{len(scenarios)}: {scenario_name}")
        print(f"[STEP1-COMP] comparison_axis={comparison_axis}")
        print(f"[STEP1-COMP] scenario_dir={scenario_dir}")
        print("=" * 80)

        scenario_args = clone_args_for_scenario(args, scenario, scenario_dir)
        restored = _try_load_completed_scenario(scenario_args, scenario_dir)
        if restored is not None:
            row, record = restored
            summary_rows.append(row)
            target_records[scenario_name] = record
            write_rows_csv(output_root / "step1_comparison_suite_summary.csv", summary_rows)
            write_comparison_key_metrics(output_root, summary_rows)
            continue

        lambda_values = [float(v) for v in getattr(scenario_args, "step1_lambda_values", [scenario_args.lam])]
        if not lambda_values:
            lambda_values = [float(scenario_args.lam)]
        prepared = prepare_tomography_problem(scenario_args)

        file_rows = _selected_observation_rows(prepared)
        write_rows_csv(scenario_dir / "step1_selected_observations.csv", file_rows)

        scan_results: List[LambdaResult] = []
        lcurve_candidate: Optional[float] = None
        if bool(getattr(scenario_args, "step1_run_lambda_scan", True)):
            scan_results, lcurve_candidate, _scan_tomo = run_lambda_scan(
                prepared, lambda_values, scenario_dir, keep_solutions=False
            )
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

        scan_gain_seed = scan_results[0].group_forward_gains if scan_results else None
        recalibrate_final = bool(getattr(scenario_args, "cross_calibration_recalibrate_after_lambda_selection", True)) or not bool(scan_gain_seed)
        tomo, solution_raw, ne, y_pred, suggested_scale = run_final_diagnostics(
            prepared=prepared, lam=final_lam, output_dir=scenario_dir,
            save_residual_npy=bool(getattr(scenario_args, "step1_save_residual_npy", False)),
            save_residual_png=bool(getattr(scenario_args, "step1_save_residual_png", False)),
            initial_group_forward_gains=scan_gain_seed,
            recalibrate_cross_calibration=recalibrate_final,
        )

        if bool(getattr(scenario_args, "step1_run_leave_one_image_out", False)):
            run_leave_one_image_out(
                prepared, lambdas=[final_lam], output_dir=scenario_dir,
                max_holdouts=getattr(scenario_args, "step1_loo_max_holdouts", None),
            )
        if bool(getattr(scenario_args, "step1_run_block_holdout", False)):
            run_time_block_holdout(
                prepared, final_lam, scenario_dir,
                block_days=float(getattr(scenario_args, "step1_block_holdout_days", 1.0)),
                max_blocks=getattr(scenario_args, "step1_block_holdout_max_blocks", 6),
            )

        metadata = scenario_parameter_metadata(scenario, scenario_args)
        summary_row = make_final_summary_row(
            prepared=prepared, tomo=tomo, solution_raw=solution_raw, ne=ne, y_pred=y_pred,
            suggested_scale=suggested_scale, final_lambda=final_lam, output_dir=scenario_dir,
            scenario_name=scenario_name, comparison_axis=comparison_axis,
            scan_results=scan_results, lcurve_candidate=lcurve_candidate,
            scenario_metadata=metadata,
        )
        write_rows_csv(scenario_dir / "step1_final_summary.csv", [summary_row])
        print(f"[STEP1-COMP] Saved scenario summary: {scenario_dir / 'step1_final_summary.csv'}")

        if getattr(scenario_args, "save_ne_npz", ""):
            save_final_npz(prepared, tomo, solution_raw, ne, y_pred, suggested_scale, final_lam, Path(scenario_args.save_ne_npz))
        if bool(getattr(scenario_args, "save_png", False)):
            save_final_png(prepared, ne, Path(scenario_args.png_path))

        freq_list = list(scenario_args.freq_mhz_list) if scenario_args.freq_mhz_list is not None else [float(scenario_args.freq_mhz)]
        first_freq = float(freq_list[0])
        target_records[scenario_name] = (
            prepared.grid,
            target_frequency_mask(prepared.grid, ne, first_freq, int(scenario_args.harmonic)),
        )
        summary_rows.append(summary_row)
        write_rows_csv(output_root / "step1_comparison_suite_summary.csv", summary_rows)
        write_comparison_key_metrics(output_root, summary_rows)
        print(f"[STEP1-COMP] Updated: {output_root / 'step1_comparison_suite_summary.csv'}")
        print(f"[STEP1-COMP] Updated: {output_root / 'step1_comparison_key_metrics.csv'}")

    if bool(getattr(args, "step1_write_overlap_metrics", True)) and target_records:
        reference_name = str(getattr(args, "step1_overlap_reference_scenario", "lambda_5"))
        if reference_name not in target_records:
            reference_name = next(iter(target_records))
            print(f"[WARN] Requested overlap reference not found; using {reference_name!r}.")
        ref_grid, ref_mask = target_records[reference_name]
        overlap_rows = []
        summary_by_name = {str(r["scenario"]): r for r in summary_rows}
        for name, (grid, mask) in target_records.items():
            ref_on_grid = ref_mask if grid is ref_grid else resample_mask_to_grid(ref_mask, ref_grid, grid)
            metrics = mask_overlap_metrics(ref_on_grid, mask, grid)
            overlap_row = {"reference_scenario": reference_name, "scenario": name, **metrics}
            overlap_rows.append(overlap_row)
            row = summary_by_name[name]
            row["overlap_reference_scenario"] = reference_name
            row["target_jaccard_vs_reference"] = metrics["jaccard"]
            row["target_dice_vs_reference"] = metrics["dice"]
            row["target_intersection_volume_rsun3"] = metrics["intersection_volume_rsun3"]
            row["target_union_volume_rsun3"] = metrics["union_volume_rsun3"]
        write_rows_csv(output_root / "step1_overlap_vs_reference.csv", overlap_rows)
        write_rows_csv(output_root / "step1_comparison_suite_summary.csv", summary_rows)
        write_comparison_key_metrics(output_root, summary_rows)
        print(f"[STEP1-COMP] Saved overlap metrics: {output_root / 'step1_overlap_vs_reference.csv'}")

    print("[STEP1-COMP] Completed comparison suite.")
    print(f"[STEP1-COMP] Summary CSV: {output_root / 'step1_comparison_suite_summary.csv'}")
    print(f"[STEP1-COMP] Key metrics CSV: {output_root / 'step1_comparison_key_metrics.csv'}")
    print(f"[STEP1-COMP] Long-form plotting CSV: {output_root / 'step1_comparison_long_form.csv'}")


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
    file_rows = [{
        "index": i, "name": p.name, "path": str(p),
        "group": base.tomography_observation_group_key(p),
        "datetime": str(base.parse_pb_filename_datetime(p)),
        "used_pixels": int(prepared.obs_list[i].idx_map.size),
        "r_use_min": prepared.obs_r_bounds[i][0], "r_use_max": prepared.obs_r_bounds[i][1],
    } for i, p in enumerate(prepared.pb_paths)]
    write_rows_csv(step1_output_dir / "step1_selected_observations.csv", file_rows)

    scan_results: List[LambdaResult] = []
    lcurve_candidate: Optional[float] = None
    if bool(getattr(args, "step1_run_lambda_scan", True)):
        scan_results, lcurve_candidate, _scan_tomo = run_lambda_scan(
            prepared, lambda_values, step1_output_dir, keep_solutions=False
        )
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

    scan_gain_seed = scan_results[0].group_forward_gains if scan_results else None
    recalibrate_final = bool(getattr(args, "cross_calibration_recalibrate_after_lambda_selection", True)) or not bool(scan_gain_seed)
    tomo, solution_raw, ne, y_pred, suggested_scale = run_final_diagnostics(
        prepared=prepared, lam=final_lam, output_dir=step1_output_dir,
        save_residual_npy=bool(getattr(args, "step1_save_residual_npy", True)),
        save_residual_png=bool(getattr(args, "step1_save_residual_png", False)),
        initial_group_forward_gains=scan_gain_seed,
        recalibrate_cross_calibration=recalibrate_final,
    )
    if bool(getattr(args, "step1_run_leave_one_image_out", False)):
        run_leave_one_image_out(
            prepared,
            lambdas=[final_lam] if bool(getattr(args, "step1_loo_final_lambda_only", True)) else lambda_values,
            output_dir=step1_output_dir,
            max_holdouts=getattr(args, "step1_loo_max_holdouts", None),
        )
    if bool(getattr(args, "step1_run_block_holdout", False)):
        run_time_block_holdout(
            prepared, final_lam, step1_output_dir,
            block_days=float(getattr(args, "step1_block_holdout_days", 1.0)),
            max_blocks=getattr(args, "step1_block_holdout_max_blocks", 6),
        )
    if args.save_ne_npz:
        save_final_npz(prepared, tomo, solution_raw, ne, y_pred, suggested_scale, final_lam, Path(args.save_ne_npz))
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
    SEARCH_WINDOW_DAYS = 5.0
    EXCLUDE_EARTH_TIMES = [
        "20220606_0258",
        "20220612_0258",
        "20220614_0258",
        "20220616_2104",
        "20220617_0258",
        "20220617_2104",
    ]
    KEEP_COR1A_FOR_EXCLUDED_EARTH_TIMES = True

    AUTO_FIND_PB_FITS = True
    INCLUDE_KCOR_LASCO = True
    INCLUDE_COR1A = True
    INCLUDE_LASCO_ONLY = False
    DEDUPLICATE_PB_FITS = True

    DEFAULT_LONLAT = ""
    LONLAT_FILE = ""
    
    ##########################
    OUT_N = 256 # ボクセル数（OUT_N × OUT_N）
    # OUT_N = 128

    R_MIN, R_MAX = 1.5, 4.0
    NR = 0  # 有効pBデータ数を数えた後に自動決定する。
    NTH = 60 # (NTH, NPH) を固定。(60, 120)のとき約3.0度の角度分解能。NTH=30のとき約6.0度の角度分解能。
    NPH = 2 * NTH
    AUTO_GRID_MAX_M_OVER_N = 1.5 # 観測数/voxel数の最大比率。AUTO_GRID_MAX_M_OVER_N=1.5のとき、voxel数は観測数の2/3以下に制限される。


    DS = 0.02 # 視線方向のボクセル分割数（DS=0.01 なら 1/100 ボクセル分割）

    HM = 5 # 方位角方向の低次Fourier成分だけを残して、背景pBプロファイル ybk(r) を作る

    # Extended comparisons use a scalar; scenarios can override wt_nr with 0 or 1.
    WT_NR = 1 # 半径方向の重みづけ（０：つけない、１：つける）
    LAM = 1.0 # 平滑化度合い
    Q_LOW = 0.0 # 方位角方向の緩やかな成分を差し引く。0の場合、無効。
    WIDTH_PIX = 1.0 # 各PAのradial cutを、横方向に何pixel広げて平均するか

    MAXITER = 10000 # 最終解の最大反復回数
    TOL = 1e-4 # 最終解の収束判定の閾値
    SOLVER_USE_PRECONDITIONER = True # 同じ正規方程式にJacobi前処理を適用
    SOLVER_PRECONDITIONER_FLOOR = 1e-12 # 前処理対角成分の相対下限
    ###########################

    DESPIKE_NSIG = 4.0 # 局所medianから何σ以上明るいpixelをspikeと判断するか
    DESPIKE_MED = 5 # despike判定に使う局所median filterの窓サイズ(DESPIKE_MED × DESPIKE_MED pixel)

    # Physically preferred instrument/bandpass-dependent limb-darkening model.
    LIMB_U = base.DEFAULT_LIMB_U
    LIMB_U_MODE = "instrument_bandpass"
    LIMB_U_USE_ALLEN = False
    LIMB_U_BANDPASS_NM_BY_INSTRUMENT = dict(base.DEFAULT_LIMB_BANDPASS_NM)
    LIMB_U_OVERRIDE_BY_INSTRUMENT = {}
    LIMB_U_BANDPASS_SAMPLES = 401
    LIMB_U_WEIGHT_HDU_NAMES = base.DEFAULT_LIMB_WEIGHT_HDU_NAMES

    FILT = 1
    PB_FLOOR = ""

    DPA_DEG = 0.5 # 背景pBプロファイルを作るためのPosition Angle方向のサンプリング間隔
    R_USE_MIN, R_USE_MAX = 1.5, 4.0
    R_USE_MIN_BY_GROUP = {"cor1a": 1.5}
    R_USE_MAX_BY_GROUP = {}
    PB_SCALE_BY_GROUP = {}

    APPLY_BRIGHTNESS_SCALE = False
    USE_DENSITY_PRIOR = True
    FITTED_DENSITY_PRIOR_NPZ = (
        "/mnt/d/wsl/home/kinno-7010/Research_data/SDO_Mk4_SOHO/pB/"
        "pB_spherical_median_prior_errorbar_20220613_0258_"
        "fit1.5-4.0_prior_model.npz"
    )
    
    # DENSITY_PRIOR_MODEL = "saito_equatorial"
    DENSITY_PRIOR_MODEL = "fitted_pchip_npz"
    
    
    if DENSITY_PRIOR_MODEL == "saito_equatorial":
        DENSITY_PRIOR_SCALE = 2.8
        DENSITY_PRIOR_FILE = ""  # Used only when DENSITY_PRIOR_MODEL="fitted_pchip_npz".
    elif DENSITY_PRIOR_MODEL == "fitted_pchip_npz":
        DENSITY_PRIOR_SCALE = 1.0
        DENSITY_PRIOR_FILE = FITTED_DENSITY_PRIOR_NPZ

    POSITIVITY_METHOD = "clip"

    LAMBDA_SCAN_VALUES = [1.0]
    LAMBDA_SELECT_MODE = "fixed"

    THOMSON_NORMALIZE_MSB = True
    THOMSON_KERNEL_SCALE = 1.0
    RUN_PB_UNIT_DIAGNOSTICS = False
    PB_DIAGNOSTIC_PATHS = []

    CALIBRATION_REFERENCE_GROUP = "earth_merged"
    AUTO_CROSS_CALIBRATE_GROUPS = True
    FIXED_GROUP_FORWARD_GAINS = {}
    CROSS_CALIBRATION_INITIAL_GAIN_BY_GROUP = {"cor1a": 0.843} # COR1A gainの初期値。設定した値から反復を開始。
    CROSS_CALIBRATION_MAX_ITERATIONS = 5 # 密度解と機器gainを交互に更新する最大回数。1回の反復は概ね、1. 現在のgainで密度を復元, 2.復元密度からpBを予測, 3. COR1AとEarth-viewの相対gainを再推定
    CROSS_CALIBRATION_TOLERANCE = 0.01 # gainの相対変化が収束する割合
    CROSS_CALIBRATION_SOLVER_MAXITER = 5000 # gain更新用の中間solve。最終solveとは分離
    CROSS_CALIBRATION_SOLVER_TOL = 3e-3 # gain更新用の緩和した収束閾値
    CROSS_CALIBRATION_DAMPING = 0.7 # 新しく推定したgainをどの程度反映するか。設定値：新しい推定値の割合、1-設定値：以前の値の割合 を対数空間で混合
    CROSS_CALIBRATION_GAIN_MIN = 0.25 # 推定gainが非現実的な値へ発散しないための範囲。数値的不安定を防止する安全境界。
    CROSS_CALIBRATION_GAIN_MAX = 4.0 # gain推定時の外れ値除去閾値
    CROSS_CALIBRATION_R_MIN = 1.5 # gain推定に使う画像面上の半径範囲。内側のocculter境界を避ける・K-Cor/LASCO接続部の影響を抑える・外側の低S/N領域を除く・COR1AとEarth-viewの共通して信頼できる範囲を使う
    CROSS_CALIBRATION_R_MAX = 4
    CROSS_CALIBRATION_MIN_COUNT = 1000 # 一つの画像からgainを推定するために必要な最低有効pixel数
    CROSS_CALIBRATION_CLIP_SIGMA = 4.0 # 予測と観測のweighted residualが中央値から約{sigma}σ以上外れたpixelを除外して、gainを再推定
    CROSS_CALIBRATION_RECALIBRATE_AFTER_LAMBDA_SELECTION = True # lambda scan後に採用するλが決まったら、そのlambdaを使ってgainをもう一度推定する設定. True：最終lambdaに対して自己整合的なgainを再計算. False：scan開始時のgainをそのまま使用

    USE_TEMPORAL_DESPIKE = False
    NE3DTOMO_GLOBAL_YBK = True
    SHOW_RAY_PROGRESS = True
    USE_RAY_CACHE = True

    SHOW_GUI = False
    HARMONIC = 2
    FREQ_MHZ_LIST = [33]
    ISO_COLORS = None

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
    STEP1_LAMBDA_SCAN_DESCENDING = True # 大きいlambdaから小さいlambdaへcontinuation
    STEP1_LAMBDA_SCAN_WARM_START = True # 直前lambdaの解を次の初期値に使用
    STEP1_FAST_LAMBDA_SUITE = True # lambda=1..19で前処理・A・gainを共有
    STEP1_FORWARD_MATRIX_CACHE_MAX_ENTRIES = 2 # 同一ray構成のCSR行列をメモリ再利用
    STEP1_SKIP_COMPLETED_SCENARIOS = True # 同一設定かつ収束済みの結果を再利用
    STEP1_SKIP_COMPLETED_UNCONVERGED = False # 未収束結果は原則として再計算

    # Use the same candidates as the current main_multi_tomo.py settings.
    STEP1_LAMBDA_VALUES = list(LAMBDA_SCAN_VALUES)

    # "default" corresponds to the current fixed-lambda setting.
    STEP1_FINAL_LAM_MODE = "default" if LAMBDA_SELECT_MODE == "fixed" else LAMBDA_SELECT_MODE

    # Save residual maps.  NPY is light and useful; PNG can create many files.
    STEP1_SAVE_RESIDUAL_NPY = False
    STEP1_SAVE_RESIDUAL_PNG = False

    # Residual/calibration/coverage diagnostics.
    STEP1_RADIAL_RESIDUAL_BINS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    STEP1_SAVE_CALIBRATION_DIAGNOSTICS = True
    STEP1_CALIBRATION_RADIAL_BINS = [1.7, 2.0, 2.5, 3.0, 3.5]
    STEP1_COMPUTE_COVERAGE = False
    STEP1_SAVE_COVERAGE_NPZ = False
    STEP1_COVERAGE_LOW_RELATIVE_THRESHOLD = 1e-3
    STEP1_SAVE_TARGET_MASK_NPZ = False

    # Holdout validation is solver-expensive; ray cache avoids rebuilding rays but
    # does not avoid the inverse solve. Enable the dedicated validation scenario below
    # only when needed.
    STEP1_RUN_LEAVE_ONE_IMAGE_OUT = False
    STEP1_LOO_FINAL_LAMBDA_ONLY = True
    STEP1_LOO_MAX_HOLDOUTS = None  # e.g., 6 representative images
    STEP1_RUN_BLOCK_HOLDOUT = False
    STEP1_BLOCK_HOLDOUT_DAYS = 1.0
    STEP1_BLOCK_HOLDOUT_MAX_BLOCKS = 6
    RUN_EXPENSIVE_VALIDATION_SCENARIOS = False

    TIME_WINDOW_COMPARISON_DAYS = [3.0, 5.0, 7.0]
    STEP1_OVERLAP_REFERENCE_SCENARIO = "time_window_5d_earth_cor1a"
    STEP1_WRITE_OVERLAP_METRICS = True

    # Comparison suite.  When True, all 3 time windows x 3 observation sets are
    # run sequentially.  Each scenario receives its own output directory, NPZ,
    # PNG, selected-observation CSV, residual diagnostics, and final summary.
    STEP1_RUN_COMPARISON_SUITE = True
    STEP1_COMPARISON_OUTPUT_DIR = (
        f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/"
        f"step1_timewindow_viewpoint_9cases_{TARGET_TAG}_{FREQ_TAG}MHz"
    )
    STEP1_COMPARISON_SCENARIOS = (
        build_time_window_observation_set_scenarios(TIME_WINDOW_COMPARISON_DAYS)
        + [
        # Additional optional scenarios can still be appended below.

        # 2) Keep the requested lambda=1..19 comparison unchanged.
        # *[
        #     {
        #         "name": f"lambda_{lambda_para}",
        #         "comparison_axis": "lambda",
        #         "lam": float(lambda_para),
        #         "step1_lambda_values": [float(lambda_para)],
        #         "step1_run_lambda_scan": False,
        #         # Coverage is geometry-expensive; save it only for the lambda=5 reference.
        #         "step1_compute_coverage": bool(lambda_para == 5),
        #         "step1_save_coverage_npz": bool(lambda_para == 5),
        #         "step1_save_target_mask_npz": bool(lambda_para == 5),
        #     }
        #     for lambda_para in np.arange(1, 20)
        # ],

        # # 3) Observation-derived fitted prior versus Saito equatorial x2.8.
        # {
        #     "name": "density_prior_fitted_pchip",
        #     "comparison_axis": "density_prior",
        #     "use_density_prior": True,
        #     "density_prior_model": "fitted_pchip_npz",
        #     "density_prior_scale": 1.0,
        #     "density_prior_file": FITTED_DENSITY_PRIOR_NPZ,
        #     "step1_run_lambda_scan": False,
        #     "note": "Observation-derived spherical radial prior from the fitted NPZ line.",
        # },
        # {
        #     "name": "density_prior_saito_2p8",
        #     "comparison_axis": "density_prior",
        #     "use_density_prior": True,
        #     "density_prior_model": "saito_equatorial",
        #     "density_prior_scale": 2.8,
        #     "density_prior_file": "",
        #     "step1_run_lambda_scan": False,
        #     "note": "Saito equatorial radial prior multiplied by 2.8.",
        # },

        # # 3) Density-prior on/off and scale comparison.
        # {"name": "density_prior_on", "comparison_axis": "density_prior", "density_prior_model": "saito_equatorial", "density_prior_scale": 2.8, "step1_run_lambda_scan": False},
        # {"name": "density_prior_off", "comparison_axis": "density_prior", "use_density_prior": False, "density_prior_model": "none", "density_prior_scale": 1.0, "step1_run_lambda_scan": False},
        # {"name": "density_prior_scale_1p0", "comparison_axis": "density_prior_scale", "density_prior_model": "saito_equatorial", "density_prior_scale": 1.0, "step1_run_lambda_scan": False},
        # {"name": "density_prior_scale_2p0", "comparison_axis": "density_prior_scale", "density_prior_model": "saito_equatorial", "density_prior_scale": 2.0, "step1_run_lambda_scan": False},
        # {"name": "density_prior_scale_2p8", "comparison_axis": "density_prior_scale", "density_prior_model": "saito_equatorial", "density_prior_scale": 2.8, "step1_run_lambda_scan": False},
        # {"name": "density_prior_scale_3p0", "comparison_axis": "density_prior_scale", "density_prior_model": "saito_equatorial", "density_prior_scale": 3.0, "step1_run_lambda_scan": False},
        # {"name": "density_prior_scale_4p0", "comparison_axis": "density_prior_scale", "density_prior_model": "saito_equatorial", "density_prior_scale": 4.0, "step1_run_lambda_scan": False},

        # # 4) Radial weighting and global-background comparison.
        # {"name": "weighting_on", "comparison_axis": "weighting", "wt_nr": 1, "step1_run_lambda_scan": False},
        # {"name": "weighting_off", "comparison_axis": "weighting", "wt_nr": 0, "step1_run_lambda_scan": False},
        # {"name": "global_ybk_on", "comparison_axis": "global_ybk", "ne3dtomo_global_ybk": True, "step1_run_lambda_scan": False},
        # {"name": "global_ybk_off", "comparison_axis": "global_ybk", "ne3dtomo_global_ybk": False, "step1_run_lambda_scan": False},

        # 5) Resolution-convergence comparison. Grid changes intentionally produce cache misses.
        # {"name": "grid_64x64x128", "comparison_axis": "grid_resolution", "nr": 64, "nth": 64, "nph": 128, "step1_run_lambda_scan": False},
        # {"name": "grid_80x80x160", "comparison_axis": "grid_resolution", "nr": 80, "nth": 80, "nph": 160, "step1_run_lambda_scan": False},
        # {"name": "grid_96x96x192", "comparison_axis": "grid_resolution", "nr": 96, "nth": 96, "nph": 192, "step1_run_lambda_scan": False},

        # # 6) Automatic/fixed/no cross-calibration while retaining both viewpoints.
        # {"name": "cross_calibration_auto", "comparison_axis": "cross_calibration", "auto_cross_calibrate_groups": True, "fixed_group_forward_gains": {}, "step1_run_lambda_scan": False},
        # {"name": "cross_calibration_fixed_0p76", "comparison_axis": "cross_calibration", "auto_cross_calibrate_groups": False, "fixed_group_forward_gains": {"cor1a": 0.76}, "step1_run_lambda_scan": False},
        # {"name": "cross_calibration_none", "comparison_axis": "cross_calibration", "auto_cross_calibrate_groups": False, "fixed_group_forward_gains": {}, "step1_run_lambda_scan": False},

        # # 7) Calibration convergence from different initial values.
        # {"name": "cal_initial_0p60", "comparison_axis": "calibration_initial_gain", "auto_cross_calibrate_groups": True, "cross_calibration_initial_gain_by_group": {"cor1a": 0.60}, "step1_run_lambda_scan": False},
        # {"name": "cal_initial_0p76", "comparison_axis": "calibration_initial_gain", "auto_cross_calibrate_groups": True, "cross_calibration_initial_gain_by_group": {"cor1a": 0.76}, "step1_run_lambda_scan": False},
        # {"name": "cal_initial_1p00", "comparison_axis": "calibration_initial_gain", "auto_cross_calibrate_groups": True, "cross_calibration_initial_gain_by_group": {"cor1a": 1.00}, "step1_run_lambda_scan": False},

        # # 8) Calibration radial-range sensitivity.
        # {"name": "cal_range_1p6_3p7", "comparison_axis": "calibration_radial_range", "cross_calibration_r_min": 1.6, "cross_calibration_r_max": 3.7, "step1_run_lambda_scan": False},
        # {"name": "cal_range_1p7_3p5", "comparison_axis": "calibration_radial_range", "cross_calibration_r_min": 1.7, "cross_calibration_r_max": 3.5, "step1_run_lambda_scan": False},
        # {"name": "cal_range_1p8_3p2", "comparison_axis": "calibration_radial_range", "cross_calibration_r_min": 1.8, "cross_calibration_r_max": 3.2, "step1_run_lambda_scan": False},

        # # 9) Background/LOS-integration sensitivity.
        # {"name": "hm_3", "comparison_axis": "hm", "hm": 3, "step1_run_lambda_scan": False},
        # {"name": "hm_5", "comparison_axis": "hm", "hm": 5, "step1_run_lambda_scan": False},
        # {"name": "width_pix_0", "comparison_axis": "width_pix", "width_pix": 0.0, "step1_run_lambda_scan": False},
        # {"name": "width_pix_1", "comparison_axis": "width_pix", "width_pix": 1.0, "step1_run_lambda_scan": False},
        # {"name": "ds_0p01", "comparison_axis": "ds", "ds": 0.01, "step1_run_lambda_scan": False},
        # {"name": "ds_0p02", "comparison_axis": "ds", "ds": 0.02, "step1_run_lambda_scan": False},

        # # 10) Positivity diagnostic. The 'none' case reveals negative-solution dependence.
        # {"name": "positivity_clip", "comparison_axis": "positivity", "positivity_method": "clip", "step1_run_lambda_scan": False},
        # {"name": "positivity_none", "comparison_axis": "positivity", "positivity_method": "none", "step1_run_lambda_scan": False},
        ]
    ) # + ([
    #     {
    #         "name": "validation_leave_one_out",
    #         "comparison_axis": "validation",
    #         "step1_run_lambda_scan": False,
    #         "step1_run_leave_one_image_out": True,
    #         "step1_loo_max_holdouts": 6,
    #     },
    #     {
    #         "name": "validation_time_block",
    #         "comparison_axis": "validation",
    #         "step1_run_lambda_scan": False,
    #         "step1_run_block_holdout": True,
    #         "step1_block_holdout_days": 1.0,
    #         "step1_block_holdout_max_blocks": 6,
    #     },
    # ] if RUN_EXPENSIVE_VALIDATION_SCENARIOS else [])

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
        exclude_earth_times=EXCLUDE_EARTH_TIMES,
        keep_cor1a_for_excluded_earth_times=(
            KEEP_COR1A_FOR_EXCLUDED_EARTH_TIMES
        ),
        observation_groups=[],

        default_lonlat=DEFAULT_LONLAT,
        lonlat_file=LONLAT_FILE,

        r_min=R_MIN,
        r_max=R_MAX,
        nr=NR,
        nth=NTH,
        nph=NPH,
        auto_grid_max_m_over_n=AUTO_GRID_MAX_M_OVER_N,
        ds=DS,
        limb_u=LIMB_U,
        limb_u_mode=LIMB_U_MODE,
        limb_u_use_allen=LIMB_U_USE_ALLEN,
        limb_u_bandpass_nm_by_instrument=LIMB_U_BANDPASS_NM_BY_INSTRUMENT,
        limb_u_override_by_instrument=LIMB_U_OVERRIDE_BY_INSTRUMENT,
        limb_u_bandpass_samples=LIMB_U_BANDPASS_SAMPLES,
        limb_u_weight_hdu_names=LIMB_U_WEIGHT_HDU_NAMES,

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
        solver_use_preconditioner=SOLVER_USE_PRECONDITIONER,
        solver_preconditioner_floor=SOLVER_PRECONDITIONER_FLOOR,
        positivity_method=POSITIVITY_METHOD,
        apply_brightness_scale=APPLY_BRIGHTNESS_SCALE,
        use_density_prior=USE_DENSITY_PRIOR,
        density_prior_model=DENSITY_PRIOR_MODEL,
        density_prior_scale=DENSITY_PRIOR_SCALE,
        density_prior_file=DENSITY_PRIOR_FILE,
        lambda_scan_values=LAMBDA_SCAN_VALUES,
        lambda_select_mode=LAMBDA_SELECT_MODE,
        thomson_normalize_msb=THOMSON_NORMALIZE_MSB,
        thomson_kernel_scale=THOMSON_KERNEL_SCALE,
        run_pb_unit_diagnostics=RUN_PB_UNIT_DIAGNOSTICS,
        pb_diagnostic_paths=PB_DIAGNOSTIC_PATHS,
        calibration_reference_group=CALIBRATION_REFERENCE_GROUP,
        auto_cross_calibrate_groups=AUTO_CROSS_CALIBRATE_GROUPS,
        fixed_group_forward_gains=FIXED_GROUP_FORWARD_GAINS,
        cross_calibration_initial_gain_by_group=CROSS_CALIBRATION_INITIAL_GAIN_BY_GROUP,
        cross_calibration_max_iterations=CROSS_CALIBRATION_MAX_ITERATIONS,
        cross_calibration_tolerance=CROSS_CALIBRATION_TOLERANCE,
        cross_calibration_solver_maxiter=CROSS_CALIBRATION_SOLVER_MAXITER,
        cross_calibration_solver_tol=CROSS_CALIBRATION_SOLVER_TOL,
        cross_calibration_damping=CROSS_CALIBRATION_DAMPING,
        cross_calibration_gain_min=CROSS_CALIBRATION_GAIN_MIN,
        cross_calibration_gain_max=CROSS_CALIBRATION_GAIN_MAX,
        cross_calibration_r_min=CROSS_CALIBRATION_R_MIN,
        cross_calibration_r_max=CROSS_CALIBRATION_R_MAX,
        cross_calibration_min_count=CROSS_CALIBRATION_MIN_COUNT,
        cross_calibration_clip_sigma=CROSS_CALIBRATION_CLIP_SIGMA,
        cross_calibration_recalibrate_after_lambda_selection=CROSS_CALIBRATION_RECALIBRATE_AFTER_LAMBDA_SELECTION,

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
        step1_lambda_scan_descending=STEP1_LAMBDA_SCAN_DESCENDING,
        step1_lambda_scan_warm_start=STEP1_LAMBDA_SCAN_WARM_START,
        step1_fast_lambda_suite=STEP1_FAST_LAMBDA_SUITE,
        step1_forward_matrix_cache_max_entries=STEP1_FORWARD_MATRIX_CACHE_MAX_ENTRIES,
        step1_skip_completed_scenarios=STEP1_SKIP_COMPLETED_SCENARIOS,
        step1_skip_completed_unconverged=STEP1_SKIP_COMPLETED_UNCONVERGED,
        step1_save_residual_npy=STEP1_SAVE_RESIDUAL_NPY,
        step1_save_residual_png=STEP1_SAVE_RESIDUAL_PNG,
        step1_radial_residual_bins=STEP1_RADIAL_RESIDUAL_BINS,
        step1_save_calibration_diagnostics=STEP1_SAVE_CALIBRATION_DIAGNOSTICS,
        step1_calibration_radial_bins=STEP1_CALIBRATION_RADIAL_BINS,
        step1_compute_coverage=STEP1_COMPUTE_COVERAGE,
        step1_save_coverage_npz=STEP1_SAVE_COVERAGE_NPZ,
        step1_coverage_low_relative_threshold=STEP1_COVERAGE_LOW_RELATIVE_THRESHOLD,
        step1_save_target_mask_npz=STEP1_SAVE_TARGET_MASK_NPZ,
        step1_run_leave_one_image_out=STEP1_RUN_LEAVE_ONE_IMAGE_OUT,
        step1_loo_final_lambda_only=STEP1_LOO_FINAL_LAMBDA_ONLY,
        step1_loo_max_holdouts=STEP1_LOO_MAX_HOLDOUTS,
        step1_run_block_holdout=STEP1_RUN_BLOCK_HOLDOUT,
        step1_block_holdout_days=STEP1_BLOCK_HOLDOUT_DAYS,
        step1_block_holdout_max_blocks=STEP1_BLOCK_HOLDOUT_MAX_BLOCKS,
        step1_run_comparison_suite=STEP1_RUN_COMPARISON_SUITE,
        step1_comparison_output_dir=STEP1_COMPARISON_OUTPUT_DIR,
        step1_comparison_scenarios=STEP1_COMPARISON_SCENARIOS,
        step1_overlap_reference_scenario=STEP1_OVERLAP_REFERENCE_SCENARIO,
        step1_write_overlap_metrics=STEP1_WRITE_OVERLAP_METRICS,
    )

    if STEP1_RUN_COMPARISON_SUITE:
        run_comparison_suite(args)
    else:
        main(args)
