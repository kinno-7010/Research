#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_tomo_spheroid_alfven_pfss.py

Recompute the 3-D electron density with the same observational selection and
inversion settings as main_multi_tomo.py, evaluate the PFSS magnetic-field
magnitude on the CME Spheroid surface, and color that surface by the Alfvén
speed

    v_A = |B| / sqrt(mu_0 * rho),
    rho = MU_E * m_p * n_e.

The original files are imported as companion modules and are not modified.

Coordinate convention
---------------------
- Tomography density grid: Heliographic Carrington Cartesian coordinates.
- Spheroid: generated in Heliographic Stonyhurst and transformed to Carrington
  by main_npz_tomo_sphe_pfss.py.
- PFSS vectors: sampled with pfsspy.Output.get_bvec() after transforming the
  Spheroid points into the PFSS output coordinate frame.

PFSS domain
-----------
PFSS is defined from 1 R_sun to R_SS.  The default Spheroid apex is above the
usual R_SS=2.5 R_sun.  PFSS_OUTSIDE_RSS_MODE controls this region:

- "radial_extrapolation" (default): sample B_r at R_SS and continue it as
  B_r(r) = B_r(R_SS) * (R_SS/r)^2.
- "mask": do not evaluate v_A outside R_SS; those cells remain gray.
- "raise": stop when any valid tomography point lies outside R_SS.

Outputs
-------
- PNG rendering.
- VTP Spheroid mesh with point arrays: ne_cm3, B_G, alfven_speed_km_s,
  pfss_extrapolated, valid_alfven.
- NPZ with the same sampled quantities and Cartesian coordinates.

Required companion files
------------------------
- main_multi_tomo.py
- main_npz_tomo_sphe_pfss.py

Edit TOMO_MODULE_PATH and OVERLAY_MODULE_PATH below if necessary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional, Sequence, Tuple
import importlib.util
import os
import sys
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import astropy.units as u
from astropy.constants import m_p, mu0
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy.time import Time
from sunpy.coordinates import frames
import sunpy.map

try:
    import pyvista as pv
except Exception as exc:
    raise SystemExit("PyVista is required: pip install pyvista") from exc


# =============================================================================
# Companion-module paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def _first_existing_path(candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return Path(candidates[0])


TOMO_MODULE_PATH = _first_existing_path(
    [
        SCRIPT_DIR / "main_multi_tomo.py",
    ]
)
OVERLAY_MODULE_PATH = _first_existing_path(
    [
        SCRIPT_DIR / "main_npz_tomo_sphe_pfss.py",
    ]
)


# =============================================================================
# User settings
# =============================================================================

# ---- Run control ----
RECOMPUTE_TOMOGRAPHY = False
SHOW_GUI = True
SAVE_PNG = True
SAVE_SPHEROID_VTP = True
SAVE_SAMPLED_NPZ = True
# ---- Input paths copied from main_multi_tomo.py / overlay code ----
DATA_DIR = Path(
    "/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata"
)
COR1A_DATA_DIR = Path(
    "/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/pB/Rawdata"
)
HMI_FITS = Path(
    "/mnt/d/wsl/home/kinno-7010/Research_data/SDO/HMI/Rawdata/"
    "hmi.synoptic_mr_polfil_720s.2258.Mr_polfil.fits"
)
DENSITY_PRIOR_FILE = Path(
    "/mnt/d/wsl/home/kinno-7010/Research_data/SDO_Mk4_SOHO/pB/"
    "pB_spherical_median_prior_errorbar_20220613_0258_"
    "fit1.5-4.0_prior_model.npz"
)

# ---- Times ----
TARGET_TIME = "20220613_030000"       # Tomography/PFSS reference time [UTC]
SPHEROID_TIME_ISO = "2022-06-13T03:26:29"  # CME Spheroid time [UTC]
SEARCH_WINDOW_DAYS = 5.0



TARGET_TAG = "20220613_030000"
WINDOW_TAG = f"pm{int(SEARCH_WINDOW_DAYS)}d"
TOMO_FREQ_MHZ_LIST = [33.0, 43.0]
FREQ_TAG = "-".join(f"{f:g}" for f in TOMO_FREQ_MHZ_LIST)

TOMOGRAPHY_NPZ = (
    DATA_DIR
    / "ne_npz"
    / f"ne3d_solution_{TARGET_TAG}_{WINDOW_TAG}_{FREQ_TAG}MHz.npz"
)
RAY_CACHE_DIR = DATA_DIR / f"ray_cache_{TARGET_TAG}"

OUTPUT_DIR = Path(
    "/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/output/"
    "spheroid_alfven"
)
OUTPUT_PNG = OUTPUT_DIR / f"spheroid_alfven_{TARGET_TAG}_{WINDOW_TAG}.png"
OUTPUT_VTP = OUTPUT_DIR / f"spheroid_alfven_{TARGET_TAG}_{WINDOW_TAG}.vtp"
OUTPUT_NPZ = OUTPUT_DIR / f"spheroid_alfven_{TARGET_TAG}_{WINDOW_TAG}.npz"

# ---- Spheroid parameters copied from main_npz_tomo_sphe_pfss.py ----
SPHEROID_ANCHOR_LON_DEG = -30.0
SPHEROID_ANCHOR_LAT_DEG = +19.0
SPHEROID_APEX_LON_DEG = -55.0
SPHEROID_APEX_LAT_DEG = +5.0
SPHEROID_APEX_RSUN = 3.27
SPHEROID_KAPPA = 0.50
SPHEROID_EPSILON = -0.45
SPHEROID_N_ALPHA = 128
SPHEROID_N_BETA = 240

# ---- PFSS ----
PFSS_RSS = 2.5
PFSS_NRHO = 80
PFSS_HELIO_SHAPE = (180, 360)
PFSS_OUTSIDE_RSS_MODE = "mask"  # radial_extrapolation: Rssより外側をr^{-2}で外挿 | mask: Rssより外側は計算しない | raise: SpheroidがRssを超えた場合に計算を停止
PFSS_SAMPLE_CHUNK_SIZE = 20_000

PFSS_GLOBAL_SEEDS = True
SHOW_PFSS_FIELD_LINES = True
PFSS_SEED_N_X = 50
PFSS_SEED_N_Y = 50
PFSS_FIELD_THRESHOLD_G = 30.0
PFSS_MAX_LINES = 1000
PFSS_LINE_WIDTH = 5
# PFSS_SEED_VIEW_QUADRANT = "upper_left"
PFSS_SEED_VIEW_QUADRANT = None

# ---- Plasma composition ----
# For n_He/n_H = 0.10 and complete ionization:
# rho/(m_p n_e) = (1 + 4*0.10)/(1 + 2*0.10) = 1.1667.
MU_E = 1.1666666666666667

# ---- Density interpolation ----
INTERPOLATE_LOG_DENSITY = True

# ---- Rendering ----
SHOW_SUN = True
SHOW_TOMOGRAPHY_ISOSURFACES = True
ISOSURFACE_OPACITY = 0.12
ISOSURFACE_HARMONIC = 2
ALFVEN_CMAP = "turbo"
ALFVEN_LOG_SCALE = True
ALFVEN_CLIM_PERCENTILES = (10.0, 99.0)
INVALID_SURFACE_COLOR = "lightgray"
INVALID_SURFACE_OPACITY = 0.30
VALID_SURFACE_OPACITY = 0.95


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class TomographyVolume:
    grid: object
    ne_1d_cm3: np.ndarray
    observer: object
    target_time_iso: str
    harmonic: int
    npz_path: Path


@dataclass
class SpheroidSampling:
    surface: pv.PolyData
    density_cm3: np.ndarray
    magnetic_field_g: np.ndarray
    alfven_speed_km_s: np.ndarray
    valid: np.ndarray
    pfss_extrapolated: np.ndarray


# =============================================================================
# Generic helpers
# =============================================================================

def load_python_module(path: Path, module_name: str) -> ModuleType:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Companion module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot construct import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _scalar_from_npz(npz, keys: Sequence[str], default=None):
    for key in keys:
        if key not in npz.files:
            continue
        arr = np.asarray(npz[key])
        if arr.size == 0:
            continue
        value = arr.reshape(-1)[0]
        if isinstance(value, bytes):
            return value.decode()
        return value.item() if hasattr(value, "item") else value
    return default


def _hgc_frame(obstime_iso: str):
    t = Time(obstime_iso)
    try:
        return frames.HeliographicCarrington(obstime=t, observer="earth")
    except TypeError:
        return frames.HeliographicCarrington(obstime=t)


def _xyz_rsun_to_skycoord_hgc(points_xyz_rsun: np.ndarray, obstime_iso: str) -> SkyCoord:
    points = np.asarray(points_xyz_rsun, dtype=np.float64)
    rep = CartesianRepresentation(
        x=points[:, 0] * u.R_sun,
        y=points[:, 1] * u.R_sun,
        z=points[:, 2] * u.R_sun,
    )
    return SkyCoord(rep, frame=_hgc_frame(obstime_iso))


def _quantity_to_gauss(values, *, fallback_unit=None) -> np.ndarray:
    """Convert a pfsspy magnetic-field Quantity to gauss."""
    if hasattr(values, "unit"):
        try:
            return np.asarray(values.to_value(u.G), dtype=np.float64)
        except Exception:
            if fallback_unit is not None:
                try:
                    q = np.asarray(values.value, dtype=np.float64) * fallback_unit
                    return np.asarray(q.to_value(u.G), dtype=np.float64)
                except Exception:
                    pass
            if getattr(values.unit, "is_equivalent", lambda _: False)(u.dimensionless_unscaled):
                print(
                    "[WARN] PFSS output is dimensionless. Interpreting numerical values as gauss "
                    "because the HMI boundary map is expected to be in gauss."
                )
                return np.asarray(values.value, dtype=np.float64)
            raise
    return np.asarray(values, dtype=np.float64)


# =============================================================================
# Tomography execution and loading
# =============================================================================

def build_tomography_args(tomo_module: ModuleType) -> SimpleNamespace:
    """Build the inversion settings currently used in main_multi_tomo.py."""
    exclude_earth_times = [
        "20220606_0258",
        "20220612_0258",
        "20220614_0258",
        "20220616_2104",
        "20220617_0258",
        "20220617_2104",
    ]

    return SimpleNamespace(
        pb_fits=[],
        out_n=256,
        data_dir=str(DATA_DIR),
        cor1a_data_dir=str(COR1A_DATA_DIR),
        target_time=TARGET_TIME,
        search_window_days=SEARCH_WINDOW_DAYS,
        auto_find_pb_fits=True,
        include_kcor_lasco=True,
        include_cor1a=True,
        include_lasco_only=False,
        deduplicate_pb_fits=True,
        exclude_earth_times=exclude_earth_times,
        keep_cor1a_for_excluded_earth_times=True,
        default_lonlat="",
        lonlat_file="",

        r_min=1.5,
        r_max=4.0,
        nr=0,
        nth=60,
        nph=120,
        auto_grid_max_m_over_n=1.5,
        ds=0.01,

        limb_u=tomo_module.DEFAULT_LIMB_U,
        limb_u_mode="instrument_bandpass",
        limb_u_use_allen=False,
        limb_u_bandpass_nm_by_instrument=dict(
            tomo_module.DEFAULT_LIMB_BANDPASS_NM
        ),
        limb_u_override_by_instrument={},
        limb_u_bandpass_samples=401,
        limb_u_weight_hdu_names=tomo_module.DEFAULT_LIMB_WEIGHT_HDU_NAMES,

        filt=1,
        despike_nsig=5.0,
        despike_med=5,
        pb_floor="",
        dpa_deg=0.5,
        r_use_min=1.5,
        r_use_max=4.0,
        r_use_min_by_group={"cor1a": 1.5},
        r_use_max_by_group={},
        pb_scale_by_group={},
        hm=5,
        wt_nr=1,

        lam=1.0,
        lambda_scan_values=[1.0],
        lambda_select_mode="fixed",
        q_low=0.0,
        width_pix=1.0,
        maxiter=15000,
        tol=1.0e-5,
        solver_use_preconditioner=True,
        solver_preconditioner_floor=1.0e-12,
        positivity_method="clip",

        apply_brightness_scale=False,
        use_density_prior=True,
        density_prior_model="fitted_pchip_npz",
        density_prior_scale=1.0,
        density_prior_file=str(DENSITY_PRIOR_FILE),

        thomson_normalize_msb=True,
        thomson_kernel_scale=1.0,
        run_pb_unit_diagnostics=False,
        pb_diagnostic_paths=[],

        calibration_reference_group="earth_merged",
        auto_cross_calibrate_groups=True,
        fixed_group_forward_gains={},
        cross_calibration_initial_gain_by_group={"cor1a": 0.815},
        cross_calibration_max_iterations=5,
        cross_calibration_tolerance=1.0e-2,
        cross_calibration_solver_maxiter=5000,
        cross_calibration_solver_tol=3.0e-3,
        cross_calibration_damping=0.7,
        cross_calibration_gain_min=0.25,
        cross_calibration_gain_max=4.0,
        cross_calibration_r_min=1.5,
        cross_calibration_r_max=4.0,
        cross_calibration_min_count=1000,
        cross_calibration_clip_sigma=4.0,
        cross_calibration_recalibrate_after_lambda_selection=True,

        use_temporal_despike=False,
        ne3dtomo_global_ybk=True,
        show_ray_progress=True,
        use_ray_cache=True,
        ray_cache_dir=str(RAY_CACHE_DIR),

        # These products are not needed for the Alfvén-speed scene.
        save_prepped_dir="",
        save_ne_npz=str(TOMOGRAPHY_NPZ),
        save_summary_csv=False,
        summary_csv_path="",
        step1_run_diagnostics=False,
        step1_output_dir="",

        # The normal visualization is temporarily replaced by a no-op as well.
        show_gui=False,
        freq_mhz_list=TOMO_FREQ_MHZ_LIST,
        harmonic=ISOSURFACE_HARMONIC,
        iso_colors=None,
        save_png=False,
        png_path="",
    )


def run_tomography_to_npz(tomo_module: ModuleType) -> Path:
    """Run main_multi_tomo.main() without its separate isosurface window."""
    args = build_tomography_args(tomo_module)
    TOMOGRAPHY_NPZ.parent.mkdir(parents=True, exist_ok=True)

    original_visualize = tomo_module.visualize_isosurface

    def _skip_original_visualization(*_args, **_kwargs):
        print(
            "[INFO] main_multi_tomo visualization skipped; "
            "the combined Alfvén-speed scene will be rendered instead."
        )

    tomo_module.visualize_isosurface = _skip_original_visualization
    try:
        tomo_module.main(args)
    finally:
        tomo_module.visualize_isosurface = original_visualize

    if not TOMOGRAPHY_NPZ.exists():
        raise FileNotFoundError(
            f"Tomography completed without producing the requested NPZ: {TOMOGRAPHY_NPZ}"
        )
    return TOMOGRAPHY_NPZ


def load_tomography_volume(
    npz_path: Path,
    overlay_module: ModuleType,
) -> TomographyVolume:
    npz_path = Path(npz_path).expanduser()
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    with np.load(npz_path, allow_pickle=False) as npz:
        required = ("ne", "r_edges", "th_edges", "ph_edges")
        missing = [key for key in required if key not in npz.files]
        if missing:
            raise KeyError(f"Tomography NPZ missing arrays {missing}: {npz_path}")

        ne = np.asarray(npz["ne"], dtype=np.float64).ravel(order="C")
        grid = overlay_module.SphericalGrid(
            r_edges=np.asarray(npz["r_edges"], dtype=np.float64),
            th_edges=np.asarray(npz["th_edges"], dtype=np.float64),
            ph_edges=np.asarray(npz["ph_edges"], dtype=np.float64),
        )
        target_time_value = str(
            _scalar_from_npz(
                npz,
                ("target_time", "target_time_iso", "observer_time_iso"),
                TARGET_TIME,
            )
        )
        try:
            target_time_iso = overlay_module.parse_target_datetime(
                target_time_value
            ).isoformat()
        except Exception:
            target_time_iso = Time(target_time_value).isot

        harmonic = int(_scalar_from_npz(npz, ("harmonic",), ISOSURFACE_HARMONIC))
        observer = overlay_module._simple_observer_from_npz_or_time(
            npz,
            target_time_iso,
        )

    expected = int(grid.nr * grid.nth * grid.nph)
    if ne.size != expected:
        raise ValueError(
            f"Density vector size {ne.size} does not match grid size {expected}."
        )

    print(
        f"[INFO] Loaded tomography: grid={grid.nr}x{grid.nth}x{grid.nph}, "
        f"ne={np.nanmin(ne):.3e}..{np.nanmax(ne):.3e} cm^-3"
    )
    return TomographyVolume(
        grid=grid,
        ne_1d_cm3=ne,
        observer=observer,
        target_time_iso=target_time_iso,
        harmonic=harmonic,
        npz_path=npz_path,
    )


# =============================================================================
# Tomography interpolation
# =============================================================================

def build_density_interpolator(volume: TomographyVolume):
    """Return a periodic Carrington (r, theta, phi) density interpolator."""
    grid = volume.grid
    nr, nth, nph = grid.nr, grid.nth, grid.nph
    ne3 = np.asarray(volume.ne_1d_cm3, dtype=np.float64).reshape(
        (nr, nth, nph), order="C"
    )

    r_cent = 0.5 * (grid.r_edges[:-1] + grid.r_edges[1:])
    th_cent = 0.5 * (grid.th_edges[:-1] + grid.th_edges[1:])
    ph_cent = 0.5 * (grid.ph_edges[:-1] + grid.ph_edges[1:])

    values = ne3.copy()
    if INTERPOLATE_LOG_DENSITY:
        values = np.where(
            np.isfinite(values) & (values > 0.0),
            np.log(values),
            np.nan,
        )

    # Extend cell-centred values to the physical radial and polar boundaries by
    # nearest-cell continuation over the outer half cells.
    values = np.concatenate([values[:1], values, values[-1:]], axis=0)
    r_nodes = np.concatenate(
        [[grid.r_edges[0]], r_cent, [grid.r_edges[-1]]]
    )

    values = np.concatenate(
        [values[:, :1, :], values, values[:, -1:, :]], axis=1
    )
    th_nodes = np.concatenate([[0.0], th_cent, [np.pi]])

    # Periodic extension around Carrington longitude phi=0/2pi.
    values = np.concatenate(
        [values[:, :, -1:], values, values[:, :, :1]], axis=2
    )
    ph_nodes = np.concatenate(
        [[ph_cent[-1] - 2.0 * np.pi], ph_cent, [ph_cent[0] + 2.0 * np.pi]]
    )

    interpolator = RegularGridInterpolator(
        (r_nodes, th_nodes, ph_nodes),
        values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    def evaluate(points_xyz_rsun: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz_rsun, dtype=np.float64)
        r = np.linalg.norm(points, axis=1)
        theta = np.full(r.shape, np.nan, dtype=np.float64)
        nonzero = np.isfinite(r) & (r > 0.0)
        theta[nonzero] = np.arccos(
            np.clip(points[nonzero, 2] / r[nonzero], -1.0, 1.0)
        )
        phi = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
        query = np.column_stack([r, theta, phi])
        out = np.asarray(interpolator(query), dtype=np.float64)
        if INTERPOLATE_LOG_DENSITY:
            out = np.exp(out)
        domain = (
            np.isfinite(r)
            & (r >= float(grid.r_edges[0]))
            & (r <= float(grid.r_edges[-1]))
        )
        out[~domain] = np.nan
        out[~np.isfinite(out) | (out <= 0.0)] = np.nan
        return out

    return evaluate


# =============================================================================
# PFSS solution from HMI
# =============================================================================

def compute_pfss_output_from_hmi(
    overlay_module: ModuleType,
    hmi_fits: Path,
    *,
    nrho: int = 50,
    rss: float = 2.5,
    helio_shape: Tuple[int, int] = (180, 360),
    fill_nan: float = 0.0,
    obstime_iso: Optional[str] = None,
):
    """
    Compute PFSS from either a polar-filled HMI synoptic CEA radial-field map
    or the legacy single full-disk HMI magnetogram input.

    This is the PFSS-input procedure used by main_npz_thetaBn.py.  A synoptic
    CEA map is metadata-corrected, resampled in its native projection, retimed
    to the requested observation time, and passed directly to pfsspy.Input.
    """
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    pfsspy, utils, _ = overlay_module._safe_import_pfsspy()
    hmi = sunpy.map.Map(str(hmi_fits))

    nlat, nlon = int(helio_shape[0]), int(helio_shape[1])
    if nlat < 2 or nlon < 2:
        raise ValueError(f"helio_shape must be >= (2,2); got {helio_shape}")

    ctype1 = str(hmi.meta.get("CTYPE1", "")).strip().upper()
    ctype2 = str(hmi.meta.get("CTYPE2", "")).strip().upper()
    is_synoptic_cea = (
        ctype1 in {"CRLN-CEA", "HGLN-CEA"}
        and ctype2 in {"CRLT-CEA", "HGLT-CEA"}
    )

    if is_synoptic_cea:
        if hasattr(utils, "fix_hmi_meta"):
            try:
                utils.fix_hmi_meta(hmi)
            except Exception as exc:
                print(f"[WARN] HMI synoptic metadata correction was skipped: {exc}")

        if hmi.data.shape != (nlat, nlon):
            bmap_cea = hmi.resample(
                u.Quantity([nlon, nlat], u.pix),
                method="linear",
            )
        else:
            bmap_cea = hmi

        cea_data = np.asarray(bmap_cea.data, dtype=float).copy()
        cea_data[~np.isfinite(cea_data)] = float(fill_nan)
        cea_meta = bmap_cea.meta.copy()

        bunit_text = str(cea_meta.get("BUNIT", "")).strip().lower().replace(" ", "")
        if bunit_text in {
            "mx/cm^2",
            "mx/cm2",
            "mxcm-2",
            "mxcm^-2",
            "maxwell/cm^2",
        }:
            cea_meta["BUNIT"] = "G"

        if obstime_iso is not None:
            pfss_time = Time(obstime_iso)
            cea_meta["DATE-OBS"] = pfss_time.isot
            cea_meta["T_OBS"] = pfss_time.isot
            try:
                from sunpy.coordinates import get_body_heliographic_stonyhurst

                earth_hgs = get_body_heliographic_stonyhurst("earth", pfss_time)
                cea_meta["HGLN_OBS"] = float(earth_hgs.lon.to_value(u.deg))
                cea_meta["HGLT_OBS"] = float(earth_hgs.lat.to_value(u.deg))
                cea_meta["DSUN_OBS"] = float(earth_hgs.radius.to_value(u.m))
                earth_hgc = earth_hgs.transform_to(
                    frames.HeliographicCarrington(
                        obstime=pfss_time,
                        observer="earth",
                    )
                )
                cea_meta["CRLN_OBS"] = (
                    float(earth_hgc.lon.to_value(u.deg)) % 360.0
                )
                cea_meta["CRLT_OBS"] = float(earth_hgc.lat.to_value(u.deg))
            except Exception as exc:
                print(f"[WARN] PFSS observer metadata retime was incomplete: {exc}")

        bmap_cea = sunpy.map.Map(cea_data, cea_meta)
        print(
            "[INFO] PFSS input: HMI global radial-field synoptic CEA map; "
            f"resampled to {bmap_cea.data.shape[1]}x{bmap_cea.data.shape[0]}, "
            f"BUNIT={bmap_cea.meta.get('BUNIT', 'unknown')}, "
            f"obstime={getattr(bmap_cea.date, 'isot', bmap_cea.date)}"
        )
    else:
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

        for key in (
            "HGLN_OBS", "HGLT_OBS", "DSUN_OBS", "RSUN_REF", "RSUN_OBS",
            "CRLN_OBS", "CRLT_OBS", "SOLAR_B0", "SOLAR_P0",
        ):
            if key in hmi.meta:
                car_header[key] = hmi.meta[key]

        if "RSUN_REF" not in car_header and "rsun_ref" in hmi.meta:
            car_header["RSUN_REF"] = hmi.meta["rsun_ref"]
        if "DSUN_OBS" not in car_header and "dsun_obs" in hmi.meta:
            car_header["DSUN_OBS"] = hmi.meta["dsun_obs"]

        bmap_car = hmi.reproject_to(car_header)
        car_data = np.array(bmap_car.data, dtype=float)
        car_data[~np.isfinite(car_data)] = float(fill_nan)
        bmap_car = sunpy.map.Map(car_data, bmap_car.meta)

        cea_out = utils.car_to_cea(bmap_car)
        if isinstance(cea_out, tuple) and len(cea_out) == 2:
            cea_data, cea_meta = cea_out
            bmap_cea = sunpy.map.Map(cea_data, cea_meta)
        else:
            bmap_cea = cea_out

        cea_data = np.array(bmap_cea.data, dtype=float)
        cea_data[~np.isfinite(cea_data)] = float(fill_nan)
        bmap_cea = sunpy.map.Map(cea_data, bmap_cea.meta)
        print(
            "[WARN] PFSS input is not a Carrington CEA synoptic radial-field map; "
            "using the legacy full-disk conversion."
        )

    rss_val = float(rss)
    try:
        pfss_input = pfsspy.Input(bmap_cea, nrho=nrho, rss=rss_val)
    except TypeError as exc:
        if "unexpected keyword argument" in str(exc) and "nrho" in str(exc):
            try:
                pfss_input = pfsspy.Input(bmap_cea, nrho, rss=rss_val)
            except TypeError:
                pfss_input = pfsspy.Input(bmap_cea, nrho, rss_val)
        else:
            raise

    pfss_output = pfsspy.pfss(pfss_input)
    return pfss_output, bmap_cea


# =============================================================================
# PFSS field sampling
# =============================================================================

def _pfss_get_bvec_chunked(
    pfss_output,
    coordinates: SkyCoord,
    chunk_size: int,
    out_type: str,
) -> np.ndarray:
    """Sample pfsspy.Output.get_bvec() in chunks and return gauss values."""
    n = coordinates.size
    output = np.full((n, 3), np.nan, dtype=np.float64)
    fallback_unit = getattr(pfss_output, "bunit", None)
    for start in range(0, n, int(chunk_size)):
        stop = min(n, start + int(chunk_size))
        bq = pfss_output.get_bvec(coordinates[start:stop], out_type=out_type)
        output[start:stop] = _quantity_to_gauss(
            bq,
            fallback_unit=fallback_unit,
        )
    return output


def sample_pfss_magnitude_on_points(
    pfss_output,
    points_hgc_rsun: np.ndarray,
    *,
    obstime_iso: str,
    rss: float,
    outside_mode: str,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return |B| [G] and a boolean flag for source-surface radial extrapolation.
    """
    points = np.asarray(points_hgc_rsun, dtype=np.float64)
    radii = np.linalg.norm(points, axis=1)
    bmag = np.full(radii.shape, np.nan, dtype=np.float64)
    extrapolated = np.zeros(radii.shape, dtype=bool)

    finite = np.all(np.isfinite(points), axis=1) & np.isfinite(radii)
    inside = finite & (radii >= 1.0) & (radii <= float(rss))
    outside = finite & (radii > float(rss))

    all_hgc = _xyz_rsun_to_skycoord_hgc(points, obstime_iso)
    all_pfss = all_hgc.transform_to(pfss_output.coordinate_frame)

    if np.any(inside):
        b_cart = _pfss_get_bvec_chunked(
            pfss_output,
            all_pfss[inside],
            chunk_size=chunk_size,
            out_type="cartesian",
        )
        bmag[inside] = np.linalg.norm(b_cart, axis=1)

    mode = str(outside_mode).strip().lower()
    if np.any(outside):
        if mode == "mask":
            print(
                f"[INFO] PFSS: masking {np.count_nonzero(outside)} Spheroid points "
                f"outside Rss={float(rss):g} R_sun."
            )
        elif mode == "raise":
            raise ValueError(
                f"Spheroid contains {np.count_nonzero(outside)} points outside "
                f"PFSS Rss={float(rss):g} R_sun."
            )
        elif mode == "radial_extrapolation":
            # Evaluate just inside Rss to avoid endpoint interpolation issues.
            r_sample = float(rss) * (1.0 - 1.0e-6)
            outside_coords = all_pfss[outside]
            ss_coords = SkyCoord(
                outside_coords.spherical.lon,
                outside_coords.spherical.lat,
                r_sample * u.R_sun,
                frame=pfss_output.coordinate_frame,
            )
            b_sph = _pfss_get_bvec_chunked(
                pfss_output,
                ss_coords,
                chunk_size=chunk_size,
                out_type="spherical",
            )
            br_ss = np.abs(b_sph[:, 0])
            bmag[outside] = br_ss * (float(rss) / radii[outside]) ** 2
            extrapolated[outside] = True
            print(
                f"[INFO] PFSS: radially extrapolated {np.count_nonzero(outside)} "
                f"points beyond Rss={float(rss):g} R_sun with B_r proportional to r^-2."
            )
        else:
            raise ValueError(
                "PFSS_OUTSIDE_RSS_MODE must be 'radial_extrapolation', 'mask', or 'raise'."
            )

    bmag[~np.isfinite(bmag) | (bmag <= 0.0)] = np.nan
    return bmag, extrapolated


# =============================================================================
# Alfvén speed and Spheroid sampling
# =============================================================================

def alfven_speed_km_s(
    magnetic_field_g: np.ndarray,
    electron_density_cm3: np.ndarray,
    mu_e: float = MU_E,
) -> np.ndarray:
    """Compute non-relativistic Alfvén speed from |B| and electron density."""
    b_g = np.asarray(magnetic_field_g, dtype=np.float64)
    ne_cm3 = np.asarray(electron_density_cm3, dtype=np.float64)
    out = np.full(np.broadcast_shapes(b_g.shape, ne_cm3.shape), np.nan)

    b_t = b_g * 1.0e-4
    ne_m3 = ne_cm3 * 1.0e6
    rho_kg_m3 = float(mu_e) * m_p.value * ne_m3
    good = (
        np.isfinite(b_t)
        & np.isfinite(rho_kg_m3)
        & (b_t > 0.0)
        & (rho_kg_m3 > 0.0)
    )
    out[good] = (
        b_t[good] / np.sqrt(mu0.value * rho_kg_m3[good]) / 1000.0
    )
    return out


def build_and_sample_spheroid(
    overlay_module: ModuleType,
    volume: TomographyVolume,
    pfss_output,
):
    params = overlay_module.SpheroidDome3DParams(
        kappa=float(SPHEROID_KAPPA),
        epsilon=float(SPHEROID_EPSILON),
        anchor_lon_deg=float(SPHEROID_ANCHOR_LON_DEG),
        anchor_lat_deg=float(SPHEROID_ANCHOR_LAT_DEG),
        apex_lon_deg=float(SPHEROID_APEX_LON_DEG),
        apex_lat_deg=float(SPHEROID_APEX_LAT_DEG),
        apex_r_rsun=float(SPHEROID_APEX_RSUN),
        n_meridians=72,
        n_parallels=72,
        n_line_pts=360,
        only_above_surface=True,
        only_visible=True,
    )
    surface = overlay_module.build_spheroid_surface_hgc(
        params,
        obstime_iso=SPHEROID_TIME_ISO,
        observer="earth",
        n_alpha=SPHEROID_N_ALPHA,
        n_beta=SPHEROID_N_BETA,
    )
    if surface is None or surface.n_points == 0:
        raise RuntimeError("Spheroid surface construction returned no points.")

    points = np.asarray(surface.points, dtype=np.float64)
    density_interp = build_density_interpolator(volume)
    density = density_interp(points)

    bmag, extrapolated = sample_pfss_magnitude_on_points(
        pfss_output,
        points,
        obstime_iso=SPHEROID_TIME_ISO,
        rss=PFSS_RSS,
        outside_mode=PFSS_OUTSIDE_RSS_MODE,
        chunk_size=PFSS_SAMPLE_CHUNK_SIZE,
    )
    va = alfven_speed_km_s(bmag, density, mu_e=MU_E)
    valid = (
        np.isfinite(density)
        & (density > 0.0)
        & np.isfinite(bmag)
        & (bmag > 0.0)
        & np.isfinite(va)
        & (va > 0.0)
    )

    surface["ne_cm3"] = density
    surface["B_G"] = bmag
    surface["alfven_speed_km_s"] = va
    surface["pfss_extrapolated"] = extrapolated.astype(np.uint8)
    surface["valid_alfven"] = valid.astype(np.uint8)
    surface["r_rsun"] = np.linalg.norm(points, axis=1)

    print(
        f"[INFO] Spheroid sampling: points={surface.n_points}, "
        f"valid={np.count_nonzero(valid)}, invalid={np.count_nonzero(~valid)}"
    )
    if np.any(valid):
        print(
            f"[INFO] ne(valid)={np.nanmin(density[valid]):.3e}.."
            f"{np.nanmax(density[valid]):.3e} cm^-3"
        )
        print(
            f"[INFO] |B|(valid)={np.nanmin(bmag[valid]):.3e}.."
            f"{np.nanmax(bmag[valid]):.3e} G"
        )
        print(
            f"[INFO] v_A(valid)={np.nanmin(va[valid]):.3f}.."
            f"{np.nanmax(va[valid]):.3f} km/s"
        )

    return params, SpheroidSampling(
        surface=surface,
        density_cm3=density,
        magnetic_field_g=bmag,
        alfven_speed_km_s=va,
        valid=valid,
        pfss_extrapolated=extrapolated,
    )


# =============================================================================
# PFSS line rendering without recomputing the PFSS solution
# =============================================================================

def hmi_roi_pixels_like_2d_script(hmi_map):
    """Return the legacy full-disk HMI ROI used by main_npz_thetaBn.py."""
    data = hmi_map.data
    ny, nx = data.shape
    center_x, center_y = nx // 2, ny // 2

    x_min_pix, x_max_pix = center_x - 512, center_x + 100
    y_min_pix, y_max_pix = center_y - 100, center_y + 512

    x_min_pix = int(np.clip(x_min_pix, 0, nx - 1))
    x_max_pix = int(np.clip(x_max_pix, 1, nx))
    y_min_pix = int(np.clip(y_min_pix, 0, ny - 1))
    y_max_pix = int(np.clip(y_max_pix, 1, ny))

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
    observer_lonlat_deg: Optional[Tuple[float, float]] = None,
    visible_lon_half_width_deg: float = 90.0,
    observer_screen_quadrant: Optional[str] = None,
):
    """Build the strong-field PFSS seeds used by main_npz_thetaBn.py."""
    xmin, xmax = map(int, x_lims_pix)
    ymin, ymax = map(int, y_lims_pix)

    if use_strong_field:
        roi = np.array(hmi_map.data[ymin:ymax, xmin:xmax], dtype=float)
        abs_roi = np.abs(roi)
        mask = np.isfinite(abs_roi) & (abs_roi > float(field_threshold))
        yy, xx = np.where(mask)

        if xx.size == 0:
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
        x0 = xmin + int(margin_pix)
        x1 = xmax - int(margin_pix)
        y0 = ymin + int(margin_pix)
        y1 = ymax - int(margin_pix)

        if x1 <= x0:
            x0, x1 = xmin, xmax
        if y1 <= y0:
            y0, y1 = ymin, ymax

        x_1d = np.linspace(x0, x1, int(n_seeds_x))
        y_1d = np.linspace(y0, y1, int(n_seeds_y))
        x_grid, y_grid = np.meshgrid(x_1d, y_1d, indexing="xy")
        x_pixels = x_grid.ravel()
        y_pixels = y_grid.ravel()

    seeds_world = hmi_map.pixel_to_world(
        x_pixels * u.pixel,
        y_pixels * u.pixel,
    )
    if not isinstance(seeds_world, SkyCoord):
        seeds_world = SkyCoord(seeds_world)

    try:
        seeds_pfss = seeds_world.transform_to(pfss_output.coordinate_frame)
    except Exception:
        seeds_pfss = SkyCoord(seeds_world).transform_to(
            pfss_output.coordinate_frame
        )

    lon = getattr(seeds_pfss, "lon", None)
    lat = getattr(seeds_pfss, "lat", None)
    if lon is None or lat is None:
        raise RuntimeError(
            "Seeds could not be represented with lon/lat in the PFSS coordinate frame."
        )

    lon = lon.to(u.deg)
    lat = lat.to(u.deg)
    good = np.isfinite(lon.value) & np.isfinite(lat.value)

    if observer_lonlat_deg is not None:
        obs_lon_deg = float(observer_lonlat_deg[0]) % 360.0
        half_width = float(abs(visible_lon_half_width_deg))
        lon_deg = lon.to_value(u.deg) % 360.0
        dlon_deg = ((lon_deg - obs_lon_deg + 180.0) % 360.0) - 180.0
        good &= np.abs(dlon_deg) <= half_width

        if observer_screen_quadrant is not None:
            quadrant = str(observer_screen_quadrant).strip().lower()
            valid_quadrants = {
                "upper_left",
                "upper_right",
                "lower_left",
                "lower_right",
            }
            if quadrant not in valid_quadrants:
                raise ValueError(
                    "observer_screen_quadrant must be one of "
                    f"{sorted(valid_quadrants)}; got {observer_screen_quadrant!r}."
                )

            obs_lat_deg = float(observer_lonlat_deg[1])
            obs_lon_rad = np.deg2rad(obs_lon_deg)
            obs_lat_rad = np.deg2rad(obs_lat_deg)
            observer_hat = np.array(
                [
                    np.cos(obs_lat_rad) * np.cos(obs_lon_rad),
                    np.cos(obs_lat_rad) * np.sin(obs_lon_rad),
                    np.sin(obs_lat_rad),
                ],
                dtype=float,
            )
            view_direction = -observer_hat
            screen_right = np.cross(
                view_direction,
                np.array([0.0, 0.0, 1.0]),
            )
            right_norm = np.linalg.norm(screen_right)
            if not np.isfinite(right_norm) or right_norm == 0.0:
                raise RuntimeError(
                    "Could not construct the observer-plane horizontal axis."
                )
            screen_right /= right_norm
            screen_up = np.cross(screen_right, view_direction)
            screen_up /= np.linalg.norm(screen_up)

            lat_rad = lat.to_value(u.rad)
            lon_rad = lon.to_value(u.rad)
            surface_xyz = np.column_stack(
                [
                    np.cos(lat_rad) * np.cos(lon_rad),
                    np.cos(lat_rad) * np.sin(lon_rad),
                    np.sin(lat_rad),
                ]
            )
            screen_x = surface_xyz @ screen_right
            screen_y = surface_xyz @ screen_up

            if quadrant.endswith("left"):
                good &= screen_x <= 0.0
            else:
                good &= screen_x >= 0.0
            if quadrant.startswith("upper"):
                good &= screen_y >= 0.0
            else:
                good &= screen_y <= 0.0

    if np.count_nonzero(good) == 0:
        raise RuntimeError(
            "All ROI seeds became invalid after transforming to PFSS frame "
            "or applying the visibility filter."
        )

    return SkyCoord(
        lon[good],
        lat[good],
        float(r_seed_rsun) * u.R_sun,
        frame=pfss_output.coordinate_frame,
    )


def add_pfss_field_lines_from_output(
    plotter: pv.Plotter,
    overlay_module: ModuleType,
    pfss_output,
    pfss_seed_map,
    obs0,
    *,
    hmi_fits: Path,
    obstime_iso: str,
) -> dict:
    """Render PFSS lines with the seed selection used by main_npz_thetaBn.py."""
    _, _, tracing = overlay_module._safe_import_pfsspy()

    source_hmi_map = sunpy.map.Map(str(hmi_fits))
    ctype1 = str(source_hmi_map.meta.get("CTYPE1", "")).strip().upper()
    ctype2 = str(source_hmi_map.meta.get("CTYPE2", "")).strip().upper()
    is_synoptic_cea = (
        ctype1 in {"CRLN-CEA", "HGLN-CEA"}
        and ctype2 in {"CRLT-CEA", "HGLT-CEA"}
    )

    if is_synoptic_cea:
        hmi_map = pfss_seed_map
        ny, nx = hmi_map.data.shape
        x_lims = (0, int(nx))
        y_lims = (0, int(ny))
        seed_map_mode = "Earth-visible HMI synoptic CEA radial field"
        # observer_lonlat_deg = getattr(obs0, "lonlat_deg", None)
        # observer_screen_quadrant = PFSS_SEED_VIEW_QUADRANT
        observer_lonlat_deg = None
        observer_screen_quadrant = None
    else:
        hmi_map = source_hmi_map
        x_lims, y_lims = hmi_roi_pixels_like_2d_script(hmi_map)
        seed_map_mode = "legacy full-disk HMI ROI"
        observer_lonlat_deg = None
        observer_screen_quadrant = None

    seeds = build_pfss_seeds_from_hmi_roi(
        hmi_map,
        pfss_output,
        x_lims_pix=x_lims,
        y_lims_pix=y_lims,
        n_seeds_x=PFSS_SEED_N_X,
        n_seeds_y=PFSS_SEED_N_Y,
        use_strong_field=False,
        field_threshold=PFSS_FIELD_THRESHOLD_G,
        r_seed_rsun=1.001,
        rng_seed=42,
        observer_lonlat_deg=observer_lonlat_deg,
        observer_screen_quadrant=observer_screen_quadrant,
    )

    tracer = None
    tracer_name = None

    if hasattr(tracing, "FortranTracer"):
        try:
            tracer = tracing.FortranTracer(max_steps="auto", step_size=1.0)
            tracer_name = "FortranTracer(max_steps='auto', step_size=1)"
        except TypeError:
            try:
                tracer = tracing.FortranTracer(max_steps=2000, step_size=1.0)
                tracer_name = "FortranTracer(max_steps=2000, step_size=1)"
            except Exception:
                tracer = None
        except Exception:
            tracer = None

    if tracer is None and hasattr(tracing, "PythonTracer"):
        try:
            tracer = tracing.PythonTracer(atol=1.0e-4, rtol=1.0e-4)
            tracer_name = "PythonTracer"
        except Exception:
            tracer = None

    if tracer is None:
        raise RuntimeError(
            "No available PFSS tracer backend "
            "(neither PythonTracer nor FortranTracer)."
        )

    print(f"[INFO] PFSS tracer (ROI-seeds): {tracer_name}")
    print(
        f"[INFO] PFSS seed map: {seed_map_mode}; "
        f"pixels x={x_lims}, y={y_lims}, "
        f"strong=True, thr={PFSS_FIELD_THRESHOLD_G:g} G, "
        f"view_quadrant={observer_screen_quadrant}, seeds={len(seeds)}"
    )

    target_frame = _hgc_frame(obstime_iso)
    n_lines = 0
    n_open = 0
    n_closed = 0

    for seed in seeds:
        if n_lines >= PFSS_MAX_LINES:
            break
        try:
            container = tracer.trace(seed, pfss_output)
        except Exception:
            continue
        if container is None:
            continue

        field_lines = getattr(container, "field_lines", None)
        if field_lines is None:
            field_lines = [container]

        for field_line in field_lines:
            if n_lines >= PFSS_MAX_LINES:
                break

            is_open = None
            if hasattr(field_line, "open"):
                is_open = bool(field_line.open)
            elif hasattr(field_line, "is_open"):
                is_open = bool(field_line.is_open)

            coords = getattr(
                field_line,
                "coords",
                getattr(field_line, "coordinates", None),
            )
            if coords is None:
                continue
            try:
                coords_hgc = coords.transform_to(target_frame)
                xyz = coords_hgc.cartesian.xyz.to_value(u.R_sun).T
            except Exception:
                continue

            if xyz.ndim != 2 or xyz.shape[0] < 2 or xyz.shape[1] != 3:
                continue
            if not np.all(np.isfinite(xyz)):
                continue
            if np.nanmax(np.abs(xyz)) > 1.0e3:
                continue

            if is_open is None:
                try:
                    end_r = coords_hgc.radius[-1].to_value(u.R_sun)
                    is_open = bool(end_r > 2.0)
                except Exception:
                    is_open = False

            if is_open:
                n_open += 1
                color = "red"
            else:
                n_closed += 1
                color = "black"

            plotter.add_mesh(
                overlay_module._polyline_from_points(xyz),
                color=color,
                line_width=max(2, int(PFSS_LINE_WIDTH)),
                opacity=1.0,
                lighting=False,
                render_lines_as_tubes=True,
                pickable=False,
            )
            n_lines += 1

    print(
        f"[INFO] PFSS(ROI-seeds) lines: total={n_lines}, "
        f"open={n_open}, closed={n_closed}"
    )
    return {
        "rss": float(PFSS_RSS),
        "nrho": int(PFSS_NRHO),
        "n_lines": int(n_lines),
        "n_open": int(n_open),
        "n_closed": int(n_closed),
        "tracer": tracer_name,
        "seed_mode": (
            "HMI_synoptic_CEA_strong_field"
            if is_synoptic_cea
            else "HMI_ROI_strong_field"
        ),
        "field_threshold_G": float(PFSS_FIELD_THRESHOLD_G),
        "seed_view_quadrant": observer_screen_quadrant,
        "x_lims_pix": tuple(map(int, x_lims)),
        "y_lims_pix": tuple(map(int, y_lims)),
    }


# =============================================================================
# Output and rendering
# =============================================================================

def save_sampled_products(sampled: SpheroidSampling) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_SPHEROID_VTP:
        sampled.surface.save(str(OUTPUT_VTP))
        print(f"[OK] Saved Spheroid VTP: {OUTPUT_VTP}")
    if SAVE_SAMPLED_NPZ:
        np.savez_compressed(
            OUTPUT_NPZ,
            points_hgc_rsun=np.asarray(sampled.surface.points, dtype=np.float32),
            faces=np.asarray(sampled.surface.faces, dtype=np.int64),
            ne_cm3=sampled.density_cm3.astype(np.float32),
            B_G=sampled.magnetic_field_g.astype(np.float32),
            alfven_speed_km_s=sampled.alfven_speed_km_s.astype(np.float32),
            valid_alfven=sampled.valid.astype(np.uint8),
            pfss_extrapolated=sampled.pfss_extrapolated.astype(np.uint8),
            mu_e=float(MU_E),
            pfss_rss=float(PFSS_RSS),
            pfss_outside_rss_mode=str(PFSS_OUTSIDE_RSS_MODE),
            target_time=str(TARGET_TIME),
            spheroid_time_iso=str(SPHEROID_TIME_ISO),
            tomography_npz=str(TOMOGRAPHY_NPZ),
        )
        print(f"[OK] Saved sampled NPZ: {OUTPUT_NPZ}")


def render_scene(
    overlay_module: ModuleType,
    volume: TomographyVolume,
    pfss_output,
    pfss_seed_map,
    spheroid_params,
    sampled: SpheroidSampling,
) -> None:
    off_screen = not SHOW_GUI
    if SHOW_GUI and not os.environ.get("DISPLAY"):
        print("[WARN] DISPLAY is not set; forcing off-screen rendering.")
        off_screen = True
        try:
            pv.start_xvfb()
        except Exception as exc:
            print(f"[WARN] pv.start_xvfb failed: {exc}")

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")
    try:
        plotter.enable_depth_peeling()
    except Exception:
        pass
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass

    if SHOW_SUN:
        plotter.add_mesh(
            pv.Sphere(radius=1.0, theta_resolution=80, phi_resolution=80),
            color="magenta",
            opacity=0.18,
        )
        overlay_module.add_solar_latlon_grid(
            plotter,
            radius=1.002,
            dlon_deg=30.0,
            dlat_deg=30.0,
            line_width=1,
            opacity=0.45,
        )

    if SHOW_TOMOGRAPHY_ISOSURFACES:
        sg = overlay_module.build_tomography_structured_grid(
            volume.grid,
            volume.ne_1d_cm3,
        )
        overlay_module.add_isosurfaces(
            plotter,
            sg,
            TOMO_FREQ_MHZ_LIST,
            harmonic=ISOSURFACE_HARMONIC,
            opacity=ISOSURFACE_OPACITY,
            colors=overlay_module._colors_for_iso_freqs(TOMO_FREQ_MHZ_LIST),
            return_surfaces=False,
            range_text_mode="none",
        )

    if SHOW_PFSS_FIELD_LINES:
        try:
            add_pfss_field_lines_from_output(
                plotter,
                overlay_module,
                pfss_output,
                pfss_seed_map,
                volume.observer,
                hmi_fits=HMI_FITS,
                obstime_iso=SPHEROID_TIME_ISO,
            )
        except Exception as exc:
            print(f"[WARN] PFSS field-line rendering skipped: {exc}")

    # Gray base reveals portions where either tomography density or PFSS B is unavailable.
    plotter.add_mesh(
        sampled.surface,
        color=INVALID_SURFACE_COLOR,
        opacity=INVALID_SURFACE_OPACITY,
        smooth_shading=True,
        lighting=True,
        pickable=False,
    )

    valid_mesh = sampled.surface.threshold(
        value=(0.5, 1.5),
        scalars="valid_alfven",
        preference="point",
        all_scalars=True,
    )
    valid_values = sampled.alfven_speed_km_s[sampled.valid]
    if valid_mesh.n_points > 0 and valid_values.size > 0:
        low, high = np.nanpercentile(
            valid_values,
            [float(ALFVEN_CLIM_PERCENTILES[0]), float(ALFVEN_CLIM_PERCENTILES[1])],
        )
        if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high <= low:
            low = float(np.nanmin(valid_values))
            high = float(np.nanmax(valid_values))
        plotter.add_mesh(
            valid_mesh,
            scalars="alfven_speed_km_s",
            cmap=ALFVEN_CMAP,
            clim=(float(low), float(high)),
            log_scale=bool(ALFVEN_LOG_SCALE),
            opacity=VALID_SURFACE_OPACITY,
            smooth_shading=True,
            lighting=True,
            pickable=False,
            scalar_bar_args={
                "title": "Alfven speed [km/s]",
                "vertical": True,
                "position_x": 0.84,
                "position_y": 0.15,
                "height": 0.68,
                "width": 0.10,
                "title_font_size": 14,
                "label_font_size": 12,
            },
        )

    # Retain the Spheroid wireframe, footprint, and anchor/apex markers, but do
    # not add the original magenta surface because it would cover the v_A map.
    overlay_module.add_spheroid_dome_3d(
        plotter,
        spheroid_params,
        obstime_iso=SPHEROID_TIME_ISO,
        observer="earth",
        obs0=volume.observer,
        color="magenta",
        surface_opacity=0.0,
        wire_opacity=0.6,
        footprint_opacity=1.0,
        line_width=1,
        footprint_width=3,
        marker_radius=0.045,
        show_surface=False,
        show_wireframe=True,
        show_footprint=True,
        show_markers=True,
        return_surface=False,
    )

    overlay_module.add_sun_earth_line(
        plotter,
        volume.observer,
        length_rsun=5.0,
        start_rsun=1.0,
        color="orange",
        line_width=4,
    )
    overlay_module.add_physical_axes_triad(
        plotter,
        volume.observer,
        origin_rsun=1.0,
        axis_len=1.4,
        shaft_radius=0.025,
        tip_radius=0.055,
        tip_length=0.20,
        label_font_size=11,
    )

    n_valid = int(np.count_nonzero(sampled.valid))
    n_extrap = int(np.count_nonzero(sampled.valid & sampled.pfss_extrapolated))
    info = (
        f"Tomography target: {TARGET_TIME} UTC, window: +/-{SEARCH_WINDOW_DAYS:g} d\n"
        f"Spheroid: {SPHEROID_TIME_ISO} UTC, apex={SPHEROID_APEX_RSUN:.2f} R_sun\n"
        f"PFSS: Rss={PFSS_RSS:.2f} R_sun, outer mode={PFSS_OUTSIDE_RSS_MODE}\n"
        f"rho = {MU_E:.4f} m_p n_e; valid points={n_valid}; extrapolated={n_extrap}"
    )
    plotter.add_text(info, position="upper_right", font_size=11, color="black")

    overlay_module.set_camera_from_observation(
        plotter,
        volume.observer,
        distance_rsun=5.0,
    )
    try:
        plotter.reset_camera_clipping_range()
        plotter.render()
    except Exception:
        pass

    if SAVE_PNG:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if off_screen:
        screenshot = str(OUTPUT_PNG) if SAVE_PNG else None
        plotter.show(screenshot=screenshot, auto_close=True)
        if screenshot:
            print(f"[OK] Saved PNG: {OUTPUT_PNG}")
        return

    if SHOW_GUI:
        if SAVE_PNG:
            print("[INFO] Close the PyVista window to save its final camera view.")
            plotter.show(auto_close=False)
            try:
                plotter.screenshot(str(OUTPUT_PNG))
                print(f"[OK] Saved PNG: {OUTPUT_PNG}")
            finally:
                plotter.close()
        else:
            plotter.show()
    elif SAVE_PNG:
        plotter.show(screenshot=str(OUTPUT_PNG), auto_close=True)
        print(f"[OK] Saved PNG: {OUTPUT_PNG}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    started = time.perf_counter()
    print(f"[INFO] Tomography module: {TOMO_MODULE_PATH}")
    print(f"[INFO] Overlay module:    {OVERLAY_MODULE_PATH}")

    tomo_module = load_python_module(TOMO_MODULE_PATH, "main_multi_tomo_companion")
    overlay_module = load_python_module(
        OVERLAY_MODULE_PATH,
        "main_npz_tomo_sphe_pfss_companion",
    )

    if RECOMPUTE_TOMOGRAPHY:
        npz_path = run_tomography_to_npz(tomo_module)
    else:
        npz_path = TOMOGRAPHY_NPZ
        print(f"[INFO] Reusing tomography NPZ: {npz_path}")

    volume = load_tomography_volume(npz_path, overlay_module)

    print(f"[INFO] Computing PFSS from HMI: {HMI_FITS}")
    pfss_output, pfss_seed_map = compute_pfss_output_from_hmi(
        overlay_module,
        HMI_FITS,
        nrho=PFSS_NRHO,
        rss=PFSS_RSS,
        helio_shape=PFSS_HELIO_SHAPE,
        obstime_iso=SPHEROID_TIME_ISO,
    )

    spheroid_params, sampled = build_and_sample_spheroid(
        overlay_module,
        volume,
        pfss_output,
    )
    save_sampled_products(sampled)
    render_scene(
        overlay_module,
        volume,
        pfss_output,
        pfss_seed_map,
        spheroid_params,
        sampled,
    )
    print(f"[TIME] Total elapsed: {time.perf_counter() - started:.2f} s")


if __name__ == "__main__":
    main()
