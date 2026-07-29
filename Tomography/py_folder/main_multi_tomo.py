#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_multi_tomo.py

Regularized tomography (Tikhonov) for coronal electron density from time-series pB images,
aiming to be *algorithmically consistent* with the SSC/Ne3dTomo (V1.1) preprocessing logic:

- "preview_data.pro": (optional) rebin to 128x128, noise floor, and basic QC/preview.
- "pbmap_despike.pro" + "fix_nan.pro": robust despike and NaN repair on the rebinned maps.
- "cor1_getpbr.pro" + "get_pbrlc.pro": polar (r, PA) sampling used for background/noise proxy.
- "get_cor1_bbk.pro": low-harmonic (FFT) smoothing over PA to estimate a radial background ybk(r).
- "map_get_coord.pro" / "map_get_pixel.pro": image<->coordinate mapping (handled here via full WCS).

Important practical note:
The uploaded IDL/F90 sources are used to port the scientifically relevant operations: single-pass
fix_nan, FREBIN-like rebinning, SSW temporal unspiking, Ne3dTomo pB(r) background, and
second-order spherical regularization. The forward model and
regularization are implemented in the same *form* (weighted least-squares + smoothness penalty),
but this is not a byte-for-byte reproduction of SSC's Fortran toolchain.

Dependencies:
  pip install numpy astropy scipy pyvista pyvistaqt

- LASCO-C2 pB / K-Cor pB のダウンロードと結合は `SOHO/SECCHI/LASCO/py_folder/download_integrate_kcor_lasco_pb.py` で事前に完了させる。
 
- このファイルは、作成済みの `pB_Kcor_LASCO_axi_*.fits` と `COR1A_pb_pre_*.fits` を読み込み、tomography のみを実行する。

- STEREO-A/SECCHI/COR1 pB: STEREO-A/SECCHI/COR1/py_folder/download_cor1a_pb.pyを実行して、STEREO-A/SECCHI/COR1/Rawdata/<YYYYMMDD>_<HHMMSS>_n4c1A.fts, dc1A_p000/120/240.ftsをダウンロード、"/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/pB/Rawdata/COR1A_pb_pre_<YYYYMMDD>_<HHMMSS>.fits"を作成。その後、STEREO-A/SECCHI/COR1/make_cor1a_pb.proをSSWIDLで回す。

- pB_Kcor_LASCO_axi_*.fitsとCOR1A_ne_<YYYYMMDD>_<HHMMSS>.fitsを使ってtomographyを行う。(自動)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from datetime import datetime, timedelta
import hashlib
import pickle
import re
import csv
import json
import time

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import median_filter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg
try:
    from scipy.optimize import minimize
except Exception:
    minimize = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy import ndimage
except Exception:
    ndimage = None

try:
    import pyvista as pv
except ImportError as e:
    raise SystemExit(
        "PyVista is required for GUI rendering. Install e.g.:\n"
        "  pip install pyvista pyvistaqt\n"
    ) from e


# ----------------------------
# Physical constants
# ----------------------------
RSUN_CM = 6.957e10  # cm
RSUN_M = 6.957e8    # m
RE_CM = 2.8179403262e-13  # classical electron radius [cm]
SIGMA_T = 6.6524587321e-25  # Thomson cross section [cm^2]
ICEN = 1.0  # SSC normalization constant (matches provided port)
DEFAULT_LIMB_U = 0.56  # Legacy SSC-compatible fallback only.

# Nominal continuum bandpasses used when a measured total system response is not
# supplied.  The forward kernel is averaged over each bandpass in coefficient
# space, rather than evaluating one arbitrary central wavelength.
DEFAULT_LIMB_BANDPASS_NM = {
    "lasco_c2": (540.0, 640.0),       # LASCO-C2 orange filter
    "kcor": (720.0, 750.0),           # K-Cor 735 nm, ~30 nm FWHM
    "cor1a": (644.75, 667.25),        # SECCHI/COR1 656 nm, ~22.5 nm FWHM
}
DEFAULT_LIMB_WEIGHT_HDU_NAMES = ("KCOR_WEIGHT", "KCOR_WGT", "KCORFRAC")


def limb_dark_coefficient(lambda_nm, allen: bool = False):
    """Return the linear solar limb-darkening coefficient u(lambda).

    This is the SSWIDL ``limb_dark.pro`` interpolation used by the supplied
    ``set_u_from_instrument.py``.  The default table is Pierce & Allen;
    ``allen=True`` selects the coarser Allen table.
    """
    scalar_input = np.isscalar(lambda_nm)
    lam = np.asarray(lambda_nm, dtype=np.float64)

    if bool(allen):
        wavelength = 10.0 * np.array(
            [30, 32, 35, 37, 38, 40, 45, 50, 55, 60, 80, 100],
            dtype=np.float64,
        )
        mean_to_center = np.array(
            [648, 685, 705, 710, 710, 718, 755, 782, 803, 817, 862, 886],
            dtype=np.float64,
        ) / 1000.0
    else:
        wavelength = 305.0 + 10.0 * np.arange(70, dtype=np.float64)
        mean_to_center = np.concatenate([
            np.array([0.636, 0.657, 0.674, 0.685, 0.694, 0.703, 0.691, 0.680, 0.687, 0.698]),
            np.array([0.710, 0.718, 0.726, 0.734, 0.744, 0.753, 0.759, 0.765, 0.769, 0.775]),
            np.array([0.779, 0.784, 0.788, 0.793, 0.798, 0.802, 0.805, 0.809, 0.812, 0.815]),
            np.array([0.817, 0.820, 0.823, 0.825, 0.828, 0.830, 0.834, 0.836, 0.838, 0.841]),
            np.array([0.843, 0.845, 0.847, 0.849, 0.851, 0.853, 0.855, 0.857, 0.859, 0.861]),
            np.array([0.863, 0.864, 0.865, 0.867, 0.868, 0.869, 0.870, 0.872, 0.873, 0.874]),
            np.array([0.875, 0.876, 0.877, 0.878, 0.879, 0.880, 0.881, 0.882, 0.883, 0.884]),
        ]).astype(np.float64)

    if np.any((lam < wavelength[0]) | (lam > wavelength[-1])):
        raise ValueError(
            f"lambda outside limb-darkening table range "
            f"[{wavelength[0]:.1f}, {wavelength[-1]:.1f}] nm: {lambda_nm}"
        )
    mean_flux_ratio = np.interp(lam, wavelength, mean_to_center)
    out = 3.0 * (1.0 - mean_flux_ratio)
    return float(out) if scalar_input else out


def limb_kernel_coefficients_from_u(
    u: float,
    normalize_msb: bool = True,
) -> Tuple[float, float]:
    """Return coefficients multiplying the van-de-Hulst A and B functions."""
    u = float(u)
    if not np.isfinite(u):
        raise ValueError(f"Invalid limb-darkening coefficient: {u}")
    norm = thomson_msb_normalization_factor(u) if bool(normalize_msb) else 1.0
    return float((1.0 - u) * norm), float(u * norm)


def instrument_limb_kernel_coefficients(
    instrument: str,
    normalize_msb: bool = True,
    use_allen: bool = False,
    bandpass_nm_by_instrument=None,
    u_override_by_instrument=None,
    samples: int = 401,
) -> Tuple[float, float, float, Tuple[float, float]]:
    """Return bandpass-averaged (u_eff, A coefficient, B coefficient, bandpass).

    The average is performed on the actual kernel coefficients
    ``(1-u)/(1-u/3)`` and ``u/(1-u/3)``.  This is more faithful than inserting
    a single central-wavelength u into a mean-solar-brightness-normalized kernel.
    The nominal filter is treated as top-hat when a measured throughput curve is
    unavailable.
    """
    inst = str(instrument).strip().lower()
    bandpasses = dict(DEFAULT_LIMB_BANDPASS_NM)
    if isinstance(bandpass_nm_by_instrument, dict):
        bandpasses.update(bandpass_nm_by_instrument)
    overrides = dict(u_override_by_instrument or {}) if isinstance(u_override_by_instrument, dict) else {}

    if inst in overrides:
        u_values = np.array([float(overrides[inst])], dtype=np.float64)
        band = tuple(float(v) for v in bandpasses.get(inst, (np.nan, np.nan)))
    else:
        if inst not in bandpasses:
            raise ValueError(
                f"No limb-darkening bandpass configured for instrument {inst!r}. "
                "Set limb_u_bandpass_nm_by_instrument or limb_u_override_by_instrument."
            )
        lo, hi = (float(v) for v in bandpasses[inst])
        if not (np.isfinite(lo) and np.isfinite(hi) and hi >= lo):
            raise ValueError(f"Invalid bandpass for {inst!r}: {(lo, hi)}")
        band = (lo, hi)
        n = max(1, int(samples))
        wavelengths = np.array([lo], dtype=np.float64) if hi == lo else np.linspace(lo, hi, n)
        u_values = np.asarray(limb_dark_coefficient(wavelengths, allen=bool(use_allen)), dtype=np.float64)

    coeff_a = []
    coeff_b = []
    for u_value in u_values:
        a, b = limb_kernel_coefficients_from_u(float(u_value), normalize_msb=bool(normalize_msb))
        coeff_a.append(a)
        coeff_b.append(b)
    a_mean = float(np.mean(coeff_a))
    b_mean = float(np.mean(coeff_b))
    u_eff = float(b_mean / max(a_mean + b_mean, 1e-300))
    return u_eff, a_mean, b_mean, band


def _read_optional_kcor_weight_hdu(
    pb_fits: Path,
    orig_n: int,
    out_n: int,
    hdu_names=DEFAULT_LIMB_WEIGHT_HDU_NAMES,
) -> Optional[np.ndarray]:
    """Read an exact K-Cor contribution map when the merged FITS provides one."""
    wanted = {str(name).strip().upper() for name in hdu_names}
    try:
        with fits.open(pb_fits) as hdul:
            for hdu in hdul[1:]:
                name = str(getattr(hdu, "name", "")).strip().upper()
                if name not in wanted or getattr(hdu, "data", None) is None:
                    continue
                arr = np.asarray(hdu.data, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                    raise ValueError(
                        f"{pb_fits.name} extension {name!r} must be a square 2-D K-Cor weight map; "
                        f"got {arr.shape}."
                    )
                if arr.shape != (out_n, out_n):
                    arr = rebin_idl_linear(arr, int(out_n))
                return np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0)
    except Exception as exc:
        print(f"[WARN] Could not read optional K-Cor weight extension from {pb_fits.name}: {exc}")
    return None


def build_observation_limb_kernel_model(
    pb_fits: Path,
    hdr,
    rho: np.ndarray,
    orig_n: int,
    out_n: int,
    fallback_u: float,
    mode: str = "instrument_bandpass",
    normalize_msb: bool = True,
    use_allen: bool = False,
    bandpass_nm_by_instrument=None,
    u_override_by_instrument=None,
    bandpass_samples: int = 401,
    weight_hdu_names=DEFAULT_LIMB_WEIGHT_HDU_NAMES,
) -> Tuple[Tuple[str, ...], Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], Optional[np.ndarray], str]:
    """Construct per-observation, and when needed per-pixel, Thomson-kernel coefficients."""
    mode = str(mode or "fixed").strip().lower()
    if mode in ("fixed", "legacy", "legacy_fixed", "scalar"):
        a, b = limb_kernel_coefficients_from_u(float(fallback_u), normalize_msb=bool(normalize_msb))
        return ("fixed",), (float(fallback_u),), (a,), (b,), None, "fixed_scalar"
    if mode not in ("instrument_bandpass", "instrument", "bandpass", "best"):
        raise ValueError(f"Unknown limb_u_mode={mode!r}")

    group = tomography_observation_group_key(Path(pb_fits))
    if group == "cor1a":
        instruments = ("cor1a",)
    elif group == "earth_lasco_only":
        instruments = ("lasco_c2",)
    elif group == "earth_merged":
        instruments = ("kcor", "lasco_c2")
    else:
        header_text = " ".join(
            str(hdr.get(key, "")) for key in ("INSTRUME", "DETECTOR", "TELESCOP", "OBSRVTRY")
        ).lower()
        if "cor1" in header_text:
            instruments = ("cor1a",)
        elif "k-cor" in header_text or "kcor" in header_text:
            instruments = ("kcor",)
        elif "lasco" in header_text or "c2" in header_text:
            instruments = ("lasco_c2",)
        else:
            a, b = limb_kernel_coefficients_from_u(float(fallback_u), normalize_msb=bool(normalize_msb))
            print(
                f"[WARN] Could not identify instrument for {pb_fits.name}; "
                f"using fallback fixed u={float(fallback_u):.6g}."
            )
            return ("fixed",), (float(fallback_u),), (a,), (b,), None, "fallback_fixed"

    u_values = []
    coeff_a = []
    coeff_b = []
    band_text = []
    for inst in instruments:
        u_eff, a, b, band = instrument_limb_kernel_coefficients(
            inst,
            normalize_msb=bool(normalize_msb),
            use_allen=bool(use_allen),
            bandpass_nm_by_instrument=bandpass_nm_by_instrument,
            u_override_by_instrument=u_override_by_instrument,
            samples=int(bandpass_samples),
        )
        u_values.append(u_eff)
        coeff_a.append(a)
        coeff_b.append(b)
        band_text.append(f"{inst}:{band[0]:g}-{band[1]:g}nm")

    weight_maps = None
    model_source = "pure_instrument"
    if group == "earth_merged":
        kcor_fraction = _read_optional_kcor_weight_hdu(
            Path(pb_fits), orig_n=int(orig_n), out_n=int(out_n), hdu_names=weight_hdu_names
        )
        if kcor_fraction is not None:
            model_source = "exact_kcor_weight_hdu"
        else:
            blend_in = float(hdr.get("BLENDIN", 2.2))
            blend_out = float(hdr.get("BLENDOUT", 2.5))
            kcor_rmin = float(hdr.get("KCORRMIN", 0.0))
            kcor_rmax = float(hdr.get("KCORRMAX", blend_out))
            if not (np.isfinite(blend_in) and np.isfinite(blend_out) and blend_out > blend_in):
                raise ValueError(
                    f"Invalid/missing BLENDIN/BLENDOUT in merged FITS {pb_fits.name}: "
                    f"{blend_in}, {blend_out}"
                )
            kcor_fraction = np.zeros_like(rho, dtype=np.float64)
            eligible = np.isfinite(rho) & (rho >= kcor_rmin) & (rho <= kcor_rmax)
            inner = eligible & (rho <= blend_in)
            kcor_fraction[inner] = 1.0
            overlap = eligible & (rho > blend_in) & (rho < blend_out)
            kcor_fraction[overlap] = (blend_out - rho[overlap]) / (blend_out - blend_in)
            kcor_fraction = np.clip(kcor_fraction, 0.0, 1.0)
            model_source = "header_radial_blend_approximation"
            if int(hdr.get("KCORHPX", 0) or 0) > 0:
                print(
                    f"[WARN] {pb_fits.name} contains K-Cor-filled LASCO holes but no exact "
                    "KCOR_WEIGHT extension. Their limb kernel uses the header radial blend approximation."
                )
        weight_maps = np.stack([kcor_fraction, 1.0 - kcor_fraction], axis=0)

    print(
        f"[LIMB-U] {pb_fits.name}: mode=instrument_bandpass, components={instruments}, "
        f"u_eff={tuple(round(v, 6) for v in u_values)}, bands={band_text}, source={model_source}"
    )
    return (
        tuple(instruments),
        tuple(float(v) for v in u_values),
        tuple(float(v) for v in coeff_a),
        tuple(float(v) for v in coeff_b),
        weight_maps,
        f"instrument_bandpass:{model_source}",
    )


# ----------------------------
# Ray cache helpers
# ----------------------------
RAY_CACHE_VERSION = "ray_cache_v3_instrument_limb_kernel"
_RAY_MEMORY_CACHE: dict[str, object] = {}


def ensure_dir(path: Path | str) -> Path:
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


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
    """Return a stable cache key for one observation/grid/ray-construction setup.

    The key intentionally depends only on quantities that change the ray geometry
    or the Thomson/LOS weights.  It does not include the modification time or file
    size of this Python script or of the input FITS files.  Therefore changing
    solver/output parameters such as LAM, WT_NR, MAXITER, Summary CSV, or PNG paths
    will not invalidate the ray cache.

    If the ray-building algorithm itself is changed, manually update
    RAY_CACHE_VERSION to invalidate old ray files.
    """
    h = hashlib.sha256()
    _hash_update_value(h, RAY_CACHE_VERSION)

    p = Path(pb_path).expanduser().resolve()
    _hash_update_value(h, str(p))

    _hash_update_value(h, float(ds_rsun))
    _hash_update_value(h, float(r_min))
    _hash_update_value(h, float(r_max))
    _hash_update_value(h, float(limb_u))
    _hash_update_value(h, getattr(obs, "limb_u_model", "fixed_scalar"))
    _hash_update_value(h, getattr(obs, "limb_component_names", ("fixed",)))
    _hash_update_array(h, np.asarray(getattr(obs, "limb_u_values", (float(limb_u),)), dtype=np.float64))
    _hash_update_array(h, np.asarray(getattr(obs, "limb_coeff_a", (1.0 - float(limb_u),)), dtype=np.float64))
    _hash_update_array(h, np.asarray(getattr(obs, "limb_coeff_b", (float(limb_u),)), dtype=np.float64))
    limb_weight_maps = getattr(obs, "limb_component_weight_maps", None)
    if limb_weight_maps is not None:
        _hash_update_array(
            h,
            np.asarray(limb_weight_maps, dtype=np.float64).reshape(
                len(getattr(obs, "limb_u_values", (float(limb_u),))), -1
            )[:, np.asarray(obs.idx_map, dtype=np.int64)],
        )
    _hash_update_value(h, bool(thomson_normalize_msb))
    _hash_update_value(h, float(thomson_kernel_scale))
    _hash_update_value(h, getattr(obs, "lonlat_deg", None))

    _hash_update_array(h, np.asarray(grid.r_edges, dtype=np.float64))
    _hash_update_array(h, np.asarray(grid.th_edges, dtype=np.float64))
    _hash_update_array(h, np.asarray(grid.ph_edges, dtype=np.float64))
    _hash_update_array(h, np.asarray(obs.idx_map, dtype=np.int64))
    _hash_update_array(h, np.asarray(obs.x, dtype=np.float64))
    _hash_update_array(h, np.asarray(obs.y, dtype=np.float64))
    _hash_update_array(h, np.asarray(obs.cam_x, dtype=np.float64))
    _hash_update_array(h, np.asarray(obs.cam_y, dtype=np.float64))
    _hash_update_array(h, np.asarray(obs.cam_z, dtype=np.float64))
    return h.hexdigest()


def ray_cache_path(cache_dir: Path | str, key: str) -> Path:
    return Path(cache_dir).expanduser() / f"ray_{key}.pkl"


def load_cached_ray(key: str, cache_dir: Path | str = "") -> Optional[object]:
    """Load one cached RayBundle from memory first, then from disk if available."""
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

        if "vox_idx" in payload and "vox_w" in payload:
            ray = RayBundle(vox_idx=payload["vox_idx"], vox_w=payload["vox_w"])
        else:
            ray = payload.get("ray")
        if ray is None:
            return None

        _RAY_MEMORY_CACHE[key] = ray
        return ray
    except Exception as exc:
        print(f"[WARN] Failed to load cached rays from {fp}: {exc}")
        return None


def save_cached_ray(key: str, ray: object, cache_dir: Path | str = "") -> None:
    """Save one RayBundle to memory and, optionally, to disk."""
    _RAY_MEMORY_CACHE[key] = ray

    if not cache_dir:
        return

    fp = ray_cache_path(cache_dir, key)
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = fp.with_suffix(fp.suffix + ".tmp")
        payload = {
            "version": RAY_CACHE_VERSION,
            "vox_idx": getattr(ray, "vox_idx", None),
            "vox_w": getattr(ray, "vox_w", None),
        }
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(fp)
    except Exception as exc:
        print(f"[WARN] Failed to save cached rays to {fp}: {exc}")


def thomson_msb_normalization_factor(u: float) -> float:
    """Return the limb-darkening normalization factor for pB in mean-solar-brightness units.

    The commonly used van de Hulst/Kramar pB kernel contains the factor
    1/(1 - u/3) when the brightness is normalized to the mean solar disk
    brightness.  Keeping this factor explicit avoids hiding an absolute-density
    scale choice inside the forward matrix.
    """
    den = 1.0 - float(u) / 3.0
    if not np.isfinite(den) or den <= 0:
        raise ValueError(f"Invalid limb-darkening coefficient u={u}; 1-u/3 must be positive.")
    return 1.0 / den



# ----------------------------
# Tomography input time helpers
# ----------------------------
def parse_target_datetime(value: str | datetime) -> datetime:
    """
    Parse a target time used for the rotational-tomography data window.

    Accepted string formats:
      - YYYYMMDD_HHMM
      - YYYYMMDD_HHMMSS
      - YYYY-MM-DD HH:MM
      - YYYY-MM-DD HH:MM:SS
      - YYYY-MM-DDTHH:MM
      - YYYY-MM-DDTHH:MM:SS
    """
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
    raise ValueError(
        f"Cannot parse target_time={value!r}. Use e.g. '20220613_0300', "
        f"'20220613_030000', or '2022-06-13T03:00:00'."
    )

def parse_pb_filename_datetime(path: Path) -> Optional[datetime]:
    """
    Extract observation time from supported pB FITS filenames.

    Supported patterns:
      - pB_Kcor_LASCO_axi_<YYYYMMDD>_<HHMM>.fits
      - COR1A_pb_pre_<YYYYMMDD>_<HHMMSS>.fits
      - C2-PB-<YYYYMMDD>_<HHMM>.fts / .fits
      - pB_LASCO_C2_only_<YYYYMMDD>_<HHMM>.fits
      - <YYYYMMDD>_<HHMMSS>_kcor_l2_pb.fts / .fits
      - <YYYYMMDD>_<HHMMSS>_kcor_l2_pbavg.fts / .fits
      - <YYYYMMDD>_<HHMMSS>_kcor_l2_pb_avg.fts / .fits
    """
    name = Path(path).name

    m = re.fullmatch(r"pB_Kcor_LASCO_axi_(\d{8})_(\d{4})\.fits", name)
    if m:
        return datetime.strptime(m.group(1) + "_" + m.group(2), "%Y%m%d_%H%M")

    m = re.fullmatch(r"COR1A_pb_pre_(\d{8})_(\d{6})\.fits", name)
    if m:
        return datetime.strptime(m.group(1) + "_" + m.group(2), "%Y%m%d_%H%M%S")

    m = re.fullmatch(r"C2-PB-(\d{8})_(\d{4})\.(?:fts|fits)(?:\.gz)?", name)
    if m:
        return datetime.strptime(m.group(1) + "_" + m.group(2), "%Y%m%d_%H%M")

    m = re.fullmatch(r"pB_LASCO_C2_only_(\d{8})_(\d{4})\.fits", name)
    if m:
        return datetime.strptime(m.group(1) + "_" + m.group(2), "%Y%m%d_%H%M")

    m = re.fullmatch(r"(\d{8})_(\d{6})_kcor_l2_pb(?:_?avg)?\.(?:fts|fits)(?:\.gz)?", name)
    if m:
        return datetime.strptime(m.group(1) + "_" + m.group(2), "%Y%m%d_%H%M%S")

    return None


# ----------------------------
# Time-window search / download helpers
# ----------------------------

def find_pb_fits_in_time_window(
    data_dir: Path,
    target_time: str | datetime,
    window_days: float = 7.0,
    include_kcor_lasco: bool = True,
    include_cor1a: bool = True,
    include_lasco_only: bool = True,
    cor1a_data_dir: Optional[Path | str] = None,
    match_cor1a_to_earth: bool = True,
    max_cor1a_match_minutes: Optional[float] = 90.0,
    exclude_earth_times: Optional[Sequence[str | datetime]] = None,
    keep_cor1a_for_excluded_earth_times: bool = True,
) -> List[Path]:
    """
    Find tomography-ready pB FITS files within target_time +/- window_days.

    Earth-view pB products are searched in ``data_dir``:
      - pB_Kcor_LASCO_axi_YYYYMMDD_HHMM.fits
      - pB_LASCO_C2_only_YYYYMMDD_HHMM.fits

    COR1A pB products are searched in ``cor1a_data_dir``:
      - COR1A_pb_pre_YYYYMMDD_HHMMSS.fits

    Parameters
    ----------
    exclude_earth_times
        Earth-viewだけを除外する観測時刻。
        例:
            ["20220616_2104", "20220617_0258"]

        この指定はEarth-viewファイルだけに適用され、COR1Aファイルを
        直接除外しない。

    keep_cor1a_for_excluded_earth_times
        Trueの場合、exclude_earth_timesで除外したEarth-view時刻も
        COR1A選択用の時刻アンカーとして使用する。

        したがって、Earth-viewは除外されるが、その時刻に最も近い
        COR1Aはtomography入力に保持される。

    Notes
    -----
    ``match_cor1a_to_earth=True`` の場合、各Earth-view時刻に最も近い
    COR1Aを1つ選択する。近接時刻のCOR1Aをすべて読み込んでCOR1Aが
    過剰に重み付けされることを防ぐ。
    """
    earth_dir = Path(data_dir).expanduser()
    cor1a_dir = (
        Path(cor1a_data_dir).expanduser()
        if cor1a_data_dir
        else earth_dir
    )

    if (include_kcor_lasco or include_lasco_only) and not earth_dir.exists():
        raise FileNotFoundError(
            f"Earth-view pB directory not found: {earth_dir}"
        )

    if include_cor1a and not cor1a_dir.exists():
        raise FileNotFoundError(
            f"COR1A pB directory not found: {cor1a_dir}"
        )

    target_dt = parse_target_datetime(target_time)
    half_width = timedelta(days=float(window_days))
    t0 = target_dt - half_width
    t1 = target_dt + half_width

    # Earth-viewだけに適用する除外時刻を正規化する。
    excluded_earth_datetimes: set[datetime] = set()
    for value in exclude_earth_times or []:
        excluded_dt = parse_target_datetime(value)
        excluded_earth_datetimes.add(excluded_dt)

    earth_candidates: List[Path] = []
    cor1a_candidates: List[Path] = []

    if include_kcor_lasco:
        earth_candidates.extend(
            earth_dir.glob(
                "pB_Kcor_LASCO_axi_????????_????.fits"
            )
        )

    if include_lasco_only:
        earth_candidates.extend(
            earth_dir.glob(
                "pB_LASCO_C2_only_????????_????.fits"
            )
        )

    if include_cor1a:
        cor1a_candidates.extend(
            cor1a_dir.glob(
                "COR1A_pb_pre_????????_??????.fits"
            )
        )

    # ------------------------------------------------------------
    # 1. 時間窓内のEarth-viewをすべて取得する。
    #
    # このリストは、後で一部のEarth-viewを除外した後も、
    # COR1Aマッチング用の時刻アンカーとして利用できる。
    # ------------------------------------------------------------
    earth_selected_all: List[Tuple[datetime, Path]] = []

    for path in earth_candidates:
        obs_dt = parse_pb_filename_datetime(path)
        if obs_dt is None:
            continue

        if t0 <= obs_dt <= t1:
            earth_selected_all.append(
                (obs_dt, Path(path))
            )

    earth_selected_all.sort(
        key=lambda item: (item[0], item[1].name)
    )

    # ------------------------------------------------------------
    # 2. tomographyへ実際に入力するEarth-viewだけを選ぶ。
    #
    # exclude_earth_timesはEarth-viewにしか適用しない。
    # ------------------------------------------------------------
    earth_selected: List[Tuple[datetime, Path]] = []

    for obs_dt, path in earth_selected_all:
        if obs_dt in excluded_earth_datetimes:
            print(
                "[INFO][EARTH-EXCLUDE] "
                f"Earth-view only excluded: {path.name} "
                f"({obs_dt:%Y-%m-%d %H:%M:%S} UT)"
            )
            continue

        earth_selected.append(
            (obs_dt, path)
        )

    # ------------------------------------------------------------
    # 3. COR1Aを選ぶためのEarth-view時刻アンカーを決める。
    #
    # True:
    #   除外されたEarth-view時刻もCOR1Aマッチングに使用する。
    #
    # False:
    #   残されたEarth-view時刻だけにCOR1Aをマッチングする。
    # ------------------------------------------------------------
    if keep_cor1a_for_excluded_earth_times:
        earth_matching_anchors = earth_selected_all
    else:
        earth_matching_anchors = earth_selected

    # 時間窓内のCOR1A候補をすべて取得する。
    cor1a_selected_all: List[Tuple[datetime, Path]] = []

    for path in cor1a_candidates:
        obs_dt = parse_pb_filename_datetime(path)
        if obs_dt is None:
            continue

        if t0 <= obs_dt <= t1:
            cor1a_selected_all.append(
                (obs_dt, Path(path))
            )

    cor1a_selected_all.sort(
        key=lambda item: (item[0], item[1].name)
    )

    # tomography入力には、除外後のEarth-viewだけを追加する。
    selected: List[Tuple[datetime, Path]] = []
    selected.extend(earth_selected)

    matched_cor1a: List[Tuple[datetime, Path]] = []
    matched_keys: set[str] = set()

    if include_cor1a:
        if match_cor1a_to_earth and earth_matching_anchors:
            if not cor1a_selected_all:
                print(
                    "[WARN] COR1A matching requested, but no COR1A "
                    "pB files were found in the time window."
                )
            else:
                print(
                    "[INFO] Nearest COR1A pB selected for each "
                    "Earth-view time anchor:"
                )

                for earth_dt, earth_path in earth_matching_anchors:
                    best_cor_dt, best_cor_path = min(
                        cor1a_selected_all,
                        key=lambda item: abs(
                            (
                                item[0] - earth_dt
                            ).total_seconds()
                        ),
                    )

                    delta_min = abs(
                        (
                            best_cor_dt - earth_dt
                        ).total_seconds()
                    ) / 60.0

                    earth_is_excluded = (
                        earth_dt in excluded_earth_datetimes
                    )

                    if (
                        max_cor1a_match_minutes is not None
                        and delta_min
                        > float(max_cor1a_match_minutes)
                    ):
                        print(
                            f"       {earth_path.name} "
                            f"-> no COR1A within "
                            f"{float(max_cor1a_match_minutes):.1f} min "
                            f"(nearest={best_cor_path.name}, "
                            f"dt={delta_min:.2f} min)"
                        )
                        continue

                    key = (
                        str(best_cor_path.resolve())
                        if best_cor_path.exists()
                        else str(best_cor_path)
                    )

                    if key not in matched_keys:
                        matched_keys.add(key)
                        matched_cor1a.append(
                            (best_cor_dt, best_cor_path)
                        )

                    if earth_is_excluded:
                        status = (
                            " [Earth-view excluded; "
                            "COR1A retained]"
                        )
                    else:
                        status = ""

                    print(
                        f"       {earth_path.name} "
                        f"-> {best_cor_path.name} "
                        f"(dt={delta_min:.2f} min)"
                        f"{status}"
                    )
        else:
            # match_cor1a_to_earth=Falseの場合は、
            # 時間窓内のCOR1Aをすべて使用する。
            matched_cor1a = cor1a_selected_all

    selected.extend(matched_cor1a)

    # 完全に同一のファイルだけを除重複する。
    unique_selected: List[Tuple[datetime, Path]] = []
    seen: set[str] = set()

    for obs_dt, path in selected:
        key = (
            str(path.resolve())
            if path.exists()
            else str(path)
        )

        if key in seen:
            continue

        seen.add(key)
        unique_selected.append(
            (obs_dt, path)
        )

    unique_selected.sort(
        key=lambda item: (item[0], item[1].name)
    )

    n_earth = sum(
        1
        for _, path in unique_selected
        if (
            re.fullmatch(
                r"pB_Kcor_LASCO_axi_\d{8}_\d{4}\.fits",
                path.name,
            )
            or re.fullmatch(
                r"pB_LASCO_C2_only_\d{8}_\d{4}\.fits",
                path.name,
            )
        )
    )

    n_cor1a = sum(
        1
        for _, path in unique_selected
        if re.fullmatch(
            r"COR1A_pb_pre_\d{8}_\d{6}\.fits",
            path.name,
        )
    )

    print("[INFO] Search window:")
    print(
        f"       target_time = "
        f"{target_dt:%Y-%m-%d %H:%M:%S}"
    )
    print(
        f"       t0          = "
        f"{t0:%Y-%m-%d %H:%M:%S}"
    )
    print(
        f"       t1          = "
        f"{t1:%Y-%m-%d %H:%M:%S}"
    )

    print("[INFO] Search directories:")

    if include_kcor_lasco or include_lasco_only:
        print(
            f"       Earth-view pB = {earth_dir}"
        )

    if include_cor1a:
        print(
            f"       COR1A pB      = {cor1a_dir}"
        )

    if excluded_earth_datetimes:
        excluded_text = ", ".join(
            dt_value.strftime("%Y-%m-%d %H:%M:%S")
            for dt_value
            in sorted(excluded_earth_datetimes)
        )

        print(
            "[INFO] Earth-view-only exclusions:"
        )
        print(
            f"       times         = {excluded_text} UT"
        )
        print(
            "       keep matched COR1A = "
            f"{bool(keep_cor1a_for_excluded_earth_times)}"
        )

    print("[INFO] Selected pB files:")
    print(
        f"       Earth-view pB = {n_earth}"
    )
    print(
        f"       COR1A pB      = {n_cor1a}"
    )
    print(
        f"       Total         = "
        f"{len(unique_selected)}"
    )

    return [
        path
        for _, path in unique_selected
    ]
    
def _tomography_file_priority(path: Path) -> Tuple[int, str]:
    """
    Return priority information used when duplicate Earth-view pB constraints exist.

    Priority is intentionally applied only to files that represent the same Earth-view
    observing time. A K-Cor/LASCO merged file contains the LASCO constraint plus any
    available K-Cor inner-corona information, so it should replace the corresponding
    LASCO-only file rather than be used together with it. COR1A files are different
    viewpoints and are kept independently.
    """
    name = Path(path).name
    if re.fullmatch(r"pB_Kcor_LASCO_axi_\d{8}_\d{4}\.fits", name):
        return 30, "earth_merged"
    if re.fullmatch(r"pB_LASCO_C2_only_\d{8}_\d{4}\.fits", name):
        return 20, "earth_lasco_only"
    if re.fullmatch(r"COR1A_pb_pre_\d{8}_\d{6}\.fits", name):
        return 10, "cor1a"
    return 0, "other"


def deduplicate_tomography_pb_paths(paths: List[Path], verbose: bool = True) -> List[Path]:
    """
    Remove duplicate tomography-ready pB constraints.

    For the same Earth-view observing time, prefer:
      pB_Kcor_LASCO_axi_*  >  pB_LASCO_C2_only_*

    This prevents a single LASCO time from being counted twice when both a merged
    K-Cor/LASCO product and its LASCO-only fallback file are present. COR1A files are
    not considered duplicates of Earth-view files because they provide a different
    line of sight.
    """
    chosen: dict[tuple[str, datetime], Path] = {}
    kept: List[Path] = []
    dropped: List[Tuple[Path, Path]] = []

    for path in paths:
        path = Path(path)
        obs_dt = parse_pb_filename_datetime(path)
        if obs_dt is None:
            kept.append(path)
            continue

        priority, kind = _tomography_file_priority(path)
        if kind.startswith("earth"):
            key = ("earth", obs_dt)
            old = chosen.get(key)
            if old is None:
                chosen[key] = path
            else:
                old_priority, _ = _tomography_file_priority(old)
                if priority > old_priority:
                    chosen[key] = path
                    dropped.append((old, path))
                else:
                    dropped.append((path, old))
        else:
            # Keep non-Earth-view constraints independently. For COR1A this preserves
            # the additional viewpoint even when it is close in time to Earth-view pB.
            kept.append(path)

    kept.extend(chosen.values())
    kept.sort(key=lambda p: (parse_pb_filename_datetime(p) or datetime.min, Path(p).name))

    if verbose and dropped:
        print(f"[INFO] Removed duplicate Earth-view pB constraints: {len(dropped)}")
        for removed, retained in dropped:
            print(f"       drop {Path(removed).name}  -> keep {Path(retained).name}")

    return kept


def earth_view_camera_lonlat_from_target_time(
    target_time: str | datetime,
) -> Tuple[float, float]:
    """
    Return the Earth-view rendering camera lon/lat at target_time.

    The returned longitude is the apparent Carrington longitude of the solar
    disk center as seen from Earth, i.e. the sub-Earth Carrington longitude L0.
    The returned latitude is the apparent heliographic latitude of disk center B0.

    These values define the Sun-to-Earth direction in the Carrington frame and
    are used only for final rendering metadata and PNG camera placement.  They do
    not change the tomography inversion or the LOS geometry of individual input
    observations.
    """
    if target_time is None or str(target_time).strip() == "":
        raise ValueError(
            "target_time is required to force the final rendering camera to Earth view."
        )

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

    target_dt = parse_target_datetime(target_time)
    obstime = Time(target_dt, scale="utc")

    try:
        lon_deg = float(sun.L0(obstime).to_value(u.deg)) % 360.0
        lat_deg = float(sun.B0(obstime).to_value(u.deg))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to compute Earth-view Carrington lon/lat for target_time={target_time!r}."
        ) from exc

    if not (np.isfinite(lon_deg) and np.isfinite(lat_deg)):
        raise ValueError(
            f"Invalid Earth-view Carrington lon/lat computed for target_time={target_time!r}: "
            f"lon={lon_deg}, lat={lat_deg}"
        )

    return lon_deg, lat_deg


def choose_camera_lonlat_near_target(
    obs_list: List["Observation"],
    pb_paths: List[Path],
    target_time: str | datetime,
) -> Optional[Tuple[float, float]]:
    """
    Force the final visualization camera to the Earth-view direction at target_time.

    This function intentionally ignores the observation list and pB file times.
    COR1A observations are still used as tomography constraints, but they never
    determine the final rendered viewpoint.

    This affects only the rendered viewpoint, not the reconstructed 3-D density.
    """
    lonlat_deg = earth_view_camera_lonlat_from_target_time(target_time)
    target_dt = parse_target_datetime(target_time)

    print(
        "[INFO] Earth-view rendering camera forced from target_time: "
        f"time_utc={target_dt:%Y-%m-%d %H:%M:%S}, "
        f"Carrington_L0={lonlat_deg[0]:.6f} deg, "
        f"B0={lonlat_deg[1]:.6f} deg"
    )

    return lonlat_deg



# ----------------------------
# Plasma frequency conversions
# ----------------------------
def ne_cm3_from_fp_mhz(fp_mhz: float, harmonic: int = 1) -> float:
    """
    Convert plasma frequency (MHz) to electron density (cm^-3).
    f_pe[Hz] = 8980 * sqrt(ne[cm^-3]).
    If harmonic=2, input frequency is assumed to be 2 f_pe.
    """
    fp_hz = float(fp_mhz) * 1e6
    fpe = fp_hz / harmonic
    return (fpe / 8980.0) ** 2


def fp_mhz_from_ne_cm3(ne_cm3: float, harmonic: int = 1) -> float:
    fpe = 8980.0 * np.sqrt(float(ne_cm3))
    return harmonic * fpe / 1e6


def frequency_range_mhz_from_ne(ne: np.ndarray, harmonic: int = 1) -> Optional[Tuple[float, float, float, float]]:
    """
    Return (ne_min, ne_max, f_min, f_max) for positive finite electron density.

    The reconstructed scalar is electron density in cm^-3.  The displayed
    frequency range is therefore a diagnostic derived from that density, not a
    separate inversion variable.
    """
    arr = np.asarray(ne, dtype=np.float64)
    pos = np.isfinite(arr) & (arr > 0)
    if not np.any(pos):
        return None
    ne_min = float(np.min(arr[pos]))
    ne_max = float(np.max(arr[pos]))
    f_min = fp_mhz_from_ne_cm3(ne_min, harmonic=harmonic)
    f_max = fp_mhz_from_ne_cm3(ne_max, harmonic=harmonic)
    return ne_min, ne_max, min(f_min, f_max), max(f_min, f_max)


# ----------------------------
# Calibration / density-prior helpers
# ----------------------------
def normalize_group_float_map(value, name: str = "group_float_map") -> dict[str, float]:
    """
    Normalize a group->float mapping used for per-instrument settings.

    Accepted input:
      - dict, e.g. {"cor1a": 1.8}
      - empty string / None -> {}
      - comma-separated string, e.g. "cor1a:1.8,earth_merged:1.5"
    """
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k is None or str(k).strip() == "":
                continue
            out[str(k).strip()] = float(v)
        return out
    if isinstance(value, str):
        out = {}
        text = value.strip()
        if not text:
            return out
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item and "=" not in item:
                raise ValueError(f"Invalid {name} item {item!r}; use group:value")
            sep = ":" if ":" in item else "="
            k, v = item.split(sep, 1)
            out[k.strip()] = float(v)
        return out
    raise TypeError(f"{name} must be dict, str, None, or empty string; got {type(value)!r}")


def scale_observation_pb(obs: "Observation", scale: float) -> "Observation":
    """Return a copy of an Observation with pB and its noise weights scaled consistently."""
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"pB scale must be positive finite, got {scale}")
    return Observation(
        pb=np.asarray(obs.pb, dtype=np.float64) * scale,
        hdr=obs.hdr,
        x=obs.x,
        y=obs.y,
        mask=obs.mask,
        w=np.asarray(obs.w, dtype=np.float64) / scale,
        idx_map=obs.idx_map,
        cam_x=obs.cam_x,
        cam_y=obs.cam_y,
        cam_z=obs.cam_z,
        lonlat_deg=obs.lonlat_deg,
        limb_component_names=obs.limb_component_names,
        limb_u_values=obs.limb_u_values,
        limb_coeff_a=obs.limb_coeff_a,
        limb_coeff_b=obs.limb_coeff_b,
        limb_component_weight_maps=obs.limb_component_weight_maps,
        limb_u_model=obs.limb_u_model,
    )


def _saito_equatorial_ne_cm3(r_rsun: np.ndarray) -> np.ndarray:
    """Saito-like equatorial density model in cm^-3."""
    r = np.asarray(r_rsun, dtype=np.float64)
    r = np.maximum(r, 1.0001)
    return 1.0e8 * (3.09 * r**(-16.0) + 1.58 * r**(-6.0) + 0.0251 * r**(-2.5))


def _fitted_density_prior_ne_cm3_from_npz(
    r_rsun: np.ndarray,
    prior_file: Path | str,
) -> np.ndarray:
    """
    Evaluate the observation-derived fitted radial prior stored in an NPZ file.

    The generated prior NPZ stores the fitted line as ``radius_rsun`` and
    ``ne_prior_cm3``.  The table is interpolated in log(r)-log(ne) space so that
    the positive multiplicative density gradient is retained.  No Saito
    exponents are imposed.
    """
    path = Path(prior_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Observation-derived density-prior NPZ not found: {path}")

    with np.load(path, allow_pickle=False) as npz:
        required = ("radius_rsun", "ne_prior_cm3")
        missing = [key for key in required if key not in npz.files]
        if missing:
            raise KeyError(
                f"Density-prior NPZ {path} is missing required arrays: {missing}. "
                "Expected radius_rsun and ne_prior_cm3."
            )
        radius = np.asarray(npz["radius_rsun"], dtype=np.float64).ravel()
        density = np.asarray(npz["ne_prior_cm3"], dtype=np.float64).ravel()

    valid = (
        np.isfinite(radius)
        & np.isfinite(density)
        & (radius > 0)
        & (density > 0)
    )
    radius = radius[valid]
    density = density[valid]
    if radius.size < 2:
        raise ValueError(
            f"Density-prior NPZ {path} contains fewer than two valid positive samples."
        )

    order = np.argsort(radius)
    radius = radius[order]
    density = density[order]
    radius, unique_index = np.unique(radius, return_index=True)
    density = density[unique_index]

    query = np.asarray(r_rsun, dtype=np.float64)
    if np.any(~np.isfinite(query)) or np.any(query <= 0):
        raise ValueError("Tomography-grid radii must be positive and finite.")

    tolerance = 1.0e-8
    if float(np.nanmin(query)) < float(radius[0]) - tolerance:
        raise ValueError(
            f"Tomography radius {float(np.nanmin(query)):.6g} Rsun is below the "
            f"observation-derived prior range {radius[0]:.6g}..{radius[-1]:.6g} Rsun."
        )
    if float(np.nanmax(query)) > float(radius[-1]) + tolerance:
        raise ValueError(
            f"Tomography radius {float(np.nanmax(query)):.6g} Rsun is above the "
            f"observation-derived prior range {radius[0]:.6g}..{radius[-1]:.6g} Rsun."
        )

    log_density = np.interp(
        np.log(query),
        np.log(radius),
        np.log(density),
    )
    return np.exp(log_density)


def density_basis_from_grid(
    grid: "SphericalGrid",
    model: str = "none",
    scale: float = 1.0,
    prior_file: Path | str = "",
) -> Optional[np.ndarray]:
    """
    Build a positive density prior/basis for solving ne = basis * q.

    model='none' disables this behavior and solves absolute electron density ne.
    model='saito_equatorial' uses a Saito-like radial density profile.
    model='fitted_pchip_npz' uses the observation-derived fitted line stored in
    ``prior_file`` without imposing the Saito radial exponents.
    """
    model = str(model or "none").strip().lower()
    if model in ("", "none", "off", "false", "0"):
        return None

    rr, _, _ = grid.voxel_centers_sph()
    if model in ("saito", "saito_equatorial", "saito-equatorial"):
        basis = float(scale) * _saito_equatorial_ne_cm3(rr)
    elif model in (
        "fitted_pchip_npz",
        "observation_fitted_pchip",
        "observation_npz",
    ):
        if not str(prior_file).strip():
            raise ValueError(
                f"density_prior_file is required for density_prior_model={model!r}."
            )
        basis = float(scale) * _fitted_density_prior_ne_cm3_from_npz(
            rr,
            prior_file=prior_file,
        )
    else:
        raise ValueError(f"Unknown density prior model: {model!r}")

    basis = np.asarray(basis, dtype=np.float64).ravel(order="C")
    if np.any(~np.isfinite(basis)) or np.any(basis <= 0):
        raise ValueError(
            f"Density prior {model!r} produced non-positive or non-finite values."
        )
    return basis


def weighted_projection_scale(y_obs, y_pred, W, min_count: int = 100) -> float:
    """Weighted least-squares scalar s minimizing ||W*(s*y_pred-y_obs)||."""
    y_obs = np.asarray(y_obs, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    W = np.asarray(W, dtype=np.float64).ravel()
    m = np.isfinite(y_obs) & np.isfinite(y_pred) & np.isfinite(W) & (y_pred != 0)
    if np.count_nonzero(m) < int(min_count):
        return 1.0
    w2 = W[m] * W[m]
    den = float(np.sum(w2 * y_pred[m] * y_pred[m]))
    if den <= 0 or not np.isfinite(den):
        return 1.0
    num = float(np.sum(w2 * y_pred[m] * y_obs[m]))
    s = num / den
    if not np.isfinite(s) or s <= 0:
        return 1.0
    return float(s)


def _weighted_projection_stats(y_obs, y_pred, W, scale_for_residual: float = 1.0):
    y_obs = np.asarray(y_obs, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel() * float(scale_for_residual)
    W = np.asarray(W, dtype=np.float64).ravel()
    m = np.isfinite(y_obs) & np.isfinite(y_pred) & np.isfinite(W)
    if not np.any(m):
        return None
    yo = y_obs[m]
    yp = y_pred[m]
    ww = W[m]
    denom = np.maximum(np.abs(yo), 1e-30)
    rms_rel = float(np.sqrt(np.sum((ww * (yp - yo) / denom) ** 2) / max(1, np.sum(ww > 0))))
    good_ratio = np.isfinite(yo) & np.isfinite(yp) & (np.abs(yp) > 0)
    med_ratio = float(np.nanmedian(yo[good_ratio] / yp[good_ratio])) if np.any(good_ratio) else np.nan
    med_obs = float(np.nanmedian(yo)) if yo.size else np.nan
    return {"n": int(yo.size), "weighted_rms_rel": rms_rel, "median_obs_over_pred": med_ratio, "median_obs": med_obs}


def print_projection_fit_diagnostic(label: str, y_obs, y_pred, W, scale_for_residual: float = 1.0):
    stats = _weighted_projection_stats(y_obs, y_pred, W, scale_for_residual=scale_for_residual)
    fit_scale = weighted_projection_scale(y_obs, y_pred, W, min_count=100)
    if stats is None:
        print(f"[DIAG] projection_fit {label}: no usable points")
        return
    print(
        f"[DIAG] projection_fit {label}: "
        f"n={stats['n']}, fit_scale={fit_scale:.6g}, eval_scale={float(scale_for_residual):.6g}, "
        f"weighted_rms_rel={stats['weighted_rms_rel']:.4g}, "
        f"median_obs/eval_pred={stats['median_obs_over_pred']:.6g}, "
        f"median_obs={stats['median_obs']:.3e}"
    )


def print_projection_fit_diagnostics_by_group(
    label: str,
    pb_paths: List[Path],
    tomo: "RegularizedTomography",
    y_obs,
    y_pred,
    W,
    scale_for_residual: float = 1.0,
):
    groups: dict[str, List[int]] = {}
    for i, p in enumerate(pb_paths):
        groups.setdefault(tomography_observation_group_key(Path(p)), []).append(i)
    for key, indices in groups.items():
        chunks_obs = []
        chunks_pred = []
        chunks_w = []
        for i in indices:
            sl = tomo.slices[i]
            chunks_obs.append(np.asarray(y_obs)[sl])
            chunks_pred.append(np.asarray(y_pred)[sl])
            chunks_w.append(np.asarray(W)[sl])
        if not chunks_obs:
            continue
        print_projection_fit_diagnostic(
            f"{label}/{key}",
            np.concatenate(chunks_obs),
            np.concatenate(chunks_pred),
            np.concatenate(chunks_w),
            scale_for_residual=scale_for_residual,
        )


def print_group_calibration_hints(
    label: str,
    pb_paths: List[Path],
    tomo: "RegularizedTomography",
    y_obs,
    y_pred,
    W,
    reference_group: str = "earth_merged",
):
    groups: dict[str, List[int]] = {}
    scales: dict[str, float] = {}
    for i, p in enumerate(pb_paths):
        groups.setdefault(tomography_observation_group_key(Path(p)), []).append(i)
    for key, indices in groups.items():
        yo = []
        yp = []
        ww = []
        for i in indices:
            sl = tomo.slices[i]
            yo.append(np.asarray(y_obs)[sl])
            yp.append(np.asarray(y_pred)[sl])
            ww.append(np.asarray(W)[sl])
        if yo:
            scales[key] = weighted_projection_scale(np.concatenate(yo), np.concatenate(yp), np.concatenate(ww), min_count=100)
    if not scales:
        return
    ref = scales.get(str(reference_group), None)
    print(f"[DIAG] calibration_hint {label}: group fit scales = {scales}")
    if ref is not None and ref > 0:
        rel = {k: v / ref for k, v in scales.items()}
        print(f"[DIAG] calibration_hint {label}: relative to {reference_group!r} = {rel}")


def robust_weighted_projection_scale(
    y_obs,
    y_pred,
    W,
    min_count: int = 100,
    clip_sigma: float = 4.0,
    max_clip_iter: int = 2,
) -> Tuple[float, int]:
    """
    Estimate a multiplicative forward-model gain robustly.

    The returned scalar g minimizes ||W * (g*y_pred - y_obs)|| after iterative
    clipping in weighted-residual space.  This is used only for inter-instrument
    relative calibration; the reference instrument remains fixed at gain 1.
    """
    yo = np.asarray(y_obs, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    ww = np.asarray(W, dtype=np.float64).ravel()

    use = (
        np.isfinite(yo)
        & np.isfinite(yp)
        & np.isfinite(ww)
        & (ww > 0)
        & (yp != 0)
    )
    if np.count_nonzero(use) < int(min_count):
        return 1.0, int(np.count_nonzero(use))

    for _ in range(max(0, int(max_clip_iter)) + 1):
        w2 = ww[use] * ww[use]
        den = float(np.sum(w2 * yp[use] * yp[use]))
        if not np.isfinite(den) or den <= 0:
            return 1.0, int(np.count_nonzero(use))
        num = float(np.sum(w2 * yp[use] * yo[use]))
        gain = num / den
        if not np.isfinite(gain) or gain <= 0:
            return 1.0, int(np.count_nonzero(use))

        resid = ww * (gain * yp - yo)
        rr = resid[use]
        if rr.size < int(min_count) or not np.isfinite(float(clip_sigma)) or float(clip_sigma) <= 0:
            break
        med = float(np.nanmedian(rr))
        mad = float(np.nanmedian(np.abs(rr - med)))
        sig = 1.4826 * mad
        if not np.isfinite(sig) or sig <= 0:
            break
        new_use = use & (np.abs(resid - med) <= float(clip_sigma) * sig)
        if np.count_nonzero(new_use) < int(min_count) or np.array_equal(new_use, use):
            break
        use = new_use

    return float(gain), int(np.count_nonzero(use))


def _optional_positive_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        out = float(value)
        if not np.isfinite(out) or out <= 0:
            return None
        return out
    except Exception:
        return None


def build_group_measurement_scale_vector(
    pb_paths: List[Path],
    tomo: "RegularizedTomography",
    group_forward_gains: dict[str, float],
) -> np.ndarray:
    """
    Build one row-wise multiplicative response vector for the forward operator.

    For group g, the observation model is y_g = gain_g * A_g x + epsilon_g.
    The reference group must therefore have gain 1.  This changes neither the
    selected Earth/STEREO-A observations nor the ray geometry.
    """
    if len(pb_paths) != len(tomo.slices):
        raise ValueError(
            f"pb_paths/tomography slice mismatch: {len(pb_paths)} != {len(tomo.slices)}"
        )
    scale = np.ones(tomo.n_meas, dtype=np.float64)
    for path, sl in zip(pb_paths, tomo.slices):
        key = tomography_observation_group_key(Path(path))
        gain = float(group_forward_gains.get(key, 1.0))
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError(f"Invalid forward gain for group {key!r}: {gain}")
        scale[sl] = gain
    return scale


def estimate_relative_group_forward_gains(
    pb_paths: List[Path],
    tomo: "RegularizedTomography",
    y_obs: np.ndarray,
    y_pred_unscaled: np.ndarray,
    W: np.ndarray,
    reference_group: str,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    min_count: int = 100,
    clip_sigma: float = 4.0,
) -> Tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Estimate group gains relative to a fixed reference group.

    A robust gain is first fitted for every image.  The group representative is
    the median of its per-image gains, preventing a group with more images from
    dominating the calibration.  Dividing by the reference-group representative
    removes a global density-amplitude bias caused by regularization.
    """
    y_obs = np.asarray(y_obs, dtype=np.float64).ravel()
    y_pred_unscaled = np.asarray(y_pred_unscaled, dtype=np.float64).ravel()
    W = np.asarray(W, dtype=np.float64).ravel()

    per_group: dict[str, list[float]] = {}
    per_group_used: dict[str, list[int]] = {}

    for path, obs, sl in zip(pb_paths, tomo.observations, tomo.slices):
        key = tomography_observation_group_key(Path(path))
        yo = y_obs[sl]
        yp = y_pred_unscaled[sl]
        ww = W[sl]

        rho_use = np.hypot(
            obs.x.ravel()[obs.idx_map],
            obs.y.ravel()[obs.idx_map],
        )
        radial = np.isfinite(rho_use)
        if r_min is not None:
            radial &= rho_use >= float(r_min)
        if r_max is not None:
            radial &= rho_use <= float(r_max)

        gain_i, n_used = robust_weighted_projection_scale(
            yo[radial],
            yp[radial],
            ww[radial],
            min_count=int(min_count),
            clip_sigma=float(clip_sigma),
            max_clip_iter=2,
        )
        if np.isfinite(gain_i) and gain_i > 0 and n_used >= int(min_count):
            per_group.setdefault(key, []).append(float(gain_i))
            per_group_used.setdefault(key, []).append(int(n_used))

    group_fit: dict[str, float] = {}
    stats: dict[str, dict[str, float]] = {}
    for key, values in per_group.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            continue
        q16, q50, q84 = np.nanpercentile(arr, [16.0, 50.0, 84.0])
        group_fit[key] = float(q50)
        stats[key] = {
            "n_images": float(arr.size),
            "fit_scale_median": float(q50),
            "fit_scale_p16": float(q16),
            "fit_scale_p84": float(q84),
            "median_used_pixels": float(np.nanmedian(per_group_used.get(key, [0]))),
        }

    ref = group_fit.get(str(reference_group))
    if ref is None or not np.isfinite(ref) or ref <= 0:
        raise ValueError(
            f"Cannot estimate inter-instrument calibration: reference group "
            f"{reference_group!r} has no valid per-image fit scale."
        )

    relative = {
        key: float(value / ref)
        for key, value in group_fit.items()
        if np.isfinite(value) and value > 0
    }
    relative[str(reference_group)] = 1.0
    return relative, stats


def run_group_cross_calibration(
    tomo: "RegularizedTomography",
    pb_paths: List[Path],
    y_obs: np.ndarray,
    reference_group: str = "earth_merged",
    initial_group_forward_gains=None,
    max_iterations: int = 3,
    convergence_tol: float = 0.01,
    damping: float = 0.7,
    gain_min: float = 0.25,
    gain_max: float = 4.0,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    min_count: int = 100,
    clip_sigma: float = 4.0,
    solve_maxiter: int = 10000,
    solve_tol: float = 1e-3,
    positivity_method: str = "clip",
    solve_use_preconditioner: bool = True,
    solve_preconditioner_floor: float = 1e-12,
) -> Tuple[dict[str, float], list[dict], Optional[np.ndarray]]:
    """
    Alternately solve the density and relative instrument gains.

    Earth-view and STEREO-A observations are both retained.  Only one scalar
    response gain per observation group is fitted, with the Earth-view reference
    fixed at 1 to remove the global scale degeneracy.  The reciprocal of a fitted
    forward gain is the multiplicative correction that would place that group's
    observed pB on the reference-group brightness scale.
    """
    group_names = sorted(
        {tomography_observation_group_key(Path(path)) for path in pb_paths}
    )
    reference_group = str(reference_group)
    if reference_group not in group_names:
        raise ValueError(
            f"Cross-calibration reference group {reference_group!r} is absent; "
            f"available groups={group_names}"
        )
    if len(group_names) < 2:
        print("[CAL] Cross-calibration skipped because only one observation group is present.")
        gains = {group_names[0]: 1.0} if group_names else {}
        tomo.set_measurement_scale(
            build_group_measurement_scale_vector(pb_paths, tomo, gains)
        )
        return gains, [], None

    initial = normalize_group_float_map(
        initial_group_forward_gains,
        "cross_calibration_initial_gain_by_group",
    )
    gains = {key: float(initial.get(key, 1.0)) for key in group_names}
    gains[reference_group] = 1.0

    gain_min = float(gain_min)
    gain_max = float(gain_max)
    if not (np.isfinite(gain_min) and np.isfinite(gain_max) and 0 < gain_min < gain_max):
        raise ValueError(
            f"Invalid cross-calibration gain bounds: {gain_min}, {gain_max}"
        )
    for key in group_names:
        gains[key] = float(np.clip(gains[key], gain_min, gain_max))
    gains[reference_group] = 1.0
    damping = float(np.clip(float(damping), 0.0, 1.0))
    max_iterations = max(1, int(max_iterations))

    history: list[dict] = []
    x_seed: Optional[np.ndarray] = None

    print(
        "[CAL] Automatic inter-instrument calibration enabled: "
        f"reference={reference_group!r}, groups={group_names}, "
        f"gain_bounds=({gain_min:.4g}, {gain_max:.4g}), damping={damping:.3g}"
    )
    if r_min is not None or r_max is not None:
        print(f"[CAL] Calibration radial range: {r_min} .. {r_max} Rsun")

    for iteration in range(1, max_iterations + 1):
        tomo.set_measurement_scale(
            build_group_measurement_scale_vector(pb_paths, tomo, gains)
        )
        solution, info = tomo.solve(
            y_obs,
            maxiter=int(solve_maxiter),
            tol=float(solve_tol),
            positivity=True,
            positivity_method=str(positivity_method),
            x0=x_seed,
            use_preconditioner=bool(solve_use_preconditioner),
            preconditioner_floor=float(solve_preconditioner_floor),
        )
        x_seed = solution

        base_pred = tomo.A_times_unscaled(solution)
        desired, stats = estimate_relative_group_forward_gains(
            pb_paths=pb_paths,
            tomo=tomo,
            y_obs=y_obs,
            y_pred_unscaled=base_pred,
            W=tomo.W,
            reference_group=reference_group,
            r_min=r_min,
            r_max=r_max,
            min_count=int(min_count),
            clip_sigma=float(clip_sigma),
        )

        updated = dict(gains)
        max_log_change = 0.0
        for key in group_names:
            if key == reference_group:
                updated[key] = 1.0
                continue
            target = float(desired.get(key, gains[key]))
            target = float(np.clip(target, gain_min, gain_max))
            old = float(np.clip(gains[key], gain_min, gain_max))
            if damping <= 0:
                new = old
            else:
                new = float(np.exp((1.0 - damping) * np.log(old) + damping * np.log(target)))
            new = float(np.clip(new, gain_min, gain_max))
            updated[key] = new
            max_log_change = max(max_log_change, abs(float(np.log(new / old))))
            max_log_change_percent = 100.0 * np.expm1(max_log_change)

        row = {
            "iteration": int(iteration),
            "solver_info": int(info),
            "max_abs_log_gain_change": float(max_log_change),
            "max_log_gain_change_percent": float(max_log_change_percent),
            "gains": dict(updated),
            "desired_relative_gains": dict(desired),
            "per_group_stats": stats,
        }
        history.append(row)

        correction = {key: 1.0 / value for key, value in updated.items()}
        print(
            f"[CAL] iteration={iteration}: forward_gains={updated}, "
            f"data_corrections={correction}, "
            f"max_abs_log_change={max_log_change:.4g}, max_log_change_equivalent_percent={max_log_change_percent:.4g} ,solver_info={info}"
        )
        gains = updated

        if max_log_change <= float(convergence_tol):
            print(
                f"[CAL] Cross-calibration converged at iteration {iteration}: "
                f"tolerance={float(convergence_tol):.4g}"
            )
            break

    tomo.set_measurement_scale(
        build_group_measurement_scale_vector(pb_paths, tomo, gains)
    )
    return gains, history, x_seed


# ----------------------------
# Thomson pB kernel
# ----------------------------
def thomsonscatter_pB_per_electron(impact_rsun: float, theta_from_pos_rad: float,
                                  u: float = DEFAULT_LIMB_U,
                                  normalize_msb: bool = True,
                                  kernel_scale: float = 1.0,
                                  limb_coefficients: Optional[Tuple[float, float]] = None) -> float:
    """
    Polarized brightness contribution per single electron at a given position along a LOS,
    using the SSC/van-de-Hulst kernel form.

    Parameters
    ----------
    normalize_msb : bool
        If True, multiply by 1/(1-u/3), appropriate when the input pB is in
        mean-solar-brightness-like units (e.g. MSB / Mean Solar Brightness).
        If an upstream SSC/IDL calibration has already absorbed this factor, set
        this to False and document that choice.
    kernel_scale : float
        Explicit final multiplicative factor for controlled unit-calibration tests.
        Keep at 1.0 unless a documented photometric conversion requires otherwise.

    Notes
    -----
    The original closed-form for B involves log((1+sinw)/cosw) and can produce 0*inf -> NaN
    numerically when cosw -> 0. We therefore regularize cosw and the log argument to keep
    the expression finite and avoid contaminating the forward operator with NaNs/Infs.
    """
    sinchi = float(np.cos(theta_from_pos_rad))
    if not np.isfinite(sinchi):
        return 0.0
    if impact_rsun <= 0:
        return 0.0

    # SSC-style definition
    sinw = sinchi / float(impact_rsun)
    sinw = float(np.clip(sinw, 0.0, 1.0))

    # Guard for numerical edge (sinw ~ 1 -> cosw ~ 0)
    cosw = float(np.sqrt(max(0.0, 1.0 - sinw * sinw)))
    cosw_safe = max(cosw, 1e-12)

    # SSC A,B
    A = cosw * (sinw ** 2)

    if sinw <= 0.0:
        B = 0.0
    else:
        # Ensure log argument is positive and finite
        arg = (1.0 + sinw) / cosw_safe
        arg = max(arg, 1.0 + 1e-12)
        logterm = float(np.log(arg))

        # Use cosw_safe to avoid 0*inf numerical NaNs
        term = (cosw_safe ** 2) / sinw * (1.0 + 3.0 * (sinw ** 2)) * logterm
        B = -0.125 * (1.0 - 3.0 * (sinw ** 2) - term)

    if limb_coefficients is None:
        coeff_a, coeff_b = limb_kernel_coefficients_from_u(
            float(u), normalize_msb=bool(normalize_msb)
        )
    else:
        coeff_a, coeff_b = (float(v) for v in limb_coefficients)
        if not (np.isfinite(coeff_a) and np.isfinite(coeff_b)):
            raise ValueError(f"Invalid limb kernel coefficients: {limb_coefficients}")

    pB = (3.0 / 16.0) * ICEN * SIGMA_T * (sinchi ** 2) * (coeff_a * A + coeff_b * B)
    pB *= float(kernel_scale)

    # Final safety: do not allow NaN/Inf to leak
    if not np.isfinite(pB):
        return 0.0
    return float(pB)


# ----------------------------
# FITS + WCS helpers
# ----------------------------
def read_fits_image(path: Path) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64)
        hdr = hdul[0].header
    return data, hdr


def summarize_pb_fits_calibration(path: Path) -> dict:
    """Return a compact photometric/unit summary for a pB-related FITS file.

    This is a diagnostic only.  It does not change the inversion data.  It is
    intended to prevent accidental mixing of raw DN images with calibrated pB/MSB
    products and to make the absolute-kernel normalization choice auditable.
    """
    data, hdr = read_fits_image(Path(path))
    arr = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(arr)
    v = arr[finite]
    pos = v[v > 0] if v.size else np.array([], dtype=np.float64)

    def _pct(a, q):
        return float(np.nanpercentile(a, q)) if a.size else np.nan

    bunit = str(hdr.get("BUNIT", "")).strip()
    unit_text = bunit.lower()
    is_raw_dn = unit_text in ("dn", "adu", "counts") or "dn" == unit_text
    is_msb_like = (
        "msb" in unit_text
        or "mean solar brightness" in unit_text
        or bunit == "pB"
    )

    return {
        "path": str(Path(path)),
        "name": Path(path).name,
        "shape": tuple(arr.shape),
        "instrument": str(hdr.get("INSTRUME", "")),
        "detector": str(hdr.get("DETECTOR", "")),
        "telescope": str(hdr.get("TELESCOP", "")),
        "observatory": str(hdr.get("OBSRVTRY", "")),
        "date_obs": str(hdr.get("DATE-OBS", hdr.get("DATE_OBS", ""))),
        "bunit": bunit,
        "is_raw_dn": bool(is_raw_dn),
        "is_msb_like": bool(is_msb_like),
        "finite_count": int(v.size),
        "min": float(np.nanmin(v)) if v.size else np.nan,
        "median": float(np.nanmedian(v)) if v.size else np.nan,
        "max": float(np.nanmax(v)) if v.size else np.nan,
        "p01": _pct(v, 1),
        "p50": _pct(v, 50),
        "p99": _pct(v, 99),
        "positive_p50": _pct(pos, 50),
        "positive_p99": _pct(pos, 99),
    }


def print_pb_calibration_report(paths: List[Path]) -> None:
    """Print pB/DN unit and scale diagnostics for selected FITS files."""
    if not paths:
        return
    print("[CAL] Input pB/DN calibration diagnostics:")
    for path in paths:
        try:
            s = summarize_pb_fits_calibration(Path(path))
        except Exception as exc:
            print(f"[CAL] {Path(path).name}: failed to read ({exc})")
            continue
        flags = []
        if s["is_raw_dn"]:
            flags.append("RAW_DN_NOT_FOR_TOMO")
        if s["is_msb_like"]:
            flags.append("MSB_LIKE")
        flag_text = ",".join(flags) if flags else "unit_unknown"
        print(
            f"[CAL] {s['name']}: BUNIT={s['bunit']!r}, {flag_text}, "
            f"instrument={s['instrument']}/{s['detector']}, shape={s['shape']}, "
            f"median={s['median']:.3e}, p99={s['p99']:.3e}, "
            f"positive_median={s['positive_p50']:.3e}"
        )


def block_reduce_mean(img: np.ndarray, out_n: int) -> np.ndarray:
    """
    Downsample by block averaging to out_n x out_n.
    Requires img be square and divisible by out_n.
    """
    n = img.shape[0]
    if img.shape[0] != img.shape[1]:
        raise ValueError(f"Expected square image, got {img.shape}")
    if n % out_n != 0:
        raise ValueError(f"Image size {n} not divisible by out_n={out_n}")

    f = n // out_n
    return img.reshape(out_n, f, out_n, f).mean(axis=(1, 3))


def _rsun_arcsec_from_header(hdr: fits.Header) -> float:
    for k in ("RSUN", "RSUN_OBS"):
        if k in hdr and np.isfinite(hdr[k]):
            return float(hdr[k])
    raise ValueError("RSUN/RSUN_OBS not found in FITS header.")


def xy_rsun_for_rebinned_image(hdr, orig_n: int, out_n: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Return helioprojective x/y coordinate maps for the *rebinned* image in **Rsun units**.

    This codebase frequently uses r_use_min/r_use_max in Rsun. Therefore x_map and y_map must be
    expressed as x/RSUN_OBS and y/RSUN_OBS (dimensionless solar radii), not in arcsec.

    Important
    ---------
    For COR1A HPLN-TAN/HPLT-TAN headers, astropy.wcs.pixel_to_world_values can return
    negative helioprojective longitudes as wrapped values near 360 deg.  If those values are
    converted directly to arcsec, a normal COR1A field of view can appear to extend to
    ~10^3 Rsun.  For tomography, the required coordinates are local image-plane offsets from
    Sun center, so this routine uses the FITS linear image-plane transformation
    (CRPIX/CRVAL/CDELT with PC or CD matrix) and avoids longitude wrapping.

    Returns
    -------
    x_map_rsun, y_map_rsun : (out_n,out_n)
        Helioprojective coordinates in Rsun (dimensionless), centered at Sun center.
    rsun_arcsec : float
        Apparent solar radius in arcsec (RSUN_OBS/RSUN keyword).
    """
    rsun_arcsec = float(hdr.get("RSUN_OBS", hdr.get("RSUN", 959.63)))

    # scale factor from original -> rebinned pixels
    s = float(orig_n) / float(out_n)

    # Pixel centers on rebinned grid, expressed in the original FITS pixel system.
    yy, xx = np.mgrid[0:out_n, 0:out_n]
    xpix = (xx + 0.5) * s - 0.5
    ypix = (yy + 0.5) * s - 0.5

    def _cd_matrix_from_header(h: fits.Header) -> np.ndarray:
        if all(k in h for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
            return np.array(
                [[float(h["CD1_1"]), float(h["CD1_2"])],
                 [float(h["CD2_1"]), float(h["CD2_2"])]],
                dtype=np.float64,
            )
        if all(k in h for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2")):
            pc = np.array(
                [[float(h["PC1_1"]), float(h["PC1_2"])],
                 [float(h["PC2_1"]), float(h["PC2_2"])]],
                dtype=np.float64,
            )
            cdelt1 = float(h.get("CDELT1", 1.0))
            cdelt2 = float(h.get("CDELT2", 1.0))
            return np.diag([cdelt1, cdelt2]) @ pc

        crota = float(h.get("CROTA2", h.get("CROTA", 0.0)))
        cdelt1 = float(h.get("CDELT1", 1.0))
        cdelt2 = float(h.get("CDELT2", 1.0))
        th = np.deg2rad(crota)
        rot = np.array([[np.cos(th), -np.sin(th)],
                        [np.sin(th),  np.cos(th)]], dtype=np.float64)
        return np.diag([cdelt1, cdelt2]) @ rot

    cd = _cd_matrix_from_header(hdr)

    # FITS CRPIX is 1-based.  xpix/ypix are 0-based pixel coordinates in the original image.
    crpix1 = float(hdr.get("CRPIX1", (orig_n + 1) / 2.0))
    crpix2 = float(hdr.get("CRPIX2", (orig_n + 1) / 2.0))
    crval1 = float(hdr.get("CRVAL1", 0.0))
    crval2 = float(hdr.get("CRVAL2", 0.0))

    dx = xpix + 1.0 - crpix1
    dy = ypix + 1.0 - crpix2

    x_map = crval1 + cd[0, 0] * dx + cd[0, 1] * dy
    y_map = crval2 + cd[1, 0] * dx + cd[1, 1] * dy

    cunit1 = str(hdr.get("CUNIT1", "")).lower()
    if "deg" in cunit1:
        x_map *= 3600.0
        y_map *= 3600.0
    elif "arcmin" in cunit1:
        x_map *= 60.0
        y_map *= 60.0

    # ---- convert arcsec -> Rsun ----
    if not np.isfinite(rsun_arcsec) or rsun_arcsec <= 0:
        rsun_arcsec = 959.63
    x_map_rsun = x_map / rsun_arcsec
    y_map_rsun = y_map / rsun_arcsec

    return x_map_rsun.astype(np.float64), y_map_rsun.astype(np.float64), rsun_arcsec

def infer_carrington_lonlat_deg(hdr: fits.Header) -> Optional[Tuple[float, float]]:
    """
    Prefer Carrington observer longitude/latitude from header (CRLN_OBS/CRLT_OBS).
    As a fallback, use HGLN_OBS/HGLT_OBS if present (often Stonyhurst/heliographic).
    """
    for lon_k, lat_k in (("CRLN_OBS", "CRLT_OBS"), ("HGLN_OBS", "HGLT_OBS")):
        if lon_k in hdr and lat_k in hdr:
            try:
                lon = float(hdr[lon_k])
                lat = float(hdr[lat_k])
                if np.isfinite(lon) and np.isfinite(lat):
                    return lon, lat
            except Exception:
                pass
    return None


def camera_basis_from_lonlat(lon_deg: float, lat_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (x_hat, y_hat, z_hat) basis vectors in Carrington Cartesian coordinates:
      - z_hat points from Sun to observer (sub-observer direction).
      - y_hat is solar north projected onto the POS.
      - x_hat completes right-handed basis.
    """
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)

    z_hat = np.array([np.cos(lat) * np.cos(lon),
                      np.cos(lat) * np.sin(lon),
                      np.sin(lat)], dtype=np.float64)
    z_hat /= np.linalg.norm(z_hat)

    north = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_hat = north - np.dot(north, z_hat) * z_hat
    yn = np.linalg.norm(y_hat)
    if yn < 1e-8:
        # Observer near pole: choose arbitrary y in POS
        y_hat = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        y_hat /= yn

    x_hat = np.cross(y_hat, z_hat)
    x_hat /= np.linalg.norm(x_hat)

    return x_hat, y_hat, z_hat


# ----------------------------
# "fix_nan.pro" analog
# ----------------------------
def fill_nan_by_neighbor_mean(img: np.ndarray, max_passes: int = 1) -> np.ndarray:
    """Fill NaNs using the same single-pass 4-neighbor rule as fix_nan.pro.

    The IDL routine fix_nan.pro does not iteratively diffuse values into extended
    NaN regions. It loops over the NaN pixels in the original image and replaces
    only interior NaNs whose four direct neighbors include at least one finite
    value. Boundary NaNs and connected NaN regions without finite 4-neighbors are
    left as NaN here instead of being extrapolated, which avoids inventing pB
    structure before the tomography step.

    Parameters
    ----------
    img : np.ndarray
        2D image.
    max_passes : int
        Kept for backward compatibility. Values larger than one are intentionally
        ignored in Ne3dTomo-compatible mode.
    """
    if img.ndim != 2:
        raise ValueError("fill_nan_by_neighbor_mean expects a 2D array")

    image = img.astype(np.float64, copy=True)
    out = image.copy()
    ny, nx = image.shape

    nan_y, nan_x = np.where(~np.isfinite(image))
    for y, x in zip(nan_y, nan_x):
        # IDL fix_nan.pro treats only interior pixels.
        if x <= 0 or x >= nx - 1 or y <= 0 or y >= ny - 1:
            continue
        neigh = np.array(
            [image[y, x - 1], image[y, x + 1], image[y - 1, x], image[y + 1, x]],
            dtype=np.float64,
        )
        good = np.isfinite(neigh)
        if np.any(good):
            out[y, x] = float(np.mean(neigh[good]))

    return out


# ----------------------------
# "pbmap_despike.pro" analog
# ----------------------------
def despike_pb_map(
    pb: np.ndarray,
    mask: np.ndarray,
    med_size: int = 3,
    nsig: float = 6.0,
    use_log: bool = True,
) -> np.ndarray:
    """
    Robust despike: compare to local median and replace high outliers.
    Implemented on log10(pB) by default to stabilize multiplicative spikes.
    """
    out = pb.copy()

    work = out.copy()
    work[~mask] = np.nan
    if use_log:
        # Avoid evaluating log10 on non-positive values. np.where would still evaluate both branches.
        tmp = np.full_like(work, np.nan, dtype=np.float64)
        good_log = np.isfinite(work) & (work > 0)
        tmp[good_log] = np.log10(work[good_log])
        work = tmp

    med = median_filter(np.nan_to_num(work, nan=np.nanmedian(work[np.isfinite(work)])), size=med_size)
    resid = work - med

    # Robust sigma (MAD)
    rr = resid[np.isfinite(resid)]
    if rr.size < 100:
        return out
    mad = np.median(np.abs(rr - np.median(rr)))
    sig = 1.4826 * mad if mad > 0 else np.std(rr)
    if not np.isfinite(sig) or sig <= 0:
        return out

    bad = mask & np.isfinite(resid) & (resid > nsig * sig)
    if not np.any(bad):
        return out

    # Replace with median in the same domain
    rep = med[bad]
    if use_log:
        out[bad] = 10.0 ** rep
    else:
        out[bad] = rep
    return out




def despike_pb_cube(
    pbs: np.ndarray,
    nsig: float = 6.0,
    use_log: bool = False,
) -> np.ndarray:
    """Temporal despike following the core algorithm of ssw_unspike_cube.pro.

    ssw_unspike_cube detects a spike by comparing the current pixel to a local
    temporal prediction built from the previous and following images. The
    simultaneous 3x3 spatial neighbors are deliberately not used. This function
    ports that scientifically relevant behavior:

      - For each image iz, choose neighbor images iz-1 and iz+1; at the first and
        last frames use the same asymmetric neighbor choices as the IDL routine.
      - For each row, form 3x3 spatial neighborhoods in the previous and next
        frames using IDL SHIFT-like circular shifts along the x direction.
      - Compute zav8 from the 2 x 3x3 non-simultaneous neighborhoods and zav1
        from the two temporal same-pixel neighbors.
      - If zav1 is itself anomalously high relative to zav8, use zav8; otherwise
        use zav1 as the replacement/prediction.
      - Replace only high spikes in the current frame.

    Parameters
    ----------
    pbs : ndarray
        Cube with shape (nt, ny, nx).
    nsig : float
        Threshold factor. Ne3dTomo preview_data.pro uses threshold=6.0.
    use_log : bool
        Kept for API compatibility; ignored because ssw_unspike_cube works in the
        native data domain, not log space.
    """
    data = np.asarray(pbs, dtype=np.float64)
    out = data.copy()
    if data.ndim != 3:
        raise ValueError("despike_pb_cube expects a cube with shape (nt, ny, nx)")

    nt, ny, nx = data.shape
    # The IDL header states NTIMES > 3. For shorter groups, avoid applying a
    # temporal filter with ill-defined edge neighbor choices.
    if nt <= 3 or ny < 3 or nx < 3:
        return out

    threshold = float(nsig) if np.isfinite(nsig) and nsig > 0 else 6.0

    def _shift_x(row: np.ndarray, shift: int) -> np.ndarray:
        # IDL SHIFT(array,-1) moves elements left with circular wrap; np.roll
        # uses the same sign convention for positive right shifts, so this is direct.
        return np.roll(row, shift)

    for iz in range(nt):
        iz1 = iz - 1
        if iz == 0:
            iz1 = iz + 2
        iz2 = iz + 1
        if iz == nt - 1:
            iz2 = iz - 3

        for j in range(ny):
            j1 = j - 1
            if j == 0:
                j1 = j + 2
            j2 = j + 1
            if j == ny - 1:
                j2 = j - 3

            a0 = data[iz1, j,  :]
            a1 = _shift_x(data[iz1, j1, :], -1)
            a2 =          data[iz1, j1, :]
            a3 = _shift_x(data[iz1, j1, :], +1)
            a4 = _shift_x(data[iz1, j,  :], -1)
            a5 = _shift_x(data[iz1, j,  :], +1)
            a6 = _shift_x(data[iz1, j2, :], -1)
            a7 =          data[iz1, j2, :]
            a8 = _shift_x(data[iz1, j2, :], +1)

            b0 = data[iz2, j,  :]
            b1 = _shift_x(data[iz2, j1, :], -1)
            b2 =          data[iz2, j1, :]
            b3 = _shift_x(data[iz2, j1, :], +1)
            b4 = _shift_x(data[iz2, j,  :], -1)
            b5 = _shift_x(data[iz2, j,  :], +1)
            b6 = _shift_x(data[iz2, j2, :], -1)
            b7 =          data[iz2, j2, :]
            b8 = _shift_x(data[iz2, j2, :], +1)

            # Keep the original IDL normalization by 8, even though nine terms
            # appear in each sum. This is part of the SSW routine behavior.
            a_avg = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8) / 8.0
            b_avg = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8) / 8.0
            zav8 = np.maximum((a_avg + b_avg) / 2.0, 0.0)
            zav1 = np.maximum((a0 + b0) / 2.0, 0.0)
            zgood = zav1.copy()

            hot_temporal = np.isfinite(zav1) & np.isfinite(zav8) & (zav1 > zav8 * threshold)
            zgood[hot_temporal] = zav8[hot_temporal]

            cur = data[iz, j, :]
            spike = np.isfinite(cur) & np.isfinite(zgood) & (cur > zgood * threshold)
            if np.any(spike):
                out[iz, j, spike] = zgood[spike]

    return out

def _frebin_area_average_2d(img: np.ndarray, out_n: int) -> np.ndarray:
    """Area-overlap rebinning analogous to SSW rebin_map.pro's default FREBIN.

    For integer downsampling ratios this reduces to block averaging. For general
    ratios, each output pixel is the area-weighted average of the input pixels it
    overlaps. This preserves the mean surface brightness scale appropriate for pB
    maps while avoiding interpolation artifacts in the tomography input images.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError(f"_frebin_area_average_2d expects a square 2D image, got {img.shape}")

    n = img.shape[0]
    out_n = int(out_n)
    if n == out_n:
        return img.copy()

    if n % out_n == 0:
        f = n // out_n
        return img.reshape(out_n, f, out_n, f).mean(axis=(1, 3))

    scale = n / float(out_n)
    out = np.empty((out_n, out_n), dtype=np.float64)

    # Separable area-overlap averaging. NaNs are not ignored, matching IDL/FREBIN
    # behavior after fix_nan.pro has already repaired isolated NaNs.
    for oy in range(out_n):
        y0 = oy * scale
        y1 = (oy + 1) * scale
        iy0 = int(np.floor(y0))
        iy1 = int(np.ceil(y1))
        yw = np.array([max(0.0, min(y1, iy + 1.0) - max(y0, iy)) for iy in range(iy0, iy1)])

        for ox in range(out_n):
            x0 = ox * scale
            x1 = (ox + 1) * scale
            ix0 = int(np.floor(x0))
            ix1 = int(np.ceil(x1))
            xw = np.array([max(0.0, min(x1, ix + 1.0) - max(x0, ix)) for ix in range(ix0, ix1)])

            patch = img[iy0:iy1, ix0:ix1]
            weights = yw[:, None] * xw[None, :]
            out[oy, ox] = np.sum(patch * weights) / np.sum(weights)

    return out


def rebin_idl_linear(img: np.ndarray, out_n: int) -> np.ndarray:
    """Rebin a square image to out_n x out_n using rebin_map.pro/FREBIN logic.

    Despite the historical function name, the Ne3dTomo-compatible behavior is now
    area-weighted rebinning, because rebin_map.pro uses FREBIN by default unless
    /CONGRID is explicitly requested. This is preferable for pB tomography because
    it avoids creating interpolated small-scale structure during preprocessing.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError(f"rebin_idl_linear expects a square 2D image, got {img.shape}")
    return _frebin_area_average_2d(img, int(out_n))


# ----------------------------
# "cor1_getpbr.pro" + "get_pbrlc.pro" analog (polar sampling)
# ----------------------------
def _plate_scale_arcsec_per_pix(hdr: fits.Header) -> float:
    # Effective pixel scale from WCS (use |CDELT| if available)
    c1 = abs(float(hdr.get("CDELT1", np.nan)))
    c2 = abs(float(hdr.get("CDELT2", np.nan)))
    if np.isfinite(c1) and np.isfinite(c2) and (c1 > 0) and (c2 > 0):
        return 0.5 * (c1 + c2)
    return 1.0


def _sample_pb_bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Bilinear sample img at floating (x,y) pixel coordinates (0-based).

    This is the Python analogue of IDL's bilinear(pmap.data, ix, jy) used in
    get_pbrlc.pro. Points outside the image or touching an invalid corner are
    returned as NaN rather than being clipped to the edge, because clipped edge
    samples would create artificial pB along radial cuts.
    """
    img = np.asarray(img, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x, y = np.broadcast_arrays(x, y)

    out = np.full(x.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return out

    x0 = np.full(x.shape, -1, dtype=np.int64)
    y0 = np.full(y.shape, -1, dtype=np.int64)
    x0[finite] = np.floor(x[finite]).astype(np.int64)
    y0[finite] = np.floor(y[finite]).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    inside = finite & (x0 >= 0) & (y0 >= 0) & (x1 < img.shape[1]) & (y1 < img.shape[0])
    if not np.any(inside):
        return out

    wx = x - x0
    wy = y - y0

    ia = img[y0[inside], x0[inside]]
    ib = img[y0[inside], x1[inside]]
    ic = img[y1[inside], x0[inside]]
    idd = img[y1[inside], x1[inside]]

    valid_corners = np.isfinite(ia) & np.isfinite(ib) & np.isfinite(ic) & np.isfinite(idd)
    if not np.any(valid_corners):
        return out

    ii = np.flatnonzero(inside)[valid_corners]
    wxv = wx.ravel()[ii]
    wyv = wy.ravel()[ii]
    out.ravel()[ii] = (
        (1.0 - wxv) * (1.0 - wyv) * ia[valid_corners]
        + wxv * (1.0 - wyv) * ib[valid_corners]
        + (1.0 - wxv) * wyv * ic[valid_corners]
        + wxv * wyv * idd[valid_corners]
    )
    return out


def _world_to_pixel_for_hpc_offsets(
    hdr: fits.Header,
    x_arc: np.ndarray,
    y_arc: np.ndarray,
    out_n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert helioprojective offsets in arcsec to rebinned image pixel coordinates.

    The IDL routines use map_get_pixel(pmap, pxy), where pxy contains map-plane
    coordinates in arcsec.  For the same reason as xy_rsun_for_rebinned_image(),
    this helper uses the local linear image-plane WCS rather than full spherical
    HPLN/HPLT longitude coordinates.  This avoids 0/360 deg wrapping for COR1A.
    """
    x_arc = np.asarray(x_arc, dtype=np.float64)
    y_arc = np.asarray(y_arc, dtype=np.float64)

    def _cd_matrix_from_header(h: fits.Header) -> np.ndarray:
        if all(k in h for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
            return np.array(
                [[float(h["CD1_1"]), float(h["CD1_2"])],
                 [float(h["CD2_1"]), float(h["CD2_2"])]],
                dtype=np.float64,
            )
        if all(k in h for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2")):
            pc = np.array(
                [[float(h["PC1_1"]), float(h["PC1_2"])],
                 [float(h["PC2_1"]), float(h["PC2_2"])]],
                dtype=np.float64,
            )
            cdelt1 = float(h.get("CDELT1", 1.0))
            cdelt2 = float(h.get("CDELT2", 1.0))
            return np.diag([cdelt1, cdelt2]) @ pc

        crota = float(h.get("CROTA2", h.get("CROTA", 0.0)))
        cdelt1 = float(h.get("CDELT1", 1.0))
        cdelt2 = float(h.get("CDELT2", 1.0))
        th = np.deg2rad(crota)
        rot = np.array([[np.cos(th), -np.sin(th)],
                        [np.sin(th),  np.cos(th)]], dtype=np.float64)
        return np.diag([cdelt1, cdelt2]) @ rot

    cd = _cd_matrix_from_header(hdr)
    inv = np.linalg.pinv(cd)

    orig_n = int(hdr.get("NAXIS1", out_n))
    if orig_n <= 0:
        orig_n = int(out_n)
    scale = float(orig_n) / float(out_n)

    crpix1 = float(hdr.get("CRPIX1", (orig_n + 1) / 2.0))
    crpix2 = float(hdr.get("CRPIX2", (orig_n + 1) / 2.0))
    crval1 = float(hdr.get("CRVAL1", 0.0))
    crval2 = float(hdr.get("CRVAL2", 0.0))

    cunit1 = str(hdr.get("CUNIT1", "")).lower()
    if "deg" in cunit1:
        xw = x_arc / 3600.0
        yw = y_arc / 3600.0
    elif "arcmin" in cunit1:
        xw = x_arc / 60.0
        yw = y_arc / 60.0
    else:
        xw = x_arc
        yw = y_arc

    dxw = xw - crval1
    dyw = yw - crval2

    dpx = inv[0, 0] * dxw + inv[0, 1] * dyw
    dpy = inv[1, 0] * dxw + inv[1, 1] * dyw

    # Convert original FITS pixel coordinates to the rebinned image coordinates.
    xpix_orig = (dpx + crpix1) - 1.0
    ypix_orig = (dpy + crpix2) - 1.0
    xpix = (xpix_orig + 0.5) / scale - 0.5
    ypix = (ypix_orig + 0.5) / scale - 0.5
    return xpix, ypix

def _clean_pbr_profiles_like_get_pbrlc(
    pbr_r_pa: np.ndarray,
    err_r_pa: np.ndarray,
    r_grid: np.ndarray,
) -> np.ndarray:
    """
    Apply the scientifically relevant bad-track removal from get_pbrlc.pro.

    Ported operations:
      1) reject the dark arc-like track near the occulter before the first pB peak,
      2) reject non-positive samples,
      3) below 2 Rsun, reject samples lower than the mean level in 2.0--2.5 Rsun.

    Interactive plotting and IDL window-control logic are intentionally omitted.
    """
    pbr = np.asarray(pbr_r_pa, dtype=np.float64).copy()
    err = np.asarray(err_r_pa, dtype=np.float64)
    r_grid = np.asarray(r_grid, dtype=np.float64)

    if pbr.ndim != 2:
        raise ValueError("_clean_pbr_profiles_like_get_pbrlc expects a 2-D (nr,npa) array")

    nr, npa = pbr.shape
    for j in range(npa):
        col = pbr[:, j]
        ecol = err[:, j] if err.shape == pbr.shape else np.full(nr, np.nan, dtype=np.float64)

        finite = np.isfinite(col)
        if not np.any(finite):
            continue

        # get_pbrlc.pro: mlc=max(lc); im=where(lc eq mlc); for i=0,im[0] ...
        # Use the first maximum, following IDL's im[0].
        finite_idx = np.flatnonzero(finite)
        imax = finite_idx[np.argmax(col[finite])]
        epeak = ecol[imax]
        if not np.isfinite(epeak) or epeak < 0:
            epeak = 0.1 * abs(col[imax])
        threshold = col[imax] - 0.5 * epeak
        pre = np.arange(nr) <= imax
        bad_pre = pre & np.isfinite(col) & (col <= threshold)
        col[bad_pre] = np.nan

        # get_pbrlc.pro: w=where(lc gt 0)
        col[~np.isfinite(col) | (col <= 0)] = np.nan

        # get_pbrlc.pro: remove bad pixels below r=2 with values lower than
        # the mean between 2.0 and 2.5 Rsun.
        ref = (r_grid > 2.0) & (r_grid < 2.5) & np.isfinite(col) & (col > 0)
        if np.any(ref):
            m0 = float(np.nanmean(col[ref]))
            bad_low = (r_grid <= 2.0) & np.isfinite(col) & (col < m0)
            col[bad_low] = np.nan

        pbr[:, j] = col

    return pbr


def polar_sample_pb(
    pb: np.ndarray,
    hdr,
    out_n: int,
    r_use_min: float,
    r_use_max: float,
    limb_u: float,
    dpa_deg: float = 3.0,
    nr: int = 240,
    hm: int = 3,
    width_pix: int = 0,
    q_low: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Sample a pB map into the Ne3dTomo-style (PA, r) representation.

    This ports the scientifically relevant parts of cor1_getpbr.pro/get_pbrlc.pro:
      - PA is defined relative to solar north and increases anticlockwise.
        The IDL conversion is xpa = pa + 90 deg, so x = -r sin(PA), y = r cos(PA).
      - If width_pix > 0, each radial cut is averaged over 2*width_pix+1 samples
        perpendicular to the cut, reproducing get_pbrlc.pro's /avg behavior.
      - Occulter-adjacent dark-track rejection is applied to the radial profiles.

    Returns flattened arrays in the same order as the previous implementation:
      y_flat, rho, pa, x_pix, y_pix, rsun_arcsec
    where y_flat is arranged as (npa, nr) in C order.
    """
    pb = np.asarray(pb, dtype=np.float64)
    if pb.shape != (out_n, out_n):
        raise ValueError(f"polar_sample_pb expects pb shape {(out_n, out_n)}, got {pb.shape}")

    rsun_arcsec = float(hdr.get("RSUN_OBS", hdr.get("RSUN", 959.63)))
    if not np.isfinite(rsun_arcsec) or rsun_arcsec <= 0:
        rsun_arcsec = 959.63

    rgrid = np.linspace(float(r_use_min), float(r_use_max), int(nr), dtype=np.float64)
    pa_grid = np.arange(0.0, 360.0, float(dpa_deg), dtype=np.float64)
    npa = int(pa_grid.size)

    rr, pp = np.meshgrid(rgrid, pa_grid, indexing="xy")  # (npa,nr)
    pa_rad = np.deg2rad(pp)

    # cor1_getpbr.pro defines PA from the north pole and converts to x-axis angle
    # with xpa=pa+90 deg. Therefore x=r*cos(xpa)=-r*sin(pa), y=r*sin(xpa)=r*cos(pa).
    x_arc_center = -rr * np.sin(pa_rad) * rsun_arcsec
    y_arc_center =  rr * np.cos(pa_rad) * rsun_arcsec

    xpix_center, ypix_center = _world_to_pixel_for_hpc_offsets(
        hdr, x_arc_center, y_arc_center, out_n=out_n
    )

    width_i = int(round(float(width_pix))) if width_pix is not None else 0
    if width_i > 0:
        # get_pbrlc.pro uses offsets in image pixels, converted here to arcsec
        # using the effective plate scale of the rebinned map.
        try:
            scale_arcsec = _plate_scale_arcsec_per_pix(hdr)
            orig_n = int(hdr.get("NAXIS1", out_n))
            if orig_n > 0 and orig_n != out_n:
                scale_arcsec *= float(orig_n) / float(out_n)
        except Exception:
            scale_arcsec = rsun_arcsec * (2.0 * float(r_use_max)) / float(out_n)

        offsets = np.arange(-width_i, width_i + 1, dtype=np.float64) * scale_arcsec
        # Tangential direction used in get_pbrlc.pro: th=xpa+90 deg = PA+180 deg.
        tx = -np.cos(pa_rad)
        ty = -np.sin(pa_rad)

        x_arc = x_arc_center[:, :, None] + tx[:, :, None] * offsets[None, None, :]
        y_arc = y_arc_center[:, :, None] + ty[:, :, None] * offsets[None, None, :]
        xpix, ypix = _world_to_pixel_for_hpc_offsets(hdr, x_arc, y_arc, out_n=out_n)
        samples = _sample_pb_bilinear(pb, xpix, ypix)

        with np.errstate(invalid="ignore", divide="ignore"):
            y = np.nanmean(samples, axis=2)
            err = np.nanstd(samples, axis=2)
        y = np.where(np.isfinite(y), y, np.nan)
        err = np.where(np.isfinite(err), err, np.nan)
    else:
        y = _sample_pb_bilinear(pb, xpix_center, ypix_center)
        err = 0.1 * np.abs(y)

    # Remove points inside the nominal COR1 occulter radius in the same spirit as
    # get_pbrlc.pro's pmap.data[wct] = -100. This remains harmless for K-Cor/LASCO
    # maps because rgrid is explicitly controlled by r_use_min/r_use_max.
    y = np.where(rr <= 1.5, np.nan, y)
    y = _clean_pbr_profiles_like_get_pbrlc(y.T, err.T, rgrid).T  # clean expects (nr,npa)

    if q_low is not None and q_low > 0:
        # This option is not part of get_pbrlc.pro. Keep it only as an explicitly
        # requested extra high-pass-like diagnostic; default settings use q_low=0.
        try:
            from scipy.ndimage import median_filter  # type: ignore
            y2 = median_filter(y.copy(), size=(2 * int(hm) + 1, 1), mode="nearest")
            y = y - y2
        except Exception:
            pass

    rho = rr.ravel(order="C")
    pa = pp.ravel(order="C")
    x_pix = np.asarray(xpix_center, dtype=np.float64).ravel(order="C")
    y_pix = np.asarray(ypix_center, dtype=np.float64).ravel(order="C")
    y_flat = y.ravel(order="C")

    return y_flat, rho, pa, x_pix, y_pix, rsun_arcsec


# ----------------------------
# "get_cor1_bbk.pro" analog (FFT background)
# ----------------------------
def _idl_like_histogram_peak_pb(data: np.ndarray, binsize: float = 0.2e-10) -> float:
    """
    Return the pB histogram peak level in the same spirit as preview_data.pro:

        y = histogram(dat, binsize=0.2e-10, location=xbin)
        pbns = xbin[imax]

    IDL's `location` array corresponds to bin locations. For this Python analogue,
    the lower edge of the most populated bin is used. The value is used only as a
    conservative noise/floor proxy, not as a photometric calibration factor.
    """
    v = np.asarray(data, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 50:
        return float(np.nanstd(v)) if v.size else 1e-30

    vmin = float(np.nanmin(v))
    vmax = float(np.nanmax(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 1e-30

    lo = min(0.0, vmin)
    hi = float(np.nanpercentile(v, 99.9))
    if not np.isfinite(hi) or hi <= lo:
        hi = vmax
    nbins = int(np.ceil((hi - lo) / binsize))
    nbins = int(max(10, min(5000, nbins)))
    edges = lo + binsize * np.arange(nbins + 1, dtype=np.float64)
    if edges[-1] < hi:
        edges = np.append(edges, hi)

    hist, edges = np.histogram(v, bins=edges)
    if hist.size == 0 or np.max(hist) <= 0:
        return 1e-30
    imax = int(np.argmax(hist))
    pbns = float(edges[imax])
    if not np.isfinite(pbns) or pbns <= 0:
        pos_edges = edges[:-1][edges[:-1] > 0]
        pbns = float(pos_edges[0]) if pos_edges.size else 1e-30
    return pbns


def _smooth_boxcar_1d_idl_like(y: np.ndarray, width: int = 5) -> np.ndarray:
    """
    Approximate IDL smooth(y, width) with a centered boxcar and edge padding.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0 or width <= 1:
        return y.copy()
    width = int(width)
    if width % 2 == 0:
        width += 1
    pad = width // 2
    kernel = np.ones(width, dtype=np.float64) / float(width)
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


def _low_harmonic_pa_background(pbr_r_pa: np.ndarray, hm: int) -> np.ndarray:
    """
    Apply the get_cor1_bbk.pro low-harmonic PA filtering for one pB(r,PA) map.

    IDL reference:
        ft = fft(reform(pb[i,*,k]), -1)
        ft[hm:np-hm] = 0.
        mbk[i,k] = max(float(fft(ft, 1)))

    This keeps m=0..hm-1 and the corresponding negative harmonics, then takes the
    maximum over PA at each radius.
    """
    pbr = np.asarray(pbr_r_pa, dtype=np.float64)
    if pbr.ndim != 2:
        raise ValueError("_low_harmonic_pa_background expects a 2-D (nr,npa) array")

    nr, npa = pbr.shape
    hm_i = int(hm)
    if hm_i < 1:
        hm_i = 1
    if hm_i >= npa // 2:
        hm_i = max(1, npa // 2 - 1)

    mbk = np.full(nr, np.nan, dtype=np.float64)
    for i in range(nr):
        row = pbr[i, :]
        finite = np.isfinite(row)
        if np.count_nonzero(finite) < 4:
            continue

        fill = float(np.nanmedian(row[finite]))
        row_f = np.where(finite, row, fill)

        ft = np.fft.fft(row_f)
        ft[hm_i:npa - hm_i] = 0.0
        smoothed = np.real(np.fft.ifft(ft))
        mbk[i] = float(np.nanmax(smoothed))

    return mbk


def ybk_profile_fft(
    pb: np.ndarray,
    hdr,
    rmin: float,
    rmax: float,
    dpa_deg: float = 3.0,
    nr: int = 240,
    hm: int = 3,
    width_pix: int = 10,
    q_low: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimate radial pB background ybk(r) using the get_cor1_bbk.pro logic.

    For a single map this is the one-image equivalent of the Ne3dTomo IDL routine:
      1) sample the pB map into a (r, PA) grid,
      2) FFT each radial PA profile,
      3) keep only Fourier modes m=0..hm-1 and the corresponding negative modes,
      4) inverse FFT and take max over PA at each radius,
      5) apply smooth(..., 5) along radius,
      6) append a final r=4.0 point when needed, as get_cor1_bbk.pro does.

    The previous implementation included additional low-r artifact suppression. That
    has been removed here because it is not present in the uploaded get_cor1_bbk.pro.
    """
    pb = np.asarray(pb, dtype=np.float64)
    if pb.ndim != 2 or pb.shape[0] != pb.shape[1]:
        raise ValueError(f"ybk_profile_fft expects a square 2-D pb map, got {pb.shape}")

    out_n = int(pb.shape[0])
    r_grid = np.linspace(float(rmin), float(rmax), int(nr))
    pa_grid = np.arange(0.0, 360.0, float(dpa_deg), dtype=np.float64)
    npa = int(pa_grid.size)

    y_flat, _, _, _, _, _ = polar_sample_pb(
        pb, hdr,
        out_n=out_n,
        r_use_min=float(rmin),
        r_use_max=float(rmax),
        limb_u=float(DEFAULT_LIMB_U),
        dpa_deg=float(dpa_deg),
        nr=int(nr),
        hm=int(hm),
        width_pix=int(width_pix),
        q_low=float(q_low) if q_low is not None else 0.0,
    )

    if y_flat.size != npa * int(nr):
        if int(nr) > 0 and (y_flat.size % int(nr) == 0):
            npa = int(y_flat.size // int(nr))
        else:
            raise ValueError(f"polar_sample_pb returned unexpected length {y_flat.size} for nr={nr}")

    pbr = y_flat.reshape((npa, int(nr)), order="C").T  # (nr,npa)
    finite_cnt = int(np.count_nonzero(np.isfinite(pbr)))
    if finite_cnt < 10:
        data = pb[np.isfinite(pb)]
        y0 = float(np.nanmedian(data)) if data.size else np.nan
        ybk = np.full(int(nr), y0, dtype=np.float64)
        pb_noise = _idl_like_histogram_peak_pb(data)
        return r_grid, ybk, pb_noise

    mbk = _low_harmonic_pa_background(pbr, hm=hm)

    good = np.isfinite(mbk)
    if np.count_nonzero(good) >= 2:
        mbk_clean = np.interp(r_grid, r_grid[good], mbk[good])
    elif np.count_nonzero(good) == 1:
        mbk_clean = np.full_like(r_grid, float(mbk[good][0]), dtype=np.float64)
    else:
        data = pb[np.isfinite(pb)]
        mbk_clean = np.full_like(r_grid, float(np.nanmedian(data)) if data.size else np.nan)

    ybk = _smooth_boxcar_1d_idl_like(mbk_clean, width=5)

    if abs(float(r_grid[-1]) - 4.0) > 1e-8:
        r_out = np.concatenate([r_grid, np.array([4.0], dtype=np.float64)])
        y_out = np.concatenate([ybk, np.array([ybk[-1]], dtype=np.float64)])
    else:
        r_out = r_grid
        y_out = ybk

    pb_noise = _idl_like_histogram_peak_pb(pb)
    return r_out, y_out, pb_noise


def ybk_profile_fft_stack(
    observations: List["Observation"],
    rmin: float,
    rmax: float,
    dpa_deg: float = 3.0,
    nr: int = 240,
    hm: int = 3,
    width_pix: int = 10,
    q_low: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Multi-image version of get_cor1_bbk.pro.

    The uploaded IDL code computes mbk(r,k) for every image k, then uses the mean
    over k before smooth(abk,5):

        aer[i] = stdev(mbk[i,*], m0)
        abk[i] = m0
        ybk = smooth(abk,5)

    This function follows that logic for a homogeneous group of observations
    (e.g. all COR1A maps, or all Earth-view merged maps). Mixing instruments in a
    single background estimate is deliberately avoided in `main()`.
    """
    if not observations:
        raise ValueError("ybk_profile_fft_stack requires at least one observation")

    r_grid = np.linspace(float(rmin), float(rmax), int(nr))
    mbk_all: List[np.ndarray] = []
    all_pb_values: List[np.ndarray] = []

    for obs in observations:
        pb = np.asarray(obs.pb, dtype=np.float64)
        out_n = int(pb.shape[0])
        pa_grid = np.arange(0.0, 360.0, float(dpa_deg), dtype=np.float64)
        npa = int(pa_grid.size)

        y_flat, _, _, _, _, _ = polar_sample_pb(
            pb, obs.hdr,
            out_n=out_n,
            r_use_min=float(rmin),
            r_use_max=float(rmax),
            limb_u=float(DEFAULT_LIMB_U),
            dpa_deg=float(dpa_deg),
            nr=int(nr),
            hm=int(hm),
            width_pix=int(width_pix),
            q_low=float(q_low) if q_low is not None else 0.0,
        )
        if y_flat.size != npa * int(nr):
            if int(nr) > 0 and (y_flat.size % int(nr) == 0):
                npa = int(y_flat.size // int(nr))
            else:
                continue

        pbr = y_flat.reshape((npa, int(nr)), order="C").T
        mbk_all.append(_low_harmonic_pa_background(pbr, hm=hm))
        all_pb_values.append(pb[np.isfinite(pb)])

    if not mbk_all:
        data = np.concatenate(all_pb_values) if all_pb_values else np.array([], dtype=np.float64)
        y0 = float(np.nanmedian(data)) if data.size else np.nan
        return r_grid, np.full(int(nr), y0, dtype=np.float64), _idl_like_histogram_peak_pb(data)

    mbk_stack = np.stack(mbk_all, axis=1)  # (nr,nimg)
    abk = np.full(int(nr), np.nan, dtype=np.float64)
    for i in range(int(nr)):
        row = mbk_stack[i, :]
        finite = np.isfinite(row)
        if np.any(finite):
            abk[i] = float(np.nanmean(row[finite]))

    good = np.isfinite(abk)
    if np.count_nonzero(good) >= 2:
        abk_clean = np.interp(r_grid, r_grid[good], abk[good])
    elif np.count_nonzero(good) == 1:
        abk_clean = np.full_like(r_grid, float(abk[good][0]), dtype=np.float64)
    else:
        data = np.concatenate(all_pb_values) if all_pb_values else np.array([], dtype=np.float64)
        abk_clean = np.full_like(r_grid, float(np.nanmedian(data)) if data.size else np.nan)

    ybk = _smooth_boxcar_1d_idl_like(abk_clean, width=5)

    if abs(float(r_grid[-1]) - 4.0) > 1e-8:
        r_out = np.concatenate([r_grid, np.array([4.0], dtype=np.float64)])
        y_out = np.concatenate([ybk, np.array([ybk[-1]], dtype=np.float64)])
    else:
        r_out = r_grid
        y_out = ybk

    data = np.concatenate(all_pb_values) if all_pb_values else np.array([], dtype=np.float64)
    pb_noise = _idl_like_histogram_peak_pb(data)
    return r_out, y_out, pb_noise


def _parse_positive_float_or_none(value) -> Optional[float]:
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
        out = float(value)
        if not np.isfinite(out) or out <= 0:
            return None
        return out
    except Exception:
        return None


def update_observation_weights_from_ybk(
    obs: Observation,
    rgrid: np.ndarray,
    ybk: np.ndarray,
    pb_noise: float,
    pb_floor: float | str = "",
) -> Observation:
    """
    Replace per-pixel weights with weights derived from the Ne3dTomo-style global
    ybk(r) profile for the corresponding homogeneous observation group.
    """
    rho = np.hypot(obs.x, obs.y)
    mask = obs.mask & np.isfinite(obs.pb) & np.isfinite(rho)

    pb_floor_user = _parse_positive_float_or_none(pb_floor)
    floor = pb_floor_user if pb_floor_user is not None else float(pb_noise)
    if not np.isfinite(floor) or floor <= 0:
        floor = 1e-30
    if np.isfinite(pb_noise) and pb_noise > 0:
        floor = max(floor, float(pb_noise))

    ybk_pix = np.interp(rho[mask], rgrid, ybk)
    ybk_pix = np.where(np.isfinite(ybk_pix) & (ybk_pix > 0), ybk_pix, floor)

    w = 1.0 / np.maximum(ybk_pix, floor)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

    idx_map = np.flatnonzero(mask.ravel())
    keep = w > 0
    if np.any(~keep):
        idx_map = idx_map[keep]
        w = w[keep]
        mask2 = np.zeros(mask.size, dtype=bool)
        mask2[idx_map] = True
        mask = mask2.reshape(mask.shape)

    return Observation(
        pb=obs.pb,
        hdr=obs.hdr,
        x=obs.x,
        y=obs.y,
        mask=mask,
        w=w,
        idx_map=idx_map,
        cam_x=obs.cam_x,
        cam_y=obs.cam_y,
        cam_z=obs.cam_z,
        lonlat_deg=obs.lonlat_deg,
        limb_component_names=obs.limb_component_names,
        limb_u_values=obs.limb_u_values,
        limb_coeff_a=obs.limb_coeff_a,
        limb_coeff_b=obs.limb_coeff_b,
        limb_component_weight_maps=obs.limb_component_weight_maps,
        limb_u_model=obs.limb_u_model,
    )


def tomography_observation_group_key(path: Path) -> str:
    """
    Group observations for Ne3dTomo-style temporal despike/background estimation.

    Ne3dTomo get_cor1_bbk.pro estimates one background profile from a homogeneous
    set of COR1 maps. Here we avoid mixing COR1A and Earth-view K-Cor/LASCO maps in
    the same ybk(r) estimate.
    """
    name = Path(path).name
    if re.fullmatch(r"COR1A_pb_pre_\d{8}_\d{6}\.fits", name):
        return "cor1a"
    if re.fullmatch(r"pB_Kcor_LASCO_axi_\d{8}_\d{4}\.fits", name):
        return "earth_merged"
    if re.fullmatch(r"pB_LASCO_C2_only_\d{8}_\d{4}\.fits", name):
        return "earth_lasco_only"
    return "other"


def build_ne3dtomo_temporal_despike_overrides(
    pb_paths: List[Path],
    out_n: int,
    nsig: float = 6.0,
) -> dict[Path, np.ndarray]:
    """
    Approximate preview_data.pro's /filt branch for homogeneous groups.

    IDL reference:
        data[*,*,i] = fix_nan(data[*,*,i])
        cleaned = ssw_unspike_cube(index, data, newindex, thresh=6.0)
        smap = rebin_map(pbmaps, 128, 128)

    Python analogue:
      - fill NaNs by 4-neighbor means before rebinning,
      - rebin to out_n,
      - apply the SSW ssw_unspike_cube-style temporal despike along the image axis.

    Groups with fewer than four files are left untouched because ssw_unspike_cube requires NTIMES > 3; build_observation() can still apply the per-image spatial despike fallback.
    """
    groups: dict[str, List[Path]] = {}
    for path in pb_paths:
        groups.setdefault(tomography_observation_group_key(path), []).append(Path(path))

    overrides: dict[Path, np.ndarray] = {}
    for key, paths in groups.items():
        if len(paths) <= 3:
            print(f"[INFO] Ne3dTomo-style temporal despike skipped for group '{key}' (n={len(paths)}; need >3).")
            continue

        cube = []
        used_paths = []
        for path in paths:
            pb0, _ = read_fits_image(path)
            pb0 = fill_nan_by_neighbor_mean(pb0, max_passes=1)
            pb1 = rebin_idl_linear(pb0, out_n) if pb0.shape[0] != out_n else pb0.astype(np.float64, copy=True)
            cube.append(pb1.astype(np.float64))
            used_paths.append(path)

        cleaned = despike_pb_cube(np.stack(cube, axis=0), nsig=nsig, use_log=False)
        for path, arr in zip(used_paths, cleaned):
            overrides[path] = arr
        print(f"[INFO] Ne3dTomo-style temporal despike applied for group '{key}' (n={len(used_paths)}).")

    return overrides


# ----------------------------
# Data containers
# ----------------------------
@dataclass
class Observation:
    pb: np.ndarray               # pB image (rebinned)
    hdr: fits.Header             # header (original, used for WCS)
    x: np.ndarray                # x map [Rsun] (rebinned grid)
    y: np.ndarray                # y map [Rsun]
    mask: np.ndarray             # boolean mask for used pixels
    w: np.ndarray                # weights for used pixels (vector)
    idx_map: np.ndarray          # flat indices of used pixels
    cam_x: np.ndarray            # camera basis x-hat in Carrington
    cam_y: np.ndarray            # camera basis y-hat in Carrington
    cam_z: np.ndarray            # camera basis z-hat in Carrington
    lonlat_deg: Optional[Tuple[float, float]] = None
    limb_component_names: Tuple[str, ...] = ("fixed",)
    limb_u_values: Tuple[float, ...] = (DEFAULT_LIMB_U,)
    limb_coeff_a: Tuple[float, ...] = (1.0 - DEFAULT_LIMB_U,)
    limb_coeff_b: Tuple[float, ...] = (DEFAULT_LIMB_U,)
    limb_component_weight_maps: Optional[np.ndarray] = None
    limb_u_model: str = "fixed_scalar"


@dataclass
class SphericalGrid:
    r_edges: np.ndarray
    th_edges: np.ndarray
    ph_edges: np.ndarray

    @property
    def nr(self) -> int:
        return self.r_edges.size - 1

    @property
    def nth(self) -> int:
        return self.th_edges.size - 1

    @property
    def nph(self) -> int:
        return self.ph_edges.size - 1

    @property
    def nvox(self) -> int:
        return self.nr * self.nth * self.nph

    def voxel_centers_sph(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])
        th = 0.5 * (self.th_edges[:-1] + self.th_edges[1:])
        ph = 0.5 * (self.ph_edges[:-1] + self.ph_edges[1:])
        rr, tt, pp = np.meshgrid(r, th, ph, indexing="ij")
        return rr, tt, pp

    def voxel_centers_xyz(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rr, tt, pp = self.voxel_centers_sph()
        x = rr * np.sin(tt) * np.cos(pp)
        y = rr * np.sin(tt) * np.sin(pp)
        z = rr * np.cos(tt)
        return x, y, z

    def flat_index(self, ir: int, ith: int, iph: int) -> int:
        return (ir * self.nth + ith) * self.nph + iph


@dataclass
class RayBundle:
    vox_idx: List[np.ndarray]  # list (per-ray) of voxel indices
    vox_w: List[np.ndarray]    # list (per-ray) of weights for those voxels


# ----------------------------
# Forward model construction
# ----------------------------
def build_rays_for_observation(
    obs: Observation,
    grid: SphericalGrid,
    ds_rsun: float,
    r_min: float,
    r_max: float,
    limb_u: float,
    thomson_normalize_msb: bool = True,
    thomson_kernel_scale: float = 1.0,
) -> RayBundle:
    """
    For each used pixel, build a sparse ray (list of voxel indices + weights).
    The LOS is sampled in image coordinate s, and mapped into Carrington frame
    using obs camera basis (cam_x, cam_y, cam_z).
    """
    x_use = obs.x.ravel()[obs.idx_map]
    y_use = obs.y.ravel()[obs.idx_map]

    component_names = tuple(getattr(obs, "limb_component_names", ("fixed",)))
    coeff_a = np.asarray(getattr(obs, "limb_coeff_a", (np.nan,)), dtype=np.float64)
    coeff_b = np.asarray(getattr(obs, "limb_coeff_b", (np.nan,)), dtype=np.float64)
    if coeff_a.size != coeff_b.size or coeff_a.size != len(component_names):
        raise ValueError(
            f"Invalid limb-kernel component metadata for observation: "
            f"names={len(component_names)}, A={coeff_a.size}, B={coeff_b.size}"
        )
    if np.any(~np.isfinite(coeff_a)) or np.any(~np.isfinite(coeff_b)):
        # Compatibility fallback for manually constructed legacy Observation objects.
        a0, b0 = limb_kernel_coefficients_from_u(
            float(limb_u), normalize_msb=bool(thomson_normalize_msb)
        )
        component_names = ("fixed",)
        coeff_a = np.array([a0], dtype=np.float64)
        coeff_b = np.array([b0], dtype=np.float64)

    weight_maps = getattr(obs, "limb_component_weight_maps", None)
    if weight_maps is None:
        component_weights_use = np.ones((1, obs.idx_map.size), dtype=np.float64)
    else:
        weight_maps = np.asarray(weight_maps, dtype=np.float64)
        if weight_maps.shape != (coeff_a.size,) + obs.pb.shape:
            raise ValueError(
                f"limb_component_weight_maps shape {weight_maps.shape} does not match "
                f"expected {(coeff_a.size,) + obs.pb.shape}."
            )
        component_weights_use = weight_maps.reshape(coeff_a.size, -1)[:, obs.idx_map]
        component_weights_use = np.clip(np.nan_to_num(component_weights_use, nan=0.0), 0.0, 1.0)
        sums = np.sum(component_weights_use, axis=0)
        bad = sums <= 0
        if np.any(bad):
            component_weights_use[:, bad] = 0.0
            component_weights_use[0, bad] = 1.0
            sums[bad] = 1.0
        component_weights_use /= sums[None, :]

    rE = grid.r_edges
    tE = grid.th_edges
    pE = grid.ph_edges

    cx, cy, cz = obs.cam_x, obs.cam_y, obs.cam_z

    vox_idx_list: List[np.ndarray] = []
    vox_w_list: List[np.ndarray] = []

    for iray, (xp, yp) in enumerate(zip(x_use, y_use)):
        rho = float(np.hypot(xp, yp))
        ray_component_weights = component_weights_use[:, iray]
        if rho >= r_max:
            vox_idx_list.append(np.array([], dtype=np.int32))
            vox_w_list.append(np.array([], dtype=np.float64))
            continue

        s_max = np.sqrt(max(0.0, r_max * r_max - rho * rho))
        nstep = int(np.ceil((2.0 * s_max) / ds_rsun)) + 1
        s_arr = np.linspace(-s_max, +s_max, nstep)

        acc: dict[int, float] = {}

        # uniform ds in Rsun
        ds = (2.0 * s_max) / max(1, (nstep - 1))

        for s in s_arr:
            r = np.sqrt(rho * rho + s * s)
            if (r < r_min) or (r > r_max):
                continue

            # Carrington Cartesian position [Rsun]
            pos = xp * cx + yp * cy + s * cz
            rr = float(np.linalg.norm(pos))
            if rr <= 0:
                continue

            # Spherical coords in Carrington frame
            th = np.arccos(np.clip(pos[2] / rr, -1.0, 1.0))
            ph = np.arctan2(pos[1], pos[0])
            if ph < 0:
                ph += 2.0 * np.pi

            ir = np.searchsorted(rE, rr) - 1
            ith = np.searchsorted(tE, th) - 1
            iph = np.searchsorted(pE, ph) - 1
            if ir < 0 or ir >= grid.nr or ith < 0 or ith >= grid.nth or iph < 0 or iph >= grid.nph:
                continue

            theta_from_pos = np.arccos(np.clip(rho / r, -1.0, 1.0))
            pb_per_e = 0.0
            for component_weight, a_component, b_component in zip(
                ray_component_weights, coeff_a, coeff_b
            ):
                if component_weight <= 0:
                    continue
                pb_per_e += float(component_weight) * thomsonscatter_pB_per_electron(
                    rho,
                    theta_from_pos,
                    u=limb_u,
                    normalize_msb=bool(thomson_normalize_msb),
                    kernel_scale=float(thomson_kernel_scale),
                    limb_coefficients=(float(a_component), float(b_component)),
                )

            # Weight: kernel * ds * Rsun(cm)
            w = pb_per_e * ds * RSUN_CM

            vidx = grid.flat_index(ir, ith, iph)
            acc[vidx] = acc.get(vidx, 0.0) + w

        if len(acc) == 0:
            vox_idx_list.append(np.array([], dtype=np.int32))
            vox_w_list.append(np.array([], dtype=np.float64))
        else:
            idx = np.fromiter(acc.keys(), dtype=np.int32)
            ww = np.fromiter(acc.values(), dtype=np.float64)
            vox_idx_list.append(idx)
            vox_w_list.append(ww)

    return RayBundle(vox_idx=vox_idx_list, vox_w=vox_w_list)


# ----------------------------
# Regularization operator (L^T L)
# ----------------------------
def _regularization_radial_scale(grid: SphericalGrid, wt_r: Optional[np.ndarray]) -> np.ndarray:
    """Return the per-radius row scale used by the Ne3dTomo regularizer."""
    nr = grid.nr
    if wt_r is not None:
        wr = np.asarray(wt_r, dtype=np.float64).copy()
        if wr.size != nr:
            raise ValueError(f"wt_r must have length nr={nr}, got {wr.size}")
        wr = np.where(np.isfinite(wr) & (wr > 0), wr, np.nan)
        if np.any(~np.isfinite(wr)):
            good = np.isfinite(wr)
            if np.count_nonzero(good) >= 2:
                r_cent = 0.5 * (grid.r_edges[:-1] + grid.r_edges[1:])
                wr[~good] = np.interp(r_cent[~good], r_cent[good], wr[good])
            else:
                wr = np.ones(nr, dtype=np.float64)
        wr = np.maximum(wr, 1e-12)
    else:
        wr = np.ones(nr, dtype=np.float64)
    return 1.0 / wr


def apply_LTL(x: np.ndarray, grid: SphericalGrid, wt_r: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply exactly the same ``R.T @ R`` operator as the original loop code.

    This implementation is vectorized over radius, colatitude, and longitude.
    It preserves the Ne3dTomo/Fortran row definitions, including periodic phi,
    omitted theta-boundary rows, and one-sided radial first differences at both
    radial boundaries.  Only the computational representation is changed.
    """
    nr, nth, nph = grid.nr, grid.nth, grid.nph
    X = np.asarray(x, dtype=np.float64).reshape((nr, nth, nph))
    out = np.zeros_like(X)
    if nr == 0 or nth < 3 or nph == 0:
        return out.ravel()

    sc = _regularization_radial_scale(grid, wt_r)
    sc3 = sc[:, None, None]
    Xi = X[:, 1:-1, :]

    # Longitude rows: [1, -2, 1], periodic in phi.
    res_p = (np.roll(Xi, 1, axis=2) - 2.0 * Xi + np.roll(Xi, -1, axis=2)) * sc3
    out[:, 1:-1, :] += (-2.0 * sc3) * res_p
    out[:, 1:-1, :] += sc3 * np.roll(res_p, -1, axis=2)
    out[:, 1:-1, :] += sc3 * np.roll(res_p, 1, axis=2)

    # Colatitude rows: [1, -2, 1] for theta-interior row centers only.
    res_t = (X[:, :-2, :] - 2.0 * X[:, 1:-1, :] + X[:, 2:, :]) * sc3
    out[:, :-2, :] += sc3 * res_t
    out[:, 1:-1, :] += (-2.0 * sc3) * res_t
    out[:, 2:, :] += sc3 * res_t

    if nr > 1:
        # Inner radial boundary row: [-1, 1], scaled by sc[0].
        res_r0 = (-X[0, 1:-1, :] + X[1, 1:-1, :]) * sc[0]
        out[0, 1:-1, :] += (-sc[0]) * res_r0
        out[1, 1:-1, :] += sc[0] * res_r0

        # Outer radial boundary row: [-1, 1], scaled by sc[-1].
        res_rn = (-X[-2, 1:-1, :] + X[-1, 1:-1, :]) * sc[-1]
        out[-2, 1:-1, :] += (-sc[-1]) * res_rn
        out[-1, 1:-1, :] += sc[-1] * res_rn

    if nr > 2:
        # Interior radial rows: [1, -2, 1].
        sci = sc[1:-1, None, None]
        res_ri = (
            X[:-2, 1:-1, :]
            - 2.0 * X[1:-1, 1:-1, :]
            + X[2:, 1:-1, :]
        ) * sci
        out[:-2, 1:-1, :] += sci * res_ri
        out[1:-1, 1:-1, :] += (-2.0 * sci) * res_ri
        out[2:, 1:-1, :] += sci * res_ri

    return out.ravel()


def regularization_diagonal(
    grid: SphericalGrid,
    wt_r: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the exact diagonal of the current ``R.T @ R`` operator.

    The diagonal is used only for preconditioning.  It does not alter the
    objective function or the reconstructed solution.
    """
    nr, nth, nph = grid.nr, grid.nth, grid.nph
    diag = np.zeros((nr, nth, nph), dtype=np.float64)
    if nr == 0 or nth < 3 or nph == 0:
        return diag.ravel()

    sc2 = _regularization_radial_scale(grid, wt_r) ** 2

    # Phi: every theta-interior voxel appears once as center (-2) and twice as
    # a periodic neighbor (+1), giving 4 + 1 + 1 = 6 times sc^2.
    diag[:, 1:-1, :] += 6.0 * sc2[:, None, None]

    # Theta rows centered at j=1..nth-2.
    theta_coeff = np.zeros(nth, dtype=np.float64)
    theta_coeff[1:-1] += 4.0
    theta_coeff[:-2] += 1.0
    theta_coeff[2:] += 1.0
    diag += sc2[:, None, None] * theta_coeff[None, :, None]

    # Radial rows exist only at theta-interior locations.
    radial_coeff = np.zeros(nr, dtype=np.float64)
    if nr > 1:
        radial_coeff[0] += sc2[0]
        radial_coeff[1] += sc2[0]
        radial_coeff[-2] += sc2[-1]
        radial_coeff[-1] += sc2[-1]
    if nr > 2:
        radial_coeff[:-2] += sc2[1:-1]
        radial_coeff[1:-1] += 4.0 * sc2[1:-1]
        radial_coeff[2:] += sc2[1:-1]
    diag[:, 1:-1, :] += radial_coeff[:, None, None]
    return diag.ravel()


# ----------------------------
# Tomography solver
# ----------------------------
class RegularizedTomography:
    def __init__(
        self,
        grid: SphericalGrid,
        observations: List[Observation],
        rays: List[RayBundle],
        lam: float = 1e-2,
        wt_r: Optional[np.ndarray] = None,
        density_basis: Optional[np.ndarray] = None,
        measurement_scale: Optional[np.ndarray] = None,
        forward_matrix=None,
        use_preconditioner: bool = True,
        preconditioner_floor: float = 1e-12,
    ):
        self.grid = grid
        self.observations = observations
        self.rays = rays
        self.lam = float(lam)
        self.wt_r = wt_r
        self.density_basis = None
        if density_basis is not None:
            db = np.asarray(density_basis, dtype=np.float64).ravel()
            if db.size != grid.nvox:
                raise ValueError(f"density_basis must have size {grid.nvox}, got {db.size}")
            db = np.where(np.isfinite(db) & (db > 0), db, 1.0)
            self.density_basis = db
        self.W = np.concatenate([o.w for o in observations]).astype(np.float64)
        self._build_slices()
        self.measurement_scale = np.ones(self.n_meas, dtype=np.float64)
        self.use_preconditioner = bool(use_preconditioner)
        self.preconditioner_floor = float(preconditioner_floor)
        self._measurement_scale_version = 0
        self._data_diagonal_cache_version = -1
        self._data_diagonal_cache = None
        self._regularization_diagonal_cache = None
        self.last_solver_iterations = 0
        self.last_solver_used_preconditioner = False
        if measurement_scale is not None:
            self.set_measurement_scale(measurement_scale)
        if forward_matrix is None:
            self._build_sparse_forward_matrix()
        else:
            if tuple(forward_matrix.shape) != (self.n_meas, self.grid.nvox):
                raise ValueError(
                    f"forward_matrix shape must be {(self.n_meas, self.grid.nvox)}, "
                    f"got {forward_matrix.shape}"
                )
            self.A_csr = forward_matrix
            print(
                f"[MATRIX-CACHE-HIT] Reusing sparse forward matrix A: "
                f"shape={self.A_csr.shape}, nnz={self.A_csr.nnz}"
            )

    def set_measurement_scale(self, measurement_scale: np.ndarray) -> None:
        """
        Set row-wise forward-response gains.

        The effective observation operator is diag(measurement_scale) @ A.  This
        permits one relative radiometric gain per instrument group without
        modifying the observations, data weights, selected viewpoints, or ray
        cache.
        """
        scale = np.asarray(measurement_scale, dtype=np.float64).ravel()
        if scale.size != self.n_meas:
            raise ValueError(
                f"measurement_scale must have size {self.n_meas}, got {scale.size}"
            )
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("measurement_scale must contain positive finite values.")
        self.measurement_scale = scale.copy()
        self._measurement_scale_version = getattr(self, "_measurement_scale_version", 0) + 1
        self._data_diagonal_cache_version = -1
        self._data_diagonal_cache = None

    def solution_to_density(self, x: np.ndarray) -> np.ndarray:
        """Convert the solver variable to physical electron density [cm^-3]."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != self.grid.nvox:
            raise ValueError(f"solution_to_density expected x.size={self.grid.nvox}, got {x.size}")
        if self.density_basis is None:
            return x
        return self.density_basis * x

    def _build_slices(self):
        self.slices: List[slice] = []
        start = 0
        for obs in self.observations:
            n = obs.idx_map.size
            self.slices.append(slice(start, start + n))
            start += n
        self.n_meas = start

    def _build_sparse_forward_matrix(self) -> None:
        """
        Build the sparse forward operator A in CSR format.

        This is a computational representation change only.  Each row contains
        exactly the same voxel indices and Thomson/LOS weights that were already
        stored in RayBundle.vox_idx and RayBundle.vox_w.  Therefore A_times() and
        AT_times() remain scientifically identical to the original Python loops,
        but the actual matrix-vector products are executed by SciPy's optimized
        sparse linear algebra routines.
        """
        indptr = np.zeros(self.n_meas + 1, dtype=np.int64)
        indices_chunks: List[np.ndarray] = []
        data_chunks: List[np.ndarray] = []

        row = 0
        nnz = 0
        for ray in self.rays:
            for idx, ww in zip(ray.vox_idx, ray.vox_w):
                idx = np.asarray(idx, dtype=np.int32)
                ww = np.asarray(ww, dtype=np.float64)

                if idx.size:
                    finite = np.isfinite(ww) & (idx >= 0) & (idx < self.grid.nvox)
                    if np.any(finite):
                        idx = idx[finite]
                        ww = ww[finite]
                    else:
                        idx = np.empty(0, dtype=np.int32)
                        ww = np.empty(0, dtype=np.float64)

                if idx.size:
                    indices_chunks.append(idx.astype(np.int32, copy=False))
                    data_chunks.append(ww.astype(np.float64, copy=False))
                    nnz += int(idx.size)

                row += 1
                indptr[row] = nnz

        if row != self.n_meas:
            raise RuntimeError(f"Sparse A row count mismatch: row={row}, n_meas={self.n_meas}")

        if nnz > 0:
            indices = np.concatenate(indices_chunks).astype(np.int32, copy=False)
            data = np.concatenate(data_chunks).astype(np.float64, copy=False)
        else:
            indices = np.empty(0, dtype=np.int32)
            data = np.empty(0, dtype=np.float64)

        self.A_csr = csr_matrix(
            (data, indices, indptr),
            shape=(self.n_meas, self.grid.nvox),
            dtype=np.float64,
        )
        self.A_csr.sum_duplicates()
        self.A_csr.eliminate_zeros()

        print(
            f"[INFO] Built sparse forward matrix A: "
            f"shape={self.A_csr.shape}, nnz={self.A_csr.nnz}, "
            f"density={self.A_csr.nnz / max(1, self.A_csr.shape[0] * self.A_csr.shape[1]):.3e}"
        )

    def A_times_unscaled(self, x: np.ndarray) -> np.ndarray:
        """Forward projection with the physical Thomson operator before group gains."""
        x = np.asarray(x, dtype=np.float64)
        if x.size != self.grid.nvox:
            raise ValueError(
                f"A_times_unscaled expected x.size={self.grid.nvox}, got {x.size}"
            )
        ne = self.solution_to_density(x)
        return np.asarray(self.A_csr.dot(ne), dtype=np.float64).ravel()

    def A_times(self, x: np.ndarray) -> np.ndarray:
        """
        Forward projection including the current inter-instrument response gains.
        """
        return self.measurement_scale * self.A_times_unscaled(x)

    def AT_times(self, y: np.ndarray) -> np.ndarray:
        """
        Backprojection through the transpose of the gain-scaled forward operator.
        """
        y = np.asarray(y, dtype=np.float64)
        if y.size != self.n_meas:
            raise ValueError(f"AT_times expected y.size={self.n_meas}, got {y.size}")
        out = np.asarray(
            self.A_csr.T.dot(self.measurement_scale * y),
            dtype=np.float64,
        ).ravel()
        if self.density_basis is not None:
            out = self.density_basis * out
        return out

    def _data_normal_diagonal(self, chunk_rows: int = 32768) -> np.ndarray:
        """Return diag(A_eff.T W^2 A_eff) in solver-variable space.

        The calculation is chunked and cached for the current measurement gains.
        It changes only CG scaling, never the inverse problem being solved.
        """
        if (
            self._data_diagonal_cache is not None
            and self._data_diagonal_cache_version == self._measurement_scale_version
        ):
            return self._data_diagonal_cache

        diag = np.zeros(self.grid.nvox, dtype=np.float64)
        row_weight = (self.W * self.measurement_scale) ** 2
        indptr = self.A_csr.indptr
        indices = self.A_csr.indices
        data = self.A_csr.data
        chunk_rows = max(1, int(chunk_rows))
        for r0 in range(0, self.n_meas, chunk_rows):
            r1 = min(self.n_meas, r0 + chunk_rows)
            start = int(indptr[r0])
            stop = int(indptr[r1])
            if stop <= start:
                continue
            counts = np.diff(indptr[r0:r1 + 1])
            rw = np.repeat(row_weight[r0:r1], counts)
            vals = data[start:stop]
            diag += np.bincount(
                indices[start:stop],
                weights=(vals * vals) * rw,
                minlength=self.grid.nvox,
            )

        if self.density_basis is not None:
            diag *= self.density_basis * self.density_basis
        self._data_diagonal_cache = diag
        self._data_diagonal_cache_version = self._measurement_scale_version
        return diag

    def _build_diagonal_preconditioner(
        self,
        floor: Optional[float] = None,
    ) -> LinearOperator:
        """Build a positive Jacobi preconditioner for the current lambda/gains."""
        data_diag = self._data_normal_diagonal()
        if self._regularization_diagonal_cache is None:
            self._regularization_diagonal_cache = regularization_diagonal(
                self.grid, wt_r=self.wt_r
            )
        diag = data_diag + float(self.lam) * self._regularization_diagonal_cache
        positive = diag[np.isfinite(diag) & (diag > 0)]
        rel_floor = self.preconditioner_floor if floor is None else float(floor)
        rel_floor = max(float(rel_floor), 0.0)
        reference = float(np.nanmedian(positive)) if positive.size else 1.0
        absolute_floor = max(reference * rel_floor, np.finfo(np.float64).tiny)
        diag = np.where(np.isfinite(diag) & (diag > absolute_floor), diag, absolute_floor)
        inv_diag = 1.0 / diag
        return LinearOperator(
            (self.grid.nvox, self.grid.nvox),
            matvec=lambda v: inv_diag * np.asarray(v, dtype=np.float64),
            dtype=np.float64,
        )

    def solve(
        self,
        y_obs: np.ndarray,
        maxiter: int = 50,
        tol: float = 1e-4,
        positivity: bool = True,
        positivity_method: str = "clip",
        x0: Optional[np.ndarray] = None,
        use_preconditioner: Optional[bool] = None,
        preconditioner_floor: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Solve the weighted Tikhonov problem.

        positivity_method options:
          - "clip"  : solve the unconstrained normal equation with CG, then set
                        negative components to zero.  Fast, but not a strict
                        non-negative least-squares solution.
          - "lbfgsb": solve the bound-constrained quadratic objective with
                        L-BFGS-B and bounds x>=0.  More faithful to x>=0, but
                        usually slower for large tomography runs.
          - "none"  : no positivity enforcement.

        x0 can be supplied to warm-start repeated solves during gain calibration
        or a lambda continuation scan.  The optional Jacobi preconditioner uses
        the exact diagonal of the unchanged normal equation and therefore changes
        convergence speed only, not the physical objective or its solution.
        """
        W = self.W
        y_obs = np.asarray(y_obs, dtype=np.float64).ravel()
        method = str(positivity_method or "clip").strip().lower()
        if not positivity:
            method = "none"

        def matvec(v: np.ndarray) -> np.ndarray:
            Av = self.A_times(v)
            W2Av = (W * W) * Av
            lhs = self.AT_times(W2Av) + self.lam * apply_LTL(v, self.grid, wt_r=self.wt_r)
            return lhs

        b = self.AT_times((W * W) * y_obs)
        Aop = LinearOperator((self.grid.nvox, self.grid.nvox), matvec=matvec, dtype=np.float64)

        if x0 is None:
            x0_use = np.zeros(self.grid.nvox, dtype=np.float64)
        else:
            x0_use = np.asarray(x0, dtype=np.float64).ravel().copy()
            if x0_use.size != self.grid.nvox:
                raise ValueError(
                    f"x0 must have size {self.grid.nvox}, got {x0_use.size}"
                )
            x0_use = np.where(np.isfinite(x0_use), x0_use, 0.0)
            if method in ("clip", "lbfgsb", "l-bfgs-b", "strict", "bounds"):
                x0_use = np.maximum(x0_use, 0.0)

        if method in ("lbfgsb", "l-bfgs-b", "strict", "bounds"):
            if minimize is None:
                raise RuntimeError("positivity_method='lbfgsb' requires scipy.optimize.minimize.")

            def fun_and_grad(v: np.ndarray):
                Av_minus_y = self.A_times(v) - y_obs
                Wres = W * Av_minus_y
                ltlv = apply_LTL(v, self.grid, wt_r=self.wt_r)
                f = 0.5 * float(np.dot(Wres, Wres)) + 0.5 * self.lam * float(np.dot(v, ltlv))
                g = self.AT_times((W * W) * Av_minus_y) + self.lam * ltlv
                return f, g

            res = minimize(
                fun_and_grad,
                x0_use,
                method="L-BFGS-B",
                jac=True,
                bounds=[(0.0, None)] * self.grid.nvox,
                options={"maxiter": int(maxiter), "ftol": float(tol), "gtol": float(tol), "maxls": 20},
            )
            x = np.asarray(res.x, dtype=np.float64)
            info = 0 if bool(res.success) else 1
            if info != 0:
                print(f"[WARN] L-BFGS-B positivity solve did not fully converge: {res.message}")
            return x, info

        # Default fast path: preconditioned CG on the same normal equations.
        use_pc = self.use_preconditioner if use_preconditioner is None else bool(use_preconditioner)
        Mop = self._build_diagonal_preconditioner(preconditioner_floor) if use_pc else None
        iteration_counter = [0]

        def _count_iteration(_xk):
            iteration_counter[0] += 1

        try:
            x, info = cg(
                Aop, b, x0=x0_use, maxiter=maxiter, rtol=tol, atol=0.0,
                M=Mop, callback=_count_iteration,
            )
        except TypeError:
            try:
                x, info = cg(
                    Aop, b, x0=x0_use, maxiter=maxiter, tol=tol,
                    M=Mop, callback=_count_iteration,
                )
            except TypeError:
                x, info = cg(
                    Aop, b, x0=x0_use, maxiter=maxiter,
                    M=Mop, callback=_count_iteration,
                )
        self.last_solver_iterations = int(iteration_counter[0])
        self.last_solver_used_preconditioner = bool(use_pc)
        print(
            f"[SOLVER] CG iterations={self.last_solver_iterations}, info={info}, "
            f"preconditioner={'jacobi' if use_pc else 'none'}"
        )

        if method == "clip":
            x = np.maximum(x, 0.0)
        elif method in ("none", "off", "false", "0"):
            pass
        else:
            raise ValueError(f"Unknown positivity_method={positivity_method!r}; use 'clip', 'lbfgsb', or 'none'.")

        return x, info


# ----------------------------
# Lambda diagnostics
# ----------------------------
def regularization_norm_value(solution_raw: np.ndarray, grid: SphericalGrid, wt_r: Optional[np.ndarray]) -> float:
    """Return sqrt(x^T L^T L x) for the current regularization operator."""
    x = np.asarray(solution_raw, dtype=np.float64).ravel()
    ltlx = apply_LTL(x, grid, wt_r=wt_r)
    val = float(np.dot(x, ltlx))
    if not np.isfinite(val) or val < 0:
        return np.nan
    return float(np.sqrt(val))


def maybe_run_lambda_scan(
    tomo: RegularizedTomography,
    y_obs: np.ndarray,
    lambda_values,
    harmonic: int = 1,
    maxiter: int = 50,
    tol: float = 1e-4,
    positivity_method: str = "clip",
) -> list[dict]:
    """Run a lightweight lambda scan using the already-built sparse forward matrix."""
    values = [float(v) for v in (lambda_values or [])]
    if not values:
        return []
    rows = []
    old_lam = float(tomo.lam)
    print(f"[LAMBDA] Running lambda scan with values={values}")
    for lam in values:
        tomo.lam = float(lam)
        sol, info = tomo.solve(y_obs, maxiter=maxiter, tol=tol, positivity=True, positivity_method=positivity_method)
        y_pred = tomo.A_times(sol)
        wres = tomo.W * (y_pred - y_obs)
        misfit_rms = float(np.sqrt(np.mean(wres * wres))) if wres.size else np.nan
        reg_norm = regularization_norm_value(sol, tomo.grid, tomo.wt_r)
        ne = tomo.solution_to_density(sol)
        fr = frequency_range_mhz_from_ne(ne, harmonic=int(harmonic))
        if fr is None:
            fmin = fmax = np.nan
        else:
            _, _, fmin, fmax = fr
        row = {
            "lambda": float(lam),
            "cg_info": int(info),
            "weighted_misfit_rms": misfit_rms,
            "regularization_norm": reg_norm,
            "f_min_mhz": float(fmin),
            "f_max_mhz": float(fmax),
        }
        rows.append(row)
        print(
            f"[LAMBDA] lambda={lam:.6g}: misfit_rms={misfit_rms:.4e}, "
            f"reg_norm={reg_norm:.4e}, f_range={fmin:.3f}..{fmax:.3f} MHz, info={info}"
        )
    tomo.lam = old_lam
    return rows


def choose_lambda_from_scan(rows: list[dict], mode: str, fallback: float) -> float:
    """Choose lambda from a lightweight scan.  'fixed' leaves fallback unchanged."""
    mode = str(mode or "fixed").strip().lower()
    if not rows or mode in ("fixed", "manual", "none", "off"):
        return float(fallback)
    good = [r for r in rows if np.isfinite(r.get("weighted_misfit_rms", np.nan))]
    if not good:
        return float(fallback)
    if mode in ("min_misfit", "misfit"):
        return float(min(good, key=lambda r: r["weighted_misfit_rms"])["lambda"])
    if mode in ("lcurve", "corner") and len(good) >= 3:
        # Simple normalized-distance L-curve corner in log(misfit)-log(reg) space.
        xs = np.array([np.log10(r["weighted_misfit_rms"]) for r in good], dtype=float)
        ys = np.array([np.log10(r["regularization_norm"]) for r in good], dtype=float)
        lams = np.array([r["lambda"] for r in good], dtype=float)
        order = np.argsort(lams)
        xs, ys, lams = xs[order], ys[order], lams[order]
        if np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)):
            def norm(v):
                return (v - np.min(v)) / max(np.max(v) - np.min(v), 1e-300)
            p = np.c_[norm(xs), norm(ys)]
            v = p[-1] - p[0]
            vn = np.linalg.norm(v)
            if vn > 0:
                d = np.abs(np.cross(v, p - p[0])) / vn
                return float(lams[int(np.argmax(d))])
    return float(fallback)


# ============================================================
# Single-run diagnostics transplanted from main_extended_multi_tomo.py
# Parameter-comparison suite code is intentionally excluded.
# ============================================================

@dataclass
class PreparedProblem:
    args: object
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


def _default_group_forward_gains(prepared: PreparedProblem) -> Dict[str, float]:
    return {
        key: 1.0
        for key in sorted(
            {tomography_observation_group_key(Path(path)) for path in prepared.pb_paths}
        )
    }


def _resolve_cross_calibration_radial_range(
    prepared: PreparedProblem,
) -> Tuple[Optional[float], Optional[float]]:
    """Resolve the explicit or common radial range used for group-gain fitting."""
    r_min = _optional_positive_float(
        getattr(prepared.args, "cross_calibration_r_min", "")
    )
    r_max = _optional_positive_float(
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
        gains, history, seed = run_group_cross_calibration(
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
        fixed = normalize_group_float_map(
            getattr(args, "fixed_group_forward_gains", {}),
            "fixed_group_forward_gains",
        )
        gains.update(fixed)
        gains[str(args.calibration_reference_group)] = 1.0
        for key, value in gains.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid fixed forward gain for {key!r}: {value}")
        tomo.set_measurement_scale(
            build_group_measurement_scale_vector(prepared.pb_paths, tomo, gains)
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
    ne_target = ne_cm3_from_fp_mhz(float(freq_mhz), harmonic=int(harmonic))
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
    ltlx = apply_LTL(x, prepared.grid, wt_r=prepared.wt_r)
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
        group = tomography_observation_group_key(path)
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
        group_indices = [i for i, p in enumerate(prepared.pb_paths) if tomography_observation_group_key(p) == group]
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
        gain, n_used = robust_weighted_projection_scale(
            prepared.y_obs[sl][use],
            base_pred[sl][use],
            tomo.W[sl][use],
            min_count=int(prepared.args.cross_calibration_min_count),
            clip_sigma=float(prepared.args.cross_calibration_clip_sigma),
            max_clip_iter=2,
        )
        group = tomography_observation_group_key(path)
        if np.isfinite(gain) and gain > 0:
            group_values.setdefault(group, []).append(float(gain))
        rows.append({
            "pb_name": path.name,
            "pb_path": str(path),
            "group": group,
            "obs_datetime": str(parse_pb_filename_datetime(path)),
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
    groups = sorted({tomography_observation_group_key(p) for p in prepared.pb_paths})
    for group in groups:
        indices = [i for i, p in enumerate(prepared.pb_paths) if tomography_observation_group_key(p) == group]
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
            gain, n_used = robust_weighted_projection_scale(
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


def target_frequency_metrics(grid, ne: np.ndarray, freq_mhz: float, harmonic: int) -> Dict[str, float]:
    """Compute volumetric, positional, component, and extent diagnostics above a target density."""
    nr, nth, nph = grid.nr, grid.nth, grid.nph
    ne3 = np.asarray(ne, dtype=np.float64).reshape((nr, nth, nph), order="C")
    ne_target = ne_cm3_from_fp_mhz(float(freq_mhz), harmonic=int(harmonic))
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
    ltlx = apply_LTL(x, grid, wt_r=wt_r)
    val = float(np.dot(x, ltlx))
    if not np.isfinite(val) or val < 0:
        return float("nan")
    return float(np.sqrt(val))


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
            "group": tomography_observation_group_key(path),
            "obs_datetime": str(parse_pb_filename_datetime(path)),
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

        train_tomo = RegularizedTomography(
            prepared.grid,
            train_obs,
            train_rays,
            lam=float(lambdas[0]),
            wt_r=prepared.wt_r,
            density_basis=prepared.density_basis,
        )
        held_tomo = RegularizedTomography(
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
            build_group_measurement_scale_vector([held_path], held_tomo, gains)
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
                "holdout_group": tomography_observation_group_key(held_path),
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
    dated = [(i, parse_pb_filename_datetime(p)) for i, p in enumerate(prepared.pb_paths)]
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
        t = t + timedelta(seconds=block_seconds) if hasattr(base, 'timedelta') else t + __import__('datetime').timedelta(seconds=block_seconds)
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
        train_groups = {tomography_observation_group_key(prepared.pb_paths[i]) for i in train}
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
        train_tomo = RegularizedTomography(
            prepared.grid, train_obs, train_rays, lam=float(lam),
            wt_r=prepared.wt_r, density_basis=prepared.density_basis,
        )
        gains, history, seed = configure_group_cross_calibration(train_prepared, train_tomo)
        sol, info = train_tomo.solve(
            y_train, maxiter=int(prepared.args.maxiter), tol=float(prepared.args.tol),
            positivity=True, positivity_method=str(prepared.args.positivity_method), x0=seed,
        )

        for i in held:
            held_tomo = RegularizedTomography(
                prepared.grid, [prepared.obs_list[i]], [prepared.rays[i]], lam=float(lam),
                wt_r=prepared.wt_r, density_basis=prepared.density_basis,
            )
            held_tomo.set_measurement_scale(
                build_group_measurement_scale_vector([prepared.pb_paths[i]], held_tomo, gains)
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
                "holdout_group": tomography_observation_group_key(prepared.pb_paths[i]),
                "lambda": float(lam),
                "solver_info": int(info),
                "cross_calibration_iterations": len(history),
                "cross_calibration_forward_gains": json.dumps(gains, ensure_ascii=False),
                **st,
            })
        print(f"[STEP1-BLOCK] Completed block {iblock + 1}/{len(starts)}: {start}..{end}, held={len(held)}")

    write_rows_csv(output_dir / "step1_time_block_holdout.csv", rows)
    print(f"[STEP1-BLOCK] Saved: {output_dir / 'step1_time_block_holdout.csv'}")


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
        groups.setdefault(tomography_observation_group_key(path), []).append(i)

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
            fit_scales[group_key] = weighted_projection_scale(
                np.concatenate(y_obs_parts),
                np.concatenate(y_pred_parts),
                np.concatenate(w_parts),
                min_count=100,
            )

    return fit_scales


def _selected_observation_rows(prepared: PreparedProblem) -> List[Dict[str, object]]:
    return [{
        "index": i,
        "name": path.name,
        "path": str(path),
        "group": tomography_observation_group_key(path),
        "datetime": str(parse_pb_filename_datetime(path)),
        "used_pixels": int(prepared.obs_list[i].idx_map.size),
        "r_use_min": prepared.obs_r_bounds[i][0],
        "r_use_max": prepared.obs_r_bounds[i][1],
    } for i, path in enumerate(prepared.pb_paths)]


def make_final_summary_row(
    prepared: PreparedProblem,
    tomo,
    solution_raw: np.ndarray,
    ne: np.ndarray,
    y_pred: np.ndarray,
    suggested_scale: float,
    final_lambda: float,
    output_dir: Path,
    scan_results: Optional[Sequence[object]] = None,
    lcurve_candidate: Optional[float] = None,
) -> Dict[str, object]:
    """Build one scalar-rich summary row for a single tomography run."""
    output_dir = Path(output_dir)
    stats = weighted_stats(prepared.y_obs, y_pred, tomo.W)
    reg_norm = regularization_norm(solution_raw, prepared.grid, prepared.wt_r)
    solver_diag = getattr(tomo, "final_solver_diagnostics", solver_objective_diagnostics(tomo, prepared, solution_raw))
    positivity_diag = getattr(tomo, "final_positivity_diagnostics", solution_positivity_diagnostics(solution_raw))
    coverage_diag = getattr(tomo, "coverage_diagnostics", {})

    fr = frequency_range_mhz_from_ne(ne, harmonic=int(prepared.args.harmonic))
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

    scan_lambdas = [
        float(r.get("lambda", np.nan)) if isinstance(r, dict) else float(getattr(r, "lam", np.nan))
        for r in (scan_results or [])
    ]
    finite_scan = []
    for r in (scan_results or []):
        lam_value = float(r.get("lambda", np.nan)) if isinstance(r, dict) else float(getattr(r, "lam", np.nan))
        misfit_value = (
            float(r.get("weighted_misfit_rms", r.get("data_misfit_norm", np.nan)))
            if isinstance(r, dict)
            else float(getattr(r, "data_misfit_norm", np.nan))
        )
        if np.isfinite(lam_value) and np.isfinite(misfit_value):
            finite_scan.append((lam_value, misfit_value))
    scan_best_misfit_lambda = float(min(finite_scan, key=lambda item: item[1])[0]) if finite_scan else np.nan
    render_camera_time = parse_target_datetime(prepared.args.target_time)
    render_camera_lonlat = earth_view_camera_lonlat_from_target_time(prepared.args.target_time)

    calibration_rows = list(getattr(tomo, "per_image_calibration_rows", []))
    cor_rows = [r for r in calibration_rows if r.get("group") == "cor1a" and np.isfinite(finite_or_nan(r.get("fit_gain_model_to_observed")))]
    cor_vals = np.asarray([float(r["fit_gain_model_to_observed"]) for r in cor_rows], dtype=float) if cor_rows else np.array([], dtype=float)
    if cor_vals.size:
        cor_p16, cor_med, cor_p84 = np.nanpercentile(cor_vals, [16, 50, 84])
    else:
        cor_p16 = cor_med = cor_p84 = np.nan

    row: Dict[str, object] = {
        "output_dir": str(output_dir),
        "target_time": str(getattr(prepared.args, "target_time", "")),
        "render_camera_is_earth_view": True, "render_camera_mode": "forced_target_time_sub_earth",
        "render_camera_source": "sunpy.coordinates.sun.L0_B0",
        "render_camera_time_utc": render_camera_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "render_camera_lon_deg": float(render_camera_lonlat[0]), "render_camera_lat_deg": float(render_camera_lonlat[1]),
        "render_camera_lonlat_frame": "Carrington sub-Earth L0 / B0",
        "search_window_days": finite_or_nan(getattr(prepared.args, "search_window_days", np.nan)),
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


def save_summary_csv_local(
    args,
    pb_paths: List[Path],
    grid: SphericalGrid,
    obs_list: List[Observation],
    rays: List[RayBundle],
    y_obs: np.ndarray,
    wt_r: Optional[np.ndarray],
    density_basis: Optional[np.ndarray],
    obs_r_bounds: List[Tuple[float, float]],
    ybk_list: List[Tuple[np.ndarray, np.ndarray]],
    tomo: RegularizedTomography,
    solution_raw: np.ndarray,
    ne: np.ndarray,
    y_pred: np.ndarray,
    suggested_scale: float,
    lambda_scan_rows: list[dict],
    ray_cache_stats: Optional[Dict[str, object]] = None,
) -> Path:
    """Write the final scalar-rich Summary CSV without importing another script."""
    summary_path_text = str(getattr(args, "summary_csv_path", "") or "").strip()
    if summary_path_text:
        summary_path = Path(summary_path_text)
    elif getattr(args, "save_ne_npz", ""):
        npz_path = Path(args.save_ne_npz)
        summary_path = npz_path.with_name(f"{npz_path.stem}_summary.csv")
    else:
        summary_path = Path("step1_final_summary.csv")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cache_stats = dict(ray_cache_stats or {})
    prepared = PreparedProblem(
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
        ray_cache_hits=int(cache_stats.get("ray_cache_hits", 0)),
        ray_cache_memory_hits=int(cache_stats.get("ray_cache_memory_hits", 0)),
        ray_cache_disk_hits=int(cache_stats.get("ray_cache_disk_hits", 0)),
        ray_cache_misses=int(cache_stats.get("ray_cache_misses", 0)),
        ray_cache_load_failures=int(cache_stats.get("ray_cache_load_failures", 0)),
        ray_cache_disabled=bool(cache_stats.get("ray_cache_disabled", not bool(getattr(args, "use_ray_cache", True)))),
        ray_cache_keys=list(cache_stats.get("ray_cache_keys", [])),
    )

    summary_row = make_final_summary_row(
        prepared=prepared,
        tomo=tomo,
        solution_raw=solution_raw,
        ne=ne,
        y_pred=y_pred,
        suggested_scale=suggested_scale,
        final_lambda=float(args.lam),
        output_dir=summary_path.parent,
        scan_results=lambda_scan_rows,
        lcurve_candidate=None,
    )
    write_rows_csv(summary_path, [summary_row])
    print(f"[OK] Saved final summary CSV: {summary_path}")
    return summary_path


def run_transplanted_diagnostics(
    args,
    pb_paths: List[Path],
    grid: SphericalGrid,
    obs_list: List[Observation],
    rays: List[RayBundle],
    y_obs: np.ndarray,
    wt_r: Optional[np.ndarray],
    density_basis: Optional[np.ndarray],
    obs_r_bounds: List[Tuple[float, float]],
    ybk_list: List[Tuple[np.ndarray, np.ndarray]],
    tomo: RegularizedTomography,
    solution_raw: np.ndarray,
    ne: np.ndarray,
    y_pred: np.ndarray,
    suggested_scale: float,
    solver_info: int,
    lambda_scan_rows: list[dict],
    ray_cache_stats: Optional[Dict[str, object]] = None,
) -> PreparedProblem:
    """
    Save the single-run diagnostics transplanted from main_extended_multi_tomo.py.

    Parameter-comparison scenarios and comparison-suite aggregation are intentionally
    excluded.  The already-computed tomography solution is reused; this function
    does not repeat the final inversion.
    """
    output_text = str(getattr(args, "step1_output_dir", "") or "").strip()
    if output_text:
        output_dir = ensure_dir(output_text)
    elif getattr(args, "save_ne_npz", ""):
        npz_path = Path(args.save_ne_npz)
        output_dir = ensure_dir(npz_path.with_name(f"{npz_path.stem}_diagnostics"))
    else:
        output_dir = ensure_dir("step1_diagnostics")

    cache_stats = dict(ray_cache_stats or {})
    prepared = PreparedProblem(
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
        ray_cache_hits=int(cache_stats.get("ray_cache_hits", 0)),
        ray_cache_memory_hits=int(cache_stats.get("ray_cache_memory_hits", 0)),
        ray_cache_disk_hits=int(cache_stats.get("ray_cache_disk_hits", 0)),
        ray_cache_misses=int(cache_stats.get("ray_cache_misses", 0)),
        ray_cache_load_failures=int(cache_stats.get("ray_cache_load_failures", 0)),
        ray_cache_disabled=bool(cache_stats.get("ray_cache_disabled", not bool(getattr(args, "use_ray_cache", True)))),
        ray_cache_keys=list(cache_stats.get("ray_cache_keys", [])),
    )

    tomo.final_solver_info = int(solver_info)
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
        f"method={getattr(args, 'positivity_method', 'clip')}, "
        f"negative_fraction={positivity_diag['solution_negative_voxel_fraction']:.4e}, "
        f"zero_fraction={positivity_diag['solution_zero_voxel_fraction']:.4e}, "
        f"min={positivity_diag['solution_minimum']:.4e}, "
        f"max={positivity_diag['solution_maximum']:.4e}"
    )

    write_rows_csv(output_dir / "step1_selected_observations.csv", _selected_observation_rows(prepared))

    image_rows = per_image_residual_rows(prepared, tomo, solution_raw, y_pred)
    write_rows_csv(output_dir / "step1_per_image_residuals.csv", image_rows)
    write_rows_csv(output_dir / "step1_group_residuals.csv", group_residual_rows(image_rows))
    radial_rows = radial_residual_rows(
        prepared,
        tomo,
        y_pred,
        radial_bins=getattr(args, "step1_radial_residual_bins", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
    )
    write_rows_csv(output_dir / "step1_radial_residuals.csv", radial_rows)

    if bool(getattr(args, "step1_save_calibration_diagnostics", True)):
        calibration_rows = per_image_calibration_rows(prepared, tomo, solution_raw)
        calibration_radial_rows = calibration_radial_gain_rows(
            prepared,
            tomo,
            solution_raw,
            radial_bins=getattr(args, "step1_calibration_radial_bins", [1.7, 2.0, 2.5, 3.0, 3.5]),
        )
        write_rows_csv(output_dir / "step1_per_image_calibration.csv", calibration_rows)
        write_rows_csv(output_dir / "step1_calibration_by_radius.csv", calibration_radial_rows)
        tomo.per_image_calibration_rows = calibration_rows
        tomo.calibration_radial_rows = calibration_radial_rows
    else:
        tomo.per_image_calibration_rows = []
        tomo.calibration_radial_rows = []

    coverage_stats: Dict[str, float] = {}
    if bool(getattr(args, "step1_compute_coverage", False)):
        coverage_path = (
            output_dir / "step1_coverage_map.npz"
            if bool(getattr(args, "step1_save_coverage_npz", False))
            else None
        )
        coverage_stats = compute_coverage_diagnostics(prepared, tomo, coverage_path)
        freq_list = list(args.freq_mhz_list) if args.freq_mhz_list is not None else [float(args.freq_mhz)]
        for freq in freq_list:
            mask = target_frequency_mask(grid, ne, float(freq), int(args.harmonic)).ravel()
            cov = np.asarray(tomo.coverage_relative).ravel()
            prefix = f"coverage_target_{float(freq):g}MHz_"
            coverage_stats[prefix + "median_relative"] = float(np.nanmedian(cov[mask])) if np.any(mask) else np.nan
            coverage_stats[prefix + "p16_relative"] = float(np.nanpercentile(cov[mask], 16)) if np.any(mask) else np.nan
            coverage_stats[prefix + "low_fraction"] = (
                float(np.mean(cov[mask] < float(args.step1_coverage_low_relative_threshold)))
                if np.any(mask) else np.nan
            )
    tomo.coverage_diagnostics = coverage_stats

    if bool(getattr(args, "step1_save_target_mask_npz", False)):
        masks = {}
        freq_list = list(args.freq_mhz_list) if args.freq_mhz_list is not None else [float(args.freq_mhz)]
        for freq in freq_list:
            masks[f"mask_{float(freq):g}MHz"] = target_frequency_mask(
                grid, ne, float(freq), int(args.harmonic)
            ).astype(np.uint8)
        np.savez_compressed(
            output_dir / "step1_target_frequency_masks.npz",
            **masks,
            r_edges=grid.r_edges.astype(np.float32),
            th_edges=grid.th_edges.astype(np.float32),
            ph_edges=grid.ph_edges.astype(np.float32),
        )

    save_residual_maps(
        prepared,
        tomo,
        y_pred,
        output_dir=output_dir / "residual_maps",
        save_npy=bool(getattr(args, "step1_save_residual_npy", True)),
        save_png=bool(getattr(args, "step1_save_residual_png", False)),
    )

    if bool(getattr(args, "step1_run_leave_one_image_out", False)):
        loo_lambdas = [float(args.lam)]
        if not bool(getattr(args, "step1_loo_final_lambda_only", True)):
            loo_lambdas = [float(v) for v in getattr(args, "lambda_scan_values", [])] or loo_lambdas
        run_leave_one_image_out(
            prepared,
            lambdas=loo_lambdas,
            output_dir=output_dir,
            max_holdouts=getattr(args, "step1_loo_max_holdouts", None),
        )

    if bool(getattr(args, "step1_run_block_holdout", False)):
        run_time_block_holdout(
            prepared,
            float(args.lam),
            output_dir,
            block_days=float(getattr(args, "step1_block_holdout_days", 1.0)),
            max_blocks=getattr(args, "step1_block_holdout_max_blocks", 6),
        )

    if bool(getattr(args, "save_summary_csv", False)):
        save_summary_csv_local(
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
            tomo=tomo,
            solution_raw=solution_raw,
            ne=ne,
            y_pred=y_pred,
            suggested_scale=suggested_scale,
            lambda_scan_rows=lambda_scan_rows,
            ray_cache_stats=cache_stats,
        )

    print(f"[STEP1] Single-run diagnostics directory: {output_dir}")
    return prepared


# ----------------------------
# Visualization (GUI)
# ----------------------------
def visualize_isosurface(
    grid: SphericalGrid,
    ne: np.ndarray,
    iso_freqs_mhz,
    harmonic: int = 1,
    show_sun: bool = True,
    opacity: float = 0.5,
    camera_lonlat: Optional[Tuple[float, float]] = None,
    show_gui: bool = True,
    save_png: bool = False,
    png_path: Optional[Path] = None,
    colors: Optional[List[str]] = None,
):
    """
    Render all requested equal-plasma-frequency surfaces in one 3-D plot.

    Important behavior:
      - Every valid frequency in ``iso_freqs_mhz`` is contoured on the same
        PyVista plotter, so multiple requested frequencies are overlaid in one
        three-dimensional scene.
      - When multiple frequencies are requested, a qualitative high-contrast
        palette is assigned automatically so that adjacent surfaces do not use
        duplicate or same-family colors. A user-supplied color is retained only
        for the single-frequency case.
      - The requested frequency is never silently replaced by the nearest
        reconstructed frequency. If it is outside the reconstructed density
        range, the contour is skipped with a clear warning.
      - The contour scalar is the derived plasma frequency field itself. This
        makes the rendered surface a true equal-frequency surface rather than a
        nearest-density fallback surface.
      - If ``save_png=True``, a PNG is still written even when no contour exists,
        so the diagnostic warning text is preserved in the output image.
    """

    if np.isscalar(iso_freqs_mhz):
        freq_list = [float(iso_freqs_mhz)]
    else:
        freq_list = [float(f) for f in list(iso_freqs_mhz)]

    if not freq_list:
        raise ValueError("iso_freqs_mhz must contain at least one frequency.")

    # Qualitative palette ordered so that adjacent entries are deliberately from
    # different hue families. The common two-frequency case therefore uses
    # gold and deepskyblue. This avoids the previous behavior in which a
    # one-element ISO_COLORS list was repeated for every requested surface.
    auto_palette = [
        "gold",
        "deepskyblue",
        "magenta",
        "limegreen",
        "orangered",
        "purple",
        "turquoise",
        "sienna",
        "black",
        "hotpink",
    ]

    if len(freq_list) == 1 and colors:
        selected_colors = [str(colors[0])]
    else:
        if colors is not None:
            print(
                "[INFO] Multiple frequencies were requested; ISO_COLORS is "
                "replaced by the automatic high-contrast palette to prevent "
                "duplicate or same-family colors."
            )

        if len(freq_list) <= len(auto_palette):
            selected_colors = auto_palette[:len(freq_list)]
        else:
            repeats = (
                len(freq_list) + len(auto_palette) - 1
            ) // len(auto_palette)

            selected_colors = (
                auto_palette * repeats
            )[:len(freq_list)]

            print(
                f"[WARN] {len(freq_list)} isosurfaces exceed the "
                f"{len(auto_palette)}-color palette; some colors must be reused."
            )

    print(f"[INFO] Isosurface colors: {selected_colors}")

    if png_path is None:
        png_path = Path("tomo_isosurface.png")

    png_path = Path(png_path)

    nr = grid.nr
    nth = grid.nth
    nph = grid.nph

    rr, tt, pp = grid.voxel_centers_sph()

    ne3 = np.asarray(
        ne,
        dtype=np.float64,
    ).reshape(
        (nr, nth, nph),
        order="C",
    )

    # Close the periodic phi boundary to reduce seam artifacts.
    pp2 = np.concatenate(
        [
            pp,
            pp[:, :, :1] + 2.0 * np.pi,
        ],
        axis=2,
    )

    rr2 = np.concatenate(
        [
            rr,
            rr[:, :, :1],
        ],
        axis=2,
    )

    tt2 = np.concatenate(
        [
            tt,
            tt[:, :, :1],
        ],
        axis=2,
    )

    ne2 = np.concatenate(
        [
            ne3,
            ne3[:, :, :1],
        ],
        axis=2,
    )

    xx = rr2 * np.sin(tt2) * np.cos(pp2)
    yy = rr2 * np.sin(tt2) * np.sin(pp2)
    zz = rr2 * np.cos(tt2)

    # Frequency field corresponding to the reconstructed electron density.
    # Non-positive or invalid density cannot define a plasma frequency.
    fp2 = np.full_like(
        ne2,
        np.nan,
        dtype=np.float64,
    )

    good2 = np.isfinite(ne2) & (ne2 > 0)

    fp2[good2] = (
        harmonic
        * 8980.0
        * np.sqrt(ne2[good2])
        / 1.0e6
    )

    sg = pv.StructuredGrid(
        xx,
        yy,
        zz,
    )

    sg["ne"] = ne2.ravel(order="F")
    sg["fp_mhz"] = fp2.ravel(order="F")

    fr = frequency_range_mhz_from_ne(
        ne,
        harmonic=harmonic,
    )

    has_ne = fr is not None

    if has_ne:
        ne_min, ne_max, flo, fhi = fr
    else:
        ne_min = np.nan
        ne_max = np.nan
        flo = np.nan
        fhi = np.nan

    p = pv.Plotter(
        off_screen=(not show_gui),
    )

    p.set_background("white")

    if show_sun:
        p.add_mesh(
            pv.Sphere(
                radius=1.0,
                theta_resolution=60,
                phi_resolution=60,
            ),
            opacity=0.15,
            color="gray",
        )

    if camera_lonlat is not None:
        lon = np.deg2rad(camera_lonlat[0])
        lat = np.deg2rad(camera_lonlat[1])

        cam_dir = np.array(
            [
                np.cos(lat) * np.cos(lon),
                np.cos(lat) * np.sin(lon),
                np.sin(lat),
            ],
            dtype=np.float64,
        )

        cam_pos = (
            cam_dir * 15.0
        ).tolist()

        p.camera_position = [
            cam_pos,
            [0, 0, 0],
            [0, 0, 1],
        ]

    legend_entries = []
    any_mesh = False

    if has_ne:
        for f, col in zip(
            freq_list,
            selected_colors,
        ):
            f = float(f)

            ne_iso = ne_cm3_from_fp_mhz(
                f,
                harmonic=harmonic,
            )

            print(
                f"[INFO] Requested isosurface: "
                f"f={f:.3f} MHz, "
                f"harmonic={harmonic}, "
                f"ne={ne_iso:.3e} cm^-3, "
                f"color={col}"
            )

            if (
                not np.isfinite(f)
                or f < flo
                or f > fhi
            ):
                print(
                    f"[WARN] Requested f={f:.3f} MHz is outside "
                    f"reconstructed range {flo:.3f}..{fhi:.3f} MHz; "
                    "no nearest-frequency fallback is used."
                )
                continue

            contours = sg.contour(
                isosurfaces=[f],
                scalars="fp_mhz",
            )

            if contours.n_points == 0:
                print(
                    f"[WARN] Empty contour for f={f:.3f} MHz "
                    f"(ne={ne_iso:.3e} cm^-3)."
                )
                continue

            p.add_mesh(
                contours,
                color=col,
                opacity=float(opacity),
                name=f"isosurface_{f:g}MHz",
            )

            legend_entries.append(
                [
                    f"f={f:g} MHz (H={harmonic})",
                    col,
                ]
            )

            any_mesh = True

    if harmonic == 1:
        f_label = "Fundamental"
    else:
        f_label = "Second Harmonic"

    if not any_mesh:
        p.add_text(
            "No requested isosurface rendered.\n"
            "Check density scale, r_use_min/max, pb_floor, "
            "and requested frequency.",
            position="upper_left",
            font_size=12,
            color="black",
        )

    if (
        has_ne
        and np.isfinite(flo)
        and np.isfinite(fhi)
    ):
        p.add_text(
            f"Reconstructed f-range: "
            f"{flo:.2f} .. {fhi:.2f} MHz "
            f"({f_label})",
            position="lower_left",
            font_size=10,
            color="black",
        )

        p.add_text(
            f"ne range: "
            f"{ne_min:.3e} .. {ne_max:.3e} cm^-3",
            position="lower_right",
            font_size=10,
            color="black",
        )

    if any_mesh and legend_entries:
        try:
            p.add_legend(
                legend_entries,
                bcolor="white",
                border=True,
            )
        except Exception as exc:
            print(
                f"[WARN] Could not add isosurface legend: {exc}"
            )

    if save_png:
        png_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        p.show(
            screenshot=str(png_path),
            auto_close=True,
        )
    else:
        p.show()

# ----------------------------
# Observation builder (SSC prep analog)
# ----------------------------
def build_observation(
    pb_fits: Path,
    out_n: int,
    r_use_min: float,
    r_use_max: float,
    limb_u: float,
    limb_u_mode: str = "instrument_bandpass",
    limb_u_use_allen: bool = False,
    limb_u_bandpass_nm_by_instrument=None,
    limb_u_override_by_instrument=None,
    limb_u_bandpass_samples: int = 401,
    limb_u_weight_hdu_names=DEFAULT_LIMB_WEIGHT_HDU_NAMES,
    thomson_normalize_msb: bool = True,
    pb_override: Optional[np.ndarray] = None,
    apply_spatial_despike: bool = True,
    filt: bool = False,
    despike_nsig: float = 6.0,
    despike_med: int = 3,
    pb_floor: float | str = 1e-13,
    dpa_deg: float = 3.0,
    hm: int = 3,
    width_pix: int = 0,
    q_low: float = 10.0,
    lonlat_override: Optional[Tuple[float, float]] = None,
    lonlat_default: Optional[Tuple[float, float]] = None,
    save_prepped_dir: Optional[Path] = None,
) -> Observation:
    """
    Load one pB FITS, apply SSC/IDL-like preprocessing (rebin, optional despike, NaN-fix),
    derive per-pixel weights from an azimuthally sampled background profile, and build the
    camera geometry needed by the forward model.

    NOTE: pb_floor may be given as a float or as a string (including "" meaning "auto").
          This function sanitizes it BEFORE any numerical comparisons to avoid dtype errors.
    """
    print(f"Reading {pb_fits}...")
    pb0, hdr = read_fits_image(pb_fits)
    if pb0.shape[0] != pb0.shape[1]:
        raise ValueError(f"{pb_fits} is not square: {pb0.shape}")
    orig_n = pb0.shape[0]

    def _parse_pb_floor(val) -> Optional[float]:
        try:
            if isinstance(val, str):
                s = val.strip()
                if s == "":
                    return None
                x = float(s)
            else:
                x = float(val)
            if not np.isfinite(x) or x <= 0:
                return None
            return x
        except Exception:
            return None

    def _estimate_pb_noise(arr: np.ndarray) -> float:
        v = arr[np.isfinite(arr) & (arr > 0)]
        if v.size < 200:
            return 1e-30
        vmax = float(np.nanpercentile(v, 99.5))
        if not np.isfinite(vmax) or vmax <= 0:
            return 1e-30

        binsize = 0.2e-10
        nbins = int(max(50, min(2000, np.ceil(vmax / binsize))))
        hist, edges = np.histogram(v, bins=nbins, range=(0.0, nbins * binsize))
        if hist.size == 0:
            return 1e-30
        k = int(np.argmax(hist))
        pb_noise = 0.5 * (edges[k] + edges[k + 1])
        if not np.isfinite(pb_noise) or pb_noise <= 0:
            return 1e-30
        return float(pb_noise)

    pb_floor_user = _parse_pb_floor(pb_floor)

    if pb_override is not None:
        pb = pb_override.astype(np.float64, copy=False)
        if pb.shape != (out_n, out_n):
            raise ValueError(f"pb_override must be shape {(out_n, out_n)}, got {pb.shape}")
    else:
        # preview_data.pro applies fix_nan before temporal filtering/rebinning.
        # For single-image fallback preprocessing, repair isolated NaNs before FREBIN-like rebinning.
        pb0_for_rebin = fill_nan_by_neighbor_mean(pb0, max_passes=1) if filt else pb0
        pb = rebin_idl_linear(pb0_for_rebin, out_n) if orig_n != out_n else pb0_for_rebin.copy()
    pb = np.asarray(pb, dtype=np.float64)

    # x/y in Rsun (critical)
    x_map, y_map, rsun_arcsec = xy_rsun_for_rebinned_image(hdr, orig_n=orig_n, out_n=out_n)
    rho = np.hypot(x_map, y_map)  # Rsun

    (
        limb_component_names,
        limb_u_values,
        limb_coeff_a,
        limb_coeff_b,
        limb_component_weight_maps,
        limb_u_model,
    ) = build_observation_limb_kernel_model(
        pb_fits=Path(pb_fits),
        hdr=hdr,
        rho=rho,
        orig_n=int(orig_n),
        out_n=int(out_n),
        fallback_u=float(limb_u),
        mode=str(limb_u_mode),
        normalize_msb=bool(thomson_normalize_msb),
        use_allen=bool(limb_u_use_allen),
        bandpass_nm_by_instrument=limb_u_bandpass_nm_by_instrument,
        u_override_by_instrument=limb_u_override_by_instrument,
        bandpass_samples=int(limb_u_bandpass_samples),
        weight_hdu_names=limb_u_weight_hdu_names,
    )

    mask = (rho >= r_use_min) & (rho <= r_use_max) & np.isfinite(pb)
    if not np.any(mask):
        rmin = float(np.nanmin(rho)) if np.any(np.isfinite(rho)) else np.nan
        rmax = float(np.nanmax(rho)) if np.any(np.isfinite(rho)) else np.nan
        raise ValueError(
            f"No valid pB pixels within r_use=[{r_use_min},{r_use_max}] Rsun for {pb_fits.name}. "
            f"rho range (Rsun) ~ {rmin:.3f}..{rmax:.3f}. "
            f"Check r_use_min/max and ensure x/y are in Rsun (rsun_arcsec={rsun_arcsec:.2f})."
        )

    if filt and apply_spatial_despike:
        pb = despike_pb_map(pb, mask=mask, med_size=despike_med, nsig=despike_nsig, use_log=True)

        pb_noise_pre = _estimate_pb_noise(pb[mask] if np.any(mask) else pb)
        pb_floor_clip = pb_floor_user if pb_floor_user is not None else pb_noise_pre

        pb = np.where(pb > pb_floor_clip, pb, np.nan)
        pb = fill_nan_by_neighbor_mean(pb, max_passes=10)
        mask = (
            (rho >= r_use_min) & (rho <= r_use_max)
            & np.isfinite(pb)
            & np.isfinite(x_map) & np.isfinite(y_map) & np.isfinite(rho)
        )

        if not np.any(mask):
            raise ValueError(
                f"All pixels became invalid after despike/threshold for {pb_fits.name}. "
                f"Consider relaxing despike_nsig/med or adjusting pb_floor."
            )

    rgrid, ybk, pb_noise = ybk_profile_fft(
        pb=pb, hdr=hdr, rmin=r_use_min, rmax=r_use_max,
        dpa_deg=dpa_deg, nr=240, hm=hm, width_pix=width_pix, q_low=q_low
    )

    pb_floor_val = pb_floor_user if pb_floor_user is not None else float(pb_noise)
    if not np.isfinite(pb_floor_val) or pb_floor_val <= 0:
        pb_floor_val = float(pb_noise) if (np.isfinite(pb_noise) and pb_noise > 0) else 1e-30
    floor = max(pb_floor_val, float(pb_noise) if (np.isfinite(pb_noise) and pb_noise > 0) else 1e-30)

    ybk_pix = np.interp(rho[mask], rgrid, ybk)
    ybk_pix = np.where(np.isfinite(ybk_pix) & (ybk_pix > 0), ybk_pix, floor)

    w = 1.0 / np.maximum(ybk_pix, floor)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

    # mask に対応するフラット添字
    idx_map = np.flatnonzero(mask.ravel())

    # w==0 の点は “行ごと除外” して idx_map と w を必ず同じ長さに保つ
    keep = (w > 0)
    if np.any(~keep):
        idx_map = idx_map[keep]
        w = w[keep]

        # mask も idx_map と一致するよう再構成（後段が mask を参照しても破綻しない）
        mask2 = np.zeros(mask.size, dtype=bool)
        mask2[idx_map] = True
        mask = mask2.reshape(mask.shape)

    lonlat_deg = None
    if lonlat_override is not None:
        lonlat_deg = lonlat_override
    else:
        lon = hdr.get("CRLN_OBS", hdr.get("HGLN_OBS", hdr.get("CRLN", None)))
        lat = hdr.get("CRLT_OBS", hdr.get("HGLT_OBS", hdr.get("CRLT", None)))
        if lon is not None and lat is not None:
            try:
                lonlat_deg = (float(lon), float(lat))
            except Exception:
                lonlat_deg = None
        if lonlat_deg is None and lonlat_default is not None:
            lonlat_deg = lonlat_default

    # Build camera basis vectors in Carrington coordinates.
    # cam_z points from observer to Sun center; cam_y is north projected on the plane of sky;
    # cam_x completes right-handed set (approx. solar west).
    lonlat_for_cam = lonlat_deg if lonlat_deg is not None else (0.0, 0.0)
    lon_rad = np.deg2rad(lonlat_for_cam[0])
    lat_rad = np.deg2rad(lonlat_for_cam[1])
    obs_vec = np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    ], dtype=float)
    norm_obs = np.linalg.norm(obs_vec)
    if norm_obs <= 0:
        obs_vec = np.array([1.0, 0.0, 0.0], dtype=float)
        norm_obs = 1.0
    obs_vec /= norm_obs

    cam_z = -obs_vec
    north = np.array([0.0, 0.0, 1.0], dtype=float)
    cam_y_tmp = north - np.dot(north, cam_z) * cam_z
    norm_y = np.linalg.norm(cam_y_tmp)
    if norm_y <= 0:
        cam_y = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        cam_y = cam_y_tmp / norm_y
    cam_x = np.cross(cam_y, cam_z)
    norm_x = np.linalg.norm(cam_x)
    if norm_x <= 0:
        cam_x = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        cam_x = cam_x / norm_x

    if save_prepped_dir is not None:
        save_prepped_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_prepped_dir / f"{pb_fits.stem}_pb.npy", pb.astype(np.float32))
        np.save(save_prepped_dir / f"{pb_fits.stem}_mask.npy", mask.astype(np.uint8))
        np.save(save_prepped_dir / f"{pb_fits.stem}_weights.npy", w.astype(np.float32))
        np.save(save_prepped_dir / f"{pb_fits.stem}_rho_rsun.npy", rho.astype(np.float32))
        np.save(save_prepped_dir / f"{pb_fits.stem}_x_rsun.npy", x_map.astype(np.float32))
        np.save(save_prepped_dir / f"{pb_fits.stem}_y_rsun.npy", y_map.astype(np.float32))
        np.save(save_prepped_dir / f"{pb_fits.stem}_ybk_r.npy", np.vstack([rgrid, ybk]).astype(np.float32))
        if limb_component_weight_maps is not None:
            np.save(
                save_prepped_dir / f"{pb_fits.stem}_limb_component_weights.npy",
                np.asarray(limb_component_weight_maps, dtype=np.float32),
            )

    return Observation(
        pb=pb,
        hdr=hdr,
        x=x_map,
        y=y_map,
        mask=mask,
        w=w,
        idx_map=idx_map,
        cam_x=cam_x,
        cam_y=cam_y,
        cam_z=cam_z,
        lonlat_deg=lonlat_deg,
        limb_component_names=limb_component_names,
        limb_u_values=limb_u_values,
        limb_coeff_a=limb_coeff_a,
        limb_coeff_b=limb_coeff_b,
        limb_component_weight_maps=limb_component_weight_maps,
        limb_u_model=limb_u_model,
    )


def choose_nr_for_measurement_to_voxel_ratio(
    n_measurements: int,
    max_m_over_n: float = 1.5,
) -> Tuple[int, int, int, float]:
    """
    Choose integer tomography dimensions from the final number of measurements.

    NTH is fixed to 60 and NPH is fixed to 2 * NTH.  NR is the smallest
    positive integer satisfying

        M / (NR * NTH * NPH) <= max_m_over_n.

    Therefore the achieved M/N is as close as possible to the requested upper
    limit without exceeding it.
    """
    n_measurements = int(n_measurements)
    max_m_over_n = float(max_m_over_n)
    nth = 60
    nph = 2 * nth

    if n_measurements <= 0:
        raise ValueError(
            f"n_measurements must be positive, got {n_measurements}."
        )
    if not np.isfinite(max_m_over_n) or max_m_over_n <= 0:
        raise ValueError(
            "max_m_over_n must be positive and finite, "
            f"got {max_m_over_n}."
        )

    nr = max(
        1,
        int(
            np.ceil(
                n_measurements
                / (max_m_over_n * nth * nph)
            )
        ),
    )
    n_voxels = nr * nth * nph
    achieved_m_over_n = n_measurements / float(n_voxels)

    if achieved_m_over_n > max_m_over_n + 1.0e-12:
        raise RuntimeError(
            "Automatic grid selection failed to satisfy M/N <= target: "
            f"M={n_measurements}, grid={nr}x{nth}x{nph}, "
            f"M/N={achieved_m_over_n:.12g}, target={max_m_over_n:.12g}."
        )

    return nr, nth, nph, float(achieved_m_over_n)


def main(args):
    """
    Run SSC/Ne3dTomo-like preprocessing + regularized tomography WITHOUT argparse.
    Edit the parameters in the `if __name__ == "__main__"` block at the bottom.
    """

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
        limb_u=DEFAULT_LIMB_U,
        limb_u_mode="instrument_bandpass",
        limb_u_use_allen=False,
        limb_u_bandpass_nm_by_instrument=dict(DEFAULT_LIMB_BANDPASS_NM),
        limb_u_override_by_instrument={},
        limb_u_bandpass_samples=401,
        limb_u_weight_hdu_names=DEFAULT_LIMB_WEIGHT_HDU_NAMES,

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

        step1_run_diagnostics=True,
        step1_output_dir="",
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

        show_gui=True,
        freq_mhz=25.0,
        freq_mhz_list=None,
        harmonic=1,
        iso_colors=None,
        save_png=True,
        png_path="",
    )

    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    if bool(getattr(args, "run_pb_unit_diagnostics", False)):
        diag_paths = [Path(p) for p in getattr(args, "pb_diagnostic_paths", [])]
        print_pb_calibration_report(diag_paths)

    if not bool(getattr(args, "use_density_prior", False)):
        args.density_prior_model = "none"

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
        found = find_pb_fits_in_time_window(
            data_dir=Path(args.data_dir),
            target_time=args.target_time,
            window_days=float(args.search_window_days),
            include_kcor_lasco=bool(args.include_kcor_lasco),
            include_cor1a=bool(args.include_cor1a),
            include_lasco_only=bool(args.include_lasco_only),
            cor1a_data_dir=(
                args.cor1a_data_dir
                if getattr(args, "cor1a_data_dir", "")
                else None
            ),
            match_cor1a_to_earth=True,
            max_cor1a_match_minutes=90.0,
            exclude_earth_times=getattr(
                args,
                "exclude_earth_times",
                [],
            ),
            keep_cor1a_for_excluded_earth_times=bool(
                getattr(
                    args,
                    "keep_cor1a_for_excluded_earth_times",
                    True,
                )
            ),
        )
        if bool(args.deduplicate_pb_fits):
            found = deduplicate_tomography_pb_paths(found, verbose=True)
        args.pb_fits = [str(p) for p in found]
        print(f"[INFO] Tomography-ready pB files selected: {len(args.pb_fits)}")
        for path in args.pb_fits:
            print(f"       {path}")

    if not args.pb_fits:
        raise ValueError(
            "pb_fits is empty. Run prepare_kcor_lasco_pb.py first if needed, then set "
            "PB_FITS manually or set AUTO_FIND_PB_FITS=True in the __main__ block."
        )

    default_lonlat = None
    if args.default_lonlat:
        a, b = args.default_lonlat.split(",")
        default_lonlat = (float(a), float(b))

    lonlat_map = {}
    if args.lonlat_file:
        fp = Path(args.lonlat_file)
        if not fp.exists():
            raise FileNotFoundError(fp)
        import csv
        with fp.open("r", newline="") as f:
            for row in csv.reader(f):
                if not row or row[0].strip().startswith("#") or len(row) < 3:
                    continue
                lonlat_map[row[0].strip()] = (float(row[1]), float(row[2]))

    pb_paths = [Path(p) for p in args.pb_fits]
    for p in pb_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    r_use_min_by_group = normalize_group_float_map(args.r_use_min_by_group, "r_use_min_by_group")
    r_use_max_by_group = normalize_group_float_map(args.r_use_max_by_group, "r_use_max_by_group")
    pb_scale_by_group = normalize_group_float_map(args.pb_scale_by_group, "pb_scale_by_group")

    pb_overrides = {}
    if args.filt and len(pb_paths) >= 2 and bool(args.use_temporal_despike):
        pb_overrides = build_ne3dtomo_temporal_despike_overrides(
            pb_paths=pb_paths,
            out_n=int(args.out_n),
            nsig=float(args.despike_nsig),
        )
        if not pb_overrides:
            print("[INFO] Temporal despike requested, but no homogeneous group was usable; applying spatial despike per image only.")
    elif args.filt and len(pb_paths) >= 2:
        print("[INFO] Global temporal despike disabled; applying spatial despike per image only.")

    save_prepped_dir = Path(args.save_prepped_dir) if args.save_prepped_dir else None

    obs_list: List[Observation] = []
    local_ybk_list: List[Tuple[np.ndarray, np.ndarray]] = []
    obs_r_bounds: List[Tuple[float, float]] = []

    for p in pb_paths:
        group_key = tomography_observation_group_key(p)
        obs_r_use_min = float(r_use_min_by_group.get(group_key, args.r_use_min))
        obs_r_use_max = float(r_use_max_by_group.get(group_key, args.r_use_max))
        if obs_r_use_min >= obs_r_use_max:
            raise ValueError(f"Invalid r_use bounds for {p.name}: {obs_r_use_min} >= {obs_r_use_max} Rsun.")
        if obs_r_use_min < float(args.r_min) - 1e-8:
            raise ValueError(
                f"{p.name} uses r_use_min={obs_r_use_min} Rsun, smaller than "
                f"the reconstruction r_min={args.r_min} Rsun. Lower r_min or raise the observation cut."
            )
        if obs_r_use_max > float(args.r_max) + 1e-8:
            raise ValueError(
                f"{p.name} uses r_use_max={obs_r_use_max} Rsun, larger than "
                f"the reconstruction r_max={args.r_max} Rsun. Raise r_max or lower the observation cut."
            )

        obs = build_observation(
            pb_fits=p,
            out_n=args.out_n,
            pb_override=pb_overrides.get(p),
            apply_spatial_despike=(p not in pb_overrides),
            r_use_min=obs_r_use_min,
            r_use_max=obs_r_use_max,
            limb_u=args.limb_u,
            limb_u_mode=str(args.limb_u_mode),
            limb_u_use_allen=bool(args.limb_u_use_allen),
            limb_u_bandpass_nm_by_instrument=args.limb_u_bandpass_nm_by_instrument,
            limb_u_override_by_instrument=args.limb_u_override_by_instrument,
            limb_u_bandpass_samples=int(args.limb_u_bandpass_samples),
            limb_u_weight_hdu_names=args.limb_u_weight_hdu_names,
            thomson_normalize_msb=bool(args.thomson_normalize_msb),
            filt=args.filt,
            despike_nsig=args.despike_nsig,
            despike_med=args.despike_med,
            pb_floor=args.pb_floor,
            dpa_deg=args.dpa_deg,
            hm=args.hm,
            width_pix=args.width_pix,
            q_low=args.q_low,
            lonlat_override=lonlat_map.get(p.name) or lonlat_map.get(str(p)) or lonlat_map.get(p.stem),
            lonlat_default=default_lonlat,
            save_prepped_dir=save_prepped_dir,
        )

        pb_scale = float(pb_scale_by_group.get(group_key, 1.0))
        if abs(pb_scale - 1.0) > 1e-12:
            obs = scale_observation_pb(obs, pb_scale)
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
            f"pb_scale={pb_scale:.6g}, rho={np.nanmin(rho):.3f}..{np.nanmax(rho):.3f} Rs, "
            f"limb_model={obs.limb_u_model}, components={obs.limb_component_names}"
        )

        rgrid, ybk, _ = ybk_profile_fft(
            pb=obs.pb,
            hdr=obs.hdr,
            rmin=obs_r_use_min,
            rmax=obs_r_use_max,
            dpa_deg=args.dpa_deg,
            nr=240,
            hm=args.hm,
            width_pix=args.width_pix,
            q_low=args.q_low,
        )
        local_ybk_list.append((rgrid, ybk))

    if bool(getattr(args, "ne3dtomo_global_ybk", False)):
        grouped_indices: dict[str, List[int]] = {}
        for i, p in enumerate(pb_paths):
            grouped_indices.setdefault(tomography_observation_group_key(p), []).append(i)

        ybk_list: List[Tuple[np.ndarray, np.ndarray]] = [local_ybk_list[i] for i in range(len(obs_list))]
        for key, indices in grouped_indices.items():
            group_obs = [obs_list[i] for i in indices]
            group_bounds = [obs_r_bounds[i] for i in indices]
            group_rmin = group_bounds[0][0]
            group_rmax = group_bounds[0][1]
            if any((abs(b0 - group_rmin) > 1e-8 or abs(b1 - group_rmax) > 1e-8) for b0, b1 in group_bounds):
                raise ValueError(
                    f"Group {key!r} has mixed r_use bounds; global ybk requires one radial range per homogeneous observation group."
                )
            rgrid_g, ybk_g, pb_noise_g = ybk_profile_fft_stack(
                observations=group_obs,
                rmin=group_rmin,
                rmax=group_rmax,
                dpa_deg=args.dpa_deg,
                nr=240,
                hm=args.hm,
                width_pix=args.width_pix,
                q_low=args.q_low,
            )
            for i in indices:
                obs_list[i] = update_observation_weights_from_ybk(
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

    y_obs = np.concatenate(y_list) if y_list else np.array([], dtype=float)
    if y_obs.size == 0 or (not np.any(np.isfinite(y_obs))):
        raise ValueError("y_obs is empty or all-NaN. Check masks and preprocessing (r_use_min/max, pb_floor).")

    args.nr, args.nth, args.nph, achieved_m_over_n = (
        choose_nr_for_measurement_to_voxel_ratio(
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

    r_edges = np.linspace(args.r_min, args.r_max, args.nr + 1)
    th_edges = np.linspace(0.0, np.pi, args.nth + 1)
    ph_edges = np.linspace(0.0, 2.0 * np.pi, args.nph + 1)
    grid = SphericalGrid(r_edges=r_edges, th_edges=th_edges, ph_edges=ph_edges)

    density_basis = density_basis_from_grid(
        grid,
        model=str(args.density_prior_model),
        scale=float(args.density_prior_scale),
        prior_file=str(getattr(args, "density_prior_file", "")),
    )
    if density_basis is not None:
        basis_range = frequency_range_mhz_from_ne(density_basis, harmonic=int(args.harmonic))
        prior_source = str(getattr(args, "density_prior_file", "") or "")
        source_text = f", file={prior_source!r}" if prior_source else ""
        print(
            "[INFO] Density prior enabled: "
            f"model={str(args.density_prior_model)!r}, "
            f"scale={float(args.density_prior_scale):.6g}{source_text}, "
            "solving ne = prior * q."
        )
        if basis_range is not None:
            b_ne_min, b_ne_max, b_fmin, b_fmax = basis_range
            print(
                f"[INFO] Prior density range: {b_ne_min:.3e} .. {b_ne_max:.3e} cm^-3; "
                f"plasma-frequency range (harm={int(args.harmonic)}) {b_fmin:.3f} .. {b_fmax:.3f} MHz"
            )
    else:
        print("[INFO] Density prior disabled: solving absolute electron density ne.")

    rays: List[RayBundle] = []
    n_obs = len(obs_list)
    use_ray_cache = bool(getattr(args, "use_ray_cache", True))
    ray_cache_dir = str(getattr(args, "ray_cache_dir", "") or "")
    cache_hits = 0
    cache_memory_hits = 0
    cache_disk_hits = 0
    cache_misses = 0
    cache_load_failures = 0
    ray_cache_keys: List[str] = []

    if use_ray_cache and ray_cache_dir:
        ensure_dir(ray_cache_dir)
        print(f"[CACHE] Ray cache enabled: disk+memory, directory={ray_cache_dir}")
    elif use_ray_cache:
        print("[CACHE] Ray cache enabled: memory only; cache will not survive process restart.")
    else:
        print("[CACHE-DISABLED] Ray cache is disabled; every observation will rebuild its rays.")
    print("[CACHE] Note: ray cache reuses LOS/Thomson ray bundles only; every lambda still requires a solver run.")

    for i, (obs, p) in enumerate(zip(obs_list, pb_paths), start=1):
        cache_key = ray_cache_key(
            obs=obs,
            pb_path=p,
            grid=grid,
            ds_rsun=float(args.ds),
            r_min=float(args.r_min),
            r_max=float(args.r_max),
            limb_u=float(args.limb_u),
            thomson_normalize_msb=bool(args.thomson_normalize_msb),
            thomson_kernel_scale=float(args.thomson_kernel_scale),
        )
        ray_cache_keys.append(cache_key)

        ray = None
        source = None
        disk_file = None
        disk_existed = False
        memory_existed = False

        if use_ray_cache:
            memory_existed = cache_key in _RAY_MEMORY_CACHE
            if ray_cache_dir:
                disk_file = ray_cache_path(ray_cache_dir, cache_key)
                disk_existed = disk_file.exists()

            ray = load_cached_ray(cache_key, ray_cache_dir)
            if ray is not None:
                cache_hits += 1
                if memory_existed:
                    cache_memory_hits += 1
                    source = "memory"
                else:
                    cache_disk_hits += 1
                    source = "disk"

                if bool(getattr(args, "show_ray_progress", True)):
                    nonempty = sum(1 for idx in ray.vox_idx if idx.size > 0)
                    print(
                        f"[CACHE-HIT:{source.upper()}] {i}/{n_obs}: {Path(p).name} "
                        f"(non-empty rays={nonempty}/{len(ray.vox_idx)})",
                        flush=True,
                    )
            else:
                cache_misses += 1
                if disk_existed:
                    cache_load_failures += 1
                    print(
                        f"[CACHE-MISS] {i}/{n_obs}: {Path(p).name}; cache file exists "
                        f"but could not be loaded: {disk_file}",
                        flush=True,
                    )
                elif bool(getattr(args, "show_ray_progress", True)):
                    location = str(disk_file) if disk_file is not None else "memory cache"
                    print(
                        f"[CACHE-MISS] {i}/{n_obs}: {Path(p).name}; "
                        f"no matching cache entry at {location}",
                        flush=True,
                    )

        if ray is None:
            if bool(getattr(args, "show_ray_progress", True)):
                print(
                    f"[INFO] Building rays {i}/{n_obs}: {Path(p).name} "
                    f"(used pixels={obs.idx_map.size})",
                    flush=True,
                )

            ray = build_rays_for_observation(
                obs=obs,
                grid=grid,
                ds_rsun=args.ds,
                r_min=args.r_min,
                r_max=args.r_max,
                limb_u=args.limb_u,
                thomson_normalize_msb=bool(args.thomson_normalize_msb),
                thomson_kernel_scale=float(args.thomson_kernel_scale),
            )

            if use_ray_cache:
                save_cached_ray(cache_key, ray, ray_cache_dir)
                if ray_cache_dir:
                    print(
                        f"[CACHE-WRITE] {Path(p).name}: "
                        f"{ray_cache_path(ray_cache_dir, cache_key)}",
                        flush=True,
                    )

            if bool(getattr(args, "show_ray_progress", True)):
                nonempty = sum(1 for idx in ray.vox_idx if idx.size > 0)
                print(
                    f"[INFO] Finished rays {i}/{n_obs}: {Path(p).name} "
                    f"(non-empty rays={nonempty}/{len(ray.vox_idx)})",
                    flush=True,
                )

        rays.append(ray)

    if use_ray_cache:
        print(
            f"[CACHE-SUMMARY] total={n_obs}, hits={cache_hits} "
            f"(memory={cache_memory_hits}, disk={cache_disk_hits}), "
            f"misses={cache_misses}, load_failures={cache_load_failures}"
        )
        if cache_hits == 0:
            print("[WARN][CACHE-NOT-USED] No ray bundle was reused in this run; all rays were rebuilt.")
        elif cache_misses > 0:
            print("[WARN][CACHE-PARTIAL] Ray cache was used only partially.")
        else:
            print("[CACHE-OK] All ray bundles were reused from cache.")

    ray_cache_stats = {
        "ray_cache_hits": cache_hits,
        "ray_cache_memory_hits": cache_memory_hits,
        "ray_cache_disk_hits": cache_disk_hits,
        "ray_cache_misses": cache_misses,
        "ray_cache_load_failures": cache_load_failures,
        "ray_cache_disabled": not use_ray_cache,
        "ray_cache_keys": ray_cache_keys,
    }

    wt_r = None
    if args.wt_nr:
        r_cent = 0.5 * (r_edges[:-1] + r_edges[1:])
        ybks = [np.interp(r_cent, rgi, ybki) for (rgi, ybki) in ybk_list]
        ybk_mean = np.nanmean(np.stack(ybks, axis=0), axis=0)

        good = np.isfinite(ybk_mean) & (ybk_mean > 0)
        if np.count_nonzero(good) < 3:
            print("[WARN] wt_nr requested, but ybk_mean is not usable (too many NaNs). Disabling radial weighting.")
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

    tomo = RegularizedTomography(
        grid,
        obs_list,
        rays,
        lam=args.lam,
        wt_r=wt_r,
        density_basis=density_basis,
        use_preconditioner=bool(getattr(args, "solver_use_preconditioner", True)),
        preconditioner_floor=float(getattr(args, "solver_preconditioner_floor", 1e-12)),
    )

    group_forward_gains = {
        key: 1.0
        for key in sorted(
            {tomography_observation_group_key(Path(path)) for path in pb_paths}
        )
    }
    cross_calibration_history: list[dict] = []
    cross_calibration_seed: Optional[np.ndarray] = None

    cal_r_min = _optional_positive_float(
        getattr(args, "cross_calibration_r_min", "")
    )
    cal_r_max = _optional_positive_float(
        getattr(args, "cross_calibration_r_max", "")
    )
    if cal_r_min is None:
        cal_r_min = max(float(bounds[0]) for bounds in obs_r_bounds)
    if cal_r_max is None:
        cal_r_max = min(float(bounds[1]) for bounds in obs_r_bounds)
    if cal_r_min >= cal_r_max:
        raise ValueError(
            f"Cross-calibration has no common radial range: "
            f"{cal_r_min} >= {cal_r_max} Rsun"
        )

    if bool(getattr(args, "auto_cross_calibrate_groups", False)):
        print(
            "[CAL] Calibration solver settings: "
            f"maxiter={int(getattr(args, 'cross_calibration_solver_maxiter', args.maxiter))}, "
            f"tol={float(getattr(args, 'cross_calibration_solver_tol', args.tol)):.3g}, "
            f"preconditioner={bool(getattr(args, 'solver_use_preconditioner', True))}"
        )
        group_forward_gains, cross_calibration_history, cross_calibration_seed = (
            run_group_cross_calibration(
                tomo=tomo,
                pb_paths=pb_paths,
                y_obs=y_obs,
                reference_group=str(args.calibration_reference_group),
                initial_group_forward_gains=getattr(
                    args, "cross_calibration_initial_gain_by_group", {}
                ),
                max_iterations=int(args.cross_calibration_max_iterations),
                convergence_tol=float(args.cross_calibration_tolerance),
                damping=float(args.cross_calibration_damping),
                gain_min=float(args.cross_calibration_gain_min),
                gain_max=float(args.cross_calibration_gain_max),
                r_min=cal_r_min,
                r_max=cal_r_max,
                min_count=int(args.cross_calibration_min_count),
                clip_sigma=float(args.cross_calibration_clip_sigma),
                solve_maxiter=int(
                    getattr(args, "cross_calibration_solver_maxiter", args.maxiter)
                ),
                solve_tol=float(
                    getattr(args, "cross_calibration_solver_tol", args.tol)
                ),
                positivity_method=str(args.positivity_method),
                solve_use_preconditioner=bool(
                    getattr(args, "solver_use_preconditioner", True)
                ),
                solve_preconditioner_floor=float(
                    getattr(args, "solver_preconditioner_floor", 1e-12)
                ),
            )
        )
        calibration_mode = "automatic"
    else:
        fixed_group_forward_gains = normalize_group_float_map(
            getattr(args, "fixed_group_forward_gains", {}),
            "fixed_group_forward_gains",
        )
        group_forward_gains.update(fixed_group_forward_gains)
        group_forward_gains[str(args.calibration_reference_group)] = 1.0
        tomo.set_measurement_scale(
            build_group_measurement_scale_vector(
                pb_paths, tomo, group_forward_gains
            )
        )
        calibration_mode = "fixed" if fixed_group_forward_gains else "none"
        if fixed_group_forward_gains:
            print(
                "[CAL] Automatic inter-instrument calibration disabled; "
                f"fixed forward gains applied: {group_forward_gains}"
            )
        else:
            print(
                "[CAL] Automatic inter-instrument calibration disabled; "
                "all group forward gains remain 1.0."
            )

    lambda_scan_rows = maybe_run_lambda_scan(
        tomo,
        y_obs,
        getattr(args, "lambda_scan_values", []),
        harmonic=int(args.harmonic),
        maxiter=int(args.maxiter),
        tol=float(args.tol),
        positivity_method=str(args.positivity_method),
    )
    old_lam = float(args.lam)
    chosen_lam = choose_lambda_from_scan(
        lambda_scan_rows,
        getattr(args, "lambda_select_mode", "fixed"),
        old_lam,
    )
    if abs(chosen_lam - old_lam) > 1e-12:
        print(
            f"[LAMBDA] lambda_select_mode={args.lambda_select_mode!r}: "
            f"using lambda={chosen_lam:.6g} instead of {old_lam:.6g}"
        )
    args.lam = float(chosen_lam)
    tomo.lam = float(args.lam)

    if (
        bool(getattr(args, "auto_cross_calibrate_groups", False))
        and bool(
            getattr(
                args,
                "cross_calibration_recalibrate_after_lambda_selection",
                True,
            )
        )
        and abs(chosen_lam - old_lam) > 1e-12
    ):
        gains_after_lambda, history_after_lambda, cross_calibration_seed = (
            run_group_cross_calibration(
                tomo=tomo,
                pb_paths=pb_paths,
                y_obs=y_obs,
                reference_group=str(args.calibration_reference_group),
                initial_group_forward_gains=group_forward_gains,
                max_iterations=int(args.cross_calibration_max_iterations),
                convergence_tol=float(args.cross_calibration_tolerance),
                damping=float(args.cross_calibration_damping),
                gain_min=float(args.cross_calibration_gain_min),
                gain_max=float(args.cross_calibration_gain_max),
                r_min=cal_r_min,
                r_max=cal_r_max,
                min_count=int(args.cross_calibration_min_count),
                clip_sigma=float(args.cross_calibration_clip_sigma),
                solve_maxiter=int(
                    getattr(args, "cross_calibration_solver_maxiter", args.maxiter)
                ),
                solve_tol=float(
                    getattr(args, "cross_calibration_solver_tol", args.tol)
                ),
                positivity_method=str(args.positivity_method),
                solve_use_preconditioner=bool(
                    getattr(args, "solver_use_preconditioner", True)
                ),
                solve_preconditioner_floor=float(
                    getattr(args, "solver_preconditioner_floor", 1e-12)
                ),
            )
        )
        group_forward_gains = gains_after_lambda
        cross_calibration_history.extend(history_after_lambda)

    group_data_corrections = {
        key: 1.0 / value for key, value in group_forward_gains.items()
    }
    args.cross_calibration_final_forward_gains = dict(group_forward_gains)
    args.cross_calibration_final_data_corrections = dict(group_data_corrections)
    args.cross_calibration_history = list(cross_calibration_history)

    calibration_converged = bool(
        calibration_mode in ("fixed", "none")
        or (
            cross_calibration_history
            and float(cross_calibration_history[-1].get("max_abs_log_gain_change", np.inf))
            <= float(args.cross_calibration_tolerance)
        )
    )
    tomo.cross_calibration_mode = calibration_mode
    tomo.cross_calibration_forward_gains = dict(group_forward_gains)
    tomo.cross_calibration_data_corrections = dict(group_data_corrections)
    tomo.cross_calibration_history = list(cross_calibration_history)
    tomo.cross_calibration_converged = calibration_converged
    tomo.cross_calibration_seed = cross_calibration_seed

    print(f"[CAL] Calibration mode: {calibration_mode}")
    print(f"[CAL] Final forward gains (model -> observed pB): {group_forward_gains}")
    print(
        "[CAL] Equivalent data corrections (observed pB -> reference scale): "
        f"{group_data_corrections}"
    )

    solution_raw, info = tomo.solve(
        y_obs,
        maxiter=args.maxiter,
        tol=args.tol,
        positivity=True,
        positivity_method=str(args.positivity_method),
        x0=cross_calibration_seed,
        use_preconditioner=bool(getattr(args, "solver_use_preconditioner", True)),
        preconditioner_floor=float(getattr(args, "solver_preconditioner_floor", 1e-12)),
    )
    ne_raw = tomo.solution_to_density(solution_raw)

    if info != 0:
        print(f"[WARN] CG did not fully converge (info={info}). Consider stronger regularization or more images.")

    y_pred = tomo.A_times(solution_raw)
    W = tomo.W
    suggested_scale = weighted_projection_scale(y_obs, y_pred, W, min_count=100)
    print_projection_fit_diagnostic("raw/global", y_obs, y_pred, W)
    print_projection_fit_diagnostics_by_group("raw", pb_paths, tomo, y_obs, y_pred, W)
    print_group_calibration_hints(
        "raw",
        pb_paths,
        tomo,
        y_obs,
        y_pred,
        W,
        reference_group=str(args.calibration_reference_group),
    )

    raw_range = frequency_range_mhz_from_ne(ne_raw, harmonic=args.harmonic)
    if raw_range is not None:
        raw_ne_min, raw_ne_max, raw_fmin, raw_fmax = raw_range
        print(f"[INFO] Raw reconstructed density range: {raw_ne_min:.3e} .. {raw_ne_max:.3e} cm^-3")
        print(f"[INFO] Raw reconstructed plasma-frequency range (harm={args.harmonic}): {raw_fmin:.3f} .. {raw_fmax:.3f} MHz")
    else:
        print("[WARN] ne_raw has no positive finite values before scaling.")

    scale = suggested_scale if bool(args.apply_brightness_scale) else 1.0
    if bool(args.apply_brightness_scale):
        print(
            f"[INFO] Applied global brightness-scale correction: scale={scale:.6g}. "
            "This is an explicit post-fit scalar calibration; the Thomson A matrix itself is unchanged."
        )
    else:
        print(f"[INFO] Brightness-scale correction disabled; suggested scale={suggested_scale:.6g}")
        if suggested_scale > 1.0:
            expected_factor = np.sqrt(suggested_scale)
            print(
                f"[WARN] Frequency range will be smaller by about sqrt(suggested_scale)="
                f"{expected_factor:.3g}. Set APPLY_BRIGHTNESS_SCALE=True to use the fitted pB scale."
            )

    ne = ne_raw * scale
    if bool(args.apply_brightness_scale):
        print_projection_fit_diagnostic("scaled/global", y_obs, y_pred, W, scale_for_residual=scale)
        print_projection_fit_diagnostics_by_group("scaled", pb_paths, tomo, y_obs, y_pred, W, scale_for_residual=scale)

    scaled_range = frequency_range_mhz_from_ne(ne, harmonic=args.harmonic)
    if scaled_range is not None:
        ne_min, ne_max, fmin, fmax = scaled_range
        print(f"[INFO] Scaled electron-density range: {ne_min:.3e} .. {ne_max:.3e} cm^-3")
        print(f"[INFO] Reconstructed plasma-frequency range (harm={args.harmonic}): {fmin:.3f} .. {fmax:.3f} MHz")
    else:
        print("[WARN] ne has no positive finite values after scaling.")

    diagnostic_freqs = list(args.freq_mhz_list) if args.freq_mhz_list is not None else [float(args.freq_mhz)]
    for f_req in diagnostic_freqs:
        ne_req = ne_cm3_from_fp_mhz(float(f_req), harmonic=int(args.harmonic))
        print(
            f"[INFO] Requested f={float(f_req):.3f} MHz (harm={int(args.harmonic)}) "
            f"corresponds to ne={ne_req:.3e} cm^-3"
        )
        if scaled_range is not None and (ne_req < ne_min or ne_req > ne_max):
            print(
                f"[WARN] Requested density {ne_req:.3e} cm^-3 is outside the scaled "
                f"reconstruction range {ne_min:.3e}..{ne_max:.3e} cm^-3."
            )

    if bool(getattr(args, "step1_run_diagnostics", True)):
        run_transplanted_diagnostics(
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
            tomo=tomo,
            solution_raw=solution_raw,
            ne=ne,
            y_pred=y_pred,
            suggested_scale=suggested_scale,
            solver_info=info,
            lambda_scan_rows=lambda_scan_rows,
            ray_cache_stats=ray_cache_stats,
        )
    elif bool(getattr(args, "save_summary_csv", False)):
        save_summary_csv_local(
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
            tomo=tomo,
            solution_raw=solution_raw,
            ne=ne,
            y_pred=y_pred,
            suggested_scale=suggested_scale,
            lambda_scan_rows=lambda_scan_rows,
            ray_cache_stats=ray_cache_stats,
        )

    if args.save_ne_npz:
        out = Path(args.save_ne_npz)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Save enough metadata to reproduce which observations and settings were used
        # for this tomography solution.  These arrays are intentionally saved as simple
        # numeric/string arrays so the NPZ remains easy to inspect with numpy alone.
        obs_lonlat_deg = np.array(
            [
                (obs.lonlat_deg if obs.lonlat_deg is not None else (np.nan, np.nan))
                for obs in obs_list
            ],
            dtype=np.float64,
        )
        obs_group_keys = np.array(
            [tomography_observation_group_key(p) for p in pb_paths],
            dtype="U64",
        )
        obs_r_use_min = np.array([b[0] for b in obs_r_bounds], dtype=np.float64)
        obs_r_use_max = np.array([b[1] for b in obs_r_bounds], dtype=np.float64)
        obs_used_pixels = np.array([obs.idx_map.size for obs in obs_list], dtype=np.int64)
        obs_limb_models = np.array([obs.limb_u_model for obs in obs_list], dtype="U128")
        obs_limb_component_names_json = np.array(
            [json.dumps(list(obs.limb_component_names)) for obs in obs_list], dtype="U512"
        )
        obs_limb_u_values_json = np.array(
            [json.dumps([float(v) for v in obs.limb_u_values]) for obs in obs_list], dtype="U512"
        )
        obs_limb_coeff_a_json = np.array(
            [json.dumps([float(v) for v in obs.limb_coeff_a]) for obs in obs_list], dtype="U512"
        )
        obs_limb_coeff_b_json = np.array(
            [json.dumps([float(v) for v in obs.limb_coeff_b]) for obs in obs_list], dtype="U512"
        )
        freq_list_to_save = np.array(diagnostic_freqs, dtype=np.float64)
        if getattr(args, "target_time", ""):
            render_camera_time = parse_target_datetime(args.target_time)
            render_camera_lonlat = earth_view_camera_lonlat_from_target_time(args.target_time)
            render_camera_is_earth_view = True
            render_camera_time_utc = render_camera_time.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            render_camera_lonlat = (np.nan, np.nan)
            render_camera_is_earth_view = False
            render_camera_time_utc = ""

        solver_diag = getattr(tomo, "final_solver_diagnostics", {})
        positivity_diag = getattr(tomo, "final_positivity_diagnostics", {})
        coverage_diag = getattr(tomo, "coverage_diagnostics", {})

        np.savez_compressed(
            out,
            ne=ne.astype(np.float32),
            ne_raw=ne_raw.astype(np.float32),
            solution_raw=solution_raw.astype(np.float32),
            density_basis=(density_basis.astype(np.float32) if density_basis is not None else np.ones_like(ne_raw, dtype=np.float32)),
            scale_brightness=float(scale),
            suggested_scale_brightness=float(suggested_scale),
            apply_brightness_scale=bool(args.apply_brightness_scale),
            density_prior_model=str(args.density_prior_model),
            density_prior_scale=float(args.density_prior_scale),
            density_prior_file=str(getattr(args, "density_prior_file", "")),
            pb_scale_group_keys=np.array(list(pb_scale_by_group.keys()), dtype="U64"),
            pb_scale_group_values=np.array(list(pb_scale_by_group.values()), dtype=np.float64),
            r_use_min_group_keys=np.array(list(r_use_min_by_group.keys()), dtype="U64"),
            r_use_min_group_values=np.array(list(r_use_min_by_group.values()), dtype=np.float64),
            r_use_max_group_keys=np.array(list(r_use_max_by_group.keys()), dtype="U64"),
            r_use_max_group_values=np.array(list(r_use_max_by_group.values()), dtype=np.float64),
            pb_paths=np.array([str(p) for p in pb_paths], dtype="U2048"),
            pb_names=np.array([p.name for p in pb_paths], dtype="U256"),
            obs_group_keys=obs_group_keys,
            obs_lonlat_deg=obs_lonlat_deg,
            obs_r_use_min=obs_r_use_min.astype(np.float32),
            obs_r_use_max=obs_r_use_max.astype(np.float32),
            obs_used_pixels=obs_used_pixels,
            data_dir=str(args.data_dir),
            cor1a_data_dir=str(args.cor1a_data_dir),
            target_time=str(args.target_time),
            render_camera_is_earth_view=bool(render_camera_is_earth_view),
            render_camera_mode=str("forced_target_time_sub_earth" if render_camera_is_earth_view else "none"),
            render_camera_source=str("sunpy.coordinates.sun.L0_B0" if render_camera_is_earth_view else ""),
            render_camera_time_utc=str(render_camera_time_utc),
            render_camera_lon_deg=float(render_camera_lonlat[0]),
            render_camera_lat_deg=float(render_camera_lonlat[1]),
            render_camera_lonlat_deg=np.array(render_camera_lonlat, dtype=np.float64),
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
            filt=bool(args.filt),
            despike_nsig=float(args.despike_nsig),
            despike_med=int(args.despike_med),
            pb_floor=str(args.pb_floor),
            dpa_deg=float(args.dpa_deg),
            r_use_min=float(args.r_use_min),
            r_use_max=float(args.r_use_max),
            hm=int(args.hm),
            wt_nr=bool(args.wt_nr),
            lam=float(args.lam),
            q_low=float(args.q_low),
            width_pix=float(args.width_pix),
            maxiter=int(args.maxiter),
            tol=float(args.tol),
            solver_use_preconditioner=bool(getattr(args, "solver_use_preconditioner", True)),
            solver_preconditioner_floor=float(getattr(args, "solver_preconditioner_floor", 1e-12)),
            final_solver_info=int(getattr(tomo, "final_solver_info", info)),
            solver_iterations=int(getattr(tomo, "last_solver_iterations", 0)),
            solver_used_preconditioner=bool(getattr(tomo, "last_solver_used_preconditioner", False)),
            normal_equation_relative_residual=float(solver_diag.get("normal_equation_relative_residual", np.nan)),
            data_objective=float(solver_diag.get("data_objective", np.nan)),
            regularization_objective=float(solver_diag.get("regularization_objective", np.nan)),
            total_objective=float(solver_diag.get("total_objective", np.nan)),
            solution_negative_voxel_fraction=float(positivity_diag.get("solution_negative_voxel_fraction", np.nan)),
            solution_zero_voxel_fraction=float(positivity_diag.get("solution_zero_voxel_fraction", np.nan)),
            solution_minimum=float(positivity_diag.get("solution_minimum", np.nan)),
            solution_maximum=float(positivity_diag.get("solution_maximum", np.nan)),
            coverage_diagnostics_json=json.dumps(coverage_diag, ensure_ascii=False),
            positivity_method=str(args.positivity_method),
            use_density_prior=bool(args.use_density_prior),
            thomson_normalize_msb=bool(args.thomson_normalize_msb),
            thomson_kernel_scale=float(args.thomson_kernel_scale),
            lambda_scan_values=np.array([float(v) for v in getattr(args, "lambda_scan_values", [])], dtype=np.float64),
            lambda_select_mode=str(getattr(args, "lambda_select_mode", "fixed")),
            use_temporal_despike=bool(args.use_temporal_despike),
            ne3dtomo_global_ybk=bool(args.ne3dtomo_global_ybk),
            use_ray_cache=bool(getattr(args, "use_ray_cache", True)),
            ray_cache_dir=str(getattr(args, "ray_cache_dir", "")),
            calibration_reference_group=str(args.calibration_reference_group),
            auto_cross_calibrate_groups=bool(
                getattr(args, "auto_cross_calibrate_groups", False)
            ),
            cross_calibration_reference_group=str(args.calibration_reference_group),
            cross_calibration_group_keys=np.array(
                list(group_forward_gains.keys()), dtype="U64"
            ),
            cross_calibration_forward_gains=np.array(
                list(group_forward_gains.values()), dtype=np.float64
            ),
            cross_calibration_data_corrections=np.array(
                [group_data_corrections[key] for key in group_forward_gains],
                dtype=np.float64,
            ),
            cross_calibration_iterations=int(len(cross_calibration_history)),
            cross_calibration_tolerance=float(
                getattr(args, "cross_calibration_tolerance", 0.01)
            ),
            cross_calibration_damping=float(
                getattr(args, "cross_calibration_damping", 0.7)
            ),
            cross_calibration_gain_min=float(
                getattr(args, "cross_calibration_gain_min", 0.25)
            ),
            cross_calibration_gain_max=float(
                getattr(args, "cross_calibration_gain_max", 4.0)
            ),
            cross_calibration_r_min=str(
                getattr(args, "cross_calibration_r_min", "")
            ),
            cross_calibration_r_max=str(
                getattr(args, "cross_calibration_r_max", "")
            ),
            harmonic=int(args.harmonic),
            freq_mhz_list=freq_list_to_save,
            r_edges=r_edges.astype(np.float32),
            th_edges=th_edges.astype(np.float32),
            ph_edges=ph_edges.astype(np.float32),
        )
        print(f"[OK] Saved solution NPZ: {out}")

    freq_list = list(args.freq_mhz_list) if args.freq_mhz_list is not None else [float(args.freq_mhz)]

    if args.png_path:
        png_path = Path(args.png_path)
    else:
        base = Path(args.save_ne_npz).with_suffix("") if args.save_ne_npz else Path("ne3d_solution")
        tag = "_".join([f"{float(f):.2f}" for f in freq_list])
        png_path = base.parent / f"{base.name}_iso_{tag}MHz_h{int(args.harmonic)}.png"

    print("Save png to", png_path)

    if getattr(args, "target_time", ""):
        cam_ll = choose_camera_lonlat_near_target(obs_list, pb_paths, args.target_time)
    else:
        cam_ll = obs_list[0].lonlat_deg if (obs_list and obs_list[0].lonlat_deg) else None

    visualize_isosurface(
        grid=grid,
        ne=ne,
        iso_freqs_mhz=freq_list,
        harmonic=int(args.harmonic),
        show_sun=True,
        opacity=0.5,
        camera_lonlat=cam_ll,
        show_gui=bool(args.show_gui),
        save_png=bool(args.save_png),
        png_path=png_path,
        colors=getattr(args, "iso_colors", None),
    )


if __name__ == "__main__":
    from types import SimpleNamespace

    # ------------------------------------------------------------------
    # Latest tomography settings used in this conversation.
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

    # Empty string is intentional.  The updated K-Cor/LASCO FITS files now carry
    # CRLN_OBS/CRLT_OBS, so the code should not fall back to (0,0).
    DEFAULT_LONLAT = ""
    LONLAT_FILE = ""

    # ====================
    OUT_N = 256
    
    R_MIN, R_MAX = 1.5, 4.0
    NR = 0  # Valid pB measurements are counted before NR is selected automatically.
    NTH = 60 # (NTH, NPH) を固定。(60, 120)のとき約3.0度の角度分解能。NTH=30のとき約6.0度の角度分解能。
    NPH = 2 * NTH
    AUTO_GRID_MAX_M_OVER_N = 1.5 # 観測数/voxel数の最大比率。AUTO_GRID_MAX_M_OVER_N=1.5のとき、voxel数は観測数の2/3以下に制限される。

    DS = 0.02
    
    HM = 5

    WT_NR = 1 # weighting
    # WT_NR = [0] # no-weighting
    LAM = 1.0
    Q_LOW = 0.0
    WIDTH_PIX = 1.0

    MAXITER = 10000
    TOL = 1e-4
    
    DESPIKE_NSIG = 4.0
    DESPIKE_MED = 5
    
    # ====================
    
    # Physically preferred limb-darkening model.  The scalar LIMB_U is retained
    # only as a fallback for unidentified/legacy inputs.
    LIMB_U = DEFAULT_LIMB_U
    LIMB_U_MODE = "instrument_bandpass"
    LIMB_U_USE_ALLEN = False
    LIMB_U_BANDPASS_NM_BY_INSTRUMENT = dict(DEFAULT_LIMB_BANDPASS_NM)
    LIMB_U_OVERRIDE_BY_INSTRUMENT = {}
    LIMB_U_BANDPASS_SAMPLES = 401
    LIMB_U_WEIGHT_HDU_NAMES = DEFAULT_LIMB_WEIGHT_HDU_NAMES

    FILT = 1

    PB_FLOOR = ""

    DPA_DEG = 0.5
    R_USE_MIN, R_USE_MAX = 1.5, 4.0
    # Since R_MIN=1.8, COR1A cannot use r_use_min=1.4 unless R_MIN is also lowered.
    R_USE_MIN_BY_GROUP = {"cor1a": 1.5}
    R_USE_MAX_BY_GROUP = {}
    PB_SCALE_BY_GROUP = {}
    
    print("[INFO] Parameter Setting")
    print(f"OUT_N={OUT_N}, R=({R_MIN}, {R_MAX}), N=({NR}, {NTH}, {NPH}), DS={DS}, HM={HM},")
    print(f"LAM={LAM}, MAXITER={MAXITER}, TOL={TOL}, DESPIKE_NSIG={DESPIKE_NSIG}, DESPIKE_MED={DESPIKE_MED}")
    print(f"R_USE=({R_USE_MIN}, {R_USE_MAX}), R_USE_MIN_BY_GROUP={R_USE_MIN_BY_GROUP}")


    # With density prior enabled, we solve ne = prior*q.
    # Set USE_DENSITY_PRIOR=False to solve absolute ne directly.
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
    
    

    # Positivity: "clip" is fast but approximate; "lbfgsb" enforces x>=0 more strictly and is slower.
    POSITIVITY_METHOD = "clip"

    # Lambda: keep fixed by default.  Set LAMBDA_SELECT_MODE="lcurve" or "min_misfit" after checking diagnostics.
    LAMBDA_SCAN_VALUES = [1.0] #, 5.0, 8.0, 10.0, 15.0]  # e.g. [5.0, 10.0, 20.0, 40.0, 80.0]
    LAMBDA_SELECT_MODE = "fixed"

    # pB unit/kernel diagnostics.  MSB normalization is appropriate for MSB / Mean Solar Brightness inputs.
    THOMSON_NORMALIZE_MSB = True
    THOMSON_KERNEL_SCALE = 1.0
    RUN_PB_UNIT_DIAGNOSTICS = False
    PB_DIAGNOSTIC_PATHS = []

    CALIBRATION_REFERENCE_GROUP = "earth_merged"

    # Joint relative radiometric calibration.  Earth-view remains the fixed
    # reference; COR1A is retained and fitted with one scalar forward gain.
    AUTO_CROSS_CALIBRATE_GROUPS = True
    CROSS_CALIBRATION_INITIAL_GAIN_BY_GROUP = {"cor1a":0.843}
    CROSS_CALIBRATION_MAX_ITERATIONS = 5
    CROSS_CALIBRATION_TOLERANCE = 1e-2
    CROSS_CALIBRATION_DAMPING = 0.7
    CROSS_CALIBRATION_GAIN_MIN = 0.25
    CROSS_CALIBRATION_GAIN_MAX = 4.0
    CROSS_CALIBRATION_R_MIN = 1.5
    CROSS_CALIBRATION_R_MAX = 4
    CROSS_CALIBRATION_MIN_COUNT = 1000
    CROSS_CALIBRATION_CLIP_SIGMA = 4.0
    CROSS_CALIBRATION_RECALIBRATE_AFTER_LAMBDA_SELECTION = True

    USE_TEMPORAL_DESPIKE = False
    NE3DTOMO_GLOBAL_YBK = True
    SHOW_RAY_PROGRESS = True
    USE_RAY_CACHE = True

    SHOW_GUI = True
    HARMONIC = 2
    FREQ_MHZ_LIST = [33, 43]
    ISO_COLORS = None

    TARGET_TAG = parse_target_datetime(TARGET_TIME).strftime("%Y%m%d_%H%M%S")
    WINDOW_TAG = f"pm{int(SEARCH_WINDOW_DAYS)}d"
    FREQ_TAG = "-".join(str(float(f)).rstrip("0").rstrip(".") for f in FREQ_MHZ_LIST)
    CALIBRATION_TAG = "_auto-cal" if AUTO_CROSS_CALIBRATE_GROUPS else ""
    RAY_CACHE_DIR = (
        f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/"
        f"ray_cache_{TARGET_TAG}"
    )

    SAVE_PREPPED_DIR = (
        f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/"
        f"tomo_prepped_{TARGET_TAG}_{WINDOW_TAG}"
    )
    
    SAVE_NE_NPZ = (
        f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/ne_npz/"
        f"ne3d_solution_{TARGET_TAG}_{WINDOW_TAG}_{FREQ_TAG}MHz{CALIBRATION_TAG}.npz"
    )
    SAVE_SUMMARY_CSV = True
    SUMMARY_CSV_PATH = (
        f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/ne_npz/"
        f"ne3d_solution_{TARGET_TAG}_{WINDOW_TAG}_{FREQ_TAG}MHz{CALIBRATION_TAG}_summary.csv"
    )
    SAVE_PNG_PATH = (
        f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/output/multi-tomo/"
        f"tomo_{TARGET_TAG}_{WINDOW_TAG}_{FREQ_TAG}MHz{CALIBRATION_TAG}.png"
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
        lambda_scan_values=LAMBDA_SCAN_VALUES,
        lambda_select_mode=LAMBDA_SELECT_MODE,
        q_low=Q_LOW,
        width_pix=WIDTH_PIX,
        maxiter=MAXITER,
        tol=TOL,
        positivity_method=POSITIVITY_METHOD,
        apply_brightness_scale=APPLY_BRIGHTNESS_SCALE,
        use_density_prior=USE_DENSITY_PRIOR,
        density_prior_model=DENSITY_PRIOR_MODEL,
        density_prior_scale=DENSITY_PRIOR_SCALE,
        density_prior_file=DENSITY_PRIOR_FILE,
        thomson_normalize_msb=THOMSON_NORMALIZE_MSB,
        thomson_kernel_scale=THOMSON_KERNEL_SCALE,
        run_pb_unit_diagnostics=RUN_PB_UNIT_DIAGNOSTICS,
        pb_diagnostic_paths=PB_DIAGNOSTIC_PATHS,
        calibration_reference_group=CALIBRATION_REFERENCE_GROUP,
        auto_cross_calibrate_groups=AUTO_CROSS_CALIBRATE_GROUPS,
        cross_calibration_initial_gain_by_group=(
            CROSS_CALIBRATION_INITIAL_GAIN_BY_GROUP
        ),
        cross_calibration_max_iterations=(
            CROSS_CALIBRATION_MAX_ITERATIONS
        ),
        cross_calibration_tolerance=(
            CROSS_CALIBRATION_TOLERANCE
        ),
        cross_calibration_damping=(
            CROSS_CALIBRATION_DAMPING
        ),
        cross_calibration_gain_min=(
            CROSS_CALIBRATION_GAIN_MIN
        ),
        cross_calibration_gain_max=(
            CROSS_CALIBRATION_GAIN_MAX
        ),
        cross_calibration_r_min=(
            CROSS_CALIBRATION_R_MIN
        ),
        cross_calibration_r_max=(
            CROSS_CALIBRATION_R_MAX
        ),
        cross_calibration_min_count=(
            CROSS_CALIBRATION_MIN_COUNT
        ),
        cross_calibration_clip_sigma=(
            CROSS_CALIBRATION_CLIP_SIGMA
        ),
        cross_calibration_recalibrate_after_lambda_selection=(
            CROSS_CALIBRATION_RECALIBRATE_AFTER_LAMBDA_SELECTION
        ),

        use_temporal_despike=USE_TEMPORAL_DESPIKE,
        ne3dtomo_global_ybk=NE3DTOMO_GLOBAL_YBK,
        show_ray_progress=SHOW_RAY_PROGRESS,
        use_ray_cache=USE_RAY_CACHE,
        ray_cache_dir=RAY_CACHE_DIR,

        save_prepped_dir=SAVE_PREPPED_DIR,
        save_ne_npz=SAVE_NE_NPZ,
        save_summary_csv=SAVE_SUMMARY_CSV,
        summary_csv_path=SUMMARY_CSV_PATH,

        show_gui=SHOW_GUI,
        freq_mhz_list=FREQ_MHZ_LIST,
        harmonic=HARMONIC,
        iso_colors=ISO_COLORS,

        save_png=True,
        png_path=SAVE_PNG_PATH,
    )

    main(args)
