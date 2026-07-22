#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot two K-COR/LASCO combined pB FITS products as separate 2D maps.

Default target files
--------------------
1. /mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/pB_Kcor_LASCO_axi_20220619_2104.fits
2. /mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/pB_Kcor_LASCO_edge_smooth_20220619_2104.fits

The script is intentionally standalone.  It does not depend on the van de Hulst
inversion modules used by pB_line_main.py.  It only reads FITS pB images and
maps them on an x-y plane in units of solar radii.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, List
from datetime import datetime
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

try:
    from astropy.io import fits
    from astropy.wcs import WCS
except ImportError as exc:
    raise SystemExit(
        "This script requires astropy. Install it in the environment where you run the script, "
        "for example: pip install astropy"
    ) from exc


# =============================================================================
# User settings
# =============================================================================
# Target time used to select the nearest available FITS files.
# Accepted formats: YYYYMMDD_HHMM, YYYYMMDD_HHMMSS, YYYY-MM-DDTHH:MM[:SS].


# Directory containing tomography-ready pB products.
PB_FITS_SEARCH_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata")

# If None, the nearest file is used regardless of time offset.
# Set e.g. 120.0 to reject files more than +/-120 minutes from TARGET_TIME.
MAX_FILE_TIME_DELTA_MINUTES: Optional[float] = None

# Product patterns to select for the requested time.
PRODUCT_SPECS = [
    ("pB_Kcor_LASCO_axi_*.fits", "K-COR + LASCO-C2 (normal/hard transition)"),
    # ("pB_Kcor_LASCO_edge_smooth_*.fits", "K-COR + LASCO-C2 (edge smooth)"),
]

OUTPUT_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/SDO_Mk4_SOHO/pB/pB_output")

# Plot range in projected solar radii.  Use None to show the full image extent.
XY_LIMIT_RSUN: Optional[float] = 4.5

# Color normalization.  If None, percentiles of the positive finite values are used.
VMIN: Optional[float] = 1e-10
VMAX: Optional[float] = 1e-6
PERCENTILE_VMIN = 0.5
PERCENTILE_VMAX = 99.7

# Radial guide circles in Rsun.
RADIAL_CIRCLES = [1.0, 2.0, 4.0]

# Raw source directories used to recover the actual K-COR/LASCO boundary geometry.
LASCOPB_SEARCH_DIRS = [
    Path("/mnt/d/wsl/home/kinno-7010/Research_data/SOHO/pB"),
]
KCORPB_SEARCH_DIRS = [
    Path("/mnt/d/wsl/home/kinno-7010/Research_data/MK4_coronagraph/MK4_coronagraph_KCOR/pB/Rawdata"),
]

# Plot the K-COR/C2 boundary as 1-degree sampled white scatter points.
BOUNDARY_SCATTER_DEG = 1.0
BOUNDARY_SCATTER_SIZE = 3.0

# Output options.
DPI = 300
SAVE_NPZ_COORDINATES = False
SHOW_FIGURES = False


# =============================================================================
# Small utilities
# =============================================================================
@dataclass
class PBMap:
    path: Path
    data: np.ndarray
    header: fits.Header
    x_rsun: np.ndarray
    y_rsun: np.ndarray
    r_rsun: np.ndarray
    rsun_arcsec: float
    extent: Tuple[float, float, float, float]


def parse_target_datetime(value: str | datetime) -> datetime:
    """Parse a target time used to select the nearest pB FITS file."""
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
        f"Cannot parse TARGET_TIME={value!r}. Use e.g. '20220619_2104', "
        f"'20220619_210400', or '2022-06-19T21:04:00'."
    )


def parse_pb_filename_datetime(path: Path) -> Optional[datetime]:
    """Extract an observation time from supported tomography-ready pB FITS filenames."""
    name = Path(path).name
    patterns = (
        r"pB_Kcor_LASCO_axi_(\d{8})_(\d{4})\.fits",
        r"pB_Kcor_LASCO_edge_smooth_(\d{8})_(\d{4})\.fits",
        r"pB_LASCO_C2_only_(\d{8})_(\d{4})\.fits",
    )
    for pat in patterns:
        m = re.fullmatch(pat, name)
        if m:
            return datetime.strptime(m.group(1) + "_" + m.group(2), "%Y%m%d_%H%M")
    return None


def find_nearest_pb_file_by_time(
    search_dir: Path,
    glob_pattern: str,
    target_dt: datetime,
    max_delta_minutes: Optional[float] = None,
) -> Tuple[Path, datetime, float]:
    """Return the nearest pB FITS file matching glob_pattern to target_dt."""
    search_dir = Path(search_dir).expanduser()
    if not search_dir.exists():
        raise FileNotFoundError(f"Search directory does not exist: {search_dir}")

    candidates: List[Tuple[float, datetime, Path]] = []
    for path in sorted(search_dir.glob(glob_pattern)):
        obs_dt = parse_pb_filename_datetime(path)
        if obs_dt is None:
            continue
        delta_seconds = abs((obs_dt - target_dt).total_seconds())
        candidates.append((delta_seconds, obs_dt, path))

    if not candidates:
        raise FileNotFoundError(f"No files matching {glob_pattern!r} were found in {search_dir}")

    delta_seconds, obs_dt, path = min(candidates, key=lambda item: (item[0], item[2].name))
    delta_minutes = delta_seconds / 60.0
    if max_delta_minutes is not None and delta_minutes > float(max_delta_minutes):
        raise FileNotFoundError(
            f"Nearest file for pattern {glob_pattern!r} is {path.name} "
            f"({delta_minutes:.1f} min from target), which exceeds "
            f"MAX_FILE_TIME_DELTA_MINUTES={max_delta_minutes:g}."
        )
    return path, obs_dt, delta_minutes


def resolve_input_files_for_target_time() -> List[Tuple[Path, str]]:
    """Resolve normal and edge-smooth pB files nearest to TARGET_TIME."""
    target_dt = parse_target_datetime(TARGET_TIME)
    selected: List[Tuple[Path, str]] = []
    print(f"[INFO] Target time for plotting: {target_dt:%Y-%m-%d %H:%M:%S}")
    for pattern, label in PRODUCT_SPECS:
        path, obs_dt, delta_minutes = find_nearest_pb_file_by_time(
            search_dir=PB_FITS_SEARCH_DIR,
            glob_pattern=pattern,
            target_dt=target_dt,
            max_delta_minutes=MAX_FILE_TIME_DELTA_MINUTES,
        )
        print(
            f"[INFO] Selected {path.name} for pattern {pattern!r} "
            f"(obs={obs_dt:%Y-%m-%d %H:%M}, |dt|={delta_minutes:.1f} min)"
        )
        selected.append((path, label))
    return selected


def _header_float(header: fits.Header, keys: Sequence[str], default: Optional[float] = None) -> Optional[float]:
    """Return the first finite float found in a FITS header."""
    for key in keys:
        if key in header:
            try:
                val = float(header[key])
            except Exception:
                continue
            if np.isfinite(val):
                return val
    return default


def _cdelt_arcsec(header: fits.Header, axis: int) -> float:
    """Return CDELT for the requested axis in arcsec/pixel."""
    cdelt = _header_float(header, [f"CDELT{axis}", f"CD{axis}_{axis}"], default=None)
    if cdelt is None:
        raise ValueError(f"FITS header has neither CDELT{axis} nor CD{axis}_{axis}.")

    unit = str(header.get(f"CUNIT{axis}", "arcsec")).strip().lower()
    if unit in {"deg", "degree", "degrees"}:
        return cdelt * 3600.0
    if unit in {"arcmin", "arcminute", "arcminutes"}:
        return cdelt * 60.0
    # If missing or already arcsec-like, keep as is.
    return cdelt


def _rsun_arcsec(header: fits.Header) -> float:
    """Get apparent solar radius in arcsec, falling back to a standard value."""
    val = _header_float(header, ["RSUN_OBS", "RSUN"], default=None)
    if val is None or not np.isfinite(val) or val <= 0:
        print("[WARN] RSUN_OBS/RSUN not found or invalid; using 959.63 arcsec.")
        return 959.63
    return float(val)


def read_pb_fits_as_xy_map(path: Path) -> PBMap:
    """Read a pB FITS file and construct x/y/r maps in solar radii."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D FITS image, got shape={data.shape} for {path}")

    ny, nx = data.shape
    crpix1 = _header_float(header, ["CRPIX1"], default=(nx + 1) / 2.0)
    crpix2 = _header_float(header, ["CRPIX2"], default=(ny + 1) / 2.0)
    crval1 = _header_float(header, ["CRVAL1"], default=0.0)
    crval2 = _header_float(header, ["CRVAL2"], default=0.0)
    cdelt1 = _cdelt_arcsec(header, 1)
    cdelt2 = _cdelt_arcsec(header, 2)
    rsun = _rsun_arcsec(header)

    # FITS pixel coordinates are 1-based.  x/y below are helioprojective-like
    # coordinates in arcsec, converted to solar radii.
    ix = np.arange(nx, dtype=np.float64) + 1.0
    iy = np.arange(ny, dtype=np.float64) + 1.0
    x_arcsec = (ix - float(crpix1)) * float(cdelt1) + float(crval1)
    y_arcsec = (iy - float(crpix2)) * float(cdelt2) + float(crval2)

    x_rsun_1d = x_arcsec / rsun
    y_rsun_1d = y_arcsec / rsun
    x_rsun, y_rsun = np.meshgrid(x_rsun_1d, y_rsun_1d)
    r_rsun = np.hypot(x_rsun, y_rsun)

    extent = (float(x_rsun_1d[0]), float(x_rsun_1d[-1]), float(y_rsun_1d[0]), float(y_rsun_1d[-1]))
    return PBMap(path=path, data=data, header=header, x_rsun=x_rsun, y_rsun=y_rsun,
                 r_rsun=r_rsun, rsun_arcsec=rsun, extent=extent)


def _auto_lognorm_values(data: np.ndarray, vmin: Optional[float], vmax: Optional[float]) -> Tuple[float, float]:
    """Choose robust positive log-normalization bounds."""
    positive = data[np.isfinite(data) & (data > 0)]
    if positive.size == 0:
        raise ValueError("No positive finite pB values are available for LogNorm plotting.")

    lo = float(vmin) if vmin is not None else float(np.nanpercentile(positive, PERCENTILE_VMIN))
    hi = float(vmax) if vmax is not None else float(np.nanpercentile(positive, PERCENTILE_VMAX))

    if not np.isfinite(lo) or lo <= 0:
        lo = float(np.nanmin(positive))
    if not np.isfinite(hi) or hi <= lo:
        hi = float(np.nanmax(positive))
    if hi <= lo:
        hi = lo * 10.0
    return lo, hi


def _add_radial_guides(ax: plt.Axes, radii: Iterable[float]) -> None:
    """Overlay heliocentric radial guide circles."""
    for rr in radii:
        circ = Circle((0.0, 0.0), float(rr), fill=False, linestyle="--", linewidth=0.8, alpha=0.8)
        ax.add_patch(circ)
        # Put labels on the +x side with a small vertical offset to avoid overlap.
        ax.text(float(rr), 0.04, f"{rr:g} Rs", fontsize=8, ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.0))


def _header_value_text(header: fits.Header, keys: Sequence[str], default: str = "N/A") -> str:
    """Return a compact display string for the first available FITS-header keyword."""
    for key in keys:
        if key not in header:
            continue
        value = header[key]
        try:
            value_float = float(value)
        except Exception:
            return str(value)
        if np.isfinite(value_float):
            return f"{value_float:g}"
    return default


def _add_processing_legend(ax: plt.Axes, header: fits.Header) -> None:
    """Add a legend-style box with K-COR/LASCO merge and boundary-smoothing parameters."""
    legend_items = [
        ("KCOR_LASCO_BLEND_INNER_RSUN", ("BLENDIN",)),
        ("KCOR_LASCO_BLEND_OUTER_RSUN", ("BLENDOUT",)),
        ("BOUNDARY_WIDTH_RSUN", ("KCORBDW", "KCOREDR")),
        ("BOUNDARY_ANGULAR_BIN_DEG", ("KCORBDAD",)),
        ("BOUNDARY_WEIGHT_THRESHOLD", ("KCORBDTH",)),
    ]
    lines = ["Merge / boundary-smoothing parameters"]
    for label, keys in legend_items:
        lines.append(f"{label} = {_header_value_text(header, keys)}")
    lines.append("Boundary plot = white scatter (1 deg bins)")
    if "WCSALIGN" in header:
        lines.append(f"WCSALIGN = {header['WCSALIGN']}")
    if "LASCOROT" in header:
        lines.append(f"LASCOROT = {_header_value_text(header, ('LASCOROT',))} deg")

    ax.text(
        0.02, 0.02, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=7,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.78, pad=3.0),
    )


def _header_unit_to_arcsec_scale(unit: str) -> float:
    """Return multiplicative scale from a FITS WCS angular unit to arcsec."""
    unit = str(unit or "").strip().lower()
    if "deg" in unit:
        return 3600.0
    if "arcmin" in unit:
        return 60.0
    return 1.0


def _linear_wcs_matrix_arcsec_per_pixel(hdr: fits.Header) -> np.ndarray:
    """Return the 2x2 linear FITS WCS matrix in arcsec per pixel."""
    cdelt1 = float(hdr.get("CDELT1", 1.0))
    cdelt2 = float(hdr.get("CDELT2", 1.0))
    if all(k in hdr for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
        mat = np.array(
            [[float(hdr["CD1_1"]), float(hdr["CD1_2"])],
             [float(hdr["CD2_1"]), float(hdr["CD2_2"])]],
            dtype=np.float64,
        )
    elif any(k in hdr for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2")):
        pc11 = float(hdr.get("PC1_1", 1.0))
        pc12 = float(hdr.get("PC1_2", 0.0))
        pc21 = float(hdr.get("PC2_1", 0.0))
        pc22 = float(hdr.get("PC2_2", 1.0))
        mat = np.array(
            [[pc11 * cdelt1, pc12 * cdelt2],
             [pc21 * cdelt1, pc22 * cdelt2]],
            dtype=np.float64,
        )
    elif "CROTA2" in hdr or "CROTA1" in hdr:
        theta = np.deg2rad(float(hdr.get("CROTA2", hdr.get("CROTA1", 0.0))))
        c = np.cos(theta)
        s = np.sin(theta)
        mat = np.array(
            [[cdelt1 * c, -cdelt2 * s],
             [cdelt1 * s,  cdelt2 * c]],
            dtype=np.float64,
        )
    else:
        mat = np.array([[cdelt1, 0.0], [0.0, cdelt2]], dtype=np.float64)

    unit_scale = np.array(
        [_header_unit_to_arcsec_scale(str(hdr.get("CUNIT1", "arcsec"))),
         _header_unit_to_arcsec_scale(str(hdr.get("CUNIT2", "arcsec")))],
        dtype=np.float64,
    )
    return mat * unit_scale[:, None]


def _linear_pixel_to_arcsec(hdr: fits.Header, xpix: np.ndarray, ypix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback FITS-linear pixel -> helioprojective arcsec conversion."""
    crpix1 = float(hdr.get("CRPIX1", (int(hdr.get("NAXIS1", 1)) + 1) / 2.0))
    crpix2 = float(hdr.get("CRPIX2", (int(hdr.get("NAXIS2", 1)) + 1) / 2.0))
    crval1 = float(hdr.get("CRVAL1", 0.0)) * _header_unit_to_arcsec_scale(str(hdr.get("CUNIT1", "arcsec")))
    crval2 = float(hdr.get("CRVAL2", 0.0)) * _header_unit_to_arcsec_scale(str(hdr.get("CUNIT2", "arcsec")))
    dx = np.asarray(xpix, dtype=np.float64) + 1.0 - crpix1
    dy = np.asarray(ypix, dtype=np.float64) + 1.0 - crpix2
    mat = _linear_wcs_matrix_arcsec_per_pixel(hdr)
    xw = crval1 + mat[0, 0] * dx + mat[0, 1] * dy
    yw = crval2 + mat[1, 0] * dx + mat[1, 1] * dy
    return xw, yw


def _linear_arcsec_to_pixel(hdr: fits.Header, x_arcsec: np.ndarray, y_arcsec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback helioprojective arcsec -> FITS-linear pixel conversion."""
    crpix1 = float(hdr.get("CRPIX1", (int(hdr.get("NAXIS1", 1)) + 1) / 2.0))
    crpix2 = float(hdr.get("CRPIX2", (int(hdr.get("NAXIS2", 1)) + 1) / 2.0))
    crval1 = float(hdr.get("CRVAL1", 0.0)) * _header_unit_to_arcsec_scale(str(hdr.get("CUNIT1", "arcsec")))
    crval2 = float(hdr.get("CRVAL2", 0.0)) * _header_unit_to_arcsec_scale(str(hdr.get("CUNIT2", "arcsec")))
    rhs0 = np.asarray(x_arcsec, dtype=np.float64) - crval1
    rhs1 = np.asarray(y_arcsec, dtype=np.float64) - crval2
    mat = _linear_wcs_matrix_arcsec_per_pixel(hdr)
    try:
        inv = np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(mat)
    dx = inv[0, 0] * rhs0 + inv[0, 1] * rhs1
    dy = inv[1, 0] * rhs0 + inv[1, 1] * rhs1
    return dx + crpix1 - 1.0, dy + crpix2 - 1.0


def _has_complete_celestial_wcs(hdr: fits.Header) -> bool:
    """Return True if the header looks safe for astropy WCS celestial transforms."""
    ctype1 = str(hdr.get("CTYPE1", "")).upper()
    ctype2 = str(hdr.get("CTYPE2", "")).upper()
    return bool(ctype1 and ctype2 and ("HPL" in ctype1 or "RA" in ctype1) and ("HPL" in ctype2 or "DEC" in ctype2))


def sample_image_bilinear_safe(img: np.ndarray, xpix: np.ndarray, ypix: np.ndarray) -> np.ndarray:
    """Bilinear sample img at floating pixel coordinates. Invalid/outside points become NaN."""
    img = np.asarray(img, dtype=np.float64)
    xpix = np.asarray(xpix, dtype=np.float64)
    ypix = np.asarray(ypix, dtype=np.float64)
    xpix, ypix = np.broadcast_arrays(xpix, ypix)
    out = np.full(xpix.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(xpix) & np.isfinite(ypix)
    if not np.any(finite):
        return out
    x0 = np.floor(xpix[finite]).astype(np.int64)
    y0 = np.floor(ypix[finite]).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    inside = (x0 >= 0) & (y0 >= 0) & (x1 < img.shape[1]) & (y1 < img.shape[0])
    if not np.any(inside):
        return out
    flat_indices = np.flatnonzero(finite)[inside]
    x0 = x0[inside]
    y0 = y0[inside]
    x1 = x1[inside]
    y1 = y1[inside]
    x = xpix.ravel()[flat_indices]
    y = ypix.ravel()[flat_indices]
    wx = x - x0
    wy = y - y0
    vals = (
        (1.0 - wx) * (1.0 - wy) * img[y0, x0]
        + wx * (1.0 - wy) * img[y0, x1]
        + (1.0 - wx) * wy * img[y1, x0]
        + wx * wy * img[y1, x1]
    )
    out.ravel()[flat_indices] = vals
    return out


def reproject_image_to_header(src_img: np.ndarray, src_hdr: fits.Header, dst_hdr: fits.Header, dst_shape: Tuple[int, int]) -> np.ndarray:
    """Reproject src_img onto dst_hdr/dst_shape using WCS with linear fallback."""
    yy, xx = np.mgrid[0:dst_shape[0], 0:dst_shape[1]]
    use_astropy_wcs = _has_complete_celestial_wcs(src_hdr) and _has_complete_celestial_wcs(dst_hdr)
    if use_astropy_wcs:
        try:
            w_dst = WCS(dst_hdr)
            w_src = WCS(src_hdr)
            xw_dst, yw_dst = w_dst.pixel_to_world_values(xx, yy)
            x_src_pix, y_src_pix = w_src.world_to_pixel_values(xw_dst, yw_dst)
            out = sample_image_bilinear_safe(src_img, x_src_pix, y_src_pix)
            if np.count_nonzero(np.isfinite(out)) > 0:
                return out
        except Exception:
            pass
    x_arc, y_arc = _linear_pixel_to_arcsec(dst_hdr, xx, yy)
    x_src_pix, y_src_pix = _linear_arcsec_to_pixel(src_hdr, x_arc, y_arc)
    return sample_image_bilinear_safe(src_img, x_src_pix, y_src_pix)


def _find_named_file(name: str, search_dirs: Sequence[Path]) -> Optional[Path]:
    """Find a file by exact name under the provided search directories."""
    if not name or str(name).strip().upper() == "NONE":
        return None
    for root in search_dirs:
        root = Path(root)
        cand = root / name
        if cand.exists():
            return cand
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    return None


def _boundary_points_by_angle(pbmap: PBMap, boundary_scatter_deg: float = 1.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Recover the actual K-COR/C2 boundary and sample it every boundary_scatter_deg in angle."""
    kcor_name = str(pbmap.header.get("KCORFILE", "")).strip()
    lasco_name = str(pbmap.header.get("LASCOPB", "")).strip()
    kcor_path = _find_named_file(kcor_name, KCORPB_SEARCH_DIRS)
    lasco_path = _find_named_file(lasco_name, LASCOPB_SEARCH_DIRS)
    if kcor_path is None or lasco_path is None:
        return None

    with fits.open(lasco_path) as hdul:
        lasco_img = np.asarray(hdul[0].data, dtype=np.float64)
        lasco_hdr = hdul[0].header.copy()
    with fits.open(kcor_path) as hdul:
        kcor_img = np.asarray(hdul[0].data, dtype=np.float64)
        kcor_hdr = hdul[0].header.copy()

    # Recover the boundary on the same output grid that is being plotted.
    # This is important when the combined FITS has been derotated to the
    # Earth-view grid: using the raw LASCO-C2 header here would reintroduce
    # the original LASCO roll angle and shift the boundary scatter points.
    output_hdr = pbmap.header.copy()
    output_shape = pbmap.data.shape
    lasco_on_output = reproject_image_to_header(lasco_img, lasco_hdr, output_hdr, output_shape)
    kcor_on_lasco = reproject_image_to_header(kcor_img, kcor_hdr, output_hdr, output_shape)

    rho = pbmap.r_rsun
    has_lasco = np.isfinite(lasco_on_output) & (lasco_on_output > 0)
    kcor_rmin = _header_float(pbmap.header, ["KCORRMIN"], default=1.0)
    kcor_rmax = _header_float(pbmap.header, ["KCORRMAX", "BLENDOUT"], default=3.0)
    blend_inner = _header_float(pbmap.header, ["BLENDIN"], default=2.0)
    blend_outer = _header_float(pbmap.header, ["BLENDOUT"], default=3.0)
    threshold = _header_float(pbmap.header, ["KCORBDTH"], default=0.5)
    if None in (kcor_rmin, kcor_rmax, blend_inner, blend_outer, threshold):
        return None

    kcor_radius_mask = np.isfinite(rho) & (rho >= float(kcor_rmin)) & (rho <= float(kcor_rmax))
    has_kcor = np.isfinite(kcor_on_lasco) & (kcor_on_lasco > 0) & kcor_radius_mask

    kcor_weight = np.zeros_like(rho, dtype=np.float64)
    inner = has_kcor & (rho <= float(blend_inner))
    kcor_weight[inner] = 1.0
    overlap = has_kcor & has_lasco & (rho > float(blend_inner)) & (rho < float(blend_outer))
    if np.any(overlap):
        alpha = (rho[overlap] - float(blend_inner)) / max(1e-6, (float(blend_outer) - float(blend_inner)))
        kcor_weight[overlap] = 1.0 - alpha
    fill_lasco_holes = has_kcor & (~has_lasco)
    kcor_weight[fill_lasco_holes] = 1.0

    valid = has_kcor | has_lasco
    dominant = valid & (kcor_weight >= float(threshold))

    boundary_mask = np.zeros_like(dominant, dtype=bool)
    boundary_mask[:-1, :] |= valid[:-1, :] & valid[1:, :] & (dominant[:-1, :] != dominant[1:, :])
    boundary_mask[1:, :]  |= valid[:-1, :] & valid[1:, :] & (dominant[:-1, :] != dominant[1:, :])
    boundary_mask[:, :-1] |= valid[:, :-1] & valid[:, 1:] & (dominant[:, :-1] != dominant[:, 1:])
    boundary_mask[:, 1:]  |= valid[:, :-1] & valid[:, 1:] & (dominant[:, :-1] != dominant[:, 1:])

    if not np.any(boundary_mask):
        return None

    theta = (np.degrees(np.arctan2(pbmap.y_rsun, pbmap.x_rsun)) + 360.0) % 360.0
    step = float(boundary_scatter_deg)
    if not np.isfinite(step) or step <= 0:
        step = 1.0

    x_pts: List[float] = []
    y_pts: List[float] = []
    for th0 in np.arange(0.0, 360.0, step):
        th1 = th0 + step
        ang_mask = (theta >= th0) & (theta < th1) & boundary_mask
        if not np.any(ang_mask):
            continue
        r_vals = rho[ang_mask]
        r_vals = r_vals[np.isfinite(r_vals)]
        if r_vals.size == 0:
            continue
        r_med = float(np.nanmedian(r_vals))
        th_c = th0 + 0.5 * step
        x_pts.append(r_med * np.cos(np.deg2rad(th_c)))
        y_pts.append(r_med * np.sin(np.deg2rad(th_c)))

    if not x_pts:
        return None
    return np.asarray(x_pts, dtype=np.float64), np.asarray(y_pts, dtype=np.float64)


def plot_single_pb_map(pbmap: PBMap, label: str, output_dir: Path) -> Path:
    """Create and save one 2D pB map."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # LogNorm cannot show non-positive values.  Keep them as masked values.
    plot_data = np.array(pbmap.data, dtype=np.float64, copy=True)
    plot_data[~np.isfinite(plot_data) | (plot_data <= 1e-11)] = np.nan
    vmin, vmax = _auto_lognorm_values(plot_data, VMIN, VMAX)

    bunit = str(pbmap.header.get("BUNIT", "pB"))
    date_obs = str(pbmap.header.get("DATE-OBS", pbmap.header.get("DATE_OBS", "")))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")
    
    

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(
        plot_data,
        origin="lower",
        extent=pbmap.extent,
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
        aspect="equal",
    )

    _add_radial_guides(ax, RADIAL_CIRCLES)

    boundary_points = _boundary_points_by_angle(pbmap, boundary_scatter_deg=BOUNDARY_SCATTER_DEG)
    if boundary_points is not None:
        bx, by = boundary_points
        ax.scatter(bx, by, s=BOUNDARY_SCATTER_SIZE, c="white", marker="o", linewidths=0.0, alpha=0.95)

    ax.axhline(0.0, color="white", linewidth=0.5, alpha=0.7)
    ax.axvline(0.0, color="white", linewidth=0.5, alpha=0.7)
    ax.set_xlabel(r"X [$R_\odot$]")
    ax.set_ylabel(r"Y [$R_\odot$]")
    ax.set_title(f"{label}\n{pbmap.path.name}\n{date_obs}")

    if XY_LIMIT_RSUN is not None:
        lim = float(XY_LIMIT_RSUN)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.86)
    cbar.set_label(f"pB [{bunit}]")

    _add_processing_legend(ax, pbmap.header)

    out_name = pbmap.path.with_suffix("").name + "_2d_map.png"
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

    print(f"[OK] Saved 2D pB map: {out_path}")
    print(f"     file={pbmap.path}")
    print(f"     BUNIT={bunit!r}, RSUN={pbmap.rsun_arcsec:.3f} arcsec, LogNorm={vmin:.3e}..{vmax:.3e}")
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_items = resolve_input_files_for_target_time()

    maps = []
    for path, label in input_items:
        pbmap = read_pb_fits_as_xy_map(Path(path))
        maps.append(pbmap)
        plot_single_pb_map(pbmap, label=label, output_dir=OUTPUT_DIR)

        if SAVE_NPZ_COORDINATES:
            npz_path = OUTPUT_DIR / (pbmap.path.with_suffix("").name + "_xy_rsun_map.npz")
            np.savez_compressed(
                npz_path,
                data=pbmap.data.astype(np.float32),
                x_rsun=pbmap.x_rsun.astype(np.float32),
                y_rsun=pbmap.y_rsun.astype(np.float32),
                r_rsun=pbmap.r_rsun.astype(np.float32),
                rsun_arcsec=float(pbmap.rsun_arcsec),
                source_path=str(pbmap.path),
                target_time=str(TARGET_TIME),
            )
            print(f"[OK] Saved coordinate NPZ: {npz_path}")

    print("[DONE] Finished plotting separate 2D maps.")


if __name__ == "__main__":
    TARGET_TIME = "20220607_0258"
    main()
