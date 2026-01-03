#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_regularized_tomography_fixed.py

Regularized tomography (Tikhonov) for coronal electron density from time-series pB images,
aiming to be *algorithmically consistent* with the SSC/Ne3dTomo (V1.1) preprocessing logic:

- "preview_data.pro": (optional) rebin to 128x128, noise floor, and basic QC/preview.
- "pbmap_despike.pro" + "fix_nan.pro": robust despike and NaN repair on the rebinned maps.
- "cor1_getpbr.pro" + "get_pbrlc.pro": polar (r, PA) sampling used for background/noise proxy.
- "get_cor1_bbk.pro": low-harmonic (FFT) smoothing over PA to estimate a radial background ybk(r).
- "map_get_coord.pro" / "map_get_pixel.pro": image<->coordinate mapping (handled here via full WCS).

Important practical note:
The uploaded IDL sources include "..." placeholders (truncated blocks), so the despike and
background estimation are implemented as robust, conservative analogs. The forward model and
regularization are implemented in the same *form* (weighted least-squares + smoothness penalty),
but this is not a byte-for-byte reproduction of SSC's Fortran toolchain.

Dependencies:
  pip install numpy astropy scipy pyvista pyvistaqt

Author: (generated/edited with ChatGPT)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import median_filter
from scipy.sparse.linalg import LinearOperator, cg

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
DEFAULT_LIMB_U = 0.56  # SSC default used in provided Python port  # typical optical limb-darkening coefficient


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


# ----------------------------
# Thomson pB kernel
# ----------------------------
def thomsonscatter_pB_per_electron(impact_rsun: float, theta_from_pos_rad: float,
                                  u: float = DEFAULT_LIMB_U) -> float:
    """
    Polarized brightness contribution per single electron at a given position along a LOS,
    using the SSC Ne3dTomo V1.1 kernel form.

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

    pB = (3.0 / 16.0) * ICEN * SIGMA_T * (sinchi ** 2) * ((1.0 - u) * A + u * B)

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

    # Pixel centers on rebinned grid
    yy, xx = np.mgrid[0:out_n, 0:out_n]
    xpix = (xx + 0.5) * s - 0.5
    ypix = (yy + 0.5) * s - 0.5

    # Preferred: WCS
    try:
        w = WCS(hdr)
        xw, yw = w.pixel_to_world_values(xpix, ypix)

        x_map = np.array(xw, dtype=np.float64, copy=False)
        y_map = np.array(yw, dtype=np.float64, copy=False)

        # Convert deg -> arcsec if needed
        try:
            cu1 = str(getattr(w.wcs, "cunit", [None, None])[0]).lower()
            if "deg" in cu1:
                x_map *= 3600.0
                y_map *= 3600.0
        except Exception:
            pass

    except Exception:
        # Fallback: linear mapping using CRPIX/CDELT in arcsec
        cdelt1 = float(hdr.get("CDELT1", 1.0))
        cdelt2 = float(hdr.get("CDELT2", 1.0))
        crpix1 = float(hdr.get("CRPIX1", (orig_n + 1) / 2.0)) - 1.0
        crpix2 = float(hdr.get("CRPIX2", (orig_n + 1) / 2.0)) - 1.0
        crval1 = float(hdr.get("CRVAL1", 0.0))
        crval2 = float(hdr.get("CRVAL2", 0.0))

        x_map = (xpix - crpix1) * cdelt1 + crval1
        y_map = (ypix - crpix2) * cdelt2 + crval2

        cunit1 = str(hdr.get("CUNIT1", "")).lower()
        if "deg" in cunit1:
            x_map *= 3600.0
            y_map *= 3600.0

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
def fill_nan_by_neighbor_mean(img: np.ndarray, max_passes: int = 10) -> np.ndarray:
    """Fill NaNs using 4-neighbor means (IDL fix_nan.pro analogue).

    The SSC IDL routine fix_nan.pro replaces NaNs with the mean of the four
    nearest neighbors (up, down, left, right) when available.

    Parameters
    ----------
    img : np.ndarray
        2D array.
    max_passes : int
        Maximum number of passes to propagate values into connected NaN regions.

    Returns
    -------
    np.ndarray
        Filled array (a copy).
    """
    if img.ndim != 2:
        raise ValueError("fill_nan_by_neighbor_mean expects a 2D array")
    out = img.astype(float, copy=True)

    for _ in range(max_passes):
        nan_mask = ~np.isfinite(out)
        if not np.any(nan_mask):
            break

        ny, nx = out.shape
        sumv = np.zeros_like(out, dtype=float)
        cnt = np.zeros_like(out, dtype=int)

        # up neighbor (y-1)
        v = np.isfinite(out[:-1, :])
        sumv[1:, :] += np.where(v, out[:-1, :], 0.0)
        cnt[1:, :] += v.astype(int)

        # down neighbor (y+1)
        v = np.isfinite(out[1:, :])
        sumv[:-1, :] += np.where(v, out[1:, :], 0.0)
        cnt[:-1, :] += v.astype(int)

        # left neighbor (x-1)
        v = np.isfinite(out[:, :-1])
        sumv[:, 1:] += np.where(v, out[:, :-1], 0.0)
        cnt[:, 1:] += v.astype(int)

        # right neighbor (x+1)
        v = np.isfinite(out[:, 1:])
        sumv[:, :-1] += np.where(v, out[:, 1:], 0.0)
        cnt[:, :-1] += v.astype(int)

        fillable = nan_mask & (cnt > 0)
        if not np.any(fillable):
            break

        out[fillable] = sumv[fillable] / cnt[fillable]

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
        # Avoid log of non-positive
        work = np.where(work > 0, np.log10(work), np.nan)

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
    use_log: bool = True,
) -> np.ndarray:
    """
    Temporal despike over a cube (nt, ny, nx), aligned with the intent of SSW's
    ssw_unspike_cube used in SSC/Ne3dTomo IDL preprocessing.

    Strategy (per-pixel, along the time axis):
      1) Work in log10(domain) by default to reduce dynamic-range issues in pB.
      2) Compute temporal median and robust sigma via MAD.
      3) Replace positive outliers (> nsig*sigma above the median) with the median.

    Notes:
      - This is a pragmatic analogue; SSW's exact ssw_unspike_cube implementation differs,
        but this captures the core behavior for cosmic-ray-like single-frame spikes.
      - NaNs are supported and ignored in the statistics.
    """
    out = np.array(pbs, dtype=np.float64, copy=True)
    if out.ndim != 3 or out.shape[0] < 2:
        return out

    # Work array for robust stats
    work = out.copy()
    if use_log:
        # log10 only on positive values; keep others as NaN to exclude from statistics
        work = np.where(work > 0, np.log10(work), np.nan)

    med = np.nanmedian(work, axis=0)
    mad = np.nanmedian(np.abs(work - med[None, :, :]), axis=0)

    # Convert MAD to robust sigma (Gaussian equivalent)
    sigma = 1.4826 * mad
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)

    # Threshold (upper outliers only; keep dim pixels)
    thr = med + nsig * sigma

    # Build mask of outliers frame-by-frame
    for t in range(out.shape[0]):
        wt = work[t]
        bad = np.isfinite(wt) & np.isfinite(thr) & (wt > thr)
        if np.any(bad):
            out[t][bad] = np.where(use_log, np.power(10.0, med[bad]), med[bad])

    return out

def rebin_idl_linear(img: np.ndarray, out_n: int) -> np.ndarray:
    """
    IDL-like REBIN for square 2D images: interpolative resampling to (out_n, out_n).

    The SSC/Ne3dTomo IDL pipeline uses rebin_map()/rebin(), which is interpolative rather than
    strict block-averaging. For consistency with that design, we provide a linear-interpolation
    resampler.

    If the input is already (out_n, out_n), it is returned as float64 copy.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError(f"rebin_idl_linear expects a square 2D image, got {img.shape}")
    n = img.shape[0]
    if n == out_n:
        return img.copy()

    # Prefer SciPy's ndimage zoom (bilinear; order=1)
    try:
        from scipy.ndimage import zoom  # type: ignore
        z = out_n / float(n)
        out = zoom(img, (z, z), order=1, mode="nearest", prefilter=False)
        out = out[:out_n, :out_n]
        if out.shape != (out_n, out_n):
            tmp = np.full((out_n, out_n), np.nan, dtype=np.float64)
            yy = min(out_n, out.shape[0])
            xx = min(out_n, out.shape[1])
            tmp[:yy, :xx] = out[:yy, :xx]
            out = tmp
        return out
    except Exception:
        # Fallback to the existing block average
        return block_reduce_mean(img, out_n)


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
    Bilinear sample img at floating (y,x) pixel coordinates (0-based).
    Robust against NaN/Inf in x,y (returns NaN there) and avoids invalid-cast warnings.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x, y = np.broadcast_arrays(x, y)

    out = np.full(x.shape, np.nan, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return out

    xv = x[m]
    yv = y[m]

    x0 = np.floor(xv).astype(np.int64)
    y0 = np.floor(yv).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    x0c = np.clip(x0, 0, img.shape[1] - 1)
    x1c = np.clip(x1, 0, img.shape[1] - 1)
    y0c = np.clip(y0, 0, img.shape[0] - 1)
    y1c = np.clip(y1, 0, img.shape[0] - 1)

    Ia = img[y0c, x0c]
    Ib = img[y0c, x1c]
    Ic = img[y1c, x0c]
    Id = img[y1c, x1c]

    wa = (x1 - xv) * (y1 - yv)
    wb = (xv - x0) * (y1 - yv)
    wc = (x1 - xv) * (yv - y0)
    wd = (xv - x0) * (yv - y0)

    out[m] = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out


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
    pb = np.asarray(pb, dtype=np.float64)
    if pb.shape != (out_n, out_n):
        raise ValueError(f"polar_sample_pb expects pb shape {(out_n,out_n)}, got {pb.shape}")

    rsun_arcsec = float(hdr.get("RSUN_OBS", hdr.get("RSUN", 959.63)))

    rgrid = np.linspace(r_use_min, r_use_max, nr)
    pa_grid = np.arange(0.0, 360.0, dpa_deg, dtype=np.float64)
    rr, pp = np.meshgrid(rgrid, pa_grid, indexing="xy")

    pa_rad = np.deg2rad(pp)
    x_arc = rr * np.sin(pa_rad) * rsun_arcsec
    y_arc = rr * np.cos(pa_rad) * rsun_arcsec

    xpix = np.full_like(x_arc, np.nan, dtype=np.float64)
    ypix = np.full_like(y_arc, np.nan, dtype=np.float64)
    used_wcs = False

    try:
        w = WCS(hdr)
        cunit1 = str(hdr.get("CUNIT1", "")).lower()
        if "deg" in cunit1:
            xw = x_arc / 3600.0
            yw = y_arc / 3600.0
        else:
            xw = x_arc
            yw = y_arc
        xpix, ypix = w.world_to_pixel_values(xw, yw)
        used_wcs = True
    except Exception:
        used_wcs = False

    if not used_wcs:
        def _cd_matrix_from_header(h):
            if all(k in h for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
                return np.array([[float(h["CD1_1"]), float(h["CD1_2"])],
                                 [float(h["CD2_1"]), float(h["CD2_2"])]], dtype=np.float64)
            if all(k in h for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2")):
                pc = np.array([[float(h["PC1_1"]), float(h["PC1_2"])],
                               [float(h["PC2_1"]), float(h["PC2_2"])]], dtype=np.float64)
                cdelt1 = float(h.get("CDELT1", 1.0))
                cdelt2 = float(h.get("CDELT2", 1.0))
                return pc @ np.diag([cdelt1, cdelt2])
            crota = float(h.get("CROTA2", h.get("CROTA", 0.0)))
            cdelt1 = float(h.get("CDELT1", 1.0))
            cdelt2 = float(h.get("CDELT2", 1.0))
            th = np.deg2rad(crota)
            rot = np.array([[np.cos(th), -np.sin(th)],
                            [np.sin(th),  np.cos(th)]], dtype=np.float64)
            return rot @ np.diag([cdelt1, cdelt2])

        cd = _cd_matrix_from_header(hdr)
        inv = np.linalg.pinv(cd)

        crpix1 = float(hdr.get("CRPIX1", out_n / 2.0))
        crpix2 = float(hdr.get("CRPIX2", out_n / 2.0))
        crval1 = float(hdr.get("CRVAL1", 0.0))
        crval2 = float(hdr.get("CRVAL2", 0.0))

        cunit1 = str(hdr.get("CUNIT1", "")).lower()
        if "deg" in cunit1:
            xw = x_arc / 3600.0
            yw = y_arc / 3600.0
        else:
            xw = x_arc
            yw = y_arc

        dxw = xw - crval1
        dyw = yw - crval2

        dpx = inv[0, 0] * dxw + inv[0, 1] * dyw
        dpy = inv[1, 0] * dxw + inv[1, 1] * dyw
        xpix = (dpx + crpix1) - 1.0
        ypix = (dpy + crpix2) - 1.0

    x0 = np.floor(xpix).astype(int)
    y0 = np.floor(ypix).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    inside = (x0 >= 0) & (y0 >= 0) & (x1 < out_n) & (y1 < out_n)
    y = np.full_like(xpix, np.nan, dtype=np.float64)

    wx = xpix - x0
    wy = ypix - y0

    v00 = pb[y0.clip(0, out_n - 1), x0.clip(0, out_n - 1)]
    v10 = pb[y0.clip(0, out_n - 1), x1.clip(0, out_n - 1)]
    v01 = pb[y1.clip(0, out_n - 1), x0.clip(0, out_n - 1)]
    v11 = pb[y1.clip(0, out_n - 1), x1.clip(0, out_n - 1)]

    y[inside] = (
        (1 - wx[inside]) * (1 - wy[inside]) * v00[inside]
        + wx[inside] * (1 - wy[inside]) * v10[inside]
        + (1 - wx[inside]) * wy[inside] * v01[inside]
        + wx[inside] * wy[inside] * v11[inside]
    )

    if q_low is not None and q_low > 0:
        try:
            from scipy.ndimage import median_filter  # type: ignore
            y2 = median_filter(y.copy(), size=(2 * hm + 1, 1), mode="nearest")
            y = y - y2
        except Exception:
            pass

    rho = rr.ravel()
    pa = pp.ravel()
    x_pix = xpix.ravel()
    y_pix = ypix.ravel()
    y_flat = y.ravel()

    return y_flat, rho, pa, x_pix, y_pix, rsun_arcsec


# ----------------------------
# "get_cor1_bbk.pro" analog (FFT background)
# ----------------------------
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
    """Estimate radial pB background ybk(r) using SSC/IDL design.

    This follows the core logic of get_cor1_bbk.pro:
      1) sample pB into (r, PA) grid via radial cuts
      2) remove low-r artifacts (occulter/track heuristic)
      3) FFT along PA, keep only low harmonics (<hm), inverse FFT
      4) take max over PA for each r => mbk(r)
      5) smooth mbk(r) along r

    Notes
    -----
    * This codebase's `polar_sample_pb()` now generates its own (r,PA) grids from
      (r_use_min,r_use_max,dpa_deg,nr). We keep ybk_profile_fft()'s API unchanged
      and adapt internally.
    * q_low is kept for backward compatibility; it is not used here.

    Returns
    -------
    r_grid : (nr,) Rsun
    ybk    : (nr,) background pB
    pb_noise : float, noise proxy (mode of histogram peak)
    """
    pb = np.asarray(pb, dtype=np.float64)
    if pb.ndim != 2 or pb.shape[0] != pb.shape[1]:
        raise ValueError(f"ybk_profile_fft expects a square 2-D pb map, got {pb.shape}")

    out_n = int(pb.shape[0])

    # r/PA grids (must match polar_sample_pb's internal grids)
    r_grid = np.linspace(float(rmin), float(rmax), int(nr))
    pa_grid = np.arange(0.0, 360.0, float(dpa_deg), dtype=float)
    npa = int(pa_grid.size)

    # Sample pB into (r,PA)
    # polar_sample_pb returns y_flat arranged as (npa,nr) in C order; transpose to (nr,npa)
    y_flat, _, _, _, _, _ = polar_sample_pb(
        pb, hdr,
        out_n=out_n,
        r_use_min=float(rmin),
        r_use_max=float(rmax),
        limb_u=float(DEFAULT_LIMB_U),  # sampler currently doesn't use it; kept for signature compatibility
        dpa_deg=float(dpa_deg),
        nr=int(nr),
        hm=int(hm),
        width_pix=int(width_pix),
        q_low=float(q_low) if q_low is not None else 0.0,
    )

    if y_flat.size != npa * int(nr):
        # fallback if upstream PA grid logic changed
        if int(nr) > 0 and (y_flat.size % int(nr) == 0):
            npa = int(y_flat.size // int(nr))
        else:
            raise ValueError(f"polar_sample_pb returned unexpected length {y_flat.size} for nr={nr}")

    pbr = y_flat.reshape((npa, int(nr)), order="C").T  # (nr,npa)

    # If sampling failed everywhere (WCS mismatch etc.), return a conservative constant profile.
    finite_cnt = int(np.count_nonzero(np.isfinite(pbr) & (pbr > 0)))
    if finite_cnt < 10:
        data = pb[np.isfinite(pb) & (pb > 0)]
        if data.size:
            y0 = float(np.nanmedian(data))
            ybk = np.full(int(nr), y0, dtype=float)
        else:
            ybk = np.full(int(nr), np.nan, dtype=float)

        # noise proxy
        if data.size < 50:
            pb_noise = float(np.nanstd(data)) if data.size else 0.0
        else:
            binsize = 0.2e-10
            dmax = float(np.nanmax(data))
            nbins = max(10, int(dmax / binsize)) if dmax > 0 else 10
            hist, edges = np.histogram(data, bins=nbins, range=(0.0, nbins * binsize))
            imax = int(np.argmax(hist))
            pb_noise = 0.5 * (edges[imax] + edges[imax + 1])

        return r_grid, ybk, pb_noise

    # --- IDL-like artifact removal near occulter / low-r track ---
    occ_r = 1.5
    pbr = pbr.astype(float, copy=True)
    pbr[r_grid <= occ_r, :] = np.nan

    # For each PA, compute mean level in 2.0-2.5 Rsun and suppress values below that level for r<=2.0
    for j in range(npa):
        col = pbr[:, j]
        ref = (r_grid > 2.0) & (r_grid < 2.5) & np.isfinite(col)
        if np.count_nonzero(ref) >= 3:
            m0 = float(np.nanmean(col[ref]))
            bad = (r_grid <= 2.0) & np.isfinite(col) & (col < m0)
            if np.any(bad):
                col[bad] = np.nan
                pbr[:, j] = col

    # --- FFT smoothing along PA (low harmonics) ---
    work = pbr.copy()
    hm_i = int(hm)
    if hm_i < 1:
        hm_i = 1

    for i in range(int(nr)):
        row = work[i, :]
        fin = np.isfinite(row) & (row > 0)
        if np.count_nonzero(fin) < 4:
            continue

        med = float(np.nanmedian(row[fin]))
        row_f = np.where(np.isfinite(row), row, med)

        ft = np.fft.rfft(row_f, n=npa)
        if hm_i < ft.size:
            ft[hm_i:] = 0.0

        work[i, :] = np.fft.irfft(ft, n=npa)

    # Take max over PA safely
    mbk = np.full(int(nr), np.nan, dtype=float)
    for i in range(int(nr)):
        row = work[i, :]
        fin = np.isfinite(row)
        if np.any(fin):
            mbk[i] = float(np.nanmax(row[fin]))

    # Smooth along r: IDL smooth(abk,5) ~ boxcar of width 5
    ybk = mbk.copy()
    if int(nr) >= 5:
        k = 5
        ker = np.ones(k, dtype=float) / float(k)
        pad = k // 2
        ypad = np.pad(ybk, (pad, pad), mode="edge")
        ybk = np.convolve(ypad, ker, mode="valid")

    # Fill remaining NaNs by 1D interpolation
    good = np.isfinite(ybk)
    if np.count_nonzero(good) >= 2:
        ybk = np.interp(r_grid, r_grid[good], ybk[good])
    elif np.count_nonzero(good) == 1:
        ybk[:] = ybk[good][0]
    else:
        data = pb[np.isfinite(pb) & (pb > 0)]
        ybk[:] = float(np.nanmedian(data)) if data.size else np.nan

    # --- Noise proxy from histogram peak (IDL-like intent) ---
    data = pb[np.isfinite(pb) & (pb > 0)]
    if data.size < 50:
        pb_noise = float(np.nanstd(data)) if data.size else 0.0
    else:
        binsize = 0.2e-10
        dmax = float(np.nanmax(data))
        nbins = max(10, int(dmax / binsize)) if dmax > 0 else 10
        hist, edges = np.histogram(data, bins=nbins, range=(0.0, nbins * binsize))
        imax = int(np.argmax(hist))
        pb_noise = 0.5 * (edges[imax] + edges[imax + 1])

    return r_grid, ybk, pb_noise


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
) -> RayBundle:
    """
    For each used pixel, build a sparse ray (list of voxel indices + weights).
    The LOS is sampled in image coordinate s, and mapped into Carrington frame
    using obs camera basis (cam_x, cam_y, cam_z).
    """
    x_use = obs.x.ravel()[obs.idx_map]
    y_use = obs.y.ravel()[obs.idx_map]

    rE = grid.r_edges
    tE = grid.th_edges
    pE = grid.ph_edges

    cx, cy, cz = obs.cam_x, obs.cam_y, obs.cam_z

    vox_idx_list: List[np.ndarray] = []
    vox_w_list: List[np.ndarray] = []

    for xp, yp in zip(x_use, y_use):
        rho = float(np.hypot(xp, yp))
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
            pb_per_e = thomsonscatter_pB_per_electron(rho, theta_from_pos, u=limb_u)

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
def apply_LTL(x: np.ndarray, grid: SphericalGrid, wt_r: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Discrete 3D Laplacian-like smoothness penalty in (r,theta,phi) voxel space.
    wt_r: optional radial weights per r-bin (length nr) to emulate SSC "wt_nr" concept.
    """
    nr, nth, nph = grid.nr, grid.nth, grid.nph
    X = x.reshape((nr, nth, nph))
    out = np.zeros_like(X)

    # finite-difference second derivative in each dimension
    # r
    out[1:-1, :, :] += (2 * X[1:-1, :, :] - X[:-2, :, :] - X[2:, :, :])
    # theta
    out[:, 1:-1, :] += (2 * X[:, 1:-1, :] - X[:, :-2, :] - X[:, 2:, :])
    # phi (periodic)
    out[:, :, :] += (2 * X[:, :, :] - np.roll(X, 1, axis=2) - np.roll(X, -1, axis=2))

    if wt_r is not None:
        wr = wt_r[:, None, None]
        out = out * wr

    return out.ravel()


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
    ):
        self.grid = grid
        self.observations = observations
        self.rays = rays
        self.lam = float(lam)
        self.wt_r = wt_r
        self.W = np.concatenate([o.w for o in observations]).astype(np.float64)
        self._build_slices()

    def _build_slices(self):
        self.slices: List[slice] = []
        start = 0
        for obs in self.observations:
            n = obs.idx_map.size
            self.slices.append(slice(start, start + n))
            start += n

    def A_times(self, x: np.ndarray) -> np.ndarray:
        """
        Forward projection y = A x, concatenated over observations.
        Each measurement is a ray integral: sum(w_ij * x_j).
        """
        y_list: List[np.ndarray] = []
        for ray in self.rays:
            y = np.zeros(len(ray.vox_idx), dtype=np.float64)
            for i, (idx, ww) in enumerate(zip(ray.vox_idx, ray.vox_w)):
                if idx.size == 0:
                    y[i] = 0.0
                else:
                    y[i] = float(np.dot(ww, x[idx]))
            y_list.append(y)
        return np.concatenate(y_list)

    def AT_times(self, y: np.ndarray) -> np.ndarray:
        """
        Backprojection x = A^T y.
        """
        x = np.zeros(self.grid.nvox, dtype=np.float64)
        for slc, ray in zip(self.slices, self.rays):
            yy = y[slc]
            for i, (idx, ww) in enumerate(zip(ray.vox_idx, ray.vox_w)):
                if idx.size:
                    x[idx] += ww * yy[i]
        return x

    def solve(self, y_obs: np.ndarray, maxiter: int = 50, tol: float = 1e-4, positivity: bool = True) -> Tuple[np.ndarray, int]:
        """
        Solve (A^T W^2 A + lam L^T L) x = A^T W^2 y.
        """
        W = self.W

        def matvec(v: np.ndarray) -> np.ndarray:
            Av = self.A_times(v)
            W2Av = (W * W) * Av
            lhs = self.AT_times(W2Av) + self.lam * apply_LTL(v, self.grid, wt_r=self.wt_r)
            return lhs

        b = self.AT_times((W * W) * y_obs)
        Aop = LinearOperator((self.grid.nvox, self.grid.nvox), matvec=matvec, dtype=np.float64)

        x0 = np.zeros(self.grid.nvox, dtype=np.float64)
        # Some SciPy builds may not accept `tol` keyword; fall back if so.
        try:
            x, info = cg(Aop, b, x0=x0, maxiter=maxiter, tol=tol)
        except TypeError:
            x, info = cg(Aop, b, x0=x0, maxiter=maxiter)

        if positivity:
            x = np.maximum(x, 0.0)

        return x, info


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
    Render isosurfaces specified by plasma frequency (MHz).

    - If `save_png=True`, a PNG is always written (even if contours are empty).
    - Adds legend entries per frequency.
    """

    if np.isscalar(iso_freqs_mhz):
        freq_list = [float(iso_freqs_mhz)]
    else:
        freq_list = [float(f) for f in list(iso_freqs_mhz)]

    if colors is None:
        colors = ["tomato", "deepskyblue", "gold", "limegreen", "violet", "orange"]
    if len(colors) < len(freq_list):
        k = (len(freq_list) + len(colors) - 1) // len(colors)
        colors = (colors * k)[: len(freq_list)]

    if png_path is None:
        png_path = Path("tomo_isosurface.png")
    png_path = Path(png_path)

    nr, nth, nph = grid.nr, grid.nth, grid.nph

    rr, tt, pp = grid.voxel_centers_sph()
    ne3 = ne.reshape((nr, nth, nph), order="C")

    # close periodic boundary in phi (reduce seam artifacts)
    pp2 = np.concatenate([pp, pp[:, :, :1] + 2.0 * np.pi], axis=2)
    rr2 = np.concatenate([rr, rr[:, :, :1]], axis=2)
    tt2 = np.concatenate([tt, tt[:, :, :1]], axis=2)
    ne2 = np.concatenate([ne3, ne3[:, :, :1]], axis=2)

    xx = rr2 * np.sin(tt2) * np.cos(pp2)
    yy = rr2 * np.sin(tt2) * np.sin(pp2)
    zz = rr2 * np.cos(tt2)

    sg = pv.StructuredGrid(xx, yy, zz)
    sg["ne"] = ne2.ravel(order="F")

    pos = np.isfinite(ne) & (ne > 0)
    has_ne = bool(np.any(pos))

    if has_ne:
        ne_min = float(np.min(ne[pos]))
        ne_max = float(np.max(ne[pos]))
        fmin = fp_mhz_from_ne_cm3(ne_min, harmonic=harmonic)
        fmax = fp_mhz_from_ne_cm3(ne_max, harmonic=harmonic)
        flo, fhi = (min(fmin, fmax), max(fmin, fmax))
    else:
        flo, fhi = (np.nan, np.nan)

    p = pv.Plotter(off_screen=(not show_gui))
    p.set_background("white")

    if show_sun:
        p.add_mesh(pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60),
                   opacity=0.15, color="gray")

    if camera_lonlat is not None:
        lon, lat = np.deg2rad(camera_lonlat[0]), np.deg2rad(camera_lonlat[1])
        cam_dir = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=np.float64)
        cam_pos = (cam_dir * 15.0).tolist()
        p.camera_position = [cam_pos, [0, 0, 0], [0, 0, 1]]

    legend_entries = []
    any_mesh = False

    if has_ne:
        used_freqs = []
        used_ne_iso = []
        for f in freq_list:
            ff = float(np.clip(f, flo, fhi))
            if ff != f:
                print(f"[WARN] Requested f={f:.3f} MHz is outside reconstructed range; using nearest f={ff:.3f} MHz")
            used_freqs.append(ff)
            used_ne_iso.append(ne_cm3_from_fp_mhz(ff, harmonic=harmonic))

        for ff, ne_iso, col in zip(used_freqs, used_ne_iso, colors):
            contours = sg.contour(isosurfaces=[ne_iso], scalars="ne")
            if contours.n_points == 0:
                print(f"[WARN] Empty contour for f={ff:.3f} MHz (ne={ne_iso:.3e} cm^-3).")
                continue
            p.add_mesh(contours, color=col, opacity=opacity)
            legend_entries.append([f"f={ff:.1f} MHz (H={harmonic})", col])
            any_mesh = True

    if not any_mesh:
        p.add_text(
            "No isosurface rendered.\n"
            "Check r_use_min/max, pb_floor, and whether ne contains valid positive values.",
            position="upper_left",
            font_size=12,
            color="black",
        )
        if has_ne and np.isfinite(flo) and np.isfinite(fhi):
            if harmonic==1:
                p.add_text(f"Reconstructed f-range: {flo:.2f} .. {fhi:.2f} MHz (Fundamental)",
                       position="lower_left", font_size=10, color="black")
            else:
                p.add_text(f"Reconstructed f-range: {flo:.2f} .. {fhi:.2f} MHz (Second Harmonic)",
                       position="lower_left", font_size=10, color="black")
    else:
        if harmonic==1:
            p.add_text(f"Reconstructed f-range: {flo:.2f} .. {fhi:.2f} MHz (Fundamental)",
                       position="lower_left", font_size=10, color="black")
        else:
            p.add_text(f"Reconstructed f-range: {flo:.2f} .. {fhi:.2f} MHz (Second Harmonic)",
                       position="lower_left", font_size=10, color="black")
        if legend_entries:
            try:
                p.add_legend(legend_entries, bcolor="white", border=True)
            except Exception:
                pass

    if save_png:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        p.show(screenshot=str(png_path), auto_close=True)
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
        pb = block_reduce_mean(pb0, out_n) if orig_n != out_n else pb0.copy()
    pb = np.asarray(pb, dtype=np.float64)

    # x/y in Rsun (critical)
    x_map, y_map, rsun_arcsec = xy_rsun_for_rebinned_image(hdr, orig_n=orig_n, out_n=out_n)
    rho = np.hypot(x_map, y_map)  # Rsun

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
        mask = mask & np.isfinite(pb)

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

    idx_map = np.flatnonzero(mask.ravel())

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
    )


def main(args):
    """
    Run SSC/Ne3dTomo-like preprocessing + regularized tomography WITHOUT argparse.
    Edit the parameters in the `if __name__ == "__main__":` block at the bottom.
    """

    defaults = dict(
        pb_fits=[],
        out_n=128,

        default_lonlat="",
        lonlat_file="",

        r_min=1.5,
        r_max=6.0,
        nr=40,
        nth=60,
        nph=120,

        ds=0.02,
        limb_u=DEFAULT_LIMB_U,

        filt=1,
        despike_nsig=6.0,
        despike_med=5,
        pb_floor="",
        dpa_deg=1.0,
        r_use_min=1.5,
        r_use_max=4.0,
        hm=6,
        wt_nr=1,

        lam=1.0,
        q_low=0.0,
        width_pix=2.0,
        maxiter=40,
        tol=1e-4,

        save_prepped_dir="",
        save_ne_npz="",

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

    if not args.pb_fits:
        raise ValueError("pb_fits is empty. Set PB_FITS list in the __main__ block.")

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

    pb_overrides = {}
    if args.filt and len(pb_paths) >= 2:
        cube = []
        for p in pb_paths:
            pb0, _ = read_fits_image(p)
            pb1 = block_reduce_mean(pb0, args.out_n) if pb0.shape[0] != args.out_n else pb0
            cube.append(pb1.astype(np.float64))
        cube = despike_pb_cube(np.stack(cube, axis=0), nsig=args.despike_nsig, use_log=True)
        for p, arr in zip(pb_paths, cube):
            pb_overrides[p] = arr

    r_edges = np.linspace(args.r_min, args.r_max, args.nr + 1)
    th_edges = np.linspace(0.0, np.pi, args.nth + 1)
    ph_edges = np.linspace(0.0, 2.0 * np.pi, args.nph + 1)
    grid = SphericalGrid(r_edges=r_edges, th_edges=th_edges, ph_edges=ph_edges)

    save_prepped_dir = Path(args.save_prepped_dir) if args.save_prepped_dir else None

    obs_list: List[Observation] = []
    y_list: List[np.ndarray] = []
    ybk_list: List[Tuple[np.ndarray, np.ndarray]] = []

    for p in pb_paths:
        obs = build_observation(
            pb_fits=p,
            out_n=args.out_n,
            pb_override=pb_overrides.get(p),
            apply_spatial_despike=(p not in pb_overrides),
            r_use_min=args.r_use_min,
            r_use_max=args.r_use_max,
            limb_u=args.limb_u,
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
        obs_list.append(obs)

        y_vec = obs.pb.ravel()[obs.idx_map]
        y_list.append(y_vec)

        rgrid, ybk, _ = ybk_profile_fft(
            pb=obs.pb, hdr=obs.hdr,
            rmin=args.r_use_min, rmax=args.r_use_max,
            dpa_deg=args.dpa_deg, nr=240, hm=args.hm,
            width_pix=args.width_pix, q_low=args.q_low
        )
        ybk_list.append((rgrid, ybk))

        vv = y_vec[np.isfinite(y_vec)]
        if vv.size:
            print(f"[INFO] {p.name}: pB (used pixels) min/med/max = {np.min(vv):.3e} / {np.median(vv):.3e} / {np.max(vv):.3e}")

    y_obs = np.concatenate(y_list) if y_list else np.array([], dtype=float)
    if y_obs.size == 0 or (not np.any(np.isfinite(y_obs))):
        raise ValueError("y_obs is empty or all-NaN. Check masks and preprocessing (r_use_min/max, pb_floor).")

    rays = [build_rays_for_observation(obs=o, grid=grid, ds_rsun=args.ds, r_min=args.r_min, r_max=args.r_max, limb_u=args.limb_u)
            for o in obs_list]

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

            floor = float(np.nanpercentile(ybk_clean[good], 5))
            if not np.isfinite(floor) or floor <= 0:
                floor = float(np.nanmin(ybk_clean[good]))
            floor = max(floor, 1e-30)

            wt_r = 1.0 / np.maximum(ybk_clean, floor)
            wt_r = np.where(np.isfinite(wt_r) & (wt_r > 0), wt_r, 0.0)

    tomo = RegularizedTomography(grid, obs_list, rays, lam=args.lam, wt_r=wt_r)
    ne_raw, info = tomo.solve(y_obs, maxiter=args.maxiter, tol=args.tol, positivity=True)

    if info != 0:
        print(f"[WARN] CG did not fully converge (info={info}). Consider stronger regularization or more images.")

    y_pred = tomo.A_times(ne_raw)
    W = tomo.W
    m = np.isfinite(y_obs) & np.isfinite(y_pred) & np.isfinite(W) & (y_pred != 0)
    if np.count_nonzero(m) > 100:
        w2 = (W[m] * W[m])
        num = float(np.sum(w2 * y_pred[m] * y_obs[m]))
        den = float(np.sum(w2 * y_pred[m] * y_pred[m]))
        scale = num / den if den > 0 else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
    else:
        scale = 1.0

    ne = ne_raw * scale

    pos = np.isfinite(ne) & (ne > 0)
    if np.any(pos):
        fmin = fp_mhz_from_ne_cm3(float(np.min(ne[pos])), harmonic=args.harmonic)
        fmax = fp_mhz_from_ne_cm3(float(np.max(ne[pos])), harmonic=args.harmonic)
        print(f"[INFO] Reconstructed plasma-frequency range (harm={args.harmonic}): {fmin:.3f} .. {fmax:.3f} MHz")
    else:
        print("[WARN] ne has no positive finite values after scaling.")

    if args.save_ne_npz:
        out = Path(args.save_ne_npz)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            ne=ne.astype(np.float32),
            ne_raw=ne_raw.astype(np.float32),
            scale_brightness=float(scale),
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
        save_png=True,
        png_path=png_path,
        colors=getattr(args, "iso_colors", None),
    )


if __name__ == "__main__":
    from types import SimpleNamespace

    PB_FITS = [
        "/mnt/d/wsl/home/kinno-7010/Research/Tomography/Rawdata/pB_Kcor_LASCO_axi_20220613_0300.fits",
        "/mnt/d/wsl/home/kinno-7010/Research/Tomography/Rawdata/COR1A_pb_pre_20220613_030100.fits",
    ]

    DEFAULT_LONLAT = "0.0,0.0"
    LONLAT_FILE = ""

    OUT_N = 128

    # Reconstruction grid
    R_MIN, R_MAX = 2.2, 4.0
    NR, NTH, NPH = 40, 60, 120

    DS = 0.01
    LIMB_U = DEFAULT_LIMB_U

    FILT = 1
    DESPIKE_NSIG = 6.0
    DESPIKE_MED = 5

    PB_FLOOR = ""

    DPA_DEG = 1.0
    R_USE_MIN, R_USE_MAX = 1.5, 4.0
    HM = 6

    WT_NR = 1
    LAM = 1.0
    Q_LOW = 0.0
    WIDTH_PIX = 2.0

    MAXITER = 40
    TOL = 1e-4
    
    # Visualization (isosurfaces specified by plasma frequency)
    SHOW_GUI = True
    HARMONIC = 2

    FREQ_MHZ_LIST = [25.0] #, 31.0, 40.0]
    ISO_COLORS = ["tomato"] #, "deepskyblue", "gold"]

    SAVE_PREPPED_DIR = "/mnt/d/wsl/home/kinno-7010/Research/Tomography/Rawdata/tomo_prepped"
    SAVE_NE_NPZ = f"/mnt/d/wsl/home/kinno-7010/Research/Tomography/Rawdata/ne3d_solution_"+\
        "-".join(str(f) for f in FREQ_MHZ_LIST)+"MHz.npz"



    SAVE_PNG_PATH = f"/mnt/d/wsl/home/kinno-7010/Research/Tomography/output/tomo_" + \
        "-".join(str(f) for f in FREQ_MHZ_LIST) + "MHz.png"

    args = SimpleNamespace(
        pb_fits=PB_FITS,
        out_n=OUT_N,
        default_lonlat=DEFAULT_LONLAT,
        lonlat_file=LONLAT_FILE,

        r_min=R_MIN, r_max=R_MAX, nr=NR, nth=NTH, nph=NPH,
        ds=DS, limb_u=LIMB_U,

        filt=FILT,
        despike_nsig=DESPIKE_NSIG,
        despike_med=DESPIKE_MED,
        pb_floor=PB_FLOOR,
        dpa_deg=DPA_DEG,
        r_use_min=R_USE_MIN,
        r_use_max=R_USE_MAX,
        hm=HM,
        wt_nr=WT_NR,

        lam=LAM,
        q_low=Q_LOW,
        width_pix=WIDTH_PIX,
        maxiter=MAXITER,
        tol=TOL,

        save_prepped_dir=SAVE_PREPPED_DIR,
        save_ne_npz=SAVE_NE_NPZ,

        show_gui=SHOW_GUI,
        freq_mhz_list=FREQ_MHZ_LIST,
        harmonic=HARMONIC,
        iso_colors=ISO_COLORS,

        save_png=True,
        png_path=SAVE_PNG_PATH,
    )

    main(args)
