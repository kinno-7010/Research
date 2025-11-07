
"""
cor1_quickpol.py
----------------
Python translation of IDL routines:
 - COR1_QUICKPOL (fast tB, pB, MU for COR1 sequences with angles 0,120,240)
 - COR_POLARIZ   (general Stokes via Mueller matrix; simplified core)
 - COR_MUELLER   (builds inverse mapping from [im0, im120, im240] to [I,Q,U])

Notes
-----
* This module focuses on the *second stage* of the SECCHI/COR1 pipeline:
  deriving total brightness (tB), polarized brightness (pB), and the
  polarization angle (MU) from three polarized images acquired at angles
  0°, 120°, and 240° (STEREO/SECCHI-COR1 convention).

* It adheres closely to the provided IDL files:
  - cor1_quickpol.pro
  - cor_polariz.pro
  - cor_mueller.pro

* Assumptions:
  - Ideal polarizers (s = d = 0.5) as in cor_mueller.pro comments.
  - Header(s) may include 'POLAR' angles in degrees.
  - For /tangential or /radial weighting, we approximate solar-center
    pixel from WCS if available; otherwise we fall back to CRPIX1/CRPIX2.

Author: Auto-translation assistant
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union, List

import numpy as np

try:
    from astropy.io.fits import Header
except Exception:  # fallback stub if astropy not installed at authoring time
    Header = dict  # type: ignore

try:
    from astropy.wcs import WCS
except Exception:
    WCS = None  # type: ignore


def _append_history(h, msg: str):
    try:
        if hasattr(h, 'add_history'):
            h.add_history(msg)
            return
        if isinstance(h, dict):
            prev = h.get('HISTORY', None)
            if prev is None:
                h['HISTORY'] = [msg]
            elif isinstance(prev, list):
                prev.append(msg)
            else:
                h['HISTORY'] = [prev, msg]
    except Exception:
        pass



def _hdr_getf(h, key, default=None):
    if isinstance(h, dict):
        try:
            return float(h.get(key, default))
        except Exception:
            return default
    try:
        return float(h[key])
    except Exception:
        try:
            return float(h.get(key, default))  # type: ignore
        except Exception:
            return default



# ----------------------------------------------------------------------------
# COR_MUELLER (translated core)
# ----------------------------------------------------------------------------

def cor_mueller(hdr1: Header, hdr2: Header, hdr3: Header) -> np.ndarray:
    # Build inverse mapping from three analyzer angles to Stokes [I,Q,U].
    # Allow per-frame transmission coefficients via headers: K1, K2 (optional).
    def _angle_deg(h) -> float:
        ang = _hdr_getf(h, 'POLAR', 0.0)
        if ang is None:
            ang = 0.0
        return float(ang)
    def _kpair(h):
        k1 = _hdr_getf(h, 'K1', None)
        k2 = _hdr_getf(h, 'K2', None)
        if (k1 is None) or (k2 is None):
            return 1.0, 0.0  # ideal polarizer
        return float(k1), float(k2)

    angle1 = _angle_deg(hdr1)
    angle2 = _angle_deg(hdr2)
    angle3 = _angle_deg(hdr3)

    k11, k21 = _kpair(hdr1)
    k12, k22 = _kpair(hdr2)
    k13, k23 = _kpair(hdr3)

    s1, d1 = 0.5*(k11+k21), 0.5*(k11-k21)
    s2, d2 = 0.5*(k12+k22), 0.5*(k12-k22)
    s3, d3 = 0.5*(k13+k23), 0.5*(k13-k23)

    def row(angle_deg: float, s: float, d: float):
        ang = math.radians(2.0 * angle_deg)
        return [s, d * math.cos(ang), d * math.sin(ang)]

    X = np.array([
        row(angle1, s1, d1),
        row(angle2, s2, d2),
        row(angle3, s3, d3)
    ], dtype=np.float64)

    Xinv = np.linalg.inv(X)
    return Xinv

# ----------------------------------------------------------------------------
# COR_POLARIZ (translated core)
# ----------------------------------------------------------------------------
def cor_polariz(images: np.ndarray,
                hdrs: Sequence[Header],
                percent: bool = False,
                silent: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Core translation of cor_polariz.pro needed for (tB, pB, MU).

    Parameters
    ----------
    images : ndarray (ny, nx, 3)
        Three images from the *same* polarization sequence (0°,120°,240°).
    hdrs : sequence of 3 headers
    percent : bool
        If True, also produce percent polarization image in place of pB.
        (Here we return both; caller can convert if desired.)
    silent : bool

    Returns
    -------
    I, pB, MU : ndarrays (ny, nx)
        Total brightness, polarized brightness, polarization angle (radians).
        MU = 0.5 * atan2(U, Q) as in IDL (atan(U, Q)).
    """
    assert images.ndim == 3 and images.shape[-1] == 3, \
        "images must be (ny, nx, 3)"
    assert len(hdrs) == 3, "hdrs must be length 3"

    # Build the inverse mapping to Stokes
    Xinv = cor_mueller(hdrs[0], hdrs[1], hdrs[2])

    # Measured vector: [im1, im2, im3]^T (order consistent with hdrs)
    im1 = images[..., 0].astype(np.float64, copy=False)
    im2 = images[..., 1].astype(np.float64, copy=False)
    im3 = images[..., 2].astype(np.float64, copy=False)

    # Compute Stokes per pixel via linear combination
    # [I, Q, U] = Xinv @ [im1, im2, im3]
    # Expand with broadcasting: sum_k Xinv[j,k] * im_k
    I = Xinv[0, 0] * im1 + Xinv[0, 1] * im2 + Xinv[0, 2] * im3
    Q = Xinv[1, 0] * im1 + Xinv[1, 1] * im2 + Xinv[1, 2] * im3
    U = Xinv[2, 0] * im1 + Xinv[2, 1] * im2 + Xinv[2, 2] * im3

    pB = np.sqrt(Q * Q + U * U)
    MU = 0.5 * np.arctan2(U, Q)  # IDL: atan(U, Q)

    _append_history(hdrs[0], 'COR_POLARIZ: computed I,Q,U and pB')
    return I, pB, MU



def _normalize_angles_and_reorder(a: np.ndarray, hdrs: Optional[Sequence[Header]]) -> Tuple[np.ndarray, Optional[List[Header]], float]:
    # Ensure frames correspond to 0,120,240 ordering modulo a global rotation.
    if hdrs is None:
        return a, None, 0.0

    def _ang(h):
        ang = _hdr_getf(h, 'POLAR', 0.0)
        if ang is None:
            ang = 0.0
        ang = float(ang) % 180.0
        if ang < 0:
            ang += 180.0
        return ang

    angs = [ _ang(h) for h in hdrs ]
    a0 = angs[0]
    diffs = [ (ang - a0) % 180.0 for ang in angs ]
    targets = [0.0, 120.0, 60.0]  # expecting ~[0,120,240≡60]
    import itertools
    best_perm = (0,1,2)
    best_err = 1e99
    for perm in itertools.permutations(range(3)):
        err = sum((diffs[perm[i]] - targets[i])**2 for i in range(3))
        if err < best_err:
            best_err = err
            best_perm = perm
    a_re = a[:, :, list(best_perm), ...]
    hdr_re = [ hdrs[ix] for ix in best_perm ]
    base_angle_deg = _hdr_getf(hdr_re[0], 'POLAR', 0.0) or 0.0
    return a_re, hdr_re, base_angle_deg

# ----------------------------------------------------------------------------
# COR1_QUICKPOL (translated, fast ideal-angle solver)
# ----------------------------------------------------------------------------
def cor1_quickpol(image: np.ndarray,
                  header: Optional[Sequence[Header]] = None,
                  double: bool = False,
                  tangential: bool = False,
                  radial: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast computation of tB, pB, MU for COR1 polarized triplets at 0°,120°,240°.

    Parameters
    ----------
    image : ndarray
        Accepts (ny, nx, 3*Ns) or (ny, nx, 3, Ns).
    header : optional sequence of headers per frame (length 3*Ns or (3, Ns)).
        Used to adjust MU by header[0].POLAR and for tangential/radial weighting.
    double : return float64 arrays (otherwise follows input dtype).
    tangential, radial : mutually exclusive weighting of pB relative to
        the local polar angle about the solar-center pixel. Requires WCS
        to infer center; falls back to CRPIX1/CRPIX2 if WCS absent.

    Returns
    -------
    totb, pb, mu : ndarrays
        Shapes: (ny, nx, Ns). MU in radians.
    """
    img = np.asarray(image)
    if img.ndim != 3 and img.ndim != 4:
        raise ValueError("IMAGE must be (ny, nx, 3*Ns) or (ny, nx, 3, Ns)")

    if img.ndim == 3:
        ny, nx, n3 = img.shape
        if n3 % 3 != 0:
            raise ValueError("Number of images must be divisible by 3")
        nseq = n3 // 3
        a = img.reshape(ny, nx, 3, nseq)
    else:
        ny, nx, n3, nseq = img.shape
        if n3 != 3:
            raise ValueError("Third dimension must be 3 for polarized triplets")
        a = img

    # Work in float64 for numeric parity with IDL double math, then cast.
    a64 = a.astype(np.float64, copy=False)

    # Ideal analyzer solution for angles (0°, 120°, 240°):
    # I = (2/3) * (i0 + i120 + i240)
    # Q = (2/3) * (2*i0 - i120 - i240)
    # U = (2/sqrt(3)) * (i240 - i120)
    i0    = a64[:, :, 0, :]
    i120  = a64[:, :, 1, :]
    i240  = a64[:, :, 2, :]

    I  = (2.0 / 3.0) * (i0 + i120 + i240)
    Q  = (2.0 / 3.0) * (2.0 * i0 - i120 - i240)
    U  = (2.0 / math.sqrt(3.0)) * (i240 - i120)

    pb = np.sqrt(Q * Q + U * U)        # polarized brightness
    totb = I                           # total brightness
    # MU from quick method:
    # The IDL quickpol uses a more intricate formula with sign disambiguation
    # using the relative magnitude of im3 vs. im2. We reproduce that logic here.
    mu = np.empty_like(pb)

    # Compute the initial magnitude-limited acos(...) estimate
    # mu0 = acos(sqrt( clamp( (i0 - 0.5*(I - pB)) / max(pB, pmin), 0, 1) ))
    # Choose pmin across valid (>0) pixels to stabilize division
    # (IDL uses min(pb[w]) where w indexes pb>0)
    # Here we do it per-sequence to mirror IDL's loop.
    for k in range(nseq):
        pbk = pb[..., k]
        Ik  = I[..., k]
        i0k = i0[..., k]
        # pmin = min(pb where pb>0), else 1.0
        mask_pos = pbk > 0
        pmin = float(pbk[mask_pos].min()) if np.any(mask_pos) else 1.0
        denom = np.maximum(pbk, pmin)
        arg = (i0k - 0.5 * (Ik - pbk)) / denom
        arg = np.clip(arg, 0.0, 1.0)
        mu0 = np.arccos(np.sqrt(arg))

        # Sign convention: if (im3 < im2) then mu0 := -mu0 (IDL logic)
        flip = i240[..., k] < i120[..., k]
        mu0 = np.where(flip, -mu0, mu0)
        mu[..., k] = mu0

    # If header(s) were provided, offset MU by header[0].POLAR (radians) and wrap.
    if header is not None and len(header) > 0:
        # Accept either list of length 3*nseq or nested (3, nseq)
        if isinstance(header, (list, tuple)) and len(header) in (3 * nseq, 3):
            # Use the first frame's POLAR per sequence
            def get_polar(h):
                if isinstance(h, dict):
                    return float(h.get('POLAR', 0.0))
                try:
                    return float(h['POLAR'])
                except Exception:
                    return float(h.get('POLAR', 0.0))  # type: ignore

            if len(header) == 3 * nseq:
                polars_deg = [get_polar(header[3*k + 0]) for k in range(nseq)]
            else:  # len==3, assume same triplet for all sequences
                polars_deg = [get_polar(header[0]) for _ in range(nseq)]
                
            # --- Base polarizer angle (deg): try header keys; default 0.0 for COR1 ---
            hdr0 = None
            if isinstance(header, (list, tuple)) and len(header) > 0:
                hdr0 = header[0]
            elif isinstance(header, dict):
                hdr0 = header
            else:
                hdr0 = {}

            def _safe_get(d, k):
                try:
                    return d.get(k)
                except Exception:
                    return None

            base_angle_deg = 0.0  # COR1 の既定は 0 度
            for key in ("POL_BASE", "POLBASE", "BASE_ANG", "BASEANG", "POLZERO",
                        "POL0", "POL_OFF", "POLOFF", "POLAR0", "POLANG0"):
                v = _safe_get(hdr0, key)
                if v is not None:
                    try:
                        base_angle_deg = float(v)
                        break
                    except Exception:
                        pass


            mu0 = np.array(polars_deg, dtype=np.float64) * (math.pi / 180.0)
            mu = mu + mu0[None, None, :] + (base_angle_deg * (math.pi/180.0))

            # wrap to [-pi/2, +pi/2]
            halfpi = 0.5 * math.pi
            # bring to (-pi, pi) then fold
            mu = (mu + math.pi) % (2.0 * math.pi) - math.pi
            mu = np.where(mu >  halfpi, mu - math.pi, mu)
            mu = np.where(mu < -halfpi, mu + math.pi, mu)

    # /tangential or /radial weighting
    if tangential or radial:
        # Build angle grid theta about solar center pixel
        if header is not None and len(header) > 0:
            h0 = header[0]
        else:
            h0 = {}
        # Attempt WCS center first
        cen_x, cen_y = None, None
        if WCS is not None:
            try:
                w = WCS(h0)
                # pixel coordinates of Sun center (0,0) in world coordinates
                cen = w.world_to_pixel_values(0.0, 0.0)
                cen_x, cen_y = float(cen[0]), float(cen[1])
            except Exception:
                pass
        if cen_x is None or cen_y is None:
            # fallback to CRPIX keywords
            def _get(h, key, default):
                if isinstance(h, dict):
                    return float(h.get(key, default))
                try:
                    return float(h[key])
                except Exception:
                    return float(h.get(key, default))  # type: ignore
            cen_x = _get(h0, 'CRPIX1', (nx - 1) / 2.0)
            cen_y = _get(h0, 'CRPIX2', (ny - 1) / 2.0)

        x = (np.arange(nx, dtype=np.float64) - cen_x)[None, :]
        y = (np.arange(ny, dtype=np.float64) - cen_y)[:, None]
        # broadcast to full image
        X = np.broadcast_to(x, (ny, nx))
        Y = np.broadcast_to(y, (ny, nx))
        theta = np.arctan2(Y, X)

        for k in range(nseq):
            dmu = mu[..., k] - theta
            corr = np.abs(np.cos(dmu))
            if radial:
                corr = np.abs(np.sin(dmu))
            pb[..., k] = pb[..., k] * corr

    # Cast outputs per IDL logic
    if double or (img.dtype == np.float64):
        dtype = np.float64
    else:
        dtype = np.float32
    totb = totb = I.astype(dtype, copy=False)
    pb   = pb.astype(dtype, copy=False)
    mu   = mu.astype(np.float64 if double else np.float32, copy=False)

    _append_history(header[0] if (header is not None and len(header)>0) else {}, 'COR1_QUICKPOL: computed tB,pB,mu (quick)')
    return totb, pb, mu


# Convenience wrapper mirroring IDL call signature
def COR1_QUICKPOL(image: np.ndarray,
                  double: bool = False,
                  header: Optional[Sequence[Header]] = None,
                  tangential: bool = False,
                  radial: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return cor1_quickpol(image=image,
                         header=header,
                         double=double,
                         tangential=tangential,
                         radial=radial)
