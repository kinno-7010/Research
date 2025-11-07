"""
secchi_prep.py — Python translation of SECCHI first-stage calibration + background subtraction

This module mirrors the entry-stage logic in IDL:
  secchi_prep -> cor_prep -> cor1_calibrate (+optional background subtraction for COR1)

Implemented steps:
  (1) EXPTIME normalization (DN -> DN/s)
  (2) constant bias subtraction (BIASMEAN / OFFSETCR)
  (3) CALFAC multiplication if present
  (4) flat-field/vignetting division if a calibration frame is provided
  (5) background subtraction if a background frame is provided (COR1 practice)
  (6) optional rectify placeholder (no-op by default)

Notes:
  - The official background-frame retrieval (e.g., SCC_GETBKGIMG) is not in the provided IDL files,
    so this module exposes 'bkgimg' hooks to pass a background image explicitly.
  - For full COR1 polarization processing (tB/pB/μ), use a separate module that reproduces cor_polariz/cor_mueller.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import os
import numpy as np

# ----------------------
# Point-object (cosmic-ray) discriminator (IDL: discri_pobj analogue)
# ----------------------
import numpy as _np

def _local_cross_median(img: _np.ndarray) -> _np.ndarray:
    img = _np.asarray(img, dtype=_np.float64)
    pad = _np.pad(img, 1, mode="edge")
    up    = pad[:-2, 1:-1]
    down  = pad[ 2:, 1:-1]
    left  = pad[1:-1, :-2]
    right = pad[1:-1,  2:]
    stk = _np.stack([up, down, left, right], axis=0)
    return _np.median(stk, axis=0)

def discri_pobj_filter(data: _np.ndarray, *, thres: float, bias: float, max_iter: int = 1):
    im = _np.asarray(data, dtype=_np.float64)
    nflag_total = 0
    for _ in range(max_iter):
        loc = _local_cross_median(im)
        denom = _np.abs(loc) + 1.0
        diff = im - loc
        mask = (diff > bias) & ((diff / denom) > thres)
        if not mask.any():
            break
        nflag = int(mask.sum())
        nflag_total += nflag
        im = _np.where(mask, loc, im)
    return im, nflag_total


try:
    from astropy.io import fits
    
    # Import SEB IP correction shim (IDL: SCC_SEBIP)
    try:
        from scc_sebip import scc_sebip
    except Exception:
        scc_sebip = None
except Exception as e:
    fits = None

def _ensure_astropy():
    if fits is None:
        raise RuntimeError(
            "astropy is required but not available in this notebook environment. "
            "Please `pip install astropy` in your target environment."
        )


def read_fits(path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a FITS image and return (data, header_dict) with data as float64."""
    _ensure_astropy()
    with fits.open(str(path), memmap=False) as hdul:
        hdu = hdul[0]
        data = np.array(hdu.data, dtype=np.float64)
        # Make a pythonic header dictionary (case-insensitive via upper keys)
        hdr = {k.upper(): v for k, v in hdu.header.items()}
    return data, hdr


def write_fits(path: Union[str, Path], data: np.ndarray, header: Dict[str, Any]) -> None:
    """Write a single-HDU FITS with the provided header dict."""
    _ensure_astropy()
    # Convert header dict back to fits.Header
    hdu = fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32))
    hdr = fits.Header()
    for k, v in header.items():
        # Basic sanitation: FITS keywords must be <= 8 characters; pass-through otherwise
        key = (k or "").strip()
        if not key:
            continue
        try:
            hdr[key] = v
        except Exception:
            # Fall back to COMMENT if type is unsupported
            try:
                hdr.add_comment(f"{key}={v!r}")
            except Exception:
                pass
    hdu.header.extend(hdr, strip=True, update=True)
    hdu.writeto(str(path), overwrite=True)


def normalize_exptime(data: np.ndarray, hdr: Dict[str, Any], exptime_off: bool=False) -> Tuple[np.ndarray, float]:
    """Divide by exposure time (DN -> DN/s). If EXPTIME missing, assume 1.0 when exptime_off is True."""
    if exptime_off:
        return data, 1.0
    exptime = hdr.get("EXPTIME", 1.0)
    if exptime in (None, 0.0):
        exptime = 1.0
    out = data / float(exptime)
    return out, float(exptime)


def subtract_bias(data: np.ndarray, hdr: Dict[str, Any], bias_off: bool=False) -> Tuple[np.ndarray, float]:
    """Subtract constant CCD bias. Use BIASMEAN if available, else OFFSETCR, else 0."""
    if bias_off:
        return data, 0.0
    bias = hdr.get("BIASMEAN", None)
    if bias is None:
        bias = hdr.get("OFFSETCR", 0.0)
    try:
        bias = float(bias)
    except Exception:
        bias = 0.0
    return data - bias, bias


def apply_calfac(data: np.ndarray, hdr: Dict[str, Any], calfac_off: bool=False) -> Tuple[np.ndarray, float]:
    """Apply photometric calibration factor if available. If absent or 0, use 1.0."""
    if calfac_off:
        return data, 1.0
    calfac = hdr.get("CALFAC", 1.0)
    try:
        calfac = float(calfac)
        if calfac == 0.0:
            calfac = 1.0
    except Exception:
        calfac = 1.0
    return data * calfac, calfac


def apply_flatfield_vignetting(data: np.ndarray, calimg: Optional[np.ndarray], calimg_off: bool=False) -> np.ndarray:
    """Divide by flat-field/vignetting image if provided and not disabled."""
    if calimg_off or calimg is None:
        return data
    # Safeguard against zeros
    safe = np.where(calimg == 0, 1.0, calimg.astype(np.float64))
    return data / safe


def subtract_background(
    data: np.ndarray,
    bkgimg: Optional[np.ndarray],
    bkgimg_off: bool=True
) -> np.ndarray:
    """Subtract a background image if provided and if not disabled.
    By default this is OFF because the official retrieval logic (SCC_GETBKGIMG)
    is not included in the provided .pro files.
    """
    if bkgimg_off or bkgimg is None:
        return data
    return data - bkgimg.astype(np.float64)


def rectify_if_needed(data: np.ndarray, hdr: Dict[str, Any], rectify: bool=True) -> np.ndarray:
    """Apply a simple rotation if RECTIFY keyword indicates it. 
    The official SECCHI_RECTIFY does more than just rotate; here we handle the common 90° CCW case.
    """
    if not rectify:
        return data
    rect = hdr.get("RECTIFY", None)
    rectrota = hdr.get("RECTROTA", None)
    # In many COR1 files, RECTIFY=T & RECTROTA=1 => rotate 90 deg CCW applied already.
    # If RECTIFY is already True in header, we assume it is applied. No-op by default.
    # If you want to force a rotation based on RECTROTA, uncomment below.
    # if isinstance(rectrota, (int, float)) and int(rectrota) == 1:
    #     return np.rot90(data, k=1)
    return data


def first_stage_calibration_and_background(
    filename: Union[str, Path],
    *,
    detector_hint: Optional[str]="COR1",
    exptime_off: bool=False,
    bias_off: bool=False,
    calfac_off: bool=False,
    calimg_off: bool=True,
    bkgimg_off: bool=True,
    calimg: Optional[np.ndarray]=None,
    bkgimg: Optional[np.ndarray]=None,
    rectify: bool=True,
    auto_bkg: bool=False,
    secchi_bkg_dir: Optional[str]=None,
    bkg_mode_priority=None,
    totalb: bool=False,
    double_totalb: bool=False,
    interpolate_bkg: bool=False,
    search_window_days: int=60,
    roll_tag: bool=False,
    silent: bool=False,
    nocalfac_butcorrforipsum: bool=False,
    discri_pobj_on: Optional[Union[bool, Tuple[float, float]]]=None,
    sebip_off: bool=False
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:

    """Translate the first-stage steps of secchi_prep/cor_prep/cor1_calibrate into Python for one image.
    
    Steps:
      1) Read FITS
      2) Normalize by EXPTIME  (unless exptime_off)
      3) Subtract BIASMEAN/OFFSETCR (unless bias_off)
      4) Multiply by CALFAC (if present; unless calfac_off)
      5) Divide by flat-field/vignetting `calimg` (unless calimg_off or None)
      6) Subtract background `bkgimg` (unless bkgimg_off or None)  [COR1-specific practice]
      7) Optionally rectify (no-op placeholder here)
    
    Returns:
      (processed_data, updated_header, info_dict)
        - updated_header contains simple history notes for your own record
        - info_dict reports the numerical factors used
    """
    data, hdr = read_fits(filename)
    # Pre-SEBIP raw fields for later background conditioning
    exptime_raw = hdr.get('EXPTIME', 1.0)
    ipsum_raw = hdr.get('IPSUM', hdr.get('IP_SUM', 1))
    bkg_hdr = None
    # --- SEB IP correction (IDL: SCC_SEBIP) applied first ---
    if (not sebip_off) and (scc_sebip is not None):
        try:
            data, hdr, ip_flag = scc_sebip(data, hdr, silent=silent)
        except Exception as _e:
            if not silent:
                print(f'[secchi_prep.py] SEBIP correction failed: {_e}')


    # Auto background retrieval (if requested)
    if auto_bkg and bkgimg is None:
        try:
            _img, _bhdr, _path = auto_background_for_cor1(hdr, base_dir=secchi_bkg_dir, mode_priority=bkg_mode_priority, totalb=totalb, double_totalb=double_totalb, interpolate=interpolate_bkg, search_window_days=search_window_days, roll_tag=roll_tag, silent=silent)
            if _img is not None:
                bkgimg = _img
                bkg_hdr = _bhdr
                if not silent:
                    print(f"[secchi_prep.py] Auto background: {_path}")
        except Exception as e:
            if not silent:
                print(f"[secchi_prep.py] Auto background retrieval failed: {e}")


    # Snapshot initial info
    info: Dict[str, Any] = {
        "filename": str(filename),
        "detector": hdr.get("DETECTOR", detector_hint),
        "exptime_used": None,
        "bias_used": None,
        "calfac_used": None,
        "ipsum_correction": None,
        "discri_pobj_flags": None,
        "applied_flatfield": calimg is not None and not calimg_off,
        "applied_background": bkgimg is not None and not bkgimg_off,
    }

    # -- Optional: Correct for IP summing (IDL: NOCALFAC_BUTCORRFORIPSUM)
    if nocalfac_butcorrforipsum:
        try:
            ip = int(ipsum_raw) if ipsum_raw not in (None, '') else 1
        except Exception:
            ip = 1
        divfactor = (2 ** max(0, ip-1)) ** 2
        if divfactor != 0:
            data = data / float(divfactor)
            hdr['IPSUM'] = 1
            bunit = str(hdr.get('BUNIT','DN')).strip()
            if '/CCDPIX' not in bunit.upper():
                hdr['BUNIT'] = (bunit + '/CCDPIX') if bunit else 'DN/CCDPIX'
        hist = hdr.get('HISTORY', [])
        if not isinstance(hist, (list, tuple)):
            hist = [hist] if hist else []
        hist.append(f'Applied IPSUM-only correction (divfactor={divfactor}) and set IPSUM=1 (NOCALFAC_BUTCORRFORIPSUM)')
        hdr['HISTORY'] = hist

# 1) Exposure normalization
    data, exptime_used = normalize_exptime(data, hdr, exptime_off=exptime_off)
    info["exptime_used"] = exptime_used
    # 2) Bias subtraction
    data, bias_used = subtract_bias(data, hdr, bias_off=bias_off)
    info["bias_used"] = bias_used
    # 3) Photometric calibration factor
    data, calfac_used = apply_calfac(data, hdr, calfac_off=calfac_off)
    info["calfac_used"] = calfac_used
    # 4) Flat-field / vignetting
    data = apply_flatfield_vignetting(data, calimg=calimg, calimg_off=calimg_off)
    # 5) Background subtraction (COR1 practice, angle-dependent)
    #    Precondition background to match IPSUM/EXPTIME/CCDSUM if header is available
    if (bkgimg is not None) and (not bkgimg_off):
        try:
            ccdsum_img = int(hdr.get('CCDSUM', 1))
            ccdsum_bkg = int(bkg_hdr.get('CCDSUM', 1)) if bkg_hdr else ccdsum_img
        except Exception:
            ccdsum_img = ccdsum_bkg = 1
        if bkg_hdr is not None and (ccdsum_img != ccdsum_bkg):
            if not silent:
                print(f"[secchi_prep.py] WARNING: CCDSUM mismatch (img={ccdsum_img}, bkg={ccdsum_bkg}); background subtraction skipped.")
            bkgimg = None
        else:
            try:
                ip_img = int(hdr.get('IPSUM', 1))
                ip_bkg = int(bkg_hdr.get('IPSUM', 1)) if bkg_hdr else ip_img
            except Exception:
                ip_img = ip_bkg = 1
            sumdif = ip_img - ip_bkg
            if sumdif != 0:
                bkgimg = bkgimg * (4.0 ** sumdif)
            if exptime_off:
                try:
                    ex = float(exptime_raw) if exptime_raw not in (None, 0.0) else 1.0
                except Exception:
                    ex = 1.0
                bkgimg = bkgimg * ex
    data = subtract_background(data, bkgimg=bkgimg, bkgimg_off=bkgimg_off)
    # 6) Cosmic-ray / point-object discrimination (IDL: discri_pobj)
    if discri_pobj_on:
        det = str(hdr.get('DETECTOR', 'COR1')).upper()
        if isinstance(discri_pobj_on, (tuple, list)) and len(discri_pobj_on) >= 2:
            th, bs = float(discri_pobj_on[0]), float(discri_pobj_on[1])
        else:
            if 'COR2' in det:
                th, bs = 0.01, 0.0
            else:
                th, bs = 0.10, 800.0
        data, nflag = discri_pobj_filter(data, thres=th, bias=bs, max_iter=1)
        info['discri_pobj_flags'] = int(nflag)
        hist = hdr.get('HISTORY', [])
        if not isinstance(hist, (list, tuple)):
            hist = [hist] if hist else []
        hist.append(f'Applied discri_pobj (thres={th}, bias={bs}, nflag={nflag})')
        hdr['HISTORY'] = hist
    
    # 7) Rectify (placeholder)
    data = rectify_if_needed(data, hdr, rectify=rectify)

    # Update header minimally to reflect ops performed (non-destructive)
    hist = hdr.get("HISTORY", [])
    if not isinstance(hist, (list, tuple)):
        hist = [hist] if hist else []
    def add_hist(msg: str):
        hist.append(msg)
    if not exptime_off:
        add_hist(f"Exposure Normalized to 1 Second (EXPTIME={exptime_used})")
        hdr["EXPTIME"] = 1.0  # reflect normalization
    if not bias_off:
        add_hist(f"Bias Subtracted (BIAS={bias_used})")
        hdr["OFFSETCR"] = float(bias_used)
    if not calfac_off:
        add_hist(f"Applied Calibration Factor (CALFAC={calfac_used})")
        hdr["CALFAC"] = float(calfac_used)
    if calimg is not None and not calimg_off:
        add_hist("Applied Flat-field/Vignetting (calimg provided)")
    if bkgimg is not None and not bkgimg_off:
        add_hist("Applied Background Subtraction (bkgimg provided)")
    hdr["HISTORY"] = hist

    if not silent:
        print(f"[secchi_prep.py] Processed {filename}")
        print(f"  Detector     : {info['detector']}")
        print(f"  EXPTIME used : {info['exptime_used']}")
        print(f"  Bias used    : {info['bias_used']}")
        print(f"  CALFAC used  : {info['calfac_used']}")
        print(f"  Flat/Vignette: {info['applied_flatfield']}")
        print(f"  Background   : {info['applied_background']}")

    return data, hdr, info


def save_processed(
    in_fits: Union[str, Path],
    out_fits: Union[str, Path],
    *,
    calimg: Optional[np.ndarray]=None,
    bkgimg: Optional[np.ndarray]=None,
    exptime_off: bool=False,
    bias_off: bool=False,
    calfac_off: bool=False,
    calimg_off: bool=True,
    bkgimg_off: bool=True,
    rectify: bool=True,
    silent: bool=False,
    auto_bkg: bool=False,
    secchi_bkg_dir: Optional[str]=None,
    bkg_mode_priority=None,
    totalb: bool=False,
    double_totalb: bool=False,
    interpolate_bkg: bool=False,
    search_window_days: int=60,
    roll_tag: bool=False,
    nocalfac_butcorrforipsum: bool=False,
    discri_pobj_on: Optional[tuple]=None,
    sebip_off: bool=False
) -> Dict[str, Any]:
    """Convenience wrapper to read, process, and write a FITS file."""
    data, hdr, info = first_stage_calibration_and_background(
        in_fits,
        exptime_off=exptime_off,
        bias_off=bias_off,
        calfac_off=calfac_off,
        calimg_off=calimg_off,
        bkgimg_off=bkgimg_off,
        calimg=calimg,
        bkgimg=bkgimg,
        rectify=rectify,
        auto_bkg=auto_bkg,
        secchi_bkg_dir=secchi_bkg_dir,
        bkg_mode_priority=bkg_mode_priority,
        totalb=totalb,
        double_totalb=double_totalb,
        interpolate_bkg=interpolate_bkg,
        search_window_days=search_window_days,
        roll_tag=roll_tag,
        nocalfac_butcorrforipsum=nocalfac_butcorrforipsum,
        discri_pobj_on=discri_pobj_on,
        sebip_off=sebip_off,
        silent=silent
    )
    write_fits(out_fits, data, hdr)
    info["out_fits"] = str(out_fits)
    return info


def example_usage():
    """Example (edit paths to your environment):
    
    in_path = '/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/20220613_030136_n4c1A.fts'
    out_path = '/path/to/output/20220613_030136_n4c1A_L1_firststage.fits'

    # If you have calibration and background frames, load them as numpy arrays:
    calimg = None  # e.g., read_fits('/path/to/calimg.fits')[0]
    bkgimg = None  # e.g., read_fits('/path/to/background_for_angle240.fits')[0]

    info = save_processed(
        in_path, out_path,
        calimg=calimg,     # divide by calimg if provided
        bkgimg=bkgimg,     # subtract bkg if provided
        exptime_off=False, # apply EXPTIME normalization
        bias_off=False,    # subtract bias
        calfac_off=False,  # apply CALFAC if present in header
        calimg_off=(calimg is None),
        bkgimg_off=(bkgimg is None),
        rectify=True,
        silent=False
    )
    print(info)
    """
    pass




# ----------------------
# Background retrieval (Python port of SCC_GETBKGIMG key logic)
# ----------------------

import os
from datetime import datetime, timedelta
from glob import glob

def _hdr_get(hdr: Dict[str, Any], key: str, default=None):
    return hdr.get(key, default)

def _parse_obs_time(hdr: Dict[str, Any]) -> datetime:
    # Prefer DATE-AVG then DATE-OBS, else DATE
    for k in ("DATE-AVG", "DATE_AVG", "DATE-OBS", "DATE_OBS", "DATE"):
        v = hdr.get(k)
        if v:
            try:
                # Accept fractional seconds
                return datetime.fromisoformat(str(v).replace('Z',''))
            except Exception:
                pass
    # Fallback: now
    from datetime import datetime as _dt
    return _dt.utcnow()

def _spacecraft_from_hdr(hdr: Dict[str, Any]) -> str:
    # OBSRVTRY = 'STEREO_A' or 'STEREO_B'
    obs = str(hdr.get("OBSRVTRY", "")).upper()
    if "A" in obs:
        return "a"
    if "B" in obs:
        return "b"
    # try filename (e.g., ..._c1A.fts)
    fn = str(hdr.get("FILENAME",""))
    import re as _re
    m = _re.search(r'([c|h|e]\d?)([AB])\.ft', fn, flags=_re.IGNORECASE)
    if m:
        return m.group(2).lower()
    return "a"

def _camera_from_hdr(hdr: Dict[str, Any]) -> str:
    tel = str(hdr.get("DETECTOR", "")).upper()
    if tel.startswith("COR1"):
        return "c1"
    if tel.startswith("COR2"):
        return "c2"
    if tel.startswith("HI1"):
        return "h1"
    if tel.startswith("HI2"):
        return "h2"
    if "EUVI" in tel:
        return "eu"
    # fallback
    return (tel[:1] + tel[-1:]).lower()

def _polstring_from_hdr(hdr: Dict[str, Any], *, totalb: bool=False, double_totalb: bool=False) -> str:
    if totalb and double_totalb:
        return "_dbTB_"
    if totalb:
        return "_pTBr_"
    try:
        pol = float(hdr.get("POLAR", -999))
    except Exception:
        pol = -999
    if pol >= 0 and pol < 361:
        return f"_p{int(round(pol)):03d}_"
    # "DOUBLE" program special-case
    if str(hdr.get("SEB_PROG","")).strip().upper().startswith("DOUBLE"):
        return "_dbTB_"
    # default to TBr
    return "_pTBr_"

def _mode_to_root(sc: str, mode: str, base_dir: Optional[str]) -> Tuple[str, str]:
    """
    Map background mode to (subdir, fchar).
    mode ∈ {'roll_min','monthly_roll','daily_med','monthly_min'}
    """
    if mode == "roll_min":
        return ("roll_min", "r")
    if mode == "monthly_roll":
        return ("monthly_roll", "mr")
    if mode == "daily_med":
        return ("daily_med", "d")
    if mode == "monthly_min":
        return ("monthly_min", "m")
    # default
    return ("monthly_min", "m")

def _format_month_dir(dt: datetime) -> str:
    # IDL uses sdir=strmid(cal,0,6) -> 'YYYYMM'
    return dt.strftime("%Y%m")

def _format_date_for_filename(dt: datetime) -> str:
    # IDL uses sfil=strmid(cal,2,6) -> 'YYMMDD'
    return dt.strftime("%y%m%d")

def _make_search_glob(base_root: str, month_dir: str, fchar: str, cam: str, sc: str, polstring: str, datestr: str, postd: str="") -> str:
    # filesrch0 = rootdir + sdir + (fchar + cam + SC + polstring + sfil + postd + '.fts')
    from pathlib import Path as _Path
    return str(_Path(base_root) / month_dir / f"{fchar}{cam}{sc.upper()}{polstring}{datestr}{postd}.fts")

def _roll_suffix(hdr: Dict[str, Any], tel: str, *, enable: bool=False) -> str:
    # In IDL they sometimes append 'rX*' based on CROTA when using monthly_roll/roll_min for non-COR1.
    if not enable:
        return ""
    crota = hdr.get("CROTA", None)
    if crota is None:
        try:
            crota = float(hdr.get("SC_ROLL", 0.0))
        except Exception:
            crota = 0.0
    # round and take first digit (IDL uses r+ first of I3.3 string then wildcard)
    try:
        d = int(round(float(crota)))
    except Exception:
        d = 0
    d = (d + 360) % 360
    return f"r{d//100}*"  # coarse bin, e.g., r0*, r1*, r2*, r3*

def find_cor_background(
    hdr: Dict[str, Any],
    *,
    base_dir: Optional[str]=None,
    mode: str="monthly_min",
    totalb: bool=False,
    double_totalb: bool=False,
    interpolate: bool=False,
    search_window_days: int=60,
    roll_tag: bool=False,
    silent: bool=False,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], Optional[str]]:
    _ensure_astropy()
    tel = str(hdr.get("DETECTOR","")).upper()
    cam = _camera_from_hdr(hdr)
    sc = _spacecraft_from_hdr(hdr)
    polstring = _polstring_from_hdr(hdr, totalb=totalb, double_totalb=double_totalb)
    dt0 = _parse_obs_time(hdr)

    if base_dir is None:
        base_dir = os.environ.get("SECCHI_BKG")
    if not base_dir:
        if not silent:
            print("[find_cor_background] SECCHI_BKG is not set. Provide base_dir.")
        return None, None, None

    subdir, fchar = _mode_to_root(sc, mode, base_dir)
    from pathlib import Path as _Path
    root = str(_Path(base_dir) / sc / subdir)

    # primary attempt: same month, exact YYMMDD
    month_dir = _format_month_dir(dt0)
    datestr = _format_date_for_filename(dt0)
    postd = _roll_suffix(hdr, tel, enable=roll_tag and tel != 'COR1')
    pattern = _make_search_glob(root, month_dir, fchar, cam, sc, polstring, datestr, postd)
    from glob import glob as _glob
    files = sorted(_glob(pattern))
    if files:
        path = files[0]
        with fits.open(path, memmap=False) as hdul:
            return np.array(hdul[0].data, dtype=np.float64), {k.upper():v for k,v in hdul[0].header.items()}, path

    # If not found, search within +/- search_window_days, scanning by day.
    # When interpolate=True, we try to get the nearest before and after and average.
    before = None
    after = None
    from datetime import timedelta as _td
    for delta in range(1, search_window_days+1):
        for sgn in (-1, +1):
            dt = dt0 + _td(days=sgn*delta)
            month_dir = _format_month_dir(dt)
            datestr = _format_date_for_filename(dt)
            pattern = _make_search_glob(root, month_dir, fchar, cam, sc, polstring, datestr, postd)
            cand = sorted(_glob(pattern))
            if cand:
                if sgn < 0 and before is None:
                    before = cand[0]
                if sgn > 0 and after is None:
                    after = cand[0]
                if not interpolate:
                    path = cand[0]
                    with fits.open(path, memmap=False) as hdul:
                        return np.array(hdul[0].data, dtype=np.float64), {k.upper():v for k,v in hdul[0].header.items()}, path
                if before and after:
                    break
        if before and after:
            break

    if interpolate and (before or after):
        # If one side missing, fall back to the one we have.
        if before and after:
            with fits.open(before, memmap=False) as h1, fits.open(after, memmap=False) as h2:
                im1 = np.array(h1[0].data, dtype=np.float64)
                im2 = np.array(h2[0].data, dtype=np.float64)
                # Linear time interpolation weight
                t1 = _parse_obs_time({k.upper():v for k,v in h1[0].header.items()})
                t2 = _parse_obs_time({k.upper():v for k,v in h2[0].header.items()})
                t0 = dt0
                if t2 == t1:
                    w = 0.5
                else:
                    w = (t0 - t1).total_seconds() / (t2 - t1).total_seconds()
                    w = max(0.0, min(1.0, w))
                bkg = im1*(1.0-w) + im2*w
                # Return header of nearest
                if abs((t0-t1).total_seconds()) <= abs((t2-t0).total_seconds()):
                    hh = {k.upper():v for k,v in h1[0].header.items()}
                else:
                    hh = {k.upper():v for k,v in h2[0].header.items()}
                return bkg, hh, f"{before} (interp {w:.2f}) + {after}"
        else:
            path = before or after
            with fits.open(path, memmap=False) as hdul:
                return np.array(hdul[0].data, dtype=np.float64), {k.upper():v for k,v in hdul[0].header.items()}, path

    if not silent:
        print(f"[find_cor_background] No background found under {root} for pattern {fchar}{cam}{sc.upper()}{polstring}YYMMDD*.fts")
    return None, None, None


def auto_background_for_cor1(hdr: Dict[str, Any], *, mode_priority=None, **kwargs):
    if mode_priority is None:
        # Order inspired by SCC_GETBKGIMG options commonly used for COR: monthly_min, daily_med, monthly_roll, roll_min
        mode_priority = ["monthly_min", "daily_med", "monthly_roll", "roll_min"]
    for mode in mode_priority:
        img, h, path = find_cor_background(hdr, mode=mode, **kwargs)
        if img is not None:
            return img, h, path
    return None, None, None
