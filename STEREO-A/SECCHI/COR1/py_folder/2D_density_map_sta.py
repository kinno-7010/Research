#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D_density_map_sta.py
=====================

Compute a 2D electron-density (N_e) map from STEREO-A / SECCHI-COR1
polarized-brightness (pB) images.

Design goals
------------
- Follow the COR1 User's Guide pipeline:
  SECCHI_PREP  →  background subtraction (per polarizer)  →  COR1_QUICKPOL (tB, pB, MU)
  (cf. https://cor1.gsfc.nasa.gov/guide/ — "Subtracting backgrounds" and
   "Calculating Polarized Brightness" sections)
- Reuse the inversion machinery already used for K-Cor + LASCO in
  `/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/pB/py_folder/2D_density_map.py`:
  specifically, `invert_per_theta_profiles` and
  `build_density_map_from_profiles`.
- Apply the *same* COR1-specific background treatment used in `plot_sta_pB.py`
  (daily-median per-angle backgrounds; safe fallbacks if some angles missing).

Usage (example)
---------------
Update the paths in the __main__ block and run:

  $ python3 2D_density_map_sta.py

Notes
-----
* Field of view used here: 1.4–4.0 R_sun (typical COR1). Pixels inside r<1.4
  or outside r>4.0 are masked.
* Sector width for pB(r) extraction is ±5° (same as io_and_processing.extract_pB_profile).
* Radial binning uses ~2 px per shell (dr = 2 / px_per_rsun), clamped to [0.01, 0.2] R_sun.
"""

from __future__ import annotations

import os
import math
import numpy as np
from typing import Dict, Tuple, Sequence, Optional

from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -------------------------------------------------------------------
# Import local modules (two code-bases; add both to PYTHONPATH at run)
# -------------------------------------------------------------------
import sys
_THIS = os.path.abspath(os.path.dirname(__file__))
if _THIS not in sys.path:
    sys.path.append(_THIS)

# Path to the Mk4+LASCO utilities (for inversion/plot helpers)
SDO_MK4_SOHO_DIR = "/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/pB/py_folder"
if os.path.isdir(SDO_MK4_SOHO_DIR) and (SDO_MK4_SOHO_DIR not in sys.path):
    sys.path.append(SDO_MK4_SOHO_DIR)

# Local (COR1) helper modules
from secchi_prep import first_stage_calibration_and_background
from cor1_quickpol import COR1_QUICKPOL

# Dynamically import '2D_density_map.py' (filename begins with a digit)
import importlib.util as _ilu
_d2_path = os.path.join(SDO_MK4_SOHO_DIR, "2D_density_map.py")
_spec = _ilu.spec_from_file_location("d2map", _d2_path)
_d2 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_d2)  # type: ignore

# Re-export what we need under local names
angle_map_deg = _d2.angle_map_deg
digitize_radius = _d2.digitize_radius
invert_per_theta_profiles = _d2.invert_per_theta_profiles
build_density_map_from_profiles = _d2.build_density_map_from_profiles

# pB → N_e line-of-sight physics (kernels etc.) comes from constants_vdh,
# but we only need it indirectly via `invert_per_theta_profiles`.
from constants_vdh import R_SUN  # for physical consistency if needed


# -----------------------------
# Header / geometry convenience
# -----------------------------
def _params_from_header(hdr: dict) -> Dict[str, float]:
    nx = int(hdr.get('NAXIS1'))
    ny = int(hdr.get('NAXIS2'))
    cx = float(hdr.get('CRPIX1')) - 1.0
    cy = float(hdr.get('CRPIX2')) - 1.0
    scale = abs(float(hdr.get('CDELT1')))  # arcsec/px
    rsun_arc = float(hdr.get('RSUN_OBS', hdr.get('RSUN', 959.2)))  # arcsec
    px_per_rsun = rsun_arc / scale
    return dict(
        nx=nx, ny=ny, cx=cx, cy=cy, scale=scale,
        rsun_arc=rsun_arc,          # ← 従来キー
        rsun_arcsec=rsun_arc,       # ← 同義キー（可読名）
        px_per_rsun=px_per_rsun,
        arcsec_per_pix=scale        # ← これも明示
    )


def _radius_map(shape: Tuple[int, int], cx: float, cy: float, px_per_rsun: float) -> np.ndarray:
    ny, nx = shape
    y, x = np.indices((ny, nx))
    r_pix = np.hypot(x - cx, y - cy)
    return r_pix / float(px_per_rsun)


# ---------------------------------------------
# Daily-median per-angle background (safe I/O)
# ---------------------------------------------
def _read_fits(path: str) -> Tuple[np.ndarray, dict]:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float64)
        hdr = {k.upper(): v for k, v in hdul[0].header.items()}
    return data, hdr

def _parse_iso_as_datetime(dt_str):
    """'YYYY-MM-DDThh:mm:ss(.sss)' などを素直に datetime（naive UTC）へ。失敗時 None。"""
    from datetime import datetime
    if dt_str is None:
        return None
    s = str(dt_str).strip().replace('Z', '')
    try:
        if 'T' not in s and ' ' in s:
            s = s.replace(' ', 'T')
        return datetime.fromisoformat(s)
    except Exception:
        return None



def load_daily_med_backgrounds(p000: Optional[str] = None,
                               p120: Optional[str] = None,
                               p240: Optional[str] = None,
                               *,
                               bkg_dict: Optional[dict] = None) -> dict:
    """
    daily-med背景を読み込むユーティリティ。
    - 個別パス指定 (p000/p120/p240) か、
    - {0: (img,hdr),120:...,240:...} 形式の bkg_dict のどちらかで渡す。
    いずれも無い場合は例外。角度欠損はp000でフォールバック。
    """
    if bkg_dict is not None:
        # すでに読み込み済み
        out = {}
        for ang in (0, 120, 240):
            if ang in bkg_dict:
                out[ang] = bkg_dict[ang]
        if 0 not in out:
            raise FileNotFoundError("bkg_dict に 0° 背景が必要です。")
        if 120 not in out:
            out[120] = out[0]
            print("[daily_med] p120 background missing in dict; using p000 fallback.")
        if 240 not in out:
            out[240] = out[0]
            print("[daily_med] p240 background missing in dict; using p000 fallback.")
        return out

    def _read_fits(path: str) -> Tuple[np.ndarray, dict]:
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float64)
            hdr = {k.upper(): v for k, v in hdul[0].header.items()}
        return data, hdr

    im000 = h000 = im120 = h120 = im240 = h240 = None
    if p000 and os.path.exists(p000):
        im000, h000 = _read_fits(p000)
    if p120 and os.path.exists(p120):
        im120, h120 = _read_fits(p120)
    if p240 and os.path.exists(p240):
        im240, h240 = _read_fits(p240)

    if im000 is None:
        raise FileNotFoundError("p000 背景が必須です（p120/p240は欠損時p000で代替）。")
    if im120 is None:
        print("[daily_med] p120 background missing; using p000 fallback.")
        im120, h120 = im000, h000
    if im240 is None:
        print("[daily_med] p240 background missing; using p000 fallback.")
        im240, h240 = im000, h000

    return {0: (im000, h000), 120: (im120, h120), 240: (im240, h240)}


# --------------------------------------------------------
# Stage-1 calibration with COR1 daily-med background apply
# --------------------------------------------------------
def _secchi_prep_with_daily_bkg(raw_path: str,
                                bkg_img: Optional[np.ndarray],
                                *,
                                discri_pobj_on: bool | tuple = False,
                                sebip_off: bool = False,
                                silent: bool = True) -> Tuple[np.ndarray, dict]:
    """
    COR1 1枚の前処理：
      - secchi_prep では「IPSUM補正のみ」を画像に適用（CALFACは掛けない）
      - 戻りヘッダから CALFAC を IDL get_calfac と同じ式で計算し、画像に乗ずる
      - daily-med 背景は角度ごとに適用（bkg_img）
    戻り値: (前処理済み画像[物理単位 Bsun], 更新済みヘッダ)
    """
    # 1) まず secchi_prep 相当（CALFACは適用しない、IPSUMは画像側で補正）
    img, hdr, _info = first_stage_calibration_and_background(
        raw_path,
        exptime_off=False,
        bias_off=False,
        calfac_off=True,                 # ★ CALFAC はここでは適用しない
        nocalfac_butcorrforipsum=True,   # ★ 画像側で IPSUM 補正は必ず実行
        calimg_off=True,                 # 必要なら False に
        bkgimg_off=(bkg_img is None),
        calimg=None,
        bkgimg=bkg_img,
        rectify=False,
        auto_bkg=False,
        secchi_bkg_dir=None,
        discri_pobj_on=discri_pobj_on,
        sebip_off=sebip_off,
        silent=silent
    )

    # 2) ヘッダキーを大文字dict化（安全策）
    try:
        h = {str(k).upper(): v for k, v in hdr.items()}
    except Exception:
        h = hdr

    # 3) IDL get_calfac と同等のCALFACを計算（IPSUM割りはしない：画像側で済）
    calfac = compute_calfac_cor1_like_idl(h,
                                          apply_ipsum_correction=False,
                                          is_totalb=False)
    img = img * float(calfac)
    h['CALFAC'] = float(calfac)

    # 4) 履歴
    hist = h.get('HISTORY', [])
    if not isinstance(hist, (list, tuple)):
        hist = [hist] if hist else []
    hist.append(f'Applied external CALFAC(get_calfac-like): {calfac:.3e}')
    h['HISTORY'] = hist

    return img, h

def _prep_one_frame(raw_path: str,
                    bkg_img: Optional[np.ndarray],
                    bkg_hdr: Optional[dict],
                    *,
                    # CALFACが0/未使用でもIPSUM補正だけは必ず行う
                    nocalfac_butcorrforipsum: bool = True,
                    # 以下は既定の前処理フラグ（必要なら呼び出し側で変更）
                    exptime_off: bool = False,
                    bias_off: bool = False,
                    calfac_off: bool = False,
                    calimg_off: bool = True,
                    auto_bkg: bool = False,
                    rectify: bool = False,
                    discri_pobj_on: bool | tuple = False,
                    sebip_off: bool = False,
                    silent: bool = True) -> Tuple[np.ndarray, dict]:
    """
    1枚のCOR1フレームを前処理。CALFACが無効でもIPSUM(2x2サミング)補正を実施し、
    daily-med背景があれば角度ごとに適用する。
    """
    img, hdr, _info = first_stage_calibration_and_background(
        raw_path,
        exptime_off=exptime_off,
        bias_off=bias_off,
        calfac_off=calfac_off,
        nocalfac_butcorrforipsum=nocalfac_butcorrforipsum,  # ★これが重要
        calimg_off=calimg_off,
        bkgimg_off=(bkg_img is None),
        calimg=None,
        bkgimg=bkg_img,
        rectify=rectify,
        auto_bkg=auto_bkg,         # daily-medを渡すときはFalse推奨
        secchi_bkg_dir=None,
        discri_pobj_on=discri_pobj_on,
        sebip_off=sebip_off,
        silent=silent
    )
    return img, hdr


def build_pB_from_triplet(raw0: str, raw120: str, raw240: str,
                          daily_med_bkg: dict,
                          *,
                          # 互換性維持のため受け取るだけ。処理本体では使いません。
                          nocalfac_butcorrforipsum: bool = True,
                          discri_pobj_on: bool | tuple = False,
                          sebip_off: bool = False,
                          silent: bool = True) -> Tuple[np.ndarray, dict]:
    """
    0/120/240° を前処理 → COR1_QUICKPOL で pB を算出。
    ここでは secchi_prep では CALFAC を掛けず（IPSUMのみ画像補正）、
    IDL get_calfac と同等式で外部CALFACを乗ずる。
    """
    (b0,   _h0b)   = daily_med_bkg.get(0,   (None, None))
    (b120, _h120b) = daily_med_bkg.get(120, (None, None))
    (b240, _h240b) = daily_med_bkg.get(240, (None, None))

    im0,   h0   = _secchi_prep_with_daily_bkg(raw0,   b0,   discri_pobj_on=discri_pobj_on,
                                              sebip_off=sebip_off, silent=silent)
    im120, h120 = _secchi_prep_with_daily_bkg(raw120, b120, discri_pobj_on=discri_pobj_on,
                                              sebip_off=sebip_off, silent=silent)
    im240, h240 = _secchi_prep_with_daily_bkg(raw240, b240, discri_pobj_on=discri_pobj_on,
                                              sebip_off=sebip_off, silent=silent)

    ims = np.dstack([im0, im120, im240])   # (ny,nx,3)
    tB, pB, mu = COR1_QUICKPOL(ims, header=[h0, h120, h240],
                               double=True, tangential=False, radial=False)
    pB = pB[..., 0] if pB.ndim == 3 else pB
    return pB, h0

# --------------------------------------------
# Masking and simple COR1 radial-bin utilities
# --------------------------------------------
def _mask_cor1_pb(pB: np.ndarray, r_map: np.ndarray,
                  r_keep_min: float = 1.4, r_keep_max: float = 4.0) -> np.ndarray:
    """
    Keep pB only in [r_keep_min, r_keep_max] R_sun; mask elsewhere.
    """
    out = np.array(pB, dtype=float, copy=True)
    m = (r_map < float(r_keep_min)) | (r_map > float(r_keep_max))
    out[m] = np.nan
    return out


def _build_uniform_r_edges(r_min: float, r_max: float, px_per_rsun: float,
                           px_per_shell: float = 2.0,
                           clamp: Tuple[float,float]=(0.01, 0.20)) -> np.ndarray:
    """
    Build uniform radial bin edges with ~px_per_shell pixels per shell.
    """
    dr = max(clamp[0], min(clamp[1], px_per_shell / float(px_per_rsun)))
    edges = np.arange(r_min, r_max + 1e-9, dr, dtype=float)
    if edges[-1] < r_max - 1e-9:
        edges = np.append(edges, r_max)
    return edges

def compute_calfac_cor1_like_idl(hdr: dict,
                                 *,
                                 apply_ipsum_correction: bool = False,
                                 is_totalb: bool = False) -> float:
    """
    IDL get_calfac.pro の COR1 ロジック（Thompson 2018 の劣化率）を Python で再現。
    既定では apply_ipsum_correction=False（＝画像側でIPSUM補正済みを想定）。
    返り値単位は Bsun/DN（EXPTIME正規化後は Bsun / (DN/s) 相当）。
    """
    det = str(hdr.get('DETECTOR', '')).upper()
    if det != 'COR1':
        return 1.0

    obs = str(hdr.get('OBSRVTRY', '')).upper()
    if 'STEREO_A' in obs or obs.endswith('_A') or obs == 'A':
        base = 6.578e-11  # Bsun/DN
        tai0 = _parse_iso_as_datetime('2007-12-01T03:41:48.174')
        rate = 0.00648    # frac decline / year
    elif 'STEREO_B' in obs or obs.endswith('_B') or obs == 'B':
        base = 7.080e-11
        tai0 = _parse_iso_as_datetime('2008-01-17T02:20:15.717')
        rate = 0.00258
    else:
        base = 6.578e-11
        tai0 = _parse_iso_as_datetime('2007-12-01T03:41:48.174')
        rate = 0.00648

    dt = _parse_iso_as_datetime(hdr.get('DATE-AVG')) or _parse_iso_as_datetime(hdr.get('DATE-OBS'))
    if (dt is None) or (tai0 is None):
        years = 0.0
    else:
        years = (dt - tai0).total_seconds() / (3600.0 * 24.0 * 365.25)

    calfac = base / (1.0 - rate * years)

    # IDLは calfac!=1.0 かつ IPSUM>1 のとき、「CALFAC 側」を area factor で割る
    if apply_ipsum_correction:
        try:
            ipsum = int(hdr.get('IPSUM', 1))
        except Exception:
            ipsum = 1
        if (ipsum > 1) and (abs(calfac - 1.0) > 0.0):
            divfactor = (2 ** (ipsum - 1)) ** 2
            if divfactor != 0:
                calfac = calfac / float(divfactor)

    # tBの2倍規則（POLAR==1001 かつ DOUBLE でない）の再現（pB用途では通常不要）
    if is_totalb:
        try:
            polar = int(hdr.get('POLAR', 0))
        except Exception:
            polar = 0
        seb_prog = str(hdr.get('SEB_PROG', '')).upper()
        if (polar == 1001) and (seb_prog != 'DOUBLE'):
            calfac = 2.0 * calfac

    return float(calfac)


def select_triplet_by_polar_header(raw_files: Sequence[str]) -> Dict[int, str]:
    """
    与えられたRAW .fts群から、ヘッダの POLAR 値で {0: path0, 120:..., 240:...} を選ぶ。
    同角度が複数ある場合は最もDATE-OBSが近い組を別途選ぶことを推奨（簡易版）。
    """
    from astropy.io import fits
    cand: Dict[int, Tuple[str, str]] = {}  # ang -> (path, dateobs)
    for p in raw_files:
        if not os.path.exists(p):
            continue
        with fits.open(p, memmap=False) as hdul:
            hdr = hdul[0].header
            polar = float(hdr.get('POLAR', np.nan))
            dateobs = str(hdr.get('DATE-OBS', ''))
        # 角度は0/120/240にラウンド
        if not np.isfinite(polar):
            continue
        ang = int(round(((polar % 360) + 360) % 360))
        if ang in (0, 120, 240):
            # 最初に見つけたものを採用（必要なら日時で最適化）
            if ang not in cand:
                cand[ang] = (p, dateobs)
    found = {k: v[0] for k, v in cand.items()}
    missing = [ang for ang in (0, 120, 240) if ang not in found]
    if missing:
        raise RuntimeError(f"POLAR {missing}° のRAWが見つかりません。入力を確認してください。")
    return found
def smooth_profiles_along_r(profiles: dict,
                            window_r: float = 0.03,   # 平滑スケール ~0.03 Rsun（調整可）
                            min_bins: int = 5,
                            logspace: bool = True) -> dict:
    """
    Ne_profiles(θ×r) を半径方向にだけ滑らか化する。
    - window_r: Rsun単位の平滑幅（典型 0.02–0.05）
    - r_mid, Ne_profiles を in-place で更新して返す
    """
    import numpy as np
    Ne = profiles['Ne_profiles'].copy()
    r_mid = np.asarray(profiles['r_mid'], float)
    if r_mid.ndim != 1 or Ne.ndim != 2 or Ne.shape[1] != r_mid.size:
        return profiles

    # ウィンドウ長（奇数）
    dr = np.median(np.diff(r_mid))
    k = int(max(min_bins, round(window_r / max(dr, 1e-6))))
    k = k + 1 - (k % 2)  # odd
    if k < min_bins: k = min_bins
    # ハニング窓
    w = np.hanning(k); w /= w.sum()

    def _conv_nanaware(arr1d):
        m = np.isfinite(arr1d) & (arr1d > 0)
        if m.sum() < 3:
            return arr1d
        arr = np.log(arr1d, where=m) if logspace else arr1d.copy()
        arr[~m] = 0.0
        ww = np.where(m, 1.0, 0.0)
        num = np.convolve(ww, w, mode='same')
        s   = np.convolve(arr, w, mode='same')
        out = np.empty_like(arr1d); out[:] = np.nan
        ok = num > 1e-6
        if logspace:
            out[ok] = np.exp(s[ok] / num[ok])
        else:
            out[ok] = s[ok] / num[ok]
        # 元の有限値は優先的に上書き（過度な平滑を避ける）
        out[m] = out[m]
        return out

    for i in range(Ne.shape[0]):
        Ne[i, :] = _conv_nanaware(Ne[i, :])

    profiles['Ne_profiles'] = Ne
    return profiles

def build_density_map_from_profiles_radinterp(
    r_map_rsun: np.ndarray,
    params_ref: dict,
    profiles: dict,
    valid_rmin: float,
    valid_rmax: float
) -> np.ndarray:
    """
    Ne_profiles(θ×r) を 2D に再配置：半径方向＋角度方向の双一次補間（log空間）。
    “段差塗り”によるリングを抑える。
    """
    import numpy as np
    ny, nx = r_map_rsun.shape
    cx = float(params_ref['cx']); cy = float(params_ref['cy'])

    # 角度マップ [0,360)
    yy, xx = np.indices((ny, nx))
    theta = (np.degrees(np.arctan2(yy - cy, xx - cx)) % 360.0).ravel()
    r = r_map_rsun.ravel()

    th_cent = np.asarray(profiles['theta_centers_deg'], float)
    r_mid   = np.asarray(profiles['r_mid'], float)
    Ne_tr   = np.asarray(profiles['Ne_profiles'], float)  # shape: (n_th, n_r)

    n_th, n_r = Ne_tr.shape
    step = float(np.median(np.diff(th_cent))) if n_th > 1 else 360.0
    base = float(th_cent[0]) if n_th > 0 else 0.0

    # 角度方向インデックスと重み
    idx_f = ((theta - base) % 360.0) / step
    i0 = np.floor(idx_f).astype(int) % n_th
    i1 = (i0 + 1) % n_th
    wt_th = idx_f - np.floor(idx_f)

    # 半径方向インデックスと重み
    r_cl = np.clip(r, r_mid[0], r_mid[-1])
    j1 = np.searchsorted(r_mid, r_cl, side='right')
    j1 = np.clip(j1, 1, n_r - 1)
    j0 = j1 - 1
    r0 = r_mid[j0]; r1 = r_mid[j1]
    wt_r = np.clip((r_cl - r0) / np.maximum(r1 - r0, 1e-12), 0.0, 1.0)

    # log 空間で補間
    eps = 1e-30
    lnN_i0_j0 = np.log(np.clip(Ne_tr[i0, j0], eps, None))
    lnN_i0_j1 = np.log(np.clip(Ne_tr[i0, j1], eps, None))
    lnN_i1_j0 = np.log(np.clip(Ne_tr[i1, j0], eps, None))
    lnN_i1_j1 = np.log(np.clip(Ne_tr[i1, j1], eps, None))

    lnN_i0 = (1 - wt_r) * lnN_i0_j0 + wt_r * lnN_i0_j1
    lnN_i1 = (1 - wt_r) * lnN_i1_j0 + wt_r * lnN_i1_j1
    lnN    = (1 - wt_th) * lnN_i0     + wt_th * lnN_i1

    Ne = np.exp(lnN)

    # r 範囲外は NaN
    valid = (r_map_rsun.ravel() >= valid_rmin) & (r_map_rsun.ravel() <= valid_rmax)
    out = np.full(r.size, np.nan, float)
    out[valid] = Ne[valid]
    return out.reshape(ny, nx)


def sanitize_pb_for_inversion(pB_masked: np.ndarray,
                              *,
                              floor_bsun: float = 0.0,
                              n_iter_fill: int = 2,
                              min_neighbors: int = 4) -> np.ndarray:
    """
    反転を安定化するための前処理：
      - rマスク済みの pB について、負値や極小値（<floor_bsun）を近傍平均で補間
      - 3x3 近傍の有限値の平均でNaN/負値を埋める（min_neighbors以上ある画素のみ）
      - これを n_iter_fill 回繰り返して小さな穴を消す
    """
    import numpy as np
    pb = np.array(pB_masked, dtype=float, copy=True)

    # まず floor 未満を NaN 扱いにする（物理的に負の pB は無い）
    bad = ~np.isfinite(pb) | (pb < float(floor_bsun))
    pb[bad] = np.nan

    # 近傍平均での小穴埋め（3x3）
    for _ in range(int(max(0, n_iter_fill))):
        # 8 近傍のシフト配列
        neighs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighs.append(np.roll(np.roll(pb, dy, axis=0), dx, axis=1))
        neighs = np.stack(neighs, axis=0)  # (8, ny, nx)
        finite = np.isfinite(neighs)
        num = finite.sum(axis=0)
        ssum = np.nansum(neighs, axis=0)
        fill = ssum / np.clip(num, 1, None)
        to_fill = (~np.isfinite(pb)) & (num >= int(min_neighbors))
        if not np.any(to_fill):
            break
        pb[to_fill] = fill[to_fill]

    return pb


def guess_daily_med_paths_yyyymmdd(bkg_dir: str, example_raw_path: str,
                                   prefix: str = "dc1A",
                                   angles: Sequence[int] = (0, 120, 240)) -> Dict[int, str]:
    """
    RAWヘッダのDATE-OBSから YYMMDD を抽出し、dc1A_p{000|120|240}_YYMMDD.fts 形式を組む。
    実ファイルが無い角度はキーから除外（後段でp000フォールバック）。
    """
    from astropy.io import fits
    with fits.open(example_raw_path, memmap=False) as hdul:
        dateobs = str(hdul[0].header.get('DATE-OBS', '')).replace('-', '').replace(':', '')
    # '2022-06-13T03:01:00.008' → '220613'
    yymmdd = dateobs[2:4] + dateobs[5:7] + dateobs[8:10] if len(dateobs) >= 10 else ""
    out = {}
    for ang in angles:
        tag = "000" if ang == 0 else ("120" if ang == 120 else "240")
        cand = os.path.join(bkg_dir, f"{prefix}_p{tag}_{yymmdd}.fts")
        if os.path.exists(cand):
            out[ang] = cand
    return out  # 欠損は後段でp000流用

# ----------------------------------------------------
# Top-level: pB -> N_e using the shared inversion code
# ----------------------------------------------------
def invert_cor1_pb_to_density_spherical(pB_masked: np.ndarray,
                              r_map_rsun: np.ndarray,
                              params: dict,
                              r_min: float = 1.4,
                              r_max: float = 4.0,
                              radial_step_pix: float = 1.2,   # ほんの少し細かく
                              n_theta: int = 360,
                              regularization: float = 1e-5,
                              theta_neighbor_blend: int = 2
                              ) -> Tuple[np.ndarray, dict]:
    """
    Mk4+LASCO と同様に、“ビン単位の段差塗り（nearest bin）”で 2D Ne を作る。
    双一次補間は使わない。
    """
    import numpy as np

    # --- r ビン境界 ---
    r_edges = _build_uniform_r_edges(r_min, r_max, params['px_per_rsun'],
                                     px_per_shell=radial_step_pix,
                                     clamp=(0.008, 0.20))
    theta_step_deg = 360.0 / float(max(1, n_theta))

    # --- θごとの pB(r) → Ne(r) 反転プロファイル ---
    profiles = invert_per_theta_profiles(
        pb_image=pB_masked,
        r_map_rsun=r_map_rsun,
        params_ref=params,
        r_min=r_min,
        r_max=r_max,
        dr=(r_max - r_min) / max(len(r_edges) - 1, 1),
        theta_step_deg=theta_step_deg,
        r_edges=r_edges,
        # 段差塗りに合わせ、角度ブレンドは抑える（0〜1推奨）
        theta_neighbor_blend=0
    )

    # （任意）半径方向の穏やかな log 平滑は維持（ビン段差は保たれます）
    profiles = smooth_profiles_along_r(profiles, window_r=0.03, min_bins=5, logspace=True)

    # --- 段差塗りで 2D 展開（nearest bin in θ と r） ---
    ny, nx = r_map_rsun.shape
    cx = float(params['cx']); cy = float(params['cy'])
    yy, xx = np.indices((ny, nx))
    theta_pix = (np.degrees(np.arctan2(yy - cy, xx - cx)) % 360.0).ravel()
    r_pix = r_map_rsun.ravel()

    th_cent = np.asarray(profiles['theta_centers_deg'], float)  # shape: (n_th,)
    r_mid   = np.asarray(profiles['r_mid'], float)              # shape: (n_r,)
    Ne_tr   = np.asarray(profiles['Ne_profiles'], float)        # shape: (n_th, n_r)

    n_th, n_r = Ne_tr.shape
    if n_th < 1 or n_r < 1:
        raise RuntimeError("Empty profiles in inversion result.")

    # 角度：最近中心（round to nearest center）
    step = float(360.0 / n_th) if n_th > 1 else 360.0
    base = float(th_cent[0]) if n_th > 0 else 0.0
    idx_f = ((theta_pix - base) % 360.0) / step
    i_th = (np.floor(idx_f + 0.5).astype(int)) % n_th

    # 半径：ビン段差（digitize で [edge_i, edge_{i+1}) に割当て）
    j_r = np.digitize(r_pix, r_edges) - 1
    valid_r = (j_r >= 0) & (j_r < n_r)

    # 2D マップを段差塗りで構成
    flat = np.full(r_pix.size, np.nan, float)
    ok = valid_r
    flat[ok] = Ne_tr[i_th[ok], j_r[ok]]

    # r 範囲外は NaN
    valid_shell = (r_pix >= r_min) & (r_pix <= r_max)
    out = np.full(r_pix.size, np.nan, float)
    out[valid_shell] = flat[valid_shell]
    ne_map = out.reshape(ny, nx)

    aux = dict(theta_centers_deg=profiles['theta_centers_deg'],
               r_edges=profiles['r_edges'],
               r_mid=profiles['r_mid'],
               Ne_profiles=profiles['Ne_profiles'],
               pB_profiles=profiles.get('pB_profiles'))
    return ne_map, aux


def invert_cor1_pb_to_density_axisymmetric(
    pB_masked: np.ndarray,
    r_map: np.ndarray,
    params: dict,
    r_min: float = 1.4,
    r_max: float = 4.0,
    radial_step_pix: int = 2,
    n_theta: int = 360,
    regularization: float = 1e-5,
):
    """
    Axisymmetric version (Hayes et al., 2001) of the pB->Ne inversion.

    Changes from the previous sector-wise (per-θ) implementation:
      - Assume axisymmetry about the solar rotation axis and invert a single
        azimuthally-averaged pB(r) annular profile.
      - Use the same van de Hulst / Billings kernel as the original code.
      - Map the resulting Ne(r) back to the 2D image by assigning each pixel
        the Ne corresponding to its radial bin.

    Returns
    -------
    ne_map : 2D array (cm^-3)
    extra  : (r_edges, theta_edges=None, r_bin_centers, ne_profile_1d)
    """
    import numpy as _np
    from constants_vdh import invert_ablation  # local import to avoid touching global imports

    px_per_rsun = float(params["px_per_rsun"])
    dr = float(radial_step_pix) / px_per_rsun

    # --- radial bins in R_sun ---
    r_edges = _np.arange(float(r_min), float(r_max) + dr, dr, dtype=float)
    if r_edges.size < 2:
        raise ValueError("r_edges has fewer than 2 points")
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    n_r = r_mid.size

    # --- azimuthally-averaged annular pB(r) ---
    pb_prof = _np.full(n_r, _np.nan, dtype=float)
    for i in range(n_r):
        rin, rout = r_edges[i], r_edges[i+1]
        mask = (r_map >= rin) & (r_map < rout) & _np.isfinite(pB_masked) & (pB_masked > 0)
        if _np.any(mask):
            pb_prof[i] = _np.nanmedian(pB_masked[mask])  # robust against outliers

    # --- repair small radial gaps (keep edge NaNs) ---
    def _interp_small_gaps_1d(row: _np.ndarray, max_gap_bins: int = 3) -> _np.ndarray:
        x = _np.arange(row.size, dtype=float)
        out = row.copy()
        i = 0
        while i < out.size:
            if _np.isnan(out[i]):
                j = i
                while j < out.size and _np.isnan(out[j]):
                    j += 1
                L = j - i
                if L <= max_gap_bins and i > 0 and j < out.size and _np.isfinite(out[i-1]) and _np.isfinite(out[j]):
                    out[i:j] = _np.interp(_np.arange(i, j, dtype=float), [i-1, j], [out[i-1], out[j]])
                i = j
            else:
                i += 1
        return out

    pb_prof = _interp_small_gaps_1d(pb_prof, max_gap_bins=3)

    valid = _np.isfinite(pb_prof) & (pb_prof > 0)
    if valid.sum() < 3:
        raise RuntimeError("Too few valid annular samples in pB to invert. Check masks/background.")

    # --- van de Hulst 殻ごと反転（1D） ---
    ne_1d = invert_ablation(pb_prof[valid], r_mid[valid], r_edges[_np.r_[valid, False]], valid.sum())

    # 全長に復元（無効部は NaN のまま）
    ne_full = _np.full_like(pb_prof, _np.nan, dtype=float)
    ne_full[valid] = ne_1d

    # --- 2D 展開（半径最近傍） ---
    ne_map = _np.full_like(pB_masked, _np.nan, dtype=float)
    idx = _np.digitize(r_map.ravel(), r_edges) - 1
    ok = (idx >= 0) & (idx < n_r)
    flat = _np.full(idx.size, _np.nan, dtype=float)
    flat[ok] = ne_full[idx[ok]]
    ne_map = flat.reshape(r_map.shape)
    ne_map[(r_map < r_min) | (r_map >= r_max)] = _np.nan

    # --- 小島状 NaN の穏やかな補間（内側だけ、境界は保持） ---
    def _fill_small_nan_islands(img: _np.ndarray, passes: int = 2, min_neighbors: int = 5) -> _np.ndarray:
        out = img.copy()
        for _ in range(int(passes)):
            nanmask = ~_np.isfinite(out)
            if not _np.any(nanmask):
                break
            neighs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighs.append(_np.roll(_np.roll(out, dy, axis=0), dx, axis=1))
            neighs = _np.stack(neighs, axis=0)
            finite = _np.isfinite(neighs)
            num = finite.sum(axis=0)
            ssum = _np.nansum(neighs, axis=0)
            fill = ssum / _np.clip(num, 1, None)
            to_fill = nanmask & (num >= int(min_neighbors))
            out[to_fill] = fill[to_fill]
        return out

    ne_map = _fill_small_nan_islands(ne_map, passes=2, min_neighbors=5)

    theta_edges = None  # 軸対称なので θ 分割はありません
    return ne_map, (r_edges, theta_edges, r_mid, ne_full)

def inpaint_small_nans(ne_map: np.ndarray,
                       valid_mask: Optional[np.ndarray] = None,
                       max_passes: int = 3,
                       min_neighbors: int = 3) -> np.ndarray:
    """
    2D 密度マップの小さな NaN 島を、8近傍の有限値の平均で段階的に埋める。
    r 範囲外などの大域マスク(valid_mask=False)は埋めない。
    """
    import numpy as np
    ne = np.array(ne_map, dtype=float, copy=True)
    if valid_mask is None:
        valid_mask = np.isfinite(ne)

    for _ in range(int(max_passes)):
        nan_island = (~np.isfinite(ne)) & valid_mask
        if not np.any(nan_island):
            break
        neighs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighs.append(np.roll(np.roll(ne, dy, axis=0), dx, axis=1))
        neighs = np.stack(neighs, axis=0)
        finite = np.isfinite(neighs)
        num = finite.sum(axis=0)
        ssum = np.nansum(neighs, axis=0)
        fill = ssum / np.clip(num, 1, None)
        to_fill = nan_island & (num >= int(min_neighbors))
        if not np.any(to_fill):
            break
        ne[to_fill] = fill[to_fill]
    return ne


def export_density_csv_sta(
    density_map: np.ndarray,
    r_map_rsun: np.ndarray,
    params_ref: dict,
    csv_path: str,
    include_all: bool = False
) -> None:
    import numpy as np
    ny, nx = density_map.shape
    cx = float(params_ref['cx']); cy = float(params_ref['cy'])
    px_per_rsun = float(params_ref.get('px_per_rsun', np.nan))
    rsun_arcsec = float(params_ref.get('rsun_arcsec', np.nan))
    arcsec_per_pix = params_ref.get('arcsec_per_pix', np.nan)

    if np.isfinite(arcsec_per_pix):
        s_arc = float(arcsec_per_pix)
    elif np.isfinite(px_per_rsun) and np.isfinite(rsun_arcsec) and px_per_rsun > 0:
        s_arc = rsun_arcsec / px_per_rsun
    else:
        s_arc = np.nan

    yy, xx = np.indices((ny, nx))
    x_Rsun = (xx - cx) / px_per_rsun if np.isfinite(px_per_rsun) else np.nan*(xx-cx)
    y_Rsun = (yy - cy) / px_per_rsun if np.isfinite(px_per_rsun) else np.nan*(yy-cy)
    th_deg  = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360.0

    if include_all:
        mask = np.ones_like(density_map, dtype=bool)
    else:
        mask = np.isfinite(density_map) & (density_map > 0)

    data = np.column_stack([
        yy[mask].ravel().astype(int),
        xx[mask].ravel().astype(int),
        ((xx - cx) * s_arc)[mask].ravel() if np.isfinite(s_arc) else (xx - cx)[mask].ravel(),
        ((yy - cy) * s_arc)[mask].ravel() if np.isfinite(s_arc) else (yy - cy)[mask].ravel(),
        x_Rsun[mask].ravel(), y_Rsun[mask].ravel(),
        r_map_rsun[mask].ravel(), th_deg[mask].ravel(),
        density_map[mask].ravel()
    ])
    header = "y_pix,x_pix,x_arcsec,y_arcsec,x_Rsun,y_Rsun,r_Rsun,theta_deg,Ne_cm^-3"
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

def save_density_fits_sta(density_map: np.ndarray, params_ref: dict, fits_path: str) -> None:
    from astropy.io import fits
    import numpy as np

    # Let astropy build the required FITS keywords (SIMPLE, BITPIX, NAXIS, …)
    hdu = fits.PrimaryHDU(density_map.astype('float32'))
    h = hdu.header
    h['BUNIT']   = 'cm^-3'
    h['COMMENT'] = 'Electron density from pB inversion (COR1)'
    h['CRPIX1']  = float(params_ref.get('cx', 0.0)) + 1.0  # FITS 1-origin
    h['CRPIX2']  = float(params_ref.get('cy', 0.0)) + 1.0
    # WCS-like keywords (arcsec/pix if available)
    arcsec_per_pix = params_ref.get('arcsec_per_pix', np.nan)
    px_per_rsun = float(params_ref.get('px_per_rsun', np.nan))
    rsun_arcsec = float(params_ref.get('rsun_arcsec', np.nan))
    if np.isfinite(arcsec_per_pix):
        cdelt = float(arcsec_per_pix)
    elif np.isfinite(px_per_rsun) and np.isfinite(rsun_arcsec) and px_per_rsun > 0:
        cdelt = rsun_arcsec / px_per_rsun
    else:
        cdelt = 1.0
    h['CDELT1']  = cdelt
    h['CDELT2']  = cdelt
    h['CUNIT1']  = 'arcsec'
    h['CUNIT2']  = 'arcsec'
    h['CTYPE1']  = 'HPLN-TAN'
    h['CTYPE2']  = 'HPLT-TAN'
    hdu.writeto(fits_path, overwrite=True)


# --------------------
# Minimal plot helper
# --------------------
def plot_density_map_sta(ne_map: np.ndarray,
                         params: dict,
                         r_keep_min: float = 1.4,
                         r_keep_max: float = 4.0,
                         title: str = "STEREO-A COR1 2D density (pB inversion) 2022-06-13 03:01:00 UT",
                         savepath: Optional[str] = None):
    """
    ne_map を対数カラースケールで描画。
    軸は可能なら arcsec（太陽中心からのオフセット）に統一する。
    savepath を省略した場合は既定の出力先に PNG を保存する。
    """
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ny, nx = ne_map.shape
    cx = float(params['cx'])
    cy = float(params['cy'])
    pps = float(params['px_per_rsun'])  # px per Rsun

    # --- arcsec/px の推定 ---
    s_arc = None
    if 'arcsec_per_pix' in params and np.isfinite(params['arcsec_per_pix']):
        s_arc = float(params['arcsec_per_pix'])
    elif 'rsun_arcsec' in params and np.isfinite(params['rsun_arcsec']):
        s_arc = float(params['rsun_arcsec']) / pps
    elif 'rsun_arc' in params and np.isfinite(params['rsun_arc']):
        s_arc = float(params['rsun_arc']) / pps

    # --- r_map を計算（単位: Rsun） ---
    r_map = _radius_map(ne_map.shape, cx, cy, pps)

    # --- extent & 座標グリッド ---
    if (s_arc is not None) and np.isfinite(s_arc):
        extent = [(-cx) * s_arc, (nx - cx) * s_arc, (-cy) * s_arc, (ny - cy) * s_arc]
        xlabel = 'X [arcsec from Sun center]'
        ylabel = 'Y [arcsec from Sun center]'
        yy, xx = np.indices((ny, nx))
        X = (xx - cx) * s_arc
        Y = (yy - cy) * s_arc
    else:
        extent = [-cx, nx - cx, -cy, ny - cy]
        xlabel = 'X [pixel from Sun center]'
        ylabel = 'Y [pixel from Sun center]'
        yy, xx = np.indices((ny, nx))
        X = xx - cx
        Y = yy - cy

    # --- ステップ表示（補間 OFF） ---
    vmin, vmax = 1e5, 1e9  # cm^-3
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(ne_map, origin='lower', extent=extent,
                   norm=LogNorm(vmin=vmin, vmax=vmax),
                   cmap='plasma',
                   interpolation='nearest')   # ★ ここを追加（段差表示）

    # --- 整数 Rsun の等高線 ---
    int_levels = np.arange(1, int(math.floor(r_keep_max)) + 1)
    if len(int_levels) > 0:
        ax.contour(X, Y, r_map, levels=int_levels,
                   colors='white', linewidths=1.0, linestyles='--', alpha=0.7)

    # --- 解析範囲リング ---
    for R, col, lab in [(r_keep_min, 'cyan',   f"{r_keep_min:.1f} R$_\\odot$"),
                        (r_keep_max, 'magenta',f"{r_keep_max:.1f} R$_\\odot$")]:
        if R <= np.nanmax(r_map):
            ax.contour(X, Y, r_map, levels=[R],
                       colors=[col], linewidths=1.2, linestyles='-.')
            ax.plot([], [], color=col, linestyle='-.', label=lab)

    ax.plot(0, 0, '+', color='k', markersize=10, markeredgewidth=1.5)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_aspect('equal')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.01)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=14)
    cb.set_label('N$_e$ [cm$^{-3}$]', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    if not savepath or not isinstance(savepath, str):
        savepath = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/2D_density_map_sta.png"
    fig.savefig(savepath, dpi=300)
    print(f"[save] Wrote {savepath}")
    plt.show()
    return fig, ax


# --------------------
# Script entry point
# --------------------
if __name__ == "__main__":
    RAW_DIR = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata"
    BKG_DIR = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata"

    # === ここで反転モードを選ぶ ===
    # "spherical"（= 局所球対称, 扇形ごと反転） or "axisym"（= 軸対称, 全周平均1D→2D）
    MODE = "spherical"   # ← "axisym" に変えるだけで軸対称版に切替

    raw_files = [
        os.path.join(RAW_DIR, "20220613_030100_n4c1A.fts"),  # POLAR=0°
        os.path.join(RAW_DIR, "20220613_030118_n4c1A.fts"),  # POLAR=120°
        os.path.join(RAW_DIR, "20220613_030136_n4c1A.fts"),  # POLAR=240°
    ]
    triplet = select_triplet_by_polar_header(raw_files)

    # 背景（欠損角はp000で代替）
    p000 = os.path.join(BKG_DIR, "dc1A_p000_220613.fts")
    p120 = os.path.join(BKG_DIR, "dc1A_p120_220613.fts")
    p240 = os.path.join(BKG_DIR, "dc1A_p240_220613.fts")
    bkg_map = load_daily_med_backgrounds(p000=p000, p120=p120, p240=p240)

    # pB作成（IPSUM補正→IDL同等CALFAC→QUICKPOL）
    pB, h0 = build_pB_from_triplet(triplet[0], triplet[120], triplet[240], bkg_map,
                                   nocalfac_butcorrforipsum=True,
                                   discri_pobj_on=False,
                                   sebip_off=False,
                                   silent=True)

    # 幾何／マスク
    params = _params_from_header(h0)
    r_map = _radius_map((params['ny'], params['nx']), params['cx'], params['cy'], params['px_per_rsun'])
    pB_masked = _mask_cor1_pb(pB, r_map, r_keep_min=1.4, r_keep_max=4.0)

    # 反転安定化のための前処理（小穴を埋める／負値をNaN扱い）
    pB_pre = sanitize_pb_for_inversion(pB_masked, floor_bsun=0.0, n_iter_fill=2)

    # === 反転の選択 ===
    if MODE.lower().startswith("axis"):
        ne_map, aux = invert_cor1_pb_to_density_axisymmetric(
            pB_masked=pB_pre,     # ★ sanitize後を渡す
            r_map=r_map,
            params=params,
            r_min=1.4, r_max=4.0,
            radial_step_pix=1.5
        )
        title = "STEREO-A COR1 2D density (Axisymmetric inversion)\n2022-06-13 03:01:00 UT"
    else:
        ne_map, aux = invert_cor1_pb_to_density_spherical(
            pB_pre, r_map, params,
            r_min=1.4, r_max=4.0,
            radial_step_pix=0.5, n_theta=720,
            theta_neighbor_blend=1
        )
        title = "STEREO-A COR1 2D density (Spherical symmetric inversion)\n2022-06-13 03:01:00 UT"

    # 小さな NaN 島の穏やかな補間（r範囲内のみ）
    valid = (r_map >= 1.4) & (r_map <= 4.0)
    ne_map = inpaint_small_nans(ne_map, valid_mask=valid, max_passes=3, min_neighbors=3)

    # 出力
    output_csv_path  = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/2D_density_map_sta.csv"
    output_fits_path = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/2D_density_map_sta_result.fits"
    export_density_csv_sta(ne_map, r_map, params, output_csv_path)
    # save_density_fits_sta(ne_map, params, output_fits_path)

    plot_density_map_sta(
        ne_map, params,
        title=title,
        savepath="/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/2D_density_map_sta.png"
    )
