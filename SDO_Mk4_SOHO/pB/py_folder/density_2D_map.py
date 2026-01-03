# -*- coding: utf-8 -*-
"""
2D_density_map.py

Convert a 2D pB map into a 2D electron density map using the van de Hulst
inversion (shell ablation) performed independently along multiple
position-angle sectors.

Design requirements from the user:
- Reuse functions from the current code base (import from your modules).
- Keep the inversion (physics) separated from plotting.
- Provide small helper functions to retrieve density at given coordinates.

Assumptions & notes:
- Axisymmetry is applied *per position angle sector*: for each PA center θ,
  we construct a 1D pB(r) profile (sector average; ±5° by the current
  extract_pB_profile implementation) and invert to Ne(r). This yields
  Ne(r, θ_center). The 2D map assigns to each pixel the Ne value from the
  nearest θ_center and corresponding radial bin r.
- pB values are assumed to be in B_sun-normalized units, consistent with
  the current "constants_vdh.invert_ablation" implementation.
- The angular tolerance inside extract_pB_profile is currently fixed at ±5°
  in your repository code, so using θ bin step = 10° gives contiguous
  non-overlapping coverage. If you choose a smaller step, angular overlap
  may occur (not harmful, but redundant).

Author: (your name)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# 追加（density_2D_map.py の import 付近）
import os
import argparse
from astropy.io import fits


# --- Reuse from your existing modules ---
from io_and_processing import (
    load_and_prepare_instrument_data,
    combine_corona_data,
    extract_pB_profile
)
from constants_vdh import (
    invert_ablation,
    set_u_from_instrument
)
from plotting_utils import (
    plot_combined_image, 
    save_density_map_to_fits
)

# -----------------------------
# Core utilities (polar helpers)
# -----------------------------
def save_combined_pb_fits_for_tomography(
    pb_image: np.ndarray,
    lasco_fits_path: str,
    kcor_fits_path: str,
    out_fits_path: str,
    *,
    bunit: str = "B/Bsun",
) -> None:
    """
    Save an Earth-view combined pB map (K-COR inner + LASCO C2 outer) as a FITS file
    suitable for regularized tomography.

    - Uses LASCO header as the base (keeps plate scale etc.)
    - Copies RSUN_OBS + observer lon/lat (CRLN_OBS/CRLT_OBS) from K-COR if missing
    - Ensures minimal HPC WCS (CTYPE1/2, CUNIT1/2)
    - Removes BLANK for float images (avoids astropy warning)
    """
    from astropy.io import fits
    import numpy as _np
    import os as _os

    hdr = fits.getheader(lasco_fits_path)
    hdr_k = fits.getheader(kcor_fits_path)

    # BLANK is only valid for integer images
    if "BLANK" in hdr:
        try:
            del hdr["BLANK"]
        except Exception:
            pass

    # Minimal HPC WCS
    if "CTYPE1" not in hdr:
        hdr["CTYPE1"] = hdr_k.get("CTYPE1", "HPLN-TAN")
    if "CTYPE2" not in hdr:
        hdr["CTYPE2"] = hdr_k.get("CTYPE2", "HPLT-TAN")
    if "CUNIT1" not in hdr:
        hdr["CUNIT1"] = hdr_k.get("CUNIT1", "arcsec")
    if "CUNIT2" not in hdr:
        hdr["CUNIT2"] = hdr_k.get("CUNIT2", "arcsec")

    # Observer lon/lat (Carrington preferred)
    for key in ("CRLN_OBS", "CRLT_OBS", "HGLN_OBS", "HGLT_OBS", "CAR_ROT", "DSUN_OBS"):
        if key in hdr_k and key not in hdr:
            hdr[key] = hdr_k[key]

    # RSUN in arcsec (required by tomography normalization)
    rsun_arcsec = None
    for key in ("RSUN_OBS", "RSUN"):
        if key in hdr_k:
            rsun_arcsec = float(hdr_k[key])
            break
    if rsun_arcsec is None and ("R_SUN" in hdr_k) and ("CDELT1" in hdr_k):
        try:
            rsun_arcsec = float(hdr_k["R_SUN"]) * abs(float(hdr_k["CDELT1"]))
        except Exception:
            rsun_arcsec = None
    if rsun_arcsec is not None:
        hdr["RSUN_OBS"] = float(rsun_arcsec)
        hdr["RSUN"] = float(rsun_arcsec)

    hdr["POLAR"] = "PB"
    hdr["BUNIT"] = str(bunit)
    hdr.add_history("Combined K-COR (inner) and LASCO-C2 (outer) pB for Earth-view tomography.")
    hdr.add_history("Copied observer keywords (CRLN_OBS/CRLT_OBS, RSUN_OBS) from K-COR header when missing.")

    # Ensure dimensions consistent
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = int(_np.asarray(pb_image).shape[1])
    hdr["NAXIS2"] = int(_np.asarray(pb_image).shape[0])

    _os.makedirs(_os.path.dirname(out_fits_path), exist_ok=True)
    fits.PrimaryHDU(_np.asarray(pb_image, dtype=_np.float32), header=hdr).writeto(out_fits_path, overwrite=True)


def _pixel_size_rsun(params: dict) -> float:
    """
    Return pixel size in units of R_sun for the given instrument params.
    px_per_rsun is pixels per R_sun, so 1 px = 1/px_per_rsun [R_sun].
    """
    return 1.0 / float(params['px_per_rsun'])
def _angdiff_deg(a, b):
    # 差を (-180,180] に折り返す
    d = (a - b + 180.0) % 360.0 - 180.0
    return d

def _theta_linear_weights(theta_map_deg: np.ndarray, centers_deg: np.ndarray):
    """
    等間隔のセクタ中心 centers_deg (0..360) に対し、
    各ピクセル角度の両側インデックス (i0,i1) と重み (w0,w1) を返す。
    w0+w1=1, 円周をまたぐ境界も連続。
    """
    step = float(np.diff(centers_deg[:2])[0]) if len(centers_deg) > 1 else 360.0
    n = centers_deg.size
    # 0..360 に正規化
    th = np.where(theta_map_deg < 0.0, theta_map_deg + 360.0, theta_map_deg)
    th = np.where(th >= 360.0, th - 360.0, th)

    # 左側中心の整数 index
    i0 = np.floor(th / step).astype(int) % n
    i1 = (i0 + 1) % n
    # 左中心角と小数部
    t0 = i0.astype(float) * step
    frac = (th - t0) / step
    w1 = frac
    w0 = 1.0 - frac
    return i0, i1, w0, w1

def _nanblend(a, b, w_b):
    """
    a と b を重み w_b で線形合成。NaNに頑健：
    - 両方有限→ (1-w_b)*a + w_b*b
    - 片方だけ有限→ その値
    - 両方 NaN → NaN
    """
    out = np.where(np.isfinite(a) & np.isfinite(b),
                   (1.0 - w_b) * a + w_b * b,
                   np.where(np.isfinite(a), a, np.where(np.isfinite(b), b, np.nan)))
    return out


def _adaptive_pb_profile_for_theta(
    pb_image: np.ndarray,
    r_bin_idx: np.ndarray,           # digitize_radius(...) の結果
    theta_map_deg: np.ndarray,       # angle_map_deg(...) の結果
    theta_center_deg: float,
    n_r: int,
    base_halfwidth_deg: float = 5.0,
    max_halfwidth_deg: float = 20.0,
    halfwidth_step_deg: float = 2.0,
    min_samples: int = 60
) -> np.ndarray:
    """
    既存抽出で NaN の半径ビンだけ、角半幅を広げつつ pB 平均を取り直す。
    pB>=0 を有効値としてカウント。戻り値は pb_profile (len=n_r)。
    """
    pb_prof = np.full(n_r, np.nan, dtype=float)

    # 角度差（全画素）
    dth = np.abs(_angdiff_deg(theta_map_deg, theta_center_deg))

    # ビンごとにフォールバック
    for i in range(n_r):
        # r=i の画素マスク
        rmask = (r_bin_idx == i)

        # 角半幅を段階的に拡大
        hw = base_halfwidth_deg
        filled = False
        while hw <= max_halfwidth_deg and not filled:
            mask = rmask & (dth <= hw)
            if np.any(mask):
                vals = pb_image[mask]
                # ゼロ以上を有効、負値・NaNは除外
                good = np.isfinite(vals) & (vals >= 0.0)
                if np.count_nonzero(good) >= min_samples:
                    pb_prof[i] = np.nanmean(vals[good])
                    filled = True
            hw += halfwidth_step_deg
        # filled=False のままなら NaN のまま（後段のθ/2D補間が対応）
    return pb_prof


def build_adaptive_radial_edges(
    r_min: float,
    r_transition: float,
    r_max: float,
    params_mk4: dict,
    params_lasco: dict,
    inner_px_factor: float = 3.0,
    outer_px_factor: float = 2.0,
    min_dr: float = 0.01,
    max_dr: float = 0.25
) -> np.ndarray:
    """
    Build non-uniform radial edges so that:
      - r < r_transition : dr ≈ inner_px_factor * (Mk4 pixel size in R_sun)
      - r ≥ r_transition : dr ≈ outer_px_factor * (LASCO pixel size in R_sun)
    dr is clamped to [min_dr, max_dr] to avoid over/under sampling.
    """
    ps_mk4   = _pixel_size_rsun(params_mk4)    # [R_sun / px]
    ps_lasco = _pixel_size_rsun(params_lasco)  # [R_sun / px]

    dr_inner = np.clip(inner_px_factor * ps_mk4,   min_dr, max_dr)
    dr_outer = np.clip(outer_px_factor * ps_lasco, min_dr, max_dr)

    edges = [float(r_min)]
    # inner part
    r = r_min
    while r < r_transition - 1e-9:
        r = min(r + dr_inner, r_transition)
        edges.append(r)

    # ensure exact transition included once
    if abs(edges[-1] - r_transition) > 1e-9:
        edges.append(r_transition)

    # outer part
    r = r_transition
    while r < r_max - 1e-9:
        r = min(r + dr_outer, r_max)
        edges.append(r)

    # uniquify & ensure strictly increasing
    edges = np.array(edges, dtype=float)
    edges = np.unique(np.clip(edges, r_min, r_max))
    edges.sort()
    if edges.size < 2:
        raise ValueError("Adaptive r-edges construction failed (fewer than 2 edges).")
    return edges

def _find_nan_runs(mask_nan: np.ndarray) -> list[tuple[int,int]]:
    """
    Return list of (start, end_exclusive) index pairs for consecutive True runs in mask_nan.
    """
    runs = []
    n = mask_nan.size
    i = 0
    while i < n:
        if mask_nan[i]:
            j = i + 1
            while j < n and mask_nan[j]:
                j += 1
            runs.append((i, j))  # [i, j)
            i = j
        else:
            i += 1
    return runs

def _interp_small_radial_gaps(pb_row: np.ndarray, max_gap_bins: int = 4) -> np.ndarray:
    """
    Fill small NaN gaps (length <= max_gap_bins) in a 1D pB(r) profile by linear interpolation
    between the nearest valid neighbors. Large gaps or edge gaps are kept as NaN.
    """
    x = np.arange(pb_row.size, dtype=float)
    out = pb_row.copy()
    isn = ~np.isfinite(out)
    if not np.any(isn):
        return out

    runs = _find_nan_runs(isn)
    for (i0, i1) in runs:
        L = i1 - i0
        # Require small interior gap with valid neighbors on both sides
        if L <= max_gap_bins and i0 > 0 and i1 < out.size and np.isfinite(out[i0-1]) and np.isfinite(out[i1]):
            # linear interpolation between (i0-1, i1)
            out[i0:i1] = np.interp(x[i0:i1], [x[i0-1], x[i1]], [out[i0-1], out[i1]])
        # else: leave as NaN
    return out

def _blend_theta_neighbors(pb_matrix: np.ndarray, max_neighbors: int = 1) -> np.ndarray:
    """
    For each theta row, fill remaining NaNs by taking the median across ±d neighbors (same radial bin).
    max_neighbors=1 checks ±1; if still NaN can set >1 to check wider.
    """
    if max_neighbors <= 0:
        return pb_matrix

    th, nr = pb_matrix.shape
    out = pb_matrix.copy()

    # Work bin-by-bin to avoid large memory temp
    for i in range(nr):
        col = out[:, i]
        nan_mask = ~np.isfinite(col)
        if not np.any(nan_mask):
            continue

        # For each theta with NaN, build a neighbor list
        idx_nan = np.where(nan_mask)[0]
        for k in idx_nan:
            vals = []
            for d in range(1, max_neighbors+1):
                vals.append(out[(k-d) % th, i])
                vals.append(out[(k+d) % th, i])
            vals = np.array(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                col[k] = np.median(vals)  # robust to outliers
        out[:, i] = col

    return out


def angle_map_deg(ny: int, nx: int, cy: float, cx: float, mode: str = '0to360') -> np.ndarray:
    """
    Build a theta map in degrees for each pixel relative to (cx, cy).
    mode: '0to360' or 'pm180'.
    """
    yy, xx = np.indices((ny, nx))
    th = np.degrees(np.arctan2(yy - cy, xx - cx))  # range: (-180, 180]
    if mode == '0to360':
        th = np.where(th < 0.0, th + 360.0, th)
    return th

def digitize_theta(theta_map_deg: np.ndarray, theta_centers_deg: np.ndarray) -> np.ndarray:
    """
    Map per-pixel theta (deg, 0..360) to nearest theta bin index (0..n_theta-1).
    """
    # Compute distance on circle
    # Expand dims: (H,W,1) vs (1,1,N)
    diff = np.abs((theta_map_deg[..., None] - theta_centers_deg[None, None, :] + 180.0) % 360.0 - 180.0)
    idx = np.argmin(diff, axis=-1)
    return idx

def digitize_radius(r_map_rsun: np.ndarray, r_edges: np.ndarray) -> np.ndarray:
    """
    Map per-pixel r (in R_sun) to radial bin index via np.digitize, clipped to valid range.
    Returns -1 for out-of-range pixels (before masking).
    """
    # np.digitize returns indices in 1..len(edges)-1 for values in [edges[i-1], edges[i]).
    ridx = np.digitize(r_map_rsun, r_edges) - 1
    # mark invalid
    invalid = (ridx < 0) | (ridx >= (len(r_edges) - 1))
    ridx[invalid] = -1
    return ridx

# ---------------------------------
# Inversion & 2D map construction
# ---------------------------------
def invert_pb_to_density_axisymmetric(instrument: str,
                                      pb_image,
                                      r_map_rsun,
                                      params_ref,
                                      r_min,
                                      r_max,
                                      dr=0.02,
                                      theta_step_deg=10.0,
                                      r_edges=None,
                                      theta_neighbor_blend=4,
                                      theta_neighbor_fallback=10,
                                      spatial_fill_mode="nearest",
                                      spatial_fill_iters=2,
                                      use_bilinear=True):
    """
    既存の per-θ 反転＋2D再配置をまとめて呼ぶユーティリティ。
    - 角度方向は θ_step ごとに pB(r) を抽出し invert_ablation で Ne(r)。
    - 2D への展開は build_density_map_from_profiles() を使用。
    """
    set_u_from_instrument(instrument)
    
    profiles = invert_per_theta_profiles(
        pb_image=pb_image,
        r_map_rsun=r_map_rsun,
        params_ref=params_ref,
        r_min=float(r_min),
        r_max=float(r_max),
        dr=float(dr),
        theta_step_deg=float(theta_step_deg),
        theta_mode='0to360',
        theta_neighbor_blend=int(theta_neighbor_blend),
        r_edges=r_edges
    )

    density_map = build_density_map_from_profiles(
        r_map_rsun=r_map_rsun,
        params_ref=params_ref,
        profiles=profiles,
        valid_rmin=float(r_min),
        valid_rmax=float(r_max),
        mask_nan_outside=True,
        theta_neighbor_fallback=int(theta_neighbor_fallback),
        spatial_fill_mode=str(spatial_fill_mode),
        spatial_fill_iters=int(spatial_fill_iters),
        use_bilinear=bool(use_bilinear)   # ← 双一次補間のON/OFF
    )
    return density_map, profiles

def invert_pb_to_density_spherical(instrument: str,
                                   pb_image,
                                   r_map_rsun,
                                   params_ref,
                                   r_min,
                                   r_max,
                                   dr=0.02,
                                   r_edges=None,
                                   spatial_fill_mode="nearest",
                                   spatial_fill_iters=1):
    """
    球対称（θ平均）で pB(r) → Ne(r) を 1D 反転し、半径ビン最近傍で 2D へ展開。
    - 角度方向は平均化（pB>=0 を有効値）。
    - 2D 展開は “ビン段差塗り（nearest）”で、補間は行わない。
    """
    import numpy as np
    
    set_u_from_instrument(instrument)
    # --- 半径ビン（edges, mid）
    if r_edges is None:
        r_edges = np.arange(float(r_min), float(r_max) + float(dr), float(dr))
    r_edges = np.asarray(r_edges, dtype=float)
    if r_edges.ndim != 1 or r_edges.size < 2:
        raise ValueError("r_edges must be 1D with at least 2 elements.")
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    n_r = r_mid.size

    # --- 半径ビンごとに pB 平均（θは全周を一括平均、pB>=0 を採用）
    pB_prof = np.full(n_r, np.nan, float)
    for i in range(n_r):
        m = (r_map_rsun >= r_edges[i]) & (r_map_rsun < r_edges[i+1])
        if np.any(m):
            vals = pb_image[m]
            good = np.isfinite(vals) & (vals >= 0.0)
            if np.any(good):
                pB_prof[i] = np.nanmean(vals[good])

    # 小欠損の線形補間（端は残す）
    def _interp_small_gaps_1d(arr, max_gap_bins=4):
        x = np.arange(arr.size, dtype=float)
        out = arr.copy()
        isn = ~np.isfinite(out)
        if not np.any(isn):
            return out
        # run 検出
        runs = []
        n = out.size; j = 0
        while j < n:
            if isn[j]:
                k = j + 1
                while k < n and isn[k]:
                    k += 1
                runs.append((j, k))  # [j, k)
                j = k
            else:
                j += 1
        for j, k in runs:
            L = k - j
            if L <= max_gap_bins and j > 0 and k < n and np.isfinite(out[j-1]) and np.isfinite(out[k]):
                out[j:k] = np.interp(x[j:k], [x[j-1], x[k]], [out[j-1], out[k]])
        return out

    pB_prof = _interp_small_gaps_1d(pB_prof, max_gap_bins=4)

    # --- 1D 反転（van de Hulst ablation）
    from constants_vdh import invert_ablation
    Ne_prof = np.full_like(pB_prof, np.nan)
    finite = np.where(np.isfinite(pB_prof) & (pB_prof > 0))[0]
    if finite.size >= 2:
        last = int(finite[-1])
        try:
            Ne_sub = invert_ablation(pB_prof[:last+1], r_mid[:last+1], r_edges[:last+2], last+1)
            Ne_prof[:last+1] = Ne_sub
        except Exception as e:
            print(f"[WARN] spherical inversion failed: {e}")

    # --- 2D 段差塗り（最近傍ビンに割当て）
    ny, nx = r_map_rsun.shape
    ridx = np.digitize(r_map_rsun, r_edges) - 1
    valid = (ridx >= 0) & (ridx < n_r)
    density_map = np.full((ny, nx), np.nan, float)
    density_map[valid] = Ne_prof[ridx[valid]]

    # --- 2D 最近傍で最小限の穴埋め（任意）
    if spatial_fill_mode in ("nearest", "itermean"):
        from scipy.ndimage import distance_transform_edt, convolve
        def _fill_nearest(arr, valid_mask):
            filled = arr.copy()
            src = valid_mask & np.isfinite(filled)
            miss = valid_mask & ~src
            if not (np.any(miss) and np.any(src)):
                return filled
            _, (iy, ix) = distance_transform_edt(~src, return_indices=True)
            filled[miss] = filled[iy[miss], ix[miss]]
            return filled

        def _iter_mean(arr, valid_mask, max_iters=1):
            out = arr.copy()
            ker = np.array([[1,1,1],[1,0,1],[1,1,1]], float)
            for _ in range(int(max_iters)):
                nanm = valid_mask & ~np.isfinite(out)
                if not np.any(nanm): break
                fin = np.isfinite(out).astype(float)
                cnt = convolve(fin, ker, mode='constant', cval=0.0)
                sm  = convolve(np.nan_to_num(out, nan=0.0), ker, mode='constant', cval=0.0)
                mean_nb = np.divide(sm, np.maximum(cnt, 1.0), where=(cnt>0))
                fillable = nanm & (cnt > 0)
                out[fillable] = mean_nb[fillable]
            return out

        if spatial_fill_mode == "nearest":
            density_map = _fill_nearest(density_map, valid)
        else:
            density_map = _iter_mean(density_map, valid, max_iters=int(spatial_fill_iters))

    aux = dict(r_edges=r_edges, r_mid=r_mid, Ne_profile_1d=Ne_prof)
    return density_map, aux
def invert_pb_to_density_2D(instrument,
                            pb_image,
                            r_map_rsun,
                            params_ref,
                            r_min,
                            r_max,
                            symmetry="axisymmetric",
                            **kwargs):
    """
    symmetry in {"axisymmetric","spherical"} でモード切替。
    返り値: (density_map_2D, aux_profiles_dict)

    Parameters
    ----------
    instrument : str
        set_u_from_instrument() に渡す観測機器名。
        例: "Mk4", "K-Cor", "SOHO/LASCO", "lasco_c2" など。
    """
    sym = str(symmetry).lower()
    if sym in ("axisymmetric", "axis", "axi"):
        return invert_pb_to_density_axisymmetric(
            instrument,
            pb_image,
            r_map_rsun,
            params_ref,
            r_min=r_min,
            r_max=r_max,
            **kwargs
        )
    elif sym in ("spherical", "sph", "sphere"):
        return invert_pb_to_density_spherical(
            instrument,
            pb_image,
            r_map_rsun,
            params_ref,
            r_min=r_min,
            r_max=r_max,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown symmetry='{symmetry}'. Use 'axisymmetric' or 'spherical'.")

def invert_per_theta_profiles(
    pb_image: np.ndarray,
    r_map_rsun: np.ndarray,
    params_ref: dict,
    r_min: float,
    r_max: float,
    dr: float,
    theta_step_deg: float = 10.0,
    theta_mode: str = '0to360',
    trim_to_last_finite: bool = True,
    rmax_margin: float = 0.02,
    radial_gap_bins: int = 4,
    theta_neighbor_blend: int = 4,
    r_edges: np.ndarray | None = None    # ← 追加: 非一様ビンに対応
):
    """
    Build Ne(r, theta_center) by extracting pB(r) per θ-sector and inverting.
    If r_edges is provided (non-uniform), it overrides r_min/r_max/dr.
    """
    ny, nx = pb_image.shape
    cy = params_ref['cy']
    cx = params_ref['cx']
    
    th_map_deg = angle_map_deg(ny, nx, cy, cx, mode='0to360')
    

    # --- radial bins ---
    if r_edges is not None:
        r_edges = np.asarray(r_edges, dtype=float)
        if r_edges.ndim != 1 or r_edges.size < 2:
            raise ValueError("r_edges must be 1D with at least 2 elements.")
        # safety: trim to available map range a bit inside
        r_edges = r_edges[(r_edges >= r_min - 1e-6) & (r_edges <= r_max + 1e-6)]
        if r_edges.size < 2:
            raise ValueError("r_edges collapsed after trimming to [r_min, r_max].")
        r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    else:
        r_max_eff = min(r_max, np.nanmax(r_map_rsun) - 1e-6) - abs(rmax_margin)
        r_edges = np.arange(r_min, r_max_eff + dr, dr)
        if len(r_edges) < 2:
            raise ValueError("r_edges has fewer than 2 points")
        r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    n_r = len(r_mid)
    
    r_bin_idx_full = digitize_radius(r_map_rsun, r_edges)

    # --- theta centers ---
    if theta_mode == '0to360':
        theta_centers = np.arange(0.0, 360.0, theta_step_deg)
    else:
        theta_centers = np.arange(-180.0, 180.0, theta_step_deg)
        theta_centers = np.where(theta_centers < 0.0, theta_centers + 360.0, theta_centers)
    n_theta = len(theta_centers)

    # --- First pass: pB(r) per theta ---
    pB_profiles = np.full((n_theta, n_r), np.nan, dtype=float)
    for k, th in enumerate(theta_centers):
        # 既存の抽出（±5°固定）
        pB_k = extract_pB_profile(
            pb_image, r_map_rsun, th, r_edges,
            cy_center=cy, cx_center=cx, ny_grid=ny, nx_grid=nx
        )
        pB_k = _interp_small_radial_gaps(pB_k, max_gap_bins=int(radial_gap_bins))

        # --- 追加：NaNが目立つビンだけ、角半幅を広げて取り直す ---
        nan_mask = ~np.isfinite(pB_k)
        # しきい値はお好みで：NaN>10% なら発動、など
        if np.count_nonzero(nan_mask) > 0:
            pB_k_fb = _adaptive_pb_profile_for_theta(
                pb_image=pb_image,
                r_bin_idx=r_bin_idx_full,
                theta_map_deg=th_map_deg,
                theta_center_deg=float(th),
                n_r=n_r,
                base_halfwidth_deg=5.0,
                max_halfwidth_deg=20.0,
                halfwidth_step_deg=2.0,
                min_samples=60
            )
            # 既存値を優先し、NaN のところだけ置き換え
            pB_k[nan_mask] = pB_k_fb[nan_mask]

        pB_profiles[k, :] = pB_k
    # --- Second pass: blend across theta for remaining NaNs ---
    if theta_neighbor_blend > 0:
        pB_profiles = _blend_theta_neighbors(pB_profiles, max_neighbors=int(theta_neighbor_blend))

    # --- Inversion per theta ---
    Ne_profiles = np.full_like(pB_profiles, np.nan)
    for k in range(n_theta):
        pB_k = pB_profiles[k, :]
        if not np.any(np.isfinite(pB_k)):
            continue

        finite_idx = np.where(np.isfinite(pB_k) & (pB_k > 0))[0]
        if finite_idx.size == 0:
            finite_idx = np.where(np.isfinite(pB_k))[0]
        last = int(finite_idx[-1])

        pB_sub     = pB_k[:last+1]
        r_mid_sub  = r_mid[:last+1]
        r_edges_sub = r_edges[:last+2]

        try:
            Ne_sub = invert_ablation(pB_sub, r_mid_sub, r_edges_sub, len(r_mid_sub))
            Ne_k = np.full_like(pB_k, np.nan)
            Ne_k[:last+1] = Ne_sub
        except Exception as e:
            print(f"[WARN] inversion failed at θ index={k} (center={theta_centers[k]:.1f}°): {e}")
            Ne_k = np.full_like(pB_k, np.nan)
        Ne_k = _interp_small_radial_gaps(Ne_k, max_gap_bins=4)
        Ne_profiles[k, :] = Ne_k

    return {
        'theta_centers_deg': theta_centers,
        'r_edges': r_edges,
        'r_mid': r_mid,
        'Ne_profiles': Ne_profiles,
        'pB_profiles': pB_profiles
    }

from scipy.ndimage import distance_transform_edt, convolve

def _fill_nan_nearest_2d(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    filled = arr.copy()
    src_mask = valid_mask & np.isfinite(filled)
    invalid = valid_mask & ~src_mask
    if not np.any(invalid) or not np.any(src_mask):
        return filled
    _, (iy, ix) = distance_transform_edt(~src_mask, return_indices=True)
    filled[invalid] = filled[iy[invalid], ix[invalid]]
    return filled

def _iterative_neighbor_mean_fill(arr: np.ndarray, valid_mask: np.ndarray, max_iters: int = 2) -> np.ndarray:
    out = arr.copy()
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=float)
    for _ in range(max_iters):
        nan_mask = valid_mask & ~np.isfinite(out)
        if not np.any(nan_mask):
            break
        finite = np.isfinite(out).astype(float)
        count  = convolve(finite, kernel, mode='constant', cval=0.0)
        sumv   = convolve(np.nan_to_num(out, nan=0.0), kernel, mode='constant', cval=0.0)
        mean_nb = np.divide(sumv, np.maximum(count, 1.0), where=(count>0))
        fillable = nan_mask & (count > 0)
        out[fillable] = mean_nb[fillable]
    return out


def _fill_missing_with_theta_neighbors(
    density_map: np.ndarray,
    valid_mask: np.ndarray,
    theta_idx: np.ndarray,
    r_idx: np.ndarray,
    Ne_profiles: np.ndarray,
    max_theta_neighbors: int
):
    """
    Fill NaNs in 'density_map' by looking at ±Δθ sector neighbors (nearest-neighbor in r).
    Operates in-place on density_map.
    """
    if max_theta_neighbors <= 0:
        return

    ny, nx = density_map.shape
    n_theta = Ne_profiles.shape[0]

    # 欠損ピクセルの一次元位置
    miss_flat = np.where(valid_mask.ravel() & ~np.isfinite(density_map).ravel())[0]
    if miss_flat.size == 0:
        return

    t_miss = theta_idx.ravel()[miss_flat]
    r_miss = r_idx.ravel()[miss_flat]

    for d in range(1, max_theta_neighbors + 1):
        if miss_flat.size == 0:
            break

        filled_any = False
        for sgn in (-1, +1):
            t_alt = (t_miss + sgn * d) % n_theta
            vals = Ne_profiles[t_alt, r_miss]
            ok = np.isfinite(vals)

            if np.any(ok):
                density_map.ravel()[miss_flat[ok]] = vals[ok]
                # 残りの欠損を更新
                keep = ~ok
                miss_flat = miss_flat[keep]
                t_miss = t_miss[keep]
                r_miss = r_miss[keep]
                filled_any = True

        if not filled_any:
            # この距離では何も埋まらなかった → 次の距離へ
            continue

def _fill_missing_with_theta_and_radial_neighbors(
    density_map: np.ndarray,
    valid_mask: np.ndarray,
    theta_idx: np.ndarray,
    r_idx: np.ndarray,
    Ne_profiles: np.ndarray,
    max_theta_neighbors: int = 2,
    max_radial_neighbors: int = 2
):
    """
    Fill NaNs in density_map by looking at neighbors in θ (±dθ sectors) and r (±dr bins).
    1) まず同一θで r±dr（dr=1..max_radial_neighbors）
    2) まだNaNなら θ±dθ（dθ=1..max_theta_neighbors）で r 同じ
    3) それでも残れば (θ±dθ, r±dr) の組を小さい順にチェック
    """
    ny, nx = density_map.shape
    n_theta, n_r = Ne_profiles.shape

    miss_flat = np.where(valid_mask.ravel() & ~np.isfinite(density_map).ravel())[0]
    if miss_flat.size == 0:
        return

    t_miss = theta_idx.ravel()[miss_flat]
    r_miss = r_idx.ravel()[miss_flat]

    # 1) 同一θで r±dr
    for dr in range(1, max_radial_neighbors + 1):
        ok = np.zeros_like(miss_flat, dtype=bool)
        for sgn in (-1, +1):
            r_alt = np.clip(r_miss + sgn * dr, 0, n_r - 1)
            vals = Ne_profiles[t_miss, r_alt]
            has = np.isfinite(vals)
            density_map.ravel()[miss_flat[has]] = vals[has]
            ok |= has
        # 未充填のみ更新
        keep = ~ok
        miss_flat, t_miss, r_miss = miss_flat[keep], t_miss[keep], r_miss[keep]
        if miss_flat.size == 0:
            return

    # 2) θ±dθで rは同じ
    for dt in range(1, max_theta_neighbors + 1):
        ok = np.zeros_like(miss_flat, dtype=bool)
        for sgn in (-1, +1):
            t_alt = (t_miss + sgn * dt) % n_theta
            vals = Ne_profiles[t_alt, r_miss]
            has = np.isfinite(vals)
            density_map.ravel()[miss_flat[has]] = vals[has]
            ok |= has
        keep = ~ok
        miss_flat, t_miss, r_miss = miss_flat[keep], t_miss[keep], r_miss[keep]
        if miss_flat.size == 0:
            return

    # 3) (θ±dθ, r±dr) の組
    for dt in range(1, max_theta_neighbors + 1):
        for dr in range(1, max_radial_neighbors + 1):
            ok = np.zeros_like(miss_flat, dtype=bool)
            for sgn_t in (-1, +1):
                for sgn_r in (-1, +1):
                    t_alt = (t_miss + sgn_t * dt) % n_theta
                    r_alt = np.clip(r_miss + sgn_r * dr, 0, n_r - 1)
                    vals = Ne_profiles[t_alt, r_alt]
                    has = np.isfinite(vals)
                    density_map.ravel()[miss_flat[has]] = vals[has]
                    ok |= has
            keep = ~ok
            miss_flat, t_miss, r_miss = miss_flat[keep], t_miss[keep], r_miss[keep]
            if miss_flat.size == 0:
                return


def build_density_map_from_profiles(
    r_map_rsun: np.ndarray,
    params_ref: dict,
    profiles: dict,
    valid_rmin: float,
    valid_rmax: float,
    mask_nan_outside: bool = True,
    theta_neighbor_fallback: int = 0,
    spatial_fill_mode: str = "nearest",
    spatial_fill_iters: int = 2,
    use_bilinear: bool = True    # ← 追加: 双一次補間を有効化
) -> np.ndarray:
    ny, nx = r_map_rsun.shape
    cy = params_ref['cy']; cx = params_ref['cx']

    # 角度マップ（0..360）
    th_map = angle_map_deg(ny, nx, cy, cx, mode='0to360')

    theta_centers = profiles['theta_centers_deg']   # 等間隔前提
    r_edges = profiles['r_edges']
    r_mid = profiles['r_mid']
    Ne_profiles = profiles['Ne_profiles']           # (n_theta, n_r)
    n_theta, n_r = Ne_profiles.shape

    # 有効領域マスク
    valid = np.ones((ny, nx), dtype=bool)
    if mask_nan_outside:
        valid &= (r_map_rsun >= valid_rmin) & (r_map_rsun <= valid_rmax)

    # r のビンと小数部
    r_idx = np.digitize(r_map_rsun, r_edges) - 1  # 左ビン
    r_idx = np.where(r_idx < 0, -1, np.where(r_idx >= n_r, -1, r_idx))
    valid &= (r_idx >= 0)
    r_idx_safe = np.clip(r_idx, 0, n_r - 1)  # n_r = len(r_centers)

    # 小数部（右端では 0）
    re_left  = r_edges[r_idx_safe]
    re_right = r_edges[r_idx_safe + 1]
    dr = np.maximum(re_right - re_left, 1e-12)
    r_frac = np.clip((r_map_rsun - re_left) / dr, 0.0, 1.0)
    r_idxp1 = np.clip(r_idx + 1, 0, n_r - 1)

    density_map = np.full((ny, nx), np.nan, dtype=float)

    if use_bilinear:
        # θ の両側セクタと角度重み
        i0, i1, w0_th, w1_th = _theta_linear_weights(th_map, theta_centers)

        # 2×2 サンプルを取得（θ×r）
        Ne_i0_k   = Ne_profiles[i0, r_idx]
        Ne_i0_k1  = Ne_profiles[i0, r_idxp1]
        Ne_i1_k   = Ne_profiles[i1, r_idx]
        Ne_i1_k1  = Ne_profiles[i1, r_idxp1]

        # まず r 方向で線形補間（NaN に頑健）
        Ne_i0 = _nanblend(Ne_i0_k, Ne_i0_k1, r_frac)
        Ne_i1 = _nanblend(Ne_i1_k, Ne_i1_k1, r_frac)
        # 次に θ 方向で線形補間（NaN に頑健）
        Ne_pix = _nanblend(Ne_i0, Ne_i1, w1_th)
        density_map[valid] = Ne_pix[valid]
    else:
        # 旧：最近傍（残したい場合）
        theta_idx = digitize_theta(th_map, theta_centers)
        flat_idx = np.where(valid.ravel())[0]
        t_sel = theta_idx.ravel()[flat_idx]
        r_sel = r_idx.ravel()[flat_idx]
        density_map.ravel()[flat_idx] = Ne_profiles[t_sel, r_sel]

    # なお残る穴は θ・r 近傍フォールバックで穴埋め
    _fill_missing_with_theta_and_radial_neighbors(
        density_map=density_map,
        valid_mask=valid,
        theta_idx=np.argmin(np.abs((th_map[...,None]-theta_centers[None,None,:]+180)%360-180), axis=-1),
        r_idx=r_idx,
        Ne_profiles=Ne_profiles,
        max_theta_neighbors=int(theta_neighbor_fallback),
        max_radial_neighbors=2
    )

    # 2D 補間（任意）
    if spatial_fill_mode == "nearest":
        density_map = _fill_nan_nearest_2d(density_map, valid)
    elif spatial_fill_mode == "itermean":
        density_map = _iterative_neighbor_mean_fill(density_map, valid, max_iters=int(spatial_fill_iters))
    return density_map


# ---------------------------------
# Plotting (separate from inversion)
# ---------------------------------

def plot_density_map(
    density_map: np.ndarray,
    r_map_plot: np.ndarray,
    params_ref: dict,
    r_ranges: dict,
    title: str = "2D Electron Density (from pB inversion)",
    vmin: float | None = None,
    vmax: float | None = None,
    xlim_pix: tuple[float, float] | None = None,
    ylim_pix: tuple[float, float] | None = None,
    min_angle_deg: float = 140.0,
    max_angle_deg: float = 201.0,
    angle_step_deg: float = 10.0,
):
    """
    Plot a 2D density map with logarithmic color scale.
    Axes coordinates are in *pixels* relative to the solar disk center (0,0).

    Parameters
    ----------
    density_map : 2D array
        Electron density [cm^-3].
    r_map_plot : 2D array
        Radial distance in R_sun (used only for contour levels).
    params_ref : dict
        Must contain 'cx', 'cy', 'nx', 'ny', 'px_per_rsun'.
    r_ranges : dict
        Keys: 'mk4_inner', 'mk4_outer_lasco_inner', 'lasco_outer' (in R_sun).
    title : str
        Plot title.
    vmin, vmax : float or None
        Color scale limits for Ne [cm^-3].
    xlim_pix, ylim_pix : (min, max) or None
        Display range in *pixels*, in the same coordinates as the axes
        (0,0 at disk center). Example: xlim_pix=(-700, 700), ylim_pix=(0, 1400).
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # local import
    from matplotlib.ticker import LogLocator

    ny, nx = density_map.shape

    # 軸の単位は「太陽中心からのピクセル」
    extent_pixels = [-params_ref['cx'], params_ref['nx'] - params_ref['cx'],
                     -params_ref['cy'], params_ref['ny'] - params_ref['cy']]

    # カラースケール
    if vmin is None:
        vmin = 1e5
    if vmax is None:
        vmax = 1e9

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color='lightgray')

    im = ax.imshow(
        density_map,
        origin='lower',
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=extent_pixels,
        aspect='equal'
    )

    # 整数 Rsun の等高線
    int_levels = np.arange(1, int(np.floor(r_ranges['lasco_outer'])) + 1)
    ax.contour(
        r_map_plot,
        levels=int_levels,
        colors='white',
        linewidths=1,
        linestyles='--',
        extent=extent_pixels,
        alpha=0.7
    )

    # 境界リング
    boundary_lines_for_legend = []
    for level_val, (label_text, color) in [
        (r_ranges['mk4_inner'], (f"{r_ranges['mk4_inner']:.1f} $R_\\odot$ (Mk4 inner)", 'magenta')),
        (r_ranges['mk4_outer_lasco_inner'], (f"{r_ranges['mk4_outer_lasco_inner']:.1f} $R_\\odot$ (Mk4/LASCO)", 'green')),
        (r_ranges['lasco_outer'], (f"{r_ranges['lasco_outer']:.1f} $R_\\odot$ (LASCO outer)", 'blue')),
    ]:
        if level_val <= np.nanmax(r_map_plot) and level_val >= np.nanmin(r_map_plot):
            ax.contour(
                r_map_plot,
                levels=[level_val],
                colors=[color],
                linewidths=1.2,
                linestyles='-.',
                extent=extent_pixels
            )
            proxy_line = plt.Line2D([0], [0], linestyle='-.', color=color, linewidth=1.2, label=label_text)
            boundary_lines_for_legend.append(proxy_line)

    # 太陽中心マーク
    ax.plot(0, 0, '+', color='black', markersize=12, markeredgewidth=1.5)

    # 140–200°を10°刻みでガイドラインを描画（プロファイル計算は行わない）
    try:
        angles_deg = np.arange(min_angle_deg, max_angle_deg, angle_step_deg)
        cmap = plt.cm.get_cmap("viridis")
        norm = plt.Normalize(vmin=angles_deg.min(), vmax=angles_deg.max())
        r_line = np.linspace(r_ranges['mk4_inner'], r_ranges['lasco_outer'], 200)
        scale = params_ref['px_per_rsun']
        for th in angles_deg:
            angle_rad = np.deg2rad(th)
            x_vals = r_line * scale * np.cos(angle_rad)
            y_vals = r_line * scale * np.sin(angle_rad)
            color = cmap(norm(th))
            ax.plot(x_vals, y_vals, color=color, linewidth=1.8, alpha=0.9)
        # 簡易凡例（範囲のみ表示）
        proxy = plt.Line2D([0], [0], color=cmap(norm(angles_deg.mean())), linewidth=2, label="θ=140–200° (10° step)")
        ax.legend(handles=[proxy], loc='upper right', fontsize=10)
    except Exception as e:
        print(f"Multi-angle guideline draw error: {e}")

    # ★ ここで表示範囲をピクセル単位で指定 ★
    if xlim_pix is not None:
        ax.set_xlim(xlim_pix)
    if ylim_pix is not None:
        ax.set_ylim(ylim_pix)
    # ---------------------------------------

    # カラーバー
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label('N$_e$ [cm$^{-3}$]', fontsize=16)
    cb.ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    cb.ax.tick_params(labelsize=14)

    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    if boundary_lines_for_legend:
        ax.legend(handles=boundary_lines_for_legend, loc='upper right', fontsize=12)

    plt.tight_layout()
    return fig, ax

def plot_density_maps_combined(
    density_mk4: np.ndarray,
    r_map_mk4: np.ndarray,
    params_mk4: dict,
    density_lasco: np.ndarray,
    r_map_lasco: np.ndarray,
    params_lasco: dict,
    r_ranges: dict,
    title_mk4: str = "K-Cor Electron Density",
    title_lasco: str = "SOHO/LASCO-C2 Electron Density",
    vmin: float | None = None,
    vmax: float | None = None,
    xlim_pix_mk4: tuple[float, float] | None = None,
    ylim_pix_mk4: tuple[float, float] | None = None,
    xlim_pix_lasco: tuple[float, float] | None = None,
    ylim_pix_lasco: tuple[float, float] | None = None,
):
    """
    K-COR と LASCO-C2 の 2D density map を 1 つの Figure に 2 パネルとして描画する。
    軸はどちらも「太陽中心を原点とした Pixel 座標」。
    カラースケール (vmin, vmax) は 2 パネルで共通。
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import LogLocator

    if vmin is None:
        vmin = 1e5
    if vmax is None:
        vmax = 1e9

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False)
    ax_k, ax_l = axes

    # --------- K-COR パネル ---------
    ny_k, nx_k = density_mk4.shape
    cx_k = params_mk4['cx']; cy_k = params_mk4['cy']
    extent_k = [-cx_k, nx_k - cx_k, -cy_k, ny_k - cy_k]

    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color='lightgray')

    im_k = ax_k.imshow(
        density_mk4,
        origin='lower',
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=extent_k,
        aspect='equal'
    )

    # Rs=整数の等高線
    int_levels = np.arange(1, int(np.floor(r_ranges['lasco_outer'])) + 1)
    # ピクセル座標グリッド
    yy_k, xx_k = np.indices((ny_k, nx_k))
    x_pix_k = xx_k - cx_k
    y_pix_k = yy_k - cy_k
    ax_k.contour(
        x_pix_k, y_pix_k, r_map_mk4,
        levels=int_levels,
        colors='white',
        linewidths=1,
        linestyles='--',
        alpha=0.7
    )

    # 境界リング
    boundary_lines_for_legend = []
    for level_val, (label_text, color) in [
        (r_ranges['mk4_inner'], (f"{r_ranges['mk4_inner']:.1f} $R_\\odot$ (Mk4 inner)", 'magenta')),
        (r_ranges['mk4_outer_lasco_inner'], (f"{r_ranges['mk4_outer_lasco_inner']:.1f} $R_\\odot$ (Mk4/LASCO)", 'green')),
        (r_ranges['lasco_outer'], (f"{r_ranges['lasco_outer']:.1f} $R_\\odot$ (LASCO outer)", 'blue')),
    ]:
        if np.nanmin(r_map_mk4) <= level_val <= np.nanmax(r_map_mk4):
            cs = ax_k.contour(
                x_pix_k, y_pix_k, r_map_mk4,
                levels=[level_val],
                colors=[color],
                linewidths=1.2,
                linestyles='-.'
            )
            proxy_line = plt.Line2D([0], [0], linestyle='-.', color=color, linewidth=1.2, label=label_text)
            boundary_lines_for_legend.append(proxy_line)
            
    theta_deg_overlay = 150.0
    angle_rad = np.deg2rad(theta_deg_overlay)

    # 軸は「太陽中心を原点とした pixel 座標」なので、Rsun→pixel変換が必要
    px_per_rsun_k = float(params_mk4['px_per_rsun'])

    # どこまで線を出すか（ここではLASCO outerまで）
    r_line = np.linspace(r_ranges['mk4_inner'], r_ranges['lasco_outer'], 400)
    x_vals = r_line * px_per_rsun_k * np.cos(angle_rad)
    y_vals = r_line * px_per_rsun_k * np.sin(angle_rad)

    line_artist_theta, = ax_k.plot(
        x_vals, y_vals,
        color='cyan', linestyle='-', linewidth=2,
        label=f'θ={theta_deg_overlay:.0f}°'
    )

    
    ax_k.plot(0, 0, '+', color='black', markersize=10, markeredgewidth=1.5)
    if xlim_pix_mk4 is not None:
        ax_k.set_xlim(xlim_pix_mk4)
    if ylim_pix_mk4 is not None:
        ax_k.set_ylim(ylim_pix_mk4)
    ax_k.set_title(title_mk4, fontsize=14)
    ax_k.set_xlabel("Solar-X [pixel]", fontsize=12)
    ax_k.set_ylabel("Solar-Y [pixel]", fontsize=12)
    ax_k.tick_params(axis='both', which='major', labelsize=10)
    if boundary_lines_for_legend:
        ax_k.legend(handles=boundary_lines_for_legend, loc='upper right', fontsize=9)

    # --------- LASCO パネル ---------
    ny_l, nx_l = density_lasco.shape
    cx_l = params_lasco['cx']; cy_l = params_lasco['cy']
    extent_l = [-cx_l, nx_l - cx_l, -cy_l, ny_l - cy_l]

    im_l = ax_l.imshow(
        density_lasco,
        origin='lower',
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=extent_l,
        aspect='equal'
    )

    yy_l, xx_l = np.indices((ny_l, nx_l))
    x_pix_l = xx_l - cx_l
    y_pix_l = yy_l - cy_l
    ax_l.contour(
        x_pix_l, y_pix_l, r_map_lasco,
        levels=int_levels,
        colors='white',
        linewidths=1,
        linestyles='--',
        alpha=0.7
    )

    # LASCO 側にも境界リング（凡例は K-COR 側にまとめたのでここは線だけ）
    for level_val, color in [
        (r_ranges['mk4_inner'], 'magenta'),
        (r_ranges['mk4_outer_lasco_inner'], 'green'),
        (r_ranges['lasco_outer'], 'blue'),
    ]:
        if np.nanmin(r_map_lasco) <= level_val <= np.nanmax(r_map_lasco):
            ax_l.contour(
                x_pix_l, y_pix_l, r_map_lasco,
                levels=[level_val],
                colors=[color],
                linewidths=1.2,
                linestyles='-.'
            )

    ax_l.plot(0, 0, '+', color='black', markersize=10, markeredgewidth=1.5)
    if xlim_pix_lasco is not None:
        ax_l.set_xlim(xlim_pix_lasco)
    if ylim_pix_lasco is not None:
        ax_l.set_ylim(ylim_pix_lasco)
    ax_l.set_title(title_lasco, fontsize=14)
    ax_l.set_xlabel("Solar-X [pixel]", fontsize=12)
    ax_l.tick_params(axis='both', which='major', labelsize=10)

    # --------- 共通カラーバー ---------
    # 右側に 1 本だけ付ける
    divider = make_axes_locatable(ax_l)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cb = plt.colorbar(im_l, cax=cax)
    cb.set_label('N$_e$ [cm$^{-3}$]', fontsize=12)
    cb.ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    cb.ax.tick_params(labelsize=10)

    fig.tight_layout()
    return fig, (ax_k, ax_l)

# ---------------------------------
# Sampling helpers
# ---------------------------------

def get_density_at_pixels(density_map: np.ndarray, yx_list: list[tuple[int,int]]) -> np.ndarray:
    """
    Sample the 2D density_map at integer pixel coordinates [(y,x), ...].
    Returns array of sampled values (NaN where out of bounds or NaN in map).
    """
    ny, nx = density_map.shape
    vals = []
    for (y, x) in yx_list:
        if 0 <= y < ny and 0 <= x < nx:
            vals.append(density_map[y, x])
        else:
            vals.append(np.nan)
    return np.array(vals, dtype=float)

def get_density_at_polar(
    r_rsun: np.ndarray,
    theta_deg: np.ndarray,
    profiles: dict
) -> np.ndarray:
    """
    Sample Ne at given (r [R_sun], theta [deg]) from the per-θ profiles (without using the 2D map).

    This uses nearest-neighbor in θ (circular) and nearest bin in r.
    If you need interpolation, you can replace the nearest-neighbor picks with
    linear interpolation over r and circular interpolation over θ.
    """
    r_edges = profiles['r_edges']
    r_mid = profiles['r_mid']
    Ne_profiles = profiles['Ne_profiles']
    theta_centers = profiles['theta_centers_deg']

    # Map theta to nearest sector
    th = np.asarray(theta_deg).copy()
    th = np.where(th < 0.0, th + 360.0, th)
    # [N, 1] vs [M]
    diff = np.abs(((th[..., None] - theta_centers[None, :]) + 180.0) % 360.0 - 180.0)
    t_idx = np.argmin(diff, axis=-1)  # [N]

    # Map r to nearest bin
    r = np.asarray(r_rsun)
    r_bin = np.digitize(r, r_edges) - 1
    r_bin = np.clip(r_bin, 0, len(r_mid)-1)  # clip

    return Ne_profiles[t_idx, r_bin]

def export_density_csv(
    density_map: np.ndarray,
    r_map_rsun: np.ndarray,
    params_ref: dict,
    csv_path: str,
    include_all: bool = False
) -> None:
    """
    Save per-pixel position and density to CSV.

    Columns:
      y_pix, x_pix         : integer pixel indices
      x_Rsun, y_Rsun       : solar-centric Cartesian coords [R_sun]
      r_Rsun, theta_deg    : polar coords [R_sun], [deg, 0..360)
      Ne_cm^-3             : electron density [cm^-3]

    include_all=False のとき、有限かつ >0 の Ne のみ出力。
    True にするとマスクせず全部（NaNは空欄として）を書き出します。
    """
    # 形状と中心・スケール
    ny, nx = density_map.shape
    cx = params_ref['cx']; cy = params_ref['cy']
    px_per_rsun = params_ref['px_per_rsun']

    # 角度マップ（0..360deg）
    th_map = angle_map_deg(ny, nx, cy, cx, mode='0to360')

    # ピクセル座標と Rsun 座標
    yy, xx = np.indices((ny, nx))
    x_Rsun = (xx - cx) / px_per_rsun
    y_Rsun = (yy - cy) / px_per_rsun

    # 書き出しマスク
    if include_all:
        mask = np.ones_like(density_map, dtype=bool)
    else:
        mask = np.isfinite(density_map) & (density_map > 0)

    # フラット化して列に束ねる
    y_pix_col   = yy[mask].ravel().astype(int)
    x_pix_col   = xx[mask].ravel().astype(int)
    x_Rsun_col  = x_Rsun[mask].ravel()
    y_Rsun_col  = y_Rsun[mask].ravel()
    r_Rsun_col  = r_map_rsun[mask].ravel()
    th_deg_col  = th_map[mask].ravel()
    Ne_col      = density_map[mask].ravel()

    # CSV 保存（ヘッダ付き）
    header = "y_pix,x_pix,x_Rsun,y_Rsun,r_Rsun,theta_deg,Ne_cm^-3"
    data = np.column_stack([
        y_pix_col, x_pix_col,
        x_Rsun_col, y_Rsun_col,
        r_Rsun_col, th_deg_col,
        Ne_col
    ])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")




def ne_to_fpe(ne_cm3: np.ndarray, harmonic: int = 1, out_unit: str = "MHz") -> np.ndarray:
    """
    f_pe[Hz] = 8980 * sqrt(n_e[cm^-3])
    harmonic=1: fpe, harmonic=2: 2fpe
    """
    ne = np.asarray(ne_cm3, dtype=float)
    ne = np.where(ne > 0, ne, np.nan)
    f_hz = 8980.0 * np.sqrt(ne) * float(harmonic)

    u = out_unit.lower()
    if u == "hz":
        return f_hz
    if u == "khz":
        return f_hz / 1e3
    if u == "mhz":
        return f_hz / 1e6
    raise ValueError(f"out_unit must be Hz/kHz/MHz, got {out_unit}")


def write_2d_fits_like(ref_header, data2d: np.ndarray, out_path: str, *, bunit: str, comment: str = "") -> None:
    """
    ref_header（LASCOヘッダなど）を可能な限り保持して2D FITSを書き出す。
    """
    hdr = fits.Header(ref_header)  # Headerでもdictでも受けられる
    # 形が違う場合は軸サイズだけ合わせる（WCSが破綻するなら前処理側を見直す）
    hdr["NAXIS"]  = 2
    hdr["NAXIS1"] = int(data2d.shape[1])
    hdr["NAXIS2"] = int(data2d.shape[0])
    hdr["BUNIT"]  = bunit
    if comment:
        hdr.add_comment(comment)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fits.PrimaryHDU(np.asarray(data2d, dtype=np.float32), header=hdr).writeto(out_path, overwrite=True)

# ---------------------------------
# Example main (optional demo)
# ---------------------------------

if __name__ == "__main__":
    # === 入力ファイル（既存と同じ） ===
    filename_mk4   = r'/mnt/d/wsl/home/kinno-7010/Research/MK4_coronagraph/MK4_coronagraph_KCOR/pB/Rawdata/20220613_025810_kcor_l2_pb.fts'
    filename_lasco = r'/mnt/d/wsl/home/kinno-7010/Research/SOHO/pB/C2-PB-20220613_0258.fts'

    # === 反転モードを選択 ===
    #   'axisymmetric'  : セクタ毎の軸対称反転（従来方式）
    #   'spherical'     : 全周平均 → 1D 反転 → 2D 段差塗り
    SYMMETRY = "axisymmetric"  # ← "spherical" に変えるだけで球対称版
    
    # === FITS保存と出力量（parseはしない：ここを編集して切替） ===
    SAVE_FITS = True

    # "ne" か "fpe" を選ぶ（fpeの場合は MHz 等に変換して保存）
    OUTPUT_QUANTITY = "ne"     # or "fpe"
    FPE_HARMONIC = 1           # 1: fpe, 2: 2fpe
    FPE_UNIT = "MHz"           # "Hz" / "kHz" / "MHz"

    # === 読み込み・前処理 ===
    # instrument 名は set_u_from_instrument() が解釈可能な文字列にしておく
    inst_mk4   = "K-Cor"
    inst_lasco = "SOHO/LASCO"

    data_mk4,   params_mk4   = load_and_prepare_instrument_data(filename_mk4, inst_mk4)
    data_lasco, params_lasco = load_and_prepare_instrument_data(filename_lasco, inst_lasco, is_lasco=True)

    # --- LASCO グリッド上の r マップ（従来通り） ---
    r_ranges = {'mk4_inner': 1.0, 'mk4_outer_lasco_inner': 2.2, 'lasco_outer': 6.0}
    _y_l, _x_l = np.indices((params_lasco['ny'], params_lasco['nx']))
    r_map_lasco = np.hypot((_x_l - params_lasco['cx']) / params_lasco['px_per_rsun'],
                           (_y_l - params_lasco['cy']) / params_lasco['px_per_rsun'])

    # --- K-COR グリッド上の r マップ（新規：K-COR 単独反転用） ---
    _y_k, _x_k = np.indices((params_mk4['ny'], params_mk4['nx']))
    r_map_mk4 = np.hypot((_x_k - params_mk4['cx']) / params_mk4['px_per_rsun'],
                         (_y_k - params_mk4['cy']) / params_mk4['px_per_rsun'])

    # --- Mk4 + LASCO の合成 pB（必要なら今後も利用できるように残す） ---
    final_pb = combine_corona_data(
        data_lasco, params_lasco,
        data_mk4,   params_mk4,
        r_map_lasco,
        r_ranges
    )

    # --- 自動 r-edges（Mk4 内側 + LASCO 外側に対応） ---
    r_min = r_ranges['mk4_inner']
    r_trn = r_ranges['mk4_outer_lasco_inner']   # Mk4→LASCO の遷移半径
    r_max = r_ranges['lasco_outer']

    r_edges = build_adaptive_radial_edges(
        r_min=r_min,
        r_transition=r_trn,
        r_max=r_max,
        params_mk4=params_mk4,
        params_lasco=params_lasco,
        inner_px_factor=3.0,   # Mk4 側は ~2–3 px 程度の厚み
        outer_px_factor=2.0,   # LASCO 側は ~2 px 程度の厚み
        min_dr=0.01,
        max_dr=0.25
    )

    # --- 各機器ごとの反転半径範囲 ---
    # K-COR: 内側 1.1 R_sun 〜 合成切替半径 2.2 R_sun までを主に担当
    r_min_mk4 = r_ranges['mk4_inner']
    r_max_mk4 = r_ranges['mk4_outer_lasco_inner']

    # LASCO-C2: 2.2 R_sun 〜 7 R_sun を主に担当
    r_min_lasco = r_ranges['mk4_outer_lasco_inner']
    r_max_lasco = r_ranges['lasco_outer']

    # === 反転の実行（モード & instrument ごと） ===
    if SYMMETRY.lower().startswith("sph"):
        # --- 球対称：K-COR 単独 ---
        density_map_mk4, aux_mk4 = invert_pb_to_density_2D(
            instrument=inst_mk4,
            pb_image=data_mk4,
            r_map_rsun=r_map_mk4,
            params_ref=params_mk4,
            r_min=r_min_mk4,
            r_max=r_max_mk4,
            symmetry="spherical",
            r_edges=r_edges,              # 共通 r-edges（必要部分のみ内部で切り出し）
            spatial_fill_mode="nearest",  # ごく最小限の穴埋め
            spatial_fill_iters=1
        )

        # --- 球対称：LASCO-C2 単独 ---
        density_map_lasco, aux_lasco = invert_pb_to_density_2D(
            instrument=inst_lasco,
            pb_image=data_lasco,
            r_map_rsun=r_map_lasco,
            params_ref=params_lasco,
            r_min=r_min_lasco,
            r_max=r_max_lasco,
            symmetry="spherical",
            r_edges=r_edges,
            spatial_fill_mode="nearest",
            spatial_fill_iters=1
        )

        suffix = "sph"

    else:
        # --- 軸対称：K-COR 単独 ---
        density_map_mk4, aux_mk4 = invert_pb_to_density_2D(
            instrument=inst_mk4,
            pb_image=data_mk4,
            r_map_rsun=r_map_mk4,
            params_ref=params_mk4,
            r_min=r_min_mk4,
            r_max=r_max_mk4,
            symmetry="axisymmetric",
            theta_step_deg=2,           # extract_pB_profile(±5°) とタイル一致
            r_edges=r_edges,              # 非一様ビン（内側は Mk4 解像度）
            use_bilinear=False,           # ビン段差塗り（nearest bin）
            theta_neighbor_blend=2,       # 角度近傍の最小限ブレンド
            theta_neighbor_fallback=6,    # 残欠損の θ 近傍フォールバック
            spatial_fill_mode="nearest",  # 2D 最近傍で最小限穴埋め
            spatial_fill_iters=1
        )

        # --- 軸対称：LASCO-C2 単独 ---
        density_map_lasco, aux_lasco = invert_pb_to_density_2D(
            instrument=inst_lasco,
            pb_image=data_lasco,
            r_map_rsun=r_map_lasco,
            params_ref=params_lasco,
            r_min=r_min_lasco,
            r_max=r_max_lasco,
            symmetry="axisymmetric",
            theta_step_deg=2,
            r_edges=r_edges,              # 外側は LASCO 解像度のビン幅
            use_bilinear=False,
            theta_neighbor_blend=2,
            theta_neighbor_fallback=6,
            spatial_fill_mode="nearest",
            spatial_fill_iters=1
        )

        suffix = "axi"

    # --- LASCO グリッド上で K-COR 範囲を重ね合わせる ---
    if 'Ne_profiles' in aux_mk4:
        mk4_on_lasco = build_density_map_from_profiles(
            r_map_rsun=r_map_lasco,
            params_ref=params_lasco,
            profiles=aux_mk4,
            valid_rmin=r_min_mk4,
            valid_rmax=r_max_mk4,
            mask_nan_outside=True,
            theta_neighbor_fallback=6,
            spatial_fill_mode="nearest",
            spatial_fill_iters=1,
            use_bilinear=True
        )
    else:
        mk4_on_lasco = np.full_like(r_map_lasco, np.nan, dtype=float)
        ridx_mk4 = np.digitize(r_map_lasco, aux_mk4['r_edges']) - 1
        valid_bins = (ridx_mk4 >= 0) & (ridx_mk4 < aux_mk4['Ne_profile_1d'].size)
        mk4_on_lasco[valid_bins] = aux_mk4['Ne_profile_1d'][ridx_mk4[valid_bins]]
        mk4_on_lasco[(r_map_lasco < r_min_mk4) | (r_map_lasco > r_max_mk4)] = np.nan

    combined_density = np.full_like(r_map_lasco, np.nan, dtype=float)
    lasco_mask = (r_map_lasco >= r_min_lasco) & (r_map_lasco <= r_max_lasco) & np.isfinite(density_map_lasco)
    combined_density[lasco_mask] = density_map_lasco[lasco_mask]
    mk4_mask = (r_map_lasco >= r_min_mk4) & (r_map_lasco < r_max_mk4) & np.isfinite(mk4_on_lasco)
    combined_density[mk4_mask] = mk4_on_lasco[mk4_mask]

    # === プロット & 保存（重ね合わせ結果） ===
    out_base = r"/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/pB/2D_density_map"

    title_combined = (
        f"K-COR (1.0-2.2 $R_\\odot$) + SOHO/LASCO-C2 (2.2-6.0 $R_\\odot$)\n"
        f"Electron Density ({SYMMETRY} inversion) 2022-06-13 03:01:00 UT"
    )
    fig, ax = plot_density_map(
        density_map=combined_density,
        r_map_plot=r_map_lasco,
        params_ref=params_lasco,
        r_ranges=r_ranges,
        title=title_combined,
        xlim_pix=(-200, 0),
        ylim_pix=(-100, 150),
        min_angle_deg=140.0,
        max_angle_deg=161.0,
        angle_step_deg=10.0
    )
    png_path = f"{out_base}_Mk4_LASCO_{suffix}_20220613_0300.png"
    csv_path = f"{out_base}_Mk4_LASCO_{suffix}_20220613_0300.csv"
    fig.savefig(png_path, dpi=200)
    print(f"Saved combined 2D density map to {png_path}")
    #     # === FITS保存（トモグラフィ入力の pB と、QA用の ne / fpe） ===
    if SAVE_FITS:
        # LASCOの元ヘッダ（観測者座標・WCSをできるだけ保持したいのでLASCOを基準にする）
        hdr_lasco = fits.getheader(filename_lasco)
        
        try:
            hdr_kcor = fits.getheader(filename_mk4)
        except Exception as e:
            hdr_kcor = None
            print(f"[WARN] Could not read K-COR header for observer keywords: {e}")

        # (2) WCS 最低限（LASCO pb の元ヘッダに CTYPE が無いケースへの対処）
        if "CTYPE1" not in hdr_lasco:
            hdr_lasco["CTYPE1"] = "HPLN-TAN"
        if "CTYPE2" not in hdr_lasco:
            hdr_lasco["CTYPE2"] = "HPLT-TAN"
        if "CUNIT1" not in hdr_lasco:
            hdr_lasco["CUNIT1"] = "arcsec"
        if "CUNIT2" not in hdr_lasco:
            hdr_lasco["CUNIT2"] = "arcsec"
        hdr_lasco.add_history("Added/verified minimal HPC WCS keywords (CTYPE1/2, CUNIT1/2)")
        
        out_dir = "/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/pB/Rawdata"
        # 1) 合成pB（SSCトモグラフィで本質的に必要になるのはこれの“時系列”）
        pb_fits_path = os.path.join(out_dir, f"pB_Kcor_LASCO_{suffix}_20220613_0300.fits")
        # === Save Earth-view combined pB FITS for regularized tomography ===
        SAVE_COMBINED_PB_FITS = True
        pb_fits_path = f"{out_base}_Mk4_LASCO_pB_{suffix}_20220613_0300.fits"

        if SAVE_COMBINED_PB_FITS:
            # (A) トモグラフィ入力用：必須ヘッダ(RSUN_OBS等)を整形して保存
            pb_tomo_path = pb_fits_path  # 既存の変数名をそのまま使うならこれでOK
            save_combined_pb_fits_for_tomography(
                pb_image=final_pb,
                lasco_fits_path=filename_lasco,
                kcor_fits_path=filename_mk4,   # ここはあなたのK-COR(or Mk4)ヘッダのFITS
                out_fits_path=pb_tomo_path,
            )
            print(f"Saved combined pB FITS (tomography input) to {pb_tomo_path}")

            # (B) 任意：QA/比較用に「素の合成pB」を別名で保存（上書きしない）
            pb_plain_path = pb_tomo_path.replace(".fits", "_plain.fits")
            write_2d_fits_like(
                hdr_lasco, final_pb, pb_plain_path,
                bunit="Bsun",
                comment="Combined pB map (MK4 inner + LASCO outer) on LASCO grid (plain header)"
            )
            print(f"Saved combined pB FITS (plain) to {pb_plain_path}")

        # 2) 2D密度（QA用。SSCトモグラフィの入力は基本pBだが、比較用に残す）
        if OUTPUT_QUANTITY.lower() == "ne":
            out_map = combined_density
            out_unit = "cm^-3"
            tag = "ne"
        else:
            out_map = ne_to_fpe(combined_density, harmonic=FPE_HARMONIC, out_unit=FPE_UNIT)
            out_unit = FPE_UNIT
            tag = f"fpe_{FPE_UNIT.lower()}_h{FPE_HARMONIC}"

        ne_fits_path = os.path.join(out_dir, f"ne_Kcor_LASCO_{suffix}_20220613_0300.fits")

    export_density_csv(combined_density, r_map_lasco, params_lasco, csv_path, include_all=False)
    print(f"Saved combined CSV to {csv_path}")

    plt.show()

