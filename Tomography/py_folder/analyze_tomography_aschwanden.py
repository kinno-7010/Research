#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_tomography_aschwanden.py

目的
- 同時刻2視点(pB)の voxel tomography が「左/裏側」に出る原因を、
  (1) 観測者幾何の取り違え
  (2) 2視点の不良設定性 + 正則化/重みの支配
  (3) データセット間の較正スケール不整合
  のどれが支配的か切り分ける。

特徴
- Earth の座標系(=triad)は固定したまま、COR1 側だけ lon反転などを試せる
- 重みモード: const / ybk_inv / ybk_sqrt / ybk_inv_capped / sigma_rel
- 観測ごとのゲイン(スケール因子)を反復推定してスケール不整合を吸収
- 自由度低下(Aschwanden 2011的): nphi/ntheta を小さめにして安定成分に限定

参考
- Aschwanden et al., 2011, doi: 10.12942/lrsp-2011-5
- Quémerais & Lamy, 2002, doi: 10.1051/0004-6361:20021019
- Kramar et al., 2009, doi: 10.1007/s11207-009-9401-2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits

from main_regularized_tomography import (
    SphericalGrid,
    RegularizedTomography,
    build_observation,
    build_rays_for_observation,
    infer_carrington_lonlat_deg,
    ybk_profile_fft,
    ne_cm3_from_fp_mhz,
)
import main_regularized_tomography as mrt
print("[DEBUG] main_regularized_tomography imported from:", mrt.__file__)

# ---------------------------
# Utilities
# ---------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0 or not np.isfinite(n):
        return v * np.nan
    return v / n

def _z_from_lonlat_math(lon_deg: float, lat_deg: float) -> np.ndarray:
    """数学的経度(+lon が +y 方向)として球面→直交。"""
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    return _unit(np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], float))

def _safe_out_n(orig_n: int, desired_out_n: int) -> int:
    """
    block_reduce_mean の制約（orig_n % out_n == 0）を満たす out_n を返す。
    原則：desired_out_n 以下で最大の割り切れる値を選ぶ（= downsample を維持）。
    """
    n = int(orig_n)
    d = int(desired_out_n)

    if n <= 0:
        return max(1, d)
    if d >= n:
        return n
    if d <= 1:
        return 1
    if n % d == 0:
        return d

    # d 以下で最大の約数を探す（512なら 256,128,64,... が候補になる）
    # O(d) だが d はせいぜい 256 程度なので十分軽い
    for cand in range(d - 1, 0, -1):
        if n % cand == 0:
            return cand
    return 1


def _earth_triad_from_lonlat_math(lon_deg: float, lat_deg: float):
    """
    Earth triad:
      x_hat: Sun -> Earth (観測者方向)
      z_hat: solar north (Carrington +z)
      y_hat: z_hat x x_hat  (右手系)
    注意: この y_hat は、一般に「solar west」側が + になりやすい。
    """
    x_hat = _z_from_lonlat_math(lon_deg, lat_deg)
    z_hat = _unit(np.array([0.0, 0.0, 1.0]))
    y_hat = _unit(np.cross(z_hat, x_hat))
    z_hat = _unit(np.cross(x_hat, y_hat))
    return x_hat, y_hat, z_hat

def _flip_lon_360(lon_deg: float) -> float:
    return (360.0 - lon_deg) % 360.0

def _sep_angle_deg(lon1, lat1, lon2, lat2) -> float:
    a = _z_from_lonlat_math(lon1, lat1)
    b = _z_from_lonlat_math(lon2, lat2)
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(c)))

# ---------------------------
# Weighting
def estimate_radial_bg_vec(
    obs,
    *,
    r_fit_min: float,
    r_fit_max: float,
    dpa_deg: float = 3.0,
    hm: int = 3,
    width_pix: int = 0,
    q_low: float = 10.0,
    nr: int = 240,
):
    """
    観測pBから方位平均の放射状背景 ybk(r) を推定し、
    obs.idx_map に対応する 1D ベクトル bg_vec を返す。

    NaN/inf を含む ybk が返ってきた場合でも、有限点で補間して NaN を極力排除する。
    """
    r_map = np.hypot(obs.x, obs.y)
    r_vec = r_map.ravel()[obs.idx_map]

    rgrid, ybk, pb_noise = ybk_profile_fft(
        pb=obs.pb, hdr=obs.hdr,
        rmin=float(r_fit_min), rmax=float(r_fit_max),
        dpa_deg=float(dpa_deg), nr=int(nr),
        hm=int(hm), width_pix=int(width_pix),
        q_low=float(q_low),
    )

    rgrid = np.asarray(rgrid, float)
    ybk = np.asarray(ybk, float)

    m = np.isfinite(rgrid) & np.isfinite(ybk)
    if np.sum(m) < 5:
        # bg 推定が壊れている（NaNだらけ等）→ bg=0 に退避
        return np.zeros_like(r_vec, dtype=float)

    rgrid_f = rgrid[m]
    ybk_f = ybk[m]

    # r_vec 側の NaN も除去して 0 埋め
    bg_vec = np.interp(np.nan_to_num(r_vec, nan=rgrid_f[0]), rgrid_f, ybk_f)
    bg_vec = np.where(np.isfinite(bg_vec), bg_vec, 0.0)
    return bg_vec

def attach_bg_vec_to_obs(obs, bg_vec: np.ndarray):
    """
    観測 obs に bg_vec を保持する（pb は変更しない）。
    """
    obs._bg_vec = np.array(bg_vec, dtype=float, copy=True)
    return
def solve_with_per_obs_gain_and_bg(
    tomo,
    y_obs: np.ndarray,
    groups,
    bg_concat: np.ndarray,
    *,
    n_iter: int = 4,
    maxiter: int = 500,
    tol: float = 2e-3,
    positivity: bool = True,
    verbose: bool = True,
):
    """
    観測ごとに y_k ≈ g_k * (A_k x) + b_k * bg_k を仮定して反復推定する。

    手順:
      1) (g,b) を仮定して y_adj = (y - b*bg)/g で x を解く
      2) p = A x
      3) 各観測kで [g_k, b_k] を重み付き最小二乗で更新（2変数線形回帰）
    """
    y_obs = np.asarray(y_obs, float)
    bg_concat = np.asarray(bg_concat, float)

    g = np.ones(len(groups), dtype=float)
    b = np.ones(len(groups), dtype=float)
    x = None
    info = None

    W = tomo.W  # concatenated weights

    # 初期 b：x=0 と仮定して y ≈ b*bg を当てはめ
    for k, (s, e) in enumerate(groups):
        wk = W[s:e]
        yk = y_obs[s:e]
        bk = bg_concat[s:e]
        m = np.isfinite(yk) & np.isfinite(bk) & np.isfinite(wk)
        if np.sum(m) < 10:
            b[k] = 0.0
            continue
        w2 = (wk[m] * wk[m])
        num = float(np.sum(w2 * yk[m] * bk[m]))
        den = float(np.sum(w2 * bk[m] * bk[m]))
        b[k] = (num / den) if (den > 0 and np.isfinite(num) and np.isfinite(den)) else 0.0

    for it in range(int(n_iter)):
        # y_adj = (y - b*bg)/g
        y_adj = y_obs.copy()
        for k, (s, e) in enumerate(groups):
            gg = g[k] if (np.isfinite(g[k]) and g[k] != 0) else 1.0
            bb = b[k] if np.isfinite(b[k]) else 0.0
            y_adj[s:e] = (y_obs[s:e] - bb * bg_concat[s:e]) / gg

        x, info = tomo.solve(y_adj, maxiter=int(maxiter), tol=float(tol), positivity=bool(positivity))
        p = tomo.A_times(x)

        # update (g,b) per observation via weighted LS:
        # minimize ||W ( y - g p - b bg )||^2
        for k, (s, e) in enumerate(groups):
            wk = W[s:e]
            yk = y_obs[s:e]
            pk = p[s:e]
            bk = bg_concat[s:e]

            m = np.isfinite(wk) & np.isfinite(yk) & np.isfinite(pk) & np.isfinite(bk)
            if np.sum(m) < 10:
                g[k] = 1.0
                b[k] = 0.0
                continue

            w2 = wk[m] * wk[m]
            yy = yk[m]
            pp = pk[m]
            bbv = bk[m]

            M11 = float(np.sum(w2 * pp * pp))
            M12 = float(np.sum(w2 * pp * bbv))
            M22 = float(np.sum(w2 * bbv * bbv))
            r1 = float(np.sum(w2 * pp * yy))
            r2 = float(np.sum(w2 * bbv * yy))

            det = M11 * M22 - M12 * M12
            if det > 0 and np.isfinite(det):
                gk = (r1 * M22 - r2 * M12) / det
                bk2 = (M11 * r2 - M12 * r1) / det
            else:
                # 退避：gのみ更新、bは維持
                gk = (r1 / M11) if (M11 > 0 and np.isfinite(M11) and np.isfinite(r1)) else 1.0
                bk2 = b[k]

            # 物理的に g,b は非負を期待 → クリップ（不安定化防止）
            g[k] = float(max(1e-6, gk)) if np.isfinite(gk) else 1.0
            b[k] = float(max(0.0, bk2)) if np.isfinite(bk2) else 0.0

        if verbose:
            gs = "  ".join([f"g[{k}]={gk:.4g}, b[{k}]={bk:.4g}" for k, (gk, bk) in enumerate(zip(g, b))])
            print(f"[INFO] gain+bg-iter {it+1}/{n_iter}: {gs}")

    # final prediction y_pred = g*p + b*bg
    p = tomo.A_times(x)
    y_pred = p.copy()
    for k, (s, e) in enumerate(groups):
        y_pred[s:e] = g[k] * p[s:e] + b[k] * bg_concat[s:e]

    return x, info, g, b, p, y_pred


def apply_radial_bg_subtraction_inplace(obs, bg_vec: np.ndarray):
    """
    obs.pb (2D) から、idx_map の点に対して bg_vec を引く（in-place）。
    bg_vec は idx_map と同順の 1D ベクトル。
    """
    pb_flat = obs.pb.ravel().astype(float, copy=True)
    idx = obs.idx_map
    pb_flat[idx] = pb_flat[idx] - bg_vec
    obs.pb = pb_flat.reshape(obs.pb.shape)

    # 後で "total相関" を計算するために保持
    obs._bg_vec = np.array(bg_vec, dtype=float, copy=True)
    return

def balance_observation_weight_alpha(obs_list, alpha: float = 0.5):
    """
    sum(w^2) を揃えるスケーリングを、強さ alpha で部分適用する。
      alpha=0: 何もしない
      alpha=1: 既存equalizeと同じ（完全に揃える）
    """
    a = float(alpha)
    if a <= 0:
        return obs_list
    if a > 1:
        a = 1.0

    sums = [float(np.sum(o.w**2)) for o in obs_list]
    valid = [s for s in sums if s > 0 and np.isfinite(s)]
    target = float(np.median(valid)) if len(valid) else 1.0

    for obs, s in zip(obs_list, sums):
        if s > 0 and np.isfinite(s):
            # 完全equalizeは sqrt(target/s)。
            # 部分適用は (target/s)^(alpha/2)。
            obs.w *= (target / s) ** (0.5 * a)
    return obs_list

def _parse_float_list(s: str, default: list[float]) -> list[float]:
    if s is None or str(s).strip() == "":
        return list(default)
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out if len(out) else list(default)


def autotune_hyperparams_and_solve(
    *,
    grid,
    obs_list,
    rays_list,
    y_obs: np.ndarray,
    tags: list[str],
    bg_concat: np.ndarray | None,
    wt_r: np.ndarray,
    lam_grid: list[float],
    alpha_grid: list[float],
    positivity_grid: list[bool],
    maxiter_scan: int = 250,
    tol_scan: float = 3e-3,
    maxiter_final: int = 2000,
    tol_final: float = 1e-3,
):
    """
    (lam, alpha, positivity) を粗探索し、平均相関（total）最大の組で最終解を計算して返す。
    """
    # 元の重みを保存
    w0 = [o.w.copy() for o in obs_list]

    # groups は tomo 初期化後に得る必要があるので、各試行で作る
    def _corr(a, b) -> float:
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        if np.sum(m) < 10:
            return np.nan
        aa = a[m] - np.mean(a[m])
        bb = b[m] - np.mean(b[m])
        den = np.sqrt(np.sum(aa * aa) * np.sum(bb * bb))
        return float(np.sum(aa * bb) / den) if den > 0 else np.nan

    def _score(y, yp, groups):
        # total相関（bgを足し戻して評価）
        if bg_concat is not None:
            yT = y + bg_concat
            pT = yp + bg_concat
        else:
            yT = y
            pT = yp
        cs = []
        for (s, e) in groups:
            cs.append(_corr(yT[s:e], pT[s:e]))
        cs = [c for c in cs if np.isfinite(c)]
        return float(np.mean(cs)) if len(cs) else -np.inf

    best = None  # (score, lam, alpha, pos, info, y_pred, groups)
    for lam in lam_grid:
        for alpha in alpha_grid:
            for pos in positivity_grid:
                # 重みを元に戻してから alpha バランス
                for o, ww in zip(obs_list, w0):
                    o.w = ww.copy()
                balance_observation_weight_alpha(obs_list, alpha=float(alpha))

                tomo = RegularizedTomography(
                    grid=grid,
                    observations=obs_list,
                    rays=rays_list,
                    lam=float(lam),
                    wt_r=wt_r,
                )
                groups = [(slc.start, slc.stop) for slc in tomo.slices]

                x, info = tomo.solve(y_obs, maxiter=int(maxiter_scan), tol=float(tol_scan), positivity=bool(pos))
                y_pred = tomo.A_times(x)

                sc = _score(y_obs, y_pred, groups)

                # 収束しない場合はペナルティ（ただし「相関最大」が目的なので軽め）
                if info != 0:
                    sc -= 0.05

                if (best is None) or (sc > best[0]):
                    best = (sc, float(lam), float(alpha), bool(pos), int(info), y_pred.copy(), groups)

    if best is None:
        raise RuntimeError("autotune failed: no valid trial produced a score.")

    _, lam_best, alpha_best, pos_best, info_best, _, _ = best
    print(f"[TUNE] best lam={lam_best:g}, alpha={alpha_best:g}, positivity={pos_best}, scan_info={info_best}")

    # 最終解（best設定で本計算）
    for o, ww in zip(obs_list, w0):
        o.w = ww.copy()
    balance_observation_weight_alpha(obs_list, alpha=float(alpha_best))

    tomo = RegularizedTomography(
        grid=grid,
        observations=obs_list,
        rays=rays_list,
        lam=float(lam_best),
        wt_r=wt_r,
    )
    groups = [(slc.start, slc.stop) for slc in tomo.slices]

    x, info = tomo.solve(y_obs, maxiter=int(maxiter_final), tol=float(tol_final), positivity=bool(pos_best))
    y_pred = tomo.A_times(x)

    return x, info, y_pred, groups, lam_best, alpha_best, pos_best

# ---------------------------



def apply_weight_mode(
    obs,
    mode: str,
    *,
    r_use_min: float,
    r_use_max: float,
    ybk_cut: float = 1e-12,
    cap_pctl: float = 10.0,
    rel_sigma: float = 0.05,
    abs_sigma_floor: float | None = None,
    normalize_median: bool = True,
    # 追加: 重みの暴走抑制（中央値に対する比でクリップ）
    w_clip_ratio: tuple[float, float] = (0.2, 5.0),
):
    """
    main_regularized_tomography.Observation に整合した重み付け。
    obs.w は「使用ピクセル(idx_map)に対応する 1D ベクトル」。

    追加仕様:
      - w_clip_ratio=(lo,hi) を指定すると、median(w[w>0]) を基準に [lo,hi] にクリップ。
        少数視点トモグラフィでは、極端重みが条件数を悪化させ相関を落としやすいので、既定で有効化。
    """
    mode = (mode or "").strip().lower()
    if mode == "ybk_inv_capped":
        mode = "ybk_inv"

    y_vec = obs.pb.ravel()[obs.idx_map]
    n = y_vec.size
    if n == 0:
        obs.w = np.array([], dtype=float)
        return

    if mode in ("", "ybk_inv", "ybk_sqrt"):
        r_map = np.hypot(obs.x, obs.y)  # Rsun
        r_vec = r_map.ravel()[obs.idx_map]

        rgrid, ybk, pb_noise = ybk_profile_fft(
            pb=obs.pb, hdr=obs.hdr,
            rmin=float(r_use_min), rmax=float(r_use_max),
            dpa_deg=3.0, nr=240, hm=3, width_pix=0, q_low=10.0
        )

        ybk_vec = np.interp(r_vec, rgrid, ybk)

        finite = np.isfinite(ybk_vec) & (ybk_vec > 0)
        if np.any(finite):
            floor = float(np.nanpercentile(ybk_vec[finite], cap_pctl))
            if (not np.isfinite(floor)) or (floor <= 0):
                floor = float(np.nanmin(ybk_vec[finite]))
        else:
            floor = float(ybk_cut)

        pb_noise = float(pb_noise) if np.isfinite(pb_noise) and pb_noise > 0 else 0.0
        floor = max(float(ybk_cut), pb_noise, floor if np.isfinite(floor) and floor > 0 else float(ybk_cut))

        ybk_vec = np.where(np.isfinite(ybk_vec) & (ybk_vec > 0), ybk_vec, floor)

        if mode in ("", "ybk_inv"):
            w = 1.0 / np.maximum(ybk_vec, floor)
        else:
            w = 1.0 / np.sqrt(np.maximum(ybk_vec, floor))

    elif mode == "const":
        w = np.ones(n, dtype=float)

    elif mode == "sigma_rel":
        # 相対誤差モデル: sigma = max(rel_sigma*|y|, abs_sigma_floor)
        if abs_sigma_floor is None:
            # 既存の「median*1e-3」は床が小さすぎて w が暴れやすい。
            # ここでは lower percentile を使って、床を現実的に大きくする。
            ay = np.abs(y_vec[np.isfinite(y_vec)])
            if ay.size == 0:
                abs_sigma_floor = 1e-30
            else:
                floor_y = float(np.nanpercentile(ay, 20.0))  # 20%点
                abs_sigma_floor = max(1e-30, 0.05 * floor_y)  # 5% を床に

        sigma = np.maximum(float(rel_sigma) * np.abs(y_vec), float(abs_sigma_floor))
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, float(abs_sigma_floor))
        w = 1.0 / sigma

    else:
        raise ValueError(f"Unknown weight_mode='{mode}'. Use const/ybk_inv/ybk_sqrt/ybk_inv_capped/sigma_rel.")

    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

    # 正規化（中央値=1）
    if normalize_median:
        ww = w[w > 0]
        if ww.size > 0:
            m = float(np.median(ww))
            if np.isfinite(m) and m > 0:
                w = w / m

    # ---- 追加: クリップで条件数悪化を抑制 ----
    lo, hi = w_clip_ratio
    if lo is not None and hi is not None:
        lo = float(lo); hi = float(hi)
        if (lo > 0) and (hi > lo):
            w = np.where(w > 0, np.clip(w, lo, hi), 0.0)

    obs.w = w
    print(f"[WEIGHT] {mode}: w_min={float(np.nanmin(w)):.3g}, w_max={float(np.nanmax(w)):.3g}, w_mean={float(np.nanmean(w)):.3g}")
    return

def equalize_observation_weight(obs_list):
    """
    各 obs の “重み総量” を揃える（あるデータセットが解を支配しないようにする）。
    ここでは sum(w^2) を揃えるスケーリングを採用。
    """
    sums = []
    for obs in obs_list:
        sums.append(float(np.sum(obs.w**2)))
    target = float(np.median([s for s in sums if s > 0])) if any(s > 0 for s in sums) else 1.0
    for obs, s in zip(obs_list, sums):
        if s > 0:
            obs.w *= np.sqrt(target / s)
    return obs_list

# ---------------------------
# Gain (per-observation scale) estimation
# ---------------------------

def solve_with_per_obs_gain(
    tomo: RegularizedTomography,
    y_obs: np.ndarray,
    groups: list[tuple[int, int]],
    *,
    n_iter: int = 4,
    maxiter: int = 50,
    tol: float = 1e-4,
    positivity: bool = True,
    verbose: bool = True,
):
    """
    観測ごとに未知の乗算ゲイン g_k を持つとして反復推定する。
      y_k ≈ g_k * (A_k x)

    手順:
      1) g を仮定して y_adj = y/g で x を解く
      2) p=A x を計算
      3) 各観測kで g_k = argmin ||W(y - g p)||^2 の解で更新
    """
    g = np.ones(len(groups), dtype=float)
    x = None
    info = None

    W = tomo.W  # concatenated weights (vector)

    for it in range(int(n_iter)):
        # y_adj = y / g_k
        y_adj = y_obs.copy()
        for k, (s, e) in enumerate(groups):
            gg = g[k]
            if not np.isfinite(gg) or gg == 0:
                gg = 1.0
            y_adj[s:e] = y_obs[s:e] / gg

        x, info = tomo.solve(y_adj, maxiter=int(maxiter), tol=float(tol), positivity=bool(positivity))
        p = tomo.A_times(x)

        # update gains
        for k, (s, e) in enumerate(groups):
            wk = W[s:e]
            yk = y_obs[s:e]
            pk = p[s:e]

            w2 = wk * wk
            num = float(np.sum(w2 * yk * pk))
            den = float(np.sum(w2 * pk * pk))

            if den > 0 and np.isfinite(num) and np.isfinite(den):
                g[k] = num / den
            else:
                g[k] = 1.0

        if verbose:
            gs = "  ".join([f"g[{k}]={gk:.4g}" for k, gk in enumerate(g)])
            print(f"[INFO] gain-iter {it+1}/{n_iter}: {gs}")

    # final prediction in original scale: y_pred = g_k * p
    p = tomo.A_times(x)
    y_pred = p.copy()
    for k, (s, e) in enumerate(groups):
        y_pred[s:e] = g[k] * p[s:e]

    return x, info, g, y_pred

# ---------------------------
# Diagnostics
# ---------------------------

def corr(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 3:
        return np.nan
    aa = a[m] - np.mean(a[m])
    bb = b[m] - np.mean(b[m])
    den = np.sqrt(np.sum(aa**2) * np.sum(bb**2))
    if den == 0:
        return np.nan
    return float(np.sum(aa * bb) / den)

def save_scatter(y, ypred, outpng: Path, title: str):
    fig = plt.figure(figsize=(5,5), dpi=150)
    ax = fig.add_subplot(111)
    m = np.isfinite(y) & np.isfinite(ypred)
    ax.scatter(y[m], ypred[m], s=2, alpha=0.3)
    ax.set_xlabel("Observed pB")
    ax.set_ylabel("Forward pB")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpng)
    plt.close(fig)

# ---------------------------
# Main pipeline
# ---------------------------

@dataclass
class ObsSpec:
    tag: str
    path: Path
    rmin: float
    rmax: float
    limb_u: float

def build_obs_and_rays(
    spec: ObsSpec,
    grid: SphericalGrid,
    *,
    # 追加: 観測で使う r 範囲を明示指定（グリッド整合のため）
    r_use_min: float,
    r_use_max: float,
    lon_override: float | None = None,
    lat_override: float | None = None,
    n_pix: int = 256,
    filt: bool = False,
    apply_spatial_despike: bool = True,
    despike_nsig: float = 6.0,
    despike_med: int = 3,
    pb_floor: float | str = 1e-13,
    dpa_deg: float = 3.0,
    hm: int = 3,
    width_pix: int = 0,
    q_low: float = 10.0,
    ds_rsun: float = 0.05,
    r_min: float = 1.0,
    r_max: float = 6.0,
):
    """
    ObsSpec + Grid から Observation と RayBundle を構築する。
    返り値:
      obs, rays, bounds, (lon_h,lat_h), (lon_used,lat_used)

    重要:
      - r_use_max は “観測ピクセルの選別半径”。
      - r_max は “ray 積分の打ち切り半径”。
      - 今回の相関低下の主因は、r_use_max > r_max（観測は外側まで含むのにモデルは含まない）になること。
        ここで整合を保証する。
    """
    hdr = fits.getheader(spec.path)
    lon_h, lat_h = infer_carrington_lonlat_deg(hdr)

    lon_used = float(lon_override) if lon_override is not None else lon_h
    lat_used = float(lat_override) if lat_override is not None else lat_h

    # ---- グリッド整合のガード ----
    r_use_min = float(r_use_min)
    r_use_max = float(r_use_max)
    r_min = float(r_min)
    r_max = float(r_max)

    if r_use_min < r_min:
        r_use_min = r_min
    if r_use_max > r_max:
        # 観測がモデル外に出ないように強制クリップ
        r_use_max = r_max

    if not (r_use_max > r_use_min):
        raise ValueError(
            f"[{spec.tag}] invalid r-range after clipping: r_use_min={r_use_min}, r_use_max={r_use_max}, r_model=[{r_min},{r_max}]"
        )

    lonlat_override = None
    if lon_used is not None and lat_used is not None:
        lonlat_override = (float(lon_used), float(lat_used))

    orig_n = int(hdr.get("NAXIS1", 0))
    out_n = _safe_out_n(orig_n, int(n_pix))
    if out_n != int(n_pix):
        print(f"[INFO] {spec.tag}: adjust out_n {int(n_pix)} -> {out_n} (orig_n={orig_n})")

    obs = build_observation(
        pb_fits=spec.path,
        out_n=int(out_n),
        r_use_min=float(spec.rmin),
        r_use_max=float(spec.rmax),
        limb_u=float(spec.limb_u),
        apply_spatial_despike=bool(apply_spatial_despike),
        filt=bool(filt),
        despike_nsig=float(despike_nsig),
        despike_med=int(despike_med),
        pb_floor=pb_floor,
        dpa_deg=float(dpa_deg),
        hm=int(hm),
        width_pix=int(width_pix),
        q_low=float(q_low),
        lonlat_override=lonlat_override,
    )
    rays = build_rays_for_observation(
        obs=obs,
        grid=grid,
        ds_rsun=float(ds_rsun),
        r_min=float(r_min),
        r_max=float(r_max),
        limb_u=float(spec.limb_u),
    )

    x_use = obs.x.ravel()[obs.idx_map]
    y_use = obs.y.ravel()[obs.idx_map]
    bounds = (float(np.nanmin(x_use)), float(np.nanmin(y_use)),
              float(np.nanmax(x_use)), float(np.nanmax(y_use)))

    return obs, rays, bounds, (lon_h, lat_h), (lon_used, lat_used)

def build_dataset(
    *,
    specs: list[ObsSpec],
    grid: SphericalGrid,
    lonE: float,
    latE: float,
    n_pix_map: dict[str, int],
    ds_rsun: float,
    r_use_cap: float,
    rmin_grid: float,
    rmax_grid: float,
    weight_mode: str,
    do_bg: bool,
    bg_fit_margin: float,
    verbose: bool = True,
):
    obs_list, rays_list, tags = [], [], []
    bg_vec_list = []

    for spec in specs:
        n_pix = int(n_pix_map.get(spec.tag, 256))

        r_use_min = max(float(spec.rmin), float(rmin_grid))
        r_use_max = min(float(spec.rmax), float(r_use_cap))

        obs, rays, bounds, (lh, bh), (lu, bu) = build_obs_and_rays(
            spec, grid,
            r_use_min=r_use_min,
            r_use_max=r_use_max,
            lon_override=lonE if spec.tag in ("KCOR-part", "LASCO-part") else None,
            lat_override=latE if spec.tag in ("KCOR-part", "LASCO-part") else None,
            n_pix=n_pix,
            filt=False,
            ds_rsun=float(ds_rsun),
            r_min=float(rmin_grid),
            r_max=float(rmax_grid),
        )

        apply_weight_mode(
            obs, weight_mode,
            r_use_min=float(r_use_min),
            r_use_max=float(r_use_max),
            normalize_median=True,
        )

        if do_bg:
            m = float(bg_fit_margin)
            rfit_min = float(r_use_min + m)
            rfit_max = float(r_use_max - m)
            if rfit_max <= rfit_min + 0.05:
                rfit_min, rfit_max = float(r_use_min), float(r_use_max)

            bg_vec = estimate_radial_bg_vec(obs, r_fit_min=rfit_min, r_fit_max=rfit_max)
            attach_bg_vec_to_obs(obs, bg_vec)
            bg_vec_list.append(bg_vec)
        else:
            bg_vec_list.append(np.zeros(obs.idx_map.size, dtype=float))

        obs_list.append(obs)
        rays_list.append(rays)
        tags.append(spec.tag)

        if verbose:
            print(f"[INFO] {spec.tag}: header lon/lat={lh:.6f}/{bh:.6f} deg  -> used lon/lat={lu:.6f}/{bu:.6f} deg")
            print(f"[INFO] {spec.tag}: r_use=[{r_use_min:.3f},{r_use_max:.3f}] Rsun (grid rmax={rmax_grid:.3f}, cap={r_use_cap:.3f})")
            print(f"[INFO] {spec.tag}: used-pixel bounds (xmin,ymin,xmax,ymax) [Rsun] = "
                  f"({bounds[0]:.3f},{bounds[1]:.3f},{bounds[2]:.3f},{bounds[3]:.3f})  n_pix={n_pix}")

    # ★ここが重要：y_obs は「元の pb」（bg を引かない）
    y_obs = np.concatenate([obs.pb.ravel()[obs.idx_map] for obs in obs_list], axis=0)
    bg_concat = np.concatenate(bg_vec_list, axis=0) if do_bg else None
    return obs_list, rays_list, tags, y_obs, bg_concat

def iso_ne_from_freq(freq_mhz: float, harmonic: int = 2) -> float:
    """
    観測周波数 freq_mhz [MHz] が plasma emission (harmonic=1 or 2) に対応すると仮定して
    等電子密度 iso_ne [cm^-3] を返す。

    f_obs = harmonic * f_pe
    f_pe [kHz] = 8.98 * sqrt(ne [cm^-3])

    => ne = ( (f_obs/harmonic)*1000 / 8.98 )^2
    """
    f = float(freq_mhz)
    H = int(harmonic)
    if H <= 0:
        raise ValueError("harmonic must be positive (1 or 2).")
    fpe_khz = (f / H) * 1.0e3  # MHz -> kHz
    ne = (fpe_khz / 8.98) ** 2
    return float(ne)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--nr", type=int, default=18)
    ap.add_argument("--ntheta", type=int, default=24)
    ap.add_argument("--nphi", type=int, default=32)
    ap.add_argument("--rmin", type=float, default=1.5)
    ap.add_argument("--rmax", type=float, default=4.0)

    ap.add_argument("--ds_rsun", type=float, default=0.05)

    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--wt_r", type=float, default=1.0)
    ap.add_argument("--maxiter", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--no_positivity", action="store_true")

    ap.add_argument("--gain_iter", type=int, default=0)

    ap.add_argument("--n_pix_kcor", type=int, default=256)
    ap.add_argument("--n_pix_lasco", type=int, default=256)
    ap.add_argument("--n_pix_cor1", type=int, default=256)

    ap.add_argument("--weight_mode", type=str, default="const",
                    help="const / ybk_inv / ybk_sqrt / sigma_rel")

    ap.add_argument("--obs_rmax_margin", type=float, default=0.15)

    # --- 背景分離 ---
    ap.add_argument("--no_bg_subtract", action="store_true")
    ap.add_argument("--bg_fit_margin", type=float, default=0.0)

    # --- autotune ---
    ap.add_argument("--no_autotune", action="store_true")

    # 既定は「軽い」方へ（40試行を避ける）
    ap.add_argument("--lam_grid", type=str, default="0.6,2.0,6.0")
    ap.add_argument("--alpha_grid", type=str, default="0.0,0.4,1.0")
    ap.add_argument("--scan_maxiter", type=int, default=60)
    ap.add_argument("--scan_tol", type=float, default=1e-2)

    # ★追加：tuneは小さく解く
    ap.add_argument("--tune_npix", type=int, default=96)
    ap.add_argument("--tune_ds_rsun", type=float, default=0.15)
    ap.add_argument("--tune_use_bg", action="store_true",
                    help="enable bg subtraction also in tuning stage (default: off for speed)")

    ap.add_argument("--iso_freq_mhz", type=float, default=33.8)
    ap.add_argument("--harmonic", type=int, default=2)

    args = ap.parse_args()

    # -------------------------
    # Grid
    # -------------------------
    r_edges = np.linspace(float(args.rmin), float(args.rmax), int(args.nr) + 1)
    th_edges = np.linspace(0.0, np.pi, int(args.ntheta) + 1)
    ph_edges = np.linspace(0.0, 2.0 * np.pi, int(args.nphi) + 1)
    grid = SphericalGrid(r_edges=r_edges, th_edges=th_edges, ph_edges=ph_edges)
    nvox = int(grid.nr * grid.nth * grid.nph)
    print(f"[INFO] Grid nvox={nvox} (nr={grid.nr}, ntheta={grid.nth}, nphi={grid.nph}), "
          f"r=[{args.rmin},{args.rmax}] Rsun")

    # -------------------------
    # Specs
    # -------------------------
    specs = [
        ObsSpec("KCOR-part", Path("../Rawdata/pB_Kcor_LASCO_axi_20220613_0300.fits"), 1.50, 2.20, 0.453),
        ObsSpec("LASCO-part", Path("../Rawdata/pB_Kcor_LASCO_axi_20220613_0300.fits"), 2.20, 5.50, 0.614),
        ObsSpec("COR1A",      Path("../Rawdata/COR1A_pb_pre_20220613_030100.fits"),    1.50, 5.50, 0.560),
    ]

    # Earth lon/lat
    hdrE = fits.getheader(specs[0].path)
    lonE_h, latE_h = infer_carrington_lonlat_deg(hdrE)
    lonE = lonE_h
    latE = latE_h
    print("[INFO] lon/lat (header -> used)")
    print(f"  Earth: lon={lonE_h:.6f}, lat={latE_h:.6f}  -> used lon={lonE:.6f}")

    hdrA = fits.getheader(specs[-1].path)
    lonA_h, latA_h = infer_carrington_lonlat_deg(hdrA)
    print(f"  COR1A: lon={lonA_h:.6f}, lat={latA_h:.6f}  -> used lon={lonA_h:.6f}")

    sep = _sep_angle_deg(lonE, latE, lonA_h, latA_h)
    print(f"[INFO] separation angle (used) = {sep:.3f} deg")

    x_hat, y_hat, z_hat = _earth_triad_from_lonlat_math(lonE, latE)
    print("[INFO] Earth triad definition:")
    print("  x_hat: Sun->Earth")
    print("  z_hat: solar north")
    print("  y_hat=z_hat×x_hat  (右手系; 典型的には +y が solar WEST 側になりやすい)")

    # 解析に使う最大半径（pixel選別）をグリッドに合わせる
    r_use_cap = float(args.rmax) - float(args.obs_rmax_margin)
    if r_use_cap <= float(args.rmin) + 0.05:
        r_use_cap = float(args.rmax)

    # -------------------------
    # Stage 1: tuning (small)
    # -------------------------
    lam_best = float(args.lam)
    alpha_best = 0.0
    pos_best = (not bool(args.no_positivity))

    if not bool(args.no_autotune):
        n_pix_map_tune = {
            "KCOR-part": int(args.tune_npix),
            "LASCO-part": int(args.tune_npix),
            "COR1A":      int(args.tune_npix),
        }

        do_bg_tune = bool(args.tune_use_bg) and (not bool(args.no_bg_subtract))
        obs_t, rays_t, tags_t, y_t, bg_t = build_dataset(
            specs=specs,
            grid=grid,
            lonE=lonE,
            latE=latE,
            n_pix_map=n_pix_map_tune,
            ds_rsun=float(args.tune_ds_rsun),
            r_use_cap=float(r_use_cap),
            rmin_grid=float(args.rmin),
            rmax_grid=float(args.rmax),
            weight_mode=str(args.weight_mode),
            do_bg=bool(do_bg_tune),
            bg_fit_margin=float(args.bg_fit_margin),
            verbose=False,  # tuneはログ最小限
        )

        wt_r = np.ones(int(grid.nr), dtype=float) * float(args.wt_r)

        lam_grid = _parse_float_list(args.lam_grid, [float(args.lam)])
        alpha_grid = _parse_float_list(args.alpha_grid, [0.0])
        positivity_grid = [not bool(args.no_positivity)]  # tuneは1種類で十分（ここが効く）

        x_t, info_t, ypred_t, groups_t, lam_best, alpha_best, pos_best = autotune_hyperparams_and_solve(
            grid=grid,
            obs_list=obs_t,
            rays_list=rays_t,
            y_obs=y_t,
            tags=tags_t,
            bg_concat=bg_t,
            wt_r=wt_r,
            lam_grid=lam_grid,
            alpha_grid=alpha_grid,
            positivity_grid=positivity_grid,
            maxiter_scan=int(args.scan_maxiter),
            tol_scan=float(args.scan_tol),
            maxiter_final=200,   # tune finalも軽く
            tol_final=2e-3,
        )
        print(f"[INFO] tune best: lam={lam_best:g}, alpha={alpha_best:g}, positivity={pos_best}")

    # -------------------------
    # Stage 2: final (full)
    # -------------------------
    n_pix_map_full = {
        "KCOR-part": int(args.n_pix_kcor),
        "LASCO-part": int(args.n_pix_lasco),
        "COR1A":      int(args.n_pix_cor1),
    }
    do_bg_full = (not bool(args.no_bg_subtract))

    obs_list, rays_list, tags, y_obs, bg_concat = build_dataset(
        specs=specs,
        grid=grid,
        lonE=lonE,
        latE=latE,
        n_pix_map=n_pix_map_full,
        ds_rsun=float(args.ds_rsun),
        r_use_cap=float(r_use_cap),
        rmin_grid=float(args.rmin),
        rmax_grid=float(args.rmax),
        weight_mode=str(args.weight_mode),
        do_bg=bool(do_bg_full),
        bg_fit_margin=float(args.bg_fit_margin),
        verbose=True,
    )

    # alpha を適用（tuneで決まった強さ）
    balance_observation_weight_alpha(obs_list, alpha=float(alpha_best))

    wt_r = np.ones(int(grid.nr), dtype=float) * float(args.wt_r)

    # --- solve ---
    tomo = RegularizedTomography(
        grid=grid,
        observations=obs_list,
        rays=rays_list,
        lam=float(lam_best if not bool(args.no_autotune) else args.lam),
        wt_r=wt_r,
    )
    groups = [(slc.start, slc.stop) for slc in tomo.slices]
    positivity = (not bool(args.no_positivity)) and bool(pos_best)

    if bg_concat is not None:
        # bg を「引く」のではなく「係数 b を同時推定」する
        x, info, g, b, p, y_pred = solve_with_per_obs_gain_and_bg(
            tomo, y_obs, groups, bg_concat,
            n_iter=4,
            maxiter=min(int(args.maxiter), 800),   # まずは軽めで
            tol=max(float(args.tol), 2e-3),        # 収束しない場合に備えて少し緩め
            positivity=bool(positivity),
            verbose=True,
        )
    else:
        if int(args.gain_iter) > 0:
            x, info, g, y_pred = solve_with_per_obs_gain(
                tomo, y_obs, groups,
                n_iter=int(args.gain_iter),
                maxiter=int(args.maxiter),
                tol=float(args.tol),
                positivity=bool(positivity),
                verbose=True,
            )
        else:
            x, info = tomo.solve(y_obs, maxiter=int(args.maxiter), tol=float(args.tol), positivity=bool(positivity))
            y_pred = tomo.A_times(x)

    print(f"[INFO] CG info={info}  (0=converged, >0 not fully converged)")

    # -------------------------
    # Stats
    # -------------------------
    def _corr(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        if np.sum(m) < 10:
            return np.nan
        aa = a[m] - np.mean(a[m])
        bb = b[m] - np.mean(b[m])
        den = np.sqrt(np.sum(aa * aa) * np.sum(bb * bb))
        return float(np.sum(aa * bb) / den) if den > 0 else np.nan

    print("[STATS] Per-observation corr")
    for k, (s, e) in enumerate(groups):
        c_tot = corr(y_obs[s:e], y_pred[s:e])

        if bg_concat is not None:
            # 構造相関：y - b*bg vs g*p（p は solve_with_per_obs_gain_and_bg が返す）
            yy = y_obs[s:e] - b[k] * bg_concat[s:e]
            pp = g[k] * p[s:e]
            c_str = corr(yy, pp)
            print(f"  {tags[k]:9s}: N={e-s:6d}  corr_total={c_tot:.6f}  corr_struct={c_str:.6f}")
        else:
            print(f"  {tags[k]:9s}: N={e-s:6d}  corr_total={c_tot:.6f}")

    iso_ne = iso_ne_from_freq(float(args.iso_freq_mhz), harmonic=int(args.harmonic))
    print(f"[INFO] iso_ne({args.iso_freq_mhz:.2f} MHz, H={args.harmonic}) = {iso_ne:.3e} cm^-3")

    xc, yc, zc = grid.voxel_centers_xyz()
    xyz = np.column_stack([xc.ravel(), yc.ravel(), zc.ravel()])
    mask = np.isfinite(x) & (x >= iso_ne)
    if np.any(mask):
        w = x[mask].astype(float)
        com = np.average(xyz[mask], axis=0, weights=w)
        print(f"[INFO] COM (ne>=iso_ne), Carrington xyz [Rsun] = ({com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f})")
        com_earth = np.array([np.dot(com, x_hat), np.dot(com, y_hat), np.dot(com, z_hat)])
        print(f"[INFO] COM Earth-triad x/y/z [Rsun] = ({com_earth[0]:.3f}, {com_earth[1]:.3f}, {com_earth[2]:.3f})")
    else:
        print("[WARN] No voxels above iso_ne; COM skipped.")

if __name__ == "__main__":
    main()
