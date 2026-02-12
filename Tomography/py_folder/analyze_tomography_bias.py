#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_tomography_bias.py

目的
----
STEREO-A(COR1) と Earth(K-COR+LASCO) の2視点で行っている pB inversion/tomography が
"左に偏る" ように見える原因を、(A) forward projection による整合性評価 と
(B) Carrington longitude の符号規約の違い(反転)テスト により判定するための診断スクリプト。

本スクリプトは *新規* の解析用ファイルです。
既存の main_tomo_gcs_pfss.py や main_regularized_tomography.py は変更しません。

出力
----
- case_nominal: 現状(ヘッダの CRLN_OBS/CRLT_OBS をそのまま使用)
- case_flip_lon: Carrington lon を (360 - lon) に反転して再構成

各ケースについて
- 画像ごとの y_obs vs y_pred の散布図 (png)
- 残差マップ (png)
- 残差統計 (標準出力)
- 3D voxel の重心(iso密度以上) の座標 (標準出力)

実行例
------
(推奨) Tomography/py_folder で実行:
  python3 analyze_tomography_bias.py

パスを明示する場合:
  python3 analyze_tomography_bias.py \
    --kcor_lasco ../Rawdata/pB_Kcor_LASCO_axi_20220613_0300.fits \
    --cor1a      ../Rawdata/COR1A_pb_pre_20220613_030100.fits

注意
----
- 本スクリプトは "2視点・単一時刻" の inversion を扱います。これは厳密には
  solar rotational tomography (SRT) のような多視点(多数時刻)問題よりも不良設定になりやすく、
  正則化やカーネル(Thomson散乱のPOS重み)によって形状が強く支配され得ます。
  (例: Quémerais & Lamy 2002, doi:10.1051/0004-6361:20021019;
       Aschwanden et al., 2011, doi:10.12942/lrsp-2011-5)
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

# 同じフォルダにある前提
from main_regularized_tomography import (
    SphericalGrid,
    RegularizedTomography,
    build_observation,
    build_rays_for_observation,
    infer_carrington_lonlat_deg,
    ne_cm3_from_fp_mhz,
)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0 or not np.isfinite(n):
        return v * np.nan
    return v / n


def _sun_to_observer_unit_vector_from_lonlat(lon_deg: float, lat_deg: float) -> np.ndarray:
    """Carrington lon/lat(度) -> 太陽中心から観測者方向の単位ベクトル (簡易球面→直交).

    注意: Carrington longitude の符号規約(西向きを正)が、数学的な経度(東向きを正)と
    逆になっている実装/データが混在し得ます。
    """
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    return _unit(np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]))


def _earth_triad_from_lonlat(lon_deg: float, lat_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """main_tomo_gcs_pfss.py の triad と同等の考え方で Earth基底を作る。
    x_hat: Sun->Earth
    z_hat: +solar north
    y_hat: east/right = z × x
    """
    x_hat = _sun_to_observer_unit_vector_from_lonlat(lon_deg, lat_deg)
    z_hat = _unit(np.array([0.0, 0.0, 1.0]))
    y_hat = _unit(np.cross(z_hat, x_hat))
    return x_hat, y_hat, z_hat


def _lon_transform_nominal(lon_deg: float) -> float:
    return float(lon_deg) % 360.0


def _lon_transform_flip(lon_deg: float) -> float:
    # 360-lon は、sin 成分の符号反転に相当（左右反転テスト）
    return float((360.0 - lon_deg) % 360.0)


def _apply_brightness_scale_like_main(tomo: RegularizedTomography, ne_raw: np.ndarray, y_obs: np.ndarray) -> tuple[np.ndarray, float]:
    """main_tomo_gcs_pfss.py と同様に、y_pred = A x のスカラー最小二乗で密度スケールを合わせる。"""
    y_pred = tomo.A_times(ne_raw)
    den = float(np.dot(y_pred, y_pred))
    if not np.isfinite(den) or den <= 0:
        return ne_raw, 1.0
    a = float(np.dot(y_obs, y_pred) / den)
    if (not np.isfinite(a)) or (a <= 0):
        a = 1.0
    return ne_raw * a, a


@dataclass(frozen=True)
class ObsSpec:
    path: Path
    tag: str
    limb_u: float
    r_use_min: float
    r_use_max: float


def _residual_stats(y_obs: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """weight 付き/なしの誤差統計を返す。"""
    eps = 1e-30
    r = y_obs - y_pred
    rel = r / (np.abs(y_obs) + eps)

    # weighted residual
    wr = w * r
    out = {
        "N": float(y_obs.size),
        "rmse": float(np.sqrt(np.mean(r**2))),
        "rmse_w": float(np.sqrt(np.mean(wr**2))),
        "median_rel": float(np.nanmedian(rel)),
        "mad_rel": float(np.nanmedian(np.abs(rel - np.nanmedian(rel)))),
        "corr": float(np.corrcoef(y_obs, y_pred)[0, 1]) if y_obs.size > 3 else np.nan,
    }
    return out


def run_case(
    case_name: str,
    obs_specs: list[ObsSpec],
    outdir: Path,
    *,
    out_n: int,
    r_min: float,
    r_max: float,
    nr: int,
    ntheta: int,
    nphi: int,
    ds: float,
    lam: float,
    wt_r: float,
    maxiter: int,
    tol: float,
    filt: str,
    apply_spatial_despike: bool,
    despike_nsig: float,
    despike_med: int,
    pb_floor: str | float,
    dpa_deg: float,
    hm: int,
    width_pix: int,
    q_low: float,
    lon_transform,
    iso_freq_mhz: float,
    harmonic: int,
) -> None:
    print("\n" + "=" * 80)
    print(f"[CASE] {case_name}")
    print("=" * 80)

    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Grid ----
    r_edges = np.linspace(r_min, r_max, nr + 1)
    th_edges = np.linspace(0.0, np.pi, ntheta + 1)
    ph_edges = np.linspace(0.0, 2.0 * np.pi, nphi + 1)
    grid = SphericalGrid(r_edges=r_edges, th_edges=th_edges, ph_edges=ph_edges)
    nvox = grid.nvox
    print(f"[INFO] Grid nvox={nvox} (nr={nr}, ntheta={ntheta}, nphi={nphi}), r=[{r_min},{r_max}] Rsun")

    # ---- Build observations & rays ----
    obs_list = []
    rays_list = []
    y_segments = []
    w_segments = []
    seg_meta = []

    for spec in obs_specs:
        hdr = fits.getheader(spec.path)
        lonlat = infer_carrington_lonlat_deg(hdr)
        if lonlat is None:
            lonlat = (0.0, 0.0)
            print(f"[WARN] {spec.tag}: header lacks Carrington lon/lat, using (0,0)")
        lon0, lat0 = lonlat
        lon1 = lon_transform(lon0)

        print(f"[INFO] {spec.tag}: header lon/lat={lon0:.6f}/{lat0:.6f} deg  -> used lon={lon1:.6f} deg")

        obs = build_observation(
            pb_fits=spec.path,
            out_n=out_n,
            r_use_min=spec.r_use_min,
            r_use_max=spec.r_use_max,
            limb_u=spec.limb_u,
            lonlat_override=(lon1, lat0),
            pb_floor=pb_floor,
            filt=filt,
            apply_spatial_despike=apply_spatial_despike,
            despike_nsig=despike_nsig,
            despike_med=despike_med,
            dpa_deg=dpa_deg,
            hm=hm,
            width_pix=width_pix,
            q_low=q_low,
        )

        # ここが重い場合は 6000 を 2000 などへ下げてください（偏り判定の初期診断なら十分です）
        obs = subsample_observation(obs, max_rays=6000, seed=0)

        rays = build_rays_for_observation(
            obs,
            grid=grid,
            ds_rsun=ds,
            r_min=r_min,
            r_max=r_max,
            limb_u=spec.limb_u,
        )

        # y_obs segment and weights
        y_seg = obs.pb.ravel()[obs.idx_map].astype(float)
        w_seg = obs.w.astype(float)

        obs_list.append(obs)
        rays_list.append(rays)
        y_segments.append(y_seg)
        w_segments.append(w_seg)
        seg_meta.append((spec.tag, y_seg.size))

    y_obs = np.concatenate(y_segments)
    w_all = np.concatenate(w_segments)
    print(f"[INFO] Total equations: {y_obs.size}")

    # ---- Solve tomography ----
    # main_regularized_tomography.apply_LTL は wt_r[:,None,None] を使うため、
    # wt_r は “nr長の配列” に正規化して渡す（スカラーのままだと後で落ちます）
    wt_r_arr = None
    if wt_r is not None:
        wt_r_arr = np.asarray(wt_r, dtype=float)
        if wt_r_arr.ndim == 0:
            wt_r_arr = np.full(grid.nr, float(wt_r_arr), dtype=float)
        else:
            wt_r_arr = wt_r_arr.ravel().astype(float)
            if wt_r_arr.size != grid.nr:
                raise ValueError(f"wt_r must have length nr={grid.nr}, got {wt_r_arr.size}")

    # 正しいAPI: observations=..., rays=..., solve()
    tomo = RegularizedTomography(grid=grid, observations=obs_list, rays=rays_list, lam=lam, wt_r=wt_r_arr)
    ne_raw, info = tomo.solve(y_obs, maxiter=maxiter, tol=tol, positivity=True)
    print(f"[INFO] CG info={info}  (0=converged, >0 not fully converged)")

    ne_scaled, scale = _apply_brightness_scale_like_main(tomo, ne_raw, y_obs)
    print(f"[INFO] brightness scale = {scale:.6e}")

    # ---- Forward projection and residuals ----
    y_pred = tomo.A_times(ne_scaled)

    # Global stats
    stats_global = _residual_stats(y_obs, y_pred, w_all)
    print("[STATS] Global")
    for k, v in stats_global.items():
        print(f"  {k:>12s} : {v:.6g}")

    # Per segment stats + diagnostic plots
    start = 0
    for i, (tag, n) in enumerate(seg_meta):
        y_o = y_obs[start:start + n]
        y_p = y_pred[start:start + n]
        w_s = w_all[start:start + n]
        start += n

        st = _residual_stats(y_o, y_p, w_s)
        print(f"[STATS] {tag}")
        for k, v in st.items():
            print(f"  {k:>12s} : {v:.6g}")

    # ---- Where is the iso-density mass center? ----
    iso_ne = ne_cm3_from_fp_mhz(iso_freq_mhz / harmonic)  # plasma freq = f/H
    # voxel 座標を flat 化して ne と同じ一次元に揃える
    xg, yg, zg = grid.voxel_centers_xyz()
    xyz = np.stack((xg, yg, zg), axis=-1).reshape(-1, 3)  # [nvox,3] in Rsun
    ne_flat = ne_scaled.ravel()
    m = np.isfinite(ne_flat) & (ne_flat >= iso_ne)

    if np.any(m):
        w = ne_flat[m]
        xyz_sel = xyz[m]
        com = np.sum(xyz_sel * w[:, None], axis=0) / np.sum(w)
        print(f"[INFO] iso_ne({iso_freq_mhz:.2f} MHz, H={harmonic}) = {iso_ne:.3e} cm^-3")
        print(f"[INFO] COM (ne>=iso_ne), Carrington xyz [Rsun] = ({com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f})")
    else:
        print(f"[WARN] No voxels above iso_ne={iso_ne:.3e} cm^-3")

    # ---- COM in Earth-triad coordinates (for intuitive left/right) ----
    earth_hdr = fits.getheader(obs_specs[0].path)
    e_lonlat = infer_carrington_lonlat_deg(earth_hdr)
    if e_lonlat is None:
        e_lonlat = (0.0, 0.0)
        print("[WARN] Earth HDR lacks Carrington lon/lat, using (0,0)")
    e_lon0, e_lat0 = e_lonlat
    e_lon1 = lon_transform(e_lon0)
    x_hat, y_hat, z_hat = _earth_triad_from_lonlat(e_lon1, e_lat0)
    if np.any(m):
        xyzm = xyz[m]
        yp = xyzm @ y_hat
        xp = xyzm @ x_hat
        zp = xyzm @ z_hat
        w = ne_flat[m]
        y_mean = float(np.sum(yp * w) / np.sum(w))
        x_mean = float(np.sum(xp * w) / np.sum(w))
        z_mean = float(np.sum(zp * w) / np.sum(w))
        print(f"[INFO] COM (Earth-triad x/y/z) [Rsun] = ({x_mean:.3f}, {y_mean:.3f}, {z_mean:.3f})")
        print("      (y>0 は図の 'East/right' 側に相当)")

    # ---- Save plots: per observation residual maps & scatter ----
    start = 0
    for i, obs in enumerate(obs_list):
        tag, n = seg_meta[i]
        y_o = y_obs[start:start + n]
        y_p = y_pred[start:start + n]
        start += n

        shape = obs.pb.shape
        map_o = np.full(shape, np.nan, dtype=float)
        map_p = np.full(shape, np.nan, dtype=float)
        idx = obs.idx_map
        map_o.ravel()[idx] = y_o
        map_p.ravel()[idx] = y_p
        map_r = (map_o - map_p) / (np.abs(map_o) + 1e-30)

        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        im1 = ax1.imshow(map_o, origin="lower")
        ax1.set_title(f"{case_name}: {tag} y_obs")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(1, 2, 2)
        im2 = ax2.imshow(map_r, origin="lower", vmin=-1, vmax=1)
        ax2.set_title(f"{case_name}: {tag} rel_residual")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(outdir / f"{case_name}_{i:02d}_{tag}_residual.png", dpi=200)
        # plt.close(fig)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(y_o, y_p, s=3)
        ax.set_xlabel("y_obs")
        ax.set_ylabel("y_pred")
        ax.set_title(f"{case_name}: {tag} obs vs pred")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls=":")
        fig.tight_layout()
        fig.savefig(outdir / f"{case_name}_{i:02d}_{tag}_scatter.png", dpi=200)
        # plt.close(fig)

    # ---- Save one summary histogram (where density sits along y in Earth-triad) ----
    if np.any(m):
        yp_all = (xyz[m] @ y_hat)
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(yp_all, bins=60)
        ax.set_xlabel("y (Earth-triad) [Rsun]")
        ax.set_ylabel("# voxels (ne>=iso_ne)")
        ax.set_title(f"{case_name}: distribution along East/right axis")
        fig.tight_layout()
        fig.savefig(outdir / f"{case_name}_y_distribution.png", dpi=200)
        # plt.close(fig)

from dataclasses import replace

def subsample_observation(obs, max_rays: int, seed: int = 0):
    """
    Observation の idx_map / w をランダムに間引いて ray 本数を減らす。
    w は「maskでTrueになった順（idx_mapの順）」と同じ並びなので “位置” で間引く。
    """
    n = int(obs.idx_map.size)
    if (max_rays is None) or (max_rays <= 0) or (n <= max_rays):
        return obs

    rng = np.random.default_rng(seed)
    pos = rng.choice(n, size=int(max_rays), replace=False)
    pos.sort()

    idx_map2 = obs.idx_map[pos]
    w2 = obs.w[pos]
    return replace(obs, idx_map=idx_map2, w=w2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kcor_lasco", type=str, default="../Rawdata/pB_Kcor_LASCO_axi_20220613_0300.fits")
    p.add_argument("--cor1a", type=str, default="../Rawdata/COR1A_pb_pre_20220613_030100.fits")
    p.add_argument("--outdir", type=str, default="../output/diagnostics")

    # ISO surface reference (Type II band etc.)
    p.add_argument("--iso_mhz", type=float, default=33.8)
    p.add_argument("--harmonic", type=int, default=2)

    args = p.parse_args()

    # ---- Match main_tomo_gcs_pfss.py defaults (必要ならここだけ調整) ----
    # 元データ 512x512 のため割り切れるよう 512 を使用
    OUT_N = 256
    R_MIN, R_MAX = 1.0, 6.0
    NR, NTH, NPH = 24, 36, 48
    DS = 0.1

    # regularization/solver
    LAM = 5e-3
    WT_R = 0.1
    MAXITER = 20
    TOL = 1e-6

    # preprocessing
    FILT = "ybk"
    APPLY_SPATIAL_DESPIKE = True
    DESPIKE_NSIG = 6.0
    DESPIKE_MED = 5
    PB_FLOOR = "auto"

    # ybk profile options (keep same as main)
    DPA_DEG = 2.0
    HM = 12
    WIDTH_PIX = 7
    Q_LOW = 0.05

    # limb darkening u values used in your run
    kcor_lasco = Path(args.kcor_lasco).expanduser().resolve()
    cor1a = Path(args.cor1a).expanduser().resolve()

    obs_specs = [
        ObsSpec(kcor_lasco, "KCOR-part", 0.453, 1.50, 2.20),
        ObsSpec(kcor_lasco, "LASCO-part", 0.614, 2.20, 5.50),
        ObsSpec(cor1a, "COR1A", 0.560, 1.50, 5.50),
    ]

    outdir = Path(args.outdir).expanduser().resolve()

    # ---- Run both cases ----
    run_case(
        "case_nominal",
        obs_specs,
        outdir,
        out_n=OUT_N,
        r_min=R_MIN,
        r_max=R_MAX,
        nr=NR,
        ntheta=NTH,
        nphi=NPH,
        ds=DS,
        lam=LAM,
        wt_r=WT_R,
        maxiter=MAXITER,
        tol=TOL,
        filt=FILT,
        apply_spatial_despike=APPLY_SPATIAL_DESPIKE,
        despike_nsig=DESPIKE_NSIG,
        despike_med=DESPIKE_MED,
        pb_floor=PB_FLOOR,
        dpa_deg=DPA_DEG,
        hm=HM,
        width_pix=WIDTH_PIX,
        q_low=Q_LOW,
        lon_transform=_lon_transform_nominal,
        iso_freq_mhz=args.iso_mhz,
        harmonic=args.harmonic,
    )

    run_case(
        "case_flip_lon",
        obs_specs,
        outdir,
        out_n=OUT_N,
        r_min=R_MIN,
        r_max=R_MAX,
        nr=NR,
        ntheta=NTH,
        nphi=NPH,
        ds=DS,
        lam=LAM,
        wt_r=WT_R,
        maxiter=MAXITER,
        tol=TOL,
        filt=FILT,
        apply_spatial_despike=APPLY_SPATIAL_DESPIKE,
        despike_nsig=DESPIKE_NSIG,
        despike_med=DESPIKE_MED,
        pb_floor=PB_FLOOR,
        dpa_deg=DPA_DEG,
        hm=HM,
        width_pix=WIDTH_PIX,
        q_low=Q_LOW,
        lon_transform=_lon_transform_flip,
        iso_freq_mhz=args.iso_mhz,
        harmonic=args.harmonic,
    )

    print("\n[OK] Diagnostics saved to:")
    print(f"  {outdir}")


if __name__ == "__main__":
    main()
