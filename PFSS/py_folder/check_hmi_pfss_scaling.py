#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_hmi_pfss_scaling_extended.py

HMI → PFSS → 2D磁場マップ → POS サンプリングの各段階で
磁場スケーリングと典型値を確認するためのデバッグ用スクリプト。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import astropy.units as u
import sunpy.map  # type: ignore

# =========================
# パス設定（必要に応じて書き換えてください）
# =========================

# HMI 磁場ファイル
HMI_FILE = Path(
    "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/Rawdata/"
    "hmi.M_720s.20220613_030000_TAI.fits"
)

# あなたの Python モジュールの場所
HMI_PY_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/py_folder")
PB_PY_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research/SOHO/pB/py_folder")
PFSS_PY_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research/PFSS/py_folder")  # このスクリプトのあるフォルダ想定

for p in (HMI_PY_DIR, PB_PY_DIR, PFSS_PY_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

# あなたの既存コードを import
from hmi_analysis_wcs import read_hmi_quick  # type: ignore
from plot_hmi_pfss_overlay import prepare_hmi_for_pfss, compute_pfss_solution  # type: ignore
from magnetic_field_2D_map import (  # type: ignore
    resolve_pfss_axes,
    resolve_pfss_components,
    make_lasco_grid_and_final_image,
    sample_pfss_on_pos_grid,
)

# LASCO / Mk4 pB ファイル（Stage 3 用・必要なら書き換え）
F_LASCO = Path(
    "/mnt/d/wsl/home/kinno-7010/Research/SOHO/pB/"
    "C2-PB-20220613_0258.fts"
)
F_MK4 = Path(
    "/mnt/d/wsl/home/kinno-7010/Research/MK4_coronagraph/"
    "MK4_coronagraph_KCOR/pB/20220613_025810_kcor_l2.fts"
)

R_RANGES = {
    "mk4_inner": 1.1,
    "mk4_outer_lasco_inner": 2.2,
    "lasco_outer": 6.0,
}


# =========================
# ユーティリティ
# =========================

def summarize(arr: np.ndarray, label: str) -> None:
    """配列の統計を簡単に表示"""
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        print(f"[{label}] no finite data")
        return
    print(
        f"[{label}]  "
        f"min={arr.min():8.3f},  max={arr.max():8.3f},  "
        f"mean={arr.mean():8.3f},  std={arr.std():8.3f}"
    )


def find_radius_index(r_ax: np.ndarray, r_target: float) -> int:
    """r 軸から、target Rsun に最も近いインデックスを取得"""
    r_ax = np.asarray(r_ax, dtype=float)
    return int(np.argmin(np.abs(r_ax - r_target)))


# =========================
# Stage 0: 元 FITS の生データ
# =========================

def stage0_raw_hmi() -> None:
    """元 FITS の BSCALE, BZERO, datamin/max を確認"""
    print("\n=== Stage 0: raw HMI FITS ===")
    with fits.open(HMI_FILE) as hdul:
        # HMI lev1.5 は通常 HDU 1 に画像が入っている
        hdu = hdul[1]
        hdr = hdu.header
        raw = hdu.data.astype(float)

        bscale = hdr.get("BSCALE", 1.0)
        bzero = hdr.get("BZERO", 0.0)
        bunit = hdr.get("BUNIT", "UNKNOWN")

        print(f"BSCALE={bscale}, BZERO={bzero}, BUNIT={bunit}")
        summarize(raw, "raw integer data")
        summarize(raw * bscale + bzero, "raw * BSCALE + BZERO  [Gauss]")

        for k in ("DATAMIN", "DATAMAX", "DATARMS", "DATAMEAN"):
            if k in hdr:
                print(f"  header {k} = {hdr[k]:.3f}")


# =========================
# Stage 1: SunPy Map & read_hmi_quick
# =========================

def stage1_sunpy_map_and_read_hmi_quick() -> None:
    """sunpy.Map と read_hmi_quick の出力を比較"""
    print("\n=== Stage 1: sunpy.Map と read_hmi_quick ===")

    m = sunpy.map.Map(HMI_FILE)
    summarize(m.data, "sunpy.Map.data [Gauss?]")

    hmi_q = read_hmi_quick(str(HMI_FILE))
    data_q = hmi_q["data"]
    summarize(data_q, "read_hmi_quick['data']")

    if "sunpy_map" in hmi_q:
        summarize(hmi_q["sunpy_map"].data, "read_hmi_quick['sunpy_map'].data")


# =========================
# Stage 2: PFSS 入力 & 3D 格子上の |B|(r)
# =========================

def stage2_pfss_input_and_output():
    """
    PFSS 入力マップと、PFSS 解の 3D 格子上での Br, |B| を
    r=1.0, 1.5, 2.0 Rsun で評価する。
    """
    print("\n=== Stage 2: PFSS input (CEA) and PFSS 3D field ===")

    # PFSS 入力（CEA）を作成
    hmi_info = prepare_hmi_for_pfss(str(HMI_FILE))
    hmi_map_full = hmi_info["full_map"]
    summarize(hmi_map_full.data, "prepare_hmi_for_pfss: full_map.data [Gauss]")

    # PFSS を計算
    print("\nPFSS解を計算中...")
    rss = 2.5
    pfss_output = compute_pfss_solution(hmi_map_full, nrho=25, rss=rss)

    # 軸 & コンポーネントを取得
    Br, Bt, Bp = resolve_pfss_components(pfss_output)
    r_ax, th_ax, ph_ax = resolve_pfss_axes(pfss_output, fallback={"rss": rss})

    Br = np.asarray(Br)
    Bt = np.asarray(Bt)
    Bp = np.asarray(Bp)

    print(f"PFSS grid shapes: Br={Br.shape}, r_ax={r_ax.shape}, th_ax={th_ax.shape}, ph_ax={ph_ax.shape}")
    summarize(Br, "PFSS Br (all radii) [Gauss]")

    # |B| を作成
    B_abs = np.sqrt(Br**2 + Bt**2 + Bp**2)
    summarize(B_abs, "PFSS |B| (all radii) [Gauss]")

    # 代表的な半径でスライス
    for r_target in (1.0, 1.5, 2.0):
        idx = find_radius_index(r_ax, r_target)
        r_val = float(r_ax[idx])
        Br_slice = Br[idx, :, :]
        Babs_slice = B_abs[idx, :, :]

        print(f"\n--- PFSS field at r ≃ {r_target:.1f} Rs (index {idx}, actual {r_val:.3f} Rs) ---")
        summarize(Br_slice, f"Br(r≈{r_val:.2f} Rs) [Gauss]")
        summarize(Babs_slice, f"|B|(r≈{r_val:.2f} Rs) [Gauss]")

    # r=1 Rs に限定した Br の統計（従来と比較用）
    idx1 = find_radius_index(r_ax, 1.0)
    Br_surface = Br[idx1, :, :]
    summarize(Br_surface, "PFSS Br at r≈1 Rs [Gauss]")

    # 戻り値として解を返しておく（Stage 3 で再利用）
    return pfss_output, hmi_map_full, r_ax, th_ax, ph_ax


# =========================
# Stage 3: POS グリッド上での B (LASCO グリッド)
# =========================

def stage3_pos_sampling(pfss_output, hmi_map_full) -> None:
    """
    LASCO グリッド上で PFSS をサンプリングし、
    Br, |B| の典型値を r≃2Rs などで確認する。
    """
    print("\n=== Stage 3: POS sampling on LASCO grid ===")

    if not F_LASCO.exists():
        print(f"[WARN] LASCO pB ファイルが見つかりません: {F_LASCO}")
        print("       Stage 3 をスキップします。パスを書き換えてください。")
        return

    filename_mk4 = str(F_MK4) if F_MK4.exists() else None

    # LASCO グリッドと r_map, theta_map を取得
    final_image, r_map, theta_map, params_lasco = make_lasco_grid_and_final_image(
        str(F_LASCO),
        filename_mk4,
        R_RANGES,
    )

    print(f"LASCO grid shape: r_map={r_map.shape}, theta_map={theta_map.shape}")
    summarize(r_map, "r_map [Rsun]")
    summarize(theta_map, "theta_map [rad]")

    # B0（観測者の緯度）を HMI メタから取得
    b0_deg = float(hmi_map_full.meta.get("crlt_obs", 0.0))

    # POS（東西 limb のどちらを見るか）は、とりあえず 90°（東側）とする
    lon_pos = 90.0

    # Br の POS サンプリング
    Br_pos = sample_pfss_on_pos_grid(
        pfss_output,
        r_map_rsun=r_map,
        theta_map=theta_map,
        pos_longitude_deg=lon_pos,
        component="Br",
        b0_deg=b0_deg,
        pfss_fallback={"rss": 2.5},
    )
    summarize(Br_pos, f"POS Br (lon={lon_pos} deg) [Gauss]")

    # |B| の POS サンプリング
    Babs_pos = sample_pfss_on_pos_grid(
        pfss_output,
        r_map_rsun=r_map,
        theta_map=theta_map,
        pos_longitude_deg=lon_pos,
        component="abs",   # "Br" 以外なら |B| を返す実装前提
        b0_deg=b0_deg,
        pfss_fallback={"rss": 2.5},
    )
    summarize(Babs_pos, f"POS |B| (lon={lon_pos} deg) [Gauss]")

    # r≃2Rs 付近だけマスクして統計
    mask_2R = (r_map > 1.9) & (r_map < 2.1) & np.isfinite(Babs_pos)
    summarize(Babs_pos[mask_2R], "POS |B| at ~2 Rs [Gauss]")
    summarize(Br_pos[mask_2R], "POS Br at ~2 Rs [Gauss]")


# =========================
# メイン
# =========================

def main() -> None:
    print(f"対象 HMI ファイル: {HMI_FILE}")
    stage0_raw_hmi()
    stage1_sunpy_map_and_read_hmi_quick()
    pfss_output, hmi_map_full, r_ax, th_ax, ph_ax = stage2_pfss_input_and_output()
    stage3_pos_sampling(pfss_output, hmi_map_full)


if __name__ == "__main__":
    main()
