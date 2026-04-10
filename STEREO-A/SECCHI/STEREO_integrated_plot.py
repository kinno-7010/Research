#!/usr/bin/env python3
"""
STEREO-A/SECCHI/EUVI と STEREO-A/SECCHI/COR1 の統合差分画像を作成するスクリプト

設計方針
--------
- EUVI 195 Å running-difference を内側背景として使用
- COR1 TBr difference を外側オーバーレイとして使用
- 基準グリッドは COR1 diff map
- EUVI diff は COR1 グリッドへ補間
- 半径境界で統合して 1 枚の図として描画

前提
----
同じディレクトリに以下が存在すること
- cor1_diff_plot.py
- euvi_diff_plot.py

使い方例
--------
python3 stereo_integrated_euvi_cor1.py
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple
import warnings
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.ndimage import map_coordinates
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import sunpy.map

sys.path.append(r"/home/kinno-7010/Research_code/STEREO-A/SECCHI/EUVI/py_folder")
sys.path.append(r"/home/kinno-7010/Research_code/STEREO-A/SECCHI/COR1/py_folder")
from cor1_diff_plot import (
    RAWDATA_DIR as COR1_RAWDATA_DIR,
    select_nearest_cor1_fits_path,
    load_cor1_tbr_sequence,
    get_solar_radius_arcsec_and_pixel,
)
from euvi_diff_plot import (
    BASE_DATA_DIR as EUVI_BASE_DATA_DIR,
    collect_euvi_files_in_range,
    find_nearest_euvi_file,
    load_map as load_euvi_map,
)



OUT_DIR = Path(
    "/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/integrated_output"
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_input_time(target_time: str) -> datetime:
    candidate_formats = [
        "%Y%m%d_%H%M%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in candidate_formats:
        try:
            return datetime.strptime(target_time, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"target_time の形式が不正です: {target_time}. "
        "使用可能形式: YYYYMMDD_HHMMSS / YYYY-MM-DDTHH:MM:SS / YYYY-MM-DD HH:MM:SS / YYYY-MM-DD HH:MM"
    )


def robust_symmetric_limits(data: np.ndarray, low_q: float = 10.0, high_q: float = 90.0,
                            fallback: float = 1.0) -> Tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -fallback, fallback

    lo, hi = np.nanpercentile(finite, [low_q, high_q])
    vmax = max(abs(lo), abs(hi))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(np.abs(finite))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = fallback
    return -float(vmax), float(vmax)


def robust_positive_limits(data: np.ndarray, low_q: float = 1.0, high_q: float = 99.5,
                           fallback: Tuple[float, float] = (0.0, 1.0)) -> Tuple[float, float]:
    finite = data[np.isfinite(data) & (data > 0)]
    if finite.size == 0:
        return fallback

    lo, hi = np.nanpercentile(finite, [low_q, high_q])
    if not np.isfinite(lo):
        lo = fallback[0]
    if not np.isfinite(hi) or hi <= lo:
        hi = np.nanmax(finite)
    if not np.isfinite(hi) or hi <= lo:
        return fallback
    return float(lo), float(hi)


def get_params(m: sunpy.map.GenericMap) -> dict:
    """
    Map の中心・スケール情報を返す。
    中心は CRPIX 直読ではなく、WCS上の太陽中心 (0,0 arcsec) を使う。
    """
    px_per_rsun = m.rsun_obs.to_value(u.arcsec) / abs(m.scale.axis1.to_value(u.arcsec / u.pix))
    if not np.isfinite(px_per_rsun) or px_per_rsun <= 0:
        raise ValueError("px_per_rsun が計算できませんでした。")

    sun_center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=m.coordinate_frame)
    sun_center_pix = m.world_to_pixel(sun_center)
    cx = float(sun_center_pix.x.value)
    cy = float(sun_center_pix.y.value)

    return {
        "nx": m.data.shape[1],
        "ny": m.data.shape[0],
        "cx": cx,
        "cy": cy,
        "px_per_rsun": float(px_per_rsun),
    }
    
    
def calculate_r_map(params: dict) -> np.ndarray:
    ny, nx = params["ny"], params["nx"]
    y_idx, x_idx = np.indices((ny, nx))
    x_norm = (x_idx - params["cx"]) / params["px_per_rsun"]
    y_norm = (y_idx - params["cy"]) / params["px_per_rsun"]
    return np.sqrt(x_norm ** 2 + y_norm ** 2)


def resample_data_to_reference_grid(source_data: np.ndarray, source_params: dict,
                                    ref_params: dict, cval=np.nan) -> np.ndarray:
    ny, nx = ref_params["ny"], ref_params["nx"]
    y_idx, x_idx = np.indices((ny, nx))

    x_norm = (x_idx - ref_params["cx"]) / ref_params["px_per_rsun"]
    y_norm = (y_idx - ref_params["cy"]) / ref_params["px_per_rsun"]

    coords_in_source = np.vstack([
        (y_norm * source_params["px_per_rsun"] + source_params["cy"]).ravel(),
        (x_norm * source_params["px_per_rsun"] + source_params["cx"]).ravel(),
    ])

    interp = map_coordinates(
        source_data,
        coords_in_source,
        order=1,
        mode="constant",
        cval=cval,
    )
    return interp.reshape((ny, nx))


def build_cor1_diff_map(target_time: str,
                        base_minutes_before: int = 10,
                        rawdata_dir: str = COR1_RAWDATA_DIR,
                        background_dir=None,
                        subtract_tbr_background: bool = True) -> Tuple[sunpy.map.GenericMap, datetime, datetime]:
    target_fits_path, target_nearest_time, _ = select_nearest_cor1_fits_path(
        target_time,
        rawdata_dir=rawdata_dir,
    )

    base_requested = target_nearest_time - timedelta(minutes=base_minutes_before)
    base_request_str = base_requested.strftime("%Y%m%d_%H%M%S")
    base_fits_path, base_nearest_time, _ = select_nearest_cor1_fits_path(
        base_request_str,
        rawdata_dir=rawdata_dir,
    )

    base_data, base_header = load_cor1_tbr_sequence(
        base_fits_path,
        rawdata_dir=rawdata_dir,
        background_dir=background_dir,
        subtract_tbr_background=subtract_tbr_background,
    )
    target_data, target_header = load_cor1_tbr_sequence(
        target_fits_path,
        rawdata_dir=rawdata_dir,
        background_dir=background_dir,
        subtract_tbr_background=subtract_tbr_background,
    )

    if base_data.shape != target_data.shape:
        min_shape = tuple(min(b, t) for b, t in zip(base_data.shape, target_data.shape))
        base_data = base_data[:min_shape[0], :min_shape[1]]
        target_data = target_data[:min_shape[0], :min_shape[1]]

    diff_data = target_data.astype(np.float32) - base_data.astype(np.float32)
    diff_data = np.where(diff_data < 0, np.nan, diff_data)
    
    diff_header = target_header.copy()
    rsun_arcsec, _ = get_solar_radius_arcsec_and_pixel(diff_header)
    diff_header["RSUN_OBS"] = rsun_arcsec
    if "RSUN_REF" not in diff_header:
        diff_header["RSUN_REF"] = 6.957e8
    diff_header["BUNIT"] = "DN/s"
    diff_header.add_history("DIFFERENCE IMAGE: Target TBr - Base TBr")
    diff_header.add_history(f"Base anchor file: {Path(base_fits_path).name}")
    diff_header.add_history(f"Target anchor file: {Path(target_fits_path).name}")

    diff_map = sunpy.map.Map((diff_data, diff_header))
    diff_map = diff_map.rotate(order=1, missing=np.nan, clip=False)
    return diff_map, target_nearest_time, base_nearest_time


def build_euvi_diff_map(target_time: str,
                        wavelength_angstrom: int = 195,
                        dt_minutes: int = 10,
                        search_window_minutes: int | None = None,
                        max_time_error_seconds: int = 300) -> Tuple[sunpy.map.GenericMap, datetime, datetime]:
    target_dt = parse_input_time(target_time)
    if search_window_minutes is None:
        search_window_minutes = dt_minutes + 2

    t_prev_req = target_dt - timedelta(minutes=dt_minutes)
    t_cur_req = target_dt
    start_utc = t_prev_req - timedelta(minutes=search_window_minutes)
    end_utc = t_cur_req + timedelta(minutes=search_window_minutes)

    files = collect_euvi_files_in_range(
        start_utc=start_utc,
        end_utc=end_utc,
        wavelength_angstrom=wavelength_angstrom,
        step_minutes=None,
    )
    tol = timedelta(seconds=max_time_error_seconds)
    prev_file = find_nearest_euvi_file(files, t_prev_req, max_abs_diff=tol)
    cur_file = find_nearest_euvi_file(files, t_cur_req, max_abs_diff=tol)

    prev_map = load_euvi_map(prev_file.path, reference_map=None, rotate_north_up=True)
    cur_map = load_euvi_map(cur_file.path, reference_map=prev_map, rotate_north_up=True)

    if prev_map.data.shape != cur_map.data.shape:
        raise RuntimeError(
            f"EUVI shape mismatch after alignment: prev={prev_map.data.shape}, cur={cur_map.data.shape}"
        )

    diff_data = cur_map.data.astype(np.float32) - prev_map.data.astype(np.float32)
    diff_map = sunpy.map.Map((diff_data, prev_map.meta.copy()))
    return diff_map, prev_file.obs_time, cur_file.obs_time

def build_common_reference_map(
    template_map: sunpy.map.GenericMap,
    outer_rsun: float = 4.0,
) -> sunpy.map.GenericMap:
    """
    EUVI の pixel scale を保ったまま、outer_rsun まで含む共通キャンバスを作る。
    中心は WCS 上の太陽中心 (0,0 arcsec)。
    """
    px_scale_x = float(template_map.scale.axis1.to_value(u.arcsec / u.pix))
    px_scale_y = float(template_map.scale.axis2.to_value(u.arcsec / u.pix))
    rsun_arcsec = float(template_map.rsun_obs.to_value(u.arcsec))

    half_width_arcsec = outer_rsun * rsun_arcsec
    half_nx = int(np.ceil(half_width_arcsec / abs(px_scale_x)))
    half_ny = int(np.ceil(half_width_arcsec / abs(px_scale_y)))

    nx = 2 * half_nx + 1
    ny = 2 * half_ny + 1

    data = np.full((ny, nx), np.nan, dtype=np.float32)
    meta = template_map.meta.copy()

    meta["NAXIS1"] = nx
    meta["NAXIS2"] = ny
    meta["CRPIX1"] = half_nx + 1.0
    meta["CRPIX2"] = half_ny + 1.0
    meta["CRVAL1"] = 0.0
    meta["CRVAL2"] = 0.0
    meta["CDELT1"] = px_scale_x
    meta["CDELT2"] = px_scale_y

    # 回転はここでは持たせない
    meta["PC1_1"] = 1.0
    meta["PC1_2"] = 0.0
    meta["PC2_1"] = 0.0
    meta["PC2_2"] = 1.0
    if "CROTA" in meta:
        meta["CROTA"] = 0.0
    if "CROTA1" in meta:
        meta["CROTA1"] = 0.0
    if "CROTA2" in meta:
        meta["CROTA2"] = 0.0

    return sunpy.map.Map((data, meta))

def create_integrated_stereo_image(
    ax,
    target_time: str,
    cor1_base_minutes_before: int = 10,
    euvi_dt_minutes: int = 10,
    euvi_outer_rsun: float = 1.30,
    cor1_outer_rsun: float = 4.0,
    cor1_alpha: float = 0.75,
    transition_width_rsun: float = 0.10,
):
    """
    EUVI の解像度を保ちつつ、COR1 の外側視野まで描ける統合図を作る。
    """
    cor1_diff_map, cor1_target_time, cor1_base_time = build_cor1_diff_map(
        target_time,
        base_minutes_before=cor1_base_minutes_before,
    )
    euvi_diff_map, euvi_prev_time, euvi_cur_time = build_euvi_diff_map(
        target_time,
        wavelength_angstrom=195,
        dt_minutes=euvi_dt_minutes,
    )

    # -------------------------------------------------
    # 共通キャンバス:
    # EUVI の pixel scale を使い、COR1 outer_rsun まで入るサイズを新規作成
    # -------------------------------------------------
    common_map = build_common_reference_map(
        euvi_diff_map,
        outer_rsun=cor1_outer_rsun,
    )

    p_common = get_params(common_map)
    p_cor1 = get_params(cor1_diff_map)
    p_euvi = get_params(euvi_diff_map)
    r_map = calculate_r_map(p_common)

    # 共通キャンバスへ補間
    euvi_on_common = resample_data_to_reference_grid(
        euvi_diff_map.data.astype(np.float32),
        p_euvi,
        p_common,
        cval=np.nan,
    )
    cor1_on_common = resample_data_to_reference_grid(
        cor1_diff_map.data.astype(np.float32),
        p_cor1,
        p_common,
        cval=np.nan,
    )

    # ----------------------------
    # EUVI 背景
    # 元の euvi_diff_plot.py の gray + TwoSlopeNorm
    # ----------------------------
    euvi_background = euvi_on_common.copy()
    euvi_background[r_map > euvi_outer_rsun] = np.nan

    finite_euvi = euvi_background[np.isfinite(euvi_background)]
    if finite_euvi.size > 0:
        euvi_vmax = np.nanpercentile(np.abs(finite_euvi), 95.0)
        if not np.isfinite(euvi_vmax) or euvi_vmax <= 0:
            euvi_vmax = np.nanmax(np.abs(finite_euvi))
        if not np.isfinite(euvi_vmax) or euvi_vmax <= 0:
            euvi_vmax = 1.0
    else:
        euvi_vmax = 1.0

    euvi_norm = TwoSlopeNorm(vmin=-euvi_vmax, vcenter=0.0, vmax=euvi_vmax)
    euvi_cmap = plt.get_cmap("gray").copy()
    euvi_cmap.set_bad(alpha=0.0)

    # ----------------------------
    # COR1 オーバーレイ
    # 元の cor1_diff_plot.py の RdBu_r + TwoSlopeNorm
    # ----------------------------
    cor1_overlay = cor1_on_common.copy()
    cor1_overlay[r_map > cor1_outer_rsun] = np.nan
    cor1_overlay[r_map < max(1.0, euvi_outer_rsun - transition_width_rsun)] = np.nan

    finite_cor1 = cor1_overlay[np.isfinite(cor1_overlay)]
    if finite_cor1.size > 0:
        cor1_vmax = np.percentile(np.abs(finite_cor1), 90.0)
        if not np.isfinite(cor1_vmax) or cor1_vmax <= 0:
            cor1_vmax = np.nanmax(np.abs(finite_cor1))
        if not np.isfinite(cor1_vmax) or cor1_vmax <= 0:
            cor1_vmax = 1.0
    else:
        cor1_vmax = 1.0

    # # 固定下限を入れたいなら弱める
    # if cor1_vmax < 1.0:
    #     cor1_vmax = 1.0

    cor1_norm = TwoSlopeNorm(vmin=-cor1_vmax, vcenter=0.0, vmax=cor1_vmax)
    cor1_cmap = plt.get_cmap("RdBu_r").copy()
    cor1_cmap.set_bad(alpha=0.0)

    # COR1 を外側で徐々に出す
    alpha_map = np.zeros_like(cor1_overlay, dtype=np.float32)
    valid = np.isfinite(cor1_overlay)
    if np.any(valid):
        scaled = np.abs(cor1_overlay[valid]) / cor1_vmax
        scaled = np.clip(scaled, 0.0, 1.0)
        alpha_map[valid] = (scaled ** 0.6) * cor1_alpha

    r0 = max(1.0, euvi_outer_rsun - transition_width_rsun)
    r1 = euvi_outer_rsun + transition_width_rsun
    ramp = np.clip((r_map - r0) / (r1 - r0), 0.0, 1.0)
    alpha_map *= ramp

    # -------------------------------------------------
    # 描画
    # -------------------------------------------------
    if hasattr(ax, "figure") and ax.figure is not None:
        fig = ax.figure
        subplot_spec = ax.get_subplotspec()
        ax.remove()
        ax = fig.add_subplot(subplot_spec, projection=common_map.wcs)
    else:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection=common_map.wcs)

    im_euvi = ax.imshow(
        euvi_background,
        origin="lower",
        cmap=euvi_cmap,
        norm=euvi_norm,
        interpolation="bilinear",
        zorder=0,
    )

    im_cor1 = ax.imshow(
        cor1_overlay,
        origin="lower",
        cmap=cor1_cmap,
        norm=cor1_norm,
        alpha=alpha_map,
        interpolation="bilinear",
        zorder=1,
    )

    # limb / grid は EUVI 元コード寄り
    try:
        common_map.draw_limb(axes=ax, color="red", linestyle="dashed", linewidth=1.0)
    except Exception:
        pass
    try:
        common_map.draw_grid(
            axes=ax,
            grid_spacing=15 * u.deg,
            color="red",
            linestyle="dotted",
            linewidth=0.8,
            alpha=0.7,
        )
    except Exception:
        pass

    # 円は world_to_pixel ベース
    rsun_arcsec = common_map.rsun_obs.to_value(u.arcsec)
    sun_center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=common_map.coordinate_frame)
    sun_center_pix = common_map.world_to_pixel(sun_center)
    sun_x_pix = float(sun_center_pix.x.value)
    sun_y_pix = float(sun_center_pix.y.value)

    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    for radius_factor in [1, 2, 3, 4]:
        r_arcsec = radius_factor * rsun_arcsec * u.arcsec
        circle_coord = SkyCoord(
            r_arcsec * np.cos(theta),
            r_arcsec * np.sin(theta),
            frame=common_map.coordinate_frame
        )
        circle_pix = common_map.world_to_pixel(circle_coord)

        ax.plot(
            circle_pix.x.value,
            circle_pix.y.value,
            color="black",
            linewidth=1.5,
            alpha=0.8,
            linestyle="--",
            zorder=3,
        )

    half_width_pix = cor1_outer_rsun * p_common["px_per_rsun"]
    ax.set_xlim(sun_x_pix - half_width_pix, sun_x_pix + half_width_pix)
    ax.set_ylim(sun_y_pix - half_width_pix, sun_y_pix + half_width_pix)

    ax.set_title(
        "STEREO-A/SECCHI/EUVI 195 Å + COR1 Integrated Difference\n"
        f"EUVI: {euvi_cur_time.strftime('%H:%M:%S')} - {euvi_prev_time.strftime('%H:%M:%S')} | "
        f"COR1: {cor1_target_time.strftime('%H:%M:%S')} - {cor1_base_time.strftime('%H:%M:%S')}",
        fontsize=14,
    )
    ax.set_xlabel("Solar X [arcsec]")
    ax.set_ylabel("Solar Y [arcsec]")

    # カラーバー
    # cbar1 = fig.colorbar(im_euvi, ax=ax, shrink=0.82, pad=0.03)
    # cbar1.set_label("EUVI diff", rotation=270, labelpad=12)

    # cbar2 = fig.colorbar(im_cor1, ax=ax, shrink=0.82, pad=0.10)
    # cbar2.set_label("COR1 diff (DN/s)", rotation=270, labelpad=12)

    return ax, cor1_diff_map, euvi_diff_map, im_euvi, im_cor1, euvi_cur_time


def save_integrated_stereo_image(
    target_time: str,
    cor1_base_minutes_before: int = 10,
    euvi_dt_minutes: int = 10,
    euvi_outer_rsun: float = 1.30,
    cor1_outer_rsun: float = 4.0,
):
    ensure_dir(OUT_DIR)

    fig, ax = plt.subplots(figsize=(10, 10))

    result = create_integrated_stereo_image(
        ax,
        target_time=target_time,
        cor1_base_minutes_before=cor1_base_minutes_before,
        euvi_dt_minutes=euvi_dt_minutes,
        euvi_outer_rsun=euvi_outer_rsun,
        cor1_outer_rsun=cor1_outer_rsun,
    )

    if result is None:
        raise RuntimeError("統合画像の生成に失敗しました。")

    safe_name = result[5].replace(":", "").replace(" ", "_").replace("T", "_")
    out_png = OUT_DIR / f"stereo_integrated_{safe_name}.png"

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    # plt.show()
    print(f"[DONE] saved: {out_png}")
    return out_png

def main():
    import datetime
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # 2022-06-13 01:00:00 から 2022-06-13 04:00:00 まで1分刻みでリスト作成
    start_time = datetime.datetime(2022, 6, 13, 1, 0, 0)
    end_time = datetime.datetime(2022, 6, 13, 4, 0, 0)
    delta = datetime.timedelta(minutes=1)
    target_time_list = []
    cur_time = start_time
    while cur_time <= end_time:
        target_time_list.append(cur_time.strftime("%Y-%m-%d %H:%M:%S"))
        cur_time += delta

    for target_time in target_time_list:
        save_integrated_stereo_image(
            target_time=target_time,
            cor1_base_minutes_before=10,
            euvi_dt_minutes=10,
            euvi_outer_rsun=1.30,
            cor1_outer_rsun=4.0,
        )


if __name__ == "__main__":
    main()
