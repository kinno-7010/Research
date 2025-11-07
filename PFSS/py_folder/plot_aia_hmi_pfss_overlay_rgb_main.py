#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIA(171/193/211)のRGB合成を背景に、HMI由来のPFSS磁力線を重ね描画する単体スクリプト。
・引数は if __name__ == "__main__": で指定（コマンドライン引数は不要）
・AIAとHMIの空間分解能差は AIA→HMI への reproject で吸収し、描画上の太陽サイズを一致
・ユーザーの AIA Composite コードの正規化/描画スタイルを踏襲
・pfsspy が使えない環境では実行できません（導入: pip install pfsspy）

依存:
  pip install sunpy pfsspy reproject aiapy astropy numpy matplotlib
"""

from __future__ import annotations

import io
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst as HGS
from astropy.visualization import ImageNormalize
from astropy.visualization import PowerStretch

import sunpy.map
from sunpy.map import Map
from sunpy.coordinates import frames
from sunpy.coordinates import SphericalScreen
from pathlib import Path
# --- optional (aia_prep 相当) ---
try:
    from aiapy.calibrate import register as aia_register
    _HAS_AIAPY = True
except Exception:
    _HAS_AIAPY = False

# --- PFSS ---
try:
    import pfsspy
    import pfsspy.tracing as tracing
    _HAS_PFSSPY = True
except Exception:
    _HAS_PFSSPY = False


# =========================
# 基本I/Oヘルパー
# =========================
def read_hmi(hmi_file: str | Path) -> Map:
    m = Map(hmi_file)
    return m


def read_aia(aia_root: str | Path, dt, wavelength: str) -> Map:
    """
    ユーザーの命名規則:
      {AIA_ROOT}/{wavelength}/AIA{YYYYMMDD}_{HHMM}_{wwww}.fits
      例: .../171/AIA20220613_0300_0171.fits
    """
    aia_root = Path(aia_root)
    ww = wavelength.zfill(4)
    fname = f"AIA{dt.strftime('%Y%m%d')}_{dt.strftime('%H%M')}_{ww}.fits"
    fpath = aia_root / wavelength / fname
    m = Map(fpath)
    if _HAS_AIAPY:
        try:
            m = aia_register(m)  # plate scale/回転補正（aia_prep相当）
        except Exception:
            pass
    return m


def reproject_aia_to_hmi(aia_map: Map, hmi_map: Map) -> Map:
    """AIAをHMI WCS/shapeへreprojectして、太陽見かけ半径の差をなくす"""
    target_wcs = hmi_map.wcs
    target_shape = hmi_map.data.shape
    return aia_map.reproject_to(target_wcs, shape_out=target_shape)


# =========================
# AIAの正規化（ユーザーの流儀を踏襲）
# =========================
def normalize_logish(data: np.ndarray, pmin=1.0, pmax=99.0, power=0.5) -> np.ndarray:
    """パーセンタイルでクリップ後、PowerStretch(γ)で0-1に近似正規化"""
    arr = data.astype(float)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)
    vmin = np.nanpercentile(valid, pmin)
    vmax = np.nanpercentile(valid, pmax)
    if vmax <= vmin:
        return np.zeros_like(arr)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=PowerStretch(power), clip=True)
    out = norm(arr)
    return np.clip(out, 0, 1)


def make_rgb_from_three(a211: Map, a193: Map, a171: Map) -> np.ndarray:
    """R=211, G=193, B=171 を 0-1 合成RGBへ"""
    R = normalize_logish(a211.data, pmin=1.0, pmax=99.0, power=0.5)
    G = normalize_logish(a193.data, pmin=1.0, pmax=99.0, power=0.5)
    B = normalize_logish(a171.data, pmin=1.0, pmax=99.0, power=0.5)
    return np.dstack([R, G, B])


# =========================
# PFSS（フォールバック実装：HMI→CEA→PFSS）
# =========================
def compute_pfss_solution(hmi_map: Map, nrho: int = 50, rss: float = 2.5):
    if not _HAS_PFSSPY:
        raise RuntimeError("pfsspy が見つかりません。`pip install pfsspy` を実施してください。")

    # NaN埋め＋コピー
    data = hmi_map.data.astype(float).copy()
    data[~np.isfinite(data)] = 0.0
    hmi_clean = Map(data, hmi_map.meta)

    # CEA投影 360x180（lon×lat, 1°/px）へ変換（全太陽向け粗いグリッド）
    from astropy.coordinates import SkyCoord
    import sunpy.map
    ref = SkyCoord(0*u.deg, 0*u.deg,
                   frame='heliographic_stonyhurst',
                   obstime=hmi_map.date,
                   rsun=hmi_map.rsun_meters)
    header_cea = sunpy.map.make_fitswcs_header(
        data=(180, 360),  # (ny, nx) = (lat, lon)
        coordinate=ref,
        scale=u.Quantity([3600, 3600 * (2 / np.pi)], u.arcsec / u.pixel),
        projection_code='CEA'
    )
    # CEAの緯度の向き合わせ（pfsspy慣習に合わせるための簡易設定）
    hmi_cea = hmi_clean.reproject_to(header_cea)
    hmi_cea.data[~np.isfinite(hmi_cea.data)] = 0.0

    pfss_in = pfsspy.Input(hmi_cea, nrho, rss)
    pfss_out = pfsspy.pfss(pfss_in)
    return pfss_out


def define_seed_points(hmi_map: Map, x_pix_range, y_pix_range,
                       n_x=18, n_y=18, strong_only=True, thr=200.0):
    x0, x1 = x_pix_range
    y0, y1 = y_pix_range

    if strong_only:
        sub = hmi_map.data[y0:y1, x0:x1]
        mask = np.abs(sub) > thr
        ys, xs = np.where(mask)
        if xs.size > 0:
            n_seeds = min(n_x*n_y, xs.size)
            idx = np.random.choice(xs.size, n_seeds, replace=False)
            xs, ys = xs[idx] + x0, ys[idx] + y0
        else:
            strong_only = False

    if not strong_only:
        xs = np.linspace(x0+10, x1-10, n_x)
        ys = np.linspace(y0+10, y1-10, n_y)
        xs, ys = np.meshgrid(xs, ys)
        xs, ys = xs.ravel(), ys.ravel()

    seeds = hmi_map.pixel_to_world(xs * u.pixel, ys * u.pixel)
    return seeds


def trace_field_lines(seeds, pfss_out):
    tracer = tracing.FortranTracer(max_steps=100000, step_size=0.01)
    return tracer.trace(seeds, pfss_out)


# =========================
# 可視化補助
# =========================
def compute_window_in_pixels(hmi_map: Map, cx=0*u.arcsec, cy=0*u.arcsec,
                             width_pix=1024, height_pix=1024):
    c0 = SkyCoord(cx, cy, frame=hmi_map.coordinate_frame)
    center_pix = hmi_map.world_to_pixel(c0)
    try:
        cx_pix = center_pix.x.value
        cy_pix = center_pix.y.value
    except AttributeError:
        cx_pix = center_pix[0].value
        cy_pix = center_pix[1].value
    half_w = width_pix / 2.0
    half_h = height_pix / 2.0
    x0 = int(np.floor(cx_pix - half_w))
    x1 = int(np.ceil(cx_pix + half_w))
    y0 = int(np.floor(cy_pix - half_h))
    y1 = int(np.ceil(cy_pix + half_h))
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(hmi_map.data.shape[1] - 1, x1)
    y1 = min(hmi_map.data.shape[0] - 1, y1)
    return (x0, x1), (y0, y1)


def print_rsun_check(hmi_map: Map, aia_r_map: Map):
    def rs_pix(m: Map):
        rs_arcsec = m.rsun_obs.to_value(u.arcsec)
        scale = m.scale
        try:
            comps = [scale.x, scale.y]
        except AttributeError:
            try:
                comps = [scale.axis1, scale.axis2]
            except AttributeError:
                comps = list(scale)
        pixscale = np.mean([abs(val.to_value(u.arcsec/u.pixel)) for val in comps])
        return rs_arcsec / pixscale
    hr, ar = rs_pix(hmi_map), rs_pix(aia_r_map)
    print(f"[半径チェック] HMI ≈ {hr:.2f} pix,  AIA(reproj) ≈ {ar:.2f} pix,  差={abs(hr-ar):.2f} pix")


def draw_field_lines(ax, hmi_map: Map, field_lines, rss: float):
    with SphericalScreen(hmi_map.observer_coordinate):
        for fl in field_lines:
            coords = fl.coords  # HGS 3D
            # open判定（終端半径がRss到達近傍）
            try:
                is_open = (coords.radius[-1].to_value(u.Rsun) >= (rss - 0.05))
            except Exception:
                is_open = False

            # Helioprojectiveへ変換（SphericalScreenでオフディスクも有効化）
            try:
                coords_hpc = coords.transform_to(hmi_map.coordinate_frame)
            except Exception:
                continue
            pixel_coords = hmi_map.world_to_pixel(coords_hpc)
            if hasattr(pixel_coords, 'x'):
                xs = np.array(pixel_coords.x.value)
                ys = np.array(pixel_coords.y.value)
            else:
                xs = np.array(pixel_coords[0].value)
                ys = np.array(pixel_coords[1].value)
            valid = np.isfinite(xs) & np.isfinite(ys)
            if not np.any(valid):
                continue
            xs_valid, ys_valid = xs[valid], ys[valid]

            # 始点極性で色分け
            try:
                start_idx = np.argmax(valid)
                x_idx = int(round(xs[start_idx]))
                y_idx = int(round(ys[start_idx]))
                if (0 <= y_idx < hmi_map.data.shape[0]) and (0 <= x_idx < hmi_map.data.shape[1]):
                    pol = hmi_map.data[y_idx, x_idx]
                else:
                    pol = 0
            except Exception:
                pol = 0

            color = 'black'
            if is_open:
                color = 'red' if pol > 0 else 'blue'
            ax.plot(xs_valid, ys_valid, color=color, linewidth=0.6, alpha=0.85)


# =========================
# メイン処理
# =========================
def run_pipeline(HMI_FILE, AIA_ROOT, TIME_OBJ,
                 RCH='211', GCH='193', BCH='171',
                 RSS=2.5, NRHO=50,
                 USE_STRONG=True, THRESH=200.0,
                 WIN_PIXELS=(1024, 1024),
                 X_RANGE_PIX=None, Y_RANGE_PIX=None,
                 RMAX=None, STEP_DR=0.01, BLEND_DR=0.6,
                 SAVE_PATH=None):
    # 1) 読み込み
    hmi = read_hmi(HMI_FILE)
    a211 = read_aia(AIA_ROOT, TIME_OBJ, RCH)  # R
    a193 = read_aia(AIA_ROOT, TIME_OBJ, GCH)  # G
    a171 = read_aia(AIA_ROOT, TIME_OBJ, BCH)  # B

    # 2) AIA -> HMI reproject（太陽サイズ合わせ）
    a211_r = reproject_aia_to_hmi(a211, hmi)
    a193_r = reproject_aia_to_hmi(a193, hmi)
    a171_r = reproject_aia_to_hmi(a171, hmi)
    print_rsun_check(hmi, a171_r)

    # 3) RGB合成（0-1）
    rgb = make_rgb_from_three(a211_r, a193_r, a171_r)

    # 4) PFSS解
    pfss_out = compute_pfss_solution(hmi, nrho=NRHO, rss=RSS)

    # 5) 表示窓とシード点
    if X_RANGE_PIX is not None or Y_RANGE_PIX is not None:
        if X_RANGE_PIX is None or Y_RANGE_PIX is None:
            raise ValueError("X_RANGE_PIX と Y_RANGE_PIX は両方とも指定してください。")
        if len(X_RANGE_PIX) != 2 or len(Y_RANGE_PIX) != 2:
            raise ValueError("X_RANGE_PIX, Y_RANGE_PIX は (min, max) の2要素で指定してください。")
        x0, x1 = sorted(int(v) for v in X_RANGE_PIX)
        y0, y1 = sorted(int(v) for v in Y_RANGE_PIX)
        x0 = max(0, min(x0, hmi.data.shape[1]-1))
        x1 = max(0, min(x1, hmi.data.shape[1]-1))
        y0 = max(0, min(y0, hmi.data.shape[0]-1))
        y1 = max(0, min(y1, hmi.data.shape[0]-1))
        xlim, ylim = (x0, x1), (y0, y1)
    else:
        xlim, ylim = compute_window_in_pixels(hmi,
                                              cx=0*u.arcsec, cy=0*u.arcsec,
                                              width_pix=WIN_PIXELS[0],
                                              height_pix=WIN_PIXELS[1])
    seeds = define_seed_points(hmi, xlim, ylim, n_x=25, n_y=25,
                               strong_only=USE_STRONG, thr=THRESH)
    flines = trace_field_lines(seeds, pfss_out)

    # 6) 描画
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=hmi)

    ax.imshow(rgb, origin='lower', aspect='equal')
    try:
        hmi.draw_limb(axes=ax, color='white', linestyle='dashed', linewidth=1.2)
        hmi.draw_grid(axes=ax, grid_spacing=15*u.deg, color='white', linestyle='dotted', linewidth=0.8, alpha=0.7)
    except Exception:
        pass

        # 既存の描画の直後に
    draw_field_lines(ax, hmi, flines, rss=RSS)
    # if RMAX is not None and RMAX > RSS:
    #     extend_open_fieldlines_smooth(
    #         ax, hmi, flines, rss=RSS,
    #         rmax=RMAX,          # RMAX の代わり
    #         step_dr=STEP_DR,      # STEP_DR の代わり
    #         blend_dr=BLEND_DR,      # BLEND_DR の代わり
    #         linewidth=0.8, alpha=0.9, linestyle='--'
    #     )


    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("Solar X (arcsec)"); ax.set_ylabel("Solar Y (arcsec)")
    ax.set_title(f"AIA RGB (211/193/171) + HMI PFSS  Rss={RSS}, nrho={NRHO}\n"
                 f"{hmi.date.strftime('%Y-%m-%d %H:%M:%S UT')}")

    # 凡例（簡易）
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], color='red', lw=0.8, label='Open (+polarity)'),
        Line2D([0],[0], color='blue', lw=0.8, label='Open (-polarity)'),
        Line2D([0],[0], color='black', lw=0.8, label='Closed'),
    ]
    ax.legend(handles=legend_elems, loc='upper right', fontsize=9)

    plt.tight_layout()
    if SAVE_PATH:
        Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
        print(f"[保存] {SAVE_PATH}")
        plt.show()
    else:
        plt.show()

# def extend_open_fieldlines_smooth(ax, hmi_map, field_lines, rss,
#                                   rmax=6.0, step_dr=0.02, blend_dr=0.5,
#                                   linewidth=0.8, alpha=0.9, linestyle='--'):
#     """
#     Rssでの接線方向を保ちながら、半径方向へC1連続（cubic smoothstep）で収束。
#     """
#     with SphericalScreen(hmi_map.observer_coordinate):
#         for fl in field_lines:
#             coords = fl.coords
#             try:
#                 r_end = coords.radius[-1].to_value(u.Rsun)
#             except Exception:
#                 continue
#             if r_end < (rss - 0.02):
#                 continue

#             # 始点極性→色
#             try:
#                 pxy0 = hmi_map.world_to_pixel(coords[0].transform_to(hmi_map.coordinate_frame))
#                 x0, y0 = int(round(pxy0.x.value)), int(round(pxy0.y.value))
#                 pol = hmi_map.data[y0, x0] if (0 <= y0 < hmi_map.data.shape[0] and 0 <= x0 < hmi_map.data.shape[1]) else 0
#             except Exception:
#                 pol = 0
#             color = 'red' if pol > 0 else 'blue'

#             # 末端接線（HGSの3D cartesianで推定）
#             c_prev = coords[-2].transform_to(HGS(obstime=hmi_map.date))
#             c_end  = coords[-1].transform_to(HGS(obstime=hmi_map.date))
#             p_prev = c_prev.cartesian.xyz.to_value(u.Rsun)
#             p_end  = c_end.cartesian.xyz.to_value(u.Rsun)
#             v0 = p_end - p_prev
#             if np.dot(v0, p_end) < 0:  # 内向きなら反転
#                 v0 = -v0
#             v0 = v0 / (np.linalg.norm(v0) + 1e-12)
#             er = p_end / (np.linalg.norm(p_end) + 1e-12)

#             # 前進積分
#             pts_x, pts_y = [], []
#             P = p_end.copy()
#             r_now = np.linalg.norm(P)
#             while r_now < rmax - 1e-6:
#                 s = np.clip((r_now - rss)/max(blend_dr, 1e-6), 0.0, 1.0)
#                 w = 1 - (3*s**2 - 2*s**3)   # 多項式ブレンド
#                 u_dir = w*v0 + (1-w)*er
#                 u_dir /= (np.linalg.norm(u_dir) + 1e-12)

#                 P = P + step_dr * u_dir
#                 r_now = np.linalg.norm(P)
#                 er = P / (r_now + 1e-12)    # 放射方向を更新

#                 ext = SkyCoord(x=P[0]*u.Rsun, y=P[1]*u.Rsun, z=P[2]*u.Rsun,
#                                frame=HGS, obstime=hmi_map.date, representation_type='cartesian')
#                 ext_hpc = ext.transform_to(hmi_map.coordinate_frame)
#                 pix = hmi_map.world_to_pixel(ext_hpc)
#                 xs, ys = float(pix.x.value), float(pix.y.value)
#                 if np.isfinite(xs) and np.isfinite(ys):
#                     pts_x.append(xs); pts_y.append(ys)
#             if len(pts_x) >= 2:
#                 ax.plot(pts_x, pts_y, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)





if __name__ == "__main__":
    # ===== ユーザー環境に合わせてここを編集 =====
    from datetime import datetime

    # 入力ファイル/ディレクトリ
    HMI_FILE = r"/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    AIA_ROOT = r"/mnt/d/wsl/home/kinno-7010/Research/SDO/AIA/Rawdata"

    # 対象時刻（AIAファイル名は分解能1分の想定）
    TIME_OBJ = datetime.strptime("2022-06-13 03:00", "%Y-%m-%d %H:%M")

    # AIA RGB の各チャンネル（R/G/B）
    RCH, GCH, BCH = "211", "193", "171"

    # PFSS設定
    RSS = 2.0
    NRHO = 50

    # シード点抽出
    USE_STRONG = True
    THRESH = 150.0  # Gauss

    # 表示窓サイズ[pixel]（HMI座標で統一）
    WIN_PIXELS = (1024, 1024)
    X_RANGE_PIX = (-512, 0)
    Y_RANGE_PIX = (0, 512)
    
    RMAX = 6.0          # 例: 6 R⊙ まで延長
    CONT_MODE = 'radial'  # もしくは 'parker'
    STEP_DR  = 0.01    # 積分ステップ幅 [Rsun]（~14 Mm 相当）
    BLEND_DR = 0.6     # 非放射成分をゼロにするまでの厚み [Rsun]    

    # 出力先（Noneなら表示のみ）
    SAVE_PATH = fr"/mnt/d/wsl/home/kinno-7010/Research/PFSS/aia_pfss_Rss{RSS}_nrho{NRHO}_{TIME_OBJ.strftime('%Y%m%d_%H%M')}.png"

    # 実行
    run_pipeline(HMI_FILE, AIA_ROOT, TIME_OBJ,
                 RCH=RCH, GCH=GCH, BCH=BCH,
                 RSS=RSS, NRHO=NRHO,
                 USE_STRONG=USE_STRONG, THRESH=THRESH,
                 WIN_PIXELS=WIN_PIXELS,
                #  X_RANGE_PIX=X_RANGE_PIX, Y_RANGE_PIX=Y_RANGE_PIX,
                 RMAX=RMAX, STEP_DR=STEP_DR, BLEND_DR=BLEND_DR,
                 SAVE_PATH=SAVE_PATH)
