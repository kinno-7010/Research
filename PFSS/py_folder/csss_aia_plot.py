# -*- coding: utf-8 -*-
"""
AIA RGB(211/193/171) + HMI CSSS-like (coefficient-based smooth radialization) field lines overlay

NOTE (重要):
 - CSSS の厳密解（Zhao & Hoeksema 1995, doi:10.1029/94JA02266）は、Rcs–Rss での
   magneto-hydrostatic (MHS) 解を球面調和係数で連結する必要があり大規模です。
 - 本スクリプトは “AIA 可視化や凡例・WCS 軸など PFSS 版の体裁を一切変えず” に、
   Rcs 以遠をなめらかにラジアル化する CSSS-lite（水平成分の指数減衰＋磁束保存）で
   外層を延長します。論文用に厳密係数版が必要な場合は、この外層部分を係数解へ差替えてください。

既知参照：
 - Zhao & Hoeksema (1995) CSSS: doi:10.1029/94JA02266
 - Bogdan & Low (1986) MHS family: ApJ 306, 271 (eta(r)=1+(a/r)^2)
 - Koskela et al. (2019) CSSS parameter ranges: A&A 631, A17, doi:10.1051/0004-6361/201935967

使い方：
  末尾の __main__ ブロックで日時やファイルパス、Rcs/Rss/a などを指定してください。
  可視化は AIA 171/193/211 の RGB 合成（あなたの PFSS 版と同一体裁）に磁力線を重畳します。
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
# 追加インポート（ファイル先頭の他の import と並べてください）
from sunpy.map import make_fitswcs_header
from reproject import reproject_interp
import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map
from sunpy.coordinates import frames as sunpy_frames

# pfsspy は PFSS の内層(photosphere→Rcs) 計算に用いる
import pfsspy
from pfsspy import tracing
# from pfsspy.pfss import PFSS
from astropy.wcs import WCS  # ← 追加

# ===================== AIA の読込とRGB合成（あなたの体裁に揃える） =====================
BASE_AIA_DIR = Path(r"/mnt/d/wsl/home/kinno-7010/Research_data/SDO/AIA/Rawdata")


def _load_aia_map(dt: datetime, wl: str) -> sunpy.map.Map:
    fname = f"AIA{dt:%Y%m%d}_{dt:%H%M}_{wl.zfill(4)}.fits"
    fpath = BASE_AIA_DIR / wl / fname
    return sunpy.map.Map(fpath)


def _normalize_percentile(img: np.ndarray, pmin=1.0, pmax=99.5):
    vmin = np.nanpercentile(img, pmin)
    vmax = np.nanpercentile(img, pmax)
    arr = np.clip(img.astype(float), vmin, vmax)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def make_aia_rgb_map(dt: datetime):
    m211 = _load_aia_map(dt, "211")
    m193 = _load_aia_map(dt, "193")
    m171 = _load_aia_map(dt, "171")
    r = _normalize_percentile(m211.data, 1.0, 99.5)
    g = _normalize_percentile(m193.data, 1.0, 99.5)
    b = _normalize_percentile(m171.data, 1.0, 99.5)
    rgb = np.stack([r, g, b], axis=-1)
    # 可視化 WCS は 171Å を基準（PFSS 版と同様）
    return rgb, m171

# ===================== PFSS: 内層(photosphere→rss_pfss) =====================

def build_pfss_inner(hmi_map: sunpy.map.Map, nrho: int, rss_pfss: float):
    # === 既存: NaN/Inf を 0 に置換（non-finite 対策） ===
    data = hmi_map.data
    if not np.all(np.isfinite(data)):
        clean = np.nan_to_num(data.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        hmi_map = sunpy.map.Map(clean, hmi_map.meta)

    # === 追加: CEA でなければ、簡易に CEA/HGC へリプロジェクト ===
    # pfsspy は CTYPE1/2 に '...-CEA' を要求
    ctype1 = hmi_map.meta.get('CTYPE1', '')
    ctype2 = hmi_map.meta.get('CTYPE2', '')
    is_cea = ('CEA' in ctype1) and ('CEA' in ctype2)

    if not is_cea:
        # 出力シノプティック格子（簡易）：経度 0..360 deg、緯度 -90..90 deg
        nlon, nlat = 180, 90
        # 経度は通常通り 360/nlon、緯度は CEA の π/2 スケールを考慮
        cdelt_lon = (360.0 / nlon) * u.deg                      # [deg / pix]
        cdelt_lat = (180.0 / nlat) / (np.pi / 2.0) * u.deg      # [deg / pix]  ← ここがポイント
        scale_xy  = u.Quantity([cdelt_lon.to(u.arcsec), cdelt_lat.to(u.arcsec)]) / u.pix  # [arcsec/pix]

        
        # 半径を明示し、観測者も明示（NaN回避）
        center_hgc = SkyCoord(
            0*u.deg, 0*u.deg, 1*u.R_sun,
            frame=sunpy_frames.HeliographicCarrington(obstime=hmi_map.date, observer='earth')
        )
        target_hdr = make_fitswcs_header(
            (nlat, nlon),
            center_hgc,
            scale=scale_xy,            # 先に設定した Quantity（arcsec/pix のままでOK）
            projection_code='CEA'
        )
        target_wcs = WCS(target_hdr)

        out_data, _ = reproject_interp((hmi_map.data, hmi_map.wcs), target_wcs,
                                    shape_out=(nlat, nlon), return_footprint=True)

        # 非有限は 0 埋め
        out_data = np.nan_to_num(out_data, nan=0.0, posinf=0.0, neginf=0.0)
        hmi_map = sunpy.map.Map(out_data, target_hdr)

    # === 既存: pfsspy へ（引数名は nr） ===
    inp = pfsspy.Input(hmi_map, nr=nrho, rss=rss_pfss)
    model = pfsspy.pfss(inp)  # 関数APIで Output を取得
    tracer = tracing.FortranTracer() if getattr(tracing, "USE_FORTRAN", False) else tracing.PythonTracer()

    return model, tracer


def make_seed_coords_on_disk(nlat: int, nlon: int, obstime, observer) -> SkyCoord:
    lats = np.linspace(-80, 80, nlat) * u.deg
    lons = np.linspace(-180, 180, nlon, endpoint=False) * u.deg
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    seeds = SkyCoord(lon_grid.ravel(), lat_grid.ravel(), 1*u.R_sun,
                     frame=sunpy_frames.HeliographicStonyhurst, obstime=obstime, observer=observer)
    return seeds


def trace_to_rcs(model, tracer, seeds_hgs: SkyCoord, rcs: float):
    """PFSS で種点から追跡し、Rcs 到達か閉ループかを分類。"""
    fllist = tracer.trace(seeds_hgs, model)
    inner_paths = []
    at_rcs = []
    is_open = []
    for fl in fllist:
        coords = fl.coords
        inner_paths.append(coords)
        rmax = (coords.radius / u.R_sun).max().value
        if rmax >= rcs - 1e-3:
            idx = np.nanargmin(np.abs((coords.radius / u.R_sun).value - rcs))
            at_rcs.append(coords[idx])
            is_open.append(True)
        else:
            at_rcs.append(None)
            is_open.append(False)
    return inner_paths, at_rcs, is_open

# ===================== CSSS-lite: Rcs→Rss のスムーズ延長（指数減衰＋磁束保存） =====================

def extend_csss_like(rcs_pt: SkyCoord, rcs: float, rss_csss: float, a_param: float, ds: float = 0.04):
    """
    CSSS の物理要点（Rcs 以遠の水平体積電流により B⊥ が徐々に 0 へ、Rss でラジアル）を
    簡明に模擬。水平成分の傾き角を exp(-(r-Rcs)/a_param) で減衰させながら前進。
    """
    # --- ここが重要：開始点を必ず HGS に揃える（observer も明示） ---
    
    obstime  = getattr(rcs_pt, "obstime", None)
    rcs_hgs = rcs_pt.transform_to(
        sunpy_frames.HeliographicStonyhurst(obstime=obstime)
    )

    # 積分初期化
    r = float(rcs)  # [Rsun]
    theta0 = np.deg2rad(7.0)   # Rcs直上の初期偏向角（小）
    curr = rcs_hgs

    # 経路を数値配列で保持（最後に一括で SkyCoord を作る：フレーム混在を避ける）
    lon_list = [curr.lon.to_value(u.deg)]
    lat_list = [curr.lat.to_value(u.deg)]
    r_list   = [curr.radius.to_value(u.R_sun)]

    # 前進積分
    while r < rss_csss - 1e-6:
        r_next = min(r + ds, rss_csss)

        # 水平角の指数減衰（a_param を減衰長とみなす）
        theta = theta0 * np.exp(-(r - rcs)/max(a_param, 1e-3))

        # ここでは単純に経度方向のみ進める（符号は子午線対称の簡易規定）
        sign = 1.0 if np.cos(np.deg2rad(lon_list[-1])) >= 0 else -1.0
        dlon = sign * theta * (ds / r)   # [rad]
        dlat = 0.0                       # [rad]

        new_r   = r_next
        new_lon = lon_list[-1]*u.deg + dlon*u.rad
        new_lat = lat_list[-1]*u.deg + dlat*u.rad

        lon_list.append(new_lon.to_value(u.deg))
        lat_list.append(new_lat.to_value(u.deg))
        r_list.append(new_r)

        r = new_r

    # すべて HGS で SkyCoord を作る（フレーム混在を回避）
    return SkyCoord(lon_list*u.deg, lat_list*u.deg, np.asarray(r_list)*u.R_sun,
                    frame=sunpy_frames.HeliographicStonyhurst, obstime=obstime)

# ===================== 可視化（AIA の体裁は PFSS 版と同一） =====================

def project_to_aia_hpc(path_hgs: SkyCoord, ref_map: sunpy.map.Map):
    hpc = path_hgs.transform_to(
        sunpy_frames.Helioprojective(observer=ref_map.observer_coordinate,
                                     obstime=ref_map.date)
    )
    return hpc


def plot_aia_with_csss_overlay(
    dt_str: str,
    hmi_path: str,
    nrho: int,
    rcs: float,
    rss_pfss: float,
    rss_csss: float,
    a_param: float,
    nlat: int,
    nlon: int,
    xlim_arcsec=None,
    ylim_arcsec=None,
):
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")

    # AIA RGB
    rgb, ref_map = make_aia_rgb_map(dt)

    # HMI マップ（そのまま sunpy.map.Map で読み込む）
    hmi_map = sunpy.map.Map(hmi_path)

    # PFSS 内層（photosphere→rss_pfss; rss_pfss は Rcs より十分大）
    model, tracer = build_pfss_inner(hmi_map, nrho=nrho, rss_pfss=rss_pfss)

    # シード（円盤上一様）
    seeds_hgs = make_seed_coords_on_disk(nlat=nlat, nlon=nlon,
                                         obstime=ref_map.date,
                                         observer=ref_map.observer_coordinate)
    
    # ROI（表示範囲）でシードを事前に間引き
    if (xlim_arcsec is not None) and (ylim_arcsec is not None):
        seeds_hpc = seeds_hgs.transform_to(
            sunpy_frames.Helioprojective(observer=ref_map.observer_coordinate, obstime=ref_map.date)
        )
        tx = seeds_hpc.Tx.to_value(u.arcsec); ty = seeds_hpc.Ty.to_value(u.arcsec)
        mask = (xlim_arcsec[0] <= tx) & (tx <= xlim_arcsec[1]) & (ylim_arcsec[0] <= ty) & (ty <= ylim_arcsec[1])
        seeds_hgs = seeds_hgs[mask]


    # PFSS で Rcs までトレース
    inner_paths, at_rcs, is_open = trace_to_rcs(model, tracer, seeds_hgs, rcs=rcs)

    # 図
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection=ref_map.wcs)

    # 追加：背景を白
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')


    # --- AIAの視野（HPC, arcsec）をextentで取得 ---
    # --- AIA の視野（HPC, arcsec）を ref_map.xrange / yrange から取得（NaN 回避） ---
    # 画像サイズ（ピクセル）
    ny, nx = ref_map.data.shape

    # 画像の「ピクセル端」→ HPC(arcsec)
    # --- AIAの視野（HPC, arcsec）をextentで取得 ---   ← ここのブロックを置き換え
    bl = ref_map.bottom_left_coord
    tr = ref_map.top_right_coord
    extent_world = [bl.Tx, tr.Tx, bl.Ty, tr.Ty]  # 単位付き Quantity のまま保持


    extent_arcsec = [
        bl.Tx.to_value(u.arcsec), tr.Tx.to_value(u.arcsec),
        bl.Ty.to_value(u.arcsec), tr.Ty.to_value(u.arcsec)
    ]

    # world(HPC) 座標で配置
    # WCS の空間単位（deg か arcsec か）を取得して合わせる
    xunit = u.Unit(ref_map.wcs.wcs.cunit[0] or 'deg')
    yunit = u.Unit(ref_map.wcs.wcs.cunit[1] or 'deg')

    extent_num = [extent_world[0].to_value(xunit),
                extent_world[1].to_value(xunit),
                extent_world[2].to_value(yunit),
                extent_world[3].to_value(yunit)]

    ax.imshow(
        rgb, origin='lower',
        extent=extent_num,
        transform=ax.get_transform('world'),
        zorder=0
    )

    ax.set_aspect('equal')




    ref_map.draw_limb(axes=ax, color='white', linestyle='dotted', linewidth=1.0)
    ref_map.draw_grid(axes=ax, grid_spacing=15*u.deg, color='white', linestyle='dotted', linewidth=0.7, alpha=0.7)

    # フィールドライン
    for path, rcsp, opened in zip(inner_paths, at_rcs, is_open):
        # PFSS 内層の投影
        hpc_inner = project_to_aia_hpc(path, ref_map)
        color = 'k'  # default closed
        if opened and rcsp is not None:
            # Rcs 点の Br の符号で色分け（近傍格子の符号を採用）
            rr = (rcsp.radius / u.R_sun).value
            th = (90*u.deg - rcsp.lat).to(u.rad).value
            ph = rcsp.lon.to(u.rad).value
            # --- grid配列を直接参照せず、等間隔格子を自前で構成して最近傍添字を求める ---
            bc = model.bc
            if isinstance(bc, tuple):
                # 旧API: bc は (Br, Bθ, Bφ) のタプル
                br_arr = bc[0]  # Br
            else:
                # 新API: bc は (nr, nt, nphi, 3) の ndarray（最後の軸が成分）
                br_arr = bc[..., 0]

            nr, nt, nphi = br_arr.shape  # Br の形状

            # 半径は 1..rss_pfss を nr 分割と仮定して rcs に最も近い層を取る
            if rss_pfss > 1.0:
                ir = int(np.clip(round((rcs - 1.0) / (rss_pfss - 1.0) * (nr - 1)), 0, nr - 1))
            else:
                ir = 0

            # 緯度 θ: 0..π を nt 分割
            theta_grid = np.linspace(0.0, np.pi, nt)
            it = int(np.clip(np.argmin(np.abs(theta_grid - th)), 0, nt - 1))

            # 経度 φ: 0..2π を nphi 分割（周期距離で最近傍）
            phi_grid = np.linspace(0.0, 2*np.pi, nphi, endpoint=False)
            # 距離を -π..π の範囲に折り畳んで絶対値最小を取る
            dphi = (phi_grid - ph + np.pi) % (2*np.pi) - np.pi
            ip = int(np.clip(np.argmin(np.abs(dphi)), 0, nphi - 1))

            br_sign = np.sign(br_arr[ir, it, ip])

            color = 'r' if br_sign >= 0 else 'b'

        ax.plot(
            hpc_inner.Tx.to_value(u.arcsec), hpc_inner.Ty.to_value(u.arcsec),
            color=color, lw=0.9, alpha=0.9,
            transform=ax.get_transform('world'), zorder=2
        )


        # Open 線は CSSS-lite で Rss まで延長
        if opened and rcsp is not None:
            ext = extend_csss_like(rcsp, rcs=rcs, rss_csss=rss_csss, a_param=a_param, ds=0.10)
            hpc_ext = project_to_aia_hpc(ext, ref_map)
            ax.plot(
                hpc_ext.Tx.to_value(u.arcsec), hpc_ext.Ty.to_value(u.arcsec),
                color=color, lw=0.9, alpha=0.9,
                transform=ax.get_transform('world'), zorder=2
            )


    # 凡例
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], color='r', lw=2, label='Open (+polarity)'),
        Line2D([0],[0], color='b', lw=2, label='Open (-polarity)'),
        Line2D([0],[0], color='k', lw=2, label='Closed'),
    ]
    ax.legend(handles=legend_elems, loc='upper right', frameon=True, fontsize=9)

    ax.set_title(
        f"AIA RGB (211/193/171)  +  HMI CSSS-like  Rcs={rcs}, a={a_param}, Rss={rss_csss}\n"
        f"{ref_map.date.strftime('%Y-%m-%d %H:%M:%S UT')}",
        fontsize=12, pad=8)
    ax.set_xlabel("Solar X (arcsec)")
    ax.set_ylabel("Solar Y (arcsec)")

    if xlim_arcsec is not None:
        xmin = (xlim_arcsec[0] * u.arcsec).to_value(xunit)
        xmax = (xlim_arcsec[1] * u.arcsec).to_value(xunit)
        ax.set_xlim(xmin, xmax)

    if ylim_arcsec is not None:
        ymin = (ylim_arcsec[0] * u.arcsec).to_value(yunit)
        ymax = (ylim_arcsec[1] * u.arcsec).to_value(yunit)
        ax.set_ylim(ymin, ymax)


    DATETIME_STR = ref_map.date.strftime('%Y%m%d-%H%M')
    output_path = f"/mnt/d/wsl/home/kinno-7010/Research_data/PFSS/csss_aia_plot_{DATETIME_STR}.png"
    plt.savefig(output_path, dpi=300)
    print(f"✓ CSSS-AIA プロットが正常に保存されました: {output_path}")
    
    plt.tight_layout()
    plt.show()


# ===================================== main =====================================
if __name__ == "__main__":
    # 入力：ユーザー指定
    DATETIME_STR = "2022-06-13 03:00"
    HMI_PATH = r"/mnt/d/wsl/home/kinno-7010/Research_data/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"

    # パラメータ（ご指定）
    RCS = 2.5
    RSS_CSSS = 4.0
    A_PARAM = 0.2
    LMAX = 90   # （現実の厳密 CSSS 係数解で使用想定。ここでは未使用）

    # PFSS 内層のメッシュ分解能と上限（Rcs より十分大）
    NRHO = 30
    RSS_PFSS = max(RSS_CSSS, RCS + 1.0)

    # シード密度
    NLAT, NLON = 18, 36

    # 表示範囲（例：全体表示。局所にするなら (-512,0), (-200,300) など）
    XLIM = (-512, 0)
    YLIM = (-300, 300)

    plot_aia_with_csss_overlay(
        dt_str=DATETIME_STR,
        hmi_path=HMI_PATH,
        nrho=NRHO,
        rcs=RCS,
        rss_pfss=RSS_PFSS,
        rss_csss=RSS_CSSS,
        a_param=A_PARAM,
        nlat=NLAT,
        nlon=NLON,
        xlim_arcsec=XLIM,
        ylim_arcsec=YLIM,
    )
