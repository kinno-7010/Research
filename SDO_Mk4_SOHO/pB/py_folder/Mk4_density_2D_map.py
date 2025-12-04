from __future__ import annotations

"""
density_2D_map_mk4.py

Mk4 pB (2011-09-20) + SOHO/LASCO-C2 pB を用いて
density_2D_map.py と同じ設定・プロットで 2D 電子密度マップを作る専用スクリプト。

- 幾何・反転ロジック・描画は density_2D_map.py 内の関数を再利用する。
- Mk4 特有の FITS ヘッダ (RSUN, CRRADIUS など) に対応するため、
  ここで専用の読み込み関数 load_mk4_pb_data() を定義する。
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from io_and_processing import (
    combine_corona_data,
    load_and_prepare_instrument_data,   # LASCO 側で使用
)

from density_2D_map import (
    invert_pb_to_density_2D,
    build_adaptive_radial_edges,
    plot_density_map,
    export_density_csv,
)


def load_mk4_pb_data(filename: str):
    """
    Mk4 pB FITS を読み込み、pB を B_sun 単位に変換しつつ
    density_2D_map 系で使う params 辞書を返す。

    期待ヘッダ (例):
      NAXIS1, NAXIS2
      CRPIX1, CRPIX2
      CDELT1  : [arcsec/pixel]
      RSUN    : [arcsec] (または RSUN_OBS / R_SUN / CRRADIUS)
      BSCALE, BZERO, BUNIT='Bsun'
    """
    with fits.open(filename) as hdul:
        raw = hdul[0].data.astype(float)
        header = hdul[0].header
        print(f"Successfully loaded Mk4 file: {filename}")

    # --- pB スケーリング: physical = raw * BSCALE + BZERO ---
    bscale = float(header.get("BSCALE", 1.0))
    bzero  = float(header.get("BZERO", 0.0))
    bunit  = str(header.get("BUNIT", "")).strip()
    data = raw * bscale + bzero

    if bunit.lower() not in ("bsun", "b/bsun", "b-bsun"):
        print(f"Warning: Mk4 BUNIT='{bunit}' (expected something like 'Bsun'). "
              "Ensure this really is normalized to mean solar disk brightness.")

    # --- 幾何情報 ---
    params: dict[str, float] = {}
    params["nx"] = int(header["NAXIS1"])
    params["ny"] = int(header["NAXIS2"])
    params["cx"] = float(header["CRPIX1"]) - 1.0  # 0-based index
    params["cy"] = float(header["CRPIX2"]) - 1.0
    params["scale"] = abs(float(header["CDELT1"]))  # [arcsec/px]

    # 太陽半径 [arcsec]
    if "RSUN_OBS" in header:
        rsun_arc = float(header["RSUN_OBS"])
    elif "RSUN" in header:
        rsun_arc = float(header["RSUN"])
    elif "R_SUN" in header:
        # R_SUN は [px] のことが多いので CDELT1 で [arcsec] へ
        rsun_arc = float(header["R_SUN"]) * params["scale"]
    elif "CRRADIUS" in header:
        # CRRADIUS: solar radius [pixels]
        rsun_arc = float(header["CRRADIUS"]) * params["scale"]
    else:
        raise KeyError(
            "Mk4 header missing RSUN / RSUN_OBS / R_SUN / CRRADIUS; "
            "cannot determine solar radius."
        )

    params["rsun_arc"] = rsun_arc
    params["px_per_rsun"] = rsun_arc / params["scale"]

    print(
        f"Mk4 params: cx={params['cx']:.2f}, cy={params['cy']:.2f}, "
        f"scale={params['scale']:.3f} arcsec/px, rsun={params['rsun_arc']:.2f} arcsec, "
        f"R_sun_px={params['px_per_rsun']:.2f} px"
    )

    return data, params


if __name__ == "__main__":
    # ============================================================
    #  入力ファイルパス（必要に応じて環境に合わせて書き換えてください）
    # ============================================================
    filename_mk4 = (
        r"/mnt/d/wsl/home/kinno-7010/Research/MK4_coronagraph/"
        r"MK4_coronagraph_KCOR/pB/Rawdata/20110920.021602.mk4.rpb.fts"
    )
    filename_lasco = (
        r"/mnt/d/wsl/home/kinno-7010/Research/SOHO/pB/C2-PB-20110920_0257.fts"
    )

    # 反転モード:
    #   'axisymmetric' : θ セクタ毎に軸対称反転（density_2D_map.py と同じ既定）
    #   'spherical'    : 全周平均→1D 反転→2D 段差塗り
    SYMMETRY = "axisymmetric"

    # ============================================================
    #  データ読み込み
    # ============================================================
    # Mk4 側（専用ローダー）
    data_mk4, params_mk4 = load_mk4_pb_data(filename_mk4)

    # LASCO 側（既存ユーティリティ）
    data_lasco, params_lasco = load_and_prepare_instrument_data(
        filename_lasco, "SOHO/LASCO-C2", is_lasco=True
    )

    # LASCO グリッド上での radial map [R_sun]
    y_idx, x_idx = np.indices((params_lasco["ny"], params_lasco["nx"]))
    r_map_lasco = np.hypot(
        (x_idx - params_lasco["cx"]) / params_lasco["px_per_rsun"],
        (y_idx - params_lasco["cy"]) / params_lasco["px_per_rsun"],
    )

    # ============================================================
    #  Mk4 + LASCO の pB 合成（density_2D_map.py と同じ設定）
    # ============================================================
    r_ranges = {
        "mk4_inner": 1.10,   # Mk4 データが有効になる内側境界
        "mk4_outer_lasco_inner": 2.20,  # Mk4 と LASCO の接合半径
        "lasco_outer": 7.00,  # LASCO C2 の外側端まで
    }

    final_pb = combine_corona_data(
        data_lasco,
        params_lasco,
        data_mk4,
        params_mk4,
        r_map_lasco,
        r_ranges,
    )

    # ============================================================
    #  反転に使う半径ビン (adaptive Edges)
    # ============================================================
    r_min = r_ranges["mk4_inner"]
    r_trn = r_ranges["mk4_outer_lasco_inner"]
    r_max = r_ranges["lasco_outer"]

    r_edges = build_adaptive_radial_edges(
        r_min=r_min,
        r_transition=r_trn,
        r_max=r_max,
        params_mk4=params_mk4,
        params_lasco=params_lasco,
        inner_px_factor=3.0,   # Mk4 側: ~2–3 px 厚
        outer_px_factor=2.0,   # LASCO 側: ~2 px 厚
        min_dr=0.01,
        max_dr=0.25,
    )

    # ============================================================
    #  pB → Ne 2D 反転
    # ============================================================
    if SYMMETRY.lower().startswith("sph"):
        # 球対称反転
        density_map, aux = invert_pb_to_density_2D(
            pb_image=final_pb,
            r_map_rsun=r_map_lasco,
            params_ref=params_lasco,
            r_min=r_min,
            r_max=r_max,
            symmetry="spherical",
            r_edges=r_edges,
            spatial_fill_mode="nearest",
            spatial_fill_iters=1,
        )
        title = (
            "Mk4 + SOHO/LASCO-C2 Electron Density "
            "(Spherical symmetric inversion)\n"
            "2011-09-20 02:57 UT"
        )
        suffix = "sph"
    else:
        # 軸対称反転（推奨）
        density_map, aux = invert_pb_to_density_2D(
            pb_image=final_pb,
            r_map_rsun=r_map_lasco,
            params_ref=params_lasco,
            r_min=r_min,
            r_max=r_max,
            symmetry="axisymmetric",
            theta_step_deg=1.0,          # θ=1° ステップ（extract_pB_profile の ±5° と整合）
            r_edges=r_edges,
            use_bilinear=False,          # Mk4+LASCO pB の段差ビン表示に揃える
            theta_neighbor_blend=2,      # θ 近傍での最小限のブレンド
            theta_neighbor_fallback=6,   # 残欠損に対する θ 近傍フォールバック
            spatial_fill_mode="nearest",
            spatial_fill_iters=1,
        )
        title = (
            "Mk4 + SOHO/LASCO-C2 Electron Density "
            "(Axisymmetric inversion)\n"
            "2011-09-20 02:57 UT"
        )
        suffix = "axi"

    # ============================================================
    #  プロット & 保存（density_2D_map.py と同じ描画範囲・単位: pixel）
    # ============================================================
    fig, ax = plot_density_map(
        density_map=density_map,
        r_map_plot=r_map_lasco,
        params_ref=params_lasco,
        r_ranges=r_ranges,
        title=title,
        # 目盛り単位は「太陽中心からのピクセル」
        # xlim_pix=(-150, 0),
        # ylim_pix=(-100, 150),
    )

    out_base = (
        r"/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/pB/2D_density_map_mk4"
    )
    png_path = f"{out_base}_{suffix}_20110920_0257.png"
    csv_path = f"{out_base}_{suffix}_20110920_0257.csv"

    fig.savefig(png_path, dpi=300)
    print(f"Saved 2D density map PNG to: {png_path}")

    export_density_csv(
        density_map,
        r_map_lasco,
        params_lasco,
        csv_path,
        include_all=False,
    )
    print(f"Saved per-pixel density CSV to: {csv_path}")

    plt.show()
