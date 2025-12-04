"""
AIA RGB running-difference + GCS wireframe overlay

前提:
- aia_diff_plot_analysis.py 内の AIA 用ユーティリティ
  (BASE_DATA_DIR, parse_datetime_str, normalize_log_stretch, get_dn_per_s)
- plot_GCS.py 内で import されている GCS 関係
  (GCSParams, sample_gcs_wireframe_points)

をそのまま import して利用する。
"""

from pathlib import Path
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames as sunpy_frames
import sunpy.map

# --- 既存コードから必要なものを import ---
# モジュール名は実際のファイル名に合わせて変更してください
import sys
sys.path.append("/mnt/d/wsl/home/kinno-7010/Research/SDO/AIA")
from aia_diff_plot_analysis import (
    BASE_DATA_DIR,
    parse_datetime_str,
    normalize_log_stretch,
    get_dn_per_s,
    _format_time_str,
)


# 太陽中心＋Rsun 円を描く関数（既存と整合）
from aia_MGN_diff_plot import add_center_and_rsun

sys.path.append("/mnt/d/wsl/home/kinno-7010/Research/GCS/gcs_overlay")

# GCS パラメータ & ワイヤーフレーム生成
from gcs_geometry import GCSParams, sample_gcs_wireframe_points


# =========================================================
# GCS ワイヤーフレームを AIA WCS（Helioprojective）上に投影するヘルパー
# =========================================================
def _project_gcs_wireframe_to_aia(
    gcs_params: GCSParams,
    obstime: Time,
    reference_map: sunpy.map.Map,
    n_parallels: int = 8,
    n_meridians: int = 32,
    include_legs: bool = True,
):
    """
    GCSParams から 3D ワイヤーフレームをサンプリングし，
    AIA 視点（reference_map）から見た Helioprojective 座標列と
    Apex 情報を返す。

    戻り値:
        curves_hpc : list of (kind, hpc_coords)
            kind: "parallels", "meridians", "legs" のいずれか
            hpc_coords: SkyCoord (N,) in Helioprojective
        apex_info : dict or None
            {
              "hpc": hpc_apex (SkyCoord in Helioprojective),
              "radius_rsun": float,
              "phi_deg": float,
              "theta_deg": float,
            }
    """
    # sample_gcs_wireframe_points は plot_GCS.py と同じ呼び出しに合わせる
    wireframe = sample_gcs_wireframe_points(
        gcs_params,
        obstime=obstime,
        n_parallels=n_parallels,
        n_meridians=n_meridians,
        include_legs=include_legs,
    )

    curves_hpc: list[tuple[str, SkyCoord]] = []
    stacks: list[np.ndarray] = []

    def project_curve(curve_points: np.ndarray):
        """
        3D カルテシアン (x,y,z) [Rsun] → HeliographicStonyhurst → Helioprojective
        """
        if not isinstance(curve_points, np.ndarray):
            return None
        if curve_points.ndim != 2 or curve_points.shape[1] != 3:
            return None

        x = curve_points[:, 0]
        y = curve_points[:, 1]
        z = curve_points[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)

        mask = r > 0
        if not np.any(mask):
            return None

        # 球座標へ（lon, lat, r）= (φ, θ, r)
        lon = np.arctan2(y[mask], x[mask])              # [-π, π]
        lat = np.arcsin(np.clip(z[mask] / r[mask], -1, 1))  # [-π/2, π/2]

        hg = SkyCoord(
            lon * u.rad,
            lat * u.rad,
            r[mask] * u.R_sun,
            frame=sunpy_frames.HeliographicStonyhurst,
            obstime=obstime,
        )

        hpc = hg.transform_to(reference_map.coordinate_frame)
        return hpc

    # 各ポリラインを HPC に投影しつつ，Apex 探索用に 3D 点も貯める
    for kind in ("parallels", "meridians", "legs"):
        curves = wireframe.get(kind, []) if isinstance(wireframe, dict) else []
        for curve in curves:
            if isinstance(curve, np.ndarray) and curve.size:
                stacks.append(curve)
            hpc_curve = project_curve(curve)
            if hpc_curve is not None:
                curves_hpc.append((kind, hpc_curve))

    # Apex 計算
    apex_info = None
    if stacks:
        points = np.vstack(stacks)        # shape (N, 3)
        radii = np.linalg.norm(points, axis=1)
        if radii.size:
            idx_max = int(np.argmax(radii))
            apex_point = points[idx_max]
            apex_r = float(radii[idx_max])

            if apex_r > 0:
                z_norm = np.clip(apex_point[2] / apex_r, -1.0, 1.0)
            else:
                z_norm = 0.0

            phi_rad = float(np.arctan2(apex_point[1], apex_point[0]))
            theta_rad = float(np.arcsin(z_norm))
            phi_deg = float(np.degrees(phi_rad))
            theta_deg = float(np.degrees(theta_rad))

            apex_coord = SkyCoord(
                lon=phi_rad * u.rad,
                lat=theta_rad * u.rad,
                radius=apex_r * u.R_sun,
                frame=sunpy_frames.HeliographicStonyhurst,
                obstime=obstime,
            )
            hpc_apex = apex_coord.transform_to(reference_map.coordinate_frame)

            apex_info = {
                "hpc": hpc_apex,
                "radius_rsun": apex_r,
                "phi_deg": phi_deg,
                "theta_deg": theta_deg,
            }

    return curves_hpc, apex_info


# =========================================================
# AIA RGB ランニング差分 + GCS ワイヤーフレームのメイン関数
# =========================================================
def plot_sdo_aia_rgb_diff_with_gcs(
    datetime_str: str,
    # AIA 3ch
    channel_r_str: str = "211",
    channel_g_str: str = "193",
    channel_b_str: str = "171",
    delta_minutes: int = 2,
    # GCS パラメータ（未指定時は既存 GCS デフォルト）
    gcs_params: GCSParams | None = None,
    h_apex: float = 3.92,
    kappa: float = 0.10,
    alpha_deg: float = 23.0,
    tilt_deg: float = 87.0,
    lon_deg: float = -44.0,
    lat_deg: float = 10.0,
    # 表示系
    xlim_arcsec: tuple[float, float] | None = None,
    ylim_arcsec: tuple[float, float] | None = None,
    vmax_gray: float = 0.015,  # 元関数と同じ引数は残すが，ロジックは合わせる
    save_path: str | Path | None = None,
    n_parallels: int = 8,
    n_meridians: int = 32,
):
    """
    aia_diff_plot_analysis.py の plot_sdo_aia_rgb_diff と
    同じ処理で AIA 差分マップを描き，その上に GCS ワイヤーフレームを重ねる。
    """

    # -----------------------------
    # 1. 日時の処理（現在時刻 & delta 分前）
    # -----------------------------
    dt_cur = parse_datetime_str(datetime_str)
    if dt_cur is None:
        return

    dt_cur = dt_cur.replace(second=0, microsecond=0)
    dt_prev = dt_cur - timedelta(minutes=delta_minutes)

    date_cur = dt_cur.strftime("%Y%m%d")
    time_cur_tag = dt_cur.strftime("%H%M")
    date_prev = dt_prev.strftime("%Y%m%d")
    time_prev_tag = dt_prev.strftime("%H%M")

    channels = {"r": channel_r_str, "g": channel_g_str, "b": channel_b_str}

    # ------------------------------------------------
    # 2. 各時刻ごとに、3波長のMapオブジェクトをロード
    # ------------------------------------------------
    def load_maps_for_time(date_str: str, time_str: str):
        maps = {}
        loaded = 0
        for color, ch_str in channels.items():
            wavelength_part_in_fname = ch_str.zfill(4)
            filename = f"AIA{date_str}_{time_str}_{wavelength_part_in_fname}.fits"
            file_path = BASE_DATA_DIR / ch_str / filename
            print(
                f"[{date_str} {time_str}] 読み込み試行: "
                f"{color.upper()} ({ch_str}Å) - {file_path}"
            )
            try:
                maps[color] = sunpy.map.Map(file_path)
                print(f"  成功: {ch_str}Å")
                loaded += 1
            except Exception as e:
                print(f"  失敗: {ch_str}Å のファイル読み込みエラー: {e}")
                maps[color] = None
        return maps, loaded

    maps_cur, loaded_cur = load_maps_for_time(date_cur, time_cur_tag)
    maps_prev, loaded_prev = load_maps_for_time(date_prev, time_prev_tag)

    if loaded_cur < 3 or loaded_prev < 3:
        print("エラー: 3波長すべて読み込めませんでした。")
        return

    reference_map = (
        maps_cur["b"] if maps_cur["b"] else maps_cur["g"] if maps_cur["g"] else maps_cur["r"]
    )
    if not reference_map:
        print("エラー: 基準となるMapオブジェクトがありません。")
        return

    wcs_info = reference_map.wcs

    def get_map_time(maps_dict):
        for key in ("b", "g", "r"):
            m = maps_dict.get(key)
            if m is not None:
                return m.date
        return None

    time_cur = get_map_time(maps_cur)
    time_prev = get_map_time(maps_prev)
    time_cur_str = _format_time_str(time_cur)
    time_prev_str = _format_time_str(time_prev)

    # ------------------------------------------------
    # 3. 各時刻・各チャンネルのデータを正規化 → RGB画像に変換
    # ------------------------------------------------
    def make_rgb_image(maps_dict):
        try:
            red_channel_data = normalize_log_stretch(maps_dict["r"].data)
            green_channel_data = normalize_log_stretch(maps_dict["g"].data)
            blue_channel_data = normalize_log_stretch(maps_dict["b"].data)
        except Exception as e_norm:
            print(f"データ正規化中にエラー: {e_norm}")
            return None

        def scale_to_01(data):
            d_min = np.nanmin(data)
            d_max = np.nanmax(data)
            if d_max == d_min:
                return np.zeros_like(data)
            return (data - d_min) / (d_max - d_min)

        red_final = scale_to_01(red_channel_data)
        green_final = scale_to_01(green_channel_data)
        blue_final = scale_to_01(blue_channel_data)

        return np.stack([red_final, green_final, blue_final], axis=-1)

    rgb_cur = make_rgb_image(maps_cur)
    rgb_prev = make_rgb_image(maps_prev)

    if rgb_cur is None or rgb_prev is None:
        print("エラー: RGB画像の生成に失敗しました。")
        return

    # ------------------------------------------------
    # 4. RGB差分画像 → 1チャンネルのスカラー差分に潰す
    # ------------------------------------------------
    diff_211 = get_dn_per_s(maps_cur["r"]) - get_dn_per_s(maps_prev["r"])
    diff_193 = get_dn_per_s(maps_cur["g"]) - get_dn_per_s(maps_prev["g"])
    diff_171 = get_dn_per_s(maps_cur["b"]) - get_dn_per_s(maps_prev["b"])

    diff_scalar = (diff_211 + diff_193 + diff_171) / 3.0

    if isinstance(diff_scalar, np.ma.MaskedArray):
        diff_scalar = diff_scalar.filled(np.nan)

    finite = np.isfinite(diff_scalar)
    if np.any(finite):
        lo, hi = np.nanpercentile(diff_scalar[finite], [10, 90])
        vmax = max(abs(lo), abs(hi))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = np.nanmax(np.abs(diff_scalar[finite]))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1e-3
        vmin = -vmax
        print("vmin", vmin, "vmax", vmax)
    else:
        diff_scalar = np.zeros_like(diff_scalar, dtype=float)
        vmin, vmax = -1.0, 1.0

    # ------------------------------------------------
    # 5. プロット準備（ここまでが元の plot_sdo_aia_rgb_diff と同じ）
    # ------------------------------------------------
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection=wcs_info)

    im = ax.imshow(
        diff_scalar,
        origin="lower",
        aspect="equal",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    # 太陽リムとグリッド
    try:
        reference_map.draw_limb(
            axes=ax, color="red", linestyle="dashed", linewidth=1.2
        )
        reference_map.draw_grid(
            axes=ax,
            grid_spacing=15 * u.deg,
            color="red",
            linestyle="dotted",
            linewidth=0.8,
            alpha=0.7,
        )
        add_center_and_rsun(ax, reference_map)
    except Exception as e_draw:
        print(f"警告: 太陽リム/グリッド/円の描画に失敗しました: {e_draw}")

    # 6. 描画範囲（arcsec指定）
    cdelt1 = reference_map.meta.get("cdelt1")
    cdelt2 = reference_map.meta.get("cdelt2")
    crval1 = reference_map.meta.get("crval1", 0.0)
    crval2 = reference_map.meta.get("crval2", 0.0)
    crpix1 = reference_map.meta.get("crpix1", reference_map.data.shape[1] / 2.0)
    crpix2 = reference_map.meta.get("crpix2", reference_map.data.shape[0] / 2.0)

    if (
        cdelt1 is not None
        and cdelt2 is not None
        and cdelt1 != 0
        and cdelt2 != 0
    ):
        if xlim_arcsec is not None:
            x1_arc, x2_arc = xlim_arcsec
            x1_pix = (x1_arc - crval1) / cdelt1 + crpix1
            x2_pix = (x2_arc - crval1) / cdelt1 + crpix1
            ax.set_xlim(x1_pix, x2_pix)

        if ylim_arcsec is not None:
            y1_arc, y2_arc = ylim_arcsec
            y1_pix = (y1_arc - crval2) / cdelt2 + crpix2
            y2_pix = (y2_arc - crval2) / cdelt2 + crpix2
            ax.set_ylim(y1_pix, y2_pix)
    else:
        print("警告: CDELT が取得できないため、arcsec での描画範囲指定は無効です。")

    # 7. タイトル & 軸ラベル
    title_str_parts = [
        f"SDO/AIA RGB Running Difference: ({channel_r_str}+{channel_g_str}+{channel_b_str}Å)\n",
        f"{time_prev_str} → {time_cur_str} UT",
    ]
    ax.set_title("\n".join(title_str_parts), fontsize=12, pad=5)

    ax.coords[0].set_axislabel("Solar X (arcsec)")
    ax.coords[1].set_axislabel("Solar Y (arcsec)")
    ax.coords[0].set_format_unit(u.arcsec)
    ax.coords[1].set_format_unit(u.arcsec)
    ax.tick_params(axis="both", which="major", labelsize=10, direction="in")

    # ------------------------------------------------
    # 8. ここから GCS ワイヤーフレームのみ追加
    # ------------------------------------------------
    if gcs_params is None:
        gcs_params = GCSParams(
            h_apex=h_apex,
            kappa=kappa,
            alpha_deg=alpha_deg,
            tilt_deg=tilt_deg,
            lon_deg=lon_deg,
            lat_deg=lat_deg,
        )

    obstime = Time(time_cur.isot if time_cur is not None else dt_cur.isoformat())

    # ワイヤーフレーム + Apex 情報を取得
    curves_hpc, apex_info = _project_gcs_wireframe_to_aia(
        gcs_params,
        obstime=obstime,
        reference_map=reference_map,
        n_parallels=n_parallels,
        n_meridians=n_meridians,
        include_legs=True,
    )

    # WCSAxes の plot_coord を使って AIA 上にワイヤーフレームを描画
    for kind, hpc_curve in curves_hpc:
        # legs は少し細く / 透過度高めにするなど
        if kind == "legs":
            lw = 0.8
            alpha_line = 0.3
            z = 5
        else:
            lw = 1.0
            alpha_line = 0.3
            z = 6

        ax.plot_coord(
            hpc_curve,
            color="lime",
            linewidth=lw,
            alpha=alpha_line,
            zorder=z,
        )

    # --- Apex にオレンジ丸を描く部分（plot_GCS.py と同等の表現） ---
    if apex_info is not None:
        hpc_apex = apex_info["hpc"]
        apex_r = apex_info["radius_rsun"]

        # ラベルは plot_GCS.py と同じフォーマットに合わせる
        apex_label = (
            f"Apex height (r={apex_r:.3f} $R_\\odot$, "
            f"lon={gcs_params.lon_deg:.1f}$^\\circ$, "
            f"lat={gcs_params.lat_deg:.1f}$^\\circ$)"
        )

        # world 座標(SkyCoord)を WCSAxes 上にプロット
        ax.plot_coord(
            hpc_apex,
            marker="o",
            linestyle="None",
            markerfacecolor="orange",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=7.0,
            zorder=7,
            label=apex_label,
        )

    # 凡例用に GCS ワイヤーフレームのエントリを追加
    ax.plot([], [], color="lime", lw=1.0, label=gcs_params.legend_label())
    ax.legend(loc="upper right", fontsize=9)

    fig.subplots_adjust(bottom=0.20)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved AIA+GCS figure to {save_path}")

    return fig, ax


# =========================================================
# スクリプトとして実行されたときの例
# =========================================================
if __name__ == "__main__":
    # 例: 2022-06-13 03:33 に対して 2 分ランニング差分 + GCS
    target_time_str = "2022-06-13 02:00"
    target_time_str_no_colon = target_time_str.replace(':', '')

    # 表示範囲 (Zucca イベントで使っている範囲の一例)
    xlim_arcsec = (-1230.0, -400.0)
    ylim_arcsec = (0.0, 800.0)

    out_png = (
        Path("/mnt/d/wsl/home/kinno-7010/Research/GCS/output")
        / "AIA_GCS"
        / f"aiaRGB_diff_GCS_{target_time_str_no_colon.replace(' ', '_')}.png"
    )

    plot_sdo_aia_rgb_diff_with_gcs(
        target_time_str,
        channel_r_str="211",
        channel_g_str="193",
        channel_b_str="171",
        delta_minutes=2,
        # 必要ならここをフィット済みパラメータに書き換え
        h_apex=1.718,
        kappa=0.04,
        alpha_deg=5.6,
        tilt_deg=72.0,
        lon_deg=-44.0,
        lat_deg=19,
        xlim_arcsec=xlim_arcsec,
        ylim_arcsec=ylim_arcsec,
        vmax_gray=0.5,
        save_path=out_png,
        n_parallels=8,
        n_meridians=32,
    )

    plt.show()
