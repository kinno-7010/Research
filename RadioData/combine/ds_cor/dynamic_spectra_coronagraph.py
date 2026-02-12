from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any
from datetime import timedelta
import astropy.units as u
import matplotlib
matplotlib.use('Agg')  # TkAgg の代わりに Agg バックエンドを使用（GUIなし）
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.dates import SecondLocator
import math
import gc
import sunpy.map
from sunpy.coordinates import frames
sys.path.append(r"F:\wsl\home\kinno-7010\Research\RadioData\combine")
from wind_hf_assa_dynamic_spectrum import draw_line_between_points, load_wind_rad2, load_hf, load_callisto, create_dataframe, resample_to_grid, normalize_by_median, combine_spectra

import sys
sys.path.append(r"F:\wsl\home\kinno-7010\Research\SDO_Mk4_SOHO\py_folder")
from integrated_analysis import create_single_diff_from_time_image
from predict_type2_const_speed import overlay_prediction_fullspan, frequency_to_r_saito_factor, r_to_frequency_saito_factor
sys.path.append(r"F:\wsl\home\kinno-7010\Research\SDO\AIA")
from aia_diff_plot_analysis import parse_datetime_str, normalize_log_stretch, get_dn_per_s, add_center_and_rsun, _format_time_str

# from 

# Fixed data locations
# WIND_CDF_PATH = Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/Wind/Rawdata/wi_l2_wav_rad2_20220613_v01.cdf")
# HF_CDF_PATH = Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/HF_plot/Rawdata/it_h1_hf_20220613_v01.cdf")
# ASSA_FITS_PATHS = [
#     Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_010001_62.fit"),
#     Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_011501_62.fit"),
#     Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_013001_62.fit"),
#     Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_014501_62.fit"),
#     Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_020001_62.fit"),
#     Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_021501_62.fit"),
#     Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_023001_62.fit"),
#     Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_024500_62.fit"),
#     Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_030000_62.fit"),
#     Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_031500_62.fit"),
#     Path("/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_033000_62.fit"),
# ]
AIA_BASE_DATA_DIR = Path(r"F:\wsl\home\kinno-7010\Research\SDO\AIA\Rawdata")
WIND_CDF_PATH = Path(r"F:\wsl\home\kinno-7010\Research\RadioData\Wind\Rawdata\wi_l2_wav_rad2_20220613_v01.cdf")
HF_CDF_PATH = Path(r"F:\wsl\home\kinno-7010\Research\RadioData\HF_plot\Rawdata\it_h1_hf_20220613_v01.cdf")
ASSA_FITS_PATHS = [
    Path(r"F:\wsl\home\kinno-7010\Research\RadioData\e-Callisto\Rawdata\Australia-ASSA_20220613_010001_62.fit"),
    Path(r"F:\wsl\home\kinno-7010\Research\RadioData\e-Callisto\Rawdata\Australia-ASSA_20220613_011501_62.fit"),
    Path(r"F:\wsl\home\kinno-7010\Research\RadioData\e-Callisto\Rawdata\Australia-ASSA_20220613_013001_62.fit"),
    Path(r"F:\wsl\home\kinno-7010\Research\RadioData\e-Callisto\Rawdata\Australia-ASSA_20220613_014501_62.fit"),
    Path(r"F:\wsl\home\kinno-7010\Research\RadioData\e-Callisto\Rawdata\Australia-ASSA_20220613_020001_62.fit"),
    Path(r"F:\wsl\home\kinno-7010\Research\RadioData\e-Callisto\Rawdata\Australia-ASSA_20220613_021501_62.fit"),
    Path(r"F:\wsl\home\kinno-7010\Research\RadioData\e-Callisto\Rawdata\Australia-ASSA_20220613_023001_62.fit"),
    Path(r"F:\wsl\home\kinno-7010\Research\RadioData\e-Callisto\Rawdata\Australia-ASSA_20220613_024500_62.fit"),
    Path(r"F:\wsl\home\kinno-7010\Research\RadioData\e-Callisto\Rawdata\Australia-ASSA_20220613_030000_62.fit"),
    Path(r"F:\wsl\home\kinno-7010\Research\RadioData\e-Callisto\Rawdata\Australia-ASSA_20220613_031500_62.fit"),
    Path(r"F:\wsl\home\kinno-7010\Research\RadioData\e-Callisto\Rawdata\Australia-ASSA_20220613_033000_62.fit"),
]


def plot_dynamic_spectrum(
    ax,
    spectrum: pd.DataFrame,
    output_path: Path | None,
    title: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    log_scale: bool,
    show: bool,
    start_time, 
    end_time,
    min_frequency,
    max_frequency,
    cor_target_time: str,
    pts=None,
    speed_kms: float = 394.0,
    factor: float = 2.8,
    dt_s: float = 1.0,
    seed_point=None,
    extra_predictions=None,
    start_point_time=None,
    start_point_frequency=None,
    end_point_time=None,
    end_point_frequency=None,
) -> None:
    """Generate and optionally save the combined dynamic spectrum figure with Type II prediction overlay.
    
    Plots both Fundamental and Harmonic branches if start_point_time/frequency are provided.
    """
    print("----------------figure export----------------")
    if spectrum.empty:
        raise ValueError("Combined spectrum is empty. Check the time range and input files.")

    time_axis = spectrum.index.to_pydatetime()
    freq_axis = spectrum.columns.to_numpy()
    values = spectrum.to_numpy().T  # shape -> (freq, time)

    # ensure we have a reference to the figure that owns this axis
    fig = ax.figure
    mesh = ax.pcolormesh(
        mdates.date2num(time_axis),
        freq_axis,
        values,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    #------------------------------------------------------------
    # Type II の点
    points_handle = None
    if pts:
        xs = [mdates.date2num(t) for t, _ in pts]
        ys = [f for _, f in pts]
        (points_handle,) = ax.plot(
            xs, ys, "o", color="magenta", ms=10, mec="k", mew=1, alpha=0.95, label="Type II points"
        )

    def resolve_seed_point(
        sp: Tuple[pd.Timestamp | str, float] | None,
    ) -> Tuple[pd.Timestamp, float]:
        if sp is not None:
            return pd.Timestamp(sp[0]), float(sp[1])
        if pts:
            t_last, f_last = pts[-1]
            return pd.Timestamp(t_last), float(f_last)
        default_time = start_time + (end_time - start_time) / 2
        default_freq = 0.5 * (min_frequency + max_frequency)
        return pd.Timestamp(default_time), float(default_freq)

    prediction_handles: List[Any] = []

    # Fundamental と Harmonic の両ブランチを描画
    # seed_point が指定されていない場合は start_point_time/frequency を使用
    if start_point_time is not None and start_point_frequency is not None:
        seed_point = (start_point_time, start_point_frequency)
    
    primary_seed_time, primary_seed_freq = resolve_seed_point(seed_point)
    
    # Fundamental (F) ブランチ: 周波数を 1/2 にする
    f_seed_freq = primary_seed_freq / 2.0
    f_style: Dict[str, Any] = {
        "color": "#A0E4FF",
        "linestyle": "--",
        "alpha": 0.7,
        "label": f"{factor}× Saito 1977 (F)"
    }
    
    line_f = overlay_prediction_fullspan(
        ax,
        start_time=start_time,
        end_time=end_time,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        t_seed=primary_seed_time,
        f_seed_mhz=f_seed_freq,
        speed_kms=speed_kms,
        branch="F",
        factor=factor,
        dt_s=dt_s,
        **f_style,
    )
    if line_f is not None:
        prediction_handles.append(line_f)
    
    # Harmonic (H) ブランチ: 元の周波数を使用
    h_style: Dict[str, Any] = {
        "color": "red",
        "linestyle": "--",
        "alpha": 0.7,
        "label": f"{factor}× Saito 1977 (H)"
    }
    
    line_h = overlay_prediction_fullspan(
        ax,
        start_time=start_time,
        end_time=end_time,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        t_seed=primary_seed_time,
        f_seed_mhz=primary_seed_freq,
        speed_kms=speed_kms,
        branch="H",
        factor=factor,
        dt_s=dt_s,
        **h_style,
    )
    if line_h is not None:
        prediction_handles.append(line_h)

    legend_handles: List[Any] = []
    if points_handle is not None:
        points_handle.set_label(f"Type II points\nconst speed = {speed_kms:.0f} km/s")
        legend_handles.append(points_handle)
    legend_handles.extend(prediction_handles)
    if legend_handles:
        ax.legend(
            legend_handles,
            [handle.get_label() for handle in legend_handles],
            loc="lower right",
            fontsize=12,
        )

    ax.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
    ax.set_ylim(min_frequency, max_frequency)

    # -------------------------
    # 軸スケール（ここで先に確定）
    # -------------------------
    ax.set_ylabel("Frequency [MHz]", fontsize=16)
    if log_scale is not False:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")

    # -------------------------
    # ★追加：y軸（Frequency）の minor tick を 0.5 MHz 刻み
    # ※ linear のときだけ有効（log では等間隔 0.5 MHz は作れません）
    # -------------------------
    if ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        # 目盛（tick mark）をはっきり出す
        ax.tick_params(axis="y", which="minor", length=4)
        # minor tick ラベルも出したい場合は以下を有効化（混雑しやすいので注意）
        # ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
        # ax.tick_params(axis="y", which="minor", labelsize=10)
    else:
        # log のときは 0.5 MHz 等間隔が不可能なので、少なくとも minor tick 自体はONにしておく
        ax.minorticks_on()

    # -------------------------
    # 第2軸（右軸）：Radial distance
    # -------------------------
    secax = ax.secondary_yaxis(
        "right",
        functions=(
            lambda f_mhz: frequency_to_r_saito_factor(f_mhz, branch="F", factor=factor),
            lambda r_rs: r_to_frequency_saito_factor(r_rs, branch="F", factor=factor),
        ),
    )
    secax.set_ylabel(f"Radial distance (F)[R$_\\odot$] ({factor}× Saito 1977)", fontsize=14)
    secax.tick_params(axis="y", labelsize=12)

    # 第2軸（右軸）の目盛は 1.5–6.0 Rs のみに固定（主軸の周波数レンジは維持）
    rmin, rmax = 1.5, 6.0
    secax.set_ylim(rmin, rmax)  # 第2軸の表示範囲を1.5-6.0 Rsに固定

    # メジャー目盛：0.5 Rs刻み
    major_ticks = np.arange(rmin, rmax + 0.01, 0.5)
    secax.set_yticks(major_ticks)

    # マイナー目盛：0.1 Rs刻み
    minor_ticks = np.arange(rmin, rmax + 0.01, 0.1)
    secax.set_yticks(minor_ticks, minor=True)

    # ★（任意だが有用）minor tick を見やすく
    secax.tick_params(axis="y", which="minor", length=4)

    secax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))

    ax.axhline(14.0, color="white", linestyle="--", linewidth=1)
    ax.text(mdates.date2num(end_time), 14.0, "Wind/RAD2", color="white", fontsize=16, ha="right", va="top", fontweight="bold")
    ax.axhline(40.0, color="white", linestyle="--", linewidth=1)
    ax.text(mdates.date2num(end_time), 40.0, "Iitate HF antenna", color="white", fontsize=16, ha="right", va="top", fontweight="bold")
    ax.text(mdates.date2num(end_time), 85.0, "Australia-ASSA", color="white", fontsize=16, ha="right", va="top", fontweight="bold")
    
    # 特定の時間を縦線
    specific_times = [
        ("01:25:00", "Faint CME \nerupted \n01:25 UT ", "right"),
        ("02:00:00", "", "center"),
        ("03:12:00", "Main CME erupted \n03:12 UT ", "right"),
        ("03:25:45", " SRBII start\n 03:25:45 UT", "left"),
    ]
    for time, label, ha in specific_times:
        timestamp = pd.Timestamp(f"{start_time.date()} {time}")
        ax.axvline(mdates.date2num(timestamp), color="magenta", linestyle="--", linewidth=1)
        ax.text(mdates.date2num(timestamp), 85.0, label, color="magenta", fontsize=16, ha=ha, va="top", fontweight="bold")
    
    ax.axvline(mdates.date2num(pd.Timestamp(cor_target_time)), color="cyan", linestyle="-", linewidth=4, label="Coronagraph image time")

    ax.set_ylabel("Frequency [MHz]", fontsize=16)
    if log_scale is not False:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
    ax.set_xlabel("Time [UT]", fontsize=16)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
    ax.set_ylim(min_frequency, max_frequency)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(SecondLocator(interval=60*30))
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.0f}"))
    ax.tick_params(axis="x", rotation=0, labelrotation=0, labelsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="lower right", fontsize=14)


    # if output_path is not None:
    #     fig.savefig(output_path, dpi=100)
    #     print(f"Figure saved to {output_path} \n")

    # if show:
    #     # show the specific figure that contains this axis
    #     try:
    #         fig.show()
    #     except Exception:
    #         plt.show()
    # else:
    #     plt.close(fig)

def plot_sdo_aia_rgb_diff(
    ax_aia,
    datetime_str,
    channel_r_str="211",
    channel_g_str="193",
    channel_b_str="171",
    delta_minutes=2,
    save_path=None,
    xlim_arcsec=None,
    ylim_arcsec=None,
    vmax_gray=0.015,
):
    """
    指定された「現在時刻」と、その delta_minutes 分前の時刻のSDO/AIAデータ (3波長) から
    RGB合成画像を作成し、その「RGB画像同士の差分画像」をWCSベースでプロットする。

    重要:
      - AIAパネルのAxesが通常Axesの場合、内部でWCSAxesに置換してから描画する
      - 目盛りはSolar X/Y (arcsec) を表示し、(0,0)が太陽中心になる
    """
    from astropy.coordinates import SkyCoord

    # -----------------------------
    # 1. 日時文字列のパース（現在時刻）
    # -----------------------------
    dt_cur = parse_datetime_str(datetime_str)
    if dt_cur is None:
        return

    dt_cur = dt_cur.replace(second=0, microsecond=0)
    dt_prev = dt_cur - timedelta(minutes=delta_minutes)

    date_cur = dt_cur.strftime("%Y%m%d")
    time_cur = dt_cur.strftime("%H%M")
    date_prev = dt_prev.strftime("%Y%m%d")
    time_prev = dt_prev.strftime("%H%M")

    channels = {"r": channel_r_str, "g": channel_g_str, "b": channel_b_str}

    # ------------------------------------------------
    # 2. 各時刻ごとに、3波長のMapオブジェクトをロード
    # ------------------------------------------------
    def load_maps_for_time(date_str, time_str):
        maps = {}
        loaded = 0
        for color, ch_str in channels.items():
            wavelength_part_in_fname = ch_str.zfill(4)
            filename = f"AIA{date_str}_{time_str}_{wavelength_part_in_fname}.fits"
            file_path = AIA_BASE_DATA_DIR / ch_str / filename
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

    maps_cur, loaded_cur = load_maps_for_time(date_cur, time_cur)
    maps_prev, loaded_prev = load_maps_for_time(date_prev, time_prev)

    if loaded_cur < 3 or loaded_prev < 3:
        print("エラー: 3波長すべて読み込めませんでした。")
        return

    reference_map = (
        maps_cur["b"] if maps_cur["b"] else maps_cur["g"] if maps_cur["g"] else maps_cur["r"]
    )
    if not reference_map:
        print("エラー: 基準となるMapオブジェクトがありません。")
        return

    def get_map_time(maps_dict):
        for key in ("b", "g", "r"):
            m = maps_dict.get(key)
            if m is not None:
                return m.date
        return None

    time_cur_map = get_map_time(maps_cur)
    time_prev_map = get_map_time(maps_prev)
    time_cur_str = _format_time_str(time_cur_map)
    time_prev_str = _format_time_str(time_prev_map)

    # ------------------------------------------------
    # 3. 差分（DN/s）を作る
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
        if vmax_gray is not None and vmax_gray > 0:
            vmax = vmax_gray
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
    # 4. WCS付きMapとして扱い、AxesをWCSAxesにする
    # ------------------------------------------------
    diff_map = sunpy.map.Map(diff_scalar, reference_map.meta)

    # もし以前この関数がax_aiaをWCSAxesに置換済みなら、それを使う
    if hasattr(ax_aia, "_wcs_replacement_ax") and ax_aia._wcs_replacement_ax in ax_aia.figure.axes:
        ax = ax_aia._wcs_replacement_ax
    else:
        # 通常Axesなら、同じSubplot位置にWCSAxesを作って置換
        if not hasattr(ax_aia, "coords"):
            fig = ax_aia.figure
            try:
                subspec = ax_aia.get_subplotspec()
                ax_aia.remove()
                ax = fig.add_subplot(subspec, projection=diff_map.wcs)
            except Exception:
                pos = ax_aia.get_position()
                ax_aia.remove()
                ax = fig.add_axes(pos, projection=diff_map.wcs)
            # “元のax_aiaハンドル”に、置換後axesへの参照を残す（再呼び出し対策）
            ax_aia._wcs_replacement_ax = ax
        else:
            ax = ax_aia

    # ------------------------------------------------
    # 5. 描画（WCSAxes上に配列をそのまま描く）
    # ------------------------------------------------
    im = ax.imshow(
        diff_scalar,
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
    )

    # 太陽リム/グリッド（WCS整合で描く）
    try:
        diff_map.draw_limb(axes=ax, color="red", linestyle="dashed", linewidth=3)
        diff_map.draw_grid(
            axes=ax,
            grid_spacing=15 * u.deg,
            color="red",
            linestyle="dotted",
            linewidth=2,
            alpha=0.7,
        )
    except Exception as e_draw:
        print(f"警告: 太陽リム/グリッド描画に失敗しました: {e_draw}")

    # 太陽中心マーカー（(0,0) arcsec をピクセルに変換して描く）
    try:
        center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=diff_map.coordinate_frame)
        cx, cy = diff_map.world_to_pixel(center)
        ax.plot(float(cx.value), float(cy.value), marker="+", color="red", markersize=10, markeredgewidth=2)
    except Exception as e_center:
        print(f"警告: 太陽中心マーカーの描画に失敗しました: {e_center}")

    # ------------------------------------------------
    # 6. 描画範囲（arcsec指定）: 太陽中心原点のarcsecをWCSでピクセルに変換
    # ------------------------------------------------
    def arcsec_to_pixel(tx_arcsec, ty_arcsec):
        c = SkyCoord(tx_arcsec * u.arcsec, ty_arcsec * u.arcsec, frame=diff_map.coordinate_frame)
        xpix, ypix = diff_map.world_to_pixel(c)
        return float(xpix.value), float(ypix.value)

    try:
        if xlim_arcsec is not None:
            x1_arc, x2_arc = xlim_arcsec
            x1_pix, _ = arcsec_to_pixel(x1_arc, 0.0)
            x2_pix, _ = arcsec_to_pixel(x2_arc, 0.0)
            ax.set_xlim(x1_pix, x2_pix)

        if ylim_arcsec is not None:
            y1_arc, y2_arc = ylim_arcsec
            _, y1_pix = arcsec_to_pixel(0.0, y1_arc)
            _, y2_pix = arcsec_to_pixel(0.0, y2_arc)
            ax.set_ylim(y1_pix, y2_pix)
    except Exception as e_lim:
        print(f"警告: xlim/ylim の設定に失敗しました: {e_lim}")

    # ------------------------------------------------
    # 7. タイトル & 軸ラベル（Solar X/Y, arcsec）
    # ------------------------------------------------
    # title_str_parts = [
    #     f"SDO/AIA RGB Running Difference: ({channel_r_str}+{channel_g_str}+{channel_b_str}Å)\n",
    #     f"{time_prev_str} → {time_cur_str} UT",
    # ]
    # ax.set_title("\n".join(title_str_parts), fontsize=12, pad=5)
    ax.set_title(f"SDO/AIA RGB Running Difference\n{time_prev_str} → {time_cur_str} UT", fontsize=12, pad=5)

    try:
        ax.coords[0].set_axislabel("Solar X (arcsec)")
        ax.coords[1].set_axislabel("Solar Y (arcsec)")
        ax.coords[0].set_format_unit(u.arcsec)
        ax.coords[1].set_format_unit(u.arcsec)
    except Exception:
        pass

    # 既存スタイル
    ax.tick_params(axis="both", which="major", labelsize=10, direction="in")

    # 保存（standalone時のみ）
    if save_path is not None:
        try:
            fig_for_save = ax.get_figure()
            fig_for_save.subplots_adjust(bottom=0.20)
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig_for_save.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saving AIA figure to: {save_path}")
        except Exception as e:
            print(f"Warning: Could not save AIA figure: {e}")


def main(
        fig_ds, ax_ds, fig_cor, ax_cor, ax_aia,
    start_time,
    end_time,
    min_frequency,
    max_frequency,
    cor_target_time,
    cor_base_time,
    mk4_inner=1.3, mk4_outer_lasco_inner=3.0, lasco_outer=6.0,
    xlim_min=-250, xlim_max=0, ylim_min=-100, ylim_max=200,
) -> None:
    # ----- Configuration section (edit as needed) -----
    cadence = "0.5s"
    polarization = "RH"  # or "LH"
    show_plot = True
    log_scale = True
    cmap = "viridis"


    # -------------------------------------------------

    if start_time >= end_time:
        raise ValueError("Start time must be earlier than end time.")

    wind_times, wind_freqs, wind_values = load_wind_rad2(WIND_CDF_PATH)
    hf_times, hf_freqs, hf_values = load_hf(HF_CDF_PATH, polarization)
    assa_times, assa_freqs, assa_values = load_callisto(ASSA_FITS_PATHS)

    target_index = pd.date_range(start=start_time, end=end_time, freq=cadence)
    if len(target_index) == 0:
        raise ValueError("Target index is empty. Check cadence and time range.")

    time_margin = pd.to_timedelta(cadence)
    extended_range = (start_time - time_margin, end_time + time_margin)

    wind_df = create_dataframe(wind_times, wind_freqs, wind_values, extended_range)
    wind_df = resample_to_grid(wind_df, target_index, cadence)
    wind_df = normalize_by_median(wind_df)
    wind_freq_index = wind_df.columns.astype(float)
    wind_df = wind_df.loc[:, (wind_freq_index >= min_frequency) & (wind_freq_index <= max_frequency)]

    hf_df = create_dataframe(hf_times, hf_freqs, hf_values, extended_range)
    hf_df = resample_to_grid(hf_df, target_index, cadence)
    hf_df = normalize_by_median(hf_df)
    hf_freq_index = hf_df.columns.astype(float)
    hf_df = hf_df.loc[:, (hf_freq_index >= min_frequency) & (hf_freq_index <= max_frequency)]

    assa_df = create_dataframe(assa_times, assa_freqs, assa_values, extended_range)
    assa_df = resample_to_grid(assa_df, target_index, cadence)
    assa_df = normalize_by_median(assa_df)
    assa_freq_index = assa_df.columns.astype(float)
    assa_df = assa_df.loc[:, (assa_freq_index >= min_frequency) & (assa_freq_index <= max_frequency)]

    combined = combine_spectra([wind_df, hf_df, assa_df])

    frequency_index = combined.columns.astype(float)
    freq_mask = (frequency_index >= min_frequency) & (frequency_index <= max_frequency)
    combined = combined.loc[:, freq_mask]

    title = "Dynamic Spectrum; Wind/RAD2 (1-14 MHz) + HF antenna (14-40 MHz) + Australia-ASSA (40-85 MHz)"
    if combined.empty:
        raise ValueError("No data remains after applying the frequency bounds.")

    # data_output_path = Path(f"/mnt/d/wsl/home/kinno-7010/Research/RadioData/combine/wind_hf_assa_dynamic_spectrum_{start_time.strftime('%Y-%m-%d_%H%M%S')}_{end_time.strftime('%H%M%S')}.csv")
    # figure_path = Path(f"/mnt/d/wsl/home/kinno-7010/Research/RadioData/combine/wind_hf_assa_dynamic_spectrum_{start_time.strftime('%Y-%m-%d_%H%M%S')}_{end_time.strftime('%H%M%S')}.png")
    data_output_path = Path(r"F:\wsl\home\kinno-7010\Research\RadioData\combine\ds_cor")
    # create a safe filename from the target time (works for str or pd.Timestamp)
    if isinstance(cor_target_time, pd.Timestamp):
        ts_str = cor_target_time.strftime("%Y%m%dT%H%M%S")
    else:
        try:
            ts = pd.Timestamp(cor_target_time)
            ts_str = ts.strftime("%Y%m%dT%H%M%S")
        except Exception:
            ts_str = str(cor_target_time).replace(":", "")
    figure_path = data_output_path / f"ds_cor_{ts_str}.png"
    data_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    # export_dataframe(combined, data_output_path)
    
    plot_dynamic_spectrum(ax_ds,
        combined,
        figure_path,
        title=title,
        cmap=cmap,
        vmin=1.0,
        vmax=1.1,
        log_scale=log_scale,
        show=show_plot,
        start_time=start_time,
        end_time=end_time,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        cor_target_time=cor_target_time,
        speed_kms=394.0,
        factor=2.8,
        dt_s=1.0,
        start_point_time=pd.Timestamp("2022-06-13T03:25:30"),
        start_point_frequency=35.0,
        end_point_time=pd.Timestamp("2022-06-13T03:31:20"),
        end_point_frequency=28.0,
    )

    create_single_diff_from_time_image(ax_cor, cor_target_time, cor_base_time, mk4_inner=mk4_inner, mk4_outer_lasco_inner=mk4_outer_lasco_inner, lasco_outer=lasco_outer, xlim_min=xlim_min, xlim_max=xlim_max, ylim_min=ylim_min, ylim_max=ylim_max)

    # AIA RGB difference plot (right side, same time as coronagraph)
    try:
        cor_target_time_str = cor_target_time.strftime("%Y-%m-%d %H:%M:%S")
        xlim_arcsec = (-1240.0, -100.0)
        ylim_arcsec = (-500.0, 1240.0)
        plot_sdo_aia_rgb_diff(
            ax_aia,
            cor_target_time_str,
            channel_r_str="193",
            channel_g_str="193",
            channel_b_str="193",
            delta_minutes=10,
            save_path=None,
            xlim_arcsec=xlim_arcsec,
            ylim_arcsec=ylim_arcsec,
            vmax_gray=None,
        )
    except Exception as e:
        print(f"WARNING: AIA RGB diff plot failed: {e}")
        ax_aia.text(0.5, 0.5, f"AIA plot error: {str(e)[:50]}", transform=ax_aia.transAxes, ha="center", va="center", fontsize=10)


def create_figure(ax_ds_width, ax_ds_height, ax_cor_height, ax_cor_width, ax_aia_height, ax_aia_width):
    """図を作成し、サブプロットの位置を調整する"""
    # レイアウト: 1段目 (ds全幅)、2段目 (cor左側、aia右側)
    total_width = ax_ds_width + ax_aia_width
    total_height = ax_ds_height + max(ax_cor_height, ax_aia_height)
    fig = plt.figure(figsize=(total_width, total_height))
    
    # GridSpec: 2行2列
    # 1行目：ds全幅、2行目：cor左側、aia右側
    gs_main = fig.add_gridspec(2, 2, width_ratios=[ax_ds_width, ax_aia_width], 
                               height_ratios=[ax_ds_height, max(ax_cor_height, ax_aia_height)],
                                wspace=-0.4, hspace=0.18)
    
    # 1段目：ds（全幅）
    ax_ds = fig.add_subplot(gs_main[0, :])  # 0行、全列
    
    # 2段目：cor（左）、aia（右）
    ax_cor = fig.add_subplot(gs_main[1, 0])  # 1行、0列
    ax_aia = fig.add_subplot(gs_main[1, 1])  # 1行、1列
    
    return fig, ax_ds, ax_cor, ax_aia


def process_time_range(cor_time_list, ds_start_time, ds_end_time, ds_min_freq, ds_max_freq,
                       ax_ds_width, ax_ds_height, ax_cor_height, ax_cor_width, ax_aia_height, ax_aia_width, output_dir,
                       mk4_inner, mk4_outer_lasco_inner, lasco_outer,
                       xlim_min, xlim_max, ylim_min, ylim_max):
    """時刻リストの各時刻に対してプロットを実行"""
    for idx, cor_target_time in enumerate(cor_time_list):
        print(f"\n処理中: {idx+1}/{len(cor_time_list)} - {cor_target_time}")
        
        fig, ax_ds, ax_cor, ax_aia = create_figure(ax_ds_width, ax_ds_height, ax_cor_height, ax_cor_width, ax_aia_height, ax_aia_width)
        cor_base_time = cor_target_time - pd.Timedelta(minutes=10)
        
        main(fig, ax_ds, fig, ax_cor, ax_aia,
            start_time=ds_start_time,
            end_time=ds_end_time,
            min_frequency=ds_min_freq,
            max_frequency=ds_max_freq,
            cor_target_time=cor_target_time,
            cor_base_time=cor_base_time,
            mk4_inner=mk4_inner,
            mk4_outer_lasco_inner=mk4_outer_lasco_inner,
            lasco_outer=lasco_outer,
            xlim_max=xlim_max,
            xlim_min=xlim_min,
            ylim_min=ylim_min,
            ylim_max=ylim_max,
        )
        
        fig.tight_layout()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"ds_cor_{cor_target_time.strftime('%Y%m%dT%H%M%S')}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        plt.close('all')
        gc.collect()


if __name__ == "__main__":
    # 設定値
    ax_ds_height = 10
    ax_ds_width = 12
    ax_cor_height = 10
    ax_cor_width = 10
    ax_aia_height = 10
    ax_aia_width = 10
    
    ds_start_time = pd.Timestamp("2022-06-13 01:00:00")
    ds_end_time = pd.Timestamp("2022-06-13 05:00:00")
    ds_min_frequency = 1.0
    ds_max_frequency = 86.0
    
    output_dir = Path(r"F:\wsl\home\kinno-7010\Research\RadioData\combine\ds_cor")
    
    # # 時間帯1: 01:00 - 03:13 (mk4_inner=1.4)
    # cor_start_time_1 = pd.Timestamp("2022-06-13T03:08:00")
    # cor_end_time_1 = pd.Timestamp("2022-06-13T03:13:00")
    # cor_target_time_list_1 = pd.date_range(start=cor_start_time_1, end=cor_end_time_1, freq='2min')
    
    # print(f"時間帯1 プロット対象時刻数: {len(cor_target_time_list_1)}")
    # process_time_range(cor_target_time_list_1, ds_start_time, ds_end_time, ds_min_frequency, ds_max_frequency,
    #                    ax_ds_width, ax_ds_height, ax_cor_height, ax_cor_width, ax_aia_height, ax_aia_width, output_dir,
    #                    mk4_inner=1.4, mk4_outer_lasco_inner=2.2, lasco_outer=7.0, xlim_min=-300, xlim_max=0, ylim_min=-100, ylim_max=250)
    
    # 時間帯2: 03:14 - 05:01 (mk4_inner=1.4)
    # cor_start_time_2 = pd.Timestamp("2022-06-13T03:44:00")
    # cor_end_time_2 = pd.Timestamp("2022-06-13T04:00:00")
    # cor_target_time_list_2 = pd.date_range(start=cor_start_time_2, end=cor_end_time_2, freq='2min')
    
    # print(f"時間帯2 プロット対象時刻数: {len(cor_target_time_list_2)}")
    # process_time_range(cor_target_time_list_2, ds_start_time, ds_end_time, ds_min_frequency, ds_max_frequency,
    #                    ax_ds_width, ax_ds_height, ax_cor_height, ax_cor_width, ax_aia_height, ax_aia_width, output_dir,
    #                    mk4_inner=1.4, mk4_outer_lasco_inner=3.0, lasco_outer=7.0, xlim_min=-300, xlim_max=0, ylim_min=-100, ylim_max=250)
    
    # 時間帯3: 04:01 - 05:01 (mk4_inner=1.6)
    cor_start_time_3 = pd.Timestamp("2022-06-13T03:44:10")
    cor_end_time_3 = pd.Timestamp("2022-06-13T05:01:00")
    cor_target_time_list_3 = pd.date_range(start=cor_start_time_3, end=cor_end_time_3, freq='2min')
    
    print(f"時間帯3 プロット対象時刻数: {len(cor_target_time_list_3)}")
    process_time_range(cor_target_time_list_3, ds_start_time, ds_end_time, ds_min_frequency, ds_max_frequency,
                       ax_ds_width, ax_ds_height, ax_cor_height, ax_cor_width, ax_aia_height, ax_aia_width, output_dir,
                       mk4_inner=1.4, mk4_outer_lasco_inner=3.0, lasco_outer=7.0, xlim_min=-512, xlim_max=0, ylim_min=-300, ylim_max=300)
    
    print("\n全てのプロットが完了しました")

