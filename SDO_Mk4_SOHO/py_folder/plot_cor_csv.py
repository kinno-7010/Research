import numpy as np
from astropy.time import Time
from astropy import units as u
import pandas as pd
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# 修正前:
# from integrated_analysis import _grid_from_csv, _estimate_px_per_rsun



def _select_cor_csv_by_time(target_time_obj, csv_dir="/mnt/d/wsl/home/kinno-7010/Research_data/SDO_Mk4_SOHO/cor_csv"):
    """target_time に最も近い <YYYYMMDD-HHMMSS>_cor.csv を選択する。"""
    csv_dir = Path(csv_dir)
    csv_files = sorted(csv_dir.glob("*_cor.csv"))

    if not csv_files:
        raise FileNotFoundError(f"CSV ファイルが見つかりません: {csv_dir}")

    candidates = []
    for csv_path in csv_files:
        stem = csv_path.stem  # 例: 20220613-032500_cor
        time_part = stem.replace("_cor", "")
        try:
            dt = datetime.strptime(time_part, "%Y%m%d-%H%M%S")
            csv_time_obj = Time(dt)
            candidates.append((abs((csv_time_obj - target_time_obj).sec), csv_path, csv_time_obj))
        except ValueError:
            continue

    if not candidates:
        raise ValueError(f"時刻形式に一致する CSV が見つかりません: {csv_dir}")

    _, best_path, best_time = min(candidates, key=lambda x: x[0])
    return best_path, best_time

def _read_csv_metadata(csv_path):
    """CSV 先頭のコメント行からメタデータを読む。"""
    meta = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            key, value = line[1:].split(":", 1)
            meta[key.strip()] = value.strip()
    return meta

def _grid_from_csv(df, value_col):
    """
    CSV の x_pix, y_pix と value_col から 2D 配列と extent を復元する。
    """
    required = ["x_pix", "y_pix", value_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"_grid_from_csv に必要な列がありません: {missing}")

    x_vals = np.sort(df["x_pix"].dropna().unique())
    y_vals = np.sort(df["y_pix"].dropna().unique())

    if len(x_vals) == 0 or len(y_vals) == 0:
        raise ValueError("x_pix または y_pix が空です。")

    x_to_i = {x: i for i, x in enumerate(x_vals)}
    y_to_i = {y: i for i, y in enumerate(y_vals)}

    grid = np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)

    for row in df[["x_pix", "y_pix", value_col]].itertuples(index=False):
        x_pix, y_pix, val = row
        if pd.isna(x_pix) or pd.isna(y_pix):
            continue
        ix = x_to_i.get(x_pix)
        iy = y_to_i.get(y_pix)
        if ix is not None and iy is not None:
            grid[iy, ix] = val

    extent = [x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()]
    return grid, extent

def _estimate_px_per_rsun(df):
    """
    CSV 内の x_pix/x_rsun または y_pix/y_rsun から px_per_rsun を推定する。
    """
    candidates = []

    if {"x_pix", "x_rsun"}.issubset(df.columns):
        mask = np.isfinite(df["x_pix"].to_numpy(dtype=float)) & np.isfinite(df["x_rsun"].to_numpy(dtype=float))
        x_pix = df.loc[mask, "x_pix"].to_numpy(dtype=float)
        x_rsun = df.loc[mask, "x_rsun"].to_numpy(dtype=float)
        valid = np.abs(x_rsun) > 0
        if np.any(valid):
            candidates.extend(np.abs(x_pix[valid] / x_rsun[valid]))

    if {"y_pix", "y_rsun"}.issubset(df.columns):
        mask = np.isfinite(df["y_pix"].to_numpy(dtype=float)) & np.isfinite(df["y_rsun"].to_numpy(dtype=float))
        y_pix = df.loc[mask, "y_pix"].to_numpy(dtype=float)
        y_rsun = df.loc[mask, "y_rsun"].to_numpy(dtype=float)
        valid = np.abs(y_rsun) > 0
        if np.any(valid):
            candidates.extend(np.abs(y_pix[valid] / y_rsun[valid]))

    candidates = np.array(candidates, dtype=float)
    candidates = candidates[np.isfinite(candidates)]

    if candidates.size == 0:
        raise ValueError("px_per_rsun を推定できませんでした。x_rsun/y_rsun を確認してください。")

    return float(np.nanmedian(candidates))

def _load_lasco_map_from_metadata(meta, fallback_time_obj):
    """CSV コメントの lasco_c2_time から最も近い LASCO map を復元する。"""
    import config
    from integrated_analysis import scan_multi_directories, _as_path_list

    lasco_time_str = meta.get("lasco_c2_time")
    if lasco_time_str is not None:
        lasco_time_obj = Time(lasco_time_str)
    else:
        lasco_time_obj = fallback_time_obj

    lasco_dirs = _as_path_list(config.data_folder_dict.get("lasco", ""))
    if not lasco_dirs:
        raise FileNotFoundError("config.data_folder_dict['lasco'] が見つかりません。")

    found = []
    try_ranges = [
        (lasco_time_obj - 30 * u.min, lasco_time_obj + 30 * u.min),
        (lasco_time_obj - 3 * u.hour, lasco_time_obj + 3 * u.hour),
        (lasco_time_obj - 1 * u.day, lasco_time_obj + 1 * u.day),
        (Time("1900-01-01"), Time("2100-01-01")),
    ]

    for s, e in try_ranges:
        try:
            cand = scan_multi_directories(lasco_dirs, s.iso, e.iso, use_cache=False)
            if cand:
                found = cand
                break
        except Exception as ex:
            print(f"[WARN] LASCO search failed in range {s.iso} - {e.iso}: {ex}")

    if not found:
        raise FileNotFoundError(f"LASCO map が見つかりませんでした: {lasco_time_obj.iso}")

    closest = min(
        found,
        key=lambda mp: abs((mp[0].date - lasco_time_obj).to_value(u.s))
    )
    return closest[0]

def create_single_diff_from_csv_image(ax, target_time_str: str, delta_time: int,
                                      mk4_inner=1.3, mk4_outer_lasco_inner=3.0, lasco_outer=6.0,
                                      xlim_min=-250, xlim_max=0, ylim_min=-100, ylim_max=200,
                                      aia_time=None, lasco_time=None,
                                      mk4_vmin=-4.0, mk4_vmax=4.0,
                                      lasco_vmin=-10.0, lasco_vmax=10.0,
                                      aia_vmin=None, aia_vmax=None):
    """CSV から 2分前ベース差分統合画像を再描画する関数。"""

    target_time_obj = Time(target_time_str)
    if isinstance(delta_time, Time):
        base_time_obj = delta_time
    elif isinstance(delta_time, (int, float)):
        base_time_obj = target_time_obj - delta_time * u.min
    else:
        base_time_obj = Time(delta_time)

    csv_path, csv_time_obj = _select_cor_csv_by_time(target_time_obj)
    meta = _read_csv_metadata(csv_path)
    df = pd.read_csv(csv_path, comment="#")

    required_cols = [
        "x_pix", "y_pix", "x_rsun", "y_rsun", "r_rsun",
        "aia193_background"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV に必要な列がありません: {missing}")

    numeric_cols = [
        "x_pix", "y_pix", "x_rsun", "y_rsun", "r_rsun",
        "aia193_background", "combined_ml",
        "mk4_component", "lasco_component",
        "mk4_diff_raw", "lasco_diff_raw",
        "aia193_diff_raw", "aia193_background_raw"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def normalize_linear_stretch(arr, vmin=None, vmax=None):
        arr = np.asarray(arr, dtype=float)
        out = np.full_like(arr, np.nan, dtype=float)

        finite = np.isfinite(arr)
        if not np.any(finite):
            return out

        if vmin is None:
            vmin = np.nanmin(arr)
        if vmax is None:
            vmax = np.nanmax(arr)

        if vmax <= vmin:
            out[finite] = 0.0
            return out

        out[finite] = (arr[finite] - vmin) / (vmax - vmin)
        out[finite] = np.clip(out[finite], 0, 1)
        return out

    # 基本グリッド復元
    r_map, _ = _grid_from_csv(df, "r_rsun")
    px_per_rsun = _estimate_px_per_rsun(df)

    if "aia193_diff_raw" in df.columns:
        aia_base_grid, extent_global = _grid_from_csv(df, "aia193_diff_raw")
    elif "aia193_background_raw" in df.columns:
        aia_base_grid, extent_global = _grid_from_csv(df, "aia193_background_raw")
    else:
        aia_base_grid, extent_global = _grid_from_csv(df, "aia193_background")

    # CSV コメント行のメタデータを優先し、必要なら引数で上書き
    mk4_time = meta.get("mk4_time", csv_time_obj.strftime("%Y-%m-%d %H:%M:%S"))
    aia_time = aia_time if aia_time is not None else meta.get("aia193_time", target_time_obj.strftime("%Y-%m-%d %H:%M:%S"))
    lasco_time = lasco_time if lasco_time is not None else meta.get("lasco_c2_time", csv_time_obj.strftime("%Y-%m-%d %H:%M:%S"))

    # spheroid overlay 用に LASCO map を復元
    lasco_map = _load_lasco_map_from_metadata(meta, csv_time_obj)

    # ---- AIA 背景を再正規化 ----
    aia193_background = normalize_linear_stretch(aia_base_grid, aia_vmin, aia_vmax)

    # ---- K-COR を再正規化 ----
    if "mk4_diff_raw" in df.columns:
        mk4_raw, _ = _grid_from_csv(df, "mk4_diff_raw")
        mk4_raw = np.clip(mk4_raw, a_min=0, a_max=None)
        mk4_display = normalize_linear_stretch(mk4_raw, mk4_vmin, mk4_vmax)
    elif "mk4_component" in df.columns:
        mk4_component, _ = _grid_from_csv(df, "mk4_component")
        mk4_display = normalize_linear_stretch(mk4_component, mk4_vmin, mk4_vmax)
    else:
        raise ValueError("CSV に mk4_diff_raw も mk4_component もありません。")

    # ---- LASCO を再正規化 ----
    if "lasco_diff_raw" in df.columns:
        lasco_raw, _ = _grid_from_csv(df, "lasco_diff_raw")
        lasco_raw = np.clip(lasco_raw, a_min=0, a_max=None)
        lasco_display = normalize_linear_stretch(lasco_raw, lasco_vmin, lasco_vmax)
    elif "lasco_component" in df.columns:
        lasco_component, _ = _grid_from_csv(df, "lasco_component")
        lasco_display = normalize_linear_stretch(lasco_component, lasco_vmin, lasco_vmax)
    else:
        raise ValueError("CSV に lasco_diff_raw も lasco_component もありません。")

    # ---- 半径で再合成 ----
    ranges = dict(mk4_inner=mk4_inner,
                  mk4_outer_lasco_inner=mk4_outer_lasco_inner,
                  lasco_outer=lasco_outer)

    mask_mk4 = (r_map >= ranges['mk4_inner']) & (r_map < ranges['mk4_outer_lasco_inner'])
    mask_lasco = (r_map >= ranges['mk4_outer_lasco_inner']) & (r_map <= ranges['lasco_outer'])

    combined_ml = np.full_like(r_map, np.nan, dtype=float)
    combined_ml[mask_mk4] = mk4_display[mask_mk4]
    combined_ml[mask_lasco] = lasco_display[mask_lasco]

    # AIA 背景画像
    try:
        aia193_cmap = plt.cm.gray
    except Exception as e:
        print(f"AIA193カラーマップの読み込みに失敗: {e}")
        aia193_cmap = plt.cm.gray

    ax.imshow(aia193_background, origin='lower', extent=extent_global,
              cmap=aia193_cmap, aspect='equal', zorder=0)
    ax.imshow(combined_ml, origin='lower', cmap=plt.cm.seismic,
              norm=Normalize(0, 1), extent=extent_global, alpha=0.7, zorder=1)

    # 境界円
    scale = px_per_rsun
    theta = np.linspace(0, 2 * np.pi, 400)

    # 太陽Limb（1 Rsun）
    r_limb = 1.0 * scale
    ax.plot(r_limb * np.cos(theta), r_limb * np.sin(theta),
            ':', color='red', linewidth=2.0)

    for i in range(2, int(ranges['lasco_outer']) + 1):
        ax.plot(scale * i * np.cos(theta),
                scale * i * np.sin(theta),
                ':', color='black', linewidth=0.8)

    # 境界円＆凡例
    r1 = ranges['mk4_inner'] * scale
    ax.plot(r1 * np.cos(theta), r1 * np.sin(theta),
            '--', color='yellow', linewidth=1.5,
            label=f"{ranges['mk4_inner']} $R_\\odot$")
    r2 = ranges['mk4_outer_lasco_inner'] * scale
    ax.plot(r2 * np.cos(theta), r2 * np.sin(theta),
            '--', color='cyan', linewidth=1.5,
            label=f"{ranges['mk4_outer_lasco_inner']} $R_\\odot$")

    # 太陽中心を通る30°直線を描画（元コードではコメントアウトされているためここでも非表示）
    theta_line_deg = 152.0
    theta_line_rad = np.radians(theta_line_deg)
    r_line_min_rsun = 0
    r_line_max_rsun = ranges['lasco_outer']
    r_coords_rsun = np.array([r_line_min_rsun, r_line_max_rsun])
    x_line_pix = r_coords_rsun * scale * np.cos(theta_line_rad)
    y_line_pix = r_coords_rsun * scale * np.sin(theta_line_rad)
    # ax.plot(x_line_pix, y_line_pix, color='red', linestyle='-', linewidth=2,
    #         label=f'θ={theta_line_deg:.0f}°')

    # 軸範囲を元コードと同じ設定に固定
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)

    title_lines = (
        f"SDO/AIA 193 Å: {aia_time}\n"
        f"Mk4: {mk4_time.split()[-1]} | LASCO-C2: {lasco_time.split()[-1]}\n"
        f"Base: {base_time_obj.iso}"
    )
    ax.set_title(title_lines)
    ax.legend(loc='upper right', fontsize=12)

    return {
        'csv_path': str(csv_path),
        'csv_time': csv_time_obj,
        'px_per_rsun': px_per_rsun,
        'mk4_time': mk4_time,
        'aia_time': aia_time,
        'lasco_time': lasco_time,
        'params_lasco': {'px_per_rsun': px_per_rsun},
        'lasco_map': lasco_map,
    }
    
            
if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(10,10), dpi=300)
    target_time_str = "2022-06-13T03:20:00"
    create_single_diff_from_csv_image(
        ax, target_time_str, delta_time=10,
        mk4_inner=1.4, mk4_outer_lasco_inner=3.0, lasco_outer=6.0,
        xlim_min=-250, xlim_max=0, ylim_min=-100, ylim_max=200,
        aia_time=None, lasco_time=None
    )
    output_path = Path(f"/mnt/d/wsl/home/kinno-7010/Research_data/SDO_Mk4_SOHO/cor_plot/diff_from_csv_image_{target_time_str.replace(':', '')}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    plt.show()