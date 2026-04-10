import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import Angle
import gc
import sunpy
from reproject import reproject_interp
from scipy.ndimage import map_coordinates
from sunpy.map import Map
from config import *
import config
from astropy.visualization import ImageNormalize, LinearStretch
from matplotlib.colors import Normalize

from integrated_analysis import (
    get_data_list,
    scan_multi_directories,
    determine_aia193_diff_ranges,
    create_fully_corrected_lasco_map,
    select_by_midpoint,
    _as_path_list,
    calculate_r_map,
    combine_corona_data
)

def save_cor_csv_file(ax, target_time_str: str, delta_time: int, mk4_inner=1.3, mk4_outer_lasco_inner=3.0, lasco_outer=6.0, xlim_min=-250, xlim_max=0, ylim_min=-100, ylim_max=200):
    """2分前ベース差分統合画像を描画し、その描画結果をCSVとしても保存する関数"""
    
    # sunpyの警告を抑制
    import warnings
    from sunpy.util.exceptions import SunpyMetadataWarning
    warnings.filterwarnings('ignore', category=SunpyMetadataWarning)
    
    # out_dir_str = "/mnt/d/wsl/home/kinno-7010/Research_data/SDO_Mk4_SOHO/diff"
    # --- 1. 時刻パースとデータリスト取得 ---
    target_time_obj = Time(target_time_str)
    if isinstance(delta_time, Time):
        base_time_obj = delta_time
    elif isinstance(delta_time, (int, float)):
        base_time_obj = target_time_obj - delta_time * u.min
    else:
        base_time_obj = Time(delta_time)
    
    # LASCOの固定ベース時刻を定義
    lasco_base_time = Time(target_time_str)
    
    # スキャン範囲を計算（より広い範囲で LASCO ファイルを探す）
    # Time オブジェクトのリストを作成してmin/maxを計算
    time_list = [target_time_obj, base_time_obj]
    earliest_time = min(time_list)
    latest_time = max([target_time_obj, base_time_obj])
    
    # LASCO ファイルをより広い範囲で探すため、±3時間のマージンを設定
    scan_start = earliest_time - 1*u.hour
    scan_end = latest_time + 1*u.hour
    
    # out_dir = Path(out_dir_str)
    
    print(f"2分前ベース差分画像作成: target={target_time_str}, base={base_time_obj.iso}")
    print(f"LASCO用固定基準時刻: {lasco_base_time.iso}")
    print(f"LASCO選択ロジック: 指定時間を超えてから次の画像に移り変わる方式を使用")
    
    # 出力ディレクトリを作成
    # out_dir.mkdir(parents=True, exist_ok=True)
    
    # データリスト取得（スキャン重複回避のため先に実行）
    mk4_list, lasco_list, aia193_list = get_data_list(scan_start, scan_end)

    # Fallback: 指定範囲にLASCOデータが見つからない場合は、
    # 近傍（±1日, ±3日, 最終的には全期間）から最も近い時刻のデータを選択する
    if not lasco_list:
        lasco_dirs = _as_path_list(config.data_folder_dict.get('lasco', ''))
        print("警告: 指定範囲にLASCOデータが見つかりません。近傍から最近接データを探します:")
        for d in lasco_dirs:
            print(f"  -> {d}")
        found = []
        try_ranges = [
            (target_time_obj - 1*u.day, target_time_obj + 1*u.day),
            (target_time_obj - 3*u.day, target_time_obj + 3*u.day),
            (Time('1900-01-01'), Time('2100-01-01'))
        ]
        for s, e in try_ranges:
            try:
                cand = scan_multi_directories(lasco_dirs, s.iso, e.iso, use_cache=False)
                if cand:
                    found = cand
                    break
            except Exception as ex:
                print(f"近傍検索で例外: {ex}")
                continue

        if found:
            # 最も時刻差が小さいものを選択
            closest = min(found, key=lambda mp: abs(mp[0].date - target_time_obj))
            lasco_list = [closest]
            print(f"代替LASCOデータを使用: {closest[0].date.iso} ({closest[1]})")
        else:
            raise ValueError("利用可能なLASCOデータが見つかりません")

    # AIA193差分範囲計算（既取得データを再利用）
    aia_norm_ranges = determine_aia193_diff_ranges(aia193_list, base_time_obj.iso, percentile_range=[10, 90])
    data_dict = {'mk4': mk4_list, 'lasco': lasco_list, 'aia193': aia193_list}

    # LASCO マップのサンプルを読み込んで最大shapeを特定（メモリ効率化）
    lasco_shapes = []
    ref_map = None
    
    # 最初の数個のマップのみでshapeを確認
    sample_size = min(3, len(lasco_list))
    for i, (_, path) in enumerate(lasco_list[:sample_size]):
        try:
            temp_map = sunpy.map.Map(path)
            lasco_shapes.append(temp_map.data.shape)
            if ref_map is None:
                ref_map = temp_map
            else:
                # より大きいshapeを基準にする
                if temp_map.data.shape[0] * temp_map.data.shape[1] > ref_map.data.shape[0] * ref_map.data.shape[1]:
                    ref_map = temp_map
            # メモリ解放
            del temp_map
            gc.collect()
        except Exception as e:
            print(f"警告: LASCOマップ読み込みエラー {path}: {e}")
    
    if not lasco_shapes:
        raise ValueError("利用可能なLASCOデータが見つかりません")
    
    # 最大shapeを決定
    max_ny = max(shape[0] for shape in lasco_shapes)
    max_nx = max(shape[1] for shape in lasco_shapes)
    global_shape = (max_ny, max_nx)
    target_wcs = ref_map.wcs

    # グローバル extent 固定
    def get_params(m):
        px = m.rsun_obs.to_value(u.arcsec) / m.scale.axis1.to_value(u.arcsec/u.pix)
        # ゼロ除算を回避
        if px <= 0:
            print(f"警告: px_per_rsun が無効な値です: {px}")
            px = 1  # デフォルト値を設定
        return dict(nx=m.data.shape[1],
                   ny=m.data.shape[0],
                   cx=m.meta['crpix1']-1,
                   cy=m.meta['crpix2']-1,
                   px_per_rsun=px)

    p_glob = get_params(ref_map)
    extent_global = [
        -p_glob['cx'], p_glob['nx'] - p_glob['cx'],
        -p_glob['cy'], p_glob['ny'] - p_glob['cy']
    ]

    # フレームごとに処理
    def resample_map(m):
        try:
            return m.resample(global_shape * u.pix)
        except AttributeError:
            data, _ = reproject_interp((m.data, m.wcs), target_wcs, global_shape)
            return sunpy.map.Map(data, ref_map.meta)
        
    data_selected_dict, data_selected_base_dict, data_dict_resampled, data_dict_base_resampled, p_dict, diff_dict = {}, {}, {}, {}, {}, {}
    # 各波長・Instrument のマップ取得
    for key, value in data_dict.items():
        # LASCOは指定時間を超えてから次の画像に移り変わる専用関数を使用
        if key == 'lasco':
            # LASCO target選択: target_time より前で最も新しいファイルを target とする
            # （要求: target_time を越すまではデータを切り替えない方式）
            candidates_before = [m for m, _ in value if m.date <= target_time_obj]
            if candidates_before:
                target_map = max(candidates_before, key=lambda m: m.date)
            else:
                # target_time より前のファイルがない場合は最初の利用可能ファイル（将来の最初）を使用
                target_map = min(value, key=lambda mp: mp[0].date)[0]
            data_selected_dict[key] = target_map
            
            # base_time は target_map より前で、最も時刻差が小さい（直前の）ファイルを選択
            base_candidates = [m for m, _ in value if m.date < target_map.date]
            if base_candidates:
                # 複数の候補がある場合は、target より前で最も近いものを選択
                data_selected_base_dict[key] = max(base_candidates, key=lambda m: m.date)
                print(f"LASCO: target より前で最も近いファイルをbase として使用")
            else:
                # target より前のファイルがない場合は target と同じものを使用
                data_selected_base_dict[key] = target_map
                print(f"LASCO: target より前のファイルが見つからないため target と同じファイルを使用")
            
            print(f"LASCO target時刻: {data_selected_dict[key].date.iso}")
            print(f"LASCO base時刻: {data_selected_base_dict[key].date.iso}")
        else:
            data_selected_dict[key], _ = select_by_midpoint(target_time_obj, value)
            data_selected_base_dict[key], _ = select_by_midpoint(base_time_obj, value)
        
        if key == 'lasco':
            print(f"INFO: Processing full correction for LASCO target map ({data_selected_dict[key].date})...")
            data_selected_dict[key] = create_fully_corrected_lasco_map(data_selected_dict[key])
            
            print(f"INFO: Processing full correction for LASCO base map ({data_selected_base_dict[key].date})...")
            data_selected_base_dict[key] = create_fully_corrected_lasco_map(data_selected_base_dict[key])
            print("INFO: LASCO correction complete.")
            
        data_dict_resampled[key] = resample_map(data_selected_dict[key])
        data_dict_base_resampled[key] = resample_map(data_selected_base_dict[key])
        
        p_dict[key] = get_params(data_dict_resampled[key])
        diff_dict[key] = data_dict_resampled[key].data - data_dict_base_resampled[key].data
        
        # メモリ効率化：元のマップは差分計算後に削除
        del data_selected_dict[key], data_selected_base_dict[key]
        gc.collect()
        
    mk4_map, lasco_map, aia193_map = data_dict_resampled.values()
    p_mk4, p_lasco, p_aia = p_dict.values()
    mk4_diff, lasco_diff, aia193_diff = diff_dict.values()

    # K-COR と LASCO の負の値を0に変換
    mk4_diff = np.clip(mk4_diff, a_min=0, a_max=None)
    lasco_diff = np.clip(lasco_diff, a_min=0, a_max=None)

    # 正規化（Mk4 / LASCO）
    mk4_vmin, mk4_vmax = -3.5, 3.5
    print('mk4_vmin', mk4_vmin, 'mk4_vmax', mk4_vmax)
    mk4_norm = ImageNormalize(mk4_diff, vmin=mk4_vmin, vmax=mk4_vmax, stretch=LinearStretch(), clip=True)
    n_mk4 = mk4_norm(mk4_diff)

    lasco_vmin, lasco_vmax = -10, 10
    print('lasco_vmin', lasco_vmin, 'lasco_vmax', lasco_vmax)
    
    lasco_norm = ImageNormalize(lasco_diff, vmin=lasco_vmin, vmax=lasco_vmax, stretch=LinearStretch(), clip=True)
    n_lasco = lasco_norm(lasco_diff)

    # AIA 193 差分画像用正規化
    def normalize_linear_stretch(arr, vmin, vmax):
        norm = ImageNormalize(arr, vmin=vmin, vmax=vmax, stretch=LinearStretch(), clip=True)
        return norm(arr)

    def scale01(a):
        mn, mx = np.nanmin(a), np.nanmax(a)
        return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a)
    
    # AIA193差分画像のみを使用
    aia193_ch = normalize_linear_stretch(aia193_diff, vmin=aia_norm_ranges[0], vmax=aia_norm_ranges[1])
    aia193_scaled = scale01(aia193_ch)

    # 単色画像として使用（グレースケール）
    aia193_image = aia193_scaled

    # 半径マップ・合成
    r_map = calculate_r_map(p_lasco)
    # mk4_inner, mk4_outer_lasco_inner, lasco_outer = 1.4, 3.0, 6.0
    ranges = dict(mk4_inner=mk4_inner, mk4_outer_lasco_inner=mk4_outer_lasco_inner, lasco_outer=lasco_outer)
    composite, imk4, ia = combine_corona_data(
        n_lasco, p_lasco,
        n_mk4, p_mk4,
        aia193_image, p_aia,  # AIA193差分画像を使用
        r_map, ranges
    )
    # MK4/LASCO 合成
    mask_mk4 = (r_map >= ranges['mk4_inner']) & (r_map < ranges['mk4_outer_lasco_inner'])
    mask_lasco = (r_map >= ranges['mk4_outer_lasco_inner']) & (r_map <= ranges['lasco_outer'])
    combined_ml = np.full_like(composite, np.nan)
    combined_ml[mask_mk4] = imk4[mask_mk4]
    combined_ml[mask_lasco] = n_lasco[mask_lasco]

    # ──────────────── 描画 ────────────────
    # 1) AIA 193差分画像を背景として変換
    ny, nx = p_lasco['ny'], p_lasco['nx']
    y_idx, x_idx = np.indices((ny, nx))
    x_norm = (x_idx - p_lasco['cx']) / p_lasco['px_per_rsun']
    y_norm = (y_idx - p_lasco['cy']) / p_lasco['px_per_rsun']
    coords = np.vstack([
        (y_norm * p_aia['px_per_rsun'] + p_aia['cy']).ravel(),
        (x_norm * p_aia['px_per_rsun'] + p_aia['cx']).ravel()
    ])
    
    # AIA193差分画像のみを座標変換
    aia193_background = map_coordinates(aia193_ch, coords, order=1, mode='constant', cval=np.nan).reshape((ny, nx))
    
    # 太陽半径1.3Rs以内で切り取り
    aia193_background[r_map > mk4_inner] = np.nan

    # 背景 AIA193差分画像（専用カラーマップ使用）
    try:
        # グレースケールで表示
        aia193_cmap = plt.cm.gray
    except Exception as e:
        print(f"AIA193カラーマップの読み込みに失敗: {e}")
        # フォールバック：グレースケール
        aia193_cmap = plt.cm.gray
    
    ax.imshow(aia193_background, origin='lower', extent=extent_global, cmap=aia193_cmap, aspect='equal', zorder=0)
    ax.imshow(combined_ml, origin='lower', cmap=plt.cm.seismic,
             norm=Normalize(0,1), extent=extent_global, alpha=0.7, zorder=1)

    # 境界円
    scale = p_lasco['px_per_rsun']
    theta = np.linspace(0, 2*np.pi, 400)
    # 太陽Limb（1 Rsun）
    r_limb = 1.0 * scale
    ax.plot(r_limb*np.cos(theta), r_limb*np.sin(theta),
            ':', color='red', linewidth=2.0)
    for i in range(2, int(ranges['lasco_outer'])+1):
        ax.plot(p_lasco['px_per_rsun']*i*np.cos(theta),
                p_lasco['px_per_rsun']*i*np.sin(theta),
                ':', color='black', linewidth=0.8)

    # 境界円＆凡例
    r1 = ranges['mk4_inner']*scale
    ax.plot(r1*np.cos(theta), r1*np.sin(theta),
            '--',color='yellow',linewidth=1.5,
            label=f"{ranges['mk4_inner']} $R_\\odot$")
    r2 = ranges['mk4_outer_lasco_inner']*scale
    ax.plot(r2*np.cos(theta), r2*np.sin(theta),
            '--',color='cyan',linewidth=1.5,
            label=f"{ranges['mk4_outer_lasco_inner']} $R_\\odot$")

    # 太陽中心を通る30°直線を描画（北側0°から反時計回り）
    theta_line_deg = 152.0
    theta_line_rad = np.radians(theta_line_deg)
    
    # 線の描画範囲（太陽半径単位）
    r_line_min_rsun = 0  # 中心から描画開始
    r_line_max_rsun = ranges['lasco_outer']  # LASCO外側まで線を伸ばす
    
    r_coords_rsun = np.array([r_line_min_rsun, r_line_max_rsun])
    
    # ピクセル座標へ変換
    x_line_pix = r_coords_rsun * p_lasco['px_per_rsun'] * np.cos(theta_line_rad)
    y_line_pix = r_coords_rsun * p_lasco['px_per_rsun'] * np.sin(theta_line_rad)
    
    # 直線を描画
    # ax.plot(x_line_pix, y_line_pix, 
    #         color='red', linestyle='-', linewidth=2, 
    #         label=f'θ={theta_line_deg:.0f}°')

    # 軸範囲を global に固定
    ax.set_xlim(xlim_min, xlim_max); ax.set_ylim(ylim_min, ylim_max)
    # ax.set_xlabel('X [pixel]'); ax.set_ylabel('Y [pixel]'); ax.set_facecolor('gray')
    title_lines = (
        f"SDO/AIA 193 Å: {aia193_map.date.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Mk4: {mk4_map.date.strftime('%H:%M:%S')} | LASCO-C2: {lasco_map.date.strftime('%H:%M:%S')}\n"
        f"Base: {base_time_obj.iso}"
    )
    ax.set_title(title_lines)
    ax.legend(loc='upper right', fontsize=12)

    # 描画に使った結果をCSVとして保存
    x_plot = x_idx - p_lasco['cx']
    y_plot = y_idx - p_lasco['cy']
    x_rsun = x_plot / p_lasco['px_per_rsun']
    y_rsun = y_plot / p_lasco['px_per_rsun']

    display_mask = (x_plot >= xlim_min) & (x_plot <= xlim_max) & (y_plot >= ylim_min) & (y_plot <= ylim_max)

    source_region = np.full(r_map.shape, 'outside', dtype=object)
    source_region[r_map < ranges['mk4_inner']] = 'aia193'
    source_region[mask_mk4] = 'mk4'
    source_region[mask_lasco] = 'lasco'

    csv_df = pd.DataFrame({
        'x_pix': x_plot[display_mask].ravel(),
        'y_pix': y_plot[display_mask].ravel(),
        'x_rsun': x_rsun[display_mask].ravel(),
        'y_rsun': y_rsun[display_mask].ravel(),
        'r_rsun': r_map[display_mask].ravel(),
        'aia193_background': aia193_background[display_mask].ravel(),
        'combined_ml': combined_ml[display_mask].ravel(),
        'mk4_component': imk4[display_mask].ravel(),
        'lasco_component': n_lasco[display_mask].ravel(),
        'source_region': source_region[display_mask].ravel(),
    })

    output_dir = Path('/mnt/d/wsl/home/kinno-7010/Research_data/SDO_Mk4_SOHO/cor_csv')
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_filename = f"{mk4_map.date.strftime('%Y%m%d-%H%M%S')}_cor.csv"
    csv_path = output_dir / csv_filename

    mk4_time_str = mk4_map.date.strftime('%Y-%m-%d %H:%M:%S')
    aia193_time_str = aia193_map.date.strftime('%Y-%m-%d %H:%M:%S')
    lasco_time_str = lasco_map.date.strftime('%Y-%m-%d %H:%M:%S')

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        f.write(f"# mk4_time: {mk4_time_str}\n")
        f.write(f"# aia193_time: {aia193_time_str}\n")
        f.write(f"# lasco_c2_time: {lasco_time_str}\n")
        csv_df.to_csv(f, index=False)

    print(f"CSV saved to: {csv_path}")

    # 描画結果もPNGとして保存（Aggバックエンドでも確認できるようにする）
    png_output_dir = Path('/mnt/d/wsl/home/kinno-7010/Research_data/SDO_Mk4_SOHO/cor_plot')
    png_output_dir.mkdir(parents=True, exist_ok=True)
    png_filename = f"{mk4_map.date.strftime('%Y%m%d-%H%M%S')}_cor.png"
    png_path = png_output_dir / png_filename
    ax.figure.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {png_path}")

    return {
        'params_lasco': p_lasco,
        'params_mk4': p_mk4,
        'lasco_map': lasco_map,
        'mk4_map': mk4_map,
        'csv_path': str(csv_path),
        'png_path': str(png_path)
    }



if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(10,10), dpi=300)
    target_time_str = "2022-06-13T03:20:00"
    save_cor_csv_file(ax, target_time_str, delta_time=10, mk4_inner=1.4, mk4_outer_lasco_inner=3.0, lasco_outer=6.0, xlim_min=-250, xlim_max=0, ylim_min=-100, ylim_max=200)
    plt.show()
