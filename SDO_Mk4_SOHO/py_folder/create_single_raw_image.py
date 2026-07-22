# sunpyの警告を抑制
import warnings
from sunpy.util.exceptions import SunpyMetadataWarning
warnings.filterwarnings('ignore', category=SunpyMetadataWarning)

import config
import numpy as np
from astropy.time import Time
import astropy.units as u
from pathlib import Path
import matplotlib.pyplot as plt
import sunpy.map
from astropy.visualization import ImageNormalize, LinearStretch
from matplotlib.colors import Normalize
import gc
from scipy.ndimage import map_coordinates
from reproject import reproject_interp

out_dir_str = "/mnt/d/wsl/home/kinno-7010/Research_data/SDO_Mk4_SOHO/raw"
# --- 1. 時刻パースとデータリスト取得 ---

out_dir = Path(out_dir_str)

# print(f"生データ統合画像作成: target={target_time_str}")

# 出力ディレクトリを作成
# out_dir.mkdir(parents=True, exist_ok=True)

# integrated_analysis.pyから必要な関数をインポート
from integrated_analysis import (
    get_data_list, create_fully_corrected_lasco_map, 
    select_by_midpoint, calculate_r_map, combine_corona_data
)

def plot_multi_degree_lines(ax, angles_deg, r_ranges: dict, params_ref: dict, extent_global):
    """
    角度リストに従って複数のラインを描画。プロファイル計算は行わず幾何学線のみ。
    """
    try:
        cmap = plt.cm.get_cmap("viridis")
        norm = plt.Normalize(vmin=min(angles_deg), vmax=max(angles_deg))
        scale = params_ref['px_per_rsun']
        max_r_pix = max(abs(extent_global[0]), abs(extent_global[1]), abs(extent_global[2]), abs(extent_global[3]))
        max_r = max_r_pix / scale
        r_line = np.linspace(0, max_r, 300)
        for th in angles_deg:
            angle_rad = np.deg2rad(th)
            x_vals = r_line * scale * np.cos(angle_rad)
            y_vals = r_line * scale * np.sin(angle_rad)
            color = cmap(norm(th))
            ax.plot(x_vals, y_vals, color=color, linewidth=1.8, alpha=0.9, label=f"θ={th:.0f}°")
    except Exception as e:
        print(f"Multi-angle guideline draw error: {e}")


def create_single_raw_image(ax, target_time_str: str):
    """生データ統合画像作成関数（差分なし）"""

    target_time_obj = Time(target_time_str)
    scan_start = target_time_obj - 20*u.min
    scan_end = target_time_obj + 20*u.min
    # データリスト取得
    mk4_list, lasco_list, aia193_list = get_data_list(scan_start, scan_end)
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
        
    data_selected_dict, data_dict_resampled, p_dict = {}, {}, {}
    # 各波長・Instrument のマップ取得（生データのみ）
    for key, value in data_dict.items():
        data_selected_dict[key], _ = select_by_midpoint(target_time_obj, value)
        
        if key == 'lasco':
            print(f"INFO: Processing full correction for LASCO raw map ({data_selected_dict[key].date})...")
            data_selected_dict[key] = create_fully_corrected_lasco_map(data_selected_dict[key])
            print("INFO: LASCO correction complete.")
            
        data_dict_resampled[key] = resample_map(data_selected_dict[key])
        p_dict[key] = get_params(data_dict_resampled[key])
        
        # メモリ効率化：元のマップは処理後に削除
        del data_selected_dict[key]
        gc.collect()
        
    mk4_map, lasco_map, aia193_map = data_dict_resampled.values()
    p_mk4, p_lasco, p_aia = p_dict.values()

    # 正規化（生データ用）
    # MK4生データ正規化
    mk4_vmin, mk4_vmax = -1, 4
    print('mk4_vmin', mk4_vmin, 'mk4_vmax', mk4_vmax)
    mk4_norm = ImageNormalize(mk4_map.data, vmin=mk4_vmin, vmax=mk4_vmax, stretch=LinearStretch(), clip=True)
    n_mk4 = mk4_norm(mk4_map.data)

    # LASCO生データ正規化
    lasco_vmin, lasco_vmax = 60, 380
    print('lasco_vmin', lasco_vmin, 'lasco_vmax', lasco_vmax)
    lasco_norm = ImageNormalize(lasco_map.data, vmin=lasco_vmin, vmax=lasco_vmax, stretch=LinearStretch(), clip=True)
    n_lasco = lasco_norm(lasco_map.data)

    # AIA 193 生データ用正規化
    def normalize_linear_stretch(arr, vmin, vmax):
        norm = ImageNormalize(arr, vmin=vmin, vmax=vmax, stretch=LinearStretch(), clip=True)
        return norm(arr)

    def scale01(a):
        mn, mx = np.nanmin(a), np.nanmax(a)
        return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a)
    
    # AIA193生データを使用
    aia193_vmin, aia193_vmax = np.percentile(aia193_map.data[np.isfinite(aia193_map.data)], [1, 99])
    aia193_ch = normalize_linear_stretch(aia193_map.data, vmin=aia193_vmin, vmax=aia193_vmax)
    aia193_scaled = scale01(aia193_ch)

    # 単色画像として使用（グレースケール）
    aia193_image = aia193_scaled

    # 半径マップ・合成
    r_map = calculate_r_map(p_lasco)
    ranges = dict(mk4_inner=1.1, mk4_outer_lasco_inner=2.2, lasco_outer=6.0)
    composite, imk4, ia = combine_corona_data(
        n_lasco, p_lasco,
        n_mk4, p_mk4,
        aia193_image, p_aia,  # AIA193生データ画像を使用
        r_map, ranges
    )
    # MK4/LASCO 合成
    mask_mk4 = (r_map >= ranges['mk4_inner']) & (r_map < ranges['mk4_outer_lasco_inner'])
    mask_lasco = (r_map >= ranges['mk4_outer_lasco_inner']) & (r_map <= ranges['lasco_outer'])
    combined_ml = np.full_like(composite, np.nan)
    combined_ml[mask_mk4] = imk4[mask_mk4]
    combined_ml[mask_lasco] = n_lasco[mask_lasco]

    # ──────────────── 描画 ────────────────
    # 1) AIA 193生データ画像を背景として変換
    ny, nx = p_lasco['ny'], p_lasco['nx']
    y_idx, x_idx = np.indices((ny, nx))
    x_norm = (x_idx - p_lasco['cx']) / p_lasco['px_per_rsun']
    y_norm = (y_idx - p_lasco['cy']) / p_lasco['px_per_rsun']
    coords = np.vstack([
        (y_norm * p_aia['px_per_rsun'] + p_aia['cy']).ravel(),
        (x_norm * p_aia['px_per_rsun'] + p_aia['cx']).ravel()
    ])
    
    # AIA193生データ画像のみを座標変換
    aia193_background = map_coordinates(aia193_ch, coords, order=1, mode='constant', cval=np.nan).reshape((ny, nx))
    
    # 太陽半径1.1Rs以内で切り取り
    aia193_background[r_map > 1.1] = np.nan

    # 背景 AIA193生データ画像（専用カラーマップ使用）
    try:
        # sunpyのAIA193専用カラーマップを使用
        import sunpy.visualization.colormaps as cm
        aia193_cmap = cm.cm.sdoaia193
    except Exception as e:
        print(f"AIA193カラーマップの読み込みに失敗: {e}")
        # フォールバック：標準カラーマップ
        aia193_cmap = plt.cm.plasma
    
    ax.imshow(aia193_background, origin='lower', extent=extent_global, cmap=aia193_cmap, aspect='equal', zorder=0)
    ax.imshow(combined_ml, origin='lower', cmap=plt.cm.plasma,
             norm=Normalize(0,1), extent=extent_global, alpha=0.7, zorder=1)

    # 境界円
    scale = p_lasco['px_per_rsun']
    theta = np.linspace(0, 2*np.pi, 400)
    for i in range(2, int(ranges['lasco_outer'])+1):
        ax.plot(p_lasco['px_per_rsun']*i*np.cos(theta),
                p_lasco['px_per_rsun']*i*np.sin(theta),
                ':', color='white', linewidth=0.8)

    # 境界円＆凡例
    r1 = ranges['mk4_inner']*scale
    ax.plot(r1*np.cos(theta), r1*np.sin(theta),
            '-.',color='magenta',linewidth=1.5,
            label=f"{ranges['mk4_inner']} $R_\\odot$")
    r2 = ranges['mk4_outer_lasco_inner']*scale
    ax.plot(r2*np.cos(theta), r2*np.sin(theta),
            '-.',color='green',linewidth=1.5,
            label=f"{ranges['mk4_outer_lasco_inner']} $R_\\odot$")

    # 140–200°を10°刻みでガイドラインを描画
    # angles_deg = np.arange(140.0, 201.0, 10.0)
    # plot_multi_degree_lines(ax, angles_deg, ranges, p_lasco, extent_global)
    
    # 軸範囲を global に固定
    ax.set_xlim(-300, 300); ax.set_ylim(-300, 300)
    # ax.set_xlabel('X [pixel]'); ax.set_ylabel('Y [pixel]'); ax.set_facecolor('gray')
    ax.set_title(
        f"Raw Data | SDO/AIA 193 Å: {aia193_map.date.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"K-Cor: {mk4_map.date.strftime('%H:%M:%S')} | LASCO-C2: {lasco_map.date.strftime('%H:%M:%S')}",
        fontsize=18
    )
    ax.legend(loc='upper right', fontsize=12, framealpha=0.5)
    
    # パラメータ情報を返す（重複スキャン回避のため）
    return {
        'params_lasco': p_lasco,
        'params_mk4': p_mk4,
        'lasco_map': lasco_map,
        'mk4_map': mk4_map
    }

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from astropy.time import Time
    
    target_time_str = "2022-06-13T03:00:00"
    target_time_obj = Time(target_time_str)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    create_single_raw_image(ax, target_time_str)
    
    output_dir = Path(f"/mnt/d/wsl/home/kinno-7010/Research_data/SDO_Mk4_SOHO/raw/single_raw_image_multiline_{target_time_obj.strftime('%Y%m%d_%H%M%S')}.png")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_dir}")
    
    plt.show()