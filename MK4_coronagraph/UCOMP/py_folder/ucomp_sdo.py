"""
UCOMPとSDO/AIA 211データの統合プロット
ucomp_ext12.pyの関数をそのまま使用してUCoMP Ext12とSDO/AIA 211を統合
"""

import sys
import os
sys.path.append('/home/kinno-7010/Research_code/MK4_coronagraph/UCOMP/py_folder')

# UCoMP関連のインポート
from ucomp_config import *
from ucomp_scanner import find_closest_ucomp_data
from ucomp_plotting import *

# ucomp_ext12の関数をインポート
from ucomp_ext12 import *

# SDO/AIA関連のインポート
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ImageNormalize, PowerStretch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sunpy.map
import astropy.units as u
from matplotlib.ticker import FuncFormatter
import glob
from datetime import datetime
import astropy.visualization as vis

# 定数
SDO_AIA_211_DATA_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/SDO/AIA/Rawdata/211")
SOLAR_RADIUS_THRESHOLD = 1.2  # Rs（太陽半径） - AIA 211を内側に表示
BASE_DATA_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/SDO/AIA/Rawdata")

def normalize_log_stretch(data):
    """
    AIA_analysis.ipynbから移植したログ正規化関数
    LogStretch normalizationをデータに適用
    """
    # LogStretchは負の値を扱えないため、ゼロや負の値を避けるためにクリッピングする
    data_clipped = np.maximum(data, 1e-5)
    normalizer = vis.ImageNormalize(data_clipped, stretch=vis.LogStretch(), clip=True)
    return normalizer(data_clipped)

def find_closest_aia_211_data(target_time, time_window_minutes=5):
    """
    指定時刻に最も近いSDO/AIA 211データを検索（AIA_analysis.ipynbのスキャン方式）
    
    Parameters
    ----------
    target_time : str or Time
        目標時刻
    time_window_minutes : int
        検索時間窓（分）
        
    Returns
    -------
    tuple or None
        (sunpy.map.Map, file_path) または None
    """
    if isinstance(target_time, str):
        target_time = Time(target_time)
    
    # 検索時間範囲を設定
    start_time = target_time - time_window_minutes * u.min
    end_time = target_time + time_window_minutes * u.min
    
    print(f"Searching AIA 211 data from {start_time.iso} to {end_time.iso}")
    
    # AIA 211データの直接検索
    pattern = f"{SDO_AIA_211_DATA_DIR}/AIA*.fits"
    aia_211_files = []
    
    for file_path in glob.glob(pattern):
        try:
            aia_map = sunpy.map.Map(file_path)
            file_time = aia_map.date
            if start_time <= file_time <= end_time:
                aia_211_files.append((aia_map, file_path))
        except Exception as e:
            continue
    
    if not aia_211_files:
        print("No AIA 211 data found in the specified time range")
        return None
    
    # 最も近い時刻のファイルを選択
    closest_file = None
    min_time_diff = float('inf')
    
    for aia_map, file_path in aia_211_files:
        time_diff = abs((aia_map.date - target_time).to_value('sec'))
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_file = (aia_map, file_path)
    
    if closest_file:
        print(f"Found closest AIA 211 data: {closest_file[0].date.iso}")
        print(f"Time difference: {min_time_diff:.1f} seconds")
        return closest_file
    
    return None

def find_closest_aia_rgb_data(target_time, time_window_minutes=5):
    """
    指定時刻に最も近いSDO/AIA RGB用データ（211, 193, 171）を検索
    
    Parameters
    ----------
    target_time : str or Time
        目標時刻
    time_window_minutes : int
        検索時間窓（分）
        
    Returns
    -------
    dict or None
        {'211': (sunpy.map.Map, file_path), '193': (sunpy.map.Map, file_path), '171': (sunpy.map.Map, file_path)} または None
    """
    if isinstance(target_time, str):
        target_time = Time(target_time)
    
    # 検索時間範囲を設定
    start_time = target_time - time_window_minutes * u.min
    end_time = target_time + time_window_minutes * u.min
    
    print(f"Searching AIA RGB data (211, 193, 171) from {start_time.iso} to {end_time.iso}")
    
    channels = ['211', '193', '171']
    rgb_data = {}
    
    for channel in channels:
        channel_dir = BASE_DATA_DIR / channel
        pattern = f"{channel_dir}/AIA*.fits"
        channel_files = []
        
        for file_path in glob.glob(pattern):
            try:
                aia_map = sunpy.map.Map(file_path)
                file_time = aia_map.date
                if start_time <= file_time <= end_time:
                    channel_files.append((aia_map, file_path))
            except Exception as e:
                continue
        
        if not channel_files:
            print(f"No AIA {channel} data found in the specified time range")
            return None
        
        # 最も近い時刻のファイルを選択
        closest_file = None
        min_time_diff = float('inf')
        
        for aia_map, file_path in channel_files:
            time_diff = abs((aia_map.date - target_time).to_value('sec'))
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_file = (aia_map, file_path)
        
        if closest_file:
            rgb_data[channel] = closest_file
            print(f"Found closest AIA {channel} data: {closest_file[0].date.iso} (diff: {min_time_diff:.1f}s)")
        else:
            print(f"No suitable AIA {channel} data found")
            return None
    
    return rgb_data

def plot_sdo_aia_211(datetime_str, channel_str="211"):
    """
    指定された日時と波長チャンネルのSDO/AIA画像をWCSベースでプロットし、
    軸の目盛りラベルのみを太陽中心を(0,0)とするピクセル単位で表示します。
    AIA_analysis.ipynbのplot_sdo_aia関数をそのまま使用
    """
    # 1. 日時とファイルパスの処理
    try:
        dt_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
        date_fmtd_for_fname = dt_obj.strftime("%Y%m%d")
        time_fmtd_for_fname = dt_obj.strftime("%H%M")
    except ValueError:
        print(
            f"エラー: 日時文字列 '{datetime_str}' の形式が無効です。"
            " 'YYYY-MM-DD HH:MM' 形式で指定してください。"
        )
        return None

    wavelength_part_in_fname = channel_str.zfill(4)
    filename = f"AIA{date_fmtd_for_fname}_{time_fmtd_for_fname}_{wavelength_part_in_fname}.fits"
    file_path = BASE_DATA_DIR / channel_str / filename

    print(f"ターゲットファイルパス: {file_path}")

    try:
        aia_map = sunpy.map.Map(file_path)
        print(f"ファイル '{file_path}' を正常に読み込みました。")
        return aia_map
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {file_path}")
        return None
    except Exception as e:
        print(f"ファイルの読み込み・初期処理中にエラーが発生しました: {e}")
        return None

def plot_ucomp_aia_211_integrated(ax, target_time, start_time, end_time, extension_num=12, 
                                  wavelength=DEFAULT_WAVELENGTH, save_path=None, 
                                  smooth_ext12=False, ext12_sigma=1.0):
    """
    UCoMP Ext12データとSDO/AIA 211データを統合してプロット
    ucomp_ext12.pyの関数をベースに、AIA 211を内側に追加
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        プロット対象のaxes
    target_time : str or Time
        プロット対象時刻
    start_time : str or Time
        スキャン開始時刻
    end_time : str or Time
        スキャン終了時刻
    extension_num : int
        Extension番号 (1-12)
    wavelength : int
        波長
    save_path : str, optional
        保存パス
    smooth_ext12 : bool, optional
        Extension 12の平滑化を適用するか (default: False)
    ext12_sigma : float, optional
        Extension 12の平滑化シグマ値 (default: 1.0)
    """
    if extension_num < 1 or extension_num > 12:
        raise ValueError("Extension number must be between 1 and 12")
    
    # 1. UCoMPデータの取得
    closest_ucomp_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
    
    if closest_ucomp_data is None:
        print("No UCOMP data found for the specified time range")
        return None
    
    closest_ucomp_info, ucomp_file_path = closest_ucomp_data
    
    # 指定Extensionを読み込み
    extensions = read_ucomp_extensions(ucomp_file_path, max_extensions=extension_num)
    
    if extension_num not in extensions:
        print(f"Extension {extension_num} not found in the file")
        return None
    
    ucomp_data, ucomp_header = extensions[extension_num]
    
    # UCOMPのWCS情報を取得
    rsun_obs, cdelt1, crpix1, crpix2 = get_header_info(ucomp_file_path)
    solar_radius_pixels = rsun_obs / cdelt1
    
    # 2. AIA 211データの取得
    closest_aia_data = find_closest_aia_211_data(target_time, time_window_minutes=5)
    
    if closest_aia_data is None:
        print("No AIA 211 data found, plotting UCOMP only")
        # UCOMPのみをプロット
        plot_single_extension(ax, target_time, start_time, end_time, extension_num, 
                            wavelength, save_path, smooth_ext12, ext12_sigma)
        return
    
    aia_map, aia_file_path = closest_aia_data
    
    # 3. AIA 211データのプロット（背景として）
    aia_data = aia_map.data
    
    # データの正規化
    valid_data = aia_data[np.isfinite(aia_data)]
    if len(valid_data) > 0:
        vmin = np.percentile(valid_data, 1.0)
        vmax = np.percentile(valid_data, 99.5)
        norm = ImageNormalize(
            vmin=vmin, 
            vmax=vmax, 
            stretch=PowerStretch(0.5),
            clip=True
        )
    else:
        norm = None
    
    # AIA座標系のパラメータ
    if 'rsun_obs' in aia_map.meta:
        aia_rsun_arcsec = aia_map.meta['rsun_obs']
    else:
        aia_rsun_arcsec = 959.63
    
    if 'cdelt1' in aia_map.meta:
        aia_pixel_scale = abs(aia_map.meta['cdelt1'])
    else:
        aia_pixel_scale = 0.6
    
    aia_solar_radius_pixels = aia_rsun_arcsec / aia_pixel_scale
    
    # UCOMPピクセル座標系への変換スケール
    scale_factor = solar_radius_pixels / aia_solar_radius_pixels
    
    # AIA座標系での太陽中心
    aia_center_x = aia_map.meta['crpix1'] - 1
    aia_center_y = aia_map.meta['crpix2'] - 1
    
    # AIA内側マスクの作成（SOLAR_RADIUS_THRESHOLD以内）
    height, width = aia_data.shape
    y_idx, x_idx = np.indices((height, width))
    distance = np.sqrt((x_idx - aia_center_x)**2 + (y_idx - aia_center_y)**2)
    distance_rs = distance / aia_solar_radius_pixels
    aia_mask = distance_rs <= SOLAR_RADIUS_THRESHOLD
    
    # マスクを適用
    masked_aia_data = np.where(aia_mask, aia_data, np.nan)
    
    # UCOMPピクセル座標系でのextent設定
    aia_extent = [
        -aia_center_x * scale_factor, 
        (width - aia_center_x) * scale_factor, 
        -aia_center_y * scale_factor, 
        (height - aia_center_y) * scale_factor
    ]
    
    # AIA 211データをプロット（背景）
    im_aia = ax.imshow(masked_aia_data, origin='lower', cmap='sdoaia211', 
                       norm=norm, extent=aia_extent, aspect='equal', alpha=0.8)
    
    # 4. UCOMPデータの処理とプロット
    if extension_num == 12:
        # Extension 12の平滑化処理
        processed_ucomp_data = smooth_ext12_data(ucomp_data, smooth=smooth_ext12, sigma=ext12_sigma)
        
        # Extension 12は規格化しない
        vmin_ucomp, vmax_ucomp = -45, 45
        cmap_ucomp = 'RdBu_r'
        colorbar_label = get_colorbar_label(extension_num)
        
        # 太陽中心を原点とする座標系
        ucomp_center_x = processed_ucomp_data.shape[1] // 2
        ucomp_center_y = processed_ucomp_data.shape[0] // 2
        
        ucomp_extent = [
            -ucomp_center_x, 
            processed_ucomp_data.shape[1] - ucomp_center_x,
            -ucomp_center_y, 
            processed_ucomp_data.shape[0] - ucomp_center_y
        ]
        
        # UCOMPの外側マスク作成（SOLAR_RADIUS_THRESHOLD以上）
        ucomp_y_idx, ucomp_x_idx = np.indices(processed_ucomp_data.shape)
        ucomp_distance = np.sqrt((ucomp_x_idx - ucomp_center_x)**2 + (ucomp_y_idx - ucomp_center_y)**2)
        ucomp_distance_rs = ucomp_distance / solar_radius_pixels
        ucomp_outer_mask = ucomp_distance_rs >= SOLAR_RADIUS_THRESHOLD
        
        # 外側マスクを適用したUCOMPデータ
        masked_ucomp_data = np.where(ucomp_outer_mask, processed_ucomp_data, np.nan)
        
        # UCOMPデータをプロット（前景）
        im_ucomp = ax.imshow(masked_ucomp_data, origin='lower', cmap=cmap_ucomp, aspect='equal', 
                           vmin=vmin_ucomp, vmax=vmax_ucomp, extent=ucomp_extent, alpha=0.9)
        
        # 磁場ベクトル場を追加
        x_grid, y_grid = create_vector_field_grid(processed_ucomp_data.shape, rsun_obs, cdelt1, crpix1, crpix2,
                                                 radial_interval_rs=0.1, 
                                                 angular_interval_deg=5)
        
        # SOLAR_RADIUS_THRESHOLD以上の領域のみをフィルタリング
        grid_distance_rs = np.sqrt(x_grid**2 + y_grid**2) / solar_radius_pixels
        valid_grid_mask = grid_distance_rs >= SOLAR_RADIUS_THRESHOLD
        
        if np.any(valid_grid_mask):
            x_grid_filtered = x_grid[valid_grid_mask]
            y_grid_filtered = y_grid[valid_grid_mask]
            
            draw_magnetic_field_vectors(ax, ucomp_data, processed_ucomp_data, 
                                      x_grid_filtered, y_grid_filtered, crpix1, crpix2,
                                      arrow_length_scale=15, 
                                      arrow_color='black',
                                      arrow_alpha=0.9)
    else:
        # 他のExtensionは従来通り正規化
        normalized_ucomp_data = normalize_ucomp_data(ucomp_data)
        im_ucomp = ax.imshow(normalized_ucomp_data, origin='lower', cmap='plasma', aspect='equal')
        colorbar_label = 'Normalized Intensity'
    
    # 5. カラーバー
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)
    cbar = plt.colorbar(im_ucomp, cax=cax)
    cbar.set_label(colorbar_label, fontsize=12)
    
    # 6. 境界円の描画
    theta = np.linspace(0, 2*np.pi, 360)
    
    # 1Rs円（太陽リム）
    rs_circle_x = solar_radius_pixels * np.cos(theta)
    rs_circle_y = solar_radius_pixels * np.sin(theta)
    ax.plot(rs_circle_x, rs_circle_y, '-', color='white', linewidth=1, 
            label='1 Rs (Solar limb)')
    
    # 境界円（SOLAR_RADIUS_THRESHOLD Rs）
    boundary_radius = SOLAR_RADIUS_THRESHOLD * solar_radius_pixels
    boundary_x = boundary_radius * np.cos(theta)
    boundary_y = boundary_radius * np.sin(theta)
    ax.plot(boundary_x, boundary_y, '--', color='yellow', linewidth=2, 
            label=f'{SOLAR_RADIUS_THRESHOLD} Rs boundary')
    
    # 7. タイトルとラベル
    title_parts = [
        f'UCoMP Ext{extension_num} + SDO/AIA 211 Integrated Plot',
        f'UCoMP: {closest_ucomp_info["date"].strftime("%Y-%m-%d %H:%M:%S")} | AIA 211: {aia_map.date.strftime("%Y-%m-%d %H:%M:%S")}'
    ]
    
    if smooth_ext12 and extension_num == 12:
        title_parts[0] += f' (σ={ext12_sigma})'
    
    ax.set_title('\n'.join(title_parts), fontsize=14)
    ax.set_xlabel('X [pixels]', fontsize=10)
    ax.set_ylabel('Y [pixels]', fontsize=10)
    ax.set_xlim(-650, 650)
    ax.set_ylim(-550, 550)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

def plot_ucomp_aia_rgb_integrated(ax, target_time, start_time, end_time, extension_num=12, 
                                  wavelength=DEFAULT_WAVELENGTH, save_path=None, 
                                  smooth_ext12=False, ext12_sigma=1.0):
    """
    UCoMP Ext12データとSDO/AIA RGB（211, 193, 171）データを統合してプロット
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        プロット対象のaxes
    target_time : str or Time
        プロット対象時刻
    start_time : str or Time
        スキャン開始時刻
    end_time : str or Time
        スキャン終了時刻
    extension_num : int
        Extension番号 (1-12)
    wavelength : int
        波長
    save_path : str, optional
        保存パス
    smooth_ext12 : bool, optional
        Extension 12の平滑化を適用するか (default: False)
    ext12_sigma : float, optional
        Extension 12の平滑化シグマ値 (default: 1.0)
    """
    if extension_num < 1 or extension_num > 12:
        raise ValueError("Extension number must be between 1 and 12")
    
    # 1. UCoMPデータの取得
    closest_ucomp_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
    
    if closest_ucomp_data is None:
        print("No UCOMP data found for the specified time range")
        return None
    
    closest_ucomp_info, ucomp_file_path = closest_ucomp_data
    
    # 指定Extensionを読み込み
    extensions = read_ucomp_extensions(ucomp_file_path, max_extensions=extension_num)
    
    if extension_num not in extensions:
        print(f"Extension {extension_num} not found in the file")
        return None
    
    ucomp_data, ucomp_header = extensions[extension_num]
    
    # UCOMPのWCS情報を取得
    rsun_obs, cdelt1, crpix1, crpix2 = get_header_info(ucomp_file_path)
    solar_radius_pixels = rsun_obs / cdelt1
    
    # 2. AIA RGBデータの取得
    closest_aia_rgb_data = find_closest_aia_rgb_data(target_time, time_window_minutes=5)
    
    if closest_aia_rgb_data is None:
        print("No AIA RGB data found, plotting UCOMP only")
        # UCOMPのみをプロット
        plot_single_extension(ax, target_time, start_time, end_time, extension_num, 
                            wavelength, save_path, smooth_ext12, ext12_sigma)
        return
    
    # 3. AIA RGBデータの処理とプロット
    channels = ['211', '193', '171']  # R, G, B
    aia_maps = {}
    
    for channel in channels:
        aia_map, aia_file_path = closest_aia_rgb_data[channel]
        aia_maps[channel] = aia_map
    
    # RGB合成用のデータ正規化
    red_channel_data = normalize_log_stretch(aia_maps['211'].data)
    green_channel_data = normalize_log_stretch(aia_maps['193'].data)
    blue_channel_data = normalize_log_stretch(aia_maps['171'].data)
    
    # 0-1にスケーリング
    def scale_to_01(data):
        d_min = np.nanmin(data)
        d_max = np.nanmax(data)
        if d_max == d_min:
            return np.zeros_like(data)
        return (data - d_min) / (d_max - d_min)
    
    red_channel_final = scale_to_01(red_channel_data)
    green_channel_final = scale_to_01(green_channel_data)
    blue_channel_final = scale_to_01(blue_channel_data)
    rgb_image = np.stack([red_channel_final, green_channel_final, blue_channel_final], axis=-1)
    
    # AIA座標系のパラメータ（基準マップとして171を使用）
    reference_map = aia_maps['171']
    if 'rsun_obs' in reference_map.meta:
        aia_rsun_arcsec = reference_map.meta['rsun_obs']
    else:
        aia_rsun_arcsec = 959.63
    
    if 'cdelt1' in reference_map.meta:
        aia_pixel_scale = abs(reference_map.meta['cdelt1'])
    else:
        aia_pixel_scale = 0.6
    
    aia_solar_radius_pixels = aia_rsun_arcsec / aia_pixel_scale
    
    # UCOMPピクセル座標系への変換スケール
    scale_factor = solar_radius_pixels / aia_solar_radius_pixels
    
    # AIA座標系での太陽中心
    aia_center_x = reference_map.meta['crpix1'] - 1
    aia_center_y = reference_map.meta['crpix2'] - 1
    
    # AIA内側マスクの作成（SOLAR_RADIUS_THRESHOLD以内）
    height, width = rgb_image.shape[:2]
    y_idx, x_idx = np.indices((height, width))
    distance = np.sqrt((x_idx - aia_center_x)**2 + (y_idx - aia_center_y)**2)
    distance_rs = distance / aia_solar_radius_pixels
    aia_mask = distance_rs <= SOLAR_RADIUS_THRESHOLD
    
    # マスクを適用
    masked_rgb_image = rgb_image.copy()
    masked_rgb_image[~aia_mask] = np.nan
    
    # UCOMPピクセル座標系でのextent設定
    aia_extent = [
        -aia_center_x * scale_factor, 
        (width - aia_center_x) * scale_factor, 
        -aia_center_y * scale_factor, 
        (height - aia_center_y) * scale_factor
    ]
    
    # AIA RGBデータをプロット（背景）
    im_aia = ax.imshow(masked_rgb_image, origin='lower', extent=aia_extent, aspect='equal', alpha=0.8)
    
    # 4. UCOMPデータの処理とプロット
    if extension_num == 12:
        # Extension 12の平滑化処理
        processed_ucomp_data = smooth_ext12_data(ucomp_data, smooth=smooth_ext12, sigma=ext12_sigma)
        
        # Extension 12は規格化しない
        vmin_ucomp, vmax_ucomp = -45, 45
        cmap_ucomp = 'RdBu_r'
        colorbar_label = get_colorbar_label(extension_num)
        
        # 太陽中心を原点とする座標系
        ucomp_center_x = processed_ucomp_data.shape[1] // 2
        ucomp_center_y = processed_ucomp_data.shape[0] // 2
        
        ucomp_extent = [
            -ucomp_center_x, 
            processed_ucomp_data.shape[1] - ucomp_center_x,
            -ucomp_center_y, 
            processed_ucomp_data.shape[0] - ucomp_center_y
        ]
        
        # UCOMPの外側マスク作成（SOLAR_RADIUS_THRESHOLD以上）
        ucomp_y_idx, ucomp_x_idx = np.indices(processed_ucomp_data.shape)
        ucomp_distance = np.sqrt((ucomp_x_idx - ucomp_center_x)**2 + (ucomp_y_idx - ucomp_center_y)**2)
        ucomp_distance_rs = ucomp_distance / solar_radius_pixels
        ucomp_outer_mask = ucomp_distance_rs >= SOLAR_RADIUS_THRESHOLD
        
        # 外側マスクを適用したUCOMPデータ
        masked_ucomp_data = np.where(ucomp_outer_mask, processed_ucomp_data, np.nan)
        
        # UCOMPデータをプロット（前景）
        im_ucomp = ax.imshow(masked_ucomp_data, origin='lower', cmap=cmap_ucomp, aspect='equal', 
                           vmin=vmin_ucomp, vmax=vmax_ucomp, extent=ucomp_extent, alpha=0.9)
        
        # 磁場ベクトル場を追加
        x_grid, y_grid = create_vector_field_grid(processed_ucomp_data.shape, rsun_obs, cdelt1, crpix1, crpix2,
                                                 radial_interval_rs=0.1, 
                                                 angular_interval_deg=5)
        
        # SOLAR_RADIUS_THRESHOLD以上の領域のみをフィルタリング
        grid_distance_rs = np.sqrt(x_grid**2 + y_grid**2) / solar_radius_pixels
        valid_grid_mask = grid_distance_rs >= SOLAR_RADIUS_THRESHOLD
        
        if np.any(valid_grid_mask):
            x_grid_filtered = x_grid[valid_grid_mask]
            y_grid_filtered = y_grid[valid_grid_mask]
            
            draw_magnetic_field_vectors(ax, ucomp_data, processed_ucomp_data, 
                                      x_grid_filtered, y_grid_filtered, crpix1, crpix2,
                                      arrow_length_scale=15, 
                                      arrow_color='black',
                                      arrow_alpha=0.9)
    else:
        # 他のExtensionは従来通り正規化
        normalized_ucomp_data = normalize_ucomp_data(ucomp_data)
        im_ucomp = ax.imshow(normalized_ucomp_data, origin='lower', cmap='plasma', aspect='equal')
        colorbar_label = 'Normalized Intensity'
    
    # 5. カラーバー
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)
    cbar = plt.colorbar(im_ucomp, cax=cax)
    cbar.set_label(colorbar_label, fontsize=12)
    
    # 6. 境界円の描画
    theta = np.linspace(0, 2*np.pi, 360)
    
    # 1Rs円（太陽リム）
    rs_circle_x = solar_radius_pixels * np.cos(theta)
    rs_circle_y = solar_radius_pixels * np.sin(theta)
    ax.plot(rs_circle_x, rs_circle_y, '-', color='white', linewidth=1, 
            label='1 Rs (Solar limb)')
    
    # 境界円（SOLAR_RADIUS_THRESHOLD Rs）
    boundary_radius = SOLAR_RADIUS_THRESHOLD * solar_radius_pixels
    boundary_x = boundary_radius * np.cos(theta)
    boundary_y = boundary_radius * np.sin(theta)
    ax.plot(boundary_x, boundary_y, '--', color='yellow', linewidth=2, 
            label=f'{SOLAR_RADIUS_THRESHOLD} Rs boundary')
    
    # 7. タイトルとラベル
    title_parts = [
        f'UCoMP Ext{extension_num} + SDO/AIA RGB Integrated Plot',
        f'UCoMP: {closest_ucomp_info["date"].strftime("%Y-%m-%d %H:%M:%S")} | AIA RGB: {reference_map.date.strftime("%Y-%m-%d %H:%M:%S")}'
    ]
    
    if smooth_ext12 and extension_num == 12:
        title_parts[0] += f' (σ={ext12_sigma})'
    
    ax.set_title('\n'.join(title_parts), fontsize=14)
    ax.set_xlabel('X [pixels]', fontsize=10)
    ax.set_ylabel('Y [pixels]', fontsize=10)
    ax.set_xlim(-650, 650)
    ax.set_ylim(-550, 550)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"RGB integrated plot saved: {save_path}")

def plot_ucomp_aia_211_diff(ax, target_start_time, target_end_time, start_time, end_time, 
                           extension_num=12, wavelength=DEFAULT_WAVELENGTH, save_path=None, 
                           smooth_ext12=False, ext12_sigma=1.0):
    """
    UCoMP Ext12差分データとSDO/AIA 211差分データを統合してプロット
    ucomp_ext12.pyのplot_ext_12_diff関数をベースに、AIA 211差分を内側に追加
    """
    # 1. UCOMPデータの取得（開始・終了時刻）
    closest_ucomp_start_data = find_closest_ucomp_data(target_start_time, start_time, end_time, wavelength)
    closest_ucomp_end_data = find_closest_ucomp_data(target_end_time, start_time, end_time, wavelength)
    
    if closest_ucomp_start_data is None or closest_ucomp_end_data is None:
        print("No UCOMP data found for the specified time range")
        return None
    
    ucomp_info_start, ucomp_file_path_start = closest_ucomp_start_data
    ucomp_info_end, ucomp_file_path_end = closest_ucomp_end_data
    
    # UCOMPデータの読み込み
    ucomp_extensions_start = read_ucomp_extensions(ucomp_file_path_start, max_extensions=extension_num)
    ucomp_extensions_end = read_ucomp_extensions(ucomp_file_path_end, max_extensions=extension_num)
    
    if extension_num not in ucomp_extensions_start or extension_num not in ucomp_extensions_end:
        print(f"Extension {extension_num} not found in UCOMP data")
        return None
    
    ucomp_data_start, _ = ucomp_extensions_start[extension_num]
    ucomp_data_end, _ = ucomp_extensions_end[extension_num]
    
    # UCOMPの差分計算
    diff_ucomp_data = ucomp_data_end - ucomp_data_start
    
    # UCOMPのWCS情報
    rsun_obs, cdelt1, crpix1, crpix2 = get_header_info(ucomp_file_path_start)
    solar_radius_pixels = rsun_obs / cdelt1
    
    # 2. AIA 211データの取得（開始・終了時刻）
    closest_aia_start_data = find_closest_aia_211_data(target_start_time, time_window_minutes=5)
    closest_aia_end_data = find_closest_aia_211_data(target_end_time, time_window_minutes=5)
    
    if closest_aia_start_data is None or closest_aia_end_data is None:
        print("No AIA 211 data found, plotting UCOMP diff only")
        # UCOMPのみの差分をプロット
        plot_ext_12_diff(ax, target_start_time, target_end_time, start_time, end_time, 
                        extension_num, wavelength, save_path, smooth_ext12, ext12_sigma)
        return
    
    aia_map_start, _ = closest_aia_start_data
    aia_map_end, _ = closest_aia_end_data
    
    # AIA差分データ
    diff_aia_data = aia_map_end.data - aia_map_start.data
    
    # 3. AIA差分データのプロット処理
    valid_aia_data = diff_aia_data[np.isfinite(diff_aia_data)]
    if len(valid_aia_data) > 0:
        vmin_aia = np.percentile(valid_aia_data, 1.0)
        vmax_aia = np.percentile(valid_aia_data, 99.5)
        norm_aia = ImageNormalize(
            vmin=vmin_aia, 
            vmax=vmax_aia, 
            stretch=PowerStretch(0.5),
            clip=True
        )
    else:
        norm_aia = None
    
    # AIA座標系のパラメータ
    if 'rsun_obs' in aia_map_start.meta:
        aia_rsun_arcsec = aia_map_start.meta['rsun_obs']
    else:
        aia_rsun_arcsec = 959.63
    
    if 'cdelt1' in aia_map_start.meta:
        aia_pixel_scale = abs(aia_map_start.meta['cdelt1'])
    else:
        aia_pixel_scale = 0.6
    
    aia_solar_radius_pixels = aia_rsun_arcsec / aia_pixel_scale
    scale_factor = solar_radius_pixels / aia_solar_radius_pixels
    
    # AIA差分データのマスクとプロット
    aia_center_x = aia_map_start.meta['crpix1'] - 1
    aia_center_y = aia_map_start.meta['crpix2'] - 1
    
    height, width = diff_aia_data.shape
    y_idx, x_idx = np.indices((height, width))
    distance = np.sqrt((x_idx - aia_center_x)**2 + (y_idx - aia_center_y)**2)
    distance_rs = distance / aia_solar_radius_pixels
    aia_mask = distance_rs <= SOLAR_RADIUS_THRESHOLD
    
    masked_diff_aia_data = np.where(aia_mask, diff_aia_data, np.nan)
    
    aia_extent = [
        -aia_center_x * scale_factor, 
        (width - aia_center_x) * scale_factor, 
        -aia_center_y * scale_factor, 
        (height - aia_center_y) * scale_factor
    ]
    
    # AIA差分データをプロット
    im_aia = ax.imshow(masked_diff_aia_data, origin='lower', cmap='sdoaia211', 
                       norm=norm_aia, extent=aia_extent, aspect='equal', alpha=0.8)
    
    # 4. UCOMP差分データの処理とプロット
    if extension_num == 12:
        # 差分データの平滑化処理
        if smooth_ext12:
            processed_ucomp_data = smooth_ext12_data(diff_ucomp_data, smooth=True, sigma=ext12_sigma)
        else:
            processed_ucomp_data = diff_ucomp_data
        
        vmin_ucomp, vmax_ucomp = -45, 45
        cmap_ucomp = 'RdBu_r'
        colorbar_label = "UCOMP magnetic field differential vector angle $\\Delta \\phi$ [deg]"
        
        # UCOMP座標系
        ucomp_center_x = processed_ucomp_data.shape[1] // 2
        ucomp_center_y = processed_ucomp_data.shape[0] // 2
        
        ucomp_extent = [
            -ucomp_center_x, 
            processed_ucomp_data.shape[1] - ucomp_center_x,
            -ucomp_center_y, 
            processed_ucomp_data.shape[0] - ucomp_center_y
        ]
        
        # UCOMPの外側マスク
        ucomp_y_idx, ucomp_x_idx = np.indices(processed_ucomp_data.shape)
        ucomp_distance = np.sqrt((ucomp_x_idx - ucomp_center_x)**2 + (ucomp_y_idx - ucomp_center_y)**2)
        ucomp_distance_rs = ucomp_distance / solar_radius_pixels
        ucomp_outer_mask = ucomp_distance_rs >= SOLAR_RADIUS_THRESHOLD
        
        masked_ucomp_data = np.where(ucomp_outer_mask, processed_ucomp_data, np.nan)
        
        # UCOMP差分データをプロット
        im_ucomp = ax.imshow(masked_ucomp_data, origin='lower', cmap=cmap_ucomp, aspect='equal', 
                           vmin=vmin_ucomp, vmax=vmax_ucomp, extent=ucomp_extent, alpha=0.9)
        
        # 磁場ベクトル場を追加（差分データ用）
        x_grid, y_grid = create_vector_field_grid(processed_ucomp_data.shape, rsun_obs, cdelt1, crpix1, crpix2,
                                                 radial_interval_rs=0.1, 
                                                 angular_interval_deg=5)
        
        grid_distance_rs = np.sqrt(x_grid**2 + y_grid**2) / solar_radius_pixels
        valid_grid_mask = grid_distance_rs >= SOLAR_RADIUS_THRESHOLD
        
        if np.any(valid_grid_mask):
            x_grid_filtered = x_grid[valid_grid_mask]
            y_grid_filtered = y_grid[valid_grid_mask]
            
            draw_magnetic_field_vectors(ax, diff_ucomp_data, processed_ucomp_data, 
                                      x_grid_filtered, y_grid_filtered, crpix1, crpix2,
                                      arrow_length_scale=15, 
                                      arrow_color='black',
                                      arrow_alpha=0.9)
    
    # 5. カラーバー
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)
    cbar = plt.colorbar(im_ucomp, cax=cax)
    cbar.set_label(colorbar_label, fontsize=12)
    
    # 6. 境界円の描画
    theta = np.linspace(0, 2*np.pi, 360)
    
    rs_circle_x = solar_radius_pixels * np.cos(theta)
    rs_circle_y = solar_radius_pixels * np.sin(theta)
    ax.plot(rs_circle_x, rs_circle_y, '-', color='white', linewidth=1, 
            label='1 Rs (Solar limb)')
    
    boundary_radius = SOLAR_RADIUS_THRESHOLD * solar_radius_pixels
    boundary_x = boundary_radius * np.cos(theta)
    boundary_y = boundary_radius * np.sin(theta)
    ax.plot(boundary_x, boundary_y, '--', color='yellow', linewidth=2, 
            label=f'{SOLAR_RADIUS_THRESHOLD} Rs boundary')
    
    # 7. タイトルとラベル
    title_parts = [
        f'UCoMP Ext{extension_num} + SDO/AIA 211 Differential Plot',
        f'UCoMP: {ucomp_info_start["date"].strftime("%Y-%m-%d %H:%M:%S")} - {ucomp_info_end["date"].strftime("%Y-%m-%d %H:%M:%S")}',
        f'AIA 211: {aia_map_start.date.strftime("%Y-%m-%d %H:%M:%S")} - {aia_map_end.date.strftime("%Y-%m-%d %H:%M:%S")}',
    ]
    
    if smooth_ext12 and extension_num == 12:
        title_parts[0] += f' (σ={ext12_sigma})'
    
    ax.set_title('\n'.join(title_parts), fontsize=14)
    ax.set_xlabel('X [pixels]', fontsize=10)
    ax.set_ylabel('Y [pixels]', fontsize=10)
    ax.set_xlim(-650, 650)
    ax.set_ylim(-550, 550)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

def plot_ucomp_aia_rgb_diff(ax, target_start_time, target_end_time, start_time, end_time, 
                           extension_num=12, wavelength=DEFAULT_WAVELENGTH, save_path=None, 
                           smooth_ext12=False, ext12_sigma=1.0):
    """
    UCoMP Ext12差分データとSDO/AIA RGB差分データを統合してプロット
    """
    # 1. UCOMPデータの取得（開始・終了時刻）
    closest_ucomp_start_data = find_closest_ucomp_data(target_start_time, start_time, end_time, wavelength)
    closest_ucomp_end_data = find_closest_ucomp_data(target_end_time, start_time, end_time, wavelength)
    
    if closest_ucomp_start_data is None or closest_ucomp_end_data is None:
        print("No UCOMP data found for the specified time range")
        return None
    
    ucomp_info_start, ucomp_file_path_start = closest_ucomp_start_data
    ucomp_info_end, ucomp_file_path_end = closest_ucomp_end_data
    
    # UCOMPデータの読み込み
    ucomp_extensions_start = read_ucomp_extensions(ucomp_file_path_start, max_extensions=extension_num)
    ucomp_extensions_end = read_ucomp_extensions(ucomp_file_path_end, max_extensions=extension_num)
    
    if extension_num not in ucomp_extensions_start or extension_num not in ucomp_extensions_end:
        print(f"Extension {extension_num} not found in UCOMP data")
        return None
    
    ucomp_data_start, _ = ucomp_extensions_start[extension_num]
    ucomp_data_end, _ = ucomp_extensions_end[extension_num]
    
    # UCOMPの差分計算
    diff_ucomp_data = ucomp_data_end - ucomp_data_start
    
    # UCOMPのWCS情報
    rsun_obs, cdelt1, crpix1, crpix2 = get_header_info(ucomp_file_path_start)
    solar_radius_pixels = rsun_obs / cdelt1
    
    # 2. AIA RGBデータの取得（開始・終了時刻）
    closest_aia_rgb_start_data = find_closest_aia_rgb_data(target_start_time, time_window_minutes=5)
    closest_aia_rgb_end_data = find_closest_aia_rgb_data(target_end_time, time_window_minutes=5)
    
    if closest_aia_rgb_start_data is None or closest_aia_rgb_end_data is None:
        print("No AIA RGB data found, plotting UCOMP diff only")
        # UCOMPのみの差分をプロット
        plot_ext_12_diff(ax, target_start_time, target_end_time, start_time, end_time, 
                        extension_num, wavelength, save_path, smooth_ext12, ext12_sigma)
        return
    
    # 3. AIA RGB差分データの処理
    channels = ['211', '193', '171']  # R, G, B
    aia_maps_start = {}
    aia_maps_end = {}
    
    for channel in channels:
        aia_map_start, _ = closest_aia_rgb_start_data[channel]
        aia_map_end, _ = closest_aia_rgb_end_data[channel]
        aia_maps_start[channel] = aia_map_start
        aia_maps_end[channel] = aia_map_end
    
    # RGB差分データの作成
    red_diff_data = aia_maps_end['211'].data - aia_maps_start['211'].data
    green_diff_data = aia_maps_end['193'].data - aia_maps_start['193'].data
    blue_diff_data = aia_maps_end['171'].data - aia_maps_start['171'].data
    
    # 差分データの正規化
    def scale_diff_to_01(data):
        """差分データを-1～1の範囲から0～1の範囲に変換"""
        d_min = np.nanmin(data)
        d_max = np.nanmax(data)
        if d_max == d_min:
            return np.zeros_like(data)
        # 差分データは負の値も含むため、適切にスケーリング
        data_normalized = (data - d_min) / (d_max - d_min)
        return data_normalized
    
    red_diff_final = scale_diff_to_01(red_diff_data)
    green_diff_final = scale_diff_to_01(green_diff_data)
    blue_diff_final = scale_diff_to_01(blue_diff_data)
    rgb_diff_image = np.stack([red_diff_final, green_diff_final, blue_diff_final], axis=-1)
    
    # AIA座標系のパラメータ（基準マップとして171を使用）
    reference_map = aia_maps_start['171']
    if 'rsun_obs' in reference_map.meta:
        aia_rsun_arcsec = reference_map.meta['rsun_obs']
    else:
        aia_rsun_arcsec = 959.63
    
    if 'cdelt1' in reference_map.meta:
        aia_pixel_scale = abs(reference_map.meta['cdelt1'])
    else:
        aia_pixel_scale = 0.6
    
    aia_solar_radius_pixels = aia_rsun_arcsec / aia_pixel_scale
    scale_factor = solar_radius_pixels / aia_solar_radius_pixels
    
    # AIA差分データのマスクとプロット
    aia_center_x = reference_map.meta['crpix1'] - 1
    aia_center_y = reference_map.meta['crpix2'] - 1
    
    height, width = rgb_diff_image.shape[:2]
    y_idx, x_idx = np.indices((height, width))
    distance = np.sqrt((x_idx - aia_center_x)**2 + (y_idx - aia_center_y)**2)
    distance_rs = distance / aia_solar_radius_pixels
    aia_mask = distance_rs <= SOLAR_RADIUS_THRESHOLD
    
    masked_rgb_diff_image = rgb_diff_image.copy()
    masked_rgb_diff_image[~aia_mask] = np.nan
    
    aia_extent = [
        -aia_center_x * scale_factor, 
        (width - aia_center_x) * scale_factor, 
        -aia_center_y * scale_factor, 
        (height - aia_center_y) * scale_factor
    ]
    
    # RGB差分画像をグレースケールに変換（輝度値計算）
    # RGB to grayscale conversion using standard weights
    rgb_diff_gray = 0.299 * masked_rgb_diff_image[:,:,0] + 0.587 * masked_rgb_diff_image[:,:,1] + 0.114 * masked_rgb_diff_image[:,:,2]
    
    # AIA RGB差分データをプロット（reversed grayカラーマップを使用）
    im_aia = ax.imshow(rgb_diff_gray, origin='lower', extent=aia_extent, aspect='equal', 
                       cmap='gray_r', alpha=0.8)
    
    # 4. UCOMP差分データの処理とプロット
    if extension_num == 12:
        # 差分データの平滑化処理
        if smooth_ext12:
            processed_ucomp_data = smooth_ext12_data(diff_ucomp_data, smooth=True, sigma=ext12_sigma)
        else:
            processed_ucomp_data = diff_ucomp_data
        
        vmin_ucomp, vmax_ucomp = -45, 45
        cmap_ucomp = 'RdBu_r'
        colorbar_label = "UCOMP magnetic field differential vector angle $\\Delta \\phi$ [deg]"
        
        # UCOMP座標系
        ucomp_center_x = processed_ucomp_data.shape[1] // 2
        ucomp_center_y = processed_ucomp_data.shape[0] // 2
        
        ucomp_extent = [
            -ucomp_center_x, 
            processed_ucomp_data.shape[1] - ucomp_center_x,
            -ucomp_center_y, 
            processed_ucomp_data.shape[0] - ucomp_center_y
        ]
        
        # UCOMPの外側マスク
        ucomp_y_idx, ucomp_x_idx = np.indices(processed_ucomp_data.shape)
        ucomp_distance = np.sqrt((ucomp_x_idx - ucomp_center_x)**2 + (ucomp_y_idx - ucomp_center_y)**2)
        ucomp_distance_rs = ucomp_distance / solar_radius_pixels
        ucomp_outer_mask = ucomp_distance_rs >= SOLAR_RADIUS_THRESHOLD
        
        masked_ucomp_data = np.where(ucomp_outer_mask, processed_ucomp_data, np.nan)
        
        # UCOMP差分データをプロット
        im_ucomp = ax.imshow(masked_ucomp_data, origin='lower', cmap=cmap_ucomp, aspect='equal', 
                           vmin=vmin_ucomp, vmax=vmax_ucomp, extent=ucomp_extent, alpha=0.9)
        
        # 磁場ベクトル場を追加（差分データ用）
        x_grid, y_grid = create_vector_field_grid(processed_ucomp_data.shape, rsun_obs, cdelt1, crpix1, crpix2,
                                                 radial_interval_rs=0.1, 
                                                 angular_interval_deg=5)
        
        grid_distance_rs = np.sqrt(x_grid**2 + y_grid**2) / solar_radius_pixels
        valid_grid_mask = grid_distance_rs >= SOLAR_RADIUS_THRESHOLD
        
        if np.any(valid_grid_mask):
            x_grid_filtered = x_grid[valid_grid_mask]
            y_grid_filtered = y_grid[valid_grid_mask]
            
            draw_magnetic_field_vectors(ax, diff_ucomp_data, processed_ucomp_data, 
                                      x_grid_filtered, y_grid_filtered, crpix1, crpix2,
                                      arrow_length_scale=15, 
                                      arrow_color='black',
                                      arrow_alpha=0.9)
    
    # 5. カラーバー
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)
    cbar = plt.colorbar(im_ucomp, cax=cax)
    cbar.set_label(colorbar_label, fontsize=12)
    
    # 6. 境界円の描画
    theta = np.linspace(0, 2*np.pi, 360)
    
    rs_circle_x = solar_radius_pixels * np.cos(theta)
    rs_circle_y = solar_radius_pixels * np.sin(theta)
    ax.plot(rs_circle_x, rs_circle_y, '-', color='white', linewidth=1, 
            label='1 Rs (Solar limb)')
    
    boundary_radius = SOLAR_RADIUS_THRESHOLD * solar_radius_pixels
    boundary_x = boundary_radius * np.cos(theta)
    boundary_y = boundary_radius * np.sin(theta)
    ax.plot(boundary_x, boundary_y, '--', color='yellow', linewidth=2, 
            label=f'{SOLAR_RADIUS_THRESHOLD} Rs boundary')
    
    # 7. タイトルとラベル
    title_parts = [
        f'UCoMP Ext{extension_num} + SDO/AIA RGB Differential Plot',
        f'UCoMP: {ucomp_info_start["date"].strftime("%Y-%m-%d %H:%M:%S")} - {ucomp_info_end["date"].strftime("%Y-%m-%d %H:%M:%S")}',
        f'AIA RGB: {aia_maps_start["171"].date.strftime("%Y-%m-%d %H:%M:%S")} - {aia_maps_end["171"].date.strftime("%Y-%m-%d %H:%M:%S")}'
    ]
    
    if smooth_ext12 and extension_num == 12:
        title_parts[0] += f' (σ={ext12_sigma})'
    
    ax.set_title('\n'.join(title_parts), fontsize=14)
    ax.set_xlabel('X [pixels]', fontsize=10)
    ax.set_ylabel('Y [pixels]', fontsize=10)
    ax.set_xlim(-650, 650)
    ax.set_ylim(-550, 550)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"RGB differential plot saved: {save_path}")

def main(aia_mode="211"):
    """
    メイン実行関数
    
    Parameters
    ----------
    aia_mode : str
        AIA データモード ("211" または "rgb")
    """
    print("UCoMP + SDO/AIA Integration Tool")
    print("=" * 50)
    print(f"AIA Mode: {aia_mode.upper()}")
    print("=" * 50)
    
    start_time = "2022-06-13T03:00:00"
    end_time = "2022-06-13T04:01:00"
    ext_num = 12
    smooth_ext12 = True
    ext12_sigma = 1.0
    
    fig, axes = plt.subplots(1, 3, figsize=(27, 8), tight_layout=True)
    
    if aia_mode.lower() == "211":
        print("=== UCoMP + AIA 211 Integration ===")
        
        # AIA 211統合プロット（2つの時刻）
        for ax, target_time in zip(axes[0:2], ["2022-06-13T03:06:00", "2022-06-13T03:36:00"]):
            plot_ucomp_aia_211_integrated(ax, target_time, start_time, end_time, ext_num,
                                         wavelength=None, save_path=None,
                                         smooth_ext12=smooth_ext12, ext12_sigma=ext12_sigma)
        
        # AIA 211差分プロット
        plot_ucomp_aia_211_diff(axes[2], "2022-06-13T03:06:00", "2022-06-13T03:36:00", 
                               start_time, end_time, ext_num, wavelength=None, save_path=None,
                               smooth_ext12=smooth_ext12, ext12_sigma=ext12_sigma)
        
    elif aia_mode.lower() == "rgb":
        print("=== UCoMP + AIA RGB Integration ===")
        
        # AIA RGB統合プロット（2つの時刻）
        for ax, target_time in zip(axes[0:2], ["2022-06-13T03:06:00", "2022-06-13T03:36:00"]):
            plot_ucomp_aia_rgb_integrated(ax, target_time, start_time, end_time, ext_num,
                                         wavelength=None, save_path=None,
                                         smooth_ext12=smooth_ext12, ext12_sigma=ext12_sigma)
        
        # AIA RGB差分プロット
        plot_ucomp_aia_rgb_diff(axes[2], "2022-06-13T03:06:00", "2022-06-13T03:36:00", 
                               start_time, end_time, ext_num, wavelength=None, save_path=None,
                               smooth_ext12=smooth_ext12, ext12_sigma=ext12_sigma)
        
    else:
        print(f"Error: Invalid AIA mode '{aia_mode}'. Use '211' or 'rgb'.")
        return
    
    plt.show()

if __name__ == "__main__":
    import sys
    
    # コマンドライン引数でモードを指定可能
    if len(sys.argv) > 1:
        aia_mode = sys.argv[1]
    else:
        # デフォルトは211モード
        aia_mode = "211"
    
    # 使用方法の表示
    if aia_mode in ["-h", "--help", "help"]:
        print("Usage: python ucomp_sdo.py [MODE]")
        print("MODE:")
        print("  211  : UCoMP + AIA 211 integration (default)")
        print("  rgb  : UCoMP + AIA RGB integration (211+193+171)")
        print("  help : Show this help message")
        print("\nExamples:")
        print("  python ucomp_sdo.py        # Use AIA 211 mode")
        print("  python ucomp_sdo.py 211    # Use AIA 211 mode")
        print("  python ucomp_sdo.py rgb    # Use AIA RGB mode")
    else:
        main(aia_mode)