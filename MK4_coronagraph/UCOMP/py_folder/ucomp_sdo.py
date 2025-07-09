"""
UCOMPとSDO/AIA 304データの統合プロット
UCoMP Ext12磁場データ（1.7Rs以上）とSDO/AIA 304データ（1.7Rs以内）を統合してプロット
"""

import sys
import os
sys.path.append('/mnt/d/wsl/home/kinno-7010/Research/MK4_coronagraph/UCOMP/py_folder')
sys.path.append('/mnt/d/wsl/home/kinno-7010/Research/SDO/AIA')
sys.path.append('/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/py_folder')

# UCoMP関連のインポート
from ucomp_config import *
from ucomp_scanner import find_closest_ucomp_data
from ucomp_plotting import (
    read_ucomp_extensions, 
    smooth_ext12_data, 
    create_vector_field_grid,
    draw_magnetic_field_vectors,
    get_data_range
)

# SDO/AIA関連のインポート
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ImageNormalize, LinearStretch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sunpy.map
import astropy.units as u
from astropy.visualization import PowerStretch
from matplotlib.ticker import FuncFormatter

# 統合解析関数のインポート
from integrated_analysis import scan_directory_for_maps

# 定数
SDO_AIA_304_DATA_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research/SDO/AIA/Rawdata/304")
SOLAR_RADIUS_THRESHOLD = 1.7  # Rs（太陽半径）

def find_closest_aia_304_data(target_time, time_window_minutes=30):
    """
    指定時刻に最も近いSDO/AIA 304データを検索
    
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
    
    print(f"Searching AIA 304 data from {start_time.iso} to {end_time.iso}")
    
    # scan_directory_for_maps関数を使用してデータを検索
    aia_304_files = scan_directory_for_maps(
        SDO_AIA_304_DATA_DIR, 
        start_time.iso, 
        end_time.iso, 
        use_cache=True
    )
    
    if not aia_304_files:
        print("No AIA 304 data found in the specified time range")
        return None
    
    # 最も近い時刻のファイルを選択
    closest_file = None
    min_time_diff = float('inf')
    
    for aia_map, file_path in aia_304_files:
        time_diff = abs((aia_map.date - target_time).to_value('sec'))
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_file = (aia_map, file_path)
    
    if closest_file:
        print(f"Found closest AIA 304 data: {closest_file[0].date.iso}")
        print(f"Time difference: {min_time_diff:.1f} seconds")
        return closest_file
    
    return None

def plot_sdo_aia_304_with_wcs(aia_map, ax, inner_radius_rs=1.7):
    """
    SDO/AIA 304データをWCSベースでプロットし、太陽中心を原点とする
    
    Parameters
    ----------
    aia_map : sunpy.map.Map
        AIA 304 マップ
    ax : matplotlib.axes.Axes
        プロット対象のaxes
    inner_radius_rs : float
        内側半径（太陽半径単位）
        
    Returns
    -------
    dict
        プロットに使用したパラメータ
    """
    # データの正規化
    data = aia_map.data
    valid_data = data[np.isfinite(data)]
    
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
    
    # 太陽中心を原点とする座標系の設定
    center_x = aia_map.meta['crpix1'] - 1  # FITS → Python indexing
    center_y = aia_map.meta['crpix2'] - 1
    extent = [-center_x, data.shape[1] - center_x, -center_y, data.shape[0] - center_y]
    
    # AIA 304専用カラーマップでプロット
    im = ax.imshow(data, origin='lower', cmap='sdoaia304', 
                   norm=norm, extent=extent, aspect='equal')
    
    # 太陽半径の計算
    if 'rsun_obs' in aia_map.meta:
        rsun_arcsec = aia_map.meta['rsun_obs']  # arcsec
    else:
        rsun_arcsec = 959.63  # デフォルト値
    
    if 'cdelt1' in aia_map.meta:
        pixel_scale = abs(aia_map.meta['cdelt1'])  # arcsec/pixel
    else:
        pixel_scale = 0.6  # デフォルト値
    
    solar_radius_pixels = rsun_arcsec / pixel_scale
    
    # 内側半径の円を描画
    theta = np.linspace(0, 2*np.pi, 360)
    inner_radius_pixels = inner_radius_rs * solar_radius_pixels
    circle_x = inner_radius_pixels * np.cos(theta)
    circle_y = inner_radius_pixels * np.sin(theta)
    ax.plot(circle_x, circle_y, '--', color='yellow', linewidth=2, 
            label=f'{inner_radius_rs} Rs boundary')
    
    # 1Rs円も描画
    rs_circle_x = solar_radius_pixels * np.cos(theta)
    rs_circle_y = solar_radius_pixels * np.sin(theta)
    ax.plot(rs_circle_x, rs_circle_y, '-', color='white', linewidth=1, 
            label='1 Rs (Solar limb)')
    
    return {
        'center_x': center_x,
        'center_y': center_y,
        'solar_radius_pixels': solar_radius_pixels,
        'pixel_scale': pixel_scale,
        'extent': extent,
        'im': im
    }

def create_solar_radius_mask(data_shape, center_x, center_y, solar_radius_pixels, 
                           inner_radius_rs=1.7):
    """
    太陽半径に基づくマスクを作成
    
    Parameters
    ----------
    data_shape : tuple
        データの形状 (height, width)
    center_x, center_y : float
        太陽中心のピクセル座標
    solar_radius_pixels : float
        太陽半径（ピクセル単位）
    inner_radius_rs : float
        内側半径（太陽半径単位）
        
    Returns
    -------
    dict
        各領域のマスク
    """
    height, width = data_shape
    y_idx, x_idx = np.indices((height, width))
    
    # 太陽中心からの距離（ピクセル単位）
    distance = np.sqrt((x_idx - center_x)**2 + (y_idx - center_y)**2)
    
    # 太陽半径単位に変換
    distance_rs = distance / solar_radius_pixels
    
    # マスクを作成
    inner_mask = distance_rs <= inner_radius_rs  # 1.7Rs以内（AIA 304領域）
    outer_mask = distance_rs > inner_radius_rs   # 1.7Rs以上（UCoMP領域）
    
    return {
        'inner_mask': inner_mask,
        'outer_mask': outer_mask,
        'distance_rs': distance_rs
    }

def plot_ucomp_sdo_integrated(target_time, wavelength=None, 
                             smooth_ext12=False, ext12_sigma=1.0, 
                             save_path=None, figsize=(12, 10)):
    """
    UCoMP Ext12磁場データとSDO/AIA 304データを統合してプロット
    
    Parameters
    ----------
    target_time : str or Time
        プロット対象時刻（自動的に前後30分の範囲でデータを検索）
    wavelength : int, optional
        UCoMP波長
    smooth_ext12 : bool, optional
        Extension 12の平滑化を適用するか
    ext12_sigma : float, optional
        Extension 12の平滑化シグマ値
    save_path : str, optional
        保存パス
    figsize : tuple, optional
        図のサイズ
        
    Returns
    -------
    matplotlib.figure.Figure
        作成された図
    """
    print("=" * 60)
    print("UCoMP + SDO/AIA 304 Integrated Plot")
    print("=" * 60)
    
    # 時間範囲の自動設定（target_timeの前後30分）
    if isinstance(target_time, str):
        target_time_obj = Time(target_time)
    else:
        target_time_obj = target_time
    
    start_time = target_time_obj - 30 * u.min
    end_time = target_time_obj + 30 * u.min
    
    print(f"Target time: {target_time_obj.iso}")
    print(f"Search range: {start_time.iso} - {end_time.iso}")
    
    # 1. UCoMP Ext12データの取得
    print("\n--- Loading UCoMP Ext12 data ---")
    closest_ucomp_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
    
    if closest_ucomp_data is None:
        print("No UCoMP data found for the specified time range")
        return None
    
    ucomp_info, ucomp_file_path = closest_ucomp_data
    ucomp_extensions = read_ucomp_extensions(ucomp_file_path, max_extensions=12)
    
    if 12 not in ucomp_extensions:
        print("Extension 12 not found in UCoMP data")
        return None
    
    ucomp_ext12_data, ucomp_ext12_header = ucomp_extensions[12]
    
    # Extension 12の平滑化処理
    if smooth_ext12:
        ucomp_ext12_data = smooth_ext12_data(ucomp_ext12_data, smooth=True, sigma=ext12_sigma)
    
    # 2. SDO/AIA 304データの取得
    print("\n--- Loading SDO/AIA 304 data ---")
    closest_aia_data = find_closest_aia_304_data(target_time, time_window_minutes=30)
    
    if closest_aia_data is None:
        print("No AIA 304 data found for the specified time range")
        return None
    
    aia_map, aia_file_path = closest_aia_data
    
    # 3. プロットの準備
    fig, ax = plt.subplots(figsize=figsize)
    
    # 4. AIA 304データのプロット（背景）
    print("\n--- Plotting AIA 304 data ---")
    aia_params = plot_sdo_aia_304_with_wcs(aia_map, ax, inner_radius_rs=SOLAR_RADIUS_THRESHOLD)
    
    # 5. UCoMP Ext12データの座標系変換とプロット
    print("\n--- Processing UCoMP Ext12 data ---")
    
    # UCoMP Ext12データの中心とスケーリング
    ucomp_center_x = ucomp_ext12_data.shape[1] // 2
    ucomp_center_y = ucomp_ext12_data.shape[0] // 2
    ucomp_solar_radius_pixels = min(ucomp_ext12_data.shape) // 4
    
    # 太陽半径マスクの作成
    mask_info = create_solar_radius_mask(
        ucomp_ext12_data.shape, 
        ucomp_center_x, 
        ucomp_center_y, 
        ucomp_solar_radius_pixels, 
        inner_radius_rs=SOLAR_RADIUS_THRESHOLD
    )
    
    # UCoMP Ext12データをAIA座標系に変換
    # 座標系のスケーリング比を計算
    scale_factor = aia_params['solar_radius_pixels'] / ucomp_solar_radius_pixels
    
    # UCoMP Ext12データをAIA座標系でプロット
    ucomp_extent = [
        -ucomp_center_x * scale_factor, 
        (ucomp_ext12_data.shape[1] - ucomp_center_x) * scale_factor,
        -ucomp_center_y * scale_factor, 
        (ucomp_ext12_data.shape[0] - ucomp_center_y) * scale_factor
    ]
    
    # 外側マスクを適用したUCoMP Ext12データをプロット
    masked_ucomp_data = np.where(mask_info['outer_mask'], ucomp_ext12_data, np.nan)
    
    # Ext12の表示範囲設定
    vmin, vmax = -45, 45
    
    im_ucomp = ax.imshow(masked_ucomp_data, origin='lower', cmap='RdBu_r', 
                        vmin=vmin, vmax=vmax, extent=ucomp_extent, 
                        aspect='equal', alpha=0.8)
    
    # 6. UCoMP磁場ベクトルの描画
    print("\n--- Drawing magnetic field vectors ---")
    
    # ベクトル場グリッドを作成（1.7Rs以上の領域のみ）
    x_grid, y_grid = create_vector_field_grid(
        ucomp_ext12_data.shape, 
        radial_interval_rs=0.15, 
        angular_interval_deg=10
    )
    
    # グリッドをAIA座標系に変換
    x_grid_aia = x_grid * scale_factor
    y_grid_aia = y_grid * scale_factor
    
    # 1.7Rs以上の領域のみをフィルタリング
    grid_distance_rs = np.sqrt(x_grid**2 + y_grid**2) / ucomp_solar_radius_pixels
    valid_grid_mask = grid_distance_rs >= SOLAR_RADIUS_THRESHOLD
    
    if np.any(valid_grid_mask):
        x_grid_filtered = x_grid_aia[valid_grid_mask]
        y_grid_filtered = y_grid_aia[valid_grid_mask]
        
        # 磁場ベクトルを描画
        draw_magnetic_field_vectors(
            ax, ucomp_ext12_data, masked_ucomp_data, 
            x_grid_filtered, y_grid_filtered,
            arrow_length_scale=20 * scale_factor, 
            arrow_width=0.004, 
            arrow_color='white',
            arrow_alpha=0.9
        )
    
    # 7. カラーバーの追加
    divider = make_axes_locatable(ax)
    cax_aia = divider.append_axes("right", size="2%", pad=0.1)
    cbar_aia = plt.colorbar(aia_params['im'], cax=cax_aia)
    cbar_aia.set_label('AIA 304 Intensity', fontsize=10)
    
    cax_ucomp = divider.append_axes("right", size="2%", pad=0.3)
    cbar_ucomp = plt.colorbar(im_ucomp, cax=cax_ucomp)
    cbar_ucomp.set_label('UCoMP Ext12 Azimuth [deg]', fontsize=10)
    
    # 8. 軸の設定
    ax.set_xlim(-800, 800)
    ax.set_ylim(-800, 800)
    ax.set_xlabel('X [pixels from Sun center]', fontsize=12)
    ax.set_ylabel('Y [pixels from Sun center]', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 9. タイトルと凡例
    title_parts = [
        f'UCoMP Ext12 + SDO/AIA 304 Integrated Plot',
        f'UCoMP: {ucomp_info["date"].strftime("%Y-%m-%d %H:%M:%S")}',
        f'AIA 304: {aia_map.date.strftime("%Y-%m-%d %H:%M:%S")}',
        f'Inner region (<{SOLAR_RADIUS_THRESHOLD}Rs): AIA 304',
        f'Outer region (≥{SOLAR_RADIUS_THRESHOLD}Rs): UCoMP Ext12'
    ]
    
    if smooth_ext12:
        title_parts.append(f'UCoMP Ext12 smoothed (σ={ext12_sigma})')
    
    ax.set_title('\n'.join(title_parts), fontsize=12, pad=20)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # 10. 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    print("\n--- Integration completed ---")
    return fig

def main():
    """
    メイン実行関数
    """
    print("UCoMP + SDO/AIA 304 Integration Tool")
    print("=" * 50)
    
    # デフォルト設定
    default_target_time = "2022-06-13T03:36:00"
    
    print(f"Default target time: {default_target_time}")
    print("Data will be searched within ±30 minutes of target time")
    
    # 統合プロットの実行
    fig = plot_ucomp_sdo_integrated(
        target_time=default_target_time,
        wavelength=None,
        smooth_ext12=True,
        ext12_sigma=1.0,
        save_path="/mnt/d/wsl/home/kinno-7010/Research/MK4_coronagraph/UCOMP/ucomp_sdo_integrated.png"
    )
    
    if fig:
        plt.show()
    else:
        print("Failed to create integrated plot")

if __name__ == "__main__":
    main()