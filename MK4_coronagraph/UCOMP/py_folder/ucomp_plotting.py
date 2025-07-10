"""
UCOMPデータ描画関数
Level 2 FITSのext1~12を3×4の画像列で描画
"""

from ucomp_config import *
from ucomp_scanner import find_closest_ucomp_data
from astropy.io import fits
import gc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
from scipy.ndimage import map_coordinates

# ベクトル場描画関数をインポート
import numpy as np

def create_vector_field_grid(data_shape, rsun_obs, cdelt1, crpix1, crpix2, 
                             radial_interval_rs=0.1, angular_interval_deg=5):
    """
    太陽中心座標系でのベクトル場グリッドを作成
    """
    height, width = data_shape
    center_x, center_y = width // 2, height // 2
    
    # 太陽半径をピクセル単位で計算
    solar_radius_pixels = rsun_obs / cdelt1
    
    # 動径方向の範囲を設定（0.5Rsから2.5Rsまで）
    r_min_rs = 0.5
    r_max_rs = 2.5
    
    # 動径方向のグリッド（太陽半径単位）
    r_rs_values = np.arange(r_min_rs, r_max_rs + radial_interval_rs, radial_interval_rs)
    
    # 角度方向のグリッド（度→ラジアン変換）
    theta_deg_values = np.arange(0, 360, angular_interval_deg)
    theta_rad_values = np.deg2rad(theta_deg_values)
    
    # グリッド座標を計算
    x_grid = []
    y_grid = []
    
    for r_rs in r_rs_values:
        # 太陽半径単位をピクセル単位に変換
        r_pixels = r_rs * solar_radius_pixels
        
        for theta_rad in theta_rad_values:
            # 極座標から直交座標への変換（太陽中心原点）
            x = r_pixels * np.cos(theta_rad)
            y = r_pixels * np.sin(theta_rad)
            
            # データ範囲内かチェック
            x_pixel = x + center_x
            y_pixel = y + center_y
            
            if (0 <= x_pixel < width) and (0 <= y_pixel < height):
                x_grid.append(x)  # 太陽中心原点座標
                y_grid.append(y)  # 太陽中心原点座標
    
    return np.array(x_grid), np.array(y_grid)


def draw_magnetic_field_vectors(ax, data, azimuth_data, x_grid, y_grid, crpix1, crpix2,
                               arrow_length_scale=20, arrow_width=0.004, 
                               arrow_color='white', arrow_alpha=0.9):
    """
    磁場ベクトル場を矢印で描画（vector_field.py手法を適用した改善版）
    """
    if azimuth_data is None or azimuth_data.size == 0:
        print("Warning: No azimuth data available for vector field")
        return
    
    height, width = data.shape
    center_x, center_y = width // 2, height // 2
    
    # 各グリッド点で方位角を取得してベクトルを描画
    for i, (x_pos, y_pos) in enumerate(zip(x_grid, y_grid)):
        # 太陽中心原点座標をピクセル座標に変換
        pixel_x = int(x_pos + center_x)
        pixel_y = int(y_pos + center_y)
        
        # データ範囲内かチェック
        if (0 <= pixel_x < width) and (0 <= pixel_y < height):
            # 方位角を取得（度単位）
            azimuth_deg = azimuth_data[pixel_y, pixel_x]
            
            # NaNや無効値をスキップ
            if not np.isfinite(azimuth_deg):
                continue
            
            # vector_field.pyの手法を適用
            # 1. 法線方向角度を計算（円の中心からの方向）
            normal_angle = np.arctan2(y_pos, x_pos)
            
            # 2. 方位角データを偏角として扱う
            deviation_angle = np.deg2rad(azimuth_deg)
            
            # 3. 最終的なベクトル角度 = 法線角度 + 偏角
            vector_angle = normal_angle + deviation_angle
            
            # 4. ベクトル成分を計算
            dx = arrow_length_scale * np.cos(vector_angle)
            dy = arrow_length_scale * np.sin(vector_angle)
            
            # ベクトル参照点を黒い小さな点で表示
            ax.plot(x_pos, y_pos, 'o', markersize=1, color='white')
            
            # 矢印を描画（太陽中心原点座標系、黒枠付き）
            ax.arrow(x_pos, y_pos, dx, dy, 
                    head_width=arrow_length_scale*0.3, 
                    head_length=arrow_length_scale*0.2,
                    fc=arrow_color, ec=arrow_color,
                    alpha=arrow_alpha, linewidth=arrow_width*700,
                    length_includes_head=True)
    
    print(f"Drew {len(x_grid)} magnetic field vectors")

def get_data_range(data, percentile_range=[1, 99]):
    """
    UCOMPデータの表示範囲を取得（正規化なし）
    """
    if data is None or data.size == 0:
        return (0, 1)
    
    # NaNや無限値を除外
    valid_data = data[np.isfinite(data)]
    
    if len(valid_data) == 0:
        return (0, 1)
    
    # パーセンタイルで範囲を決定
    vmin, vmax = np.percentile(valid_data, percentile_range)
    
    return (vmin, vmax)

def smooth_ext12_data(data, smooth=True, sigma=1.0, kernel_size=None):
    """
    Extension 12データの平滑化
    """
    if not smooth or data is None or data.size == 0:
        return data
    
    try:
        # NaNや無限値のマスクを作成
        valid_mask = np.isfinite(data)
        
        if not np.any(valid_mask):
            print("Warning: No valid data points for smoothing")
            return data
        
        # カーネルサイズの自動計算
        if kernel_size is None:
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)  # 3σ範囲をカバー
        
        # 有効なデータのみを平滑化
        smoothed_data = data.copy()
        
        # ガウシアンフィルタを適用（NaNは無視）
        if np.all(valid_mask):
            # 全データが有効な場合
            smoothed_data = ndimage.gaussian_filter(data, sigma=sigma)
        else:
            # NaNがある場合の処理
            smoothed_data = ndimage.gaussian_filter(
                np.where(valid_mask, data, 0), sigma=sigma
            )
            # 重みマップも作成して正規化
            weight_map = ndimage.gaussian_filter(
                valid_mask.astype(float), sigma=sigma
            )
            # ゼロ除算を避けて正規化
            smoothed_data = np.where(
                weight_map > 0,
                smoothed_data / weight_map,
                data
            )
            # 元々NaNだった場所はNaNに戻す
            smoothed_data = np.where(valid_mask, smoothed_data, np.nan)
        
        print(f"Applied Gaussian smoothing to Ext12 data (sigma={sigma}, kernel_size={kernel_size})")
        return smoothed_data
        
    except Exception as e:
        print(f"Warning: Smoothing failed ({e}), using original data")
        return data

def read_ucomp_extensions(file_path, max_extensions=12):
    """
    UCOMPファイルからext1~12を読み込む
    
    Parameters
    ----------
    file_path : str or Path
        UCOMPファイルのパス
    max_extensions : int
        読み込む最大Extension数 (default: 12)
        
    Returns
    -------
    dict
        {extension_number: (data, header), ...}
    """
    extensions = {}
    
    print(f"Reading UCOMP file: {Path(file_path).name}")
    
    try:
        with fits.open(file_path) as hdul:
            print(f"Total HDUs: {len(hdul)}")
            
            # ext1~12を読み込み
            for i in range(1, min(len(hdul), max_extensions + 1)):
                try:
                    data = hdul[i].data
                    header = hdul[i].header
                    extensions[i] = (data, header)
                    print(f"  Ext {i}: {data.shape} - {get_extension_title(i)}")
                except Exception as e:
                    print(f"  Ext {i}: Error reading - {e}")
                    
    except Exception as e:
        print(f"Error opening UCOMP file: {e}")
        return {}
    
    return extensions

def normalize_ucomp_data(data, percentile_range=[1, 99]):
    """
    UCOMPデータを正規化
    
    Parameters
    ----------
    data : np.ndarray
        画像データ
    percentile_range : list
        パーセンタイル範囲 [min, max]
        
    Returns
    -------
    np.ndarray
        正規化されたデータ
    """
    if data is None or data.size == 0:
        return np.zeros((100, 100))
    
    # NaNや無限値を除外
    valid_data = data[np.isfinite(data)]
    
    if len(valid_data) == 0:
        return np.zeros_like(data)
    
    # パーセンタイルで範囲を決定
    vmin, vmax = np.percentile(valid_data, percentile_range)
    
    # 正規化
    norm = ImageNormalize(data, vmin=vmin, vmax=vmax, stretch=LinearStretch(), clip=True)
    normalized = norm(data)
    
    return normalized

def plot_ucomp_extensions(target_time, start_time, end_time, wavelength=DEFAULT_WAVELENGTH, 
                         save_path=None, figsize=(15, 12)):
    """
    指定時刻のUCOMPデータのext1~12を3×4で描画
    
    Parameters
    ----------
    target_time : str or Time
        プロット対象時刻
    start_time : str or Time
        スキャン開始時刻
    end_time : str or Time
        スキャン終了時刻
    wavelength : int
        波長 (default: 1074)
    save_path : str, optional
        保存パス
    figsize : tuple
        図のサイズ (default: (15, 12))
        
    Returns
    -------
    matplotlib.figure.Figure
        作成された図
    """
    # 最も近いデータを見つける
    closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
    
    if closest_data is None:
        print("No UCOMP data found for the specified time range")
        return None
    
    closest_info, file_path = closest_data
    
    # Extensionを読み込み
    extensions = read_ucomp_extensions(file_path)
    
    if not extensions:
        print("Failed to read UCOMP extensions")
        return None
    
    # 3×4のサブプロットを作成
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    fig.suptitle(f'UCOMP {wavelength}nm Level 2 Data\n'
                f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}\n'
                f'File: {closest_info["filename"]}', 
                fontsize=14, y=0.95)
    
    # 各Extensionをプロット
    for i in range(12):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        ext_num = i + 1
        
        if ext_num in extensions:
            data, header = extensions[ext_num]
            
            # データを正規化
            normalized_data = normalize_ucomp_data(data)
            
            # プロット
            im = ax.imshow(normalized_data, origin='lower', cmap='plasma', aspect='equal')
            
            # カラーバーを追加
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        else:
            # データがない場合はダミー画像
            ax.imshow(np.zeros((100, 100)), origin='lower', cmap='gray', aspect='equal')
            ax.text(50, 50, 'No Data', ha='center', va='center', 
                   fontsize=12, color='white', weight='bold')
        
        # タイトルとラベル設定
        ax.set_title(f'Ext {ext_num}: {get_extension_title(ext_num)}', fontsize=10)
        ax.set_xlabel('X [pixels]', fontsize=8)
        ax.set_ylabel('Y [pixels]', fontsize=8)
        
        # 太陽円盤の境界を描画（シアンの点線）
        if ext_num in extensions:
            data, _ = extensions[ext_num]
            if data is not None and data.size > 0:
                center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
                # 太陽半径を仮定（実際の値は観測データのヘッダーから取得すべき）
                solar_radius = min(data.shape) // 4
                circle = plt.Circle((center_x, center_y), solar_radius, 
                                  fill=False, color='cyan', linewidth=2, linestyle='--', alpha=0.8)
                ax.add_patch(circle)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    # メモリ解放
    del extensions
    gc.collect()
    
    return fig

def plot_single_extension(ax, target_time, start_time, end_time, extension_num=1, 
                         wavelength=DEFAULT_WAVELENGTH, save_path=None, 
                         smooth_ext12=False, ext12_sigma=1.0):
    """
    指定Extension単体を詳細プロット
    
    Parameters
    ----------
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
        
    Returns
    -------
    matplotlib.figure.Figure
        作成された図
    """
    if extension_num < 1 or extension_num > 12:
        raise ValueError("Extension number must be between 1 and 12")
    
    # 最も近いデータを見つける
    closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
    
    if closest_data is None:
        print("No UCOMP data found for the specified time range")
        return None
    
    closest_info, file_path = closest_data
    
    # 指定Extensionを読み込み
    extensions = read_ucomp_extensions(file_path, max_extensions=extension_num)
    
    if extension_num not in extensions:
        print(f"Extension {extension_num} not found in the file")
        return None
    
    data, header = extensions[extension_num]
    
    # プロット
    # fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extension 12用の特別処理
    if extension_num == 12:
        # Extension 12の平滑化処理
        processed_data = smooth_ext12_data(data, smooth=smooth_ext12, sigma=ext12_sigma)
        
        # Extension 12は規格化しない、hsvカラーマップ使用
        vmin, vmax = -45, 45
        cmap = 'RdBu_r'
        colorbar_label = get_colorbar_label(extension_num)
        
        # 太陽中心を原点とする座標系
        center_x, center_y = processed_data.shape[1] // 2, processed_data.shape[0] // 2
        extent = [-center_x, center_x, -center_y, center_y]
        
        # 画像表示
        im = ax.imshow(processed_data, origin='lower', cmap=cmap, aspect='equal', 
                      vmin=vmin, vmax=vmax, extent=extent)
    else:
        # 他のExtensionは従来通り正規化
        normalized_data = normalize_ucomp_data(data)
        im = ax.imshow(normalized_data, origin='lower', cmap='plasma', aspect='equal')
        colorbar_label = 'Normalized Intensity'
    
    # カラーバー
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(colorbar_label, fontsize=12)
    
    # タイトルとラベル
    ax.set_title(f'Mk4/UCOMP - Ext {extension_num}: {get_extension_title(extension_num)}\n'
                f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', fontsize=14)
    ax.set_xlabel('X [pixels]', fontsize=10)
    ax.set_ylabel('Y [pixels]', fontsize=10)
    ax.set_xlim(-650, 650)
    ax.set_ylim(-550, 550)
    
    # 太陽円盤の境界を描画（シアンの点線）
    if data is not None and data.size > 0:
        if extension_num == 12:
            # Extension 12では太陽中心原点座標系の円
            solar_radius = min(data.shape) // 4
            circle = plt.Circle((0, 0), solar_radius, 
                              fill=False, color='cyan', linewidth=2, linestyle='--', alpha=0.8)
            ax.add_patch(circle)
            
            # Extension 12には磁場ベクトル場を追加
            x_grid, y_grid = create_vector_field_grid(processed_data.shape, 
                                                     radial_interval_rs=0.1, 
                                                     angular_interval_deg=5)
            draw_magnetic_field_vectors(ax, processed_data, processed_data, x_grid, y_grid,
                                      arrow_length_scale=15, 
                                      arrow_color='black',
                                      arrow_alpha=0.9)
        else:
            # 他のExtensionでは従来の座標系
            center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
            solar_radius = min(data.shape) // 4
            circle = plt.Circle((center_x, center_y), solar_radius, 
                              fill=False, color='cyan', linewidth=2, linestyle='--', alpha=0.8)
            ax.add_patch(circle)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    # return fig

def plot_ext_12_diff(ax, target_start_time, target_end_time, start_time, end_time, extension_num=12, 
                         wavelength=DEFAULT_WAVELENGTH, save_path='/mnt/d/wsl/home/kinno-7010/Research/MK4_coronagraph/UCOMP/ucomp_diff_plot.png', 
                         smooth_ext12=False, ext12_sigma=1.0):
    
    # 最も近いデータを見つける
    closest_start_data = find_closest_ucomp_data(target_start_time, start_time, end_time, wavelength)
    closest_end_data = find_closest_ucomp_data(target_end_time, start_time, end_time, wavelength)
    
    closest_start_info, start_file_path = closest_start_data
    closest_end_info, end_file_path = closest_end_data
    
    # 指定Extensionを読み込み
    start_extensions = read_ucomp_extensions(start_file_path, max_extensions=extension_num)
    end_extensions = read_ucomp_extensions(end_file_path, max_extensions=extension_num)
    
    start_data, start_header = start_extensions[extension_num]
    end_data, end_header = end_extensions[extension_num]
    
    diff_data = end_data - start_data
    # プロット
    # fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extension 12用の特別処理
    if extension_num == 12:
        # Extension 12の平滑化処理
        processed_data = smooth_ext12_data(diff_data, smooth=smooth_ext12, sigma=ext12_sigma)
        
        # Extension 12は規格化しない
        vmin, vmax = -45, 45
        cmap = 'RdBu_r'
        colorbar_label = "Radial Azimuthal differencial angle [deg] $\\Delta \\phi$"
        
        # 太陽中心を原点とする座標系
        center_x, center_y = processed_data.shape[1] // 2, processed_data.shape[0] // 2
        extent = [-center_x, center_x, -center_y, center_y]
        
        # 画像表示
        im = ax.imshow(processed_data, origin='lower', cmap=cmap, aspect='equal', 
                      vmin=vmin, vmax=vmax, extent=extent)
    
    # カラーバー
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(colorbar_label, fontsize=12)
    
    # タイトルとラベル
    ax.set_title(f'Mk4/UCOMP diff - Ext {extension_num}: {get_extension_title(extension_num)}\n'
                f'Time: {closest_start_info["date"].strftime("%Y-%m-%d %H:%M:%S")} - {closest_end_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', fontsize=14)
    ax.set_xlabel('X [pixels]', fontsize=10)
    ax.set_ylabel('Y [pixels]', fontsize=10)
    ax.set_xlim(-650, 650)
    ax.set_ylim(-550, 550)
    
    # 太陽円盤の境界を描画（シアンの点線）
    if processed_data is not None and processed_data.size > 0:
        if extension_num == 12:
            # Extension 12では太陽中心原点座標系の円
            solar_radius = min(processed_data.shape) // 4
            circle = plt.Circle((0, 0), solar_radius, 
                              fill=False, color='cyan', linewidth=2, linestyle='--', alpha=0.8)
            ax.add_patch(circle)
            
            # Extension 12には磁場ベクトル場を追加
            x_grid, y_grid = create_vector_field_grid(processed_data.shape, 
                                                     radial_interval_rs=0.1, 
                                                     angular_interval_deg=5)
            draw_magnetic_field_vectors(ax, processed_data, processed_data, x_grid, y_grid,
                                      arrow_length_scale=15, 
                                      arrow_color='black',
                                      arrow_alpha=0.9)
        else:
            # 他のExtensionでは従来の座標系
            center_x, center_y = processed_data.shape[1] // 2, processed_data.shape[0] // 2
            solar_radius = min(processed_data.shape) // 4
            circle = plt.Circle((center_x, center_y), solar_radius, 
                              fill=False, color='cyan', linewidth=2, linestyle='--', alpha=0.8)
            ax.add_patch(circle)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    # return fig

def create_ucomp_summary_plot(target_time, start_time, end_time, wavelength=DEFAULT_WAVELENGTH):
    """
    UCOMPデータのサマリープロットを作成
    主要なExtension（1,2,4,10）のみを表示
    
    Parameters
    ----------
    target_time : str or Time
        プロット対象時刻
    start_time : str or Time
        スキャン開始時刻
    end_time : str or Time
        スキャン終了時刻
    wavelength : int
        波長
        
    Returns
    -------
    matplotlib.figure.Figure
        作成された図
    """
    # 主要なExtensionのみ
    key_extensions = [1, 2, 4, 10]  # Center Line, Enhanced, Doppler, Linear Pol
    
    closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
    
    if closest_data is None:
        print("No UCOMP data found")
        return None
    
    closest_info, file_path = closest_data
    extensions = read_ucomp_extensions(file_path)
    
    # 2×2のサブプロット
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'UCOMP {wavelength}nm Key Parameters\n'
                f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', 
                fontsize=14)
    
    for i, ext_num in enumerate(key_extensions):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        if ext_num in extensions:
            data, _ = extensions[ext_num]
            normalized_data = normalize_ucomp_data(data)
            
            im = ax.imshow(normalized_data, origin='lower', cmap='plasma', aspect='equal')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # 太陽円盤
            if data is not None and data.size > 0:
                center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
                solar_radius = min(data.shape) // 4
                circle = plt.Circle((center_x, center_y), solar_radius, 
                                  fill=False, color='cyan', linewidth=2, linestyle='--', alpha=0.8)
                ax.add_patch(circle)
        else:
            ax.imshow(np.zeros((100, 100)), origin='lower', cmap='gray')
            ax.text(50, 50, 'No Data', ha='center', va='center', 
                   fontsize=12, color='white', weight='bold')
        
        ax.set_title(f'Ext {ext_num}: {get_extension_title(ext_num)}', fontsize=11)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
    
    plt.tight_layout()
    return fig