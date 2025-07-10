from ucomp_plotting import *
from ucomp_config import *
from ucomp_scanner import find_closest_ucomp_data
from astropy.io import fits
import gc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

def plot_single_extension(ax, target_time, start_time, end_time, extension_num=12, 
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
    
    rsun_obs, cdelt1, crpix1, crpix2 = get_header_info(file_path)
    
    # 正確な太陽半径をピクセル単位で計算
    solar_radius_pixels = rsun_obs / cdelt1
    solar_center = (crpix1, crpix2)
    
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
        extent = [-solar_center[0], solar_center[0], -solar_center[1], solar_center[1]]
        
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
                f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")} ({ext12_sigma}$\\sigma$)', fontsize=14)
    ax.set_xlabel('X [pixels]', fontsize=10)
    ax.set_ylabel('Y [pixels]', fontsize=10)
    ax.set_xlim(-650, 650)
    ax.set_ylim(-550, 550)
    
    # 太陽円盤の境界を描画（シアンの点線）
    if data is not None and data.size > 0:
        if extension_num == 12:
            # Extension 12では太陽中心原点座標系の円（正確な太陽半径を使用）
            circle = plt.Circle((0, 0), solar_radius_pixels, 
                              fill=False, color='cyan', linewidth=2, linestyle='--', alpha=0.8)
            ax.add_patch(circle)
            
            # Extension 12には磁場ベクトル場を追加
            x_grid, y_grid = create_vector_field_grid(processed_data.shape, rsun_obs, cdelt1, crpix1, crpix2, 
                                                     radial_interval_rs=0.1, 
                                                     angular_interval_deg=5)
            draw_magnetic_field_vectors(ax, data, processed_data, x_grid, y_grid, crpix1, crpix2,
                                      arrow_length_scale=15, 
                                      arrow_color='black',
                                      arrow_alpha=0.9)
        else:
            # 他のExtensionでは従来の座標系（正確な太陽半径を使用）
            center_x, center_y = crpix1, crpix2
            circle = plt.Circle((center_x, center_y), solar_radius_pixels, 
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
    
    # メインヘッダーから太陽半径情報を取得
    rsun_obs, cdelt1, crpix1, crpix2 = get_header_info(start_file_path)
    
    # 正確な太陽半径をピクセル単位で計算
    solar_radius_pixels = rsun_obs / cdelt1
    solar_center = (crpix1, crpix2)
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
        extent = [-solar_center[0], solar_center[0], -solar_center[1], solar_center[1]]
        
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
                f'Time: {closest_start_info["date"].strftime("%Y-%m-%d %H:%M:%S")} - {closest_end_info["date"].strftime("%Y-%m-%d %H:%M:%S")} ({ext12_sigma}$\\sigma$)', fontsize=14)
    ax.set_xlabel('X [pixels]', fontsize=10)
    ax.set_ylabel('Y [pixels]', fontsize=10)
    ax.set_xlim(-650, 650)
    ax.set_ylim(-550, 550)
    
    # 太陽円盤の境界を描画（シアンの点線）
    if processed_data is not None and processed_data.size > 0:
        if extension_num == 12:
            # Extension 12では太陽中心原点座標系の円（正確な太陽半径を使用）
            circle = plt.Circle((0, 0), solar_radius_pixels, 
                              fill=False, color='cyan', linewidth=2, linestyle='--', alpha=0.8)
            ax.add_patch(circle)
            
            # Extension 12には磁場ベクトル場を追加
            x_grid, y_grid = create_vector_field_grid(processed_data.shape, rsun_obs, cdelt1, crpix1, crpix2, 
                                                     radial_interval_rs=0.1, 
                                                     angular_interval_deg=5)
            draw_magnetic_field_vectors(ax, diff_data, processed_data, x_grid, y_grid, crpix1, crpix2,
                                      arrow_length_scale=15, 
                                      arrow_color='black',
                                      arrow_alpha=0.9)
        else:
            # 他のExtensionでは従来の座標系（正確な太陽半径を使用）
            center_x, center_y = crpix1, crpix2
            circle = plt.Circle((center_x, center_y), solar_radius_pixels, 
                              fill=False, color='cyan', linewidth=2, linestyle='--', alpha=0.8)
            ax.add_patch(circle)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")


if __name__ == "__main__":
    start_time = "2022-06-13T03:00:00"
    end_time = "2022-06-13T04:01:00"
    ext_num = 12
    smooth_ext12 = True
    ext12_sigma = 1.0
    
    fig, axes = plt.subplots(1,3,figsize=(27,8),tight_layout=True)
    for ax, target_time in zip(axes[0:2], ["2022-06-13T03:06:00", "2022-06-13T03:36:00"]):
        plot_single_extension(ax, target_time, start_time, end_time, ext_num, None,
                                smooth_ext12=smooth_ext12, ext12_sigma=ext12_sigma)

    plot_ext_12_diff(axes[2], "2022-06-13T03:06:00", "2022-06-13T03:36:00", start_time, end_time, extension_num=ext_num, wavelength=None, smooth_ext12=smooth_ext12, ext12_sigma=ext12_sigma)
    plt.show()