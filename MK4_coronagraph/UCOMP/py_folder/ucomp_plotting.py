"""
UCOMPデータ描画関数
Level 2 FITSのext1~12を3×4の画像列で描画
"""

from ucomp_config import *
from ucomp_scanner import find_closest_ucomp_data
from astropy.io import fits
import gc
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def plot_single_extension(target_time, start_time, end_time, extension_num=1, 
                         wavelength=DEFAULT_WAVELENGTH, save_path=None):
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
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # データを正規化
    normalized_data = normalize_ucomp_data(data)
    
    # 画像表示
    im = ax.imshow(normalized_data, origin='lower', cmap='plasma', aspect='equal')
    
    # カラーバー
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1, shrink=0.5)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Normalized Intensity', fontsize=12)
    
    # タイトルとラベル
    ax.set_title(f'UCOMP {wavelength}nm - Ext {extension_num}: {get_extension_title(extension_num)}\n'
                f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', fontsize=12)
    ax.set_xlabel('X [pixels]', fontsize=10)
    ax.set_ylabel('Y [pixels]', fontsize=10)
    
    # 太陽円盤の境界を描画（シアンの点線）
    if data is not None and data.size > 0:
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
    
    return fig

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