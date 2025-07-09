"""
UCOMP Extension グループ別描画関数
ext1~3, 4~6, 7~10, 11~12 に分けた描画機能
"""

from ucomp_config import *
from ucomp_scanner import find_closest_ucomp_data, get_available_ucomp_times
from astropy.io import fits
import gc
import matplotlib.animation as animation
from datetime import timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess
import tempfile
import shutil
from scipy import ndimage

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

def get_data_range(data, percentile_range=[1, 99]):
    """
    UCOMPデータの表示範囲を取得（正規化なし）
    
    Parameters
    ----------
    data : np.ndarray
        画像データ
    percentile_range : list
        パーセンタイル範囲 [min, max]
        
    Returns
    -------
    tuple
        (vmin, vmax) 表示範囲
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

def draw_solar_radius(ax, data, color='cyan', linewidth=2, linestyle='--', alpha=0.8):
    """太陽半径を描画（太陽中心原点座標系）"""
    if data is not None and data.size > 0:
        solar_radius = min(data.shape) // 4
        circle = plt.Circle((0, 0), solar_radius, 
                          fill=False, color=color, linewidth=linewidth, 
                          linestyle=linestyle, alpha=alpha)
        ax.add_patch(circle)

def create_vector_field_grid(data_shape, radial_interval_rs=0.1, angular_interval_deg=5):
    """
    太陽中心座標系でのベクトル場グリッドを作成
    
    Parameters
    ----------
    data_shape : tuple
        データの形状 (height, width)
    radial_interval_rs : float
        動径方向の間隔（太陽半径単位） (default: 0.1)
    angular_interval_deg : float
        角度方向の間隔（度） (default: 5)
        
    Returns
    -------
    tuple
        (x_grid, y_grid) ベクトル場のグリッド座標（ピクセル単位）
    """
    height, width = data_shape
    center_x, center_y = width // 2, height // 2
    
    # 太陽半径をピクセル単位で計算
    solar_radius_pixels = min(data_shape) // 4
    
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

def draw_magnetic_field_vectors(ax, data, azimuth_data, x_grid, y_grid, 
                               arrow_length_scale=20, arrow_width=0.004, 
                               arrow_color='white', arrow_alpha=0.8):
    """
    磁場ベクトル場を矢印で描画（vector_field.py手法を適用した改善版）
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        描画対象のaxes
    data : np.ndarray
        画像データ（形状確認用）
    azimuth_data : np.ndarray
        方位角データ（Extension 12）
    x_grid, y_grid : np.ndarray
        ベクトル場のグリッド座標（太陽中心原点）
    arrow_length_scale : float
        矢印の長さスケール (default: 20)
    arrow_width : float
        矢印の幅 (default: 0.004)
    arrow_color : str
        矢印の色 (default: 'white')
    arrow_alpha : float
        矢印の透明度 (default: 0.8)
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
            ax.plot(x_pos, y_pos, 'ko', markersize=2, alpha=0.8)
            
            # 矢印を描画（太陽中心原点座標系、白い塗りつぶし、黒枠線）
            ax.arrow(x_pos, y_pos, dx, dy, 
                    head_width=arrow_length_scale*0.35, 
                    head_length=arrow_length_scale*0.25,
                    fc='white', ec='black', 
                    alpha=0.9, linewidth=0.8,
                    length_includes_head=True)
    
    print(f"Drew {len(x_grid)} magnetic field vectors")

def smooth_ext12_data(data, smooth=True, sigma=1.0, kernel_size=None):
    """
    Extension 12データの平滑化
    
    Parameters
    ----------
    data : np.ndarray
        Extension 12のデータ（方位角データ）
    smooth : bool
        平滑化を適用するかどうか (default: True)
    sigma : float
        ガウシアンカーネルのシグマ値 (default: 1.0)
    kernel_size : int, optional
        カーネルサイズ。Noneの場合は自動計算 (default: None)
        
    Returns
    -------
    np.ndarray
        平滑化されたデータ（smooth=Falseの場合は元データをそのまま返す）
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
        # scipyのndimage.gaussian_filterはNaNを適切に処理しないため、
        # 有効なピクセルのみを処理する
        if np.all(valid_mask):
            # 全データが有効な場合
            smoothed_data = ndimage.gaussian_filter(data, sigma=sigma)
        else:
            # NaNがある場合の処理
            # 1. 有効データのみを抽出して平滑化
            # 2. NaN位置はそのまま保持
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

def plot_ucomp_extensions_group1(target_time, start_time, end_time, wavelength=None):
    """
    UCOMP ext1~3を1×3で描画 (Stokes I基本パラメータ)
    """
    # 最も近いデータを見つける
    closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
    
    if closest_data is None:
        print("No UCOMP data found for the specified time range")
        return None
    
    closest_info, file_path = closest_data
    extensions = read_ucomp_extensions(file_path, max_extensions=3)
    
    # 1×3のサブプロット
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    wl_text = f'{closest_info["wavelength"]}nm' if wavelength is None else f'{wavelength}nm'
    fig.suptitle(f'UCOMP {wl_text} - Stokes I Basic Parameters (Ext 1-3)\n'
                f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', fontsize=14, y=0.92)
    
    for i in range(3):
        ext_num = i + 1
        ax = axes[i]
        
        if ext_num in extensions:
            data, header = extensions[ext_num]
            vmin, vmax = get_data_range(data)
            
            # 太陽中心原点の座標系を設定
            center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
            extent = [-center_x, center_x, -center_y, center_y]
            
            im = ax.imshow(data, origin='lower', cmap='plasma', aspect='equal', 
                          vmin=vmin, vmax=vmax, extent=extent)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="1%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(get_colorbar_label(ext_num), fontsize=10)
            
            # 太陽半径描画
            draw_solar_radius(ax, data)
        else:
            # ダミーデータの場合も太陽中心原点座標系を使用
            dummy_data = np.zeros((100, 100))
            center_x, center_y = 50, 50
            extent = [-center_x, center_x, -center_y, center_y]
            ax.imshow(dummy_data, origin='lower', cmap='gray', aspect='equal', extent=extent)
            ax.text(0, 0, 'No Data', ha='center', va='center', 
                   fontsize=12, color='white', weight='bold')
        
        ax.set_title(f'Ext {ext_num}: {get_extension_title(ext_num)}', fontsize=12)
        ax.set_xlabel('X [pixels]', fontsize=10)
        ax.set_ylabel('Y [pixels]', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    
    # 保存
    output_path = get_ucomp_output_path()
    filename = create_output_filename(closest_info["date"], "group1")
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig

def plot_ucomp_extensions_group2(target_time, start_time, end_time, wavelength=None):
    """
    UCOMP ext4~6を1×3で描画 (Doppler & Line Width)
    """
    closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
    
    if closest_data is None:
        print("No UCOMP data found for the specified time range")
        return None
    
    closest_info, file_path = closest_data
    extensions = read_ucomp_extensions(file_path, max_extensions=6)
    
    # 1×3のサブプロット
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f'UCOMP {wavelength}nm - Doppler & Line Width (Ext 4-6)\n'
                f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', fontsize=14, y=0.92)
    
    for i in range(3):
        ext_num = i + 4  # ext4~6
        ax = axes[i]
        
        if ext_num in extensions:
            data, header = extensions[ext_num]
            vmin, vmax = get_data_range(data)
            
            # 太陽中心原点の座標系を設定
            center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
            extent = [-center_x, center_x, -center_y, center_y]
            
            # ext4はドップラー速度なので異なるカラーマップを使用
            cmap = 'RdBu_r' if ext_num == 4 else 'plasma'
            im = ax.imshow(data, origin='lower', cmap=cmap, aspect='equal', 
                          vmin=vmin, vmax=vmax, extent=extent)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="1%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(get_colorbar_label(ext_num), fontsize=10)
            
            draw_solar_radius(ax, data)
        else:
            # ダミーデータの場合も太陽中心原点座標系を使用
            dummy_data = np.zeros((100, 100))
            center_x, center_y = 50, 50
            extent = [-center_x, center_x, -center_y, center_y]
            ax.imshow(dummy_data, origin='lower', cmap='gray', aspect='equal', extent=extent)
            ax.text(0, 0, 'No Data', ha='center', va='center', 
                   fontsize=12, color='white', weight='bold')
        
        ax.set_title(f'Ext {ext_num}: {get_extension_title(ext_num)}', fontsize=12)
        ax.set_xlabel('X [pixels]', fontsize=10)
        ax.set_ylabel('Y [pixels]', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    
    # 保存
    output_path = get_ucomp_output_path()
    filename = create_output_filename(closest_info["date"], "group2")
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig

def plot_ucomp_extensions_group3(target_time, start_time, end_time, wavelength=None):
    """
    UCOMP ext7~10を2×2で描画 (Stokes Parameters)
    """
    closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
    
    if closest_data is None:
        print("No UCOMP data found for the specified time range")
        return None
    
    closest_info, file_path = closest_data
    extensions = read_ucomp_extensions(file_path, max_extensions=10)
    
    # 2×2のサブプロット
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'UCOMP {wavelength}nm - Stokes Parameters (Ext 7-10)\n'
                f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', fontsize=14, y=0.92)
    
    for i in range(4):
        ext_num = i + 7  # ext7~10
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        if ext_num in extensions:
            data, header = extensions[ext_num]
            vmin, vmax = get_data_range(data)
            
            # 太陽中心原点の座標系を設定
            center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
            extent = [-center_x, center_x, -center_y, center_y]
            
            # Stokes Q, Uは異なるカラーマップを使用
            cmap = 'RdBu_r' if ext_num in [8, 9] else 'plasma'
            im = ax.imshow(data, origin='lower', cmap=cmap, aspect='equal', 
                          vmin=vmin, vmax=vmax, extent=extent)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="1%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(get_colorbar_label(ext_num), fontsize=10)
            
            draw_solar_radius(ax, data)
        else:
            # ダミーデータの場合も太陽中心原点座標系を使用
            dummy_data = np.zeros((100, 100))
            center_x, center_y = 50, 50
            extent = [-center_x, center_x, -center_y, center_y]
            ax.imshow(dummy_data, origin='lower', cmap='gray', aspect='equal', extent=extent)
            ax.text(0, 0, 'No Data', ha='center', va='center', 
                   fontsize=12, color='white', weight='bold')
        
        ax.set_title(f'Ext {ext_num}: {get_extension_title(ext_num)}', fontsize=12)
        ax.set_xlabel('X [pixels]', fontsize=10)
        ax.set_ylabel('Y [pixels]', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    
    # 保存
    output_path = get_ucomp_output_path()
    filename = create_output_filename(closest_info["date"], "group3")
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig

def plot_ucomp_extensions_group4(target_time, start_time, end_time, wavelength=None):
    """
    UCOMP ext11~12を1×2で描画 (Magnetic Field Azimuth)
    """
    closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
    
    if closest_data is None:
        print("No UCOMP data found for the specified time range")
        return None
    
    closest_info, file_path = closest_data
    extensions = read_ucomp_extensions(file_path, max_extensions=12)
    
    # 1×2のサブプロット
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'UCOMP {wavelength}nm - Magnetic Field Azimuth (Ext 11-12)\n'
                f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', fontsize=14, y=0.92)
    
    for i in range(2):
        ext_num = i + 11  # ext11~12
        ax = axes[i]
        
        if ext_num in extensions:
            data, header = extensions[ext_num]
            vmin, vmax = get_data_range(data)
            
            # 太陽中心原点の座標系を設定
            center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
            extent = [-center_x, center_x, -center_y, center_y]
            
            # 方位角なのでhsvカラーマップを使用
            im = ax.imshow(data, origin='lower', cmap='hsv', aspect='equal', 
                          vmin=vmin, vmax=vmax, extent=extent)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="1%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(get_colorbar_label(ext_num), fontsize=10)
            
            draw_solar_radius(ax, data)
        else:
            # ダミーデータの場合も太陽中心原点座標系を使用
            dummy_data = np.zeros((100, 100))
            center_x, center_y = 50, 50
            extent = [-center_x, center_x, -center_y, center_y]
            ax.imshow(dummy_data, origin='lower', cmap='gray', aspect='equal', extent=extent)
            ax.text(0, 0, 'No Data', ha='center', va='center', 
                   fontsize=12, color='white', weight='bold')
        
        ax.set_title(f'Ext {ext_num}: {get_extension_title(ext_num)}', fontsize=12)
        ax.set_xlabel('X [pixels]', fontsize=10)
        ax.set_ylabel('Y [pixels]', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    
    # 保存
    output_path = get_ucomp_output_path()
    filename = create_output_filename(closest_info["date"], "group4")
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig

def plot_all_ucomp_groups(target_time, start_time, end_time, wavelength=None):
    """
    全4つのグループを順番に描画・保存
    """
    print("=" * 60)
    print("UCOMP All Groups Plotting")
    print("=" * 60)
    
    print("\n--- Group 1: Stokes I Basic Parameters (Ext 1-3) ---")
    fig1 = plot_ucomp_extensions_group1(target_time, start_time, end_time, wavelength)
    
    print("\n--- Group 2: Doppler & Line Width (Ext 4-6) ---")
    fig2 = plot_ucomp_extensions_group2(target_time, start_time, end_time, wavelength)
    
    print("\n--- Group 3: Stokes Parameters (Ext 7-10) ---")
    fig3 = plot_ucomp_extensions_group3(target_time, start_time, end_time, wavelength)
    
    print("\n--- Group 4: Magnetic Field Azimuth (Ext 11-12) ---")
    fig4 = plot_ucomp_extensions_group4(target_time, start_time, end_time, wavelength)
    
    print("\n--- All groups completed ---")
    
    return [fig1, fig2, fig3, fig4]

def get_available_writer():
    """
    利用可能な動画ライターを取得
    
    Returns
    -------
    str or None
        利用可能なライター名、または None
    """
    # 利用可能なライターのリスト（優先順）
    preferred_writers = ['ffmpeg', 'pillow', 'imagemagick']
    
    for writer_name in preferred_writers:
        try:
            if writer_name in animation.writers.list():
                # テスト用に実際にライターを作成してみる
                Writer = animation.writers[writer_name]
                test_writer = Writer(fps=1, metadata=dict(artist='Test'), bitrate=1800)
                print(f"Available writer found: {writer_name}")
                return writer_name
        except Exception as e:
            print(f"Writer {writer_name} not available: {e}")
            continue
    
    print("No matplotlib animation writers available")
    return None

def create_video_from_images(image_folder, output_path, fps=2):
    """
    画像フォルダから動画を作成（外部ffmpegコマンドを使用）
    
    Parameters
    ----------
    image_folder : str
        画像が保存されているフォルダのパス
    output_path : str
        出力動画ファイルのパス
    fps : int
        フレームレート
        
    Returns
    -------
    bool
        成功した場合True、失敗した場合False
    """
    try:
        # ffmpegコマンドで動画作成（画像サイズを2で割り切れるように調整）
        cmd = [
            'ffmpeg', '-y',  # -y: overwrite output file
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', f'{image_folder}/*.png',
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # 幅と高さを2で割り切れるように調整
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(output_path)
        ]
        
        print(f"Creating video with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"Video created successfully: {output_path}")
            return True
        else:
            print(f"ffmpeg failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("ffmpeg command not found. Please install ffmpeg.")
        return False
    except subprocess.TimeoutExpired:
        print("ffmpeg command timed out.")
        return False
    except Exception as e:
        print(f"Error creating video: {e}")
        return False

def save_frame_as_image(fig, image_path):
    """
    matplotlibの図を画像として保存
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        保存する図
    image_path : str
        保存先のパス
    """
    fig.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved frame: {image_path}")

def create_ucomp_animation_group1(start_time, end_time, wavelength=None, 
                                 fps=2, interval=500, output_filename=None):
    """
    Group1 (Ext 1-3) の時系列動画を作成
    
    Parameters
    ----------
    start_time : str or Time
        開始時刻
    end_time : str or Time
        終了時刻
    wavelength : int
        波長 (default: 1074)
    fps : int
        フレームレート (default: 2)
    interval : int
        フレーム間隔（ミリ秒） (default: 500)
    output_filename : str, optional
        出力ファイル名（指定しない場合は自動生成）
        
    Returns
    -------
    str
        保存されたファイルのパス
    """
    print("=" * 60)
    print("Creating UCOMP Group 1 Animation (Ext 1-3)")
    print("=" * 60)
    
    # 利用可能な時刻を取得（波長に関係なく時間範囲内のデータ）
    times = get_available_ucomp_times(start_time, end_time, wavelength)
    
    if len(times) == 0:
        print("No UCOMP data found for the specified time range")
        return None
    
    print(f"Found {len(times)} time points")
    
    # start_timeのデータからカラースケール範囲を取得
    start_data = find_closest_ucomp_data(start_time, start_time, end_time, wavelength)
    color_ranges = {}  # {ext_num: (vmin, vmax)}
    
    if start_data:
        start_info, start_file_path = start_data
        start_extensions = read_ucomp_extensions(start_file_path, max_extensions=3)
        for ext_num in range(1, 4):
            if ext_num in start_extensions:
                data, _ = start_extensions[ext_num]
                vmin, vmax = get_data_range(data)
                color_ranges[ext_num] = (vmin, vmax)
                print(f"Fixed color range for Ext {ext_num}: [{vmin:.3f}, {vmax:.3f}]")
    
    # 出力ファイル名の設定
    if output_filename is None:
        from astropy.time import Time
        start_str = Time(start_time).strftime("%Y%m%d_%H%M")
        end_str = Time(end_time).strftime("%Y%m%d_%H%M")
        if wavelength is not None:
            output_filename = f"ucomp_group1_animation_{wavelength}nm_{start_str}-{end_str}.mp4"
        else:
            output_filename = f"ucomp_group1_animation_allwl_{start_str}-{end_str}.mp4"
    
    output_path = get_ucomp_output_path()
    save_path = output_path / output_filename
    
    # 図の初期化
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle('', fontsize=14, y=0.92)
    
    # 初期プロット用のダミーデータ
    ims = []
    cbars = []
    
    for i in range(3):
        ax = axes[i]
        ext_num = i + 1
        
        # ダミー画像でプロットを初期化（太陽中心原点座標系）
        dummy_data = np.zeros((100, 100))
        center_x, center_y = 50, 50
        extent = [-center_x, center_x, -center_y, center_y]
        im = ax.imshow(dummy_data, origin='lower', cmap='plasma', aspect='equal', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(get_colorbar_label(ext_num), fontsize=10)
        
        ax.set_title(f'Ext {ext_num}: {get_extension_title(ext_num)}', fontsize=12)
        ax.set_xlabel('X [pixels]', fontsize=10)
        ax.set_ylabel('Y [pixels]', fontsize=10)
        
        ims.append(im)
        cbars.append(cbar)
    
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    
    def animate(frame_num):
        """アニメーション更新関数"""
        target_time = times[frame_num]
        print(f"Processing frame {frame_num+1}/{len(times)}: {target_time.iso}")
        
        # データを取得
        closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
        
        if closest_data is None:
            return ims
        
        closest_info, file_path = closest_data
        extensions = read_ucomp_extensions(file_path, max_extensions=3)
        
        # タイトル更新
        wl_text = f'{closest_info["wavelength"]}nm' if wavelength is None else f'{wavelength}nm'
        fig.suptitle(f'UCOMP {wl_text} - Stokes I Basic Parameters (Ext 1-3)\n'
                    f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', 
                    fontsize=14, y=0.92)
        
        # 各extensionの画像を更新
        for i in range(3):
            ext_num = i + 1
            ax = axes[i]
            
            # 既存の円を削除
            for patch in ax.patches[:]:
                patch.remove()
            
            if ext_num in extensions:
                data, header = extensions[ext_num]
                # 固定されたカラースケール範囲を使用
                if ext_num in color_ranges:
                    vmin, vmax = color_ranges[ext_num]
                else:
                    vmin, vmax = get_data_range(data)
                
                ims[i].set_array(data)
                ims[i].set_clim(vmin, vmax)
                
                # 太陽半径描画
                draw_solar_radius(ax, data)
            else:
                ims[i].set_array(np.zeros((100, 100)))
        
        return ims
    
    # 利用可能なライターを確認
    available_writer = get_available_writer()
    
    if available_writer:
        # matplotlib animationを使用
        print("Creating animation with matplotlib...")
        anim = animation.FuncAnimation(fig, animate, frames=len(times), 
                                     interval=interval, blit=False, repeat=True)
        
        print(f"Saving animation to: {save_path}")
        Writer = animation.writers[available_writer]
        writer = Writer(fps=fps, metadata=dict(artist='UCOMP Analysis'), bitrate=1800)
        
        anim.save(save_path, writer=writer)
        plt.close(fig)
        
        print(f"Animation saved: {save_path}")
        return str(save_path)
    
    else:
        # 代替方法：個別画像を作成してから動画にする
        print("Creating animation using individual frames...")
        
        # 一時ディレクトリを作成
        temp_dir = tempfile.mkdtemp(prefix='ucomp_group1_')
        print(f"Creating temporary images in: {temp_dir}")
        
        try:
            # 各フレームを個別に保存
            for frame_num in range(len(times)):
                target_time = times[frame_num]
                print(f"Creating frame {frame_num+1}/{len(times)}: {target_time.iso}")
                
                # データを取得
                closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
                if closest_data is None:
                    continue
                
                closest_info, file_path = closest_data
                extensions = read_ucomp_extensions(file_path, max_extensions=3)
                
                # 新しい図を作成
                frame_fig, frame_axes = plt.subplots(1, 3, figsize=(13, 4))
                wl_text = f'{closest_info["wavelength"]}nm' if wavelength is None else f'{wavelength}nm'
                frame_fig.suptitle(f'UCOMP {wl_text} - Stokes I Basic Parameters (Ext 1-3)\\n'
                                  f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', 
                                  fontsize=14, y=0.92)
                
                for i in range(3):
                    ext_num = i + 1
                    ax = frame_axes[i]
                    
                    if ext_num in extensions:
                        data, header = extensions[ext_num]
                        # 固定されたカラースケール範囲を使用
                        if ext_num in color_ranges:
                            vmin, vmax = color_ranges[ext_num]
                        else:
                            vmin, vmax = get_data_range(data)
                        
                        center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
                        extent = [-center_x, center_x, -center_y, center_y]
                        
                        im = ax.imshow(data, origin='lower', cmap='plasma', aspect='equal', 
                                      vmin=vmin, vmax=vmax, extent=extent)
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="1%", pad=0.1)
                        cbar = plt.colorbar(im, cax=cax)
                        cbar.set_label(get_colorbar_label(ext_num), fontsize=10)
                        
                        draw_solar_radius(ax, data)
                    else:
                        dummy_data = np.zeros((100, 100))
                        center_x, center_y = 50, 50
                        extent = [-center_x, center_x, -center_y, center_y]
                        ax.imshow(dummy_data, origin='lower', cmap='gray', aspect='equal', extent=extent)
                        ax.text(0, 0, 'No Data', ha='center', va='center', 
                               fontsize=12, color='white', weight='bold')
                    
                    ax.set_title(f'Ext {ext_num}: {get_extension_title(ext_num)}', fontsize=12)
                    ax.set_xlabel('X [pixels]', fontsize=10)
                    ax.set_ylabel('Y [pixels]', fontsize=10)
                
                plt.tight_layout(rect=[0, 0, 1, 0.88])
                
                # フレームを保存
                frame_path = temp_dir + f"/frame_{frame_num:04d}.png"
                save_frame_as_image(frame_fig, frame_path)
                plt.close(frame_fig)
            
            # 画像から動画を作成
            if create_video_from_images(temp_dir, save_path, fps):
                print(f"Animation saved: {save_path}")
                return str(save_path)
            else:
                print("Failed to create video from images")
                return None
                
        finally:
            # 一時ディレクトリを削除
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
            plt.close(fig)

def create_ucomp_animation_group2(start_time, end_time, wavelength=None, 
                                 fps=2, interval=500, output_filename=None):
    """
    Group2 (Ext 4-6) の時系列動画を作成
    """
    print("=" * 60)
    print("Creating UCOMP Group 2 Animation (Ext 4-6)")
    print("=" * 60)
    
    times = get_available_ucomp_times(start_time, end_time, wavelength)
    
    if len(times) == 0:
        print("No UCOMP data found for the specified time range")
        return None
    
    print(f"Found {len(times)} time points")
    
    if output_filename is None:
        from astropy.time import Time
        start_str = Time(start_time).strftime("%Y%m%d_%H%M")
        end_str = Time(end_time).strftime("%Y%m%d_%H%M")
        output_filename = f"ucomp_group2_animation_{start_str}-{end_str}.mp4"
    
    output_path = get_ucomp_output_path()
    save_path = output_path / output_filename
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle('', fontsize=14, y=0.92)
    
    ims = []
    cbars = []
    
    for i in range(3):
        ext_num = i + 4  # ext4~6
        ax = axes[i]
        
        cmap = 'RdBu_r' if ext_num == 4 else 'plasma'
        dummy_data = np.zeros((100, 100))
        center_x, center_y = 50, 50
        extent = [-center_x, center_x, -center_y, center_y]
        im = ax.imshow(dummy_data, origin='lower', cmap=cmap, aspect='equal', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(get_colorbar_label(ext_num), fontsize=10)
        
        ax.set_title(f'Ext {ext_num}: {get_extension_title(ext_num)}', fontsize=12)
        ax.set_xlabel('X [pixels]', fontsize=10)
        ax.set_ylabel('Y [pixels]', fontsize=10)
        
        ims.append(im)
        cbars.append(cbar)
    
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    
    def animate(frame_num):
        target_time = times[frame_num]
        print(f"Processing frame {frame_num+1}/{len(times)}: {target_time.iso}")
        
        closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
        
        if closest_data is None:
            return ims
        
        closest_info, file_path = closest_data
        extensions = read_ucomp_extensions(file_path, max_extensions=6)
        
        fig.suptitle(f'UCOMP {wavelength}nm - Doppler & Line Width (Ext 4-6)\n'
                    f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', 
                    fontsize=14, y=0.92)
        
        for i in range(3):
            ext_num = i + 4
            ax = axes[i]
            
            for patch in ax.patches[:]:
                patch.remove()
            
            if ext_num in extensions:
                data, header = extensions[ext_num]
                vmin, vmax = get_data_range(data)
                
                ims[i].set_array(data)
                ims[i].set_clim(vmin, vmax)
                
                draw_solar_radius(ax, data)
            else:
                ims[i].set_array(np.zeros((100, 100)))
        
        return ims
    
    # 利用可能なライターを確認
    available_writer = get_available_writer()
    
    if available_writer:
        print("Creating animation with matplotlib...")
        anim = animation.FuncAnimation(fig, animate, frames=len(times), 
                                     interval=interval, blit=False, repeat=True)
        
        print(f"Saving animation to: {save_path}")
        Writer = animation.writers[available_writer]
        writer = Writer(fps=fps, metadata=dict(artist='UCOMP Analysis'), bitrate=1800)
        
        anim.save(save_path, writer=writer)
        plt.close(fig)
        
        print(f"Animation saved: {save_path}")
        return str(save_path)
    
    else:
        # 代替方法：個別画像を作成してから動画にする
        print("Creating animation using individual frames...")
        temp_dir = tempfile.mkdtemp(prefix='ucomp_group2_')
        print(f"Creating temporary images in: {temp_dir}")
        
        try:
            for frame_num in range(len(times)):
                target_time = times[frame_num]
                print(f"Creating frame {frame_num+1}/{len(times)}: {target_time.iso}")
                
                closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
                if closest_data is None:
                    continue
                
                closest_info, file_path = closest_data
                extensions = read_ucomp_extensions(file_path, max_extensions=6)
                
                frame_fig, frame_axes = plt.subplots(1, 3, figsize=(13, 4))
                frame_fig.suptitle(f'UCOMP {wavelength}nm - Doppler & Line Width (Ext 4-6)\\n'
                                  f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', 
                                  fontsize=14, y=0.92)
                
                for i in range(3):
                    ext_num = i + 4
                    ax = frame_axes[i]
                    
                    if ext_num in extensions:
                        data, header = extensions[ext_num]
                        vmin, vmax = get_data_range(data)
                        
                        center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
                        extent = [-center_x, center_x, -center_y, center_y]
                        
                        cmap = 'RdBu_r' if ext_num == 4 else 'plasma'
                        im = ax.imshow(data, origin='lower', cmap=cmap, aspect='equal', 
                                      vmin=vmin, vmax=vmax, extent=extent)
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="1%", pad=0.1)
                        cbar = plt.colorbar(im, cax=cax)
                        cbar.set_label(get_colorbar_label(ext_num), fontsize=10)
                        
                        draw_solar_radius(ax, data)
                    else:
                        dummy_data = np.zeros((100, 100))
                        center_x, center_y = 50, 50
                        extent = [-center_x, center_x, -center_y, center_y]
                        ax.imshow(dummy_data, origin='lower', cmap='gray', aspect='equal', extent=extent)
                        ax.text(0, 0, 'No Data', ha='center', va='center', 
                               fontsize=12, color='white', weight='bold')
                    
                    ax.set_title(f'Ext {ext_num}: {get_extension_title(ext_num)}', fontsize=12)
                    ax.set_xlabel('X [pixels]', fontsize=10)
                    ax.set_ylabel('Y [pixels]', fontsize=10)
                
                plt.tight_layout(rect=[0, 0, 1, 0.88])
                
                frame_path = temp_dir + f"/frame_{frame_num:04d}.png"
                save_frame_as_image(frame_fig, frame_path)
                plt.close(frame_fig)
            
            if create_video_from_images(temp_dir, save_path, fps):
                print(f"Animation saved: {save_path}")
                return str(save_path)
            else:
                print("Failed to create video from images")
                return None
                
        finally:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
            plt.close(fig)

def create_ucomp_animation_group3(start_time, end_time, wavelength=None, 
                                 fps=2, interval=500, output_filename=None):
    """
    Group3 (Ext 7-10) の時系列動画を作成
    """
    print("=" * 60)
    print("Creating UCOMP Group 3 Animation (Ext 7-10)")
    print("=" * 60)
    
    times = get_available_ucomp_times(start_time, end_time, wavelength)
    
    if len(times) == 0:
        print("No UCOMP data found for the specified time range")
        return None
    
    print(f"Found {len(times)} time points")
    
    if output_filename is None:
        from astropy.time import Time
        start_str = Time(start_time).strftime("%Y%m%d_%H%M")
        end_str = Time(end_time).strftime("%Y%m%d_%H%M")
        output_filename = f"ucomp_group3_animation_{start_str}-{end_str}.mp4"
    
    output_path = get_ucomp_output_path()
    save_path = output_path / output_filename
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('', fontsize=14, y=0.92)
    
    ims = []
    cbars = []
    
    for i in range(4):
        ext_num = i + 7  # ext7~10
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        cmap = 'RdBu_r' if ext_num in [8, 9] else 'plasma'
        dummy_data = np.zeros((100, 100))
        center_x, center_y = 50, 50
        extent = [-center_x, center_x, -center_y, center_y]
        im = ax.imshow(dummy_data, origin='lower', cmap=cmap, aspect='equal', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(get_colorbar_label(ext_num), fontsize=10)
        
        ax.set_title(f'Ext {ext_num}: {get_extension_title(ext_num)}', fontsize=12)
        ax.set_xlabel('X [pixels]', fontsize=10)
        ax.set_ylabel('Y [pixels]', fontsize=10)
        
        ims.append(im)
        cbars.append(cbar)
    
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    
    def animate(frame_num):
        target_time = times[frame_num]
        print(f"Processing frame {frame_num+1}/{len(times)}: {target_time.iso}")
        
        closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
        
        if closest_data is None:
            return ims
        
        closest_info, file_path = closest_data
        extensions = read_ucomp_extensions(file_path, max_extensions=10)
        
        fig.suptitle(f'UCOMP {wavelength}nm - Stokes Parameters (Ext 7-10)\n'
                    f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', 
                    fontsize=14, y=0.92)
        
        for i in range(4):
            ext_num = i + 7
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            for patch in ax.patches[:]:
                patch.remove()
            
            if ext_num in extensions:
                data, header = extensions[ext_num]
                vmin, vmax = get_data_range(data)
                
                ims[i].set_array(data)
                ims[i].set_clim(vmin, vmax)
                
                draw_solar_radius(ax, data)
            else:
                ims[i].set_array(np.zeros((100, 100)))
        
        return ims
    
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=len(times), 
                                 interval=interval, blit=False, repeat=True)
    
    print(f"Saving animation to: {save_path}")
    available_writer = get_available_writer()
    if available_writer:
        Writer = animation.writers[available_writer]
        writer = Writer(fps=fps, metadata=dict(artist='UCOMP Analysis'), bitrate=1800)
    else:
        print("No animation writers available, falling back to frame-by-frame method")
        return None
    
    anim.save(save_path, writer=writer)
    plt.close(fig)
    
    print(f"Animation saved: {save_path}")
    return str(save_path)

def create_ucomp_animation_group4(start_time, end_time, wavelength=None, 
                                 fps=2, interval=500, output_filename=None):
    """
    Group4 (Ext 11-12) の時系列動画を作成
    """
    print("=" * 60)
    print("Creating UCOMP Group 4 Animation (Ext 11-12)")
    print("=" * 60)
    
    times = get_available_ucomp_times(start_time, end_time, wavelength)
    
    if len(times) == 0:
        print("No UCOMP data found for the specified time range")
        return None
    
    print(f"Found {len(times)} time points")
    
    if output_filename is None:
        from astropy.time import Time
        start_str = Time(start_time).strftime("%Y%m%d_%H%M")
        end_str = Time(end_time).strftime("%Y%m%d_%H%M")
        output_filename = f"ucomp_group4_animation_{start_str}-{end_str}.mp4"
    
    output_path = get_ucomp_output_path()
    save_path = output_path / output_filename
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('', fontsize=14, y=0.92)
    
    ims = []
    cbars = []
    
    for i in range(2):
        ext_num = i + 11  # ext11~12
        ax = axes[i]
        
        dummy_data = np.zeros((100, 100))
        center_x, center_y = 50, 50
        extent = [-center_x, center_x, -center_y, center_y]
        im = ax.imshow(dummy_data, origin='lower', cmap='hsv', aspect='equal', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(get_colorbar_label(ext_num), fontsize=10)
        
        ax.set_title(f'Ext {ext_num}: {get_extension_title(ext_num)}', fontsize=12)
        ax.set_xlabel('X [pixels]', fontsize=10)
        ax.set_ylabel('Y [pixels]', fontsize=10)
        
        ims.append(im)
        cbars.append(cbar)
    
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    
    def animate(frame_num):
        target_time = times[frame_num]
        print(f"Processing frame {frame_num+1}/{len(times)}: {target_time.iso}")
        
        closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
        
        if closest_data is None:
            return ims
        
        closest_info, file_path = closest_data
        extensions = read_ucomp_extensions(file_path, max_extensions=12)
        
        fig.suptitle(f'UCOMP {wavelength}nm - Magnetic Field Azimuth (Ext 11-12)\n'
                    f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', 
                    fontsize=14, y=0.92)
        
        for i in range(2):
            ext_num = i + 11
            ax = axes[i]
            
            for patch in ax.patches[:]:
                patch.remove()
            
            if ext_num in extensions:
                data, header = extensions[ext_num]
                vmin, vmax = get_data_range(data)
                
                ims[i].set_array(data)
                ims[i].set_clim(vmin, vmax)
                
                draw_solar_radius(ax, data)
            else:
                ims[i].set_array(np.zeros((100, 100)))
        
        return ims
    
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=len(times), 
                                 interval=interval, blit=False, repeat=True)
    
    print(f"Saving animation to: {save_path}")
    available_writer = get_available_writer()
    if available_writer:
        Writer = animation.writers[available_writer]
        writer = Writer(fps=fps, metadata=dict(artist='UCOMP Analysis'), bitrate=1800)
    else:
        print("No animation writers available, falling back to frame-by-frame method")
        return None
    
    anim.save(save_path, writer=writer)
    plt.close(fig)
    
    print(f"Animation saved: {save_path}")
    return str(save_path)

def create_all_ucomp_animations(start_time, end_time, wavelength=None, 
                               fps=2, interval=500):
    """
    全4つのグループの時系列動画を順番に作成
    
    Parameters
    ----------
    start_time : str or Time
        開始時刻
    end_time : str or Time
        終了時刻
    wavelength : int
        波長 (default: 1074)
    fps : int
        フレームレート (default: 2)
    interval : int
        フレーム間隔（ミリ秒） (default: 500)
        
    Returns
    -------
    list
        作成された動画ファイルのパスのリスト
    """
    print("=" * 80)
    print("UCOMP All Groups Animation Creation")
    print("=" * 80)
    
    animation_paths = []
    
    print("\n--- Group 1: Stokes I Basic Parameters (Ext 1-3) Animation ---")
    path1 = create_ucomp_animation_group1(start_time, end_time, wavelength, fps, interval)
    if path1:
        animation_paths.append(path1)
    
    print("\n--- Group 2: Doppler & Line Width (Ext 4-6) Animation ---")
    path2 = create_ucomp_animation_group2(start_time, end_time, wavelength, fps, interval)
    if path2:
        animation_paths.append(path2)
    
    print("\n--- Group 3: Stokes Parameters (Ext 7-10) Animation ---")
    path3 = create_ucomp_animation_group3(start_time, end_time, wavelength, fps, interval)
    if path3:
        animation_paths.append(path3)
    
    print("\n--- Group 4: Magnetic Field Azimuth (Ext 11-12) Animation ---")
    path4 = create_ucomp_animation_group4(start_time, end_time, wavelength, fps, interval)
    if path4:
        animation_paths.append(path4)
    
    print("\n--- All animations completed ---")
    print(f"Created {len(animation_paths)} animation files:")
    for path in animation_paths:
        print(f"  - {path}")
    
    return animation_paths

def plot_ucomp_extensions_custom1(target_time, start_time, end_time, wavelength=None, 
                                  smooth_ext12=False, ext12_sigma=1.0):
    """
    UCOMP ext3,4,5,12を2×2で描画 (カスタムグループ1)
    
    Parameters
    ----------
    target_time : str or Time
        プロット対象時刻
    start_time : str or Time
        スキャン開始時刻
    end_time : str or Time
        スキャン終了時刻
    wavelength : int or None
        波長 (Noneの場合は全波長データから自動選択)
    smooth_ext12 : bool, optional
        Extension 12の平滑化を適用するか (default: False)
    ext12_sigma : float, optional
        Extension 12の平滑化シグマ値 (default: 1.0)
        
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
    extensions = read_ucomp_extensions(file_path, max_extensions=12)
    
    if not extensions:
        print("Failed to read UCOMP extensions")
        return None
    
    # 2×2のサブプロットを作成
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'UCOMP {wavelength}nm - Custom Group 1 (Ext 3,4,5,12)\n'
                f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', 
                fontsize=14, y=0.95)
    
    # Extensionのマッピング（3,4,5,12を2×2に配置）
    ext_mapping = [3, 4, 5, 12]
    
    for i in range(4):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        ext_num = ext_mapping[i]
        
        if ext_num in extensions:
            data, header = extensions[ext_num]
            
            # Extension 12の平滑化処理
            if ext_num == 12:
                processed_data = smooth_ext12_data(data, smooth=smooth_ext12, sigma=ext12_sigma)
            else:
                processed_data = data
            
            # Extension番号に応じたカラーバー範囲とカラーマップを設定
            if ext_num == 3:
                vmin, vmax = 0, 20
                cmap = 'plasma'
            elif ext_num == 4:
                vmin, vmax = -10, 40
                cmap = 'RdBu_r'
            elif ext_num == 5:
                vmin, vmax = 0, 140
                cmap = 'plasma'
            elif ext_num == 12:
                vmin, vmax = get_data_range(processed_data)
                cmap = 'hsv'
            else:
                vmin, vmax = get_data_range(processed_data)
                cmap = 'plasma'
            
            # 太陽中心を原点とする座標系
            center_x, center_y = processed_data.shape[1] // 2, processed_data.shape[0] // 2
            extent = [-center_x, center_x, -center_y, center_y]
            
            # プロット
            im = ax.imshow(processed_data, origin='lower', cmap=cmap, aspect='equal', 
                          vmin=vmin, vmax=vmax, extent=extent)
            
            # カラーバーを追加（統一形式）
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="1%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(get_colorbar_label(ext_num), fontsize=10)
            
            # 太陽半径を描画
            draw_solar_radius(ax, processed_data)
            
            # Extension 12には磁場ベクトル場を追加
            if ext_num == 12:
                # ベクトル場グリッドを作成
                x_grid, y_grid = create_vector_field_grid(processed_data.shape, 
                                                         radial_interval_rs=0.1, 
                                                         angular_interval_deg=5)
                # 磁場ベクトルを描画
                draw_magnetic_field_vectors(ax, processed_data, processed_data, x_grid, y_grid,
                                          arrow_length_scale=15, 
                                          arrow_color='white',
                                          arrow_alpha=0.7)
            
        else:
            # データがない場合はダミー画像
            dummy_data = np.zeros((100, 100))
            center_x, center_y = 50, 50
            extent = [-center_x, center_x, -center_y, center_y]
            ax.imshow(dummy_data, origin='lower', cmap='gray', aspect='equal', extent=extent)
            ax.text(0, 0, 'No Data', ha='center', va='center', 
                   fontsize=12, color='white', weight='bold')
        
        # タイトルとラベル設定
        ax.set_title(f'Ext {ext_num}: {get_extension_title(ext_num)}', fontsize=12)
        ax.set_xlabel('X [pixels]', fontsize=10)
        ax.set_ylabel('Y [pixels]', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # メモリ解放
    del extensions
    gc.collect()
    
    return fig

def plot_ucomp_extensions_custom2(target_time, start_time, end_time, wavelength=None):
    """
    UCOMP ext7~10を2×2で描画 (カスタムグループ2)
    """
    # 最も近いデータを見つける
    closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
    
    if closest_data is None:
        print("No UCOMP data found for the specified time range")
        return None
    
    closest_info, file_path = closest_data
    
    # Extensionを読み込み
    extensions = read_ucomp_extensions(file_path, max_extensions=10)
    
    if not extensions:
        print("Failed to read UCOMP extensions")
        return None
    
    # 2×2のサブプロットを作成
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'UCOMP {wavelength}nm - Custom Group 2 (Ext 7-10)\n'
                f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', 
                fontsize=14, y=0.95)
    
    # Extensionのマッピング（7,8,9,10を2×2に配置）
    ext_mapping = [7, 8, 9, 10]
    
    for i in range(4):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        ext_num = ext_mapping[i]
        
        if ext_num in extensions:
            data, header = extensions[ext_num]
            
            # Extension番号に応じたカラーバー範囲とカラーマップを設定
            if ext_num == 7:
                vmin, vmax = 0, 20
                cmap = 'plasma'
            elif ext_num == 8:
                vmin, vmax = -0.3, 0.8
                cmap = 'RdBu_r'
            elif ext_num == 9:
                vmin, vmax = -0.5, 0.6
                cmap = 'RdBu_r'
            elif ext_num == 10:
                vmin, vmax = 0, 1.0
                cmap = 'plasma'
            else:
                vmin, vmax = get_data_range(data)
                cmap = 'plasma'
            
            # 太陽中心を原点とする座標系
            center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
            extent = [-center_x, center_x, -center_y, center_y]
            
            # プロット
            im = ax.imshow(data, origin='lower', cmap=cmap, aspect='equal', 
                          vmin=vmin, vmax=vmax, extent=extent)
            
            # カラーバーを追加（統一形式）
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="1%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(get_colorbar_label(ext_num), fontsize=10)
            
            # 太陽半径を描画
            draw_solar_radius(ax, data)
            
        else:
            # データがない場合はダミー画像
            dummy_data = np.zeros((100, 100))
            center_x, center_y = 50, 50
            extent = [-center_x, center_x, -center_y, center_y]
            ax.imshow(dummy_data, origin='lower', cmap='gray', aspect='equal', extent=extent)
            ax.text(0, 0, 'No Data', ha='center', va='center', 
                   fontsize=12, color='white', weight='bold')
        
        # タイトルとラベル設定
        ax.set_title(f'Ext {ext_num}: {get_extension_title(ext_num)}', fontsize=12)
        ax.set_xlabel('X [pixels]', fontsize=10)
        ax.set_ylabel('Y [pixels]', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # メモリ解放
    del extensions
    gc.collect()
    
    return fig

def create_ucomp_animation_custom1(start_time, end_time, wavelength=None, 
                                  fps=2, interval=500, output_filename=None,
                                  smooth_ext12=False, ext12_sigma=1.0):
    """
    Custom Group1 (Ext 3,4,5,12) の時系列動画を作成
    
    Parameters
    ----------
    start_time : str or Time
        開始時刻
    end_time : str or Time
        終了時刻
    wavelength : int or None
        波長 (Noneの場合は全波長データから自動選択)
    fps : int, optional
        フレームレート (default: 2)
    interval : int, optional
        フレーム間隔（ミリ秒） (default: 500)
    output_filename : str, optional
        出力ファイル名 (default: None)
    smooth_ext12 : bool, optional
        Extension 12の平滑化を適用するか (default: False)
    ext12_sigma : float, optional
        Extension 12の平滑化シグマ値 (default: 1.0)
        
    Returns
    -------
    str
        保存されたファイルのパス
    """
    print("=" * 60)
    print("Creating UCOMP Custom Group 1 Animation (Ext 3,4,5,12)")
    print("=" * 60)
    
    times = get_available_ucomp_times(start_time, end_time, wavelength)
    
    if len(times) == 0:
        print("No UCOMP data found for the specified time range")
        return None
    
    print(f"Found {len(times)} time points")
    
    if output_filename is None:
        from astropy.time import Time
        start_str = Time(start_time).strftime("%Y%m%d_%H%M")
        end_str = Time(end_time).strftime("%Y%m%d_%H%M")
        output_filename = f"ucomp_custom1_animation_{start_str}-{end_str}.mp4"
    
    output_path = get_ucomp_output_path()
    save_path = output_path / output_filename
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('', fontsize=14, y=0.95)
    
    ims = []
    
    # Writer確認とフォールバック
    writer = get_available_writer()
    print(f"Available animation writer: {writer}")
    
    # 現在の実装では、直接フォールバック方式を使用（より安定）
    print("Using fallback method (individual frames -> video).")
    
    # UCOMPフォルダー内にframesディレクトリを作成
    output_path = get_ucomp_output_path()
    frames_dir = output_path / "frames" / f"ucomp_custom1_{Time(start_time).strftime('%Y%m%d_%H%M')}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating temporary images in: {frames_dir}")
    
    try:
        ext_mapping = [3, 4, 5, 12]
        
        for frame_num in range(len(times)):
            target_time = times[frame_num]
            print(f"Creating frame {frame_num+1}/{len(times)}: {target_time.iso}")
            
            closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
            if closest_data is None:
                print(f"  No data found for frame {frame_num+1}")
                continue
            
            closest_info, file_path = closest_data
            extensions = read_ucomp_extensions(file_path, max_extensions=12)
            
            # 2×2形式で画像を作成
            time_str = closest_info["date"].strftime("%y%m%d-%H%M%S")
            wl_value = closest_info["wavelength"]
            
            frame_fig, frame_axes = plt.subplots(2, 2, figsize=(10, 8))
            frame_fig.suptitle(f'UCOMP {wl_value}nm - Custom Group 1 (Ext 3,4,5,12)\n'
                              f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', 
                              fontsize=14, y=0.95)
            
            for i in range(4):
                row = i // 2
                col = i % 2
                ax = frame_axes[row, col]
                ext_num = ext_mapping[i]
                
                if ext_num in extensions:
                    data, header = extensions[ext_num]
                    
                    # Extension 12の平滑化処理
                    if ext_num == 12:
                        processed_data = smooth_ext12_data(data, smooth=smooth_ext12, sigma=ext12_sigma)
                    else:
                        processed_data = data
                    
                    # Extension番号に応じたカラーバー範囲とカラーマップを設定
                    if ext_num == 3:
                        vmin, vmax = 0, 20
                        cmap = 'plasma'
                    elif ext_num == 4:
                        vmin, vmax = -10, 40
                        cmap = 'RdBu_r'
                    elif ext_num == 5:
                        vmin, vmax = 0, 140
                        cmap = 'plasma'
                    elif ext_num == 12:
                        vmin, vmax = get_data_range(processed_data)
                        cmap = 'hsv'
                    else:
                        vmin, vmax = get_data_range(processed_data)
                        cmap = 'plasma'
                    
                    center_x, center_y = processed_data.shape[1] // 2, processed_data.shape[0] // 2
                    extent = [-center_x, center_x, -center_y, center_y]
                    
                    im = ax.imshow(processed_data, origin='lower', cmap=cmap, aspect='equal', 
                                  vmin=vmin, vmax=vmax, extent=extent)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="1%", pad=0.1)
                    cbar = plt.colorbar(im, cax=cax)
                    cbar.set_label(get_colorbar_label(ext_num), fontsize=10)
                    
                    draw_solar_radius(ax, processed_data)
                    
                    # Extension 12には磁場ベクトル場を追加
                    if ext_num == 12:
                        # ベクトル場グリッドを作成
                        x_grid, y_grid = create_vector_field_grid(processed_data.shape, 
                                                                 radial_interval_rs=0.1, 
                                                                 angular_interval_deg=5)
                        # 磁場ベクトルを描画
                        draw_magnetic_field_vectors(ax, processed_data, processed_data, x_grid, y_grid,
                                                  arrow_length_scale=15, 
                                                  arrow_color='white',
                                                  arrow_alpha=0.7)
                else:
                    dummy_data = np.zeros((100, 100))
                    center_x, center_y = 50, 50
                    extent = [-center_x, center_x, -center_y, center_y]
                    ax.imshow(dummy_data, origin='lower', cmap='gray', aspect='equal', extent=extent)
                    ax.text(0, 0, 'No Data', ha='center', va='center', 
                           fontsize=12, color='white', weight='bold')
                
                ax.set_title(f'Ext {ext_num}: {get_extension_title(ext_num)}', fontsize=12)
                ax.set_xlabel('X [pixels]', fontsize=10)
                ax.set_ylabel('Y [pixels]', fontsize=10)
            
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            
            # 新しいファイル名形式で保存
            frame_filename = f"{time_str}_{wl_value}_ext3-4-5-12.png"
            frame_path = frames_dir / frame_filename
            save_frame_as_image(frame_fig, str(frame_path))
            plt.close(frame_fig)
            print(f"  Frame saved: {frame_path}")
        
        # 全フレーム作成後に動画を作成
        print(f"\nCreating video from {len(times)} frames...")
        if create_video_from_images(str(frames_dir), save_path, fps):
            print(f"Animation saved: {save_path}")
            return str(save_path)
        else:
            print("Failed to create video from images")
            return None
            
    except Exception as e:
        print(f"Error during animation creation: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # フレームディレクトリを保持（デバッグ用）
        print(f"Frames preserved in: {frames_dir}")
        plt.close(fig)

def create_ucomp_animation_custom2(start_time, end_time, wavelength=None, 
                                  fps=2, interval=500, output_filename=None):
    """
    Custom Group2 (Ext 7-10) の時系列動画を作成
    """
    print("=" * 60)
    print("Creating UCOMP Custom Group 2 Animation (Ext 7-10)")
    print("=" * 60)
    
    times = get_available_ucomp_times(start_time, end_time, wavelength)
    
    if len(times) == 0:
        print("No UCOMP data found for the specified time range")
        return None
    
    print(f"Found {len(times)} time points")
    
    if output_filename is None:
        from astropy.time import Time
        start_str = Time(start_time).strftime("%Y%m%d_%H%M")
        end_str = Time(end_time).strftime("%Y%m%d_%H%M")
        output_filename = f"ucomp_custom2_animation_{start_str}-{end_str}.mp4"
    
    output_path = get_ucomp_output_path()
    save_path = output_path / output_filename
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('', fontsize=14, y=0.95)
    
    ims = []
    
    # Writer確認とフォールバック
    writer = get_available_writer()
    print(f"Available animation writer: {writer}")
    
    # 現在の実装では、直接フォールバック方式を使用（より安定）
    print("Using fallback method (individual frames -> video).")
    
    # UCOMPフォルダー内にframesディレクトリを作成
    output_path = get_ucomp_output_path()
    frames_dir = output_path / "frames" / f"ucomp_custom2_{Time(start_time).strftime('%Y%m%d_%H%M')}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating temporary images in: {frames_dir}")
    
    try:
        ext_mapping = [7, 8, 9, 10]
        
        for frame_num in range(len(times)):
            target_time = times[frame_num]
            print(f"Creating frame {frame_num+1}/{len(times)}: {target_time.iso}")
            
            closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
            if closest_data is None:
                print(f"  No data found for frame {frame_num+1}")
                continue
            
            closest_info, file_path = closest_data
            extensions = read_ucomp_extensions(file_path, max_extensions=10)
            
            # 2×2形式で画像を作成
            time_str = closest_info["date"].strftime("%y%m%d-%H%M%S")
            wl_value = closest_info["wavelength"]
            
            frame_fig, frame_axes = plt.subplots(2, 2, figsize=(10, 8))
            frame_fig.suptitle(f'UCOMP {wl_value}nm - Custom Group 2 (Ext 7-10)\n'
                              f'Time: {closest_info["date"].strftime("%Y-%m-%d %H:%M:%S")}', 
                              fontsize=14, y=0.95)
            
            for i in range(4):
                row = i // 2
                col = i % 2
                ax = frame_axes[row, col]
                ext_num = ext_mapping[i]
                
                if ext_num in extensions:
                    data, header = extensions[ext_num]
                    
                    # Extension番号に応じたカラーバー範囲とカラーマップを設定
                    if ext_num == 7:
                        vmin, vmax = 0, 20
                        cmap = 'plasma'
                    elif ext_num == 8:
                        vmin, vmax = -0.3, 0.8
                        cmap = 'RdBu_r'
                    elif ext_num == 9:
                        vmin, vmax = -0.5, 0.6
                        cmap = 'RdBu_r'
                    elif ext_num == 10:
                        vmin, vmax = 0, 1.0
                        cmap = 'plasma'
                    else:
                        vmin, vmax = get_data_range(data)
                        cmap = 'plasma'
                    
                    center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
                    extent = [-center_x, center_x, -center_y, center_y]
                    
                    im = ax.imshow(data, origin='lower', cmap=cmap, aspect='equal', 
                                  vmin=vmin, vmax=vmax, extent=extent)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="1%", pad=0.1)
                    cbar = plt.colorbar(im, cax=cax)
                    cbar.set_label(get_colorbar_label(ext_num), fontsize=10)
                    
                    draw_solar_radius(ax, data)
                else:
                    dummy_data = np.zeros((100, 100))
                    center_x, center_y = 50, 50
                    extent = [-center_x, center_x, -center_y, center_y]
                    ax.imshow(dummy_data, origin='lower', cmap='gray', aspect='equal', extent=extent)
                    ax.text(0, 0, 'No Data', ha='center', va='center', 
                           fontsize=12, color='white', weight='bold')
                
                ax.set_title(f'Ext {ext_num}: {get_extension_title(ext_num)}', fontsize=12)
                ax.set_xlabel('X [pixels]', fontsize=10)
                ax.set_ylabel('Y [pixels]', fontsize=10)
            
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            
            # 新しいファイル名形式で保存
            frame_filename = f"{time_str}_{wl_value}_ext7-10.png"
            frame_path = frames_dir / frame_filename
            save_frame_as_image(frame_fig, str(frame_path))
            plt.close(frame_fig)
            print(f"  Frame saved: {frame_path}")
        
        # 全フレーム作成後に動画を作成
        print(f"\nCreating video from {len(times)} frames...")
        if create_video_from_images(str(frames_dir), save_path, fps):
            print(f"Animation saved: {save_path}")
            return str(save_path)
        else:
            print("Failed to create video from images")
            return None
            
    except Exception as e:
        print(f"Error during animation creation: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # フレームディレクトリを保持（デバッグ用）
        print(f"Frames preserved in: {frames_dir}")
        plt.close(fig)