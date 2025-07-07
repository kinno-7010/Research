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

def plot_ucomp_extensions_group1(target_time, start_time, end_time, wavelength=DEFAULT_WAVELENGTH):
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
    fig.suptitle(f'UCOMP {wavelength}nm - Stokes I Basic Parameters (Ext 1-3)\n'
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

def plot_ucomp_extensions_group2(target_time, start_time, end_time, wavelength=DEFAULT_WAVELENGTH):
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

def plot_ucomp_extensions_group3(target_time, start_time, end_time, wavelength=DEFAULT_WAVELENGTH):
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

def plot_ucomp_extensions_group4(target_time, start_time, end_time, wavelength=DEFAULT_WAVELENGTH):
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

def plot_all_ucomp_groups(target_time, start_time, end_time, wavelength=DEFAULT_WAVELENGTH):
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

def create_ucomp_animation_group1(start_time, end_time, wavelength=DEFAULT_WAVELENGTH, 
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
    
    # 利用可能な時刻を取得
    times = get_available_ucomp_times(start_time, end_time, wavelength)
    
    if len(times) == 0:
        print("No UCOMP data found for the specified time range")
        return None
    
    print(f"Found {len(times)} time points")
    
    # 出力ファイル名の設定
    if output_filename is None:
        from astropy.time import Time
        start_str = Time(start_time).strftime("%Y%m%d_%H%M")
        end_str = Time(end_time).strftime("%Y%m%d_%H%M")
        output_filename = f"ucomp_group1_animation_{start_str}-{end_str}.mp4"
    
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
        fig.suptitle(f'UCOMP {wavelength}nm - Stokes I Basic Parameters (Ext 1-3)\n'
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
                vmin, vmax = get_data_range(data)
                
                ims[i].set_array(data)
                ims[i].set_clim(vmin, vmax)
                
                # 太陽半径描画
                draw_solar_radius(ax, data)
            else:
                ims[i].set_array(np.zeros((100, 100)))
        
        return ims
    
    # アニメーション作成
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=len(times), 
                                 interval=interval, blit=False, repeat=True)
    
    # MP4として保存
    print(f"Saving animation to: {save_path}")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='UCOMP Analysis'), bitrate=1800)
    
    anim.save(save_path, writer=writer)
    plt.close(fig)
    
    print(f"Animation saved: {save_path}")
    return str(save_path)

def create_ucomp_animation_group2(start_time, end_time, wavelength=DEFAULT_WAVELENGTH, 
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
    
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=len(times), 
                                 interval=interval, blit=False, repeat=True)
    
    print(f"Saving animation to: {save_path}")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='UCOMP Analysis'), bitrate=1800)
    
    anim.save(save_path, writer=writer)
    plt.close(fig)
    
    print(f"Animation saved: {save_path}")
    return str(save_path)

def create_ucomp_animation_group3(start_time, end_time, wavelength=DEFAULT_WAVELENGTH, 
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
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='UCOMP Analysis'), bitrate=1800)
    
    anim.save(save_path, writer=writer)
    plt.close(fig)
    
    print(f"Animation saved: {save_path}")
    return str(save_path)

def create_ucomp_animation_group4(start_time, end_time, wavelength=DEFAULT_WAVELENGTH, 
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
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='UCOMP Analysis'), bitrate=1800)
    
    anim.save(save_path, writer=writer)
    plt.close(fig)
    
    print(f"Animation saved: {save_path}")
    return str(save_path)

def create_all_ucomp_animations(start_time, end_time, wavelength=DEFAULT_WAVELENGTH, 
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