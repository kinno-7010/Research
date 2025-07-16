#!/usr/bin/env python3
"""
HMI磁場データの高速プロット（簡易版）
時刻: 2022-06-13T03:00:00
修正版: SphericalScreen()コンテキストマネージャーを使用してSunPy座標変換警告を解決
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# 現在のディレクトリをPythonパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from astropy.io import fits
import astropy.units as u
import sunpy.map
from sunpy.coordinates import SphericalScreen  # 重要: SphericalScreenを追加
import warnings
warnings.filterwarnings("ignore", category=sunpy.util.exceptions.SunpyUserWarning)

def draw_hmi_solar_grid(hmi_map, ax, color='gray'):
    """
    HMI データに対してSunPyの draw_grid メソッドを使用して太陽座標グリッドを描画
    
    Parameters:
    -----------
    hmi_map : sunpy.map.Map
        HMI Map オブジェクト（WCS情報付き）
    ax : matplotlib.axes.Axes
        WCS projection を持つプロット軸
    """
    try:
        # SunPyの標準的な太陽グリッド描画メソッドを使用
        # AIA_analysis.ipynbのplot_sdo_aia関数と同じパラメータ
        hmi_map.draw_grid(
            axes=ax, 
            grid_spacing=15*u.deg,  # 15度間隔のグリッド
            color=color, 
            linestyle='dotted', 
            linewidth=0.8, 
            alpha=0.7
        )
        print("情報: SunPyの draw_grid メソッドで太陽座標グリッドを描画しました")
        return True
    except Exception as e:
        print(f"警告: SunPy グリッド描画に失敗: {e}")
        return False

def draw_solar_coordinate_lines(hmi_map, ax, target_lat, target_lon):
    """
    指定した緯度・経度に沿って太陽座標系の線を描画
    SphericalScreen()を使用して座標変換警告を解決
    
    Parameters:
    -----------
    hmi_map : sunpy.map.Map
        HMI Map オブジェクト（WCS情報付き）
    ax : matplotlib.axes.Axes
        WCS projection を持つプロット軸
    target_lat : float
        描画する緯度線（度）
    target_lon : float
        描画する経度線（度）
    """
    try:
        from astropy.coordinates import SkyCoord
        from sunpy.coordinates import frames
        
        # SphericalScreen()コンテキストマネージャーを使用して座標変換警告を解決
        with SphericalScreen(hmi_map.observer_coordinate):
            
            # 緯度線を描画（固定緯度で経度を変化）
            lon_range = np.linspace(-90, 90, 300)  # 経度範囲（より密に）
            lat_constant = np.full_like(lon_range, target_lat)  # 固定緯度
            
            # Helioprojective座標系で直接作成（observer情報を使用）
            # 緯度線を描画
            coords_lat_line = SkyCoord(
                lon_range * u.deg, 
                lat_constant * u.deg, 
                frame=frames.HeliographicStonyhurst, # <- 修正後
                obstime=hmi_map.date,
                observer=hmi_map.observer_coordinate
            )
            
            # Helioprojective座標系に変換（SphericalScreen内で実行）
            coords_lat_hpc = coords_lat_line.transform_to(hmi_map.coordinate_frame)
            
            # WCSを使用してピクセル座標に変換
            pixel_coords_lat = hmi_map.world_to_pixel(coords_lat_hpc)
            
            # ピクセル座標を数値配列として取得（単位を除去）
            if hasattr(pixel_coords_lat, 'x'):
                x_lat = pixel_coords_lat.x.value if hasattr(pixel_coords_lat.x, 'value') else pixel_coords_lat.x
                y_lat = pixel_coords_lat.y.value if hasattr(pixel_coords_lat.y, 'value') else pixel_coords_lat.y
            else:
                x_lat = pixel_coords_lat[0].value if hasattr(pixel_coords_lat[0], 'value') else pixel_coords_lat[0]
                y_lat = pixel_coords_lat[1].value if hasattr(pixel_coords_lat[1], 'value') else pixel_coords_lat[1]
            
            # 有効な座標のみを選択（画像範囲内かつ有限値）
            valid_mask_lat = (
                (x_lat >= 0) & (x_lat < hmi_map.data.shape[1]) &
                (y_lat >= 0) & (y_lat < hmi_map.data.shape[0]) &
                np.isfinite(x_lat) & np.isfinite(y_lat)
            )
            
            if valid_mask_lat.sum() > 0:
                # 緯度線を描画
                ax.plot(x_lat[valid_mask_lat], y_lat[valid_mask_lat], 
                       color='magenta', linestyle='-', linewidth=2.5, alpha=0.9, 
                       label=f'Latitude {target_lat}°')
                print(f"情報: 緯度線 {target_lat}° を描画しました ({valid_mask_lat.sum()} 点)")
            else:
                print(f"警告: 緯度線 {target_lat}° の有効な点が見つかりませんでした")
            
            # 経度線を描画（固定経度で緯度を変化）
            lat_range = np.linspace(-90, 90, 300)  # 緯度範囲（より密に）
            lon_constant = np.full_like(lat_range, target_lon)  # 固定経度
            
            # 経度線を描画
            coords_lon_line = SkyCoord(
                lon_constant * u.deg, 
                lat_range * u.deg, 
                frame=frames.HeliographicStonyhurst, # <- 修正後
                obstime=hmi_map.date,
                observer=hmi_map.observer_coordinate
            )
            
            # Helioprojective座標系に変換（SphericalScreen内で実行）
            coords_lon_hpc = coords_lon_line.transform_to(hmi_map.coordinate_frame)
            
            # WCSを使用してピクセル座標に変換
            pixel_coords_lon = hmi_map.world_to_pixel(coords_lon_hpc)
            
            # ピクセル座標を数値配列として取得（単位を除去）
            if hasattr(pixel_coords_lon, 'x'):
                x_lon = pixel_coords_lon.x.value if hasattr(pixel_coords_lon.x, 'value') else pixel_coords_lon.x
                y_lon = pixel_coords_lon.y.value if hasattr(pixel_coords_lon.y, 'value') else pixel_coords_lon.y
            else:
                x_lon = pixel_coords_lon[0].value if hasattr(pixel_coords_lon[0], 'value') else pixel_coords_lon[0]
                y_lon = pixel_coords_lon[1].value if hasattr(pixel_coords_lon[1], 'value') else pixel_coords_lon[1]
            
            # 有効な座標のみを選択（画像範囲内かつ有限値）
            valid_mask_lon = (
                (x_lon >= 0) & (x_lon < hmi_map.data.shape[1]) &
                (y_lon >= 0) & (y_lon < hmi_map.data.shape[0]) &
                np.isfinite(x_lon) & np.isfinite(y_lon)
            )
            
            if valid_mask_lon.sum() > 0:
                # 経度線を描画
                ax.plot(x_lon[valid_mask_lon], y_lon[valid_mask_lon], 
                       color='cyan', linestyle='-', linewidth=2.5, alpha=0.9, 
                       label=f'Longitude {target_lon}°')
                print(f"情報: 経度線 {target_lon}° を描画しました ({valid_mask_lon.sum()} 点)")
            else:
                print(f"警告: 経度線 {target_lon}° の有効な点が見つかりませんでした")
        
        print(f"情報: 太陽座標線を描画しました (lat={target_lat}°, lon={target_lon}°)")
        return True
        
    except Exception as e:
        print(f"警告: 太陽座標線描画に失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def add_solar_grid_fallback(ax, extent, grid_spacing_deg=30):
    """
    フォールバック用の太陽座標グリッド（SunPyが使用できない場合）
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        プロット軸
    extent : list
        [lon_min, lon_max, lat_min, lat_max]
    grid_spacing_deg : float
        グリッド間隔（度）
    """
    lon_min, lon_max, lat_min, lat_max = extent
    
    # 表示範囲に応じてグリッド間隔を調整
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    
    # 動的グリッド間隔
    if lon_range > 180:
        major_spacing = 60
        minor_spacing = 30
    elif lon_range > 90:
        major_spacing = 30
        minor_spacing = 15
    else:
        major_spacing = 15
        minor_spacing = 5
    
    # メジャーグリッド線（太い線）
    lon_major = np.arange(0, 361, major_spacing)
    lat_major = np.arange(-90, 91, major_spacing)
    
    # マイナーグリッド線（細い線）
    lon_minor = np.arange(0, 361, minor_spacing)
    lat_minor = np.arange(-90, 91, minor_spacing)
    
    # マイナーグリッド（経度）
    for lon in lon_minor:
        if lon_min <= lon <= lon_max and lon not in lon_major:
            ax.axvline(x=lon, color='white', linestyle='-', alpha=0.2, linewidth=0.5)
    
    # マイナーグリッド（緯度）
    for lat in lat_minor:
        if lat_min <= lat <= lat_max and lat not in lat_major:
            ax.axhline(y=lat, color='white', linestyle='-', alpha=0.2, linewidth=0.5)
    
    # メジャーグリッド（経度）
    for lon in lon_major:
        if lon_min <= lon <= lon_max:
            ax.axvline(x=lon, color='white', linestyle='-', alpha=0.5, linewidth=1.0)
            # 経度ラベル（赤道付近に表示）
            label_lat = max(lat_min + (lat_max - lat_min) * 0.05, lat_min + 5)
            if lat_min <= label_lat <= lat_max:
                ax.text(lon, label_lat, f'{lon:.0f}°', ha='center', va='bottom', 
                       color='white', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # メジャーグリッド（緯度）
    for lat in lat_major:
        if lat_min <= lat <= lat_max:
            ax.axhline(y=lat, color='white', linestyle='-', alpha=0.5, linewidth=1.0)
            # 緯度ラベル（左端に表示）
            label_lon = lon_min + (lon_max - lon_min) * 0.02
            if lon_min <= label_lon <= lon_max:
                ax.text(label_lon, lat, f'{lat:.0f}°', ha='left', va='center', 
                       color='white', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # 特別な線（赤道）
    if lat_min <= 0 <= lat_max:
        ax.axhline(y=0, color='yellow', linestyle='-', alpha=0.8, linewidth=2.0)
        # 赤道ラベル
        ax.text(lon_min + (lon_max - lon_min) * 0.95, 0, 'Equator', ha='right', va='bottom', 
               color='yellow', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
    
    # 中央子午線
    lon_center = (lon_min + lon_max) / 2
    if lon_min <= lon_center <= lon_max:
        ax.axvline(x=lon_center, color='orange', linestyle='--', alpha=0.8, linewidth=1.5)
        # 中央子午線ラベル
        ax.text(lon_center, lat_max - (lat_max - lat_min) * 0.05, f'CM: {lon_center:.0f}°', 
               ha='center', va='top', color='orange', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    print("情報: フォールバック太陽座標グリッドを描画しました")

def read_hmi_quick(filepath):
    """
    HMI磁場マップを高速読み込み（座標情報付き）
    SunPy Map オブジェクトも作成して WCS 情報を保持
    """
    print(f"HMI磁場データを読み込み中: {os.path.basename(filepath)}")
    
    try:
        # SunPy Map オブジェクトを作成（WCS情報を適切に処理）

        hmi_map = sunpy.map.Map(filepath)
        data = hmi_map.data
        header = hmi_map.meta
        print(f"  SunPy Map オブジェクトを作成しました")
        use_sunpy_map = True

        
        # フォールバック: 直接FITS読み込み
        if not use_sunpy_map:
            with fits.open(filepath) as hdul:
                # データの取得
                if len(hdul) > 1 and hdul[1].data is not None:
                    data = hdul[1].data
                    header = hdul[1].header
                else:
                    data = hdul[0].data
                    header = hdul[0].header
        
        print(f"  データ形状: {data.shape}")
        print(f"  観測時刻: {header.get('T_OBS', 'Unknown')}")
        
        # 統計情報
        valid_data = data[np.isfinite(data)]
        if len(valid_data) > 0:
            print(f"  磁場範囲: {np.min(valid_data):.1f} ～ {np.max(valid_data):.1f} Gauss")
        
        # 座標情報を取得
        naxis1 = header.get('NAXIS1', data.shape[1])  # 経度方向
        naxis2 = header.get('NAXIS2', data.shape[0])  # 緯度方向
        
        # Carrington座標系での経度・緯度
        crval1 = header.get('CRVAL1', 180.0)  # 中心経度
        cdelt1 = header.get('CDELT1', 360.0/naxis1)  # 経度刻み
        crpix1 = header.get('CRPIX1', naxis1/2 + 0.5)
        
        crval2 = header.get('CRVAL2', 0.0)    # 中心緯度
        cdelt2 = header.get('CDELT2', 180.0/naxis2)  # 緯度刻み
        crpix2 = header.get('CRPIX2', naxis2/2 + 0.5)
        
        # 経度・緯度配列を計算
        lon = crval1 + (np.arange(naxis1) + 1 - crpix1) * cdelt1
        lat = crval2 + (np.arange(naxis2) + 1 - crpix2) * cdelt2
        
        # 経度を0-360度範囲に正規化
        lon = (lon + 360) % 360
        
        # 緯度を-90～90度範囲に制限
        lat = np.clip(lat, -90, 90)
        
        print(f"  座標範囲: 経度 {lon[0]:.1f}°-{lon[-1]:.1f}°, 緯度 {lat[0]:.1f}°-{lat[-1]:.1f}°")
        
        return {
            'data': data,
            'header': header,
            'time': header.get('T_OBS', '2022-06-13T03:00:00'),
            'lon': lon,
            'lat': lat,
            'sunpy_map': hmi_map,  # SunPy Map オブジェクト（WCS情報付き）
            'coord_info': {
                'crval1': crval1, 'cdelt1': cdelt1, 'crpix1': crpix1,
                'crval2': crval2, 'cdelt2': cdelt2, 'crpix2': crpix2
            }
        }
        
    except Exception as e:
        print(f"読み込みエラー: {e}")
        return None

def plot_hmi_quick(hmi_data, downsample=1):
    """
    HMI磁場マップの高速プロット（座標系完全修正版）
    
    Parameters:
    -----------
    hmi_data : dict
        HMIデータ
    downsample : int
        ダウンサンプリング率（高速化のため）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u
    import os
    
    # 必要な関数をグローバルスコープから参照
    global draw_hmi_solar_grid, draw_solar_coordinate_lines
    
    print(f"\n=== HMI磁場マップ（高速プロット） ===")
    
    data = hmi_data['data']
    time_str = hmi_data['time']
    hmi_map = hmi_data.get('sunpy_map')
    
    if hmi_map is None:
        print("エラー: SunPy Mapオブジェクトが必要です")
        return
    
    # ダウンサンプリング
    if downsample > 1:
        data_plot = data[::downsample, ::downsample]
        print(f"ダウンサンプリング: {data.shape} → {data_plot.shape}")
    else:
        data_plot = data
    
    # NaN値をマスク
    data_masked = np.ma.masked_invalid(data_plot)
    
    # 統計情報
    valid_data = data_plot[np.isfinite(data_plot)]
    if len(valid_data) == 0:
        print("有効データがありません")
        return
    
    data_min = np.min(valid_data)
    data_max = np.max(valid_data)
    data_std = np.std(valid_data)
    
    print(f"磁場統計: {data_min:.1f} ～ {data_max:.1f} Gauss (σ={data_std:.1f})")
    
    # プロット作成（figureサイズを大きく調整）
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f'HMI Magnetogram Quick View\n{time_str}', fontsize=16, fontweight='bold', y=0.98)
    
    # GridSpecを使用してサブプロットの配置を調整
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.3, 
                  left=0.08, right=0.95, top=0.93, bottom=0.05)
    
    # 通常のaxesを先に作成（プロファイル用）
    # axesをNumPy配列に変換して、[row, col]形式のインデックスを使えるようにする
    axes = np.array([[None, None], [None, None]], dtype=object)
    axes[1, 0] = fig.add_subplot(gs[1, 0])
    axes[1, 1] = fig.add_subplot(gs[1, 1])
    
    # === 座標系の理解と変換 ===
    # 元のコードの座標系（太陽中心基準のピクセル座標）
    x_min_solar_center = -512  # 太陽中心から左に512ピクセル
    x_max_solar_center = 0   # 太陽中心から左に50ピクセル
    y_min_solar_center = -100   # 太陽中心から下に50ピクセル
    y_max_solar_center = 512   # 太陽中心から上に512ピクセル
    
    # 画像サイズと中心
    ny, nx = data_plot.shape
    center_x = nx // 2
    center_y = ny // 2
    
    # 配列インデックスに変換（0始まり）
    x_min_pix = center_x + x_min_solar_center
    x_max_pix = center_x + x_max_solar_center
    y_min_pix = center_y + y_min_solar_center
    y_max_pix = center_y + y_max_solar_center
    
    print(f"\n座標変換の詳細:")
    print(f"画像サイズ: {nx} x {ny}")
    print(f"太陽中心（ピクセル）: ({center_x}, {center_y})")
    print(f"太陽中心基準座標: X=[{x_min_solar_center}, {x_max_solar_center}], Y=[{y_min_solar_center}, {y_max_solar_center}]")
    print(f"配列インデックス: X=[{x_min_pix}, {x_max_pix}], Y=[{y_min_pix}, {y_max_pix}]")
    
    # 実際のarcsec値を確認
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames
    
    # 表示範囲の角のarcsec座標を取得
    corner_coords = []
    for x_pix, y_pix in [(x_min_pix, y_min_pix), (x_max_pix, y_max_pix)]:
        world = hmi_map.pixel_to_world(x_pix * u.pix, y_pix * u.pix)
        corner_coords.append((world.Tx.value, world.Ty.value))
    
    print(f"対応するarcsec範囲: X=[{corner_coords[0][0]:.1f}, {corner_coords[1][0]:.1f}], Y=[{corner_coords[0][1]:.1f}, {corner_coords[1][1]:.1f}]")
    
    # プロファイル用の座標（太陽中心基準ピクセル座標で指定）
    x_profile_solar_center = -250  # 太陽中心から左に250ピクセル
    y_profile_solar_center = 150   # 太陽中心から上に150ピクセル
    
    # 配列インデックスに変換
    profile_x_pix = center_x + x_profile_solar_center
    profile_y_pix = center_y + y_profile_solar_center
    
    # 対応するarcsec座標を取得
    profile_world = hmi_map.pixel_to_world(profile_x_pix * u.pix, profile_y_pix * u.pix)
    x_profile_arcsec = profile_world.Tx.value
    y_profile_arcsec = profile_world.Ty.value
    
    print(f"\nプロファイル位置:")
    print(f"  太陽中心基準: ({x_profile_solar_center}, {y_profile_solar_center}) pixel")
    print(f"  配列インデックス: ({profile_x_pix}, {profile_y_pix})")
    print(f"  arcsec: ({x_profile_arcsec:.1f}, {y_profile_arcsec:.1f})")
    
    # ---------------------------------------------------------------------------------
    # プロット作成
    lat, lon = 20, -40
    
    # 1. 磁場マップ（WCS座標系）
    axes[0,0] = fig.add_subplot(gs[0, 0], projection=hmi_map.wcs)
    
    im1 = axes[0,0].imshow(data_masked, cmap='RdBu_r', origin='lower', 
                           vmin=-200, vmax=200)
    axes[0,0].set_title('Radial Magnetic Field')
    axes[0,0].set_xlabel('Solar X (arcsec)')
    axes[0,0].set_ylabel('Solar Y (arcsec)')
    
    # === 重要: ピクセル座標でxlim, ylimを設定 ===
    axes[0,0].set_xlim(x_min_pix, x_max_pix)
    axes[0,0].set_ylim(y_min_pix, y_max_pix)
    
    # グリッドと座標線を描画
    draw_hmi_solar_grid(hmi_map, axes[0,0])
    draw_solar_coordinate_lines(hmi_map, axes[0,0], target_lat=lat, target_lon=lon)
    
    # プロファイル位置を示す線（ピクセル座標で描画）
    axes[0,0].legend(loc='upper right')
    
    # 2. 磁場強度（フルディスク表示）
    magnitude = np.abs(data_masked)
    
    axes[0,1] = fig.add_subplot(gs[0, 1], projection=hmi_map.wcs)
    
    im2 = axes[0,1].imshow(magnitude, cmap='plasma', origin='lower', vmax=200)
    axes[0,1].set_title('Magnetic Field Strength (Full Disk)', fontsize=14, pad=10)
    axes[0,1].set_xlabel('Solar X (arcsec)', fontsize=12)
    axes[0,1].set_ylabel('Solar Y (arcsec)', fontsize=12)
    
    # フルディスク表示（ピクセル座標で0から画像サイズまで）
    axes[0,1].set_xlim(x_min_pix, x_max_pix)
    axes[0,1].set_ylim(y_min_pix, y_max_pix)
    
    draw_hmi_solar_grid(hmi_map, axes[0,1])
    draw_solar_coordinate_lines(hmi_map, axes[0,1], target_lat=lat, target_lon=lon)
    axes[0,1].legend(loc='upper right')
    
    # フォントサイズを調整
    axes[0,1].tick_params(axis='both', which='major', labelsize=10)
    
    # 3. Y方向プロファイル（X固定）
    # 表示範囲をピクセル座標で制限
    y_pix_min = max(0, int(y_min_pix))
    y_pix_max = min(data_plot.shape[0], int(y_max_pix))
    
    if 0 <= profile_x_pix < data_plot.shape[1]:
        y_profile_data = data_plot[y_pix_min:y_pix_max, profile_x_pix]
        y_pixels = np.arange(y_pix_min, y_pix_max)
        
        # ピクセル座標をarcsec座標に変換
        y_coords_arcsec = []
        valid_values = []
        for idx, y_pix in enumerate(y_pixels):
            try:
                world = hmi_map.pixel_to_world(profile_x_pix * u.pix, y_pix * u.pix)
                y_coords_arcsec.append(world.Ty.value)
                valid_values.append(y_profile_data[idx])
            except:
                pass
        
        if len(valid_values) > 0:
            y_coords_arcsec = np.array(y_coords_arcsec)
            valid_values = np.array(valid_values)
            valid_data = valid_values[np.isfinite(valid_values)]
            
            if len(valid_data) > 0:
                profile_mean = np.mean(valid_data)
                profile_std = np.std(valid_data)
                
                axes[1,0].plot(y_coords_arcsec, valid_values, '-', color='cyan', 
                              linewidth=1.5, label=f'Ave±std: {profile_mean:.1f} ± {profile_std:.1f} G')
                axes[1,0].fill_between(y_coords_arcsec,
                                      profile_mean - profile_std,
                                      profile_mean + profile_std,
                                      color='gray', alpha=0.2)
                axes[1,0].axhline(y=profile_mean, color='cyan', linestyle='--', alpha=0.8)
    
    axes[1,0].set_title(f'Profile at X={x_profile_arcsec}" (arcsec)')
    axes[1,0].set_xlabel('Y (arcsec)')
    axes[1,0].set_ylabel('Br (Gauss)')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1,0].axvline(x=0, color='orange', linestyle=':', alpha=0.7)
    axes[1,0].set_xlim(corner_coords[0][1], corner_coords[1][1])  # Y方向のarcsec範囲
    axes[1,0].legend(loc='upper right')
    
    # 4. X方向プロファイル（Y固定）
    # 表示範囲をピクセル座標で制限
    x_pix_min = max(0, int(x_min_pix))
    x_pix_max = min(data_plot.shape[1], int(x_max_pix))
    
    if 0 <= profile_y_pix < data_plot.shape[0]:
        x_profile_data = data_plot[profile_y_pix, x_pix_min:x_pix_max]
        x_pixels = np.arange(x_pix_min, x_pix_max)
        
        # ピクセル座標をarcsec座標に変換
        x_coords_arcsec = []
        valid_values = []
        for idx, x_pix in enumerate(x_pixels):
            try:
                world = hmi_map.pixel_to_world(x_pix * u.pix, profile_y_pix * u.pix)
                x_coords_arcsec.append(world.Tx.value)
                valid_values.append(x_profile_data[idx])
            except:
                pass
        
        if len(valid_values) > 0:
            x_coords_arcsec = np.array(x_coords_arcsec)
            valid_values = np.array(valid_values)
            valid_data = valid_values[np.isfinite(valid_values)]
            
            if len(valid_data) > 0:
                profile_mean = np.mean(valid_data)
                profile_std = np.std(valid_data)
                
                axes[1,1].plot(x_coords_arcsec, valid_values, '-', color='magenta', 
                              linewidth=1.5, label=f'Ave±std: {profile_mean:.1f} ± {profile_std:.1f} G')
                axes[1,1].fill_between(x_coords_arcsec,
                                      profile_mean - profile_std,
                                      profile_mean + profile_std,
                                      color='gray', alpha=0.2)
                axes[1,1].axhline(y=profile_mean, color='magenta', linestyle='--', alpha=0.8)
    
    axes[1,1].set_title(f'Profile at Y={y_profile_arcsec}" (arcsec)')
    axes[1,1].set_xlabel('X (arcsec)')
    axes[1,1].set_ylabel('Br (Gauss)')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1,1].axvline(x=0, color='orange', linestyle=':', alpha=0.7)
    axes[1,1].set_xlim(corner_coords[0][0], corner_coords[1][0])  # X方向のarcsec範囲
    axes[1,1].legend(loc='upper right')
    
    # tight_layoutは使用しない（GridSpecで調整済み）
    # plt.tight_layout()
    
    # 保存
    save_dir = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI"
    filename = f"hmi_quick_view_{time_str.replace(':', '').replace('.', '').replace('-', '')}.png"
    full_path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    print(f"✓ 高速プロット保存: {full_path}")
    
    plt.show()
    
    # デバッグ情報
    print(f"\n=== デバッグ情報 ===")
    print(f"データ形状: {data_plot.shape}")
    print(f"太陽中心ピクセル: ({data_plot.shape[1]//2}, {data_plot.shape[0]//2})")
    print(f"フルディスク（ピクセル）: X=[0, {data_plot.shape[1]}], Y=[0, {data_plot.shape[0]}]")
    
    # 太陽中心のarcsec座標を確認
    center_pix_x = data_plot.shape[1] // 2
    center_pix_y = data_plot.shape[0] // 2
    center_world = hmi_map.pixel_to_world(center_pix_x * u.pix, center_pix_y * u.pix)
    print(f"太陽中心（arcsec）: ({center_world.Tx.value:.1f}, {center_world.Ty.value:.1f})")

def main():
    """
    メイン実行関数
    """
    print("HMI磁場マップ高速プロットツール")
    print("対象時刻: 2022-06-13T03:00:00")
    print("=" * 40)
    
    # HMI FITSファイルのパス
    hmi_file = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    
    if not os.path.exists(hmi_file):
        print(f"エラー: ファイルが見つかりません: {hmi_file}")
        return
    
    try:
        # HMI読み込み
        hmi_data = read_hmi_quick(hmi_file)
        if hmi_data is None:
            print("HMI データの読み込みに失敗しました")
            return
        
        # 高速プロット
        plot_hmi_quick(hmi_data)
        
        print("\n✓ HMI高速プロット完了")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()