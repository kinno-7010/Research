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
    HMI磁場マップの高速プロット
    
    Parameters:
    -----------
    hmi_data : dict
        HMIデータ
    downsample : int
        ダウンサンプリング率（高速化のため）
    """
    print(f"\n=== HMI磁場マップ（高速プロット） ===")
    
    data = hmi_data['data']
    time_str = hmi_data['time']
    
    # 座標情報を取得
    if 'lon' in hmi_data and 'lat' in hmi_data:
        lon_full = hmi_data['lon']
        lat_full = hmi_data['lat']
        use_solar_coords = True
        print("太陽座標系を使用します")
    else:
        use_solar_coords = False
        print("ピクセル座標系を使用します")
    
    # ダウンサンプリングで高速化
    if downsample > 1:
        data_plot = data[::downsample, ::downsample]
        if use_solar_coords:
            lon_plot = lon_full[::downsample]
            lat_plot = lat_full[::downsample]
        print(f"ダウンサンプリング: {data.shape} → {data_plot.shape}")
    else:
        data_plot = data
        if use_solar_coords:
            lon_plot = lon_full
            lat_plot = lat_full
    
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
    
    # プロット作成
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'HMI Magnetogram Quick View\n{time_str}', fontsize=14, fontweight='bold')
    
    # カラースケールの調整
    vmax = min(abs(data_max), abs(data_min), 500)
    
    # 太陽の中心を原点に合わせるための座標設定
    ny, nx = data_plot.shape
    center_y, center_x = ny // 2, nx // 2
    extent = [-center_x, nx-center_x, -center_y, ny-center_y]
    
    # プロファイル用の座標設定（太陽中心からのピクセル数で指定）
    x_from_center = -250   # 太陽中心からX方向にピクセル
    y_from_center = 150  # 太陽中心からY方向にピクセル
    
    # ピクセルスケールを取得してarcsecに変換
    cdelt1 = hmi_data['coord_info']['cdelt1']  # X方向のピクセルスケール (arcsec/pixel)
    cdelt2 = hmi_data['coord_info']['cdelt2']  # Y方向のピクセルスケール (arcsec/pixel)
    
    x_from_center_arcsec = x_from_center * cdelt1
    y_from_center_arcsec = y_from_center * cdelt2
    
    print(f"ピクセルスケール: X={cdelt1:.2f} arcsec/pixel, Y={cdelt2:.2f} arcsec/pixel")
    print(f"指定位置: X={x_from_center} pixel = {x_from_center_arcsec:.1f} arcsec")
    print(f"指定位置: Y={y_from_center} pixel = {y_from_center_arcsec:.1f} arcsec")
    
    # 実際の配列インデックスに変換
    center_col = x_from_center + center_x
    center_row = y_from_center + center_y
    
    
    # 3. 中央行プロファイル（xlim範囲内のデータのみ使用）
    # axes[0,0]のxlim設定を取得
    x_min, x_max = -512, -50  # xlim範囲（ピクセル単位）
    x_min_arcsec, x_max_arcsec = -1024, 0  # arcsecに変換
    
    # 全行プロファイルを取得
    row_profile_full = data_plot[center_row, :]
    x_coords_full = np.arange(len(row_profile_full)) - center_x
    
    # xlim範囲内のインデックスを特定
    x_mask = (x_coords_full >= x_min_arcsec) & (x_coords_full <= x_max_arcsec)
    row_profile = row_profile_full[x_mask]
    x_coords = x_coords_full[x_mask]
    x_coords_arcsec = x_coords * cdelt1  # arcsecに変換

    # 4. 中央列プロファイル（ylim範囲内のデータのみ使用）
    y_min, y_max = -50, 512  # ylim範囲（ピクセル単位）
    y_min_arcsec, y_max_arcsec = -200, 1024  # arcsecに変換
   
    
    # 全列プロファイルを取得
    col_profile_full = data_plot[:, center_col]
    y_coords_full = np.arange(len(col_profile_full)) - center_y
    
    
    # ylim範囲内のインデックスを特定
    y_mask = (y_coords_full >= y_min_arcsec) & (y_coords_full <= y_max_arcsec)
    col_profile = col_profile_full[y_mask]
    y_coords = y_coords_full[y_mask]
    y_coords_arcsec = y_coords * cdelt2  # arcsecに変換
    
    # ---------------------------------------------------------------------------------
    # プロット作成
    lat, lon = 20, -40
    
    if use_solar_coords:
        # 太陽座標系を使用（SunPy Map オブジェクトが利用可能な場合はWCS軸を使用）
        extent_solar = [lon_plot[0], lon_plot[-1], lat_plot[0], lat_plot[-1]]
        
        # SunPy Map オブジェクトが利用可能かチェック
        hmi_map = hmi_data.get('sunpy_map')
        use_wcs_projection = hmi_map is not None
        
        if use_wcs_projection:
            # WCS projection を使用してプロット
            print("WCS projection を使用します")
            # 既存のaxes[0,0]を削除してWCS軸で再作成
            axes[0,0].remove()
            axes[0,0] = fig.add_subplot(2, 2, 1, projection=hmi_map.wcs)
            
            # 1. 磁場マップ（WCS座標系）
            im1 = axes[0,0].imshow(data_masked, cmap='RdBu_r', origin='lower', 
                                   vmin=-200, vmax=200)
            axes[0,0].set_title('Radial Magnetic Field')
            axes[0,0].set_xlabel('Solar X (arcsec)')
            axes[0,0].set_ylabel('Solar Y (arcsec)')
            # axes[0,0].set_xlim(x_min_arcsec, x_max_arcsec)
            # axes[0,0].set_ylim(y_min_arcsec, y_max_arcsec)
            
            # SunPy の draw_grid メソッドを使用
            grid_success = draw_hmi_solar_grid(hmi_map, axes[0,0])
            if not grid_success:
                print("フォールバック: 手動グリッドを使用")
                add_solar_grid_fallback(axes[0,0], extent_solar)
            
            # 指定した緯度・経度の線を描画（修正版）
            draw_solar_coordinate_lines(hmi_map, axes[0,0], target_lat=lat, target_lon=lon)
          
            axes[0,0].legend(loc='upper right')
            
    
    # 2. 磁場強度
    magnitude = np.abs(data_masked)
    
    if use_solar_coords:
        # 2番目のプロット: 磁場強度
        if use_wcs_projection:
            # WCS projection を使用
            axes[0,1].remove()
            axes[0,1] = fig.add_subplot(2, 2, 2, projection=hmi_map.wcs)
            
            im2 = axes[0,1].imshow(magnitude, cmap='plasma', origin='lower', 
                                   vmax=200)
            axes[0,1].set_title('Magnetic Field Strength')
            axes[0,1].set_xlabel('Solar X (arcsec)')
            axes[0,1].set_ylabel('Solar Y (arcsec)')
            # axes[0,1].set_xlim(x_min_arcsec, x_max_arcsec)
            # axes[0,1].set_ylim(y_min_arcsec, y_max_arcsec)
            # axes[0,1].axhline(y=y_from_center_arcsec, color='magenta', linestyle='--', alpha=0.5, linewidth=1, label=f'Y={y_from_center_arcsec:.1f}(arcsec)')
            # axes[0,1].axvline(x=x_from_center_arcsec, color='cyan', linestyle='--', alpha=0.5, linewidth=1, label=f'X={x_from_center_arcsec:.1f}(arcsec)')
            
            # SunPy の draw_grid メソッドを使用
            grid_success = draw_hmi_solar_grid(hmi_map, axes[0,1])
            if not grid_success:
                add_solar_grid_fallback(axes[0,1], extent_solar)
            
            # 指定した緯度・経度の線を描画（修正版）
            draw_solar_coordinate_lines(hmi_map, axes[0,1], target_lat=lat, target_lon=lon)
            
            axes[0,1].legend(loc='upper right')
    

    # col_profileは既にy_maskで制限されているので、直接計算
    valid_col_data = col_profile[np.isfinite(col_profile)]
    col_profile_mean = np.mean(valid_col_data)
    col_profile_std = np.std(valid_col_data)

    axes[1,0].plot(y_coords_arcsec, col_profile, '-', color='cyan', linewidth=1, label=f'Ave$\\pm$std: {col_profile_mean:.1f} $\\pm$ {col_profile_std:.1f} Gauss')
    axes[1,0].set_title(f'On X={x_from_center_arcsec:.1f} arcsec line profile')
    axes[1,0].set_xlabel('Y (arcsec from Sun center)')
    axes[1,0].set_ylabel('Br (Gauss)')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1,0].axvline(x=0, color='orange', linestyle=':', alpha=0.7, linewidth=1)  # 太陽中心線
    axes[1,0].set_xlim(y_min_arcsec, y_max_arcsec)  # arcsec単位で設定
    axes[1,0].legend(loc='upper right')
    axes[1,0].axhline(y=col_profile_mean, color='cyan', linestyle='--', alpha=0.8, linewidth=2)
    # col_profileのave±stdの範囲にシェードをかける
    axes[1,0].fill_between(y_coords_arcsec,  # arcsec座標を使用
                          col_profile_mean - col_profile_std,
                          col_profile_mean + col_profile_std,
                          color='gray', alpha=0.2)

    valid_row_data = row_profile[np.isfinite(row_profile)]
    row_profile_mean = np.mean(valid_row_data)
    row_profile_std = np.std(valid_row_data)


    axes[1,1].plot(x_coords_arcsec, row_profile, '-', color='magenta', linewidth=1, label=f'Ave$\\pm$std: {row_profile_mean:.1f} $\\pm$ {row_profile_std:.1f} Gauss')
    axes[1,1].set_title(f'On Y={y_from_center_arcsec:.1f} arcsec line profile')
    axes[1,1].set_xlabel('X (arcsec from Sun center)')
    axes[1,1].set_ylabel('Br (Gauss)')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1,1].axvline(x=0, color='orange', linestyle=':', alpha=0.7, linewidth=1)  # 太陽中心線
    axes[1,1].set_xlim(x_min_arcsec, x_max_arcsec)  # arcsec単位で設定
    axes[1,1].legend(loc='upper right')
    axes[1,1].axhline(y=row_profile_mean, color='magenta', linestyle='--', alpha=0.8, linewidth=2)
    # row_profileのave±stdの範囲にシェードをかける
    axes[1,1].fill_between(x_coords_arcsec,  # arcsec座標を使用
                          row_profile_mean - row_profile_std,
                          row_profile_mean + row_profile_std,
                          color='gray', alpha=0.2)

    plt.tight_layout()
    
    # 保存場所をResearch/SDO/HMIに変更
    save_dir = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI"
    filename = f"hmi_quick_view_{time_str.replace(':', '').replace('.', '').replace('-', '')}.png"
    full_path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    print(f"✓ 高速プロット保存: {full_path}")
    
    plt.show()
    
    # 簡易統計
    print(f"\n統計情報:")
    print(f"  元データサイズ: {data.shape}")
    print(f"  プロット解像度: {data_plot.shape}")
    print(f"  磁場範囲: {data_min:.1f} ～ {data_max:.1f} Gauss")
    print(f"  平均磁場: {np.mean(valid_data):.1f} ± {data_std:.1f} Gauss")
    print(f"  有効ピクセル率: {100*len(valid_data)/data_plot.size:.1f}%")

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