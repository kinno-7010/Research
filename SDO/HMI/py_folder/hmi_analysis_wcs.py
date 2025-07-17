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
    
# ==============================================================================
# <<< 修正: データ抽出関数に表示範囲の引数を追加 >>>
# ==============================================================================
def extract_data_along_solar_lines(hmi_map, target_lat, target_lon, x_lim_pix, y_lim_pix):
    """
    指定した緯度・経度線に沿って磁場データを抽出する。
    指定されたピクセル表示範囲(x_lim_pix, y_lim_pix)内のデータのみを対象とする。
    
    Parameters:
    -----------
    hmi_map : sunpy.map.Map
        HMI Map オブジェクト
    target_lat : float
        対象の緯度（度）
    target_lon : float
        対象の経度（度）
    x_lim_pix : tuple
        プロット領域のXピクセル範囲 (xmin, xmax)
    y_lim_pix : tuple
        プロット領域のYピクセル範囲 (ymin, ymax)
        
    Returns:
    --------
    dict: 緯度線・経度線プロファイルデータを格納した辞書
    """
    profiles = {'lat_profile': None, 'lon_profile': None}
    try:
        from astropy.coordinates import SkyCoord
        from sunpy.coordinates import frames
        
        data = hmi_map.data

        with SphericalScreen(hmi_map.observer_coordinate):
            # --- 緯度線プロファイルの抽出 (緯度固定、経度を変化) ---
            lon_range = np.linspace(-90, 90, 500) # 解像度を少し上げる
            lat_constant = np.full_like(lon_range, target_lat)
            
            coords_lat_line = SkyCoord(lon_range * u.deg, lat_constant * u.deg, 
                                       frame=frames.HeliographicStonyhurst, obstime=hmi_map.date,
                                       observer=hmi_map.observer_coordinate)
            
            pixel_coords = hmi_map.world_to_pixel(coords_lat_line)
            x_pix, y_pix = pixel_coords.x.value, pixel_coords.y.value
            
            # <<< 修正: valid_maskにxlim, ylimの条件を追加 >>>
            valid_mask = (
                (x_pix >= x_lim_pix[0]) & (x_pix < x_lim_pix[1]) &
                (y_pix >= y_lim_pix[0]) & (y_pix < y_lim_pix[1]) &
                np.isfinite(x_pix) & np.isfinite(y_pix)
            )
            
            if valid_mask.sum() > 0:
                valid_x = x_pix[valid_mask].astype(int)
                valid_y = y_pix[valid_mask].astype(int)
                magnetic_values = data[valid_y, valid_x]
                valid_lon_coords = lon_range[valid_mask]
                
                profiles['lat_profile'] = {'lon_coords': valid_lon_coords, 'values': magnetic_values}
                print(f"情報: 緯度 {target_lat}° 線上の限定領域プロファイルを抽出 ({len(magnetic_values)} 点)")

            # --- 経度線プロファイルの抽出 (経度固定、緯度を変化) ---
            lat_range = np.linspace(-90, 90, 500) # 解像度を少し上げる
            lon_constant = np.full_like(lat_range, target_lon)

            coords_lon_line = SkyCoord(lon_constant * u.deg, lat_range * u.deg, 
                                       frame=frames.HeliographicStonyhurst, obstime=hmi_map.date,
                                       observer=hmi_map.observer_coordinate)

            pixel_coords = hmi_map.world_to_pixel(coords_lon_line)
            x_pix, y_pix = pixel_coords.x.value, pixel_coords.y.value

            # <<< 修正: valid_maskにxlim, ylimの条件を追加 >>>
            valid_mask = (
                (x_pix >= x_lim_pix[0]) & (x_pix < x_lim_pix[1]) &
                (y_pix >= y_lim_pix[0]) & (y_pix < y_lim_pix[1]) &
                np.isfinite(x_pix) & np.isfinite(y_pix)
            )

            if valid_mask.sum() > 0:
                valid_x = x_pix[valid_mask].astype(int)
                valid_y = y_pix[valid_mask].astype(int)
                magnetic_values = data[valid_y, valid_x]
                valid_lat_coords = lat_range[valid_mask]
                
                profiles['lon_profile'] = {'lat_coords': valid_lat_coords, 'values': magnetic_values}
                print(f"情報: 経度 {target_lon}° 線上の限定領域プロファイルを抽出 ({len(magnetic_values)} 点)")

        return profiles

    except Exception as e:
        print(f"警告: プロファイルデータ抽出に失敗: {e}")
        import traceback
        traceback.print_exc()
        return profiles

def extract_ar_region_data(data, x_min, x_max, y_min, y_max):
    """
    指定されたピクセル範囲内のデータを抽出
    
    Parameters:
    -----------
    data : numpy.ndarray
        HMI磁場データ
    x_min, x_max, y_min, y_max : int
        活動領域のピクセル座標範囲
        
    Returns:
    --------
    numpy.ndarray : 活動領域内のデータ
    """
    # 範囲をデータサイズに制限
    x_min = max(0, int(x_min))
    x_max = min(data.shape[1], int(x_max))
    y_min = max(0, int(y_min))
    y_max = min(data.shape[0], int(y_max))
    
    # 指定範囲のデータを抽出
    ar_data = data[y_min:y_max, x_min:x_max]
    
    print(f"活動領域データ抽出:")
    print(f"  ピクセル範囲: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]")
    print(f"  抽出データサイズ: {ar_data.shape}")
    
    return ar_data

def plot_magnetic_field_histogram(ax, ar_data, region_name="AR Region"):
    """
    活動領域内の磁場強度を符号別にヒストグラム表示
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        プロット軸
    ar_data : numpy.ndarray
        活動領域の磁場データ
    region_name : str
        領域名（タイトル用）
    """
    import numpy as np
    
    # 有効なデータのみを抽出
    valid_data = ar_data[np.isfinite(ar_data)]
    
    if len(valid_data) == 0:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        return
    
    # 正負のデータを分離
    positive_data = valid_data[valid_data > 0]
    negative_data = valid_data[valid_data < 0]
    
    # 統計情報
    print(f"\n磁場統計（{region_name}）:")
    print(f"  全体: {len(valid_data)} pixels")
    print(f"  正の磁場（N極）: {len(positive_data)} pixels ({100*len(positive_data)/len(valid_data):.1f}%)")
    print(f"  負の磁場（S極）: {len(negative_data)} pixels ({100*len(negative_data)/len(valid_data):.1f}%)")
    
    if len(positive_data) > 0:
        print(f"  正の磁場: 平均={np.mean(positive_data):.1f} G, 最大={np.max(positive_data):.1f} G")
    if len(negative_data) > 0:
        print(f"  負の磁場: 平均={np.mean(negative_data):.1f} G, 最小={np.min(negative_data):.1f} G")
    
    # ビンの設定（符号別に適切な範囲を設定）
    max_abs_value = valid_data.max()
    min_abs_value = valid_data.min()
    
    # より細かいビン幅でヒストグラムを作成
    bin_width = 10  # 20 Gaussごと
    
    # 正の磁場用のビン
    if len(positive_data) > 0:
        pos_bins = np.arange(0, np.max(positive_data) + bin_width, bin_width)
    else:
        pos_bins = np.array([0, 1])
    
    # 負の磁場用のビン
    if len(negative_data) > 0:
        neg_bins = np.arange(np.min(negative_data), 0 + bin_width, bin_width)
    else:
        neg_bins = np.array([-1, 0])
    
    # ヒストグラムの作成
    # 正の磁場（赤色）
    if len(positive_data) > 0:
        n_pos, bins_pos, patches_pos = ax.hist(positive_data, bins=pos_bins, 
                                                color='red', alpha=0.7, 
                                                label=f'Positive (N={len(positive_data)})', 
                                                edgecolor='darkred', linewidth=1.2)
    
    # 負の磁場（青色）
    if len(negative_data) > 0:
        n_neg, bins_neg, patches_neg = ax.hist(negative_data, bins=neg_bins, 
                                                color='blue', alpha=0.7, 
                                                label=f'Negative (N={len(negative_data)})', 
                                                edgecolor='darkblue', linewidth=1.2)
    
    # プロットの装飾
    ax.set_xlabel('Magnetic Field Strength (Gauss)', fontsize=12)
    ax.set_ylabel('Number of Pixels', fontsize=12)
    ax.set_title(f'Magnetic Field Distribution in {region_name}', fontsize=14, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # x軸の範囲を調整（ゼロを中心に対称的に）
    ax.set_xlim(min_abs_value, max_abs_value)
    
    # ゼロラインを強調
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8)
    
    # 統計情報をプロット上に表示
    stats_text = f'Total pixels: {len(valid_data):,}\n'
    stats_text += f'Mean: {np.mean(valid_data):.1f} G\n'
    stats_text += f'Std: {np.std(valid_data):.1f} G'
    
    # 平均と標準偏差の位置に縦線を追加
    mean = np.mean(valid_data)
    std = np.std(valid_data)
    
    # 平均値の縦線（緑）
    ax.axvline(x=mean, color='green', linestyle='--', linewidth=1.5, alpha=0.8,
               label=f'Mean ({mean:.1f} G)')
    
    # 平均±3σの縦線（オレンジ）
    ax.axvline(x=mean + 3*std, color='orange', linestyle=':', linewidth=1.5, alpha=0.8,
               label=f'Mean+3σ ({mean+3*std:.1f} G)')
    ax.axvline(x=mean - 3*std, color='orange', linestyle=':', linewidth=1.5, alpha=0.8, label=f'Mean-3σ ({mean-3*std:.1f} G)')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 磁束バランスの表示
    if len(positive_data) > 0 and len(negative_data) > 0:
        total_pos_flux = np.sum(positive_data)
        total_neg_flux = np.sum(np.abs(negative_data))
        flux_imbalance = (total_pos_flux - total_neg_flux) / (total_pos_flux + total_neg_flux) * 100
        
        flux_text = f'Flux imbalance: {flux_imbalance:.1f}%'
        ax.text(0.98, 0.98, flux_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Y軸を対数スケールにするオプション（必要に応じて）
    ax.set_yscale('log')


# plot_hmi_quick関数内のaxes[0][1]部分を以下のように修正：
def add_histogram_to_plot(fig, gs, data, ar_region_bounds):
    """
    plot_hmi_quick関数に組み込むためのヒストグラム追加関数
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        図オブジェクト
    gs : matplotlib.gridspec.GridSpec
        GridSpecオブジェクト
    data : numpy.ndarray
        HMI磁場データ
    ar_region_bounds : tuple
        (x_min, x_max, y_min, y_max) 活動領域の境界
    """
    # 活動領域のデータを抽出
    x_min, x_max, y_min, y_max = ar_region_bounds
    ar_data = extract_ar_region_data(data, x_min, x_max, y_min, y_max)
    
    # axes[0][1]の位置にヒストグラムを作成
    ax2 = fig.add_subplot(gs[0, 1])
    plot_magnetic_field_histogram(ax2, ar_data, "AR 13030, 13032")
    
    return ax2

def plot_hmi_quick(hmi_data, downsample=1):
    """
    HMI磁場マップの高速プロット（完全手動レイアウトで表示と保存を一致）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u
    from matplotlib.gridspec import GridSpec
    
    print(f"\n=== HMI磁場マップ（完全手動レイアウト版） ===")
    
    data = hmi_data['data']
    time_str = hmi_data['time']
    hmi_map = hmi_data.get('sunpy_map')
    
    if hmi_map is None:
        print("エラー: SunPy Mapオブジェクトが必要です")
        return
        
    # --- プロット設定 (変更なし) ---
    ny, nx = data.shape
    center_x, center_y = nx // 2, ny // 2
    x_min_pix, x_max_pix = center_x - 512, center_x + 0
    y_min_pix, y_max_pix = center_y - 100, center_y + 512
    # CME発生位置：N21E44 (https://kauai.ccmc.gsfc.nasa.gov/DONKI/view/CME/20476/1)
    target_lat, target_lon = 21, -44
    
    # --- データ抽出 (変更なし) ---
    x_lims_pix = (x_min_pix, x_max_pix)
    y_lims_pix = (y_min_pix, y_max_pix)
    profile_data = extract_data_along_solar_lines(hmi_map, target_lat, target_lon, x_lims_pix, y_lims_pix)
    
    # --- プロット描画 ---
    
    # <<< 修正点 1: constrained_layout=True を削除 >>>
    fig = plt.figure(figsize=(15, 14))
    
    # suptitleはGridSpecの手動設定と競合しにくいため維持
    fig.suptitle(f'HMI Magnetogram (Profile in Zoomed Region)  {time_str}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # <<< 修正点 2: GridSpecで全ての余白を手動で厳密に設定 >>>
    # wspace/hspaceを小さくしてプロット間の隙間を詰める
    gs = GridSpec(2, 2, figure=fig, 
                  left=0.08, right=0.95, bottom=0.08, top=0.92,
                  wspace=0.2, hspace=0.15)
    
    # --- 1. 磁場マップ（WCS座標系） ---
    ax1 = fig.add_subplot(gs[0, 0], projection=hmi_map.wcs)
    im1 = ax1.imshow(data, cmap='RdBu_r', origin='lower', vmin=-200, vmax=200)
    cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', pad=0.12, shrink=0.8) 
    cbar1.ax.set_ylabel('$B_r$ (Gauss)', fontsize=12)
    ax1.set_title('Radial Magnetic Field', fontsize=14)
    ax1.set_xlabel('Solar X (arcsec)', fontsize=12)
    ax1.set_ylabel('Solar Y (arcsec)', fontsize=12)
    ax1.set_xlim(x_lims_pix); ax1.set_ylim(y_lims_pix)
    
    # --- 描画したい長方形の範囲を定義 (arcsec単位) ---
    AR_region_x_min, AR_region_x_max = center_x - 350, center_x - 200
    AR_region_y_min, AR_region_y_max = center_y + 100, center_y + 250
    
    # <<< 修正点: plt.Rectangle の代わりに ax.plot() を使用 >>>
    # 長方形の頂点のX、Y座標のリストを作成
    rect_x = [AR_region_x_min, AR_region_x_max, AR_region_x_max, AR_region_x_min, AR_region_x_min]
    rect_y = [AR_region_y_min, AR_region_y_min, AR_region_y_max, AR_region_y_max, AR_region_y_min]
    
    # ax.plotで長方形の枠線を描画する
    # transform=ax1.get_transform('world') で座標がarcsec単位であることを明示
    ax1.plot(rect_x, rect_y,
             color='green', linestyle='--', alpha=0.9, linewidth=2, label='AR Region\n(13030, 13032)')
    
    draw_hmi_solar_grid(hmi_map, ax1)
    draw_solar_coordinate_lines(hmi_map, ax1, target_lat, target_lon)
    ax1.legend(loc='upper right')
    
        # --- 2. 磁場強度ヒストグラム（活動領域内） ---
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 活動領域のデータを抽出
    ar_data = extract_ar_region_data(data, AR_region_x_min, AR_region_x_max, 
                                    AR_region_y_min, AR_region_y_max)
    # ヒストグラムをプロット
    plot_magnetic_field_histogram(ax2, ar_data, "AR 13030, 13032")
    
    
    # --- 3. 経度線プロファイル (変更なし) ---
    ax3 = fig.add_subplot(gs[1, 0])
    lon_profile = profile_data.get('lon_profile')
    if lon_profile and len(lon_profile.get('values', [])) > 0:
        coords = lon_profile['lat_coords']
        values = lon_profile['values']
        if len(coords) > 0: # Check if coords is not empty
            valid_mask = np.isfinite(values)
            if valid_mask.sum() > 0:
                profile_mean = np.mean(values[valid_mask])
                profile_std = np.std(values[valid_mask])
                label = f'Ave±std: {profile_mean:.1f} ± {profile_std:.1f} G'
                ax3.plot(coords, values, '-', color='cyan', linewidth=1.5, label=label)
                # target_latの位置に縦線を引き，その位置の値をプロット・テキストで記入する
                # coords（緯度配列）からtarget_latに最も近いインデックスを取得
                idx_nearest = np.argmin(np.abs(np.array(coords) - target_lat))
                value_at_target = values[idx_nearest]
                lat_at_target = coords[idx_nearest]
                # 縦線を描画
                ax3.fill_between(coords, profile_mean - profile_std, profile_mean + profile_std,
                                 color='gray', alpha=0.2)
                ax3.axhline(y=profile_mean, color='cyan', linestyle='--', alpha=0.8)
                ax3.set_xlim(np.min(coords), np.max(coords))
    # peak値（最大値）のインデックスを取得
    if lon_profile and len(lon_profile.get('values', [])) > 0 and len(coords) > 0:
        peak_idx = np.nanargmax(values)
        peak_lat = coords[peak_idx]
        peak_value = values[peak_idx]
        # peak点をマーカーで強調
        ax3.plot(peak_lat, peak_value, 's', color='red', markersize=10, label=f'Peak={peak_value:.1f} G (Lon={peak_lat:.2f}°)')
        # 縦線を描画
        ax3.axvline(x=peak_lat, color='red', linestyle=':', linewidth=1.5, alpha=0.8)

    ax3.set_title(f'Profile along Lon = {target_lon}°', fontsize=14)
    ax3.set_xlabel('Latitude (°)', fontsize=12)
    ax3.set_ylabel('$B_r$ (Gauss)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right')

    # --- 4. 緯度線プロファイル (変更なし) ---
    ax4 = fig.add_subplot(gs[1, 1])
    lat_profile = profile_data.get('lat_profile')
    if lat_profile and len(lat_profile.get('values', [])) > 0:
        coords = lat_profile['lon_coords']
        values = lat_profile['values']
        if len(coords) > 0: # Check if coords is not empty
            valid_mask = np.isfinite(values)
            if valid_mask.sum() > 0:
                profile_mean = np.mean(values[valid_mask])
                profile_std = np.std(values[valid_mask])
                label = f'Ave±std: {profile_mean:.1f} ± {profile_std:.1f} G'
                ax4.plot(coords, values, '-', color='magenta', linewidth=1.5, label=label)
                # ax3と同様に，target_lonの位置に縦線と値をプロットする
                # coords（経度配列）からtarget_lonに最も近いインデックスを取得
                idx_nearest = np.argmin(np.abs(np.array(coords) - target_lon))
                value_at_target = values[idx_nearest]
                lon_at_target = coords[idx_nearest]
                ax4.fill_between(coords, profile_mean - profile_std, profile_mean + profile_std,
                                 color='gray', alpha=0.2)
                ax4.axhline(y=profile_mean, color='magenta', linestyle='--', alpha=0.8)
                ax4.set_xlim(np.min(coords), np.max(coords))
                
    # 緯度線プロファイルでもピーク値（最大値）のインデックスを取得し、強調表示
    if lat_profile and len(lat_profile.get('values', [])) > 0 and len(coords) > 0:
        peak_idx_lat = np.nanargmax(values)
        peak_lon = coords[peak_idx_lat]
        peak_value_lat = values[peak_idx_lat]
        # peak点をマーカーで強調
        ax4.plot(peak_lon, peak_value_lat, 's', color='red', markersize=10, label=f'Peak={peak_value_lat:.1f} G (Lon={peak_lon:.2f}°)')
        ax4.axvline(x=peak_lon, color='red', linestyle=':', linewidth=1.5, alpha=0.8)

    ax4.set_title(f'Profile along Lat = {target_lat}°', fontsize=14)
    ax4.set_xlabel('Longitude (°)', fontsize=12)
    ax4.set_ylabel('$B_r$ (Gauss)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.legend(loc='upper right')
    
    # --- 保存 ---
    save_dir = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI"
    filename = f"hmi_latlon_profile_view_{time_str.replace(':', '').replace('.', '').replace('-', '')}.png"
    full_path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)
    
    # <<< 修正点 3: bbox_inches='tight' を再度有効化 >>>
    plt.savefig(full_path, dpi=300, bbox_inches='tight') 
    print(f"✓ プロット保存: {full_path}")
    
    plt.show()
    
        
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