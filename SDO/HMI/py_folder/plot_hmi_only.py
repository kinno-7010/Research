#!/usr/bin/env python3
"""
HMI磁場データのみをプロットする専用スクリプト
時刻: 2022-06-13T03:00:00
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 現在のディレクトリをPythonパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from astropy.io import fits
    import astropy.units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    print("astropy パッケージが必要です: pip install astropy")
    ASTROPY_AVAILABLE = False

def read_hmi_magnetogram(filepath):
    """
    HMI磁場マップを読み込み
    
    Parameters:
    -----------
    filepath : str
        HMI FITSファイルのパス
        
    Returns:
    --------
    dict : 磁場データと座標情報
    """
    if not ASTROPY_AVAILABLE:
        return None
    
    print(f"HMI磁場マップを読み込み中: {os.path.basename(filepath)}")
    
    try:
        with fits.open(filepath) as hdul:
            # HMIデータは通常HDU 1にある
            if len(hdul) > 1 and hdul[1].data is not None:
                data = hdul[1].data
                header = hdul[1].header
                print(f"  HDU 1からデータを読み込み")
            else:
                data = hdul[0].data
                header = hdul[0].header
                print(f"  HDU 0からデータを読み込み")
            
            print(f"  データ形状: {data.shape}")
            print(f"  観測時刻: {header.get('T_OBS', 'Unknown')}")
            
            # NaN値の処理
            valid_data = data[np.isfinite(data)]
            if len(valid_data) > 0:
                print(f"  磁場範囲: {np.min(valid_data):.1f} ～ {np.max(valid_data):.1f} Gauss")
                print(f"  有効データ点数: {len(valid_data):,} / {data.size:,}")
            else:
                print(f"  警告: 有効データがありません")
            
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
            
            result = {
                'magnetogram': data,
                'lon': lon,
                'lat': lat,
                'header': header,
                'time': header.get('T_OBS', '2022-06-13T03:00:00'),
                'filepath': filepath,
                'instrument': header.get('INSTRUME', 'HMI'),
                'telescope': header.get('TELESCOP', 'SDO')
            }
            
            print(f"  経度範囲: {lon[0]:.1f} ～ {lon[-1]:.1f} 度")
            print(f"  緯度範囲: {lat[0]:.1f} ～ {lat[-1]:.1f} 度")
            print(f"  観測装置: {result['telescope']}/{result['instrument']}")
            
            return result
            
    except Exception as e:
        print(f"FITSファイル読み込みエラー: {e}")
        return None

def plot_hmi_magnetogram(hmi_data, save_plots=True):
    """
    HMI磁場マップを様々な形式でプロット
    
    Parameters:
    -----------
    hmi_data : dict
        read_hmi_magnetogram()の結果
    save_plots : bool
        プロットを保存するかどうか
    """
    print("\n=== HMI磁場マップの可視化 ===")
    
    data = hmi_data['magnetogram']
    lon = hmi_data['lon']
    lat = hmi_data['lat']
    time_str = hmi_data['time']
    
    # NaN値をマスク
    data_masked = np.ma.masked_invalid(data)
    
    # 統計情報
    valid_data = data[np.isfinite(data)]
    data_min = np.min(valid_data) if len(valid_data) > 0 else 0
    data_max = np.max(valid_data) if len(valid_data) > 0 else 0
    data_std = np.std(valid_data) if len(valid_data) > 0 else 0
    
    print(f"磁場統計:")
    print(f"  範囲: {data_min:.1f} ～ {data_max:.1f} Gauss")
    print(f"  標準偏差: {data_std:.1f} Gauss")
    print(f"  有効ピクセル数: {len(valid_data):,}")
    
    # 複数のプロットを作成
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'HMI Magnetogram Analysis\n{time_str}', fontsize=16, fontweight='bold')
    
    # 1. 基本磁場マップ
    ax1 = plt.subplot(3, 3, 1)
    extent = [lon[0], lon[-1], lat[0], lat[-1]]
    
    # カラーマップの範囲を調整
    vmax = min(abs(data_max), abs(data_min), 500)  # 強すぎる磁場をクリップ
    
    im1 = ax1.imshow(data_masked, extent=extent, cmap='RdBu_r', 
                     origin='lower', aspect='auto', vmin=-vmax, vmax=vmax)
    ax1.set_title('Radial Magnetic Field')
    ax1.set_xlabel('Longitude (deg)')
    ax1.set_ylabel('Latitude (deg)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax1, label='Br (Gauss)', shrink=0.8)
    
    # 2. 磁場強度分布
    ax2 = plt.subplot(3, 3, 2)
    magnitude = np.abs(data_masked)
    im2 = ax2.imshow(magnitude, extent=extent, cmap='plasma', 
                     origin='lower', aspect='auto', vmax=vmax)
    ax2.set_title('Magnetic Field Strength')
    ax2.set_xlabel('Longitude (deg)')
    ax2.set_ylabel('Latitude (deg)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax2, label='|Br| (Gauss)', shrink=0.8)
    
    # 3. 強磁場領域のマーク
    ax3 = plt.subplot(3, 3, 3)
    threshold = 3 * data_std  # 3σ以上を強磁場とする
    strong_field = np.abs(data_masked) > threshold
    
    ax3.imshow(data_masked, extent=extent, cmap='RdBu_r', 
               origin='lower', aspect='auto', vmin=-vmax, vmax=vmax, alpha=0.7)
    ax3.contour(strong_field, extent=extent, levels=[0.5], colors='black', linewidths=2)
    ax3.set_title(f'Strong Field Regions (|B| > {threshold:.0f}G)')
    ax3.set_xlabel('Longitude (deg)')
    ax3.set_ylabel('Latitude (deg)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 緯度プロファイル（赤道付近）
    ax4 = plt.subplot(3, 3, 4)
    eq_idx = len(lat) // 2  # 赤道付近のインデックス
    eq_range = slice(max(0, eq_idx-5), min(len(lat), eq_idx+6))
    
    lon_profile = np.nanmean(data[eq_range, :], axis=0)
    ax4.plot(lon, lon_profile, 'b-', linewidth=2)
    ax4.set_title(f'Longitude Profile (Equatorial ±{5*abs(lat[1]-lat[0]):.1f}°)')
    ax4.set_xlabel('Longitude (deg)')
    ax4.set_ylabel('Br (Gauss)')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 5. 経度プロファイル（中央子午線）
    ax5 = plt.subplot(3, 3, 5)
    central_lon_idx = len(lon) // 2
    lon_range = slice(max(0, central_lon_idx-5), min(len(lon), central_lon_idx+6))
    
    lat_profile = np.nanmean(data[:, lon_range], axis=1)
    ax5.plot(lat, lat_profile, 'r-', linewidth=2)
    ax5.set_title(f'Latitude Profile (Central ±{5*abs(lon[1]-lon[0]):.1f}°)')
    ax5.set_xlabel('Latitude (deg)')
    ax5.set_ylabel('Br (Gauss)')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 6. ヒストグラム
    ax6 = plt.subplot(3, 3, 6)
    hist_range = (-3*data_std, 3*data_std)
    ax6.hist(valid_data, bins=100, range=hist_range, alpha=0.7, color='blue', density=True)
    ax6.set_title('Magnetic Field Distribution')
    ax6.set_xlabel('Br (Gauss)')
    ax6.set_ylabel('Probability Density')
    ax6.grid(True, alpha=0.3)
    ax6.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # 7. 活動領域の検出
    ax7 = plt.subplot(3, 3, 7)
    
    # 活動領域の検出（簡易版）
    from scipy import ndimage
    
    # 強磁場領域をラベリング
    strong_positive = data_masked > 2 * data_std
    strong_negative = data_masked < -2 * data_std
    
    labeled_pos, num_pos = ndimage.label(strong_positive)
    labeled_neg, num_neg = ndimage.label(strong_negative)
    
    ax7.imshow(data_masked, extent=extent, cmap='RdBu_r', 
               origin='lower', aspect='auto', vmin=-vmax, vmax=vmax, alpha=0.6)
    
    # 活動領域を囲む
    ax7.contour(labeled_pos, extent=extent, levels=range(1, num_pos+1), 
                colors='red', linewidths=1, alpha=0.8)
    ax7.contour(labeled_neg, extent=extent, levels=range(1, num_neg+1), 
                colors='blue', linewidths=1, alpha=0.8)
    
    ax7.set_title(f'Active Regions (Pos:{num_pos}, Neg:{num_neg})')
    ax7.set_xlabel('Longitude (deg)')
    ax7.set_ylabel('Latitude (deg)')
    ax7.grid(True, alpha=0.3)
    
    # 8. 磁束分布
    ax8 = plt.subplot(3, 3, 8)
    
    # 緯度帯別の磁束
    lat_bands = np.linspace(-90, 90, 19)  # 10度刻み
    flux_positive = []
    flux_negative = []
    
    for i in range(len(lat_bands)-1):
        lat_mask = (lat >= lat_bands[i]) & (lat < lat_bands[i+1])
        band_data = data[lat_mask, :]
        
        pos_flux = np.sum(band_data[band_data > 0])
        neg_flux = np.sum(band_data[band_data < 0])
        
        flux_positive.append(pos_flux)
        flux_negative.append(abs(neg_flux))
    
    lat_centers = (lat_bands[:-1] + lat_bands[1:]) / 2
    
    ax8.bar(lat_centers, flux_positive, width=8, alpha=0.7, color='red', 
            label='Positive Flux')
    ax8.bar(lat_centers, flux_negative, width=8, alpha=0.7, color='blue', 
            label='Negative Flux')
    
    ax8.set_title('Magnetic Flux by Latitude')
    ax8.set_xlabel('Latitude (deg)')
    ax8.set_ylabel('Total |Flux| (G·pixel)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. データ統計
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    stats_text = f"""
HMI Magnetogram Statistics
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Observation Time: {time_str}
Instrument: {hmi_data['telescope']}/{hmi_data['instrument']}

Data Properties:
• Image size: {data.shape[1]} × {data.shape[0]} pixels
• Valid pixels: {len(valid_data):,} ({100*len(valid_data)/data.size:.1f}%)
• NaN/Invalid: {data.size - len(valid_data):,} pixels

Magnetic Field:
• Range: {data_min:.1f} to {data_max:.1f} Gauss
• Mean: {np.mean(valid_data):.1f} ± {data_std:.1f} Gauss
• RMS: {np.sqrt(np.mean(valid_data**2)):.1f} Gauss

Coordinate Coverage:
• Longitude: {lon[0]:.1f}° to {lon[-1]:.1f}°
• Latitude: {lat[0]:.1f}° to {lat[-1]:.1f}°
• Pixel scale: {abs(lon[1]-lon[0]):.3f}°/pixel

Active Regions:
• Positive flux regions: {num_pos}
• Negative flux regions: {num_neg}
• Strong field threshold: ±{threshold:.0f} Gauss

Total Magnetic Flux:
• Positive: {np.sum(valid_data[valid_data > 0]):.2e} G·pixel
• Negative: {np.sum(valid_data[valid_data < 0]):.2e} G·pixel
• Net flux: {np.sum(valid_data):.2e} G·pixel
"""
    
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=10, 
             fontfamily='monospace', verticalalignment='top')
    
    plt.tight_layout()
    
    if save_plots:
        # 保存場所をResearch/SDO/HMIに変更
        save_dir = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI"
        filename = f"hmi_magnetogram_analysis_{time_str.replace(':', '').replace('.', '').replace('-', '')}.png"
        full_path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(full_path, dpi=200, bbox_inches='tight')
        print(f"✓ プロット保存: {full_path}")
    
    plt.show()
    
    return {
        'data_stats': {
            'min': data_min,
            'max': data_max,
            'std': data_std,
            'mean': np.mean(valid_data),
            'rms': np.sqrt(np.mean(valid_data**2)),
            'valid_pixels': len(valid_data),
            'total_pixels': data.size
        },
        'active_regions': {
            'positive': num_pos,
            'negative': num_neg,
            'threshold': threshold
        },
        'coordinate_info': {
            'lon_range': (lon[0], lon[-1]),
            'lat_range': (lat[0], lat[-1]),
            'pixel_scale_lon': abs(lon[1]-lon[0]),
            'pixel_scale_lat': abs(lat[1]-lat[0])
        }
    }

def main():
    """
    メイン実行関数
    """
    print("HMI磁場マップ専用プロットツール")
    print("対象時刻: 2022-06-13T03:00:00")
    print("=" * 50)
    
    # HMI FITSファイルのパス
    hmi_file = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    
    print(f"対象ファイル: {os.path.basename(hmi_file)}")
    
    if not os.path.exists(hmi_file):
        print(f"エラー: ファイルが見つかりません: {hmi_file}")
        print("ファイルパスを確認してください。")
        return
    
    # 必要なパッケージのチェック
    if not ASTROPY_AVAILABLE:
        print("astropy パッケージをインストールしてください: pip install astropy")
        return
    
    try:
        # 1. HMI磁場マップの読み込み
        hmi_data = read_hmi_magnetogram(hmi_file)
        if hmi_data is None:
            print("HMI磁場マップの読み込みに失敗しました")
            return
        
        # 2. 磁場マップの可視化
        plot_results = plot_hmi_magnetogram(hmi_data, save_plots=True)
        
        print("\n=== 解析完了 ===")
        print("✓ HMI磁場マップの詳細解析とプロットが完了しました")
        
        # 結果の要約表示
        stats = plot_results['data_stats']
        ar_info = plot_results['active_regions']
        
        print(f"\n主要統計:")
        print(f"  磁場範囲: {stats['min']:.1f} ～ {stats['max']:.1f} Gauss")
        print(f"  RMS磁場: {stats['rms']:.1f} Gauss")
        print(f"  活動領域数: 正極性={ar_info['positive']}, 負極性={ar_info['negative']}")
        print(f"  有効ピクセル率: {100*stats['valid_pixels']/stats['total_pixels']:.1f}%")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()