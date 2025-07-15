#!/usr/bin/env python3
"""
簡単なPFSSデータ取得スクリプト
2022-06-13T03:00:00のデータを例として使用
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import urllib.request
import os

def download_sample_hmi_data():
    """
    HMIサンプルデータの取得例
    """
    print("=== HMI Synoptic Magnetogram データ取得例 ===")
    
    # 実際のHMIデータURL（例）
    # 注意: 実際のJSOCからのダウンロードは登録とエクスポート手続きが必要
    sample_urls = [
        "http://jsoc.stanford.edu/",  # JSOC メインページ
        "http://wso.stanford.edu/synopticl.html",  # WSO データ
        "https://gong.nso.edu/data/magmap/",  # GONG データ
    ]
    
    print("実際のデータ取得先:")
    for i, url in enumerate(sample_urls, 1):
        print(f"  {i}. {url}")
    
    print("\n手順:")
    print("1. JSOC (http://jsoc.stanford.edu/) にユーザー登録")
    print("2. データシリーズ 'hmi.synoptic_mr_720s' を検索")
    print("3. 時刻 '2022.06.13_00:00:00_TAI' を指定")
    print("4. FITS形式でエクスポート要求")
    print("5. ダウンロードリンクがメールで送信される")

def create_sample_pfss_data():
    """
    PFSSデータ構造のサンプルを作成
    実際のHMIデータの代わりに使用
    """
    print("\n=== サンプルPFSSデータ作成 ===")
    
    # グリッド設定（実際のPFSSデータと同じ構造）
    nr, nlat, nlon = 35, 48, 96
    
    # 座標配列
    rix = np.linspace(1.0, 2.5, nr)          # 1.0-2.5 太陽半径
    lat = np.linspace(-90, 90, nlat)          # 緯度 -90～90度
    lon = np.linspace(0, 360, nlon, endpoint=False)  # 経度 0～360度
    
    # 角度変換
    theta = (90 - lat) * np.pi / 180  # colatitude
    phi = lon * np.pi / 180           # longitude
    
    # メッシュグリッド作成
    LON, LAT, R = np.meshgrid(lon, lat, rix, indexing='ij')
    THETA = (90 - LAT) * np.pi / 180
    PHI = LON * np.pi / 180
    
    print(f"グリッドサイズ: {nlon}×{nlat}×{nr}")
    print(f"動径範囲: {rix[0]:.1f} - {rix[-1]:.1f} Rs")
    
    # 2022年6月13日の活動状況を模擬した磁場
    print("2022年6月13日の太陽活動を模擬中...")
    
    # 基本双極子磁場
    br = 2.0 * np.cos(THETA) / R**3
    
    # 活動領域を追加（2022年6月の実際の活動を模擬）
    active_regions = [
        # (lon_center, lat_center, strength, width)
        (60, 20, 15.0, 400),    # AR30030系（大規模活動領域）
        (180, -15, -12.0, 350), # 反対極性領域
        (300, 10, 8.0, 250),    # 高緯度活動領域
        (120, -30, -6.0, 200),  # 南半球活動領域
        (240, 35, 4.0, 150),    # 北半球小規模領域
        (15, -5, 3.0, 120),     # 新興領域
    ]
    
    for lon_c, lat_c, strength, width in active_regions:
        ar = strength * np.exp(-((LON - lon_c)**2 + (LAT - lat_c)**2) / width) / R**2
        br += ar
    
    # 高次多重極成分
    br += 1.5 * np.sin(2 * THETA) * np.cos(2 * PHI) / R**3
    br += 1.0 * np.sin(THETA)**2 * np.cos(4 * PHI) / R**3
    br += 0.7 * np.cos(3 * THETA) * np.sin(3 * PHI) / R**3
    
    # θ成分（子午線流）
    bth = np.sin(THETA) / R**3
    bth += 0.8 * np.cos(2 * THETA) * np.cos(2 * PHI) / R**3
    bth += 0.5 * np.sin(3 * THETA) * np.cos(PHI) / R**3
    
    # φ成分（差動回転）
    bph = 0.6 * np.sin(2 * PHI) * np.sin(THETA) / R**2
    bph += 0.4 * np.sin(THETA) * np.sin(4 * PHI) / R**2
    bph += 0.2 * np.cos(THETA) * np.sin(6 * PHI) / R**2
    
    # データ保存
    data = {
        'br': br,
        'bth': bth, 
        'bph': bph,
        'rix': rix,
        'lat': lat,
        'lon': lon,
        'theta': theta,
        'phi': phi,
        'time': '2022-06-13T03:00:00',
        'active_regions': active_regions
    }
    
    print(f"磁場データ作成完了:")
    print(f"  動径磁場範囲: {np.min(br):.2f} ～ {np.max(br):.2f}")
    print(f"  活動領域数: {len(active_regions)}")
    
    return data

def save_pfss_data_hdf5(data, filename='sample_pfss_20220613.h5'):
    """
    PFSSデータをHDF5形式で保存
    """
    try:
        import h5py
        
        print(f"\nHDF5ファイルに保存中: {filename}")
        
        with h5py.File(filename, 'w') as f:
            # 磁場成分
            f.create_dataset('br', data=data['br'])
            f.create_dataset('bth', data=data['bth'])
            f.create_dataset('bph', data=data['bph'])
            
            # 座標配列
            coord_group = f.create_group('coordinates')
            coord_group.create_dataset('rix', data=data['rix'])
            coord_group.create_dataset('lat', data=data['lat'])
            coord_group.create_dataset('lon', data=data['lon'])
            coord_group.create_dataset('theta', data=data['theta'])
            coord_group.create_dataset('phi', data=data['phi'])
            
            # メタデータ
            meta_group = f.create_group('metadata')
            meta_group.attrs['time'] = data['time']
            meta_group.attrs['source_surface_height'] = 2.5
            meta_group.attrs['grid_size'] = f"{len(data['lon'])}x{len(data['lat'])}x{len(data['rix'])}"
            meta_group.attrs['description'] = "Sample PFSS data for 2022-06-13T03:00:00"
        
        print(f"✓ ファイル保存完了: {filename}")
        return filename
        
    except ImportError:
        print("h5py モジュールが必要です: pip install h5py")
        return None

def visualize_sample_data(data):
    """
    サンプルデータを可視化
    """
    print("\n=== データ可視化 ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sample PFSS Data for 2022-06-13T03:00:00 UTC', fontsize=16)
    
    # 1. 太陽表面の動径磁場 (r=1.0)
    br_surface = data['br'][:, :, 0]  # 最内層
    im1 = axes[0,0].imshow(br_surface.T, extent=[0, 360, -90, 90], 
                          cmap='RdBu_r', origin='lower', aspect='auto')
    axes[0,0].set_title('Radial Field at Solar Surface (r=1.0 Rs)')
    axes[0,0].set_xlabel('Longitude (deg)')
    axes[0,0].set_ylabel('Latitude (deg)')
    axes[0,0].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[0,0], label='Br (Gauss)')
    
    # 活動領域をマーク
    for lon_c, lat_c, strength, _ in data['active_regions']:
        axes[0,0].plot(lon_c, lat_c, 'ko', markersize=6)
        axes[0,0].annotate(f'{strength:.0f}G', (lon_c, lat_c), 
                          xytext=(3, 3), textcoords='offset points',
                          fontsize=8, color='white', fontweight='bold')
    
    # 2. ソース面での動径磁場 (r=2.5)
    br_source = data['br'][:, :, -1]  # 最外層
    im2 = axes[0,1].imshow(br_source.T, extent=[0, 360, -90, 90], 
                          cmap='RdBu_r', origin='lower', aspect='auto')
    axes[0,1].set_title('Radial Field at Source Surface (r=2.5 Rs)')
    axes[0,1].set_xlabel('Longitude (deg)')
    axes[0,1].set_ylabel('Latitude (deg)')
    axes[0,1].grid(True, alpha=0.3)
    plt.colorbar(im2, ax=axes[0,1], label='Br (Gauss)')
    
    # 3. 動径プロファイル（赤道での変化）
    eq_idx = len(data['lat']) // 2  # 赤道のインデックス
    lon_idx = len(data['lon']) // 4  # 90度経度
    br_radial = data['br'][lon_idx, eq_idx, :]
    
    axes[1,0].plot(data['rix'], br_radial, 'b-', linewidth=2, marker='o')
    axes[1,0].set_title(f'Radial Profile at Equator (Lon={data["lon"][lon_idx]:.0f}°)')
    axes[1,0].set_xlabel('Radius (Rs)')
    axes[1,0].set_ylabel('Br (Gauss)')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 4. 磁場強度分布
    bmag_surface = np.sqrt(data['br'][:, :, 0]**2 + 
                          data['bth'][:, :, 0]**2 + 
                          data['bph'][:, :, 0]**2)
    
    im4 = axes[1,1].imshow(bmag_surface.T, extent=[0, 360, -90, 90], 
                          cmap='plasma', origin='lower', aspect='auto')
    axes[1,1].set_title('Magnetic Field Strength at Surface')
    axes[1,1].set_xlabel('Longitude (deg)')
    axes[1,1].set_ylabel('Latitude (deg)')
    axes[1,1].grid(True, alpha=0.3)
    plt.colorbar(im4, ax=axes[1,1], label='|B| (Gauss)')
    
    plt.tight_layout()
    plt.savefig('sample_pfss_data_20220613.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ 可視化完了 (sample_pfss_data_20220613.png として保存)")

def main():
    """
    メイン実行関数
    """
    print("PFSS磁力線描画用データファイル準備ツール")
    print("=" * 50)
    
    # 1. 実データ取得方法の説明
    download_sample_hmi_data()
    
    # 2. サンプルデータ作成
    data = create_sample_pfss_data()
    
    # 3. HDF5形式で保存
    filename = save_pfss_data_hdf5(data)
    
    # 4. データ可視化
    visualize_sample_data(data)
    
    print(f"\n=== 完了 ===")
    print(f"作成されたファイル:")
    if filename:
        print(f"  - {filename} (HDF5 PFSSデータ)")
    print(f"  - sample_pfss_data_20220613.png (可視化画像)")
    
    print(f"\n使用方法:")
    print(f"1. working_fieldlines.py でこのサンプルデータを読み込み")
    print(f"2. 実際のHMIデータが必要な場合はJSOCから取得")
    print(f"3. pfss_restore でデータを読み込んで磁力線描画")

if __name__ == "__main__":
    main()