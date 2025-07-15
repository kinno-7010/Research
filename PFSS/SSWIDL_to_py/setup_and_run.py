#!/usr/bin/env python3
"""
セットアップとモジュールインストール後に磁力線描画を実行
"""

import sys
import os
import subprocess

def install_packages():
    """必要なパッケージをインストール"""
    print("=== 必要なパッケージのインストール ===")
    
    packages = [
        'numpy',
        'matplotlib', 
        'scipy',
        'h5py',
        'astropy'
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} - 既にインストール済み")
        except ImportError:
            print(f"⚠ {package} - インストール中...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✓ {package} - インストール完了")
            except subprocess.CalledProcessError:
                print(f"✗ {package} - インストール失敗")

def setup_environment():
    """環境を設定"""
    print("\n=== 環境設定 ===")
    
    # 現在のディレクトリをPythonパスに追加
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"✓ Pythonパスに追加: {current_dir}")
    
    # 必要なファイルの存在確認
    required_files = [
        'spherical_field_data__define.py',
        'spherical_draw_field.py',
        'spherical_trace_field.py',
        'spherical_field_start_coord.py',
        'pfss_data_block.py'
    ]
    
    missing_files = []
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✓ {filename} - 存在")
        else:
            print(f"✗ {filename} - 不存在")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n警告: {len(missing_files)}個のファイルが見つかりません")
        print("デモモードでの実行になります")
    
    return len(missing_files) == 0

def main():
    """メイン実行"""
    print("太陽磁力線描画プログラム - セットアップ版")
    print("対象時刻: 2022-06-13T03:00:00")
    print("=" * 60)
    
    # パッケージインストール
    install_packages()
    
    # 環境設定
    files_ok = setup_environment()
    
    # working_fieldlines.pyを実行
    print("\n=== プログラム実行 ===")
    
    args = sys.argv[1:] if len(sys.argv) > 1 else ['--demo']
    
    try:
        # working_fieldlines.pyを直接実行
        exec(open('working_fieldlines.py').read())
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        
        # 最小限のデモを実行
        print("\n最小限のデモを実行します...")
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            
            # 簡単な磁場可視化
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 磁場データ
            nlat, nlon = 90, 180
            lat = np.linspace(-90, 90, nlat)
            lon = np.linspace(0, 360, nlon)
            LON, LAT = np.meshgrid(lon, lat)
            
            # 基本双極子磁場
            theta = np.radians(90 - LAT)
            br = 2 * np.cos(theta)
            
            # 活動領域を追加
            ar1 = 5 * np.exp(-((LON - 60)**2 + (LAT - 20)**2) / 300)
            ar2 = -4 * np.exp(-((LON - 180)**2 + (LAT + 15)**2) / 250)
            br += ar1 + ar2
            
            # プロット
            im = ax.imshow(br, extent=[0, 360, -90, 90], 
                          cmap='RdBu_r', origin='lower', aspect='auto')
            ax.set_title('Solar Magnetic Field Simulation\n2022-06-13T03:00:00 UTC')
            ax.set_xlabel('Longitude (deg)')
            ax.set_ylabel('Latitude (deg)')
            ax.grid(True, alpha=0.3)
            plt.colorbar(im, ax=ax, label='Br (Gauss)')
            
            plt.tight_layout()
            plt.show()
            
            print("✓ 最小限のデモ実行完了")
            
        except Exception as e2:
            print(f"最小限のデモでもエラー: {e2}")
            print("matplotlibがインストールされていない可能性があります")

if __name__ == "__main__":
    main()