#!/usr/bin/env python3
"""
HMI磁場データの最小限プロット（超高速版）
時刻: 2022-06-13T03:00:00
"""

import numpy as np
import matplotlib.pyplot as plt
import os

try:
    from astropy.io import fits
    ASTROPY_AVAILABLE = True
except ImportError:
    print("astropy パッケージが必要です: pip install astropy")
    ASTROPY_AVAILABLE = False

def plot_hmi_minimal():
    """
    最小限のHMI磁場プロット
    """
    print("HMI磁場データ最小限プロット")
    print("=" * 30)
    
    # HMI FITSファイルのパス
    hmi_file = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    
    if not os.path.exists(hmi_file):
        print(f"エラー: ファイルが見つかりません")
        return
    
    if not ASTROPY_AVAILABLE:
        print("astropy パッケージが必要です")
        return
    
    try:
        print("データ読み込み中...")
        with fits.open(hmi_file) as hdul:
            # データ取得
            data = hdul[1].data if len(hdul) > 1 else hdul[0].data
            header = hdul[1].header if len(hdul) > 1 else hdul[0].header
            
            # 大幅ダウンサンプリング（超高速化）
            step = 8  # 8x8ピクセルごとに1点
            data_small = data[::step, ::step]
            
            print(f"元サイズ: {data.shape}")
            print(f"表示サイズ: {data_small.shape}")
            print(f"観測時刻: {header.get('T_OBS', 'Unknown')}")
            
            # 統計
            valid = data_small[np.isfinite(data_small)]
            if len(valid) > 0:
                vmin, vmax = np.percentile(valid, [5, 95])  # 5-95パーセンタイル
                print(f"磁場範囲: {np.min(valid):.1f} ～ {np.max(valid):.1f} Gauss")
            else:
                vmin, vmax = -100, 100
            
            # シンプルプロット
            plt.figure(figsize=(10, 8))
            plt.imshow(data_small, cmap='RdBu_r', origin='lower', 
                      vmin=vmin, vmax=vmax, aspect='auto')
            plt.colorbar(label='Br (Gauss)')
            plt.title(f'HMI Magnetogram\n{header.get("T_OBS", "2022-06-13")}')
            plt.xlabel('Pixel X (8x downsampled)')
            plt.ylabel('Pixel Y (8x downsampled)')
            
            # 保存場所をResearch/SDO/HMIに変更
            save_dir = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI"
            filename = os.path.join(save_dir, "hmi_minimal_view.png")
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            print(f"✓ 保存: {filename}")
            
            plt.show()
            
            print("✓ 最小限プロット完了")
            
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    plot_hmi_minimal()