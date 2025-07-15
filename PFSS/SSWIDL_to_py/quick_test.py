#!/usr/bin/env python3
"""
クイックテスト
"""

import sys
import os
import numpy as np

# 現在のディレクトリをPythonパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def quick_test():
    """
    高速テスト
    """
    print("=== クイックテスト ===")
    
    # HMI FITSファイルのパス
    hmi_file = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    
    try:
        # 1. HMI読み込みのみテスト
        print("1. HMI読み込みテスト")
        from hmi_pfss_fieldlines import read_hmi_fits
        
        hmi_data = read_hmi_fits(hmi_file)
        if hmi_data is None:
            print("✗ 読み込み失敗")
            return False
        
        print("✓ HMI読み込み成功")
        print(f"  形状: {hmi_data['br_surface'].shape}")
        print(f"  範囲: {np.min(hmi_data['br_surface']):.1f} ～ {np.max(hmi_data['br_surface']):.1f} G")
        
        # 2. 超低解像度でPFSS計算テスト
        print("\n2. 超低解像度PFSS計算テスト")
        from hmi_pfss_fieldlines import create_pfss_from_hmi
        
        # 極端に低解像度
        pfss_data = create_pfss_from_hmi(hmi_data, nr=5, rss=2.0)
        
        print("✓ PFSS計算成功")
        print(f"  最終グリッド: {pfss_data['nlon']}×{pfss_data['nlat']}×{pfss_data['nr']}")
        print(f"  表面磁場: {np.min(pfss_data['br'][:,:,0]):.1f} ～ {np.max(pfss_data['br'][:,:,0]):.1f} G")
        
        print("\n=== クイックテスト成功 ===")
        return True
        
    except Exception as e:
        print(f"✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    quick_test()