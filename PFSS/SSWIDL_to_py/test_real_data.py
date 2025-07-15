#!/usr/bin/env python3
"""
実観測データのテスト
"""

import sys
import os

# 現在のディレクトリをPythonパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_real_data_processing():
    """
    実観測データ処理のテスト
    """
    print("=== 実観測データ処理テスト ===")
    
    # HMI FITSファイルのパス
    hmi_file = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    
    if not os.path.exists(hmi_file):
        print(f"✗ HMI FITSファイルが見つかりません: {hmi_file}")
        return False
    
    try:
        # 1. HMI FITS読み込みテスト
        print("\n1. HMI FITSファイル読み込みテスト")
        from hmi_pfss_fieldlines import read_hmi_fits
        
        hmi_data = read_hmi_fits(hmi_file)
        if hmi_data is None:
            print("✗ HMI データの読み込みに失敗")
            return False
        
        print("✓ HMI データ読み込み成功")
        
        # 2. PFSS計算テスト（低解像度）
        print("\n2. PFSS磁場計算テスト（低解像度）")
        from hmi_pfss_fieldlines import create_pfss_from_hmi
        
        # 低解像度で高速計算
        pfss_data = create_pfss_from_hmi(hmi_data, nr=15, rss=2.5)
        
        print("✓ PFSS計算成功")
        print(f"  グリッドサイズ: {pfss_data['nlon']}×{pfss_data['nlat']}×{pfss_data['nr']}")
        
        # 3. SSWIDL形式変換テスト
        print("\n3. SSWIDL形式変換テスト")
        from working_fieldlines import convert_pfss_to_sswidl_format
        
        sph_data = convert_pfss_to_sswidl_format(pfss_data)
        if sph_data is None:
            print("✗ SSWIDL形式変換に失敗")
            return False
        
        print("✓ SSWIDL形式変換成功")
        
        # 4. 可視化テスト
        print("\n4. データ可視化テスト")
        from hmi_pfss_fieldlines import visualize_hmi_and_pfss
        
        visualize_hmi_and_pfss(hmi_data, pfss_data)
        print("✓ 可視化完了")
        
        print("\n=== 全テスト成功 ===")
        print("実観測データを使用した磁力線描画の準備ができました")
        return True
        
    except Exception as e:
        print(f"✗ エラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_data_processing()
    sys.exit(0 if success else 1)