#!/usr/bin/env python3
"""
HMI FITSファイルの構造を確認
"""

import os
try:
    from astropy.io import fits
    import numpy as np
    ASTROPY_AVAILABLE = True
except ImportError:
    print("astropy パッケージが必要です: pip install astropy")
    ASTROPY_AVAILABLE = False

def check_fits_structure(filepath):
    """
    FITSファイルの構造を詳細に確認
    """
    if not ASTROPY_AVAILABLE:
        return
    
    print(f"FITSファイル構造の確認: {filepath}")
    print("=" * 60)
    
    if not os.path.exists(filepath):
        print(f"エラー: ファイルが見つかりません: {filepath}")
        return
    
    try:
        with fits.open(filepath) as hdul:
            print(f"HDU数: {len(hdul)}")
            print()
            
            for i, hdu in enumerate(hdul):
                print(f"HDU {i}: {type(hdu).__name__}")
                print(f"  ヘッダー項目数: {len(hdu.header)}")
                
                if hasattr(hdu, 'data') and hdu.data is not None:
                    print(f"  データ形状: {hdu.data.shape}")
                    print(f"  データ型: {hdu.data.dtype}")
                    
                    if hdu.data.size > 0:
                        valid_data = hdu.data[np.isfinite(hdu.data)]
                        if len(valid_data) > 0:
                            print(f"  データ範囲: {np.min(valid_data):.2e} ～ {np.max(valid_data):.2e}")
                            print(f"  有効データ点数: {len(valid_data)} / {hdu.data.size}")
                        else:
                            print(f"  有効データなし（全てNaN）")
                else:
                    print(f"  データなし")
                
                # 重要なヘッダー情報
                important_keys = ['TELESCOP', 'INSTRUME', 'T_OBS', 'DATE-OBS', 
                                'NAXIS1', 'NAXIS2', 'CDELT1', 'CDELT2', 
                                'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                                'BUNIT', 'BZERO', 'BSCALE']
                
                header_info = {}
                for key in important_keys:
                    if key in hdu.header:
                        header_info[key] = hdu.header[key]
                
                if header_info:
                    print(f"  重要なヘッダー:")
                    for key, value in header_info.items():
                        print(f"    {key}: {value}")
                
                print()
                
                # 最初のHDUのヘッダーを詳細表示
                if i == 0:
                    print(f"  全ヘッダー情報:")
                    for key in hdu.header:
                        try:
                            value = hdu.header[key]
                            comment = hdu.header.comments[key]
                            print(f"    {key:8s} = {str(value):20s} / {comment}")
                        except:
                            pass
                    print()
    
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    メイン実行
    """
    hmi_file = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    check_fits_structure(hmi_file)

if __name__ == "__main__":
    main()