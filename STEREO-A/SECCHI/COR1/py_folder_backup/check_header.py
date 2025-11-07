#!/usr/bin/env python3
"""
COR1 FITSファイルのヘッダー情報を確認
"""

from astropy.io import fits
import os

# テストファイルのパス
filepath = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/20220613_032136_n4c1A.fts"

try:
    with fits.open(filepath) as hdul:
        header = hdul[0].header
        
    print("=== COR1 FITSヘッダー情報 ===")
    print(f"ファイル: {os.path.basename(filepath)}")
    print("\n太陽半径関連のキーワード:")
    
    # 太陽半径関連のキーワードを検索
    solar_radius_keywords = [
        'RSUN', 'RSUN_REF', 'RSUN_OBS', 'SOLAR_R', 'R_SUN', 
        'CDELT1', 'CDELT2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
        'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2', 'DSUN_OBS', 'DSUN_REF'
    ]
    
    found_keywords = {}
    for keyword in solar_radius_keywords:
        if keyword in header:
            found_keywords[keyword] = header[keyword]
            print(f"{keyword}: {header[keyword]}")
    
    print("\n全てのヘッダーキーワード:")
    for key, value in header.items():
        if 'SUN' in key.upper() or 'SOLAR' in key.upper() or 'RADIUS' in key.upper():
            print(f"{key}: {value}")
    
    print(f"\n主要な情報:")
    print(f"観測時間: {header.get('DATE-OBS', 'Unknown')}")
    print(f"検出器: {header.get('DETECTOR', 'Unknown')}")
    print(f"データ形状: {header.get('NAXIS1', 'Unknown')} x {header.get('NAXIS2', 'Unknown')}")
    
except Exception as e:
    print(f"エラー: {e}")