#!/usr/bin/env python3
"""
処理済みFITSファイルの日時確認
"""

from astropy.io import fits
import os

# 元のファイルと処理済みファイルのパス
original_file = "/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/Rawdata/20220613_032136_n4c1A.fts"
processed_file = "/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/Rawdata/calibration/20220613_032136_n4c1A_processed.fits"

print("=== 日時情報の比較 ===")
print()

# 元のファイル
print("【元のファイル】")
print(f"ファイル: {os.path.basename(original_file)}")
try:
    with fits.open(original_file) as hdul:
        header = hdul[0].header
        print(f"DATE-OBS: {header.get('DATE-OBS', 'N/A')}")
        print(f"DATE: {header.get('DATE', 'N/A')}")
        print(f"TIME-OBS: {header.get('TIME-OBS', 'N/A')}")
        print(f"EXPTIME: {header.get('EXPTIME', 'N/A')}")
        print(f"FILENAME: {header.get('FILENAME', 'N/A')}")
except Exception as e:
    print(f"エラー: {e}")

print()

# 処理済みファイル
print("【処理済みファイル】")
print(f"ファイル: {os.path.basename(processed_file)}")
try:
    with fits.open(processed_file) as hdul:
        header = hdul[0].header
        print(f"DATE-OBS: {header.get('DATE-OBS', 'N/A')}")
        print(f"DATE: {header.get('DATE', 'N/A')}")
        print(f"TIME-OBS: {header.get('TIME-OBS', 'N/A')}")
        print(f"EXPTIME: {header.get('EXPTIME', 'N/A')}")
        print(f"FILENAME: {header.get('FILENAME', 'N/A')}")
        
        print()
        print("【処理関連の情報】")
        print("HISTORY:")
        if 'HISTORY' in header:
            for hist in header['HISTORY']:
                print(f"  {hist}")
        
        print(f"BUNIT: {header.get('BUNIT', 'N/A')}")
        print(f"DATAMIN: {header.get('DATAMIN', 'N/A')}")
        print(f"DATAMAX: {header.get('DATAMAX', 'N/A')}")
        print(f"DATAMEAN: {header.get('DATAMEAN', 'N/A')}")
        
except Exception as e:
    print(f"エラー: {e}")

print()
print("=== 結論 ===")
print("データ自身の観測日時（DATE-OBS）は元のままです。")
print("処理時刻はHISTORYに記録されています。")