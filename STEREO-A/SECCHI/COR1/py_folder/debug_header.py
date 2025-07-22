#!/usr/bin/env python3
"""
ヘッダー情報の転送をデバッグ
"""

from astropy.io import fits
from cor_prep import CORPrep
import os

# テストファイル
test_file = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/20220613_032136_n4c1A.fts"

print("=== ヘッダー情報の転送デバッグ ===")

# 元のファイルのヘッダー
print("【元のヘッダー】")
with fits.open(test_file) as hdul:
    original_header = hdul[0].header
    print("重要な情報:")
    important_keys = ['DATE-OBS', 'DATE', 'EXPTIME', 'FILENAME', 'DETECTOR', 'INSTRUME', 'RSUN']
    for key in important_keys:
        if key in original_header:
            print(f"  {key}: {original_header[key]}")

print("\n【CORPrep処理後のヘッダー】")
# CORPrep処理を実行
prep = CORPrep(silent=True)
processed_image, processed_header = prep.cor_prep(filepath=test_file)

if processed_header:
    print("重要な情報:")
    for key in important_keys:
        if key in processed_header:
            print(f"  {key}: {processed_header[key]}")
        else:
            print(f"  {key}: ❌ 失われました")
    
    print(f"\n処理済みヘッダーの総キー数: {len(processed_header)}")
    print("すべてのキー:")
    for key in sorted(processed_header.keys()):
        print(f"  {key}: {processed_header[key]}")
else:
    print("❌ ヘッダーが None です")

print(f"\nデータ形状: {processed_image.shape if processed_image is not None else 'None'}")
print(f"データ範囲: {processed_image.min():.2f} - {processed_image.max():.2f}" if processed_image is not None else "None")