#!/usr/bin/env python3
"""
Custom Group動画作成テストスクリプト
"""

import sys
sys.path.append('/home/kinno-7010/Research_code/MK4_coronagraph/UCOMP/py_folder')

from ucomp_config import *
from ucomp_plotting_groups import create_ucomp_animation_custom1, create_ucomp_animation_custom2
import matplotlib.pyplot as plt

def test_custom_animations():
    """カスタムグループのアニメーション作成テスト"""
    
    # デフォルト設定
    start_time = "2022-06-13T03:00:00"
    end_time = "2022-06-13T04:01:00"
    wavelength = 1074
    fps = 2
    interval = 500
    
    print("=" * 80)
    print("UCOMP Custom Groups Animation Test")
    print("=" * 80)
    print(f"時間範囲: {start_time} - {end_time}")
    print(f"波長: {wavelength} nm")
    print(f"フレームレート: {fps} fps")
    print(f"間隔: {interval} ms")
    print()
    
    # Custom Group 1 動画作成
    print("1. Custom Group 1 動画作成 (Ext 3,4,5,12)")
    print("-" * 50)
    try:
        path1 = create_ucomp_animation_custom1(start_time, end_time, wavelength, fps, interval)
        if path1:
            print(f"✓ Custom Group 1 動画作成完了: {path1}")
            print(f"  ファイルサイズ: {Path(path1).stat().st_size} bytes")
        else:
            print("✗ Custom Group 1 動画作成失敗")
    except Exception as e:
        print(f"✗ Custom Group 1 エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Custom Group 2 動画作成
    print("2. Custom Group 2 動画作成 (Ext 7-10)")
    print("-" * 50)
    try:
        path2 = create_ucomp_animation_custom2(start_time, end_time, wavelength, fps, interval)
        if path2:
            print(f"✓ Custom Group 2 動画作成完了: {path2}")
            print(f"  ファイルサイズ: {Path(path2).stat().st_size} bytes")
        else:
            print("✗ Custom Group 2 動画作成失敗")
    except Exception as e:
        print(f"✗ Custom Group 2 エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)
    print("テスト完了")
    print("=" * 80)

if __name__ == "__main__":
    test_custom_animations()