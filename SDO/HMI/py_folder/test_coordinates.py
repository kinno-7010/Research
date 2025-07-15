#!/usr/bin/env python3
"""
座標系のテスト
"""

import numpy as np

def test_coordinates():
    """
    座標変換のテスト
    """
    print("=== 座標系テスト ===")
    
    # サンプルデータサイズ
    ny, nx = 256, 256  # ダウンサンプリング後のサイズ
    center_y, center_x = ny // 2, nx // 2
    
    print(f"データサイズ: {ny} x {nx}")
    print(f"太陽中心: ({center_x}, {center_y})")
    
    # 太陽中心からの座標指定
    x_from_center = 40   # 太陽中心からX方向に40ピクセル
    y_from_center = -60  # 太陽中心からY方向に-60ピクセル
    
    # 実際の配列インデックスに変換
    center_col = x_from_center + center_x
    center_row = y_from_center + center_y
    
    print(f"\n太陽中心からの座標指定:")
    print(f"  X方向: {x_from_center} ピクセル")
    print(f"  Y方向: {y_from_center} ピクセル")
    
    print(f"\n配列インデックス:")
    print(f"  center_col: {center_col}")
    print(f"  center_row: {center_row}")
    
    # 座標範囲のチェック
    extent = [-center_x, nx-center_x, -center_y, ny-center_y]
    print(f"\n表示範囲 (extent):")
    print(f"  X範囲: {extent[0]} ～ {extent[1]}")
    print(f"  Y範囲: {extent[2]} ～ {extent[3]}")
    
    # プロファイル座標の生成
    x_coords = np.arange(nx) - center_x
    y_coords = np.arange(ny) - center_y
    
    print(f"\nプロファイル座標範囲:")
    print(f"  X座標: {x_coords[0]} ～ {x_coords[-1]}")
    print(f"  Y座標: {y_coords[0]} ～ {y_coords[-1]}")
    
    print(f"\n指定位置での座標値:")
    print(f"  X={x_from_center}でのインデックス: {center_col}")
    print(f"  Y={y_from_center}でのインデックス: {center_row}")
    
    print("✓ 座標系テスト完了")

if __name__ == "__main__":
    test_coordinates()