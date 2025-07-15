#!/usr/bin/env python3
"""
プロファイル範囲制限のテスト
"""

import numpy as np
import matplotlib.pyplot as plt

def test_profile_range():
    """
    プロファイル範囲制限の動作テスト
    """
    print("=== プロファイル範囲制限テスト ===")
    
    # サンプルデータ生成
    ny, nx = 256, 256
    center_y, center_x = ny // 2, nx // 2
    
    # テストデータ（線形勾配）
    data = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            data[i, j] = (j - center_x) + (i - center_y) * 0.1
    
    print(f"データサイズ: {ny} x {nx}")
    print(f"太陽中心: ({center_x}, {center_y})")
    
    # 指定位置
    x_from_center = -60
    y_from_center = 40
    center_col = x_from_center + center_x
    center_row = y_from_center + center_y
    
    print(f"指定位置: X={x_from_center}, Y={y_from_center}")
    print(f"配列インデックス: col={center_col}, row={center_row}")
    
    # 制限範囲
    x_min, x_max = -125, 10
    y_min, y_max = -10, 125
    
    # 行プロファイル（制限前）
    row_profile_full = data[center_row, :]
    x_coords_full = np.arange(len(row_profile_full)) - center_x
    
    # 行プロファイル（制限後）
    x_mask = (x_coords_full >= x_min) & (x_coords_full <= x_max)
    row_profile = row_profile_full[x_mask]
    x_coords = x_coords_full[x_mask]
    
    print(f"\n行プロファイル:")
    print(f"  制限前: {len(row_profile_full)} 点")
    print(f"  制限後: {len(row_profile)} 点")
    print(f"  X範囲: {x_coords[0]:.0f} ～ {x_coords[-1]:.0f}")
    
    # 列プロファイル（制限前）
    col_profile_full = data[:, center_col]
    y_coords_full = np.arange(len(col_profile_full)) - center_y
    
    # 列プロファイル（制限後）
    y_mask = (y_coords_full >= y_min) & (y_coords_full <= y_max)
    col_profile = col_profile_full[y_mask]
    y_coords = y_coords_full[y_mask]
    
    print(f"\n列プロファイル:")
    print(f"  制限前: {len(col_profile_full)} 点")
    print(f"  制限後: {len(col_profile)} 点")
    print(f"  Y範囲: {y_coords[0]:.0f} ～ {y_coords[-1]:.0f}")
    
    # 統計比較
    print(f"\n統計比較:")
    print(f"  行プロファイル 制限前: 平均={np.mean(row_profile_full):.2f}, 標準偏差={np.std(row_profile_full):.2f}")
    print(f"  行プロファイル 制限後: 平均={np.mean(row_profile):.2f}, 標準偏差={np.std(row_profile):.2f}")
    print(f"  列プロファイル 制限前: 平均={np.mean(col_profile_full):.2f}, 標準偏差={np.std(col_profile_full):.2f}")
    print(f"  列プロファイル 制限後: 平均={np.mean(col_profile):.2f}, 標準偏差={np.std(col_profile):.2f}")
    
    print("✓ プロファイル範囲制限テスト完了")

if __name__ == "__main__":
    test_profile_range()