#!/usr/bin/env python3
"""
HMI磁場データの高速プロット（簡易版）
時刻: 2022-06-13T03:00:00
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 現在のディレクトリをPythonパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from astropy.io import fits
    ASTROPY_AVAILABLE = True
except ImportError:
    print("astropy パッケージが必要です: pip install astropy")
    ASTROPY_AVAILABLE = False

def read_hmi_quick(filepath):
    """
    HMI磁場マップを高速読み込み
    """
    if not ASTROPY_AVAILABLE:
        return None
    
    print(f"HMI磁場データを読み込み中: {os.path.basename(filepath)}")
    
    try:
        with fits.open(filepath) as hdul:
            # データの取得
            if len(hdul) > 1 and hdul[1].data is not None:
                data = hdul[1].data
                header = hdul[1].header
            else:
                data = hdul[0].data
                header = hdul[0].header
            
            print(f"  データ形状: {data.shape}")
            print(f"  観測時刻: {header.get('T_OBS', 'Unknown')}")
            
            # 統計情報
            valid_data = data[np.isfinite(data)]
            if len(valid_data) > 0:
                print(f"  磁場範囲: {np.min(valid_data):.1f} ～ {np.max(valid_data):.1f} Gauss")
            
            return {
                'data': data,
                'header': header,
                'time': header.get('T_OBS', '2022-06-13T03:00:00')
            }
            
    except Exception as e:
        print(f"読み込みエラー: {e}")
        return None

def plot_hmi_quick(hmi_data, downsample=1):
    """
    HMI磁場マップの高速プロット
    
    Parameters:
    -----------
    hmi_data : dict
        HMIデータ
    downsample : int
        ダウンサンプリング率（高速化のため）
    """
    print(f"\n=== HMI磁場マップ（高速プロット） ===")
    
    data = hmi_data['data']
    time_str = hmi_data['time']
    
    # ダウンサンプリングで高速化
    if downsample > 1:
        data_plot = data[::downsample, ::downsample]
        print(f"ダウンサンプリング: {data.shape} → {data_plot.shape}")
    else:
        data_plot = data
    
    # NaN値をマスク
    data_masked = np.ma.masked_invalid(data_plot)
    
    # 統計情報
    valid_data = data_plot[np.isfinite(data_plot)]
    if len(valid_data) == 0:
        print("有効データがありません")
        return
    
    data_min = np.min(valid_data)
    data_max = np.max(valid_data)
    data_std = np.std(valid_data)
    
    print(f"磁場統計: {data_min:.1f} ～ {data_max:.1f} Gauss (σ={data_std:.1f})")
    
    # プロット作成
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'HMI Magnetogram Quick View\n{time_str}', fontsize=14, fontweight='bold')
    
    # カラースケールの調整
    vmax = min(abs(data_max), abs(data_min), 500)
    
    # 太陽の中心を原点に合わせるための座標設定
    ny, nx = data_plot.shape
    center_y, center_x = ny // 2, nx // 2
    extent = [-center_x, nx-center_x, -center_y, ny-center_y]
    
    # プロファイル用の座標設定（太陽中心からのピクセル数で指定）
    x_from_center = -250   # 太陽中心からX方向に40ピクセル
    y_from_center = 150  # 太陽中心からY方向に-60ピクセル
    
    # 実際の配列インデックスに変換
    center_col = x_from_center + center_x
    center_row = y_from_center + center_y
    
    
    # 3. 中央行プロファイル（xlim範囲内のデータのみ使用）
    # axes[0,0]のxlim設定を取得
    x_min, x_max = -512, -50  # xlim範囲（axes[0,0]と同じ）
    
    # 全行プロファイルを取得
    row_profile_full = data_plot[center_row, :]
    x_coords_full = np.arange(len(row_profile_full)) - center_x
    
    # xlim範囲内のインデックスを特定
    x_mask = (x_coords_full >= x_min) & (x_coords_full <= x_max)
    row_profile = row_profile_full[x_mask]
    x_coords = x_coords_full[x_mask]

    # 4. 中央列プロファイル（ylim範囲内のデータのみ使用）
    y_min, y_max = -50, 512  # ylim範囲
    
    # 全列プロファイルを取得
    col_profile_full = data_plot[:, center_col]
    y_coords_full = np.arange(len(col_profile_full)) - center_y
    
    # ylim範囲内のインデックスを特定
    y_mask = (y_coords_full >= y_min) & (y_coords_full <= y_max)
    col_profile = col_profile_full[y_mask]
    y_coords = y_coords_full[y_mask]
    
    # ---------------------------------------------------------------------------------
    # プロット作成
    # 1. 磁場マップ
    im1 = axes[0,0].imshow(data_masked, cmap='RdBu_r', origin='lower', 
                           aspect='auto', vmin=-200, vmax=200, extent=extent)
    axes[0,0].set_title('Radial Magnetic Field')
    axes[0,0].set_xlabel('X (Pixel)')
    axes[0,0].set_ylabel('Y (Pixel)')
    axes[0,0].set_aspect('equal')
    axes[0,0].set_xlim(x_min, x_max)
    axes[0,0].set_ylim(y_min, y_max)
    # center_col, center_rowに対応する線を描画
    axes[0,0].axvline(x=x_from_center, color='cyan', linestyle='--', alpha=0.8, linewidth=2, label=f'X={x_from_center}(pixel)')
    axes[0,0].axhline(y=y_from_center, color='magenta', linestyle='--', alpha=0.8, linewidth=2, label=f'Y={y_from_center}(pixel)')
    axes[0,0].legend(loc='upper right')
    plt.colorbar(im1, ax=axes[0,0], label='Br (Gauss)', shrink=0.8)
    
    # 2. 磁場強度
    magnitude = np.abs(data_masked)
    im2 = axes[0,1].imshow(magnitude, cmap='plasma', origin='lower', 
                           aspect='auto', vmax=200, extent=extent)
    axes[0,1].set_title('Magnetic Field Strength')
    axes[0,1].set_xlabel('X (Pixel)')
    axes[0,1].set_ylabel('Y (Pixel)')
    axes[0,1].set_aspect('equal')
    axes[0,1].set_xlim(x_min, x_max)
    axes[0,1].set_ylim(y_min, y_max)
    # center_col, center_rowに対応する線を描画
    axes[0,1].axvline(x=x_from_center, color='cyan', linestyle='--', alpha=0.8, linewidth=2, label=f'X={x_from_center}(pixel)')
    axes[0,1].axhline(y=y_from_center, color='magenta', linestyle='--', alpha=0.8, linewidth=2, label=f'Y={y_from_center}(pixel)')
    axes[0,1].legend(loc='upper right')
    plt.colorbar(im2, ax=axes[0,1], label='|Br| (Gauss)', shrink=0.8)
    

    # col_profileは既にy_maskで制限されているので、直接計算
    valid_col_data = col_profile[np.isfinite(col_profile)]
    col_profile_mean = np.mean(valid_col_data)
    col_profile_std = np.std(valid_col_data)

    axes[1,0].plot(y_coords, col_profile, '-', color='cyan', linewidth=1, label=f'Ave$\\pm$std: {col_profile_mean:.1f} $\\pm$ {col_profile_std:.1f} Gauss')
    axes[1,0].set_title(f'On X={x_from_center} line profile')
    axes[1,0].set_xlabel('Y (Pixel from Sun center)')
    axes[1,0].set_ylabel('Br (Gauss)')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1,0].axvline(x=0, color='orange', linestyle=':', alpha=0.7, linewidth=1)  # 太陽中心線
    axes[1,0].set_xlim(y_min, y_max)
    axes[1,0].legend(loc='upper right')
    axes[1,0].axhline(y=col_profile_mean, color='cyan', linestyle='--', alpha=0.8, linewidth=2)
    # col_profileのave±stdの範囲にシェードをかける
    axes[1,0].fill_between(y_coords, 
                          col_profile_mean - col_profile_std,
                          col_profile_mean + col_profile_std,
                          color='gray', alpha=0.2)

    valid_row_data = row_profile[np.isfinite(row_profile)]
    row_profile_mean = np.mean(valid_row_data)
    row_profile_std = np.std(valid_row_data)


    axes[1,1].plot(x_coords, row_profile, '-', color='magenta', linewidth=1, label=f'Ave$\\pm$std: {row_profile_mean:.1f} $\\pm$ {row_profile_std:.1f} Gauss')
    axes[1,1].set_title(f'On Y={y_from_center} line profile')
    axes[1,1].set_xlabel('X (Pixel from Sun center)')
    axes[1,1].set_ylabel('Br (Gauss)')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1,1].axvline(x=0, color='orange', linestyle=':', alpha=0.7, linewidth=1)  # 太陽中心線
    axes[1,1].set_xlim(x_min, x_max)
    axes[1,1].legend(loc='upper right')
    axes[1,1].axhline(y=row_profile_mean, color='magenta', linestyle='--', alpha=0.8, linewidth=2)
    # row_profileのave±stdの範囲にシェードをかける
    axes[1,1].fill_between(x_coords, 
                          row_profile_mean - row_profile_std,
                          row_profile_mean + row_profile_std,
                          color='gray', alpha=0.2)

    plt.tight_layout()
    
    # 保存場所をResearch/SDO/HMIに変更
    save_dir = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI"
    filename = f"hmi_quick_view_{time_str.replace(':', '').replace('.', '').replace('-', '')}.png"
    full_path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    print(f"✓ 高速プロット保存: {full_path}")
    
    plt.show()
    
    # 簡易統計
    print(f"\n統計情報:")
    print(f"  元データサイズ: {data.shape}")
    print(f"  プロット解像度: {data_plot.shape}")
    print(f"  磁場範囲: {data_min:.1f} ～ {data_max:.1f} Gauss")
    print(f"  平均磁場: {np.mean(valid_data):.1f} ± {data_std:.1f} Gauss")
    print(f"  有効ピクセル率: {100*len(valid_data)/data_plot.size:.1f}%")

def main():
    """
    メイン実行関数
    """
    print("HMI磁場マップ高速プロットツール")
    print("対象時刻: 2022-06-13T03:00:00")
    print("=" * 40)
    
    # HMI FITSファイルのパス
    hmi_file = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    
    if not os.path.exists(hmi_file):
        print(f"エラー: ファイルが見つかりません: {hmi_file}")
        return
    
    if not ASTROPY_AVAILABLE:
        print("astropy パッケージをインストールしてください: pip install astropy")
        return
    
    try:
        # HMI読み込み
        hmi_data = read_hmi_quick(hmi_file)
        if hmi_data is None:
            print("HMI データの読み込みに失敗しました")
            return
        
        # 高速プロット
        plot_hmi_quick(hmi_data)
        
        print("\n✓ HMI高速プロット完了")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()