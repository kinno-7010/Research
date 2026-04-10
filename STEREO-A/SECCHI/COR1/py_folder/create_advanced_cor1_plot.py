#!/usr/bin/env python3
"""
STEREO-A/SECCHI/COR1 プロフェッショナルプロッター (sunpy統合版)

提供された cor_prep.py と cor1_colors.py を活用し、
sunpyライブラリを用いてSTEREO/COR1のFITSデータを処理・可視化します。

主な機能:
- cor_prepによるレベル0.5から1.0へのデータ前処理
- sunpy標準カラーマップの適用
- sunpyによる座標系を考慮した正確なプロット
- 太陽の輪郭、座標グリッド、太陽半径スケールの自動描画
- プロフェッショナルなアノテーションの追加
"""

import os
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import ZScaleInterval
from matplotlib.patches import Circle
import sunpy.map
import warnings

# --- 提供されたカスタムモジュールのインポート ---
try:
    from cor_prep import CORPrep
    # from cor1_colors import SECCHIColors # sunpyの標準カラーマップを使用するため不要
except ImportError:
    print("エラー: 'cor_prep.py' が同じディレクトリに存在することを確認してください。")
    exit()

# sunpyの警告を一部抑制
warnings.filterwarnings('ignore', category=sunpy.map.mapbase.SunpyUserWarning)


def create_professional_cor1_plot(fits_filepath, output_dir="."):
    """
    COR1 FITSデータの前処理を行い、sunpyを用いてプロットを作成する。

    Parameters:
    -----------
    fits_filepath : str
        入力となるCOR1 FITSファイルのパス（レベル0.5を想定）。
    output_dir : str, optional
        生成されたプロットを保存するディレクトリ。
    """
    if not os.path.exists(fits_filepath):
        print(f"エラー: 指定されたFITSファイルが見つかりません: {fits_filepath}")
        return

    print(f"処理開始: {os.path.basename(fits_filepath)}")

    # --- 1. データの前処理 (cor_prep.pyを使用) ---
    print("ステップ1: CORPrepによるデータ前処理...")
    preprocessor = CORPrep(silent=True)
    processed_data, processed_header_dict = preprocessor.cor_prep(
        filepath=fits_filepath,
        rotate_on=False,
        smask_on=True,
        calibrate_off=False,
        discri_pobj_on=True
    )
    if processed_data is None:
        print(f"エラー: {fits_filepath} の前処理に失敗しました。")
        return

    history_list = processed_header_dict.pop('HISTORY', [])
    processed_header = fits.Header(processed_header_dict)
    if history_list:
        for history_entry in history_list:
            sanitized_lines = str(history_entry).split('\n')
            for line in sanitized_lines:
                clean_line = line.strip()
                if clean_line:
                    ascii_line = ''.join(c for c in clean_line if ord(c) < 128)
                    processed_header.add_history(ascii_line)

    print("前処理完了。")

    # --- 2. sunpy.map.Mapオブジェクトの作成 ---
    print("ステップ2: sunpy.map.Mapオブジェクトの作成...")
    cor1_map = sunpy.map.Map((processed_data, processed_header))
    cor1_map = cor1_map.rotate()
    print("Mapオブジェクトの作成と回転が完了。")
    
    # --- 3. カラーマップとスケーリングの準備 ---
    print("ステップ3: カラーマップとスケーリングの準備...")
    # >>>>> ここから修正 >>>>>
    # sunpyに登録されているSTEREO/SECCHI COR1用の標準カラーマップ名を使用
    cor1_cmap_name = 'stereocor1' 
    try:
        cor1_cmap = plt.get_cmap(cor1_cmap_name)
        print(f"sunpy標準カラーマップ '{cor1_cmap_name}' を使用します。")
    except ValueError:
        print(f"警告: カラーマップ '{cor1_cmap_name}' が見つかりません。デフォルトの'gray'を使用します。")
        cor1_cmap = plt.get_cmap('gray')
    # <<<<< ここまで修正 <<<<<
    
    # interval = ZScaleInterval()
    # vmin, vmax = interval.get_limits(cor1_map.data)
    vmin, vmax = 1000, 10000
    print(f"カラースケーリング (ZScale): vmin={vmin:.2f}, vmax={vmax:.2f}")

    # --- 4. プロット作成 (sunpyとmatplotlib) ---
    print("ステップ4: sunpyによる画像のプロット...")
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection=cor1_map)
    # >>>>> ここから修正 >>>>>
    # cmapに取得したカラーマップオブジェクトを渡す
    im = cor1_map.plot(axes=ax, cmap=cor1_cmap, vmin=vmin, vmax=vmax, title=False)
    # <<<<< ここまで修正 <<<<<

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label(f"Intensity [{cor1_map.unit}]", fontsize=12)
    cor1_map.draw_limb(color='yellow', linewidth=2, alpha=0.8)
    cor1_map.draw_grid(grid_spacing=(10, 10) * u.deg, color='white', alpha=0.5, linestyle=':')
    
    center_pix = cor1_map.world_to_pixel(SkyCoord(0*u.arcsec, 0*u.arcsec, frame=cor1_map.coordinate_frame))
    rsun_pix = cor1_map.rsun_obs.to(u.arcsec).value / cor1_map.scale.axis1.to(u.arcsec/u.pix).value
    for i, color in enumerate(['yellow', 'orange', 'red']):
        radius_pix = (i + 1) * rsun_pix
        circle = Circle((center_pix.x.value, center_pix.y.value), radius_pix,
                        fill=False, edgecolor=color, linewidth=1.5,
                        alpha=0.8, label=f'{i+1} $R_\\odot$')
        ax.add_patch(circle)

    # --- 5. アノテーションの追加 ---
    print("ステップ5: アノテーションの追加...")
    ax.set_title(f'STEREO-A/SECCHI/COR1', fontsize=16, fontweight='bold', pad=20, color='white')
    time_str = cor1_map.date.strftime('%Y-%m-%d %H:%M:%S UT')
    detector_info = f"{cor1_map.observatory}/{cor1_map.instrument}-{cor1_map.detector}"
    ax.text(0.02, 0.08, time_str, transform=ax.transAxes, color='white', weight='bold', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.5))
    ax.text(0.02, 0.12, detector_info, transform=ax.transAxes, color='white', weight='bold', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.5))
    ax.text(0.98, 0.98, 'SECCHI', transform=ax.transAxes, color='white', weight='bold', fontsize=14,
            ha='right', va='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='navy', alpha=0.8))

    # --- 6. 画像の保存 ---
    print("ステップ6: プロットの保存...")
    base_name = os.path.splitext(os.path.basename(fits_filepath))[0]
    output_filename = f"{base_name}_sunpy_color_plot.png"
    save_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"✓ プロットが正常に保存されました: {save_path}")

    return save_path


if __name__ == '__main__':
    fits_file_path = "/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/Rawdata/20220613_030136_n4c1A.fts"
    output_directory = "/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1"
    
    create_professional_cor1_plot(fits_file_path, output_directory)