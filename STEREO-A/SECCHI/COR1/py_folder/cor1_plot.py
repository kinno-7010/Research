#!/usr/bin/env python3
"""
STEREO-A/SECCHI/COR1 コロナグラフデータの読み取り・プロット

このスクリプトは03:24付近の時間帯のCOR1データを読み取り、プロットします。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
import glob
from datetime import datetime
from matplotlib.patches import Circle

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

def read_cor1_data(filepath):
    """
    COR1 FITSファイルを読み込み、データとヘッダーを返す
    
    Parameters:
    -----------
    filepath : str
        FITSファイルのパス
    
    Returns:
    --------
    data : numpy.ndarray
        画像データ
    header : astropy.io.fits.Header
        FITSヘッダー
    """
    try:
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            header = hdul[0].header
        return data, header
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

def calculate_solar_radius_pixel(header):
    """
    FITSヘッダーから太陽半径をピクセル単位で計算
    
    Parameters:
    -----------
    header : astropy.io.fits.Header
        FITSヘッダー
    
    Returns:
    --------
    rsun_pixel : float
        太陽半径（ピクセル単位）
    sun_center_x : float
        太陽中心のx座標（ピクセル単位）
    sun_center_y : float
        太陽中心のy座標（ピクセル単位）
    """
    # 太陽半径（arcsec）
    rsun_arcsec = header.get('RSUN', None)
    if rsun_arcsec is None:
        return None, None, None
    
    # ピクセルスケール（arcsec/pixel）
    cdelt1 = header.get('CDELT1', None)
    cdelt2 = header.get('CDELT2', None)
    
    if cdelt1 is None or cdelt2 is None:
        return None, None, None
    
    # 太陽中心のピクセル位置
    crpix1 = header.get('CRPIX1', None)
    crpix2 = header.get('CRPIX2', None)
    
    if crpix1 is None or crpix2 is None:
        return None, None, None
    
    # 太陽半径をピクセル単位に変換
    rsun_pixel = rsun_arcsec / abs(cdelt1)
    
    # 太陽中心座標（1-indexedから0-indexedに変換）
    sun_center_x = crpix1 - 1
    sun_center_y = crpix2 - 1
    
    return rsun_pixel, sun_center_x, sun_center_y

def plot_cor1_image(data, header, filepath, save_path=None):
    """
    COR1データをプロット
    
    Parameters:
    -----------
    data : numpy.ndarray
        画像データ
    header : astropy.io.fits.Header
        FITSヘッダー
    filepath : str
        元のファイルパス
    save_path : str, optional
        保存先パス
    """
    # データの基本情報を取得
    obs_time = header.get('DATE-OBS', 'Unknown')
    instrument = header.get('INSTRUME', 'Unknown')
    detector = header.get('DETECTOR', 'Unknown')
    
    # 時間情報を解析
    try:
        time_obj = Time(obs_time)
        time_str = time_obj.datetime.strftime('%Y-%m-%d %H:%M:%S')
    except:
        time_str = obs_time
    
    # 太陽半径の計算
    rsun_pixel, sun_center_x, sun_center_y = calculate_solar_radius_pixel(header)
    rsun_arcsec = header.get('RSUN', None)
    
    # データの統計情報
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data_mean = np.nanmean(data)
    data_std = np.nanstd(data)
    
    print(f"File: {os.path.basename(filepath)}")
    print(f"Observation Time: {time_str}")
    print(f"Instrument: {instrument}")
    print(f"Detector: {detector}")
    print(f"Data shape: {data.shape}")
    print(f"Data range: {data_min:.2f} to {data_max:.2f}")
    print(f"Data mean: {data_mean:.2f} ± {data_std:.2f}")
    if rsun_pixel is not None:
        print(f"Solar radius: {rsun_arcsec:.2f} arcsec ({rsun_pixel:.2f} pixels)")
        print(f"Solar center: ({sun_center_x:.2f}, {sun_center_y:.2f}) pixels")
    print("-" * 50)
    
    # ZScaleを使用してコントラストを調整
    interval = ZScaleInterval()
    vmin, vmax = 2000, 6000
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 画像表示
    im = ax.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    
    # カラーバー追加
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Intensity [DN]', rotation=270, labelpad=20)
    
    # タイトルとラベル設定
    ax.set_title(f'STEREO-A/SECCHI/COR1\n{time_str}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Pixel X', fontsize=12)
    ax.set_ylabel('Pixel Y', fontsize=12)
    
    # グリッド追加
    ax.grid(True, alpha=0.3)
    
    # 画像の中心に十字線を追加
    center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
    ax.axhline(y=center_y, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=center_x, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # 太陽半径の円を描画
    if rsun_pixel is not None and sun_center_x is not None and sun_center_y is not None:
        # 太陽半径の円（1Rs）
        circle_1rs = Circle((sun_center_x, sun_center_y), rsun_pixel, 
                           fill=False, edgecolor='yellow', linewidth=2, 
                           alpha=0.8, label='1 Rs')
        ax.add_patch(circle_1rs)
        
        # 2Rs、3Rsの円も追加
        circle_2rs = Circle((sun_center_x, sun_center_y), 2*rsun_pixel, 
                           fill=False, edgecolor='orange', linewidth=1.5, 
                           alpha=0.6, label='2 Rs')
        ax.add_patch(circle_2rs)
        
        circle_3rs = Circle((sun_center_x, sun_center_y), 3*rsun_pixel, 
                           fill=False, edgecolor='red', linewidth=1.5, 
                           alpha=0.6, label='3 Rs')
        ax.add_patch(circle_3rs)
        
        # 太陽中心に点を追加
        ax.plot(sun_center_x, sun_center_y, 'x', color='yellow', 
                markersize=8, markeredgewidth=2, label='Solar center')
        
        # 凡例を追加
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    
    # レイアウト調整
    plt.tight_layout()
    
    # 保存またはプロット表示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig, ax

def main():
    """メイン関数"""
    # データディレクトリのパス
    data_dir = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata"
    
    # 利用可能な全てのファイルを取得
    # all_files = glob.glob(os.path.join(data_dir, "*.fts"))
    target_files = ['20220613_032136_n4c1A.fts']
    
    # 各ファイルを処理
    for filename in target_files:
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        
        print(f"\n処理中: {filename}")
        
        # データ読み込み
        data, header = read_cor1_data(filepath)
        
        if data is None:
            continue
        
        # プロット作成
        output_filename = filename.replace('.fts', '_plot.png')
        save_path = os.path.join(
            "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1",
            output_filename
        )
        
        fig, ax = plot_cor1_image(data, header, filepath, save_path)
        plt.close(fig)  # メモリ節約のため閉じる
    
    print("\nすべてのファイルの処理が完了しました。")

if __name__ == "__main__":
    main()