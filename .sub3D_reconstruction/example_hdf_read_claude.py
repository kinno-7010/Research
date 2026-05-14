"""
トモグラフィーHDF5ファイル読み込みの使用例
SOHO-LASCO プロジェクト用

Purpose:
    tomo_hdf_read の使用方法を示す例
    'CR2062_1.hf5' ファイルを読み込んで投影マップを描画
    
Author: Python translation from IDL original by J.W (December 2015)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from datetime import datetime, timedelta
from pathlib import Path
import warnings

# tomo_hdf_read関数をインポート（同じディレクトリにあると仮定）
from tomo_hdf_read import tomo_hdf_read


def example_hdf_read():
    """
    HDF5トモグラフィーファイルを読み込んで可視化する例
    
    この関数は以下を実行：
    1. HDF5ファイルからトモグラフィーデータを読み込み
    2. 指定された半径での緯度/経度マップを描画
    3. 指定された緯度での半径/経度マップを描画
    """
    
    # ステップ 0: グローバルパラメータの設定
    dataset_filename = '../CR2062_1.hf5'  # トモグラフィーデータファイル
    cr_start_date = '2007/10/08'  # Carrington回転 2062 の開始日
    # cr_mid_date = '2007/10/12'  # Carrington回転 2062 の中間日（必要に応じて切り替え可能）
    cr_date_of_interest = cr_start_date  # 注目する日付（cr_mid_dateに切り替え可能）
    radii_to_show = [3, 4]  # 表示する半径（太陽半径単位）- 緯度/経度マップ用
    latitude_to_show = 0  # 表示する緯度（度）- 半径/経度マップ用
    
    # ファイルの存在確認
    file_path = Path(dataset_filename)
    if not file_path.exists():
        warnings.warn(f"ファイルが見つかりません: {dataset_filename}")
        # デモ用の代替パスを試す
        alternative_paths = [
            './CR2062_1.hf5',
            '/mnt/d/wsl/home/kinno-7010/Research_data/CR2062_1.hf5',
            './data/CR2062_1.hf5'
        ]
        for alt_path in alternative_paths:
            if Path(alt_path).exists():
                dataset_filename = alt_path
                print(f"代替パスを使用: {dataset_filename}")
                break
        else:
            raise FileNotFoundError(f"データファイルが見つかりません。パスを確認してください: {dataset_filename}")
    
    # ステップ 1.1: ファイル読み込み
    # ファイル読み込みによって緯度、経度、半径、時間を取得
    print(f"ファイルを読み込み中: {dataset_filename}")
    lon, lat, rad, time, vol, misc = tomo_hdf_read(dataset_filename)
    
    # ステップ 1.2: 日付を数値形式に変換して時間インデックスを取得
    # 注目日を日時オブジェクトに変換
    date_of_interest = datetime.strptime(cr_date_of_interest, '%Y/%m/%d')
    
    # トモグラフィー開始日を日時オブジェクトに変換
    # startingdateの形式はyyyymmddと仮定
    start_date_str = misc['startingdate']
    if len(start_date_str) == 8 and start_date_str.isdigit():
        tomo_start_date = datetime.strptime(start_date_str, '%Y%m%d')
    else:
        # フォーマットが異なる場合の処理
        tomo_start_date = datetime.strptime(start_date_str[:8], '%Y%m%d')
    
    # 開始日からの遅延（日数）を計算
    delay_from_start = (date_of_interest - tomo_start_date).days
    
    # 最も近い時間インデックスを見つける
    time_idx = np.argmin(np.abs(delay_from_start - time))
    
    print(f"選択された時刻: 開始日から {time[time_idx]:.2f} 日後")
    print(f"対応する日付: {tomo_start_date + timedelta(days=float(time[time_idx]))}")
    
    # ステップ 2.1: 緯度/経度マップのプロット
    # 選択された半径での緯度/経度マップを描画
    for current_radius in radii_to_show:
        # 現在の半径のインデックスを取得
        r_idx = np.argmin(np.abs(rad - current_radius))
        
        # 新しい図を作成
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # データを抽出（log10変換、最小値0.001でクリップ）
        data_slice = np.log10(np.maximum(vol[r_idx, :, :, time_idx], 0.001))
        
        # 画像を表示（転置して正しい方向に）
        im = ax.imshow(data_slice.T, 
                      extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                      aspect='auto',
                      origin='lower',
                      cmap='turbo',  # IDLのrgb_table=5に相当
                      vmin=3.5, vmax=5.5)
        
        # カラーバーを追加
        cbar = plt.colorbar(im, ax=ax, label='Log₁₀(Ne)')
        
        # 軸ラベルとタイトル
        ax.set_xlabel('経度 (度)', fontsize=12)
        ax.set_ylabel('緯度 (度)', fontsize=12)
        ax.set_title(f'緯度/経度マップ @ {rad[r_idx]:.2f} R☉\n'
                    f'日付: {tomo_start_date + timedelta(days=float(time[time_idx]))}',
                    fontsize=14)
        
        # 目盛りを設定
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks([-90, -45, 0, 45, 90])
        
        # グリッドを追加
        ax.grid(True, alpha=0.3)
        
        # 観測者の経度を取得してマップ上に描画
        if 'obscl' in misc and len(misc['obscl']) > time_idx:
            obs_lon = misc['obscl'][time_idx]
            ax.axvline(x=obs_lon, color='red', linestyle='--', linewidth=2,
                      label=f'観測者経度: {obs_lon:.1f}°')
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    # ステップ 2.2: 経度/半径マップのプロット
    # 選択された緯度での半径/経度マップを描画
    if not isinstance(latitude_to_show, list):
        latitude_to_show = [latitude_to_show]
    
    for current_lat in latitude_to_show:
        # 現在の緯度のインデックスを取得
        lat_idx = np.argmin(np.abs(lat - current_lat))
        
        # 新しい図を作成
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # データを抽出（log10変換、最小値0.001でクリップ）
        data_slice = np.log10(np.maximum(vol[:, lat_idx, :, time_idx], 0.001))
        
        # 画像を表示（IDLのreverse相当の操作）
        # 半径軸を拡大表示するため、radを40倍（IDLコードに従う）
        im = ax.imshow(data_slice,
                      extent=[lon.min(), lon.max(), rad.min()*40, rad.max()*40],
                      aspect='auto',
                      origin='lower',  # IDLのreverseに対応
                      cmap='turbo',
                      vmin=3.5, vmax=5.5)
        
        # カラーバーを追加
        cbar = plt.colorbar(im, ax=ax, label='Log₁₀(Ne)')
        
        # 軸ラベルとタイトル
        ax.set_xlabel('経度 (度)', fontsize=12)
        ax.set_ylabel('半径 × 40', fontsize=12)
        ax.set_title(f'半径/経度マップ @ 緯度 {lat[lat_idx]:.1f}°\n'
                    f'日付: {tomo_start_date + timedelta(days=float(time[time_idx]))}',
                    fontsize=14)
        
        # 経度の目盛りを設定
        ax.set_xticks([0, 90, 180, 270, 360])
        
        # グリッドを追加
        ax.grid(True, alpha=0.3)
        
        # 観測者の経度を追加（オプション）
        if 'obscl' in misc and len(misc['obscl']) > time_idx:
            obs_lon = misc['obscl'][time_idx]
            ax.axvline(x=obs_lon, color='red', linestyle='--', linewidth=2,
                      label=f'観測者経度: {obs_lon:.1f}°')
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    print("\n可視化が完了しました。")


if __name__ == "__main__":
    # メイン実行
    try:
        example_hdf_read()
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()