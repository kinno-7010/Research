"""
トモグラフィーHDF5ファイル読み込みモジュール
SOHO-LASCO プロジェクト用

Purpose:
    トモグラフィー結果を格納した.hdf (または.hf5) ファイルを読み込む
    
Author: Python translation from IDL original by J.W (December 2015)
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings
import pyhdf

def tomo_hdf_read(hdf_file_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    トモグラフィーHDFファイルを読み込み、4D (3D+t) 電子密度ボリュームと軸データを提供
    
    Parameters
    ----------
    hdf_file_name : str
        入力するトモ結果ファイル（.hdfまたは.hf5フォーマット）のパス
        
    Returns
    -------
    lon : np.ndarray
        経度ベクトル（度）
    lat : np.ndarray
        緯度ベクトル（度）
    rad : np.ndarray
        半径ベクトル（太陽半径単位）
    time : np.ndarray
        時間ベクトル（開始日からの日数）
    tomo_vol : np.ndarray
        4Dトモグラフィックボリューム (3D + t)
        座標は [rad, lat, lon, time] の順
    misc : dict
        追加情報を含む辞書：
        - 'startingdate': 開始日（yyyymmdd形式の文字列）
        - 'endingdate': 終了日（yyyymmdd形式の文字列）  
        - 'obscl': 観測者のCarrington経度（時間軸と同じサイズのベクトル）
        
    Raises
    ------
    FileNotFoundError
        指定されたファイルが存在しない場合
    KeyError
        必要なデータセットがHDF5ファイル内に見つからない場合
    """
    
    # ファイル存在チェック
    file_path = Path(hdf_file_name)
    if not file_path.exists():
        raise FileNotFoundError(f"トモグラフィーファイルが見つかりません: {hdf_file_name}")
    
    # 出力用の辞書を初期化
    misc = {}
    
    try:
        # HDF5ファイルを開く
        with pyhdf.File(hdf_file_name, 'r') as hdf_file:
            
            # 軸データの読み込み
            # /axes グループ内の緯度データセットを読み込み
            lat = hdf_file['/axes/latitudes'][:]
            
            # /axes グループ内の経度データセットを読み込み
            lon = hdf_file['/axes/longitudes'][:]
            
            # /axes グループ内の半径データセットを読み込み
            rad = hdf_file['/axes/rad'][:]
            
            # /axes グループ内の時間データセットを読み込み
            time = hdf_file['/axes/time'][:]
            
            # ボリュームデータの読み込み
            # /volume グループ内の4Dボリュームデータセットを読み込み
            tomo_vol = hdf_file['/volume/dataset_4D'][:]
            
            # データセットサイズの検証（もし存在すれば）
            if '/volume/dataset_4D_size' in hdf_file:
                expected_size = hdf_file['/volume/dataset_4D_size'][:]
                # 転置が必要な場合がある（IDL/Python の配列順序の違い）
                if hasattr(expected_size, 'T'):
                    expected_size = expected_size.T
                    
                if not np.array_equal(expected_size, tomo_vol.shape):
                    warnings.warn(f"トモグラフィックボリュームサイズの不整合: "
                                f"期待値 {expected_size}, 実際 {tomo_vol.shape}")
            
            # その他の情報を読み込み
            # /misc グループ内の観測者Carrington経度を読み込み
            if '/misc/obscl' in hdf_file:
                misc['obscl'] = hdf_file['/misc/obscl'][:]
            else:
                # データが存在しない場合は、時間軸と同じサイズのゼロ配列を作成
                misc['obscl'] = np.zeros(len(time))
                warnings.warn("観測者Carrington経度データが見つかりません。ゼロで初期化しました。")
            
            # /misc グループ内の開始日を読み込み
            if '/misc/startingdate' in hdf_file:
                startdate_data = hdf_file['/misc/startingdate']
                # データタイプに応じて適切に処理
                if hasattr(startdate_data, 'dtype') and startdate_data.dtype.char == 'S':
                    # バイト文字列の場合
                    misc['startingdate'] = startdate_data[()].decode('utf-8') if isinstance(startdate_data[()], bytes) else str(startdate_data[()])
                else:
                    misc['startingdate'] = str(startdate_data[()])
            else:
                misc['startingdate'] = 'yyyymmdd'
                warnings.warn("開始日データが見つかりません。")
            
            # /misc グループ内の終了日を読み込み  
            if '/misc/endingdate' in hdf_file:
                enddate_data = hdf_file['/misc/endingdate']
                # データタイプに応じて適切に処理
                if hasattr(enddate_data, 'dtype') and enddate_data.dtype.char == 'S':
                    # バイト文字列の場合
                    misc['endingdate'] = enddate_data[()].decode('utf-8') if isinstance(enddate_data[()], bytes) else str(enddate_data[()])
                else:
                    misc['endingdate'] = str(enddate_data[()])
            else:
                misc['endingdate'] = 'yyyymmdd'
                warnings.warn("終了日データが見つかりません。")
                
            # データの形状を確認（デバッグ用）
            print(f"読み込み完了:")
            print(f"  経度: {lon.shape}, 範囲: [{lon.min():.2f}, {lon.max():.2f}]")
            print(f"  緯度: {lat.shape}, 範囲: [{lat.min():.2f}, {lat.max():.2f}]")
            print(f"  半径: {rad.shape}, 範囲: [{rad.min():.2f}, {rad.max():.2f}] Rsun")
            print(f"  時間: {time.shape}, 範囲: [{time.min():.2f}, {time.max():.2f}] days")
            print(f"  ボリューム: {tomo_vol.shape}")
            print(f"  開始日: {misc['startingdate']}")
            print(f"  終了日: {misc['endingdate']}")
                
    except KeyError as e:
        raise KeyError(f"必要なデータセットがHDF5ファイル内に見つかりません: {e}")
    except Exception as e:
        raise RuntimeError(f"HDF5ファイルの読み込みエラー: {e}")
        
    return lon, lat, rad, time, tomo_vol, misc

if __name__ == "__main__":
    for hdf_file in "/mnt/d/wsl/home/kinno-7010/Research_data/3D_reconstruction/Rawdata/*.hdf":
        lon, lat, rad, time, tomo_vol, misc = tomo_hdf_read(hdf_file)
        print(lon)
        print(lat)
        print(rad)
        print(time)
        print(tomo_vol)
        print(misc)