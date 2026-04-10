"""
UCOMP設定ファイル
UCOMPデータの読み込み、処理、描画のための設定とユーティリティ関数
"""

import sys
import os
sys.path.append('/home/kinno-7010/Research_code/SDO_Mk4_SOHO/py_folder')

# 必要なライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from astropy.visualization import ImageNormalize, LinearStretch
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# GUIが利用可能な場合はTkAggを使用、そうでなければAggを使用
try:
    import tkinter
    matplotlib.use('TkAgg')
except ImportError:
    matplotlib.use('Agg')

# UCOMPデータパス設定
UCOMP_DATA_DIR = "/mnt/d/wsl/home/kinno-7010/Research_data/MK4_coronagraph/UCOMP/Rawdata"
UCOMP_OUTPUT_DIR = "/mnt/d/wsl/home/kinno-7010/Research_data/MK4_coronagraph/UCOMP"

# UCOMPファイル名パターン設定
UCOMP_FILENAME_PATTERN = "YYYYMMDD.HHMMSS.ucomp.{wavelength}.l2.fts"

# UCOMP Level 2 FITSのExtension情報
UCOMP_EXTENSIONS = {
    1: "Center Line Intensity",
    2: "Enhanced (unsharp mask) intensity", 
    3: "Gaussian Peak Intensity",
    4: "Line-of-Sight Doppler Velocity",
    5: "FWHM Line Width",
    6: "Noise Mask",
    7: "Weighted avg Stokes I",
    8: "Weighted avg Stokes Q", 
    9: "Weighted avg Stokes U",
    10: "Weighted avg Linear Polarization (L)",
    11: "Azimuth of plane-of-sky magnetic field",
    12: "Radial Azimuth of plane-of-sky magnetic field"
}

# カラーバーのラベル情報
UCOMP_COLORBAR_LABELS = {
    1: "Intensity",
    2: "Enhanced Intensity", 
    3: "Peak Intensity",
    4: "Velocity (km/s)",
    5: "Line Width (km/s)",
    6: "Noise Level",
    7: "Stokes I",
    8: "Stokes Q", 
    9: "Stokes U",
    10: "Linear Pol (%)",
    11: "Azimuth (deg)",
    12: "Radial Azimuth (deg)"
}

# デフォルト波長
DEFAULT_WAVELENGTH = 1074

def get_header_info(file_path):
    """ヘッダー情報を取得"""
    with fits.open(file_path) as hdul:
        header = hdul[0].header
    return header

def get_ucomp_data_path():
    """UCOMPデータディレクトリのパスを取得"""
    return Path(UCOMP_DATA_DIR)

def get_ucomp_output_path():
    """UCOMP出力ディレクトリのパスを取得"""
    output_path = Path(UCOMP_OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def create_output_filename(timestamp, group_name="all"):
    """出力ファイル名を作成"""
    if isinstance(timestamp, str):
        # ISO形式文字列から変換
        from astropy.time import Time
        timestamp = Time(timestamp)
    
    time_str = timestamp.strftime("%Y-%m-%d-%H-%M-%S")
    return f"ucomp_ext_{group_name}_{time_str}.png"

def validate_wavelength(wavelength):
    """波長の妥当性をチェック"""
    valid_wavelengths = [637, 706, 789, 1074, 1079]
    if wavelength not in valid_wavelengths:
        raise ValueError(f"Invalid wavelength: {wavelength}. Valid options: {valid_wavelengths}")
    return True

def get_extension_title(ext_num):
    """Extension番号から対応するタイトルを取得"""
    return UCOMP_EXTENSIONS.get(ext_num, f"Extension {ext_num}")

def get_colorbar_label(ext_num):
    """Extension番号から対応するカラーバーラベルを取得"""
    return UCOMP_COLORBAR_LABELS.get(ext_num, "Value")

def create_filename_pattern(wavelength):
    """指定波長のファイル名パターンを作成"""
    validate_wavelength(wavelength)
    return f"*.ucomp.{wavelength}.l2.fts"