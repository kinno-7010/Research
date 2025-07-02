"""
共通設定ファイル
太陽物理学研究用のデータ解析に必要な定数、パス、インポートを管理
"""

import io
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import astropy.units as u
import astropy.visualization as vis
import imageio.v2 as imageio
import matplotlib
matplotlib.use('TkAgg')  # GUI表示用バックエンドを明示的に設定
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sunpy.map
from astropy.time import Time
from astropy.visualization import ImageNormalize, PowerStretch, AsinhStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
from astropy.io import fits
from datetime import timedelta
import glob
import re
from matplotlib.patches import Circle
from scipy.ndimage import map_coordinates
from matplotlib.colors import Normalize
import bisect
from reproject import reproject_interp
import logging
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import rotate
import warnings
from astropy.io.fits import file as fits_file
import gc

# データディレクトリの設定（WSLパス）
BASE_DATA_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research/SDO/AIA/Rawdata")
OUTPUT_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research/SDO/AIA/output")

# データフォルダ辞書（WSLパス）
data_folder_dict = {
    'mk4': '/mnt/d/wsl/home/kinno-7010/Research/MK4_coronagraph/MK4_coronagraph_KCOR/Subtraction_data/Rawdata/kcor_nrgf',
    'lasco': '/mnt/d/wsl/home/kinno-7010/Research/SOHO/LASCO-C2_rawdata',
    'aia211': '/mnt/d/wsl/home/kinno-7010/Research/SDO/AIA/Rawdata/211',
    'aia193': '/mnt/d/wsl/home/kinno-7010/Research/SDO/AIA/Rawdata/193',
    'aia171': '/mnt/d/wsl/home/kinno-7010/Research/SDO/AIA/Rawdata/171'
}

# MK4データ設定（WSLパス）
mk4_data_folder = '/mnt/d/wsl/home/kinno-7010/Research/MK4_coronagraph/MK4_coronagraph_KCOR/Subtraction_data/Rawdata'
mk4_output_folder = '/mnt/d/wsl/home/kinno-7010/Research/MK4_coronagraph/MK4_coronagraph_KCOR/Subtraction_data/output'

# ログ設定
logging.getLogger('sunpy.map.mapbase').setLevel(logging.WARNING)
logging.getLogger('sunpy').setLevel(logging.WARNING)
logging.getLogger('reproject').setLevel(logging.WARNING)
logging.getLogger('reproject.mosaicking').setLevel(logging.WARNING)

warnings.filterwarnings(
    "ignore",
    message="Could not memory map array.*",
    category=UserWarning,
    module=fits_file.__name__
)

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# 解析パラメータ
DEFAULT_RANGES = {
    'mk4_inner': 1.1,
    'mk4_outer_lasco_inner': 3,
    'lasco_outer': 6.0
}

# CME解析設定（WSLパス）
CME_ANALYSIS_OUTPUT_DIR = '/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO'

# グローバルキャッシュ（スキャン結果を保存）
_global_scan_cache = {}