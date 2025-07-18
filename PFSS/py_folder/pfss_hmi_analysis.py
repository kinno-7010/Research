#!/usr/bin/env python3
"""
PFSSモデルとHMIデータを統合した3次元磁力線プロット
AIAマップ上への磁力線重ね合わせ機能付き
masked_dataの範囲のみを使用して計算量を削減
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# 必要なパッケージのインポート
try:
    from astropy.io import fits
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames
    import sunpy.map
    import pfsspy
    import pfsspy.tracing as tracing
    import astropy.visualization as vis
    from astropy.visualization import ImageNormalize, PowerStretch
    ASTROPY_AVAILABLE = True
    PFSSPY_AVAILABLE = True
except ImportError as e:
    print(f"必要なパッケージが不足しています: {e}")
    print("pip install astropy sunpy pfsspy")
    ASTROPY_AVAILABLE = False
    PFSSPY_AVAILABLE = False

# HMI解析モジュールのインポート
sys.path.append('/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/py_folder')
try:
    from hmi_analysis_wcs import read_hmi_quick
    HMI_MODULE_AVAILABLE = True
except ImportError:
    print("HMI解析モジュールが見つかりません")
    HMI_MODULE_AVAILABLE = False

def normalize_log_stretch(data):
    """
    LogStretch正規化を適用（AIA_analysis.ipynbを参考）
    """
    data_clipped = np.maximum(data, 1e-5)
    normalizer = vis.ImageNormalize(data_clipped, stretch=vis.LogStretch(), clip=True)
    return normalizer(data_clipped)

def create_aia_rgb_image_from_overlay(target_time="2022-06-13 03:00", 
                                    channel_r="211", channel_g="193", channel_b="171",
                                    hmi_map=None):
    """
    plot_hmi_pfss_overlay.pyのplot_sdo_aia_rgb関数を参考にしたAIA RGB画像生成
    HMIマップの太陽半径に合わせてAIA画像をリスケール
    
    Parameters:
    -----------
    target_time : str
        対象時刻（"YYYY-MM-DD HH:MM"形式）
    channel_r, channel_g, channel_b : str
        R, G, B チャンネルの波長
    hmi_map : sunpy.map.Map, optional
        HMIマップ（太陽半径の基準として使用）
        
    Returns:
    --------
    tuple : (rgb_image, reference_map, success)
    """
    print(f"\nAIA RGB画像を生成中（overlay版）: {target_time}")
    
    # ベースディレクトリ（plot_hmi_pfss_overlay.pyと同じパス）
    from pathlib import Path
    BASE_DATA_DIR = Path('/mnt/d/wsl/home/kinno-7010/Research/SDO/AIA/Rawdata')
    
    # 時刻文字列を解析
    try:
        from datetime import datetime
        dt_obj = datetime.strptime(target_time, "%Y-%m-%d %H:%M")
        date_fmtd_for_fname = dt_obj.strftime("%Y%m%d")
        time_fmtd_for_fname = dt_obj.strftime("%H%M")
    except ValueError:
        print(f"エラー: 時刻文字列の形式が無効です: {target_time}")
        return None, None, False
    
    # 各チャンネルのファイルパスを組み立て、Mapオブジェクトをロード
    maps = {}
    channels = {'r': channel_r, 'g': channel_g, 'b': channel_b}
    loaded_map_count = 0
    
    for color, ch_str in channels.items():
        wavelength_part_in_fname = ch_str.zfill(4)
        filename = f"AIA{date_fmtd_for_fname}_{time_fmtd_for_fname}_{wavelength_part_in_fname}.fits"
        file_path = BASE_DATA_DIR / ch_str / filename
        print(f"  {color.upper()}チャンネル ({ch_str}Å): {filename}")
        
        try:
            if file_path.exists():
                maps[color] = sunpy.map.Map(file_path)
                print(f"    成功")
                loaded_map_count += 1
            else:
                print(f"    ファイルが見つかりません: {file_path}")
                return None, None, False
        except Exception as e:
            print(f"    読み込みエラー: {e}")
            return None, None, False
    
    if loaded_map_count < 3:
        print("エラー: 3つ全ての波長チャンネルのデータを読み込めませんでした")
        return None, None, False
    
    # 基準となるMapオブジェクトを選択
    reference_map = maps['b'] if maps['b'] else maps['g'] if maps['g'] else maps['r']
    if not reference_map:
        print("エラー: 基準となるMapオブジェクトがありません")
        return None, None, False
    
    try:
        # 各チャンネルのデータを正規化（LinearStretchを使用）
        def normalize_linear_stretch(data):
            data_clipped = np.maximum(data, 1e-5)
            normalizer = vis.ImageNormalize(data_clipped, stretch=vis.LinearStretch(), clip=True)
            return normalizer(data_clipped)
        
        red_channel_data = normalize_linear_stretch(maps['r'].data)
        green_channel_data = normalize_linear_stretch(maps['g'].data)
        blue_channel_data = normalize_linear_stretch(maps['b'].data)
        
        # 0-1にスケーリング
        def scale_to_01(data):
            d_min = np.nanmin(data)
            d_max = np.nanmax(data)
            if d_max == d_min:
                return np.zeros_like(data)
            return (data - d_min) / (d_max - d_min)
        
        red_channel_final = scale_to_01(red_channel_data)
        green_channel_final = scale_to_01(green_channel_data)
        blue_channel_final = scale_to_01(blue_channel_data)
        
        # RGB画像を作成
        rgb_image = np.stack([red_channel_final, green_channel_final, blue_channel_final], axis=-1)
        
        print(f"  RGB画像生成完了: {rgb_image.shape}")
        
        # HMIマップが提供されている場合、太陽半径に基づいてリスケール
        if hmi_map is not None:
            print(f"  HMIマップの太陽半径に合わせてリスケール中...")
            
            # 太陽半径の取得
            try:
                hmi_rsun = hmi_map.rsun_obs  # HMIの太陽半径 (arcsec)
                aia_rsun = reference_map.rsun_obs  # AIAの太陽半径 (arcsec)
                
                print(f"    HMI太陽半径: {hmi_rsun:.2f} arcsec")
                print(f"    AIA太陽半径: {aia_rsun:.2f} arcsec")
                
                # スケール比を計算
                scale_ratio = hmi_rsun.value / aia_rsun.value
                print(f"    スケール比: {scale_ratio:.4f}")
                
                # RGB画像をリスケール
                from scipy.ndimage import zoom
                
                # 各チャンネルを個別にリスケール
                resized_rgb = np.zeros((int(rgb_image.shape[0] * scale_ratio), 
                                       int(rgb_image.shape[1] * scale_ratio), 3))
                
                for i in range(3):
                    resized_rgb[:, :, i] = zoom(rgb_image[:, :, i], scale_ratio, order=1)
                
                print(f"    リスケール完了: {rgb_image.shape} -> {resized_rgb.shape}")
                
                # リスケール後のRGB画像を返す
                return resized_rgb, reference_map, True
                
            except Exception as e:
                print(f"    リスケールエラー: {e}")
                print(f"    元のRGB画像を使用します")
                return rgb_image, reference_map, True
        else:
            # HMIマップが提供されていない場合は元のRGB画像を返す
            return rgb_image, reference_map, True
        
    except Exception as e:
        print(f"RGB画像生成エラー: {e}")
        return None, None, False

def load_aia_data(aia_file=None):
    """
    AIAデータを読み込み
    
    Parameters:
    -----------
    aia_file : str, optional
        AIAファイルのパス
        
    Returns:
    --------
    sunpy.map.Map : AIAマップオブジェクト
    """
    if aia_file is None:
        aia_file = "/mnt/d/wsl/home/kinno-7010/Research/SDO/AIA/Rawdata/211/AIA20220613_0300_0211.fits"
    
    print(f"AIAデータを読み込み中: {aia_file}")
    
    if not os.path.exists(aia_file):
        print(f"エラー: AIAファイルが見つかりません: {aia_file}")
        return None
    
    try:
        aia_map = sunpy.map.Map(aia_file)
        print(f"  AIAマップ作成完了")
        print(f"  観測時刻: {aia_map.date}")
        print(f"  波長: {aia_map.wavelength}")
        print(f"  データ形状: {aia_map.data.shape}")
        print(f"  強度範囲: {np.min(aia_map.data):.1f} ～ {np.max(aia_map.data):.1f}")
        
        return aia_map
        
    except Exception as e:
        print(f"AIAデータ読み込みエラー: {e}")
        return None

def get_masked_hmi_data(hmi_file=None):
    """
    HMIデータを読み込み、masked_dataの範囲を取得
    plot_hmi_single.pyと同じ範囲を使用
    
    Returns:
    --------
    dict : masked_dataとその座標情報
    """
    if hmi_file is None:
        hmi_file = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    
    print(f"HMIデータを読み込み中: {hmi_file}")
    
    if not HMI_MODULE_AVAILABLE:
        print("HMI解析モジュールが利用できません")
        return None
    
    # HMIデータ読み込み
    hmi_data = read_hmi_quick(hmi_file)
    data = hmi_data['data']
    hmi_map = hmi_data.get('sunpy_map')
    
    # plot_hmi_single.pyと同じマスク範囲を適用
    ny, nx = data.shape
    center_x, center_y = nx // 2, ny // 2
    x_min_pix, x_max_pix = center_x - 512, center_x + 0
    y_min_pix, y_max_pix = center_y - 100, center_y + 512
    
    # masked_dataを抽出
    masked_data = data[y_min_pix:y_max_pix, x_min_pix:x_max_pix]
    
    print(f"  元データ形状: {data.shape}")
    print(f"  マスク範囲: x[{x_min_pix}:{x_max_pix}], y[{y_min_pix}:{y_max_pix}]")
    print(f"  masked_data形状: {masked_data.shape}")
    print(f"  磁場範囲: {np.nanmin(masked_data):.1f} ～ {np.nanmax(masked_data):.1f} Gauss")
    
    # WCSを使って実際の太陽座標を計算
    if hmi_map is not None:
        # ピクセル座標から太陽座標への変換
        x_pixels = np.arange(x_min_pix, x_max_pix)
        y_pixels = np.arange(y_min_pix, y_max_pix)
        
        # WCS変換（astropy Quantityとして単位を付加）
        x_mesh, y_mesh = np.meshgrid(x_pixels, y_pixels)
        world_coords = hmi_map.pixel_to_world(x_mesh * u.pixel, y_mesh * u.pixel)
        
        # arcsecond単位の座標
        x_arcsec = world_coords.Tx.to(u.arcsec).value
        y_arcsec = world_coords.Ty.to(u.arcsec).value
        
        # 太陽半径単位に変換 (1 Rs = 696,340 km ≈ 959.63 arcsec at 1 AU)
        Rs_to_arcsec = 959.63
        x_Rs = x_arcsec / Rs_to_arcsec
        y_Rs = y_arcsec / Rs_to_arcsec
        
        print(f"  X範囲: {np.min(x_arcsec):.1f} ～ {np.max(x_arcsec):.1f} arcsec")
        print(f"  Y範囲: {np.min(y_arcsec):.1f} ～ {np.max(y_arcsec):.1f} arcsec")
        print(f"  X範囲: {np.min(x_Rs):.2f} ～ {np.max(x_Rs):.2f} Rs")
        print(f"  Y範囲: {np.min(y_Rs):.2f} ～ {np.max(y_Rs):.2f} Rs")
    else:
        # WCSが利用できない場合のフォールバック
        print("  WCS情報が利用できません。ピクセル座標を使用します。")
        x_arcsec = (x_pixels - center_x) * 0.5  # 仮定: 0.5 arcsec/pixel
        y_arcsec = (y_pixels - center_y) * 0.5
        Rs_to_arcsec = 959.63
        x_Rs = x_arcsec / Rs_to_arcsec
        y_Rs = y_arcsec / Rs_to_arcsec
    
    result = {
        'masked_data': masked_data,
        'x_pixels': np.arange(x_min_pix, x_max_pix),
        'y_pixels': np.arange(y_min_pix, y_max_pix),
        'x_arcsec': x_arcsec,
        'y_arcsec': y_arcsec,
        'x_Rs': x_Rs,
        'y_Rs': y_Rs,
        'pixel_bounds': (x_min_pix, x_max_pix, y_min_pix, y_max_pix),
        'hmi_map': hmi_map,
        'time': hmi_data['time']
    }
    
    return result

def create_pfss_from_hmi(hmi_map, nr=25, rss=2.5):
    """
    HMIマップからPFSS解を計算（pfsspy使用）
    
    Parameters:
    -----------
    hmi_map : sunpy.map.Map
        HMIマップオブジェクト
    nr : int
        動径方向の格子点数
    rss : float
        ソース面の高度（太陽半径単位）
        
    Returns:
    --------
    pfsspy.Output : PFSS解オブジェクト
    """
    print(f"\nPFSS解を計算中 (ソース面: {rss} Rs, 動径点数: {nr})")
    
    if not PFSSPY_AVAILABLE:
        print("pfsspy パッケージが利用できません")
        return None
    
    try:
        # HMIマップのメタデータを修正（必要に応じて）
        if 'cunit1' not in hmi_map.meta:
            hmi_map.meta['cunit1'] = 'deg'
        if 'cunit2' not in hmi_map.meta:
            hmi_map.meta['cunit2'] = 'deg'
        
        # PFSS入力オブジェクトを作成
        pfss_input = pfsspy.Input(hmi_map, nr, rss)
        print(f"  PFSS入力オブジェクト作成完了")
        
        # PFSS解を計算
        pfss_output = pfsspy.pfss(pfss_input)
        print(f"  PFSS計算完了")
        
        return pfss_output
        
    except Exception as e:
        print(f"PFSS計算エラー: {e}")
        return None

def create_masked_pfss_field(masked_hmi_data, nr=20, rss=2.5):
    """
    masked_dataからPFSS磁場を計算
    計算量削減のため小さな領域のみを計算
    
    Parameters:
    -----------
    masked_hmi_data : dict
        get_masked_hmi_data()の結果
    nr : int
        動径方向の格子点数
    rss : float
        ソース面の高度（太陽半径単位）
        
    Returns:
    --------
    dict : PFSS磁場データ
    """
    print(f"\nmasked領域でPFSS磁場を計算中 (ソース面: {rss} Rs, 動径点数: {nr})")
    
    masked_data = masked_hmi_data['masked_data']
    x_Rs = masked_hmi_data['x_Rs']
    y_Rs = masked_hmi_data['y_Rs']
    
    ny_mask, nx_mask = masked_data.shape
    print(f"  計算領域: {nx_mask} x {ny_mask} ピクセル")
    print(f"  X範囲: {np.min(x_Rs):.2f} ～ {np.max(x_Rs):.2f} Rs")
    print(f"  Y範囲: {np.min(y_Rs):.2f} ～ {np.max(y_Rs):.2f} Rs")
    
    # 動径座標
    r_coords = np.linspace(1.0, rss, nr)
    
    # 3次元グリッドを作成
    X_grid, Y_grid, R_grid = np.meshgrid(x_Rs[0, :], y_Rs[:, 0], r_coords, indexing='ij')
    
    # 球面座標に変換
    # X, Y は既に太陽半径単位
    # Z座標を計算（太陽表面からの高度）
    Z_coords = np.sqrt(np.maximum(R_grid**2 - X_grid**2 - Y_grid**2, 0))
    
    # 実際の球面座標 (r, theta, phi)
    r_sphere = np.sqrt(X_grid**2 + Y_grid**2 + Z_coords**2)
    theta_sphere = np.arccos(np.clip(Z_coords / r_sphere, -1, 1))  # コラティチュード
    phi_sphere = np.arctan2(Y_grid, X_grid)  # 方位角
    
    print(f"  球面座標範囲:")
    print(f"    r: {np.min(r_sphere):.2f} ～ {np.max(r_sphere):.2f} Rs")
    print(f"    theta: {np.min(theta_sphere)*180/np.pi:.1f} ～ {np.max(theta_sphere)*180/np.pi:.1f} deg")
    print(f"    phi: {np.min(phi_sphere)*180/np.pi:.1f} ～ {np.max(phi_sphere)*180/np.pi:.1f} deg")
    
    # 磁場成分を計算
    br = np.zeros_like(X_grid)
    btheta = np.zeros_like(X_grid)
    bphi = np.zeros_like(X_grid)
    
    print("  磁場成分を計算中...")
    
    for k in range(nr):
        r_current = r_coords[k]
        
        if k == 0:  # 表面 (r = 1.0 Rs)
            # masked_dataをそのまま使用
            br[:, :, k] = masked_data.T  # 転置してX,Y順に
        else:
            # ポテンシャル磁場の外挿
            # 単純な双極子近似: Br ∝ r^(-2)
            decay_factor = (1.0 / r_current)**2
            br[:, :, k] = masked_data.T * decay_factor
            
            # ソース面近傍では動径成分のみ
            if r_current >= rss * 0.9:
                source_factor = np.exp(-(r_current - rss*0.9) / (rss*0.1))
                br[:, :, k] *= source_factor
        
        # θ, φ成分（簡略計算）
        # ポテンシャル磁場では ∇ × B = 0
        # 簡略版: 表面磁場の勾配から推定
        if k > 0:
            # 表面磁場の勾配を利用
            grad_x = np.gradient(masked_data.T, axis=0)
            grad_y = np.gradient(masked_data.T, axis=1)
            
            # 球面座標での成分に変換
            sin_theta = np.sin(theta_sphere[:, :, k])
            cos_theta = np.cos(theta_sphere[:, :, k])
            sin_phi = np.sin(phi_sphere[:, :, k])
            cos_phi = np.cos(phi_sphere[:, :, k])
            
            # 簡略変換（正確ではないが近似として）
            btheta[:, :, k] = 0.1 * (grad_x * cos_theta * cos_phi + grad_y * cos_theta * sin_phi) / r_current
            bphi[:, :, k] = 0.1 * (-grad_x * sin_phi + grad_y * cos_phi) / (r_current * sin_theta + 1e-10)
    
    print("  磁場計算完了")
    
    result = {
        'br': br,
        'btheta': btheta,
        'bphi': bphi,
        'X_grid': X_grid,
        'Y_grid': Y_grid,
        'Z_coords': Z_coords,
        'r_coords': r_coords,
        'r_sphere': r_sphere,
        'theta_sphere': theta_sphere,
        'phi_sphere': phi_sphere,
        'nx': nx_mask,
        'ny': ny_mask,
        'nr': nr,
        'source_surface': rss,
        'masked_hmi_data': masked_hmi_data
    }
    
    return result

def trace_pfss_fieldlines(pfss_output, aia_map, n_lines=25):
    """
    PFSS解から磁力線をトレース（pfsspy使用）
    
    Parameters:
    -----------
    pfss_output : pfsspy.Output
        PFSS解オブジェクト
    aia_map : sunpy.map.Map
        AIAマップオブジェクト（座標系参照用）
    n_lines : int
        磁力線の本数
        
    Returns:
    --------
    list : トレースされた磁力線のリスト
    """
    print(f"\n磁力線をトレース中 (本数: {n_lines})")
    
    if not PFSSPY_AVAILABLE:
        print("pfsspy パッケージが利用できません")
        return []
    
    try:
        # 磁力線の開始点を定義（AIA観測領域内）
        # AIAマップの座標系を使用
        hp_lon = np.linspace(-600, 600, int(np.sqrt(n_lines))) * u.arcsec
        hp_lat = np.linspace(-600, 600, int(np.sqrt(n_lines))) * u.arcsec
        
        # 2Dグリッドを作成
        lon_grid, lat_grid = np.meshgrid(hp_lon, hp_lat)
        
        # SkyCoordオブジェクトを作成
        seeds = SkyCoord(lon_grid.ravel(), lat_grid.ravel(),
                        frame=aia_map.coordinate_frame)
        
        print(f"  開始点数: {len(seeds)}")
        print(f"  座標系: {aia_map.coordinate_frame}")
        
        # トレーサーを初期化
        tracer = tracing.FortranTracer()
        
        # 磁力線をトレース
        fieldlines = tracer.trace(seeds, pfss_output)
        
        print(f"  磁力線トレース完了: {len(fieldlines)} 本")
        
        return fieldlines
        
    except Exception as e:
        print(f"磁力線トレースエラー: {e}")
        import traceback
        traceback.print_exc()
        return []

def trace_fieldlines_3d(pfss_data, n_lines=20, max_steps=500):
    """
    3次元磁力線をトレース（ちぢれた磁力線の除去フィルタ付き）
    
    Parameters:
    -----------
    pfss_data : dict
        create_masked_pfss_field()の結果
    n_lines : int
        磁力線の本数
    max_steps : int
        最大ステップ数
        
    Returns:
    --------
    list : 磁力線の座標リスト
    """
    print(f"\n磁力線をトレース中 (本数: {n_lines}, 最大ステップ: {max_steps})")
    
    br = pfss_data['br']
    btheta = pfss_data['btheta']
    bphi = pfss_data['bphi']
    X_grid = pfss_data['X_grid']
    Y_grid = pfss_data['Y_grid']
    Z_coords = pfss_data['Z_coords']
    
    # 開始点を選択（表面の強い磁場領域）
    surface_field = np.abs(br[:, :, 0])
    
    # 閾値以上の強い磁場領域を特定
    threshold = np.percentile(surface_field, 85)  # 上位15%
    strong_field_mask = surface_field > threshold
    
    # 開始点の候補を取得
    strong_indices = np.where(strong_field_mask)
    
    if len(strong_indices[0]) == 0:
        print("  警告: 強い磁場領域が見つかりません。ランダム点を使用します。")
        strong_indices = (np.random.randint(0, br.shape[0], n_lines),
                         np.random.randint(0, br.shape[1], n_lines))
    
    # 開始点をサンプリング
    n_candidates = len(strong_indices[0])
    if n_candidates > n_lines:
        sample_indices = np.random.choice(n_candidates, n_lines, replace=False)
        start_x_idx = strong_indices[0][sample_indices]
        start_y_idx = strong_indices[1][sample_indices]
    else:
        start_x_idx = strong_indices[0]
        start_y_idx = strong_indices[1]
        n_lines = n_candidates
    
    print(f"  実際の磁力線数: {n_lines}")
    
    fieldlines = []
    
    for i in range(n_lines):
        print(f"  磁力線 {i+1}/{n_lines} をトレース中...", end='\r')
        
        # 開始点
        x_idx, y_idx = start_x_idx[i], start_y_idx[i]
        
        # 磁力線の座標を格納
        line_x = [X_grid[x_idx, y_idx, 0]]
        line_y = [Y_grid[x_idx, y_idx, 0]]
        line_z = [Z_coords[x_idx, y_idx, 0]]
        
        # 現在位置
        current_x = X_grid[x_idx, y_idx, 0]
        current_y = Y_grid[x_idx, y_idx, 0]
        current_z = Z_coords[x_idx, y_idx, 0]
        
        # ステップサイズ
        step_size = 0.01  # Rs単位
        
        for step in range(max_steps):
            # 現在位置での磁場を補間
            try:
                # グリッド内の位置を特定
                x_indices = np.interp(current_x, X_grid[:, 0, 0], np.arange(br.shape[0]))
                y_indices = np.interp(current_y, Y_grid[0, :, 0], np.arange(br.shape[1]))
                
                # 現在の半径を計算
                current_r = np.sqrt(current_x**2 + current_y**2 + current_z**2)
                r_indices = np.interp(current_r, pfss_data['r_coords'], np.arange(br.shape[2]))
                
                # グリッド範囲外の場合は終了
                if (x_indices < 0 or x_indices >= br.shape[0]-1 or
                    y_indices < 0 or y_indices >= br.shape[1]-1 or
                    r_indices < 0 or r_indices >= br.shape[2]-1):
                    break
                
                # 線形補間で磁場を取得
                x_int = int(x_indices)
                y_int = int(y_indices)
                r_int = int(r_indices)
                
                # 安全性チェック
                if (x_int + 1 >= br.shape[0] or y_int + 1 >= br.shape[1] or 
                    r_int + 1 >= br.shape[2]):
                    break
                
                # 単純な最近傍補間
                bx_local = br[x_int, y_int, r_int]
                by_local = btheta[x_int, y_int, r_int] 
                bz_local = bphi[x_int, y_int, r_int]
                
                # 磁場が十分小さい場合は終了
                b_magnitude = np.sqrt(bx_local**2 + by_local**2 + bz_local**2)
                if b_magnitude < 1e-6:
                    break
                
                # 次のステップを計算
                dx = step_size * bx_local / b_magnitude
                dy = step_size * by_local / b_magnitude
                dz = step_size * bz_local / b_magnitude
                
                current_x += dx
                current_y += dy
                current_z += dz
                
                # 座標を記録
                line_x.append(current_x)
                line_y.append(current_y)
                line_z.append(current_z)
                
                # ソース面に到達したら終了
                if current_r > pfss_data['source_surface']:
                    break
                
                # 太陽表面より内側に入ったら終了
                if current_r < 1.0:
                    break
                
            except (IndexError, ValueError):
                break
        
        # 磁力線の品質チェック（ちぢれた磁力線を除去）
        coords = np.array([line_x, line_y, line_z]).T
        
        # フィルタリング条件
        is_valid = True
        
        # 1. 最小点数チェック
        if len(coords) < 10:
            is_valid = False
        
        # 2. 開始点の半径チェック
        start_r = np.sqrt(line_x[0]**2 + line_y[0]**2 + line_z[0]**2)
        if start_r < 0.98 or start_r > 1.02:
            is_valid = False
        
        # 3. 磁力線の長さチェック
        if len(coords) > 1:
            total_length = np.sum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)))
            if total_length < 0.5:  # 太陽半径の半分以下は除外
                is_valid = False
        
        # 4. 曲率チェック（急激な方向変化を検出）
        if len(coords) > 2:
            vectors = np.diff(coords, axis=0)
            if len(vectors) > 1:
                # 隣接するベクトル間の角度を計算
                dot_products = np.sum(vectors[:-1] * vectors[1:], axis=1)
                magnitudes = np.sqrt(np.sum(vectors[:-1]**2, axis=1)) * np.sqrt(np.sum(vectors[1:]**2, axis=1))
                # ゼロ除算を避ける
                valid_mask = magnitudes > 1e-10
                if np.any(valid_mask):
                    cos_angles = dot_products[valid_mask] / magnitudes[valid_mask]
                    cos_angles = np.clip(cos_angles, -1, 1)
                    angles = np.arccos(cos_angles)
                    max_curvature = np.max(angles)
                    if max_curvature > np.pi/2:  # 90度以上の急激な曲がりは除外
                        is_valid = False
        
        # 5. 異常な座標値チェック
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            is_valid = False
        
        # 6. 半径の異常値チェック
        radii = np.sqrt(np.sum(coords**2, axis=1))
        if np.any(radii < 0.5) or np.any(radii > pfss_data['source_surface'] * 1.2):
            is_valid = False
        
        # 有効な磁力線のみを追加
        if is_valid:
            fieldlines.append({
                'x': np.array(line_x),
                'y': np.array(line_y),
                'z': np.array(line_z),
                'start_field': br[x_idx, y_idx, 0]
            })
    
    print(f"\n  磁力線トレース完了: {len(fieldlines)} 本（フィルタリング後）")
    
    return fieldlines

def plot_aia_with_fieldlines(aia_map, hmi_map, fieldlines, save_filename=None):
    """
    AIAマップ上にPFSS磁力線を重ねてプロット
    plot_aia_overplotting.pyの手法を参考にしたWCS座標系の正確な処理
    
    Parameters:
    -----------
    aia_map : sunpy.map.Map
        AIAマップオブジェクト
    hmi_map : sunpy.map.Map
        HMIマップオブジェクト
    fieldlines : list
        トレースされた磁力線のリスト
    save_filename : str, optional
        保存ファイル名
    """
    print("\nAIA + PFSS磁力線プロットを作成中...")
    
    # 2x2のサブプロット
    fig = plt.figure(figsize=(16, 12))
    
    # 1. HMI磁場マップ
    ax1 = fig.add_subplot(221, projection=hmi_map)
    hmi_map.plot(axes=ax1)
    plt.colorbar(ax1.images[0], ax=ax1, orientation='vertical', shrink=0.8, label='Br (Gauss)')
    ax1.set_title('HMI Radial Magnetic Field')
    
    # HMI上に磁力線を重ねる
    for fline in fieldlines:
        try:
            ax1.plot_coord(fline.coords, color='black', linewidth=0.8, alpha=0.7)
        except:
            pass
    
    # HMIデータの拡張範囲を計算
    ny, nx = hmi_map.data.shape
    center_x, center_y = nx // 2, ny // 2
    x_min_pix, x_max_pix = center_x - 1024, center_x + 0
    y_min_pix, y_max_pix = center_y - 100, center_y + 512
    
    # HMI座標系での表示範囲をarcsecに変換
    try:
        # ピクセル座標の四隅を定義
        corners_x = np.array([x_min_pix, x_max_pix, x_max_pix, x_min_pix])
        corners_y = np.array([y_min_pix, y_min_pix, y_max_pix, y_max_pix])
        
        # WCS変換でarcsec座標を取得
        world_corners = hmi_map.pixel_to_world(corners_x * u.pixel, corners_y * u.pixel)
        
        # 表示範囲を取得
        x_range_arcsec = [world_corners.Tx.to(u.arcsec).value.min(), 
                         world_corners.Tx.to(u.arcsec).value.max()]
        y_range_arcsec = [world_corners.Ty.to(u.arcsec).value.min(), 
                         world_corners.Ty.to(u.arcsec).value.max()]
        
        print(f"  拡張表示範囲: X={x_range_arcsec[0]:.0f}～{x_range_arcsec[1]:.0f} arcsec, Y={y_range_arcsec[0]:.0f}～{y_range_arcsec[1]:.0f} arcsec")
        
    except Exception as e:
        print(f"  座標変換エラー: {e}")
        # フォールバック: デフォルト範囲
        x_range_arcsec = [-1200, 400]
        y_range_arcsec = [-400, 1200]
    
    # 2. AIA画像のみ（拡張範囲）
    ax2 = fig.add_subplot(222, projection=aia_map)
    aia_map.plot(axes=ax2)
    plt.colorbar(ax2.images[0], ax=ax2, orientation='vertical', shrink=0.8, label='Intensity')
    ax2.set_title(f'AIA {aia_map.wavelength} ({aia_map.date.strftime("%Y-%m-%d %H:%M:%S")})')
    
    # ax2の表示範囲を拡張
    ax2.set_xlim(x_range_arcsec)
    ax2.set_ylim(y_range_arcsec)
    
    # 3. AIA RGB + 磁力線重ね合わせ（plot_aia_overplotting.pyの手法を使用）
    # AIA RGB画像を生成（HMIマップの太陽半径に合わせてリスケール）
    target_time = aia_map.date.strftime("%Y-%m-%d %H:%M")
    rgb_image, rgb_ref_map, rgb_success = create_aia_rgb_image_from_overlay(target_time, hmi_map=hmi_map)
    
    if rgb_success and rgb_image is not None and rgb_ref_map is not None:
        # plot_aia_overplotting.pyと同じ方法でAIA RGB画像のWCS座標系を使用
        ax3 = fig.add_subplot(223, projection=rgb_ref_map)
        
        # RGB画像をAIA座標系で表示
        ax3.imshow(rgb_image, origin='lower', aspect='equal')
        
        # AIA座標系でのリムとグリッドを描画
        try:
            rgb_ref_map.draw_limb(axes=ax3, color='white', linestyle='dashed', linewidth=1.2)
        except:
            pass
        
        try:
            rgb_ref_map.draw_grid(axes=ax3, grid_spacing=15*u.deg, color='white', 
                                 linestyle='dotted', linewidth=0.8, alpha=0.7)
        except:
            pass
        
        ax3.set_title('AIA RGB (211/193/171) + PFSS Magnetic Field Lines')
        
        # 磁力線をAIA RGB上に重ねる（plot_aia_overplotting.pyと同じ方法）
        for fline in fieldlines:
            try:
                ax3.plot_coord(fline.coords, alpha=0.8, linewidth=1.2, color='white')
            except:
                pass
        
        # ax3の表示範囲を拡張
        ax3.set_xlim(x_range_arcsec)
        ax3.set_ylim(y_range_arcsec)
        
    else:
        # RGB画像生成に失敗した場合は元のAIA画像を使用
        print("  RGB画像生成に失敗、元のAIA画像を使用します")
        ax3 = fig.add_subplot(223, projection=aia_map)
        aia_map.plot(axes=ax3)
        ax3.set_title('AIA + PFSS Magnetic Field Lines (Extended View)')
        
        # 磁力線をAIA上に重ねる
        for fline in fieldlines:
            try:
                ax3.plot_coord(fline.coords, alpha=0.8, linewidth=1.2, color='white')
            except:
                pass
        
        # ax3の表示範囲を拡張
        ax3.set_xlim(x_range_arcsec)
        ax3.set_ylim(y_range_arcsec)
    
    # 軸ラベルを設定
    ax3.set_xlabel('Solar X (arcsec)')
    ax3.set_ylabel('Solar Y (arcsec)')
    
    # 4. 磁力線の統計情報
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    # 磁力線の統計を計算
    open_lines = 0
    closed_lines = 0
    line_lengths = []
    
    for fline in fieldlines:
        try:
            coords = fline.coords
            if hasattr(coords, 'radius'):
                start_r = coords.radius[0].to(u.Rsun).value
                end_r = coords.radius[-1].to(u.Rsun).value
                line_lengths.append(len(coords))
                
                # 開いた/閉じた磁力線の判定
                if start_r < 1.1 and end_r > 2.0:  # 太陽表面から外に向かう
                    open_lines += 1
                elif start_r < 1.1 and end_r < 1.1:  # 両端が太陽表面近く
                    closed_lines += 1
        except:
            pass
    
    stats_text = f"""
PFSS磁力線解析結果

データ情報:
• HMI観測時刻: {hmi_map.date.strftime("%Y-%m-%d %H:%M:%S")}
• AIA観測時刻: {aia_map.date.strftime("%Y-%m-%d %H:%M:%S")}
• AIA波長: {aia_map.wavelength}

磁力線統計:
• 総磁力線数: {len(fieldlines)}
• 開いた磁力線: {open_lines}
• 閉じた磁力線: {closed_lines}
• 平均長: {np.mean(line_lengths):.1f} 点
• 最大長: {np.max(line_lengths) if line_lengths else 0} 点

PFSS設定:
• ソース面: 2.5 Rs
• 動径格子点: 25
"""
    
    ax4.text(0.05, 0.95, stats_text, fontsize=11, fontfamily='monospace',
             transform=ax4.transAxes, verticalalignment='top')
    
    # メインタイトル
    fig.suptitle('SDO/HMI PFSS Analysis with AIA Overlay', fontsize=16)
    
    plt.tight_layout()
    
    # 保存
    if save_filename is None:
        save_filename = '/mnt/d/wsl/home/kinno-7010/Research/PFSS/hmi_aia_pfss_analysis.png'
    
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"  AIA+PFSS プロットを保存: {save_filename}")
    
    plt.show()
    
    return fig

def plot_3d_fieldlines(pfss_data, fieldlines, save_filename=None):
    """
    3次元磁力線をプロット
    
    Parameters:
    -----------
    pfss_data : dict
        PFSS磁場データ
    fieldlines : list
        磁力線データ
    save_filename : str, optional
        保存ファイル名
    """
    print("\n3次元磁力線プロットを作成中...")
    
    fig = plt.figure(figsize=(15, 12))
    
    # 2x2のサブプロット
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224)
    
    axes_3d = [ax1, ax2, ax3]
    view_angles = [(30, 45), (0, 0), (60, 135)]
    view_titles = ['Standard View', 'Front View', 'Side View']
    
    # 色の設定
    colors = plt.cm.RdBu_r(np.linspace(0, 1, len(fieldlines)))
    
    for ax, (elev, azim), title in zip(axes_3d, view_angles, view_titles):
        # 太陽表面の球体を描画
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='yellow')
        
        # ソース面の球体
        rss = pfss_data['source_surface']
        ax.plot_surface(x_sphere*rss, y_sphere*rss, z_sphere*rss, 
                       alpha=0.1, color='gray')
        
        # 磁力線を描画
        for i, line in enumerate(fieldlines):
            field_strength = np.abs(line['start_field'])
            linewidth = 1 + 2 * field_strength / np.max([l['start_field'] for l in fieldlines])
            
            ax.plot(line['x'], line['y'], line['z'], 
                   color=colors[i], linewidth=linewidth, alpha=0.8)
        
        # 軸設定
        max_range = rss * 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, max_range])
        
        ax.set_xlabel('X (Rs)')
        ax.set_ylabel('Y (Rs)')
        ax.set_zlabel('Z (Rs)')
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
    
    # 4番目のプロットに表面磁場を表示
    masked_data = pfss_data['masked_hmi_data']['masked_data']
    x_Rs = pfss_data['masked_hmi_data']['x_Rs']
    y_Rs = pfss_data['masked_hmi_data']['y_Rs']
    
    im = ax4.imshow(masked_data, extent=[np.min(x_Rs), np.max(x_Rs), 
                                       np.min(y_Rs), np.max(y_Rs)],
                   cmap='RdBu_r', origin='lower', aspect='equal')
    
    # 磁力線の開始点をプロット
    for line in fieldlines:
        ax4.plot(line['x'][0], line['y'][0], 'ko', markersize=3)
    
    ax4.set_title('Surface Magnetic Field & Field Line Start Points')
    ax4.set_xlabel('X (Rs)')
    ax4.set_ylabel('Y (Rs)')
    
    # カラーバー
    cbar = fig.colorbar(im, ax=ax4, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Br (Gauss)')
    
    # メインタイトル
    time_str = pfss_data['masked_hmi_data']['time']
    fig.suptitle(f'PFSS 3D Magnetic Field Lines (Masked Region)\n{time_str}', 
                fontsize=14)
    
    plt.tight_layout()
    
    # 保存
    if save_filename is None:
        save_filename = '/mnt/d/wsl/home/kinno-7010/Research/PFSS/pfss_hmi_3d_fieldlines.png'
    
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"  3次元プロットを保存: {save_filename}")
    
    plt.show()
    
    return fig

def main():
    """
    メイン実行関数
    """
    print("=== SDO/HMI PFSS解析 + AIA重ね合わせ ===")
    print("pfsspy使用版")
    print("=" * 50)
    
    if not ASTROPY_AVAILABLE or not PFSSPY_AVAILABLE:
        print("Error: 必要なパッケージが不足しています")
        print("pip install astropy sunpy pfsspy")
        return
    
    if not HMI_MODULE_AVAILABLE:
        print("Warning: HMI解析モジュールが見つかりません（直接読み込みを試行）")
    
    try:
        # 1. AIAデータの読み込み
        print("\n=== Step 1: AIAデータ読み込み ===")
        aia_map = load_aia_data()
        if aia_map is None:
            print("AIAデータの読み込みに失敗しました")
            return
        
        # 2. HMIデータの読み込み
        print("\n=== Step 2: HMIデータ読み込み ===")
        hmi_file = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
        
        if HMI_MODULE_AVAILABLE:
            # HMI解析モジュール使用
            hmi_data = read_hmi_quick(hmi_file)
            hmi_map = hmi_data.get('sunpy_map')
            if hmi_map is None:
                print("HMIマップの作成に失敗しました")
                return
        else:
            # 直接読み込み
            hmi_map = sunpy.map.Map(hmi_file)
        
        print(f"  HMIマップ読み込み完了")
        print(f"  観測時刻: {hmi_map.date}")
        print(f"  データ形状: {hmi_map.data.shape}")
        
        # 3. PFSS解の計算
        print("\n=== Step 3: PFSS解計算 ===")
        pfss_output = create_pfss_from_hmi(hmi_map, nr=25, rss=2.5)
        if pfss_output is None:
            print("PFSS計算に失敗しました")
            return
        
        # 4. 磁力線のトレース
        print("\n=== Step 4: 磁力線トレース ===")
        fieldlines = trace_pfss_fieldlines(pfss_output, aia_map, n_lines=25)
        if not fieldlines:
            print("磁力線トレースに失敗しました")
            return
        
        # 5. AIA + PFSS プロット
        print("\n=== Step 5: AIA + PFSS プロット ===")
        fig_aia = plot_aia_with_fieldlines(aia_map, hmi_map, fieldlines)
        
        # 6. オプション: 従来の3次元プロット（masked_data使用）
        print("\n=== Step 6: 3次元プロット（オプション） ===")
        try:
            if HMI_MODULE_AVAILABLE:
                masked_hmi_data = get_masked_hmi_data(hmi_file)
                if masked_hmi_data is not None:
                    pfss_data_3d = create_masked_pfss_field(masked_hmi_data, nr=15, rss=2.5)
                    fieldlines_3d = trace_fieldlines_3d(pfss_data_3d, n_lines=15, max_steps=300)
                    fig_3d = plot_3d_fieldlines(pfss_data_3d, fieldlines_3d)
                    print("  3次元プロットも作成しました")
                else:
                    print("  3次元プロットはスキップしました")
            else:
                print("  HMI解析モジュールが利用できないため3次元プロットはスキップしました")
        except Exception as e3d:
            print(f"  3次元プロットでエラー: {e3d}")
        
        print("\n=== 処理完了 ===")
        print("✓ SDO/HMI PFSS解析 + AIA重ね合わせが完了しました")
        
        # 統計情報
        print(f"\n統計情報:")
        print(f"  HMI観測時刻: {hmi_map.date}")
        print(f"  AIA観測時刻: {aia_map.date}")
        print(f"  AIA波長: {aia_map.wavelength}")
        print(f"  PFSS磁力線数: {len(fieldlines)}")
        print(f"  PFSS ソース面: 2.5 Rs")
        print(f"  PFSS 動径格子点: 25")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()