#!/usr/bin/env python3
"""
STEREO-A/SECCHI/COR1 差分画像プロッター (改良版)
0を確実に白で表示するための修正を加えたバージョン
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.map
import warnings

# カスタムモジュールのインポート
try:
    from cor_prep import CORPrep
except ImportError:
    print("エラー: 'cor_prep.py' が同じディレクトリに存在することを確認してください。")
    exit()

warnings.filterwarnings('ignore', category=sunpy.map.mapbase.SunpyUserWarning)

BASE_TIME_FILE = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/20220613_020136_n4c1A.fts"


def analyze_data_distribution(data):
    """
    データの分布を詳細に分析し、適切なスケーリングパラメータを決定する
    
    Parameters:
    -----------
    data : numpy.ndarray
        分析対象のデータ配列
        
    Returns:
    --------
    dict : 分析結果を含む辞書
    """
    # NaNを除外した統計値の計算
    valid_data = data[~np.isnan(data)]
    
    stats = {
        'mean': np.mean(valid_data),
        'median': np.median(valid_data),
        'std': np.std(valid_data),
        'min': np.min(valid_data),
        'max': np.max(valid_data),
        'percentiles': {
            1: np.percentile(valid_data, 1),
            5: np.percentile(valid_data, 5),
            95: np.percentile(valid_data, 95),
            99: np.percentile(valid_data, 99)
        }
    }
    
    # 対称的なスケーリング範囲の決定
    # 99パーセンタイルを基準に、0を中心とした対称範囲を設定
    abs_max = max(abs(stats['percentiles'][1]), abs(stats['percentiles'][99]))
    stats['symmetric_range'] = abs_max
    
    # データの偏りを評価
    stats['skewness'] = stats['mean'] / stats['std'] if stats['std'] > 0 else 0
    
    return stats


def create_symmetric_colormap_norm(data, method='percentile', percentile=99, min_range=5):
    """
    0を中心とした対称的なカラーマップ正規化を作成
    
    Parameters:
    -----------
    data : numpy.ndarray
        正規化対象のデータ
    method : str
        正規化方法 ('percentile', 'std', 'fixed')
    percentile : float
        パーセンタイル法で使用するパーセンタイル値
    min_range : float
        最小範囲
        
    Returns:
    --------
    TwoSlopeNorm : 0を中心とした正規化オブジェクト
    """
    stats = analyze_data_distribution(data)
    
    if method == 'percentile':
        # パーセンタイルベースの対称範囲
        vmax = np.percentile(np.abs(data[~np.isnan(data)]), percentile)
        vmin = -vmax
    elif method == 'std':
        # 標準偏差ベースの範囲（3σ）
        vmax = 3 * stats['std']
        vmin = -vmax
    elif method == 'fixed':
        # 固定範囲
        vmax = 5
        vmin = -5
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 最小範囲の確保
    if vmax < min_range:
        vmax = min_range
        vmin = -min_range
    
    # TwoSlopeNormを使用して0を中心に固定
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    print(f"カラースケーリング設定 ({method}法):")
    print(f"  vmin={vmin:.2f}, vcenter=0.00, vmax={vmax:.2f}")
    print(f"  データ平均={stats['mean']:.3f}, 中央値={stats['median']:.3f}")
    
    return norm, vmin, vmax


def load_and_preprocess_data(fits_filepath):
    """既存の関数をそのまま使用"""
    if not os.path.exists(fits_filepath):
        print(f"エラー: 指定されたFITSファイルが見つかりません: {fits_filepath}")
        return None, None

    print(f"データ読み込み・前処理: {os.path.basename(fits_filepath)}")

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
        return None, None

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

    return processed_data, processed_header


def create_cor1_difference_plot(target_fits_filepath, output_dir=".", base_time_file=BASE_TIME_FILE, 
                              colormap_method='percentile', apply_median_correction=False):
    """
    改良版: 0を確実に白で表示する差分プロット作成
    
    Parameters:
    -----------
    target_fits_filepath : str
        差分を計算する対象のCOR1 FITSファイルのパス
    output_dir : str
        出力ディレクトリ
    base_time_file : str
        Base timeファイルのパス
    colormap_method : str
        カラーマップ正規化方法 ('percentile', 'std', 'fixed')
    apply_median_correction : bool
        中央値補正を適用するかどうか
    """
    # データの読み込みと前処理
    print("=" * 60)
    print("Base timeデータの処理開始...")
    base_data, base_header = load_and_preprocess_data(base_time_file)
    if base_data is None:
        return None

    print("\n対象データの処理開始...")
    target_data, target_header = load_and_preprocess_data(target_fits_filepath)
    if target_data is None:
        return None

    # データ形状の調整
    print("\n差分計算の実行...")
    if base_data.shape != target_data.shape:
        print(f"警告: データの形状が異なります。Base: {base_data.shape}, Target: {target_data.shape}")
        min_shape = tuple(min(b, t) for b, t in zip(base_data.shape, target_data.shape))
        base_data = base_data[:min_shape[0], :min_shape[1]]
        target_data = target_data[:min_shape[0], :min_shape[1]]
        print(f"データを共通サイズにトリミング: {min_shape}")

    # 差分計算
    diff_data = target_data.astype(np.float64) - base_data.astype(np.float64)
    
    # オプション: 中央値補正（系統的バイアスの除去）
    if apply_median_correction:
        median_offset = np.nanmedian(diff_data)
        print(f"\n中央値補正を適用: offset={median_offset:.3f}")
        diff_data -= median_offset
    
    stats = analyze_data_distribution(diff_data)
    print(f"\n差分データ統計:")
    print(f"  平均={stats['mean']:.3f}, 中央値={stats['median']:.3f}, 標準偏差={stats['std']:.3f}")
    print(f"  最小={stats['min']:.3f}, 最大={stats['max']:.3f}")

    # sunpy.map.Mapオブジェクトの作成
    print("\nsunpy.map.Mapオブジェクトの作成...")
    diff_header = target_header.copy()
    diff_header.add_history("DIFFERENCE IMAGE: Target - Base")
    diff_header.add_history(f"Base time file: {os.path.basename(base_time_file)}")
    diff_header.add_history(f"Target time file: {os.path.basename(target_fits_filepath)}")
    if apply_median_correction:
        diff_header.add_history(f"Median correction applied: {median_offset:.3f}")
    
    diff_map = sunpy.map.Map((diff_data, diff_header))
    diff_map = diff_map.rotate()
    
    # カラーマップと正規化の設定
    print("\nカラーマップの設定...")
    cmap = plt.get_cmap('RdBu_r')
    norm, vmin, vmax = create_symmetric_colormap_norm(diff_data, method=colormap_method)
    
    # プロット作成
    print("\nプロット作成...")
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(projection=diff_map)
    
    # normパラメータを明示的に指定して差分画像をプロット
    im = diff_map.plot(axes=ax, cmap=cmap, norm=norm, title=False)
    
    # カラーバーの追加（目盛りを対称的に配置）
    # cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
    # cbar.set_label(f"Difference Intensity [{diff_map.unit}]", fontsize=12)
    
    # カラーバーの目盛りを対称的に設定
    tick_interval = max(1, int(vmax / 5))  # 適切な間隔を計算
    ticks = np.arange(-int(vmax), int(vmax) + 1, tick_interval)
    ticks = np.append(ticks[ticks < 0], [0] + list(ticks[ticks > 0]))  # 0を確実に含める
    # cbar.set_ticks(ticks)
    
    # 太陽の輪郭と座標グリッド
    diff_map.draw_limb(color='yellow', linewidth=2, alpha=0.9)
    diff_map.draw_grid(grid_spacing=(15, 15) * u.deg, color='black', alpha=0.6, linestyle=':', linewidth=0.8)
    
    # 太陽半径の円
    sun_center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=diff_map.coordinate_frame)
    for radius_factor in [1, 2, 3, 4]:
        radius_coord = radius_factor * diff_map.rsun_obs
        circle_coord = SkyCoord(
            radius_coord * np.cos(np.linspace(0, 2*np.pi, 100)),
            radius_coord * np.sin(np.linspace(0, 2*np.pi, 100)),
            frame=diff_map.coordinate_frame
        )
        circle_pix = diff_map.world_to_pixel(circle_coord)
        ax.plot(circle_pix.x.value, circle_pix.y.value, 
               color='black', linewidth=1.5, alpha=0.8, linestyle='--')
    
    # タイトルとアノテーション
    target_time_str = diff_map.date.strftime('%Y-%m-%d %H:%M:%S UT')
    try:
        with fits.open(base_time_file) as hdul:
            base_time_str = hdul[0].header['DATE-OBS']
    except:
        base_time_str = "Base Time"
    
    title_line1 = f'STEREO-A/SECCHI/COR1 - Difference Image'
    title_line2 = f'Target: {target_time_str}'
    title_line3 = f'Base: {base_time_str}'

    full_title = f'{title_line1}\n{title_line2}\n{title_line3}'
    
    ax.set_title(full_title, fontsize=16)
    ax.set_xlabel('Solar X [arcsec]', fontsize=14)
    ax.set_ylabel('Solar Y [arcsec]', fontsize=14)
    
    # 画像の保存
    print("\nプロットの保存...")
    base_name = os.path.splitext(os.path.basename(target_fits_filepath))[0]
    base_time_name = os.path.splitext(os.path.basename(base_time_file))[0]
    output_filename = f"{base_name}_diff_from_{base_time_name}_zerocenter.png"
    save_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"✓ 差分プロットが正常に保存されました: {save_path}")
    print("=" * 60)

    return save_path


if __name__ == '__main__':
    target_fits_file = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/20220613_032136_n4c1A.fts"
    output_directory = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1"
    
    print("=== 改良版COR1差分プロット作成プログラム ===")
    print(f"Base time: {os.path.basename(BASE_TIME_FILE)}")
    print(f"Target file: {os.path.basename(target_fits_file)}")
    
    # 異なる正規化方法を試す
    create_cor1_difference_plot(
        target_fits_file, 
        output_directory,
        colormap_method='fixed',  # 'percentile', 'std', 'fixed' から選択
        apply_median_correction=True   # 系統的バイアスの補正
    )