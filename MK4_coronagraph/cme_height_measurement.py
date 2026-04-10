"""
Mk4コロナグラフ用CME高度計測関数

このモジュールは、Mk4コロナグラフの差分画像からCMEの高度を自動的に計測する機能を提供します。
既存のcreate_single_diff_imageやcreate_integrated_image関数を参考にして作成されています。
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from tqdm import tqdm
import re
import io
import imageio.v2 as imageio
import sunpy.map
import gc
from pathlib import Path
from scipy import ndimage
from scipy.signal import find_peaks
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any


def read_fits_file(fits_path: str) -> Tuple[np.ndarray, fits.Header]:
    """FITSファイルを読み込み、データとヘッダーを返す."""
    with fits.open(fits_path, ignore_missing_end=True) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    return data, header


def calculate_statistics(data: np.ndarray, exclude_zeros: bool = False) -> Dict[str, float]:
    """データの統計情報を計算."""
    if exclude_zeros:
        data = data[data > 0]
    if data.size == 0:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
    return {
        "min": np.min(data),
        "max": np.max(data),
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data),
    }


def load_kcor_nrgf_sequence(path_or_dir: str) -> sunpy.map.MapSequence:
    """
    Mk4 KCOR NGRFデータを読み込んでMapSequenceを返す
    
    Parameters:
    -----------
    path_or_dir : str
        FITSファイルのパスまたはディレクトリパス
        
    Returns:
    --------
    sunpy.map.MapSequence
        読み込まれたMapSequence
    """
    if os.path.isdir(path_or_dir):
        fits_list = sorted(
            glob.glob(os.path.join(path_or_dir, "*.fts"))
            + glob.glob(os.path.join(path_or_dir, "*.fts.gz"))
        )
        if not fits_list:
            raise FileNotFoundError("指定ディレクトリに *.fts が見つかりません。")
        return sunpy.map.Map(fits_list, sequence=True, allow_errors=True)

    # 1 ファイルだけ渡された場合
    return sunpy.map.Map(path_or_dir, sequence=True, allow_errors=True)


def create_difference_map(base_map: sunpy.map.Map, target_map: sunpy.map.Map) -> sunpy.map.Map:
    """
    2つのマップから差分マップを作成
    
    Parameters:
    -----------
    base_map : sunpy.map.Map
        基準となるマップ
    target_map : sunpy.map.Map
        対象となるマップ
        
    Returns:
    --------
    sunpy.map.Map
        差分マップ
    """
    diff_data = target_map.data - base_map.data
    diff_map = sunpy.map.Map(diff_data, target_map.meta)
    return diff_map


def detect_cme_front_enhanced(diff_map: sunpy.map.Map,
                             prev_diff_map: Optional[sunpy.map.Map] = None,
                             threshold_factor: float = 2.0,
                             min_radius_rsun: float = 1.5,
                             max_radius_rsun: float = 6.0,
                             min_points: int = 5) -> List[float]:
    """
    差分マップからCMEフロントを検出し、複数の高度を計算
    Base Timeからの差分に加えて、前の時間との差分も使用
    
    Parameters:
    -----------
    diff_map : sunpy.map.Map
        Base Timeからの差分マップ
    prev_diff_map : Optional[sunpy.map.Map]
        前の時間との差分マップ
    threshold_factor : float
        閾値の係数（標準偏差の倍数）
    min_radius_rsun : float
        最小検出半径（太陽半径単位）
    max_radius_rsun : float
        最大検出半径（太陽半径単位）
    min_points : int
        連続する明るいピクセルの最小数
        
    Returns:
    --------
    List[float]
        検出されたCMEの高度リスト（太陽半径単位）
    """
    data = diff_map.data
    
    # 太陽中心の座標を取得
    solar_x = diff_map.meta.get('CRPIX1', data.shape[1] // 2) - 1
    solar_y = diff_map.meta.get('CRPIX2', data.shape[0] // 2) - 1
    solar_radius = diff_map.meta.get('R_SUN', 0)
    
    if solar_radius <= 0:
        print("警告: 太陽半径の情報が取得できませんでした。")
        return []
    
    # ピクセル座標から太陽半径への変換係数
    pixel_to_rsun = 1 / solar_radius
    
    # 各ピクセルの太陽中心からの距離を計算
    y, x = np.ogrid[:data.shape[0], :data.shape[1]]
    distances = np.sqrt((x - solar_x)**2 + (y - solar_y)**2) * pixel_to_rsun
    
    # 指定された半径範囲内のデータのみを対象とする
    mask = (distances >= min_radius_rsun) & (distances <= max_radius_rsun)
    
    if not np.any(mask):
        print("指定された半径範囲内にデータがありません。")
        return []
    
    # Base Timeからの差分データの統計を計算
    diff_data_masked = data[mask]
    diff_std = np.std(diff_data_masked)
    diff_mean = np.mean(diff_data_masked)
    
    # 閾値を設定
    threshold = diff_mean + threshold_factor * diff_std
    
    # 閾値を超えるピクセルを検出
    bright_pixels_base = (data >= threshold) & mask
    
    # 前の時間との差分も使用する場合
    if prev_diff_map is not None:
        prev_data = prev_diff_map.data
        prev_diff_masked = prev_data[mask]
        prev_std = np.std(prev_diff_masked)
        prev_mean = np.mean(prev_diff_masked)
        prev_threshold = prev_mean + threshold_factor * prev_std
        
        # 前の時間との差分でも閾値を超えるピクセル
        bright_pixels_prev = (prev_data >= prev_threshold) & mask
        
        # 両方で明るいピクセルの組み合わせ
        bright_pixels = bright_pixels_base | bright_pixels_prev
    else:
        bright_pixels = bright_pixels_base
    
    if not np.any(bright_pixels):
        print("閾値を超える明るいピクセルが検出されませんでした。")
        return []
    
    # 明るいピクセルの距離を取得
    bright_distances = distances[bright_pixels]
    
    # 連続する領域を検出
    labeled_array, num_features = ndimage.label(bright_pixels)
    
    cme_heights = []
    
    for i in range(1, num_features + 1):
        region_mask = (labeled_array == i)
        region_distances = distances[region_mask]
        
        # 十分な数のピクセルがある領域のみ考慮
        if len(region_distances) >= min_points:
            # その領域の最大距離をCME高度とする
            max_distance = np.max(region_distances)
            cme_heights.append(max_distance)
    
    # 距離でソート
    cme_heights.sort(reverse=True)
    
    return cme_heights


def measure_cme_height_enhanced(map_sequence: sunpy.map.MapSequence,
                               base_time_str: str,
                               output_file: Optional[str] = None,
                               threshold_factor: float = 2.0,
                               min_radius_rsun: float = 1.5,
                               max_radius_rsun: float = 6.0) -> pd.DataFrame:
    """
    マップシーケンスからCME高度を時系列で計測（拡張版）
    Base Timeからの差分を計算し、その差分データをさらに前の時間の差分データから引く（二次差分）
    
    Parameters:
    -----------
    map_sequence : sunpy.map.MapSequence
        マップシーケンス
    base_time_str : str
        基準時刻（ISO形式）
    output_file : Optional[str]
        結果を保存するファイルパス
    threshold_factor : float
        閾値の係数
    min_radius_rsun : float
        最小検出半径
    max_radius_rsun : float
        最大検出半径
        
    Returns:
    --------
    pd.DataFrame
        時刻とCME高度のデータフレーム
    """
    base_time = Time(base_time_str)
    
    # Base timeに最も近いマップを基準マップとして取得
    base_map = None
    min_time_diff = float('inf')
    for m in map_sequence:
        time_diff = abs((m.date - base_time).to(u.s).value)
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            base_map = m
    
    if base_map is None:
        print("基準マップが見つかりません。")
        return pd.DataFrame()
    
    print(f"基準時刻: {base_time.iso}")
    print(f"基準マップ時刻: {base_map.date.iso}")
    print(f"対象マップ数: {len(map_sequence)}")
    
    # 各マップからCME高度を計測
    results = []
    prev_diff_from_base = None  # 前の時間のBase差分データを保存
    
    for i, target_map in enumerate(tqdm(map_sequence, desc="CME高度計測中（拡張版・二次差分）")):
        try:
            # 1. Base Timeからの差分マップを作成
            diff_from_base = create_difference_map(base_map, target_map)
            
            # 2. Base Timeからの差分で検出
            cme_heights_base = detect_cme_front_enhanced(
                diff_from_base, 
                prev_diff_map=None,
                threshold_factor=threshold_factor,
                min_radius_rsun=min_radius_rsun,
                max_radius_rsun=max_radius_rsun
            )
            
            # Base timeからの差分結果を記録
            if cme_heights_base:
                for j, height in enumerate(cme_heights_base):
                    results.append({
                        'Time_ISO': target_map.date.iso,
                        'Height_Rsun': height,
                        'Point_Index': j + 1,
                        'Total_Points': len(cme_heights_base),
                        'Diff_Type': 'Base_Diff'
                    })
            else:
                results.append({
                    'Time_ISO': target_map.date.iso,
                    'Height_Rsun': np.nan,
                    'Point_Index': 0,
                    'Total_Points': 0,
                    'Diff_Type': 'Base_Diff'
                })
            
            # 3. 二次差分（前の時間のBase差分データから現在のBase差分データを引く）
            if prev_diff_from_base is not None:
                # 二次差分マップを作成：前の差分 - 現在の差分
                second_diff_data = prev_diff_from_base.data - diff_from_base.data
                second_diff_map = sunpy.map.Map(second_diff_data, diff_from_base.meta)
                
                # 二次差分で検出
                cme_heights_second = detect_cme_front_enhanced(
                    second_diff_map, 
                    prev_diff_map=None,
                    threshold_factor=threshold_factor,
                    min_radius_rsun=min_radius_rsun,
                    max_radius_rsun=max_radius_rsun
                )
                
                # 二次差分結果を記録
                if cme_heights_second:
                    for j, height in enumerate(cme_heights_second):
                        results.append({
                            'Time_ISO': target_map.date.iso,
                            'Height_Rsun': height,
                            'Point_Index': j + 1,
                            'Total_Points': len(cme_heights_second),
                            'Diff_Type': 'Second_Diff'
                        })
                else:
                    results.append({
                        'Time_ISO': target_map.date.iso,
                        'Height_Rsun': np.nan,
                        'Point_Index': 0,
                        'Total_Points': 0,
                        'Diff_Type': 'Second_Diff'
                    })
            
            # 次のループのために現在のBase差分マップを保存
            prev_diff_from_base = diff_from_base
                
        except Exception as e:
            print(f"時刻 {target_map.date.iso} の処理中にエラーが発生しました: {e}")
            continue
    
    # 結果をDataFrameに変換
    df = pd.DataFrame(results)
    
    if df.empty:
        print("有効なCME高度データが得られませんでした。")
        return df
    
    # 結果を保存
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"結果を {output_file} に保存しました。")
    
    return df


def plot_cme_height_evolution_enhanced(df: pd.DataFrame, 
                                     output_plot: Optional[str] = None,
                                     title: str = "Enhanced CME Height Evolution") -> None:
    """
    CME高度の時間変化をプロット（拡張版・二次差分対応）
    Base timeからの差分と二次差分を分けて表示
    
    Parameters:
    -----------
    df : pd.DataFrame
        CME高度データ
    output_plot : Optional[str]
        プロットを保存するファイルパス
    title : str
        プロットのタイトル
    """
    if df.empty:
        print("プロットするデータがありません。")
        return
    
    # NaNを除外
    df_valid = df.dropna(subset=['Height_Rsun'])
    
    if df_valid.empty:
        print("有効なCME高度データがありません。")
        return
    
    # 時刻をTimeオブジェクトに変換（スペースをTに置換してisot形式に揃える）
    time_list = [str(t).replace(' ', 'T') for t in df_valid['Time_ISO'] if pd.notnull(t)]
    times = Time(time_list, format='isot')
    
    plt.figure(figsize=(15, 12))
    
    # 2つのサブプロットを作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # Base timeからの差分データ
    df_base = df_valid[df_valid['Diff_Type'] == 'Base_Diff']
    if not df_base.empty:
        colors_base = ['blue', 'red', 'green', 'orange', 'purple']
        
        for point_idx in sorted(df_base['Point_Index'].unique()):
            if point_idx == 0:  # 検出されなかった場合はスキップ
                continue
                
            point_data = df_base[df_base['Point_Index'] == point_idx]
            point_times = Time([str(t).replace(' ', 'T') for t in point_data['Time_ISO']], format='isot')
            
            color = colors_base[(point_idx - 1) % len(colors_base)]
            ax1.scatter(point_times.datetime, point_data['Height_Rsun'], 
                       c=color, s=20, alpha=0.7, label=f'Point {point_idx}')
        
        # Base timeからの差分のトレンドライン
        max_heights_base = df_base.groupby('Time_ISO')['Height_Rsun'].max().reset_index()
        trend_times_base = Time([str(t).replace(' ', 'T') for t in max_heights_base['Time_ISO']], format='isot')
        ax1.plot(trend_times_base.datetime, max_heights_base['Height_Rsun'], 
                'k--', alpha=0.5, linewidth=1, label='Max Height Trend')
        
        ax1.set_ylabel('CME Height (R⊙)')
        ax1.set_title('Base Time差分からの検出', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 二次差分データ
    df_second = df_valid[df_valid['Diff_Type'] == 'Second_Diff']
    if not df_second.empty:
        colors_second = ['cyan', 'magenta', 'lime', 'brown', 'pink']
        
        for point_idx in sorted(df_second['Point_Index'].unique()):
            if point_idx == 0:  # 検出されなかった場合はスキップ
                continue
                
            point_data = df_second[df_second['Point_Index'] == point_idx]
            point_times = Time([str(t).replace(' ', 'T') for t in point_data['Time_ISO']], format='isot')
            
            color = colors_second[(point_idx - 1) % len(colors_second)]
            ax2.scatter(point_times.datetime, point_data['Height_Rsun'], 
                       c=color, s=20, alpha=0.7, label=f'Point {point_idx}')
        
        # 二次差分のトレンドライン
        max_heights_second = df_second.groupby('Time_ISO')['Height_Rsun'].max().reset_index()
        trend_times_second = Time([str(t).replace(' ', 'T') for t in max_heights_second['Time_ISO']], format='isot')
        ax2.plot(trend_times_second.datetime, max_heights_second['Height_Rsun'], 
                'k--', alpha=0.5, linewidth=1, label='Max Height Trend')
        
        ax2.set_ylabel('CME Height (R⊙)')
        ax2.set_title('二次差分からの検出（前の差分 - 現在の差分）', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax2.set_xlabel('Time (UTC)')
    plt.xticks(rotation=45)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if output_plot:
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"プロットを {output_plot} に保存しました。")
    
    plt.show()


def create_cme_height_measurement_movie_enhanced(map_sequence: sunpy.map.MapSequence,
                                               base_time_str: str,
                                               output_movie: str,
                                               fps: int = 10,
                                               threshold_factor: float = 2.0,
                                               min_radius_rsun: float = 1.5,
                                               max_radius_rsun: float = 6.0) -> None:
    """
    CME高度計測の過程を動画として作成（拡張版・二次差分対応）
    
    Parameters:
    -----------
    map_sequence : sunpy.map.MapSequence
        マップシーケンス
    base_time_str : str
        基準時刻
    output_movie : str
        出力動画ファイルパス
    fps : int
        フレームレート
    threshold_factor : float
        閾値の係数
    min_radius_rsun : float
        最小検出半径
    max_radius_rsun : float
        最大検出半径
    """
    base_time = Time(base_time_str)
    
    # Base timeに最も近いマップを基準マップとして取得
    base_map = None
    min_time_diff = float('inf')
    for m in map_sequence:
        time_diff = abs((m.date - base_time).to(u.s).value)
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            base_map = m
    
    if base_map is None:
        print("基準マップが見つかりません。")
        return
    
    frames = []
    prev_diff_from_base = None
    
    for i, target_map in enumerate(tqdm(map_sequence, desc="動画フレーム生成中（拡張版・二次差分）")):
        try:
            # Base Timeからの差分マップを作成
            diff_from_base = create_difference_map(base_map, target_map)
            
            # Base差分でCMEフロントを検出
            cme_heights_base = detect_cme_front_enhanced(
                diff_from_base, 
                prev_diff_map=None,
                threshold_factor=threshold_factor,
                min_radius_rsun=min_radius_rsun,
                max_radius_rsun=max_radius_rsun
            )
            
            # 二次差分でCMEフロントを検出（前の差分がある場合）
            cme_heights_second = []
            if prev_diff_from_base is not None:
                second_diff_data = prev_diff_from_base.data - diff_from_base.data
                second_diff_map = sunpy.map.Map(second_diff_data, diff_from_base.meta)
                cme_heights_second = detect_cme_front_enhanced(
                    second_diff_map, 
                    prev_diff_map=None,
                    threshold_factor=threshold_factor,
                    min_radius_rsun=min_radius_rsun,
                    max_radius_rsun=max_radius_rsun
                )
            
            # プロットを作成
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # 差分マップを表示（Base差分）
            data = diff_from_base.data
            solar_x = diff_from_base.meta.get('CRPIX1', data.shape[1] // 2) - 1
            solar_y = diff_from_base.meta.get('CRPIX2', data.shape[0] // 2) - 1
            solar_radius = diff_from_base.meta.get('R_SUN', 0)
            
            if solar_radius > 0:
                pixel_to_rsun = 1 / solar_radius
                extent = [
                    -solar_x * pixel_to_rsun, (data.shape[1] - solar_x) * pixel_to_rsun,
                    -solar_y * pixel_to_rsun, (data.shape[0] - solar_y) * pixel_to_rsun
                ]
                
                # 差分データの統計を計算して表示範囲を決定
                diff_std = np.std(data)
                diff_mean = np.mean(data)
                vmin = diff_mean - 3 * diff_std
                vmax = diff_mean + 3 * diff_std
                
                im = ax.imshow(data, cmap='gray', origin='lower', 
                             vmin=vmin, vmax=vmax, extent=extent)
                
                # 太陽中心と太陽円盤を描画
                ax.scatter(0, 0, color='red', s=50, label="Solar Center")
                circle = Circle((0, 0), 1, color='red', fill=False, lw=1.5, label="Solar Disk")
                ax.add_patch(circle)
                
                # Base差分からのCME高度検出結果を描画
                base_colors = ['yellow', 'orange', 'gold', 'khaki', 'lemonchiffon']
                if cme_heights_base:
                    for j, height in enumerate(cme_heights_base):
                        color = base_colors[j % len(base_colors)]
                        cme_circle = Circle((0, 0), height, color=color, 
                                          fill=False, lw=2, linestyle='-', 
                                          label=f"Base Diff {j+1} ({height:.2f} R⊙)")
                        ax.add_patch(cme_circle)
                
                # 二次差分からのCME高度検出結果を描画
                second_colors = ['cyan', 'magenta', 'lime', 'pink', 'lightblue']
                if cme_heights_second:
                    for j, height in enumerate(cme_heights_second):
                        color = second_colors[j % len(second_colors)]
                        cme_circle = Circle((0, 0), height, color=color, 
                                          fill=False, lw=2, linestyle='--', 
                                          label=f"2nd Diff {j+1} ({height:.2f} R⊙)")
                        ax.add_patch(cme_circle)
                
                ax.set_xlabel("X [Solar Radius R⊙]")
                ax.set_ylabel("Y [Solar Radius R⊙]")
                ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                
                title = f"Enhanced Mk4 KCOR Analysis with Second Diff\n{target_map.date.strftime('%Y-%m-%d %H:%M:%S')}"
                if cme_heights_base or cme_heights_second:
                    title += f"\nBase Diff: {len(cme_heights_base)} fronts, 2nd Diff: {len(cme_heights_second)} fronts"
                    if cme_heights_base:
                        title += f"\nMax Base Height: {max(cme_heights_base):.2f} R⊙"
                    if cme_heights_second:
                        title += f", Max 2nd Height: {max(cme_heights_second):.2f} R⊙"
                else:
                    title += "\nNo CME Fronts Detected"
                
                ax.set_title(title, fontsize=10)
            
            # フレームを保存
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()
            plt.close(fig)
            
            # 次のループのために現在のBase差分マップを保存
            prev_diff_from_base = diff_from_base
            
        except Exception as e:
            print(f"時刻 {target_map.date.iso} のフレーム生成中にエラーが発生しました: {e}")
            continue
    
    if frames:
        print(f"動画を作成中... ({len(frames)} フレーム)")
        imageio.mimwrite(output_movie, frames, fps=fps, codec='libx264', quality=8)
        print(f"動画を {output_movie} に保存しました。")
    else:
        print("有効なフレームが生成されませんでした。")


def main():
    """メイン実行関数"""
    # データパスの設定
    data_folder = r"/mnt/d/wsl/home/kinno-7010/Research_data/MK4_coronagraph/MK4_coronagraph_KCOR/Subtraction_data/Rawdata/kcor_nrgf/20220613.kcor_nrgf.fits"
    output_folder = r"/mnt/d/wsl/home/kinno-7010/Research_data/MK4_coronagraph\MK4_coronagraph_KCOR\Subtraction_data\output"
    
    # 出力フォルダを作成
    os.makedirs(output_folder, exist_ok=True)
    
    # 基準時刻の設定（CME開始時刻の推定）
    base_time_str = "2022-06-13T02:00:00"
    
    # 出力ファイル名の設定
    height_output = os.path.join(output_folder, "cme_height_mk4_20220613.csv")
    plot_output = os.path.join(output_folder, "cme_height_evolution_mk4_20220613.png")
    movie_output = os.path.join(output_folder, "cme_height_measurement_mk4_20220613.mp4")
    
    try:
        print("Mk4 KCOR NGRFデータを読み込み中...")
        map_sequence = load_kcor_nrgf_sequence(data_folder)
        print(f"読み込み完了: {len(map_sequence)} マップ")
        
        # CME高度を計測
        print("\nCME高度を計測中...")
        df = measure_cme_height_enhanced(
            map_sequence=map_sequence,
            base_time_str=base_time_str,
            output_file=height_output,
            threshold_factor=2.0,
            min_radius_rsun=1.5,
            max_radius_rsun=6.0
        )
        
        if not df.empty:
            print(f"\n計測完了: {len(df)} データポイント")
            print(f"高度範囲: {df['Height_Rsun'].min():.3f} - {df['Height_Rsun'].max():.3f} R⊙")
            
            # 高度変化をプロット
            print("\n高度変化をプロット中...")
            plot_cme_height_evolution_enhanced(
                df=df,
                output_plot=plot_output,
                title="CME Height Evolution (Mk4 KCOR NGRF)"
            )
            
            # 計測過程の動画を作成
            print("\n計測過程の動画を作成中...")
            create_cme_height_measurement_movie_enhanced(
                map_sequence=map_sequence,
                base_time_str=base_time_str,
                output_movie=movie_output,
                fps=10,
                threshold_factor=2.0,
                min_radius_rsun=1.5,
                max_radius_rsun=6.0
            )
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
