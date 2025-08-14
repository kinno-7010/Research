#!/usr/bin/env python3
"""
CME with_line統計データ専用プロットツール

このスクリプトは、cme_statistics_YYMMDD-HHMMSS_min_diff_with_line.csvファイルを使用して、
CME先端高度の時系列変化をプロットします。

機能:
- with_line統計データ専用のプロット
- ax[0]: Max height + Mean height±std
- ax[1]: Max speed + Mean speed
- 時間範囲の指定
- 太陽半径単位での高度表示

使用例:
python cme_with_line_plot.py --start_time "2022-06-13T03:00:00" --end_time "2022-06-13T03:40:00"
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import glob
import os
import argparse
from pathlib import Path

# 既存のCMEHeightPlotterクラスをインポート
from cme_height_time_plot import CMEHeightPlotter


class CMEWithLinePlotter(CMEHeightPlotter):
    """
    CME with_line統計データ専用プロッター
    
    CMEHeightPlotterクラスを継承し、with_line.csv専用の機能を提供
    """
    
    def __init__(self, csv_folder_path):
        """
        CME with_lineプロッターの初期化
        
        Parameters:
        -----------
        csv_folder_path : str
            CME統計データのCSVフォルダのパス
        """
        super().__init__(csv_folder_path)
        
    def apply_distance_correction(self, R_1, theta_s_deg=21.0, phi_s_deg=-44.0):
        """
        クリックで求めた距離R_1を球座標系補正式により実際の距離Rに変換
        
        Parameters:
        -----------
        R_1 : float or array-like
            クリックして求めた太陽中心からの距離
        theta_s_deg : float, default=21.0
            θ_s角度（度）
        phi_s_deg : float, default=-44.0
            φ_s角度（度）
            
        Returns:
        --------
        float or array-like
            補正された距離R
            
        Notes:
        ------
        補正式: R = R_1 / sqrt(sin²(θ_s) * sin²(φ_s) + cos²(θ_s))
        """
        # 度をラジアンに変換
        theta_s_rad = np.radians(theta_s_deg)
        phi_s_rad = np.radians(phi_s_deg)
        
        # 補正式の分母を計算
        denominator = np.sqrt(
            np.sin(theta_s_rad)**2 * np.sin(phi_s_rad)**2 + 
            np.cos(theta_s_rad)**2
        )
        
        # 補正された距離Rを計算
        R = R_1 / denominator
        
        return R
    
    def plot_with_line_comparison_analysis(self, start_time, end_time, save_path_base=None, figsize=(15, 8)):
        """
        with_line統計データの補正前・補正済み比較解析（別図でプロット）
        
        Parameters:
        -----------
        start_time : str
            開始時刻 (ISO format: "YYYY-MM-DDTHH:MM:SS")
        end_time : str
            終了時刻 (ISO format: "YYYY-MM-DDTHH:MM:SS")
        save_path_base : str, optional
            保存ファイルパスのベース名（自動的に_original.png、_corrected.pngが付加される）
        figsize : tuple
            図のサイズ
            
        Returns:
        --------
        tuple
            (original_height_data, corrected_height_data, fig_original, axes_original, fig_corrected, axes_corrected)
        """
        # 固定パラメータ
        theta_s_deg = 21.0
        phi_s_deg = -44.0
        
        print(f"Starting CME with_line comparison analysis: {start_time} - {end_time}")
        print(f"Distance correction parameters (fixed): θ_s={theta_s_deg}°, φ_s={phi_s_deg}°")
        
        # with_line統計データ読み込み
        print("Loading with_line statistics data...")
        with_line_df = self.load_diff_with_line_statistics_data(start_time, end_time)
        original_height_data = {}
        
        if not with_line_df.empty:
            print(f"With_line statistics data loading complete: {len(with_line_df)} records")
            original_height_data = self.extract_height_data(with_line_df)
            print(f"With_line statistics extraction complete: {len(original_height_data)} time points")
        else:
            print("No with_line statistics data found")
            print("Available files:")
            pattern = str(self.sta_folder / "cme_statistics_*_min_diff_with_line.csv")
            files = glob.glob(pattern)
            for file in files:
                print(f"  - {file}")
            return {}, {}, None, None, None, None
        
        # 距離補正を適用
        corrected_height_data = {}
        if original_height_data:
            print("Applying distance correction...")
            corrected_height_data = self.apply_height_correction_to_data(
                original_height_data, theta_s_deg, phi_s_deg
            )
            print(f"Distance correction complete: {len(corrected_height_data)} time points")
            
            # 補正係数を表示
            correction_factor = 1.0 / self.apply_distance_correction(1.0, theta_s_deg, phi_s_deg)
            print(f"Correction factor applied: {correction_factor:.4f}")
        
        # 保存パスの設定
        save_path_original = None
        save_path_corrected = None
        if save_path_base:
            if save_path_base.endswith('.png'):
                base_name = save_path_base[:-4]
            else:
                base_name = save_path_base
            save_path_original = "../speed_analysis_along_with_line_skyplane.png"
            save_path_corrected = "../speed_analysis_along_with_line_3D.png"
        
        # 補正前のプロット作成
        print("Creating original data plot...")
        fig_original, axes_original = self.plot_diff_with_line_evolution(
            original_height_data, 
            save_path=save_path_original, 
            figsize=figsize,
            title_prefix="Original Data"
        )
        
        # 補正済みのプロット作成
        print("Creating distance-corrected data plot...")
        fig_corrected, axes_corrected = self.plot_diff_with_line_evolution(
            corrected_height_data, 
            save_path=save_path_corrected, 
            figsize=figsize,
            title_prefix="Distance-Corrected Data"
        )
        
        # プロット結果の統計情報を表示
        if original_height_data and corrected_height_data:
            self._print_comparison_analysis_summary(original_height_data, corrected_height_data, theta_s_deg, phi_s_deg)
        
        print("Comparison analysis complete!")
        return original_height_data, corrected_height_data, fig_original, axes_original, fig_corrected, axes_corrected
    
    def apply_height_correction_to_data(self, height_data, theta_s_deg=21.0, phi_s_deg=-44.0):
        """
        高度データ辞書内のすべての高度値に距離補正を適用
        
        Parameters:
        -----------
        height_data : dict
            時刻をキーとした高度データの辞書
        theta_s_deg : float, default=21.0
            θ_s角度（度）
        phi_s_deg : float, default=-44.0
            φ_s角度（度）
            
        Returns:
        --------
        dict
            補正された高度データの辞書
        """
        corrected_data = {}
        
        for time_stamp, data in height_data.items():
            corrected_info = data.copy()
            
            # 各高度値に補正を適用
            for height_key in ['mean_height', 'median_height', 'max_height', 'min_height']:
                if data[height_key] is not None:
                    corrected_info[height_key] = self.apply_distance_correction(
                        data[height_key], theta_s_deg, phi_s_deg
                    )
            
            # 標準偏差値も補正（分散の性質により同じ係数で補正可能）
            for std_key in ['std_height', 'median_std_height']:
                if data[std_key] is not None:
                    corrected_info[std_key] = self.apply_distance_correction(
                        data[std_key], theta_s_deg, phi_s_deg
                    )
            
            corrected_data[time_stamp] = corrected_info
            
        return corrected_data
    
    def plot_with_line_analysis_corrected(self, start_time, end_time, theta_s_deg=21.0, phi_s_deg=-44.0, save_path=None, figsize=(15, 8)):
        """
        with_line統計データに距離補正を適用した完全解析とプロット
        
        Parameters:
        -----------
        start_time : str
            開始時刻 (ISO format: "YYYY-MM-DDTHH:MM:SS")
        end_time : str
            終了時刻 (ISO format: "YYYY-MM-DDTHH:MM:SS")
        theta_s_deg : float, default=21.0
            θ_s角度（度）
        phi_s_deg : float, default=-44.0
            φ_s角度（度）
        save_path : str, optional
            保存ファイルパス
        figsize : tuple
            図のサイズ
            
        Returns:
        --------
        tuple
            (original_height_data, corrected_height_data, fig, axes)
        """
        print(f"Starting CME with_line corrected analysis: {start_time} - {end_time}")
        print(f"Distance correction parameters: θ_s={theta_s_deg}°, φ_s={phi_s_deg}°")
        
        # 元のwith_line統計データ読み込み
        print("Loading with_line statistics data...")
        with_line_df = self.load_diff_with_line_statistics_data(start_time, end_time)
        original_height_data = {}
        
        if not with_line_df.empty:
            print(f"With_line statistics data loading complete: {len(with_line_df)} records")
            original_height_data = self.extract_height_data(with_line_df)
            print(f"With_line statistics extraction complete: {len(original_height_data)} time points")
        else:
            print("No with_line statistics data found")
            print("Available files:")
            pattern = str(self.sta_folder / "cme_statistics_*_min_diff_with_line.csv")
            files = glob.glob(pattern)
            for file in files:
                print(f"  - {file}")
        
        # 距離補正を適用
        corrected_height_data = {}
        if original_height_data:
            print("Applying distance correction...")
            corrected_height_data = self.apply_height_correction_to_data(
                original_height_data, theta_s_deg, phi_s_deg
            )
            print(f"Distance correction complete: {len(corrected_height_data)} time points")
            
            # 補正係数を表示
            correction_factor = 1.0 / self.apply_distance_correction(1.0, theta_s_deg, phi_s_deg)
            print(f"Correction factor applied: {correction_factor:.4f}")
        
        # プロット作成（補正されたデータを使用）
        print("Creating distance-corrected data plot...")
        fig, axes = self.plot_diff_with_line_evolution(
            corrected_height_data, 
            save_path=save_path, 
            figsize=figsize,
            title_prefix="Distance-Corrected Data"
        )
        
        # プロット結果の統計情報を表示
        if corrected_height_data:
            self._print_comparison_analysis_summary(original_height_data, corrected_height_data, theta_s_deg, phi_s_deg)
        
        print("Corrected with_line analysis complete!")
        return original_height_data, corrected_height_data, fig, axes
        
    def plot_with_line_analysis(self, start_time, end_time, save_path=None, figsize=(15, 8)):
        """
        with_line統計データの完全解析とプロット
        
        Parameters:
        -----------
        start_time : str
            開始時刻 (ISO format: "YYYY-MM-DDTHH:MM:SS")
        end_time : str
            終了時刻 (ISO format: "YYYY-MM-DDTHH:MM:SS")
        save_path : str, optional
            保存ファイルパス
        figsize : tuple
            図のサイズ
            
        Returns:
        --------
        tuple
            (with_line_height_data, fig, axes)
        """
        print(f"Starting CME with_line analysis: {start_time} - {end_time}")
        
        # with_line統計データ読み込み
        print("Loading with_line statistics data...")
        with_line_df = self.load_diff_with_line_statistics_data(start_time, end_time)
        with_line_height_data = {}
        
        if not with_line_df.empty:
            print(f"With_line statistics data loading complete: {len(with_line_df)} records")
            with_line_height_data = self.extract_height_data(with_line_df)
            print(f"With_line statistics extraction complete: {len(with_line_height_data)} time points")
        else:
            print("No with_line statistics data found")
            print("Available files:")
            pattern = str(self.sta_folder / "cme_statistics_*_min_diff_with_line.csv")
            files = glob.glob(pattern)
            for file in files:
                print(f"  - {file}")
        
        # プロット作成
        print("Creating original data plot...")
        fig, axes = self.plot_diff_with_line_evolution(
            with_line_height_data, 
            save_path=save_path, 
            figsize=figsize,
            title_prefix="Original Data"
        )
        
        # プロット結果の統計情報を表示
        if with_line_height_data:
            self._print_analysis_summary(with_line_height_data)
        
        print("With_line analysis complete!")
        return with_line_height_data, fig, axes
    
    def _print_comparison_analysis_summary(self, original_data, corrected_data, theta_s_deg, phi_s_deg):
        """
        距離補正解析結果の統計情報を表示
        
        Parameters:
        -----------
        original_data : dict
            元の高度データの辞書
        corrected_data : dict
            補正された高度データの辞書
        theta_s_deg : float
            θ_s角度（度）
        phi_s_deg : float
            φ_s角度（度）
        """
        if not original_data or not corrected_data:
            return
            
        print("\n=== Distance-Correction Comparison Analysis Summary ===")
        print(f"Correction parameters: θ_s={theta_s_deg}°, φ_s={phi_s_deg}°")
        
        # 補正係数を計算
        correction_factor = 1.0 / self.apply_distance_correction(1.0, theta_s_deg, phi_s_deg)
        print(f"Correction factor: {correction_factor:.4f}")
        
        # 時刻範囲
        times = sorted(corrected_data.keys())
        print(f"Time range: {times[0]} - {times[-1]}")
        print(f"Data points: {len(times)}")
        
        # 元データと補正データの高度統計比較
        original_max_heights = [data['max_height'] for data in original_data.values() if data['max_height'] is not None]
        corrected_max_heights = [data['max_height'] for data in corrected_data.values() if data['max_height'] is not None]
        
        original_median_heights = [data['median_height'] for data in original_data.values() if data['median_height'] is not None]
        corrected_median_heights = [data['median_height'] for data in corrected_data.values() if data['median_height'] is not None]
        
        original_min_heights = [data['min_height'] for data in original_data.values() if data['min_height'] is not None]  # 最小高度追加
        corrected_min_heights = [data['min_height'] for data in corrected_data.values() if data['min_height'] is not None]  # 最小高度追加
        
        if original_max_heights and corrected_max_heights:
            print(f"\nMax height comparison:")
            print(f"  Original range: {min(original_max_heights):.2f} - {max(original_max_heights):.2f} Rs")
            print(f"  Corrected range: {min(corrected_max_heights):.2f} - {max(corrected_max_heights):.2f} Rs")
            
        if original_median_heights and corrected_median_heights:
            print(f"\nMedian height comparison:")
            print(f"  Original range: {min(original_median_heights):.2f} - {max(original_median_heights):.2f} Rs")
            print(f"  Corrected range: {min(corrected_median_heights):.2f} - {max(corrected_median_heights):.2f} Rs")
        
        if original_min_heights and corrected_min_heights:  # 最小高度比較追加
            print(f"\nMin height comparison:")
            print(f"  Original range: {min(original_min_heights):.2f} - {max(original_min_heights):.2f} Rs")
            print(f"  Corrected range: {min(corrected_min_heights):.2f} - {max(corrected_min_heights):.2f} Rs")
        
        # 補正データの速度統計
        corrected_speed_stats = self.calculate_cme_speed(corrected_data)
        original_speed_stats = self.calculate_cme_speed(original_data)
        
        print(f"\nSpeed comparison:")
        if original_speed_stats['max_speed_kmps'] is not None and corrected_speed_stats['max_speed_kmps'] is not None:
            print(f"  Original max speed: {original_speed_stats['max_speed_kmps']:.1f} ± {original_speed_stats['max_speed_std']:.1f} km/s")
            print(f"  Corrected max speed: {corrected_speed_stats['max_speed_kmps']:.1f} ± {corrected_speed_stats['max_speed_std']:.1f} km/s")
        if original_speed_stats['median_speed_kmps'] is not None and corrected_speed_stats['median_speed_kmps'] is not None:
            print(f"  Original median speed: {original_speed_stats['median_speed_kmps']:.1f} ± {original_speed_stats['median_speed_std']:.1f} km/s")
            print(f"  Corrected median speed: {corrected_speed_stats['median_speed_kmps']:.1f} ± {corrected_speed_stats['median_speed_std']:.1f} km/s")
        if original_speed_stats['min_speed_kmps'] is not None and corrected_speed_stats['min_speed_kmps'] is not None:  # 最小速度比較追加
            print(f"  Original min speed: {original_speed_stats['min_speed_kmps']:.1f} ± {original_speed_stats['min_speed_std']:.1f} km/s")
            print(f"  Corrected min speed: {corrected_speed_stats['min_speed_kmps']:.1f} ± {corrected_speed_stats['min_speed_std']:.1f} km/s")
        
        print("======================================================\n")
    
    def _plot_height_data(self, ax, height_data, color, label_prefix):
        """
        高度データをプロットする共通関数（最小高度追加版）
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            プロット対象の軸
        height_data : dict
            高度データの辞書
        color : str
            プロット色
        label_prefix : str
            ラベルのプレフィックス
        """
        if not height_data or len(height_data) == 0:
            return
            
        # データを時系列順にソート
        sorted_times = sorted(height_data.keys())
        
        times = []
        median_heights = []  # 中央値を使用
        median_std_heights = []  # 中央値ベース標準偏差を使用
        max_heights = []
        min_heights = []  # 最小高度を追加
        
        for time_stamp in sorted_times:
            data = height_data[time_stamp]
            times.append(time_stamp)
            median_heights.append(data['median_height'])  # 中央値を使用
            # median_std_heightがNoneの場合は0.0をデフォルトとして使用
            median_std_val = data['median_std_height'] if data['median_std_height'] is not None else 0.0
            median_std_heights.append(median_std_val)  # 中央値ベース標準偏差を使用
            max_heights.append(data['max_height'])
            min_heights.append(data['min_height'])  # 最小高度を追加
        
        # NumPy配列に変換
        times = np.array(times)
        median_heights = np.array(median_heights, dtype=float)  # 中央値を使用
        median_std_heights = np.array(median_std_heights, dtype=float)  # 中央値ベース標準偏差を使用
        max_heights = np.array(max_heights, dtype=float)
        min_heights = np.array(min_heights, dtype=float)  # 最小高度を追加
        
        # 有効なデータのマスクを作成
        valid_median = ~np.isnan(median_heights)  # 中央値を使用
        valid_max = ~np.isnan(max_heights)
        valid_min = ~np.isnan(min_heights)  # 最小高度を追加
        valid_median_std = (~np.isnan(median_std_heights) & valid_median & (median_std_heights > 0))  # 中央値ベース標準偏差を使用
        
        # 最大高度プロット（マーカーを三角形に変更、色を薄く）
        if np.any(valid_max):
            # 色を薄くする（アルファ値を追加）
            light_color = color if color == 'lightgray' else f'{color}' 
            ax.plot(times[valid_max], max_heights[valid_max], 
                   f'{color}^-', label='Maximum height',  # label_prefixを削除
                   linewidth=1.5, markersize=6, alpha=0.6, zorder=3)  # 薄く表示
        
                # 中央値±中央値ベース標準偏差プロット
        if np.any(valid_median):  # 中央値を使用
            if np.any(valid_median_std):  # 中央値ベース標準偏差を使用
                # エラーバー付きプロット（中央値±中央値ベース標準偏差）
                ax.errorbar(times[valid_median_std], median_heights[valid_median_std],  # 中央値ベース標準偏差を使用
                           yerr=median_std_heights[valid_median_std],  # 中央値ベース標準偏差を使用
                           fmt=f'{color}o-', label='Median height ± median_std',  # label_prefixを削除
                           linewidth=2, markersize=6, capsize=5, capthick=2, zorder=2)
                
                # 中央値ベース標準偏差がないデータポイントも表示
                only_median = valid_median & ~valid_median_std  # 中央値ベース標準偏差を使用
                if np.any(only_median):  # 中央値を使用
                    ax.plot(times[only_median], median_heights[only_median],  # 中央値を使用
                           f'{color}o', markersize=6, zorder=2)
            else:
                # エラーバーなしの中央値プロット
                ax.plot(times[valid_median], median_heights[valid_median],  # 中央値を使用
                       f'{color}o-', label='Median height',  # label_prefixを削除
                       linewidth=2, markersize=6, zorder=2)
        
        # 最小高度プロット（■マーカー、最大高度と同じ色濃度）
        if np.any(valid_min):
            ax.plot(times[valid_min], min_heights[valid_min], 
                   f'{color}s-', label='Minimum height',  # label_prefixを削除
                   linewidth=1.5, markersize=6, alpha=0.6, zorder=3)  # 最大高度と同じ色濃度
        


    def calculate_cme_speed(self, height_data):
        """
        CME速度を計算（最小高度を含む拡張版）
        
        Parameters:
        -----------
        height_data : dict
            時刻をキーとした高度データの辞書
            
        Returns:
        --------
        dict
            速度統計データの辞書（最小速度を含む）
        """
        # 太陽半径をkm単位に変換する定数
        SOLAR_RADIUS_KM = 696000.0  # km
        
        # データを時系列順にソート
        sorted_times = sorted(height_data.keys())
        
        if len(sorted_times) < 2:
            return {'max_speed_kmps': None, 'max_speed_std': None,
                   'median_speed_kmps': None, 'median_speed_std': None,
                   'min_speed_kmps': None, 'min_speed_std': None}  # 最小速度を追加
        
        max_speeds = []
        median_speeds = []  # 中央値を使用
        min_speeds = []  # 最小速度を追加
        
        for i in range(1, len(sorted_times)):
            t1 = sorted_times[i-1]
            t2 = sorted_times[i]
            
            data1 = height_data[t1]
            data2 = height_data[t2]
            
            # 時間差を秒で計算
            dt_seconds = (t2 - t1).total_seconds()
            
            if dt_seconds > 0:
                # 最大高度の速度計算 (km/s)
                if data1['max_height'] is not None and data2['max_height'] is not None:
                    dh_max_km = (data2['max_height'] - data1['max_height']) * SOLAR_RADIUS_KM
                    speed_max = dh_max_km / dt_seconds
                    max_speeds.append(speed_max)
                
                # 中央値高度の速度計算 (km/s)
                if data1['median_height'] is not None and data2['median_height'] is not None:
                    dh_median_km = (data2['median_height'] - data1['median_height']) * SOLAR_RADIUS_KM
                    speed_median = dh_median_km / dt_seconds
                    median_speeds.append(speed_median)
                
                # 最小高度の速度計算 (km/s)
                if data1['min_height'] is not None and data2['min_height'] is not None:
                    dh_min_km = (data2['min_height'] - data1['min_height']) * SOLAR_RADIUS_KM
                    speed_min = dh_min_km / dt_seconds
                    min_speeds.append(speed_min)
        
        # 速度統計を計算
        result = {}
        
        # 最大速度統計
        if max_speeds:
            result['max_speed_kmps'] = np.mean(max_speeds)
            result['max_speed_std'] = np.std(max_speeds)
        else:
            result['max_speed_kmps'] = None
            result['max_speed_std'] = None
        
        # 中央値速度統計
        if median_speeds:
            result['median_speed_kmps'] = np.mean(median_speeds)
            result['median_speed_std'] = np.std(median_speeds)
        else:
            result['median_speed_kmps'] = None
            result['median_speed_std'] = None
        
        # 最小速度統計
        if min_speeds:
            result['min_speed_kmps'] = np.mean(min_speeds)
            result['min_speed_std'] = np.std(min_speeds)
        else:
            result['min_speed_kmps'] = None
            result['min_speed_std'] = None
        
        return result

    def _plot_speed_data(self, ax, height_data, color, label_prefix):
        """
        速度データをプロットする共通関数（最小速度を含む拡張版）
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            プロット対象の軸
        height_data : dict
            高度データの辞書
        color : str
            プロット色
        label_prefix : str
            ラベルのプレフィックス
        """
        if not height_data or len(height_data) < 2:
            return
            
        # 速度統計を事前に計算
        speed_stats = self.calculate_cme_speed(height_data)
            
        # データを時系列順にソート
        sorted_times = sorted(height_data.keys())
        
        speed_times = []
        max_speeds = []
        median_speeds = []  # 中央値を使用
        min_speeds = []  # 最小速度を追加
        
        # 太陽半径をkm単位に変換する定数
        SOLAR_RADIUS_KM = 696000.0  # km
        
        for i in range(1, len(sorted_times)):
            t1 = sorted_times[i-1]
            t2 = sorted_times[i]
            
            data1 = height_data[t1]
            data2 = height_data[t2]
            
            # 時間差を秒で計算
            dt_seconds = (t2 - t1).total_seconds()
            
            if dt_seconds > 0:
                # 中間時刻を記録
                mid_time = t1 + (t2 - t1) / 2
                speed_times.append(mid_time)
                
                # 最大高度の速度計算 (km/s)
                if data1['max_height'] is not None and data2['max_height'] is not None:
                    dh_max_km = (data2['max_height'] - data1['max_height']) * SOLAR_RADIUS_KM
                    speed_max = dh_max_km / dt_seconds
                    max_speeds.append(speed_max)
                else:
                    max_speeds.append(np.nan)
                
                # 中央値高度の速度計算 (km/s)
                if data1['median_height'] is not None and data2['median_height'] is not None:
                    dh_median_km = (data2['median_height'] - data1['median_height']) * SOLAR_RADIUS_KM
                    speed_median = dh_median_km / dt_seconds
                    median_speeds.append(speed_median)
                else:
                    median_speeds.append(np.nan)
                
                # 最小高度の速度計算 (km/s)
                if data1['min_height'] is not None and data2['min_height'] is not None:
                    dh_min_km = (data2['min_height'] - data1['min_height']) * SOLAR_RADIUS_KM
                    speed_min = dh_min_km / dt_seconds
                    min_speeds.append(speed_min)
                else:
                    min_speeds.append(np.nan)
        
        if not speed_times:
            return
            
        # NumPy配列に変換
        speed_times = np.array(speed_times)
        max_speeds = np.array(max_speeds, dtype=float)
        median_speeds = np.array(median_speeds, dtype=float)
        min_speeds = np.array(min_speeds, dtype=float)
        
        # 有効なデータのマスクを作成
        valid_max = ~np.isnan(max_speeds)
        valid_median = ~np.isnan(median_speeds)
        valid_min = ~np.isnan(min_speeds)
        
        # 最大速度プロット（三角形マーカー、薄い色）
        if np.any(valid_max):
            max_speed_label = f"Maximum speed"
            if speed_stats['max_speed_kmps'] is not None:
                max_speed_label += f" ({speed_stats['max_speed_kmps']:.1f}±{speed_stats['max_speed_std']:.1f} km/s)"
            ax.plot(speed_times[valid_max], max_speeds[valid_max], 
                   f'{color}^-', label=max_speed_label,
                   linewidth=1.5, markersize=6, alpha=0.6, zorder=3)
            
        # 中央値速度プロット（円マーカー）
        if np.any(valid_median):
            median_speed_label = f"Median speed"
            if speed_stats['median_speed_kmps'] is not None:
                median_speed_label += f" ({speed_stats['median_speed_kmps']:.1f}±{speed_stats['median_speed_std']:.1f} km/s)"
            ax.plot(speed_times[valid_median], median_speeds[valid_median], 
                   f'{color}o-', label=median_speed_label,
                   linewidth=2, markersize=6, zorder=2)    
        
        # 最小速度プロット（■マーカー、最大速度と同じ色濃度）
        if np.any(valid_min):
            min_speed_label = f"Minimum speed"
            if speed_stats['min_speed_kmps'] is not None:
                min_speed_label += f" ({speed_stats['min_speed_kmps']:.1f}±{speed_stats['min_speed_std']:.1f} km/s)"
            ax.plot(speed_times[valid_min], min_speeds[valid_min], 
                   f'{color}s-', label=min_speed_label,
                   linewidth=1.5, markersize=6, alpha=0.6, zorder=3)
        


    def _print_analysis_summary(self, height_data):
        """
        解析結果の統計情報を表示
        
        Parameters:
        -----------
        height_data : dict
            高度データの辞書
        """
        if not height_data:
            return
            
        print("\n=== Analysis Summary ===")
        
        # 時刻範囲
        times = sorted(height_data.keys())
        print(f"Time range: {times[0]} - {times[-1]}")
        print(f"Data points: {len(times)}")
        
        # 高度統計
        max_heights = [data['max_height'] for data in height_data.values() if data['max_height'] is not None]
        mean_heights = [data['mean_height'] for data in height_data.values() if data['mean_height'] is not None]
        min_heights = [data['min_height'] for data in height_data.values() if data['min_height'] is not None]  # 最小高度追加
        
        if max_heights:
            print(f"Max height range: {min(max_heights):.2f} - {max(max_heights):.2f} Rs")
        if mean_heights:
            print(f"Mean height range: {min(mean_heights):.2f} - {max(mean_heights):.2f} Rs")
        if min_heights:  # 最小高度統計追加
            print(f"Min height range: {min(min_heights):.2f} - {max(min_heights):.2f} Rs")
        
        # 速度統計
        speed_stats = self.calculate_cme_speed(height_data)
        if speed_stats['max_speed_kmps'] is not None:
            print(f"Average max speed: {speed_stats['max_speed_kmps']:.1f} ± {speed_stats['max_speed_std']:.1f} km/s")
        if speed_stats['median_speed_kmps'] is not None:
            print(f"Average median speed: {speed_stats['median_speed_kmps']:.1f} ± {speed_stats['median_speed_std']:.1f} km/s")
        if speed_stats['min_speed_kmps'] is not None:  # 最小速度統計追加
            print(f"Average min speed: {speed_stats['min_speed_kmps']:.1f} ± {speed_stats['min_speed_std']:.1f} km/s")
        
        print("========================\n")


def main():
    """
    メイン関数：コマンドライン引数の処理と解析実行
    """
    parser = argparse.ArgumentParser(
        description='CME with_line統計データの時系列プロット',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python cme_with_line_plot.py --start_time "2022-06-13T03:00:00" --end_time "2022-06-13T03:40:00"
  python cme_with_line_plot.py --start_time "2022-06-13T03:00:00" --end_time "2022-06-13T03:40:00" --save_path "with_line_plot"
  
注意：常に補正前と補正済みの両方のプロットが生成されます
補正パラメータ：θ_s=21.0°, φ_s=-44.0°（固定）
        """
    )
    
    parser.add_argument('--start_time', type=str, required=True,
                       help='開始時刻 (ISO format: YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('--end_time', type=str, required=True,
                       help='終了時刻 (ISO format: YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('--csv_folder', type=str, 
                       default='../CME_measurement/csv_folder',
                       help='CSVフォルダのパス (default: ../CME_measurement/csv_folder)')
    parser.add_argument('--save_path', type=str,
                       help='保存ファイルパスのベース名 (_original.png, _corrected.pngが自動追加)')
    parser.add_argument('--figsize', type=str, default='15,8',
                       help='図のサイズ (width,height) (default: 15,8)')
    
    args = parser.parse_args()
    
    # figsize処理
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
        if len(figsize) != 2:
            raise ValueError
    except ValueError:
        print("Error: figsize must be in format 'width,height' (e.g., '15,8')")
        return 1
    
    try:
        # プロッターの初期化
        plotter = CMEWithLinePlotter(args.csv_folder)
        
        # 補正前・補正済み比較解析実行（常に両方プロット）
        original_data, corrected_data, fig_orig, axes_orig, fig_corr, axes_corr = plotter.plot_with_line_comparison_analysis(
            args.start_time, 
            args.end_time, 
            args.save_path,
            figsize
        )
        
        if not original_data and not corrected_data:
            print("Warning: No data found for the specified time range.")
            return 1
            
    except FileNotFoundError as e:
        print(f"Error: CSVフォルダが見つかりません: {e}")
        return 1
    except Exception as e:
        print(f"Error: 解析中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # コマンドライン引数がない場合のデフォルト実行例
    if len(os.sys.argv) == 1:
        print("CME with_line統計データプロットツール（補正前・補正済み比較版）")
        print("====================================================")
        print()
        print("使用例:")
        print("python cme_with_line_plot.py --start_time '2022-06-13T03:00:00' --end_time '2022-06-13T03:40:00'")
        print("python cme_with_line_plot.py --start_time '2022-06-13T03:00:00' --end_time '2022-06-13T03:40:00' --save_path 'with_line_plot'")
        print()
        print("オプション:")
        print("  --csv_folder : CSVフォルダのパス")
        print("  --save_path  : 保存ファイルパスのベース名 (_original.png, _corrected.pngが自動追加)")
        print("  --figsize    : 図のサイズ (例: '15,8')")
        print()
        print("注意：常に補正前と補正済みの両方のプロットが生成されます")
        print("補正パラメータ：θ_s=21.0°, φ_s=-44.0°（固定）")
        print()
        print("直接実行例を開始します...")
        print()
        
        # デフォルトパラメータでの実行例
        csv_folder = "../CME_measurement/csv_folder"
        plotter = CMEWithLinePlotter(csv_folder)
        
        try:
            # 補正前・補正済み比較解析実行
            print("=== 補正前・補正済み比較解析実行 ===")
            original_data, corrected_data, fig_orig, axes_orig, fig_corr, axes_corr = plotter.plot_with_line_comparison_analysis(
                "2022-06-13T03:19:00",
                "2022-06-13T03:34:00",
                "default_save_path"  # save_path_baseを指定（実際のファイル名は内部で設定される）
            )
            
            if original_data and corrected_data:
                print("比較解析が正常に完了しました！")
                print("・補正前プロット: ../speed_analysis_along_with_line_skyplane.png")
                print("・補正済みプロット: ../speed_analysis_along_with_line_3D.png")
            else:
                print("データが見つかりませんでした。")
                
        except Exception as e:
            print(f"デフォルト実行でエラー: {e}")
    else:
        # コマンドライン引数がある場合は通常のmain()を実行
        exit(main())