#!/usr/bin/env python3
"""
CME with_line統計データ専用プロットツール（補正前・補正後比較版）

このスクリプトは、cme_statistics_YYMMDD-HHMMSS_min_diff_with_line.csvファイルを使用して、
CME先端高度の時系列変化をプロットします。
補正前と補正後の両方のプロットを別図で生成します。

機能:
- with_line統計データ専用のプロット
- ax[0]: Max height + Mean height±std
- ax[1]: Max speed + Mean speed
- 時間範囲の指定
- 太陽半径単位での高度表示
- 距離補正機能（θ_s=21°, φ_s=-44°固定）

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
            save_path_original = f"{base_name}_original.png"
            save_path_corrected = f"{base_name}_corrected.png"
        
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
    
    def _print_comparison_analysis_summary(self, original_data, corrected_data, theta_s_deg, phi_s_deg):
        """
        距離補正比較解析結果の統計情報を表示
        
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
        
        if original_max_heights and corrected_max_heights:
            print(f"\nMax height comparison:")
            print(f"  Original range: {min(original_max_heights):.2f} - {max(original_max_heights):.2f} Rs")
            print(f"  Corrected range: {min(corrected_max_heights):.2f} - {max(corrected_max_heights):.2f} Rs")
            
        if original_median_heights and corrected_median_heights:
            print(f"\nMedian height comparison:")
            print(f"  Original range: {min(original_median_heights):.2f} - {max(original_median_heights):.2f} Rs")
            print(f"  Corrected range: {min(corrected_median_heights):.2f} - {max(corrected_median_heights):.2f} Rs")
        
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
        
        print("======================================================\n")


def main():
    """
    メイン関数：コマンドライン引数の処理と解析実行
    """
    parser = argparse.ArgumentParser(
        description='CME with_line統計データの時系列プロット（補正前・補正後比較版）',
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
                "2022-06-13T03:33:00",
                "../CME_measurement/cme_with_line_comparison_analysis"
            )
            
            if original_data and corrected_data:
                print("比較解析が正常に完了しました！")
                print("・補正前プロット: ../CME_measurement/cme_with_line_comparison_analysis_original.png")
                print("・補正済みプロット: ../CME_measurement/cme_with_line_comparison_analysis_corrected.png")
            else:
                print("データが見つかりませんでした。")
                
        except Exception as e:
            print(f"デフォルト実行でエラー: {e}")
    else:
        # コマンドライン引数がある場合は通常のmain()を実行
        exit(main())