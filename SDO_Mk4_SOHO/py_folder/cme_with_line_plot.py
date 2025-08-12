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
        print("Creating with_line plot...")
        fig, axes = self.plot_diff_with_line_evolution(
            with_line_height_data, 
            save_path=save_path, 
            figsize=figsize
        )
        
        # プロット結果の統計情報を表示
        if with_line_height_data:
            self._print_analysis_summary(with_line_height_data)
        
        print("With_line analysis complete!")
        return with_line_height_data, fig, axes
    
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
        
        if max_heights:
            print(f"Max height range: {min(max_heights):.2f} - {max(max_heights):.2f} Rs")
        if mean_heights:
            print(f"Mean height range: {min(mean_heights):.2f} - {max(mean_heights):.2f} Rs")
        
        # 速度統計
        speed_stats = self.calculate_cme_speed(height_data)
        if speed_stats['max_speed_kmps'] is not None:
            print(f"Average max speed: {speed_stats['max_speed_kmps']:.1f} ± {speed_stats['max_speed_std']:.1f} km/s")
        if speed_stats['mean_speed_kmps'] is not None:
            print(f"Average mean speed: {speed_stats['mean_speed_kmps']:.1f} ± {speed_stats['mean_speed_std']:.1f} km/s")
        
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
  python cme_with_line_plot.py --start_time "2022-06-13T03:00:00" --end_time "2022-06-13T03:40:00" --save_path "with_line_plot.png"
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
                       help='保存ファイルパス (指定しない場合は画面表示のみ)')
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
        
        # 解析実行
        with_line_data, fig, axes = plotter.plot_with_line_analysis(
            args.start_time, 
            args.end_time, 
            args.save_path,
            figsize
        )
        
        if not with_line_data:
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
        print("CME with_line統計データプロットツール")
        print("=====================================")
        print()
        print("使用例:")
        print("python cme_with_line_plot.py --start_time '2022-06-13T03:00:00' --end_time '2022-06-13T03:40:00'")
        print("python cme_with_line_plot.py --start_time '2022-06-13T03:00:00' --end_time '2022-06-13T03:40:00' --save_path 'with_line_plot.png'")
        print()
        print("オプション:")
        print("  --csv_folder     : CSVフォルダのパス")
        print("  --save_path      : 保存ファイルパス")
        print("  --figsize        : 図のサイズ (例: '15,8')")
        print()
        print("直接実行例を開始します...")
        print()
        
        # デフォルトパラメータでの実行例
        csv_folder = "../CME_measurement/csv_folder"
        plotter = CMEWithLinePlotter(csv_folder)
        
        try:
            with_line_data, fig, axes = plotter.plot_with_line_analysis(
                "2022-06-13T03:00:00",
                "2022-06-13T04:01:00",
                "../with_line_analysis.png"
            )
            
            if with_line_data:
                print("デフォルト実行が正常に完了しました！")
            else:
                print("データが見つかりませんでした。")
                
        except Exception as e:
            print(f"デフォルト実行でエラー: {e}")
    else:
        # コマンドライン引数がある場合は通常のmain()を実行
        exit(main())