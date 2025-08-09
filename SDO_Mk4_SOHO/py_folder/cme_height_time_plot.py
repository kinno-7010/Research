#!/usr/bin/env python3
"""
CME高度の時系列プロットツール

このスクリプトは、CME_measurement/csv_folder/sta/ フォルダ内の統計データを使用して、
時間範囲を指定してCME先端高度の時系列変化をプロットします。

機能:
- 時間範囲の指定
- 最大高度のプロット
- 平均高度±標準偏差のエラーバー付きプロット
- 太陽半径単位での高度表示

使用例:
python cme_height_time_plot.py --start_time "2022-06-13T03:00:00" --end_time "2022-06-13T03:40:00"
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

class CMEHeightPlotter:
    def __init__(self, csv_folder_path):
        """
        CME高度プロッターの初期化
        
        Parameters:
        -----------
        csv_folder_path : str
            CME統計データのCSVフォルダのパス
        """
        self.csv_folder_path = Path(csv_folder_path)
        self.sta_folder = self.csv_folder_path / "sta"
        self.obs_folder = self.csv_folder_path / "obs"
        
    def load_statistics_data(self, start_time, end_time, file_pattern="nose_cme_statistics_*.csv"):
        """
        指定された時間範囲の統計データを読み込む
        
        Parameters:
        -----------
        start_time : str or datetime
            開始時刻 (ISO format: "YYYY-MM-DDTHH:MM:SS")
        end_time : str or datetime
            終了時刻 (ISO format: "YYYY-MM-DDTHH:MM:SS")
        file_pattern : str
            検索するファイルのパターン
            
        Returns:
        --------
        pd.DataFrame
            統計データのDataFrame
        """
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
            
        # 統計データファイルを検索
        pattern = str(self.sta_folder / file_pattern)
        csv_files = glob.glob(pattern)
        
        all_data = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # データが空でない場合のみ処理
                if len(df) > 0 and 'time' in df.columns:
                    # 時刻列をdatetime型に変換（YYMMDD-HHMMSS形式に対応）
                    df['datetime'] = pd.to_datetime(df['time'])
                    
                    # 指定された時間範囲内のデータをフィルタリング
                    mask = (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
                    filtered_df = df[mask]
                    
                    if len(filtered_df) > 0:
                        all_data.append(filtered_df)
                        
            except Exception as e:
                print(f"Warning: ファイル {csv_file} の読み込みでエラー: {e}")
                continue
        
        if not all_data:
            print(f"Warning: 指定された時間範囲 {start_time} - {end_time} にデータが見つかりません (pattern: {file_pattern})")
            return pd.DataFrame()
            
        # 全データを結合
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 時刻でソート
        combined_df = combined_df.sort_values('datetime')
        
        return combined_df
    
    def load_raw_statistics_data(self, start_time, end_time):
        """
        raw_cme_statistics統計データを読み込む
        
        Parameters:
        -----------
        start_time : str or datetime
            開始時刻
        end_time : str or datetime
            終了時刻
            
        Returns:
        --------
        pd.DataFrame
            raw統計データのDataFrame
        """
        return self.load_statistics_data(start_time, end_time, "raw_cme_statistics_*.csv")
    
    def load_diff_statistics_data(self, start_time, end_time):
        """
        1min_diff統計データを読み込む
        
        Parameters:
        -----------
        start_time : str or datetime
            開始時刻
        end_time : str or datetime
            終了時刻
            
        Returns:
        --------
        pd.DataFrame
            1min_diff統計データのDataFrame
        """
        return self.load_statistics_data(start_time, end_time, "cme_statistics_*_1min_diff.csv")
    
    def load_raw_observations_data(self, start_time, end_time):
        """
        指定された時間範囲のraw観測データを読み込む
        
        Parameters:
        -----------
        start_time : str or datetime
            開始時刻 (ISO format: "YYYY-MM-DDTHH:MM:SS")
        end_time : str or datetime
            終了時刻 (ISO format: "YYYY-MM-DDTHH:MM:SS")
            
        Returns:
        --------
        pd.DataFrame
            raw観測データのDataFrame
        """
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
            
        all_raw_data = []
        
        # obs/フォルダ内のcme_observations_*.csvファイルを検索
        obs_pattern = str(self.obs_folder / "cme_observations_*.csv")
        obs_files = glob.glob(obs_pattern)
        
        # メインフォルダ内のcme_measurements_*.csvファイルも検索
        main_pattern = str(self.csv_folder_path / "cme_measurements_*.csv")
        main_files = glob.glob(main_pattern)
        
        all_files = obs_files + main_files
        
        for csv_file in all_files:
            try:
                df = pd.read_csv(csv_file)
                
                # データが空でない場合のみ処理
                if len(df) > 0 and 'time' in df.columns:
                    # 時刻列をdatetime型に変換
                    df['datetime'] = pd.to_datetime(df['time'])
                    
                    # point_idが数値のデータのみ抽出（統計行を除外）
                    if 'point_id' in df.columns:
                        # point_idが数値のデータのみ取得
                        numeric_mask = pd.to_numeric(df['point_id'], errors='coerce').notna()
                        df = df[numeric_mask]
                    
                    if len(df) > 0:
                        # 指定された時間範囲内のデータをフィルタリング
                        mask = (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
                        filtered_df = df[mask]
                        
                        if len(filtered_df) > 0:
                            all_raw_data.append(filtered_df)
                        
            except Exception as e:
                print(f"Warning: ファイル {csv_file} の読み込みでエラー: {e}")
                continue
        
        if not all_raw_data:
            print(f"Warning: 指定された時間範囲 {start_time} - {end_time} にraw観測データが見つかりません")
            return pd.DataFrame()
            
        # 全データを結合
        combined_df = pd.concat(all_raw_data, ignore_index=True)
        
        # 時刻でソート
        combined_df = combined_df.sort_values('datetime')
        
        return combined_df
    
    def extract_height_data(self, df):
        """
        統計データから高度情報を抽出
        
        Parameters:
        -----------
        df : pd.DataFrame
            統計データのDataFrame
            
        Returns:
        --------
        dict
            時刻をキーとした高度データの辞書
        """
        height_data = {}
        
        # ユニークな時刻を取得
        unique_times = df['datetime'].unique()
        
        for time_stamp in unique_times:
            time_data = df[df['datetime'] == time_stamp]
            
            height_info = {
                'datetime': time_stamp,
                'mean_height': None,
                'std_height': None,
                'max_height': None,
                'min_height': None,
                'n_points': 0
            }
            
            # 各統計値を抽出
            for _, row in time_data.iterrows():
                metric = row['metric']
                value = row['value']
                
                if pd.notna(value) and value != '':
                    try:
                        value = float(value)
                        if metric == 'mean_height_rsun':
                            height_info['mean_height'] = value
                        elif metric == 'std_height_rsun':
                            height_info['std_height'] = value
                        elif metric == 'max_height_rsun':
                            height_info['max_height'] = value
                        elif metric == 'min_height_rsun':
                            height_info['min_height'] = value
                        elif metric == 'n_points':
                            height_info['n_points'] = int(value)
                    except (ValueError, TypeError):
                        continue
            
            # データポイントが存在する場合のみ追加
            if height_info['n_points'] > 0:
                height_data[time_stamp] = height_info
                
        return height_data
    
    def extract_raw_height_data(self, raw_df):
        """
        raw観測データから時刻ごとの高度統計を抽出
        
        Parameters:
        -----------
        raw_df : pd.DataFrame
            raw観測データのDataFrame
            
        Returns:
        --------
        dict
            時刻をキーとした高度統計データの辞書
        """
        if raw_df.empty or 'height_rsun' not in raw_df.columns:
            return {}
            
        raw_height_data = {}
        
        # ユニークな時刻を取得
        unique_times = raw_df['datetime'].unique()
        
        for time_stamp in unique_times:
            time_data = raw_df[raw_df['datetime'] == time_stamp]
            
            # height_rsun列から有効な数値を抽出
            heights = time_data['height_rsun'].dropna()
            heights = heights[heights > 0]  # 正の値のみ
            
            if len(heights) > 0:
                height_info = {
                    'datetime': time_stamp,
                    'mean_height': float(heights.mean()),
                    'std_height': float(heights.std()) if len(heights) > 1 else 0.0,
                    'max_height': float(heights.max()),
                    'min_height': float(heights.min()),
                    'n_points': len(heights)
                }
                
                raw_height_data[time_stamp] = height_info
                
        return raw_height_data
    
    def calculate_cme_speed(self, height_data):
        """
        CME nose速度を計算
        
        Parameters:
        -----------
        height_data : dict
            時刻をキーとした高度データの辞書
            
        Returns:
        --------
        dict
            速度統計データの辞書
        """
        # 太陽半径をkm単位に変換する定数
        SOLAR_RADIUS_KM = 696000.0  # km
        
        # データを時系列順にソート
        sorted_times = sorted(height_data.keys())
        
        if len(sorted_times) < 2:
            return {'max_speed_kmps': None, 'max_speed_std': None,
                   'mean_speed_kmps': None, 'mean_speed_std': None}
        
        max_speeds = []
        mean_speeds = []
        
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
                
                # 平均高度の速度計算 (km/s)
                if data1['mean_height'] is not None and data2['mean_height'] is not None:
                    dh_mean_km = (data2['mean_height'] - data1['mean_height']) * SOLAR_RADIUS_KM
                    speed_mean = dh_mean_km / dt_seconds
                    mean_speeds.append(speed_mean)
        
        # 統計計算
        max_speeds = np.array(max_speeds)
        mean_speeds = np.array(mean_speeds)
        
        speed_stats = {
            'max_speed_kmps': np.mean(max_speeds) if len(max_speeds) > 0 else None,
            'max_speed_std': np.std(max_speeds) if len(max_speeds) > 0 else None,
            'mean_speed_kmps': np.mean(mean_speeds) if len(mean_speeds) > 0 else None,
            'mean_speed_std': np.std(mean_speeds) if len(mean_speeds) > 0 else None
        }
        
        return speed_stats
    
    def plot_height_evolution(self, nose_height_data, raw_height_data=None, diff_height_data=None, save_path=None, figsize=(15, 12)):
        """
        CME高度の時系列変化をプロット（4つのサブプロット）
        
        Parameters:
        -----------
        nose_height_data : dict
            nose統計高度データの辞書
        raw_height_data : dict, optional
            raw統計高度データの辞書
        diff_height_data : dict, optional
            1min_diff統計高度データの辞書
        save_path : str, optional
            保存ファイルパス
        figsize : tuple
            図のサイズ
        """
        # サブプロット作成（4つ）
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # 色の定義
        raw_color = 'g'
        nose_color = 'b' 
        diff_color = 'r'
        
        # =========================
        # axes[0]: Raw Max Height + Raw Mean Height
        # =========================
        self._plot_height_data(axes[0], raw_height_data, raw_color, 'Raw')
        axes[0].set_title('Raw Max Height + Raw Mean Height', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('CME height [Solar radii]', fontsize=12)
        axes[0].legend(fontsize=10, loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # =========================
        # axes[1]: Base Diff Max height + Base Diff Mean Height  
        # =========================
        self._plot_height_data(axes[1], nose_height_data, nose_color, 'Base Diff')
        axes[1].set_title('Base Diff Max height + Base Diff Mean Height', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('CME height [Solar radii]', fontsize=12)
        axes[1].legend(fontsize=10, loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        # =========================
        # axes[2]: 2 min Diff Max Height + 2 min Diff Mean Height
        # =========================
        self._plot_height_data(axes[2], diff_height_data, diff_color, '2 min Diff')
        axes[2].set_title('2 min Diff Max Height + 2 min Diff Mean Height', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('CME height [Solar radii]', fontsize=12)
        axes[2].legend(fontsize=10, loc='upper left')
        axes[2].grid(True, alpha=0.3)
        
        # =========================
        # axes[3]: 速度プロット（ax[0]~ax[2]から導出）
        # =========================
        
        # 各データセットの速度を計算してプロット
        if raw_height_data and len(raw_height_data) > 0:
            self._plot_speed_data(axes[3], raw_height_data, raw_color, 'Raw')
        
        if nose_height_data and len(nose_height_data) > 0:
            self._plot_speed_data(axes[3], nose_height_data, nose_color, 'Base Diff')
            
        if diff_height_data and len(diff_height_data) > 0:
            self._plot_speed_data(axes[3], diff_height_data, diff_color, '2 min Diff')
        
        # 速度プロットの設定
        axes[3].set_xlabel('Observation time (UTC)', fontsize=12)
        axes[3].set_ylabel('CME speed [km/s]', fontsize=12)
        axes[3].set_title('Speed from ax[0]~ax[2]', fontsize=14, fontweight='bold')
        axes[3].legend(fontsize=10, loc='upper right')
        axes[3].grid(True, alpha=0.3)
        
        # 時刻軸の書式設定（最下段のグラフのみ）
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        axes[3].tick_params(axis='x', rotation=0)
        
        # レイアウト調整
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        
        # 保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        
        plt.show()
        
        return fig, axes
    
    def _plot_height_data(self, ax, height_data, color, label_prefix):
        """
        高度データをプロットする共通関数
        
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
        mean_heights = []
        std_heights = []
        max_heights = []
        
        for time_stamp in sorted_times:
            data = height_data[time_stamp]
            times.append(time_stamp)
            mean_heights.append(data['mean_height'])
            std_heights.append(data['std_height'])
            max_heights.append(data['max_height'])
        
        # NumPy配列に変換
        times = np.array(times)
        mean_heights = np.array(mean_heights, dtype=float)
        std_heights = np.array(std_heights, dtype=float)
        max_heights = np.array(max_heights, dtype=float)
        
        # 有効なデータのマスクを作成
        valid_mean = ~np.isnan(mean_heights)
        valid_max = ~np.isnan(max_heights)
        valid_std = (~np.isnan(std_heights) & valid_mean & (std_heights > 0))
        
        # 最大高度プロット
        if np.any(valid_max):
            ax.plot(times[valid_max], max_heights[valid_max], 
                   f'{color}o-', label=f'{label_prefix} Maximum height', 
                   linewidth=2, markersize=6, zorder=3)
        
        # 平均高度±標準偏差プロット
        if np.any(valid_mean):
            if np.any(valid_std):
                # エラーバー付きプロット
                ax.errorbar(times[valid_std], mean_heights[valid_std], 
                           yerr=std_heights[valid_std],
                           fmt=f'{color}o-', label=f'{label_prefix} Mean height ± std', 
                           linewidth=2, markersize=6, capsize=5, capthick=2, zorder=2)
                
                # 標準偏差がないデータポイントも表示
                only_mean = valid_mean & ~valid_std
                if np.any(only_mean):
                    ax.plot(times[only_mean], mean_heights[only_mean], 
                           f'{color}o', markersize=6, zorder=2)
            else:
                # エラーバーなしの平均高度プロット
                ax.plot(times[valid_mean], mean_heights[valid_mean], 
                       f'{color}o-', label=f'{label_prefix} Mean height', 
                       linewidth=2, markersize=6, zorder=2)
    
    def _plot_speed_data(self, ax, height_data, color, label_prefix):
        """
        速度データをプロットする共通関数
        
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
            
        # データを時系列順にソート
        sorted_times = sorted(height_data.keys())
        
        speed_times = []
        max_speeds = []
        mean_speeds = []
        
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
                # 最大高度の速度計算 (km/s)
                if data1['max_height'] is not None and data2['max_height'] is not None:
                    dh_max_km = (data2['max_height'] - data1['max_height']) * SOLAR_RADIUS_KM
                    speed_max = dh_max_km / dt_seconds
                    max_speeds.append(speed_max)
                
                # 平均高度の速度計算 (km/s)
                if data1['mean_height'] is not None and data2['mean_height'] is not None:
                    dh_mean_km = (data2['mean_height'] - data1['mean_height']) * SOLAR_RADIUS_KM
                    speed_mean = dh_mean_km / dt_seconds
                    mean_speeds.append(speed_mean)
                    
                speed_times.append(t2)
        
        # 速度プロット
        if len(speed_times) > 0:
            if len(max_speeds) > 0:
                max_speeds_array = np.array(max_speeds)
                avg_max_speed = np.mean(max_speeds_array)
                std_max_speed = np.std(max_speeds_array)
                ax.plot(speed_times[:len(max_speeds)], max_speeds, 
                       f'{color}o-', label=f'{label_prefix} Max Speed ({avg_max_speed:.1f} ± {std_max_speed:.1f} km/s)', 
                       linewidth=2, markersize=6)
            if len(mean_speeds) > 0:
                mean_speeds_array = np.array(mean_speeds)
                avg_mean_speed = np.mean(mean_speeds_array)
                std_mean_speed = np.std(mean_speeds_array)
                ax.plot(speed_times[:len(mean_speeds)], mean_speeds, 
                       f'{color}s-', label=f'{label_prefix} Mean Speed ({avg_mean_speed:.1f} ± {std_mean_speed:.1f} km/s)', 
                       linewidth=2, markersize=6)
    
    def run_analysis(self, start_time, end_time, save_path=None):
        """
        完全な解析を実行（4つのサブプロット版）
        
        Parameters:
        -----------
        start_time : str
            開始時刻 (ISO format: "YYYY-MM-DDTHH:MM:SS")
        end_time : str
            終了時刻 (ISO format: "YYYY-MM-DDTHH:MM:SS")
        save_path : str, optional
            保存ファイルパス
        """
        print(f"Starting CME height analysis: {start_time} - {end_time}")
        
        # Raw統計データ読み込み
        print("Loading raw statistics data...")
        raw_df = self.load_raw_statistics_data(start_time, end_time)
        raw_height_data = {}
        if not raw_df.empty:
            print(f"Raw statistics data loading complete: {len(raw_df)} records")
            raw_height_data = self.extract_height_data(raw_df)
            print(f"Raw statistics extraction complete: {len(raw_height_data)} time points")
        else:
            print("No raw statistics data found")
        
        # Nose統計データ読み込み
        print("Loading nose statistics data...")
        nose_df = self.load_statistics_data(start_time, end_time)
        nose_height_data = {}
        if not nose_df.empty:
            print(f"Nose statistics data loading complete: {len(nose_df)} records")
            nose_height_data = self.extract_height_data(nose_df)
            print(f"Nose statistics extraction complete: {len(nose_height_data)} time points")
        else:
            print("No nose statistics data found")
        
        # 1min_diff統計データ読み込み
        print("Loading diff statistics data...")
        diff_df = self.load_diff_statistics_data(start_time, end_time)
        diff_height_data = {}
        if not diff_df.empty:
            print(f"Diff statistics data loading complete: {len(diff_df)} records")
            diff_height_data = self.extract_height_data(diff_df)
            print(f"Diff statistics extraction complete: {len(diff_height_data)} time points")
        else:
            print("No diff statistics data found")
        
        # プロット作成
        print("Creating plot...")
        fig, axes = self.plot_height_evolution(nose_height_data, raw_height_data, diff_height_data, save_path)
        
        print("Analysis complete!")
        return nose_height_data, raw_height_data, diff_height_data, fig, axes


def main():
    """
    メイン関数：コマンドライン引数の処理と解析実行
    """
    parser = argparse.ArgumentParser(description='CME高度の時系列プロット')
    parser.add_argument('--start_time', type=str, required=True,
                       help='開始時刻 (ISO format: YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('--end_time', type=str, required=True,
                       help='終了時刻 (ISO format: YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('--csv_folder', type=str, 
                       default='../CME_measurement/csv_folder',
                       help='CSVフォルダのパス')
    parser.add_argument('--save_path', type=str,
                       help='保存ファイルパス')
    
    args = parser.parse_args()
    
    try:
        # プロッターの初期化
        plotter = CMEHeightPlotter(args.csv_folder)
        
        # 解析実行
        nose_data, raw_data, diff_data, fig, axes = plotter.run_analysis(
            args.start_time, 
            args.end_time, 
            args.save_path
        )
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # 使用例（直接実行時）
    if len(os.sys.argv) == 1:
        # デフォルト実行例
        print("使用例:")
        print("python cme_height_time_plot.py --start_time '2022-06-13T03:00:00' --end_time '2022-06-13T03:40:00'")
        print("\n直接実行例:")
        
        # デフォルトパラメータでの実行
        csv_folder = "../CME_measurement/csv_folder"
        plotter = CMEHeightPlotter(csv_folder)
        
        try:
            nose_data, raw_data, diff_data, fig, axes = plotter.run_analysis(
                "2022-06-13T03:00:00",
                "2022-06-13T03:40:00",
                "../CME_measurement/analysis_png/cme_height_time_plot.png"
            )
        except Exception as e:
            print(f"デフォルト実行でエラー: {e}")
    else:
        main()