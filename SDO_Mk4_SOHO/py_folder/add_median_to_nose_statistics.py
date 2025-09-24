#!/usr/bin/env python3
"""
nose統計データに中央値を追加するツール

このスクリプトは、nose_cme_observations_*.csvからheight_rsunデータを読み込み、
時刻ごとの中央値を計算して、対応するnose_cme_statistics_*.csvに追記します。

使用例:
python add_median_to_nose_statistics.py --start_time "2022-06-13T03:00:00" --end_time "2022-06-13T04:00:00"
"""

import pandas as pd
import numpy as np
import glob
import os
import argparse
import shutil
from datetime import datetime
from pathlib import Path


class MedianStatisticsAdderNose:
    """
    nose統計データに中央値を追加するクラス
    """
    
    def __init__(self, csv_folder_path):
        """
        初期化
        
        Parameters:
        -----------
        csv_folder_path : str
            CME統計データのCSVフォルダのパス
        """
        self.csv_folder_path = Path(csv_folder_path)
        self.obs_folder = self.csv_folder_path / "obs"
        self.sta_folder = self.csv_folder_path / "sta"
        
        # フォルダ存在確認
        if not self.csv_folder_path.exists():
            raise FileNotFoundError(f"CSV folder not found: {csv_folder_path}")
        if not self.obs_folder.exists():
            raise FileNotFoundError(f"Observations folder not found: {self.obs_folder}")
    
    def find_observations_files(self, start_time=None, end_time=None):
        """
        指定時間範囲のnose observations CSVファイルを検索
        """
        pattern = str(self.obs_folder / "nose_cme_observations_*.csv")
        obs_files = glob.glob(pattern)
        
        if not start_time or not end_time:
            return sorted(obs_files)
        
        # 時間範囲でフィルタリング
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        
        filtered_files = []
        for file_path in obs_files:
            try:
                # ファイル名から時刻を抽出 (例: nose_cme_observations_220613-032055.csv)
                filename = os.path.basename(file_path)
                time_part = filename.split('_')[3].split('.')[0]  # 220613-032055
                
                # YYMMDD-HHMMSS形式をdatetimeに変換
                year = 2000 + int(time_part[:2])
                month = int(time_part[2:4])
                day = int(time_part[4:6])
                hour = int(time_part[7:9])
                minute = int(time_part[9:11])
                second = int(time_part[11:13])
                
                file_dt = datetime(year, month, day, hour, minute, second)
                
                if start_dt <= file_dt <= end_dt:
                    filtered_files.append(file_path)
                    
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse time from filename {filename}: {e}")
                continue
        
        return sorted(filtered_files)
    
    def find_statistics_file(self, obs_file_path):
        """
        observations CSVファイルに対応するnose statistics CSVファイルを検索
        """
        # observations ファイル名から時刻部分を抽出
        obs_filename = os.path.basename(obs_file_path)
        time_part = obs_filename.split('_')[3].split('.')[0]  # 220613-032055
        
        # 対応するstatistics ファイルを検索
        stats_pattern = str(self.csv_folder_path / f"nose_cme_statistics_{time_part}.csv")
        stats_files = glob.glob(stats_pattern)
        
        # sta/ フォルダ内も検索
        sta_pattern = str(self.sta_folder / f"nose_cme_statistics_{time_part}.csv")
        sta_files = glob.glob(sta_pattern)
        
        all_stats_files = stats_files + sta_files
        
        if all_stats_files:
            return all_stats_files[0]  # 最初に見つかったファイルを返す
        else:
            return None
    
    def calculate_median_from_observations(self, obs_file_path):
        """
        observations CSVファイルからheight_rsunの中央値を計算
        """
        try:
            df = pd.read_csv(obs_file_path)
            
            if df.empty or 'height_rsun' not in df.columns:
                print(f"Warning: No height_rsun data in {obs_file_path}")
                return None
            
            # height_rsunデータの取得
            heights = df['height_rsun'].dropna()
            heights = heights[heights > 0]  # 正の値のみ
            
            if len(heights) == 0:
                print(f"Warning: No valid height_rsun data in {obs_file_path}")
                return None
            
            # 時刻の取得（最初の行の時刻を使用）
            time_stamp = df['time'].iloc[0] if 'time' in df.columns else None
            
            median_value = float(np.median(heights))
            # 中央値ベース標準偏差を計算
            median_std_value = float(np.sqrt(((heights - median_value) ** 2).mean())) if len(heights) > 1 else 0.0
            
            return {
                'median_height_rsun': median_value,
                'median_std_height_rsun': median_std_value,  # 中央値ベース標準偏差を追加
                'time': time_stamp,
                'n_points': len(heights),
                'file': obs_file_path
            }
            
        except Exception as e:
            print(f"Error calculating median from {obs_file_path}: {e}")
            return None
    
    def add_median_to_statistics(self, stats_file_path, median_info, backup=True):
        """
        statistics CSVファイルに中央値行を追記
        """
        try:
            # バックアップ作成
            if backup and os.path.exists(stats_file_path):
                backup_path = f"{stats_file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(stats_file_path, backup_path)
                print(f"Backup created: {backup_path}")
            
            # 既存データの読み込み
            if os.path.exists(stats_file_path):
                df = pd.read_csv(stats_file_path)
                
                # 既に中央値と中央値ベース標準偏差の両方が存在するかチェック
                median_exists = df['metric'].str.contains('median_height_rsun', na=False).any()
                median_std_exists = df['metric'].str.contains('median_std_height_rsun', na=False).any()
                if median_exists and median_std_exists:
                    print(f"Median and median_std already exist in {stats_file_path}, skipping...")
                    return True
            else:
                # ファイルが存在しない場合は新規作成
                df = pd.DataFrame(columns=['metric', 'value', 'time'])
            
            # 中央値と中央値ベース標準偏差の行を追加
            new_rows = pd.DataFrame({
                'metric': ['median_height_rsun', 'median_std_height_rsun'],
                'value': [median_info['median_height_rsun'], median_info['median_std_height_rsun']],
                'time': [median_info['time'], median_info['time']]
            })
            
            df = pd.concat([df, new_rows], ignore_index=True)
            
            # CSVファイルに保存
            df.to_csv(stats_file_path, index=False)
            print(f"Added median to: {stats_file_path}")
            print(f"  Median value: {median_info['median_height_rsun']:.3f} Rs")
            print(f"  Median std value: {median_info['median_std_height_rsun']:.3f} Rs")
            print(f"  Data points: {median_info['n_points']}")
            
            return True
            
        except Exception as e:
            print(f"Error adding median to {stats_file_path}: {e}")
            return False
    
    def process_time_range(self, start_time=None, end_time=None, backup=True):
        """
        指定時間範囲のnoseファイルを処理
        """
        print(f"Processing nose time range: {start_time} - {end_time}")
        
        # observations ファイルを検索
        obs_files = self.find_observations_files(start_time, end_time)
        print(f"Found {len(obs_files)} nose observations files")
        
        results = {
            'processed': 0,
            'successful': 0,
            'skipped': 0,
            'errors': 0
        }
        
        for obs_file in obs_files:
            print(f"\nProcessing: {os.path.basename(obs_file)}")
            results['processed'] += 1
            
            # 中央値を計算
            median_info = self.calculate_median_from_observations(obs_file)
            if median_info is None:
                results['errors'] += 1
                continue
            
            # 対応するstatistics ファイルを検索
            stats_file = self.find_statistics_file(obs_file)
            if stats_file is None:
                print(f"Warning: No corresponding statistics file found for {obs_file}")
                results['skipped'] += 1
                continue
            
            # 中央値を追記
            success = self.add_median_to_statistics(stats_file, median_info, backup)
            if success:
                results['successful'] += 1
            else:
                results['errors'] += 1
        
        # 結果サマリー
        print(f"\n=== Nose Processing Summary ===")
        print(f"Total files processed: {results['processed']}")
        print(f"Successfully updated: {results['successful']}")
        print(f"Skipped (no stats file): {results['skipped']}")
        print(f"Errors: {results['errors']}")
        print("===============================")
        
        return results


def main():
    """
    メイン関数：コマンドライン引数の処理と実行
    """
    parser = argparse.ArgumentParser(
        description='nose統計データに中央値を追加',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python add_median_to_nose_statistics.py --start_time "2022-06-13T03:00:00" --end_time "2022-06-13T04:00:00"
        """
    )
    
    parser.add_argument('--start_time', type=str,
                       help='開始時刻 (ISO format: YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('--end_time', type=str,
                       help='終了時刻 (ISO format: YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('--csv_folder', type=str, 
                       default='../CME_measurement/csv_folder',
                       help='CSVフォルダのパス (default: ../CME_measurement/csv_folder)')
    parser.add_argument('--no_backup', action='store_true',
                       help='バックアップを作成しない')
    
    args = parser.parse_args()
    
    try:
        # プロセッサーの初期化
        processor = MedianStatisticsAdderNose(args.csv_folder)
        
        # 処理実行
        results = processor.process_time_range(
            args.start_time,
            args.end_time,
            backup=not args.no_backup
        )
        
        # 終了コード決定
        if results['errors'] > 0:
            return 1
        else:
            return 0
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())