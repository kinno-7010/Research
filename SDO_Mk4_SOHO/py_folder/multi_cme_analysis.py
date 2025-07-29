# -*- coding: utf-8 -*-
"""
複数時刻CME解析モジュール
coronagraph_analysis.pyのカスタム時刻解析機能を独立化
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.time import Time
import astropy.units as u

# 同じディレクトリのモジュールをインポート
try:
    from cme_measurement import analyze_single_time_cme_multi_points
    from claude_analysis_utils import analyze_single_time_cme_with_diff_image, compare_cme_heights_multiple_times
except ImportError as e:
    print(f"モジュールのインポートエラー: {e}")
    sys.exit(1)


def save_cme_measurements_to_csv(result, target_time_str, base_dir='/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/CME_measurement/'):
    """
    CME測定結果をCSVファイルに保存（観測量と統計量を分離）
    
    Parameters:
    -----------
    result : dict
        analyze_single_time_cme_multi_points からの結果（Noneまたは空の場合も対応）
    target_time_str : str
        対象時刻文字列
    base_dir : str
        ベースディレクトリ
    """
    
    import os
    
    # 出力ディレクトリを設定
    obs_dir = os.path.join(base_dir, 'csv_folder', 'obs')
    stats_dir = os.path.join(base_dir, 'csv_folder', 'sta')
    png_dir = os.path.join(base_dir, 'analysis_png')
    
    # ディレクトリを作成（既に存在する場合はスキップ）
    os.makedirs(obs_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    
    # ファイル名を生成
    time_label = target_time_str.replace(':', '').replace('-', '').replace('T', '_')
    obs_filename = os.path.join(obs_dir, f'cme_observations_{time_label}.csv')
    stats_filename = os.path.join(stats_dir, f'cme_statistics_{time_label}.csv')
    
    # データが無い場合はNaNで埋める
    if not result or not result.get('heights') or len(result.get('heights', [])) == 0:
        print("測定データがありません。NaNでCSVを作成します。")
        
        # 観測量データ（NaN）
        obs_data = [{
            'point_id': 1,
            'time': target_time_str,
            'x_arcsec': np.nan,
            'y_arcsec': np.nan,
            'x_pixel': np.nan,
            'y_pixel': np.nan,
            'height_rsun': np.nan,
            'position_angle_deg': np.nan
        }]
        
        # 統計量データ（NaN）
        stats_data = [
            {'metric': 'mean_height_rsun', 'value': np.nan, 'time': target_time_str},
            {'metric': 'std_height_rsun', 'value': np.nan, 'time': target_time_str},
            {'metric': 'min_height_rsun', 'value': np.nan, 'time': target_time_str},
            {'metric': 'max_height_rsun', 'value': np.nan, 'time': target_time_str},
            {'metric': 'height_range_rsun', 'value': np.nan, 'time': target_time_str},
            {'metric': 'n_points', 'value': 0, 'time': target_time_str}
        ]
        
        obs_df = pd.DataFrame(obs_data)
        stats_df = pd.DataFrame(stats_data)
        
        # CSVファイルに保存
        obs_df.to_csv(obs_filename, index=False, float_format='%.3f')
        stats_df.to_csv(stats_filename, index=False, float_format='%.3f')
        
        return obs_filename, stats_filename, png_dir
    
    # 観測量データフレームを作成
    obs_data = []
    for i, (height, position, angle) in enumerate(zip(result['heights'], result['positions'], result['angles'])):
        # arcsec座標に変換（太陽中心からの角度距離）
        # 1 R☉ = 959.63 arcsec (標準値)
        x_arcsec = position[0] * 959.63 / 80  # px_per_rsun = 80 として仮定
        y_arcsec = position[1] * 959.63 / 80
        
        obs_data.append({
            'point_id': i + 1,
            'time': target_time_str,
            'x_arcsec': x_arcsec,
            'y_arcsec': y_arcsec,
            'x_pixel': position[0],
            'y_pixel': position[1],
            'height_rsun': height,
            'position_angle_deg': angle
        })
    
    obs_df = pd.DataFrame(obs_data)
    
    # 統計量データフレームを作成
    stats = result['statistics']
    stats_data = [
        {'metric': 'mean_height_rsun', 'value': stats['mean_height'], 'time': target_time_str},
        {'metric': 'std_height_rsun', 'value': stats['std_height'], 'time': target_time_str},
        {'metric': 'min_height_rsun', 'value': stats['min_height'], 'time': target_time_str},
        {'metric': 'max_height_rsun', 'value': stats['max_height'], 'time': target_time_str},
        {'metric': 'height_range_rsun', 'value': stats['height_range'], 'time': target_time_str},
        {'metric': 'n_points', 'value': stats['n_points'], 'time': target_time_str}
    ]
    
    stats_df = pd.DataFrame(stats_data)
    
    # CSVファイルに保存
    obs_df.to_csv(obs_filename, index=False, float_format='%.3f')
    stats_df.to_csv(stats_filename, index=False, float_format='%.3f')
    
    return obs_filename, stats_filename, png_dir


def single_time_analysis(target_time: str, show_diff_image: bool = True):
    """単一時刻の解析を実行"""
    
    print(f"\n=== 単一時刻解析: {target_time} ===")
    
    try:
        if show_diff_image:
            # 差分画像表示付きの解析を実行
            print("差分画像表示付きCME解析を開始します...")
            result = analyze_single_time_cme_with_diff_image(
                target_time, 
                save_results=True,
                output_dir='./cme_analysis/'
            )
        else:
            # CME複数点測定のみを実行
            print("単一時刻CME複数点測定を開始します...")
            result = analyze_single_time_cme_multi_points(
                target_time, 
                save_results=True,
                output_dir='./cme_analysis/'
            )
        
        # CSVファイルに保存（結果がNoneでも実行）
        csv_files = save_cme_measurements_to_csv(result, target_time)
        if csv_files:
            obs_file, stats_file, png_dir = csv_files
            print(f"観測量CSV保存完了: {obs_file}")
            print(f"統計量CSV保存完了: {stats_file}")
            print(f"PNGファイル保存先: {png_dir}")
        
        if result:
            stats = result['statistics']
            print(f"\n解析結果:")
            print(f"  測定点数: {stats['n_points']}")
            print(f"  平均高度: {stats['mean_height']:.2f} ± {stats['std_height']:.2f} R☉")
            print(f"  最小高度: {stats['min_height']:.2f} R☉")
            print(f"  最大高度: {stats['max_height']:.2f} R☉")
            print(f"  高度範囲: {stats['height_range']:.2f} R☉")
            
            return result
        else:
            print("測定データが取得できませんでした。")
            return None
            
    except Exception as e:
        print(f"単一時刻解析でエラー: {e}")
        # エラーの場合でもCSVファイルにNaNを保存
        csv_files = save_cme_measurements_to_csv(None, target_time)
        if csv_files:
            obs_file, stats_file, png_dir = csv_files
            print(f"エラー時観測量CSV保存完了: {obs_file}")
            print(f"エラー時統計量CSV保存完了: {stats_file}")
        return None


def multi_analysis():
    """カスタム時刻リストでの複数CME解析"""
    
    print("\n=== カスタム時刻リスト解析 ===")
    
    # 入力方法の選択
    print("1. 手動で時刻リストを入力")
    print("2. 開始・終了時刻と間隔を指定")
    input_method = input("入力方法を選択 (1/2) [1]: ").strip()
    
    time_list = []
    
    if input_method == "2":
        # 開始・終了時刻と間隔による自動生成
        start_time = input("開始時刻 (例: 2022-06-13T03:00:00): ").strip()
        end_time = input("終了時刻 (例: 2022-06-13T04:00:00): ").strip()
        interval_str = input("時間間隔（分） [12]: ").strip()
        
        try:
            interval_min = int(interval_str) if interval_str else 12
            start_t = Time(start_time)
            end_t = Time(end_time)
            
            current_t = start_t
            while current_t <= end_t:
                time_list.append(current_t.iso)
                current_t += interval_min * u.min
                
            print(f"生成された時刻リスト ({len(time_list)}個):")
            for i, t in enumerate(time_list, 1):
                print(f"  {i}. {t}")
                
        except Exception as e:
            print(f"時刻生成エラー: {e}")
            return
    else:
        # 手動入力
        print("カスタム時刻リストを入力してください (カンマ区切り):")
        time_input = input("例: 2022-06-13T03:00:00,2022-06-13T03:12:00: ").strip()
        if time_input:
            time_list = [t.strip() for t in time_input.split(",")]
        else:
            print("無効な入力です。")
            return
    
    if time_list:
        print(f"\n{len(time_list)}個の時刻で解析を実行します...")
        show_diff = input("差分画像を表示しますか？ (y/n) [y]: ").strip().lower()
        show_diff_images = show_diff != 'n'
        
        try:
            results = []
            for i, target_time in enumerate(time_list, 1):
                print(f"\n--- {i}/{len(time_list)}: {target_time} ---")
                result = single_time_analysis(target_time, show_diff_images)
                if result:
                    results.append(result)
                    # 各時刻でCSV保存を実行
                    csv_files = save_cme_measurements_to_csv(result, target_time)
                    if csv_files:
                        obs_file, stats_file, png_dir = csv_files
                        print(f"観測量CSV保存完了: {obs_file}")
                        print(f"統計量CSV保存完了: {stats_file}")
            
            if len(results) > 1:
                print(f"\n=== 複数時刻比較解析を実行中... ===")
                comparison_results = compare_cme_heights_multiple_times(
                    time_list,
                    save_comparison=True,
                    output_dir='./cme_analysis/'
                )
                print(f"解析完了: {len(results)}個の結果")
            else:
                print(f"解析完了: {len(results)}個の結果")
                
        except Exception as e:
            print(f"カスタム解析でエラー: {e}")
    else:
        print("無効な入力です。")


if __name__ == "__main__":
    multi_analysis()