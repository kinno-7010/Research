"""
Claude用太陽物理学解析スクリプト（モジュール使用版）
SDO_Mk4_SOHO/py_folder内のモジュールを使用してCME解析を実行

使用例:
    python3 claude_analysis.py
"""

import sys
from pathlib import Path

# SDO_Mk4_SOHO/py_folderをパスに追加
sys.path.append(str(Path(__file__).parent / "SDO_Mk4_SOHO" / "py_folder"))

try:
    # 必要なモジュールをインポート
    from config import *
    import config
    from claude_analysis_utils import (
        analyze_single_time_cme_with_diff_image,
        analyze_single_time_cme_with_diff_from_min_image,
        analyze_single_time_cme_with_raw_image,
        compare_cme_heights_multiple_times,
        run_cme_analysis_workflow
    )
    from integrated_analysis import create_single_diff_image, create_single_integrated_image, clear_scan_cache, get_cache_info
    from cme_measurement import analyze_single_time_cme_multi_points
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from astropy.time import Time
    
    print("=== 太陽物理学CME解析ツール ===")
    print("モジュールの読み込みが完了しました。")
    
except ImportError as e:
    print(f"モジュールのインポートエラー: {e}")
    print("SDO_Mk4_SOHO/py_folderが正しく配置されているか確認してください。")
    sys.exit(1)


def main():
    """メイン実行関数"""
    
    print("\n=== CME解析を開始します ===")
    
    # デフォルトの解析対象時刻リスト
    default_time_series = [
        '2022-06-13T03:00:00',
        '2022-06-13T03:12:00',
        '2022-06-13T03:24:00',
        '2022-06-13T03:36:00',
        '2022-06-13T03:48:00',
        '2022-06-13T04:00:00'
    ]
    
    print(f"解析対象時刻: {len(default_time_series)}個")
    for i, time_str in enumerate(default_time_series, 1):
        print(f"  {i}. {time_str}")
    
    try:
        # CME解析ワークフローを実行
        results = run_cme_analysis_workflow(default_time_series)
        
        if results and len(results) > 0:
            print(f"\n=== 解析結果サマリー ===")
            print(f"成功した解析数: {len(results)}")
            
            # 各時刻の解析結果を表示してCSV保存
            for result in results:
                stats = result['statistics']
                time_str = result['time']
                print(f"\n時刻: {time_str}")
                print(f"  測定点数: {stats['n_points']}")
                print(f"  平均高度: {stats['mean_height']:.2f} ± {stats['std_height']:.2f} R☉")
                print(f"  高度範囲: {stats['min_height']:.2f} - {stats['max_height']:.2f} R☉")
                
                # CSV保存を実行（MK4時刻を取得して渡す）
                mk4_time_str = None
                if hasattr(result, 'get') and result.get('mk4_time'):
                    mk4_time_str = result['mk4_time']
                csv_files = save_cme_measurements_to_csv(result, time_str, mk4_time_str=mk4_time_str)
                if csv_files:
                    obs_file, stats_file, png_dir = csv_files
                    print(f"  観測量CSV保存: {obs_file}")
                    print(f"  統計量CSV保存: {stats_file}")
        else:
            print("解析結果が得られませんでした。")
            
    except Exception as e:
        print(f"解析実行中にエラーが発生しました: {e}")
        print("ログを確認してトラブルシューティングを行ってください。")


def save_cme_measurements_to_csv(result, target_time_str, base_dir='/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/CME_measurement/', mk4_time_str=None):
    """
    CME測定結果をCSVファイルに保存（観測量と統計量を分離）
    
    Parameters:
    -----------
    result : dict
        analyze_single_time_cme_multi_points からの結果（Noneまたは空の場合も対応）
    target_time_str : str
        対象時刻文字列（fallback用）
    base_dir : str
        ベースディレクトリ
    mk4_time_str : str, optional
        MK4データの実際の時刻文字列（優先使用）
    """
    
    import os
    from astropy.time import Time
    
    # 出力ディレクトリを設定
    obs_dir = os.path.join(base_dir, 'csv_folder', 'obs')
    stats_dir = os.path.join(base_dir, 'csv_folder', 'sta')
    png_dir = os.path.join(base_dir, 'analysis_png')
    
    # ディレクトリを作成（既に存在する場合はスキップ）
    os.makedirs(obs_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    
    # 使用する時刻を決定（MK4時刻が優先、なければtarget_time）
    use_time_str = mk4_time_str if mk4_time_str else target_time_str
    
    # ファイル名用の時刻ラベルを生成（YYMMDD-HHMMSS形式）
    try:
        time_obj = Time(use_time_str)
        time_label = time_obj.datetime.strftime('%y%m%d-%H%M%S')
    except:
        # フォールバック：従来の方式
        time_label = target_time_str.replace(':', '').replace('-', '').replace('T', '_')
    
    # obs_filename = os.path.join(obs_dir, f'nose_cme_observations_{time_label}.csv')
    # stats_filename = os.path.join(stats_dir, f'nose_cme_statistics_{time_label}.csv')
    obs_filename = os.path.join(obs_dir, f'cme_observations_{time_label}_1min_diff.csv')
    stats_filename = os.path.join(stats_dir, f'cme_statistics_{time_label}_1min_diff.csv')
    
    
    # データが無い場合はNaNで埋める
    if not result or not result.get('heights') or len(result.get('heights', [])) == 0:
        print("測定データがありません。NaNでCSVを作成します。")
        
        # 観測量データ（NaN）
        obs_data = [{
            'point_id': 1,
            'time': use_time_str,
            'x_arcsec': np.nan,
            'y_arcsec': np.nan,
            'x_pixel': np.nan,
            'y_pixel': np.nan,
            'height_rsun': np.nan,
            'position_angle_deg': np.nan
        }]
        
        # 統計量データ（NaN）
        stats_data = [
            {'metric': 'mean_height_rsun', 'value': np.nan, 'time': use_time_str},
            {'metric': 'std_height_rsun', 'value': np.nan, 'time': use_time_str},
            {'metric': 'min_height_rsun', 'value': np.nan, 'time': use_time_str},
            {'metric': 'max_height_rsun', 'value': np.nan, 'time': use_time_str},
            {'metric': 'height_range_rsun', 'value': np.nan, 'time': use_time_str},
            {'metric': 'n_points', 'value': 0, 'time': use_time_str}
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
            'time': use_time_str,
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
        {'metric': 'mean_height_rsun', 'value': stats['mean_height'], 'time': use_time_str},
        {'metric': 'std_height_rsun', 'value': stats['std_height'], 'time': use_time_str},
        {'metric': 'min_height_rsun', 'value': stats['min_height'], 'time': use_time_str},
        {'metric': 'max_height_rsun', 'value': stats['max_height'], 'time': use_time_str},
        {'metric': 'height_range_rsun', 'value': stats['height_range'], 'time': use_time_str},
        {'metric': 'n_points', 'value': stats['n_points'], 'time': use_time_str}
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
        mk4_time_str = None
        if result and hasattr(result, 'get') and result.get('mk4_time'):
            mk4_time_str = result['mk4_time']
        csv_files = save_cme_measurements_to_csv(result, target_time, mk4_time_str=mk4_time_str)
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
        csv_files = save_cme_measurements_to_csv(None, target_time, mk4_time_str=None)
        if csv_files:
            obs_file, stats_file, png_dir = csv_files
            print(f"エラー時観測量CSV保存完了: {obs_file}")
            print(f"エラー時統計量CSV保存完了: {stats_file}")
        return None


def single_time_analysis_from_min(target_time: str, show_diff_image: bool = True):
    """
    単一時刻の解析を実行（2分前のデータから差分を作成）
    
    Parameters:
    -----------
    target_time : str
        解析対象時刻
    show_diff_image : bool
        差分画像を表示するかどうか
        
    Returns:
    --------
    dict or None
        解析結果
    """
    
    print(f"\n=== 単一時刻解析（2分前差分）: {target_time} ===")
    
    try:
        from astropy.time import Time
        import astropy.units as u
        
        # 2分前の時刻を計算
        target_t = Time(target_time)
        base_t = target_t - 2 * u.min
        base_time = base_t.iso
        
        print(f"対象時刻: {target_time}")
        print(f"ベース時刻（2分前）: {base_time}")
        
        if show_diff_image:
            # 差分画像表示付きの解析を実行（2分前専用関数を使用）
            print("2分前ベース差分画像表示付きCME解析を開始します...")
            result = analyze_single_time_cme_with_diff_from_min_image(
                target_time, 
                save_results=True,
                output_dir='./cme_analysis/'
            )
        else:
            # CME複数点測定のみを実行（2分前をベース時刻として指定）
            print("2分前ベース単一時刻CME複数点測定を開始します...")
            result = analyze_single_time_cme_multi_points(
                target_time, 
                save_results=True,
                output_dir='./cme_analysis/',
                base_time=base_time
            )
        
        # CSVファイルに保存（結果がNoneでも実行）
        mk4_time_str = None
        if result and hasattr(result, 'get') and result.get('mk4_time'):
            mk4_time_str = result['mk4_time']
        csv_files = save_cme_measurements_to_csv(result, target_time, mk4_time_str=mk4_time_str)
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
        print(f"単一時刻解析（2分前差分）でエラー: {e}")
        # エラーの場合でもCSVファイルにNaNを保存
        csv_files = save_cme_measurements_to_csv(None, target_time, mk4_time_str=None)
        if csv_files:
            obs_file, stats_file, png_dir = csv_files
            print(f"エラー時観測量CSV保存完了: {obs_file}")
            print(f"エラー時統計量CSV保存完了: {stats_file}")
        return None


def raw_data_analysis(target_time: str, show_raw_image: bool = True):
    """
    単一時刻の生データ解析を実行（差分なし）
    
    Parameters:
    -----------
    target_time : str
        解析対象時刻
    show_raw_image : bool
        生データ画像を表示するかどうか
        
    Returns:
    --------
    dict or None
        解析結果
    """
    
    print(f"\n=== 単一時刻解析（生データ）: {target_time} ===")
    
    try:
        if show_raw_image:
            # 生データ画像表示付きの解析を実行
            print("生データ画像表示付きCME解析を開始します...")
            result = analyze_single_time_cme_with_raw_image(
                target_time, 
                save_results=True,
                output_dir='./cme_analysis/'
            )
        else:
            # CME複数点測定のみを実行（生データ用）
            print("生データ単一時刻CME複数点測定を開始します...")
            result = analyze_single_time_cme_multi_points(
                target_time, 
                save_results=True,
                output_dir='./cme_analysis/'
            )
        
        # CSVファイルに保存（結果がNoneでも実行）- raw_プレフィックス付き
        mk4_time_str = None
        if result and hasattr(result, 'get') and result.get('mk4_time'):
            mk4_time_str = result['mk4_time']
        
        # raw用CSV保存関数を呼び出し
        csv_files = save_raw_cme_measurements_to_csv(result, target_time, mk4_time_str=mk4_time_str)
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
        print(f"単一時刻解析（生データ）でエラー: {e}")
        # エラーの場合でもCSVファイルにNaNを保存
        csv_files = save_raw_cme_measurements_to_csv(None, target_time, mk4_time_str=None)
        if csv_files:
            obs_file, stats_file, png_dir = csv_files
            print(f"エラー時観測量CSV保存完了: {obs_file}")
            print(f"エラー時統計量CSV保存完了: {stats_file}")
        return None


def save_raw_cme_measurements_to_csv(result, target_time_str, base_dir='/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/CME_measurement/', mk4_time_str=None):
    """
    CME測定結果をCSVファイルに保存（生データ用・raw_プレフィックス付き）
    
    Parameters:
    -----------
    result : dict
        analyze_single_time_cme_with_raw_image からの結果
    target_time_str : str
        対象時刻文字列
    base_dir : str
        ベースディレクトリ
    mk4_time_str : str, optional
        MK4データの実際の時刻文字列
    """
    
    import os
    from astropy.time import Time
    
    # 出力ディレクトリを設定
    obs_dir = os.path.join(base_dir, 'csv_folder', 'obs')
    stats_dir = os.path.join(base_dir, 'csv_folder', 'sta')
    png_dir = os.path.join(base_dir, 'analysis_png')
    
    # ディレクトリを作成（既に存在する場合はスキップ）
    os.makedirs(obs_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    
    # 使用する時刻を決定
    use_time_str = mk4_time_str if mk4_time_str else target_time_str
    
    # ファイル名用の時刻ラベルを生成
    try:
        time_obj = Time(use_time_str)
        time_label = time_obj.datetime.strftime('%y%m%d-%H%M%S')
    except:
        time_label = target_time_str.replace(':', '').replace('-', '').replace('T', '_')
    
    obs_filename = os.path.join(obs_dir, f'raw_cme_observations_{time_label}.csv')
    stats_filename = os.path.join(stats_dir, f'raw_cme_statistics_{time_label}.csv')
    
    # データが無い場合はNaNで埋める
    if not result or not result.get('heights') or len(result.get('heights', [])) == 0:
        print("測定データがありません。NaNでCSVを作成します。")
        
        # 観測量データ（NaN）
        obs_data = [{
            'point_id': 1,
            'time': use_time_str,
            'x_arcsec': np.nan,
            'y_arcsec': np.nan,
            'x_pixel': np.nan,
            'y_pixel': np.nan,
            'height_rsun': np.nan,
            'position_angle_deg': np.nan
        }]
        
        # 統計量データ（NaN）
        stats_data = [
            {'metric': 'mean_height_rsun', 'value': np.nan, 'time': use_time_str},
            {'metric': 'std_height_rsun', 'value': np.nan, 'time': use_time_str},
            {'metric': 'min_height_rsun', 'value': np.nan, 'time': use_time_str},
            {'metric': 'max_height_rsun', 'value': np.nan, 'time': use_time_str},
            {'metric': 'height_range_rsun', 'value': np.nan, 'time': use_time_str},
            {'metric': 'n_points', 'value': 0, 'time': use_time_str}
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
        # arcsec座標に変換
        x_arcsec = position[0] * 959.63 / 80
        y_arcsec = position[1] * 959.63 / 80
        
        obs_data.append({
            'point_id': i + 1,
            'time': use_time_str,
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
        {'metric': 'mean_height_rsun', 'value': stats['mean_height'], 'time': use_time_str},
        {'metric': 'std_height_rsun', 'value': stats['std_height'], 'time': use_time_str},
        {'metric': 'min_height_rsun', 'value': stats['min_height'], 'time': use_time_str},
        {'metric': 'max_height_rsun', 'value': stats['max_height'], 'time': use_time_str},
        {'metric': 'height_range_rsun', 'value': stats['height_range'], 'time': use_time_str},
        {'metric': 'n_points', 'value': stats['n_points'], 'time': use_time_str}
    ]
    
    stats_df = pd.DataFrame(stats_data)
    
    # CSVファイルに保存
    obs_df.to_csv(obs_filename, index=False, float_format='%.3f')
    stats_df.to_csv(stats_filename, index=False, float_format='%.3f')
    
    return obs_filename, stats_filename, png_dir


def interactive_analysis():
    """対話的な解析モード"""
    
    print("\n=== 対話的解析モード ===")
    print("1. 複数時刻の比較解析")
    print("2. 単一時刻の詳細解析")
    print("3. カスタム時刻リストでの解析")
    print("4. キャッシュ管理")
    print("0. 終了")
    
    while True:
        try:
            choice = input("\n選択してください (0-4): ").strip()
            
            if choice == "0":
                print("解析を終了します。")
                break
                
            elif choice == "1":
                print("複数時刻の比較解析を実行します...")
                main()
                
            elif choice == "2":
                target_time = input("解析対象時刻を入力してください (例: 2022-06-13T03:24:00): ").strip()
                if target_time:
                    show_diff = input("差分画像を表示しますか？ (y/n) [y]: ").strip().lower()
                    show_diff_image = show_diff != 'n'
                    single_time_analysis(target_time, show_diff_image)
                else:
                    print("無効な時刻です。")
                    
            elif choice == "3":
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
                        from astropy.time import Time
                        import astropy.units as u
                        
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
                        continue
                else:
                    # 手動入力
                    print("カスタム時刻リストを入力してください (カンマ区切り):")
                    time_input = input("例: 2022-06-13T03:00:00,2022-06-13T03:12:00: ").strip()
                    if time_input:
                        time_list = [t.strip() for t in time_input.split(",")]
                    else:
                        print("無効な入力です。")
                        continue
                
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
                                mk4_time_str = None
                                if hasattr(result, 'get') and result.get('mk4_time'):
                                    mk4_time_str = result['mk4_time']
                                csv_files = save_cme_measurements_to_csv(result, target_time, mk4_time_str=mk4_time_str)
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
                    
            elif choice == "4":
                print("\n=== キャッシュ管理 ===")
                print("1. キャッシュ情報を表示")
                print("2. キャッシュをクリア")
                cache_choice = input("選択 (1/2): ").strip()
                
                if cache_choice == "1":
                    print(f"現在のキャッシュ状況: {get_cache_info()}")
                elif cache_choice == "2":
                    clear_scan_cache()
                else:
                    print("無効な選択です。")
                    
            else:
                print("無効な選択です。0-4の数字を入力してください。")
                
        except KeyboardInterrupt:
            print("\n\n解析を中断しました。")
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    # コマンドライン引数をチェック

    interactive_analysis()
        # elif sys.argv[1] == "--single":
        #     if len(sys.argv) > 2:
        #         show_diff = len(sys.argv) > 3 and sys.argv[3].lower() == "--diff"
        #         single_time_analysis(sys.argv[2], show_diff)
        #     else:
        #         print("使用法: python3 claude_analysis.py --single YYYY-MM-DDTHH:MM:SS [--diff]")
        # else:
        #     print("使用法:")
        #     print("  python3 claude_analysis.py                           # デフォルト解析")
        #     print("  python3 claude_analysis.py --interactive             # 対話的モード")
        #     print("  python3 claude_analysis.py --single <時刻> [--diff]  # 単一時刻解析")
        #     print("    --diff: 差分画像を表示")
    # else:
    #     # デフォルト: 複数時刻の比較解析を実行
    #     main()