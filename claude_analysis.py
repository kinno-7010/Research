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
        compare_cme_heights_multiple_times,
        run_cme_analysis_workflow
    )
    from integrated_analysis import create_single_diff_image, create_single_integrated_image, clear_scan_cache, get_cache_info
    from cme_measurement import analyze_single_time_cme_multi_points
    import matplotlib.pyplot as plt
    
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
            
            # 各時刻の解析結果を表示
            for result in results:
                stats = result['statistics']
                print(f"\n時刻: {result['time']}")
                print(f"  測定点数: {stats['n_points']}")
                print(f"  平均高度: {stats['mean_height']:.2f} ± {stats['std_height']:.2f} R☉")
                print(f"  高度範囲: {stats['min_height']:.2f} - {stats['max_height']:.2f} R☉")
        else:
            print("解析結果が得られませんでした。")
            
    except Exception as e:
        print(f"解析実行中にエラーが発生しました: {e}")
        print("ログを確認してトラブルシューティングを行ってください。")


def single_time_analysis(target_time: str, show_diff_image: bool = True):
    """単一時刻の解析を実行"""
    
    print(f"\n=== 単一時刻解析: {target_time} ===")
    
    try:
        # 単一時刻解析も複数時刻と同じスキームを使用
        print("単一時刻解析を開始します（複数時刻と同じクリック機能）...")
        result = analyze_single_time_cme_with_diff_image(
            target_time, 
            save_results=True,
            output_dir='./cme_analysis/'
        )
        
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
            print("解析に失敗しました。")
            return None
            
    except Exception as e:
        print(f"単一時刻解析でエラー: {e}")
        return None


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
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive":
            interactive_analysis()
        elif sys.argv[1] == "--single":
            if len(sys.argv) > 2:
                show_diff = len(sys.argv) > 3 and sys.argv[3].lower() == "--diff"
                single_time_analysis(sys.argv[2], show_diff)
            else:
                print("使用法: python3 claude_analysis.py --single YYYY-MM-DDTHH:MM:SS [--diff]")
        else:
            print("使用法:")
            print("  python3 claude_analysis.py                           # デフォルト解析")
            print("  python3 claude_analysis.py --interactive             # 対話的モード")
            print("  python3 claude_analysis.py --single <時刻> [--diff]  # 単一時刻解析")
            print("    --diff: 差分画像を表示")
    else:
        # デフォルト: 複数時刻の比較解析を実行
        main()