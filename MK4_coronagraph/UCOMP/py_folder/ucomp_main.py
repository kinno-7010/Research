"""
UCOMPデータ解析メインスクリプト
使用例とメイン実行関数
"""

from ucomp_config import *
from ucomp_scanner import *
from ucomp_plotting import *
from ucomp_plotting_groups import *

def main_ucomp_analysis():
    """
    UCOMPデータ解析のメイン関数
    対話的に解析を実行
    """
    print("=" * 60)
    print("UCOMP Data Analysis Tool")
    print("=" * 60)
    
    # デフォルト設定
    default_start = "2022-06-13T03:00:00"
    default_end = "2022-06-13T04:01:00"
    default_target = "2022-06-13T03:36:00"
    
    print("\n=== 設定 ===")
    print(f"データディレクトリ: {UCOMP_DATA_DIR}")
    print(f"波長設定: 全波長（時間範囲内のデータを自動選択）")
    print(f"利用可能な波長: 637, 706, 789, 1074, 1079 nm")
    
    # ユーザー入力
    print("\n=== パラメータ設定 ===")
    
    start_time = input(f"スキャン開始時刻 [default: {default_start} (Enter)]: ").strip()
    if not start_time:
        start_time = default_start
    
    end_time = input(f"スキャン終了時刻 [default: {default_end} (Enter)]: ").strip()
    if not end_time:
        end_time = default_end
    
    target_time = input(f"プロット対象時刻 [default: {default_target} (Enter)]: ").strip()
    if not target_time:
        target_time = default_target
    
    # 解析タイプ選択
    print("\n=== 解析タイプ選択 ===")
    print("=== 画像プロット ===")
    print("1. Group 1: Stokes I Basic (Ext 1-3)")
    print("2. Group 2: Doppler & Line Width (Ext 4-6)")
    print("3. Group 3: Stokes Parameters (Ext 7-10)")
    print("4. Group 4: Magnetic Field Azimuth (Ext 11-12)")
    print("5. 全グループ表示 (1-4)")
    print("6. 全Extension表示 (3×4) [旧版]")
    print("7. 単一Extension詳細表示")
    print("8. 利用可能時刻一覧表示")
    print("9. Custom Group 1: Ext 3,4,5,12 (2×2)")
    print("10. Custom Group 2: Ext 7-10 (2×2)")
    print("")
    print("=== 動画作成 ===")
    print("11. Group 1 時系列動画作成 (Ext 1-3)")
    print("12. Group 2 時系列動画作成 (Ext 4-6)")
    print("13. Group 3 時系列動画作成 (Ext 7-10)")
    print("14. Group 4 時系列動画作成 (Ext 11-12)")
    print("15. 全グループ時系列動画作成 (1-4)")
    print("16. Custom Group 1 時系列動画作成 (Ext 3,4,5,12)")
    print("17. Custom Group 2 時系列動画作成 (Ext 7-10)")
    
    analysis_type = input("選択 (1-17) [1]: ").strip()
    if not analysis_type:
        analysis_type = "1"
    
    try:
        # 解析実行
        if analysis_type == "1":
            print("\n=== Group 1: Stokes I Basic (Ext 1-3) ===")
            fig = plot_ucomp_extensions_group1(target_time, start_time, end_time, wavelength)
            if fig:
                plt.show()
                
        elif analysis_type == "2":
            print("\n=== Group 2: Doppler & Line Width (Ext 4-6) ===")
            fig = plot_ucomp_extensions_group2(target_time, start_time, end_time, wavelength)
            if fig:
                plt.show()
                
        elif analysis_type == "3":
            print("\n=== Group 3: Stokes Parameters (Ext 7-10) ===")
            fig = plot_ucomp_extensions_group3(target_time, start_time, end_time, wavelength)
            if fig:
                plt.show()
                
        elif analysis_type == "4":
            print("\n=== Group 4: Magnetic Field Azimuth (Ext 11-12) ===")
            fig = plot_ucomp_extensions_group4(target_time, start_time, end_time, wavelength)
            if fig:
                plt.show()
                
        elif analysis_type == "5":
            print("\n=== 全グループ表示 (1-4) ===")
            figs = plot_all_ucomp_groups(target_time, start_time, end_time, wavelength)
            for fig in figs:
                if fig:
                    plt.show()
                
        elif analysis_type == "6":
            print("\n=== 全Extension表示 (旧版) ===")
            fig = plot_ucomp_extensions(target_time, start_time, end_time, None)
            if fig:
                plt.show()
                
        elif analysis_type == "7":
            print("\n=== 単一Extension詳細表示 ===")
            ext_num = input("Extension番号 (1-12) [1]: ").strip()
            if not ext_num:
                ext_num = 1
            else:
                ext_num = int(ext_num)
            
            # Extension 12の場合は平滑化オプションを提供
            smooth_ext12 = False
            ext12_sigma = 1.0
            if ext_num == 12:
                smooth_input = input("Extension 12の平滑化を適用しますか？ (y/n) [n]: ").strip().lower()
                if smooth_input == 'y' or smooth_input == 'yes':
                    smooth_ext12 = True
                    sigma_input = input("平滑化シグマ値 [1.0]: ").strip()
                    if sigma_input:
                        ext12_sigma = float(sigma_input)
            
            # fig = plot_single_extension(target_time, start_time, end_time, ext_num, None,
            #                           smooth_ext12=smooth_ext12, ext12_sigma=ext12_sigma)
            # if fig:
            #     plt.show()
                
            fig, axes = plt.subplots(1,3,figsize=(27,8),tight_layout=True)
            for ax, target_time in zip(axes[0:2], ["2022-06-13T03:06:00", "2022-06-13T03:36:00"]):
                plot_single_extension(ax, target_time, start_time, end_time, ext_num, None,
                                      smooth_ext12=smooth_ext12, ext12_sigma=ext12_sigma)
            
            plot_ext_12_diff(axes[2], "2022-06-13T03:06:00", "2022-06-13T03:36:00", start_time, end_time, extension_num=ext_num, wavelength=None, smooth_ext12=smooth_ext12, ext12_sigma=ext12_sigma)
            
            plt.show()
                
        elif analysis_type == "8":
            print("\n=== 利用可能時刻一覧 ===")
            times = get_available_ucomp_times(start_time, end_time, None)
            if times:
                print(f"Found {len(times)} UCOMP observations for all wavelengths:")
                for i, time_obj in enumerate(times[:20]):  # 最初の20個のみ表示
                    print(f"  {i+1:2d}: {time_obj.iso}")
                if len(times) > 20:
                    print(f"  ... and {len(times) - 20} more")
            else:
                print("No UCOMP data found for the specified parameters")
                
        elif analysis_type == "9":
            print("\n=== Custom Group 1: Ext 3,4,5,12 (2×2) ===")
            # Extension 12の平滑化オプション
            smooth_input = input("Extension 12の平滑化を適用しますか？ (y/n) [n]: ").strip().lower()
            smooth_ext12 = smooth_input == 'y' or smooth_input == 'yes'
            ext12_sigma = 1.0
            if smooth_ext12:
                sigma_input = input("平滑化シグマ値 [1.0]: ").strip()
                if sigma_input:
                    ext12_sigma = float(sigma_input)
            
            fig = plot_ucomp_extensions_custom1(target_time, start_time, end_time, None,
                                              smooth_ext12=smooth_ext12, ext12_sigma=ext12_sigma)
            if fig:
                plt.show()
                
        elif analysis_type == "10":
            print("\n=== Custom Group 2: Ext 7-10 (2×2) ===")
            fig = plot_ucomp_extensions_custom2(target_time, start_time, end_time, None)
            if fig:
                plt.show()
                
        elif analysis_type == "11":
            print("\n=== Group 1 時系列動画作成 (Ext 1-3) ===")
            fps = input("フレームレート (fps) [2]: ").strip()
            fps = int(fps) if fps else 2
            interval = input("フレーム間隔 (ms) [500]: ").strip()
            interval = int(interval) if interval else 500
            
            path = create_ucomp_animation_group1(start_time, end_time, None, fps, interval)
            if path:
                print(f"動画作成完了: {path}")
                
        elif analysis_type == "12":
            print("\n=== Group 2 時系列動画作成 (Ext 4-6) ===")
            fps = input("フレームレート (fps) [2]: ").strip()
            fps = int(fps) if fps else 2
            interval = input("フレーム間隔 (ms) [500]: ").strip()
            interval = int(interval) if interval else 500
            
            path = create_ucomp_animation_group2(start_time, end_time, None, fps, interval)
            if path:
                print(f"動画作成完了: {path}")
                
        elif analysis_type == "13":
            print("\n=== Group 3 時系列動画作成 (Ext 7-10) ===")
            fps = input("フレームレート (fps) [2]: ").strip()
            fps = int(fps) if fps else 2
            interval = input("フレーム間隔 (ms) [500]: ").strip()
            interval = int(interval) if interval else 500
            
            path = create_ucomp_animation_group3(start_time, end_time, None, fps, interval)
            if path:
                print(f"動画作成完了: {path}")
                
        elif analysis_type == "14":
            print("\n=== Group 4 時系列動画作成 (Ext 11-12) ===")
            fps = input("フレームレート (fps) [2]: ").strip()
            fps = int(fps) if fps else 2
            interval = input("フレーム間隔 (ms) [500]: ").strip()
            interval = int(interval) if interval else 500
            
            path = create_ucomp_animation_group4(start_time, end_time, None, fps, interval)
            if path:
                print(f"動画作成完了: {path}")
                
        elif analysis_type == "15":
            print("\n=== 全グループ時系列動画作成 (1-4) ===")
            fps = input("フレームレート (fps) [2]: ").strip()
            fps = int(fps) if fps else 2
            interval = input("フレーム間隔 (ms) [500]: ").strip()
            interval = int(interval) if interval else 500
            
            # 波長に関係なく時間範囲内のデータを使用
            paths = create_all_ucomp_animations(start_time, end_time, None, fps, interval)
            if paths:
                print(f"全動画作成完了: {len(paths)} files created")
                for path in paths:
                    print(f"  - {path}")
        
        elif analysis_type == "16":
            print("\n=== Custom Group 1 時系列動画作成 (Ext 3,4,5,12) ===")
            fps = input("フレームレート (fps) [2]: ").strip()
            fps = int(fps) if fps else 2
            interval = input("フレーム間隔 (ms) [500]: ").strip()
            interval = int(interval) if interval else 500
            
            # Extension 12の平滑化オプション
            smooth_input = input("Extension 12の平滑化を適用しますか？ (y/n) [n]: ").strip().lower()
            smooth_ext12 = smooth_input == 'y' or smooth_input == 'yes'
            ext12_sigma = 1.0
            if smooth_ext12:
                sigma_input = input("平滑化シグマ値 [1.0]: ").strip()
                if sigma_input:
                    ext12_sigma = float(sigma_input)
            
            # タイプ15の設定を参照：波長に関係なく時間範囲内のデータを使用
            path = create_ucomp_animation_custom1(start_time, end_time, None, fps, interval,
                                                smooth_ext12=smooth_ext12, ext12_sigma=ext12_sigma)
            if path:
                print(f"動画作成完了: {path}")
                
        elif analysis_type == "17":
            print("\n=== Custom Group 2 時系列動画作成 (Ext 7-10) ===")
            fps = input("フレームレート (fps) [2]: ").strip()
            fps = int(fps) if fps else 2
            interval = input("フレーム間隔 (ms) [500]: ").strip()
            interval = int(interval) if interval else 500
            
            # タイプ15の設定を参照：波長に関係なく時間範囲内のデータを使用
            path = create_ucomp_animation_custom2(start_time, end_time, None, fps, interval)
            if path:
                print(f"動画作成完了: {path}")
        
        else:
            print("Invalid selection")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # コマンドライン引数をチェック
    import sys

    main_ucomp_analysis()