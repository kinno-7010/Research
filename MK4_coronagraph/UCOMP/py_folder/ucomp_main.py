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
    default_start = "2022-06-13T02:00:00"
    default_end = "2022-06-13T04:30:00"
    default_wavelength = 1074
    default_target = "2022-06-13T03:36:00"
    
    print("\n=== 設定 ===")
    print(f"データディレクトリ: {UCOMP_DATA_DIR}")
    print(f"デフォルト波長: {default_wavelength} nm")
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
    
    wavelength_str = input(f"波長 (nm) [default: {default_wavelength} (Enter)]: ").strip()
    if wavelength_str:
        try:
            wavelength = int(wavelength_str)
            validate_wavelength(wavelength)
        except (ValueError, Exception) as e:
            print(f"Invalid wavelength: {e}")
            wavelength = default_wavelength
    else:
        wavelength = default_wavelength
    
    # 解析タイプ選択
    print("\n=== 解析タイプ選択 ===")
    print("1. Group 1: Stokes I Basic (Ext 1-3)")
    print("2. Group 2: Doppler & Line Width (Ext 4-6)")
    print("3. Group 3: Stokes Parameters (Ext 7-10)")
    print("4. Group 4: Magnetic Field Azimuth (Ext 11-12)")
    print("5. 全グループ表示 (1-4)")
    print("6. 全Extension表示 (3×4) [旧版]")
    print("7. 単一Extension詳細表示")
    print("8. 利用可能時刻一覧表示")
    print("9. Group 1 時系列動画作成 (Ext 1-3)")
    print("10. Group 2 時系列動画作成 (Ext 4-6)")
    print("11. Group 3 時系列動画作成 (Ext 7-10)")
    print("12. Group 4 時系列動画作成 (Ext 11-12)")
    print("13. 全グループ時系列動画作成 (1-4)")
    
    analysis_type = input("選択 (1-13) [1]: ").strip()
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
            fig = plot_ucomp_extensions(target_time, start_time, end_time, wavelength)
            if fig:
                plt.show()
                
        elif analysis_type == "7":
            print("\n=== 単一Extension詳細表示 ===")
            ext_num = input("Extension番号 (1-12) [1]: ").strip()
            if not ext_num:
                ext_num = 1
            else:
                ext_num = int(ext_num)
            
            fig = plot_single_extension(target_time, start_time, end_time, ext_num, wavelength)
            if fig:
                plt.show()
                
        elif analysis_type == "8":
            print("\n=== 利用可能時刻一覧 ===")
            times = get_available_ucomp_times(start_time, end_time, wavelength)
            if times:
                print(f"Found {len(times)} UCOMP observations for {wavelength}nm:")
                for i, time_obj in enumerate(times[:20]):  # 最初の20個のみ表示
                    print(f"  {i+1:2d}: {time_obj.iso}")
                if len(times) > 20:
                    print(f"  ... and {len(times) - 20} more")
            else:
                print("No UCOMP data found for the specified parameters")
                
        elif analysis_type == "9":
            print("\n=== Group 1 時系列動画作成 (Ext 1-3) ===")
            fps = input("フレームレート (fps) [2]: ").strip()
            fps = int(fps) if fps else 2
            interval = input("フレーム間隔 (ms) [500]: ").strip()
            interval = int(interval) if interval else 500
            
            path = create_ucomp_animation_group1(target_time, start_time, end_time, wavelength, fps, interval)
            if path:
                print(f"動画作成完了: {path}")
                
        elif analysis_type == "10":
            print("\n=== Group 2 時系列動画作成 (Ext 4-6) ===")
            fps = input("フレームレート (fps) [2]: ").strip()
            fps = int(fps) if fps else 2
            interval = input("フレーム間隔 (ms) [500]: ").strip()
            interval = int(interval) if interval else 500
            
            path = create_ucomp_animation_group2(target_time, start_time, end_time, wavelength, fps, interval)
            if path:
                print(f"動画作成完了: {path}")
                
        elif analysis_type == "11":
            print("\n=== Group 3 時系列動画作成 (Ext 7-10) ===")
            fps = input("フレームレート (fps) [2]: ").strip()
            fps = int(fps) if fps else 2
            interval = input("フレーム間隔 (ms) [500]: ").strip()
            interval = int(interval) if interval else 500
            
            path = create_ucomp_animation_group3(target_time, start_time, end_time, wavelength, fps, interval)
            if path:
                print(f"動画作成完了: {path}")
                
        elif analysis_type == "12":
            print("\n=== Group 4 時系列動画作成 (Ext 11-12) ===")
            fps = input("フレームレート (fps) [2]: ").strip()
            fps = int(fps) if fps else 2
            interval = input("フレーム間隔 (ms) [500]: ").strip()
            interval = int(interval) if interval else 500
            
            path = create_ucomp_animation_group4(target_time, start_time, end_time, wavelength, fps, interval)
            if path:
                print(f"動画作成完了: {path}")
                
        elif analysis_type == "13":
            print("\n=== 全グループ時系列動画作成 (1-4) ===")
            fps = input("フレームレート (fps) [2]: ").strip()
            fps = int(fps) if fps else 2
            interval = input("フレーム間隔 (ms) [500]: ").strip()
            interval = int(interval) if interval else 500
            
            paths = create_all_ucomp_animations(start_time, end_time, wavelength, fps, interval)
            if paths:
                print(f"全動画作成完了: {len(paths)} files created")
                for path in paths:
                    print(f"  - {path}")
        
        else:
            print("Invalid selection")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def example_usage():
    """
    使用例を実行
    """
    print("UCOMP Analysis Examples")
    print("=" * 40)
    
    # パラメータ設定
    start_time = "2022-06-13T00:00:00"
    end_time = "2022-06-13T05:00:00"
    target_time = "2022-06-13T02:30:00"
    wavelength = 1074
    
    print(f"Time range: {start_time} - {end_time}")
    print(f"Target time: {target_time}")
    print(f"Wavelength: {wavelength} nm")
    
    # 例1: 利用可能な時刻を確認
    print("\n--- Example 1: Available times ---")
    times = get_available_ucomp_times(start_time, end_time, wavelength)
    print(f"Found {len(times)} observations")
    
    # 例2: 最も近いデータを探す
    print("\n--- Example 2: Find closest data ---")
    closest = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
    
    # 例3: 全Extension表示
    print("\n--- Example 3: Plot all extensions ---")
    fig1 = plot_ucomp_extensions(target_time, start_time, end_time, wavelength)
    if fig1:
        plt.show()
    
    # 例4: サマリープロット
    print("\n--- Example 4: Summary plot ---")
    fig2 = create_ucomp_summary_plot(target_time, start_time, end_time, wavelength)
    if fig2:
        plt.show()

if __name__ == "__main__":
    # コマンドライン引数をチェック
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "example":
        example_usage()
    else:
        main_ucomp_analysis()