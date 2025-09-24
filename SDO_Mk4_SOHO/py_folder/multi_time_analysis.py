from coronagraph_analysis import single_time_analysis, single_time_analysis_from_min
from coronagraph_analysis import raw_data_analysis
from integrated_analysis import clear_scan_cache
import matplotlib.pyplot as plt
import gc


if __name__ == "__main__":
    target_start_time = input("1. 解析対象時刻を入力してください (例: 2022-06-13T03:24:00): ").strip()
    target_end_time = input("2. 解析対象時刻を入力してください (例: 2022-06-13T03:24:00): ").strip()
    target_interval = input("3. インターバルを入力してください (例: 15 [sec]): ").strip()
    
    time_list = []
    
    from astropy.time import Time
    import astropy.units as u
    
    # 2番目と3番目の入力が空の場合は、1番目の時刻のみを処理
    if not target_end_time and not target_interval:
        time_list = [target_start_time]
    else:
        # 従来の複数時刻解析ロジック
        interval_sec = int(target_interval) if target_interval else 15
        start_t = Time(target_start_time)
        end_t = Time(target_end_time) if target_end_time else Time(target_start_time)
        
        current_t = start_t
        while current_t <= end_t:
            time_list.append(current_t.iso)
            current_t += interval_sec * u.s
        
    print(f"生成された時刻リスト ({len(time_list)}個):")
    for i, t in enumerate(time_list, 1):
        print(f"  {i}. {t}")
        
    for target_time in time_list:
        single_time_analysis(target_time)
        # single_time_analysis_from_min(target_time)
        # raw_data_analysis(target_time)
        clear_scan_cache()
        plt.close('all')
        gc.collect()
        