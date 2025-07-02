#!/usr/bin/env python3
"""
CME高度計測のテストスクリプト（拡張版・複数点対応）
Base Timeからの差分に加えて前の時間との差分も使用し、複数のCME高度を検出
時間範囲: 03:00:00~04:01:00
"""

import os
import sys
from pathlib import Path
import sunpy.map
from astropy.time import Time

# 現在のディレクトリをパスに追加
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from cme_height_measurement import (
    load_kcor_nrgf_sequence,
    measure_cme_height_enhanced,
    plot_cme_height_evolution_enhanced,
    create_cme_height_measurement_movie_enhanced
)

def test_enhanced_cme_height_measurement():
    """
    拡張版CME高度計測のテスト（時間範囲制限付き）
    """
    print("=== 拡張版CME高度計測テスト開始 ===")
    
    # データディレクトリ
    data_dir = "MK4_coronagraph_KCOR/Subtraction_data/Rawdata/kcor_nrgf/20220613.kcor_nrgf.fits"
    
    # 基準時刻
    base_time_str = "2022-06-13T02:00:00"
    
    # 処理時間範囲
    start_time_str = "2022-06-13T03:00:00"
    end_time_str = "2022-06-13T04:01:00"
    
    # 出力ファイル名
    csv_output = "enhanced_cme_height_mk4_20220613_0300-0401.csv"
    plot_output = "enhanced_cme_height_evolution_mk4_20220613_0300-0401.png"
    movie_output = "enhanced_cme_height_measurement_mk4_20220613_0300-0401.mp4"
    
    try:
        # マップシーケンスを読み込み
        print("\nKCOR NGRFマップシーケンスを読み込み中...")
        map_sequence = load_kcor_nrgf_sequence(data_dir)
        
        # 時間範囲でフィルタリング
        start_time = Time(start_time_str)
        end_time = Time(end_time_str)
        
        print(f"\n時間範囲でフィルタリング中: {start_time_str} - {end_time_str}")
        filtered_maps = []
        for m in map_sequence:
            if start_time <= m.date <= end_time:
                filtered_maps.append(m)
        
        # フィルタされたマップシーケンスを作成
        if filtered_maps:
            filtered_sequence = sunpy.map.MapSequence(filtered_maps)
        else:
            print("指定された時間範囲にマップが見つかりませんでした。")
            return
        
        # 基本情報を表示
        print(f"\n全マップ数: {len(map_sequence)}")
        print(f"フィルタ後マップ数: {len(filtered_sequence)}")
        print(f"処理時間範囲: {filtered_sequence[0].date.iso} - {filtered_sequence[-1].date.iso}")
        
        # CME高度を計測（拡張版）
        print("\n拡張版CME高度を計測中...")
        df = measure_cme_height_enhanced(
            map_sequence=filtered_sequence,
            base_time_str=base_time_str,
            output_file=csv_output,
            threshold_factor=2.0,
            min_radius_rsun=1.5,
            max_radius_rsun=6.0
        )
        
        if not df.empty:
            print(f"\n=== 計測結果（拡張版） ===")
            print(f"データポイント数: {len(df)}")
            
            # データの詳細統計
            valid_data = df.dropna(subset=['Height_Rsun'])
            if not valid_data.empty:
                print(f"有効なデータポイント数: {len(valid_data)}")
                print(f"高度範囲: {valid_data['Height_Rsun'].min():.3f} - {valid_data['Height_Rsun'].max():.3f} R⊙")
                
                # 差分タイプ別の統計
                base_data = valid_data[valid_data['Diff_Type'] == 'Base_Diff']
                second_data = valid_data[valid_data['Diff_Type'] == 'Second_Diff']
                
                print(f"\nBase Time差分からの検出:")
                print(f"  有効データ点数: {len(base_data)}")
                if len(base_data) > 0:
                    print(f"  高度範囲: {base_data['Height_Rsun'].min():.3f} - {base_data['Height_Rsun'].max():.3f} R⊙")
                
                print(f"\n二次差分からの検出:")
                print(f"  有効データ点数: {len(second_data)}")
                if len(second_data) > 0:
                    print(f"  高度範囲: {second_data['Height_Rsun'].min():.3f} - {second_data['Height_Rsun'].max():.3f} R⊙")
                
                # 各時刻での検出数の統計（差分タイプ別）
                print(f"\nBase Time差分での検出点数統計:")
                base_stats = df[df['Diff_Type'] == 'Base_Diff'].groupby('Time_ISO')['Point_Index'].max().value_counts().sort_index()
                for num_points, count in base_stats.items():
                    if num_points == 0:
                        print(f"  検出なし: {count} 時刻")
                    else:
                        print(f"  {num_points}点検出: {count} 時刻")
                
                print(f"\n二次差分での検出点数統計:")
                second_stats = df[df['Diff_Type'] == 'Second_Diff'].groupby('Time_ISO')['Point_Index'].max().value_counts().sort_index()
                for num_points, count in second_stats.items():
                    if num_points == 0:
                        print(f"  検出なし: {count} 時刻")
                    else:
                        print(f"  {num_points}点検出: {count} 時刻")
                
                # 高度変化をプロット（拡張版）
                print("\n拡張版高度変化をプロット中...")
                plot_cme_height_evolution_enhanced(
                    df=df,
                    output_plot=plot_output,
                    title="Enhanced CME Height Evolution with Second Difference (03:00-04:01)"
                )
                
                # 最初の10レコードを表示
                print(f"\n最初の10レコード:")
                print(df.head(10).to_string(index=False))
                
                # 最大高度のレコードを表示
                max_height_row = valid_data.loc[valid_data['Height_Rsun'].idxmax()]
                print(f"\n最大高度データ:")
                print(f"時刻: {max_height_row['Time_ISO']}")
                print(f"高度: {max_height_row['Height_Rsun']:.3f} R⊙")
                print(f"点番号: {max_height_row['Point_Index']}")
                print(f"総検出点数: {max_height_row['Total_Points']}")
                
                # 計測過程の動画を作成（拡張版）
                print("\n拡張版計測過程の動画を作成中...")
                create_cme_height_measurement_movie_enhanced(
                    map_sequence=filtered_sequence,
                    base_time_str=base_time_str,
                    output_movie=movie_output,
                    fps=8,
                    threshold_factor=2.0,
                    min_radius_rsun=1.5,
                    max_radius_rsun=6.0
                )
                
            else:
                print("有効なCME高度データが計測されませんでした。")
        else:
            print("計測データが得られませんでした。")
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    print("=== 拡張版テスト完了 ===")

if __name__ == "__main__":
    test_enhanced_cme_height_measurement() 