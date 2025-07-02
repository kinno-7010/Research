"""
Claude用解析ユーティリティ関数群
claude_analysis.pyで使用される独自の解析機能を提供
"""

from config import *
import config
from integrated_analysis import create_single_diff_image, clear_scan_cache, get_cache_info
from cme_measurement import (
    measure_cme_height_manual_multi_points, 
    plot_cme_height_distribution
)
import matplotlib
import matplotlib.pyplot as plt
import gc
import threading


def analyze_single_time_cme_with_diff_image(target_time_str: str, 
                                           save_results: bool = True,
                                           output_dir: str = './cme_analysis/'):
    """
    特定時刻の統合差分画像を背景にCME高度を複数点で計測し解析
    
    Parameters:
    -----------
    target_time_str : str
        対象時刻 'YYYY-MM-DDTHH:MM:SS'
    save_results : bool
        結果を保存するかどうか
    output_dir : str
        出力ディレクトリ
        
    Returns:
    --------
    results : dict
        解析結果
    """
    
    print(f"\n=== CME高度の複数点計測（統合差分画像使用） ===")
    print(f"対象時刻: {target_time_str}")
    
    # GUIリソースの初期化とクリーンアップ
    try:
        # 既存の全てのfigureを閉じる
        plt.close('all')
        # インタラクティブモードをリセット
        plt.ioff()
        plt.ion()  # 新しいセッション用に再度有効化
    except Exception as e:
        print(f"GUI初期化警告: {e}")
    
    # 統合差分画像を作成し、実際のパラメータを取得
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # create_single_diff_image関数を使用して実際の太陽観測データを表示
    try:
        from integrated_analysis import select_by_midpoint, get_params
        
        # 実際のデータを取得してパラメータを計算
        target_time_obj = Time(target_time_str)
        
        # 統合差分画像を作成（1回のスキャンでパラメータも取得）
        image_info = create_single_diff_image(ax, target_time_str)
        
        # 戻り値からパラメータを取得（重複スキャン回避）
        params_lasco = image_info['params_lasco']
        params_mk4 = image_info['params_mk4']
        lasco_map = image_info['lasco_map']
        mk4_map = image_info['mk4_map']
        
        print(f"LASCOパラメータ: cx={params_lasco['cx']:.1f}, cy={params_lasco['cy']:.1f}, px_per_rsun={params_lasco['px_per_rsun']:.1f}")
        print(f"MK4パラメータ: cx={params_mk4['cx']:.1f}, cy={params_mk4['cy']:.1f}, px_per_rsun={params_mk4['px_per_rsun']:.1f}")
        
        # 統合画像の基準となるパラメータを使用
        params = params_lasco
        mk4_time = mk4_map.date.strftime('%Y%m%d_%H%M%S')
        
        print("統合差分画像の作成が完了しました。")
        
    except Exception as e:
        print(f"統合差分画像作成でエラー: {e}")
        # フォールバック：ダミー画像を使用
        dummy_image = np.random.rand(512, 512)
        ax.imshow(dummy_image, cmap='gray', extent=[-256, 256, -256, 256])
        
        # 太陽円盤と高度円を描画
        params = {'cx': 0, 'cy': 0, 'px_per_rsun': 80}
        mk4_time = target_time_str.replace(':', '').replace('-', '')
        
        # 太陽円盤
        sun_circle = Circle((params['cx'], params['cy']), params['px_per_rsun'], 
                           fill=False, color='yellow', linewidth=2)
        ax.add_patch(sun_circle)
        
        # 高度円
        for r in [2, 3, 4, 5, 6]:
            height_circle = Circle((params['cx'], params['cy']), 
                                 r * params['px_per_rsun'], 
                                 fill=False, color='white', linewidth=1, 
                                 linestyle='--', alpha=0.5)
            ax.add_patch(height_circle)
            ax.text(params['cx'], params['cy'] + r * params['px_per_rsun'], 
                   f'{r} R☉', color='white', ha='center', va='bottom')
        
        ax.set_xlim(-256, 256)
        ax.set_ylim(-256, 256)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title(f'CME Multi-point Measurement - {target_time_str}')
    
    # CME高度を複数点で計測（r_mapは実際には使用されない）
    r_map = None  # measure_cme_height_manual_multi_pointsでは使用されない
    heights, positions, angles = measure_cme_height_manual_multi_points(
        ax, None, params, r_map, 'combined'
    )
    
    # クリック後の画像を保存
    if len(heights) > 0:
        # 保存ディレクトリを作成
        save_dir = Path('/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/CME_measurement/integrated_coronagraph')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル名を生成
        save_filename = f'integrated_coronagraph_{mk4_time}.png'
        save_path = save_dir / save_filename
        
        # 画像を保存
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"クリック後の画像を保存しました: {save_path}")
    
    # クリック操作後にGUIリソースを適切にクリーンアップ
    try:
        plt.close(fig)
        # matplotlibのGUIリソースをクリーンアップ
        import matplotlib
        if matplotlib.get_backend() != 'Agg':
            matplotlib.pyplot.ioff()  # インタラクティブモードをオフ
            matplotlib.pyplot.close('all')  # 全てのfigureを閉じる
    except Exception as e:
        print(f"GUI クリーンアップ警告: {e}")
    
    if len(heights) > 0:
        # 結果をプロット
        fig2 = plot_cme_height_distribution(heights, angles, target_time_str)
        plt.close(fig2)  # 結果プロットは保存後に閉じる
        
        # 統計情報を計算
        stats = {
            'n_points': len(heights),
            'mean_height': np.mean(heights),
            'std_height': np.std(heights),
            'min_height': np.min(heights),
            'max_height': np.max(heights),
            'height_range': np.max(heights) - np.min(heights)
        }
        
        print(f"\n=== 測定結果の統計 ===")
        print(f"測定点数: {stats['n_points']}")
        print(f"平均高度: {stats['mean_height']:.2f} ± {stats['std_height']:.2f} R☉")
        print(f"最小高度: {stats['min_height']:.2f} R☉")
        print(f"最大高度: {stats['max_height']:.2f} R☉")
        print(f"高度範囲: {stats['height_range']:.2f} R☉")
        
        # 結果を保存
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            
            # データをCSVで保存
            df = pd.DataFrame({
                'point_id': range(1, len(heights) + 1),
                'x_pixel': [p[0] for p in positions],
                'y_pixel': [p[1] for p in positions],
                'height_rsun': heights,
                'position_angle_deg': angles
            })
            
            time_label = target_time_str.replace(':', '').replace('-', '')
            
            # プロットを保存
            plot_filename = os.path.join(output_dir, f'cme_analysis_{time_label}.png')
            fig2.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"プロットを保存: {plot_filename}")
        
        results = {
            'time': target_time_str,
            'heights': heights,
            'positions': positions,
            'angles': angles,
            'statistics': stats,
            'dataframe': df
        }
        
        return results
    
    else:
        print("測定点が選択されませんでした。")
        return None


def compare_cme_heights_multiple_times(time_list: list, 
                                     save_comparison: bool = True,
                                     output_dir: str = './cme_analysis/'):
    """
    複数時刻のCME高度測定結果を比較（最適化版）
    
    Parameters:
    -----------
    time_list : list
        時刻文字列のリスト
    save_comparison : bool
        比較結果を保存するかどうか
    output_dir : str
        出力ディレクトリ
    """
    
    print(f"\n=== 複数時刻解析開始 ({len(time_list)}時刻) ===")
    print(get_cache_info())
    
    all_results = []
    
    # 各時刻を単一時刻解析として処理（キャッシュクリア付き）
    for i, time_str in enumerate(time_list, 1):
        print(f"\n{'='*50}")
        print(f"解析進行状況: {i}/{len(time_list)} ({time_str})")
        
        # 各時刻でキャッシュとGUIリソースをクリア（メモリ効率化）
        if i > 1:
            from integrated_analysis import clear_scan_cache
            clear_scan_cache()
            print("前回のキャッシュをクリアしました")
            
            # GUIリソースの完全クリーンアップ
            try:
                import matplotlib
                matplotlib.pyplot.close('all')  # 全てのfigureを閉じる
                matplotlib.pyplot.ioff()  # インタラクティブモードをオフ
                
                # tkinterリソースのクリーンアップ
                import gc
                gc.collect()  # ガベージコレクションを強制実行
                
                print("GUIリソースをクリーンアップしました")
            except Exception as e:
                print(f"GUIクリーンアップ警告: {e}")
        
        # 単一時刻解析と同じ処理を実行
        print(f"単一時刻解析を開始します（時刻 {i}/{len(time_list)}）...")
        try:
            result = analyze_single_time_cme_with_diff_image(time_str, save_results=True, 
                                                            output_dir=output_dir)
            if result:
                all_results.append(result)
            else:
                print(f"時刻 {time_str} の解析で結果が得られませんでした")
        except Exception as e:
            print(f"時刻 {time_str} の解析中にエラー: {e}")
            import traceback
            traceback.print_exc()
            
            # エラー後のGUIリソースクリーンアップ
            try:
                plt.close('all')
                gc.collect()
            except:
                pass
            
            print("次の時刻の解析を継続します...")
            continue
        
        # 進行状況を表示
        print(f"完了: {len(all_results)}/{i}, キャッシュ状況: {get_cache_info()}")
    
    if len(all_results) > 1:
        # 時系列比較プロット
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        times = [r['time'] for r in all_results]
        mean_heights = [r['statistics']['mean_height'] for r in all_results]
        std_heights = [r['statistics']['std_height'] for r in all_results]
        max_heights = [r['statistics']['max_height'] for r in all_results]
        min_heights = [r['statistics']['min_height'] for r in all_results]
        
        # 平均高度の時間変化
        ax1.errorbar(range(len(times)), mean_heights, yerr=std_heights, 
                    fmt='o-', capsize=5, capthick=2, markersize=8)
        ax1.set_xticks(range(len(times)))
        ax1.set_xticklabels([t.split('T')[1][:5] for t in times], rotation=45)
        ax1.set_ylabel('Mean CME Height (R☉)')
        ax1.set_title('CME Height Evolution')
        ax1.grid(True, alpha=0.3)
        
        # 高度範囲の時間変化
        ax2.fill_between(range(len(times)), min_heights, max_heights, 
                        alpha=0.3, color='blue', label='Height range')
        ax2.plot(range(len(times)), mean_heights, 'ro-', markersize=8, 
                label='Mean height')
        ax2.set_xticks(range(len(times)))
        ax2.set_xticklabels([t.split('T')[1][:5] for t in times], rotation=45)
        ax2.set_ylabel('CME Height ($R_\odot$)')
        ax2.set_xlabel('Time (UT)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_comparison:
            comparison_filename = os.path.join(output_dir, 'cme_height_comparison.png')
            fig.savefig(comparison_filename, dpi=300, bbox_inches='tight')
            print(f"\n比較プロットを保存: {comparison_filename}")
        
        plt.show()
    
    return all_results


def run_cme_analysis_workflow(time_series: list = None):
    """
    CME解析ワークフローを実行
    
    Parameters:
    -----------
    time_series : list, optional
        解析対象の時刻リスト。指定しない場合はデフォルトの時系列を使用
    """
    
    if time_series is None:
        time_series = [
            '2022-06-13T03:00:00',
            '2022-06-13T03:12:00',
            '2022-06-13T03:24:00',
            '2022-06-13T03:36:00',
            '2022-06-13T03:48:00',
            '2022-06-13T04:00:00'
        ]
    
    print("=== CME解析ワークフローを開始します ===")
    print(f"対象時刻数: {len(time_series)}")
    
    try:
        all_results = compare_cme_heights_multiple_times(
            time_series, 
            save_comparison=True,
            output_dir='./cme_analysis/'
        )
        
        print(f"\n=== ワークフロー完了 ===")
        print(f"解析成功数: {len(all_results)}")
        
        return all_results
        
    except Exception as e:
        print(f"ワークフロー実行中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return []