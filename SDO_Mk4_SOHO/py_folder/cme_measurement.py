"""
CME計測関数群
コロナ質量放出(CME)の高度測定、運動学的解析、可視化を行う
"""

from config import *
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
# GUIが利用可能な場合はTkAggを使用、そうでなければAggを使用
try:
    import tkinter
    matplotlib.use('TkAgg')
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os
from astropy.time import Time
import astropy.units as u
from tqdm import tqdm
import warnings
import threading
warnings.filterwarnings('ignore', category=RuntimeWarning)


def measure_cme_height_manual(ax, map_data, params, r_map, instrument='mk4', click_points=None):
    """
    CMEの高度を手動で計測する関数
    
    Parameters:
    -----------
    ax : matplotlib.axes
        プロット用のAxes
    map_data : np.ndarray
        画像データ
    params : dict
        パラメータ辞書 (nx, ny, cx, cy, px_per_rsun)
    r_map : np.ndarray
        半径マップ (太陽半径単位)
    instrument : str
        観測機器名 ('mk4' または 'lasco')
    click_points : list of tuples, optional
        手動で指定する点のリスト [(x1, y1), (x2, y2), ...]
        
    Returns:
    --------
    heights : list
        測定されたCME先端の高度 (太陽半径単位)
    positions : list of tuples
        測定点の座標 [(x, y), ...]
    """
    
    if click_points is None:
        print(f"{instrument}画像上でCMEの先端をクリックしてください。右クリックで終了。")
        click_points = []
        
        def onclick(event):
            if event.button == 1:  # 左クリック
                if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                    click_points.append((event.xdata, event.ydata))
                    # エラーハンドリングで囲んでプロット
                    try:
                        ax.plot(event.xdata, event.ydata, 'ro', markersize=8, markeredgecolor='white')
                        plt.draw()
                        print(f"測定点 {len(click_points)}: ({event.xdata:.1f}, {event.ydata:.1f})")
                    except Exception as e:
                        print(f"プロットエラー: {e}")
            elif event.button == 3:  # 右クリック
                try:
                    plt.disconnect(cid)
                except:
                    pass
                print(f"測定終了。{len(click_points)}点を取得しました。")
                
        cid = plt.connect('button_press_event', onclick)
        
        try:
            plt.show(block=False)  # ノンブロッキング表示に戻す
            print("\ncombined画像上でCMEの先端をクリックしてください。")
            print("左クリック: 測定点を追加")
            print("右クリック: 測定終了\n")
            
            # 簡単な待機ループを復活
            input("クリック操作後、Enterキーを押して継続してください...")
        except Exception as e:
            print(f"注意: GUIが利用できません: {e}")
            print("click_pointsパラメータを使用してください。")
    
    # クリック点から高度を計算
    heights = []
    positions = []
    
    for x_pixel, y_pixel in click_points:
        # ピクセル座標を太陽中心からの距離に変換
        x_from_center = x_pixel - params['cx']
        y_from_center = y_pixel - params['cy']
        
        # 太陽半径単位での距離を計算（ゼロ除算を回避）
        if params['px_per_rsun'] <= 0:
            print(f"警告: px_per_rsun が無効な値です: {params['px_per_rsun']}")
            params['px_per_rsun'] = 1  # デフォルト値を設定
        distance_rsun = np.sqrt(x_from_center**2 + y_from_center**2) / params['px_per_rsun']
        
        heights.append(distance_rsun)
        positions.append((x_pixel, y_pixel))
        
        # 測定点をプロット
        ax.scatter(x_pixel, y_pixel, c='red', s=100, marker='o', 
                  edgecolors='white', linewidth=2, zorder=10,
                  label=f'CME leading edge: {distance_rsun:.2f} R☉')
    
    return heights, positions


def fit_cme_kinematics(times, heights):
    """
    CMEの運動学的パラメータをフィッティング
    
    Parameters:
    -----------
    times : list
        時刻リスト (astropy.time.Time objects)
    heights : list
        対応する高度リスト (太陽半径単位)
        
    Returns:
    --------
    fit_params : dict
        フィッティング結果 {'v0': 初期速度, 'a': 加速度, 'h0': 初期高度}
    fit_func : function
        フィッティング関数
    """
    
    # 時刻を秒単位の相対時間に変換
    t0 = times[0]
    t_seconds = [(t - t0).to_value('s') for t in times]
    
    # 2次多項式フィッティング: h(t) = h0 + v0*t + 0.5*a*t^2
    def kinematic_model(t, h0, v0, a):
        return h0 + v0 * np.array(t) + 0.5 * a * np.array(t)**2
    
    try:
        popt, pcov = curve_fit(kinematic_model, t_seconds, heights)
        h0, v0, a = popt
        
        fit_params = {
            'h0': h0,          # 初期高度 [R☉]
            'v0': v0,          # 初期速度 [R☉/s]
            'a': a,            # 加速度 [R☉/s²]
            'v0_km_s': v0 * 695700,  # 初期速度 [km/s]
            'a_km_s2': a * 695700    # 加速度 [km/s²]
        }
        
        return fit_params, kinematic_model
        
    except Exception as e:
        print(f"フィッティングエラー: {e}")
        return None, None


def plot_cme_height_evolution(times, heights, fit_params=None, fit_func=None):
    """
    CME高度の時間発展をプロット
    
    Parameters:
    -----------
    times : list
        時刻リスト
    heights : list
        高度リスト
    fit_params : dict, optional
        フィッティングパラメータ
    fit_func : function, optional
        フィッティング関数
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 時刻を分単位に変換
    t0 = times[0]
    t_minutes = [(t - t0).to_value('min') for t in times]
    
    # 高度-時間プロット（負値やゼロ値をチェック）
    valid_heights = [h for h in heights if h > 0]
    valid_t_minutes = [t_minutes[i] for i, h in enumerate(heights) if h > 0]
    
    if len(valid_heights) > 0:
        ax1.scatter(valid_t_minutes, valid_heights, c='red', s=100, marker='o', 
                   edgecolors='black', linewidth=1, label='CME nose height')
    else:
        print("警告: 有効な高度データがありません")
    
    if fit_params and fit_func and len(valid_t_minutes) > 0:
        t_fine = np.linspace(0, max(valid_t_minutes), 100)
        t_fine_sec = np.array(t_fine) * 60  # 分を秒に変換
        h_fit = fit_func(t_fine_sec, fit_params['h0'], fit_params['v0'], fit_params['a'])
        # フィット結果も正値のみをプロット
        valid_h_fit = np.maximum(h_fit, 1e-6)  # 最小値を設定
        ax1.plot(t_fine, valid_h_fit, 'b-', linewidth=2, 
                label=f"Fit: v₀={fit_params['v0_km_s']:.0f} km/s, a={fit_params['a_km_s2']:.1f} km/s²")
    
    ax1.set_xlabel('Time (min) since ' + times[0].iso)
    ax1.set_ylabel('CME nose height ($R_\odot$)')
    ax1.set_title('CME height evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # Y軸の範囲を正値のみに制限
    if len(valid_heights) > 0:
        ax1.set_ylim(bottom=0.1)  # 最小値を設定してログスケール対応
    
    # 速度-時間プロット
    if len(times) > 1:
        velocities = []
        t_vel = []
        for i in range(1, len(times)):
            dt = (times[i] - times[i-1]).to_value('s')
            dh = heights[i] - heights[i-1]
            v = dh / dt * 695700  # km/s に変換
            velocities.append(v)
            t_vel.append((t_minutes[i] + t_minutes[i-1]) / 2)
        
        ax2.scatter(t_vel, velocities, c='blue', s=100, marker='s',
                   edgecolors='black', linewidth=1, label='CME velocity (km/s)')
        
        if fit_params:
            v_fit = fit_params['v0_km_s'] + fit_params['a_km_s2'] * np.array(t_vel) * 60
            ax2.plot(t_vel, v_fit, 'r-', linewidth=2, label='Fit velocity (km/s)')
        
        ax2.set_xlabel('Time (min) since ' + times[0].iso)
        ax2.set_ylabel('CME velocity (km/s)')
        ax2.set_title('CME velocity evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    return fig


def analyze_cme_height_time_series(start_time_str: str, end_time_str: str, 
                                  time_interval_min: int = 1):
    """
    03:00-04:01の時間範囲でCME高度を時系列解析
    
    Parameters:
    -----------
    start_time_str : str
        開始時刻 'YYYY-MM-DDTHH:MM:SS'
    end_time_str : str
        終了時刻 'YYYY-MM-DDTHH:MM:SS'
    time_interval_min : int
        時間間隔 (分)
        
    Returns:
    --------
    results : dict
        解析結果を含む辞書
    """
    
    print("CME高度の時系列解析を開始します...")
    
    # 時刻リストを作成
    start_time = Time(start_time_str)
    end_time = Time(end_time_str)
    current_time = start_time
    time_list = []
    
    while current_time <= end_time:
        time_list.append(current_time)
        current_time += time_interval_min * u.min
    
    heights_mk4 = []
    heights_lasco = []
    times_measured = []
    
    print(f"{len(time_list)}個の時刻でCME高度を測定します...")
    
    for i, target_time in enumerate(tqdm(time_list)):
        try:
            # 統合画像を作成
            fig, ax = plt.subplots(figsize=(12, 12))
            from integrated_analysis import create_single_integrated_image
            create_single_integrated_image(ax, target_time.iso)
            
            # この時点で手動またはスクリプトでCME先端を指定
            # 実際の使用では、ユーザーが画像を見て座標を指定する
            
            # サンプル座標（実際の解析では適切な座標に置き換える）
            if target_time.iso == '2022-06-13T03:00:00.000':
                # MK4領域のCME先端例
                mk4_points = [(-150, 50)]  # サンプル座標
                lasco_points = []
            elif target_time.iso >= '2022-06-13T03:24:00.000':
                # LASCO領域にCMEが到達
                mk4_points = []
                lasco_points = [(-250, 100)]  # サンプル座標
            else:
                mk4_points = [(-180, 80)]
                lasco_points = []
            
            # 実際のパラメータを取得（簡略化）
            params = {'cx': 256, 'cy': 256, 'px_per_rsun': 80}  # 実際の値に置き換え
            r_map = np.sqrt(np.indices((512, 512))[0]**2 + np.indices((512, 512))[1]**2)
            
            # CME高度を測定
            if mk4_points:
                h_mk4, _ = measure_cme_height_manual(ax, None, params, r_map, 
                                                   'mk4', mk4_points)
                heights_mk4.extend(h_mk4)
                times_measured.append(target_time)
            
            if lasco_points:
                h_lasco, _ = measure_cme_height_manual(ax, None, params, r_map,
                                                     'lasco', lasco_points)
                heights_lasco.extend(h_lasco)
                if not mk4_points:  # LASCOのみの場合
                    times_measured.append(target_time)
            
            plt.close(fig)
            
        except Exception as e:
            print(f"時刻 {target_time.iso} でエラー: {e}")
            continue
    
    # 結果をまとめる
    all_heights = heights_mk4 + heights_lasco
    
    if len(all_heights) > 2 and len(times_measured) > 2:
        # 運動学的フィッティング
        fit_params, fit_func = fit_cme_kinematics(times_measured, all_heights)
        
        # 結果をプロット
        plot_fig = plot_cme_height_evolution(times_measured, all_heights, 
                                           fit_params, fit_func)
        
        results = {
            'times': times_measured,
            'heights': all_heights,
            'heights_mk4': heights_mk4,
            'heights_lasco': heights_lasco,
            'fit_params': fit_params,
            'plot_figure': plot_fig
        }
        
        return results
    
    else:
        print("データが不足しています。より多くの測定点が必要です。")
        return None


def plot_integrated_image_with_cme_overlay(target_time_str: str, 
                                         cme_positions: list = None,
                                         show_height_circles: bool = True):
    """
    統合画像にCME先端位置をオーバーレイしてプロット
    
    Parameters:
    -----------
    target_time_str : str
        対象時刻 'YYYY-MM-DDTHH:MM:SS'
    cme_positions : list of tuples
        CME先端位置のリスト [(x1, y1), (x2, y2), ...]
    show_height_circles : bool
        高度円を表示するかどうか
    """
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 統合画像を作成
    from integrated_analysis import create_single_integrated_image
    create_single_integrated_image(ax, target_time_str)
    
    if cme_positions:
        # CME先端位置をスキャッタープロット
        x_coords = [pos[0] for pos in cme_positions]
        y_coords = [pos[1] for pos in cme_positions]
        
        ax.scatter(x_coords, y_coords, c='red', s=150, marker='*',
                  edgecolors='white', linewidth=2, zorder=15,
                  label='CME Leading Edge')
        
        # 各点に高度ラベルを追加
        params = {'cx': 256, 'cy': 256, 'px_per_rsun': 80}  # 実際の値に置き換え
        for i, (x, y) in enumerate(cme_positions):
            distance_rsun = np.sqrt((x - params['cx'])**2 + (y - params['cy'])**2) / params['px_per_rsun']
            ax.annotate(f'{distance_rsun:.2f} R☉', 
                       (x, y), xytext=(10, 10), 
                       textcoords='offset points',
                       color='white', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
    
    ax.legend(loc='upper right')
    plt.title(f'CME Height Measurement - {target_time_str}', fontsize=14, pad=20)
    plt.tight_layout()
    
    return fig, ax


def measure_cme_height_manual_multi_points(ax, map_data, params, r_map, instrument='mk4', click_points=None):
    """
    CMEの高度を複数点で手動計測する関数
    
    Parameters:
    -----------
    ax : matplotlib.axes
        プロット用のAxes
    map_data : np.ndarray
        画像データ
    params : dict
        パラメータ辞書 (nx, ny, cx, cy, px_per_rsun)
    r_map : np.ndarray
        半径マップ (太陽半径単位)
    instrument : str
        観測機器名 ('mk4' または 'lasco')
    click_points : list of tuples, optional
        手動で指定する点のリスト [(x1, y1), (x2, y2), ...]
        
    Returns:
    --------
    heights : list
        測定されたCME先端の高度 (太陽半径単位)
    positions : list of tuples
        測定点の座標 [(x, y), ...]
    angles : list
        各測定点の位置角度 (度)
    """
    
    if click_points is None:
        print(f"\n{instrument}画像上でCMEの複数点をクリックしてください。")
        print("左クリック: 測定点を追加")
        print("右クリック: 測定終了\n")
        click_points = []
        finished = False
        
        def onclick(event):
            nonlocal finished
            if event.button == 1:  # 左クリック
                if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                    click_points.append((event.xdata, event.ydata))
                    # エラーハンドリングで囲んでプロット
                    try:
                        # 点をプロット
                        ax.plot(event.xdata, event.ydata, 'ro', markersize=8, 
                               markeredgecolor='white', markeredgewidth=2)
                        # 点番号を表示
                        ax.text(event.xdata + 5, event.ydata + 5, str(len(click_points)), 
                               color='yellow', fontsize=10, fontweight='bold')
                        plt.draw()
                        print(f"測定点 {len(click_points)}: ({event.xdata:.1f}, {event.ydata:.1f})")
                    except Exception as e:
                        print(f"プロットエラー: {e}")
            elif event.button == 3:  # 右クリック
                finished = True
                try:
                    plt.disconnect(cid)
                except Exception as e:
                    print(f"イベントハンドラーの切断エラー: {e}")
                print(f"測定終了。{len(click_points)}点を取得しました。")
                # ウィンドウを適切に閉じる
                try:
                    plt.close()
                except:
                    pass
                
        cid = plt.connect('button_press_event', onclick)
        
        # インタラクティブモードでブロック
        try:
            # メインスレッドチェックを一時的に無効化
            print(f"DEBUG: スレッド情報 - 現在: {threading.current_thread().name}, メイン: {threading.main_thread().name}")
            print("DEBUG: メインスレッドチェックをスキップしてGUIを表示します")
            
            plt.show(block=False)  # ノンブロッキング表示に戻す
            print("\ncombined画像上でCMEの複数点をクリックしてください。")
            print("左クリック: 測定点を追加")
            print("右クリック: 測定終了\n")
            print(f"DEBUG: finishedフラグの初期値: {finished}")
            
            # 右クリックで終了するまで待機ループを復活
            print("DEBUG: クリック待機ループを開始")
            while not finished:
                plt.pause(0.1)
                if len(click_points) > 0:
                    print(f"DEBUG: 途中経過 - click_points: {len(click_points)}")
            print("DEBUG: クリック待機ループ終了")
            
        except Exception as e:
            print(f"注意: GUIが利用できません: {e}")
            print("click_pointsパラメータを使用してください。")
            finished = True
    
    # クリック点から高度と位置角を計算
    heights = []
    positions = []
    angles = []
    
    for i, (x_pixel, y_pixel) in enumerate(click_points):
        # 統合画像では extent により座標が太陽中心基準になっている
        # そのため、クリック座標はすでに太陽中心からの距離を表している
        x_from_center = x_pixel  # すでに太陽中心基準
        y_from_center = y_pixel  # すでに太陽中心基準
        
        # 太陽半径単位での距離を計算（ゼロ除算を回避）
        if params['px_per_rsun'] <= 0:
            print(f"警告: px_per_rsun が無効な値です: {params['px_per_rsun']}")
            params['px_per_rsun'] = 1  # デフォルト値を設定
        distance_rsun = np.sqrt(x_from_center**2 + y_from_center**2) / params['px_per_rsun']
        
        # 位置角度を計算（北から時計回りに0-360度）
        angle = np.degrees(np.arctan2(x_from_center, -y_from_center))
        if angle < 0:
            angle += 360
        
        heights.append(distance_rsun)
        positions.append((x_pixel, y_pixel))
        angles.append(angle)
        
        print(f"測定点 {i+1}: 高度 = {distance_rsun:.2f} R☉, 位置角 = {angle:.1f}°")
        print(f"  デバッグ: x_center={x_from_center:.1f}, y_center={y_from_center:.1f}, px_per_rsun={params['px_per_rsun']:.1f}")
    
    return heights, positions, angles


def plot_cme_height_distribution(heights, angles, time_str, instrument='mk4'):
    """
    CME高度の角度分布をプロット
    
    Parameters:
    -----------
    heights : list
        高度リスト (太陽半径単位)
    angles : list
        位置角度リスト (度)
    time_str : str
        観測時刻
    instrument : str
        観測機器名
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 高度 vs 位置角度のプロット（正値のみ）
    valid_data = [(a, h) for a, h in zip(angles, heights) if h > 0]
    if len(valid_data) > 0:
        valid_angles, valid_heights = zip(*valid_data)
        ax1.scatter(valid_angles, valid_heights, c='red', s=100, marker='o', 
                   edgecolors='black', linewidth=1)
    ax1.set_xlabel('Position Angle (degrees)')
    ax1.set_ylabel('Height (R☉)')
    ax1.set_title(f'CME Height Distribution - {time_str}')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 360)
    ax1.set_ylim(bottom=0.1)  # Y軸の最小値を設定
    
    # 平均高度と標準偏差を表示（正値のみ）
    positive_heights = [h for h in heights if h > 0]
    if len(positive_heights) > 0:
        mean_height = np.mean(positive_heights)
        std_height = np.std(positive_heights)
        ax1.axhline(y=mean_height, color='blue', linestyle='--', 
                   label=f'Mean: {mean_height:.2f} ± {std_height:.2f} R☉')
        ax1.fill_between([0, 360], max(0.1, mean_height - std_height), mean_height + std_height,
                        color='blue', alpha=0.2)
    ax1.legend()
    
    # 2. 極座標プロット（正値のみ）
    ax2 = plt.subplot(122, projection='polar')
    positive_data = [(a, h) for a, h in zip(angles, heights) if h > 0]
    if len(positive_data) > 0:
        pos_angles, pos_heights = zip(*positive_data)
        theta = np.radians(pos_angles)
        ax2.scatter(theta, pos_heights, c='red', s=100, marker='o', 
                   edgecolors='black', linewidth=1)
    else:
        pos_heights = [1]  # デフォルト値を設定
    
    # 測定点を線で結ぶ
    if len(angles) > 2:
        # 角度順にソート
        sorted_indices = np.argsort(angles)
        sorted_angles = [angles[i] for i in sorted_indices]
        sorted_heights = [heights[i] for i in sorted_indices]
        sorted_theta = np.radians(sorted_angles)
        
        # 閉じた曲線にする
        sorted_theta = np.append(sorted_theta, sorted_theta[0])
        sorted_heights = np.append(sorted_heights, sorted_heights[0])
        
        ax2.plot(sorted_theta, sorted_heights, 'b-', linewidth=2, alpha=0.5)
    
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_title(f'CME Shape ({instrument.upper()})', pad=20)
    ax2.set_ylim(0, max(heights) * 1.2)
    ax2.grid(True)
    
    plt.tight_layout()
    return fig


def analyze_single_time_cme_multi_points(target_time_str: str, 
                                       save_results: bool = True,
                                       output_dir: str = './cme_analysis/'):
    """
    特定時刻のCME高度を複数点で計測し解析
    
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
    
    print(f"\n=== CME高度の複数点計測 ===")
    print(f"対象時刻: {target_time_str}")
    
    # 統合画像を作成（実際の実装では適切な関数を使用）
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # ダミー画像（実際の実装では create_single_integrated_image を使用）
    dummy_image = np.random.rand(512, 512)
    ax.imshow(dummy_image, cmap='gray', extent=[-256, 256, -256, 256])
    
    # 太陽円盤と高度円を描画
    params = {'cx': 0, 'cy': 0, 'px_per_rsun': 80}
    
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
    
    # CME高度を複数点で計測
    r_map = np.sqrt(np.indices((512, 512))[0]**2 + np.indices((512, 512))[1]**2)
    heights, positions, angles = measure_cme_height_manual_multi_points(
        ax, dummy_image, params, r_map, 'combined'
    )
    
    if len(heights) > 0:
        # 結果をプロット
        fig2 = plot_cme_height_distribution(heights, angles, target_time_str)
        
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
            csv_filename = os.path.join(output_dir, f'cme_heights_{time_label}.csv')
            df.to_csv(csv_filename, index=False)
            print(f"\nデータを保存: {csv_filename}")
            
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
    複数時刻のCME高度測定結果を比較
    
    Parameters:
    -----------
    time_list : list
        時刻文字列のリスト
    save_comparison : bool
        比較結果を保存するかどうか
    output_dir : str
        出力ディレクトリ
    """
    
    all_results = []
    
    for time_str in time_list:
        print(f"\n{'='*50}")
        result = analyze_single_time_cme_multi_points(time_str, save_results=True, 
                                                    output_dir=output_dir)
        if result:
            all_results.append(result)
    
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
        ax2.set_ylabel('CME Height (R☉)')
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