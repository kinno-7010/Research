#!/usr/bin/env python3
"""
STEREO-A/SECCHI/COR1 プロフェッショナル天体画像プロット

このスクリプトは、SSWIDL機能を統合したCOR1画像処理・プロット機能を提供します。
カラーテーブル、アノテーション、測定機能などの天文学的可視化ツールを含みます。

SSWIDL参照プログラム:
- secchi_colors.pro (カラーテーブル)
- scc_add_datetime.pro (日時スタンプ)
- scc_add_logo.pro (ロゴ配置)
- drawcoordgrid.pro (座標グリッド)
- stereo_rsun.pro (太陽半径計算)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
import glob
from datetime import datetime
from matplotlib.patches import Circle
import argparse
import sys

# 新機能モジュールのインポート
try:
    from cor1_colors import SECCHIColors
    from cor1_annotations import COR1Annotations
    from cor1_solar_utils import COR1SolarUtils
except ImportError as e:
    print(f"Warning: Could not import enhanced modules: {e}")
    print("Running in basic mode only.")
    SECCHIColors = None
    COR1Annotations = None
    COR1SolarUtils = None

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

def read_cor1_data(filepath):
    """
    COR1 FITSファイルを読み込み、データとヘッダーを返す
    
    Parameters:
    -----------
    filepath : str
        FITSファイルのパス
    
    Returns:
    --------
    data : numpy.ndarray
        画像データ
    header : astropy.io.fits.Header
        FITSヘッダー
    """
    try:
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            header = hdul[0].header
        return data, header
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

def calculate_solar_radius_pixel(header):
    """
    FITSヘッダーから太陽半径をピクセル単位で計算
    
    Parameters:
    -----------
    header : astropy.io.fits.Header
        FITSヘッダー
    
    Returns:
    --------
    rsun_pixel : float
        太陽半径（ピクセル単位）
    sun_center_x : float
        太陽中心のx座標（ピクセル単位）
    sun_center_y : float
        太陽中心のy座標（ピクセル単位）
    """
    # 太陽半径（arcsec）
    rsun_arcsec = header.get('RSUN', None)
    if rsun_arcsec is None:
        return None, None, None
    
    # ピクセルスケール（arcsec/pixel）
    cdelt1 = header.get('CDELT1', None)
    cdelt2 = header.get('CDELT2', None)
    
    if cdelt1 is None or cdelt2 is None:
        return None, None, None
    
    # 太陽中心のピクセル位置
    crpix1 = header.get('CRPIX1', None)
    crpix2 = header.get('CRPIX2', None)
    
    if crpix1 is None or crpix2 is None:
        return None, None, None
    
    # 太陽半径をピクセル単位に変換
    rsun_pixel = rsun_arcsec / abs(cdelt1)
    
    # 太陽中心座標（1-indexedから0-indexedに変換）
    sun_center_x = crpix1 - 1
    sun_center_y = crpix2 - 1
    
    return rsun_pixel, sun_center_x, sun_center_y

def plot_cor1_image(data, header, filepath, save_path=None, 
                   use_secchi_colors=True, add_annotations=True, 
                   add_coordinate_grid=False, add_measurements=False,
                   color_scaling='zscale', show_logo=True):
    """
    COR1データをプロフェッショナル天体画像としてプロット
    
    Parameters:
    -----------
    data : numpy.ndarray
        画像データ
    header : astropy.io.fits.Header
        FITSヘッダー
    filepath : str
        元のファイルパス
    save_path : str, optional
        保存先パス
    use_secchi_colors : bool, optional
        SECCHI専用カラーテーブルを使用
    add_annotations : bool, optional
        日時スタンプとロゴを追加
    add_coordinate_grid : bool, optional
        座標グリッドを追加
    add_measurements : bool, optional
        測定ツールを表示
    color_scaling : str, optional
        カラースケーリング方法（'zscale', 'percentile', 'minmax'）
    show_logo : bool, optional
        SECCHIロゴを表示
    """
    # 新機能モジュールの初期化
    if SECCHIColors is not None and use_secchi_colors:
        colors = SECCHIColors(silent=True)
    else:
        colors = None
    
    if COR1Annotations is not None and add_annotations:
        annotations = COR1Annotations(silent=True)
    else:
        annotations = None
    
    if COR1SolarUtils is not None:
        solar_utils = COR1SolarUtils(silent=True)
    else:
        solar_utils = None
    
    # データの基本情報を取得
    obs_time = header.get('DATE-OBS', header.get('DATE_OBS', 'Unknown'))
    instrument = header.get('INSTRUME', 'Unknown')
    detector = header.get('DETECTOR', 'Unknown')
    
    # 時間情報を解析
    try:
        time_obj = Time(obs_time)
        time_str = time_obj.datetime.strftime('%Y-%m-%d %H:%M:%S')
    except:
        time_str = obs_time
    
    # 太陽半径と中心座標の計算（新機能優先）
    if solar_utils is not None:
        sun_center = solar_utils.get_sun_center(header)
        sun_center_x, sun_center_y = sun_center['xcen'], sun_center['ycen']
        rsun_arcsec = solar_utils.calculate_solar_radius_arcsec(obs_time, 'A')
        cdelt1 = abs(header.get('CDELT1', 1.0))
        rsun_pixel = rsun_arcsec / cdelt1
    else:
        # フォールバック: 既存機能
        rsun_pixel, sun_center_x, sun_center_y = calculate_solar_radius_pixel(header)
        rsun_arcsec = header.get('RSUN', None)
    
    # データの統計情報
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data_mean = np.nanmean(data)
    data_std = np.nanstd(data)
    
    print(f"File: {os.path.basename(filepath)}")
    print(f"Observation Time: {time_str}")
    print(f"Instrument: {instrument}")
    print(f"Detector: {detector}")
    print(f"Data shape: {data.shape}")
    print(f"Data range: {data_min:.2f} to {data_max:.2f}")
    print(f"Data mean: {data_mean:.2f} ± {data_std:.2f}")
    if rsun_pixel is not None:
        print(f"Solar radius: {rsun_arcsec:.2f} arcsec ({rsun_pixel:.2f} pixels)")
        print(f"Solar center: ({sun_center_x:.2f}, {sun_center_y:.2f}) pixels")
    print("-" * 50)
    
    # カラースケーリングの計算
    if colors is not None:
        vmin, vmax = colors.calculate_scaling(data, method=color_scaling)
        cmap = colors.get_colormap('COR1')
    else:
        # フォールバック: 従来のスケーリング
        interval = ZScaleInterval()
        vmin, vmax = 2000, 6000
        cmap = 'gray'
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 画像表示（SECCHI専用カラーテーブル使用）
    im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    
    # カラーバー追加
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Intensity [DN]', rotation=270, labelpad=20)
    
    # プロフェッショナルなタイトル設定
    title_text = f'STEREO-A/SECCHI/COR1'
    if add_annotations and annotations is not None:
        # 日時スタンプはアノテーション機能で追加するため、タイトルは簡潔に
        ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
    else:
        ax.set_title(f'{title_text}\n{time_str}', fontsize=14, fontweight='bold')
    
    # 座標ラベル設定
    ax.set_xlabel('Pixel X', fontsize=12)
    ax.set_ylabel('Pixel Y', fontsize=12)
    
    # 座標グリッドの追加
    if add_coordinate_grid and annotations is not None:
        # 天体座標グリッドを追加
        annotations.draw_coordinate_grid(data, header, system='HCR', color='cyan', thickness=1)
    else:
        # 基本グリッド
        ax.grid(True, alpha=0.3)
    
    # 太陽中心に十字線を追加（改良版）
    if sun_center_x is not None and sun_center_y is not None:
        # 太陽中心に十字線
        ax.axhline(y=sun_center_y, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.axvline(x=sun_center_x, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    else:
        # フォールバック: 画像中心
        center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
        ax.axhline(y=center_y, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=center_x, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # 太陽半径の円を描画（SSWIDL準拠の改良版）
    if rsun_pixel is not None and sun_center_x is not None and sun_center_y is not None:
        # 太陽リム円の描画（IDL準拠）
        if annotations is not None:
            annotations.add_solar_limb_circle(data, header, thickness=2, color='yellow')
        
        # 太陽半径の円（1Rs, 2Rs, 3Rs）- 天体観測用標準表示
        colors_list = ['yellow', 'orange', 'red']
        alphas = [0.9, 0.7, 0.6]
        linewidths = [2.5, 2.0, 1.5]
        
        for i, (color, alpha, lw) in enumerate(zip(colors_list, alphas, linewidths)):
            radius = (i + 1) * rsun_pixel
            circle = Circle((sun_center_x, sun_center_y), radius, 
                           fill=False, edgecolor=color, linewidth=lw, 
                           alpha=alpha, label=f'{i+1} R☉')
            ax.add_patch(circle)
        
        # 太陽中心マーカー
        ax.plot(sun_center_x, sun_center_y, '+', color='yellow', 
                markersize=12, markeredgewidth=3, label='太陽中心')
        
        # プロフェッショナルな凡例
        legend = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                          fontsize=11, framealpha=0.9, edgecolor='gray')
        legend.get_frame().set_facecolor('black')
        for text in legend.get_texts():
            text.set_color('white')
    
    # アノテーション機能の適用
    if add_annotations and annotations is not None:
        # 日時スタンプの追加（SSWIDL scc_add_datetime準拠）
        try:
            # 画像上に直接テキストを描画する代わりに、matplotlibテキストを使用
            datetime_str, detector_info = annotations.format_datetime_string(header)
            
            # 動的フォントサイズの計算
            sum_factor = annotations._get_size_factor(data.shape)
            config = annotations.size_configs[sum_factor]
            
            # 日時スタンプの配置（左下）
            ax.text(0.02, 0.08, datetime_str, transform=ax.transAxes, 
                   fontsize=config['font_size']*0.8, color='white', 
                   weight=config['font_weight'], family='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            # 検出器情報の配置（日時スタンプの上）
            if detector_info:
                ax.text(0.02, 0.12, detector_info, transform=ax.transAxes, 
                       fontsize=config['font_size']*0.7, color='white', 
                       weight=config['font_weight'], family='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        except Exception as e:
            print(f"Warning: Could not add annotations: {e}")
        
        # ロゴの追加（右上）
        if show_logo:
            try:
                # SECCHIロゴテキスト（実際のロゴファイルがある場合は置き換え）
                ax.text(0.98, 0.98, 'SECCHI', transform=ax.transAxes, 
                       fontsize=14, color='white', weight='bold',
                       ha='right', va='top',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='navy', alpha=0.8))
            except Exception as e:
                print(f"Warning: Could not add logo: {e}")
    
    # 測定ツールの表示
    if add_measurements and solar_utils is not None:
        try:
            # サンプル測定点の表示
            test_points = [(sun_center_x + rsun_pixel, sun_center_y),
                          (sun_center_x, sun_center_y + 2*rsun_pixel)]
            
            # 距離測定の表示
            for i, point in enumerate(test_points):
                distance_rsun = solar_utils.calculate_distance(
                    (sun_center_x, sun_center_y), point, header, 'rsun')
                ax.plot(point[0], point[1], 'go', markersize=6)
                ax.text(point[0]+10, point[1]+10, f'{distance_rsun:.1f} R☉',
                       color='green', fontsize=10, weight='bold')
        except Exception as e:
            print(f"Warning: Could not add measurements: {e}")
    
    # プロフェッショナルなレイアウト調整
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # 高品質で保存またはプロット表示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        print(f"Professional plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig, ax

def main():
    """メイン関数 - コマンドライン引数対応"""
    parser = argparse.ArgumentParser(
        description='STEREO-A/SECCHI/COR1 プロフェッショナル画像プロット',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 cor1_plot.py                           # Basic mode
  python3 cor1_plot.py --enhanced                # Full enhanced mode
  python3 cor1_plot.py --colors --annotations    # Selective features
  python3 cor1_plot.py --file custom_file.fits   # Custom file
  python3 cor1_plot.py --scaling percentile      # Custom scaling
        """)
    
    parser.add_argument('--file', type=str, 
                       help='Specific FITS file to process')
    parser.add_argument('--enhanced', action='store_true',
                       help='Enable all enhanced features')
    parser.add_argument('--colors', action='store_true', 
                       help='Use SECCHI color tables')
    parser.add_argument('--annotations', action='store_true',
                       help='Add datetime stamps and logo')
    parser.add_argument('--grid', action='store_true',
                       help='Add coordinate grid')
    parser.add_argument('--measurements', action='store_true',
                       help='Show measurement tools')
    parser.add_argument('--scaling', choices=['zscale', 'percentile', 'minmax'],
                       default='zscale', help='Color scaling method')
    parser.add_argument('--no-logo', action='store_true',
                       help='Disable SECCHI logo')
    parser.add_argument('--output-dir', type=str,
                       default='/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Enhanced mode enables all features
    if args.enhanced:
        use_colors = True
        add_annotations = True
        add_grid = True
        add_measurements = True
        show_logo = not args.no_logo
    else:
        use_colors = args.colors
        add_annotations = args.annotations
        add_grid = args.grid
        add_measurements = args.measurements
        show_logo = args.annotations and not args.no_logo
    
    # 機能チェック
    if (use_colors or add_annotations or add_measurements) and SECCHIColors is None:
        print("Warning: Enhanced features requested but modules not available.")
        print("Running in basic mode. Please check module imports.")
        use_colors = False
        add_annotations = False
        add_measurements = False
    
    # ファイル処理
    if args.file:
        # 指定されたファイルを処理
        target_files = [args.file]
        data_dir = os.path.dirname(args.file) if os.path.dirname(args.file) else "."
    else:
        # デフォルトファイルを処理
        data_dir = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/calibration"
        target_files = ['20220613_032136_n4c1A_processed.fits']
    
    print(f"=== STEREO-A/SECCHI/COR1 Professional Plotter ===")
    print(f"Enhanced features: Colors={use_colors}, Annotations={add_annotations}")
    print(f"Grid={add_grid}, Measurements={add_measurements}, Logo={show_logo}")
    print(f"Scaling method: {args.scaling}")
    print("=" * 50)
    
    # 各ファイルを処理
    processed_count = 0
    for filename in target_files:
        if args.file:
            filepath = filename
        else:
            filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        
        print(f"\n処理中: {os.path.basename(filepath)}")
        
        # データ読み込み
        data, header = read_cor1_data(filepath)
        
        if data is None:
            continue
        
        # プロット作成（新機能統合版）
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        suffix = '_enhanced' if args.enhanced else '_professional'
        output_filename = f"{base_name}{suffix}.png"
        save_path = os.path.join(args.output_dir, output_filename)
        
        try:
            fig, ax = plot_cor1_image(
                data, header, filepath, save_path,
                use_secchi_colors=use_colors,
                add_annotations=add_annotations,
                add_coordinate_grid=add_grid,
                add_measurements=add_measurements,
                color_scaling=args.scaling,
                show_logo=show_logo
            )
            plt.close(fig)  # メモリ節約
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    print(f"\n=== 処理完了 ===")
    print(f"処理されたファイル数: {processed_count}/{len(target_files)}")
    if processed_count > 0:
        print(f"出力先: {args.output_dir}")

if __name__ == "__main__":
    main()