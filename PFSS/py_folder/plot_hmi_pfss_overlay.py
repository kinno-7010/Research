#!/usr/bin/env python3
"""
HMIデータの指定範囲にPFSS磁力線を重ねてプロット
plot_aia_overplotting.pyの手法をHMIデータに適用
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 必要なパッケージのインポート
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.map
import pfsspy
import pfsspy.tracing as tracing

# HMI解析モジュールのインポート
sys.path.append('/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/py_folder')
from hmi_analysis_wcs import read_hmi_quick, draw_hmi_solar_grid


def prepare_hmi_for_pfss(hmi_file):
    """
    HMIデータを読み込み、PFSS計算用に準備
    
    Parameters:
    -----------
    hmi_file : str
        HMIファイルのパス
        
    Returns:
    --------
    dict : HMIデータと範囲情報
    """
    print(f"HMIデータを読み込み中: {hmi_file}")
    
    # HMIデータ読み込み
    hmi_data = read_hmi_quick(hmi_file)
    data = hmi_data['data']
    hmi_map = hmi_data.get('sunpy_map')
    
    # plot_hmi_single.pyと同じ範囲設定
    ny, nx = data.shape
    center_x, center_y = nx // 2, ny // 2
    x_min_pix, x_max_pix = center_x - 512, center_x + 0
    y_min_pix, y_max_pix = center_y - 100, center_y + 512
    
    x_lims_pix = (x_min_pix, x_max_pix)
    y_lims_pix = (y_min_pix, y_max_pix)
    
    masked_data = data[y_min_pix:y_max_pix, x_min_pix:x_max_pix]
    
    # データの統計
    print(f"  データ形状: {data.shape}")
    print(f"  表示範囲: x[{x_min_pix}:{x_max_pix}], y[{y_min_pix}:{y_max_pix}]")
    print(f"  masked_data形状: {masked_data.shape}")
    print(f"  全体の磁場範囲: {np.nanmin(data):.1f} ～ {np.nanmax(data):.1f} Gauss")
    print(f"  表示範囲の磁場範囲: {np.nanmin(masked_data):.1f} ～ {np.nanmax(masked_data):.1f} Gauss")
    
    return {
        'full_map': hmi_map,
        'masked_data': masked_data,
        'x_lims_pix': x_lims_pix,
        'y_lims_pix': y_lims_pix,
        'time': hmi_data['time']
    }


def compute_pfss_solution(hmi_map, nrho=25, rss=2.5):
    """
    HMIマップからPFSS解を計算
    
    Parameters:
    -----------
    hmi_map : sunpy.map.Map
        HMIマップ（全球データ）
    nrho : int
        動径方向の格子点数
    rss : float
        ソース面の高度（太陽半径単位）
        
    Returns:
    --------
    pfsspy.Output : PFSS解
    """
    print(f"\nPFSS解を計算中...")
    print(f"  ソース面: {rss} Rs")
    print(f"  動径格子点数: {nrho}")
    
    # NaN値の処理
    data = hmi_map.data.copy()
    nan_mask = np.isnan(data)
    n_nan = np.sum(nan_mask)
    
    if n_nan > 0:
        print(f"  警告: {n_nan} 個のNaN値を検出しました")
        print(f"  NaN値を0に置き換えます")
        data[nan_mask] = 0.0
        
        # 新しいマップを作成
        hmi_map_clean = sunpy.map.Map(data, hmi_map.meta)
    else:
        hmi_map_clean = hmi_map
        print("  NaN値は検出されませんでした")
    
    # メタデータの修正（必要に応じて）
    if 'cunit1' not in hmi_map_clean.meta:
        hmi_map_clean.meta['cunit1'] = 'deg'
    if 'cunit2' not in hmi_map_clean.meta:
        hmi_map_clean.meta['cunit2'] = 'deg'
    
    # PFSS入力オブジェクトを作成
    pfss_input = pfsspy.Input(hmi_map_clean, nrho, rss)
    
    # PFSS解を計算
    pfss_output = pfsspy.pfss(pfss_input)
    
    print("  PFSS計算完了")
    
    return pfss_output


def define_field_line_seeds(hmi_map, x_lims_pix, y_lims_pix, n_seeds_x=7, n_seeds_y=7, 
                           use_strong_field=False, field_threshold=100):
    """
    表示範囲内に磁力線の開始点を定義
    
    Parameters:
    -----------
    hmi_map : sunpy.map.Map
        HMIマップ
    x_lims_pix, y_lims_pix : tuple
        表示範囲のピクセル座標
    n_seeds_x, n_seeds_y : int
        X, Y方向の開始点数（use_strong_field=Falseの場合）
    use_strong_field : bool
        強い磁場領域を優先するか
    field_threshold : float
        磁場強度の閾値（Gauss）
        
    Returns:
    --------
    SkyCoord : 磁力線の開始点座標
    """
    print(f"\n磁力線の開始点を定義中...")
    
    if use_strong_field:
        # 強い磁場領域を検出
        data = hmi_map.data[y_lims_pix[0]:y_lims_pix[1], x_lims_pix[0]:x_lims_pix[1]]
        abs_data = np.abs(data)
        
        # 閾値以上の磁場を持つピクセルを検出
        strong_field_mask = abs_data > field_threshold
        y_indices, x_indices = np.where(strong_field_mask)
        
        if len(x_indices) > 0:
            # 開始点数を制限
            n_seeds = min(n_seeds_x * n_seeds_y, len(x_indices))
            
            # ランダムにサンプリング
            if len(x_indices) > n_seeds:
                sample_indices = np.random.choice(len(x_indices), n_seeds, replace=False)
                x_indices = x_indices[sample_indices]
                y_indices = y_indices[sample_indices]
            
            # 表示範囲のオフセットを考慮してピクセル座標を計算
            x_pixels = x_indices + x_lims_pix[0]
            y_pixels = y_indices + y_lims_pix[0]
            
            print(f"  強い磁場領域から {len(x_pixels)} 点を選択")
            print(f"  磁場閾値: {field_threshold} Gauss")
        else:
            print(f"  警告: 閾値 {field_threshold} Gauss を超える磁場が見つかりません")
            print(f"  均等グリッドを使用します")
            use_strong_field = False
    
    if not use_strong_field:
        # 表示範囲内に均等グリッド点を作成
        x_pixels_1d = np.linspace(x_lims_pix[0] + 50, x_lims_pix[1] - 50, n_seeds_x)
        y_pixels_1d = np.linspace(y_lims_pix[0] + 50, y_lims_pix[1] - 50, n_seeds_y)
        
        # 2Dグリッドを作成
        x_grid, y_grid = np.meshgrid(x_pixels_1d, y_pixels_1d)
        x_pixels = x_grid.ravel()
        y_pixels = y_grid.ravel()
        
        print(f"  均等グリッドで {len(x_pixels)} 点を配置")
    
    # ピクセル座標から世界座標（太陽座標）に変換
    seeds = hmi_map.pixel_to_world(x_pixels * u.pixel, y_pixels * u.pixel)
    
    print(f"  開始点数: {len(seeds)}")
    print(f"  座標系: {seeds[0].frame}")
    
    return seeds


def trace_field_lines(seeds, pfss_output):
    """
    磁力線をトレース
    
    Parameters:
    -----------
    seeds : SkyCoord
        磁力線の開始点
    pfss_output : pfsspy.Output
        PFSS解
        
    Returns:
    --------
    list : トレースされた磁力線
    """
    print(f"\n磁力線をトレース中...")
    
    # トレーサーを初期化
    tracer = tracing.FortranTracer()
    
    # 磁力線をトレース
    field_lines = tracer.trace(seeds, pfss_output)
    
    print(f"  トレース完了: {len(field_lines)} 本")
    
    # 磁力線の統計情報
    if len(field_lines) > 0:
        n_open = 0
        n_closed = 0
        for fline in field_lines:
            if hasattr(fline.coords, 'radius'):
                end_r = fline.coords.radius[-1].to(u.Rsun).value
                if end_r > 2.0:
                    n_open += 1
                else:
                    n_closed += 1
        
        print(f"  開いた磁力線: {n_open} 本")
        print(f"  閉じた磁力線: {n_closed} 本")
    
    return field_lines


def plot_hmi_with_pfss(hmi_data, pfss_output, field_lines, save_filename=None):
    """
    HMI画像にPFSS磁力線を重ねてプロット
    
    Parameters:
    -----------
    hmi_data : dict
        prepare_hmi_for_pfss()の結果
    pfss_output : pfsspy.Output
        PFSS解
    field_lines : list
        トレースされた磁力線
    save_filename : str
        保存ファイル名
    """
    print(f"\nHMI + PFSS プロットを作成中...")
    
    # データを取得
    hmi_map = hmi_data['full_map']
    masked_data = hmi_data['masked_data']
    x_lims_pix = hmi_data['x_lims_pix']
    y_lims_pix = hmi_data['y_lims_pix']
    
    # Figure作成（2x2のサブプロット）
    fig = plt.figure(figsize=(16, 14))
    
    # 1. HMI全体マップ + 表示範囲の枠
    ax1 = fig.add_subplot(221, projection=hmi_map)
    hmi_map.plot(axes=ax1, cmap='hmimag', vmin=-1500, vmax=1500)
    
    # 表示範囲を矩形で示す
    from matplotlib.patches import Rectangle
    world_coords = hmi_map.pixel_to_world(
        [x_lims_pix[0], x_lims_pix[1]] * u.pixel,
        [y_lims_pix[0], y_lims_pix[1]] * u.pixel
    )
    rect = Rectangle(
        (x_lims_pix[0], y_lims_pix[0]),
        x_lims_pix[1] - x_lims_pix[0],
        y_lims_pix[1] - y_lims_pix[0],
        fill=False, edgecolor='red', linewidth=2,
        transform=ax1.get_transform('pixel')
    )
    ax1.add_patch(rect)
    
    ax1.set_title('HMI Full Disk + Region of Interest', fontsize=14)
    
    # 2. 指定範囲のHMI画像（plot_hmi_single.pyと同じ）
    ax2 = fig.add_subplot(222, projection=hmi_map)
    im2 = ax2.imshow(hmi_map.data, cmap='RdBu_r', origin='lower', 
                     vmin=-200, vmax=200, extent=[0, hmi_map.data.shape[1], 0, hmi_map.data.shape[0]])
    ax2.set_xlim(x_lims_pix)
    ax2.set_ylim(y_lims_pix)
    
    cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', pad=0.1, shrink=0.8)
    cbar2.ax.set_ylabel('$B_r$ (Gauss)', fontsize=12)
    ax2.set_title('HMI Radial Magnetic Field (Zoomed)', fontsize=14)
    
    # 太陽座標グリッドを描画
    draw_hmi_solar_grid(hmi_map, ax2)
    
    # 3. 指定範囲のHMI画像 + PFSS磁力線
    ax3 = fig.add_subplot(223, projection=hmi_map)
    im3 = ax3.imshow(hmi_map.data, cmap='RdBu_r', origin='lower', 
                     vmin=-200, vmax=200, extent=[0, hmi_map.data.shape[1], 0, hmi_map.data.shape[0]])
    ax3.set_xlim(x_lims_pix)
    ax3.set_ylim(y_lims_pix)
    
    # 磁力線を重ねる（色分けオプション付き）
    for fline in field_lines:
        coords = fline.coords
        
        # 磁力線の特性を判定
        if hasattr(coords, 'radius'):
            start_r = coords.radius[0].to(u.Rsun).value
            end_r = coords.radius[-1].to(u.Rsun).value
            
            # 開始点の磁場極性を判定
            start_coord = coords[0]
            start_pix = hmi_map.world_to_pixel(start_coord)
            x_pix = int(start_pix[0].value)
            y_pix = int(start_pix[1].value)
            
            # 境界チェック
            if (0 <= x_pix < hmi_map.data.shape[1] and 
                0 <= y_pix < hmi_map.data.shape[0]):
                polarity = hmi_map.data[y_pix, x_pix]
            else:
                polarity = 0
            
            # 開いた/閉じた磁力線の判定と色設定
            if end_r > 2.0:  # 開いた磁力線
                if polarity > 0:
                    color = 'yellow'  # 正極性から出る開いた磁力線
                else:
                    color = 'cyan'    # 負極性から出る開いた磁力線
                linewidth = 1.5
                alpha = 0.9
            else:  # 閉じた磁力線
                color = 'white'
                linewidth = 1.0
                alpha = 0.7
        else:
            # デフォルト設定
            color = 'white'
            linewidth = 1.2
            alpha = 0.8
        
        ax3.plot_coord(coords, alpha=alpha, linewidth=linewidth, color=color)
    
    ax3.set_title('HMI + PFSS Field Lines', fontsize=14)
    draw_hmi_solar_grid(hmi_map, ax3)
    
    # 凡例を追加
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='yellow', lw=2, label='開いた磁力線（正極性）'),
        Line2D([0], [0], color='cyan', lw=2, label='開いた磁力線（負極性）'),
        Line2D([0], [0], color='white', lw=2, label='閉じた磁力線')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 4. 磁力線の統計情報
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    # 統計を計算
    open_lines = 0
    closed_lines = 0
    positive_start = 0
    negative_start = 0
    
    for i, fline in enumerate(field_lines):
        coords = fline.coords
        if hasattr(coords, 'radius'):
            start_r = coords.radius[0].to(u.Rsun).value
            end_r = coords.radius[-1].to(u.Rsun).value
            
            # 開いた/閉じた磁力線の判定
            if end_r > 2.0:  # ソース面に到達
                open_lines += 1
            else:
                closed_lines += 1
            
            # 開始点の極性を判定
            start_coord = coords[0]
            # ピクセル座標に変換
            start_pix = hmi_map.world_to_pixel(start_coord)
            x_pix = int(start_pix[0].value)
            y_pix = int(start_pix[1].value)
            
            # 境界チェック
            if (0 <= x_pix < hmi_map.data.shape[1] and 
                0 <= y_pix < hmi_map.data.shape[0]):
                if hmi_map.data[y_pix, x_pix] > 0:
                    positive_start += 1
                else:
                    negative_start += 1
    
    stats_text = f"""PFSS磁力線解析結果

観測情報:
• HMI観測時刻: {hmi_map.date.strftime("%Y-%m-%d %H:%M:%S")}
• 表示範囲: [{x_lims_pix[0]}:{x_lims_pix[1]}, {y_lims_pix[0]}:{y_lims_pix[1]}] pixels

磁力線統計:
• 総磁力線数: {len(field_lines)}
• 開いた磁力線: {open_lines} ({open_lines/len(field_lines)*100:.1f}%)
• 閉じた磁力線: {closed_lines} ({closed_lines/len(field_lines)*100:.1f}%)
• 正極性開始: {positive_start}
• 負極性開始: {negative_start}

PFSS設定:
• ソース面: {pfss_output.source_surface_radius:.1f} Rs
• 動径格子点: {pfss_output.nr}

色分け:
• 黄色: 正極性から出る開いた磁力線
• 水色: 負極性から出る開いた磁力線  
• 白色: 閉じた磁力線
"""
    
    ax4.text(0.05, 0.95, stats_text, fontsize=11, fontfamily='monospace',
             transform=ax4.transAxes, verticalalignment='top')
    
    # メインタイトル
    fig.suptitle(f'SDO/HMI PFSS Analysis\n{hmi_data["time"]}', fontsize=16)
    
    plt.tight_layout()
    
    # 保存
    if save_filename is None:
        save_filename = '/mnt/d/wsl/home/kinno-7010/Research/PFSS/hmi_pfss_overlay.png'
    
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"  プロットを保存: {save_filename}")
    
    plt.show()
    
    return fig


def main():
    """
    メイン実行関数
    """
    print("=== HMI + PFSS 磁力線重ね合わせプロット ===")
    print("=" * 50)
    
    # 再現性のためのシード設定
    np.random.seed(42)
    
    # HMIファイルパス
    hmi_file = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    
    try:
        # 1. HMIデータの準備
        print("\n--- Step 1: HMIデータ準備 ---")
        hmi_data = prepare_hmi_for_pfss(hmi_file)
        
        # 2. PFSS解の計算（全球データ使用）
        print("\n--- Step 2: PFSS計算 ---")
        pfss_output = compute_pfss_solution(hmi_data['full_map'], nrho=25, rss=2.5)
        
        # 3. 磁力線の開始点を定義（表示範囲内）
        print("\n--- Step 3: 磁力線開始点定義 ---")
        # オプション1: 均等グリッド
        # seeds = define_field_line_seeds(
        #     hmi_data['full_map'], 
        #     hmi_data['x_lims_pix'], 
        #     hmi_data['y_lims_pix'],
        #     n_seeds_x=7,  # X方向の開始点数
        #     n_seeds_y=7   # Y方向の開始点数
        # )
        
        # オプション2: 強い磁場領域から選択
        seeds = define_field_line_seeds(
            hmi_data['full_map'], 
            hmi_data['x_lims_pix'], 
            hmi_data['y_lims_pix'],
            n_seeds_x=10,  # 最大開始点数
            n_seeds_y=10,
            use_strong_field=True,  # 強い磁場領域を使用
            field_threshold=150     # 磁場強度閾値 (Gauss)
        )
        
        # 4. 磁力線をトレース
        print("\n--- Step 4: 磁力線トレース ---")
        field_lines = trace_field_lines(seeds, pfss_output)
        
        # 5. プロット作成
        print("\n--- Step 5: プロット作成 ---")
        fig = plot_hmi_with_pfss(hmi_data, pfss_output, field_lines)
        
        print("\n=== 処理完了 ===")
        print("✓ HMI + PFSS 磁力線重ね合わせプロットが完了しました")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()