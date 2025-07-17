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
        hmi_map_clean = sunpy.map.Map(data, hmi_map.meta)
    else:
        hmi_map_clean = hmi_map
        print("  NaN値は検出されませんでした")
    
    # CEA投影にリプロジェクション
    print("  HMIマップをCEA投影にリプロジェクション中...")
    shape_cea = (180, 360)

    reference_coord = SkyCoord(0 * u.deg, 0 * u.deg,
                               frame='heliographic_stonyhurst',
                               obstime=hmi_map.date,
                               rsun=hmi_map.rsun_meters)

    # CEA投影用のヘッダーを作成
    header_cea = sunpy.map.make_fitswcs_header(
        shape_cea,
        reference_coord,
        projection_code="CEA"
    )

    # ======================= ここからが修正部分 (最終版) =======================
    # pfsspyライブラリの厳密な内部検証を通過するために、スケール値を仕様通りに手動設定する

    # 経度スケール (CDELT1): 360度 / 360ピクセル = 1.0
    header_cea['cdelt1'] = 360.0 / shape_cea[1]
    header_cea['cunit1'] = 'deg'

    # 緯度スケール (CDELT2): pfsspyの検証式から逆算した値 (2 / pi) を設定する
    header_cea['cdelt2'] = 2 / np.pi
    header_cea['cunit2'] = 'deg'
    # ======================= ここまでが修正部分 (最終版) =======================

    # マップをリプロジェクション
    hmi_map_cea = hmi_map_clean.reproject_to(header_cea)
    
    # リプロジェクション後のマップのNaNも0で埋める
    hmi_map_cea.data[np.isnan(hmi_map_cea.data)] = 0.0
    print(f"  リプロジェクション完了. 新しい形状: {hmi_map_cea.data.shape}")
    
    # PFSS入力オブジェクトを作成
    pfss_input = pfsspy.Input(hmi_map_cea, nrho, rss)
    
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


def plot_hmi_with_pfss(hmi_data, pfss_output, field_lines, rss, nrho, save_filename=None):
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
    rss : float
        ソース面の半径 (Rs)
    nrho : int
        動径方向の格子点数
    save_filename : str
        保存ファイル名
    """
    print(f"\nHMI + PFSS プロットを作成中...")
    
    # データを取得
    hmi_map = hmi_data['full_map']
    x_lims_pix = hmi_data['x_lims_pix']
    y_lims_pix = hmi_data['y_lims_pix']
    
    # Figure作成（2x2のサブプロット）
    fig = plt.figure(figsize=(16, 14))
    
    # 1. HMI全体マップ + 表示範囲の枠
    ax1 = fig.add_subplot(221, projection=hmi_map)
    hmi_map.plot_settings['norm'].vmin = -1500
    hmi_map.plot_settings['norm'].vmax = 1500
    hmi_map.plot(axes=ax1, cmap='hmimag')
    
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (x_lims_pix[0], y_lims_pix[0]),
        x_lims_pix[1] - x_lims_pix[0],
        y_lims_pix[1] - y_lims_pix[0],
        fill=False, edgecolor='red', linewidth=2,
        transform=ax1.get_transform('pixel')
    )
    ax1.add_patch(rect)
    ax1.set_title('HMI Full Disk + Region of Interest', fontsize=14)
    
    # 2. 指定範囲のHMI画像
    ax2 = fig.add_subplot(222, projection=hmi_map)
    im2 = ax2.imshow(hmi_map.data, cmap='RdBu_r', origin='lower', vmin=-200, vmax=200)
    ax2.set_xlim(x_lims_pix)
    ax2.set_ylim(y_lims_pix)
    cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', pad=0.1, shrink=0.8)
    cbar2.ax.set_ylabel('$B_r$ (Gauss)', fontsize=12)
    ax2.set_title('HMI Radial Magnetic Field (Zoomed)', fontsize=14)
    draw_hmi_solar_grid(hmi_map, ax2)
    
    # 3. 指定範囲のHMI画像 + PFSS磁力線
    ax3 = fig.add_subplot(223, projection=hmi_map)
    im3 = ax3.imshow(hmi_map.data, cmap='RdBu_r', origin='lower', vmin=-200, vmax=200)
    ax3.set_xlim(x_lims_pix)
    ax3.set_ylim(y_lims_pix)
    
    for fline in field_lines:
        coords = fline.coords
        if hasattr(coords, 'radius'):
            end_r = fline.coords.radius[-1].to(u.Rsun).value
            try:
                start_pix = hmi_map.world_to_pixel(coords[0])
                x_pix, y_pix = int(start_pix.x.value), int(start_pix.y.value)
            except AttributeError:
                start_pix = hmi_map.world_to_pixel(coords[0])
                x_pix, y_pix = int(start_pix[0].value), int(start_pix[1].value)
            
            polarity = hmi_map.data[y_pix, x_pix] if (0 <= y_pix < hmi_map.data.shape[0] and 0 <= x_pix < hmi_map.data.shape[1]) else 0
            
            color, linewidth, alpha = ('white', 1.0, 0.7)
            if end_r > 2.0:
                color = 'yellow' if polarity > 0 else 'cyan'
                linewidth, alpha = 1.5, 0.9
            
            ax3.plot_coord(coords, alpha=alpha, linewidth=linewidth, color=color)

    ax3.set_title('HMI + PFSS Field Lines', fontsize=14)
    draw_hmi_solar_grid(hmi_map, ax3)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='yellow', lw=2, label='Open Lines (from N-pole)'),
        Line2D([0], [0], color='cyan', lw=2, label='Open Lines (from S-pole)'),
        Line2D([0], [0], color='white', lw=2, label='Closed Lines')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 4. 磁力線の統計情報
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    open_lines = sum(1 for fline in field_lines if fline.is_open)
    closed_lines = len(field_lines) - open_lines
    
    positive_start, negative_start = 0, 0
    for fline in field_lines:
        try:
            start_pix = hmi_map.world_to_pixel(fline.coords[0])
            x_pix, y_pix = int(start_pix.x.value), int(start_pix.y.value)
        except AttributeError:
            start_pix = hmi_map.world_to_pixel(fline.coords[0])
            x_pix, y_pix = int(start_pix[0].value), int(start_pix[1].value)

        if (0 <= y_pix < hmi_map.data.shape[0] and 0 <= x_pix < hmi_map.data.shape[1]):
            if hmi_map.data[y_pix, x_pix] > 0:
                positive_start += 1
            else:
                negative_start += 1
    
    # ======================= ここからが修正部分 =======================
    # 引数で渡された rss と nrho を使用する
    stats_text = f"""PFSS Field Line Analysis

Observation Info:
- HMI Obs Time: {hmi_map.date.strftime("%Y-%m-%d %H:%M:%S")}
- Display Range: [{x_lims_pix[0]}:{x_lims_pix[1]}, {y_lims_pix[0]}:{y_lims_pix[1]}] px

Field Line Statistics:
- Total Lines: {len(field_lines)}
- Open Lines: {open_lines} ({open_lines/len(field_lines)*100:.1f}%)
- Closed Lines: {closed_lines} ({closed_lines/len(field_lines)*100:.1f}%)
- Start from N-pole: {positive_start}
- Start from S-pole: {negative_start}

PFSS Settings:
- Source Surface: {rss:.1f} Rs
- Radial Grid Pts: {nrho}

Color Legend:
- Yellow: Open (from N-pole)
- Cyan: Open (from S-pole)
- White: Closed
"""
    # ======================= ここまでが修正部分 =======================
    
    ax4.text(0.05, 0.95, stats_text, fontsize=11, fontfamily='monospace',
             transform=ax4.transAxes, verticalalignment='top')
    
    fig.suptitle(f'SDO/HMI PFSS Analysis\n{hmi_data["time"]}', fontsize=16)
    plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=1.5)
    fig.subplots_adjust(top=0.92)

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
    
    # ======================= ここからが修正部分 =======================
    # PFSSパラメータを定数として定義
    RSS_VAL = 2.5
    NRHO_VAL = 25
    # ======================= ここまでが修正部分 =======================
    
    try:
        # 1. HMIデータの準備
        print("\n--- Step 1: HMIデータ準備 ---")
        hmi_data = prepare_hmi_for_pfss(hmi_file)
        
        # 2. PFSS解の計算（全球データ使用）
        print("\n--- Step 2: PFSS計算 ---")
        pfss_output = compute_pfss_solution(hmi_data['full_map'], nrho=NRHO_VAL, rss=RSS_VAL)
        
        # 3. 磁力線の開始点を定義（表示範囲内）
        print("\n--- Step 3: 磁力線開始点定義 ---")
        seeds = define_field_line_seeds(
            hmi_data['full_map'], 
            hmi_data['x_lims_pix'], 
            hmi_data['y_lims_pix'],
            n_seeds_x=10,
            n_seeds_y=10,
            use_strong_field=True,
            field_threshold=150
        )
        
        # 4. 磁力線をトレース
        print("\n--- Step 4: 磁力線トレース ---")
        field_lines = trace_field_lines(seeds, pfss_output)
        
        # 5. プロット作成
        print("\n--- Step 5: プロット作成 ---")
        # ======================= ここからが修正部分 =======================
        # plot関数にrssとnrhoの値を渡す
        fig = plot_hmi_with_pfss(hmi_data, pfss_output, field_lines, rss=RSS_VAL, nrho=NRHO_VAL)
        # ======================= ここまでが修正部分 =======================
        
        print("\n=== 処理完了 ===")
        print("✓ HMI + PFSS 磁力線重ね合わせプロットが完了しました")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()