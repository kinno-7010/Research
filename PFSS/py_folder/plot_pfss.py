import sunpy.map
import pfsspy
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from plot_hmi_pfss_overlay import draw_hmi_solar_grid, prepare_hmi_for_pfss, compute_pfss_solution, define_field_line_seeds, trace_field_lines, plot_sdo_aia_rgb

def plot_pfss(hmi_map, field_lines, x_lims_pix, y_lims_pix, RSS_VAL):
    fig = plt.figure(figsize=(10, 10))
    ax4 = fig.add_subplot(111, projection=hmi_map)
    
    # HMI画像を表示
    im4 = ax4.imshow(hmi_map.data, cmap='RdBu_r', origin='lower', vmin=-200, vmax=200)
    
    # HMI画像範囲より広い軸範囲を設定（磁力線を画像範囲外まで表示）
    # extended_x_range = (x_lims_pix[0] - 400, x_lims_pix[1])
    # extended_y_range = (y_lims_pix[0], y_lims_pix[1] + 400)
    extended_x_range = (x_lims_pix[0], x_lims_pix[1])
    extended_y_range = (y_lims_pix[0], y_lims_pix[1])
    ax4.set_xlim(extended_x_range)
    ax4.set_ylim(extended_y_range)
    
    for fline in field_lines:
        coords = fline.coords
        
        # pfsspyの磁力線分類機能とsunpyのWCS変換を活用
        if hasattr(coords, 'radius'):
            # pfsspyのis_openプロパティを優先使用
            if hasattr(fline, 'is_open'):
                is_open_line = fline.is_open
            else:
                # フォールバック: 半径による判定
                end_r = fline.coords.radius[-1].to(u.Rsun).value
                is_open_line = end_r > 2.0
            
            # sunpyのWCS変換で磁極性を判定
            try:
                start_pix = hmi_map.world_to_pixel(coords[0])
                x_pix, y_pix = int(start_pix.x.value), int(start_pix.y.value)
            except AttributeError:
                start_pix = hmi_map.world_to_pixel(coords[0])
                x_pix, y_pix = int(start_pix[0].value), int(start_pix[1].value)
            
            polarity = hmi_map.data[y_pix, x_pix] if (0 <= y_pix < hmi_map.data.shape[0] and 0 <= x_pix < hmi_map.data.shape[1]) else 0
            
            # 磁力線の色分け
            color, linewidth, alpha = ('black', 0.5, 0.7)
            if is_open_line:
                color = 'red' if polarity > 0 else 'blue'
                linewidth, alpha = 0.5, 0.7
            
            # sunpyのplot_coord機能を使用
            ax4.plot_coord(coords, alpha=alpha, linewidth=linewidth, color=color)

    ax4.set_title(f'SDO/HMI Magnetogram + PFSS Field Lines (Rss={RSS_VAL})', fontsize=14)
    ax4.set_xlabel('Solar X (arcsec)', fontsize=12)
    ax4.set_ylabel('Solar Y (arcsec)', fontsize=12)
    draw_hmi_solar_grid(hmi_map, ax4)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=0.7, label='Open Lines (from N-pole)'),
        Line2D([0], [0], color='blue', lw=0.7, label='Open Lines (from S-pole)'),
        Line2D([0], [0], color='black', lw=0.7, label='Closed Lines')
    ]
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    fig.suptitle(f'SDO/HMI + PFSS Field Lines {hmi_map.date}', fontsize=16)
    # plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
    # fig.subplots_adjust(top=0.92)

    save_filename = f'/mnt/d/wsl/home/kinno-7010/Research/PFSS/pfss_plot_Rss_{RSS_VAL}.png'
    
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
    # PFSSパラメータを定数として定義（変更可能）
    RSS_VAL = 2.5   # Source Surface高度 (Rs) - 必要に応じて変更可能: 1.5, 2.0, 2.5, 3.0など
    NRHO_VAL = 50   # 動径方向格子点数 - 精度向上のため30→50に増加
    # ======================= ここまでが修正部分 =======================
    
    try:
        # 1. HMIデータの準備
        print("\n--- Step 1: HMIデータ準備 ---")
        hmi_data = prepare_hmi_for_pfss(hmi_file)

        
        # 2. PFSS解の計算（全球データ使用）
        print("\n--- Step 2: PFSS計算 ---")
        print(f"  使用パラメータ: Source Surface = {RSS_VAL} Rs, 動径格子点数 = {NRHO_VAL}")
        pfss_output = compute_pfss_solution(hmi_data['full_map'], nrho=NRHO_VAL, rss=RSS_VAL)
        
        # 3. 磁力線の開始点を定義（表示範囲内）
        print("\n--- Step 3: 磁力線開始点定義 ---")
        seeds = define_field_line_seeds(
            hmi_data['full_map'], 
            hmi_data['x_lims_pix'], 
            hmi_data['y_lims_pix'],
            n_seeds_x=15, # 磁力線の開始点の数
            n_seeds_y=15, # 磁力線の開始点の数
            use_strong_field=True,
            field_threshold=200 # 磁場強度の閾値
        )
        
        # 4. 磁力線をトレース
        print("\n--- Step 4: 磁力線トレース ---")
        field_lines = trace_field_lines(seeds, pfss_output)
        
        # 5. プロット作成
        print("\n--- Step 5: プロット作成 ---")
        # ======================= ここからが修正部分 =======================
        # plot関数にrssとnrhoの値を渡す
        fig = plot_pfss(hmi_data['full_map'], field_lines, hmi_data['x_lims_pix'], hmi_data['y_lims_pix'], RSS_VAL)
        # ======================= ここまでが修正部分 =======================
        
        print("\n=== 処理完了 ===")
        print("✓ HMI + PFSS 磁力線重ね合わせプロットが完了しました")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()