#!/usr/bin/env python3
"""
HMIデータの指定範囲にPFSS磁力線を重ねてプロット
plot_aia_overplotting.pyの手法をHMIデータに適用
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from astropy.time import Time
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm


# 必要なパッケージのインポート
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.map
import pfsspy
import pfsspy.tracing as tracing

import astropy.visualization as vis



# HMI解析モジュールのインポート
sys.path.append('/home/kinno-7010/Research_code/SDO/HMI/py_folder')
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
    
    # ピクセル座標から世界座標（太陽座標）に変換（sunpy WCS機能を活用）
    seeds = hmi_map.pixel_to_world(x_pixels * u.pixel, y_pixels * u.pixel)
    
    # sunpyのSkyCoordを使用して座標系の統一性を確保
    if not isinstance(seeds, SkyCoord):
        seeds = SkyCoord(seeds)
    
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
    
    # トレーサーを初期化（改善されたパラメータ）
    tracer = tracing.FortranTracer(
        max_steps=50000,    # ステップ数をさらに増加（警告対応）
        step_size=0.01      # より小さなステップサイズで精度向上
    )
    
    # 磁力線をトレース
    field_lines = tracer.trace(seeds, pfss_output)
    
    print(f"  トレース完了: {len(field_lines)} 本")
    
    # 磁力線の統計情報（pfsspyの分類機能を活用）
    if len(field_lines) > 0:
        n_open = 0
        n_closed = 0
        for fline in field_lines:
            # pfsspyの磁力線オブジェクトのプロパティを利用
            if hasattr(fline, 'is_open'):
                if fline.is_open:
                    n_open += 1
                else:
                    n_closed += 1
            elif hasattr(fline.coords, 'radius'):
                # フォールバック: 半径による判定
                end_r = fline.coords.radius[-1].to(u.Rsun).value
                if end_r > 2.0:
                    n_open += 1
                else:
                    n_closed += 1
        
        print(f"  開いた磁力線: {n_open} 本")
        print(f"  閉じた磁力線: {n_closed} 本")
    
    return field_lines

def normalize_log_stretch(data):
    """Applies LogStretch normalization to the data."""
    # LogStretchは負の値を扱えないため、ゼロや負の値を避けるためにクリッピングする
    data_clipped = np.maximum(data, 1e-5)
    normalizer = vis.ImageNormalize(data_clipped, stretch=vis.LogStretch(), clip=True)
    return normalizer(data_clipped)

def plot_sdo_aia_rgb(datetime_str,
                     channel_r_str="211",
                     channel_g_str="193",
                     channel_b_str="171",
                     return_data=False):
    """
    指定された日時のSDO/AIAデータ (3波長) を読み込み、RGB合成画像をWCSベースでプロットします。
    軸の目盛りラベルは太陽中心を(0,0)とするピクセル単位で表示します。
    
    Parameters:
    -----------
    datetime_str : str
        日時文字列
    channel_r_str, channel_g_str, channel_b_str : str
        各チャンネルの波長
    return_data : bool
        TrueならRGB画像データを返す、Falseならプロットのみ
        
    Returns:
    --------
    tuple or None : return_data=Trueの場合、(rgb_image, reference_map, success)
    """
    BASE_DATA_DIR = Path('/mnt/d/wsl/home/kinno-7010/Research_data/SDO/AIA/Rawdata')
    # 1. 日時文字列のパース
    try:
        dt_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
        date_fmtd_for_fname = dt_obj.strftime("%Y%m%d")
        time_fmtd_for_fname = dt_obj.strftime("%H%M")
    except ValueError:
        print(
            f"エラー: 日時文字列 '{datetime_str}' の形式が無効です。"
            " 'YYYY-MM-DD HH:MM' 形式で指定してください。"
        )
        return

    # 2. 各チャンネルのファイルパスを組み立て、Mapオブジェクトをロード
    maps = {}
    channels = {'r': channel_r_str, 'g': channel_g_str, 'b': channel_b_str}
    loaded_map_count = 0

    for color, ch_str in channels.items():
        wavelength_part_in_fname = ch_str.zfill(4)
        filename = f"AIA{date_fmtd_for_fname}_{time_fmtd_for_fname}_{wavelength_part_in_fname}.fits"
        file_path = BASE_DATA_DIR / ch_str / filename
        print(f"読み込み試行: {color.upper()}チャンネル ({ch_str}Å) - {file_path}")
        try:
            # sunpyの詳細なエラーハンドリングを活用
            maps[color] = sunpy.map.Map(file_path)
            # sunpyマップの基本検証
            if hasattr(maps[color], 'data') and maps[color].data is not None:
                print(f"  成功: {ch_str}Å (データ形状: {maps[color].data.shape})")
                loaded_map_count += 1
            else:
                print(f"  警告: {ch_str}Å のデータが無効です")
                maps[color] = None
        except Exception as e:
            print(f"  失敗: {ch_str}Å のファイル読み込みエラー: {e}")
            maps[color] = None

    if loaded_map_count < 3:
        print("エラー: 3つ全ての波長チャンネルのデータを読み込めませんでした。プロットを中止します。")
        if return_data:
            return None, None, False
        return

    # 基準となるMapオブジェクトを選択 (WCS情報、メタデータ、リム/グリッド描画に使用)
    reference_map = maps['b'] if maps['b'] else maps['g'] if maps['g'] else maps['r']
    if not reference_map:
        print("エラー: 基準となるMapオブジェクトがありません。")
        if return_data:
            return None, None, False
        return

    wcs_info = reference_map.wcs

    # 3. 各チャンネルのデータを正規化（線形ストレッチ使用）
    try:
        red_channel_data = normalize_log_stretch(maps['r'].data)
        green_channel_data = normalize_log_stretch(maps['g'].data)
        blue_channel_data = normalize_log_stretch(maps['b'].data)
    except Exception as e_norm:
        print(f"データ正規化中にエラー: {e_norm}")
        if return_data:
            return None, None, False
        return

    # 4. RGB画像の作成 (0-1にスケーリング、パーセンタイル使用)
    def scale_to_01_percentile(data, pmin=1.0, pmax=99.5):
        """パーセンタイルベースのスケーリング"""
        valid_data = data[np.isfinite(data)]
        if valid_data.size == 0:
            return np.zeros_like(data)
        
        vmin = np.percentile(valid_data, pmin)
        vmax = np.percentile(valid_data, pmax)
        
        if vmax == vmin:
            return np.zeros_like(data)
        
        scaled = (data - vmin) / (vmax - vmin)
        return np.clip(scaled, 0, 1)

    red_channel_final = scale_to_01_percentile(red_channel_data)
    green_channel_final = scale_to_01_percentile(green_channel_data)
    blue_channel_final = scale_to_01_percentile(blue_channel_data)
    rgb_image = np.stack([red_channel_final, green_channel_final, blue_channel_final], axis=-1)

    # 5. ピクセルスケールを取得
    cdelt1 = reference_map.meta.get('cdelt1')
    cdelt2 = reference_map.meta.get('cdelt2')
    use_pixel_formatter = False
    pixel_scale_x = 1.0
    pixel_scale_y = 1.0

    if cdelt1 is not None and cdelt2 is not None and cdelt1 != 0 and cdelt2 != 0:
        pixel_scale_x = abs(cdelt1)
        pixel_scale_y = abs(cdelt2)
        if hasattr(wcs_info, 'wcs') and hasattr(wcs_info.wcs, 'cunit'):
            if wcs_info.wcs.cunit[0] == u.arcsec and wcs_info.wcs.cunit[1] == u.arcsec:
                use_pixel_formatter = True
        else:
            print("警告: WCS単位を確認できません。ピクセル目盛りは無効の可能性があります。")
    else:
        print("警告: CDELTが取得不可または0です。ピクセル目盛りは無効です。")

    # 6. 目盛りフォーマッタ関数
    def arcsec_to_pixel_offset_formatter(arcsec_value, pos, scale_arcsec_per_pixel):
        if scale_arcsec_per_pixel == 0:
            return f"{0:.0f}"
        return f"{(arcsec_value / scale_arcsec_per_pixel):.0f}"

    # 7. プロット準備
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection=wcs_info)

    # 8. RGB画像のプロット
    ax.imshow(rgb_image, origin='lower', aspect='equal')

    # 9. 太陽リムとグリッドの描画
    try:
        reference_map.draw_limb(axes=ax, color='white', linestyle='dashed', linewidth=1.2)
        reference_map.draw_grid(axes=ax, grid_spacing=15*u.deg, color='white', linestyle='dotted', linewidth=0.8, alpha=0.7)
        print("情報: 太陽リムとグリッドをWCSベースで描画しました。")
    except Exception as e_draw:
        print(f"警告: 太陽リムまたはグリッドの描画に失敗しました: {e_draw}")

    # 10. タイトルと軸ラベルの設定
    title_str_parts = [
        f"SDO/AIA Composite: R={channel_r_str}Å, G={channel_g_str}Å, B={channel_b_str}Å",
        f"{reference_map.date.strftime('%Y-%m-%d %H:%M:%S UT')}"
    ]
    if use_pixel_formatter:
        title_str_parts.append(f"Tick Labels in Pixels (Ref. Scale ≈ {pixel_scale_x:.2f}\" /pix)")
    ax.set_title("\n".join(title_str_parts), fontsize=12, pad=15)

    if use_pixel_formatter:
        ax.coords[0].set_major_formatter(FuncFormatter(lambda val, pos: arcsec_to_pixel_offset_formatter(val, pos, pixel_scale_x)))
        ax.coords[0].set_axislabel("Solar X (pixels from Sun center, ref. WCS)")
        ax.coords[1].set_major_formatter(FuncFormatter(lambda val, pos: arcsec_to_pixel_offset_formatter(val, pos, pixel_scale_y)))
        ax.coords[1].set_axislabel("Solar Y (pixels from Sun center, ref. WCS)")
    else:
        cunit1 = wcs_info.wcs.cunit[0] if hasattr(wcs_info, 'wcs') and hasattr(wcs_info.wcs, 'cunit') else u.arcsec
        cunit2 = wcs_info.wcs.cunit[1] if hasattr(wcs_info, 'wcs') and hasattr(wcs_info.wcs, 'cunit') else u.arcsec
        ax.coords[0].set_axislabel(f"Solar X ({cunit1})")
        ax.coords[1].set_axislabel(f"Solar Y ({cunit2})")

    ax.tick_params(axis='both', which='major', labelsize=10, direction='in')

    # return_dataがTrueの場合、RGB画像データを返す
    if return_data:
        return rgb_image, reference_map, True

    plt.tight_layout()
    plt.show()


def find_files_in_time_range(start_time: str, end_time: str, time_tolerance_seconds: int = 12) -> defaultdict:
    """
    指定された時間範囲にあるFITSファイルを検索し、時刻をグループ化して返す。
    時刻のわずかなズレを許容するため、指定された秒数で時刻を丸める（タイムビン）。
    """
    BASE_DATA_DIR = Path('/mnt/d/wsl/home/kinno-7010/Research_data/SDO/AIA/Rawdata')
    print(f"ディレクトリ '{BASE_DATA_DIR}' 内の.fits/.ftsファイルを再帰的に検索しています...")
    all_files = sorted(BASE_DATA_DIR.rglob('*.fits')) + sorted(BASE_DATA_DIR.rglob('*.fts'))
    all_files = sorted(list(set(all_files))) # 重複を削除

    if not all_files:
        print("警告: FITSファイルが一つも見つかりませんでした。")
        return defaultdict(list)
    else:
        print(f"{len(all_files)}個のFITSファイルが見つかりました。")

    t_start = Time(start_time)
    t_end = Time(end_time)
    files_by_time = defaultdict(list)

    print(f"各ファイルを読み込み、約{time_tolerance_seconds}秒間の時間幅でグループ化しています...")
    for f in tqdm(all_files, desc="ファイルフィルタリング"):
        try:
            # sunpyの堅牢なファイル読み込み機能を活用
            m = sunpy.map.Map(f)
            file_time = m.date
            
            # sunpyのTime比較機能を使用
            if t_start <= file_time <= t_end:
                # --- ★★★ 新しいグループ化ロジック ★★★ ---
                dt_obj = file_time.to_datetime()
                
                # UNIXタイムスタンプ（秒）に変換し、指定秒数で丸める
                total_seconds = dt_obj.timestamp()
                binned_seconds = round(total_seconds / time_tolerance_seconds) * time_tolerance_seconds
                
                # 丸めた秒数から、グループ化のキーとなるdatetimeオブジェクトを再構築
                time_key = datetime.fromtimestamp(binned_seconds)
                
                files_by_time[time_key].append(f)
                
        except Exception as e:
            print(f"ファイル {f.name} の読み込み/解析中にエラー: {e}")
            continue

    return files_by_time




def plot_hmi_with_pfss(hmi_data, aia_rgb_data, pfss_output, field_lines, rss, nrho, save_filename=None):
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
    aia_rgb_map = aia_rgb_data['aia_rgb_map']
    x_lims_pix = hmi_data['x_lims_pix']
    y_lims_pix = hmi_data['y_lims_pix']
    
    # Figure作成（2x2のサブプロット）
    fig = plt.figure(figsize=(16, 14))
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, left=0.1, right=0.9, bottom=0.1, top=0.9,
                  wspace=0.2, hspace=0.2)

    # 1. HMI全体マップ (左上)
    ax1 = fig.add_subplot(gs[0, 0], projection=hmi_map)    
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
    ax1.set_xlabel('Solar X (arcsec)', fontsize=12)
    ax1.set_ylabel('Solar Y (arcsec)', fontsize=12)
    ax1.add_patch(rect)
    ax1.set_title('SDO/HMI Magnetogram + AR 13030, 13032', fontsize=14)
    
    # 2. 指定範囲のSDO/AIA RGB画像
    ax2 = fig.add_subplot(gs[0, 1], projection=aia_rgb_map)
    
    # AIA RGB画像をHMI画像と同じサイズにリサイズ
    from scipy.ndimage import zoom
    
    # HMI画像のサイズを取得
    hmi_shape = hmi_map.data.shape
    
    # ズーム係数を計算
    zoom_y = hmi_shape[0] / aia_rgb_data['rgb_image'].shape[0]
    zoom_x = hmi_shape[1] / aia_rgb_data['rgb_image'].shape[1]
    
    # 各チャンネルを個別にリサイズ
    resized_rgb = np.zeros((hmi_shape[0], hmi_shape[1], 3))
    for i in range(3):
        resized_rgb[:, :, i] = zoom(aia_rgb_data['rgb_image'][:, :, i], (zoom_y, zoom_x), order=1)
    
    # AIA RGB画像を表示
    im2 = ax2.imshow(resized_rgb, origin='lower')
    
    # sunpyの座標変換機能を活用してarcsec座標をピクセル座標に変換
    # X: -1024"~0", Y: -200"~1024" arcsec
    coord_x1 = SkyCoord(-880*u.arcsec, 0*u.arcsec, frame=hmi_map.coordinate_frame)
    coord_x2 = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=hmi_map.coordinate_frame)
    coord_y1 = SkyCoord(0*u.arcsec, -180*u.arcsec, frame=hmi_map.coordinate_frame)
    coord_y2 = SkyCoord(0*u.arcsec, 880*u.arcsec, frame=hmi_map.coordinate_frame)
    
    # sunpyのWCS変換機能を使用
    pix_x1 = hmi_map.world_to_pixel(coord_x1)
    pix_x2 = hmi_map.world_to_pixel(coord_x2)
    pix_y1 = hmi_map.world_to_pixel(coord_y1)
    pix_y2 = hmi_map.world_to_pixel(coord_y2)
    
    # ピクセル座標で範囲を設定（HMIの切り取り方と同様）
    x_min_aia = int(pix_x1.x.value) if hasattr(pix_x1, 'x') else int(pix_x1[0].value)
    x_max_aia = int(pix_x2.x.value) if hasattr(pix_x2, 'x') else int(pix_x2[0].value)
    y_min_aia = int(pix_y1.y.value) if hasattr(pix_y1, 'y') else int(pix_y1[1].value)
    y_max_aia = int(pix_y2.y.value) if hasattr(pix_y2, 'y') else int(pix_y2[1].value)
    
    ax2.set_xlim(x_min_aia, x_max_aia)
    ax2.set_ylim(y_min_aia, y_max_aia)
    ax2.set_xlabel('Solar X (arcsec)', fontsize=12)
    ax2.set_ylabel('Solar Y (arcsec)', fontsize=12)
    ax2.set_title('SDO/AIA RGB (211/193/171)', fontsize=14)
    
    # AIA RGB画像用のグリッド描画
    try:
        aia_rgb_map.draw_limb(axes=ax2, color='white', linestyle='dashed', linewidth=1.2)
        aia_rgb_map.draw_grid(axes=ax2, grid_spacing=15*u.deg, color='white', linestyle='dotted', linewidth=0.8, alpha=0.7)
    except Exception as e:
        print(f"警告: AIA RGB用グリッド描画に失敗: {e}")
        # フォールバックとしてHMIマップのグリッドを使用
        draw_hmi_solar_grid(hmi_map, ax2)
    
    # 3. 指定範囲のHMI画像のみ
    ax3 = fig.add_subplot(gs[1, 0], projection=hmi_map)
    
    # HMI画像を表示
    im3 = ax3.imshow(hmi_map.data, cmap='RdBu_r', origin='lower', vmin=-200, vmax=200)
    ax3.set_xlim(x_lims_pix)
    ax3.set_ylim(y_lims_pix)
    ax3.set_xlabel('Solar X (arcsec)', fontsize=12)
    ax3.set_ylabel('Solar Y (arcsec)', fontsize=12)
    cbar3 = fig.colorbar(im3, ax=ax3, orientation='vertical', pad=0.1, shrink=0.8)
    cbar3.ax.set_ylabel('$B_r$ (Gauss)', fontsize=12)
    ax3.set_title(f'SDO/HMI Radial Magnetic Field (Source Surface: {rss:.1f} Rs)', fontsize=14)
    draw_hmi_solar_grid(hmi_map, ax3)
    
    # 4. 指定範囲のHMI画像 + PFSS磁力線
    ax4 = fig.add_subplot(gs[1, 1], projection=hmi_map)
    
    # HMI画像を表示
    im4 = ax4.imshow(hmi_map.data, cmap='RdBu_r', origin='lower', vmin=-200, vmax=200)
    ax4.set_xlim(x_lims_pix)
    ax4.set_ylim(y_lims_pix)
    
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

    ax4.set_title('SDO/HMI Magnetogram + PFSS Field Lines', fontsize=14)
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
    
#     # 4. 磁力線の統計情報
#     ax4 = fig.add_subplot(224)
#     ax4.axis('off')
    
#     open_lines = sum(1 for fline in field_lines if fline.is_open)
#     closed_lines = len(field_lines) - open_lines
    
#     positive_start, negative_start = 0, 0
#     for fline in field_lines:
#         try:
#             start_pix = hmi_map.world_to_pixel(fline.coords[0])
#             x_pix, y_pix = int(start_pix.x.value), int(start_pix.y.value)
#         except AttributeError:
#             start_pix = hmi_map.world_to_pixel(fline.coords[0])
#             x_pix, y_pix = int(start_pix[0].value), int(start_pix[1].value)

#         if (0 <= y_pix < hmi_map.data.shape[0] and 0 <= x_pix < hmi_map.data.shape[1]):
#             if hmi_map.data[y_pix, x_pix] > 0:
#                 positive_start += 1
#             else:
#                 negative_start += 1
    
#     # ======================= ここからが修正部分 =======================
#     # 引数で渡された rss と nrho を使用する
#     stats_text = f"""PFSS Field Line Analysis

# Observation Info:
# - HMI Obs Time: {hmi_map.date.strftime("%Y-%m-%d %H:%M:%S")}
# - Display Range: [{x_lims_pix[0]}:{x_lims_pix[1]}, {y_lims_pix[0]}:{y_lims_pix[1]}] px

# Field Line Statistics:
# - Total Lines: {len(field_lines)}
# - Open Lines: {open_lines} ({open_lines/len(field_lines)*100:.1f}%)
# - Closed Lines: {closed_lines} ({closed_lines/len(field_lines)*100:.1f}%)
# - Start from N-pole: {positive_start}
# - Start from S-pole: {negative_start}

# PFSS Settings:
# - Source Surface: {rss:.1f} Rs
# - Radial Grid Pts: {nrho}

# Color Legend:
# - Red: Open (from N-pole)
# - Blue: Open (from S-pole)
# - Black: Closed
# """
    # # ======================= ここまでが修正部分 =======================
    
    # ax4.text(0.05, 0.95, stats_text, fontsize=11, fontfamily='monospace',
    #          transform=ax4.transAxes, verticalalignment='top')
    
    fig.suptitle(f'SDO/HMI + PFSS Field Lines {hmi_data["time"]}', fontsize=16)
    # plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
    # fig.subplots_adjust(top=0.92)

    if save_filename is None:
        save_filename = '/mnt/d/wsl/home/kinno-7010/Research_data/PFSS/hmi_pfss_overlay.png'
    
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
    hmi_file = "/mnt/d/wsl/home/kinno-7010/Research_data/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    
    # ======================= ここからが修正部分 =======================
    # PFSSパラメータを定数として定義（変更可能）
    RSS_VAL = 3.0   # Source Surface高度 (Rs) - 必要に応じて変更可能: 1.5, 2.0, 2.5, 3.0など
    NRHO_VAL = 30   # 動径方向格子点数
    # ======================= ここまでが修正部分 =======================
    
    try:
        # 1. HMIデータの準備
        print("\n--- Step 1: HMIデータ準備 ---")
        hmi_data = prepare_hmi_for_pfss(hmi_file)
        
        # AIA RGB画像の準備
        print("\n--- Step 1.5: AIA RGB画像準備 ---")
        target_time = "2022-06-13 03:00"
        
        # plot_sdo_aia_rgb関数を使用してRGB画像データを取得
        aia_rgb_image, aia_rgb_ref_map, aia_rgb_success = plot_sdo_aia_rgb(target_time, return_data=True)
        
        if aia_rgb_success:
            aia_rgb_data = {
                'aia_rgb_map': aia_rgb_ref_map,
                'rgb_image': aia_rgb_image
            }
        else:
            print("AIA RGB画像の準備に失敗しました")
            return
        
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
            n_seeds_x=20,
            n_seeds_y=20,
            use_strong_field=True,
            field_threshold=200
        )
        
        # 4. 磁力線をトレース
        print("\n--- Step 4: 磁力線トレース ---")
        field_lines = trace_field_lines(seeds, pfss_output)
        
        # 5. プロット作成
        print("\n--- Step 5: プロット作成 ---")
        # ======================= ここからが修正部分 =======================
        # plot関数にrssとnrhoの値を渡す
        fig = plot_hmi_with_pfss(hmi_data, aia_rgb_data, pfss_output, field_lines, rss=RSS_VAL, nrho=NRHO_VAL)
        # ======================= ここまでが修正部分 =======================
        
        print("\n=== 処理完了 ===")
        print("✓ HMI + PFSS 磁力線重ね合わせプロットが完了しました")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()