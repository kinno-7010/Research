"""
統合・解析関数群
SDO/AIA、MK4、LASCO-C2データの統合解析、差分画像作成、動画生成を行う
"""

from config import *
import config
from sunpy.coordinates import get_horizons_coord
from sunpy.sun import constants as sunpy_constants
import matplotlib
# GUIが利用可能な場合はTkAggを使用、そうでなければAggを使用
try:
    import tkinter
    matplotlib.use('TkAgg')
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def get_data_list(scan_start, scan_end, use_cache=True):
    """データリスト取得（キャッシュ対応）"""    
    map_lists_dict = {}
    for key, value in data_folder_dict.items():
        print(f"Scanning {key}: {value}")
        try:
            map_lists_dict[key] = scan_directory_for_maps(Path(value), scan_start.iso, scan_end.iso, use_cache=use_cache)
            print(f"  Found {len(map_lists_dict[key])} files for {key}")
        except Exception as e:
            print(f"  Error scanning {key}: {e}")
            map_lists_dict[key] = []
    
    # デバッグ情報表示
    empty_lists = [key for key, value in map_lists_dict.items() if not value]
    if empty_lists:
        print(f"警告: 以下のデータが見つかりませんでした: {empty_lists}")
        print("利用可能なデータで続行します...")
    
    if not any(map_lists_dict.values()):
        raise FileNotFoundError("指定範囲に必要なデータが一つも見つかりませんでした。")
    
    mk4_list, lasco_list, aia211_list, aia193_list, aia171_list = map_lists_dict.values()
    return mk4_list, lasco_list, aia211_list, aia193_list, aia171_list


def get_data_list_smart_range(time_list):
    """
    複数時刻のリストから最適な範囲を計算して一度だけスキャンする
    """
    if not time_list:
        raise ValueError("時刻リストが空です")
    
    time_objects = [Time(t) for t in time_list]
    
    # 全時刻を含む範囲を計算（前後にマージンを追加）
    earliest = min(time_objects) - 30*u.min  # 30分前
    latest = max(time_objects) + 30*u.min     # 30分後
    
    print(f"スマートスキャン範囲: {earliest.iso} - {latest.iso}")
    
    return get_data_list(earliest, latest, use_cache=True)
    

def create_fully_corrected_lasco_map(source: Union[str, Path, sunpy.map.Map]) -> sunpy.map.Map:
    """
    SOHO/LASCOのFITSファイルまたはMapオブジェクトを読み込み、輝度、角度、
    観測者位置、太陽視半径をすべて補正したSunPy Mapオブジェクトを生成する。
    """
    ### ▼▼▼ 変更点：チェック対象を sunpy.map.Map から sunpy.map.GenericMap に変更します ▼▼▼
    if isinstance(source, sunpy.map.GenericMap):
        raw_map = source
    else:
        raw_map = sunpy.map.Map(source)

    # --- 以降の関数の本体は変更ありません ---
    observer_coord = get_horizons_coord('SOHO', raw_map.date)
    new_meta = raw_map.meta.copy()
    
    new_meta['hgln_obs'] = observer_coord.lon.value
    new_meta['hglt_obs'] = observer_coord.lat.value
    dsun_obs = observer_coord.radius
    new_meta['dsun_obs'] = dsun_obs.to('m').value

    rsun_physical = sunpy_constants.get('radius')
    rsun_arc = np.arctan(rsun_physical / dsun_obs).to(u.arcsec).value
    new_meta['rsun_arc'] = rsun_arc

    exposure_time = new_meta.get('EXPTIME')
    if exposure_time and exposure_time > 0:
        normalized_data = raw_map.data.astype(np.float64) / exposure_time
        new_meta['BUNIT'] = 'DN / s'
    else:
        normalized_data = raw_map.data.astype(np.float64)

    intermediate_map = sunpy.map.Map(normalized_data, new_meta)
    final_map = intermediate_map.rotate(missing=0.0)
    
    return final_map


def combine_corona_data(data_lasco, params_lasco,
                        data_mk4, params_mk4,
                        data_aia, params_aia,
                        r_map, r_ranges):
    """
    AIA, Mk4, LASCO データを同一ラスコグリッド上で補間・マスクし、
    連続画像を返す。（修正版）
    """

    # 補間対象の画像データとパラメータを辞書にまとめます。
    # 処理の基準となるLASCOのデータは、このループに含める必要はありません。
    data_to_interp = {'mk4': data_mk4, 'aia': data_aia}
    params_to_interp = {'mk4': params_mk4, 'aia': params_aia}
    
    # 基準となるLASCOグリッドの座標を計算
    ny, nx = params_lasco['ny'], params_lasco['nx']
    y_idx, x_idx = np.indices((ny, nx))
    
    # LASCOグリッドの各点が、太陽中心からどのくらい離れているかを正規化して表現
    x_norm_on_lasco_grid = (x_idx - params_lasco['cx']) / params_lasco['px_per_rsun']
    y_norm_on_lasco_grid = (y_idx - params_lasco['cy']) / params_lasco['px_per_rsun']

    # 補間後のデータを格納するための辞書
    interp_dict = {}
    
    # MK4とAIAのデータを、LASCOのグリッドに合わせて補間するループ
    for key in data_to_interp.keys():
        data = data_to_interp[key]
        params = params_to_interp[key]
        
        # LASCOグリッド上の各点が、現在のデータ(AIAやMK4)のどのピクセル座標に対応するかを計算
        coords_in_source_image = np.vstack([
            (y_norm_on_lasco_grid * params['px_per_rsun'] + params['cy']).ravel(),
            (x_norm_on_lasco_grid * params['px_per_rsun'] + params['cx']).ravel()
        ])
        
        # 入力データが有効かチェック
        if not isinstance(data, np.ndarray) or data.ndim < 2:
             raise ValueError(f"'{key}' の画像データが有効なNumpy配列ではありません。")

        # ★★★ ここが最も重要な修正点 ★★★
        # map_coordinatesに「画像データ(data)」と「座標」を渡して補間を実行します。
        interp_data = map_coordinates(data, coords_in_source_image, order=1, mode='constant', cval=np.nan)
        
        # 結果を元のLASCOグリッドの形状に戻し、辞書に保存
        interp_dict[key] = interp_data.reshape((ny, nx))

    # 辞書から補間済みのデータを取り出す
    interp_aia = interp_dict['aia']
    interp_mk4 = interp_dict['mk4']

    # 各領域に対応するマスクを定義
    mask_aia = (r_map >= 0) & (r_map < r_ranges['mk4_inner'])
    mask_mk4 = (r_map >= r_ranges['mk4_inner']) & (r_map < r_ranges['mk4_outer_lasco_inner'])
    mask_lasco = (r_map >= r_ranges['mk4_outer_lasco_inner']) & (r_map <= r_ranges['lasco_outer'])

    # 各マスク領域に、対応するデータを貼り付けて1枚の画像を合成
    composite = np.full_like(data_lasco, np.nan)
    composite[mask_aia]   = interp_aia[mask_aia]
    composite[mask_mk4]  = interp_mk4[mask_mk4]
    composite[mask_lasco] = data_lasco[mask_lasco]  # LASCOは基準なので、元のデータをそのまま使用

    return composite, interp_mk4, interp_aia


def calculate_r_map(params):
    """
    params: dict with keys 'nx','ny','cx','cy','px_per_rsun'
    returns: 2D array of shape (ny, nx) giving radius in R_sun units
    """
    ny, nx = params['ny'], params['nx']
    # pixel indices
    y_idx, x_idx = np.indices((ny, nx))
    # normalized coordinates [R_sun]
    x_norm = (x_idx - params['cx']) / params['px_per_rsun']
    y_norm = (y_idx - params['cy']) / params['px_per_rsun']
    # radius map
    return np.sqrt(x_norm**2 + y_norm**2)


def _process_file(file_path: Path, start_time: Time, end_time: Time):
    """
    単一のファイルを処理するためのヘルパー関数（HDU構造に対応した最終版）。
    """
    try:
        if not file_path.is_file():
            return None

        # 【対策】まずHDU 1(2番目の箱)のヘッダー取得を試みる
        # SDO/AIA等の圧縮FITSは、主要情報がHDU 1にあることが多いため。
        try:
            header = fits.getheader(file_path, ext=1)
        except IndexError:
            # HDU 1 が存在しないファイル(非圧縮FITS等)の場合は、HDU 0(1番目の箱)にフォールバックする
            logging.debug(f"HDU 1 が見つかりません。HDU 0 を試します: {file_path}")
            header = fits.getheader(file_path, ext=0)
        
        # --- これ以降の時刻キー検索ロジックは前回と同じ ---
        datetime_str = None
        date_key_candidates = ['DATE-OBS', 'DATE_OBS', 'DATE']
        for key in date_key_candidates:
            if key in header:
                datetime_str = header[key]
                break

        if not datetime_str:
            logging.warning(f"日付キー {date_key_candidates} がどのHDUにも見つかりません: {file_path}")
            return None

        if 'TIME-OBS' in header:
            date_part = datetime_str.split('T')[0]
            time_part = header['TIME-OBS']
            datetime_str = f"{date_part.replace('/', '-')}T{time_part}"
        
        file_time = Time(datetime_str)
        
        if start_time <= file_time <= end_time:
            try:
                m = sunpy.map.Map(file_path)
                return (m, file_path)
            except Exception as e:
                logging.warning(f"SunPy Mapの作成に失敗: {file_path}: {e}")
                return None

    except ValueError as e:
        logging.warning(f"時刻文字列 '{datetime_str}' の解釈に失敗しました {file_path}: {e}")
        return None
    except Exception as e:
        logging.warning(f"ファイルの処理に失敗しました {file_path}: {e}")
        return None
        
    return None


def scan_directory_for_maps(directory: Path, start_time_iso: str, end_time_iso: str, max_workers: int = None, use_cache: bool = True) -> list:
    """
    並列処理を用いて超高速化したスキャン関数（キャッシュ機能付き）
    """
    # キャッシュキーを生成
    cache_key = (str(directory), start_time_iso, end_time_iso)
    
    # キャッシュから結果を取得（可能な場合）
    if use_cache and cache_key in config._global_scan_cache:
        print(f"キャッシュからデータを読み込み: {directory.name}")
        return config._global_scan_cache[cache_key]
    
    if not directory.is_dir():
        return []

    t0 = Time(start_time_iso)
    t1 = Time(end_time_iso)
    
    # ファイル検索の最適化：glob パターンを統合
    files = []
    for pattern in ['*.fits', '*.fts']:
        files.extend(directory.rglob(pattern))
    
    # ファイル数が多い場合の事前フィルタ
    if len(files) == 0:
        return []
    
    maps_with_paths = []
    
    # 並列処理の最適化：CPUコア数ベースで設定
    import os
    if max_workers is None:
        max_workers = min(os.cpu_count(), len(files), 8)  # 最大8並列に制限
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(_process_file, f, t0, t1): f for f in files}
        
        gen = as_completed(future_to_file)
        for future in tqdm(gen, total=len(files), desc=f"Scanning {directory.name}"):
            try:
                result = future.result()
                if result is not None:
                    maps_with_paths.append(result)
            except Exception as e:
                # 個別ファイルのエラーをログに記録して継続
                logging.warning(f"ファイル処理エラー: {e}")
                continue

    # メモリ効率化：ソート後にガベージコレクションを実行
    maps_with_paths.sort(key=lambda x: x[0].date)
    gc.collect()
    
    # 結果をキャッシュに保存
    if use_cache:
        config._global_scan_cache[cache_key] = maps_with_paths
        print(f"スキャン結果をキャッシュに保存: {directory.name} ({len(maps_with_paths)}ファイル)")
    
    return maps_with_paths


def clear_scan_cache():
    """スキャンキャッシュをクリアする"""
    config._global_scan_cache.clear()
    
    # GUIリソースのクリーンアップも実行
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
        import gc
        gc.collect()
    except Exception as e:
        print(f"GUIリソースクリーンアップ警告: {e}")
    
    print("スキャンキャッシュをクリアしました。")


def get_cache_info():
    """キャッシュの情報を取得"""
    cache_count = len(config._global_scan_cache)
    total_files = sum(len(files) for files in config._global_scan_cache.values())
    return f"キャッシュエントリ数: {cache_count}, 総ファイル数: {total_files}"


def determine_aia_ranges(start_time_str: str, end_time_str: str, base_time_str: str, percentile_range=[1, 99.9]):
    """
    指定された期間のデータから、3つのAIA波長を統合した差分画像の
    最適なグレースケール表示範囲(vmin, vmax)を自動で決定する。
    """
    print("最適なAIAグレースケール表示範囲を計算中...")
    base_time_obj = Time(base_time_str)
    
    # 解析期間全体のデータリストを取得
    aia_dir = "/mnt/d/wsl/home/kinno-7010/Research/SDO/AIA/Rawdata"
    aia_lists = {
        '211': scan_directory_for_maps(Path(aia_dir)/'211', start_time_str, end_time_str),
        '193': scan_directory_for_maps(Path(aia_dir)/'193', start_time_str, end_time_str),
        '171': scan_directory_for_maps(Path(aia_dir)/'171', start_time_str, end_time_str)
    }

    # メモリ効率化：サンプリングを使用してデータサイズを削減
    sample_values = []
    max_samples_per_image = 5000  # 各画像から最大サンプル数
    
    for wl, map_list in aia_lists.items():
        if not map_list:
            print(f"警告: {wl}Å のデータが見つかりません。スキップします。")
            continue
        
        # メモリ効率化：最大3個のマップのみ使用
        limited_map_list = map_list[:3] if len(map_list) > 3 else map_list
        
        for aia_map, _ in tqdm(limited_map_list, desc=f"AIA {wl}Å (sampled)"):
            try:
                # ランダムサンプリングでメモリ使用量を削減
                flat_data = aia_map.data.ravel()
                if len(flat_data) > max_samples_per_image:
                    indices = np.random.choice(len(flat_data), max_samples_per_image, replace=False)
                    sampled_data = flat_data[indices]
                else:
                    sampled_data = flat_data
                
                sample_values.append(sampled_data)
                
                # メモリ解放
                del flat_data, sampled_data
                gc.collect()
                
            except Exception as e:
                print(f"警告: {wl}Å データ処理中にエラー: {e}")
                continue

    if not sample_values:
        print("警告: サンプルデータが見つかりません。デフォルト値を使用します。")
        return (100.0, 3000.0)

    # サンプル値からパーセンタイルを計算
    all_samples = np.concatenate(sample_values)
    vmin, vmax = np.percentile(all_samples, percentile_range)
    
    # メモリ解放
    del sample_values, all_samples
    gc.collect()

    # 以前の辞書ではなく、(vmin, vmax) のタプルを返す
    ranges = (vmin, vmax)
    
    print("計算完了:", ranges)
    return ranges


def create_single_integrated_image(ax, target_time_str: str):
    """統合画像作成関数"""
    out_dir_str = "/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/diff"
    # --- 1. 時刻パースとデータリスト取得 (変更なし) ---
    target_time_obj = Time(target_time_str)
    scan_start = target_time_obj - 20*u.min
    scan_end = target_time_obj + 20*u.min
    out_dir = Path(out_dir_str)
    
    # 出力ディレクトリを作成
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # matplotlibの警告を抑制
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
    
    aia_norm_ranges = determine_aia_ranges(scan_start.iso, scan_end.iso, target_time_str, percentile_range=[1, 99.9])

    # データリスト取得
    mk4_list, lasco_list, aia211_list, aia193_list, aia171_list = get_data_list(scan_start, scan_end)
    data_dict = {'mk4': mk4_list, 'lasco': lasco_list, 'aia211': aia211_list, 'aia193': aia193_list, 'aia171': aia171_list}

    # LASCO マップのサンプルを読み込んで最大shapeを特定（メモリ効率化）
    lasco_shapes = []
    ref_map = None
    
    # 最初の数個のマップのみでshapeを確認
    sample_size = min(3, len(lasco_list))
    for i, (_, path) in enumerate(lasco_list[:sample_size]):
        try:
            temp_map = sunpy.map.Map(path)
            lasco_shapes.append(temp_map.data.shape)
            if ref_map is None:
                ref_map = temp_map
            else:
                # より大きいshapeを基準にする
                if temp_map.data.shape[0] * temp_map.data.shape[1] > ref_map.data.shape[0] * ref_map.data.shape[1]:
                    ref_map = temp_map
            # メモリ解放
            del temp_map
            gc.collect()
        except Exception as e:
            print(f"警告: LASCOマップ読み込みエラー {path}: {e}")
    
    if not lasco_shapes:
        raise ValueError("利用可能なLASCOデータが見つかりません")
    
    # 最大shapeを決定
    max_ny = max(shape[0] for shape in lasco_shapes)
    max_nx = max(shape[1] for shape in lasco_shapes)
    global_shape = (max_ny, max_nx)
    target_wcs = ref_map.wcs

    # グローバル extent 固定
    def get_params(m):
        px = m.rsun_obs.to_value(u.arcsec) / m.scale.axis1.to_value(u.arcsec/u.pix)
        # ゼロ除算を回避
        if px <= 0:
            print(f"警告: px_per_rsun が無効な値です: {px}")
            px = 1  # デフォルト値を設定
        return dict(nx=m.data.shape[1],
                   ny=m.data.shape[0],
                   cx=m.meta['crpix1']-1,
                   cy=m.meta['crpix2']-1,
                   px_per_rsun=px)

    p_glob = get_params(ref_map)
    extent_global = [
        -p_glob['cx'], p_glob['nx'] - p_glob['cx'],
        -p_glob['cy'], p_glob['ny'] - p_glob['cy']
    ]

    # フレームごとに処理
    def resample_map(m):
        try:
            return m.resample(global_shape * u.pix)
        except AttributeError:
            data, _ = reproject_interp((m.data, m.wcs), target_wcs, global_shape)
            return sunpy.map.Map(data, ref_map.meta)
        
    data_selected_dict, data_dict_resampled, p_dict = {}, {}, {}
    # 各波長・Instrument のマップ取得
    for key, value in data_dict.items():
        data_selected_dict[key], _ = select_by_midpoint(target_time_obj, value)
        
        if key == 'lasco':
            print(f"INFO: Processing full correction for LASCO target map ({data_selected_dict[key].date})...")
            data_selected_dict[key] = create_fully_corrected_lasco_map(data_selected_dict[key])
            
        data_dict_resampled[key] = resample_map(data_selected_dict[key])
        
        p_dict[key] = get_params(data_dict_resampled[key])
        
        # メモリ効率化：元のマップは処理後に削除
        del data_selected_dict[key]
        gc.collect()
        
    mk4_map, lasco_map, aia211_map, aia193_map, aia171_map = data_dict_resampled.values()
    p_mk4, p_lasco, p_aia, _, _ = p_dict.values()

    # 正規化（Mk4 / LASCO）
    # mk4_map_mean, mk4_map_std = np.nanmean(mk4_map.data), np.nanstd(mk4_map.data)
    # mk4_vmin, mk4_vmax = mk4_map_mean, mk4_map_mean + 1*mk4_map_std
    mk4_vmin, mk4_vmax = -1, 4
    mk4_norm = ImageNormalize(mk4_map.data, vmin=mk4_vmin, vmax=mk4_vmax, stretch=LinearStretch(), clip=True)
    n_mk4 = mk4_norm(mk4_map.data)


    lasco_map_mean, lasco_map_std = np.nanmean(lasco_map.data), np.nanstd(lasco_map.data)
    lasco_vmin, lasco_vmax = lasco_map_mean, lasco_map_mean + 5*lasco_map_std
    print('lasco_vmin', lasco_vmin, 'lasco_vmax', lasco_vmax)
    # lasco_vmin, lasco_vmax = np.nanpercentile(lasco_map.data, [1, 99])
    lasco_vmin, lasco_vmax = 60, 380
    lasco_norm = ImageNormalize(lasco_map.data, vmin=lasco_vmin, vmax=lasco_vmax, stretch=LinearStretch(), clip=True)
    n_lasco = lasco_norm(lasco_map.data)

    # AIA 193差分画像用正規化
    def normalize_linear_stretch(arr, vmin, vmax):
        norm = ImageNormalize(arr, vmin=vmin, vmax=vmax, stretch=LinearStretch(), clip=True)
        return norm(arr)

    def scale01(a):
        mn, mx = np.nanmin(a), np.nanmax(a)
        return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a)
    
    # AIA 193差分画像のみを使用（ベース画像を取得して差分計算）
    aia193_base_map, _ = select_by_midpoint(Time("2022-06-13T02:00:00"), aia193_list)
    aia193_diff = aia193_map.data - aia193_base_map.data
    
    aia193_ch = normalize_linear_stretch(aia193_diff, vmin=aia_norm_ranges[0], vmax=aia_norm_ranges[1])
    aia193_scaled = scale01(aia193_ch)

    # 単色差分画像として使用
    aia193_image = aia193_scaled

    # 半径マップ・合成
    r_map = calculate_r_map(p_lasco)
    ranges = dict(mk4_inner=1.1, mk4_outer_lasco_inner=3, lasco_outer=6.0)
    composite, imk4, ia = combine_corona_data(
        n_lasco, p_lasco,
        n_mk4, p_mk4,
        aia193_image, p_aia,  # AIA 193差分画像を使用
        r_map, ranges
    )
    # MK4/LASCO 合成
    mask_mk4 = (r_map >= ranges['mk4_inner']) & (r_map < ranges['mk4_outer_lasco_inner'])
    mask_lasco = (r_map >= ranges['mk4_outer_lasco_inner']) & (r_map <= ranges['lasco_outer'])
    combined_ml = np.full_like(composite, np.nan)
    combined_ml[mask_mk4] = imk4[mask_mk4]
    combined_ml[mask_lasco] = n_lasco[mask_lasco]

    # ──────────────── 描画 ────────────────
    # fig, ax = plt.subplots(figsize=(6, 10))

    # 1) AIA 193画像を背景として変換
    ny, nx = p_lasco['ny'], p_lasco['nx']
    y_idx, x_idx = np.indices((ny, nx))
    x_norm = (x_idx - p_lasco['cx']) / p_lasco['px_per_rsun']
    y_norm = (y_idx - p_lasco['cy']) / p_lasco['px_per_rsun']
    coords = np.vstack([
        (y_norm * p_aia['px_per_rsun'] + p_aia['cy']).ravel(),
        (x_norm * p_aia['px_per_rsun'] + p_aia['cx']).ravel()
    ])
    
    # AIA 193画像のみを座標変換
    aia193_background = map_coordinates(aia193_ch, coords, order=1, mode='constant', cval=np.nan).reshape((ny, nx))
    
    # 太陽半径1.1Rs以内で切り取り
    aia193_background[r_map > 1.1] = np.nan

    # 背景 AIA 193差分画像（専用カラーマップ使用）
    try:
        # sunpyのAIA 193専用カラーマップを使用
        import sunpy.visualization.colormaps as cm
        aia193_cmap = cm.cm.sdoaia193
    except Exception as e:
        print(f"AIA 193カラーマップの読み込みに失敗: {e}")
        # フォールバック：標準カラーマップ
        aia193_cmap = plt.cm.plasma
    
    ax.imshow(aia193_background, origin='lower', extent=extent_global, cmap=aia193_cmap, aspect='equal', zorder=0)
    ax.imshow(combined_ml, origin='lower', cmap=plt.cm.plasma,
             norm=Normalize(0,1), extent=extent_global, alpha=0.7, zorder=1)

    # 境界円
    scale = p_lasco['px_per_rsun']
    theta = np.linspace(0, 2*np.pi, 400)
    for i in range(2, int(ranges['lasco_outer'])+1):
        ax.plot(p_lasco['px_per_rsun']*i*np.cos(theta),
                p_lasco['px_per_rsun']*i*np.sin(theta),
                ':', color='white', linewidth=0.8)

    # 境界円＆凡例
    r1 = ranges['mk4_inner']*scale
    ax.plot(r1*np.cos(theta), r1*np.sin(theta),
            '--',color='yellow',linewidth=1.5,
            label=f"{ranges['mk4_inner']} $R_\\odot$")
    r2 = ranges['mk4_outer_lasco_inner']*scale
    ax.plot(r2*np.cos(theta), r2*np.sin(theta),
            '--',color='cyan',linewidth=1.5,
            label=f"{ranges['mk4_outer_lasco_inner']} $R_\\odot$")

    # 軸範囲を global に固定
    ax.set_xlim(-400, 0); ax.set_ylim(-200, 300)
    # ax.set_xlabel('X [pixel]'); ax.set_ylabel('Y [pixel]'); ax.set_facecolor('gray')
    ax.set_title(
        f"SDO/AIA 211+193+171 Å: {aia211_map.date.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Mk4: {mk4_map.date.strftime('%H:%M:%S')} | LASCO-C2: {lasco_map.date.strftime('%H:%M:%S')}"
    )
    ax.legend(loc='best')


def select_by_midpoint(target_time: Time, map_list: list[tuple]):
    """時刻の中点で最適なマップを選択"""
    # map_list already sorted by m.date
    times = [m.date for m, _ in map_list]
    if len(times) == 1:
        return map_list[0]
    mids = []
    for i in range(len(times)-1):
        delta = times[i+1] - times[i]
        mids.append(times[i] + delta/2)
    idx = bisect.bisect_left(mids, target_time)
    return map_list[idx]

    
def normalize_linear_stretch_integrated(arr, vmin, vmax):
    """AIA RGB 合成用正規化"""
    norm = ImageNormalize(arr, vmin=vmin, vmax=vmax, stretch=LinearStretch(), clip=True)
    return norm(arr)


def get_params(m):
    """マップパラメータ取得"""
    px = m.rsun_obs.to_value(u.arcsec) / m.scale.axis1.to_value(u.arcsec/u.pix)
    return dict(nx=m.data.shape[1],
                ny=m.data.shape[0],
                cx=m.meta['crpix1']-1,
                cy=m.meta['crpix2']-1,
                px_per_rsun=px)


def scale01(a):
    """0-1スケーリング"""
    mn, mx = np.nanmin(a), np.nanmax(a)
    return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a)


def determine_aia_diff_ranges(aia211_list, aia193_list, aia171_list, base_time_str: str, percentile_range=[1, 99.9]):
    """
    指定された期間のデータから、3つのAIA波長を統合した差分画像の
    最適なグレースケール表示範囲(vmin, vmax)を自動で決定する。（実データ使用版）
    """
    print("最適なAIAグレースケール表示範囲を計算中...")
    base_time_obj = Time(base_time_str)
    
    # 既に取得済みのデータリストを使用（スキャン重複を回避）
    aia_lists = {
        '211': aia211_list,
        '193': aia193_list,
        '171': aia171_list
    }

    # 実際の観測データから差分画像範囲を計算（サンプリング無効化）
    diff_values = []
    
    for wl, map_list in aia_lists.items():
        if not map_list:
            print(f"警告: {wl}Å のデータが見つかりません。スキップします。")
            continue
        
        base_map, _ = select_by_midpoint(base_time_obj, map_list)
        
        # 実際のデータを使用（最大3個のマップで計算効率化）
        limited_map_list = map_list[:3] if len(map_list) > 3 else map_list
        
        for aia_map, _ in tqdm(limited_map_list, desc=f"AIA {wl}Å diffs (full data)"):
            try:
                diff = aia_map.data - base_map.data
                # 全データを使用（サンプリングなし）
                valid_diff = diff[np.isfinite(diff)]
                if len(valid_diff) > 0:
                    diff_values.append(valid_diff)
                
                # メモリ解放
                del diff, valid_diff
                gc.collect()
                
            except Exception as e:
                print(f"警告: {wl}Å データ処理中にエラー: {e}")
                continue

    if not diff_values:
        print("警告: 差分データが見つかりません。デフォルト値を使用します。")
        return (-200.0, 600.0)

    # 全差分データからパーセンタイルを計算
    all_diffs = np.concatenate(diff_values)
    vmin, vmax = np.percentile(all_diffs, percentile_range)
    
    # メモリ解放
    del diff_values, all_diffs
    gc.collect()

    ranges = (vmin, vmax)
    print("計算完了:", ranges)
    return ranges


def create_single_diff_image(ax, target_time_str: str):
    """差分統合画像作成関数"""
    
    # sunpyの警告を抑制
    import warnings
    from sunpy.util.exceptions import SunpyMetadataWarning
    warnings.filterwarnings('ignore', category=SunpyMetadataWarning)
    
    base_time_str = "2022-06-13T02:00:00"
    out_dir_str = "/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/diff"
    # --- 1. 時刻パースとデータリスト取得 (変更なし) ---
    base_time_obj = Time(base_time_str)
    target_time_obj = Time(target_time_str)
    scan_start = min(target_time_obj, base_time_obj)
    scan_end = max(target_time_obj, base_time_obj)
    out_dir = Path(out_dir_str)
    
    # 出力ディレクトリを作成
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # データリスト取得（スキャン重複回避のため先に実行）
    mk4_list, lasco_list, aia211_list, aia193_list, aia171_list = get_data_list(scan_start, scan_end)
    
    # AIA差分範囲計算（既取得データを再利用）
    aia_norm_ranges = determine_aia_diff_ranges(aia211_list, aia193_list, aia171_list, base_time_str, percentile_range=[1, 99.9])
    data_dict = {'mk4': mk4_list, 'lasco': lasco_list, 'aia211': aia211_list, 'aia193': aia193_list, 'aia171': aia171_list}

    # LASCO マップのサンプルを読み込んで最大shapeを特定（メモリ効率化）
    lasco_shapes = []
    ref_map = None
    
    # 最初の数個のマップのみでshapeを確認
    sample_size = min(3, len(lasco_list))
    for i, (_, path) in enumerate(lasco_list[:sample_size]):
        try:
            temp_map = sunpy.map.Map(path)
            lasco_shapes.append(temp_map.data.shape)
            if ref_map is None:
                ref_map = temp_map
            else:
                # より大きいshapeを基準にする
                if temp_map.data.shape[0] * temp_map.data.shape[1] > ref_map.data.shape[0] * ref_map.data.shape[1]:
                    ref_map = temp_map
            # メモリ解放
            del temp_map
            gc.collect()
        except Exception as e:
            print(f"警告: LASCOマップ読み込みエラー {path}: {e}")
    
    if not lasco_shapes:
        raise ValueError("利用可能なLASCOデータが見つかりません")
    
    # 最大shapeを決定
    max_ny = max(shape[0] for shape in lasco_shapes)
    max_nx = max(shape[1] for shape in lasco_shapes)
    global_shape = (max_ny, max_nx)
    target_wcs = ref_map.wcs

    # グローバル extent 固定
    def get_params(m):
        px = m.rsun_obs.to_value(u.arcsec) / m.scale.axis1.to_value(u.arcsec/u.pix)
        # ゼロ除算を回避
        if px <= 0:
            print(f"警告: px_per_rsun が無効な値です: {px}")
            px = 1  # デフォルト値を設定
        return dict(nx=m.data.shape[1],
                   ny=m.data.shape[0],
                   cx=m.meta['crpix1']-1,
                   cy=m.meta['crpix2']-1,
                   px_per_rsun=px)

    p_glob = get_params(ref_map)
    extent_global = [
        -p_glob['cx'], p_glob['nx'] - p_glob['cx'],
        -p_glob['cy'], p_glob['ny'] - p_glob['cy']
    ]

    # フレームごとに処理
    def resample_map(m):
        try:
            return m.resample(global_shape * u.pix)
        except AttributeError:
            data, _ = reproject_interp((m.data, m.wcs), target_wcs, global_shape)
            return sunpy.map.Map(data, ref_map.meta)
        
    data_selected_dict, data_selected_base_dict, data_dict_resampled, data_dict_base_resampled, p_dict, diff_dict = {}, {}, {}, {}, {}, {}
    # 各波長・Instrument のマップ取得
    for key, value in data_dict.items():
        data_selected_dict[key], _ = select_by_midpoint(target_time_obj, value)
        data_selected_base_dict[key], _ = select_by_midpoint(base_time_obj, value)
        
        if key == 'lasco':
            print(f"INFO: Processing full correction for LASCO target map ({data_selected_dict[key].date})...")
            data_selected_dict[key] = create_fully_corrected_lasco_map(data_selected_dict[key])
            
            print(f"INFO: Processing full correction for LASCO base map ({data_selected_base_dict[key].date})...")
            data_selected_base_dict[key] = create_fully_corrected_lasco_map(data_selected_base_dict[key])
            print("INFO: LASCO correction complete.")
            
        data_dict_resampled[key] = resample_map(data_selected_dict[key])
        data_dict_base_resampled[key] = resample_map(data_selected_base_dict[key])
        
        p_dict[key] = get_params(data_dict_resampled[key])
        diff_dict[key] = data_dict_resampled[key].data - data_dict_base_resampled[key].data
        
        # メモリ効率化：元のマップは差分計算後に削除
        del data_selected_dict[key], data_selected_base_dict[key]
        gc.collect()
        
    mk4_map, lasco_map, aia211_map, _, _ = data_dict_resampled.values()
    p_mk4, p_lasco, p_aia, _, _ = p_dict.values()
    mk4_diff, lasco_diff, aia211_diff, aia193_diff, aia171_diff = diff_dict.values()

    # 正規化（Mk4 / LASCO）
    # mk4_map_mean, mk4_map_std = np.nanmean(mk4_map.data), np.nanstd(mk4_map.data)
    # mk4_vmin, mk4_vmax = mk4_map_mean - 2*mk4_map_std, mk4_map_mean
    mk4_vmin, mk4_vmax = np.percentile(mk4_diff, [0.1, 99.9])
    mk4_vmin, mk4_vmax = -2, 1.5
    print('mk4_vmin', mk4_vmin, 'mk4_vmax', mk4_vmax)
    mk4_norm = ImageNormalize(mk4_diff, vmin=mk4_vmin, vmax=mk4_vmax, stretch=LinearStretch(), clip=True)
    n_mk4 = mk4_norm(mk4_diff)

    # lasco_map_mean, lasco_map_std = np.nanmean(lasco_map.data), np.nanstd(lasco_map.data)
    # lasco_vmin, lasco_vmax = lasco_map_mean - lasco_map_std, lasco_map_mean + lasco_map_std
    # lasco_vmin, lasco_vmax = np.percentile(lasco_diff, [1, 99.9])
    lasco_vmin, lasco_vmax = -5, 50
    print('lasco_vmin', lasco_vmin, 'lasco_vmax', lasco_vmax)
    
    lasco_norm = ImageNormalize(lasco_diff, vmin=lasco_vmin, vmax=lasco_vmax, stretch=LinearStretch(), clip=True)
    n_lasco = lasco_norm(lasco_diff)

    # AIA 193 差分画像用正規化
    def normalize_linear_stretch(arr, vmin, vmax):
        norm = ImageNormalize(arr, vmin=vmin, vmax=vmax, stretch=LinearStretch(), clip=True)
        return norm(arr)

    def scale01(a):
        mn, mx = np.nanmin(a), np.nanmax(a)
        return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a)
    
    # AIA 193差分画像のみを使用
    aia193_ch = normalize_linear_stretch(aia193_diff, vmin=aia_norm_ranges[0], vmax=aia_norm_ranges[1])
    aia193_scaled = scale01(aia193_ch)

    # 単色画像として使用（グレースケール）
    aia193_image = aia193_scaled

    # 半径マップ・合成
    r_map = calculate_r_map(p_lasco)
    ranges = dict(mk4_inner=1.1, mk4_outer_lasco_inner=3, lasco_outer=6.0)
    composite, imk4, ia = combine_corona_data(
        n_lasco, p_lasco,
        n_mk4, p_mk4,
        aia193_image, p_aia,  # AIA 193差分画像を使用
        r_map, ranges
    )
    # MK4/LASCO 合成
    mask_mk4 = (r_map >= ranges['mk4_inner']) & (r_map < ranges['mk4_outer_lasco_inner'])
    mask_lasco = (r_map >= ranges['mk4_outer_lasco_inner']) & (r_map <= ranges['lasco_outer'])
    combined_ml = np.full_like(composite, np.nan)
    combined_ml[mask_mk4] = imk4[mask_mk4]
    combined_ml[mask_lasco] = n_lasco[mask_lasco]

    # ──────────────── 描画 ────────────────
    # fig, ax = plt.subplots(figsize=(6, 10))

    # 1) AIA 193差分画像を背景として変換
    ny, nx = p_lasco['ny'], p_lasco['nx']
    y_idx, x_idx = np.indices((ny, nx))
    x_norm = (x_idx - p_lasco['cx']) / p_lasco['px_per_rsun']
    y_norm = (y_idx - p_lasco['cy']) / p_lasco['px_per_rsun']
    coords = np.vstack([
        (y_norm * p_aia['px_per_rsun'] + p_aia['cy']).ravel(),
        (x_norm * p_aia['px_per_rsun'] + p_aia['cx']).ravel()
    ])
    
    # AIA 193差分画像のみを座標変換
    aia193_background = map_coordinates(aia193_ch, coords, order=1, mode='constant', cval=np.nan).reshape((ny, nx))
    
    # 太陽半径1.1Rs以内で切り取り
    aia193_background[r_map > 1.1] = np.nan

    # 背景 AIA 193差分画像（専用カラーマップ使用）
    try:
        # sunpyのAIA 193専用カラーマップを使用
        import sunpy.visualization.colormaps as cm
        aia193_cmap = cm.cm.sdoaia193
    except Exception as e:
        print(f"AIA 193カラーマップの読み込みに失敗: {e}")
        # フォールバック：標準カラーマップ
        aia193_cmap = plt.cm.plasma
    
    ax.imshow(aia193_background, origin='lower', extent=extent_global, cmap=aia193_cmap, aspect='equal', zorder=0)
    ax.imshow(combined_ml, origin='lower', cmap=plt.cm.seismic,
             norm=Normalize(0,1), extent=extent_global, alpha=0.7, zorder=1)

    # 境界円
    scale = p_lasco['px_per_rsun']
    theta = np.linspace(0, 2*np.pi, 400)
    for i in range(2, int(ranges['lasco_outer'])+1):
        ax.plot(p_lasco['px_per_rsun']*i*np.cos(theta),
                p_lasco['px_per_rsun']*i*np.sin(theta),
                ':', color='white', linewidth=0.8)

    # 境界円＆凡例
    r1 = ranges['mk4_inner']*scale
    ax.plot(r1*np.cos(theta), r1*np.sin(theta),
            '--',color='yellow',linewidth=1.5,
            label=f"{ranges['mk4_inner']} $R_\\odot$")
    r2 = ranges['mk4_outer_lasco_inner']*scale
    ax.plot(r2*np.cos(theta), r2*np.sin(theta),
            '--',color='cyan',linewidth=1.5,
            label=f"{ranges['mk4_outer_lasco_inner']} $R_\\odot$")

    # 軸範囲を global に固定
    ax.set_xlim(-400, 0); ax.set_ylim(-200, 300)
    # ax.set_xlabel('X [pixel]'); ax.set_ylabel('Y [pixel]'); ax.set_facecolor('gray')
    ax.set_title(
        f"Difference | Base: {base_time_str.replace('T', ' ')} \n"
        f"SDO/AIA 211+193+171 Å: {aia211_map.date.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Mk4: {mk4_map.date.strftime('%H:%M:%S')} | LASCO-C2: {lasco_map.date.strftime('%H:%M:%S')}"
    )
    ax.legend(loc='best')
    
    # パラメータ情報を返す（重複スキャン回避のため）
    return {
        'params_lasco': p_lasco,
        'params_mk4': p_mk4,
        'lasco_map': lasco_map,
        'mk4_map': mk4_map
    }