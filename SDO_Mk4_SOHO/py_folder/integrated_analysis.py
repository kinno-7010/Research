"""
統合・解析関数群
SDO/AIA、MK4、LASCO-C2データの統合解析、差分画像作成、動画生成を行う
"""

from config import *
import config
from sunpy.coordinates import get_horizons_coord
from sunpy.sun import constants as sunpy_constants
import matplotlib
# ループ処理でのスレッド問題を回避するため Agg バックエンドを使用
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def _as_path_list(value) -> list:
    if isinstance(value, (list, tuple, set)):
        return [Path(v) for v in value]
    return [Path(value)]

def build_common_reference_params(inner_params, outer_rsun):
    """
    内側観測(AIA)の px_per_rsun を保ったまま、
    outer_rsun まで含む共通キャンバスの params と extent を返す。
    extent は既存コードと同じく中心からの pixel offset。
    """
    px = inner_params['px_per_rsun']
    half_nx = int(np.ceil(outer_rsun * px))
    half_ny = int(np.ceil(outer_rsun * px))

    params = dict(
        nx=2 * half_nx + 1,
        ny=2 * half_ny + 1,
        cx=float(half_nx),
        cy=float(half_ny),
        px_per_rsun=float(px),
    )

    extent = [
        -params['cx'],
        params['nx'] - params['cx'],
        -params['cy'],
        params['ny'] - params['cy']
    ]
    return params, extent

def resample_array_between_params(data, src_params, dst_params, order=1, cval=np.nan):
    """
    src_params の画像を dst_params のグリッドへ補間する。
    """
    ny, nx = dst_params['ny'], dst_params['nx']
    y_idx, x_idx = np.indices((ny, nx))

    x_norm = (x_idx - dst_params['cx']) / dst_params['px_per_rsun']
    y_norm = (y_idx - dst_params['cy']) / dst_params['px_per_rsun']

    coords = np.vstack([
        (y_norm * src_params['px_per_rsun'] + src_params['cy']).ravel(),
        (x_norm * src_params['px_per_rsun'] + src_params['cx']).ravel()
    ])

    return map_coordinates(
        data,
        coords,
        order=order,
        mode='constant',
        cval=cval
    ).reshape((ny, nx))

def scan_multi_directories(directories, start_iso, end_iso, use_cache=True) -> list:
    combined = []
    seen_paths = set()
    for directory in directories:
        print(f"  -> {directory}")
        try:
            cand = scan_directory_for_maps(directory, start_iso, end_iso, use_cache=use_cache)
        except Exception as e:
            print(f"    Error scanning {directory}: {e}")
            continue
        for m, p in cand:
            p_str = str(p)
            if p_str in seen_paths:
                continue
            seen_paths.add(p_str)
            combined.append((m, p))
    combined.sort(key=lambda x: x[0].date)
    return combined


def get_data_list(scan_start, scan_end, use_cache=True):
    """データリスト取得（キャッシュ対応）"""    
    map_lists_dict = {}
    for key, value in data_folder_dict.items():
        paths = _as_path_list(value)
        print(f"Scanning {key}:")
        try:
            map_lists_dict[key] = scan_multi_directories(paths, scan_start.iso, scan_end.iso, use_cache=use_cache)
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
    
    mk4_list, lasco_list, aia193_list = map_lists_dict.values()
    return mk4_list, lasco_list, aia193_list


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


def determine_aia_ranges(start_time_str: str, end_time_str: str, base_time_str: str, percentile_range=[10, 90]):
    """
    指定された期間のデータから、3つのAIA波長を統合した差分画像の
    最適なグレースケール表示範囲(vmin, vmax)を自動で決定する。
    """
    print("最適なAIAグレースケール表示範囲を計算中...")
    base_time_obj = Time(base_time_str)
    
    # 解析期間全体のデータリストを取得
    aia_dir_candidates = [Path(p) for p in config.AIA_DATA_DIRS]
    aia_lists = {
        '211': scan_multi_directories([p / '211' for p in aia_dir_candidates], start_time_str, end_time_str),
        '193': scan_multi_directories([p / '193' for p in aia_dir_candidates], start_time_str, end_time_str),
        '171': scan_multi_directories([p / '171' for p in aia_dir_candidates], start_time_str, end_time_str)
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
    out_dir_str = "/mnt/d/wsl/home/kinno-7010/Research_data/SDO_Mk4_SOHO/diff"

    target_time_obj = Time(target_time_str)
    scan_start = target_time_obj - 20*u.min
    scan_end = target_time_obj + 20*u.min
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)

    aia_norm_ranges = determine_aia_ranges(
        scan_start.iso, scan_end.iso, target_time_str, percentile_range=[10, 90]
    )

    mk4_list, lasco_list, aia193_list = get_data_list(scan_start, scan_end)
    data_dict = {'mk4': mk4_list, 'lasco': lasco_list, 'aia193': aia193_list}

    # --- 既存ロジックを維持しつつ、ここでは「選択」だけ行う ---
    data_selected_dict = {}
    for key, value in data_dict.items():
        data_selected_dict[key], _ = select_by_midpoint(target_time_obj, value)

        if key == 'lasco':
            print(f"INFO: Processing full correction for LASCO target map ({data_selected_dict[key].date})...")
            data_selected_dict[key] = create_fully_corrected_lasco_map(data_selected_dict[key])

    mk4_map = data_selected_dict['mk4']
    lasco_map = data_selected_dict['lasco']
    aia193_map = data_selected_dict['aia193']

    # 既存 get_params と同じ定義をここでも使う
    def get_params_local(m):
        px = m.rsun_obs.to_value(u.arcsec) / m.scale.axis1.to_value(u.arcsec/u.pix)
        if px <= 0:
            print(f"警告: px_per_rsun が無効な値です: {px}")
            px = 1
        return dict(
            nx=m.data.shape[1],
            ny=m.data.shape[0],
            cx=m.meta['crpix1'] - 1,
            cy=m.meta['crpix2'] - 1,
            px_per_rsun=px
        )

    p_mk4 = get_params_local(mk4_map)
    p_lasco = get_params_local(lasco_map)
    p_aia = get_params_local(aia193_map)

    # --- ここが今回の本質 ---
    # AIA の px_per_rsun を保った共通キャンバスを作る
    ranges = dict(mk4_inner=1.3, mk4_outer_lasco_inner=3, lasco_outer=6.0)
    p_common, extent_common = build_common_reference_params(
        p_aia, outer_rsun=ranges['lasco_outer']
    )

    # 正規化（設定値は元コードをそのまま維持）
    mk4_vmin, mk4_vmax = -1, 3
    mk4_norm = ImageNormalize(
        mk4_map.data, vmin=mk4_vmin, vmax=mk4_vmax,
        stretch=LinearStretch(), clip=True
    )
    n_mk4 = mk4_norm(mk4_map.data)

    lasco_map_mean, lasco_map_std = np.nanmean(lasco_map.data), np.nanstd(lasco_map.data)
    lasco_vmin, lasco_vmax = lasco_map_mean, lasco_map_mean + 5*lasco_map_std
    print('lasco_vmin', lasco_vmin, 'lasco_vmax', lasco_vmax)
    lasco_vmin, lasco_vmax = 50, 350
    lasco_norm = ImageNormalize(
        lasco_map.data, vmin=lasco_vmin, vmax=lasco_vmax,
        stretch=LinearStretch(), clip=True
    )
    n_lasco = lasco_norm(lasco_map.data)

    def normalize_linear_stretch(arr, vmin, vmax):
        norm = ImageNormalize(arr, vmin=vmin, vmax=vmax, stretch=LinearStretch(), clip=True)
        return norm(arr)

    def scale01(a):
        mn, mx = np.nanmin(a), np.nanmax(a)
        return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a)

    aia193_base_map, _ = select_by_midpoint(Time("2022-06-13T02:00:00"), aia193_list)
    aia193_diff = aia193_map.data - aia193_base_map.data
    aia193_ch = normalize_linear_stretch(
        aia193_diff, vmin=aia_norm_ranges[0], vmax=aia_norm_ranges[1]
    )
    aia193_scaled = scale01(aia193_ch)
    aia193_image = aia193_scaled

    # LASCO だけ先に共通キャンバスへ補間
    n_lasco_common = resample_array_between_params(
        n_lasco, p_lasco, p_common, order=1, cval=np.nan
    )

    # 半径マップ・合成
    r_map = calculate_r_map(p_common)
    composite, imk4, ia = combine_corona_data(
        n_lasco_common, p_common,
        n_mk4, p_mk4,
        aia193_image, p_aia,
        r_map, ranges
    )

    mask_mk4 = (r_map >= ranges['mk4_inner']) & (r_map < ranges['mk4_outer_lasco_inner'])
    mask_lasco = (r_map >= ranges['mk4_outer_lasco_inner']) & (r_map <= ranges['lasco_outer'])
    combined_ml = np.full_like(composite, np.nan)
    combined_ml[mask_mk4] = imk4[mask_mk4]
    combined_ml[mask_lasco] = n_lasco_common[mask_lasco]

    # ──────────────── 描画 ────────────────
    # AIA 背景も共通キャンバスへ補間するが、基準スケールは AIA のまま
    aia193_background = resample_array_between_params(
        aia193_ch, p_aia, p_common, order=1, cval=np.nan
    )
    aia193_background[r_map > 1.3] = np.nan

    finite = np.isfinite(aia193_background)
    if np.any(finite):
        try:
            lo, hi = np.nanpercentile(aia193_background[finite], [10.0, 90.0])
            vmax = max(abs(lo), abs(hi))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = np.nanmax(np.abs(aia193_background[finite]))
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = 1e-3
            vmin = -vmax
        except Exception as e:
            print(f"AIA パーセンタイル計算エラー: {e}")
            vmin, vmax = -1e-3, 1e-3
    else:
        aia193_background = np.zeros_like(aia193_background, dtype=float)
        vmin, vmax = -1e-3, 1e-3

    im_aia = ax.imshow(
        aia193_background,
        origin='lower',
        extent=extent_common,
        cmap='gray',
        aspect='equal',
        vmin=vmin,
        vmax=vmax,
        zorder=0,
    )

    ax.imshow(
        combined_ml,
        origin='lower',
        cmap=plt.cm.plasma,
        norm=Normalize(0, 1),
        extent=extent_common,
        alpha=0.7,
        zorder=1
    )

    # 境界円（設定はそのまま、scale だけ common に変更）
    scale = p_common['px_per_rsun']
    theta = np.linspace(0, 2*np.pi, 400)

    r_limb = 1.0 * scale
    ax.plot(r_limb*np.cos(theta), r_limb*np.sin(theta),
            '--', color='red', linewidth=2.0)

    for i in range(2, int(ranges['lasco_outer'])+1):
        ax.plot(scale*i*np.cos(theta), scale*i*np.sin(theta),
                ':', color='black', linewidth=0.8)

    r1 = ranges['mk4_inner'] * scale
    ax.plot(r1*np.cos(theta), r1*np.sin(theta),
            '--', color='purple', linewidth=1.5,
            label=f"{ranges['mk4_inner']} $R_\\odot$")

    r2 = ranges['mk4_outer_lasco_inner'] * scale
    ax.plot(r2*np.cos(theta), r2*np.sin(theta),
            '--', color='cyan', linewidth=1.5,
            label=f"{ranges['mk4_outer_lasco_inner']} $R_\\odot$")

    # 既存 xlim/ylim の「物理範囲」は維持し、AIA scale へ換算
    scale_ratio = p_common['px_per_rsun'] / p_lasco['px_per_rsun']
    ax.set_xlim(-200 * scale_ratio, 0 * scale_ratio)
    ax.set_ylim(-150 * scale_ratio, 150 * scale_ratio)

    ax.set_title(
        f"SDO/AIA 193 Å: {aia193_map.date.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Mk4: {mk4_map.date.strftime('%H:%M:%S')} | LASCO-C2: {lasco_map.date.strftime('%H:%M:%S')}"
    )
    ax.legend(loc='upper right')
    


def select_by_midpoint(target_time: Time, map_list: list[tuple]):
    """時刻の中点で最適なマップを選択"""
    # 空リスト時の IndexError を防止
    if not map_list:
        raise ValueError("select_by_midpoint(): map_list が空です。データ取得結果を確認してください。")

    # map_list already sorted by m.date
    times = [m.date for m, _ in map_list]
    if len(times) == 1:
        return map_list[0]

    mids = []
    for i in range(len(times) - 1):
        delta = times[i + 1] - times[i]
        mids.append(times[i] + delta / 2)

    idx = bisect.bisect_left(mids, target_time)
    return map_list[idx]

def select_lasco_by_time_threshold(target_time: Time, map_list: list[tuple]):
    """LASCO専用：指定時間を超えてから次の画像に移り変わる選択"""
    # Ensure the list is sorted by observation time to avoid incorrect selections
    sorted_list = sorted(map_list, key=lambda mp: mp[0].date)
    times = [m.date for m, _ in sorted_list]
    if len(times) == 1:
        return sorted_list[0]

    # 指定時間以前で最も近い画像を選択（指定時間を超えるまで同じ画像を使用）
    idx = None
    for i, obs_time in enumerate(times):
        if obs_time <= target_time:
            idx = i
        else:
            break

    if idx is None:
        # すべての観測時刻が target_time より後の場合は最初のファイルを使用
        idx = 0

    print(f"LASCO画像選択: 指定時刻={target_time.iso}, 選択画像時刻={times[idx].iso}")
    return sorted_list[idx]

    
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


def determine_aia193_diff_ranges(aia193_list, base_time_str: str, percentile_range=[1, 99.9]):
    """
    AIA193差分画像の最適なグレースケール表示範囲(vmin, vmax)を自動で決定する。
    """
    print("最適なAIA193差分画像表示範囲を計算中...")
    base_time_obj = Time(base_time_str)
    
    # AIA193データリストを使用（スキャン重複を回避）
    aia_lists = {
        '193': aia193_list
    }

    # 実際の観測データから差分画像範囲を計算
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
                # 差分データから有効値のみを取得
                valid_diff = diff[np.isfinite(diff)]
                if len(valid_diff) > 0:
                    diff_values.extend(valid_diff.flatten())
            except Exception as e:
                print(f"差分計算エラー {wl}Å: {e}")
                continue
    
    if not diff_values:
        print("警告: 差分データが計算できませんでした。デフォルト範囲を使用します。")
        return (-100, 100)
    
    # パーセンタイル範囲を計算
    vmin, vmax = np.percentile(diff_values, percentile_range)
    ranges = (vmin, vmax)
    print("計算完了:", ranges)
    return ranges

# RGB画像用（現在はコメントアウト）
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
    
    base_time_str = "2022-06-13T00:48:00"
    out_dir_str = "/mnt/d/wsl/home/kinno-7010/Research_data/SDO_Mk4_SOHO/diff"
    # --- 1. 時刻パースとデータリスト取得 ---
    base_time_obj = Time(base_time_str)
    target_time_obj = Time(target_time_str)
    # LASCO用の固定ベース時刻（差分計算に必ず含める）: 00:48UTに統一
    lasco_base_time = base_time_obj
    # LASCOの02:00ベース画像を必ず含めるようスキャン範囲を拡張
    scan_start = min(target_time_obj, base_time_obj - 13*u.min, lasco_base_time)
    scan_end = max(target_time_obj, base_time_obj, lasco_base_time)
    out_dir = Path(out_dir_str)
    
    print(f"LASCO選択ロジック: 指定時間を超えてから次の画像に移り変わる方式を使用（ベース=00:48UT）")
    
    # 出力ディレクトリを作成
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # データリスト取得（スキャン重複回避のため先に実行）
    mk4_list, lasco_list, aia193_list = get_data_list(scan_start, scan_end)
    
    # AIA193差分範囲計算（既取得データを再利用）
    aia_norm_ranges = determine_aia193_diff_ranges(aia193_list, base_time_str, percentile_range=[10, 90])
    data_dict = {'mk4': mk4_list, 'lasco': lasco_list, 'aia193': aia193_list}

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
        # LASCOは指定時間を超えてから次の画像に移り変わる専用関数を使用
        if key == 'lasco':
            data_selected_dict[key], _ = select_lasco_by_time_threshold(target_time_obj, value)
            # LASCOの場合は常に00:48UT付近を基準時刻として使用
            data_selected_base_dict[key], _ = select_lasco_by_time_threshold(lasco_base_time, value)
            print(f"LASCO: 00:48UT基準時刻使用: {lasco_base_time.iso}")
            print(f"LASCO target時刻: {data_selected_dict[key].date.iso}")
            print(f"LASCO base時刻: {data_selected_base_dict[key].date.iso}")
        else:
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
        
    mk4_map, lasco_map, aia193_map = data_dict_resampled.values()
    p_mk4, p_lasco, p_aia = p_dict.values()
    mk4_diff, lasco_diff, aia193_diff = diff_dict.values()

    # 正規化（Mk4 / LASCO）
    # mk4_map_mean, mk4_map_std = np.nanmean(mk4_map.data), np.nanstd(mk4_map.data)
    # mk4_vmin, mk4_vmax = mk4_map_mean - 2*mk4_map_std, mk4_map_mean
    mk4_vmin, mk4_vmax = np.percentile(mk4_diff, [0.1, 99.9])
    mk4_vmin, mk4_vmax = -2.0, 2.0
    print('mk4_vmin', mk4_vmin, 'mk4_vmax', mk4_vmax)
    mk4_norm = ImageNormalize(mk4_diff, vmin=mk4_vmin, vmax=mk4_vmax, stretch=LinearStretch(), clip=True)
    n_mk4 = mk4_norm(mk4_diff)

    # lasco_map_mean, lasco_map_std = np.nanmean(lasco_map.data), np.nanstd(lasco_map.data)
    # lasco_vmin, lasco_vmax = lasco_map_mean - lasco_map_std, lasco_map_mean + lasco_map_std
    # lasco_vmin, lasco_vmax = np.percentile(lasco_diff, [1, 99.9])
    lasco_vmin, lasco_vmax = -8, 8
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
    
    # AIA193差分画像のみを使用
    aia193_ch = normalize_linear_stretch(aia193_diff, vmin=aia_norm_ranges[0], vmax=aia_norm_ranges[1])
    aia193_scaled = scale01(aia193_ch)

    # 単色画像として使用（グレースケール）
    aia193_image = aia193_scaled

    # 半径マップ・合成
    r_map = calculate_r_map(p_lasco)
    mk4_inner, mk4_outer_lasco_inner, lasco_outer = 1.3, 3.0, 6.0
    ranges = dict(mk4_inner=mk4_inner, mk4_outer_lasco_inner=mk4_outer_lasco_inner, lasco_outer=lasco_outer)
    composite, imk4, ia = combine_corona_data(
        n_lasco, p_lasco,
        n_mk4, p_mk4,
        aia193_image, p_aia,  # AIA193差分画像を使用
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
    
    # AIA193差分画像のみを座標変換
    aia193_background = map_coordinates(aia193_diff, coords, order=1, mode='constant', cval=np.nan).reshape((ny, nx))
    
    # 太陽半径1.3Rs以内で切り取り
    aia193_background[r_map > mk4_inner] = np.nan

    # AIA差分の表示範囲（aia_diff_plot_analysis.py 準拠）
    finite = np.isfinite(aia193_background)
    if np.any(finite):
        lo, hi = np.nanpercentile(aia193_background[finite], [10, 90])
        vmax = max(abs(lo), abs(hi))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = np.nanmax(np.abs(aia193_background[finite]))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1e-3
        vmin = -vmax
    else:
        vmin, vmax = -1.0, 1.0

    # 背景 AIA193差分画像（専用カラーマップ使用）
    try:
        # グレースケールで表示
        aia193_cmap = plt.cm.gray
    except Exception as e:
        print(f"AIA193カラーマップの読み込みに失敗: {e}")
        # フォールバック：グレースケール
        aia193_cmap = plt.cm.gray
    
    ax.imshow(
        aia193_background,
        origin='lower',
        extent=extent_global,
        cmap=aia193_cmap,
        aspect='equal',
        vmin=vmin,
        vmax=vmax,
        zorder=0
    )
    ax.imshow(combined_ml, origin='lower', cmap=plt.cm.seismic,
             norm=Normalize(0,1), extent=extent_global, alpha=0.7, zorder=1)

    # 境界円
    scale = p_lasco['px_per_rsun']
    theta = np.linspace(0, 2*np.pi, 400)
    # 太陽Limb（1 Rsun）
    r_limb = 1.0 * scale
    ax.plot(r_limb*np.cos(theta), r_limb*np.sin(theta),
            '--', color='red', linewidth=2.0)
    for i in range(2, int(ranges['lasco_outer'])+1):
        ax.plot(p_lasco['px_per_rsun']*i*np.cos(theta),
                p_lasco['px_per_rsun']*i*np.sin(theta),
                ':', color='black', linewidth=0.8)

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
    ax.set_xlim(-250, 0); ax.set_ylim(-100, 200)
    # ax.set_xlabel('X [pixel]'); ax.set_ylabel('Y [pixel]'); ax.set_facecolor('gray')
    ax.set_title(
        f"Difference | Base: {base_time_str.replace('T', ' ')} \n"
        f"SDO/AIA 193 Å: {aia193_map.date.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Mk4: {mk4_map.date.strftime('%H:%M:%S')} | LASCO-C2: {lasco_map.date.strftime('%H:%M:%S')}"
    )
    ax.legend(loc='upper right', fontsize=10)
    
    # パラメータ情報を返す（重複スキャン回避のため）
    return {
        'params_lasco': p_lasco,
        'params_mk4': p_mk4,
        'lasco_map': lasco_map,
        'mk4_map': mk4_map
    }


def create_single_diff_from_time_image(
    ax,
    target_time_str: str,
    delta_time: int,
    mk4_inner=1.3,
    mk4_outer_lasco_inner=3.0,
    lasco_outer=6.0,
    xlim_min=-250,
    xlim_max=0,
    ylim_min=-100,
    ylim_max=200
):
    """2分前ベース差分統合画像作成関数（LASCOのみ02:00固定）"""

    # sunpyの警告を抑制
    import warnings
    from sunpy.util.exceptions import SunpyMetadataWarning
    warnings.filterwarnings('ignore', category=SunpyMetadataWarning)

    target_time_obj = Time(target_time_str)

    if isinstance(delta_time, Time):
        base_time_obj = delta_time
    elif isinstance(delta_time, (int, float)):
        base_time_obj = target_time_obj - delta_time * u.min
    else:
        base_time_obj = Time(delta_time)

    # LASCOの固定ベース時刻を定義
    lasco_base_time = Time(target_time_str)

    # スキャン範囲
    time_list = [target_time_obj, base_time_obj]
    earliest_time = min(time_list)
    latest_time = max([target_time_obj, base_time_obj])
    scan_start = earliest_time - 1 * u.hour
    scan_end = latest_time + 1 * u.hour

    print(f"2分前ベース差分画像作成: target={target_time_str}, base={base_time_obj.iso}")
    print(f"LASCO用固定基準時刻: {lasco_base_time.iso}")
    print(f"LASCO選択ロジック: 指定時間を超えてから次の画像に移り変わる方式を使用")

    # データリスト取得
    mk4_list, lasco_list, aia193_list = get_data_list(scan_start, scan_end)

    # Fallback: 指定範囲にLASCOデータが見つからない場合
    if not lasco_list:
        lasco_dirs = _as_path_list(config.data_folder_dict.get('lasco', ''))
        print("警告: 指定範囲にLASCOデータが見つかりません。近傍から最近接データを探します:")
        for d in lasco_dirs:
            print(f" -> {d}")

        found = []
        try_ranges = [
            (target_time_obj - 1 * u.day, target_time_obj + 1 * u.day),
            (target_time_obj - 3 * u.day, target_time_obj + 3 * u.day),
            (Time('1900-01-01'), Time('2100-01-01'))
        ]

        for s, e in try_ranges:
            try:
                cand = scan_multi_directories(lasco_dirs, s.iso, e.iso, use_cache=False)
                if cand:
                    found = cand
                    break
            except Exception as ex:
                print(f"近傍LASCO探索エラー: {ex}")

        lasco_list = found
        if lasco_list:
            print(f"近傍LASCOデータを使用: {len(lasco_list)} files")
        else:
            raise FileNotFoundError("LASCOデータが見つかりませんでした。")

    data_dict = {
        'mk4': mk4_list,
        'lasco': lasco_list,
        'aia193': aia193_list
    }

    # LASCO基準shape確認（既存ロジック維持）
    lasco_shapes = []
    ref_map = None
    sample_size = min(3, len(lasco_list))

    for i, (_, path) in enumerate(lasco_list[:sample_size]):
        try:
            temp_map = sunpy.map.Map(path)
            lasco_shapes.append(temp_map.data.shape)
            if ref_map is None:
                ref_map = temp_map
            else:
                if temp_map.data.shape[0] * temp_map.data.shape[1] > ref_map.data.shape[0] * ref_map.data.shape[1]:
                    ref_map = temp_map
            del temp_map
            gc.collect()
        except Exception as e:
            print(f"警告: LASCOマップ読み込みエラー {path}: {e}")

    if not lasco_shapes:
        raise ValueError("利用可能なLASCOデータが見つかりません")

    max_ny = max(shape[0] for shape in lasco_shapes)
    max_nx = max(shape[1] for shape in lasco_shapes)
    global_shape = (max_ny, max_nx)
    target_wcs = ref_map.wcs

    def get_params(m):
        px = m.rsun_obs.to_value(u.arcsec) / m.scale.axis1.to_value(u.arcsec / u.pix)
        if px <= 0:
            print(f"警告: px_per_rsun が無効な値です: {px}")
            px = 1
        return dict(
            nx=m.data.shape[1],
            ny=m.data.shape[0],
            cx=m.meta['crpix1'] - 1,
            cy=m.meta['crpix2'] - 1,
            px_per_rsun=px
        )

    p_glob = get_params(ref_map)
    extent_global = [
        -p_glob['cx'],
        p_glob['nx'] - p_glob['cx'],
        -p_glob['cy'],
        p_glob['ny'] - p_glob['cy']
    ]

    def resample_map(m):
        try:
            return m.resample(global_shape * u.pix)
        except AttributeError:
            data, _ = reproject_interp((m.data, m.wcs), target_wcs, global_shape)
            return sunpy.map.Map(data, ref_map.meta)

    data_selected_dict, data_selected_base_dict = {}, {}
    data_dict_resampled, data_dict_base_resampled = {}, {}
    p_dict, diff_dict = {}, {}

    def select_by_time_threshold_single(reference_time: Time, map_list: list[tuple]):
        """
        指定時刻を超えるまで直前データを維持し、超えたら次データへ切り替える。
        """
        if not map_list:
            raise ValueError("データリストが空です。対象時刻の観測データを確認してください。")

        sorted_list = sorted(map_list, key=lambda mp: mp[0].date)
        idx = None
        for i, (m, _) in enumerate(sorted_list):
            if m.date <= reference_time:
                idx = i
            else:
                break

        if idx is None:
            idx = 0

        return sorted_list[idx][0]

    # LASCOだけは既存の resample_map を維持
    for key, value in data_dict.items():
        if not value:
            print(f"警告: {key} のデータが空のためスキップします。")
            if key == 'mk4':
                zero_data = np.zeros(global_shape, dtype=np.float64)
                zero_map = sunpy.map.Map(zero_data, ref_map.meta)
                data_dict_resampled[key] = zero_map
                data_dict_base_resampled[key] = zero_map
                p_dict[key] = get_params(zero_map)
                diff_dict[key] = zero_data
                continue
            raise ValueError(f"{key} のデータが空です。処理を継続できません。")

        data_selected_dict[key] = select_by_time_threshold_single(target_time_obj, value)

        if key == 'lasco':
            sorted_list = sorted(value, key=lambda mp: mp[0].date)
            target_idx = None
            for i, (m, _) in enumerate(sorted_list):
                if m is data_selected_dict[key] or m.date == data_selected_dict[key].date:
                    target_idx = i
                    break
            if target_idx is None:
                target_idx = 0
            base_idx = max(target_idx - 1, 0)
            data_selected_base_dict[key] = sorted_list[base_idx][0]
        else:
            data_selected_base_dict[key] = select_by_time_threshold_single(base_time_obj, value)

        print(f"{key} target時刻: {data_selected_dict[key].date.iso}")
        print(f"{key} base時刻: {data_selected_base_dict[key].date.iso}")

        if key == 'lasco':
            print(f"INFO: Processing full correction for LASCO target map ({data_selected_dict[key].date}).")
            data_selected_dict[key] = create_fully_corrected_lasco_map(data_selected_dict[key])

            print(f"INFO: Processing full correction for LASCO base map ({data_selected_base_dict[key].date}).")
            data_selected_base_dict[key] = create_fully_corrected_lasco_map(data_selected_base_dict[key])

            print("INFO: LASCO correction complete.")

        data_dict_resampled[key] = resample_map(data_selected_dict[key])
        data_dict_base_resampled[key] = resample_map(data_selected_base_dict[key])
        p_dict[key] = get_params(data_dict_resampled[key])
        diff_dict[key] = data_dict_resampled[key].data - data_dict_base_resampled[key].data

        del data_selected_dict[key], data_selected_base_dict[key]
        gc.collect()

    mk4_map, lasco_map, aia193_map = data_dict_resampled.values()
    p_mk4, p_lasco, p_aia = p_dict.values()
    mk4_diff, lasco_diff, aia193_diff = diff_dict.values()

    # ここから AIA の画質改善:
    # AIA の px_per_rsun を保った共通キャンバスを作る
    p_common, extent_common = build_common_reference_params(
        p_aia,
        outer_rsun=lasco_outer
    )

    # -----------------------------
    # K-COR / LASCO の負値を NaN にする
    # -----------------------------
    mk4_diff_plot = np.where(mk4_diff < 0, np.nan, mk4_diff)
    lasco_diff_plot = np.where(lasco_diff < 0, np.nan, lasco_diff)

    # -----------------------------
    # 0 が中央になるように各装置ごとに正規化
    # （表示用の無次元量：0 → 中央、+1 → 最大側）
    # -----------------------------
    mk4_vabs = 2.0
    lasco_vabs = 8.0
    print('mk4_vabs', mk4_vabs)
    print('lasco_vabs', lasco_vabs)

    n_mk4 = np.clip(mk4_diff_plot / mk4_vabs, 0.0, 1.0)
    n_lasco = np.clip(lasco_diff_plot / lasco_vabs, 0.0, 1.0)

    def normalize_linear_stretch(arr, vmin, vmax):
        norm = ImageNormalize(
            arr,
            vmin=vmin,
            vmax=vmax,
            stretch=LinearStretch(),
            clip=True
        )
        return norm(arr)

    def scale01(a):
        mn, mx = np.nanmin(a), np.nanmax(a)
        return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a)

    aia_norm_ranges = determine_aia193_diff_ranges(
        aia193_list,
        base_time_obj.iso,
        percentile_range=[10, 90]
    )
    aia193_ch = normalize_linear_stretch(
        aia193_diff,
        vmin=aia_norm_ranges[0],
        vmax=aia_norm_ranges[1]
    )
    aia193_scaled = scale01(aia193_ch)
    aia193_image = aia193_scaled

    # LASCO / MK4 を AIA 基準キャンバスへ補間
    n_lasco_common = resample_array_between_params(
        n_lasco,
        p_lasco,
        p_common,
        order=1,
        cval=np.nan
    )

    r_map = calculate_r_map(p_common)

    ranges = dict(
        mk4_inner=mk4_inner,
        mk4_outer_lasco_inner=mk4_outer_lasco_inner,
        lasco_outer=lasco_outer
    )

    composite, imk4, ia = combine_corona_data(
        n_lasco_common,
        p_common,
        n_mk4,
        p_mk4,
        aia193_image,
        p_aia,
        r_map,
        ranges
    )

    mask_mk4 = (r_map >= ranges['mk4_inner']) & (r_map < ranges['mk4_outer_lasco_inner'])
    mask_lasco = (r_map >= ranges['mk4_outer_lasco_inner']) & (r_map <= ranges['lasco_outer'])

    combined_ml = np.full_like(composite, np.nan)
    combined_ml[mask_mk4] = imk4[mask_mk4]
    combined_ml[mask_lasco] = n_lasco_common[mask_lasco]

    # AIA背景も AIA 基準キャンバスへ補間
    aia193_background = resample_array_between_params(
        aia193_diff,
        p_aia,
        p_common,
        order=1,
        cval=np.nan
    )
    aia193_background[r_map > mk4_inner] = np.nan

    finite = np.isfinite(aia193_background)
    if np.any(finite):
        lo, hi = np.nanpercentile(aia193_background[finite], [10, 90])
        vmax = max(abs(lo), abs(hi))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = np.nanmax(np.abs(aia193_background[finite]))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1e-3
        vmin = -vmax
    else:
        vmin, vmax = -1.0, 1.0

    try:
        aia193_cmap = plt.cm.gray
    except Exception as e:
        print(f"AIA193カラーマップの読み込みに失敗: {e}")
        aia193_cmap = plt.cm.gray

    ax.imshow(
        aia193_background,
        origin='lower',
        extent=extent_common,
        cmap=aia193_cmap,
        aspect='equal',
        vmin=vmin,
        vmax=vmax,
        zorder=0
    )

    # 0 がカラースケール中央に来るように設定
    display_norm = matplotlib.colors.TwoSlopeNorm(
        vmin=-1.0,
        vcenter=0.0,
        vmax=1.0
    )

    ax.imshow(
        combined_ml,
        origin='lower',
        cmap=plt.cm.seismic,
        norm=display_norm,
        extent=extent_common,
        alpha=0.7,
        zorder=1
    )

    # 境界円（設定はそのまま、scale だけ common に変更）
    scale = p_common['px_per_rsun']
    theta = np.linspace(0, 2 * np.pi, 400)

    r_limb = 1.0 * scale
    ax.plot(
        r_limb * np.cos(theta),
        r_limb * np.sin(theta),
        '--',
        color='red',
        linewidth=2.0
    )

    for i in range(2, int(ranges['lasco_outer']) + 1):
        ax.plot(
            scale * i * np.cos(theta),
            scale * i * np.sin(theta),
            ':',
            color='black',
            linewidth=0.8
        )

    r1 = ranges['mk4_inner'] * scale
    ax.plot(
        r1 * np.cos(theta),
        r1 * np.sin(theta),
        '--',
        color='yellow',
        linewidth=1.5,
        label=f"{ranges['mk4_inner']} $R_\\odot$"
    )

    r2 = ranges['mk4_outer_lasco_inner'] * scale
    ax.plot(
        r2 * np.cos(theta),
        r2 * np.sin(theta),
        '--',
        color='cyan',
        linewidth=1.5,
        label=f"{ranges['mk4_outer_lasco_inner']} $R_\\odot$"
    )

    # 既存の表示範囲の意味は維持し、AIA scale へ換算
    scale_ratio = p_common['px_per_rsun'] / p_lasco['px_per_rsun']
    ax.set_xlim(xlim_min * scale_ratio, xlim_max * scale_ratio)
    ax.set_ylim(ylim_min * scale_ratio, ylim_max * scale_ratio)

    title_lines = (
        f"SDO/AIA 193 Å: {aia193_map.date.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Mk4: {mk4_map.date.strftime('%H:%M:%S')} | LASCO-C2: {lasco_map.date.strftime('%H:%M:%S')}\n"
        f"Base: {base_time_obj.iso}"
    )
    ax.set_title(title_lines)
    ax.legend(loc='upper right', fontsize=12)

    return {
        'params_lasco': p_lasco,
        'params_mk4': p_mk4,
        'lasco_map': lasco_map,
        'mk4_map': mk4_map
    }