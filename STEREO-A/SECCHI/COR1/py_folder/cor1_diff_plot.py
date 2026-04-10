#!/usr/bin/env python3
"""
STEREO-A/SECCHI/COR1 差分画像プロッター (改良版)
0を確実に白で表示するための修正を加えたバージョン
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import sunpy.map
import warnings
from datetime import datetime, timedelta

# 新しい校正システムのインポート
from stereo_cor1_calibration import STEREOCOR1Calibration, STEREOCoordinates
from stereo_coordinates import STEREOCoordinateTransform

# カスタムモジュールのインポート
try:
    from cor_prep import CORPrep
except ImportError:
    print("エラー: 'cor_prep.py' が同じディレクトリに存在することを確認してください。")
    exit()

warnings.filterwarnings('ignore', category=sunpy.map.mapbase.SunpyUserWarning)

BASE_TIME_FILE = "/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/Rawdata/20220613_020136_n4c1A.fts"
RAWDATA_DIR = "/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/Rawdata"


def get_solar_radius_arcsec_and_pixel(header):
    """
    FITSヘッダーから太陽半径を arcsec と pixel で取得する。
    COR1 では RSUN は通常 arcsec とみなす。
    """
    if 'RSUN' in header and np.isfinite(float(header['RSUN'])) and float(header['RSUN']) > 0:
        rsun_arcsec = float(header['RSUN'])
    elif 'RSUN_OBS' in header and np.isfinite(float(header['RSUN_OBS'])) and float(header['RSUN_OBS']) > 0:
        rsun_arcsec = float(header['RSUN_OBS'])
    else:
        # フォールバック: 標準太陽半径
        rsun_arcsec = 959.63

    cdelt_values = []
    if 'CDELT1' in header:
        cdelt_values.append(abs(float(header['CDELT1'])))
    if 'CDELT2' in header:
        cdelt_values.append(abs(float(header['CDELT2'])))

    if len(cdelt_values) == 0:
        cdelt_arcsec_pix = 1.0
    else:
        cdelt_arcsec_pix = float(np.mean(cdelt_values))

    rsun_pix = rsun_arcsec / cdelt_arcsec_pix
    return rsun_arcsec, rsun_pix

def analyze_data_distribution(data):
    """
    データの分布を詳細に分析し、適切なスケーリングパラメータを決定する

    Parameters:
    -----------
    data : numpy.ndarray
        分析対象のデータ配列

    Returns:
    --------
    dict : 分析結果を含む辞書
    """
    valid_data = data[np.isfinite(data)]

    if valid_data.size == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'valid_count': 0,
            'percentiles': {
                1: np.nan,
                5: np.nan,
                95: np.nan,
                99: np.nan
            },
            'symmetric_range': np.nan,
            'skewness': np.nan
        }

    stats = {
        'mean': np.mean(valid_data),
        'median': np.median(valid_data),
        'std': np.std(valid_data),
        'min': np.min(valid_data),
        'max': np.max(valid_data),
        'valid_count': valid_data.size,
        'percentiles': {
            1: np.percentile(valid_data, 1),
            5: np.percentile(valid_data, 5),
            95: np.percentile(valid_data, 95),
            99: np.percentile(valid_data, 99)
        }
    }

    abs_max = max(abs(stats['percentiles'][1]), abs(stats['percentiles'][99]))
    stats['symmetric_range'] = abs_max
    stats['skewness'] = stats['mean'] / stats['std'] if stats['std'] > 0 else 0

    return stats

def create_symmetric_colormap_norm(data, method='percentile', percentile=99, min_range=5):
    """
    0を中心とした対称的なカラーマップ正規化を作成

    Parameters:
    -----------
    data : numpy.ndarray
        正規化対象のデータ
    method : str
        正規化方法 ('percentile', 'std', 'fixed')
    percentile : float
        パーセンタイル法で使用するパーセンタイル値
    min_range : float
        最小範囲

    Returns:
    --------
    TwoSlopeNorm : 0を中心とした正規化オブジェクト
    """
    valid_data = data[np.isfinite(data)]
    if valid_data.size == 0:
        raise ValueError("有効画素が存在しないため、カラーマップ正規化を作成できません。")

    stats = analyze_data_distribution(data)

    if method == 'percentile':
        vmax = np.percentile(np.abs(valid_data), percentile)
        vmin = -vmax
    elif method == 'std':
        vmax = 3 * stats['std']
        vmin = -vmax
    elif method == 'fixed':
        vmax = 5
        vmin = -5
    else:
        raise ValueError(f"Unknown method: {method}")

    if vmax < min_range:
        vmax = min_range
        vmin = -min_range

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    print(f"カラースケーリング設定 ({method}法):")
    print(f"  vmin={vmin:.2f}, vcenter=0.00, vmax={vmax:.2f}")
    print(f"  有効画素数={stats['valid_count']}, 平均={stats['mean']:.3f}, 中央値={stats['median']:.3f}")

    return norm, vmin, vmax

def load_and_preprocess_data(fits_filepath):
    """
    COR1 FITSファイルを読み込み、前処理を行う関数（新校正システム使用）

    Parameters:
    -----------
    fits_filepath : str
        読み込むFITSファイルのパス

    Returns:
    --------
    processed_data : numpy.ndarray or None
        前処理済みの画像データ
    processed_header : astropy.io.fits.Header or None
        処理済みのヘッダー情報
    """
    if not os.path.exists(fits_filepath):
        print(f"エラー: 指定されたFITSファイルが見つかりません: {fits_filepath}")
        return None, None

    print(f"データ読み込み・前処理: {os.path.basename(fits_filepath)}")

    try:
        # 旧システム（CORPrep）を直接使用
        preprocessor = CORPrep(silent=True)
        processed_data, processed_header_dict = preprocessor.cor_prep(
            filepath=fits_filepath,
            rotate_on=False,  # CORPrepでの回転を無効化（差分データレベルで統一的に回転）
            smask_on=True,
            calibrate_off=False,
            discri_pobj_on=True
        )
        
        if processed_data is None:
            print(f"エラー: {fits_filepath} の前処理に失敗しました。")
            return None, None

        # ヘッダー処理
        history_list = processed_header_dict.pop('HISTORY', [])
        processed_header = fits.Header(processed_header_dict)
        if history_list:
            for history_entry in history_list:
                sanitized_lines = str(history_entry).split('\n')
                for line in sanitized_lines:
                    clean_line = line.strip()
                    if clean_line:
                        ascii_line = ''.join(c for c in clean_line if ord(c) < 128)
                        processed_header.add_history(ascii_line)
        
        # デバッグ用データ統計
        print(f"Debug: Processed data shape={processed_data.shape}, min={np.nanmin(processed_data):.3f}, max={np.nanmax(processed_data):.3f}")
        # 回転角の確認
        crota_angle = STEREOCoordinateTransform.get_crota_angle(processed_header)
        print(f"Debug: CORPrep後の回転角={crota_angle:.3f}度")
        print(f"✓ CORPrep前処理完了: {os.path.basename(fits_filepath)}")
        return processed_data, processed_header
        
    except Exception as e:
        print(f"エラー: {fits_filepath} の前処理に失敗しました: {e}")
        return None, None


def parse_input_time(target_time):
    """
    target_time文字列をdatetimeに変換する

    受け入れ形式:
    - YYYYMMDD_HHMMSS
    - YYYY-MM-DDTHH:MM:SS
    - YYYY-MM-DD HH:MM:SS
    """
    if not isinstance(target_time, str):
        raise TypeError(f"target_time は文字列で指定してください: {target_time}")

    candidate_formats = [
        "%Y%m%d_%H%M%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in candidate_formats:
        try:
            return datetime.strptime(target_time, fmt)
        except ValueError:
            continue

    raise ValueError(
        f"target_time の形式が不正です: {target_time}. "
        "使用可能形式: YYYYMMDD_HHMMSS / YYYY-MM-DDTHH:MM:SS / YYYY-MM-DD HH:MM:SS"
    )


def extract_time_from_cor1_filename(filepath):
    """COR1ファイル名から観測時刻を抽出する。"""
    basename = os.path.basename(filepath)
    if "_n4c1A" not in basename:
        raise ValueError(f"COR1形式のファイル名ではありません: {basename}")
    time_str = basename.split("_n4c1A")[0]
    return datetime.strptime(time_str, "%Y%m%d_%H%M%S")


def select_nearest_cor1_fits_path(target_time, rawdata_dir=RAWDATA_DIR):
    """
    target_time に最も近い COR1 FITS ファイルパスを返す。
    """
    if not os.path.isdir(rawdata_dir):
        raise FileNotFoundError(f"Rawdataディレクトリが存在しません: {rawdata_dir}")

    target_dt = parse_input_time(target_time)
    fits_candidates = [
        os.path.join(rawdata_dir, filename)
        for filename in os.listdir(rawdata_dir)
        if filename.endswith("_n4c1A.fts")
    ]
    if not fits_candidates:
        raise FileNotFoundError(f"候補となるCOR1 FITSが見つかりません: {rawdata_dir}")

    nearest_path = min(
        fits_candidates,
        key=lambda p: abs((extract_time_from_cor1_filename(p) - target_dt).total_seconds())
    )
    nearest_time = extract_time_from_cor1_filename(nearest_path)
    return nearest_path, nearest_time, target_dt

def select_cor1_triplet_paths(reference_fits_filepath, rawdata_dir=RAWDATA_DIR, max_dt_seconds=45):
    """
    1つの参照COR1ファイルに対して、同一シーケンス内の
    POLAR=0,120,240 の3枚を自動取得する。
    """
    if not os.path.exists(reference_fits_filepath):
        raise FileNotFoundError(f"参照FITSが見つかりません: {reference_fits_filepath}")

    with fits.open(reference_fits_filepath) as hdul:
        ref_header = hdul[0].header
    ref_time = datetime.fromisoformat(ref_header['DATE-OBS'])

    best_paths = {0: None, 120: None, 240: None}
    best_dts = {0: np.inf, 120: np.inf, 240: np.inf}

    for filename in os.listdir(rawdata_dir):
        if not filename.endswith("_n4c1A.fts"):
            continue

        filepath = os.path.join(rawdata_dir, filename)
        try:
            with fits.open(filepath) as hdul:
                hdr = hdul[0].header
                obs_time = datetime.fromisoformat(hdr['DATE-OBS'])
                polar = float(hdr.get('POLAR', np.nan))
        except Exception:
            continue

        dt = abs((obs_time - ref_time).total_seconds())
        if dt > max_dt_seconds:
            continue

        nearest_polar = min([0, 120, 240], key=lambda p: abs(polar - p))
        if abs(polar - nearest_polar) > 5:
            continue

        if dt < best_dts[nearest_polar]:
            best_dts[nearest_polar] = dt
            best_paths[nearest_polar] = filepath

    missing = [p for p, path in best_paths.items() if path is None]
    if missing:
        raise FileNotFoundError(
            f"偏光シーケンスが揃いませんでした。missing={missing}, "
            f"reference={os.path.basename(reference_fits_filepath)}"
        )

    return best_paths


def resolve_daily_background_files(reference_fits_filepath, background_dir=None):
    """
    観測日から dc1A_p000/p120/p240/pTBr を自動で探す。
    """
    basename = os.path.basename(reference_fits_filepath)
    yyyymmdd = basename[:8]
    yymmdd = yyyymmdd[2:]

    background_names = {
        0: f"dc1A_p000_{yymmdd}.fts",
        120: f"dc1A_p120_{yymmdd}.fts",
        240: f"dc1A_p240_{yymmdd}.fts",
        "TBr": f"dc1A_pTBr_{yymmdd}.fts",
    }

    candidate_dirs = []
    if background_dir is not None:
        candidate_dirs.append(background_dir)

    ref_dir = os.path.dirname(reference_fits_filepath)
    candidate_dirs.extend([
        ref_dir,
        os.path.dirname(ref_dir),
        os.path.dirname(os.path.abspath(__file__)),
        ".",
    ])

    # 重複除去
    dedup_dirs = []
    for d in candidate_dirs:
        if d not in dedup_dirs:
            dedup_dirs.append(d)

    found = {}
    for key, name in background_names.items():
        for d in dedup_dirs:
            path = os.path.join(d, name)
            if os.path.exists(path):
                found[key] = path
                break

    # p000/p120/p240 は必須
    for key in [0, 120, 240]:
        if key not in found:
            raise FileNotFoundError(
                f"必要な背景ファイルが見つかりません: {background_names[key]}"
            )

    return found


def load_cor1_tbr_sequence(reference_fits_filepath, rawdata_dir=RAWDATA_DIR,
                           background_dir=None, subtract_tbr_background=True):
    """
    同一時刻帯の 0/120/240 偏光画像を読み込み、
    日次背景を各偏光に対して差し引いた上で Total Brightness を構成する。
    """
    triplet_paths = select_cor1_triplet_paths(reference_fits_filepath, rawdata_dir=rawdata_dir)
    background_files = resolve_daily_background_files(reference_fits_filepath, background_dir=background_dir)

    calibrator = STEREOCOR1Calibration(silent=True)

    corrected_images = {}
    corrected_headers = {}

    for polar in [0, 120, 240]:
        filepath = triplet_paths[polar]

        with fits.open(filepath) as hdul:
            raw_data = hdul[0].data.astype(np.float32)
            raw_header = hdul[0].header.copy()

        # BLANK値はNaNへ
        blank_value = raw_header.get('BLANK', None)
        if blank_value is not None:
            raw_data = np.where(raw_data == blank_value, np.nan, raw_data)

        # COR1日次背景は未絶対較正なので、
        # ここでは bias/exptime/SEB IP を合わせつつ、
        # CALFAC と CALIMG は掛けない。
        calibrated_data, calibrated_header = calibrator.cor1_calibrate(
            raw_data,
            raw_header,
            calfac_off=True,
            calimg_off=True
        )

        calibrated_data = calibrator.apply_smooth_mask(
            calibrated_data,
            calibrated_header,
            fill_value=np.nan
        )

        with fits.open(background_files[polar]) as hdul:
            bkg_data = hdul[0].data.astype(np.float32)

        if bkg_data.shape != calibrated_data.shape:
            min_shape = tuple(min(b, t) for b, t in zip(bkg_data.shape, calibrated_data.shape))
            bkg_data = bkg_data[:min_shape[0], :min_shape[1]]
            calibrated_data = calibrated_data[:min_shape[0], :min_shape[1]]

        corrected = calibrated_data - bkg_data
        corrected_images[polar] = corrected
        corrected_headers[polar] = calibrated_header

    # 3偏光から TBr を構成
    # 理想偏光子近似: TBr = 2/3 * (I0 + I120 + I240)
    tbr_data = (2.0 / 3.0) * (
        corrected_images[0] +
        corrected_images[120] +
        corrected_images[240]
    )

    # 必要なら日次 TBr 背景も補助的に差し引く
    if subtract_tbr_background and ("TBr" in background_files):
        with fits.open(background_files["TBr"]) as hdul:
            tbr_bkg = hdul[0].data.astype(np.float32)

        if tbr_bkg.shape != tbr_data.shape:
            min_shape = tuple(min(b, t) for b, t in zip(tbr_bkg.shape, tbr_data.shape))
            tbr_bkg = tbr_bkg[:min_shape[0], :min_shape[1]]
            tbr_data = tbr_data[:min_shape[0], :min_shape[1]]

        tbr_data = tbr_data - tbr_bkg

    # 代表ヘッダーは reference に最も近い偏光画像のものを採用
    with fits.open(reference_fits_filepath) as hdul:
        ref_header = hdul[0].header
    ref_polar = float(ref_header.get('POLAR', 240.0))
    rep_polar = min([0, 120, 240], key=lambda p: abs(ref_polar - p))

    tbr_header = corrected_headers[rep_polar].copy()
    tbr_header['POLAR'] = 1001.0
    tbr_header['BUNIT'] = 'DN/s'
    tbr_header.add_history("Built TBr from polarized sequence (0,120,240)")
    tbr_header.add_history(f"POL0 background: {os.path.basename(background_files[0])}")
    tbr_header.add_history(f"POL120 background: {os.path.basename(background_files[120])}")
    tbr_header.add_history(f"POL240 background: {os.path.basename(background_files[240])}")
    if subtract_tbr_background and ("TBr" in background_files):
        tbr_header.add_history(f"TBr background: {os.path.basename(background_files['TBr'])}")

    return tbr_data, tbr_header


def create_cor1_difference_plot(ax, target_time, output_dir=".", base_minutes_before=10,
                              rawdata_dir=RAWDATA_DIR, background_dir=None,
                              colormap_method='percentile', apply_median_correction=False,
                              subtract_tbr_background=True, plot_radius_rsun=4.0):
    """
    各時刻帯の 0/120/240 偏光シーケンスから TBr を作成し、
    target時刻の指定分前（既定:10分前）をbaseとして差分をプロットする。
    太陽北極が真上になるように画像回転も行う。

    Parameters
    ----------
    plot_radius_rsun : float
        太陽中心から何太陽半径までを表示するか。
        例: 4.0 なら x,y ともに ±4 R_sun の範囲を表示する。
    """
    try:
        target_fits_filepath, nearest_time, requested_time = select_nearest_cor1_fits_path(
            target_time,
            rawdata_dir=rawdata_dir
        )
    except Exception as e:
        print(f"エラー: target_time から最近傍ファイルを選択できませんでした: {e}")
        return None

    print(f"要求時刻: {requested_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
    print(f"選択ファイル: {os.path.basename(target_fits_filepath)} "
          f"({nearest_time.strftime('%Y-%m-%d %H:%M:%S')} UT)")

    base_requested_time = nearest_time - timedelta(minutes=base_minutes_before)
    base_request_str = base_requested_time.strftime("%Y%m%d_%H%M%S")
    try:
        base_fits_filepath, base_nearest_time, _ = select_nearest_cor1_fits_path(
            base_request_str,
            rawdata_dir=rawdata_dir
        )
    except Exception as e:
        print(f"エラー: Base時刻( target-{base_minutes_before}分 )の最近傍ファイル選択に失敗しました: {e}")
        return None

    print(f"Base要求時刻: {base_requested_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
    print(f"Base選択ファイル: {os.path.basename(base_fits_filepath)} "
          f"({base_nearest_time.strftime('%Y-%m-%d %H:%M:%S')} UT)")

    print("=" * 60)
    print("Base timeシーケンスの処理開始...")
    try:
        base_data, base_header = load_cor1_tbr_sequence(
            base_fits_filepath,
            rawdata_dir=rawdata_dir,
            background_dir=background_dir,
            subtract_tbr_background=subtract_tbr_background
        )
    except Exception as e:
        print(f"エラー: Base timeシーケンスのTBr構成に失敗しました: {e}")
        return None

    print("\n対象シーケンスの処理開始...")
    try:
        target_data, target_header = load_cor1_tbr_sequence(
            target_fits_filepath,
            rawdata_dir=rawdata_dir,
            background_dir=background_dir,
            subtract_tbr_background=subtract_tbr_background
        )
    except Exception as e:
        print(f"エラー: 対象シーケンスのTBr構成に失敗しました: {e}")
        return None

    print("\n差分計算の実行...")
    if base_data.shape != target_data.shape:
        print(f"警告: データの形状が異なります。Base: {base_data.shape}, Target: {target_data.shape}")
        min_shape = tuple(min(b, t) for b, t in zip(base_data.shape, target_data.shape))
        base_data = base_data[:min_shape[0], :min_shape[1]]
        target_data = target_data[:min_shape[0], :min_shape[1]]
        print(f"データを共通サイズにトリミング: {min_shape}")

    diff_data = target_data.astype(np.float64) - base_data.astype(np.float64)

    if apply_median_correction:
        median_offset = np.nanmedian(diff_data)
        print(f"\n中央値補正を適用: offset={median_offset:.3f}")
        diff_data -= median_offset

    negative_count = np.sum(diff_data < 0)
    diff_data = np.where(diff_data < 0, np.nan, diff_data)
    print(f"\n負値をNaNに置換: {negative_count} pixel")

    stats = analyze_data_distribution(diff_data)
    if stats['valid_count'] == 0:
        print("エラー: 差分画像に有効画素が存在しません。")
        return None

    print(f"\n差分データ統計:")
    print(f"  平均={stats['mean']:.3f}, 中央値={stats['median']:.3f}, 標準偏差={stats['std']:.3f}")
    print(f"  最小={stats['min']:.3f}, 最大={stats['max']:.3f}")

    print("\nSTEREO-A軌道傾き補正を適用...")
    diff_header = target_header.copy()
    diff_header.add_history("DIFFERENCE IMAGE: Target TBr - Base TBr")
    diff_header.add_history(f"Base rule: nearest file to target-{base_minutes_before}min")
    diff_header.add_history(f"Base requested time: {base_requested_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
    diff_header.add_history(f"Base anchor file: {os.path.basename(base_fits_filepath)}")
    diff_header.add_history(f"Requested target time: {target_time}")
    diff_header.add_history(f"Target anchor file: {os.path.basename(target_fits_filepath)}")
    if apply_median_correction:
        diff_header.add_history(f"Median correction applied: {median_offset:.3f}")

    # 太陽半径メタデータを整える
    rsun_arcsec, rsun_pix = get_solar_radius_arcsec_and_pixel(diff_header)
    diff_header['RSUN_OBS'] = rsun_arcsec
    if 'RSUN_REF' not in diff_header:
        diff_header['RSUN_REF'] = 6.957e8

    input_rotation = STEREOCoordinateTransform.get_crota_angle(diff_header)
    print(f"Debug: header回転角 = {input_rotation:.3f}度")
    print(f"Debug: solar radius = {rsun_arcsec:.3f} arcsec = {rsun_pix:.3f} pixel")

    print("\nsunpy.map.Mapオブジェクトの作成...")
    diff_map = sunpy.map.Map((diff_data, diff_header))

    print("Debug: SunPy rotate() により画像配列を北上へ補正中...")
    diff_map = diff_map.rotate(
        order=1,
        missing=np.nan,
        clip=False
    )
    print("✓ 画像配列の回転補正を適用: 真上が太陽北極になるよう補正済み")

    print("\nカラーマップの設定...")
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(alpha=0.0)
    norm, vmin, vmax = create_symmetric_colormap_norm(diff_map.data, method=colormap_method)

    print("\nプロット作成...")
    if hasattr(ax, 'figure') and ax.figure is not None:
        fig = ax.figure
        subplot_spec = ax.get_subplotspec()
        if hasattr(ax, 'remove'):
            ax.remove()
        ax = fig.add_subplot(subplot_spec, projection=diff_map.wcs)
    else:
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(projection=diff_map.wcs)

    im = diff_map.plot(axes=ax, cmap=cmap, norm=norm, title=False)

    diff_map.draw_limb(axes=ax, color='yellow', linewidth=2, alpha=0.9)
    diff_map.draw_grid(
        axes=ax,
        grid_spacing=(15, 15) * u.deg,
        color='black',
        alpha=0.6,
        linestyle=':',
        linewidth=0.8
    )

    # WCS上の太陽中心 (0,0 arcsec) を pixel に変換
    sun_center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=diff_map.coordinate_frame)
    sun_center_pix = diff_map.world_to_pixel(sun_center)
    sun_x_pix = float(sun_center_pix.x.value)
    sun_y_pix = float(sun_center_pix.y.value)

    # 破線円: world座標で作って pixel に変換して描く
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    for radius_factor in [1, 2, 3, 4]:
        r_arcsec = radius_factor * rsun_arcsec * u.arcsec
        circle_coord = SkyCoord(
            r_arcsec * np.cos(theta),
            r_arcsec * np.sin(theta),
            frame=diff_map.coordinate_frame
        )
        circle_pix = diff_map.world_to_pixel(circle_coord)

        ax.plot(
            circle_pix.x.value,
            circle_pix.y.value,
            color='black',
            linewidth=1.5,
            alpha=0.8,
            linestyle='--'
        )

    # 表示範囲は pixel で指定
    half_width_pix = plot_radius_rsun * rsun_pix
    ax.set_xlim(sun_x_pix - half_width_pix, sun_x_pix + half_width_pix)
    ax.set_ylim(sun_y_pix - half_width_pix, sun_y_pix + half_width_pix)

    target_time_str = diff_map.date.strftime('%Y-%m-%d %H:%M:%S UT')
    base_time_str = base_header.get('DATE-OBS', base_nearest_time.strftime('%Y-%m-%d %H:%M:%S'))

    title_line1 = 'STEREO-A/SECCHI/COR1 - TBr Difference Image'
    title_line2 = f'Target: {target_time_str}\nBase(≈target-{base_minutes_before}min): {base_time_str}'
    full_title = f'{title_line1}\n{title_line2}'

    ax.set_title(full_title, fontsize=14)
    # ax.set_xlabel('Solar X [arcsec]', fontsize=12)
    # ax.set_ylabel('Solar Y [arcsec]', fontsize=12)

    return ax, diff_map, im

 
if __name__ == '__main__':
    # target_time = ['20220613_030136',
    #                '20220613_030636',
    #                '20220613_031136',
    #                '20220613_031636',
    #                '20220613_032136',
    #                '20220613_032636',
    #                '20220613_033136',
    #                '20220613_033636',
    #                '20220613_034136',
    #                '20220613_034636',
    #                '20220613_035136',
    #                '20220613_035636',
    #                '20220613_040136',
    #                '20220613_040636',
    #                '20220613_041136']
    # target_fits_file = [f"{time}_n4c1A.fts" for time in target_time]
    output_directory = "/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1"
    
    # fig = plt.figure(figsize=(35, 24))
    
    # for i, target_time_i in enumerate(target_time):
    #     print("=== 改良版COR1差分プロット作成プログラム ===")
    #     print(f"Base time: {os.path.basename(BASE_TIME_FILE)}")
    #     print(f"Target file: {target_time_i}_n4c1A.fts")
        
    #     # 各サブプロットでWCSAxesを作成するためにまずデータを読み込む
    #     from astropy.io import fits
    #     with fits.open(os.path.join(RAWDATA_DIR, f"{target_time_i}_n4c1A.fts")) as hdul:
    #         header = hdul[0].header
        
    #     # WCSを使用してサブプロットを作成
    #     ax = fig.add_subplot(3, 5, i+1, projection=WCS(header))
        
    #     create_cor1_difference_plot(
    #         ax, target_time_i,
    #         output_directory,
    #         colormap_method='fixed',  # 'percentile', 'std', 'fixed' から選択
    #         apply_median_correction=True   # 系統的バイアスの補正
    #     )
        # 画像の保存
    fig, ax = plt.subplots(figsize=(10,10))

    target_time = '20220613_034636'
    result = create_cor1_difference_plot(ax, target_time, plot_radius_rsun=3.6)

    if result is None:
        print("✗ プロット生成に失敗したため、保存を中止します。")
    else:
        ax, diff_map, im = result
        print("\nプロットの保存...")
        plt.tight_layout()
        output_filename = f"cor1_diff_from_{target_time}.png"
        save_path = os.path.join(output_directory, output_filename)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ 差分プロットが正常に保存されました: {save_path}")
        print("=" * 60)