"""
UCOMPファイルスキャナー
integrated_analysis.pyのscan_directory_for_mapsを使用してUCOMPデータをスキャン
"""

from ucomp_config import *
from integrated_analysis import scan_directory_for_maps
import re
from datetime import datetime

def parse_ucomp_filename(filename):
    """
    UCOMPファイル名を解析してタイムスタンプを取得
    ファイル名形式: YYYYMMDD.HHMMSS.ucomp.<wavelength>.l2.fts
    
    Parameters
    ----------
    filename : str
        UCOMPファイル名
        
    Returns
    -------
    datetime
        ファイルのタイムスタンプ
    """
    # UCOMPファイル名のパターン
    pattern = r"(\d{8})\.(\d{6})\.ucomp\.(\d+)\.l2\.fts"
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(f"Invalid UCOMP filename format: {filename}")
    
    date_str, time_str, wavelength = match.groups()
    
    # 日時文字列を作成
    datetime_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
    
    return datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S"), int(wavelength)

def scan_ucomp_data(start_time, end_time, wavelength=None, use_cache=True):
    """
    指定された時間範囲でUCOMPデータをスキャン（ファイル名ベース）
    
    Parameters
    ----------
    start_time : str or Time
        スキャン開始時刻 (ISO形式)
    end_time : str or Time  
        スキャン終了時刻 (ISO形式)
    wavelength : int or None
        波長 (Noneの場合は全波長のデータを取得)
    use_cache : bool
        キャッシュを使用するか (default: True)
        
    Returns
    -------
    list
        [(sunpy.map.Map, file_path), ...] のリスト
    """
    if wavelength is not None:
        validate_wavelength(wavelength)
    
    # 時刻オブジェクトに変換
    if isinstance(start_time, str):
        start_time_obj = Time(start_time)
    else:
        start_time_obj = start_time
        
    if isinstance(end_time, str):
        end_time_obj = Time(end_time)
    else:
        end_time_obj = end_time
    
    ucomp_dir = get_ucomp_data_path()
    
    print(f"Scanning UCOMP data: {ucomp_dir}")
    print(f"Time range: {start_time_obj.iso} - {end_time_obj.iso}")
    if wavelength is not None:
        print(f"Wavelength: {wavelength} nm")
    else:
        print("Wavelength: All wavelengths")
    
    # UCOMPファイル専用スキャン（ファイル名ベース）
    try:
        # 波長フィルタリングの設定
        if wavelength is not None:
            file_pattern = f"*.ucomp.{wavelength}.l2.fts"
        else:
            file_pattern = "*.ucomp.*.l2.fts"
        files = list(ucomp_dir.glob(file_pattern))
        
        print(f"Found {len(files)} total files with pattern: {file_pattern}")
        
        maps_with_paths = []
        
        for file_path in files:
            try:
                # ファイル名から日時を解析
                filename = file_path.name
                file_datetime, file_wavelength = parse_ucomp_filename(filename)
                file_time = Time(file_datetime)
                
                # 時間範囲チェック
                if start_time_obj <= file_time <= end_time_obj:
                    # 波長フィルタリング（wavelengthがNoneの場合は全波長を受け入れる）
                    if wavelength is None or file_wavelength == wavelength:
                        # SunPy Mapを作成せず、時刻情報とファイルパスのみを保存
                        # 簡易的なオブジェクトとして辞書を使用
                        file_info = {
                            'date': file_time,
                            'filename': filename,
                            'file_path': file_path,
                            'wavelength': file_wavelength
                        }
                        maps_with_paths.append((file_info, file_path))
                        print(f"  Added: {filename} ({file_time.iso})")
                        
            except ValueError as e:
                # UCOMPファイル名でない場合はスキップ
                print(f"  Skipped: {file_path.name} (invalid filename format)")
                continue
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
                continue
        
        # 時刻順にソート
        maps_with_paths.sort(key=lambda x: x[0]['date'])
        
        if wavelength is not None:
            print(f"Found {len(maps_with_paths)} UCOMP files for wavelength {wavelength} in time range")
        else:
            print(f"Found {len(maps_with_paths)} UCOMP files for all wavelengths in time range")
        return maps_with_paths
        
    except Exception as e:
        print(f"Error scanning UCOMP data: {e}")
        import traceback
        traceback.print_exc()
        return []

def find_closest_ucomp_data(target_time, start_time, end_time, wavelength=None):
    """
    指定時刻に最も近いUCOMPデータを見つける
    
    Parameters
    ----------
    target_time : str or Time
        目標時刻
    start_time : str or Time
        スキャン開始時刻
    end_time : str or Time
        スキャン終了時刻  
    wavelength : int
        波長
        
    Returns
    -------
    tuple
        (closest_map, file_path) または None
    """
    if isinstance(target_time, str):
        target_time_obj = Time(target_time)
    else:
        target_time_obj = target_time
    
    # データをスキャン
    maps_with_paths = scan_ucomp_data(start_time, end_time, wavelength)
    
    if not maps_with_paths:
        print(f"No UCOMP data found for wavelength {wavelength}")
        return None
    
    # 最も近い時刻のデータを見つける
    min_diff = float('inf')
    closest_data = None
    
    for file_info, file_path in maps_with_paths:
        time_diff = abs((file_info['date'] - target_time_obj).sec)
        if time_diff < min_diff:
            min_diff = time_diff
            closest_data = (file_info, file_path)
    
    if closest_data:
        closest_info, closest_path = closest_data
        print(f"Closest UCOMP data:")
        print(f"  Target time: {target_time_obj.iso}")
        print(f"  Actual time: {closest_info['date'].iso}")
        print(f"  Time diff: {min_diff:.1f} seconds")
        print(f"  File: {closest_info['filename']}")
    
    return closest_data

def get_available_ucomp_times(start_time, end_time, wavelength=None):
    """
    指定時間範囲で利用可能なUCOMPデータの時刻リストを取得
    
    Parameters
    ----------
    start_time : str or Time
        開始時刻
    end_time : str or Time
        終了時刻
    wavelength : int
        波長
        
    Returns
    -------
    list
        利用可能な時刻のリスト
    """
    maps_with_paths = scan_ucomp_data(start_time, end_time, wavelength)
    times = [file_info['date'] for file_info, _ in maps_with_paths]
    return sorted(times)