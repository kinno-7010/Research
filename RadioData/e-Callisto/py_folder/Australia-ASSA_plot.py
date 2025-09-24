#!/usr/bin/env python3
"""
e-Callisto Dynamic Spectrum Plotter
太陽電波バースト観測データのダイナミックスペクトル可視化ツール
複数HDU対応版 - バイナリテーブルから実際の周波数軸を読み取る
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import matplotlib.dates as mdates
from matplotlib.colors import PowerNorm, LogNorm
from datetime import datetime, timedelta
from pathlib import Path
from scipy.ndimage import median_filter, gaussian_filter1d
import warnings
from matplotlib.ticker import MultipleLocator, LogLocator, FuncFormatter
from matplotlib.dates import SecondLocator
import os
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime as dt

def _to_datetime(dt_input):
    """
    Convert a datetime or ISO-format string to a datetime object.
    """
    return dt.datetime.fromisoformat(dt_input) if isinstance(dt_input, str) else dt_input

def _slice_data(time_array, data, start_dt, end_dt, freq_mhz, freq_min, freq_max):
    """
    Slice time and frequency ranges from a 2D data array.
    Returns: t_sel, f_sel, d_sel
    """
    # Time slice
    mask_t = (time_array >= start_dt) & (time_array <= end_dt)
    t_sel = time_array[mask_t]
    d_sel = data[mask_t, :]

    # Frequency slice
    freq_array = np.asarray(freq_mhz, dtype=float)
    freq_min_float = float(freq_min)
    freq_max_float = float(freq_max)
    mask_f = (freq_array >= freq_min_float) & (freq_array <= freq_max_float)
    f_sel = freq_array[mask_f]
    d_sel = d_sel[:, mask_f]
    return t_sel, f_sel, d_sel

class eCallistoSpectrum:
    """e-Callistoデータの処理とダイナミックスペクトル生成クラス"""
    
    def __init__(self, fits_path):
        """
        Parameters
        ----------
        fits_path : str, Path, or list
            FITSファイルへのパス（単一ファイルまたは複数ファイルのリスト）
        """
        if isinstance(fits_path, (list, tuple)):
            # 複数ファイルの場合、マージ処理を実行
            self.merge_multiple_files(fits_path)
        else:
            # 単一ファイルの場合、従来通りの処理
            self.load_single_file(fits_path)
    
    def load_single_file(self, fits_path):
        """単一FITSファイルを読み込み"""
        self.fits_path = Path(fits_path)
        if not self.fits_path.exists():
            raise FileNotFoundError(f"FITSファイル '{fits_path}' が見つかりません")
        
        # FITSファイルを読み込み（全HDUを確認）
        self.hdulist = fits.open(self.fits_path)
        
        print("=" * 60)
        print(f"FITSファイル: {self.fits_path.name}")
        print("=" * 60)
        print(f"HDUの数: {len(self.hdulist)}")
        self.hdulist.info()
        print("-" * 60)
        
        # Primary HDUからデータとヘッダーを取得
        self.header = self.hdulist[0].header
        self.data = self.hdulist[0].data
        
        # メタデータを抽出
        self.extract_metadata()
        
        # 実際の周波数軸を取得（バイナリテーブルまたはヘッダーから）
        self.extract_frequency_axis()
        
        print(f"\nデータ形状: {self.data.shape}")
        print(f"時間範囲: {self.time_obs} - {self.time_end} UT")
        print(f"周波数範囲: {min(self.freq_axis):.2f} - {max(self.freq_axis):.2f} MHz")
        print(f"時間分解能: {self.time_step} 秒")
        print(f"周波数チャンネル数: {self.n_freq}")
    
    def merge_multiple_files(self, file_paths):
        """複数のFITSファイルをマージ"""
        print("=" * 60)
        print("複数ファイルマージモード")
        print("=" * 60)
        print(f"ファイル数: {len(file_paths)}")
        
        # 各ファイルのデータを読み込み
        file_data = []
        for file_path in file_paths:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"FITSファイル '{file_path}' が見つかりません")
            
            print(f"\n読み込み中: {file_path.name}")
            
            # 一時的にeCallistoSpectrumオブジェクトを作成してデータを取得
            temp_spectrum = eCallistoSpectrum.__new__(eCallistoSpectrum)
            temp_spectrum.load_single_file(file_path)
            
            # 時間軸を生成
            temp_spectrum.create_time_axis()
            
            file_data.append({
                'time_axis': temp_spectrum.time_axis,
                'freq_axis': temp_spectrum.freq_axis,
                'data': temp_spectrum.data,
                'start_time': temp_spectrum.time_axis[0],
                'file': file_path.name,
                'header': temp_spectrum.header
            })
            
            # リソースを解放
            temp_spectrum.hdulist.close()
        
        # 時間順にソート
        file_data.sort(key=lambda x: x['start_time'])
        
        print("\n時間順ファイル一覧:")
        for fd in file_data:
            print(f"  {fd['file']}: {fd['start_time'].strftime('%H:%M:%S')} - {fd['time_axis'][-1].strftime('%H:%M:%S')}")
        
        # 最初のファイルの周波数軸を基準とする
        self.freq_axis = file_data[0]['freq_axis']
        
        # 周波数軸の整合性をチェック
        for i, fd in enumerate(file_data[1:], 1):
            if not np.allclose(fd['freq_axis'], self.freq_axis, rtol=1e-5):
                print(f"  警告: ファイル {i+1} の周波数軸が異なります。補間処理を実行します。")
                # 補間処理
                interp_func = interp1d(fd['freq_axis'], fd['data'], axis=0, 
                                      bounds_error=False, fill_value=np.nan)
                fd['data'] = interp_func(self.freq_axis)
        
        # 時間軸とデータを結合
        merged_time_axis = []
        merged_data_list = []
        
        for fd in file_data:
            merged_time_axis.extend(fd['time_axis'])
            merged_data_list.append(fd['data'])
        
        self.time_axis = merged_time_axis
        # データの形状を確認してからaxis決定
        print(f"  各ファイルのデータ形状:")
        for i, fd in enumerate(file_data):
            print(f"    ファイル{i+1}: {fd['data'].shape}")
        
        # e-Callistoデータは(freq, time)形状なので、axis=1で時間軸方向に結合
        self.data = np.concatenate(merged_data_list, axis=1)  # 時間軸方向で結合
        
        # 最初のファイルのメタデータを使用
        self.header = file_data[0]['header']
        self.extract_metadata()
        
        # 更新された情報
        self.time_obs = self.time_axis[0].strftime('%H:%M:%S.%f')[:-3]
        self.time_end = self.time_axis[-1].strftime('%H:%M:%S')
        
        print(f"\nマージ結果:")
        print(f"データ形状: {self.data.shape}")
        print(f"時間軸長さ: {len(self.time_axis)}")
        print(f"時間範囲: {self.time_obs} - {self.time_end} UT")
        print(f"周波数範囲: {min(self.freq_axis):.2f} - {max(self.freq_axis):.2f} MHz")
        print(f"総時間点数: {len(self.time_axis)}")
        print(f"周波数チャンネル数: {self.n_freq}")
        
        # データ形状の整合性チェック
        if self.data.shape[1] != len(self.time_axis):
            print(f"  警告: データの時間軸({self.data.shape[1]})と時間軸配列({len(self.time_axis)})のサイズが一致しません")
        
        # hdulistは複数ファイルなので設定しない
        self.hdulist = None
        
    def extract_metadata(self):
        """FITSヘッダーから必要なメタデータを抽出"""
        
        # 時間軸の情報
        self.n_time = self.header['NAXIS1']  # 時間軸のデータ点数
        self.time_start_sec = self.header['CRVAL1']  # 開始時刻（その日の秒数）
        self.time_step = self.header['CDELT1']  # 時間ステップ（秒）
        
        # 周波数軸の情報（ヘッダーの名目値）
        self.n_freq = self.header['NAXIS2']  # 周波数軸のデータ点数
        self.freq_start_nominal = self.header.get('CRVAL2', None)
        self.freq_step_nominal = self.header.get('CDELT2', None)
        
        # 観測情報
        self.date_obs = self.header['DATE-OBS']
        self.time_obs = self.header['TIME-OBS']
        self.time_end = self.header.get('TIME-END', self.header.get('TIME_END', ''))
        self.instrument = self.header.get('INSTRUME', 'Unknown')
        self.location = self.header.get('ORIGIN', 'Unknown')
        
        # 観測地点の情報
        self.obs_lat = self.header.get('OBS_LAT', None)
        self.obs_lon = self.header.get('OBS_LON', None)
        self.obs_alt = self.header.get('OBS_ALT', None)
        
        # データの単位とキャリブレーション状態
        self.bunit = self.header.get('BUNIT', 'digits')
        self.is_calibrated = (self.bunit == 'SFU')
        
        # 周波数設定ファイル
        self.frqfile = self.header.get('FRQFILE', None)
        
    def extract_frequency_axis(self):
        """実際の周波数軸を取得（バイナリテーブルまたは計算から）"""
        
        frequency_extracted = False
        
        # 方法1: バイナリテーブルから周波数軸を探す
        if len(self.hdulist) > 1:
            print("\nバイナリテーブルを検索中...")
            
            for i, hdu in enumerate(self.hdulist[1:], 1):
                if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                    print(f"  HDU {i}: {hdu.name} - カラム数: {len(hdu.columns.names)}")
                    
                    # カラム名を表示
                    for col_name in hdu.columns.names:
                        col_data = hdu.data[col_name]
                        print(f"    カラム '{col_name}': 形状={col_data.shape}, 型={col_data.dtype}")
                        
                        # 周波数に関連するカラムを探す
                        if 'freq' in col_name.lower() or col_name.lower() in ['frequency', 'frequencies']:
                            self.freq_axis = np.array(col_data).flatten()
                            if len(self.freq_axis) == self.n_freq:
                                frequency_extracted = True
                                print(f"    → 周波数軸として採用 ({len(self.freq_axis)}チャンネル)")
                                break
                            
                    if frequency_extracted:
                        break
                    
                    # カラム名がない場合、サイズで判定
                    if not frequency_extracted:
                        for col_name in hdu.columns.names:
                            col_data = hdu.data[col_name]
                            if len(col_data) == self.n_freq:
                                # 周波数の可能性があるデータかチェック
                                if np.all(col_data > 0) and np.all(col_data < 1000):  # MHz範囲
                                    self.freq_axis = np.array(col_data).flatten()
                                    frequency_extracted = True
                                    print(f"    → カラム '{col_name}' を周波数軸として採用")
                                    break
        
        # 方法2: ヘッダーの情報から計算（フォールバック）
        if not frequency_extracted:
            print("\nバイナリテーブルに周波数軸が見つからなかったため、ヘッダー情報から計算...")
            
            if self.freq_start_nominal and self.freq_step_nominal:
                # ヘッダーの名目値から計算
                self.freq_axis = self.freq_start_nominal + np.arange(self.n_freq) * self.freq_step_nominal
                print(f"  ヘッダーから計算: {self.freq_start_nominal:.1f} MHz から {self.freq_step_nominal:.1f} MHz刻み")
                
                # Australia ASSAの特別処理（既知の周波数範囲）
                if 'AUSTRALIA' in self.instrument.upper() and 'ASSA' in self.instrument.upper():
                    print("  Australia ASSA ステーションを検出 - 既知の周波数範囲を適用")
                    # バイナリテーブルから判明した実際の周波数範囲（降順）
                    self.freq_axis = np.linspace(86.9, 15.0, self.n_freq)  # 86.9-15.0 MHz
            else:
                # デフォルト値（Australia-ASSAに適した範囲）
                warnings.warn("周波数軸の情報が不完全です。デフォルト値を使用します。")
                self.freq_axis = np.linspace(86.9, 15.0, self.n_freq)  # デフォルト
        
        # 周波数範囲は実際のデータから取得
        # （freq_axis_min/maxは削除し、min(freq_axis)/max(freq_axis)を使用）
        
        
    def create_time_axis(self):
        """時間軸を生成"""
        
        # 観測開始時刻を基準にする
        base_datetime = datetime.strptime(f"{self.date_obs} {self.time_obs}", 
                                         "%Y/%m/%d %H:%M:%S.%f")
        
        # 時間配列を生成
        time_seconds = np.arange(self.n_time) * self.time_step
        self.time_axis = [base_datetime + timedelta(seconds=float(t)) for t in time_seconds]
        
    def preprocess_data(self, background_method, intensity_threshold):
        """
        データの前処理と背景除去
        
        Parameters
        ----------
        background_method : str
            背景除去の方法
            - 'median_time': 時間方向の中央値を減算（デフォルト）
            - 'percentile': パーセンタイル法
            - 'rolling_median': 移動中央値
            - 'quiet_time': 静穏時間帯を背景とする
        intensity_threshold : float
            intensityマスク処理の閾値（median_timeの場合のみ使用）
        """
        self.processed_data = self.data.astype(float)
        
        print(f"\n背景除去処理: {background_method}")
        
        if background_method == 'median_time':
            # 時間方向の中央値を背景とする
            background = np.median(self.processed_data, axis=1, keepdims=True)
            self.processed_data = self.processed_data - background
            print("  - 各周波数での時間中央値を減算")
            
            # intensityが閾値以下のデータをnp.nanとして扱う
            low_intensity_mask = self.processed_data <= intensity_threshold
            self.processed_data[low_intensity_mask] = np.nan

            
        elif background_method == 'percentile':
            # パーセンタイル法
            background = np.percentile(self.processed_data, 25, axis=1, keepdims=True)
            self.processed_data = self.processed_data - background
            print("  - 25パーセンタイル値を背景として減算")
            
        elif background_method == 'rolling_median':
            # 移動中央値法
            window_size = 10  # 25秒間（100点 × 0.25秒）
            self.processed_data_bg = np.zeros_like(self.processed_data)
            for i in range(self.n_freq):
                background_rolling = median_filter(self.processed_data[i, :], size=window_size)
                self.processed_data_bg[i, :] = self.processed_data[i, :] - background_rolling
            self.processed_data = self.processed_data_bg
            print(f"  - 移動中央値（窓幅: {window_size*self.time_step:.1f}秒）を減算")
            
        elif background_method == 'quiet_time':
            # 静穏時間帯を選択して背景とする
            quiet_samples = 100
            background = np.nanmean(self.processed_data[:, :quiet_samples], axis=1, keepdims=True)
            self.processed_data = self.processed_data - background
            print(f"  - 最初の{quiet_samples*self.time_step:.1f}秒間を静穏時として減算")
        
        # 負の値を0にクリップ
        self.processed_data = np.clip(self.processed_data, 0, None)
        
        # スパイクノイズの除去
        threshold = np.percentile(self.processed_data[self.processed_data > 0], 99.5)
        self.processed_data = np.clip(self.processed_data, 0, threshold)
        
        print(f"  - 処理後のデータ範囲: {self.processed_data.min():.2f} - {self.processed_data.max():.2f}")
    
    def plot_dynamic_spectrum(self, start_time, end_time, freq_min, freq_max, time_tick_sec, freq_tick_mhz, med_filter_size, title, background_method, intensity_threshold):
        """
        Plot a dynamic spectrum between start_time and end_time.
        (HF_plot方式を適用)
        """
        # 時間軸を生成（マージ済みの場合はスキップ）
        if not hasattr(self, 'time_axis') or len(self.time_axis) == 0:
            self.create_time_axis()
        else:
            print(f"マージ済み時間軸を使用: {len(self.time_axis)} 点")
        
        # データの前処理（選択されたノイズ除去方法に基づく）
        if background_method is not None:
            print(f"ノイズ除去処理を実行: {background_method}")
            self.preprocess_data(background_method, intensity_threshold)
        else:
            print("ノイズ除去処理をスキップします")
            # processed_dataを元のdataで初期化
            self.processed_data = self.data.astype(float)
            # intensityが160以下のデータをnp.nanとして扱う
            intensity_threshold_none = 165
            low_intensity_mask = self.processed_data <= intensity_threshold_none
            self.processed_data[low_intensity_mask] = np.nan
        
        # 時間軸とデータをnumpy配列に変換（HF_plot方式に合わせる）
        time_array = np.array(self.time_axis)
        frequency_array = np.array(self.freq_axis)
        
        # データ形状チェックと修正
        print(f"デバッグ情報:")
        print(f"  時間軸サイズ: {len(time_array)}")
        print(f"  周波数軸サイズ: {len(frequency_array)}")
        print(f"  処理前データ形状: {self.processed_data.shape}")
        
        # 時間軸とデータの整合性チェック
        if self.processed_data.shape[1] != len(time_array):
            print(f"  エラー: データ時間軸({self.processed_data.shape[1]})と時間軸配列({len(time_array)})が不一致")
            raise ValueError(f"データ形状とtime_axisのサイズが一致しません: {self.processed_data.shape[1]} != {len(time_array)}")
        
        data_array = self.processed_data.T  # 転置してHF_plot形式に合わせる (time, freq)
        print(f"  転置後データ形状: {data_array.shape}")
        
        # HF_plot方式のスライス処理
        start_dt = _to_datetime(start_time)
        end_dt = _to_datetime(end_time)
        t_sel, f_sel, d_sel = _slice_data(
            time_array, data_array, start_dt, end_dt,
            frequency_array, freq_min, freq_max
        )

        # ノイズ軽減
        d_filt = median_filter(d_sel.astype(float), size=med_filter_size)

        # 図を作成
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # カラーバー範囲をデータ統計情報から設定
        valid_data = d_filt[d_filt > 0]
        if len(valid_data) > 0:
            if background_method is None:
                vmin = np.percentile(valid_data, 0.5)
                vmax = np.percentile(valid_data, 99.5)
            else:
                vmin = 0
                vmax = 30
        else:
            vmin, vmax = 0, 1
        
        # 画像の描画（周波数軸を高い方が上になるよう設定）
        extent = [
            mdates.date2num(t_sel[0]), mdates.date2num(t_sel[-1]),
            f_sel[-1], f_sel[0]  # 周波数軸を反転させて高い方が上に
        ]
        im = ax.imshow(
            d_filt.T, origin='upper', aspect='auto',  # origin='upper'で高周波数が上
            extent=extent, cmap='viridis', vmin=vmin, vmax=vmax
        )
        
        # カラーバーの追加（元の形式）
        cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.5)
        unit_label = 'SFU' if self.is_calibrated else 'digits'
        if background_method is not None:
            cbar.set_label(f'Intensity [{unit_label}] (background subtracted)', fontsize=12)
        else:
            cbar.set_label(f'Intensity [{unit_label}] (raw data)', fontsize=12)

        # ラベルとフォーマット（元のe-Callisto形式）
        ax.set_xlabel('Time [UT]', fontsize=12)
        ax.set_ylabel('Frequency [MHz]', fontsize=12)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(SecondLocator(interval=time_tick_sec))
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # タイトル（元のe-Callisto形式、ユーザー指定時間範囲を表示）
        start_time_str = start_dt.strftime('%H:%M:%S')
        end_time_str = end_dt.strftime('%H:%M:%S')
        
        if background_method is not None:
            if background_method == 'median_time':
                title_text = f"e-Callisto Dynamic Spectrum (Background Subtracted {background_method}, intensity masked > {intensity_threshold})\n"
            else:
                title_text = f"e-Callisto Dynamic Spectrum (Background Subtracted {background_method})\n"
        else:
            title_text = f"e-Callisto Dynamic Spectrum (No Background Subtraction)\n"
        title_text += f"{self.instrument} @ {self.location}\n"
        title_text += f"{self.date_obs} {start_time_str} - {end_time_str} UT"
        if hasattr(self, 'frqfile') and self.frqfile:
            title_text += f" (FRQ: {self.frqfile})"
        ax.set_title(title_text, fontsize=14)
        
        # グリッド
        ax.grid(True, alpha=0.3, linestyle='--', color='white')
        
        # ファイル保存
        date_str = self.date_obs.replace('/', '')  # 2022/06/13 → 20220613
        time_start_str = start_time.split('.')[0].replace(':', '')  # 03:14:58.217 → 031458
        time_end_str = end_time.split('.')[0].replace(':', '')    # 03:29:58 → 032958
        
        filename = f'/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/output/Australia-ASSA_{date_str}_{time_start_str}_{time_end_str}_{background_method}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'画像を保存しました: {filename}')
        plt.tight_layout()
        
        # 表示
        plt.show()
        
        return fig, ax
        
        
    def close(self):
        """FITSファイルを閉じる"""
        if hasattr(self, 'hdulist') and self.hdulist is not None:
            self.hdulist.close()


def main():
    """メイン関数"""
    
    try:
        print("\n" + "=" * 60)
        print("e-Callisto Dynamic Spectrum Generator")
        print("複数HDU対応版・マージ機能対応")
        print("=" * 60)
        
        # 固定ファイルパス
        file_paths = [
            '/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_031500_62.fit',
            '/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_033000_62.fit'
        ]
        
        # ファイル存在確認
        existing_files = []
        for file_path in file_paths:
            if Path(file_path).exists():
                existing_files.append(file_path)
            else:
                print(f"  警告: ファイル '{Path(file_path).name}' が見つかりません")
        
        file_paths = existing_files
        if not file_paths:
            raise FileNotFoundError("利用可能なファイルが見つかりません")
        
        print(f"\n選択されたファイル ({len(file_paths)} 個):")
        for i, fp in enumerate(file_paths, 1):
            print(f"  {i}. {Path(fp).name}")
        
        # ファイル数に応じて自動判定（1つの場合は単一ファイル、複数の場合はマージ）
        if len(file_paths) == 1:
            print("\n単一ファイルモード")
            spectrum = eCallistoSpectrum(file_paths[0])
        else:
            print(f"\n複数ファイルマージモード ({len(file_paths)} ファイル)")
            spectrum = eCallistoSpectrum(file_paths)

        # stats = spectrum.analyze_burst_statistics()
        
        # プロットパラメータの入力
        print("\n" + "-" * 60)
        print("プロットパラメータの設定")
        print("-" * 60)
        
        start_time = input("開始時刻を入力 [default: 2022-06-13T03:25:00]: ").strip()
        if start_time == '':
            start_time = "2022-06-13T03:25:00"
        
        end_time = input("終了時刻を入力 [default: 2022-06-13T03:33:00]: ").strip()
        if end_time == '':
            end_time = "2022-06-13T03:33:00"  
            
        min_frequency = input("最小周波数を入力 (MHz) [default: 25 MHz]: ").strip()
        if min_frequency == '':
            min_frequency = 25
        else:
            min_frequency = float(min_frequency)
            
        max_frequency = input("最大周波数を入力 (MHz) [default: 45 MHz]: ").strip()
        if max_frequency == '':
            max_frequency = 45
        else:
            max_frequency = float(max_frequency)
        
        # ノイズ除去方法の選択
        print("\nノイズ除去方法を選択してください:")
        print("0. ノイズ除去しない")
        print("1. median_time (時間方向中央値)")
        print("2. percentile (パーセンタイル法)")
        print("3. rolling_median (移動中央値)")
        print("4. quiet_time (静穏時間帯)")
        
        noise_choice = input("選択してください (0-4) [default: 1]: ").strip()
        if noise_choice == '':
            noise_choice = '1'
            
        background_methods = {
            '0': None,  # ノイズ除去しない
            '1': 'median_time',
            '2': 'percentile',
            '3': 'rolling_median',
            '4': 'quiet_time'
        }
        
        background_method = background_methods.get(noise_choice, 'median_time')
        
        time_tick_sec = 30
        freq_tick_mhz = 1
        med_filter_size = (1, 1)
        # カラーバー範囲はデータに基づいて動的に設定
        title = f'e-Callisto Dynamic Spectrum - MERGED'
        intensity_threshold = 2
        
        print(f"\n設定されたパラメータ:")
        print(f"  開始時刻: {start_time}")
        print(f"  終了時刻: {end_time}")
        print(f"  最小周波数: {min_frequency} MHz")
        print(f"  最大周波数: {max_frequency} MHz")
        print(f"  ノイズ除去方法: {background_method if background_method else 'なし'}")
        
        # ダイナミックスペクトルをプロット（HF_plot方式）
        fig, ax = spectrum.plot_dynamic_spectrum(
            start_time, end_time,
            min_frequency, max_frequency,
            time_tick_sec, freq_tick_mhz,
            med_filter_size,
            title,
            background_method,
            intensity_threshold
        )
        
        # クリーンアップ
        spectrum.close()
        
        print("\n処理完了")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())