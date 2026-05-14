#!/usr/bin/env python3
# https://soleil.i4ds.ch/solarradio/data/

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
        """複数のFITSファイルを「時間×周波数」の2軸で統合してマージ"""
        print("=" * 60)
        print("複数ファイルマージモード（時間軸＋周波数軸を統合）")
        print("=" * 60)
        print(f"ファイル数: {len(file_paths)}")

        # -----------------------------
        # 1) 各ファイルを読み込み、(freq, time)・time_axis・freq_axis を取得
        # -----------------------------
        file_data = []
        for file_path in file_paths:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"FITSファイル '{file_path}' が見つかりません")

            print(f"\n読み込み中: {file_path.name}")

            temp = eCallistoSpectrum.__new__(eCallistoSpectrum)
            temp.load_single_file(file_path)
            temp.create_time_axis()

            # 標準化：freq_axis は float 1D、data は (freq, time) を想定
            freq_axis = np.asarray(temp.freq_axis, dtype=float).copy()
            data = np.asarray(temp.data, dtype=float)

            # 念のため形状整合（通常FITSは (n_freq, n_time) のはず）
            if data.shape[0] != len(freq_axis) and data.shape[1] == len(freq_axis):
                data = data.T
            if data.shape[0] != len(freq_axis):
                raise ValueError(
                    f"データ形状と周波数軸が整合しません: data={data.shape}, len(freq)={len(freq_axis)}"
                )

            # 周波数は「高い方が上」を維持するため、freq を降順にそろえる
            if np.any(np.diff(freq_axis) > 0):  # 昇順の可能性
                sidx = np.argsort(freq_axis)[::-1]
                freq_axis = freq_axis[sidx]
                data = data[sidx, :]

            file_data.append(
                {
                    "file": file_path.name,
                    "header": temp.header,
                    "time_axis": list(temp.time_axis),  # datetime のリスト
                    "freq_axis": freq_axis,             # 1D float
                    "data": data,                       # (freq, time)
                    "time_step": float(temp.time_step),
                }
            )

            temp.hdulist.close()

        # -----------------------------
        # 2) グローバル時間グリッドを作る（全ファイルを統合）
        #    - dt は最小 time_step を採用（ズレは丸めで吸収）
        # -----------------------------
        dt_global = min(fd["time_step"] for fd in file_data if fd["time_step"] > 0)
        t0 = min(fd["time_axis"][0] for fd in file_data)
        t1 = max(fd["time_axis"][-1] for fd in file_data)

        total_sec = (t1 - t0).total_seconds()
        n_time_global = int(round(total_sec / dt_global)) + 1

        self.time_axis = [t0 + timedelta(seconds=i * dt_global) for i in range(n_time_global)]
        print(f"\n[Time] Global grid: start={self.time_axis[0]}  end={self.time_axis[-1]}")
        print(f"[Time] dt_global={dt_global} s, n_time={len(self.time_axis)}")

        # numpy用（インデックス化）
        t0_np = np.datetime64(t0, "us")
        dt_us = int(round(dt_global * 1e6))
        if dt_us <= 0:
            raise ValueError(f"不正な dt_global: {dt_global}")

        # -----------------------------
        # 3) グローバル周波数軸（union）を作る
        #    - _56.fit と _63.fit の周波数帯域差を同一時刻で結合するため union にする
        #    - 微小な丸め差を吸収するため rounding して一意化
        # -----------------------------
        FREQ_ROUND_DECIMALS = 3  # 必要なら 2～4 程度で調整

        from collections import defaultdict
        freq_bucket = defaultdict(list)
        for fd in file_data:
            fr = np.round(fd["freq_axis"].astype(float), FREQ_ROUND_DECIMALS)
            for r, a in zip(fr, fd["freq_axis"].astype(float)):
                freq_bucket[float(r)].append(float(a))

        freq_keys = np.array(sorted(freq_bucket.keys()), dtype=float)[::-1]  # 降順
        self.freq_axis = np.array([np.mean(freq_bucket[k]) for k in freq_keys], dtype=float)

        # freq_key -> row index
        freq_index = {k: i for i, k in enumerate(freq_keys)}

        print(f"\n[Freq] Global union channels: n_freq={len(self.freq_axis)}")
        print(f"[Freq] Range: {np.nanmin(self.freq_axis):.3f} - {np.nanmax(self.freq_axis):.3f} MHz")

        # -----------------------------
        # 4) (freq, time) のグローバル配列を確保し、各ファイルを埋め込む
        #    - 欠損は NaN のまま保持（時間ギャップ・帯域ギャップを自然に表現）
        #    - 同一セルに複数値が入る場合は max を採用（上書きより安全）
        # -----------------------------
        merged = np.full((len(self.freq_axis), len(self.time_axis)), np.nan, dtype=float)

        for fd in file_data:
            # 時間インデックス化
            t_arr_np = np.array(fd["time_axis"], dtype="datetime64[us]")
            dt_from_t0_us = (t_arr_np - t0_np).astype("timedelta64[us]").astype(np.int64)
            idx_t = np.rint(dt_from_t0_us / dt_us).astype(int)

            valid_t = (idx_t >= 0) & (idx_t < merged.shape[1])
            idx_t = idx_t[valid_t]
            if idx_t.size == 0:
                continue

            data = fd["data"][:, valid_t]  # (freq, time_valid)

            # 周波数インデックス化
            fr_key = np.round(fd["freq_axis"].astype(float), FREQ_ROUND_DECIMALS)
            idx_f = np.array([freq_index[float(k)] for k in fr_key], dtype=int)

            # 埋め込み（freqごとに）
            for local_f, global_f in enumerate(idx_f):
                target = merged[global_f, idx_t]
                new = data[local_f, :]

                # NaN を壊さずに max 合成（両方有限なら max、片方NaNなら他方）
                out = target.copy()
                t_nan = np.isnan(target)
                n_nan = np.isnan(new)

                # target が NaN で new が有限 -> new
                out[t_nan & ~n_nan] = new[t_nan & ~n_nan]
                # target が有限で new が有限 -> max
                both = (~t_nan) & (~n_nan)
                out[both] = np.maximum(target[both], new[both])
                # target 有限 & new NaN -> target のまま
                # 両方 NaN -> NaN のまま

                merged[global_f, idx_t] = out

            print(
                f"  embedded: {fd['file']}  "
                f"time[{fd['time_axis'][0].strftime('%H:%M:%S')}..{fd['time_axis'][-1].strftime('%H:%M:%S')}]  "
                f"freq[{np.nanmin(fd['freq_axis']):.2f}..{np.nanmax(fd['freq_axis']):.2f}]"
            )

        self.data = merged

        # -----------------------------
        # 5) 表示・メタデータ（最初のファイルを代表にしつつ、サイズはマージ後に合わせる）
        # -----------------------------
        self.header = file_data[0]["header"]
        self.extract_metadata()  # instrument / location / bunit / frqfile 等を得る

        # マージ後に合わせて上書き
        self.n_freq = len(self.freq_axis)
        self.n_time = len(self.time_axis)
        self.time_step = dt_global

        # 表示用（時刻表示はマージ後の端点に）
        self.date_obs = self.time_axis[0].strftime("%Y/%m/%d")
        self.time_obs = self.time_axis[0].strftime("%H:%M:%S.%f")[:-3]
        self.time_end = self.time_axis[-1].strftime("%H:%M:%S")

        print(f"\nマージ結果:")
        print(f"  data shape (freq, time): {self.data.shape}")
        print(f"  time range: {self.time_axis[0]}  -  {self.time_axis[-1]}")
        print(f"  freq range: {np.nanmin(self.freq_axis):.2f} - {np.nanmax(self.freq_axis):.2f} MHz")
        print(f"  dt_global: {self.time_step} s")
        print(f"  filled ratio: {np.isfinite(self.data).sum() / self.data.size:.3f}")

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
        データの前処理と背景除去（マージ後のNaNを考慮して nan系統計を使用）
        """
        self.processed_data = self.data.astype(float)

        print(f"\n背景除去処理: {background_method}")

        if background_method == 'median_time':
            # 時間方向の中央値を背景とする（NaNを無視）
            background = np.nanmedian(self.processed_data, axis=1, keepdims=True)
            self.processed_data = self.processed_data - background
            print("  - 各周波数での時間中央値を減算（nanmedian）")

            low_intensity_mask = self.processed_data <= intensity_threshold
            self.processed_data[low_intensity_mask] = np.nan

        elif background_method == 'percentile':
            # パーセンタイル法（NaNを無視）
            background = np.nanpercentile(self.processed_data, 25, axis=1, keepdims=True)
            self.processed_data = self.processed_data - background
            print("  - 25パーセンタイル値を背景として減算（nanpercentile）")

        elif background_method == 'rolling_median':
            # 移動中央値法（NaNがある場合は行中央値で埋めてから実施）
            window_size = 10  # 25秒間（100点 × 0.25秒）
            self.processed_data_bg = np.zeros_like(self.processed_data)
            for i in range(self.n_freq):
                row = self.processed_data[i, :].copy()
                row_med = np.nanmedian(row)
                if not np.isfinite(row_med):
                    row_med = 0.0
                row = np.where(np.isnan(row), row_med, row)
                background_rolling = median_filter(row, size=window_size)
                self.processed_data_bg[i, :] = row - background_rolling
            self.processed_data = self.processed_data_bg
            print(f"  - 移動中央値（窓幅: {window_size*self.time_step:.1f}秒）を減算")

        elif background_method == 'quiet_time':
            quiet_samples = 100
            background = np.nanmean(self.processed_data[:, :quiet_samples], axis=1, keepdims=True)
            self.processed_data = self.processed_data - background
            print(f"  - 最初の{quiet_samples*self.time_step:.1f}秒間を静穏時として減算")

        # 負の値を0にクリップ（NaNは保持）
        self.processed_data = np.where(np.isnan(self.processed_data), np.nan, np.clip(self.processed_data, 0, None))

        # スパイクノイズの除去（有限・正のみで統計を取る）
        pos = self.processed_data[np.isfinite(self.processed_data) & (self.processed_data > 0)]
        if pos.size > 0:
            threshold = np.percentile(pos, 99.5)
            self.processed_data = np.where(
                np.isnan(self.processed_data), np.nan, np.clip(self.processed_data, 0, threshold)
            )
            print(f"  - スパイク抑制: clip to 99.5 percentile = {threshold:.2f}")
        else:
            print("  - スパイク抑制: 正の有限値が無いためスキップ")

        finite = self.processed_data[np.isfinite(self.processed_data)]
        if finite.size > 0:
            print(f"  - 処理後のデータ範囲（finite）: {finite.min():.2f} - {finite.max():.2f}")
        else:
            print("  - 処理後データ: finite値なし")    
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
            # processed_dataを元のdataで初期化（Rawdataをそのまま使用）
            self.processed_data = self.data.astype(float)
        
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

        # ノイズ軽減（ノイズ除去なしの場合はRawdataを使用）
        if background_method is None:
            d_filt = d_sel.astype(float)
        else:
            d_filt = median_filter(d_sel.astype(float), size=med_filter_size)

        # 図を作成
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # カラーバー範囲をデータ統計情報から設定
        valid_data = d_filt[d_filt > 0]
        if len(valid_data) > 0:
            if background_method is None:
                # vmin = np.percentile(valid_data, 1)
                # vmax = np.percentile(valid_data, 99)
                vmin, vmax = 130, 185
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
        ax.set_yscale('log')
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(SecondLocator(interval=time_tick_sec))
        ax.yaxis.set_major_locator(MultipleLocator(freq_tick_mhz))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
        
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
        
        filename = f'/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/e-Callisto/output/Australia-ASSA_{date_str}_{time_start_str}_{time_end_str}_{background_method}.png'
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
        # file_paths = [
        #     '/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_031500_62.fit',
        #     '/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_033000_62.fit'
        # ]
        file_paths = ["/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/e-Callisto/Rawdata/Australia-ASSA_20240621_232959_63.fit",
                      "/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/e-Callisto/Rawdata/Australia-ASSA_20240621_233000_56.fit",
                      "/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/e-Callisto/Rawdata/Australia-ASSA_20240621_234459_63.fit",
                      "/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/e-Callisto/Rawdata/Australia-ASSA_20240621_234500_56.fit",
                      "/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/e-Callisto/Rawdata/Australia-ASSA_20240622_000000_56.fit",
                      "/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/e-Callisto/Rawdata/Australia-ASSA_20240622_000000_63.fit"]
        
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
        
        start_time = "2024-06-21T23:35:00"        
        end_time = "2024-06-22T00:05:00"
            
        min_frequency = 20
        max_frequency = 200
        
        # ノイズ除去方法の選択
        print("\nノイズ除去方法を選択してください:")
        print("0. ノイズ除去しない")
        print("1. median_time (時間方向中央値)")
        print("2. percentile (パーセンタイル法)")
        print("3. rolling_median (移動中央値)")
        print("4. quiet_time (静穏時間帯)")
        
        noise_choice = input("選択してください (0-4) [default: 0]: ").strip()
        if noise_choice == '':
            noise_choice = '0'
            
        background_methods = {
            '0': None,  # ノイズ除去しない
            '1': 'median_time',
            '2': 'percentile',
            '3': 'rolling_median',
            '4': 'quiet_time'
        }
        
        background_method = background_methods.get(noise_choice, 'median_time')
        
        time_tick_sec = 5*60
        freq_tick_mhz = 10
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