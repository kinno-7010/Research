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

class eCallistoSpectrum:
    """e-Callistoデータの処理とダイナミックスペクトル生成クラス"""
    
    def __init__(self, fits_path):
        """
        Parameters
        ----------
        fits_path : str or Path
            FITSファイルへのパス
        """
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
        print(f"周波数範囲: {self.freq_axis_min:.2f} - {self.freq_axis_max:.2f} MHz")
        print(f"時間分解能: {self.time_step} 秒")
        print(f"周波数チャンネル数: {self.n_freq}")
        
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
                
                # Alaska Cohoeの特別処理（既知の周波数範囲）
                if 'ALASKA' in self.instrument.upper() and 'COHOE' in self.instrument.upper():
                    print("  Alaska Cohoe ステーションを検出 - 既知の周波数範囲を適用")
                    # Quick Lookから判明した実際の周波数範囲
                    self.freq_axis = np.linspace(90, 8, self.n_freq)  # 90-8 MHz
            else:
                # デフォルト値（45-870MHzの標準的なe-Callisto範囲から推定）
                warnings.warn("周波数軸の情報が不完全です。デフォルト値を使用します。")
                self.freq_axis = np.linspace(200, 1, self.n_freq)  # デフォルト
        
        # 周波数範囲を確定
        self.freq_axis_min = 10
        self.freq_axis_max = 90
        
        
    def create_time_axis(self):
        """時間軸を生成"""
        
        # 観測開始時刻を基準にする
        base_datetime = datetime.strptime(f"{self.date_obs} {self.time_obs}", 
                                         "%Y/%m/%d %H:%M:%S.%f")
        
        # 時間配列を生成
        time_seconds = np.arange(self.n_time) * self.time_step
        self.time_axis = [base_datetime + timedelta(seconds=float(t)) for t in time_seconds]
        
    def preprocess_data(self, background_method):
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
        """
        self.processed_data = self.data.astype(float)
        
        print(f"\n背景除去処理: {background_method}")
        
        if background_method == 'median_time':
            # 時間方向の中央値を背景とする
            background = np.median(self.processed_data, axis=1, keepdims=True)
            self.processed_data = self.processed_data - background
            print("  - 各周波数での時間中央値を減算")
            
        elif background_method == 'percentile':
            # パーセンタイル法
            background = np.percentile(self.processed_data, 25, axis=1, keepdims=True)
            self.processed_data = self.processed_data - background
            print("  - 25パーセンタイル値を背景として減算")
            
        elif background_method == 'rolling_median':
            # 移動中央値法
            window_size = 100  # 25秒間（100点 × 0.25秒）
            self.processed_data_bg = np.zeros_like(self.processed_data)
            for i in range(self.n_freq):
                background_rolling = median_filter(self.processed_data[i, :], size=window_size)
                self.processed_data_bg[i, :] = self.processed_data[i, :] - background_rolling
            self.processed_data = self.processed_data_bg
            print(f"  - 移動中央値（窓幅: {window_size*self.time_step:.1f}秒）を減算")
            
        elif background_method == 'quiet_time':
            # 静穏時間帯を選択して背景とする
            quiet_samples = 100
            background = np.mean(self.processed_data[:, :quiet_samples], axis=1, keepdims=True)
            self.processed_data = self.processed_data - background
            print(f"  - 最初の{quiet_samples*self.time_step:.1f}秒間を静穏時として減算")
        
        # 負の値を0にクリップ
        self.processed_data = np.clip(self.processed_data, 0, None)
        
        # スパイクノイズの除去
        threshold = np.percentile(self.processed_data[self.processed_data > 0], 99.5)
        self.processed_data = np.clip(self.processed_data, 0, threshold)
        
        print(f"  - 処理後のデータ範囲: {self.processed_data.min():.2f} - {self.processed_data.max():.2f}")
    
    def plot_dynamic_spectrum(self):
        """
        ダイナミックスペクトルをプロット
        """
        
        # 時間軸を生成
        self.create_time_axis()
        
        # データの前処理
        background_method = 'percentile'
        self.preprocess_data(background_method)
        
        # 図を作成
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # データの統計情報からカラースケールを設定
        valid_data = self.processed_data[self.processed_data > 0]
        if len(valid_data) > 0:
            vmin = 18
            vmax = 0
        else:
            vmin, vmax = 0, 1
            
        # ダイナミックスペクトルをプロット
        # extent: [xmin, xmax, ymin, ymax]
        extent = [mdates.date2num(self.time_axis[0]), 
                 mdates.date2num(self.time_axis[-1]),
                 self.freq_axis_min,
                 self.freq_axis_max]
        
        im = ax.imshow(self.processed_data, 
                      aspect='auto',
                      extent=extent,
                      cmap='viridis',
                      vmin=vmin,
                      vmax=vmax,
                      interpolation='nearest')
        
        # カラーバーを追加
        cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.5)
        unit_label = 'SFU' if self.is_calibrated else 'digits'
        cbar.set_label(f'Intensity [{unit_label}] (background subtracted)', 
                      )
        
        # x軸を時刻形式で表示
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
        
        # ラベルとタイトル
        ax.set_xlabel('Time [UT]', fontsize=12)
        ax.set_ylabel('Frequency [MHz]', fontsize=12)
        
        
        # タイトル
        title = f"e-Callisto Dynamic Spectrum (Background Subtracted {background_method})\n"
        title += f"{self.instrument} @ {self.location}\n"
        title += f"{self.date_obs} {self.time_obs} - {self.time_end} UT"
        if self.frqfile:
            title += f" (FRQ: {self.frqfile})"
        ax.set_title(title, fontsize=14)
        
        # グリッド
        ax.grid(True, alpha=0.3, linestyle='--', color='white')
        
        # 観測地点情報
        if self.obs_lat and self.obs_lon:
            info_text = f"Location: {abs(self.obs_lat):.2f}°{'N' if self.obs_lat > 0 else 'S'}, "
            info_text += f"{abs(self.obs_lon):.2f}°{'E' if self.obs_lon > 0 else 'W'}"
            if self.obs_alt:
                info_text += f", Alt: {self.obs_alt:.1f}m"
            info_text += f"\nData: {'Calibrated' if self.is_calibrated else 'Uncalibrated'}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=9, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # x軸ラベルを回転
        # plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # レイアウトを調整
        plt.tight_layout()
        
        # 表示
        plt.show()
        
        return fig, ax
        
    # def analyze_burst_statistics(self):
    #     """バースト統計情報の解析"""
        
    #     # 背景除去済みデータを使用
    #     if not hasattr(self, 'processed_data'):
    #         self.create_time_axis()
    #         self.preprocess_data(background_method)
        
    #     # 統計情報を計算
    #     stats = {
    #         'max_intensity': np.max(self.processed_data),
    #         'mean_intensity': np.mean(self.processed_data),
    #         'std_intensity': np.std(self.processed_data),
    #         'snr_max': np.max(self.processed_data) / (np.std(self.processed_data) + 1e-10)
    #     }
        
    #     # 最大強度の時刻と周波数
    #     max_idx = np.unravel_index(np.argmax(self.processed_data), self.processed_data.shape)
    #     stats['max_freq'] = self.freq_axis[max_idx[0]]
    #     stats['max_time'] = self.time_axis[max_idx[1]]
        
    #     unit_label = 'SFU' if self.is_calibrated else 'digits'
        
    #     print("\n" + "=" * 60)
    #     print("バースト統計情報")
    #     print("=" * 60)
    #     print(f"最大強度: {stats['max_intensity']:.1f} {unit_label}")
    #     print(f"平均強度: {stats['mean_intensity']:.1f} {unit_label}")
    #     print(f"標準偏差: {stats['std_intensity']:.1f} {unit_label}")
    #     print(f"最大S/N比: {stats['snr_max']:.1f}")
    #     print(f"最大強度の周波数: {stats['max_freq']:.2f} MHz")
    #     print(f"最大強度の時刻: {stats['max_time'].strftime('%H:%M:%S.%f')[:-3]} UT")
        
    #     return stats
        
    def close(self):
        """FITSファイルを閉じる"""
        self.hdulist.close()


def main():
    """メイン関数"""
    
    # 固定パス
    fits_path = '/mnt/d/wsl/home/kinno-7010/Research/RadioData/e-Callisto/Rawdata/ALASKA-COHOE_20220613_031458_62.fit'
    
    try:
        print("\n" + "=" * 60)
        print("e-Callisto Dynamic Spectrum Generator")
        print("複数HDU対応版")
        print("=" * 60)
        
        # e-Callistoスペクトラムオブジェクトを作成
        spectrum = eCallistoSpectrum(fits_path)
        
        # バースト統計情報を解析
        # stats = spectrum.analyze_burst_statistics()
        
        # ダイナミックスペクトルをプロット
        fig, ax = spectrum.plot_dynamic_spectrum()
        
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