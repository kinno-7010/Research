#!/usr/bin/env python3
"""
太陽コロナ3次元再構成データ解析システム - メインプログラム
Solar Corona 3D Reconstruction Data Analysis System - Main Program

このプログラムは、太陽物理学研究における日常的なデータ解析作業を
自動化し、効率的な研究活動を支援することを目的としています。

主要機能：
1. バッチ処理による複数データセットの自動解析
2. CME（コロナ質量放出）イベントの検出と追跡
3. 時系列解析による太陽活動の監視
4. 物理量の相関解析と統計処理
5. 自動レポート生成と結果の保存

使用方法：
    python solar_corona_analysis_main.py --dataset 002 --analysis all
    python solar_corona_analysis_main.py --batch --start 001 --end 010

Author: Solar Physics Research Team
Date: 2024
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from scipy import signal, ndimage
from scipy.optimize import curve_fit
import h5py

# 研究用モジュールのインポート
# これらは同じディレクトリにあることを前提としています
from tomo_hdf_read import (
    tomo_hdf_read,
    read_all_magnetic_components,
    SolarData,
    DataType,
    get_data_directory
)
from example_hdf_read import SolarDataVisualizer


class AnalysisType(Enum):
    """実行する解析タイプを定義
    
    各解析タイプは特定の物理現象や研究課題に対応しています。
    これにより、必要な解析だけを選択的に実行できます。
    """
    BASIC = "basic"              # 基本統計と可視化
    CME_DETECTION = "cme"         # CMEイベント検出
    TIME_SERIES = "timeseries"    # 時系列解析
    CORRELATION = "correlation"   # 物理量相関解析
    MAGNETIC = "magnetic"         # 磁場構造解析
    ALL = "all"                  # すべての解析


@dataclass
class CMEEvent:
    """CMEイベントの特徴を記録するデータクラス
    
    コロナ質量放出イベントの物理的特性を定量的に記述します。
    これらのパラメータは、CMEの発生機構や宇宙天気への影響を
    理解する上で重要な情報となります。
    """
    time_index: int               # イベント発生時刻インデックス
    time_julian: float           # ユリウス日での時刻
    location_lat: float          # 緯度 [度]
    location_lon: float          # 経度 [度]
    location_rad: float          # 高度 [太陽半径]
    mass_excess: float           # 質量超過 [g]
    velocity_radial: float       # 動径速度 [km/s]
    magnetic_flux: float         # 磁束 [Mx]
    energy_kinetic: float        # 運動エネルギー [erg]
    energy_magnetic: float       # 磁気エネルギー [erg]
    confidence: float            # 検出信頼度 [0-1]
    
    def to_dict(self) -> Dict:
        """辞書形式に変換（JSON保存用）"""
        return asdict(self)
    
    def get_severity_score(self) -> float:
        """CMEの深刻度スコアを計算
        
        地球への影響度を評価するための総合的な指標を提供します。
        速度、質量、エネルギーを考慮した重み付けスコアです。
        """
        # 速度の寄与（300-2000 km/sを0-1に正規化）
        v_score = np.clip((self.velocity_radial - 300) / 1700, 0, 1)
        
        # 質量の寄与（1e14-1e16 gを0-1に正規化）
        m_score = np.clip(np.log10(self.mass_excess / 1e14) / 2, 0, 1)
        
        # エネルギーの寄与（1e30-1e33 ergを0-1に正規化）
        e_total = self.energy_kinetic + self.energy_magnetic
        e_score = np.clip(np.log10(e_total / 1e30) / 3, 0, 1)
        
        # 重み付け平均（速度が最も重要）
        return 0.5 * v_score + 0.3 * m_score + 0.2 * e_score


class SolarCoronaAnalyzer:
    """太陽コロナデータの高度な解析を行うクラス
    
    このクラスは、基本的な可視化機能を超えて、科学的な解析と
    イベント検出を行います。CMEの自動検出、磁場構造の定量化、
    時系列解析などの研究用機能を提供します。
    """
    
    def __init__(self, solar_data: SolarData, output_dir: Path = None):
        """
        Parameters
        ----------
        solar_data : SolarData
            解析対象のデータ
        output_dir : Path, optional
            結果を保存するディレクトリ
        """
        self.data = solar_data
        self.output_dir = output_dir or Path("./analysis_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 解析結果を保存する辞書
        self.results = {
            'statistics': {},
            'cme_events': [],
            'time_series': {},
            'correlations': {},
            'magnetic_analysis': {}
        }
        
        # ロガーの設定
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def detect_cme_events(self, 
                          density_threshold: float = 2.0,
                          velocity_threshold: float = 300.0) -> List[CMEEvent]:
        """CMEイベントを自動検出
        
        密度と速度の時間変化を解析し、CMEの候補を検出します。
        この手法は、密度の急激な増加と外向きの高速流を特徴とする
        CMEの物理的特性に基づいています。
        
        Parameters
        ----------
        density_threshold : float
            密度増加の閾値（背景密度に対する倍率）
        velocity_threshold : float
            速度の閾値 [km/s]
            
        Returns
        -------
        events : List[CMEEvent]
            検出されたCMEイベントのリスト
        """
        
        self.logger.info("CMEイベント検出を開始")
        events = []
        
        if self.data.density is None or len(self.data.time) < 2:
            self.logger.warning("CME検出に必要なデータが不足しています")
            return events
        
        # 時系列での密度変化を計算
        # 各時刻での平均密度を計算（高度2-5 Rs の範囲）
        r_min_idx = np.argmin(np.abs(self.data.rad - 2.0))
        r_max_idx = np.argmin(np.abs(self.data.rad - 5.0))
        
        mean_density_time = np.zeros(len(self.data.time))
        for t in range(len(self.data.time)):
            density_slice = self.data.density[r_min_idx:r_max_idx, :, :, t]
            mean_density_time[t] = np.nanmean(density_slice)
        
        # 背景密度の推定（メディアンフィルタ）
        if len(mean_density_time) > 5:
            background_density = signal.medfilt(mean_density_time, kernel_size=5)
        else:
            background_density = np.median(mean_density_time) * np.ones_like(mean_density_time)
        
        # 密度超過を検出
        density_excess = mean_density_time / background_density
        
        # CME候補時刻を特定
        cme_candidates = np.where(density_excess > density_threshold)[0]
        
        self.logger.info(f"CME候補時刻: {len(cme_candidates)}個")
        
        # 各候補について詳細解析
        for t_idx in cme_candidates:
            if t_idx == 0:
                continue  # 速度計算のため、最初の時刻はスキップ
            
            # 密度増加領域を特定
            density_current = self.data.density[:, :, :, t_idx]
            density_previous = self.data.density[:, :, :, t_idx - 1]
            
            # 密度変化率
            density_change = (density_current - density_previous) / density_previous
            
            # 最大変化位置を特定
            max_change_idx = np.unravel_index(
                np.nanargmax(density_change), density_change.shape
            )
            r_idx, lat_idx, lon_idx = max_change_idx
            
            # 物理量を計算
            # 質量超過の推定（簡略化）
            volume_element = (
                4 * np.pi * self.data.rad[r_idx]**2 * 
                6.96e10**3  # cm^3に変換
            )
            mass_excess = (
                density_excess[t_idx] * 
                np.nanmean(density_current) * 
                1.67e-24 *  # 陽子質量
                volume_element * 
                0.01  # 体積の一部と仮定
            )
            
            # 速度の推定（時間差分から）
            if t_idx < len(self.data.time) - 1:
                dt = self.data.time[t_idx + 1] - self.data.time[t_idx]
                dr = 0.5  # 仮定: 0.5 Rs の移動
                velocity_radial = dr * 6.96e5 / (dt * 86400)  # km/s
            else:
                velocity_radial = velocity_threshold  # デフォルト値
            
            # 速度条件をチェック
            if velocity_radial < velocity_threshold:
                continue
            
            # 磁場関連の計算
            magnetic_flux = 0.0
            energy_magnetic = 0.0
            if self.data.br is not None:
                b_local = self.data.br[r_idx, lat_idx, lon_idx, t_idx]
                area = 4 * np.pi * (self.data.rad[r_idx] * 6.96e10)**2 / 100  # 一部の面積
                magnetic_flux = np.abs(b_local) * area
                
                # 磁気エネルギー（簡略化）
                b_magnitude = np.sqrt(
                    self.data.br[r_idx, lat_idx, lon_idx, t_idx]**2 +
                    (self.data.bt[r_idx, lat_idx, lon_idx, t_idx]**2 
                     if self.data.bt is not None else 0) +
                    (self.data.bp[r_idx, lat_idx, lon_idx, t_idx]**2 
                     if self.data.bp is not None else 0)
                )
                energy_magnetic = b_magnitude**2 / (8 * np.pi) * volume_element
            
            # 運動エネルギー
            energy_kinetic = 0.5 * mass_excess * (velocity_radial * 1e5)**2
            
            # 信頼度の計算（複数の指標を統合）
            confidence = np.clip(
                0.3 * (density_excess[t_idx] / density_threshold) +
                0.3 * (velocity_radial / velocity_threshold) +
                0.4 * (1.0 if energy_kinetic > 1e30 else energy_kinetic / 1e30),
                0, 1
            )
            
            # CMEイベントを記録
            event = CMEEvent(
                time_index=t_idx,
                time_julian=self.data.time[t_idx],
                location_lat=self.data.lat[lat_idx],
                location_lon=self.data.lon[lon_idx],
                location_rad=self.data.rad[r_idx],
                mass_excess=mass_excess,
                velocity_radial=velocity_radial,
                magnetic_flux=magnetic_flux,
                energy_kinetic=energy_kinetic,
                energy_magnetic=energy_magnetic,
                confidence=confidence
            )
            
            events.append(event)
            self.logger.info(
                f"CMEイベント検出: t={t_idx}, "
                f"位置=({event.location_lat:.1f}°, {event.location_lon:.1f}°, "
                f"{event.location_rad:.1f} Rs), "
                f"速度={event.velocity_radial:.0f} km/s, "
                f"信頼度={event.confidence:.2f}"
            )
        
        # 重複を除去（時間的に近いイベントを統合）
        events = self._merge_duplicate_events(events)
        
        self.results['cme_events'] = events
        return events
    
    def _merge_duplicate_events(self, events: List[CMEEvent], 
                                time_window: int = 2) -> List[CMEEvent]:
        """時間的に近いCMEイベントを統合
        
        同じ物理現象が複数回検出されることを防ぐため、
        時間的・空間的に近いイベントを一つに統合します。
        """
        if len(events) <= 1:
            return events
        
        merged = []
        used = set()
        
        for i, event1 in enumerate(events):
            if i in used:
                continue
            
            # 近いイベントを探す
            nearby = [event1]
            for j, event2 in enumerate(events[i+1:], start=i+1):
                if j in used:
                    continue
                
                # 時間差をチェック
                if abs(event2.time_index - event1.time_index) <= time_window:
                    # 空間的な距離もチェック
                    dlat = event2.location_lat - event1.location_lat
                    dlon = event2.location_lon - event1.location_lon
                    distance = np.sqrt(dlat**2 + dlon**2)
                    
                    if distance < 30:  # 30度以内
                        nearby.append(event2)
                        used.add(j)
            
            # 最も信頼度の高いイベントを選択
            best_event = max(nearby, key=lambda e: e.confidence)
            merged.append(best_event)
            used.add(i)
        
        return merged
    
    def analyze_time_series(self) -> Dict[str, np.ndarray]:
        """時系列解析の実行
        
        各物理量の時間変化を解析し、周期性、トレンド、
        異常値などを検出します。これは太陽活動の
        長期的な変動を理解する上で重要です。
        """
        
        self.logger.info("時系列解析を開始")
        time_series = {}
        
        # 各高度での平均値の時間変化
        radial_levels = [1.5, 2.5, 5.0, 10.0]  # Rs
        
        for r_target in radial_levels:
            r_idx = np.argmin(np.abs(self.data.rad - r_target))
            actual_r = self.data.rad[r_idx]
            
            # 密度の時系列
            if self.data.density is not None:
                density_series = np.nanmean(
                    self.data.density[r_idx, :, :, :], axis=(0, 1)
                )
                time_series[f'density_r{actual_r:.1f}'] = density_series
            
            # 磁場強度の時系列
            b_mag = self.data.get_magnetic_field_magnitude()
            if b_mag is not None:
                b_series = np.nanmean(b_mag[r_idx, :, :, :], axis=(0, 1))
                time_series[f'b_magnitude_r{actual_r:.1f}'] = b_series
        
        # 全球平均の時系列
        if self.data.density is not None:
            global_density = np.nanmean(
                self.data.density, axis=(0, 1, 2)
            )
            time_series['density_global'] = global_density
            
            # トレンド解析（線形フィット）
            if len(self.data.time) > 2:
                coeffs = np.polyfit(self.data.time, global_density, 1)
                trend = np.poly1d(coeffs)
                time_series['density_trend'] = trend(self.data.time)
                
                self.logger.info(
                    f"密度トレンド: {coeffs[0]:.2e} cm^-3/day"
                )
        
        # フーリエ解析（周期性の検出）
        if len(self.data.time) > 10:
            for key, series in time_series.items():
                if 'trend' not in key and len(series) > 10:
                    # データのデトレンド
                    detrended = signal.detrend(series)
                    
                    # FFT
                    fft_vals = np.fft.fft(detrended)
                    fft_freq = np.fft.fftfreq(len(detrended), 
                                             d=np.mean(np.diff(self.data.time)))
                    
                    # パワースペクトル
                    power = np.abs(fft_vals)**2
                    
                    # 主要な周期を特定
                    positive_freq = fft_freq[1:len(fft_freq)//2]
                    positive_power = power[1:len(power)//2]
                    
                    if len(positive_power) > 0:
                        peak_idx = np.argmax(positive_power)
                        dominant_period = 1.0 / positive_freq[peak_idx]
                        
                        self.logger.info(
                            f"{key}の主要周期: {dominant_period:.1f} days"
                        )
        
        self.results['time_series'] = time_series
        return time_series
    
    def analyze_correlations(self) -> Dict[str, float]:
        """物理量間の相関解析
        
        密度と磁場強度、プラズマベータと速度など、
        物理量間の相関を計算します。これらの相関は、
        物理過程の理解に重要な手がかりを提供します。
        """
        
        self.logger.info("相関解析を開始")
        correlations = {}
        
        # 解析する領域を限定（計算効率のため）
        r_min_idx = np.argmin(np.abs(self.data.rad - 1.5))
        r_max_idx = np.argmin(np.abs(self.data.rad - 5.0))
        
        # 密度と磁場強度の相関
        if self.data.density is not None:
            b_mag = self.data.get_magnetic_field_magnitude()
            if b_mag is not None:
                density_flat = self.data.density[r_min_idx:r_max_idx, :, :, 0].flatten()
                b_mag_flat = b_mag[r_min_idx:r_max_idx, :, :, 0].flatten()
                
                # NaNを除去
                valid = ~(np.isnan(density_flat) | np.isnan(b_mag_flat))
                if np.sum(valid) > 100:
                    corr = np.corrcoef(
                        np.log10(density_flat[valid] + 1e-10),
                        np.log10(b_mag_flat[valid] + 1e-10)
                    )[0, 1]
                    correlations['density_vs_b_magnitude'] = corr
                    
                    self.logger.info(
                        f"密度-磁場強度相関: {corr:.3f}"
                    )
        
        # プラズマベータと各成分の相関
        beta = self.data.get_plasma_beta(temperature=2e6)
        if beta is not None and self.data.density is not None:
            beta_flat = beta[r_min_idx:r_max_idx, :, :, 0].flatten()
            density_flat = self.data.density[r_min_idx:r_max_idx, :, :, 0].flatten()
            
            valid = ~(np.isnan(beta_flat) | np.isnan(density_flat) | 
                     np.isinf(beta_flat))
            if np.sum(valid) > 100:
                # ベータ値を対数空間で制限
                log_beta = np.log10(np.clip(beta_flat[valid], 1e-3, 1e3))
                log_density = np.log10(density_flat[valid] + 1e-10)
                
                corr = np.corrcoef(log_beta, log_density)[0, 1]
                correlations['beta_vs_density'] = corr
                
                self.logger.info(
                    f"プラズマβ-密度相関: {corr:.3f}"
                )
        
        # 磁場成分間の相関
        if all(comp is not None for comp in [self.data.br, self.data.bt]):
            br_flat = self.data.br[r_min_idx:r_max_idx, :, :, 0].flatten()
            bt_flat = self.data.bt[r_min_idx:r_max_idx, :, :, 0].flatten()
            
            valid = ~(np.isnan(br_flat) | np.isnan(bt_flat))
            if np.sum(valid) > 100:
                corr = np.corrcoef(br_flat[valid], bt_flat[valid])[0, 1]
                correlations['br_vs_bt'] = corr
                
                self.logger.info(
                    f"Br-Bt相関: {corr:.3f}"
                )
        
        self.results['correlations'] = correlations
        return correlations
    
    def analyze_magnetic_structure(self) -> Dict[str, Any]:
        """磁場構造の詳細解析
        
        磁場のトポロジー、開いた磁力線の領域（コロナホール）、
        電流シートの位置などを特定します。
        """
        
        self.logger.info("磁場構造解析を開始")
        magnetic_analysis = {}
        
        if self.data.br is None:
            self.logger.warning("磁場データが利用できません")
            return magnetic_analysis
        
        # 開いた磁場領域の特定（コロナホール）
        # 高高度での磁場が一方向の領域
        r_high_idx = np.argmin(np.abs(self.data.rad - 2.5))
        br_high = self.data.br[r_high_idx, :, :, 0]
        
        # 磁場の極性マップ
        polarity_map = np.sign(br_high)
        
        # コロナホールの候補（強い単極磁場）
        threshold = np.percentile(np.abs(br_high), 75)
        coronal_holes = np.abs(br_high) > threshold
        
        # 連結成分解析
        labeled, n_features = ndimage.label(coronal_holes)
        
        magnetic_analysis['n_coronal_holes'] = n_features
        magnetic_analysis['coronal_hole_area'] = np.sum(coronal_holes) / coronal_holes.size
        
        self.logger.info(
            f"コロナホール: {n_features}個, "
            f"面積率: {magnetic_analysis['coronal_hole_area']:.1%}"
        )
        
        # 電流シートの検出（磁場の急激な変化）
        if self.data.bt is not None:
            # 赤道面での磁場反転を探す
            lat_eq_idx = np.argmin(np.abs(self.data.lat))
            br_eq = self.data.br[:, lat_eq_idx, :, 0]
            
            # 磁場の勾配
            grad_r = np.gradient(br_eq, axis=0)
            grad_lon = np.gradient(br_eq, axis=1)
            grad_magnitude = np.sqrt(grad_r**2 + grad_lon**2)
            
            # 電流シートの位置（勾配が大きい場所）
            current_sheet_threshold = np.percentile(grad_magnitude, 95)
            current_sheet = grad_magnitude > current_sheet_threshold
            
            magnetic_analysis['current_sheet_strength'] = np.mean(
                grad_magnitude[current_sheet]
            )
            
            self.logger.info(
                f"電流シート強度: {magnetic_analysis['current_sheet_strength']:.2e}"
            )
        
        # 磁気ヘリシティの推定（簡略化）
        # H = ∫A·B dV （Aはベクトルポテンシャル）
        # ここでは磁場のツイスト度として簡易評価
        if all(comp is not None for comp in [self.data.br, self.data.bt, self.data.bp]):
            # 磁場のツイスト（Bt/Br の比）
            with np.errstate(divide='ignore', invalid='ignore'):
                twist = np.abs(self.data.bt / (self.data.br + 1e-10))
            
            magnetic_analysis['mean_twist'] = np.nanmean(twist)
            magnetic_analysis['max_twist'] = np.nanmax(twist[np.isfinite(twist)])
            
            self.logger.info(
                f"磁場ツイスト: 平均={magnetic_analysis['mean_twist']:.2f}, "
                f"最大={magnetic_analysis['max_twist']:.2f}"
            )
        
        self.results['magnetic_analysis'] = magnetic_analysis
        return magnetic_analysis
    
    def generate_report(self, filename: str = "analysis_report.pdf"):
        """解析レポートの生成
        
        すべての解析結果を統合し、PDF形式の包括的な
        レポートを生成します。図表と解説を含みます。
        """
        
        self.logger.info(f"レポート生成開始: {filename}")
        
        pdf_path = self.output_dir / filename
        
        with PdfPages(pdf_path) as pdf:
            # タイトルページ
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.7, '太陽コロナ3次元構造解析レポート', 
                    size=24, ha='center', weight='bold')
            fig.text(0.5, 0.6, f'生成日時: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                    size=14, ha='center')
            
            # データ概要
            info_text = f"""
データセット情報:
  座標範囲:
    - 半径: {self.data.rad.min():.1f} - {self.data.rad.max():.1f} Rs
    - 緯度: {self.data.lat.min():.1f} - {self.data.lat.max():.1f}°
    - 経度: {self.data.lon.min():.1f} - {self.data.lon.max():.1f}°
    - 時間ステップ: {len(self.data.time)}
  
  利用可能な物理量:
    - 電子密度: {"○" if self.data.density is not None else "×"}
    - 磁場Br: {"○" if self.data.br is not None else "×"}
    - 磁場Bt: {"○" if self.data.bt is not None else "×"}
    - 磁場Bp: {"○" if self.data.bp is not None else "×"}
            """
            fig.text(0.1, 0.45, info_text, size=11, family='monospace',
                    verticalalignment='top')
            
            pdf.savefig(fig)
            plt.close(fig)
            
            # CMEイベントサマリー
            if self.results['cme_events']:
                fig = plt.figure(figsize=(8.5, 11))
                fig.suptitle('検出されたCMEイベント', size=16, weight='bold')
                
                # イベントテーブル
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                events_data = []
                for event in self.results['cme_events'][:10]:  # 最大10個
                    events_data.append([
                        f"{event.time_index}",
                        f"{event.location_lat:.1f}°",
                        f"{event.location_lon:.1f}°",
                        f"{event.location_rad:.1f} Rs",
                        f"{event.velocity_radial:.0f}",
                        f"{event.confidence:.2f}"
                    ])
                
                table = ax.table(
                    cellText=events_data,
                    colLabels=['時刻', '緯度', '経度', '高度', 
                              '速度[km/s]', '信頼度'],
                    cellLoc='center',
                    loc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                
                pdf.savefig(fig)
                plt.close(fig)
            
            # 時系列解析結果
            if self.results['time_series']:
                fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
                fig.suptitle('時系列解析結果', size=16, weight='bold')
                
                # 密度の時系列
                ax = axes[0]
                for key, series in self.results['time_series'].items():
                    if 'density' in key and 'trend' not in key:
                        ax.plot(self.data.time, series, label=key, alpha=0.7)
                
                ax.set_xlabel('時間 [days]')
                ax.set_ylabel('密度 [cm⁻³]')
                ax.set_yscale('log')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # 磁場の時系列
                ax = axes[1]
                for key, series in self.results['time_series'].items():
                    if 'b_magnitude' in key:
                        ax.plot(self.data.time, series, label=key, alpha=0.7)
                
                ax.set_xlabel('時間 [days]')
                ax.set_ylabel('磁場強度 [Gauss]')
                ax.set_yscale('log')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            
            # 相関解析結果
            if self.results['correlations']:
                fig = plt.figure(figsize=(8.5, 11))
                fig.suptitle('物理量相関解析', size=16, weight='bold')
                
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                corr_text = "相関係数:\n\n"
                for key, value in self.results['correlations'].items():
                    corr_text += f"  {key}: {value:.3f}\n"
                
                corr_text += """
                
解釈:
  |r| > 0.7: 強い相関
  0.4 < |r| < 0.7: 中程度の相関
  |r| < 0.4: 弱い相関
                
相関は因果関係を意味しません。
物理的解釈には注意が必要です。
                """
                
                ax.text(0.1, 0.9, corr_text, size=11, 
                       transform=ax.transAxes,
                       verticalalignment='top')
                
                pdf.savefig(fig)
                plt.close(fig)
            
            # 磁場構造解析結果
            if self.results['magnetic_analysis']:
                fig = plt.figure(figsize=(8.5, 11))
                fig.suptitle('磁場構造解析', size=16, weight='bold')
                
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                mag_text = "磁場トポロジー解析:\n\n"
                for key, value in self.results['magnetic_analysis'].items():
                    if isinstance(value, float):
                        mag_text += f"  {key}: {value:.3e}\n"
                    else:
                        mag_text += f"  {key}: {value}\n"
                
                mag_text += """

物理的意義:
  - コロナホールは太陽風の加速領域
  - 電流シートはエネルギー解放の場
  - 磁場ツイストはヘリシティを示す
  
これらの構造は宇宙天気に
直接的な影響を与えます。
                """
                
                ax.text(0.1, 0.9, mag_text, size=11,
                       transform=ax.transAxes,
                       verticalalignment='top')
                
                pdf.savefig(fig)
                plt.close(fig)
            
            # PDFメタデータ
            d = pdf.infodict()
            d['Title'] = '太陽コロナ3次元構造解析レポート'
            d['Author'] = 'Solar Physics Research System'
            d['Subject'] = 'Solar Corona Analysis'
            d['Keywords'] = 'Solar Physics, Corona, CME, Magnetic Field'
            d['CreationDate'] = datetime.now()
        
        self.logger.info(f"レポート生成完了: {pdf_path}")
        return pdf_path
    
    def save_results(self, filename: str = "analysis_results.json"):
        """解析結果をJSON形式で保存
        
        すべての数値結果を構造化された形式で保存し、
        後続の解析や他のツールでの利用を可能にします。
        """
        
        json_path = self.output_dir / filename
        
        # NumPy配列をリストに変換
        save_dict = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                save_dict[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        save_dict[key][k] = v.tolist()
                    else:
                        save_dict[key][k] = v
            elif isinstance(value, list):
                # CMEイベントリストの処理
                if key == 'cme_events':
                    save_dict[key] = [event.to_dict() for event in value]
                else:
                    save_dict[key] = value
            else:
                save_dict[key] = value
        
        with open(json_path, 'w') as f:
            json.dump(save_dict, f, indent=2, default=str)
        
        self.logger.info(f"結果を保存: {json_path}")
        return json_path


def setup_logging(log_dir: Path = None, level: int = logging.INFO):
    """ロギングシステムの設定
    
    研究活動の記録と問題診断のために、詳細なログを
    記録します。これは再現可能な研究に不可欠です。
    """
    
    if log_dir is None:
        log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # ログファイル名（タイムスタンプ付き）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"solar_analysis_{timestamp}.log"
    
    # ロガーの設定
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('main')
    logger.info(f"ロギングシステムを初期化: {log_file}")
    
    return logger


def parse_arguments():
    """コマンドライン引数の解析
    
    研究者が様々な解析シナリオを実行できるように、
    柔軟なコマンドラインインターフェースを提供します。
    """
    
    parser = argparse.ArgumentParser(
        description='太陽コロナ3次元再構成データ解析システム',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単一データセットの基本解析
  python %(prog)s --dataset 002 --analysis basic
  
  # すべての解析を実行してレポート生成
  python %(prog)s --dataset 002 --analysis all --report
  
  # バッチ処理（複数データセット）
  python %(prog)s --batch --start 001 --end 010 --analysis cme
  
  # 時系列解析のみ
  python %(prog)s --dataset 002 --analysis timeseries --output ./results
        """
    )
    
    # データ選択オプション
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        '--dataset', 
        type=str,
        help='解析するデータセット番号（例: 002）'
    )
    data_group.add_argument(
        '--batch',
        action='store_true',
        help='バッチ処理モード（複数データセット）'
    )
    
    # バッチ処理オプション
    parser.add_argument(
        '--start',
        type=str,
        default='001',
        help='バッチ処理の開始番号（デフォルト: 001）'
    )
    parser.add_argument(
        '--end',
        type=str,
        default='010',
        help='バッチ処理の終了番号（デフォルト: 010）'
    )
    
    # 解析タイプ
    parser.add_argument(
        '--analysis',
        type=str,
        choices=[t.value for t in AnalysisType],
        default='basic',
        help='実行する解析タイプ（デフォルト: basic）'
    )
    
    # 出力オプション
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./analysis_results'),
        help='結果を保存するディレクトリ（デフォルト: ./analysis_results）'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='PDFレポートを生成'
    )
    parser.add_argument(
        '--save-figures',
        action='store_true',
        help='すべての図を個別に保存'
    )
    
    # CME検出パラメータ
    parser.add_argument(
        '--cme-density-threshold',
        type=float,
        default=2.0,
        help='CME検出の密度閾値（背景密度に対する倍率、デフォルト: 2.0）'
    )
    parser.add_argument(
        '--cme-velocity-threshold',
        type=float,
        default=300.0,
        help='CME検出の速度閾値 [km/s]（デフォルト: 300.0）'
    )
    
    # その他のオプション
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='詳細なログ出力'
    )
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='GUI表示を無効化（バッチ処理用）'
    )
    
    return parser.parse_args()


def analyze_single_dataset(dataset_number: str, 
                          args: argparse.Namespace,
                          logger: logging.Logger) -> Dict:
    """単一データセットの解析
    
    指定されたデータセットに対して、要求された
    解析を実行し、結果を返します。
    
    Parameters
    ----------
    dataset_number : str
        データセット番号
    args : argparse.Namespace
        コマンドライン引数
    logger : logging.Logger
        ロガー
        
    Returns
    -------
    results : Dict
        解析結果の辞書
    """
    
    logger.info(f"データセット {dataset_number} の解析を開始")
    
    # 出力ディレクトリの作成
    output_dir = args.output / f"dataset_{dataset_number}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # データの読み込み
        logger.info("データ読み込み中...")
        solar_data = read_all_magnetic_components(dataset_number)
        
        # 解析器の初期化
        analyzer = SolarCoronaAnalyzer(solar_data, output_dir)
        
        # 可視化器の初期化（基本解析用）
        visualizer = SolarDataVisualizer(solar_data)
        
        # 解析タイプに応じた処理
        analysis_type = AnalysisType(args.analysis)
        
        if analysis_type in [AnalysisType.BASIC, AnalysisType.ALL]:
            logger.info("基本統計と可視化を実行")
            
            # 基本統計の計算
            if solar_data.density is not None:
                analyzer.results['statistics']['density'] = {
                    'min': float(np.nanmin(solar_data.density)),
                    'max': float(np.nanmax(solar_data.density)),
                    'mean': float(np.nanmean(solar_data.density)),
                    'std': float(np.nanstd(solar_data.density))
                }
            
            # 基本的な可視化
            if not args.no_gui:
                fig1 = visualizer.plot_equatorial_slice()
                if fig1 and args.save_figures:
                    fig1.savefig(output_dir / 'equatorial_slice.png', dpi=300)
                    plt.close(fig1)
                elif fig1:
                    plt.show()
        
        if analysis_type in [AnalysisType.CME_DETECTION, AnalysisType.ALL]:
            logger.info("CMEイベント検出を実行")
            events = analyzer.detect_cme_events(
                density_threshold=args.cme_density_threshold,
                velocity_threshold=args.cme_velocity_threshold
            )
            logger.info(f"検出されたCMEイベント: {len(events)}個")
        
        if analysis_type in [AnalysisType.TIME_SERIES, AnalysisType.ALL]:
            logger.info("時系列解析を実行")
            time_series = analyzer.analyze_time_series()
        
        if analysis_type in [AnalysisType.CORRELATION, AnalysisType.ALL]:
            logger.info("相関解析を実行")
            correlations = analyzer.analyze_correlations()
        
        if analysis_type in [AnalysisType.MAGNETIC, AnalysisType.ALL]:
            logger.info("磁場構造解析を実行")
            magnetic = analyzer.analyze_magnetic_structure()
        
        # 結果の保存
        analyzer.save_results(f"results_{dataset_number}.json")
        
        # レポート生成
        if args.report:
            analyzer.generate_report(f"report_{dataset_number}.pdf")
        
        logger.info(f"データセット {dataset_number} の解析完了")
        
        return analyzer.results
        
    except Exception as e:
        logger.error(f"データセット {dataset_number} の解析中にエラー: {e}")
        logger.debug("詳細:", exc_info=True)
        return {}


def main():
    """メインプログラム
    
    このプログラムは、太陽コロナの3次元再構成データを
    包括的に解析するための統合システムです。
    
    実行フロー:
    1. コマンドライン引数の解析
    2. ロギングシステムの初期化
    3. データセットの選択（単一またはバッチ）
    4. 指定された解析の実行
    5. 結果の保存とレポート生成
    
    このシステムにより、大量の太陽観測データを
    効率的に処理し、科学的知見を抽出できます。
    """
    
    print("=" * 70)
    print(" 太陽コロナ3次元再構成データ解析システム ".center(70))
    print(" Solar Corona 3D Reconstruction Analysis System ".center(70))
    print("=" * 70)
    print()
    
    # コマンドライン引数の解析
    args = parse_arguments()
    
    # ロギングの設定
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(args.output / "logs", log_level)
    
    logger.info("解析システムを起動")
    logger.info(f"パラメータ: {vars(args)}")
    
    # matplotlibの設定（GUI無効化対応）
    if args.no_gui:
        import matplotlib
        matplotlib.use('Agg')
        logger.info("GUIモードを無効化（バッチ処理モード）")
    
    # データディレクトリの確認
    data_dir = get_data_directory()
    logger.info(f"データディレクトリ: {data_dir}")
    
    # 実行モードの判定
    if args.batch:
        # バッチ処理モード
        logger.info(f"バッチ処理モード: {args.start} から {args.end} まで")
        
        # データセット番号のリストを生成
        start_num = int(args.start)
        end_num = int(args.end)
        
        all_results = {}
        successful = 0
        failed = 0
        
        for num in range(start_num, end_num + 1):
            dataset_number = f"{num:03d}"
            logger.info(f"\n処理中: データセット {dataset_number} ({num}/{end_num})")
            
            # ファイルの存在確認
            test_file = data_dir / f"rho{dataset_number}.h5"
            if not test_file.exists():
                logger.warning(f"データファイルが見つかりません: {test_file}")
                failed += 1
                continue
            
            # 解析の実行
            results = analyze_single_dataset(dataset_number, args, logger)
            
            if results:
                all_results[dataset_number] = results
                successful += 1
            else:
                failed += 1
        
        # バッチ処理の統計
        logger.info(f"\nバッチ処理完了:")
        logger.info(f"  成功: {successful}")
        logger.info(f"  失敗: {failed}")
        logger.info(f"  合計: {successful + failed}")
        
        # 統合レポートの生成
        if args.report and all_results:
            logger.info("統合レポートを生成中...")
            # ここに統合レポート生成のコードを追加
            
    else:
        # 単一データセット処理モード
        logger.info(f"単一データセット処理: {args.dataset}")
        
        results = analyze_single_dataset(args.dataset, args, logger)
        
        if results:
            logger.info("解析が正常に完了しました")
            
            # 結果のサマリー表示
            print("\n" + "=" * 50)
            print("解析結果サマリー")
            print("=" * 50)
            
            if 'statistics' in results and results['statistics']:
                print("\n基本統計:")
                for key, stats in results['statistics'].items():
                    if isinstance(stats, dict):
                        print(f"  {key}:")
                        for stat_name, value in stats.items():
                            print(f"    {stat_name}: {value:.2e}")
            
            if 'cme_events' in results:
                print(f"\nCMEイベント: {len(results['cme_events'])}個検出")
                for i, event in enumerate(results['cme_events'][:3]):
                    print(f"  イベント{i+1}: "
                          f"速度={event.velocity_radial:.0f} km/s, "
                          f"信頼度={event.confidence:.2f}")
            
            if 'correlations' in results:
                print("\n物理量相関:")
                for key, value in results['correlations'].items():
                    print(f"  {key}: {value:.3f}")
            
            if 'magnetic_analysis' in results:
                print("\n磁場構造:")
                for key, value in results['magnetic_analysis'].items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.3e}")
                    else:
                        print(f"  {key}: {value}")
            
            print("\n" + "=" * 50)
            print(f"詳細な結果は {args.output} に保存されています")
            
        else:
            logger.error("解析に失敗しました")
            sys.exit(1)
    
    logger.info("すべての処理が完了しました")
    print("\n解析システムを終了します。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nユーザーによって中断されました")
        sys.exit(0)
    except Exception as e:
        print(f"\n予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)