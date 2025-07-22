#!/usr/bin/env python3
"""
STEREO-A/SECCHI ユーティリティ関数群

このモジュールは、STEREO SECCHIデータ処理に必要な各種ユーティリティ関数を提供します。
IDL版のSSWIDLライブラリの関数をPythonで実装しています。

主な機能:
- 校正関数（get_calfac, get_vignetting等）
- 座標変換関数
- 画像処理関数
- ヘッダー操作関数
- 時刻処理関数

参照元: SSW/STEREO/SECCHI IDL Library
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from scipy import ndimage, interpolate
from scipy.optimize import curve_fit
import warnings
from datetime import datetime
import logging
import json
from pathlib import Path

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SECCHIUtils:
    """
    STEREO SECCHIユーティリティクラス
    """
    
    def __init__(self, silent=False):
        """
        初期化
        
        Parameters:
        -----------
        silent : bool, optional
            メッセージを抑制
        """
        self.silent = silent
        
        # 太陽の物理定数
        self.solar_radius_km = 696000.0  # km
        self.solar_radius_arcsec = 959.63  # arcsec at 1 AU
        
        # 検出器の仕様
        self.detector_specs = {
            'COR1': {
                'pixel_size': 15.0,  # arcsec/pixel
                'field_of_view': [1.5, 4.0],  # Rs
                'wavelength': 'white light',
                'detector_size': [512, 512]
            },
            'COR2': {
                'pixel_size': 14.7,  # arcsec/pixel
                'field_of_view': [3.0, 15.0],  # Rs
                'wavelength': 'white light',
                'detector_size': [512, 512]
            },
            'EUVI': {
                'pixel_size': 1.6,  # arcsec/pixel
                'field_of_view': 1.7,  # Rs
                'wavelengths': [171, 195, 284, 304],  # Angstrom
                'detector_size': [2048, 2048]
            },
            'HI1': {
                'pixel_size': 70.0,  # arcsec/pixel
                'field_of_view': [4.0, 24.0],  # Rs
                'wavelength': 'white light',
                'detector_size': [512, 512]
            },
            'HI2': {
                'pixel_size': 240.0,  # arcsec/pixel
                'field_of_view': [18.0, 88.0],  # Rs
                'wavelength': 'white light',
                'detector_size': [512, 512]
            }
        }
    
    def get_calfac(self, header, detector=None):
        """
        校正係数の取得
        
        Parameters:
        -----------
        header : dict
            ヘッダー構造体
        detector : str, optional
            検出器名
            
        Returns:
        --------
        float : 校正係数
        """
        if detector is None:
            detector = header.get('DETECTOR', '')
        
        # 基本的な校正係数（実際の値は観測時期や条件によって変わる）
        base_calfac = {
            'COR1': 1.0e-12,  # MSB/s to B/s
            'COR2': 1.0e-12,  # MSB/s to B/s
            'EUVI': 1.0e-6,   # DN/s to photons/s
            'HI1': 1.0e-12,   # MSB/s to B/s
            'HI2': 1.0e-12    # MSB/s to B/s
        }
        
        calfac = base_calfac.get(detector, 1.0)
        
        # SUMMEDモードの補正
        summed = header.get('SUMMED', 1)
        if summed > 1:
            calfac *= summed**2
        
        # 時間依存補正（簡略化版）
        obs_time = header.get('DATE_OBS', '')
        if obs_time:
            try:
                obs_date = Time(obs_time)
                # 劣化補正（年率約5%と仮定）
                years_elapsed = (Time.now() - obs_date).to(u.year).value
                degradation_factor = (0.95) ** years_elapsed
                calfac *= degradation_factor
            except:
                pass
        
        return calfac
    
    def get_vignetting(self, header, detector=None):
        """
        ビネッティング関数の取得
        
        Parameters:
        -----------
        header : dict
            ヘッダー構造体
        detector : str, optional
            検出器名
            
        Returns:
        --------
        numpy.ndarray : ビネッティング補正係数
        """
        if detector is None:
            detector = header.get('DETECTOR', '')
        
        # 画像サイズを取得
        nx = header.get('NAXIS1', 512)
        ny = header.get('NAXIS2', 512)
        
        # 中心位置
        center_x = header.get('CRPIX1', nx // 2)
        center_y = header.get('CRPIX2', ny // 2)
        
        # 座標グリッドを作成
        x = np.arange(nx) - center_x
        y = np.arange(ny) - center_y
        X, Y = np.meshgrid(x, y)
        
        # 中心からの距離
        R = np.sqrt(X**2 + Y**2)
        
        # 検出器別のビネッティング関数
        if detector == 'COR1':
            # COR1のビネッティング関数（簡略化版）
            vignetting = 1.0 - 0.1 * (R / (nx // 2))**2
            vignetting = np.clip(vignetting, 0.1, 1.0)
            
        elif detector == 'COR2':
            # COR2のビネッティング関数（簡略化版）
            vignetting = 1.0 - 0.15 * (R / (nx // 2))**2
            vignetting = np.clip(vignetting, 0.1, 1.0)
            
        elif detector == 'EUVI':
            # EUVIのビネッティング関数（簡略化版）
            vignetting = 1.0 - 0.05 * (R / (nx // 2))**2
            vignetting = np.clip(vignetting, 0.8, 1.0)
            
        elif detector in ['HI1', 'HI2']:
            # HIのビネッティング関数（簡略化版）
            vignetting = 1.0 - 0.2 * (R / (nx // 2))**2
            vignetting = np.clip(vignetting, 0.05, 1.0)
            
        else:
            # デフォルト（補正なし）
            vignetting = np.ones((ny, nx))
        
        return vignetting
    
    def get_flat_field(self, header, detector=None):
        """
        フラットフィールド補正の取得
        
        Parameters:
        -----------
        header : dict
            ヘッダー構造体
        detector : str, optional
            検出器名
            
        Returns:
        --------
        numpy.ndarray : フラットフィールド補正係数
        """
        if detector is None:
            detector = header.get('DETECTOR', '')
        
        # 画像サイズを取得
        nx = header.get('NAXIS1', 512)
        ny = header.get('NAXIS2', 512)
        
        # 簡略化版：基本的なフラットフィールド補正
        # 実際は検出器固有のフラットフィールド画像を使用
        
        # 中心位置
        center_x = nx // 2
        center_y = ny // 2
        
        # 座標グリッドを作成
        x = np.arange(nx) - center_x
        y = np.arange(ny) - center_y
        X, Y = np.meshgrid(x, y)
        
        # 簡単な補正パターン（実際はより複雑）
        if detector in ['COR1', 'COR2']:
            # コロナグラフの場合
            R = np.sqrt(X**2 + Y**2)
            flat_field = 1.0 + 0.02 * np.sin(R / 50.0) + 0.01 * np.cos(2 * np.arctan2(Y, X))
            
        elif detector == 'EUVI':
            # EUVIの場合
            flat_field = 1.0 + 0.01 * (X / 1000.0) + 0.01 * (Y / 1000.0)
            
        else:
            # デフォルト
            flat_field = np.ones((ny, nx))
        
        return flat_field
    
    def get_dark_current(self, header, detector=None):
        """
        ダークカレントの取得
        
        Parameters:
        -----------
        header : dict
            ヘッダー構造体
        detector : str, optional
            検出器名
            
        Returns:
        --------
        numpy.ndarray : ダークカレント画像
        """
        if detector is None:
            detector = header.get('DETECTOR', '')
        
        # 画像サイズを取得
        nx = header.get('NAXIS1', 512)
        ny = header.get('NAXIS2', 512)
        
        # 露出時間
        exptime = header.get('EXPTIME', 1.0)
        
        # 検出器別のダークカレント率（DN/s/pixel）
        dark_rates = {
            'COR1': 0.1,
            'COR2': 0.1,
            'EUVI': 0.05,
            'HI1': 0.2,
            'HI2': 0.2
        }
        
        dark_rate = dark_rates.get(detector, 0.1)
        
        # 基本的なダークカレント
        dark_current = np.full((ny, nx), dark_rate * exptime)
        
        # 温度依存性（簡略化版）
        try:
            temp = header.get('TEMP_CCD', 273.15)  # K
            temp_factor = np.exp((temp - 273.15) / 10.0)
            dark_current *= temp_factor
        except:
            pass
        
        return dark_current
    
    def convert_coords(self, x, y, header, from_system='pixel', to_system='arcsec'):
        """
        座標変換
        
        Parameters:
        -----------
        x : array-like
            x座標
        y : array-like
            y座標
        header : dict
            ヘッダー構造体
        from_system : str, optional
            変換元の座標系
        to_system : str, optional
            変換先の座標系
            
        Returns:
        --------
        tuple : (変換後x座標, 変換後y座標)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        # 座標変換パラメータ
        crpix1 = header.get('CRPIX1', 0)
        crpix2 = header.get('CRPIX2', 0)
        crval1 = header.get('CRVAL1', 0)
        crval2 = header.get('CRVAL2', 0)
        cdelt1 = header.get('CDELT1', 1)
        cdelt2 = header.get('CDELT2', 1)
        
        if from_system == 'pixel' and to_system == 'arcsec':
            # ピクセル座標からarcsec座標へ
            x_arcsec = (x - crpix1) * cdelt1 + crval1
            y_arcsec = (y - crpix2) * cdelt2 + crval2
            return x_arcsec, y_arcsec
            
        elif from_system == 'arcsec' and to_system == 'pixel':
            # arcsec座標からピクセル座標へ
            x_pixel = (x - crval1) / cdelt1 + crpix1
            y_pixel = (y - crval2) / cdelt2 + crpix2
            return x_pixel, y_pixel
            
        elif from_system == 'arcsec' and to_system == 'Rs':
            # arcsec座標から太陽半径単位へ
            rsun_arcsec = header.get('RSUN', self.solar_radius_arcsec)
            x_rs = x / rsun_arcsec
            y_rs = y / rsun_arcsec
            return x_rs, y_rs
            
        elif from_system == 'Rs' and to_system == 'arcsec':
            # 太陽半径単位からarcsec座標へ
            rsun_arcsec = header.get('RSUN', self.solar_radius_arcsec)
            x_arcsec = x * rsun_arcsec
            y_arcsec = y * rsun_arcsec
            return x_arcsec, y_arcsec
            
        else:
            # サポートされていない変換
            logger.warning(f"Unsupported coordinate conversion: {from_system} -> {to_system}")
            return x, y
    
    def get_solar_ephemeris(self, header):
        """
        太陽の天体暦情報の取得
        
        Parameters:
        -----------
        header : dict
            ヘッダー構造体
            
        Returns:
        --------
        dict : 太陽天体暦情報
        """
        ephemeris = {
            'rsun_arcsec': header.get('RSUN', self.solar_radius_arcsec),
            'dsun_obs': header.get('DSUN_OBS', 1.5e11),  # meters
            'solar_p': header.get('SOLAR_P', 0.0),  # degrees
            'solar_b0': header.get('SOLAR_B0', 0.0),  # degrees
            'solar_l0': header.get('SOLAR_L0', 0.0),  # degrees
            'carrington_rotation': header.get('CARR_ROT', 0)
        }
        
        # 観測時刻から計算される値
        obs_time = header.get('DATE_OBS', '')
        if obs_time:
            try:
                obs_date = Time(obs_time)
                
                # 太陽のP角、B0角、L0角の計算（簡略化版）
                # 実際はより精密な計算が必要
                days_from_epoch = (obs_date - Time('2000-01-01')).to(u.day).value
                
                # 簡単な近似
                ephemeris['solar_p'] = 0.0  # 簡略化
                ephemeris['solar_b0'] = 7.25 * np.sin(2 * np.pi * days_from_epoch / 365.25)
                ephemeris['solar_l0'] = (days_from_epoch * 13.2) % 360
                
            except:
                pass
        
        return ephemeris
    
    def calculate_position_angle(self, x, y, header):
        """
        位置角の計算
        
        Parameters:
        -----------
        x : array-like
            x座標
        y : array-like
            y座標
        header : dict
            ヘッダー構造体
            
        Returns:
        --------
        numpy.ndarray : 位置角（度）
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        # 太陽中心からの位置角
        pa = np.arctan2(x, y) * 180.0 / np.pi
        
        # 太陽のP角補正
        solar_p = header.get('SOLAR_P', 0.0)
        pa_corrected = pa + solar_p
        
        # 0-360度の範囲に正規化
        pa_corrected = pa_corrected % 360
        
        return pa_corrected
    
    def apply_roll_correction(self, image, header, angle=None):
        """
        ロール補正の適用
        
        Parameters:
        -----------
        image : numpy.ndarray
            入力画像
        header : dict
            ヘッダー構造体
        angle : float, optional
            回転角（度）
            
        Returns:
        --------
        tuple : (補正済み画像, 更新されたヘッダー)
        """
        if angle is None:
            angle = -header.get('CROTA2', 0.0)
        
        if abs(angle) < 0.01:
            return image, header
        
        # 画像の回転
        rotated_image = ndimage.rotate(image, angle, reshape=False, order=1)
        
        # ヘッダーの更新
        header_copy = header.copy()
        header_copy['CROTA1'] = 0.0
        header_copy['CROTA2'] = 0.0
        
        if 'HISTORY' not in header_copy:
            header_copy['HISTORY'] = []
        header_copy['HISTORY'].append(f"Roll correction applied: {angle:.2f} degrees")
        
        return rotated_image, header_copy
    
    def create_coordinate_arrays(self, header):
        """
        座標配列の作成
        
        Parameters:
        -----------
        header : dict
            ヘッダー構造体
            
        Returns:
        --------
        dict : 座標配列の辞書
        """
        nx = header.get('NAXIS1', 512)
        ny = header.get('NAXIS2', 512)
        
        # ピクセル座標
        x_pixel = np.arange(nx)
        y_pixel = np.arange(ny)
        X_pixel, Y_pixel = np.meshgrid(x_pixel, y_pixel)
        
        # arcsec座標
        X_arcsec, Y_arcsec = self.convert_coords(X_pixel, Y_pixel, header, 
                                                'pixel', 'arcsec')
        
        # 太陽半径単位
        X_rs, Y_rs = self.convert_coords(X_arcsec, Y_arcsec, header, 
                                        'arcsec', 'Rs')
        
        # 中心からの距離
        R_arcsec = np.sqrt(X_arcsec**2 + Y_arcsec**2)
        R_rs = np.sqrt(X_rs**2 + Y_rs**2)
        
        # 位置角
        PA = self.calculate_position_angle(X_arcsec, Y_arcsec, header)
        
        return {
            'X_pixel': X_pixel,
            'Y_pixel': Y_pixel,
            'X_arcsec': X_arcsec,
            'Y_arcsec': Y_arcsec,
            'X_rs': X_rs,
            'Y_rs': Y_rs,
            'R_arcsec': R_arcsec,
            'R_rs': R_rs,
            'PA': PA
        }
    
    def estimate_background(self, image, header, method='median'):
        """
        背景レベルの推定
        
        Parameters:
        -----------
        image : numpy.ndarray
            入力画像
        header : dict
            ヘッダー構造体
        method : str, optional
            推定方法
            
        Returns:
        --------
        float : 背景レベル
        """
        if method == 'median':
            # 外側の領域からメディアンを計算
            ny, nx = image.shape
            border = min(nx, ny) // 10
            
            # 外縁部分を抽出
            border_pixels = np.concatenate([
                image[:border, :].flatten(),
                image[-border:, :].flatten(),
                image[:, :border].flatten(),
                image[:, -border:].flatten()
            ])
            
            background = np.median(border_pixels)
            
        elif method == 'mode':
            # ヒストグラムのモードを計算
            hist, bins = np.histogram(image.flatten(), bins=100)
            max_bin = np.argmax(hist)
            background = (bins[max_bin] + bins[max_bin + 1]) / 2
            
        else:
            # デフォルト：メディアン
            background = np.median(image)
        
        return background
    
    def precommcorrect(self, image, header):
        """
        試験運用期間の補正
        
        Parameters:
        -----------
        image : numpy.ndarray
            入力画像
        header : dict
            ヘッダー構造体
            
        Returns:
        --------
        tuple : (補正済み画像, 更新されたヘッダー)
        """
        # 試験運用期間の特定
        obs_time = header.get('DATE_OBS', '')
        detector = header.get('DETECTOR', '')
        
        if not obs_time:
            return image, header
        
        try:
            obs_date = Time(obs_time)
            
            # STEREO-A COR1の試験運用期間
            if (detector == 'COR1' and 
                header.get('OBSRVTRY', '') == 'STEREO_A' and
                obs_date < Time('2007-02-03T13:15')):
                
                # 露出時間の補正
                if header.get('EXTEND') == 'T' and header.get('N_IMAGES', 1) > 1:
                    # 拡張ヘッダーからの露出時間計算（簡略化版）
                    # 実際はMRDFITSで読み込む
                    pass
                
                # 試験運用期間固有の補正
                corrected_image = image * 1.1  # 例：10%の補正
                
                # ヘッダーの更新
                header_copy = header.copy()
                if 'HISTORY' not in header_copy:
                    header_copy['HISTORY'] = []
                header_copy['HISTORY'].append("Pre-commissioning correction applied")
                
                return corrected_image, header_copy
        
        except:
            pass
        
        return image, header

def main():
    """
    テスト用のメイン関数
    """
    # ユーティリティクラスのインスタンス作成
    utils = SECCHIUtils()
    
    # テスト用のヘッダー
    test_header = {
        'DETECTOR': 'COR1',
        'NAXIS1': 512,
        'NAXIS2': 512,
        'CRPIX1': 256,
        'CRPIX2': 256,
        'CRVAL1': 0,
        'CRVAL2': 0,
        'CDELT1': 15.0,
        'CDELT2': 15.0,
        'RSUN': 998.69,
        'DATE_OBS': '2022-06-13T03:21:36.012',
        'EXPTIME': 1.0
    }
    
    print("=== SECCHI Utilities Test ===")
    
    # 校正係数の取得
    calfac = utils.get_calfac(test_header)
    print(f"Calibration factor: {calfac:.2e}")
    
    # ビネッティング関数の取得
    vignetting = utils.get_vignetting(test_header)
    print(f"Vignetting shape: {vignetting.shape}")
    print(f"Vignetting range: {vignetting.min():.3f} to {vignetting.max():.3f}")
    
    # 座標変換のテスト
    x_pixel, y_pixel = 256, 256
    x_arcsec, y_arcsec = utils.convert_coords(x_pixel, y_pixel, test_header, 
                                             'pixel', 'arcsec')
    print(f"Pixel ({x_pixel}, {y_pixel}) -> Arcsec ({x_arcsec:.2f}, {y_arcsec:.2f})")
    
    # 座標配列の作成
    coords = utils.create_coordinate_arrays(test_header)
    print(f"Coordinate arrays created:")
    print(f"  R_rs range: {coords['R_rs'].min():.2f} to {coords['R_rs'].max():.2f}")
    print(f"  PA range: {coords['PA'].min():.2f} to {coords['PA'].max():.2f}")
    
    # 太陽天体暦の取得
    ephemeris = utils.get_solar_ephemeris(test_header)
    print(f"Solar ephemeris:")
    print(f"  RSUN: {ephemeris['rsun_arcsec']:.2f} arcsec")
    print(f"  B0: {ephemeris['solar_b0']:.2f} degrees")
    print(f"  L0: {ephemeris['solar_l0']:.2f} degrees")
    
    print("\nUtilities test completed successfully!")

if __name__ == "__main__":
    main()