#!/usr/bin/env python3
"""
STEREO-A/SECCHI/COR1専用太陽座標・測定機能

このモジュールは、IDL版SSWIDL/stereo_rsun.pro, scc_sun_center.proの
機能をPythonに移植したものです。
太陽座標系の計算、距離測定、座標変換などの機能を提供します。

主な機能:
- 太陽半径計算（距離補正込み）
- 太陽中心座標の計算
- 座標変換（ピクセル↔天体座標）
- 測定ツール（距離、角度）
- HPC/HCR座標系変換

参照元: SSWIDL stereo_rsun.pro, scc_sun_center.pro
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord, ICRS
from astropy.wcs import WCS
from astropy.time import Time
import logging
from typing import Tuple, Optional, Union, Dict, List
from datetime import datetime

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 太陽定数
SOLAR_RADIUS_KM = 6.95508e5  # km
ARCSEC_TO_RAD = 1 / 206265.0  # arcsec to radian conversion
AU_KM = 1.495978707e8  # km/AU


class COR1SolarUtils:
    """
    STEREO COR1画像用太陽座標・測定機能クラス
    
    IDL版SSWIDL各プログラムの機能をPythonで実装
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
        self.version = "COR1_Solar_Utils v1.0 (Python port of IDL SSWIDL)"
        
        if not self.silent:
            logger.info(self.version)
    
    def calculate_solar_radius_arcsec(self, datetime_obs, spacecraft='A', 
                                     distance_au=None):
        """
        太陽半径をarcsecで計算
        
        IDL版stereo_rsun.proの機能を再現
        
        Parameters:
        -----------
        datetime_obs : str or datetime
            観測日時
        spacecraft : str, optional
            宇宙機識別子（'A', 'B'）
        distance_au : float, optional
            太陽からの距離（AU）。指定されない場合は近似値を使用
            
        Returns:
        --------
        float : 太陽半径（arcsec）
        """
        # 距離が指定されていない場合は近似計算
        if distance_au is None:
            # STEREO軌道の近似（1 AU周辺）
            if isinstance(datetime_obs, str):
                # 簡略化: 1 AUを使用
                distance_au = 1.0
            else:
                distance_au = 1.0
        
        # 太陽半径計算（IDL版のアルゴリズム）
        # r = (太陽半径[km] * 206265[arcsec/rad]) / 距離[km]
        distance_km = distance_au * AU_KM
        solar_radius_arcsec = (SOLAR_RADIUS_KM * 206265.0) / distance_km
        
        return solar_radius_arcsec
    
    def get_sun_center(self, header, full_scale=None):
        """
        太陽中心座標を取得
        
        IDL版scc_sun_center.proの機能を再現
        
        Parameters:
        -----------
        header : dict or astropy.io.fits.Header
            FITSヘッダー情報
        full_scale : int, optional
            フルスケール画像サイズ
            
        Returns:
        --------
        dict : {'xcen': float, 'ycen': float} 太陽中心座標（ピクセル）
        """
        # ヘッダーから基本情報を取得
        if hasattr(header, 'get'):
            crpix1 = header.get('CRPIX1', header.get('CRPIX1', 0))
            crpix2 = header.get('CRPIX2', header.get('CRPIX2', 0))
            naxis1 = header.get('NAXIS1', 2048)
            naxis2 = header.get('NAXIS2', 2048)
            cdelt1 = header.get('CDELT1', 1.0)
            cdelt2 = header.get('CDELT2', 1.0)
        else:
            crpix1 = getattr(header, 'crpix1', 0)
            crpix2 = getattr(header, 'crpix2', 0)
            naxis1 = getattr(header, 'naxis1', 2048)
            naxis2 = getattr(header, 'naxis2', 2048)
            cdelt1 = getattr(header, 'cdelt1', 1.0)
            cdelt2 = getattr(header, 'cdelt2', 1.0)
        
        # 太陽中心座標の計算（IDL座標系：1始まり）
        # Python座標系（0始まり）に変換
        sun_center = {
            'xcen': float(crpix1 - 1),  # IDL -> Python 座標変換
            'ycen': float(crpix2 - 1)
        }
        
        # フルスケール補正
        if full_scale is not None and full_scale != naxis1:
            scale_factor = full_scale / naxis1
            sun_center['xcen'] *= scale_factor
            sun_center['ycen'] *= scale_factor
        
        return sun_center
    
    def pixel_to_arcsec(self, pixel_coord, header, sun_center=None):
        """
        ピクセル座標をarcsec座標に変換
        
        Parameters:
        -----------
        pixel_coord : tuple or array-like
            ピクセル座標 (x, y)
        header : dict or astropy.io.fits.Header
            FITSヘッダー情報
        sun_center : dict, optional
            太陽中心座標。指定されない場合は自動計算
            
        Returns:
        --------
        tuple : (x_arcsec, y_arcsec) 太陽中心からのarcsec座標
        """
        if sun_center is None:
            sun_center = self.get_sun_center(header)
        
        # ヘッダーから画素スケールを取得
        if hasattr(header, 'get'):
            cdelt1 = abs(header.get('CDELT1', 1.0))
            cdelt2 = abs(header.get('CDELT2', 1.0))
        else:
            cdelt1 = abs(getattr(header, 'cdelt1', 1.0))
            cdelt2 = abs(getattr(header, 'cdelt2', 1.0))
        
        # ピクセル差分を計算
        dx_pix = pixel_coord[0] - sun_center['xcen']
        dy_pix = pixel_coord[1] - sun_center['ycen']
        
        # arcsecに変換
        x_arcsec = dx_pix * cdelt1
        y_arcsec = dy_pix * cdelt2
        
        return x_arcsec, y_arcsec
    
    def arcsec_to_pixel(self, arcsec_coord, header, sun_center=None):
        """
        arcsec座標をピクセル座標に変換
        
        Parameters:
        -----------
        arcsec_coord : tuple or array-like
            arcsec座標 (x, y)
        header : dict or astropy.io.fits.Header
            FITSヘッダー情報
        sun_center : dict, optional
            太陽中心座標。指定されない場合は自動計算
            
        Returns:
        --------
        tuple : (x_pixel, y_pixel) ピクセル座標
        """
        if sun_center is None:
            sun_center = self.get_sun_center(header)
        
        # ヘッダーから画素スケールを取得
        if hasattr(header, 'get'):
            cdelt1 = abs(header.get('CDELT1', 1.0))
            cdelt2 = abs(header.get('CDELT2', 1.0))
        else:
            cdelt1 = abs(getattr(header, 'cdelt1', 1.0))
            cdelt2 = abs(getattr(header, 'cdelt2', 1.0))
        
        # ピクセルに変換
        dx_pix = arcsec_coord[0] / cdelt1
        dy_pix = arcsec_coord[1] / cdelt2
        
        x_pixel = sun_center['xcen'] + dx_pix
        y_pixel = sun_center['ycen'] + dy_pix
        
        return x_pixel, y_pixel
    
    def calculate_distance(self, point1, point2, header, unit='arcsec'):
        """
        2点間の距離を計算
        
        Parameters:
        -----------
        point1 : tuple
            開始点座標 (x, y)
        point2 : tuple
            終了点座標 (x, y)
        header : dict or astropy.io.fits.Header
            FITSヘッダー情報
        unit : str, optional
            出力単位（'arcsec', 'pixel', 'rsun'）
            
        Returns:
        --------
        float : 距離
        """
        if unit == 'pixel':
            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            distance = np.sqrt(dx**2 + dy**2)
        else:
            # arcsec座標に変換して計算
            arcsec1 = self.pixel_to_arcsec(point1, header)
            arcsec2 = self.pixel_to_arcsec(point2, header)
            
            dx_arcsec = arcsec2[0] - arcsec1[0]
            dy_arcsec = arcsec2[1] - arcsec1[1]
            distance_arcsec = np.sqrt(dx_arcsec**2 + dy_arcsec**2)
            
            if unit == 'arcsec':
                distance = distance_arcsec
            elif unit == 'rsun':
                # 太陽半径での規格化
                if hasattr(header, 'get'):
                    date_obs = header.get('DATE_OBS', header.get('DATE-OBS', ''))
                else:
                    date_obs = getattr(header, 'date_obs', '')
                
                rsun_arcsec = self.calculate_solar_radius_arcsec(date_obs)
                distance = distance_arcsec / rsun_arcsec
            else:
                distance = distance_arcsec
        
        return distance
    
    def calculate_angle(self, center, point, header=None):
        """
        中心点からの角度を計算
        
        Parameters:
        -----------
        center : tuple
            中心点座標 (x, y)
        point : tuple
            測定点座標 (x, y)
        header : dict or astropy.io.fits.Header, optional
            FITSヘッダー情報
            
        Returns:
        --------
        float : 角度（度、北から時計回り）
        """
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        
        # 角度計算（北から時計回り）
        angle_rad = np.arctan2(dx, dy)
        angle_deg = np.degrees(angle_rad)
        
        # 0-360度に正規化
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg
    
    def convert_hpc_to_hcr(self, x_arcsec, y_arcsec, header):
        """
        Helioprojective Cartesian (HPC) から Heliocentric Radial (HCR) 座標に変換
        
        Parameters:
        -----------
        x_arcsec : float or array
            X座標（arcsec）
        y_arcsec : float or array
            Y座標（arcsec）
        header : dict or astropy.io.fits.Header
            FITSヘッダー情報
            
        Returns:
        --------
        tuple : (elongation, position_angle) 離角と位置角
        """
        # 距離の計算
        r_arcsec = np.sqrt(x_arcsec**2 + y_arcsec**2)
        
        # 位置角の計算（北から時計回り）
        position_angle = np.degrees(np.arctan2(x_arcsec, y_arcsec))
        position_angle = np.where(position_angle < 0, position_angle + 360, position_angle)
        
        # 離角への変換（太陽半径での規格化も可能）
        if hasattr(header, 'get'):
            date_obs = header.get('DATE_OBS', header.get('DATE-OBS', ''))
        else:
            date_obs = getattr(header, 'date_obs', '')
        
        # 単位は arcsec のままとする（必要に応じて太陽半径単位に変換可能）
        elongation = r_arcsec
        
        return elongation, position_angle
    
    def create_measurement_overlay(self, image, measurements, header):
        """
        測定結果をオーバーレイ表示
        
        Parameters:
        -----------
        image : numpy.ndarray
            画像データ
        measurements : list
            測定データのリスト
        header : dict or astropy.io.fits.Header
            FITSヘッダー情報
            
        Returns:
        --------
        matplotlib.figure.Figure : 測定オーバーレイ図
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap='gray', origin='lower')
        
        # 太陽中心の表示
        sun_center = self.get_sun_center(header)
        ax.plot(sun_center['xcen'], sun_center['ycen'], 
                'r+', markersize=20, markeredgewidth=2, label='Sun Center')
        
        # 太陽リムの表示
        if hasattr(header, 'get'):
            date_obs = header.get('DATE_OBS', header.get('DATE-OBS', ''))
            cdelt1 = abs(header.get('CDELT1', 1.0))
        else:
            date_obs = getattr(header, 'date_obs', '')
            cdelt1 = abs(getattr(header, 'cdelt1', 1.0))
        
        rsun_arcsec = self.calculate_solar_radius_arcsec(date_obs)
        rsun_pix = rsun_arcsec / cdelt1
        
        circle = plt.Circle((sun_center['xcen'], sun_center['ycen']), 
                           rsun_pix, fill=False, color='yellow', 
                           linewidth=2, label='Solar Limb')
        ax.add_patch(circle)
        
        # 測定データの表示
        for i, measurement in enumerate(measurements):
            if 'points' in measurement:
                points = measurement['points']
                if len(points) == 2:
                    # 線分の描画
                    ax.plot([points[0][0], points[1][0]], 
                           [points[0][1], points[1][1]], 
                           'g-', linewidth=2, label=f'Measurement {i+1}')
                    
                    # 距離の表示
                    if 'distance' in measurement:
                        mid_x = (points[0][0] + points[1][0]) / 2
                        mid_y = (points[0][1] + points[1][1]) / 2
                        ax.text(mid_x, mid_y, f"{measurement['distance']:.1f}\"",
                               color='green', fontsize=12, ha='center',
                               bbox=dict(boxstyle="round,pad=0.3", 
                                       facecolor='white', alpha=0.8))
        
        ax.set_title('COR1 Measurements')
        ax.legend()
        ax.axis('equal')
        
        return fig
    
    def create_coordinate_conversion_table(self, points, header):
        """
        座標変換テーブルを作成
        
        Parameters:
        -----------
        points : list
            座標点のリスト [(x_pix, y_pix), ...]
        header : dict or astropy.io.fits.Header
            FITSヘッダー情報
            
        Returns:
        --------
        dict : 座標変換テーブル
        """
        conversion_table = []
        sun_center = self.get_sun_center(header)
        
        for i, point in enumerate(points):
            x_pix, y_pix = point
            x_arcsec, y_arcsec = self.pixel_to_arcsec(point, header, sun_center)
            elongation, pa = self.convert_hpc_to_hcr(x_arcsec, y_arcsec, header)
            
            conversion_table.append({
                'point_id': i + 1,
                'pixel_x': x_pix,
                'pixel_y': y_pix,
                'arcsec_x': x_arcsec,
                'arcsec_y': y_arcsec,
                'elongation_arcsec': elongation,
                'position_angle_deg': pa,
                'distance_from_center_pix': np.sqrt((x_pix - sun_center['xcen'])**2 + 
                                                  (y_pix - sun_center['ycen'])**2),
                'distance_from_center_arcsec': np.sqrt(x_arcsec**2 + y_arcsec**2)
            })
        
        return {
            'sun_center': sun_center,
            'points': conversion_table,
            'header_info': {
                'cdelt1': abs(header.get('CDELT1', 1.0) if hasattr(header, 'get') 
                             else getattr(header, 'cdelt1', 1.0)),
                'solar_radius_arcsec': self.calculate_solar_radius_arcsec(
                    header.get('DATE_OBS', '') if hasattr(header, 'get') 
                    else getattr(header, 'date_obs', ''))
            }
        }


def main():
    """
    テスト用のメイン関数
    """
    print("=== COR1 Solar Utils Test ===")
    
    # COR1SolarUtilsインスタンスを作成
    solar_utils = COR1SolarUtils()
    
    # モックヘッダーの作成
    test_header = {
        'DATE_OBS': '2022-06-13T03:36:50',
        'CRPIX1': 256,
        'CRPIX2': 256,
        'CDELT1': 14.7,  # arcsec/pixel for COR1
        'CDELT2': 14.7,
        'NAXIS1': 512,
        'NAXIS2': 512
    }
    
    # 太陽半径計算のテスト
    rsun_arcsec = solar_utils.calculate_solar_radius_arcsec('2022-06-13T03:36:50', 'A')
    print(f"Solar radius: {rsun_arcsec:.1f} arcsec")
    
    # 太陽中心計算のテスト
    sun_center = solar_utils.get_sun_center(test_header)
    print(f"Sun center: ({sun_center['xcen']:.1f}, {sun_center['ycen']:.1f}) pixels")
    
    # 座標変換のテスト
    test_pixel = (300, 350)
    arcsec_coord = solar_utils.pixel_to_arcsec(test_pixel, test_header, sun_center)
    print(f"Pixel {test_pixel} -> Arcsec ({arcsec_coord[0]:.1f}, {arcsec_coord[1]:.1f})")
    
    # HCR変換のテスト
    elongation, pa = solar_utils.convert_hpc_to_hcr(arcsec_coord[0], arcsec_coord[1], test_header)
    print(f"HCR: Elongation {elongation:.1f} arcsec, PA {pa:.1f} deg")
    
    # 距離計算のテスト
    point1 = (200, 200)
    point2 = (300, 350)
    distance_arcsec = solar_utils.calculate_distance(point1, point2, test_header, 'arcsec')
    distance_rsun = solar_utils.calculate_distance(point1, point2, test_header, 'rsun')
    print(f"Distance: {distance_arcsec:.1f} arcsec, {distance_rsun:.2f} Rsun")
    
    # 座標変換テーブルのテスト
    test_points = [(200, 200), (300, 350), (400, 250)]
    coord_table = solar_utils.create_coordinate_conversion_table(test_points, test_header)
    print("\nCoordinate conversion table:")
    for point_data in coord_table['points']:
        print(f"Point {point_data['point_id']}: "
              f"({point_data['pixel_x']:.0f}, {point_data['pixel_y']:.0f}) pix -> "
              f"({point_data['arcsec_x']:.1f}, {point_data['arcsec_y']:.1f}) arcsec")
    
    print("COR1 Solar Utils test completed successfully!")


if __name__ == "__main__":
    main()