#!/usr/bin/env python3
"""
STEREO座標変換ユーティリティ

STEREO衛星の軌道特性と座標変換を扱うユーティリティ
SSWIDLの座標変換機能をPythonに移植

主な機能:
- STEREO衛星軌道位置の計算
- 太陽中心座標系との変換
- 視線方向の補正
- 軌道傾斜による回転角の計算
"""

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body_barycentric, get_body
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.wcs import WCS
import sunpy.coordinates
from sunpy.coordinates import HeliographicStonyhurst, HeliographicCarrington
import warnings


class STEREOOrbit:
    """STEREO衛星の軌道計算クラス"""
    
    # STEREO衛星の軌道パラメータ（概算値）
    STEREO_LAUNCH_DATE = Time('2006-10-26T00:00:00')
    STEREO_A_ANGULAR_SPEED = 0.9856 * u.deg / u.day  # 地球より少し遅い
    STEREO_B_ANGULAR_SPEED = 1.0144 * u.deg / u.day  # 地球より少し速い
    ORBITAL_RADIUS = 1.0 * u.AU  # 概算
    
    @classmethod
    def get_stereo_position(cls, obstime, spacecraft='A'):
        """
        STEREO衛星の太陽中心座標を計算
        
        Parameters:
        -----------
        obstime : astropy.time.Time
            観測時刻
        spacecraft : str
            衛星名 ('A' or 'B')
            
        Returns:
        --------
        position : astropy.coordinates.SkyCoord
            太陽中心座標系での衛星位置
        """
        if isinstance(obstime, str):
            obstime = Time(obstime)
        
        # 打ち上げからの経過日数
        days_since_launch = (obstime - cls.STEREO_LAUNCH_DATE).to(u.day)
        
        if spacecraft.upper() == 'A':
            # STEREO-Aは地球軌道より内側を周回（太陽に近づく）
            angular_speed = cls.STEREO_A_ANGULAR_SPEED
            orbital_radius = cls.ORBITAL_RADIUS * (1.0 - 0.01 * days_since_launch.value / 365.25)
        else:
            # STEREO-Bは地球軌道より外側を周回（太陽から遠ざかる）
            angular_speed = cls.STEREO_B_ANGULAR_SPEED
            orbital_radius = cls.ORBITAL_RADIUS * (1.0 + 0.01 * days_since_launch.value / 365.25)
        
        # 軌道角度の計算
        orbital_angle = angular_speed * days_since_launch
        
        # 太陽中心直交座標
        x = orbital_radius * np.cos(orbital_angle)
        y = orbital_radius * np.sin(orbital_angle)
        z = 0.0 * u.km  # 黄道面上と仮定
        
        # HeliographicStonyhurst座標系で返す
        position = SkyCoord(
            x=x, y=y, z=z,
            frame=HeliographicStonyhurst(obstime=obstime),
            representation_type='cartesian'
        )
        
        return position
    
    @classmethod
    def get_earth_stereo_separation(cls, obstime, spacecraft='A'):
        """
        地球とSTEREO衛星の分離角を計算
        
        Parameters:
        -----------
        obstime : astropy.time.Time
            観測時刻
        spacecraft : str
            衛星名 ('A' or 'B')
            
        Returns:
        --------
        separation : astropy.units.Quantity
            分離角 (degrees)
        """
        if isinstance(obstime, str):
            obstime = Time(obstime)
        
        # 地球位置
        earth_pos = get_body_barycentric('earth', obstime)
        
        # STEREO位置
        stereo_pos = cls.get_stereo_position(obstime, spacecraft)
        
        # 太陽から見た分離角
        earth_angle = np.arctan2(earth_pos.y.to(u.AU).value, earth_pos.x.to(u.AU).value)
        stereo_angle = np.arctan2(stereo_pos.cartesian.y.to(u.AU).value, 
                                 stereo_pos.cartesian.x.to(u.AU).value)
        
        separation = abs(stereo_angle - earth_angle) * u.radian
        return separation.to(u.deg)


class STEREOCoordinateTransform:
    """STEREO座標変換クラス"""
    
    @staticmethod
    def get_solar_north_angle(obstime, spacecraft='A'):
        """
        STEREO衛星から見た太陽北極の回転角を計算
        
        SSWIDLのstereo_get_pa_north.proに対応
        
        Parameters:
        -----------
        obstime : astropy.time.Time or str
            観測時刻
        spacecraft : str
            衛星名 ('A' or 'B')
            
        Returns:
        --------
        pa_north : astropy.units.Quantity
            太陽北極の位置角 (degrees)
        """
        if isinstance(obstime, str):
            obstime = Time(obstime)
        
        # STEREO衛星位置を取得
        stereo_pos = STEREOOrbit.get_stereo_position(obstime, spacecraft)
        
        # 太陽の自転軸傾斜
        solar_inclination = 7.25 * u.deg  # 黄道面に対する太陽自転軸の傾斜
        
        # 軌道位置による補正
        days_since_launch = (obstime - STEREOOrbit.STEREO_LAUNCH_DATE).to(u.day)
        orbital_correction = 0.1 * days_since_launch.value / 365.25 * u.deg
        
        if spacecraft.upper() == 'A':
            pa_north = solar_inclination + orbital_correction
        else:
            pa_north = solar_inclination - orbital_correction
        
        return pa_north
    
    @staticmethod
    def get_solar_center_pixel(header, image_shape, origin=0):
        """
        WCS上の太陽中心 (HPLN=0 arcsec, HPLT=0 arcsec) を
        pixel座標に変換して返す。

        Parameters
        ----------
        header : astropy.io.fits.Header
            FITSヘッダー
        image_shape : tuple
            画像サイズ (ny, nx)
        origin : int
            0ならnumpy index系, 1ならFITS系

        Returns
        -------
        center_x, center_y : float, float
            太陽中心のpixel座標
        """
        updated_header = STEREOCoordinateTransform.correct_pointing(header, image_shape)

        # CROTA がある場合は WCS が解釈しやすいように CROTA2 にも入れる
        if 'CROTA2' not in updated_header:
            if 'CROTA' in updated_header:
                updated_header['CROTA2'] = float(updated_header['CROTA'])
            elif 'CROTA1' in updated_header:
                updated_header['CROTA2'] = float(updated_header['CROTA1'])

        try:
            wcs = WCS(updated_header)
            world = np.array([[0.0, 0.0]], dtype=float)  # (HPLN, HPLT) = (0, 0) arcsec
            center_x, center_y = wcs.all_world2pix(world, origin)[0]

            if np.isfinite(center_x) and np.isfinite(center_y):
                return float(center_x), float(center_y)

        except Exception as e:
            warnings.warn(f"WCS solar-center transform failed, fallback to CRPIX: {e}")

        # fallback
        if origin == 0:
            fallback_x = float(updated_header.get('CRPIX1', image_shape[1] / 2.0 + 0.5)) - 1.0
            fallback_y = float(updated_header.get('CRPIX2', image_shape[0] / 2.0 + 0.5)) - 1.0
        else:
            fallback_x = float(updated_header.get('CRPIX1', image_shape[1] / 2.0 + 0.5))
            fallback_y = float(updated_header.get('CRPIX2', image_shape[0] / 2.0 + 0.5))

        return fallback_x, fallback_y
    
    @staticmethod
    def get_crota_angle(header, spacecraft=None):
        """
        FITSヘッダーから回転角を取得・計算（実際のファイルデータを優先）
        
        Parameters:
        -----------
        header : astropy.io.fits.Header
            FITSヘッダー
        spacecraft : str, optional
            衛星名（ヘッダーから自動判定も可能）
            
        Returns:
        --------
        crota : float
            画像回転角 (degrees)
        """
        # 1. ヘッダーから直接回転角を取得（最優先）
        rotation_keywords = [
            'CROTA',                         # STEREO-A特有のCROTA（コメント付き）
            'CROTA2', 'CROTA1',             # 標準WCS回転角
            'SC_ROLL', 'ROLL_ANGLE',         # 衛星ロール角
            'PA_NORTH', 'NORTHANG',          # 太陽北極角
            'SOLAR_P0', 'P_ANGLE',           # 太陽P角
            'INST_ROT', 'INSTROT'            # 装置回転角
        ]
        
        for key in rotation_keywords:
            if key in header:
                rotation_value = float(header[key])
                print(f"Debug: Found rotation angle in header['{key}'] = {rotation_value:.3f}°")
                return rotation_value
        
        # 2. STEREO特有のキーワードをチェック
        stereo_keywords = [
            'STEREOSCOPIC_ANGLE', 'STEREO_ANGLE',
            'SPACECRAFT_ROLL', 'SC_ATT_ROLL'
        ]
        
        for key in stereo_keywords:
            if key in header:
                rotation_value = float(header[key])
                print(f"Debug: Found STEREO rotation in header['{key}'] = {rotation_value:.3f}°")
                return rotation_value
        
        # 3. ヘッダーから観測時刻と衛星を判定してフォールバック計算
        print("Debug: No rotation angle found in header, calculating from observation time...")
        
        if spacecraft is None:
            observatory = header.get('OBSRVTRY', header.get('TELESCOP', ''))
            if 'STEREO_A' in observatory.upper() or 'STA' in observatory.upper():
                spacecraft = 'A'
            elif 'STEREO_B' in observatory.upper() or 'STB' in observatory.upper():
                spacecraft = 'B'
            else:
                spacecraft = 'A'  # デフォルト
        
        # 観測時刻を取得
        obstime_str = header.get('DATE-OBS', header.get('DATE_OBS'))
        if obstime_str:
            obstime = Time(obstime_str)
            pa_north = STEREOCoordinateTransform.get_solar_north_angle(obstime, spacecraft)
            calculated_angle = pa_north.to(u.deg).value
            print(f"Debug: Calculated rotation angle = {calculated_angle:.3f}° for {spacecraft} at {obstime_str}")
            return calculated_angle
        
        print("Debug: No observation time found, returning default rotation = 0.0°")
        return 0.0  # デフォルト
    
    @staticmethod
    def correct_pointing(header, image_shape):
        """
        STEREO衛星のポインティング補正
        
        SSWIDLのcor_point.proに対応
        
        Parameters:
        -----------
        header : astropy.io.fits.Header
            FITSヘッダー
        image_shape : tuple
            画像サイズ (ny, nx)
            
        Returns:
        --------
        updated_header : astropy.io.fits.Header
            ポインティング補正済みヘッダー
        """
        updated_header = header.copy()
        
        # 基本的なWCS情報の確認・補正
        ny, nx = image_shape
        
        # CRPIXの確認・補正
        if 'CRPIX1' not in updated_header:
            updated_header['CRPIX1'] = nx / 2.0 + 0.5
        if 'CRPIX2' not in updated_header:
            updated_header['CRPIX2'] = ny / 2.0 + 0.5
        
        # CDELTの確認・補正
        if 'CDELT1' not in updated_header:
            # COR1のデフォルト画素スケール
            updated_header['CDELT1'] = 7.5  # arcsec/pixel
        if 'CDELT2' not in updated_header:
            updated_header['CDELT2'] = 7.5  # arcsec/pixel
        
        # CRVALの確認・補正
        if 'CRVAL1' not in updated_header:
            updated_header['CRVAL1'] = 0.0  # Sun center
        if 'CRVAL2' not in updated_header:
            updated_header['CRVAL2'] = 0.0  # Sun center
        
        # CTYPEの設定
        updated_header['CTYPE1'] = 'HPLN-TAN'
        updated_header['CTYPE2'] = 'HPLT-TAN'
        updated_header['CUNIT1'] = 'arcsec'
        updated_header['CUNIT2'] = 'arcsec'
        
        return updated_header
    
    @staticmethod
    def create_stereo_wcs(header, image_shape):
        """
        STEREO観測用のWCSオブジェクトを作成
        
        Parameters:
        -----------
        header : astropy.io.fits.Header
            FITSヘッダー
        image_shape : tuple
            画像サイズ (ny, nx)
            
        Returns:
        --------
        wcs : astropy.wcs.WCS
            WCSオブジェクト
        """
        # ポインティング補正
        corrected_header = STEREOCoordinateTransform.correct_pointing(header, image_shape)
        
        # 回転角の追加
        crota = STEREOCoordinateTransform.get_crota_angle(corrected_header)
        corrected_header['CROTA2'] = crota
        
        # WCSオブジェクトの作成
        wcs = WCS(corrected_header)
        
        return wcs


class STEREOVignettingCorrection:
    """STEREO/COR1 ビネッティング補正クラス"""
    
    @staticmethod
    def create_vignetting_map(image_shape, header=None, method='polynomial'):
        """
        ビネッティング補正マップを生成
        中心は画像中心ではなく、headerがあればWCS上の太陽中心を使う。
        """
        ny, nx = image_shape

        if header is not None:
            center_x, center_y = STEREOCoordinateTransform.get_solar_center_pixel(
                header, image_shape, origin=0
            )
        else:
            center_x = (nx - 1) / 2.0
            center_y = (ny - 1) / 2.0

        y, x = np.ogrid[:ny, :nx]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # WCS太陽中心から最も近い端までを有効半径とする
        max_radius = min(center_x, center_y, (nx - 1) - center_x, (ny - 1) - center_y)
        if not np.isfinite(max_radius) or max_radius <= 0:
            max_radius = min(nx, ny) / 2.0

        r_norm = r / max_radius

        if method == 'polynomial':
            vignetting_map = (1.0 - 0.15 * r_norm**2 +
                              0.08 * r_norm**4 - 0.02 * r_norm**6)
        elif method == 'radial':
            vignetting_map = 1.0 / (1.0 + 0.1 * r_norm**2)
        else:
            vignetting_map = np.exp(-0.1 * r_norm**2)

        vignetting_map = np.clip(vignetting_map, 0.1, 1.0)
        return vignetting_map.astype(np.float32)
    
        
    @staticmethod
    def apply_flat_field(image, flat_field=None):
        """
        フラットフィールド補正を適用
        
        Parameters:
        -----------
        image : numpy.ndarray
            入力画像
        flat_field : numpy.ndarray, optional
            フラットフィールド画像
            
        Returns:
        --------
        corrected_image : numpy.ndarray
            補正済み画像
        """
        if flat_field is None:
            # デフォルトのフラットフィールドを作成
            flat_field = STEREOVignettingCorrection.create_vignetting_map(image.shape)
        
        # ゼロ除算を避ける
        safe_flat = np.where(flat_field > 0.01, flat_field, 1.0)
        corrected_image = image / safe_flat
        
        return corrected_image