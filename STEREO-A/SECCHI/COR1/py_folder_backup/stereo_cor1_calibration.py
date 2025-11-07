#!/usr/bin/env python3
"""
STEREO-A/SECCHI/COR1 校正システム (SSWIDLベース)
SSWIDL cor_prep.pro, cor_calibrate.pro, secchi_prep.proの機能をPythonに移植

主な機能:
- バイアス減算
- 校正係数適用
- ビネッティング/フラットフィールド補正
- 露光時間正規化
- STEREO-A軌道の傾き補正 (太陽北極を上に回転)
- SEB IP補正
- 欠損ブロック処理
- スムージングマスク適用
"""

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from astropy.time import Time
import sunpy.map
import sunpy.coordinates
from scipy import ndimage
from scipy.interpolate import interp2d
import warnings


class STEREOCoordinates:
    """STEREO衛星の座標変換ユーティリティ"""
    
    @staticmethod
    def get_stereo_roll_angle(obstime, observatory='STEREO_A'):
        """
        STEREO衛星の軌道傾斜による太陽北極の回転角を計算
        
        Parameters:
        -----------
        obstime : str or astropy.time.Time
            観測時刻
        observatory : str
            衛星名 ('STEREO_A' or 'STEREO_B')
            
        Returns:
        --------
        roll_angle : astropy.units.Quantity
            太陽北極を上に向けるための回転角 (degrees)
        """
        if isinstance(obstime, str):
            obstime = Time(obstime)
            
        # STEREO衛星の軌道傾斜を考慮した近似計算
        # 実際のSSWIDLコードではより精密な軌道計算を行う
        days_since_launch = (obstime - Time('2006-10-26')).to(u.day).value
        
        if observatory.upper() == 'STEREO_A':
            # STEREO-Aは太陽の前方を周回、徐々に地球から離れる
            orbital_phase = (days_since_launch * 360 / 365.25) * u.degree
            roll_angle = -orbital_phase * 0.1  # 近似値
        else:
            # STEREO-Bは太陽の後方を周回
            orbital_phase = (days_since_launch * 360 / 365.25) * u.degree
            roll_angle = orbital_phase * 0.1  # 近似値
            
        return roll_angle
    
    @staticmethod
    def rotate_image_solar_north_up(image, header, missing_value=0.0, 
                                  interpolation='linear'):
        """
        太陽北極を画像の上に向けるように画像を回転
        
        Parameters:
        -----------
        image : numpy.ndarray
            回転する画像データ
        header : astropy.io.fits.Header
            FITSヘッダー
        missing_value : float
            欠損値の置換値
        interpolation : str
            補間方法 ('linear', 'cubic', 'nearest')
            
        Returns:
        --------
        rotated_image : numpy.ndarray
            回転された画像
        updated_header : astropy.io.fits.Header
            更新されたヘッダー
        """
        # CRROTキーワードから回転角を取得
        if 'CROTA2' in header:
            rotation_angle = header['CROTA2']
        elif 'CROTA1' in header:
            rotation_angle = header['CROTA1']
        elif 'CROTA' in header:
            rotation_angle = header['CROTA']
        else:
            # 観測時刻からSTEREO軌道の傾きを計算
            obstime = header.get('DATE-OBS', header.get('DATE_OBS'))
            if obstime:
                observatory = header.get('OBSRVTRY', 'STEREO_A')
                roll_angle = STEREOCoordinates.get_stereo_roll_angle(
                    obstime, observatory)
                rotation_angle = roll_angle.to(u.degree).value
            else:
                rotation_angle = 0.0
        
        if abs(rotation_angle) < 0.01:  # 0.01度未満なら回転しない
            return image.copy(), header.copy()
        
        print(f"Debug: Rotating image by {rotation_angle:.3f}° to align solar north up")
        
        # 画像中心を軸に回転
        center = (np.array(image.shape) - 1) / 2.0
        
        if interpolation == 'cubic':
            order = 3
        elif interpolation == 'linear':
            order = 1
        else:
            order = 0
            
        # 天体画像の標準的な回転（Y軸を上にして回転）
        rotated_image = ndimage.rotate(
            image, 
            rotation_angle,   # STEREO画像では時計回りが太陽北極を上に向ける
            axes=(0, 1),      # 標準的な画像軸順序
            reshape=False,
            order=order,
            cval=missing_value,
            prefilter=True
        )
        
        # ヘッダーの更新
        updated_header = header.copy()
        updated_header['CROTA1'] = 0.0
        updated_header['CROTA2'] = 0.0
        updated_header['HISTORY'] = f'Applied solar north rotation: {rotation_angle:.3f} deg'
        
        return rotated_image, updated_header


class STEREOCOR1Calibration:
    """STEREO/SECCHI/COR1 包括的校正クラス"""
    
    def __init__(self, silent=False):
        """
        初期化
        
        Parameters:
        -----------
        silent : bool
            詳細メッセージの表示を抑制
        """
        self.silent = silent
        self.history = []
        
    def log_message(self, message, level='info'):
        """ログメッセージの記録"""
        self.history.append(message)
        if not self.silent:
            print(f"[COR1 Calibration] {message}")
    
    def get_bias_mean(self, header):
        """
        CCDバイアス値を取得
        SSWIDLのget_biasmean.proに対応
        
        Parameters:
        -----------
        header : astropy.io.fits.Header
            FITSヘッダー
            
        Returns:
        --------
        bias_mean : float
            バイアス平均値
        """
        # ヘッダーからバイアス情報を取得
        if 'BIASMEAN' in header:
            bias_mean = float(header['BIASMEAN'])
        elif 'OFFSETCR' in header:
            bias_mean = float(header['OFFSETCR'])
        else:
            # COR1のデフォルトバイアス値（近似）
            bias_mean = 108.0  # DN
            
        return bias_mean
    
    def get_calibration_factor(self, header):
        """
        校正係数を取得
        SSWIDLのget_calfac.proに対応
        
        Parameters:
        -----------
        header : astropy.io.fits.Header
            FITSヘッダー
            
        Returns:
        --------
        calfac : float
            校正係数 (DN to MSB)
        """
        if 'CALFAC' in header:
            calfac = float(header['CALFAC'])
        else:
            # COR1のデフォルト校正係数
            calfac = 1.0e-9  # DN to Mean Solar Brightness
            
        # SUMMEDモードの補正
        if 'SUMMED' in header:
            summed = int(header['SUMMED'])
            if summed > 1:
                calfac *= (2 ** (summed - 1)) ** 2
                
        return calfac
    
    def get_exposure_time(self, header):
        """
        露光時間を取得
        SSWIDLのget_exptime.proに対応
        
        Parameters:
        -----------
        header : astropy.io.fits.Header
            FITSヘッダー
            
        Returns:
        --------
        exptime : float
            露光時間 (seconds)
        """
        if 'EXPTIME' in header:
            exptime = float(header['EXPTIME'])
        elif 'EXPOSURE' in header:
            exptime = float(header['EXPOSURE'])
        else:
            exptime = 1.0  # デフォルト
            
        return max(exptime, 0.001)  # 最小値制限
    
    def get_vignetting_correction(self, header, image_shape):
        """
        ビネッティング補正マップを生成
        SSWIDLのget_calimg.proに対応
        
        Parameters:
        -----------
        header : astropy.io.fits.Header
            FITSヘッダー
        image_shape : tuple
            画像サイズ (ny, nx)
            
        Returns:
        --------
        vignetting_map : numpy.ndarray
            ビネッティング補正マップ
        """
        ny, nx = image_shape
        
        # 画像中心を計算
        center_x = nx / 2.0
        center_y = ny / 2.0
        
        # 半径マップを作成
        y, x = np.ogrid[:ny, :nx]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # ビネッティング関数（多項式近似）
        # 実際のSSWIDLではより複雑な補正関数を使用
        max_radius = min(nx, ny) / 2.0
        r_norm = r / max_radius
        
        # COR1用のビネッティング補正近似
        vignetting_map = 1.0 - 0.1 * r_norm**2 + 0.05 * r_norm**4
        vignetting_map = np.clip(vignetting_map, 0.1, 1.0)
        
        return vignetting_map.astype(np.float32)
    
    def apply_seb_ip_correction(self, image, header):
        """
        SEB IP補正を適用
        SSWIDLのscc_sebip.proに対応
        
        Parameters:
        -----------
        image : numpy.ndarray
            画像データ
        header : astropy.io.fits.Header
            FITSヘッダー
            
        Returns:
        --------
        corrected_image : numpy.ndarray
            SEB IP補正済み画像
        """
        # SEB IPフラグをチェック
        if 'SEBIPFLG' in header and header['SEBIPFLG'] == 'T':
            # 既に補正済み
            return image.copy()
        
        # SEB IP補正（簡略化バージョン）
        # 実際のSSWIDLではより複雑な補正処理
        corrected_image = image.copy()
        
        # 基本的な補正処理
        if 'IPSUM' in header:
            ipsum = int(header['IPSUM'])
            if ipsum > 1:
                correction_factor = 2 ** (ipsum - 1)
                corrected_image = corrected_image / correction_factor
                self.log_message(f"Applied SEB IP correction: factor={correction_factor}")
        
        return corrected_image
    
    def get_missing_blocks(self, header):
        """
        欠損ブロック情報を取得
        SSWIDLのscc_get_missing.proに対応
        
        Parameters:
        -----------
        header : astropy.io.fits.Header
            FITSヘッダー
            
        Returns:
        --------
        missing_mask : numpy.ndarray or None
            欠損ブロックマスク
        """
        if 'MISSLIST' not in header or not header['MISSLIST']:
            return None
            
        # 欠損ブロックリストの解析
        misslist = header['MISSLIST']
        # 実際の実装では、欠損ブロック情報から座標を計算
        # ここでは簡略化
        
        return None  # 簡略化のためNoneを返す
    
    def apply_smooth_mask(self, image, header, fill_value=0.0):
        """
        スムージングマスクを適用
        SSWIDLのget_smask.proに対応
        
        Parameters:
        -----------
        image : numpy.ndarray
            画像データ
        header : astropy.io.fits.Header
            FITSヘッダー
        fill_value : float
            マスク領域の置換値
            
        Returns:
        --------
        masked_image : numpy.ndarray
            マスク適用済み画像
        """
        ny, nx = image.shape
        center_x, center_y = nx // 2, ny // 2
        
        # 基本的な円形マスクを作成
        y, x = np.ogrid[:ny, :nx]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # COR1の視野に基づくマスク半径
        if 'RSUN' in header:
            rsun_pix = float(header['RSUN'])
        else:
            rsun_pix = min(nx, ny) / 20.0  # 近似値
            
        # 内側マスク（太陽円盤）と外側マスク
        inner_radius = rsun_pix * 1.1  # 太陽半径の1.1倍
        outer_radius = min(nx, ny) / 2.0 * 0.95  # 検出器端の95%
        
        mask = np.ones_like(image, dtype=bool)
        mask[r < inner_radius] = False  # 内側をマスク
        mask[r > outer_radius] = False  # 外側をマスク
        
        masked_image = image.copy()
        masked_image[~mask] = fill_value
        
        self.log_message("Applied smooth mask")
        return masked_image
    
    def cor1_calibrate(self, image, header, 
                      bias_off=False, calfac_off=False, 
                      exptime_off=False, calimg_off=False, 
                      sebip_off=False):
        """
        COR1画像の校正処理
        SSWIDLのcor_calibrate.proに対応
        
        Parameters:
        -----------
        image : numpy.ndarray
            生画像データ (DN)
        header : astropy.io.fits.Header
            FITSヘッダー
        bias_off : bool
            バイアス減算をスキップ
        calfac_off : bool
            校正係数適用をスキップ
        exptime_off : bool
            露光時間正規化をスキップ
        calimg_off : bool
            ビネッティング補正をスキップ
        sebip_off : bool
            SEB IP補正をスキップ
            
        Returns:
        --------
        calibrated_image : numpy.ndarray
            校正済み画像 (MSB or DN/s)
        updated_header : astropy.io.fits.Header
            更新されたヘッダー
        """
        calibrated_image = image.astype(np.float32)
        updated_header = header.copy()
        
        # SEB IP補正
        if not sebip_off:
            calibrated_image = self.apply_seb_ip_correction(calibrated_image, updated_header)
        
        # 露光時間取得
        if exptime_off:
            exptime = 1.0
        else:
            exptime = self.get_exposure_time(updated_header)
            if exptime != 1.0:
                self.log_message(f"Exposure time normalization: {exptime:.3f}s")
        
        # バイアス減算
        if bias_off:
            bias_mean = 0.0
        else:
            bias_mean = self.get_bias_mean(updated_header)
            if bias_mean != 0.0:
                updated_header['OFFSETCR'] = bias_mean
                self.log_message(f"Bias subtracted: {bias_mean:.1f} DN")
        
        # ビネッティング補正
        if calimg_off:
            vignetting_map = 1.0
        else:
            vignetting_map = self.get_vignetting_correction(updated_header, calibrated_image.shape)
            self.log_message("Applied vignetting correction")
        
        # 校正係数取得
        if calfac_off:
            calfac = 1.0
        else:
            calfac = self.get_calibration_factor(updated_header)
            if calfac != 1.0:
                updated_header['CALFAC'] = calfac
                self.log_message(f"Applied calibration factor: {calfac:.2e}")
        
        # 校正の適用
        calibrated_image = ((calibrated_image - bias_mean) * calfac / exptime) / vignetting_map
        
        # ヘッダーの更新
        updated_header['BUNIT'] = 'MSB' if not calfac_off else 'DN/s'
        for msg in self.history:
            updated_header['HISTORY'] = msg
        
        return calibrated_image, updated_header
    
    def cor1_prep(self, image, header,
                 rotate_on=True, smask_on=True, calibrate_off=False,
                 fill_value=0.0, **kwargs):
        """
        COR1画像の完全な前処理
        SSWIDLのcor_prep.proに対応
        
        Parameters:
        -----------
        image : numpy.ndarray
            生画像データ
        header : astropy.io.fits.Header
            FITSヘッダー
        rotate_on : bool
            太陽北極を上に回転
        smask_on : bool
            スムージングマスクを適用
        calibrate_off : bool
            校正処理をスキップ
        fill_value : float
            欠損値の置換値
            
        Returns:
        --------
        processed_image : numpy.ndarray
            処理済み画像
        updated_header : astropy.io.fits.Header
            更新されたヘッダー
        """
        self.log_message("Starting COR1 preparation...")
        
        processed_image = image.copy()
        updated_header = header.copy()
        
        # 欠損ブロックの処理
        missing_mask = self.get_missing_blocks(updated_header)
        if missing_mask is not None:
            processed_image[missing_mask] = fill_value
            self.log_message("Applied missing block mask")
        
        # 校正処理
        if not calibrate_off:
            processed_image, updated_header = self.cor1_calibrate(
                processed_image, updated_header, **kwargs)
        
        # 太陽北極を上に回転
        if rotate_on:
            processed_image, updated_header = STEREOCoordinates.rotate_image_solar_north_up(
                processed_image, updated_header, missing_value=fill_value)
            self.log_message("Applied solar north rotation")
        
        # スムージングマスクの適用
        if smask_on and not calibrate_off:
            processed_image = self.apply_smooth_mask(
                processed_image, updated_header, fill_value=fill_value)
        
        # ヘッダーにレベル情報を追加
        updated_header['LEVEL'] = '1.0'
        updated_header['HISTORY'] = 'Processed with STEREO COR1 Python calibration'
        
        self.log_message("COR1 preparation completed")
        return processed_image, updated_header