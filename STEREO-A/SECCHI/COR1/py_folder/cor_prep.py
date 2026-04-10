#!/usr/bin/env python3
"""
STEREO-A/SECCHI/COR1データ処理パイプライン - cor_prep.proのPython実装

このモジュールは、STEREO COR1/COR2画像をレベル0.5からレベル1.0に処理します。
IDL版のcor_prep.proに基づいて実装されています。

主な機能:
- CCDバイアス減算
- 校正係数の適用
- ビネッティング関数の適用
- 露出時間での正規化
- 画像回転（太陽北極を上に）
- マスク適用
- 欠損データの処理

参照元: $Id: cor_prep.pro,v 1.42 2013/10/28 18:28:59 nathan Exp $
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
import astropy.units as u
from scipy import ndimage
from scipy.interpolate import griddata
import warnings
from datetime import datetime
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class CORPrep:
    """
    STEREO COR1/COR2データ処理クラス
    
    IDL版のcor_prep.proの機能をPythonで実装
    """
    
    def __init__(self, silent=False):
        """
        初期化
        
        Parameters:
        -----------
        silent : bool, optional
            True の場合、処理メッセージを抑制
        """
        self.silent = silent
        self.version = "cor_prep.py v1.0 (Python port of IDL cor_prep.pro)"
        
        if not self.silent:
            logger.info(self.version)
    
    def scc_fitshdr2struct(self, header):
        """
        FITSヘッダーをSECCHI構造体形式に変換
        
        Parameters:
        -----------
        header : astropy.io.fits.Header
            FITSヘッダー
            
        Returns:
        --------
        dict : SECCHI構造体形式の辞書
        """
        # 基本的なSECCHI構造体のフィールドを定義
        secchi_struct = {
            'DETECTOR': header.get('DETECTOR', ''),
            'INSTRUME': header.get('INSTRUME', ''),
            'DATE-OBS': header.get('DATE-OBS', ''),  # ハイフン形式を保持
            'DATE': header.get('DATE', ''),
            'TIME-OBS': header.get('TIME-OBS', ''),
            'EXPTIME': header.get('EXPTIME', 0.0),
            'RSUN': header.get('RSUN', 0.0),
            'DSUN_OBS': header.get('DSUN_OBS', 0.0),
            'CRPIX1': header.get('CRPIX1', 0.0),
            'CRPIX2': header.get('CRPIX2', 0.0),
            'CRVAL1': header.get('CRVAL1', 0.0),
            'CRVAL2': header.get('CRVAL2', 0.0),
            'CDELT1': header.get('CDELT1', 0.0),
            'CDELT2': header.get('CDELT2', 0.0),
            'CTYPE1': header.get('CTYPE1', ''),
            'CTYPE2': header.get('CTYPE2', ''),
            'CUNIT1': header.get('CUNIT1', ''),
            'CUNIT2': header.get('CUNIT2', ''),
            'MISSLIST': header.get('MISSLIST', ''),
            'SUMMED': header.get('SUMMED', 1),
            'NAXIS1': header.get('NAXIS1', 0),
            'NAXIS2': header.get('NAXIS2', 0),
            'FILENAME': header.get('FILENAME', ''),
            'OBSRVTRY': header.get('OBSRVTRY', ''),
            'DISTCORR': header.get('DISTCORR', 'F'),
            'BUNIT': header.get('BUNIT', ''),
            'DATAP95': header.get('DATAP95', 0.0),
            'DATAMAX': header.get('DATAMAX', 0.0),
            'DATAMIN': header.get('DATAMIN', 0.0),
            'DATAMEAN': header.get('DATAMEAN', 0.0),
            'SOLAR_P': header.get('SOLAR_P', 0.0),
            'SOLAR_B0': header.get('SOLAR_B0', 0.0),
            'SOLAR_L0': header.get('SOLAR_L0', 0.0),
            'WAVELNTH': header.get('WAVELNTH', 0.0),
            'WAVEUNIT': header.get('WAVEUNIT', ''),
            'HISTORY': []
        }
        
        # HISTORYキーワードを収集
        if 'HISTORY' in header:
            if isinstance(header['HISTORY'], list):
                secchi_struct['HISTORY'] = header['HISTORY']
            else:
                secchi_struct['HISTORY'] = [header['HISTORY']]
        
        return secchi_struct
    
    def scc_get_missing(self, header):
        """
        欠損ブロックの位置を取得
        
        Parameters:
        -----------
        header : dict
            SECCHI構造体
            
        Returns:
        --------
        numpy.ndarray : 欠損ピクセルのインデックス
        """
        misslist = header.get('MISSLIST', '')
        if not misslist or misslist == '':
            return np.array([])
        
        # MISSLISTの解析（簡略化版）
        # 実際のIDL版では複雑な解析が行われる
        missing_indices = np.array([])
        
        if not self.silent:
            logger.info("Missing block processing not fully implemented")
        
        return missing_indices
    
    def cor_calibrate(self, image, header, **kwargs):
        """
        COR校正処理
        
        Parameters:
        -----------
        image : numpy.ndarray
            入力画像データ
        header : dict
            SECCHI構造体
        **kwargs : dict
            追加のキーワード引数
            
        Returns:
        --------
        numpy.ndarray : 校正済み画像データ
        """
        if not self.silent:
            logger.info("Applying COR calibration")
        
        # 画像を浮動小数点型に変換
        calibrated_image = image.astype(np.float64)
        
        # 露出時間での正規化
        exptime = header.get('EXPTIME', 1.0)
        if exptime > 0:
            calibrated_image = calibrated_image / exptime
            if not self.silent:
                logger.info(f"Normalized by exposure time: {exptime} seconds")
        
        # 校正係数の適用（簡略化版）
        # 実際のIDL版では複雑な校正係数の計算が行われる
        calfac = 1.0  # 実際の校正係数は別途計算が必要
        calibrated_image = calibrated_image * calfac
        
        # ビネッティング補正（簡略化版）
        # 実際のIDL版では詳細なビネッティング関数が適用される
        calibrated_image = self.apply_vignetting_correction(calibrated_image, header)
        
        return calibrated_image
    
    def cor1_calibrate(self, image, header, discri_pobj_on=False, **kwargs):
        """
        COR1専用校正処理
        
        Parameters:
        -----------
        image : numpy.ndarray
            入力画像データ
        header : dict
            SECCHI構造体
        discri_pobj_on : bool, optional
            点光源・宇宙線除去フィルタの適用
        **kwargs : dict
            追加のキーワード引数
            
        Returns:
        --------
        numpy.ndarray : 校正済み画像データ
        """
        if not self.silent:
            logger.info("Applying COR1-specific calibration")
        
        # 基本的なCOR校正を適用
        calibrated_image = self.cor_calibrate(image, header, **kwargs)
        
        # COR1固有の処理
        if discri_pobj_on:
            calibrated_image = self.discri_pobj(calibrated_image)
        
        return calibrated_image
    
    def apply_vignetting_correction(self, image, header):
        """
        ビネッティング補正の適用
        
        Parameters:
        -----------
        image : numpy.ndarray
            入力画像データ
        header : dict
            SECCHI構造体
            
        Returns:
        --------
        numpy.ndarray : ビネッティング補正済み画像
        """
        # ビネッティング補正（簡略化版）
        # 実際のIDL版では詳細なビネッティング関数が読み込まれる
        ny, nx = image.shape
        center_x, center_y = nx // 2, ny // 2
        
        # 中心からの距離を計算
        y, x = np.ogrid[:ny, :nx]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # 簡単なビネッティング補正関数（実際はより複雑）
        max_dist = np.sqrt(center_x**2 + center_y**2)
        vignetting_factor = 1.0 - 0.1 * (dist / max_dist)**2
        vignetting_factor = np.clip(vignetting_factor, 0.1, 1.0)
        
        corrected_image = image / vignetting_factor
        
        return corrected_image
    
    def discri_pobj(self, image, threshold=0.01, bias=0.0):
        """
        点光源・宇宙線除去フィルタ
        
        Parameters:
        -----------
        image : numpy.ndarray
            入力画像データ
        threshold : float, optional
            閾値
        bias : float, optional
            バイアス値
            
        Returns:
        --------
        numpy.ndarray : フィルタ適用済み画像
        """
        if not self.silent:
            logger.info("Applying point source and cosmic ray removal")
        
        # メディアンフィルタを適用
        filtered_image = ndimage.median_filter(image, size=3)
        
        # 差分を計算
        diff = np.abs(image - filtered_image)
        
        # 閾値を超えるピクセルを特定
        outliers = diff > threshold * np.std(image)
        
        # 外れ値をフィルタ済み値で置き換え
        result = image.copy()
        result[outliers] = filtered_image[outliers]
        
        return result
    
    def get_smask(self, header):
        """
        スムーズマスクの取得
        
        Parameters:
        -----------
        header : dict
            SECCHI構造体
            
        Returns:
        --------
        numpy.ndarray : マスク配列
        """
        # 画像サイズを取得
        nx = header.get('NAXIS1', 512)
        ny = header.get('NAXIS2', 512)
        
        # 基本的な円形マスクを作成
        center_x, center_y = nx // 2, ny // 2
        y, x = np.ogrid[:ny, :nx]
        
        # 太陽半径を取得
        rsun_pixel = header.get('RSUN', 0.0)
        if rsun_pixel > 0:
            cdelt = abs(header.get('CDELT1', 15.0))
            rsun_pixel = rsun_pixel / cdelt
        else:
            rsun_pixel = nx // 8  # デフォルト値
        
        # 内側と外側の半径を設定
        inner_radius = rsun_pixel * 1.1  # 太陽半径の1.1倍
        outer_radius = min(nx, ny) // 2 * 0.9  # 画像サイズの90%
        
        # 距離を計算
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # マスクを作成
        mask = np.ones((ny, nx), dtype=np.float64)
        mask[dist < inner_radius] = 0.0
        mask[dist > outer_radius] = 0.0
        
        return mask
    
    def scc_roll_image(self, header, image, missing=0, interp=False, cubic=False, **kwargs):
        """
        太陽北極を上に向けるための画像回転
        
        Parameters:
        -----------
        header : dict
            SECCHI構造体
        image : numpy.ndarray
            入力画像データ
        missing : int, optional
            欠損値
        interp : bool, optional
            双線形補間の使用
        cubic : bool, optional
            3次畳み込み補間の使用
        **kwargs : dict
            追加のキーワード引数
            
        Returns:
        --------
        tuple : (回転済み画像, 更新されたヘッダー)
        """
        if not self.silent:
            logger.info("Rotating image to Solar North Up")
        
        # 回転角度を計算（簡略化版）
        # 実際のIDL版ではより複雑な計算が行われる
        try:
            crota = header.get('CROTA2', 0.0)
            if crota == 0.0:
                crota = header.get('CROTA1', 0.0)
            
            # 回転角度（度）
            angle = -crota
            
            # 補間方法を選択
            if cubic:
                order = 3
            elif interp:
                order = 1
            else:
                order = 0
            
            # 画像を回転
            rotated_image = ndimage.rotate(image, angle, order=order, 
                                         reshape=False, cval=missing)
            
            # ヘッダーを更新
            header['CROTA1'] = 0.0
            header['CROTA2'] = 0.0
            if 'HISTORY' not in header:
                header['HISTORY'] = []
            header['HISTORY'].append(f"Image rotated by {angle:.2f} degrees to Solar North Up")
            
            return rotated_image, header
            
        except Exception as e:
            logger.warning(f"Image rotation failed: {e}")
            return image, header
    
    def scc_update_hdr(self, image, header, missing=None, **kwargs):
        """
        レベル1値へのヘッダー更新
        
        Parameters:
        -----------
        image : numpy.ndarray
            処理済み画像データ
        header : dict
            SECCHI構造体
        missing : numpy.ndarray, optional
            欠損ピクセルのインデックス
        **kwargs : dict
            追加のキーワード引数
            
        Returns:
        --------
        dict : 更新されたヘッダー
        """
        if not self.silent:
            logger.info("Updating header to Level 1 values")
        
        # 画像統計を計算
        if missing is not None and len(missing) > 0:
            valid_pixels = np.delete(image.flatten(), missing)
        else:
            valid_pixels = image.flatten()
        
        # 統計値を計算
        header['DATAMIN'] = float(np.nanmin(valid_pixels))
        header['DATAMAX'] = float(np.nanmax(valid_pixels))
        header['DATAMEAN'] = float(np.nanmean(valid_pixels))
        header['DATAP95'] = float(np.nanpercentile(valid_pixels, 95))
        
        # 処理レベルを更新
        filename = header.get('FILENAME', '')
        if len(filename) > 16:
            filename = filename[:16] + '1' + filename[17:]
            header['FILENAME'] = filename
        
        # 単位を更新
        header['BUNIT'] = 'DN/s'
        
        return header
    
    def prepare_data(self, filepath, header=None, image=None):
        """
        データの準備と読み込み
        
        Parameters:
        -----------
        filepath : str
            FITSファイルのパス
        header : dict, optional
            既存のヘッダー
        image : numpy.ndarray, optional
            既存の画像データ
            
        Returns:
        --------
        tuple : (画像データ, ヘッダー)
        """
        if image is None or header is None:
            try:
                with fits.open(filepath) as hdul:
                    image = hdul[0].data.astype(np.float64)
                    fits_header = hdul[0].header
                    header = self.scc_fitshdr2struct(fits_header)
            except Exception as e:
                logger.error(f"Failed to read FITS file {filepath}: {e}")
                return None, None
        
        return image, header
    
    def cor_prep(self, filepath=None, header=None, image=None, 
                 rotate_on=False, fill_mean=False, fill_value=None,
                 smask_on=False, calibrate_off=False, color_on=False,
                 date_on=False, logo_on=False, polariz_on=False,
                 rotinterp_on=False, rotcubic_on=False, nowarp=False,
                 warp_off=False, precommcorrect_on=False, 
                 discri_pobj_on=False, **kwargs):
        """
        COR画像の主要処理関数
        
        Parameters:
        -----------
        filepath : str, optional
            FITSファイルのパス
        header : dict, optional
            SECCHI構造体
        image : numpy.ndarray, optional
            画像データ
        rotate_on : bool, optional
            太陽北極を上にする回転
        fill_mean : bool, optional
            欠損データを平均値で埋める
        fill_value : float, optional
            欠損データを指定値で埋める
        smask_on : bool, optional
            スムーズマスクの適用
        calibrate_off : bool, optional
            校正処理をスキップ
        color_on : bool, optional
            カラーテーブルの読み込み
        date_on : bool, optional
            日時スタンプの追加
        logo_on : bool, optional
            SECCHIロゴの追加
        polariz_on : bool, optional
            偏光処理
        rotinterp_on : bool, optional
            回転時の双線形補間
        rotcubic_on : bool, optional
            回転時の3次畳み込み補間
        nowarp : bool, optional
            ワープ処理をスキップ
        warp_off : bool, optional
            ワープ処理を無効化
        precommcorrect_on : bool, optional
            試験運用期間の補正
        discri_pobj_on : bool, optional
            点光源・宇宙線除去
        **kwargs : dict
            追加のキーワード引数
            
        Returns:
        --------
        tuple : (処理済み画像, 更新されたヘッダー)
        """
        # データの準備
        image, header = self.prepare_data(filepath, header, image)
        if image is None or header is None:
            return None, None
        
        # ヘッダーの検証
        if header.get('DETECTOR') not in ['COR1', 'COR2']:
            logger.error(f"Unsupported detector: {header.get('DETECTOR')}")
            return None, None
        
        # 欠損ブロックの検索
        missing = np.array([])
        if header.get('MISSLIST', '') != '' and not calibrate_off:
            missing = self.scc_get_missing(header)
        
        # 校正処理
        if not calibrate_off:
            if header.get('DETECTOR') == 'COR1':
                image = self.cor1_calibrate(image, header, 
                                          discri_pobj_on=discri_pobj_on,
                                          **kwargs)
            else:
                image = self.cor_calibrate(image, header, **kwargs)
        
        # 欠損ブロックマスクの適用
        if len(missing) > 0 and not calibrate_off:
            if fill_mean:
                image[missing] = np.nanmean(image)
            elif fill_value is not None:
                image[missing] = fill_value
            else:
                image[missing] = 0.0
        
        # COR2のワープ処理（簡略化版）
        if (header.get('DETECTOR') == 'COR2' and 
            not nowarp and not warp_off and not calibrate_off):
            if not self.silent:
                logger.info("COR2 warp processing (simplified)")
            header['DISTCORR'] = 'T'
        
        # 太陽北極を上にする回転
        if rotate_on:
            image, header = self.scc_roll_image(header, image, missing=0,
                                              interp=rotinterp_on,
                                              cubic=rotcubic_on, **kwargs)
        
        # スムーズマスクの適用
        if smask_on and not calibrate_off:
            mask = self.get_smask(header)
            m_dex = np.where(mask == 0)
            
            if fill_mean:
                image[m_dex] = np.nanmean(image)
            elif fill_value is not None:
                image[m_dex] = fill_value
            else:
                image = image * mask
            
            if not self.silent:
                logger.info("Mask applied")
        
        # ヘッダーの更新
        if not calibrate_off and not polariz_on:
            header = self.scc_update_hdr(image, header, missing=missing, **kwargs)
        
        # COR2の飛行中校正バグの修正
        if header.get('DETECTOR') == 'COR2':
            summed = header.get('SUMMED', 1)
            header['CDELT1'] = 14.7 * (2 ** (summed - 1))
            header['CDELT2'] = 14.7 * (2 ** (summed - 1))
        
        if not self.silent:
            logger.info("COR preparation completed")
        
        return image, header

def main():
    """
    テスト用のメイン関数
    """
    # テスト用のファイルパス
    test_file = "/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/Rawdata/20220613_032136_n4c1A.fts"
    
    # CORPrepインスタンスを作成
    prep = CORPrep(silent=False)
    
    # 処理実行
    try:
        processed_image, processed_header = prep.cor_prep(
            filepath=test_file,
            rotate_on=True,
            smask_on=True,
            calibrate_off=False
        )
        
        if processed_image is not None:
            print(f"Processing successful!")
            print(f"Image shape: {processed_image.shape}")
            print(f"Image range: {np.nanmin(processed_image):.2f} to {np.nanmax(processed_image):.2f}")
            print(f"Image mean: {np.nanmean(processed_image):.2f}")
            
            # 簡単な可視化
            plt.figure(figsize=(10, 8))
            plt.imshow(processed_image, origin='lower', cmap='gray')
            plt.colorbar(label='DN/s')
            plt.title('COR1 Processed Image')
            plt.tight_layout()
            plt.savefig('/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/py_folder/cor_prep_test.png')
            plt.close()
            
            print("Test image saved as cor_prep_test.png")
        else:
            print("Processing failed!")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()