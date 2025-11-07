#!/usr/bin/env python3
"""
STEREO-A/SECCHI データ処理パイプライン - secchi_prep.proのPython実装

このモジュールは、STEREO SECCHIデータ（COR1, COR2, EUVI, HI1, HI2）を
レベル0.5からレベル1.0に処理し、FITS、PNG、JPEGファイルとして出力します。
IDL版のsecchi_prep.proに基づいて実装されています。

主な機能:
- 複数ファイルの一括処理
- 各検出器（COR1, COR2, EUVI, HI1, HI2）専用の処理
- 偏光処理（COR1, COR2）
- 画像トリミング
- 宇宙線除去
- 複数形式での出力（FITS, PNG, JPEG）

参照元: $Id: secchi_prep.pro,v 1.58 2013/11/07 10:34:45 nathan Exp $
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u
from scipy import ndimage
from scipy.interpolate import griddata
import warnings
from datetime import datetime
import logging
import glob
from pathlib import Path
import json

# 自作モジュールのインポート
from cor_prep import CORPrep

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SECCHIPrep:
    """
    STEREO SECCHIデータ処理クラス
    
    IDL版のsecchi_prep.proの機能をPythonで実装
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
        self.version = "secchi_prep.py v1.0 (Python port of IDL secchi_prep.pro)"
        
        # 各検出器の処理クラスを初期化
        self.cor_prep = CORPrep(silent=silent)
        
        if not self.silent:
            logger.info(self.version)
    
    def sccreadfits(self, filepath, silent=False):
        """
        SECCHI FITS ファイルの読み込み
        
        Parameters:
        -----------
        filepath : str
            FITSファイルのパス
        silent : bool, optional
            メッセージを抑制
            
        Returns:
        --------
        tuple : (画像データ, ヘッダー構造体, 文字列ヘッダー)
        """
        try:
            with fits.open(filepath) as hdul:
                image = hdul[0].data.astype(np.float64)
                fits_header = hdul[0].header
                header = self.cor_prep.scc_fitshdr2struct(fits_header)
                
                # ファイル名を設定
                header['FILENAME'] = os.path.basename(filepath)
                
                if not silent:
                    logger.info(f"Successfully read: {filepath}")
                
                return image, header, fits_header
                
        except Exception as e:
            logger.error(f"Failed to read FITS file {filepath}: {e}")
            return None, None, None
    
    def scc_make_array(self, filenames, headers=None, outsize=None, **kwargs):
        """
        出力配列の作成
        
        Parameters:
        -----------
        filenames : list
            ファイル名のリスト
        headers : list, optional
            ヘッダーのリスト
        outsize : tuple, optional
            出力サイズ
        **kwargs : dict
            追加のキーワード引数
            
        Returns:
        --------
        tuple : (画像配列, ヘッダー配列, 出力サイズ)
        """
        if isinstance(filenames, str):
            filenames = [filenames]
        
        num_files = len(filenames)
        
        # 最初のファイルからサイズを取得
        test_image, test_header, _ = self.sccreadfits(filenames[0], silent=True)
        if test_image is None:
            return None, None, None
        
        # 出力サイズの決定
        if outsize is None:
            outsize = test_image.shape
        
        # 配列の初期化
        images = np.zeros((outsize[0], outsize[1], num_files), dtype=np.float64)
        headers = []
        
        return images, headers, outsize
    
    def scc_img_trim(self, image, header, **kwargs):
        """
        画像のトリミング（オーバースキャン除去）
        
        Parameters:
        -----------
        image : numpy.ndarray
            入力画像
        header : dict
            ヘッダー構造体
        **kwargs : dict
            追加のキーワード引数
            
        Returns:
        --------
        numpy.ndarray : トリミング済み画像
        """
        # 簡略化版：基本的なトリミング
        # 実際のIDL版ではより複雑な処理が行われる
        
        detector = header.get('DETECTOR', '')
        
        if detector in ['COR1', 'COR2']:
            # CORの場合、通常は512x512にトリミング
            if image.shape[0] > 512 or image.shape[1] > 512:
                center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
                start_y, start_x = center_y - 256, center_x - 256
                end_y, end_x = start_y + 512, start_x + 512
                
                # 境界チェック
                start_y = max(0, start_y)
                start_x = max(0, start_x)
                end_y = min(image.shape[0], end_y)
                end_x = min(image.shape[1], end_x)
                
                trimmed_image = image[start_y:end_y, start_x:end_x]
                
                # ヘッダーの更新
                header['NAXIS1'] = trimmed_image.shape[1]
                header['NAXIS2'] = trimmed_image.shape[0]
                
                if not self.silent:
                    logger.info(f"Image trimmed from {image.shape} to {trimmed_image.shape}")
                
                return trimmed_image
        
        return image
    
    def scc_putin_array(self, image, header, outsize, **kwargs):
        """
        画像を出力配列に配置
        
        Parameters:
        -----------
        image : numpy.ndarray
            入力画像
        header : dict
            ヘッダー構造体
        outsize : tuple
            出力サイズ
        **kwargs : dict
            追加のキーワード引数
            
        Returns:
        --------
        numpy.ndarray : サイズ調整済み画像
        """
        if image.shape == outsize:
            return image
        
        # 画像をoutsize にリサイズ
        if image.shape[0] != outsize[0] or image.shape[1] != outsize[1]:
            # 簡単なリサイズ（実際はより複雑な処理が必要）
            from scipy.ndimage import zoom
            
            zoom_factors = (outsize[0] / image.shape[0], outsize[1] / image.shape[1])
            resized_image = zoom(image, zoom_factors, order=1)
            
            if not self.silent:
                logger.info(f"Image resized from {image.shape} to {resized_image.shape}")
            
            return resized_image
        
        return image
    
    def euvi_prep(self, header, image, **kwargs):
        """
        EUVI画像の処理（簡略化版）
        
        Parameters:
        -----------
        header : dict
            ヘッダー構造体
        image : numpy.ndarray
            画像データ
        **kwargs : dict
            追加のキーワード引数
            
        Returns:
        --------
        numpy.ndarray : 処理済み画像
        """
        if not self.silent:
            logger.info("Processing EUVI image (simplified)")
        
        # 基本的な校正処理
        processed_image = image.astype(np.float64)
        
        # 露出時間での正規化
        exptime = header.get('EXPTIME', 1.0)
        if exptime > 0:
            processed_image = processed_image / exptime
        
        # 簡単なフラットフィールド補正
        processed_image = self._apply_flat_field(processed_image, header)
        
        return processed_image
    
    def hi_prep(self, header, image, cosmics=None, **kwargs):
        """
        HI画像の処理（簡略化版）
        
        Parameters:
        -----------
        header : dict
            ヘッダー構造体
        image : numpy.ndarray
            画像データ
        cosmics : list, optional
            宇宙線除去レポート
        **kwargs : dict
            追加のキーワード引数
            
        Returns:
        --------
        tuple : (処理済み画像, 宇宙線レポート)
        """
        if not self.silent:
            logger.info("Processing HI image (simplified)")
        
        # 基本的な校正処理
        processed_image = image.astype(np.float64)
        
        # 宇宙線除去
        cosmic_report = self._remove_cosmic_rays(processed_image)
        
        # 露出時間での正規化
        exptime = header.get('EXPTIME', 1.0)
        if exptime > 0:
            processed_image = processed_image / exptime
        
        return processed_image, cosmic_report
    
    def _apply_flat_field(self, image, header):
        """
        フラットフィールド補正の適用
        
        Parameters:
        -----------
        image : numpy.ndarray
            入力画像
        header : dict
            ヘッダー構造体
            
        Returns:
        --------
        numpy.ndarray : 補正済み画像
        """
        # 簡略化版：基本的なフラットフィールド補正
        # 実際は適切なフラットフィールド画像を使用
        
        # 画像の中心からの距離に基づく簡単な補正
        ny, nx = image.shape
        center_y, center_x = ny // 2, nx // 2
        
        y, x = np.ogrid[:ny, :nx]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # 簡単な補正係数
        correction = 1.0 + 0.05 * (dist / max_dist)
        
        return image * correction
    
    def _remove_cosmic_rays(self, image):
        """
        宇宙線除去
        
        Parameters:
        -----------
        image : numpy.ndarray
            入力画像
            
        Returns:
        --------
        dict : 宇宙線除去レポート
        """
        # 簡略化版：基本的な宇宙線除去
        original_image = image.copy()
        
        # メディアンフィルタを適用
        filtered = ndimage.median_filter(image, size=3)
        
        # 差分を計算
        diff = np.abs(image - filtered)
        threshold = 5.0 * np.std(diff)
        
        # 宇宙線候補を特定
        cosmic_mask = diff > threshold
        
        # 宇宙線を除去
        image[cosmic_mask] = filtered[cosmic_mask]
        
        # レポートを作成
        cosmic_report = {
            'num_cosmic_rays': int(np.sum(cosmic_mask)),
            'fraction_affected': float(np.sum(cosmic_mask)) / image.size,
            'threshold_used': threshold
        }
        
        return cosmic_report
    
    def scc_bytscl(self, image, header):
        """
        画像のバイトスケーリング
        
        Parameters:
        -----------
        image : numpy.ndarray
            入力画像
        header : dict
            ヘッダー構造体
            
        Returns:
        --------
        numpy.ndarray : バイトスケール済み画像
        """
        # 95パーセンタイルを使用してスケーリング
        datap95 = header.get('DATAP95', np.nanpercentile(image, 95))
        
        # 0-255の範囲にスケール
        scaled = np.clip(image * 255.0 / datap95, 0, 255)
        
        return scaled.astype(np.uint8)
    
    def sccwritefits(self, filename, image, header, savepath=None, **kwargs):
        """
        FITS ファイルの書き込み
        
        Parameters:
        -----------
        filename : str
            ファイル名
        image : numpy.ndarray
            画像データ
        header : dict
            ヘッダー構造体
        savepath : str, optional
            保存先パス
        **kwargs : dict
            追加のキーワード引数
        """
        try:
            # 保存パスの設定
            if savepath:
                filepath = os.path.join(savepath, filename)
            else:
                filepath = filename
            
            # FITSヘッダーの作成
            fits_header = fits.Header()
            
            # 基本的なヘッダー情報をコピー
            for key, value in header.items():
                if key not in ['HISTORY'] and isinstance(value, (str, int, float, bool)):
                    try:
                        fits_header[key] = value
                    except:
                        pass
            
            # HISTORYの追加
            if 'HISTORY' in header:
                for hist in header['HISTORY']:
                    fits_header['HISTORY'] = hist
            
            # HDUの作成
            hdu = fits.PrimaryHDU(data=image, header=fits_header)
            
            # ファイルの書き込み
            hdu.writeto(filepath, overwrite=True)
            
            if not self.silent:
                logger.info(f"FITS file written: {filepath}")
                
        except Exception as e:
            logger.error(f"Failed to write FITS file {filename}: {e}")
    
    def process_files(self, filenames, headers=None, images=None, 
                     savepath=None, outsize=None, polariz_on=False,
                     precommcorrect_on=False, write_fits=False, 
                     write_fts=False, write_png=False, write_jpg=False,
                     debug=False, trim_off=False, silent=False,
                     cosmics=None, update_hdr_off=False, calimg_off=False,
                     exptime_off=False, calfac_off=False, 
                     nocalfac_butcorrforipsum=False, rectify=False,
                     discri_pobj_on=False, **kwargs):
        """
        ファイルの一括処理
        
        Parameters:
        -----------
        filenames : list or str
            処理するファイル名のリスト
        headers : list, optional
            出力ヘッダーのリスト
        images : numpy.ndarray, optional
            出力画像配列
        savepath : str, optional
            保存先パス
        outsize : tuple, optional
            出力サイズ
        polariz_on : bool, optional
            偏光処理
        precommcorrect_on : bool, optional
            試験運用期間補正
        write_fits : bool, optional
            FITS出力
        write_fts : bool, optional
            FTSファイル出力
        write_png : bool, optional
            PNG出力
        write_jpg : bool, optional
            JPEG出力
        debug : bool, optional
            デバッグモード
        trim_off : bool, optional
            トリミングを無効化
        silent : bool, optional
            メッセージを抑制
        cosmics : list, optional
            宇宙線除去レポート
        update_hdr_off : bool, optional
            ヘッダー更新を無効化
        calimg_off : bool, optional
            画像校正を無効化
        exptime_off : bool, optional
            露出時間正規化を無効化
        calfac_off : bool, optional
            校正係数を無効化
        nocalfac_butcorrforipsum : bool, optional
            校正係数なしでIP summing補正
        rectify : bool, optional
            補正の適用
        discri_pobj_on : bool, optional
            点光源・宇宙線除去
        **kwargs : dict
            追加のキーワード引数
            
        Returns:
        --------
        tuple : (処理済み画像配列, ヘッダー配列, 宇宙線レポート)
        """
        # 入力検証
        if isinstance(filenames, str):
            filenames = [filenames]
        
        if not filenames:
            logger.error("No input files provided")
            return None, None, None
        
        # 出力フラグの設定
        write_flag = any([write_fits, write_fts, write_png, write_jpg])
        
        # 出力配列の準備
        if headers is None or images is None:
            images, headers, outsize = self.scc_make_array(filenames, outsize=outsize, **kwargs)
            if images is None:
                return None, None, None
        
        # 処理結果の保存
        processed_headers = []
        processed_images = []
        cosmic_reports = []
        
        # 各ファイルを処理
        for i, filename in enumerate(filenames):
            if not self.silent:
                logger.info(f"Processing image {i+1} of {len(filenames)}: {filename}")
            
            try:
                # ファイルの読み込み
                image, header, str_header = self.sccreadfits(filename, silent=self.silent)
                if image is None:
                    continue
                
                # 校正係数の初期化
                if header.get('CALFAC', 0) == 0:
                    header['CALFAC'] = 1.0
                
                # 画像トリミング
                if not trim_off:
                    image = self.scc_img_trim(image, header, **kwargs)
                
                # 画像を出力配列に配置
                image = self.scc_putin_array(image, header, outsize, **kwargs)
                
                # 点光源・宇宙線除去
                if discri_pobj_on:
                    image = self.cor_prep.discri_pobj(image)
                
                # 検出器別の処理
                detector = header.get('DETECTOR', '')
                
                if detector == 'EUVI':
                    image = self.euvi_prep(header, image, **kwargs)
                    
                elif detector in ['COR1', 'COR2']:
                    # COR処理
                    image, header = self.cor_prep.cor_prep(
                        header=header, image=image,
                        discri_pobj_on=discri_pobj_on,
                        **kwargs
                    )
                    
                elif detector in ['HI1', 'HI2']:
                    image, cosmic_report = self.hi_prep(header, image, cosmics=cosmics, **kwargs)
                    cosmic_reports.append(cosmic_report)
                    
                else:
                    logger.warning(f"Unknown detector: {detector}")
                    continue
                
                # 処理済みデータを保存
                processed_images.append(image)
                processed_headers.append(header)
                
                # ファイル出力
                if write_flag:
                    self._write_output_files(image, header, filename, savepath,
                                           write_fits, write_fts, write_png, write_jpg,
                                           **kwargs)
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
        
        # 結果の配列化
        if processed_images:
            result_images = np.stack(processed_images, axis=2)
            result_headers = processed_headers
        else:
            result_images = None
            result_headers = None
        
        if not self.silent:
            logger.info("Processing completed!")
        
        return result_images, result_headers, cosmic_reports
    
    def _write_output_files(self, image, header, filename, savepath,
                           write_fits, write_fts, write_png, write_jpg, **kwargs):
        """
        出力ファイルの書き込み
        
        Parameters:
        -----------
        image : numpy.ndarray
            画像データ
        header : dict
            ヘッダー構造体
        filename : str
            元のファイル名
        savepath : str
            保存先パス
        write_fits : bool
            FITS出力
        write_fts : bool
            FTSファイル出力
        write_png : bool
            PNG出力
        write_jpg : bool
            JPEG出力
        **kwargs : dict
            追加のキーワード引数
        """
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        
        if write_fits or write_fts:
            fits_filename = base_filename + '.fits'
            self.sccwritefits(fits_filename, image, header, savepath=savepath, **kwargs)
        
        if write_png:
            png_filename = base_filename + '.png'
            png_path = os.path.join(savepath, png_filename) if savepath else png_filename
            
            # 画像をバイトスケール
            scaled_image = self.scc_bytscl(image, header)
            
            # PNG出力
            plt.figure(figsize=(10, 10))
            plt.imshow(scaled_image, origin='lower', cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            if not self.silent:
                logger.info(f"PNG file written: {png_path}")
        
        if write_jpg:
            jpg_filename = base_filename + '.jpg'
            jpg_path = os.path.join(savepath, jpg_filename) if savepath else jpg_filename
            
            # 画像をバイトスケール
            scaled_image = self.scc_bytscl(image, header)
            
            # JPEG出力
            plt.figure(figsize=(10, 10))
            plt.imshow(scaled_image, origin='lower', cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(jpg_path, format='jpeg', dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            if not self.silent:
                logger.info(f"JPEG file written: {jpg_path}")

def main():
    """
    テスト用のメイン関数
    """
    # テスト用のファイル
    test_files = [
        "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/20220613_032136_n4c1A.fts"
    ]
    
    # SECCHIPrepインスタンスを作成
    prep = SECCHIPrep(silent=False)
    
    # 処理実行
    try:
        processed_images, processed_headers, cosmic_reports = prep.process_files(
            filenames=test_files,
            write_png=True,
            write_fits=True,
            savepath="/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/py_folder",
            rotate_on=True,
            smask_on=True,
            calibrate_off=False
        )
        
        if processed_images is not None:
            print(f"Processing successful!")
            print(f"Processed {len(processed_headers)} images")
            print(f"Output shape: {processed_images.shape}")
            
            # 統計情報の表示
            for i, header in enumerate(processed_headers):
                print(f"Image {i+1}:")
                print(f"  Detector: {header.get('DETECTOR', 'Unknown')}")
                print(f"  Date: {header.get('DATE_OBS', 'Unknown')}")
                print(f"  Exposure: {header.get('EXPTIME', 0):.2f} s")
                print(f"  Data range: {header.get('DATAMIN', 0):.2f} to {header.get('DATAMAX', 0):.2f}")
        else:
            print("Processing failed!")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()