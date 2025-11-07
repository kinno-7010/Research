#!/usr/bin/env python3
"""
STEREO-A/SECCHI 統合データ処理パイプライン

このモジュールは、STEREO SECCHIデータの統合的な処理パイプラインを提供します。
IDL版のSSWIDLライブラリの機能をPythonで実装し、使いやすいインターフェースを提供します。

主な機能:
- 単一ファイルまたは複数ファイルの一括処理
- 検出器の自動識別
- 適切な校正処理の自動選択
- 柔軟な出力オプション
- 処理履歴の管理
- エラーハンドリング

使用例:
    # 単一ファイルの処理
    pipeline = STEREOPipeline()
    result = pipeline.process_file('cor1_file.fts')
    
    # 複数ファイルの一括処理
    results = pipeline.process_directory('/path/to/data/')
    
    # カスタム処理オプション
    result = pipeline.process_file('cor1_file.fts', 
                                   calibrate=True, 
                                   rotate=True,
                                   output_format=['fits', 'png'])

参照元: STEREO/SECCHI IDL Libraries
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u
from pathlib import Path
import warnings
from datetime import datetime
import logging
import json
import traceback
from typing import List, Dict, Any, Optional, Union, Tuple

# 自作モジュールのインポート
from cor_prep import CORPrep
from secchi_prep import SECCHIPrep
from secchi_utils import SECCHIUtils

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProcessingResult:
    """
    処理結果を格納するクラス
    """
    
    def __init__(self, success: bool = False, message: str = "", 
                 data: Optional[np.ndarray] = None, 
                 header: Optional[Dict] = None,
                 processing_info: Optional[Dict] = None):
        self.success = success
        self.message = message
        self.data = data
        self.header = header
        self.processing_info = processing_info or {}
        self.timestamp = datetime.now()
    
    def __str__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return f"ProcessingResult({status}): {self.message}"
    
    def to_dict(self):
        """辞書形式で結果を返す"""
        return {
            'success': self.success,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'processing_info': self.processing_info,
            'data_shape': self.data.shape if self.data is not None else None,
            'header_keys': list(self.header.keys()) if self.header else None
        }

class STEREOPipeline:
    """
    STEREO-A/SECCHI 統合データ処理パイプライン
    """
    
    def __init__(self, silent: bool = False, output_dir: Optional[str] = None):
        """
        パイプラインの初期化
        
        Parameters:
        -----------
        silent : bool, optional
            メッセージを抑制
        output_dir : str, optional
            出力ディレクトリ
        """
        self.silent = silent
        self.output_dir = output_dir
        self.version = "STEREO Pipeline v1.0"
        
        # 処理クラスの初期化
        self.cor_prep = CORPrep(silent=silent)
        self.secchi_prep = SECCHIPrep(silent=silent)
        self.utils = SECCHIUtils(silent=silent)
        
        # 処理履歴
        self.processing_history = []
        
        # サポートされている検出器
        self.supported_detectors = ['COR1', 'COR2', 'EUVI', 'HI1', 'HI2']
        
        # デフォルト処理オプション
        self.default_options = {
            'calibrate': True,
            'rotate': True,
            'trim': True,
            'mask': True,
            'cosmic_ray_removal': True,
            'background_subtraction': False,
            'vignetting_correction': True,
            'flat_field_correction': True,
            'dark_current_subtraction': True,
            'output_format': ['fits'],
            'save_processing_log': True
        }
        
        if not self.silent:
            logger.info(f"Initialized {self.version}")
    
    def identify_detector(self, filepath: str) -> Tuple[str, Dict]:
        """
        ファイルから検出器を識別
        
        Parameters:
        -----------
        filepath : str
            FITSファイルのパス
            
        Returns:
        --------
        tuple : (検出器名, ヘッダー情報)
        """
        try:
            with fits.open(filepath) as hdul:
                header = hdul[0].header
                detector = header.get('DETECTOR', '').strip()
                
                if detector not in self.supported_detectors:
                    raise ValueError(f"Unsupported detector: {detector}")
                
                # 基本的なヘッダー情報を抽出
                header_info = {
                    'detector': detector,
                    'instrument': header.get('INSTRUME', ''),
                    'date_obs': header.get('DATE-OBS', ''),
                    'exptime': header.get('EXPTIME', 0.0),
                    'naxis1': header.get('NAXIS1', 0),
                    'naxis2': header.get('NAXIS2', 0),
                    'filename': os.path.basename(filepath)
                }
                
                return detector, header_info
                
        except Exception as e:
            logger.error(f"Failed to identify detector for {filepath}: {e}")
            return '', {}
    
    def validate_file(self, filepath: str) -> bool:
        """
        ファイルの検証
        
        Parameters:
        -----------
        filepath : str
            FITSファイルのパス
            
        Returns:
        --------
        bool : 検証結果
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return False
        
        if not filepath.lower().endswith(('.fits', '.fts')):
            logger.error(f"Unsupported file format: {filepath}")
            return False
        
        try:
            with fits.open(filepath) as hdul:
                if len(hdul) == 0:
                    logger.error(f"Empty FITS file: {filepath}")
                    return False
                
                data = hdul[0].data
                if data is None:
                    logger.error(f"No data in FITS file: {filepath}")
                    return False
                
                if data.ndim != 2:
                    logger.error(f"Expected 2D data, got {data.ndim}D: {filepath}")
                    return False
                
            return True
            
        except Exception as e:
            logger.error(f"File validation failed for {filepath}: {e}")
            return False
    
    def create_processing_options(self, detector: str, **kwargs) -> Dict:
        """
        検出器に応じた処理オプションの作成
        
        Parameters:
        -----------
        detector : str
            検出器名
        **kwargs : dict
            カスタムオプション
            
        Returns:
        --------
        dict : 処理オプション
        """
        options = self.default_options.copy()
        
        # 検出器別のデフォルト設定
        if detector in ['COR1', 'COR2']:
            options.update({
                'rotate': True,
                'mask': True,
                'cosmic_ray_removal': True,
                'background_subtraction': False,
                'vignetting_correction': True
            })
        elif detector == 'EUVI':
            options.update({
                'rotate': True,
                'mask': False,
                'cosmic_ray_removal': True,
                'background_subtraction': False,
                'flat_field_correction': True
            })
        elif detector in ['HI1', 'HI2']:
            options.update({
                'rotate': False,
                'mask': False,
                'cosmic_ray_removal': True,
                'background_subtraction': True,
                'vignetting_correction': True
            })
        
        # カスタムオプションで上書き
        options.update(kwargs)
        
        return options
    
    def process_file(self, filepath: str, **kwargs) -> ProcessingResult:
        """
        単一ファイルの処理
        
        Parameters:
        -----------
        filepath : str
            FITSファイルのパス
        **kwargs : dict
            処理オプション
            
        Returns:
        --------
        ProcessingResult : 処理結果
        """
        start_time = datetime.now()
        
        try:
            # ファイルの検証
            if not self.validate_file(filepath):
                return ProcessingResult(
                    success=False,
                    message=f"File validation failed: {filepath}"
                )
            
            # 検出器の識別
            detector, header_info = self.identify_detector(filepath)
            if not detector:
                return ProcessingResult(
                    success=False,
                    message=f"Failed to identify detector: {filepath}"
                )
            
            # 処理オプションの設定
            options = self.create_processing_options(detector, **kwargs)
            
            if not self.silent:
                logger.info(f"Processing {detector} file: {os.path.basename(filepath)}")
            
            # 検出器別の処理
            if detector in ['COR1', 'COR2']:
                result = self._process_cor_file(filepath, options)
            elif detector == 'EUVI':
                result = self._process_euvi_file(filepath, options)
            elif detector in ['HI1', 'HI2']:
                result = self._process_hi_file(filepath, options)
            else:
                return ProcessingResult(
                    success=False,
                    message=f"Unsupported detector: {detector}"
                )
            
            # 処理時間の記録
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_info.update({
                'detector': detector,
                'processing_time': processing_time,
                'options_used': options,
                'input_file': filepath
            })
            
            # 処理履歴に追加
            self.processing_history.append({
                'timestamp': start_time.isoformat(),
                'file': filepath,
                'detector': detector,
                'success': result.success,
                'processing_time': processing_time
            })
            
            # 出力ファイルの保存
            if result.success and options.get('output_format'):
                self._save_output_files(result, filepath, options)
            
            # 処理ログの保存
            if options.get('save_processing_log', True):
                self._save_processing_log(result, filepath)
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error processing {filepath}: {e}")
            return ProcessingResult(
                success=False,
                message=f"Unexpected error: {str(e)}"
            )
    
    def _process_cor_file(self, filepath: str, options: Dict) -> ProcessingResult:
        """
        COR1/COR2ファイルの処理
        
        Parameters:
        -----------
        filepath : str
            FITSファイルのパス
        options : dict
            処理オプション
            
        Returns:
        --------
        ProcessingResult : 処理結果
        """
        try:
            # COR処理の実行
            processed_image, processed_header = self.cor_prep.cor_prep(
                filepath=filepath,
                rotate_on=options.get('rotate', True),
                smask_on=options.get('mask', True),
                calibrate_off=not options.get('calibrate', True),
                discri_pobj_on=options.get('cosmic_ray_removal', True),
                **options
            )
            
            if processed_image is None:
                return ProcessingResult(
                    success=False,
                    message="COR processing failed"
                )
            
            # 処理情報の作成
            processing_info = {
                'calibration_applied': options.get('calibrate', True),
                'rotation_applied': options.get('rotate', True),
                'mask_applied': options.get('mask', True),
                'cosmic_ray_removal': options.get('cosmic_ray_removal', True),
                'final_shape': processed_image.shape,
                'data_range': [float(np.nanmin(processed_image)), 
                              float(np.nanmax(processed_image))],
                'data_mean': float(np.nanmean(processed_image)),
                'data_std': float(np.nanstd(processed_image))
            }
            
            return ProcessingResult(
                success=True,
                message="COR processing completed successfully",
                data=processed_image,
                header=processed_header,
                processing_info=processing_info
            )
            
        except Exception as e:
            logger.error(f"COR processing failed: {e}")
            return ProcessingResult(
                success=False,
                message=f"COR processing failed: {str(e)}"
            )
    
    def _process_euvi_file(self, filepath: str, options: Dict) -> ProcessingResult:
        """
        EUVIファイルの処理
        
        Parameters:
        -----------
        filepath : str
            FITSファイルのパス
        options : dict
            処理オプション
            
        Returns:
        --------
        ProcessingResult : 処理結果
        """
        try:
            # 基本的な読み込み
            image, header, _ = self.secchi_prep.sccreadfits(filepath)
            
            if image is None:
                return ProcessingResult(
                    success=False,
                    message="EUVI file reading failed"
                )
            
            # EUVI処理の実行
            processed_image = self.secchi_prep.euvi_prep(header, image, **options)
            
            # 処理情報の作成
            processing_info = {
                'calibration_applied': options.get('calibrate', True),
                'flat_field_applied': options.get('flat_field_correction', True),
                'cosmic_ray_removal': options.get('cosmic_ray_removal', True),
                'final_shape': processed_image.shape,
                'data_range': [float(np.nanmin(processed_image)), 
                              float(np.nanmax(processed_image))],
                'data_mean': float(np.nanmean(processed_image)),
                'data_std': float(np.nanstd(processed_image))
            }
            
            return ProcessingResult(
                success=True,
                message="EUVI processing completed successfully",
                data=processed_image,
                header=header,
                processing_info=processing_info
            )
            
        except Exception as e:
            logger.error(f"EUVI processing failed: {e}")
            return ProcessingResult(
                success=False,
                message=f"EUVI processing failed: {str(e)}"
            )
    
    def _process_hi_file(self, filepath: str, options: Dict) -> ProcessingResult:
        """
        HI1/HI2ファイルの処理
        
        Parameters:
        -----------
        filepath : str
            FITSファイルのパス
        options : dict
            処理オプション
            
        Returns:
        --------
        ProcessingResult : 処理結果
        """
        try:
            # 基本的な読み込み
            image, header, _ = self.secchi_prep.sccreadfits(filepath)
            
            if image is None:
                return ProcessingResult(
                    success=False,
                    message="HI file reading failed"
                )
            
            # HI処理の実行
            processed_image, cosmic_report = self.secchi_prep.hi_prep(
                header, image, **options
            )
            
            # 処理情報の作成
            processing_info = {
                'calibration_applied': options.get('calibrate', True),
                'background_subtraction': options.get('background_subtraction', True),
                'cosmic_ray_removal': options.get('cosmic_ray_removal', True),
                'cosmic_ray_report': cosmic_report,
                'final_shape': processed_image.shape,
                'data_range': [float(np.nanmin(processed_image)), 
                              float(np.nanmax(processed_image))],
                'data_mean': float(np.nanmean(processed_image)),
                'data_std': float(np.nanstd(processed_image))
            }
            
            return ProcessingResult(
                success=True,
                message="HI processing completed successfully",
                data=processed_image,
                header=header,
                processing_info=processing_info
            )
            
        except Exception as e:
            logger.error(f"HI processing failed: {e}")
            return ProcessingResult(
                success=False,
                message=f"HI processing failed: {str(e)}"
            )
    
    def _save_output_files(self, result: ProcessingResult, 
                          input_filepath: str, options: Dict):
        """
        出力ファイルの保存
        
        Parameters:
        -----------
        result : ProcessingResult
            処理結果
        input_filepath : str
            入力ファイルのパス
        options : dict
            処理オプション
        """
        if not result.success or result.data is None:
            return
        
        # 出力ディレクトリの設定
        if self.output_dir:
            output_dir = Path(self.output_dir)
        else:
            output_dir = Path(input_filepath).parent
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ベースファイル名の作成
        base_name = Path(input_filepath).stem
        
        # 処理済みファイルの接尾辞
        base_name += "_processed"
        
        # 出力形式別の保存
        output_formats = options.get('output_format', ['fits'])
        
        for format_type in output_formats:
            try:
                if format_type.lower() in ['fits', 'fts']:
                    output_path = output_dir / f"{base_name}.fits"
                    self._save_fits_file(result, output_path)
                    
                elif format_type.lower() == 'png':
                    output_path = output_dir / f"{base_name}.png"
                    self._save_png_file(result, output_path)
                    
                elif format_type.lower() in ['jpg', 'jpeg']:
                    output_path = output_dir / f"{base_name}.jpg"
                    self._save_jpeg_file(result, output_path)
                    
                elif format_type.lower() == 'npy':
                    output_path = output_dir / f"{base_name}.npy"
                    np.save(output_path, result.data)
                    
                if not self.silent:
                    logger.info(f"Saved {format_type.upper()} file: {output_path}")
                    
            except Exception as e:
                logger.error(f"Failed to save {format_type} file: {e}")
    
    def _save_fits_file(self, result: ProcessingResult, output_path: Path):
        """FITS ファイルの保存"""
        # 基本的なヘッダーを作成
        fits_header = fits.Header()
        
        # 必須のヘッダー情報を最初に追加
        essential_keys = ['NAXIS1', 'NAXIS2', 'NAXIS']
        if result.header:
            for key in essential_keys:
                if key in result.header:
                    fits_header[key] = result.header[key]
        
        # 画像サイズの情報を確実に設定
        if result.data is not None:
            fits_header['NAXIS'] = 2
            fits_header['NAXIS1'] = result.data.shape[1]
            fits_header['NAXIS2'] = result.data.shape[0]
        
        # 重要なヘッダー情報を優先的に追加
        important_keys = ['DATE-OBS', 'DATE', 'TIME-OBS', 'EXPTIME', 'FILENAME', 
                         'DETECTOR', 'INSTRUME', 'OBSRVTRY', 'RSUN', 'DSUN_OBS',
                         'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2',
                         'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2', 'SOLAR_P', 'SOLAR_B0',
                         'SOLAR_L0', 'SUMMED', 'WAVELNTH', 'WAVEUNIT']
        
        if result.header:
            # 重要なキーを最初に追加
            for key in important_keys:
                if key in result.header:
                    try:
                        value = result.header[key]
                        if isinstance(value, (str, int, float, bool)):
                            if isinstance(value, str):
                                # 文字列の長さと文字チェック
                                try:
                                    value.encode('ascii')
                                    if len(value) <= 68:
                                        fits_header[key] = value
                                except UnicodeEncodeError:
                                    clean_value = ''.join(c for c in value if ord(c) < 128)
                                    if clean_value.strip() and len(clean_value) <= 68:
                                        fits_header[key] = clean_value
                            else:
                                fits_header[key] = value
                    except Exception as e:
                        logger.warning(f"Failed to add important header key {key}: {e}")
            
            # その他のヘッダー情報を追加
            for key, value in result.header.items():
                if key not in essential_keys and key not in important_keys and key != 'HISTORY':
                    try:
                        if isinstance(value, (str, int, float, bool)):
                            # ASCII文字でない場合は文字列として保存
                            if isinstance(value, str):
                                # 非ASCII文字を含む場合はエンコード
                                try:
                                    value.encode('ascii')
                                    if len(value) <= 68:  # FITS標準の制限
                                        fits_header[key] = value
                                except UnicodeEncodeError:
                                    # 非ASCII文字を削除または置換
                                    clean_value = ''.join(c for c in value if ord(c) < 128)
                                    if clean_value.strip() and len(clean_value) <= 68:
                                        fits_header[key] = clean_value
                            else:
                                fits_header[key] = value
                    except Exception as e:
                        logger.warning(f"Failed to add header key {key}: {e}")
        
        # 処理履歴を追加
        fits_header['HISTORY'] = f"Processed with {self.version}"
        fits_header['HISTORY'] = f"Processing time: {result.timestamp.isoformat()}"
        
        # データの統計情報を追加
        if result.data is not None:
            fits_header['DATAMIN'] = float(np.nanmin(result.data))
            fits_header['DATAMAX'] = float(np.nanmax(result.data))
            fits_header['DATAMEAN'] = float(np.nanmean(result.data))
            fits_header['BUNIT'] = 'DN/s'
        
        # HDUの作成と保存
        try:
            hdu = fits.PrimaryHDU(data=result.data, header=fits_header)
            hdu.writeto(output_path, overwrite=True)
            if not self.silent:
                logger.info(f"Saved FITS file: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save FITS file {output_path}: {e}")
            # 最小限のヘッダーで再試行
            try:
                minimal_header = fits.Header()
                minimal_header['HISTORY'] = f"Processed with {self.version}"
                minimal_hdu = fits.PrimaryHDU(data=result.data, header=minimal_header)
                minimal_hdu.writeto(output_path, overwrite=True)
                logger.info(f"Saved FITS file with minimal header: {output_path}")
            except Exception as e2:
                logger.error(f"Failed to save FITS file even with minimal header: {e2}")
    
    def _save_png_file(self, result: ProcessingResult, output_path: Path):
        """PNG ファイルの保存"""
        # 画像のスケーリング
        data = result.data.copy()
        
        # 95パーセンタイルでスケーリング
        vmin = np.nanpercentile(data, 5)
        vmax = np.nanpercentile(data, 95)
        
        # プロット作成
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'STEREO/SECCHI Processed Image\n{result.timestamp.strftime("%Y-%m-%d %H:%M:%S")}')
        
        # カラーバーの追加
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity')
        
        # 太陽半径の円を描画（可能な場合）
        if result.header and 'RSUN' in result.header:
            rsun_pixel = result.header['RSUN']
            cdelt = abs(result.header.get('CDELT1', 15.0))
            rsun_pixel = rsun_pixel / cdelt
            
            center_x = result.header.get('CRPIX1', data.shape[1] // 2)
            center_y = result.header.get('CRPIX2', data.shape[0] // 2)
            
            from matplotlib.patches import Circle
            circle = Circle((center_x, center_y), rsun_pixel, 
                          fill=False, edgecolor='yellow', linewidth=2, alpha=0.7)
            ax.add_patch(circle)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_jpeg_file(self, result: ProcessingResult, output_path: Path):
        """JPEG ファイルの保存"""
        # PNGと同様の処理
        self._save_png_file(result, output_path.with_suffix('.png'))
        
        # PNGからJPEGに変換（必要に応じて）
        # 実際のプロジェクトではPILなどを使用
        pass
    
    def _save_processing_log(self, result: ProcessingResult, input_filepath: str):
        """処理ログの保存"""
        try:
            log_data = {
                'input_file': input_filepath,
                'timestamp': result.timestamp.isoformat(),
                'success': result.success,
                'message': result.message,
                'processing_info': result.processing_info,
                'pipeline_version': self.version
            }
            
            # ログファイルのパス
            if self.output_dir:
                log_dir = Path(self.output_dir)
            else:
                log_dir = Path(input_filepath).parent
            
            log_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = Path(input_filepath).stem
            log_path = log_dir / f"{base_name}_processing_log.json"
            
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save processing log: {e}")
    
    def process_directory(self, directory: str, pattern: str = "*.fts", 
                         **kwargs) -> List[ProcessingResult]:
        """
        ディレクトリ内のファイルを一括処理
        
        Parameters:
        -----------
        directory : str
            処理対象のディレクトリ
        pattern : str, optional
            ファイルパターン
        **kwargs : dict
            処理オプション
            
        Returns:
        --------
        list : 処理結果のリスト
        """
        directory = Path(directory)
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return []
        
        # ファイルの検索
        files = list(directory.glob(pattern))
        if not files:
            logger.warning(f"No files found matching pattern: {pattern}")
            return []
        
        if not self.silent:
            logger.info(f"Found {len(files)} files to process")
        
        results = []
        
        for i, filepath in enumerate(files):
            if not self.silent:
                logger.info(f"Processing file {i+1}/{len(files)}: {filepath.name}")
            
            try:
                result = self.process_file(str(filepath), **kwargs)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {e}")
                results.append(ProcessingResult(
                    success=False,
                    message=f"Processing failed: {str(e)}"
                ))
        
        # 結果のサマリー
        successful = sum(1 for r in results if r.success)
        if not self.silent:
            logger.info(f"Processing completed: {successful}/{len(results)} files successful")
        
        return results
    
    def get_processing_summary(self) -> Dict:
        """
        処理サマリーの取得
        
        Returns:
        --------
        dict : 処理サマリー
        """
        if not self.processing_history:
            return {'total_files': 0, 'successful': 0, 'failed': 0}
        
        total_files = len(self.processing_history)
        successful = sum(1 for h in self.processing_history if h['success'])
        failed = total_files - successful
        
        # 検出器別の統計
        detector_stats = {}
        for history in self.processing_history:
            detector = history.get('detector', 'unknown')
            if detector not in detector_stats:
                detector_stats[detector] = {'total': 0, 'successful': 0}
            detector_stats[detector]['total'] += 1
            if history['success']:
                detector_stats[detector]['successful'] += 1
        
        # 処理時間の統計
        processing_times = [h['processing_time'] for h in self.processing_history 
                          if 'processing_time' in h]
        
        return {
            'total_files': total_files,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_files if total_files > 0 else 0,
            'detector_stats': detector_stats,
            'processing_times': {
                'mean': np.mean(processing_times) if processing_times else 0,
                'median': np.median(processing_times) if processing_times else 0,
                'total': sum(processing_times) if processing_times else 0
            }
        }

def main():
    """
    STEREO-A/SECCHI/COR1データの自動校正処理
    
    このスクリプトを実行すると、Rawdataフォルダ内の全てのFITSファイルを
    自動的に校正処理し、校正済みFITSファイルとPNG画像を出力します。
    """
    import glob
    
    # 入力・出力ディレクトリの設定
    input_dir = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata"
    output_dir = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/calibration"
    
    print("=== STEREO-A/SECCHI/COR1 自動校正処理 ===")
    print(f"入力ディレクトリ: {input_dir}")
    print(f"出力ディレクトリ: {output_dir}")
    
    # パイプラインの初期化
    pipeline = STEREOPipeline(output_dir=output_dir)
    
    # Rawdataフォルダ内のFITSファイルを検索
    fits_files = glob.glob(os.path.join(input_dir, "*.fts"))
    fits_files.extend(glob.glob(os.path.join(input_dir, "*.fits")))
    
    if not fits_files:
        print("⚠️  処理対象のFITSファイルが見つかりません")
        return
    
    print(f"📁 {len(fits_files)} 個のFITSファイルを発見しました")
    
    # 処理オプション（IDL版cor_prep.proと同等の処理）
    processing_options = {
        'calibrate': True,              # 校正処理を実行
        'rotate': True,                 # 太陽北極を上に回転
        'mask': True,                   # マスク処理を適用
        'cosmic_ray_removal': True,     # 宇宙線除去
        'vignetting_correction': True,  # ビネッティング補正
        'background_subtraction': False, # 背景減算（COR1では通常OFF）
        'output_format': ['fits', 'png'], # FITS + PNG出力
        'save_processing_log': True     # 処理ログを保存
    }
    
    print("\n📊 処理オプション:")
    for key, value in processing_options.items():
        print(f"  {key}: {value}")
    
    print("\n🔄 処理開始...")
    
    # 各ファイルを処理
    successful_files = []
    failed_files = []
    
    for i, filepath in enumerate(fits_files):
        filename = os.path.basename(filepath)
        print(f"\n[{i+1}/{len(fits_files)}] 処理中: {filename}")
        
        try:
            # 校正処理の実行
            result = pipeline.process_file(filepath, **processing_options)
            
            if result.success:
                successful_files.append(filename)
                print(f"✅ 成功: {filename}")
                
                # 処理結果の表示
                info = result.processing_info
                print(f"   データ範囲: {info['data_range'][0]:.2f} - {info['data_range'][1]:.2f}")
                print(f"   平均値: {info['data_mean']:.2f}")
                print(f"   処理時間: {info['processing_time']:.2f}秒")
                
            else:
                failed_files.append(filename)
                print(f"❌ 失敗: {filename} - {result.message}")
                
        except Exception as e:
            failed_files.append(filename)
            print(f"❌ エラー: {filename} - {str(e)}")
    
    # 処理結果のサマリー
    print("\n" + "="*50)
    print("🎯 処理結果サマリー")
    print("="*50)
    print(f"総ファイル数: {len(fits_files)}")
    print(f"成功: {len(successful_files)}")
    print(f"失敗: {len(failed_files)}")
    print(f"成功率: {len(successful_files)/len(fits_files)*100:.1f}%")
    
    if successful_files:
        print(f"\n📁 校正済みファイルの出力先: {output_dir}")
        print("📋 成功したファイル:")
        for filename in successful_files:
            base_name = os.path.splitext(filename)[0]
            print(f"  • {base_name}_processed.fits")
            print(f"  • {base_name}_processed.png")
    
    if failed_files:
        print("\n⚠️  処理に失敗したファイル:")
        for filename in failed_files:
            print(f"  • {filename}")
    
    # 処理全体の統計
    total_summary = pipeline.get_processing_summary()
    if total_summary['total_files'] > 0:
        print(f"\n📊 総処理時間: {total_summary['processing_times']['total']:.2f}秒")
        print(f"📊 平均処理時間: {total_summary['processing_times']['mean']:.2f}秒")
    
    print("\n🎉 STEREO-A/SECCHI/COR1 自動校正処理が完了しました！")
    
    # 使用方法のヒント
    print("\n💡 使用方法:")
    print("  - 校正済みFITSファイルは科学解析に使用できます")
    print("  - PNG画像は可視化・確認用です")
    print("  - 処理ログJSONファイルには詳細な処理情報が含まれています")
    print("  - エラーが発生した場合は、入力ファイルの形式を確認してください")

if __name__ == "__main__":
    main()