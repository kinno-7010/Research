#!/usr/bin/env python3
"""
STEREO-A/SECCHI/COR1専用カラーテーブル機能

このモジュールは、IDL版SSWIDL/secchi_colors.proの機能をPythonに移植したものです。
COR1専用のカラーマップと動的スケーリング機能を提供します。

主な機能:
- COR1専用カラーマップの定義・適用
- 画像データに応じた動的カラースケーリング
- 各種SECCHI検出器のカラーテーブル対応
- matplotlibとの統合

参照元: SSWIDL secchi_colors.pro, load_secchi_color.pro
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from astropy.io import fits
import os
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SECCHIColors:
    """
    STEREO SECCHI検出器用カラーテーブルクラス
    
    IDL版secchi_colors.proの機能をPythonで実装
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
        self.version = "COR1_Colors v1.0 (Python port of IDL secchi_colors.pro)"
        
        # カラーテーブルの定義
        self._define_color_tables()
        
        if not self.silent:
            logger.info(self.version)
    
    def _define_color_tables(self):
        """
        SECCHI各検出器用のカラーテーブルを定義
        
        IDL版のカラーファイルを元に、Pythonで近似的に再現
        """
        # COR1専用カラーマップ（白コロナ用）
        # 低輝度：黒 → 高輝度：白、コントラスト強化
        self.cor1_colors = {
            'red':   [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.3, 0.2, 0.2), 
                     (0.5, 0.5, 0.5), (0.7, 0.8, 0.8), (1.0, 1.0, 1.0)],
            'green': [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.3, 0.2, 0.2), 
                     (0.5, 0.5, 0.5), (0.7, 0.8, 0.8), (1.0, 1.0, 1.0)],
            'blue':  [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.3, 0.2, 0.2), 
                     (0.5, 0.5, 0.5), (0.7, 0.8, 0.8), (1.0, 1.0, 1.0)]
        }
        
        # COR2専用カラーマップ（外部コロナ用）
        self.cor2_colors = {
            'red':   [(0.0, 0.0, 0.0), (0.2, 0.1, 0.1), (0.4, 0.3, 0.3), 
                     (0.6, 0.6, 0.6), (0.8, 0.9, 0.9), (1.0, 1.0, 1.0)],
            'green': [(0.0, 0.0, 0.0), (0.2, 0.1, 0.1), (0.4, 0.3, 0.3), 
                     (0.6, 0.6, 0.6), (0.8, 0.9, 0.9), (1.0, 1.0, 1.0)],
            'blue':  [(0.0, 0.0, 0.0), (0.2, 0.1, 0.1), (0.4, 0.3, 0.3), 
                     (0.6, 0.6, 0.6), (0.8, 0.9, 0.9), (1.0, 1.0, 1.0)]
        }
        
        # EUVI専用カラーマップ（波長別）
        self.euvi_colors = {
            171: {  # Fe IX/X (青系)
                'red':   [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.6, 0.3, 0.3), (1.0, 1.0, 1.0)],
                'green': [(0.0, 0.0, 0.0), (0.3, 0.2, 0.2), (0.6, 0.6, 0.6), (1.0, 1.0, 1.0)],
                'blue':  [(0.0, 0.0, 0.0), (0.3, 0.5, 0.5), (0.6, 0.8, 0.8), (1.0, 1.0, 1.0)]
            },
            195: {  # Fe XII (緑系)
                'red':   [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.6, 0.4, 0.4), (1.0, 1.0, 1.0)],
                'green': [(0.0, 0.0, 0.0), (0.3, 0.3, 0.3), (0.6, 0.8, 0.8), (1.0, 1.0, 1.0)],
                'blue':  [(0.0, 0.0, 0.0), (0.3, 0.1, 0.1), (0.6, 0.3, 0.3), (1.0, 1.0, 1.0)]
            },
            284: {  # Fe XV (黄系)
                'red':   [(0.0, 0.0, 0.0), (0.3, 0.4, 0.4), (0.6, 0.8, 0.8), (1.0, 1.0, 1.0)],
                'green': [(0.0, 0.0, 0.0), (0.3, 0.3, 0.3), (0.6, 0.7, 0.7), (1.0, 1.0, 1.0)],
                'blue':  [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.6, 0.2, 0.2), (1.0, 1.0, 1.0)]
            },
            304: {  # He II (赤系)
                'red':   [(0.0, 0.0, 0.0), (0.3, 0.5, 0.5), (0.6, 0.9, 0.9), (1.0, 1.0, 1.0)],
                'green': [(0.0, 0.0, 0.0), (0.3, 0.1, 0.1), (0.6, 0.4, 0.4), (1.0, 1.0, 1.0)],
                'blue':  [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.6, 0.1, 0.1), (1.0, 1.0, 1.0)]
            }
        }
        
        # HI1/HI2カラーマップ（太陽圏撮像用）
        self.hi_colors = {
            'red':   [(0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.5, 0.4, 0.4), 
                     (0.75, 0.8, 0.8), (1.0, 1.0, 1.0)],
            'green': [(0.0, 0.0, 0.0), (0.25, 0.1, 0.1), (0.5, 0.5, 0.5), 
                     (0.75, 0.9, 0.9), (1.0, 1.0, 1.0)],
            'blue':  [(0.0, 0.0, 0.0), (0.25, 0.2, 0.2), (0.5, 0.6, 0.6), 
                     (0.75, 0.9, 0.9), (1.0, 1.0, 1.0)]
        }
    
    def get_colormap(self, detector, wavelength=None):
        """
        検出器・波長に応じたカラーマップを取得
        
        Parameters:
        -----------
        detector : str
            検出器名（'COR1', 'COR2', 'EUVI', 'HI1', 'HI2'）
        wavelength : int, optional
            EUVI用の波長（171, 195, 284, 304）
            
        Returns:
        --------
        matplotlib.colors.LinearSegmentedColormap : カラーマップ
        """
        detector = detector.upper()
        
        if detector == 'COR1':
            colors = self.cor1_colors
            name = 'COR1_colormap'
        elif detector == 'COR2':
            colors = self.cor2_colors
            name = 'COR2_colormap'
        elif detector == 'EUVI':
            if wavelength in self.euvi_colors:
                colors = self.euvi_colors[wavelength]
                name = f'EUVI_{wavelength}_colormap'
            else:
                # デフォルトは195Å
                colors = self.euvi_colors[195]
                name = 'EUVI_195_colormap'
        elif detector in ['HI1', 'HI2']:
            colors = self.hi_colors
            name = f'{detector}_colormap'
        else:
            # 不明な検出器の場合はグレースケール
            if not self.silent:
                logger.warning(f"Unknown detector: {detector}, using grayscale")
            return plt.cm.gray
        
        return LinearSegmentedColormap(name, colors, N=256)
    
    def load_secchi_colors(self, header=None, detector=None, wavelength=None):
        """
        SECCHI検出器用カラーテーブルの読み込み
        
        IDL版load_secchi_color.proの機能を再現
        
        Parameters:
        -----------
        header : dict or astropy.io.fits.Header, optional
            FITSヘッダー情報
        detector : str, optional
            検出器名（ヘッダーから自動取得可能）
        wavelength : int, optional
            波長情報（ヘッダーから自動取得可能）
            
        Returns:
        --------
        matplotlib.colors.LinearSegmentedColormap : カラーマップ
        """
        # ヘッダーから検出器情報を取得
        if header is not None:
            if hasattr(header, 'get'):
                # FITS Header or dict-like object
                detector = header.get('DETECTOR', detector)
                wavelength = header.get('WAVELNTH', wavelength)
            elif hasattr(header, 'detector'):
                # SECCHI structure
                detector = header.detector
                wavelength = getattr(header, 'wavelnth', wavelength)
        
        if detector is None:
            detector = 'COR1'  # デフォルト
        
        return self.get_colormap(detector, wavelength)
    
    def calculate_scaling(self, data, method='zscale', percentile_range=(5, 95)):
        """
        画像データの動的スケーリング計算
        
        IDL版のバイトスケーリング機能を再現
        
        Parameters:
        -----------
        data : numpy.ndarray
            画像データ
        method : str, optional
            スケーリング方法（'zscale', 'percentile', 'minmax'）
        percentile_range : tuple, optional
            パーセンタイル範囲
            
        Returns:
        --------
        tuple : (vmin, vmax) スケール範囲
        """
        # NaN値を除外
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return 0, 1
        
        if method == 'zscale':
            # ZScaleアルゴリズム（簡略版）
            sorted_data = np.sort(valid_data.flatten())
            n = len(sorted_data)
            
            # 中央値周辺のデータを使用
            center_idx = n // 2
            sample_size = min(n // 10, 1000)
            start_idx = max(0, center_idx - sample_size // 2)
            end_idx = min(n, start_idx + sample_size)
            
            sample = sorted_data[start_idx:end_idx]
            
            # 線形フィットによる傾きを推定
            x = np.arange(len(sample))
            coeffs = np.polyfit(x, sample, 1)
            slope = coeffs[0]
            
            # ZScale範囲の計算
            median = np.median(valid_data)
            z1 = median - 2.5 * slope * len(sample) / 2
            z2 = median + 2.5 * slope * len(sample) / 2
            
            vmin = max(np.min(valid_data), z1)
            vmax = min(np.max(valid_data), z2)
            
        elif method == 'percentile':
            # パーセンタイル範囲
            vmin = np.percentile(valid_data, percentile_range[0])
            vmax = np.percentile(valid_data, percentile_range[1])
            
        else:  # minmax
            vmin = np.min(valid_data)
            vmax = np.max(valid_data)
        
        # 同じ値の場合は少し拡張
        if vmin == vmax:
            if vmin == 0:
                vmax = 1
            else:
                vmin = vmin * 0.9
                vmax = vmax * 1.1
        
        return vmin, vmax
    
    def get_annotation_color(self, data, detector='COR1'):
        """
        アノテーション用の色を取得
        
        IDL版の charcolor = max(image) 機能を再現
        
        Parameters:
        -----------
        data : numpy.ndarray
            画像データ
        detector : str, optional
            検出器名
            
        Returns:
        --------
        str : matplotlib色指定
        """
        max_val = np.nanmax(data)
        min_val = np.nanmin(data)
        
        # データの範囲に応じて色を決定
        if detector.upper() == 'COR1':
            # 高輝度データには黒、低輝度データには白
            mid_val = (max_val + min_val) / 2
            if max_val > mid_val * 1.5:
                return 'black'
            else:
                return 'white'
        else:
            # その他の検出器は白をデフォルト
            return 'white'

def main():
    """
    テスト用のメイン関数
    """
    print("=== SECCHI Colors Test ===")
    
    # SECCHIColorsインスタンスを作成
    colors = SECCHIColors()
    
    # COR1カラーマップのテスト
    cor1_cmap = colors.get_colormap('COR1')
    print(f"COR1 colormap created: {cor1_cmap.name}")
    
    # テスト用データの生成
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    test_data = np.exp(-((X-5)**2 + (Y-5)**2) / 4)
    
    # スケーリングのテスト
    vmin, vmax = colors.calculate_scaling(test_data, method='zscale')
    print(f"ZScale range: {vmin:.3f} to {vmax:.3f}")
    
    # アノテーション色のテスト
    ann_color = colors.get_annotation_color(test_data, 'COR1')
    print(f"Annotation color: {ann_color}")
    
    # 可視化テスト
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # COR1カラーマップ
    im1 = axes[0].imshow(test_data, cmap=cor1_cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title('COR1 Colormap')
    plt.colorbar(im1, ax=axes[0])
    
    # COR2カラーマップ
    cor2_cmap = colors.get_colormap('COR2')
    im2 = axes[1].imshow(test_data, cmap=cor2_cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title('COR2 Colormap')
    plt.colorbar(im2, ax=axes[1])
    
    # EUVI 195Åカラーマップ
    euvi_cmap = colors.get_colormap('EUVI', 195)
    im3 = axes[2].imshow(test_data, cmap=euvi_cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title('EUVI 195Å Colormap')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/py_folder/secchi_colormap_test.png')
    plt.close()
    
    print("Colormap test completed successfully!")
    print("Test image saved as: secchi_colormap_test.png")

if __name__ == "__main__":
    main()