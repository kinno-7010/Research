# STEREO-A/SECCHI/COR1 Python Data Processing Pipeline

このディレクトリには、STEREO-A/SECCHI/COR1コロナグラフデータの処理用Python実装が含まれています。IDL版のSSWIDLライブラリ（cor_prep.pro、secchi_prep.pro）をPythonに移植したものです。

## 概要

STEREO（Solar Terrestrial Relations Observatory）は、太陽コロナや太陽風を観測するNASAの双子の宇宙探査機です。このプロジェクトは、STEREO-A衛星に搭載されたSECCHI（Sun Earth Connection Coronal and Heliospheric Investigation）装置群の中でも、特にCOR1（Coronagraph 1）のデータ処理に焦点を当てています。

## ファイル構成

### 主要スクリプト

1. **cor_prep.py** - COR1/COR2データの核心的な処理
   - IDL版 `cor_prep.pro` のPython実装
   - 校正、ビネッティング補正、回転、マスク処理
   - 点光源・宇宙線除去フィルタ
   - CORPrepクラスによるオブジェクト指向設計

2. **secchi_prep.py** - SECCHI全般のデータ処理
   - IDL版 `secchi_prep.pro` のPython実装
   - 複数ファイルの一括処理
   - 各検出器（COR1, COR2, EUVI, HI1, HI2）への対応
   - SECCHIPrepクラスによる統合処理

3. **secchi_utils.py** - 共通ユーティリティ関数
   - 校正係数の計算（get_calfac）
   - ビネッティング関数（get_vignetting）
   - 座標変換（convert_coords）
   - 太陽天体暦の計算（get_solar_ephemeris）
   - フラットフィールド・ダークカレント補正
   - SECCHIUtilsクラスによる機能提供

4. **stereo_pipeline.py** - 統合処理パイプライン
   - STEREOPipelineクラスによる高レベルAPI
   - 自動的な検出器識別
   - ProcessingResultクラスによる結果管理
   - 複数の出力形式対応（FITS, PNG, JPEG, NPY）
   - エラーハンドリングと処理履歴管理

### データ処理・解析スクリプト

5. **cor1_plot.py** - COR1データの可視化
   - FITSファイルの読み込み（read_cor1_data関数）
   - 太陽半径の計算（calculate_solar_radius_pixel関数）
   - 太陽半径円の描画（1Rs, 2Rs, 3Rs）
   - 画像統計情報の表示

6. **cor1_data_download.py** - データダウンロード
   - SunPy/Fidoを使用したSTEREO-Aデータの自動取得
   - 指定時間範囲のCOR1データ検索・ダウンロード

### ユーティリティ・デバッグスクリプト

7. **check_header.py** - FITSヘッダーの検証
   - ヘッダー情報の確認
   - 太陽半径関連パラメータの抽出
   - 重要なキーワードの一覧表示

8. **check_datetime.py** - 日時情報の確認
   - 元ファイルと処理済みファイルの日時比較
   - 処理履歴（HISTORY）の確認
   - データ統計情報の表示

9. **debug_header.py** - ヘッダー情報の転送デバッグ
   - CORPrep処理前後のヘッダー比較
   - データ転送の整合性確認
   - 欠損キーの特定

## 使用方法

### 1. 統合パイプラインの使用（推奨）

```python
from stereo_pipeline import STEREOPipeline

# パイプラインの初期化
pipeline = STEREOPipeline(output_dir='./output')

# 単一ファイルの処理
result = pipeline.process_file('20220613_032136_n4c1A.fts', 
                               calibrate=True, 
                               rotate=True,
                               output_format=['fits', 'png'])

# 処理結果の確認
if result.success:
    print(f"処理成功: {result.message}")
    print(f"データ形状: {result.data.shape}")
    print(f"データ範囲: {result.processing_info['data_range']}")
    print(f"処理時間: {result.processing_info['processing_time']:.2f}秒")
```

### 2. 複数ファイルの一括処理

```python
# ディレクトリ内の全FITSファイルを自動処理
results = pipeline.process_directory('./Rawdata/', 
                                   pattern='*.fts',
                                   calibrate=True,
                                   rotate=True,
                                   mask=True,
                                   cosmic_ray_removal=True,
                                   output_format=['fits', 'png'])

# 処理結果のサマリー
summary = pipeline.get_processing_summary()
print(f"成功率: {summary['success_rate']:.1%}")
print(f"平均処理時間: {summary['processing_times']['mean']:.2f}秒")
```

### 3. 自動校正処理（メイン機能）

```bash
# stereo_pipeline.pyを直接実行すると自動校正処理が開始されます
python3 stereo_pipeline.py
```

この機能により、Rawdataフォルダ内の全FITSファイルが自動的に：
- 校正処理（calibrate=True）
- 太陽北極回転（rotate=True）
- マスク処理（mask=True）
- 宇宙線除去（cosmic_ray_removal=True）
- FITS・PNG形式で出力

### 4. 個別モジュールの使用

```python
from cor_prep import CORPrep

# COR処理の実行
prep = CORPrep()
processed_image, header = prep.cor_prep(
    filepath='20220613_032136_n4c1A.fts',
    rotate_on=True,
    smask_on=True,
    calibrate_off=False,
    discri_pobj_on=True
)
```

### 5. データの可視化

```python
# COR1データの可視化
python3 cor1_plot.py

# または個別に
from cor1_plot import plot_cor1_image, read_cor1_data

data, header = read_cor1_data('20220613_032136_n4c1A.fts')
plot_cor1_image(data, header, filepath, save_path='output.png')
```

## 主要な処理機能

### 1. 校正処理
- **CCDバイアス減算**: 検出器のバイアスレベルを除去
- **フラットフィールド補正**: 画素感度の不均一性を補正
- **ビネッティング補正**: 光学系による周辺減光を補正
- **露出時間正規化**: 異なる露出時間での観測を統一

### 2. 画像処理
- **太陽北極回転**: 太陽の北極を画像の上に向ける
- **マスク処理**: 有効な観測領域のみを抽出
- **宇宙線除去**: 高エネルギー粒子による異常値を除去
- **点光源除去**: 星などの点光源を除去

### 3. 座標系
- **ピクセル座標**: 検出器上の位置 (pixel)
- **角度座標**: 太陽中心からの角度 (arcsec)
- **太陽半径単位**: 太陽半径を基準とした座標 (Rs)

### 4. 出力形式
- **FITS**: 天文学標準形式（ヘッダー情報含む）
- **PNG**: 可視化用画像
- **JPEG**: 圧縮画像
- **NumPy**: 数値配列形式

## 技術仕様

### 対応検出器
- **COR1**: 内部コロナグラフ（1.5-4.0 Rs）
- **COR2**: 外部コロナグラフ（3.0-15.0 Rs）
- **EUVI**: 極端紫外線撮像装置
- **HI1/HI2**: 太陽圏撮像装置

### 依存関係
- `numpy`: 数値計算
- `scipy`: 科学計算
- `matplotlib`: 可視化
- `astropy`: 天文学データ処理
- `pathlib`: ファイル操作

### データ形式
- **入力**: FITS/FTS ファイル
- **処理**: 64bit浮動小数点配列
- **出力**: 複数形式対応

## 処理パラメータ

### 校正オプション
- `calibrate`: 校正処理の有効/無効
- `rotate`: 太陽北極回転の有効/無効
- `mask`: マスク処理の有効/無効
- `cosmic_ray_removal`: 宇宙線除去の有効/無効
- `background_subtraction`: 背景減算の有効/無効

### 画像処理オプション
- `vignetting_correction`: ビネッティング補正
- `flat_field_correction`: フラットフィールド補正
- `dark_current_subtraction`: ダークカレント減算
- `trim`: 画像トリミング

### 出力オプション
- `output_format`: 出力形式 ['fits', 'png', 'jpeg', 'npy']
- `save_processing_log`: 処理ログの保存
- `silent`: メッセージの抑制

## 処理例

### 現在のデータ処理結果

```
=== COR1 Processing Results ===
File: 20220613_032136_n4c1A.fts
Observation Time: 2022-06-13 03:21:36
Detector: COR1
Data shape: (512, 512)
Solar radius: 998.69 arcsec (66.54 pixels)
Solar center: (254.73, 254.47) pixels
Processing time: 0.574 seconds
Success rate: 100.0%
```

### 実際の処理例（20220613データ）
```
=== COR1 Processing Results ===
File: 20220613_032136_n4c1A.fts
Observation Time: 2022-06-13 03:21:36
Detector: COR1
Data shape: (512, 512)
Solar radius: 998.69 arcsec (66.54 pixels)
Solar center: (254.73, 254.47) pixels
Processing time: 0.574 seconds
Success rate: 100.0%
```

### 処理されたデータの特徴
- **データ範囲**: 0.0 to 5966.98 DN/s
- **平均値**: 1216.48 DN/s  
- **標準偏差**: 1570.56 DN/s
- **処理時間**: 0.57秒
- **出力単位**: DN/s (Data Numbers per second)

## 出力ファイル

### 自動生成されるファイル
1. **処理済みFITS**: `*_processed.fits`
   - 校正済み科学データ（FITS形式）
   - 元のヘッダー情報 + 処理履歴
   - 統計情報（DATAMIN, DATAMAX, DATAMEAN等）

2. **可視化画像**: `*_processed.png`
   - 95パーセンタイルスケーリング
   - 太陽半径円の描画（黄色：1Rs, オレンジ：2Rs, 赤：3Rs）
   - カラーバーと統計情報

3. **処理ログ**: `*_processing_log.json`
   - 詳細な処理情報とメタデータ

### 処理ログの内容
```json
{
  "input_file": "20220613_032136_n4c1A.fts",
  "timestamp": "2025-01-22T15:30:00",
  "success": true,
  "message": "COR processing completed successfully",
  "processing_info": {
    "detector": "COR1",
    "calibration_applied": true,
    "rotation_applied": true,
    "mask_applied": true,
    "cosmic_ray_removal": true,
    "vignetting_correction": true,
    "processing_time": 0.574,
    "data_range": [0.0, 5966.98],
    "data_mean": 1216.48,
    "final_shape": [512, 512]
  },
  "pipeline_version": "STEREO Pipeline v1.0"
}
```

## ファイル依存関係

```
stereo_pipeline.py (メインAPI)
├── cor_prep.py (COR1/COR2処理)
├── secchi_prep.py (SECCHI統合処理)
│   └── cor_prep.py
└── secchi_utils.py (共通ユーティリティ)

cor1_plot.py (独立した可視化ツール)
cor1_data_download.py (データ取得ツール)
check_*.py, debug_*.py (デバッグ・確認ツール)
```

## 開発状況

### 実装済み機能 ✅
- IDL版cor_prep.proの主要機能をPython移植
- レベル0.5→1.0への校正処理パイプライン
- 太陽北極回転、マスク処理、宇宙線除去
- 複数出力形式（FITS, PNG, JSON）
- エラーハンドリングと処理履歴管理
- 自動検出器識別
- 統合処理結果サマリー

### 今後の改善点 🔧
1. **校正精度の向上**: 実際の校正データファイル（CALファイル）の組み込み
2. **処理速度の最適化**: NumPy並列化、メモリ効率化
3. **IDL互換性**: より詳細な比較検証とパラメータ調整
4. **エラー処理**: より堅牢なエラーハンドリング
5. **文書化**: 関数・クラスのAPI文書充実

### 既知の制限事項 ⚠️
- ビネッティング関数は簡略化版（実際の関数ファイル未実装）
- 校正係数は固定値（時間・温度依存性は簡略化）
- COR2のワープ処理は基本実装のみ
- 偏光処理（polariz_on）は未実装

## トラブルシューティング

### よくある問題
1. **ModuleNotFoundError**: astropyまたは必要パッケージ未インストール
   ```bash
   pip3 install astropy matplotlib scipy numpy
   ```

2. **メモリエラー**: 大量ファイル処理時
   ```python
   # バッチサイズを小さく設定
   pipeline.process_directory('./data/', batch_size=10)
   ```

3. **FITSヘッダーエラー**: 非ASCII文字の処理
   - ヘッダークリーニングを自動実行

## 参考文献

### STEREO/SECCHI関連文献
- Howard, R. A., et al. (2008). "Sun Earth Connection Coronal and Heliospheric Investigation (SECCHI)." *Space Science Reviews*, 136(1-4), 67-115.
- Kaiser, M. L., et al. (2008). "The STEREO mission: An introduction." *Space Science Reviews*, 136(1-4), 5-16.
- Thernisien, A., et al. (2009). "The COR1 and COR2 coronagraphs on the STEREO spacecraft." *Solar Physics*, 256(1-2), 111-130.

### データ解析リソース
- STEREO SECCHI Data Analysis Guide: https://stereo-ssc.nascom.nasa.gov/
- SolarSoft IDL Library: https://hesperia.gsfc.nasa.gov/ssw/
- SunPy Documentation: https://docs.sunpy.org/

### 関連プロジェクト
- SSW/STEREO/SECCHI IDL Library (元実装)
- SunPy Project (Python solar physics toolkit)
- Astropy Project (Python astronomy library)

## 連絡先

このプロジェクトは太陽物理学研究の一環として開発されています。  
東北大学理学研究科・地球物理学専攻  

質問・バグ報告・機能要望は開発者までお問い合わせください。

---
*最終更新: 2025年1月22日*  
*Python移植版 v1.0 - IDL SSW cor_prep.proベース*