# UCOMP Data Analysis Tools

UCOMPコロナグラフデータの解析・可視化ツール

## 概要

このツールセットは、UCOMP (University of Hawaii Coronal Observatory Multi-band Polarimetry) Level 2データの読み込み、処理、可視化を行います。

### 対応データ形式
- **ファイル形式**: FITS (.fts)
- **ファイル名**: `YYYYMMDD.HHMMSS.ucomp.<wavelength>.l2.fts`
- **対応波長**: 637, 706, 789, 1074, 1079 nm
- **デフォルト波長**: 1074 nm

### Level 2 FITS構造

#### Stokes I science products (全波長)
- **Ext 1**: Center Line Intensity (中心線強度)
- **Ext 2**: Enhanced (unsharp mask) intensity (強調強度)
- **Ext 3**: Gaussian Peak Intensity (ガウシアンピーク強度)
- **Ext 4**: Line-of-Sight Doppler Velocity (視線ドップラー速度)
- **Ext 5**: FWHM Line Width (線幅)
- **Ext 6**: Noise Mask (ノイズマスク)

#### FeXIII ONLY Stokes Q & U products (1074nm, 1079nmのみ)
- **Ext 7**: Weighted avg Stokes I (重み付き平均Stokes I)
- **Ext 8**: Weighted avg Stokes Q (重み付き平均Stokes Q)
- **Ext 9**: Weighted avg Stokes U (重み付き平均Stokes U)
- **Ext 10**: Weighted avg Linear Polarization (L) (重み付き平均線偏光)
- **Ext 11**: Azimuth of plane-of-sky magnetic field (天球面磁場方位角)
- **Ext 12**: Radial Azimuth of plane-of-sky magnetic field (径方向磁場方位角)


### ストークスIから導出される物理量（全輝線で利用可能）

[cite_start]これらのデータは、主に輝線の強度プロファイルをガウス関数でフィットすることによって得られる物理量です [cite: 223, 261]。

* **Ext 1: Center wavelength intensity (輝線中心の強度)**
    * [cite_start]これは、観測された複数の波長点のうち、**輝線の中心波長で観測されたL1データの強度**そのものです [cite: 263]。画像の明るさがコロナのプラズマの強度に直接対応します。

* **Ext 2: Enhanced intensity (強調処理された強度)**
    * [cite_start]Ext 1の輝線中心強度画像に対して**アンシャープマスク処理を施した画像**です [cite: 263]。アンシャープマスクは、画像の鮮明さを強調する画像処理技術で、コロナの微細なループ構造や活動領域の細部を視覚的に捉えやすくするために用いられます。

* **Ext 3: Gaussian Peak Intensity (ガウスフィットによるピーク強度)**
    * [cite_start]観測された輝線プロファイルに対してガウス関数をフィッティングし、その**解析的なガウス関数のピーク（頂点）の高さ**を強度としたものです [cite: 263]。単一波長の強度（Ext 1）よりもノイズの影響を受けにくく、よりロバストな強度値と期待できます。

* **Ext 4: Line-of-Sight (LOS) Doppler Velocity (視線方向ドップラー速度)**
    * [cite_start]ガウスフィットによって得られた**輝線中心波長のズレ**から導出されます [cite: 263]。これにより、コロナプラズマが我々に対して近づいている（青方偏移）か、遠ざかっている（赤方偏移）かの視線方向の速度がわかります。
    * [cite_start]**注意**: 2023年12月時点のガイドでは、このデータはまだ最適化されていないと記載されています [cite: 9]。

* **Ext 5: FWHM Line Width (輝線の半値全幅)**
    * [cite_start]ガウスフィットによって得られた**輝線のスペクトル幅（半値全幅）**です [cite: 263]。この値は、プラズマの温度や視線に沿った非熱的な運動（乱流など）の大きさを示唆します。
    * [cite_start]**注意**: このバージョンから、以前の「1/e幅」ではなく「半値全幅（FWHM）」で提供されています。両者の間には `FWHM = 1/e幅 * 1.66511` の関係があります [cite: 264, 265][cite_start]。また、このデータもまだ最適化されていないと記載されています [cite: 9]。

* **Ext 6: Noise Mask (ノイズマスク)**
    * [cite_start]このマスクは、強度から導出されるデータ（上記Ext 1-5）の計算において、**信号の閾値を満たさなかった視野内のピクセル**を示します [cite: 266][cite_start]。つまり、このマスクで示されるピクセルはデータ品質が低く、解析から除外すべき領域であることを意味します。将来的には、統計的な手法から導出されるノイズレベルに置き換えられる予定です [cite: 268]。

***

### 偏光から導出される物理量（FeXIII 1074.7nmと1079.8nmのみ）

[cite_start]これらのデータは、偏光信号が強いFeXIII輝線でのみ提供される、コロナ磁場に関する情報です [cite: 275, 276]。

* **Ext 7: Weighted average I (重み付き平均 I)**
    * [cite_start]**中心3波長のストークスIの和を2で割った**ものです [cite: 278]。

* **Ext 8: Weighted average Q (重み付き平均 Q)**
    * [cite_start]**中心3波長のストークスQの和を2で割った**ものです [cite: 278]。

* **Ext 9: Weighted average U (重み付き平均 U)**
    * [cite_start]**中心3波長のストークスUの和を2で割った**ものです [cite: 278]。

* **Ext 10: Weighted average L (重み付き平均 L)**
    * [cite_start]上記の重み付き平均されたストークスQとUから計算された**直線偏光度 $L = \sqrt{Q^2 + U^2}$** です [cite: 224, 278]。

* **Ext 11: Azimuth (磁場の天体面方位角)**
    * [cite_start]ストークスUとQの比から計算される方位角（$Azimuth = 0.5 \times \arctan(U/Q)$）です [cite: 225, 279][cite_start]。これは**天球面上での磁場の向き**を示しており、水平方向から反時計回りに測られます [cite: 225][cite_start]。ただし、180度の不定性があります [cite: 225]。

* **Ext 12: Radial Azimuth (磁場の動径方向に対する方位角)**
    * [cite_start]Ext 11で求めた方位角を、**太陽中心からの動径方向を基準**として測り直したものです [cite: 226, 280][cite_start]。太陽の動径方向から反時計回りに測られます [cite: 227]。これにより、磁場が放射状に伸びているのか、それに対してどのくらい傾いているのかが分かります。

#### 科学的に意味があるデータはどれか？
結論として、すべてのデータが科学的な探求の対象となります。
- **コロナのダイナミクスやイベント（CMEなど）**に興味がある場合： 視線方向のドップラー速度 (Ext 4) や 輝線強度 (Ext 1, 3) の時間変化を追うことが特に重要です。
- コロナの温度や加熱メカニズムを研究したい場合： 輝線の半値全幅 (Ext 5) が重要な情報を提供します。異なる輝線の強度比も温度診断に利用できます。
- コロナの磁場構造そのものを解明したい場合： 天球面上の磁場の方向 (Ext 11, 12) や 直線偏光度 (Ext 10) は、他に類を見ないユニークで非常に価値の高いデータです。

## ファイル構成

```
py_folder/
├── ucomp_config.py      # 設定とユーティリティ関数
├── ucomp_scanner.py     # データスキャン機能
├── ucomp_plotting.py    # 描画・可視化機能
├── ucomp_main.py        # メイン実行スクリプト
└── README.md            # このファイル
```

## 使用方法

### 1. 対話的解析実行
```bash
cd /mnt/d/wsl/home/kinno-7010/Research/MK4_coronagraph/UCOMP/py_folder
python3 ucomp_main.py
```

### 2. 使用例実行
```bash
python3 ucomp_main.py example
```

### 3. Pythonスクリプトからの使用

#### 基本的な使用例
```python
from ucomp_scanner import *
from ucomp_plotting import *

# パラメータ設定
start_time = "2022-06-13T00:00:00"
end_time = "2022-06-13T05:00:00"
target_time = "2022-06-13T02:30:00"
wavelength = 1074

# 全Extension表示 (3×4)
fig = plot_ucomp_extensions(target_time, start_time, end_time, wavelength)
plt.show()

# 主要Extension表示 (2×2)
fig = create_ucomp_summary_plot(target_time, start_time, end_time, wavelength)
plt.show()

# 単一Extension詳細表示
fig = plot_single_extension(target_time, start_time, end_time, extension_num=1, wavelength=wavelength)
plt.show()
```

#### データスキャン
```python
# 利用可能な時刻一覧取得
times = get_available_ucomp_times(start_time, end_time, wavelength)
print(f"Found {len(times)} observations")

# 最も近いデータ検索
closest_data = find_closest_ucomp_data(target_time, start_time, end_time, wavelength)
if closest_data:
    map_obj, file_path = closest_data
    print(f"Closest time: {map_obj.date.iso}")
```

## 主要機能

### 1. データスキャン機能 (`ucomp_scanner.py`)
- `scan_ucomp_data()`: 時間範囲でデータをスキャン
- `find_closest_ucomp_data()`: 指定時刻に最も近いデータを検索
- `get_available_ucomp_times()`: 利用可能な時刻リストを取得

### 2. 描画機能 (`ucomp_plotting.py`)
- `plot_ucomp_extensions()`: 全Extension表示 (3×4レイアウト)
- `create_ucomp_summary_plot()`: 主要Extension表示 (2×2レイアウト)
- `plot_single_extension()`: 単一Extension詳細表示

### 3. 設定機能 (`ucomp_config.py`)
- Extension情報の定義
- 波長検証
- ファイルパス設定

## 解析オプション

### 1. 全Extension表示 (3×4)
- Ext 1-12を一度に表示
- 各ExtensionにColorbarを表示
- 太陽円盤境界を描画

### 2. 主要Extension表示 (2×2)
- 重要なExtension（1,2,4,10）のみ表示
- コンパクトな表示形式

### 3. 単一Extension詳細表示
- 指定したExtension単体を詳細表示
- 高解像度での解析に適用

### 4. 利用可能時刻一覧表示
- 指定時間範囲内の観測時刻を一覧表示
- データの確認に使用

## データパス設定

デフォルトデータパス:
```
/mnt/d/wsl/home/kinno-7010/Research/MK4_coronagraph/UCOMP/Rawdata
```

設定変更は `ucomp_config.py` の `UCOMP_DATA_DIR` を編集してください。

## 依存関係

- numpy
- matplotlib
- astropy
- pathlib
- integrated_analysis (SDO_Mk4_SOHO/py_folder/)

## 注意事項

1. **メモリ使用量**: Level 2データは大容量のため、メモリ使用量に注意
2. **ファイル形式**: 現在は.ftsファイルのみ対応
3. **Extension数**: ファイルによってExtension数が異なる場合があります
4. **波長依存**: FeXIII関連データ(Ext 7-12)は1074nm, 1079nmのみ

## トラブルシューティング

### よくある問題

1. **データが見つからない**
   - ファイル名形式を確認
   - 時間範囲を確認
   - データパスを確認

2. **メモリエラー**
   - 処理対象を減らす
   - 仮想環境のメモリを増やす

3. **表示エラー**
   - matplotlib backendを確認
   - GUI環境を確認

### デバッグ方法

```python
# データ存在確認
from pathlib import Path
data_dir = Path("/mnt/d/wsl/home/kinno-7010/Research/MK4_coronagraph/UCOMP/Rawdata")
files = list(data_dir.glob("*.ucomp.1074.l2.fts"))
print(f"Found {len(files)} files")

# Extension確認
from ucomp_plotting import read_ucomp_extensions
extensions = read_ucomp_extensions(files[0])
print(f"Available extensions: {list(extensions.keys())}")
```