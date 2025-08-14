# HF Radio Data Analysis Package

## 概要

このパッケージは、太陽電波観測のHF（高周波）データ解析用のPythonモジュール群です。元のJupyter Notebook (`HF_plt.ipynb`) から機能別に分割・再構成されており、太陽電波バーストの動的スペクトラム解析、ピーク検出、ドリフト線解析、統計解析などを行うことができます。

## データソース

### 観測データ
- **データサイト**: http://adrastea.gp.tohoku.ac.jp/~jupiter/
- **対象データ**: 高頻度太陽電波観測データ (it_h1_hf_20220613_v01.cdf)

### 機器情報
- **IUGONET データ詳細**: http://www.iugonet.org/data/workshop/20120222/iugonet_20120223_kumamoto.pdf
- 東北大学惑星プラズマ・大気研究センター（PPARC）による高頻度太陽電波観測装置

## ファイル構成

```
py_folder/
├── __init__.py              # パッケージ初期化ファイル
├── utils.py                 # 基本的な時間・データ処理関数
├── spectrum_plot.py         # 動的スペクトラム描画関数
├── peak_analysis.py         # ピーク検出・解析関数
├── drift_analysis.py        # ドリフト線解析関数
├── frequency_conversion.py  # 密度・周波数変換関数
├── statistics.py            # 統計解析関数
├── solar_models.py          # 太陽コロナ密度モデル
├── main_analysis.py         # メインデータ読み込み・解析スクリプト
└── README.md               # このファイル
```

## モジュール詳細

### 1. utils.py
基本的な時間・データ処理関数を提供

- `_to_datetime(dt_input)`: 日時文字列またはdatetimeオブジェクトをdatetimeに変換
- `_to_seconds(times)`: matplotlib日付数を経過秒数配列に変換
- `_slice_data(time_array, data, start_dt, end_dt, freq_mhz, freq_min, freq_max)`: 時間・周波数範囲でデータをスライス

### 2. spectrum_plot.py
動的スペクトラム描画関数を提供

- `plot_dynamic_spectrum()`: 基本的な動的スペクトラムプロット
- `plot_removed_dynamic_spectrum()`: 3σ外れ値除去後の動的スペクトラムプロット
- `plot_drift_line()`: ドリフト線の描画とエンドポイントマーキング

### 3. peak_analysis.py
ピーク検出・解析機能を提供

- `calculate_dynamic_spectrum_with_peak()`: 動的スペクトラムとピーク周波数の描画
- `calculate_peak_time_and_freq()`: 指定区間・周波数帯のピーク時刻・周波数抽出
- `plot_removed_dynamic_spectrum_with_peak()`: 3σクリーニング後の動的スペクトラムとピーク描画

### 4. drift_analysis.py
ドリフト線解析機能を提供

- `time_weighted_average(segments)`: セグメント上の時間加重平均レート計算
- `compute_drift_rates_and_durations(segments)`: ドリフトレートと持続時間の計算
- `plot_segments(ax, segments, color, label)`: 複数ドリフトセグメントの描画と平均レート計算

### 5. frequency_conversion.py
周波数・密度変換機能を提供

- `density_from_frequency(freq_mhz)`: 周波数[MHz] → 電子密度[cm^-3]変換
- `frequency_from_density(dens_cm3)`: 電子密度[cm^-3] → 周波数[MHz]変換
- `density_from_frequency_harmonic()`: 2次高調波用の密度変換
- `frequency_from_density_harmonic()`: 2次高調波用の周波数変換

### 6. statistics.py
統計解析機能を提供

- `compute_drift_stats(times, freqs, outlier_z)`: ドリフト統計の計算（外れ値除去含む）
- `calculate_fit_with_error(ax, times, freqs)`: 線形回帰フィットと誤差計算
- `plot_fit_with_error(ax, times, freqs, color, label)`: フィット直線の描画とエラー表示

### 7. solar_models.py
太陽コロナ密度モデルを提供

- `Saito1970(rho, phi)`: Saito1970太陽コロナ密度モデル
- `Saito1977(rho)`: Saito1977太陽コロナ密度モデル（2.5-5.5Rs）
- `find_rho_for_value(func, target, rho_min, rho_max)`: 指定密度値に対応する太陽半径距離の探索

### 8. main_analysis.py
メインデータ読み込み・解析スクリプト

- `load_hf_data(file_path)`: CDFファイルからHF電波データを読み込み
- `main()`: メイン解析関数（動的スペクトラムとピーク検出の例）

## 使用方法

### 基本的な使用例

```python
# メイン解析スクリプトの実行
python main_analysis.py

# 個別モジュールの使用例
from utils import _slice_data, _to_datetime
from spectrum_plot import plot_dynamic_spectrum
from peak_analysis import calculate_peak_time_and_freq

# データの読み込み
from main_analysis import load_hf_data
time, frequency_mhz, data = load_hf_data("path/to/data.cdf")

# 指定時間範囲・周波数範囲でのピーク抽出
peak_times, peak_freqs = calculate_peak_time_and_freq(
    time, frequency_mhz, data,
    "2022-06-13T03:00:00", "2022-06-13T04:00:00",
    20.0, 80.0, (3, 3), 2.0
)
```

### 必要なライブラリ

```python
import numpy as np
import cdflib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, LogLocator, FuncFormatter
from matplotlib.dates import SecondLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import median_filter
from scipy.stats import zscore, linregress
from scipy.optimize import root_scalar
import datetime as dt
```

## データフォーマット

### 入力データ（CDFファイル）
- `Epoch`: 時間データ（ミリ秒）
- `Frequency`: 周波数データ（Hz）
- `RH`: 右旋円偏波パワーフラックス密度（dB）

### 物理定数
- 電子電荷: e = 1.60217662e-19 C
- 電子質量: m_e = 9.10938356e-31 kg
- 真空誘電率: ε₀ = 8.854187817e-12 F/m

## 解析可能な現象

1. **太陽電波バーストの動的スペクトラム解析**
2. **Type II太陽電波バーストのドリフト線解析**
3. **電子プラズマ周波数からの電子密度推定**
4. **太陽コロナ密度構造の推定**
5. **電子加速過程の統計解析**

## 参考文献・リンク

- **データ提供**: 東北大学惑星プラズマ・大気研究センター
- **データサイト**: http://adrastea.gp.tohoku.ac.jp/~jupiter/
- **IUGONET プロジェクト**: http://www.iugonet.org/data/workshop/20120222/iugonet_20120223_kumamoto.pdf
- **観測装置詳細**: 高頻度太陽電波観測装置（東北大学PPARC）

## ライセンス

このソフトウェアは研究目的で作成されています。データの使用については各データ提供機関のポリシーに従ってください。

## 作成者

太陽物理学研究グループ
東北大学大学院理学研究科

## 更新履歴

- v1.0.0 (2025-01-14): 初期リリース
  - HF_plt.ipynbからの機能分割・モジュール化
  - 8つの機能別Pythonファイルに再構成
  - データパス修正とパッケージ化