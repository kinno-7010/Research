# 太陽物理学研究 - Solar Physics Research

## 🌞 研究概要
太陽コロナ質量放出（CME）の動力学とII型太陽電波バースト（Type II SRB）の関連性を多波長・多観測機器データを用いて研究。電子加速過程と太陽表面の磁場・密度構造の推定を目的とする。

## 📡 使用観測機器とデータ

### 🛰️ 宇宙観測機器

#### SDO (Solar Dynamics Observatory)
- **機器**: AIA (Atmospheric Imaging Assembly)
- **観測波長**: 131Å, 171Å, 193Å
- **観測内容**: 
  - 太陽コロナの温度構造解析
  - CME発生過程の高時間分解能観測
  - 太陽フレアとCMEの関連性調査
- **データサイト**: [JSOC Stanford](http://jsoc.stanford.edu/) / [SDO NASA](https://sdo.gsfc.nasa.gov/data/)
- **関連論文**: Lemen, J. R., et al. (2012). "The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)." *Solar Physics*, 275, 17-40. [DOI: 10.1007/s11207-011-9776-8](https://doi.org/10.1007/s11207-011-9776-8)

#### SOHO (Solar and Heliospheric Observatory)
- **機器**: LASCO-C2 (Large Angle and Spectrometric Coronagraph)
- **観測内容**: 
  - 白色光コロナグラフィ（視野：2.2-6太陽半径）
  - CME伝搬速度測定
  - pB (polarized Brightness) データによる電子密度推定
  - 電子密度トモグラフィ解析
- **データサイト**: [CDAW NASA](https://cdaw.gsfc.nasa.gov/) / [LASCO NRL](https://lasco-www.nrl.navy.mil/)
- **関連論文**: Brueckner, G. E., et al. (1995). "The Large Angle Spectroscopic Coronagraph (LASCO)." *Solar Physics*, 162, 357-402. [DOI: 10.1007/BF00733434](https://doi.org/10.1007/BF00733434)

#### Parker Solar Probe (PSP)
- **機器**: FIELDS (電場・磁場測定装置)
- **観測周波数**: 10 kHz - 19 MHz
- **観測内容**: 
  - Type II太陽電波バースト検出
  - 太陽風中の電磁場揺らぎ測定
  - 近日点での高時間分解能電波観測
- **データサイト**: [Parker Solar Probe Mission](https://parkersolarprobe.jhuapl.edu/)
- **関連論文**: Bale, S. D., et al. (2016). "The FIELDS Instrument Suite for Solar Probe Plus." *Space Science Reviews*, 204, 49-82. [DOI: 10.1007/s11214-016-0244-5](https://doi.org/10.1007/s11214-016-0244-5)

### 🏔️ 地上観測機器

#### MLSO K-COR (Mauna Loa Solar Observatory)
- **機器**: Mark-4 K-Coronagraph → COSMO K-Coronagraph
- **観測内容**:
  - 地上コロナグラフィ（視野：1.05-3太陽半径）
  - CME高度進化の詳細測定（15秒間隔）
  - 偏光度データ(pB)による密度構造解析
- **データサイト**: [MLSO HAO NCAR](https://www2.hao.ucar.edu/mlso/) / [データリクエスト](mailto:mlso_data_requests@ucar.edu)
- **関連論文**: 
  - Mark-4: Elmore, D. F., et al. (2003). "Calibration of a ground-based solar coronal polarimeter." *Proc. SPIE 4843*, 66-76. [DOI: 10.1117/12.459279](https://doi.org/10.1117/12.459279)
  - K-Cor: de Wijn, A. G., et al. (2012). "Stray light and polarimetry considerations for the COSMO K-Coronagraph." *Proc. SPIE 8444*, 84443N. [DOI: 10.1117/12.926511](https://doi.org/10.1117/12.926511)

#### 電波観測ネットワーク

##### IPRT (Institute for Plasma Research and Technology)
- **周波数帯**: 100-500 MHz
- **観測内容**: 太陽電波バースト動的スペクトラム
- **時間分解能**: 高時間分解能観測

##### Learmonth Observatory (RSTN)
- **ネットワーク**: Radio Solar Telescope Network
- **観測内容**: 広帯域太陽電波観測
- **データ形式**: SRS形式

##### Yamagawa Observatory
- **周波数帯**: 
  - 低周波: 70-1024 MHz
  - 高周波: 1000-9000 MHz
- **観測内容**: 高時間分解能太陽電波観測

## 🔬 研究内容別フォルダー構成

### データ解析・処理
- **`DensityModel/`**: LASCO偏光データからの電子密度推定、各種密度モデル検証
- **`SDO_Mk4_SOHO/`**: 多観測機器統合解析（AIA+K-COR+LASCO）
- **`cme_analysis/`**: CME動力学解析結果

### 観測機器別解析
- **`SDO/AIA/`**: コロナ温度構造・CME発生過程解析
- **`SOHO/`**: LASCO-C2白色光コロナグラフィ・CME伝搬解析
- **`MK4_coronagraph/`**: 地上コロナグラフによるCME高度測定
- **`RadioData/`**: 各観測所での太陽電波バースト解析

### 開発・ユーティリティ
- **`sswdb/`**: SolarSoft IDL環境・IDLコード移植作業
- **`SDO_Mk4_SOHO/py_folder/`**: Python解析ツール群

## 🎯 主要研究テーマ

### 1. CME動力学研究
- 多高度でのCME伝搬速度測定
- 密度構造がCME伝搬に与える影響
- CME前面衝撃波の形成・発達過程

### 2. Type II電波バースト解析
- CME衝撃波前面での電子加速過程
- 電波放射周波数からのコロナ密度推定
- 動的スペクトラムによる伝搬方向決定

### 3. 多観測機器統合解析
- AIA+K-COR+LASCOによる広視野CME追跡
- 電波観測とコロナグラフィの時間同期解析
- 異なる高度での物理量相関研究

## 📊 解析対象イベント
**主要解析日時**: 2022年6月13日 02:00-05:00 UT

このイベントは複数観測機器で同時観測され、CME発生からType II電波バースト発生までの全過程を追跡可能な貴重なデータセットとなっている。

## 📚 主要観測機器の関連論文

### 宇宙観測機器
- **SDO/AIA**: Lemen, J. R., et al. (2012). "The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)." *Solar Physics*, 275, 17-40.
- **SOHO/LASCO**: Brueckner, G. E., et al. (1995). "The Large Angle Spectroscopic Coronagraph (LASCO)." *Solar Physics*, 162, 357-402.
- **Parker Solar Probe/FIELDS**: Bale, S. D., et al. (2016). "The FIELDS Instrument Suite for Solar Probe Plus." *Space Science Reviews*, 204, 49-82.

### 地上観測機器
- **MLSO Mark-4**: Elmore, D. F., et al. (2003). "Calibration of a ground-based solar coronal polarimeter." *Proc. SPIE 4843*, 66-76.
- **MLSO K-Cor**: de Wijn, A. G., et al. (2012). "Stray light and polarimetry considerations for the COSMO K-Coronagraph." *Proc. SPIE 8444*, 84443N.

## 🔗 関連リンク
- [Virtual Solar Observatory (VSO)](https://sdac.virtualsolar.org/)
- [SolarSoft IDL](https://www.lmsal.com/solarsoft/)
- [CDAW CME Catalog](https://cdaw.gsfc.nasa.gov/CME_list/)
- [Space Weather Prediction Center](https://www.swpc.noaa.gov/)
