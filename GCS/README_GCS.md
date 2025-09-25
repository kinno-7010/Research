# README_GCS.md

## 概要

`plot_GCS.py` は、既存のコロナグラフ合成図（AIA193・Mk4・LASCO-C2）に GCS (Graduated Cylindrical Shell) モデルのワイヤーフレームを重ね描きするドライバスクリプトです。内部では `GCS/gcs_overlay/gcs_overlay.py` と `GCS/gcs_overlay/gcs_geometry.py` が利用されており、PyThea 由来の GCS 幾何学実装を呼び出して 3D 座標を生成し、SunPy を通じて LASCO の視線に投影します。

```
plot_GCS.py  → overlay_gcs_on_composite() → sample_gcs_wireframe_points()
```

## ファイル構成

- `plot_GCS.py`
  - ユーザーの入力パラメータを `GCSParams` にまとめ、`overlay_gcs_on_composite` を呼び出します。
  - オプションで footpoint から tilt を再推定するルーチン (`gcs_overlay.footpoint_fit`) を利用できます。
  - `integrated_analysis.create_single_diff_image` を呼び出し、LASCO/Mk4/AIA の差分合成図を取得した後、GCS ワイヤーフレームを描画して `plt.show()` します。

- `GCS/gcs_overlay/gcs_overlay.py`
  - `overlay_gcs_on_composite`: 合成図を生成 (`create_single_diff_image`) → LASCO 地図と投影スケールを取得 → `overlay_gcs_wireframe_on_axes` でワイヤーフレームを投影。
  - `overlay_gcs_wireframe_on_axes`: `sample_gcs_wireframe_points` から得た 3D HGS 座標を LASCO 視線に変換し、ピクセル座標に落とし込んで Matplotlib の Axes に線として描画します。
  - LASCO 観測者座標取得では SunPy の `ephemeris.get_horizons_coord` を使います。ネットワーク不通時は SOHO を 1 AU の Earth 視点とするフォールバックが組み込まれています。

- `GCS/gcs_overlay/gcs_geometry.py`
  - `GCSParams`: GCS モデルパラメータ（下記参照）。
  - `sample_gcs_wireframe_points`: PyThea の GCS モデルを用いてワイヤーフレーム点群を生成します。必要に応じて torus front・meridians・legs をサンプリングします。
  - PyThea が求める `seaborn.color_palette` 依存は最小スタブで補っています。

## GCS パラメータの意味

`GCSParams` は以下のフィールドを持ちます（単位は括弧内）。

| パラメータ | 意味 | 備考 |
|------------|------|------|
| `h_apex` (R⊙) | GCS 輸送殻の前面（apex）までの太陽中心からの距離 | モデル全体のサイズを決定。PyThea では apex height。
| `kappa` (-) | アスペクト比 = 小半径 / 大半径 | 0 に近いほど扁平、1 に近いほど球状。Thernisien (2011) の定義。
| `alpha_deg` (deg) | ハーフアングル（開き角） | トーラス前面の側方広がり、脚の開口角を制御。
| `tilt_deg` (deg) | GCS 軸周りの回転角 | 観測者から見た回転。正値で反時計回り。
| `lon_deg` (deg) | Heliographic Stonyhurst 経度 | CME 中心軸の向き。0° が地球方向。
| `lat_deg` (deg) | Heliographic Stonyhurst 緯度 | 同上。

PyThea 内部ではこれらから以下の値が算出されます。

- `rcenter`: GCS torus の中心までの距離（`rcenter_()`）。
- `rapex`: トーラス断面半径 (`rapex_()`)。
- `h`: CME 脚の長さ (`h_()`)。

### パラメータ設定の順序

1. 観測時刻 `ts` を決める（例: `2022-06-13T03:20:00`）。
2. 観測データに基づき、おおよその CME 位置・広がりを見積もる。
   - AIA/Mk4/LASCO の差分画像で CME の前縁高度や幅を確認。
3. `h_apex`, `kappa`, `alpha_deg` をセット。初期値の例:
   - `h_apex`: 2.0 – 6.0 (R⊙)
   - `kappa`: 0.2 – 0.6
   - `alpha_deg`: 20° – 45°
4. `lon_deg`, `lat_deg` で CME の中央軸方向を設定。STEREO 等の多視点情報があればそれを利用。
5. `tilt_deg` を調整。オプションで `FIT_TILT_FROM_SOURCE` を有効化し、電波源や脚根位置から最適化することも可能です。
6. `plot_GCS.py` を実行し、描画結果を確認してパラメータを微調整します。

## 実行方法

```bash
# 例: デフォルト値を使って実行
python3 plot_GCS.py

# パラメータを引数で指定する場合
python3 plot_GCS.py <ts> <h_apex> <kappa> <alpha_deg> <tilt_deg> <lon_deg> <lat_deg>

# 例
python3 plot_GCS.py 2022-06-13T03:12:00 3.5 0.30 35.0 20.0 5.0 0.0
```

### footpoint から tilt をフィットする場合

```bash
python3 plot_GCS.py 2022-06-13T03:12:00 3.5 0.3 35.0 20.0 5.0 0.0 \
                    fit <lon_src1> <lat_src1> [<lon_src2> <lat_src2>] [<tilt_lo> <tilt_hi> <tilt_step>]
```

- `FIT_TILT_FROM_SOURCE` は非対話型で tilt を再推定します。
- SDO や radio source など、脚根位置を反映したい場合に有用です。

## 注意事項

- **バックエンド**: 現在の `plot_GCS.py` は `plt.show()` を呼び出すだけなので、GUI バックエンド (例: TkAgg) が利用可能な環境で実行してください。ヘッドレス環境や Tk 未インストール環境では ImportError となります。その際は `matplotlib.use('Agg')` を明示するか、画像保存に切り替えるなど調整が必要です。
- **PyThea 依存**: `PyThea/Kouloumvakos_GitHub` ディレクトリが存在し、`PyThea.geometrical_models.gcs` が読み込めることが前提です。初回は `seaborn` が無くてもスタブで動作するようになっています。
- **データパス**: `integrated_analysis.py` が参照する各種データ（LASCO, Mk4, AIA）のパスが正しく設定されている必要があります。
- **処理時間**: 画像スキャンや LASCO 補正処理に時間を要するため、実行後プロットが表示されるまで数分かかる場合があります。
- **Horizons API**: ネットワークが利用できない環境では SOHO の視点情報を単純化したフォールバック値に置き換えています。精度が必要な解析ではオンライン環境での実行を推奨します。

## 参考

- Thernisien et al. (2006), Thernisien (2011) — GCS モデル原典。
- PyThea: https://github.com/AKoulouris/PyThea — 本リポジトリに含まれる参照実装。
