import os
from datetime import datetime, timedelta
from pathlib import Path
import re

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from astropy.visualization import ImageNormalize, PowerStretch
from matplotlib.ticker import FuncFormatter
from astropy.io import fits

# --- 定数定義 ---
BASE_DATA_DIR = Path(r"/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/EUVI/Rawdata")

def find_closest_euvi_file(target_datetime, wavelength):
    """
    指定された日時と波長に最も近いEUVIファイルを検索する
    
    Parameters:
    -----------
    target_datetime : datetime
        目標とする観測日時
    wavelength : str or int
        EUVI観測波長 (195, 171, 304, 284のいずれか)
        
    Returns:
    --------
    Path or None
        最も近いファイルのパス（見つからない場合はNone）
    """
    wavelength_str = str(wavelength)
    
    if not BASE_DATA_DIR.exists():
        print(f"エラー: Rawdataディレクトリが見つかりません - {BASE_DATA_DIR}")
        return None
    
    # 指定波長のファイルを検索
    pattern = f"*{wavelength_str}eu_R.fts"
    euvi_files = list(BASE_DATA_DIR.glob(pattern))
    
    if not euvi_files:
        print(f"エラー: 波長{wavelength_str}Åのファイルが見つかりません")
        return None
    
    print(f"波長{wavelength_str}Åのファイル{len(euvi_files)}個を検索中...")
    
    closest_file = None
    min_time_diff = float('inf')
    
    for file_path in euvi_files:
        try:
            # ファイル名から日時を抽出
            # 例: 20220613_020000_195eu_R.fts -> 2022-06-13 02:00:00
            filename = file_path.name
            match = re.match(r'(\d{8})_(\d{6})_\d{3}eu_R\.fts', filename)
            
            if match:
                date_str = match.group(1)  # YYYYMMDD
                time_str = match.group(2)  # HHMMSS
                
                # datetimeオブジェクトに変換
                file_datetime = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                
                # 時間差を計算
                time_diff = abs((target_datetime - file_datetime).total_seconds())
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_file = file_path
                    
        except Exception as e:
            print(f"ファイル {filename} の解析中にエラー: {e}")
            continue
    
    if closest_file:
        time_diff_minutes = min_time_diff / 60
        print(f"最近接ファイル: {closest_file.name} (時間差: {time_diff_minutes:.1f}分)")
        return closest_file
    else:
        print(f"波長{wavelength_str}Åの有効なファイルが見つかりませんでした")
        return None

def plot_euvi_image(datetime_str, wavelength):
    """
    指定された日時と波長のSTEREO-A/SECCHI/EUVI画像をWCSベースでプロットし、
    軸の目盛りラベルのみを太陽中心を(0,0)とするピクセル単位で表示します。
    太陽リムとグリッドはWCSに基づいて描画されます。
    指定時刻に最も近い時間帯のデータを自動選択します。
    
    Parameters:
    -----------
    datetime_str : str
        プロット対象の日時 (フォーマット: "YYYY-MM-DD HH:MM")
    wavelength : str or int
        EUVI観測波長 (195, 171, 304, 284のいずれか)
    """
    
    # 1. 日時の処理
    try:
        target_datetime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
    except ValueError:
        print(
            f"エラー: 日時文字列 '{datetime_str}' の形式が無効です。"
            " 'YYYY-MM-DD HH:MM' 形式で指定してください。"
        )
        return

    # 波長を文字列に変換
    wavelength_str = str(wavelength)
    
    print(f"指定日時: {target_datetime.strftime('%Y-%m-%d %H:%M')}")
    print(f"指定波長: {wavelength_str} Å")
    
    # 2. 最近接ファイルを検索
    file_path = find_closest_euvi_file(target_datetime, wavelength)
    if file_path is None:
        return

    try:
        euvi_map = sunpy.map.Map(file_path)
        print(f"ファイル '{file_path}' を正常に読み込みました。")
        print(f"データ形状: {euvi_map.data.shape}")
        print(f"観測日時: {euvi_map.date}")
        print(f"観測波長: {euvi_map.wavelength}")
    except Exception as e:
        print(f"ファイルの読み込み・初期処理中にエラーが発生しました: {e}")
        return

    image_data = euvi_map.data
    wcs_info = euvi_map.wcs

    # 3. データの前処理（NaN値の処理）
    if np.all(np.isnan(image_data)) or np.all(image_data == 0):
        print("警告: 画像データがすべてNaNまたは0です。")
        return

    # 4. データ正規化
    vmin_percentile = 1.0
    vmax_percentile = 99
    stretch_power = 0.5
    valid_data = image_data[np.isfinite(image_data)]
    norm = None
    if valid_data.size > 0:
        norm = ImageNormalize(
            # vmin=np.percentile(valid_data, vmin_percentile),
            # vmax=np.percentile(valid_data, vmax_percentile),
            vmin=-3,
            vmax=3,
            stretch=PowerStretch(stretch_power),
            clip=True
        )

    # 5. WCS座標系の確認
    if hasattr(wcs_info, 'wcs') and hasattr(wcs_info.wcs, 'cunit'):
        if wcs_info.wcs.cunit[0] == u.arcsec and wcs_info.wcs.cunit[1] == u.arcsec:
            print("情報: WCS座標系がarcsec単位で設定されています。")
        else:
            print(f"警告: WCS座標系の単位: X={wcs_info.wcs.cunit[0]}, Y={wcs_info.wcs.cunit[1]}")
    else:
        print("警告: WCS座標系の単位情報が取得できません。")

    # 6. プロットの準備 (WCSAxesを使用)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection=wcs_info)

    # 7. カラーマップの選択（SSWIDLのeit_colors方式を参考）
    # SSWIDLではeit_colorsがEUVI波長に応じたカラーテーブルを設定
    colormap_dict = {
        '195': 'sohoeit195',  # 緑系
        '171': 'sohoeit171',  # 金系  
        '304': 'sohoeit304',  # 赤系
        '284': 'sohoeit284'   # 青系
    }
    cmap = colormap_dict.get(wavelength_str, 'sohoeit195')  # デフォルトは195
    
    print(f"使用カラーマップ: {cmap} (波長 {wavelength_str} Å)")

    # 8. 画像のプロット
    im = ax.imshow(image_data, origin='lower', cmap=cmap, norm=norm)

    # 9. 太陽リムとグリッドの描画
    try:
        euvi_map.draw_limb(axes=ax, color='white', linestyle='dashed', linewidth=1.2)
        euvi_map.draw_grid(axes=ax, grid_spacing=15*u.deg, color='white', linestyle='dotted', linewidth=0.8, alpha=0.7)
        print("情報: 太陽リムとグリッドをWCSベースで描画しました。")
    except Exception as e_draw:
        print(f"警告: 太陽リムまたはグリッドの描画に失敗しました: {e_draw}")

    # 10. タイトルと軸ラベルの設定
    title_str_parts = [
        f"STEREO-A/SECCHI/EUVI {wavelength_str} Å",
        f"Time: {euvi_map.date.strftime('%Y-%m-%d %H:%M:%S UT')}"
    ]

    ax.set_title("\n".join(title_str_parts), fontsize=12, pad=15)

    # 11. 軸ラベルを太陽中心原点のarcsec単位に設定
    ax.coords[0].set_axislabel("Solar X (arcsec)")
    ax.coords[1].set_axislabel("Solar Y (arcsec)")

    # 12. カラーバーの追加（COR1スタイルの目盛り配置）
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label(f'Intensity (DN/s)', rotation=270, labelpad=15, fontsize=12)
    
    # SSWIDLスタイルのカラーバー目盛り配置
    if norm:
        # 正規化範囲を取得
        vmin, vmax = norm.vmin, norm.vmax
        
        # データ範囲に応じた適切な目盛り数を決定
        data_range = vmax - vmin
        
        if data_range > 1000:
            # 大きな値の場合は5-7個の目盛り
            n_ticks = 6
        elif data_range > 100:
            # 中程度の値の場合は6-8個の目盛り
            n_ticks = 7
        else:
            # 小さな値の場合は8-10個の目盛り
            n_ticks = 8
        
        # 等間隔で目盛り位置を計算
        ticks = np.linspace(vmin, vmax, n_ticks)
        
        # 目盛りラベルの精度を データ範囲に応じて調整
        if data_range > 1000:
            tick_labels = [f'{tick:.0f}' for tick in ticks]
        elif data_range > 10:
            tick_labels = [f'{tick:.1f}' for tick in ticks]
        else:
            tick_labels = [f'{tick:.3f}' for tick in ticks]
        
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        
        print(f"カラーバー設定: {n_ticks}個の目盛り, 範囲=[{vmin:.3f}, {vmax:.3f}]")

    plt.tight_layout()
    output_dir = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/EUVI"
    plt.savefig(f"{output_dir}/euvi_{wavelength_str}_{datetime_str}.png", dpi=300, bbox_inches='tight')
    plt.show()


def list_available_euvi_files():
    """
    Rawdataディレクトリ内のEUVIファイル一覧を表示
    """
    print(f"EUVI Rawdataディレクトリ: {BASE_DATA_DIR}")
    
    if not BASE_DATA_DIR.exists():
        print("エラー: Rawdataディレクトリが見つかりません。")
        return
    
    euvi_files = sorted(BASE_DATA_DIR.glob("*.fts"))
    
    if not euvi_files:
        print("EUVIファイルが見つかりません。")
        return
    
    print(f"\n利用可能なEUVIファイル ({len(euvi_files)}個):")
    print("-" * 60)
    
    wavelengths = {'195': [], '171': [], '304': [], '284': []}
    
    for file_path in euvi_files:
        filename = file_path.name
        print(f"  {filename}")
        
        # 波長別に分類
        for wl in wavelengths.keys():
            if f"{wl}eu" in filename:
                wavelengths[wl].append(filename)
                break
    
    print("\n波長別ファイル数:")
    print("-" * 30)
    for wl, files in wavelengths.items():
        print(f"  {wl} Å: {len(files)}個")


if __name__ == "__main__":
    # 使用例
    print("=== STEREO-A/SECCHI/EUVI画像プロットツール ===")
    
    # 利用可能なファイルを表示
    list_available_euvi_files()
    
    # プロット例
    print("\n=== プロット例 ===")
    plot_euvi_image("2022-06-13 03:00", 304)