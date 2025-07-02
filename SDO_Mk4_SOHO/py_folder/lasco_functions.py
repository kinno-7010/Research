"""
SOHO/LASCO-C2関連の関数群
SOHO/LASCO-C2データの読み込み、補正、描画、動画作成を行う
"""

from config import *
from sunpy.coordinates import get_horizons_coord
from sunpy.sun import constants


def create_corrected_lasco_map(filepath: Union[str, Path]) -> sunpy.map.Map:
    """SOHO/LASCOのFITSファイルを読み込み、観測者位置と太陽視半径を補正したMapを生成"""
    print(f"'{Path(filepath).name}' を読み込んでいます...")
    lasco_map_raw = sunpy.map.Map(filepath)
    print("SOHOの正確な位置をJPL HORIZONSから取得中...")
    observer_coord = get_horizons_coord('SOHO', lasco_map_raw.date)
    new_meta = lasco_map_raw.meta.copy()
    new_meta['hgln_obs'] = observer_coord.lon.value
    new_meta['hglt_obs'] = observer_coord.lat.value
    dsun_obs = observer_coord.radius
    new_meta['dsun_obs'] = dsun_obs.to('m').value
    rsun_physical = constants.get('radius')
    rsun_arc = np.arctan(rsun_physical / dsun_obs).to(u.arcsec).value
    new_meta['rsun_arc'] = rsun_arc
    print("メタデータの補正が完了しました。")
    corrected_map = sunpy.map.Map(lasco_map_raw.data, new_meta)
    return corrected_map


def plot_lasco_with_integer_contours(lasco_map: sunpy.map.Map, ax=None, show_grid: bool = True):
    """
    Mapオブジェクトをプロットし、太陽半径の整数倍の等高線を描画する。

    Args:
        lasco_map (sunpy.map.Map): プロットするMapオブジェクト。
        ax (matplotlib.axes.Axes, optional): プロット先のAxes。指定しない場合は新規作成する。
        show_grid (bool, optional): 背景の直交グリッドを表示するかどうか。
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection=lasco_map)

    # 輝度のノーマライゼーション
    norm = ImageNormalize(
        vmin=np.percentile(lasco_map.data, 0.1),
        vmax=np.percentile(lasco_map.data, 99.9),
        stretch=AsinhStretch(a=0.1)
    )

    # LASCO画像のプロット
    lasco_map.plot(axes=ax, cmap='sdoaia304', norm=norm, annotate=False)
    
    # 太陽本体のリムを描画 (1R☉)
    lasco_map.draw_limb(axes=ax, color='white', linestyle='solid', linewidth=1)
    
    # --- 等高線描画のための準備 ---
    # 1. 各ピクセルの座標グリッドから半径マップを生成 (単位: R_sun)
    x, y = np.meshgrid(*[np.arange(v.value) for v in lasco_map.dimensions]) * u.pix
    coords = lasco_map.pixel_to_world(x, y)
    rsun_arcsecs = lasco_map.meta['rsun_arc']
    radius_map = np.sqrt(coords.Tx**2 + coords.Ty**2) / (rsun_arcsecs * u.arcsec)
    
    # --- 等高線の描画 ---
    # 画像に収まる最大の整数半径まで、2R☉から等高線を引く
    max_radius = 6
    int_levels = np.arange(2, max_radius + 1)
    
    if len(int_levels) > 0:
        ax.contour(radius_map.value, levels=int_levels, colors='white', 
                   linewidths=1, linestyles='--', alpha=0.7)

    # 太陽中心のマーカー
    ax.plot(0, 0, '+', color='yellow', markersize=12, markeredgewidth=1.5,
            transform=ax.get_transform('world'))

    # 背景グリッドと軸ラベル、タイトル
    ax.grid(show_grid, color='white', linestyle=':', alpha=0.5)
    ax.set_xlabel('Solar-X (arcsec)')
    ax.set_ylabel('Solar-Y (arcsec)')
    ax.set_title(f'SOHO/LASCO-C2 | {lasco_map.date.iso}')
    
    
def find_files_in_time_range(directory: Path, start_time: str, end_time: str) -> list[Path]:
    """
    指定されたディレクトリと時間範囲に一致するFITSファイルのリストを返す。
    sunpy.map.Mapを使って確実に時刻を取得する。
    """
    print(f"ディレクトリ '{directory}' 内の.ftsファイルを検索しています...")
    all_files = sorted(directory.glob('*.fts'))
    
    t_start = Time(start_time)
    t_end = Time(end_time)
    
    filtered_files = []
    print("各ファイルを読み込み、時間範囲内か確認しています（時間がかかる場合があります）...")
    for f in tqdm(all_files, desc="ファイルフィルタリング"):
        try:
            # ★★★ 修正箇所: sunpy.map.Mapでファイル全体を読み込む ★★★
            m = sunpy.map.Map(f)
            # .date属性から直接、正確な時刻（Timeオブジェクト）を取得
            file_time = m.date
            
            if t_start <= file_time <= t_end:
                filtered_files.append(f)
        except Exception as e:
            # 読み込みに失敗したファイルはスキップ
            print(f"ファイル {f.name} の読み込み/解析中にエラー: {e}")
            continue
            
    return filtered_files


def plot_difference_map(diff_map: sunpy.map.Map, base_map_date: Time, ax=None, show_grid: bool = True):
    """
    差分Mapオブジェクトをプロットする。カラーマップやノーマライゼーションは差分表示に最適化。
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection=diff_map)

    # 差分データ用のノーマライゼーション（ゼロ中心で対称）
    vmax = np.percentile(np.abs(diff_map.data), 99.5) # 99.5%点の絶対値を取得
    norm = ImageNormalize(vmin=-vmax, vmax=vmax)

    # 差分画像なので、発散するカラーマップ（'gray'や'bwr'など）が適している
    diff_map.plot(axes=ax, cmap='gray', norm=norm, annotate=False)
    
    # 太陽本体のリムと等高線は空間的な参照のために描画する
    diff_map.draw_limb(axes=ax, color='lime', linestyle='solid', linewidth=1)
    
    x, y = np.meshgrid(*[np.arange(v.value) for v in diff_map.dimensions]) * u.pix
    coords = diff_map.pixel_to_world(x, y)
    rsun_arcsecs = diff_map.meta['rsun_arc']
    radius_map = np.sqrt(coords.Tx**2 + coords.Ty**2) / (rsun_arcsecs * u.arcsec)
    
    max_radius = 6
    int_levels = np.arange(2, max_radius + 1)
    
    if len(int_levels) > 0:
        ax.contour(radius_map.value, levels=int_levels, colors='lime', 
                   linewidths=1, linestyles='--', alpha=0.5)

    ax.plot(0, 0, '+', color='yellow', markersize=12, markeredgewidth=1.5,
            transform=ax.get_transform('world'))

    ax.grid(show_grid, color='white', linestyle=':', alpha=0.5)
    ax.set_xlabel('Solar-X (arcsec)')
    ax.set_ylabel('Solar-Y (arcsec)')
    
    # タイトルに差分元と差分先の時刻を両方表示する
    time_delta_minutes = (diff_map.date - base_map_date).to(u.min).value
    ax.set_title(
        f'SOHO/LASCO-C2 (Running Difference)\n'
        f'{base_map_date.strftime("%Y-%m-%d %H:%M:%S")} - {diff_map.date.strftime("%Y-%m-%d %H:%M:%S")}'
    )


def create_lasco_movie(file_list: list[Path], output_path: Path, fps: int):
    """ファイルリストからフレームを生成し、動画ファイルとして保存する"""
    frames = []
    print("各フレームをメモリ上にレンダリングしています...")
    for filepath in tqdm(file_list, desc="フレーム生成"):
        try:
            # create_corrected_lasco_mapは重いので、フィルタリングで読み込んだMapを再利用したいが、
            # 現状のコード構成では、ここで再度読み込む必要がある。
            current_map = create_corrected_lasco_map(filepath)
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1, projection=current_map)
            plot_lasco_with_integer_contours(current_map, ax, show_grid=False)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            image_from_plot = imageio.imread(buf)
            buf.close()
            
            frames.append(image_from_plot)
            plt.close(fig)
        except Exception as e:
            print(f"ファイル {filepath.name} の処理中にエラーが発生しました: {e}")
            plt.close('all')
            continue

    if not frames:
        print("有効なフレームが生成されなかったため、動画は作成されませんでした。")
        return

    print(f"\n全{len(frames)}フレームをレンダリングしました。動画に変換します...")
    try:
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
        print(f"\n動画ファイル '{output_path}' が正常に作成されました。")
    except Exception as e:
        print(f"\n動画の保存中にエラーが発生しました: {e}")
        
        
def create_lasco_running_difference_movie(file_list: list[Path], output_path: Path, fps: int):
    """
    ファイルリストからランニング・ディファレンス（差分）画像を生成し、動画として保存する。
    """
    if len(file_list) < 2:
        print("差分動画を作成するには少なくとも2つのファイルが必要です。")
        return
        
    frames = []
    print("連続するフレーム間の差分を計算し、レンダリングしています...")
    
    # 差分を取るため、ループは(リストの長さ - 1)回実行する
    for i in tqdm(range(len(file_list) - 1), desc="差分フレーム生成"):
        filepath_i = file_list[i]
        filepath_i_plus_1 = file_list[i+1]
        
        try:
            # 連続する2つの時刻のマップを読み込む
            map_i = create_corrected_lasco_map(filepath_i)
            map_i_plus_1 = create_corrected_lasco_map(filepath_i_plus_1)

            # データ配列の差分を計算
            # メタデータは後の時刻のものを引き継ぐ
            diff_data = map_i_plus_1.data - map_i.data
            diff_map = sunpy.map.Map(diff_data, map_i_plus_1.meta)
            
            # 差分マップをプロット
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1, projection=diff_map)
            plot_difference_map(diff_map, base_map_date=map_i.date, ax=ax, show_grid=False)
            
            # プロットした画像をメモリ上のバッファに保存
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()
            plt.close(fig)

        except Exception as e:
            print(f"ファイル {filepath_i.name} と {filepath_i_plus_1.name} の処理中にエラー: {e}")
            plt.close('all') # エラー時に開いているFigureをすべて閉じる
            continue

    if not frames:
        print("有効なフレームが生成されなかったため、動画は作成されませんでした。")
        return

    print(f"\n全{len(frames)}フレームの差分画像をレンダリングしました。動画に変換します...")
    try:
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
        print(f"\n動画ファイル '{output_path}' が正常に作成されました。")
    except Exception as e:
        print(f"\n動画の保存中にエラーが発生しました: {e}")


def plot_lasco_single_frame(file_path: Union[str, Path], corrected: bool = True, save_output: bool = False, output_dir: str = None):
    """
    LASCO-C2の単一フレームをプロットする関数
    
    Parameters:
    -----------
    file_path : str or Path
        LASCO FITSファイルのパス
    corrected : bool
        補正されたマップを使用するかどうか
    save_output : bool
        画像を保存するかどうか
    output_dir : str
        出力ディレクトリ
    """
    try:
        if corrected:
            lasco_map = create_corrected_lasco_map(file_path)
        else:
            lasco_map = sunpy.map.Map(file_path)
            
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection=lasco_map)
        plot_lasco_with_integer_contours(lasco_map, ax, show_grid=True)
        
        if save_output and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = Path(file_path).stem + '_corrected.png' if corrected else Path(file_path).stem + '.png'
            output_path = Path(output_dir) / filename
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"画像を保存しました: {output_path}")
        
        plt.show()
        return fig, ax
        
    except Exception as e:
        print(f"ファイル {file_path} の処理中にエラー: {e}")
        return None, None