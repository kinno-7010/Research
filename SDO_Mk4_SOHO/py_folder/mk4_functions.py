"""
MK4コロナグラフ関連の関数群
MK4コロナグラフデータの読み込み、処理、描画、動画作成を行う
"""

from config import *


def read_fits_file(fits_path: str):
    """FITSファイルを読み込み、データとヘッダーを返す."""
    with fits.open(fits_path, ignore_missing_end=True) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    return data, header


def calculate_statistics(data, exclude_zeros=False):
    """データの統計情報を計算."""
    if exclude_zeros:
        data = data[data > 0]
    if data.size == 0:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
    return {
        "min": np.min(data),
        "max": np.max(data),
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data),
    }


def draw_frame(data, header, ax):
    """単一フレームを描画する関数（変更なし）."""
    solar_x = header.get('CRPIX1', data.shape[1] // 2) - 1
    solar_y = header.get('CRPIX2', data.shape[0] // 2) - 1
    solar_radius = header.get('R_SUN', 0)

    pixel_to_rsun = 1 / solar_radius if solar_radius > 0 else 1

    stats = calculate_statistics(data)
    data_mean, data_std = stats["mean"], stats["std"]
    custom_vmin = data_mean - 5 * data_std
    custom_vmax = data_mean + 5 * data_std

    extent = [
        -solar_x * pixel_to_rsun, (data.shape[1] - solar_x) * pixel_to_rsun,
        -solar_y * pixel_to_rsun, (data.shape[0] - solar_y) * pixel_to_rsun
    ]
    ax.clear()
    im = ax.imshow(data, cmap='gray', origin='lower', vmin=custom_vmin, vmax=custom_vmax, extent=extent)

    ax.scatter(0, 0, color='red', s=50, label="Solar Center")
    if solar_radius > 0:
        circle = Circle((0, 0), 1, color='red', fill=False, lw=1.5, label="Solar Disk")
        ax.add_patch(circle)
        
    # こちらの円は不要であればコメントアウトしてください
    custom_radius = np.sqrt((0 - solar_x) ** 2 + (solar_y - solar_y) ** 2) * pixel_to_rsun
    circle = Circle((0, 0), custom_radius, color='white', fill=False, linestyle='--', lw=1.0, label="Custom Radius")
    ax.add_patch(circle)

    ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.5)

    date_obs = header.get('DATE-OBS', 'Unknown Time')
    ax.set_title(f"Time: {date_obs}", fontsize=12)
    ax.set_xlabel("X [Solar Radius $R_\\\\odot$]", fontsize=10)
    ax.set_ylabel("Y [Solar Radius $R_\\\\odot$]", fontsize=10)

    return im


def create_movie_from_fits_imageio(start_time_str: str, end_time_str: str, data_folder: str, output_path: str, fps: int = 15):
    """
    MK4データから動画を作成する関数
    """
    print("動画の作成を開始します...")
    
    # 1. 時刻オブジェクトの準備 (変更なし)
    try:
        start_time = Time(start_time_str, scale='utc')
        end_time = Time(end_time_str, scale='utc')
        print(f"対象期間: {start_time.iso} から {end_time.iso} まで")
    except Exception as e:
        print(f"時刻の変換に失敗しました: {e}")
        return

    # 2. 対象ファイルのフィルタリング (変更なし)
    all_files = sorted(glob.glob(os.path.join(data_folder, "*.fts")))
    files_to_process = []
    pattern = re.compile(r'(\d{8})_(\d{6})')
    
    for f_path in all_files:
        match = pattern.search(os.path.basename(f_path))
        if match:
            date_str, time_str = match.groups()
            try:
                iso_time_str = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
                file_time = Time(iso_time_str, scale='utc')
                if start_time <= file_time <= end_time:
                    files_to_process.append(f_path)
            except ValueError:
                continue
    
    if not files_to_process:
        print("指定された期間に該当するファイルが見つかりませんでした。")
        return

    print(f"{len(files_to_process)} 個のファイルからフレームを生成します。")

    # 3. フレームを1枚ずつ生成してリストに保存
    frames = []
    for file_path in tqdm(files_to_process, desc="フレーム生成中"):
        try:
            data, header = read_fits_file(file_path)
            fig, ax = plt.subplots(figsize=(10, 8))
            draw_frame(data, header, ax)
            
            # ★★★ 修正点 1: 余白を調整し、画像サイズを固定する ★★★
            plt.tight_layout()
            
            buf = io.BytesIO()
            # 'bbox_inches'を削除して、figsizeとdpiからサイズが決定されるようにする
            fig.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            
            frames.append(imageio.imread(buf))
            
            buf.close()
            plt.close(fig)

        except Exception as e:
            print(f"\nファイル {os.path.basename(file_path)} のフレーム生成中にエラーが発生しました: {e}")
            if 'fig' in locals() and plt.fignum_exists(fig.number):
                plt.close(fig)
            continue

    # 4. フレームリストから動画ファイルを生成
    if not frames:
        print("有効なフレームが生成されなかったため、動画は作成されませんでした。")
        return

    print(f"\n全{len(frames)}フレームをレンダリングしました。動画に変換します...")
    try:
        # ★★★ 修正点 2: macro_block_size=1 を追加 ★★★
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8, macro_block_size=1)
        print(f"\n動画ファイル '{output_path}' が正常に作成されました。")
    except Exception as e:
        print(f"\n動画の保存中にエラーが発生しました: {e}")
        print("imageioまたはimageio-ffmpegライブラリが正しくインストールされているか確認してください。")


def plot_mk4_single_frame(file_path: str, save_output: bool = False, output_dir: str = None):
    """
    MK4の単一フレームをプロットする関数
    
    Parameters:
    -----------
    file_path : str
        MK4 FITSファイルのパス
    save_output : bool
        画像を保存するかどうか
    output_dir : str
        出力ディレクトリ
    """
    try:
        data, header = read_fits_file(file_path)
        fig, ax = plt.subplots(figsize=(8, 8))
        draw_frame(data, header, ax)
        
        if save_output and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(file_path).replace('.fts', '.png')
            output_path = os.path.join(output_dir, filename)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"画像を保存しました: {output_path}")
        
        plt.show()
        return fig, ax
        
    except Exception as e:
        print(f"ファイル {file_path} の処理中にエラー: {e}")
        return None, None