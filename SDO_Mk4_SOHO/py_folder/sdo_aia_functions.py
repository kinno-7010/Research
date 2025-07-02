"""
SDO/AIA関連の関数群
太陽観測衛星SDO/AIAデータの解析、可視化、動画作成を行う
"""

from config import *


def plot_sdo_aia(datetime_str, channel_str):
    """
    指定された日時と波長チャンネルのSDO/AIA画像をWCSベースでプロットし、
    軸の目盛りラベルのみを太陽中心を(0,0)とするピクセル単位で表示します。
    太陽リムとグリッドはWCSに基づいて描画されます。
    """
    # 1. 日時とファイルパスの処理
    try:
        dt_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
        date_fmtd_for_fname = dt_obj.strftime("%Y%m%d")
        time_fmtd_for_fname = dt_obj.strftime("%H%M")
    except ValueError:
        print(
            f"エラー: 日時文字列 '{datetime_str}' の形式が無効です。"
            " 'YYYY-MM-DD HH:MM' 形式で指定してください。"
        )
        return

    wavelength_part_in_fname = channel_str.zfill(4)
    filename = f"AIA{date_fmtd_for_fname}_{time_fmtd_for_fname}_{wavelength_part_in_fname}.fits"
    file_path = BASE_DATA_DIR / channel_str / filename

    print(f"ターゲットファイルパス: {file_path}")

    try:
        aia_map = sunpy.map.Map(file_path)
        print(f"ファイル '{file_path}' を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {file_path}")
        return
    except Exception as e:
        print(f"ファイルの読み込み・初期処理中にエラーが発生しました: {e}")
        return

    image_data = aia_map.data
    wcs_info = aia_map.wcs

    # 5. データ正規化
    vmin_percentile = 1.0
    vmax_percentile = 99.5
    stretch_power = 0.5
    valid_data = image_data[np.isfinite(image_data)]
    norm = None
    if valid_data.size > 0:
        norm = ImageNormalize(
            vmin=np.percentile(valid_data, vmin_percentile),
            vmax=np.percentile(valid_data, vmax_percentile),
            stretch=PowerStretch(stretch_power),
            clip=True
        )

    # 6. ピクセルスケールを取得 (フォーマッタ関数で使用)
    cdelt1 = aia_map.meta.get('cdelt1')
    cdelt2 = aia_map.meta.get('cdelt2')

    use_pixel_formatter = False
    pixel_scale_x = 1.0
    pixel_scale_y = 1.0

    if cdelt1 is not None and cdelt2 is not None and cdelt1 != 0 and cdelt2 != 0:
        pixel_scale_x = abs(cdelt1)
        pixel_scale_y = abs(cdelt2)

        if hasattr(wcs_info, 'wcs') and hasattr(wcs_info.wcs, 'cunit'):
            if wcs_info.wcs.cunit[0] == u.arcsec and wcs_info.wcs.cunit[1] == u.arcsec:
                use_pixel_formatter = True
            else:
                print(f"警告: WCSの単位がarcsecではありません ({wcs_info.wcs.cunit[0]}, {wcs_info.wcs.cunit[1]})。ピクセル目盛りは無効です。")
        elif hasattr(wcs_info, 'wcs') and not hasattr(wcs_info.wcs, 'cunit'):
            print("警告: WCS CUNIT情報が見つかりません。ピクセル目盛りは無効です。")
        else:
            print("警告: WCSの単位を確認できません。ピクセル目盛りは無効です。")
    else:
        print("警告: FITSヘッダーからCDELT1またはCDELT2が取得できないか0です。ピクセル目盛りは無効です。")

    # 7. 目盛りフォーマッタ関数を定義
    def arcsec_to_pixel_offset_formatter(arcsec_value, pos, scale_arcsec_per_pixel):
        if scale_arcsec_per_pixel == 0:
            return f"{0:.0f}"
        pixel_offset = arcsec_value / scale_arcsec_per_pixel
        return f"{pixel_offset:.0f}"

    # 8. プロットの準備 (WCSAxesを使用)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection=wcs_info)

    # 9. 画像のプロット
    im = ax.imshow(image_data, origin='lower', cmap=f'sdoaia{channel_str}', norm=norm)

    # 10. 太陽リムとグリッドの描画
    try:
        aia_map.draw_limb(axes=ax, color='white', linestyle='dashed', linewidth=1.2)
        aia_map.draw_grid(axes=ax, grid_spacing=15*u.deg, color='white', linestyle='dotted', linewidth=0.8, alpha=0.7)
        print("情報: 太陽リムとグリッドをWCSベースで描画しました。")
    except Exception as e_draw:
        print(f"警告: 太陽リムまたはグリッドの描画に失敗しました: {e_draw}")

    # 11. タイトルと軸ラベル、目盛りフォーマッタの設定
    title_str_parts = [
        f"SDO/AIA {int(channel_str)} Å",
        f"{dt_obj.strftime('%Y-%m-%d %H:%M:%S UT')}"
    ]
    if use_pixel_formatter:
        title_str_parts.append(f"Tick Labels in Pixels (Scale ≈ {pixel_scale_x:.2f}\" /pix)")

    ax.set_title("\n".join(title_str_parts), fontsize=12, pad=15)

    if use_pixel_formatter:
        ax.coords[0].set_major_formatter(FuncFormatter(lambda val, pos: arcsec_to_pixel_offset_formatter(val, pos, pixel_scale_x)))
        ax.coords[0].set_axislabel("Solar X (pixels from Sun center)")
        ax.coords[1].set_major_formatter(FuncFormatter(lambda val, pos: arcsec_to_pixel_offset_formatter(val, pos, pixel_scale_y)))
        ax.coords[1].set_axislabel("Solar Y (pixels from Sun center)")
    else:
        cunit1_fallback = wcs_info.wcs.cunit[0] if hasattr(wcs_info, 'wcs') and hasattr(wcs_info.wcs, 'cunit') else u.arcsec
        cunit2_fallback = wcs_info.wcs.cunit[1] if hasattr(wcs_info, 'wcs') and hasattr(wcs_info.wcs, 'cunit') else u.arcsec
        ax.coords[0].set_axislabel(f"Solar X ({cunit1_fallback})")
        ax.coords[1].set_axislabel(f"Solar Y ({cunit2_fallback})")

    plt.tight_layout()
    plt.show()


def normalize_log_stretch(data):
    """Applies LogStretch normalization to the data."""
    # LogStretchは負の値を扱えないため、ゼロや負の値を避けるためにクリッピングする
    data_clipped = np.maximum(data, 1e-5)
    normalizer = vis.ImageNormalize(data_clipped, stretch=vis.LogStretch(), clip=True)
    return normalizer(data_clipped)


def plot_sdo_aia_rgb(datetime_str,
                     channel_r_str="211",
                     channel_g_str="193",
                     channel_b_str="171"):
    """
    指定された日時のSDO/AIAデータ (3波長) を読み込み、RGB合成画像をWCSベースでプロットします。
    軸の目盛りラベルは太陽中心を(0,0)とするピクセル単位で表示します。
    """
    # 1. 日時文字列のパース
    try:
        dt_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
        date_fmtd_for_fname = dt_obj.strftime("%Y%m%d")
        time_fmtd_for_fname = dt_obj.strftime("%H%M")
    except ValueError:
        print(
            f"エラー: 日時文字列 '{datetime_str}' の形式が無効です。"
            " 'YYYY-MM-DD HH:MM' 形式で指定してください。"
        )
        return

    # 2. 各チャンネルのファイルパスを組み立て、Mapオブジェクトをロード
    maps = {}
    channels = {'r': channel_r_str, 'g': channel_g_str, 'b': channel_b_str}
    loaded_map_count = 0

    for color, ch_str in channels.items():
        wavelength_part_in_fname = ch_str.zfill(4)
        filename = f"AIA{date_fmtd_for_fname}_{time_fmtd_for_fname}_{wavelength_part_in_fname}.fits"
        file_path = BASE_DATA_DIR / ch_str / filename
        print(f"読み込み試行: {color.upper()}チャンネル ({ch_str}Å) - {file_path}")
        try:
            maps[color] = sunpy.map.Map(file_path)
            print(f"  成功: {ch_str}Å")
            loaded_map_count += 1
        except Exception as e:
            print(f"  失敗: {ch_str}Å のファイル読み込みエラー: {e}")
            maps[color] = None

    if loaded_map_count < 3:
        print("エラー: 3つ全ての波長チャンネルのデータを読み込めませんでした。プロットを中止します。")
        return

    # 基準となるMapオブジェクトを選択 (WCS情報、メタデータ、リム/グリッド描画に使用)
    reference_map = maps['b'] if maps['b'] else maps['g'] if maps['g'] else maps['r']
    if not reference_map:
        print("エラー: 基準となるMapオブジェクトがありません。")
        return

    wcs_info = reference_map.wcs

    # 3. 各チャンネルのデータを正規化
    try:
        red_channel_data = normalize_log_stretch(maps['r'].data)
        green_channel_data = normalize_log_stretch(maps['g'].data)
        blue_channel_data = normalize_log_stretch(maps['b'].data)
    except Exception as e_norm:
        print(f"データ正規化中にエラー: {e_norm}")
        return

    # 4. RGB画像の作成 (0-1にスケーリング)
    def scale_to_01(data):
        d_min = np.nanmin(data)
        d_max = np.nanmax(data)
        if d_max == d_min:
            return np.zeros_like(data)
        return (data - d_min) / (d_max - d_min)

    red_channel_final = scale_to_01(red_channel_data)
    green_channel_final = scale_to_01(green_channel_data)
    blue_channel_final = scale_to_01(blue_channel_data)
    rgb_image = np.stack([red_channel_final, green_channel_final, blue_channel_final], axis=-1)

    # 5. ピクセルスケールを取得
    cdelt1 = reference_map.meta.get('cdelt1')
    cdelt2 = reference_map.meta.get('cdelt2')
    use_pixel_formatter = False
    pixel_scale_x = 1.0
    pixel_scale_y = 1.0

    if cdelt1 is not None and cdelt2 is not None and cdelt1 != 0 and cdelt2 != 0:
        pixel_scale_x = abs(cdelt1)
        pixel_scale_y = abs(cdelt2)
        if hasattr(wcs_info, 'wcs') and hasattr(wcs_info.wcs, 'cunit'):
            if wcs_info.wcs.cunit[0] == u.arcsec and wcs_info.wcs.cunit[1] == u.arcsec:
                use_pixel_formatter = True
        else:
            print("警告: WCS単位を確認できません。ピクセル目盛りは無効の可能性があります。")
    else:
        print("警告: CDELTが取得不可または0です。ピクセル目盛りは無効です。")

    # 6. 目盛りフォーマッタ関数
    def arcsec_to_pixel_offset_formatter(arcsec_value, pos, scale_arcsec_per_pixel):
        if scale_arcsec_per_pixel == 0:
            return f"{0:.0f}"
        return f"{(arcsec_value / scale_arcsec_per_pixel):.0f}"

    # 7. プロット準備
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection=wcs_info)

    # 8. RGB画像のプロット
    ax.imshow(rgb_image, origin='lower', aspect='equal')

    # 9. 太陽リムとグリッドの描画
    try:
        reference_map.draw_limb(axes=ax, color='white', linestyle='dashed', linewidth=1.2)
        reference_map.draw_grid(axes=ax, grid_spacing=15*u.deg, color='white', linestyle='dotted', linewidth=0.8, alpha=0.7)
        print("情報: 太陽リムとグリッドをWCSベースで描画しました。")
    except Exception as e_draw:
        print(f"警告: 太陽リムまたはグリッドの描画に失敗しました: {e_draw}")

    # 10. タイトルと軸ラベルの設定
    title_str_parts = [
        f"SDO/AIA Composite: R={channel_r_str}Å, G={channel_g_str}Å, B={channel_b_str}Å",
        f"{reference_map.date.strftime('%Y-%m-%d %H:%M:%S UT')}"
    ]
    if use_pixel_formatter:
        title_str_parts.append(f"Tick Labels in Pixels (Ref. Scale ≈ {pixel_scale_x:.2f}\" /pix)")
    ax.set_title("\n".join(title_str_parts), fontsize=12, pad=15)

    if use_pixel_formatter:
        ax.coords[0].set_major_formatter(FuncFormatter(lambda val, pos: arcsec_to_pixel_offset_formatter(val, pos, pixel_scale_x)))
        ax.coords[0].set_axislabel("Solar X (pixels from Sun center, ref. WCS)")
        ax.coords[1].set_major_formatter(FuncFormatter(lambda val, pos: arcsec_to_pixel_offset_formatter(val, pos, pixel_scale_y)))
        ax.coords[1].set_axislabel("Solar Y (pixels from Sun center, ref. WCS)")
    else:
        cunit1 = wcs_info.wcs.cunit[0] if hasattr(wcs_info, 'wcs') and hasattr(wcs_info.wcs, 'cunit') else u.arcsec
        cunit2 = wcs_info.wcs.cunit[1] if hasattr(wcs_info, 'wcs') and hasattr(wcs_info.wcs, 'cunit') else u.arcsec
        ax.coords[0].set_axislabel(f"Solar X ({cunit1})")
        ax.coords[1].set_axislabel(f"Solar Y ({cunit2})")

    ax.tick_params(axis='both', which='major', labelsize=10, direction='in')

    plt.tight_layout()
    plt.show()


def find_files_in_time_range(start_time: str, end_time: str, time_tolerance_seconds: int = 12) -> defaultdict:
    """
    指定された時間範囲にあるFITSファイルを検索し、時刻をグループ化して返す。
    時刻のわずかなズレを許容するため、指定された秒数で時刻を丸める（タイムビン）。
    """
    print(f"ディレクトリ '{BASE_DATA_DIR}' 内の.fits/.ftsファイルを再帰的に検索しています...")
    all_files = sorted(BASE_DATA_DIR.rglob('*.fits')) + sorted(BASE_DATA_DIR.rglob('*.fts'))
    all_files = sorted(list(set(all_files))) # 重複を削除

    if not all_files:
        print("警告: FITSファイルが一つも見つかりませんでした。")
        return defaultdict(list)
    else:
        print(f"{len(all_files)}個のFITSファイルが見つかりました。")

    t_start = Time(start_time)
    t_end = Time(end_time)
    files_by_time = defaultdict(list)

    print(f"各ファイルを読み込み、約{time_tolerance_seconds}秒間の時間幅でグループ化しています...")
    for f in tqdm(all_files, desc="ファイルフィルタリング"):
        try:
            m = sunpy.map.Map(f)
            file_time = m.date
            
            if t_start <= file_time <= t_end:
                # --- ★★★ 新しいグループ化ロジック ★★★ ---
                dt_obj = file_time.to_datetime()
                
                # UNIXタイムスタンプ（秒）に変換し、指定秒数で丸める
                total_seconds = dt_obj.timestamp()
                binned_seconds = round(total_seconds / time_tolerance_seconds) * time_tolerance_seconds
                
                # 丸めた秒数から、グループ化のキーとなるdatetimeオブジェクトを再構築
                time_key = datetime.fromtimestamp(binned_seconds)
                
                files_by_time[time_key].append(f)
                
        except Exception as e:
            print(f"ファイル {f.name} の読み込み/解析中にエラー: {e}")
            continue

    return files_by_time


def create_single_aia_movie(channel_str: str, start_time: str, end_time: str, output_path: Path, fps: int, pmin: float=1.0, pmax: float=99.5):
    """
    指定された単一波長(チャンネル)と時間範囲のデータから動画を作成する。
    """
    # 1. 時間範囲内のファイルを時刻ごとにグループ化
    #    単一波長なので時間許容度は小さくても良いが、汎用的に60秒のままにする
    files_by_time = find_files_in_time_range(start_time, end_time, time_tolerance_seconds=60)

    if not files_by_time:
        print("指定された時間範囲に該当するファイルのグループが見つかりませんでした。")
        return

    frames = []
    sorted_times = sorted(files_by_time.keys())
    print(f"\n波長 {channel_str}Å の動画を作成します...")
    print(f"{len(sorted_times)}個のタイムスタンプグループからフレームを生成します...")

    # 2. 各時刻グループのファイルから目的の波長の画像を生成
    for time_key in tqdm(sorted_times, desc=f"フレーム生成 ({channel_str}Å)"):
        file_list = files_by_time[time_key]
        
        target_map = None
        # グループ内のファイルを調べて目的の波長のものを探す
        for f in file_list:
            try:
                m = sunpy.map.Map(f)
                # ファイルの波長が目的の波長と一致するか確認
                wavelength = str(int(m.wavelength.to_value(u.Angstrom)))
                if wavelength == channel_str:
                    target_map = m
                    break  # 目的の波長が見つかったら、このグループの探索は終了
            except Exception:
                # FITSファイルではない、または壊れている可能性を無視して次に進む
                continue
        
        # 目的の波長のマップが見つかった場合のみフレームを生成
        if target_map:
            try:
                # プロット処理
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(projection=target_map)
                #――――――――――――――――――――――――
                # ここから変更点：パーセンタイルで vmin/vmax を計算
                #――――――――――――――――――――――――
                data = target_map.data.astype(float)
                # NaN を除いて flatten
                valid = data[np.isfinite(data)]
                if valid.size > 0:
                    vmin_val = np.nanpercentile(valid, pmin)
                    vmax_val = np.nanpercentile(valid, pmax)
                else:
                    # データがすべて NaN/無効なら、デフォルトで 0–1 をセット
                    vmin_val = 0.0
                    vmax_val = 1.0

                # SunPy が推奨する AIA カラーマップを取得
                cmap = (target_map.plot_settings.get('cmap')
                        if 'cmap' in target_map.plot_settings
                        else f'sdoaia{channel_str}')

                im = ax.imshow(
                    data,
                    origin='lower',
                    cmap=cmap,
                    vmin=vmin_val,
                    vmax=vmax_val
                )
                #――――――――――――――――――――――――
                target_map.draw_limb(axes=ax, color='white', linestyle='solid')

                # カスタムタイトルを設定
                ax.set_title(
                    f"SDO/AIA {channel_str} Å\n{target_map.date.strftime('%Y-%m-%d %H:%M:%S UT')}",
                    fontsize=12
                )
                ax.set_xlabel("Solar X (arcsec)"); ax.set_ylabel("Solar Y (arcsec)")
                
                # バッファにプロットを保存
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150)
                buf.seek(0)
                frames.append(imageio.imread(buf))
                plt.close(fig)
            except Exception as e:
                print(f"タイムスタンプ {time_key} のプロット中にエラー: {e}")
                plt.close('all')

    # 動画ファイルとして保存
    if not frames:
        print(f"波長 {channel_str}Å の有効なフレームが生成されなかったため、動画は作成されませんでした。")
        return

    print(f"\n全{len(frames)}フレームをレンダリングしました。動画に変換します...")
    try:
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
        print(f"\n動画ファイル '{output_path}' が正常に作成されました。")
    except Exception as e:
        print(f"\n動画の保存中にエラーが発生しました: {e}")


def create_aia_rgb_movie(start_time: str, end_time: str, output_path: Path, fps: int):
    """
    指定された時間範囲のデータからRGB合成動画を作成する。
    """
    files_by_time = find_files_in_time_range(start_time, end_time, time_tolerance_seconds=60)

    if not files_by_time:
        print("指定された時間範囲に該当するファイルのグループが見つかりませんでした。")
        return

    frames = []
    sorted_times = sorted(files_by_time.keys())
    print(f"\n見つかった{len(sorted_times)}個のタイムスタンプグループからフレームを生成します...")

    # ★★★ バグ修正：デバッグ表示用のカウンターを準備 ★★★
    debug_prints = 0

    # 各時刻グループのファイルからRGB画像を生成
    for time_key in tqdm(sorted_times, desc="フレーム生成"):
        file_list = files_by_time[time_key]
        maps = {}
        for f in file_list:
            try:
                m = sunpy.map.Map(f)
                wavelength = str(int(m.wavelength.to_value(u.Angstrom)))
                if wavelength in ["211", "193", "171"]:
                    maps[wavelength] = m
            except Exception as e:
                print(f"ファイル {f.name} の読み込み中にエラー: {e}")

        # 3波長が揃っているか確認
        required_channels = {"211", "193", "171"}
        available_channels = set(maps.keys())
        
        if required_channels.issubset(available_channels):
            # 3波長揃っている場合の処理（変更なし）
            try:
                ref_map = maps["171"]
                # ...（以降の画像生成処理は省略）...
                # （この部分は元のコードのままで問題ありません）
                wcs_info = ref_map.wcs
                red_data = normalize_log_stretch(maps['211'].data)
                green_data = normalize_log_stretch(maps['193'].data)
                blue_data = normalize_log_stretch(maps['171'].data)
                
                def scale_to_01(data):
                    d_min, d_max = np.nanmin(data), np.nanmax(data)
                    return (data - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(data)

                rgb_image = np.stack([
                    scale_to_01(red_data), scale_to_01(green_data), scale_to_01(blue_data)
                ], axis=-1)

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(projection=wcs_info)
                ax.imshow(rgb_image, origin='lower', aspect='equal')
                ref_map.draw_limb(axes=ax, color='white', linestyle='solid', linewidth=1.5)
                ax.set_title(
                    f"SDO/AIA Composite (211/193/171 Å)\n{time_key.strftime('%Y-%m-%d %H:%M:%S UT')}",
                    fontsize=12
                )
                ax.set_xlabel("Solar X (arcsec)")
                ax.set_ylabel("Solar Y (arcsec)")
                ax.tick_params(axis='both', which='major', labelsize=10, direction='in')
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150)
                buf.seek(0)
                frames.append(imageio.imread(buf))
                plt.close(fig)

            except Exception as e:
                print(f"タイムスタンプ {time_key} のフレーム生成中にエラー: {e}")
                plt.close('all')
        else:
            # ★★★ バグ修正：シンプルなカウンターでデバッグ表示を制御 ★★★
            if debug_prints < 5: # 最初の5回だけ表示
                missing = required_channels - available_channels
                print(f"タイムスタンプ {time_key} ではフレームをスキップしました。利用可能: {available_channels}, 不足波長: {missing}")
                debug_prints += 1

    # 動画保存部分は変更ありません
    if not frames:
        print("有効なフレームが生成されなかったため、動画は作成されませんでした。")
        print("ヒント: find_files_in_time_rangeのtime_tolerance_secondsの値を調整してみてください。")
        return

    print(f"\n全{len(frames)}フレームをレンダリングしました。動画に変換します...")
    try:
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8, macro_block_size=None)
        print(f"\n動画ファイル '{output_path}' が正常に作成されました。")
    except Exception as e:
        print(f"\n動画の保存中にエラーが発生しました: {e}")
        
        
def create_aia_rgb_diff_movie(start_time: str, end_time: str, output_path: Path, fps: int):
    """
    指定された時間範囲のデータから「RGB隣接差分画像」の動画を作成する。
    データ欠損に強いロジックに修正。
    """
    # 内部で使うヘルパー関数（変更なし）
    def _generate_rgb_image(file_list):
        maps = {}
        for f in file_list:
            m = sunpy.map.Map(f)
            wavelength = str(int(m.wavelength.to_value(u.Angstrom)))
            if wavelength in ["211", "193", "171"]:
                maps[wavelength] = m
        
        if not all(ch in maps for ch in ["211", "193", "171"]):
            return None, None

        red_data = normalize_log_stretch(maps['211'].data)
        green_data = normalize_log_stretch(maps['193'].data)
        blue_data = normalize_log_stretch(maps['171'].data)
        
        def scale_to_01(data):
            d_min, d_max = np.nanmin(data), np.nanmax(data)
            return (data - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(data)

        rgb_image = np.stack([
            scale_to_01(red_data), scale_to_01(green_data), scale_to_01(blue_data)
        ], axis=-1)
        
        return rgb_image, maps["171"]

    # 1. 時間範囲内のファイルを時刻ごとにグループ化
    files_by_time = find_files_in_time_range(start_time, end_time, time_tolerance_seconds=60)
    if not files_by_time or len(files_by_time) < 2:
        print("差分を作成するために必要な、2つ以上のタイムスタンプグループが見つかりませんでした。")
        return
    sorted_times = sorted(files_by_time.keys())
    
    # ★★★ 修正点1: まず、有効なフレームだけをすべてリストアップする ★★★
    print("有効なフレーム（3波長揃ったグループ）を事前にスキャンしています...")
    valid_frames_data = []
    for time_key in tqdm(sorted_times, desc="有効フレームのスキャン"):
        file_list = files_by_time[time_key]
        rgb_image, ref_map = _generate_rgb_image(file_list)
        if rgb_image is not None:
            valid_frames_data.append({
                "time": time_key,
                "rgb": rgb_image,
                "map": ref_map
            })

    if len(valid_frames_data) < 2:
        print("差分を作成するために必要な、2つ以上の有効なフレームが見つかりませんでした。")
        return

    frames = []
    print(f"\n{len(valid_frames_data)}個の有効なフレームから差分フレームを生成します...")

    # ★★★ 修正点2: 「有効なフレームのリスト」を使ってペアを作る ★★★
    for i in tqdm(range(len(valid_frames_data) - 1), desc="差分フレーム生成"):
        data_i = valid_frames_data[i]
        data_i_plus_1 = valid_frames_data[i+1]
        
        time_i, rgb_i = data_i["time"], data_i["rgb"]
        time_i_plus_1, rgb_i_plus_1, ref_map = data_i_plus_1["time"], data_i_plus_1["rgb"], data_i_plus_1["map"]

        try:
            diff_image = rgb_i_plus_1 - rgb_i
            scaled_diff = (diff_image + 1) / 2.0

            # プロット処理
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(projection=ref_map.wcs)
            ax.imshow(scaled_diff, origin='lower', aspect='equal')
            
            # ★★★ 修正点3: 太陽リム（白円）を描画 ★★★
            ref_map.draw_limb(axes=ax, color='white', linestyle='solid', linewidth=1.5)
            
            title = (
                f"SDO/AIA RGB Running Difference\n"
                f"{time_i.strftime('%H:%M:%S')} to {time_i_plus_1.strftime('%H:%M:%S UT on %Y-%m-%d')}"
            )
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Solar X (arcsec)")
            ax.set_ylabel("Solar Y (arcsec)")
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close(fig)

        except Exception as e:
            print(f"タイムスタンプ {time_i} と {time_i_plus_1} の処理中にエラー: {e}")
            plt.close('all')

    # 動画保存（変更なし）
    if not frames:
        print("有効な差分フレームが生成されなかったため、動画は作成されませんでした。")
        return

    print(f"\n全{len(frames)}フレームをレンダリングしました。動画に変換します...")
    try:
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8, macro_block_size=None)
        print(f"\n動画ファイル '{output_path}' が正常に作成されました。")
    except Exception as e:
        print(f"\n動画の保存中にエラーが発生しました: {e}")

        print(f"\n動画の保存中にエラーが発生しました: {e}")        


def movie_mode(MODE: str, START_TIME: str, END_TIME: str=None, FRAME_RATE: int=5, WAVELENGTH_STR: str=None):
    """
    AIAデータから動画や単一プロットを作成するメイン関数
    """
    # MODE: 'base_diff_movie', 'rgb_diff_movie', 'rgb_movie', 'single_movie', 'rgb_single_plot'
    if MODE == 'rgb_diff_movie':
        output_filename_str = (
            f"aia_rgb_diff_{Time(START_TIME).strftime('%Y%m%d_%H%M%S')}-"
            f"{Time(END_TIME).strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        OUTPUT_PATH = OUTPUT_DIR / output_filename_str
        create_aia_rgb_diff_movie(START_TIME, END_TIME, OUTPUT_PATH, FRAME_RATE)

    elif MODE == 'rgb_movie':
        output_filename_str = (
            f"aia_rgb_{Time(START_TIME).strftime('%Y%m%d_%H%M%S')}-"
            f"{Time(END_TIME).strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        OUTPUT_PATH = OUTPUT_DIR / output_filename_str
        create_aia_rgb_movie(START_TIME, END_TIME, OUTPUT_PATH, FRAME_RATE)
    
    
    elif MODE == 'single_movie':
        output_filename_str = (
            f"aia_single_{WAVELENGTH_STR}_{Time(START_TIME).strftime('%Y%m%d_%H%M%S')}-"
            f"{Time(END_TIME).strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        OUTPUT_PATH = OUTPUT_DIR / output_filename_str
        
        # ★★★ 修正箇所: 引数の順序を修正 ★★★
        create_single_aia_movie(WAVELENGTH_STR, START_TIME, END_TIME, OUTPUT_PATH, FRAME_RATE)

    elif MODE == 'rgb_single_plot':
        # --- 単一画像プロットの例 ---
        plot_sdo_aia_rgb(datetime_str=START_TIME, channel_r_str="211",
                         channel_g_str="193", channel_b_str="171")
        
    elif MODE == 'single_plot':
        plot_sdo_aia(START_TIME, WAVELENGTH_STR)
        
    else:
        print(f"無効なモードです: {MODE}")