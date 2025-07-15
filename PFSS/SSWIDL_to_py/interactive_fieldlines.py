#!/usr/bin/env python3
import numpy as np
from pfss_time2file import pfss_time2file
from pfss_restore import pfss_restore
from pfss_field_start_coord import pfss_field_start_coord
from pfss_trace_field import pfss_trace_field
from pfss_draw_field import pfss_draw_field
from pfss_to_spherical import pfss_to_spherical
from spherical_draw_field import spherical_draw_field

def main():
    print("=== 太陽磁力線描画プログラム ===")

    # ユーザー入力
    date_time = input("日時を入力してください (YYYY-MM-DD形式) [2003-04-05]: ").strip()
    if not date_time:
        date_time = '2022-06-13T03:00:00'

    # パラメータ入力
    try:
        invdens = int(input("磁力線密度の逆数 (10=デフォルト, 小さいほど密): ") or "10")
        radstart = float(input("開始半径 (太陽半径単位, 1.5=デフォルト): ") or "1.5")
        bcent = float(input("中心緯度 (度, 0=デフォルト): ") or "0")
        lcent = float(input("中心経度 (度, 0=デフォルト): ") or "0")
        width = float(input("表示幅 (太陽半径単位, 2.5=デフォルト): ") or "2.5")
    except ValueError:
        print("無効な入力です。デフォルト値を使用します。")
        invdens, radstart, bcent, lcent, width = 10, 1.5, 0, 0, 2.5

    print(f"\n設定:")
    print(f"  日時: {date_time}")
    print(f"  磁力線密度逆数: {invdens}")
    print(f"  開始半径: {radstart} Rs")
    print(f"  視点: 緯度={bcent}°, 経度={lcent}°")
    print(f"  表示幅: {width} Rs")

    try:
        print("\n1. データファイルを取得中...")
        filename = pfss_time2file(date_time, urls=True)
        print(f"   ファイル: {filename}")

        print("2. データを復元中...")
        pfss_restore(filename)

        print("3. 磁力線開始点を設定中...")
        pfss_field_start_coord(5, invdens, radstart=radstart)

        print("4. 磁力線をトレース中...")
        pfss_trace_field()

        print("5. 2D画像を描画中...")
        outim = pfss_draw_field(bcent=bcent, lcent=lcent, width=width, mag=2)

        print("6. 3D表示を準備中...")
        sph_data = pfss_to_spherical()
        spherical_draw_field(sph_data, onscreen=True, bcent=bcent, lcent=lcent)

        print("完了！")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("インターネット接続またはデータの可用性を確認してください")

if __name__ == "__main__":
    main()