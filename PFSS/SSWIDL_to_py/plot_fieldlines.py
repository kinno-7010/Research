#!/usr/bin/env python3
import sys
import numpy as np
from pfss_time2file import pfss_time2file
from pfss_restore import pfss_restore
from pfss_field_start_coord import pfss_field_start_coord
from pfss_trace_field import pfss_trace_field
from pfss_draw_field import pfss_draw_field
from pfss_to_spherical import pfss_to_spherical
from spherical_draw_field import spherical_draw_field

def main():
    # コマンドライン引数から日時を取得
    if len(sys.argv) > 1:
        date_time = sys.argv[1]
    else:
        date_time = '2022-06-13T03:00:00'  # デフォルト

    print(f"Processing date: {date_time}")

    try:
        # 1. データファイルの取得
        filename = pfss_time2file(date_time, urls=True)
        print(f"Data file: {filename}")

        # 2. データの復元
        pfss_restore(filename)
        print("Data restored successfully")

        # 3. 磁力線の開始点設定
        pfss_field_start_coord(5, 10, radstart=1.5)  # fieldtype=5, invdens=10

        # 4. 磁力線のトレース
        pfss_trace_field()

        # 5. 磁力線の描画
        outim = pfss_draw_field(bcent=0, lcent=0, width=2.5, mag=2)

        # 6. 球面座標系に変換して3D表示
        sph_data = pfss_to_spherical()
        spherical_draw_field(sph_data, onscreen=True)

        print("Field line plotting completed!")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have internet connection for data download")

if __name__ == "__main__":
    main()