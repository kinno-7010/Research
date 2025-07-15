#!/usr/bin/env python3
"""
太陽磁力線描画プログラム（修正版）
SSWIDLの動作を再現するPythonスクリプト
"""

import numpy as np
import sys
import os

# 正しいクラス名でインポート
try:
    from pfss_data_block import PFSSDataBlock  # 大文字のクラス名
    from pfss_time2file import pfss_time2file
    from pfss_restore import pfss_restore
    from pfss_field_start_coord import pfss_field_start_coord
    from pfss_trace_field import pfss_trace_field
    from pfss_draw_field import pfss_draw_field
    from pfss_to_spherical import pfss_to_spherical
    from spherical_draw_field import spherical_draw_field
    print("必要なモジュールが正常にインポートされました")
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("一部のモジュールが利用できません。デモモードで実行します。")


def sswidl_style_execution():
    """
    SSWIDLのspheirical_sample1.proを再現する実行
    """
    print("=== SSWIDL spherical_sample1.pro の再現 ===")
    
    try:
        # IDLでの実行: pfss_restore,pfss_time2file('2003-04-05',/ssw_cat,/url)
        print("1. サンプルベクトル場を取得中...")
        print("   IDL equivalent: pfss_restore,pfss_time2file('2003-04-05',/ssw_cat,/url)")
        
        filename = pfss_time2file('2003-04-05', ssw_cat=True, urls=True)
        print(f"   ファイル: {filename}")
        
        # データを復元
        pfss_restore(filename)
        print("   データが復元されました")
        
        # IDLでの実行: pfss_to_spherical,sph_data
        print("\n2. 球面構造体に格納中...")
        print("   IDL equivalent: pfss_to_spherical,sph_data")
        sph_data = pfss_to_spherical()
        
        # IDLでの実行: spherical_field_start_coord,sph_data,5,10,radstart=1.5
        print("\n3. 磁力線のトレース中...")
        print("   IDL equivalent: spherical_field_start_coord,sph_data,5,10,radstart=1.5")
        spherical_field_start_coord(sph_data, 5, 10, radstart=1.5)
        
        # IDLでの実行: spherical_trace_field,sph_data
        print("   IDL equivalent: spherical_trace_field,sph_data")
        spherical_trace_field(sph_data)
        
        # IDLでの実行: spherical_draw_field,sph_data,outim=outim,bcent=30,lcent=0,imsc=100,xsize=512,ysize=512
        print("\n4. フィールドをレンダリング中...")
        print("   IDL equivalent: spherical_draw_field,sph_data,outim=outim,bcent=30,lcent=0,imsc=100,xsize=512,ysize=512")
        
        outim = spherical_draw_field(sph_data, bcent=30, lcent=0, imsc=100, 
                                   xsize=512, ysize=512, onscreen=True)
        
        print("\n5. インタラクティブビューアを起動...")
        print("   IDL equivalent: spherical_trackball_widget,sph_data,imsc=100")
        print("   (Python版では簡易3D表示)")
        
        # 追加の表示角度
        angles = [(0, 0), (30, 90), (60, 180), (-30, 270)]
        for i, (bcent, lcent) in enumerate(angles):
            print(f"   視点 {i+1}: 緯度={bcent}°, 経度={lcent}°")
            spherical_draw_field(sph_data, bcent=bcent, lcent=lcent, 
                                imsc=100, xsize=400, ysize=400, onscreen=True)
        
        print("\n=== 実行完了 ===")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("代替としてサンプルデータで実行します...")
        demo_mode()


def demo_mode():
    """
    デモモード：合成データを使用
    """
    print("\n=== デモモード（合成データ使用） ===")
    
    try:
        from spherical_field_data__define import SphericalFieldData
        from spherical_field_start_coord import spherical_field_start_coord
        from spherical_trace_field import spherical_trace_field
        from spherical_draw_field import spherical_draw_field
        
        # 合成データの作成
        sph_data = SphericalFieldData()
        
        # グリッドの設定
        nr, nlat, nlon = 30, 45, 90
        rix = np.linspace(1.0, 2.5, nr)
        lat = np.linspace(-90, 90, nlat)
        lon = np.linspace(0, 360, nlon, endpoint=False)
        
        sph_data.set_coordinate_arrays(lon, lat, rix)
        
        # 双極子磁場の作成
        LON, LAT, R = np.meshgrid(lon, lat, rix, indexing='ij')
        theta = (90 - LAT) * np.pi / 180
        phi = LON * np.pi / 180
        
        # 磁場成分
        br = 2.0 * np.cos(theta) / R**3
        br += 0.5 * np.sin(2 * theta) * np.cos(2 * phi) / R**3
        
        bth = np.sin(theta) / R**3
        bth += 0.3 * np.cos(2 * theta) * np.cos(2 * phi) / R**3
        
        bph = 0.2 * np.sin(2 * phi) / R**2
        
        sph_data.set_vector_field(br, bth, bph)
        
        # 磁力線のトレースと描画
        spherical_field_start_coord(sph_data, fieldtype=5, spacing=10)
        spherical_trace_field(sph_data, stepmax=1000)
        spherical_draw_field(sph_data, onscreen=True, bcent=30, lcent=0)
        
        print("デモモード実行完了")
        
    except Exception as e:
        print(f"デモモードでもエラーが発生: {e}")


def interactive_mode():
    """
    インタラクティブモード
    """
    print("=== インタラクティブ磁力線描画 ===")
    
    # 日時入力
    date_time = input("日時を入力 (YYYY-MM-DD) [2003-04-05]: ").strip()
    if not date_time:
        date_time = '2003-04-05'
    
    # パラメータ入力
    try:
        print(f"\nSSWIDL equivalent parameters:")
        print(f"  fieldtype (フィールドタイプ): 5 (固定)")
        
        invdens = int(input("磁力線密度逆数 [10]: ") or "10")
        print(f"  IDL: spherical_field_start_coord, sph_data, 5, {invdens}")
        
        radstart = float(input("開始半径 (Rs) [1.5]: ") or "1.5")
        print(f"  IDL: radstart={radstart}")
        
        bcent = float(input("中心緯度 (度) [30]: ") or "30")
        lcent = float(input("中心経度 (度) [0]: ") or "0")
        print(f"  IDL: bcent={bcent}, lcent={lcent}")
        
        imsc = float(input("画像スケール [100]: ") or "100")
        print(f"  IDL: imsc={imsc}")
        
    except ValueError:
        print("デフォルト値を使用")
        invdens, radstart, bcent, lcent, imsc = 10, 1.5, 30, 0, 100
    
    print(f"\n実行するIDLコマンド:")
    print(f"  pfss_restore,pfss_time2file('{date_time}',/ssw_cat,/url)")
    print(f"  pfss_to_spherical,sph_data")
    print(f"  spherical_field_start_coord,sph_data,5,{invdens},radstart={radstart}")
    print(f"  spherical_trace_field,sph_data")
    print(f"  spherical_draw_field,sph_data,bcent={bcent},lcent={lcent},imsc={imsc}")
    
    # 実行
    try:
        sswidl_style_execution()
    except:
        demo_mode()


def main():
    """
    メイン関数
    """
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_mode()
    elif len(sys.argv) > 1 and sys.argv[1] == "--sswidl":
        sswidl_style_execution()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()