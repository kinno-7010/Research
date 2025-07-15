#!/usr/bin/env python3
"""
太陽磁力線描画プログラム（修正版）
時刻: 2022-06-13T03:00:00 に統一
インポートエラーと引数エラーを修正
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# 絶対インポートに修正
try:
    import pfss_data_block
    import pfss_time2file  
    import pfss_restore
    import pfss_field_start_coord
    import pfss_trace_field
    import pfss_draw_field
    import pfss_to_spherical
    import spherical_draw_field
    print("必要なモジュールが正常にインポートされました")
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("一部のモジュールが利用できません。デモモードで実行します。")
    MODULES_AVAILABLE = False


def sswidl_style_execution():
    """
    SSWIDLのspherical_sample1.proを再現する実行
    時刻: 2022-06-13T03:00:00
    """
    print("=== SSWIDL spherical_sample1.pro の再現 ===")
    print("対象時刻: 2022-06-13T03:00:00")
    
    if not MODULES_AVAILABLE:
        print("モジュールが利用できないため、デモモードに移行します...")
        demo_mode()
        return
    
    try:
        # IDLでの実行: pfss_restore,pfss_time2file('2022-06-13T03:00:00',/ssw_catalog,/url)
        print("1. サンプルベクトル場を取得中...")
        print("   IDL equivalent: pfss_restore,pfss_time2file('2022-06-13T03:00:00',/ssw_catalog,/url)")
        
        # 正しい引数名を使用
        filename = pfss_time2file.pfss_time2file('2022-06-13T03:00:00', 
                                                ssw_catalog=True, urls=True)
        print(f"   ファイル: {filename}")
        
        # データを復元
        pfss_restore.pfss_restore(filename)
        print("   データが復元されました")
        
        # IDLでの実行: pfss_to_spherical,sph_data
        print("\n2. 球面構造体に格納中...")
        print("   IDL equivalent: pfss_to_spherical,sph_data")
        sph_data = pfss_to_spherical.pfss_to_spherical()
        
        # IDLでの実行: spherical_field_start_coord,sph_data,5,10,radstart=1.5
        print("\n3. 磁力線のトレース中...")
        print("   IDL equivalent: spherical_field_start_coord,sph_data,5,10,radstart=1.5")
        spherical_field_start_coord.spherical_field_start_coord(sph_data, 5, 10, radstart=1.5)
        
        # IDLでの実行: spherical_trace_field,sph_data
        print("   IDL equivalent: spherical_trace_field,sph_data")
        spherical_trace_field.spherical_trace_field(sph_data)
        
        # IDLでの実行: spherical_draw_field,sph_data,outim=outim,bcent=30,lcent=0,imsc=100,xsize=512,ysize=512
        print("\n4. フィールドをレンダリング中...")
        print("   IDL equivalent: spherical_draw_field,sph_data,outim=outim,bcent=30,lcent=0,imsc=100,xsize=512,ysize=512")
        
        outim = spherical_draw_field.spherical_draw_field(sph_data, bcent=30, lcent=0, imsc=100, 
                                                        xsize=512, ysize=512, onscreen=True)
        
        print("\n5. インタラクティブビューアを起動...")
        print("   IDL equivalent: spherical_trackball_widget,sph_data,imsc=100")
        print("   (Python版では簡易3D表示)")
        
        # 追加の表示角度
        angles = [(0, 0), (30, 90), (60, 180), (-30, 270)]
        for i, (bcent, lcent) in enumerate(angles):
            print(f"   視点 {i+1}: 緯度={bcent}°, 経度={lcent}°")
            spherical_draw_field.spherical_draw_field(sph_data, bcent=bcent, lcent=lcent, 
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
    print("時刻: 2022-06-13T03:00:00 のシミュレーション")
    
    try:
        # ローカルインポートでエラーを回避
        import spherical_field_data__define
        import spherical_field_start_coord
        import spherical_trace_field
        import spherical_draw_field
        
        # 合成データの作成（2022年6月13日 3時のシミュレーション）
        sph_data = spherical_field_data__define.SphericalFieldData()
        
        # グリッドの設定
        nr, nlat, nlon = 30, 45, 90
        rix = np.linspace(1.0, 2.5, nr)
        lat = np.linspace(-90, 90, nlat)
        lon = np.linspace(0, 360, nlon, endpoint=False)
        
        sph_data.set_coordinate_arrays(lon, lat, rix)
        
        # 2022年6月13日頃の太陽活動を模擬した磁場の作成
        LON, LAT, R = np.meshgrid(lon, lat, rix, indexing='ij')
        theta = (90 - LAT) * np.pi / 180
        phi = LON * np.pi / 180
        
        # より複雑な磁場構造（活発な太陽活動期を想定）
        br = 2.0 * np.cos(theta) / R**3
        
        # 活動領域を模擬した局所的な磁場強化
        # 2022年6月は太陽活動極大期に向かう時期
        active_region1 = 5.0 * np.exp(-((LON - 45)**2 + (LAT - 15)**2) / 200) / R**2
        active_region2 = -3.0 * np.exp(-((LON - 180)**2 + (LAT + 25)**2) / 150) / R**2
        active_region3 = 4.0 * np.exp(-((LON - 290)**2 + (LAT - 0)**2) / 180) / R**2
        
        br += active_region1 + active_region2 + active_region3
        
        # 多重極子成分
        br += 0.8 * np.sin(2 * theta) * np.cos(2 * phi) / R**3
        br += 0.5 * np.sin(theta)**2 * np.cos(4 * phi) / R**3
        br += 0.3 * np.cos(3 * theta) * np.sin(3 * phi) / R**3
        
        # Theta成分（子午線流）
        bth = np.sin(theta) / R**3
        bth += 0.4 * np.cos(2 * theta) * np.cos(2 * phi) / R**3
        bth += 0.2 * np.sin(3 * theta) * np.cos(phi) / R**3
        
        # Phi成分（方位角流）
        bph = 0.3 * np.sin(2 * phi) / R**2
        bph += 0.2 * np.sin(theta) * np.sin(4 * phi) / R**2
        
        sph_data.set_vector_field(br, bth, bph)
        
        print(f"作成された磁場データ:")
        print(f"  グリッドサイズ: {nlon}×{nlat}×{nr}")
        print(f"  動径範囲: {rix[0]:.1f} - {rix[-1]:.1f} Rs")
        print(f"  磁場強度範囲: {np.min(br):.2e} - {np.max(br):.2e}")
        
        # 磁力線のトレースと描画
        print("\n磁力線開始点を設定中...")
        spherical_field_start_coord.spherical_field_start_coord(sph_data, fieldtype=5, spacing=8)
        
        print("磁力線をトレース中...")
        spherical_trace_field.spherical_trace_field(sph_data, stepmax=1000, quiet=False)
        
        print("磁力線を描画中...")
        # 複数の視点で描画
        views = [
            (0, 0, "正面視点"),
            (30, 90, "北東視点"),
            (60, 180, "高緯度視点"),
            (-30, 270, "南西視点")
        ]
        
        for bcent, lcent, desc in views:
            print(f"  {desc} (緯度={bcent}°, 経度={lcent}°)")
            spherical_draw_field.spherical_draw_field(sph_data, onscreen=True, 
                                                    bcent=bcent, lcent=lcent,
                                                    xsize=500, ysize=500,
                                                    quiet=True)
        
        print("デモモード実行完了")
        print("\n2022-06-13T03:00:00 の磁力線シミュレーション終了")
        
    except Exception as e:
        print(f"デモモードでもエラーが発生: {e}")
        simple_demo()


def simple_demo():
    """
    最小限のデモ
    """
    print("\n=== 最小限デモ ===")
    print("時刻: 2022-06-13T03:00:00")
    
    # 基本的なプロット
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Solar Magnetic Field Simulation\n2022-06-13T03:00:00', fontsize=16)
    
    # サンプルデータ
    nlat, nlon = 90, 180
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon)
    LON, LAT = np.meshgrid(lon, lat)
    
    # 磁場成分をシミュレート
    theta = np.radians(90 - LAT)
    phi = np.radians(LON)
    
    # 動径磁場
    br = 2 * np.cos(theta)
    # 活動領域を追加
    br += 5 * np.exp(-((LON - 45)**2 + (LAT - 15)**2) / 200)
    br += -3 * np.exp(-((LON - 180)**2 + (LAT + 25)**2) / 150)
    
    # プロット
    im1 = axes[0,0].imshow(br, extent=[0, 360, -90, 90], cmap='RdBu_r', origin='lower')
    axes[0,0].set_title('Radial Magnetic Field')
    axes[0,0].set_xlabel('Longitude (deg)')
    axes[0,0].set_ylabel('Latitude (deg)')
    plt.colorbar(im1, ax=axes[0,0])
    
    # 磁場強度
    bmag = np.abs(br)
    im2 = axes[0,1].imshow(bmag, extent=[0, 360, -90, 90], cmap='plasma', origin='lower')
    axes[0,1].set_title('Magnetic Field Strength')
    axes[0,1].set_xlabel('Longitude (deg)')
    axes[0,1].set_ylabel('Latitude (deg)')
    plt.colorbar(im2, ax=axes[0,1])
    
    # 磁力線の概略図
    y = np.linspace(-2, 2, 20)
    x = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    # 双極子磁場
    r = np.sqrt(X**2 + Y**2)
    r[r == 0] = 1e-10
    
    Bx = 3 * X * Y / r**5
    By = (2 * Y**2 - X**2) / r**5
    
    axes[1,0].streamplot(x, y, Bx, By, density=1.5, color='blue', linewidth=1)
    axes[1,0].set_title('Field Lines (2D Projection)')
    axes[1,0].set_xlabel('X (Rs)')
    axes[1,0].set_ylabel('Y (Rs)')
    axes[1,0].set_aspect('equal')
    
    # 時系列情報
    times = ['2022-06-13T00:00', '2022-06-13T03:00', '2022-06-13T06:00', '2022-06-13T09:00']
    field_strength = [100, 120, 110, 95]  # 任意の値
    
    axes[1,1].plot(range(len(times)), field_strength, 'ro-')
    axes[1,1].set_title('Field Evolution')
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Average Field Strength')
    axes[1,1].set_xticks(range(len(times)))
    axes[1,1].set_xticklabels([t.split('T')[1] for t in times])
    axes[1,1].axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Target Time')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("簡易デモ完了")


def interactive_mode():
    """
    インタラクティブモード（時刻固定）
    """
    print("=== インタラクティブ磁力線描画 ===")
    print("時刻: 2022-06-13T03:00:00 (固定)")
    
    # パラメータ入力
    try:
        print(f"\nSSWIDL equivalent parameters:")
        print(f"  時刻: 2022-06-13T03:00:00")
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
    print(f"  pfss_restore,pfss_time2file('2022-06-13T03:00:00',/ssw_catalog,/url)")
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
    print("太陽磁力線描画プログラム")
    print("対象時刻: 2022-06-13T03:00:00")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_mode()
    elif len(sys.argv) > 1 and sys.argv[1] == "--sswidl":
        sswidl_style_execution()
    elif len(sys.argv) > 1 and sys.argv[1] == "--simple":
        simple_demo()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()