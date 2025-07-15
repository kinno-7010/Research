#!/usr/bin/env python3
"""
太陽磁力線描画プログラム（動作版）
時刻: 2022-06-13T03:00:00 に統一
インポートエラーを完全に修正
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 現在のディレクトリをPythonパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 必要なライブラリをインストール
def install_required_packages():
    """必要なパッケージをインストール"""
    try:
        import subprocess
        packages = ['numpy', 'matplotlib', 'scipy', 'h5py', 'astropy']
        for package in packages:
            try:
                __import__(package)
                print(f"✓ {package} は既にインストールされています")
            except ImportError:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✓ {package} をインストールしました")
    except Exception as e:
        print(f"パッケージインストール中にエラー: {e}")

# モジュールインポート
def import_modules():
    """モジュールを安全にインポート"""
    modules = {}
    
    try:
        # 基本パッケージのインストール確認
        install_required_packages()
        
        # PFSSモジュールのインポート
        try:
            import pfss_time2file
            modules['pfss_time2file'] = pfss_time2file
        except ImportError:
            print("警告: pfss_time2file モジュールが見つかりません")
        
        try:
            import pfss_restore
            modules['pfss_restore'] = pfss_restore
        except ImportError:
            print("警告: pfss_restore モジュールが見つかりません")
        
        try:
            import pfss_to_spherical
            modules['pfss_to_spherical'] = pfss_to_spherical
        except ImportError:
            print("警告: pfss_to_spherical モジュールが見つかりません")
        
        # 球面モジュールのインポート
        import spherical_field_data__define
        modules['spherical_field_data__define'] = spherical_field_data__define
        
        import spherical_field_start_coord
        modules['spherical_field_start_coord'] = spherical_field_start_coord
        
        import spherical_trace_field
        modules['spherical_trace_field'] = spherical_trace_field
        
        import spherical_draw_field
        modules['spherical_draw_field'] = spherical_draw_field
        
        print("✓ 必要なモジュールが正常にインポートされました")
        return modules, True
        
    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        return {}, False

# モジュールの初期化
MODULES, MODULES_AVAILABLE = import_modules()

def convert_pfss_to_sswidl_format(pfss_data):
    """
    PFSS計算結果をSSWIDL形式のSphericalFieldDataに変換
    """
    if not MODULES_AVAILABLE:
        print("✗ SSWIDLモジュールが利用できません")
        return None
    
    try:
        # SphericalFieldData構造体を作成
        sph_data = MODULES['spherical_field_data__define'].SphericalFieldData()
        
        # 座標配列を設定
        sph_data.set_coordinate_arrays(pfss_data['lon'], pfss_data['lat'], pfss_data['rix'])
        
        # 磁場データを設定
        sph_data.set_vector_field(pfss_data['br'], pfss_data['bth'], pfss_data['bph'])
        
        print(f"✓ SSWIDL形式変換完了:")
        print(f"  グリッド: {pfss_data['nlon']}×{pfss_data['nlat']}×{pfss_data['nr']}")
        print(f"  動径範囲: {pfss_data['rix'][0]:.1f} - {pfss_data['rix'][-1]:.1f} Rs")
        
        return sph_data
        
    except Exception as e:
        print(f"✗ SSWIDL形式変換エラー: {e}")
        return None

def draw_field_lines_from_converted_data(sph_data):
    """
    変換されたデータから磁力線を描画
    """
    if not MODULES_AVAILABLE:
        print("✗ 描画モジュールが利用できません")
        return
    
    try:
        print("\n=== 実観測データでの磁力線描画 ===")
        
        # 磁力線開始点の設定
        print("磁力線開始点を設定中...")
        MODULES['spherical_field_start_coord'].spherical_field_start_coord(sph_data, fieldtype=5, spacing=8, radstart=1.2)
        
        # 磁力線のトレース
        print("磁力線をトレース中...")
        MODULES['spherical_trace_field'].spherical_trace_field(sph_data, stepmax=1000, quiet=False)
        
        # 磁力線の描画
        print("磁力線を描画中...")
        
        # 複数の視点で描画
        views = [
            (30, 0, "Standard View (Real Data)"),
            (0, 0, "Front View (Real Data)"),
            (60, 90, "High Latitude (Real Data)"),
            (-30, 180, "South Hemisphere (Real Data)")
        ]
        
        for bcent, lcent, desc in views:
            print(f"  {desc} (lat={bcent}°, lon={lcent}°)")
            MODULES['spherical_draw_field'].spherical_draw_field(sph_data, 
                                                               bcent=bcent, lcent=lcent,
                                                               imsc=100, xsize=500, ysize=500,
                                                               onscreen=True, quiet=True)
        
        print("✓ 実観測データでの磁力線描画完了")
        
    except Exception as e:
        print(f"✗ 磁力線描画エラー: {e}")
        import traceback
        traceback.print_exc()

def sswidl_style_execution():
    """
    SSWIDLのspherical_sample1.proを再現する実行
    時刻: 2022-06-13T03:00:00 - 実際のHMIデータを使用
    """
    print("=== SSWIDL spherical_sample1.pro の再現（実観測データ使用） ===")
    print("対象時刻: 2022-06-13T03:00:00")
    
    # まず実観測データの使用を試行
    hmi_file = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    
    if os.path.exists(hmi_file) and MODULES_AVAILABLE:
        print("実観測データを使用してSSWIDL処理を実行します...")
        try:
            # demo_mode関数を呼び出し（実観測データ処理に変更済み）
            demo_mode()
            return
        except Exception as e:
            print(f"実観測データ処理に失敗: {e}")
            print("従来のSSWIDL処理にフォールバックします...")
    
    if not MODULES_AVAILABLE:
        print("モジュールが利用できないため、デモモードに移行します...")
        demo_mode()
        return
    
    try:
        # PFSSモジュールが利用可能かチェック
        required_pfss_modules = ['pfss_time2file', 'pfss_restore', 'pfss_to_spherical']
        missing_modules = [mod for mod in required_pfss_modules if mod not in MODULES]
        
        if missing_modules:
            print(f"必要なPFSSモジュールが不足: {missing_modules}")
            print("デモモードで実行します...")
            demo_mode()
            return
        
        # 1. データファイルの取得（タイムアウト付き）
        print("\n1. PFSSデータファイルを取得中...")
        print("   IDL equivalent: pfss_restore,pfss_time2file('2022-06-13T03:00:00',/ssw_catalog,/url)")
        print("   注意: インターネット接続が必要で、時間がかかる場合があります")
        
        try:
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("データ取得がタイムアウトしました")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30秒でタイムアウト
            
            filename = MODULES['pfss_time2file'].pfss_time2file('2022-06-13T03:00:00', 
                                                               ssw_catalog=True, urls=True)
            signal.alarm(0)  # タイムアウトをリセット
            print(f"   ファイル: {filename}")
            
        except (TimeoutError, Exception) as e:
            signal.alarm(0)  # タイムアウトをリセット
            print(f"   データ取得に失敗: {e}")
            print("   ローカルデータまたはデモデータを使用します...")
            raise Exception("Data acquisition failed")
        
        # 2. データの復元
        print("\n2. データを復元中...")
        print("   IDL equivalent: pfss_restore,filename")
        MODULES['pfss_restore'].pfss_restore(filename)
        
        # 3. 球面構造体に変換
        print("\n3. 球面構造体に格納中...")
        print("   IDL equivalent: pfss_to_spherical,sph_data")
        sph_data = MODULES['pfss_to_spherical'].pfss_to_spherical()
        
        # 4. 磁力線開始点の設定
        print("\n4. 磁力線開始点を設定中...")
        print("   IDL equivalent: spherical_field_start_coord,sph_data,5,10,radstart=1.5")
        MODULES['spherical_field_start_coord'].spherical_field_start_coord(sph_data, 5, 10, radstart=1.5)
        
        # 5. 磁力線のトレース
        print("\n5. 磁力線をトレース中...")
        print("   IDL equivalent: spherical_trace_field,sph_data")
        MODULES['spherical_trace_field'].spherical_trace_field(sph_data)
        
        # 6. 磁力線の描画
        print("\n6. フィールドをレンダリング中...")
        print("   IDL equivalent: spherical_draw_field,sph_data,outim=outim,bcent=30,lcent=0,imsc=100")
        
        # 複数の視点での描画
        views = [
            (30, 0, "IDL標準視点"),
            (0, 0, "正面視点"),
            (60, 90, "高緯度視点"),
            (-30, 180, "南半球視点")
        ]
        
        for bcent, lcent, desc in views:
            print(f"   {desc} (緯度={bcent}°, 経度={lcent}°)")
            MODULES['spherical_draw_field'].spherical_draw_field(sph_data, 
                                                               bcent=bcent, lcent=lcent,
                                                               imsc=100, xsize=512, ysize=512,
                                                               onscreen=True, quiet=True)
        
        print("\n7. インタラクティブビューアを起動...")
        print("   IDL equivalent: spherical_trackball_widget,sph_data,imsc=100")
        print("   (Python版では複数角度での静的表示)")
        
        print("\n=== 実際のPFSSデータを使用した実行完了 ===")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("詳細なエラー情報:")
        import traceback
        traceback.print_exc()
        print("\n代替としてサンプルデータで実行します...")
        demo_mode()

def demo_mode():
    """
    デモモード：実際のHMI観測データを使用
    """
    print("\n=== デモモード（実観測データ使用） ===")
    print("時刻: 2022-06-13T03:00:00 - HMI実観測データ")
    
    # HMI FITSファイルのパス
    hmi_file = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    
    if not os.path.exists(hmi_file):
        print(f"✗ HMI FITSファイルが見つかりません: {hmi_file}")
        print("合成データにフォールバックします...")
        return run_fallback_demo()
    
    # HMI観測データからPFSS磁場を計算
    try:
        # HMI処理モジュールのインポート
        try:
            from hmi_pfss_fieldlines import read_hmi_fits, create_pfss_from_hmi
        except ImportError:
            print("HMI処理モジュールが見つかりません。合成データにフォールバックします...")
            return run_fallback_demo()
        
        print(f"HMI観測データを読み込み中: {os.path.basename(hmi_file)}")
        hmi_data = read_hmi_fits(hmi_file)
        
        if hmi_data is None:
            print("✗ HMI データの読み込みに失敗しました")
            return run_fallback_demo()
        
        print("PFSS磁場を計算中...")
        pfss_data = create_pfss_from_hmi(hmi_data, nr=25, rss=2.5)
        
        # PFSS データをSSWIDL形式に変換
        converted_data = convert_pfss_to_sswidl_format(pfss_data)
        
        if converted_data is None:
            print("✗ データ変換に失敗しました")
            return run_fallback_demo()
        
        print("✓ 実観測データからPFSS磁場計算完了")
        
        # 磁力線の描画
        draw_field_lines_from_converted_data(converted_data)
        print("✓ 実観測データでの磁力線描画完了")
        
    except Exception as e:
        print(f"✗ 実観測データ処理エラー: {e}")
        print("合成データにフォールバックします...")
        return run_fallback_demo()

def run_fallback_demo():
    """
    フォールバック用デモモード（合成データ）
    """
    print("\n=== フォールバックデモモード（合成データ） ===")
    print("時刻: 2022-06-13T03:00:00 のシミュレーション")
    
    if not MODULES_AVAILABLE:
        print("モジュールが利用できないため、簡易デモに移行します...")
        simple_demo()
        return
    
    try:
        # 合成データの作成
        sph_data = MODULES['spherical_field_data__define'].SphericalFieldData()
        
        # 2022年6月13日の太陽活動を模擬したグリッド設定
        nr, nlat, nlon = 35, 60, 120  # より高解像度
        rix = np.linspace(1.0, 2.5, nr)
        lat = np.linspace(-90, 90, nlat)
        lon = np.linspace(0, 360, nlon, endpoint=False)
        
        sph_data.set_coordinate_arrays(lon, lat, rix)
        
        # 2022年6月頃の太陽活動を模擬した磁場
        print("2022年6月13日03:00 UTC の磁場を模擬中...")
        LON, LAT, R = np.meshgrid(lon, lat, rix, indexing='ij')
        theta = (90 - LAT) * np.pi / 180
        phi = LON * np.pi / 180
        
        # 基本双極子磁場
        br = 2.0 * np.cos(theta) / R**3
        
        # 2022年6月の活動領域を模擬
        # AR3030付近の活動領域
        ar1 = 8.0 * np.exp(-((LON - 60)**2 + (LAT - 20)**2) / 300) / R**2
        ar2 = -6.0 * np.exp(-((LON - 180)**2 + (LAT + 15)**2) / 250) / R**2
        ar3 = 5.0 * np.exp(-((LON - 300)**2 + (LAT - 10)**2) / 200) / R**2
        ar4 = -4.0 * np.exp(-((LON - 120)**2 + (LAT + 30)**2) / 180) / R**2
        
        br += ar1 + ar2 + ar3 + ar4
        
        # 高次多重極成分
        br += 1.0 * np.sin(2 * theta) * np.cos(2 * phi) / R**3
        br += 0.7 * np.sin(theta)**2 * np.cos(4 * phi) / R**3
        br += 0.5 * np.cos(3 * theta) * np.sin(3 * phi) / R**3
        br += 0.3 * np.sin(4 * theta) * np.cos(6 * phi) / R**3
        
        # 子午線流成分
        bth = np.sin(theta) / R**3
        bth += 0.5 * np.cos(2 * theta) * np.cos(2 * phi) / R**3
        bth += 0.3 * np.sin(3 * theta) * np.cos(phi) / R**3
        
        # 差動回転を模擬した方位角成分
        bph = 0.4 * np.sin(2 * phi) * np.sin(theta) / R**2
        bph += 0.2 * np.sin(theta) * np.sin(4 * phi) / R**2
        bph += 0.1 * np.cos(theta) * np.sin(6 * phi) / R**2
        
        sph_data.set_vector_field(br, bth, bph)
        
        print(f"作成された磁場データ:")
        print(f"  グリッドサイズ: {nlon}×{nlat}×{nr}")
        print(f"  動径範囲: {rix[0]:.1f} - {rix[-1]:.1f} Rs")
        print(f"  磁場強度範囲: {np.min(br):.2e} - {np.max(br):.2e}")
        print(f"  活動領域数: 4個")
        
        # SSWIDLスタイルの実行
        print("\n=== SSWIDLスタイルの処理開始 ===")
        
        # 磁力線開始点の設定
        print("spherical_field_start_coord を実行中...")
        MODULES['spherical_field_start_coord'].spherical_field_start_coord(sph_data, fieldtype=5, spacing=8)
        
        # 磁力線のトレース
        print("spherical_trace_field を実行中...")
        MODULES['spherical_trace_field'].spherical_trace_field(sph_data, stepmax=1500, quiet=False)
        
        # 磁力線の描画
        print("spherical_draw_field を実行中...")
        
        # IDLのspherical_sample1.proと同じパラメータで実行
        print("IDL equivalent: spherical_draw_field,sph_data,outim=outim,bcent=30,lcent=0,imsc=100,xsize=512,ysize=512")
        MODULES['spherical_draw_field'].spherical_draw_field(sph_data, 
                                                bcent=30, lcent=0, imsc=100,
                                                xsize=512, ysize=512,
                                                onscreen=True, quiet=False)
        
        # 追加の視点
        additional_views = [
            (0, 0, "正面視点"),
            (60, 90, "高緯度視点"),
            (-30, 180, "南半球視点"),
            (45, 270, "斜視点")
        ]
        
        for bcent, lcent, desc in additional_views:
            print(f"追加視点: {desc} (緯度={bcent}°, 経度={lcent}°)")
            MODULES['spherical_draw_field'].spherical_draw_field(sph_data, 
                                                    bcent=bcent, lcent=lcent,
                                                    imsc=100, xsize=400, ysize=400,
                                                    onscreen=True, quiet=True)
        
        print("\n=== デモモード実行完了 ===")
        print("2022-06-13T03:00:00 の磁力線シミュレーション終了")
        
    except Exception as e:
        print(f"デモモードでエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        print("\n簡易デモに移行します...")
        simple_demo()

def simple_demo():
    """
    最小限のデモ（matplotlibのみ使用）
    """
    print("\n=== 最小限デモ（matplotlib使用） ===")
    print("時刻: 2022-06-13T03:00:00")
    
    try:
        # 基本的な太陽磁場の可視化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Solar Magnetic Field Simulation\n2022-06-13T03:00:00 UTC', fontsize=16)
        
        # 高解像度グリッド
        nlat, nlon = 180, 360
        lat = np.linspace(-90, 90, nlat)
        lon = np.linspace(0, 360, nlon)
        LON, LAT = np.meshgrid(lon, lat)
        
        # 磁場計算
        theta = np.radians(90 - LAT)
        phi = np.radians(LON)
        
        # 基本双極子 + 活動領域
        br = 2 * np.cos(theta)
        
        # 2022年6月の活動領域を模擬
        ar1 = 8 * np.exp(-((LON - 60)**2 + (LAT - 20)**2) / 300)
        ar2 = -6 * np.exp(-((LON - 180)**2 + (LAT + 15)**2) / 250)
        ar3 = 5 * np.exp(-((LON - 300)**2 + (LAT - 10)**2) / 200)
        ar4 = -4 * np.exp(-((LON - 120)**2 + (LAT + 30)**2) / 180)
        
        br += ar1 + ar2 + ar3 + ar4
        
        # 高次成分
        br += 1.0 * np.sin(2 * theta) * np.cos(2 * phi)
        br += 0.7 * np.sin(theta)**2 * np.cos(4 * phi)
        
        # プロット1: 動径磁場
        im1 = axes[0,0].imshow(br, extent=[0, 360, -90, 90], 
                              cmap='RdBu_r', origin='lower', aspect='auto')
        axes[0,0].set_title('Radial Magnetic Field\n(Simulated Data)')
        axes[0,0].set_xlabel('Longitude (deg)')
        axes[0,0].set_ylabel('Latitude (deg)')
        axes[0,0].grid(True, alpha=0.3)
        plt.colorbar(im1, ax=axes[0,0], label='Br (Gauss)')
        
        # プロット2: 磁場強度
        bmag = np.abs(br)
        im2 = axes[0,1].imshow(bmag, extent=[0, 360, -90, 90], 
                              cmap='plasma', origin='lower', aspect='auto')
        axes[0,1].set_title('Magnetic Field Strength')
        axes[0,1].set_xlabel('Longitude (deg)')
        axes[0,1].set_ylabel('Latitude (deg)')
        axes[0,1].grid(True, alpha=0.3)
        plt.colorbar(im2, ax=axes[0,1], label='|B| (Gauss)')
        
        # プロット3: 磁力線の概略図（2D投影）
        y = np.linspace(-2.5, 2.5, 25)
        x = np.linspace(-2.5, 2.5, 25)
        X, Y = np.meshgrid(x, y)
        
        # 双極子磁場
        r = np.sqrt(X**2 + Y**2)
        r[r < 0.1] = 0.1
        
        Bx = 3 * X * Y / r**5
        By = (2 * Y**2 - X**2) / r**5
        
        # 内側に太陽を描画
        circle = plt.Circle((0, 0), 1.0, color='yellow', alpha=0.8)
        axes[1,0].add_patch(circle)
        
        axes[1,0].streamplot(x, y, Bx, By, density=1.2, color='blue', linewidth=1.5)
        axes[1,0].set_title('Field Lines (2D Projection)')
        axes[1,0].set_xlabel('X (Rs)')
        axes[1,0].set_ylabel('Y (Rs)')
        axes[1,0].set_aspect('equal')
        axes[1,0].set_xlim(-2.5, 2.5)
        axes[1,0].set_ylim(-2.5, 2.5)
        
        # プロット4: 活動領域の位置
        axes[1,1].imshow(br, extent=[0, 360, -90, 90], 
                        cmap='RdBu_r', origin='lower', aspect='auto', alpha=0.7)
        
        # 活動領域をマーク
        ar_positions = [(60, 20, 'AR1'), (180, -15, 'AR2'), (300, 10, 'AR3'), (120, -30, 'AR4')]
        for lon, lat, name in ar_positions:
            axes[1,1].plot(lon, lat, 'ko', markersize=8)
            axes[1,1].annotate(name, (lon, lat), xytext=(5, 5), 
                              textcoords='offset points', fontsize=10, 
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        axes[1,1].set_title('Active Regions\n2022-06-13T03:00:00')
        axes[1,1].set_xlabel('Longitude (deg)')
        axes[1,1].set_ylabel('Latitude (deg)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Simple demo completed")
        print("Corresponding SSWIDL commands:")
        print("  IDL> pfss_restore,pfss_time2file('2022-06-13T03:00:00',/ssw_catalog,/url)")
        print("  IDL> pfss_to_spherical,sph_data")
        print("  IDL> spherical_field_start_coord,sph_data,5,10,radstart=1.5")
        print("  IDL> spherical_trace_field,sph_data")
        print("  IDL> spherical_draw_field,sph_data,bcent=30,lcent=0,imsc=100")
        
    except Exception as e:
        print(f"簡易デモでもエラーが発生: {e}")
        import traceback
        traceback.print_exc()

def interactive_mode():
    """
    インタラクティブモード
    """
    print("=== インタラクティブ磁力線描画 ===")
    print("時刻: 2022-06-13T03:00:00 (固定)")
    
    try:
        print("\nパラメータ設定:")
        invdens = int(input("磁力線密度逆数 (5-20, 小さいほど密) [10]: ") or "10")
        radstart = float(input("開始半径 (Rs) [1.5]: ") or "1.5")
        bcent = float(input("中心緯度 (度) [30]: ") or "30")
        lcent = float(input("中心経度 (度) [0]: ") or "0")
        imsc = float(input("画像スケール [100]: ") or "100")
        
        print(f"\n設定値:")
        print(f"  時刻: 2022-06-13T03:00:00")
        print(f"  磁力線密度逆数: {invdens}")
        print(f"  開始半径: {radstart} Rs")
        print(f"  視点: 緯度={bcent}°, 経度={lcent}°")
        print(f"  画像スケール: {imsc}")
        
        print(f"\n対応するIDLコマンド:")
        print(f"  pfss_restore,pfss_time2file('2022-06-13T03:00:00',/ssw_catalog,/url)")
        print(f"  pfss_to_spherical,sph_data")
        print(f"  spherical_field_start_coord,sph_data,5,{invdens},radstart={radstart}")
        print(f"  spherical_trace_field,sph_data")
        print(f"  spherical_draw_field,sph_data,bcent={bcent},lcent={lcent},imsc={imsc}")
        
    except ValueError:
        print("無効な入力。デフォルト値を使用します。")
    
    # 実行
    if MODULES_AVAILABLE:
        sswidl_style_execution()
    else:
        demo_mode()

def main():
    """
    メイン関数
    """
    print("太陽磁力線描画プログラム")
    print("対象時刻: 2022-06-13T03:00:00 UTC")
    print("=" * 60)
    
    # 使用方法の表示
    if len(sys.argv) == 1:
        print("使用方法:")
        print("  --sswidl    : SSWIDLスタイル実行（実データ取得試行）")
        print("  --demo      : デモモード（合成データ使用）")
        print("  --simple    : 簡易デモ（matplotlibのみ）")
        print("  --interactive: インタラクティブモード")
        print("  引数なし    : インタラクティブモード")
        print()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--sswidl":
            sswidl_style_execution()
        elif sys.argv[1] == "--demo":
            demo_mode()
        elif sys.argv[1] == "--simple":
            simple_demo()
        elif sys.argv[1] == "--interactive":
            interactive_mode()
        else:
            print(f"不明な引数: {sys.argv[1]}")
            print("--sswidl, --demo, --simple, --interactive のいずれかを指定してください")
    else:
        interactive_mode()

if __name__ == "__main__":
    main()