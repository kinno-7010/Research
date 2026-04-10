#!/usr/bin/env python3
"""
HMI FITSファイルからPFSS磁力線を描画
入力: hmi.M_720s.20220613_030000_TAI.fits
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 現在のディレクトリをPythonパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from astropy.io import fits
    import astropy.units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    print("astropy パッケージが必要です: pip install astropy")
    ASTROPY_AVAILABLE = False

# モジュールインポート
try:
    import spherical_field_data__define
    import spherical_field_start_coord
    import spherical_trace_field
    import spherical_draw_field
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"モジュールインポートエラー: {e}")
    MODULES_AVAILABLE = False

def read_hmi_fits(filepath):
    """
    HMI FITSファイルを読み込み
    
    Parameters:
    -----------
    filepath : str
        HMI FITSファイルのパス
        
    Returns:
    --------
    dict : 磁場データと座標情報
    """
    if not ASTROPY_AVAILABLE:
        return None
    
    print(f"HMI FITSファイルを読み込み中: {filepath}")
    
    try:
        with fits.open(filepath) as hdul:
            # HMIデータは通常HDU 1にある
            if len(hdul) > 1 and hdul[1].data is not None:
                data = hdul[1].data
                header = hdul[1].header
                print(f"  HDU 1からデータを読み込み")
            else:
                data = hdul[0].data
                header = hdul[0].header
                print(f"  HDU 0からデータを読み込み")
            
            print(f"  データ形状: {data.shape}")
            print(f"  観測時刻: {header.get('T_OBS', 'Unknown')}")
            print(f"  磁場範囲: {np.nanmin(data):.1f} ～ {np.nanmax(data):.1f} Gauss")
            
            # 座標情報を取得
            naxis1 = header.get('NAXIS1', data.shape[1])  # 経度方向
            naxis2 = header.get('NAXIS2', data.shape[0])  # 緯度方向
            
            # Carrington座標系での経度・緯度
            crval1 = header.get('CRVAL1', 180.0)  # 中心経度
            cdelt1 = header.get('CDELT1', 360.0/naxis1)  # 経度刻み
            crpix1 = header.get('CRPIX1', naxis1/2 + 0.5)
            
            crval2 = header.get('CRVAL2', 0.0)    # 中心緯度
            cdelt2 = header.get('CDELT2', 180.0/naxis2)  # 緯度刻み
            crpix2 = header.get('CRPIX2', naxis2/2 + 0.5)
            
            # 経度・緯度配列を計算
            lon = crval1 + (np.arange(naxis1) + 1 - crpix1) * cdelt1
            lat = crval2 + (np.arange(naxis2) + 1 - crpix2) * cdelt2
            
            # 経度を0-360度範囲に正規化
            lon = (lon + 360) % 360
            
            # 緯度を-90～90度範囲に制限
            lat = np.clip(lat, -90, 90)
            
            result = {
                'br_surface': data,  # 表面磁場（元のHMIデータ）
                'lon': lon,
                'lat': lat,
                'header': header,
                'time': header.get('T_OBS', '2022-06-13T03:00:00'),
                'filepath': filepath
            }
            
            print(f"  経度範囲: {lon[0]:.1f} ～ {lon[-1]:.1f} 度")
            print(f"  緯度範囲: {lat[0]:.1f} ～ {lat[-1]:.1f} 度")
            
            return result
            
    except Exception as e:
        print(f"FITSファイル読み込みエラー: {e}")
        return None

def create_pfss_from_hmi(hmi_data, nr=35, rss=2.5):
    """
    HMIデータからPFSS磁場を計算
    
    Parameters:
    -----------
    hmi_data : dict
        read_hmi_fits()の結果
    nr : int
        動径方向の格子点数
    rss : float
        ソース面の高度（太陽半径単位）
        
    Returns:
    --------
    dict : PFSS磁場データ
    """
    print(f"\nPFSS磁場を計算中 (ソース面: {rss} Rs, 動径点数: {nr})")
    
    # HMIデータのリサンプリング（計算効率のため）
    original_shape = hmi_data['br_surface'].shape
    
    # 適度な解像度にダウンサンプリング
    if original_shape[1] > 180:  # 経度方向
        nlon_new = 120
        nlat_new = 60
    else:
        nlon_new = original_shape[1]
        nlat_new = original_shape[0]
    
    print(f"  リサンプリング: {original_shape} → ({nlat_new}, {nlon_new})")
    
    # 新しい座標配列
    lon_new = np.linspace(0, 360, nlon_new, endpoint=False)
    lat_new = np.linspace(-90, 90, nlat_new)
    rix = np.linspace(1.0, rss, nr)
    
    # HMIデータを新しいグリッドに補間
    from scipy.interpolate import RectBivariateSpline
    
    # 元のデータでNaNを処理
    br_orig = hmi_data['br_surface'].copy()
    mask = np.isfinite(br_orig)
    if not np.all(mask):
        print(f"  NaN値を補間: {np.sum(~mask)} / {mask.size}")
        # 簡単な補間
        from scipy.ndimage import gaussian_filter
        br_orig[~mask] = 0
        br_orig = gaussian_filter(br_orig, sigma=1.0)
    
    # 補間関数を作成（新しいSciPy対応）
    try:
        from scipy.interpolate import RegularGridInterpolator
        
        # HMIデータの実際の形状を確認
        print(f"  HMIデータ形状: {br_orig.shape}")
        print(f"  座標配列長: lon={len(hmi_data['lon'])}, lat={len(hmi_data['lat'])}")
        
        # データの次元を調整
        if len(br_orig.shape) == 2:
            if br_orig.shape == (len(hmi_data['lat']), len(hmi_data['lon'])):
                # (lat, lon) の順序
                br_for_interp = br_orig
            elif br_orig.shape == (len(hmi_data['lon']), len(hmi_data['lat'])):
                # (lon, lat) の順序 - 転置が必要
                br_for_interp = br_orig.T
            else:
                print(f"  警告: データ形状が座標と一致しません。最近傍補間を使用します。")
                raise ValueError("Shape mismatch")
        else:
            print(f"  警告: データが2次元ではありません。最近傍補間を使用します。")
            raise ValueError("Not 2D data")
        
        # 座標を昇順にソート
        lat_sorted = np.sort(hmi_data['lat'])
        lon_sorted = np.sort(hmi_data['lon'])
        
        # データを対応する順序に並べ替え
        lat_indices = np.argsort(hmi_data['lat'])
        lon_indices = np.argsort(hmi_data['lon'])
        
        br_sorted = br_for_interp[np.ix_(lat_indices, lon_indices)]
        
        # RegularGridInterpolatorを使用
        interp_func = RegularGridInterpolator((lat_sorted, lon_sorted), br_sorted,
                                            method='linear', bounds_error=False, fill_value=0)
        
        # 新しいグリッドで補間
        lat_grid, lon_grid = np.meshgrid(lat_new, lon_new, indexing='ij')
        points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
        br_surface_new = interp_func(points).reshape(len(lat_new), len(lon_new))
        
        # 最終的に(lon, lat)の順序に転置
        br_surface_new = br_surface_new.T
        
        print(f"  補間完了: {br_surface_new.shape}")
        
    except Exception as e:
        print(f"  補間エラー: {e}")
        print(f"  最近傍補間にフォールバックします...")
        
        # 簡単なフォールバック：最近傍補間
        br_surface_new = np.zeros((len(lon_new), len(lat_new)))
        
        # HMIデータのグリッドを作成
        hmi_lon_grid, hmi_lat_grid = np.meshgrid(hmi_data['lon'], hmi_data['lat'], indexing='ij')
        
        for i, lon_val in enumerate(lon_new):
            for j, lat_val in enumerate(lat_new):
                # 最近傍点を見つける
                distances = np.sqrt((hmi_lon_grid - lon_val)**2 + (hmi_lat_grid - lat_val)**2)
                min_idx = np.unravel_index(np.argmin(distances), distances.shape)
                
                # データの次元に応じて値を取得
                if len(br_orig.shape) == 2:
                    if br_orig.shape == (len(hmi_data['lat']), len(hmi_data['lon'])):
                        br_surface_new[i, j] = br_orig[min_idx[1], min_idx[0]]  # (lat,lon) -> (lon,lat)
                    else:
                        br_surface_new[i, j] = br_orig[min_idx[0], min_idx[1]]  # (lon,lat)
                else:
                    br_surface_new[i, j] = 0.0  # デフォルト値
        
        print(f"  最近傍補間完了: {br_surface_new.shape}")
    
    print(f"  表面磁場範囲: {np.min(br_surface_new):.1f} ～ {np.max(br_surface_new):.1f} Gauss")
    
    # 3次元グリッドを作成
    LON, LAT, R = np.meshgrid(lon_new, lat_new, rix, indexing='ij')
    THETA = (90 - LAT) * np.pi / 180  # コラティチュード
    PHI = LON * np.pi / 180
    
    # ポテンシャル磁場の計算（簡略版）
    print("  ポテンシャル磁場を外挿中...")
    
    # 動径磁場の計算
    br = np.zeros((nlon_new, nlat_new, nr))
    bth = np.zeros((nlon_new, nlat_new, nr))
    bph = np.zeros((nlon_new, nlat_new, nr))
    
    # 各動径での磁場を計算
    for k in range(nr):
        r_current = rix[k]
        
        # 動径磁場: 双極子近似 + 表面磁場の影響
        if k == 0:  # 表面
            br[:, :, k] = br_surface_new
        else:
            # ポテンシャル磁場の外挿（簡略版）
            decay_factor = (1.0 / r_current)**2
            br[:, :, k] = br_surface_new * decay_factor
            
            # ソース面での境界条件
            if r_current >= rss * 0.9:  # ソース面近傍
                # 動径方向のみの磁場（ソース面条件）
                br[:, :, k] = br[:, :, k] * 0.1
        
        # θ, φ成分（簡略計算）
        bth[:, :, k] = 0.1 * br[:, :, k] * np.sin(THETA[:, :, k])
        bph[:, :, k] = 0.05 * br[:, :, k] * np.cos(PHI[:, :, k])
    
    result = {
        'br': br,
        'bth': bth,
        'bph': bph,
        'rix': rix,
        'lat': lat_new,
        'lon': lon_new,
        'theta': (90 - lat_new) * np.pi / 180,
        'phi': lon_new * np.pi / 180,
        'nr': nr,
        'nlat': nlat_new,
        'nlon': nlon_new,
        'source_surface': rss,
        'original_hmi': hmi_data
    }
    
    print(f"  PFSS計算完了")
    print(f"  最終グリッド: {nlon_new}×{nlat_new}×{nr}")
    
    return result

def run_pfss_fieldlines(pfss_data):
    """
    PFSS磁力線の描画
    """
    if not MODULES_AVAILABLE:
        print("球面モジュールが利用できません")
        return
    
    print("\n=== PFSS磁力線の描画 ===")
    
    # SphericalFieldData構造体を作成
    sph_data = spherical_field_data__define.SphericalFieldData()
    
    # 座標配列を設定
    sph_data.set_coordinate_arrays(pfss_data['lon'], pfss_data['lat'], pfss_data['rix'])
    
    # 磁場データを設定
    sph_data.set_vector_field(pfss_data['br'], pfss_data['bth'], pfss_data['bph'])
    
    print(f"データ設定完了:")
    print(f"  グリッド: {pfss_data['nlon']}×{pfss_data['nlat']}×{pfss_data['nr']}")
    
    # 磁力線開始点の設定
    print("磁力線開始点を設定中...")
    spherical_field_start_coord.spherical_field_start_coord(sph_data, fieldtype=5, spacing=10, radstart=1.2)
    
    # 磁力線のトレース
    print("磁力線をトレース中...")
    spherical_trace_field.spherical_trace_field(sph_data, stepmax=1000, quiet=False)
    
    # 磁力線の描画
    print("磁力線を描画中...")
    
    # 複数の視点で描画
    views = [
        (30, 0, "Standard View"),
        (0, 0, "Front View"),
        (60, 90, "High Latitude"),
        (-30, 180, "South Hemisphere")
    ]
    
    for bcent, lcent, desc in views:
        print(f"  {desc} (lat={bcent}°, lon={lcent}°)")
        spherical_draw_field.spherical_draw_field(sph_data, 
                                                bcent=bcent, lcent=lcent,
                                                imsc=100, xsize=500, ysize=500,
                                                onscreen=True, quiet=True)
    
    print("✓ 磁力線描画完了")

def visualize_hmi_and_pfss(hmi_data, pfss_data):
    """
    HMIデータとPFSS結果を可視化
    """
    print("\n=== データ可視化 ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'HMI Data and PFSS Results\n{hmi_data["time"]}', fontsize=16)
    
    # 1. 元のHMIデータ
    im1 = axes[0,0].imshow(hmi_data['br_surface'], extent=[hmi_data['lon'][0], hmi_data['lon'][-1], 
                                                          hmi_data['lat'][0], hmi_data['lat'][-1]], 
                          cmap='RdBu_r', origin='lower', aspect='auto')
    axes[0,0].set_title('Original HMI Magnetogram')
    axes[0,0].set_xlabel('Longitude (deg)')
    axes[0,0].set_ylabel('Latitude (deg)')
    plt.colorbar(im1, ax=axes[0,0], label='Br (Gauss)')
    
    # 2. PFSS表面磁場
    im2 = axes[0,1].imshow(pfss_data['br'][:, :, 0].T, extent=[0, 360, -90, 90], 
                          cmap='RdBu_r', origin='lower', aspect='auto')
    axes[0,1].set_title('PFSS Surface Field (r=1.0 Rs)')
    axes[0,1].set_xlabel('Longitude (deg)')
    axes[0,1].set_ylabel('Latitude (deg)')
    plt.colorbar(im2, ax=axes[0,1], label='Br (Gauss)')
    
    # 3. ソース面磁場
    im3 = axes[0,2].imshow(pfss_data['br'][:, :, -1].T, extent=[0, 360, -90, 90], 
                          cmap='RdBu_r', origin='lower', aspect='auto')
    axes[0,2].set_title(f'Source Surface Field (r={pfss_data["source_surface"]} Rs)')
    axes[0,2].set_xlabel('Longitude (deg)')
    axes[0,2].set_ylabel('Latitude (deg)')
    plt.colorbar(im3, ax=axes[0,2], label='Br (Gauss)')
    
    # 4. 動径プロファイル
    eq_idx = pfss_data['nlat'] // 2
    lon_idx = pfss_data['nlon'] // 4
    br_radial = pfss_data['br'][lon_idx, eq_idx, :]
    
    axes[1,0].plot(pfss_data['rix'], br_radial, 'b-', linewidth=2, marker='o')
    axes[1,0].set_title(f'Radial Profile (Equator, Lon={pfss_data["lon"][lon_idx]:.0f}°)')
    axes[1,0].set_xlabel('Radius (Rs)')
    axes[1,0].set_ylabel('Br (Gauss)')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 5. 磁場強度
    bmag_surface = np.sqrt(pfss_data['br'][:, :, 0]**2 + 
                          pfss_data['bth'][:, :, 0]**2 + 
                          pfss_data['bph'][:, :, 0]**2)
    
    im5 = axes[1,1].imshow(bmag_surface.T, extent=[0, 360, -90, 90], 
                          cmap='plasma', origin='lower', aspect='auto')
    axes[1,1].set_title('Magnetic Field Strength')
    axes[1,1].set_xlabel('Longitude (deg)')
    axes[1,1].set_ylabel('Latitude (deg)')
    plt.colorbar(im5, ax=axes[1,1], label='|B| (Gauss)')
    
    # 6. データ統計
    axes[1,2].text(0.1, 0.9, 'Data Statistics', fontsize=14, fontweight='bold',
                  transform=axes[1,2].transAxes)
    
    stats_text = f"""
HMI Original:
• Data shape: {hmi_data['br_surface'].shape}
• Br range: {np.nanmin(hmi_data['br_surface']):.1f} to {np.nanmax(hmi_data['br_surface']):.1f} G
• Valid pixels: {np.sum(np.isfinite(hmi_data['br_surface']))}

PFSS Result:
• Grid: {pfss_data['nlon']}×{pfss_data['nlat']}×{pfss_data['nr']}
• Radial range: {pfss_data['rix'][0]:.1f} - {pfss_data['rix'][-1]:.1f} Rs
• Surface Br: {np.min(pfss_data['br'][:,:,0]):.1f} to {np.max(pfss_data['br'][:,:,0]):.1f} G
• Source Br: {np.min(pfss_data['br'][:,:,-1]):.2f} to {np.max(pfss_data['br'][:,:,-1]):.2f} G

File: {os.path.basename(hmi_data['filepath'])}
Time: {hmi_data['time']}
"""
    
    axes[1,2].text(0.1, 0.85, stats_text, fontsize=10, fontfamily='monospace',
                  transform=axes[1,2].transAxes, verticalalignment='top')
    axes[1,2].set_xlim(0, 1)
    axes[1,2].set_ylim(0, 1)
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig('hmi_pfss_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ 可視化完了 (hmi_pfss_analysis.png として保存)")

def main():
    """
    メイン実行関数
    """
    print("HMI FITSファイルからPFSS磁力線描画")
    print("=" * 50)
    
    # HMI FITSファイルのパス（WSL形式）
    hmi_file = "/mnt/d/wsl/home/kinno-7010/Research_data/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    
    print(f"対象ファイル: {hmi_file}")
    
    if not os.path.exists(hmi_file):
        print(f"エラー: ファイルが見つかりません: {hmi_file}")
        print("\nファイルパスを確認してください。")
        print("現在のディレクトリ:", os.getcwd())
        return
    
    # 必要なパッケージのチェック
    if not ASTROPY_AVAILABLE:
        print("astropy パッケージをインストールしてください: pip install astropy")
        return
    
    try:
        # 1. HMI FITSファイルの読み込み
        hmi_data = read_hmi_fits(hmi_file)
        if hmi_data is None:
            print("HMIデータの読み込みに失敗しました")
            return
        
        # 2. PFSS磁場の計算
        pfss_data = create_pfss_from_hmi(hmi_data, nr=25, rss=2.5)
        
        # 3. データの可視化
        visualize_hmi_and_pfss(hmi_data, pfss_data)
        
        # 4. 磁力線の描画
        if MODULES_AVAILABLE:
            run_pfss_fieldlines(pfss_data)
        else:
            print("磁力線描画モジュールが利用できません（可視化のみ実行）")
        
        print("\n=== 処理完了 ===")
        print("✓ HMI FITSファイルからPFSS磁力線描画が完了しました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()