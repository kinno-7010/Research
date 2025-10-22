# -*- coding: utf-8 -*-
import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates

def load_and_prepare_instrument_data(filename, instrument_name, is_lasco=False):
    try:
        with fits.open(filename) as hdul:
            data = hdul[0].data.astype(float)
            header = hdul[0].header
            print(f"Successfully loaded {instrument_name} file: {filename}")
    except FileNotFoundError:
        print(f"CRITICAL Error: {instrument_name} file '{filename}' not found. Please check the path.")
        raise SystemExit
    except Exception as e:
        print(f"CRITICAL Error: Could not read {instrument_name} file '{filename}'. Error: {e}")
        raise SystemExit()

    params = {}
    try:
        params['nx'] = header['NAXIS1']
        params['ny'] = header['NAXIS2']
        params['cx'] = header['CRPIX1'] - 1
        params['cy'] = header['CRPIX2'] - 1
        params['scale'] = abs(header['CDELT1'])

        if is_lasco:
            params['rsun_arc'] = header.get('RSUN_OBS', 959.2)
            if 'RSUN_OBS' not in header:
                print(f"Warning: {instrument_name} 'RSUN_OBS' not found in header. Using default value: {params['rsun_arc']} arcsec.")
        else:
            if 'RSUN_OBS' in header:
                params['rsun_arc'] = header['RSUN_OBS']
            elif 'R_SUN' in header and 'CDELT1' in header:
                params['rsun_arc'] = header['R_SUN'] * abs(header['CDELT1'])
                print(f"Warning: {instrument_name} 'RSUN_OBS' not found, calculated from 'R_SUN' and 'CDELT1'.")
            else:
                raise KeyError(f"{instrument_name} header missing solar radius information (RSUN_OBS or R_SUN/CDELT1).")
        
        params['px_per_rsun'] = params['rsun_arc'] / params['scale']
        
        print(f"{instrument_name} params: cx={params['cx']:.2f}, cy={params['cy']:.2f}, "
              f"scale={params['scale']:.3f} arcsec/px, rsun={params['rsun_arc']:.2f} arcsec, "
              f"R_sun_px={params['px_per_rsun']:.2f} px")
        
    except KeyError as e:
        print(f"CRITICAL Error: Missing essential keyword in {instrument_name} header: {e}. Cannot proceed.")
        raise SystemExit
        
    return data, params

def combine_corona_data(data_lasco, params_lasco, data_mk4, params_mk4, r_map_lasco, r_ranges,
                        blend_r_inner=None, blend_r_outer=None):
    """
    Mk4をLASCOグリッドに一次補間し、半径で結合する。
    2.2 R_sun 近傍にブレンディング帯を設け、pBの外側ビン抜けを防ぐ。
    """
    # --- LASCO グリッドでの正規化座標
    y_l_idx, x_l_idx = np.indices((params_lasco['ny'], params_lasco['nx']))
    x_norm_on_lasco_grid = (x_l_idx - params_lasco['cx']) / params_lasco['px_per_rsun']
    y_norm_on_lasco_grid = (y_l_idx - params_lasco['cy']) / params_lasco['px_per_rsun']

    # --- Mk4 データを LASCO グリッドへ補間
    coords_for_mk4_sampling_y = y_norm_on_lasco_grid * params_mk4['px_per_rsun'] + params_mk4['cy']
    coords_for_mk4_sampling_x = x_norm_on_lasco_grid * params_mk4['px_per_rsun'] + params_mk4['cx']
    coords_mk4_sampling = np.vstack([coords_for_mk4_sampling_y.ravel(),
                                     coords_for_mk4_sampling_x.ravel()])
    interp_mk4_on_lasco_grid = map_coordinates(
        data_mk4, coords_mk4_sampling, order=1, mode='constant', cval=np.nan
    ).reshape((params_lasco['ny'], params_lasco['nx']))

    final_image = np.full_like(data_lasco, np.nan)

    # --- ラジアル領域
    r_mk4_in   = float(r_ranges['mk4_inner'])
    r_switch   = float(r_ranges['mk4_outer_lasco_inner'])
    r_lasco_out = float(r_ranges['lasco_outer'])

    mask_lasco = (r_map_lasco >= r_switch) & (r_map_lasco <= r_lasco_out)
    mask_mk4   = (r_map_lasco >= r_mk4_in) & (r_map_lasco <  r_switch)

    final_image[mask_lasco] = data_lasco[mask_lasco]
    final_image[mask_mk4]   = interp_mk4_on_lasco_grid[mask_mk4]

    # --- 境界ブレンド帯（既定：±0.15 R_sun）
    if blend_r_inner is None:
        blend_r_inner = max(r_mk4_in, r_switch - 0.15)
    if blend_r_outer is None:
        blend_r_outer = min(r_lasco_out, r_switch + 0.15)
    if blend_r_outer > blend_r_inner:
        bm = (r_map_lasco >= blend_r_inner) & (r_map_lasco < blend_r_outer)
        if np.any(bm):
            # 0→1 の線形ウェイト（内側=K-Cor優先、外側=LASCO優先）
            w = (r_map_lasco - blend_r_inner) / (blend_r_outer - blend_r_inner)
            w = np.clip(w, 0.0, 1.0)

            mk4v  = interp_mk4_on_lasco_grid
            lasv  = data_lasco
            # どちらかが NaN のときはもう片方を採用
            blended = (1.0 - w) * mk4v + w * lasv
            blended = np.where(np.isnan(mk4v), lasv, blended)
            blended = np.where(np.isnan(lasv), mk4v, blended)

            final_image[bm] = blended[bm]

    # 範囲外マスク
    final_image[r_map_lasco < r_mk4_in]   = np.nan
    final_image[r_map_lasco > r_lasco_out] = np.nan
    return final_image

def extract_pB_profile(image_data_extract, r_map_extract, theta_deg_extract, r_bins_extract, 
                       cy_center, cx_center, ny_grid, nx_grid,
                       angle_halfwidth_deg: float = 5.0):
    """
    θ=theta_deg_extract ± angle_halfwidth_deg の扇形で、半径ビンごとに pB の平均を返す。
    K-Cor の 0 値を有効とみなすため、pB>=0 を採用。
    """
    theta_rad = np.radians(theta_deg_extract)
    angle_tol = np.radians(angle_halfwidth_deg)

    y_idx, x_idx = np.indices((ny_grid, nx_grid))
    angle_map = np.arctan2(y_idx - cy_center, x_idx - cx_center)
    # wrap-safe な角度差
    angle_diff = np.arctan2(np.sin(angle_map - theta_rad), np.cos(angle_map - theta_rad))

    mask_angle = (np.abs(angle_diff) < angle_tol)
    # ★ ここを >=0 に変更（負値と NaN は除外）
    mask_valid_pb = (~np.isnan(image_data_extract)) & (image_data_extract >= 0.0)

    combined_mask = mask_angle & mask_valid_pb
    pB_profile = np.full(len(r_bins_extract) - 1, np.nan, dtype=float)

    for i in range(len(r_bins_extract) - 1):
        rmin = r_bins_extract[i]
        rmax = r_bins_extract[i+1]
        mrad = (r_map_extract >= rmin) & (r_map_extract < rmax)
        m = mrad & combined_mask
        if np.any(m):
            pB_profile[i] = np.nanmean(image_data_extract[m])
    return pB_profile
