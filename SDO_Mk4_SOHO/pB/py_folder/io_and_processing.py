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

def combine_corona_data(data_lasco, params_lasco, data_mk4, params_mk4, r_map_lasco, r_ranges):
    y_l_idx, x_l_idx = np.indices((params_lasco['ny'], params_lasco['nx']))
    x_norm_on_lasco_grid = (x_l_idx - params_lasco['cx']) / params_lasco['px_per_rsun']
    y_norm_on_lasco_grid = (y_l_idx - params_lasco['cy']) / params_lasco['px_per_rsun']

    coords_for_mk4_sampling_y = y_norm_on_lasco_grid * params_mk4['px_per_rsun'] + params_mk4['cy']
    coords_for_mk4_sampling_x = x_norm_on_lasco_grid * params_mk4['px_per_rsun'] + params_mk4['cx']
    coords_mk4_sampling = np.vstack([coords_for_mk4_sampling_y.ravel(),
                                      coords_for_mk4_sampling_x.ravel()])

    interp_mk4_on_lasco_grid = map_coordinates(data_mk4, coords_mk4_sampling,
                                               order=1, mode='constant', cval=np.nan)
    interp_mk4_on_lasco_grid = interp_mk4_on_lasco_grid.reshape((params_lasco['ny'], params_lasco['nx']))

    final_image = np.full_like(data_lasco, np.nan)

    mask_lasco_region = (r_map_lasco >= r_ranges['mk4_outer_lasco_inner']) & \
                        (r_map_lasco <= r_ranges['lasco_outer'])
    final_image[mask_lasco_region] = data_lasco[mask_lasco_region]

    mask_mk4_region = (r_map_lasco >= r_ranges['mk4_inner']) & \
                      (r_map_lasco < r_ranges['mk4_outer_lasco_inner'])
    final_image[mask_mk4_region] = interp_mk4_on_lasco_grid[mask_mk4_region]

    final_image[r_map_lasco < r_ranges['mk4_inner']] = np.nan
    final_image[r_map_lasco > r_ranges['lasco_outer']] = np.nan
    
    return final_image

def extract_pB_profile(image_data_extract, r_map_extract, theta_deg_extract, r_bins_extract, 
                       cy_center, cx_center, ny_grid, nx_grid):
    theta_rad_extract = np.radians(theta_deg_extract)
    angle_tolerance_rad = np.radians(5)
    y_idx_grid, x_idx_grid = np.indices((ny_grid, nx_grid))
    angle_map_extract = np.arctan2(y_idx_grid - cy_center, x_idx_grid - cx_center)
    angle_diff = np.arctan2(np.sin(angle_map_extract - theta_rad_extract), np.cos(angle_map_extract - theta_rad_extract))
    mask_angle = (np.abs(angle_diff) < angle_tolerance_rad)
    mask_valid_pb = (~np.isnan(image_data_extract)) & (image_data_extract > 0)
    combined_mask = mask_angle & mask_valid_pb
    pB_profile_extracted = np.zeros(len(r_bins_extract) - 1)
    for i in range(len(r_bins_extract) - 1):
        r_min_bin = r_bins_extract[i]
        r_max_bin = r_bins_extract[i+1]
        bin_mask_radius = (r_map_extract >= r_min_bin) & (r_map_extract < r_max_bin)
        final_bin_mask = bin_mask_radius & combined_mask
        if np.any(final_bin_mask):
            pB_profile_extracted[i] = np.nanmean(image_data_extract[final_bin_mask])
        else:
            pB_profile_extracted[i] = np.nan
    return pB_profile_extracted
