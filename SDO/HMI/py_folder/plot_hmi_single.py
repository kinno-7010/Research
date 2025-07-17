import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

from hmi_analysis_wcs import draw_hmi_solar_grid, draw_solar_coordinate_lines, read_hmi_quick, extract_ar_region_data


def plot_hmi_single(hmi_data, downsample=1):
    data = hmi_data['data']
    time_str = hmi_data['time']
    hmi_map = hmi_data.get('sunpy_map')
    
    # --- プロット設定 (変更なし) ---
    ny, nx = data.shape
    center_x, center_y = nx // 2, ny // 2
    x_min_pix, x_max_pix = center_x - 512, center_x + 0
    y_min_pix, y_max_pix = center_y - 100, center_y + 512

    x_lims_pix = (x_min_pix, x_max_pix)
    y_lims_pix = (y_min_pix, y_max_pix)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=hmi_map.wcs)
    
    im1 = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-200, vmax=200)
    cbar1 = fig.colorbar(im1, ax=ax, orientation='vertical', pad=0.1, shrink=0.8, fraction=0.02) 
    cbar1.ax.set_ylabel('$B_r$ (Gauss)', fontsize=12)
    ax.set_title('SDO/HMI Radial Magnetic Field (2022-06-13 03:00:00)', fontsize=16)
    ax.set_xlabel('Solar X (arcsec)', fontsize=14)
    ax.set_ylabel('Solar Y (arcsec)', fontsize=14)
    ax.set_xlim(x_lims_pix); ax.set_ylim(y_lims_pix)
    draw_hmi_solar_grid(hmi_map, ax)

    
    masked_data = data[y_min_pix:y_max_pix, x_min_pix:x_max_pix]
    
    return fig, ax, masked_data, hmi_map

if __name__ == "__main__":
    hmi_file = "/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits"
    hmi_data = read_hmi_quick(hmi_file)
    fig, ax, masked_data, hmi_map = plot_hmi_single(hmi_data)
    plt.savefig("/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/hmi_single.png", dpi=300)
    print("Saved figure to /mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/hmi_single.png")
    plt.show()
