#!/usr/bin/env python3
"""
STEREO-A/SECCHI/COR1 Professional Astronomical Image Plotter

This script provides integrated COR1 image processing and plotting functionality 
with SSWIDL features. Includes astronomical visualization tools such as color tables, 
annotations, and measurement capabilities.

SSWIDL Reference Programs:
- secchi_colors.pro (color tables)
- scc_add_datetime.pro (date-time stamps)
- scc_add_logo.pro (logo placement)
- drawcoordgrid.pro (coordinate grids)
- stereo_rsun.pro (solar radius calculation)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
import glob
from datetime import datetime
from matplotlib.patches import Circle
import argparse
import sys

# Enhanced feature module imports
try:
    from cor1_colors import SECCHIColors
    from cor1_annotations import COR1Annotations
    from cor1_solar_utils import COR1SolarUtils
except ImportError as e:
    print(f"Warning: Could not import enhanced modules: {e}")
    print("Running in basic mode only.")
    SECCHIColors = None
    COR1Annotations = None
    COR1SolarUtils = None

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

def read_cor1_data(filepath):
    """
    COR1 FITSファイルを読み込み、データとヘッダーを返す
    
    Parameters:
    -----------
    filepath : str
        FITSファイルのパス
    
    Returns:
    --------
    data : numpy.ndarray
        画像データ
    header : astropy.io.fits.Header
        FITSヘッダー
    """
    try:
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            header = hdul[0].header
        return data, header
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

def calculate_solar_radius_pixel(header):
    """
    FITSヘッダーから太陽半径をピクセル単位で計算
    
    Parameters:
    -----------
    header : astropy.io.fits.Header
        FITSヘッダー
    
    Returns:
    --------
    rsun_pixel : float
        太陽半径（ピクセル単位）
    sun_center_x : float
        太陽中心のx座標（ピクセル単位）
    sun_center_y : float
        太陽中心のy座標（ピクセル単位）
    """
    # Solar radius (arcsec)
    rsun_arcsec = header.get('RSUN', None)
    if rsun_arcsec is None:
        return None, None, None
    
    # Pixel scale (arcsec/pixel)
    cdelt1 = header.get('CDELT1', None)
    cdelt2 = header.get('CDELT2', None)
    
    if cdelt1 is None or cdelt2 is None:
        return None, None, None
    
    # Solar center pixel position
    crpix1 = header.get('CRPIX1', None)
    crpix2 = header.get('CRPIX2', None)
    
    if crpix1 is None or crpix2 is None:
        return None, None, None
    
    # Convert solar radius to pixel units
    rsun_pixel = rsun_arcsec / abs(cdelt1)
    
    # Solar center coordinates (convert from 1-indexed to 0-indexed)
    sun_center_x = crpix1 - 1
    sun_center_y = crpix2 - 1
    
    return rsun_pixel, sun_center_x, sun_center_y

def plot_cor1_image(data, header, filepath, save_path=None, 
                   use_secchi_colors=True, add_annotations=True, 
                   add_coordinate_grid=False, add_measurements=False,
                   color_scaling='zscale', show_logo=True):
    """
    COR1データをプロフェッショナル天体画像としてプロット
    
    Parameters:
    -----------
    data : numpy.ndarray
        画像データ
    header : astropy.io.fits.Header
        FITSヘッダー
    filepath : str
        元のファイルパス
    save_path : str, optional
        保存先パス
    use_secchi_colors : bool, optional
        SECCHI専用カラーテーブルを使用
    add_annotations : bool, optional
        日時スタンプとロゴを追加
    add_coordinate_grid : bool, optional
        座標グリッドを追加
    add_measurements : bool, optional
        測定ツールを表示
    color_scaling : str, optional
        カラースケーリング方法（'zscale', 'percentile', 'minmax'）
    show_logo : bool, optional
        SECCHIロゴを表示
    """
    # Enhanced module initialization
    if SECCHIColors is not None and use_secchi_colors:
        colors = SECCHIColors(silent=True)
    else:
        colors = None
    
    if COR1Annotations is not None and add_annotations:
        annotations = COR1Annotations(silent=True)
    else:
        annotations = None
    
    if COR1SolarUtils is not None:
        solar_utils = COR1SolarUtils(silent=True)
    else:
        solar_utils = None
    
    # Get basic data information
    obs_time = header.get('DATE-OBS', header.get('DATE_OBS', 'Unknown'))
    instrument = header.get('INSTRUME', 'Unknown')
    detector = header.get('DETECTOR', 'Unknown')
    
    # Parse time information
    try:
        time_obj = Time(obs_time)
        time_str = time_obj.datetime.strftime('%Y-%m-%d %H:%M:%S')
    except:
        time_str = obs_time
    
    # Calculate solar radius and center coordinates (enhanced version priority)
    if solar_utils is not None:
        sun_center = solar_utils.get_sun_center(header)
        sun_center_x, sun_center_y = sun_center['xcen'], sun_center['ycen']
        rsun_arcsec = solar_utils.calculate_solar_radius_arcsec(obs_time, 'A')
        cdelt1 = abs(header.get('CDELT1', 1.0))
        rsun_pixel = rsun_arcsec / cdelt1
    else:
        # Fallback: existing functionality
        rsun_pixel, sun_center_x, sun_center_y = calculate_solar_radius_pixel(header)
        rsun_arcsec = header.get('RSUN', None)
    
    # Data statistics
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data_mean = np.nanmean(data)
    data_std = np.nanstd(data)
    
    print(f"File: {os.path.basename(filepath)}")
    print(f"Observation Time: {time_str}")
    print(f"Instrument: {instrument}")
    print(f"Detector: {detector}")
    print(f"Data shape: {data.shape}")
    print(f"Data range: {data_min:.2f} to {data_max:.2f}")
    print(f"Data mean: {data_mean:.2f} ± {data_std:.2f}")
    if rsun_pixel is not None:
        print(f"Solar radius: {rsun_arcsec:.2f} arcsec ({rsun_pixel:.2f} pixels)")
        print(f"Solar center: ({sun_center_x:.2f}, {sun_center_y:.2f}) pixels")
    print("-" * 50)
    
    # Color scaling calculation
    if colors is not None:
        vmin, vmax = colors.calculate_scaling(data, method=color_scaling)
        cmap = colors.get_colormap('COR1')
    else:
        # Fallback: traditional scaling
        interval = ZScaleInterval()
        vmin, vmax = 2000, 6000
        cmap = 'gray'
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Image display (using SECCHI-specific color tables)
    im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Intensity [DN]', labelpad=20)
    
    # Professional title setting
    title_text = f'STEREO-A/SECCHI/COR1'
    if add_annotations and annotations is not None:
        # Date-time stamp is added by annotation feature, keep title simple
        ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
    else:
        ax.set_title(f'{title_text}\n{time_str}', fontsize=14, fontweight='bold')
    
    # Coordinate labels
    ax.set_xlabel('Pixel X', fontsize=12)
    ax.set_ylabel('Pixel Y', fontsize=12)
    
    # Coordinate grid addition
    if add_coordinate_grid and annotations is not None:
        # Add astronomical coordinate grid
        annotations.draw_coordinate_grid(data, header, system='HCR', color='cyan', thickness=1)
    else:
        # Basic grid
        ax.grid(True, alpha=0.3)
    
    # Draw solar radius circles (SSWIDL-compliant enhanced version)
    if rsun_pixel is not None and sun_center_x is not None and sun_center_y is not None:
        # Solar limb circle drawing (IDL-compliant)
        if annotations is not None:
            annotations.add_solar_limb_circle(data, header, thickness=2, color='yellow')
        
        # Solar radius circles (1Rs, 2Rs, 3Rs) - astronomical observation standard display
        colors_list = ['yellow', 'orange', 'red']
        alphas = [0.9, 0.7, 0.6]
        linewidths = [2.5, 2.0, 1.5]
        
        for i, (color, alpha, lw) in enumerate(zip(colors_list, alphas, linewidths)):
            radius = (i + 1) * rsun_pixel
            circle = Circle((sun_center_x, sun_center_y), radius, 
                           fill=False, edgecolor=color, linewidth=lw, 
                           alpha=alpha, label=f'{i+1} Rs')
            ax.add_patch(circle)
        
        # Solar center marker
        ax.plot(sun_center_x, sun_center_y, '+', color='yellow', 
                markersize=12, markeredgewidth=3, label='Solar Center')
        
        # Professional legend
        legend = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                          fontsize=11, framealpha=0.9, edgecolor='gray')
        legend.get_frame().set_facecolor('white')
        for text in legend.get_texts():
            text.set_color('white')
    
    # Apply annotation features
    if add_annotations and annotations is not None:
        # Add date-time stamp (SSWIDL scc_add_datetime compliant)
        try:
            # Use matplotlib text instead of drawing directly on image
            datetime_str, detector_info = annotations.format_datetime_string(header)
            
            # Dynamic font size calculation
            sum_factor = annotations._get_size_factor(data.shape)
            config = annotations.size_configs[sum_factor]
            
            # Date-time stamp placement (lower left)
            ax.text(0.02, 0.08, datetime_str, transform=ax.transAxes, 
                   fontsize=config['font_size']*0.8, color='white', 
                   weight=config['font_weight'], family='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            # Detector information placement (above date-time stamp)
            if detector_info:
                ax.text(0.02, 0.12, detector_info, transform=ax.transAxes, 
                       fontsize=config['font_size']*0.7, color='white', 
                       weight=config['font_weight'], family='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        except Exception as e:
            print(f"Warning: Could not add annotations: {e}")
        
        # Logo addition (upper right)
        if show_logo:
            try:
                # SECCHI logo text (replace with actual logo file if available)
                ax.text(0.98, 0.98, 'SECCHI', transform=ax.transAxes, 
                       fontsize=14, color='white', weight='bold',
                       ha='right', va='top',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='navy', alpha=0.8))
            except Exception as e:
                print(f"Warning: Could not add logo: {e}")
    
    # Measurement tools display
    if add_measurements and solar_utils is not None:
        try:
            # Sample measurement points display
            test_points = [(sun_center_x + rsun_pixel, sun_center_y),
                          (sun_center_x, sun_center_y + 2*rsun_pixel)]
            
            # Distance measurement display
            for i, point in enumerate(test_points):
                distance_rsun = solar_utils.calculate_distance(
                    (sun_center_x, sun_center_y), point, header, 'rsun')
                # ax.plot(point[0], point[1], 'go', markersize=6)
                # ax.text(point[0]+10, point[1]+10, f'{distance_rsun:.1f} R☉',
                #        color='green', fontsize=10, weight='bold')
        except Exception as e:
            print(f"Warning: Could not add measurements: {e}")
    
    # Professional layout adjustment
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # High-quality save or plot display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        print(f"Professional plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig, ax

def main():
    """Main function with command-line argument support"""
    parser = argparse.ArgumentParser(
        description='STEREO-A/SECCHI/COR1 Professional Astronomical Image Plotter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 cor1_plot.py                           # Basic mode
  python3 cor1_plot.py --enhanced                # Full enhanced mode
  python3 cor1_plot.py --colors --annotations    # Selective features
  python3 cor1_plot.py --file custom_file.fits   # Custom file
  python3 cor1_plot.py --scaling percentile      # Custom scaling
        """)
    
    parser.add_argument('--file', type=str, 
                       help='Specific FITS file to process')
    parser.add_argument('--enhanced', action='store_true',
                       help='Enable all enhanced features')
    parser.add_argument('--colors', action='store_true', 
                       help='Use SECCHI color tables')
    parser.add_argument('--annotations', action='store_true',
                       help='Add datetime stamps and logo')
    parser.add_argument('--grid', action='store_true',
                       help='Add coordinate grid')
    parser.add_argument('--measurements', action='store_true',
                       help='Show measurement tools')
    parser.add_argument('--scaling', choices=['zscale', 'percentile', 'minmax'],
                       default='zscale', help='Color scaling method')
    parser.add_argument('--no-logo', action='store_true',
                       help='Disable SECCHI logo')
    parser.add_argument('--output-dir', type=str,
                       default='/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Default: Enhanced mode enables all features by default
    # User can disable features with specific flags
    use_colors = args.enhanced or args.colors or True  # Default enabled
    add_annotations = args.enhanced or args.annotations or True  # Default enabled
    add_grid = args.enhanced or args.grid or True  # Default enabled
    add_measurements = args.enhanced or args.measurements or True  # Default enabled
    show_logo = (args.enhanced or args.annotations or True) and not args.no_logo
    
    # Feature availability check
    if (use_colors or add_annotations or add_measurements) and SECCHIColors is None:
        print("Warning: Enhanced features requested but modules not available.")
        print("Running in basic mode. Please check module imports.")
        use_colors = False
        add_annotations = False
        add_measurements = False
    
    # File processing
    if args.file:
        # Process specified file
        target_files = [args.file]
        data_dir = os.path.dirname(args.file) if os.path.dirname(args.file) else "."
    else:
        # Process default files
        data_dir = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/calibration"
        target_files = ['20220613_032136_n4c1A_processed.fits']
    
    print(f"=== STEREO-A/SECCHI/COR1 Professional Plotter ===")
    print(f"Enhanced features: Colors={use_colors}, Annotations={add_annotations}")
    print(f"Grid={add_grid}, Measurements={add_measurements}, Logo={show_logo}")
    print(f"Scaling method: {args.scaling}")
    print("=" * 50)
    
    # Process each file
    processed_count = 0
    for filename in target_files:
        if args.file:
            filepath = filename
        else:
            filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        
        print(f"\nProcessing: {os.path.basename(filepath)}")
        
        # Data loading
        data, header = read_cor1_data(filepath)
        
        if data is None:
            continue
        
        # Create plot (integrated enhanced version)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        suffix = '_enhanced' if args.enhanced else '_professional'
        output_filename = f"{base_name}{suffix}.png"
        save_path = os.path.join(args.output_dir, output_filename)
        
        try:
            fig, ax = plot_cor1_image(
                data, header, filepath, save_path,
                use_secchi_colors=use_colors,
                add_annotations=add_annotations,
                add_coordinate_grid=add_grid,
                add_measurements=add_measurements,
                color_scaling=args.scaling,
                show_logo=show_logo
            )
            plt.show()
            plt.close(fig)  # Memory conservation
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    print(f"\n=== Processing Complete ===")
    print(f"Processed files: {processed_count}/{len(target_files)}")
    if processed_count > 0:
        print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()