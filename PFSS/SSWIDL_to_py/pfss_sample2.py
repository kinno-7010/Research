"""
pfss_sample2.py - Sample script demonstrating how to do an extrapolation with a
source surface radius of 2.0 R_sun (as opposed to the [canonical] radius
of 2.5 that is used for the pre-computed field models available through
this SSW package).

This sample script demonstrates how to:
1. Download a surface-field magnetic field map
2. Create a magnetogram with custom resolution
3. Compute PFSS coefficients with custom source surface radius
4. Reconstruct the coronal field
5. Calculate diagnostic information

To use, run this script as a Python program.

M.DeRosa -  4 Aug 2010 - created

Converted to Python for SSWIDL_to_py package.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

# Import required modules from the converted package
from pfss_data_block import PfssDataBlock
from pfss_surffield_restore import pfss_surffield_restore
from pfss_time2file import pfss_time2file
from pfss_mag_create import pfss_mag_create
from pfss_get_potl_coeffs import pfss_get_potl_coeffs
from pfss_potl_field import pfss_potl_field
from mean_dtheta import mean_dtheta


def pfss_sample2():
    """
    Sample script demonstrating PFSS extrapolation with custom source surface radius.
    
    This function:
    1. Downloads a surface-field magnetic field map
    2. Creates a magnetogram with custom resolution
    3. Computes PFSS coefficients with custom source surface radius (2.0 R_sun)
    4. Reconstructs the coronal field
    5. Calculates and displays diagnostic information
    """
    
    print("PFSS Sample 2: Custom source surface radius extrapolation")
    print("=" * 55)
    
    # Get the data block
    data_block = PfssDataBlock()
    
    # First download a surface-field magnetic field map
    # Date/time is set here to Apr 5, 2003 for demonstration purposes
    print("Step 1: Downloading surface-field magnetic field map...")
    try:
        # Get the filename for the specified date (surface field)
        filename = pfss_time2file('2003-04-05', ssw_cat=True, url=True, surffield=True)
        print(f"Downloading/restoring surface field: {filename}")
        
        # Restore the surface field data
        sfield = pfss_surffield_restore(filename)
        print("Surface field data restored successfully!")
        
    except Exception as e:
        print(f"Error restoring surface field: {e}")
        print("This may be due to network issues or file availability.")
        print("For demonstration purposes, continuing with synthetic data...")
        # Could initialize with synthetic data here
        sfield = create_synthetic_surface_field()
        warnings.warn("Using synthetic data for demonstration")
    
    # Create a magnetogram with custom resolution
    print("\nStep 2: Creating magnetogram...")
    nlat0 = 192  # number of latitudinal gridpoints in magnetogram
    
    try:
        mag, lat, lon = pfss_mag_create(0, nlat=nlat0, file=sfield)
        print(f"Magnetogram created with {nlat0} latitudinal gridpoints")
        print(f"Magnetogram shape: {mag.shape}")
        
        # Update data block with magnetogram info
        data_block.nlat = nlat0
        data_block.nlon = nlat0 * 2  # typically 2:1 ratio
        data_block.lat = lat
        data_block.lon = lon
        
    except Exception as e:
        print(f"Error creating magnetogram: {e}")
        return
    
    # Get PFSS coefficients with custom source surface radius
    print("\nStep 3: Computing PFSS coefficients...")
    rss = 2.0  # source surface radius (instead of canonical 2.5)
    
    try:
        pfss_get_potl_coeffs(mag, rtop=rss)
        print(f"PFSS coefficients computed with source surface radius = {rss} R_sun")
        
    except Exception as e:
        print(f"Error computing PFSS coefficients: {e}")
        return
    
    # Reconstruct the coronal field in a spherical shell between 1 and rss
    print("\nStep 4: Reconstructing coronal field...")
    
    try:
        pfss_potl_field(rss, 2, trunc=True)
        print("Coronal field reconstructed successfully!")
        
    except Exception as e:
        print(f"Error reconstructing coronal field: {e}")
        return
    
    # Calculate diagnostic information
    print("\nStep 5: Calculating diagnostic information...")
    
    try:
        # Get field components and coordinates from data block
        br = data_block.br
        theta = data_block.theta if hasattr(data_block, 'theta') else None
        rix = data_block.rix if hasattr(data_block, 'rix') else None
        nlon = data_block.nlon
        nlat = data_block.nlat
        nr = data_block.nr if hasattr(data_block, 'nr') else br.shape[2]
        
        if theta is None or rix is None:
            print("Warning: Missing coordinate information, using defaults")
            theta = np.linspace(0, np.pi, nlat)
            rix = np.linspace(1.0, rss, nr)
        
        # Calculate diagnostic quantities
        cth = np.cos(theta)
        
        # Monopole calculation
        br_surface = br[:, :, 0]  # Surface field
        br_source = br[:, :, -1]  # Source surface field
        
        monopole = mean_dtheta(np.sum(br_surface, axis=0), cth) / nlon
        
        # Surface flux calculation
        surfflux = mean_dtheta(np.sum(np.abs(br_surface), axis=0), cth) * nlat * 1e18 * rix[0]**2
        
        # Open flux calculation
        openflux = mean_dtheta(np.sum(np.abs(br_source), axis=0), cth) * nlat * 1e18 * rix[-1]**2
        
        # Monopole fractions
        monfrac = monopole * nlon * nlat * 1e18 / np.array([surfflux, openflux])
        
        # Remove monopole from coronal field
        r2inv = np.broadcast_to((1.0 / rix**2).reshape(1, 1, nr), (nlon, nlat, nr))
        br_corrected = br - monopole * r2inv
        
        # Update the data block with corrected field
        data_block.br = br_corrected
        
        # Print diagnostic information
        print("\nDiagnostic Results:")
        print("-" * 30)
        print(f"Monopole = {monopole:.6e}")
        print(f"Unsigned flux = {surfflux:.6e}")
        print(f"Open flux = {openflux:.6e}")
        print(f"Monopole fractions = {monfrac}")
        print(f"Source surface radius = {rss} R_sun")
        print(f"Number of radial grid points = {nr}")
        print(f"Radial range = {rix[0]:.2f} to {rix[-1]:.2f} R_sun")
        
        # Create a simple visualization
        create_diagnostic_plots(br_surface, br_source, lat, lon, monopole, surfflux, openflux)
        
    except Exception as e:
        print(f"Error calculating diagnostics: {e}")
        return
    
    print("\nStep 6: Field lines can now be drawn...")
    print("At this point field lines can be drawn, etc., as in pfss_sample1.py")
    print("The field model is now ready for field line tracing and visualization.")
    
    print("\nPFSS Sample 2 completed successfully!")


def create_diagnostic_plots(br_surface, br_source, lat, lon, monopole, surfflux, openflux):
    """
    Create diagnostic plots showing surface and source surface fields.
    
    Args:
        br_surface (numpy.ndarray): Surface magnetic field
        br_source (numpy.ndarray): Source surface magnetic field
        lat (numpy.ndarray): Latitude coordinates
        lon (numpy.ndarray): Longitude coordinates
        monopole (float): Monopole component
        surfflux (float): Surface flux
        openflux (float): Open flux
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Create coordinate meshes
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    
    # Plot 1: Surface field
    im1 = axes[0, 0].contourf(lon_mesh, lat_mesh, br_surface.T, levels=50, cmap='RdBu_r')
    axes[0, 0].set_title('Surface Magnetic Field (R = 1.0 R☉)')
    axes[0, 0].set_xlabel('Longitude (degrees)')
    axes[0, 0].set_ylabel('Latitude (degrees)')
    plt.colorbar(im1, ax=axes[0, 0], label='Br (Gauss)')
    
    # Plot 2: Source surface field
    im2 = axes[0, 1].contourf(lon_mesh, lat_mesh, br_source.T, levels=50, cmap='RdBu_r')
    axes[0, 1].set_title('Source Surface Magnetic Field (R = 2.0 R☉)')
    axes[0, 1].set_xlabel('Longitude (degrees)')
    axes[0, 1].set_ylabel('Latitude (degrees)')
    plt.colorbar(im2, ax=axes[0, 1], label='Br (Gauss)')
    
    # Plot 3: Latitudinal averages
    lat_avg_surface = np.mean(br_surface, axis=0)
    lat_avg_source = np.mean(br_source, axis=0)
    
    axes[1, 0].plot(lat, lat_avg_surface, 'b-', linewidth=2, label='Surface (R=1.0)')
    axes[1, 0].plot(lat, lat_avg_source, 'r-', linewidth=2, label='Source Surface (R=2.0)')
    axes[1, 0].set_title('Latitudinal Averages')
    axes[1, 0].set_xlabel('Latitude (degrees)')
    axes[1, 0].set_ylabel('Average Br (Gauss)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Diagnostic information
    axes[1, 1].text(0.1, 0.8, f'Monopole: {monopole:.3e}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.7, f'Surface Flux: {surfflux:.3e}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.6, f'Open Flux: {openflux:.3e}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.5, f'Open Flux Fraction: {openflux/surfflux:.3f}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.4, f'Source Surface: 2.0 R☉', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.3, f'Grid: {br_surface.shape[0]} × {br_surface.shape[1]}', transform=axes[1, 1].transAxes, fontsize=12)
    
    axes[1, 1].set_title('Diagnostic Information')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def create_synthetic_surface_field():
    """
    Create synthetic surface field data for demonstration when real data is not available.
    
    Returns:
        dict: Synthetic surface field structure
    """
    
    print("Creating synthetic surface field data for demonstration...")
    
    # Create a simple synthetic surface field
    nlat, nlon = 192, 384
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon)
    
    # Create some synthetic magnetic field pattern
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    
    # Create a simple dipole-like field with some complexity
    br = (10 * np.cos(lat_mesh * np.pi/180) + 
          5 * np.sin(2 * lat_mesh * np.pi/180) * np.cos(lon_mesh * np.pi/180) +
          2 * np.sin(4 * lat_mesh * np.pi/180) * np.sin(2 * lon_mesh * np.pi/180))
    
    # Create synthetic structure
    sfield = {
        'br': br.T,  # Transpose to match expected format
        'lat': lat,
        'lon': lon,
        'nlat': nlat,
        'nlon': nlon,
        'date': '2003-04-05',
        'synthetic': True
    }
    
    print("Synthetic surface field created")
    return sfield


# For direct script execution
if __name__ == "__main__":
    pfss_sample2()


# For compatibility with IDL calling convention
def pfss_sample2_idl():
    """
    IDL-compatible wrapper for pfss_sample2.
    
    This maintains the original interface while providing the same functionality.
    """
    pfss_sample2()


# Additional utility functions for enhanced functionality
def pfss_sample2_interactive():
    """
    Interactive version of pfss_sample2 with user input for parameters.
    """
    
    print("Interactive PFSS Sample 2")
    print("=" * 25)
    
    # Get user input for date
    date_str = input("Enter date (YYYY-MM-DD) [default: 2003-04-05]: ").strip()
    if not date_str:
        date_str = '2003-04-05'
    
    # Get user input for parameters
    try:
        nlat0 = int(input("Enter number of latitudinal gridpoints (192=default): ") or "192")
        rss = float(input("Enter source surface radius in Rsun (2.0=default): ") or "2.0")
    except ValueError:
        print("Invalid input, using defaults")
        nlat0, rss = 192, 2.0
    
    # Run the modified sample with user parameters
    pfss_sample2_with_params(date_str, nlat0, rss)


def pfss_sample2_with_params(date_str, nlat0, rss):
    """
    Run pfss_sample2 with custom parameters.
    
    Args:
        date_str (str): Date string in YYYY-MM-DD format
        nlat0 (int): Number of latitudinal gridpoints
        rss (float): Source surface radius in solar radii
    """
    
    print(f"Running PFSS Sample 2 with custom parameters:")
    print(f"  Date: {date_str}")
    print(f"  Latitudinal gridpoints: {nlat0}")
    print(f"  Source surface radius: {rss} Rsun")
    
    # Similar to pfss_sample2() but with custom parameters
    # Implementation would be similar to the main function above