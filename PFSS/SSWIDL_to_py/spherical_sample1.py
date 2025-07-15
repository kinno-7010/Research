"""
Sample Script for Spherical Field Tools

This script demonstrates how to use the tools in the spherical directory
for visualizing and analyzing spherical vector field data.

To use, run this script from Python:
    python spherical_sample1.py

MODIFICATION HISTORY:
    M.DeRosa - 19 Jan 2006 - created (IDL version)
    Converted to Python - 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

# Import required modules (assuming they exist)
try:
    from spherical_field_data__define import SphericalFieldData
    from spherical_field_start_coord import spherical_field_start_coord
    from spherical_trace_field import spherical_trace_field
    from spherical_draw_field import spherical_draw_field
    from spherical_image_create import spherical_image_create
    print("All required modules imported successfully")
except ImportError as e:
    print(f"Warning: Some modules not yet implemented: {e}")
    print("This is a demonstration script showing the intended workflow")


def create_sample_pfss_data() -> SphericalFieldData:
    """
    Create a sample PFSS-like vector field for demonstration.
    
    This creates a simplified spherical vector field that resembles
    potential field source surface (PFSS) data.
    
    Returns:
    --------
    SphericalFieldData
        A sample spherical field data structure
    """
    
    print("Creating sample vector field data...")
    
    # Set up grid parameters
    nr, nlat, nlon = 30, 45, 90  # Modest resolution for demo
    
    # Create coordinate arrays
    rix = np.linspace(1.0, 2.5, nr)  # Solar radii from 1 to 2.5
    lat = np.linspace(-90, 90, nlat)  # Latitude in degrees
    lon = np.linspace(0, 360, nlon, endpoint=False)  # Longitude in degrees
    
    # Create meshgrids
    LON, LAT, R = np.meshgrid(lon, lat, rix, indexing='ij')
    
    # Convert to spherical coordinates
    theta = (90 - LAT) * np.pi / 180  # Colatitude in radians
    phi = LON * np.pi / 180  # Longitude in radians
    
    # Create a simplified dipole-like field
    # Radial component (strongest at poles, weaker at equator)
    br = 2.0 * np.cos(theta) / R**3
    
    # Add some higher-order multipole components
    br += 0.5 * np.sin(2 * theta) * np.cos(2 * phi) / R**3
    br += 0.3 * np.sin(theta)**2 * np.cos(4 * phi) / R**3
    
    # Theta component (meridional flow)
    bth = np.sin(theta) / R**3
    bth += 0.3 * np.cos(2 * theta) * np.cos(2 * phi) / R**3
    
    # Phi component (azimuthal flow)
    bph = 0.2 * np.sin(2 * phi) / R**2
    bph += 0.1 * np.sin(theta) * np.sin(4 * phi) / R**2
    
    # Create the spherical field data structure
    sph_data = SphericalFieldData()
    sph_data.set_coordinate_arrays(lon, lat, rix)
    sph_data.set_vector_field(br, bth, bph)
    
    print(f"Created vector field with dimensions: {br.shape}")
    
    return sph_data


def demonstrate_field_tracing(sph_data: SphericalFieldData):
    """
    Demonstrate fieldline tracing functionality.
    
    Parameters:
    -----------
    sph_data : SphericalFieldData
        The spherical field data structure
    """
    
    print("Demonstrating fieldline tracing...")
    
    try:
        # Choose starting points for fieldlines
        # fieldtype=5 means uniform grid, spacing=10 controls density
        print("Setting up fieldline starting coordinates...")
        spherical_field_start_coord(sph_data, fieldtype=5, spacing=10, radstart=1.5)
        
        # Trace fieldlines through the data
        print("Tracing fieldlines through the vector field...")
        spherical_trace_field(sph_data)
        
        print(f"Successfully traced {len(sph_data.nstep)} fieldlines")
        
        # Show some statistics
        valid_lines = sph_data.nstep[sph_data.nstep > 0]
        if len(valid_lines) > 0:
            print(f"Valid fieldlines: {len(valid_lines)}")
            print(f"Average line length: {np.mean(valid_lines):.1f} points")
            print(f"Max line length: {np.max(valid_lines)} points")
        
    except Exception as e:
        print(f"Note: Fieldline tracing not fully implemented yet: {e}")
        print("This demonstrates the intended workflow")


def demonstrate_visualization(sph_data: SphericalFieldData):
    """
    Demonstrate field visualization.
    
    Parameters:
    -----------
    sph_data : SphericalFieldData
        The spherical field data structure
    """
    
    print("Demonstrating field visualization...")
    
    try:
        # Create a simple visualization using matplotlib
        # This replaces the IDL spherical_draw_field functionality
        
        # Get the radial component at the inner boundary for visualization
        br_inner = sph_data.br[:, :, 0]  # (nlon, nlat)
        
        # Create image data structure
        im_data = spherical_image_create(br_inner, sph_data.lon, sph_data.lat)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Radial field at inner boundary
        im1 = ax1.imshow(br_inner.T, extent=[0, 360, -90, 90], 
                        origin='lower', aspect='auto', cmap='RdBu_r')
        ax1.set_title('Radial Field at Inner Boundary')
        ax1.set_xlabel('Longitude (degrees)')
        ax1.set_ylabel('Latitude (degrees)')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(im1, ax=ax1, label='Br')
        
        # Plot 2: Field magnitude at a middle radius
        mid_r_idx = sph_data.nr // 2
        br_mid = sph_data.br[:, :, mid_r_idx]
        bth_mid = sph_data.bth[:, :, mid_r_idx]
        bph_mid = sph_data.bph[:, :, mid_r_idx]
        
        bmag = np.sqrt(br_mid**2 + bth_mid**2 + bph_mid**2)
        
        im2 = ax2.imshow(bmag.T, extent=[0, 360, -90, 90], 
                        origin='lower', aspect='auto', cmap='plasma')
        ax2.set_title(f'Field Magnitude at r={sph_data.rix[mid_r_idx]:.2f}')
        ax2.set_xlabel('Longitude (degrees)')
        ax2.set_ylabel('Latitude (degrees)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(im2, ax=ax2, label='|B|')
        
        plt.tight_layout()
        plt.show()
        
        print("Visualization complete")
        
    except Exception as e:
        print(f"Note: Visualization not fully implemented yet: {e}")
        print("This demonstrates the intended workflow")


def demonstrate_interactive_viewer(sph_data: SphericalFieldData):
    """
    Demonstrate interactive field viewer (placeholder).
    
    Parameters:
    -----------
    sph_data : SphericalFieldData
        The spherical field data structure
    """
    
    print("Interactive viewer demonstration...")
    print("Note: Interactive 3D viewer (spherical_trackball_widget) would be launched here")
    print("This would provide:")
    print("  - Click and drag left mouse button to rotate")
    print("  - Click and drag right mouse button to zoom")
    print("  - Interactive 3D visualization of fieldlines")
    print("  - Export capabilities for images")
    
    # In the full implementation, this would launch an interactive widget
    # For now, we'll create a simple 3D plot as a placeholder
    
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the spherical boundaries
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_inner = np.outer(np.cos(u), np.sin(v))
        y_inner = np.outer(np.sin(u), np.sin(v))
        z_inner = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Inner boundary
        ax.plot_surface(x_inner, y_inner, z_inner, alpha=0.3, color='blue')
        
        # Outer boundary
        r_outer = sph_data.rix[-1]
        ax.plot_surface(r_outer * x_inner, r_outer * y_inner, r_outer * z_inner, 
                       alpha=0.2, color='red')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Field Domain (Interactive viewer placeholder)')
        
        plt.show()
        
    except ImportError:
        print("3D plotting not available - install matplotlib with 3D support")


def main():
    """
    Main demonstration function.
    """
    
    print("=== Spherical Field Tools Demonstration ===")
    print()
    
    # Step 1: Create sample vector field data
    sph_data = create_sample_pfss_data()
    print(sph_data)
    print()
    
    # Step 2: Demonstrate fieldline tracing
    demonstrate_field_tracing(sph_data)
    print()
    
    # Step 3: Demonstrate visualization
    demonstrate_visualization(sph_data)
    print()
    
    # Step 4: Demonstrate interactive viewer
    demonstrate_interactive_viewer(sph_data)
    print()
    
    print("=== Demonstration Complete ===")
    print()
    print("In the full implementation, this workflow would:")
    print("1. Load real PFSS data from solar magnetogram")
    print("2. Trace fieldlines through the 3D vector field")
    print("3. Render fieldlines with proper 3D visualization")
    print("4. Provide interactive manipulation and export")


if __name__ == "__main__":
    main()