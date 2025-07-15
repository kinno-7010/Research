"""
Spherical Texture Map Create

This function creates a texture-mapped spherical surface object from
spherically gridded data.

CALLING SEQUENCE:
    result = spherical_texmap_create(sphim_data, radius, imsc=None, lonbounds=None)

INPUTS:
    sphim_data = a SphericalImageData object
    radius = radius of spherical surface onto which the image is to be mapped
    imsc = data value(s) to which to scale central image; if only
           one value is given then range is [-value,+value] (default = 
           [-image_max,+image_max], centered around 0)
    lonbounds = array of bounds [lonmin,lonmax], in degrees between 0 and 360,
                defining data that is bounded in longitude (otherwise, the data
                is assumed to wrap around)

OUTPUTS:
    result = dictionary containing texture map data for 3D visualization

NOTES:
    - Assumes phi data is monotonically increasing
    - This is a simplified version that returns data structures instead of IDL graphics objects
    - For full 3D visualization, integrate with matplotlib/mayavi/vtk

MODIFICATION HISTORY:
    M.DeRosa - 23 Aug 2006 - created (IDL version)
    Converted to Python - 2025 (simplified version)
"""

import numpy as np
from typing import Optional, List, Dict, Any
from spherical_image_data__define import SphericalImageData


def spherical_texmap_create(sphim_data: SphericalImageData,
                           radius: float,
                           imsc: Optional[float] = None,
                           lonbounds: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Create a texture-mapped spherical surface from spherical image data.
    
    Parameters:
    -----------
    sphim_data : SphericalImageData
        Spherical image data structure
    radius : float
        Radius of spherical surface for texture mapping
    imsc : float, optional
        Image scaling value
    lonbounds : list, optional
        Longitude bounds [lonmin, lonmax] in degrees
        
    Returns:
    --------
    dict
        Dictionary containing texture map data:
        - 'vertices': 3D vertex coordinates
        - 'faces': Face connectivity
        - 'texture_coords': Texture coordinates
        - 'image_data': Scaled image data
        - 'bounded': Whether data is bounded in longitude
    """
    
    # Input validation
    if sphim_data is None or sphim_data.image is None:
        raise ValueError("ERROR in spherical_texmap_create: invalid input data")
    
    if radius <= 0:
        raise ValueError("ERROR in spherical_texmap_create: radius must be positive")
    
    # Deal with phi bounds
    bounded = False
    if lonbounds is not None and len(lonbounds) >= 2:
        if lonbounds[0] >= 0 and lonbounds[1] != 0:
            ph1 = lonbounds[0] * np.pi / 180
            ph2 = lonbounds[1] * np.pi / 180
            bounded = True
    
    # Deal with theta bounds
    thmin, thmax = np.min(sphim_data.theta), np.max(sphim_data.theta)
    
    # If global data, replicate last longitude to avoid seam
    imd2 = sphim_data.copy()
    if not bounded:
        # Add periodic boundary
        image_extended = np.concatenate([sphim_data.image, sphim_data.image[0:1, :]], axis=0)
        lon_extended = np.concatenate([sphim_data.lon, [360 + sphim_data.lon[0]]])
        phi_extended = np.concatenate([sphim_data.phi, [2*np.pi + sphim_data.phi[0]]])
        
        imd2.set_image_data(image_extended)
        imd2.lon = lon_extended
        imd2.phi = phi_extended
        imd2.nlon = len(lon_extended)
    
    # Get byte image and scale it
    if imsc is None:
        imsc = np.max(np.abs(sphim_data.image))
    
    # Scale image to [0, 255] range
    scaled_image = np.clip((imd2.image + imsc) / (2 * imsc) * 255, 0, 255).astype(np.uint8)
    
    # Create vertex list for polygonal spherical surface
    phi_grid, theta_grid = np.meshgrid(imd2.phi, imd2.theta, indexing='ij')
    
    # Convert to 3D cartesian coordinates
    # Note: theta is colatitude, phi is longitude
    x = radius * np.sin(theta_grid) * np.cos(phi_grid)
    y = radius * np.sin(theta_grid) * np.sin(phi_grid)
    z = radius * np.cos(theta_grid)
    
    # Flatten coordinates for vertex array
    vertices = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    
    # Create connectivity array (faces)
    faces = []
    for i in range(imd2.nlat - 1):
        for j in range(imd2.nlon - 1):
            # Four vertices of a quad
            v0 = i * imd2.nlon + j
            v1 = i * imd2.nlon + (j + 1)
            v2 = (i + 1) * imd2.nlon + (j + 1)
            v3 = (i + 1) * imd2.nlon + j
            
            # Create two triangles from the quad
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    
    faces = np.array(faces)
    
    # Determine texture coordinates
    if bounded:
        # For bounded data, map to [0,1] range within bounds
        u_coords = (phi_grid - ph1) / (ph2 - ph1)
        v_coords = (theta_grid - thmin) / (thmax - thmin)
    else:
        # For global data, map phi to [0,1] and theta to [0,1]
        u_coords = phi_grid / (2 * np.pi)
        v_coords = (theta_grid - thmin) / (thmax - thmin)
    
    # Flatten texture coordinates
    texture_coords = np.column_stack([u_coords.flatten(), v_coords.flatten()])
    
    # Create result dictionary
    result = {
        'vertices': vertices,
        'faces': faces,
        'texture_coords': texture_coords,
        'image_data': scaled_image,
        'bounded': bounded,
        'radius': radius,
        'phi_range': (np.min(imd2.phi), np.max(imd2.phi)),
        'theta_range': (thmin, thmax),
        'dimensions': (imd2.nlon, imd2.nlat)
    }
    
    return result


def plot_spherical_texture_map(texmap_data: Dict[str, Any], 
                              show_wireframe: bool = False,
                              elevation: float = 30,
                              azimuth: float = 45):
    """
    Plot the spherical texture map using matplotlib.
    
    Parameters:
    -----------
    texmap_data : dict
        Output from spherical_texmap_create
    show_wireframe : bool, optional
        Whether to show wireframe
    elevation : float, optional
        Viewing elevation angle
    azimuth : float, optional
        Viewing azimuth angle
    """
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        vertices = texmap_data['vertices']
        faces = texmap_data['faces']
        image_data = texmap_data['image_data']
        
        # Create a simple surface plot
        nlon, nlat = texmap_data['dimensions']
        x = vertices[:, 0].reshape(nlat, nlon)
        y = vertices[:, 1].reshape(nlat, nlon)
        z = vertices[:, 2].reshape(nlat, nlon)
        
        # Plot surface with color mapping
        surf = ax.plot_surface(x, y, z, 
                              facecolors=plt.cm.RdBu_r(image_data.T/255),
                              alpha=0.8, linewidth=0, antialiased=True)
        
        if show_wireframe:
            ax.plot_wireframe(x, y, z, alpha=0.3, color='gray', linewidth=0.5)
        
        # Set equal aspect ratio
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set viewing angle
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Set title
        radius = texmap_data['radius']
        bounded = texmap_data['bounded']
        bound_str = "bounded" if bounded else "global"
        ax.set_title(f'Spherical Texture Map (r={radius:.2f}, {bound_str})')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not available for plotting")
        print("Texture map data structure created successfully")


if __name__ == "__main__":
    # Example usage
    from spherical_image_create import spherical_image_create
    
    # Create sample image data
    nlon, nlat = 72, 36  # 5-degree resolution
    lon = np.linspace(0, 360, nlon, endpoint=False)
    lat = np.linspace(-90, 90, nlat)
    
    # Create a simple test pattern
    LON, LAT = np.meshgrid(lon, lat, indexing='ij')
    
    # Create a pattern with latitude bands and longitude variation
    data = np.sin(np.radians(LAT)) * np.cos(4 * np.radians(LON))
    
    # Create spherical image data
    sphim_data = spherical_image_create(data, lon, lat, radius=1.0)
    
    # Create texture map
    texmap_data = spherical_texmap_create(sphim_data, radius=1.5)
    
    print("Texture map created successfully:")
    print(f"  Vertices shape: {texmap_data['vertices'].shape}")
    print(f"  Faces shape: {texmap_data['faces'].shape}")
    print(f"  Texture coords shape: {texmap_data['texture_coords'].shape}")
    print(f"  Image data shape: {texmap_data['image_data'].shape}")
    print(f"  Bounded: {texmap_data['bounded']}")
    print(f"  Radius: {texmap_data['radius']}")
    
    # Plot the texture map
    plot_spherical_texture_map(texmap_data, show_wireframe=True)
    
    # Test with bounded data
    print("\nTesting with bounded data:")
    bounded_texmap = spherical_texmap_create(sphim_data, radius=2.0, 
                                           lonbounds=[0, 180])
    print(f"  Bounded texture map created: {bounded_texmap['bounded']}")
    
    # Example of how to use the texture map data with other 3D libraries
    print("\nTexture map data can be used with:")
    print("  - VTK/PyVista for advanced 3D visualization")
    print("  - Mayavi for scientific visualization")
    print("  - Three.js for web-based 3D graphics")
    print("  - OpenGL for high-performance rendering")