"""
Spherical Image Dilate

This function dilates binary data mapped on a spherical surface using a
circular structural element (kernel).

CALLING SEQUENCE:
    result = spherical_image_dilate(sph_image, radius)

INPUTS:
    sph_image = a SphericalImageData object with all fields defined.
                If sph_image.image is not binary, then generate a binary
                by doing a abs(sph_image.image) >= 1.
    radius = the radius of the circular dilation kernel in degrees

OUTPUTS:
    result = a SphericalImageData object containing the dilated image.
             This image is a binary image.

NOTES:
    1.  Future improvement: allow arrays of structures and/or radii to be
        passed into this routine.
    2.  Another future improvement: allow routine to work on grayscale images
        (i.e. assign to each point the maximum value of the surrounding
        pixels defined by the kernel)

MODIFICATION HISTORY:
    M.DeRosa - 15 Dec 2005 - created (IDL version)
    Converted to Python - 2025
"""

import numpy as np
from typing import Union
from spherical_image_data__define import SphericalImageData


def spherical_image_dilate(sph_image: SphericalImageData, 
                          radius: float) -> SphericalImageData:
    """
    Dilate binary data mapped on a spherical surface using a circular kernel.
    
    Parameters:
    -----------
    sph_image : SphericalImageData
        Spherical image data structure with all fields defined
    radius : float
        Radius of the circular dilation kernel in degrees
        
    Returns:
    --------
    SphericalImageData
        A SphericalImageData object containing the dilated binary image
        
    Raises:
    -------
    ValueError
        If input parameters are invalid
    """
    
    # Input validation
    if sph_image is None:
        raise ValueError("ERROR in spherical_image_dilate: no input data provided")
    
    if sph_image.image is None:
        raise ValueError("ERROR in spherical_image_dilate: image data not defined")
    
    if radius <= 0:
        raise ValueError("ERROR in spherical_image_dilate: negative dilation radius")
    
    # Convert radius to radians
    rad = radius * np.pi / 180.0
    
    # Get a binary input image
    imin = np.abs(sph_image.image) >= 1
    
    # Get grid dimensions
    nlat = sph_image.nlat
    nlon = sph_image.nlon
    
    # Create grids holding the theta and phi values of each point
    phi_grid = sph_image.phi[:, np.newaxis]  # (nlon, 1)
    theta_grid = sph_image.theta[np.newaxis, :]  # (1, nlat)
    
    # Initialize result array
    result = np.zeros((nlon, nlat), dtype=bool)
    
    # Loop through each row (latitude) in input image
    for i in range(nlat):
        # Current latitude angle
        anga = theta_grid[0, i]
        
        # All theta values in the map
        angb = theta_grid[0, :]  # (nlat,)
        
        # Angle differences on surface between meridian lines
        # Broadcasting: phi_grid[0, i] is scalar, phi_grid is (nlon, 1)
        capc = sph_image.phi[0] - phi_grid[:, 0]  # (nlon,)
        
        # Compute spherical distances using great circle formula
        # Broadcasting: cos(anga) and cos(angb) are scalars/arrays
        cos_anga = np.cos(anga)
        cos_angb = np.cos(angb)  # (nlat,)
        sin_anga = np.sin(anga)
        sin_angb = np.sin(angb)  # (nlat,)
        
        # Create meshgrids for broadcasting
        cos_angb_grid = cos_angb[np.newaxis, :]  # (1, nlat)
        sin_angb_grid = sin_angb[np.newaxis, :]  # (1, nlat)
        capc_grid = capc[:, np.newaxis]  # (nlon, 1)
        
        # Compute spherical distance
        cos_sdist = (cos_anga * cos_angb_grid + 
                    sin_anga * sin_angb_grid * np.cos(capc_grid))
        
        # Clip to valid range for arccos
        cos_sdist = np.clip(cos_sdist, -1.0, 1.0)
        sdist = np.arccos(cos_sdist)
        
        # Create kernel: points within radius
        kernel = sdist <= rad
        
        # Do convolution for current latitude
        # Find all points in current latitude that are set in input image
        active_points = np.where(imin[:, i])[0]
        
        if len(active_points) > 0:
            for j in active_points:
                # Shift kernel to center on point j
                # Use circular shift for longitude (wrapping)
                shifted_kernel = np.roll(kernel, j, axis=0)
                result = result | shifted_kernel
    
    # Create output structure
    sph_dilated = sph_image.copy()
    sph_dilated.set_image_data(result.astype(np.uint8))
    
    return sph_dilated


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    from spherical_image_create import spherical_image_create
    
    # Create sample data
    nlon, nlat = 72, 36  # 5-degree resolution
    lon = np.linspace(0, 360, nlon, endpoint=False)
    lat = np.linspace(-90, 90, nlat)
    
    # Create a simple binary image with a few active regions
    data = np.zeros((nlon, nlat))
    
    # Add some circular regions
    LON, LAT = np.meshgrid(lon, lat, indexing='ij')
    
    # Region 1: around equator and 0 longitude
    mask1 = (np.abs(LAT) < 20) & (np.abs(LON) < 20)
    data[mask1] = 1
    
    # Region 2: around 45N, 180E
    mask2 = (np.abs(LAT - 45) < 10) & (np.abs(LON - 180) < 15)
    data[mask2] = 1
    
    # Region 3: around -30S, 90E
    mask3 = (np.abs(LAT + 30) < 8) & (np.abs(LON - 90) < 12)
    data[mask3] = 1
    
    # Create spherical image data
    sph_image = spherical_image_create(data, lon, lat)
    
    # Dilate the image
    dilation_radius = 15.0  # degrees
    sph_dilated = spherical_image_dilate(sph_image, dilation_radius)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    im1 = ax1.imshow(sph_image.image.T, extent=[0, 360, -90, 90], 
                     origin='lower', aspect='auto', cmap='binary')
    ax1.set_title('Original Binary Image')
    ax1.set_xlabel('Longitude (degrees)')
    ax1.set_ylabel('Latitude (degrees)')
    ax1.grid(True, alpha=0.3)
    
    # Dilated image
    im2 = ax2.imshow(sph_dilated.image.T, extent=[0, 360, -90, 90], 
                     origin='lower', aspect='auto', cmap='binary')
    ax2.set_title(f'Dilated Image (radius={dilation_radius}°)')
    ax2.set_xlabel('Longitude (degrees)')
    ax2.set_ylabel('Latitude (degrees)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Original image statistics:")
    print(sph_image.get_image_statistics())
    print("\nDilated image statistics:")
    print(sph_dilated.get_image_statistics())