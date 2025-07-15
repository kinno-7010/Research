"""
Spherical Image Erode

This function erodes binary data mapped on a spherical surface using a
circular structural element (kernel).

CALLING SEQUENCE:
    result = spherical_image_erode(sph_image, radius)

INPUTS:
    sph_image = a SphericalImageData object with all fields defined.
                If sph_image.image is not binary, then generate a binary
                by doing a abs(sph_image.image) < 1.
    radius = the radius of the circular erosion kernel in degrees

OUTPUTS:
    result = a SphericalImageData object containing the eroded image.
             This image is a binary image.

NOTES:
    1.  Future improvement: allow arrays of structures and/or radii to be
        passed into this routine.
    2.  Another future improvement: allow routine to work on grayscale images
        (i.e. assign to each point the minimum value of the surrounding
        pixels defined by the kernel)

MODIFICATION HISTORY:
    M.DeRosa - 15 Dec 2005 - created (IDL version)
    Converted to Python - 2025
"""

import numpy as np
from typing import Union
from spherical_image_data__define import SphericalImageData
from spherical_image_dilate import spherical_image_dilate


def spherical_image_erode(sph_image: SphericalImageData, 
                         radius: float) -> SphericalImageData:
    """
    Erode binary data mapped on a spherical surface using a circular kernel.
    
    Erosion is implemented by dilating the background (complement of the image)
    and then taking the complement of the result.
    
    Parameters:
    -----------
    sph_image : SphericalImageData
        Spherical image data structure with all fields defined
    radius : float
        Radius of the circular erosion kernel in degrees
        
    Returns:
    --------
    SphericalImageData
        A SphericalImageData object containing the eroded binary image
        
    Raises:
    -------
    ValueError
        If input parameters are invalid
    """
    
    # Input validation
    if sph_image is None:
        raise ValueError("ERROR in spherical_image_erode: no input data provided")
    
    if sph_image.image is None:
        raise ValueError("ERROR in spherical_image_erode: image data not defined")
    
    if radius <= 0:
        raise ValueError("ERROR in spherical_image_erode: negative erosion radius")
    
    # Get a binary input image (note: different condition from dilation)
    # For erosion, we consider values < 1 as background
    imin = np.abs(sph_image.image) < 1
    
    # Create a temporary image structure with the inverted image
    sph_image2 = sph_image.copy()
    sph_image2.set_image_data(imin.astype(np.uint8))
    
    # Dilate the background
    result = spherical_image_dilate(sph_image2, radius)
    
    # Invert the result to get the eroded foreground
    sph_eroded = sph_image.copy()
    eroded_data = 1 - result.image  # Complement of the dilated background
    sph_eroded.set_image_data(eroded_data)
    
    return sph_eroded


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    from spherical_image_create import spherical_image_create
    
    # Create sample data
    nlon, nlat = 72, 36  # 5-degree resolution
    lon = np.linspace(0, 360, nlon, endpoint=False)
    lat = np.linspace(-90, 90, nlat)
    
    # Create a binary image with some thick regions
    data = np.zeros((nlon, nlat))
    
    # Add some larger circular regions
    LON, LAT = np.meshgrid(lon, lat, indexing='ij')
    
    # Region 1: large region around equator and 0 longitude
    mask1 = (np.abs(LAT) < 30) & (np.abs(LON) < 30)
    data[mask1] = 1
    
    # Region 2: large region around 45N, 180E
    mask2 = (np.abs(LAT - 45) < 20) & (np.abs(LON - 180) < 25)
    data[mask2] = 1
    
    # Region 3: smaller region around -30S, 90E
    mask3 = (np.abs(LAT + 30) < 15) & (np.abs(LON - 90) < 18)
    data[mask3] = 1
    
    # Create spherical image data
    sph_image = spherical_image_create(data, lon, lat)
    
    # Erode the image
    erosion_radius = 10.0  # degrees
    sph_eroded = spherical_image_erode(sph_image, erosion_radius)
    
    # Also create a dilated version for comparison
    sph_dilated = spherical_image_dilate(sph_image, erosion_radius)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    im1 = ax1.imshow(sph_image.image.T, extent=[0, 360, -90, 90], 
                     origin='lower', aspect='auto', cmap='binary')
    ax1.set_title('Original Binary Image')
    ax1.set_xlabel('Longitude (degrees)')
    ax1.set_ylabel('Latitude (degrees)')
    ax1.grid(True, alpha=0.3)
    
    # Eroded image
    im2 = ax2.imshow(sph_eroded.image.T, extent=[0, 360, -90, 90], 
                     origin='lower', aspect='auto', cmap='binary')
    ax2.set_title(f'Eroded Image (radius={erosion_radius}°)')
    ax2.set_xlabel('Longitude (degrees)')
    ax2.set_ylabel('Latitude (degrees)')
    ax2.grid(True, alpha=0.3)
    
    # Dilated image (for comparison)
    im3 = ax3.imshow(sph_dilated.image.T, extent=[0, 360, -90, 90], 
                     origin='lower', aspect='auto', cmap='binary')
    ax3.set_title(f'Dilated Image (radius={erosion_radius}°)')
    ax3.set_xlabel('Longitude (degrees)')
    ax3.set_ylabel('Latitude (degrees)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Original image statistics:")
    print(sph_image.get_image_statistics())
    print(f"Original active pixels: {np.sum(sph_image.image > 0)}")
    
    print("\nEroded image statistics:")
    print(sph_eroded.get_image_statistics())
    print(f"Eroded active pixels: {np.sum(sph_eroded.image > 0)}")
    
    print("\nDilated image statistics:")
    print(sph_dilated.get_image_statistics())
    print(f"Dilated active pixels: {np.sum(sph_dilated.image > 0)}")
    
    # Show the effect of erosion
    print(f"\nErosion reduced active pixels by: {np.sum(sph_image.image > 0) - np.sum(sph_eroded.image > 0)}")
    print(f"Dilation increased active pixels by: {np.sum(sph_dilated.image > 0) - np.sum(sph_image.image > 0)}")