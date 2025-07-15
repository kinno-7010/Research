"""
Spherical Image Create

This function creates a SphericalImageData object, containing
data spanning a spherical surface and its associated index arrays.

CALLING SEQUENCE:
    result = spherical_image_create(data, lon, lat, radius=1.0, palette=None)

INPUTS:
    data = a two-dimensional image gridded in longitude and latitude
    lon = index array defining the longitudes of the image [in degrees]
    lat = index array defining the latitudes of the image [in degrees]
    radius = (optional) radius of the spherical surface (default=1.0)
    palette = (optional) colormap for the image

OUTPUTS:
    result = a SphericalImageData object with all fields defined

MODIFICATION HISTORY:
    M.DeRosa - 23 Aug 2006 - created (IDL version)
               27 Jun 2007 - added optional radius argument
               13 Jul 2007 - added optional palette keyword
    Converted to Python - 2025
"""

import numpy as np
from typing import Optional, Union
import matplotlib.colors as mcolors
from spherical_image_data__define import SphericalImageData


def spherical_image_create(data: np.ndarray, 
                          lon: np.ndarray, 
                          lat: np.ndarray, 
                          radius: float = 1.0,
                          palette: Optional[Union[str, mcolors.Colormap]] = None) -> SphericalImageData:
    """
    Create a SphericalImageData object from image data and coordinate arrays.
    
    Parameters:
    -----------
    data : np.ndarray
        Two-dimensional image data gridded in longitude and latitude
    lon : np.ndarray
        Index array defining the longitudes of the image [in degrees]
    lat : np.ndarray
        Index array defining the latitudes of the image [in degrees]
    radius : float, optional
        Radius of the spherical surface (default=1.0)
    palette : str or matplotlib.colors.Colormap, optional
        Color palette/colormap for the image
        
    Returns:
    --------
    SphericalImageData
        A SphericalImageData object with all fields defined
        
    Raises:
    -------
    ValueError
        If input dimensions are inconsistent
    """
    
    # Input validation
    data = np.asarray(data)
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    
    if data.ndim != 2:
        raise ValueError("ERROR in spherical_image_create: data argument must have 2 dimensions")
    
    nlon, nlat = data.shape
    
    if len(lon) != nlon:
        raise ValueError("ERROR in spherical_image_create: length of lon array does not match data")
    
    if len(lat) != nlat:
        raise ValueError("ERROR in spherical_image_create: length of lat array does not match data")
    
    # Create the SphericalImageData object
    im_data = SphericalImageData()
    
    # Set the image data
    im_data.set_image_data(data)
    
    # Set the coordinate arrays
    im_data.set_coordinate_arrays(lon, lat)
    
    # Set the radius
    im_data.set_radius(radius)
    
    # Set the palette if provided
    if palette is not None:
        im_data.set_palette(palette)
    
    return im_data


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create sample data
    nlon, nlat = 180, 90
    lon = np.linspace(0, 360, nlon, endpoint=False)
    lat = np.linspace(-90, 90, nlat)
    
    # Create a simple test pattern
    LON, LAT = np.meshgrid(lon, lat, indexing='ij')
    data = np.sin(np.radians(LON)) * np.cos(np.radians(LAT))
    
    # Create the spherical image data structure
    im_data = spherical_image_create(data, lon, lat, radius=1.5, palette='viridis')
    
    print("Created spherical image data:")
    print(im_data)
    
    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.imshow(data.T, extent=[0, 360, -90, 90], origin='lower', 
               aspect='auto', cmap='viridis')
    plt.colorbar(label='Data value')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.title('Example Spherical Image Data')
    plt.tight_layout()
    plt.show()
    
    print(f"Image statistics: {im_data.get_image_statistics()}")
    print(f"Longitude range: {im_data.get_longitude_range()}")
    print(f"Latitude range: {im_data.get_latitude_range()}")