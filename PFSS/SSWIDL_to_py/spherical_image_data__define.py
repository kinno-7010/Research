"""
Spherical Image Data Structure Definition

Creates a class for spherical_image_data, containing fields for
spherical image data (i.e. data mapped on a spherical surface) and its
associated metadata.

OUTPUTS:
    Defines a class with the following fields:
      image: two-dimensional data, indexed (lon,lat), typically binary
      rad: the radius of the spherical surface
      nlat: number of gridpoints in latitudinal (theta) dimension
      nlon: number of gridpoints in longitudinal (phi) dimension
      lat: index array for meridional dimension, in degrees
            between [0,180]
      lon: index array for zonal dimension, in degrees
            between [0,360]
      theta: index array for theta dimension, should be set to 
              (90-lat)*pi/180.
      phi: index array for phi dimension, should be set to lon*pi/180.
      palette: color palette/colormap for the image (optional)

CALLING SEQUENCE:
    data = SphericalImageData()

NOTES:
    1.  The data needs to be on a rectangular (i.e., longitude-latitude)
        grid, but the grid spacing does not need to be uniform as long as
        both the lon,lat fields each increase monotonically.

MODIFICATION HISTORY:
    M.DeRosa - 14 Dec 2005 - created (IDL version)
               13 Jul 2007 - added palette tag to structure definition
    Converted to Python - 2025
"""

import numpy as np
from typing import Optional, Union
import matplotlib.colors as mcolors


class SphericalImageData:
    """
    Container class for spherical image data.
    
    This class holds image data mapped on a spherical surface along with
    the associated coordinate arrays and metadata.
    """
    
    def __init__(self):
        """Initialize a SphericalImageData structure with default values."""
        
        # Image data
        self.image: Optional[np.ndarray] = None   # 2D image data (lon, lat)
        self.rad: float = 1.0                     # radius of spherical surface
        
        # Grid dimensions
        self.nlat: int = -1        # number of gridpoints in latitudinal dimension
        self.nlon: int = -1        # number of gridpoints in longitudinal dimension
        
        # Coordinate arrays
        self.theta: Optional[np.ndarray] = None   # theta array (radians)
        self.phi: Optional[np.ndarray] = None     # phi array (radians)
        self.lat: Optional[np.ndarray] = None     # latitude array (degrees)
        self.lon: Optional[np.ndarray] = None     # longitude array (degrees)
        
        # Color palette
        self.palette: Optional[mcolors.Colormap] = None
        
    def __repr__(self) -> str:
        """Return a string representation of the SphericalImageData object."""
        info = []
        info.append(f"SphericalImageData:")
        info.append(f"  Grid dimensions: nlon={self.nlon}, nlat={self.nlat}")
        info.append(f"  Radius: {self.rad}")
        
        if self.image is not None:
            info.append(f"  Image shape: {self.image.shape}")
            info.append(f"  Image range: [{np.min(self.image):.3f}, {np.max(self.image):.3f}]")
        else:
            info.append(f"  Image: not defined")
            
        if self.lon is not None:
            info.append(f"  Longitude range: [{np.min(self.lon):.1f}, {np.max(self.lon):.1f}] degrees")
        
        if self.lat is not None:
            info.append(f"  Latitude range: [{np.min(self.lat):.1f}, {np.max(self.lat):.1f}] degrees")
            
        if self.palette is not None:
            info.append(f"  Palette: {type(self.palette).__name__}")
        
        return "\n".join(info)
    
    def set_image_data(self, image: np.ndarray):
        """Set the image data and update grid dimensions."""
        self.image = np.asarray(image)
        
        if self.image.ndim != 2:
            raise ValueError("Image data must be 2-dimensional")
            
        self.nlon, self.nlat = self.image.shape
        
    def set_coordinate_arrays(self, lon: np.ndarray, lat: np.ndarray):
        """Set the coordinate arrays and compute theta, phi arrays."""
        self.lon = np.asarray(lon)
        self.lat = np.asarray(lat)
        
        # Compute theta and phi arrays
        self.theta = (90.0 - self.lat) * np.pi / 180.0
        self.phi = self.lon * np.pi / 180.0
        
        # Update grid dimensions
        self.nlon = len(self.lon)
        self.nlat = len(self.lat)
        
    def set_radius(self, radius: float):
        """Set the radius of the spherical surface."""
        self.rad = float(radius)
        
    def set_palette(self, palette: Union[str, mcolors.Colormap]):
        """Set the color palette for the image."""
        if isinstance(palette, str):
            self.palette = mcolors.get_cmap(palette)
        else:
            self.palette = palette
            
    def get_longitude_range(self) -> tuple:
        """Get the longitude range in degrees."""
        if self.lon is not None:
            return float(np.min(self.lon)), float(np.max(self.lon))
        return None, None
        
    def get_latitude_range(self) -> tuple:
        """Get the latitude range in degrees."""
        if self.lat is not None:
            return float(np.min(self.lat)), float(np.max(self.lat))
        return None, None
        
    def get_theta_range(self) -> tuple:
        """Get the theta range in radians."""
        if self.theta is not None:
            return float(np.min(self.theta)), float(np.max(self.theta))
        return None, None
        
    def get_phi_range(self) -> tuple:
        """Get the phi range in radians."""
        if self.phi is not None:
            return float(np.min(self.phi)), float(np.max(self.phi))
        return None, None
        
    def get_image_statistics(self) -> dict:
        """Get basic statistics about the image data."""
        if self.image is None:
            return {}
            
        return {
            'min': float(np.min(self.image)),
            'max': float(np.max(self.image)),
            'mean': float(np.mean(self.image)),
            'std': float(np.std(self.image)),
            'shape': self.image.shape
        }
        
    def validate_consistency(self) -> bool:
        """Check if the coordinate arrays are consistent with image dimensions."""
        if self.image is None:
            return False
            
        image_nlon, image_nlat = self.image.shape
        
        if self.lon is not None and len(self.lon) != image_nlon:
            return False
            
        if self.lat is not None and len(self.lat) != image_nlat:
            return False
            
        return True
        
    def copy(self):
        """Create a copy of the SphericalImageData object."""
        new_obj = SphericalImageData()
        
        if self.image is not None:
            new_obj.image = self.image.copy()
        new_obj.rad = self.rad
        new_obj.nlat = self.nlat
        new_obj.nlon = self.nlon
        
        if self.theta is not None:
            new_obj.theta = self.theta.copy()
        if self.phi is not None:
            new_obj.phi = self.phi.copy()
        if self.lat is not None:
            new_obj.lat = self.lat.copy()
        if self.lon is not None:
            new_obj.lon = self.lon.copy()
            
        new_obj.palette = self.palette
        
        return new_obj


# Factory function to create a SphericalImageData instance (for compatibility)
def spherical_image_data():
    """Factory function to create a SphericalImageData instance."""
    return SphericalImageData()


if __name__ == "__main__":
    # Example usage
    im_data = SphericalImageData()
    
    # Set up a simple example
    nlon, nlat = 360, 180
    lon = np.linspace(0, 360, nlon, endpoint=False)
    lat = np.linspace(-90, 90, nlat)
    
    im_data.set_coordinate_arrays(lon, lat)
    
    # Create example image data (simple pattern)
    image = np.sin(np.radians(lon)[:, np.newaxis]) * np.cos(np.radians(lat)[np.newaxis, :])
    im_data.set_image_data(image)
    
    im_data.set_radius(1.5)
    im_data.set_palette('viridis')
    
    print(im_data)
    print(f"Image statistics: {im_data.get_image_statistics()}")
    print(f"Validation: {im_data.validate_consistency()}")