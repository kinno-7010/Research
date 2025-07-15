"""
Spherical Field Data Structure Definition

Creates a class for spherical_field_data, containing fields for
spherical vector field data, fieldline trajectories, and its associated
metadata.

OUTPUTS:
    Defines a class with the following fields:
      br: r-component of vector field, indexed (lon,lat,r)
      bth: theta-component of vector field, indexed (lon,lat,r)
      bph: phi-component of vector field, indexed (lon,lat,r)
      bderivs: structure/dict of derivative arrays
      nr: number of gridpoints in r dimension
      nlat: number of gridpoints in latitudinal (theta) dimension
      nlon: number of gridpoints in longitudinal (phi) dimension
      rix: index array for r dimension
      lat: index array for meridional dimension, in degrees between [0,180]
      lon: index array for zonal dimension, in degrees between [0,360]
      lonbounds: array of bounds [lonmin,lonmax], in degrees between 0 and
                360, defining data that is bounded in longitude. The
                convention is if lonbounds[0] is <0 or undefined, then it is
                assumed that the data span all longitudes. Note that this
                means that the default setting of lonbounds=[0,0] indicates
                that the field is bounded, but since lonmin=lonmax this may
                cause some problems with some routines.
      theta: index array for theta dimension, should be set to
              (90-lat)*pi/180.
      phi: index array for phi dimension, should be set to lon*pi/180.
      str: n-element array for r coordinate of fieldline start points
      stth: n-element array for theta coordinate of fieldline start points
      stph: n-element array for phi coordinate of fieldline start points
      ptr: (n,stepmax)-element array of r coordinates of fieldlines
      ptth: (n,stepmax)-element array of theta coordinates of fieldlines
      ptph: (n,stepmax)-element array of phi coordinates of fieldlines
      nstep: n-element array of fieldline lengths
      extra_objects: array of extra objects added to the view for this field
                      when rendered by the spherical_trackball_widget or
                      spherical_draw_field routines 

CALLING SEQUENCE:
    data = SphericalFieldData()

NOTES:
    1.  In the above, n is the number of fieldlines, stepmax is the length in
        points of the longest fieldline.
    2.  The data needs to be on a longitude-latitude-radius grid, but the
        grid spacing does not need to be uniform as long as the lon,lat, and
        rix fields each increase monotonically.

MODIFICATION HISTORY:
    M.DeRosa - 13 Dec 2005 - created (IDL version)
               24 Jan 2006 - added lonbounds tag
               28 Jan 2011 - added bderivs tag
               25 Apr 2011 - added extra_objects tag
               23 Mar 2016 - corrected language in description above
    Converted to Python - 2025
"""

import numpy as np
from typing import Optional, Union, Dict, Any


class SphericalFieldData:
    """
    Container class for spherical vector field data and fieldline trajectories.
    
    This class holds magnetic field data on a spherical grid along with
    fieldline trajectory information and associated metadata.
    """
    
    def __init__(self):
        """Initialize a SphericalFieldData structure with default values."""
        
        # Vector field components
        self.br: Optional[np.ndarray] = None      # r-component of vector field
        self.bth: Optional[np.ndarray] = None     # theta-component of vector field  
        self.bph: Optional[np.ndarray] = None     # phi-component of vector field
        self.bderivs: Optional[Dict[str, np.ndarray]] = None  # derivative arrays
        
        # Grid dimensions
        self.nr: int = -1          # number of gridpoints in r dimension
        self.nlat: int = -1        # number of gridpoints in latitudinal dimension
        self.nlon: int = -1        # number of gridpoints in longitudinal dimension
        
        # Index arrays
        self.rix: Optional[np.ndarray] = None     # r index array
        self.theta: Optional[np.ndarray] = None   # theta index array (radians)
        self.phi: Optional[np.ndarray] = None     # phi index array (radians)
        self.lat: Optional[np.ndarray] = None     # latitude array (degrees)
        self.lon: Optional[np.ndarray] = None     # longitude array (degrees)
        
        # Longitude bounds
        self.lonbounds: np.ndarray = np.array([-1.0, -1.0])  # longitude bounds
        
        # Fieldline starting points
        self.str: Optional[np.ndarray] = None     # r coordinates of start points
        self.stth: Optional[np.ndarray] = None    # theta coordinates of start points
        self.stph: Optional[np.ndarray] = None    # phi coordinates of start points
        
        # Fieldline trajectories
        self.ptr: Optional[np.ndarray] = None     # r coordinates of trajectories
        self.ptth: Optional[np.ndarray] = None    # theta coordinates of trajectories
        self.ptph: Optional[np.ndarray] = None    # phi coordinates of trajectories
        self.nstep: Optional[np.ndarray] = None   # number of steps in each line
        
        # Extra objects for rendering
        self.extra_objects: Optional[list] = None
        
    def __repr__(self) -> str:
        """Return a string representation of the SphericalFieldData object."""
        info = []
        info.append(f"SphericalFieldData:")
        info.append(f"  Grid dimensions: nr={self.nr}, nlat={self.nlat}, nlon={self.nlon}")
        
        if self.br is not None:
            info.append(f"  Vector field shapes: br={self.br.shape}")
        else:
            info.append(f"  Vector field: not defined")
            
        if self.str is not None:
            info.append(f"  Starting points: {len(self.str)} points")
        else:
            info.append(f"  Starting points: not defined")
            
        if self.nstep is not None:
            info.append(f"  Fieldlines: {len(self.nstep)} lines")
        else:
            info.append(f"  Fieldlines: not traced")
            
        info.append(f"  Longitude bounds: {self.lonbounds}")
        
        return "\n".join(info)
    
    def set_grid_dimensions(self, nr: int, nlat: int, nlon: int):
        """Set the grid dimensions."""
        self.nr = nr
        self.nlat = nlat
        self.nlon = nlon
        
    def set_vector_field(self, br: np.ndarray, bth: np.ndarray, bph: np.ndarray):
        """Set the vector field components."""
        self.br = np.asarray(br)
        self.bth = np.asarray(bth)
        self.bph = np.asarray(bph)
        
        # Update grid dimensions from field shape
        if self.br.ndim == 3:
            self.nlon, self.nlat, self.nr = self.br.shape
        
    def set_coordinate_arrays(self, lon: np.ndarray, lat: np.ndarray, rix: np.ndarray):
        """Set the coordinate arrays and compute theta, phi arrays."""
        self.lon = np.asarray(lon)
        self.lat = np.asarray(lat)
        self.rix = np.asarray(rix)
        
        # Compute theta and phi arrays
        self.theta = (90.0 - self.lat) * np.pi / 180.0
        self.phi = self.lon * np.pi / 180.0
        
        # Update grid dimensions
        self.nlon = len(self.lon)
        self.nlat = len(self.lat)
        self.nr = len(self.rix)
        
    def set_longitude_bounds(self, lonmin: float, lonmax: float):
        """Set the longitude bounds for bounded data."""
        self.lonbounds = np.array([lonmin, lonmax])
        
    def set_starting_points(self, str_coords: np.ndarray, stth_coords: np.ndarray, 
                           stph_coords: np.ndarray):
        """Set the fieldline starting points."""
        self.str = np.asarray(str_coords)
        self.stth = np.asarray(stth_coords)
        self.stph = np.asarray(stph_coords)
        
    def set_fieldline_trajectories(self, ptr: np.ndarray, ptth: np.ndarray, 
                                 ptph: np.ndarray, nstep: np.ndarray):
        """Set the fieldline trajectory data."""
        self.ptr = np.asarray(ptr)
        self.ptth = np.asarray(ptth)
        self.ptph = np.asarray(ptph)
        self.nstep = np.asarray(nstep)
        
    def is_bounded_in_longitude(self) -> bool:
        """Check if the data is bounded in longitude."""
        return self.lonbounds[0] >= 0
        
    def get_radial_range(self) -> tuple:
        """Get the radial range of the grid."""
        if self.rix is not None:
            return float(np.min(self.rix)), float(np.max(self.rix))
        return None, None
        
    def get_theta_range(self) -> tuple:
        """Get the theta range of the grid."""
        if self.theta is not None:
            return float(np.min(self.theta)), float(np.max(self.theta))
        return None, None
        
    def get_phi_range(self) -> tuple:
        """Get the phi range of the grid."""
        if self.phi is not None:
            return float(np.min(self.phi)), float(np.max(self.phi))
        return None, None


# Factory function to create a SphericalFieldData instance (for compatibility)
def spherical_field_data():
    """Factory function to create a SphericalFieldData instance."""
    return SphericalFieldData()


if __name__ == "__main__":
    # Example usage
    sph_data = SphericalFieldData()
    
    # Set up a simple example grid
    nr, nlat, nlon = 10, 20, 30
    sph_data.set_grid_dimensions(nr, nlat, nlon)
    
    # Create example coordinate arrays
    lon = np.linspace(0, 360, nlon, endpoint=False)
    lat = np.linspace(-90, 90, nlat)
    rix = np.linspace(1.0, 2.5, nr)
    
    sph_data.set_coordinate_arrays(lon, lat, rix)
    
    # Create example vector field
    br = np.random.random((nlon, nlat, nr))
    bth = np.random.random((nlon, nlat, nr))
    bph = np.random.random((nlon, nlat, nr))
    
    sph_data.set_vector_field(br, bth, bph)
    
    print(sph_data)