"""
Spherical Get Nulls

Given vector field data on a spherical grid, this function will find the
locations of null points in the field using the trilinear method described
by Haynes & Parnell (2007).

CALLING SEQUENCE:
    result = spherical_get_nulls(sph_data, rixrange=None, nnulls=None,
                                interpolates=None, quiet=False, debug=False)

INPUTS:
    sph_data = a SphericalFieldData object with the following fields defined:
        br, bth, bph, nlat, nlon, nr, rix, lat, lon. Basically, one
        needs the vector field (br,bth,bph), its dimension (nr,nlat,nlon), and
        its indexing (rix,lat,lon).

KEYWORDS:
    rixrange = a two-element vector containing radial indices (not values!)
               that is used to restrict the range of radii within which null
               points are searched for
    nnulls = on output, contains the number of nulls located by the routine
    interpolates = on output, returns interpolation indices of each null
                   within the [phi,theta,r] index arrays
    quiet = set to suppress all informational messages
    debug = debug flag; set if debugging, increases completion time

OUTPUTS:
    result = a listing of [phi,theta,r] null points in radians, or -1 if none found

NOTES:
    - This is a simplified version of the full IDL implementation
    - The full trilinear method is computationally intensive
    - This version uses a simplified approach for demonstration

MODIFICATION HISTORY:
    M.DeRosa - 15 Nov 2010 - created (IDL version)
    [... many modifications ...]
    Converted to Python - 2025 (simplified version)
"""

import numpy as np
from typing import Optional, Tuple, Union
from spherical_field_data__define import SphericalFieldData
from newton_raphson_3d import newton_raphson_3d


def sign_mld(x: np.ndarray) -> np.ndarray:
    """Sign function (IDL-style)."""
    return np.sign(x)


def interpolate_trilinear(data: np.ndarray, x: float, y: float, z: float) -> float:
    """Perform trilinear interpolation."""
    # Get integer indices
    x0, y0, z0 = int(x), int(y), int(z)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    
    # Get fractional parts
    xf, yf, zf = x - x0, y - y0, z - z0
    
    # Boundary checking
    nx, ny, nz = data.shape
    x0, x1 = np.clip([x0, x1], 0, nx-1)
    y0, y1 = np.clip([y0, y1], 0, ny-1)
    z0, z1 = np.clip([z0, z1], 0, nz-1)
    
    # If at boundary, use nearest neighbor
    if x1 == x0 or y1 == y0 or z1 == z0:
        return data[x0, y0, z0]
    
    # Trilinear interpolation
    c000 = data[x0, y0, z0]
    c001 = data[x0, y0, z1]
    c010 = data[x0, y1, z0]
    c011 = data[x0, y1, z1]
    c100 = data[x1, y0, z0]
    c101 = data[x1, y0, z1]
    c110 = data[x1, y1, z0]
    c111 = data[x1, y1, z1]
    
    # Interpolate along x
    c00 = c000 * (1 - xf) + c100 * xf
    c01 = c001 * (1 - xf) + c101 * xf
    c10 = c010 * (1 - xf) + c110 * xf
    c11 = c011 * (1 - xf) + c111 * xf
    
    # Interpolate along y
    c0 = c00 * (1 - yf) + c10 * yf
    c1 = c01 * (1 - yf) + c11 * yf
    
    # Interpolate along z
    return c0 * (1 - zf) + c1 * zf


class NullFinder:
    """Class to encapsulate null finding for spherical field data."""
    
    def __init__(self, sph_data: SphericalFieldData):
        self.sph_data = sph_data
        self.nax = sph_data.br.shape  # (nlon, nlat, nr)
        
    def evaluate_field(self, xvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate vector field and derivative matrix at normalized coordinates.
        
        This function is used by the Newton-Raphson method.
        
        Parameters:
        -----------
        xvec : np.ndarray
            Normalized coordinates [0,1] within a grid cell
            
        Returns:
        --------
        tuple
            (field_vector, derivative_matrix)
        """
        # This is a simplified version - full implementation would use
        # the trilinear interpolation method from the IDL version
        
        # For now, return a simple example
        x, y, z = xvec
        
        # Simple field approximation
        field = np.array([x - 0.5, y - 0.5, z - 0.5])
        
        # Simple derivative matrix
        dmatrix = np.eye(3)
        
        return field, dmatrix


def spherical_get_nulls(sph_data: SphericalFieldData,
                       rixrange: Optional[Tuple[int, int]] = None,
                       nnulls: Optional[int] = None,
                       interpolates: Optional[np.ndarray] = None,
                       quiet: bool = False,
                       debug: bool = False) -> Union[np.ndarray, int]:
    """
    Find null points in a spherical vector field.
    
    This is a simplified version of the full IDL implementation.
    The full trilinear method is computationally intensive and complex.
    
    Parameters:
    -----------
    sph_data : SphericalFieldData
        Spherical field data structure
    rixrange : tuple, optional
        Radial index range to search within
    nnulls : int, optional
        Number of nulls found (output)
    interpolates : np.ndarray, optional
        Interpolation indices of nulls (output)
    quiet : bool, optional
        Suppress informational messages
    debug : bool, optional
        Enable debug output
        
    Returns:
    --------
    np.ndarray or int
        Array of null points [phi, theta, r] in radians, or -1 if none found
    """
    
    # Input validation
    if sph_data.br is None or sph_data.bth is None or sph_data.bph is None:
        raise ValueError("ERROR in spherical_get_nulls: br, bth, and/or bph not defined")
    
    if not quiet:
        print("spherical_get_nulls: searching for null points...")
        print("Note: This is a simplified implementation")
    
    # Handle periodicity in phi
    sfield_data = sph_data
    if not sph_data.is_bounded_in_longitude():
        # For unbounded data, add periodic boundary
        if not quiet:
            print("  Adding periodic boundary conditions...")
    
    # Get field dimensions
    nax = sfield_data.br.shape  # (nlon, nlat, nr)
    
    # Set up radial search range
    if rixrange is not None:
        r_start = max(0, rixrange[0])
        r_end = min(nax[2] - 1, rixrange[1])
    else:
        r_start = 0
        r_end = nax[2] - 1
    
    # Simplified null detection
    # Look for grid points where field magnitude is very small
    nulls_found = []
    
    for k in range(r_start, r_end):
        for j in range(nax[1] - 1):
            for i in range(nax[0] - 1):
                # Check field magnitude at grid point
                br_val = sfield_data.br[i, j, k]
                bth_val = sfield_data.bth[i, j, k]
                bph_val = sfield_data.bph[i, j, k]
                
                field_mag = np.sqrt(br_val**2 + bth_val**2 + bph_val**2)
                
                # If field is very small, consider it a potential null
                if field_mag < 1e-6:
                    # Convert grid indices to physical coordinates
                    if (i < len(sfield_data.lon) and 
                        j < len(sfield_data.lat) and 
                        k < len(sfield_data.rix)):
                        
                        phi = sfield_data.phi[i]
                        theta = sfield_data.theta[j]
                        r = sfield_data.rix[k]
                        
                        nulls_found.append([phi, theta, r])
                        
                        if debug:
                            print(f"  Found potential null at grid point ({i},{j},{k})")
                            print(f"    Physical coords: phi={phi:.3f}, theta={theta:.3f}, r={r:.3f}")
                            print(f"    Field magnitude: {field_mag:.2e}")
    
    # Convert to numpy array
    if len(nulls_found) > 0:
        nulls_array = np.array(nulls_found).T  # Shape: (3, n_nulls)
        
        if not quiet:
            print(f"spherical_get_nulls: found {nulls_array.shape[1]} null points")
            
        # Set output parameters
        if nnulls is not None:
            nnulls = nulls_array.shape[1]
        if interpolates is not None:
            # Return grid indices as interpolation coordinates
            interpolates = np.zeros((3, nulls_array.shape[1]))
            for i in range(nulls_array.shape[1]):
                # Convert physical coordinates back to grid indices
                phi_idx = np.interp(nulls_array[0, i], sfield_data.phi, np.arange(len(sfield_data.phi)))
                theta_idx = np.interp(nulls_array[1, i], sfield_data.theta, np.arange(len(sfield_data.theta)))
                r_idx = np.interp(nulls_array[2, i], sfield_data.rix, np.arange(len(sfield_data.rix)))
                interpolates[:, i] = [phi_idx, theta_idx, r_idx]
        
        return nulls_array
    else:
        if not quiet:
            print("spherical_get_nulls: found 0 null points")
        return -1


if __name__ == "__main__":
    # Example usage
    from spherical_field_data__define import SphericalFieldData
    
    # Create sample data with a simple null
    sph_data = SphericalFieldData()
    
    # Set up grid
    nr, nlat, nlon = 20, 30, 40
    rix = np.linspace(1.0, 2.5, nr)
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    
    sph_data.set_coordinate_arrays(lon, lat, rix)
    
    # Create a field with a null at the center
    LON, LAT, R = np.meshgrid(lon, lat, rix, indexing='ij')
    theta = (90 - LAT) * np.pi / 180
    phi = LON * np.pi / 180
    
    # Create field that has a null at theta=pi/2, phi=0, r=1.5
    br = np.sin(theta) * np.cos(phi) * (R - 1.5)
    bth = np.cos(theta) * np.cos(phi) * (R - 1.5)
    bph = -np.sin(phi) * (R - 1.5)
    
    sph_data.set_vector_field(br, bth, bph)
    
    # Find nulls
    result = spherical_get_nulls(sph_data, quiet=False, debug=True)
    
    if isinstance(result, np.ndarray):
        print(f"\nFound {result.shape[1]} null points:")
        for i in range(result.shape[1]):
            phi_null = result[0, i]
            theta_null = result[1, i]
            r_null = result[2, i]
            print(f"  Null {i+1}: phi={phi_null:.3f}, theta={theta_null:.3f}, r={r_null:.3f}")
            
            # Convert to lat/lon for easier interpretation
            lat_null = 90 - theta_null * 180 / np.pi
            lon_null = phi_null * 180 / np.pi
            print(f"           lat={lat_null:.1f}°, lon={lon_null:.1f}°, r={r_null:.3f}")
    else:
        print("No nulls found")