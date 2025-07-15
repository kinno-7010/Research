"""
Spherical Get Radial Crossing

Given vector fieldline data on a spherical grid, this function
determines the latitudes and longitudes at which each fieldline
intersects a surface of constant radius.

CALLING SEQUENCE:
    result = spherical_get_radial_crossing(sph_data, rcut, interpindex=None)

INPUTS:
    sph_data = a SphericalFieldData object with the following fields defined:
        br, bth, bph, nlat, nlon, nr, rix, lat, lon, str, stth, stph.
        Basically, one needs the vector field (br,bth,bph), its dimension
        (nr,nlat,nlon), its indexing (rix,lat,lon), and the starting
        points (str,stth,stph).
    rcut = radius of spherical surface at which to determine crossings

OUTPUTS:
    result = [3,n] array of data where each of the n crossings has a
        row in the array: [lineno,bcross,lcross] where lineno = the
        line number of the input fieldline array in the sph_data
        structure, and (bcross,lcross) are the colatitude and longitude
        in radians of the crossing.  If no crossings are detected, or
        if there is an error, the result will be -1.
    interpindex = an n-element array of interpolation
        coordinates giving the fractional gridpoint along each
        fieldline at which the nth crossing occurs

MODIFICATION HISTORY:
    M.DeRosa - 23 Mar 2007 - created, based on pfss_rad_field_crossing.pro (IDL)
               7 Jan 2010 - added interpindex keyword
    Converted to Python - 2025
"""

import numpy as np
from typing import Optional, Tuple, Union
from spherical_field_data__define import SphericalFieldData


def spherical_get_radial_crossing(sph_data: SphericalFieldData, 
                                 rcut: float) -> Tuple[Union[np.ndarray, int], np.ndarray]:
    """
    Determine fieldline crossings with a spherical surface of constant radius.
    
    Parameters:
    -----------
    sph_data : SphericalFieldData
        Spherical field data structure with fieldline trajectories
    rcut : float
        Radius of spherical surface at which to determine crossings
        
    Returns:
    --------
    result : np.ndarray or int
        [3,n] array of crossings: [lineno, bcross, lcross] where
        lineno is the line number, bcross is colatitude in radians,
        lcross is longitude in radians. Returns -1 if no crossings.
    interpindex : np.ndarray
        n-element array of interpolation coordinates along each fieldline
        
    Raises:
    -------
    ValueError
        If required input data is missing
    """
    
    # Input validation
    if sph_data is None:
        raise ValueError("ERROR in spherical_get_radial_crossing: no input data provided")
    
    if sph_data.rix is None:
        raise ValueError("ERROR in spherical_get_radial_crossing: rix not defined")
        
    if sph_data.nstep is None or sph_data.ptr is None:
        raise ValueError("ERROR in spherical_get_radial_crossing: fieldline data not defined")
    
    # Check rcut bounds
    rmin, rmax = np.min(sph_data.rix), np.max(sph_data.rix)
    if rcut < rmin or rcut > rmax:
        print(f"WARNING in spherical_get_radial_crossing: rcut out of bounds,")
        print(f"    using closest radial gridpoint")
        rcut = np.clip(rcut, rmin, rmax)
    
    # Initialize output arrays
    result_list = []
    interpindex_list = []
    
    # Number of fieldlines
    nlines = len(sph_data.nstep)
    
    # Loop through individual field lines
    for i in range(nlines):
        # Extract coordinates of current field line
        ns = sph_data.nstep[i]
        if ns <= 0:
            continue
            
        lr = sph_data.ptr[:ns, i]
        lth = sph_data.ptth[:ns, i]
        lph = sph_data.ptph[:ns, i]
        
        # Look for crossings only if rcut is in range
        lmin, lmax = np.min(lr), np.max(lr)
        if rcut <= lmax and rcut >= lmin:
            
            # First find out if any points exactly match
            exact_matches = np.where(lr == rcut)[0]
            for j in exact_matches:
                result_list.append([i, lth[j], lph[j]])
                interpindex_list.append(j)
            
            # Now find crossings between consecutive points
            # Look for sign changes in (lr - rcut)
            diff_vals = lr - rcut
            crossing_indices = np.where(diff_vals[:-1] * diff_vals[1:] < 0)[0]
            
            for j in crossing_indices:
                # Avoid double-counting exact matches
                if j not in exact_matches:
                    # Linear interpolation to get crossing location
                    coeff = (rcut - lr[j]) / (lr[j+1] - lr[j])
                    ptcrth = (1 - coeff) * lth[j] + coeff * lth[j+1]
                    ptcrph = (1 - coeff) * lph[j] + coeff * lph[j+1]
                    
                    # Handle phi wraparound
                    ptcrph = (ptcrph + 2*np.pi) % (2*np.pi)
                    
                    result_list.append([i, ptcrth, ptcrph])
                    interpindex_list.append(j + coeff)
    
    # Convert lists to arrays
    if len(result_list) == 0:
        print("WARNING in spherical_get_radial_crossing: no crossings detected")
        return -1, np.array([-1.0])
    else:
        result = np.array(result_list).T  # Transpose to get [3,n] array
        interpindex = np.array(interpindex_list)
        return result, interpindex


if __name__ == "__main__":
    # Example usage
    from spherical_field_data__define import SphericalFieldData
    
    # Create sample data
    sph_data = SphericalFieldData()
    
    # Set up grid
    nr, nlat, nlon = 20, 30, 40
    rix = np.linspace(1.0, 2.5, nr)
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    
    sph_data.set_coordinate_arrays(lon, lat, rix)
    
    # Create sample fieldline data
    nlines = 10
    stepmax = 100
    
    # Simple radial fieldlines for testing
    ptr = np.zeros((stepmax, nlines))
    ptth = np.zeros((stepmax, nlines))
    ptph = np.zeros((stepmax, nlines))
    nstep = np.zeros(nlines, dtype=int)
    
    for i in range(nlines):
        # Create a simple fieldline from inner to outer radius
        nstep[i] = 50
        ptr[:nstep[i], i] = np.linspace(1.0, 2.5, nstep[i])
        ptth[:nstep[i], i] = np.pi/2 + 0.1 * np.sin(np.linspace(0, 2*np.pi, nstep[i]))
        ptph[:nstep[i], i] = i * 2*np.pi/nlines
    
    sph_data.set_fieldline_trajectories(ptr, ptth, ptph, nstep)
    
    # Find crossings at radius 1.5
    result, interpindex = spherical_get_radial_crossing(sph_data, 1.5)
    
    if isinstance(result, np.ndarray):
        print(f"Found {result.shape[1]} crossings:")
        for i in range(result.shape[1]):
            line_no = int(result[0, i])
            theta = result[1, i]
            phi = result[2, i]
            interp_idx = interpindex[i]
            print(f"  Line {line_no}: theta={theta:.3f}, phi={phi:.3f}, interp_idx={interp_idx:.3f}")
    else:
        print("No crossings found")