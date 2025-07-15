"""
pfss_rad_field_crossing.py - This procedure determines the latitude and
longitude at which a set of field lines crosses a specified radial level

usage: data, interpindex = pfss_rad_field_crossing(rcut)

Parameters:
    rcut: radial level to evaluate line crossings

Returns:
    data: [3,n] array of data where each of the n crossings has
          [lineno,bcross,lcross] with (bcross,lcross)=(lat,lon)
          coordinate in degrees of planar crossing and lineno=line
          number of input field line arrays
    interpindex: n-element array of interpolation coordinates giving the
                fractional gridpoint along each fieldline at which the
                nth crossing occurs

M.DeRosa - 21 Apr 2004 - created, loosely based on slice_field.pro
          14 Aug 2007 - added interpindex keyword
           7 Jan 2010 - routine is now merely a wrapper for the identical
                        spherical_get_radial_crossing function
"""

import numpy as np
from pfss_data_block import PfssDataBlock
from pfss_to_spherical import pfss_to_spherical
from spherical_to_pfss import spherical_to_pfss
from spherical_get_radial_crossing import spherical_get_radial_crossing


def pfss_rad_field_crossing(rcut):
    """
    Determine the latitude and longitude at which field lines cross a radial level.
    
    Args:
        rcut (float): Radial level to evaluate line crossings
        
    Returns:
        tuple: (data, interpindex) where:
            - data: [3,n] array with [lineno, lat, lon] for each crossing
            - interpindex: n-element array of interpolation coordinates
    """
    
    # Get the data block
    data_block = PfssDataBlock()
    
    # Convert to spherical field data structure
    pfss_data = pfss_to_spherical(free=True, no_copy=True)
    
    # Call the spherical routine
    data, interpindex = spherical_get_radial_crossing(pfss_data, rcut)
    
    # Deal with differing units of output
    if data is not None and len(data) > 0:
        # Convert from colatitude (radians) to latitude (degrees)
        data[1, :] = 90 - data[1, :] * 180 / np.pi
        # Convert from longitude (radians) to longitude (degrees)
        data[2, :] = data[2, :] * 180 / np.pi
    
    # Convert back to pfss common block
    spherical_to_pfss(pfss_data, noreset=True, no_copy=True)
    
    return data, interpindex


def pfss_rad_field_crossing_legacy(rcut):
    """
    Legacy implementation of pfss_rad_field_crossing for reference.
    
    This is the original algorithm that was commented out in the IDL version.
    """
    
    # Get the data block
    data_block = PfssDataBlock()
    
    # Preliminaries
    nlines = len(data_block.nstep)
    rmin = np.min(data_block.rix)
    rmax = np.max(data_block.rix)
    
    # Check rcut
    if rcut is None:
        print(f'  pfss_rad_field_crossing: rcut being set to {data_block.rix[0]}')
        rcut = data_block.rix[0]
    else:
        rcut = max(rmin, min(rcut, rmax))
    
    # Initialize data arrays
    data_list = []
    interpindex_list = []
    
    # Loop through the individual field lines
    for i in range(nlines):
        # Extract coordinates of current field line
        ns = data_block.nstep[i]
        lr = data_block.ptr[:ns, i]
        lth = data_block.ptth[:ns, i]
        lph = data_block.ptph[:ns, i]
        
        # Look for crossings only if rcut is in range
        lmin = np.min(lr)
        lmax = np.max(lr)
        
        # First find out if any points exactly match
        ixm = np.where(lr == rcut)[0]
        if len(ixm) > 0:
            for j in range(len(ixm)):
                data_list.append([i, lth[ixm[j]], lph[ixm[j]]])
                interpindex_list.append(ixm[j])
        
        # Now find crossings
        if ns > 1:
            crossings = (lr[:-1] - rcut) * (lr[1:] - rcut) < 0
            ixpt = np.where(crossings)[0]
            
            if len(ixpt) > 0:
                # Linearly interpolate to get locus of crossing
                for j in range(len(ixpt)):
                    # Avoid double-counting exact matches
                    if ixpt[j] not in ixm:
                        coeff = (rcut - lr[ixpt[j]]) / (lr[ixpt[j]+1] - lr[ixpt[j]])
                        ptcrth = 90 - ((1-coeff)*lth[ixpt[j]] + coeff*lth[ixpt[j]+1]) * 180/np.pi
                        ptcrph = (((1-coeff)*lph[ixpt[j]] + coeff*lph[ixpt[j]+1]) * 180/np.pi + 360) % 360
                        data_list.append([i, ptcrth, ptcrph])
                        interpindex_list.append(ixpt[j] + coeff)
    
    # Convert to arrays
    if len(data_list) == 0:
        print('  pfss_rad_field_crossing: no crossings detected')
        return None, None
    else:
        data = np.array(data_list).T  # Transpose to get [3,n] format
        interpindex = np.array(interpindex_list)
        return data, interpindex


# For compatibility with IDL calling convention
def pfss_rad_field_crossing_idl(rcut):
    """
    IDL-compatible wrapper for pfss_rad_field_crossing.
    
    This function modifies the global data block and returns the crossing data.
    """
    return pfss_rad_field_crossing(rcut)