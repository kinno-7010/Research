"""
Spherical to PFSS Conversion

This procedure populates PFSS common block variables with the data
stored in the given SphericalFieldData structure.

CALLING SEQUENCE:
    spherical_to_pfss(sph_data, noreset=False, no_copy=False)

INPUTS:
    sph_data = a SphericalFieldData object
    noreset = if True, routine does not set to undefined variables which are
              not defined in the spherical field data structure or which
              exist in the pfss common block but not in the spherical data
              structure 
    no_copy = if True, will clear the data from the sph_data structure after
              the data is copied to the variables (for memory management)

OUTPUTS: 
    Returns a dictionary containing the PFSS common block variables

NOTES:
    1.  Those variables in the SphericalFieldData structure for which there
        are no corresponding variables in the PFSS common block are not
        carried over.
    2.  If variables are added to the PFSS common block, one needs to add a
        line to this routine.
    3.  This is a Python adaptation of the IDL version that used common blocks.
        Instead of common blocks, we return a dictionary of variables.

MODIFICATION HISTORY:
    M.DeRosa - 15 Dec 2005 - created (IDL version)
               19 Aug 2006 - added noreset keyword
               13 Apr 2007 - added no_copy keyword
                3 Aug 2011 - added check to see if sph_data is a structure
    Converted to Python - 2025
"""

import numpy as np
from typing import Dict, Any, Optional
from spherical_field_data__define import SphericalFieldData


def spherical_to_pfss(sph_data: SphericalFieldData, 
                     noreset: bool = False, 
                     no_copy: bool = False) -> Dict[str, Any]:
    """
    Convert spherical field data to PFSS common block format.
    
    Parameters:
    -----------
    sph_data : SphericalFieldData
        Input spherical field data structure
    noreset : bool, optional
        If True, don't reset variables that aren't defined in sph_data
    no_copy : bool, optional
        If True, clear data from sph_data after copying (for memory management)
        
    Returns:
    --------
    dict
        Dictionary containing PFSS common block variables
        
    Raises:
    -------
    ValueError
        If sph_data is not a SphericalFieldData object
    """
    
    # Input validation
    if not isinstance(sph_data, SphericalFieldData):
        raise ValueError("ERROR in spherical_to_pfss: no input structure provided")
    
    # Initialize output dictionary
    pfss_data = {}
    
    # Copy vector field components
    if sph_data.br is not None:
        pfss_data['br'] = sph_data.br.copy() if not no_copy else sph_data.br
        if no_copy:
            sph_data.br = None
    elif not noreset:
        pfss_data['br'] = None
        
    if sph_data.bth is not None:
        pfss_data['bth'] = sph_data.bth.copy() if not no_copy else sph_data.bth
        if no_copy:
            sph_data.bth = None
    elif not noreset:
        pfss_data['bth'] = None
        
    if sph_data.bph is not None:
        pfss_data['bph'] = sph_data.bph.copy() if not no_copy else sph_data.bph
        if no_copy:
            sph_data.bph = None
    elif not noreset:
        pfss_data['bph'] = None
    
    # Copy grid dimensions
    if sph_data.nr > 0:
        pfss_data['nr'] = sph_data.nr
    elif not noreset:
        pfss_data['nr'] = None
        
    if sph_data.nlat > 0:
        pfss_data['nlat'] = sph_data.nlat
    elif not noreset:
        pfss_data['nlat'] = None
        
    if sph_data.nlon > 0:
        pfss_data['nlon'] = sph_data.nlon
    elif not noreset:
        pfss_data['nlon'] = None
    
    # Copy coordinate arrays
    if sph_data.rix is not None:
        pfss_data['rix'] = sph_data.rix.copy() if not no_copy else sph_data.rix
        if no_copy:
            sph_data.rix = None
    elif not noreset:
        pfss_data['rix'] = None
        
    if sph_data.lat is not None:
        pfss_data['lat'] = sph_data.lat.copy() if not no_copy else sph_data.lat
        if no_copy:
            sph_data.lat = None
    elif not noreset:
        pfss_data['lat'] = None
        
    if sph_data.lon is not None:
        pfss_data['lon'] = sph_data.lon.copy() if not no_copy else sph_data.lon
        if no_copy:
            sph_data.lon = None
    elif not noreset:
        pfss_data['lon'] = None
        
    if sph_data.theta is not None:
        pfss_data['theta'] = sph_data.theta.copy() if not no_copy else sph_data.theta
        if no_copy:
            sph_data.theta = None
    elif not noreset:
        pfss_data['theta'] = None
        
    if sph_data.phi is not None:
        pfss_data['phi'] = sph_data.phi.copy() if not no_copy else sph_data.phi
        if no_copy:
            sph_data.phi = None
    elif not noreset:
        pfss_data['phi'] = None
    
    # Copy fieldline starting points
    if sph_data.str is not None:
        pfss_data['str'] = sph_data.str.copy() if not no_copy else sph_data.str
        if no_copy:
            sph_data.str = None
    elif not noreset:
        pfss_data['str'] = None
        
    if sph_data.stth is not None:
        pfss_data['stth'] = sph_data.stth.copy() if not no_copy else sph_data.stth
        if no_copy:
            sph_data.stth = None
    elif not noreset:
        pfss_data['stth'] = None
        
    if sph_data.stph is not None:
        pfss_data['stph'] = sph_data.stph.copy() if not no_copy else sph_data.stph
        if no_copy:
            sph_data.stph = None
    elif not noreset:
        pfss_data['stph'] = None
    
    # Copy fieldline trajectories
    if sph_data.ptr is not None:
        pfss_data['ptr'] = sph_data.ptr.copy() if not no_copy else sph_data.ptr
        if no_copy:
            sph_data.ptr = None
    elif not noreset:
        pfss_data['ptr'] = None
        
    if sph_data.ptth is not None:
        pfss_data['ptth'] = sph_data.ptth.copy() if not no_copy else sph_data.ptth
        if no_copy:
            sph_data.ptth = None
    elif not noreset:
        pfss_data['ptth'] = None
        
    if sph_data.ptph is not None:
        pfss_data['ptph'] = sph_data.ptph.copy() if not no_copy else sph_data.ptph
        if no_copy:
            sph_data.ptph = None
    elif not noreset:
        pfss_data['ptph'] = None
        
    if sph_data.nstep is not None:
        pfss_data['nstep'] = sph_data.nstep.copy() if not no_copy else sph_data.nstep
        if no_copy:
            sph_data.nstep = None
    elif not noreset:
        pfss_data['nstep'] = None
    
    # Variables in the PFSS common block that have no corresponding field in sph_data
    # These are set to None if not resetting
    if not noreset:
        pfss_data['l0'] = None      # Central meridian longitude
        pfss_data['b0'] = None      # Solar B0 angle
        pfss_data['now'] = None     # Current time
        pfss_data['phiat'] = None   # Phi at something
        pfss_data['phibt'] = None   # Phi bt something
        pfss_data['rimage'] = None  # Image radius
    
    return pfss_data


def pfss_to_spherical(pfss_data: Dict[str, Any]) -> SphericalFieldData:
    """
    Convert PFSS common block data back to SphericalFieldData structure.
    
    This is the inverse operation of spherical_to_pfss.
    
    Parameters:
    -----------
    pfss_data : dict
        Dictionary containing PFSS common block variables
        
    Returns:
    --------
    SphericalFieldData
        Spherical field data structure
    """
    
    sph_data = SphericalFieldData()
    
    # Set grid dimensions
    if 'nr' in pfss_data and pfss_data['nr'] is not None:
        sph_data.nr = pfss_data['nr']
    if 'nlat' in pfss_data and pfss_data['nlat'] is not None:
        sph_data.nlat = pfss_data['nlat']
    if 'nlon' in pfss_data and pfss_data['nlon'] is not None:
        sph_data.nlon = pfss_data['nlon']
    
    # Set coordinate arrays
    if all(key in pfss_data and pfss_data[key] is not None 
           for key in ['lon', 'lat', 'rix']):
        sph_data.set_coordinate_arrays(pfss_data['lon'], pfss_data['lat'], pfss_data['rix'])
    
    # Set vector field
    if all(key in pfss_data and pfss_data[key] is not None 
           for key in ['br', 'bth', 'bph']):
        sph_data.set_vector_field(pfss_data['br'], pfss_data['bth'], pfss_data['bph'])
    
    # Set fieldline starting points
    if all(key in pfss_data and pfss_data[key] is not None 
           for key in ['str', 'stth', 'stph']):
        sph_data.set_starting_points(pfss_data['str'], pfss_data['stth'], pfss_data['stph'])
    
    # Set fieldline trajectories
    if all(key in pfss_data and pfss_data[key] is not None 
           for key in ['ptr', 'ptth', 'ptph', 'nstep']):
        sph_data.set_fieldline_trajectories(pfss_data['ptr'], pfss_data['ptth'], 
                                          pfss_data['ptph'], pfss_data['nstep'])
    
    return sph_data


if __name__ == "__main__":
    # Example usage
    from spherical_field_data__define import SphericalFieldData
    
    # Create sample data
    sph_data = SphericalFieldData()
    
    # Set up a simple test case
    nr, nlat, nlon = 10, 20, 30
    rix = np.linspace(1.0, 2.5, nr)
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    
    sph_data.set_coordinate_arrays(lon, lat, rix)
    
    # Create sample vector field
    br = np.random.random((nlon, nlat, nr))
    bth = np.random.random((nlon, nlat, nr))
    bph = np.random.random((nlon, nlat, nr))
    
    sph_data.set_vector_field(br, bth, bph)
    
    print("Original spherical data:")
    print(sph_data)
    
    # Convert to PFSS format
    pfss_data = spherical_to_pfss(sph_data)
    
    print(f"\nPFSS data keys: {list(pfss_data.keys())}")
    print(f"PFSS data shapes: br={pfss_data['br'].shape if pfss_data['br'] is not None else None}")
    
    # Convert back to spherical format
    sph_data2 = pfss_to_spherical(pfss_data)
    
    print(f"\nConverted back to spherical:")
    print(sph_data2)
    
    # Test with no_copy option
    print(f"\nTesting with no_copy=True...")
    sph_data3 = SphericalFieldData()
    sph_data3.set_coordinate_arrays(lon, lat, rix)
    sph_data3.set_vector_field(br.copy(), bth.copy(), bph.copy())
    
    pfss_data2 = spherical_to_pfss(sph_data3, no_copy=True)
    print(f"After no_copy=True, sph_data3.br is None: {sph_data3.br is None}")
    print(f"pfss_data2['br'] shape: {pfss_data2['br'].shape}")