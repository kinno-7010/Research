"""
Spherical Field Start Coordinate Selection

This procedure chooses points through which to trace fieldlines
of the vector field.

CALLING SEQUENCE:
    spherical_field_start_coord(sph_data, fieldtype, spacing, bbox=None,
                               radstart=None, add=False)

INPUTS:
    sph_data = a SphericalFieldData object with the following fields defined:
        br, bth, bph, nlat, nlon, nr, rix, lat, lon. If starting points are
        already defined (str, stth, stph), the new fieldlines will either
        overwrite or be added to the existing set, depending on the add flag.
    fieldtype = 1 = starting points fall along the equator
                2 = uniform grid, with a random offset
                3 = points are distributed randomly in latitude and longitude
                4 = read in from a file (not implemented)
                5 = uniform grid (default)
                6 = points are weighted by radial flux at the start radius
                7 = rectangular grid
    spacing = controls density of points, does different things depending on 
              fieldtype
    bbox = either [lon1, lon2] (for fieldtype 1) or [lon1, lat1, lon2, lat2] 
           (for fieldtypes 2,3,5,6,7) defining bounding box (in degrees) outside
           of which no fieldline starting points lie
    radstart = a scalar equal to the radius at which all fieldlines should
               start (default=minimum radius in the domain)
    add = if True, the new starting points are added to the existing set
          already defined in sph_data (default is to overwrite)

OUTPUTS:
    sph_data = SphericalFieldData object with str, stth, stph fields set.
               These are the r-, theta-, and phi-components of the fieldline
               starting points.

NOTES:
    1. fieldtype=4 option is not implemented.

MODIFICATION HISTORY:
    M.DeRosa - 13 Dec 2005 - created from guts of pfss_field_start_coord.pro (IDL)
               19 Dec 2005 - now handles "wraparound" bounding boxes
               24 Jan 2006 - now deals with bounded datasets
               14 Dec 2009 - added fieldtype 7
               23 Dec 2009 - added latitudinal boundary check
    Converted to Python - 2025
"""

import numpy as np
from typing import Optional, List, Union
from spherical_field_data__define import SphericalFieldData


def linrange(n: int, start: float, stop: float) -> np.ndarray:
    """Create linearly spaced array (IDL linrange equivalent)."""
    return np.linspace(start, stop, n)


def get_interpolation_index(arr: np.ndarray, value: float) -> float:
    """Get interpolation index for a value in a sorted array."""
    return np.interp(value, arr, np.arange(len(arr)))


def spherical_field_start_coord(sph_data: SphericalFieldData, 
                               fieldtype: int, 
                               spacing: Union[float, List[float]], 
                               bbox: Optional[List[float]] = None,
                               radstart: Optional[float] = None, 
                               add: bool = False) -> None:
    """
    Choose starting points for fieldline tracing.
    
    Parameters:
    -----------
    sph_data : SphericalFieldData
        Spherical field data structure
    fieldtype : int
        Type of starting point distribution (1-7)
    spacing : float or list
        Controls density of points
    bbox : list, optional
        Bounding box for starting points
    radstart : float, optional
        Starting radius (default = minimum radius)
    add : bool, optional
        If True, add to existing points instead of replacing
    """
    
    # Input validation
    if sph_data is None:
        raise ValueError("ERROR in spherical_field_start_coord: no input data provided")
    
    if sph_data.rix is None:
        raise ValueError("ERROR in spherical_field_start_coord: coordinate arrays not defined")
    
    # Get radial range
    rmin, rmax = np.min(sph_data.rix), np.max(sph_data.rix)
    
    # Set radstart if not provided
    if radstart is None:
        radstart = rmin
    
    # Convert spacing to list if needed
    if not isinstance(spacing, (list, tuple, np.ndarray)):
        spacing = [spacing]
    
    # Generate starting points based on fieldtype
    if fieldtype == 1:  # Points along the equator
        npt = int(round(spacing[0]))
        str_coords = np.full(npt, radstart)
        stth_coords = np.full(npt, np.pi/2)
        
        if bbox is not None and len(bbox) == 2:
            phmin, phmax = bbox[0], bbox[1]
            if phmax < phmin:
                phmax += 360
            stph_coords = linrange(npt, phmin, phmax)
            stph_coords = ((stph_coords + 360) % 360) * np.pi / 180
        else:
            stph_coords = linrange(npt + 1, 0, 360)[:npt] * np.pi / 180
            
    elif fieldtype == 2:  # Uniform grid with random offset
        str_coords, stth_coords, stph_coords = _generate_uniform_grid_random(
            sph_data, spacing[0], radstart, bbox)
            
    elif fieldtype == 3:  # Random distribution
        str_coords, stth_coords, stph_coords = _generate_random_points(
            sph_data, spacing[0], radstart, bbox)
            
    elif fieldtype == 4:  # Read from file (not implemented)
        raise NotImplementedError("fieldtype=4 not implemented")
        
    elif fieldtype == 5:  # Uniform grid (default)
        str_coords, stth_coords, stph_coords = _generate_uniform_grid(
            sph_data, spacing[0], radstart, bbox)
            
    elif fieldtype == 6:  # Flux-weighted points
        str_coords, stth_coords, stph_coords = _generate_flux_weighted_points(
            sph_data, spacing[0], radstart, bbox)
            
    elif fieldtype == 7:  # Rectangular grid
        str_coords, stth_coords, stph_coords = _generate_rectangular_grid(
            sph_data, spacing[0], radstart, bbox)
            
    else:
        raise ValueError("ERROR in spherical_field_start_coord: invalid fieldtype")
    
    # Apply longitude bounds if dataset is bounded
    if sph_data.is_bounded_in_longitude():
        str_coords, stth_coords, stph_coords = _apply_longitude_bounds(
            str_coords, stth_coords, stph_coords, sph_data.lonbounds)
    
    # Apply latitudinal bounds
    str_coords, stth_coords, stph_coords = _apply_latitudinal_bounds(
        str_coords, stth_coords, stph_coords, sph_data.theta)
    
    # Handle adding to existing points or replacing
    if add and sph_data.str is not None:
        # Add to existing points
        str_coords = np.concatenate([sph_data.str, str_coords])
        stth_coords = np.concatenate([sph_data.stth, stth_coords])
        stph_coords = np.concatenate([sph_data.stph, stph_coords])
    
    # Set the starting points in the data structure
    sph_data.set_starting_points(str_coords, stth_coords, stph_coords)


def _generate_uniform_grid_random(sph_data: SphericalFieldData, 
                                 spacing: float, 
                                 radstart: float, 
                                 bbox: Optional[List[float]]) -> tuple:
    """Generate uniform grid with random offset."""
    # Set up binning
    nlatbin = int(round(sph_data.nlat / spacing))
    dlatbin = np.pi / nlatbin
    latbin = linrange(nlatbin, dlatbin/2, np.pi - dlatbin/2)
    nlonbin = np.round(nlatbin * 2 * np.sin(latbin)).astype(int)
    dlonbin = 2 * np.pi / nlonbin
    
    # Calculate cumulative point counts
    nloncum = np.cumsum(nlonbin)
    npt = int(nloncum[-1])
    
    # Add random offsets
    np.random.seed(42)  # For reproducibility
    stth = np.random.random(npt) - 0.5
    stph = np.random.random(npt) - 0.5
    
    for i in range(npt):
        lonbinix = np.where(i < nloncum)[0][0]
        stth[i] = latbin[lonbinix] + stth[i] * dlatbin / 2
        stph[i] = ((i - (nloncum[lonbinix] - nlonbin[lonbinix])) * dlonbin[lonbinix] + 
                   stph[i] * dlonbin[lonbinix] / 2)
    
    stph = (stph + 2*np.pi) % (2*np.pi)
    
    # Apply bounding box if provided
    if bbox is not None and len(bbox) == 4:
        stth, stph = _apply_bbox_filter(stth, stph, bbox)
    
    str_coords = np.full(len(stth), radstart)
    return str_coords, stth, stph


def _generate_random_points(sph_data: SphericalFieldData, 
                           spacing: float, 
                           radstart: float, 
                           bbox: Optional[List[float]]) -> tuple:
    """Generate randomly distributed points."""
    npt = int(round(spacing))
    str_coords = np.full(npt, radstart)
    
    if bbox is not None and len(bbox) == 4:
        thmin = (90 - bbox[3]) * np.pi / 180
        thmax = (90 - bbox[1]) * np.pi / 180
        stth = np.random.random(npt) * (thmax - thmin) + thmin
        
        phmin, phmax = bbox[0], bbox[2]
        if phmax < phmin:
            phmax += 360
        stph = ((np.random.random(npt) * (phmax - phmin) + phmin) % 360) * np.pi / 180
    else:
        thmin, thmax = np.min(sph_data.theta), np.max(sph_data.theta)
        stth = np.random.random(npt) * (thmax - thmin) + thmin
        stph = np.random.random(npt) * 2 * np.pi
    
    return str_coords, stth, stph


def _generate_uniform_grid(sph_data: SphericalFieldData, 
                          spacing: float, 
                          radstart: float, 
                          bbox: Optional[List[float]]) -> tuple:
    """Generate uniform grid."""
    # Set up binning
    nlatbin = int(round(sph_data.nlat / spacing))
    dlatbin = np.pi / nlatbin
    latbin = linrange(nlatbin, dlatbin/2, np.pi - dlatbin/2)
    nlonbin = np.round(nlatbin * 2 * np.sin(latbin)).astype(int)
    dlonbin = 2 * np.pi / nlonbin
    
    # Calculate cumulative point counts
    nloncum = np.cumsum(nlonbin)
    npt = int(nloncum[-1])
    
    # Set stth, stph
    stth = np.zeros(npt)
    stph = np.zeros(npt)
    
    for i in range(npt):
        lonbinix = np.where(i < nloncum)[0][0]
        stth[i] = latbin[lonbinix]
        stph[i] = (i - (nloncum[lonbinix] - nlonbin[lonbinix])) * dlonbin[lonbinix]
    
    # Apply bounding box if provided
    if bbox is not None and len(bbox) == 4:
        stth, stph = _apply_bbox_filter(stth, stph, bbox)
    
    str_coords = np.full(len(stth), radstart)
    return str_coords, stth, stph


def _generate_flux_weighted_points(sph_data: SphericalFieldData, 
                                  spacing: float, 
                                  radstart: float, 
                                  bbox: Optional[List[float]]) -> tuple:
    """Generate flux-weighted points."""
    npt = int(round(spacing))
    oversampling = 10
    
    # Get random starting points with oversampling
    if bbox is not None and len(bbox) == 4:
        thmin = (90 - bbox[3]) * np.pi / 180
        thmax = (90 - bbox[1]) * np.pi / 180
        stth = np.random.random(npt * oversampling) * (thmax - thmin) + thmin
        
        phmin = bbox[0] * np.pi / 180
        phmax = bbox[2] * np.pi / 180
        stph = np.random.random(npt * oversampling) * (phmax - phmin) + phmin
    else:
        thmin, thmax = np.min(sph_data.theta), np.max(sph_data.theta)
        stth = np.random.random(npt * oversampling) * (thmax - thmin) + thmin
        stph = np.random.random(npt * oversampling) * 2 * np.pi
    
    str_coords = np.full(npt * oversampling, radstart)
    
    # Get br at starting points
    ir = get_interpolation_index(sph_data.rix, str_coords)
    ith = get_interpolation_index(sph_data.lat, 90 - stth * 180 / np.pi)
    iph = get_interpolation_index(sph_data.lon, stph * 180 / np.pi)
    
    # Interpolate br values
    br0 = np.zeros(len(str_coords))
    for i in range(len(str_coords)):
        ix, iy, iz = int(iph[i]), int(ith[i]), int(ir[i])
        # Simple nearest neighbor interpolation for now
        br0[i] = sph_data.br[ix, iy, iz]
    
    # Sort by magnitude and select top points
    sorted_indices = np.argsort(np.abs(br0))[::-1]
    selected_indices = sorted_indices[:npt]
    
    return str_coords[selected_indices], stth[selected_indices], stph[selected_indices]


def _generate_rectangular_grid(sph_data: SphericalFieldData, 
                              spacing: float, 
                              radstart: float, 
                              bbox: Optional[List[float]]) -> tuple:
    """Generate rectangular grid."""
    # Set up binning
    nlatbin = int(round(sph_data.nlat / spacing))
    dlatbin = np.pi / nlatbin
    latbin = linrange(nlatbin, dlatbin/2, np.pi - dlatbin/2)
    nlonbin = nlatbin * 2
    lonbin = linrange(nlonbin + 1, 0, 2*np.pi)[:nlonbin]
    
    # Create grid
    npt = nlonbin * nlatbin
    stph = np.tile(lonbin, nlatbin)
    stth = np.repeat(latbin, nlonbin)
    
    # Apply bounding box if provided
    if bbox is not None and len(bbox) == 4:
        stth, stph = _apply_bbox_filter(stth, stph, bbox)
    
    str_coords = np.full(len(stth), radstart)
    return str_coords, stth, stph


def _apply_bbox_filter(stth: np.ndarray, stph: np.ndarray, bbox: List[float]) -> tuple:
    """Apply bounding box filter."""
    # Filter in latitude
    lat_mask = ((stth >= (90 - bbox[3]) * np.pi / 180) & 
                (stth <= (90 - bbox[1]) * np.pi / 180))
    stth = stth[lat_mask]
    stph = stph[lat_mask]
    
    # Filter in longitude
    lon1 = ((bbox[0] + 360) % 360) * np.pi / 180
    lon2 = ((bbox[2] + 360) % 360) * np.pi / 180
    
    if lon2 > lon1:
        lon_mask = (stph >= lon1) & (stph <= lon2)
    else:
        lon_mask = (stph <= lon2) | (stph >= lon1)
    
    stth = stth[lon_mask]
    stph = stph[lon_mask]
    
    return stth, stph


def _apply_longitude_bounds(str_coords: np.ndarray, stth: np.ndarray, stph: np.ndarray,
                           lonbounds: np.ndarray) -> tuple:
    """Apply longitude bounds for bounded datasets."""
    if lonbounds[0] >= 0:
        lonb2 = lonbounds * np.pi / 180
        
        if lonb2[1] > lonb2[0]:
            mask = (stph >= lonb2[0]) & (stph <= lonb2[1])
        else:
            mask = (stph >= lonb2[0]) | (stph <= lonb2[1])
        
        if np.sum(mask) == 0:
            raise ValueError("ERROR in spherical_field_start_coord: no points in bounds")
        
        return str_coords[mask], stth[mask], stph[mask]
    
    return str_coords, stth, stph


def _apply_latitudinal_bounds(str_coords: np.ndarray, stth: np.ndarray, stph: np.ndarray,
                             theta: np.ndarray) -> tuple:
    """Apply latitudinal bounds."""
    thmin, thmax = np.min(theta), np.max(theta)
    mask = (stth >= thmin) & (stth <= thmax)
    
    if np.sum(mask) > 0:
        return str_coords[mask], stth[mask], stph[mask]
    
    return str_coords, stth, stph


if __name__ == "__main__":
    # Example usage
    from spherical_field_data__define import SphericalFieldData
    
    # Create sample data
    sph_data = SphericalFieldData()
    
    # Set up grid
    nr, nlat, nlon = 20, 30, 60
    rix = np.linspace(1.0, 2.5, nr)
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    
    sph_data.set_coordinate_arrays(lon, lat, rix)
    
    # Create sample vector field
    br = np.random.random((nlon, nlat, nr))
    bth = np.random.random((nlon, nlat, nr))
    bph = np.random.random((nlon, nlat, nr))
    
    sph_data.set_vector_field(br, bth, bph)
    
    # Test different fieldtypes
    for fieldtype in [1, 2, 3, 5, 6, 7]:
        try:
            print(f"\nTesting fieldtype {fieldtype}:")
            
            # Choose appropriate spacing and bbox
            if fieldtype == 1:
                spacing = 20
                bbox = [0, 180]
            else:
                spacing = 5
                bbox = [0, -60, 180, 60]
            
            spherical_field_start_coord(sph_data, fieldtype, spacing, bbox=bbox)
            
            if sph_data.str is not None:
                print(f"  Generated {len(sph_data.str)} starting points")
                print(f"  Radial range: {np.min(sph_data.str):.2f} - {np.max(sph_data.str):.2f}")
                print(f"  Theta range: {np.min(sph_data.stth):.2f} - {np.max(sph_data.stth):.2f}")
                print(f"  Phi range: {np.min(sph_data.stph):.2f} - {np.max(sph_data.stph):.2f}")
            
        except Exception as e:
            print(f"  Error: {e}")