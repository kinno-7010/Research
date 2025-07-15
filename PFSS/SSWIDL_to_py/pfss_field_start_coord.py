#!/usr/bin/env python3
"""
pfss_field_start_coord - This chooses gridpoints from which to trace the
                         magnetic field lines we will later visualize

Usage: pfss_field_start_coord(fieldtype, spacing, radstart=radstart, top=top,
                              bbox=bbox, add=add)

Parameters:
    fieldtype: 1=starting points fall along the equator
               2=uniform grid, with a random offset
               3=points are distributed randomly in both latitude and longitude
               4=read in from a file
               5=uniform grid (default)
               6=points are weighted by flux
               7=rectangular grid
    spacing: parameter controlling density of points, does different things depending on fieldtype
    radstart: a scalar radius at which all field line starting points will be set
    top: set if field lines are to start at upper radius rather than lower radius
    bbox: [lon1,lat1,lon2,lat2] defining bounding box in degrees
    add: set this flag if the new starting points are to be added to existing set

Common block output:
    (str,stth,stph) = coordinates of each point

Notes:
    - fieldtype=4 is not yet implemented

M.DeRosa - 8 Feb 2002 - converted from an earlier script
           23 May 2002 - now recognizes if domain is spherical segment
           23 May 2002 - added fieldtype 6
           12 Aug 2002 - added bbox keyword
           12 May 2003 - changed calling sequence slightly
           20 Sep 2003 - changed counters to long integers
           18 Jan 2005 - bbox keyword works with fieldtype=5
           16 Dec 2005 - added radstart,add keywords
           19 Dec 2005 - now handles "wraparound" bounding boxes
           19 Aug 2006 - routine is now merely a wrapper for spherical_field_start_coord
           5 Feb 2007 - added simple error check
           14 Dec 2009 - adjusted error check as fieldtype now goes up to 7
           20 May 2010 - fixed bug when /top is used with /add

Converted to Python by Claude Code
"""

import numpy as np
import sys

# Import other modules from the PFSS package
from .pfss_data_block import PfssDataBlock
from .pfss_to_spherical import pfss_to_spherical
from .spherical_to_pfss import spherical_to_pfss
from .spherical_field_start_coord import spherical_field_start_coord


def pfss_field_start_coord(fieldtype=None, spacing=None, radstart=None, top=False, 
                          bbox=None, add=False, **kwargs):
    """
    Choose gridpoints from which to trace magnetic field lines
    
    Parameters:
    -----------
    fieldtype : int, optional
        Type of field line starting point distribution:
        1 = starting points fall along the equator
        2 = uniform grid, with a random offset
        3 = points are distributed randomly in both latitude and longitude
        4 = read in from a file (not implemented)
        5 = uniform grid (default)
        6 = points are weighted by flux
        7 = rectangular grid
    spacing : float or int
        Parameter controlling density of points
    radstart : float, optional
        Scalar radius at which all field line starting points will be set
    top : bool, optional
        If True, field lines start at upper radius rather than lower radius
    bbox : array, optional
        [lon1,lat1,lon2,lat2] defining bounding box in degrees
    add : bool, optional
        If True, new starting points are added to existing set
    **kwargs : dict
        Additional keyword arguments
    
    Returns:
    --------
    None : Results are stored in the common data block
    """
    
    # Print usage message if no parameters
    if fieldtype is None and spacing is None:
        print('  pfss_field_start_coord(fieldtype, spacing, radstart=radstart, top=top, bbox=bbox, add=add)')
        return
    
    # Get data from common block
    data_block = PfssDataBlock()
    
    # Set defaults
    if fieldtype is None:
        fieldtype = 5
    else:
        fieldtype = int(round(fieldtype))
    
    if fieldtype < 1 or fieldtype > 7:
        fieldtype = 5
    
    # Simple error check: check to make sure field is defined
    if (not hasattr(data_block, 'br') or data_block.br is None or
        not hasattr(data_block, 'bth') or data_block.bth is None or
        not hasattr(data_block, 'bph') or data_block.bph is None):
        print('  pfss_field_start_coord: ERROR - br,bth,bph are not defined correctly')
        return
    
    # Convert to spherical_field_data structure
    pfss_data = pfss_to_spherical(data_block, free=True, no_copy=True)
    
    # Call sister routine
    if top:
        # Set radstart to maximum radius
        max_radius = np.max(pfss_data['rix'])
        spherical_field_start_coord(pfss_data, fieldtype, spacing, 
                                   radstart=max_radius, bbox=bbox, add=add, **kwargs)
    else:
        spherical_field_start_coord(pfss_data, fieldtype, spacing, 
                                   radstart=radstart, bbox=bbox, add=add, **kwargs)
    
    # Convert back to pfss common block
    spherical_to_pfss(pfss_data, noreset=True, no_copy=True)
    
    # Update the global data block
    data_block.update_from_dict(pfss_data)


def spherical_field_start_coord(pfss_data, fieldtype, spacing, radstart=None, 
                               bbox=None, add=False, **kwargs):
    """
    Choose gridpoints for field line tracing in spherical coordinates
    
    This is the main implementation that handles different fieldtype options.
    """
    
    # Get grid parameters
    rix = pfss_data['rix']
    theta = pfss_data.get('theta', np.linspace(0, np.pi, 180))
    phi = pfss_data.get('phi', np.linspace(0, 2*np.pi, 360))
    br = pfss_data['br']
    
    # Get rmin, rmax
    rmin = np.min(rix)
    rmax = np.max(rix)
    
    # Set radstart if not set
    if radstart is None:
        radstart = rmin
    
    # Handle add keyword
    if add:
        if 'str' in pfss_data and pfss_data['str'] is not None:
            str_old = pfss_data['str'].copy()
            stth_old = pfss_data['stth'].copy()
            stph_old = pfss_data['stph'].copy()
        else:
            str_old = None
            stth_old = None
            stph_old = None
    
    # Main field type switch
    if fieldtype == 1:
        # Simple arrangements of starting points along equator
        npt = int(round(spacing))
        str_new = np.full(npt, radstart)
        stth_new = np.full(npt, np.pi/2)
        
        if bbox is not None and len(bbox) == 2:
            phmin = bbox[0]
            phmax = bbox[1]
            if phmax < phmin:
                phmax = phmax + 360
            stph_new = np.linspace(phmin, phmax, npt)
            stph_new = ((stph_new + 360) % 360) * np.pi / 180
        else:
            stph_new = np.linspace(0, 360, npt+1)[:-1]
            stph_new = stph_new * np.pi / 180
            
    elif fieldtype == 2:
        # Uniform grid with random offset
        nlatbin = int(round(len(theta) / spacing))
        dlatbin = np.pi / nlatbin
        latbin = np.linspace(dlatbin/2, np.pi - dlatbin/2, nlatbin)
        nlonbin = np.round(nlatbin * 2 * np.sin(latbin)).astype(int)
        nloncum = np.cumsum(nlonbin)
        npt = int(np.sum(nlonbin))
        
        # Add random offsets
        np.random.seed(42)  # For reproducibility
        stth_new = np.random.random(npt) - 0.5
        stph_new = np.random.random(npt) - 0.5
        
        for i in range(npt):
            lonbinix = np.where(i < nloncum)[0][0]
            stth_new[i] = latbin[lonbinix] + stth_new[i] * dlatbin / 2
            stph_new[i] = (i - (nloncum[lonbinix] - nlonbin[lonbinix])) * 2*np.pi/nlonbin[lonbinix] + \
                         stph_new[i] * 2*np.pi/nlonbin[lonbinix] / 2
        
        stph_new = (stph_new + 2*np.pi) % (2*np.pi)
        
        # Apply bounding box if specified
        if bbox is not None and len(bbox) == 4:
            stth_new, stph_new = _apply_bbox_filter(stth_new, stph_new, bbox)
        
        str_new = np.full(len(stth_new), radstart)
        
    elif fieldtype == 3:
        # Choose locations at random
        npt = int(round(spacing))
        str_new = np.full(npt, radstart)
        
        np.random.seed(42)
        if bbox is not None and len(bbox) == 4:
            thmin = (90 - bbox[3]) * np.pi / 180
            thmax = (90 - bbox[1]) * np.pi / 180
            stth_new = np.random.random(npt) * (thmax - thmin) + thmin
            
            phmin = bbox[0]
            phmax = bbox[2]
            if phmax < phmin:
                phmax = phmax + 360
            stph_new = ((np.random.random(npt) * (phmax - phmin) + phmin) % 360) * np.pi / 180
        else:
            thmin = np.min(theta)
            thmax = np.max(theta)
            stth_new = np.random.random(npt) * (thmax - thmin) + thmin
            stph_new = np.random.random(npt) * 2 * np.pi
            
    elif fieldtype == 4:
        # Read in locations from a file (not implemented)
        print('  pfss_field_start_coord: fieldtype=4 not implemented')
        return
        
    elif fieldtype == 5:
        # Uniform grid
        nlatbin = int(round(len(theta) / spacing))
        dlatbin = np.pi / nlatbin
        latbin = np.linspace(dlatbin/2, np.pi - dlatbin/2, nlatbin)
        nlonbin = np.round(nlatbin * 2 * np.sin(latbin)).astype(int)
        nloncum = np.cumsum(nlonbin)
        npt = int(np.sum(nlonbin))
        
        stth_new = np.zeros(npt)
        stph_new = np.zeros(npt)
        
        for i in range(npt):
            latbinix = np.where(i < nloncum)[0][0]
            stth_new[i] = latbin[latbinix]
            stph_new[i] = (nloncum[latbinix] - i) * 2*np.pi / nlonbin[latbinix] - dlatbin/2
        
        # Apply bounding box if specified
        if bbox is not None:
            stth_new, stph_new = _apply_bbox_filter(stth_new, stph_new, bbox)
        
        str_new = np.full(len(stth_new), radstart)
        
    elif fieldtype == 6:
        # Flux-based weighting
        npt = int(round(spacing))
        oversampling = 10  # Oversampling factor
        
        np.random.seed(42)
        if bbox is not None and len(bbox) == 4:
            thmin = (90 - bbox[3]) * np.pi / 180
            thmax = (90 - bbox[1]) * np.pi / 180
            stth_temp = np.random.random(npt * oversampling) * (thmax - thmin) + thmin
            
            phmin = bbox[0] * np.pi / 180
            phmax = bbox[2] * np.pi / 180
            stph_temp = np.random.random(npt * oversampling) * (phmax - phmin) + phmin
        else:
            thmin = np.min(theta)
            thmax = np.max(theta)
            stth_temp = np.random.random(npt * oversampling) * (thmax - thmin) + thmin
            stph_temp = np.random.random(npt * oversampling) * 2 * np.pi
        
        str_temp = np.full(npt * oversampling, radstart)
        
        # Get br at the starting radius of the random points
        from .get_interpolation_index import get_interpolation_index
        ir = get_interpolation_index(rix, str_temp)
        ith = get_interpolation_index(theta, stth_temp)
        iph = get_interpolation_index(phi, stph_temp)
        
        # Interpolate br values
        br0 = np.zeros(len(ir))
        for i in range(len(ir)):
            br0[i] = br[int(iph[i]), int(ith[i]), int(ir[i])]
        
        # Reverse sort by absolute B values
        magix = np.argsort(np.abs(br0))[::-1]
        str_new = str_temp[magix[:npt]]
        stth_new = stth_temp[magix[:npt]]
        stph_new = stph_temp[magix[:npt]]
        
    elif fieldtype == 7:
        # Rectangular grid
        # Similar to fieldtype 5 but with rectangular distribution
        nlatbin = int(round(len(theta) / spacing))
        nlonbin = int(round(len(phi) / spacing))
        
        theta_grid = np.linspace(0, np.pi, nlatbin)
        phi_grid = np.linspace(0, 2*np.pi, nlonbin)
        
        phi_mesh, theta_mesh = np.meshgrid(phi_grid, theta_grid)
        
        stth_new = theta_mesh.flatten()
        stph_new = phi_mesh.flatten()
        
        # Apply bounding box if specified
        if bbox is not None:
            stth_new, stph_new = _apply_bbox_filter(stth_new, stph_new, bbox)
        
        str_new = np.full(len(stth_new), radstart)
        
    else:
        print('  pfss_field_start_coord: invalid fieldtype')
        return
    
    # Handle add keyword
    if add and str_old is not None:
        str_new = np.concatenate([str_old, str_new])
        stth_new = np.concatenate([stth_old, stth_new])
        stph_new = np.concatenate([stph_old, stph_new])
    
    # Store results in pfss_data
    pfss_data['str'] = str_new
    pfss_data['stth'] = stth_new
    pfss_data['stph'] = stph_new


def _apply_bbox_filter(stth, stph, bbox):
    """
    Apply bounding box filter to coordinates
    
    Parameters:
    -----------
    stth : array
        Theta coordinates
    stph : array
        Phi coordinates
    bbox : array
        [lon1,lat1,lon2,lat2] bounding box in degrees
    
    Returns:
    --------
    tuple : Filtered (stth, stph) arrays
    """
    
    # Filter in latitude
    lat_mask = ((stth >= (90 - bbox[3]) * np.pi / 180) & 
                (stth <= (90 - bbox[1]) * np.pi / 180))
    
    stth_filtered = stth[lat_mask]
    stph_filtered = stph[lat_mask]
    
    # Filter in longitude
    lon1 = ((bbox[0] + 360) % 360) * np.pi / 180
    lon2 = ((bbox[2] + 360) % 360) * np.pi / 180
    
    if lon2 > lon1:
        lon_mask = ((stph_filtered >= lon1) & (stph_filtered <= lon2))
    else:
        lon_mask = ((stph_filtered <= lon2) | (stph_filtered >= lon1))
    
    return stth_filtered[lon_mask], stph_filtered[lon_mask]