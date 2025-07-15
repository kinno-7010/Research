"""
pfss_get_chfootprint - This procedure determines the "footprint" of coronal
hole boundaries from an input field model at the input radius. By "footprint" 
is meant the contiguous regions on a radial shell through which open fieldlines pass.

usage: opfield = pfss_get_chfootprint(spacing=spacing, rad=rad, sinlat=sinlat,
                                     dilate=dilate, close=close, usecurrent=usecurrent,
                                     keepnew=keepnew, quiet=quiet)

Parameters:
    spacing: density of fieldline starting points that gets passed to
             pfss_field_start_coord with fieldtype=5 (default=50)
    rad: the radial shell on which to perform this analysis (default = 1.0 = photosphere)
    sinlat: if set, the output opfield array will be on a sin(latitude)-longitude grid,
            rather than the Legendre grid used throughout the pfss package
    dilate: if set, will dilate the output array commensurate with the gridpoint spacing
    close: if set, will perform a morphological close (dilate followed by erode)
    usecurrent: if set, computes the coronal hole footprint from existing field lines
    keepnew: if set, the field lines traced remain in the common block
    quiet: set to disable screen output

Returns:
    opfield: a trinary array of open field locations at the same resolution as the
             input magnetic field data. Contains -1 or 1 in pixels that have open
             field, with sign indicating polarity. All other pixels are 0.

Notes:
    - Holes and narrow isthmi can be removed by using morphological operators
    - Sometimes returns open polar cap holes that are too sparsely filled
    - If both positive and negative polarity open field lines intersect the same
      coordinate, the grid is coded with the polarity of the last field line traced
    - If using usecurrent, remember to specify the gridpoint spacing used when
      tracing the current set of fieldlines

M.DeRosa - 21 Apr 2004 - created, adapted from chbounds.pro
           27 Apr 2004 - added usecurrent and keepnew flags
           11 May 2004 - added rad keyword
           14 Dec 2005 - added spacing,dilate,close,sinlat keywords
           14 Aug 2007 - fixed "problem with line crossings" problem
            3 May 2021 - /close and /dilate now both work with /sinlat
"""

import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import binary_dilation, binary_erosion
import warnings

# Import required modules from the converted package
from pfss_data_block import PfssDataBlock
from pfss_field_start_coord import pfss_field_start_coord
from pfss_trace_field import pfss_trace_field
from pfss_rad_field_crossing import pfss_rad_field_crossing
from get_interpolation_index import get_interpolation_index
from sign_mld import sign_mld


def pfss_get_chfootprint(spacing=50, rad=None, sinlat=False, dilate=False, 
                        close=False, usecurrent=False, keepnew=False, quiet=False):
    """
    Determine the footprint of coronal hole boundaries from an input field model.
    
    Args:
        spacing (int): Density of fieldline starting points (default=50)
        rad (float): Radial shell for analysis (default=min(rix))
        sinlat (bool): Use sin(latitude)-longitude grid instead of Legendre grid
        dilate (bool): Dilate the output array
        close (bool): Perform morphological close operation
        usecurrent (bool): Use existing field lines from common block
        keepnew (bool): Keep traced field lines in common block
        quiet (bool): Disable screen output
        
    Returns:
        numpy.ndarray: Trinary array of open field locations
    """
    
    # Access the common block data
    data_block = PfssDataBlock()
    
    # Get radius
    if rad is not None:
        radius = max(min(data_block.rix), min(rad, max(data_block.rix)))
    else:
        radius = min(data_block.rix)
    
    # Get spacing
    spacing = int(round(spacing))
    
    # Trace field lines
    if not usecurrent:
        # Save current field line data
        if not keepnew:
            saved_data = {}
            if hasattr(data_block, 'str') and data_block.str is not None:
                saved_data['str'] = data_block.str.copy()
            if hasattr(data_block, 'stth') and data_block.stth is not None:
                saved_data['stth'] = data_block.stth.copy()
            if hasattr(data_block, 'stph') and data_block.stph is not None:
                saved_data['stph'] = data_block.stph.copy()
            if hasattr(data_block, 'ptr') and data_block.ptr is not None:
                saved_data['ptr'] = data_block.ptr.copy()
            if hasattr(data_block, 'ptth') and data_block.ptth is not None:
                saved_data['ptth'] = data_block.ptth.copy()
            if hasattr(data_block, 'ptph') and data_block.ptph is not None:
                saved_data['ptph'] = data_block.ptph.copy()
            if hasattr(data_block, 'nstep') and data_block.nstep is not None:
                saved_data['nstep'] = data_block.nstep.copy()
        
        # Get coordinates of field line starting points: from source surface inward
        pfss_field_start_coord(5, spacing, radstart=max(data_block.rix))
        
        # Get coordinates of field line starting points: from photosphere outward
        pfss_field_start_coord(5, spacing, radstart=data_block.rix[1], add=True)
        
        # Trace field
        pfss_trace_field(data_block.kind, data_block.stbr, quiet=quiet, oneway=True)
    
    # Get lat/lon coordinates at which each field line intersects photosphere
    sldata, interpindex = pfss_rad_field_crossing(radius)
    
    # Create logical open field grid
    nlines = len(data_block.nstep)
    open_lines = np.zeros(nlines, dtype=bool)
    
    for j in range(nlines):
        open_lines[j] = np.max(data_block.ptr[:data_block.nstep[j], j]) >= max(data_block.rix)
    
    if sinlat:
        # Sin(latitude)-longitude grid
        opfield = np.zeros((data_block.nlon, data_block.nlat), dtype=float)
        dslat = 2.0 / data_block.nlat
        opx = data_block.lon
        opy = np.arange(data_block.nlat) * dslat + dslat/2 - 1
        
        for j in range(nlines):
            if open_lines[j]:
                wh = np.where(np.round(sldata[0, :]) == j)[0]
                nwh = len(wh)
                
                if nwh == 0:
                    continue
                elif nwh == 1:
                    thci = get_interpolation_index(opy, np.sin(sldata[1, wh] * np.pi/180))
                    phci = get_interpolation_index(data_block.lon, sldata[2, wh])
                    
                    if not hasattr(data_block, 'stbr') or data_block.stbr is None:
                        thcix = get_interpolation_index(data_block.lat, 
                                                      90 - data_block.stth[j] * 180/np.pi)
                        phcix = get_interpolation_index(data_block.phi, data_block.stph[j])
                        # Interpolate br field
                        coords = np.array([phcix, thcix, data_block.nr-1]).reshape(1, -1)
                        signbr = sign_mld(interpn((data_block.phi, data_block.lat, 
                                                 np.arange(data_block.nr)), 
                                                data_block.br, coords, method='linear')[0])
                    else:
                        signbr = sign_mld(data_block.stbr[j])
                    
                    opfield[int(round(phci)), int(round(thci))] = signbr
                    
                elif nwh > 1:
                    # Choose point closest to open endpoint
                    warnings.warn("Multiple intersection points - using closest to open endpoint")
                    temp = np.abs(data_block.ptr[:data_block.nstep[j], j] - max(data_block.rix))
                    indexopen = np.argmin(temp)
                    indexslice = interpindex[wh]
                    whindex = np.argmin(np.abs(indexslice - indexopen))
                    
                    thci = get_interpolation_index(opy, np.sin(sldata[1, wh[whindex]] * np.pi/180))
                    phci = get_interpolation_index(data_block.lon, sldata[2, wh[whindex]])
                    
                    if not hasattr(data_block, 'stbr') or data_block.stbr is None:
                        thcix = get_interpolation_index(data_block.lat, 
                                                      90 - data_block.stth[j] * 180/np.pi)
                        phcix = get_interpolation_index(data_block.phi, data_block.stph[j])
                        coords = np.array([phcix, thcix, data_block.nr-1]).reshape(1, -1)
                        signbr = sign_mld(interpn((data_block.phi, data_block.lat, 
                                                 np.arange(data_block.nr)), 
                                                data_block.br, coords, method='linear')[0])
                    else:
                        signbr = sign_mld(data_block.stbr[j])
                    
                    opfield[int(round(phci)), int(round(thci))] = signbr
        
        # Convert back to degrees for dilate/close operations
        opy = np.arcsin(opy) * 180 / np.pi
        
    else:
        # Regular Legendre grid
        opfield = np.zeros((data_block.nlon, data_block.nlat), dtype=float)
        opx = data_block.lon
        opy = data_block.lat
        
        for j in range(nlines):
            if open_lines[j]:
                wh = np.where(np.round(sldata[0, :]) == j)[0]
                nwh = len(wh)
                
                if nwh == 0:
                    continue
                elif nwh == 1:
                    thci = get_interpolation_index(data_block.lat, sldata[1, wh])
                    phci = get_interpolation_index(data_block.lon, sldata[2, wh])
                    
                    if not hasattr(data_block, 'stbr') or data_block.stbr is None:
                        coords = np.array([phci, thci, data_block.nr-1]).reshape(1, -1)
                        signbr = sign_mld(interpn((data_block.phi, data_block.lat, 
                                                 np.arange(data_block.nr)), 
                                                data_block.br, coords, method='linear')[0])
                    else:
                        signbr = sign_mld(data_block.stbr[j])
                    
                    opfield[int(round(phci)), int(round(thci))] = signbr
                    
                elif nwh > 1:
                    # Choose point closest to open endpoint
                    temp = np.abs(data_block.ptr[:data_block.nstep[j], j] - max(data_block.rix))
                    indexopen = np.argmin(temp)
                    indexslice = interpindex[wh]
                    whindex = np.argmin(np.abs(indexslice - indexopen))
                    
                    thci = get_interpolation_index(data_block.lat, sldata[1, wh[whindex]])
                    phci = get_interpolation_index(data_block.lon, sldata[2, wh[whindex]])
                    
                    if not hasattr(data_block, 'stbr') or data_block.stbr is None:
                        coords = np.array([phci, thci, data_block.nr-1]).reshape(1, -1)
                        signbr = sign_mld(interpn((data_block.phi, data_block.lat, 
                                                 np.arange(data_block.nr)), 
                                                data_block.br, coords, method='linear')[0])
                    else:
                        signbr = sign_mld(data_block.stbr[j])
                    
                    opfield[int(round(phci)), int(round(thci))] = signbr
    
    # Apply morphological operations if requested
    if (close or dilate) and (spacing > 1):
        # Calculate dilation/erosion radius in pixels
        radius_deg = 360.0 * spacing / data_block.nlon / 2
        radius_pix = int(np.ceil(radius_deg * data_block.nlon / 360.0))
        
        # Create structuring element
        struct = np.ones((2*radius_pix + 1, 2*radius_pix + 1))
        
        # Separate positive and negative regions
        pos_field = (opfield > 0).astype(float)
        neg_field = (opfield < 0).astype(float)
        
        # Apply dilation
        if dilate or close:
            pos_dilated = binary_dilation(pos_field, structure=struct).astype(float)
            neg_dilated = binary_dilation(neg_field, structure=struct).astype(float)
            
            if not close:
                opfield = pos_dilated - neg_dilated
        
        # Apply erosion for close operation
        if close:
            pos_closed = binary_erosion(pos_dilated, structure=struct).astype(float)
            neg_closed = binary_erosion(neg_dilated, structure=struct).astype(float)
            opfield = pos_closed - neg_closed
    
    # Restore field line data in common block
    if not usecurrent and not keepnew:
        for key, value in saved_data.items():
            setattr(data_block, key, value)
    
    return opfield


# For compatibility with IDL calling convention
def pfss_get_chfootprint_idl(spacing=50, rad=None, sinlat=False, dilate=False, 
                            close=False, usecurrent=False, keepnew=False, quiet=False):
    """
    IDL-compatible wrapper for pfss_get_chfootprint.
    
    This function modifies the global data block and returns the opfield array.
    """
    return pfss_get_chfootprint(spacing=spacing, rad=rad, sinlat=sinlat, 
                               dilate=dilate, close=close, usecurrent=usecurrent,
                               keepnew=keepnew, quiet=quiet)