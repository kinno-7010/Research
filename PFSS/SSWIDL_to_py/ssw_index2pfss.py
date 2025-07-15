"""
ssw_index2pfss.py - Interface from SSW 'index' or 'map' to PFSS derived field lines

Purpose: interface from SSW 'index' or 'map' to PFSS derived field lines.

Input Parameters:
    sswindex - an ssw 'index' record or Map per D.M.Zarro et al

Output Parameters:
   pfssindex - sswindex and pfss parameter derived composite index
   pfssdata - the 2D field line projection  

Keyword Parameters:
    open_color - if set, value to set OPEN lines
    closed_color - if set, value to set CLOSED lines
    fieldtype - per pfss_field_start_coord field type
    strref - desired start point for field; units=radius 
    bbox - optional subfield for field line restriction tie points
    before/after - optional switches to force Bfield dbase selection
    ecliptic - if set, limit lines to ecliptic open lines
    chboundries - if set, include Coronal Hole boundaries
    draw_field2 - if set, use draw_field2 instead of draw_field
"""

import numpy as np
import warnings
from datetime import datetime

# Import required modules from the converted package
from pfss_data_block import PfssDataBlock
from pfss_restore import pfss_restore
from pfss_time2file import pfss_time2file
from pfss_field_start_coord import pfss_field_start_coord
from pfss_trace_field import pfss_trace_field
from pfss_draw_field import pfss_draw_field
from pfss_draw_field2 import pfss_draw_field2


def ssw_index2pfss(sswindex, open_color=None, closed_color=None, refresh=False,
                   force_remote=False, strref=None, fieldtype=3, 
                   image_reference=None, nlines=None, bbox=None, debug=False,
                   lcent=None, bcent=None, earth_view=False, before=False,
                   after=False, ecliptic=False, spacing=None, chboundries=False,
                   draw_field2=False, crop=False, **kwargs):
    """
    Interface from SSW 'index' or 'map' to PFSS derived field lines.
    
    Args:
        sswindex: SSW index record or map structure
        open_color: Value to set for OPEN field lines
        closed_color: Value to set for CLOSED field lines
        refresh: Force refresh of PFSS data
        force_remote: Force remote access to PFSS data
        strref: Start point for field lines (radius units)
        fieldtype: Field line starting point distribution type
        image_reference: Reference for image coordinates
        nlines: Number of field lines to trace
        bbox: Bounding box for field line restriction
        debug: Enable debug output
        lcent: Central longitude
        bcent: Central latitude
        earth_view: Use Earth view perspective
        before: Select closest preceding B-field
        after: Select closest following B-field
        ecliptic: Limit to ecliptic open lines
        spacing: Spacing parameter for field lines
        chboundries: Include coronal hole boundaries
        draw_field2: Use draw_field2 instead of draw_field
        crop: Crop the output image
        **kwargs: Additional parameters passed to pfss_draw_field
        
    Returns:
        tuple: (pfssindex, pfssdata) where pfssindex is composite index
               and pfssdata is the 2D field line projection
    """
    
    # Validate input
    if sswindex is None:
        raise ValueError("sswindex is required")
    
    # Extract time information from sswindex
    if isinstance(sswindex, dict):
        # Handle dictionary-like structure
        obs_time = sswindex.get('date_obs', sswindex.get('time', None))
    else:
        # Handle other formats
        obs_time = getattr(sswindex, 'date_obs', 
                          getattr(sswindex, 'time', None))
    
    if obs_time is None:
        warnings.warn("No observation time found in sswindex, using current time")
        obs_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    
    # Get PFSS field model file
    if debug:
        print(f"Looking for PFSS model for time: {obs_time}")
    
    try:
        # Get PFSS filename for the observation time
        pfss_file = pfss_time2file(obs_time, before=before, after=after,
                                  ssw_catalog=True, urls=force_remote)
        
        if debug:
            print(f"PFSS file: {pfss_file}")
        
        # Restore PFSS field model
        pfss_restore(pfss_file, refresh=refresh)
        
    except Exception as e:
        print(f"Error loading PFSS model: {e}")
        return None, None
    
    # Set up field line tracing parameters
    if strref is None:
        strref = 1.0  # Default to photosphere
    
    if fieldtype is None:
        fieldtype = 3  # Default to random distribution
    
    if spacing is None:
        spacing = 50  # Default spacing
    
    # Handle bounding box
    if bbox is not None:
        # Convert bbox to appropriate format
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            # Already in correct format
            pass
        elif isinstance(bbox, dict):
            # Extract from structure
            bbox = [bbox.get('xmin', -180), bbox.get('ymin', -90),
                   bbox.get('xmax', 180), bbox.get('ymax', 90)]
        else:
            warnings.warn("Invalid bbox format, ignoring")
            bbox = None
    
    # Set up field line starting coordinates
    try:
        pfss_field_start_coord(fieldtype, spacing, radstart=strref, bbox=bbox)
        
        if debug:
            print(f"Field line starting coordinates set up")
        
    except Exception as e:
        print(f"Error setting up field line coordinates: {e}")
        return None, None
    
    # Trace field lines
    try:
        pfss_trace_field(debug=debug)
        
        if debug:
            print("Field lines traced successfully")
        
    except Exception as e:
        print(f"Error tracing field lines: {e}")
        return None, None
    
    # Extract viewing parameters from sswindex
    if lcent is None:
        lcent = extract_viewing_parameter(sswindex, 'crln_obs', 0.0)
    
    if bcent is None:
        bcent = extract_viewing_parameter(sswindex, 'crlt_obs', 0.0)
    
    # Set up drawing parameters
    draw_params = {
        'bcent': bcent,
        'lcent': lcent,
        'width': kwargs.get('width', 2.5),
        'mag': kwargs.get('mag', 1),
        'imsc': kwargs.get('imsc', 200),
        'thick': kwargs.get('thick', 1),
        'drawopen': kwargs.get('drawopen', True),
        'drawclosed': kwargs.get('drawclosed', True),
        'crop': crop
    }
    
    # Handle color settings
    if open_color is not None:
        draw_params['open_color'] = open_color
    
    if closed_color is not None:
        draw_params['closed_color'] = closed_color
    
    # Draw field lines
    try:
        if draw_field2:
            pfssdata = pfss_draw_field2(**draw_params)
        else:
            pfssdata = pfss_draw_field(**draw_params)
        
        if debug:
            print("Field lines drawn successfully")
        
    except Exception as e:
        print(f"Error drawing field lines: {e}")
        return None, None
    
    # Create composite index
    pfssindex = create_composite_index(sswindex, obs_time, draw_params)
    
    # Handle special cases
    if ecliptic:
        pfssdata = filter_ecliptic_lines(pfssdata)
    
    if chboundries:
        pfssdata = add_coronal_hole_boundaries(pfssdata)
    
    return pfssindex, pfssdata


def extract_viewing_parameter(sswindex, param_name, default_value):
    """
    Extract viewing parameter from sswindex.
    
    Args:
        sswindex: SSW index structure
        param_name: Parameter name to extract
        default_value: Default value if not found
        
    Returns:
        float: Parameter value
    """
    
    if isinstance(sswindex, dict):
        return sswindex.get(param_name, default_value)
    else:
        return getattr(sswindex, param_name, default_value)


def create_composite_index(sswindex, obs_time, draw_params):
    """
    Create composite index combining SSW index and PFSS parameters.
    
    Args:
        sswindex: Original SSW index
        obs_time: Observation time
        draw_params: Drawing parameters
        
    Returns:
        dict: Composite index
    """
    
    # Start with original index
    if isinstance(sswindex, dict):
        pfssindex = sswindex.copy()
    else:
        pfssindex = {}
        # Copy attributes from object
        for attr in dir(sswindex):
            if not attr.startswith('_'):
                pfssindex[attr] = getattr(sswindex, attr, None)
    
    # Add PFSS-specific parameters
    pfssindex.update({
        'pfss_obs_time': obs_time,
        'pfss_bcent': draw_params['bcent'],
        'pfss_lcent': draw_params['lcent'],
        'pfss_width': draw_params['width'],
        'pfss_mag': draw_params['mag'],
        'pfss_processed': True
    })
    
    return pfssindex


def filter_ecliptic_lines(pfssdata):
    """
    Filter field lines to show only ecliptic open lines.
    
    Args:
        pfssdata: Field line data
        
    Returns:
        array: Filtered field line data
    """
    
    # This is a placeholder implementation
    # In practice, would filter based on field line geometry
    warnings.warn("Ecliptic filtering not fully implemented")
    return pfssdata


def add_coronal_hole_boundaries(pfssdata):
    """
    Add coronal hole boundaries to field line data.
    
    Args:
        pfssdata: Field line data
        
    Returns:
        array: Data with coronal hole boundaries
    """
    
    # This is a placeholder implementation
    # In practice, would add boundary detection and overlay
    warnings.warn("Coronal hole boundary detection not fully implemented")
    return pfssdata


def ssw_index2pfss_simple(sswindex, **kwargs):
    """
    Simplified version of ssw_index2pfss for basic usage.
    
    Args:
        sswindex: SSW index record
        **kwargs: Additional parameters
        
    Returns:
        tuple: (pfssindex, pfssdata)
    """
    
    return ssw_index2pfss(sswindex, **kwargs)


def validate_sswindex(sswindex):
    """
    Validate SSW index structure.
    
    Args:
        sswindex: SSW index to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    
    if sswindex is None:
        return False
    
    # Check for required fields
    required_fields = ['date_obs', 'time']
    
    if isinstance(sswindex, dict):
        has_time = any(field in sswindex for field in required_fields)
    else:
        has_time = any(hasattr(sswindex, field) for field in required_fields)
    
    if not has_time:
        warnings.warn("No time information found in sswindex")
        return False
    
    return True


def print_pfss_info(pfssindex, pfssdata):
    """
    Print information about PFSS results.
    
    Args:
        pfssindex: PFSS composite index
        pfssdata: PFSS field line data
    """
    
    print('PFSS Processing Results:')
    print('-' * 24)
    
    if pfssindex is not None:
        print(f'Observation time: {pfssindex.get("pfss_obs_time", "N/A")}')
        print(f'Central longitude: {pfssindex.get("pfss_lcent", "N/A")}°')
        print(f'Central latitude: {pfssindex.get("pfss_bcent", "N/A")}°')
        print(f'Image width: {pfssindex.get("pfss_width", "N/A")} R☉')
        print(f'Magnification: {pfssindex.get("pfss_mag", "N/A")}')
    
    if pfssdata is not None:
        print(f'Field data shape: {pfssdata.shape}')
        print(f'Data range: {np.min(pfssdata):.2f} to {np.max(pfssdata):.2f}')
        print(f'Non-zero pixels: {np.count_nonzero(pfssdata)}')


# For compatibility with IDL calling convention
def ssw_index2pfss_idl(sswindex, **kwargs):
    """
    IDL-compatible wrapper for ssw_index2pfss.
    
    This function maintains the original interface.
    """
    return ssw_index2pfss(sswindex, **kwargs)


# Example usage
def ssw_index2pfss_example():
    """
    Example usage of ssw_index2pfss.
    """
    
    print("SSW Index to PFSS Example")
    print("=" * 25)
    
    # Create synthetic SSW index
    sswindex = {
        'date_obs': '2003-04-05T12:00:00',
        'crln_obs': 90.0,
        'crlt_obs': 7.0,
        'instrume': 'EIT',
        'wavelnth': 195
    }
    
    # Process with PFSS
    pfssindex, pfssdata = ssw_index2pfss(sswindex, debug=True)
    
    if pfssindex is not None and pfssdata is not None:
        print_pfss_info(pfssindex, pfssdata)
        print("Processing completed successfully!")
    else:
        print("Processing failed")


if __name__ == "__main__":
    ssw_index2pfss_example()