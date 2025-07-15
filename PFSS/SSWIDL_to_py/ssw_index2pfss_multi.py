"""
ssw_index2pfss_multi.py - Multi-image version of ssw_index2pfss

This routine processes multiple SSW index records to create PFSS field line overlays.
"""

import numpy as np
from ssw_index2pfss import ssw_index2pfss, print_pfss_info


def ssw_index2pfss_multi(sswindex_array, combine_method='average', **kwargs):
    """
    Process multiple SSW index records for PFSS field line overlays.
    
    Args:
        sswindex_array: Array of SSW index records
        combine_method: Method to combine results ('average', 'overlay', 'sequence')
        **kwargs: Additional parameters passed to ssw_index2pfss
        
    Returns:
        tuple: (pfssindex_array, pfssdata_array) with results for each input
    """
    
    print(f"Processing {len(sswindex_array)} SSW index records")
    
    pfssindex_array = []
    pfssdata_array = []
    
    for i, sswindex in enumerate(sswindex_array):
        print(f"Processing index {i+1}/{len(sswindex_array)}")
        
        pfssindex, pfssdata = ssw_index2pfss(sswindex, **kwargs)
        
        pfssindex_array.append(pfssindex)
        pfssdata_array.append(pfssdata)
    
    # Combine results if requested
    if combine_method != 'sequence':
        pfssdata_combined = combine_pfss_data(pfssdata_array, combine_method)
        return pfssindex_array, pfssdata_combined
    
    return pfssindex_array, pfssdata_array


def combine_pfss_data(pfssdata_array, method='average'):
    """Combine multiple PFSS data arrays."""
    
    # Remove None entries
    valid_data = [data for data in pfssdata_array if data is not None]
    
    if not valid_data:
        return None
    
    if method == 'average':
        return np.mean(valid_data, axis=0)
    elif method == 'overlay':
        return np.sum(valid_data, axis=0)
    else:
        return valid_data[0]


if __name__ == "__main__":
    print("ssw_index2pfss_multi - Multi-image PFSS processing")