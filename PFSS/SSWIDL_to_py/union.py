"""
====================================================================

union.py - This function returns the union of arr1 and arr2

usage:  result = union(arr1, arr2)

notes:  - Values are sorted and will only appear once in the union
          even if they are repeated in either arr1 or arr2.

M.DeRosa - 1 Nov 1999 - created (IDL version)
Converted to Python - 2025

====================================================================
"""

import numpy as np
from typing import Union, Optional


def union(arr1: Optional[Union[np.ndarray, list]], 
          arr2: Optional[Union[np.ndarray, list]]) -> Optional[np.ndarray]:
    """
    Returns the union of arr1 and arr2.
    
    Parameters:
    -----------
    arr1 : array-like or None
        First array
    arr2 : array-like or None
        Second array
        
    Returns:
    --------
    numpy.ndarray or None
        Sorted union of the two arrays, or None if both are empty
    """
    
    # Handle None/empty cases
    if arr1 is None:
        na1 = 0
    else:
        arr1 = np.asarray(arr1)
        na1 = arr1.size
    
    if arr2 is None:
        na2 = 0
    else:
        arr2 = np.asarray(arr2)
        na2 = arr2.size
    
    # Determine union
    if na1 == 0 and na2 == 0:
        return None
    elif na1 == 0:
        both = arr2.flatten()
    elif na2 == 0:
        both = arr1.flatten()
    else:
        both = np.concatenate([arr1.flatten(), arr2.flatten()])
    
    # Return unique sorted values
    return np.unique(both)