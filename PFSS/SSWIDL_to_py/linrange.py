"""
====================================================================

linrange.py - creates a double precision array of n numbers starting
              with min and ending with max, in a linear fashion.

usage:  result = linrange(n, min_val, max_val)
        where result = double-precision array with n elements
                   n = number of points in the array
             max_val = highest number
             min_val = lowest number

notes:  raises ValueError if error detected

M.DeRosa - 12 Apr 1995 - created (IDL version)
         - 30 Nov 1998 - added usage message
Converted to Python - 2025

====================================================================
"""

import numpy as np
from typing import Union


def linrange(n: int, min_val: Union[int, float], max_val: Union[int, float]) -> np.ndarray:
    """
    Creates a linearly spaced array of n numbers from min_val to max_val.
    
    Parameters:
    -----------
    n : int
        Number of points in the array
    min_val : int or float
        Lowest number
    max_val : int or float
        Highest number
        
    Returns:
    --------
    numpy.ndarray
        Double-precision array with n elements linearly spaced from min_val to max_val
    """
    
    if n < 1:
        raise ValueError("n must be at least 1")
    
    np_val = int(n)
    mn = float(min_val)
    mx = float(max_val)
    
    if np_val == 1:
        return np.array([mn], dtype=np.float64)
    
    vec = np.arange(np_val, dtype=np.float64)
    vec = vec * (mx - mn) / (np_val - 1)
    vec = vec + mn
    
    return vec