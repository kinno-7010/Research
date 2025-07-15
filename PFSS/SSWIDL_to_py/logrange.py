"""
====================================================================

logrange.py - creates a double precision array of n numbers starting
              with min and ending with max, in a logarithmic fashion.

usage:  result = logrange(n, min_val, max_val)
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


def logrange(n: int, min_val: Union[int, float], max_val: Union[int, float]) -> np.ndarray:
    """
    Creates a logarithmically spaced array of n numbers from min_val to max_val.
    
    Parameters:
    -----------
    n : int
        Number of points in the array
    min_val : int or float
        Lowest number (must be positive)
    max_val : int or float
        Highest number (must be positive)
        
    Returns:
    --------
    numpy.ndarray
        Double-precision array with n elements logarithmically spaced from min_val to max_val
    """
    
    if n < 1:
        raise ValueError("n must be at least 1")
    
    if min_val <= 0 or max_val <= 0:
        raise ValueError("min_val and max_val must be positive for logarithmic spacing")
    
    np_val = int(n)
    lmin = np.log10(float(min_val))
    lmax = np.log10(float(max_val))
    
    # Use our own linrange function - import it locally
    from .linrange import linrange
    
    vec = 10 ** linrange(np_val, lmin, lmax)
    
    return vec