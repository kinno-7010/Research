"""
====================================================================

sign_mld.py - this program takes an input array and returns an integer
              array of the same shape where every element is either -1
              0 or 1, signifying whether the corresponding element in
              the input array is negative, zero, or positive.

usage:  result = sign_mld(array)
     where:  array = input array on which to perform the signing
             result = output array which contains the sign

notes: Originally named sign_mld to avoid confusion with SSW version

M.DeRosa - 24 Jun 1999 - created (IDL version)
         - 15 May 2002 - changed name to sign_mld so as not to confuse
                         with SSW version by the same name
Converted to Python - 2025

====================================================================
"""

import numpy as np
from typing import Union


def sign_mld(array: Union[np.ndarray, list, int, float]) -> np.ndarray:
    """
    Returns the sign of each element in the input array.
    
    Parameters:
    -----------
    array : array-like
        Input array on which to perform the signing
        
    Returns:
    --------
    numpy.ndarray
        Array of same shape containing -1, 0, or 1 for each element
    """
    
    # Convert to numpy array
    array = np.asarray(array)
    
    # Calculate signs using boolean operations
    pos = array > 0
    neg = array < 0
    
    return -1 * neg + pos