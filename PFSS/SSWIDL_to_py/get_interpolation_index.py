"""
====================================================================

get_interpolation_index.py - This function returns the fractional coordinate 
                             index of an input array

usage:  result = get_interpolation_index(array, value)
          where array = input array
                value = values for which index is desired

notes:  -useful for determining the index for interpolation,  
         get_interpolation_index(linrange(11,0,2),.68) returns 3.4, since 
         the value .68 would lie at (fractional) index coordinate 3.4
        -if out of bounds, then function returns either 0 or N-1, where N
         is the number of elements in array
        -assumes that array is monotonically increasing
        -linear interpolation performed, watch out if array is not equally 
         spaced

M.DeRosa - 12 Oct 2001 - created (IDL version)
         - 12 Aug 2002 - value argument now takes arrays as well as scalars
Converted to Python - 2025

====================================================================
"""

import numpy as np
from typing import Union


def get_interpolation_index(array: np.ndarray, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Returns the fractional coordinate index of an input array.
    
    Parameters:
    -----------
    array : numpy.ndarray
        Input array (assumed monotonically increasing)
    value : float or numpy.ndarray
        Values for which index is desired
        
    Returns:
    --------
    float or numpy.ndarray
        Fractional indices corresponding to input values
    """
    
    # Convert inputs to numpy arrays if needed
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    if not isinstance(value, np.ndarray):
        value = np.array([value])
        scalar_input = True
    else:
        scalar_input = False
    
    npt = len(value)
    out = np.zeros(npt, dtype=np.float64)
    
    for i in range(npt):
        if value[i] <= array[0]:
            out[i] = 0.0
        else:
            # Find where array is greater than value[i]
            indices = np.where(array > value[i])[0]
            if len(indices) > 0:
                nix = indices[0] - 1
                extract = array[nix:nix+2]
                out[i] = nix + (value[i] - extract[0]) / (extract[1] - extract[0])
            else:
                out[i] = float(len(array) - 1)
    
    # Return scalar if input was scalar
    if scalar_input:
        return out[0]
    else:
        return out