"""
====================================================================

get_string_number.py - This function creates a string out of a number

usage:  out = get_string_number(number, pad=None)
           where out = output string
                 number = input number
                 pad = number of characters the string must have, routine
                       pads with leading zeroes

M.DeRosa -  1 Aug 2000 - created (IDL version)
         - 14 Nov 2002 - added capability to process an array of numbers
         - 29 Jan 2003 - now returns a scalar (instead of a one-element
                         vector) if number is a scalar
Converted to Python - 2025

====================================================================
"""

import numpy as np
from typing import Union, List, Optional


def get_string_number(number: Union[int, float, np.ndarray, List], 
                     pad: Optional[int] = None) -> Union[str, List[str]]:
    """
    Creates a string representation of a number with optional zero padding.
    
    Parameters:
    -----------
    number : int, float, numpy.ndarray, or list
        Input number(s)
    pad : int, optional
        Number of characters the string must have, pads with leading zeroes
        
    Returns:
    --------
    str or list of str
        String representation(s) of the input number(s)
    """
    
    # Convert to numpy array for processing
    if not isinstance(number, np.ndarray):
        if isinstance(number, (list, tuple)):
            number = np.array(number)
        else:
            number = np.array([number])
            scalar_input = True
    else:
        scalar_input = (number.size == 1)
    
    nel = number.size
    strout = []
    
    for i in range(nel):
        # Round the number and convert to string
        st = str(int(round(number.flat[i])))
        
        # Apply padding if specified
        if pad is not None:
            st = st.zfill(pad)
        
        strout.append(st)
    
    # Return scalar string if input was scalar
    if scalar_input:
        return strout[0]
    else:
        return strout