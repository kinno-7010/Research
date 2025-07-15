"""
====================================================================

mean_dtheta.py - This function calculates the mean of a function mapped
                 onto the Legendre collocation points

usage:  result = mean_dtheta(A, costheta)

                     1           1
     where result =  -  integral     dx f(x)
                     2          -1
                   
                     1           pi/2
                  =  -  integral     d(theta) sin(theta) f(theta)
                     2          -pi/2

           A = input array/vector with theta dimension first
           costheta = cosine of Legendre collocation points for theta grid

M.DeRosa - 20 Oct 2000 - created (IDL version)
Converted to Python - 2025

====================================================================
"""

import numpy as np
from .weights_legendre import weights_legendre


def mean_dtheta(A: np.ndarray, costheta: np.ndarray) -> np.ndarray:
    """
    Calculates the mean of a function mapped onto the Legendre collocation points.
    
    Parameters:
    -----------
    A : numpy.ndarray
        Input array/vector with theta dimension first
    costheta : numpy.ndarray
        Cosine of Legendre collocation points for theta grid
        
    Returns:
    --------
    numpy.ndarray
        Mean integrated over theta dimension
    """
    
    # Convert inputs to numpy arrays
    A = np.array(A)
    costheta = np.array(costheta, dtype=np.float64)
    
    # Preliminaries
    naxin = A.shape
    ndim = len(naxin)
    nx = len(costheta)
    
    # Error checking
    if nx != naxin[0]:
        raise ValueError("size of costheta and first dim of A do not agree")
    
    # Set output axes
    if ndim == 1:
        naxout = 1
    else:
        naxout = naxin[1:]
    
    # Reform input array
    if ndim > 1:
        dim2 = np.prod(naxin[1:])
        AA = A.reshape(nx, dim2).T
    else:
        AA = A.reshape(1, nx)
        dim2 = 1
    
    # Get integration weights
    weights = weights_legendre(costheta) / (4 * np.pi)
    
    # Integrate
    result = np.dot(weights, AA.T)
    
    # Reshape result
    if ndim > 1:
        result = result.reshape(naxout)
    elif dim2 == 1:
        result = float(result)
    
    return result