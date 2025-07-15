"""
====================================================================

weights_legendre.py - This function returns the Legendre integration
                      weights given an array of collocation points x.

usage:  weights = weights_legendre(x)
     where weights = Legendre integration weights
           x = array of collocation points, i.e. the zeroes of the Legendre
               function (of the first kind) of order nx (where nx =
               number of elements in the array x)

notes:  -normalized to 4pi (unlike gaussquad_legendre.py routine,
         which is normalized to 2)

references: -Arfken, G. 1985, Mathematical Methods for Physicists (San
             Diego, Academic Press) 

M.DeRosa - 13 Oct 2000 - cannibalized from spherical_transform.pro (IDL version)
Converted to Python - 2025

====================================================================
"""

import numpy as np


def weights_legendre(x: np.ndarray) -> np.ndarray:
    """
    Returns the Legendre integration weights given an array of collocation points.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Array of collocation points (zeroes of Legendre function of order nx)
        
    Returns:
    --------
    numpy.ndarray
        Legendre integration weights normalized to 4π
    """
    
    # Convert to numpy array and ensure double precision
    x = np.array(x, dtype=np.float64)
    
    # Preliminaries
    nx = len(x)
    weights = np.zeros(nx, dtype=np.float64)
    costheta = x
    sintheta = np.sqrt(1 - costheta**2)
    
    # Set first two Legendre functions (evaluated at the collocation points)
    Pm2 = np.ones_like(x)
    Pm1 = x
    
    # Iterate through the rest of the functions
    for l in range(2, nx):
        lr = 1.0 / float(l)
        P = (2 - lr) * Pm1 * costheta - (1 - lr) * Pm2  # recursion relation Arfken 12.17a
        Pm2 = Pm1
        Pm1 = P
    
    # Calculate dP_nx/dx evaluated at the collocation points
    # NOTE: P in the expression below is actually P_(nx-1), and in the
    # recursion relation listed below P_nx evaluated at the collocation points
    # is zero, by definition 
    p_deriv = (nx * Pm1) / (sintheta**2)  # recursion relation Arfken 12.26
    
    # Calculate and then renormalize the weights
    weights = 2 / (sintheta * p_deriv)**2
    weights = weights * (2 * np.pi)
    
    return weights