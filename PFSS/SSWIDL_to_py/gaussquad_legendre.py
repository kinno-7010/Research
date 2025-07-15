"""
====================================================================

gaussquad_legendre.py - This procedure computes the Legendre collocation 
                        points and integration weights over x in (-1,1)

usage:  x, w = gaussquad_legendre(n)
        where n = number of gridpoints
              x = n-element array of collocation points
              w = n-element array of integration weights

notes:  -based on Numerical Recipes routine gauleg, section 4.5, p.145
        -maximum number of iterations is about n for eps=1e-6
        -probably need higher precision variables for n>500 or so
        -sum(w) should equal 2

references: -Press, W.H., Flannery, B.P., Teukolsky, S.A., Vitterling, W.T.
             1992, Numerical Recipes: The Art of Scientific Computing
             (Cambridge: Cambridge University Press)
            -Arfken, G. 1985, Mathematical Methods for Physicists (San
             Diego, Academic Press) 

M.DeRosa - 2 Oct 2001 - created (IDL version)
Converted to Python - 2025

====================================================================
"""

import numpy as np
from typing import Tuple


def gaussquad_legendre(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Legendre collocation points and integration weights.
    
    Parameters:
    -----------
    n : int
        Number of gridpoints
        
    Returns:
    --------
    tuple
        (x, w) where:
        x : numpy.ndarray
            n-element array of collocation points
        w : numpy.ndarray
            n-element array of integration weights
    """
    
    # Check input
    if n < 2:
        raise ValueError("n must be at least 2")
    
    order = int(n)
    
    # Initialize arrays
    x = np.zeros(order, dtype=np.float64)
    pprime = np.zeros(order, dtype=np.float64)
    
    # Set tolerance
    eps = 1e-6  # probably adequate for float64 double precision
    
    # Loop through points
    for i in range(1, (order + 1) // 2 + 1):  # symmetric domain, only do half
        
        # Starting guess for ith root
        guess = np.cos(np.pi * (i - 0.25) / (order + 0.5))
        
        # Iterate until zero is found
        while True:
            # Starting values for P_(n-1) and P_n, evaluated at guess point
            pnm1 = 1.0
            pn = guess
            
            # Find P_n evaluated at guess point, Arfken (12.17a) after n+1 replaces n
            for j in range(2, order + 1):
                pnm2 = pnm1
                pnm1 = pn
                pn = (guess * (2 * j - 1) * pnm1 - (j - 1) * pnm2) / j
            
            # Compute d P_n / dx evaluated at guess point, basically Arfken (12.26)
            dpndx = order * (guess * pn - pnm1) / (guess * guess - 1)
            
            # Use Newton's method to improve guess
            oldguess = guess
            guess = oldguess - pn / dpndx
            
            # Check tolerance
            if abs(guess - oldguess) <= eps:
                break
        
        # Fill x and pprime arrays
        x[i-1] = -guess
        x[order-i] = guess
        pprime[i-1] = dpndx  # may be off by a minus sign, but it gets squared below
        pprime[order-i] = dpndx  # ditto above
    
    # Calculate weights
    w = 2 / ((1 - x*x) * pprime * pprime)
    
    return x, w