"""
====================================================================

forw_euler.py - This function integrates a system of equations of the
                form dy1/dx = f1(x,y1,y2,...) forward in x space
                     dy2/dx = f2(x,y1,y2,...)
                       etc.

usage:  result = forw_euler(x, y, h, derivs)
        where result = a vector of values of y(x+h)
              x = the current value of the independent variable
              y = a vector of values of y(x)
              h = amount to step forward in x
              derivs = either: a string specifying a user-supplied python 
                               function that calculates the derivatives 
                               dy/dx evaluated at an arbitrary point x,
                       or: a vector of values of dy/dx evaluated at x

M.DeRosa - 18 Oct 2001 - created (IDL version)
         - 28 Jun 2012 - added y argument to call of user-supplied function
Converted to Python - 2025

====================================================================
"""

import numpy as np
from typing import Union, Callable, List


def forw_euler(x: Union[float, np.ndarray], y: np.ndarray, h: Union[float, np.ndarray], 
               derivs: Union[str, Callable, np.ndarray]) -> np.ndarray:
    """
    Forward Euler integration method for systems of ODEs.
    
    Parameters:
    -----------
    x : float or array-like
        Current value of independent variable
    y : array-like
        Vector of values of y(x)
    h : float or array-like
        Step size to advance in x
    derivs : str, callable, or array-like
        Either a function name/callable that computes derivatives,
        or an array of derivative values
        
    Returns:
    --------
    numpy.ndarray
        Values of y(x+h)
    """
    
    # Convert inputs to numpy arrays if needed
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(x, np.ndarray):
        x = np.array([x])
    if not isinstance(h, np.ndarray):
        h = np.array([h])
    
    # Get order of system
    order = len(y)
    
    # Determine if derivs is a callable or array
    if callable(derivs) or isinstance(derivs, str):
        # If string, we assume it's a function name in the calling scope
        if isinstance(derivs, str):
            import inspect
            frame = inspect.currentframe().f_back
            if derivs in frame.f_locals:
                func = frame.f_locals[derivs]
            elif derivs in frame.f_globals:
                func = frame.f_globals[derivs]
            else:
                raise ValueError(f"Function '{derivs}' not found in scope")
        else:
            func = derivs
        
        # Call the function to get derivatives
        dydx = func(x[0], y)
    else:
        # derivs is an array of derivative values
        derivs = np.array(derivs)
        if len(derivs) < order:
            raise ValueError("derivs argument has fewer elements than y")
        dydx = derivs[:order]
    
    # Advance solution and return
    return y + h[0] * dydx