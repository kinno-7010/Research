"""
Newton-Raphson 3D Root Finding Algorithm

A quick-and-dirty three-dimensional version of the iterative
Newton-Raphson method for finding a null point in a vector field.  For a
vector field Bvec(xvec), the main step is:
  xvec_{n+1} = xvec_{n} - [ dB_i/dx_j | xvec_{n} ] ^ {-1} * Bvec(xvec)

CALLING SEQUENCE:
    result = newton_raphson_3d(funcname, xstart, status=None, itmax=10, loud=False)

INPUTS:
    funcname = a callable function that evaluates Bvec given a field point
               xvec (syntax = Bvec, dmatrix = funcname(xvec)), where
               xvec and Bvec are 3-element arrays of type float or double,
               and dmatrix is the matrix of dB_i/dx_j values at xvec 
    xstart = a 3-element array containing the starting point for the iteration

KEYWORDS:
    status = a status parameter that indicates problems with the minimization
             process: 
               0 = no problems encountered
               1 = inversion attempted on a singular array (which indicates
                   that the inversion is invalid)
               2 = warning that a small pivot element was used during the
                   matrix inversion step, and that significant accuracy was
                   probably lost
               3 = iteration step limit was reached before an acceptable
                   minimum was found
    itmax = iteration limit (default=10)
    loud = if True, prints iteration number and value of x and B

OUTPUTS:
    result = a 3-element array containing a root of Bvec, assuming everything
             went well 

NOTES:
    -The routine has not undergone extensive testing (so be careful!).

MODIFICATION HISTORY:
    M.DeRosa - 22 Nov 2010 - created (IDL version)
    Converted to Python - 2025
"""

import numpy as np
from typing import Callable, Tuple, Optional


def newton_raphson_3d(funcname: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]], 
                      xstart: np.ndarray, 
                      status: Optional[int] = None,
                      itmax: int = 10, 
                      loud: bool = False) -> np.ndarray:
    """
    Newton-Raphson method for finding a null point in a 3D vector field.
    
    Parameters:
    -----------
    funcname : callable
        Function that evaluates Bvec and returns (Bvec, dmatrix)
    xstart : np.ndarray
        3-element array containing the starting point
    status : int, optional
        Status parameter (returned by reference in original IDL)
    itmax : int, default=10
        Maximum number of iterations
    loud : bool, default=False
        If True, prints iteration information
        
    Returns:
    --------
    np.ndarray
        3-element array containing a root of Bvec
    """
    
    # Usage check
    if not callable(funcname):
        raise ValueError("ERROR in newton_raphson_3d: funcname must be callable")
        
    # Input validation
    xstart = np.asarray(xstart)
    if xstart.size != 3:
        raise ValueError("ERROR in newton_raphson_3d: xstart must have 3 elements")
        
    # Starting point
    p0 = xstart.copy()
    bp0, dbidxj = funcname(p0)
    
    # Initialize iteration variables
    iter_count = 0
    status_val = 0
    flag = 0
    
    if loud:
        print(f"{iter_count}: {np.concatenate([p0, bp0])}")
    
    # Main iteration loop
    while flag == 0:
        try:
            # Invert matrix
            dbidxj_inv = np.linalg.inv(dbidxj)
            
            # Check for numerical issues
            cond_num = np.linalg.cond(dbidxj)
            if cond_num > 1e12:
                status_val = max(status_val, 2)  # Warning about accuracy loss
                
        except np.linalg.LinAlgError:
            status_val = max(status_val, 1)  # Singular matrix
            dbidxj_inv = np.linalg.pinv(dbidxj)  # Use pseudoinverse
            
        # Compute next point
        p1 = p0 - np.dot(dbidxj_inv, bp0)
        
        # Test new value
        bp1, dbidxj = funcname(p1)
        
        # Check for convergence
        if np.sqrt(np.sum(bp1**2)) < 1e-5:
            flag = 1  # Hit null
            
        # Check iteration limit
        if iter_count >= itmax:
            flag = 2
            status_val = 3
            
        # Update for next iteration
        p0 = p1
        bp0 = bp1
        iter_count += 1
        
        if loud:
            print(f"{iter_count}: {np.concatenate([p0, bp0])}")
    
    # Set status if it was provided as mutable reference
    if status is not None and hasattr(status, '__setitem__'):
        status[0] = status_val
    
    return p0


# Example usage function for testing
def example_vector_field(xvec):
    """
    Example vector field function that returns both the field value and derivative matrix.
    This is just for demonstration - real usage would provide an actual field function.
    """
    x, y, z = xvec
    
    # Simple example: B = (x^2 + y, y^2 + z, z^2 + x)
    bvec = np.array([x**2 + y, y**2 + z, z**2 + x])
    
    # Derivative matrix dB_i/dx_j
    dmatrix = np.array([
        [2*x, 1, 0],      # dB_x/dx, dB_x/dy, dB_x/dz
        [0, 2*y, 1],      # dB_y/dx, dB_y/dy, dB_y/dz
        [1, 0, 2*z]       # dB_z/dx, dB_z/dy, dB_z/dz
    ])
    
    return bvec, dmatrix


if __name__ == "__main__":
    # Test the function
    start_point = np.array([1.0, 1.0, 1.0])
    result = newton_raphson_3d(example_vector_field, start_point, loud=True)
    print(f"Final result: {result}")