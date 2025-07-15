"""
====================================================================

inv_spherical_transform.py - This routine performs an inverse spherical
                             harmonic transform on a 2-D array.

usage:  A = inv_spherical_transform(B, cp, period=1, lmax=None,
              mrange=None, phirange=None, cprange=None, 
              thindex=None, phindex=None)
     where B(lmax,lmax) = complex array to be transformed ordered (l,m)
           A(n_phi,n_theta) = transformed array ordered (phi,theta)
           cp = cosine of theta collocation points for theta grid
           period = periodicity factor in phi, assumes input array
                    contains m values which are integral multiples
                    of period
           lmax = set to max l value we want to use
           mrange = set to be range of m values we want to use
           phirange = (optional) a one- or two-element array containing the
                     range of phi to return, in radians.  This option is
                     useful for high-resolution transforms where the region
                     of interest is bounded in longitude.  If phirange is
                     one element, range is [0,phirange].  If two elements,
                     range is [phirange(0),phirange(1)].  If not specified,
                     the range is set to is [0,2*pi/period].
           cprange = (optional) a one- or two-element array containing the
                     range of cp to return.  This option is useful for
                     high-resolution transforms where the region of
                     interest is bounded in latitude.  If cprange is one
                     element, range is [-cprange,cprange].  If two
                     elements, range is [min(cprange),max(cprange)].  If
                     not specified, default range is [-1,1] is used.
           thindex = (optional) custom array of theta (colatitude)
                       coordinates, in radians, for output grid.  If not
                       specified, the argument cp and optional keyword
                       cprange are used to determine the theta grid.
           phindex = (optional) custom array of phi (longitude)
                     coordinates, in radians, for output grid.  If not
                     specified, an equally spaced grid with
                     2*n_elements(cp) elements is used.

notes: - All calculations are done in double precision.
       - Default is to return an array of size
         (2*n_elements(cp),n_elements(cp)) unless limits are put on cp
         and/or phi ranges via the cprange and phirange keywords, or a
         custom grid is specified using thindex and phindex (in which case
         phirange and cprange are ignored).
       - Routine is increasingly less accurate for higher l.  To see why,
         look at table of Legendre functions (m=0 example) in Arfken - 
         they are alternating series with increasingly larger numbers being
         added and subtracted from each other.

M.DeRosa - 12 Sep 2000 - created (IDL version)
         - 24 Oct 2001 - fixed nasty bug related to sign of m=0 components
         -  2 Apr 2003 - added phirange and cprange keywords
         -  2 Apr 2007 - now calculates phases of B in a cleaner fashion
         -  2 Apr 2007 - added thindex and phindex keywords
Converted to Python - 2025

====================================================================
"""

import numpy as np
from typing import Optional, Union, Tuple, List


def inv_spherical_transform(B: np.ndarray, cp: np.ndarray, 
                           period: int = 1, lmax: Optional[int] = None,
                           mrange: Optional[Union[int, List[int]]] = None,
                           phirange: Optional[Union[float, List[float]]] = None,
                           cprange: Optional[Union[float, List[float]]] = None,
                           thindex: Optional[np.ndarray] = None,
                           phindex: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Performs an inverse spherical harmonic transform on a 2-D array.
    
    Parameters:
    -----------
    B : numpy.ndarray
        Complex array to be transformed ordered (l,m)
    cp : numpy.ndarray
        Cosine of theta collocation points for theta grid
    period : int, optional
        Periodicity factor in phi (default=1)
    lmax : int, optional
        Max l value to use (default=size of B minus 1)
    mrange : int or list, optional
        Range of m values to use
    phirange : float or list, optional
        Range of phi to return, in radians
    cprange : float or list, optional
        Range of cp to return
    thindex : numpy.ndarray, optional
        Custom theta coordinates for output grid
    phindex : numpy.ndarray, optional
        Custom phi coordinates for output grid
        
    Returns:
    --------
    numpy.ndarray
        Transformed array ordered (phi,theta)
    """
    
    # Convert inputs to numpy arrays
    B = np.array(B, dtype=complex)
    cp = np.array(cp, dtype=np.float64)
    
    # Set defaults
    if lmax is None:
        lmax = B.shape[0] - 1
    else:
        lmax = int(lmax)
    
    period = int(period)
    
    # Handle mrange
    if mrange is None:
        mrange = [0, lmax]
    elif isinstance(mrange, (int, float)):
        mrange = [0, min(int(mrange), lmax)]
    else:
        mrange = [int(mrange[0]), min(int(mrange[1]), lmax)]
    
    # Determine output (co-)latitude grid
    if thindex is not None:
        thindex = np.array(thindex, dtype=np.float64)
        if np.max(thindex) > np.pi or np.min(thindex) < 0:
            raise ValueError("thindex out of range")
        ntheta = len(thindex)
        costheta = np.cos(thindex)
        sintheta = np.sqrt(1 - costheta * costheta)
    else:
        if cprange is None:
            cp1i = -1.0
            cp2i = 1.0
        elif isinstance(cprange, (int, float)):
            cp2i = min(abs(cprange), 1.0)
            cp1i = -cp2i
        else:
            cp1i = max(min(cprange), -1.0)
            cp2i = min(max(cprange), 1.0)
        
        mask = (cp >= cp1i) & (cp <= cp2i)
        if not np.any(mask):
            raise ValueError("invalid cprange")
        
        costheta = cp[mask].astype(np.float64)
        sintheta = np.sqrt(1 - costheta * costheta)
        thindex = np.arccos(costheta)
        ntheta = len(costheta)
    
    # Determine output longitude grid
    if phindex is not None:
        phindex = np.array(phindex, dtype=np.float64)
        nphi = len(phindex)
        phiix = phindex
    else:
        nphi = 2 * len(cp)
        phiix = 2 * np.pi * np.arange(nphi // period) / nphi
        
        if phirange is None:
            ph1i = 0.0
            ph2i = 2 * np.pi / period
        elif isinstance(phirange, (int, float)):
            ph1i = 0.0
            ph2i = min(2 * np.pi / period, phirange)
        else:
            ph1i = max(0.0, min(phirange))
            ph2i = min(2 * np.pi / period, max(phirange))
        
        mask = (phiix >= ph1i) & (phiix <= ph2i)
        if not np.any(mask):
            raise ValueError("invalid phirange")
        
        phiix = phiix[mask]
        nphi = len(phiix)
    
    # Calculate array of amplitudes and phases of B
    Bamp = np.abs(B)
    phase = np.angle(B)
    
    # Set up output array A
    A = np.zeros((nphi, ntheta), dtype=np.float64)
    
    # Take care of modes where m=0
    CP_0_0 = 1.0 / np.sqrt(4 * np.pi)
    
    if mrange[0] == 0:
        # Start with m=l=0 mode
        A += Bamp[0, 0] * np.cos(phase[0, 0]) * CP_0_0
        
        # Now do l=1 m=0 mode
        if lmax >= 1:
            CP_1_0 = np.sqrt(3.0) * costheta * CP_0_0
            Y = np.outer(np.cos(phase[1, 0]), CP_1_0)
            A += Bamp[1, 0] * Y
        
        # Do other l modes for which m=0
        if lmax > 1:
            CP_lm1_0 = CP_0_0 * np.ones_like(costheta)
            CP_l_0 = CP_1_0
            
            for l in range(2, lmax + 1):
                ld = float(l)
                CP_lm2_0 = CP_lm1_0
                CP_lm1_0 = CP_l_0
                c1 = np.sqrt(4 * ld**2 - 1) / ld
                c2 = np.sqrt((2 * ld + 1) / (2 * ld - 3)) * ((ld - 1) / ld)
                CP_l_0 = c1 * costheta * CP_lm1_0 - c2 * CP_lm2_0
                Y = np.outer(np.cos(phase[l, 0]), CP_l_0)
                A += Bamp[l, 0] * Y
    
    # Loop through m's for m>0, and then loop through l's for each m
    CP_m_m = CP_0_0 * np.ones_like(costheta)
    
    for m in range(1, mrange[1] + 1):
        md = float(m)
        
        # Do l=m mode first
        CP_mm1_mm1 = CP_m_m
        CP_m_m = -np.sqrt(1 + 1 / (2 * md)) * sintheta * CP_mm1_mm1
        
        if (mrange[0] <= m) and ((m % period) == 0):
            m_period = m // period
            
            angpart = np.cos(md * phiix + phase[m, m_period])
            A += Bamp[m, m_period] * np.outer(angpart, CP_m_m)
            
            # Now do l=m+1 mode
            if lmax >= m + 1:
                CP_mp1_m = np.sqrt(2 * md + 3) * costheta * CP_m_m
                angpart = np.cos(md * phiix + phase[m + 1, m_period])
                A += Bamp[m + 1, m_period] * np.outer(angpart, CP_mp1_m)
            
            # Now do other l's
            if lmax >= m + 2:
                CP_lm1_m = CP_m_m
                CP_l_m = CP_mp1_m
                
                for l in range(m + 2, lmax + 1):
                    ld = float(l)
                    CP_lm2_m = CP_lm1_m
                    CP_lm1_m = CP_l_m
                    c1 = np.sqrt((4 * ld**2 - 1) / (ld**2 - md**2))
                    c2 = np.sqrt(((2 * ld + 1) * ((ld - 1)**2 - md**2)) / 
                                ((2 * ld - 3) * (ld**2 - md**2)))
                    CP_l_m = c1 * costheta * CP_lm1_m - c2 * CP_lm2_m
                    angpart = np.cos(md * phiix + phase[l, m_period])
                    A += Bamp[l, m_period] * np.outer(angpart, CP_l_m)
    
    return A