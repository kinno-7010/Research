"""
====================================================================

spherical_transform.py - This routine performs a spherical harmonic
                         transform on a 2-D array.

usage:  B = spherical_transform(A, cp, lmax=None, period=1)
     where B(lmax,lmax) = transformed array ordered (l,m)
           A(n_phi,n_theta) = array to be transformed ordered (phi,theta)
           cp = cosine of theta collocation points for theta grid
           lmax = maximum l in expansion (default is (2*n_theta-1)/3)
           period = periodicity factor in phi

notes: - All calculations are done in double precision.
       - Companion routine inv_spherical_transform.py contains the
         normalization coefficient 1/sqrt(4*pi), such that the mean of A,
         given by mean_dtheta(total((br(*,*,0)),1),cp)/n_phi, will be
         a factor of sqrt(4*pi) less than B(0,0)

M.Miesch - 15 Oct 1997 - acquired from Mark (IDL version)
M.DeRosa - 25 Oct 1999 - modified slightly (basic algorithm is the
                         same as Mark's original version)
         - 27 Oct 1999 - added period keyword
         - 12 Sep 2000 - added weights keyword
         - 12 Sep 2000 - changed normalization to something more sensible
         - 14 Sep 2000 - fixed bug in m,l loop (near end of routine)
         - 13 Oct 2000 - uses weights_legendre to compute theta integration
                         weights, weights keyword now obsolete
Converted to Python - 2025

====================================================================
"""

import numpy as np
from typing import Optional
from .weights_legendre import weights_legendre


def spherical_transform(A: np.ndarray, cp: np.ndarray, 
                       lmax: Optional[int] = None, 
                       period: int = 1) -> np.ndarray:
    """
    Performs a spherical harmonic transformation on a 2-D array.
    
    Parameters:
    -----------
    A : numpy.ndarray
        Array to be transformed, shape (n_phi, n_theta)
    cp : numpy.ndarray
        Cosine of theta collocation points for theta grid
    lmax : int, optional
        Maximum l in expansion (default is (2*n_theta-1)/3)
    period : int, optional
        Periodicity factor in phi (default=1)
        
    Returns:
    --------
    numpy.ndarray
        Transformed array ordered (l,m), shape (lmax+1, lmax/period+1)
    """
    
    # Convert inputs to numpy arrays
    A = np.array(A)
    costheta = np.array(cp, dtype=np.float64)
    sintheta = np.sqrt(1 - costheta**2)
    
    # Get array dimensions
    n_phi, n_theta = A.shape
    
    # Set defaults
    if lmax is None:
        lmax = (2 * n_theta - 1) // 3
    else:
        lmax = int(lmax)
    
    period = int(period)
    
    # First compute the integration weights
    weights = weights_legendre(costheta)
    
    # Next do the Fourier transform: phi -> m
    Bm = np.zeros((n_phi, n_theta), dtype=complex)
    for i in range(n_theta):
        Bm[:, i] = np.fft.fft(A[:, i])
    
    # Finally the Legendre transform: theta -> l
    B = np.zeros((lmax + 1, lmax // period + 1), dtype=complex)
    
    # Define N_mm such that Y_mm = N_mm sin^m(theta) exp(i m phi) i.e. it's the
    # normalization for the sectoral harmonics.  It will be useful below in
    # computing the spherical harmonics recursively.
    N_mm = np.zeros(lmax + 1, dtype=np.float64)
    N_mm[0] = 1.0 / np.sqrt(4.0 * np.pi)
    for m in range(1, lmax + 1):
        N_mm[m] = -N_mm[m - 1] * np.sqrt(1 + 1.0 / (2.0 * m))
    
    # First do m=0
    P_lm2 = N_mm[0] * np.ones_like(costheta)
    P_lm1 = P_lm2 * costheta * np.sqrt(3.0)
    
    # Set l=0 m=0 term
    B[0, 0] = np.sum(Bm[0, :] * P_lm2 * weights)
    
    # Set l=1 m=0 term
    if lmax >= 1:
        B[1, 0] = np.sum(Bm[0, :] * P_lm1 * weights)
    
    # Set m=0 term for all other l's
    for l in range(2, lmax + 1):
        lr = float(l)
        c1 = np.sqrt(4.0 - 1.0 / lr**2)
        c2 = -(1 - 1.0 / lr) * np.sqrt((2 * lr + 1) / (2 * lr - 3))
        P_l = c1 * costheta * P_lm1 + c2 * P_lm2
        B[l, 0] = np.sum(Bm[0, :] * P_l * weights)
        P_lm2 = P_lm1
        P_lm1 = P_l
    
    # Note factor of 2 below accounts for the way FFT distributes power
    # since only the l modes from 1 to lmax are used below
    Bm = 2 * Bm
    
    # Now the rest of the m's
    old_Pmm = N_mm[0] * np.ones_like(costheta)
    
    for m in range(1, lmax + 1):
        P_lm2 = old_Pmm * sintheta * N_mm[m] / N_mm[m - 1]
        P_lm1 = P_lm2 * costheta * np.sqrt(2.0 * m + 3)
        old_Pmm = P_lm2
        
        if (m % period) == 0:
            m_period = m // period
            
            # Set l=m term
            B[m, m_period] = np.sum(Bm[m_period, :] * P_lm2 * weights)
            
            # Set l=m+1 term
            if m < lmax:
                B[m + 1, m_period] = np.sum(Bm[m_period, :] * P_lm1 * weights)
            
            mr = float(m)
            for l in range(m + 2, lmax + 1):
                lr = float(l)
                c1 = np.sqrt((4 * lr**2 - 1) / (lr**2 - mr**2))
                c2 = -np.sqrt(((2 * lr + 1) * ((lr - 1)**2 - mr**2)) / 
                             ((2 * lr - 3) * (lr**2 - mr**2)))
                P_l = c1 * costheta * P_lm1 + c2 * P_lm2
                
                B[l, m_period] = np.sum(Bm[m_period, :] * P_l * weights)
                
                P_lm2 = P_lm1
                P_lm1 = P_l
    
    return B