#!/usr/bin/env python3
"""
pfss_get_potl_coeffs.py - Computes spherical harmonic coefficients of
                          field potential in (l,m,r)-space from a map of
                          the radial magnetic field located at r=1

Usage: pfss_get_potl_coeffs(mag, rtop=rtop, quiet=quiet)

Parameters:
    mag: input magnetogram
    rtop: if source surface upper BC is desired, set to the radius
          of source surface (assuming input magnetogram is located 
          at r=1), otherwise field spans 1<r<infinity 
    quiet: set keyword to disable screen output

Common block output:
    phiat: (l,m) array of complex coeffs, corresponding to r^l eigenfunction
    phibt: (l,m) array of complex coeffs, corresponding to 1/r^(l+1) eigenfunction

M.DeRosa - 30 Jan 2002 - converted from earlier script
           12 May 2003 - converted common block to PFSS package format
           12 May 2004 - added check for overflow numbers when computing coefficients

Converted to Python by Claude Code
"""

import numpy as np
import sys

# Import other modules from the PFSS package
from .pfss_data_block import PfssDataBlock
from .spherical_transform import spherical_transform


def pfss_get_potl_coeffs(mag, rtop=None, quiet=False):
    """
    Compute spherical harmonic coefficients of field potential
    
    Parameters:
    -----------
    mag : ndarray
        Input magnetogram
    rtop : float, optional
        Radius of source surface (if source surface upper BC is desired)
    quiet : bool, optional
        Set to disable screen output
    
    Returns:
    --------
    dict : Dictionary containing 'phiat' and 'phibt' coefficient arrays
    """
    
    # Print usage message if no parameters
    if mag is None:
        print('  pfss_get_potl_coeffs(mag, rtop=rtop, quiet=quiet)')
        return None
    
    # Get data from common block
    data_block = PfssDataBlock()
    
    # Preliminaries
    cth = np.cos(data_block.theta)
    sth = np.sqrt(1 - cth * cth)
    nlat = len(cth)
    nlon = 2 * nlat
    
    # Check magnetogram dimensions
    nax = mag.shape
    if nlon != nax[0]:
        print('  ERROR in pfss_get_potl_coeffs: nlon is off')
        return None
    
    if nlat != nax[1]:
        print('  ERROR in pfss_get_potl_coeffs: nlat is off')
        return None
    
    # Get spherical harmonic transform of magnetogram
    lmax = nlat
    magt = spherical_transform(mag, cth, lmax=lmax)
    
    # Get l and m index arrays of transform
    lix = np.arange(lmax + 1)
    mix = lix
    larr = lix[:, np.newaxis] * np.ones((1, lmax + 1))
    marr = np.ones((lmax + 1, 1)) * mix[np.newaxis, :]
    
    # Set coefficients to zero where m > l
    mask = marr > larr
    larr = larr.astype(float)
    marr = marr.astype(float)
    larr[mask] = 0
    marr[mask] = 0
    
    # Determine coefficients
    if rtop is not None:
        # Source surface at rtop
        rtop = float(rtop)
        
        # Avoid division by zero and overflow
        denominator = 1 + larr * (1 + rtop**(-2*larr - 1))
        
        # Handle the case where larr = 0 separately
        denominator[larr == 0] = 1.0
        
        phibt = -magt / denominator
        
        # Calculate phiat
        phiat = -phibt / (rtop**(2*larr + 1))
        
        # Handle infinite values
        finite_mask = np.isfinite(phiat)
        phiat[~finite_mask] = 0.0 + 0.0j
        
    else:
        # Potential field extends to infinity, all A's are 0
        phiat = np.zeros_like(magt, dtype=complex)
        
        # Avoid division by zero
        denominator = 1 + larr
        denominator[larr == 0] = 1.0
        
        phibt = -magt / denominator
    
    # Store results in data block
    data_block.phiat = phiat
    data_block.phibt = phibt
    
    if not quiet:
        print('  pfss_get_potl_coeffs: forward transform completed')
    
    return {
        'phiat': phiat,
        'phibt': phibt
    }