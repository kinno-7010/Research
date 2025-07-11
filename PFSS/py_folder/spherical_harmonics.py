#!/usr/bin/env python3
"""
Spherical Harmonics Expansion
=============================

Spherical harmonic expansion for PFSS model calculations.

Author: Solar Physics Research System
Date: 2025-01-11
"""

import numpy as np
import scipy.special
import logging
from typing import Dict, Tuple
from numba import jit
from constants import DEFAULT_L_MAX

# Set up logging
logger = logging.getLogger(__name__)

class SphericalHarmonics:
    """
    Spherical harmonic expansion following SSWIDL normalization conventions
    """
    
    def __init__(self, l_max: int = DEFAULT_L_MAX):
        """
        Initialize spherical harmonic calculator
        
        Parameters
        ----------
        l_max : int
            Maximum harmonic degree (typically 60-120)
        """
        self.l_max = l_max
        self._plm_cache = {}
        logger.info(f"Initialized spherical harmonics with l_max={l_max}")
    
    @staticmethod
    @jit(nopython=True)
    def _associated_legendre_normalized(l: int, m: int, x: np.ndarray) -> np.ndarray:
        """
        Compute normalized associated Legendre polynomials
        Using SSWIDL normalization: sqrt((2-delta_0m)/2)
        """
        # Use scipy's sph_harm normalization and adjust
        norm_factor = np.sqrt(2.0) if m == 0 else 1.0
        return norm_factor * scipy.special.lpmv(m, l, x)
    
    def compute_ylm(self, l: int, m: int, theta: np.ndarray, 
                    phi: np.ndarray) -> np.ndarray:
        """
        Compute spherical harmonic Y_lm with proper normalization
        """
        cos_theta = np.cos(theta)
        
        # Get cached or compute Legendre polynomial
        cache_key = (l, m, cos_theta.shape)
        if cache_key not in self._plm_cache:
            self._plm_cache[cache_key] = self._associated_legendre_normalized(
                l, m, cos_theta)
        
        plm = self._plm_cache[cache_key]
        
        # Spherical harmonic
        if m >= 0:
            ylm = plm * np.exp(1j * m * phi)
        else:
            # Y_l,-m = (-1)^m * conj(Y_l,m)
            ylm = (-1)**(-m) * np.conj(plm * np.exp(1j * (-m) * phi))
        
        return ylm
    
    def expand_map(self, br_map: np.ndarray, theta: np.ndarray, 
                   phi: np.ndarray) -> Dict[Tuple[int, int], complex]:
        """
        Expand photospheric Br map into spherical harmonic coefficients
        
        Parameters
        ----------
        br_map : ndarray
            Radial magnetic field map at photosphere
        theta, phi : ndarray
            Coordinate arrays
        
        Returns
        -------
        coeffs : dict
            Spherical harmonic coefficients {(l,m): coefficient}
        """
        coeffs = {}
        
        # Integration weights for uniform sampling in sin(theta)
        d_sin_theta = 2.0 / len(theta)
        d_phi = 2.0 * np.pi / len(phi[0])
        
        logger.info("Computing spherical harmonic expansion...")
        
        for l in range(self.l_max + 1):
            for m in range(-l, l + 1):
                # Compute Y_lm
                ylm = self.compute_ylm(l, m, theta, phi)
                
                # Integration over sphere
                integrand = br_map * np.conj(ylm) * np.sin(theta)
                coeff = np.sum(integrand) * d_sin_theta * d_phi
                
                coeffs[(l, m)] = coeff
                
            if l % 10 == 0:
                logger.debug(f"Computed harmonics up to l={l}")
        
        return coeffs