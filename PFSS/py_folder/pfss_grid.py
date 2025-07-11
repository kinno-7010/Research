#!/usr/bin/env python3
"""
PFSS Grid Configuration
=======================

Grid management for PFSS model calculations using spherical coordinates.

Author: Solar Physics Research System
Date: 2025-01-11
"""

import numpy as np
import logging
from dataclasses import dataclass
from constants import DEFAULT_N_PHI, DEFAULT_N_THETA, DEFAULT_N_R, DEFAULT_RSS

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class PFSSGrid:
    """Data structure for PFSS grid following SSWIDL conventions"""
    n_phi: int = DEFAULT_N_PHI      # Longitude points
    n_theta: int = DEFAULT_N_THETA  # Latitude points  
    n_r: int = DEFAULT_N_R          # Radial points
    r_min: float = 1.0              # Minimum radius (R_sun)
    r_max: float = DEFAULT_RSS      # Maximum radius (source surface)
    
    def __post_init__(self):
        """Initialize coordinate arrays with logarithmic radial spacing"""
        # Phi: uniform azimuthal spacing
        self.phi = np.linspace(0, 2*np.pi, self.n_phi, endpoint=False)
        
        # Theta: cosine-latitude distribution for uniform area elements
        # Using sine latitude for numerical stability
        sin_theta = np.linspace(-1, 1, self.n_theta)
        self.theta = np.arcsin(sin_theta) + np.pi/2
        
        # R: logarithmic spacing as in SSWIDL dumfric coordinates
        log_r = np.linspace(np.log(self.r_min), np.log(self.r_max), self.n_r)
        self.r = np.exp(log_r)
        
        # Create 3D coordinate grids
        self.R, self.THETA, self.PHI = np.meshgrid(self.r, self.theta, self.phi, 
                                                   indexing='ij')
        
        # Coordinate arrays for interpolation
        self.coords = (self.r, self.theta, self.phi)
        
        logger.info(f"Initialized PFSS grid: {self.n_phi}×{self.n_theta}×{self.n_r}")