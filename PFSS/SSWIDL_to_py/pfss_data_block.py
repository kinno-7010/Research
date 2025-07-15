"""
====================================================================

pfss_data_block.py - This module contains the PFSS common block variables
                     converted to a Python class for state management.

Original IDL common block variables:
br, bth, bph, nlat, nlon, nr, lat, lon, rix, theta, phi, l0, b0,
now, phiat, phibt, str, stth, stph, ptr, ptth, ptph, nstep, rimage, 
pfss_data, bderivs

Converted to Python - 2025

====================================================================
"""

import numpy as np
from typing import Optional, Any


class PFSSDataBlock:
    """
    Class to manage PFSS common block variables.
    This replaces the IDL common block with a Python class.
    """
    
    def __init__(self):
        """Initialize all PFSS data block variables."""
        
        # Magnetic field components
        self.br: Optional[np.ndarray] = None      # r-component of magnetic field
        self.bth: Optional[np.ndarray] = None     # theta-component of magnetic field
        self.bph: Optional[np.ndarray] = None     # phi-component of magnetic field
        
        # Grid dimensions
        self.nlat: Optional[int] = None           # number of latitude points
        self.nlon: Optional[int] = None           # number of longitude points
        self.nr: Optional[int] = None             # number of radial points
        
        # Coordinate arrays
        self.lat: Optional[np.ndarray] = None     # latitude array
        self.lon: Optional[np.ndarray] = None     # longitude array
        self.rix: Optional[np.ndarray] = None     # radial coordinate array
        self.theta: Optional[np.ndarray] = None   # theta coordinate array
        self.phi: Optional[np.ndarray] = None     # phi coordinate array
        
        # Carrington rotation parameters
        self.l0: Optional[float] = None           # Carrington longitude
        self.b0: Optional[float] = None           # Carrington latitude
        
        # Time information
        self.now: Optional[Any] = None            # current time
        
        # Field line starting coordinates
        self.phiat: Optional[np.ndarray] = None   # phi at start
        self.phibt: Optional[np.ndarray] = None   # phi at end
        
        # Field line starting coordinates (spherical)
        self.str: Optional[np.ndarray] = None     # r starting coordinates
        self.stth: Optional[np.ndarray] = None    # theta starting coordinates
        self.stph: Optional[np.ndarray] = None    # phi starting coordinates
        
        # Field line trace coordinates
        self.ptr: Optional[np.ndarray] = None     # r trace coordinates
        self.ptth: Optional[np.ndarray] = None    # theta trace coordinates
        self.ptph: Optional[np.ndarray] = None    # phi trace coordinates
        
        # Field line properties
        self.nstep: Optional[np.ndarray] = None   # number of steps in each field line
        
        # Image data
        self.rimage: Optional[np.ndarray] = None  # rendered image
        
        # PFSS data structure
        self.pfss_data: Optional[Any] = None      # PFSS data structure
        
        # Derivative information
        self.bderivs: Optional[np.ndarray] = None # field derivatives
    
    def reset(self):
        """Reset all variables to None."""
        self.__init__()
    
    def is_initialized(self) -> bool:
        """Check if basic magnetic field data is initialized."""
        return (self.br is not None and 
                self.bth is not None and 
                self.bph is not None and
                self.nlat is not None and
                self.nlon is not None and
                self.nr is not None)
    
    def set_magnetic_field(self, br: np.ndarray, bth: np.ndarray, bph: np.ndarray):
        """Set the magnetic field components."""
        self.br = np.array(br)
        self.bth = np.array(bth)
        self.bph = np.array(bph)
        
        # Update grid dimensions
        if self.br.ndim >= 3:
            self.nlon, self.nlat, self.nr = self.br.shape
        elif self.br.ndim == 2:
            self.nlon, self.nlat = self.br.shape
            self.nr = 1
    
    def set_coordinates(self, lat: np.ndarray, lon: np.ndarray, rix: np.ndarray):
        """Set the coordinate arrays."""
        self.lat = np.array(lat)
        self.lon = np.array(lon)
        self.rix = np.array(rix)
        
        # Compute theta and phi from lat and lon
        self.theta = np.pi/2 - np.radians(lat)  # colatitude
        self.phi = np.radians(lon)


# Global instance to mimic IDL common block behavior
pfss_data_block = PFSSDataBlock()