#!/usr/bin/env python3
"""
PFSS Model Core
===============

Main PFSS model implementation with magnetic field calculations.

Author: Solar Physics Research System
Date: 2025-01-11
"""

import numpy as np
import sunpy.map
import sunpy.coordinates
from sunpy.net import Fido, attrs as a
from sunpy.time import parse_time
import astropy.units as u
from datetime import timedelta
import scipy.special
from scipy.interpolate import RegularGridInterpolator
import logging
from typing import Tuple, Optional, Dict, Union

from constants import DEFAULT_RSS
from pfss_grid import PFSSGrid
from spherical_harmonics import SphericalHarmonics

# Set up logging
logger = logging.getLogger(__name__)

class PFSSModel:
    """
    Main PFSS model class implementing field calculation and analysis
    """
    
    def __init__(self, grid: Optional[PFSSGrid] = None, 
                 source_surface: float = DEFAULT_RSS):
        """
        Initialize PFSS model
        
        Parameters
        ----------
        grid : PFSSGrid, optional
            Grid specification (uses default if None)
        source_surface : float
            Source surface radius in solar radii
        """
        self.grid = grid or PFSSGrid(r_max=source_surface)
        self.source_surface = source_surface
        self.harmonics = SphericalHarmonics()
        
        # Field component arrays (following SSWIDL BR, BTH, BPH convention)
        self.br = None
        self.btheta = None  
        self.bphi = None
        
        # Harmonic coefficients
        self.coeffs = None
        
        logger.info(f"Initialized PFSS model with Rss={source_surface} R_sun")
    
    def compute_potential(self, coeffs: Dict[Tuple[int, int], complex], 
                         r: np.ndarray, theta: np.ndarray, 
                         phi: np.ndarray) -> np.ndarray:
        """
        Compute magnetic potential from spherical harmonic coefficients
        
        Implements the potential:
        Ψ(r,θ,φ) = Σ_l Σ_m [A_lm r^l + B_lm r^(-l-1)] Y_lm(θ,φ)
        
        With source surface boundary condition:
        B_lm = -A_lm * (R_sun/R_ss)^(2l+1)
        """
        potential = np.zeros_like(r, dtype=complex)
        
        r0 = self.grid.r_min
        rss = self.grid.r_max
        
        for (l, m), alm in coeffs.items():
            # Source surface boundary condition
            blm = -alm * (r0 / rss)**(2*l + 1)
            
            # Radial dependence
            radial = alm * (r/r0)**l + blm * (r0/r)**(l+1)
            
            # Angular dependence
            ylm = self.harmonics.compute_ylm(l, m, theta, phi)
            
            potential += radial * ylm
        
        return np.real(potential)
    
    def calculate_field_components(self, coeffs: Dict[Tuple[int, int], complex]):
        """
        Calculate magnetic field components from harmonic coefficients
        
        B = -∇Ψ in spherical coordinates
        """
        logger.info("Calculating magnetic field components...")
        
        # Initialize field arrays
        shape = (self.grid.n_r, self.grid.n_theta, self.grid.n_phi)
        self.br = np.zeros(shape)
        self.btheta = np.zeros(shape)
        self.bphi = np.zeros(shape)
        
        r0 = self.grid.r_min
        rss = self.grid.r_max
        
        # Compute field components from derivatives
        for (l, m), alm in coeffs.items():
            # Boundary condition coefficient
            blm = -alm * (r0 / rss)**(2*l + 1)
            
            # Spherical harmonic
            ylm = self.harmonics.compute_ylm(l, m, self.grid.THETA, self.grid.PHI)
            
            # Radial component: Br = -∂Ψ/∂r
            dr_radial = l * alm * (self.grid.R/r0)**(l-1) / r0 - \
                       (l+1) * blm * (r0/self.grid.R)**(l+2) / r0
            self.br += np.real(dr_radial * ylm)
            
            # Theta component: Bθ = -(1/r)∂Ψ/∂θ
            if l > 0:  # Avoid division by zero
                radial = alm * (self.grid.R/r0)**l + blm * (r0/self.grid.R)**(l+1)
                
                # Derivative of Plm with respect to theta
                cos_theta = np.cos(self.grid.THETA)
                sin_theta = np.sin(self.grid.THETA)
                
                # Using recurrence relation for derivative
                if abs(m) <= l-1:
                    plm_deriv = self._legendre_derivative(l, m, cos_theta, sin_theta)
                    d_ylm = plm_deriv * np.exp(1j * m * self.grid.PHI)
                    self.btheta -= np.real(radial * d_ylm / self.grid.R)
            
            # Phi component: Bφ = -(1/(r sin θ))∂Ψ/∂φ  
            if m != 0:
                radial = alm * (self.grid.R/r0)**l + blm * (r0/self.grid.R)**(l+1)
                d_ylm_phi = 1j * m * ylm
                sin_theta = np.sin(self.grid.THETA)
                self.bphi -= np.real(radial * d_ylm_phi / (self.grid.R * sin_theta))
        
        self.coeffs = coeffs
        logger.info("Field calculation complete")
    
    def _legendre_derivative(self, l: int, m: int, cos_theta: np.ndarray,
                           sin_theta: np.ndarray) -> np.ndarray:
        """
        Compute derivative of associated Legendre polynomial
        """
        # Use recurrence relation for derivative
        # dP_l^m/dθ = -l*cot(θ)*P_l^m + (l+m)*P_(l-1)^m / sin(θ)
        
        plm = scipy.special.lpmv(m, l, cos_theta)
        
        if l > abs(m):
            plm_prev = scipy.special.lpmv(m, l-1, cos_theta)
            deriv = -l * cos_theta * plm / sin_theta + \
                    (l + m) * plm_prev / sin_theta
        else:
            deriv = -l * cos_theta * plm / sin_theta
            
        return deriv
    
    def load_synoptic_map(self, time_str: str, 
                         instrument: str = 'HMI') -> sunpy.map.Map:
        """
        Load synoptic magnetogram for specified time
        
        Parameters
        ----------
        time_str : str
            Time in format 'YYYY-MM-DDTHH:MM:SS'
        instrument : str
            'HMI' or 'STEREO' 
        
        Returns
        -------
        synoptic_map : sunpy.map.Map
            Synoptic magnetogram
        """
        logger.info(f"Loading {instrument} synoptic map for {time_str}")
        
        # Parse time and find nearest available data
        target_time = parse_time(time_str)
        
        if instrument.upper() == 'HMI':
            # Search for HMI synoptic maps
            result = Fido.search(
                a.Time(target_time - timedelta(days=14), 
                      target_time + timedelta(days=14)),
                a.Instrument('HMI'),
                a.Physobs('los_magnetic_field'),
                a.vso.Sample(24*u.hour)
            )
            
            if len(result) == 0:
                raise ValueError(f"No HMI data found near {time_str}")
            
            # Download closest file
            files = Fido.fetch(result[0, 0])
            synoptic_map = sunpy.map.Map(files[0])
            
        else:
            raise NotImplementedError(f"Instrument {instrument} not yet supported")
        
        # Preprocess map
        synoptic_map = self._preprocess_magnetogram(synoptic_map)
        
        return synoptic_map
    
    def _preprocess_magnetogram(self, mag_map: sunpy.map.Map) -> sunpy.map.Map:
        """
        Preprocess magnetogram: monopole removal and coordinate alignment
        """
        logger.info("Preprocessing magnetogram...")
        
        # Remove monopole term
        data = mag_map.data
        monopole = np.mean(data)
        data_corrected = data - monopole
        
        logger.info(f"Removed monopole term: {monopole:.2f} G")
        
        # Create corrected map
        new_map = sunpy.map.Map(data_corrected, mag_map.meta)
        
        return new_map
    
    def compute_from_magnetogram(self, mag_map: Union[sunpy.map.Map, np.ndarray]):
        """
        Compute PFSS model from magnetogram data
        
        Parameters
        ----------
        mag_map : sunpy.map.Map or ndarray
            Input magnetogram (Br at photosphere)
        """
        if isinstance(mag_map, sunpy.map.Map):
            br_data = mag_map.data
            # Resample to model grid if needed
            if br_data.shape != (self.grid.n_theta, self.grid.n_phi):
                br_data = self._resample_magnetogram(br_data)
        else:
            br_data = mag_map
        
        # Ensure proper dimensions
        assert br_data.shape == (self.grid.n_theta, self.grid.n_phi), \
            f"Magnetogram shape {br_data.shape} doesn't match grid"
        
        # Expand in spherical harmonics
        theta_2d, phi_2d = np.meshgrid(self.grid.theta, self.grid.phi, indexing='ij')
        coeffs = self.harmonics.expand_map(br_data, theta_2d, phi_2d)
        
        # Calculate field components
        self.calculate_field_components(coeffs)
        
        logger.info("PFSS model computation complete")
    
    def _resample_magnetogram(self, data: np.ndarray) -> np.ndarray:
        """Resample magnetogram to model grid resolution"""
        from scipy.ndimage import zoom
        
        zoom_factors = (self.grid.n_theta / data.shape[0],
                       self.grid.n_phi / data.shape[1])
        
        return zoom(data, zoom_factors, order=3)
    
    def get_field_interpolator(self) -> RegularGridInterpolator:
        """
        Get interpolator for magnetic field components
        
        Returns
        -------
        interpolator : RegularGridInterpolator
            Interpolator for (Br, Btheta, Bphi) field components
        """
        if self.br is None:
            raise ValueError("Field not yet computed. Run compute_from_magnetogram first.")
        
        # Stack field components
        field_data = np.stack([self.br, self.btheta, self.bphi], axis=-1)
        
        # Create interpolator
        interpolator = RegularGridInterpolator(
            self.grid.coords, field_data,
            bounds_error=False, fill_value=0.0
        )
        
        return interpolator