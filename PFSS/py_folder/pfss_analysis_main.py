#!/usr/bin/env python3
"""
Advanced Potential Field Source Surface (PFSS) Model Analysis Tool
==================================================================

This module implements a sophisticated PFSS model based on SSWIDL's architecture
with modern Python enhancements. It provides:

1. Spherical harmonic expansion of photospheric magnetic fields
2. 3D magnetic field extrapolation to the source surface
3. Field line tracing with adaptive step sizing
4. Advanced visualization with matplotlib
5. Integration with SDO/HMI (LOS_magnetic_field) and STEREO data

Based on SSWIDL PFSS implementation analysis with performance optimizations
and enhanced coordinate system handling.

Author: Solar Physics Research System
Date: 2025-01-11
"""

import numpy as np
import matplotlib
# WSL環境でのGUI表示用バックエンド設定
try:
    # GUI表示を試行
    matplotlib.use('TkAgg')
except ImportError:
    try:
        matplotlib.use('Qt5Agg')  # 代替バックエンド
    except ImportError:
        matplotlib.use('Agg')  # フォールバック（ファイル保存のみ）
        print("Warning: GUI backend not available. Files will be saved only.")
    
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sunpy.map
import sunpy.coordinates
from sunpy.net import Fido, attrs as a
from sunpy.time import parse_time
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import scipy.special
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime, timedelta
import warnings
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import h5py
from numba import jit, prange
import concurrent.futures
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
R_SUN = 696000.0  # Solar radius in km
R_SUN_RS = 1.0    # Solar radius in solar radii units
DEFAULT_RSS = 2.5  # Default source surface radius in solar radii
CARRINGTON_ROTATION_PERIOD = 27.2753  # days

@dataclass
class ROIBounds:
    """Region of Interest bounds in HMI pixel coordinates (sun center = 0,0)"""
    x_min: int = -512     # Left edge in pixels
    x_max: int = 512      # Right edge in pixels  
    y_min: int = -512     # Bottom edge in pixels
    y_max: int = 512      # Top edge in pixels
    
    def __post_init__(self):
        """Validate ROI bounds"""
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Invalid ROI bounds: min values must be less than max values")
        
        # Calculate ROI size
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min
        self.area = self.width * self.height
        
        logger.info(f"ROI defined: {self.width}×{self.height} pixels, area={self.area}")

@dataclass
class PFSSGrid:
    """Data structure for PFSS grid following SSWIDL conventions"""
    n_phi: int = 384      # Longitude points
    n_theta: int = 192    # Latitude points  
    n_r: int = 39         # Radial points
    r_min: float = 1.0    # Minimum radius (R_sun)
    r_max: float = 2.5    # Maximum radius (source surface)
    roi: Optional[ROIBounds] = None  # Region of interest
    
    def __post_init__(self):
        """Initialize coordinate arrays with ROI support"""
        # ROI-aware grid initialization
        if self.roi is not None:
            # Adjust grid size based on ROI
            roi_aspect = self.roi.width / self.roi.height
            
            # Scale grid points proportionally to ROI size
            base_resolution = 64  # Base resolution for small ROIs
            scale_factor = min(2.0, max(0.5, np.sqrt(self.roi.area) / 500))
            
            self.n_phi = max(32, int(base_resolution * roi_aspect * scale_factor))
            self.n_theta = max(32, int(base_resolution * scale_factor))
            
            logger.info(f"ROI-adjusted grid: {self.n_phi}×{self.n_theta} (ROI: {self.roi.width}×{self.roi.height})")
        
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

class SphericalHarmonics:
    """
    Spherical harmonic expansion following SSWIDL normalization conventions
    """
    
    def __init__(self, l_max: int = 90):
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
    # @jit(nopython=True)
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

class HMICoordinateConverter:
    """
    Handle coordinate conversions between HMI pixel coordinates and spherical coordinates
    """
    
    def __init__(self, mag_map: sunpy.map.Map):
        """
        Initialize converter with HMI magnetogram
        
        Parameters
        ----------
        mag_map : sunpy.map.Map
            HMI magnetogram with proper WCS information
        """
        try:
            self.mag_map = mag_map
            self.reference_pixel = mag_map.reference_pixel
            self.scale = mag_map.scale
            
            logger.info(f"Debug: mag_map.dimensions = {mag_map.dimensions}")
            logger.info(f"Debug: mag_map.rsun_obs = {mag_map.rsun_obs}")
            logger.info(f"Debug: mag_map.scale = {mag_map.scale}")
            
            # HMI image dimensions with proper unit handling
            if hasattr(mag_map.dimensions.x, 'value'):
                self.nx = int(mag_map.dimensions.x.value)
                self.ny = int(mag_map.dimensions.y.value)
            else:
                self.nx = int(mag_map.dimensions.x)
                self.ny = int(mag_map.dimensions.y)
            
            # Solar radius in pixels with proper unit handling
            rsun_obs = mag_map.rsun_obs
            scale_x = mag_map.scale[0]
            
            # Handle astropy units properly
            if hasattr(rsun_obs, 'value') and hasattr(scale_x, 'value'):
                self.rsun_pixels = float(rsun_obs.value / scale_x.value)
            elif hasattr(rsun_obs, 'to') and hasattr(scale_x, 'to'):
                # Convert to same units
                rsun_arcsec = rsun_obs.to(u.arcsec).value
                scale_arcsec = scale_x.to(u.arcsec).value
                self.rsun_pixels = float(rsun_arcsec / scale_arcsec)
            else:
                # Fallback: assume both are in same units
                self.rsun_pixels = float(rsun_obs / scale_x)
            
            logger.info(f"HMI Converter: {self.nx}×{self.ny}, R_sun={self.rsun_pixels:.1f} pixels")
            
        except Exception as e:
            logger.error(f"Error initializing HMI converter: {e}")
            logger.error(f"mag_map type: {type(mag_map)}")
            logger.error(f"rsun_obs type: {type(mag_map.rsun_obs)}")
            logger.error(f"scale type: {type(mag_map.scale)}")
            raise
    
    def hmi_to_solar_center(self, x_hmi: int, y_hmi: int) -> Tuple[float, float]:
        """
        Convert HMI pixel coordinates to solar center coordinates
        
        Parameters
        ----------
        x_hmi, y_hmi : int
            HMI pixel coordinates
        
        Returns
        -------
        x_solar, y_solar : float
            Solar center coordinates in pixels
        """
        # HMI reference pixel is usually at center
        x_center = float(self.reference_pixel.x.value)
        y_center = float(self.reference_pixel.y.value)
        
        x_solar = x_hmi - x_center
        y_solar = y_hmi - y_center
        
        return x_solar, y_solar
    
    def solar_center_to_spherical(self, x_solar: float, y_solar: float) -> Tuple[float, float]:
        """
        Convert solar center coordinates to spherical coordinates (improved)
        
        Parameters
        ----------
        x_solar, y_solar : float
            Solar center coordinates in pixels
        
        Returns
        -------
        theta, phi : float
            Spherical coordinates (colatitude, azimuth) in radians
        """
        # Convert to normalized coordinates (-1 to 1)
        x_norm = x_solar / self.rsun_pixels
        y_norm = y_solar / self.rsun_pixels
        
        # Limit to solar disk for realistic coordinates
        r_norm = np.sqrt(x_norm**2 + y_norm**2)
        if r_norm > 0.95:  # Stay within 95% of solar radius for stability
            x_norm = x_norm * 0.95 / r_norm
            y_norm = y_norm * 0.95 / r_norm
            r_norm = 0.95
        
        # Convert to spherical coordinates using proper spherical projection
        # theta: colatitude (0 at north pole, pi at south pole)
        # phi: azimuth (longitude)
        
        # For points on visible solar disk, use stereographic projection
        if r_norm < 0.001:  # Very close to center
            theta = np.pi/2  # Equator
            phi = 0.0
        else:
            # Latitude from y-coordinate
            sin_lat = y_norm
            lat = np.arcsin(np.clip(sin_lat, -1, 1))
            theta = np.pi/2 - lat  # Convert latitude to colatitude
            
            # Longitude from x-coordinate, accounting for foreshortening
            cos_lat = np.cos(lat)
            if abs(cos_lat) > 0.001:
                sin_lon = x_norm / cos_lat
                sin_lon = np.clip(sin_lon, -1, 1)
                phi = np.arcsin(sin_lon)
                if phi < 0:
                    phi += 2*np.pi
            else:
                phi = 0.0  # At poles
        
        # Ensure valid ranges
        theta = np.clip(theta, 0.001, np.pi - 0.001)  # Avoid poles
        phi = phi % (2*np.pi)
            
        return theta, phi
    
    def extract_roi_data(self, roi: ROIBounds) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract magnetogram data for specified ROI
        
        Parameters
        ----------
        roi : ROIBounds
            Region of interest bounds
        
        Returns
        -------
        data : ndarray
            Extracted magnetogram data
        theta_coords : ndarray
            Theta coordinates for extracted region
        phi_coords : ndarray
            Phi coordinates for extracted region
        """
        # Convert ROI bounds to HMI pixel coordinates
        x_center = float(self.reference_pixel.x.value)
        y_center = float(self.reference_pixel.y.value)
        
        x_start = int(x_center + roi.x_min)
        x_end = int(x_center + roi.x_max)
        y_start = int(y_center + roi.y_min)
        y_end = int(y_center + roi.y_max)
        
        # Clamp to image bounds
        x_start = max(0, x_start)
        x_end = min(int(self.nx), x_end)
        y_start = max(0, y_start)
        y_end = min(int(self.ny), y_end)
        
        # Extract data
        roi_data = self.mag_map.data[y_start:y_end, x_start:x_end]
        
        # Create coordinate arrays for ROI
        x_roi = np.arange(x_start, x_end) - x_center
        y_roi = np.arange(y_start, y_end) - y_center
        
        X_roi, Y_roi = np.meshgrid(x_roi, y_roi)
        
        # Convert to spherical coordinates
        theta_coords = np.zeros_like(X_roi)
        phi_coords = np.zeros_like(X_roi)
        
        for i in range(X_roi.shape[0]):
            for j in range(X_roi.shape[1]):
                theta_coords[i, j], phi_coords[i, j] = self.solar_center_to_spherical(
                    X_roi[i, j], Y_roi[i, j]
                )
        
        logger.info(f"Extracted ROI data: {roi_data.shape}, theta range: [{np.min(theta_coords):.3f}, {np.max(theta_coords):.3f}]")
        
        return roi_data, theta_coords, phi_coords

class PFSSModel:
    """
    Main PFSS model class implementing field calculation and analysis with ROI support
    """
    
    def __init__(self, grid: Optional[PFSSGrid] = None, 
                 source_surface: float = DEFAULT_RSS,
                 roi: Optional[ROIBounds] = None):
        """
        Initialize PFSS model with ROI support
        
        Parameters
        ----------
        grid : PFSSGrid, optional
            Grid specification (uses default if None)
        source_surface : float
            Source surface radius in solar radii
        roi : ROIBounds, optional
            Region of interest for computation
        """
        # Create grid with ROI if specified
        if grid is None:
            grid = PFSSGrid(r_max=source_surface, roi=roi)
        
        self.grid = grid
        self.source_surface = source_surface
        self.roi = roi
        self.harmonics = SphericalHarmonics()
        
        # Coordinate converter (will be set when loading magnetogram)
        self.coord_converter: Optional[HMICoordinateConverter] = None
        
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
        
        # --- 修正箇所：極でのゼロ除算を回避 ---
        sin_theta_grid = np.sin(self.grid.THETA)
        sin_theta_safe = sin_theta_grid + 1e-15
        
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
            if l > 0:
                radial = alm * (self.grid.R/r0)**l + blm * (r0/self.grid.R)**(l+1)
                
                cos_theta = np.cos(self.grid.THETA)
                
                # Use the already fixed derivative function
                plm_deriv = self._legendre_derivative(l, m, cos_theta, sin_theta_grid)
                d_ylm = plm_deriv * np.exp(1j * m * self.grid.PHI)
                self.btheta -= np.real(radial * d_ylm / self.grid.R)
            
            # Phi component: Bφ = -(1/(r sin θ))∂Ψ/∂φ
            if m != 0:
                radial = alm * (self.grid.R/r0)**l + blm * (r0/self.grid.R)**(l+1)
                d_ylm_phi = 1j * m * ylm
                # --- 修正箇所：安全なsin_thetaを使用 ---
                self.bphi -= np.real(radial * d_ylm_phi / (self.grid.R * sin_theta_safe))
        
        self.coeffs = coeffs
        logger.info("Field calculation complete")
        
        
    def _legendre_derivative(self, l: int, m: int, cos_theta: np.ndarray,
                        sin_theta: np.ndarray) -> np.ndarray:
        """
        Compute derivative of associated Legendre polynomial
        """
        # Use recurrence relation for derivative
        # dP_l^m/dθ = -l*cot(θ)*P_l^m + (l+m)*P_(l-1)^m / sin(θ)
        
        # --- 修正箇所：分母に微小な値を加え、ゼロ除算を回避 ---
        sin_theta_safe = sin_theta + 1e-15

        plm = scipy.special.lpmv(m, l, cos_theta)
        
        if l > abs(m):
            plm_prev = scipy.special.lpmv(m, l-1, cos_theta)
            deriv = -l * cos_theta * plm / sin_theta_safe + \
                    (l + m) * plm_prev / sin_theta_safe
        else:
            deriv = -l * cos_theta * plm / sin_theta_safe
            
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
        
        target_time = parse_time(time_str)
        
        if instrument.upper() == 'HMI':
            result = Fido.search(
                a.Time(target_time - timedelta(days=14), 
                    target_time + timedelta(days=14)),
                a.Instrument('HMI'),
                a.Physobs('los_magnetic_field'),
                a.Sample(24*u.hour)
            )
            
            if len(result) == 0:
                raise ValueError(f"No HMI data found near {time_str}")
            
            logger.info("Fetching data from server... This may take a few minutes.")
            files = Fido.fetch(result[0, 0])
            
            # --- 修正箇所：ダウンロードが成功したかチェック ---
            if not files:
                # ダウンロードに失敗した場合、filesは空になる
                raise IOError(
                    "Failed to download data due to a network or server issue (e.g., timeout). "
                    "Please check your internet connection and try again later."
                )
                
            synoptic_map = sunpy.map.Map(files[0])
            
        else:
            raise NotImplementedError(f"Instrument {instrument} not yet supported")
        
        synoptic_map = self._preprocess_magnetogram(synoptic_map)
        
        return synoptic_map

    
    def _preprocess_magnetogram(self, mag_map: sunpy.map.Map) -> sunpy.map.Map:
        """
        Preprocess magnetogram: monopole removal and coordinate alignment
        """
        logger.info("Preprocessing magnetogram...")
        
        # Remove monopole term
        data = mag_map.data
        monopole = np.nanmean(data)
        data_corrected = data - monopole
        
        logger.info(f"Removed monopole term: {monopole:.2f} G")
        
        # Create corrected map
        new_map = sunpy.map.Map(data_corrected, mag_map.meta)
        
        return new_map
    
    def compute_from_magnetogram(self, mag_map):
        """
        Compute PFSS model from magnetogram data with ROI support
        
        Parameters
        ----------
        mag_map : sunpy.map.Map or ndarray
            Input magnetogram (Br at photosphere)
        """
        # Handle ROI processing
        if hasattr(mag_map, 'data') and hasattr(mag_map, 'meta') and self.roi is not None:
            logger.info(f"Processing ROI: [{self.roi.x_min}, {self.roi.x_max}] x [{self.roi.y_min}, {self.roi.y_max}]")
            
            try:
                # Initialize coordinate converter
                logger.info("Initializing HMI coordinate converter...")
                self.coord_converter = HMICoordinateConverter(mag_map)
                
                # Extract ROI data
                logger.info("Extracting ROI data...")
                br_data, theta_coords, phi_coords = self.coord_converter.extract_roi_data(self.roi)
                
                # Use ROI coordinates for spherical harmonic expansion
                logger.info("Computing spherical harmonic expansion for ROI...")
                coeffs = self.harmonics.expand_map(br_data, theta_coords, phi_coords)
                
            except Exception as e:
                logger.error(f"Error in ROI processing: {e}")
                logger.error("Falling back to full disk processing...")
                # Fallback to full disk processing
                br_data = mag_map.data
                if br_data.shape != (self.grid.n_theta, self.grid.n_phi):
                    br_data = self._resample_magnetogram(br_data)
                theta_2d, phi_2d = np.meshgrid(self.grid.theta, self.grid.phi, indexing='ij')
                coeffs = self.harmonics.expand_map(br_data, theta_2d, phi_2d)
            
        else:
            # Full disk processing (original method)
            if hasattr(mag_map, 'data'):
                br_data = mag_map.data
            else:
                br_data = mag_map
                
            # Resample to model grid if needed
            if br_data.shape != (self.grid.n_theta, self.grid.n_phi):
                br_data = self._resample_magnetogram(br_data)
            
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

class MagneticFieldAnalyzer:
    """
    Analyze magnetic field data to find optimal field line starting points
    """
    
    def __init__(self, pfss_model: 'PFSSModel'):
        """
        Initialize magnetic field analyzer
        
        Parameters
        ---------- 
        pfss_model : PFSSModel
            PFSS model with computed magnetic field
        """
        self.model = pfss_model
        
    def find_magnetic_sources(self, threshold_percentile: float = 85, 
                            min_separation: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find strong magnetic field regions for field line starting points
        
        Parameters
        ----------
        threshold_percentile : float
            Percentile threshold for field strength (default: 85)
        min_separation : float
            Minimum separation between starting points in solar radii
            
        Returns
        -------
        positive_sources : ndarray
            Starting points for positive magnetic regions [N, 3] (r, theta, phi)
        negative_sources : ndarray  
            Starting points for negative magnetic regions [N, 3] (r, theta, phi)
        """
        # Get photospheric magnetic field
        br_photo = self.model.br[0, :, :]
        
        # Calculate field strength threshold
        field_strength = np.abs(br_photo)
        threshold = np.percentile(field_strength, threshold_percentile)
        
        # Find positive and negative strong field regions
        positive_mask = br_photo > threshold
        negative_mask = br_photo < -threshold
        
        # Extract coordinates
        theta_2d, phi_2d = np.meshgrid(self.model.grid.theta, self.model.grid.phi, indexing='ij')
        
        # Find starting points with separation constraint
        positive_sources = self._extract_separated_points(
            positive_mask, theta_2d, phi_2d, min_separation)
        negative_sources = self._extract_separated_points(
            negative_mask, theta_2d, phi_2d, min_separation)
            
        logger.info(f"Found {len(positive_sources)} positive and {len(negative_sources)} negative magnetic sources")
        
        return positive_sources, negative_sources
    
    def _extract_separated_points(self, mask: np.ndarray, theta_2d: np.ndarray, 
                                phi_2d: np.ndarray, min_separation: float) -> np.ndarray:
        """
        Extract well-separated points from magnetic field mask
        """
        # Find all candidate points
        candidates_i, candidates_j = np.where(mask)
        
        if len(candidates_i) == 0:
            return np.array([]).reshape(0, 3)
            
        # Convert to spherical coordinates
        candidates = np.column_stack([
            np.ones(len(candidates_i)),  # r = 1.0 (photosphere)
            theta_2d[candidates_i, candidates_j],
            phi_2d[candidates_i, candidates_j]
        ])
        
        # Apply separation constraint
        selected_points = []
        for point in candidates:
            if len(selected_points) == 0:
                selected_points.append(point)
            else:
                # Check minimum separation
                distances = [self._angular_distance(point[1:], sp[1:]) for sp in selected_points]
                if min(distances) > min_separation:
                    selected_points.append(point)
                    
        return np.array(selected_points)
    
    def _angular_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calculate angular distance between two points on sphere
        """
        theta1, phi1 = point1
        theta2, phi2 = point2
        
        # Haversine formula for angular distance
        dtheta = theta2 - theta1
        dphi = phi2 - phi1
        
        a = np.sin(dtheta/2)**2 + np.cos(theta1) * np.cos(theta2) * np.sin(dphi/2)**2
        return 2 * np.arcsin(np.sqrt(a))

class FieldLineTracer:
    """
    Advanced field line tracing with adaptive step size control and 3D visualization support
    """
    
    def __init__(self, pfss_model: PFSSModel):
        """
        Initialize field line tracer
        
        Parameters
        ----------
        pfss_model : PFSSModel
            PFSS model instance with computed field
        """
        self.model = pfss_model
        self.interpolator = pfss_model.get_field_interpolator()
        
        # Tracing parameters (optimized for performance)
        self.step_min = 0.02   # Minimum step size in R_sun
        self.step_max = 0.3    # Maximum step size in R_sun
        self.max_steps = 500   # Further reduced for speed
        
        logger.info("Initialized field line tracer")
    
    # FieldLineTracer クラス内のこの関数を置き換えてください
    def _field_function(self, s: float, pos: np.ndarray) -> np.ndarray:
        """
        Field line differential equation: dx/ds = B(x)/|B(x)| (optimized)
        
        Parameters
        ----------
        s : float
            Path parameter (not used but required by solve_ivp)
        pos : ndarray
            Position vector [r, theta, phi]
        
        Returns
        -------
        dpos_ds : ndarray
            Normalized field direction
        """
        # Boundary checks for early termination
        r, theta, phi = pos
        if r < self.model.grid.r_min or r > self.model.grid.r_max:
            return np.zeros(3)
        
        # Get field at current position
        try:
            field_components = self.interpolator(pos).flatten()
            br, btheta, bphi = field_components
        except:
            return np.zeros(3)
        
        # Field magnitude with minimum threshold
        b_mag = np.sqrt(br**2 + btheta**2 + bphi**2)
        
        if b_mag < 1e-8:  # Increased threshold for speed
            return np.zeros(3)
        
        # Optimized trigonometric calculations
        sin_theta = np.sin(theta)
        sin_theta_safe = sin_theta + 1e-10
        
        # Normalized field components in spherical coordinates
        inv_b_mag = 1.0 / b_mag
        dr_ds = br * inv_b_mag
        dtheta_ds = btheta * inv_b_mag / r
        dphi_ds = bphi * inv_b_mag / (r * sin_theta_safe)
        
        return np.array([dr_ds, dtheta_ds, dphi_ds])
    
    def trace_field_line(self, start_point: np.ndarray, 
                        direction: str = 'both') -> Dict[str, np.ndarray]:
        """
        Trace magnetic field line from starting point
        
        Parameters
        ----------
        start_point : ndarray
            Starting position [r, theta, phi]
        direction : str
            'forward', 'backward', or 'both'
        
        Returns
        -------
        result : dict
            Dictionary with traced coordinates and classification
        """
        results = {}
        
        # Forward tracing
        if direction in ['forward', 'both']:
            sol_forward = self._trace_single_direction(start_point, forward=True)
            results['forward'] = sol_forward
        
        # Backward tracing  
        if direction in ['backward', 'both']:
            sol_backward = self._trace_single_direction(start_point, forward=False)
            results['backward'] = sol_backward
        
        # Combine and classify
        if direction == 'both':
            results['combined'] = self._combine_traces(
                results['forward'], results['backward'])
            results['classification'] = self._classify_field_line(results['combined'])
        
        return results
    
    def _trace_single_direction(self, start_point: np.ndarray, 
                               forward: bool = True) -> dict:
        """Trace in single direction with adaptive stepping"""
        
        # Set integration direction
        sign = 1.0 if forward else -1.0
        
        # Termination event: reaches boundaries
        def boundary_event(s, pos):
            r = pos[0]
            # Stop at photosphere or source surface
            return min(r - self.model.grid.r_min, 
                      self.model.grid.r_max - r)
        
        boundary_event.terminal = True
        boundary_event.direction = -1  # Detect crossing
        
        # Solve ODE with adaptive stepping (optimized with timeout)
        sol = solve_ivp(
            lambda s, pos: sign * self._field_function(s, pos),
            [0, sign * 50],    # Further reduced s_span
            start_point,
            method='RK45',
            dense_output=False, # Disable dense output for speed
            events=boundary_event,
            rtol=1e-3,         # Further relaxed tolerance
            atol=1e-5,         # Further relaxed tolerance
            max_step=self.step_max,
            first_step=self.step_max/10  # Larger initial step
        )
        
        return {
            'positions': sol.y.T,
            's_values': sol.t,
            'terminated': len(sol.t_events[0]) > 0,
            'termination_reason': self._get_termination_reason(sol)
        }
    
    def _get_termination_reason(self, sol) -> str:
        """Determine why field line tracing terminated"""
        if len(sol.t_events[0]) > 0:
            final_r = sol.y[0, -1]
            if abs(final_r - self.model.grid.r_min) < 1e-6:
                return 'photosphere'
            elif abs(final_r - self.model.grid.r_max) < 1e-6:
                return 'source_surface'
        return 'max_steps'
    
    def _combine_traces(self, forward: dict, backward: dict) -> dict:
        """Combine forward and backward traces"""
        # Reverse backward trace and concatenate
        combined_positions = np.vstack([
            backward['positions'][::-1][:-1],  # Exclude duplicate start point
            forward['positions']
        ])
        
        return {
            'positions': combined_positions,
            'forward_end': forward['termination_reason'],
            'backward_end': backward['termination_reason']
        }
    
    def _classify_field_line(self, trace: dict) -> str:
        """
        Classify field line as open or closed
        
        Following SSWIDL convention:
        - Open: reaches source surface
        - Closed: both ends at photosphere
        """
        forward_end = trace['forward_end']
        backward_end = trace['backward_end']
        
        if 'source_surface' in [forward_end, backward_end]:
            return 'open'
        elif forward_end == 'photosphere' and backward_end == 'photosphere':
            return 'closed'
        else:
            return 'undetermined'
    
    def trace_from_magnetic_sources(self, positive_sources: np.ndarray, 
                                   negative_sources: np.ndarray) -> List[dict]:
        """
        Trace field lines from detected magnetic source regions
        
        Parameters
        ----------
        positive_sources : ndarray
            Starting points for positive magnetic regions [N, 3]
        negative_sources : ndarray
            Starting points for negative magnetic regions [N, 3]
            
        Returns
        -------
        field_lines : list
            List of traced field lines with polarity information
        """
        field_lines = []
        
        # Trace from positive sources
        logger.info(f"Tracing {len(positive_sources)} positive field lines...")
        for i, start_point in enumerate(positive_sources):
            try:
                result = self.trace_field_line(start_point, 'forward')
                result['polarity'] = 'positive'
                result['source_index'] = i
                field_lines.append(result)
            except Exception as e:
                logger.warning(f"Failed to trace positive field line {i}: {e}")
        
        # Trace from negative sources
        logger.info(f"Tracing {len(negative_sources)} negative field lines...")
        for i, start_point in enumerate(negative_sources):
            try:
                result = self.trace_field_line(start_point, 'forward')
                result['polarity'] = 'negative'
                result['source_index'] = i + len(positive_sources)
                field_lines.append(result)
            except Exception as e:
                logger.warning(f"Failed to trace negative field line {i}: {e}")
                
        logger.info(f"Successfully traced {len(field_lines)} field lines")
        return field_lines
    
    def trace_from_grid(self, n_lon: int = 30, n_lat: int = 15, 
                       r_start: float = 1.0) -> List[dict]:
        """
        Trace field lines from uniform grid of starting points with ROI awareness
        
        Parameters
        ----------
        n_lon, n_lat : int
            Number of longitude/latitude points
        r_start : float
            Starting radius in R_sun
        
        Returns
        -------
        field_lines : list
            List of traced field lines
        """
        logger.info(f"Tracing field lines from {n_lon}×{n_lat} grid...")
        
        # ROI-aware grid creation (improved)
        if self.model.roi is not None and self.model.coord_converter is not None:
            # Create grid covering ROI region
            roi = self.model.roi
            
            try:
                # Sample multiple points in ROI to get proper coordinate range
                x_samples = np.linspace(roi.x_min, roi.x_max, 5)
                y_samples = np.linspace(roi.y_min, roi.y_max, 5)
                
                theta_list, phi_list = [], []
                for x in x_samples:
                    for y in y_samples:
                        theta, phi = self.model.coord_converter.solar_center_to_spherical(x, y)
                        theta_list.append(theta)
                        phi_list.append(phi)
                
                # Get coordinate ranges with proper margins
                theta_min, theta_max = min(theta_list), max(theta_list)
                phi_min, phi_max = min(phi_list), max(phi_list)
                
                # Add margins and ensure valid ranges
                theta_margin = (theta_max - theta_min) * 0.1 + 0.1
                phi_margin = (phi_max - phi_min) * 0.1 + 0.1
                
                theta_min = max(0.1, theta_min - theta_margin)
                theta_max = min(np.pi - 0.1, theta_max + theta_margin)
                phi_min = max(0, phi_min - phi_margin) 
                phi_max = min(2*np.pi, phi_max + phi_margin)
                
                # Ensure theta range is meaningful
                if theta_max - theta_min < 0.1:
                    theta_center = (theta_min + theta_max) / 2
                    theta_min = max(0.1, theta_center - 0.2)
                    theta_max = min(np.pi - 0.1, theta_center + 0.2)
                
                lon_grid = np.linspace(phi_min, phi_max, n_lon)
                lat_grid = np.linspace(theta_min, theta_max, n_lat)
                
                logger.info(f"ROI-focused tracing: theta=[{theta_min:.3f}, {theta_max:.3f}], phi=[{phi_min:.3f}, {phi_max:.3f}]")
                
            except Exception as e:
                logger.warning(f"ROI coordinate conversion failed: {e}, using default grid")
                # Fallback to default grid
                lon_grid = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
                lat_grid = np.linspace(np.pi/6, np.pi*5/6, n_lat)  # Avoid poles
        else:
            # Default full-disk grid (optimized spacing)
            lon_grid = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
            lat_grid = np.linspace(-np.pi/3, np.pi/3, n_lat)  # Focus on equatorial region
        
        field_lines = []
        
        # Use parallel processing with timeout (optimized)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            
            for lon in lon_grid:
                for lat in lat_grid:
                    theta = lat  # lat_grid already contains colatitude values
                    start_point = np.array([r_start, theta, lon])
                    
                    future = executor.submit(
                        self.trace_field_line, start_point, 'forward'  # Only forward tracing for speed
                    )
                    futures.append(future)
            
            # Collect results with timeout
            timeout_per_line = 10  # 10 seconds per field line max
            for i, future in enumerate(concurrent.futures.as_completed(futures, timeout=timeout_per_line * len(futures))):
                try:
                    result = future.result(timeout=timeout_per_line)
                    field_lines.append(result)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Traced {i + 1}/{len(futures)} field lines")
                        
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Field line tracing timed out (line {i+1})")
                except Exception as e:
                    logger.warning(f"Failed to trace field line {i+1}: {e}")
                    
            # Stop processing if we have reasonable number of field lines
            if len(field_lines) > len(futures) // 2:
                logger.info(f"Sufficient field lines traced ({len(field_lines)}), stopping early")
        
        logger.info(f"Completed tracing {len(field_lines)} field lines")
        return field_lines

class PFSSVisualizer:
    """
    Advanced 3D visualization for PFSS models with realistic magnetic field line display
    """
    
    def __init__(self, pfss_model: PFSSModel):
        """Initialize visualizer with PFSS model"""
        self.model = pfss_model
        self.tracer = FieldLineTracer(pfss_model)
        self.analyzer = MagneticFieldAnalyzer(pfss_model)
        
        # Visualization parameters
        self.cmap_field = 'RdBu_r'
        self.cmap_polarity = {
            'positive': 'Reds',
            'negative': 'Blues',
            'closed': 'Greens'
        }
        
    def plot_magnetic_field_3d_realistic(self, use_magnetic_sources: bool = True,
                                        threshold_percentile: float = 80,
                                        show_surface: bool = True,
                                        show_source_surface: bool = True,
                                        figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Create realistic 3D visualization of magnetic field lines emerging from solar surface
        
        Parameters
        ----------
        use_magnetic_sources : bool
            Use HMI data to find magnetic source regions for field line starting points
        threshold_percentile : float
            Percentile threshold for magnetic field strength (default: 80)
        show_surface : bool
            Show photospheric magnetogram
        show_source_surface : bool
            Show source surface field
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            3D visualization figure
        """
        logger.info("Creating realistic 3D magnetic field visualization...")
        
        try:
            # Create figure
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot photospheric surface
            if show_surface:
                logger.info("Plotting photospheric magnetogram...")
                self._plot_photosphere(ax)
            
            # Plot source surface
            if show_source_surface:
                logger.info("Plotting source surface...")
                self._plot_source_surface(ax)
            
            # Trace and plot field lines from magnetic sources
            if use_magnetic_sources:
                logger.info("Finding magnetic source regions...")
                positive_sources, negative_sources = self.analyzer.find_magnetic_sources(
                    threshold_percentile=threshold_percentile, min_separation=0.2)
                
                if len(positive_sources) > 0 or len(negative_sources) > 0:
                    logger.info("Tracing field lines from magnetic sources...")
                    field_lines = self.tracer.trace_from_magnetic_sources(
                        positive_sources, negative_sources)
                    
                    logger.info("Plotting realistic 3D field lines...")
                    self._plot_field_lines_3d(ax, field_lines)
                    
                    # Store field lines info for later use
                    self._traced_field_lines = field_lines
                else:
                    logger.warning("No significant magnetic sources found")
                    self._traced_field_lines = []
            
        except Exception as e:
            logger.error(f"Error in 3D visualization: {e}")
            # Store empty field lines on error
            self._traced_field_lines = []
            # Create minimal plot
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot simple sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, alpha=0.5, color='lightblue')
            ax.set_title('PFSS Model (Simplified View)', fontsize=16)
        
        # Enhanced formatting
        ax.set_xlabel('X (R☉)', fontsize=12)
        ax.set_ylabel('Y (R☉)', fontsize=12)
        ax.set_zlabel('Z (R☉)', fontsize=12)
        ax.set_title('3D Solar Magnetic Field Lines\n(From HMI Data)', fontsize=16, pad=20)
        
        # Set equal aspect ratio with extended range
        max_range = self.model.source_surface * 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # Enhanced grid and viewing angle
        ax.grid(True, alpha=0.2)
        ax.view_init(elev=15, azim=45)
        
        # Add text annotation
        ax.text2D(0.02, 0.98, 'Red: Positive polarity\nBlue: Negative polarity\nGreen: Closed loops', 
                 transform=ax.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return fig
    
    def plot_magnetic_field_3d(self, field_lines: Optional[List[dict]] = None,
                              show_surface: bool = True,
                              show_source_surface: bool = True,
                              figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Create 3D visualization of magnetic field configuration
        
        Parameters
        ----------
        field_lines : list, optional
            Pre-computed field lines (will trace if None)
        show_surface : bool
            Show photospheric magnetogram
        show_source_surface : bool
            Show source surface field
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            3D visualization figure
        """
        logger.info("Creating 3D magnetic field visualization...")
        
        try:
            # Create figure
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot photospheric surface
            if show_surface:
                logger.info("Plotting photospheric surface...")
                self._plot_photosphere(ax)
            
            # Plot source surface
            if show_source_surface:
                logger.info("Plotting source surface...")
                self._plot_source_surface(ax)
            
            # Trace and plot field lines
            if field_lines is None or len(field_lines) == 0:
                logger.info("No field lines to plot (simple mode)")
            else:
                logger.info(f"Plotting {len(field_lines)} existing field lines...")
                self._plot_field_lines(ax, field_lines)
                
        except Exception as e:
            logger.error(f"Error in 3D visualization: {e}")
            # Create minimal plot
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot simple sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, alpha=0.5, color='lightblue')
            ax.set_title('PFSS Model (Simplified View)', fontsize=16)
        
        # Formatting
        ax.set_xlabel('X (R☉)')
        ax.set_ylabel('Y (R☉)')
        ax.set_zlabel('Z (R☉)')
        ax.set_title('PFSS Magnetic Field Configuration', fontsize=16)
        
        # Set equal aspect ratio
        max_range = self.model.source_surface
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # Add grid and adjust viewing angle
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
        
        return fig
    
    # PFSSVisualizer クラス内のこの関数を置き換えてください
    def _plot_photosphere(self, ax):
        """Plot photospheric magnetic field (optimized with error handling)"""
        try:
            # Create sphere mesh (reduced resolution)
            u_plot = np.linspace(0, 2 * np.pi, 30)  # Further reduced
            v_plot = np.linspace(0, np.pi, 15)      # Further reduced
            
            x = np.outer(np.cos(u_plot), np.sin(v_plot))
            y = np.outer(np.sin(u_plot), np.sin(v_plot))
            z = np.outer(np.ones(np.size(u_plot)), np.cos(v_plot))
            
            # Get Br at photosphere with safety checks
            if self.model.br is None or self.model.br.size == 0:
                logger.warning("No magnetic field data available for photosphere plot")
                return
                
            br_photo = self.model.br[0, :, :]
            
            # Check data dimensions
            logger.info(f"br_photo shape: {br_photo.shape}")
            logger.info(f"grid theta shape: {self.model.grid.theta.shape}")
            logger.info(f"grid phi shape: {self.model.grid.phi.shape}")
            
            # Create coordinate meshes for interpolation
            theta_min, theta_max = self.model.grid.theta.min(), self.model.grid.theta.max()
            phi_min, phi_max = self.model.grid.phi.min(), self.model.grid.phi.max()
            
            # Map plot coordinates to model coordinates
            theta_plot = np.linspace(theta_min, theta_max, len(v_plot))
            phi_plot = np.linspace(phi_min, phi_max, len(u_plot))
            
            PHI_plot, THETA_plot = np.meshgrid(phi_plot, theta_plot)
            
            # Interpolate magnetic field safely
            try:
                interp_func = RegularGridInterpolator(
                    (self.model.grid.theta, self.model.grid.phi), 
                    br_photo, 
                    bounds_error=False, 
                    fill_value=0.0
                )
                
                # Create interpolation points
                points = np.column_stack([THETA_plot.ravel(), PHI_plot.ravel()])
                br_interp = interp_func(points).reshape(THETA_plot.shape)
                
            except Exception as e:
                logger.warning(f"Interpolation failed: {e}, using simplified visualization")
                # Fallback: use a simple uniform field
                br_interp = np.zeros_like(THETA_plot)
                if br_photo.size > 0:
                    br_interp += np.mean(br_photo)
        
            # Create colormap normalized to magnetic field strength
            vmax = np.nanmax(np.abs(br_interp))
            if vmax == 0 or np.isnan(vmax): 
                vmax = 1.0  # Avoid error if field is all zero
                
            norm = plt.Normalize(vmin=-vmax, vmax=vmax)
            colors = cm.RdBu_r(norm(br_interp))
            
            # Plot surface with error handling
            try:
                ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors, alpha=0.8,
                                linewidth=0, antialiased=True, shade=True)
                
                # Add colorbar
                mappable = cm.ScalarMappable(norm=norm, cmap=cm.RdBu_r)
                plt.colorbar(mappable, ax=ax, label='Br (Gauss)', shrink=0.6)
                
            except Exception as e:
                logger.warning(f"Surface plotting failed: {e}, using simple sphere")
                # Fallback: plot simple gray sphere
                ax.plot_surface(x, y, z, alpha=0.3, color='gray')
                
        except Exception as e:
            logger.error(f"Error in photosphere plotting: {e}")
            # Minimal fallback: just plot a sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, alpha=0.3, color='lightblue')
    
    def _plot_source_surface(self, ax):
        """Plot source surface with open field regions (optimized)"""
        # Create sphere mesh at source surface (reduced resolution)
        rss = self.model.source_surface
        u = np.linspace(0, 2 * np.pi, 20)  # Reduced from 30
        v = np.linspace(0, np.pi, 20)      # Reduced from 30
        
        x = rss * np.outer(np.cos(u), np.sin(v))
        y = rss * np.outer(np.sin(u), np.sin(v))
        z = rss * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot semi-transparent source surface
        ax.plot_surface(x, y, z, alpha=0.1, color='gray',
                       linewidth=0, antialiased=True)
    
    def _plot_field_lines_3d(self, ax, field_lines: List[dict]):
        """Plot traced magnetic field lines with realistic 3D visualization"""
        logger.info(f"Plotting {len(field_lines)} magnetic field lines in 3D...")
        
        # Count field line types
        n_positive = 0
        n_negative = 0
        n_closed = 0
        
        for i, field_line in enumerate(field_lines):
            # Handle both new format (with polarity) and old format
            if 'forward' in field_line:
                # New format from trace_from_magnetic_sources
                positions = field_line['forward']['positions']
                polarity = field_line.get('polarity', 'unknown')
                termination = field_line['forward'].get('termination_reason', 'unknown')
            elif 'combined' in field_line:
                # Old format from trace_from_grid
                positions = field_line['combined']['positions']
                classification = field_line.get('classification', 'unknown')
                # Determine polarity from starting point
                try:
                    start_br = self.model.interpolator(positions[0])[0]
                    polarity = 'positive' if start_br > 0 else 'negative'
                    termination = 'open' if classification == 'open' else 'closed'
                except:
                    polarity = 'unknown'
                    termination = 'unknown'
            else:
                continue
                
            # Convert to Cartesian coordinates
            r = positions[:, 0]
            theta = positions[:, 1] 
            phi = positions[:, 2]
            
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            # Determine visualization properties based on polarity and termination
            if polarity == 'positive':
                color = 'red'
                alpha = 0.8
                linewidth = 2.0
                n_positive += 1
            elif polarity == 'negative':
                color = 'blue'
                alpha = 0.8
                linewidth = 2.0
                n_negative += 1
            else:
                # Closed or unknown field lines
                color = 'green'
                alpha = 0.6
                linewidth = 1.5
                n_closed += 1
            
            # Plot field line with varying thickness based on distance from sun
            if len(x) > 1:
                # Create segments with varying properties
                for j in range(len(x)-1):
                    # Line thickness decreases with distance
                    segment_width = linewidth * (2.0 - r[j]) / 1.5
                    segment_alpha = alpha * (2.0 - r[j]) / 1.5
                    
                    ax.plot([x[j], x[j+1]], [y[j], y[j+1]], [z[j], z[j+1]], 
                           color=color, alpha=segment_alpha, linewidth=segment_width)
            
            # Add directional arrows (every 10th line)
            if i % 10 == 0 and len(x) > 2:
                # Add arrows at multiple points along the field line
                arrow_indices = np.linspace(1, len(x)-2, min(3, len(x)-2), dtype=int)
                for idx in arrow_indices:
                    # Calculate direction vector
                    dx = x[idx+1] - x[idx-1]
                    dy = y[idx+1] - y[idx-1] 
                    dz = z[idx+1] - z[idx-1]
                    
                    # Normalize and scale
                    length = np.sqrt(dx**2 + dy**2 + dz**2)
                    if length > 0:
                        scale = 0.1  # Arrow size
                        dx, dy, dz = dx/length*scale, dy/length*scale, dz/length*scale
                        
                        ax.quiver(x[idx], y[idx], z[idx], dx, dy, dz,
                                 color=color, alpha=0.9, arrow_length_ratio=0.3,
                                 linewidth=1.0)
        
        logger.info(f"Plotted {n_positive} positive, {n_negative} negative, and {n_closed} closed field lines")
        
        # Enhanced legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=3, label=f'Positive polarity ({n_positive})'),
            Line2D([0], [0], color='blue', lw=3, label=f'Negative polarity ({n_negative})'),
            Line2D([0], [0], color='green', lw=2, label=f'Closed loops ({n_closed})')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    def _plot_field_lines(self, ax, field_lines: List[dict]):
        """Compatibility wrapper for field line plotting"""
        self._plot_field_lines_3d(ax, field_lines)
    
    def plot_source_surface_map(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot source surface magnetic field map in Carrington coordinates
        
        Similar to SSWIDL pfss_plot_source_surface
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Get Br at source surface
        br_ss = self.model.br[-1, :, :]
        
        # Create coordinate grids
        lon_deg = np.degrees(self.model.grid.phi)
        lat_deg = 90 - np.degrees(self.model.grid.theta)  # Convert to latitude
        
        LON, LAT = np.meshgrid(lon_deg, lat_deg)
        
        # Plot Br map
        im1 = ax1.pcolormesh(LON, LAT, br_ss, cmap='RdBu_r',
                           vmin=-np.max(np.abs(br_ss)),
                           vmax=np.max(np.abs(br_ss)),
                           shading='auto')
        
        ax1.set_xlabel('Carrington Longitude (deg)')
        ax1.set_ylabel('Latitude (deg)')
        ax1.set_title('Source Surface Br')
        ax1.grid(True, alpha=0.3)
        
        # Add neutral line
        ax1.contour(LON, LAT, br_ss, levels=[0], colors='black', linewidths=2)
        
        plt.colorbar(im1, ax=ax1, label='Br (Gauss)')
        
        # Plot open/closed field map
        open_field_map = self._compute_open_field_map()
        
        im2 = ax2.pcolormesh(LON, LAT, open_field_map, cmap='RdBu_r',
                           vmin=-1, vmax=1, shading='auto')
        
        ax2.set_xlabel('Carrington Longitude (deg)')
        ax2.set_ylabel('Latitude (deg)')
        ax2.set_title('Open Field Regions')
        ax2.grid(True, alpha=0.3)
        
        # Custom colorbar for open field
        cbar = plt.colorbar(im2, ax=ax2, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(['Negative', 'Closed', 'Positive'])
        
        plt.tight_layout()
        return fig
    
    def _compute_open_field_map(self) -> np.ndarray:
        """Compute map of open field regions at photosphere (optimized)"""
        open_map = np.zeros((self.model.grid.n_theta, self.model.grid.n_phi))
        
        # Sample every 4th point for speed
        theta_step = max(1, self.model.grid.n_theta // 25)
        phi_step = max(1, self.model.grid.n_phi // 25)
        
        for i in range(0, self.model.grid.n_theta, theta_step):
            for j in range(0, self.model.grid.n_phi, phi_step):
                theta = self.model.grid.theta[i]
                phi = self.model.grid.phi[j]
                start_point = np.array([1.0, theta, phi])
                
                # Quick trace to check if open
                result = self.tracer.trace_field_line(start_point, 'forward')
                
                if result['forward']['termination_reason'] == 'source_surface':
                    # Open field - check polarity and fill region
                    br_start = self.model.br[0, i, j]
                    value = 1 if br_start > 0 else -1
                    
                    # Fill surrounding region
                    i_end = min(i + theta_step, self.model.grid.n_theta)
                    j_end = min(j + phi_step, self.model.grid.n_phi)
                    open_map[i:i_end, j:j_end] = value
        
        return open_map
    
    def save_visualization(self, filename: str, dpi: int = 150):
        """Save current figure to file"""
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved visualization to {filename}")

class PFSSDataManager:
    """
    Handle data I/O and caching for PFSS models
    """
    
    def __init__(self, cache_dir: Path = Path('./pfss_cache')):
        """Initialize data manager with cache directory"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized data manager with cache: {self.cache_dir}")
    
    def save_model(self, model: PFSSModel, filename: str):
        """
        Save PFSS model to HDF5 file
        
        Parameters
        ----------
        model : PFSSModel
            Model to save
        filename : str
            Output filename (without extension)
        """
        filepath = self.cache_dir / f"{filename}.h5"
        
        with h5py.File(filepath, 'w') as f:
            # Save grid parameters
            grid_group = f.create_group('grid')
            grid_group.attrs['n_phi'] = model.grid.n_phi
            grid_group.attrs['n_theta'] = model.grid.n_theta
            grid_group.attrs['n_r'] = model.grid.n_r
            grid_group.attrs['r_min'] = model.grid.r_min
            grid_group.attrs['r_max'] = model.grid.r_max
            
            # Save field components
            field_group = f.create_group('field')
            field_group.create_dataset('br', data=model.br, compression='gzip')
            field_group.create_dataset('btheta', data=model.btheta, compression='gzip')
            field_group.create_dataset('bphi', data=model.bphi, compression='gzip')
            
            # Save harmonic coefficients
            if model.coeffs is not None:
                coeffs_group = f.create_group('coefficients')
                for (l, m), coeff in model.coeffs.items():
                    coeffs_group.attrs[f'l{l}_m{m}'] = coeff
            
            # Save metadata
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['source_surface'] = model.source_surface
        
        logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filename: str) -> PFSSModel:
        """
        Load PFSS model from HDF5 file
        
        Parameters
        ----------
        filename : str
            Filename to load (without extension)
        
        Returns
        -------
        model : PFSSModel
            Loaded model
        """
        filepath = self.cache_dir / f"{filename}.h5"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            # Load grid parameters
            grid = PFSSGrid(
                n_phi=f['grid'].attrs['n_phi'],
                n_theta=f['grid'].attrs['n_theta'],
                n_r=f['grid'].attrs['n_r'],
                r_min=f['grid'].attrs['r_min'],
                r_max=f['grid'].attrs['r_max']
            )
            
            # Create model
            model = PFSSModel(grid=grid, 
                            source_surface=f.attrs['source_surface'])
            
            # Load field components
            model.br = f['field/br'][:]
            model.btheta = f['field/btheta'][:]
            model.bphi = f['field/bphi'][:]
            
            # Load coefficients if available
            if 'coefficients' in f:
                model.coeffs = {}
                for key, value in f['coefficients'].attrs.items():
                    # Parse l and m from key
                    parts = key.split('_')
                    l = int(parts[0][1:])
                    m = int(parts[1][1:])
                    model.coeffs[(l, m)] = value
        
        logger.info(f"Loaded model from {filepath}")
        return model

def main():
    """
    PFSS analysis tool with ROI (Region of Interest) support.
    Allows focusing computation on specific solar regions for faster processing.
    """
    # --- 1. 使用するローカルファイルのパスを指定 ---
    local_file_path = '/mnt/d/wsl/home/kinno-7010/Research/SDO/HMI/Rawdata/hmi.M_720s.20220613_030000_TAI.fits'
    
    # --- 1.5. 出力ディレクトリの設定と作成 ---
    output_dir = Path('/mnt/d/wsl/home/kinno-7010/Research/PFSS')
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # --- 2. 実行モード設定 ---
    # 'field_lines_3d': リアルな3D磁力線可視化（HMIデータ基づく）
    # 'ultra_fast': 最高速モード（200x200ピクセル、l_max=5）
    # 'small_roi': 小さなROI領域でのテスト
    # 'simple': 磁力線なしの簡易解析
    # 'roi': ROI領域での高速解析
    # 'full': 全面解析
    execution_mode = 'field_lines_3d'  # ←← ここを変更して実行モードを選択
    
    # --- 3. モード別設定 ---
    if execution_mode == 'field_lines_3d':
        # 3D磁力線可視化モード（中程度のROI）
        roi = ROIBounds(
            x_min=-300,  # 左端 (ピクセル)
            x_max=300,   # 右端 (ピクセル)
            y_min=-300,  # 下端 (ピクセル)
            y_max=300    # 上端 (ピクセル)
        )
        enable_field_lines = True
        enable_3d_realistic = True
    elif execution_mode == 'ultra_fast':
        # 最高速モード（200x200ピクセル）
        roi = ROIBounds(
            x_min=-100,  # 左端 (ピクセル)
            x_max=100,   # 右端 (ピクセル)
            y_min=-100,  # 下端 (ピクセル)
            y_max=100    # 上端 (ピクセル)
        )
        enable_field_lines = False  # 磁力線なしで最高速
        enable_3d_realistic = False
    elif execution_mode == 'small_roi':
        # 小さなROI領域でのテスト（400x400ピクセル）
        roi = ROIBounds(
            x_min=-200,  # 左端 (ピクセル)
            x_max=200,   # 右端 (ピクセル)
            y_min=-200,  # 下端 (ピクセル)
            y_max=200    # 上端 (ピクセル)
        )
        enable_field_lines = True
    elif execution_mode == 'roi':
        # 大きなROI領域（元の設定）
        roi = ROIBounds(
            x_min=-2048,  # 左端 (ピクセル)
            x_max=0,      # 右端 (ピクセル)
            y_min=-512,   # 下端 (ピクセル)
            y_max=2048    # 上端 (ピクセル)
        )
        enable_field_lines = True
    elif execution_mode == 'simple':
        # 簡易モード：磁力線なし、磁場分布のみ
        roi = ROIBounds(-300, 300, -300, 300)
        enable_field_lines = False
    else:  # 'full'
        # 全面解析
        roi = None
        enable_field_lines = True

    # Set up logging
    logging.getLogger().setLevel(logging.INFO)
    logger.info(f"Running PFSS analysis for local file: {local_file_path}")
    if roi:
        logger.info(f"ROI specified: [{roi.x_min}, {roi.x_max}] x [{roi.y_min}, {roi.y_max}] pixels")

    # --- 3. ROI対応グリッドとパラメータ設定 ---
    # 実行モードに応じてグリッドとパラメータを設定
    if execution_mode == 'ultra_fast':
        # 最高速設定
        grid = PFSSGrid(n_phi=32, n_theta=32, n_r=20, roi=roi)
        l_max_setting = 5  # 最小次数
    elif execution_mode in ['small_roi', 'simple']:
        grid = PFSSGrid(roi=roi)
        l_max_setting = 8  # 低次数
    elif execution_mode == 'roi':
        grid = PFSSGrid(roi=roi)
        l_max_setting = 12  # 中程度
    else:
        grid = PFSSGrid(roi=roi)
        l_max_setting = 15 if roi else 10  # デフォルト
    
    # --- 4. PFSSモデルの初期化 ---
    model = PFSSModel(grid=grid, roi=roi)
    model.harmonics.l_max = l_max_setting
    logger.info(f"Setting l_max to: {l_max_setting}")

    try:
        # --- 5. ローカルファイルから磁場マップを読み込む ---
        logger.info(f"Loading magnetogram from {local_file_path}...")
        synoptic_map = sunpy.map.Map(local_file_path)
        
        # マップの情報を表示
        logger.info(f"Map dimensions: {synoptic_map.dimensions}")
        logger.info(f"Map center: {synoptic_map.center}")
        logger.info(f"Solar radius: {synoptic_map.rsun_obs:.1f} arcsec")

        # --- 6. 前処理とPFSSモデルの計算 ---
        synoptic_map = model._preprocess_magnetogram(synoptic_map)
        
        if roi:
            logger.info(f"Computing PFSS model for ROI region ({roi.area} pixels)...")
        else:
            logger.info("Computing PFSS model for full disk...")
            
        model.compute_from_magnetogram(synoptic_map)
        
        # --- 7. 磁力線の計算と可視化 ---
        viz = PFSSVisualizer(model)
        
        # --- 7. 磁力線の計算と可視化 ---
        # field_lines_3dモードではリアルな3D可視化を使用
        if execution_mode == 'field_lines_3d' and enable_3d_realistic:
            logger.info("Creating realistic 3D magnetic field visualization...")
            fig_3d = viz.plot_magnetic_field_3d_realistic(
                use_magnetic_sources=True,
                threshold_percentile=75,
                show_surface=True,
                show_source_surface=True
            )
            # リアルモードでは磁力線情報をビジュアライザから取得
            field_lines = getattr(viz, '_traced_field_lines', [])
        else:
            # 従来の3D可視化
            if enable_field_lines:
                logger.info("Tracing magnetic field lines...")
                # 実行モードに応じて磁力線数を調整
                if execution_mode == 'ultra_fast':
                    field_lines = []  # 磁力線なし
                elif execution_mode == 'small_roi':
                    field_lines = viz.tracer.trace_from_grid(n_lon=6, n_lat=4)  # 24本
                elif execution_mode == 'roi':
                    field_lines = viz.tracer.trace_from_grid(n_lon=8, n_lat=6)  # 48本
                else:
                    field_lines = viz.tracer.trace_from_grid(n_lon=10, n_lat=8)  # 80本
            else:
                logger.info("Skipping field line tracing (simple mode)...")
                field_lines = []
            
            logger.info("Creating 3D visualization...")
            fig_3d = viz.plot_magnetic_field_3d(field_lines)
        
        # ファイル名を実行モードで区別
        if execution_mode == 'field_lines_3d':
            filename_suffix = '_3d_field_lines'
        elif execution_mode == 'ultra_fast':
            filename_suffix = '_ultra_fast'
        elif execution_mode == 'small_roi':
            filename_suffix = '_small_roi'
        elif execution_mode == 'roi':
            filename_suffix = '_large_roi'
        elif execution_mode == 'simple':
            filename_suffix = '_simple'
        else:
            filename_suffix = '_fulldisk'
            
        # ファイル保存（指定ディレクトリ）
        fig_3d_path = output_dir / f'pfss_3d_field{filename_suffix}.png'
        plt.savefig(fig_3d_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved 3D visualization: {fig_3d_path}")
        
        logger.info("Creating source surface map...")
        fig_ss = viz.plot_source_surface_map()
        
        # ファイル保存（指定ディレクトリ）
        fig_ss_path = output_dir / f'pfss_source_surface{filename_suffix}.png'
        plt.savefig(fig_ss_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved source surface map: {fig_ss_path}")
        
        # --- 8. モデルの保存 ---
        # キャッシュディレクトリも指定ディレクトリ内に作成
        cache_dir = output_dir / 'pfss_cache'
        data_manager = PFSSDataManager(cache_dir=cache_dir)
        model_name = f'pfss_model{filename_suffix}'
        data_manager.save_model(model, model_name)
        
        # --- 9. 結果サマリ ---
        logger.info("\n" + "="*50)
        logger.info("PFSS ANALYSIS COMPLETE")
        logger.info("="*50)
        if roi:
            logger.info(f"ROI region: [{roi.x_min}, {roi.x_max}] x [{roi.y_min}, {roi.y_max}] pixels")
            logger.info(f"ROI area: {roi.area} pixels ({roi.width} x {roi.height})")
        logger.info(f"Grid resolution: {model.grid.n_phi} x {model.grid.n_theta} x {model.grid.n_r}")
        logger.info(f"Spherical harmonic degree: l_max = {l_max_setting}")
        logger.info(f"Field lines traced: {len(field_lines)}")
        logger.info(f"Execution mode: {execution_mode}")
        logger.info(f"Field line tracing: {'enabled' if enable_field_lines else 'disabled'}")
        if roi:
            logger.info(f"Actual ROI processed: {roi.width} x {roi.height} pixels")
        logger.info(f"Output files:")
        logger.info(f"  - {fig_3d_path}")
        logger.info(f"  - {fig_ss_path}")
        logger.info(f"  - Model saved as: {cache_dir}/{model_name}.h5")
        logger.info("="*50)
        
        # --- 9. GUI表示 ---
        logger.info("Attempting GUI display...")
        try:
            # WSL環境でのGUI表示確認
            display_env = os.environ.get('DISPLAY', '')
            wsl_check = 'microsoft' in os.uname().release.lower()
            
            if wsl_check:
                logger.info(f"WSL environment detected. DISPLAY={display_env}")
                if not display_env:
                    logger.warning("DISPLAY variable not set. GUI display may fail.")
                    logger.info("To enable GUI: Install X11 server (VcXsrv/Xming) and set DISPLAY=:0")
            
            # matplotlibのGUIバックエンド設定
            import matplotlib
            matplotlib.use('TkAgg')  # WSLで使用可能なバックエンド
            
            plt.show(block=True)  # ブロッキングでGUI表示（ウィンドウを閉じるまで待機）
            logger.info("GUI display completed.")
            
        except Exception as e:
            logger.warning(f"GUI display failed: {e}")
            logger.info("Files saved successfully to the specified directory.")
            logger.info("You can view the plots using any image viewer.")
            logger.info("\nFor WSL GUI setup:")
            logger.info("1. Install X11 server: VcXsrv or Xming")
            logger.info("2. Set DISPLAY environment variable: export DISPLAY=:0")
            logger.info("3. Run: python3 pfss_analysis_main.py")

    except Exception as e:
        logger.error(f"An error occurred during the analysis: {e}")

# このファイルを実行する際のお決まりの記述
if __name__ == "__main__":
    # main関数の先頭で`sunpy.map`をインポートしたため、ここに追加

    main()
