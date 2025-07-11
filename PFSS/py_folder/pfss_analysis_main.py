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
5. Integration with SDO/HMI and STEREO data

Based on SSWIDL PFSS implementation analysis with performance optimizations
and enhanced coordinate system handling.

Author: Solar Physics Research System
Date: 2025-01-11
"""

import numpy as np
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
class PFSSGrid:
    """Data structure for PFSS grid following SSWIDL conventions"""
    n_phi: int = 384      # Longitude points
    n_theta: int = 192    # Latitude points  
    n_r: int = 39         # Radial points
    r_min: float = 1.0    # Minimum radius (R_sun)
    r_max: float = 2.5    # Maximum radius (source surface)
    
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

        br_data = mag_map.data
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

class FieldLineTracer:
    """
    Advanced field line tracing with adaptive step size control
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
        
        # Tracing parameters (following SSWIDL conventions)
        self.step_min = 0.001  # Minimum step size in R_sun
        self.step_max = 0.1    # Maximum step size in R_sun
        self.max_steps = 10000
        
        logger.info("Initialized field line tracer")
    
    def _field_function(self, s: float, pos: np.ndarray) -> np.ndarray:
        """
        Field line differential equation: dx/ds = B(x)/|B(x)|
        
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
        # Get field at current position
        field = self.interpolator(pos)
        br, btheta, bphi = field
        
        # Convert to Cartesian for integration
        r, theta, phi = pos
        
        # Field magnitude
        b_mag = np.sqrt(br**2 + btheta**2 + bphi**2)
        
        if b_mag < 1e-10:
            return np.zeros(3)
        
        # Normalized field components in spherical coordinates
        dr_ds = br / b_mag
        dtheta_ds = btheta / (r * b_mag)
        dphi_ds = bphi / (r * np.sin(theta) * b_mag)
        
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
        
        # Solve ODE with adaptive stepping
        sol = solve_ivp(
            lambda s, pos: sign * self._field_function(s, pos),
            [0, sign * 1000],  # Large s_span
            start_point,
            method='RK45',
            dense_output=True,
            events=boundary_event,
            rtol=1e-6,
            atol=1e-8,
            max_step=self.step_max
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
    
    def trace_from_grid(self, n_lon: int = 30, n_lat: int = 15, 
                       r_start: float = 1.0) -> List[dict]:
        """
        Trace field lines from uniform grid of starting points
        
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
        
        # Create starting grid
        lon_grid = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        lat_grid = np.linspace(-np.pi/2 + 0.1, np.pi/2 - 0.1, n_lat)
        
        field_lines = []
        
        # Use parallel processing for efficiency
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for lon in lon_grid:
                for lat in lat_grid:
                    theta = lat + np.pi/2  # Convert to colatitude
                    start_point = np.array([r_start, theta, lon])
                    
                    future = executor.submit(
                        self.trace_field_line, start_point, 'both'
                    )
                    futures.append(future)
            
            # Collect results
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    field_lines.append(result)
                    
                    if (i + 1) % 50 == 0:
                        logger.debug(f"Traced {i + 1}/{len(futures)} field lines")
                        
                except Exception as e:
                    logger.warning(f"Failed to trace field line: {e}")
        
        logger.info(f"Completed tracing {len(field_lines)} field lines")
        return field_lines

class PFSSVisualizer:
    """
    Advanced 3D visualization for PFSS models
    """
    
    def __init__(self, pfss_model: PFSSModel):
        """Initialize visualizer with PFSS model"""
        self.model = pfss_model
        self.tracer = FieldLineTracer(pfss_model)
        
        # Visualization parameters
        self.cmap_field = 'RdBu_r'
        self.cmap_polarity = {
            'positive': 'Reds',
            'negative': 'Blues',
            'closed': 'Greens'
        }
        
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
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot photospheric surface
        if show_surface:
            self._plot_photosphere(ax)
        
        # Plot source surface
        if show_source_surface:
            self._plot_source_surface(ax)
        
        # Trace and plot field lines
        if field_lines is None:
            field_lines = self.tracer.trace_from_grid(n_lon=40, n_lat=20)
        
        self._plot_field_lines(ax, field_lines)
        
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
    
    def _plot_photosphere(self, ax):
        """Plot photospheric magnetic field"""
        # Create sphere mesh
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Get Br at photosphere
        br_photo = self.model.br[0, :, :]
        
        # Interpolate to mesh
        theta_mesh, phi_mesh = np.meshgrid(
            np.linspace(0, np.pi, 50),
            np.linspace(0, 2*np.pi, 50),
            indexing='ij'
        )
        
        from scipy.interpolate import interp2d
        interp_func = interp2d(self.model.grid.phi, self.model.grid.theta, 
                              br_photo.T, kind='linear')
        br_mesh = interp_func(phi_mesh.ravel(), theta_mesh.ravel())
        br_mesh = br_mesh.reshape(theta_mesh.shape)
        
        # Create colormap normalized to magnetic field strength
        norm = plt.Normalize(vmin=-np.max(np.abs(br_mesh)), 
                           vmax=np.max(np.abs(br_mesh)))
        colors = cm.RdBu_r(norm(br_mesh))
        
        # Plot surface
        ax.plot_surface(x, y, z, facecolors=colors, alpha=0.8,
                       linewidth=0, antialiased=True, shade=True)
        
        # Add colorbar
        mappable = cm.ScalarMappable(norm=norm, cmap=cm.RdBu_r)
        plt.colorbar(mappable, ax=ax, label='Br (Gauss)', shrink=0.6)
    
    def _plot_source_surface(self, ax):
        """Plot source surface with open field regions"""
        # Create sphere mesh at source surface
        rss = self.model.source_surface
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        
        x = rss * np.outer(np.cos(u), np.sin(v))
        y = rss * np.outer(np.sin(u), np.sin(v))
        z = rss * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot semi-transparent source surface
        ax.plot_surface(x, y, z, alpha=0.1, color='gray',
                       linewidth=0, antialiased=True)
    
    def _plot_field_lines(self, ax, field_lines: List[dict]):
        """Plot traced magnetic field lines with classification coloring"""
        logger.info(f"Plotting {len(field_lines)} field lines...")
        
        # Count field line types
        n_open = 0
        n_closed = 0
        
        for i, field_line in enumerate(field_lines):
            if 'combined' not in field_line:
                continue
                
            # Get field line data
            positions = field_line['combined']['positions']
            classification = field_line['classification']
            
            # Convert to Cartesian coordinates
            r = positions[:, 0]
            theta = positions[:, 1]
            phi = positions[:, 2]
            
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            # Determine color based on classification and polarity
            if classification == 'open':
                # Check polarity at photosphere
                start_br = self.model.interpolator(positions[0])[0]
                color = 'red' if start_br > 0 else 'blue'
                alpha = 0.7
                linewidth = 1.5
                n_open += 1
            else:
                color = 'green'
                alpha = 0.5
                linewidth = 1.0
                n_closed += 1
            
            # Plot field line
            ax.plot(x, y, z, color=color, alpha=alpha, linewidth=linewidth)
            
            # Add arrows to show direction (every 10th line)
            if i % 10 == 0 and len(x) > 10:
                mid_idx = len(x) // 2
                ax.quiver(x[mid_idx], y[mid_idx], z[mid_idx],
                         x[mid_idx+1] - x[mid_idx],
                         y[mid_idx+1] - y[mid_idx], 
                         z[mid_idx+1] - z[mid_idx],
                         color=color, alpha=0.8, arrow_length_ratio=0.3)
        
        logger.info(f"Plotted {n_open} open and {n_closed} closed field lines")
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Open (positive)'),
            Line2D([0], [0], color='blue', lw=2, label='Open (negative)'),
            Line2D([0], [0], color='green', lw=2, label='Closed')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
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
        """Compute map of open field regions at photosphere"""
        open_map = np.zeros((self.model.grid.n_theta, self.model.grid.n_phi))
        
        # Trace field lines from photosphere
        for i, theta in enumerate(self.model.grid.theta):
            for j, phi in enumerate(self.model.grid.phi):
                start_point = np.array([1.0, theta, phi])
                
                # Quick trace to check if open
                result = self.tracer.trace_field_line(start_point, 'forward')
                
                if result['forward']['termination_reason'] == 'source_surface':
                    # Open field - check polarity
                    br_start = self.model.br[0, i, j]
                    open_map[i, j] = 1 if br_start > 0 else -1
        
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
    Example usage of PFSS analysis tool
    """
    # Set up logging
    logging.getLogger().setLevel(logging.INFO)
    
    # Create test magnetogram (dipole + quadrupole)
    logger.info("Creating test magnetogram...")
    
    grid = PFSSGrid()
    theta, phi = np.meshgrid(grid.theta, grid.phi, indexing='ij')
    
    # Simple dipole + quadrupole field
    br_photo = (10 * np.cos(theta) +  # Dipole
               5 * np.sin(theta)**2 * np.cos(2*phi))  # Quadrupole
    
    # Create and compute PFSS model
    logger.info("Computing PFSS model...")
    model = PFSSModel(grid=grid)
    model.compute_from_magnetogram(br_photo)
    
    # Create visualizer
    viz = PFSSVisualizer(model)
    
    # Trace field lines
    logger.info("Tracing magnetic field lines...")
    field_lines = model.tracer.trace_from_grid(n_lon=30, n_lat=15)
    
    # Create 3D visualization
    logger.info("Creating 3D visualization...")
    fig_3d = viz.plot_magnetic_field_3d(field_lines)
    viz.save_visualization('pfss_3d_field.png')
    
    # Create source surface map
    fig_ss = viz.plot_source_surface_map()
    viz.save_visualization('pfss_source_surface.png')
    
    # Save model
    data_manager = PFSSDataManager()
    data_manager.save_model(model, 'test_model')
    
    logger.info("Analysis complete!")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()