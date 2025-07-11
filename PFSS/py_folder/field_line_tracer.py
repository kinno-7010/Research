#!/usr/bin/env python3
"""
Field Line Tracer
=================

Advanced magnetic field line tracing with adaptive step size control.

Author: Solar Physics Research System
Date: 2025-01-11
"""

import numpy as np
from scipy.integrate import solve_ivp
import logging
from typing import Dict, List
import concurrent.futures

# Set up logging
logger = logging.getLogger(__name__)

class FieldLineTracer:
    """
    Advanced field line tracing with adaptive step size control
    """
    
    def __init__(self, pfss_model):
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