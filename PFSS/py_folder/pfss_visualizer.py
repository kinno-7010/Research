#!/usr/bin/env python3
"""
PFSS Visualizer
===============

Advanced 3D visualization for PFSS models.

Author: Solar Physics Research System
Date: 2025-01-11
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import logging
from typing import Tuple, Optional, List
from scipy.interpolate import interp2d

from field_line_tracer import FieldLineTracer

# Set up logging
logger = logging.getLogger(__name__)

class PFSSVisualizer:
    """
    Advanced 3D visualization for PFSS models
    """
    
    def __init__(self, pfss_model):
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