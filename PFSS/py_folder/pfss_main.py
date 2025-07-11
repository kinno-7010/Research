#!/usr/bin/env python3
"""
PFSS Analysis Main
==================

Main execution script for PFSS analysis with integrated modules.

Author: Solar Physics Research System
Date: 2025-01-11
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import warnings

# Import PFSS modules
from constants import LOG_FORMAT
from pfss_grid import PFSSGrid
from pfss_model import PFSSModel
from pfss_visualizer import PFSSVisualizer
from pfss_data_manager import PFSSDataManager

# Set up logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

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
    field_lines = viz.tracer.trace_from_grid(n_lon=30, n_lat=15)
    
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