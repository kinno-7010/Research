#!/usr/bin/env python3
"""
Constants for PFSS Analysis
==========================

Physical and numerical constants used in PFSS model calculations.

Author: Solar Physics Research System
Date: 2025-01-11
"""

# Physical constants
R_SUN = 696000.0  # Solar radius in km
R_SUN_RS = 1.0    # Solar radius in solar radii units
DEFAULT_RSS = 2.5  # Default source surface radius in solar radii
CARRINGTON_ROTATION_PERIOD = 27.2753  # days

# Numerical parameters
DEFAULT_N_PHI = 384      # Default longitude points
DEFAULT_N_THETA = 192    # Default latitude points  
DEFAULT_N_R = 39         # Default radial points
DEFAULT_L_MAX = 90       # Default maximum harmonic degree

# Logging format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'