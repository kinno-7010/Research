"""
PFSS IDL to Python Conversion Package

This package contains Python conversions of IDL routines for Potential Field
Source Surface (PFSS) modeling and analysis.

Key modules:
- forw_euler: Forward Euler integration
- gaussquad_legendre: Legendre collocation points and weights
- spherical_transform: Forward spherical harmonic transform
- inv_spherical_transform: Inverse spherical harmonic transform
- pfss_data_block: Data structure management
- weights_legendre: Legendre integration weights
- mean_dtheta: Mean calculation over theta dimension
- pfss_draw_field: Field line visualization
- pfss_get_potl_coeffs: Spherical harmonic coefficients
- pfss_field_start_coord: Field line starting points
- Various utility functions

Converted from IDL to Python - 2025
"""

# Import key functions for easy access
from .forw_euler import forw_euler
from .gaussquad_legendre import gaussquad_legendre
from .get_interpolation_index import get_interpolation_index
from .get_string_number import get_string_number
from .inv_spherical_transform import inv_spherical_transform
from .linrange import linrange
from .logrange import logrange
from .mean_dtheta import mean_dtheta
from .pfss_data_block import pfss_data_block, PFSSDataBlock
from .pfss_print_time import pfss_print_time
from .sign_mld import sign_mld
from .spherical_transform import spherical_transform
from .union import union
from .weights_legendre import weights_legendre

# Import newly converted modules
from .pfss_draw_field import pfss_draw_field
from .pfss_draw_field2 import pfss_draw_field2
from .pfss_draw_field3 import pfss_draw_field3
from .pfss_draw_field_vrml import pfss_draw_field_vrml
from .pfss_fake_index import pfss_fake_index
from .pfss_field_start_coord import pfss_field_start_coord
from .pfss_get_potl_coeffs import pfss_get_potl_coeffs
from .pfss_to_spherical import pfss_to_spherical

__version__ = "1.1.0"
__author__ = "Converted from IDL by Claude"