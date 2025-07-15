#!/usr/bin/env python3
"""
pfss_to_spherical.py - Given magnetic field data created via the pfss package, this
                       procedure creates a spherical_field_data structure for use with
                       the spherical_* routines.

Usage: pfss_to_spherical(sph_data, free_heap=free_heap, no_copy=no_copy)

Parameters:
    sph_data: a structure of type spherical_field_data
    free_heap: if set, will perform a ptr_free operation on all valid
               pointers before creating a new pointer
    no_copy: if set, will set the /no_copy keyword when calling ptr_new

Common Blocks:
    uses pfss common block (see pfss_data_block.py) as input data

Notes:
    1. Those variables in the pfss common block for which there is no
       corresponding field in the sph_data structure are not carried over
    2. If no_copy is set, variables in the pfss_data_block will be
       undefined afterward.

M.DeRosa - 13 Dec 2005 - created
           27 Jan 2006 - sets sph_data.lonbounds
           9 Apr 2007 - added no_copy keyword

Converted to Python by Claude Code
"""

import numpy as np
import copy

# Import other modules from the PFSS package
from pfss_data_block import PFSSDataBlock
from spherical_field_data__define import SphericalFieldData


def pfss_to_spherical(data_block=None, free_heap=False, no_copy=False):
    """
    Convert PFSS data to spherical field data structure
    
    Parameters:
    -----------
    data_block : PfssDataBlock, optional
        PFSS data block (if None, uses global instance)
    free_heap : bool, optional
        If True, will perform cleanup operations on pointers
    no_copy : bool, optional
        If True, will move data instead of copying
    
    Returns:
    --------
    SphericalFieldData : Spherical field data structure
    """
    
    # Access pfss common block
    if data_block is None:
        data_block = PFSSDataBlock()
    
    # Create output data structure
    sph_data = SphericalFieldData()
    
    # Transfer data fields
    if hasattr(data_block, 'br') and data_block.br is not None:
        if no_copy:
            sph_data.br = data_block.br
            data_block.br = None
        else:
            sph_data.br = copy.deepcopy(data_block.br)
    
    if hasattr(data_block, 'bth') and data_block.bth is not None:
        if no_copy:
            sph_data.bth = data_block.bth
            data_block.bth = None
        else:
            sph_data.bth = copy.deepcopy(data_block.bth)
    
    if hasattr(data_block, 'bph') and data_block.bph is not None:
        if no_copy:
            sph_data.bph = data_block.bph
            data_block.bph = None
        else:
            sph_data.bph = copy.deepcopy(data_block.bph)
    
    # Transfer coordinate arrays
    if hasattr(data_block, 'rix') and data_block.rix is not None:
        if no_copy:
            sph_data.rix = data_block.rix
            data_block.rix = None
        else:
            sph_data.rix = copy.deepcopy(data_block.rix)
    
    if hasattr(data_block, 'lat') and data_block.lat is not None:
        if no_copy:
            sph_data.lat = data_block.lat
            data_block.lat = None
        else:
            sph_data.lat = copy.deepcopy(data_block.lat)
    
    if hasattr(data_block, 'lon') and data_block.lon is not None:
        if no_copy:
            sph_data.lon = data_block.lon
            data_block.lon = None
        else:
            sph_data.lon = copy.deepcopy(data_block.lon)
    
    # Transfer theta and phi if they exist
    if hasattr(data_block, 'theta') and data_block.theta is not None:
        if no_copy:
            sph_data.theta = data_block.theta
            data_block.theta = None
        else:
            sph_data.theta = copy.deepcopy(data_block.theta)
    
    if hasattr(data_block, 'phi') and data_block.phi is not None:
        if no_copy:
            sph_data.phi = data_block.phi
            data_block.phi = None
        else:
            sph_data.phi = copy.deepcopy(data_block.phi)
    
    # Transfer field line data if it exists
    if hasattr(data_block, 'ptr') and data_block.ptr is not None:
        if no_copy:
            sph_data.ptr = data_block.ptr
            data_block.ptr = None
        else:
            sph_data.ptr = copy.deepcopy(data_block.ptr)
    
    if hasattr(data_block, 'ptth') and data_block.ptth is not None:
        if no_copy:
            sph_data.ptth = data_block.ptth
            data_block.ptth = None
        else:
            sph_data.ptth = copy.deepcopy(data_block.ptth)
    
    if hasattr(data_block, 'ptph') and data_block.ptph is not None:
        if no_copy:
            sph_data.ptph = data_block.ptph
            data_block.ptph = None
        else:
            sph_data.ptph = copy.deepcopy(data_block.ptph)
    
    if hasattr(data_block, 'nstep') and data_block.nstep is not None:
        if no_copy:
            sph_data.nstep = data_block.nstep
            data_block.nstep = None
        else:
            sph_data.nstep = copy.deepcopy(data_block.nstep)
    
    # Transfer starting point data if it exists
    if hasattr(data_block, 'str') and data_block.str is not None:
        if no_copy:
            sph_data.str = data_block.str
            data_block.str = None
        else:
            sph_data.str = copy.deepcopy(data_block.str)
    
    if hasattr(data_block, 'stth') and data_block.stth is not None:
        if no_copy:
            sph_data.stth = data_block.stth
            data_block.stth = None
        else:
            sph_data.stth = copy.deepcopy(data_block.stth)
    
    if hasattr(data_block, 'stph') and data_block.stph is not None:
        if no_copy:
            sph_data.stph = data_block.stph
            data_block.stph = None
        else:
            sph_data.stph = copy.deepcopy(data_block.stph)
    
    # Set longitude bounds if longitude data exists
    if hasattr(sph_data, 'lon') and sph_data.lon is not None:
        sph_data.lonbounds = [np.min(sph_data.lon), np.max(sph_data.lon)]
    
    # Transfer dimensions
    if hasattr(data_block, 'nr') and data_block.nr is not None:
        sph_data.nr = data_block.nr
    
    if hasattr(data_block, 'nlat') and data_block.nlat is not None:
        sph_data.nlat = data_block.nlat
    
    if hasattr(data_block, 'nlon') and data_block.nlon is not None:
        sph_data.nlon = data_block.nlon
    
    # Transfer potential field coefficients if they exist
    if hasattr(data_block, 'phiat') and data_block.phiat is not None:
        if no_copy:
            sph_data.phiat = data_block.phiat
            data_block.phiat = None
        else:
            sph_data.phiat = copy.deepcopy(data_block.phiat)
    
    if hasattr(data_block, 'phibt') and data_block.phibt is not None:
        if no_copy:
            sph_data.phibt = data_block.phibt
            data_block.phibt = None
        else:
            sph_data.phibt = copy.deepcopy(data_block.phibt)
    
    return sph_data


class SphericalFieldData:
    """
    Spherical field data structure
    """
    
    def __init__(self):
        self.br = None      # Radial magnetic field
        self.bth = None     # Theta component of magnetic field
        self.bph = None     # Phi component of magnetic field
        self.rix = None     # Radial grid
        self.lat = None     # Latitude grid
        self.lon = None     # Longitude grid
        self.theta = None   # Theta coordinates
        self.phi = None     # Phi coordinates
        self.ptr = None     # Field line radial coordinates
        self.ptth = None    # Field line theta coordinates
        self.ptph = None    # Field line phi coordinates
        self.nstep = None   # Number of steps in each field line
        self.str = None     # Starting radial coordinates
        self.stth = None    # Starting theta coordinates
        self.stph = None    # Starting phi coordinates
        self.nr = None      # Number of radial points
        self.nlat = None    # Number of latitude points
        self.nlon = None    # Number of longitude points
        self.lonbounds = None  # Longitude bounds
        self.phiat = None   # Potential field coefficients (A)
        self.phibt = None   # Potential field coefficients (B)
    
    def __str__(self):
        """String representation of the spherical field data"""
        return f"SphericalFieldData(nr={self.nr}, nlat={self.nlat}, nlon={self.nlon})"
    
    def __repr__(self):
        return self.__str__()