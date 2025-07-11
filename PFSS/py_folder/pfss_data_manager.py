#!/usr/bin/env python3
"""
PFSS Data Manager
=================

Handle data I/O and caching for PFSS models.

Author: Solar Physics Research System
Date: 2025-01-11
"""

import h5py
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

from pfss_grid import PFSSGrid

# Set up logging
logger = logging.getLogger(__name__)

class PFSSDataManager:
    """
    Handle data I/O and caching for PFSS models
    """
    
    def __init__(self, cache_dir: Path = Path('./pfss_cache')):
        """Initialize data manager with cache directory"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized data manager with cache: {self.cache_dir}")
    
    def save_model(self, model, filename: str):
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
    
    def load_model(self, filename: str):
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
        # Import here to avoid circular imports
        from pfss_model import PFSSModel
        
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