"""
pfss_mag_create.py - This procedure resamples a magnetic map of the full
solar surface (such as a synoptic map) onto either a Legendre grid or an
equally spaced lat-lon grid. The Legendre grid is compatible with the PFSS
field extrapolation software included in this package.

usage: magout, lat, lon = pfss_mag_create(magtype, nlat=nlat, gridtype=gridtype, 
                                          file=file, quiet=quiet)

Parameters:
    magtype: 0=IDL save file or structure containing flux concentrations
             1=Wilcox text file 
             2=GONG 360x180 magnetogram in FITS format
             3=MDI 3600x1080 synoptic charts in FITS format
             4=HMI 3600x1440 diachronic charts in FITS format
    nlat: number of gridpoints in latitude (default=48)
    gridtype: 1=regularly spaced grid in lat and lon
              2=Legendre grid (default)
    file: filename of input magnetic data (or structure containing the data)
    quiet: set if minimal screen output is desired

Returns:
    magout: output magnetogram image in (lon,lat) format
    lat: latitude grid
    lon: longitude grid

Notes:
    - This procedure merely resamples the input data. It does NOT convert from
      line-of-sight magnetic fields to Br (as would be needed for PFSS extrapolation).
    - The total flux should be approximately conserved during resampling.
    - One can display the resulting magnetogram using plotting functions.

M.DeRosa - 30 Jan 2002 - converted from earlier script
           5 Mar 2002 - now successfully reads Wilcox magnetogram tables
          27 Jun 2002 - added quiet keyword
           8 Nov 2002 - added rectangular grid capability
          12 May 2003 - converted common block to PFSS package format
          21 May 2003 - added magtype = 0
           9 Mar 2005 - now parses WSO text files
G.Petrie - 14 Jul 2006 - added magtype=2 (GONG synoptic maps)
M.DeRosa - 25 Sep 2007 - enabled structure input for magtype=0
          15 Apr 2009 - added magtype=3 (MDI synoptic charts)
           4 Mar 2010 - for magtype 1 (Wilcox) converts to radial field
          19 Mar 2012 - added magtype=4 (HMI diachronic charts)
           1 Jun 2012 - added better treatment of missing data for MDI
           1 Feb 2022 - added .h5 file support for magtype=0
"""

import numpy as np
import os
import h5py
from scipy.interpolate import interpn, interp2d
from astropy.io import fits
import struct
import warnings

# Import required modules from the converted package
from pfss_data_block import PfssDataBlock
from gaussquad_legendre import gaussquad_legendre
from linrange import linrange
from get_interpolation_index import get_interpolation_index
from pfss_print_time import pfss_print_time


def pfss_mag_create(magtype, nlat=48, gridtype=2, file=None, quiet=False):
    """
    Resamples a magnetic map onto either a Legendre grid or equally spaced lat-lon grid.
    
    Args:
        magtype (int): Type of magnetic data input
        nlat (int): Number of gridpoints in latitude (default=48)
        gridtype (int): 1=regularly spaced grid, 2=Legendre grid (default)
        file (str or dict): Filename or structure containing magnetic data
        quiet (bool): Set if minimal screen output is desired
        
    Returns:
        tuple: (magout, lat, lon) where magout is the magnetogram array
    """
    
    # Set up grid dimensions
    nlat = int(round(nlat))
    nlon = nlat * 2
    
    # Set up coordinate grid
    if gridtype == 1:
        # Uniformly spaced grid
        lat_edges = linrange(nlat + 1, -90, 90)
        lat = (lat_edges[:-1] + lat_edges[1:]) / 2
        theta = (90 - lat) * np.pi / 180
        weights = np.sin(theta)
        weights[0] = 0.5
        weights[-1] = 0.5
    else:
        # Legendre grid
        cth, weights = gaussquad_legendre(nlat)
        theta = np.arccos(cth)  # radians
        lat = 90 - theta * 180 / np.pi
    
    lon_edges = linrange(nlon + 1, 0, 360)
    lon = lon_edges[:-1]
    phi = lon * np.pi / 180
    
    # Process different magnetic data types
    if magtype == 0:
        # IDL save file filled with flux concentrations
        if file is None:
            raise ValueError("pfss_mag_create: file must be specified for magtype=0")
        
        if isinstance(file, str):
            # String filename
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext == '.sav':
                # IDL save file - would need scipy.io.readsav or similar
                raise NotImplementedError("IDL save file support not implemented")
            elif file_ext == '.h5':
                # HDF5 file
                with h5py.File(file, 'r') as h5file:
                    sfield = h5file['evolving_model_snapshot']
                    phis = sfield['phis'][:]
                    thetas = sfield['thetas'][:]
                    fluxs = sfield['fluxs'][:]
                    nflux = sfield['nflux'][()]
            else:
                raise ValueError(f"pfss_mag_create: file type {file_ext} not recognized")
        elif isinstance(file, dict):
            # Structure/dictionary
            phis = file['phis']
            thetas = file['thetas']
            fluxs = file['fluxs']
            nflux = file['nflux']
        else:
            raise ValueError("pfss_mag_create: file keyword not valid")
        
        # Normalize phi to [0, 2π]
        phis = (phis + 2 * np.pi) % (2 * np.pi)
        
        # Create magnetogram by adding in each flux
        mag = np.zeros((nlon, nlat))
        if not quiet:
            print(f"  pfss_mag_create: adding {nflux} sources")
        
        for i in range(nflux):
            if not quiet:
                pfss_print_time(f"  adding sources: ", i+1, nflux)
            
            # Find where this source is on our grid
            thc = get_interpolation_index(lat, 90 - np.degrees(thetas[i]))
            thc1 = int(thc)
            thc2 = min(thc1 + 1, nlat - 1)
            
            phi_extended = np.append(phi, 2 * np.pi)
            phc = get_interpolation_index(phi_extended, phis[i])
            phc1 = int(phc)
            phc2 = (phc1 + 1) % nlon
            
            # Add flux to grid, divided by areal factor
            bco = thc - thc1
            aco = phc - phc1
            
            mag[phc1, thc1] += (1 - aco) * (1 - bco) * fluxs[i] / weights[thc1]
            mag[phc2, thc1] += aco * (1 - bco) * fluxs[i] / weights[thc1]
            mag[phc1, thc2] += (1 - aco) * bco * fluxs[i] / weights[thc2]
            mag[phc2, thc2] += aco * bco * fluxs[i] / weights[thc2]
        
        magout = mag * np.mean(weights)  # normalization
        
    elif magtype == 1:
        # Wilcox line-of-sight synoptic map
        if file is None:
            raise ValueError("pfss_mag_create: filename must be specified")
        
        # Parse Wilcox text file
        with open(file, 'rb') as f:
            table_bytes = f.read()
            table = table_bytes.decode('ascii', errors='ignore')
        
        # Find CT markers
        ct_positions = []
        pos = 0
        while True:
            pos = table.find('CT', pos)
            if pos == -1:
                break
            ct_positions.append(pos)
            pos += 1
        
        # Extract longitudes
        longitudes = []
        for pos in ct_positions:
            lon_str = table[pos+7:pos+10]
            try:
                longitudes.append(int(float(lon_str)))
            except ValueError:
                continue
        
        # Read data
        data = np.zeros((72, 30))  # 72 lon bins, 30 slat bins
        
        with open(file, 'r') as f:
            for i in range(72):
                try:
                    f.seek(ct_positions[i] + 18)
                    # Read the data in chunks as specified in original
                    buff1 = list(map(float, f.read(54).split()))[:6]
                    buff2 = list(map(float, f.read(72).split()))[:8]
                    buff3 = list(map(float, f.read(72).split()))[:8]
                    buff4 = list(map(float, f.read(72).split()))[:8]
                    data[i, :] = np.array(buff1 + buff2 + buff3 + buff4)
                except:
                    continue
        
        # Align and reverse arrays
        shift_idx = np.where(np.array(longitudes) == 360)[0]
        if len(shift_idx) > 0:
            data = np.roll(data, -shift_idx[0], axis=0)
        data = np.flip(data, axis=0)  # longitude increasing
        data = np.flip(data, axis=1)  # latitude increasing
        
        # Convert from line-of-sight to radial field
        dlonix = linrange(72, 0, 355) + 2.5
        dslatix = linrange(30, 14.5, -14.5) / 15
        dlatix = np.arcsin(dslatix) * 180 / np.pi
        slatgrid = np.outer(np.ones(72), dslatix)
        data = data / np.sqrt(1 - slatgrid**2)
        
        # Remap onto our grid
        dlatinterp = get_interpolation_index(np.flip(dlatix), lat)
        dloninterp = get_interpolation_index(dlonix, lon)
        
        # Use scipy interpolation
        lon_grid, lat_grid = np.meshgrid(dloninterp, dlatinterp, indexing='ij')
        magout = interpn((np.arange(72), np.arange(30)), data, 
                        np.stack([lon_grid, lat_grid], axis=-1), 
                        method='linear', bounds_error=False, fill_value=0.0)
        
    elif magtype == 2:
        # GONG 360x180 magnetogram in FITS format
        if file is None:
            raise ValueError("pfss_mag_create: filename must be specified")
        
        # Read FITS file
        with fits.open(file) as hdul:
            data = hdul[0].data
        
        # Set up coordinate grids
        dlatix = np.arcsin(linrange(180, 89.5, -89.5) / 90.0) * 180 / np.pi
        dlonix = linrange(360, 1, 360)
        
        # Remap onto our grid
        dlatinterp = get_interpolation_index(np.flip(dlatix), lat)
        dloninterp = get_interpolation_index(dlonix, lon)
        
        lon_grid, lat_grid = np.meshgrid(dloninterp, dlatinterp, indexing='ij')
        magout = interpn((np.arange(360), np.arange(180)), data, 
                        np.stack([lon_grid, lat_grid], axis=-1), 
                        method='linear', bounds_error=False, fill_value=0.0)
        
    elif magtype == 3:
        # MDI synoptic charts
        if file is None:
            raise ValueError("pfss_mag_create: filename must be specified")
        
        # Read FITS file
        with fits.open(file) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header
        
        # Set out of bounds points to zero
        blank = hdr.get('BLANK', 0)
        mask = np.abs(data - blank) < 1e-4
        data[mask] = 0.0
        
        # Set up coordinate grids
        dlatix = np.arcsin(linrange(1080, 539.5, -539.5) / 540) * 180 / np.pi
        dlonix = linrange(3600, 0.1, 360)
        
        # Remap onto our grid
        dlatinterp = get_interpolation_index(np.flip(dlatix), lat)
        dloninterp = get_interpolation_index(dlonix, lon)
        
        lon_grid, lat_grid = np.meshgrid(dloninterp, dlatinterp, indexing='ij')
        magout = interpn((np.arange(3600), np.arange(1080)), data, 
                        np.stack([lon_grid, lat_grid], axis=-1), 
                        method='linear', bounds_error=False, fill_value=0.0)
        
    elif magtype == 4:
        # HMI synoptic charts
        if file is None:
            raise ValueError("pfss_mag_create: filename must be specified")
        
        # Read FITS file
        with fits.open(file) as hdul:
            data = hdul[0].data
        
        # Set out of bounds points to zero
        mask = data < -3e4
        data[mask] = 0.0
        
        # Set up coordinate grids
        dlatix = np.arcsin(linrange(1440, 719.5, -719.5) / 720) * 180 / np.pi
        dlonix = linrange(3600, 0.1, 360)
        
        # Remap onto our grid
        dlatinterp = get_interpolation_index(np.flip(dlatix), lat)
        dloninterp = get_interpolation_index(dlonix, lon)
        
        lon_grid, lat_grid = np.meshgrid(dloninterp, dlatinterp, indexing='ij')
        magout = interpn((np.arange(3600), np.arange(1440)), data, 
                        np.stack([lon_grid, lat_grid], axis=-1), 
                        method='linear', bounds_error=False, fill_value=0.0)
        
    else:
        raise ValueError("pfss_mag_create: invalid magtype")
    
    if not quiet:
        print("  pfss_mag_create: magnetogram created")
    
    return magout, lat, lon


# For compatibility with IDL calling convention  
def pfss_mag_create_idl(magtype, nlat=48, gridtype=2, file=None, quiet=False):
    """
    IDL-compatible wrapper for pfss_mag_create.
    
    This function updates the global data block with the created magnetogram.
    """
    magout, lat, lon = pfss_mag_create(magtype, nlat=nlat, gridtype=gridtype, 
                                      file=file, quiet=quiet)
    
    # Update the global data block
    data_block = PfssDataBlock()
    data_block.lat0 = lat
    data_block.lon0 = lon
    
    return magout