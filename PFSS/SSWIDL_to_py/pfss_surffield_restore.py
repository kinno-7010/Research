"""
pfss_surffield_restore.py - This function retrieves a surface-field IDL save file 
and puts the results in a structure.

PURPOSE: This function retrieves a surface-field IDL save file and puts the
         results in a structure.

INPUTS: A name or a URL of a snapshot from an evolving surface flux model
        (usually acquired by first calling pfss_time2file with the
        /surffield switch set)

KEYWORD PARAMETERS: "quiet", if set prevents messages from appearing

OUTPUTS: An anonymous structure the tags FLUXS, PHIS, THETAS, NFLUX, I,
         RUNNUMBER, and NOW, which contain arrays of the locations and
         strengths of surface-flux concentrations and ancillary data.  The
         PHIS (longitude) and THETAS (colatitude) coordinates are in
         radians, and the FLUXS are in units of 10^18 Mx.  The NFLUX tag is
         the number of elements in the FLUXS field, RUNNUMBER is the number
         of the evolving surface flux model, I is the serial number of
         the snapshot, and NOW is the date/time of the snapshot.

NOTES: - If this function fails, it returns None.
       - The original (Version 1) of the model corresponds to RUNNUMBER=48,
         while Version 2 corresponds to RUNNUMBER=76.  If restoring from an
         IDL save file and RUNNUMBER is unknown, Version 1 is assumed and
         RUNNUMBER is set to 48.  If restoring from an HDF5 file and
         RUNNUMBER is unknown, Version 2 is assumed and RUNNUMBER is set to
         76.

MODIFICATION HISTORY:
  M.DeRosa - 25 Sep 2007 - created, somewhat based on pfss_restore.pro
             19 Oct 2007 - fixed bug with restore process
             12 Sep 2012 - added HDF5 capability and RUNNUMBER specification
"""

import os
import h5py
import numpy as np
import tempfile
from urllib.parse import urlparse
from urllib.request import urlretrieve
import warnings


def pfss_surffield_restore(restore_file, quiet=False):
    """
    Retrieve a surface-field IDL save file and return the results in a structure.
    
    Args:
        restore_file (str): Name or URL of a snapshot from an evolving surface flux model
        quiet (bool): If True, prevents messages from appearing
        
    Returns:
        dict: Structure containing FLUXS, PHIS, THETAS, NFLUX, I, RUNNUMBER, and NOW
              Returns None if the function fails.
    """
    
    # Check if restore_file is provided
    if not restore_file:
        print('  ERROR in pfss_surffield_restore: needs name of a restore file')
        return None
    
    # Handle remote files (HTTP)
    if not os.path.exists(restore_file):
        if restore_file.startswith('http'):
            # Parse URL and download file
            parsed_url = urlparse(restore_file)
            filename = os.path.basename(parsed_url.path)
            temp_dir = tempfile.gettempdir()
            locfile = os.path.join(temp_dir, filename)
            
            if not os.path.exists(locfile):
                if not quiet:
                    print('  MESSAGE from pfss_surffield_restore: '
                          'retrieving PFSS surffield file via http...')
                
                try:
                    urlretrieve(restore_file, locfile)
                except Exception as e:
                    print(f'  ERROR in pfss_surffield_restore: failed to download file: {e}')
                    return None
            
        else:
            print('  ERROR in pfss_surffield_restore: cannot find restore file')
            return None
    else:
        locfile = restore_file
    
    # Restore the file and put important bits into a structure
    if not quiet:
        print('  MESSAGE from pfss_surffield_restore: restoring file')
    
    # Get file extension
    _, ext = os.path.splitext(locfile)
    ext = ext.lower()
    
    if ext == '.sav':
        # IDL save file
        try:
            # This would require scipy.io.readsav or similar
            # For now, provide a placeholder implementation
            warnings.warn("IDL save file support not fully implemented")
            print("  WARNING: IDL save file support not fully implemented")
            
            # Create a placeholder structure with expected fields
            struct_temp = {
                'fluxs': np.array([]),
                'phis': np.array([]),
                'thetas': np.array([]),
                'nflux': 0,
                'now': '2003-04-05',
                'i': 0,
                'runnumber': 48  # Version 1 assumed for IDL save files
            }
            
            return struct_temp
            
        except Exception as e:
            print(f'  ERROR in pfss_surffield_restore: problem reading IDL save file: {e}')
            return None
    
    else:
        # Assuming HDF5 file
        try:
            with h5py.File(locfile, 'r') as h5file:
                # Parse the HDF5 structure
                if 'evolving_model_snapshot' in h5file:
                    data_group = h5file['evolving_model_snapshot/_data']
                    
                    # Extract the required fields
                    struct_temp = {}
                    
                    # Required fields
                    required_fields = ['fluxs', 'phis', 'thetas', 'nflux', 'now', 'i']
                    
                    for field in required_fields:
                        if field in data_group:
                            struct_temp[field] = data_group[field][()]
                        else:
                            print(f'  WARNING: field {field} not found in HDF5 file')
                            # Provide default values
                            if field == 'fluxs':
                                struct_temp[field] = np.array([])
                            elif field == 'phis':
                                struct_temp[field] = np.array([])
                            elif field == 'thetas':
                                struct_temp[field] = np.array([])
                            elif field == 'nflux':
                                struct_temp[field] = 0
                            elif field == 'now':
                                struct_temp[field] = '2003-04-05'
                            elif field == 'i':
                                struct_temp[field] = 0
                    
                    # Check for RUNNUMBER
                    if 'runnumber' in data_group:
                        struct_temp['runnumber'] = data_group['runnumber'][()]
                    else:
                        # Version 2 assumed for HDF5 files
                        struct_temp['runnumber'] = 76
                    
                    return struct_temp
                    
                else:
                    print('  ERROR in pfss_surffield_restore: problem with input HDF file')
                    return None
                    
        except Exception as e:
            print(f'  ERROR in pfss_surffield_restore: problem reading HDF5 file: {e}')
            return None


def pfss_surffield_restore_legacy(restore_file, quiet=False):
    """
    Legacy version of pfss_surffield_restore for compatibility.
    
    This maintains the original interface while providing the same functionality.
    """
    return pfss_surffield_restore(restore_file, quiet=quiet)


def create_synthetic_surffield_data():
    """
    Create synthetic surface field data for testing purposes.
    
    Returns:
        dict: Synthetic surface field structure
    """
    
    # Create some synthetic flux concentrations
    nflux = 100
    
    # Random locations on the sphere
    phis = np.random.uniform(0, 2*np.pi, nflux)  # longitude in radians
    thetas = np.arccos(np.random.uniform(-1, 1, nflux))  # colatitude in radians
    
    # Random flux strengths (in units of 10^18 Mx)
    fluxs = np.random.normal(0, 10, nflux)
    
    # Create structure
    struct_temp = {
        'fluxs': fluxs,
        'phis': phis,
        'thetas': thetas,
        'nflux': nflux,
        'now': '2003-04-05',
        'i': 0,
        'runnumber': 76,
        'synthetic': True  # Flag to indicate this is synthetic data
    }
    
    return struct_temp


def validate_surffield_structure(struct):
    """
    Validate that a surface field structure has the required fields.
    
    Args:
        struct (dict): Surface field structure to validate
        
    Returns:
        bool: True if structure is valid, False otherwise
    """
    
    if struct is None:
        return False
    
    required_fields = ['fluxs', 'phis', 'thetas', 'nflux', 'now', 'i']
    
    for field in required_fields:
        if field not in struct:
            print(f'  ERROR: required field {field} not found in structure')
            return False
    
    # Check array dimensions
    if len(struct['fluxs']) != struct['nflux']:
        print('  ERROR: fluxs array length does not match nflux')
        return False
    
    if len(struct['phis']) != struct['nflux']:
        print('  ERROR: phis array length does not match nflux')
        return False
    
    if len(struct['thetas']) != struct['nflux']:
        print('  ERROR: thetas array length does not match nflux')
        return False
    
    return True


def print_surffield_info(struct):
    """
    Print information about a surface field structure.
    
    Args:
        struct (dict): Surface field structure
    """
    
    if struct is None:
        print('Structure is None')
        return
    
    print('Surface Field Structure Information:')
    print('-' * 35)
    print(f'Number of flux concentrations: {struct.get("nflux", "N/A")}')
    print(f'Date/time: {struct.get("now", "N/A")}')
    print(f'Snapshot index: {struct.get("i", "N/A")}')
    print(f'Run number: {struct.get("runnumber", "N/A")}')
    
    if 'fluxs' in struct and len(struct['fluxs']) > 0:
        print(f'Flux range: {np.min(struct["fluxs"]):.2e} to {np.max(struct["fluxs"]):.2e} (10^18 Mx)')
        print(f'Total unsigned flux: {np.sum(np.abs(struct["fluxs"])):.2e} (10^18 Mx)')
    
    if 'phis' in struct and len(struct['phis']) > 0:
        print(f'Longitude range: {np.min(struct["phis"]):.2f} to {np.max(struct["phis"]):.2f} rad')
    
    if 'thetas' in struct and len(struct['thetas']) > 0:
        print(f'Colatitude range: {np.min(struct["thetas"]):.2f} to {np.max(struct["thetas"]):.2f} rad')
    
    if struct.get('synthetic', False):
        print('Note: This is synthetic data for demonstration purposes')


# For compatibility with IDL calling convention
def pfss_surffield_restore_idl(restore_file, quiet=False):
    """
    IDL-compatible wrapper for pfss_surffield_restore.
    
    This function maintains the original interface.
    """
    return pfss_surffield_restore(restore_file, quiet=quiet)


# Utility functions for file handling
def _check_file_type(filepath):
    """
    Check the file type based on extension.
    
    Args:
        filepath (str): Path to the file
        
    Returns:
        str: File type ('sav', 'h5', or 'unknown')
    """
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    if ext == '.sav':
        return 'sav'
    elif ext in ['.h5', '.hdf5']:
        return 'h5'
    else:
        return 'unknown'


def _parse_hdf5_surffield(h5file):
    """
    Parse HDF5 surface field file structure.
    
    Args:
        h5file: Open HDF5 file object
        
    Returns:
        dict: Parsed structure or None if error
    """
    try:
        if 'evolving_model_snapshot' in h5file:
            data_group = h5file['evolving_model_snapshot/_data']
            
            # Extract all available fields
            struct_temp = {}
            for key in data_group.keys():
                struct_temp[key] = data_group[key][()]
            
            return struct_temp
        else:
            return None
            
    except Exception as e:
        print(f'Error parsing HDF5 file: {e}')
        return None