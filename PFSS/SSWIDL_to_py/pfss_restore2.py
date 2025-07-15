"""
pfss_restore2.py - Modified version of pfss_restore with debugging additions

This is a variant of pfss_restore.py with slight modifications for debugging purposes.
The main differences from pfss_restore.py are:
1. locfile is initialized to empty string
2. A debug stop point is added (converted to optional debug print)

Purpose: routine to restore input pfss 'save' file and update pfss common

Input Parameters:
    restfile - the pfss file to restore - (.sav or .h5)

Output Parameters:
    NONE - (all output via common block)

Keyword Parameters:
    refresh - if set, force restore even if same as last restored file 
    bfield_struct - optional output HDF5 pfss structure
    loud - if set, more details
    debug - if set, print debug information

Side Effects:
    Updates pfss common block: pfss_data_block

History:
    5-Dec-2004 - S.L.Freeland
   26-Jan-2004 - S.L.Freeland - add /RELAX to 
   25-sep-2006 - S.L.Freeland - use pfss_data_block.pro for common define
   11-oct-2012 - S.L.Freeland - allow pfss Version 2 / HDF5 format
"""

import os
import h5py
import numpy as np
import pickle
import warnings
from urllib.parse import urlparse
from urllib.request import urlretrieve
import tempfile

# Import required modules from the converted package
from pfss_data_block import PfssDataBlock


# Global variable to track last restored file
_last_restored_file = None


def pfss_restore2(restfile, refresh=False, loud=False, debug=False, bfield_struct=None):
    """
    Restore input pfss 'save' file and update pfss common block (version 2).
    
    This is a variant of pfss_restore with slight modifications for debugging.
    
    Args:
        restfile (str): The pfss file to restore (.sav or .h5)
        refresh (bool): If True, force restore even if same as last restored file
        loud (bool): If True, print more details
        debug (bool): If True, print debug information
        bfield_struct (dict): Optional output HDF5 pfss structure
        
    Returns:
        dict: HDF5 structure if applicable, None otherwise
    """
    
    global _last_restored_file
    
    # Check if restfile is a string
    if not isinstance(restfile, str):
        print('Need an input pfss (idl) save file name...., bailing')
        return None
    
    # Initialize last restored file if needed
    if _last_restored_file is None:
        _last_restored_file = ''
    
    # Check if we need to restore
    restoreit = refresh or restfile != _last_restored_file
    
    # Handle remote files (HTTP)
    if restfile.startswith('http') and restoreit:
        if not os.path.exists(restfile):
            parsed_url = urlparse(restfile)
            filename = os.path.basename(parsed_url.path)
            temp_dir = tempfile.gettempdir()
            local_file = os.path.join(temp_dir, filename)
            
            # Initialize local_file to empty string (difference from pfss_restore)
            local_file = ''
            
            if not os.path.exists(local_file):
                if loud:
                    print('Retrieving pfss Bfield file via http...')
                try:
                    urlretrieve(restfile, local_file)
                    # Debug stop point (converted to optional debug print)
                    if debug:
                        print(f'DEBUG: restfile[0] = {restfile}')
                    restfile = local_file
                except Exception as e:
                    print(f"Error downloading file: {e}")
                    return None
            else:
                if loud:
                    print(f'File> {local_file} already local...')
                restfile = local_file
    
    # Check if file exists
    if not os.path.exists(restfile) and restoreit:
        print(f'Cannot find restore file> {restfile}, bailing..')
        return None
    
    # Get the data block
    data_block = PfssDataBlock()
    
    if restoreit:
        if loud:
            print(f'Restoring file>> {restfile}')
            print('...Please be patient')
        
        # Get file extension
        _, extension = os.path.splitext(restfile)
        extension = extension.lower()
        
        if extension == '.sav':
            # IDL save file - would need scipy.io.readsav or similar
            try:
                # This is a placeholder - actual implementation would depend on
                # the specific format of the IDL save file
                warnings.warn("IDL save file support not fully implemented")
                print("IDL save file support not fully implemented")
                return None
            except Exception as e:
                print(f"Error reading IDL save file: {e}")
                return None
                
        elif extension == '.h5':
            # HDF5 file
            try:
                with h5py.File(restfile, 'r') as h5file:
                    # Parse the HDF5 structure
                    if 'ssw_pfss_extrapolation' not in h5file:
                        print('Unexpected HDF5 output, bailing...')
                        return None
                    
                    # Create bfield_struct dictionary
                    bfield_struct = {}
                    
                    # Get the main data structure
                    pfss_data = h5file['ssw_pfss_extrapolation']
                    
                    # Extract common block variables
                    # This is a simplified version - actual implementation would
                    # need to match the specific HDF5 structure
                    if '_data' in pfss_data:
                        data_group = pfss_data['_data']
                        
                        # Extract key variables into the data block
                        for key in data_group.keys():
                            if loud:
                                print(f'{key} >> found')
                            
                            # Map HDF5 variables to data block attributes
                            value = data_group[key][()]
                            if hasattr(data_block, key):
                                setattr(data_block, key, value)
                            else:
                                # Store in a general dictionary for unknown variables
                                if not hasattr(data_block, 'extra_vars'):
                                    data_block.extra_vars = {}
                                data_block.extra_vars[key] = value
                        
                        # Handle special cases
                        if 'model_date' in data_group:
                            data_block.now = data_group['model_date'][()]
                        
                        # Handle complex variables
                        if all(k in data_group for k in ['phiat_re', 'phiat_im', 'phibt_re', 'phibt_im']):
                            data_block.phiat = (data_group['phiat_re'][()] + 
                                              1j * data_group['phiat_im'][()])
                            data_block.phibt = (data_group['phibt_re'][()] + 
                                              1j * data_group['phibt_im'][()])
                        
                        # Store the structure for output
                        bfield_struct['ssw_pfss_extrapolation'] = {
                            '_data': {k: v[()] for k, v in data_group.items()}
                        }
                        
            except Exception as e:
                print(f"Error reading HDF5 file: {e}")
                return None
                
        else:
            print(f'Unknown file extension>> {extension}')
            return None
        
        _last_restored_file = restfile
        if loud:
            print('Done with restore...')
    
    elif restfile == _last_restored_file and loud:
        print('Same file as last time so not re-restored')
        print('Use refresh=True to force re-restore')
    
    return bfield_struct


def pfss_restore2_legacy(restfile, refresh=False, loud=False, debug=False):
    """
    Legacy version of pfss_restore2 for compatibility.
    
    This maintains the original interface while providing the same functionality.
    """
    return pfss_restore2(restfile, refresh=refresh, loud=loud, debug=debug)


# For compatibility with IDL calling convention
def pfss_restore2_idl(restfile, refresh=False, loud=False, debug=False):
    """
    IDL-compatible wrapper for pfss_restore2.
    
    This function updates the global data block and doesn't return anything.
    """
    pfss_restore2(restfile, refresh=refresh, loud=loud, debug=debug)


# Utility functions for file handling (same as pfss_restore.py)
def _check_file_exists(filepath):
    """Check if a file exists, handling both local and remote files."""
    if filepath.startswith('http'):
        # For remote files, we'll try to download them
        return True
    else:
        return os.path.exists(filepath)


def _download_file(url, local_path):
    """Download a file from a URL to a local path."""
    try:
        urlretrieve(url, local_path)
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False


def _get_file_extension(filepath):
    """Get the file extension from a filepath."""
    return os.path.splitext(filepath)[1].lower()


def _parse_hdf5_structure(h5file):
    """Parse HDF5 file structure into a dictionary."""
    def _parse_group(group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                result[key] = _parse_group(item)
            else:
                result[key] = item[()]
        return result
    
    return _parse_group(h5file)