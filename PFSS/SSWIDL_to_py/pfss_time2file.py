"""
pfss_time2file.py - Convert any SSW time to PFSS save file (MDI extrap); NFS or URL

Purpose: any ssw time -> PFSS save file (MDI extrap); NFS or URL

Input Parameters:
    time0 - time desired, any SSW format (string, index record..)
    time1 - optional stop time for range

Output:
    function returns files or urls matching users input time/range

Keyword Parameters:
    urls - if set, return URLs 
    before - if set, match closest BEFORE (default is closest)
    after - if set, match closest AFTER   (default is closest)
    count (output) - number of files returned (zero implies error)
    ssw_catalog - if set, use $SSWDB/pfss/catalog 
    generate - if set (and permission and local), generate SSWDB cat
    version - version number ; default=2 (.h5)
    h5,hdf5 - switch (synonyms) - equivalent to version=2

Calling Sequence:
    mdisav = pfss_time2file(time [,before=True] [,after=True] )   
    mdisavs = pfss_time2file(start_time, end_time, [,before=True] [,after=True] )

History:
    5-December-2003 - S.L.Freeland - ssw/pfss integration helper
   13-April-2007    - S.L.Freeland - fixed COUNT output calculation
   10-Sep-2007 - S.L.F. - add SURFFIELD keyword & function
    7-Feb-2007 - S.L.F. - tweak local nfs parent path per Marc 
    8-Mar-2010 - S.L.F. - avoid common block hiccups due to 
                          multiple field-type toggles
    4-Oct-2012 - S.L.F - version 2 support = HDF5
    1-Jan-2013 - M.DeRosa - version 2 is now the default as of 1/1/2013
   11-Sep-2017 - M.DeRosa - updated catalog generation
    6-Jul-2018 - M.DeRosa - accommodates the fact that the database file is
                            now compressed
"""

import os
import tempfile
import re
import numpy as np
from datetime import datetime
from urllib.parse import urlparse
from urllib.request import urlretrieve
import h5py
import warnings
import glob


# Global variables to simulate IDL common blocks
_times = None
_links = None
_sswlinks = None
_lastbfieldcat = None
_lastftype = None
_lastversion = None


def pfss_time2file(time0, time1=None, urls=False, debug=False, before=False, 
                   after=False, refresh=False, ssw_catalog=False, generate=False,
                   loud=False, surffield=False, version=None, hdf5=False, h5=False):
    """
    Convert any SSW time to PFSS save file (MDI extrap); NFS or URL.
    
    Args:
        time0 (str or datetime): Time desired, any SSW format
        time1 (str or datetime, optional): Stop time for range
        urls (bool): If True, return URLs instead of local file paths
        debug (bool): Enable debug output
        before (bool): Match closest BEFORE (default is closest)
        after (bool): Match closest AFTER (default is closest)
        refresh (bool): Force refresh of file list
        ssw_catalog (bool): Use $SSWDB/pfss/catalog
        generate (bool): Generate SSWDB catalog (if authorized)
        loud (bool): Verbose output
        surffield (bool): Look for surface field files instead of Bfield
        version (int): Version number (1 or 2, default=2)
        hdf5 (bool): Use HDF5 format (equivalent to version=2)
        h5 (bool): Use HDF5 format (equivalent to version=2)
        
    Returns:
        str or list: Files or URLs matching input time/range
        Empty string if no matches found
    """
    
    global _times, _links, _sswlinks, _lastbfieldcat, _lastftype, _lastversion
    
    # Determine version
    if version is not None:
        ver = max(1, min(2, int(version)))
    elif hdf5 or h5:
        ver = 2
    else:
        ver = 2  # Default as of 1/1/2013
    
    # Initialize globals if needed
    if _lastftype is None:
        _lastftype = ''
    if _lastversion is None:
        _lastversion = -1
    if _lastbfieldcat is None:
        _lastbfieldcat = ''
    
    # Determine field type
    ftype = 'surffield' if surffield else 'Bfield'
    
    # Check if we need to refresh globals
    if _lastftype != ftype or _lastversion != ver:
        _times = None
        _links = None
        _sswlinks = None
        _lastbfieldcat = None
    
    _lastftype = ftype
    _lastversion = ver
    
    # Set up paths and parameters based on version
    run = '76' if ver == 2 else '48'
    ext = '.h5' if ver == 2 else '.sav'
    
    # Set up environment variables (these would normally come from environment)
    pfss_path = os.environ.get('PFSS_PATH', '')
    pfss_http = os.environ.get('PFSS_HTTP', '')
    
    fpat = f'kitrun0{run}' if surffield else ftype
    date_sub = ftype + '-bydate' + ('/hdf5' if ver == 2 else '')
    
    if not pfss_path:
        pfss_path = f'/archive/pfss/kitrun{run}/{date_sub}'
    
    if not pfss_http:
        pfss_http = f'http://www.lmsal.com/solarsoft{"" if ver == 1 else "/archive/ssw"}/pfss_links'
    
    # Set up catalog paths
    sswdb_path = os.environ.get('SSWDB', '/tmp')
    pfsscatdir = os.path.join(sswdb_path, 'packages', 'pfss', 'genxcat')
    
    vstring = '_v2' if ver == 2 else ''
    bfieldcatname = f'{ftype.lower()}{vstring}.geny'
    bfieldcat = os.path.join(pfsscatdir, bfieldcatname)
    bfieldurl = f'http://www.lmsal.com/solarsoft/pfss_genxcat/{bfieldcatname}'
    
    # Handle SSW catalog
    if ssw_catalog:
        if not os.path.exists(bfieldcat):
            if loud:
                print('No local catalog in SSWDB, trying remote http access..')
            
            temp_dir = tempfile.gettempdir()
            bfieldcat = os.path.join(temp_dir, bfieldcatname)
            
            try:
                urlretrieve(bfieldurl, bfieldcat)
                _lastbfieldcat = bfieldcat
                if loud:
                    print(f'Downloaded catalog to {bfieldcat}')
            except Exception as e:
                print(f'Problem with remote http access: {e}')
                return ''
        
        # Load catalog (this would need to be implemented based on the actual format)
        try:
            _times, _links, _sswlinks = load_catalog(bfieldcat)
            if loud:
                print(f'Loaded catalog with {len(_times)} entries')
        except Exception as e:
            print(f'Error loading catalog: {e}')
            return ''
    
    # Refresh file list if needed
    need_refresh = (os.path.exists(pfss_path) and 
                   (_links is None or refresh or generate))
    
    if need_refresh:
        if loud:
            print('Initializing file list')
        
        # Find files matching pattern
        try:
            pattern = os.path.join(pfss_path, f'*{fpat}*{ext}')
            files = glob.glob(pattern)
            
            if not files:
                print(f'No files found matching pattern: {pattern}')
                return ''
            
            _links = files
            
            # Extract times from filenames
            _times = []
            _sswlinks = []
            
            for file in files:
                filename = os.path.basename(file)
                # Extract time from filename (this is a simplified version)
                time_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', filename)
                if time_match:
                    time_str = time_match.group(1)
                    _times.append(parse_time(time_str))
                    # Create SSW-style link name
                    ssw_name = filename.replace('-', ' ').replace(':', ' ').replace('T', '_')
                    ssw_name = re.sub(r'\s+', '', ssw_name)
                    _sswlinks.append(ssw_name)
            
            _times = np.array(_times)
            
            if loud:
                print(f'Found {len(_links)} files')
                
        except Exception as e:
            print(f'Error initializing file list: {e}')
            return ''
    
    # Parse input times
    if time1 is None:
        # Single time
        t0 = parse_time(time0)
        t1 = t0
        is_range = False
    else:
        # Time range
        t0 = parse_time(time0)
        t1 = parse_time(time1)
        is_range = True
    
    # Find matching files
    if _times is None or len(_times) == 0:
        return ''
    
    if not is_range:
        # Single time - find closest match
        dtimes = _times - t0
        
        if before:
            # Find closest before
            before_mask = _times <= t0
            if np.any(before_mask):
                valid_indices = np.where(before_mask)[0]
                ss = valid_indices[np.argmax(_times[valid_indices])]
            else:
                return ''
        elif after:
            # Find closest after
            after_mask = _times >= t0
            if np.any(after_mask):
                valid_indices = np.where(after_mask)[0]
                ss = valid_indices[np.argmin(_times[valid_indices])]
            else:
                return ''
        else:
            # Find closest overall
            ss = np.argmin(np.abs(dtimes))
        
        selected_indices = [ss]
        
    else:
        # Time range
        selected_indices = np.where((_times >= t0) & (_times <= t1))[0]
    
    # Build return values
    if len(selected_indices) == 0:
        return ''
    
    if urls:
        # Return URLs
        url_base = pfss_http + ('_sfield' if surffield else '') + vstring + '/'
        retval = [url_base + _sswlinks[i] for i in selected_indices]
    else:
        # Return local file paths
        retval = [_links[i] for i in selected_indices]
    
    # Return single string if only one result, otherwise list
    if len(retval) == 1:
        return retval[0]
    else:
        return retval


def parse_time(time_input):
    """
    Parse time input into a comparable format.
    
    Args:
        time_input: Time in various formats (string, datetime, etc.)
        
    Returns:
        float: Time as a comparable number (e.g., timestamp)
    """
    
    if isinstance(time_input, str):
        # Try to parse common time formats
        formats = [
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%Y%m%d_%H%M%S',
            '%Y%m%d'
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(time_input, fmt)
                return dt.timestamp()
            except ValueError:
                continue
        
        # If no format matches, try basic parsing
        try:
            # Remove common separators and try again
            cleaned = re.sub(r'[T_-:]', '', time_input)
            if len(cleaned) >= 8:
                year = int(cleaned[:4])
                month = int(cleaned[4:6])
                day = int(cleaned[6:8])
                hour = int(cleaned[8:10]) if len(cleaned) > 8 else 0
                minute = int(cleaned[10:12]) if len(cleaned) > 10 else 0
                second = int(cleaned[12:14]) if len(cleaned) > 12 else 0
                dt = datetime(year, month, day, hour, minute, second)
                return dt.timestamp()
        except:
            pass
    
    elif isinstance(time_input, datetime):
        return time_input.timestamp()
    
    elif isinstance(time_input, (int, float)):
        return float(time_input)
    
    # Default: return current time
    return datetime.now().timestamp()


def load_catalog(catalog_path):
    """
    Load catalog file (placeholder implementation).
    
    Args:
        catalog_path (str): Path to catalog file
        
    Returns:
        tuple: (times, links, sswlinks)
    """
    
    # This is a placeholder implementation
    # In practice, this would need to read the actual catalog format
    warnings.warn("Catalog loading not fully implemented")
    
    # Return empty arrays
    return np.array([]), [], []


def create_synthetic_catalog():
    """
    Create a synthetic catalog for testing purposes.
    
    Returns:
        tuple: (times, links, sswlinks)
    """
    
    # Create synthetic data for demonstration
    base_time = datetime(2003, 4, 5, 0, 0, 0).timestamp()
    
    times = []
    links = []
    sswlinks = []
    
    # Create entries for several days
    for i in range(10):
        t = base_time + i * 24 * 3600  # Daily entries
        times.append(t)
        
        dt = datetime.fromtimestamp(t)
        date_str = dt.strftime('%Y-%m-%dT%H:%M:%S')
        filename = f'Bfield_{date_str}.h5'
        
        links.append(f'/archive/pfss/kitrun76/Bfield-bydate/hdf5/{filename}')
        sswlinks.append(filename.replace('-', ' ').replace(':', ' ').replace('T', '_'))
    
    return np.array(times), links, sswlinks


# For compatibility with IDL calling convention
def pfss_time2file_idl(time0, time1=None, **kwargs):
    """
    IDL-compatible wrapper for pfss_time2file.
    
    This function maintains the original interface.
    """
    return pfss_time2file(time0, time1, **kwargs)


# Utility functions
def get_temp_dir():
    """Get temporary directory."""
    return tempfile.gettempdir()


def file_exist(filepath):
    """Check if file exists."""
    return os.path.exists(filepath)


def concat_dir(dir1, dir2):
    """Concatenate directory paths."""
    return os.path.join(dir1, dir2)


def get_user():
    """Get current user."""
    return os.environ.get('USER', 'unknown')


def get_logenv(varname):
    """Get environment variable."""
    return os.environ.get(varname, '')


def box_message(message):
    """Print a box message."""
    if isinstance(message, list):
        for msg in message:
            print(f'  {msg}')
    else:
        print(f'  {message}')


def anytim(time_input, tai=False):
    """
    Convert time to internal format.
    
    Args:
        time_input: Time in various formats
        tai (bool): Return TAI time
        
    Returns:
        float: Time as timestamp
    """
    return parse_time(time_input)


def strextract(strings, start_pattern, end_pattern):
    """
    Extract substring between patterns.
    
    Args:
        strings (list): List of strings
        start_pattern (str): Start pattern
        end_pattern (str): End pattern
        
    Returns:
        list: Extracted substrings
    """
    results = []
    for s in strings:
        try:
            start_idx = s.find(start_pattern)
            if start_idx != -1:
                start_idx += len(start_pattern)
                end_idx = s.find(end_pattern, start_idx)
                if end_idx != -1:
                    results.append(s[start_idx:end_idx])
                else:
                    results.append('')
            else:
                results.append('')
        except:
            results.append('')
    
    return results


def str_replace(string, old, new):
    """Replace substring in string."""
    return string.replace(old, new)


def is_member(item, list_items, ignore_case=False):
    """Check if item is member of list."""
    if ignore_case:
        item = item.lower()
        list_items = [x.lower() for x in list_items]
    
    return item in list_items


def str2arr(string, delimiter=','):
    """Convert comma-separated string to array."""
    return string.split(delimiter)