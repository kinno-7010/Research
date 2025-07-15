"""
pfss_to_wsa.py - Convert LMSAL assimilation model snapshot to WSA-compatible FITS format

PURPOSE:
   Given a LMSAL assimilation model snapshot, such as the structure returned
   from pfss_surffield_restore, this routine will write out a FITS file in a
   format that can be read by the WSA model.

CALLING SEQUENCE:
   pfss_to_wsa(sfield, outdir=outdir, xsiz=xsiz, ysiz=ysiz)

INPUTS:
   sfield = output from pfss_surffield_restore.py, or another compatible
            structure, having the tags FLUXS, PHIS, THETAS, NFLUX, I,
            RUNNUMBER, and NOW, which contain arrays of the locations and
            strengths of surface-flux concentrations and ancillary data.

KEYWORD PARAMETERS:
   outdir = directory in which to save file (default is current directory)
   xsiz = number of gridpoints in longitude (default=360)
   ysiz = number of gridpoints in latitude (default=180)

OUTPUTS:
   One FITS file that can be read in by the WSA pipeline

NOTES: -The output FITS files is based on file spec from Shaela Jones 
       -The output filename is based on date of model.
       -Essentially, the routine takes the list of flux elements and assigns
        them to pixels in a synoptic map.
       -Because the LMSAL assimilation model maps are snapshots at a   
        point in time, so:
        CRROTEDG = Carr rot number of central meridian, as viewed from Earth
        MAPCR = Carr rot number of central meridian, as viewed from Earth
        MODELDA = not sure what to put here (-1 is used as a dummy value)
        MODELVER = not sure what to put here (-1 is used as a dummy value)

EXAMPLE:
   date = '2014-04-08'  # sample date
   fname = pfss_time2file(date, ssw_catalog=True, urls=True, surffield=True)
   sfield = pfss_surffield_restore(fname[0])
   pfss_to_wsa(sfield)

MODIFICATION HISTORY:
   M.DeRosa - 5 Apr 2021 - created
"""

import os
import numpy as np
from astropy.io import fits
from astropy.time import Time
from datetime import datetime
import re


def pfss_to_wsa(sfield, outdir=None, xsiz=360, ysiz=180):
    """
    Convert LMSAL assimilation model snapshot to WSA-compatible FITS format.
    
    Args:
        sfield (dict): Surface field structure from pfss_surffield_restore
        outdir (str): Directory to save output file (default: current directory)
        xsiz (int): Number of gridpoints in longitude (default: 360)
        ysiz (int): Number of gridpoints in latitude (default: 180)
    """
    
    # Validate input structure
    if not validate_sfield_structure(sfield):
        raise ValueError("Invalid sfield structure")
    
    # Parameters
    if xsiz is None:
        xsiz = 360  # number of elements in longitude in map
    if ysiz is None:
        ysiz = 180  # number of elements in latitude in map
    
    # WCS coordinate parameters
    crval1 = 180.0        # coordinate of reference pixel in x
    crval2 = 0.0          # coordinate of reference pixel in y
    cunit1 = 'deg     '   # unit of x coordinate
    cunit2 = 'deg     '   # unit of y coordinate
    ctype1 = 'CRLN-CAR'   # axis label for x
    ctype2 = 'CRLT-CAR'   # axis label for y
    
    # Define map projection and WCS keywords
    crpix1 = 0.5 * xsiz + 0.5  # reference pixel in x
    crpix2 = 0.5 * ysiz + 0.5  # reference pixel in y
    cdelt1 = 360.0 / xsiz      # delta x
    cdelt2 = 180.0 / ysiz      # delta y
    
    # Create carrington map based on above WCS projection
    synoptic = np.zeros((xsiz, ysiz), dtype=np.float32)
    
    # Convert coordinates to pixel indices
    xix = (np.fix(sfield['phis'] * 180 / np.pi / cdelt1) % xsiz).astype(int)
    yix = (np.fix((180 - sfield['thetas'] * 180 / np.pi) / cdelt2) % ysiz).astype(int)
    
    # Accumulate flux values in the synoptic map
    for ii in range(sfield['nflux']):
        synoptic[xix[ii], yix[ii]] += sfield['fluxs'][ii]
    
    # Create FITS header
    header = fits.Header()
    header['CRPIX1'] = crpix1
    header['CRPIX2'] = crpix2
    header['CRVAL1'] = (crval1, 'center of map: Carrington long = 180 deg')
    header['CRVAL2'] = (crval2, 'center of map: heliographic lat = equator')
    header['CDELT1'] = cdelt1
    header['CDELT2'] = cdelt2
    header['CUNIT1'] = cunit1
    header['CUNIT2'] = cunit2
    header['CTYPE1'] = ctype1
    header['CTYPE2'] = ctype2
    
    # Calculate Carrington rotation number
    try:
        # Convert time to Carrington rotation number
        crrot = time_to_carr(sfield['now'])
        header['CRROTEDG'] = int(crrot)
        header['MAPCR'] = crrot
    except:
        # Use dummy values if conversion fails
        header['CRROTEDG'] = -1
        header['MAPCR'] = -1.0
    
    header['CRLNGEDG'] = 0.0
    header['LATTYPE'] = 0  # assume 0=latitude and 1=colatitude
    header['MAPDATA'] = 'HMI     '
    header['GRID'] = cdelt1  # assume cdelt1=cdelt2
    
    # Time-related keywords
    try:
        # Convert time to Julian day
        time_obj = parse_time_string(sfield['now'])
        julian_day = time_obj.jd
        header['MAPJUL'] = julian_day
        header['MAPTIME'] = time_obj.strftime('%Y-%m-%dT%H:%M:%S')
    except:
        header['MAPJUL'] = -1
        header['MAPTIME'] = 'unknown'
    
    header['MODEL'] = ('LMSALsft', 'LMSAL surface flux transport model')
    header['MODELDA'] = -1      # dummy value
    header['MODELVER'] = -1.0   # dummy value
    
    # Create output filename
    outfile = create_output_filename(sfield['now'])
    if outdir is not None:
        outfile = os.path.join(outdir, outfile)
    
    # Write out FITS file
    hdu = fits.PrimaryHDU(synoptic, header=header)
    hdu.writeto(outfile, overwrite=True)
    
    print(f'  pfss_to_wsa: wrote {outfile}')
    
    return outfile


def validate_sfield_structure(sfield):
    """
    Validate that the surface field structure has required fields.
    
    Args:
        sfield (dict): Surface field structure
        
    Returns:
        bool: True if valid, False otherwise
    """
    
    if sfield is None:
        return False
    
    required_fields = ['fluxs', 'phis', 'thetas', 'nflux', 'now']
    
    for field in required_fields:
        if field not in sfield:
            print(f'Error: Required field {field} not found in sfield structure')
            return False
    
    # Check array dimensions
    if len(sfield['fluxs']) != sfield['nflux']:
        print('Error: fluxs array length does not match nflux')
        return False
    
    if len(sfield['phis']) != sfield['nflux']:
        print('Error: phis array length does not match nflux')
        return False
    
    if len(sfield['thetas']) != sfield['nflux']:
        print('Error: thetas array length does not match nflux')
        return False
    
    return True


def time_to_carr(time_string):
    """
    Convert time string to Carrington rotation number.
    
    Args:
        time_string (str): Time string in various formats
        
    Returns:
        float: Carrington rotation number
    """
    
    try:
        # Parse the time string
        time_obj = parse_time_string(time_string)
        
        # Convert to Carrington rotation number
        # Reference: Carrington rotation 1 started on 1853-11-09 14:20:00
        # Period is approximately 27.2753 days
        carr_epoch = datetime(1853, 11, 9, 14, 20, 0)
        carr_period = 27.2753  # days
        
        # Calculate time difference in days
        time_diff = (time_obj - carr_epoch).total_seconds() / (24 * 3600)
        
        # Calculate Carrington rotation number
        carr_rot = time_diff / carr_period + 1
        
        return carr_rot
        
    except Exception as e:
        print(f'Warning: Could not convert time to Carrington rotation: {e}')
        return -1.0


def parse_time_string(time_string):
    """
    Parse time string in various formats.
    
    Args:
        time_string (str): Time string
        
    Returns:
        datetime: Parsed datetime object
    """
    
    # Try various time formats
    formats = [
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%Y%m%d_%H%M%S',
        '%Y%m%d',
        '%d-%b-%Y %H:%M:%S',
        '%d-%b-%Y'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(time_string, fmt)
        except ValueError:
            continue
    
    # If no format works, try to extract date components
    try:
        # Look for YYYY-MM-DD pattern
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', time_string)
        if match:
            year, month, day = map(int, match.groups())
            return datetime(year, month, day)
    except:
        pass
    
    # Default to epoch if all else fails
    return datetime(1970, 1, 1)


def create_output_filename(time_string):
    """
    Create output filename based on time string.
    
    Args:
        time_string (str): Time string
        
    Returns:
        str: Output filename
    """
    
    try:
        # Parse the time
        time_obj = parse_time_string(time_string)
        
        # Create filename in format: LMSALassim_YYYYMMDD_HHMMSS.fits
        filename = f"LMSALassim_{time_obj.strftime('%Y%m%d_%H%M%S')}.fits"
        
        return filename
        
    except:
        # Create a generic filename if parsing fails
        # Clean up the time string by removing problematic characters
        cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', str(time_string))
        return f"LMSALassim_{cleaned}.fits"


def print_synoptic_info(synoptic):
    """
    Print information about the synoptic map.
    
    Args:
        synoptic (numpy.ndarray): Synoptic map
    """
    
    print('Synoptic Map Information:')
    print('-' * 25)
    print(f'Dimensions: {synoptic.shape}')
    print(f'Total flux: {np.sum(synoptic):.2e}')
    print(f'Positive flux: {np.sum(synoptic[synoptic > 0]):.2e}')
    print(f'Negative flux: {np.sum(synoptic[synoptic < 0]):.2e}')
    print(f'Min value: {np.min(synoptic):.2e}')
    print(f'Max value: {np.max(synoptic):.2e}')
    print(f'Non-zero pixels: {np.count_nonzero(synoptic)}')


def plot_synoptic_map(synoptic, title='Synoptic Map'):
    """
    Plot the synoptic map using matplotlib.
    
    Args:
        synoptic (numpy.ndarray): Synoptic map
        title (str): Plot title
    """
    
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create coordinate grids
        xsiz, ysiz = synoptic.shape
        lon = np.linspace(0, 360, xsiz)
        lat = np.linspace(-90, 90, ysiz)
        
        # Plot the map
        im = ax.imshow(synoptic.T, extent=[0, 360, -90, 90], 
                      origin='lower', aspect='auto', cmap='RdBu_r')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Magnetic Flux Density')
        
        # Set labels and title
        ax.set_xlabel('Carrington Longitude (degrees)')
        ax.set_ylabel('Latitude (degrees)')
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")


# For compatibility with IDL calling convention
def pfss_to_wsa_idl(sfield, outdir=None, xsiz=360, ysiz=180):
    """
    IDL-compatible wrapper for pfss_to_wsa.
    
    This function maintains the original interface.
    """
    return pfss_to_wsa(sfield, outdir=outdir, xsiz=xsiz, ysiz=ysiz)


# Example usage function
def pfss_to_wsa_example():
    """
    Example usage of pfss_to_wsa function.
    """
    
    print("PFSS to WSA Example")
    print("=" * 18)
    
    # This would normally use real data
    print("Note: This example uses synthetic data for demonstration")
    
    # Create synthetic surface field data
    sfield = create_synthetic_sfield()
    
    # Convert to WSA format
    outfile = pfss_to_wsa(sfield)
    
    print(f"Created WSA-compatible FITS file: {outfile}")
    
    # Optionally read back and verify
    try:
        with fits.open(outfile) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            
            print(f"File successfully created with shape: {data.shape}")
            print(f"Header contains {len(header)} keywords")
            
            # Print some diagnostic info
            print_synoptic_info(data)
            
    except Exception as e:
        print(f"Error reading back file: {e}")


def create_synthetic_sfield():
    """
    Create synthetic surface field data for testing.
    
    Returns:
        dict: Synthetic surface field structure
    """
    
    nflux = 1000
    
    # Random locations on the sphere
    phis = np.random.uniform(0, 2*np.pi, nflux)     # longitude in radians
    thetas = np.arccos(np.random.uniform(-1, 1, nflux))  # colatitude in radians
    
    # Random flux strengths
    fluxs = np.random.normal(0, 10, nflux)
    
    # Create structure
    sfield = {
        'fluxs': fluxs,
        'phis': phis,
        'thetas': thetas,
        'nflux': nflux,
        'now': '2003-04-05T12:00:00',
        'i': 0,
        'runnumber': 76
    }
    
    return sfield


if __name__ == "__main__":
    pfss_to_wsa_example()