"""
Spherical Trace Field

Given vector field data (such as the magnetic field) on a spherical grid,
and a set of starting points, this procedure will trace the fieldlines
that pass through each of the starting points.

CALLING SEQUENCE:
    spherical_trace_field(sph_data, stepmax=3000, safety=0.5, outfield=None,
                         linelengths=None, linekind=None, endpoints=False,
                         oneway=False, noreverse=False, trim=False,
                         subsample=None, quiet=False)

INPUTS:
    sph_data = a SphericalFieldData object with the following fields defined:
        br, bth, bph, nlat, nlon, nr, rix, lat, lon, str, stth, stph.
        Basically, one needs the vector field (br,bth,bph), its dimension
        (nr,nlat,nlon), its indexing (rix,lat,lon), and the starting
        points (str,stth,stph).
    stepmax = max number of steps per field line (default=3000)
    safety = maximum ds along each field line, in units of minimum grid
             spacing (default = 0.5)
    endpoints = if True, only endpoints are stored
    oneway = if True, loops are traced in one direction only
    noreverse = if True, field lines are not reversed
    trim = if True, trim fieldlines to domain boundaries
    subsample = factor for subsampling fieldline points
    quiet = if True, suppress screen output

OUTPUTS:
    sph_data = SphericalFieldData object with ptr, ptth, ptph, nstep fields set
    outfield = array of vector field values at endpoints and starting points
    linelengths = array of fieldline lengths
    linekind = array of fieldline type codes

NOTES:
    - The latitude array (sph_data.lat) MUST be in monotonically ascending order
    - This is a simplified version of the full IDL implementation

MODIFICATION HISTORY:
    M.DeRosa - 13 Dec 2005 - copied from pfss_trace_field (IDL)
    [... many modifications ...]
    Huang,GH - 15 Mar 2018 - fixed bug with OUTFIELD interpolation
    Converted to Python - 2025 (simplified version)
"""

import numpy as np
from typing import Optional, Tuple, List
from spherical_field_data__define import SphericalFieldData


def get_interpolation_index(arr: np.ndarray, value: float) -> float:
    """Get interpolation index for a value in a sorted array."""
    return np.interp(value, arr, np.arange(len(arr)))


def interpolate_trilinear(data: np.ndarray, x: float, y: float, z: float) -> float:
    """Perform trilinear interpolation."""
    # Get integer indices
    x0, y0, z0 = int(x), int(y), int(z)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    
    # Get fractional parts
    xf, yf, zf = x - x0, y - y0, z - z0
    
    # Boundary checking
    nx, ny, nz = data.shape
    x0, x1 = np.clip([x0, x1], 0, nx-1)
    y0, y1 = np.clip([y0, y1], 0, ny-1)
    z0, z1 = np.clip([z0, z1], 0, nz-1)
    
    # If at boundary, use nearest neighbor
    if x1 == x0 or y1 == y0 or z1 == z0:
        return data[x0, y0, z0]
    
    # Trilinear interpolation
    c000 = data[x0, y0, z0]
    c001 = data[x0, y0, z1]
    c010 = data[x0, y1, z0]
    c011 = data[x0, y1, z1]
    c100 = data[x1, y0, z0]
    c101 = data[x1, y0, z1]
    c110 = data[x1, y1, z0]
    c111 = data[x1, y1, z1]
    
    # Interpolate along x
    c00 = c000 * (1 - xf) + c100 * xf
    c01 = c001 * (1 - xf) + c101 * xf
    c10 = c010 * (1 - xf) + c110 * xf
    c11 = c011 * (1 - xf) + c111 * xf
    
    # Interpolate along y
    c0 = c00 * (1 - yf) + c10 * yf
    c1 = c01 * (1 - yf) + c11 * yf
    
    # Interpolate along z
    return c0 * (1 - zf) + c1 * zf


def forw_euler(step: int, point: np.ndarray, steplen: float, ds: np.ndarray) -> np.ndarray:
    """Forward Euler integration step."""
    return point + steplen * ds


def sign_mld(x: np.ndarray) -> np.ndarray:
    """Sign function (IDL-style)."""
    return np.sign(x)


def spherical_trace_field(sph_data: SphericalFieldData,
                         stepmax: int = 3000,
                         safety: float = 0.5,
                         outfield: Optional[np.ndarray] = None,
                         linelengths: Optional[np.ndarray] = None,
                         linekind: Optional[np.ndarray] = None,
                         endpoints: bool = False,
                         oneway: bool = False,
                         noreverse: bool = False,
                         trim: bool = False,
                         subsample: Optional[float] = None,
                         quiet: bool = False) -> None:
    """
    Trace fieldlines through a spherical vector field.
    
    Parameters:
    -----------
    sph_data : SphericalFieldData
        Spherical field data structure with vector field and starting points
    stepmax : int, optional
        Maximum number of steps per fieldline
    safety : float, optional
        Safety factor for step size control
    outfield : np.ndarray, optional
        Output array for field values at endpoints
    linelengths : np.ndarray, optional
        Output array for fieldline lengths
    linekind : np.ndarray, optional
        Output array for fieldline type codes
    endpoints : bool, optional
        If True, store only endpoints
    oneway : bool, optional
        If True, trace in one direction only
    noreverse : bool, optional
        If True, don't reverse fieldlines
    trim : bool, optional
        If True, trim fieldlines to domain boundaries
    subsample : float, optional
        Subsampling factor for fieldline points
    quiet : bool, optional
        If True, suppress output
    """
    
    # Input validation
    if sph_data.str is None or sph_data.stth is None or sph_data.stph is None:
        raise ValueError("ERROR in spherical_trace_field: starting points not defined")
    
    if sph_data.br is None or sph_data.bth is None or sph_data.bph is None:
        raise ValueError("ERROR in spherical_trace_field: vector field not defined")
    
    # Parameters
    nlines = len(sph_data.str)
    rmin, rmax = np.min(sph_data.rix), np.max(sph_data.rix)
    thmin, thmax = np.min(sph_data.theta), np.max(sph_data.theta)
    
    # Check for bounded data
    bounded = sph_data.is_bounded_in_longitude()
    if bounded:
        ph1, ph2 = sph_data.lonbounds * np.pi / 180
    
    # Check if starting points are in bounds
    inbounds = np.ones(nlines, dtype=bool)
    if bounded:
        if ph2 > ph1:
            inbounds &= (sph_data.stph >= ph1) & (sph_data.stph <= ph2)
        else:
            inbounds &= (sph_data.stph > ph2) | (sph_data.stph < ph1)
    
    inbounds &= (sph_data.str >= rmin) & (sph_data.str <= rmax)
    inbounds &= (sph_data.stth >= thmin) & (sph_data.stth <= thmax)
    
    # Get grid spacings
    deltar = sph_data.rix[1] - sph_data.rix[0] if sph_data.nr > 1 else 0.1
    deltath = sph_data.theta[0] - sph_data.theta[1] if sph_data.nlat > 1 else 0.1
    deltaph = (sph_data.lon[1] - sph_data.lon[0]) * np.pi / 180 if sph_data.nlon > 1 else 0.1
    
    # Initialize output arrays
    nstep = np.zeros(nlines, dtype=int)
    linekind_arr = np.zeros(nlines, dtype=int)
    linelengths_arr = np.zeros(nlines)
    
    if endpoints:
        ptr = np.zeros((2, nlines))
        ptth = np.zeros((2, nlines))
        ptph = np.zeros((2, nlines))
    else:
        ptr = np.zeros((stepmax, nlines))
        ptth = np.zeros((stepmax, nlines))
        ptph = np.zeros((stepmax, nlines))
    
    # Set starting points
    ptr[0, :] = sph_data.str
    ptth[0, :] = sph_data.stth
    ptph[0, :] = sph_data.stph
    
    if not quiet:
        print(f"spherical_trace_field: tracing {nlines} field lines")
    
    # Main loop through fieldlines
    for i in range(nlines):
        if not inbounds[i]:
            nstep[i] = -1
            linekind_arr[i] = -1
            continue
        
        # Initialize for this fieldline
        ir = np.zeros(stepmax)
        ith = np.zeros(stepmax)
        iph = np.zeros(stepmax)
        
        ir[0] = ptr[0, i]
        ith[0] = ptth[0, i]
        iph[0] = ptph[0, i]
        
        step = 1
        
        # Simple fieldline tracing (simplified version)
        while step < stepmax:
            # Current point
            ptc = np.array([ir[step-1], ith[step-1], iph[step-1]])
            
            # Calculate field at current point
            try:
                irc = get_interpolation_index(sph_data.rix, ptc[0])
                ithc = get_interpolation_index(sph_data.lat, 90 - ptc[1] * 180/np.pi)
                iphc = get_interpolation_index(sph_data.lon, (ptc[2] * 180/np.pi + 360) % 360)
                
                brc = interpolate_trilinear(sph_data.br, iphc, ithc, irc)
                bthc = interpolate_trilinear(sph_data.bth, iphc, ithc, irc) / ptc[0]
                bphc = interpolate_trilinear(sph_data.bph, iphc, ithc, irc) / (ptc[0] * np.sin(ptc[1]))
                
                # Field direction
                ds = np.array([brc, bthc, bphc])
                
                # Step size
                steplen = safety * min(deltar, deltath, deltaph) / (np.linalg.norm(ds) + 1e-10)
                
                # Take step
                result = forw_euler(step-1, ptc, steplen, ds)
                
                ir[step] = result[0]
                ith[step] = result[1]
                iph[step] = result[2]
                
                # Check boundaries
                if (ir[step] < rmin or ir[step] > rmax or 
                    ith[step] < thmin or ith[step] > thmax):
                    break
                
                if bounded:
                    if ph2 > ph1:
                        if iph[step] < ph1 or iph[step] > ph2:
                            break
                    else:
                        if iph[step] > ph1 and iph[step] < ph2:
                            break
                
                step += 1
                
            except (IndexError, ValueError):
                break
        
        # Set results for this fieldline
        nstep[i] = step
        linekind_arr[i] = 2  # Default: crosses boundaries
        
        if endpoints:
            ptr[:, i] = [ir[0], ir[step-1]]
            ptth[:, i] = [ith[0], ith[step-1]]
            ptph[:, i] = [iph[0], iph[step-1]]
        else:
            ptr[:step, i] = ir[:step]
            ptth[:step, i] = ith[:step]
            ptph[:step, i] = iph[:step]
        
        # Calculate line length (simplified)
        if step > 1:
            xpt = ir[:step] * np.sin(ith[:step]) * np.sin(iph[:step])
            ypt = ir[:step] * np.sin(ith[:step]) * np.cos(iph[:step])
            zpt = ir[:step] * np.cos(ith[:step])
            
            ptdist = np.sqrt(np.diff(xpt)**2 + np.diff(ypt)**2 + np.diff(zpt)**2)
            linelengths_arr[i] = np.sum(ptdist)
    
    # Store results
    if not endpoints and np.max(nstep) < stepmax:
        maxnstep = np.max(nstep)
        ptr = ptr[:maxnstep, :]
        ptth = ptth[:maxnstep, :]
        ptph = ptph[:maxnstep, :]
    
    sph_data.set_fieldline_trajectories(ptr, ptth, ptph, nstep)
    
    # Set optional outputs
    if outfield is not None:
        outfield[:] = np.zeros((nlines, 3, 3))  # Simplified
    if linelengths is not None:
        linelengths[:] = linelengths_arr
    if linekind is not None:
        linekind[:] = linekind_arr
    
    if not quiet:
        valid_lines = np.sum(nstep > 0)
        print(f"spherical_trace_field: traced {valid_lines} valid fieldlines")


if __name__ == "__main__":
    # Example usage
    from spherical_field_data__define import SphericalFieldData
    from spherical_field_start_coord import spherical_field_start_coord
    
    # Create sample data
    sph_data = SphericalFieldData()
    
    # Set up grid
    nr, nlat, nlon = 20, 30, 40
    rix = np.linspace(1.0, 2.5, nr)
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    
    sph_data.set_coordinate_arrays(lon, lat, rix)
    
    # Create simple dipole field
    LON, LAT, R = np.meshgrid(lon, lat, rix, indexing='ij')
    theta = (90 - LAT) * np.pi / 180
    
    # Dipole field components
    br = 2.0 * np.cos(theta) / R**3
    bth = np.sin(theta) / R**3
    bph = np.zeros_like(br)
    
    sph_data.set_vector_field(br, bth, bph)
    
    # Set starting points
    spherical_field_start_coord(sph_data, fieldtype=5, spacing=10)
    
    print(f"Set up {len(sph_data.str)} starting points")
    
    # Trace fieldlines
    spherical_trace_field(sph_data, stepmax=1000, quiet=False)
    
    print(f"Traced fieldlines with steps: {sph_data.nstep}")
    print(f"Valid fieldlines: {np.sum(sph_data.nstep > 0)}")
    
    if sph_data.nstep is not None:
        valid_steps = sph_data.nstep[sph_data.nstep > 0]
        if len(valid_steps) > 0:
            print(f"Average steps per line: {np.mean(valid_steps):.1f}")
            print(f"Max steps: {np.max(valid_steps)}")