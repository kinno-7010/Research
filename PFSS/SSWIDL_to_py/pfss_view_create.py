"""
pfss_view_create.py - Create a view object containing a PFSS magnetic field model

PURPOSE: To create a view object containing a PFSS magnetic field model

CALLING SEQUENCE:
   object_list = pfss_view_create()

INPUTS: None specified explicitly, but from the common block we have
                br = r-component of magnetic field
                (ptr,ptth,ptph) = on input, contains a (n,stepmax)-array of
                                  field line coordinates
                nstep = an n-vector (where n=number of field lines) 
                        containing the number of points comprising each 
                        field line
                rimage = on output, image of z=buffer is read into this
                         variable

OUTPUTS:
   object_list = on output, contains a structure containing the objects
                 created by this routine

NOTES: -Object is created such that the viewer is looking down at the north
        pole, with 0 degrees longitude pointing to the right.
       -Once a valid PFSS model has been loaded, and some fieldlines have
        been traced, for simple testing at the Python prompt type:
          olist = pfss_view_create()
          # Use plotting library to display the field lines
       -To rotate the view to the ephemeral values before drawing, apply
        appropriate rotation transformations

MODIFICATION HISTORY: 
   M.DeRosa - 22 Aug 2006 - created
"""

import numpy as np
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

# Import required modules from the converted package
from pfss_data_block import PfssDataBlock
from pfss_to_spherical import pfss_to_spherical
from get_interpolation_index import get_interpolation_index


def pfss_view_create():
    """
    Create a view object containing a PFSS magnetic field model.
    
    Returns:
        dict: Structure containing the view objects and field line data
    """
    
    # First create spherical_field_data_structure
    pfss_data = pfss_to_spherical(free=True)
    
    if pfss_data is None:
        print("Error: No PFSS data available")
        return None
    
    # Set some RGB triplets of line colors
    colors = {
        'black': [0, 0, 0],
        'yellow': [255, 255, 0],
        'green': [0, 255, 0],
        'red': [255, 0, 255],
        'white': [255, 255, 255]
    }
    
    # Determine rmin and rmax and thmin and thmax
    rix = pfss_data.get('rix', np.array([1.0, 2.5]))
    rmax = np.max(rix)
    rmin = np.min(rix)
    
    theta = pfss_data.get('theta', np.linspace(0, np.pi, 48))
    thmin = np.min(theta)
    thmax = np.max(theta)
    
    # Create field line objects
    nstep = pfss_data.get('nstep', np.array([]))
    if len(nstep) == 0:
        print("Warning: No field lines found")
        return create_empty_object_list()
    
    nlines = len(nstep)
    open_status = np.zeros(nlines, dtype=int)
    
    # Arrays to store field line data
    field_lines = []
    line_colors = []
    
    # Process each field line
    for i in range(nlines):
        ns = nstep[i]
        if ns > 0:
            # Extract field line coordinates
            ptr = pfss_data.get('ptr', np.array([]))
            ptth = pfss_data.get('ptth', np.array([]))
            ptph = pfss_data.get('ptph', np.array([]))
            
            if len(ptr) == 0:
                continue
            
            # Determine whether field lines are open or closed
            if (np.max(ptr[:ns, i]) - rmin) / (rmax - rmin) > 0.99:
                # Calculate interpolation indices
                try:
                    irc = get_interpolation_index(rix, ptr[0, i])
                    ithc = get_interpolation_index(
                        pfss_data.get('lat', np.array([])), 
                        90 - ptth[0, i] * 180/np.pi
                    )
                    iphc = get_interpolation_index(
                        pfss_data.get('lon', np.array([])), 
                        (ptph[0, i] * 180/np.pi + 360) % 360
                    )
                    
                    # Interpolate magnetic field
                    br = pfss_data.get('br', np.array([]))
                    if br.size > 0:
                        # Use simplified interpolation
                        brc = interpolate_field(br, iphc, ithc, irc)
                        open_status[i] = 1 if brc > 0 else -1
                    else:
                        open_status[i] = 0
                        
                except Exception as e:
                    print(f"Warning: Could not determine open status for line {i}: {e}")
                    open_status[i] = 0
            
            # Flag those lines that go higher than the first radial gridpoint
            heightflag = np.max(ptr[:ns, i]) > rix[1] if len(rix) > 1 else True
            
            # Create an object for this line if it has sufficient height
            if heightflag:
                # Set appropriate color
                if open_status[i] == -1:
                    col = colors['red']
                elif open_status[i] == 0:
                    col = colors['white']
                else:  # open_status[i] == 1
                    col = colors['green']
                
                # Transform from spherical to cartesian coordinates
                linecoords = spherical_to_cartesian(
                    ptph[:ns, i], ptth[:ns, i], ptr[:ns, i]
                )
                
                # Store field line data
                field_lines.append(linecoords)
                line_colors.append(col)
    
    # Create structure of objects
    object_list = {
        'field_lines': field_lines,
        'line_colors': line_colors,
        'open_status': open_status,
        'rmin': rmin,
        'rmax': rmax,
        'thmin': thmin,
        'thmax': thmax,
        'nlines': nlines
    }
    
    return object_list


def spherical_to_cartesian(phi, theta, r):
    """
    Convert spherical coordinates to cartesian coordinates.
    
    Args:
        phi (array): Azimuthal angle (longitude) in radians
        theta (array): Polar angle (colatitude) in radians
        r (array): Radial distance
        
    Returns:
        array: Cartesian coordinates [x, y, z]
    """
    
    # Convert to cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.column_stack([x, y, z])


def interpolate_field(br, iphc, ithc, irc):
    """
    Interpolate magnetic field at given indices.
    
    Args:
        br (array): Magnetic field array
        iphc (float): Longitude index
        ithc (float): Latitude index
        irc (float): Radial index
        
    Returns:
        float: Interpolated field value
    """
    
    try:
        # Ensure indices are within bounds
        shape = br.shape
        iphc = max(0, min(iphc, shape[0] - 1))
        ithc = max(0, min(ithc, shape[1] - 1))
        irc = max(0, min(irc, shape[2] - 1))
        
        # Simple trilinear interpolation
        i0, i1 = int(iphc), min(int(iphc) + 1, shape[0] - 1)
        j0, j1 = int(ithc), min(int(ithc) + 1, shape[1] - 1)
        k0, k1 = int(irc), min(int(irc) + 1, shape[2] - 1)
        
        # Interpolation weights
        fi = iphc - i0
        fj = ithc - j0
        fk = irc - k0
        
        # Trilinear interpolation
        c00 = br[i0, j0, k0] * (1 - fi) + br[i1, j0, k0] * fi
        c01 = br[i0, j0, k1] * (1 - fi) + br[i1, j0, k1] * fi
        c10 = br[i0, j1, k0] * (1 - fi) + br[i1, j1, k0] * fi
        c11 = br[i0, j1, k1] * (1 - fi) + br[i1, j1, k1] * fi
        
        c0 = c00 * (1 - fj) + c10 * fj
        c1 = c01 * (1 - fj) + c11 * fj
        
        return c0 * (1 - fk) + c1 * fk
        
    except Exception as e:
        print(f"Warning: Field interpolation failed: {e}")
        return 0.0


def create_empty_object_list():
    """
    Create an empty object list structure.
    
    Returns:
        dict: Empty object list structure
    """
    
    return {
        'field_lines': [],
        'line_colors': [],
        'open_status': np.array([]),
        'rmin': 1.0,
        'rmax': 2.5,
        'thmin': 0.0,
        'thmax': np.pi,
        'nlines': 0
    }


def plot_field_lines(object_list, title='PFSS Field Lines'):
    """
    Plot the field lines using matplotlib.
    
    Args:
        object_list (dict): Object list from pfss_view_create
        title (str): Plot title
    """
    
    if not object_list or len(object_list['field_lines']) == 0:
        print("No field lines to plot")
        return
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each field line
    for i, (line, color) in enumerate(zip(object_list['field_lines'], 
                                         object_list['line_colors'])):
        if len(line) > 0:
            # Convert color to matplotlib format
            mpl_color = [c/255.0 for c in color]
            
            # Plot the line
            ax.plot(line[:, 0], line[:, 1], line[:, 2], 
                   color=mpl_color, linewidth=1.0)
    
    # Draw solar surface
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.3, color='yellow')
    
    # Set labels and title
    ax.set_xlabel('X (Solar Radii)')
    ax.set_ylabel('Y (Solar Radii)')
    ax.set_zlabel('Z (Solar Radii)')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = object_list['rmax']
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.tight_layout()
    plt.show()


def rotate_view(object_list, rotations):
    """
    Apply rotation transformations to the view.
    
    Args:
        object_list (dict): Object list from pfss_view_create
        rotations (list): List of rotation parameters
        
    Returns:
        dict: Rotated object list
    """
    
    # This is a placeholder for rotation functionality
    # In practice, would apply rotation matrices to field line coordinates
    warnings.warn("Rotation functionality not fully implemented")
    
    return object_list


def print_view_info(object_list):
    """
    Print information about the view object.
    
    Args:
        object_list (dict): Object list from pfss_view_create
    """
    
    if not object_list:
        print("No view object available")
        return
    
    print('PFSS View Information:')
    print('-' * 22)
    print(f'Number of field lines: {object_list["nlines"]}')
    print(f'Number of plotted lines: {len(object_list["field_lines"])}')
    print(f'Radial range: {object_list["rmin"]:.2f} to {object_list["rmax"]:.2f} R☉')
    print(f'Theta range: {object_list["thmin"]:.2f} to {object_list["thmax"]:.2f} rad')
    
    # Count open/closed lines
    open_count = np.sum(object_list['open_status'] == 1)
    closed_count = np.sum(object_list['open_status'] == 0)
    negative_count = np.sum(object_list['open_status'] == -1)
    
    print(f'Open field lines (positive): {open_count}')
    print(f'Open field lines (negative): {negative_count}')
    print(f'Closed field lines: {closed_count}')


# For compatibility with IDL calling convention
def pfss_view_create_idl():
    """
    IDL-compatible wrapper for pfss_view_create.
    
    This function maintains the original interface.
    """
    return pfss_view_create()


# Example usage
def pfss_view_create_example():
    """
    Example usage of pfss_view_create.
    """
    
    print("PFSS View Create Example")
    print("=" * 24)
    
    # Create view object
    object_list = pfss_view_create()
    
    if object_list is not None:
        # Print information
        print_view_info(object_list)
        
        # Plot field lines
        plot_field_lines(object_list)
        
        print("View created successfully!")
    else:
        print("Failed to create view - check PFSS data availability")


if __name__ == "__main__":
    pfss_view_create_example()