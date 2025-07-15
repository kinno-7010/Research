"""
Spherical Draw Field - Simplified Version

Given vector fieldline data on a spherical grid, this procedure renders
these fieldlines using matplotlib for 2D projections or 3D visualization.

CALLING SEQUENCE:
    spherical_draw_field(sph_data, xsize=512, ysize=512, bcent=0, lcent=0,
                        width=1, im_data=None, imsc=None, thick=1, outim=None,
                        onscreen=True, movie=False, drawopen=True, drawclosed=True,
                        nolines=False, noimage=False, for_ps=False, quiet=False)

INPUTS:
    sph_data = a SphericalFieldData object with fieldline trajectories
    xsize = number of pixels in width of rendered image (default=512)
    ysize = number of pixels in height of rendered image (default=512)
    bcent = central latitude in degrees of projection centroid (default=0)
    lcent = central longitude in degrees of projection centroid (default=0)
    width = width of horizontal extent relative to outer radius (default=1)
    im_data = SphericalImageData object for background image
    imsc = image scaling value(s)
    thick = thickness of field lines
    outim = output image array
    onscreen = if True, display image on screen
    movie = if True, create movie sequence (simplified)
    drawopen = if True, draw open field lines
    drawclosed = if True, draw closed field lines
    nolines = if True, don't draw field lines
    noimage = if True, don't draw background image
    for_ps = if True, interchange black and white for printing
    quiet = if True, suppress output

OUTPUTS:
    outim = rendered image array (if requested)

NOTES:
    - This is a simplified version of the full IDL implementation
    - Uses matplotlib for 2D projections instead of IDL Object Graphics
    - For full 3D functionality, integrate with mayavi/vtk

MODIFICATION HISTORY:
    M.DeRosa - 11 Jan 2006 - created (IDL version)
    [... many modifications ...]
    Converted to Python - 2025 (simplified version)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
from spherical_field_data__define import SphericalFieldData
from spherical_image_data__define import SphericalImageData
from spherical_image_create import spherical_image_create


def spherical_draw_field(sph_data: SphericalFieldData,
                        xsize: int = 512,
                        ysize: int = 512,
                        bcent: float = 0,
                        lcent: float = 0,
                        width: float = 1,
                        im_data: Optional[SphericalImageData] = None,
                        imsc: Optional[float] = None,
                        thick: float = 1,
                        outim: Optional[np.ndarray] = None,
                        onscreen: bool = True,
                        movie: bool = False,
                        drawopen: bool = True,
                        drawclosed: bool = True,
                        nolines: bool = False,
                        noimage: bool = False,
                        for_ps: bool = False,
                        quiet: bool = False) -> Optional[np.ndarray]:
    """
    Render spherical field lines using matplotlib (simplified version).
    
    Parameters:
    -----------
    sph_data : SphericalFieldData
        Spherical field data with trajectories
    xsize, ysize : int
        Image dimensions
    bcent, lcent : float
        Central latitude and longitude for projection
    width : float
        Field of view width
    im_data : SphericalImageData, optional
        Background image data
    imsc : float, optional
        Image scaling
    thick : float
        Line thickness
    outim : np.ndarray, optional
        Output image array
    onscreen : bool
        Display on screen
    movie : bool
        Create movie sequence
    drawopen, drawclosed : bool
        Draw open/closed field lines
    nolines, noimage : bool
        Skip lines/image
    for_ps : bool
        Postscript mode
    quiet : bool
        Suppress output
        
    Returns:
    --------
    np.ndarray or None
        Rendered image array if requested
    """
    
    # Input validation
    if sph_data is None:
        raise ValueError("ERROR in spherical_draw_field: no input data provided")
    
    if not quiet:
        print("spherical_draw_field: rendering field lines...")
        print("Note: This is a simplified 2D implementation")
    
    # Get field ranges
    if sph_data.rix is not None:
        rmax = np.max(sph_data.rix)
        rmin = np.min(sph_data.rix)
    else:
        rmax, rmin = 2.5, 1.0
    
    if sph_data.theta is not None:
        thmin, thmax = np.min(sph_data.theta), np.max(sph_data.theta)
    else:
        thmin, thmax = 0, np.pi
    
    # Create figure
    fig_width = xsize / 100
    fig_height = ysize / 100
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    
    # Set up projection (simplified orthographic)
    extent = rmax * width
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect('equal')
    
    # Set background color
    if for_ps:
        ax.set_facecolor('white')
        line_color = 'black'
        bg_color = 'white'
    else:
        ax.set_facecolor('black')
        line_color = 'white'
        bg_color = 'black'
    
    # Draw background image if provided
    if not noimage:
        if im_data is None and sph_data.br is not None:
            # Create image from radial field at inner boundary
            im_data = spherical_image_create(sph_data.br[:, :, 0], 
                                           sph_data.lon, sph_data.lat, 
                                           radius=rmin)
        
        if im_data is not None:
            _draw_background_image(ax, im_data, bcent, lcent, imsc, rmin)
    
    # Draw field lines
    if not nolines and sph_data.nstep is not None:
        _draw_field_lines(ax, sph_data, bcent, lcent, rmin, rmax, thick,
                         drawopen, drawclosed, for_ps, quiet)
    
    # Remove axes for clean appearance
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Display or save
    if onscreen:
        plt.tight_layout()
        plt.show()
    
    # Get image data if requested
    if outim is not None or movie:
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        if outim is not None:
            outim[:] = buf
        
        if movie:
            if not quiet:
                print("Movie creation simplified - returning single frame")
            return buf
    
    plt.close(fig)
    
    if not quiet:
        print("spherical_draw_field: rendering complete")
    
    return None


def _draw_background_image(ax, im_data: SphericalImageData, bcent: float, 
                          lcent: float, imsc: Optional[float], radius: float):
    """Draw background image on spherical surface."""
    
    # Get image scaling
    if imsc is None:
        imsc = np.max(np.abs(im_data.image))
    
    # Create simple projection of image
    # This is a very simplified version - full implementation would use
    # proper spherical projection
    
    lon_grid, lat_grid = np.meshgrid(im_data.lon, im_data.lat)
    
    # Simple orthographic projection
    # Rotate coordinates to center on (bcent, lcent)
    lon_rot = lon_grid - lcent
    lat_rot = lat_grid - bcent
    
    # Convert to radians
    lon_rot_rad = np.radians(lon_rot)
    lat_rot_rad = np.radians(lat_rot)
    
    # Project to x, y
    x_proj = radius * np.cos(lat_rot_rad) * np.sin(lon_rot_rad)
    y_proj = radius * np.sin(lat_rot_rad)
    
    # Plot image
    im = ax.contourf(x_proj, y_proj, im_data.image.T, 
                    levels=50, cmap='RdBu_r', alpha=0.7)
    
    # Add circular boundary
    circle = plt.Circle((0, 0), radius, fill=False, color='white', linewidth=2)
    ax.add_patch(circle)


def _draw_field_lines(ax, sph_data: SphericalFieldData, bcent: float, 
                     lcent: float, rmin: float, rmax: float, thick: float,
                     drawopen: bool, drawclosed: bool, for_ps: bool, quiet: bool):
    """Draw field lines on the plot."""
    
    if sph_data.nstep is None or sph_data.ptr is None:
        return
    
    nlines = len(sph_data.nstep)
    lines_drawn = 0
    
    # Color scheme
    if for_ps:
        open_color = 'red'
        closed_color = 'black'
    else:
        open_color = 'green'
        closed_color = 'white'
    
    for i in range(nlines):
        ns = sph_data.nstep[i]
        if ns <= 0:
            continue
        
        # Get line coordinates
        r_line = sph_data.ptr[:ns, i]
        th_line = sph_data.ptth[:ns, i]
        ph_line = sph_data.ptph[:ns, i]
        
        # Determine if line is open or closed (simplified)
        is_open = (np.max(r_line) - rmin) / (rmax - rmin) > 0.99
        
        # Decide whether to draw this line
        if (is_open and not drawopen) or (not is_open and not drawclosed):
            continue
        
        # Convert to Cartesian coordinates for projection
        x_line = r_line * np.sin(th_line) * np.cos(ph_line)
        y_line = r_line * np.sin(th_line) * np.sin(ph_line)
        z_line = r_line * np.cos(th_line)
        
        # Simple orthographic projection
        # Rotate to center on (bcent, lcent) - simplified
        lon_deg = ph_line * 180 / np.pi - lcent
        lat_deg = 90 - th_line * 180 / np.pi - bcent
        
        x_proj = r_line * np.cos(np.radians(lat_deg)) * np.sin(np.radians(lon_deg))
        y_proj = r_line * np.sin(np.radians(lat_deg))
        
        # Choose color
        color = open_color if is_open else closed_color
        
        # Draw line
        ax.plot(x_proj, y_proj, color=color, linewidth=thick, alpha=0.8)
        lines_drawn += 1
    
    if not quiet:
        print(f"  Drew {lines_drawn} field lines")


def create_movie_sequence(sph_data: SphericalFieldData, 
                         output_dir: str = "movie_frames",
                         n_frames: int = 36,
                         **kwargs) -> None:
    """
    Create a movie sequence by rotating the view.
    
    Parameters:
    -----------
    sph_data : SphericalFieldData
        Field data structure
    output_dir : str
        Directory to save frames
    n_frames : int
        Number of frames in movie
    **kwargs
        Additional arguments for spherical_draw_field
    """
    
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating movie sequence with {n_frames} frames...")
    
    # Rotate through longitudes
    for i in range(n_frames):
        lcent = i * 360 / n_frames
        
        # Render frame
        outim = np.zeros((kwargs.get('ysize', 512), kwargs.get('xsize', 512), 3), dtype=np.uint8)
        spherical_draw_field(sph_data, lcent=lcent, outim=outim, 
                           onscreen=False, quiet=True, **kwargs)
        
        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        plt.imsave(frame_path, outim)
        
        if i % 10 == 0:
            print(f"  Rendered frame {i+1}/{n_frames}")
    
    print(f"Movie frames saved to {output_dir}")
    print("Use ffmpeg to create video: ffmpeg -i frame_%03d.png -r 10 movie.mp4")


if __name__ == "__main__":
    # Example usage
    from spherical_field_data__define import SphericalFieldData
    from spherical_field_start_coord import spherical_field_start_coord
    from spherical_trace_field import spherical_trace_field
    
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
    
    br = 2.0 * np.cos(theta) / R**3
    bth = np.sin(theta) / R**3
    bph = np.zeros_like(br)
    
    sph_data.set_vector_field(br, bth, bph)
    
    # Set starting points and trace fieldlines
    spherical_field_start_coord(sph_data, fieldtype=5, spacing=8)
    spherical_trace_field(sph_data, stepmax=500, quiet=True)
    
    print(f"Traced {np.sum(sph_data.nstep > 0)} fieldlines")
    
    # Render field
    spherical_draw_field(sph_data, xsize=600, ysize=600, bcent=30, lcent=0,
                        onscreen=True, quiet=False)
    
    # Example with different viewing angles
    print("\nRendering different viewing angles...")
    for bcent in [0, 30, -30]:
        for lcent in [0, 90, 180]:
            title = f"bcent={bcent}, lcent={lcent}"
            spherical_draw_field(sph_data, xsize=400, ysize=400, 
                               bcent=bcent, lcent=lcent, onscreen=True, 
                               quiet=True)
            plt.title(title)
            plt.show()