#!/usr/bin/env python3
"""
pfss_draw_field3 - This procedure renders an image (or a series of images)
                   of a magnetogram with field lines, but uses object
                   graphics to plot line crossings more accurately

Usage: pfss_draw_field3(bcent=bcent, lcent=lcent, mag=mag, width=width,
                        crop=crop, imsc=imsc, thick=thick, file=file,
                        outim=outim, onscreen=onscreen, movie=movie,
                        for_ps=for_ps, nolines=nolines, noimage=noimage,
                        drawopen=drawopen, drawclosed=drawclosed, quiet=quiet)

Parameters:
    bcent, lcent: central (lat,lon) in degrees of centroid of projection (default = (0,0))
    mag: magnification of central image (default=1)
    width: width of final image relative to central magnetogram image (default=2.5)
    crop: [x0,y0,x1,y1] cropping coordinates in normalized units
    imsc: data value(s) to which to scale central magnetogram image
    thick: thickness of field lines
    file: if set, FITS files of image(s) are created
    outim: on output, final image is read into this variable
    onscreen: if set, then display image onscreen
    movie: if set, creates movie sequence of field-line data
    for_ps: if set, then interchange white and black colors
    nolines: if set, no field lines are drawn
    noimage: if set, an opaque black sphere appears where the central image would have been
    drawopen: if set, only open field lines are drawn
    drawclosed: if set, only closed field lines are drawn
    quiet: set to inhibit screen output

Notes:
    - Unlike pfss_draw_field and pfss_draw_field2, this routine produces
      a 24-bit (true-color) image.
    - Uses modern 3D graphics rendering for accurate line crossings

M.DeRosa - 22 Aug 2006 - created
           12 Nov 2010 - updated for IDL version 8 compatibility

Converted to Python by Claude Code
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as mpatches
from astropy.io import fits
import time
import sys
import os
from scipy.spatial.transform import Rotation

# Import other modules from the PFSS package
from .pfss_data_block import PfssDataBlock
from .pfss_print_time import pfss_print_time
from .pfss_view_create import pfss_view_create
from .get_string_number import get_string_number
from .scim import scim


def pfss_draw_field3(bcent=None, lcent=None, mag=None, width=None, crop=None,
                    imsc=None, thick=None, file=None, outim=None, onscreen=False,
                    movie=False, for_ps=False, nolines=False, noimage=False,
                    drawopen=None, drawclosed=None, quiet=False):
    """
    Render an image (or a series of images) of a magnetogram with field lines,
    using object graphics to plot line crossings more accurately
    
    Parameters:
    -----------
    bcent : float, optional
        Central latitude in degrees (default=0)
    lcent : float, optional
        Central longitude in degrees (default=0)
    mag : float, optional
        Magnification of central image (default=1)
    width : float, optional
        Width of final image relative to central magnetogram image (default=2.5)
    crop : array, optional
        [x0,y0,x1,y1] cropping coordinates in normalized units
    imsc : float or array, optional
        Data value(s) to which to scale central magnetogram image
    thick : float, optional
        Thickness of field lines (default=1)
    file : str, optional
        If set, FITS files of image(s) are created
    outim : array, optional
        On output, final image is read into this variable
    onscreen : bool, optional
        If True, display image onscreen
    movie : bool, optional
        If True, creates movie sequence of field-line data
    for_ps : bool, optional
        If True, interchange white and black colors
    nolines : bool, optional
        If True, no field lines are drawn
    noimage : bool, optional
        If True, an opaque black sphere appears where the central image would have been
    drawopen : bool, optional
        If True, only open field lines are drawn
    drawclosed : bool, optional
        If True, only closed field lines are drawn
    quiet : bool, optional
        Set to inhibit screen output
    
    Returns:
    --------
    dict : Dictionary containing results including 'rimage' (24-bit color image)
    """
    
    # Get data from common block
    data_block = PfssDataBlock()
    
    # Some error checking and default values
    if bcent is None:
        bcent = 0.0
    if lcent is None:
        lcent = 0.0
    if mag is None:
        mag = 1
    if width is None:
        width = 2.5
    if crop is None:
        crop = np.array([0, 0, 1, 1], dtype=float)
    
    # Handle movie mode
    if movie:
        for_movie = True
        if file is None:
            file = 'test'
    else:
        for_movie = False
    
    # Handle file output
    if file is not None:
        onscreen = False
        if not isinstance(file, str):
            file = 'test'
    
    if thick is None:
        thick = 1
    
    # Set image scaling
    if imsc is None:
        br_data = data_block.br[:, :, 0]
        imsc = np.max(np.abs([np.min(br_data), np.max(br_data)]))
    
    # Handle drawing flags
    if drawclosed is None and drawopen is None:
        drawclosed = True
        drawopen = True
    else:
        if drawclosed is None:
            drawclosed = False
        else:
            drawclosed = bool(drawclosed)
        if drawopen is None:
            drawopen = False
        else:
            drawopen = bool(drawopen)
    
    # Some RGB colors
    bla = np.array([0, 0, 0])
    gre = np.array([0, 255, 0])
    red = np.array([255, 0, 255])
    whi = np.array([255, 255, 255])
    
    # Create view object (this would need to be implemented)
    view_objects = pfss_view_create()
    
    # Create and add a texture map to the view
    if not hasattr(data_block, 'im_data') or data_block.im_data is None:
        # Create spherical image (this would need implementation)
        data_block.im_data = spherical_image_create(data_block.br[:, :, 0], 
                                                   data_block.lon, 
                                                   data_block.lat)
    
    # Create spherical texture map (this would need implementation)
    sphere_texture = spherical_texmap_create(data_block.im_data, imsc=imsc)
    
    # Preliminaries
    nlines = len(view_objects.get('fieldlines', []))
    if for_movie:
        nframe = 360
    else:
        nframe = 1
    
    rmax = np.max(data_block.rix)
    
    # Correct background color and correct colors for fieldlines
    if for_ps:
        background_color = whi
        # Correct field line colors for postscript
        for i, fieldline in enumerate(view_objects.get('fieldlines', [])):
            if np.allclose(fieldline.get('color', []), whi):
                fieldline['color'] = bla
    else:
        background_color = bla
    
    # Remove central image if desired
    if noimage:
        sphere_texture = None
    
    # Set line thicknesses
    for fieldline in view_objects.get('fieldlines', []):
        fieldline['thick'] = thick
    
    # Remove lines if desired
    if nolines:
        view_objects['fieldlines'] = []
    else:
        # Filter field lines based on draw flags
        filtered_fieldlines = []
        for fieldline in view_objects.get('fieldlines', []):
            color = fieldline.get('color', bla)
            do_draw = False
            
            # Check if this is an open field line (green or red)
            if (np.allclose(color, gre) or np.allclose(color, red)) and drawopen:
                do_draw = True
            # Check if this is a closed field line (black or white)
            elif (np.allclose(color, bla) or np.allclose(color, whi)) and drawclosed:
                do_draw = True
            
            if do_draw:
                filtered_fieldlines.append(fieldline)
        
        view_objects['fieldlines'] = filtered_fieldlines
    
    # Loop through images
    if not quiet:
        print('  pfss_draw_field3: rendering image ...')
    
    for j in range(nframe):
        if for_movie and not quiet:
            pfss_print_time('  ', j+1, nframe)
        
        if for_movie:
            lcent_current = j
        else:
            lcent_current = lcent
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up the view
        ax.view_init(elev=90-bcent, azim=-(90+lcent_current))
        
        # Set background color
        fig.patch.set_facecolor(background_color/255.0)
        ax.set_facecolor(background_color/255.0)
        
        # Draw the sphere (magnetogram) if not disabled
        if sphere_texture is not None:
            # Create sphere mesh
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Apply texture mapping (simplified)
            # In a full implementation, this would map the magnetogram data
            # to the sphere surface with proper projection
            colors_sphere = plt.cm.gray(np.clip(sphere_texture, 0, 1))
            ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                           facecolors=colors_sphere, alpha=0.8)
        
        # Draw field lines
        for fieldline in view_objects.get('fieldlines', []):
            # Get field line coordinates (this would come from the view object)
            # For now, we'll use placeholder coordinates
            field_coords = fieldline.get('coordinates', [])
            if len(field_coords) > 0:
                x_coords = field_coords[:, 0]
                y_coords = field_coords[:, 1]
                z_coords = field_coords[:, 2]
                
                # Convert color to matplotlib format
                line_color = fieldline.get('color', bla) / 255.0
                line_width = fieldline.get('thick', thick)
                
                ax.plot(x_coords, y_coords, z_coords, 
                       color=line_color, linewidth=line_width, alpha=0.8)
        
        # Set equal aspect ratio and limits
        max_range = rmax * width
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # Remove axes
        ax.set_axis_off()
        
        # Render to get image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        outim = np.asarray(buf)
        
        # Convert to RGB (remove alpha channel)
        outim_rgb = outim[:, :, :3]
        
        # Resize according to magnification
        target_size = int(len(data_block.lat) * mag * rmax)
        if outim_rgb.shape[0] != target_size or outim_rgb.shape[1] != target_size:
            from scipy.ndimage import zoom
            zoom_factor = target_size / min(outim_rgb.shape[:2])
            outim_rgb = zoom(outim_rgb, (zoom_factor, zoom_factor, 1), order=1)
        
        # Reduce to proper width
        nax_outim = outim_rgb.shape
        winxsz = nax_outim[1]
        winysz = nax_outim[0]
        
        wfrac = width / rmax
        wcrp = np.array([(1-wfrac)/2, (1-wfrac)/2, 1-(1-wfrac)/2, 1-(1-wfrac)/2])
        
        x1 = max(0, int(wcrp[0] * winxsz))
        x2 = min(winxsz, int(wcrp[2] * winxsz))
        y1 = max(0, int(wcrp[1] * winysz))
        y2 = min(winysz, int(wcrp[3] * winysz))
        
        outim_rgb = outim_rgb[y1:y2, x1:x2, :]
        
        # Apply cropping
        nax_outim = outim_rgb.shape
        winxsz = nax_outim[1]
        winysz = nax_outim[0]
        
        crop_x1 = max(0, int(crop[0] * winxsz))
        crop_x2 = min(winxsz, int(crop[2] * winxsz))
        crop_y1 = max(0, int(crop[1] * winysz))
        crop_y2 = min(winysz, int(crop[3] * winysz))
        
        outim_rgb = outim_rgb[crop_y1:crop_y2, crop_x1:crop_x2, :]
        
        # Handle output
        if onscreen:
            # Display using scim with true color
            scim(outim_rgb, quiet=quiet, true_color=True)
        elif file is not None:
            if for_movie:
                filename = file + get_string_number(j, pad=3) + '.fits'
            else:
                filename = file + '.fits'
            
            # Save as FITS file
            fits.writeto(filename, outim_rgb, overwrite=True)
        
        plt.close(fig)
    
    # Return results - rimage is a 24-bit color image
    results = {
        'rimage': outim_rgb
    }
    
    return results


def spherical_image_create(br_data, lon, lat):
    """
    Create a spherical image from magnetogram data
    
    Parameters:
    -----------
    br_data : ndarray
        Magnetic field data
    lon : ndarray
        Longitude array
    lat : ndarray
        Latitude array
    
    Returns:
    --------
    ndarray : Processed image data for spherical mapping
    """
    # Simple implementation - in reality this would do proper spherical projection
    # Normalize the data
    normalized_data = (br_data - np.min(br_data)) / (np.max(br_data) - np.min(br_data))
    return normalized_data


def spherical_texmap_create(im_data, imsc=None):
    """
    Create a spherical texture map from image data
    
    Parameters:
    -----------
    im_data : ndarray
        Image data
    imsc : float, optional
        Image scaling factor
    
    Returns:
    --------
    ndarray : Texture map data
    """
    if imsc is not None:
        # Scale the image data
        scaled_data = im_data * imsc
        # Clip to valid range
        scaled_data = np.clip(scaled_data, 0, 1)
        return scaled_data
    else:
        return im_data