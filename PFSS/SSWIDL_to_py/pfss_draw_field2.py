#!/usr/bin/env python3
"""
pfss_draw_field2 - This procedure renders an image (or a series of images)
                   of a magnetogram with field lines, but plots line
                   crossings more accurately

Usage: pfss_draw_field2(bcent=bcent, lcent=lcent, mag=mag, width=width,
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
    outim: on output, image of z-buffer is read into this variable
    onscreen: if set, then display image onscreen
    movie: if set, creates movie sequence of field-line data
    for_ps: if set, then interchange white and black colors
    nolines: if set, no field lines are drawn
    noimage: if set, an opaque black sphere appears where the central image would have been
    drawopen: if set, only open field lines are drawn
    drawclosed: if set, only closed field lines are drawn
    quiet: set to inhibit screen output

Notes:
    - crop keyword does not work with for_ps flag set
    - if drawopen is set and drawclosed is not, would be faster to use
      pfss_draw_field instead of this routine

M.DeRosa - 8 Feb 2002 - converted from an earlier script
           29 May 2002 - added crop keyword
           30 Jul 2002 - added for_ps,outim,onscreen keyword
           1 Aug 2002 - added thick keyword
           6 Aug 2002 - corrected for improper front-back projection using "better Z-buffer"
           28 Oct 2002 - added quiet keyword
           6 Nov 2002 - moved cropping to end of routine
           19 Nov 2002 - removed open keyword, obsolete
           11 Mar 2003 - added nolines keyword
           25 Apr 2003 - added drawopen,drawclosed keywords
           12 May 2003 - fixed logic with nolines,drawopen,drawclosed
           4 Jun 2003 - for /onscreen, removed call to loadct_mld
           26 Aug 2003 - fixed bug related to lcent being a 1-element vector
           28 Jan 2004 - changed set_plot,'x' to SSW procedure set_x
           15 Jun 2005 - changed set_x command to set_plot,olddname
           12 Jul 2005 - added noimage keyword

Converted to Python by Claude Code
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from astropy.io import fits
import time
import sys
import os
from scipy.interpolate import interp1d

# Import other modules from the PFSS package
from .pfss_data_block import PfssDataBlock
from .pfss_print_time import pfss_print_time
from .scim import scim
from .get_interpolation_index import get_interpolation_index
from .get_string_number import get_string_number
from .union import union


def pfss_draw_field2(bcent=None, lcent=None, mag=None, width=None, crop=None,
                    imsc=None, thick=None, file=None, outim=None, onscreen=False,
                    movie=False, for_ps=False, nolines=False, noimage=False,
                    drawopen=None, drawclosed=None, quiet=False):
    """
    Render an image (or a series of images) of a magnetogram with field lines,
    but plots line crossings more accurately using a better Z-buffer
    
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
        On output, image of z-buffer is read into this variable
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
    dict : Dictionary containing results including 'open' array and 'rimage'
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
        br_min = np.min(data_block.br[:, :, 0])
        br_max = np.max(data_block.br[:, :, 0])
        imsc = max([-br_min, br_max])
    
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
    
    # Set colors (corresponding to ANA color table 47)
    gre = 250
    blu = 252
    whi = 254
    bla = 0
    
    # Preliminaries
    dtor = np.pi / 180.0
    cb = np.cos(bcent * dtor)
    sb = -np.sin(bcent * dtor)
    
    rmin = np.min(data_block.rix)
    rmax = np.max(data_block.rix)
    
    # Determine if field lines start from top or bottom
    top = abs(data_block.ptr[0] - rmax) < abs(data_block.ptr[0] - rmin)
    
    # Set number of frames
    if for_movie:
        nframe = 360
    else:
        nframe = 1
    
    npt = len(data_block.nstep)  # Number of field lines to be drawn
    
    # Loop through images
    if not quiet:
        print('  pfss_draw_field2: rendering image ...')
    
    for j in range(nframe):
        if for_movie and not quiet:
            pfss_print_time('  ', j+1, nframe)
        
        if for_movie:
            lcent_current = j
        else:
            lcent_current = lcent
        
        # Get image of full-disk magnetogram
        if hasattr(data_block, 'bounds') and data_block.bounds is not None:
            # Handle bounds case
            mgram = np.concatenate([data_block.br[:, :, 0], 
                                   data_block.br[:, :, 0], 
                                   data_block.br[:, :, 0]], axis=0)
            if noimage:
                mgram = np.full_like(mgram, min([imsc, -imsc]))
            
            radeg = 180.0 / np.pi
            bounds_diff = data_block.bounds[3] - data_block.bounds[2]
            olon = np.array([data_block.lon - radeg * bounds_diff,
                           data_block.lon,
                           data_block.lon + radeg * bounds_diff])
            
            outim = scim(mgram, m=mag, ortho=[lcent_current % 360, bcent],
                        olon=olon, olat=data_block.lat, sc=imsc, quiet=True,
                        top=248, white=for_ps)
        else:
            mgram = data_block.br[:, :, 0]
            if noimage:
                mgram = np.full_like(mgram, min([imsc, -imsc]))
            
            outim = scim(mgram, m=mag, ortho=[lcent_current % 360, bcent],
                        olon=data_block.lon, olat=data_block.lat, sc=imsc,
                        quiet=True, top=248, white=for_ps)
        
        # Set up canvas dimensions
        nax = outim.shape
        winxsz = int(nax[0] * width)
        winysz = int(nax[1] * width)
        xcent = int(nax[0] * width * 0.5)
        zcent = int(nax[1] * width * 0.5)
        rad = 0.5 * nax[0]
        
        # Position for placing the magnetogram
        pos = np.array([xcent, zcent, xcent, zcent]) + np.array([-1, -1, 1, 1]) * int(rad)
        
        # Create larger canvas
        canvas = np.zeros((winxsz, winysz), dtype=outim.dtype)
        if for_ps:
            canvas.fill(whi)
        else:
            canvas.fill(bla)
        
        # Place magnetogram on canvas
        canvas[pos[0]:pos[0]+nax[0], pos[1]:pos[1]+nax[1]] = outim
        outim = canvas.copy()
        
        # Create depth map array for better Z-buffer
        x_indices = np.arange(winxsz)
        z_indices = np.arange(winysz)
        xgrid, zgrid = np.meshgrid(x_indices, z_indices, indexing='ij')
        
        r2grid = ((xgrid - xcent)**2 + (zgrid - zcent)**2) / (rad**2)
        
        # Initialize depth map
        dmap = np.full((winxsz, winysz), -width, dtype=float)
        sphere_mask = r2grid <= 1
        dmap[sphere_mask] = 0.0
        
        # Draw field lines if requested
        if not nolines:
            open_array = np.zeros(npt, dtype=int)
            
            for i in range(npt):
                # Print update message
                if not quiet and not for_movie:
                    pfss_print_time('  pfss_draw_field2: ', i+1, npt)
                
                # Transform from spherical to cartesian coordinates
                ns = data_block.nstep[i]
                
                # Get field line coordinates
                ptr_line = data_block.ptr[0:ns, i]
                ptth_line = data_block.ptth[0:ns, i]
                ptph_line = data_block.ptph[0:ns, i]
                
                # Spherical to Cartesian transformation
                xp = ptr_line * np.sin(ptth_line) * np.sin(ptph_line - lcent_current * dtor)
                yp = ptr_line * np.sin(ptth_line) * np.cos(ptph_line - lcent_current * dtor)
                zp = ptr_line * np.cos(ptth_line)
                
                # Apply latitudinal tilt
                xpp = xp
                ypp = cb * yp - sb * zp
                zpp = sb * yp + cb * zp
                
                # Determine whether line is open or closed
                if (np.max(ptr_line) - rmin) / (rmax - rmin) > 0.99:
                    irc = get_interpolation_index(data_block.rix, ptr_line[0])
                    ithc = get_interpolation_index(data_block.lat, 90 - ptth_line[0] * 180/np.pi)
                    iphc = get_interpolation_index(data_block.lon, 
                                                 (ptph_line[0] * 180/np.pi + 360) % 360)
                    
                    # Interpolate magnetic field
                    brc = np.interp(irc, range(len(data_block.br)), 
                                   data_block.br[int(iphc), int(ithc), :])
                    
                    if brc > 0:
                        open_array[i] = 1
                    else:
                        open_array[i] = -1
                
                # Only plot lines that go higher than the first radial gridpoint
                heightflag = np.max(ptr_line) > data_block.rix[1]
                drawflag = (drawopen and (open_array[i] != 0)) or (drawclosed and (open_array[i] == 0))
                
                if heightflag and drawflag:
                    # Hide line segments that are behind disk
                    wh1 = np.where(ypp >= 0)[0]
                    wh2 = np.where((ypp < 0) & ((xpp**2 + zpp**2) > data_block.rix[0]**2))[0]
                    
                    if len(wh1) > 0 and len(wh2) > 0:
                        wh = union(wh1, wh2)
                    elif len(wh1) > 0:
                        wh = wh1
                    elif len(wh2) > 0:
                        wh = wh2
                    else:
                        continue
                    
                    if len(wh) > 0:
                        # Select visible coordinates
                        xpp_vis = xpp[wh]
                        ypp_vis = ypp[wh]
                        zpp_vis = zpp[wh]
                        
                        # Determine color value
                        if open_array[i] == -1:
                            col = blu
                        elif open_array[i] == 0:
                            col = bla if for_ps else whi
                        else:  # open_array[i] == 1
                            col = gre
                        
                        # Draw line using better Z-buffer approach
                        # Create a temporary buffer for this line
                        temp_buffer = np.zeros_like(canvas)
                        if for_ps:
                            temp_buffer.fill(whi)
                        else:
                            temp_buffer.fill(bla)
                        
                        # Convert 3D coordinates to 2D screen coordinates
                        screen_x = xpp_vis * rad + xcent
                        screen_z = zpp_vis * rad + zcent
                        
                        # Draw the line on the temporary buffer
                        for k in range(len(screen_x) - 1):
                            x1, z1 = int(screen_x[k]), int(screen_z[k])
                            x2, z2 = int(screen_x[k+1]), int(screen_z[k+1])
                            
                            # Simple line drawing (could be improved with anti-aliasing)
                            if (0 <= x1 < winxsz and 0 <= z1 < winysz and
                                0 <= x2 < winxsz and 0 <= z2 < winysz):
                                
                                # Draw line segment
                                dx = abs(x2 - x1)
                                dz = abs(z2 - z1)
                                
                                if dx > dz:
                                    steps = dx
                                else:
                                    steps = dz
                                
                                if steps > 0:
                                    x_step = (x2 - x1) / steps
                                    z_step = (z2 - z1) / steps
                                    
                                    for step in range(steps + 1):
                                        x = int(x1 + step * x_step)
                                        z = int(z1 + step * z_step)
                                        
                                        if 0 <= x < winxsz and 0 <= z < winysz:
                                            temp_buffer[x, z] = col
                        
                        # Find pixels that were drawn
                        line_pixels = np.where(temp_buffer == col)
                        
                        if len(line_pixels[0]) > 0:
                            # Calculate depth for each pixel
                            for pix_idx in range(len(line_pixels[0])):
                                x_pix = line_pixels[0][pix_idx]
                                z_pix = line_pixels[1][pix_idx]
                                
                                # Find corresponding depth from the 3D line
                                x_dist = (x_pix - xcent) / rad
                                z_dist = (z_pix - zcent) / rad
                                
                                # Find closest point on the line
                                distances = np.sqrt((xpp_vis - x_dist)**2 + (zpp_vis - z_dist)**2)
                                closest_idx = np.argmin(distances)
                                line_depth = ypp_vis[closest_idx]
                                
                                # Update depth map and output image if this pixel is closer
                                if line_depth > dmap[x_pix, z_pix]:
                                    dmap[x_pix, z_pix] = line_depth
                                    outim[x_pix, z_pix] = col
        
        # Crop image if desired
        crop_x1 = max(0, int(crop[0] * winxsz))
        crop_z1 = max(0, int(crop[1] * winysz))
        crop_x2 = min(winxsz, int(crop[2] * winxsz))
        crop_z2 = min(winysz, int(crop[3] * winysz))
        
        outim = outim[crop_x1:crop_x2, crop_z1:crop_z2]
        
        # Handle output
        if onscreen:
            # Set up color table
            plt.figure(figsize=(10, 10))
            plt.imshow(outim, cmap='gray')
            plt.axis('off')
            plt.show()
        elif file is not None:
            if for_movie:
                filename = file + get_string_number(j, pad=3) + '.fits'
            else:
                filename = file + '.fits'
            
            # Save as FITS file
            fits.writeto(filename, outim, overwrite=True)
    
    # Return results
    results = {
        'open': open_array if not nolines else None,
        'rimage': outim
    }
    
    return results