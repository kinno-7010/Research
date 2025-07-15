#!/usr/bin/env python3
"""
pfss_draw_field - This procedure draws an image (or a series of images) of a
                  magnetogram with field lines

Usage: pfss_draw_field(open=open, bcent=bcent, lcent=lcent, mag=mag,
                       width=width, crop=crop, imsc=imsc, thick=thick, file=file,
                       outim=outim, onscreen=onscreen, movie=movie, nolines=nolines,
                       noimage=noimage, drawopen=drawopen, drawclosed=drawclosed,
                       for_ps=for_ps, quiet=quiet)
       
Parameters:
    bcent, lcent: central (lat,lon) in degrees of centroid of projection (default = (0,0))
    open: output array indicating polarity of each field line (-1=negative, 0=closed, 1=positive)
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
    - width keyword does not work when less than 1

M.DeRosa - 8 Feb 2002 - converted from an earlier script
           29 May 2002 - added crop keyword
           30 Jul 2002 - added for_ps,outim,onscreen keyword
           1 Aug 2002 - added thick keyword
           29 Jan 2003 - fixed cropping (I think)
           25 Apr 2003 - added nolines,drawopen,drawclosed keywords
           12 May 2003 - fixed logic with nolines,drawopen,drawclosed
           3 Jun 2003 - added quiet keyword
           4 Jun 2003 - for /onscreen, removed call to loadct_mld
           26 Aug 2003 - fixed bug related to lcent being a 1-element vector
           28 Jan 2004 - changed set_plot,'x' to SSW procedure set_x
           31 May 2005 - changed set_x command to set_plot,olddname
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

# Import other modules from the PFSS package
from .pfss_data_block import PfssDataBlock
from .pfss_print_time import pfss_print_time
from .scim import scim
from .get_interpolation_index import get_interpolation_index
from .get_string_number import get_string_number
from .union import union


def pfss_draw_field(open=None, bcent=None, lcent=None, mag=None, width=None,
                   crop=None, imsc=None, thick=None, file=None, outim=None,
                   onscreen=False, movie=False, nolines=False, noimage=False,
                   drawopen=None, drawclosed=None, for_ps=False, quiet=False):
    """
    Draw an image (or a series of images) of a magnetogram with field lines
    
    Parameters:
    -----------
    open : array, optional
        Output array indicating polarity of each field line (-1=negative, 0=closed, 1=positive)
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
    nolines : bool, optional
        If True, no field lines are drawn
    noimage : bool, optional
        If True, an opaque black sphere appears where the central image would have been
    drawopen : bool, optional
        If True, only open field lines are drawn
    drawclosed : bool, optional
        If True, only closed field lines are drawn
    for_ps : bool, optional
        If True, interchange white and black colors
    quiet : bool, optional
        Set to inhibit screen output
    
    Returns:
    --------
    dict : Dictionary containing results including 'open' array and 'rimage'
    """
    
    # Get data from common block (assuming it's a global object)
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
    
    # Initialize open array
    if open is None:
        open = np.zeros(npt, dtype=int)
    
    # Loop through images
    if not quiet:
        print('  pfss_draw_field: rendering image ...')
    
    for j in range(nframe):
        if for_movie and not quiet:
            pfss_print_time('  ', j+1, nframe)
        
        if for_movie:
            lcent_current = j
        else:
            lcent_current = lcent
        
        # Create figure for plotting
        fig, ax = plt.subplots(figsize=(10, 10))
        
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
        
        # Get image dimensions
        nax = outim.shape
        
        # Create larger image with border
        new_size = (int(nax[0] * width), int(nax[1] * width))
        border_img = np.zeros(new_size, dtype=outim.dtype)
        if for_ps:
            border_img.fill(whi)
        else:
            border_img.fill(bla)
        
        # Place original image in center
        start_x = int(nax[0] * (width - 1) / 2)
        start_y = int(nax[1] * (width - 1) / 2)
        border_img[start_x:start_x + nax[0], start_y:start_y + nax[1]] = outim
        
        # Apply cropping
        bbox = np.array([nax[0], nax[1], nax[0], nax[1]]) * width * crop
        bbox = bbox.astype(int)
        
        winxsz = bbox[2] - bbox[0]
        winysz = bbox[3] - bbox[1]
        
        # Crop the image
        cropped_img = border_img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        
        # Set up plot bounds
        bval = (2 * crop - 1) * width
        
        # Display the magnetogram
        ax.imshow(cropped_img, extent=[bval[0], bval[2], bval[1], bval[3]], 
                 cmap='gray', origin='lower')
        
        # Now draw the individual field lines
        if not nolines:
            for i in range(npt):
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
                        open[i] = 1
                    else:
                        open[i] = -1
                # else open[i] = 0, which is already set
                
                # Only plot lines that go higher than the first radial gridpoint
                heightflag = np.max(ptr_line) > data_block.rix[1]
                drawflag = (drawopen and (open[i] != 0)) or (drawclosed and (open[i] == 0))
                
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
                        
                        # Determine color
                        if open[i] == -1:
                            color = 'blue'
                        elif open[i] == 0:
                            color = 'black' if for_ps else 'white'
                        else:  # open[i] == 1
                            color = 'green'
                        
                        # Plot the field line
                        ax.plot(xpp_vis, zpp_vis, color=color, linewidth=thick)
        
        # Set plot limits and remove axes
        ax.set_xlim(bval[0], bval[2])
        ax.set_ylim(bval[1], bval[3])
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Capture the image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        outim = np.asarray(buf)
        
        # Handle output
        if onscreen:
            plt.show()
        elif file is not None:
            if for_movie:
                filename = file + get_string_number(j, pad=3) + '.fits'
            else:
                filename = file + '.fits'
            
            # Save as FITS file
            fits.writeto(filename, outim, overwrite=True)
        
        plt.close(fig)
    
    # Return results
    results = {
        'open': open,
        'rimage': outim
    }
    
    return results