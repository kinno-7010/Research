#!/usr/bin/env python3
"""
pfss_draw_field_vrml.py - This procedure creates a VRML rendering of a 
                          field line extrapolation

Usage: pfss_draw_field_vrml(bcent=bcent, lcent=lcent, imsc=imsc, file=file,
                            subsample=subsample, extrude=extrude, quiet=quiet)

Parameters:
    bcent, lcent: central (lat,lon) in degrees of centroid of projection (default = (0,0))
    imsc: data value(s) to which to scale central magnetogram image
    file: name of VRML file (default = 'test')
    subsample: factor by which to subsample the number of points used to define each fieldline
    extrude: set to render each field line as a tube
    quiet: if set, disables screen output

Notes:
    - does not yet work for spherical segments, only full spheres

M.DeRosa - 19 Nov 2002 - created, adapted from draw_field.pro
           4 Jun 2003 - added to PFSS package
           14 Oct 2003 - resolved conflict with SSW get_pid process
           28 Jan 2004 - changed set_plot,'x' to SSW procedure set_x
           19 Apr 2006 - changed set_x command to set_plot,olddname
           10 Dec 2011 - discovered that texture map for VRML was inverted
F.Breitling - 16 Feb 2016 - added subsample keyword, writes PNG instead of TIFF

Converted to Python by Claude Code
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Import other modules from the PFSS package
from .pfss_data_block import PfssDataBlock
from .pfss_print_time import pfss_print_time
from .scim import scim
from .get_interpolation_index import get_interpolation_index
from .linrange import linrange


def pfss_draw_field_vrml(bcent=None, lcent=None, imsc=None, file=None,
                        subsample=None, extrude=False, quiet=False):
    """
    Create a VRML rendering of a field line extrapolation
    
    Parameters:
    -----------
    bcent : float, optional
        Central latitude in degrees (default=0)
    lcent : float, optional
        Central longitude in degrees (default=0)
    imsc : float or array, optional
        Data value(s) to which to scale central magnetogram image
    file : str, optional
        Name of VRML file (default='test')
    subsample : float, optional
        Factor by which to subsample the number of points used to define each fieldline
    extrude : bool, optional
        Set to render each field line as a tube
    quiet : bool, optional
        If set, disables screen output
    
    Returns:
    --------
    None : Creates VRML file on disk
    """
    
    # Get data from common block
    data_block = PfssDataBlock()
    
    # Some error checking and default values
    if bcent is None:
        bcent = 0
    if lcent is None:
        lcent = 0
    if file is None:
        file = 'test'
    if subsample is not None:
        subsample = max(float(subsample), 1.0)
    
    # Preliminaries
    dtor = np.pi / 180.0
    cb = np.cos(bcent * dtor)
    sb = -np.sin(bcent * dtor)
    
    rmin = np.min(data_block.rix)
    rmax = np.max(data_block.rix)
    
    # Determine if field lines start from top or bottom
    top = abs(data_block.ptr[0] - rmax) < abs(data_block.ptr[0] - rmin)
    
    npt = len(data_block.nstep)  # Number of field lines to be drawn
    
    # VRML color table (equivalent to color table 47)
    re = np.concatenate([np.arange(250, dtype=np.uint8), [0, 0, 255, 255, 255, 255]])
    gr = np.concatenate([np.arange(250, dtype=np.uint8), [255, 255, 0, 0, 255, 255]])
    bl = np.concatenate([np.arange(250, dtype=np.uint8), [0, 0, 255, 255, 255, 255]])
    
    # Convert colors to VRML format (0-1 range)
    blu = f"{re[252]/255.:.2f} {gr[252]/255.:.2f} {bl[252]/255.:.2f}"
    gre = f"{re[250]/255.:.2f} {gr[250]/255.:.2f} {bl[250]/255.:.2f}"
    whi = f"{re[254]/255.:.2f} {gr[254]/255.:.2f} {bl[254]/255.:.2f}"
    bla = f"{re[0]/255.:.2f} {gr[0]/255.:.2f} {bl[0]/255.:.2f}"
    
    # Open VRML file
    with open(file + '.wrl', 'w') as vrml_file:
        # Write header
        vrml_file.write('#VRML V2.0 utf8\n')
        vrml_file.write('\n')
        vrml_file.write('WorldInfo {\n')
        vrml_file.write('  title "VRML field lines" }\n')
        vrml_file.write('\n')
        
        # Set up entry viewpoint
        vrml_file.write('Viewpoint { position 0.0 0.0 10.0 description "Entry View" }\n')
        vrml_file.write('\n')
        
        # Set up north pole viewpoint
        vrml_file.write('Viewpoint { \n')
        north_pos = 10 * np.array([np.cos(bcent * dtor), np.sin(bcent * dtor)])
        vrml_file.write(f'  position 0.0 {north_pos[0]:8.3f} {north_pos[1]:8.3f}\n')
        vrml_file.write(f'  orientation 1.0 0.0 0.0 {-(90-bcent)*dtor:8.3f}\n')
        vrml_file.write('  description "North Pole" }\n')
        
        # Set up south pole viewpoint
        vrml_file.write('Viewpoint { \n')
        south_pos = -10 * np.array([np.cos(bcent * dtor), np.sin(bcent * dtor)])
        vrml_file.write(f'  position 0.0 {south_pos[0]:8.3f} {south_pos[1]:8.3f}\n')
        vrml_file.write(f'  orientation 1.0 0.0 0.0 {(90-bcent)*dtor:8.3f}\n')
        vrml_file.write('  description "South Pole" }\n')
        
        # Display central image
        # Create magnetogram image
        outim = scim(data_block.br[:, :, 0], sc=imsc, quiet=True, 
                    m=512/len(data_block.lat), interp=True)
        
        # Create color table
        cmap = plt.cm.get_cmap('hot')
        colors = cmap(np.linspace(0, 1, 256))
        
        # Convert to PIL Image and save as PNG
        if outim.dtype != np.uint8:
            outim = (outim * 255).astype(np.uint8)
        
        # Create RGB image
        img_rgb = np.zeros((outim.shape[0], outim.shape[1], 3), dtype=np.uint8)
        img_rgb[:, :, 0] = re[outim]
        img_rgb[:, :, 1] = gr[outim]
        img_rgb[:, :, 2] = bl[outim]
        
        # Save as PNG
        img = Image.fromarray(img_rgb)
        img.save(file + '.wrl.png')
        
        # Write texture sphere
        rotang = np.pi - lcent * np.pi / 180.0
        vrml_file.write('Transform {\n')
        vrml_file.write(f'  rotation 0.0 1.0 0.0 {rotang}\n')
        vrml_file.write('  children [\n')
        vrml_file.write('    Shape {\n')
        vrml_file.write('      appearance Appearance {\n')
        
        # Extract filename
        fn = os.path.basename(file)
        vrml_file.write(f'        texture ImageTexture {{ url "{fn}.wrl.png"}} }}\n')
        vrml_file.write('      geometry Sphere { radius 1.0 } } ] }\n')
        vrml_file.write('\n')
        
        # Display field lines
        for i in range(npt):
            # Print update message
            if not quiet:
                pfss_print_time('  pfss_draw_field_vrml: ', i+1, npt)
            
            # Transform from spherical to cartesian coordinates
            ns = data_block.nstep[i]
            
            # Get field line coordinates
            ptr_line = data_block.ptr[0:ns, i]
            ptth_line = data_block.ptth[0:ns, i]
            ptph_line = data_block.ptph[0:ns, i]
            
            # Spherical to Cartesian transformation
            xp = ptr_line * np.sin(ptth_line) * np.sin(ptph_line - lcent * dtor)
            yp = ptr_line * np.sin(ptth_line) * np.cos(ptph_line - lcent * dtor)
            zp = ptr_line * np.cos(ptth_line)
            
            # Apply latitudinal tilt
            xpp = xp
            ypp = cb * yp - sb * zp
            zpp = sb * yp + cb * zp
            
            # Determine whether line is open or closed
            if np.max(ptr_line) >= data_block.rix[-1]:
                irc = get_interpolation_index(data_block.rix, ptr_line[0])
                ithc = get_interpolation_index(data_block.lat, 90 - ptth_line[0] * 180/np.pi)
                iphc = get_interpolation_index(data_block.lon, 
                                             (ptph_line[0] * 180/np.pi + 360) % 360)
                
                # Interpolate magnetic field
                brc = np.interp(irc, range(len(data_block.br)), 
                               data_block.br[int(iphc), int(ithc), :])
                
                if brc > 0:
                    open_flag = 1
                else:
                    open_flag = -1
            else:
                open_flag = 0
            
            # Only plot those lines that are higher than the first radial gridpoint
            if np.max(ptr_line) > data_block.rix[1]:
                # Preamble to lines
                vrml_file.write('Shape {\n')
                
                # Set color
                if open_flag == -1:
                    col = blu
                elif open_flag == 0:
                    col = whi  # Assuming not for_ps
                else:  # open_flag == 1
                    col = gre
                
                vrml_file.write('  appearance Appearance { material Material { \n')
                vrml_file.write(f'    diffuseColor {col}\n')
                vrml_file.write(f'    emissiveColor {col} }} }}\n')
                
                if extrude:
                    # Extrusion commands
                    vrml_file.write('  geometry Extrusion {\n')
                    vrml_file.write('    crossSection [\n')
                    vrml_file.write('      .01 0, .00866 -.005, .00707 -.00707, .005 -.00866,\n')
                    vrml_file.write('      0 -.01, -.005 -.00866, -.00707 -.00707, -.00866 -.005,\n')
                    vrml_file.write('      -.01 0, -.00866 .005, -.00707 .00707, -.005 .00866,\n')
                    vrml_file.write('      0 .01, .005 .00866, .00707 .00707, .00866 .005,\n')
                    vrml_file.write('      .01 0]\n')
                    vrml_file.write('    spine [\n')
                else:
                    # Indexed line set commands
                    vrml_file.write('  geometry IndexedLineSet {\n')
                    vrml_file.write('    coord Coordinate {\n')
                    vrml_file.write('      point [\n')
                
                # Apply subsampling if requested
                if subsample is not None:
                    newstep = min(max(int(ns / subsample) + 1, 3), ns)
                    newix = np.round(linrange(newstep, 0, ns-1)).astype(int)
                    xpp = xpp[newix]
                    ypp = ypp[newix]
                    zpp = zpp[newix]
                    ns = len(newix)
                
                # Print coordinates
                for j in range(ns):
                    vrml_file.write(f'      {xpp[j]:8.3f} {zpp[j]:8.3f} {ypp[j]:8.3f},\n')
                
                # Finish geometry commands
                if extrude:
                    vrml_file.write('      ] } }\n')
                else:
                    vrml_file.write('      ] }\n')
                    vrml_file.write('    coordIndex [\n')
                    for j in range(ns):
                        vrml_file.write(f'{j:4d},\n')
                    vrml_file.write('    -1,] } }\n')
                
                vrml_file.write('\n')
    
    if not quiet:
        print(f"VRML file '{file}.wrl' created successfully")
        print(f"Texture image '{file}.wrl.png' created successfully")