"""
ssw_pfss2linecolors.py - Convert PFSS field line data to line colors

This routine converts PFSS field line data to appropriate line colors for visualization.
"""

import numpy as np
import matplotlib.colors as mcolors


def ssw_pfss2linecolors(pfssdata, color_scheme='default', normalize=True):
    """
    Convert PFSS field line data to line colors.
    
    Args:
        pfssdata: PFSS field line data array
        color_scheme: Color scheme to use ('default', 'magnetic', 'custom')
        normalize: Whether to normalize color values
        
    Returns:
        array: RGB color values for each data point
    """
    
    if pfssdata is None:
        return None
    
    # Define color schemes
    if color_scheme == 'default':
        # Standard PFSS colors: red for negative, green for positive, white for closed
        pos_color = [0, 1, 0]  # Green
        neg_color = [1, 0, 0]  # Red
        closed_color = [1, 1, 1]  # White
    elif color_scheme == 'magnetic':
        # Magnetic field colors
        pos_color = [0, 0, 1]  # Blue
        neg_color = [1, 0, 0]  # Red
        closed_color = [0.5, 0.5, 0.5]  # Gray
    else:
        # Custom color scheme
        pos_color = [0, 1, 0]
        neg_color = [1, 0, 0]
        closed_color = [1, 1, 1]
    
    # Create color array
    colors = np.zeros(pfssdata.shape + (3,))
    
    # Assign colors based on field line type
    positive_mask = pfssdata > 0
    negative_mask = pfssdata < 0
    closed_mask = pfssdata == 0
    
    colors[positive_mask] = pos_color
    colors[negative_mask] = neg_color
    colors[closed_mask] = closed_color
    
    if normalize:
        colors = colors / 255.0 if colors.max() > 1 else colors
    
    return colors


def create_colormap_from_pfss(pfssdata):
    """Create a custom colormap from PFSS data."""
    
    # Create custom colormap
    colors = ['red', 'white', 'green']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('pfss', colors, N=n_bins)
    
    return cmap


if __name__ == "__main__":
    print("ssw_pfss2linecolors - PFSS field line color conversion")