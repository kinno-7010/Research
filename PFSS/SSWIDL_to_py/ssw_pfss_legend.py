"""
ssw_pfss_legend.py - Create legend for PFSS field line plots

This routine creates legends for PFSS field line visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def ssw_pfss_legend(ax=None, location='upper right', open_positive_label='Open (+)', 
                    open_negative_label='Open (-)', closed_label='Closed'):
    """
    Create a legend for PFSS field line plots.
    
    Args:
        ax: Matplotlib axis object (if None, uses current axis)
        location: Legend location
        open_positive_label: Label for positive open field lines
        open_negative_label: Label for negative open field lines
        closed_label: Label for closed field lines
        
    Returns:
        legend: Matplotlib legend object
    """
    
    if ax is None:
        ax = plt.gca()
    
    # Create legend elements
    legend_elements = [
        plt.Line2D([0], [0], color='green', lw=2, label=open_positive_label),
        plt.Line2D([0], [0], color='red', lw=2, label=open_negative_label),
        plt.Line2D([0], [0], color='white', lw=2, label=closed_label)
    ]
    
    # Create legend
    legend = ax.legend(handles=legend_elements, loc=location, 
                      fancybox=True, shadow=True)
    
    return legend


def create_pfss_colorbar(fig, ax, pfssdata, label='Field Line Type'):
    """
    Create a colorbar for PFSS field line data.
    
    Args:
        fig: Matplotlib figure object
        ax: Matplotlib axis object
        pfssdata: PFSS field line data
        label: Colorbar label
        
    Returns:
        colorbar: Matplotlib colorbar object
    """
    
    # Create colorbar
    im = ax.imshow(pfssdata, cmap='RdBu_r', aspect='auto')
    cbar = fig.colorbar(im, ax=ax, label=label)
    
    return cbar


if __name__ == "__main__":
    print("ssw_pfss_legend - PFSS field line legend creation")