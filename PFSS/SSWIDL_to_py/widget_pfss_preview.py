"""
widget_pfss_preview.py - Preview widget for PFSS field line visualization

This is a simplified Python version of the IDL preview widget for PFSS fields.
Provides quick preview capabilities for PFSS field line data.
"""

import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np


class PfssPreviewWidget:
    """
    Preview widget for PFSS field line visualization.
    """
    
    def __init__(self, pfssdata=None, preview_size=(400, 300)):
        """
        Initialize the PFSS preview widget.
        
        Args:
            pfssdata: PFSS field line data
            preview_size: Size of preview window
        """
        
        self.pfssdata = pfssdata
        self.preview_size = preview_size
        self.fig = None
        self.ax = None
        self.preview_active = False
        
        self.setup_preview()
    
    def setup_preview(self):
        """Set up the preview interface."""
        
        # Create compact figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        plt.subplots_adjust(bottom=0.15)
        
        # Create preview controls
        self.create_preview_controls()
        
        # Initial preview
        self.update_preview()
    
    def create_preview_controls(self):
        """Create preview control widgets."""
        
        # Create control axes
        ax_prev_button = plt.axes([0.1, 0.02, 0.1, 0.05])
        ax_next_button = plt.axes([0.25, 0.02, 0.1, 0.05])
        ax_zoom_button = plt.axes([0.4, 0.02, 0.1, 0.05])
        ax_full_button = plt.axes([0.55, 0.02, 0.15, 0.05])
        
        # Create buttons
        self.prev_button = widgets.Button(ax_prev_button, 'Prev')
        self.next_button = widgets.Button(ax_next_button, 'Next')
        self.zoom_button = widgets.Button(ax_zoom_button, 'Zoom')
        self.full_button = widgets.Button(ax_full_button, 'Full View')
        
        # Connect events
        self.prev_button.on_clicked(self.on_prev)
        self.next_button.on_clicked(self.on_next)
        self.zoom_button.on_clicked(self.on_zoom)
        self.full_button.on_clicked(self.on_full_view)
    
    def update_preview(self):
        """Update the preview display."""
        
        self.ax.clear()
        
        if self.pfssdata is not None:
            # Create preview of PFSS data
            # Downsample for faster preview
            preview_data = self.downsample_data(self.pfssdata)
            
            # Plot preview
            im = self.ax.imshow(preview_data, cmap='RdBu_r', origin='lower', 
                               aspect='auto')
            self.ax.set_title('PFSS Preview')
            self.ax.set_xlabel('Longitude (preview)')
            self.ax.set_ylabel('Latitude (preview)')
            
            # Minimal colorbar
            if hasattr(self, 'cbar'):
                self.cbar.remove()
            self.cbar = self.fig.colorbar(im, ax=self.ax, shrink=0.8)
        else:
            self.ax.text(0.5, 0.5, 'No data for preview', 
                        ha='center', va='center', transform=self.ax.transAxes)
        
        self.fig.canvas.draw()
    
    def downsample_data(self, data, factor=4):
        """
        Downsample data for faster preview.
        
        Args:
            data: Input data array
            factor: Downsampling factor
            
        Returns:
            array: Downsampled data
        """
        
        if data.ndim == 2:
            return data[::factor, ::factor]
        else:
            return data[::factor]
    
    def on_prev(self, event):
        """Handle previous button click."""
        print("Previous view")
        # Implement view navigation
        
    def on_next(self, event):
        """Handle next button click."""
        print("Next view")
        # Implement view navigation
        
    def on_zoom(self, event):
        """Handle zoom button click."""
        print("Zoom view")
        # Implement zoom functionality
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        
        # Zoom in by factor of 2
        zoom_factor = 0.5
        center_x = np.mean(current_xlim)
        center_y = np.mean(current_ylim)
        
        new_width = (current_xlim[1] - current_xlim[0]) * zoom_factor
        new_height = (current_ylim[1] - current_ylim[0]) * zoom_factor
        
        self.ax.set_xlim(center_x - new_width/2, center_x + new_width/2)
        self.ax.set_ylim(center_y - new_height/2, center_y + new_height/2)
        
        self.fig.canvas.draw()
    
    def on_full_view(self, event):
        """Handle full view button click."""
        print("Opening full view...")
        # This would launch the full widget_pfss_field
        try:
            from widget_pfss_field import widget_pfss_field
            widget_pfss_field(self.pfssdata)
        except ImportError:
            print("Full view widget not available")
    
    def show(self):
        """Display the preview widget."""
        plt.show()
    
    def set_data(self, pfssdata):
        """Set new PFSS data for preview."""
        self.pfssdata = pfssdata
        self.update_preview()


def widget_pfss_preview(pfssdata=None):
    """
    Create and display PFSS preview widget.
    
    Args:
        pfssdata: PFSS field line data
        
    Returns:
        PfssPreviewWidget: Preview widget instance
    """
    
    widget = PfssPreviewWidget(pfssdata)
    widget.show()
    return widget


def create_quick_preview(pfssdata, title="PFSS Quick Preview"):
    """
    Create a quick preview plot without interactive widgets.
    
    Args:
        pfssdata: PFSS field line data
        title: Plot title
    """
    
    if pfssdata is None:
        print("No data available for preview")
        return
    
    # Create simple preview plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(pfssdata, cmap='RdBu_r', origin='lower', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Field Strength')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Create synthetic data for testing
    data = np.random.randn(50, 50)
    widget = widget_pfss_preview(data)
    print("PFSS Preview Widget launched")