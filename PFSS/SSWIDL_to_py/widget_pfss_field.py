"""
widget_pfss_field.py - Widget for PFSS field line visualization

This is a simplified Python version of the IDL widget for PFSS field visualization.
Uses matplotlib widgets for interactive control.
"""

import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np


class PfssFieldWidget:
    """
    Widget for interactive PFSS field line visualization.
    """
    
    def __init__(self, pfssdata=None):
        """
        Initialize the PFSS field widget.
        
        Args:
            pfssdata: PFSS field line data
        """
        
        self.pfssdata = pfssdata
        self.fig = None
        self.ax = None
        self.widgets = {}
        self.current_view = 'field_lines'
        
        self.setup_gui()
    
    def setup_gui(self):
        """Set up the GUI interface."""
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.2)
        
        # Create control widgets
        self.create_widgets()
        
        # Initial plot
        self.update_plot()
    
    def create_widgets(self):
        """Create interactive widgets."""
        
        # Create widget axes
        ax_button1 = plt.axes([0.1, 0.05, 0.1, 0.04])
        ax_button2 = plt.axes([0.25, 0.05, 0.1, 0.04])
        ax_button3 = plt.axes([0.4, 0.05, 0.1, 0.04])
        ax_slider = plt.axes([0.6, 0.05, 0.3, 0.03])
        
        # Create buttons
        self.widgets['refresh'] = widgets.Button(ax_button1, 'Refresh')
        self.widgets['save'] = widgets.Button(ax_button2, 'Save')
        self.widgets['reset'] = widgets.Button(ax_button3, 'Reset')
        
        # Create slider
        self.widgets['threshold'] = widgets.Slider(ax_slider, 'Threshold', 
                                                  0.0, 1.0, valinit=0.5)
        
        # Connect events
        self.widgets['refresh'].on_clicked(self.on_refresh)
        self.widgets['save'].on_clicked(self.on_save)
        self.widgets['reset'].on_clicked(self.on_reset)
        self.widgets['threshold'].on_changed(self.on_threshold_changed)
    
    def update_plot(self):
        """Update the plot display."""
        
        self.ax.clear()
        
        if self.pfssdata is not None:
            # Plot PFSS field data
            im = self.ax.imshow(self.pfssdata, cmap='RdBu_r', origin='lower')
            self.ax.set_title('PFSS Field Lines')
            self.ax.set_xlabel('Longitude')
            self.ax.set_ylabel('Latitude')
            
            # Add colorbar
            if hasattr(self, 'cbar'):
                self.cbar.remove()
            self.cbar = self.fig.colorbar(im, ax=self.ax)
        else:
            self.ax.text(0.5, 0.5, 'No PFSS data available', 
                        ha='center', va='center', transform=self.ax.transAxes)
        
        self.fig.canvas.draw()
    
    def on_refresh(self, event):
        """Handle refresh button click."""
        print("Refreshing PFSS field data...")
        self.update_plot()
    
    def on_save(self, event):
        """Handle save button click."""
        filename = 'pfss_field_widget.png'
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    
    def on_reset(self, event):
        """Handle reset button click."""
        print("Resetting view...")
        self.widgets['threshold'].reset()
        self.update_plot()
    
    def on_threshold_changed(self, val):
        """Handle threshold slider change."""
        print(f"Threshold changed to: {val:.2f}")
        # Apply threshold to data and update plot
        self.update_plot()
    
    def show(self):
        """Display the widget."""
        plt.show()
    
    def set_data(self, pfssdata):
        """Set new PFSS data."""
        self.pfssdata = pfssdata
        self.update_plot()


def widget_pfss_field(pfssdata=None):
    """
    Create and display PFSS field widget.
    
    Args:
        pfssdata: PFSS field line data
        
    Returns:
        PfssFieldWidget: Widget instance
    """
    
    widget = PfssFieldWidget(pfssdata)
    widget.show()
    return widget


if __name__ == "__main__":
    # Create synthetic data for testing
    data = np.random.randn(100, 100)
    widget = widget_pfss_field(data)
    print("PFSS Field Widget launched")