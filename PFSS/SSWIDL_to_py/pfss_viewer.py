"""
pfss_viewer.py - Interactive PFSS field line viewer (simplified Python version)

This is a simplified Python version of the IDL pfss_viewer widget.
The original IDL version was a complex widget-based interactive viewer.
This Python version provides basic visualization capabilities using matplotlib.

PURPOSE: Interactive visualization of PFSS magnetic field models and field lines

CALLING SEQUENCE:
   pfss_viewer(pfss_data=None, interactive=True)

INPUTS:
   pfss_data - PFSS data structure (optional, will use common block if not provided)

KEYWORD PARAMETERS:
   interactive - if True, enables interactive features
   save_plots - if True, saves plots to files
   plot_format - format for saved plots ('png', 'pdf', 'svg')

OUTPUTS:
   Interactive plots and optionally saved figure files

MODIFICATION HISTORY:
   Original IDL version - S.L.Freeland and others
   Python conversion - Simplified version for SSWIDL_to_py package
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.widgets as widgets
from matplotlib.colors import LinearSegmentedColormap
import warnings

# Import required modules from the converted package
from pfss_data_block import PfssDataBlock
from pfss_view_create import pfss_view_create, plot_field_lines
from pfss_draw_field import pfss_draw_field


class PfssViewer:
    """
    Simplified PFSS viewer class for interactive visualization.
    """
    
    def __init__(self, pfss_data=None):
        """
        Initialize the PFSS viewer.
        
        Args:
            pfss_data: PFSS data structure (optional)
        """
        
        self.pfss_data = pfss_data
        self.object_list = None
        self.current_view = 'field_lines'
        self.fig = None
        self.ax = None
        
        # Load data if not provided
        if self.pfss_data is None:
            self.load_pfss_data()
    
    def load_pfss_data(self):
        """Load PFSS data from common block."""
        try:
            data_block = PfssDataBlock()
            # Check if data is available
            if hasattr(data_block, 'br') and data_block.br is not None:
                self.pfss_data = data_block
                print("PFSS data loaded from common block")
            else:
                print("No PFSS data available in common block")
                self.pfss_data = None
        except Exception as e:
            print(f"Error loading PFSS data: {e}")
            self.pfss_data = None
    
    def create_view(self):
        """Create the view object."""
        self.object_list = pfss_view_create()
        if self.object_list is not None:
            print("View object created successfully")
        else:
            print("Failed to create view object")
    
    def show_field_lines(self, interactive=True):
        """
        Display field lines in 3D.
        
        Args:
            interactive (bool): Enable interactive features
        """
        
        if self.object_list is None:
            self.create_view()
        
        if self.object_list is None:
            print("No field line data available")
            return
        
        # Create the plot
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Plot field lines
        self.plot_field_lines_3d()
        
        # Add interactive controls if requested
        if interactive:
            self.add_interactive_controls()
        
        plt.show()
    
    def plot_field_lines_3d(self):
        """Plot field lines in 3D."""
        
        if not self.object_list or len(self.object_list['field_lines']) == 0:
            print("No field lines to plot")
            return
        
        self.ax.clear()
        
        # Plot each field line
        for i, (line, color) in enumerate(zip(self.object_list['field_lines'], 
                                             self.object_list['line_colors'])):
            if len(line) > 0:
                # Convert color to matplotlib format
                mpl_color = [c/255.0 for c in color]
                
                # Plot the line
                self.ax.plot(line[:, 0], line[:, 1], line[:, 2], 
                           color=mpl_color, linewidth=1.0)
        
        # Draw solar surface
        self.draw_solar_surface()
        
        # Set labels and title
        self.ax.set_xlabel('X (Solar Radii)')
        self.ax.set_ylabel('Y (Solar Radii)')
        self.ax.set_zlabel('Z (Solar Radii)')
        self.ax.set_title('PFSS Field Lines')
        
        # Set equal aspect ratio
        max_range = self.object_list['rmax']
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([-max_range, max_range])
    
    def draw_solar_surface(self):
        """Draw the solar surface."""
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax.plot_surface(x, y, z, alpha=0.3, color='yellow')
    
    def add_interactive_controls(self):
        """Add interactive controls to the plot."""
        
        # Add buttons for different views
        ax_button1 = plt.axes([0.02, 0.9, 0.1, 0.04])
        ax_button2 = plt.axes([0.02, 0.85, 0.1, 0.04])
        ax_button3 = plt.axes([0.02, 0.8, 0.1, 0.04])
        
        button1 = widgets.Button(ax_button1, 'Reset View')
        button2 = widgets.Button(ax_button2, 'Top View')
        button3 = widgets.Button(ax_button3, 'Side View')
        
        button1.on_clicked(self.reset_view)
        button2.on_clicked(self.top_view)
        button3.on_clicked(self.side_view)
        
        # Store buttons to prevent garbage collection
        self.buttons = [button1, button2, button3]
    
    def reset_view(self, event):
        """Reset to default view."""
        self.ax.view_init(elev=20, azim=45)
        self.fig.canvas.draw()
    
    def top_view(self, event):
        """Set top view (looking down at north pole)."""
        self.ax.view_init(elev=90, azim=0)
        self.fig.canvas.draw()
    
    def side_view(self, event):
        """Set side view."""
        self.ax.view_init(elev=0, azim=0)
        self.fig.canvas.draw()
    
    def show_magnetogram(self, projection='orthographic'):
        """
        Display the magnetogram.
        
        Args:
            projection (str): Projection type for display
        """
        
        try:
            # Create magnetogram plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # This would use pfss_draw_field or similar
            outim = pfss_draw_field(bcent=0, lcent=0, width=2.5, mag=2)
            
            if outim is not None:
                im = ax.imshow(outim, origin='lower', extent=[-2.5, 2.5, -2.5, 2.5])
                ax.set_xlabel('Solar Radii')
                ax.set_ylabel('Solar Radii')
                ax.set_title('PFSS Magnetogram')
                plt.colorbar(im, ax=ax)
                plt.show()
            else:
                print("No magnetogram data available")
                
        except Exception as e:
            print(f"Error creating magnetogram: {e}")
    
    def save_view(self, filename=None, format='png'):
        """
        Save the current view to a file.
        
        Args:
            filename (str): Output filename (optional)
            format (str): File format ('png', 'pdf', 'svg')
        """
        
        if self.fig is None:
            print("No figure to save")
            return
        
        if filename is None:
            filename = f'pfss_view.{format}'
        
        try:
            self.fig.savefig(filename, format=format, dpi=300, bbox_inches='tight')
            print(f"View saved to {filename}")
        except Exception as e:
            print(f"Error saving view: {e}")
    
    def print_info(self):
        """Print information about the current view."""
        
        if self.object_list is None:
            print("No view object available")
            return
        
        print('PFSS Viewer Information:')
        print('-' * 24)
        print(f'Number of field lines: {self.object_list["nlines"]}')
        print(f'Number of plotted lines: {len(self.object_list["field_lines"])}')
        print(f'Radial range: {self.object_list["rmin"]:.2f} to {self.object_list["rmax"]:.2f} R☉')
        
        # Count open/closed lines
        open_count = np.sum(self.object_list['open_status'] == 1)
        closed_count = np.sum(self.object_list['open_status'] == 0)
        negative_count = np.sum(self.object_list['open_status'] == -1)
        
        print(f'Open field lines (positive): {open_count}')
        print(f'Open field lines (negative): {negative_count}')
        print(f'Closed field lines: {closed_count}')


def pfss_viewer(pfss_data=None, interactive=True, save_plots=False, plot_format='png'):
    """
    Launch the PFSS viewer.
    
    Args:
        pfss_data: PFSS data structure (optional)
        interactive (bool): Enable interactive features
        save_plots (bool): Save plots to files
        plot_format (str): Format for saved plots
    
    Returns:
        PfssViewer: Viewer instance
    """
    
    # Create viewer instance
    viewer = PfssViewer(pfss_data)
    
    # Print information
    viewer.print_info()
    
    # Show field lines
    viewer.show_field_lines(interactive=interactive)
    
    # Save plots if requested
    if save_plots:
        viewer.save_view(format=plot_format)
    
    return viewer


def pfss_viewer_simple():
    """
    Simple non-interactive PFSS viewer.
    
    Returns:
        PfssViewer: Viewer instance
    """
    
    viewer = PfssViewer()
    
    if viewer.pfss_data is not None:
        # Create simple 3D plot
        object_list = pfss_view_create()
        if object_list is not None:
            plot_field_lines(object_list)
        
        # Show magnetogram
        viewer.show_magnetogram()
    else:
        print("No PFSS data available for viewing")
    
    return viewer


# For compatibility with IDL calling convention
def pfss_viewer_idl():
    """
    IDL-compatible wrapper for pfss_viewer.
    
    This launches the viewer using data from the common block.
    """
    return pfss_viewer()


# Example usage
def pfss_viewer_example():
    """
    Example usage of pfss_viewer.
    """
    
    print("PFSS Viewer Example")
    print("=" * 19)
    
    # Launch viewer
    viewer = pfss_viewer(interactive=True)
    
    print("Viewer launched successfully!")
    print("Use mouse to rotate the 3D view")
    print("Use buttons for different viewing angles")
    
    return viewer


# Widget-based interface (simplified)
class PfssViewerWidget:
    """
    Widget-based PFSS viewer (simplified version).
    """
    
    def __init__(self):
        self.viewer = None
        self.setup_gui()
    
    def setup_gui(self):
        """Set up the GUI interface."""
        
        # This would set up a more complex GUI using tkinter or PyQt
        # For now, just provide a placeholder
        print("GUI setup not implemented in this simplified version")
        print("Use pfss_viewer() for interactive matplotlib-based viewing")
    
    def launch(self):
        """Launch the widget viewer."""
        self.viewer = pfss_viewer(interactive=True)
        return self.viewer


# Utility functions
def create_synthetic_pfss_data():
    """
    Create synthetic PFSS data for testing the viewer.
    
    Returns:
        dict: Synthetic PFSS data structure
    """
    
    # Create basic synthetic data
    nlat, nlon, nr = 48, 96, 35
    
    # Create coordinate arrays
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon)
    rix = np.linspace(1.0, 2.5, nr)
    
    # Create synthetic field components
    br = np.random.randn(nlon, nlat, nr) * 10
    bth = np.random.randn(nlon, nlat, nr) * 5
    bph = np.random.randn(nlon, nlat, nr) * 5
    
    # Create synthetic field lines
    nlines = 100
    nstep = np.random.randint(10, 50, nlines)
    
    ptr = np.random.uniform(1.0, 2.5, (max(nstep), nlines))
    ptth = np.random.uniform(0, np.pi, (max(nstep), nlines))
    ptph = np.random.uniform(0, 2*np.pi, (max(nstep), nlines))
    
    # Create data structure
    pfss_data = {
        'br': br,
        'bth': bth,
        'bph': bph,
        'lat': lat,
        'lon': lon,
        'rix': rix,
        'nlat': nlat,
        'nlon': nlon,
        'nr': nr,
        'ptr': ptr,
        'ptth': ptth,
        'ptph': ptph,
        'nstep': nstep,
        'nlines': nlines
    }
    
    return pfss_data


if __name__ == "__main__":
    # Run example
    pfss_viewer_example()