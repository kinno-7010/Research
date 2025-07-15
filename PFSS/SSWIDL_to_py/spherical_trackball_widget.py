"""
Spherical Trackball Widget - Simplified Version

Given vector fieldline data on a spherical grid, this procedure opens an
interactive widget for display and manipulation of the fieldline data.

CALLING SEQUENCE:
    spherical_trackball_widget(sph_data, im_data=None, imsc=None, 
                              nolines=False, extra_objects=None)

INPUTS:
    sph_data = a SphericalFieldData object with fieldline trajectories
    im_data = optional SphericalImageData object for background image
    imsc = image scaling
    nolines = if True, ignore fieldline data
    extra_objects = extra objects to be added to the view

OUTPUTS:
    Interactive 3D widget for field visualization

NOTES:
    - This is a simplified version using matplotlib for basic 3D interaction
    - Full implementation would use Qt/Tkinter with OpenGL or VTK
    - For production use, consider integrating with PyQt + VTK or mayavi

MODIFICATION HISTORY:
    M.DeRosa - 18 Jan 2006 - created (IDL version)
    [... many modifications ...]
    Converted to Python - 2025 (simplified version)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Any
from spherical_field_data__define import SphericalFieldData
from spherical_image_data__define import SphericalImageData
from spherical_image_create import spherical_image_create


class SphericalTrackballWidget:
    """
    Simplified interactive 3D viewer for spherical field data.
    
    This is a basic implementation using matplotlib's 3D capabilities.
    For production use, consider PyQt + VTK or mayavi for full interactivity.
    """
    
    def __init__(self, sph_data: SphericalFieldData,
                 im_data: Optional[SphericalImageData] = None,
                 imsc: Optional[float] = None,
                 nolines: bool = False,
                 extra_objects: Optional[List[Any]] = None):
        """
        Initialize the trackball widget.
        
        Parameters:
        -----------
        sph_data : SphericalFieldData
            Field data with trajectories
        im_data : SphericalImageData, optional
            Background image data
        imsc : float, optional
            Image scaling
        nolines : bool
            Skip fieldline rendering
        extra_objects : list, optional
            Additional objects to render
        """
        
        self.sph_data = sph_data
        self.im_data = im_data
        self.imsc = imsc
        self.nolines = nolines
        self.extra_objects = extra_objects or []
        
        # Get field ranges
        if sph_data.rix is not None:
            self.rmax = np.max(sph_data.rix)
            self.rmin = np.min(sph_data.rix)
        else:
            self.rmax, self.rmin = 2.5, 1.0
        
        # Initialize figure
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set up initial view
        self._setup_view()
        
        # Render field
        self._render_field()
        
        # Add interactivity
        self._setup_interactivity()
    
    def _setup_view(self):
        """Set up the 3D view."""
        # Set axis limits
        lim = self.rmax * 1.1
        self.ax.set_xlim([-lim, lim])
        self.ax.set_ylim([-lim, lim])
        self.ax.set_zlim([-lim, lim])
        
        # Set labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Set initial viewing angle
        self.ax.view_init(elev=20, azim=45)
        
        # Set background color
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('gray')
        self.ax.yaxis.pane.set_edgecolor('gray')
        self.ax.zaxis.pane.set_edgecolor('gray')
        self.ax.xaxis.pane.set_alpha(0.1)
        self.ax.yaxis.pane.set_alpha(0.1)
        self.ax.zaxis.pane.set_alpha(0.1)
    
    def _render_field(self):
        """Render the field data."""
        # Draw spherical boundaries
        self._draw_spherical_boundaries()
        
        # Draw background image if available
        if self.im_data is not None:
            self._draw_background_image()
        elif self.sph_data.br is not None:
            # Create image from radial field
            im_data = spherical_image_create(self.sph_data.br[:, :, 0],
                                           self.sph_data.lon, self.sph_data.lat,
                                           radius=self.rmin)
            self._draw_background_image(im_data)
        
        # Draw fieldlines
        if not self.nolines:
            self._draw_fieldlines()
        
        # Draw extra objects
        self._draw_extra_objects()
    
    def _draw_spherical_boundaries(self):
        """Draw inner and outer spherical boundaries."""
        # Create sphere coordinates
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        
        # Inner sphere
        x_inner = self.rmin * np.outer(np.cos(u), np.sin(v))
        y_inner = self.rmin * np.outer(np.sin(u), np.sin(v))
        z_inner = self.rmin * np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.ax.plot_surface(x_inner, y_inner, z_inner, 
                           alpha=0.3, color='blue', label='Inner boundary')
        
        # Outer sphere
        x_outer = self.rmax * np.outer(np.cos(u), np.sin(v))
        y_outer = self.rmax * np.outer(np.sin(u), np.sin(v))
        z_outer = self.rmax * np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.ax.plot_surface(x_outer, y_outer, z_outer, 
                           alpha=0.2, color='red', label='Outer boundary')
    
    def _draw_background_image(self, im_data: Optional[SphericalImageData] = None):
        """Draw background image on spherical surface."""
        if im_data is None:
            im_data = self.im_data
        
        if im_data is None:
            return
        
        # Get image scaling
        imsc = self.imsc
        if imsc is None:
            imsc = np.max(np.abs(im_data.image))
        
        # Create spherical surface coordinates
        phi_grid, theta_grid = np.meshgrid(im_data.phi, im_data.theta, indexing='ij')
        
        # Convert to Cartesian
        r = im_data.rad
        x = r * np.sin(theta_grid) * np.cos(phi_grid)
        y = r * np.sin(theta_grid) * np.sin(phi_grid)
        z = r * np.cos(theta_grid)
        
        # Normalize image for color mapping
        image_norm = (im_data.image + imsc) / (2 * imsc)
        image_norm = np.clip(image_norm, 0, 1)
        
        # Plot surface with image texture
        self.ax.plot_surface(x, y, z, facecolors=plt.cm.RdBu_r(image_norm),
                           alpha=0.7, linewidth=0, antialiased=True)
    
    def _draw_fieldlines(self):
        """Draw fieldlines in 3D."""
        if (self.sph_data.nstep is None or self.sph_data.ptr is None):
            return
        
        nlines = len(self.sph_data.nstep)
        lines_drawn = 0
        
        for i in range(nlines):
            ns = self.sph_data.nstep[i]
            if ns <= 0:
                continue
            
            # Get line coordinates
            r_line = self.sph_data.ptr[:ns, i]
            th_line = self.sph_data.ptth[:ns, i]
            ph_line = self.sph_data.ptph[:ns, i]
            
            # Convert to Cartesian
            x_line = r_line * np.sin(th_line) * np.cos(ph_line)
            y_line = r_line * np.sin(th_line) * np.sin(ph_line)
            z_line = r_line * np.cos(th_line)
            
            # Determine line type (open/closed)
            is_open = (np.max(r_line) - self.rmin) / (self.rmax - self.rmin) > 0.99
            
            # Choose color
            if is_open:
                color = 'green'
            else:
                color = 'white'
            
            # Draw line
            self.ax.plot(x_line, y_line, z_line, color=color, linewidth=1.5, alpha=0.8)
            lines_drawn += 1
        
        print(f"Drew {lines_drawn} fieldlines")
    
    def _draw_extra_objects(self):
        """Draw extra objects."""
        # Placeholder for additional objects
        # In full implementation, this would handle various object types
        pass
    
    def _setup_interactivity(self):
        """Set up interactive controls."""
        # Add title and instructions
        self.ax.set_title('Spherical Field Viewer\n'
                         'Use mouse to rotate, zoom with scroll wheel\n'
                         'Press keys for additional controls')
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Add control panel
        self._add_control_panel()
    
    def _add_control_panel(self):
        """Add control panel with buttons."""
        # Create control panel
        ax_reset = plt.axes([0.02, 0.02, 0.1, 0.04])
        ax_save = plt.axes([0.02, 0.07, 0.1, 0.04])
        ax_rotate = plt.axes([0.02, 0.12, 0.1, 0.04])
        
        from matplotlib.widgets import Button
        
        # Reset view button
        self.btn_reset = Button(ax_reset, 'Reset View')
        self.btn_reset.on_clicked(self._reset_view)
        
        # Save image button
        self.btn_save = Button(ax_save, 'Save Image')
        self.btn_save.on_clicked(self._save_image)
        
        # Auto-rotate button
        self.btn_rotate = Button(ax_rotate, 'Auto Rotate')
        self.btn_rotate.on_clicked(self._toggle_auto_rotate)
        
        self.auto_rotate = False
    
    def _on_key_press(self, event):
        """Handle key press events."""
        if event.key == 'r':
            self._reset_view(None)
        elif event.key == 's':
            self._save_image(None)
        elif event.key == 'a':
            self._toggle_auto_rotate(None)
        elif event.key == 'h':
            self._show_help()
    
    def _reset_view(self, event):
        """Reset the 3D view."""
        self.ax.view_init(elev=20, azim=45)
        self.fig.canvas.draw()
    
    def _save_image(self, event):
        """Save the current view."""
        filename = 'spherical_field_view.png'
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved view to {filename}")
    
    def _toggle_auto_rotate(self, event):
        """Toggle auto-rotation."""
        self.auto_rotate = not self.auto_rotate
        if self.auto_rotate:
            self._start_auto_rotate()
        print(f"Auto-rotate: {'ON' if self.auto_rotate else 'OFF'}")
    
    def _start_auto_rotate(self):
        """Start auto-rotation animation."""
        # Simple animation loop
        import matplotlib.animation as animation
        
        def animate(frame):
            if self.auto_rotate:
                self.ax.view_init(elev=20, azim=frame * 2)
                return self.ax.collections
            return []
        
        self.ani = animation.FuncAnimation(self.fig, animate, interval=50, blit=False)
        self.fig.canvas.draw()
    
    def _show_help(self):
        """Show help information."""
        help_text = """
        Spherical Field Viewer - Controls:
        
        Mouse:
        - Left drag: Rotate view
        - Right drag: Pan view
        - Scroll: Zoom in/out
        
        Keyboard:
        - 'r': Reset view
        - 's': Save image
        - 'a': Toggle auto-rotation
        - 'h': Show this help
        
        Buttons:
        - Reset View: Return to default view
        - Save Image: Save current view as PNG
        - Auto Rotate: Toggle automatic rotation
        """
        print(help_text)
    
    def show(self):
        """Display the widget."""
        plt.tight_layout()
        plt.show()


def spherical_trackball_widget(sph_data: SphericalFieldData,
                              im_data: Optional[SphericalImageData] = None,
                              imsc: Optional[float] = None,
                              nolines: bool = False,
                              extra_objects: Optional[List[Any]] = None):
    """
    Launch interactive 3D viewer for spherical field data.
    
    Parameters:
    -----------
    sph_data : SphericalFieldData
        Field data with trajectories
    im_data : SphericalImageData, optional
        Background image data
    imsc : float, optional
        Image scaling
    nolines : bool
        Skip fieldline rendering
    extra_objects : list, optional
        Additional objects to render
    """
    
    widget = SphericalTrackballWidget(sph_data, im_data, imsc, nolines, extra_objects)
    widget.show()


if __name__ == "__main__":
    # Example usage
    from spherical_field_data__define import SphericalFieldData
    from spherical_field_start_coord import spherical_field_start_coord
    from spherical_trace_field import spherical_trace_field
    
    # Create sample data
    sph_data = SphericalFieldData()
    
    # Set up grid
    nr, nlat, nlon = 15, 25, 30
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
    print("Launching interactive 3D viewer...")
    print("Use mouse to rotate, keyboard shortcuts for controls")
    
    # Launch interactive viewer
    spherical_trackball_widget(sph_data, imsc=100)