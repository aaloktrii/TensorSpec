# File: tensorspec/plotting/backends/matplotlib_engine.py
import matplotlib
matplotlib.use('qtagg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class MatplotlibCrystalBackend:
    """
    Crash-proof CPU rendering backend for legacy systems.
    Uses Matplotlib to simulate 3D graphics safely without VTK/OpenGL.
    """
    def __init__(self, parent=None):
        self.figure = Figure(facecolor="#1e1e24")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_facecolor("#1e1e24")
        self.ax.axis('off')
        
        # Duck-typing: Pretend to be a PyVista plotter so the GUI doesn't crash
        self.plotter = self.canvas
        self.plotter.render = self.canvas.draw_idle

        

    def clear_scene(self):
        self.ax.clear()
        self.ax.set_facecolor("#1e1e24")
        self.ax.axis('off')

    def draw_atoms(self, supercell, active_colors: dict, scale_mod: float = 0.5, is_shiny: bool = False, erased_atoms: set = None):
        if erased_atoms is None: erased_atoms = set()
        
        for i, site in enumerate(supercell):
            if i in erased_atoms: continue
            tag = supercell.site_properties.get("layer_tag", [site.specie.symbol] * len(supercell))[i]
            color = active_colors.get(tag, "#008080")
            rad = (site.specie.atomic_radius if site.specie.atomic_radius else 1.0) * scale_mod
            
            x, y, z = site.coords
            # Multiply radius to look decent in matplotlib scatter sizing
            self.ax.scatter(x, y, z, s=(rad * 400), c=color, edgecolors='black', depthshade=True)

    def draw_bonds(self, supercell, active_colors: dict, cyl_radius: float = 0.1, thresh_multiplier: float = 1.15, is_shiny: bool = False, erased_bonds: set = None, erased_atoms: set = None):
        if len(supercell) > 2000: return # Protect CPU from freezing on massive arrays
        if erased_bonds is None: erased_bonds = set()
        if erased_atoms is None: erased_atoms = set()

        bond_color = active_colors.get("Bonds", "#FFFFFF")
        for i in range(len(supercell)):
            if i in erased_atoms: continue
            for j in range(i + 1, len(supercell)):
                if j in erased_atoms or (i, j) in erased_bonds: continue
                
                a, b = supercell[i], supercell[j]
                dist = np.linalg.norm(a.coords - b.coords)
                ra = a.specie.atomic_radius if a.specie.atomic_radius else 1.2
                rb = b.specie.atomic_radius if b.specie.atomic_radius else 1.2
                
                if 0.5 < dist <= (ra + rb) * thresh_multiplier:
                    self.ax.plot([a.coords[0], b.coords[0]], 
                                 [a.coords[1], b.coords[1]], 
                                 [a.coords[2], b.coords[2]], 
                                 color=bond_color, linewidth=2)

    def draw_axes(self, conventional_matrix=None, primitive_matrix=None):
        """Draws the crystallographic a, b, c axes vectors at the origin."""
        import numpy as np
        
        # Determine visual scale based on the lattice size
        scale = 5.0
        if conventional_matrix is not None:
            scale = np.max(np.linalg.norm(conventional_matrix, axis=1)) * 0.5
        elif primitive_matrix is not None:
            scale = np.max(np.linalg.norm(primitive_matrix, axis=1)) * 0.5

        axes = np.eye(3) * scale
        colors = ['#ff4444', '#44ff44', '#4444ff'] # Red for 'a', Green for 'b', Blue for 'c'
        labels = ['a', 'b', 'c']

        for i in range(3):
            vec = axes[i]
            # Draw vector line
            self.ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]], color=colors[i], linewidth=2.5)
            # Add text label slightly offset from the tip
            self.ax.text(vec[0]*1.1, vec[1]*1.1, vec[2]*1.1, labels[i], color=colors[i], fontsize=14, fontweight='bold')

    def draw_lattice_boxes(self, conventional_matrix=None, primitive_matrix=None):
        """Draws the 3D parallelepiped bounding boxes for the unit cells."""
        import numpy as np

        def _draw_box(matrix, color, linestyle):
            if matrix is None: return
            
            # Extract the 3 basis vectors (rows of the matrix)
            v1, v2, v3 = matrix[0], matrix[1], matrix[2]
            
            # Generate the 8 corners of the parallelepiped
            corners = [
                np.array([0, 0, 0]), 
                v1, 
                v2, 
                v3,
                v1 + v2, 
                v1 + v3, 
                v2 + v3, 
                v1 + v2 + v3
            ]
            
            # Map the 12 connecting edges between the corners
            edges = [
                (0, 1), (0, 2), (0, 3),
                (1, 4), (1, 5),
                (2, 4), (2, 6),
                (3, 5), (3, 6),
                (4, 7), (5, 7), (6, 7)
            ]
            
            # Render the lines
            for p1_idx, p2_idx in edges:
                p1, p2 = corners[p1_idx], corners[p2_idx]
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                             color=color, linestyle=linestyle, linewidth=1.5, alpha=0.7)

        # Draw Conventional Cell in solid white, Primitive in dashed cyan
        if conventional_matrix is not None:
            _draw_box(conventional_matrix, color='white', linestyle='-')
        if primitive_matrix is not None:
            _draw_box(primitive_matrix, color='#00FFFF', linestyle='--')


    def draw_moire_envelope(self, m_moire, z_min, z_max):
        """Draws the 2D Moiré superlattice bounding box in Matplotlib."""
        import numpy as np
        
        # Extract the 2D basis vectors for the Moiré cell
        v1 = np.array([m_moire[0][0], m_moire[0][1], 0])
        v2 = np.array([m_moire[1][0], m_moire[1][1], 0])
        
        # Calculate the 4 corners of the base
        p0 = np.array([0, 0, 0])
        p1 = v1
        p2 = v1 + v2
        p3 = v2
        
        # Generate Bottom and Top face coordinates
        corners_bottom = [p0 + [0, 0, z_min], p1 + [0, 0, z_min], p2 + [0, 0, z_min], p3 + [0, 0, z_min], p0 + [0, 0, z_min]]
        corners_top = [p0 + [0, 0, z_max], p1 + [0, 0, z_max], p2 + [0, 0, z_max], p3 + [0, 0, z_max], p0 + [0, 0, z_max]]
        
        # Draw bottom and top rings
        cb_x, cb_y, cb_z = zip(*corners_bottom)
        ct_x, ct_y, ct_z = zip(*corners_top)
        
        self.ax.plot(cb_x, cb_y, cb_z, color="#00FFFF", linestyle="--", linewidth=2)
        self.ax.plot(ct_x, ct_y, ct_z, color="#00FFFF", linestyle="--", linewidth=2)
        
        # Draw vertical pillars to complete the wireframe box
        for b, t in zip(corners_bottom[:-1], corners_top[:-1]):
            self.ax.plot([b[0], t[0]], [b[1], t[1]], [b[2], t[2]], color="#00FFFF", linestyle="--", linewidth=2)

    def set_camera_preset(self, preset: str):
        if preset == 'x': self.ax.view_init(elev=0, azim=0)
        elif preset == 'y': self.ax.view_init(elev=0, azim=90)
        elif preset == 'z': self.ax.view_init(elev=90, azim=-90)
        elif preset == 'iso': self.ax.view_init(elev=35, azim=45)
        self.canvas.draw_idle()
    
    def draw_brillouin_zone(self, bz_points, bz_simplices, style_idx=0, edges=None):
        """Renders the BZ using Line3D collections to bypass Poly3DCollection projection crashes.

        ``edges`` is accepted for API parity with the PyVista backend (unused here).
        """
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        import numpy as np

        # Create a set of unique line segments (edges) from the simplices
        edges = set()
        for simplex in bz_simplices:
            for i in range(len(simplex)):
                p1, p2 = simplex[i], simplex[(i + 1) % len(simplex)]
                edges.add(tuple(sorted((p1, p2))))
        
        line_segments = [[bz_points[e[0]], bz_points[e[1]]] for e in edges]
        
        # Draw edges as a Line3DCollection (much more stable than polygons)
        line_col = Line3DCollection(line_segments, colors="#FF00FF", linewidths=2.0)
        self.ax.add_collection3d(line_col)
            
        # Draw Reciprocal Axes safely
        arrow_scale = np.max(bz_points) * 1.25
        colors = ["#ff6666", "#66ff66", "#6666ff"]
        labels = ["k_x", "k_y", "k_z"]
        
        for i, vec in enumerate(np.eye(3) * arrow_scale):
            self.ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=colors[i], arrow_length_ratio=0.1)
            self.ax.text(vec[0]*1.15, vec[1]*1.15, vec[2]*1.15, labels[i], color=colors[i], fontsize=12)

    def draw_surface_bz(self, base_plane, hover_plane, simplices, proj_lines):
        """Draws the dual-plane projection and dashed lines in Matplotlib."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # 1. Base Plane
        base_faces = [base_plane[s] for s in simplices]
        poly_base = Poly3DCollection(base_faces, alpha=0.15, facecolor="#00FFFF", edgecolor="gray", linewidths=0.5)
        self.ax.add_collection3d(poly_base)
        
        # 2. Hover Plane
        hover_faces = [hover_plane[s] for s in simplices]
        poly_hover = Poly3DCollection(hover_faces, alpha=0.5, facecolor="#00FFFF", edgecolor="white", linewidths=1.5)
        self.ax.add_collection3d(poly_hover)
        
        # 3. Dashed Projection Lines
        for p1, p2 in proj_lines:
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="yellow", linestyle="--", linewidth=1.5)
    
    def draw_polyhedra(self, supercell, active_colors: dict, is_shiny: bool = False):
        """
        Compatibility fallback for polyhedra style in CPU mode. 
        Draws atoms and bonds instead to prevent heavy Matplotlib spatial projection crashes.
        """
        # Safely fall back to ball-and-stick rendering
        self.draw_atoms(supercell, active_colors, scale_mod=0.5, is_shiny=is_shiny)
        self.draw_bonds(supercell, active_colors, is_shiny=is_shiny)