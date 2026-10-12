# File: tensorspec/gui/components/crystal_panel.py
import os
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QFileDialog, QLabel, QSpinBox, QDoubleSpinBox,
                               QComboBox, QColorDialog, QCheckBox, QTabWidget,
                               QGroupBox, QGridLayout, QSlider, QScrollArea, QFrame, QMessageBox,
                               QRadioButton)
from PySide6.QtCore import Qt
from pymatgen.core import Structure

from tensorspec.core.crystallography import CrystalEngine
from tensorspec.core.io.exporters import SceneExporter
import platform

is_legacy_mac = (platform.system() == "Darwin" and platform.machine() == "x86_64")


def _scroll_page(*widgets):
    """Wrap widgets in a scroll area so dense controls stay usable."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(4, 4, 4, 4)
    for w in widgets:
        layout.addWidget(w)
    layout.addStretch()
    scroll.setWidget(page)
    return scroll


class TabViewEdit(QWidget):
    """Modular Tab 1: CIF load / style / camera / export split into sub-tabs."""
    def __init__(self, main_suite):
        super().__init__()
        self.main_suite = main_suite
        self.init_ui()

    def init_ui(self):
        main_tab_layout = QVBoxLayout(self)
        main_tab_layout.setContentsMargins(0, 0, 0, 0)

        sub = QTabWidget()
        main_tab_layout.addWidget(sub)

        # --- Structure page: load + geometry + style + colors + render ---
        struct_box = QWidget()
        struct_layout = QVBoxLayout(struct_box)
        struct_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_load = QPushButton("📂 Load CIF File")
        self.btn_load.clicked.connect(self.handle_load_cif)
        struct_layout.addWidget(self.btn_load)

        self.lbl_file = QLabel("No file loaded"); self.lbl_file.setAlignment(Qt.AlignCenter)
        struct_layout.addWidget(self.lbl_file)

        self.lbl_sym = QLabel("Space Group: N/A")
        self.lbl_sym.setStyleSheet("color: #aaaaaa; font-style: italic;")
        struct_layout.addWidget(self.lbl_sym)

        struct_layout.addWidget(QLabel("Graphics Backend:"))
        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["CPU Safe Mode (Matplotlib)", "GPU Fast Mode (PyVista)"])
        self.combo_backend.setCurrentIndex(0 if is_legacy_mac else 1)
        self.combo_backend.currentIndexChanged.connect(self.switch_backend)
        struct_layout.addWidget(self.combo_backend)

        group_geom = QGroupBox("Geometry & Scaling")
        geom_layout = QVBoxLayout(group_geom)

        self.radio_conv = QRadioButton("Conventional Basis")
        self.radio_prim = QRadioButton("Primitive Basis")
        self.radio_conv.setChecked(True)
        self.radio_conv.toggled.connect(self.handle_draw)
        basis_layout = QHBoxLayout()
        basis_layout.addWidget(self.radio_conv); basis_layout.addWidget(self.radio_prim)
        geom_layout.addLayout(basis_layout)

        sc_layout = QHBoxLayout()
        sc_layout.addWidget(QLabel("Supercell (X,Y,Z):"))
        self.spin_scx = QSpinBox(); self.spin_scx.setValue(1); self.spin_scx.setRange(1, 20)
        self.spin_scy = QSpinBox(); self.spin_scy.setValue(1); self.spin_scy.setRange(1, 20)
        self.spin_scz = QSpinBox(); self.spin_scz.setValue(1); self.spin_scz.setRange(1, 20)
        sc_layout.addWidget(self.spin_scx); sc_layout.addWidget(self.spin_scy); sc_layout.addWidget(self.spin_scz)
        geom_layout.addLayout(sc_layout)

        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Atom Radius Scale:"))
        self.spin_radius = QDoubleSpinBox(); self.spin_radius.setRange(0.1, 3.0); self.spin_radius.setValue(0.5); self.spin_radius.setSingleStep(0.1)
        scale_layout.addWidget(self.spin_radius)
        scale_layout.addWidget(QLabel("Bond Thick:"))
        self.spin_bond_thick = QDoubleSpinBox(); self.spin_bond_thick.setRange(0.01, 1.0); self.spin_bond_thick.setValue(0.10); self.spin_bond_thick.setSingleStep(0.02)
        scale_layout.addWidget(self.spin_bond_thick)
        geom_layout.addLayout(scale_layout)
        struct_layout.addWidget(group_geom)

        group_render = QGroupBox("Styles & Rendering")
        render_layout = QVBoxLayout(group_render)

        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("Connections:"))
        self.combo_style = QComboBox()
        self.combo_style.addItems(["Bonds (Sticks)", "Polyhedra (Planes)", "None"])
        self.combo_style.currentIndexChanged.connect(self.main_suite.refresh_render)
        style_layout.addWidget(self.combo_style)
        style_layout.addWidget(QLabel("Bond Thresh:"))
        self.spin_bond_thresh = QDoubleSpinBox(); self.spin_bond_thresh.setRange(0.1, 5.0); self.spin_bond_thresh.setValue(1.15)
        style_layout.addWidget(self.spin_bond_thresh)
        render_layout.addLayout(style_layout)

        chk_layout = QHBoxLayout()
        self.chk_shiny = QCheckBox("PBR Shiny"); self.chk_shiny.stateChanged.connect(self.main_suite.refresh_render)
        self.chk_axes = QCheckBox("Show Axes"); self.chk_axes.setChecked(True); self.chk_axes.stateChanged.connect(self.main_suite.refresh_render)
        self.chk_show_conventional = QCheckBox("Show Conv. Box"); self.chk_show_conventional.setChecked(True); self.chk_show_conventional.stateChanged.connect(self.main_suite.refresh_render)
        self.chk_show_primitive = QCheckBox("Show Prim. Box"); self.chk_show_primitive.stateChanged.connect(self.main_suite.refresh_render)
        chk_layout.addWidget(self.chk_shiny); chk_layout.addWidget(self.chk_axes)
        chk_layout.addWidget(self.chk_show_conventional); chk_layout.addWidget(self.chk_show_primitive)
        render_layout.addLayout(chk_layout)
        struct_layout.addWidget(group_render)

        self.chk_edit_mode = QCheckBox("Enable Interactive Eraser Brush")
        self.chk_edit_mode.setStyleSheet("color: #d9534f; font-weight: bold;")
        self.chk_edit_mode.stateChanged.connect(self.toggle_edit_mode)
        struct_layout.addWidget(self.chk_edit_mode)

        self.group_colors = QGroupBox("Dynamic Element Colors")
        self.colors_layout = QGridLayout(self.group_colors)
        struct_layout.addWidget(self.group_colors)

        self.btn_draw = QPushButton("🎨 Render Structure")
        self.btn_draw.setStyleSheet("background-color: #2b5c8f; color: white; font-weight: bold; padding: 6px;")
        self.btn_draw.clicked.connect(self.handle_draw)
        struct_layout.addWidget(self.btn_draw)

        sub.addTab(_scroll_page(struct_box), "Structure")

        # --- Camera page ---
        self.group_camera = QGroupBox("Camera & Projection")
        self.cam_layout = QVBoxLayout(self.group_camera)

        self.combo_projection = QComboBox()
        self.combo_projection.addItems(["Perspective Projection", "Orthogonal Projection"])
        self.combo_projection.currentIndexChanged.connect(self.handle_camera_update)
        self.cam_layout.addWidget(self.combo_projection)

        quick_view_layout = QHBoxLayout()
        quick_view_layout.addWidget(QLabel("Quick View:"))
        for axis, label in [('x', '+a'), ('y', '+b'), ('z', '+c'), ('iso', '111')]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, a=axis: self.main_suite.renderer.set_camera_preset(a) if hasattr(self.main_suite.renderer, 'set_camera_preset') else None)
            quick_view_layout.addWidget(btn)
        self.cam_layout.addLayout(quick_view_layout)

        azel_layout = QHBoxLayout()
        azel_layout.addWidget(QLabel("Azimuth:")); self.spin_azimuth = QDoubleSpinBox(); self.spin_azimuth.setRange(-360.0, 360.0); self.spin_azimuth.setSingleStep(5.0)
        self.spin_azimuth.valueChanged.connect(self.update_azel)
        azel_layout.addWidget(self.spin_azimuth)
        azel_layout.addWidget(QLabel("Elevation:")); self.spin_elevation = QDoubleSpinBox(); self.spin_elevation.setRange(-90.0, 90.0); self.spin_elevation.setSingleStep(5.0)
        self.spin_elevation.valueChanged.connect(self.update_azel)
        azel_layout.addWidget(self.spin_elevation)
        self.cam_layout.addLayout(azel_layout)

        self.group_cryst = QGroupBox("Crystallography Tools")
        self.cryst_layout = QVBoxLayout(self.group_cryst)

        hkl_layout = QHBoxLayout()
        hkl_layout.addWidget(QLabel("View [h k l]:"))
        self.spin_h = QDoubleSpinBox(); self.spin_h.setRange(-10, 10); self.spin_k = QDoubleSpinBox(); self.spin_k.setRange(-10, 10); self.spin_l = QDoubleSpinBox(); self.spin_l.setRange(-10, 10)
        hkl_layout.addWidget(self.spin_h); hkl_layout.addWidget(self.spin_k); hkl_layout.addWidget(self.spin_l)
        self.btn_align_hkl = QPushButton("Align"); self.btn_align_hkl.clicked.connect(self.align_to_hkl)
        hkl_layout.addWidget(self.btn_align_hkl)
        self.cryst_layout.addLayout(hkl_layout)

        plane_layout = QHBoxLayout()
        self.chk_show_plane = QCheckBox("Show Cut Plane")
        self.chk_show_plane.stateChanged.connect(self.toggle_cut_plane)
        plane_layout.addWidget(self.chk_show_plane)
        self.combo_plane_color = QComboBox(); self.combo_plane_color.addItems(["cyan", "magenta", "yellow", "white", "gray"])
        self.combo_plane_color.currentTextChanged.connect(self.update_cut_plane)
        plane_layout.addWidget(self.combo_plane_color)
        self.combo_plane_orient = QComboBox(); self.combo_plane_orient.addItems(["Lock to Camera", "Lock to [h k l]"])
        self.combo_plane_orient.currentIndexChanged.connect(self.lock_plane_normal)
        plane_layout.addWidget(self.combo_plane_orient)
        self.cryst_layout.addLayout(plane_layout)

        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Depth:")); self.slider_plane_depth = QSlider(Qt.Horizontal); self.slider_plane_depth.setRange(-100, 100); self.slider_plane_depth.setValue(0)
        self.slider_plane_depth.valueChanged.connect(self.update_cut_plane)
        depth_layout.addWidget(self.slider_plane_depth)
        self.cryst_layout.addLayout(depth_layout)

        sub.addTab(_scroll_page(self.group_camera, self.group_cryst), "Camera")

        # --- Export page ---
        self.group_export = QGroupBox("Export Elements")
        exp_layout = QVBoxLayout(self.group_export)
        exp_chk = QHBoxLayout()
        self.chk_exp_atoms = QCheckBox("Atoms/Bonds"); self.chk_exp_atoms.setChecked(True)
        self.chk_exp_cell = QCheckBox("Unit Cell"); self.chk_exp_cell.setChecked(True)
        self.chk_exp_bz = QCheckBox("Brillouin Zone"); self.chk_exp_bz.setChecked(True)
        exp_chk.addWidget(self.chk_exp_atoms); exp_chk.addWidget(self.chk_exp_cell); exp_chk.addWidget(self.chk_exp_bz)
        exp_layout.addLayout(exp_chk)

        btn_layout = QHBoxLayout()
        self.btn_export_max = QPushButton("Export 3ds Max"); self.btn_export_max.setStyleSheet("background-color: #0F6A8B; color: white; font-weight: bold;"); self.btn_export_max.clicked.connect(self.export_scripts)
        self.btn_export_blend = QPushButton("Export Blender"); self.btn_export_blend.setStyleSheet("background-color: #E87D0D; color: white; font-weight: bold;"); self.btn_export_blend.clicked.connect(self.export_scripts)
        btn_layout.addWidget(self.btn_export_max); btn_layout.addWidget(self.btn_export_blend)
        exp_layout.addLayout(btn_layout)

        export_actions = QWidget()
        export_actions_layout = QVBoxLayout(export_actions)
        export_actions_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_save = QPushButton("🖼 Save High-Res Image")
        self.btn_save.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 6px;")
        self.btn_save.clicked.connect(self.save_image)
        export_actions_layout.addWidget(self.btn_save)

        self.btn_export_cif = QPushButton("💾 Export Structure to CIF")
        self.btn_export_cif.setStyleSheet("background-color: #0F6A8B; color: white; font-weight: bold; padding: 6px;")
        self.btn_export_cif.clicked.connect(self.export_cif)
        export_actions_layout.addWidget(self.btn_export_cif)

        sub.addTab(_scroll_page(self.group_export, export_actions), "Export")

    # --- LOGIC METHODS ---
    def handle_load_cif(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open CIF', '', "CIF files (*.cif)")
        if not fname: return
        try:
            import warnings
            # Silences the specific pymatgen CIF parsing warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.main_suite.current_structure = Structure.from_file(fname)
            
            filename_only = fname.split('/')[-1]
            self.lbl_file.setText(filename_only)
            self.main_suite.current_filename = filename_only.replace('.cif', '')
            
            sym_info = CrystalEngine.get_symmetry_info(self.main_suite.current_structure)
            self.lbl_sym.setText(f"Space Group: {sym_info['spacegroup']} | V_conv = {sym_info['volume_ratio']}× V_prim")
            
            unique_els = sorted(list(set([s.specie.symbol for s in self.main_suite.current_structure])))
            self.build_color_panel(unique_els)
            
            if hasattr(self.main_suite, 'combo_cdw_el'):
                self.main_suite.combo_cdw_el.clear()
                self.main_suite.combo_cdw_el.addItem("All Elements")
                self.main_suite.combo_cdw_el.addItems(unique_els)
                
            if self.main_suite.workspace:
                self.main_suite.workspace[fname.split('/')[-1]] = {"type": "Structure", "data": self.main_suite.current_structure}
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to parse CIF: {e}")

    def build_color_panel(self, elements: list):
        while self.colors_layout.count():
            item = self.colors_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        for idx, el in enumerate(elements + ["Bonds"]):
            color = self.main_suite.active_colors.get(el, "#008080")
            lbl = QLabel(f"{el}:")
            btn = QPushButton()
            btn.setStyleSheet(f"background-color: {color}; border: 1px solid white;")
            btn.clicked.connect(lambda checked, key=el, b=btn: self.pick_color(key, b))
            self.colors_layout.addWidget(lbl, idx, 0)
            self.colors_layout.addWidget(btn, idx, 1)

    def pick_color(self, key: str, button: QPushButton):
        color = QColorDialog.getColor()
        if color.isValid():
            hex_str = color.name()
            self.main_suite.active_colors[key] = hex_str
            button.setStyleSheet(f"background-color: {hex_str}; border: 1px solid white;")
            self.main_suite.refresh_render()

    def handle_draw(self):
        if not getattr(self.main_suite, 'current_structure', None): return
        
        base_struct = self.main_suite.current_structure
        
        # 1. Convert to primitive BEFORE multiplying if the Primitive radio button is active
        if self.radio_prim.isChecked():
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sga = SpacegroupAnalyzer(base_struct)
                primitive_struct = sga.find_primitive()
                if primitive_struct:
                    base_struct = primitive_struct
            except Exception as e:
                print(f"Primitive conversion failed, falling back to conventional: {e}")
                
        # 2. Build the supercell
        self.main_suite.active_supercell = base_struct * (self.spin_scx.value(), self.spin_scy.value(), self.spin_scz.value())
        
        # 3. Reset eraser caches because the underlying geometry just changed
        self.main_suite.erased_atoms = set()
        self.main_suite.erased_bonds = set()
        
        # 4. Push to renderer
        self.main_suite.refresh_render()
        if hasattr(self.main_suite.renderer, 'set_camera_preset'):
            self.main_suite.renderer.set_camera_preset('iso')

    def switch_backend(self, index):
        if index == 1 and is_legacy_mac:
            reply = QMessageBox.warning(
                self, "Hardware Warning", 
                "Your older MacBook (Intel/OCLP) does not support the modern Metal drivers required for PyVista.\n\nSwitching to GPU Fast Mode will likely cause a crash. Proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                self.combo_backend.blockSignals(True)
                self.combo_backend.setCurrentIndex(0)
                self.combo_backend.blockSignals(False)
                return

        self.main_suite.viewer_stack.setCurrentIndex(index)
        self.main_suite.renderer = self.main_suite.renderer_cpu if index == 0 else self.main_suite.renderer_gpu
        self.handle_draw()

    def handle_camera_update(self):
        is_ortho = self.combo_projection.currentIndex() == 1
        
        # 1. Matplotlib Support
        if hasattr(self.main_suite.renderer, 'ax'):
            self.main_suite.renderer.ax.set_proj_type('ortho' if is_ortho else 'persp')
            if hasattr(self.main_suite.renderer, 'canvas'):
                self.main_suite.renderer.canvas.draw_idle()
                
        # 2. PyVista Support
        if hasattr(self.main_suite.renderer, 'plotter') and hasattr(self.main_suite.renderer.plotter, 'camera'):
            if is_ortho:
                self.main_suite.renderer.plotter.camera.enable_parallel_projection()
            else:
                self.main_suite.renderer.plotter.camera.disable_parallel_projection()
            
            # Support both rendering styles safely
            if hasattr(self.main_suite.renderer.plotter, 'render'):
                self.main_suite.renderer.plotter.render()

    def toggle_edit_mode(self, *args):
        is_erasing = self.chk_edit_mode.isChecked()
        
        # 1. Matplotlib Backend Support
        if self.combo_backend.currentIndex() == 0:
            if hasattr(self.main_suite.renderer, 'ax'):
                if is_erasing:
                    self.main_suite.renderer.ax.disable_mouse_rotation()
                    QMessageBox.information(self, "Eraser Mode", "Matplotlib eraser is active (Click to delete). GPU mode is recommended for brush sweeps!")
                else:
                    self.main_suite.renderer.ax.mouse_init()
                    
        # 2. PyVista GPU Backend Support
        elif self.combo_backend.currentIndex() == 1:
            if not hasattr(self.main_suite.renderer, 'plotter'): return
            
            if is_erasing:
                self.is_erasing = False
                
                # Lock camera rotation during left-click drag
                self.main_suite.renderer.plotter.enable_trackball_actor_style()
                
                # Attach continuous brush VTK listeners
                self.press_obs = self.main_suite.renderer.plotter.iren.add_observer("LeftButtonPressEvent", self.start_erase)
                self.move_obs = self.main_suite.renderer.plotter.iren.add_observer("MouseMoveEvent", self.do_erase)
                self.release_obs = self.main_suite.renderer.plotter.iren.add_observer("LeftButtonReleaseEvent", self.stop_erase)
                self.right_obs = self.main_suite.renderer.plotter.iren.add_observer("RightButtonPressEvent", self.exit_eraser_mode)
                
                QMessageBox.information(self, "Eraser Mode", "Left-click and drag to sweep-delete atoms/bonds. Right-click to exit.")
            else:
                self.main_suite.renderer.plotter.enable_trackball_style()
                
                # Safely detach brush listeners
                if hasattr(self, 'press_obs'):
                    self.main_suite.renderer.plotter.iren.remove_observer(self.press_obs)
                    self.main_suite.renderer.plotter.iren.remove_observer(self.move_obs)
                    self.main_suite.renderer.plotter.iren.remove_observer(self.release_obs)
                    self.main_suite.renderer.plotter.iren.remove_observer(self.right_obs)
                    del self.press_obs; del self.move_obs; del self.release_obs; del self.right_obs
                
                self.main_suite.renderer.plotter.update()

    def start_erase(self, obj, event):
        self.is_erasing = True
        self.do_erase(obj, event)

    def stop_erase(self, obj, event):
        self.is_erasing = False

    def do_erase(self, obj, event):
        """Sweeps and deletes any atom or bond using 3D Spatial GPU Intersection."""
        if not getattr(self, 'is_erasing', False) or not self.chk_edit_mode.isChecked(): 
            return
            
        import vtk
        click_pos = obj.GetEventPosition()
        
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005) 
        picker.Pick(click_pos[0], click_pos[1], 0, self.main_suite.renderer.plotter.renderer)
        
        if picker.GetActor():
            pick_pos = picker.GetPickPosition()
            erased_something = False
            
            # Check for Atom collisions
            if hasattr(self.main_suite.renderer, 'atom_tree') and self.main_suite.renderer.atom_tree is not None:
                dist, idx = self.main_suite.renderer.atom_tree.query(pick_pos)
                if dist < 1.0: 
                    if idx not in self.main_suite.erased_atoms:
                        self.main_suite.erased_atoms.add(idx)
                        erased_something = True
                        
            # Check for Bond collisions
            if not erased_something and hasattr(self.main_suite.renderer, 'bond_tree') and self.main_suite.renderer.bond_tree is not None:
                dist, idx = self.main_suite.renderer.bond_tree.query(pick_pos)
                if dist < 0.6: 
                    pair = self.main_suite.renderer.bond_pairs_list[idx]
                    if pair not in self.main_suite.erased_bonds:
                        self.main_suite.erased_bonds.add(pair)
                        erased_something = True
                        
            if erased_something:
                # Save the camera state so the screen doesn't jerk during rapid deletion
                cam_pos = self.main_suite.renderer.plotter.camera_position
                cam_fp = self.main_suite.renderer.plotter.camera.focal_point
                
                self.main_suite.refresh_render()
                
                # Restore the exact camera orientation
                self.main_suite.renderer.plotter.camera_position = cam_pos
                self.main_suite.renderer.plotter.camera.focal_point = cam_fp
                self.main_suite.renderer.plotter.render()

    def exit_eraser_mode(self, *args):
        """Quickly escapes eraser mode via right-click."""
        if hasattr(self, 'chk_edit_mode'):
            self.chk_edit_mode.setChecked(False)

    def align_to_hkl(self):
        if not getattr(self.main_suite, 'current_structure', None): return
        h, k, l = self.spin_h.value(), self.spin_k.value(), self.spin_l.value()
        if h == 0 and k == 0 and l == 0: return
        
        # We MUST use the reciprocal lattice to find the normal vector to the (h k l) plane
        recip_matrix = self.main_suite.current_structure.lattice.reciprocal_lattice.matrix
        cart_vec = h * recip_matrix[0] + k * recip_matrix[1] + l * recip_matrix[2]
        
        if hasattr(self.main_suite.renderer, 'plotter') and hasattr(self.main_suite.renderer.plotter, 'camera'):
            fp = np.array(self.main_suite.renderer.plotter.camera.focal_point)
            dist = self.main_suite.renderer.plotter.camera.distance
            
            norm_vec = cart_vec / np.linalg.norm(cart_vec)
            self.main_suite.renderer.plotter.camera.position = fp + norm_vec * dist
            
            # Prevent PyVista camera crash (gimbal lock) if looking straight down Z
            if abs(norm_vec[2]) > 0.99:
                self.main_suite.renderer.plotter.camera.up = (0, 1, 0)
            else:
                self.main_suite.renderer.plotter.camera.up = (0, 0, 1)
                
            self.main_suite.renderer.plotter.render()
            
        elif hasattr(self.main_suite.renderer, 'ax'):
            dist = np.linalg.norm(cart_vec)
            az = np.degrees(np.arctan2(cart_vec[1], cart_vec[0]))
            el = np.degrees(np.arcsin(cart_vec[2] / dist))
            self.main_suite.renderer.ax.view_init(elev=el, azim=az)
            if hasattr(self.main_suite.renderer, 'canvas'):
                self.main_suite.renderer.canvas.draw_idle()

    def save_image(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
        if not fname: return
        if hasattr(self.main_suite.renderer, 'plotter') and hasattr(self.main_suite.renderer.plotter, 'screenshot'):
            self.main_suite.renderer.plotter.screenshot(fname, transparent_background=True, window_size=[3840, 2160])
        elif hasattr(self.main_suite.renderer, 'figure'):
            self.main_suite.renderer.figure.savefig(fname, dpi=600, bbox_inches='tight', transparent=True)
        QMessageBox.information(self, "Saved", f"High resolution image saved to:\n{fname}")

    def export_scripts(self):
        if not getattr(self.main_suite, 'active_supercell', None):
            QMessageBox.warning(self, "Export Error", "Please draw a structure first!")
            return
            
        sender = self.sender()
        software = "3ds Max" if sender == self.btn_export_max else "Blender"
        file_path, _ = QFileDialog.getSaveFileName(self, f"Export {software} Script", "", "Python Files (*.py)")
        if not file_path: return

        atoms_data, bonds_data, lattice_data = [], [], []
        bz_solid_data = None

        if self.chk_exp_atoms.isChecked():
            scale_mod = self.spin_radius.value()
            for i, site in enumerate(self.main_suite.active_supercell):
                if i in self.main_suite.erased_atoms: continue
                radius = float((site.specie.atomic_radius if site.specie.atomic_radius else 1.0) * scale_mod)
                color = self.main_suite.active_colors.get(site.specie.symbol, "#008080")
                atoms_data.append((float(site.coords[0]), float(site.coords[1]), float(site.coords[2]), radius, color))

            if self.combo_style.currentIndex() == 0:
                cyl_radius = float(self.spin_bond_thick.value())
                bond_color = self.main_suite.active_colors.get("Bonds", "#FFFFFF")
                coords = self.main_suite.active_supercell.cart_coords
                radii = np.array([s.specie.atomic_radius if s.specie.atomic_radius else 1.2 for s in self.main_suite.active_supercell])
                thresh = self.spin_bond_thresh.value()
                
                dist_mat = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
                threshold_mat = (radii[:, np.newaxis] + radii[np.newaxis, :]) * thresh
                valid_pairs = np.triu((dist_mat > 0.5) & (dist_mat <= threshold_mat), k=1)
                
                for i, j in np.argwhere(valid_pairs):
                    if (i, j) in self.main_suite.erased_bonds or i in self.main_suite.erased_atoms or j in self.main_suite.erased_atoms:
                        continue
                    bonds_data.append((float(coords[i][0]), float(coords[i][1]), float(coords[i][2]), 
                                       float(coords[j][0]), float(coords[j][1]), float(coords[j][2]), 
                                       cyl_radius, bond_color))

        if self.chk_exp_cell.isChecked() and getattr(self.main_suite, 'current_structure', None):
            def get_edges(matrix, color):
                a, b, c = matrix[0], matrix[1], matrix[2]
                v = np.zeros((8, 3))
                v[1], v[2], v[3] = a, b, c; v[4], v[5], v[6], v[7] = a + b, a + c, b + c, a + b + c
                edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
                for p1, p2 in edges:
                    lattice_data.append((float(v[p1][0]), float(v[p1][1]), float(v[p1][2]), float(v[p2][0]), float(v[p2][1]), float(v[p2][2]), color))

            if self.chk_show_conventional.isChecked():
                get_edges(self.main_suite.current_structure.lattice.matrix, "#FF5733")
            if self.chk_show_primitive.isChecked():
                prim_matrix = CrystalEngine.get_universal_primitive_matrix(self.main_suite.current_structure)
                get_edges(prim_matrix, "#33FF57")

        if self.chk_exp_bz.isChecked():
            if hasattr(self.main_suite, 'active_bz_edges') and getattr(self.main_suite, 'active_bz_style', 0) in [1, 2]:
                for p1, p2 in self.main_suite.active_bz_edges:
                    lattice_data.append((float(p1[0]), float(p1[1]), float(p1[2]), float(p2[0]), float(p2[1]), float(p2[2]), "#FF00FF"))
            
            if hasattr(self.main_suite, 'active_bz_faces') and getattr(self.main_suite, 'active_bz_style', 0) in [0, 2]:
                bz_solid_data = {'verts': self.main_suite.active_bz_points.tolist(), 'faces': self.main_suite.active_bz_faces}
                
            if hasattr(self.main_suite, 'active_surf_bz_edges') and self.main_suite.active_surf_bz_edges:
                for p1, p2 in self.main_suite.active_surf_bz_edges:
                    lattice_data.append((float(p1[0]), float(p1[1]), float(p1[2]), float(p2[0]), float(p2[1]), float(p2[2]), "#00FFFF"))

        if software == "3ds Max":
            SceneExporter.export_3dsmax(file_path, atoms_data, bonds_data, lattice_data, bz_solid_data)
        else:
            SceneExporter.export_blender(file_path, atoms_data, bonds_data, lattice_data, bz_solid_data)
        QMessageBox.information(self, "Export Complete", f"Successfully saved {software} script!")

    def update_azel(self):
        az = self.spin_azimuth.value()
        el = self.spin_elevation.value()
        
        if self.combo_backend.currentIndex() == 0: 
            if hasattr(self.main_suite.renderer, 'ax'):
                self.main_suite.renderer.ax.view_init(elev=el, azim=az)
                if hasattr(self.main_suite.renderer, 'canvas'):
                    self.main_suite.renderer.canvas.draw_idle()
        else: 
            if hasattr(self.main_suite.renderer, 'plotter') and hasattr(self.main_suite.renderer.plotter, 'camera'):
                fp = np.array(self.main_suite.renderer.plotter.camera.focal_point)
                dist = self.main_suite.renderer.plotter.camera.distance
                az_rad, el_rad = np.radians(az), np.radians(el)
                
                dx = dist * np.cos(el_rad) * np.cos(az_rad)
                dy = dist * np.cos(el_rad) * np.sin(az_rad)
                dz = dist * np.sin(el_rad)
                
                self.main_suite.renderer.plotter.camera.position = fp + np.array([dx, dy, dz])
                self.main_suite.renderer.plotter.camera.up = (0, 0, 1)
                self.main_suite.renderer.plotter.render()

    def toggle_cut_plane(self):
        if self.chk_show_plane.isChecked():
            self.lock_plane_normal()
        self.update_cut_plane()

    def lock_plane_normal(self, *args):
        if not getattr(self.main_suite, 'current_structure', None) or self.combo_backend.currentIndex() == 0: return
        if not hasattr(self.main_suite.renderer, 'plotter'): return
        
        self.plane_center_base = np.array(self.main_suite.renderer.plotter.camera.focal_point)
        
        if self.combo_plane_orient.currentIndex() == 0: 
            fp = np.array(self.main_suite.renderer.plotter.camera.focal_point)
            pos = np.array(self.main_suite.renderer.plotter.camera.position)
            normal = pos - fp
        else: 
            h, k, l = self.spin_h.value(), self.spin_k.value(), self.spin_l.value()
            if h == 0 and k == 0 and l == 0: 
                normal = np.array([0,0,1])
            else:
                recip_matrix = self.main_suite.current_structure.lattice.reciprocal_lattice.matrix
                normal = h * recip_matrix[0] + k * recip_matrix[1] + l * recip_matrix[2]
                
        dist = np.linalg.norm(normal)
        self.plane_normal = normal / dist if dist != 0 else np.array([0,0,1])
        
        if self.chk_show_plane.isChecked():
            self.update_cut_plane()

    def update_cut_plane(self):
        if self.combo_backend.currentIndex() == 0: return 
        if not hasattr(self.main_suite.renderer, 'plotter'): return
        
        import pyvista as pv
        
        if hasattr(self.main_suite.renderer, 'plane_actor') and self.main_suite.renderer.plane_actor is not None:
            self.main_suite.renderer.plotter.remove_actor(self.main_suite.renderer.plane_actor)
            self.main_suite.renderer.plane_actor = None
            
        if not self.chk_show_plane.isChecked() or not getattr(self.main_suite, 'current_structure', None):
            self.main_suite.renderer.plotter.render()
            return
            
        if getattr(self, 'plane_normal', None) is None:
            self.lock_plane_normal()
            
        normal = self.plane_normal
        base_center = self.plane_center_base
        
        depth_val = self.slider_plane_depth.value() / 100.0 
        bounds = self.main_suite.renderer.plotter.bounds
        diag = np.linalg.norm([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
        
        center = base_center + normal * (depth_val * diag / 2.0)
        
        plane = pv.Plane(center=center, direction=normal, i_size=diag*1.5, j_size=diag*1.5)
        color = self.combo_plane_color.currentText()
        
        self.main_suite.renderer.plane_actor = self.main_suite.renderer.plotter.add_mesh(plane, color=color, opacity=0.35, pickable=False)
        self.main_suite.renderer.plotter.render()

    def sync_ui_to_camera(self, *args):
        vec = None
        import numpy as np
        if self.combo_backend.currentIndex() == 0:
            if hasattr(self.main_suite.renderer, 'ax'):
                az = self.main_suite.renderer.ax.azim
                el = self.main_suite.renderer.ax.elev
                az_rad, el_rad = np.radians(az), np.radians(el)
                vec = np.array([np.cos(el_rad)*np.cos(az_rad), np.cos(el_rad)*np.sin(az_rad), np.sin(el_rad)])
        else:
            if hasattr(self.main_suite.renderer, 'plotter') and hasattr(self.main_suite.renderer.plotter, 'camera'):
                fp = np.array(self.main_suite.renderer.plotter.camera.focal_point)
                pos = np.array(self.main_suite.renderer.plotter.camera.position)
                vec = pos - fp
                dist = np.linalg.norm(vec)
                if dist == 0: return
                az = np.degrees(np.arctan2(vec[1], vec[0]))
                el = np.degrees(np.arcsin(vec[2] / dist))

        if vec is not None:
            self.spin_azimuth.blockSignals(True)
            self.spin_elevation.blockSignals(True)
            self.spin_azimuth.setValue(az)
            self.spin_elevation.setValue(el)
            self.spin_azimuth.blockSignals(False)
            self.spin_elevation.blockSignals(False)
            
            if getattr(self.main_suite, 'current_structure', None):
                recip_inv_matrix = self.main_suite.current_structure.lattice.reciprocal_lattice.inv_matrix
                frac_vec = np.dot(vec, recip_inv_matrix)
                max_val = np.max(np.abs(frac_vec))
                if max_val > 0: frac_vec = frac_vec / max_val
                
                self.spin_h.blockSignals(True)
                self.spin_k.blockSignals(True)
                self.spin_l.blockSignals(True)
                
                self.spin_h.setValue(frac_vec[0])
                self.spin_k.setValue(frac_vec[1])
                self.spin_l.setValue(frac_vec[2])
                
                self.spin_h.blockSignals(False)
                self.spin_k.blockSignals(False)
                self.spin_l.blockSignals(False)

    def export_cif(self):
        """Exports the currently active supercell/cleaved structure to a CIF file."""
        if not getattr(self.main_suite, 'current_structure', None):
            QMessageBox.warning(self, "Export Error", "No structure available to export!")
            return
            
        fname, _ = QFileDialog.getSaveFileName(self, "Save CIF", "cleaved_structure.cif", "CIF Files (*.cif)")
        if fname:
            try:
                # PyMatgen's built-in exporter
                self.main_suite.current_structure.to(filename=fname, fmt="cif")
                QMessageBox.information(self, "Success", f"Structure successfully saved to:\n{fname}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save CIF:\n{str(e)}")