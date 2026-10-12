# File: tensorspec/gui/suites/crystal_suite.py
import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QFileDialog, QLabel, QSpinBox, QDoubleSpinBox, 
                               QComboBox, QColorDialog, QTabWidget, QCheckBox, 
                               QGroupBox, QGridLayout, QSlider, QSplitter, 
                               QInputDialog, QMessageBox, QFrame, QScrollArea,
                               QDialog, QDialogButtonBox)
from PySide6.QtCore import Qt
from pymatgen.core import Structure

# Clean modular imports from our established architecture
from tensorspec.core.crystallography import CrystalEngine
from tensorspec.core.bernevig_twist import classify_bravais
from tensorspec.core.workspace import global_workspace
import platform

# --- SMART ENGINE ROUTER ---
is_legacy_mac = (platform.system() == "Darwin" and platform.machine() == "x86_64")

if is_legacy_mac:
    print("🔌 Crystal Suite: Routing to crash-proof Matplotlib backend.")
    from tensorspec.plotting.backends.matplotlib_engine import MatplotlibCrystalBackend as ActiveCrystalBackend
else:
    print("🔌 Crystal Suite: Routing to high-performance PyVista backend.")
    from tensorspec.plotting.backends.pyvista_engine import PyVistaCrystalBackend as ActiveCrystalBackend


# Default CPK Colors for dynamic UI generation
CPK_COLORS = {
    "H": "#FFFFFF", "C": "#333333", "N": "#2233FF", "O": "#FF2200",
    "Te": "#FF8C00", "Fe": "#E06633", "Ta": "#B041FF", "Ir": "#0080FF",
    "Nb": "#7A378B", "W": "#4682B4", "Mo": "#5F9EA0", "Bonds": "#d3d3d3"
}


class StackLayerRow(QFrame):
    """Mini UI Row Controller for individual 2D sheets in the Heterostructure Tab."""
    def __init__(self, name: str, struct: Structure, default_z: float):

        # --- NEW: Terminal Logging ---
        print("open suite Crystal Viewer")

        super().__init__()
        self.struct = struct
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { background-color: #2b2b36; border: 1px solid #555555; border-radius: 5px; padding: 4px; color: #ffffff; } QLabel { color: #ffffff; }")
        
        main_layout = QVBoxLayout(self) # Stacked vertically now!
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # --- Top Row: Name and Supercell ---
        top_row = QHBoxLayout()
        self.lbl_name = QLabel(f"<b>{name}</b>")
        top_row.addWidget(self.lbl_name)
        
        top_row.addWidget(QLabel("SC:"))
        self.spin_x = QSpinBox(); self.spin_x.setRange(1, 100); self.spin_x.setValue(1)
        self.spin_y = QSpinBox(); self.spin_y.setRange(1, 100); self.spin_y.setValue(1)
        top_row.addWidget(self.spin_x); top_row.addWidget(self.spin_y)
        main_layout.addLayout(top_row)
        
        # --- Bottom Row: Z-shift, Twist, and Action Buttons ---
        bot_row = QHBoxLayout()
        bot_row.addWidget(QLabel("z (Å):"))
        self.spin_z = QDoubleSpinBox(); self.spin_z.setRange(-100.0, 100.0); self.spin_z.setSingleStep(0.5); self.spin_z.setValue(default_z)
        bot_row.addWidget(self.spin_z)
        
        bot_row.addWidget(QLabel("θ (°):"))
        self.spin_twist = QDoubleSpinBox(); self.spin_twist.setRange(-360.0, 360.0); self.spin_twist.setSingleStep(1.0)
        bot_row.addWidget(self.spin_twist)
        
        self.btn_up = QPushButton("▲"); self.btn_up.setFixedWidth(25)
        self.btn_down = QPushButton("▼"); self.btn_down.setFixedWidth(25)
        self.btn_save_tpl = QPushButton("💾"); self.btn_save_tpl.setStyleSheet("background-color: #5cb85c; color: white;"); self.btn_save_tpl.setFixedWidth(25)
        self.btn_delete = QPushButton("X"); self.btn_delete.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;"); self.btn_delete.setFixedWidth(28)
        
        bot_row.addWidget(self.btn_up); bot_row.addWidget(self.btn_down)
        bot_row.addWidget(self.btn_save_tpl); bot_row.addWidget(self.btn_delete)
        main_layout.addLayout(bot_row)

    def get_layer_dict(self) -> dict:
        return {
            "struct": self.struct, "sc_x": self.spin_x.value(),
            "sc_y": self.spin_y.value(), "z_shift": self.spin_z.value(), "twist": self.spin_twist.value()
        }


class CrystalViewerSuite(QWidget):
    """
    Main Crystal Suite UI Controller.
    Links user actions to core crystallography math and 3D PyVista rendering.
    """
    def __init__(self, workspace_manager=None, parent=None):
        print("open suite Crystal Suite")
        super().__init__(parent)

        self.setAttribute(Qt.WA_DeleteOnClose)
        self._is_closing = False  # NEW: Kill-switch to prevent ghost rendering

        self.workspace = workspace_manager
        
        self.current_structure = None
        self.active_supercell = None
        self.active_colors = CPK_COLORS.copy()
        self.stack_layer_rows = []
        self.erased_atoms = set()
        self.erased_bonds = set()
        
        self.init_layout()

    def init_layout(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        #1. Left Control Panel (Tabs)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(340)
        left_layout.addWidget(self.tabs)
        
        # --- NEW: Push to Workspace Button ---
        self.btn_push_workspace = QPushButton("📥 Push Structure to Central Workspace")
        self.btn_push_workspace.setStyleSheet("background-color: #5cb85c; color: white; font-weight: bold; padding: 10px;")
        self.btn_push_workspace.clicked.connect(self.push_current_to_workspace)
        left_layout.addWidget(self.btn_push_workspace)
        
        splitter.addWidget(left_container)
        
        # --- NEW: Import and Mount Modular Tab 1 ---
        from tensorspec.gui.components.crystal_panel import TabViewEdit
        self.tab_view = TabViewEdit(self)
        self.tabs.addTab(self.tab_view, "1. View & Edit")
        
        # Keep the rest initialized locally for now until we move them
        self.init_tab_cdw()
        self.init_tab_heterostructure()
        self.init_tab_bz()

        # 2. Right 3D Viewport (Stacked Backends)
        from PySide6.QtWidgets import QStackedWidget
        from tensorspec.plotting.backends.matplotlib_engine import MatplotlibCrystalBackend
        from tensorspec.plotting.backends.pyvista_engine import PyVistaCrystalBackend
        
        self.viewer_stack = QStackedWidget()
        self.renderer_cpu = MatplotlibCrystalBackend(parent=self)
        self.renderer_gpu = PyVistaCrystalBackend(parent=self)
        
        self.viewer_stack.addWidget(self.renderer_cpu.plotter) # Index 0: Matplotlib
        self.viewer_stack.addWidget(self.renderer_gpu.plotter) # Index 1: PyVista
        
        # Hook the camera sync to mouse release
        self.renderer_gpu.plotter.iren.add_observer("EndInteractionEvent", self.tab_view.sync_ui_to_camera)
        splitter.addWidget(self.viewer_stack)
        
        # Auto-detect default based on hardware, but allow manual switching
        if is_legacy_mac:
            self.viewer_stack.setCurrentIndex(0)
            self.renderer = self.renderer_cpu
        else:
            self.viewer_stack.setCurrentIndex(1)
            self.renderer = self.renderer_gpu
        
        splitter.setSizes([380, 820])


    
    # ================= TAB 2: CDW MODULATOR =================
    def init_tab_cdw(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.chk_cdw = QCheckBox("Enable CDW Modulation")
        self.chk_cdw.setStyleSheet("font-weight: bold; color: #2b5c8f;")
        self.chk_cdw.stateChanged.connect(self.refresh_render)
        layout.addWidget(self.chk_cdw)
        
        layout.addWidget(QLabel("Target Element:"))
        self.combo_cdw_el = QComboBox(); self.combo_cdw_el.addItem("All Elements")
        layout.addWidget(self.combo_cdw_el)

        group_q = QGroupBox("Wavevector q (rlu)")
        q_layout = QVBoxLayout(group_q)
        self.spin_qx = QDoubleSpinBox(); self.spin_qx.setRange(-5.0, 5.0); self.spin_qx.setSingleStep(0.05)
        self.spin_qy = QDoubleSpinBox(); self.spin_qy.setRange(-5.0, 5.0); self.spin_qy.setSingleStep(0.05)
        self.spin_qz = QDoubleSpinBox(); self.spin_qz.setRange(-5.0, 5.0); self.spin_qz.setSingleStep(0.05)
        for spin, lbl in [(self.spin_qx, "q_a:"), (self.spin_qy, "q_b:"), (self.spin_qz, "q_c:")]:
            row = QHBoxLayout(); row.addWidget(QLabel(lbl)); row.addWidget(spin); q_layout.addLayout(row)
            spin.valueChanged.connect(self.refresh_render)
        layout.addWidget(group_q)

        group_amp = QGroupBox("Amplitude A (Å)")
        amp_layout = QVBoxLayout(group_amp)
        self.spin_ax = QDoubleSpinBox(); self.spin_ax.setRange(-2.0, 2.0); self.spin_ax.setSingleStep(0.02)
        self.spin_ay = QDoubleSpinBox(); self.spin_ay.setRange(-2.0, 2.0); self.spin_ay.setSingleStep(0.02)
        self.spin_az = QDoubleSpinBox(); self.spin_az.setRange(-2.0, 2.0); self.spin_az.setSingleStep(0.02)
        for spin, lbl in [(self.spin_ax, "Δx:"), (self.spin_ay, "Δy:"), (self.spin_az, "Δz:")]:
            row = QHBoxLayout(); row.addWidget(QLabel(lbl)); row.addWidget(spin); amp_layout.addLayout(row)
            spin.valueChanged.connect(self.refresh_render)
        layout.addWidget(group_amp)

        # --- RESTORED: Phase Adjuster ---
        layout.addWidget(QLabel("Phase Shift φ (degrees):"))
        self.spin_cdw_phase = QDoubleSpinBox()
        self.spin_cdw_phase.setRange(0.0, 360.0); self.spin_cdw_phase.setSingleStep(15.0)
        self.spin_cdw_phase.valueChanged.connect(self.refresh_render)
        layout.addWidget(self.spin_cdw_phase)

        layout.addStretch()
        self.tabs.addTab(tab, "2. CDW Modulator")

    # ================= TAB 3: STACK & TWIST =================
    def init_tab_heterostructure(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        tpl_group = QGroupBox("2D Material Templates")
        tpl_layout = QVBoxLayout(tpl_group)
        
        self.btn_add_layer = QPushButton("📂 Load 2D Monolayer CIF")
        self.btn_add_layer.setStyleSheet("background-color: #5cb85c; color: white; font-weight: bold; padding: 5px; border-radius: 4px;")
        self.btn_add_layer.clicked.connect(self.load_cif_layer)
        tpl_layout.addWidget(self.btn_add_layer)
        
        self.combo_tpl = QComboBox()
        base_templates = ["Graphene (Monolayer)", "Graphene (AB Bilayer)", "TaIrTe4 (1T')", "MoS2", "h-BN"]
        if os.path.exists("user_templates.json"):
            try:
                import json
                with open("user_templates.json", "r") as f:
                    custom_templates = json.load(f)
                    base_templates.extend(list(custom_templates.keys()))
            except Exception:
                pass
                
        self.combo_tpl.addItems(base_templates)
        tpl_layout.addWidget(self.combo_tpl)
        
        btn_add_tpl = QPushButton("➕ Add Template Sheet")
        btn_add_tpl.clicked.connect(self.handle_add_template)
        tpl_layout.addWidget(btn_add_tpl)
        layout.addWidget(tpl_group)

        # --- RESTORED: Bulk Exfoliator with Dynamic HKL UI ---
        exf_group = QGroupBox("Bulk Exfoliator")
        exf_layout = QVBoxLayout(exf_group)

        exf_layout.addWidget(QLabel("Cleavage / Cut Engine Mode:"))
        self.combo_exfoliate_mode = QComboBox()
        self.combo_exfoliate_mode.addItems([
            "Automatic van der Waals Gap Detection",
            "Manual [h k l] Miller Index Cleavage"
        ])
        exf_layout.addWidget(self.combo_exfoliate_mode)
        
        # Embedded HKL Inputs
        self.hkl_widget = QWidget()
        hkl_layout = QHBoxLayout(self.hkl_widget)
        hkl_layout.setContentsMargins(0, 0, 0, 0)
        hkl_layout.addWidget(QLabel("Layers:"))
        self.spin_exf_layers = QSpinBox(); self.spin_exf_layers.setRange(1, 10); self.spin_exf_layers.setValue(1)
        hkl_layout.addWidget(self.spin_exf_layers)
        
        hkl_layout.addWidget(QLabel("Vacuum (Å):"))
        self.spin_exf_vacuum = QDoubleSpinBox(); self.spin_exf_vacuum.setRange(0.0, 100.0); self.spin_exf_vacuum.setValue(15.0)
        hkl_layout.addWidget(self.spin_exf_vacuum)
        
        hkl_layout.addWidget(QLabel("Plane [h k l]:"))
        self.spin_exf_h = QSpinBox(); self.spin_exf_h.setRange(-10, 10); self.spin_exf_h.setValue(0)
        self.spin_exf_k = QSpinBox(); self.spin_exf_k.setRange(-10, 10); self.spin_exf_k.setValue(0)
        self.spin_exf_l = QSpinBox(); self.spin_exf_l.setRange(-10, 10); self.spin_exf_l.setValue(1)
        hkl_layout.addWidget(self.spin_exf_h); hkl_layout.addWidget(self.spin_exf_k); hkl_layout.addWidget(self.spin_exf_l)
        exf_layout.addWidget(self.hkl_widget)
        
        # Hide by default, show when Manual is selected
        self.hkl_widget.setVisible(False)
        self.combo_exfoliate_mode.currentTextChanged.connect(
            lambda text: self.hkl_widget.setVisible("Manual" in text)
        )

        self.btn_extract_bulk = QPushButton("✂️ Extract Monolayer from Bulk CIF")
        self.btn_extract_bulk.setStyleSheet("background-color: #f0ad4e; color: black; font-weight: bold; padding: 5px;")
        self.btn_extract_bulk.clicked.connect(self.extract_monolayer_from_bulk)
        exf_layout.addWidget(self.btn_extract_bulk)
        layout.addWidget(exf_group)

        self.scroll_area = QScrollArea(); self.scroll_area.setWidgetResizable(True)
        self.scroll_container = QWidget(); self.scroll_layout = QVBoxLayout(self.scroll_container)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.scroll_container)
        layout.addWidget(self.scroll_area)

        btn_draw_stack = QPushButton("🧱 Render Heterostructure Stack")
        btn_draw_stack.setStyleSheet("background-color: #2b5c8f; color: white; font-weight: bold; padding: 6px;")
        btn_draw_stack.clicked.connect(self.handle_draw_stack)
        layout.addWidget(btn_draw_stack)

        vac_row = QHBoxLayout()
        vac_row.addWidget(QLabel("DFT vacuum (Å):"))
        self.spin_stack_vacuum = QDoubleSpinBox()
        self.spin_stack_vacuum.setRange(5.0, 100.0)
        self.spin_stack_vacuum.setSingleStep(1.0)
        self.spin_stack_vacuum.setValue(20.0)
        self.spin_stack_vacuum.setToolTip(
            "Vacuum padding along c for Push→DFT cell. Interlayer spacing uses each layer's z spinbox."
        )
        vac_row.addWidget(self.spin_stack_vacuum)
        vac_row.addStretch()
        layout.addLayout(vac_row)

        bernevig_group = QGroupBox("Bernevig twist")
        bernevig_layout = QGridLayout(bernevig_group)

        bernevig_layout.addWidget(QLabel("θ (°):"), 0, 0)
        self.spin_bernevig_theta = QDoubleSpinBox()
        self.spin_bernevig_theta.setRange(-180.0, 180.0)
        self.spin_bernevig_theta.setDecimals(5)
        self.spin_bernevig_theta.setValue(7.34)
        bernevig_layout.addWidget(self.spin_bernevig_theta, 0, 1)

        bernevig_layout.addWidget(QLabel("Stacking:"), 1, 0)
        self.combo_bernevig_stacking = QComboBox()
        self.combo_bernevig_stacking.addItems(["AA", "AB"])
        bernevig_layout.addWidget(self.combo_bernevig_stacking, 1, 1)

        bernevig_layout.addWidget(QLabel("Valley:"), 2, 0)
        self.combo_bernevig_valley = QComboBox()
        self.combo_bernevig_valley.addItems(["Γ", "K", "M", "nHSP"])
        self.combo_bernevig_valley.setCurrentText("K")
        bernevig_layout.addWidget(self.combo_bernevig_valley, 2, 1)

        bernevig_layout.addWidget(QLabel("Max cells:"), 3, 0)
        self.spin_bernevig_max_cells = QSpinBox()
        self.spin_bernevig_max_cells.setRange(1, 2000)
        self.spin_bernevig_max_cells.setValue(217)
        bernevig_layout.addWidget(self.spin_bernevig_max_cells, 3, 1)

        self.btn_bernevig = QPushButton("Build twisted bilayer")
        self.btn_bernevig.clicked.connect(self.handle_bernevig_twist)
        bernevig_layout.addWidget(self.btn_bernevig, 4, 0, 1, 2)
        layout.addWidget(bernevig_group)

        self.btn_moire = QPushButton("🌀 Calculate Moiré Superlattice")
        self.btn_moire.setStyleSheet("background-color: #8A2BE2; color: white; font-weight: bold; padding: 6px;")
        self.btn_moire.clicked.connect(self.handle_moire)
        layout.addWidget(self.btn_moire)

        self.lbl_moire = QLabel("Status: Waiting for stack...")
        self.lbl_moire.setStyleSheet("background-color: #1e1e24; color: white; padding: 4px;")
        layout.addWidget(self.lbl_moire)

        self.tabs.addTab(tab, "3. Stack & Twist")

    # ================= TAB 4: BRILLOUIN ZONE =================
    def init_tab_bz(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("<b>Reciprocal Space (Wigner-Seitz Cell)</b>"))
        
        self.chk_overlay_atoms = QCheckBox("Overlay Real-Space Crystal Structure")
        self.chk_overlay_atoms.setStyleSheet("font-weight: bold; color: #d9534f;")
        self.chk_overlay_atoms.setChecked(True)
        layout.addWidget(self.chk_overlay_atoms)
        
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("BZ Visual Scale:"))
        self.spin_bz_scale = QDoubleSpinBox()
        self.spin_bz_scale.setRange(0.1, 10.0); self.spin_bz_scale.setValue(1.0); self.spin_bz_scale.setSingleStep(0.1)
        scale_layout.addWidget(self.spin_bz_scale)
        layout.addLayout(scale_layout)
        
        # --- NEW: Skeleton vs Solid Control ---
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("Render Style:"))
        self.combo_bz_style = QComboBox()
        self.combo_bz_style.addItems(["Solid Faces", "Skeleton (Cylinder Frame)", "Both Solid & Skeleton"])
        style_layout.addWidget(self.combo_bz_style)
        layout.addLayout(style_layout)
        
        # --- NEW: Surface Projection Controls ---
        group_surf = QGroupBox("Projected Surface BZ")
        surf_layout = QVBoxLayout(group_surf)
        self.chk_surf_bz = QCheckBox("Enable Surface Projection")
        self.chk_surf_bz.setStyleSheet("font-weight: bold; color: #0F6A8B;")
        surf_layout.addWidget(self.chk_surf_bz)
        
        hkl_layout = QHBoxLayout()
        hkl_layout.addWidget(QLabel("Plane [h k l]:"))
        self.spin_bz_h = QSpinBox(); self.spin_bz_h.setRange(-10, 10); self.spin_bz_h.setValue(0)
        self.spin_bz_k = QSpinBox(); self.spin_bz_k.setRange(-10, 10); self.spin_bz_k.setValue(0)
        self.spin_bz_l = QSpinBox(); self.spin_bz_l.setRange(-10, 10); self.spin_bz_l.setValue(1)
        hkl_layout.addWidget(self.spin_bz_h); hkl_layout.addWidget(self.spin_bz_k); hkl_layout.addWidget(self.spin_bz_l)
        surf_layout.addLayout(hkl_layout)
        layout.addWidget(group_surf)
        
        btn_bz = QPushButton("🛑 Render Brillouin Zone")
        btn_bz.setStyleSheet("background-color: #8A2BE2; color: white; font-weight: bold; padding: 6px;")
        btn_bz.clicked.connect(self.handle_draw_bz)
        layout.addWidget(btn_bz)
        
        layout.addStretch()
        self.tabs.addTab(tab, "4. Brillouin Zone")

    # ================= LOGIC AND EVENT CONTROLLERS ================

    

    def refresh_render(self):
        """Routes structure state through core math transformations, then pushes to PyVista."""
        if getattr(self, '_is_closing', False): return  # ABORT if the window is tearing down
        if not self.active_supercell: return
        
        render_struct = self.active_supercell.copy()
        if getattr(self, 'chk_cdw', None) and self.chk_cdw.isChecked():
            import numpy as np
            q_vec = (self.spin_qx.value(), self.spin_qy.value(), self.spin_qz.value())
            amp_vec = (self.spin_ax.value(), self.spin_ay.value(), self.spin_az.value())
            phase_rad = np.radians(self.spin_cdw_phase.value()) if hasattr(self, 'spin_cdw_phase') else 0.0
            render_struct = CrystalEngine.apply_cdw_distortion(render_struct, self.combo_cdw_el.currentText(), q_vec, amp_vec, phase_rad)

        self.renderer.clear_scene()
        
        # Point to the variables inside the new tab_view
        is_shiny = self.tab_view.chk_shiny.isChecked()
        scale = self.tab_view.spin_radius.value()
        
        style = self.tab_view.combo_style.currentIndex()
        
        # Pass the erased_atoms list to the renderer so it knows what to hide!
        self.renderer.draw_atoms(render_struct, self.active_colors, scale_mod=scale, is_shiny=is_shiny, erased_atoms=self.erased_atoms)
        
        if style == 0:
            # Fetch the exact widget names defined in tab_view.py
            b_rad = self.tab_view.spin_bond_thick.value() if hasattr(self.tab_view, 'spin_bond_thick') else 0.1
            b_thresh = self.tab_view.spin_bond_thresh.value() if hasattr(self.tab_view, 'spin_bond_thresh') else 1.15
            
            # Pass both the radius AND the threshold to the rendering engine
            self.renderer.draw_bonds(render_struct, self.active_colors, cyl_radius=b_rad, thresh_multiplier=b_thresh, is_shiny=is_shiny, erased_bonds=self.erased_bonds, erased_atoms=self.erased_atoms)
        elif style == 1:
            if hasattr(self.renderer, 'draw_polyhedra'):
                self.renderer.draw_polyhedra(render_struct, self.active_colors)
        
        if self.current_structure:
            # Prevent the massive 500A dummy lattice from ruining the zoom scale
            is_dummy_canvas = self.current_structure.lattice.a >= 499.0
            
            conv_mat = self.current_structure.lattice.matrix if (self.tab_view.chk_show_conventional.isChecked() and not is_dummy_canvas) else None
            prim_mat = CrystalEngine.get_universal_primitive_matrix(self.current_structure) if (self.tab_view.chk_show_primitive.isChecked() and not is_dummy_canvas) else None
            
            if self.tab_view.chk_axes.isChecked() and not is_dummy_canvas:
                self.renderer.draw_axes(conventional_matrix=conv_mat, primitive_matrix=prim_mat)
            if conv_mat is not None or prim_mat is not None:
                self.renderer.draw_lattice_boxes(conventional_matrix=conv_mat, primitive_matrix=prim_mat)
            
        if hasattr(self.renderer, 'plotter') and hasattr(self.renderer.plotter, 'render'):
            self.renderer.plotter.render()
        elif hasattr(self.renderer, 'canvas'):
            self.renderer.canvas.draw_idle()
        
    def push_current_to_workspace(self):
        """Push Tab 1 structure, or rebuild DFT-ready cell from Tab 3 stack rows."""
        # Non-stack path: Tab 1 / no hetero rows
        if not getattr(self, "stack_layer_rows", None):
            if getattr(self, "current_structure", None) is None:
                QMessageBox.warning(self, "Warning", "No active structure to push!")
                return
            default_name = getattr(self, "current_filename", "My_Crystal")
            name, ok = QInputDialog.getText(
                self, "Workspace Export", "Enter a variable name for this structure:", text=default_name
            )
            if ok and name:
                global_workspace.push_crystal_structure(name, self.current_structure)
                QMessageBox.information(
                    self, "Success",
                    f"Structure '{name}' sent to Global Workspace!\nYou can now load it in the DFT or ARPES Suites.",
                )
            return

        layers = [row.get_layer_dict() for row in self.stack_layer_rows]
        # Row twists decide the path. Bernevig spin is NOT consulted unless the user opted in
        # (AB stacking or a nonzero row twist); the 7.34 default must not force a twist Push.
        kind = CrystalEngine.classify_stack_for_dft(layers)
        if (
            kind == "aligned"
            and len(layers) == 2
            and self.combo_bernevig_stacking.currentText() == "AB"
        ):
            kind = "twist"

        if kind == "empty":
            QMessageBox.warning(self, "Warning", "No layers in the stack to push.")
            return

        if kind == "reject_multitwist":
            QMessageBox.information(
                self,
                "Multi-twist stacks — coming later",
                "DFT cell rebuild for more than two twisted layers is not available yet.\n\n"
                "This week: N layers with all twists = 0°, or exactly 2 layers with twist "
                "(e.g. graphene/hBN). Render/view still works.",
            )
            return

        vacuum = float(self.spin_stack_vacuum.value())
        suggested, strains = CrystalEngine.suggest_reference_layer(layers)
        ref_idx = suggested
        need_dialog = False
        dialog_status = ""

        if kind == "aligned":
            need_dialog = CrystalEngine.aligned_needs_strain_dialog(layers, suggested)
            dialog_status = "lattice mismatch"

        if need_dialog:
            names = [
                row.lbl_name.text().replace("<b>", "").replace("</b>", "")
                for row in self.stack_layer_rows
            ]
            dlg = QDialog(self)
            dlg.setWindowTitle("Forced strain for DFT")
            lay = QVBoxLayout(dlg)
            sug_name = names[suggested]
            sug_pct = strains[suggested]
            a_ref = float(layers[suggested]["struct"].lattice.a)
            iso_lines = []
            for i, (name_i, layer) in enumerate(zip(names, layers)):
                if i == suggested:
                    continue
                a_native = float(layer["struct"].lattice.a)
                iso = CrystalEngine.isotropic_match_strain_percent(a_native, a_ref)
                iso_lines.append(f"Isotropic lattice strain on {name_i}: {iso:+.2f}%")
            suggest_line = (
                f"Suggested reference (smallest total strain): {sug_name} (~{sug_pct:.2f}%)."
            )
            dialog_text = (
                f"Status: {dialog_status}.\n\n"
                "To run DFT, non-reference layer(s) will be stretched/compressed to match the "
                "reference layer's in-plane cell. Band structures will reflect that forced strain.\n\n"
                + "\n".join(iso_lines)
                + ("\n\n" if iso_lines else "\n")
                + suggest_line
                + "\n\nRun QE Relax ions + vdW after Push for better interlayer geometry."
            )
            lay.addWidget(QLabel(dialog_text))
            lay.addWidget(QLabel("Reference layer:"))
            combo = QComboBox()
            for i, name_i in enumerate(names):
                label = name_i + ("  ← suggested (min strain)" if i == suggested else "")
                combo.addItem(label)
            combo.setCurrentIndex(suggested)
            lay.addWidget(combo)
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.button(QDialogButtonBox.Ok).setText("Push anyway")
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)
            lay.addWidget(buttons)
            if dlg.exec() != QDialog.Accepted:
                return
            ref_idx = combo.currentIndex()

        try:
            if kind == "aligned":
                dft_struct = CrystalEngine.build_dft_aligned_stack(layers, vacuum, ref_idx)
                info_note = "aligned"
            else:
                theta = float(self.spin_bernevig_theta.value())
                layers[0]["twist"] = -theta / 2.0
                layers[1]["twist"] = theta / 2.0
                self._refresh_bernevig_valleys(layers)
                dft_struct, info = CrystalEngine.build_dft_twist_stack(
                    layers,
                    vacuum,
                    0,
                    stacking=self.combo_bernevig_stacking.currentText(),
                    max_cells=int(self.spin_bernevig_max_cells.value()),
                    valley=self.combo_bernevig_valley.currentText(),
                )
                info_note = info["status"]
        except Exception as e:
            QMessageBox.critical(self, "DFT cell rebuild failed", str(e))
            return

        default_name = getattr(self, "current_filename", "Heterostructure_Stack")
        name, ok = QInputDialog.getText(
            self, "Workspace Export", "Enter a variable name for this structure:", text=default_name
        )
        if not (ok and name):
            return

        global_workspace.push_crystal_structure(name, dft_struct)
        a, b, c = dft_struct.lattice.a, dft_struct.lattice.b, dft_struct.lattice.c
        QMessageBox.information(
            self,
            "Success",
            f"DFT structure '{name}' sent to Global Workspace ({info_note}).\n"
            f"Cell a={a:.3f} Å, b={b:.3f} Å, c={c:.3f} Å (vacuum={vacuum:.1f} Å).\n"
            f"Load it in the DFT Suite — not the 500 Å viewer canvas.",
        )

    def handle_add_template(self):
        name = self.combo_tpl.currentText()
        struct = CrystalEngine.generate_template_structure(name)
        if struct: self.append_stack_layer(name, struct)
    
    def load_cif_layer(self):
        """Directly loads a 2D CIF into the heterostructure stack without exfoliation."""
        fname, _ = QFileDialog.getOpenFileName(self, 'Open 2D CIF', '', "CIF files (*.cif)")
        if not fname: return
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                struct = Structure.from_file(fname)
            name = fname.split('/')[-1].replace('.cif', '')
            self.append_stack_layer(name, struct)
        except Exception as e:
            QMessageBox.critical(self, "Parser Error", f"Could not parse CIF: {e}")
    
    def move_layer_up(self, row):
        idx = self.stack_layer_rows.index(row)
        if idx > 0:
            self.stack_layer_rows.pop(idx)
            self.stack_layer_rows.insert(idx - 1, row)
            self.scroll_layout.insertWidget(idx - 1, row)

    def move_layer_down(self, row):
        idx = self.stack_layer_rows.index(row)
        if idx < len(self.stack_layer_rows) - 1:
            self.stack_layer_rows.pop(idx)
            self.stack_layer_rows.insert(idx + 1, row)
            self.scroll_layout.insertWidget(idx + 1, row)

    def save_layer_as_template(self, layer_widget):
        import json
        clean_name = layer_widget.lbl_name.text().replace("<b>", "").replace("</b>", "")
        name, ok = QInputDialog.getText(self, "Save Template", "Enter a name for this custom template:", text=clean_name)
        if ok and name:
            templates = {}
            if os.path.exists("user_templates.json"):
                with open("user_templates.json", "r") as f:
                    templates = json.load(f)
            templates[name] = layer_widget.struct.as_dict()
            with open("user_templates.json", "w") as f:
                json.dump(templates, f)
            QMessageBox.information(self, "Success", f"'{name}' has been saved to your templates!")

    def extract_monolayer_from_bulk(self, *args):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Bulk CIF', '', "CIF files (*.cif)")
        if not fname: return

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bulk_struct = Structure.from_file(fname, primitive=False) 
                bulk_struct.translate_sites(list(range(len(bulk_struct))), [0.01, 0.01, 0.01], to_unit_cell=True)
        except Exception as e:
            QMessageBox.critical(self, "Parser Error", f"Could not parse structure file: {str(e)}")
            return

        mode = self.combo_exfoliate_mode.currentText()

        if "van der Waals" in mode:
            try:
                mono, gap = CrystalEngine.extract_monolayer_vdw(bulk_struct)
                name = fname.split('/')[-1].replace('.cif', '') + " (vdW Auto-Mono)"
                self.append_stack_layer(name, mono)
                self.current_filename = name
            except Exception as e:
                QMessageBox.critical(self, "vdW Cleave Error", str(e))
        else:
            # Read directly from our new persistent UI boxes
            num_layers = self.spin_exf_layers.value()
            vacuum_thickness = self.spin_exf_vacuum.value()
            h = self.spin_exf_h.value()
            k = self.spin_exf_k.value()
            l = self.spin_exf_l.value()
            hkl = (h, k, l)
            hkl_str = f"{h} {k} {l}"
            
            try:
                from pymatgen.core.surface import SlabGenerator
                
                # 1. Ask PyMatgen for a MASSIVE slab to guarantee it contains complete molecules
                safe_thickness = max(30.0, num_layers * 15.0) 
                
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    slabgen = SlabGenerator(bulk_struct, miller_index=hkl, min_slab_size=safe_thickness, min_vacuum_size=vacuum_thickness, center_slab=True)
                    slabs = slabgen.get_slabs()
                    
                if not slabs: raise ValueError(f"Could not generate slabs for plane {hkl}")
                
                # 2. 3D Topological Chemical Graphing with Surface Dust Filtering[cite: 2]
                raw_slab = slabs[0]
                dist_mat = raw_slab.distance_matrix 
                
                bond_threshold = 3.2 # Safely connects all tilted intra-layer bonds
                num_sites = len(raw_slab)
                visited = set()
                all_clusters = []
                import numpy as np
                
                # Breadth-First Search (BFS) to map all covalently connected molecules[cite: 2]
                for i in range(num_sites):
                    if i not in visited:
                        queue = [i]
                        current_cluster = []
                        while queue:
                            curr = queue.pop(0)
                            if curr not in visited:
                                visited.add(curr)
                                current_cluster.append(curr)
                                neighbors = np.where((dist_mat[curr] > 0.1) & (dist_mat[curr] < bond_threshold))[0]
                                queue.extend([n for n in neighbors if n not in visited])
                        all_clusters.append(current_cluster)
                
                # FILTER: A full sandwich spans the whole supercell width and has max atoms.
                # We throw away the broken "surface dust" slabs left over by PyMatgen's cut.
                max_size = max([len(c) for c in all_clusters]) if all_clusters else 0
                valid_layers = [c for c in all_clusters if len(c) > max_size * 0.75]
                
                # Sort the surviving valid layers bottom-to-top
                valid_layers.sort(key=lambda indices: np.mean([raw_slab[idx].coords[2] for idx in indices]))
                
                # Grab exactly the number of complete layers requested
                target_layer_indices = valid_layers[:min(num_layers, len(valid_layers))]
                
                sites_to_keep = []
                for indices in target_layer_indices:
                    sites_to_keep.extend([raw_slab[idx] for idx in indices])
                
                
                # Reconstruct the structure safely with dynamic vacuum
                final_coords_temp = [s.coords for s in sites_to_keep]
                z_vals = [c[2] for c in final_coords_temp]
                slab_thickness = max(z_vals) - min(z_vals) if z_vals else 0.0
                total_c = slab_thickness + vacuum_thickness

                new_matrix = raw_slab.lattice.matrix.copy()
                new_matrix[2] = [0, 0, total_c] 
                from pymatgen.core import Lattice
                new_lat = Lattice(new_matrix)
                
                z_center = (max(z_vals) + min(z_vals)) / 2.0
                final_coords = [[c[0], c[1], c[2] - z_center + (total_c / 2.0)] for c in final_coords_temp]
                
                mono_struct = Structure(new_lat, [s.specie for s in sites_to_keep], final_coords, coords_are_cartesian=True)
                
                name = fname.split('/')[-1].replace('.cif', '') + f" ({num_layers}-Layer {hkl_str.replace(' ', '')})"
                self.append_stack_layer(name, mono_struct)
                
                # --- NEW: Set the tight slab as active so we don't push the 500A dummy box! ---
                self.current_structure = mono_struct
                self.current_filename = name
                self.refresh_render()
            except Exception as e:
                QMessageBox.critical(self, "Exfoliation Error", str(e))

    def append_stack_layer(self, name: str, struct: Structure):
        default_z = max([r.spin_z.value() for r in self.stack_layer_rows]) + 3.4 if self.stack_layer_rows else 0.0
        row = StackLayerRow(name, struct, default_z)
        
        # Connect all buttons!
        row.btn_delete.clicked.connect(lambda: self.remove_stack_layer(row))
        row.btn_up.clicked.connect(lambda: self.move_layer_up(row))
        row.btn_down.clicked.connect(lambda: self.move_layer_down(row))
        row.btn_save_tpl.clicked.connect(lambda: self.save_layer_as_template(row))
        
        self.scroll_layout.addWidget(row)
        self.stack_layer_rows.append(row)
        
        for el in set([s.specie.symbol for s in struct]):
            if el not in self.active_colors: self.active_colors[el] = CPK_COLORS.get(el, "#008080")

    def remove_stack_layer(self, row_widget):
        self.scroll_layout.removeWidget(row_widget)
        row_widget.deleteLater()
        self.stack_layer_rows.remove(row_widget)

    def _bernevig_opted_in(self) -> bool:
        """True when a two-layer stack should use the Bernevig builder (not the 7.34 default alone)."""
        if len(self.stack_layer_rows) != 2:
            return False
        if self.combo_bernevig_stacking.currentText() == "AB":
            return True
        return any(abs(float(row.spin_twist.value())) > 1e-6 for row in self.stack_layer_rows)

    def handle_draw_stack(self):
        if not self.stack_layer_rows: return
        if self._bernevig_opted_in():
            ok, err = self._try_bernevig_build()
            if ok:
                return
            self.lbl_moire.setText(f"Bernevig build failed ({err}); drew plain stack.")

        layers_data = [row.get_layer_dict() for row in self.stack_layer_rows]
        
        # 1. Build the math supercell
        self.active_supercell = CrystalEngine.build_heterostructure_stack(layers_data)
        self.current_structure = self.active_supercell
        self.current_filename = "Heterostructure_Stack"
        
        # 2. Extract elements to wake up Tab 1's Color Panel
        if "layer_tag" in self.active_supercell.site_properties:
            unique_elements = sorted(list(set(self.active_supercell.site_properties["layer_tag"])))
        else:
            unique_elements = sorted(list(set([site.specie.symbol for site in self.active_supercell])))

        # Ensure CPK fallback colors are loaded for any new elements
        for el in unique_elements:
            base_el = el.split('_')[0] if '_' in el else el
            if el not in self.active_colors: 
                self.active_colors[el] = CPK_COLORS.get(base_el, "#008080")

        self.tab_view.build_color_panel(unique_elements)
        
        # Update Tab 2 CDW Target Elements Dropdown
        self.combo_cdw_el.blockSignals(True)
        self.combo_cdw_el.clear()
        self.combo_cdw_el.addItem("All Elements")
        for el in unique_elements:
            # Strip layer tags (e.g., V_L1 becomes V) so CDW targets the element
            clean_el = el.split('_')[0] if '_' in el else el
            if self.combo_cdw_el.findText(clean_el) == -1:
                self.combo_cdw_el.addItem(clean_el)
        self.combo_cdw_el.blockSignals(False)
        
        # 3. Draw to screen
        self.refresh_render()
        if hasattr(self.renderer, 'set_camera_preset'):
            self.renderer.set_camera_preset('z')

    def _refresh_bernevig_valleys(self, layers):
        current = self.combo_bernevig_valley.currentText()
        bravais = classify_bravais(layers[0]["struct"].lattice.matrix[:2, :2])
        items = {
            "hexagonal": ["Γ", "K", "M", "nHSP"],
            "square": ["Γ", "X", "M", "nHSP"],
            "rectangular": ["HSP", "nHSP"],
            "oblique": ["HSP", "nHSP"],
        }[bravais]
        self.combo_bernevig_valley.blockSignals(True)
        self.combo_bernevig_valley.clear()
        self.combo_bernevig_valley.addItems(items)
        if current in items:
            self.combo_bernevig_valley.setCurrentText(current)
        self.combo_bernevig_valley.blockSignals(False)

    def _try_bernevig_build(self) -> tuple[bool, str]:
        """Run the Bernevig builder; on success update viewer + row twists. No modal dialogs."""
        if len(self.stack_layer_rows) != 2:
            return False, "Requires exactly 2 stacked layers."

        layers = [row.get_layer_dict() for row in self.stack_layer_rows]
        theta = float(self.spin_bernevig_theta.value())
        layers[0]["twist"] = -theta / 2.0
        layers[1]["twist"] = theta / 2.0
        try:
            self._refresh_bernevig_valleys(layers)
            struct, info = CrystalEngine.build_dft_twist_stack(
                layers,
                float(self.spin_stack_vacuum.value()),
                0,
                stacking=self.combo_bernevig_stacking.currentText(),
                max_cells=int(self.spin_bernevig_max_cells.value()),
                valley=self.combo_bernevig_valley.currentText(),
            )
        except Exception as e:
            return False, str(e)

        # Later Push follows this twist.
        self.stack_layer_rows[0].spin_twist.setValue(-theta / 2.0)
        self.stack_layer_rows[1].spin_twist.setValue(theta / 2.0)

        self.active_supercell = struct
        self.current_structure = struct
        self.current_filename = "Bernevig_Twisted_Bilayer"
        self.refresh_render()
        self.lbl_moire.setText(
            f"Commensurate θ={info['theta_used']:.5f}° "
            f"(asked {info['theta_requested']:.2f}°)  "
            f"N={info['n_cells']}  {info['stacking']}  "
            f"Q={info['q_lattice']}  atoms={len(struct)}"
        )
        return True, ""

    def handle_bernevig_twist(self, *_args):
        ok, err = self._try_bernevig_build()
        if ok:
            return
        self.lbl_moire.setText(err if len(self.stack_layer_rows) == 2 else f"Error: {err}")
        if len(self.stack_layer_rows) == 2:
            QMessageBox.critical(self, "Bernevig twist build failed", err)

    def handle_moire(self):
        self.handle_bernevig_twist()

    def handle_draw_bz(self):
        if not getattr(self, 'current_structure', None): return
        
        # 1. Ask Engine for pure math
        bz_data = CrystalEngine.calculate_brillouin_zone(self.current_structure)
        if not bz_data: return
        
        self.renderer.clear_scene()
        import numpy as np
        
        # --- OVERLAY CRYSTAL STRUCTURE ---
        if getattr(self, 'chk_overlay_atoms', None) and self.chk_overlay_atoms.isChecked():
            is_shiny = self.tab_view.chk_shiny.isChecked()
            scale_mod = self.tab_view.spin_radius.value()
            style = self.tab_view.combo_style.currentIndex()
            
            # Draw real-space atoms and bonds
            self.renderer.draw_atoms(self.current_structure, self.active_colors, scale_mod=scale_mod, is_shiny=is_shiny, erased_atoms=self.erased_atoms)
            if style == 0:
                self.renderer.draw_bonds(self.current_structure, self.active_colors, is_shiny=is_shiny, erased_bonds=self.erased_bonds, erased_atoms=self.erased_atoms)
            
            # Draw Real-Space Axes to correlate with BZ direction
            if self.tab_view.chk_axes.isChecked():
                is_dummy = self.current_structure.lattice.a >= 499.0
                if not is_dummy:
                    prim_mat = CrystalEngine.get_universal_primitive_matrix(self.current_structure) if self.tab_view.chk_show_primitive.isChecked() else None
                    conv_mat = self.current_structure.lattice.matrix if self.tab_view.chk_show_conventional.isChecked() else None
                    self.renderer.draw_axes(conventional_matrix=conv_mat, primitive_matrix=prim_mat)
        
        scale = self.spin_bz_scale.value()
        style_idx = self.combo_bz_style.currentIndex()
        scaled_points = np.array(bz_data["points"]) * scale
        scaled_hull_points = np.array(bz_data["hull_points"]) * scale
        
        self.active_bz_points = scaled_hull_points
        self.active_bz_faces = bz_data["simplices"]
        self.active_bz_edges = bz_data["edges"]
        self.active_bz_style = style_idx
        self.active_surf_bz_edges = []
        
        # 2. Command Renderer to draw Bulk BZ
        if hasattr(self.renderer, 'draw_brillouin_zone'):
            self.renderer.draw_brillouin_zone(scaled_hull_points, np.array(bz_data["simplices"]), style_idx, edges=bz_data["edges"])
            
        # 3. Route Surface Projection (if checked)
        if getattr(self, 'chk_surf_bz', None) and self.chk_surf_bz.isChecked():
            h, k, l = self.spin_bz_h.value(), self.spin_bz_k.value(), self.spin_bz_l.value()
            if h != 0 or k != 0 or l != 0:
                surf_data = CrystalEngine.calculate_surface_projection(scaled_points, self.current_structure, h, k, l)
                if surf_data and hasattr(self.renderer, 'draw_surface_bz'):
                    
                    normal = np.array(surf_data["normal"])
                    base_plane = np.array(surf_data["origin_plane"])
                    
                    # Calculate visual hover distance
                    offset_dist = np.max(scaled_points) * 1.5
                    hover_plane = base_plane + normal * offset_dist
                    
                    silh_3d = np.array(surf_data["silhouette_3d"])
                    proj_bounds = np.array(surf_data["projected_bounds"])
                    hover_bounds = proj_bounds + normal * offset_dist
                    
                    proj_lines = [(silh_3d[i], hover_bounds[i]) for i in range(len(silh_3d))]
                    
                    # Command renderer
                    self.renderer.draw_surface_bz(
                        base_plane=base_plane, 
                        hover_plane=hover_plane, 
                        simplices=np.array(surf_data["simplices"]), 
                        proj_lines=proj_lines
                    )
                    
                    # Cache both planes + dashed lines for 3ds Max/Blender
                    for i in range(len(proj_bounds)):
                        self.active_surf_bz_edges.append((proj_bounds[i], proj_bounds[(i+1)%len(proj_bounds)]))
                    for i in range(len(hover_bounds)):
                        self.active_surf_bz_edges.append((hover_bounds[i], hover_bounds[(i+1)%len(hover_bounds)]))
                    for line in proj_lines:
                        self.active_surf_bz_edges.append(line)

        # 4. Push updates to screen
        if hasattr(self.renderer, 'plotter') and hasattr(self.renderer.plotter, 'render'):
            self.renderer.plotter.render()
        elif hasattr(self.renderer, 'canvas'):
            self.renderer.canvas.draw_idle()

    def closeEvent(self, event):
        # 1. Trip the kill-switch immediately to block all incoming UI signals
        self._is_closing = True 
        print("close suite Crystal Viewer")
        
        # 2. Safely isolate the PyVista GPU Engine without nuking the shared context
        if hasattr(self, 'renderer_gpu') and hasattr(self.renderer_gpu, 'plotter'):
            try:
                # Stop VTK from listening to any more Qt events
                if hasattr(self.renderer_gpu.plotter, 'iren'):
                    self.renderer_gpu.plotter.iren.remove_all_observers()
                
                # Detach the PyVista widget from the main Qt window so destruction is localized
                self.renderer_gpu.plotter.setParent(None)
                
                # Cleanly close the plotter (DO NOT use Finalize() in multi-window apps)
                self.renderer_gpu.plotter.close()
            except Exception:
                pass
                
        # 3. Safely kill the Matplotlib CPU Engine
        if hasattr(self, 'renderer_cpu') and hasattr(self.renderer_cpu, 'figure'):
            try:
                import matplotlib.pyplot as plt
                plt.close(self.renderer_cpu.figure)
            except Exception:
                pass
        
        event.accept()