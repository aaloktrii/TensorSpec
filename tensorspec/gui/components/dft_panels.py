import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
                               QDoubleSpinBox, QPushButton, QGroupBox, QLabel, 
                               QSpinBox, QLineEdit, QComboBox, QCheckBox, QFileDialog)

from tensorspec.gui.services.cluster_utils import (
    is_remote_target,
    populate_compute_target_combo,
    selected_cluster,
)
from tensorspec.gui.services.compute_mode import (
    effective_band_diag,
    hybrid_exec_summary,
    is_hybrid_mode,
)

class TightBindingPanel(QWidget):
    """
    Isolated UI Component containing all inputs for the Tight Binding engine.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_w90_file = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        tb_group = QGroupBox("Tight Binding Parameters")
        tb_form = QFormLayout(tb_group)

        # --- Compute mode: local only vs hybrid (server compute) ---
        self.combo_tb_target = QComboBox()
        populate_compute_target_combo(self.combo_tb_target)
        self.combo_tb_target.setToolTip(
            "Local only = everything on this Mac.\n"
            "Hybrid = upload TB job to your remote cluster, "
            "run band diag there, download result, plot here."
        )
        self.lbl_tb_exec = QLabel("")
        self.lbl_tb_exec.setWordWrap(True)
        self.lbl_tb_exec.setStyleSheet("color: #666; font-size: 10px;")

        self.chk_hybrid_fast = QCheckBox("Hybrid fast path: Grizzly CUDA on server")
        self.chk_hybrid_fast.setChecked(True)
        self.chk_hybrid_fast.setToolTip(
            "When Hybrid is selected, auto-use GrizzlyME + CUDA on the remote "
            "cluster for band diagonalization (best for large Wannier90 models)."
        )
        self.chk_hybrid_fast.setStyleSheet("color: #2b5c8f; font-weight: bold;")

        tb_form.addRow("Compute mode:", self.combo_tb_target)
        tb_form.addRow(self.lbl_tb_exec)
        tb_form.addRow(self.chk_hybrid_fast)
        self.combo_tb_target.currentIndexChanged.connect(self._sync_tb_target)
        self.chk_hybrid_fast.stateChanged.connect(self._sync_tb_target)
        
        # --- Dimension Toggle ---
        self.combo_k_mode = QComboBox()
        self.combo_k_mode.addItems(["1D High-Symmetry Path", "2D k-Mesh (kx, ky)"])
        tb_form.addRow("k-Space Grid:", self.combo_k_mode)

        # --- ARPES Isoenergy Control ---
        self.spin_iso = QDoubleSpinBox()
        self.spin_iso.setRange(-20.0, 20.0)
        self.spin_iso.setValue(0.0)
        self.spin_iso.setSingleStep(0.1)
        self.spin_iso.setEnabled(False) 
        tb_form.addRow("2D Isoenergy Cut (eV):", self.spin_iso)
        
        self.combo_k_mode.currentTextChanged.connect(
            lambda text: self.spin_iso.setEnabled("2D" in text)
        )
        
        # --- K-Path Template Toggle ---
        self.combo_k_template = QComboBox()
        self.combo_k_template.addItems([
            "Auto-Detect BZ Path (PyMatgen)",
            "Arbitrary Custom Path",
            "Hexagonal (Template)", 
            "Rectangular / Orthorhombic (Template)", 
            "Square / Tetragonal (Template)"
        ])
        tb_form.addRow("1D Path Template:", self.combo_k_template)

        # --- Spin-Orbit Coupling Controls ---
        soc_layout = QHBoxLayout()
        self.chk_soc = QCheckBox("Enable Spin-Orbit Coupling")
        self.chk_soc.setStyleSheet("font-weight: bold; color: #8A2BE2;")
        
        self.spin_soc_val = QDoubleSpinBox()
        self.spin_soc_val.setRange(0.0, 5.0)
        self.spin_soc_val.setValue(0.5)
        self.spin_soc_val.setSingleStep(0.1)
        self.spin_soc_val.setSuffix(" eV")
        self.spin_soc_val.setToolTip("SOC Strength (\u03BB)")
        
        soc_layout.addWidget(self.chk_soc)
        soc_layout.addWidget(self.spin_soc_val)
        tb_form.addRow("Relativistic:", soc_layout)

        # --- Fat Band Projection Controls ---
        self.combo_projection = QComboBox()
        self.combo_projection.addItem("None (Standard Lines)")
        tb_form.addRow("Fat Band Target:", self.combo_projection)
        
        self.spin_soc_val.setEnabled(False)
        self.chk_soc.stateChanged.connect(lambda state: self.spin_soc_val.setEnabled(bool(state)))

        # --- Wannier90 Importer ---
        w90_group = QGroupBox("Ab Initio Import (Wannier90)")
        w90_layout = QVBoxLayout(w90_group)
        
        self.btn_load_w90 = QPushButton("📂 Load wannier90_hr.dat")
        self.btn_load_w90.setToolTip("Select the Wannier90 hopping file. Make sure wannier90.wout or scf.out is in the same folder!")
        self.btn_load_w90.setStyleSheet("background-color: #5cb85c; color: white; font-weight: bold;")
        
        self.lbl_w90_warning = QLabel("⚠️ Note: Keep wannier90.wout or scf.out in same folder for lattice alignment!")
        self.lbl_w90_warning.setStyleSheet("color: #d9534f; font-size: 10px; font-weight: bold;")
        
        self.lbl_w90_status = QLabel("Status: Using Manual Slater-Koster parameters.")
        self.lbl_w90_status.setStyleSheet("color: gray; font-size: 10px;")
        
        # --- NEW OVERLAY CHECKBOX ---
        self.chk_overlay_w90 = QCheckBox("Overlay Native W90 Bands (Red Dashed)")
        self.chk_overlay_w90.setChecked(True)
        self.chk_overlay_w90.setEnabled(False) # Disabled until a file is loaded

        self.spin_hop_tol = QDoubleSpinBox()
        self.spin_hop_tol.setDecimals(8)
        self.spin_hop_tol.setRange(1e-10, 0.1)
        self.spin_hop_tol.setValue(1e-4)
        self.spin_hop_tol.setSingleStep(1e-5)
        self.spin_hop_tol.setToolTip(
            "Drop hoppings with |t| ≤ this (eV). Higher = fewer hops = faster "
            "Prepare-for-ARPES on huge third-party hr.dat. Default 1e-4."
        )
        
        w90_layout.addWidget(self.btn_load_w90)
        w90_layout.addWidget(self.lbl_w90_warning)
        w90_layout.addWidget(self.chk_overlay_w90)
        w90_layout.addWidget(self.lbl_w90_status)
        hop_tol_row = QHBoxLayout()
        hop_tol_row.addWidget(QLabel("Hop cutoff |t|>"))
        hop_tol_row.addWidget(self.spin_hop_tol)
        hop_tol_row.addWidget(QLabel("eV (ARPES prep)"))
        w90_layout.addLayout(hop_tol_row)
        tb_form.addRow(w90_group)
        
        self.btn_load_w90.clicked.connect(self.load_w90_file)
        
        # --- Custom Arbitrary Path Inputs ---
        self.custom_k_widget = QWidget()
        custom_layout = QFormLayout(self.custom_k_widget)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        
        self.line_k_coords = QLineEdit("0,0,0; 0.5,0,0; 0.5,0.5,0; 0,0,0")
        self.line_k_labels = QLineEdit("G; X; M; G")
        
        # --- Hamiltonian Mode Toggle ---
        self.combo_tb_mode = QComboBox()
        self.combo_tb_mode.addItems(["Simple Scalar (Isotropic)", "Slater-Koster (Rigorous)"])
        tb_form.addRow("Hamiltonian Mode:", self.combo_tb_mode)

        custom_layout.addRow("Coords (frac):", self.line_k_coords)
        custom_layout.addRow("Labels:", self.line_k_labels)
        tb_form.addRow(self.custom_k_widget)
        
        self.custom_k_widget.setVisible(False)
        self.combo_k_template.currentTextChanged.connect(
            lambda text: self.custom_k_widget.setVisible("Arbitrary" in text)
        )
        
        # --- Resolution Control ---
        self.spin_k_res = QSpinBox()
        self.spin_k_res.setRange(10, 2000)
        self.spin_k_res.setValue(100)
        self.spin_k_res.setSingleStep(50)
        tb_form.addRow("Points per Segment:", self.spin_k_res)

        # --- Band diagonalization backend (explicit; default Chinook CPU) ---
        self.combo_band_diag = QComboBox()
        self.combo_band_diag.addItem("Chinook (CPU)", "chinook")
        self.combo_band_diag.addItem("GrizzlyME", "grizzly")
        self.combo_band_diag.setToolTip(
            "Chinook = default CPU reference. GrizzlyME = optional PyTorch accel "
            "(pick CPU or CUDA below; requires pip install grizzlyme)."
        )
        self.combo_band_device = QComboBox()
        self.combo_band_device.addItem("CPU", "cpu")
        self.combo_band_device.addItem("CUDA (GPU)", "cuda")
        self.combo_band_device.setEnabled(False)
        self.combo_band_device.setToolTip(
            "GrizzlyME only. Default CPU; pick CUDA for GPU clusters."
        )
        self.combo_band_diag.currentIndexChanged.connect(self._sync_band_diag_device)
        self.combo_band_device.currentIndexChanged.connect(self._sync_band_diag_device)
        tb_form.addRow("Band diag engine:", self.combo_band_diag)
        tb_form.addRow("Grizzly device:", self.combo_band_device)

        self.spin_qe_fermi = QDoubleSpinBox()
        self.spin_qe_fermi.setRange(-100.0, 100.0)
        self.spin_qe_fermi.setDecimals(4)
        self.spin_qe_fermi.setValue(0.0)
        self.spin_qe_fermi.setSingleStep(0.1)
        self.spin_qe_fermi.setToolTip(
            "Fermi for W90 bands (eV). Auto: nscf/scf/FERMI_ENERGY.txt, or "
            "W90 midgap at a near-touch (Dirac) when that sits off QE EF. "
            "Folded into H at Prepare; not the same as On-site E."
        )
        self.lbl_qe_fermi_src = QLabel("QE EF source: none — enter if known")
        self.lbl_qe_fermi_src.setStyleSheet("color: gray; font-size: 10px;")
        self._qe_fermi_auto = True
        self.spin_qe_fermi.valueChanged.connect(self._on_qe_fermi_edited)
        tb_form.addRow("QE Fermi (eV):", self.spin_qe_fermi)
        tb_form.addRow("", self.lbl_qe_fermi_src)

        self.spin_onsite = QDoubleSpinBox()
        self._sync_tb_target()
        self.spin_onsite.setRange(-10.0, 10.0)
        self.spin_onsite.setValue(0.0)
        self.spin_onsite.setSingleStep(0.1)
        self.spin_onsite.setToolTip(
            "Extra rigid on-site shift (eV) after QE Fermi is folded into H."
        )
        tb_form.addRow("On-site E (eV):", self.spin_onsite)

        # --- NEW: Orbital-Specific Energy Shifts ---
        self.spin_onsite_s = QDoubleSpinBox()
        self.spin_onsite_s.setRange(-50.0, 50.0); self.spin_onsite_s.setValue(-10.0); self.spin_onsite_s.setSingleStep(0.5)
        self.spin_onsite_s.setToolTip("s-orbital on-site energy (eV)")

        self.spin_onsite_p = QDoubleSpinBox()
        self.spin_onsite_p.setRange(-50.0, 50.0); self.spin_onsite_p.setValue(-2.0); self.spin_onsite_p.setSingleStep(0.5)
        self.spin_onsite_p.setToolTip("p-orbital on-site energy (eV)")

        self.spin_onsite_d = QDoubleSpinBox()
        self.spin_onsite_d.setRange(-50.0, 50.0); self.spin_onsite_d.setValue(0.0); self.spin_onsite_d.setSingleStep(0.5)
        self.spin_onsite_d.setToolTip("d-orbital on-site energy (eV)")

        orb_layout = QHBoxLayout()
        orb_layout.addWidget(QLabel("s:"))
        orb_layout.addWidget(self.spin_onsite_s)
        orb_layout.addWidget(QLabel("p:"))
        orb_layout.addWidget(self.spin_onsite_p)
        orb_layout.addWidget(QLabel("d:"))
        orb_layout.addWidget(self.spin_onsite_d)
        tb_form.addRow("Orbital Shifts:", orb_layout)

        # --- NEW: Energy Zoom Controls ---
        self.spin_emin = QDoubleSpinBox()
        self.spin_emin.setRange(-50.0, 50.0); self.spin_emin.setValue(-6.0); self.spin_emin.setSingleStep(0.5)
        self.spin_emax = QDoubleSpinBox()
        self.spin_emax.setRange(-50.0, 50.0); self.spin_emax.setValue(6.0); self.spin_emax.setSingleStep(0.5)
        
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Min:"))
        range_layout.addWidget(self.spin_emin)
        range_layout.addWidget(QLabel("Max:"))
        range_layout.addWidget(self.spin_emax)
        tb_form.addRow("Y-Axis Zoom (eV):", range_layout)

        # Hopping Shells
        self.spin_t1 = QDoubleSpinBox(); self.spin_t1.setRange(-10.0, 10.0); self.spin_t1.setValue(2.7); self.spin_t1.setMinimumWidth(65)
        self.spin_r1 = QDoubleSpinBox(); self.spin_r1.setRange(0.0, 10.0); self.spin_r1.setValue(1.6); self.spin_r1.setMinimumWidth(65)
        
        self.spin_t2 = QDoubleSpinBox(); self.spin_t2.setRange(-10.0, 10.0); self.spin_t2.setValue(0.0); self.spin_t2.setMinimumWidth(65)
        self.spin_r2 = QDoubleSpinBox(); self.spin_r2.setRange(0.0, 10.0); self.spin_r2.setValue(2.6); self.spin_r2.setMinimumWidth(65)
        
        self.spin_t3 = QDoubleSpinBox(); self.spin_t3.setRange(-10.0, 10.0); self.spin_t3.setValue(0.0); self.spin_t3.setMinimumWidth(65)
        self.spin_r3 = QDoubleSpinBox(); self.spin_r3.setRange(0.0, 10.0); self.spin_r3.setValue(3.1); self.spin_r3.setMinimumWidth(65)
        
        self.spin_t4 = QDoubleSpinBox(); self.spin_t4.setRange(-10.0, 10.0); self.spin_t4.setValue(-0.3); self.spin_t4.setMinimumWidth(65)
        self.spin_r4 = QDoubleSpinBox(); self.spin_r4.setRange(0.0, 15.0); self.spin_r4.setValue(4.5); self.spin_r4.setMinimumWidth(65)
        
        shells = [
            (self.spin_t1, self.spin_r1), 
            (self.spin_t2, self.spin_r2), 
            (self.spin_t3, self.spin_r3),
            (self.spin_t4, self.spin_r4)
        ]
        for i, (t_spin, r_spin) in enumerate(shells, start=1):
            row = QHBoxLayout()
            row.addWidget(t_spin)
            row.addWidget(QLabel("Max Å:"))
            row.addWidget(r_spin)
            tb_form.addRow(f"Hopping t{i}:", row)

        main_layout.addWidget(tb_group)

    def is_remote_target(self) -> bool:
        return is_remote_target(self.combo_tb_target)

    def is_hybrid_mode(self) -> bool:
        return is_hybrid_mode(self.combo_tb_target)

    def get_selected_cluster(self):
        return selected_cluster(self.combo_tb_target)

    def hybrid_auto_gpu(self) -> bool:
        return bool(self.chk_hybrid_fast.isChecked())

    def _sync_tb_target(self, _index=None) -> None:
        hybrid = is_hybrid_mode(self.combo_tb_target)
        self.chk_hybrid_fast.setVisible(hybrid)
        self.lbl_tb_exec.setText(hybrid_exec_summary(self.combo_tb_target))
        if hybrid and self.active_w90_file:
            self.lbl_tb_exec.setStyleSheet("color: #2b5c8f; font-size: 10px;")
        else:
            self.lbl_tb_exec.setStyleSheet("color: #666; font-size: 10px;")

    def resolved_band_diag_settings(self) -> tuple[str, str]:
        """UI engine/device, with hybrid fast-path override when enabled."""
        ui_engine, ui_device = self.band_diag_settings()
        return effective_band_diag(
            self.combo_tb_target,
            ui_engine,
            ui_device,
            auto_gpu=self.hybrid_auto_gpu(),
            w90_loaded=bool(self.active_w90_file),
        )

    def hop_tol(self) -> float:
        return float(self.spin_hop_tol.value())

    def band_diag_settings(self) -> tuple[str, str]:
        """(diag_engine, device) exactly as selected in the UI."""
        engine = str(self.combo_band_diag.currentData() or "chinook")
        if engine != "grizzly":
            return engine, "cpu"
        device = str(self.combo_band_device.currentData() or "cpu")
        return engine, device

    def _sync_band_diag_device(self, _index=None):
        use_grizzly = self.combo_band_diag.currentData() == "grizzly"
        self.combo_band_device.setEnabled(use_grizzly)

    def qe_fermi_eV(self) -> float:
        return float(self.spin_qe_fermi.value())

    def _set_qe_fermi_ui(self, value: float, source: str) -> None:
        self._qe_fermi_auto = True
        self.spin_qe_fermi.blockSignals(True)
        self.spin_qe_fermi.setValue(float(value))
        self.spin_qe_fermi.blockSignals(False)
        if source == "none":
            self.lbl_qe_fermi_src.setText("QE EF source: none — enter if known")
            self.lbl_qe_fermi_src.setStyleSheet("color: #d9534f; font-size: 10px;")
        else:
            self.lbl_qe_fermi_src.setText(
                f"QE EF: {float(value):.4f} eV (from {source})"
            )
            self.lbl_qe_fermi_src.setStyleSheet("color: #2b5c8f; font-size: 10px;")

    def _on_qe_fermi_edited(self, _value=None) -> None:
        if self._qe_fermi_auto:
            self._qe_fermi_auto = False
            return
        self.lbl_qe_fermi_src.setText(
            f"QE EF: {self.qe_fermi_eV():.4f} eV (manual)"
        )
        self.lbl_qe_fermi_src.setStyleSheet("color: #8a6d3b; font-size: 10px;")

    def load_w90_file(self):
        """Opens a file dialog to load the Wannier90 hopping data."""
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Wannier90 HR File', '', "Data files (*.dat);;All files (*.*)")
        if fname:
            import os
            from tensorspec.core.dft.qe_fermi import detect_plot_fermi_eV

            work_dir = os.path.dirname(fname)
            wout = os.path.join(work_dir, "wannier90.wout")
            scf_out = os.path.join(work_dir, "scf.out")
            if not os.path.exists(wout) and not os.path.exists(scf_out):
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Missing Grid Alignment File", 
                                  "No 'wannier90.wout' or 'scf.out' found in the same folder as the hr.dat file.\n\n"
                                  "The band structure might look completely wrong because TensorSpec cannot align the crystal lattice.\n"
                                  "Please place 'wannier90.wout' in the same folder!")
            
            self.active_w90_file = fname
            filename_short = fname.split('/')[-1]
            self.lbl_w90_status.setText(f"Status: Using {filename_short}")
            self.lbl_w90_status.setStyleSheet("color: blue; font-weight: bold;")
            # Prefer W90 midgap@K when QE EF sits off a Dirac/semimetal touch.
            ef, src = detect_plot_fermi_eV(work_dir)
            self._set_qe_fermi_ui(ef, src)
            self._qe_fermi_auto = False
            self._sync_tb_target()
            
            self.spin_t1.setEnabled(False)
            self.spin_t2.setEnabled(False)
            self.spin_t3.setEnabled(False)

            self.chk_overlay_w90.setEnabled(True)