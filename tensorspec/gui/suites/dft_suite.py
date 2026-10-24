import sys
import os
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QGroupBox, QComboBox, QMessageBox, 
                               QInputDialog, QSplitter, QScrollArea, QFileDialog, QLabel, QTextEdit,
                               QTabWidget, QStackedWidget, QFrame)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import the core math router and workspace
from tensorspec.core.dft_engine import DFTEngineRouter
from tensorspec.core.workspace import global_workspace

# Import the isolated UI components
from tensorspec.gui.components.dft_panels import TightBindingPanel
from tensorspec.gui.components.qe_generator_panel import QEGeneratorPanel
from tensorspec.gui.components.sprkkr_panels import SPRKKRDftPanel
from tensorspec.core.dft.tb_remote_client import (
    TBCancelled,
    build_job_payload,
    kill_remote_tb_job,
    run_remote_tb_bands,
)
from tensorspec.core.dft.w90_hr_stats import format_w90_cost_warning, quick_w90_hr_stats
from tensorspec.core.dft.w90_tb_cache import cache_key as w90_cache_key_for
from tensorspec.gui.services.cluster_utils import cluster_display_name


def _recip_matrix_for_workspace(chinook_engine, structure):
    """Match band k-path frame: QE A_qe when parsed from scf/wout, else CIF pymatgen."""
    if structure is None:
        return None, "none"
    a_qe = getattr(chinook_engine, "A_qe", None) if chinook_engine is not None else None
    if a_qe is not None:
        b = 2 * np.pi * np.linalg.inv(np.asarray(a_qe, dtype=float)).T
        return b, "qe"
    return structure.lattice.reciprocal_lattice.matrix, "cif"


def _orbital_positions_for_workspace(chinook_engine, basis, structure):
    """Prefer Wannier WF centres from Chinook basis; fall back to CIF atom sites."""
    if chinook_engine is not None and basis is not None:
        orbs = chinook_engine._orbital_list_from_basis(basis)
        if orbs:
            pos = []
            for orb in orbs:
                raw = getattr(orb, "pos", None)
                if raw is None:
                    pos = []
                    break
                pos.append(np.asarray(raw, dtype=float).tolist())
            if len(pos) == len(orbs):
                return pos
    if structure is not None:
        return [site.coords.tolist() for site in structure]
    return []


class TBBandRunnerThread(QThread):
    """Run TB band diag on remote cluster; download tb_bands_result.npz."""

    log_signal = Signal(str)
    finished_signal = Signal(bool, str, object)

    def __init__(self, cluster, job, w90_filepath, parent=None):
        super().__init__(parent)
        self.cluster = cluster
        self.job = job
        self.w90_filepath = w90_filepath
        self._cancel = False

    def request_cancel(self):
        self._cancel = True

    def run(self):
        try:
            result = run_remote_tb_bands(
                self.cluster,
                self.job,
                self.w90_filepath,
                log_fn=self.log_signal.emit,
                cancel_check=lambda: self._cancel,
            )
            if self._cancel:
                self.finished_signal.emit(False, "Cancelled.", None)
                return
            self.finished_signal.emit(True, "Hybrid bands downloaded.", result)
        except TBCancelled:
            self.finished_signal.emit(False, "Hybrid run cancelled.", None)
        except Exception as exc:
            self.finished_signal.emit(False, str(exc), None)


class TBBandLocalRunnerThread(QThread):
    """Local band diag off GUI thread (large W90 still blocks this worker)."""

    finished_signal = Signal(bool, str, object)

    def __init__(self, engine, k_vecs, solve_kwargs, fermi_energy, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.k_vecs = k_vecs
        self.solve_kwargs = solve_kwargs
        self.fermi_energy = fermi_energy
        self._cancel = False

    def request_cancel(self):
        self._cancel = True

    def run(self):
        try:
            if self._cancel:
                self.finished_signal.emit(False, "Cancelled.", None)
                return
            eigenvalues, eigenvectors, orb_labels = self.engine.solve_bands(
                self.k_vecs, **self.solve_kwargs
            )
            if self._cancel:
                self.finished_signal.emit(False, "Cancelled.", None)
                return
            result = (
                np.asarray(eigenvalues, dtype=float)
                if self.solve_kwargs.get("w90_filepath")
                else np.asarray(eigenvalues, dtype=float) - self.fermi_energy,
                eigenvectors,
                orb_labels,
            )
            self.finished_signal.emit(True, "Local bands done.", result)
        except Exception as exc:
            self.finished_signal.emit(False, str(exc), None)


class PrepareTBForARPESThread(QThread):
    """Parse/build Wannier TB for workspace push without full band path."""

    log_signal = Signal(str)
    finished_signal = Signal(bool, str, object)

    def __init__(self, engine, w90_filepath, prepare_kwargs, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.w90_filepath = w90_filepath
        self.prepare_kwargs = prepare_kwargs
        self._cancel = False

    def request_cancel(self):
        self._cancel = True

    def run(self):
        try:
            if self._cancel:
                self.finished_signal.emit(False, "Cancelled.", None)
                return
            result = self.engine.prepare_tb_for_arpes(
                self.w90_filepath, **self.prepare_kwargs
            )
            if self._cancel:
                self.finished_signal.emit(False, "Cancelled.", None)
                return
            self.finished_signal.emit(True, "TB ready for ARPES push.", result)
        except Exception as exc:
            self.finished_signal.emit(False, str(exc), None)


class DFTSuite(QWidget):
    """
    Main UI Coordinator for the DFT Suite.
    Manages layout and plots, delegating physics to DFTEngineRouter 
    and UI inputs to modular panels.
    """
    def __init__(self, parent=None):
        print("open suite DFT")
        super().__init__(parent)
        self.setWindowTitle("TensorSpec - DFT & Tight Binding Suite")
        self.resize(900, 700)
        
        # Initialize Core Math Engine Router
        self.engine = DFTEngineRouter()
        
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Left: tabbed controls (no mile-long scroll of everything at once) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.left_tabs = QTabWidget()

        # Tab 1 — Structure + engine pick
        tab_struct = QWidget()
        struct_layout = QVBoxLayout(tab_struct)
        ws_group = QGroupBox("Crystal Structure (Workspace)")
        ws_layout = QVBoxLayout(ws_group)
        row1 = QHBoxLayout()
        self.ws_combo = QComboBox()
        self.ws_combo.addItem("No structures available")
        self.btn_ws_refresh = QPushButton("🔄 Refresh")
        row1.addWidget(self.ws_combo)
        row1.addWidget(self.btn_ws_refresh)
        self.btn_ws_load = QPushButton("📥 Load Basis into Engine")
        ws_layout.addLayout(row1)
        ws_layout.addWidget(self.btn_ws_load)
        struct_layout.addWidget(ws_group)

        engine_group = QGroupBox("First Principles Engine")
        engine_layout = QVBoxLayout(engine_group)
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(
            ["Quantum Espresso (Pseudopotential)", "SPRKKR (All-Electron KKR)"]
        )
        engine_layout.addWidget(self.engine_combo)
        struct_layout.addWidget(engine_group)
        struct_layout.addStretch()
        self.left_tabs.addTab(tab_struct, "Structure")

        # Tab 2 — QE / SPRKKR inputs (each panel scrolls inside its own tab)
        tab_dft = QWidget()
        dft_layout = QVBoxLayout(tab_dft)
        dft_layout.setContentsMargins(0, 0, 0, 0)
        self.engine_stack = QStackedWidget()
        self.qe_panel = QEGeneratorPanel(self.engine)
        self.engine_stack.addWidget(self.qe_panel)
        self.sprkkr_panel = SPRKKRDftPanel(self.engine)
        self.engine_stack.addWidget(self.sprkkr_panel)
        dft_inner = QWidget()
        dft_inner_layout = QVBoxLayout(dft_inner)
        dft_inner_layout.setContentsMargins(0, 0, 0, 0)
        dft_inner_layout.addWidget(self.engine_stack)
        dft_scroll = QScrollArea()
        dft_scroll.setWidget(dft_inner)
        dft_scroll.setWidgetResizable(True)
        dft_scroll.setFrameShape(QFrame.Shape.NoFrame)
        dft_layout.addWidget(dft_scroll)
        self.left_tabs.addTab(tab_dft, "DFT / Wannier")

        self.engine_combo.currentIndexChanged.connect(self.engine_stack.setCurrentIndex)
        self.engine_combo.currentIndexChanged.connect(self.on_engine_changed)

        # Tab 3 — Tight binding + band actions
        tab_tb = QWidget()
        tb_layout = QVBoxLayout(tab_tb)
        tb_layout.setContentsMargins(0, 0, 0, 0)
        self.tb_panel = TightBindingPanel()
        tb_scroll = QScrollArea()
        tb_scroll.setWidget(self.tb_panel)
        tb_scroll.setWidgetResizable(True)
        tb_scroll.setFrameShape(QFrame.Shape.NoFrame)
        tb_layout.addWidget(tb_scroll)

        self.btn_calculate = QPushButton("⚙️ Calculate Band Structure")
        self.btn_calculate.setStyleSheet(
            "background-color: #2b5c8f; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_cancel_bands = QPushButton("🛑 Cancel Run")
        self.btn_cancel_bands.setStyleSheet(
            "background-color: #c0392b; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_cancel_bands.setEnabled(False)
        self.btn_prepare_arpes = QPushButton("🚀 Prepare TB for ARPES")
        self.btn_prepare_arpes.setStyleSheet(
            "background-color: #e67e22; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_prepare_arpes.setToolTip(
            "Fast path: parse Wannier90 with hop cutoff, build H_dict, enable "
            "Push to Workspace — skips full band-path diagonalization. "
            "Use for large / third-party hr.dat before ARPES."
        )
        self.btn_push_bands = QPushButton("📤 Push Bands to Workspace")
        self.btn_push_bands.setStyleSheet(
            "background-color: #5cb85c; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_push_bands.setEnabled(False)
        self.btn_load_qe_bands = QPushButton("📊 Load QE Ab-Initio Bands (XML)")
        self.btn_load_qe_bands.setStyleSheet(
            "background-color: #9C27B0; color: white; font-weight: bold; padding: 8px;"
        )
        for btn in (
            self.btn_calculate,
            self.btn_prepare_arpes,
            self.btn_cancel_bands,
            self.btn_push_bands,
            self.btn_load_qe_bands,
        ):
            tb_layout.addWidget(btn)
        self.left_tabs.addTab(tab_tb, "Tight Binding")

        left_layout.addWidget(self.left_tabs)

        # --- Right: band canvas ---
        self.right_container = QWidget()
        self.current_structure_name = "Unknown"
        right_layout = QVBoxLayout(self.right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_sprkkr_info = QLabel(
            "<h3>SPRKKR Band Structures</h3><br>"
            "SPRKKR computes band structures directly via the <b>Bloch Spectral Function (BSF)</b>.<br><br>"
            "1. Run your SPRKKR SCF here.<br>"
            "2. The job is automatically saved to your Workspace Vault.<br>"
            "3. Switch to the <b>ARPES Suite</b>, select the Vault, choose <b>BSF</b> task, and run it to plot!"
        )
        self.lbl_sprkkr_info.setWordWrap(True)
        self.lbl_sprkkr_info.setAlignment(Qt.AlignCenter)
        self.lbl_sprkkr_info.setStyleSheet(
            "background-color: #1e1e1e; color: #a0a0a0; font-size: 14px; padding: 20px; border-radius: 10px;"
        )
        self.lbl_sprkkr_info.hide()

        self.live_log_widget = QWidget()
        live_log_layout = QVBoxLayout(self.live_log_widget)
        live_log_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_start_live = QPushButton("📡 Start Embedded Live Monitor")
        self.btn_start_live.setStyleSheet(
            "background-color: #d35400; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_start_live.clicked.connect(self.start_embedded_monitor)
        self.txt_live_logs = QTextEdit()
        self.txt_live_logs.setReadOnly(True)
        self.txt_live_logs.setStyleSheet(
            "background-color: #0c0c0c; color: #00ff00; font-family: monospace; font-size: 12px;"
        )
        live_log_layout.addWidget(self.btn_start_live)
        live_log_layout.addWidget(self.txt_live_logs)
        self.live_log_widget.hide()

        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        right_layout.addWidget(self.canvas)
        right_layout.addWidget(self.lbl_sprkkr_info)
        right_layout.addWidget(self.live_log_widget)

        left_panel.setMinimumWidth(340)
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(self.right_container)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([400, 600])
        main_layout.addWidget(main_splitter)

        self.monitor_thread = None
        self._tb_band_thread = None
        self._tb_local_thread = None
        self._tb_prepare_thread = None
        self._tb_run_cluster = None

    def _set_tb_running(self, running: bool, label: str = "⚙️ Calculate Band Structure"):
        self.btn_calculate.setEnabled(not running)
        self.btn_prepare_arpes.setEnabled(not running)
        self.btn_cancel_bands.setEnabled(running)
        self.btn_calculate.setText(label if running else "⚙️ Calculate Band Structure")

    def cancel_tb_run(self):
        """Stop in-flight local or hybrid TB band calculation."""
        if self._tb_band_thread and self._tb_band_thread.isRunning():
            self._tb_band_thread.request_cancel()
            if self._tb_run_cluster:
                try:
                    kill_remote_tb_job(
                        self._tb_run_cluster,
                        log_fn=lambda m: print(m, flush=True),
                    )
                except Exception as exc:
                    print(f"WARN: remote kill: {exc}", flush=True)
        if self._tb_local_thread and self._tb_local_thread.isRunning():
            self._tb_local_thread.request_cancel()
            self._tb_local_thread.terminate()
        if getattr(self, "_tb_prepare_thread", None) and self._tb_prepare_thread.isRunning():
            self._tb_prepare_thread.request_cancel()
            self._tb_prepare_thread.terminate()
        self._set_tb_running(False)
        print("TB band run cancel requested.", flush=True)

    def start_embedded_monitor(self, force_start=False):
        if self.monitor_thread and self.monitor_thread.isRunning():
            if force_start:
                return # Already running, do nothing
            self.monitor_thread.running = False
            self.monitor_thread.wait()
            self.btn_start_live.setText("📡 Start Embedded Live Monitor")
            self.btn_start_live.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; padding: 8px;")
            self.monitor_thread = None
            return

        import os, json
        config_file = os.path.expanduser('~/.tensorspec_clusters.json')
        if not os.path.exists(config_file):
            QMessageBox.warning(self, "Error", "No clusters configured. Add one in Cluster Manager first.")
            return
            
        with open(config_file, 'r') as f:
            clusters = json.load(f)
            
        if not clusters:
            QMessageBox.warning(self, "Error", "No clusters found.")
            return

        target = None
        if hasattr(self, "tb_panel") and self.tb_panel.is_hybrid_mode():
            target = self.tb_panel.get_selected_cluster()
        target = target or clusters[0]

        # DFT suite: tail QE or SPR-KKR logs — not ARPES chinook_gui_run.
        if hasattr(self, "engine_dropdown") and self.engine_dropdown.currentIndex() == 1:
            log_jobs = ["sprkkr"]
        else:
            log_jobs = ["qe", "tb"]
        
        from tensorspec.gui.components.compute_panel import LiveMonitorThread
        self.monitor_thread = LiveMonitorThread(target, log_jobs=log_jobs)
        self.monitor_thread.data_ready.connect(self.update_embedded_logs)
        self.monitor_thread.error_occurred.connect(self.monitor_error)
        self.monitor_thread.start()
        
        self.live_log_widget.show()
        self.btn_start_live.setText("🛑 Stop Live Monitor")
        self.btn_start_live.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; padding: 8px;")
        
    def update_embedded_logs(self, data):
        content = "=== LIVE TASK MANAGER ===\n\n"
        content += data.get('text_info', '')
        if not content.strip().endswith("==="):
            content += "\n----------------------------------------\n"
        content += data.get('full_log_tail', '')
        
        self.txt_live_logs.setPlainText(content)
        sb = self.txt_live_logs.verticalScrollBar()
        sb.setValue(sb.maximum())
        sb = self.txt_live_logs.verticalScrollBar()
        sb.setValue(sb.maximum())
            
    def monitor_error(self, err):
        self.txt_live_logs.setPlainText(f"MONITOR ERROR:\n{err}")

    def on_engine_changed(self, idx):
        # If SPRKKR is selected (idx == 1), hide the Tight-Binding panel and plotting tools.
        if idx == 1:
            self.tb_panel.hide()
            self.btn_calculate.hide()
            self.btn_prepare_arpes.hide()
            self.btn_push_bands.hide()
            self.btn_load_qe_bands.hide()
            self.canvas.hide()
            self.lbl_sprkkr_info.show()
            self.live_log_widget.show()
        else:
            self.tb_panel.show()
            self.btn_calculate.show()
            self.btn_prepare_arpes.show()
            self.btn_push_bands.show()
            self.btn_load_qe_bands.show()
            self.lbl_sprkkr_info.hide()
            self.live_log_widget.hide()
            self.canvas.show()

    def _connect_signals(self):
        self.btn_calculate.clicked.connect(self.calculate_bands)
        self.btn_prepare_arpes.clicked.connect(self.prepare_tb_for_arpes)
        self.btn_cancel_bands.clicked.connect(self.cancel_tb_run)
        self.btn_ws_refresh.clicked.connect(self.refresh_workspace_list)
        self.btn_ws_load.clicked.connect(self.load_workspace_structure)
        self.btn_push_bands.clicked.connect(self.push_bands_to_workspace)
        
        # --- NEW: Connect Ab-Initio Plotter ---
        self.sprkkr_panel.job_started.connect(lambda: self.start_embedded_monitor(force_start=True))
        self.btn_load_qe_bands.clicked.connect(self.load_qe_xml_bands)
        
        # Connect spin box for live plotting updates using the panel reference
        self.tb_panel.spin_iso.valueChanged.connect(self.update_2d_plot)

    def refresh_workspace_list(self):
        self.ws_combo.clear()
        structures = global_workspace.list_crystal_structures()
        if structures:
            self.ws_combo.addItems(structures)
        else:
            self.ws_combo.addItem("No structures available")

    def load_workspace_structure(self):
        target = self.ws_combo.currentText()
        if not target or target == "No structures available":
            QMessageBox.warning(self, "Warning", "No valid structure selected.")
            return
            
        if self.engine.load_structure_from_workspace(target):
            self.current_structure_name = target
            formula = self.engine.crystal_structure.composition.reduced_formula
            # Slabs / 2D: default QE k_z = 1 (material-agnostic vacuum heuristic)
            try:
                if self.engine.crystal_structure.lattice.c > 10.0:
                    self.qe_panel.spin_kz.setValue(1)
            except Exception:
                pass
            hopping = self.engine.get_default_hopping(formula)
            
            self.active_hopping_keys = list(hopping.keys())
            vals = list(hopping.values())
            
            # Pre-fill the UI SpinBoxes inside the panel
            if len(vals) > 0: self.tb_panel.spin_t1.setValue(vals[0])
            if len(vals) > 1: self.tb_panel.spin_t2.setValue(vals[1])
            if len(vals) > 2: self.tb_panel.spin_t3.setValue(vals[2])
            if len(vals) > 3: self.tb_panel.spin_t4.setValue(vals[3]) 
            else: self.tb_panel.spin_t4.setValue(0.0) 
            
            # Populate Projection Dropdown
            self.tb_panel.combo_projection.clear()
            self.tb_panel.combo_projection.addItem("None (Standard Lines)")
            
            num_wann_bands = 0
            
            if self.engine.crystal_structure:
                for site in self.engine.crystal_structure:
                    el = site.specie
                    
                    # --- NEW: Automatically calculate required Wannier Bands ---
                    if el.is_transition_metal or el.number > 30:
                        num_wann_bands += 9 # s(1) + p(3) + d(5)
                    else:
                        num_wann_bands += 4 # s(1) + p(3)
                    # -----------------------------------------------------------
                    
                    elem = site.species_string
                    for orb_str in self.engine._get_orbital_basis(elem):
                        if orb_str.endswith("0"): orb_name = "s"
                        elif orb_str[1] == "1": orb_name = "p" + orb_str[2:]
                        elif orb_str[1] == "2":
                            if orb_str.endswith("ZR"): orb_name = "dz2"
                            elif orb_str.endswith("XY"): orb_name = "dx2-y2"
                            else: orb_name = "d" + orb_str[2:]
                        else: orb_name = "unknown"
                        
                        label = f"{elem}_{orb_name}"
                        existing_items = [self.tb_panel.combo_projection.itemText(i) for i in range(self.tb_panel.combo_projection.count())]
                        if label not in existing_items:
                            self.tb_panel.combo_projection.addItem(label)
            
            # --- NEW: Smart UI Popup Message ---
            msg = f"Loaded '{target}' ({formula}) into DFT engine.\n\n"
            msg += f"Based on the atomic composition, you must set:\n"
            msg += f"Number of Bands (nbnd) = {num_wann_bands}"
            
            # If your QE panel's spinbox is named spin_nbnd, this will automatically set it!
            try:
                if hasattr(self.qe_panel, 'spin_nbnd'):
                    self.qe_panel.spin_nbnd.setValue(num_wann_bands)
                    msg += "\n\n(The UI value has been updated for you automatically!)"
            except Exception:
                pass
                
            QMessageBox.information(self, "Structure Loaded", msg)
            
        else:
            QMessageBox.critical(self, "Error", "Failed to load structure.")

    def prepare_tb_for_arpes(self):
        """Parse/build Wannier TB for ARPES push — no full band-path diagonalization."""
        w90_file = self.tb_panel.active_w90_file
        if not w90_file:
            QMessageBox.warning(
                self,
                "Prepare TB for ARPES",
                "Load a wannier90_hr.dat first (yours or someone else's).",
            )
            return
        if self.engine.crystal_structure is None:
            QMessageBox.warning(
                self,
                "Prepare TB for ARPES",
                "Load a crystal structure from Workspace first.",
            )
            return

        hop_tol = self.tb_panel.hop_tol()
        is_soc = self.tb_panel.chk_soc.isChecked()
        onsite_e = float(self.tb_panel.spin_onsite.value())
        fermi_energy = float(self.tb_panel.qe_fermi_eV())
        orbital_shifts = {
            "0": float(self.tb_panel.spin_onsite_s.value()),
            "1": float(self.tb_panel.spin_onsite_p.value()),
            "2": float(self.tb_panel.spin_onsite_d.value()),
        }

        self._pending_prepare_ctx = {
            "w90_file": w90_file,
            "is_soc": is_soc,
            "onsite_e": onsite_e,
            "orbital_shifts": orbital_shifts,
            "tb_mode": self.tb_panel.combo_tb_mode.currentText(),
            "fermi_energy": fermi_energy,
            "hop_tol": hop_tol,
        }
        self._set_tb_running(True, "⏳ Preparing TB for ARPES...")
        print(
            f"Prepare TB for ARPES: {w90_file} hop_tol={hop_tol:g} "
            f"qe_fermi={fermi_energy:.4f} eV "
            f"(Wannier hr.dat — TB mode '{self.tb_panel.combo_tb_mode.currentText()}' ignored)",
            flush=True,
        )
        self._tb_prepare_thread = PrepareTBForARPESThread(
            self.engine,
            w90_file,
            {
                "use_soc": is_soc,
                "onsite_e": onsite_e,
                "hop_tol": hop_tol,
                "quick_diag": True,
                "qe_fermi": fermi_energy,
            },
            parent=self,
        )
        self._tb_prepare_thread.finished_signal.connect(self._on_prepare_arpes_done)
        self._tb_prepare_thread.start()

    def _on_prepare_arpes_done(self, success, message, result):
        self._set_tb_running(False)
        if message == "Cancelled.":
            return
        if not success:
            QMessageBox.warning(self, "Prepare TB for ARPES", message)
            return

        ctx = getattr(self, "_pending_prepare_ctx", None) or {}
        stats = result.get("stats", {})
        w90_file = ctx.get("w90_file")
        onsite_e = ctx.get("onsite_e", 0.0)
        fermi_energy = ctx.get("fermi_energy", 0.0)
        orbital_shifts = ctx.get("orbital_shifts", {})
        tb_mode = ctx.get("tb_mode", "Simple Scalar")

        basis_coords = _orbital_positions_for_workspace(
            self.engine.chinook,
            result.get("basis"),
            self.engine.crystal_structure,
        )
        recip_matrix, recip_source = _recip_matrix_for_workspace(
            self.engine.chinook, self.engine.crystal_structure
        )
        basis_quality = getattr(
            self.engine.chinook, "_last_wannier_basis_quality", None
        )

        eigenvalues = result.get("eigenvalues")
        if eigenvalues is not None:
            # W90 path folds QE EF into H during parse — do not subtract again.
            eigenvalues = np.asarray(eigenvalues, dtype=float)

        self.active_bands_data = {
            "type": "band_structure",
            "is_2d": False,
            "k_vecs": np.array([[0.0, 0.0, 0.0]]),
            "eigenvalues": eigenvalues,
            "eigenvectors": None,
            "orbital_positions": basis_coords,
            # Prefer TB_model orbital list (ARPES remote packing); keep gen_basis dict too.
            "basis": (
                getattr(result.get("tb_model"), "basis", None)
                or self.engine.chinook._orbital_list_from_basis(result.get("basis"))
                or result.get("basis")
            ),
            "H_dict": result.get("tb_dict")
            or getattr(self.engine.chinook, "H_dict", None),
            "tb_model": result.get("tb_model"),
            "fermi_energy": fermi_energy,
            "e_fermi": fermi_energy,
            "qe_fermi_eV": fermi_energy,
            "onsite_e": onsite_e,
            "orbital_shifts": orbital_shifts,
            "tb_mode": tb_mode,
            "w90_filepath": w90_file or "",
            "h_includes_onsite": True,
            "h_includes_qe_fermi_shift": True,
            "arpes_e_fermi_shift": 0.0,
            "hop_tol": stats.get("hop_tol", ctx.get("hop_tol")),
            "prepared_for_arpes": True,
            "title": "Wannier90 TB (prepared for ARPES)",
            "structure": self.engine.crystal_structure,
            "recip_matrix": recip_matrix,
            "recip_matrix_source": recip_source,
            "wannier_basis_quality": basis_quality,
            "k_dist": np.array([0.0]),
            "node_idx": [0],
            "labels": ["Γ"],
        }
        self.btn_push_bands.setEnabled(True)

        # Light Γ-point plot if we have eigenvalues
        if eigenvalues is not None:
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            n_bands = eigenvalues.shape[1] if eigenvalues.ndim == 2 else len(eigenvalues)
            e_plot = eigenvalues[0] if eigenvalues.ndim == 2 else eigenvalues
            self.ax.plot(np.zeros(n_bands), e_plot, "o", markersize=2, color="C0")
            self.ax.axhline(0.0, color="k", lw=0.8, ls="--")
            self.ax.set_ylabel("E − E_F (eV)")
            self.ax.set_title(
                f"TB ready for ARPES  |  hops={stats.get('n_hops', '?')}  "
                f"hop_tol={stats.get('hop_tol', '?')}"
            )
            self.figure.tight_layout()
            self.canvas.draw()

        QMessageBox.information(
            self,
            "Prepare TB for ARPES",
            f"{message}\n\n"
            f"Hops kept: {stats.get('n_hops', '?'):,}\n"
            f"Orbitals: {stats.get('n_orbs', '?')}\n"
            f"hop_tol: {stats.get('hop_tol', '?')}\n"
            f"Time: {stats.get('total_s', 0):.1f}s\n\n"
            f"Next: Push Bands to Workspace → ARPES Suite.",
        )

    def calculate_bands(self):
        is_2d = self.tb_panel.combo_k_mode.currentIndex() == 1
        
        try:
            # 1. Generate k-points
            if not is_2d:
                template_name = self.tb_panel.combo_k_template.currentText()
                
                if "Auto-Detect" in template_name:
                    k_points, k_labels = self.engine.get_auto_kpath()
                elif "Arbitrary" in template_name:
                    k_points, k_labels = self.engine.get_custom_kpath(self.tb_panel.line_k_coords.text(), self.tb_panel.line_k_labels.text())
                else:
                    temp_key = template_name.split()[0].lower()
                    lattice_a, lattice_b = 3.0, 3.0
                    if hasattr(self.engine.crystal_structure, 'lattice'):
                        lattice_a = self.engine.crystal_structure.lattice.a
                        lattice_b = self.engine.crystal_structure.lattice.b
                    k_points, k_labels = self.engine.get_kpath_template(temp_key, a=lattice_a, b=lattice_b)

                if hasattr(self.engine.crystal_structure, 'lattice'):
                    # --- NEW: Use A_qe reciprocal matrix if loaded from W90 to prevent rotation mismatch ---
                    if hasattr(self.engine.chinook, 'A_qe') and self.engine.chinook.A_qe is not None:
                        A_qe = self.engine.chinook.A_qe
                        # Reciprocal matrix: B = 2*pi * (A^-1)^T
                        # PyMatgen convention is without 2*pi, chinook also uses without 2*pi here?
                        # Let's match PyMatgen: 2*pi * np.linalg.inv(A_qe).T
                        recip_matrix = 2 * np.pi * np.linalg.inv(A_qe).T
                    else:
                        recip_matrix = self.engine.crystal_structure.lattice.reciprocal_lattice.matrix
                        
                    k_points = np.dot(k_points, recip_matrix)

                k_vecs, k_dist, node_idx, labels = self.engine.generate_k_path(
                    k_points, k_labels, points_per_segment=self.tb_panel.spin_k_res.value()
                )
            else:
                self.res = self.tb_panel.spin_k_res.value() 
                self.kx_vals = np.linspace(-4.5, 4.5, self.res)
                self.ky_vals = np.linspace(-4.5, 4.5, self.res)
                Kx, Ky = np.meshgrid(self.kx_vals, self.ky_vals)
                k_vecs = np.column_stack([Kx.ravel(), Ky.ravel(), np.zeros_like(Kx.ravel())])
                k_dist, node_idx, labels = None, None, None

            # 2. Re-pack UI values into the custom hopping dictionary
            custom_hopping = {}
            if hasattr(self, 'active_hopping_keys'):
                keys = self.active_hopping_keys
                if len(keys) > 0: custom_hopping[keys[0]] = self.tb_panel.spin_t1.value()
                if len(keys) > 1: custom_hopping[keys[1]] = self.tb_panel.spin_t2.value()
                if len(keys) > 2: custom_hopping[keys[2]] = self.tb_panel.spin_t3.value()
                if len(keys) > 3: custom_hopping[keys[3]] = self.tb_panel.spin_t4.value()

            # 3. Route to solver using custom physical shells or Wannier90
            is_soc = self.tb_panel.chk_soc.isChecked()
            soc_val = self.tb_panel.spin_soc_val.value()
            w90_file = self.tb_panel.active_w90_file

            ui_cutoffs = [
                self.tb_panel.spin_r1.value(),
                self.tb_panel.spin_r2.value(),
                self.tb_panel.spin_r3.value(),
                self.tb_panel.spin_r4.value()
            ]
            
            need_evecs = (
                not is_2d
                and self.tb_panel.combo_projection.currentText()
                != "None (Standard Lines)"
            )
            diag_engine, diag_device = self.tb_panel.resolved_band_diag_settings()
            ui_engine, ui_device = self.tb_panel.band_diag_settings()
            if (ui_engine, ui_device) != (diag_engine, diag_device):
                print(
                    f"TB bands: hybrid fast path overrides UI "
                    f"{ui_engine}/{ui_device} -> {diag_engine}/{diag_device}",
                    flush=True,
                )
            onsite_e = float(self.tb_panel.spin_onsite.value())
            orbital_shifts = {
                '0': float(self.tb_panel.spin_onsite_s.value()),
                '1': float(self.tb_panel.spin_onsite_p.value()),
                '2': float(self.tb_panel.spin_onsite_d.value()),
            }
            tb_mode = self.tb_panel.combo_tb_mode.currentText()
            fermi_energy = float(self.tb_panel.qe_fermi_eV())

            if w90_file:
                warn = format_w90_cost_warning(
                    quick_w90_hr_stats(w90_file), len(k_vecs)
                )
                if warn:
                    reply = QMessageBox.warning(
                        self,
                        "Large Wannier90 model",
                        warn + "\n\nContinue anyway?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No,
                    )
                    if reply != QMessageBox.Yes:
                        return

            ctx = {
                "is_2d": is_2d,
                "k_vecs": k_vecs,
                "k_dist": k_dist,
                "node_idx": node_idx,
                "labels": labels,
                "template_name": template_name if not is_2d else "2D",
                "w90_file": w90_file,
                "is_soc": is_soc,
                "soc_val": soc_val,
                "onsite_e": onsite_e,
                "orbital_shifts": orbital_shifts,
                "tb_mode": tb_mode,
                "fermi_energy": fermi_energy,
                "custom_hopping": custom_hopping,
                "ui_cutoffs": ui_cutoffs,
                "need_evecs": need_evecs,
                "diag_engine": diag_engine,
                "diag_device": diag_device,
            }

            if self.tb_panel.is_hybrid_mode():
                cluster = self.tb_panel.get_selected_cluster()
                if not cluster:
                    raise ValueError(
                        "Select a server under Compute mode (Hybrid) in the Tight Binding panel."
                    )
                w90_key = None
                if w90_file:
                    w90_key = w90_cache_key_for(
                        w90_file, is_soc, onsite_e, qe_fermi=fermi_energy
                    )
                job = build_job_payload(
                    self.engine.crystal_structure,
                    k_vecs,
                    fermi_energy=fermi_energy,
                    need_eigenvectors=need_evecs,
                    diag_engine=diag_engine,
                    diag_device=diag_device,
                    use_soc=is_soc,
                    soc_strength=soc_val,
                    onsite_e=onsite_e,
                    orbital_shifts=orbital_shifts,
                    custom_hopping=custom_hopping,
                    cutoffs=ui_cutoffs,
                    tb_mode=tb_mode,
                    w90_basename="wannier90_hr.dat" if w90_file else None,
                    w90_cache_key=w90_key,
                )
                self._pending_band_ctx = ctx
                self._tb_run_cluster = cluster
                self._set_tb_running(True, "⏳ Hybrid: server run + download...")
                print(
                    f"TB bands: hybrid submit -> {cluster_display_name(cluster)} "
                    f"({diag_engine}/{diag_device})",
                    flush=True,
                )
                self._tb_band_thread = TBBandRunnerThread(
                    cluster, job, w90_file, parent=self
                )
                self._tb_band_thread.log_signal.connect(
                    lambda m: print(m, flush=True)
                )
                self._tb_band_thread.finished_signal.connect(self._on_tb_bands_done)
                self._tb_band_thread.start()
                self.start_embedded_monitor(force_start=True)
                return

            self._pending_band_ctx = ctx
            self._tb_run_cluster = None
            self._set_tb_running(True, "⏳ Local: calculating...")
            solve_kwargs = dict(
                custom_hopping=custom_hopping,
                onsite_e=onsite_e,
                use_soc=is_soc,
                soc_strength=soc_val,
                w90_filepath=w90_file,
                cutoffs=ui_cutoffs,
                tb_mode=tb_mode,
                orbital_shifts=orbital_shifts,
                need_eigenvectors=need_evecs,
                diag_engine=diag_engine,
                diag_device=diag_device,
                qe_fermi=fermi_energy,
            )
            self._tb_local_thread = TBBandLocalRunnerThread(
                self.engine, k_vecs, solve_kwargs, fermi_energy, parent=self
            )
            self._tb_local_thread.finished_signal.connect(self._on_tb_bands_done)
            self._tb_local_thread.start()
            return
            
        except Exception as e:
            QMessageBox.warning(self, "Calculation Error", str(e))
            return

    def _fermi_energy_from_w90(self, w90_file):
        """Legacy helper — prefer tb_panel.qe_fermi_eV() in UI paths."""
        if not w90_file:
            return 0.0
        from tensorspec.core.dft.qe_fermi import detect_plot_fermi_eV

        val, _src = detect_plot_fermi_eV(os.path.dirname(w90_file))
        return float(val)

    def _attach_w90_h_dict_from_cache(self, ctx):
        """Hybrid path: H_dict for ARPES push without rebuilding gen_TB locally."""
        w90_file = ctx.get("w90_file")
        if not w90_file:
            return
        existing = getattr(self.engine.chinook, "H_dict", None)
        n_existing = 0
        if isinstance(existing, dict):
            n_existing = len(existing.get("list") or existing.get("H") or [])
        if n_existing > 0:
            return
        from tensorspec.core.dft.w90_tb_cache import load_parsed_tb

        disk = load_parsed_tb(
            w90_file,
            ctx["is_soc"],
            ctx["onsite_e"],
            qe_fermi=float(ctx.get("fermi_energy", 0.0) or 0.0),
        )
        if not disk:
            print(
                "[DFT] WARN: no local W90 TB cache after Hybrid — "
                "ARPES Push will fail until Prepare TB for ARPES or a new Hybrid "
                "run with cache sync.",
                flush=True,
            )
            return
        tb_dict, _basis_args, a_qe = disk
        n_hop = len(tb_dict.get("list") or tb_dict.get("H") or [])
        if n_hop == 0:
            print("[DFT] WARN: cached H_dict has zero hoppings", flush=True)
            return
        self.engine.chinook.H_dict = tb_dict
        if a_qe is not None:
            self.engine.chinook.A_qe = a_qe
        print(f"[DFT] Attached H_dict from W90 cache ({n_hop} hoppings)", flush=True)

    def _on_tb_bands_done(self, success, message, result):
        self._tb_run_cluster = None
        self._set_tb_running(False)
        if message == "Cancelled." or message == "Hybrid run cancelled.":
            return
        if not success:
            QMessageBox.warning(self, "TB band calculation", message)
            return

        ctx = getattr(self, "_pending_band_ctx", None)
        if not ctx:
            QMessageBox.warning(self, "TB band calculation", "Internal state lost.")
            return
        ctx = dict(ctx)

        if len(result) == 4:
            eigenvalues, eigenvectors, orb_labels, fermi_energy = result
            ctx["fermi_energy"] = fermi_energy
        else:
            eigenvalues, eigenvectors, orb_labels = result
            fermi_energy = ctx.get("fermi_energy", 0.0)

        ctx["eigenvalues"] = eigenvalues
        ctx["eigenvectors"] = eigenvectors
        ctx["orb_labels"] = orb_labels
        self._attach_w90_h_dict_from_cache(ctx)
        self._finalize_and_plot_bands(ctx)
        if self.tb_panel.is_hybrid_mode():
            QMessageBox.information(
                self,
                "Hybrid TB bands",
                f"{message}\nPlotted locally.",
            )

    def _on_tb_remote_done(self, success, message, result):
        """Back-compat wrapper."""
        self._on_tb_bands_done(success, message, result)

    @staticmethod
    def _norm_k_label(lab: str) -> str:
        s = str(lab).replace("$", "").replace("\\", "").strip()
        if "Gamma" in s or s in ("G", "Γ"):
            return "G"
        return s.upper() if len(s) <= 2 else s

    def _overlay_w90_native(
        self, w90_file, k_dist, node_idx, labels, fermi_energy
    ) -> bool:
        """Overlay wannier90_band.dat with per-segment x alignment via labelinfo.

        Global stretch of the full W90 path onto the TB path is wrong whenever
        HighSymmKpath and wannier90.win kpoint_path differ (common for 2D).
        """
        import os

        band_dat_path = w90_file.replace("_hr.dat", "_band.dat")
        labelinfo_path = w90_file.replace("_hr.dat", "_band.labelinfo.dat")
        if not os.path.isfile(band_dat_path):
            print(f"Native band file not found at: {band_dat_path}", flush=True)
            return False

        # Parse band blocks (one block per Wannier band)
        blocks = []
        xs, ys = [], []
        with open(band_dat_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]) - float(fermi_energy))
                elif not parts and xs:
                    blocks.append((np.asarray(xs), np.asarray(ys)))
                    xs, ys = [], []
        if xs:
            blocks.append((np.asarray(xs), np.asarray(ys)))
        if not blocks:
            return False

        # labelinfo: first occurrence of each label → W90 path distance
        w90_lab_x = {}
        if os.path.isfile(labelinfo_path):
            with open(labelinfo_path, "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3:
                        lab = self._norm_k_label(parts[0])
                        if lab not in w90_lab_x:
                            w90_lab_x[lab] = float(parts[2])

        blue_labs = [self._norm_k_label(l) for l in labels]
        blue_x = [float(k_dist[i]) for i in node_idx]

        # Shared consecutive label pairs for piecewise map
        pairs = []
        for a in range(len(blue_labs) - 1):
            la, lb = blue_labs[a], blue_labs[a + 1]
            if la in w90_lab_x and lb in w90_lab_x:
                pairs.append(
                    (blue_x[a], blue_x[a + 1], w90_lab_x[la], w90_lab_x[lb])
                )

        label_added = False
        for wx, wy in blocks:
            if pairs:
                mapped_x = np.full_like(wx, np.nan, dtype=float)
                for bx0, bx1, wx0, wx1 in pairs:
                    if abs(wx1 - wx0) < 1e-12:
                        continue
                    lo, hi = (wx0, wx1) if wx0 <= wx1 else (wx1, wx0)
                    mask = (wx >= lo - 1e-9) & (wx <= hi + 1e-9)
                    if not np.any(mask):
                        continue
                    t = (wx[mask] - wx0) / (wx1 - wx0)
                    mapped_x[mask] = bx0 + t * (bx1 - bx0)
                ok = np.isfinite(mapped_x)
                if not np.any(ok):
                    continue
                order = np.argsort(mapped_x[ok])
                lbl = "Wannier90 Native" if not label_added else None
                self.ax.plot(
                    mapped_x[ok][order],
                    wy[ok][order],
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    zorder=0,
                    label=lbl,
                )
                label_added = True
            else:
                # No shared labels — refuse misleading global stretch
                print(
                    "W90 overlay skipped: band.labelinfo labels do not match "
                    "TB path (would misalign Dirac/K). Blue = TensorSpec FT.",
                    flush=True,
                )
                return False
        return label_added

    def _finalize_and_plot_bands(self, ctx):
        is_2d = ctx["is_2d"]
        k_vecs = ctx["k_vecs"]
        k_dist = ctx.get("k_dist")
        node_idx = ctx.get("node_idx")
        labels = ctx.get("labels")
        w90_file = ctx.get("w90_file")
        is_soc = ctx["is_soc"]
        onsite_e = ctx["onsite_e"]
        orbital_shifts = ctx["orbital_shifts"]
        tb_mode = ctx["tb_mode"]
        fermi_energy = ctx["fermi_energy"]
        eigenvalues = ctx["eigenvalues"]
        eigenvectors = ctx.get("eigenvectors")
        orb_labels = ctx["orb_labels"]
        template_name = ctx.get("template_name", "Path")

        soc_title_tag = " (with SOC)" if is_soc and not w90_file else ""
        mode_tag = "Wannier90" if w90_file else "Chinook"
        title = (
            f"{mode_tag} 2D Mesh"
            if is_2d
            else f"{mode_tag} Bands ({template_name.split()[0]} Path){soc_title_tag}"
        )

        # Ultra-Deep Hunt for Chinook objects
        found_basis, found_h_dict, found_tb_model = None, None, None
        chinook_engine = self.engine.chinook
        found_basis = getattr(chinook_engine, "basis", None)
        found_h_dict = getattr(chinook_engine, "H_dict", None)

        for attr_name in dir(chinook_engine):
            if attr_name.startswith("__"):
                continue
            attr_val = getattr(chinook_engine, attr_name)
            type_name = type(attr_val).__name__
            if "TB_model" in type_name:
                found_tb_model = attr_val
                break

        if found_tb_model is not None:
            if found_basis is None:
                found_basis = getattr(found_tb_model, "basis", None)
            if found_h_dict is None:
                found_h_dict = getattr(found_tb_model, "H_dict", None)
            tb_basis = getattr(found_tb_model, "basis", None)
            if isinstance(tb_basis, (list, tuple)) and tb_basis:
                found_basis = tb_basis

        basis_coords = _orbital_positions_for_workspace(
            chinook_engine, found_basis, self.engine.crystal_structure
        )
        recip_matrix, recip_source = _recip_matrix_for_workspace(
            chinook_engine, self.engine.crystal_structure
        )
        basis_quality = getattr(chinook_engine, "_last_wannier_basis_quality", None)

        # 4. Cache data for pushing (onsite/EF must travel to ARPES)
        # Wannier/SK solve already folds onsite_e into H_dict diagonals.
        # Wannier also folds QE EF into those diagonals → ARPES must NOT
        # subtract fermi_energy again (would double-shift → empty window).
        is_w90 = bool(w90_file)
        self.active_bands_data = {
            'type': 'band_structure',
            'is_2d': is_2d,
            'k_vecs': k_vecs,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'orbital_positions': basis_coords, 
            'basis': found_basis,
            'H_dict': found_h_dict,
            'tb_model': found_tb_model,
            'fermi_energy': fermi_energy,
            'e_fermi': fermi_energy,
            'onsite_e': onsite_e,
            'orbital_shifts': orbital_shifts,
            'tb_mode': tb_mode,
            'w90_filepath': w90_file or '',
            'h_includes_onsite': True,
            'h_includes_qe_fermi_shift': is_w90,
            # Eigenvalue shift applied inside remote/local ARPES solve_H:
            # 0 for Wannier (already in H); QE EF for non-W90 if needed later.
            'arpes_e_fermi_shift': 0.0 if is_w90 else float(fermi_energy),
            'title': title,
            'structure': self.engine.crystal_structure,
            'recip_matrix': recip_matrix,
            'recip_matrix_source': recip_source,
            'wannier_basis_quality': basis_quality,
        }
        
        if is_2d:
            self.active_bands_data.update({'kx': self.kx_vals, 'ky': self.ky_vals, 'grid_shape': (self.res, self.res)})
        else:
            self.active_bands_data.update({'k_dist': k_dist, 'node_idx': node_idx, 'labels': labels})
        self.btn_push_bands.setEnabled(True)
        
        # --- ADD THESE DEBUG LINES ---
        print("\n[LOG 1 - DFT SUITE]")
        print(f"Fermi Energy packed into dict: {self.active_bands_data.get('fermi_energy')}")
        print(f"onsite_e packed: {self.active_bands_data.get('onsite_e')}")
        print(f"orbital_shifts packed: {self.active_bands_data.get('orbital_shifts')}")
        print(f"arpes_e_fermi_shift: {self.active_bands_data.get('arpes_e_fermi_shift')}")
        print(f"recip_matrix_source: {self.active_bands_data.get('recip_matrix_source')}")
        if basis_quality:
            print(
                f"wannier_basis: {basis_quality.get('unique_centers')} unique centres, "
                f"ok={basis_quality.get('ok')}, source={basis_quality.get('basis_source')}"
            )
        print(f"e_fermi packed into dict: {self.active_bands_data.get('e_fermi', 'NOT DEFINED')}")
        # -----------------------------

        # 5. Render Plot
        if hasattr(self, 'cbar') and self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
            
        self.figure.clear() 
        
        if not is_2d:
            self.ax = self.figure.add_subplot(111)
            num_bands = eigenvalues.shape[1]
            projection_mode = self.tb_panel.combo_projection.currentText()
            
            if projection_mode != "None (Standard Lines)":
                if eigenvectors is None:
                    QMessageBox.warning(
                        self,
                        "Fat bands",
                        "Eigenvectors were skipped (fast line mode). "
                        "Re-calculate with fat-band target selected.",
                    )
                    return
                target_el = projection_mode.replace("Element: ", "")
                target_indices = [i for i, lbl in enumerate(orb_labels) if lbl.startswith(target_el)]
                
                if target_indices:
                    probs = np.abs(eigenvectors)**2
                    if probs.shape[1] == len(orb_labels):
                        weights = np.sum(probs[:, target_indices, :], axis=1) 
                    else:
                        weights = np.sum(probs[:, :, target_indices], axis=2)
                        
                    x = np.tile(k_dist, (num_bands, 1)).T.flatten()
                    y = eigenvalues.flatten()
                    c = weights.flatten()
                    
                    scatter = self.ax.scatter(x, y, c=c, cmap='coolwarm', s=8, zorder=2, vmin=0, vmax=1)
                    self.cbar = self.figure.colorbar(scatter, ax=self.ax)
                    self.cbar.set_label(f"Orbital Character (Red = {target_el})", fontsize=10)
            else:
                for b in range(num_bands):
                    self.ax.plot(k_dist, eigenvalues[:, b], color='blue', linewidth=2)
            
            # --- NATIVE W90 OVERLAY (label-aligned; no global x-stretch) ---
            if not is_2d and w90_file and self.tb_panel.chk_overlay_w90.isChecked():
                if self._overlay_w90_native(
                    w90_file, k_dist, node_idx, labels, fermi_energy
                ):
                    self.ax.legend(loc="upper right")
            
            self.ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=1)
            self.ax.set_xlim(0, k_dist[-1])
            
            # --- Apply Y-Axis Zoom ---
            e_min = self.tb_panel.spin_emin.value()
            e_max = self.tb_panel.spin_emax.value()
            self.ax.set_ylim(e_min, e_max)
            
            self.ax.set_xticks([k_dist[i] for i in node_idx])
            self.ax.set_xticklabels(labels, fontsize=14)
            self.ax.set_ylabel("Energy (eV)", fontsize=12)
            for i in node_idx:
                self.ax.axvline(k_dist[i], color='black', linewidth=0.8, zorder=1)
                
            self.ax.set_title(title, fontsize=14)
            self.figure.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.15)
            self.canvas.draw()
            
        else:
            self.update_2d_plot()

    def update_2d_plot(self):
        if not hasattr(self, 'active_bands_data') or not self.active_bands_data.get('is_2d'):
            return
            
        if hasattr(self, 'cbar') and self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
            
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        eigenvalues = self.active_bands_data['eigenvalues']
        num_bands = eigenvalues.shape[1]
        res = self.active_bands_data['grid_shape'][0]
        
        omega = self.tb_panel.spin_iso.value()
        eta = 0.1 
        
        spectral_weight = np.zeros((res, res))
        for b in range(num_bands):
            band_energy = eigenvalues[:, b].reshape((res, res))
            spectral_weight += (eta / np.pi) / ((omega - band_energy)**2 + eta**2)
        
        im = self.ax.pcolormesh(self.active_bands_data['kx'], self.active_bands_data['ky'], spectral_weight, cmap='magma', shading='auto')
        
        self.ax.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)", fontsize=12)
        self.ax.set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)", fontsize=12)
        
        title = self.active_bands_data.get('title', "2D Mesh")
        self.ax.set_title(f"{title}\nIsoenergy Cut at {omega:.2f} eV", fontsize=14)
        self.ax.set_aspect('equal') 
        
        self.cbar = self.figure.colorbar(im, ax=self.ax)
        self.cbar.set_label("Spectral Weight (A.U.)", fontsize=10)
        
        self.figure.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
        self.canvas.draw()
            
    def _confirm_arpes_basis_quality(self, payload: dict) -> bool:
        """Warn before workspace push when Wannier basis looks too coarse for ARPES."""
        w90 = payload.get("w90_filepath")
        if not w90:
            return True

        quality = payload.get("wannier_basis_quality")
        if quality is None:
            quality = getattr(
                self.engine.chinook, "_last_wannier_basis_quality", None
            )
        if not quality or quality.get("ok"):
            return True

        msg = quality.get("message") or "Wannier basis may be wrong for ARPES."
        reply = QMessageBox.warning(
            self,
            "ARPES Basis Warning",
            f"{msg}\n\nPush to workspace anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def push_bands_to_workspace(self):
        if not hasattr(self, 'active_bands_data'):
            QMessageBox.warning(self, "Warning", "No band structure calculated yet.")
            return

        if not self._confirm_arpes_basis_quality(self.active_bands_data):
            return
            
        dim_str = "2D" if self.active_bands_data.get('is_2d') else "1D"
        struct_name = getattr(self, 'current_structure_name', 'Unknown')
        default_name = f"TB_{struct_name}_{dim_str}"
        
        name, ok = QInputDialog.getText(self, "Save to Workspace", 
                                        "Enter variable name for workspace:", 
                                        text=default_name)
        
        if not ok or not name.strip():
            return
            
        name = name.strip()
        # Re-assert energy metadata so older in-memory calcs still get keys if patched mid-session
        payload = dict(self.active_bands_data)
        payload.setdefault('onsite_e', float(self.tb_panel.spin_onsite.value()))
        payload.setdefault('orbital_shifts', {
            '0': float(self.tb_panel.spin_onsite_s.value()),
            '1': float(self.tb_panel.spin_onsite_p.value()),
            '2': float(self.tb_panel.spin_onsite_d.value()),
        })
        payload.setdefault('e_fermi', payload.get('fermi_energy', 0.0))
        global_workspace._data[name] = payload
        
        dim_str_display = "2D Mesh" if payload.get('is_2d') else "1D Path"
        recip_src = payload.get("recip_matrix_source", "cif")
        bq = payload.get("wannier_basis_quality") or {}
        basis_line = ""
        if bq:
            basis_line = (
                f"\nWannier centres: {bq.get('unique_centers', '?')} unique "
                f"(source={bq.get('basis_source', '?')})"
            )
        QMessageBox.information(
            self,
            "Success",
            f"Band structure '{name}' ({dim_str_display}) pushed to Workspace!\n\n"
            f"onsite_e = {payload.get('onsite_e')} eV\n"
            f"QE Fermi = {payload.get('fermi_energy')} eV\n"
            f"ARPES eigenvalue shift = {payload.get('arpes_e_fermi_shift', 0.0)} eV\n"
            f"Reciprocal lattice for ARPES: {recip_src}"
            f"{basis_line}\n\n"
            f"(onsite is baked into H_dict for ARPES)\n\n"
            f"Load it in the ARPES Suite.",
        )

    def load_qe_xml_bands(self):
        """Loads and overlays raw Quantum ESPRESSO bands onto the current plot."""
        fname, _ = QFileDialog.getOpenFileName(self, 'Open QE XML File', '', "XML files (*.xml);;All files (*.*)")
        if not fname:
            return

        try:
            qe_k_dist, qe_eigenvals, qe_fermi = self.engine.parse_qe_xml(fname)

            # Scale the QE x-axis to perfectly align with the PyMatgen tight-binding axis
            if hasattr(self, 'active_bands_data') and not self.active_bands_data.get('is_2d'):
                tb_k_dist = self.active_bands_data.get('k_dist')
                if tb_k_dist is not None and len(tb_k_dist) > 0 and qe_k_dist[-1] > 0:
                    qe_k_dist = qe_k_dist * (tb_k_dist[-1] / qe_k_dist[-1])

            num_bands = qe_eigenvals.shape[1]
            label_added = False
            
            # Plot the solid black DFT bands behind the current bands (zorder=0)
            for b in range(num_bands):
                lbl = "Ab-Initio (QE)" if not label_added else None
                self.ax.plot(qe_k_dist, qe_eigenvals[:, b] - qe_fermi, color='black', linewidth=1.5, zorder=0, label=lbl)
                label_added = True

            self.ax.legend(loc='upper right')
            self.canvas.draw()
            QMessageBox.information(self, "Success", f"Successfully loaded and overlaid {num_bands} Ab-Initio bands from QE.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse QE XML:\n{str(e)}")
    
    def closeEvent(self, event):
        print("close suite DFT Suite")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DFTSuite()
    window.show()
    sys.exit(app.exec())