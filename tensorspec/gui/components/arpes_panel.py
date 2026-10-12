import numpy as np
from typing import Optional
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QComboBox, QPushButton, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QMessageBox, QSlider,
                               QSpinBox, QScrollArea, QApplication, QInputDialog, QSplitter,
                               QCheckBox, QTextEdit, QLineEdit, QFileDialog)
from PySide6.QtCore import Qt, QThread, Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from tensorspec.core.arpes_engine import ARPESEngineRouter
from tensorspec.core.arpes.photon_energy_scan import (
    resolve_photon_energies,
    stack_hv_cubes,
    tensor_from_stacked_sim,
)
from tensorspec.core.workspace import global_workspace
from tensorspec.core.data_models import TensorData
from tensorspec.core.compute import cluster_paths as cp
from tensorspec.gui.services.cluster_utils import cluster_display_name
from tensorspec.core.dft.sprkkr import (
    ArpesParams as SprkkrArpesParams,
    EtaModel as SprkkrEtaModel,
    format_eta as sprkkr_format_eta,
    LocalLauncher as SprkkrLocalLauncher,
    RemoteLauncher as SprkkrRemoteLauncher,
    load_settings as sprkkr_load_settings,
    local_binaries_present as sprkkr_local_binaries_present,
    parse_spc as sprkkr_parse_spc,
    stitch_spc as sprkkr_stitch_spc,
    spc_paths_to_results as sprkkr_spc_paths_to_results,
)


def _remote_hv_cli_args(hv_list, hv_start, hv_finish, hv_step):
    """Build one remote runner's photon-energy CLI arguments."""
    if len(hv_list) == 1:
        return f"--hv {hv_list[0]}"
    return (
        f"--hv_start {hv_start} --hv_finish {hv_finish} "
        f"--hv_step {hv_step}"
    )


def _remote_cube_results(data):
    """Map a fetched Chinook npz payload to the GUI result contract."""
    cube = data["cube"]
    results = {
        "intensity_broadened": cube,
        "energy": data["energy"] if "energy" in data else None,
        "theta": data["theta"] if "theta" in data else None,
        "phi": data["phi"] if "phi" in data else None,
    }
    if "hv" in data and cube.ndim == 4:
        results["photon_energies"] = np.asarray(data["hv"], dtype=float)
    return results


class ARPESRunnerThread(QThread):
    """Runs the heavy 2D matrix element loop in the background to prevent UI freezing."""
    finished_signal = Signal(bool, object, str)
    progress = Signal(str)

    def __init__(
        self,
        engine_router,
        model_choice,
        crystal_data,
        experiment_kwargs,
        photon_energies=None,
    ):
        super().__init__()
        self.engine_router = engine_router
        self.model_choice = model_choice
        self.crystal_data = crystal_data
        self.experiment_kwargs = experiment_kwargs
        self.photon_energies = photon_energies

    def run(self):
        try:
            if self.photon_energies and len(self.photon_energies) > 1:
                cubes = []
                results = None
                for index, hv in enumerate(self.photon_energies):
                    kwargs = dict(self.experiment_kwargs)
                    kwargs["photon_energy"] = float(hv)
                    kwargs["photon_energies"] = list(self.photon_energies)
                    results = self.engine_router.run_simulation(
                        model_choice=self.model_choice,
                        crystal_data=self.crystal_data,
                        experiment_kwargs=kwargs,
                    )
                    cubes.append(results["intensity_broadened"])
                    self.progress.emit(
                        f"hv {index + 1}/{len(self.photon_energies)} ({hv:g} eV)"
                    )
                stacked, hv = stack_hv_cubes(
                    cubes, np.asarray(self.photon_energies, dtype=float)
                )
                results = dict(results)
                results["intensity_broadened"] = stacked
                results["photon_energies"] = hv
            else:
                results = self.engine_router.run_simulation(
                    model_choice=self.model_choice,
                    crystal_data=self.crystal_data,
                    experiment_kwargs=self.experiment_kwargs,
                )
            self.finished_signal.emit(True, results, "Success")
        except Exception as e:
            self.finished_signal.emit(False, None, str(e))


class ARPESPanel(QWidget):
    """
    Standalone modular panel for ARPES Matrix Element Simulations.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine_router = ARPESEngineRouter()
        self._setup_ui()

    def _setup_ui(self):
        # Use QSplitter instead of QHBoxLayout for resizable panels
        main_sim_layout = QVBoxLayout(self)
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_sim_layout.addWidget(self.main_splitter)
        
        # --- LEFT PANEL: CONTROLS ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        # 1. Engine Selector
        engine_group = QGroupBox("1. Simulation Engine Selector")
        engine_layout = QFormLayout(engine_group)
        self.engine_dropdown = QComboBox()
        self.engine_dropdown.addItem("Option B1: One-Step Model (Chinook TB)", "B1")
        self.engine_dropdown.addItem("Option B2: Bare Spectral Function (ME Off)", "B2")
        self.engine_dropdown.addItem("Option B3: Full Multiple Scattering (SPR-KKR)", "B3")
        self.engine_dropdown.addItem("Option A: Phenomenological Three-Step Model", "A")
        
        self.target_dropdown = QComboBox()
        from tensorspec.gui.services.compute_mode import populate_compute_mode_combo

        populate_compute_mode_combo(self.target_dropdown)
        
        engine_layout.addRow(QLabel("Physics Model:"), self.engine_dropdown)
        engine_layout.addRow(QLabel("Compute mode:"), self.target_dropdown)
        control_layout.addWidget(engine_group)

        # Remote Chinook/Grizzly debug toggles (shown for B1/B2/A + Remote)
        self.remote_tb_opts = QGroupBox("1b. Remote TB backend (Chinook ↔ Grizzly debug)")
        remote_tb_form = QFormLayout(self.remote_tb_opts)
        self.combo_remote_me = QComboBox()
        self.combo_remote_me.addItem("Chinook (CPU reference)", "chinook")
        self.combo_remote_me.addItem("GrizzlyME", "grizzly")
        self.combo_remote_me.addItem("Auto (Grizzly if installed)", "auto")
        self.combo_remote_me.setCurrentIndex(0)  # default chinook while debugging
        self.combo_remote_device = QComboBox()
        self.combo_remote_device.addItem("CUDA", "cuda")
        self.combo_remote_device.addItem("CPU", "cpu")
        self.combo_remote_layout = QComboBox()
        self.combo_remote_layout.addItem("θ-slices (safe; large TB)", "slices")
        self.combo_remote_layout.addItem("Full cube (θ-chunked; may OOM)", "full")
        self.combo_remote_layout.addItem("Auto (full if CUDA Grizzly)", "auto")
        self.combo_remote_layout.setCurrentIndex(0)
        self.remote_ngpus_spin = QSpinBox()
        self.remote_ngpus_spin.setRange(1, 8)
        self.remote_ngpus_spin.setValue(1)
        self.remote_ngpus_spin.setSuffix(" GPU")
        self._remote_gpu_host_max: Optional[int] = None
        self._remote_gpu_safe_max: Optional[int] = None
        gpu_row = QHBoxLayout()
        gpu_row.addWidget(self.remote_ngpus_spin)
        self.lbl_remote_gpu_limits = QLabel("host max: —")
        self.lbl_remote_gpu_limits.setStyleSheet("color: #9ad; font-size: 11px;")
        gpu_row.addWidget(self.lbl_remote_gpu_limits)
        self.btn_refresh_remote_gpus = QPushButton("Refresh")
        self.btn_refresh_remote_gpus.setFixedWidth(72)
        self.btn_refresh_remote_gpus.clicked.connect(self._refresh_remote_gpu_count)
        gpu_row.addWidget(self.btn_refresh_remote_gpus)
        gpu_row.addStretch()
        self.remote_tb_cores_spin = QSpinBox()
        self.remote_tb_cores_spin.setRange(1, 128)
        self.remote_tb_cores_spin.setValue(40)
        self.remote_tb_cores_spin.setPrefix("Cores: ")
        remote_tb_form.addRow("ME engine:", self.combo_remote_me)
        remote_tb_form.addRow("Grizzly device:", self.combo_remote_device)
        remote_tb_form.addRow("Layout:", self.combo_remote_layout)
        remote_tb_form.addRow("CUDA GPUs:", gpu_row)
        remote_tb_form.addRow("CPU workers:", self.remote_tb_cores_spin)
        self.combo_remote_me.setToolTip(
            "Chinook = reference CPU path. GrizzlyME = GPU/CPU accelerated ME. "
            "Toggle to A/B debug intensity and speed."
        )
        self.combo_remote_layout.setToolTip(
            "Full cube runs GrizzlyME on the full (θ,φ,E) grid with automatic θ-chunking "
            "when VRAM is tight. Large Wannier TB often OOMs without chunks — use θ-slices."
        )
        self.remote_ngpus_spin.setToolTip(
            "Type any count from 1 up to host max. Capped again at submit and on the cluster."
        )
        self.combo_remote_device.currentIndexChanged.connect(self._sync_remote_gpu_ui)
        self.combo_remote_layout.currentIndexChanged.connect(self._sync_remote_gpu_ui)
        self.target_dropdown.currentIndexChanged.connect(self._refresh_remote_gpu_count)
        control_layout.addWidget(self.remote_tb_opts)
        self.remote_tb_opts.hide()

        # Workspace & SOC Settings
        self.ws_group = QGroupBox("0. Crystal Structure & Bands (Workspace)")
        ws_layout = QVBoxLayout(self.ws_group)
        ws_row = QHBoxLayout()
        self.ws_combo = QComboBox()
        self.btn_ws_refresh = QPushButton("🔄 Refresh")
        self.btn_ws_refresh.clicked.connect(self.refresh_workspace)
        ws_row.addWidget(self.ws_combo)
        ws_row.addWidget(self.btn_ws_refresh)
        ws_layout.addLayout(ws_row)
        self.lbl_band_energy_meta = QLabel("onsite / EF: (select a pushed band structure)")
        self.lbl_band_energy_meta.setWordWrap(True)
        self.lbl_band_energy_meta.setStyleSheet("color: #9ad; font-size: 11px;")
        ws_layout.addWidget(self.lbl_band_energy_meta)
        self.ws_combo.currentIndexChanged.connect(self._update_band_energy_meta_label)
        control_layout.addWidget(self.ws_group)
        
        # --- Remote Vaults for SPRKKR ---
        self.vault_group = QGroupBox("0. SPRKKR Remote Vault")
        vault_layout = QHBoxLayout(self.vault_group)
        self.vault_combo = QComboBox()
        self.btn_vault_refresh = QPushButton("🔄 Refresh")
        self.btn_vault_refresh.clicked.connect(self.refresh_workspace)
        self.btn_vault_delete = QPushButton("🗑️ Delete")
        self.btn_vault_delete.clicked.connect(self.delete_vault)
        
        self.remote_cores_spin = QSpinBox()
        self.remote_cores_spin.setRange(1, 128)
        self.remote_cores_spin.setValue(40)
        self.remote_cores_spin.setPrefix("Cores: ")
        
        self.btn_load_local_spc = QPushButton("📂 Load local .spc")
        self.btn_load_local_spc.setToolTip(
            "Load an already-fetched SPR-KKR *_ARPES_data.spc from this mac "
            "(no cluster needed). Multi-select to stitch energy chunks."
        )
        self.btn_load_local_spc.clicked.connect(self.load_local_spc)

        vault_layout.addWidget(self.vault_combo)
        vault_layout.addWidget(self.remote_cores_spin)
        vault_layout.addWidget(self.btn_vault_refresh)
        vault_layout.addWidget(self.btn_vault_delete)
        vault_layout.addWidget(self.btn_load_local_spc)
        control_layout.addWidget(self.vault_group)
        self.vault_group.hide()

        # --- SPR-KKR (B3) settings ---
        _kkr_settings = sprkkr_load_settings()
        self.sprkkr_group = QGroupBox("SPR-KKR (B3) settings")
        sprkkr_form = QFormLayout(self.sprkkr_group)

        self.sprkkr_bin_edit = QLineEdit(_kkr_settings.bin_dir)
        self.sprkkr_bin_edit.setToolTip("dir with kkrspec9.7[MPI]; local runs only")
        self.lbl_kkr_bins = QLabel("")
        self.lbl_kkr_bins.setStyleSheet("color: #9ad; font-size: 11px;")
        self.sprkkr_bin_edit.textChanged.connect(self._update_kkr_bin_status)
        sprkkr_form.addRow("Bin dir:", self.sprkkr_bin_edit)
        sprkkr_form.addRow("", self.lbl_kkr_bins)

        pot_row = QHBoxLayout()
        self.sprkkr_pot_edit = QLineEdit()
        self.sprkkr_pot_edit.setToolTip("local converged .pot_new (used when compute target is local)")
        self.btn_sprkkr_pot_browse = QPushButton("Browse")
        self.btn_sprkkr_pot_browse.clicked.connect(self._browse_sprkkr_pot)
        pot_row.addWidget(self.sprkkr_pot_edit)
        pot_row.addWidget(self.btn_sprkkr_pot_browse)
        sprkkr_form.addRow("Local pot:", pot_row)

        self.spin_kkr_nproc = QSpinBox()
        self.spin_kkr_nproc.setRange(1, 256)
        self.spin_kkr_nproc.setValue(_kkr_settings.nproc)
        sprkkr_form.addRow("nproc:", self.spin_kkr_nproc)

        self.combo_kkr_mode = QComboBox()
        self.combo_kkr_mode.addItems(["auto", "mpi", "chunks"])
        sprkkr_form.addRow("Fan-out mode:", self.combo_kkr_mode)

        self.chk_pointwise = QCheckBox("Point-wise deflector cut (1 job per slit angle)")
        self.chk_pointwise.setChecked(True)
        self.chk_pointwise.setToolTip(
            "Off = legacy: deflector goes straight to SPEC_EL PHI -> a cut THROUGH Gamma "
            "(only correct at deflector 0). On = one kkrspec job per slit angle, each with "
            "its own (THETA, PHI) -> a cut PARALLEL to the slit, offset from Gamma. "
            "Costs NT jobs."
        )
        sprkkr_form.addRow(self.chk_pointwise)

        self.spin_phi_offset = QDoubleSpinBox(); self.spin_phi_offset.setRange(-180.0, 180.0)
        self.spin_phi_offset.setValue(0.0); self.spin_phi_offset.setSuffix(" °")
        self.spin_phi_offset.setToolTip(
            "UNCALIBRATED. Angle between the analyzer slit (at azimuth 0) and SPR-KKR's own "
            "in-plane x axis, which ase2sprkkr picks when it slices by MILLER_HKL. The manual "
            "does not pin it. 0 assumes slit || x (held empirically for the VTe2 PHI=0 run)."
        )
        sprkkr_form.addRow("PHI zero offset (UNCALIBRATED):", self.spin_phi_offset)

        self.spin_ref_energy = QDoubleSpinBox(); self.spin_ref_energy.setRange(-20.0, 5.0)
        self.spin_ref_energy.setValue(0.0); self.spin_ref_energy.setSuffix(" eV")
        self.spin_ref_energy.setToolTip(
            "Energy (rel. E_F) at which (THETA_i, PHI_i) are evaluated; held fixed over the "
            "whole window. Residual drift ~0.13 deg/eV, ~0.7 %/eV in k_par."
        )
        sprkkr_form.addRow("Angle reference energy:", self.spin_ref_energy)

        self.spin_n_layer = QSpinBox()
        self.spin_n_layer.setRange(5, 200)
        self.spin_n_layer.setValue(50)
        self.spin_n_layer.setToolTip("slab layers; cost ∝ N_LAYER")
        sprkkr_form.addRow("N_LAYER:", self.spin_n_layer)

        self.spin_nlat_g_vec = QSpinBox()
        self.spin_nlat_g_vec.setRange(9, 300)
        self.spin_nlat_g_vec.setValue(57)
        self.spin_nlat_g_vec.setToolTip("2D reciprocal vectors; cost ∝ NLAT_G_VEC")
        sprkkr_form.addRow("NLAT_G_VEC:", self.spin_nlat_g_vec)

        self.spin_imv_ini = QDoubleSpinBox()
        self.spin_imv_ini.setRange(0.0, 5.0)
        self.spin_imv_ini.setDecimals(3)
        self.spin_imv_ini.setValue(0.05)
        self.spin_imv_ini.setSuffix(" eV")
        sprkkr_form.addRow("IMV_INI:", self.spin_imv_ini)

        self.spin_imv_fin = QDoubleSpinBox()
        self.spin_imv_fin.setRange(0.0, 10.0)
        self.spin_imv_fin.setDecimals(3)
        self.spin_imv_fin.setValue(2.0)
        self.spin_imv_fin.setSuffix(" eV")
        sprkkr_form.addRow("IMV_FIN:", self.spin_imv_fin)

        self.spin_kkr_nl = QSpinBox()
        self.spin_kkr_nl.setRange(2, 5)
        self.spin_kkr_nl.setValue(3)
        sprkkr_form.addRow("NL:", self.spin_kkr_nl)

        self.combo_hkl_frame = QComboBox()
        self.combo_hkl_frame.addItem("Auto (CIF if lattice known)", "auto")
        self.combo_hkl_frame.addItem("CIF (convert → ABAS)", "cif")
        self.combo_hkl_frame.addItem("ABAS (direct / CRYS_VECS)", "abas")
        self.combo_hkl_frame.setToolTip(
            "How Cleavage Plane [h k l] is read for SPR-KKR.\n"
            "• CIF: Miller index in the structure/CIF cell; converted to pot ABAS "
            "(needs matching cif_lattice on the vault).\n"
            "• ABAS: indices already in the .pot A(1..3) frame; emit CRYS_VECS, "
            "no conversion (use for VTe₂ e.g. −2 0 1 when CIF convert fails).\n"
            "• Auto: CIF when vault/crystal lattice exists, else ABAS."
        )
        sprkkr_form.addRow("HKL frame:", self.combo_hkl_frame)

        self.chk_iq_auto = QCheckBox("Auto surface site (top atom)")
        self.chk_iq_auto.setChecked(True)
        self.chk_iq_auto.setToolTip(
            "Surface IQ auto-picked (topmost non-vacancy) along the chosen HKL "
            "normal unless you pin one below. HKL frame is set by the combo above."
        )
        self.spin_iq_surf = QSpinBox()
        self.spin_iq_surf.setRange(1, 999)
        self.spin_iq_surf.setEnabled(False)
        self.chk_iq_auto.toggled.connect(lambda on: self.spin_iq_surf.setEnabled(not on))
        sprkkr_form.addRow("", self.chk_iq_auto)
        sprkkr_form.addRow("IQ_AT_SURF:", self.spin_iq_surf)

        self.lbl_kkr_geom = QLabel("geometry: —")
        self.lbl_kkr_geom.setStyleSheet("color: #9ad; font-size: 11px;")
        self.lbl_kkr_geom.setWordWrap(True)
        sprkkr_form.addRow(self.lbl_kkr_geom)

        self.lbl_kkr_eta = QLabel("ETA: —")
        self.lbl_kkr_eta.setStyleSheet("color: #9ad; font-size: 11px;")
        sprkkr_form.addRow(self.lbl_kkr_eta)

        self.btn_kkr_kill = QPushButton("⛔ Kill SPR-KKR jobs")
        self.btn_kkr_kill.clicked.connect(self._kill_sprkkr_jobs)
        sprkkr_form.addRow(self.btn_kkr_kill)

        control_layout.addWidget(self.sprkkr_group)
        self.sprkkr_group.hide()
        self._update_kkr_bin_status()

        self.engine_dropdown.currentIndexChanged.connect(self.on_engine_changed)
        self.target_dropdown.currentIndexChanged.connect(self.on_target_changed)
        
        # 2. Final State & Thermodynamics
        param_group = QGroupBox("2. Final State & Thermodynamics")
        param_layout = QFormLayout(param_group)
        self.param_layout = param_layout  # kept for label lookup (_set_row_visible)
        
        self.hv_mode_combo = QComboBox()
        self.hv_mode_combo.addItem("Single", "single")
        self.hv_mode_combo.addItem("Range", "range")
        self.hv_mode_combo.setToolTip(
            "hv-dependent scan = Chinook (Grizzly) only. SPR-KKR later."
        )

        self.photon_energy_spin = QDoubleSpinBox(); self.photon_energy_spin.setRange(5.0, 2000.0); self.photon_energy_spin.setValue(90.0); self.photon_energy_spin.setSuffix(" eV")

        self.hv_start_spin = QDoubleSpinBox()
        self.hv_start_spin.setRange(5.0, 2000.0)
        self.hv_start_spin.setValue(80.0)
        self.hv_start_spin.setSuffix(" eV")
        self.hv_finish_spin = QDoubleSpinBox()
        self.hv_finish_spin.setRange(5.0, 2000.0)
        self.hv_finish_spin.setValue(100.0)
        self.hv_finish_spin.setSuffix(" eV")
        self.hv_step_spin = QDoubleSpinBox()
        self.hv_step_spin.setRange(0.01, 2000.0)
        self.hv_step_spin.setValue(5.0)
        self.hv_step_spin.setSuffix(" eV")

        self.hv_range_widget = QWidget()
        hv_range_layout = QHBoxLayout(self.hv_range_widget)
        hv_range_layout.setContentsMargins(0, 0, 0, 0)
        hv_range_layout.addWidget(QLabel("Start"))
        hv_range_layout.addWidget(self.hv_start_spin)
        hv_range_layout.addWidget(QLabel("Finish"))
        hv_range_layout.addWidget(self.hv_finish_spin)
        hv_range_layout.addWidget(QLabel("Step"))
        hv_range_layout.addWidget(self.hv_step_spin)
        self.hv_range_widget.hide()

        hv_row_widget = QWidget()
        hv_row_layout = QHBoxLayout(hv_row_widget)
        hv_row_layout.setContentsMargins(0, 0, 0, 0)
        hv_row_layout.addWidget(self.hv_mode_combo)
        hv_row_layout.addWidget(self.photon_energy_spin)
        hv_row_layout.addWidget(self.hv_range_widget)
        self.hv_mode_combo.currentIndexChanged.connect(self.update_hv_mode_ui)

        self.work_function_spin = QDoubleSpinBox(); self.work_function_spin.setRange(0.0, 10.0); self.work_function_spin.setValue(4.5); self.work_function_spin.setSuffix(" eV")
        self.inner_potential_spin = QDoubleSpinBox(); self.inner_potential_spin.setRange(0.0, 30.0); self.inner_potential_spin.setValue(15.0); self.inner_potential_spin.setSuffix(" eV")
        self.temperature_spin = QDoubleSpinBox(); self.temperature_spin.setRange(0.1, 1000.0); self.temperature_spin.setValue(10.0); self.temperature_spin.setSuffix(" K")

        self.rad_type_combo = QComboBox()
        self.rad_type_combo.addItems(["slater", "hydrogenic", "grid"])
        # grid needs rad_args — disable until supported
        _rad_model = self.rad_type_combo.model()
        _grid_item = _rad_model.item(2)
        if _grid_item is not None:
            _grid_item.setEnabled(False)
        self.rad_type_combo.setItemData(
            2, "needs rad_args (unsupported)", Qt.ToolTipRole
        )
        self.rad_type_combo.setToolTip(
            "Final-state radial type. 'grid' needs rad_args (unsupported)."
        )

        self.mfp_spin = QDoubleSpinBox()
        self.mfp_spin.setRange(0.5, 50.0)
        self.mfp_spin.setValue(10.0)
        self.mfp_spin.setSingleStep(0.5)
        self.mfp_spin.setDecimals(2)
        self.mfp_spin.setSuffix(" Å")
        
        param_layout.addRow("Photon (hv):", hv_row_widget)
        param_layout.addRow("Work Func (Φ):", self.work_function_spin)
        param_layout.addRow("Inner Pot (V0):", self.inner_potential_spin)
        param_layout.addRow("Radial Type:", self.rad_type_combo)
        param_layout.addRow("MFP (λ):", self.mfp_spin)
        param_layout.addRow("Temperature:", self.temperature_spin)
        control_layout.addWidget(param_group)

        # 3. Beam & Manipulator Geometry
        beam_group = QGroupBox("3. Beam & Manipulator Geometry")
        beam_layout = QFormLayout(beam_group)
        self.beam_layout = beam_layout  # kept for label lookup (_set_row_visible)
        
        self.manip_theta_spin = QDoubleSpinBox(); self.manip_theta_spin.setRange(-180.0, 180.0); self.manip_theta_spin.setSuffix(" °")
        self.manip_azi_spin = QDoubleSpinBox(); self.manip_azi_spin.setRange(-180.0, 180.0); self.manip_azi_spin.setSuffix(" °")
        self.manip_azi_spin.setToolTip("Sample azimuth (physical manipulator rotation about the surface normal). For SPR-KKR it adds to the PHI zero offset; calibrate that offset at azimuth 0.")
        self.manip_tilt_spin = QDoubleSpinBox(); self.manip_tilt_spin.setRange(-90.0, 90.0); self.manip_tilt_spin.setSuffix(" °")
        self.incidence_angle_spin = QDoubleSpinBox(); self.incidence_angle_spin.setRange(0.0, 90.0); self.incidence_angle_spin.setValue(55.0); self.incidence_angle_spin.setSuffix(" °")

        self.kz_halfwidth_spin = QDoubleSpinBox()
        self.kz_halfwidth_spin.setRange(0.0, 2.0)
        self.kz_halfwidth_spin.setValue(0.0)
        self.kz_halfwidth_spin.setSingleStep(0.05)
        self.kz_halfwidth_spin.setDecimals(3)
        self.kz_halfwidth_spin.setSuffix(" Å⁻¹")
        self.kz_halfwidth_spin.setToolTip(
            "Lorentzian k_z HWHM. 0 = off (legacy single cut)."
        )

        self.kz_npoints_spin = QSpinBox()
        self.kz_npoints_spin.setRange(3, 51)
        self.kz_npoints_spin.setSingleStep(2)
        self.kz_npoints_spin.setValue(7)
        self.kz_npoints_spin.setEnabled(False)
        self.kz_npoints_spin.setToolTip(
            "Odd sample count for k_z Lorentzian sum. Enabled when halfwidth > 0."
        )
        self.kz_halfwidth_spin.valueChanged.connect(self._on_kz_halfwidth_changed)
        self.kz_npoints_spin.valueChanged.connect(self._ensure_kz_npoints_odd)

        self.fresnel_enabled_chk = QCheckBox("Fresnel local field")
        self.fresnel_enabled_chk.setChecked(True)
        self.fresnel_enabled_chk.setToolTip(
            "Apply Fresnel transmission to lab-frame vector potential A."
        )
        self.optical_n_spin = QDoubleSpinBox()
        self.optical_n_spin.setRange(0.01, 20.0)
        self.optical_n_spin.setValue(1.0)
        self.optical_n_spin.setSingleStep(0.05)
        self.optical_n_spin.setDecimals(3)
        self.optical_n_spin.setToolTip("Real refractive index n (n=1 → vacuum parity).")
        self.optical_k_spin = QDoubleSpinBox()
        self.optical_k_spin.setRange(0.0, 20.0)
        self.optical_k_spin.setValue(0.0)
        self.optical_k_spin.setSingleStep(0.05)
        self.optical_k_spin.setDecimals(3)
        self.optical_k_spin.setToolTip("Extinction coefficient k (Im n).")
        self.include_photon_momentum_chk = QCheckBox("Photon momentum (soft X-ray)")
        self.include_photon_momentum_chk.setChecked(False)
        self.include_photon_momentum_chk.setToolTip(
            "Subtract full photon wavevector q from bulk crystal momentum."
        )
        self.fresnel_enabled_chk.toggled.connect(self._on_fresnel_enabled_changed)
        self._on_fresnel_enabled_changed(True)

        self.polarization_combo = QComboBox()
        self.polarization_combo.addItems([
            "Linear Horizontal (p-pol)", 
            "Linear Vertical (s-pol)", 
            "Linear Arbitrary", 
            "Circular Right (CR)", 
            "Circular Left (CL)"
        ])
        
        self.lin_pol_angle_spin = QDoubleSpinBox()
        self.lin_pol_angle_spin.setRange(0.0, 360.0)
        self.lin_pol_angle_spin.setValue(45.0)
        self.lin_pol_angle_spin.setSuffix(" °")
        self.lin_pol_angle_spin.setVisible(False)
        
        self.polarization_combo.currentTextChanged.connect(
            lambda text: self.lin_pol_angle_spin.setVisible("Arbitrary" in text)
        )
        
        self.matrix_element_combo = QComboBox()
        self.matrix_element_combo.addItems(["Full Matrix Elements", "Polarization Dipole Only", "Bare Spectral Function (ME Off)"])
        

        beam_layout.addRow("Manipulator Θ (Lab Z):", self.manip_theta_spin)
        beam_layout.addRow("Manipulator Azimuth:", self.manip_azi_spin)
        beam_layout.addRow("Manipulator Tilt:", self.manip_tilt_spin)

        # --- ADD THIS INSIDE _setup_ui() (e.g., right before beam_layout.addRow("Manipulator Θ...")) ---
        hkl_layout = QHBoxLayout()
        self.spin_h = QSpinBox(); self.spin_h.setRange(-10, 10); self.spin_h.setValue(0)
        self.spin_k = QSpinBox(); self.spin_k.setRange(-10, 10); self.spin_k.setValue(0)
        self.spin_l = QSpinBox(); self.spin_l.setRange(-10, 10); self.spin_l.setValue(1)
        
        hkl_layout.addWidget(self.spin_h)
        hkl_layout.addWidget(self.spin_k)
        hkl_layout.addWidget(self.spin_l)
        for _sp in (self.spin_h, self.spin_k, self.spin_l):
            _sp.setToolTip(
                "Cleavage Miller index. For SPR-KKR (B3), interpretation is set by "
                "'HKL frame' under SPR-KKR settings (CIF convert vs ABAS direct)."
            )

        beam_layout.addRow("Cleavage Plane [h k l]:", hkl_layout)
        # ---------------------------------------------------------------------------------------------
        beam_layout.addRow("Beam Incidence (Lab):", self.incidence_angle_spin)
        beam_layout.addRow(self.fresnel_enabled_chk)
        beam_layout.addRow("Optical n:", self.optical_n_spin)
        beam_layout.addRow("Optical k:", self.optical_k_spin)
        beam_layout.addRow(self.include_photon_momentum_chk)
        beam_layout.addRow("k_z Halfwidth:", self.kz_halfwidth_spin)
        beam_layout.addRow("k_z N Points:", self.kz_npoints_spin)
        beam_layout.addRow("Polarization:", self.polarization_combo)
        beam_layout.addRow("Pol. Angle (Arbitrary):", self.lin_pol_angle_spin)
        beam_layout.addRow("Intensity Mode:", self.matrix_element_combo)
        control_layout.addWidget(beam_group)

        # --- SARPES CONFIGURATION ---
        self.group_sarpes = QGroupBox("Spin-Resolved ARPES (SARPES)")
        sarpes_layout = QFormLayout()
        
        self.chk_enable_sarpes = QCheckBox("Enable SARPES Filter")
        self.chk_enable_sarpes.setChecked(False)
        
        self.combo_spin_axis = QComboBox()
        self.combo_spin_axis.addItems(["Sz (Z-Axis)", "Sx (X-Axis)", "Sy (Y-Axis)"])
        
        self.combo_spin_comp = QComboBox()
        self.combo_spin_comp.addItems(["Spin Up (+)", "Spin Down (-)"])
        
        sarpes_layout.addRow(self.chk_enable_sarpes)
        sarpes_layout.addRow("Projection Axis:", self.combo_spin_axis)
        sarpes_layout.addRow("Spin Filter:", self.combo_spin_comp)
        self.group_sarpes.setLayout(sarpes_layout)
        
        control_layout.addWidget(self.group_sarpes)


        
        
        # 4. Analyzer Domain & Resolution
        domain_group = QGroupBox("4. Analyzer Domain & Resolution")
        domain_layout = QVBoxLayout(domain_group)
        
        analyzer_layout = QFormLayout()
        self.slit_size_spin = QDoubleSpinBox(); self.slit_size_spin.setRange(0.1, 5.0); self.slit_size_spin.setValue(0.5); self.slit_size_spin.setSingleStep(0.1); self.slit_size_spin.setSuffix(" mm")
        self.slit_angle_spin = QDoubleSpinBox(); self.slit_angle_spin.setRange(-180.0, 180.0); self.slit_angle_spin.setValue(0.0); self.slit_angle_spin.setSuffix(" °")
        self.deflector_angle_spin = QDoubleSpinBox(); self.deflector_angle_spin.setRange(-15.0, 15.0); self.deflector_angle_spin.setValue(0.0); self.deflector_angle_spin.setSuffix(" °")
        analyzer_layout.addRow("Analyzer Slit Size:", self.slit_size_spin)
        analyzer_layout.addRow("Analyzer Slit Angle:", self.slit_angle_spin)
        analyzer_layout.addRow("Deflector Angle:", self.deflector_angle_spin)
        domain_layout.addLayout(analyzer_layout)
        
        def create_range_row(label_text, min_val, max_val, pts_val):
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            vmin = QDoubleSpinBox(); vmin.setRange(-90.0, 90.0); vmin.setValue(min_val); vmin.setSingleStep(0.1)
            vmax = QDoubleSpinBox(); vmax.setRange(-90.0, 90.0); vmax.setValue(max_val); vmax.setSingleStep(0.1)
            vpts = QSpinBox(); vpts.setRange(10, 1000); vpts.setValue(40); vpts.setSingleStep(10)
            row.addWidget(QLabel("Min:")); row.addWidget(vmin)
            row.addWidget(QLabel("Max:")); row.addWidget(vmax)
            row.addWidget(QLabel("Pts:")); row.addWidget(vpts)
            return row, vmin, vmax, vpts

        # Relabeled strictly to Angular axes (Theta and Phi)
        row_kx, self.spin_kx_min, self.spin_kx_max, self.spin_kx_steps = create_range_row("Θ (Slit) [°]:", -15.0, 15.0, 100)
        row_ky, self.spin_ky_min, self.spin_ky_max, self.spin_ky_steps = create_range_row("Φ (Deflect) [°]:", -15.0, 15.0, 100)
        row_e, self.spin_e_min, self.spin_e_max, self.spin_e_steps = create_range_row("E:", -2.0, 0.5, 100)

        domain_layout.addLayout(row_kx)
        domain_layout.addLayout(row_ky)
        domain_layout.addLayout(row_e)

        # SPR-KKR (B3) ETA depends on these grid sizes + its own nproc spin.
        self.spin_e_steps.valueChanged.connect(self._update_kkr_eta)
        self.spin_kx_steps.valueChanged.connect(self._update_kkr_eta)
        self.spin_ky_steps.valueChanged.connect(self._update_kkr_eta)
        self.spin_kkr_nproc.valueChanged.connect(self._update_kkr_eta)
        self.chk_pointwise.toggled.connect(self._update_kkr_eta)
        self.deflector_angle_spin.valueChanged.connect(self._update_kkr_eta)
        self._update_kkr_eta()

        res_layout = QFormLayout()
        self.ui_se_spinbox = QDoubleSpinBox(); self.ui_se_spinbox.setRange(0.001, 1.0); self.ui_se_spinbox.setValue(0.01); self.ui_se_spinbox.setSingleStep(0.01); self.ui_se_spinbox.setDecimals(3); self.ui_se_spinbox.setSuffix(" eV")
        self.ui_res_e_spinbox = QDoubleSpinBox(); self.ui_res_e_spinbox.setRange(0.001, 1.0); self.ui_res_e_spinbox.setValue(0.02); self.ui_res_e_spinbox.setSingleStep(0.01); self.ui_res_e_spinbox.setDecimals(3); self.ui_res_e_spinbox.setSuffix(" eV")
        self.ui_res_k_spinbox = QDoubleSpinBox(); self.ui_res_k_spinbox.setRange(0.001, 1.0); self.ui_res_k_spinbox.setValue(0.02); self.ui_res_k_spinbox.setSingleStep(0.01); self.ui_res_k_spinbox.setDecimals(3); self.ui_res_k_spinbox.setSuffix(" 1/A")
        
        res_layout.addRow("Peak Linewidth (SE):", self.ui_se_spinbox)
        res_layout.addRow("Energy Resolution (dE):", self.ui_res_e_spinbox)
        res_layout.addRow("Momentum Resolution (dk):", self.ui_res_k_spinbox)
        domain_layout.addLayout(res_layout)
        
        control_layout.addWidget(domain_group)
        
        control_layout.addStretch()
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(control_panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(350) 
        self.main_splitter.addWidget(scroll_area)

        # --- MIDDLE PANEL: SCHEMATIC VIEWER ---
        schematic_panel = QWidget()
        schematic_layout = QVBoxLayout(schematic_panel)
        self.schematic_figure = Figure(figsize=(4, 5))
        self.schematic_canvas = FigureCanvas(self.schematic_figure)
        
        self.ax_schematic = self.schematic_figure.add_subplot(111, projection='3d')
        schematic_layout.addWidget(self.schematic_canvas)
        self.main_splitter.addWidget(schematic_panel)

        self.incidence_angle_spin.valueChanged.connect(self.update_schematic)
        self.polarization_combo.currentTextChanged.connect(self.update_schematic)
        self.deflector_angle_spin.valueChanged.connect(self.update_schematic)
        self.slit_size_spin.valueChanged.connect(self.update_schematic)
        self.manip_theta_spin.valueChanged.connect(self.update_schematic)
        self.manip_azi_spin.valueChanged.connect(self.update_schematic)
        self.manip_tilt_spin.valueChanged.connect(self.update_schematic)
        self.lin_pol_angle_spin.valueChanged.connect(self.update_schematic)
        self.slit_angle_spin.valueChanged.connect(self.update_schematic)

        # --- RIGHT PANEL: INTERACTIVE PLOTTER ---
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Simulated ARPES Intensity")
        plot_layout.addWidget(self.canvas, stretch=1)
        
        slider_layout = QHBoxLayout()
        self.energy_label = QLabel("Binding Energy (eV):")
        
        # New QDoubleSpinBox for keyboard input
        self.energy_spinbox = QDoubleSpinBox()
        self.energy_spinbox.setDecimals(3)
        self.energy_spinbox.setSingleStep(0.01)
        self.energy_spinbox.setEnabled(False)
        self.energy_spinbox.setKeyboardTracking(False) # Only update when Enter is pressed or focus is lost
        
        self.energy_slider = QSlider(Qt.Horizontal)
        self.energy_slider.setEnabled(False)
        
        # Connect both widgets to sync and update the plot
        self.energy_slider.valueChanged.connect(self._sync_slider_to_spinbox)
        self.energy_spinbox.valueChanged.connect(self._sync_spinbox_to_slider)
        
        slider_layout.addWidget(self.energy_label)
        slider_layout.addWidget(self.energy_spinbox)
        slider_layout.addWidget(self.energy_slider)
        plot_layout.addLayout(slider_layout)
        
        contrast_layout = QHBoxLayout()
        self.vmin_spin = QDoubleSpinBox(); self.vmin_spin.setRange(0.0, 1.0); self.vmin_spin.setValue(0.0); self.vmin_spin.setSingleStep(0.05)
        self.vmax_spin = QDoubleSpinBox(); self.vmax_spin.setRange(0.0, 1.0); self.vmax_spin.setValue(1.0); self.vmax_spin.setSingleStep(0.05)
        self.gamma_spin = QDoubleSpinBox(); self.gamma_spin.setRange(0.1, 5.0); self.gamma_spin.setValue(1.0); self.gamma_spin.setSingleStep(0.1)
        
        # --- NEW: BZ OVERLAY TOGGLE ---
        self.chk_overlay_bz = QCheckBox("Overlay Surface BZ")
        self.chk_overlay_bz.setStyleSheet("font-weight: bold; color: #0F6A8B;")
        self.chk_overlay_bz.stateChanged.connect(lambda: self.update_plot_slice())
        # ------------------------------

        contrast_layout.addWidget(QLabel("Min:"))
        contrast_layout.addWidget(self.vmin_spin)
        contrast_layout.addWidget(QLabel("Max:"))
        contrast_layout.addWidget(self.vmax_spin)
        contrast_layout.addWidget(QLabel("γ (Gamma):"))
        contrast_layout.addWidget(self.gamma_spin)
        contrast_layout.addWidget(self.chk_overlay_bz) # <-- Added here
        plot_layout.addLayout(contrast_layout)

        
        
        self.vmin_spin.valueChanged.connect(lambda: self.update_plot_slice(self.energy_slider.value()))
        self.vmax_spin.valueChanged.connect(lambda: self.update_plot_slice(self.energy_slider.value()))
        self.gamma_spin.valueChanged.connect(lambda: self.update_plot_slice(self.energy_slider.value()))
        
        # --- Live Logs (Hidden by default, shown for SPRKKR) ---
        self.live_log_widget = QWidget()
        live_log_layout = QVBoxLayout(self.live_log_widget)
        live_log_layout.setContentsMargins(0, 5, 0, 0)
        
        self.btn_start_live = QPushButton("▶️ Start Live Monitor")
        self.btn_start_live.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; padding: 4px;")
        self.btn_start_live.clicked.connect(self.toggle_embedded_monitor)
        
        self.btn_fetch_results = QPushButton("📥 Fetch Remote Results")
        self.btn_fetch_results.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 4px;")
        self.btn_fetch_results.clicked.connect(self.fetch_remote_results)
        self.btn_fetch_results.setEnabled(False)
        self.btn_fetch_results.setToolTip(
            "Pull finished cube from the selected remote cluster. Enabled whenever "
            "Compute Target is Remote (Chinook → chinook_gui_run; SPR-KKR → vault / sprkkr_gui_run)."
        )
        
        self.txt_live_logs = QTextEdit()
        self.txt_live_logs.setReadOnly(True)
        self.txt_live_logs.setFixedHeight(120)
        self.txt_live_logs.setStyleSheet("background-color: #0c0c0c; color: #00ff00; font-family: monospace; font-size: 10px;")
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_start_live)
        btn_layout.addWidget(self.btn_fetch_results)
        
        live_log_layout.addLayout(btn_layout)
        live_log_layout.addWidget(self.txt_live_logs)
        plot_layout.addWidget(self.live_log_widget)
        self.live_log_widget.hide()
        
        self.live_monitor = None

        # After remote widgets exist: sync Fetch/Monitor vs Physics×Target.
        self._sync_remote_ui()
        
        self.main_splitter.addWidget(plot_panel)
        
        # Set initial relative sizes for the splitter panels (e.g., 30% left, 35% middle, 35% right)
        self.main_splitter.setSizes([350, 400, 400])
        
        self.update_schematic()

        # Run Button
        self.run_sim_btn = QPushButton("🚀 Run ARPES Simulation")
        self.run_sim_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #2b5c8f; color: white;")
        self.run_sim_btn.clicked.connect(self.trigger_simulation)
        control_layout.addWidget(self.run_sim_btn)
        
        # --- NEW: Save & Push Buttons ---
        self.btn_push_workspace = QPushButton("📊 Push to Workspace")
        self.btn_push_workspace.setStyleSheet("font-weight: bold; padding: 10px; background-color: #E67E22; color: white;")
        self.btn_push_workspace.setEnabled(False)
        self.btn_push_workspace.clicked.connect(self.push_arpes_to_workspace)
        
        self.btn_save_disk = QPushButton("💾 Save to Disk (.npz)")
        self.btn_save_disk.setStyleSheet("font-weight: bold; padding: 10px; background-color: #D35400; color: white;")
        self.btn_save_disk.setEnabled(False)
        self.btn_save_disk.clicked.connect(self.save_arpes_to_disk)
        
        control_layout.addWidget(self.btn_push_workspace)
        control_layout.addWidget(self.btn_save_disk)

    def _hv_scan_allowed(self) -> bool:
        # B1 Chinook TB; B2 bare still uses chinook_wrapper path in panel — allow only
        # engines that call Chinook/Grizzly ME (B1). Exclude A (three-step) and B3 (SPR-KKR).
        return self.engine_dropdown.currentData() == "B1"

    def update_hv_mode_ui(self, *_args):
        scan_allowed = self._hv_scan_allowed()
        range_item = self.hv_mode_combo.model().item(1)
        if range_item is not None:
            range_item.setEnabled(scan_allowed)
        if not scan_allowed and self.hv_mode_combo.currentData() == "range":
            self.hv_mode_combo.setCurrentIndex(0)

        is_range = (
            scan_allowed and self.hv_mode_combo.currentData() == "range"
        )
        self.photon_energy_spin.setVisible(not is_range)
        self.hv_range_widget.setVisible(is_range)

    def get_photon_energies(self) -> list[float]:
        mode = self.hv_mode_combo.currentData()
        if mode == "range":
            return resolve_photon_energies(
                mode="range",
                start=self.hv_start_spin.value(),
                finish=self.hv_finish_spin.value(),
                step=self.hv_step_spin.value(),
            )
        return resolve_photon_energies(
            mode="single", single=self.photon_energy_spin.value()
        )

    def _on_kz_halfwidth_changed(self, value):
        """Enable kz_npoints only when broadening is on (halfwidth > 0)."""
        self.kz_npoints_spin.setEnabled(float(value) > 0.0)

    def _on_fresnel_enabled_changed(self, checked):
        """Enable optical n/k spins only when Fresnel local field is on."""
        on = bool(checked)
        self.optical_n_spin.setEnabled(on)
        self.optical_k_spin.setEnabled(on)

    def _ensure_kz_npoints_odd(self, value):
        """Force odd sample count for Lorentzian k_z quadrature."""
        v = int(value)
        if v % 2 == 0:
            self.kz_npoints_spin.blockSignals(True)
            self.kz_npoints_spin.setValue(max(3, v + 1))
            self.kz_npoints_spin.blockSignals(False)

    def _kz_npoints_for_physics(self):
        """
        When kz_halfwidth > 0, must send npoints >= 3 (spin default 7).
        physics_from_experiment_kwargs defaults npoints=1; lorentzian treats
        npoints<=1 as OFF even if halfwidth>0.
        """
        if self.kz_halfwidth_spin.value() > 0.0:
            return max(3, int(self.kz_npoints_spin.value()))
        return 1

    def _critic_gap_physics_kwargs(self):
        """Approach B/C knobs shared by local + remote + metadata paths."""
        return {
            "rad_type": self.rad_type_combo.currentText(),
            "mfp": self.mfp_spin.value(),
            "kz_halfwidth": self.kz_halfwidth_spin.value(),
            "kz_npoints": self._kz_npoints_for_physics(),
            "fresnel_enabled": self.fresnel_enabled_chk.isChecked(),
            "optical_n": self.optical_n_spin.value(),
            "optical_k": self.optical_k_spin.value(),
            "include_photon_momentum": self.include_photon_momentum_chk.isChecked(),
        }

    def update_schematic(self, *args):
        self.draw_hemisphere_schematic(
            self.manip_theta_spin.value(),
            self.manip_azi_spin.value(),
            self.manip_tilt_spin.value(),
            self.incidence_angle_spin.value(),
            self.polarization_combo.currentText(),
            self.deflector_angle_spin.value(),
            self.slit_size_spin.value(),
            self.slit_angle_spin.value()
        )

    def draw_hemisphere_schematic(self, theta, azimuth, tilt, incidence_angle, pol_mode, deflector_angle, slit_width, slit_angle):
        self.ax_schematic.clear()
        t_rad, a_rad, tilt_rad = np.radians(theta), np.radians(azimuth), np.radians(tilt)
        R_z = np.array([[np.cos(t_rad), -np.sin(t_rad), 0], [np.sin(t_rad), np.cos(t_rad), 0], [0, 0, 1]])
        R_y = np.array([[np.cos(a_rad), 0, np.sin(a_rad)], [0, 1, 0], [-np.sin(a_rad), 0, np.cos(a_rad)]])
        R_x = np.array([[1, 0, 0], [0, np.cos(tilt_rad), -np.sin(tilt_rad)], [0, np.sin(tilt_rad), np.cos(tilt_rad)]])
        R_total = R_z @ R_y @ R_x
        R_inv = np.linalg.inv(R_total)

        inc_rad = np.radians(incidence_angle)
        lin_ang = np.radians(self.lin_pol_angle_spin.value())
        is_circular = "Circular" in pol_mode
        
        if "Horizontal" in pol_mode: eps_lab = np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0])
        elif "Vertical" in pol_mode: eps_lab = np.array([0.0, 0.0, 1.0])
        elif "Arbitrary" in pol_mode:
            eps_lab = np.cos(lin_ang)*np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0]) + np.sin(lin_ang)*np.array([0.0, 0.0, 1.0])
        else:
            sign = 1 if "Right" in pol_mode else -1
            eps_lab = (np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0]) + sign*1j*np.array([0.0, 0.0, 1.0])) / np.sqrt(2)
        
        eps_sample = R_inv @ eps_lab

        phi = np.linspace(0, 2*np.pi, 40)
        theta_emi = np.linspace(0, np.pi/2, 20)
        PHI, THETA_EMI = np.meshgrid(phi, theta_emi)
        
        X_hemi_loc = 1.2 * np.sin(THETA_EMI) * np.cos(PHI)
        Z_hemi_loc = 1.2 * np.sin(THETA_EMI) * np.sin(PHI)
        Y_hemi_loc = 1.2 * np.cos(THETA_EMI)
        
        dot_product = eps_sample[0]*(X_hemi_loc/1.2) + eps_sample[1]*(Y_hemi_loc/1.2) + eps_sample[2]*(Z_hemi_loc/1.2)
        ME_heatmap = np.abs(dot_product)**2
        
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=ME_heatmap.min(), vmax=ME_heatmap.max())
        colors = cm.afmhot(norm(ME_heatmap))
        
        shape = X_hemi_loc.shape
        coords = np.vstack([X_hemi_loc.flatten(), Y_hemi_loc.flatten(), Z_hemi_loc.flatten()])
        coords_rot = R_total @ coords
        X_hemi_lab, Y_hemi_lab, Z_hemi_lab = coords_rot[0].reshape(shape), coords_rot[1].reshape(shape), coords_rot[2].reshape(shape)
        
        self.ax_schematic.plot_surface(X_hemi_lab, Y_hemi_lab, Z_hemi_lab, facecolors=colors, alpha=0.7, shade=False)

        s_range = np.linspace(-1, 1, 2)
        X_s, Z_s = np.meshgrid(s_range, s_range)
        Y_s = np.zeros_like(X_s)
        s_coords = R_total @ np.vstack([X_s.flatten(), Y_s.flatten(), Z_s.flatten()])
        self.ax_schematic.plot_surface(s_coords[0].reshape(2,2), s_coords[1].reshape(2,2), s_coords[2].reshape(2,2), color='darkgray', alpha=0.9, edgecolor='k')

        axes_lab = R_total @ np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]]).T
        self.ax_schematic.quiver(0,0,0, axes_lab[0,0], axes_lab[1,0], axes_lab[2,0], color='red', linewidth=2, label='Sample X')
        self.ax_schematic.quiver(0,0,0, axes_lab[0,1], axes_lab[1,1], axes_lab[2,1], color='lime', linewidth=2, label='Sample Y (Normal)')
        self.ax_schematic.quiver(0,0,0, axes_lab[0,2], axes_lab[1,2], axes_lab[2,2], color='blue', linewidth=2, label='Sample Z')

        beam_dir = np.array([-np.sin(inc_rad), -np.cos(inc_rad), 0.0]) 
        start_pt = -4 * beam_dir
        t_beam = np.linspace(0, 4, 150)
        
        if is_circular:
            freq = 6 * np.pi
            radius = 0.3
            sign = 1 if "Right" in pol_mode else -1
            perp1 = np.array([0, 0, 1])
            perp2 = np.cross(beam_dir, perp1)
            wave_pts = start_pt[:, None] + beam_dir[:, None] * t_beam + radius * (np.cos(freq * t_beam) * perp1[:, None] + sign * np.sin(freq * t_beam) * perp2[:, None])
            self.ax_schematic.plot(wave_pts[0], wave_pts[1], wave_pts[2], color='magenta', linewidth=2, label='Circular Light')
        else:
            freq = 5 * np.pi
            amplitude = 0.3
            osc_dir = np.real(eps_lab)
            wave_pts = start_pt[:, None] + beam_dir[:, None] * t_beam + osc_dir[:, None] * amplitude * np.sin(freq * t_beam)
            self.ax_schematic.plot(wave_pts[0], wave_pts[1], wave_pts[2], color='gold', linewidth=2, label='Linear Light')
            self.ax_schematic.quiver(start_pt[0], start_pt[1], start_pt[2], np.real(eps_lab)[0], np.real(eps_lab)[1], np.real(eps_lab)[2], length=1.5, color='magenta', linewidth=2, label='E-Field')
        
        defl_rad, slit_rot = np.radians(deflector_angle), np.radians(slit_angle)
        
        # --- FIXED DETECTOR PLANE ---
        # The detector is physically bolted to the Lab Frame at Y = 1.25
        y_det = 1.25
        plane_size = 0.5
        px, pz = np.meshgrid(np.linspace(-plane_size, plane_size, 5), np.linspace(-plane_size, plane_size, 5))
        py = np.full_like(px, y_det)
        
        self.ax_schematic.plot_surface(px, py, pz, color='gray', alpha=0.3, edgecolor='blue', label='Detector Plane')

        # --- DRAW ANALYZER SLIT (Fixed Cyan Line) ---
        slit_len = 0.5
        sx = slit_len * np.cos(slit_rot)
        sz = slit_len * np.sin(slit_rot)
        self.ax_schematic.plot([-sx, sx], [y_det, y_det], [-sz, sz], color='cyan', linewidth=4, zorder=10, label='Analyzer Slit')

        # --- DRAW DEFLECTOR CUT (Animated Yellow Line) ---
        # The deflector shifts the measurement orthogonally to the slit
        shift_dist = y_det * np.tan(defl_rad)
        dx_shift = shift_dist * (-np.sin(slit_rot))
        dz_shift = shift_dist * (np.cos(slit_rot))
        
        self.ax_schematic.plot([-sx + dx_shift, sx + dx_shift], 
                               [y_det, y_det], 
                               [-sz + dz_shift, sz + dz_shift], 
                               color='yellow', linestyle='--', linewidth=3, zorder=11, label='Deflector Cut')
                               
        # Draw a small arrow showing the deflection direction
        if abs(deflector_angle) > 0.1:
            self.ax_schematic.quiver(0, y_det, 0, dx_shift, 0, dz_shift, color='yellow', linewidth=2, arrow_length_ratio=0.3)

        self.ax_schematic.plot([0, 0], [0, 0], [-3, 3], color='black', linestyle='-.', linewidth=1, label='Lab Z (Theta)')

        self.ax_schematic.set_xlim([-3, 3]); self.ax_schematic.set_ylim([-3, 3]); self.ax_schematic.set_zlim([-3, 3])
        self.ax_schematic.set_xlabel('Lab X (Side)'); self.ax_schematic.set_ylabel('Lab Y (Forward)'); self.ax_schematic.set_zlabel('Lab Z (Vertical)')
        self.ax_schematic.set_title("Experimental Geometry Schematic", pad=0)
        self.ax_schematic.legend(loc='center left', fontsize=7, bbox_to_anchor=(1.05, 0.5))
        self.ax_schematic.xaxis.pane.fill = False; self.ax_schematic.yaxis.pane.fill = False; self.ax_schematic.zaxis.pane.fill = False
        
        self.schematic_figure.subplots_adjust(left=0.0, right=0.8, top=1.0, bottom=0.0) 
        self.schematic_canvas.draw()

    def trigger_simulation(self):
        model_choice = self.engine_dropdown.currentData()
        try:
            hv_list = self.get_photon_energies()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid photon-energy range", str(exc))
            return
        if len(hv_list) > 64:
            reply = QMessageBox.question(
                self,
                "Large photon-energy scan",
                f"This scan contains {len(hv_list)} photon energies. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        
        if model_choice == "B3":
            # --- SPR-KKR (B3) native execution: workflow.run_arpes via ARPESRunnerThread ---
            import os
            import time as _time

            kx_min, kx_max, kx_steps = self.spin_kx_min.value(), self.spin_kx_max.value(), self.spin_kx_steps.value()
            ky_min, ky_max, ky_steps = self.spin_ky_min.value(), self.spin_ky_max.value(), self.spin_ky_steps.value()
            e_min, e_max, e_steps = self.spin_e_min.value(), self.spin_e_max.value(), self.spin_e_steps.value()
            ts = _time.strftime("%Y%m%d_%H%M%S")

            # Same min==max -> 1 sample rule as chinook branch / _update_kkr_eta.
            kx_steps_eff = 1 if kx_min == kx_max else kx_steps
            ky_steps_eff = 1 if ky_min == ky_max else ky_steps
            e_steps_eff = 1 if e_min == e_max else e_steps
            nproc = self.spin_kkr_nproc.value()
            try:
                eta_store = os.path.expanduser("~/.tensorspec_sprkkr_eta.json")
                eta_model = SprkkrEtaModel(store_path=eta_store)
                eta_model.load()
                eta_params = SprkkrArpesParams(ne=e_steps_eff, nt=kx_steps_eff, np_=ky_steps_eff)
                eta_seconds = eta_model.estimate_seconds(eta_params, nproc)
            except Exception:
                eta_seconds = 0.0
            if self.chk_pointwise.isChecked() and kx_steps_eff > 64:
                reply = QMessageBox.question(
                    self,
                    "Point-wise mode: many jobs",
                    f"Point-wise mode launches {kx_steps_eff} kkrspec jobs at once on {nproc} ranks. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return
            if eta_seconds > 900:
                n_points = e_steps_eff * kx_steps_eff * ky_steps_eff
                reply = QMessageBox.question(
                    self,
                    "Long SPR-KKR run",
                    f"SPR-KKR run ≈ {sprkkr_format_eta(eta_seconds)}, "
                    f"{n_points} points on {nproc} ranks. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

            experiment_kwargs = {
                'photon_energy': float(hv_list[0]),
                'polarization': self.polarization_combo.currentText(),
                'work_function': self.work_function_spin.value(),
                # SPEC_PH THETA (light incidence angle): was hardcoded at
                # ArpesParams' 45.0 default and never read from this widget --
                # "Beam Incidence (Lab)" looked editable for B3 but had zero
                # effect on the SPR-KKR .inp. theta_ph passes straight through
                # _build_arpes_params' generic passthrough (real ArpesParams
                # field, not in _EXPLICIT_FIELDS).
                'theta_ph': self.incidence_angle_spin.value(),
                'pointwise': self.chk_pointwise.isChecked(),
                'slit_angle': self.slit_angle_spin.value(),
                # Deflector = the "Φ (Deflect)" range row (ky), same knob every run so far
                # used (Sep-13 run: ky -10.3..-10.3 -> PHI=-10.3). The analyzer-group
                # "Deflector Angle" spin only drives the schematic. A ky RANGE (min!=max)
                # is a Fermi-map ask -> None here, and kkr_wrapper raises in pointwise mode.
                'deflector_angle': (ky_min if ky_min == ky_max else None),
                'manip_theta': self.manip_theta_spin.value(),
                'manip_azimuth': self.manip_azi_spin.value(),
                'manip_tilt': self.manip_tilt_spin.value(),
                'phi_offset_deg': self.spin_phi_offset.value(),
                'ref_energy_eV': self.spin_ref_energy.value(),
                'k_bounds': {'X': [kx_min, kx_max, kx_steps], 'Y': [ky_min, ky_max, ky_steps]},
                'e_min': e_min,
                'e_max': e_max,
                'e_steps': e_steps,
                'hkl': (self.spin_h.value(), self.spin_k.value(), self.spin_l.value()),
                'n_layer': self.spin_n_layer.value(),
                'nlat_g_vec': self.spin_nlat_g_vec.value(),
                'imv_ini_eV': self.spin_imv_ini.value(),
                'imv_fin_eV': self.spin_imv_fin.value(),
                'nl': self.spin_kkr_nl.value(),
                'nproc': self.spin_kkr_nproc.value(),
                'mode': self.combo_kkr_mode.currentText(),
                'workdir': f"scratch/sprkkr_gui_run/{ts}",
                'temperature_K': self.temperature_spin.value(),
            }
            # SARPES filter (same widgets as B1): axis -> SPR-KKR POL_E, comp -> I_up/I_dn.
            if self.chk_enable_sarpes.isChecked():
                axis_map = {"Sz (Z-Axis)": "PZ", "Sx (X-Axis)": "PX", "Sy (Y-Axis)": "PY"}
                comp_map = {"Spin Up (+)": "up", "Spin Down (-)": "down"}
                experiment_kwargs['pol_e'] = axis_map.get(self.combo_spin_axis.currentText(), "PZ")
                experiment_kwargs['spin_filter'] = comp_map.get(self.combo_spin_comp.currentText(), "up")

            # Surface geometry (design doc §0): HKL frame combo picks CIF vs ABAS.
            frame_choice = self.combo_hkl_frame.currentData() or "auto"
            try:
                hkl_frame, cif_lattice = self._resolve_sprkkr_hkl_frame(frame_choice)
            except ValueError as exc:
                QMessageBox.warning(self, "HKL frame", str(exc))
                return
            experiment_kwargs["hkl_frame"] = hkl_frame
            if cif_lattice is not None:
                experiment_kwargs["cif_lattice"] = cif_lattice
            experiment_kwargs['iq_at_surf'] = None if self.chk_iq_auto.isChecked() else self.spin_iq_surf.value()

            try:
                if self._is_remote_target():
                    cluster = self.get_selected_cluster()
                    if not cluster:
                        QMessageBox.critical(self, "Error", "No remote cluster found in configuration!")
                        return
                    vault_name = self.vault_combo.currentText()
                    if not vault_name or vault_name == "No SPRKKR Vaults Found":
                        QMessageBox.warning(self, "Error", "No valid SPRKKR Vault selected! Run an SCF job first.")
                        return

                    from tensorspec.gui.components.sprkkr_panels import (
                        format_pot_summary,
                        remote_latest_scf_dir,
                        remote_pot_summary,
                    )

                    if vault_name == "Temporary Scratch Run (sprkkr_gui_run)":
                        scratch = cp.job_dir(cluster, "sprkkr")
                        # Newest CONVERGED scf_*/scf.pot_new only. No fallback to the
                        # scratch-root pot: 2026-09-12 that was a stale unconverged Cu
                        # test pot and a run silently went out on it.
                        pot_dir = remote_latest_scf_dir(cluster, scratch)
                        if not pot_dir:
                            QMessageBox.warning(
                                self, "Error",
                                f"No CONVERGED scf_*/scf.pot_new under {scratch} on "
                                f"{cluster.get('name', 'cluster')}. Run SCF first, or "
                                "pick a named vault.",
                            )
                            return
                        remote_dir = scratch
                        experiment_kwargs["pot_path"] = f"{pot_dir}/scf.pot_new"
                        experiment_kwargs["remote_workdir"] = f"{remote_dir}/arpes_{ts}"
                    else:
                        vault = global_workspace.get(vault_name)
                        remote_dir = vault.get('remote_path') if vault else None
                        if not remote_dir:
                            QMessageBox.warning(
                                self, "Error",
                                f"Vault '{vault_name}' has no remote_path in the workspace. "
                                "Hit Refresh on the vault list and pick it again.",
                            )
                            return
                        experiment_kwargs['pot_path'] = f"{remote_dir}/scf.pot_new"
                        experiment_kwargs['remote_workdir'] = f"{remote_dir}/arpes_{ts}"

                    # Guard: read the remote pot header and make the user eyeball it.
                    pot_info = remote_pot_summary(cluster, experiment_kwargs["pot_path"])
                    if not pot_info["exists"]:
                        QMessageBox.critical(
                            self, "Error",
                            f"Pot not found on cluster:\n{experiment_kwargs['pot_path']}",
                        )
                        return
                    if pot_info["scfstatus"].upper() != "CONVERGED":
                        QMessageBox.critical(
                            self, "Refusing to run",
                            "Selected pot is NOT a converged SCF result:\n\n"
                            + format_pot_summary(pot_info)
                            + "\n\nPick a converged vault or rerun SCF.",
                        )
                        return
                    reply = QMessageBox.question(
                        self, "Confirm SPR-KKR ARPES pot",
                        "kkrspec will run on this potential:\n\n"
                        + format_pot_summary(pot_info)
                        + "\n\nIs this the right material?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
                    )
                    if reply != QMessageBox.Yes:
                        return

                    experiment_kwargs['workdir'] = f"scratch/sprkkr_gui_run/arpes_{ts}"
                    experiment_kwargs['launcher'] = SprkkrRemoteLauncher(cluster)
                else:
                    pot_path = self.sprkkr_pot_edit.text().strip()
                    if not pot_path or not os.path.isfile(pot_path):
                        QMessageBox.warning(self, "Error", "Pick a valid local converged .pot_new file first (Local pot: Browse).")
                        return
                    experiment_kwargs['pot_path'] = pot_path
                    experiment_kwargs['launcher'] = SprkkrLocalLauncher(bin_dir=self.sprkkr_bin_edit.text().strip())
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to prepare SPR-KKR ARPES run:\n{str(e)}")
                return

            self.ax.clear()
            self.ax.set_title("Calculating SPR-KKR (B3) ARPES... (Running in Background)")
            self.canvas.draw()

            self.run_sim_btn.setEnabled(False)
            self.run_sim_btn.setText("⏳ SPR-KKR running... Please Wait")
            self.run_sim_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #555555; color: white;")

            self.arpes_thread = ARPESRunnerThread(
                self.engine_router,
                "B3",
                {},
                experiment_kwargs,
                photon_energies=hv_list,
            )
            self.arpes_thread.progress.connect(
                lambda text: self.run_sim_btn.setText(f"⏳ {text}")
            )
            self.arpes_thread.finished_signal.connect(self.on_simulation_finished)
            self.arpes_thread.start()

            if self._is_remote_target():
                self.btn_fetch_results.setEnabled(True)
                # Auto-start the live monitor so the user sees the output immediately
                self.toggle_embedded_monitor(force_start=True)
            return

        # --- CHINOOK EXECUTION (LOCAL OR REMOTE CLUSTER) ---
        target_crystal = self.ws_combo.currentText()
        me_mode = self.matrix_element_combo.currentText()

        print("\n" + "="*50)
        print("🚀 INITIATING ARPES MATRIX ELEMENT SIMULATION")
        print("="*50)
        print(f"Target Structure : {target_crystal}")
        print(f"Selected Physics : {model_choice} ({me_mode})")
        print(f"Compute Target   : {self.target_dropdown.currentText()}")
        print(f"Polarization     : {self.polarization_combo.currentText()}")
        print(f"Incidence Angle  : {self.incidence_angle_spin.value()}\u00B0")
        print("="*50 + "\n")

        e_min, e_max, e_steps = self.spin_e_min.value(), self.spin_e_max.value(), self.spin_e_steps.value()
        kx_min, kx_max, kx_steps = self.spin_kx_min.value(), self.spin_kx_max.value(), self.spin_kx_steps.value()
        ky_min, ky_max, ky_steps = self.spin_ky_min.value(), self.spin_ky_max.value(), self.spin_ky_steps.value()
        # Same as local chinook_wrapper: min==max → 1 sample (avoid linspace(0,0,40) waste).
        if kx_min == kx_max:
            kx_steps = 1
        if ky_min == ky_max:
            ky_steps = 1
        if e_min == e_max:
            e_steps = 1

        band_data = global_workspace.pull_band_structure(target_crystal)

        if not band_data:
            print(">> WARNING: No pre-calculated band structure found.")
            return
        
        print("\n[LOG 2 - ARPES PANEL]")
        print(f"Dictionary keys pulled: {list(band_data.keys())}")
        print(">> SUCCESS: Band structure loaded. Routing to Matrix Element Engine...")
        
        from tensorspec.gui.services.compute_mode import is_hybrid_mode

        if is_hybrid_mode(self.target_dropdown):
            # --- REMOTE CHINOOK DISPATCH ---
            cluster = None
            cluster_label = "remote cluster"
            try:
                import json
                from pathlib import Path

                cluster = self.get_selected_cluster()
                if not cluster:
                    QMessageBox.critical(self, "Error", "No remote cluster found in configuration!")
                    return
                cluster_label = cluster.get("name") or cluster.get("host") or cluster_label

                remote_dir = cp.job_dir(cluster, "chinook")
                py = cp.python_bin(cluster)
                ssh = self._ssh_connect(cluster)
                sftp = ssh.open_sftp()
                
                try:
                    sftp.mkdir(cp.heavy_root(cluster))
                except OSError:
                    pass
                try:
                    sftp.mkdir(remote_dir)
                except OSError:
                    pass
                    
                # Upload runner modules (absolute paths — cwd is not always repo root)
                one_step = (
                    Path(__file__).resolve().parents[2]
                    / "core"
                    / "arpes"
                    / "one_step"
                )
                local_template = one_step / "chinook_remote_runner_template.py"
                local_kmesh = one_step / "chinook_arpes_kmesh.py"
                local_schedule = one_step / "grizzly_cuda_schedule.py"
                local_collect = one_step / "grizzly_multigpu_collect.py"
                local_fresnel = one_step / "fresnel.py"
                local_photon_q = one_step / "photon_momentum.py"
                for src, dst_name in (
                    (local_template, "chinook_remote_runner.py"),
                    (local_kmesh, "chinook_arpes_kmesh.py"),
                    (local_schedule, "grizzly_cuda_schedule.py"),
                    (local_collect, "grizzly_multigpu_collect.py"),
                    (local_fresnel, "fresnel.py"),
                    (local_photon_q, "photon_momentum.py"),
                ):
                    if not src.is_file():
                        raise FileNotFoundError(f"Missing ARPES runner module: {src}")
                    sftp.put(str(src), f"{remote_dir}/{dst_name}")
                
                # Save and upload TB data
                scratch = Path.home() / ".tensorspec_cache" / "chinook_gui_run"
                scratch.mkdir(parents=True, exist_ok=True)
                tb_path_local = str(scratch / "tb_data.npz")
                physics_path_local = str(scratch / "arpes_physics.json")
                
                # Compress massive H_dict into binary arrays.
                # R vectors are Cartesian floats (Wannier/SK) — must stay float64.
                # (int32 truncation zeroed remote ARPES intensity at finite angle.)
                h_dict = band_data.get("H_dict") or {}
                h_list = h_dict.get("list") or h_dict.get("H") or []
                if len(h_list) == 0:
                    # Hybrid can plot bands before W90 cache finishes syncing;
                    # refill from ~/.tensorspec_cache/w90_tb if present.
                    filled = self._try_fill_h_dict_from_w90_cache(band_data)
                    if filled:
                        h_dict = band_data.get("H_dict") or {}
                        h_list = h_dict.get("list") or h_dict.get("H") or []
                if len(h_list) > 0:
                    indices = np.array(
                        [[h[0], h[1], h[2], h[3], h[4]] for h in h_list],
                        dtype=np.float64,
                    )
                    values = np.array([h[5] for h in h_list], dtype=np.complex128)
                else:
                    indices = np.empty((0, 5), dtype=np.float64)
                    values = np.empty(0, dtype=np.complex128)

                if indices.shape[0] == 0:
                    raise ValueError(
                        "Workspace TB has zero hoppings (H_dict.list empty). "
                        "Re-run Prepare TB for ARPES / Calculate Bands, then Push again."
                    )

                basis_list = self._pack_basis_list_for_remote(band_data)
                if not basis_list:
                    raise ValueError(
                        "Workspace TB has no orbital basis for remote ARPES. "
                        "Re-run Prepare TB for ARPES (needs Wannier + structure), then Push."
                    )
                
                b_matrix = band_data.get("recip_matrix")
                if b_matrix is None and band_data.get("structure") is not None:
                    b_matrix = band_data["structure"].lattice.reciprocal_lattice.matrix
                elif b_matrix is None:
                    a_list = h_dict.get("a")
                    if a_list is not None:
                        A = np.array(a_list, dtype=float)
                        b_matrix = 2 * np.pi * np.linalg.inv(A).T
                if b_matrix is None:
                    b_matrix = 2 * np.pi * np.eye(3)

                np.savez_compressed(
                    tb_path_local,
                    indices=indices,
                    values=values,
                    basis_list=basis_list,
                    a_mat=h_dict.get("a", []),
                    b_matrix=np.asarray(b_matrix, dtype=float),
                    e_fermi=float(
                        band_data.get(
                            "arpes_e_fermi_shift",
                            band_data.get("fermi_energy", 0.0),
                        )
                        or 0.0
                    ),
                    onsite_e=float(band_data.get("onsite_e", 0.0) or 0.0),
                    fermi_energy_qe=float(band_data.get("fermi_energy", 0.0) or 0.0),
                    h_includes_onsite=bool(band_data.get("h_includes_onsite", False)),
                )
                print(
                    f"[ARPES remote] TB upload: hops={len(indices)} "
                    f"indices.dtype={indices.dtype} "
                    f"onsite_e={band_data.get('onsite_e', 'MISSING')} "
                    f"QE_fermi={band_data.get('fermi_energy', 'MISSING')} "
                    f"arpes_e_shift={band_data.get('arpes_e_fermi_shift', band_data.get('fermi_energy', 0.0))} "
                    f"R_sample={indices[0, 2:5] if len(indices) else 'n/a'}",
                    flush=True,
                )

                from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import (
                    physics_from_experiment_kwargs,
                )

                remote_physics = physics_from_experiment_kwargs(
                    {
                        "photon_energy": self.photon_energy_spin.value(),
                        "work_function": self.work_function_spin.value(),
                        "inner_potential": self.inner_potential_spin.value(),
                        "temperature": self.temperature_spin.value(),
                        "incidence_angle": self.incidence_angle_spin.value(),
                        "polarization": self.polarization_combo.currentText(),
                        "lin_pol_angle": self.lin_pol_angle_spin.value(),
                        "matrix_element_mode": me_mode,
                        "manip_theta": self.manip_theta_spin.value(),
                        "manip_azimuth": self.manip_azi_spin.value(),
                        "manip_tilt": self.manip_tilt_spin.value(),
                        "hkl": (
                            self.spin_h.value(),
                            self.spin_k.value(),
                            self.spin_l.value(),
                        ),
                        "slit_angle": self.slit_angle_spin.value(),
                        "se_width": self.ui_se_spinbox.value(),
                        "res_E": self.ui_res_e_spinbox.value(),
                        "res_k": self.ui_res_k_spinbox.value(),
                        **self._critic_gap_physics_kwargs(),
                    }
                )
                with open(physics_path_local, "w") as pf:
                    json.dump(
                        {**remote_physics, "hkl": list(remote_physics["hkl"])},
                        pf,
                        indent=2,
                    )
                    
                sftp.put(tb_path_local, f"{remote_dir}/tb_data.npz")
                sftp.put(physics_path_local, f"{remote_dir}/arpes_physics.json")
                sftp.close()
                
                cores = self.remote_tb_cores_spin.value()
                hv_list = self.get_photon_energies()
                hv_args = _remote_hv_cli_args(
                    hv_list,
                    self.hv_start_spin.value(),
                    self.hv_finish_spin.value(),
                    self.hv_step_spin.value(),
                )
                workf = self.work_function_spin.value()
                v0 = self.inner_potential_spin.value()
                temp = self.temperature_spin.value()
                polar = "P" if "Linear" in self.polarization_combo.currentText() else "S"
                e_fermi = float(
                    band_data.get(
                        'arpes_e_fermi_shift',
                        band_data.get('fermi_energy', 0.0),
                    )
                    or 0.0
                )
                me_engine = self.combo_remote_me.currentData()
                me_device = self.combo_remote_device.currentData()
                me_layout = self.combo_remote_layout.currentData()
                me_ngpus = self._resolve_ngpus_for_submit(me_device, me_layout)

                run_args = (
                    f"--tb_file tb_data.npz --theta_min {kx_min} --theta_max {kx_max} --ntheta {kx_steps} "
                    f"--phi_min {ky_min} --phi_max {ky_max} --nphi {ky_steps} "
                    f"--e_min {e_min} --e_max {e_max} --ne {e_steps} "
                    f"{hv_args} --workf {workf} --v0 {v0} --temp {temp} --polar {polar} "
                    f"--cores {cores} --engine {me_engine} --device {me_device} "
                    f"--layout {me_layout} --e_fermi {e_fermi} --theta_chunk 0 --ngpus {me_ngpus}"
                )

                # SARPES Args (forces chinook fallback inside runner; GrizzlyME v0.1 is spinless)
                if self.chk_enable_sarpes.isChecked():
                    axis_map = {"Sz (Z-Axis)": "Z", "Sx (X-Axis)": "X", "Sy (Y-Axis)": "Y"}
                    comp_map = {"Spin Up (+)": 1, "Spin Down (-)": -1}
                    run_args += f" --sarpes True --spin_axis {axis_map[self.combo_spin_axis.currentText()]} --spin_comp {comp_map[self.combo_spin_comp.currentText()]}"

                
                is_slurm = cluster.get('mode', '').upper() == 'SLURM'
                if is_slurm:
                    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=chinook_arpes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cores}
#SBATCH --output=sys.out.full
#SBATCH --error=sys.out.full

cd {remote_dir}
{cp.shell_export_tmp(cluster)}
{cp.shell_thread_limits()}

{py} -u chinook_remote_runner.py {run_args}
"""
                    sftp = ssh.open_sftp()
                    with sftp.file(f"{remote_dir}/job.sbatch", "w") as f:
                        f.write(sbatch_content)
                    sftp.close()
                    cmd = f"cd {remote_dir} && sbatch job.sbatch"
                else:
                    cmd = (
                        f"bash -c 'cd {remote_dir} && "
                        f"{cp.shell_export_tmp(cluster, one_line=True)} && "
                        f"{cp.shell_thread_limits(one_line=True)} && "
                        f"nohup {py} -u chinook_remote_runner.py {run_args} > sys.out.full 2>&1 &'"
                    )
                
                ssh.exec_command(cmd)
                ssh.close()
                
                self.btn_fetch_results.setEnabled(True)
                cluster_label = cluster.get('name', cluster['host'])
                msg_type = "via SLURM Queue" if is_slurm else "via Background Daemon"
                QMessageBox.information(self, "Success", f"CHINOOK ARPES map dispatched to {cluster_label} ({msg_type}) with {cores} cores!\n\nLogs: {remote_dir}/sys.out.full\n\nCheck 'Calculation Live Logs' tab to watch progress!")
                self.toggle_embedded_monitor(force_start=True)
                return
            except Exception as e:
                import traceback

                traceback.print_exc()
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to dispatch CHINOOK to {cluster_label}:\n{e}",
                )
                return
        
        # --- LOCAL EXECUTION ---
        experiment_kwargs = {
            'photon_energy': float(hv_list[0]),
            'photon_energies': hv_list,
            'work_function': self.work_function_spin.value(),
            'inner_potential': self.inner_potential_spin.value(), 
            'temperature': self.temperature_spin.value(),         
            'incidence_angle': self.incidence_angle_spin.value(),
            'polarization': self.polarization_combo.currentText(),
            'lin_pol_angle': self.lin_pol_angle_spin.value(),
            'matrix_element_mode': me_mode,
            'manip_theta': self.manip_theta_spin.value(),
            'manip_azimuth': self.manip_azi_spin.value(),
            'manip_tilt': self.manip_tilt_spin.value(),
            'hkl': (self.spin_h.value(), self.spin_k.value(), self.spin_l.value()),

            # Send standard relative bounds (e.g., -2.0 to 0.5) to the engine
            'k_bounds': {'X': [kx_min, kx_max, kx_steps], 'Y': [ky_min, ky_max, ky_steps], 'E': [e_min, e_max, e_steps]},
            'se_width': self.ui_se_spinbox.value(),
            'res_E': self.ui_res_e_spinbox.value(),
            'res_k': self.ui_res_k_spinbox.value(),
            'slit_angle': self.slit_angle_spin.value(),
            **self._critic_gap_physics_kwargs(),
        }
        
        # --- THREADED LAUNCH ---
        self.ax.clear()
        self.ax.set_title("Calculating 2D Matrix Elements... (Running in Background)")
        self.canvas.draw()
        
        self.run_sim_btn.setEnabled(False)
        self.run_sim_btn.setText("⏳ Calculating... Please Wait")
        self.run_sim_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #555555; color: white;")

        self.arpes_thread = ARPESRunnerThread(
            self.engine_router,
            model_choice,
            band_data,
            experiment_kwargs,
            photon_energies=hv_list,
        )
        self.arpes_thread.progress.connect(
            lambda text: self.run_sim_btn.setText(f"⏳ {text}")
        )
        self.arpes_thread.finished_signal.connect(self.on_simulation_finished)
        self.arpes_thread.start()

    def on_simulation_finished(self, success, results, message):
        self.run_sim_btn.setEnabled(True)
        self.run_sim_btn.setText("🚀 Run ARPES Simulation")
        self.run_sim_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #2b5c8f; color: white;")

        if success:
            self.btn_push_workspace.setEnabled(True)
            self.btn_save_disk.setEnabled(True)

            # SPR-KKR (B3) only: show the resolved surface geometry (hkl_abas / IQ).
            geometry = results.get('geometry') if isinstance(results, dict) else None
            if geometry is not None:
                self.lbl_kkr_geom.setText(
                    f"geometry: hkl_abas={tuple(geometry.hkl_abas)} "
                    f"IQ_AT_SURF={geometry.iq_at_surf} ({geometry.planes_summary})"
                )

            e_min, e_max, e_steps = self.spin_e_min.value(), self.spin_e_max.value(), self.spin_e_steps.value()
            kx_min, kx_max, kx_steps = self.spin_kx_min.value(), self.spin_kx_max.value(), self.spin_kx_steps.value()
            ky_min, ky_max, ky_steps = self.spin_ky_min.value(), self.spin_ky_max.value(), self.spin_ky_steps.value()
            if kx_min == kx_max:
                kx_steps = 1
            if ky_min == ky_max:
                ky_steps = 1
            if e_min == e_max:
                e_steps = 1

            self.sim_intensity = results['intensity_broadened']
            if self.sim_intensity.ndim == 4:
                photon_energies = results.get("photon_energies")
                self.sim_hv = (
                    np.asarray(photon_energies, dtype=float)
                    if photon_energies is not None
                    else None
                )
                preview_intensity = self.sim_intensity[
                    self.sim_intensity.shape[0] // 2
                ]
            else:
                self.sim_hv = None
                preview_intensity = self.sim_intensity
            
            # Prefer axes from remote npz when present (truth for remote fetches).
            nx, ny, ne = preview_intensity.shape
            if results.get('energy') is not None:
                self.sim_E_axis = np.asarray(results['energy'], dtype=float)
            else:
                self.sim_E_axis = np.linspace(e_min, e_max, ne)
            if results.get('theta') is not None:
                self.sim_kx = np.asarray(results['theta'], dtype=float)
            else:
                self.sim_kx = np.linspace(kx_min, kx_max, nx)
            if results.get('phi') is not None:
                self.sim_ky = np.asarray(results['phi'], dtype=float)
            else:
                self.sim_ky = np.linspace(ky_min, ky_max, ny)

            # Remote/UI grids are emission angles (deg), not Å⁻¹.
            self.sim_axes_are_angles = True
            
            if (
                ny == 1
                or nx == 1
                or self._axis_degenerate(self.sim_ky)
                or self._axis_degenerate(self.sim_kx)
            ):
                # 2D Band Dispersion Map (Energy vs angle)
                self.energy_slider.setEnabled(False)
                self.energy_spinbox.setEnabled(False)
                self.update_plot_slice(index=None)
            else:
                # 3D Cube (Constant Energy Contours)
                self.energy_slider.setRange(0, ne - 1)
                
                # Setup Spinbox Range
                self.energy_spinbox.setRange(float(self.sim_E_axis[0]), float(self.sim_E_axis[-1]))
                
                fermi_idx = int(np.abs(self.sim_E_axis).argmin())
                self.energy_slider.setEnabled(True)
                self.energy_spinbox.setEnabled(True)
                
                # Block signals to prevent double-firing during initialization
                self.energy_slider.blockSignals(True)
                self.energy_spinbox.blockSignals(True)
                self.energy_slider.setValue(fermi_idx)
                self.energy_spinbox.setValue(self.sim_E_axis[fermi_idx])
                self.energy_slider.blockSignals(False)
                self.energy_spinbox.blockSignals(False)
                
                self.update_plot_slice(fermi_idx)
        else:
            QMessageBox.critical(self, "Simulation Error", f"An error occurred in the physics router:\n{message}")
            self.ax.set_title("Simulation Failed")
            self.canvas.draw()

    def _sync_slider_to_spinbox(self, index):
        """Updates the spinbox text when the slider moves."""
        if not hasattr(self, 'sim_E_axis'): return
        self.energy_spinbox.blockSignals(True)
        self.energy_spinbox.setValue(self.sim_E_axis[index])
        self.energy_spinbox.blockSignals(False)
        self.update_plot_slice(index)

    def _sync_spinbox_to_slider(self, val):
        """Finds the nearest array index when the user types a value in the spinbox."""
        if not hasattr(self, 'sim_E_axis'): return
        nearest_idx = (np.abs(self.sim_E_axis - val)).argmin()
        self.energy_slider.blockSignals(True)
        self.energy_slider.setValue(nearest_idx)
        self.energy_slider.blockSignals(False)
        self.update_plot_slice(nearest_idx)

    def update_plot_slice(self, index=None):
        if not hasattr(self, 'sim_intensity'):
            return

        intensity = self.sim_intensity
        if intensity.ndim == 4:
            intensity = intensity[intensity.shape[0] // 2]
        nx, ny, ne = intensity.shape
        self.ax.clear()

        use_angles = getattr(self, 'sim_axes_are_angles', True)
        x_lab = r"$\Theta$ (slit) [deg]" if use_angles else r"$k_x$ ($\mathrm{\AA}^{-1}$)"
        y_lab = r"$\Phi$ (deflect) [deg]" if use_angles else r"$k_y$ ($\mathrm{\AA}^{-1}$)"

        # Degenerate φ=0…0 (or θ) → band map, not a squeezed contour.
        deg_y = ny == 1 or self._axis_degenerate(self.sim_ky)
        deg_x = nx == 1 or self._axis_degenerate(self.sim_kx)

        # --- ROUTE 1: PLOT BAND DISPERSION (E vs angle) ---
        if deg_y or deg_x:
            self.energy_label.setText("Band Dispersion Map")
            
            if deg_y:
                slice_2d = intensity[:, 0, :].T
                x_axis = self.sim_kx
                x_label = x_lab
            else:
                slice_2d = intensity[0, :, :].T
                x_axis = self.sim_ky
                x_label = y_lab
                
            slice_max = np.max(slice_2d) if np.max(slice_2d) > 0 else 1.0
            norm_slice = slice_2d / slice_max
            gamma_corrected = np.power(norm_slice, self.gamma_spin.value())
            
            self.ax.pcolormesh(x_axis, self.sim_E_axis, gamma_corrected, shading='auto', cmap='afmhot', 
                               vmin=self.vmin_spin.value(), vmax=self.vmax_spin.value())
            
            self.ax.set_aspect('auto')
            self.ax.set_title("Simulated ARPES Band Dispersion")
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(r"$E - E_F$ [eV]")

        # --- ROUTE 2: PLOT CONSTANT ENERGY CONTOUR ---
        else:
            if index is None: 
                index = self.energy_slider.value()
                
            E_val = self.sim_E_axis[index]
            
            slice_2d = intensity[:, :, index].T
            
            slice_max = np.max(slice_2d) if np.max(slice_2d) > 0 else 1.0
            norm_slice = slice_2d / slice_max
            gamma_corrected = np.power(norm_slice, self.gamma_spin.value())
            
            self.ax.pcolormesh(self.sim_kx, self.sim_ky, gamma_corrected, shading='auto', cmap='afmhot', 
                               vmin=self.vmin_spin.value(), vmax=self.vmax_spin.value())

            span_x = float(np.ptp(self.sim_kx))
            span_y = float(np.ptp(self.sim_ky))
            if span_x > 1e-9 and span_y > 1e-9 and 0.05 < (span_y / span_x) < 20:
                self.ax.set_aspect('equal')
            else:
                self.ax.set_aspect('auto')
            self.ax.set_title(f"Constant Energy Contour: {E_val:.2f} eV")
            self.ax.set_xlabel(x_lab)
            self.ax.set_ylabel(y_lab)
            
            if getattr(self, 'chk_overlay_bz', None) and self.chk_overlay_bz.isChecked():
                self._draw_bz_overlay()
            
        self.figure.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        self.canvas.draw()
        
    def populate_clusters(self):
        """Refresh cluster list (Hybrid entries) after Compute Manager changes."""
        from tensorspec.gui.services.compute_mode import populate_compute_mode_combo

        populate_compute_mode_combo(self.target_dropdown)

    def get_selected_cluster(self):
        from tensorspec.gui.services.compute_mode import combo_cluster

        return combo_cluster(self.target_dropdown)

    def _ssh_connect(self, cluster):
        """Paramiko connect with longer banner/kex timeouts (slow remote hosts)."""
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        pwd = cluster.get('password', '') or None
        # banner_timeout / auth_timeout matter more than connect timeout for kex stalls
        ssh.connect(
            cluster['host'],
            port=cluster.get('port', 22),
            username=cluster['user'],
            password=pwd,
            timeout=30,
            banner_timeout=90,
            auth_timeout=60,
            allow_agent=True,
            look_for_keys=True,
        )
        return ssh

    @staticmethod
    def _axis_degenerate(axis) -> bool:
        axis = np.asarray(axis, dtype=float)
        return axis.size <= 1 or float(np.ptp(axis)) < 1e-9

    def toggle_embedded_monitor(self, force_start=False):
        if self.live_monitor and self.live_monitor.isRunning():
            if force_start:
                return
            # Stop it
            self.live_monitor.stop()
            self.live_monitor.wait()
            self.btn_start_live.setText("▶️ Start Live Monitor")
            self.btn_start_live.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; padding: 4px;")
            return
            
        active_cluster = self.get_selected_cluster()
        if not active_cluster:
            QMessageBox.warning(self, "No Cluster", "No remote cluster configured in Cluster Connections!")
            return
        
        from tensorspec.gui.components.compute_panel import LiveMonitorThread

        model_choice = self.engine_dropdown.currentData() if hasattr(self, "engine_dropdown") else None
        if model_choice == "B3":
            log_jobs = ["sprkkr"]
        else:
            log_jobs = ["chinook"]

        self.live_monitor = LiveMonitorThread(active_cluster, log_jobs=log_jobs)
        self.live_monitor.data_ready.connect(self.update_embedded_logs)
        self.live_monitor.start()
        self.btn_start_live.setText("⏹️ Stop Live Monitor")
        self.btn_start_live.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; padding: 4px;")

    def update_embedded_logs(self, data):
        content = "=== LIVE TASK MANAGER ===\n\n"
        content += data.get('text_info', '') + "\n"
        content += "----------------------------------------\n"
        content += data.get('full_log_tail', '')
        
        self.txt_live_logs.setPlainText(content)
        sb = self.txt_live_logs.verticalScrollBar()
        sb.setValue(sb.maximum())

    def get_simulation_metadata(self):
        """Helper to grab all current UI parameters for saving."""
        azi = self.manip_azi_spin.value()
        sim_hv = getattr(self, "sim_hv", None)
        photon_energies = (
            np.asarray(sim_hv, dtype=float).tolist()
            if sim_hv is not None
            else [float(self.photon_energy_spin.value())]
        )
        return {
            'crystal': self.ws_combo.currentText(),
            'engine': self.engine_dropdown.currentText(),
            'photon_energy': float(photon_energies[0]),
            'photon_energies': photon_energies,
            'work_function': self.work_function_spin.value(),
            'inner_potential': self.inner_potential_spin.value(),
            'temperature': self.temperature_spin.value(),
            'polarization': self.polarization_combo.currentText(),
            'lin_pol_angle': self.lin_pol_angle_spin.value(),
            'incidence_angle': self.incidence_angle_spin.value(),
            'matrix_element_mode': self.matrix_element_combo.currentText(),
            'manip_theta': self.manip_theta_spin.value(),
            'manip_azimuth': azi,
            'manip_azi': azi,  # dual key during transition
            'manip_tilt': self.manip_tilt_spin.value(),
            'slit_angle': self.slit_angle_spin.value(),
            'hkl': [self.spin_h.value(), self.spin_k.value(), self.spin_l.value()],
            **self._critic_gap_physics_kwargs(),
        }



    @staticmethod
    def _spc_paths_to_results(local_paths, points_json: Optional[str] = None) -> dict:
        """Parse (+stitch) SPR-KKR *_ARPES_data.spc files into the GUI results dict.
        Delegates to the core layer sprkkr_spc_paths_to_results (no Qt here)."""
        return sprkkr_spc_paths_to_results(local_paths, points_json=points_json)

    def load_local_spc(self):
        """Load already-synced SPR-KKR .spc file(s) from disk -- no cluster.
        2026-09-12: Fetch always takes the NEWEST remote arpes_* dir, which can be
        a wrong/aborted run; this lets Sandy open a known-good local result."""
        import os
        start_dir = "scratch/sprkkr_gui_run" if os.path.isdir("scratch/sprkkr_gui_run") else "scratch"
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load SPR-KKR ARPES .spc", start_dir,
            "SPR-KKR spectra (*_ARPES_data.spc *.spc);;All files (*)",
        )
        if not paths:
            return
        empty = [p for p in paths if os.path.getsize(p) == 0]
        if empty:
            QMessageBox.warning(
                self, "Empty .spc",
                "These files are 0 bytes (run aborted before writing?):\n" + "\n".join(empty),
            )
            return

        # Look for pointwise_points.json: files' dir, its parent, its grandparent
        # (per-point subdir case: <run>/arpes/pot_arpes_pNNNN/*.spc), and any
        # immediate subdir (sprkkr_e2e copies the .spc flat into <run>/ while the
        # sidecar stays in <run>/arpes/).
        import glob as _glob
        file_dir = os.path.dirname(paths[0])
        candidates = [
            os.path.join(file_dir, "pointwise_points.json"),
            os.path.join(os.path.dirname(file_dir), "pointwise_points.json"),
            os.path.join(os.path.dirname(os.path.dirname(file_dir)), "pointwise_points.json"),
        ] + sorted(_glob.glob(os.path.join(file_dir, "*", "pointwise_points.json")))
        points_json = next((c for c in candidates if os.path.isfile(c)), None)

        try:
            results = self._spc_paths_to_results(sorted(paths), points_json=points_json)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse .spc:\n{e}")
            return
        label = os.path.basename(os.path.dirname(paths[0])) or paths[0]
        self.on_simulation_finished(True, results, f"Loaded local .spc ({label})")

    def fetch_remote_results(self):
        """Fetch remote ARPES cube (Chinook or SPR-KKR)."""
        return self.fetch_sprkkr_results()

    def fetch_sprkkr_results(self):
        vault_name = self.vault_combo.currentText()
        model_choice = self.engine_dropdown.currentData()

        import json, os, paramiko
        config_file = os.path.expanduser('~/.tensorspec_clusters.json')
        with open(config_file, 'r') as f:
            clusters = json.load(f)
        cluster = self.get_selected_cluster()
        if not cluster: return

        # Chinook / bare / three-step remote → chinook_gui_run npz cube (unchanged).
        # Only SPR-KKR (B3) uses the native .spc fetch path below.
        is_chinook_remote = model_choice != "B3" and self._is_remote_target()

        if is_chinook_remote:
            remote_dir = cp.job_dir(cluster, "chinook")
            target_cube_name = "chinook_arpes_cube.npz"

            try:
                ssh = self._ssh_connect(cluster)

                sftp = ssh.open_sftp()
                os.makedirs("scratch", exist_ok=True)
                local_path = f"scratch/{target_cube_name}"
                sftp.get(f"{remote_dir}/{target_cube_name}", local_path)
                sftp.close()
                ssh.close()

                # Load the file
                data = np.load(local_path, allow_pickle=True)
                if 'cube' in data:
                    results = _remote_cube_results(data)
                else:
                    intensity_3d = data['intensity']
                    results = {
                        'intensity_broadened': intensity_3d,
                        'energy': data['e_axis'] if 'e_axis' in data else None,
                        'theta': data['kx'] if 'kx' in data else None,
                        'phi': data['ky'] if 'ky' in data else None,
                    }

                self.on_simulation_finished(True, results, "Fetched successfully")

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to fetch results:\n{str(e)}\n\n"
                    "If this is a Paramiko key-exchange timeout, retry Fetch "
                    "(slow network / VPN). Connect timeout was increased to 90s banner.",
                )
            return

        # --- SPR-KKR (B3): native *_ARPES_data.spc fetch + parse ---
        if vault_name == "Temporary Scratch Run (sprkkr_gui_run)":
            remote_dir = cp.job_dir(cluster, "sprkkr")
        else:
            vault = global_workspace.get(vault_name)
            remote_dir = vault.get('remote_path') if vault else cp.job_dir(cluster, "sprkkr")

        try:
            import stat as _stat

            ssh = self._ssh_connect(cluster)
            sftp = ssh.open_sftp()

            def _is_dir(attr) -> bool:
                return _stat.S_ISDIR(attr.st_mode)

            # workflow.run_arpes writes into workdir=f"{remote_dir}/arpes_<ts>" —
            # pick the newest such subdir if any exist, else search remote_dir itself.
            try:
                entries = sftp.listdir_attr(remote_dir)
            except IOError:
                entries = []
            run_dirs = sorted(
                (e for e in entries if e.filename.startswith("arpes_") and _is_dir(e)),
                key=lambda e: e.st_mtime,
            )
            search_dir = f"{remote_dir}/{run_dirs[-1].filename}" if run_dirs else remote_dir

            def _find_spc(d):
                found = []
                for e in sftp.listdir_attr(d):
                    if e.filename.endswith("_ARPES_data.spc"):
                        found.append(f"{d}/{e.filename}")
                    elif _is_dir(e):
                        found.extend(_find_spc(f"{d}/{e.filename}"))
                return found

            spc_remote_paths = sorted(_find_spc(search_dir))
            if not spc_remote_paths:
                raise FileNotFoundError(f"No *_ARPES_data.spc found under {search_dir}")

            local_dir = os.path.join("scratch", "sprkkr_gui_fetch")
            os.makedirs(local_dir, exist_ok=True)
            local_paths = []
            for i, rp in enumerate(spc_remote_paths):
                lp = os.path.join(local_dir, f"{i}_{os.path.basename(rp)}")
                sftp.get(rp, lp)
                local_paths.append(lp)

            # Try to fetch pointwise_points.json if it exists
            points_json_local = None
            try:
                points_json_remote = f"{search_dir}/pointwise_points.json"
                points_json_local = os.path.join(local_dir, "pointwise_points.json")
                sftp.get(points_json_remote, points_json_local)
            except IOError:
                points_json_local = None

            sftp.close()
            ssh.close()

            results = self._spc_paths_to_results(local_paths, points_json=points_json_local)
            self.on_simulation_finished(True, results, "Fetched successfully")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to fetch SPR-KKR results:\n{str(e)}\n\n"
                "If this is a Paramiko key-exchange timeout, retry Fetch "
                "(slow network / VPN). Connect timeout was increased to 90s banner.",
            )

    def push_arpes_to_workspace(self):
        if not hasattr(self, 'sim_intensity'): return
        
        name, ok = QInputDialog.getText(self, "Push to Workspace", "Enter dataset name (e.g., WTe2_75eV_CR):")
        if ok and name:
            try:
                if self.sim_hv is not None and self.sim_intensity.ndim == 4:
                    sim_tensor = tensor_from_stacked_sim(
                        self.sim_intensity,
                        self.sim_hv,
                        self.sim_kx,
                        self.sim_ky,
                        self.sim_E_axis,
                        metadata=self.get_simulation_metadata(),
                    )
                else:
                    # Transpose dimensions for the viewer (kx, ky, E) -> (E, kx, ky)
                    tensor_value = np.transpose(self.sim_intensity, (2, 0, 1))
                    sim_tensor = TensorData(
                        value=tensor_value,
                        axes=[self.sim_E_axis, self.sim_kx, self.sim_ky],
                        labels=["Energy", "Θ (Slit)", "Φ (Deflect)"],
                        units=["eV", "deg", "deg"],
                        data_type="Simulated ARPES Matrix Elements",
                        metadata=self.get_simulation_metadata()
                    )
                
                global_workspace.push_spectroscopy_data(name, sim_tensor)
                QMessageBox.information(self, "Success", f"Data pushed to Workspace as '{name}'.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to push data:\n{e}")

    def save_arpes_to_disk(self):
        if not hasattr(self, 'sim_intensity'): return
        
        name, ok = QInputDialog.getText(self, "Save to Disk", "Enter file name (e.g., WTe2_75eV_CR):")
        if ok and name:
            try:
                global_workspace.save_simulated_arpes(
                    name, 
                    self.sim_intensity, 
                    self.sim_kx, 
                    self.sim_ky, 
                    self.sim_E_axis, 
                    self.get_simulation_metadata(),
                    hv=self.sim_hv,
                )
                QMessageBox.information(self, "Success", f"Data safely saved to disk as '{name}.npz'.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save data:\n{e}")
    

    def _try_fill_h_dict_from_w90_cache(self, band_data: dict) -> bool:
        """Load H_dict from local W90 TB cache into band_data. Returns True on success."""
        w90 = band_data.get("w90_filepath") or band_data.get("w90_file") or ""
        if not w90:
            return False
        from tensorspec.core.dft.w90_tb_cache import load_parsed_tb

        use_soc = bool(band_data.get("use_soc", False))
        onsite_e = float(band_data.get("onsite_e", 0.0) or 0.0)
        hop_tol = float(band_data.get("hop_tol", 1e-6) or 1e-6)
        qe_fermi = float(band_data.get("fermi_energy", 0.0) or 0.0)
        disk = load_parsed_tb(
            w90, use_soc, onsite_e, hop_tol=hop_tol, qe_fermi=qe_fermi
        )
        if not disk:
            print(
                f"[ARPES] No local W90 cache for refill ({w90})",
                flush=True,
            )
            return False
        tb_dict, _basis_args, a_qe = disk
        n_hop = len(tb_dict.get("list") or tb_dict.get("H") or [])
        if n_hop == 0:
            return False
        band_data["H_dict"] = tb_dict
        if a_qe is not None and not band_data.get("a_mat"):
            band_data["a_mat"] = a_qe
        print(f"[ARPES] Refilled H_dict from W90 cache ({n_hop} hoppings)", flush=True)
        return True

    def _pack_basis_list_for_remote(self, band_data: dict) -> list:
        """Serialize orbital basis for tb_data.npz (handles gen_basis dict + TB_model)."""
        basis = band_data.get("basis")
        tb_model = band_data.get("tb_model")

        if isinstance(basis, dict):
            if "bulk" in basis:
                basis = basis["bulk"]
            elif "orbitals" in basis:
                basis = basis["orbitals"]

        if not isinstance(basis, (list, tuple)) or not basis:
            if tb_model is not None and getattr(tb_model, "basis", None) is not None:
                basis = list(tb_model.basis)

        basis_list = []
        if isinstance(basis, (list, tuple)):
            for b in basis:
                if hasattr(b, "pos"):
                    basis_list.append(
                        {
                            "pos": np.asarray(b.pos, dtype=float).tolist(),
                            "label": getattr(b, "label", "10"),
                            "spin": 1.0,
                            "Z": int(getattr(b, "Z", 1) or 1),
                        }
                    )
        return basis_list

    def _is_remote_target(self) -> bool:
        from tensorspec.gui.services.compute_mode import is_hybrid_mode

        return is_hybrid_mode(self.target_dropdown)

    def _set_row_visible(self, widget, visible: bool):
        """Hide/show ``widget`` and its QFormLayout label (if any) together.

        Checked against both self.param_layout and self.beam_layout since the
        Chinook-only knobs this feeds (_sync_sprkkr_irrelevant_knobs) live in
        either group. A single-widget addRow() (e.g. a bare QCheckBox row)
        has no separate label -- labelForField returns None, so we just
        toggle the widget itself.
        """
        for layout in (getattr(self, "param_layout", None), getattr(self, "beam_layout", None)):
            if layout is None:
                continue
            label = layout.labelForField(widget)
            if label is not None:
                label.setVisible(visible)
        widget.setVisible(visible)

    def _sync_sprkkr_irrelevant_knobs(self, is_b3: bool):
        """Hide Chinook-only physics knobs that SPR-KKR (B3) never reads.

        None of these reach kkr_wrapper.py / params.py / workflow.py -- SPR-
        KKR's one-step method has no Fresnel transmission step, no separate
        escape-depth (k_z) broadening, no dipole mean-free-path, and encodes
        sample/surface orientation directly via MILLER_HKL + CRYS_VECS rather
        than the lab-frame manipulator knobs (which have been moved to pointwise
        angle points via sample_to_bulk_frame in the core layer). Leaving them
        visible+editable under B3 implied they mattered when they silently did nothing.
        Beam Incidence stays visible for B3 -- it's wired to SPEC_PH THETA.
        Manipulator theta/azimuth/tilt now reach SPR-KKR via pointwise angle points.
        """
        chinook_only = [
            self.inner_potential_spin,
            self.rad_type_combo,
            self.mfp_spin,
            self.fresnel_enabled_chk,
            self.optical_n_spin,
            self.optical_k_spin,
            self.include_photon_momentum_chk,
            self.kz_halfwidth_spin,
            self.kz_npoints_spin,
        ]
        for widget in chinook_only:
            self._set_row_visible(widget, not is_b3)
        if is_b3:
            self._set_row_visible(self.lin_pol_angle_spin, False)
        else:
            # Restore the pre-existing rule: only shown for Arbitrary pol.
            self._set_row_visible(
                self.lin_pol_angle_spin, "Arbitrary" in self.polarization_combo.currentText()
            )

    def _sync_remote_ui(self):
        """Show/enable live-monitor + fetch from Physics Model × Compute Target.

        Rules:
        - Local + (B1/B2/A): hide remote panel; fetch off.
        - Remote + (B1/B2/A): show panel; fetch on → chinook_gui_run cube.
        - B3 (SPR-KKR): always show panel + vault; fetch on (vault / sprkkr path).
        Fetch must NOT require a fresh submit this session (finished jobs on remote cluster).
        """
        model = self.engine_dropdown.currentData()
        remote = self._is_remote_target()
        self._sync_sprkkr_irrelevant_knobs(model == "B3")

        if model == "B3":
            self.ws_group.hide()
            self.vault_group.show()
            self.sprkkr_group.show()
            self.remote_tb_opts.hide()
            self.matrix_element_combo.hide()
            self.live_log_widget.show()
            self.btn_fetch_results.setEnabled(True)
            self.btn_start_live.setEnabled(True)
        else:
            self.ws_group.show()
            self.vault_group.hide()
            self.sprkkr_group.hide()
            self.matrix_element_combo.show()
            if remote:
                self.remote_tb_opts.show()
                self.live_log_widget.show()
                self.btn_fetch_results.setEnabled(True)
                self.btn_start_live.setEnabled(True)
            else:
                self.remote_tb_opts.hide()
                self.live_log_widget.hide()
                self.btn_fetch_results.setEnabled(False)
                self.btn_start_live.setEnabled(False)
        self.update_hv_mode_ui()
        self._sync_remote_gpu_ui()
        if remote:
            self._refresh_remote_gpu_count()

    def _probe_remote_cuda_gpus(self) -> tuple[Optional[int], Optional[int]]:
        """Return (total_gpus, gpus_with_headroom) on selected remote host."""
        cluster = self.get_selected_cluster()
        if not cluster:
            return None, None
        min_free_mib = 10 * 1024  # ~10 GiB ARPES worker headroom guideline
        try:
            ssh = self._ssh_connect(cluster)
            _, stdout, _ = ssh.exec_command(
                "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null",
                timeout=15,
            )
            lines = [ln.strip() for ln in stdout.read().decode().splitlines() if ln.strip()]
            ssh.close()
            if not lines:
                return None, None
            free_vals = [int(float(x)) for x in lines]
            total = len(free_vals)
            safe = sum(1 for f in free_vals if f >= min_free_mib)
            return total, safe
        except Exception:
            return None, None

    def _probe_remote_cuda_gpu_count(self) -> Optional[int]:
        total, _ = self._probe_remote_cuda_gpus()
        return total

    def _refresh_remote_gpu_count(self):
        """Set GPU spin max from remote nvidia-smi (any cluster)."""
        if not self._is_remote_target():
            self.lbl_remote_gpu_limits.setText("host max: —")
            return
        total, safe = self._probe_remote_cuda_gpus()
        self._remote_gpu_host_max = total
        self._remote_gpu_safe_max = safe
        prev = self.remote_ngpus_spin.value()
        if total is not None:
            self.remote_ngpus_spin.setMaximum(max(1, total))
            if prev > total:
                self.remote_ngpus_spin.setValue(total)
            safe_txt = f", {safe} with headroom" if safe is not None else ""
            self.lbl_remote_gpu_limits.setText(f"host max: {total}{safe_txt}")
        else:
            self.remote_ngpus_spin.setMaximum(8)
            self.lbl_remote_gpu_limits.setText("host max: unknown")

    def _resolve_ngpus_for_submit(self, me_device: str, me_layout: str) -> int:
        """Clamp manual GPU request to probed host limits before dispatch."""
        if me_device != "cuda" or me_layout not in ("full", "auto"):
            return 1
        requested = int(self.remote_ngpus_spin.value())
        total, safe = self._probe_remote_cuda_gpus()
        if total is None:
            return max(1, min(requested, self.remote_ngpus_spin.maximum()))
        effective = max(1, min(requested, total))
        if effective != requested:
            QMessageBox.warning(
                self,
                "GPU count capped",
                f"Requested {requested} GPU(s) but host has {total}. Using {effective}.",
            )
            self.remote_ngpus_spin.setValue(effective)
        elif requested > 1 and safe is not None and safe < requested:
            QMessageBox.information(
                self,
                "GPU headroom",
                f"{safe} of {total} GPU(s) have ≥10 GiB free. "
                f"Job may still run but cluster can clamp further if VRAM is tight.",
            )
        return effective

    def _sync_remote_gpu_ui(self):
        """CUDA GPU count applies only to full-cube Grizzly layout."""
        cuda = self.combo_remote_device.currentData() == "cuda"
        layout = self.combo_remote_layout.currentData()
        fullish = layout in ("full", "auto")
        enabled = cuda and fullish
        self.remote_ngpus_spin.setEnabled(enabled)
        self.btn_refresh_remote_gpus.setEnabled(enabled)
        self.lbl_remote_gpu_limits.setEnabled(enabled)
        if not enabled:
            self.remote_ngpus_spin.setToolTip(
                "GPU count applies to full-cube CUDA Grizzly runs only."
            )
        else:
            self.remote_ngpus_spin.setToolTip(
                "Type any count from 1 up to host max. Capped again at submit and on the cluster."
            )

    def on_target_changed(self):
        self._sync_remote_ui()

    def on_engine_changed(self):
        self._sync_remote_ui()
        # Beam Incidence now feeds SPEC_PH THETA for B3 too, and hiding the
        # Chinook-only manipulator/Fresnel rows changes what's on screen --
        # redraw so the schematic never shows a stale geometry after a
        # engine switch.
        self.update_schematic()

    def _update_kkr_bin_status(self):
        """Refresh the 'which SPR-KKR binaries were found' status label."""
        bin_dir = self.sprkkr_bin_edit.text().strip() or "."
        try:
            bins = sprkkr_local_binaries_present(bin_dir)
        except Exception:
            bins = {}
        found = [k for k, v in bins.items() if v]
        if found:
            self.lbl_kkr_bins.setText("found: " + ", ".join(found))
        else:
            self.lbl_kkr_bins.setText("no kkrscf/kkrspec binaries found in this dir")

    def _lattice_matrix_from_obj(self, obj):
        """Extract 3x3 cartesian lattice matrix (Angstrom) from Structure/Lattice-like."""
        if obj is None:
            return None
        lattice = getattr(obj, "lattice", obj)
        matrix = getattr(lattice, "matrix", None)
        if matrix is None:
            return None
        try:
            return [list(map(float, row)) for row in matrix]
        except Exception:
            return None

    def _lookup_cif_lattice(self):
        """Vault meta cif_lattice, else any crystal_structure in workspace.

        NOTE: do NOT use ``ws_combo`` here — that lists band structures, not CIFs.
        """
        vault_name = self.vault_combo.currentText()
        vault_entry = global_workspace.get(vault_name) if vault_name else None
        if vault_entry and vault_entry.get("meta", {}).get("cif_lattice"):
            return vault_entry["meta"]["cif_lattice"]

        # Crystal Suite pushes pymatgen Structure under type=crystal_structure.
        for name in global_workspace.list_crystal_structures():
            crystal = global_workspace.pull_crystal_structure(name)
            lat = self._lattice_matrix_from_obj(crystal)
            if lat is not None:
                return lat

        # Band push sometimes embeds a structure — last resort only.
        band_name = self.ws_combo.currentText()
        band = global_workspace.pull_band_structure(band_name) if band_name else None
        if isinstance(band, dict):
            for key in ("structure", "crystal", "pymatgen_structure"):
                lat = self._lattice_matrix_from_obj(band.get(key))
                if lat is not None:
                    return lat
        return None

    def _resolve_sprkkr_hkl_frame(self, frame_choice: str):
        """Return (hkl_frame, cif_lattice|None) for B3 kwargs.

        frame_choice: 'auto' | 'cif' | 'abas'
        Raises ValueError when CIF is required but no lattice is available.
        """
        choice = (frame_choice or "auto").lower()
        if choice == "abas":
            return "abas", None

        cif_lattice = self._lookup_cif_lattice()
        if choice == "cif":
            if cif_lattice is None:
                crystals = global_workspace.list_crystal_structures()
                raise ValueError(
                    "HKL frame is CIF, but no cif_lattice on this vault "
                    "and no usable crystal Structure in workspace "
                    f"(crystal entries: {crystals or 'none'}). "
                    "Load a CIF in Crystal Suite, or switch HKL frame to ABAS "
                    "and type the pot-frame Miller index (for this VTe₂ pot, "
                    "CIF (−2,0,1) ≡ ABAS (0,1,−1))."
                )
            return "cif", cif_lattice

        # auto
        if cif_lattice is not None:
            return "cif", cif_lattice
        QMessageBox.warning(
            self,
            "No CIF lattice",
            "No CIF lattice for this vault; treating h k l as ABAS (pot) frame.\n"
            "Or set HKL frame to ABAS explicitly to skip this warning.",
        )
        return "abas", None

    def _browse_sprkkr_pot(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select converged potential", "", "SPR-KKR potential (*.pot_new);;All files (*)"
        )
        if path:
            self.sprkkr_pot_edit.setText(path)

    def _update_kkr_eta(self):
        """SPR-KKR B3 ETA label from grid size (NE, NT, NP) × nproc."""
        import os as _os

        try:
            store_path = _os.path.expanduser("~/.tensorspec_sprkkr_eta.json")
            model = SprkkrEtaModel(store_path=store_path)
            model.load()
            e_steps_eff = 1 if self.spin_e_min.value() == self.spin_e_max.value() else self.spin_e_steps.value()
            kx_steps_eff = 1 if self.spin_kx_min.value() == self.spin_kx_max.value() else self.spin_kx_steps.value()
            ky_steps_eff = 1 if self.spin_ky_min.value() == self.spin_ky_max.value() else self.spin_ky_steps.value()
            params = SprkkrArpesParams(
                ne=e_steps_eff,
                nt=kx_steps_eff,
                np_=ky_steps_eff,
            )
            seconds = model.estimate_seconds(params, self.spin_kkr_nproc.value())
            self.lbl_kkr_eta.setText(f"ETA: {sprkkr_format_eta(seconds)}")
        except Exception:
            self.lbl_kkr_eta.setText("ETA: —")

    def _kill_sprkkr_jobs(self):
        """Force-kill any running kkrspec9.7/kkrscf9.7 processes (local or remote target)."""
        reply = QMessageBox.question(
            self,
            "Kill SPR-KKR jobs",
            "Kill all running kkrspec9.7 / kkrscf9.7 processes on the "
            "current compute target?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            if self._is_remote_target():
                cluster = self.get_selected_cluster()
                if not cluster:
                    QMessageBox.critical(self, "Error", "No remote cluster found in configuration!")
                    return
                ssh = self._ssh_connect(cluster)
                try:
                    _, stdout, _ = ssh.exec_command(
                        "pkill -f kkrspec9.7; pkill -f kkrscf9.7; sleep 1; "
                        "pgrep -fl 'kkrspec9.7|kkrscf9.7' || echo none"
                    )
                    out = stdout.read().decode(errors="ignore")
                finally:
                    ssh.close()
                QMessageBox.information(self, "Kill SPR-KKR jobs", f"Remote result:\n{out}")
            else:
                import subprocess
                subprocess.run(["pkill", "-f", "kkrspec9.7"])
                subprocess.run(["pkill", "-f", "kkrscf9.7"])
                QMessageBox.information(self, "Kill SPR-KKR jobs", "Sent pkill for kkrspec9.7 / kkrscf9.7 locally.")
            self.run_sim_btn.setEnabled(True)
            self.run_sim_btn.setText("🚀 Run ARPES Simulation")
            self.run_sim_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #2b5c8f; color: white;")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to kill SPR-KKR jobs:\n{str(e)}")


    def _update_band_energy_meta_label(self):
        name = self.ws_combo.currentText()
        if not name or name == "No band structures found":
            self.lbl_band_energy_meta.setText("onsite / EF: (select a pushed band structure)")
            return
        band = global_workspace.pull_band_structure(name)
        if not band:
            self.lbl_band_energy_meta.setText("onsite / EF: (not found in workspace)")
            return
        onsite = band.get('onsite_e', 'MISSING')
        ef = band.get('fermi_energy', band.get('e_fermi', 'MISSING'))
        shift = band.get('arpes_e_fermi_shift', ef)
        baked = band.get('h_includes_onsite', False)
        self.lbl_band_energy_meta.setText(
            f"From DFT push: onsite_e={onsite} eV | QE Fermi={ef} eV | "
            f"ARPES E-shift={shift} eV | onsite baked in H={baked}"
        )

    def refresh_workspace(self):
        bands = global_workspace.list_band_structures()
        self.ws_combo.blockSignals(True)
        self.ws_combo.clear()
        if not bands:
            self.ws_combo.addItem("No band structures found")
        else:
            self.ws_combo.addItems(bands)
        self.ws_combo.blockSignals(False)
        self._update_band_energy_meta_label()

        # Pull SPR-KKR vault dirs from the remote cluster into workspace memory
        # so Refresh shows VTe2_* even after a GUI restart.
        if self._is_remote_target():
            try:
                from tensorspec.gui.components.sprkkr_panels import (
                    discover_remote_vault_dirs,
                )

                cluster = self.get_selected_cluster()
                if cluster:
                    for name, path in discover_remote_vault_dirs(cluster):
                        existing = global_workspace.get(name)
                        if existing and existing.get("type") == "remote_run":
                            continue
                        global_workspace.push_remote_run(
                            name=name,
                            cluster_name=cluster_display_name(cluster),
                            engine="SPRKKR",
                            remote_path=path,
                            meta={},
                        )
            except Exception as exc:
                print(f"[ARPES] remote vault discover failed: {exc}", flush=True)

        vaults = global_workspace.list_remote_runs(engine="SPRKKR")
        self.vault_combo.clear()
        
        # Always add the Scratch Run as the default!
        self.vault_combo.addItem("Temporary Scratch Run (sprkkr_gui_run)")
        
        if vaults:
            self.vault_combo.addItems(vaults)
            
    def delete_vault(self):
        vault_name = self.vault_combo.currentText()
        if not vault_name or "Temporary Scratch Run" in vault_name or vault_name == "No SPRKKR Vaults Found":
            QMessageBox.warning(self, "Invalid", "You cannot delete the temporary scratch run or an empty vault.")
            return
            
        reply = QMessageBox.question(
            self,
            'Delete Vault',
            f"Are you sure you want to delete '{vault_name}' from the Workspace?\n\n"
            "(This will also SSH into the remote cluster and delete the remote folder!)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        
        if reply == QMessageBox.Yes:
            vault = global_workspace.get(vault_name)
            remote_dir = vault.get('remote_path')
            
            # SSH and delete
            try:
                import json, paramiko, os
                cluster = self.get_selected_cluster()
                if cluster:
                    ssh = paramiko.SSHClient()
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    pwd = cluster.get('password', '') or None
                    ssh.connect(cluster['host'], port=cluster.get('port', 22), username=cluster['user'], password=pwd, timeout=10)
                    
                    cmd = f"rm -rf {remote_dir}"
                    ssh.exec_command(cmd)
                    ssh.close()
            except Exception as e:
                QMessageBox.warning(self, "SSH Warning", f"Removed from workspace, but failed to delete remote folder on cluster: {e}")
                
            global_workspace.remove(vault_name)
            self.refresh_workspace()
            QMessageBox.information(self, "Deleted", f"Vault '{vault_name}' has been deleted!")
    
    def _draw_bz_overlay(self):
        """Fetches the 3D Brillouin Zone, projects it to the hkl surface, and rotates it onto the (kx, ky) plot axes."""
        target_crystal = self.ws_combo.currentText()
        band_data = global_workspace.pull_band_structure(target_crystal)
        
        # Ensure the DFT suite provided the structure and reciprocal matrix
        if not band_data or 'structure' not in band_data or 'recip_matrix' not in band_data:
            return

        structure = band_data['structure']
        recip_matrix = band_data['recip_matrix']
        
        from tensorspec.core.crystallography import CrystalEngine
        import numpy as np
        
        # 1. Calculate the 3D Bulk Brillouin Zone vertices
        bz_data = CrystalEngine.calculate_brillouin_zone(structure)
        if not bz_data: return
        
        # 2. Recreate the exact hkl -> Sample Frame rotation matrix used by Chinook
        hkl = (self.spin_h.value(), self.spin_k.value(), self.spin_l.value())
        Z_surf, Y_surf = CrystalEngine.get_hkl_surface_frame(hkl, recip_matrix, azimuthal_ref=None)
        X_surf = np.cross(Y_surf, Z_surf)
        
        # R_bulk_to_hkl projects 3D Bulk vectors into the (kx, ky, kz) sample frame
        R_hkl_to_bulk = np.column_stack((X_surf, Y_surf, Z_surf))
        R_bulk_to_hkl = np.linalg.inv(R_hkl_to_bulk)
        
        # 3. Calculate the 2D projected boundary silhouette
        scaled_points = np.array(bz_data["points"])
        surf_data = CrystalEngine.calculate_surface_projection(scaled_points, structure, hkl[0], hkl[1], hkl[2])
        
        if surf_data:
            # The silhouette vertices are returned in the 3D Bulk Cartesian frame
            proj_bounds_bulk = np.array(surf_data["projected_bounds"])
            
            # Rotate the vertices into the ARPES (kx, ky) Frame
            bounds_sample = proj_bounds_bulk @ R_bulk_to_hkl.T
            
            kx_poly = bounds_sample[:, 0]
            ky_poly = bounds_sample[:, 1]
            
            # Close the polygon loop so the drawn line connects the last vertex back to the first
            kx_poly = np.append(kx_poly, kx_poly[0])
            ky_poly = np.append(ky_poly, ky_poly[0])
            
            # Overlay on the plot
            self.ax.plot(kx_poly, ky_poly, color='cyan', linestyle='-', linewidth=1.5, label="Projected Surface BZ")
            self.ax.legend(loc='upper right', fontsize=8, facecolor='black', edgecolor='white', labelcolor='white')