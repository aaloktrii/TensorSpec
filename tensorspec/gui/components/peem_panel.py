"""PEEM analysis panel — direct core access via PeemService."""
from __future__ import annotations

import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tensorspec.gui.components.data_viewer_panel import DataViewerPanel
from tensorspec.gui.services.peem_service import PeemService


class PeemLoadThread(QThread):
    finished_signal = Signal(bool, object, str)

    def __init__(
        self,
        service: PeemService,
        path=None,
        name=None,
        csv_path=None,
        *,
        beamline_paths=None,
    ):
        super().__init__()
        self.service = service
        self.path = path
        self.name = name
        self.csv_path = csv_path
        self.beamline_paths = beamline_paths

    def run(self):
        try:
            if self.beamline_paths is not None:
                summary = self.service.load_beamline(self.beamline_paths, name=self.name)
            else:
                summary = self.service.load_path(
                    self.path, name=self.name, csv_path=self.csv_path
                )
            self.finished_signal.emit(True, summary, "Loaded")
        except Exception as exc:
            self.finished_signal.emit(False, None, str(exc))


class PeemPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.service = PeemService()
        self._dataset_name: str | None = None
        self._meta: dict | None = None
        self._load_thread: PeemLoadThread | None = None
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        # --- LEFT: controls (forms only — plots live on right tabs) ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        load_group = QGroupBox("1. Load")
        load_form = QFormLayout(load_group)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Dataset name (auto from path)")
        load_form.addRow("Name:", self.name_edit)
        btn_row = QHBoxLayout()
        self.btn_load_stack = QPushButton("Load TIF Stack")
        self.btn_load_stack.clicked.connect(self._load_tif_stack)
        self.btn_load_folder = QPushButton("Load Folder")
        self.btn_load_folder.clicked.connect(self._load_folder)
        btn_row.addWidget(self.btn_load_stack)
        btn_row.addWidget(self.btn_load_folder)
        load_form.addRow(btn_row)
        self.btn_load_beamline = QPushButton("Load beamline…")
        self.btn_load_beamline.setToolTip(
            "UE49 scan .txt, UView .dat, or ALS .P3B (files or folders).\n"
            "UView CP/CM: pick CL folder, say Yes to add CR (or LH/LV).\n"
            "Then Pair / Separate → Drift → Make PCD/PLD (Frame) → BG → Sum Rule."
        )
        self.btn_load_beamline.clicked.connect(self._load_beamline)
        load_form.addRow(self.btn_load_beamline)
        self.btn_attach_csv = QPushButton("Attach CSV")
        self.btn_attach_csv.clicked.connect(self._attach_csv)
        load_form.addRow(self.btn_attach_csv)
        self.status_label = QLabel("No PEEM data loaded")
        self.status_label.setWordWrap(True)
        load_form.addRow(self.status_label)
        control_layout.addWidget(load_group)

        pair_group = QGroupBox("2. Pair / Separate")
        pair_form = QFormLayout(pair_group)
        self.pair_mode = QComboBox()
        self.pair_mode.addItem("Auto", "auto")
        self.pair_mode.addItem("CP / CM", "CP_CM")
        self.pair_mode.addItem("LH / LV", "LH_LV")
        pair_form.addRow("Pair mode:", self.pair_mode)
        pair_btns = QHBoxLayout()
        self.btn_pair = QPushButton("Stack Pairs")
        self.btn_pair.clicked.connect(self._pair)
        self.btn_separate = QPushButton("Separate")
        self.btn_separate.clicked.connect(self._separate)
        pair_btns.addWidget(self.btn_pair)
        pair_btns.addWidget(self.btn_separate)
        pair_form.addRow(pair_btns)
        control_layout.addWidget(pair_group)

        drift_group = QGroupBox("3. Drift")
        drift_form = QFormLayout(drift_group)
        self.drift_algo_label = QLabel("NCC ROI (ncc_roi)")
        drift_form.addRow("Algorithm:", self.drift_algo_label)
        self.drift_source = QComboBox()
        self.drift_source.addItem("Raw", "raw")
        self.drift_source.addItem("Processed", "processed")
        drift_form.addRow("Source:", self.drift_source)
        self.drift_ref = QSpinBox()
        self.drift_ref.setRange(0, 9999)
        drift_form.addRow("Ref frame:", self.drift_ref)
        self.drift_radius = QSpinBox()
        self.drift_radius.setRange(1, 200)
        self.drift_radius.setValue(10)
        drift_form.addRow("Search radius:", self.drift_radius)
        self.btn_drift = QPushButton("Apply Drift")
        self.btn_drift.clicked.connect(self._apply_drift)
        drift_form.addRow(self.btn_drift)
        control_layout.addWidget(drift_group)

        bg_group = QGroupBox("4. Background")
        bg_form = QFormLayout(bg_group)
        self.bg_node = QComboBox()
        self.bg_node.addItem("Raw", "raw")
        self.bg_node.addItem("Processed", "processed")
        bg_form.addRow("Source node:", self.bg_node)
        self.bg_method = QComboBox()
        self.bg_method.addItem("Linear pre-edge", "linear")
        self.bg_method.addItem("Two-step pre/post", "two_step")
        self.bg_method.currentIndexChanged.connect(self._sync_bg_method_ui)
        bg_form.addRow("Method:", self.bg_method)
        self.bg_e0 = QDoubleSpinBox()
        self.bg_e1 = QDoubleSpinBox()
        self.bg_post_e0 = QDoubleSpinBox()
        self.bg_post_e1 = QDoubleSpinBox()
        for spin in (self.bg_e0, self.bg_e1, self.bg_post_e0, self.bg_post_e1):
            spin.setRange(-1e4, 1e4)
            spin.setDecimals(3)
        self.bg_e0.setValue(0.0)
        self.bg_e1.setValue(2.0)
        self.bg_post_e0.setValue(3.0)
        self.bg_post_e1.setValue(4.0)
        bg_form.addRow("Pre e0 / e1:", self._pair_row(self.bg_e0, self.bg_e1))
        self.bg_post_row_label = QLabel("Post e0 / e1:")
        self.bg_post_row = self._pair_row(self.bg_post_e0, self.bg_post_e1)
        bg_form.addRow(self.bg_post_row_label, self.bg_post_row)
        bg_btns = QHBoxLayout()
        self.btn_bg_preview = QPushButton("Preview BG")
        self.btn_bg_preview.clicked.connect(self._bg_preview)
        self.btn_bg_apply = QPushButton("Apply BG")
        self.btn_bg_apply.clicked.connect(self._bg_apply)
        bg_btns.addWidget(self.btn_bg_preview)
        bg_btns.addWidget(self.btn_bg_apply)
        bg_form.addRow(bg_btns)
        control_layout.addWidget(bg_group)
        self._sync_bg_method_ui()

        map_group = QGroupBox("5. Dichroism map (Frame viewer)")
        map_form = QFormLayout(map_group)
        self.map_register = QComboBox()
        self.map_register.addItem("Register helicities (recommended)", True)
        self.map_register.addItem("No extra registration", False)
        map_form.addRow("Align:", self.map_register)
        map_btns = QHBoxLayout()
        self.btn_make_pcd = QPushButton("Make PCD (XMCD)")
        self.btn_make_pcd.setToolTip(
            "CP/CM → /analysis/pcd. Prefer Separate first, or beamline load with pol tags."
        )
        self.btn_make_pcd.clicked.connect(self._make_pcd)
        self.btn_make_pld = QPushButton("Make PLD (XLD)")
        self.btn_make_pld.clicked.connect(self._make_pld)
        map_btns.addWidget(self.btn_make_pcd)
        map_btns.addWidget(self.btn_make_pld)
        map_form.addRow(map_btns)
        self.btn_export_map = QPushButton("Export current map…")
        self.btn_export_map.clicked.connect(self._export_map)
        map_form.addRow(self.btn_export_map)
        control_layout.addWidget(map_group)

        sum_group = QGroupBox("6. XMCD Sum Rule")
        sum_form = QFormLayout(sum_group)
        self.sum_nh = QDoubleSpinBox()
        self.sum_nh.setRange(0.0, 20.0)
        self.sum_nh.setValue(1.0)
        self.sum_nh.setDecimals(2)
        sum_form.addRow("n_h:", self.sum_nh)
        self.sum_l3_lo = QDoubleSpinBox()
        self.sum_l3_hi = QDoubleSpinBox()
        self.sum_l2_lo = QDoubleSpinBox()
        self.sum_l2_hi = QDoubleSpinBox()
        self.sum_r_lo = QDoubleSpinBox()
        self.sum_r_hi = QDoubleSpinBox()
        for spin in (
            self.sum_l3_lo,
            self.sum_l3_hi,
            self.sum_l2_lo,
            self.sum_l2_hi,
            self.sum_r_lo,
            self.sum_r_hi,
        ):
            spin.setRange(-1e4, 1e4)
            spin.setDecimals(3)
        self.sum_l3_lo.setValue(0.0)
        self.sum_l3_hi.setValue(1.0)
        self.sum_l2_lo.setValue(2.0)
        self.sum_l2_hi.setValue(3.0)
        self.sum_r_lo.setValue(0.0)
        self.sum_r_hi.setValue(4.0)
        sum_form.addRow("L3 lo/hi:", self._pair_row(self.sum_l3_lo, self.sum_l3_hi))
        sum_form.addRow("L2 lo/hi:", self._pair_row(self.sum_l2_lo, self.sum_l2_hi))
        sum_form.addRow("R lo/hi:", self._pair_row(self.sum_r_lo, self.sum_r_hi))
        sum_btns = QHBoxLayout()
        self.btn_sum_preview = QPushButton("Preview Sum Rule")
        self.btn_sum_preview.clicked.connect(self._sumrule_preview)
        self.btn_sum_apply = QPushButton("Apply Sum Rule")
        self.btn_sum_apply.clicked.connect(self._sumrule_apply)
        sum_btns.addWidget(self.btn_sum_preview)
        sum_btns.addWidget(self.btn_sum_apply)
        sum_form.addRow(sum_btns)
        self.sum_results = QLabel("(no sum-rule result yet)")
        self.sum_results.setWordWrap(True)
        sum_form.addRow(self.sum_results)
        control_layout.addWidget(sum_group)

        control_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(control_panel)
        scroll.setMinimumWidth(360)
        splitter.addWidget(scroll)

        # --- RIGHT: Frame (DataViewer) | BG | Sum Rule ---
        self.view_tabs = QTabWidget()
        self.view_tabs.setDocumentMode(True)

        view_panel = QWidget()
        view_layout = QVBoxLayout(view_panel)
        view_layout.setContentsMargins(4, 4, 4, 4)
        nav = QHBoxLayout()
        self.node_combo = QComboBox()
        self.node_combo.currentIndexChanged.connect(self._on_node_changed)
        nav.addWidget(QLabel("Show:"))
        nav.addWidget(self.node_combo, stretch=1)
        self.btn_refresh_view = QPushButton("Reload")
        self.btn_refresh_view.clicked.connect(self._refresh_frame_view)
        nav.addWidget(self.btn_refresh_view)
        view_layout.addLayout(nav)
        self.frame_viewer = DataViewerPanel(compact=True)
        view_layout.addWidget(self.frame_viewer, stretch=1)
        self.view_tabs.addTab(view_panel, "Frame")

        bg_page = QWidget()
        bg_page_layout = QVBoxLayout(bg_page)
        bg_page_layout.setContentsMargins(4, 4, 4, 4)
        self.bg_figure = Figure(figsize=(6, 4), dpi=100, layout="tight")
        self.bg_canvas = FigureCanvas(self.bg_figure)
        self.bg_ax = self.bg_figure.add_subplot(111)
        bg_page_layout.addWidget(self.bg_canvas, stretch=1)
        self.view_tabs.addTab(bg_page, "BG")

        sum_page = QWidget()
        sum_page_layout = QVBoxLayout(sum_page)
        sum_page_layout.setContentsMargins(4, 4, 4, 4)
        self.sum_figure = Figure(figsize=(6, 4), dpi=100, layout="tight")
        self.sum_canvas = FigureCanvas(self.sum_figure)
        self.sum_ax = self.sum_figure.add_subplot(111)
        sum_page_layout.addWidget(self.sum_canvas, stretch=1)
        self.view_tabs.addTab(sum_page, "Sum Rule")

        splitter.addWidget(self.view_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 720])

    @staticmethod
    def _pair_row(a, b):
        row = QHBoxLayout()
        row.addWidget(a)
        row.addWidget(b)
        wrap = QWidget()
        wrap.setLayout(row)
        return wrap

    def _sync_bg_method_ui(self):
        two_step = self.bg_method.currentData() == "two_step"
        self.bg_post_row_label.setVisible(two_step)
        self.bg_post_row.setVisible(two_step)

    def _require_dataset(self) -> str | None:
        if not self._dataset_name:
            QMessageBox.warning(self, "No data", "Load PEEM data first.")
            return None
        return self._dataset_name

    def _load_tif_stack(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load TIF stack", "", "TIFF (*.tif *.tiff)"
        )
        if path:
            self._start_load(path)

    def _load_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Load TIF folder")
        if path:
            self._start_load(path)

    def _load_beamline(self):
        """Pick UE49/UView/P3B files, or one+ acquisition folders (CL then CR, etc.)."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Beamline files (Cancel → pick folders)",
            "",
            "Beamline (*.txt *.dat *.P3B *.p3b);;All (*)",
        )
        if not paths:
            paths = []
            while True:
                folder = QFileDialog.getExistingDirectory(
                    self, f"Beamline folder {len(paths) + 1} (Cancel when done)"
                )
                if not folder:
                    break
                paths.append(folder)
                more = QMessageBox.question(
                    self,
                    "Add another folder?",
                    "Add another acquisition folder?\n"
                    "(e.g. CR with CL, or LV with LH)",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes if len(paths) == 1 else QMessageBox.No,
                )
                if more != QMessageBox.Yes:
                    break
        if not paths:
            return
        self._start_beamline_load(paths)

    def _start_load(self, path: str):
        name = self.name_edit.text().strip() or None
        if name is None:
            base = os.path.basename(path.rstrip(os.sep))
            if base.lower().endswith((".tif", ".tiff")):
                base = os.path.splitext(base)[0]
            self.name_edit.setText(base.replace(" ", "_"))
            name = self.name_edit.text().strip() or None

        self.status_label.setText(f"Loading {path}…")
        self._set_load_buttons_enabled(False)
        self._load_thread = PeemLoadThread(self.service, path, name=name)
        self._load_thread.finished_signal.connect(self._on_load_finished)
        self._load_thread.start()

    def _start_beamline_load(self, paths: list[str]):
        name = self.name_edit.text().strip() or None
        if name is None:
            base = os.path.basename(paths[0].rstrip(os.sep))
            base = os.path.splitext(base)[0]
            self.name_edit.setText(base.replace(" ", "_"))
            name = self.name_edit.text().strip() or None

        shown = ", ".join(os.path.basename(p.rstrip(os.sep)) for p in paths[:3])
        if len(paths) > 3:
            shown += f", … (+{len(paths) - 3})"
        self.status_label.setText(f"Loading beamline ({len(paths)}): {shown}…")
        self._set_load_buttons_enabled(False)
        self._load_thread = PeemLoadThread(
            self.service, name=name, beamline_paths=paths
        )
        self._load_thread.finished_signal.connect(self._on_load_finished)
        self._load_thread.start()

    def _set_load_buttons_enabled(self, enabled: bool):
        self.btn_load_stack.setEnabled(enabled)
        self.btn_load_folder.setEnabled(enabled)
        self.btn_load_beamline.setEnabled(enabled)

    def _on_load_finished(self, ok: bool, summary, message: str):
        self._set_load_buttons_enabled(True)
        if not ok:
            QMessageBox.critical(self, "Load error", message)
            self.status_label.setText(f"Load failed: {message}")
            return
        self._dataset_name = summary["name"]
        self.name_edit.setText(summary["name"])
        self._refresh_meta()
        parts = [
            f"Loaded '{summary['name']}'",
            f"{summary['n_frames']} frames",
            str(summary.get("pol_summary", {})),
        ]
        if summary.get("format"):
            parts.append(f"format={summary['format']}")
        if summary.get("csv_prompt"):
            parts.append("Attach CSV if needed.")
        self.status_label.setText(" | ".join(parts))

    def _attach_csv(self):
        name = self._require_dataset()
        if not name:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Attach beamline CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            summary = self.service.attach_csv(name, path)
            self._refresh_meta()
            self.status_label.setText(
                f"CSV attached to '{summary['name']}' (I0 present: {summary['I0_present']})"
            )
        except Exception as exc:
            QMessageBox.critical(self, "CSV error", str(exc))

    def _pair(self):
        name = self._require_dataset()
        if not name:
            return
        try:
            result = self.service.pair(name, self.pair_mode.currentData())
            self._refresh_meta()
            self.status_label.setText(
                f"Paired {result['n_pairs']} pairs ({result.get('unpaired_count', 0)} unpaired)"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Pair error", str(exc))

    def _separate(self):
        name = self._require_dataset()
        if not name:
            return
        try:
            result = self.service.separate(name)
            self._refresh_meta()
            self.status_label.setText(f"Separated channels: {', '.join(result['channels'])}")
        except Exception as exc:
            QMessageBox.critical(self, "Separate error", str(exc))

    def _apply_drift(self):
        name = self._require_dataset()
        if not name:
            return
        try:
            frame = self.service.get_view_tensor(
                name, node=self.drift_source.currentData(), frame_index=self.drift_ref.value()
            )
            ny, nx = frame.shape
            roi = self.service.default_drift_roi(ny, nx)
            result = self.service.drift(
                name,
                ref_index=self.drift_ref.value(),
                roi_dict=roi,
                search_radius=self.drift_radius.value(),
                source=self.drift_source.currentData(),
            )
            self._refresh_meta()
            algo = result.get("drift_method", "ncc_roi")
            self.drift_algo_label.setText(f"NCC ROI ({algo})")
            self.status_label.setText(
                f"Drift applied (max |dx|={result['max_abs_dx']}, |dy|={result['max_abs_dy']})"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Drift error", str(exc))

    def _bg_kwargs(self) -> dict:
        method = self.bg_method.currentData()
        kwargs = {
            "node": self.bg_node.currentData(),
            "method": method,
            "e0": self.bg_e0.value(),
            "e1": self.bg_e1.value(),
        }
        if method == "two_step":
            kwargs["post_e0"] = self.bg_post_e0.value()
            kwargs["post_e1"] = self.bg_post_e1.value()
        return kwargs

    def _bg_preview(self):
        name = self._require_dataset()
        if not name:
            return
        try:
            result = self.service.bg_preview(name, **self._bg_kwargs())
            self._plot_bg(result)
        except Exception as exc:
            QMessageBox.critical(self, "BG preview error", str(exc))

    def _bg_apply(self):
        name = self._require_dataset()
        if not name:
            return
        try:
            self.service.bg_apply(name, **self._bg_kwargs())
            self._refresh_meta()
            self.status_label.setText("Background subtracted and stored.")
        except Exception as exc:
            QMessageBox.critical(self, "BG apply error", str(exc))

    def _sumrule_kwargs(self) -> dict:
        return {
            "nh": self.sum_nh.value(),
            "l3_lo": self.sum_l3_lo.value(),
            "l3_hi": self.sum_l3_hi.value(),
            "l2_lo": self.sum_l2_lo.value(),
            "l2_hi": self.sum_l2_hi.value(),
            "r_lo": self.sum_r_lo.value(),
            "r_hi": self.sum_r_hi.value(),
        }

    def _sumrule_preview(self):
        name = self._require_dataset()
        if not name:
            return
        try:
            result = self.service.sumrule_preview(name, **self._sumrule_kwargs())
            self._plot_sumrule(result)
            self._show_sumrule_text(result)
        except Exception as exc:
            QMessageBox.critical(self, "Sum rule preview error", str(exc))

    def _sumrule_apply(self):
        name = self._require_dataset()
        if not name:
            return
        try:
            result = self.service.sumrule_apply(name, **self._sumrule_kwargs())
            stored = self.service.sumrule_preview(name, **self._sumrule_kwargs())
            self._plot_sumrule(stored)
            self._show_sumrule_text(stored)
            self._refresh_meta()
            self.status_label.setText(
                f"Sum rule stored ({result['tag_plus']}/{result['tag_minus']}, "
                f"source={result['source_kind']})"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Sum rule apply error", str(exc))

    def _show_sumrule_text(self, result: dict):
        self.sum_results.setText(
            f"p={result['p']:.4g}±{result['p_std']:.4g}  "
            f"q={result['q']:.4g}±{result['q_std']:.4g}  "
            f"r={result['r']:.4g}±{result['r_std']:.4g}\n"
            f"m_orb={result['m_orb']:.4g}±{result['m_orb_std']:.4g}  "
            f"m_spin+dipole={result['m_spin_plus_dipole']:.4g}±"
            f"{result['m_spin_plus_dipole_std']:.4g}"
        )

    def _plot_bg(self, result: dict):
        self.bg_ax.clear()
        energy = result["energy"]
        self.bg_ax.plot(energy, result["spectrum"], label="spectrum", color="#60a5fa")
        self.bg_ax.plot(energy, result["bg"], label="bg", color="#f97316")
        self.bg_ax.plot(energy, result["subtracted"], label="subtracted", color="#34d399")
        self.bg_ax.set_xlabel("Energy")
        self.bg_ax.set_ylabel("Intensity")
        self.bg_ax.legend(fontsize=7)
        self.bg_ax.set_title(f"BG preview ({result['method']})")
        self.bg_canvas.draw()
        self.view_tabs.setCurrentIndex(1)

    def _plot_sumrule(self, result: dict):
        self.sum_ax.clear()
        energy = result["energy"]
        self.sum_ax.plot(energy, result["mu_plus"], label=result["tag_plus"], color="#60a5fa")
        self.sum_ax.plot(energy, result["mu_minus"], label=result["tag_minus"], color="#f87171")
        self.sum_ax.plot(energy, result["dichroism"], label="dichroism", color="#a78bfa")
        self.sum_ax.set_xlabel("Energy")
        self.sum_ax.set_ylabel("μ")
        self.sum_ax.legend(fontsize=7)
        self.sum_ax.set_title("XMCD sum rule")
        self.sum_canvas.draw()
        self.view_tabs.setCurrentIndex(2)

    def _refresh_meta(self):
        name = self._dataset_name
        if not name:
            return
        try:
            self._meta = self.service.get_meta(name)
        except Exception as exc:
            QMessageBox.critical(self, "Meta error", str(exc))
            return
        self._populate_node_combo()
        n = int((self._meta or {}).get("n_frames", 1) or 1)
        self.drift_ref.setMaximum(max(0, n - 1))
        self._refresh_frame_view()

    def _populate_node_combo(self):
        prev = self.node_combo.currentData()
        self.node_combo.blockSignals(True)
        self.node_combo.clear()
        meta = self._meta or {}
        self.node_combo.addItem("Raw", "raw")
        if meta.get("has_processed"):
            self.node_combo.addItem("Processed", "processed")
        for child in meta.get("processed_children", []):
            self.node_combo.addItem(f"Processed / {child}", f"processed/{child}")
        for kind in meta.get("analysis_maps") or []:
            self.node_combo.addItem(f"Map / {kind.upper()}", f"analysis/{kind}")
        # restore prior selection if still present
        if prev is not None:
            idx = self.node_combo.findData(prev)
            if idx >= 0:
                self.node_combo.setCurrentIndex(idx)
        self.node_combo.blockSignals(False)

        self.bg_node.blockSignals(True)
        self.bg_node.clear()
        self.bg_node.addItem("Raw", "raw")
        if meta.get("has_processed"):
            self.bg_node.addItem("Processed", "processed")
        for child in meta.get("processed_children", []):
            if not str(child).endswith("_bg"):
                self.bg_node.addItem(f"Processed / {child}", f"processed/{child}")
        self.bg_node.blockSignals(False)

    def _on_node_changed(self):
        self._refresh_frame_view()

    def _refresh_frame_view(self):
        name = self._dataset_name
        if not name:
            return
        node = self.node_combo.currentData() or "raw"
        try:
            td = self.service.get_display_tensor(name, node)
            self.frame_viewer.load_data(td)
            view = self.frame_viewer.primary_view()
            if view is not None and td.ndim >= 2:
                # Prefer spatial axes last two dims when stack-like
                if td.ndim >= 3 and "y" in td.labels and "x" in td.labels:
                    view.combo_y.setCurrentIndex(td.labels.index("y"))
                    view.combo_x.setCurrentIndex(td.labels.index("x"))
                if node.startswith("analysis/"):
                    # dichroism maps: diverging cmap if available
                    if hasattr(view, "combo_cmap"):
                        idx = view.combo_cmap.findText("bwr")
                        if idx >= 0:
                            view.combo_cmap.setCurrentIndex(idx)
        except Exception as exc:
            self.status_label.setText(f"View error: {exc}")

    def _make_pcd(self):
        name = self._require_dataset()
        if not name:
            return
        try:
            result = self.service.make_pcd_map(
                name, register=bool(self.map_register.currentData())
            )
            self._refresh_meta()
            idx = self.node_combo.findData(result["display_node"])
            if idx >= 0:
                self.node_combo.setCurrentIndex(idx)
            self.view_tabs.setCurrentIndex(0)
            self.status_label.setText(
                f"PCD map ({result['formula']}, source={result['source']}, "
                f"mean={result['mean']:.4g})"
            )
        except Exception as exc:
            QMessageBox.critical(self, "PCD map error", str(exc))

    def _make_pld(self):
        name = self._require_dataset()
        if not name:
            return
        try:
            result = self.service.make_pld_map(
                name, register=bool(self.map_register.currentData())
            )
            self._refresh_meta()
            idx = self.node_combo.findData(result["display_node"])
            if idx >= 0:
                self.node_combo.setCurrentIndex(idx)
            self.view_tabs.setCurrentIndex(0)
            self.status_label.setText(
                f"PLD map ({result['formula']}, source={result['source']}, "
                f"mean={result['mean']:.4g})"
            )
        except Exception as exc:
            QMessageBox.critical(self, "PLD map error", str(exc))

    def _export_map(self):
        name = self._require_dataset()
        if not name:
            return
        node = self.node_combo.currentData() or ""
        kind = None
        if isinstance(node, str) and node.startswith("analysis/"):
            kind = node.split("/", 1)[1]
        else:
            maps = (self._meta or {}).get("analysis_maps") or []
            if not maps:
                QMessageBox.warning(self, "No map", "Make a PCD/PLD map first.")
                return
            kind = maps[-1]
        path, _ = QFileDialog.getSaveFileName(
            self, "Export map TIFF", f"{name}_{kind}.tif", "TIFF (*.tif)"
        )
        if not path:
            return
        try:
            out = self.service.export_analysis_map(name, kind, path, png=True)
            self.status_label.setText(f"Exported {out} (+ .png)")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))
