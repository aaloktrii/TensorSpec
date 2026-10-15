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
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

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
        images_path=None,
    ):
        super().__init__()
        self.service = service
        self.path = path
        self.name = name
        self.csv_path = csv_path
        self.beamline_paths = beamline_paths
        self.images_path = images_path

    def run(self):
        try:
            if self.beamline_paths is not None:
                summary = self.service.load_beamline(self.beamline_paths, name=self.name)
            elif self.images_path is not None:
                summary = self.service.load_images(self.images_path, name=self.name)
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
        self.btn_load_folder.setToolTip("Any PEEM folder in one step: TIF / TIFF, PNG / JPEG, UE49 scan,\n"
                                        "UView .dat, ALS .P3B, or a folder of acquisitions (pick which).\n"
                                        "Other files (.rtf, .ipf, notes ...) are skipped.")
        self.btn_load_folder.clicked.connect(self._load_folder)
        btn_row.addWidget(self.btn_load_stack)
        btn_row.addWidget(self.btn_load_folder)
        load_form.addRow(btn_row)
        self.btn_load_beamline = QPushButton("Load beamline…")
        self.btn_load_beamline.setToolTip(
            "UE49 scan .txt, UView .dat, or ALS .P3B (files or folders).\n"
            "UView CP/CM: pick CL folder, say Yes to add CR (or LH/LV).\n"
            "Then use Pair / Separate → Drift → BG → Sum Rule as usual."
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

        # 4-6: calculate + analyse (asymmetry map, image analysis, ROIs / domains); background and
        # sum rule are in the Sum Rule window (XAS suite), reached with "→ Sum Rule"
        from tensorspec.gui.components.peem_map_tools import PeemMapTools
        self.map_tools = PeemMapTools(self.service, lambda: self._dataset_name, self,
                                      send_to_sum_rule=self._to_sum_rule)
        control_layout.addWidget(self.map_tools.controls)

        control_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(control_panel)
        scroll.setMinimumWidth(360)
        splitter.addWidget(scroll)

        # --- RIGHT: Frame | BG | Sum Rule tabs (plots leave left column) ---
        self.view_tabs = QTabWidget()
        self.view_tabs.setDocumentMode(True)

        view_panel = QWidget()
        view_layout = QVBoxLayout(view_panel)
        view_layout.setContentsMargins(4, 4, 4, 4)
        self.view_figure = Figure(figsize=(6, 5), dpi=100, layout="tight")
        self.view_canvas = FigureCanvas(self.view_figure)
        self.view_ax = self.view_figure.add_subplot(111)
        view_layout.addWidget(self.view_canvas, stretch=1)

        nav = QHBoxLayout()
        self.node_combo = QComboBox()
        self.node_combo.currentIndexChanged.connect(self._on_node_changed)
        nav.addWidget(QLabel("Node:"))
        nav.addWidget(self.node_combo)

        self.channel_combo = QComboBox()
        self.channel_combo.currentIndexChanged.connect(self._refresh_frame_view)
        nav.addWidget(QLabel("Channel:"))
        nav.addWidget(self.channel_combo)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._refresh_frame_view)
        nav.addWidget(QLabel("Frame:"))
        nav.addWidget(self.frame_slider)
        view_layout.addLayout(nav)

        contrast = QHBoxLayout()
        self.vmin_spin = QDoubleSpinBox()
        self.vmax_spin = QDoubleSpinBox()
        for spin in (self.vmin_spin, self.vmax_spin):
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(4)
            spin.valueChanged.connect(self._refresh_frame_view)
        contrast.addWidget(QLabel("vmin:"))
        contrast.addWidget(self.vmin_spin)
        contrast.addWidget(QLabel("vmax:"))
        contrast.addWidget(self.vmax_spin)
        view_layout.addLayout(contrast)
        self.view_tabs.addTab(view_panel, "Frame")

        self.view_tabs.addTab(self.map_tools.map_view, "Map")
        self.view_tabs.addTab(self.map_tools.plot_view, "Analysis")

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
        """Any PEEM folder in one step (service.inspect_folder): TIF / TIFF, PNG / JPEG, UE49 scan,
        UView .dat, ALS .P3B, or a folder of acquisitions (pick which). Other files are skipped."""
        path = QFileDialog.getExistingDirectory(
            self, "Load folder (TIF, PNG / JPG, UE49 scan, UView .dat, ALS .P3B, or a folder of these)")
        if not path:
            return
        try:
            info = self.service.inspect_folder(path)
        except ValueError as exc:
            QMessageBox.warning(self, "Load folder", str(exc))
            return
        skipped = ", ".join(f"{n} {ext}" for ext, n in info.get("ignored", {}).items())
        self._skipped_note = f" | skipped: {skipped}" if skipped else ""
        kind = info["kind"]
        if kind == "tif":
            self._start_load(path)
        elif kind == "images":
            self._start_images_load(path)
        elif kind == "beamline":
            self._start_beamline_load([path])
        else:
            chosen = self._choose_acquisitions(path, info)
            if chosen:
                if not self.name_edit.text().strip():  # one dataset: named after the chosen folder
                    self.name_edit.setText(os.path.basename(path.rstrip(os.sep)).replace(" ", "_"))
                self._start_beamline_load(chosen)

    def _choose_acquisitions(self, root: str, info: dict) -> list[str]:
        """A folder of acquisitions: all listed and selected; click / shift / cmd to choose, then load
        the selected ones as one dataset (e.g. the CL and CR folders)."""
        from PySide6.QtWidgets import QAbstractItemView, QDialog, QDialogButtonBox, QListWidget

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Load from {os.path.basename(root)}")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(f"{len(info['paths'])} acquisition(s) found, all selected. "
                             "Click / shift-click / cmd-click to choose; they load as one dataset."))
        lst = QListWidget()
        lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for fmt, p in zip(info.get("formats", []), info["paths"]):
            lst.addItem(f"[{fmt}] {os.path.relpath(p, root)}")
            lst.item(lst.count() - 1).setData(Qt.UserRole, p)
        lst.selectAll()
        lst.setMinimumSize(560, 320)
        lay.addWidget(lst)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok = bb.button(QDialogButtonBox.Ok)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lst.itemSelectionChanged.connect(lambda: (ok.setText(f"Load selected ({len(lst.selectedItems())})"),
                                                  ok.setEnabled(bool(lst.selectedItems()))))
        ok.setText(f"Load selected ({lst.count()})")
        lay.addWidget(bb)
        if self._exec_dialog(dlg) != QDialog.Accepted:
            return []
        return [lst.item(i).data(Qt.UserRole) for i in range(lst.count()) if lst.item(i).isSelected()]

    def _exec_dialog(self, dlg) -> int:
        return dlg.exec()

    def _start_images_load(self, path: str):
        name = self.name_edit.text().strip() or os.path.basename(path.rstrip(os.sep)).replace(" ", "_")
        self.name_edit.setText(name)
        self.status_label.setText(f"Loading images {path}…")
        self._set_load_buttons_enabled(False)
        self._load_thread = PeemLoadThread(self.service, name=name, images_path=path)
        self._load_thread.finished_signal.connect(self._on_load_finished)
        self._load_thread.start()

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
        self.status_label.setText(" | ".join(parts) + getattr(self, "_skipped_note", ""))
        self._skipped_note = ""

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

    def _to_sum_rule(self, name: str):
        """Corrected CP / CM pairs -> the Sum Rule window (XAS suite), as a new dataset
        <name>_sumrule with an energy per pair (PeemService.to_sum_rule)."""
        res = self.service.to_sum_rule(name)
        from tensorspec.gui.suites.xas_suite import open_sum_rule
        open_sum_rule(res["name"], self.service)
        note = f" ({res['note']})" if res.get("note") else ""
        self.status_label.setText(f"Sent to the Sum Rule window as '{res['name']}'{note}.")

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
        self._configure_frame_slider()
        self._refresh_frame_view()
        self.map_tools.refresh()

    def _populate_node_combo(self):
        self.node_combo.blockSignals(True)
        self.node_combo.clear()
        meta = self._meta or {}
        self.node_combo.addItem("Raw", "raw")
        if meta.get("has_processed"):
            self.node_combo.addItem("Processed", "processed")
        for child in meta.get("processed_children", []):
            self.node_combo.addItem(f"Processed / {child}", f"processed/{child}")
        self.node_combo.blockSignals(False)


        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        tags = meta.get("channel_tags") or []
        if tags and meta.get("processed_is_paired"):
            for idx, tag in enumerate(tags):
                self.channel_combo.addItem(str(tag), idx)
        else:
            self.channel_combo.addItem("0", 0)
        self.channel_combo.blockSignals(False)

    def _on_node_changed(self):
        self._configure_frame_slider()
        self._refresh_frame_view()

    def _configure_frame_slider(self):
        meta = self._meta or {}
        name = self._dataset_name
        node = self.node_combo.currentData() or "raw"
        n = meta.get("n_frames", 1)
        if node == "processed" and meta.get("processed_is_paired"):
            n = meta.get("n_pairs") or n
        elif node == "processed" and meta.get("n_processed_frames") is not None:
            n = meta.get("n_processed_frames") or n
        elif node.startswith("processed/") and name:
            from tensorspec.core.workspace import global_workspace
            child = global_workspace.pull_tensor_data(name, node)
            if child is not None:
                n = int(child.value.shape[0])
            else:
                n = meta.get("n_pairs") or meta.get("n_frames", 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setEnabled(n > 1)
        self.frame_slider.setRange(0, max(0, n - 1))
        if self.frame_slider.value() > max(0, n - 1):
            self.frame_slider.setValue(0)
        self.drift_ref.setMaximum(max(0, n - 1))
        self.frame_slider.blockSignals(False)

    def _refresh_frame_view(self):
        name = self._dataset_name
        if not name:
            return
        node = self.node_combo.currentData() or "raw"
        channel = self.channel_combo.currentData()
        if channel is None:
            channel = 0
        frame_index = self.frame_slider.value()
        try:
            frame = self.service.get_view_tensor(
                name, node=node, frame_index=frame_index, channel=channel
            )
        except Exception:
            return

        finite = frame[np.isfinite(frame)]
        if finite.size:
            p1, p99 = np.percentile(finite, [1, 99])
            if p1 == p99:
                p1, p99 = float(finite.min()), float(finite.max())
        else:
            p1 = p99 = 0.0

        if not self.vmin_spin.hasFocus() and not self.vmax_spin.hasFocus():
            self.vmin_spin.blockSignals(True)
            self.vmax_spin.blockSignals(True)
            self.vmin_spin.setValue(float(p1))
            self.vmax_spin.setValue(float(p99))
            self.vmin_spin.blockSignals(False)
            self.vmax_spin.blockSignals(False)

        self.view_ax.clear()
        self.view_ax.imshow(
            frame,
            cmap="gray",
            vmin=self.vmin_spin.value(),
            vmax=self.vmax_spin.value(),
            origin="upper",
        )
        title_bits = [name, node, f"frame {frame_index}"]
        if node == "processed" and self._meta and self._meta.get("processed_is_paired"):
            tag = self.channel_combo.currentText()
            title_bits.append(f"ch={tag}")
        self.view_ax.set_title(" | ".join(title_bits))
        self.view_ax.axis("off")
        self.view_canvas.draw()
