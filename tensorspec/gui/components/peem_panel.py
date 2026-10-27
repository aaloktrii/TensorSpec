"""PEEM analysis panel — direct core access via PeemService."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tensorspec.gui.components.data_viewer_panel import DataViewerPanel
from tensorspec.core.peem import domain_analysis as da
from tensorspec.gui.components.peem_map_common import _dspin, action_style, bold_titles, tidy_forms
from tensorspec.gui.services.peem_service import PeemService
from tensorspec.plotting import peem_plots


class PeemLoadThread(QThread):
    """Load jobs in order, each ``(kind, source, name)``: ``("path", path, name)`` TIF stack / folder,
    ``("images", folder, name)`` PNG / JPEG frames, ``("beamline", [paths], name)`` UE49 / UView / P3B.
    A failed job does not stop the others. Emits (any loaded, [summaries], error lines)."""

    finished_signal = Signal(bool, object, str)

    def __init__(self, service: PeemService, jobs: list[tuple[str, object, str | None]]):
        super().__init__()
        self.service = service
        self.jobs = jobs

    def run(self):
        summaries, errors = [], []
        for kind, source, name in self.jobs:
            try:
                if kind == "beamline":
                    summary = self.service.load_beamline(source, name=name)
                elif kind == "images":
                    summary = self.service.load_images(source, name=name)
                else:
                    summary = self.service.load_path(source, name=name)
                summary["sources"] = [str(q) for q in (source if isinstance(source, list) else [source])]
                summaries.append(summary)
            except Exception as exc:
                first = source[0] if isinstance(source, list) else source
                errors.append(f"{os.path.basename(str(first).rstrip(os.sep))}: {exc}")
        self.finished_signal.emit(bool(summaries), summaries, "\n".join(errors))


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
        bold_titles(load_group)
        load_form = QFormLayout(load_group)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Dataset name (auto from path)")
        load_form.addRow("Name:", self.name_edit)
        btn_row = QHBoxLayout()
        self.btn_load_stack = QPushButton("Load TIF Stack")
        self.btn_load_stack.clicked.connect(self._load_tif_stack)
        self.btn_load_folder = QPushButton("Load Folder")
        self.btn_load_folder.setToolTip("One or more PEEM folders in one step (shift / cmd-click for several):\n"
                                        "TIF / TIFF, PNG / JPEG, UE49 scan, UView .dat, ALS .P3B, or a folder\n"
                                        "of acquisitions. Other files (.rtf, .ipf, notes ...) are skipped.")
        self.btn_load_folder.clicked.connect(self._load_folder)
        action_style(self.btn_load_folder, "load")
        btn_row.addWidget(self.btn_load_stack)
        btn_row.addWidget(self.btn_load_folder)
        load_form.addRow(btn_row)
        self.btn_attach_csv = QPushButton("Attach CSV")
        self.btn_attach_csv.clicked.connect(self._attach_csv)
        load_form.addRow(self.btn_attach_csv)
        self.dataset_combo = QComboBox()
        self.dataset_combo.setToolTip("Loaded datasets (e.g. several .nc map files); pick one to work on")
        self.dataset_combo.activated.connect(lambda *_: self.select_dataset(self.dataset_combo.currentText()))
        load_form.addRow("Dataset:", self.dataset_combo)
        self.status_label = QLabel("No PEEM data loaded")
        self.status_label.setWordWrap(True)
        load_form.addRow(self.status_label)
        control_layout.addWidget(load_group)

        pair_group = QGroupBox("2. Pair / Separate")
        bold_titles(pair_group)
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
        bold_titles(drift_group)
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



        # 4-7: calculate + analyse (dichroism map, image analysis, ROIs / domains, compare); background and
        # sum rule are in the Sum Rule window (XAS suite), reached with "→ Sum Rule"
        from tensorspec.gui.components.peem_map_tools import PeemMapTools
        self.map_tools = PeemMapTools(self.service, lambda: self._dataset_name, self,
                                      send_to_sum_rule=self._to_sum_rule, on_map_made=self._refresh_meta)
        if hasattr(self.map_tools, "btn_sum_rule"):  # the corrected pairs go to the Sum Rule window after Drift
            control_layout.insertWidget(control_layout.indexOf(drift_group) + 1, self.map_tools.btn_sum_rule)
        control_layout.addWidget(self.map_tools.controls)

        control_layout.addStretch()
        tidy_forms(control_panel)  # same clean form layout on macOS / Windows / Linux
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(control_panel)
        scroll.setMinimumWidth(400)
        splitter.addWidget(scroll)
        self._controls_scroll = scroll

        # --- RIGHT: Frame (DataViewer) | Map | Analysis ---
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

        self.view_tabs.addTab(self.map_tools.map_view, "Map")
        self.view_tabs.addTab(self.map_tools.analysis_view, "Analysis")
        # Analysis -> Domains has its own controls: the left column (1-7) is hidden there
        self.view_tabs.currentChanged.connect(lambda *_: self._sync_controls())
        self.map_tools.analysis_view.currentChanged.connect(lambda *_: self._sync_controls())

        splitter.addWidget(self.view_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 720])

    def _sync_controls(self):
        mt = self.map_tools
        on_domains = (self.view_tabs.currentWidget() is mt.analysis_view
                      and mt.analysis_view.currentWidget() is mt.domain_view)
        self._controls_scroll.setVisible(not on_domains)

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
            self, "Load TIF stack", self._last_dir(), "TIFF (*.tif *.tiff)"
        )
        if path:
            self._remember_dir(os.path.dirname(path))
            name = self._typed_name() or os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
            self._start_jobs([("path", path, name)])

    # --- Load Folder: one or more folders, recognized in one step ---------------------------------
    @staticmethod
    def _settings():
        from PySide6.QtCore import QSettings
        return QSettings("TensorSpec", "PEEM")

    def _last_dir(self) -> str:
        return str(self._settings().value("last_dir", "") or "")

    def _remember_dir(self, folder: str):
        self._settings().setValue("last_dir", folder)

    def _typed_name(self) -> str | None:
        """The Name box, unless it still shows the current dataset (then loading would overwrite it)."""
        text = self.name_edit.text().strip()
        return text if text and text != self._dataset_name else None

    def _pick_folders(self) -> list[str]:
        """Folder dialog that takes several folders (shift / cmd-click); starts where the last load was."""
        from PySide6.QtWidgets import QAbstractItemView, QDialog, QListView, QTreeView

        dlg = QFileDialog(self, "Load folder(s) — shift / cmd-click to pick several", self._last_dir())
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)  # the native dialog picks one folder only
        for view in dlg.findChildren(QListView) + dlg.findChildren(QTreeView):
            if view.objectName() in ("listView", "treeView"):
                view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        if self._exec_dialog(dlg) != QDialog.Accepted:
            return []
        folders = [f for f in dlg.selectedFiles() if os.path.isdir(f)]
        if folders:
            self._remember_dir(os.path.dirname(folders[0].rstrip(os.sep)))
        return folders

    def _load_folder(self):
        folders = self._pick_folders()
        if folders:
            self.load_folders(folders)

    def load_folders(self, folders: list[str]):
        """Recognize each folder (service.inspect_folder) and load them all in one go: TIF / image
        folders become one dataset each; beamline acquisitions (picked directly or inside a parent
        folder) are listed to choose from, then load as one dataset each or combined (e.g. CL + CR)."""
        from collections import Counter

        jobs, acqs, problems, skipped = [], [], [], Counter()
        for path in folders:
            try:
                info = self.service.inspect_folder(path)
            except ValueError as exc:
                problems.append(str(exc))
                continue
            skipped.update(info.get("ignored", {}))
            base = os.path.basename(path.rstrip(os.sep)).replace(" ", "_")
            if info["kind"] == "tif":
                jobs.append(("path", path, base))
            elif info["kind"] == "images":
                jobs.append(("images", path, base))
            else:
                acqs += [(fmt, q, base if info["kind"] == "container" else None)
                         for fmt, q in zip(info.get("formats") or [info["format"]], info["paths"])]
        if acqs:
            show_list = len(acqs) > 1 or any(parent for *_, parent in acqs)
            groups = self._choose_acquisitions(acqs) if show_list else [[acqs[0][1]]]
            for group in groups:
                parent = next((par for _, q, par in acqs if q == group[0]), None)
                jobs.append(("beamline", group, parent if len(group) > 1 and parent else self._group_name(group)))
        if problems:
            QMessageBox.warning(self, "Load folder", "\n\n".join(problems))
        if not jobs:
            return
        if len(jobs) == 1 and self._typed_name():
            jobs[0] = (jobs[0][0], jobs[0][1], self._typed_name())
        self._skipped_note = (" | skipped: " + ", ".join(f"{n} {e}" for e, n in skipped.items())) if skipped else ""
        self._start_jobs(jobs)

    @staticmethod
    def _group_name(paths: list[str]) -> str:
        """Dataset name from its folders: the parts they share, with the parts that differ joined,
        e.g. ..._Cr2p_LH_-15deg + ..._Cr2p_LV_-15deg -> ..._Cr2p_LH_LV_-15deg."""
        names = [os.path.basename(p.rstrip(os.sep)).replace(" ", "_") for p in paths]
        if len(names) == 1:
            return names[0]
        parts = [n.split("_") for n in names]
        lead = 0
        while all(len(q) > lead for q in parts) and len({q[lead] for q in parts}) == 1:
            lead += 1
        tail = 0
        while all(len(q) - lead > tail for q in parts) and len({q[-1 - tail] for q in parts}) == 1:
            tail += 1
        middle = list(dict.fromkeys("_".join(q[lead:len(q) - tail]) for q in parts))  # LH, LV
        middle = [m for m in middle if m]
        return "_".join(parts[0][:lead] + middle + (parts[0][len(parts[0]) - tail:] if tail else []))[:64]

    def _choose_acquisitions(self, acqs: list[tuple[str, str, str | None]]) -> list[list[str]]:
        """List the acquisitions (all selected; click / shift / cmd-click to choose) and how to load
        them: one dataset each (ALS .P3B: CP and CM are in every acquisition) or one combined dataset
        (UView CL + CR folders, UE49 CP + CM scans). Returns the groups to load."""
        from PySide6.QtWidgets import QAbstractItemView, QDialog, QDialogButtonBox, QListWidget

        dlg = QDialog(self)
        dlg.setWindowTitle("Load acquisitions")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(f"{len(acqs)} acquisition(s) found, all selected. Click / shift-click / "
                             "cmd-click to choose."))
        lst = QListWidget()
        lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for fmt, path, parent in acqs:
            shown = f"{parent}/{os.path.basename(path)}" if parent else os.path.basename(path)
            lst.addItem(f"[{fmt}] {shown}")
            lst.item(lst.count() - 1).setData(Qt.UserRole, path)
        lst.selectAll()
        lst.setMinimumSize(560, 320)
        lay.addWidget(lst)
        mode = QComboBox()
        mode.addItem("One dataset per acquisition", "each")
        mode.addItem("One combined dataset (e.g. CL + CR folders, CP + CM scans)", "combine")
        mode.setCurrentIndex(0 if all(fmt == "p3b" for fmt, *_ in acqs) else 1)
        form = QFormLayout()
        form.addRow("Load as:", mode)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok = bb.button(QDialogButtonBox.Ok)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lst.itemSelectionChanged.connect(lambda: (ok.setText(f"Load selected ({len(lst.selectedItems())})"),
                                                  ok.setEnabled(bool(lst.selectedItems()))))
        ok.setText(f"Load selected ({lst.count()})")
        lay.addWidget(bb)
        self._acq_mode = mode  # tests pick the mode here
        if self._exec_dialog(dlg) != QDialog.Accepted:
            return []
        chosen = [lst.item(i).data(Qt.UserRole) for i in range(lst.count()) if lst.item(i).isSelected()]
        if not chosen:
            return []
        return [chosen] if mode.currentData() == "combine" else [[c] for c in chosen]

    def _exec_dialog(self, dlg) -> int:
        return dlg.exec()

    def _start_jobs(self, jobs: list[tuple[str, object, str | None]]):
        shown = ", ".join(name or "?" for *_, name in jobs[:3]) + (f", … (+{len(jobs) - 3})" if len(jobs) > 3 else "")
        self.status_label.setText(f"Loading {len(jobs)} dataset(s): {shown}…")
        self._set_load_buttons_enabled(False)
        self._load_thread = PeemLoadThread(self.service, jobs)
        self._load_thread.finished_signal.connect(self._on_load_finished)
        self._load_thread.start()

    def _set_load_buttons_enabled(self, enabled: bool):
        self.btn_load_stack.setEnabled(enabled)
        self.btn_load_folder.setEnabled(enabled)

    def _on_load_finished(self, ok: bool, summaries, message: str):
        self._set_load_buttons_enabled(True)
        if message:
            QMessageBox.critical(self, "Load error", message)
        if not ok:
            self.status_label.setText(f"Load failed: {message}")
            return
        first = summaries[0]
        if not hasattr(self, "_sources"):
            self._sources = {}
        for summary in summaries:
            self._sources[summary["name"]] = summary.get("sources", [])
        self._dataset_name = first["name"]
        self.name_edit.setText(first["name"])
        self._refresh_meta()
        self._fill_dataset_combo()
        if len(summaries) == 1:
            parts = [f"Loaded '{first['name']}'", f"{first['n_frames']} frames", str(first.get("pol_summary", {}))]
            if len(first.get("sources", [])) > 1:
                parts.insert(1, f"{len(first['sources'])} folders: "
                             + ", ".join(os.path.basename(q.rstrip(os.sep)) for q in first["sources"]))
            if first.get("format"):
                parts.append(f"format={first['format']}")
            if first.get("csv_prompt"):
                parts.append("Attach CSV if needed.")
        else:
            parts = [f"Loaded {len(summaries)} datasets: "
                     + ", ".join(f"{s['name']} ({s['n_frames']} frames)" for s in summaries),
                     "pick one under Dataset"]
        self.status_label.setText(" | ".join(parts) + getattr(self, "_skipped_note", ""))
        self._skipped_note = ""

    def _fill_dataset_combo(self):
        names = self.service.list_datasets()
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItems(names)
        for i, n in enumerate(names):  # hover a dataset to see the folders it came from
            folders = getattr(self, "_sources", {}).get(n)
            if folders:
                self.dataset_combo.setItemData(i, "\n".join(folders), Qt.ToolTipRole)
        if self._dataset_name in names:
            self.dataset_combo.setCurrentText(self._dataset_name)
        self.dataset_combo.blockSignals(False)

    def select_dataset(self, name: str):
        """Work on another loaded dataset (frames, maps and tools follow)."""
        if not name or name == self._dataset_name:
            return
        self._dataset_name = name
        self.name_edit.setText(name)
        self._refresh_meta()
        self.status_label.setText(f"Dataset '{name}'")

    def open_files(self, paths: list[str]) -> list[str]:
        """Load .nc map files (and TIF stacks / PEEM folders) given on the command line; the first
        becomes the current dataset. Returns the dataset names."""
        names, errors = [], []
        for p in paths:
            try:
                if str(p).lower().endswith((".nc", ".nc4", ".netcdf")):
                    summary = self.service.load_maps_nc(p)
                else:
                    summary = self.service.load_path(Path(p))
                names.append(summary["name"])
            except Exception as exc:  # report every file, keep the others
                errors.append(f"{Path(p).name}: {exc}")
        if names:
            self._dataset_name = None
            self.select_dataset(names[0])
            self._fill_dataset_combo()
            self.view_tabs.setCurrentWidget(self.map_tools.map_view)
            self.status_label.setText(f"Opened {len(names)} file(s): {', '.join(names)}"
                                      + (f" | failed: {'; '.join(errors)}" if errors else ""))
        elif errors:
            QMessageBox.critical(self, "Open", "\n".join(errors))
        return names

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
            msg = str(exc)
            if "(pair, channel=2, y, x)" in msg:  # Drift on Raw replaced the pairs with single frames
                msg = ("The pairs are gone: Drift with Source 'Raw' replaced them with drift-corrected single "
                       "frames.\n\nRun 2. Stack Pairs again, then 3. Drift with Source 'Processed' "
                       "(now chosen automatically after Stack Pairs), then Separate.")
            QMessageBox.critical(self, "Separate error", msg)

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
            from PySide6.QtGui import QGuiApplication
            QGuiApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                result = self.service.drift(
                    name,
                    ref_index=self.drift_ref.value(),
                    roi_dict=roi,
                    search_radius=self.drift_radius.value(),
                    source=self.drift_source.currentData(),
                )
            finally:
                QGuiApplication.restoreOverrideCursor()
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
        n = int((self._meta or {}).get("n_frames", 1) or 1)
        self.drift_ref.setMaximum(max(0, n - 1))
        self._refresh_frame_view()
        meta = self._meta or {}
        pols = set(meta.get("pol") or [])  # no Stack Pairs yet: name the pair from the frames' polarizations
        self.map_tools.set_pair(meta.get("channel_tags") or next(
            (list(p) for p in (("CP", "CM"), ("LH", "LV")) if set(p) <= pols), None))
        if (self._meta or {}).get("processed_is_paired"):  # drift the pairs, not the raw frames
            self.drift_source.setCurrentIndex(self.drift_source.findData("processed"))
        self.map_tools.refresh()

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


# ------------------------------------------------------------------ Analysis tab, "Domains"
# Spatial domain analysis of the maps pushed from the Map view (6. Push to analysis): pushed maps,
# pipeline steps (SDA toolkit defaults), Run, one step with axes in um, numbers table, export.
# Every step is a PeemService call; math in core/peem/domain_analysis.py, drawing in plotting/peem_plots.py.
def _domain_combo(items: list[tuple[str, object]], current=None) -> QComboBox:
    c = QComboBox()
    for label, data in items:
        c.addItem(label, data)
    if current is not None:
        c.setCurrentIndex(max(0, c.findData(current)))
    return c


class PeemDomainView(QWidget):
    def __init__(self, service, parent=None):
        super().__init__(parent)
        self.service = service
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        root.addWidget(split)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        controls = self._build_controls()
        tidy_forms(controls)
        scroll.setWidget(controls)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(controls.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 4)
        split.addWidget(scroll)
        split.addWidget(self._build_results())
        split.setStretchFactor(1, 1)
        split.setSizes([400, 700])
        tidy_forms(self)

    # ------------------------------------------------------------------ building
    def _build_controls(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("Pushed maps")
        bold_titles(g)
        v = QVBoxLayout(g)
        self.items = QListWidget()
        self.items.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.items.setToolTip("Maps pushed from the Map view (6. Push to analysis); select one or more")
        self.items.setMaximumHeight(130)
        self.items.itemSelectionChanged.connect(self._on_item_changed)
        v.addWidget(self.items)
        row = QHBoxLayout()
        b = QPushButton("Remove selected")
        b.clicked.connect(self._remove)
        row.addWidget(b)
        b = QPushButton("↻")
        b.setFixedWidth(30)
        b.setToolTip("Update the list")
        b.clicked.connect(lambda: self.refresh())
        row.addWidget(b)
        v.addLayout(row)
        lay.addWidget(g)

        g = QGroupBox("ROI and background")
        bold_titles(g)
        f = QFormLayout(g)
        row = QHBoxLayout()
        for label, kind in (("Draw rectangle", "rect"), ("Draw polygon", "polygon")):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, k=kind: self._start_draw("roi", k))
            row.addWidget(b)
        b = QPushButton("Clear")
        b.clicked.connect(lambda: self._set_region("roi", None))
        row.addWidget(b)
        f.addRow("ROI:", row)
        row = QHBoxLayout()
        self.roi_edit = QLineEdit()
        self.roi_edit.setPlaceholderText("x0, x1, y0, y1 (px)")
        self.roi_edit.returnPressed.connect(self._set_roi_typed)
        row.addWidget(self.roi_edit, 1)
        b = QPushButton("Set")
        b.clicked.connect(self._set_roi_typed)
        row.addWidget(b)
        f.addRow("Type ROI:", row)
        self.auto_flake = QCheckBox("Auto flake outline")
        self.auto_flake.setToolTip("Find the flake on the pushed map (denoise, local contrast, Otsu, largest region;\n"
                                   "not the drift strips at the edges) and use its outline as the ROI; untick to clear")
        self.auto_flake.toggled.connect(self._auto_flake_toggled)
        f.addRow(self.auto_flake)
        row = QHBoxLayout()
        for label, fn in (("Save outline…", self._save_outline), ("Load outline…", self._load_outline)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        f.addRow(row)
        self.use_bg = QCheckBox("Background region")
        self.use_bg.setToolTip("A non-magnetic area: its mean is the zero for ROI statistics (and for the\n"
                               "analysis with Background: Background region). Used only while ticked.")
        f.addRow(self.use_bg)
        row = QHBoxLayout()
        for label, kind in (("Draw rectangle", "rect"), ("Draw polygon", "polygon")):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, k=kind: self._start_draw("bg", k))
            row.addWidget(b)
        f.addRow(row)
        b = QPushButton("ROI statistics")
        b.setToolTip("Of the pushed map in the ROI: f_up / f_down split at zero (the background mean if ticked)")
        b.clicked.connect(self._roi_stats)
        f.addRow(b)
        lay.addWidget(g)

        g = QGroupBox("Steps")
        bold_titles(g)
        f = QFormLayout(g)
        d = da.DEFAULTS
        self.denoise = _domain_combo([("TV-Chambolle", "tv"), ("Median", "median"), ("None", "none")], d["denoise"])
        self.tv_weight = _dspin(0.0, 2.0, d["tv_weight"], 0.01, 2)
        f.addRow("Denoise:", self.denoise)
        f.addRow("TV weight:", self.tv_weight)
        self.flake = _domain_combo([("ROI (drawn / outline)", "pushed"), ("Automatic", "auto"), ("Whole map", "all")],
                                   d["flake"])
        f.addRow("Flake:", self.flake)
        self.background = _domain_combo([("Gaussian high-pass", "gaussian"), ("Background region", "region"),
                                  ("None (asymmetry 0)", "none")], d["background"])
        self.background.setToolTip("Gaussian: subtract a large-scale blur (σ must exceed the domains).\n"
                                   "Background region: its mean is the magnetic zero (push with the region ticked).\n"
                                   "None: the map's own zero.")
        self.sigma = _dspin(1.0, 1000.0, d["flatten_sigma"], 5.0, 0)
        f.addRow("Background:", self.background)
        f.addRow("Gaussian σ (px):", self.sigma)
        self.fft = QCheckBox()
        self.fft_cut = QSpinBox()
        self.fft_cut.setRange(1, 2000)
        self.fft_cut.setValue(d["fft_cutoff"])
        self.fft_mode = _domain_combo([("radial", "radial"), ("along x", "x")], d["fft_mode"])
        self.fft.setText("FFT filter")
        f.addRow(self.fft)
        row = QHBoxLayout()
        row.addWidget(self.fft_cut, 1)
        row.addWidget(self.fft_mode, 1)
        f.addRow("FFT cutoff:", row)
        self.m0 = _dspin(0.0, 1.0, d["m_threshold"], 0.01, 3)
        f.addRow("Three states m₀:", self.m0)
        self.min_px = QSpinBox()
        self.min_px.setRange(0, 100000)
        self.min_px.setValue(d["min_px"])
        f.addRow("Min domain (px):", self.min_px)
        self.conn = _domain_combo([("4 (edges)", 1), ("8 (+ corners)", 2)], d["connectivity"])
        f.addRow("Connectivity:", self.conn)
        row = QHBoxLayout()
        b = action_style(QPushButton("Run selected"), "domains")
        b.clicked.connect(lambda: self._run(all_items=False))
        row.addWidget(b)
        b = QPushButton("Run all")
        b.clicked.connect(lambda: self._run(all_items=True))
        row.addWidget(b)
        f.addRow(row)
        lay.addWidget(g)

        g = QGroupBox("Show and export")
        bold_titles(g)
        f = QFormLayout(g)
        self.step = _domain_combo([(da.STEP_LABELS[s], s) for s in da.STEPS], "width_map")  # right panel
        self.step.currentIndexChanged.connect(self._show)
        f.addRow("Show (right):", self.step)
        self.sizes = QCheckBox("Domain-size distribution")
        self.sizes.toggled.connect(self._show)
        f.addRow(self.sizes)
        row = QHBoxLayout()
        for label, fn, tip in (("Pushed map PDF…", self._export_pushed_pdf,
                                "Left panel: the pushed map as in the Map view, with the ROI / background outlines"),
                               ("Step PDF…", self._export_pdf, "Right panel: the step chosen under Show (right)")):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        f.addRow("Export:", row)
        row = QHBoxLayout()
        for label, fn in (("Table CSV…", self._export_table), ("Sizes CSV…", self._export_sizes)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        f.addRow(row)
        row = QHBoxLayout()
        for label, fn, tip in (("Save analysis…", self._save_analysis,
                                "The selected pushed map with its ROI / background, look, settings and results (.nc)"),
                               ("Load analysis…", self._load_analysis,
                                "A saved analysis as a pushed map, with its settings: run it again or change them")):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        f.addRow("Analysis file:", row)
        lay.addWidget(g)
        self.status = QLabel("")  # shown under the image (see _build_results)
        self.status.setWordWrap(True)
        lay.addStretch(1)
        return w

    def _build_results(self) -> QWidget:
        split = QSplitter(Qt.Vertical)
        split.setChildrenCollapsible(False)
        top = QWidget()
        v = QVBoxLayout(top)
        v.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(6, 4.5), layout="constrained")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(Qt.StrongFocus)  # arrow keys move the selected outline
        self._move = {"selected": None, "start": None, "region": None, "preview": None}
        for e, fn in (("button_press_event", self._move_press), ("motion_notify_event", self._move_motion),
                      ("button_release_event", self._move_release), ("key_press_event", self._move_key)):
            self.canvas.mpl_connect(e, fn)  # click an outline on the pushed map to select it
        v.addWidget(self.canvas, 1)
        self.caption = QLabel("")
        self.caption.setWordWrap(True)
        v.addWidget(self.caption)
        v.addWidget(self.status)
        split.addWidget(top)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        split.addWidget(self.table)
        split.setSizes([520, 200])
        return split

    # ------------------------------------------------------------------ helpers
    def _err(self, msg: str):
        QMessageBox.warning(self, "Domain analysis", msg)

    def params(self) -> dict:
        return {"denoise": self.denoise.currentData(), "tv_weight": self.tv_weight.value(),
                "flake": self.flake.currentData(), "background": self.background.currentData(),
                "flatten_sigma": self.sigma.value(), "fft_filter": self.fft.isChecked(),
                "fft_cutoff": self.fft_cut.value(), "fft_mode": self.fft_mode.currentData(),
                "m_threshold": self.m0.value(), "min_px": self.min_px.value(),
                "connectivity": self.conn.currentData()}

    def _selected(self) -> list[tuple[str, str]]:
        return [tuple(i.data(Qt.UserRole)) for i in self.items.selectedItems()]

    def refresh(self, select: tuple[str, str] | None = None):
        """Reload the pushed maps and the table; select (name, label) if given."""
        keep = [select] if select else self._selected()
        self.items.blockSignals(True)
        self.items.clear()
        for it in self.service.domain_inputs():
            item = QListWidgetItem(it["title"] + ("" if it["analysed"] else "  · not run"))
            item.setData(Qt.UserRole, [it["name"], it["label"]])
            self.items.addItem(item)
            if (it["name"], it["label"]) in keep:
                item.setSelected(True)
        if not self.items.selectedItems() and self.items.count():
            self.items.item(self.items.count() - 1).setSelected(True)
        self.items.blockSignals(False)
        self._fill_table()
        self._show()

    def _fill_table(self):
        rows = self.service.domain_table()
        self.table.clear()
        self.table.setRowCount(len(rows))
        cols = list(rows[0]) if rows else []
        cols = [c for c in cols if c != "label"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        for r, row in enumerate(rows):
            for c, k in enumerate(cols):
                v = row[k]
                text = f"{v:.3g}" if isinstance(v, float) else str(v)
                self.table.setItem(r, c, QTableWidgetItem(text))

    # ------------------------------------------------------------------ actions
    def _on_item_changed(self):
        self._move["selected"] = None
        self.auto_flake.blockSignals(True)  # the tick belongs to the map it was ticked on
        self.auto_flake.setChecked(False)
        self.auto_flake.blockSignals(False)
        self._show()

    def _run(self, all_items: bool):
        items = [(it["name"], it["label"]) for it in self.service.domain_inputs()] if all_items else self._selected()
        if not items:
            self._err("Push a map first (Map view, 6. Push to analysis).")
            return
        from PySide6.QtGui import QGuiApplication

        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        done, errors = 0, []
        try:
            for name, label in items:
                try:
                    self.service.run_domain_analysis(name, label, **self.params())
                    done += 1
                except (ValueError, KeyError) as exc:
                    errors.append(f"{name} · {label}: {exc}")
        finally:
            QGuiApplication.restoreOverrideCursor()
        if errors:
            self._err("\n".join(errors))
        self.status.setText(f"Analysed {done} map(s).")
        self.refresh(select=items[-1] if not all_items else None)

    def _remove(self):
        for name, label in self._selected():
            self.service.remove_domain_input(name, label)
        self.refresh()

    def _show(self):
        """Left: the pushed map as in the Map view (ROI / background drawn on it). Right: the chosen step,
        zoomed to the flake. Third panel: the domain-size distribution, when ticked."""
        sel = self._selected()
        self.fig.clear()
        self.caption.setText("")
        self._tool = None
        if not sel:
            self.canvas.draw_idle()
            return
        name, label = sel[0]
        step = self.step.currentData()
        result = self.service.domain_result(name, label)
        look = self.service.domain_look(name, label)
        um = look["um_per_px"]
        sizes = self.sizes.isChecked() and bool(result)
        n = 3 if sizes else 2
        ax0 = self.fig.add_subplot(1, n, 1)
        im0 = peem_plots.pushed_map(ax0, self.service.domain_layer(name, label, "input"), look, f"{name} · pushed map")
        self.fig.colorbar(im0, ax=ax0, fraction=0.04, pad=0.02)
        self._ax, self._scale = ax0, (um or 1.0)  # drawing happens on the pushed map
        sel_w = self._move.get("selected")
        region = self.service.domain_regions(name, label).get(sel_w) if sel_w else None
        if region is not None:  # the selected outline, thicker (axes limits unchanged)
            xs, ys = self._region_xy(region)
            lims = ax0.get_xlim(), ax0.get_ylim()
            ax0.plot([v * self._scale for v in xs], [v * self._scale for v in ys],
                     color="yellow" if sel_w == "roi" else "cyan", lw=2.6, zorder=15, scalex=False, scaley=False)
            ax0.set_xlim(*lims[0])
            ax0.set_ylim(*lims[1])
        ax1 = self.fig.add_subplot(1, n, 2)
        if result:
            im1 = peem_plots.domain_step(ax1, self.service.domain_layer(name, label, step), step, um,
                                         da.STEP_LABELS.get(step, step))
            try:  # zoom to the flake (axes keep the field's coordinates)
                peem_plots.zoom_to_mask(ax1, self.service.domain_layer(name, label, "flake"), um)
            except ValueError:
                pass
            if im1 is not None:
                self.fig.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02)
        else:
            ax1.set_axis_off()
            ax1.text(0.5, 0.5, f"{da.STEP_LABELS.get(step, step)}\n\nRun selected to compute it", ha="center",
                     va="center", transform=ax1.transAxes, fontsize=9, color="0.4")
        if sizes:
            peem_plots.domain_sizes(self.fig.add_subplot(1, n, 3), self.service.domain_areas(name, label))
        self.caption.setText(peem_plots.domain_numbers_text(result) if result else "Not run yet: Run selected.")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ ROI and background
    def _current(self) -> tuple[str, str] | None:
        sel = self._selected()
        if not sel:
            self._err("Push a map first (Map view, 6. Push to analysis) and select it.")
            return None
        return sel[0]

    def _start_draw(self, which: str, kind: str):
        """Draw the ROI / background region on the pushed map (shown as in the Map view)."""
        from matplotlib.widgets import PolygonSelector, RectangleSelector

        cur = self._current()
        if cur is None:
            return
        self._show()
        s, color = self._scale, ("yellow" if which == "roi" else "cyan")
        if kind == "rect":
            def done(press, release):
                xs = sorted((press.xdata / s, release.xdata / s))
                ys = sorted((press.ydata / s, release.ydata / s))
                self._set_region(which, {"kind": "rect", "x0": int(round(xs[0])), "x1": int(round(xs[1])),
                                         "y0": int(round(ys[0])), "y1": int(round(ys[1]))})
            self._tool = RectangleSelector(self._ax, done, useblit=False, interactive=False,
                                           props=dict(ec=color, fill=False, lw=1.2))
            self.status.setText("Press and drag on the pushed map.")
        else:
            def done(verts):
                self._set_region(which, {"kind": "polygon", "points": [[x / s, y / s] for x, y in verts]})
            self._tool = PolygonSelector(self._ax, done, useblit=False, props=dict(color=color, lw=1.2))
            self.status.setText("Click the corners on the pushed map; click the first corner again to close.")

    def _set_region(self, which: str, region: dict | None):
        cur = self._current()
        if cur is None:
            return
        try:
            self.service.set_domain_region(cur[0], cur[1], which, region)
        except ValueError as exc:
            self._err(str(exc))
            return
        if which == "bg" and region is not None:
            self.use_bg.setChecked(True)
        if which == "roi" and region is not None and region.get("kind") == "rect":
            self.roi_edit.setText(f"{region['x0']}, {region['x1']}, {region['y0']}, {region['y1']}")
        self.status.setText({"roi": "ROI", "bg": "Background region"}[which] + (" set." if region else " cleared."))
        self.refresh(select=cur)

    # ---- move an outline (drawn, typed, auto or loaded)
    @staticmethod
    def _shift_region(region: dict, dx: float, dy: float) -> dict:
        r = dict(region)
        if r["kind"] == "rect":
            dx, dy = int(round(dx)), int(round(dy))
            r.update(x0=r["x0"] + dx, x1=r["x1"] + dx, y0=r["y0"] + dy, y1=r["y1"] + dy)
        elif r["kind"] == "polygon":
            r["points"] = [[float(x) + dx, float(y) + dy] for x, y in r["points"]]
        elif r["kind"] == "ellipse":
            r.update(cx=r["cx"] + dx, cy=r["cy"] + dy)
        return r

    @classmethod
    def _rotate_region(cls, region: dict, deg: float) -> dict:
        """Rotate a region about its centre by deg (positive = clockwise as displayed, rows going down);
        a rectangle or ellipse becomes a polygon."""
        xs, ys = cls._region_xy(region)
        pts = np.column_stack([xs[:-1], ys[:-1]]).astype(float)  # without the closing point
        c = pts.mean(axis=0)
        t = np.radians(deg)
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        out = (pts - c) @ rot.T + c
        return {"kind": "polygon", "points": [[float(x), float(y)] for x, y in out]}

    @staticmethod
    def _region_xy(region: dict) -> tuple[list[float], list[float]]:
        """Outline of a region in px (closed), for the move preview."""
        if region["kind"] == "rect":
            xs = [region["x0"], region["x1"], region["x1"], region["x0"], region["x0"]]
            ys = [region["y0"], region["y0"], region["y1"], region["y1"], region["y0"]]
        elif region["kind"] == "polygon":
            xs = [p[0] for p in region["points"]] + [region["points"][0][0]]
            ys = [p[1] for p in region["points"]] + [region["points"][0][1]]
        else:
            t = np.linspace(0, 2 * np.pi, 64)
            xs = list(region["cx"] + region["rx"] * np.cos(t))
            ys = list(region["cy"] + region["ry"] * np.sin(t))
        return xs, ys

    def _move_press(self, ev):
        """Click inside the ROI / background region on the pushed map: select it (arrows move it, a drag too);
        click elsewhere: deselect. Ignored while a rectangle / polygon is being drawn."""
        cur = self._selected()[0] if self._selected() else None
        if cur is None or self._tool is not None or ev.inaxes is not getattr(self, "_ax", None) or ev.xdata is None:
            return
        self.canvas.setFocus()
        s = self._scale
        x, y = ev.xdata / s, ev.ydata / s
        regions = self.service.domain_regions(*cur)
        look = self.service.domain_look(*cur)
        which = None
        for w in ("roi", "bg"):  # the region under the mouse
            m = look.get(w)
            if m is not None and 0 <= int(y) < m.shape[0] and 0 <= int(x) < m.shape[1] and m[int(y), int(x)]:
                which = w
                break
        if which != self._move["selected"]:  # this click only selects (or deselects)
            self._move["selected"] = which
            self._show()
            self.status.setText(f"{'ROI' if which == 'roi' else 'Background region'} selected: arrow keys move it "
                                "(Shift: 10 px), [ / ] rotate it 1° (Shift: 5°), or drag it; click elsewhere to "
                                "deselect." if which else "")
            return
        if which is None or regions.get(which) is None:
            return
        from matplotlib.lines import Line2D

        line = Line2D([], [], color="yellow" if which == "roi" else "cyan", lw=1.2, ls="--", zorder=20)
        self._ax.add_line(line)
        self._move.update(start=(x, y), region=regions[which], preview=line)

    def _move_motion(self, ev):
        mv = self._move
        if mv["start"] is None or ev.inaxes is not self._ax or ev.xdata is None:
            return
        s = self._scale
        dx, dy = ev.xdata / s - mv["start"][0], ev.ydata / s - mv["start"][1]
        xs, ys = self._region_xy(self._shift_region(mv["region"], dx, dy))
        mv["preview"].set_data([v * s for v in xs], [v * s for v in ys])
        self.canvas.draw_idle()

    def _move_release(self, ev):
        mv = self._move
        if mv["start"] is None:
            return
        s = self._scale
        if ev.xdata is not None and ev.inaxes is self._ax:
            dx, dy = ev.xdata / s - mv["start"][0], ev.ydata / s - mv["start"][1]
        else:
            dx = dy = 0.0
        region, which = mv["region"], mv["selected"]
        mv.update(start=None, region=None, preview=None)
        if abs(dx) < 0.5 and abs(dy) < 0.5:  # a click: selection only
            return
        self._set_region(which, self._shift_region(region, dx, dy))
        self.status.setText(f"{'ROI' if which == 'roi' else 'Background region'} moved by ({dx:+.0f}, {dy:+.0f}) px.")

    def _move_key(self, ev):
        """Selected outline: arrows move it 1 px (Shift: 10 px); ] / [ rotate it 1 deg clockwise /
        anticlockwise about its centre (Shift: 5 deg)."""
        steps = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
        turns = {"]": 1.0, "[": -1.0, "}": 5.0, "{": -5.0, "shift+]": 5.0, "shift+[": -5.0}
        key = str(ev.key or "")
        base = key.replace("shift+", "")
        if (base not in steps and key not in turns) or not self._selected():
            return
        which = self._move.get("selected")
        if which is None:
            return
        region = self.service.domain_regions(*self._selected()[0]).get(which)
        if region is None:
            return
        what = "ROI" if which == "roi" else "Background region"
        if key in turns:
            self._set_region(which, self._rotate_region(region, turns[key]))
            self.status.setText(f"{what} rotated {turns[key]:+.0f}° (clockwise +). [ / ] 1°, Shift 5°; arrows move.")
        else:
            k = 10 if "shift" in key else 1
            dx, dy = steps[base]
            self._set_region(which, self._shift_region(region, dx * k, dy * k))
            self.status.setText(f"{what} moved ({dx * k:+d}, {dy * k:+d}) px. Arrows 1 px, Shift 10 px; [ / ] rotate.")
        self.canvas.setFocus()

    def _set_roi_typed(self):
        from tensorspec.gui.components.peem_map_common import _numbers

        try:
            x0, x1, y0, y1 = (int(round(v)) for v in _numbers(self.roi_edit.text(), 4))
        except ValueError as exc:
            self._err(str(exc))
            return
        self._set_region("roi", {"kind": "rect", "x0": min(x0, x1), "x1": max(x0, x1), "y0": min(y0, y1), "y1": max(y0, y1)})

    def _auto_flake_toggled(self, on: bool):
        cur = self._selected()[0] if self._selected() else None
        if cur is None:
            return
        if not on:
            self._set_region("roi", None)
            return
        try:
            roi = self.service.domain_auto_flake(*cur)
        except ValueError as exc:
            self._err(str(exc))
            self.auto_flake.blockSignals(True)
            self.auto_flake.setChecked(False)
            self.auto_flake.blockSignals(False)
            return
        self.refresh(select=cur)
        self.status.setText(f"Flake outline found ({len(roi['points'])} points): it is the ROI now.")

    def _save_outline(self):
        cur = self._current()
        if cur is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save ROI outline", f"{cur[0]}_roi.json", "ROI (*.json)")
        if path:
            try:
                self.status.setText(f"Saved {self.service.save_domain_region(cur[0], cur[1], path, 'roi')}")
            except ValueError as exc:
                self._err(str(exc))

    def _load_outline(self):
        cur = self._current()
        if cur is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load ROI outline", "", "ROI (*.json)")
        if path:
            try:
                self.service.load_domain_region(cur[0], cur[1], path, "roi")
            except ValueError as exc:
                self._err(str(exc))
                return
            self.refresh(select=cur)
            self.status.setText(f"ROI loaded from {os.path.basename(path)}")

    def _roi_stats(self):
        cur = self._current()
        if cur is None:
            return
        try:
            st = self.service.domain_roi_statistics(cur[0], cur[1], use_bg=self.use_bg.isChecked())
        except ValueError as exc:
            self._err(str(exc))
            return
        where = "In the ROI" if st["in_roi"] else "In the whole map"
        zero = f"background mean {st['zero']:+.4g}" if st["zero_from"] != "0" else "0 (tick Background region for the map's own zero)"
        self.status.setText(f"{where}: {st['n_px']} px, up {st['n_up_px']}, down {st['n_down_px']}, "
                            f"<m_z> {st['mz']:+.3f}; mean {st['mean']:+.4g} (zero = {zero})")

    def _export_pushed_pdf(self):
        """The left panel as an editable PDF: pushed map (Map-view look), ROI / background outlines, um axes."""
        sel = self._selected()
        if not sel:
            return
        name, label = sel[0]
        path, _ = QFileDialog.getSaveFileName(self, "Export pushed map", f"{name}_{label}_pushed.pdf", "PDF (*.pdf)")
        if path:
            try:
                self.service.export_domain_figure(path, name, label, "input")
                self.status.setText(f"Saved {path}")
            except ValueError as exc:
                self._err(str(exc))

    def _export_pdf(self):
        sel = self._selected()
        if not sel:
            return
        name, label = sel[0]
        step = self.step.currentData()
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF", f"{name}_{label}_{step}.pdf", "PDF (*.pdf)")
        if path:
            try:
                self.service.export_domain_figure(path, name, label, step)
                if self.sizes.isChecked() and self.service.domain_result(name, label):
                    peem_plots.save_domain_sizes_pdf(os.path.splitext(path)[0] + "_sizes.pdf",
                                                     self.service.domain_areas(name, label), name)
                self.status.setText(f"Saved {path}")
            except ValueError as exc:
                self._err(str(exc))

    def _save_analysis(self):
        cur = self._current()
        if cur is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save analysis", f"{cur[0]}_{cur[1]}.nc", "Domain analysis (*.nc)")
        if path:
            try:
                self.status.setText(f"Saved {self.service.save_domain_analysis(cur[0], cur[1], path)}")
            except (ValueError, OSError) as exc:
                self._err(str(exc))

    def _load_analysis(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load analysis", "", "Domain analysis (*.nc)")
        if not path:
            return
        try:
            res = self.service.load_domain_analysis(path)
        except (ValueError, OSError) as exc:
            self._err(str(exc))
            return
        self._apply_params(res["params"])
        self.refresh(select=(res["name"], res["label"]))
        self.status.setText(f"Loaded {os.path.basename(path)} as {res['name']} · {res['label']}, with its settings; "
                            "change them and Run selected to analyse again.")

    def _apply_params(self, params: dict):
        """Set the step controls from saved settings (missing ones keep their value)."""
        for combo, key in ((self.denoise, "denoise"), (self.flake, "flake"), (self.background, "background"),
                           (self.fft_mode, "fft_mode"), (self.conn, "connectivity")):
            if key in params:
                i = combo.findData(params[key])
                if i < 0 and isinstance(params[key], (int, float, np.integer, np.floating)):
                    i = combo.findData(int(params[key]))
                if i >= 0:
                    combo.setCurrentIndex(i)
        for spin, key in ((self.tv_weight, "tv_weight"), (self.sigma, "flatten_sigma"), (self.m0, "m_threshold")):
            if key in params:
                spin.setValue(float(params[key]))
        for spin, key in ((self.fft_cut, "fft_cutoff"), (self.min_px, "min_px")):
            if key in params:
                spin.setValue(int(params[key]))
        if "fft_filter" in params:
            self.fft.setChecked(bool(int(params["fft_filter"])))

    def _export_table(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export table", "domain_analysis.csv", "CSV (*.csv)")
        if path:
            try:
                self.status.setText(f"Saved {self.service.export_domain_table(path)}")
            except ValueError as exc:
                self._err(str(exc))

    def _export_sizes(self):
        sel = self._selected()
        if not sel:
            return
        name, label = sel[0]
        path, _ = QFileDialog.getSaveFileName(self, "Export domain sizes", f"{name}_{label}_sizes.csv", "CSV (*.csv)")
        if path:
            try:
                self.status.setText(f"Saved {self.service.export_domain_areas(path, name, label)}")
            except ValueError as exc:
                self._err(str(exc))
