"""PEEM analysis panel — direct core access via PeemService."""
from __future__ import annotations

import os
from pathlib import Path

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

        map_group = QGroupBox("4. Dichroism map (Frame viewer)")
        map_form = QFormLayout(map_group)
        self.map_register = QComboBox()
        self.map_register.addItem("Register polarizations (recommended)", True)
        self.map_register.addItem("No extra registration", False)
        map_form.addRow("Align:", self.map_register)
        map_btns = QHBoxLayout()
        self.btn_make_pcd = QPushButton("Make PCD (XMCD)")
        self.btn_make_pcd.setToolTip(
            "CP/CM → /analysis/pcd (LH/LV: Make PLD). Prefer Separate first, or beamline load with pol tags."
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


        # 5-8: calculate + analyse (asymmetry map, image analysis, ROIs / domains); background and
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

    def _make_pcd(self):
        name = self._require_dataset()
        if not name:
            return
        try:
            result = self.service.make_pcd_map(
                name, register=bool(self.map_register.currentData())
            )
            self.service.map_to_tools(name, "pcd")  # also in the Map tab, for sections 6-8
            self._refresh_meta()
            self.map_tools.refresh(select=("pcd", "data"))
            idx = self.node_combo.findData(result["display_node"])
            if idx >= 0:
                self.node_combo.setCurrentIndex(idx)
            self.view_tabs.setCurrentIndex(0)
            self.status_label.setText(
                f"PCD map ({result['formula']}, source={result['source']}, "
                f"mean={result['mean']:.4g}) | also in the Map tab for 6-8"
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
            self.service.map_to_tools(name, "pld")  # also in the Map tab, for sections 6-8
            self._refresh_meta()
            self.map_tools.refresh(select=("pld", "data"))
            idx = self.node_combo.findData(result["display_node"])
            if idx >= 0:
                self.node_combo.setCurrentIndex(idx)
            self.view_tabs.setCurrentIndex(0)
            self.status_label.setText(
                f"PLD map ({result['formula']}, source={result['source']}, "
                f"mean={result['mean']:.4g}) | also in the Map tab for 6-8"
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
