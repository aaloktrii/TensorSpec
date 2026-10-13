"""PEEM XPEEM-map section: 1. Load data (several files / folders at once) and the load options.

Mixin of ``PeemXmcdPanel`` (gui/components/peem_xmcd_panel.py); it uses the panel's shared state
(service, _name, ROIs, viewer, _log / _err / _run). Moved verbatim from the former ue49_panel.py."""
from __future__ import annotations

import os

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListView,
    QTreeView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from tensorspec.core.io import peem_beamline_loaders

_DIVERGING = {"pcd", "pld", "pmld", "pmnld"}


class _DropPaths(QObject):
    """Files / folders dropped from Finder onto the load list -> callback(paths)."""

    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self._callback = callback

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t in (QEvent.DragEnter, QEvent.DragMove) and ev.mimeData().hasUrls():
            ev.acceptProposedAction()
            return True
        if t == QEvent.Drop and ev.mimeData().hasUrls():
            ev.acceptProposedAction()
            self._callback([u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()])
            return True
        return False


class _DeleteKey(QObject):
    """Delete / Backspace on the load list -> callback()."""

    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self._callback = callback

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.KeyPress and ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._callback()
            return True
        return False


class LoadSection:
    def _build_load(self, cl):
        """Controls of this section, added to the control column ``cl``."""
        # ---------------------------------------------------------------- 1. Load
        load = QGroupBox("1. Load data")
        lf = QFormLayout(load)
        self.scan_list = QListWidget()
        self.scan_list.setMaximumHeight(90)
        self.scan_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.scan_list.setToolTip("Drop files / folders here from Finder (several at once)")
        self.scan_list.setAcceptDrops(True)
        self.scan_list.viewport().setAcceptDrops(True)
        self._drop_filter = _DropPaths(self._add_any, self)
        self.scan_list.viewport().installEventFilter(self._drop_filter)
        lf.addRow(self.scan_list)
        self.pol_combo = QComboBox()
        for label, tag in (("auto (from the data)", ""), ("circular + (CP)", "CP"),
                           ("circular − (CM)", "CM"), ("linear horizontal (LH)", "LH"),
                           ("linear vertical (LV)", "LV")):
            self.pol_combo.addItem(label, tag)
        self.pol_combo.setToolTip("Polarization of the selected item(s). Set it when the data do not record it.")
        self.pol_combo.activated.connect(self._set_item_pol)
        self.scan_list.itemSelectionChanged.connect(self._show_item_pol)
        lf.addRow("Polarization:", self.pol_combo)
        row = QHBoxLayout()
        b_add = QPushButton("Add files…")
        b_add.setToolTip("Scan tables (.txt), .dat or .P3B images")
        b_add.clicked.connect(self._add_scans)
        b_dirs = QPushButton("Add folders…")
        b_dirs.setToolTip("Pick one acquisition, or a folder that holds several and tick the ones to add")
        b_dirs.clicked.connect(self._add_folders)
        b_remove = QPushButton("Remove selected")
        b_remove.setToolTip("Take the selected item(s) off the list (Delete / Backspace key too). "
                            "Click Load again to load the rest.")
        b_remove.clicked.connect(self._remove_selected)
        b_clear = QPushButton("Clear all")
        b_clear.setToolTip("Empty the list and unload the current data")
        b_clear.clicked.connect(self._clear_scans)
        row.addWidget(b_add)
        row.addWidget(b_dirs)
        lf.addRow(row)
        row = QHBoxLayout()
        row.addWidget(b_remove)
        row.addWidget(b_clear)
        lf.addRow(row)
        self._delete_key = _DeleteKey(self._remove_selected, self)
        self.scan_list.installEventFilter(self._delete_key)
        row = QHBoxLayout()
        self.btn_load = QPushButton("Load all as one dataset")
        self.btn_load.setToolTip("Every item in the list as one dataset")
        self.btn_load.clicked.connect(self._load)
        self.btn_load_sel = QPushButton("Load selected as one dataset")
        self.btn_load_sel.setToolTip("Only the selected items (e.g. one CP / CM pair) as one dataset")
        self.btn_load_sel.clicked.connect(lambda: self._load(selected=True))
        self.btn_load_sel.setEnabled(False)
        row.addWidget(self.btn_load)
        row.addWidget(self.btn_load_sel)
        lf.addRow(row)
        self.btn_load_pairs = QPushButton("Load each pair separately")
        self.btn_load_pairs.setToolTip("One dataset per polarization pair (e.g. CL/CR at each angle), all kept "
                                       "loaded: pick one under Dataset, analyse it, save, unload, pick the next")
        self.btn_load_pairs.clicked.connect(self._load_pairs)
        lf.addRow(self.btn_load_pairs)
        self.scan_list.itemSelectionChanged.connect(self._update_load_button)

        row = QHBoxLayout()
        self.dataset_combo = QComboBox()
        self.dataset_combo.setToolTip("The dataset every step below works on")
        self.dataset_combo.activated.connect(lambda *_: self._switch_dataset(self.dataset_combo.currentData()))
        b_save = QPushButton("Save results…")
        b_save.setToolTip("Every map (TIFF + PNG), drift.csv, history.txt and summary.json of this dataset "
                          "into <folder>/<dataset>/")
        b_save.clicked.connect(self._save_results)
        b_unload = QPushButton("Unload")
        b_unload.setToolTip("Remove this dataset from memory (save its results first)")
        b_unload.clicked.connect(self._unload_current)
        row.addWidget(self.dataset_combo, 1)
        row.addWidget(b_save)
        row.addWidget(b_unload)
        lf.addRow("Dataset:", row)
        self.load_status = QLabel("Add files or a folder, then click Load.")
        self.load_status.setWordWrap(True)
        lf.addRow(self.load_status)
        self.scan_list.itemDoubleClicked.connect(lambda *_: self._load())

        det = QGroupBox("Load options: detector / I0 corrections (set from the format)")
        det.setCheckable(True)
        det.setChecked(False)
        det_body = QWidget()
        df = QFormLayout(det_body)
        df.setContentsMargins(0, 0, 0, 0)
        self.dead_edit = QLineEdit()
        self.norm_edit = QLineEdit()
        for edit, label in ((self.dead_edit, "Dead pixel"), (self.norm_edit, "Norm image")):
            r = QHBoxLayout()
            r.addWidget(edit)
            b = QPushButton("…")
            b.setFixedWidth(28)
            b.clicked.connect(lambda _=False, e=edit: self._pick_tif(e))
            r.addWidget(b)
            df.addRow(label + ":", r)
        self.chk_tpx3 = QCheckBox("TPX3 detector 512→514 cross correction")
        self.chk_tpx3.setChecked(True)
        self.chk_i0 = QCheckBox("I0 normalization")
        self.chk_i0.setChecked(True)
        self.chk_bad = QCheckBox("Drop I0-spike / unstable-beam frames")
        df.addRow(self.chk_tpx3)
        df.addRow(self.chk_i0)
        df.addRow(self.chk_bad)
        QVBoxLayout(det).addWidget(det_body)
        det_body.setVisible(False)
        det.toggled.connect(det_body.setVisible)  # collapsed by default; the settings apply either way
        lf.addRow(det)
        cl.addWidget(load)

    def _pick_tif(self, edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Detector map", os.path.expanduser("~"), "TIFF (*.tif *.tiff)")
        if path:
            edit.setText(path)

    def _start_dir(self) -> str:
        """Reopen next to the last added item (its parent folder)."""
        if not self._scan_paths:
            return os.path.expanduser("~")
        return os.path.dirname(self._scan_paths[-1].rstrip("/"))

    def _add_paths(self, paths):
        for p in paths:
            p = os.path.normpath(p)  # dialogs can return a trailing "/"
            if p in self._scan_paths:
                continue
            try:
                fmt, _ = peem_beamline_loaders.detect_format(p)
                self._scan_detected[p] = peem_beamline_loaders.detect_pol(p)
            except ValueError as exc:
                self._err(str(exc))
                continue
            self._scan_paths.append(p)
            self.scan_list.addItem("")
            self.scan_list.item(self.scan_list.count() - 1).setToolTip(p)
            self._item_text(len(self._scan_paths) - 1, fmt)
            self._set_format(fmt)
            self._log(f"Added [{fmt}] {p}")
        self._update_load_status()

    def _item_text(self, i: int, fmt: str | None = None):
        p = self._scan_paths[i]
        item = self.scan_list.item(i)
        fmt = fmt or item.text()[1:].split("]")[0]
        tag, src = self._scan_detected.get(p, (None, ""))
        if p in self._scan_pol:
            pol = f"{self._scan_pol[p]} (set)"
        else:
            pol = f"{tag} ({src})" if tag else "? — set the polarization"
        item.setText(f"[{fmt}] {os.path.basename(p)}{'/' if os.path.isdir(p) else ''} · {pol}")

    def _show_item_pol(self):
        rows = [self.scan_list.row(it) for it in self.scan_list.selectedItems()]
        tags = {self._scan_pol.get(self._scan_paths[r], "") for r in rows}
        self.pol_combo.setCurrentIndex(max(0, self.pol_combo.findData(tags.pop())) if len(tags) == 1 else 0)

    def _set_item_pol(self):
        rows = [self.scan_list.row(it) for it in self.scan_list.selectedItems()]
        if not rows:
            self._err("Select an item in the list first, then choose its polarization.")
            return
        tag = self.pol_combo.currentData()
        for r in rows:
            p = self._scan_paths[r]
            if tag and self._scan_detected.get(p, ("",))[0] == "per frame":
                self._err(f"{os.path.basename(p)}: these frames record their polarization; not changed.")
                continue
            if tag:
                self._scan_pol[p] = tag
            else:
                self._scan_pol.pop(p, None)
            self._item_text(r)
        self._update_load_status()

    def _unset_pol(self) -> list[str]:
        return [os.path.basename(p) for p in self._scan_paths
                if p not in self._scan_pol and not self._scan_detected.get(p, (None,))[0]]

    def _update_load_status(self):
        n = len(self._scan_paths)
        if not n:
            self.load_status.setText("Add files or a folder, then click Load.")
            return
        unset = self._unset_pol()
        if unset:
            self.load_status.setText(f"Set the polarization of: {', '.join(unset)} (select it, then choose "
                                     "Polarization). The data do not record it.")
            return
        missing = [os.path.basename(str(q)) for q in map(peem_beamline_loaders.partner_folder, self._scan_paths)
                   if q is not None and os.path.normpath(str(q)) not in self._scan_paths]
        msg = f"{n} item(s) ready. Click Load."
        if missing:
            msg += f" Partner not added yet: {', '.join(missing)}."
        self.load_status.setText(msg)

    def _set_format(self, fmt: str):
        """Enable the corrections that exist for this format, with its default I0 choice."""
        ue49 = fmt == "ue49"
        for w in (self.chk_tpx3, self.chk_bad, self.dead_edit, self.norm_edit):
            w.setEnabled(ue49)
        self.chk_i0.setEnabled(fmt != "uview")  # UView files carry no I0
        self.chk_i0.setChecked(ue49)  # ALS: off by default (I0 did not track the image on 2021-06-26)

    def _add_scans(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "PEEM data", self._start_dir(), "PEEM data (*.txt *.dat *.P3B *.p3b)")
        self._add_paths(paths)

    def _add_folders(self):
        """Pick one or several folders (⌘ / ⇧-click): acquisitions are added directly; a folder of
        acquisitions opens a checklist of everything inside (all ticked)."""
        paths = self._pick_folders()
        if paths:
            self._add_any(paths)

    def _pick_folders(self) -> list[str]:
        """Qt folder dialog with multi-selection. Its own Choose button refuses several folders, so
        Choose takes the selected rows directly (macOS native dialogs pick one folder only)."""
        dlg = QFileDialog(self, "Select folders (⌘ / ⇧-click for several, or a parent folder)", self._start_dir())
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        views = [v for v in (dlg.findChild(QListView, "listView"), dlg.findChild(QTreeView, "treeView")) if v is not None]
        for v in views:
            v.setSelectionMode(QAbstractItemView.ExtendedSelection)
        chosen: list[str] = []

        def take():
            paths: list[str] = []
            here = dlg.directory().absolutePath()
            for v in views:
                if v.selectionModel() is None:
                    continue
                for idx in v.selectionModel().selectedRows():
                    p = os.path.normpath(os.path.join(here, str(idx.data())))  # selected rows of the shown folder
                    if os.path.isdir(p) and p not in paths:
                        paths.append(p)
            chosen[:] = paths or [p for p in dlg.selectedFiles() if os.path.isdir(p)]
            dlg.done(QDialog.Accepted)

        box = dlg.findChild(QDialogButtonBox)
        if box is not None:
            box.accepted.disconnect()
            box.accepted.connect(take)
        self._folder_dialog = dlg  # for tests
        if self._exec_dialog(dlg) != QDialog.Accepted:
            return []
        return chosen

    def _add_any(self, paths):
        """Add files / folders; folders that hold several acquisitions open the checklist."""
        direct = []
        for p in map(os.path.normpath, paths):
            if os.path.isdir(p) and peem_beamline_loaders.is_container(p):
                found = peem_beamline_loaders.find_acquisitions(p)
                if not found:
                    self._err(f"{os.path.basename(p)}: no loadable acquisitions inside.")
                    continue
                direct += self._choose_acquisitions(p, found)
            else:
                direct.append(p)
        if direct:
            self._add_paths(direct)

    def _choose_acquisitions(self, root: str, found) -> list[str]:
        """Checklist of the acquisitions inside ``root``; returns the ticked paths."""
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Add from {os.path.basename(root)}")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(f"{len(found)} acquisition(s) found, all ticked. Untick the ones not to add "
                             "(or select several and press Space / Tick selected)."))
        lst = QListWidget()
        lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for fmt, p in found:
            rel = os.path.relpath(str(p), root)
            lst.addItem(f"[{fmt}] {rel}")
            it = lst.item(lst.count() - 1)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)  # the whole folder by default; untick what is not wanted
            it.setData(Qt.UserRole, str(p))
        lst.setMinimumSize(560, 360)

        def toggle_selected():
            sel = lst.selectedItems()
            on = any(i.checkState() != Qt.Checked for i in sel)
            for i in sel:
                i.setCheckState(Qt.Checked if on else Qt.Unchecked)

        lst.itemActivated.connect(lambda _it: toggle_selected())
        lay.addWidget(lst)
        row = QHBoxLayout()
        for label, state in (("Tick all", Qt.Checked), ("Tick none", Qt.Unchecked)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, s=state: [lst.item(i).setCheckState(s) for i in range(lst.count())])
            row.addWidget(b)
        b = QPushButton("Tick selected")
        b.clicked.connect(toggle_selected)
        row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Add ticked")
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if self._exec_dialog(dlg) != QDialog.Accepted:
            return []
        return [lst.item(i).data(Qt.UserRole) for i in range(lst.count()) if lst.item(i).checkState() == Qt.Checked]

    def _exec_dialog(self, dlg) -> int:
        return dlg.exec()

    def _remove_selected(self):
        """Take the selected items off the list; the loaded data stay until Load is clicked again."""
        rows = sorted({self.scan_list.row(it) for it in self.scan_list.selectedItems()}, reverse=True)
        if not rows:
            self._err("Select the item(s) to remove in the list first.")
            return
        for r in rows:
            p = self._scan_paths.pop(r)
            self._scan_pol.pop(p, None)
            self._scan_detected.pop(p, None)
            self.scan_list.takeItem(r)
            self._log(f"Removed {p}")
        self._update_load_status()
        if self._name and self._scan_paths:
            self.load_status.setText(self.load_status.text() + " The data shown are still the previous load.")

    def _clear_scans(self):
        """Clear the file list AND unload the current dataset (frames, drift, stack, maps) and the view."""
        if self._worker is not None and self._worker.isRunning():
            self._err("Busy: wait for the current operation to finish.")
            return
        for name in self.service.loaded_datasets():
            self.service.unload(name)
            self._log(f"Unloaded '{name}'.")
        self._name = None
        self._scan_paths.clear()
        self._scan_pol.clear()
        self._scan_detected.clear()
        self.scan_list.clear()
        self._clear_view()
        self._fill_datasets()
        self._update_load_status()

    def _clear_view(self):
        self.plot_box.setVisible(False)
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        self.view_combo.blockSignals(False)
        self.viewer._clear_tree()
        self.viewer.views.clear()
        self.viewer.tensor_data = None
        self._viewer_key = None
        self._update_load_status()

    def _load(self, selected: bool = False):
        if not self._scan_paths:
            self._err("Add files or folders first.")
            return
        if not self._ready_to_load():
            return
        paths = self._selected_paths() if selected else list(self._scan_paths)
        if not paths:
            self._err("Select the item(s) to load in the list first.")
            return
        self.load_status.setText(f"Loading {len(paths)} item(s)… (online-only Dropbox files download first)")
        self._log("Loading: " + "; ".join(paths))
        self._run(self.service.load_beamline, self._on_loaded, paths, None,  # dataset name is automatic
                  pol={p: t for p, t in self._scan_pol.items() if p in paths} or None, **self._load_kw())

    def _load_kw(self) -> dict:
        return {"i0_normalize": self.chk_i0.isChecked() if self.chk_i0.isEnabled() else None,
                "tpx3": self.chk_tpx3.isChecked(), "drop_bad_frames": self.chk_bad.isChecked()}

    def _selected_paths(self) -> list[str]:
        rows = sorted(self.scan_list.row(it) for it in self.scan_list.selectedItems())
        return [self._scan_paths[r] for r in rows]

    def _update_load_button(self):
        n = len(self._selected_paths())
        self.btn_load_sel.setEnabled(n > 0)
        self.btn_load_sel.setText(f"Load selected ({n}) as one dataset" if n else "Load selected as one dataset")

    def _load_pairs(self):
        if not self._scan_paths:
            self._err("Add files or folders first.")
            return
        if not self._ready_to_load():
            return
        self.load_status.setText(f"Loading {len(self._scan_paths)} item(s), one dataset per pair…")
        self._log("Loading pairs: " + "; ".join(self._scan_paths))
        self._run(self.service.load_pairs, self._on_loaded_pairs, list(self._scan_paths),
                  pol=dict(self._scan_pol) or None, **self._load_kw())

    def _on_loaded_pairs(self, results: list[dict]):
        for res in results[1:]:
            self._log(f"Loaded '{res['name']}': {res['shape'][0]} frames")
        self._on_loaded(results[0])
        names = ", ".join(r["name"] for r in results)
        self.load_status.setText(f"Loaded {len(results)} datasets: {names}. Pick one under Dataset; "
                                 "Save results, then Unload it before the next.")

    def _ready_to_load(self) -> bool:
        try:
            self.service.set_detector_maps(self.dead_edit.text().strip() or None,
                                           self.norm_edit.text().strip() or None)
        except ValueError as exc:
            self._err(str(exc))
            return False
        unset = self._unset_pol()
        if unset:
            self._err(f"Set the polarization of {', '.join(unset)} first: select it in the list and choose "
                      "Polarization (its files do not record it).")
            return False
        return True

    # ------------------------------------------------------------- datasets
    def _fill_datasets(self, select: str | None = None):
        names = self.service.loaded_datasets()
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        for n in names:
            self.dataset_combo.addItem(n, n)
        if not names:
            self.dataset_combo.addItem("(none loaded)", None)
        i = self.dataset_combo.findData(select or self._name)
        self.dataset_combo.setCurrentIndex(max(0, i))
        self.dataset_combo.blockSignals(False)

    def _switch_dataset(self, name: str | None):
        """Make ``name`` the dataset every step works on and show it."""
        if not name or name == self._name:
            return
        if self._worker is not None and self._worker.isRunning():
            self._err("Busy: wait for the current operation to finish.")
            self._fill_datasets()
            return
        self._name = name
        meta = self.service.xmcd_meta(name)
        self.drift_ref.setRange(0, max(0, int(meta.get("n_frames") or 1) - 1))
        self._viewer_key = None
        self._fill_show(self.stack_out, self.service.stack_outputs(name), "Show: (after Stack)")
        self._refresh_asym_outputs()
        products = self.service.list_products(name)
        self._refresh_views(select="pcd" if "pcd" in products else "pld" if "pld" in products else "raw")
        self._fill_datasets(name)
        self.load_status.setText(f"Working on '{name}'.")
        self._log(f"Dataset: {name}")

    def _save_results(self):
        name = self._require()
        if not name:
            return
        folder = QFileDialog.getExistingDirectory(self, f"Save the results of '{name}' into (a sub-folder "
                                                  f"'{name}' is made)", self._start_dir())
        if folder:
            self._run(self.service.save_results, lambda r: (
                self._log(f"Saved {len(r['files'])} files to {r['folder']}"),
                self.load_status.setText(f"Saved the results of '{name}' to {r['folder']}.")), name, folder)

    def _unload_current(self):
        name = self._require()
        if not name:
            return
        if self._worker is not None and self._worker.isRunning():
            self._err("Busy: wait for the current operation to finish.")
            return
        self.service.unload(name)
        self._log(f"Unloaded '{name}'.")
        rest = self.service.loaded_datasets()
        self._name = None
        if rest:
            self._switch_dataset(rest[0])
        else:
            self._clear_view()
            self._fill_datasets()
            self.load_status.setText(f"Unloaded '{name}'. Nothing loaded.")

    def _on_loaded(self, res: dict):
        # regions of the previous dataset (or set before loading) carry over when the new one has none
        carry = self.service.get_rois(self._name) if self._name else dict(self._pending_rois)
        self._name = res["name"]
        have = self.service.get_rois(self._name)
        for which, roi in carry.items():
            if roi is not None and have.get(which) is None:
                self.service.set_roi(self._name, which, roi)
        self._pending_rois.clear()
        self.drift_ref.setRange(0, max(0, res["shape"][0] - 1))
        pc = ", ".join(f"{k} {v}" for k, v in res["pol_counts"].items())
        self.load_status.setText(f"Loaded '{res['name']}': {res['shape'][0]} frames ({pc}).")
        if res.get("energy_range") and self.e_a.value() == 0:
            lo, hi = res["energy_range"]
            self.e_a.setValue(lo)
            self.e_b.setValue(hi)
        self._fill_show(self.stack_out, self.service.stack_outputs(self._name), "Show: (after Stack)")
        self._refresh_asym_outputs()
        self._fill_datasets(self._name)
        self._show_result(res, show_kind="raw")
