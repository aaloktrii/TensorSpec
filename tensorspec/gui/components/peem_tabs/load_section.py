"""PEEM XMCD-tab section: 1. Load data (UE49 / BNL UView / ALS P3B) and the load options.

Mixin of ``PeemXmcdPanel`` (gui/components/peem_xmcd_panel.py); it uses the panel's shared state
(service, _name, ROIs, viewer, _log / _err / _run). Moved verbatim from the former ue49_panel.py."""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QCheckBox,
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



class LoadSection:
    def _build_load(self, cl):
        """Controls of this section, added to the control column ``cl``."""
        # ---------------------------------------------------------------- 1. Load
        load = QGroupBox("1. Load data (UE49 · BNL UView · ALS P3B)")
        lf = QFormLayout(load)
        self.scan_list = QListWidget()
        self.scan_list.setMaximumHeight(90)
        lf.addRow(self.scan_list)
        row = QHBoxLayout()
        b_add = QPushButton("Add files…")
        b_add.setToolTip("UE49 scan .txt, UView .dat or ALS .P3B frames")
        b_add.clicked.connect(self._add_scans)
        b_dirs = QPushButton("Add folder…")
        b_dirs.setToolTip("One acquisition folder per click: UE49 image folder, UView .dat folder (one polarization, "
                          "_CL_/_CR_/_LH_/_LV_ in the name) or ALS .P3B folder (EPU ±0.9 per frame)")
        b_dirs.clicked.connect(self._add_folders)
        b_clear = QPushButton("Clear")
        b_clear.clicked.connect(self._clear_scans)
        row.addWidget(b_add)
        row.addWidget(b_dirs)
        row.addWidget(b_clear)
        lf.addRow(row)
        self.btn_load = QPushButton("Load")
        self.btn_load.clicked.connect(self._load)
        lf.addRow(self.btn_load)
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
        self.chk_tpx3 = QCheckBox("UE49 TPX3 512→514 cross correction")
        self.chk_tpx3.setChecked(True)
        self.chk_i0 = QCheckBox("I0 normalization (ALS: only if I0 passes the check)")
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
            except ValueError as exc:
                self._err(str(exc))
                continue
            self._scan_paths.append(p)
            name = os.path.basename(p) + ("/" if os.path.isdir(p) else "")
            self.scan_list.addItem(f"[{fmt}] {name}")
            self.scan_list.item(self.scan_list.count() - 1).setToolTip(p)
            self._set_format(fmt)
            self._log(f"Added [{fmt}] {p}")
        self._update_load_status()

    def _update_load_status(self):
        n = len(self._scan_paths)
        if not n:
            self.load_status.setText("Add files or a folder, then click Load.")
            return
        missing = [os.path.basename(str(q)) for q in map(peem_beamline_loaders.partner_folder, self._scan_paths)
                   if q is not None and os.path.normpath(str(q)) not in self._scan_paths]
        msg = f"{n} item(s) ready. Click Load."
        if missing:
            msg += f" Partner not added yet: {', '.join(missing)} (Add folder… again for XMCD/XLD)."
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
            self, "PEEM beamline files", self._start_dir(),
            "PEEM beamline data (*.txt *.dat *.P3B *.p3b);;UE49 scan tables (*.txt);;UView (*.dat);;ALS PEEM-3 (*.P3B *.p3b)")
        self._add_paths(paths)

    def _add_folders(self):
        """One acquisition folder per click (native picker; multi-select folder dialogs are unreliable on macOS)."""
        path = QFileDialog.getExistingDirectory(self, "Acquisition folder (add one at a time)", self._start_dir())
        if not path:
            return
        self._add_paths([path])
        partner = self._partner_folder(path)
        if partner and partner not in self._scan_paths:
            self._log(f"Partner folder found: {os.path.basename(partner)}. Click 'Add folder…' again to add it.")

    @staticmethod
    def _partner_folder(path: str) -> str | None:
        partner = peem_beamline_loaders.partner_folder(path)
        return str(partner) if partner else None

    def _clear_scans(self):
        """Clear the file list AND unload the current dataset (frames, drift, stack, maps) and the view."""
        if self._worker is not None and self._worker.isRunning():
            self._err("Busy: wait for the current operation to finish.")
            return
        if self._name:
            self.service.unload(self._name)
            self._log(f"Unloaded '{self._name}'.")
        self._name = None
        self._scan_paths.clear()
        self.scan_list.clear()
        self.plot_box.setVisible(False)
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        self.view_combo.blockSignals(False)
        self.viewer._clear_tree()
        self.viewer.views.clear()
        self.viewer.tensor_data = None
        self._viewer_key = None
        self._update_load_status()

    def _load(self):
        if not self._scan_paths:
            self._err("Add files or folders first (UE49 X_0002.txt, UView or P3B acquisition folders).")
            return
        try:
            self.service.set_detector_maps(self.dead_edit.text().strip() or None,
                                           self.norm_edit.text().strip() or None)
        except ValueError as exc:
            self._err(str(exc))
            return
        i0 = self.chk_i0.isChecked() if self.chk_i0.isEnabled() else None
        self.load_status.setText(f"Loading {len(self._scan_paths)} item(s)… (online-only Dropbox files download first)")
        self._log("Loading: " + "; ".join(self._scan_paths))
        self._run(self.service.load_beamline, self._on_loaded, list(self._scan_paths),
                  None, i0_normalize=i0,  # dataset name is automatic
                  tpx3=self.chk_tpx3.isChecked(), drop_bad_frames=self.chk_bad.isChecked())

    def _on_loaded(self, res: dict):
        self._name = res["name"]
        self.drift_ref.setRange(0, max(0, res["shape"][0] - 1))
        pc = ", ".join(f"{(res.get('pol_source') or {}).get(k, k)} {v}" for k, v in res["pol_counts"].items())
        self.load_status.setText(f"Loaded '{res['name']}': {res['shape'][0]} frames ({pc}).")
        conv = res.get("default_convention")  # the beamline's own sign (UE49 CM-CP; BNL CL-CR; ALS +0.9 - -0.9)
        if conv and self.conv_combo.findData(conv) >= 0:
            self.conv_combo.setCurrentIndex(self.conv_combo.findData(conv))
        if res.get("energy_range") and self.e_a.value() == 0:
            lo, hi = res["energy_range"]
            self.e_a.setValue(lo)
            self.e_b.setValue(hi)
        self._show_result(res, show_kind="raw")
