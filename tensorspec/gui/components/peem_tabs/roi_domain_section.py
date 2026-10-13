"""XPEEM panel section: 7. Map ROIs, 8. ROI statistics, 9. Domain map and TIFF export.

Mixin of ``PeemXmcdPanel`` (gui/components/peem_xmcd_panel.py); it uses the panel's shared state
(service, _name, ROIs, viewer, _log / _err / _run). Moved verbatim from the former ue49_panel.py."""
from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


_DIVERGING = {"pcd", "pld", "pmld", "pmnld"}

from tensorspec.gui.components.peem_tabs.common import _dspin


class RoiDomainSection:
    def _build_roi_domain(self, cl):
        """Controls of this section, added to the control column ``cl``."""
        # ---------------------------------------------------------------- 7. Map ROIs
        roi = QGroupBox("7. Map ROIs (statistics, background, blanking)")
        rf = QVBoxLayout(roi)
        r1 = QHBoxLayout()
        b = QPushButton("Draw ROI")
        b.clicked.connect(lambda: self._start_polygon("roi"))
        r1.addWidget(b)
        b = QPushButton("Draw BG ROI")
        b.clicked.connect(lambda: self._start_polygon("bg"))
        r1.addWidget(b)
        rf.addLayout(r1)
        r2 = QHBoxLayout()
        b = QPushButton("Auto FOV circle")
        b.clicked.connect(self._auto_fov)
        r2.addWidget(b)
        b = QPushButton("Clear ROIs")
        b.clicked.connect(self._clear_rois)
        r2.addWidget(b)
        rf.addLayout(r2)
        r3 = QHBoxLayout()
        b = QPushButton("Save ROI…")
        b.clicked.connect(self._save_roi)
        r3.addWidget(b)
        b = QPushButton("Load ROI…")
        b.clicked.connect(self._load_roi)
        r3.addWidget(b)
        rf.addLayout(r3)
        self.chk_crop = QCheckBox("Blank maps outside ROI")
        self.chk_crop.setChecked(True)
        rf.addWidget(self.chk_crop)
        self.map_roi_label = QLabel("ROI: none · BG: none")
        rf.addWidget(self.map_roi_label)
        cl.addWidget(roi)

        stats = QGroupBox("8. ROI statistics")
        sf = QFormLayout(stats)
        self.stats_kind = QComboBox()
        self.stats_kind.addItems(["pcd", "pld", "pmld", "pmnld"])
        sf.addRow("Map:", self.stats_kind)
        self.band = _dspin(0, 1, 0.0, 0.005, 4)
        sf.addRow("Gray band |v| <:", self.band)
        self.thr_edit = QLineEdit()
        self.thr_edit.setPlaceholderText("blank = ROI mean")
        sf.addRow("Up/down threshold:", self.thr_edit)
        b = QPushButton("Compute ROI stats")
        b.clicked.connect(self._roi_stats)
        sf.addRow(b)
        cl.addWidget(stats)

        dom = QGroupBox("9. Domain map (SDA)")
        mf = QFormLayout(dom)
        self.dom_source = QComboBox()
        self.dom_source.addItems(["pcd", "pmld", "pld", "pmnld"])
        mf.addRow("Source:", self.dom_source)
        self.dom_thr = _dspin(0, 1, 0.10, 0.01, 3)
        mf.addRow("Threshold t:", self.dom_thr)
        self.dom_norm = QComboBox()
        self.dom_norm.addItem("m = A / P99|A|", "p99")
        self.dom_norm.addItem("raw A", "none")
        mf.addRow("Normalize:", self.dom_norm)
        self.dom_smooth = _dspin(0, 20, 1.5, 0.5, 1)
        mf.addRow("Smooth σ (px):", self.dom_smooth)
        self.dom_flat = _dspin(0, 500, 0.0, 5, 0)
        mf.addRow("Flatten σ (px, 0=off):", self.dom_flat)
        self.dom_minpx = QSpinBox()
        self.dom_minpx.setRange(0, 10000)
        self.dom_minpx.setValue(12)
        mf.addRow("Min domain px:", self.dom_minpx)
        b = QPushButton("Compute domain map")
        b.clicked.connect(self._domain_map)
        mf.addRow(b)
        cl.addWidget(dom)

        b = QPushButton("Export shown map as TIFF…")
        b.clicked.connect(self._export)
        cl.addWidget(b)

    def _start_polygon(self, target: str):
        self._start_tool("polygon", lambda sh: self._on_polygon(target, sh), "yellow" if target == "roi" else "cyan",
                         f"Click vertices on the image; close the polygon on the first point ({target}).")

    def _on_polygon(self, target: str, sh: dict):
        roi = {"kind": "polygon", "points": [[float(x), float(y)] for x, y in sh["points"]]}
        if target == "bg":
            self._bg_roi = roi
        else:
            self._roi = roi
        self._update_roi_label()
        self._refresh_view()

    def _auto_fov(self):
        img = self._current_image()
        if img is None or self._view_kind() != "raw":
            name = self._require()
            if not name:
                return
            img = self.service.get_view_tensor(name, "raw", 0)
        self._roi = self.service.auto_fov_roi(img)
        self._update_roi_label()
        self._refresh_view()

    def _clear_rois(self):
        self._roi = self._bg_roi = self._reg_roi = None
        self.drift_method.setCurrentIndex(self.drift_method.findData("central"))
        self._update_roi_label()
        self._refresh_view()

    def _save_roi(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save ROIs", "rois.json", "JSON (*.json)")
        if path:
            with open(path, "w") as fh:
                json.dump({"roi": self._roi, "bg_roi": self._bg_roi, "reg_roi": self._reg_roi}, fh, indent=1)

    def _load_roi(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load ROIs", "", "JSON (*.json)")
        if path:
            with open(path) as fh:
                data = json.load(fh)
            self._roi, self._bg_roi = data.get("roi"), data.get("bg_roi")
            self._reg_roi = data.get("reg_roi")
            if self._reg_roi and self.drift_method.currentData() == "central":
                self.drift_method.setCurrentIndex(self.drift_method.findData("marquee"))
            self._update_roi_label()
            self._refresh_view()

    def _update_roi_label(self):
        def d(r):
            return "none" if r is None else r["kind"]
        self.map_roi_label.setText(f"ROI: {d(self._roi)} · BG: {d(self._bg_roi)}")
        b = self._reg_roi
        box = "none drawn" if b is None else f"x {b['x0']}-{b['x1']}, y {b['y0']}-{b['y1']} px"
        used = "box used" if self.drift_method.currentData() in self._BOX_METHODS else "box not used by this method"
        self.roi_label.setText(f"Box: {box} ({used})")

    def _roi_stats(self):
        name = self._require()
        if not name:
            return
        txt = self.thr_edit.text().strip()
        try:
            thr = float(txt) if txt else None
        except ValueError:
            self._err("Threshold must be a number or blank.")
            return
        self._run(self.service.roi_statistics, lambda r: self._show_result(r), name,
                  kind=self.stats_kind.currentText(), roi=self._roi, bg_roi=self._bg_roi,
                  band=self.band.value(), threshold=thr)

    def _domain_map(self):
        name = self._require()
        if name:
            self._run(self.service.compute_domain_map, lambda r: self._show_result(r, "domain_map"), name,
                      source_kind=self.dom_source.currentText(), roi=self._roi,
                      threshold=self.dom_thr.value(), normalize=self.dom_norm.currentData(),
                      min_px=self.dom_minpx.value(), smooth_sigma=self.dom_smooth.value(),
                      flatten_sigma=self.dom_flat.value() or None, bg_roi=self._bg_roi)

    def _export(self):
        name = self._require()
        kind = self._view_kind()
        if not name or kind in self._FRAME_KINDS:
            self._err("Show a computed map first (viewer Layer menu).")
            return
        var = self._view_var() or "data"
        path, _ = QFileDialog.getSaveFileName(self, "Export map", f"{name}_{kind}_{var}.tif", "TIFF (*.tif)")
        if path:
            try:
                self._log("Saved " + self.service.export_map_tiff(name, kind, path, var))
            except Exception as exc:
                self._err(str(exc))
