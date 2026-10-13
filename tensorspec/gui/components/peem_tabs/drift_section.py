"""PEEM XMCD-tab section: 3. Drift correction (all polarizations together) and manual shift.

Mixin of ``PeemXmcdPanel`` (gui/components/peem_xmcd_panel.py); it uses the panel's shared state
(service, _name, ROIs, viewer, _log / _err / _run). Moved verbatim from the former ue49_panel.py."""
from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
)


_DIVERGING = {"pcd", "pld", "pmld", "pmnld"}

from tensorspec.gui.components.peem_tabs.common import _coords_row, _dspin, _numbers
from tensorspec.plotting import peem_plots


class DriftSection:
    _BOX_METHODS = ("marquee", "roi", "ncc_roi", "igor_marquee", "igor_registration")

    def _build_drift(self, cl):
        """Controls of this section, added to the control column ``cl``."""
        # ---------------------------------------------------------------- 3. Drift
        drift = QGroupBox("3. Drift correction (all frames together)")
        dfm = QFormLayout(drift)
        info = QLabel("Every frame of both polarizations is aligned to one reference frame.")
        info.setWordWrap(True)
        dfm.addRow(info)
        self.drift_method = QComboBox()
        self.drift_method.addItem("Cross-correlation · central half", "central")
        self.drift_method.addItem("Cross-correlation · marquee (drawn box)", "marquee")
        self.drift_method.addItem("ROI template tracking (NCC, drawn box)", "roi")
        self.drift_method.addItem("NCC ROI (original TensorSpec, whole px, drawn box)", "ncc_roi")
        self.drift_method.addItem("Igor autoImageCorrelation (fixed window, whole px)", "igor_auto")
        self.drift_method.addItem("Igor driftCorrection (co-add, one shift between the polarizations)", "igor_pair")
        self.drift_method.addItem("Igor marqueeDriftCorrection (drawn box)", "igor_marquee")
        self.drift_method.addItem("Igor marqueeImageRegistration (drawn box)", "igor_registration")
        self.drift_method.setToolTip(
            "central: FFT cross-correlation of the central half (Igor default)\n"
            "marquee: FFT cross-correlation inside the box you draw (Igor marquee drift correction)\n"
            "ROI: the box in the reference frame is a template, found in every frame by normalized\n"
            "cross-correlation near its previous position (follows large drift with a small radius)\n"
            "NCC ROI: the original PEEM-panel drift (core/peem/engine.drift_correct): box vs the reference\n"
            "    frame, whole pixels, brute-force search within the radius (slow for large radii)\n"
            "Igor autoImageCorrelation: each frame vs the reference, fixed window, whole pixels\n"
            "Igor driftCorrection: co-add each polarization WITHOUT shifting (autoCoaddXMCD), then ONE\n"
            "    shift of the second polarization onto the first; for stable acquisitions only\n"
            "Igor marqueeDriftCorrection: box correlation, aligned frames added to the reference,\n"
            "    up to 3 passes, optional x N interpolation (1/N px steps)\n"
            "Igor marqueeImageRegistration: least squares in the box; translation + rotation / skew / scale")
        self.drift_method.currentIndexChanged.connect(self._on_drift_method)
        dfm.addRow("Method:", self.drift_method)
        b = QPushButton("Draw box (marquee / ROI)")
        b.setToolTip("Drag a rectangle over sharp, non-magnetic features (flake edges, particles)")
        b.clicked.connect(self._start_drift_rect)
        dfm.addRow(b)
        r, self.box_coords = _coords_row("x0, y0, x1, y1 (px)", self._set_box_coords)
        dfm.addRow("Box:", r)
        self.drift_filter = QComboBox()
        self.drift_filter.addItem("the images as they are", "none")
        self.drift_filter.addItem("edges, smoothed (best for opposite polarizations)", "smooth_edges")
        self.drift_filter.addItem("fine detail only (high-pass)", "highpass")
        self.drift_filter.addItem("edges, not smoothed", "find_edges")
        self.drift_filter.addItem("sharpened images", "sharpen")
        self.drift_filter.setToolTip(
            "What the frames are matched on to find their shifts. Only copies used for the matching are\n"
            "filtered; the frames themselves are shifted unfiltered.\n"
            "Opposite polarizations show reversed magnetic contrast but the same edges, so matching on\n"
            "edges does not confuse the domain pattern with a shift.")
        dfm.addRow("Match frames on:", self.drift_filter)
        self.drift_ref = QSpinBox()
        self.drift_ref.setRange(0, 0)
        dfm.addRow("Reference frame:", self.drift_ref)
        self.drift_radius = QSpinBox()
        self.drift_radius.setRange(1, 200)
        self.drift_radius.setValue(20)
        self.drift_radius.setToolTip("ROI method: search this far around the previous frame's position")
        dfm.addRow("Search radius (px):", self.drift_radius)
        self.drift_max = QSpinBox()
        self.drift_max.setRange(0, 500)
        self.drift_max.setValue(0)
        self.drift_max.setToolTip("Cross-correlation: ignore peaks farther than this from zero shift")
        dfm.addRow("Max shift (px, 0 = off):", self.drift_max)
        self.drift_interp = QSpinBox()
        self.drift_interp.setRange(1, 16)
        self.drift_interp.setValue(4)
        self.drift_interp.setToolTip("Igor marqueeDriftCorrection: interpolate by this factor (1 = off) -> 1/N px steps")
        dfm.addRow("Interpolate (×, 1 = off):", self.drift_interp)
        self.drift_transform = QComboBox()
        for label, val in (("translation", "translation"), ("+ rotation", "rotation"), ("+ skew", "skew"),
                           ("+ isotropic scaling", "isoscale")):
            self.drift_transform.addItem(label, val)
        self.drift_transform.setToolTip("Igor marqueeImageRegistration: rotation, skew and scaling are mutually exclusive")
        dfm.addRow("Registration:", self.drift_transform)
        self.chk_skip_pre = QCheckBox("Ignore pre-edge frames (name contains \"pre\")")
        dfm.addRow(self.chk_skip_pre)
        r = QHBoxLayout()
        b = QPushButton("Run drift correction")
        b.clicked.connect(self._drift_step)
        r.addWidget(b)
        b = QPushButton("Show drift")
        b.clicked.connect(self._show_drift)
        r.addWidget(b)
        dfm.addRow(r)
        self.roi_label = QLabel("Box: none")
        dfm.addRow(self.roi_label)
        r = QHBoxLayout()
        self.man_dy = _dspin(-500, 500, 0.0, 0.25, 2)
        self.man_dx = _dspin(-500, 500, 0.0, 0.25, 2)
        r.addWidget(QLabel("dy"))
        r.addWidget(self.man_dy)
        r.addWidget(QLabel("dx (px)"))
        r.addWidget(self.man_dx)
        b = QPushButton("Shift current frame")
        b.setToolTip("Igor manualDrift: move the shown drift-corrected frame by (dy, dx) px")
        b.clicked.connect(self._manual_shift)
        r.addWidget(b)
        dfm.addRow("Manual:", r)
        cl.addWidget(drift)

    def _drift_roi(self) -> dict | None:
        """Drawn box when the method uses it (marquee / ROI), else None = central half."""
        combo = getattr(self, "drift_method", None)
        if combo is None or combo.currentData() not in self._BOX_METHODS:
            return None
        return self._reg_roi

    def _on_drift_method(self, *_):
        m = self.drift_method.currentData()
        self.drift_radius.setEnabled(m in ("roi", "ncc_roi"))
        self.drift_max.setEnabled(m in ("central", "marquee"))
        self.drift_interp.setEnabled(m == "igor_marquee")
        self.drift_transform.setEnabled(m == "igor_registration")
        self.chk_skip_pre.setEnabled(m in ("igor_marquee", "igor_registration"))
        self.drift_ref.setEnabled(m != "igor_pair")
        if m in self._BOX_METHODS and self._reg_roi is None:
            self._log("Click 'Draw box (marquee / ROI)' and drag a box over sharp features.")
        self._update_roi_label()
        self._refresh_view()

    def _start_drift_rect(self):
        self._start_tool("rect", self._on_drift_rect, "magenta",
                         "Drag a rectangle over sharp, non-magnetic features for the drift correction.")

    def _set_box_coords(self):
        try:
            x0, y0, x1, y1 = _numbers(self.box_coords.text(), 4)
        except ValueError as exc:
            self._err(str(exc))
            return
        if self._current_image() is None:
            self._err("Load data first.")
            return
        self._on_drift_rect({"kind": "rect", "x0": int(min(x0, x1)), "x1": int(max(x0, x1)),
                             "y0": int(min(y0, y1)), "y1": int(max(y0, y1))})

    def _on_drift_rect(self, sh: dict):
        img = self._current_image()
        ny, nx = img.shape[-2:]
        x0, x1, y0, y1 = max(sh["x0"], 0), min(sh["x1"], nx), max(sh["y0"], 0), min(sh["y1"], ny)
        if x1 - x0 < 16 or y1 - y0 < 16:
            self._err("Box is too small (needs at least 16 x 16 px).")
            return
        self._reg_roi = {"kind": "rect", "x0": int(x0), "x1": int(x1), "y0": int(y0), "y1": int(y1)}
        self.box_coords.setText(f"{int(x0)}, {int(y0)}, {int(x1)}, {int(y1)}")
        if self.drift_method.currentData() == "central":
            self.drift_method.setCurrentIndex(self.drift_method.findData("marquee"))
        self._log(f"Box: x {x0}-{x1}, y {y0}-{y1} px. Click 'Run drift correction'.")
        self._update_roi_label()
        self._refresh_view()

    def _show_drift(self):
        name = self._require()
        if not name:
            return
        kind = self._view_kind()
        have = self.service.list_products(name)
        if kind not in ("processed", "stack", "pcd", "pld"):
            kind = "processed" if self.service.get_drift(name, "processed") is not None else \
                next((k for k in ("stack", "pcd", "pld") if k in have), "processed")
        try:
            d = self.service.get_drift(name, kind)
        except Exception as exc:
            self._err(str(exc))
            return
        if d is None:
            self._err("No drift yet: click 'Run drift correction' first.")
            return
        y0, y1, x0, x1 = d["box"]
        ax = self._plot(f"drift · {d.get('method') or d['source']} · box x {x0}-{x1}, y {y0}-{y1} px")
        peem_plots.drift(ax, d)
        self.pfig.tight_layout()
        self.pcanvas.draw_idle()

    def _drift_step(self):
        name = self._require()
        if name:
            m = self.drift_method.currentData()
            if m in self._BOX_METHODS and self._reg_roi is None:
                self._err("Draw the box first: 'Draw box (marquee / ROI)', then drag over sharp features.")
                return
            self._run(self.service.drift_correct, lambda r: self._show_result(r, "processed"), name,
                      reg_roi=self._drift_roi(), method=m, filt=self.drift_filter.currentData(),
                      ref_index=self.drift_ref.value(), search_radius=self.drift_radius.value(),
                      max_shift=self.drift_max.value() or None, interp_factor=self.drift_interp.value(),
                      transform=self.drift_transform.currentData(), skip_pre=self.chk_skip_pre.isChecked(),
                      pair=self.pair_combo.currentData())

    def _manual_shift(self):
        name = self._require()
        if not name:
            return
        if self._view_kind() != "processed":
            self._err("Show Frames: 'drift-corrected frames' (Layer: Intensity) and pick the frame with its slider.")
            return
        try:
            r = self.service.manual_shift(name, self._frame(), self.man_dy.value(), self.man_dx.value())
        except ValueError as exc:
            self._err(str(exc))
            return
        self._log(json.dumps(r))
        self._frames_dirty = True
        self._refresh_views(select="processed")
