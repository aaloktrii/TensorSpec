"""PEEM XMCD-tab section: 6. Image analysis on the shown image (SPEEM Igor ports).

Mixin of ``PeemXmcdPanel`` (gui/components/peem_xmcd_panel.py); it uses the panel's shared state
(service, _name, ROIs, viewer, _log / _err / _run). Moved verbatim from the former ue49_panel.py."""
from __future__ import annotations

import json

import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


_DIVERGING = {"pcd", "pld", "pmld", "pmnld"}

from tensorspec.gui.components.peem_tabs.common import _coords_row, _dspin, _numbers
from tensorspec.plotting import peem_plots


class AnalysisSection:
    def _build_analysis(self, cl):
        """Controls of this section, added to the control column ``cl``."""
        # ---------------------------------------------------------------- 6. Image analysis
        ana = QGroupBox("6. Image analysis (on the shown image; tick to show)")
        ana.setCheckable(True)
        ana.setChecked(False)
        ana_body = QWidget()
        af = QFormLayout(ana_body)
        af.setContentsMargins(0, 0, 0, 0)
        r = QHBoxLayout()
        b = QPushButton("Draw line (click A, then B)")
        b.clicked.connect(self._start_line)
        r.addWidget(b)
        self.line_hw = QSpinBox()
        self.line_hw.setRange(0, 100)
        self.line_hw.setValue(3)
        self.line_hw.setToolTip("Profile averaged over 2 x half-width + 1 parallel lines (Igor ImageLineProfile width)")
        r.addWidget(QLabel("half-width (px)"))
        r.addWidget(self.line_hw)
        af.addRow(r)
        self.line_label = QLabel("Line: none")
        af.addRow(self.line_label)
        r, self.line_coords = _coords_row("x1, y1, x2, y2 (px)", self._set_line_coords)
        af.addRow("Line A→B:", r)
        r = QHBoxLayout()
        for label, fn, tip in (("Profile", self._an_profile, "Igor LineProfile"),
                               ("Gauss fit", lambda: self._an_fit(False), "Igor getGaussProfile: width (FWHM) of a peak/stripe"),
                               ("Edge fit (d/dx)", lambda: self._an_fit(True), "Igor getDiffGaussProfile: wall / edge width"),
                               ("Profile vs frame", self._an_stack_profile, "Igor displayLines / stackLines")):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            r.addWidget(b)
        af.addRow(r)
        r = QHBoxLayout()
        b = QPushButton("Histogram")
        b.setToolTip("Igor calcFrameHistogram: 255 bins, inside the map ROI (section 7) if drawn")
        b.clicked.connect(self._an_histogram)
        r.addWidget(b)
        self.hist_range = QComboBox()
        self.hist_range.addItem("range 0.5–99.5 %", "auto")
        self.hist_range.addItem("range 0.9–1.1 (Igor, ratios)", "igor")
        self.hist_range.addItem("range min…max below", "custom")
        r.addWidget(self.hist_range, 1)
        af.addRow(r)
        r = QHBoxLayout()
        self.hist_min = _dspin(-1e9, 1e9, -0.1, 0.01, 4)
        self.hist_max = _dspin(-1e9, 1e9, 0.1, 0.01, 4)
        r.addWidget(QLabel("min"))
        r.addWidget(self.hist_min)
        r.addWidget(QLabel("max"))
        r.addWidget(self.hist_max)
        af.addRow("Custom range:", r)
        r = QHBoxLayout()
        b = QPushButton("2-D FFT + period")
        b.setToolTip("Igor imageFFT (power-of-2 window, Hann); the azimuthal average gives the dominant period")
        b.clicked.connect(self._an_fft)
        r.addWidget(b)
        b = QPushButton("Power spectrum (line)")
        b.setToolTip("Igor powerSpectrum of the line profile: period along the line")
        b.clicked.connect(self._an_power)
        r.addWidget(b)
        af.addRow(r)
        r = QHBoxLayout()
        self.fft_cut = QSpinBox()
        self.fft_cut.setRange(1, 4096)
        self.fft_cut.setValue(40)
        r.addWidget(QLabel("cutoff"))
        r.addWidget(self.fft_cut)
        self.fft_mode = QComboBox()
        self.fft_mode.addItem("along x (Igor)", "x")
        self.fft_mode.addItem("radial", "radial")
        r.addWidget(self.fft_mode)
        b = QPushButton("FFT filter")
        b.setToolTip("Igor FFTFilter: low-pass, gain 1 below the cutoff index then 1/(i+1-cutoff) -> 'fft_filtered'")
        b.clicked.connect(self._an_fft_filter)
        r.addWidget(b)
        af.addRow(r)
        r = QHBoxLayout()
        self.chk_flat_off = QCheckBox("remove offset")
        self.chk_flat_pts = QCheckBox("end points only")
        b = QPushButton("Flatten along line")
        b.setToolTip("Igor Flatten (map -> 'flattened') / FlattenSequence (frames -> whole stack); half-width as above")
        b.clicked.connect(self._an_flatten)
        r.addWidget(b)
        r.addWidget(self.chk_flat_off)
        r.addWidget(self.chk_flat_pts)
        af.addRow(r)
        r = QHBoxLayout()
        self.bg_order = QSpinBox()
        self.bg_order.setRange(0, 6)
        self.bg_order.setValue(2)
        self.bg_cell = QSpinBox()
        self.bg_cell.setRange(1, 200)
        self.bg_cell.setValue(5)
        b = QPushButton("Background (rows/cols)")
        b.setToolTip("Igor backgroundremoval: polynomial of every row and column (map -> 'bg_removed'; frames -> whole stack)")
        b.clicked.connect(self._an_bg_poly)
        r.addWidget(b)
        r.addWidget(QLabel("order"))
        r.addWidget(self.bg_order)
        r.addWidget(QLabel("cell (px)"))
        r.addWidget(self.bg_cell)
        af.addRow(r)
        r = QHBoxLayout()
        b = QPushButton("Background from BG ROI")
        b.setToolTip("Igor bkgRemove: order-3 polynomial surface fitted in the BG ROI (section 7) -> 'bg_removed'")
        b.clicked.connect(self._an_bg_roi)
        r.addWidget(b)
        b = QPushButton("Use shown variable as map data")
        b.setToolTip("ROI statistics and domain maps then use it; the old data are kept as 'data_original'")
        b.clicked.connect(self._an_use_as_data)
        r.addWidget(b)
        af.addRow(r)
        QVBoxLayout(ana).addWidget(ana_body)
        ana_body.setVisible(False)
        ana.toggled.connect(ana_body.setVisible)
        ana.toggled.connect(lambda _on: ana_body.setEnabled(True))
        cl.addWidget(ana)

    def _view_args(self):
        return (self._view_kind(), self._view_var() or "data", self._frame())

    def _plot(self, title: str):
        self.pfig.clear()
        self.plot_title.setText(title)
        self.plot_box.setVisible(True)
        return self.pfig.add_subplot(111)

    def _start_line(self):
        self._start_tool("line", self._on_line, "orange", "Click point A, then point B on the image.")

    def _set_line_coords(self):
        try:
            x1, y1, x2, y2 = _numbers(self.line_coords.text(), 4)
        except ValueError as exc:
            self._err(str(exc))
            return
        self._on_line({"a": (y1, x1), "b": (y2, x2)})

    def _on_line(self, sh: dict):
        self._line = (tuple(sh["a"]), tuple(sh["b"]))
        (y1, x1), (y2, x2) = self._line
        self.line_coords.setText(f"{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}")
        self.line_label.setText(f"Line: A (x {x1:.0f}, y {y1:.0f}) → B (x {x2:.0f}, y {y2:.0f}), "
                                f"{np.hypot(y2 - y1, x2 - x1):.0f} px")
        self._refresh_view()

    def _need_line(self) -> bool:
        if not getattr(self, "_line", None):
            self._err("Draw a line first (Draw line, then click A and B).")
            return False
        return bool(self._require())

    def _an_call(self, fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self._err(str(exc))
            return None

    def _an_profile(self):
        if not self._need_line():
            return None
        kind, var, fr = self._view_args()
        pr = self._an_call(self.service.line_profile, self._name, kind, var, fr, *self._line, self.line_hw.value())
        if pr is None:
            return None
        self._last_profile = pr
        ax = self._plot(f"Profile A→B · {kind}/{var} · half-width {pr['halfwidth']} px")
        peem_plots.line_profile(ax, pr)
        self.pfig.tight_layout()
        self.pcanvas.draw_idle()
        return pr

    def _an_fit(self, edge: bool):
        pr = self._an_profile()
        if pr is None:
            return
        from tensorspec.core.peem import image_analysis as ia
        res = self._an_call(ia.diff_gauss_fit if edge else ia.gauss_fit, pr["distance"], pr["values"])
        if res is None:
            return
        peem_plots.profile_fit(self.pfig.axes[0], pr, res, edge)
        u = pr["unit"]
        txt = (f"{'edge' if edge else 'peak'} at {res['x0']:.3f} {u}, FWHM {res['fwhm']:.3f} ± "
               f"{res['fwhm_err']:.3f} {u}")
        self.plot_title.setText(self.plot_title.text() + " · " + txt)
        self._log(txt)
        self.pcanvas.draw_idle()

    def _an_stack_profile(self):
        if not self._need_line():
            return
        kind = self._view_kind()
        if kind not in self._FRAME_KINDS:
            kind = self.view_combo.currentData() or "raw"  # map shown: use the current frame stack
        st = self._an_call(self.service.stack_line_profiles, self._name, kind, *self._line, self.line_hw.value())
        if st is None:
            return
        ax = self._plot(f"Profile vs frame · {kind} · half-width {self.line_hw.value()} px")
        peem_plots.stack_line_profiles(ax, st)
        self.pfig.tight_layout()
        self.pcanvas.draw_idle()

    def _an_histogram(self):
        if not self._require():
            return
        kind, var, fr = self._view_args()
        mode = self.hist_range.currentData()
        rng = None if mode == "auto" else (0.9, 1.1) if mode == "igor" else (self.hist_min.value(), self.hist_max.value())
        h = self._an_call(self.service.histogram, self._name, kind, var, fr, self._roi, rng)
        if h is None:
            return
        ax = self._plot(f"Histogram · {kind}/{var} · {'map ROI' if self._roi else 'whole image'} · n {h['n']} · "
                        f"mean {h['mean']:.4g} · std {h['std']:.4g}" + (f" · {h['outside_range']} outside" if h["outside_range"] else ""))
        peem_plots.histogram(ax, h)
        self.pfig.tight_layout()
        self.pcanvas.draw_idle()

    def _an_fft(self):
        if not self._require():
            return
        kind, var, fr = self._view_args()
        f = self._an_call(self.service.image_fft, self._name, kind, var, fr)
        if f is None:
            return
        rad = f["radial"]
        per = f"{rad['period_um']:.3f} µm" if rad["period_um"] else f"{rad['period_px']:.1f} px"
        self._plot(f"2-D FFT · {kind}/{var} · dominant period {per}")
        peem_plots.image_fft(self.pfig, f)
        self.pfig.tight_layout()
        self.pcanvas.draw_idle()
        self._log(f"FFT dominant period {per} (f = {rad['peak_f']:.4f} 1/px)")

    def _an_power(self):
        pr = self._an_profile()
        if pr is None:
            return
        from tensorspec.core.peem import image_analysis as ia
        ps = ia.power_spectrum(pr["values"])
        upp = self.service.xmcd_meta(self._name).get("um_per_px")
        per = f"{ps['period_px'] * upp:.3f} µm" if upp else f"{ps['period_px']:.1f} px"
        ax = self._plot(f"Power spectrum of the line · peak period {per}")
        peem_plots.power_spectrum(ax, ps)
        self.pfig.tight_layout()
        self.pcanvas.draw_idle()

    def _an_after_op(self, res):
        if res is None:
            return
        self._log(json.dumps(res, default=lambda o: round(float(o), 6) if isinstance(o, (np.floating, float)) else str(o)))
        if "variable" in res:
            self._refresh_views(select=res["kind"], var=res["variable"])
        else:  # stack clean-up edit
            self._frames_dirty = True
            self.edits_label.setText(f"{res['summary']} · {res['frames_used']}/{res['frames_total']} frames used"
                                     + (f" · {res['note']}" if res.get("note") else ""))
            self._refresh_views()

    def _an_flatten(self):
        if not self._need_line():
            return
        kind, var, fr = self._view_args()
        self._an_after_op(self._an_call(self.service.flatten, self._name, kind, var, fr, *self._line,
                                        max(1, self.line_hw.value()), self.chk_flat_off.isChecked(),
                                        self.chk_flat_pts.isChecked()))

    def _an_bg_poly(self):
        if self._require():
            kind, var, fr = self._view_args()
            self._an_after_op(self._an_call(self.service.background_poly, self._name, kind, var, fr,
                                            self.bg_order.value(), self.bg_cell.value()))

    def _an_bg_roi(self):
        if not self._require():
            return
        if self._bg_roi is None:
            self._err("Draw a BG ROI first (section 7, Draw BG ROI).")
            return
        kind, var, fr = self._view_args()
        self._an_after_op(self._an_call(self.service.background_roi, self._name, kind, var, fr, self._bg_roi, 3))

    def _an_fft_filter(self):
        if self._require():
            kind, var, fr = self._view_args()
            self._an_after_op(self._an_call(self.service.fft_filter, self._name, kind, var, fr,
                                            self.fft_cut.value(), self.fft_mode.currentData()))

    def _an_use_as_data(self):
        if self._require():
            kind, var, _ = self._view_args()
            if kind in ("raw", "cleaned", "processed"):
                self._err("Show a map (pcd, pld, ...) and pick the variable first.")
                return
            res = self._an_call(self.service.use_as_map_data, self._name, kind, var)
            if res:
                self._log(json.dumps(res, default=str))
                self._refresh_views(select=kind)
