"""Tab 1 · Map Explorer: FWHM(2D) / G height / 2D height map, pixel grid, optical overlay,
orientation and auto-alignment, raw spectrum and fit of a clicked pixel. The figures are drawn by
plotting/backends/raman_views/map_explorer.py.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Tab 1 - Map Explorer)."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

from tensorspec.plotting.backends.raman_matplotlib_engine import draw_placeholder
from tensorspec.plotting.backends.raman_views import map_explorer as views
from tensorspec.plotting.viewers.raman_canvas import Canvas

if TYPE_CHECKING:
    from tensorspec.gui.components.raman_panel import RamanPanel

# ---------------------------------------------------------------------------
# Tab 1 - Map Explorer
# ---------------------------------------------------------------------------

class MapExplorerTab(QtWidgets.QWidget):

    def __init__(self, main: "RamanPanel"):
        super().__init__()
        self.main = main
        self.sel: Optional[Tuple[int, int]] = None
        self._spec: Optional[dict] = None     # last plotted spectrum, cached
        self._grid_artists: List = []
        self._build()

    # -- UI -----------------------------------------------------------------
    def _build(self):
        root = QtWidgets.QVBoxLayout(self)

        bar = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load Raman map…")
        self.btn_load.clicked.connect(self.main.load_raman)
        self.btn_optical = QtWidgets.QPushButton("Load optical image…")
        self.btn_optical.clicked.connect(self.main.load_optical)
        self.chk_grid = QtWidgets.QCheckBox("Grid overlay")
        self.chk_grid.toggled.connect(self.refresh_map)
        self.chk_optical = QtWidgets.QCheckBox("Overlay optical")
        self.chk_optical.toggled.connect(self.refresh_map)
        self.chk_area = QtWidgets.QCheckBox("Select area")
        self.chk_area.setToolTip("Drag a rectangle on the map: the spectra of the pixels inside are "
                                 "summed (2 · Map ▸ Area spectrum). Drag inside it to move it, drag "
                                 "its corners or edges to resize it.")
        self.chk_area.toggled.connect(self.refresh_map)
        self._area_selector = None
        self.sld_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_alpha.setRange(0, 100)
        self.sld_alpha.setValue(50)
        self.sld_alpha.setFixedWidth(120)
        self.sld_alpha.valueChanged.connect(self._alpha_changed)
        self.cmb_cmap = QtWidgets.QComboBox()
        self.cmb_cmap.addItems(["viridis", "plasma", "inferno", "magma",
                                "cividis", "jet", "turbo", "coolwarm"])
        self.cmb_cmap.currentTextChanged.connect(self._cmap_changed)

        for w in (self.btn_load, self.btn_optical):
            bar.addWidget(w)
        bar.addSpacing(12)
        bar.addWidget(self.chk_grid)
        bar.addWidget(self.chk_optical)
        bar.addWidget(self.chk_area)
        bar.addWidget(QtWidgets.QLabel("Opacity"))
        bar.addWidget(self.sld_alpha)
        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel("Map"))
        self.cmb_quantity = QtWidgets.QComboBox()
        self.cmb_quantity.addItems(["Total intensity", "Band integral", "Peak map", "FWHM(2D)",
                                    "G height", "2D height"])
        self.cmb_quantity.setToolTip("Total intensity / Band integral: raw counts integrated over "
                                     "the whole spectrum / a window, no fitting, no background "
                                     "removed.\nFWHM(2D): graphene 2D-band Voigt fit, after "
                                     "'Fit 2D band on all pixels' (3 · Analysis ▸ Classification)."
                                     "\nG / 2D height: raw graphene peak height above the "
                                     "flanking baseline, no fitting.")
        self.cmb_quantity.currentIndexChanged.connect(self._quantity_changed)
        bar.addWidget(self.cmb_quantity)
        # window of the band integral: a band of the 1 · Spectra fit model, D, Si, or custom
        self.cmb_band = QtWidgets.QComboBox()
        self.cmb_band.setToolTip("Window to integrate: the bands of 1 · Spectra ▸ Peak fitting, "
                                 "windows of an applied preset, or Custom")
        self.cmb_band.activated.connect(self._band_picked)
        self.spn_w0 = QtWidgets.QDoubleSpinBox()
        self.spn_w1 = QtWidgets.QDoubleSpinBox()
        for sp, v in ((self.spn_w0, 2560.0), (self.spn_w1, 2810.0)):
            sp.setRange(0, 10000)
            sp.setDecimals(0)
            sp.setSingleStep(10)
            sp.setValue(v)
            sp.setFixedWidth(72)
            sp.setSuffix(" cm⁻¹")
            sp.editingFinished.connect(self._window_edited)
        self._band_row = (self.cmb_band, self.spn_w0, self.spn_w1)
        for w in self._band_row:
            bar.addWidget(w)
            w.setVisible(False)
        # peak map: band, peak (count or peak k, low to high wavenumber), fitted quantity
        self.cmb_pband = QtWidgets.QComboBox()
        self.cmb_peak = QtWidgets.QComboBox()
        self.cmb_pq = QtWidgets.QComboBox()
        for label, key in (("position", "pos"), ("FWHM", "fwhm"), ("height", "height"),
                           ("area", "area")):
            self.cmb_pq.addItem(label, key)
        self.cmb_peak.setToolTip("Peaks are numbered in each pixel from low to high wavenumber")
        self.cmb_pband.currentIndexChanged.connect(self._fill_peaks)
        for w in (self.cmb_peak, self.cmb_pq):
            w.currentIndexChanged.connect(self.refresh_map)
        self._peak_row = (self.cmb_pband, self.cmb_peak, self.cmb_pq)
        for w in self._peak_row:
            bar.addWidget(w)
            w.setVisible(False)
        self.btn_fit_pixels = QtWidgets.QPushButton("Fit every pixel")
        self.btn_fit_pixels.setToolTip("Fit every pixel's spectrum with the model of 1 · Spectra ▸ "
                                       "Peak fitting and map each peak (Map ▸ Peak map). Runs in "
                                       "the background; nothing is fitted until pressed.")
        self.btn_fit_pixels.clicked.connect(self.main.start_peak_maps)
        bar.addWidget(self.btn_fit_pixels)
        bar.addWidget(QtWidgets.QLabel("Colour map"))
        bar.addWidget(self.cmb_cmap)
        bar.addStretch(1)
        root.addLayout(bar)

        # ---- map orientation relative to the camera image ----
        obar = QtWidgets.QHBoxLayout()
        obar.addWidget(QtWidgets.QLabel("Map orientation:"))
        self.chk_swap = QtWidgets.QCheckBox("Swap X/Y")
        self.chk_fx = QtWidgets.QCheckBox("Flip X")
        self.chk_fy = QtWidgets.QCheckBox("Flip Y")
        for w in (self.chk_swap, self.chk_fx, self.chk_fy):
            w.toggled.connect(self._transform_edited)
            obar.addWidget(w)
        obar.addSpacing(8)
        obar.addWidget(QtWidgets.QLabel("offset X"))
        self.spn_dx = QtWidgets.QDoubleSpinBox()
        obar.addWidget(self.spn_dx)
        obar.addWidget(QtWidgets.QLabel("Y"))
        self.spn_dy = QtWidgets.QDoubleSpinBox()
        obar.addWidget(self.spn_dy)
        for sp in (self.spn_dx, self.spn_dy):
            sp.setRange(-200.0, 200.0)
            sp.setDecimals(2)
            sp.setSingleStep(0.5)
            sp.setSuffix(" µm")
            sp.setFixedWidth(92)
            sp.valueChanged.connect(self._transform_edited)
        obar.addSpacing(14)
        self.chk_zoom = QtWidgets.QCheckBox("Zoom to map")
        self.chk_zoom.setChecked(bool(self.main.cfg.get("zoom_to_map", True)))
        self.chk_zoom.setToolTip("Frame the maps on the Raman grid. Untick to see the "
                                 "whole camera frame around it.")
        self.chk_zoom.toggled.connect(self._zoom_toggled)
        obar.addWidget(self.chk_zoom)
        obar.addStretch(1)
        root.addLayout(obar)
        obar = QtWidgets.QHBoxLayout()          # second line: alignment controls
        obar.addSpacing(18)
        obar.addWidget(QtWidgets.QLabel("Flake appears"))
        self.cmb_contrast = QtWidgets.QComboBox()
        self.cmb_contrast.addItems(["auto", "brighter", "darker"])
        self.cmb_contrast.setToolTip(
            "How the flake looks against the substrate in the camera image. "
            "'auto' works either way but can confuse the flake with its "
            "180°-rotated look-alike; telling it removes that.")
        obar.addWidget(self.cmb_contrast)
        self.btn_align = QtWidgets.QPushButton("Auto-align to optical")
        self.btn_align.setToolTip(
            "Try every swap / flip combination, each with a search over "
            "offsets, and keep the one where the graphene band signal best "
            "matches the flake in the camera image.")
        self.btn_align.clicked.connect(self.main.run_auto_align)
        obar.addWidget(self.btn_align)
        self.chk_allorient = QtWidgets.QCheckBox("Search all orientations")
        self.chk_allorient.setToolTip(
            "Off (recommended): keep the orientation, search only the offset.\n"
            "On: try all eight swap/flip combinations — slower, and easily fooled "
            "by uneven illumination in the camera image.")
        obar.addWidget(self.chk_allorient)
        self.chk_ydown = QtWidgets.QCheckBox("Y down (as LabSpec)")
        self.chk_ydown.setChecked(bool(self.main.cfg.get("y_down", True)))
        self.chk_ydown.setToolTip("Display only: draw +Y pointing down, as LabSpec does, "
                                  "so maps look the same as on the instrument")
        self.chk_ydown.toggled.connect(self._ydown_toggled)
        obar.addWidget(self.chk_ydown)
        self.btn_hdr = QtWidgets.QPushButton("Instrument default")
        self.btn_hdr.setToolTip("X = −col0, Y = col1 with no offset — this "
                                "LabRAM's convention, verified against LabSpec")
        self.btn_hdr.clicked.connect(lambda: self.main.set_transform(None))
        obar.addWidget(self.btn_hdr)
        obar.addStretch(1)
        root.addLayout(obar)
        self.lbl_transform = QtWidgets.QLabel("")
        self.lbl_transform.setStyleSheet("color:#555; font-size:11px;")
        root.addWidget(self.lbl_transform)
        self._syncing = False

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.map_canvas = Canvas(figsize=(6, 5))
        self.map_canvas.canvas.mpl_connect("button_press_event", self._on_click)
        self.map_canvas.setMinimumHeight(240)

        spec_box = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(spec_box)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(2)

        sbar = QtWidgets.QHBoxLayout()
        sbar.setContentsMargins(4, 0, 4, 0)
        sbar.addWidget(QtWidgets.QLabel("View"))
        self.cmb_view = QtWidgets.QComboBox()
        self.cmb_view.addItems(["2D band", "Full spectrum", "Custom"])
        self.cmb_view.currentIndexChanged.connect(self._view_changed)
        sbar.addWidget(self.cmb_view)
        sbar.addSpacing(8)
        sbar.addWidget(QtWidgets.QLabel("from"))
        self.spn_x0 = QtWidgets.QDoubleSpinBox()
        self.spn_x1 = QtWidgets.QDoubleSpinBox()
        for sp, val in ((self.spn_x0, 2580.0), (self.spn_x1, 2820.0)):
            sp.setRange(0, 10000)
            sp.setDecimals(0)
            sp.setSingleStep(10)
            sp.setValue(val)
            sp.setFixedWidth(80)
            sp.valueChanged.connect(self._range_edited)
            sbar.addWidget(sp if sp is self.spn_x0 else QtWidgets.QLabel("to"))
            if sp is self.spn_x1:
                sbar.addWidget(sp)
        sbar.addWidget(QtWidgets.QLabel("cm⁻¹"))
        sbar.addSpacing(10)
        self.chk_yfit = QtWidgets.QCheckBox("Fit Y to view")
        self.chk_yfit.setChecked(True)
        self.chk_yfit.setToolTip(
            "Rescale the intensity axis to whatever is inside the wavenumber "
            "range, instead of to the whole spectrum. A weak 2D band sitting "
            "on a tall background is invisible without this.")
        self.chk_yfit.toggled.connect(self._apply_view)
        sbar.addWidget(self.chk_yfit)
        self.btn_reset_view = QtWidgets.QPushButton("Reset view")
        self.btn_reset_view.clicked.connect(self._reset_view)
        sbar.addWidget(self.btn_reset_view)
        self.btn_to_spec = QtWidgets.QPushButton("Add to Spectra tab")
        self.btn_to_spec.setToolTip("Add the selected pixel's spectrum to the "
                                    "comparison in 2 · Map ▸ Compare cells")
        self.btn_to_spec.clicked.connect(self._send_to_spectra)
        sbar.addWidget(self.btn_to_spec)
        sbar.addStretch(1)
        sv.addLayout(sbar)

        self.spec_canvas = Canvas(figsize=(6, 3.2))
        self.spec_canvas.setMinimumHeight(150)
        sv.addWidget(self.spec_canvas, 1)

        split.addWidget(self.map_canvas)
        split.addWidget(spec_box)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([640, 280])
        root.addWidget(split, 1)

        self.info = QtWidgets.QLabel("Load a Raman map file to begin.")
        self.info.setStyleSheet("color:#555;")
        root.addWidget(self.info)

        self._show_placeholder()

    # -- reactions ----------------------------------------------------------
    def _alpha_changed(self, v):
        self.main.cfg["optical_alpha"] = v / 100.0
        self.refresh_map()
        self.main.tab_class.refresh_map()

    def _quantity_changed(self):
        integral = self.cmb_quantity.currentText() == "Band integral"
        for w in self._band_row:
            w.setVisible(integral)
        if integral:
            self._fill_bands()
        for w in self._peak_row:
            w.setVisible(self.cmb_quantity.currentText() == "Peak map")
        self.refresh_map()

    # -- peak maps ---------------------------------------------------------
    def peak_maps_ready(self):
        """Offer the bands of the latest pixel fits (or none) in the Peak map pickers."""
        grids = self.main.service.peak_grids
        bands = [k[:-2] for k in grids if k.endswith("_n")]
        keep = self.cmb_pband.currentText()
        self.cmb_pband.blockSignals(True)
        self.cmb_pband.clear()
        self.cmb_pband.addItems(bands)
        if keep in bands:
            self.cmb_pband.setCurrentText(keep)
        self.cmb_pband.blockSignals(False)
        self._fill_peaks()
        if bands and self.cmb_quantity.currentText() != "Peak map":
            self.cmb_quantity.setCurrentText("Peak map")

    def _fill_peaks(self):
        band = self.cmb_pband.currentText()
        grids = self.main.service.peak_grids
        n = 0
        while "%s_p%d_pos" % (band, n + 1) in grids:
            n += 1
        keep = self.cmb_peak.currentText()
        self.cmb_peak.blockSignals(True)
        self.cmb_peak.clear()
        self.cmb_peak.addItem("peak count", "n")
        for k in range(1, n + 1):
            self.cmb_peak.addItem("peak %d" % k, k)
        i = self.cmb_peak.findText(keep)
        self.cmb_peak.setCurrentIndex(i if i >= 0 else min(1, self.cmb_peak.count() - 1))
        self.cmb_peak.blockSignals(False)
        self.refresh_map()

    def peak_key(self):
        """(grid key, colour-bar label, title) of the peak map chosen, or None."""
        band, peak = self.cmb_pband.currentText(), self.cmb_peak.currentData()
        if not band or peak is None:
            return None
        if peak == "n":
            return "%s_n" % band, "peaks (0 = no fit)", "%s peak count (fit)" % band
        q = self.cmb_pq.currentData()
        unit = {"pos": "cm$^{-1}$", "fwhm": "cm$^{-1}$", "height": "counts",
                "area": "counts·cm$^{-1}$"}[q]
        return ("%s_p%d_%s" % (band, peak, q), "%s  [%s]" % (self.cmb_pq.currentText(), unit),
                "%s peak %d %s (fit)" % (band, peak, self.cmb_pq.currentText()))

    def _window_choices(self):
        """[(name, lo, hi)]: the step-1 fit bands, then any windows of the applied preset."""
        svc = self.main.service
        out = [(name, lo, hi) for name, lo, hi, *_ in svc.fit_model.bands]
        names = {n for n, _, _ in out}
        out += [(n, lo, hi) for n, (lo, hi) in svc.preset_windows.items() if n not in names]
        return out

    def _fill_bands(self):
        keep = self.cmb_band.currentText()
        self.cmb_band.blockSignals(True)
        self.cmb_band.clear()
        for name, lo, hi in self._window_choices():
            self.cmb_band.addItem("%s  %g–%g" % (name, lo, hi), (name, lo, hi))
        self.cmb_band.addItem("Custom", None)
        i = self.cmb_band.findText(keep)
        self.cmb_band.setCurrentIndex(i if i >= 0 else 0)
        self.cmb_band.blockSignals(False)
        self._band_picked()

    def _band_picked(self, *_):
        choice = self.cmb_band.currentData()
        if choice is not None:
            for sp, v in ((self.spn_w0, choice[1]), (self.spn_w1, choice[2])):
                sp.blockSignals(True)
                sp.setValue(v)
                sp.blockSignals(False)
            self.refresh_map()

    def _window_edited(self):
        """Typing a window makes it Custom unless it matches the chosen band."""
        choice = self.cmb_band.currentData()
        if choice is not None and (self.spn_w0.value(), self.spn_w1.value()) != (choice[1], choice[2]):
            self.cmb_band.setCurrentIndex(self.cmb_band.count() - 1)
        self.refresh_map()

    def window(self):
        """(name, lo, hi) of the band integral currently chosen."""
        choice = self.cmb_band.currentData()
        lo, hi = sorted((self.spn_w0.value(), self.spn_w1.value()))
        return (choice[0] if choice is not None else "window"), lo, hi

    def _cmap_changed(self, name):
        self.main.cfg["cmap"] = name
        self.refresh_map()

    def _zoom_toggled(self, on):
        self.main.cfg["zoom_to_map"] = bool(on)
        self.refresh_map()
        self.main.tab_class.refresh_map()

    def _ydown_toggled(self, on):
        self.main.cfg["y_down"] = bool(on)
        self.refresh_map()
        self.main.tab_class.refresh_map()
        self.main.tab_spec.refresh_mini()

    def _send_to_spectra(self):
        mf = self.main.mapfile
        if mf is None or self.sel is None:
            self.main.status("Click a pixel on the map first.")
            return
        i = int(mf.index_map[self.sel])
        if i >= 0:
            self.main.tab_spec.add_spectrum(i)
            self.main.status("Added %s to the Spectra tab (%d there now)."
                             % (self.main.tab_spec._name(i).split("  ")[0],
                                len(self.main.tab_spec.items)))

    def _view_changed(self):
        mode = self.cmb_view.currentText()
        custom = mode == "Custom"
        self.spn_x0.setEnabled(custom)
        self.spn_x1.setEnabled(custom)
        if not custom:
            lo, hi = self._view_range()
            for sp, v in ((self.spn_x0, lo), (self.spn_x1, hi)):
                sp.blockSignals(True)
                sp.setValue(v)
                sp.blockSignals(False)
        self._apply_view()

    def _range_edited(self):
        """Editing the bounds by hand implies a custom view."""
        if self.cmb_view.currentText() != "Custom":
            self.cmb_view.blockSignals(True)
            self.cmb_view.setCurrentText("Custom")
            self.cmb_view.blockSignals(False)
            self.spn_x0.setEnabled(True)
            self.spn_x1.setEnabled(True)
        self._apply_view()

    def _reset_view(self):
        self.cmb_view.setCurrentText("2D band")
        self.chk_yfit.setChecked(True)
        self._view_changed()

    def _view_range(self) -> Tuple[float, float]:
        cfg = self.main.cfg
        mode = self.cmb_view.currentText()
        if mode == "Full spectrum":
            if self._spec is not None:
                wn = self._spec["wn"]
                return float(wn[0]), float(wn[-1])
            return 1000.0, 3200.0
        if mode == "Custom":
            lo, hi = self.spn_x0.value(), self.spn_x1.value()
            return (lo, hi) if hi > lo else (lo, lo + 10.0)
        pad = 0.2 * (cfg["band_max"] - cfg["band_min"])
        return cfg["band_min"] - pad, cfg["band_max"] + pad

    def _apply_view(self):
        """Re-limit the axes without refitting anything."""
        if self._spec is None:
            return
        lo, hi = self._view_range()
        views.set_spectrum_view(self.spec_canvas.ax, self._spec["wn"], self._spec["inten"],
                                lo, hi, self.chk_yfit.isChecked())
        self.spec_canvas.draw()

    def sync_transform_ui(self):
        """Mirror the active transform into the controls without re-firing."""
        t = self.main.current_transform()
        self._syncing = True
        try:
            if t is not None:
                self.chk_swap.setChecked(bool(t["swap"]))
                self.chk_fx.setChecked(bool(t["flip_x"]))
                self.chk_fy.setChecked(bool(t["flip_y"]))
                self.spn_dx.setValue(float(t["dx"]))
                self.spn_dy.setValue(float(t["dy"]))
                src = (("instrument default, offset not aligned yet"
                        if self.main.cfg.get("orientation_default", "instrument") == "instrument"
                        else "from the #AxisType headers — unverified")
                       if self.main.cfg.get("map_transform") is None else "set")
                self.lbl_transform.setText("Placement (%s):  %s"
                                           % (src, self.main.service.describe_transform(t)))
        finally:
            self._syncing = False

    def _transform_edited(self, *_):
        if self._syncing or self.main.mapfile_raw is None:
            return
        self.main.set_transform({
            "swap": self.chk_swap.isChecked(), "flip_x": self.chk_fx.isChecked(),
            "flip_y": self.chk_fy.isChecked(),
            "dx": self.spn_dx.value(), "dy": self.spn_dy.value()})

    def _show_placeholder(self, text="No map loaded", spectrum_too=True):
        self.map_canvas.clear()
        draw_placeholder(self.map_canvas.ax, text)
        self.map_canvas.draw()
        if not spectrum_too:
            return
        self.spec_canvas.clear()
        draw_placeholder(self.spec_canvas.ax)
        self.spec_canvas.draw()

    # -- drawing ------------------------------------------------------------
    def height_grid(self, which: str) -> np.ndarray:
        """Raw G or 2D peak height on the current grid (no fitting)."""
        return self.main.service.height_map(which)

    def refresh_map(self):
        mf = self.main.mapfile
        quantity = self.cmb_quantity.currentText()
        fwhm = quantity == "FWHM(2D)"
        if mf is None:
            self._show_placeholder()
            return
        if fwhm and self.main.fwhm_map is None:
            self._show_placeholder("FWHM(2D) needs the 2D-band fit:\n3 · Analysis > Classification"
                                   " > Fit 2D band on all pixels", spectrum_too=False)
            return

        cv = self.map_canvas
        cv.clear()
        svc = self.main.service
        cbar_label = title = window = peak_key = None
        if quantity == "Peak map":
            choice = self.peak_key() if svc.peak_grids else None
            if choice is None:
                self._show_placeholder("No pixel fits yet: define bands in 1 · Spectra > Peak fitting,"
                                       "\nthen press 'Fit every pixel'", spectrum_too=False)
                return
            peak_key, cbar_label, title = choice
        elif quantity == "Total intensity":
            cbar_label = "∫ I dω  [counts·cm$^{-1}$]"
            title = "Total intensity, %g–%g cm$^{-1}$ (raw counts)" % (mf.axis.min(), mf.axis.max())
        elif quantity == "Band integral":
            window = self.window()
            cbar_label = "∫ I dω  [counts·cm$^{-1}$]"
            title = "%s integral, %g–%g cm$^{-1}$ (raw counts)" % window
        data = svc.quantity_map(quantity, window, peak_key)
        area_mode = self.chk_area.isChecked()
        area = self.main.service.area_rect
        views.draw_quantity_map(cv.figure, cv.ax, mf, data, quantity, self.main.fit_tally,
                                self.main.cfg, self.main.optical, self.main.optical_image,
                                self.chk_optical.isChecked(), self.chk_grid.isChecked(),
                                self.sel, None if area_mode else area, cbar_label, title)
        # the selector belongs to the axes just drawn; a new one after every redraw. In area
        # mode it shows the current area itself (with handles to move / resize it).
        views.remove_area_selector(self._area_selector)
        self._area_selector = None
        if area_mode:
            ny, nx = mf.shape
            l, r, b, t = mf.extent
            step = min((r - l) / nx, (t - b) / ny)
            self._area_selector = views.make_area_selector(cv.ax, self._area_selected, area,
                                                           min_span=0.3 * step)
        cv.draw()

    def _area_selected(self, x0, x1, y0, y1):
        self.main.set_area((x0, x1, y0, y1))

    def show_area(self):
        """The summed spectrum of the selected area in the spectrum panel."""
        svc = self.main.service
        if svc.area_spectrum is None:
            return
        cv = self.spec_canvas
        cv.clear()
        wn, spec = svc.area_axis, svc.area_spectrum
        self._spec = dict(wn=wn, inten=spec, row=None, col=None)
        views.draw_area_spectrum(cv.ax, wn, spec, len(svc.area_indices),
                                 (self.main.cfg["band_min"], self.main.cfg["band_max"]),
                                 svc.AREA_MODE)
        self._apply_view()
        cv.draw()

    # -- interaction --------------------------------------------------------
    def _on_click(self, event):
        mf = self.main.mapfile
        if mf is None or event.inaxes is None or event.xdata is None or self.chk_area.isChecked():
            return
        left, right, bottom, top = mf.extent
        ny, nx = mf.shape
        col = int(np.floor((event.xdata - left) / (right - left) * nx))
        row = int(np.floor((event.ydata - bottom) / (top - bottom) * ny))
        if not (0 <= col < nx and 0 <= row < ny):
            return
        self.sel = (row, col)
        self.refresh_map()
        self.show_spectrum(row, col)

    def show_spectrum(self, row: int, col: int):
        px = self.main.service.pixel(row, col)
        cv = self.spec_canvas
        cv.clear()
        ax = cv.ax
        if px is None:
            views.draw_no_spectrum(ax)
            cv.draw()
            return
        self._spec = dict(wn=px["wn"], inten=px["inten"], row=row, col=col)
        res, txt = px["fit"], ""
        if px["center"] is not None:
            asym, err = px["asym"], res.get("fwhm_err", np.nan)
            txt = ("x = %.2f µm, y = %.2f µm\nFWHM = %.1f %s cm⁻¹   centre = %.1f cm⁻¹\n"
                   "asymmetry (%s) = %s   SNR = %.1f%s"
                   % (px["x"], px["y"], res["fwhm"],
                      ("± %.1f" % err) if np.isfinite(err) else "",
                      res["center"],
                      self.main.cfg["asym_metric"],
                      "%.3f" % asym if np.isfinite(asym) else "n/a",
                      res["snr"],
                      "" if res["ok"] else "   [rejected: %s]" % res["note"]))
        views.draw_pixel_spectrum(ax, px["wn"], px["inten"], px["band"], px["fit_curve"],
                                  px["center"], row, col)
        self._apply_view()
        gf = px["g_fit"]
        if gf is not None and txt:
            txt += ("\nI(2D)/I(G) = %.2f  →  %s" % (px["ratio"], px["layer"])
                    if px["layer"] is not None else
                    "\nI(2D)/I(G) unavailable (%s)" % (gf.get("why") or "G fit rejected"))
            if px["g_curve"] is not None:
                views.draw_g_fit(ax, *px["g_curve"])
        cv.draw()
        self.info.setText(txt or "No fit available for this pixel.")
        rejected = bool(res) and not res.get("ok", False)
        self.info.setStyleSheet("color:#b71c1c;" if rejected else "color:#555;")
