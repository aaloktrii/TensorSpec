"""Tab 1 · Map Explorer: FWHM(2D) / G height / 2D height map, pixel grid, optical overlay,
orientation and auto-alignment, raw spectrum and fit of a clicked pixel. The figures are drawn by
plotting/backends/raman_views/map_explorer.py.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Tab 1 - Map Explorer)."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

from tensorspec.core.raman.classification import g_band_cfg, label_of, layer_from_ratio, pixel_ratio
from tensorspec.core.raman.band_maps import EXTRA_WINDOWS
from tensorspec.core.raman.map_geometry import band_height, describe_transform
from tensorspec.core.raman.voigt_2d import asym_of
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
        self.chk_area.setToolTip("Drag a rectangle on the map: the spectra inside are summed and "
                                 "fitted with the 1 · Spectra model (2 · Map ▸ Area spectrum).")
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
        self.cmb_quantity.addItems(["FWHM(2D)", "G height", "2D height", "Band integral"])
        self.cmb_quantity.setToolTip("G / 2D height: raw peak height above the "
                                     "flanking baseline, no fitting.\nBand integral: raw counts "
                                     "integrated over a wavenumber window, no background removed")
        self.cmb_quantity.currentIndexChanged.connect(self._quantity_changed)
        bar.addWidget(self.cmb_quantity)
        # window of the band integral: a band of the 1 · Spectra fit model, D, Si, or custom
        self.cmb_band = QtWidgets.QComboBox()
        self.cmb_band.setToolTip("Window to integrate: the bands of 1 · Spectra ▸ Peak fitting, "
                                 "D, Si, or Custom")
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
        self.refresh_map()

    def _window_choices(self):
        """[(name, lo, hi)]: the step-1 fit bands, then D and Si."""
        out = [(name, lo, hi) for name, lo, hi, *_ in self.main.tab_fit.bands()]
        names = {n for n, _, _ in out}
        out += [(n, lo, hi) for n, (lo, hi) in EXTRA_WINDOWS.items() if n not in names]
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
                                           % (src, describe_transform(t)))
        finally:
            self._syncing = False

    def _transform_edited(self, *_):
        if self._syncing or self.main.mapfile_raw is None:
            return
        self.main.set_transform({
            "swap": self.chk_swap.isChecked(), "flip_x": self.chk_fx.isChecked(),
            "flip_y": self.chk_fy.isChecked(),
            "dx": self.spn_dx.value(), "dy": self.spn_dy.value()})

    def _show_placeholder(self):
        self.map_canvas.clear()
        draw_placeholder(self.map_canvas.ax, "No map loaded")
        self.map_canvas.draw()
        self.spec_canvas.clear()
        draw_placeholder(self.spec_canvas.ax)
        self.spec_canvas.draw()

    # -- drawing ------------------------------------------------------------
    def height_grid(self, which: str) -> np.ndarray:
        """Raw G or 2D peak height on the current grid (no fitting)."""
        mf = self.main.mapfile
        if which == "G height":
            v = band_height(mf.axis, mf.data, 1582.0, 22.0)
        else:
            c = 0.5 * (self.main.cfg["band_min"] + self.main.cfg["band_max"])
            v = band_height(mf.axis, mf.data, c, 45.0)
        g = np.full(mf.shape, np.nan)
        ok = mf.index_map >= 0
        g[ok] = v[mf.index_map[ok]]
        return g

    def refresh_map(self):
        mf = self.main.mapfile
        quantity = self.cmb_quantity.currentText()
        fwhm = quantity == "FWHM(2D)"
        if mf is None or (fwhm and self.main.fwhm_map is None):
            self._show_placeholder()
            return

        cv = self.map_canvas
        cv.clear()
        cbar_label = title = None
        if quantity == "Band integral":
            name, lo, hi = self.window()
            data = self.main.service.band_integral_map(name, lo, hi)
            cbar_label = "∫ I dω  [counts·cm$^{-1}$]"
            title = "%s integral, %g–%g cm$^{-1}$ (raw counts)" % (name, lo, hi)
        else:
            data = self.main.fwhm_map if fwhm else self.height_grid(quantity)
        views.draw_quantity_map(cv.figure, cv.ax, mf, data, quantity, self.main.fit_tally,
                                self.main.cfg, self.main.optical, self.main.optical_image,
                                self.chk_optical.isChecked(), self.chk_grid.isChecked(),
                                self.sel, self.main.service.area_rect, cbar_label, title)
        # the selector belongs to the axes just drawn; a new one after every redraw
        self._area_selector = (views.make_area_selector(cv.ax, self._area_selected)
                               if self.chk_area.isChecked() else None)
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
        mf = self.main.mapfile
        idx = mf.index_map[row, col]
        cv = self.spec_canvas
        cv.clear()
        ax = cv.ax
        if idx < 0:
            views.draw_no_spectrum(ax)
            cv.draw()
            return

        wn, inten = mf.axis, mf.data[idx]
        self._spec = dict(wn=wn, inten=inten, row=row, col=col)
        lo, hi = self.main.cfg["band_min"], self.main.cfg["band_max"]

        res = self.main.fits[idx] if self.main.fits else None
        txt = ""
        fit_curve, center = None, None
        if res and np.isfinite(res.get("center", np.nan)):
            m = (wn >= lo) & (wn <= hi)
            if m.sum() > 3:
                fit_curve = (wn[m], self.main.engine.rebuild_fit(wn[m], res))
            center = res["center"]
            asym = asym_of(res, self.main.cfg["asym_metric"])
            err = res.get("fwhm_err", np.nan)
            txt = ("x = %.2f µm, y = %.2f µm\nFWHM = %.1f %s cm⁻¹   centre = %.1f cm⁻¹\n"
                   "asymmetry (%s) = %s   SNR = %.1f%s"
                   % (mf.x[idx], mf.y[idx], res["fwhm"],
                      ("± %.1f" % err) if np.isfinite(err) else "",
                      res["center"],
                      self.main.cfg["asym_metric"],
                      "%.3f" % asym if np.isfinite(asym) else "n/a",
                      res["snr"],
                      "" if res["ok"] else "   [rejected: %s]" % res["note"]))

        views.draw_pixel_spectrum(ax, wn, inten, (lo, hi), fit_curve, center, row, col)
        self._apply_view()
        gf = self.main.g_fits[idx] if self.main.g_fits else None
        if gf is not None and txt:
            r = pixel_ratio(res, gf, self.main.cfg.get("ratio_metric", "height"))
            n = layer_from_ratio(r, self.main.cfg)
            txt += ("\nI(2D)/I(G) = %.2f  →  %s" % (r, label_of(n, "N/A"))
                    if n is not None else
                    "\nI(2D)/I(G) unavailable (%s)" % (gf.get("why") or "G fit rejected"))
            if gf.get("ok") and np.isfinite(gf.get("center", np.nan)):
                gcfg = g_band_cfg(self.main.cfg)
                gm = (wn >= gcfg["band_min"]) & (wn <= gcfg["band_max"])
                views.draw_g_fit(ax, wn[gm], self.main.engine.rebuild_fit(wn[gm], gf))
        cv.draw()
        self.info.setText(txt or "No fit available for this pixel.")
        rejected = bool(res) and not res.get("ok", False)
        self.info.setStyleSheet("color:#b71c1c;" if rejected else "color:#555;")
