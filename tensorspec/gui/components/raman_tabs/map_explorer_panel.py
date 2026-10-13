"""Tab 1 · Map Explorer: FWHM(2D) / G height / 2D height map, pixel grid, optical overlay,
orientation and auto-alignment, raw spectrum and fit of a clicked pixel.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Tab 1 - Map Explorer)."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import numpy as np
from PySide6 import QtCore, QtWidgets

from tensorspec.core.io.raman_loaders import MapFile
from tensorspec.core.raman.classification import g_band_cfg, label_of, layer_from_ratio, pixel_ratio
from tensorspec.core.raman.map_geometry import band_height, describe_transform
from tensorspec.core.raman.voigt_2d import asym_of
from tensorspec.plotting.backends.raman_matplotlib_engine import frame_axes, orient_axes, side_colorbar
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
        bar.addWidget(QtWidgets.QLabel("Opacity"))
        bar.addWidget(self.sld_alpha)
        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel("Map"))
        self.cmb_quantity = QtWidgets.QComboBox()
        self.cmb_quantity.addItems(["FWHM(2D)", "G height", "2D height"])
        self.cmb_quantity.setToolTip("G / 2D height: raw peak height above the "
                                     "flanking baseline, no fitting")
        self.cmb_quantity.currentIndexChanged.connect(self.refresh_map)
        bar.addWidget(self.cmb_quantity)
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
                                    "comparison in tab 3")
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
        ax = self.spec_canvas.ax
        wn, inten = self._spec["wn"], self._spec["inten"]
        lo, hi = self._view_range()
        ax.set_xlim(lo, hi)
        if self.chk_yfit.isChecked():
            sel = (wn >= lo) & (wn <= hi) & np.isfinite(inten)
            if sel.sum() >= 2:
                y0, y1 = float(np.min(inten[sel])), float(np.max(inten[sel]))
                pad = 0.08 * (y1 - y0) if y1 > y0 else max(1.0, abs(y1) * 0.1)
                ax.set_ylim(y0 - pad, y1 + pad)
        else:
            good = np.isfinite(inten)
            if good.any():
                y0, y1 = float(np.min(inten[good])), float(np.max(inten[good]))
                pad = 0.08 * (y1 - y0) if y1 > y0 else 1.0
                ax.set_ylim(y0 - pad, y1 + pad)
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
        self.map_canvas.ax.text(0.5, 0.5, "No map loaded",
                                ha="center", va="center", color="#999")
        self.map_canvas.ax.set_axis_off()
        self.map_canvas.draw()
        self.spec_canvas.clear()
        self.spec_canvas.ax.set_axis_off()
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
        ax = cv.ax
        data = self.main.fwhm_map if fwhm else self.height_grid(quantity)
        finite = np.isfinite(data)
        if finite.any():
            vmin = float(np.nanpercentile(data[finite], 2))
            vmax = float(np.nanpercentile(data[finite], 98))
            if vmin == vmax:
                vmin, vmax = vmin - 1, vmax + 1
        else:
            vmin, vmax = 0, 1

        im = ax.imshow(np.ma.masked_invalid(data), origin="lower",
                       extent=mf.extent, cmap=self.main.cfg["cmap"],
                       vmin=vmin, vmax=vmax, interpolation="nearest",
                       aspect="equal", zorder=1)
        cb = side_colorbar(cv.figure, ax, im)
        cb.set_label("FWHM(2D)  [cm$^{-1}$]" if fwhm else "%s  [counts]" % quantity)

        if fwhm and not finite.any():
            # An all-NaN map is otherwise indistinguishable from a map that
            # simply has not been fitted yet, so spell it out.
            tally = self.main.fit_tally or {}
            lines = ["No spectrum passed the quality filters",
                     "(%d pixels, none fitted)" % mf.n_points]
            for why, n in sorted(tally.items(), key=lambda kv: -kv[1])[:4]:
                lines.append("%d × %s" % (n, why))
            lines.append("Adjust under Analysis > Fit parameters")
            ax.text(0.5, 0.5, "\n".join(lines), transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="#b71c1c",
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.6", fc="white",
                              ec="#b71c1c", alpha=0.92))

        self._draw_optical(ax)
        if self.chk_grid.isChecked():
            self._draw_grid(ax, mf)
        if self.sel is not None:
            self._draw_selection(ax, mf, *self.sel)

        ax.set_xlabel("X  [µm]")
        ax.set_ylabel("Y  [µm]")
        ax.set_title("%s   (%d × %d pixels)"
                     % ("FWHM of the 2D band" if fwhm else quantity + " (raw)",
                        mf.shape[1], mf.shape[0]))
        # An equal-aspect image in a wide canvas is anchored right by default
        # once the colorbar has taken its slice, stranding the map in the
        # corner with empty figure beside it.
        ax.set_anchor("C")
        frame_axes(ax, mf, self.main.cfg,
                   self.chk_optical.isChecked() and self.main.optical is not None)
        orient_axes(ax, self.main.cfg)
        cv.draw()

    def _draw_optical(self, ax):
        opt = self.main.optical
        if opt is None or not self.chk_optical.isChecked():
            return
        img = self.main.optical_image
        if img is None:
            return
        ax.imshow(np.ma.masked_invalid(img), origin="lower", extent=opt.extent,
                  cmap="gray", alpha=self.main.cfg["optical_alpha"],
                  interpolation="bilinear", aspect="equal", zorder=2)

    @staticmethod
    def _draw_grid(ax, mf: MapFile):
        left, right, bottom, top = mf.extent
        ny, nx = mf.shape
        for xv in np.linspace(left, right, nx + 1):
            ax.axvline(xv, color="w", lw=0.4, alpha=0.6, zorder=3)
        for yv in np.linspace(bottom, top, ny + 1):
            ax.axhline(yv, color="w", lw=0.4, alpha=0.6, zorder=3)
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

    @staticmethod
    def _draw_selection(ax, mf: MapFile, row: int, col: int):
        left, right, bottom, top = mf.extent
        ny, nx = mf.shape
        dx = (right - left) / nx
        dy = (top - bottom) / ny
        rect = mpatches.Rectangle(
            (left + col * dx, bottom + row * dy), dx, dy,
            fill=False, edgecolor="red", lw=2.0, zorder=4)
        ax.add_patch(rect)

    # -- interaction --------------------------------------------------------
    def _on_click(self, event):
        mf = self.main.mapfile
        if mf is None or event.inaxes is None or event.xdata is None:
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
            ax.text(0.5, 0.5, "No spectrum at this pixel",
                    ha="center", va="center", color="#999")
            ax.set_axis_off()
            cv.draw()
            return

        wn, inten = mf.axis, mf.data[idx]
        self._spec = dict(wn=wn, inten=inten, row=row, col=col)
        ax.plot(wn, inten, lw=0.8, color="#333", label="raw spectrum")
        lo, hi = self.main.cfg["band_min"], self.main.cfg["band_max"]
        ax.axvspan(lo, hi, color="#ffd54f", alpha=0.30, label="2D band region")

        res = self.main.fits[idx] if self.main.fits else None
        txt = ""
        if res and np.isfinite(res.get("center", np.nan)):
            m = (wn >= lo) & (wn <= hi)
            if m.sum() > 3:
                ax.plot(wn[m], self.main.engine.rebuild_fit(wn[m], res), lw=1.4, color="#d81b60",
                        label="Voigt + baseline")
            ax.axvline(res["center"], color="#1e88e5", lw=0.8, ls="--")
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

        ax.set_xlabel("Raman shift  [cm$^{-1}$]")
        ax.set_ylabel("Intensity  [a.u.]")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=3,
                  borderpad=0.3, handlelength=1.4)
        ax.set_title("Pixel (row %d, col %d)" % (row, col), fontsize=9)
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
                ax.plot(wn[gm], self.main.engine.rebuild_fit(wn[gm], gf), lw=1.4, color="#1e88e5",
                        label="G fit")
                ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=4,
                          borderpad=0.3, handlelength=1.4)
        cv.draw()
        self.info.setText(txt or "No fit available for this pixel.")
        rejected = bool(res) and not res.get("ok", False)
        self.info.setStyleSheet("color:#b71c1c;" if rejected else "color:#555;")
