"""Tab 3 · Spectra: pick grid cells, compare their spectra stacked, fit Lorentzians, suggest the
peak count, CSV export. Figures are drawn by plotting/backends/raman_views/spectra.py.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (SpectraTab)."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import List, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from tensorspec.core.raman.classification import label_of
from tensorspec.plotting.backends.raman_matplotlib_engine import draw_placeholder
from tensorspec.plotting.backends.raman_views import spectra as views
from tensorspec.plotting.viewers.raman_canvas import Canvas

if TYPE_CHECKING:
    from tensorspec.gui.components.raman_panel import RamanPanel

class SpectraTab(QtWidgets.QWidget):
    """Pick grid cells, compare their spectra stacked, fit Lorentzians.

    Selections are kept by spectrum index, so they survive a change of map
    orientation; they are cleared when a new map is loaded.
    """

    PRESETS = [("2D band", (2550.0, 2850.0)), ("G band", (1500.0, 1680.0)),
               ("D band", (1270.0, 1430.0)), ("Full spectrum", None), ("Custom", None)]

    def __init__(self, main: "RamanPanel"):
        super().__init__()
        self.main = main
        self._build()

    # -- UI ------------------------------------------------------------------
    def _build(self):
        root = QtWidgets.QHBoxLayout(self)

        self.split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(self.split)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)
        vr = QtWidgets.QHBoxLayout()
        vr.addWidget(QtWidgets.QLabel("Show"))
        self.cmb_view = QtWidgets.QComboBox()
        self.cmb_view.addItems(["FWHM(2D)", "Classification", "Layer map (N)"])
        self.cmb_view.setToolTip(
            "FWHM(2D) — the fitted 2D width, no classification\n"
            "Classification — the result from 3 · Analysis ▸ Classification, whichever layer mode "
            "produced it (one N, Auto, or layer regions)\n"
            "Layer map (N) — the per-region layer counts from the regions dialog")
        self.cmb_view.currentIndexChanged.connect(self.refresh_mini)
        vr.addWidget(self.cmb_view, 1)
        lv.addLayout(vr)
        orow = QtWidgets.QHBoxLayout()
        self.chk_opt = QtWidgets.QCheckBox("Optical image")
        self.chk_opt.setChecked(True)
        self.chk_opt.toggled.connect(self.refresh_mini)
        orow.addWidget(self.chk_opt)
        orow.addWidget(QtWidgets.QLabel("map opacity"))
        self.sld_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_alpha.setRange(10, 100)
        self.sld_alpha.setValue(75)
        self.sld_alpha.setToolTip("Opacity of the map over the optical image")
        self.sld_alpha.valueChanged.connect(self.refresh_mini)
        orow.addWidget(self.sld_alpha, 1)
        lv.addLayout(orow)
        hint = QtWidgets.QLabel("Click grid cells to add or remove their spectra")
        hint.setWordWrap(True)
        lv.addWidget(hint)
        self.mini = Canvas(figsize=(4.4, 4.6))
        self.mini.setMinimumWidth(240)
        self.mini.canvas.mpl_connect("button_press_event", self._mini_click)
        self.mini.setMinimumHeight(360)
        lv.addWidget(self.mini, 5)
        self.lst = QtWidgets.QListWidget()
        self.lst.setMaximumHeight(170)
        self.lst.itemChanged.connect(self._item_toggled)
        lv.addWidget(self.lst, 1)
        br = QtWidgets.QHBoxLayout()
        b_rm = QtWidgets.QPushButton("Remove selected")
        b_rm.clicked.connect(self._remove_selected)
        b_cl = QtWidgets.QPushButton("Clear all")
        b_cl.clicked.connect(self.clear)
        br.addWidget(b_rm)
        br.addWidget(b_cl)
        lv.addLayout(br)
        self.split.addWidget(left)

        right = QtWidgets.QVBoxLayout()
        r1 = QtWidgets.QHBoxLayout()
        r1.addWidget(QtWidgets.QLabel("Range"))
        self.cmb_range = QtWidgets.QComboBox()
        self.cmb_range.addItems([n for n, _ in self.PRESETS])
        self.cmb_range.currentIndexChanged.connect(self._preset_changed)
        r1.addWidget(self.cmb_range)
        r1.addWidget(QtWidgets.QLabel("from"))
        self.spn_lo = QtWidgets.QDoubleSpinBox()
        r1.addWidget(self.spn_lo)
        r1.addWidget(QtWidgets.QLabel("to"))
        self.spn_hi = QtWidgets.QDoubleSpinBox()
        r1.addWidget(self.spn_hi)
        for sp, v in ((self.spn_lo, 2550.0), (self.spn_hi, 2850.0)):
            sp.setRange(0, 10000)
            sp.setDecimals(0)
            sp.setSingleStep(10)
            sp.setValue(v)
            sp.setSuffix(" cm⁻¹")
            sp.setFixedWidth(100)
            sp.valueChanged.connect(self._range_edited)
        self._lbl_stack = QtWidgets.QLabel("Stack offset")
        self.sld_off = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_off.setRange(0, 150)
        self.sld_off.setValue(60)
        self.sld_off.setFixedWidth(120)
        self.sld_off.valueChanged.connect(self.redraw)
        self.chk_norm = QtWidgets.QCheckBox("Normalise")
        self.chk_norm.setToolTip("Scale each spectrum to its own maximum within the range")
        self.chk_norm.setChecked(True)
        self.chk_norm.toggled.connect(self.redraw)
        r1.addStretch(1)
        right.addLayout(r1)

        r2 = QtWidgets.QHBoxLayout()
        r2.addWidget(QtWidgets.QLabel("Lorentzian peaks"))
        self.spn_n = QtWidgets.QSpinBox()
        self.spn_n.setRange(1, 4)
        self.spn_n.valueChanged.connect(self._n_changed)
        r2.addWidget(self.spn_n)
        self.btn_suggest = QtWidgets.QPushButton("Suggest peak count")
        self.btn_suggest.setToolTip(
            "Fewest peaks within 10 BIC units of the best fit, rejecting any fit "
            "with a component narrower than the spectral resolution")
        self.btn_suggest.clicked.connect(self.suggest)
        r2.addWidget(self.btn_suggest)
        self.btn_fit = QtWidgets.QPushButton("Fit")
        self.btn_fit.setStyleSheet("font-weight:bold;")
        self.btn_fit.clicked.connect(self.fit_all)
        r2.addWidget(self.btn_fit)
        self.chk_auto = QtWidgets.QCheckBox("Refit on change")
        self.chk_auto.setChecked(False)
        r2.addWidget(self.chk_auto)
        r2.addStretch(1)
        right.addLayout(r2)
        r2 = QtWidgets.QHBoxLayout()            # second line: display and export
        self.chk_fit = QtWidgets.QCheckBox("Show fits")
        self.chk_fit.setChecked(True)
        self.chk_fit.toggled.connect(self.redraw)
        self.chk_comp = QtWidgets.QCheckBox("Show components")
        self.chk_comp.setChecked(True)
        self.chk_comp.toggled.connect(self.redraw)
        r2.addWidget(self._lbl_stack)
        r2.addWidget(self.sld_off)
        r2.addWidget(self.chk_norm)
        r2.addSpacing(12)
        r2.addWidget(self.chk_fit)
        r2.addWidget(self.chk_comp)
        r2.addStretch(1)
        self.btn_csv = QtWidgets.QPushButton("Export CSV…")
        self.btn_csv.clicked.connect(self.export_csv)
        r2.addWidget(self.btn_csv)
        right.addLayout(r2)

        self.plot = Canvas(figsize=(8, 5))
        right.addWidget(self.plot, 3)
        self.lbl = QtWidgets.QLabel("")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("font-size:11px; color:#555;")
        right.addWidget(self.lbl)
        self.tbl = QtWidgets.QTableWidget(0, 9)
        self.tbl.setHorizontalHeaderLabels(["Spectrum", "Peak", "Centre", "FWHM",
                                            "± FWHM", "Height", "Area", "Band FWHM", "R²"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tbl.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        right.addWidget(self.tbl, 2)
        rightw = QtWidgets.QWidget()
        rightw.setLayout(right)
        self.split.addWidget(rightw)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 3)
        self.split.setSizes([420, 1100])
        self.refresh_mini()
        self.redraw()

    # -- helpers ------------------------------------------------------------
    @property
    def items(self) -> List[dict]:
        """The compared cells, held by the service: {idx, color, visible, fit}."""
        return self.main.service.cells

    def resolution(self) -> float:
        return self.main.service.cells_resolution()

    def rng(self) -> Tuple[float, float]:
        lo, hi = self.spn_lo.value(), self.spn_hi.value()
        return (lo, hi) if hi > lo else (lo, lo + 10.0)

    def _name(self, idx: int) -> str:
        mf = self.main.mapfile
        r, c = np.argwhere(mf.index_map == idx)[0]
        name = "R%dC%d (%.1f, %.1f) µm" % (r, c, mf.x[idx], mf.y[idx])
        tc = self.main.tab_class
        if tc.layers is not None:
            name += "  " + label_of(int(tc.layers[r, c]), tc.calls[r, c])
        return name

    # -- selection ---------------------------------------------------------------
    def add_spectrum(self, idx: int):
        why = self.main.service.cells_add(idx)
        if why:
            if why != "already there":
                self.main.status(why)
            return
        if self.chk_auto.isChecked():
            self._fit_item(self.items[-1])
        self._rebuild_list()
        self.refresh_mini()
        self.redraw()

    def _toggle(self, idx: int):
        if any(it["idx"] == idx for it in self.items):
            self.main.service.cells_toggle(idx)
            self._rebuild_list()
            self.refresh_mini()
            self.redraw()
            return
        self.add_spectrum(idx)

    def clear(self):
        self.items.clear()
        self._rebuild_list()
        self.refresh_mini()
        self.redraw()

    def _remove_selected(self):
        self.main.service.cells_remove({i.row() for i in self.lst.selectedIndexes()})
        self._rebuild_list()
        self.refresh_mini()
        self.redraw()

    def _rebuild_list(self):
        self.lst.blockSignals(True)
        self.lst.clear()
        for it in self.items:
            li = QtWidgets.QListWidgetItem(self._name(it["idx"]))
            li.setFlags(li.flags() | QtCore.Qt.ItemIsUserCheckable)
            li.setCheckState(QtCore.Qt.Checked if it["visible"] else QtCore.Qt.Unchecked)
            pm = QtGui.QPixmap(14, 14)
            pm.fill(QtGui.QColor(it["color"]))
            li.setIcon(QtGui.QIcon(pm))
            self.lst.addItem(li)
        self.lst.blockSignals(False)

    def _item_toggled(self, li):
        r = self.lst.row(li)
        if 0 <= r < len(self.items):
            self.items[r]["visible"] = li.checkState() == QtCore.Qt.Checked
            self.refresh_mini()
            self.redraw()

    def _mini_click(self, event):
        mf = self.main.mapfile
        if mf is None or event.inaxes is None or event.xdata is None:
            return
        left, right, bottom, top = mf.extent
        ny, nx = mf.shape
        col = int(np.floor((event.xdata - left) / (right - left) * nx))
        row = int(np.floor((event.ydata - bottom) / (top - bottom) * ny))
        if 0 <= col < nx and 0 <= row < ny and mf.index_map[row, col] >= 0:
            self._toggle(int(mf.index_map[row, col]))

    def reset(self):
        self.items.clear()
        self._rebuild_list()
        self.refresh_mini()
        self.redraw()

    # -- range ------------------------------------------------------------------
    def _preset_changed(self):
        name = self.cmb_range.currentText()
        rngd = dict(self.PRESETS).get(name)
        mf = self.main.mapfile
        if name == "Full spectrum" and mf is not None:
            rngd = (float(mf.axis[0]), float(mf.axis[-1]))
        if rngd is not None:
            for sp, v in ((self.spn_lo, rngd[0]), (self.spn_hi, rngd[1])):
                sp.blockSignals(True)
                sp.setValue(v)
                sp.blockSignals(False)
            self._changed()

    def _range_edited(self):
        if self.cmb_range.currentText() != "Custom":
            self.cmb_range.blockSignals(True)
            self.cmb_range.setCurrentText("Custom")
            self.cmb_range.blockSignals(False)
        self._changed()

    def _n_changed(self):
        self._changed()

    def _changed(self):
        if self.chk_auto.isChecked():
            self.fit_all(redraw=False)
        self.redraw()

    # -- fitting ----------------------------------------------------------------
    def _fit_item(self, it):
        self.main.service.cell_fit(it, *self.rng(), self.spn_n.value())

    def fit_all(self, redraw=True):
        if self.main.mapfile is None:
            return
        self.main.service.cells_fit(*self.rng(), self.spn_n.value())
        self._fill_table()
        if redraw:
            self.redraw()

    def suggest(self):
        if self.main.mapfile is None or not self.items:
            return
        res = self.resolution()
        picks, best = self.main.service.cells_suggest(*self.rng())
        if not picks:
            return
        self.spn_n.setValue(best)
        self.lbl.setText("Suggested peak counts (resolution %.1f cm⁻¹): %s  →  using %d."
                         % (res, ",  ".join("%s: %d" % (self._name(it["idx"]).split()[0], n)
                                            for it, n in picks), best))
        self.fit_all()

    # -- drawing ---------------------------------------------------------------
    def refresh_mini(self):
        cv = self.mini
        cv.clear()
        ax = cv.ax
        mf = self.main.mapfile
        if mf is None or mf.layout != "spectra":
            draw_placeholder(ax, "No map loaded")
            cv.draw()
            return
        cfg = self.main.cfg
        view = self.cmb_view.currentText()
        opt = self.main.optical
        with_opt = (self.chk_opt.isChecked() and opt is not None
                    and self.main.optical_image is not None)
        alpha = self.sld_alpha.value() / 100.0 if with_opt else 1.0
        self.sld_alpha.setEnabled(with_opt)
        tc = self.main.tab_class
        layer_grid = (self.main.layer_grid() if view == "Layer map (N)"
                      and self.main.layer_by_spec is not None else None)
        views.draw_cell_picker(cv.figure, ax, mf, view, cfg, self.items, opt,
                               self.main.optical_image, with_opt, alpha,
                               tc.layers, tc.calls, layer_grid, self.main.fwhm_map)
        cv.draw()

    def redraw(self):
        cv = self.plot
        cv.clear()
        ax = cv.ax
        mf = self.main.mapfile
        vis = [it for it in self.items if it["visible"]]
        if mf is None or not vis:
            views.draw_no_selection(ax)
            cv.draw()
            self._fill_table()
            return
        lo, hi = self.rng()
        views.draw_stacked_selection(ax, mf, vis, [self._name(it["idx"]) for it in vis], lo, hi,
                                     self.spn_n.value(), self.chk_norm.isChecked(),
                                     self.sld_off.value() / 100.0, self.chk_fit.isChecked(),
                                     self.chk_comp.isChecked(), self.resolution())
        cv.draw()
        self._fill_table()

    def _fill_table(self):
        lo, hi = self.rng()
        res_lim = self.resolution()
        rows = []
        for it in self.items:
            fr = it.get("fit")
            if not it["visible"] or fr is None or fr[0] != (lo, hi, self.spn_n.value()):
                continue
            r = fr[1]
            if not r["ok"]:
                rows.append((it, None, r["note"], r))
                continue
            for k, pk in enumerate(r["peaks"]):
                rows.append((it, k, pk, r))
        self.tbl.setRowCount(len(rows))
        for i, (it, k, pk, r) in enumerate(rows):
            name = self._name(it["idx"]).split("  ")[0]
            if k is None:
                vals = [name, "—", pk, "", "", "", "", "", ""]
            else:
                narrow = pk["fwhm"] < res_lim
                vals = [name, "%d" % (k + 1), "%.1f" % pk["center"],
                        "%.1f%s" % (pk["fwhm"], " ⚠" if narrow else ""),
                        "%.1f" % pk["err_fwhm"] if np.isfinite(pk["err_fwhm"]) else "—",
                        "%.1f" % pk["height"], "%.0f" % pk["area"],
                        "%.1f" % r["band_fwhm"] if k == 0 else "", "%.4f" % r["r2"] if k == 0 else ""]
            for c, v in enumerate(vals):
                ti = QtWidgets.QTableWidgetItem(v)
                ti.setTextAlignment(QtCore.Qt.AlignCenter)
                if c == 0:
                    ti.setForeground(QtGui.QBrush(QtGui.QColor(it["color"])))
                if c == 3 and k is not None and pk["fwhm"] < res_lim:
                    ti.setForeground(QtGui.QBrush(QtGui.QColor("#b71c1c")))
                    ti.setToolTip("Narrower than the %.1f cm⁻¹ spectral resolution — "
                                  "this component is fitting noise, not a band. "
                                  "Try fewer peaks." % res_lim)
                self.tbl.setItem(i, c, ti)

    # -- export -------------------------------------------------------------------
    def export_csv(self):
        mf = self.main.mapfile
        vis = [it for it in self.items if it["visible"]]
        if mf is None or not vis:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export spectra and fits", self.main.suggest_name("_spectra.csv"),
            "CSV file (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        fpath = self.main.service.cells_export(path, *self.rng(), [self._name(it["idx"]) for it in vis])
        self.main.status("Saved %s and %s" % (os.path.basename(path), os.path.basename(fpath)))
