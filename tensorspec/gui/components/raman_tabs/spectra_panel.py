"""Tab 3 · Spectra: pick grid cells, compare their spectra stacked, fit Lorentzians, suggest the
peak count, CSV export.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (SpectraTab)."""
from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING
from typing import List, Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import Patch
from PySide6 import QtCore, QtGui, QtWidgets

from tensorspec.core.raman.classification import label_of
from tensorspec.gui.components.raman_tabs.layout import Canvas
from tensorspec.plotting.raman_figures import class_rgba, n_color, na_color, orient_axes, side_colorbar

if TYPE_CHECKING:
    from tensorspec.gui.components.raman_panel import RamanPanel

TRACE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
                "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


class SpectraTab(QtWidgets.QWidget):
    """Pick grid cells, compare their spectra stacked, fit Lorentzians.

    Selections are kept by spectrum index, so they survive a change of map
    orientation; they are cleared when a new map is loaded.
    """

    PRESETS = [("2D band", (2550.0, 2850.0)), ("G band", (1500.0, 1680.0)),
               ("D band", (1270.0, 1430.0)), ("Full spectrum", None), ("Custom", None)]
    MAX_SPECTRA = 12

    def __init__(self, main: "RamanPanel"):
        super().__init__()
        self.main = main
        self.items: List[dict] = []
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
            "Classification — the result from tab 2, whichever layer mode "
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
        self.chk_auto.setChecked(True)
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
    def resolution(self) -> float:
        mf = self.main.mapfile
        return float((mf.spectral_resolution if mf is not None else None)
                     or self.main.cfg.get("min_fwhm", 5.0))

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
        if any(it["idx"] == idx for it in self.items):
            return
        if len(self.items) >= self.MAX_SPECTRA:
            self.main.status("The comparison holds at most %d spectra." % self.MAX_SPECTRA)
            return
        used = {it["color"] for it in self.items}
        color = next((c for c in TRACE_COLORS if c not in used), TRACE_COLORS[0])
        self.items.append({"idx": int(idx), "color": color, "visible": True, "fit": None})
        if self.chk_auto.isChecked():
            self._fit_item(self.items[-1])
        self._rebuild_list()
        self.refresh_mini()
        self.redraw()

    def _toggle(self, idx: int):
        for k, it in enumerate(self.items):
            if it["idx"] == idx:
                self.items.pop(k)
                self._rebuild_list()
                self.refresh_mini()
                self.redraw()
                return
        self.add_spectrum(idx)

    def clear(self):
        self.items = []
        self._rebuild_list()
        self.refresh_mini()
        self.redraw()

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.lst.selectedIndexes()}, reverse=True)
        for r in rows:
            self.items.pop(r)
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
        self.items = []
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
        mf = self.main.mapfile
        lo, hi = self.rng()
        m = (mf.axis >= lo) & (mf.axis <= hi)
        it["fit"] = ((lo, hi, self.spn_n.value()),
                     self.main.engine.fit_band("lorentz_n", mf.axis[m], mf.data[it["idx"]][m],
                                                {"n": self.spn_n.value()}))

    def fit_all(self, redraw=True):
        if self.main.mapfile is None:
            return
        for it in self.items:
            if it["visible"]:
                self._fit_item(it)
        self._fill_table()
        if redraw:
            self.redraw()

    def suggest(self):
        mf = self.main.mapfile
        if mf is None or not self.items:
            return
        lo, hi = self.rng()
        m = (mf.axis >= lo) & (mf.axis <= hi)
        res = self.resolution()
        picks = []
        for it in self.items:
            if it["visible"]:
                n, _ = self.main.engine.suggest_peak_count(mf.axis[m], mf.data[it["idx"]][m], res)
                picks.append((it, n))
        if not picks:
            return
        counts = [n for _, n in picks]
        best = max(set(counts), key=counts.count)
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
            ax.text(0.5, 0.5, "No map loaded", ha="center", va="center", color="#999")
            ax.set_axis_off()
            cv.draw()
            return
        cfg = self.main.cfg
        view = self.cmb_view.currentText()
        opt = self.main.optical
        with_opt = (self.chk_opt.isChecked() and opt is not None
                    and self.main.optical_image is not None)
        alpha = self.sld_alpha.value() / 100.0 if with_opt else 1.0
        self.sld_alpha.setEnabled(with_opt)
        if with_opt:
            ax.imshow(np.ma.masked_invalid(self.main.optical_image), origin="lower",
                      extent=opt.extent, cmap="gray", aspect="equal", zorder=0)
        handles, note, title = [], "", view

        if view == "Classification":
            tc = self.main.tab_class
            if tc.layers is None or tc.layers.shape != mf.shape:
                note = "No classification yet —\nrun 'Classify map' in tab 2"
            else:
                rgba = class_rgba(tc.layers, tc.calls, cfg)
                rgba[..., 3] *= alpha
                ax.imshow(rgba, origin="lower", extent=mf.extent,
                          interpolation="nearest", aspect="equal", zorder=1)
                present = sorted({(int(tc.layers[r, c]), tc.calls[r, c])
                                  for r in range(mf.shape[0]) for c in range(mf.shape[1])},
                                 key=lambda t: (t[0] < 0, t[0], str(t[1])))
                for lay, call in present:
                    if call is None or (lay < 0 and call != "N/A"):
                        continue
                    col = (na_color(lay, cfg) if call == "N/A"
                           else cfg["colors"][lay][call])
                    handles.append(Patch(facecolor=col, edgecolor="#555",
                                         label=label_of(lay, call)))
                handles.append(Patch(facecolor=cfg["unclassified_color"],
                                     edgecolor="#555", label="Unclassified"))
                mode = {"manual": "Manual %dL" % int(cfg.get("manual_layer", 3)),
                        "auto": "Auto, I(2D)/I(G)",
                        "map": "layer regions"}.get(cfg.get("layer_mode"), "")
                title = "Classification (%s)" % mode
        elif view == "Layer map (N)":
            g = self.main.layer_grid() if self.main.layer_by_spec is not None else None
            if g is None or not (g > 0).any():
                note = "No layer map yet —\ndefine layer regions in tab 2"
            else:
                rgba = np.zeros(g.shape + (4,))
                for n in sorted(set(g[g > 0].tolist())):
                    rgba[g == n] = mcolors.to_rgba(n_color(n, cfg), alpha)
                    handles.append(Patch(facecolor=n_color(n, cfg), edgecolor="#555",
                                         label="%dL" % n))
                ax.imshow(rgba, origin="lower", extent=mf.extent,
                          interpolation="nearest", aspect="equal", zorder=1)
                title = "Layer map (N per region)"
        if view == "FWHM(2D)" or note:
            data = (self.main.fwhm_map if self.main.fwhm_map is not None
                    else mf.greyscale())
            im = ax.imshow(np.ma.masked_invalid(data), origin="lower", extent=mf.extent,
                           cmap=cfg.get("cmap", "viridis"), interpolation="nearest",
                           aspect="equal", alpha=alpha, zorder=1)
            if not note:
                cb = side_colorbar(cv.figure, ax, im, size="5%", pad=0.05)
                cb.set_label("FWHM(2D) [cm$^{-1}$]", fontsize=7)
                cb.ax.tick_params(labelsize=7)
                title = "FWHM(2D)"
        if note:
            ax.text(0.5, 0.5, note, transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#b71c1c", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#b71c1c", alpha=0.9))

        left, right, bottom, top = mf.extent
        ny, nx = mf.shape
        dx, dy = (right - left) / nx, (top - bottom) / ny
        for xv in np.linspace(left, right, nx + 1):
            ax.axvline(xv, color="w", lw=0.3, alpha=0.5, zorder=2)
        for yv in np.linspace(bottom, top, ny + 1):
            ax.axhline(yv, color="w", lw=0.3, alpha=0.5, zorder=2)
        for it in self.items:
            r, c = np.argwhere(mf.index_map == it["idx"])[0]
            ax.add_patch(mpatches.Rectangle((left + c * dx, bottom + r * dy), dx, dy,
                                            fill=False, edgecolor=it["color"],
                                            lw=2.6 if it["visible"] else 1.0,
                                            ls="-" if it["visible"] else ":", zorder=4))
        pad = 0.08 * max(right - left, top - bottom) if with_opt else 0.0
        ax.set_xlim(left - pad, right + pad)
        ax.set_ylim(bottom - pad, top + pad)
        ax.set_title(title + " — click cells", fontsize=9)
        ax.tick_params(labelsize=7)
        if handles:
            ax.legend(handles=handles, fontsize=7, loc="upper center",
                      bbox_to_anchor=(0.5, -0.08), ncol=min(4, len(handles)),
                      frameon=False, handlelength=1.2, columnspacing=0.8)
        orient_axes(ax, cfg)
        cv.draw()

    def redraw(self):
        cv = self.plot
        cv.clear()
        ax = cv.ax
        mf = self.main.mapfile
        vis = [it for it in self.items if it["visible"]]
        if mf is None or not vis:
            ax.text(0.5, 0.5, "Select grid cells on the left to compare their spectra",
                    ha="center", va="center", color="#999", transform=ax.transAxes)
            ax.set_axis_off()
            cv.draw()
            self._fill_table()
            return
        lo, hi = self.rng()
        m = (mf.axis >= lo) & (mf.axis <= hi)
        x = mf.axis[m]
        norm = self.chk_norm.isChecked()
        scaled = []
        for it in vis:
            y = mf.data[it["idx"]][m]
            ymin = float(np.nanpercentile(y, 2))
            ymax = float(np.nanmax(y))
            k = (ymax - ymin) if norm and ymax > ymin else 1.0
            scaled.append((it, y, ymin if norm else 0.0, k))
        span = max(((np.nanmax(y) - (b if norm else np.nanmin(y))) / k) for _, y, b, k in scaled)
        step = self.sld_off.value() / 100.0 * span
        for j, (it, y, b, k) in enumerate(scaled):
            off = j * step
            ax.plot(x, (y - b) / k + off, color=it["color"], lw=0.9, label=self._name(it["idx"]))
            fr = it.get("fit")
            txt = ""
            if fr is not None and fr[0] == (lo, hi, self.spn_n.value()) and fr[1]["ok"]:
                res = fr[1]
                fx, fy = res["curve"]
                if self.chk_fit.isChecked():
                    ax.plot(fx, (fy - b) / k + off, color="k", lw=1.1, ls="--")
                if self.chk_comp.isChecked() and len(res["peaks"]) > 1:
                    sl, ic, xm = res["baseline"]
                    basel = sl * (fx - xm) + ic
                    for (h, c, w) in res["components"]:
                        comp = h * (w / 2) ** 2 / ((fx - c) ** 2 + (w / 2) ** 2)
                        ax.fill_between(fx, (basel - b) / k + off, (basel + comp - b) / k + off,
                                        color=it["color"], alpha=0.18, lw=0)
                txt = "FWHM " + " / ".join("%.1f" % p["fwhm"] for p in res["peaks"])
                if len(res["peaks"]) > 1:
                    txt += "  (band %.1f)" % res["band_fwhm"]
            ax.text(x[-1], float(np.nanmedian((y[-8:] - b) / k)) + off, "  " + txt,
                    color=it["color"], fontsize=8, va="bottom", ha="right")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Raman shift  [cm$^{-1}$]")
        ax.set_ylabel("Intensity (normalised, stacked)" if norm else "Intensity [counts], stacked")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
        ax.set_title("%d spectra · %d Lorentzian peak%s · resolution %.1f cm$^{-1}$"
                     % (len(vis), self.spn_n.value(), "" if self.spn_n.value() == 1 else "s",
                        self.resolution()), fontsize=9)
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
        lo, hi = self.rng()
        m = (mf.axis >= lo) & (mf.axis <= hi)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["raman_shift_cm-1"] + [self._name(it["idx"]) for it in vis])
            for j, xv in enumerate(mf.axis[m]):
                w.writerow(["%.3f" % xv] + ["%.3f" % mf.data[it["idx"]][m][j] for it in vis])
        fpath = path[:-4] + "_fits.csv"
        with open(fpath, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["spectrum", "x_um", "y_um", "range_lo", "range_hi", "n_peaks", "peak",
                        "centre_cm-1", "fwhm_cm-1", "fwhm_err_cm-1", "height", "area",
                        "band_fwhm_cm-1", "r2", "bic"])
            for it in vis:
                fr = it.get("fit")
                if fr is None or not fr[1]["ok"]:
                    continue
                (flo, fhi, n), r = fr
                for k, pk in enumerate(r["peaks"]):
                    w.writerow([self._name(it["idx"]), "%.3f" % mf.x[it["idx"]],
                                "%.3f" % mf.y[it["idx"]], flo, fhi, n, k + 1,
                                "%.2f" % pk["center"], "%.2f" % pk["fwhm"],
                                "%.2f" % pk["err_fwhm"], "%.2f" % pk["height"],
                                "%.1f" % pk["area"], "%.2f" % r["band_fwhm"],
                                "%.5f" % r["r2"], "%.1f" % r["bic"]])
        self.main.status("Saved %s and %s" % (os.path.basename(path), os.path.basename(fpath)))
