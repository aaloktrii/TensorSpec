"""Classification helper dialog: layer regions (terraces) from optical contrast, each given its
layer count; save / load regions.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Tab 2 - Classification: LayerRegionsDialog)."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from tensorspec.core.raman.classification import terrace_edges
from tensorspec.core.raman.constants import APP_NAME
from tensorspec.core.raman.map_geometry import describe_transform, kmeans_1d, optical_contrast
from tensorspec.gui.components.raman_tabs.layout import Canvas, fit_to_screen
from tensorspec.plotting.raman_figures import orient_axes

if TYPE_CHECKING:
    from tensorspec.gui.components.raman_panel import RamanPanel

LAYER_NM = 0.335          # graphene interlayer spacing
REGION_COLORS = ["#1e88e5", "#fb8c00", "#8e24aa", "#43a047", "#e53935", "#00acc1"]


class LayerRegionsDialog(QtWidgets.QDialog):
    """Define terraces from optical contrast and give each its layer count.

    Contrast groups the pixels into regions; it cannot count layers. Each
    region's N is entered from AFM, either directly (a terrace step worked
    out in Gwyddion) or as a height measured down to bare SiO2, converted
    with the editable SiO2 offset. Non-modal, so it can stay open beside
    Gwyddion; Apply re-runs the classification behind it.
    """

    def __init__(self, main: "RamanPanel", parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("Define layer regions")
        fit_to_screen(self, 1400, 780)
        self.main = main
        mf = main.mapfile
        self.region_of = np.full(mf.n_points, -1, dtype=int)
        self.levels = None
        self.contrast = None
        self.selected = -1
        self.rows = []
        self._texts = []

        root = QtWidgets.QHBoxLayout(self)
        self.canvas = Canvas(figsize=(7.4, 6.8))
        root.addWidget(self.canvas, 1)
        side = QtWidgets.QVBoxLayout()
        root.addLayout(side, 1)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Regions"))
        self.spn_k = QtWidgets.QSpinBox()
        self.spn_k.setRange(1, len(REGION_COLORS))
        self.spn_k.setValue(int(main.cfg.get("contrast_levels", 2)))
        top.addWidget(self.spn_k)
        top.addSpacing(8)
        top.addWidget(QtWidgets.QLabel("Substrate if |contrast| <"))
        self.spn_cut = QtWidgets.QDoubleSpinBox()
        self.spn_cut.setRange(0.0, 1.0)
        self.spn_cut.setDecimals(3)
        self.spn_cut.setSingleStep(0.005)
        self.spn_cut.setValue(float(main.cfg.get("contrast_cut", 0.03)))
        top.addWidget(self.spn_cut)
        self.btn_seg = QtWidgets.QPushButton("Segment")
        self.btn_seg.clicked.connect(self.segment)
        top.addWidget(self.btn_seg)
        top.addStretch(1)
        side.addLayout(top)

        self.lbl_place = QtWidgets.QLabel()
        self.lbl_place.setWordWrap(True)
        self.lbl_place.setStyleSheet("font-size:11px;")
        side.addWidget(self.lbl_place)

        self.tbl = QtWidgets.QTableWidget(0, 7)
        self.tbl.setHorizontalHeaderLabels(["Region", "Contrast", "Pixels",
                                            "FWHM", "Entry", "Value", "N"])
        self.tbl.horizontalHeaderItem(3).setToolTip("Median 2D FWHM of the region, cm⁻¹")
        self.tbl.verticalHeader().setVisible(False)
        hh = self.tbl.horizontalHeader()
        # Every column visible at once: Value and N are the ones that matter,
        # so nothing may be pushed behind a scrollbar.
        for col in range(7):
            hh.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents
                                    if col < 4 else QtWidgets.QHeaderView.Stretch)
        self.tbl.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl.itemSelectionChanged.connect(self._row_selected)
        self.tbl.setMinimumHeight(190)
        side.addWidget(self.tbl, 1)

        orow = QtWidgets.QHBoxLayout()
        orow.addWidget(QtWidgets.QLabel("SiO₂ offset"))
        self.spn_off = QtWidgets.QDoubleSpinBox()
        self.spn_off.setRange(0.0, 3.0)
        self.spn_off.setDecimals(2)
        self.spn_off.setSingleStep(0.05)
        self.spn_off.setSuffix(" nm")
        self.spn_off.setValue(float(main.cfg.get("sio2_offset", 0.5)))
        self.spn_off.setToolTip(
            "Extra apparent height of a flake-to-SiO₂ step in AFM (tip and "
            "water-layer effects), typically 0.3–1.0 nm. Only used for rows "
            "entered as 'Height from SiO₂'.")
        self.spn_off.valueChanged.connect(self.update_status)
        orow.addWidget(self.spn_off)
        orow.addWidget(QtWidgets.QLabel("   N = (height − offset) / %.3f nm" % LAYER_NM))
        orow.addStretch(1)
        side.addLayout(orow)

        self.chk_edges = QtWidgets.QCheckBox("Exclude step-edge pixels")
        self.chk_edges.setChecked(bool(main.cfg.get("exclude_edges", True)))
        self.chk_edges.setToolTip(
            "Pixels next to a region of different N: the laser spot spans "
            "both terraces there, so the spectrum mixes two thicknesses.")
        self.chk_edges.toggled.connect(self.update_status)
        side.addWidget(self.chk_edges)

        self.lbl_warn = QtWidgets.QLabel()
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setStyleSheet("font-size:11px;")
        side.addWidget(self.lbl_warn)

        bot = QtWidgets.QHBoxLayout()
        b_sv = QtWidgets.QPushButton("Save…")
        b_sv.clicked.connect(self.save_regions)
        b_ld = QtWidgets.QPushButton("Load…")
        b_ld.clicked.connect(self.load_regions)
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.setStyleSheet("font-weight:bold;")
        self.btn_apply.clicked.connect(self.apply)
        b_close = QtWidgets.QPushButton("Close")
        b_close.clicked.connect(self.close)
        for w in (b_sv, b_ld):
            bot.addWidget(w)
        bot.addStretch(1)
        bot.addWidget(self.btn_apply)
        bot.addWidget(b_close)
        side.addLayout(bot)

        self._build_figure()
        self.canvas.canvas.mpl_connect("button_press_event", self._clicked)

        if main.optical is None or main.optical_image is None:
            self.lbl_place.setText("No camera image loaded — regions come from optical "
                                   "contrast, so load one in tab 1 first.")
            self.lbl_place.setStyleSheet("color:#b71c1c; font-size:11px;")
            for w in (self.spn_k, self.spn_cut, self.btn_seg, self.btn_apply):
                w.setEnabled(False)
            return
        if main.cfg.get("map_transform") is None:
            self.lbl_place.setText("⚠ The map's offset has not been aligned yet — run "
                                   "Auto-align in tab 1 first, or contrast will be "
                                   "read from slightly the wrong place.")
            self.lbl_place.setStyleSheet("color:#b71c1c; font-size:11px;")
        else:
            self.lbl_place.setText("Placement: %s" %
                                   describe_transform(main.cfg["map_transform"]))
            self.lbl_place.setStyleSheet("color:#555; font-size:11px;")
        self.segment()

    # -- figure -------------------------------------------------------------
    def _build_figure(self):
        mf = self.main.mapfile
        fig = self.canvas.figure
        fig.clear()
        gsp = fig.add_gridspec(3, 1)
        self.ax = fig.add_subplot(gsp[:2, 0])
        self.axh = fig.add_subplot(gsp[2, 0])
        opt = self.main.optical
        if opt is not None and self.main.optical_image is not None:
            self.ax.imshow(np.ma.masked_invalid(self.main.optical_image),
                           origin="lower", extent=opt.extent, cmap="gray",
                           aspect="equal", zorder=0)
        ny, nx = mf.shape
        self.overlay = self.ax.imshow(np.zeros((ny, nx, 4)), origin="lower",
                                      extent=mf.extent, interpolation="nearest",
                                      aspect="equal", zorder=1)
        left, right, bottom, top = mf.extent
        pad = 0.15 * max(right - left, top - bottom)
        self.ax.set_xlim(left - pad, right + pad)
        self.ax.set_ylim(bottom - pad, top + pad)
        self.ax.set_xlabel("X [µm]")
        self.ax.set_ylabel("Y [µm]")
        self.ax.set_title("Regions (click one to select its row)", fontsize=10)
        orient_axes(self.ax, self.main.cfg)

    def _n_by_spec(self) -> np.ndarray:
        n = np.zeros(self.main.mapfile.n_points, dtype=int)
        for j in range(len(self.rows)):
            nj = self.row_n(j)[1]
            n[self.region_of == j] = max(nj, 0)
        return n

    def refresh_map(self):
        mf = self.main.mapfile
        ny, nx = mf.shape
        rgba = np.zeros((ny, nx, 4))
        reg = np.full((ny, nx), -1, dtype=int)
        ok = mf.index_map >= 0
        reg[ok] = self.region_of[mf.index_map[ok]]
        for j in range(len(self.rows)):
            a = 0.85 if j == self.selected else 0.5
            rgba[reg == j] = mcolors.to_rgba(REGION_COLORS[j % len(REGION_COLORS)], a)
        if self.chk_edges.isChecked():
            ng = np.zeros((ny, nx), dtype=int)
            nb = self._n_by_spec()
            ng[ok] = nb[mf.index_map[ok]]
            rgba[terrace_edges(ng)] = (1.0, 1.0, 1.0, 0.85)
        self.overlay.set_data(rgba)
        for t in self._texts:
            t.remove()
        self._texts = []
        for j in range(len(self.rows)):
            m = self.region_of == j
            if m.any():
                self._texts.append(self.ax.text(
                    float(np.median(mf.x[m])), float(np.median(mf.y[m])), "R%d" % (j + 1),
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc=REGION_COLORS[j % 6],
                              ec="none", alpha=0.9)))

        self.axh.clear()
        if self.contrast is not None:
            fin = self.contrast[np.isfinite(self.contrast)]
            if fin.size:
                self.axh.hist(fin, bins=max(10, min(60, fin.size // 3)),
                              color="#90a4ae", edgecolor="white", linewidth=0.4)
            cut = self.spn_cut.value()
            for v in (-cut, cut):
                self.axh.axvline(v, color="#555", ls=":", lw=1)
            if self.levels is not None:
                top = self.axh.get_ylim()[1]
                for j, c in enumerate(self.levels):
                    self.axh.axvline(c, color=REGION_COLORS[j % 6], lw=1.6)
                    self.axh.text(c, top * 0.92, " R%d" % (j + 1),
                                  color=REGION_COLORS[j % 6], fontsize=9,
                                  fontweight="bold")
                srt = np.sort(self.levels)
                for a, b in zip(srt[:-1], srt[1:]):
                    self.axh.axvline(0.5 * (a + b), color="#555", ls="--", lw=0.8)
            self.axh.set_xlabel(r"optical contrast  $(I - I_{\mathrm{SiO_2}})\,/\,I_{\mathrm{SiO_2}}$")
            self.axh.set_ylabel("pixels")
        self.canvas.draw()

    # -- segmentation --------------------------------------------------------
    def segment(self):
        mf, opt = self.main.mapfile, self.main.optical
        if opt is None:
            return
        self.contrast, _ = optical_contrast(mf, opt)
        cut = self.spn_cut.value()
        on = np.isfinite(self.contrast) & (np.abs(self.contrast) >= cut)
        k = self.spn_k.value()
        self.region_of[:] = -1
        if on.sum() < k:
            self.levels = None
            self._build_rows([])
            self.lbl_warn.setText("Too few pixels above the substrate cut to make "
                                  "%d regions — lower the cut or the region count." % k)
            self.lbl_warn.setStyleSheet("color:#b71c1c; font-size:11px;")
            self.refresh_map()
            return
        c = kmeans_1d(self.contrast[on], k)
        c = c[np.argsort(np.abs(c))]          # R1 = faintest = thinnest
        self.levels = c
        idx = np.where(on)[0]
        self.region_of[idx] = np.argmin(np.abs(self.contrast[idx, None] - c[None, :]), axis=1)

        # Defaults: a region with almost no 2D-band fits, or contrast opposite
        # to the flake's, is not a graphene terrace and starts unassigned.
        fit_ok = np.array([bool(r.get("ok")) for r in self.main.fits])
        n_pos = int(((self.contrast >= cut) & fit_ok).sum())
        n_neg = int(((self.contrast <= -cut) & fit_ok).sum())
        sign = 1.0 if n_pos >= n_neg else -1.0
        nxt = int(self.main.cfg.get("manual_layer", 3))
        entries = []
        for j, cj in enumerate(c):
            m = self.region_of == j
            frac = float(fit_ok[m].mean()) if m.any() else 0.0
            if np.sign(cj) == sign and frac >= 0.2:
                entries.append(("N", float(min(9, nxt))))
                nxt += 1
            else:
                entries.append(("N", 0.0))
        self._build_rows(entries)

    def _build_rows(self, entries):
        fw = np.array([r["fwhm"] if r.get("ok") else np.nan for r in self.main.fits])
        fit_ok = np.isfinite(fw)
        self.tbl.blockSignals(True)
        self.tbl.setRowCount(len(entries))
        self.rows = []
        for j, (kind, val) in enumerate(entries):
            m = self.region_of == j
            med = np.nanmedian(fw[m]) if (m & fit_ok).any() else np.nan
            cells = ("R%d" % (j + 1),
                     "%+.3f" % self.levels[j] if self.levels is not None else "—",
                     "%d" % m.sum(),
                     "—" if not np.isfinite(med) else "%.1f" % med)
            for col, txt in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(txt)
                it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                if col == 0:
                    it.setBackground(QtGui.QBrush(QtGui.QColor(REGION_COLORS[j % 6])))
                    it.setForeground(QtGui.QBrush(QtGui.QColor("white")))
                self.tbl.setItem(j, col, it)
            cmb = QtWidgets.QComboBox()
            cmb.addItems(["N", "SiO₂ height"])
            cmb.setToolTip("N — layer count, e.g. from a terrace step in Gwyddion\n"
                           "SiO₂ height — AFM step down to bare SiO₂, converted "
                           "with the offset below")
            cmb.setCurrentIndex(1 if kind == "height" else 0)
            spn = QtWidgets.QDoubleSpinBox()
            self.tbl.setCellWidget(j, 4, cmb)
            self.tbl.setCellWidget(j, 5, spn)
            nit = QtWidgets.QTableWidgetItem("—")
            nit.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            nit.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tbl.setItem(j, 6, nit)
            self.rows.append({"entry": cmb, "value": spn})
            self._configure_value(j, val)
            cmb.currentIndexChanged.connect(lambda _i, j=j: self._configure_value(j, None))
            spn.valueChanged.connect(self.update_status)
        self.tbl.blockSignals(False)
        self.update_status()

    def _configure_value(self, j: int, val):
        spn = self.rows[j]["value"]
        is_h = self.rows[j]["entry"].currentIndex() == 1
        spn.blockSignals(True)
        if is_h:
            spn.setDecimals(2)
            spn.setRange(0.0, 60.0)
            spn.setSingleStep(0.05)
            spn.setSuffix(" nm")
            spn.setToolTip("Step height from the flake down to bare SiO₂, from AFM")
        else:
            spn.setDecimals(0)
            spn.setRange(0, 9)
            spn.setSingleStep(1)
            spn.setSuffix("")
            spn.setToolTip("Layer count, e.g. from a terrace step in Gwyddion. 0 = unassigned")
        if val is not None:
            spn.setValue(val)
        elif is_h:
            spn.setValue(0.0)
        spn.blockSignals(False)
        self.update_status()

    # -- layer numbers + checks ---------------------------------------------
    def row_n(self, j: int):
        """(unrounded N or None, N used (0 = unassigned), note)."""
        is_h = self.rows[j]["entry"].currentIndex() == 1
        v = self.rows[j]["value"].value()
        if not is_h:
            n = int(round(v))
            return (float(n) if n > 0 else None), n, ""
        if v <= 0:
            return None, 0, ""
        raw = (v - self.spn_off.value()) / LAYER_NM
        n = int(round(raw))
        note = ""
        if n < 1:
            note = "height is below the offset"
            n = 0
        elif abs(raw - n) > 0.25:
            note = "%.2f is not close to a whole number of layers" % raw
        return raw, n, note

    def update_status(self, *_):
        if not self.rows:
            self.refresh_map()
            return
        warns, infos = [], []
        assigned = []
        for j in range(len(self.rows)):
            raw, n, note = self.row_n(j)
            it = self.tbl.item(j, 6)
            is_h = self.rows[j]["entry"].currentIndex() == 1
            if n <= 0:
                it.setText("—" if not note else "⚠ —")
                it.setForeground(QtGui.QBrush(QtGui.QColor("#999")))
            else:
                it.setText(("%.2f → %d" % (raw, n) if is_h else "%d" % n) +
                           (" ⚠" if note else ""))
                it.setForeground(QtGui.QBrush(QtGui.QColor("#b71c1c" if note else "#1b5e20")))
                assigned.append((abs(self.levels[j]) if self.levels is not None else 0, j, n))
                if n not in self.main.cfg["fwhm"] and n not in (1, 2):
                    warns.append("R%d: no reference widths for %dL in the table." % (j + 1, n))
            if note:
                warns.append("R%d: %s — check the SiO₂ offset or the line "
                             "profile." % (j + 1, note))
        assigned.sort()
        for (ca, ja, na), (cb, jb, nb) in zip(assigned[:-1], assigned[1:]):
            if nb < na:
                warns.append("R%d has stronger contrast than R%d but fewer layers "
                             "(%d vs %d) — check the values or which terrace is "
                             "which." % (jb + 1, ja + 1, nb, na))
            elif nb == na:
                infos.append("R%d and R%d are both %dL and will be treated as one "
                             "terrace." % (ja + 1, jb + 1, na))
        nb = self._n_by_spec()
        mf = self.main.mapfile
        g = np.zeros(mf.shape, dtype=int)
        ok = mf.index_map >= 0
        g[ok] = nb[mf.index_map[ok]]
        ne = int(terrace_edges(g).sum())
        infos.append("%d pixels assigned; %d step-edge pixels %s."
                     % (int((nb > 0).sum()), ne,
                        "excluded" if self.chk_edges.isChecked() else "kept"))
        html = "".join("<div style='color:#b71c1c'>⚠ %s</div>" % w for w in warns) + \
               "".join("<div style='color:#555'>%s</div>" % i for i in infos)
        self.lbl_warn.setText(html)
        self.btn_apply.setEnabled(bool(assigned))
        self.refresh_map()

    # -- selection ------------------------------------------------------------
    def _row_selected(self):
        rows = self.tbl.selectionModel().selectedRows()
        self.selected = rows[0].row() if rows else -1
        self.refresh_map()

    def _clicked(self, event):
        mf = self.main.mapfile
        if event.inaxes is not self.ax or event.xdata is None:
            return
        left, right, bottom, top = mf.extent
        ny, nx = mf.shape
        col = int(np.floor((event.xdata - left) / (right - left) * nx))
        row = int(np.floor((event.ydata - bottom) / (top - bottom) * ny))
        if 0 <= col < nx and 0 <= row < ny and mf.index_map[row, col] >= 0:
            j = int(self.region_of[mf.index_map[row, col]])
            if j >= 0:
                self.tbl.selectRow(j)

    # -- apply / persistence ----------------------------------------------------
    def apply(self):
        cfg = self.main.cfg
        nb = self._n_by_spec()
        if not (nb > 0).any():
            return
        self.main.layer_by_spec = nb
        cfg["exclude_edges"] = self.chk_edges.isChecked()
        cfg["contrast_levels"] = self.spn_k.value()
        cfg["contrast_cut"] = self.spn_cut.value()
        cfg["sio2_offset"] = self.spn_off.value()
        cfg["layer_mode"] = "map"
        tab = self.main.tab_class
        tab.rb_map.setChecked(True)
        tab.update_map_status()
        tab.run_classification()

    def save_regions(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save layer regions", self.main.suggest_name("_regions.json"),
            "JSON (*.json)")
        if not path:
            return
        mf = self.main.mapfile
        data = {"sio2_offset": self.spn_off.value(), "cut": self.spn_cut.value(),
                "levels": [float(v) for v in (self.levels if self.levels is not None else [])],
                "regions": [{"entry": "height" if r["entry"].currentIndex() else "N",
                             "value": r["value"].value()} for r in self.rows],
                "spectra": [[round(float(mf.raw0[i]), 4), round(float(mf.raw1[i]), 4),
                             int(self.region_of[i])] for i in range(mf.n_points)]}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        self.main.status("Regions saved to %s" % path)

    def load_regions(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load layer regions", "", "JSON (*.json)")
        if not path:
            return
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        mf = self.main.mapfile
        key = {(round(float(a), 3), round(float(b), 3)): i
               for i, (a, b) in enumerate(zip(mf.raw0, mf.raw1))}
        self.region_of[:] = -1
        hit = 0
        for a, b, r in data.get("spectra", []):
            i = key.get((round(a, 3), round(b, 3)))
            if i is not None:
                self.region_of[i] = int(r)
                hit += 1
        if hit == 0:
            QtWidgets.QMessageBox.warning(
                self, APP_NAME, "No positions in that file match this map — it "
                "was saved for a different acquisition.")
            return
        self.spn_off.blockSignals(True)
        self.spn_off.setValue(float(data.get("sio2_offset", 0.5)))
        self.spn_off.blockSignals(False)
        lv = np.asarray(data.get("levels", []), dtype=float)
        self.levels = lv if lv.size else None
        self._build_rows([(r["entry"], float(r["value"])) for r in data.get("regions", [])])
