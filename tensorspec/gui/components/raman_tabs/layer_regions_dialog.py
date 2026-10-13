"""Classification helper dialog: layer regions (terraces) from optical contrast, each given its
layer count; save / load regions. Figures are drawn by plotting/backends/raman_views/classification.py.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Tab 2 - Classification: LayerRegionsDialog)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from tensorspec.core.raman.constants import APP_NAME, LAYER_NM
from tensorspec.gui.components.raman_tabs.layout import fit_to_screen
from tensorspec.plotting.backends.raman_views import classification as views
from tensorspec.plotting.backends.raman_views.classification import REGION_COLORS
from tensorspec.plotting.viewers.raman_canvas import Canvas

if TYPE_CHECKING:
    from tensorspec.gui.components.raman_panel import RamanPanel



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
                                   "contrast, so load one in 2 · Map ▸ Map Explorer first.")
            self.lbl_place.setStyleSheet("color:#b71c1c; font-size:11px;")
            for w in (self.spn_k, self.spn_cut, self.btn_seg, self.btn_apply):
                w.setEnabled(False)
            return
        if main.cfg.get("map_transform") is None:
            self.lbl_place.setText("⚠ The map's offset has not been aligned yet — run "
                                   "Auto-align in 2 · Map ▸ Map Explorer first, or contrast will be "
                                   "read from slightly the wrong place.")
            self.lbl_place.setStyleSheet("color:#b71c1c; font-size:11px;")
        else:
            self.lbl_place.setText("Placement: %s" %
                                   self.main.service.describe_transform(main.cfg["map_transform"]))
            self.lbl_place.setStyleSheet("color:#555; font-size:11px;")
        self.segment()

    # -- figure -------------------------------------------------------------
    def _build_figure(self):
        fig = self.canvas.figure
        fig.clear()
        self.ax, self.axh, self.overlay = views.build_regions_figure(
            fig, self.main.mapfile, self.main.cfg, self.main.optical, self.main.optical_image)

    def _n_per_region(self):
        return [self.row_n(j)[1] for j in range(len(self.rows))]

    def refresh_map(self):
        mf = self.main.mapfile
        edges = None
        if self.chk_edges.isChecked():
            edges = self.main.service.region_edges(self.region_of, self._n_per_region())
        self.overlay.set_data(views.region_rgba(mf, self.region_of, len(self.rows),
                                                self.selected, edges))
        self._texts = views.draw_region_labels(self.ax, mf, self.region_of, len(self.rows),
                                               self._texts)
        views.draw_contrast_histogram(self.axh, self.contrast, self.spn_cut.value(), self.levels)
        self.canvas.draw()

    # -- segmentation --------------------------------------------------------
    def segment(self):
        if self.main.optical is None:
            return
        k = self.spn_k.value()
        seg = self.main.service.segment_regions(self.spn_cut.value(), k)
        self.contrast, self.levels = seg["contrast"], seg["levels"]
        self.region_of[:] = seg["region_of"]
        if self.levels is None:
            self._build_rows([])
            self.lbl_warn.setText("Too few pixels above the substrate cut to make "
                                  "%d regions — lower the cut or the region count." % k)
            self.lbl_warn.setStyleSheet("color:#b71c1c; font-size:11px;")
            self.refresh_map()
            return
        self._build_rows(seg["entries"])

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
        return self.main.service.region_n(self.rows[j]["entry"].currentIndex() == 1,
                                          self.rows[j]["value"].value(), self.spn_off.value())

    def update_status(self, *_):
        if not self.rows:
            self.refresh_map()
            return
        rows = [self.row_n(j) for j in range(len(self.rows))]
        for j, (raw, n, note) in enumerate(rows):
            it = self.tbl.item(j, 6)
            is_h = self.rows[j]["entry"].currentIndex() == 1
            if n <= 0:
                it.setText("—" if not note else "⚠ —")
                it.setForeground(QtGui.QBrush(QtGui.QColor("#999")))
            else:
                it.setText(("%.2f → %d" % (raw, n) if is_h else "%d" % n) +
                           (" ⚠" if note else ""))
                it.setForeground(QtGui.QBrush(QtGui.QColor("#b71c1c" if note else "#1b5e20")))
        warns, infos, assigned = self.main.service.check_regions(
            self.levels, self.region_of, rows, self.chk_edges.isChecked())
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
        if not self.main.service.apply_regions(self.region_of, self._n_per_region(),
                                               self.chk_edges.isChecked(), self.spn_k.value(),
                                               self.spn_cut.value(), self.spn_off.value()):
            return
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
        self.main.service.save_regions(
            path, self.spn_off.value(), self.spn_cut.value(), self.levels,
            [("height" if r["entry"].currentIndex() else "N", r["value"].value()) for r in self.rows],
            self.region_of)
        self.main.status("Regions saved to %s" % path)

    def load_regions(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load layer regions", "", "JSON (*.json)")
        if not path:
            return
        data = self.main.service.load_regions(path)
        if data is None:
            QtWidgets.QMessageBox.warning(
                self, APP_NAME, "No positions in that file match this map — it "
                "was saved for a different acquisition.")
            return
        self.region_of[:] = data["region_of"]
        self.spn_off.blockSignals(True)
        self.spn_off.setValue(data["sio2_offset"])
        self.spn_off.blockSignals(False)
        self.levels = data["levels"]
        self._build_rows(data["entries"])
