"""Tab 4 · Stacked spectra: publication figure (closed frame, // axis breaks, no y ticks, Arial,
8.7 cm, vector PDF) from files or from the loaded map. Nothing here fits anything.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Tab 4 - Stacked spectra); draw_stack is in plotting/backends/raman_matplotlib_engine.py."""
from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING
from typing import List, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from tensorspec.core.io.raman_loaders import (Spectrum, is_single_spectrum, parse_map_file,
                                              read_single_spectrum, read_spot_folder, short_label)
from tensorspec.core.raman.constants import APP_NAME
from tensorspec.plotting.backends.raman_matplotlib_engine import raman_style, stack_figure
from tensorspec.plotting.viewers.raman_canvas import _EXPORT_DIR

if TYPE_CHECKING:
    from tensorspec.gui.components.raman_panel import RamanPanel

STACK_COLORS = ["#000000", "#e8402a", "#1f5fa8", "#2a8c4a", "#8a5a00",
                "#7b3fa0", "#17becf", "#7f7f7f"]
_NUM = r"(\d+(?:\.\d+)?)"


def parse_windows(text: str) -> List[Tuple[float, float]]:
    """'1450-1650, 2550-2800' -> [(1450, 1650), (2550, 2800)]."""
    out = []
    for lo, hi in re.findall(_NUM + r"\s*[-–:]\s*" + _NUM, text):
        lo, hi = float(lo), float(hi)
        if hi > lo:
            out.append((lo, hi))
    return out


def parse_band_labels(text: str) -> List[Tuple[str, float]]:
    """'G 1583, 2D 2690' -> [('G', 1583.0), ('2D', 2690.0)]."""
    out = []
    for piece in re.split(r"[,;]", text):
        m = re.match(r"\s*(.+?)\s*@?\s*" + _NUM + r"\s*$", piece)
        if m:
            out.append((m.group(1), float(m.group(2))))
    return out


class StackTab(QtWidgets.QWidget):
    """Stack spectra from files or from the loaded map into one figure and
    export it. Nothing here fits anything."""

    def __init__(self, main: "RamanPanel"):
        super().__init__()
        self.main = main
        self.items: List[dict] = []          # {spec, color, visible}
        self._build()

    # -- UI -----------------------------------------------------------------
    def _build(self):
        root = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()

        self.tbl = QtWidgets.QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["Label", "", "Source"])
        self.tbl.verticalHeader().setVisible(False)
        hh = self.tbl.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.tbl.setColumnWidth(1, 26)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl.itemChanged.connect(self._item_changed)
        self.tbl.cellClicked.connect(self._cell_clicked)
        left.addWidget(self.tbl, 1)

        r = QtWidgets.QHBoxLayout()
        for text, fn in (("Add files…", self.add_files),
                         ("Add selected pixel", self.add_map_pixel),
                         ("Add map median", self.add_map_median)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            r.addWidget(b)
        left.addLayout(r)
        r = QtWidgets.QHBoxLayout()
        for text, fn in (("Up", lambda: self._move(-1)), ("Down", lambda: self._move(1)),
                         ("Remove", self._remove), ("Clear", self.clear)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            r.addWidget(b)
        left.addLayout(r)

        form = QtWidgets.QFormLayout()
        self.ed_win = QtWidgets.QLineEdit("")
        self.ed_win.setPlaceholderText("e.g. 1450-1650, 2550-2800 — empty: full range")
        form.addRow("Ranges (cm⁻¹)", self.ed_win)
        self.spn_step = QtWidgets.QDoubleSpinBox()
        self.spn_step.setRange(0.0, 5.0)
        self.spn_step.setSingleStep(0.05)
        self.spn_step.setValue(1.15)
        form.addRow("Offset", self.spn_step)
        self.chk_norm = QtWidgets.QCheckBox("Normalise each spectrum")
        self.chk_norm.setChecked(True)
        form.addRow(self.chk_norm)
        self.chk_labels = QtWidgets.QCheckBox("Spectrum labels")
        form.addRow(self.chk_labels)
        self.chk_bands = QtWidgets.QCheckBox("Peak labels")
        self.ed_bands = QtWidgets.QLineEdit("")
        self.ed_bands.setPlaceholderText("e.g. G 1583, 2D 2690")
        rb = QtWidgets.QHBoxLayout()
        rb.addWidget(self.chk_bands)
        rb.addWidget(self.ed_bands, 1)
        form.addRow(rb)
        self.ed_title = QtWidgets.QLineEdit("")
        form.addRow("Title", self.ed_title)
        self.spn_w = QtWidgets.QDoubleSpinBox()
        self.spn_w.setRange(3.0, 30.0)
        self.spn_w.setValue(8.7)
        self.spn_w.setSuffix(" cm")
        self.spn_h = QtWidgets.QDoubleSpinBox()
        self.spn_h.setRange(0.0, 30.0)
        self.spn_h.setSpecialValueText("auto")
        self.spn_h.setSuffix(" cm")
        rs = QtWidgets.QHBoxLayout()
        rs.addWidget(self.spn_w)
        rs.addWidget(QtWidgets.QLabel("×"))
        rs.addWidget(self.spn_h)
        form.addRow("Size", rs)
        left.addLayout(form)
        b = QtWidgets.QPushButton("Export PDF…")
        b.clicked.connect(self.export_pdf)
        left.addWidget(b)

        for w in (self.ed_win, self.ed_bands, self.ed_title):
            w.editingFinished.connect(self.redraw)
        for w in (self.spn_step, self.spn_w, self.spn_h):
            w.valueChanged.connect(self.redraw)
        for w in (self.chk_norm, self.chk_labels, self.chk_bands):
            w.toggled.connect(self.redraw)

        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(430)
        root.addWidget(lw)
        self.preview = QtWidgets.QLabel()
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setStyleSheet("background:white;")
        sc = QtWidgets.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.preview)
        root.addWidget(sc, 1)
        self.redraw()

    # -- items ----------------------------------------------------------------
    def add_spectrum(self, spec: Spectrum):
        used = {it["color"] for it in self.items}
        color = next((c for c in STACK_COLORS if c not in used), STACK_COLORS[0])
        self.items.append({"spec": spec, "color": color, "visible": True})
        self._rebuild()
        self.redraw()

    def add_paths(self, paths: List[str]):
        bad = []
        for p in paths:
            try:
                if os.path.isdir(p):
                    mf = read_spot_folder(p)
                    self.add_spectrum(Spectrum(
                        mf.axis, np.nanmedian(mf.data, 0),
                        short_label(p) + " median", p))
                elif is_single_spectrum(p):
                    self.add_spectrum(read_single_spectrum(p))
                else:
                    mf = parse_map_file(p)
                    if mf.layout != "spectra":
                        raise ValueError("not a spectrum")
                    self.add_spectrum(Spectrum(
                        mf.axis, np.nanmedian(mf.data, 0),
                        short_label(p) + " median", p))
            except Exception as exc:
                bad.append("%s: %s" % (os.path.basename(p), exc))
        if bad:
            QtWidgets.QMessageBox.warning(self, APP_NAME, "Could not read:\n\n"
                                          + "\n".join(bad))

    def add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add spectra", "",
            "Spectra (*.txt *.dat *.csv *.h5);;All files (*)")
        if paths:
            self.add_paths(paths)
            self.main.show_tab(self)

    def add_map_pixel(self):
        mf, sel = self.main.mapfile, self.main.tab_map.sel
        if mf is None or sel is None or mf.index_map[sel] < 0:
            self.main.status("Click a pixel in the Map Explorer first.")
            return
        i = int(mf.index_map[sel])
        name = short_label(mf.path)
        self.add_spectrum(Spectrum(mf.axis, mf.data[i].astype(float),
                                   "%s (%.1f, %.1f)" % (name, mf.x[i], mf.y[i]),
                                   mf.path))

    def add_map_median(self):
        mf = self.main.mapfile
        if mf is None:
            self.main.status("Load a map first.")
            return
        name = short_label(mf.path)
        self.add_spectrum(Spectrum(mf.axis, np.nanmedian(mf.data, 0),
                                   "%s median" % name, mf.path))

    def clear(self):
        self.items = []
        self._rebuild()
        self.redraw()

    def _remove(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            self.items.pop(r)
        self._rebuild()
        self.redraw()

    def _move(self, d: int):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()})
        if len(rows) != 1:
            return
        r, t = rows[0], rows[0] + d
        if 0 <= t < len(self.items):
            self.items[r], self.items[t] = self.items[t], self.items[r]
            self._rebuild()
            self.tbl.selectRow(t)
            self.redraw()

    def _rebuild(self):
        self.tbl.blockSignals(True)
        self.tbl.setRowCount(len(self.items))
        for r, it in enumerate(self.items):
            a = QtWidgets.QTableWidgetItem(it["spec"].label)
            a.setFlags(a.flags() | QtCore.Qt.ItemIsUserCheckable)
            a.setCheckState(QtCore.Qt.Checked if it["visible"] else QtCore.Qt.Unchecked)
            self.tbl.setItem(r, 0, a)
            c = QtWidgets.QTableWidgetItem("")
            c.setFlags(QtCore.Qt.ItemIsEnabled)
            c.setBackground(QtGui.QColor(it["color"]))
            self.tbl.setItem(r, 1, c)
            s = QtWidgets.QTableWidgetItem(os.path.basename(it["spec"].path.rstrip(os.sep)))
            s.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            s.setToolTip(it["spec"].path)
            self.tbl.setItem(r, 2, s)
        self.tbl.blockSignals(False)

    def _item_changed(self, item):
        r = item.row()
        if item.column() == 0 and 0 <= r < len(self.items):
            self.items[r]["spec"].label = item.text()
            self.items[r]["visible"] = item.checkState() == QtCore.Qt.Checked
            self.redraw()

    def _cell_clicked(self, r: int, c: int):
        if c != 1 or not (0 <= r < len(self.items)):
            return
        col = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.items[r]["color"]), self)
        if col.isValid():
            self.items[r]["color"] = col.name()
            self._rebuild()
            self.redraw()

    # -- figure ---------------------------------------------------------------
    def _figure(self):
        traces = [{"wn": it["spec"].wn, "inten": it["spec"].inten,
                   "label": it["spec"].label, "color": it["color"]}
                  for it in self.items if it["visible"]]
        w = self.spn_w.value() / 2.54
        h = self.spn_h.value() / 2.54 or (2.6 + 1.25 * max(1, len(traces))) / 2.54
        windows = parse_windows(self.ed_win.text())
        if not windows and traces:              # no ranges given: the whole common range
            windows = [(max(float(t["wn"].min()) for t in traces),
                        min(float(t["wn"].max()) for t in traces))]
            windows = [w for w in windows if w[1] > w[0]]
        return stack_figure(w, h, traces, windows,
                            step=self.spn_step.value(), normalise=self.chk_norm.isChecked(),
                            trace_labels=self.chk_labels.isChecked(),
                            band_labels=parse_band_labels(self.ed_bands.text())
                            if self.chk_bands.isChecked() else [],
                            title=self.ed_title.text().strip())

    def redraw(self):
        import io
        fig = self._figure()
        buf = io.BytesIO()
        ratio = self.devicePixelRatioF() if hasattr(self, "devicePixelRatioF") else 1.0
        with raman_style():
            fig.savefig(buf, format="png", dpi=150 * ratio, bbox_inches="tight",
                        pad_inches=0.03, facecolor="white")
        pm = QtGui.QPixmap()
        pm.loadFromData(buf.getvalue())
        pm.setDevicePixelRatio(ratio)
        self.preview.setPixmap(pm)

    def export_pdf(self):
        stem = "stacked_spectra.pdf"
        vis = [it for it in self.items if it["visible"]]
        if vis and vis[0]["spec"].path:
            stem = os.path.join(os.path.dirname(vis[0]["spec"].path.rstrip(os.sep)), stem)
        start = os.path.join(_EXPORT_DIR["path"], os.path.basename(stem)) \
            if _EXPORT_DIR["path"] else stem
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PDF", start, "Vector PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        _EXPORT_DIR["path"] = os.path.dirname(path)
        with raman_style():
            self._figure().savefig(path, bbox_inches="tight", pad_inches=0.03,
                                   facecolor="white")
        self.main.status("Exported %s" % path)
