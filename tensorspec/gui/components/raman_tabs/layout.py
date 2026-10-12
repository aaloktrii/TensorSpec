"""Shared widgets of the Raman tabs: screen-fitted sizing, figure export (vector PDF/SVG or
600-dpi PNG, saved in the Raman figure style), the compact matplotlib toolbar, the Canvas widget
(figure + toolbar, drawn in the Raman figure style) and the editable Bouhafs threshold table.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Small widgets, _fmt)."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from tensorspec.core.raman.classification import windows_for
from tensorspec.core.raman.constants import LAYERS
from tensorspec.plotting.raman_figures import raman_style


class FigureCanvas(FigureCanvasQTAgg):
    """Qt canvas that draws in the Raman figure style (plotting.raman_figures.GUI_RC), which
    the original application set globally; here it applies to the Raman suite only."""

    def draw(self):
        with raman_style():
            super().draw()

# ---------------------------------------------------------------------------
# Small widgets
# ---------------------------------------------------------------------------

def fit_to_screen(widget, want_w: int, want_h: int, frac_w=0.94, frac_h=0.90):
    """Open at the wanted size, but never larger than the screen it is on.

    A fixed size can push a window partly off a laptop screen: a 15-inch
    MacBook is about 1470 × 956 points at default scaling.
    """
    scr = widget.screen() if hasattr(widget, "screen") and widget.screen() else \
        QtWidgets.QApplication.primaryScreen()
    if scr is None:
        widget.resize(want_w, want_h)
        return
    g = scr.availableGeometry()
    widget.resize(min(want_w, int(g.width() * frac_w)), min(want_h, int(g.height() * frac_h)))


_EXPORT_DIR = {"path": ""}
EXPORT_FILTERS = ("Vector PDF (*.pdf);;Vector SVG (*.svg);;"
                  "PNG image, 600 dpi (*.png)")


def export_figure(fig, parent, suggested: str = "figure.pdf") -> Optional[str]:
    """Save a figure for publication.

    PDF and SVG are vector: axes, text, lines, grid, outlines and legends
    stay editable shapes and text. Pixel images inside a figure (the Raman map
    cells, the camera image) are embedded at 600 dpi, with no smoothing, so
    the map cells stay sharp-edged at any zoom.
    """
    start = os.path.join(_EXPORT_DIR["path"], os.path.basename(suggested)) \
        if _EXPORT_DIR["path"] else suggested
    path, flt = QtWidgets.QFileDialog.getSaveFileName(
        parent, "Export figure", start, EXPORT_FILTERS)
    if not path:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".pdf", ".svg", ".png"):
        ext = ".svg" if "SVG" in flt else ".png" if "PNG" in flt else ".pdf"
        path += ext
    _EXPORT_DIR["path"] = os.path.dirname(path)
    with raman_style():
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    return path


class CompactToolbar(NavigationToolbar):
    """Matplotlib's navigation toolbar, sized like a toolbar.

    On a Retina display Qt hands this toolbar pixmaps scaled by the device
    pixel ratio without shrinking the buttons to match, so the icon row ends
    up taller than the plot it controls. Pinning the icon size and the widget
    height keeps it to one slim strip on every display.
    """

    ICON_PX = 16

    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        self.setIconSize(QtCore.QSize(self.ICON_PX, self.ICON_PX))
        self.setFixedHeight(self.ICON_PX + 12)
        self.setStyleSheet(
            "QToolBar { border: 0px; padding: 0px; spacing: 0px; }"
            "QToolButton { padding: 1px; margin: 0px; }")
        # The coordinate readout is set in a large default face on macOS.
        for lbl in self.findChildren(QtWidgets.QLabel):
            lbl.setStyleSheet("font-size: 10px; color: #666;")
        self.addSeparator()
        act = self.addAction("Export…")
        act.setToolTip("Save this figure as vector PDF or SVG (editable in Inkscape "
                       "or Illustrator, then upload to Overleaf), or 600-dpi PNG")
        act.triggered.connect(self._export)

    def _export(self):
        win = self.window()
        main = win if hasattr(win, "suggest_name") else getattr(win, "main", None)
        if main is None:
            par = win.parent()
            main = par.window() if par is not None else None
        stem = "figure.pdf"
        if main is not None and hasattr(main, "suggest_name"):
            stem = main.suggest_name("_figure.pdf")
        path = export_figure(self.canvas.figure, self, stem)
        if path and main is not None and hasattr(main, "status"):
            main.status("Exported %s" % path)


class Canvas(QtWidgets.QWidget):
    """A matplotlib figure plus navigation toolbar in one widget."""

    def __init__(self, parent=None, toolbar=True, figsize=(5, 4)):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        if toolbar:
            self.toolbar = CompactToolbar(self.canvas, self)
            lay.addWidget(self.toolbar, 0)
        lay.addWidget(self.canvas, 1)
        self.ax = self.figure.add_subplot(111)

    def clear(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

    def draw(self):
        self.canvas.draw_idle()


class ThresholdTable(QtWidgets.QTableWidget):
    """Editable Bouhafs threshold + colour table. Writes straight into config."""

    changed = QtCore.Signal()

    COLS = ["Layer", "ABA FWHM", "ABA ±", "ABC FWHM", "ABC ±",
            "ABA window", "ABC window", "ABA colour", "ABC colour"]
    C_ABA, C_TA, C_ABC, C_TC, C_WA, C_WC, C_CA, C_CC = 1, 2, 3, 4, 5, 6, 7, 8

    def __init__(self, cfg: dict, parent=None):
        super().__init__(len(LAYERS), len(self.COLS), parent)
        self.cfg = cfg
        self._loading = False
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().setVisible(False)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        hh.setStretchLastSection(True)
        self.setMinimumHeight(len(LAYERS) * 28 + 34)
        self.reload()
        self.itemChanged.connect(self._on_item_changed)
        self.cellClicked.connect(self._on_cell_clicked)

    # -- population ---------------------------------------------------------
    def reload(self):
        self._loading = True
        for r, layer in enumerate(LAYERS):
            d = self.cfg["fwhm"][layer]
            name = QtWidgets.QTableWidgetItem("%dL" % layer)
            name.setFlags(QtCore.Qt.ItemIsEnabled)
            self.setItem(r, 0, name)
            tol = self.cfg["tol"][layer]
            for c, val in ((self.C_ABA, d["ABA"]), (self.C_TA, tol["ABA"]),
                           (self.C_ABC, d["ABC"]), (self.C_TC, tol["ABC"])):
                it = QtWidgets.QTableWidgetItem("%.1f" % val)
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                self.setItem(r, c, it)
            self._set_window_items(r, layer)
            for c, key in ((self.C_CA, "ABA"), (self.C_CC, "ABC")):
                self._set_color_item(r, c, self.cfg["colors"][layer][key])
        self._loading = False

    def _set_window_items(self, row: int, layer: int):
        """The two acceptance windows, and a warning when they collide."""
        (a0, a1), (c0, c1) = windows_for(layer, self.cfg)
        overlap = a1 >= c0
        gap = c0 - a1
        for col, (lo, hi) in ((self.C_WA, (a0, a1)), (self.C_WC, (c0, c1))):
            it = QtWidgets.QTableWidgetItem("%.1f – %.1f" % (lo, hi))
            it.setFlags(QtCore.Qt.ItemIsEnabled)
            it.setTextAlignment(QtCore.Qt.AlignCenter)
            if overlap:
                it.setForeground(QtGui.QBrush(QtGui.QColor("#c62828")))
                it.setToolTip("The windows overlap by %.1f cm⁻¹; pixels in the "
                              "overlap are settled by the overlap policy."
                              % -gap)
            elif gap > 0.5:
                it.setForeground(QtGui.QBrush(QtGui.QColor("#8a6d00")))
                it.setToolTip("A %.1f cm⁻¹ dead zone sits between the windows; "
                              "widths landing there stay Unclassified." % gap)
            self.setItem(row, col, it)

    def _set_color_item(self, row: int, col: int, hexcol: str):
        it = QtWidgets.QTableWidgetItem(hexcol)
        it.setFlags(QtCore.Qt.ItemIsEnabled)
        it.setTextAlignment(QtCore.Qt.AlignCenter)
        qc = QtGui.QColor(hexcol)
        it.setBackground(QtGui.QBrush(qc))
        lum = 0.299 * qc.red() + 0.587 * qc.green() + 0.114 * qc.blue()
        it.setForeground(QtGui.QBrush(QtGui.QColor("#000000" if lum > 140 else "#ffffff")))
        self.setItem(row, col, it)

    def set_all_tolerances(self, value: float):
        for layer in LAYERS:
            self.cfg["tol"][layer]["ABA"] = float(value)
            self.cfg["tol"][layer]["ABC"] = float(value)
        self.reload()
        self.changed.emit()

    def set_tolerances_to_half_gap(self):
        """Windows meet exactly at the midpoint: the old boundary rule, but
        bounded on the outside so far-off widths no longer get a label."""
        for layer in LAYERS:
            d = self.cfg["fwhm"][layer]
            half = abs(d["ABC"] - d["ABA"]) / 2.0
            self.cfg["tol"][layer]["ABA"] = half
            self.cfg["tol"][layer]["ABC"] = half
        self.reload()
        self.changed.emit()

    # -- interaction --------------------------------------------------------
    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem):
        col = item.column()
        if self._loading or col not in (self.C_ABA, self.C_TA,
                                        self.C_ABC, self.C_TC):
            return
        layer = LAYERS[item.row()]
        is_tol = col in (self.C_TA, self.C_TC)
        key = "ABA" if col in (self.C_ABA, self.C_TA) else "ABC"
        store = self.cfg["tol"] if is_tol else self.cfg["fwhm"]
        try:
            value = float(item.text().replace(",", "."))
            # A zero tolerance is legitimate (accept only the exact value);
            # a negative one is not.
            if value < 0 or (value <= 0 and not is_tol):
                raise ValueError
        except ValueError:
            self._loading = True
            item.setText("%.1f" % store[layer][key])
            self._loading = False
            return
        store[layer][key] = value
        self._loading = True
        item.setText("%.1f" % value)
        self._set_window_items(item.row(), layer)
        self._loading = False
        self.changed.emit()

    def _on_cell_clicked(self, row: int, col: int):
        if col not in (self.C_CA, self.C_CC):
            return
        layer = LAYERS[row]
        key = "ABA" if col == self.C_CA else "ABC"
        current = QtGui.QColor(self.cfg["colors"][layer][key])
        chosen = QtWidgets.QColorDialog.getColor(
            current, self, "Colour for %dL-%s" % (layer, key))
        if chosen.isValid():
            self.cfg["colors"][layer][key] = chosen.name()
            self._loading = True
            self._set_color_item(row, col, chosen.name())
            self._loading = False
            self.changed.emit()


def _fmt(v, nd=3) -> str:
    return "" if v is None or not np.isfinite(v) else ("%.*f" % (nd, v))
