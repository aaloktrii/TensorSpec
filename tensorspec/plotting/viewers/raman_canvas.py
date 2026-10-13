"""Reusable Qt figure widgets of the Raman suite: a canvas that draws in the Raman figure style,
figure export (vector PDF/SVG or 600-dpi PNG, saved in that style), the compact matplotlib
toolbar with an Export button, and Canvas (figure + toolbar in one widget).

Moved from gui/components/raman_tabs/layout.py (originally the "Small widgets" of Raman_Classifier
``raman_map_classifier_8 (1).py``)."""
from __future__ import annotations

import os
from typing import Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

from tensorspec.plotting.backends.raman_matplotlib_engine import raman_style


class FigureCanvas(FigureCanvasQTAgg):
    """Qt canvas that draws in the Raman figure style (plotting.backends.raman_matplotlib_engine.GUI_RC), which
    the original application set globally; here it applies to the Raman suite only."""

    def draw(self):
        with raman_style():
            super().draw()

# ---------------------------------------------------------------------------
# Small widgets
# ---------------------------------------------------------------------------


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
