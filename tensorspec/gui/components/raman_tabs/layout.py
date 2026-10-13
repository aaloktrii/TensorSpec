"""Shared widgets of the Raman tabs: screen-fitted dialog sizing, the editable Bouhafs threshold
table and the number formatter used in the tabs' tables. The figure widgets (Canvas, toolbar,
export) are in plotting/viewers/raman_canvas.py.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Small widgets, _fmt)."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from tensorspec.core.raman.classification import windows_for
from tensorspec.core.raman.constants import LAYERS


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
