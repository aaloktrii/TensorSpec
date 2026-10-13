"""2 · Map ▸ Area spectrum: the spectra of the area selected on the map (Map Explorer ▸ Select area)
summed into one spectrum and fitted with the model of 1 · Spectra ▸ Peak fitting; it can be sent to
Peak fitting or to the stacked figure. Written to /analysis/raman_roi.
Figures are drawn by plotting/backends/raman_views/peak_fit.py."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PySide6 import QtWidgets

from tensorspec.core.io.raman_loaders import Spectrum, short_label
from tensorspec.gui.components.raman_tabs.layout import fill_peak_table
from tensorspec.plotting.backends.raman_matplotlib_engine import draw_figure_message
from tensorspec.plotting.backends.raman_views import peak_fit as views
from tensorspec.plotting.viewers.raman_canvas import Canvas

if TYPE_CHECKING:
    from tensorspec.gui.components.raman_panel import RamanPanel


class AreaTab(QtWidgets.QWidget):

    def __init__(self, main: "RamanPanel"):
        super().__init__()
        self.main = main
        root = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.lbl = QtWidgets.QLabel("Tick 'Select area' in Map Explorer and drag a rectangle on the map.")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("color:#555;")
        left.addWidget(self.lbl)
        for text, fn in (("Fit with Peak-fitting model", self.fit),
                         ("Send to Peak fitting", self.send_to_fit),
                         ("Send to Stacked spectra", self.send_to_stack)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            left.addWidget(b)
        self.tbl_peaks = QtWidgets.QTableWidget(0, 6)
        self.tbl_peaks.setHorizontalHeaderLabels(["Band", "#", "Centre", "FWHM", "Height", "Area"])
        self.tbl_peaks.verticalHeader().setVisible(False)
        self.tbl_peaks.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tbl_peaks.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        left.addWidget(self.tbl_peaks, 1)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(460)
        root.addWidget(lw)
        self.plot = Canvas(figsize=(8, 5))
        root.addWidget(self.plot, 1)
        self.clear()

    def clear(self):
        self.tbl_peaks.setRowCount(0)
        self.plot.figure.clear()
        draw_figure_message(self.plot.figure, "No area selected")
        self.plot.draw()

    def show_area(self):
        """The summed spectrum of the selected area, not fitted."""
        svc = self.main.service
        if svc.area_spectrum is None:
            self.clear()
            return
        fig = self.plot.figure
        fig.clear()
        views.draw_spectra_overlay(fig, [{"label": self._label(), "wn": svc.area_axis,
                                          "y": svc.area_spectrum}], svc.fit_model.bands)
        self.tbl_peaks.setRowCount(0)
        x0, x1, y0, y1 = svc.area_rect
        self.lbl.setText("%d pixels, X %.1f–%.1f µm, Y %.1f–%.1f µm; spectra summed. Not fitted:\n"
                         "'Fit with Peak-fitting model' uses the bands of 1 · Spectra ▸ Peak fitting."
                         % (len(svc.area_indices), x0, x1, y0, y1))
        self.plot.draw()

    def _label(self) -> str:
        svc = self.main.service
        return "%s area (%d px)" % (short_label(self.main.mapfile.path), len(svc.area_indices))

    def fit(self):
        svc = self.main.service
        if svc.area_spectrum is None:
            self.clear()
            return
        if not svc.fit_model.bands:
            self.main.status("No bands to fit: define them in 1 · Spectra ▸ Peak fitting "
                             "(or apply a preset).")
            return
        res = svc.fit_area(svc.fit_model)
        bands = list(res["bands"].items())
        fig = self.plot.figure
        fig.clear()
        if bands:
            views.draw_band_fits(fig, bands)
        fill_peak_table(self.tbl_peaks, bands)
        x0, x1, y0, y1 = svc.area_rect
        self.lbl.setText("%d pixels, X %.1f–%.1f µm, Y %.1f–%.1f µm; spectra summed.\n"
                         "Fit: the model of 1 · Spectra ▸ Peak fitting (Si shift %+.2f cm⁻¹)."
                         % (len(svc.area_indices), x0, x1, y0, y1, res["shift"]))
        self.plot.draw()

    def send_to_fit(self):
        svc = self.main.service
        if svc.area_spectrum is None:
            return
        tf = self.main.tab_fit
        tf._add(self._label(), svc.area_axis, svc.area_spectrum, self.main.mapfile.path)
        tf._added()
        self.main.show_tab(tf)

    def send_to_stack(self):
        svc = self.main.service
        if svc.area_spectrum is None:
            return
        self.main.tab_stack.add_spectrum(Spectrum(svc.area_axis, svc.area_spectrum, self._label(),
                                                  self.main.mapfile.path))
        self.main.show_tab(self.main.tab_stack)
        self.main.status("Added the area spectrum to 1 · Spectra ▸ Stacked spectra (%s)."
                         % os.path.basename(self.main.mapfile.path.rstrip(os.sep)))
