"""Tab 6 · Peak fitting: optional Si calibration, background model, Lorentzians with a fixed or
data-chosen peak count, batch over files / folders / the loaded map, and the omega_G - omega_2D
strain/doping correlation (shown in 3 · Analysis ▸ Strain / doping). Results for the loaded map are
written to /analysis/raman_peakfit.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Tab 6 - Peak fitting); calls go through RamanEngineRouter (main.engine),
figures through plotting/backends/raman_views/peak_fit.py."""
from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING
from typing import Dict, List

import numpy as np
from PySide6 import QtCore, QtWidgets

from tensorspec.core.raman.fit_model import FitModel, as_band, band
from tensorspec.core.raman.ranges import parse_windows
from tensorspec.gui.components.raman_tabs.layout import fill_peak_table
from tensorspec.plotting.backends.raman_matplotlib_engine import draw_figure_message
from tensorspec.plotting.backends.raman_views import peak_fit as views
from tensorspec.plotting.viewers.raman_canvas import Canvas

#: The strain/doping and peak-fit modules were optional next to the original script; they are
#: part of tensorspec (core/raman/) now.
HAVE_GREF = True

if TYPE_CHECKING:
    from tensorspec.gui.components.raman_panel import RamanPanel



class PeakFitTab(QtWidgets.QWidget):
    """Fit any spectra band by band: optional Si calibration, a background
    model, and Lorentzians with the number of peaks set by hand or chosen
    from the data. Batch over files, folders or the loaded map."""

    def __init__(self, main: "RamanPanel"):
        super().__init__()
        self.main = main
        self._build()

    # -- UI -------------------------------------------------------------------
    def _build(self):
        root = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()

        self.lst = QtWidgets.QListWidget()
        self.lst.currentRowChanged.connect(self._selected)
        left.addWidget(self.lst, 2)
        r = QtWidgets.QHBoxLayout()
        for text, fn in (("Add files…", self.add_files), ("Add folder…", self.add_folder),
                         ("Use loaded map", self.use_map), ("Plot all", self.plot_all),
                         ("Clear", self.clear)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            r.addWidget(b)
        left.addLayout(r)

        form = QtWidgets.QFormLayout()
        self.chk_si = QtWidgets.QCheckBox("Calibrate on Si (520.7 cm⁻¹)")
        self.chk_si.setChecked(False)
        form.addRow(self.chk_si)
        self.cmb_bg = QtWidgets.QComboBox()
        self.cmb_bg.addItems(["linear", "polynomial", "als", "none"])
        form.addRow("Background", self.cmb_bg)
        rb = QtWidgets.QHBoxLayout()
        self.spn_order = QtWidgets.QSpinBox()
        self.spn_order.setRange(1, 8)
        self.spn_order.setValue(3)
        self.ed_lam = QtWidgets.QLineEdit("1e5")
        self.ed_p = QtWidgets.QLineEdit("0.01")
        for lab, w in (("order", self.spn_order), ("λ", self.ed_lam), ("p", self.ed_p)):
            rb.addWidget(QtWidgets.QLabel(lab))
            rb.addWidget(w)
        form.addRow(rb)
        left.addLayout(form)

        # fit ranges typed as text: one band per range
        r = QtWidgets.QHBoxLayout()
        r.addWidget(QtWidgets.QLabel("Ranges"))
        self.ed_ranges = QtWidgets.QLineEdit("")
        self.ed_ranges.setPlaceholderText("e.g. 1400-1600; 2500-2800  (cm⁻¹)")
        self.ed_ranges.returnPressed.connect(self.set_ranges)
        r.addWidget(self.ed_ranges, 1)
        b = QtWidgets.QPushButton("Set bands")
        b.setToolTip("One band per range; existing bands are replaced")
        b.clicked.connect(self.set_ranges)
        r.addWidget(b)
        left.addLayout(r)

        self.tbl_bands = QtWidgets.QTableWidget(0, 7)
        self.tbl_bands.setHorizontalHeaderLabels(["Band", "From", "To", "Peaks", "Max", "Shape",
                                                  "Manual centres"])
        self.tbl_bands.verticalHeader().setVisible(False)
        hh = self.tbl_bands.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch)
        self.tbl_bands.setMaximumHeight(160)
        self.tbl_bands.setToolTip(
            "Peaks: 'Auto: find count' lets the fit decide how many (up to Max); 'Auto: N' fits N "
            "peaks it places itself; 'Manual' fits one peak at each manual centre — select the "
            "band's row, then left-click the spectrum to add a peak, right-click to remove one.")
        self.tbl_bands.itemChanged.connect(self._bands_changed)
        left.addWidget(self.tbl_bands)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(QtWidgets.QLabel("Manual peaks may move ±"))
        self.spn_tol = QtWidgets.QDoubleSpinBox()
        self.spn_tol.setRange(0.5, 200.0)
        self.spn_tol.setDecimals(1)
        self.spn_tol.setValue(10.0)
        self.spn_tol.setSuffix(" cm⁻¹")
        r.addWidget(self.spn_tol)
        r.addStretch(1)
        left.addLayout(r)
        r = QtWidgets.QHBoxLayout()
        for text, fn in (("Add band", lambda: self._add_band_row("band", 1000, 1100, "auto", 3)),
                         ("Remove band", self._remove_band),
                         ("Graphene preset", lambda: self.main.apply_preset("Graphene"))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            r.addWidget(b)
        left.addLayout(r)

        r = QtWidgets.QHBoxLayout()
        for text, fn in (("Fit selected", self.fit_selected), ("Fit all…", self.fit_all),
                         ("ω_G–ω_2D plot…", self.correlation)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            b.setEnabled(HAVE_GREF)
            r.addWidget(b)
        left.addLayout(r)

        self.tbl_peaks = QtWidgets.QTableWidget(0, 6)
        self.tbl_peaks.setHorizontalHeaderLabels(["Band", "#", "Centre", "FWHM", "Height", "Area"])
        self.tbl_peaks.verticalHeader().setVisible(False)
        self.tbl_peaks.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tbl_peaks.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        left.addWidget(self.tbl_peaks, 2)
        self.lbl = QtWidgets.QLabel("")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("color:#555; font-size:11px;")
        left.addWidget(self.lbl)

        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(500)
        root.addWidget(lw)
        self.plot = Canvas(figsize=(8, 5))
        self.plot.canvas.mpl_connect("button_press_event", self._plot_clicked)
        root.addWidget(self.plot, 1)
        # every fit setting goes to the backend as it changes
        self.chk_si.toggled.connect(self._sync_model)
        self.cmb_bg.currentIndexChanged.connect(self._sync_model)
        self.spn_order.valueChanged.connect(self._sync_model)
        self.ed_lam.editingFinished.connect(self._sync_model)
        self.ed_p.editingFinished.connect(self._sync_model)
        self.spn_tol.valueChanged.connect(self._sync_model)
        self._placeholder()

    PEAK_CHOICES = ([("Auto: find count", "auto")] +
                    [("Auto: %d peak%s" % (k, "" if k == 1 else "s"), k) for k in range(1, 7)] +
                    [("Manual (click peaks)", "manual")])
    SHAPE_CHOICES = (("Lorentzian", "lorentzian"), ("Gaussian", "gaussian"),
                     ("Voigt (pseudo)", "voigt"))

    def _add_band_row(self, name, lo, hi, peaks="auto", nmax=3, shape="lorentzian", centres=()):
        self.tbl_bands.blockSignals(True)            # no redraws while the row is half built
        r = self.tbl_bands.rowCount()
        self.tbl_bands.insertRow(r)
        self.tbl_bands.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
        self.tbl_bands.setItem(r, 1, QtWidgets.QTableWidgetItem("%g" % lo))
        self.tbl_bands.setItem(r, 2, QtWidgets.QTableWidgetItem("%g" % hi))
        cmb = QtWidgets.QComboBox()
        for label, data in self.PEAK_CHOICES:
            cmb.addItem(label, data)
        peaks = "auto" if peaks in ("Auto", "auto") else peaks
        cmb.setCurrentIndex(max(0, cmb.findData(peaks if peaks in ("auto", "manual") else int(peaks))))
        self.tbl_bands.setCellWidget(r, 3, cmb)
        sp = QtWidgets.QSpinBox()
        sp.setRange(1, 6)
        sp.setValue(int(nmax))
        sp.setToolTip("Most peaks 'Auto: find count' may use")
        self.tbl_bands.setCellWidget(r, 4, sp)
        shp = QtWidgets.QComboBox()
        for label, data in self.SHAPE_CHOICES:
            shp.addItem(label, data)
        shp.setCurrentIndex(max(0, shp.findData(shape)))
        self.tbl_bands.setCellWidget(r, 5, shp)
        self.tbl_bands.setItem(r, 6, QtWidgets.QTableWidgetItem(
            ", ".join("%g" % c for c in centres)))
        cmb.currentIndexChanged.connect(lambda *_, c=cmb, s=sp: s.setEnabled(c.currentData() == "auto"))
        sp.setEnabled(cmb.currentData() == "auto")
        cmb.currentIndexChanged.connect(self._bands_changed)
        shp.currentIndexChanged.connect(self._bands_changed)
        self.tbl_bands.blockSignals(False)

    def set_bands(self, bands, si_calibrate: bool):
        """Replace the band table: rows of (name, from, to, peaks, max[, shape, centres])."""
        self.tbl_bands.setRowCount(0)
        for row in bands:
            self._add_band_row(*as_band(row))
        self.chk_si.setChecked(bool(si_calibrate))
        self._bands_changed()

    def set_ranges(self):
        """One band per range typed in 'Ranges' (e.g. 1400-1600; 2500-2800); peaks found
        automatically until set otherwise."""
        ranges = parse_windows(self.ed_ranges.text().replace(";", ","))
        if not ranges:
            self.main.status("No range understood — type e.g. 1400-1600; 2500-2800")
            return
        self.tbl_bands.setRowCount(0)
        for lo, hi in ranges:
            self._add_band_row("%g-%g" % (lo, hi), lo, hi)
        self._bands_changed()
        self.main.status("%d band%s set from the ranges; nothing fitted yet."
                         % (len(ranges), "" if len(ranges) == 1 else "s"))

    def _bands_changed(self, *_):
        """Hand the bands to the backend and re-show the current view, so band windows and manual
        centres stay visible."""
        self._sync_model()
        i = self.lst.currentRow()
        if 0 <= i < len(self.items) and i not in self.results:
            self._selected(i)
        elif self.items and i < 0:
            self.plot_all()

    def _remove_band(self):
        r = self.tbl_bands.currentRow()
        if r >= 0:
            self.tbl_bands.removeRow(r)

    def bands(self) -> List[tuple]:
        out = []
        for r in range(self.tbl_bands.rowCount()):
            try:
                lo = float(self.tbl_bands.item(r, 1).text())
                hi = float(self.tbl_bands.item(r, 2).text())
            except (AttributeError, ValueError):
                continue
            if self.tbl_bands.cellWidget(r, 3) is None or self.tbl_bands.cellWidget(r, 5) is None:
                continue
            item = self.tbl_bands.item(r, 6)
            centres = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?", item.text() if item else "")]
            out.append(band(self.tbl_bands.item(r, 0).text(), lo, hi,
                            self.tbl_bands.cellWidget(r, 3).currentData(),
                            self.tbl_bands.cellWidget(r, 4).value(),
                            self.tbl_bands.cellWidget(r, 5).currentData(), sorted(centres)))
        return out

    # -- manual peaks: click on the plot ------------------------------------------
    def _plot_clicked(self, event):
        """Left-click adds a manual peak to the band under the cursor (the selected row first),
        right-click removes the nearest one."""
        if event.inaxes is None or event.xdata is None:
            return
        x = float(event.xdata)
        rows = [r for r, b in enumerate(self.bands()) if b[3] == "manual" and b[1] < x < b[2]]
        if not rows:
            if any(b[3] == "manual" for b in self.bands()):
                self.main.status("Click inside a Manual band's range to place a peak.")
            return
        cur = self.tbl_bands.currentRow()
        r = cur if cur in rows else rows[0]
        centres = list(self.bands()[r][6])
        if event.button == 1:
            centres.append(x)
        elif event.button == 3 and centres:
            centres.remove(min(centres, key=lambda c: abs(c - x)))
        else:
            return
        self.tbl_bands.item(r, 6).setText(", ".join("%.1f" % c for c in sorted(centres)))
        i = self.lst.currentRow()
        self.results.pop(i, None)                      # placement changed: show it unfitted
        self._bands_changed()

    def fit_model(self) -> FitModel:
        """The fit defined here (Si calibration, background, bands); 2 · Map applies the same."""
        return FitModel(bands=self.bands(), background=self.cmb_bg.currentText(),
                        bg_opts=self.bg_opts(), si_calibrate=self.chk_si.isChecked(),
                        centre_tol=self.spn_tol.value())

    def bg_opts(self) -> dict:
        try:
            return {"order": self.spn_order.value(), "lam": float(self.ed_lam.text()),
                    "p": float(self.ed_p.text())}
        except ValueError:
            return {"order": self.spn_order.value()}

    # -- spectra (held by the service) ----------------------------------------------
    @property
    def items(self) -> List[dict]:
        return self.main.service.spectra

    @property
    def results(self) -> Dict[int, dict]:
        return self.main.service.spectra_fits

    def _add(self, label, wn, y, path, x=np.nan, ypos=np.nan):
        self.main.service.add_spectrum(label, wn, y, path, x, ypos)
        self.lst.addItem(label)

    def add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add spectra", "", "Spectra (*.txt *.dat *.csv *.h5);;All files (*)")
        _, bad = self.main.service.add_spectra_files(paths)
        for p, why in bad:
            self.main.status("Skipped %s: %s" % (os.path.basename(p), why))
        self._added()

    def add_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder of spectra")
        if not folder:
            return
        self.main.service.add_spectra_folder(folder)
        self._added()

    def use_map(self):
        if self.main.mapfile is None:
            self.main.status("Load a map first.")
            return
        self.main.service.add_map_spectra()
        self._added()

    def _refresh_list(self):
        self.lst.blockSignals(True)
        self.lst.clear()
        self.lst.addItems([it["label"] for it in self.items])
        self.lst.setCurrentRow(-1)
        self.lst.blockSignals(False)

    def _added(self):
        """After adding spectra: plot them all; nothing is fitted until a Fit button is pressed."""
        self._refresh_list()
        self.plot_all()

    def plot_all(self):
        if not self.items:
            self._placeholder()
            return
        fig = self.plot.figure
        fig.clear()
        views.draw_spectra_overlay(fig, self.items, self.bands())
        self.tbl_peaks.setRowCount(0)
        self.lbl.setText("%d spectra loaded. Define the bands to fit (or a preset), then "
                         "'Fit selected' or 'Fit all…'." % len(self.items))
        self.plot.draw()

    def clear(self):
        self.main.service.clear_spectra()
        self.lst.clear()
        self.tbl_peaks.setRowCount(0)
        self._placeholder()

    # -- fitting --------------------------------------------------------------
    def fit_item(self, i: int) -> dict:
        self._sync_model()
        return self.main.service.fit_spectrum(i)

    def _selected(self, i: int):
        """Show the chosen spectrum (with its fit, if it has been fitted); fitting is a button."""
        if not (0 <= i < len(self.items)):
            return
        if i in self.results:
            self.draw_fit(i, self.results[i])
            return
        fig = self.plot.figure
        fig.clear()
        views.draw_spectra_overlay(fig, [self.items[i]], self.bands())
        self.tbl_peaks.setRowCount(0)
        self.lbl.setText("Not fitted. Define bands (or a preset), then 'Fit selected' or 'Fit all…'.")
        self.plot.draw()

    def fit_selected(self):
        i = self.lst.currentRow()
        if not (0 <= i < len(self.items)):
            return
        if not self.bands():
            self.main.status("No bands to fit: add a band (From/To, peaks) or apply a preset.")
            return
        self.draw_fit(i, self.fit_item(i))

    def fit_all(self):
        if not self.items:
            return
        if not self.bands():
            self.main.status("No bands to fit: add a band (From/To, peaks) or apply a preset.")
            return
        self._sync_model()
        prog = QtWidgets.QProgressDialog("Fitting…", "Cancel", 0, len(self.items), self)
        prog.setWindowModality(QtCore.Qt.WindowModal)

        def tick(k, n):
            prog.setValue(k)
            QtWidgets.QApplication.processEvents()
            return not prog.wasCanceled()
        path = self.main.service.fit_all_spectra(tick)
        prog.setValue(len(self.items))
        if path is None:
            return
        self.draw_summary()
        self.main.status("Fitted %d spectra — saved %s" % (len(self.items), path))

    def _from_map(self) -> bool:
        return self.main.service.spectra_are_map()

    def _sync_model(self, *_):
        """Hand the fit settings of this tab to the backend (used by every fit, here and in 2 · Map)."""
        self.main.service.set_fit_model(self.fit_model())

    # -- drawing --------------------------------------------------------------
    def _placeholder(self):
        self.plot.figure.clear()
        draw_figure_message(self.plot.figure, "Add spectra")
        self.plot.draw()

    def draw_fit(self, i: int, res: dict):
        fig = self.plot.figure
        fig.clear()
        bands = [(n, b) for n, b in res["bands"].items()]
        if not bands:
            self._placeholder()
            return
        views.draw_band_fits(fig, bands)
        fill_peak_table(self.tbl_peaks, bands)
        notes = ["Si shift %+.2f cm⁻¹" % res["shift"] if self.chk_si.isChecked() else "no Si calibration",
                 "noise %.1f counts" % res["noise"]]
        for name, b in bands:
            if b.get("stopped"):
                notes.append("%s: stopped at %d — %s" % (name, b.get("n", 0), b["stopped"]))
        self.lbl.setText(";  ".join(notes))
        self.plot.draw()

    def draw_summary(self):
        """Peak count per band: a map when the spectra have positions,
        otherwise a histogram."""
        fig = self.plot.figure
        fig.clear()
        names, counts, xy = self.main.service.peak_counts()
        views.draw_peak_counts(fig, names, counts, xy, self.main.cfg)
        self.plot.draw()

    def correlation(self):
        """ω_G–ω_2D plot of these spectra: 3 · Analysis ▸ Strain / doping."""
        if not self.items:
            return
        self.main.show_tab(self.main.tab_strain)
        self.main.tab_strain.analyse()
