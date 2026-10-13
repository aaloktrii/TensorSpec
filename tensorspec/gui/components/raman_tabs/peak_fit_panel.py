"""Tab 6 · Peak fitting: optional Si calibration, background model, Lorentzians with a fixed or
data-chosen peak count, batch over files / folders / the loaded map, and the omega_G - omega_2D
strain/doping correlation (shown in 3 · Analysis ▸ Strain / doping). Results for the loaded map are
written to /analysis/raman_peakfit.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Tab 6 - Peak fitting); calls go through RamanEngineRouter (main.engine),
figures through plotting/backends/raman_views/peak_fit.py."""
from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING
from typing import Dict, List

import numpy as np
from PySide6 import QtCore, QtWidgets

from tensorspec.core.io.raman_loaders import _SPOT_RE, is_single_spectrum, read_single_spectrum, short_label
from tensorspec.core.raman.fit_model import FitModel
from tensorspec.gui.components.raman_tabs.layout import _fmt, fill_peak_table
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
        self.items: List[dict] = []      # {label, wn, y, path, x, ypos}
        self.results: Dict[int, dict] = {}
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

        self.tbl_bands = QtWidgets.QTableWidget(0, 5)
        self.tbl_bands.setHorizontalHeaderLabels(["Band", "From", "To", "Peaks", "Max"])
        self.tbl_bands.verticalHeader().setVisible(False)
        self.tbl_bands.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tbl_bands.setMaximumHeight(140)
        left.addWidget(self.tbl_bands)
        r = QtWidgets.QHBoxLayout()
        for text, fn in (("Add band", lambda: self._add_band_row("band", 1000, 1100, "Auto", 3)),
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
        root.addWidget(self.plot, 1)
        self._placeholder()

    def _add_band_row(self, name, lo, hi, peaks, nmax):
        r = self.tbl_bands.rowCount()
        self.tbl_bands.insertRow(r)
        self.tbl_bands.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
        self.tbl_bands.setItem(r, 1, QtWidgets.QTableWidgetItem("%g" % lo))
        self.tbl_bands.setItem(r, 2, QtWidgets.QTableWidgetItem("%g" % hi))
        cmb = QtWidgets.QComboBox()
        cmb.addItems(["Auto"] + [str(k) for k in range(1, 7)])
        cmb.setCurrentText(str(peaks))
        self.tbl_bands.setCellWidget(r, 3, cmb)
        sp = QtWidgets.QSpinBox()
        sp.setRange(1, 6)
        sp.setValue(int(nmax))
        self.tbl_bands.setCellWidget(r, 4, sp)

    def set_bands(self, bands, si_calibrate: bool):
        """Replace the band table: rows of (name, from, to, peaks or "auto", max)."""
        self.tbl_bands.setRowCount(0)
        for name, lo, hi, n, nmax in bands:
            self._add_band_row(name, lo, hi, "Auto" if n == "auto" else n, nmax)
        self.chk_si.setChecked(bool(si_calibrate))

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
            n = self.tbl_bands.cellWidget(r, 3).currentText()
            out.append((self.tbl_bands.item(r, 0).text(), lo, hi,
                        "auto" if n == "Auto" else int(n),
                        self.tbl_bands.cellWidget(r, 4).value()))
        return out

    def fit_model(self) -> FitModel:
        """The fit defined here (Si calibration, background, bands); 2 · Map applies the same."""
        return FitModel(bands=self.bands(), background=self.cmb_bg.currentText(),
                        bg_opts=self.bg_opts(), si_calibrate=self.chk_si.isChecked())

    def bg_opts(self) -> dict:
        try:
            return {"order": self.spn_order.value(), "lam": float(self.ed_lam.text()),
                    "p": float(self.ed_p.text())}
        except ValueError:
            return {"order": self.spn_order.value()}

    # -- spectra --------------------------------------------------------------
    def _add(self, label, wn, y, path, x=np.nan, ypos=np.nan):
        self.items.append({"label": label, "wn": np.asarray(wn, float),
                           "y": np.asarray(y, float), "path": path, "x": x, "ypos": ypos})
        self.lst.addItem(label)

    def add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add spectra", "", "Spectra (*.txt *.dat *.csv *.h5);;All files (*)")
        for p in paths:
            try:
                s = read_single_spectrum(p)
                self._add(s.label, s.wn, s.inten, p)
            except Exception as exc:
                self.main.status("Skipped %s: %s" % (os.path.basename(p), exc))
        self._added()

    def add_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder of spectra")
        if not folder:
            return
        for name in sorted(os.listdir(folder)):
            p = os.path.join(folder, name)
            if not name.lower().endswith((".txt", ".dat", ".csv")) or not is_single_spectrum(p):
                continue
            s = read_single_spectrum(p)
            m = _SPOT_RE.search(name)
            self._add(short_label(p), s.wn, s.inten, p,
                      float(m.group(1)) if m else np.nan, float(m.group(2)) if m else np.nan)
        self._added()

    def use_map(self):
        mf = self.main.mapfile
        if mf is None:
            self.main.status("Load a map first.")
            return
        for i in range(mf.n_points):
            self._add("spot %d (%.1f, %.1f)" % (i, mf.x[i], mf.y[i]), mf.axis, mf.data[i],
                      mf.path, float(mf.x[i]), float(mf.y[i]))
        self._added()

    def _added(self):
        """After adding spectra: plot them all; nothing is fitted until a Fit button is pressed."""
        self.lst.blockSignals(True)
        self.lst.setCurrentRow(-1)
        self.lst.blockSignals(False)
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
        self.items, self.results = [], {}
        self.lst.clear()
        self.tbl_peaks.setRowCount(0)
        self._placeholder()

    # -- fitting --------------------------------------------------------------
    def fit_item(self, i: int) -> dict:
        it = self.items[i]
        res = self.main.engine.fit_spectrum(it["wn"], it["y"], self.fit_model())
        self.results[i] = res
        return res

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
        prog = QtWidgets.QProgressDialog("Fitting…", "Cancel", 0, len(self.items), self)
        prog.setWindowModality(QtCore.Qt.WindowModal)
        for i in range(len(self.items)):
            if prog.wasCanceled():
                return
            prog.setValue(i)
            self.fit_item(i)
        prog.setValue(len(self.items))
        src = self.items[0]["path"].rstrip(os.sep)
        base = src if os.path.isdir(src) else os.path.dirname(src)
        out = os.path.join(base, "output")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "peak_fits.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["spectrum", "x", "y", "si_shift", "band", "n_peaks", "peak",
                        "centre", "centre_err", "fwhm", "fwhm_err", "height", "height_err",
                        "area", "area_err", "r2", "note"])
            for i, it in enumerate(self.items):
                res = self.results[i]
                for name, b in res["bands"].items():
                    rows = b.get("comps") or [None]
                    for k, c in enumerate(rows):
                        w.writerow([it["label"], _fmt(it["x"]), _fmt(it["ypos"]),
                                    _fmt(res["shift"]), name, b.get("n", 0), k + 1 if c else ""]
                                   + (["%.3f" % c[q] for q in ("pos", "pos_err", "fwhm", "fwhm_err",
                                                                "height", "height_err", "area",
                                                                "area_err")] if c else [""] * 8)
                                   + [_fmt(b.get("r2", np.nan)),
                                      b.get("why") or b.get("stopped", "")])
        self.draw_summary()
        if self._from_map():
            self.main.service.write_table("raman_peakfit", self._result_rows(), "si_shift",
                                          {"background": self.cmb_bg.currentText()})
        self.main.status("Fitted %d spectra — saved %s" % (len(self.items), path))

    def _from_map(self) -> bool:
        """True when the list holds exactly the spots of the loaded map ('Use loaded map')."""
        mf = self.main.mapfile
        return (mf is not None and len(self.items) == mf.n_points
                and all(it["path"] == mf.path for it in self.items))

    def _result_rows(self) -> List[dict]:
        """One row per spectrum for /analysis/raman_peakfit: position, FWHM and height of
        each fitted peak per band."""
        rows = []
        for i, it in enumerate(self.items):
            res = self.results.get(i)
            if res is None:
                continue
            row = {"spectrum": it["label"], "x": it["x"], "y": it["ypos"],
                   "si_shift": res["shift"]}
            for name, b in res["bands"].items():
                row["%s_n_peaks" % name] = b.get("n", 0)
                for k, c in enumerate(b.get("comps") or []):
                    for q in ("pos", "fwhm", "height"):
                        row["%s_%d_%s" % (name, k + 1, q)] = c[q]
            rows.append(row)
        return rows

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
        names = [b[0] for b in self.bands()]
        has_xy = sum(np.isfinite(it["x"]) for it in self.items) > 3
        counts = [np.array([self.results[i]["bands"].get(name, {}).get("n", 0)
                            if self.results[i]["bands"].get(name, {}).get("ok") else 0
                            for i in range(len(self.items))]) for name in names]
        xy = ((np.array([it["x"] for it in self.items]),
               np.array([it["ypos"] for it in self.items])) if has_xy else None)
        views.draw_peak_counts(fig, names, counts, xy, self.main.cfg)
        self.plot.draw()

    def correlation(self):
        """ω_G–ω_2D plot of these spectra: 3 · Analysis ▸ Strain / doping."""
        if not self.items:
            return
        self.main.show_tab(self.main.tab_strain)
        self.main.tab_strain.analyse()
