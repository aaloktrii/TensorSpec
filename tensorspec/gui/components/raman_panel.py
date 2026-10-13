"""Raman suite panel: the Raman workflow as three step tabs — 1 · Spectra (Peak fitting, Stacked
spectra), 2 · Map (Map Explorer, Compare cells), 3 · Analysis (Classification, Layer prediction,
Strain / doping) — with File / Analysis / Help menus, status bar and progress.

Was ``MainWindow`` of Raman_Classifier ``raman_map_classifier_8 (1).py``, now a QWidget so the
suite can embed it. The shared state and non-GUI logic live in gui/services/raman_service.py
(RamanService); the state attributes the tabs read as ``self.main.<name>`` are forwarded to it.
Each tab is a panel in gui/components/raman_tabs/."""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

from tensorspec.core.io.raman_loaders import MapFile, is_single_spectrum, parse_map_file, read_spot_folder
from tensorspec.core.raman.constants import APP_NAME
from tensorspec.core.raman.map_geometry import auto_align, describe_transform
from tensorspec.core.raman.voigt_2d import HAVE_LMFIT
from tensorspec.gui.components.raman_tabs.classification_panel import ClassificationTab
from tensorspec.gui.components.raman_tabs.map_explorer_panel import MapExplorerTab
from tensorspec.gui.components.raman_tabs.peak_fit_panel import PeakFitTab
from tensorspec.gui.components.raman_tabs.predict_panel import PredictTab
from tensorspec.gui.components.raman_tabs.spectra_panel import SpectraTab
from tensorspec.gui.components.raman_tabs.stack_panel import StackTab
from tensorspec.gui.components.raman_tabs.strain_doping_panel import StrainDopingTab
from tensorspec.gui.services.raman_service import RamanService

#: MainWindow attributes that now live on RamanService; the tabs keep reading them as main.<name>.
_SERVICE_STATE = ("cfg", "mapfile", "mapfile_raw", "layer_by_spec", "optical", "optical_image",
                  "fits", "g_fits", "ratio_maps", "ratio_ok", "ratio_note", "fit_tally",
                  "fwhm_map", "asym_maps", "engine")


class FitWorker(QtCore.QThread):
    """Fits every spectrum off the GUI thread (core.raman.batch.MapFitter does the work)."""

    progress = QtCore.Signal(int, str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, service: RamanService, parent=None):
        super().__init__(parent)
        self.fitter = service.map_fitter(progress=self.progress.emit)

    def cancel(self):
        self.fitter.cancel()

    def run(self):
        try:
            self.done.emit(self.fitter.run())
        except RuntimeError as exc:
            self.failed.emit(str(exc))


class RamanPanel(QtWidgets.QWidget):

    def __init__(self, service: Optional[RamanService] = None, parent=None):
        super().__init__(parent)
        self.service = service or RamanService()
        self.worker: Optional[FitWorker] = None

        # The tabs refer to each other while building, so create them all first.
        self.tab_map = MapExplorerTab(self)
        self.tab_class = ClassificationTab(self)
        self.tab_spec = SpectraTab(self)
        self.tab_stack = StackTab(self)
        self.tab_pred = PredictTab(self)
        self.tab_fit = PeakFitTab(self)
        self.tab_strain = StrainDopingTab(self)

        # Workflow order: 1 · Spectra → 2 · Map → 3 · Analysis, each step with its sub-tabs.
        self.tabs = QtWidgets.QTabWidget()
        self.step_tabs = {}
        for step, subs in (("1 · Spectra", ((self.tab_fit, "Peak fitting"),
                                            (self.tab_stack, "Stacked spectra"))),
                           ("2 · Map", ((self.tab_map, "Map Explorer"),
                                        (self.tab_spec, "Compare cells"))),
                           ("3 · Analysis", ((self.tab_class, "Classification"),
                                             (self.tab_pred, "Layer prediction"),
                                             (self.tab_strain, "Strain / doping")))):
            inner = QtWidgets.QTabWidget()
            inner.setDocumentMode(True)
            for w, name in subs:
                inner.addTab(w, name)
                self.step_tabs[w] = inner
            self.tabs.addTab(inner, step)

        self._menu_bar = QtWidgets.QMenuBar(self)
        self._status_bar = QtWidgets.QStatusBar(self)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setMenuBar(self._menu_bar)
        lay.addWidget(self.tabs, 1)
        lay.addWidget(self._status_bar)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.status("Ready. %s" % ("lmfit available." if HAVE_LMFIT
                                   else "lmfit not found — using scipy fallback."))
        self._menus()

    # -- MainWindow API kept for the tabs --------------------------------------------------------
    def menuBar(self) -> QtWidgets.QMenuBar:
        return self._menu_bar

    def statusBar(self) -> QtWidgets.QStatusBar:
        return self._status_bar

    # -- menus --------------------------------------------------------------
    def _menus(self):
        m = self.menuBar().addMenu("&File")
        self._add_action(m, "Open Raman map…", self.load_raman, "Ctrl+O")
        self._add_action(m, "Open map folder…", self.load_folder)
        self._add_action(m, "Open optical image…", self.load_optical)
        self._add_action(m, "Add spectra to stack…", self.tab_stack.add_files)
        m.addSeparator()
        self._add_action(m, "Save settings…", self.save_config)
        self._add_action(m, "Load settings…", self.load_config)
        m.addSeparator()
        self._add_action(m, "Quit", self.window().close, "Ctrl+Q")

        a = self.menuBar().addMenu("&Analysis")
        self._add_action(a, "Fit parameters…", self.edit_fit_params)
        self._add_action(a, "Re-fit all spectra", self.start_fitting)

        h = self.menuBar().addMenu("&Help")
        self._add_action(h, "About", self.about)

    @staticmethod
    def _add_action(menu, text, slot, shortcut=None):
        act = menu.addAction(text)
        act.triggered.connect(lambda *_: slot())
        if shortcut:
            act.setShortcut(shortcut)
        return act

    def show_tab(self, widget: QtWidgets.QWidget):
        """Bring a sub-tab (e.g. self.tab_stack) to the front, with its step tab."""
        inner = self.step_tabs[widget]
        self.tabs.setCurrentWidget(inner)
        inner.setCurrentWidget(widget)

    def status(self, text: str):
        self.statusBar().showMessage(text, 8000)

    def suggest_name(self, suffix: str) -> str:
        return self.service.suggest_name(suffix)

    # -- loading ------------------------------------------------------------
    def load_raman(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Raman map", "",
            "Text/map files (*.txt *.dat *.csv *.h5);;All files (*)")
        if path:
            self.open_raman(path)

    def load_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open map folder (one file per spot)")
        if path:
            self.open_raman(path)

    def open_raman(self, path: str):
        # A single spectrum is not a map: it goes to the stack tab instead.
        if not os.path.isdir(path) and is_single_spectrum(path):
            self.tab_stack.add_paths([path])
            self.show_tab(self.tab_stack)
            self.status("%s is a single spectrum — added to 1 · Spectra ▸ Stacked "
                        "spectra." % os.path.basename(path))
            return
        try:
            mf = read_spot_folder(path) if os.path.isdir(path) else parse_map_file(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, APP_NAME, "Could not parse the file:\n\n%s" % exc)
            return
        if mf.layout == "image":
            QtWidgets.QMessageBox.warning(
                self, APP_NAME,
                "This file is a camera image, not a Raman map — its rows carry "
                "one spatial coordinate and a row of pixel values, with no "
                "spectra to fit.\n\nLoading it as the optical image instead.")
            self.service.set_optical(mf)
            self.tab_map.chk_optical.setChecked(True)
            self.tab_map.refresh_map()
            return
        if mf.axis.max() < self.cfg["band_min"] or mf.axis.min() > self.cfg["band_max"]:
            QtWidgets.QMessageBox.warning(
                self, APP_NAME,
                "The axis of this file spans %.0f–%.0f, which does not cover the "
                "2D-band window %.0f–%.0f cm⁻¹.\n\nChange the window under "
                "Analysis ▸ Fit parameters."
                % (mf.axis.min(), mf.axis.max(),
                   self.cfg["band_min"], self.cfg["band_max"]))
        self.tab_class.close_regions_dialog()
        layer_mode_was_map = self.cfg.get("layer_mode") == "map"
        res = self.service.set_map(mf)
        if layer_mode_was_map:
            self.tab_class.rb_manual.setChecked(True)
        self.tab_class.update_map_status()
        t1 = self.tab_map
        t1.sync_transform_ui()
        for sp in (t1.spn_x0, t1.spn_x1):
            sp.blockSignals(True)
            sp.setRange(float(mf.axis[0]), float(mf.axis[-1]))
            sp.blockSignals(False)
        t1._spec = None
        self.tab_spec.reset()
        self.tab_class.update_auto_availability()
        self.tab_class.layers = None
        bits = ["Loaded %s — %d spectra on a %d × %d grid"
                % (os.path.basename(path), mf.n_points, mf.shape[1], mf.shape[0])]
        if mf.swapped:
            bits.append("X/Y taken from the #AxisType headers (column 1 is X)")
        if res:
            bits.append("narrow-peak cutoff set to %.1f cm⁻¹ from the header" % res)
        self.status("; ".join(bits))
        self.start_fitting()

    def load_optical(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open optical image", "",
            "Text/map files (*.txt *.dat *.csv);;All files (*)")
        if not path:
            return
        try:
            opt = parse_map_file(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, APP_NAME, "Could not parse the file:\n\n%s" % exc)
            return
        self.service.set_optical(opt)
        kind = "camera image" if opt.is_optical else "image-like map"
        self.status("Loaded %s (%s) — %d × %d pixels, X %.1f–%.1f µm, "
                    "Y %.1f–%.1f µm%s"
                    % (os.path.basename(path), kind,
                       opt.shape[1], opt.shape[0],
                       opt.ux.min(), opt.ux.max(), opt.uy.min(), opt.uy.max(),
                       self._coverage_note(opt)))
        self.tab_map.chk_optical.setChecked(True)
        self.tab_map.refresh_map()
        self.tab_class.refresh_map()

    def _coverage_note(self, opt: MapFile) -> str:
        return self.service.coverage_note(opt)

    # -- fitting ------------------------------------------------------------
    def start_fitting(self):
        if self.mapfile is None:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Load a map first.")
            return
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.worker = FitWorker(self.service, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_fits)
        self.worker.failed.connect(self._on_fit_failed)
        self.worker.start()

    def _on_progress(self, pct: int, msg: str):
        self.progress.setValue(pct)
        self.statusBar().showMessage(msg)

    def _on_fit_failed(self, msg: str):
        self.progress.setVisible(False)
        self.status(msg.strip().splitlines()[-1] if msg else "Fitting failed.")

    def _on_fits(self, payload):
        self.progress.setVisible(False)
        ok = self.service.set_fits(payload)
        tally = self.fit_tally
        self.tab_class.update_auto_availability()
        self.tab_map.refresh_map()
        self.tab_spec.refresh_mini()
        mf = self.mapfile
        med = np.nanmedian(self.fwhm_map) if ok else np.nan
        if ok == 0:
            top = max(tally, key=tally.get) if tally else "rejected"
            self.status("No spectrum passed the quality filters — %d of %d "
                        "failed: %s. See the map panel for the full breakdown."
                        % (tally.get(top, 0), mf.n_points, top))
        else:
            self.status("Fitted %d / %d spectra (%d rejected) — median FWHM(2D) "
                        "%.1f cm⁻¹. Click a pixel to inspect it."
                        % (ok, mf.n_points, mf.n_points - ok, med))

    def rebuild_maps(self) -> int:
        return self.service.rebuild_maps()

    def layer_grid(self) -> np.ndarray:
        return self.service.layer_grid()

    # -- orientation ---------------------------------------------------------
    def current_transform(self) -> Optional[dict]:
        return self.service.current_transform()

    def set_transform(self, t: Optional[dict]):
        """Re-place the same spectra under a new orientation / offset."""
        if not self.service.apply_transform(t):
            return
        # Regions were read from the camera image at the old placement; the
        # assigned layer counts survive (they are per spectrum), the dialog
        # does not.
        self.tab_class.close_regions_dialog()
        self.tab_map.sel = None
        self.tab_map._spec = None
        self.tab_class.layers = None
        self.tab_map.sync_transform_ui()
        self.tab_map.refresh_map()
        self.tab_class.refresh_map()
        self.tab_spec.refresh_mini()
        self.tab_spec._rebuild_list()
        note = self._coverage_note(self.optical) if self.optical is not None else ""
        self.status("Map placed as  %s%s" % (describe_transform(self.current_transform()), note))

    def run_auto_align(self):
        """Fit the map's offset to the camera image.

        By default the orientation is locked to the current one (the
        instrument convention unless changed) and only the offset is searched.
        Searching all eight orientations is optional: on images with strong
        illumination gradients it can lock onto the background instead of the
        flake, which is what happened on 004_01_8.
        """
        if self.mapfile_raw is None or self.optical is None:
            QtWidgets.QMessageBox.information(
                self, APP_NAME, "Auto-align needs both a Raman map and an optical "
                "image loaded.")
            return
        if self.mapfile_raw.layout != "spectra":
            return
        t0 = self.current_transform()
        rng = float(self.cfg.get("align_search_um", 15.0))
        full = bool(self.tab_map.chk_allorient.isChecked())
        self.status("Aligning map to the optical image…")
        QtWidgets.QApplication.processEvents()
        orient = None if full else [(t0["swap"], t0["flip_x"], t0["flip_y"])]
        want = self.tab_map.cmb_contrast.currentIndex()   # 0 auto, 1 brighter, 2 darker
        sign = {0: None, 1: 1, 2: -1}[want]
        cands, sig = auto_align(self.mapfile_raw, self.optical, self.cfg,
                                search_um=rng, orientations=orient, sign=sign)
        cands = [c for c in cands if c[0] > 0]
        if not cands:
            QtWidgets.QMessageBox.warning(
                self, APP_NAME, "Could not find an alignment — is there a band "
                "signal in this map and a flake in the optical image?")
            return
        best = cands[0]
        t = best[2]
        at_edge = max(abs(t["dx"]), abs(t["dy"])) >= rng - 0.5
        self.set_transform(t)
        if full:
            runner = next((c for c in cands[1:]
                           if (c[2]["swap"], c[2]["flip_x"], c[2]["flip_y"]) !=
                           (t["swap"], t["flip_x"], t["flip_y"])), None)
            margin = best[0] - (runner[0] if runner else 0.0)
            verdict = ("clear" if best[0] >= 0.5 and margin >= 0.2 else
                       "tentative — check it by eye, or against LabSpec's overlay")
            extra = ("Next best orientation:  %s   (r = %+.2f)\n\n"
                     % (describe_transform(runner[2]), runner[1]) if runner else "")
        else:
            verdict = ("offset found" if best[0] >= 0.3 else
                       "weak match — check it by eye")
            extra = "Orientation locked; only the offset was searched (±%.0f µm).\n\n" % rng
        msg = ("Aligned using the %s height against the optical image.\n\n"
               "Placement:  %s   (r = %+.2f)\n%s"
               "Result: %s.%s"
               % (sig, describe_transform(t), best[1], extra, verdict,
                  "\n\n⚠ The best offset lies at the edge of the search range — "
                  "the true offset may be larger. Raise the search range." if at_edge else ""))
        QtWidgets.QMessageBox.information(self, "Auto-align", msg)

    # -- settings -----------------------------------------------------------
    def edit_fit_params(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Fit parameters")
        form = QtWidgets.QFormLayout(dlg)
        s_lo = QtWidgets.QDoubleSpinBox(); s_lo.setRange(0, 5000); s_lo.setDecimals(1)
        s_lo.setValue(self.cfg["band_min"])
        s_hi = QtWidgets.QDoubleSpinBox(); s_hi.setRange(0, 5000); s_hi.setDecimals(1)
        s_hi.setValue(self.cfg["band_max"])
        s_snr = QtWidgets.QDoubleSpinBox(); s_snr.setRange(0, 1000); s_snr.setDecimals(1)
        s_snr.setValue(self.cfg["min_snr"])
        s_max = QtWidgets.QDoubleSpinBox(); s_max.setRange(5, 2000); s_max.setDecimals(1)
        s_max.setValue(self.cfg["max_fwhm"])
        s_min = QtWidgets.QDoubleSpinBox(); s_min.setRange(0, 200); s_min.setDecimals(1)
        s_min.setValue(self.cfg["min_fwhm"])
        s_min.setToolTip("Peaks narrower than the spectral resolution are "
                         "artefacts, not bands. Set from the file header "
                         "where one is present.")
        chk_ds = QtWidgets.QCheckBox("Remove cosmic-ray spikes before fitting")
        chk_ds.setChecked(bool(self.cfg["despike"]))
        s_span = QtWidgets.QDoubleSpinBox(); s_span.setRange(0.2, 10.0)
        s_span.setSingleStep(0.1); s_span.setValue(self.cfg["asym_span"])
        s_smooth = QtWidgets.QSpinBox(); s_smooth.setRange(0, 51); s_smooth.setSingleStep(2)
        s_smooth.setValue(int(self.cfg["smooth_window"]))
        cmb_engine = QtWidgets.QComboBox()
        cmb_engine.addItems(["scipy", "lmfit"])
        cmb_engine.setCurrentText(self.cfg.get("engine", "scipy"))
        if not HAVE_LMFIT:
            cmb_engine.model().item(1).setEnabled(False)
            cmb_engine.setToolTip("lmfit is not installed")
        chk_g = QtWidgets.QCheckBox("Let the Lorentzian width vary independently")
        chk_g.setChecked(bool(self.cfg["free_gamma"]))
        chk_p = QtWidgets.QCheckBox("Fit across CPU cores (large maps)")
        chk_p.setChecked(bool(self.cfg["parallel"]))
        form.addRow("2D band window, lower [cm⁻¹]", s_lo)
        form.addRow("2D band window, upper [cm⁻¹]", s_hi)
        form.addRow("Minimum SNR to accept a fit", s_snr)
        form.addRow("Maximum credible FWHM [cm⁻¹]", s_max)
        form.addRow("Minimum credible FWHM [cm⁻¹]", s_min)
        form.addRow(chk_ds)
        form.addRow("Asymmetry window (× FWHM)", s_span)
        form.addRow("Smoothing points (shape metrics)", s_smooth)
        form.addRow("Fitting engine", cmb_engine)
        form.addRow(chk_g)
        form.addRow(chk_p)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                        QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        if s_hi.value() <= s_lo.value():
            QtWidgets.QMessageBox.warning(self, APP_NAME,
                                          "Upper bound must exceed lower bound.")
            return
        self.cfg.update(band_min=s_lo.value(), band_max=s_hi.value(),
                        min_snr=s_snr.value(), max_fwhm=s_max.value(),
                        min_fwhm=s_min.value(), despike=chk_ds.isChecked(),
                        asym_span=s_span.value(),
                        smooth_window=int(s_smooth.value()),
                        engine=cmb_engine.currentText(),
                        free_gamma=chk_g.isChecked(),
                        parallel=chk_p.isChecked())
        if self.mapfile is not None:
            self.start_fitting()

    def save_config(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save settings", "raman_classifier_settings.json",
            "JSON (*.json)")
        if not path:
            return
        self.tab_class._pull_config()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.service.config_to_json(), fh, indent=2)
        self.status("Settings saved to %s" % path)

    def load_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load settings", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            cfg = self.service.config_from_json(raw)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, APP_NAME,
                                           "Could not read settings:\n\n%s" % exc)
            return
        self.cfg = cfg
        if self.mapfile_raw is not None:
            self.set_transform(cfg.get("map_transform"))
        self.tab_class.spn_tol.setValue(
            float(np.median([t["ABA"] for t in cfg["tol"].values()])))
        self.tab_class.push_config()
        self.tab_map.sld_alpha.setValue(int(cfg["optical_alpha"] * 100))
        self.tab_map.cmb_cmap.setCurrentText(cfg["cmap"])
        self.status("Settings loaded from %s" % path)

    def about(self):
        QtWidgets.QMessageBox.about(
            self, "About " + APP_NAME,
            "<b>%s</b><br><br>"
            "FWHM(2D) mapping and ABA/ABC stacking classification of "
            "multilayer graphene Raman maps.<br><br>"
            "Fit: single Voigt + linear baseline over %.0f–%.0f cm⁻¹ "
            "(%s).<br>FWHM from the Olivero–Longbothum approximation.<br><br>"
            "Default thresholds after Bouhafs <i>et al.</i>, "
            "Carbon <b>177</b> (2021) 282–290, Fig. 2c. These were read off a "
            "published figure — treat them as starting values and calibrate "
            "against your own reference samples."
            % (APP_NAME, self.cfg["band_min"], self.cfg["band_max"],
               self.cfg.get("engine", "scipy")))

    def closeEvent(self, event):
        self.stop_worker()
        event.accept()

    def stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)

    # Embedded in a suite window, the panel gets no closeEvent of its own: watch the window.
    def showEvent(self, event):
        win = self.window()
        if win is not self and getattr(self, "_watching", None) is not win:
            win.installEventFilter(self)
            self._watching = win
        super().showEvent(event)

    def eventFilter(self, obj, event):
        if obj is getattr(self, "_watching", None) and event.type() == QtCore.QEvent.Close:
            self.stop_worker()
        return super().eventFilter(obj, event)


def _forward(name):
    return property(lambda self: getattr(self.service, name),
                    lambda self, value: setattr(self.service, name, value))


for _name in _SERVICE_STATE:
    setattr(RamanPanel, _name, _forward(_name))
