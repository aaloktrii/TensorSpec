"""Tab 2 · Classification: layer number (manual / auto from I(2D)/I(G) / layer map) and ABA/ABC
stacking per pixel, editable threshold table, histograms, PNG / CSV export. Results are written to
/analysis/raman_classification; figures are drawn by plotting/backends/raman_views/classification.py.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Tab 2 - Classification: ClassificationTab)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

from tensorspec.core.raman.constants import (APP_NAME, BOUHAFS_FWHM, DEFAULT_COLORS, DEFAULT_TOL,
                                             LAYERS)
from tensorspec.gui.components.raman_tabs.layer_regions_dialog import LayerRegionsDialog
from tensorspec.gui.components.raman_tabs.layout import ThresholdTable
from tensorspec.gui.components.raman_tabs.thickness_dialog import ThicknessDialog
from tensorspec.plotting.backends.raman_matplotlib_engine import draw_placeholder
from tensorspec.plotting.backends.raman_views import classification as views
from tensorspec.plotting.viewers.raman_canvas import Canvas, export_figure

if TYPE_CHECKING:
    from tensorspec.gui.components.raman_panel import RamanPanel

class ClassificationTab(QtWidgets.QWidget):

    def __init__(self, main: "RamanPanel"):
        super().__init__()
        self.main = main
        self._build()

    # -- UI -----------------------------------------------------------------
    def _build(self):
        root = QtWidgets.QHBoxLayout(self)

        # ---------------- left: inputs ----------------
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(620)
        pl = QtWidgets.QVBoxLayout(panel)

        # Layer assignment
        gb_layer = QtWidgets.QGroupBox("Layer count")
        gl = QtWidgets.QGridLayout(gb_layer)
        self.rb_manual = QtWidgets.QRadioButton("Manual")
        self.rb_auto = QtWidgets.QRadioButton("Auto — thickness from I(2D)/I(G)")
        self.rb_manual.setChecked(True)
        self.cmb_layer = QtWidgets.QComboBox()
        self.cmb_layer.addItems(["%dL" % n for n in LAYERS])
        self.rb_manual.toggled.connect(self.cmb_layer.setEnabled)
        gl.addWidget(self.rb_manual, 0, 0)
        gl.addWidget(self.cmb_layer, 0, 1)
        gl.addWidget(self.rb_auto, 1, 0, 1, 2)
        self.btn_thick = QtWidgets.QPushButton("Calculate thickness from I(2D)/I(G)…")
        self.btn_thick.clicked.connect(self.show_thickness)
        gl.addWidget(self.btn_thick, 2, 0, 1, 2)
        self.lbl_auto = QtWidgets.QLabel("Load a map to see whether Auto is available.")
        self.lbl_auto.setWordWrap(True)
        self.lbl_auto.setStyleSheet("font-size:11px;")
        gl.addWidget(self.lbl_auto, 3, 0, 1, 2)
        note = QtWidgets.QLabel(
            "Auto sets each pixel's layer count from the ratio of the 2D and "
            "G band intensities, so it needs a wide acquisition (600 gr/mm) "
            "containing both. The ratio drops steeply from 1L to 3L and then "
            "flattens, so it resolves 1L, 2L, 3L and '≥4L' — not 5L versus "
            "7L. ABA/ABC is only defined from 3 layers up, so 1L and 2L "
            "pixels get a thickness but no stacking call. A 2D-only file "
            "(1800 gr/mm) has no G band: Manual only.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666; font-size:11px;")
        gl.addWidget(note, 4, 0, 1, 2)
        self.rb_map = QtWidgets.QRadioButton("Layer map — N per region (from AFM)")
        gl.addWidget(self.rb_map, 5, 0, 1, 2)
        self.btn_layermap = QtWidgets.QPushButton("Define layer regions…")
        self.btn_layermap.clicked.connect(self.show_layermap)
        gl.addWidget(self.btn_layermap, 6, 0, 1, 2)
        self.lbl_map = QtWidgets.QLabel("No layer map yet.")
        self.lbl_map.setWordWrap(True)
        self.lbl_map.setStyleSheet("font-size:11px;")
        gl.addWidget(self.lbl_map, 7, 0, 1, 2)
        note2 = QtWidgets.QLabel(
            "Optional, for flakes with terraces of different thickness. Optical "
            "contrast draws the regions, you enter each region's N or its AFM "
            "height from SiO₂, and each region is classified against its own "
            "width windows. Simple flakes: just use Manual.")
        note2.setWordWrap(True)
        note2.setStyleSheet("color:#666; font-size:11px;")
        gl.addWidget(note2, 8, 0, 1, 2)
        pl.addWidget(gb_layer)

        # Technique
        gb_tech = QtWidgets.QGroupBox("Technique")
        tl = QtWidgets.QVBoxLayout(gb_tech)
        self.rb_w = QtWidgets.QRadioButton("FWHM(2D) width only")
        self.rb_a = QtWidgets.QRadioButton("2D shape asymmetry only")
        self.rb_b = QtWidgets.QRadioButton("Both combined (AND)")
        self.rb_b.setChecked(True)
        for rb in (self.rb_w, self.rb_a, self.rb_b):
            tl.addWidget(rb)
        arow = QtWidgets.QHBoxLayout()
        arow.addWidget(QtWidgets.QLabel("Metric"))
        self.cmb_metric = QtWidgets.QComboBox()
        self.cmb_metric.addItems(["hwhm", "area", "residual"])
        self.cmb_metric.setToolTip(
            "hwhm     — ratio of the two half-widths at half maximum\n"
            "area     — ratio of the two lobe areas\n"
            "residual — low-frequency shoulder left over by the Voigt fit")
        arow.addWidget(self.cmb_metric)
        arow.addWidget(QtWidgets.QLabel("Threshold"))
        self.spn_asym = QtWidgets.QDoubleSpinBox()
        self.spn_asym.setRange(-1.0, 1.0)
        self.spn_asym.setSingleStep(0.01)
        self.spn_asym.setDecimals(3)
        self.spn_asym.setValue(0.15)
        arow.addWidget(self.spn_asym)
        arow.addStretch(1)
        tl.addLayout(arow)
        anote = QtWidgets.QLabel(
            "Asymmetry is positive when the band carries extra weight on its "
            "low-frequency shoulder (~2680 cm⁻¹), the ABC signature. All three "
            "metrics are measured about the maximum of the smoothed band, not "
            "the fitted centre — a single Voigt drifts into the shoulder and "
            "masks the asymmetry. The 0.15 default is a placeholder: use "
            "'Threshold histograms' to set it from your own data.")
        anote.setWordWrap(True)
        anote.setStyleSheet("color:#666; font-size:11px;")
        tl.addWidget(anote)
        pl.addWidget(gb_tech)

        # Threshold table
        gb_tab = QtWidgets.QGroupBox("FWHM thresholds and colours (editable)")
        tabl = QtWidgets.QVBoxLayout(gb_tab)
        self.table = ThresholdTable(self.main.cfg)
        self.table.changed.connect(self._table_changed)
        tabl.addWidget(self.table)
        hint = QtWidgets.QLabel(
            "A width must land inside a window to be called that stacking; one "
            "matching neither is left Unclassified rather than pushed to the "
            "nearer side. Windows shown in red overlap, in amber leave a dead "
            "zone between them. Click a colour cell to change it. Reference "
            "values from Bouhafs et al., Carbon 177 (2021) Fig. 2c — verify "
            "them, and the tolerances, against your own samples.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:11px;")
        tabl.addWidget(hint)
        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(QtWidgets.QLabel("Set every ± to"))
        self.spn_tol = QtWidgets.QDoubleSpinBox()
        self.spn_tol.setRange(0.0, 200.0)
        self.spn_tol.setDecimals(1)
        self.spn_tol.setSingleStep(1.0)
        self.spn_tol.setValue(DEFAULT_TOL)
        self.spn_tol.setSuffix(" cm⁻¹")
        self.spn_tol.setFixedWidth(100)
        trow.addWidget(self.spn_tol)
        btn_apply = QtWidgets.QPushButton("Apply to all")
        btn_apply.clicked.connect(
            lambda: self.table.set_all_tolerances(self.spn_tol.value()))
        trow.addWidget(btn_apply)
        btn_half = QtWidgets.QPushButton("Set ± to half-gap")
        btn_half.setToolTip(
            "Each window reaches exactly to the midpoint, so the ABA/ABC cut "
            "is the old boundary rule — but a width far from the table is now "
            "left Unclassified instead of being forced to a side.")
        btn_half.clicked.connect(self.table.set_tolerances_to_half_gap)
        trow.addWidget(btn_half)
        trow.addStretch(1)
        tabl.addLayout(trow)

        orow = QtWidgets.QHBoxLayout()
        orow.addWidget(QtWidgets.QLabel("Widths inside both windows:"))
        self.cmb_overlap = QtWidgets.QComboBox()
        self.cmb_overlap.addItems(["nearest reference value",
                                   "leave Unclassified"])
        self.cmb_overlap.setToolTip(
            "Only reached when a tolerance is wide enough that the ABA and "
            "ABC windows collide. 'nearest' keeps the midpoint behaviour "
            "inside the overlap; 'Unclassified' declines to call it.")
        orow.addWidget(self.cmb_overlap)
        orow.addStretch(1)
        tabl.addLayout(orow)

        rowb = QtWidgets.QHBoxLayout()
        btn_reset = QtWidgets.QPushButton("Reset to Bouhafs defaults")
        btn_reset.clicked.connect(self._reset_defaults)
        rowb.addWidget(btn_reset)
        rowb.addStretch(1)
        tabl.addLayout(rowb)
        pl.addWidget(gb_tab)

        # Run + export
        self.btn_hist = QtWidgets.QPushButton("Threshold histograms…")
        self.btn_hist.clicked.connect(self.show_histograms)
        pl.addWidget(self.btn_hist)

        self.btn_fit2d = QtWidgets.QPushButton("Fit 2D band on all pixels")
        self.btn_fit2d.setToolTip("Voigt fit of every pixel's 2D band (and G, when covered): "
                                  "FWHM(2D), asymmetry and I(2D)/I(G) maps used by the "
                                  "classification. Settings: Analysis ▸ Fit parameters.")
        self.btn_fit2d.clicked.connect(self.main.start_fitting)
        pl.addWidget(self.btn_fit2d)
        self.btn_run = QtWidgets.QPushButton("Classify map")
        self.btn_run.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn_run.clicked.connect(self.run_classification)
        pl.addWidget(self.btn_run)

        erow = QtWidgets.QHBoxLayout()
        self.btn_png = QtWidgets.QPushButton("Export figure (PDF/SVG/PNG)…")
        self.btn_csv = QtWidgets.QPushButton("Export CSV…")
        self.btn_png.clicked.connect(self.export_png)
        self.btn_csv.clicked.connect(self.export_csv)
        erow.addWidget(self.btn_png)
        erow.addWidget(self.btn_csv)
        pl.addLayout(erow)

        self.stats = QtWidgets.QTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setMaximumHeight(150)
        self.stats.setStyleSheet("font-family:monospace; font-size:11px;")
        pl.addWidget(self.stats)
        pl.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(640)
        root.addWidget(scroll)

        # ---------------- right: output map ----------------
        right = QtWidgets.QVBoxLayout()
        obar = QtWidgets.QHBoxLayout()
        self.chk_optical = QtWidgets.QCheckBox("Overlay optical")
        self.chk_optical.toggled.connect(self.refresh_map)
        self.sld_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_alpha.setRange(0, 100)
        self.sld_alpha.setValue(50)
        self.sld_alpha.setFixedWidth(140)
        self.sld_alpha.valueChanged.connect(self._alpha_changed)
        self.chk_grid = QtWidgets.QCheckBox("Grid overlay")
        self.chk_grid.toggled.connect(self.refresh_map)
        obar.addWidget(self.chk_optical)
        obar.addWidget(QtWidgets.QLabel("Opacity"))
        obar.addWidget(self.sld_alpha)
        obar.addWidget(self.chk_grid)
        obar.addStretch(1)
        right.addLayout(obar)
        self.canvas = Canvas(figsize=(6, 6))
        right.addWidget(self.canvas, 1)
        root.addLayout(right, 1)

        self._placeholder()

    # -- reactions ----------------------------------------------------------
    def _alpha_changed(self, v):
        self.main.cfg["optical_alpha"] = v / 100.0
        self.refresh_map()

    def _table_changed(self):
        self.stats.append("Thresholds edited — press 'Classify map' to re-run.")

    def _reset_defaults(self):
        self.main.cfg["fwhm"] = {k: dict(v) for k, v in BOUHAFS_FWHM.items()}
        self.main.cfg["colors"] = {k: dict(v) for k, v in DEFAULT_COLORS.items()}
        self.main.cfg["tol"] = {k: {"ABA": DEFAULT_TOL, "ABC": DEFAULT_TOL}
                                for k in BOUHAFS_FWHM}
        self.spn_tol.setValue(DEFAULT_TOL)
        self.table.reload()

    def _pull_config(self):
        cfg = self.main.cfg
        cfg["layer_mode"] = ("manual" if self.rb_manual.isChecked()
                             else "map" if self.rb_map.isChecked() else "auto")
        cfg["manual_layer"] = LAYERS[self.cmb_layer.currentIndex()]
        cfg["technique"] = ("fwhm" if self.rb_w.isChecked()
                            else "asym" if self.rb_a.isChecked() else "both")
        cfg["asym_metric"] = self.cmb_metric.currentText()
        cfg["asym_threshold"] = float(self.spn_asym.value())
        cfg["overlap_policy"] = ("unclassified"
                                 if self.cmb_overlap.currentIndex() else "nearest")

    def push_config(self):
        """Mirror a freshly loaded config into the widgets."""
        cfg = self.main.cfg
        self.rb_manual.setChecked(cfg["layer_mode"] == "manual")
        self.rb_auto.setChecked(cfg["layer_mode"] == "auto")
        self.rb_map.setChecked(cfg["layer_mode"] == "map")
        if cfg["manual_layer"] in LAYERS:
            self.cmb_layer.setCurrentIndex(LAYERS.index(cfg["manual_layer"]))
        self.rb_w.setChecked(cfg["technique"] == "fwhm")
        self.rb_a.setChecked(cfg["technique"] == "asym")
        self.rb_b.setChecked(cfg["technique"] == "both")
        self.cmb_metric.setCurrentText(cfg["asym_metric"])
        self.spn_asym.setValue(cfg["asym_threshold"])
        self.cmb_overlap.setCurrentIndex(
            1 if cfg.get("overlap_policy") == "unclassified" else 0)
        self.table.cfg = cfg
        self.table.reload()

    # -- classification -----------------------------------------------------
    def update_auto_availability(self):
        """Enable Auto only when the file holds both bands."""
        ok, note = self.main.ratio_ok, self.main.ratio_note
        self.rb_auto.setEnabled(ok)
        if ok:
            self.lbl_auto.setText("✓ Auto available: " + note + ".")
            self.lbl_auto.setStyleSheet("color:#1b5e20; font-size:11px;")
        else:
            if self.rb_auto.isChecked():
                self.rb_manual.setChecked(True)
            self.lbl_auto.setText("✗ Auto not applicable for this file: %s. "
                                  "Only Manual is possible." % note)
            self.lbl_auto.setStyleSheet("color:#b71c1c; font-size:11px;")

    def update_map_status(self):
        lb = self.main.layer_by_spec
        if lb is None or not (lb > 0).any():
            self.lbl_map.setText("No layer map yet.")
            self.lbl_map.setStyleSheet("color:#666; font-size:11px;")
            return
        parts = "  ".join("%dL: %d" % (n, int((lb == n).sum()))
                          for n in sorted(set(lb[lb > 0].tolist())))
        self.lbl_map.setText("Layer map: %s   (unassigned %d)"
                             % (parts, int((lb == 0).sum())))
        self.lbl_map.setStyleSheet("color:#1b5e20; font-size:11px;")

    def show_layermap(self):
        if self.main.mapfile is None or not self.main.fits:
            QtWidgets.QMessageBox.information(
                self, APP_NAME, "Load a Raman map and press 'Fit 2D band on all pixels' first.")
            return
        dlg = getattr(self, "_regions_dlg", None)
        if dlg is not None and dlg.isVisible():
            dlg.raise_()
            dlg.activateWindow()
            return
        self._regions_dlg = LayerRegionsDialog(self.main, self)
        self._regions_dlg.show()

    def close_regions_dialog(self):
        dlg = getattr(self, "_regions_dlg", None)
        if dlg is not None:
            dlg.close()
            self._regions_dlg = None

    def show_thickness(self):
        if self.main.mapfile is None:
            QtWidgets.QMessageBox.information(
                self, APP_NAME, "Load a Raman map and press 'Fit 2D band on all pixels' first.")
            return
        ThicknessDialog(self.main, self).exec()

    # -- the classification result, held by the service ---------------------------------
    def _result(self, key):
        c = self.main.service.classification
        return None if c is None else c[key]

    @property
    def layers(self):
        return self._result("layers")

    @layers.setter
    def layers(self, value):
        if value is None:                   # a new map / placement: the result no longer applies
            self.main.service.classification = None

    @property
    def calls(self):
        return self._result("calls")

    @property
    def ncands(self):
        return self._result("ncands")

    @property
    def reasons(self):
        return self._result("reasons")

    def run_classification(self):
        self._pull_config()
        try:
            notes = self.main.service.classify()
        except ValueError as exc:
            QtWidgets.QMessageBox.information(self, APP_NAME, str(exc))
            return
        for note in notes:
            QtWidgets.QMessageBox.information(self, APP_NAME, note)
        if self.main.cfg["layer_mode"] == "manual":
            self.rb_manual.setChecked(True)
        self.refresh_map()
        self._report()
        ts = getattr(self.main, "tab_spec", None)
        if ts is not None:
            ts._rebuild_list()           # names carry the class label
            ts.refresh_mini()

    def _report(self):
        self.stats.setPlainText(self.main.service.classification_report())

    def show_histograms(self):
        """Show where the thresholds actually fall relative to the data.

        Thresholds copied from a paper describe that paper's samples. This is
        how you check whether they describe yours: a bimodal FWHM histogram
        with the boundary sitting in the valley means the cut is doing real
        work; a single broad hump means it is slicing one population in half.
        """
        if self.main.fwhm_map is None:
            QtWidgets.QMessageBox.information(
                self, APP_NAME, "Load a Raman map and press 'Fit 2D band on all pixels' first.")
            return
        self._pull_config()
        fw, am, shown = self.main.service.histogram_data()
        if fw.size == 0:
            QtWidgets.QMessageBox.information(self, APP_NAME, "No successful fits.")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Threshold histograms")
        dlg.resize(900, 460)
        lay = QtWidgets.QVBoxLayout(dlg)
        cv = Canvas(figsize=(9, 4))
        cv.figure.clear()
        views.draw_threshold_histograms(cv.figure, fw, am, self.main.cfg, shown)

        lay.addWidget(cv)
        msg = QtWidgets.QLabel(
            "Widen each ± until the shaded windows cover the humps your data "
            "actually has, and no further. Windows that between them cover the "
            "whole axis classify everything and test nothing; windows that "
            "cover neither hump mean the reference values do not describe this "
            "sample. A window edge cutting through the middle of a hump is "
            "splitting one population in two, and that map will look decisive "
            "without meaning much.")
        msg.setWordWrap(True)
        msg.setStyleSheet("color:#555; font-size:11px;")
        lay.addWidget(msg)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        bb.accepted.connect(dlg.accept)
        lay.addWidget(bb)
        cv.draw()
        dlg.exec()

    # -- drawing ------------------------------------------------------------
    def _placeholder(self):
        self.canvas.clear()
        draw_placeholder(self.canvas.ax, "No classification yet")
        self.canvas.draw()

    def refresh_map(self):
        mf = self.main.mapfile
        if mf is None or self.layers is None:
            self._placeholder()
            return
        cv = self.canvas
        cv.clear()
        views.draw_class_map(cv.ax, mf, self.layers, self.calls, self.main.cfg,
                             self.main.optical, self.main.optical_image,
                             self.chk_optical.isChecked(), self.chk_grid.isChecked())
        cv.draw()

    # -- export -------------------------------------------------------------
    def export_png(self):
        if self.layers is None:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Nothing to export yet.")
            return
        path = export_figure(self.canvas.figure, self,
                             self.main.suggest_name("_classified.pdf"))
        if path:
            self.main.status("Saved %s" % path)

    def export_csv(self):
        if self.layers is None:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Nothing to export yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save classification results",
            self.main.suggest_name("_classification.csv"), "CSV file (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        self.main.service.export_classification_csv(path)
        self.main.status("Saved %s" % path)
