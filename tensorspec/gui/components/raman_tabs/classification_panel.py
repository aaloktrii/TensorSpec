"""Tab 2 · Classification: layer number (manual / auto from I(2D)/I(G) / layer map) and ABA/ABC
stacking per pixel, editable threshold table, histograms, PNG / CSV export. Results are written to
/analysis/raman_classification.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Tab 2 - Classification: ClassificationTab)."""
from __future__ import annotations

import csv
from typing import TYPE_CHECKING
from typing import Dict, Optional

import numpy as np
from matplotlib.patches import Patch
from PySide6 import QtCore, QtWidgets

from tensorspec.core.raman.classification import (LAYER_THICK, classify_map, label_of, pixel_ratio,
                                                  terrace_edges, windows_for)
from tensorspec.core.raman.constants import (APP_NAME, BOUHAFS_FWHM, DEFAULT_COLORS, DEFAULT_TOL,
                                             LAYERS)
from tensorspec.gui.components.raman_tabs.layer_regions_dialog import LayerRegionsDialog
from tensorspec.gui.components.raman_tabs.layout import ThresholdTable, _fmt
from tensorspec.gui.components.raman_tabs.map_explorer_panel import MapExplorerTab
from tensorspec.gui.components.raman_tabs.thickness_dialog import ThicknessDialog
from tensorspec.plotting.backends.raman_matplotlib_engine import class_rgba, frame_axes, na_color, orient_axes
from tensorspec.plotting.viewers.raman_canvas import Canvas, export_figure

if TYPE_CHECKING:
    from tensorspec.gui.components.raman_panel import RamanPanel

class ClassificationTab(QtWidgets.QWidget):

    def __init__(self, main: "RamanPanel"):
        super().__init__()
        self.main = main
        self.layers: Optional[np.ndarray] = None
        self.calls: Optional[np.ndarray] = None
        self.ncands: Optional[np.ndarray] = None
        self.reasons: Optional[np.ndarray] = None
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
                self, APP_NAME, "Load and fit a Raman map first (tab 1).")
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
                self, APP_NAME, "Load a Raman map first (tab 1).")
            return
        ThicknessDialog(self.main, self).exec()

    def run_classification(self):
        if self.main.fwhm_map is None:
            QtWidgets.QMessageBox.information(
                self, APP_NAME, "Load a Raman map first (tab 1).")
            return
        self._pull_config()
        cfg = self.main.cfg
        if cfg["layer_mode"] == "auto" and not self.main.ratio_ok:
            QtWidgets.QMessageBox.information(
                self, APP_NAME,
                "Auto mode is not applicable for this file: %s.\n\nUsing "
                "Manual." % self.main.ratio_note)
            cfg["layer_mode"] = "manual"
            self.rb_manual.setChecked(True)
        asym = self.main.asym_maps[cfg["asym_metric"]]
        if cfg["layer_mode"] == "map":
            lb = self.main.layer_by_spec
            if lb is None or not (lb > 0).any():
                QtWidgets.QMessageBox.information(
                    self, APP_NAME, "Layer-map mode needs a layer map — use "
                    "'Define layer regions…' first.")
                return
        ratio = self.main.ratio_maps.get(cfg.get("ratio_metric", "height"))
        self.layers, self.calls, self.ncands, self.reasons = classify_map(
            self.main.fwhm_map, asym, cfg, ratio, self.main.layer_grid())
        self.main.service.write_classification(self.layers, self.calls, self.ncands)
        self.refresh_map()
        self._report()
        ts = getattr(self.main, "tab_spec", None)
        if ts is not None:
            ts._rebuild_list()           # names carry the class label
            ts.refresh_mini()

    def _report(self):
        labels: Dict[str, int] = {}
        ny, nx = self.layers.shape
        for r in range(ny):
            for c in range(nx):
                lab = label_of(int(self.layers[r, c]), self.calls[r, c])
                labels[lab] = labels.get(lab, 0) + 1
        total = ny * nx
        cfg = self.main.cfg
        fitted = int(np.isfinite(self.main.fwhm_map).sum())
        lines = ["Pixels: %d   (%d × %d)   fitted: %d" % (total, nx, ny, fitted),
                 "Mode: %s   technique: %s   metric: %s   asym ≥ %.3f"
                 % (cfg["layer_mode"], cfg["technique"],
                    cfg["asym_metric"], cfg["asym_threshold"])]
        if fitted:
            errs = np.array([r.get("fwhm_err", np.nan) for r in self.main.fits
                             if r.get("ok")], dtype=float)
            errs = errs[np.isfinite(errs)]
            if errs.size:
                med_err = float(np.median(errs))
                gaps = [abs(cfg["fwhm"][L]["ABC"] - cfg["fwhm"][L]["ABA"]) / 2.0
                        for L in cfg["fwhm"]]
                lines.append("Median FWHM uncertainty: ± %.1f cm⁻¹  "
                             "(threshold-to-boundary gap ≈ %.0f cm⁻¹)%s"
                             % (med_err, float(np.median(gaps)),
                                "   ← too large to separate ABA from ABC"
                                if med_err > 0.5 * float(np.median(gaps)) else ""))
            out = 0
            for r in range(ny):
                for c in range(nx):
                    why = self.reasons[r, c] or ""
                    if self.calls[r, c] is None and "outside both" in why:
                        out += 1
            if out:
                lines.append("Outside every tolerance window: %d of %d fitted "
                             "pixels (%.1f %%)" % (out, fitted, 100.0 * out / fitted))
            if cfg["layer_mode"] == "map":
                g = self.main.layer_grid()
                fitted_mask = np.isfinite(self.main.fwhm_map)
                per = "  ".join("%dL %d" % (n, int(((g == n) & fitted_mask).sum()))
                                for n in sorted(set(g[g > 0].tolist())))
                lines.append("Layer map: fitted pixels per region — %s   "
                             "(no N assigned: %d)"
                             % (per, int(((g == 0) & fitted_mask).sum())))
                if cfg.get("exclude_edges", True):
                    e = int((terrace_edges(g) & fitted_mask).sum())
                    lines.append("Terrace-edge pixels excluded: %d" % e)
            elif cfg["layer_mode"] == "manual":
                amb = int(((self.ncands > 1) & np.isfinite(self.main.fwhm_map)).sum())
                lines.append("Ambiguous layer assignment: %d of %d fitted pixels "
                             "(%.1f %%) matched more than one layer window"
                             % (amb, fitted, 100.0 * amb / fitted))
            else:
                k = cfg["ratio_cut"]
                rm = self.main.ratio_maps.get(cfg.get("ratio_metric", "height"))
                nr = int(np.isfinite(rm).sum()) if rm is not None else 0
                lines.append("Thickness from I(2D)/I(G) (%s), cutoffs %.2f / %.2f / "
                             "%.2f: ratio available for %d of %d fitted pixels"
                             % (cfg.get("ratio_metric", "height"), k["1_2"],
                                k["2_3"], k["3_thick"], nr, fitted))
                if cfg.get("thick_policy") == "assume":
                    lines.append("≥4L pixels assumed to be %dL for the stacking call"
                                 % int(cfg.get("thick_assume", 5)))
            res = self.main.mapfile.spectral_resolution if self.main.mapfile else None
            if res and res > 8.0 and cfg["technique"] != "asym":
                lines.append("⚠ Spectral resolution %.1f cm⁻¹: the 2D widths are "
                             "instrument-broadened, so ABA/ABC calls against the "
                             "Bouhafs table are unreliable. Use this map for "
                             "thickness and a 1800 gr/mm map for stacking." % res)
        rejected = total - fitted
        if rejected:
            lines.append("")
            lines.append("Rejected before classification (%d pixels):" % rejected)
            for why, n in sorted((self.main.fit_tally or {}).items(),
                                 key=lambda kv: -kv[1]):
                lines.append("   %5d   %s" % (n, why))
        lines.append("")
        for lab in sorted(labels, key=lambda s: (s == "Unclassified", s)):
            n = labels[lab]
            lines.append("%-16s %6d   %5.1f %%" % (lab, n, 100.0 * n / total))
        self.stats.setPlainText("\n".join(lines))

    def show_histograms(self):
        """Show where the thresholds actually fall relative to the data.

        Thresholds copied from a paper describe that paper's samples. This is
        how you check whether they describe yours: a bimodal FWHM histogram
        with the boundary sitting in the valley means the cut is doing real
        work; a single broad hump means it is slicing one population in half.
        """
        if self.main.fwhm_map is None:
            QtWidgets.QMessageBox.information(
                self, APP_NAME, "Load and fit a map first (tab 1).")
            return
        self._pull_config()
        cfg = self.main.cfg
        fw = self.main.fwhm_map[np.isfinite(self.main.fwhm_map)]
        am = self.main.asym_maps[cfg["asym_metric"]]
        am = am[np.isfinite(am)]
        if fw.size == 0:
            QtWidgets.QMessageBox.information(self, APP_NAME, "No successful fits.")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Threshold histograms")
        dlg.resize(900, 460)
        lay = QtWidgets.QVBoxLayout(dlg)
        cv = Canvas(figsize=(9, 4))
        cv.figure.clear()
        ax1 = cv.figure.add_subplot(121)
        ax2 = cv.figure.add_subplot(122)

        ax1.hist(fw, bins=max(10, min(60, fw.size // 8)),
                 color="#607d8b", edgecolor="white", linewidth=0.4)
        if cfg["layer_mode"] == "manual":
            shown = [int(cfg["manual_layer"])]
        elif cfg["layer_mode"] == "map" and self.main.layer_by_spec is not None:
            shown = sorted(n for n in set(self.main.layer_by_spec.tolist())
                           if n in cfg["fwhm"]) or sorted(cfg["fwhm"])
        else:
            shown = sorted(cfg["fwhm"])
        top = ax1.get_ylim()[1]
        for layer in shown:
            (a0, a1), (c0, c1) = windows_for(layer, cfg)
            for (lo, hi), col, key in (((a0, a1), "#1e88e5", "ABA"),
                                       ((c0, c1), "#e53935", "ABC")):
                if hi < fw.min() - 5 or lo > fw.max() + 5:
                    continue
                ax1.axvspan(lo, hi, color=col, alpha=0.16, lw=0)
                ax1.axvline(cfg["fwhm"][layer][key], color=col, lw=1.0, ls="--")
            if len(shown) == 1:
                ax1.text(cfg["fwhm"][layer]["ABA"], top * 0.96, " ABA",
                         color="#1e88e5", fontsize=8, va="top")
                ax1.text(cfg["fwhm"][layer]["ABC"], top * 0.96, " ABC",
                         color="#e53935", fontsize=8, va="top")
        ax1.set_xlabel("FWHM(2D)  [cm$^{-1}$]")
        ax1.set_ylabel("pixels")
        ax1.set_title("Width, with the ABA (blue) and ABC (red) windows"
                      if len(shown) == 1 else
                      "Width, with every layer's ABA/ABC windows", fontsize=9)

        if am.size:
            ax2.hist(am, bins=max(10, min(60, am.size // 8)),
                     color="#8d6e63", edgecolor="white", linewidth=0.4)
            ax2.axvline(cfg["asym_threshold"], color="#d81b60", lw=1.4, ls="--",
                        label="threshold %.3f" % cfg["asym_threshold"])
            ax2.legend(fontsize=8)
        ax2.set_xlabel("asymmetry (%s)" % cfg["asym_metric"])
        ax2.set_ylabel("pixels")
        ax2.set_title("Shape asymmetry, with the ABC cut", fontsize=10)

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
        self.canvas.ax.text(0.5, 0.5, "No classification yet",
                            ha="center", va="center", color="#999")
        self.canvas.ax.set_axis_off()
        self.canvas.draw()

    def refresh_map(self):
        mf = self.main.mapfile
        if mf is None or self.layers is None:
            self._placeholder()
            return
        cv = self.canvas
        cv.clear()
        ax = cv.ax
        rgba = class_rgba(self.layers, self.calls, self.main.cfg)
        ax.imshow(rgba, origin="lower", extent=mf.extent,
                  interpolation="nearest", aspect="equal", zorder=1)

        opt = self.main.optical
        if opt is not None and self.chk_optical.isChecked() \
                and self.main.optical_image is not None:
            ax.imshow(np.ma.masked_invalid(self.main.optical_image),
                      origin="lower", extent=opt.extent, cmap="gray",
                      alpha=self.main.cfg["optical_alpha"],
                      interpolation="bilinear", aspect="equal", zorder=2)
        if self.chk_grid.isChecked():
            MapExplorerTab._draw_grid(ax, mf)

        present = set()
        ny, nx = self.layers.shape
        for r in range(ny):
            for c in range(nx):
                present.add((int(self.layers[r, c]), self.calls[r, c]))
        handles = []
        for lay, call in sorted(present, key=lambda t: (t[0] < 0, t[0], str(t[1]))):
            if call is None or (lay < 0 and call != "N/A"):
                continue
            col = (na_color(lay, self.main.cfg) if call == "N/A"
                   else self.main.cfg["colors"][lay][call])
            handles.append(Patch(facecolor=col, edgecolor="#555",
                                 label=label_of(lay, call)))
        handles.append(Patch(facecolor=self.main.cfg["unclassified_color"],
                             edgecolor="#555", label="Unclassified"))
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=9, frameon=True, title="Class")
        ax.set_xlabel("X  [µm]")
        ax.set_ylabel("Y  [µm]")
        ax.set_title("Stacking-order classification")
        ax.set_anchor("C")
        frame_axes(ax, mf, self.main.cfg,
                   self.chk_optical.isChecked() and self.main.optical is not None)
        orient_axes(ax, self.main.cfg)
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
        mf = self.main.mapfile
        ny, nx = self.layers.shape
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["row", "col", "x_um", "y_um", "fwhm_2d_cm-1",
                        "fwhm_err_cm-1", "center_cm-1", "asym_hwhm", "asym_area",
                        "asym_residual", "snr", "spikes_removed",
                        "g_height", "twod_height", "ratio_2d_g", "n_layers",
                        "stacking", "label", "n_candidate_layers",
                        "unclassified_reason", "fit_note"])
            for r in range(ny):
                for c in range(nx):
                    i = mf.index_map[r, c]
                    res = self.main.fits[i] if i >= 0 and self.main.fits else {}
                    gr = (self.main.g_fits[i] if i >= 0 and self.main.g_fits
                          else {})
                    lay = int(self.layers[r, c])
                    call = self.calls[r, c]
                    w.writerow([
                        r, c,
                        "" if i < 0 else "%.4f" % mf.x[i],
                        "" if i < 0 else "%.4f" % mf.y[i],
                        _fmt(res.get("fwhm")), _fmt(res.get("fwhm_err"), 2),
                        _fmt(res.get("center")),
                        _fmt(res.get("asym_hwhm"), 4),
                        _fmt(res.get("asym_area"), 4),
                        _fmt(res.get("asym_residual"), 4),
                        _fmt(res.get("snr"), 2), res.get("spikes", 0),
                        _fmt(gr.get("height") if gr.get("ok") else None, 2),
                        _fmt(res.get("height") if res.get("ok") else None, 2),
                        _fmt(pixel_ratio(res, gr, self.main.cfg.get("ratio_metric", "height")), 3),
                        (">=4" if lay == LAYER_THICK else ("" if lay < 0 else lay)),
                        call or "",
                        label_of(lay, call),
                        int(self.ncands[r, c]),
                        self.reasons[r, c] or "",
                        res.get("note", ""),
                    ])
        self.main.status("Saved %s" % path)
