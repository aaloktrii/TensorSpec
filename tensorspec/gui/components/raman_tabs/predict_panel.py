"""Tab 5 · Layer prediction: 1L/2L reference values (Lorentzian fits of the reference datasets,
mean ± SD) and layer-number / stacking predictions for the loaded map or a folder. Map predictions
are written to /analysis/raman_layer_prediction.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Tab 5 - Layer prediction); calls go through RamanEngineRouter
(main.engine), figures through plotting/raman_figures.py."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

from tensorspec.core.io.raman_loaders import short_label
from tensorspec.core.raman.constants import APP_NAME
from tensorspec.core.raman.reference import DEFAULT_REF
from tensorspec.gui.components.raman_tabs.layout import Canvas
from tensorspec.plotting import raman_figures
from tensorspec.plotting.raman_figures import REF_COLORS, draw_reference, orient_axes

#: The reference / prediction modules were optional next to the original script; they are part
#: of tensorspec (core/raman/reference.py, core/raman/batch.py) now.
HAVE_GREF = True

if TYPE_CHECKING:
    from tensorspec.gui.components.raman_panel import RamanPanel

REF_ROWS = [("2D_fwhm", "2D FWHM (cm⁻¹)"), ("I2D_IG", "I(2D)/I(G)"),
            ("A2D_AG", "A(2D)/A(G)"), ("2D_pos", "2D position (cm⁻¹)"),
            ("2D_height", "2D intensity"), ("G_pos", "G position (cm⁻¹)"),
            ("G_fwhm", "G FWHM (cm⁻¹)"), ("G_height", "G intensity"),
            ("Gminus_pos", "G⁻ position (cm⁻¹)"), ("Gplus_pos", "G⁺ position (cm⁻¹)")]


class PredictTab(QtWidgets.QWidget):
    """Reference values (Lorentzian fits of the 1L and 2L datasets, mean ±
    SD) and layer-number / stacking predictions for new data."""

    def __init__(self, main: "RamanPanel"):
        super().__init__()
        self.main = main
        self.ref = None
        self.ref_path = None
        self.preds: Optional[List[dict]] = None
        self.pred_title = ""
        self._build()
        if HAVE_GREF and os.path.exists(DEFAULT_REF):
            self._set_ref(self.main.engine.load_reference(DEFAULT_REF), DEFAULT_REF)

    def _build(self):
        root = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        r = QtWidgets.QHBoxLayout()
        for text, fn in (("Build reference…", self.build_ref),
                         ("Load reference…", self.load_ref)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            b.setEnabled(HAVE_GREF)
            r.addWidget(b)
        left.addLayout(r)
        self.lbl_ref = QtWidgets.QLabel("No reference loaded")
        self.lbl_ref.setWordWrap(True)
        self.lbl_ref.setStyleSheet("color:#555; font-size:11px;")
        left.addWidget(self.lbl_ref)
        self.tbl_ref = QtWidgets.QTableWidget(0, 0)
        self.tbl_ref.verticalHeader().setDefaultSectionSize(22)
        self.tbl_ref.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        left.addWidget(self.tbl_ref, 2)
        r = QtWidgets.QHBoxLayout()
        self.btn_map = QtWidgets.QPushButton("Predict loaded map")
        self.btn_map.clicked.connect(self.predict_map)
        self.btn_dir = QtWidgets.QPushButton("Predict folder…")
        self.btn_dir.clicked.connect(self.predict_dir)
        for b in (self.btn_map, self.btn_dir):
            b.setEnabled(False)
            r.addWidget(b)
        left.addLayout(r)
        self.tbl_pred = QtWidgets.QTableWidget(0, 2)
        self.tbl_pred.setHorizontalHeaderLabels(["Prediction", "Spectra"])
        self.tbl_pred.verticalHeader().setVisible(False)
        self.tbl_pred.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tbl_pred.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        left.addWidget(self.tbl_pred, 1)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMinimumWidth(600)
        lw.setMaximumWidth(640)
        root.addWidget(lw)
        self.plot = Canvas(figsize=(8, 5))
        root.addWidget(self.plot, 1)
        self.redraw()

    # -- reference --------------------------------------------------------------
    def _set_ref(self, ref, path):
        self.ref = ref
        self.ref_path = path
        ds = ref["datasets"]
        self.lbl_ref.setText("Reference %s (built %s). Predictions use the 'matching' "
                             "spots: fitted 2D FWHM %s." % (
                                 os.path.basename(path), ref.get("created", "?"),
                                 ", ".join("%s %g–%g" % (k, *v)
                                           for k, v in ref["match_rule"].items())))
        cols = []
        for d in ds:
            for which in ("all", "matching"):
                cols.append((d, which))
        self.tbl_ref.clear()
        self.tbl_ref.setColumnCount(len(cols))
        self.tbl_ref.setRowCount(len(REF_ROWS) + 1)
        self.tbl_ref.setHorizontalHeaderLabels(
            ["%s %s\n%s" % (d["label"], d["name"].split("_")[0], w) for d, w in cols])
        self.tbl_ref.setVerticalHeaderLabels(["spectra"] + [lab for _, lab in REF_ROWS])
        for j, (d, w) in enumerate(cols):
            n = d["n_fitted"] if w == "all" else d["n_matching"]
            self.tbl_ref.setItem(0, j, QtWidgets.QTableWidgetItem("%d" % n))
            for i, (q, _) in enumerate(REF_ROWS, start=1):
                s = d[w].get(q, {})
                txt = "" if s.get("mean") is None else (
                    "%.4g ± %.2g" % (s["mean"], s["std"] or 0))
                self.tbl_ref.setItem(i, j, QtWidgets.QTableWidgetItem(txt))
        self.tbl_ref.resizeColumnsToContents()
        for b in (self.btn_map, self.btn_dir):
            b.setEnabled(True)
        self.redraw()

    def build_ref(self):
        f1 = QtWidgets.QFileDialog.getExistingDirectory(
            self, "1L reference dataset (folder of spectra, e.g. the ML map)")
        if not f1:
            return
        f2 = QtWidgets.QFileDialog.getExistingDirectory(
            self, "2L reference dataset (folder of spectra, e.g. the BL map)")
        if not f2:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            ref = self.main.engine.build_reference([("1L", f1), ("2L", f2)], DEFAULT_REF,
                                                 figures=raman_figures)
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, APP_NAME, "Could not build:\n\n%s" % exc)
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self._set_ref(ref, DEFAULT_REF)
        self.main.status("Reference saved to %s" % os.path.dirname(DEFAULT_REF))

    def load_ref(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load reference", os.path.dirname(DEFAULT_REF), "JSON (*.json)")
        if path:
            self._set_ref(self.main.engine.load_reference(path), path)

    # -- prediction -------------------------------------------------------------
    def predict_map(self):
        mf = self.main.mapfile
        if mf is None:
            self.main.status("Load a map first.")
            return
        prog = QtWidgets.QProgressDialog("Fitting G and 2D…", "Cancel", 0, mf.n_points, self)
        prog.setWindowModality(QtCore.Qt.WindowModal)
        rows = []
        for i in range(mf.n_points):
            if prog.wasCanceled():
                return
            prog.setValue(i)
            r, _, _ = self.main.engine.fit_band("g_2d", mf.axis, mf.data[i].astype(float))
            r.update(file="spot %d" % i, x=float(mf.x[i]), y=float(mf.y[i]), idx=i)
            rows.append(r)
        prog.setValue(mf.n_points)
        p = mf.path.rstrip(os.sep)
        out = os.path.join(p, "output") if os.path.isdir(p) else \
            os.path.splitext(p)[0] + "_output"
        preds = self.main.engine.write_predictions(rows, self.ref, out, short_label(p),
                                                   figures=raman_figures)
        self.main.service.write_table("raman_layer_prediction", preds, "2D_fwhm",
                                      {"reference": self.ref_path or ""})
        self._show(preds, short_label(p), out)

    def predict_dir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Dataset to predict")
        if not folder:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            preds = self.main.engine.predict_folder(folder, self.ref, figures=raman_figures)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._show(preds, short_label(folder), os.path.join(folder, "output"))

    def _show(self, preds, title, out):
        self.preds, self.pred_title = preds, title
        counts: Dict[str, int] = {}
        for p in preds:
            key = p["layer"] + ("  " + p["polytype"] if p["polytype"] not in ("", "—") else "")
            counts[key] = counts.get(key, 0) + 1
        self.tbl_pred.setRowCount(len(counts))
        for i, (k, n) in enumerate(sorted(counts.items(), key=lambda kv: -kv[1])):
            self.tbl_pred.setItem(i, 0, QtWidgets.QTableWidgetItem(k))
            self.tbl_pred.setItem(i, 1, QtWidgets.QTableWidgetItem("%d" % n))
        self.redraw()
        self.main.status("Predictions saved to %s" % os.path.join(out, "predictions.csv"))

    # -- figure -----------------------------------------------------------------
    def redraw(self):
        fig = self.plot.figure
        fig.clear()
        if not HAVE_GREF or self.ref is None:
            fig.text(0.5, 0.5, "No reference" if HAVE_GREF else
                     "raman_peak_fit.py / graphene_reference.py not found",
                     ha="center", va="center", color="#999")
            self.plot.draw()
            return
        preds = self.preds or []
        has_xy = sum(np.isfinite(p["x"]) for p in preds) > 3
        axs = np.atleast_1d(fig.subplots(1, 2 if has_xy else 1))
        ax = axs[0]
        draw_reference(ax, self.ref, matching_only=not preds)
        for lab, c in REF_COLORS.items():
            if lab == "no fit":                 # failed fits carry no usable values
                continue
            sel = [p for p in preds if p["layer"] == lab and np.isfinite(p["2D_fwhm"])]
            if sel:
                ax.scatter([p["2D_fwhm"] for p in sel], [p["I2D_IG"] for p in sel], s=18,
                           marker="x", lw=0.8, color="k" if lab in ("1L", "2L") else c,
                           label="%s: %s (%d)" % (self.pred_title, lab, len(sel)))
        ax.legend(fontsize=7, frameon=False)
        if has_xy:
            ax = axs[1]
            for lab, c in REF_COLORS.items():
                sel = [p for p in preds if p["layer"] == lab]
                if sel:
                    ax.scatter([p["x"] for p in sel], [p["y"] for p in sel], s=40,
                               marker="s", color=c, lw=0, label="%s (%d)" % (lab, len(sel)))
            ax.set_aspect("equal")
            ax.set_xlabel("X  [µm]")
            ax.set_ylabel("Y  [µm]")
            ax.set_title(self.pred_title, fontsize=9)
            ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
            orient_axes(ax, self.main.cfg)
        self.plot.draw()
