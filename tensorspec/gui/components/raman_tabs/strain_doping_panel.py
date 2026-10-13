"""3 · Analysis ▸ Strain / doping: omega_G - omega_2D correlation (Lee et al. 2012 decomposition) of the
spectra loaded in 1 · Spectra ▸ Peak fitting, with editable grid parameters. Results for the loaded
map are written to /analysis/raman_strain_doping.

Was the "ω_G–ω_2D plot…" dialog of the Peak fitting tab (Raman_Classifier ``raman_map_classifier_8 (1).py``,
Tab 6); the button there now opens this tab. Figures are drawn by
plotting/backends/raman_views/peak_fit.py."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from PySide6 import QtCore, QtWidgets

from tensorspec.core.raman import strain_doping
from tensorspec.core.raman.constants import APP_NAME
from tensorspec.plotting.backends.raman_matplotlib_engine import draw_figure_message
from tensorspec.plotting.backends.raman_views import peak_fit as views
from tensorspec.plotting.viewers.raman_canvas import Canvas

if TYPE_CHECKING:
    from tensorspec.gui.components.raman_panel import RamanPanel

PARAM_ROWS = (("G0", "ω_G0 (cm⁻¹)"), ("D0", "ω_2D0 (cm⁻¹)"),
              ("a_G", "∂ω_G/∂ε (cm⁻¹/%)"), ("s_strain", "strain slope"),
              ("b_G", "∂ω_G/∂n (cm⁻¹/10¹³)"), ("s_doping", "doping slope"))


class StrainDopingTab(QtWidgets.QWidget):

    def __init__(self, main: "RamanPanel"):
        super().__init__()
        self.main = main
        self.rows: Optional[List[dict]] = None
        root = QtWidgets.QHBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.eds = {}
        for k, lab in PARAM_ROWS:
            self.eds[k] = QtWidgets.QLineEdit("%g" % strain_doping.PARAMS[k])
            form.addRow(lab, self.eds[k])
        self.btn_run = QtWidgets.QPushButton("Analyse spectra from Peak fitting")
        self.btn_run.clicked.connect(self.analyse)
        form.addRow(self.btn_run)
        btn = QtWidgets.QPushButton("Redraw")
        btn.clicked.connect(self.redraw)
        form.addRow(btn)
        fw = QtWidgets.QWidget()
        fw.setLayout(form)
        fw.setMaximumWidth(320)
        root.addWidget(fw)
        self.cv = Canvas(figsize=(5, 5))
        root.addWidget(self.cv, 1)
        self.cv.figure.clear()
        draw_figure_message(self.cv.figure, "Load spectra in 1 · Spectra > Peak fitting, then Analyse")
        self.cv.draw()

    def params(self) -> Optional[dict]:
        try:
            return {k: float(e.text()) for k, e in self.eds.items()}
        except ValueError:
            return None

    def analyse(self):
        items = self.main.tab_fit.items
        if not items:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Add spectra in "
                                              "1 · Spectra ▸ Peak fitting first.")
            return
        ref = self.main.tab_pred.ref
        if ref is None:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Build or load a 1L/2L "
                                              "reference in 3 · Analysis ▸ Layer prediction first.")
            return
        prog = QtWidgets.QProgressDialog("Analysing…", "Cancel", 0, len(items), self)
        prog.setWindowModality(QtCore.Qt.WindowModal)

        def tick(k, n):
            prog.setValue(k)
            QtWidgets.QApplication.processEvents()
            return not prog.wasCanceled()
        rows = self.main.engine.strain_doping_many([(it["wn"], it["y"]) for it in items],
                                                   ref, progress=tick)
        prog.setValue(len(items))
        if rows is None:
            return
        self.rows = rows
        self.redraw()

    def redraw(self):
        p = self.params()
        if p is None or self.rows is None:
            return
        self.cv.figure.clear()
        views.draw_correlation_figure(self.cv.figure, self.rows, p)
        self.cv.draw()
        if self.main.tab_fit._from_map():
            self.main.service.write_strain_doping(self.rows, p)
