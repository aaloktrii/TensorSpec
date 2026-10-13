"""Classification helper dialog: thickness from I(2D)/I(G) (maps, histogram, editable cutoffs).
Figures are drawn by plotting/backends/raman_views/classification.py.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Tab 2 - Classification: ThicknessDialog)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtWidgets

from tensorspec.core.raman.classification import LAYER_THICK
from tensorspec.core.raman.constants import LAYERS
from tensorspec.gui.components.raman_tabs.layout import fit_to_screen
from tensorspec.gui.services.raman_service import RamanService
from tensorspec.plotting.backends.raman_views import classification as views
from tensorspec.plotting.viewers.raman_canvas import Canvas

if TYPE_CHECKING:
    from tensorspec.gui.components.raman_panel import RamanPanel

class ThicknessDialog(QtWidgets.QDialog):
    """Thickness from I(2D)/I(G): maps, histogram, editable cutoffs.

    Nothing here touches the main config until 'Use as Auto mode' is pressed,
    so the cutoffs can be explored freely.
    """

    LAYER_ORDER = RamanService.THICKNESS_LAYERS

    def __init__(self, main: "RamanPanel", parent=None):
        super().__init__(parent)
        self.main = main
        self.setWindowTitle("Thickness from I(2D)/I(G)")
        fit_to_screen(self, 1180, 660)
        lay = QtWidgets.QVBoxLayout(self)

        if not main.ratio_ok or main.ratio_maps.get("height") is None:
            why = main.ratio_note if not main.ratio_ok else \
                "the map has not finished fitting yet"
            msg = QtWidgets.QLabel(
                "<b>Not applicable for this file.</b><br><br>"
                "Thickness from I(2D)/I(G) needs both the G band (~1580 cm⁻¹) "
                "and the 2D band (~2700 cm⁻¹) in the same spectrum — here %s."
                "<br><br>Auto mode is unavailable; only <b>Manual</b> layer "
                "count is possible. A wide 600 gr/mm acquisition of the same "
                "area would give you the ratio." % why)
            msg.setWordWrap(True)
            lay.addWidget(msg)
            bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            bb.rejected.connect(self.reject)
            lay.addWidget(bb)
            self.resize(520, 220)
            return

        cfg = main.cfg
        form = QtWidgets.QHBoxLayout()
        form.addWidget(QtWidgets.QLabel("Intensity"))
        self.cmb_metric = QtWidgets.QComboBox()
        self.cmb_metric.addItems(["height", "area"])
        self.cmb_metric.setCurrentText(cfg.get("ratio_metric", "height"))
        self.cmb_metric.setToolTip(
            "height — fitted peak heights, what most layer-counting papers "
            "quote and what the default cutoffs assume\n"
            "area — integrated areas; unaffected by instrument broadening, "
            "but the cutoffs must then be recalibrated")
        form.addWidget(self.cmb_metric)
        form.addSpacing(14)
        self.spins = {}
        for key, label in (("1_2", "1L | 2L"), ("2_3", "2L | 3L"),
                           ("3_thick", "3L | ≥4L")):
            form.addWidget(QtWidgets.QLabel(label))
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(0.01, 20.0)
            sp.setDecimals(2)
            sp.setSingleStep(0.05)
            sp.setValue(float(cfg["ratio_cut"][key]))
            sp.setFixedWidth(76)
            form.addWidget(sp)
            self.spins[key] = sp
        form.addSpacing(14)
        form.addWidget(QtWidgets.QLabel("≥4L pixels:"))
        self.cmb_thick = QtWidgets.QComboBox()
        self.cmb_thick.addItems(["no stacking call", "assume N ="])
        self.cmb_thick.setCurrentIndex(1 if cfg.get("thick_policy") == "assume" else 0)
        form.addWidget(self.cmb_thick)
        self.spn_n = QtWidgets.QSpinBox()
        self.spn_n.setRange(min(LAYERS), max(LAYERS))
        self.spn_n.setValue(int(cfg.get("thick_assume", 5)))
        self.spn_n.setSuffix(" L")
        form.addWidget(self.spn_n)
        form.addStretch(1)
        lay.addLayout(form)

        self.canvas = Canvas(figsize=(11.5, 4.6))
        lay.addWidget(self.canvas, 1)

        self.lbl = QtWidgets.QLabel()
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("font-family:monospace; font-size:11px;")
        lay.addWidget(self.lbl)

        caveat = QtWidgets.QLabel(
            "The default cutoffs are placeholders taken from typical literature "
            "values (1L ~4, 2L ~2, 3L ~1.1, thick ~0.3), and published values "
            "for 2L vary between roughly 1 and 2. The ratio also moves with "
            "doping, strain, oxide thickness and laser energy, and at "
            "14.7 cm⁻¹ resolution the narrow G band is broadened more than "
            "the 2D, which inflates height ratios. Calibrate the cutoffs on "
            "this instrument against a region of known thickness (AFM, "
            "or a clear monolayer) before trusting the layer counts.")
        caveat.setWordWrap(True)
        caveat.setStyleSheet("color:#8a6d00; font-size:11px;")
        lay.addWidget(caveat)

        bb = QtWidgets.QDialogButtonBox()
        self.btn_use = bb.addButton("Use as Auto mode", QtWidgets.QDialogButtonBox.AcceptRole)
        bb.addButton(QtWidgets.QDialogButtonBox.Close)
        bb.accepted.connect(self._use)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        for w in list(self.spins.values()) + [self.spn_n]:
            w.valueChanged.connect(self.redraw)
        self.cmb_metric.currentIndexChanged.connect(self.redraw)
        self.cmb_thick.currentIndexChanged.connect(self.redraw)
        self.redraw()

    # -- current, uncommitted settings --------------------------------------
    def _local_cfg(self) -> dict:
        c = dict(self.main.cfg)
        c["ratio_cut"] = {k: sp.value() for k, sp in self.spins.items()}
        c["ratio_metric"] = self.cmb_metric.currentText()
        c["thick_policy"] = "assume" if self.cmb_thick.currentIndex() else "leave"
        c["thick_assume"] = int(self.spn_n.value())
        return c

    def _valid(self, c: dict) -> bool:
        return self.main.service.thickness_valid(c)

    def redraw(self):
        c = self._local_cfg()
        self.spn_n.setEnabled(c["thick_policy"] == "assume")
        valid = self._valid(c)
        self.btn_use.setEnabled(valid)
        mf = self.main.mapfile
        pv = self.main.service.thickness_preview(c)
        rm, lm, counts, with_ratio = pv["ratios"], pv["layer_index"], pv["counts"], pv["with_ratio"]

        fig = self.canvas.figure
        fig.clear()
        fin = rm[np.isfinite(rm)]
        views.draw_thickness(fig, mf, rm, lm, counts, c, self.main.cfg, self.LAYER_ORDER, fin,
                             self.main.optical, self.main.optical_image)
        self.canvas.draw()

        if not valid:
            self.lbl.setText("Cutoffs must decrease: 1L|2L > 2L|3L > 3L|≥4L > 0.")
            self.lbl.setStyleSheet("color:#b71c1c; font-family:monospace; font-size:11px;")
            return
        self.lbl.setStyleSheet("font-family:monospace; font-size:11px;")
        med = float(np.median(fin)) if fin.size else float("nan")
        self.lbl.setText(
            "Pixels with both bands fitted: %d of %d   median ratio %.2f\n"
            "1L %d   2L %d   3L %d   ≥4L %d"
            % (with_ratio, mf.n_points, med, counts[1], counts[2],
               counts[3], counts[LAYER_THICK]))

    def _use(self):
        c = self._local_cfg()
        if not self._valid(c):
            return
        self.main.service.use_thickness(c)
        self.main.tab_class.rb_auto.setChecked(True)
        self.accept()
        self.main.tab_class.run_classification()
