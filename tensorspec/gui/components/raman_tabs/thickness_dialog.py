"""Classification helper dialog: thickness from I(2D)/I(G) (maps, histogram, editable cutoffs).

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Tab 2 - Classification: ThicknessDialog)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Patch
from PySide6 import QtWidgets

from tensorspec.core.raman.classification import LAYER_THICK, label_of, layer_from_ratio
from tensorspec.core.raman.constants import LAYERS
from tensorspec.gui.components.raman_tabs.layout import Canvas, fit_to_screen
from tensorspec.plotting.raman_figures import na_color, orient_axes

if TYPE_CHECKING:
    from tensorspec.gui.components.raman_panel import RamanPanel

class ThicknessDialog(QtWidgets.QDialog):
    """Thickness from I(2D)/I(G): maps, histogram, editable cutoffs.

    Nothing here touches the main config until 'Use as Auto mode' is pressed,
    so the cutoffs can be explored freely.
    """

    LAYER_ORDER = (1, 2, 3, LAYER_THICK)

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
        k = c["ratio_cut"]
        return k["1_2"] > k["2_3"] > k["3_thick"] > 0

    def redraw(self):
        c = self._local_cfg()
        self.spn_n.setEnabled(c["thick_policy"] == "assume")
        valid = self._valid(c)
        self.btn_use.setEnabled(valid)
        mf = self.main.mapfile
        rm = self.main.ratio_maps[c["ratio_metric"]]
        ny, nx = rm.shape
        lm = np.full((ny, nx), np.nan)
        counts = {k: 0 for k in self.LAYER_ORDER}
        for r in range(ny):
            for cc in range(nx):
                n = layer_from_ratio(rm[r, cc], c) if valid else None
                if n is not None:
                    lm[r, cc] = self.LAYER_ORDER.index(n)
                    counts[n] += 1
        with_ratio = int(np.isfinite(rm).sum())

        fig = self.canvas.figure
        fig.clear()
        a1 = fig.add_subplot(131)
        a2 = fig.add_subplot(132)
        a3 = fig.add_subplot(133)
        fin = rm[np.isfinite(rm)]

        opt = self.main.optical
        for ax in (a1, a2):
            if opt is not None and self.main.optical_image is not None:
                ax.imshow(np.ma.masked_invalid(self.main.optical_image),
                          origin="lower", extent=opt.extent, cmap="gray",
                          aspect="equal", zorder=0)
        im = a1.imshow(np.ma.masked_invalid(rm), origin="lower", extent=mf.extent,
                       cmap="magma", aspect="equal", interpolation="nearest",
                       alpha=0.85, zorder=1)
        fig.colorbar(im, ax=a1, fraction=0.046, pad=0.03).set_label(
            "I(2D)/I(G) (%s)" % c["ratio_metric"])
        a1.set_title("I(2D)/I(G)", fontsize=10)

        cols = [na_color(n, c) for n in self.LAYER_ORDER]
        cmap = mcolors.ListedColormap(cols)
        a2.imshow(np.ma.masked_invalid(lm), origin="lower", extent=mf.extent,
                  cmap=cmap, vmin=-0.5, vmax=3.5, aspect="equal",
                  interpolation="nearest", alpha=0.9, zorder=1)
        a2.legend(handles=[Patch(facecolor=cols[i], edgecolor="#555",
                                 label="%s  (%d)" % (label_of(n, "N/A"), counts[n]))
                           for i, n in enumerate(self.LAYER_ORDER)],
                  fontsize=8, loc="upper left", framealpha=0.9)
        a2.set_title("Thickness from the ratio", fontsize=10)
        for ax in (a1, a2):
            ax.set_xlim(mf.extent[0], mf.extent[1])
            ax.set_ylim(mf.extent[2], mf.extent[3])
            orient_axes(ax, self.main.cfg)
            ax.set_xlabel("X [µm]")
        a1.set_ylabel("Y [µm]")

        if fin.size:
            a3.hist(fin, bins=max(8, min(50, fin.size // 3)), color="#607d8b",
                    edgecolor="white", linewidth=0.4)
            for key, lab in (("1_2", "1|2"), ("2_3", "2|3"), ("3_thick", "3|≥4")):
                v = c["ratio_cut"][key]
                a3.axvline(v, color="#d81b60", ls="--", lw=1.1)
                a3.text(v, a3.get_ylim()[1] * 0.96, " " + lab, color="#d81b60",
                        fontsize=8, va="top")
        a3.set_xlabel("I(2D)/I(G)")
        a3.set_ylabel("pixels")
        a3.set_title("Distribution, with the cutoffs", fontsize=10)
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
        for k in ("ratio_cut", "ratio_metric", "thick_policy", "thick_assume"):
            self.main.cfg[k] = c[k]
        self.main.cfg["layer_mode"] = "auto"
        self.main.tab_class.rb_auto.setChecked(True)
        self.accept()
        self.main.tab_class.run_classification()
