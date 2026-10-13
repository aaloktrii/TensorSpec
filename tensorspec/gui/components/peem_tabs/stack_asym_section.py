"""PEEM XMCD-tab section: 4. Stack and 5. Asymmetry (plus one-click / PLD / PMLD).

Mixin of ``PeemXmcdPanel`` (gui/components/peem_xmcd_panel.py); it uses the panel's shared state
(service, _name, ROIs, viewer, _log / _err / _run). Moved verbatim from the former ue49_panel.py."""
from __future__ import annotations


from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


_DIVERGING = {"pcd", "pld", "pmld", "pmnld"}

from tensorspec.core.peem import dichroism
from tensorspec.gui.components.peem_tabs.common import _dspin


class StackAsymSection:
    def _build_stack_asym(self, cl):
        """Controls of this section, added to the control column ``cl``."""
        # ---------------------------------------------------------------- 3. Stack
        stk = QGroupBox("4. Stack each polarization into one image")
        sk = QFormLayout(stk)
        self.pair_combo = QComboBox()
        self.pair_combo.addItem("XMCD (CP / CM)", "xmcd")
        self.pair_combo.addItem("XLD (LH / LV)", "xld")
        sk.addRow("Polarization pair:", self.pair_combo)
        self.pair_combo.currentIndexChanged.connect(lambda *_: self._refresh_asym_outputs())
        b = QPushButton("Stack")
        b.setToolTip("Average the drift-corrected frames of each polarization, then align the two images on edges")
        b.clicked.connect(self._stack_step)
        sk.addRow(b)
        self.stack_out = QComboBox()
        self._fill_show(self.stack_out, [], "Show: (after Stack)")
        self.stack_out.setToolTip("Stacked images in the viewer (after Stack)")
        self.stack_out.currentIndexChanged.connect(self._show_stack_output)
        sk.addRow(self.stack_out)
        self.send_source = QComboBox()
        self.send_source.addItem("stacked images (per polarization / energy)", "stacked")
        self.send_source.addItem("drift-corrected frames (all, paired + separated)", "processed")
        self.btn_to_xas = QPushButton("→ Sum Rule")
        self.btn_to_xas.setToolTip("Push these images to the Sum Rule tab (ROI XAS, background, sum rule);\n"
                                   "they arrive paired and separated into the two polarizations")
        self.btn_to_xas.clicked.connect(self._send_to_xas)
        self._send_row = QWidget()
        sr = QHBoxLayout(self._send_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.addWidget(QLabel("Send:"))
        sr.addWidget(self.send_source, 1)
        sr.addWidget(self.btn_to_xas)
        self._send_row.setVisible(False)  # shown when the PEEM suite connects its Sum Rule tab
        sk.addRow(self._send_row)
        cl.addWidget(stk)

        # ---------------------------------------------------------------- 4. Asymmetry
        dich = QGroupBox("5. Asymmetry")
        dl = QFormLayout(dich)
        sign = QLabel("XMCD = (CM−CP)/(CM+CP)   ·   XLD = (LV−LH)/(LV+LH)")
        dl.addRow(sign)
        b = QPushButton("Subtract + asymmetry")
        b.setToolTip("Difference and asymmetry of the two stacked images")
        b.clicked.connect(self._asym_step)
        dl.addRow(b)
        self.asym_out = QComboBox()
        self._fill_show(self.asym_out, [], "Show: (after the asymmetry step)")
        self.asym_out.currentIndexChanged.connect(self._show_asym_output)
        dl.addRow(self.asym_out)
        adv = QGroupBox("One-click and more maps")
        adv.setCheckable(True)
        adv.setChecked(False)
        adv_body = QWidget()
        al = QFormLayout(adv_body)
        al.setContentsMargins(0, 0, 0, 0)
        self.node_combo = QComboBox()
        for label, val in (("Raw frames", "raw"), ("Processed (drift/paired)", "processed"),
                           ("Separated channels", "separated")):
            self.node_combo.addItem(label, val)
        al.addRow("Source:", self.node_combo)
        self.chk_register = QCheckBox("Drift correction (cross-correlation; box if marquee)")
        self.chk_register.setChecked(True)
        al.addRow(self.chk_register)
        row = QHBoxLayout()
        b = QPushButton("Drift-correct + XMCD (PCD)")
        b.clicked.connect(self._pcd)
        row.addWidget(b)
        b = QPushButton("PLD (one energy)")
        b.clicked.connect(self._pld)
        row.addWidget(b)
        al.addRow(row)
        self.e_a = _dspin(0, 2000, 0.0, 0.05, 2)
        self.e_b = _dspin(0, 2000, 0.0, 0.05, 2)
        self.e_hw = _dspin(0, 10, 0.1, 0.05, 2)
        al.addRow("E_A (eV):", self.e_a)
        al.addRow("E_B (eV):", self.e_b)
        al.addRow("± window (eV):", self.e_hw)
        b = QPushButton("PMLD / PMNLD")
        b.clicked.connect(self._pmld)
        al.addRow(b)
        QVBoxLayout(adv).addWidget(adv_body)
        adv_body.setVisible(False)
        adv.toggled.connect(adv_body.setVisible)
        dl.addRow(adv)
        cl.addWidget(dich)

    def _pcd(self):
        name = self._require()
        if name:
            self._run(self.service.compute_pcd, lambda r: self._show_result(r, "pcd"), name,
                      node=self.node_combo.currentData(), convention=dichroism.UNIVERSAL_CONVENTION,
                      register=self.chk_register.isChecked(), reg_roi=self._drift_roi(), crop_roi=self._crop())

    def _stack_step(self):
        name = self._require()
        if name:
            self._run(self.service.stack, self._on_stacked, name, pair=self.pair_combo.currentData())

    @staticmethod
    def _fill_show(combo: QComboBox, outputs: list[tuple[str, str]], empty: str):
        """Fill a Show list with the service's (label, variable) pairs; keep the chosen position."""
        pos = max(0, combo.currentIndex())
        combo.blockSignals(True)
        combo.clear()
        for label, var in outputs:
            combo.addItem(f"Show: {label}", var)
        if not outputs:
            combo.addItem(empty, None)
        combo.setCurrentIndex(min(pos, combo.count() - 1))
        combo.blockSignals(False)

    def set_xas_target(self, send):
        """send() hands the stacked images to the stack / BG / sum-rule tab (set by the PEEM suite)."""
        self._xas_target = send
        self._send_row.setVisible(send is not None)

    def _send_to_xas(self):
        if self._require() and getattr(self, "_xas_target", None):
            self._xas_target()

    def _on_stacked(self, res: dict):
        self._fill_show(self.stack_out, self.service.stack_outputs(self._name), "Show: (after Stack)")
        self._show_result(res, "stack")
        self._show_stack_output()

    def _show_stack_output(self, *_):
        """Show the chosen stacked image (layer in the viewer)."""
        var = self.stack_out.currentData()
        if self._name and var:
            self._select("stack", var)

    def _show_asym_output(self, *_):
        """Show the chosen output of the asymmetry step (layer in the viewer)."""
        var = self.asym_out.currentData()
        if self._name and var:
            self._select("pcd" if self.pair_combo.currentData() == "xmcd" else "pld", var)

    def _asym_step(self):
        name = self._require()
        if name:
            pair = self.pair_combo.currentData()
            self._run(self.service.asymmetry,
                      lambda r: (self._show_result(r, "pcd" if pair == "xmcd" else "pld"),
                                 self._refresh_asym_outputs(), self._show_asym_output()),
                      name, convention=dichroism.UNIVERSAL_CONVENTION, crop_roi=self._crop())

    def _refresh_asym_outputs(self, *_):
        outputs = self.service.asymmetry_outputs(self._name, self.pair_combo.currentData()) if self._name else []
        self._fill_show(self.asym_out, outputs, "Show: (after the asymmetry step)")

    def _pld(self):
        name = self._require()
        if name:
            self._run(self.service.compute_pld, lambda r: self._show_result(r, "pld"), name,
                      node=self.node_combo.currentData(), register=self.chk_register.isChecked(),
                      reg_roi=self._drift_roi(), crop_roi=self._crop())

    def _pmld(self):
        name = self._require()
        if name:
            node = self.node_combo.currentData()
            self._run(self.service.compute_pmld, lambda r: self._show_result(r, "pmld"), name,
                      self.e_a.value(), self.e_b.value(), half_width=self.e_hw.value(),
                      node="raw" if node == "separated" else node,
                      register=self.chk_register.isChecked(), reg_roi=self._drift_roi(), crop_roi=self._crop())
