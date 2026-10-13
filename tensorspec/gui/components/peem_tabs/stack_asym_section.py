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
    QPushButton,
    QVBoxLayout,
    QWidget,
)


_DIVERGING = {"pcd", "pld", "pmld", "pmnld"}

from tensorspec.gui.components.peem_tabs.common import _dspin


class StackAsymSection:
    def _build_stack_asym(self, cl):
        """Controls of this section, added to the control column ``cl``."""
        # ---------------------------------------------------------------- 3. Stack
        stk = QGroupBox("4. Stack each polarization into one image")
        sk = QFormLayout(stk)
        self.pair_combo = QComboBox()
        self.pair_combo.addItem("XMCD (CL/CR · ±0.9 · CP/CM)", "xmcd")
        self.pair_combo.addItem("XLD (LH/LV)", "xld")
        sk.addRow("Polarization pair:", self.pair_combo)
        b = QPushButton("Stack")
        b.setToolTip("Average the drift-corrected frames of each polarization, then align the two images on edges")
        b.clicked.connect(self._stack_step)
        sk.addRow(b)
        self.stack_out = QComboBox()
        self._set_stack_labels("CP", "CM")
        self.stack_out.setToolTip("Stacked images in the viewer (after Stack)")
        self.stack_out.currentIndexChanged.connect(self._show_stack_output)
        sk.addRow(self.stack_out)
        cl.addWidget(stk)

        # ---------------------------------------------------------------- 4. Asymmetry
        dich = QGroupBox("5. Asymmetry")
        dl = QFormLayout(dich)
        self.conv_combo = QComboBox()
        self.conv_combo.addItem("UE49: (CM−CP)/(CM+CP)", "ue49")
        self.conv_combo.addItem("(CP−CM)/(CP+CM) = BNL (CL−CR) · ALS (+0.9 − −0.9)", "cp_minus_cm")
        dl.addRow("XMCD sign:", self.conv_combo)
        b = QPushButton("Subtract + asymmetry")
        b.setToolTip("From the stacked images A (CL) and B (CR): difference A-B and asymmetry (A-B)/(A+B)")
        b.clicked.connect(self._asym_step)
        dl.addRow(b)
        self.asym_out = QComboBox()
        self.asym_out.addItem("Show: asymmetry (A−B)/(A+B)", "data")
        self.asym_out.addItem("Show: difference A−B (subtraction)", "difference")
        self.asym_out.addItem("Show: A (stacked CL)", "a")
        self.asym_out.addItem("Show: B (stacked CR)", "b")
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
                      node=self.node_combo.currentData(), convention=self.conv_combo.currentData(),
                      register=self.chk_register.isChecked(), reg_roi=self._drift_roi(), crop_roi=self._crop())

    def _stack_step(self):
        name = self._require()
        if name:
            self._run(self.service.stack, self._on_stacked, name, pair=self.pair_combo.currentData())

    def _set_stack_labels(self, a: str, b: str, n_a: int | None = None, n_b: int | None = None):
        na = f", {n_a} frames" if n_a else ""
        nb = f", {n_b} frames" if n_b else ""
        cur = self.stack_out.currentData() or "a_mean"
        self.stack_out.blockSignals(True)
        self.stack_out.clear()
        for label, key in ((f"Show: {a} stacked (mean{na})", "a_mean"), (f"Show: {b} stacked (mean{nb})", "b_mean"),
                           (f"Show: {a} + {b} (sum of the two stacked images)", "data"),
                           (f"Show: {a}, sum of frames", "a_sum"), (f"Show: {b}, sum of frames", "b_sum")):
            self.stack_out.addItem(label, key)
        self.stack_out.setCurrentIndex(max(0, self.stack_out.findData(cur)))
        self.stack_out.blockSignals(False)

    def _on_stacked(self, res: dict):
        at = self.service.get_product_attrs(self._name, "stack")
        self._set_stack_labels(at.get("channel_a_name", "A"), at.get("channel_b_name", "B"),
                               at.get("n_a"), at.get("n_b"))
        self._show_result(res, "stack")
        self._show_stack_output()

    def _show_stack_output(self, *_):
        """Show the chosen stacked image (layer in the viewer)."""
        if not self._name or "stack" not in self.service.list_products(self._name):
            return
        want = self.stack_out.currentData()
        if want != "data":
            at = self.service.get_product_attrs(self._name, "stack")
            tag = at.get("channel_a" if want.startswith("a") else "channel_b", "")
            want = f"{'mean' if want.endswith('mean') else 'sum'}_{str(tag).lower()}"
            if want not in self.service.map_variables(self._name, "stack"):
                self._err("Run Stack again to get the sums of the frames (older result).")
                return
        self._select("stack", want)

    def _show_asym_output(self, *_):
        """Show the chosen output of the asymmetry step (layer in the viewer)."""
        if not self._name:
            return
        kind = "pcd" if self.pair_combo.currentData() == "xmcd" else "pld"
        if kind not in self.service.list_products(self._name):
            return
        want = self.asym_out.currentData()
        if want in ("a", "b"):
            means = [v for v in self.service.map_variables(self._name, kind) if v.startswith("mean_")]
            want = means[0 if want == "a" else 1] if len(means) == 2 else "data"
        self._select(kind, want)

    def _asym_step(self):
        name = self._require()
        if name:
            pair = self.pair_combo.currentData()
            self._run(self.service.asymmetry,
                      lambda r: (self._show_result(r, "pcd" if pair == "xmcd" else "pld"), self._show_asym_output()),
                      name, convention=self.conv_combo.currentData(), crop_roi=self._crop())

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
