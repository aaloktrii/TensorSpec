"""PEEM suite container (main PEEM suite of the launcher ribbon).

Tabs on one shared workspace:
  Stack · BG · sum rule   the original PEEM panel (pair / separate, NCC drift, background, sum rule)
  XPEEM map               loading (format and polarization from the data), stack clean-up, drift
                          correction, stack + asymmetry, image analysis, ROIs and domain maps (data viewer)
The XPEEM-map result can be handed to the first tab ("Load from XPEEM map") for background
subtraction and the sum rule. XAS spectra are a separate suite (xas_suite.py).
"""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTabWidget, QVBoxLayout, QWidget

from tensorspec.gui.components.peem_panel import PeemPanel
from tensorspec.gui.components.peem_xmcd_panel import PeemXmcdPanel
from tensorspec.gui.services.peem_service import PeemService


class PEEMSuite(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        layout = QVBoxLayout(self)
        self.service = PeemService()  # the one PEEM service; both tabs use it
        self.tabs = QTabWidget()
        self.xmcd_panel = PeemXmcdPanel(service=self.service)
        self.panel = PeemPanel()
        self.panel.service = self.service
        self.panel.set_xmcd_source(self._xmcd_to_stack)
        self.tabs.addTab(self.xmcd_panel, "XPEEM map")
        self.tabs.addTab(self.panel, "Stack · BG · sum rule")
        layout.addWidget(self.tabs)

    def _xmcd_to_stack(self) -> dict:
        """XPEEM-map result -> stack tab (background subtraction, sum rule)."""
        name = self.xmcd_panel._name
        if not name:
            raise ValueError("Load data in the XPEEM map tab first (and run its drift correction).")
        return self.service.to_peem_stack(name, self.xmcd_panel.pair_combo.currentData())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PEEMSuite()
    window.resize(1400, 950)
    window.setWindowTitle("TensorSpec - PEEM Suite")
    window.show()
    sys.exit(app.exec())
