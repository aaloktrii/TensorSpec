"""PEEM suite container.

Tabs on one shared workspace (one PeemService):
  PEEM        the original panel, unchanged: load, pair / separate, drift, background, sum rule
  XPEEM map   added: UE49 / UView / P3B loading (format and polarization from the data), stack
              clean-up, drift correction, stack + asymmetry, image analysis, ROIs and domain maps
              (data viewer). Its result can be sent to the PEEM tab ("→ PEEM tab") for the
              background and the sum rule; it arrives there as a loaded dataset.
"""
from __future__ import annotations

import sys
from collections import Counter

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
        self.panel = PeemPanel()
        self.service = self.panel.service  # one service for both tabs
        self.xmcd_panel = PeemXmcdPanel(service=self.service)
        self.xmcd_panel.set_xas_target(self._to_peem_tab)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.panel, "PEEM")
        self.tabs.addTab(self.xmcd_panel, "XPEEM map")
        layout.addWidget(self.tabs)

    def _to_peem_tab(self):
        """XPEEM-map images -> a new dataset (paired + separated) shown in the PEEM tab, through the
        panel's own load-finished path (the panel itself is not changed)."""
        name = self.xmcd_panel._name
        if not name:
            self.xmcd_panel._err("Load data in the XPEEM map tab first (and run its drift correction).")
            return
        try:
            h = self.service.to_peem_stack(name, self.xmcd_panel.pair_combo.currentData(),
                                           source=self.xmcd_panel.send_source.currentData())
            meta = self.service.get_meta(h["name"])
        except ValueError as exc:
            self.xmcd_panel._err(str(exc))
            return
        summary = {"name": h["name"], "n_frames": meta["n_frames"], "pol_summary": dict(Counter(meta["pol"]))}
        self.panel._on_load_finished(True, summary, "")
        self.tabs.setCurrentWidget(self.panel)
        self.xmcd_panel._log(f"Sent to the PEEM tab as '{h['name']}'.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PEEMSuite()
    window.resize(1100, 750)
    window.setWindowTitle("TensorSpec - PEEM Suite")
    window.show()
    sys.exit(app.exec())
