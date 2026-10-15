"""XAS / XMCD suite container: the Sum Rule window.

Tabs:
  From PEEM          corrected CP / CM pairs pushed from the PEEM panel -> spectra of a region ->
                     background -> XMCD sum rule (peem_sumrule_panel.py)
  XAS spectra (.txt) 1D XAS / XMCD files -> background -> sum rule (xas_panel.py)
"""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTabWidget, QVBoxLayout, QWidget

from tensorspec.gui.components.peem_sumrule_panel import PeemSumRulePanel
from tensorspec.gui.components.xas_panel import XasPanel


class XASSuite(QWidget):
    def __init__(self, parent=None, peem_service=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        layout = QVBoxLayout(self)
        self.peem_panel = PeemSumRulePanel(service=peem_service)
        self.panel = XasPanel()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.peem_panel, "From PEEM")
        self.tabs.addTab(self.panel, "XAS spectra (.txt)")
        layout.addWidget(self.tabs)

    def show_peem(self, name: str):
        """Show a PEEM dataset in the From PEEM tab (used by the PEEM panel's '→ Sum Rule')."""
        self.tabs.setCurrentWidget(self.peem_panel)
        self.peem_panel.select_dataset(name)


_OPENED: list[XASSuite] = []  # windows opened from the PEEM panel (a Python reference keeps them alive)


def open_sum_rule(name: str, peem_service=None) -> XASSuite:
    """Bring up the Sum Rule window (an open one, else a new one) showing PEEM dataset ``name``."""
    suites = [w for w in QApplication.allWidgets() if isinstance(w, XASSuite) and w.isVisible()]
    suite = suites[0] if suites else XASSuite(peem_service=peem_service)
    if not suites:
        _OPENED.append(suite)
        suite.destroyed.connect(lambda *_: _OPENED.remove(suite) if suite in _OPENED else None)
        suite.resize(1100, 750)
        suite.setWindowTitle("TensorSpec - XAS / XMCD Suite")
        suite.show()
    suite.show_peem(name)
    win = suite.window()
    win.raise_()
    win.activateWindow()
    return suite


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XASSuite()
    window.resize(1000, 700)
    window.setWindowTitle("TensorSpec - XAS / XMCD Suite")
    window.show()
    sys.exit(app.exec())
