"""Raman suite container (main browser ▸ Raman Suite).

Tabs (gui/components/raman_panel.py): Map Explorer · Classification · Spectra · Stacked spectra ·
Layer prediction · Peak fitting, on one RamanService (gui/services/raman_service.py) that keeps the
loaded map and its /analysis results in the global workspace.
"""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from tensorspec.gui.components.raman_panel import RamanPanel


class RamanSuite(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.panel = RamanPanel()
        layout.addWidget(self.panel)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RamanSuite()
    window.resize(1500, 960)
    window.setWindowTitle("TensorSpec - Raman Suite")
    window.show()
    if len(sys.argv) > 1:
        window.panel.open_raman(sys.argv[1])
    sys.exit(app.exec())
