"""PEEM suite container."""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from tensorspec.gui.components.peem_panel import PeemPanel


class PEEMSuite(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        layout = QVBoxLayout(self)
        self.panel = PeemPanel()
        layout.addWidget(self.panel)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PEEMSuite()
    window.resize(1100, 750)
    window.setWindowTitle("TensorSpec - PEEM Suite")
    window.show()
    sys.exit(app.exec())
