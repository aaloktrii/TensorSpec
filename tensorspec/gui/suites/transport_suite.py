"""Transport suite — magnetoresistance / Hall / R(T) shell (engine pending)."""

import sys

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


class TransportSuite(QWidget):
    """PEEM-style split panel for transport curves (demo until transport_engine ships)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._build_ui()
        self._redraw_demo()

    def _build_ui(self):
        root = QVBoxLayout(self)
        header = QLabel("<h2>Transport Suite</h2><p>Demo curves — transport_engine.py not wired yet.</p>")
        root.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        load_group = QGroupBox("1. Load")
        load_form = QFormLayout(load_group)
        self.name_edit = QLineEdit()
        load_form.addRow("Dataset name:", self.name_edit)
        btn_load = QPushButton("Load Transport Data…")
        btn_load.clicked.connect(self._load_file)
        load_form.addRow(btn_load)
        self.status = QLabel("No curves loaded")
        load_form.addRow(self.status)
        left_layout.addWidget(load_group)

        type_group = QGroupBox("2. Measurement Type")
        type_form = QFormLayout(type_group)
        self.curve_type = QComboBox()
        self.curve_type.addItem("Magnetoresistance R(H)", "mr")
        self.curve_type.addItem("Hall R_xy(H)", "hall")
        self.curve_type.addItem("Resistivity R(T)", "rt")
        self.curve_type.currentIndexChanged.connect(self._redraw_demo)
        type_form.addRow("Curve type:", self.curve_type)
        self.geometry = QComboBox()
        self.geometry.addItems(["Hall bar", "van der Pauw", "Four-point linear"])
        type_form.addRow("Geometry:", self.geometry)
        left_layout.addWidget(type_group)

        dim_group = QGroupBox("3. Sample Dimensions")
        dim_form = QFormLayout(dim_group)
        self.thickness = QDoubleSpinBox()
        self.width = QDoubleSpinBox()
        self.length = QDoubleSpinBox()
        self.current = QDoubleSpinBox()
        self.thickness.setRange(0, 1e6)
        self.thickness.setValue(100)
        self.width.setRange(0, 1e6)
        self.width.setValue(10)
        self.length.setRange(0, 1e6)
        self.length.setValue(50)
        self.current.setRange(0, 1e6)
        self.current.setValue(10.0)
        dim_form.addRow("Thickness (nm):", self.thickness)
        dim_form.addRow("Width (µm):", self.width)
        dim_form.addRow("Length (µm):", self.length)
        dim_form.addRow("Current (µA):", self.current)
        left_layout.addWidget(dim_group)

        proc_group = QGroupBox("4. Processing")
        proc_layout = QVBoxLayout(proc_group)
        self.symmetrize = QCheckBox("Symmetrize in field")
        self.symmetrize.setChecked(True)
        self.hall_bg = QCheckBox("Subtract ordinary Hall background")
        proc_layout.addWidget(self.symmetrize)
        proc_layout.addWidget(self.hall_bg)
        self.fit_model = QComboBox()
        self.fit_model.addItems(["Single carrier", "Two-band", "Power law (R-T scaling)"])
        row = QHBoxLayout()
        row.addWidget(QLabel("Fit model:"))
        row.addWidget(self.fit_model)
        wrap = QWidget()
        wrap.setLayout(row)
        proc_layout.addWidget(wrap)
        btn_extract = QPushButton("Extract Transport Parameters")
        btn_extract.clicked.connect(lambda: self._pending("Parameter extraction"))
        proc_layout.addWidget(btn_extract)
        left_layout.addWidget(proc_group)

        left_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        splitter.addWidget(scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.fig = Figure(figsize=(5, 4), dpi=100, layout="tight")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        right_layout.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 660])

    def _curve_kind(self) -> str:
        return self.curve_type.currentData() or "mr"

    def _demo_curve(self):
        kind = self._curve_kind()
        n = 101
        if kind == "rt":
            x = np.linspace(2, 298, n)
            y = 1.0 + 0.003 * x
            xlab, ylab = "T (K)", "R (Ω)"
        elif kind == "hall":
            x = np.linspace(-5, 5, n)
            y = 0.05 * x
            xlab, ylab = "H (T)", "R_xy (Ω)"
        else:
            x = np.linspace(-5, 5, n)
            y = 1.0 + 0.02 * x * x
            xlab, ylab = "H (T)", "R (Ω)"
        return x, y, xlab, ylab

    def _redraw_demo(self):
        x, y, xlab, ylab = self._demo_curve()
        self.ax.clear()
        self.ax.plot(x, y, color="#60a5fa", lw=1.5)
        self.ax.set_xlabel(xlab)
        self.ax.set_ylabel(ylab)
        self.ax.set_title("Demo transport curve")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
        self.status.setText(f"Demo {xlab.split()[0]} curve ready")

    def _pending(self, action: str):
        self.status.setText(f"{action} — engine pending")
        QMessageBox.information(self, "Engine pending", "transport_engine.py is not wired on TensorSpec_GUI yet.")

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load transport data", "", "Data (*.csv *.txt *.dat)")
        if not path:
            return
        if not self.name_edit.text().strip():
            self.name_edit.setText(path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
        self._pending(f"Load {path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TransportSuite()
    win.resize(1000, 700)
    win.setWindowTitle("TensorSpec - Transport Suite")
    win.show()
    sys.exit(app.exec())
