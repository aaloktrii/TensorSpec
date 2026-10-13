import sys
import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                               QLabel, QApplication, QPushButton, QFileDialog,
                               QMessageBox, QComboBox, QListWidget, QSplitter,
                               QAbstractItemView, QPlainTextEdit)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from tensorspec.gui.components.arpes_panel import ARPESPanel
from PySide6.QtWidgets import QStackedWidget, QGroupBox
from tensorspec.gui.components.data_viewer_panel import DataViewerPanel
from tensorspec.core.io.arpes_loader import ARPESLoader
from tensorspec.core.workspace import global_workspace

from tensorspec.core.io.simulated_loader import SimulatedARPESLoader


class ARPESSuite(QWidget):
    """
    Main container for all ARPES-related tools. 
    Integrates the Simulation Engine and the Experimental Data Viewer.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._is_closing = False
        
        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        
        # Tab 1: Hierarchical Simulation Engine
        self.simulation_tab = QWidget()
        sim_layout = QVBoxLayout(self.simulation_tab)
        
        # Panel 0: Unified ARPES Panel
        self.arpes_panel = ARPESPanel()
        sim_layout.addWidget(self.arpes_panel)
        
        self.tabs.addTab(self.simulation_tab, "Simulations (Chinook/SPRKKR)")
        
        # Tab 2: Experimental Data Viewer & Loader
        self.init_data_viewer_tab()
        
        self.layout.addWidget(self.tabs)

    def init_data_viewer_tab(self):
        """Builds the Data Viewer tab with a Fetcher sidebar and Universal Dashboard."""
        self.data_tab = QWidget()
        data_layout = QVBoxLayout(self.data_tab)

        # Use a QSplitter to separate the Fetcher list from the Viewer
        self.viewer_splitter = QSplitter(Qt.Horizontal)
        self.viewer_splitter.setChildrenCollapsible(False)
        data_layout.addWidget(self.viewer_splitter)

        # --- LEFT SIDE: THE FETCHER ---
        fetcher_widget = QWidget()
        fetcher_layout = QVBoxLayout(fetcher_widget)

        self.btn_load = QPushButton("📂 Load ARPES Data...")
        self.btn_load.setStyleSheet("background-color: #0F6A8B; color: white; font-weight: bold; padding: 6px;")
        self.btn_load.clicked.connect(self.load_arpes_files)
        fetcher_layout.addWidget(self.btn_load)

        fetcher_layout.addWidget(QLabel("<b>ARPES Workspace:</b>"))

        self.data_list = QListWidget()
        self.data_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.data_list.itemDoubleClicked.connect(self.fetch_data_to_viewer)
        self.data_list.itemSelectionChanged.connect(self.update_metadata_display)
        fetcher_layout.addWidget(self.data_list)

        self.btn_refresh = QPushButton("🔄 Refresh List")
        self.btn_refresh.clicked.connect(self.refresh_fetcher_list)
        fetcher_layout.addWidget(self.btn_refresh)

        # --- Metadata Display ---
        fetcher_layout.addWidget(QLabel("<b>Metadata:</b>"))
        self.metadata_text = QPlainTextEdit()
        self.metadata_text.setReadOnly(True)
        self.metadata_text.setMaximumHeight(200)
        self.metadata_text.setFont(self.get_monospace_font())
        fetcher_layout.addWidget(self.metadata_text)

        self.viewer_splitter.addWidget(fetcher_widget)

        # --- RIGHT SIDE: THE VIEWER ---
        self.viewer_panel = DataViewerPanel()
        self.viewer_splitter.addWidget(self.viewer_panel)
        self.viewer_splitter.setStretchFactor(0, 0)
        self.viewer_splitter.setStretchFactor(1, 1)
        # Set initial sizes (25% sidebar, 75% viewer)
        self.viewer_splitter.setSizes([300, 700])

        self.tabs.addTab(self.data_tab, "Data Viewer")
        self.refresh_fetcher_list()

    @staticmethod
    def get_monospace_font():
        font = QFont()
        font.setFamily("Courier")
        font.setPointSize(9)
        return font

    def load_arpes_files(self):
        """Allows multi-selection of experimental HDF5 and simulated NPZ files."""
        # Update the file filter to include .npz
        paths, _ = QFileDialog.getOpenFileNames(self, "Load ARPES Data", "", "ARPES Data (*.h5 *.hdf5 *.npz)")
        if not paths: return
        
        success_count = 0
        for path in paths:
            var_name = os.path.basename(path).split('.')[0]
            try:
                # Route the file to the appropriate loader based on its extension
                if path.endswith('.npz'):
                    tensor_data = SimulatedARPESLoader.load(path)
                else:
                    tensor_data = ARPESLoader.load(path)
                
                # Push it silently to the central memory tree
                global_workspace.push_spectroscopy_data(var_name, tensor_data)
                success_count += 1
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load '{var_name}':\n{str(e)}")
        
        if success_count > 0:
            self.refresh_fetcher_list()
            QMessageBox.information(self, "Load Complete", f"Successfully loaded {success_count} dataset(s) into the Workspace.")

    def refresh_fetcher_list(self):
        """Scans the global workspace and lists only ARPES-compatible data."""
        self.data_list.clear()
        
        # Check all stored spectroscopy trees in the workspace
        for name, item in global_workspace._data.items():
            if item.get('type') == 'spectroscopy_tree':
                # We can add further logic here to strictly check if it's ARPES data
                # by pulling the /raw node and checking data_type, but for now we list all trees.
                self.data_list.addItem(name)

    def fetch_data_to_viewer(self, item):
        """Triggered when user double-clicks an item in the sidebar."""
        var_name = item.text()

        # Safely pull data, letting the workspace use its default root node
        official_data = global_workspace.pull_tensor_data(var_name)

        # Fallback: Try with and without the slash
        if not official_data:
            official_data = global_workspace.pull_tensor_data(var_name, node="raw")
        if not official_data:
            official_data = global_workspace.pull_tensor_data(var_name, node="/raw")

        if official_data:
            self.viewer_panel.load_data(official_data)
            self.display_metadata(official_data)
        else:
            QMessageBox.warning(self, "Fetch Error", f"Could not retrieve {var_name} from the Workspace.")

    def update_metadata_display(self):
        """Update metadata when selection changes in the list."""
        if self.data_list.selectedItems():
            item = self.data_list.selectedItems()[0]
            var_name = item.text()

            # Pull data to get metadata
            official_data = global_workspace.pull_tensor_data(var_name)
            if not official_data:
                official_data = global_workspace.pull_tensor_data(var_name, node="raw")
            if not official_data:
                official_data = global_workspace.pull_tensor_data(var_name, node="/raw")

            if official_data:
                self.display_metadata(official_data)

    def display_metadata(self, tensor_data):
        """Display tensor data metadata in the metadata panel."""
        lines = []

        if tensor_data is None:
            self.metadata_text.setPlainText("No data loaded")
            return

        # Basic info
        lines.append(f"Type: {tensor_data.data_type}")
        lines.append(f"Shape: {tensor_data.value.shape}")
        lines.append(f"Ndim: {tensor_data.ndim}")
        lines.append("")

        # Axes info
        lines.append("AXES:")
        for i, (label, unit) in enumerate(zip(tensor_data.labels, tensor_data.units)):
            axis_data = tensor_data.axes[i]
            lines.append(f"  [{i}] {label} ({unit}): {len(axis_data)} points")
            if len(axis_data) > 0:
                lines.append(f"       Range: [{axis_data[0]:.3g}, {axis_data[-1]:.3g}]")

        # Metadata
        if tensor_data.metadata:
            lines.append("")
            lines.append("METADATA:")

            meta = tensor_data.metadata

            # Temperatures
            if 'temperatures' in meta and meta['temperatures']:
                lines.append("  Temperatures:")
                for key, val in meta['temperatures'].items():
                    if isinstance(val, dict):
                        lines.append(f"    {key}: {val.get('mean', val):.2f} K")
                    else:
                        lines.append(f"    {key}: {val:.2f} K")

            # Note: Voltages removed as they are device-specific

            # Photon energy
            if 'photon_energy' in meta:
                lines.append(f"  Photon Energy: {meta['photon_energy']}")
            if 'photon_energy_eV' in meta:
                lines.append(f"  Photon Energy: {meta['photon_energy_eV']:.2f} eV")

            # Facility
            if 'facility' in meta:
                lines.append(f"  Facility: {meta['facility']}")

        self.metadata_text.setPlainText("\n".join(lines))

    def closeEvent(self, event):
        self._is_closing = True
        event.accept()

# Standalone runner for independent testing
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ARPESSuite()
    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec())