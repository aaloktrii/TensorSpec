import sys
import os
# --- Python Version Check ---
if sys.version_info < (3, 11):
    sys.exit("\n[ERROR] TensorSpec requires Python 3.11 or newer.\n"
             f"You are currently running Python {sys.version_info.major}.{sys.version_info.minor}.\n"
             "Please upgrade your Python environment or create a new virtual environment using python3.11.\n")
             
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QTreeWidget, QTreeWidgetItem, QToolBar, 
                               QSplitter, QTextEdit, QLabel, QApplication, 
                               QPushButton, QListWidget, QMessageBox, QMenu)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QIcon

from tensorspec.gui.suites.crystal_suite import CrystalViewerSuite
from tensorspec.gui.suites.arpes_suite import ARPESSuite
from tensorspec.gui.suites.dft_suite import DFTSuite
from tensorspec.gui.components.data_viewer_panel import DataViewerPanel
from tensorspec.core.workspace import global_workspace
from tensorspec.core.io.arpes_loader import ARPESLoader
from tensorspec.core.io.simulated_loader import SimulatedARPESLoader

import time
import platform
import psutil
from collections import deque
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QMessageBox

class TelemetryWindow(QWidget):
    """A hidden developer dashboard tracking and recording live system resource consumption."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensorSpec Telemetry - Hardware Profiler")
        self.resize(800, 450)
        self.setWindowFlags(Qt.WindowStaysOnTopHint) 
        
        # --- Recording States ---
        # Capped: a 1 Hz recording left running for hours used to grow without
        # bound. 5000 samples is a little over 80 minutes of history.
        self.max_history = 5000
        self.is_recording = False
        self.history_time = deque(maxlen=self.max_history)
        self.history_cpu = deque(maxlen=self.max_history)
        self.history_ram = deque(maxlen=self.max_history)
        self.history_rss = deque(maxlen=self.max_history)
        self.start_time = None

        # Our own resident set size, so a run that is about to exhaust memory is
        # distinguishable from unrelated system load.
        self.process = psutil.Process(os.getpid())
        
        layout = QVBoxLayout(self)
        
        # --- UI: Top Control Ribbon ---
        ctrl_layout = QHBoxLayout()
        
        self.btn_record = QPushButton("🔴 Start Recording")
        self.btn_record.setStyleSheet("font-weight: bold; background-color: #d9534f; color: white; padding: 6px;")
        self.btn_record.clicked.connect(self.toggle_recording)
        ctrl_layout.addWidget(self.btn_record)
        
        self.btn_save = QPushButton("💾 Save Recorded Graph")
        self.btn_save.setEnabled(False) # Disabled until a recording finishes
        self.btn_save.clicked.connect(self.save_graph)
        ctrl_layout.addWidget(self.btn_save)
        
        layout.addLayout(ctrl_layout)
        
        # --- UI: Live Matplotlib Canvas ---
        self.fig = Figure(figsize=(8, 4), dpi=100, layout='tight')
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        self.ax_cpu = self.fig.add_subplot(121)
        self.ax_ram = self.fig.add_subplot(122)
        
        # Start tracking
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(1000)
        self.update_metrics()

    def toggle_recording(self):
        """Switches between recording and stopped states."""
        if not self.is_recording:
            # Begin Recording
            self.is_recording = True
            self.history_time.clear()
            self.history_cpu.clear()
            self.history_ram.clear()
            self.history_rss.clear()
            self.start_time = time.time()
            
            self.btn_record.setText("⏹ Stop Recording")
            self.btn_record.setStyleSheet("font-weight: bold; background-color: #5bc0de; color: black; padding: 6px;")
            self.btn_save.setEnabled(False)
        else:
            # Stop Recording
            self.is_recording = False
            self.btn_record.setText("🔴 Start Recording")
            self.btn_record.setStyleSheet("font-weight: bold; background-color: #d9534f; color: white; padding: 6px;")
            self.btn_save.setEnabled(len(self.history_time) > 0)

    def update_metrics(self):
        """Polls the hardware every 1 second and draws the live bar charts."""
        self.ax_cpu.clear()
        self.ax_ram.clear()
        
        # 1. Hardware Polling
        cpu_percents = psutil.cpu_percent(percpu=True)
        total_cpu = psutil.cpu_percent()
        cores = range(len(cpu_percents))
        
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)
        rss_gb = self.process.memory_info().rss / (1024**3)
        rss_percent = 100.0 * rss_gb / total_gb if total_gb else 0.0
        
        # 2. Draw Live CPU Bar Chart
        self.ax_cpu.bar(cores, cpu_percents, color='#5bc0de', edgecolor='black')
        self.ax_cpu.set_ylim(0, 100)
        self.ax_cpu.set_title(f"CPU Usage per Thread (Total: {total_cpu}%)")
        self.ax_cpu.set_xlabel("Logical Core ID")
        self.ax_cpu.set_ylabel("Utilization (%)")
        
        # 3. Draw Live RAM Bar Chart
        mem_label = "Unified Memory" if platform.system() == "Darwin" else "System RAM"
        self.ax_ram.bar([mem_label, "TensorSpec"], [mem.percent, rss_percent],
                        color=['#d9534f', '#f0ad4e'], edgecolor='black', width=0.4)
        self.ax_ram.set_ylim(0, 100)
        self.ax_ram.set_title(f"RAM: {used_gb:.1f} / {total_gb:.1f} GB"
                              f"   |   TensorSpec: {rss_gb:.2f} GB")
        self.ax_ram.set_ylabel("Capacity Used (%)")
        
        self.canvas.draw()
        
        # 4. Log the data if recording is active
        if self.is_recording:
            elapsed_seconds = time.time() - self.start_time
            self.history_time.append(elapsed_seconds)
            self.history_cpu.append(total_cpu)
            self.history_ram.append(mem.percent)
            self.history_rss.append(rss_gb)

    def save_graph(self):
        """Generates a beautiful time-series plot of the recorded session and saves it."""
        if not self.history_time:
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save Telemetry Graph", "tensorspec_stress_test.png", "PNG Images (*.png);;PDF Files (*.pdf)")
        if not path:
            return
            
        try:
            # Create a separate, hidden figure specifically for the exported line graph
            export_fig = Figure(figsize=(8, 5), dpi=150, layout='tight')
            ax = export_fig.add_subplot(111)
            
            # Plot the recorded data
            lines = []
            lines += ax.plot(self.history_time, self.history_cpu, label='Total CPU Usage (%)', color='#5bc0de', linewidth=2.5)
            lines += ax.plot(self.history_time, self.history_ram, label='RAM Usage (%)', color='#d9534f', linewidth=2.5)
            
            # Style it professionally for a presentation
            ax.set_title("TensorSpec Hardware Stress Test over Time", fontsize=14, fontweight='bold')
            ax.set_xlabel("Elapsed Time (Seconds)", fontsize=12)
            ax.set_ylabel("System Utilization (%)", fontsize=12)
            ax.set_ylim(0, 105)
            ax.grid(True, linestyle='--', alpha=0.7)

            # Process memory keeps its own axis: absolute GB is what tells you
            # whether a long run is heading for an out-of-memory kill.
            ax_rss = ax.twinx()
            lines += ax_rss.plot(self.history_time, self.history_rss,
                                 label='TensorSpec Process RAM (GB)',
                                 color='#f0ad4e', linewidth=2.5, linestyle='--')
            ax_rss.set_ylabel("Process Memory (GB)", fontsize=12, color='#b8860b')
            ax_rss.tick_params(axis='y', labelcolor='#b8860b')
            ax_rss.set_ylim(0, max(1.0, max(self.history_rss) * 1.5))

            ax.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=11)
            
            export_fig.savefig(path)
            QMessageBox.information(self, "Export Successful", f"Time-series graph saved successfully to:\n{path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save telemetry graph:\n{str(e)}")

# --- NEW: Floating Window Wrapper ---
class FloatingViewerWindow(QMainWindow):
    """A wrapper that turns any panel into a tracked floating window."""
    window_closed = Signal(str)

    def __init__(self, win_id: str, title: str, inner_widget: QWidget, parent=None):
        super().__init__(parent)
        self.win_id = win_id
        self.setWindowTitle(f"TensorSpec Viewer: {title}")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(1200, 800)
        
        # Ensures the window floats independently but retains OS event focus
        self.setWindowFlags(Qt.Window) 

        self.setCentralWidget(inner_widget)

    def closeEvent(self, event):
        # Notify the Main Browser registry that this window is closing
        self.window_closed.emit(self.win_id)
        super().closeEvent(event)


class TensorSpecMainBrowser(QMainWindow):
    """
    The Central Data Workspace Browser for TensorSpec.
    Manages in-memory data trees and coordinates independent analytical suites.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensorSpec Workspace Browser")
        self.resize(1000, 700)
        
        # Central Memory Workspace Pool (Dictionary tracking active variables)
        self.workspace_data = {}
        self.active_windows = {}  # NEW: Window Registry
        
        # Initialize UI Components
        self.init_menubar()
        self.init_launcher_toolbar()
        self.init_central_layout()
        
        self.refresh_workspace_tree()
        
        # NEW: Secret keystroke buffer for the telemetry Easter egg
        self._secret_buffer = ""

    def keyPressEvent(self, event):
        """Listens for the secret Easter egg phrase to unlock the telemetry panel."""
        if event.text():
            # Append the typed character and keep the buffer lowercase
            self._secret_buffer += event.text().lower()
            
            # Keep the buffer from growing infinitely (the phrase is 21 chars long)
            self._secret_buffer = self._secret_buffer[-30:]
            
            # Check if the secret phrase is inside the buffer
            if "should i buy new mac?" in self._secret_buffer:
                self._secret_buffer = "" # Reset buffer
                self.launch_telemetry_panel()
                
        super().keyPressEvent(event)
    
    def show_secret_menu(self, pos):
        """Displays the hidden developer menu when the user right-clicks the designated label."""
        menu = QMenu(self)
        secret_act = menu.addAction("Should I buy a new mac?")
        
        # Map the local click position to global screen coordinates so the menu spawns on your mouse
        action = menu.exec(self.lbl_active_vars.mapToGlobal(pos))
        
        if action == secret_act:
            self.launch_telemetry_panel()

    def launch_telemetry_panel(self):
        """Spawns the hidden hardware profiler."""
        if not hasattr(self, 'telemetry_window') or not self.telemetry_window.isVisible():
            self.telemetry_window = TelemetryWindow()
            self.telemetry_window.show()

    def init_menubar(self):
        """Initializes top-level system menus."""
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        
        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def init_launcher_toolbar(self):
        """Builds the App Launcher Ribbon for launching independent functional suites."""
        launcher_toolbar = QToolBar("Suite Launcher")
        launcher_toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.TopToolBarArea, launcher_toolbar)
        
        # Define actions for all roadmap suites
        suites = [
            ("Crystal Viewer", self.launch_crystal_suite),
            ("DFT Suite", self.launch_dft_suite),
            ("ARPES Suite", self.launch_arpes_suite),
            ("PEEM Suite", self.launch_peem_suite),
            ("XAS Suite", self.launch_xas_suite),
            ("Raman Suite", self.launch_raman_suite),
            ("Transport Suite", self.launch_transport_suite),
            ("Machine Learning", self.launch_ml_suite),
        ]
        
        for name, slot in suites:
            action = QAction(name, self)
            action.triggered.connect(slot)
            launcher_toolbar.addAction(action)
            
        launcher_toolbar.addSeparator()
        
        compute_action = QAction("☁️ Remote Compute", self)
        compute_action.triggered.connect(self.launch_compute_manager)
        launcher_toolbar.addAction(compute_action)

    def init_central_layout(self):
        """Creates the split workspace: Left Data Tree and Right Inspector Panel."""
        main_splitter = QSplitter(Qt.Horizontal)
        
        # --- Left Panel: Data Tree Explorer & Window Registry ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        top_left_layout = QHBoxLayout()
        
        # --- NEW: Secret Right-Click Target ---
        self.lbl_active_vars = QLabel("<b>Active Workspace Variables</b>")
        self.lbl_active_vars.setContextMenuPolicy(Qt.CustomContextMenu)
        self.lbl_active_vars.customContextMenuRequested.connect(self.show_secret_menu)
        top_left_layout.addWidget(self.lbl_active_vars)
        
        self.btn_load_workspace = QPushButton("Load ARPES / Maestro Data...")
        self.btn_load_workspace.clicked.connect(self.load_workspace_files)
        top_left_layout.addWidget(self.btn_load_workspace)

        self.btn_refresh_tree = QPushButton("🔄 Refresh Workspace")
        self.btn_refresh_tree.clicked.connect(self.refresh_workspace_tree)
        top_left_layout.addWidget(self.btn_refresh_tree)
        left_layout.addLayout(top_left_layout)
        
        self.data_tree_widget = QTreeWidget()
        self.data_tree_widget.setHeaderLabels(["Variable Name", "Type", "Dimensions / Shape"])
        self.data_tree_widget.currentItemChanged.connect(self.on_item_selected)
        left_layout.addWidget(self.data_tree_widget)
        
        # NEW: Active Windows Registry Tracker
        left_layout.addWidget(QLabel("<b>Active Floating Windows</b>"))
        self.window_tracker_list = QListWidget()
        self.window_tracker_list.setMaximumHeight(150)
        self.window_tracker_list.itemClicked.connect(self.bring_window_to_front)
        left_layout.addWidget(self.window_tracker_list)
        
        main_splitter.addWidget(left_widget)
        
        # --- Right Panel: Quick Inspector / Preview ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("<b>Variable Metadata Inspector</b>"))
        
        self.metadata_inspector = QTextEdit()
        self.metadata_inspector.setReadOnly(True)
        self.metadata_inspector.setPlaceholderText("Select a workspace item to view properties...")
        right_layout.addWidget(self.metadata_inspector)
        
        self.btn_launch_viewer = QPushButton("📊 Launch Data Viewer for Selected Item")
        self.btn_launch_viewer.setStyleSheet("font-weight: bold; padding: 10px; background-color: #2b5c8f; color: white;")
        self.btn_launch_viewer.setEnabled(False)
        self.btn_launch_viewer.clicked.connect(self.launch_standalone_viewer)
        right_layout.addWidget(self.btn_launch_viewer)
        
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([350, 650])
        self.setCentralWidget(main_splitter)

    def refresh_workspace_tree(self):
        """Pulls the actual active data from global_workspace and populates the tree."""
        self.data_tree_widget.clear()
        
        # Render mock data safely using .get()
        for name, info in self.workspace_data.items():
            item = QTreeWidgetItem(self.data_tree_widget)
            item.setText(0, name)
            item.setText(1, info.get("type", "Unknown"))
            item.setText(2, str(info.get("dims", "N/A")))

        # Fetch Crystal Structures
        for name in global_workspace.list_crystal_structures():
            item = QTreeWidgetItem(self.data_tree_widget)
            item.setText(0, name)
            item.setText(1, "Crystal Structure")
            item.setText(2, "3D Basis Vectors")
            
        # Fetch Band Structures
        for name in global_workspace.list_band_structures():
            item = QTreeWidgetItem(self.data_tree_widget)
            item.setText(0, name)
            item.setText(1, "Band Structure")
            item.setText(2, "E(k) Dispersion Data")

        # 3. Fetch DataTrees (Spectroscopy)
        for name, item_data in global_workspace._data.items():
            if item_data.get('type') == 'spectroscopy_tree':
                item = QTreeWidgetItem(self.data_tree_widget)
                item.setText(0, name)
                item.setText(1, "Spectroscopy Dataset")
                
                # Try to extract shape/type from the raw node if available
                try:
                    raw_node = item_data['tree']['raw']
                    if 'data' in raw_node:
                        shape = raw_node['data'].shape
                        item.setText(2, f"Shape: {shape}")
                except Exception:
                    item.setText(2, "DataTree")

        # 4. Fetch Remote DFT Vaults
        for name in global_workspace.list_remote_runs():
            vault_info = global_workspace.pull_remote_run(name)
            item = QTreeWidgetItem(self.data_tree_widget)
            item.setText(0, name)
            item.setText(1, f"Remote Vault ({vault_info['engine']})")
            item.setText(2, f"Cluster: {vault_info['cluster']}")
            # Give it a special color to stand out as remote!
            item.setForeground(0, Qt.cyan)
            item.setForeground(1, Qt.cyan)
            item.setForeground(2, Qt.cyan)

        # Expand all by default
        self.data_tree_widget.expandAll()

    def load_workspace_files(self):
        """Load ARPES / Maestro files into the central workspace tree."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load ARPES / Maestro Data",
            "",
            "ARPES Data (*.h5 *.hdf5 *.npz)",
        )
        if not paths:
            return

        success_count = 0
        for path in paths:
            var_name = os.path.basename(path).split('.')[0]
            try:
                if path.endswith('.npz'):
                    tensor_data = SimulatedARPESLoader.load(path)
                else:
                    tensor_data = ARPESLoader.load(path)
                global_workspace.push_spectroscopy_data(var_name, tensor_data)
                success_count += 1
            except Exception as e:
                QMessageBox.critical(
                    self, "Load Error", f"Failed to load '{var_name}':\n{str(e)}"
                )

        if success_count > 0:
            self.refresh_workspace_tree()
            QMessageBox.information(
                self,
                "Load Complete",
                f"Successfully loaded {success_count} dataset(s) into the Workspace.",
            )

    def on_item_selected(self, current_item, previous_item):
        """Triggers preview update in the Inspector Panel upon selecting a variable."""
        if not current_item: 
            return
            
        var_name = current_item.text(0)
        self.current_selected_var = var_name
        
        if var_name in self.workspace_data:
            meta_text = self.workspace_data[var_name].get("metadata", "No metadata available for this object.")
            self.metadata_inspector.setText(f"Variable: {var_name}\n" + "-"*40 + f"\n{meta_text}")
            self.btn_launch_viewer.setEnabled(False) # Cannot view mock string data
        else:
            # Check real global workspace
            ws_item = global_workspace._data.get(var_name)
            if ws_item:
                item_type = ws_item.get('type', 'Unknown')
                if item_type == 'remote_run':
                    meta = f"Remote Engine: {ws_item.get('engine')}\nCluster: {ws_item.get('cluster')}\nPath: {ws_item.get('remote_path')}"
                    self.metadata_inspector.setText(f"Workspace Item: {var_name}\nType: Remote DFT Vault\n" + "-"*40 + f"\n{meta}")
                    self.btn_launch_viewer.setEnabled(False)
                else:
                    self.metadata_inspector.setText(f"Workspace Item: {var_name}\nType: {item_type}\n" + "-"*40 + "\nData object ready for inspection.")
                    
                if item_type == 'spectroscopy_tree':
                    self.btn_launch_viewer.setEnabled(True)
                else:
                    self.btn_launch_viewer.setEnabled(False)

    def launch_standalone_viewer(self):
        """Reads the workspace data type and spawns the correct floating component."""
        if not hasattr(self, 'current_selected_var'): return
        var_name = self.current_selected_var
        
        # Fetch item info from workspace
        item_info = global_workspace._data.get(var_name)
        if not item_info: return
        
        data_type = item_info.get('type')
        
        # Prevent opening the exact same dataset twice; bring it to front instead
        if var_name in self.active_windows:
            items = self.window_tracker_list.findItems(var_name, Qt.MatchExactly)
            if items:
                self.bring_window_to_front(items[0])
            return

        viewer_widget = None

        # Universal Dispatch Logic
        if data_type == 'spectroscopy_tree':
            viewer_widget = DataViewerPanel()
            
            # Safely pull the data with fallbacks
            tensor_data = global_workspace.pull_tensor_data(var_name)
            if not tensor_data:
                tensor_data = global_workspace.pull_tensor_data(var_name, node="raw")
            if not tensor_data:
                tensor_data = global_workspace.pull_tensor_data(var_name, node="/raw")
                
            if tensor_data:
                viewer_widget.load_data(tensor_data)
        
        # Wrap, Track, and Launch
        if viewer_widget:
            # Pass `self` as the parent to prevent the macOS focus-loss bug
            wrapper = FloatingViewerWindow(win_id=var_name, title=var_name, inner_widget=viewer_widget, parent=self)
            wrapper.window_closed.connect(self.unregister_window)
            
            self.active_windows[var_name] = wrapper
            self.window_tracker_list.addItem(var_name)
            wrapper.show()
            
            # Crucial: Load data AFTER the widget is anchored to a window structure
            if data_type == 'spectroscopy_tree' and tensor_data:
                viewer_widget.load_data(tensor_data)

    def bring_window_to_front(self, item):
        """Raises a specific floating window to the top of the user's screen."""
        win_id = item.text()
        if win_id in self.active_windows:
            window = self.active_windows[win_id]
            window.showNormal()
            window.raise_()
            window.activateWindow()

    def unregister_window(self, win_id: str):
        """Removes a closed window from the tracking registry and UI list."""
        if win_id in self.active_windows:
            del self.active_windows[win_id]
            
        items = self.window_tracker_list.findItems(win_id, Qt.MatchExactly)
        if items:
            self.window_tracker_list.takeItem(self.window_tracker_list.row(items[0]))

    def launch_dft_suite(self):
        win_id = "DFT Suite"
        if win_id in self.active_windows:
            items = self.window_tracker_list.findItems(win_id, Qt.MatchExactly)
            if items: self.bring_window_to_front(items[0])
            return
            
        dft_widget = DFTSuite()
        wrapper = FloatingViewerWindow(win_id=win_id, title="TensorSpec - DFT Suite", inner_widget=dft_widget, parent=self)
        wrapper.window_closed.connect(self.unregister_window)
        wrapper.resize(900, 600)
        self.active_windows[win_id] = wrapper
        self.window_tracker_list.addItem(win_id)
        wrapper.show()

    def launch_arpes_suite(self):
        self.arpes_window = ARPESSuite()
        self.arpes_window.resize(1100, 700)
        self.arpes_window.setWindowTitle("TensorSpec - ARPES Suite")
        self.arpes_window.show()

    def _launch_panel_suite(
        self,
        win_id: str,
        title: str,
        module_path: str,
        class_name: str,
        size: tuple[int, int] = (1100, 750),
    ):
        """Open a tracked floating suite window; surface import/build errors in a dialog."""
        if win_id in self.active_windows:
            items = self.window_tracker_list.findItems(win_id, Qt.MatchExactly)
            if items:
                self.bring_window_to_front(items[0])
            return

        try:
            import importlib

            suite_cls = getattr(importlib.import_module(module_path), class_name)
            inner = suite_cls()
            wrapper = FloatingViewerWindow(
                win_id=win_id,
                title=title,
                inner_widget=inner,
                parent=None,
            )
            wrapper.setWindowTitle(title)
            wrapper.window_closed.connect(self.unregister_window)
            wrapper.resize(*size)
            self.active_windows[win_id] = wrapper
            self.window_tracker_list.addItem(win_id)
            wrapper.show()
            wrapper.raise_()
            wrapper.activateWindow()
        except Exception as exc:
            import traceback

            traceback.print_exc()
            QMessageBox.critical(
                self,
                f"Failed to launch {win_id}",
                f"{exc}\n\nSee terminal for full traceback.",
            )

    def launch_peem_suite(self):
        self._launch_panel_suite(
            "PEEM Suite",
            "TensorSpec - PEEM Suite",
            "tensorspec.gui.suites.peem_suite",
            "PEEMSuite",
            size=(1100, 750),
        )

    def launch_xas_suite(self):
        self._launch_panel_suite(
            "XAS Suite",
            "TensorSpec - XAS / XMCD Suite",
            "tensorspec.gui.suites.xas_suite",
            "XASSuite",
            size=(1000, 700),
        )

    def launch_raman_suite(self):
        self._launch_panel_suite(
            "Raman Suite",
            "TensorSpec - Raman Suite",
            "tensorspec.gui.suites.raman_suite",
            "RamanSuite",
            size=(1500, 960),  # original Raman Map Classifier window size
        )

    def launch_transport_suite(self):
        self._launch_panel_suite(
            "Transport Suite",
            "TensorSpec - Transport Suite",
            "tensorspec.gui.suites.transport_suite",
            "TransportSuite",
            size=(1000, 700),
        )

    def launch_compute_manager(self):
        try:
            from tensorspec.gui.components.compute_panel import ComputeManagerPanel
            self.compute_window = ComputeManagerPanel(self)
            self.compute_window.show()
        except Exception as e:
            print(f"Failed to launch Compute Manager: {e}")

    def launch_ml_suite(self):
        win_id = "ML Suite"
        if win_id in self.active_windows:
            items = self.window_tracker_list.findItems(win_id, Qt.MatchExactly)
            if items: self.bring_window_to_front(items[0])
            return

        print("Launching Machine Learning Suite Window...")
        try:
            from tensorspec.gui.suites.ml_suite import MLSuite
            ml_widget = MLSuite()
            wrapper = FloatingViewerWindow(
                win_id=win_id,
                title="TensorSpec - Machine Learning Suite",
                inner_widget=ml_widget,
                parent=self,
            )
            wrapper.window_closed.connect(self.unregister_window)
            wrapper.resize(1500, 900)
            self.active_windows[win_id] = wrapper
            self.window_tracker_list.addItem(win_id)
            wrapper.show()
        except Exception as e:
            print(f"Failed to launch ML Suite: {e}")
    
    def launch_crystal_suite(self):
        win_id = "Crystal Suite"
        if win_id in self.active_windows:
            items = self.window_tracker_list.findItems(win_id, Qt.MatchExactly)
            if items:
                self.bring_window_to_front(items[0])
            return

        crystal_widget = CrystalViewerSuite(workspace_manager=self.workspace_data)
        wrapper = FloatingViewerWindow(
            win_id=win_id,
            title="TensorSpec - Crystal Suite",
            inner_widget=crystal_widget,
            parent=None,
        )
        wrapper.setWindowTitle("TensorSpec - Crystal Suite")
        wrapper.window_closed.connect(self.unregister_window)
        wrapper.resize(1100, 700)
        wrapper.setMinimumSize(640, 480)
        self.active_windows[win_id] = wrapper
        self.window_tracker_list.addItem(win_id)
        self.crystal_window = wrapper
        wrapper.show()
        wrapper.raise_()
        wrapper.activateWindow()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TensorSpecMainBrowser()
    window.show()
    sys.exit(app.exec())