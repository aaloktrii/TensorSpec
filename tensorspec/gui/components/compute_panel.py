import os
import json
import time
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QFormLayout, QGroupBox, QAbstractItemView, QTabWidget, QWidget, QTextEdit,
    QApplication, QMainWindow, QSplitter, QFileDialog, QScrollArea, QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from tensorspec.core.compute.cluster_live_monitor import fetch_cluster_snapshot
from tensorspec.core.compute import cluster_paths as cp
from tensorspec.gui.services.sshproxy_auth import refresh_sshproxy_login

CONFIG_FILE = os.path.expanduser('~/.tensorspec_clusters.json')

class LiveMonitorThread(QThread):
    data_ready = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, cluster, log_jobs=None):
        super().__init__()
        self.cluster = cluster
        self.log_jobs = log_jobs
        self.running = True
        self.ssh = None

    def run(self):
        try:
            import paramiko
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            pwd = self.cluster.get('password', '')
            if not pwd: pwd = None
            cp.ssh_connect(self.ssh, self.cluster, timeout=15)
            
            while self.running:
                try:
                    # Fetch memory
                    stdin, stdout, stderr = self.ssh.exec_command("free -m", timeout=5)
                    mem_out = stdout.read().decode().strip().split('\n')
                    mem_used = 0
                    mem_total = 100
                    for line in mem_out:
                        if line.startswith("Mem:"):
                            parts = line.split()
                            mem_total = float(parts[1])
                            mem_used = float(parts[2])
                    
                    # Fetch CPU overall load (1 min avg) and scale to percentage for rough estimate
                    stdin, stdout, stderr = self.ssh.exec_command("top -b -n 1 | grep 'Cpu(s)'", timeout=5)
                    cpu_out = stdout.read().decode().strip()
                    cpu_usage = 0
                    if cpu_out:
                        parts = cpu_out.replace(',', ' ').split()
                        try:
                            if 'us' in parts:
                                idx = parts.index('us')
                                cpu_usage = float(parts[idx-1])
                            else:
                                cpu_usage = float(parts[1])
                        except Exception:
                            cpu_usage = 10.0

                    snap = fetch_cluster_snapshot(
                        self.ssh, self.cluster, log_jobs=self.log_jobs
                    )
                    text_info = snap["text_info"]
                    full_log_tail = snap.get("full_log_tail", "")
                    max_elapsed = snap.get("kkr_elapsed_seconds", 0)

                    self.data_ready.emit({
                        'cpu': min(100.0, cpu_usage * 2),
                        'mem_percent': (mem_used / mem_total) * 100 if mem_total > 0 else 0,
                        'mem_used': mem_used / 1024,
                        'mem_total': mem_total / 1024,
                        'text_info': text_info,
                        'full_log_tail': full_log_tail,
                        'dft_jobs_text': snap.get("dft_jobs_text", ""),
                        'kkr_elapsed_seconds': max_elapsed,
                        'my_gpu_count': snap.get("my_gpu_count", 0),
                        'other_gpu_users': snap.get("other_gpu_users", []),
                    })
                except Exception as e:
                    self.error_occurred.emit(f"Fetch loop error: {e}")
                
                # Sleep for 2.5 seconds before next poll
                for _ in range(12):
                    if not self.running: break
                    time.sleep(0.2)
                    
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if self.ssh:
                self.ssh.close()

    def stop(self):
        self.running = False


class LiveClusterMonitor(QMainWindow):
    def __init__(self, cluster, parent=None):
        super().__init__(None)
        self.cluster = cluster
        self.setWindowTitle(f"Live Task Manager: {cluster['name']} ({cluster['host']})")
        self.resize(1000, 750)
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        self.is_recording = False
        self.history_time = []
        self.history_cpu = []
        self.history_ram = []
        self.start_time = None
        
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        # --- Top Control Ribbon ---
        ctrl_layout = QHBoxLayout()
        
        self.status_lbl = QLabel(f"Connecting to {cluster['host']}... Please wait.")
        self.status_lbl.setStyleSheet("font-weight: bold; color: orange; font-size: 14px;")
        ctrl_layout.addWidget(self.status_lbl)
        
        ctrl_layout.addStretch()
        
        self.btn_record = QPushButton("🔴 Start Recording")
        self.btn_record.setStyleSheet("font-weight: bold; background-color: #d9534f; color: white; padding: 6px;")
        self.btn_record.clicked.connect(self.toggle_recording)
        ctrl_layout.addWidget(self.btn_record)
        
        self.btn_save = QPushButton("💾 Save Recorded Graph")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_graph)
        ctrl_layout.addWidget(self.btn_save)
        
        layout.addLayout(ctrl_layout)
        
        # Tabs for Hardware vs Logs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # TAB 1: Hardware Performance
        self.tab_hw = QWidget()
        hw_layout = QVBoxLayout(self.tab_hw)
        
        splitter = QSplitter(Qt.Vertical)
        
        # Top: Graphs
        self.fig = Figure(figsize=(8, 3), dpi=100, layout='tight')
        self.canvas = FigureCanvas(self.fig)
        splitter.addWidget(self.canvas)
        
        self.ax_cpu = self.fig.add_subplot(121)
        self.ax_ram = self.fig.add_subplot(122)
        self.ax_cpu.set_title("CPU Load (Waiting for Recording...)")
        self.ax_ram.set_title("RAM Usage (Waiting for Recording...)")
        
        # Bottom: Text feed
        self.text_feed = QTextEdit()
        self.text_feed.setReadOnly(True)
        self.text_feed.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: monospace; font-size: 13px;")
        splitter.addWidget(self.text_feed)
        
        splitter.setSizes([350, 400])
        hw_layout.addWidget(splitter)
        self.tabs.addTab(self.tab_hw, "📊 Hardware & Tasks")
        
        # TAB 2: DFT Live Log Viewer
        self.tab_logs = QWidget()
        logs_layout = QVBoxLayout(self.tab_logs)
        
        self.timer_lbl = QLabel("⏱ Calculation Elapsed Time: Not Running")
        self.timer_lbl.setStyleSheet("font-weight: bold; color: #007bff; font-size: 18px;")
        logs_layout.addWidget(self.timer_lbl)
        
        self.log_feed = QTextEdit()
        self.log_feed.setReadOnly(True)
        self.log_feed.setLineWrapMode(QTextEdit.NoWrap)
        self.log_feed.setStyleSheet("background-color: #1e1e1e; color: #ffffff; font-family: monospace; font-size: 14px;")
        logs_layout.addWidget(self.log_feed)
        
        self.tabs.addTab(self.tab_logs, "📜 Calculation Live Logs")
        
        self.monitor_thread = LiveMonitorThread(self.cluster, log_jobs=None)
        self.monitor_thread.data_ready.connect(self.update_data)
        self.monitor_thread.error_occurred.connect(self.handle_error)
        self.monitor_thread.start()

    def toggle_recording(self):
        if not self.is_recording:
            # Start Recording
            self.is_recording = True
            self.history_time.clear()
            self.history_cpu.clear()
            self.history_ram.clear()
            self.start_time = time.time()
            
            self.btn_record.setText("⏹ Stop Recording")
            self.btn_record.setStyleSheet("font-weight: bold; background-color: #5bc0de; color: black; padding: 6px;")
            self.btn_save.setEnabled(False)
            self.status_lbl.setText(f"🔴 Recording {self.cluster['host']} Live!")
            self.status_lbl.setStyleSheet("font-weight: bold; color: #d9534f; font-size: 14px;")
        else:
            # Stop Recording
            self.is_recording = False
            self.btn_record.setText("🔴 Start Recording")
            self.btn_record.setStyleSheet("font-weight: bold; background-color: #d9534f; color: white; padding: 6px;")
            self.btn_save.setEnabled(len(self.history_time) > 0)
            self.btn_save.setStyleSheet("font-weight: bold; background-color: #007bff; color: white; padding: 6px;")
            self.status_lbl.setText("Recording Stopped. Graph is frozen.")
            self.status_lbl.setStyleSheet("font-weight: bold; color: gray; font-size: 14px;")

    def update_data(self, data):
        # Text always updates regardless of recording state
        self.text_feed.setText(data['text_info'])
        self.log_feed.setText(data.get('full_log_tail', ''))
        
        # Update timer / task summary
        elapsed_sec = data.get('kkr_elapsed_seconds', 0)
        my_gpus = data.get('my_gpu_count', 0)
        others = data.get('other_gpu_users', [])
        if elapsed_sec > 0:
            m, s = divmod(elapsed_sec, 60)
            h, m = divmod(m, 60)
            extra = f" | your GPUs in use: {my_gpus}" if my_gpus else ""
            if others:
                extra += f" | other GPU users: {', '.join(others)}"
            self.timer_lbl.setText(
                f"⏱ Active job: {h:02d}:{m:02d}:{s:02d}{extra}"
            )
            self.timer_lbl.setStyleSheet("font-weight: bold; color: #28a745; font-size: 18px;")
        elif others:
            self.timer_lbl.setText(
                f"⏱ No your jobs — other GPU users: {', '.join(others)}"
            )
            self.timer_lbl.setStyleSheet("font-weight: bold; color: #e67e22; font-size: 18px;")
        else:
            self.timer_lbl.setText("⏱ No active TensorSpec / GPU jobs detected")
            self.timer_lbl.setStyleSheet("font-weight: bold; color: #007bff; font-size: 18px;")
        
        if not self.is_recording:
            if not self.status_lbl.text().startswith("Recording Stopped") and not self.status_lbl.text().startswith("🔴 Error"):
                self.status_lbl.setText(f"🟢 Connected to {self.cluster['host']}. (Not Recording)")
                self.status_lbl.setStyleSheet("font-weight: bold; color: #28a745; font-size: 14px;")
            return
            
        cpu = data['cpu']
        mem_pct = data['mem_percent']
        mem_u = data['mem_used']
        mem_t = data['mem_total']
        
        elapsed = time.time() - self.start_time
        self.history_time.append(elapsed)
        self.history_cpu.append(cpu)
        self.history_ram.append(mem_pct)
        
        # Keep memory bounds reasonable so it doesn't crash from out of memory
        if len(self.history_time) > 1000:
            self.history_time.pop(0)
            self.history_cpu.pop(0)
            self.history_ram.pop(0)
            
        self.ax_cpu.clear()
        self.ax_ram.clear()
        
        self.ax_cpu.plot(self.history_time, self.history_cpu, color='#5bc0de', linewidth=3)
        self.ax_cpu.fill_between(self.history_time, self.history_cpu, color='#5bc0de', alpha=0.3)
        self.ax_cpu.set_ylim(0, 100)
        self.ax_cpu.set_title(f"CPU Load ({cpu:.1f}%)")
        self.ax_cpu.set_xlabel("Time (s)")
        
        self.ax_ram.plot(self.history_time, self.history_ram, color='#d9534f', linewidth=3)
        self.ax_ram.fill_between(self.history_time, self.history_ram, color='#d9534f', alpha=0.3)
        self.ax_ram.set_ylim(0, 100)
        self.ax_ram.set_title(f"RAM Usage: {mem_u:.1f} GB / {mem_t:.1f} GB")
        self.ax_ram.set_xlabel("Time (s)")
        
        self.canvas.draw()

    def save_graph(self):
        if not self.history_time: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Cluster Telemetry Graph", "cluster_stress_test.png", "PNG Images (*.png);;PDF Files (*.pdf)")
        if not path: return
        
        try:
            export_fig = Figure(figsize=(8, 5), dpi=150, layout='tight')
            ax = export_fig.add_subplot(111)
            
            ax.plot(self.history_time, self.history_cpu, label='Total CPU Usage (%)', color='#5bc0de', linewidth=2.5)
            ax.plot(self.history_time, self.history_ram, label='RAM Usage (%)', color='#d9534f', linewidth=2.5)
            
            ax.set_title(f"Hardware Stress Test on {self.cluster['host']}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Elapsed Time (Seconds)", fontsize=12)
            ax.set_ylabel("System Utilization (%)", fontsize=12)
            ax.set_ylim(0, 105)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper left', fontsize=11)
            
            export_fig.savefig(path)
            QMessageBox.information(self, "Export Successful", f"Time-series graph saved successfully to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save telemetry graph:\n{str(e)}")

    def handle_error(self, err):
        self.status_lbl.setText(f"🔴 Error: {err}")
        self.status_lbl.setStyleSheet("font-weight: bold; color: red; font-size: 14px;")

    def closeEvent(self, event):
        self.is_recording = False
        self.monitor_thread.stop()
        self.monitor_thread.wait(2000)
        super().closeEvent(event)


class StatusFetchThread(QThread):
    result_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, cluster):
        super().__init__()
        self.cluster = cluster

    def run(self):
        try:
            import paramiko
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            pwd = self.cluster.get('password', '')
            if not pwd:
                pwd = None
            cp.ssh_connect(ssh, self.cluster, timeout=15)
            
            output_text = f"=== Status for {self.cluster['host']} ===\n\n"
            
            stdin, stdout, stderr = ssh.exec_command("df -h / /mnt/data", timeout=10)
            output_text += "[ Disk Space ]\n" + stdout.read().decode() + "\n"
            
            stdin, stdout, stderr = ssh.exec_command("free -h", timeout=10)
            output_text += "[ Memory Usage ]\n" + stdout.read().decode() + "\n"
            
            stdin, stdout, stderr = ssh.exec_command("uptime", timeout=10)
            output_text += "[ System Uptime & Load ]\n" + stdout.read().decode() + "\n"

            snap = fetch_cluster_snapshot(ssh, self.cluster)
            output_text += snap["text_info"]
            if snap.get("full_log_tail"):
                output_text += "\n" + snap["full_log_tail"] + "\n"
                
            ssh.close()
            self.result_ready.emit(output_text)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class ComputeManagerPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compute Manager")
        self.setMinimumSize(640, 480)
        self.resize(860, 680)
        self.setSizeGripEnabled(True)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        
        self.clusters = []
        self.fetch_thread = None
        self.live_monitor = None
        self._init_ui()
        self.load_config()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Tab 1: Connections
        self.tab_connections = QWidget()
        self._init_connections_tab()
        self.tabs.addTab(self.tab_connections, "Cluster Connections")
        
        # Tab 2: Remote Cluster Status
        self.tab_status = QWidget()
        self._init_status_tab()
        self.tabs.addTab(self.tab_status, "Remote Cluster Status")
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        main_layout.addWidget(close_btn, alignment=Qt.AlignRight)
        
    def _init_connections_tab(self):
        outer = QVBoxLayout(self.tab_connections)
        outer.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QScrollArea.NoFrame)

        scroll_body = QWidget()
        layout = QVBoxLayout(scroll_body)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Form Area ---
        form_group = QGroupBox("Add New Cluster")
        form_layout = QFormLayout()
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. Production Cluster")
        self.host_input = QLineEdit()
        self.host_input.setPlaceholderText("e.g. 192.168.1.100 or cluster.local")
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Username")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Daemon", "SLURM"])

        self.heavy_root_input = QLineEdit()
        self.heavy_root_input.setPlaceholderText("optional — default /mnt/data/{user}/tensorspec_heavy")
        self.tmp_dir_input = QLineEdit()
        self.tmp_dir_input.setPlaceholderText("optional — default /mnt/data/{user}/tmp")
        self.repo_root_input = QLineEdit()
        self.repo_root_input.setPlaceholderText("optional — default /home/{user}/TensorSpec")
        self.python_input = QLineEdit()
        self.python_input.setPlaceholderText("optional — default {repo}/TensorSpec_env/bin/python")
        self.slurm_account_input = QLineEdit()
        self.slurm_account_input.setPlaceholderText("SLURM only — e.g. m1234")
        self.slurm_qos_input = QLineEdit()
        self.slurm_qos_input.setPlaceholderText("SLURM only — e.g. regular (not regular_0 on als)")
        self.slurm_constraint_input = QLineEdit()
        self.slurm_constraint_input.setPlaceholderText("SLURM only — cpu or gpu")
        self.slurm_walltime_input = QLineEdit()
        self.slurm_walltime_input.setPlaceholderText("SLURM only — HH:MM:SS e.g. 06:00:00")
        self.qe_module_input = QLineEdit()
        self.qe_module_input.setPlaceholderText("optional — e.g. espresso/7.5-libxc-7.0.0-cpu")
        self.ssh_key_input = QLineEdit()
        self.ssh_key_input.setPlaceholderText("optional — ~/.ssh/key (passwordless SSH)")
        
        form_layout.addRow("Name:", self.name_input)
        form_layout.addRow("Host:", self.host_input)
        form_layout.addRow("User:", self.user_input)
        form_layout.addRow("Password:", self.password_input)
        form_layout.addRow("Mode:", self.mode_combo)

        paths_group = QGroupBox("Remote paths (optional — see docs/REMOTE_GPU_SETUP.md)")
        paths_form = QFormLayout()
        paths_form.addRow("heavy_root:", self.heavy_root_input)
        paths_form.addRow("tmp_dir:", self.tmp_dir_input)
        paths_form.addRow("repo_root:", self.repo_root_input)
        paths_form.addRow("python:", self.python_input)
        paths_group.setLayout(paths_form)
        form_layout.addRow(paths_group)

        slurm_group = QGroupBox("Slurm (SLURM mode only)")
        slurm_form = QFormLayout()
        slurm_form.addRow("account:", self.slurm_account_input)
        slurm_form.addRow("qos:", self.slurm_qos_input)
        slurm_form.addRow("constraint:", self.slurm_constraint_input)
        slurm_form.addRow("walltime:", self.slurm_walltime_input)
        slurm_form.addRow("qe module:", self.qe_module_input)
        slurm_form.addRow("ssh key:", self.ssh_key_input)
        slurm_group.setLayout(slurm_form)
        form_layout.addRow(slurm_group)
        
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add Cluster")
        self.add_btn.clicked.connect(self.add_cluster)
        btn_layout.addStretch()
        btn_layout.addWidget(self.add_btn)
        
        form_layout.addRow(btn_layout)
        form_group.setLayout(form_layout)
        layout.addWidget(form_group)

        scroll.setWidget(scroll_body)
        outer.addWidget(scroll, stretch=1)
        
        # --- Table Area (always visible below scroll) ---
        table_group = QGroupBox("Configured Clusters")
        table_layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Name", "Host", "User", "Mode"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        
        table_layout.addWidget(self.table)
        
        action_layout = QHBoxLayout()
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self.remove_cluster)
        
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self.test_connection)

        self.sshproxy_login_btn = QPushButton("🔑 Refresh sshproxy Login")
        self.sshproxy_login_btn.setToolTip(
            "For sshproxy clusters only (auth: sshproxy) — runs sshproxy (MFA). "
            "Hidden / disabled for Daemon / password-style clusters."
        )
        self.sshproxy_login_btn.clicked.connect(self.refresh_sshproxy_login)

        self.provision_btn = QPushButton("⚙️ Auto-Setup Remote Environment")
        self.provision_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 5px;")
        self.provision_btn.clicked.connect(self.provision_cluster_environment)
        
        action_layout.addWidget(self.remove_btn)
        action_layout.addStretch()
        action_layout.addWidget(self.sshproxy_login_btn)
        action_layout.addWidget(self.test_btn)
        action_layout.addWidget(self.provision_btn)
        
        table_layout.addLayout(action_layout)
        table_group.setLayout(table_layout)
        outer.addWidget(table_group)

    def _init_status_tab(self):
        layout = QVBoxLayout(self.tab_status)
        
        ctrl_layout = QHBoxLayout()
        self.combo_cluster_status = QComboBox()
        self.btn_refresh_status = QPushButton("🔄 Fetch Static Text Report")
        self.btn_refresh_status.clicked.connect(self.refresh_cluster_status)
        
        self.btn_live_graph = QPushButton("📈 Launch Live Graphical Monitor")
        self.btn_live_graph.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
        self.btn_live_graph.clicked.connect(self.launch_live_monitor)
        
        ctrl_layout.addWidget(QLabel("Select Cluster:"))
        ctrl_layout.addWidget(self.combo_cluster_status)
        ctrl_layout.addWidget(self.btn_refresh_status)
        ctrl_layout.addWidget(self.btn_live_graph)
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)
        
        # Hardware utilization and running processes
        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)
        self.status_display.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: monospace; font-size: 13px;")
        layout.addWidget(self.status_display)

    def launch_live_monitor(self):
        cluster_name = self.combo_cluster_status.currentText()
        if not cluster_name: return
        cluster = next((c for c in self.clusters if c['name'] == cluster_name), None)
        if not cluster: return
        
        if self.live_monitor:
            try:
                self.live_monitor.close()
            except RuntimeError:
                pass
            
        self.live_monitor = LiveClusterMonitor(cluster, parent=None)
        self.live_monitor.show()

    def add_cluster(self):
        name = self.name_input.text().strip()
        host = self.host_input.text().strip()
        user = self.user_input.text().strip()
        password = self.password_input.text().strip()
        mode = self.mode_combo.currentText()
        
        if not name or not host or not user:
            QMessageBox.warning(self, "Validation Error", "Name, Host, and User are required.")
            return
            
        cluster_data = {
            "name": name,
            "host": host,
            "user": user,
            "password": password,
            "mode": mode,
        }
        paths = {}
        for key, widget in (
            ("heavy_root", self.heavy_root_input),
            ("tmp_dir", self.tmp_dir_input),
            ("repo_root", self.repo_root_input),
            ("python", self.python_input),
            ("qe_module", self.qe_module_input),
            ("ssh_key", self.ssh_key_input),
        ):
            val = widget.text().strip()
            if val:
                paths[key] = val
        slurm = {}
        for key, widget in (
            ("account", self.slurm_account_input),
            ("qos", self.slurm_qos_input),
            ("constraint", self.slurm_constraint_input),
            ("walltime", self.slurm_walltime_input),
        ):
            val = widget.text().strip()
            if val:
                slurm[key] = val
        if slurm:
            paths["slurm"] = slurm
        if paths:
            cluster_data["paths"] = paths
        
        self.clusters.append(cluster_data)
        self.update_table()
        self.save_config()
        self.clear_inputs()

    def remove_cluster(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return
            
        # Remove from bottom to top
        for index in sorted([r.row() for r in selected_rows], reverse=True):
            del self.clusters[index]
            
        self.update_table()
        self.save_config()

    def refresh_sshproxy_login(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(
                self,
                "Info",
                "Select an sshproxy cluster in the table first.",
            )
            return
        row = selected_rows[0].row()
        cluster = self.clusters[row]
        if not cp.uses_sshproxy(cluster):
            QMessageBox.information(
                self,
                "Not sshproxy",
                f"'{cluster.get('name')}' is not an sshproxy cluster.\n"
                "Use Test Connection for Daemon / password hosts.",
            )
            return
        refresh_sshproxy_login(self, cluster)

    def test_connection(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "Info", "Please select a cluster to test.")
            return
            
        row = selected_rows[0].row()
        cluster = self.clusters[row]
        try:
            import paramiko
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cp.ssh_connect(ssh, cluster, timeout=8)
            
            # Quick check for python / slurm
            stdin, stdout, stderr = ssh.exec_command("which python3 sbatch", timeout=5)
            detected = stdout.read().decode().strip().split('\n')
            detected_str = ", ".join([os.path.basename(p) for p in detected if p])
            ssh.close()
            QMessageBox.information(self, "Connection Successful", f"Connected to '{cluster['name']}' ({cluster['host']})!\nDetected on cluster: {detected_str or 'Basic shell'}")
        except Exception as e:
            hint = ""
            if cp.uses_sshproxy(cluster):
                hint = "\n\nTip: click 🔑 Refresh sshproxy Login first (MFA)."
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to connect to '{cluster['name']}':\n{str(e)}{hint}",
            )

    def provision_cluster_environment(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "Info", "Please select a cluster from the table to set up.")
            return
            
        row = selected_rows[0].row()
        cluster = self.clusters[row]
        
        reply = QMessageBox.question(
            self,
            "Auto-Setup Remote Server",
            f"Would you like TensorSpec to automatically verify and set up directories and Python environments on '{cluster['name']}'?\n\nThis will:\n- Create `{cp.heavy_root(cluster)}` and job subdirs\n- Create `{cp.tmp_dir(cluster)}`\n- Verify or create Python venv at `{cp.repo_root(cluster)}`\n- Install numpy/scipy/matplotlib in the remote venv",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
            
        try:
            import paramiko
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cp.ssh_connect(ssh, cluster, timeout=10)
            
            setup_cmd = cp.provision_script(cluster)
            stdin, stdout, stderr = ssh.exec_command(setup_cmd, timeout=60)
            out = stdout.read().decode()
            ssh.close()
            
            if "SETUP_COMPLETE" in out:
                cluster['provisioned'] = True
                self._save_clusters()
                QMessageBox.information(self, "Setup Complete", f"✅ Cluster '{cluster['name']}' is now 100% configured for TensorSpec calculations!")
            else:
                QMessageBox.warning(self, "Setup Notice", f"Setup executed with response:\n{out}\n{stderr.read().decode()}")
        except Exception as e:
            QMessageBox.critical(self, "Setup Error", f"Failed to auto-provision cluster:\n{str(e)}")

    def refresh_cluster_status(self):
        cluster_name = self.combo_cluster_status.currentText()
        if not cluster_name:
            self.status_display.setText("No cluster selected.")
            return
            
        cluster = next((c for c in self.clusters if c['name'] == cluster_name), None)
        if not cluster:
            self.status_display.setText("Cluster config not found.")
            return
            
        self.btn_refresh_status.setEnabled(False)
        self.btn_refresh_status.setText("Connecting...")
        self.status_display.setText(
            "Connecting and fetching status in the background...\n"
            "Please wait — remote clusters can be slow to respond (up to 30s).\n"
        )
        
        self.fetch_thread = StatusFetchThread(cluster)
        self.fetch_thread.result_ready.connect(self._on_status_success)
        self.fetch_thread.error_occurred.connect(self._on_status_error)
        self.fetch_thread.start()

    def _on_status_success(self, text):
        self.status_display.setText(text)
        self.btn_refresh_status.setEnabled(True)
        self.btn_refresh_status.setText("🔄 Fetch Static Text Report")

    def _on_status_error(self, err_text):
        self.status_display.setText(f"Error fetching status:\n{err_text}")
        self.btn_refresh_status.setEnabled(True)
        self.btn_refresh_status.setText("🔄 Fetch Static Text Report")

    def clear_inputs(self):
        self.name_input.clear()
        self.host_input.clear()
        self.user_input.clear()
        self.password_input.clear()
        self.mode_combo.setCurrentIndex(0)
        self.heavy_root_input.clear()
        self.tmp_dir_input.clear()
        self.repo_root_input.clear()
        self.python_input.clear()

    def update_table(self):
        self.table.setRowCount(0)
        
        self.combo_cluster_status.clear()
        
        for row, cluster in enumerate(self.clusters):
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(cluster.get("name", "")))
            self.table.setItem(row, 1, QTableWidgetItem(cluster.get("host", "")))
            self.table.setItem(row, 2, QTableWidgetItem(cluster.get("user", "")))
            self.table.setItem(row, 3, QTableWidgetItem(cluster.get("mode", "")))
            
            self.combo_cluster_status.addItem(cluster.get("name", ""))

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    self.clusters = json.load(f)
                self.update_table()
            except Exception as e:
                print(f"Error loading cluster config: {e}")

    def save_config(self):
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.clusters, f, indent=4)
        except Exception as e:
            print(f"Error saving cluster config: {e}")
