"""Machine Learning suite: data browser, shared N-D viewer, ML panels."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from tensorspec.gui.components.data_viewer_panel import DataViewerPanel
from tensorspec.gui.components.ml_tabs.active_learning_panel import ActiveLearningPanel
from tensorspec.gui.components.ml_tabs.alignment_panel import AlignmentPanel
from tensorspec.gui.components.ml_tabs.cluster_panel import ClusterPanel
from tensorspec.gui.components.ml_tabs.data_browser_panel import DataBrowserPanel
from tensorspec.gui.components.ml_tabs.layout import scrollable, tab_group
from tensorspec.gui.components.ml_tabs.simulate_al_panel import SimulateALPanel
from tensorspec.gui.components.ml_tabs.ssl_panel import SSLTrainingPanel
from tensorspec.gui.components.ml_tabs.supervised_panel import SupervisedPanel
from tensorspec.gui.ml.build_pipeline_tab import BuildPipelineTab
from tensorspec.gui.ml.model_warehouse_tab import ModelWarehouseTab
from tensorspec.gui.ml.train_model_tab import TrainModelTab
from tensorspec.gui.ml.session import MLSession


class MLSuite(QWidget):
    """Machine Learning suite: data browser, shared N-D viewer, ML panels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(1000, 620)
        self.session = MLSession()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.viewer = DataViewerPanel()
        self.session.viewer = self.viewer

        self.browser_panel = DataBrowserPanel(self.session)
        self.ssl_panel = SSLTrainingPanel(self.session)
        self.cluster_panel = ClusterPanel(self.session)
        self.supervised_panel = SupervisedPanel(self.session)
        self.al_panel = ActiveLearningPanel(self.session)
        self.sim_panel = SimulateALPanel(self.session)
        self.alignment_panel = AlignmentPanel(self.session)

        self.model_warehouse_tab = ModelWarehouseTab(self)
        self.build_pipeline_tab = BuildPipelineTab(self)
        self.train_model_tab = TrainModelTab(self)

        mid_panel = QWidget()
        mid_layout = QVBoxLayout(mid_panel)
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.addWidget(self.viewer)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setMovable(True)
        tabs.setUsesScrollButtons(True)
        tabs.setElideMode(Qt.TextElideMode.ElideNone)
        for group_label, pages in (
            ("Train", [
                ("SSL Training", self.ssl_panel),
                ("Supervised Learning", self.supervised_panel),
            ]),
            ("Cluster", [("Clustering", self.cluster_panel)]),
            ("Align", [("3D Alignment", self.alignment_panel)]),
            ("Steer", [
                ("Active Learning", self.al_panel),
                ("Simulate AL", self.sim_panel),
            ]),
            ("Models", [
                ("Model Warehouse", self.model_warehouse_tab),
                ("Build Pipeline", scrollable(self.build_pipeline_tab)),
                ("Train Model", self.train_model_tab),
            ]),
        ):
            tabs.addTab(tab_group(pages), group_label)
        tabs.setCurrentIndex(0)

        self.browser_panel.setMinimumWidth(220)
        mid_panel.setMinimumWidth(320)
        tabs.setMinimumWidth(380)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.browser_panel)
        splitter.addWidget(mid_panel)
        splitter.addWidget(tabs)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([260, 680, 560])
        layout.addWidget(splitter, 1)

        status_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.prog_bar = QProgressBar()
        self.prog_bar.setVisible(False)
        self.prog_bar.setMaximumWidth(220)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.prog_bar)
        layout.addLayout(status_row)

        self.session.status_changed.connect(self._on_status)

    def _on_status(self, value, message):
        if message:
            self.status_label.setText(message)
        self.prog_bar.setVisible(0 < value < 100)
        self.prog_bar.setValue(value)

    # Compatibility shims used by older tests / context helpers.
    @property
    def workspace(self):
        return self.session.workspace

    @property
    def current_view_data(self):
        return self.session.current_view_data

    def activate_data(self, item):
        self.browser_panel.activate_data(item)
