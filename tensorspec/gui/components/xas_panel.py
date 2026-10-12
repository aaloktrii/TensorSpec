"""1D XAS / XMCD panel — shared PEEM BG and sum-rule core."""
from __future__ import annotations

import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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

from tensorspec.gui.services.xas_service import XasService


class XasPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.service = XasService()
        self._dataset_name: str | None = None
        self._meta: dict | None = None
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        load_group = QGroupBox("1. Load")
        load_form = QFormLayout(load_group)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("From source name")
        load_form.addRow("Dataset name:", self.name_edit)
        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load CSV / TXT…")
        self.btn_load.clicked.connect(self._load_single)
        self.btn_load_pair = QPushButton("Load CP + CM files…")
        self.btn_load_pair.clicked.connect(self._load_pair)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_load_pair)
        load_form.addRow(btn_row)
        self.status_label = QLabel("No spectra loaded")
        load_form.addRow(self.status_label)
        control_layout.addWidget(load_group)

        bg_group = QGroupBox("2. Background (PEEM core)")
        bg_form = QFormLayout(bg_group)
        self.bg_node = QComboBox()
        self.bg_node.addItem("Raw", "raw")
        self.bg_node.addItem("Processed / paired", "processed")
        bg_form.addRow("Source:", self.bg_node)
        self.bg_channel = QComboBox()
        self.bg_channel.addItem("Channel 0", 0)
        self.bg_channel.addItem("Channel 1", 1)
        bg_form.addRow("Channel:", self.bg_channel)
        self.bg_method = QComboBox()
        self.bg_method.addItem("Linear pre-edge", "linear")
        self.bg_method.addItem("Two-step pre+post", "two_step")
        self.bg_method.currentIndexChanged.connect(self._sync_bg_method_ui)
        bg_form.addRow("Method:", self.bg_method)
        self.bg_e0 = QDoubleSpinBox()
        self.bg_e1 = QDoubleSpinBox()
        for spin in (self.bg_e0, self.bg_e1):
            spin.setRange(-1e4, 1e4)
            spin.setDecimals(4)
        bg_form.addRow("Pre-edge e0 / e1:", self._pair_row(self.bg_e0, self.bg_e1))
        self.bg_post_row_label = QLabel("Post-edge e0 / e1:")
        self.bg_post_e0 = QDoubleSpinBox()
        self.bg_post_e1 = QDoubleSpinBox()
        for spin in (self.bg_post_e0, self.bg_post_e1):
            spin.setRange(-1e4, 1e4)
            spin.setDecimals(4)
        self.bg_post_row = self._pair_row(self.bg_post_e0, self.bg_post_e1)
        bg_form.addRow(self.bg_post_row_label, self.bg_post_row)
        bg_btns = QHBoxLayout()
        self.btn_bg_preview = QPushButton("Preview Background")
        self.btn_bg_apply = QPushButton("Apply Background")
        self.btn_bg_preview.clicked.connect(self._bg_preview)
        self.btn_bg_apply.clicked.connect(self._bg_apply)
        bg_btns.addWidget(self.btn_bg_preview)
        bg_btns.addWidget(self.btn_bg_apply)
        bg_form.addRow(bg_btns)
        self.bg_figure = Figure(figsize=(3, 2), dpi=90, layout="tight")
        self.bg_canvas = FigureCanvas(self.bg_figure)
        self.bg_ax = self.bg_figure.add_subplot(111)
        bg_form.addRow(self.bg_canvas)
        control_layout.addWidget(bg_group)

        sum_group = QGroupBox("3. XMCD Sum Rule (PEEM core)")
        sum_form = QFormLayout(sum_group)
        self.sum_nh = QDoubleSpinBox()
        self.sum_nh.setRange(0.001, 1e6)
        self.sum_nh.setValue(1.0)
        sum_form.addRow("n_h:", self.sum_nh)
        self.sum_l3_lo = QDoubleSpinBox()
        self.sum_l3_hi = QDoubleSpinBox()
        self.sum_l2_lo = QDoubleSpinBox()
        self.sum_l2_hi = QDoubleSpinBox()
        self.sum_r_lo = QDoubleSpinBox()
        self.sum_r_hi = QDoubleSpinBox()
        for spin in (
            self.sum_l3_lo,
            self.sum_l3_hi,
            self.sum_l2_lo,
            self.sum_l2_hi,
            self.sum_r_lo,
            self.sum_r_hi,
        ):
            spin.setRange(-1e4, 1e4)
            spin.setDecimals(4)
        sum_form.addRow("L3 lo / hi:", self._pair_row(self.sum_l3_lo, self.sum_l3_hi))
        sum_form.addRow("L2 lo / hi:", self._pair_row(self.sum_l2_lo, self.sum_l2_hi))
        sum_form.addRow("r lo / hi:", self._pair_row(self.sum_r_lo, self.sum_r_hi))
        sum_btns = QHBoxLayout()
        self.btn_sum_preview = QPushButton("Preview Sum Rule")
        self.btn_sum_apply = QPushButton("Apply Sum Rule")
        self.btn_sum_preview.clicked.connect(self._sumrule_preview)
        self.btn_sum_apply.clicked.connect(self._sumrule_apply)
        sum_btns.addWidget(self.btn_sum_preview)
        sum_btns.addWidget(self.btn_sum_apply)
        sum_form.addRow(sum_btns)
        self.sum_results = QLabel("(no sum-rule result yet)")
        self.sum_results.setWordWrap(True)
        sum_form.addRow(self.sum_results)
        self.sum_figure = Figure(figsize=(3, 2), dpi=90, layout="tight")
        self.sum_canvas = FigureCanvas(self.sum_figure)
        self.sum_ax = self.sum_figure.add_subplot(111)
        sum_form.addRow(self.sum_canvas)
        control_layout.addWidget(sum_group)

        control_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(control_panel)
        scroll.setMinimumWidth(360)
        splitter.addWidget(scroll)

        view_panel = QWidget()
        view_layout = QVBoxLayout(view_panel)
        self.plot_node = QComboBox()
        self.plot_node.currentIndexChanged.connect(self._refresh_main_plot)
        view_layout.addWidget(QLabel("Main plot source:"))
        view_layout.addWidget(self.plot_node)
        self.main_figure = Figure(figsize=(6, 4), dpi=100, layout="tight")
        self.main_canvas = FigureCanvas(self.main_figure)
        self.main_ax = self.main_figure.add_subplot(111)
        view_layout.addWidget(self.main_canvas, stretch=1)
        splitter.addWidget(view_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 620])

        self._sync_bg_method_ui()

    @staticmethod
    def _pair_row(a, b):
        row = QHBoxLayout()
        row.addWidget(a)
        row.addWidget(b)
        wrap = QWidget()
        wrap.setLayout(row)
        return wrap

    def _sync_bg_method_ui(self):
        two = self.bg_method.currentData() == "two_step"
        self.bg_post_row_label.setVisible(two)
        self.bg_post_row.setVisible(two)

    def _require_dataset(self) -> str | None:
        if not self._dataset_name:
            QMessageBox.warning(self, "No data", "Load XAS spectra first.")
            return None
        return self._dataset_name

    def _load_single(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load XAS spectrum", "", "Spectra (*.csv *.txt)")
        if not path:
            return
        name = self.name_edit.text().strip() or None
        if name is None:
            base = os.path.splitext(os.path.basename(path))[0]
            self.name_edit.setText(base.replace(" ", "_"))
            name = self.name_edit.text().strip()
        try:
            meta = self.service.load_path(path, name=name)
            self._on_loaded(meta)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))

    def _load_pair(self):
        plus, _ = QFileDialog.getOpenFileName(self, "Load plus (CP) spectrum", "", "Spectra (*.csv *.txt)")
        if not plus:
            return
        minus, _ = QFileDialog.getOpenFileName(self, "Load minus (CM) spectrum", "", "Spectra (*.csv *.txt)")
        if not minus:
            return
        name = self.name_edit.text().strip() or None
        try:
            meta = self.service.load_pair(plus, minus, name=name)
            self._on_loaded(meta)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))

    def _on_loaded(self, meta: dict):
        self._dataset_name = meta["name"]
        self._meta = meta
        self.name_edit.setText(meta["name"])
        e_span = meta.get("n_points", 0)
        parts = [f"Loaded '{meta['name']}'", f"{e_span} points"]
        if meta.get("paired"):
            parts.append(f"paired {meta.get('channel_tags', [])}")
        if meta.get("I0_present"):
            parts.append("I0 in metadata")
        self.status_label.setText(" | ".join(parts))
        self._refresh_node_lists()
        self._refresh_main_plot()
        if meta.get("paired"):
            e_max = float(meta.get("n_points", 10))
            self.sum_l3_hi.setValue(max(1.0, e_max * 0.25))
            self.sum_l2_hi.setValue(max(2.0, e_max * 0.5))
            self.sum_r_hi.setValue(max(4.0, e_max * 0.9))

    def _refresh_node_lists(self):
        self.plot_node.blockSignals(True)
        self.bg_node.blockSignals(True)
        self.plot_node.clear()
        self.bg_node.clear()
        self.bg_node.addItem("Raw", "raw")
        self.plot_node.addItem("Raw", "raw")
        if self._meta and self._meta.get("has_processed"):
            self.bg_node.addItem("Processed / paired", "processed")
            self.plot_node.addItem("Processed / paired", "processed")
        for tag in (self._meta or {}).get("processed_children", []):
            node = f"processed/{tag}"
            self.plot_node.addItem(node, node)
            self.bg_node.addItem(node, node)
        self.plot_node.blockSignals(False)
        self.bg_node.blockSignals(False)
        paired = bool(self._meta and self._meta.get("paired"))
        self.bg_channel.setEnabled(paired)

    def _refresh_main_plot(self):
        name = self._dataset_name
        if not name:
            return
        node = self.plot_node.currentData() or "raw"
        channel = self.bg_channel.currentData() or 0
        try:
            if node == "processed" and self._meta and self._meta.get("paired"):
                tags = self._meta.get("channel_tags") or ["CP", "CM"]
                data0 = self.service.get_plot_spectrum(name, node="processed", channel=0)
                data1 = self.service.get_plot_spectrum(name, node="processed", channel=1)
                self.main_ax.clear()
                self.main_ax.plot(data0["energy"], data0["spectrum"], label=str(tags[0]))
                self.main_ax.plot(data1["energy"], data1["spectrum"], label=str(tags[1]))
                self.main_ax.legend(fontsize=8)
            else:
                data = self.service.get_plot_spectrum(name, node=node, channel=channel)
                self.main_ax.clear()
                self.main_ax.plot(data["energy"], data["spectrum"], color="#60a5fa")
            self.main_ax.set_xlabel("Energy (eV)")
            self.main_ax.set_ylabel("Intensity")
            self.main_ax.grid(True, alpha=0.3)
            self.main_canvas.draw()
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _bg_kwargs(self) -> dict:
        kwargs = {
            "node": self.bg_node.currentData(),
            "channel": self.bg_channel.currentData() or 0,
            "method": self.bg_method.currentData(),
            "e0": self.bg_e0.value(),
            "e1": self.bg_e1.value(),
        }
        if kwargs["method"] == "two_step":
            kwargs["post_e0"] = self.bg_post_e0.value()
            kwargs["post_e1"] = self.bg_post_e1.value()
        return kwargs

    def _bg_preview(self):
        name = self._require_dataset()
        if not name:
            return
        try:
            result = self.service.bg_preview(name, **self._bg_kwargs())
            self._plot_bg(result)
        except Exception as exc:
            QMessageBox.critical(self, "BG preview error", str(exc))

    def _bg_apply(self):
        name = self._require_dataset()
        if not name:
            return
        try:
            self.service.bg_apply(name, **self._bg_kwargs())
            self._meta = self.service.get_meta(name)
            self._refresh_node_lists()
            self._refresh_main_plot()
            self.status_label.setText("Background subtracted and stored.")
        except Exception as exc:
            QMessageBox.critical(self, "BG apply error", str(exc))

    def _plot_bg(self, result: dict):
        self.bg_ax.clear()
        e = result["energy"]
        self.bg_ax.plot(e, result["spectrum"], color="#ddd", label="raw")
        self.bg_ax.plot(e, result["bg"], color="#ff8c00", label="bg")
        self.bg_ax.plot(e, result["subtracted"], color="#4dd0e1", label="sub")
        self.bg_ax.legend(fontsize=7)
        self.bg_ax.set_xlabel("Energy (eV)")
        self.bg_ax.grid(True, alpha=0.3)
        self.bg_canvas.draw()

    def _sumrule_kwargs(self) -> dict:
        return {
            "nh": self.sum_nh.value(),
            "l3_lo": self.sum_l3_lo.value(),
            "l3_hi": self.sum_l3_hi.value(),
            "l2_lo": self.sum_l2_lo.value(),
            "l2_hi": self.sum_l2_hi.value(),
            "r_lo": self.sum_r_lo.value(),
            "r_hi": self.sum_r_hi.value(),
        }

    def _sumrule_preview(self):
        name = self._require_dataset()
        if not name:
            return
        try:
            result = self.service.sumrule_preview(name, **self._sumrule_kwargs())
            self._plot_sumrule(result)
            self._show_sumrule_text(result)
        except Exception as exc:
            QMessageBox.critical(self, "Sum rule preview error", str(exc))

    def _sumrule_apply(self):
        name = self._require_dataset()
        if not name:
            return
        try:
            self.service.sumrule_apply(name, **self._sumrule_kwargs())
            self._meta = self.service.get_meta(name)
            stored = self.service.sumrule_preview(name, **self._sumrule_kwargs())
            self._plot_sumrule(stored)
            self._show_sumrule_text(stored)
            self.status_label.setText(
                f"Sum rule stored ({stored['tag_plus']}/{stored['tag_minus']}, "
                f"source={stored['source_kind']})"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Sum rule apply error", str(exc))

    def _plot_sumrule(self, result: dict):
        self.sum_ax.clear()
        e = result["energy"]
        self.sum_ax.plot(e, result["mu_plus"], color="#4dd0e1", label="μ+")
        self.sum_ax.plot(e, result["mu_minus"], color="#ff8c00", label="μ−")
        self.sum_ax.plot(e, result["dichroism"], color="#ddd", label="dichro")
        self.sum_ax.legend(fontsize=7)
        self.sum_ax.set_xlabel("Energy (eV)")
        self.sum_ax.grid(True, alpha=0.3)
        self.sum_canvas.draw()

    def _show_sumrule_text(self, result: dict):
        self.sum_results.setText(
            f"p={result['p']:.4g}±{result['p_std']:.4g}  "
            f"q={result['q']:.4g}±{result['q_std']:.4g}  "
            f"r={result['r']:.4g}±{result['r_std']:.4g}\n"
            f"m_orb={result['m_orb']:.4g}±{result['m_orb_std']:.4g}  "
            f"m_spin+dipole={result['m_spin_plus_dipole']:.4g}±"
            f"{result['m_spin_plus_dipole_std']:.4g}"
        )
