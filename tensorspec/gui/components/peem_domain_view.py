"""Analysis tab, "Domains": spatial domain analysis of the maps pushed from the Map view.

Widgets only; every step is a ``PeemService`` call (peem_steps/domain_analysis.py, math in
core/peem/domain_analysis.py, drawing in plotting/peem_plots.py). Left: the pushed maps, the pipeline
parameters (the SDA toolkit's defaults), Run and export. Right: one step of the selected map with axes
in um, and a table of the numbers of every analysed map.
"""
from __future__ import annotations

import os

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tensorspec.core.peem import domain_analysis as da
from tensorspec.gui.components.peem_map_common import _dspin, action_style, bold_titles
from tensorspec.plotting import peem_plots


def _combo(items: list[tuple[str, object]], current=None) -> QComboBox:
    c = QComboBox()
    for label, data in items:
        c.addItem(label, data)
    if current is not None:
        c.setCurrentIndex(max(0, c.findData(current)))
    return c


class PeemDomainView(QWidget):
    def __init__(self, service, parent=None):
        super().__init__(parent)
        self.service = service
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        root.addWidget(split)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._build_controls())
        scroll.setMinimumWidth(390)
        split.addWidget(scroll)
        split.addWidget(self._build_results())
        split.setStretchFactor(1, 1)
        split.setSizes([400, 700])

    # ------------------------------------------------------------------ building
    def _build_controls(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("Pushed maps")
        bold_titles(g)
        v = QVBoxLayout(g)
        self.items = QListWidget()
        self.items.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.items.setToolTip("Maps pushed from the Map view (6. Push to analysis); select one or more")
        self.items.setMaximumHeight(130)
        self.items.itemSelectionChanged.connect(self._show)
        v.addWidget(self.items)
        row = QHBoxLayout()
        b = QPushButton("Remove selected")
        b.clicked.connect(self._remove)
        row.addWidget(b)
        b = QPushButton("↻")
        b.setFixedWidth(30)
        b.setToolTip("Update the list")
        b.clicked.connect(lambda: self.refresh())
        row.addWidget(b)
        v.addLayout(row)
        lay.addWidget(g)

        g = QGroupBox("Steps")
        bold_titles(g)
        f = QFormLayout(g)
        d = da.DEFAULTS
        self.denoise = _combo([("TV-Chambolle", "tv"), ("Median", "median"), ("None", "none")], d["denoise"])
        self.tv_weight = _dspin(0.0, 2.0, d["tv_weight"], 0.01, 2)
        row = QHBoxLayout()
        row.addWidget(self.denoise, 1)
        row.addWidget(QLabel("w"))
        row.addWidget(self.tv_weight)
        f.addRow("Denoise:", row)
        self.flake = _combo([("Pushed ROI", "pushed"), ("Automatic", "auto"), ("Whole map", "all")], d["flake"])
        f.addRow("Flake:", self.flake)
        self.background = _combo([("Gaussian high-pass", "gaussian"), ("Background region", "region"),
                                  ("None (asymmetry 0)", "none")], d["background"])
        self.background.setToolTip("Gaussian: subtract a large-scale blur (σ must exceed the domains).\n"
                                   "Background region: its mean is the magnetic zero (push with the region ticked).\n"
                                   "None: the map's own zero.")
        self.sigma = _dspin(1.0, 1000.0, d["flatten_sigma"], 5.0, 0)
        row = QHBoxLayout()
        row.addWidget(self.background, 1)
        row.addWidget(QLabel("σ"))
        row.addWidget(self.sigma)
        f.addRow("Background:", row)
        self.fft = QCheckBox()
        self.fft_cut = QSpinBox()
        self.fft_cut.setRange(1, 2000)
        self.fft_cut.setValue(d["fft_cutoff"])
        self.fft_mode = _combo([("radial", "radial"), ("along x", "x")], d["fft_mode"])
        row = QHBoxLayout()
        row.addWidget(self.fft)
        row.addWidget(QLabel("cutoff"))
        row.addWidget(self.fft_cut)
        row.addWidget(self.fft_mode)
        f.addRow("FFT filter:", row)
        self.m0 = _dspin(0.0, 1.0, d["m_threshold"], 0.01, 3)
        f.addRow("Three states m₀:", self.m0)
        self.min_px = QSpinBox()
        self.min_px.setRange(0, 100000)
        self.min_px.setValue(d["min_px"])
        f.addRow("Min domain (px):", self.min_px)
        self.conn = _combo([("4 (edges)", 1), ("8 (+ corners)", 2)], d["connectivity"])
        f.addRow("Connectivity:", self.conn)
        row = QHBoxLayout()
        b = action_style(QPushButton("Run selected"), "domains")
        b.clicked.connect(lambda: self._run(all_items=False))
        row.addWidget(b)
        b = QPushButton("Run all")
        b.clicked.connect(lambda: self._run(all_items=True))
        row.addWidget(b)
        f.addRow(row)
        lay.addWidget(g)

        g = QGroupBox("Show and export")
        bold_titles(g)
        f = QFormLayout(g)
        self.step = _combo([("input map", "input")] + [(da.STEP_LABELS[s], s) for s in da.STEPS], "three_states")
        self.step.currentIndexChanged.connect(self._show)
        f.addRow("Show:", self.step)
        self.sizes = QCheckBox("Domain-size distribution")
        self.sizes.toggled.connect(self._show)
        f.addRow(self.sizes)
        row = QHBoxLayout()
        for label, fn in (("PDF…", self._export_pdf), ("Table CSV…", self._export_table), ("Sizes CSV…", self._export_sizes)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        f.addRow("Export:", row)
        lay.addWidget(g)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        lay.addStretch(1)
        return w

    def _build_results(self) -> QWidget:
        split = QSplitter(Qt.Vertical)
        split.setChildrenCollapsible(False)
        top = QWidget()
        v = QVBoxLayout(top)
        v.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(6, 4.5), layout="constrained")
        self.canvas = FigureCanvas(self.fig)
        v.addWidget(self.canvas, 1)
        self.caption = QLabel("")
        self.caption.setWordWrap(True)
        v.addWidget(self.caption)
        split.addWidget(top)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        split.addWidget(self.table)
        split.setSizes([520, 200])
        return split

    # ------------------------------------------------------------------ helpers
    def _err(self, msg: str):
        QMessageBox.warning(self, "Domain analysis", msg)

    def params(self) -> dict:
        return {"denoise": self.denoise.currentData(), "tv_weight": self.tv_weight.value(),
                "flake": self.flake.currentData(), "background": self.background.currentData(),
                "flatten_sigma": self.sigma.value(), "fft_filter": self.fft.isChecked(),
                "fft_cutoff": self.fft_cut.value(), "fft_mode": self.fft_mode.currentData(),
                "m_threshold": self.m0.value(), "min_px": self.min_px.value(),
                "connectivity": self.conn.currentData()}

    def _selected(self) -> list[tuple[str, str]]:
        return [tuple(i.data(Qt.UserRole)) for i in self.items.selectedItems()]

    def refresh(self, select: tuple[str, str] | None = None):
        """Reload the pushed maps and the table; select (name, label) if given."""
        keep = [select] if select else self._selected()
        self.items.blockSignals(True)
        self.items.clear()
        for it in self.service.domain_inputs():
            item = QListWidgetItem(it["title"] + ("" if it["analysed"] else "  · not run"))
            item.setData(Qt.UserRole, [it["name"], it["label"]])
            self.items.addItem(item)
            if (it["name"], it["label"]) in keep:
                item.setSelected(True)
        if not self.items.selectedItems() and self.items.count():
            self.items.item(self.items.count() - 1).setSelected(True)
        self.items.blockSignals(False)
        self._fill_table()
        self._show()

    def _fill_table(self):
        rows = self.service.domain_table()
        self.table.clear()
        self.table.setRowCount(len(rows))
        cols = list(rows[0]) if rows else []
        cols = [c for c in cols if c != "label"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        for r, row in enumerate(rows):
            for c, k in enumerate(cols):
                v = row[k]
                text = f"{v:.3g}" if isinstance(v, float) else str(v)
                self.table.setItem(r, c, QTableWidgetItem(text))

    # ------------------------------------------------------------------ actions
    def _run(self, all_items: bool):
        items = [(it["name"], it["label"]) for it in self.service.domain_inputs()] if all_items else self._selected()
        if not items:
            self._err("Push a map first (Map view, 6. Push to analysis).")
            return
        from PySide6.QtGui import QGuiApplication

        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        done, errors = 0, []
        try:
            for name, label in items:
                try:
                    self.service.run_domain_analysis(name, label, **self.params())
                    done += 1
                except (ValueError, KeyError) as exc:
                    errors.append(f"{name} · {label}: {exc}")
        finally:
            QGuiApplication.restoreOverrideCursor()
        if errors:
            self._err("\n".join(errors))
        self.status.setText(f"Analysed {done} map(s).")
        self.refresh(select=items[-1] if not all_items else None)

    def _remove(self):
        for name, label in self._selected():
            self.service.remove_domain_input(name, label)
        self.refresh()

    def _show(self):
        sel = self._selected()
        self.fig.clear()
        self.caption.setText("")
        if not sel:
            self.canvas.draw_idle()
            return
        name, label = sel[0]
        step = self.step.currentData()
        result = self.service.domain_result(name, label)
        try:
            img = self.service.domain_layer(name, label, step)
        except ValueError:
            img, step = self.service.domain_layer(name, label, "input"), "input"
        um = result.get("um_per_px") if result else None
        um = um if um and um == um else None
        sizes = self.sizes.isChecked() and bool(result)
        ax = self.fig.add_subplot(1, 2 if sizes else 1, 1)
        title = f"{name} · {da.STEP_LABELS.get(step, 'input map')}"
        im = peem_plots.domain_step(ax, img, step, um, title)
        try:  # zoom to the flake (axes keep the field's coordinates)
            peem_plots.zoom_to_mask(ax, self.service.domain_layer(name, label, "flake"), um)
        except ValueError:
            pass
        if im is not None:
            self.fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        if sizes:
            peem_plots.domain_sizes(self.fig.add_subplot(1, 2, 2), self.service.domain_areas(name, label))
        self.caption.setText(peem_plots.domain_numbers_text(result) if result else "Not run yet: Run selected.")
        self.canvas.draw_idle()

    def _export_pdf(self):
        sel = self._selected()
        if not sel:
            return
        name, label = sel[0]
        step = self.step.currentData()
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF", f"{name}_{label}_{step}.pdf", "PDF (*.pdf)")
        if path:
            try:
                self.service.export_domain_figure(path, name, label, step)
                if self.sizes.isChecked() and self.service.domain_result(name, label):
                    peem_plots.save_domain_sizes_pdf(os.path.splitext(path)[0] + "_sizes.pdf",
                                                     self.service.domain_areas(name, label), name)
                self.status.setText(f"Saved {path}")
            except ValueError as exc:
                self._err(str(exc))

    def _export_table(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export table", "domain_analysis.csv", "CSV (*.csv)")
        if path:
            try:
                self.status.setText(f"Saved {self.service.export_domain_table(path)}")
            except ValueError as exc:
                self._err(str(exc))

    def _export_sizes(self):
        sel = self._selected()
        if not sel:
            return
        name, label = sel[0]
        path, _ = QFileDialog.getSaveFileName(self, "Export domain sizes", f"{name}_{label}_sizes.csv", "CSV (*.csv)")
        if path:
            try:
                self.status.setText(f"Saved {self.service.export_domain_areas(path, name, label)}")
            except ValueError as exc:
                self._err(str(exc))
