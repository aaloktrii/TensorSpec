"""Sum Rule window, "From PEEM" tab: the corrected CP / CM pairs of a PEEM dataset (pushed from the
PEEM panel) -> spectra of a region -> background -> XMCD sum rule.

The Background and XMCD Sum Rule forms, handlers and plots are the PEEM panel's former Background and Sum Rule sections,
moved here unchanged; they call the same ``PeemService.bg_* / sumrule_*`` with the chosen region
(``use_roi`` / ``roi_dict``). Spectra of the region: ``PeemService.roi_spectra``.
"""
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tensorspec.gui.components.peem_map_common import action_style, bold_titles, tidy_forms
from tensorspec.gui.services.peem_service import PeemService


class PeemSumRulePanel(QWidget):
    def __init__(self, service: PeemService | None = None, parent=None):
        super().__init__(parent)
        self.service = service or PeemService()
        self._dataset_name: str | None = None
        self._roi: dict | None = None
        self._setup_ui()
        tidy_forms(self)
        self.refresh_datasets()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        data_group = QGroupBox("1. PEEM data")
        bold_titles(data_group)
        data_form = QFormLayout(data_group)
        row = QHBoxLayout()
        self.dataset_combo = QComboBox()
        self.dataset_combo.activated.connect(lambda *_: self.select_dataset(self.dataset_combo.currentData()))
        row.addWidget(self.dataset_combo, 1)
        b = QPushButton("Refresh")
        b.clicked.connect(self.refresh_datasets)
        row.addWidget(b)
        data_form.addRow("Dataset:", row)
        self.status_label = QLabel("Corrected CP / CM pairs from the PEEM panel (→ Sum Rule).")
        self.status_label.setWordWrap(True)
        data_form.addRow(self.status_label)
        control_layout.addWidget(data_group)

        roi_group = QGroupBox("2. Region")
        bold_titles(roi_group)
        roi_form = QFormLayout(roi_group)
        self.roi_mode = QComboBox()
        self.roi_mode.addItem("Whole image", "all")
        self.roi_mode.addItem("Rectangle", "rect")
        self.roi_mode.currentIndexChanged.connect(self._roi_changed)
        roi_form.addRow("Spectra from:", self.roi_mode)
        self.roi_edit = QLineEdit()
        self.roi_edit.setPlaceholderText("x0, x1, y0, y1 (px), or drag on the Image tab")
        self.roi_edit.returnPressed.connect(self._roi_typed)
        roi_form.addRow("Rectangle:", self.roi_edit)
        b = QPushButton("Show spectra")
        b.clicked.connect(self._show_spectra)
        roi_form.addRow(b)
        control_layout.addWidget(roi_group)

        # --- sections moved unchanged from the PEEM panel (4. Background, 5. XMCD Sum Rule) ---
        bg_group = QGroupBox("3. Background")
        bold_titles(bg_group)
        bg_form = QFormLayout(bg_group)
        self.bg_node = QComboBox()
        self.bg_node.addItem("Raw", "raw")
        self.bg_node.addItem("Processed", "processed")
        bg_form.addRow("Source node:", self.bg_node)
        self.bg_method = QComboBox()
        self.bg_method.addItem("Linear pre-edge", "linear")
        self.bg_method.addItem("Two-step pre/post", "two_step")
        self.bg_method.currentIndexChanged.connect(self._sync_bg_method_ui)
        bg_form.addRow("Method:", self.bg_method)
        self.bg_e0 = QDoubleSpinBox()
        self.bg_e1 = QDoubleSpinBox()
        self.bg_post_e0 = QDoubleSpinBox()
        self.bg_post_e1 = QDoubleSpinBox()
        for spin in (self.bg_e0, self.bg_e1, self.bg_post_e0, self.bg_post_e1):
            spin.setRange(-1e4, 1e4)
            spin.setDecimals(3)
        self.bg_e0.setValue(0.0)
        self.bg_e1.setValue(2.0)
        self.bg_post_e0.setValue(3.0)
        self.bg_post_e1.setValue(4.0)
        bg_form.addRow("Pre e0 / e1:", self._pair_row(self.bg_e0, self.bg_e1))
        self.bg_post_row_label = QLabel("Post e0 / e1:")
        self.bg_post_row = self._pair_row(self.bg_post_e0, self.bg_post_e1)
        bg_form.addRow(self.bg_post_row_label, self.bg_post_row)
        bg_btns = QHBoxLayout()
        self.btn_bg_preview = QPushButton("Preview BG")
        self.btn_bg_preview.clicked.connect(self._bg_preview)
        self.btn_bg_apply = QPushButton("Apply BG")
        self.btn_bg_apply.clicked.connect(self._bg_apply)
        bg_btns.addWidget(self.btn_bg_preview)
        bg_btns.addWidget(self.btn_bg_apply)
        bg_form.addRow(bg_btns)
        control_layout.addWidget(bg_group)
        self._sync_bg_method_ui()

        sum_group = QGroupBox("4. XMCD Sum Rule")
        bold_titles(sum_group)
        sum_form = QFormLayout(sum_group)
        self.sum_nh = QDoubleSpinBox()
        self.sum_nh.setRange(0.0, 20.0)
        self.sum_nh.setValue(1.0)
        self.sum_nh.setDecimals(2)
        sum_form.addRow("n_h:", self.sum_nh)
        self.sum_l3_lo = QDoubleSpinBox()
        self.sum_l3_hi = QDoubleSpinBox()
        self.sum_l2_lo = QDoubleSpinBox()
        self.sum_l2_hi = QDoubleSpinBox()
        self.sum_r_lo = QDoubleSpinBox()
        self.sum_r_hi = QDoubleSpinBox()
        for spin in (self.sum_l3_lo, self.sum_l3_hi, self.sum_l2_lo, self.sum_l2_hi, self.sum_r_lo, self.sum_r_hi):
            spin.setRange(-1e4, 1e4)
            spin.setDecimals(3)
        self.sum_l3_lo.setValue(0.0)
        self.sum_l3_hi.setValue(1.0)
        self.sum_l2_lo.setValue(2.0)
        self.sum_l2_hi.setValue(3.0)
        self.sum_r_lo.setValue(0.0)
        self.sum_r_hi.setValue(4.0)
        sum_form.addRow("L3 lo/hi:", self._pair_row(self.sum_l3_lo, self.sum_l3_hi))
        sum_form.addRow("L2 lo/hi:", self._pair_row(self.sum_l2_lo, self.sum_l2_hi))
        sum_form.addRow("R lo/hi:", self._pair_row(self.sum_r_lo, self.sum_r_hi))
        sum_btns = QHBoxLayout()
        self.btn_sum_preview = QPushButton("Preview Sum Rule")
        self.btn_sum_preview.clicked.connect(self._sumrule_preview)
        self.btn_sum_apply = action_style(QPushButton("Apply Sum Rule"), "map")
        self.btn_sum_apply.clicked.connect(self._sumrule_apply)
        sum_btns.addWidget(self.btn_sum_preview)
        sum_btns.addWidget(self.btn_sum_apply)
        sum_form.addRow(sum_btns)
        self.sum_results = QLabel("(no sum-rule result yet)")
        self.sum_results.setWordWrap(True)
        sum_form.addRow(self.sum_results)
        control_layout.addWidget(sum_group)

        control_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(control_panel)
        scroll.setMinimumWidth(360)
        splitter.addWidget(scroll)

        self.view_tabs = QTabWidget()
        self.view_tabs.setDocumentMode(True)
        self.img_figure, self.img_canvas, self.img_ax = self._figure_tab("Image")
        self.spec_figure, self.spec_canvas, self.spec_ax = self._figure_tab("Spectra")
        self.bg_figure, self.bg_canvas, self.bg_ax = self._figure_tab("BG")
        self.sum_figure, self.sum_canvas, self.sum_ax = self._figure_tab("Sum Rule")
        self._selector = RectangleSelector(self.img_ax, self._on_drag, useblit=False, interactive=False,
                                           button=[1], minspanx=3, minspany=3, spancoords="data")
        splitter.addWidget(self.view_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 720])

    def _figure_tab(self, label: str):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)
        fig = Figure(figsize=(6, 4), dpi=100, layout="tight")
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        lay.addWidget(canvas, stretch=1)
        self.view_tabs.addTab(page, label)
        return fig, canvas, ax

    @staticmethod
    def _pair_row(a, b):
        row = QHBoxLayout()
        row.addWidget(a)
        row.addWidget(b)
        wrap = QWidget()
        wrap.setLayout(row)
        return wrap

    def _sync_bg_method_ui(self):
        two_step = self.bg_method.currentData() == "two_step"
        self.bg_post_row_label.setVisible(two_step)
        self.bg_post_row.setVisible(two_step)

    def _require_dataset(self) -> str | None:
        if not self._dataset_name:
            QMessageBox.warning(self, "No data", "Push corrected pairs (CP / CM or LH / LV) from the PEEM panel first (→ Sum Rule).")
            return None
        return self._dataset_name

    # ------------------------------------------------------------------ dataset + region
    def refresh_datasets(self):
        names = self.service.sum_rule_datasets()
        self.dataset_combo.clear()
        for n in names:
            self.dataset_combo.addItem(n, n)
        if not names:
            self.dataset_combo.addItem("(none: push from the PEEM panel)", None)
        if self._dataset_name in names:
            self.dataset_combo.setCurrentIndex(names.index(self._dataset_name))

    def select_dataset(self, name: str | None):
        """Show ``name`` (called by the PEEM panel's '→ Sum Rule')."""
        if not name:
            return
        self._dataset_name = name
        self.refresh_datasets()
        meta = self.service.get_meta(name)
        self.bg_node.clear()
        self.bg_node.addItem("Raw", "raw")
        if meta.get("has_processed"):
            self.bg_node.addItem("Processed", "processed")
        for child in meta.get("processed_children", []):
            if not str(child).endswith("_bg"):
                self.bg_node.addItem(f"Processed / {child}", f"processed/{child}")
        n = int(meta.get("n_pairs") or meta.get("n_frames") or 0)
        self.status_label.setText(f"'{name}': {n} {'energy' if n == 1 else 'energies'} (one {' / '.join(meta.get('channel_tags') or ['?'])} "
                                  f"image each)" + ("; a single photon energy: no spectrum to fit (see Spectra)" if n == 1 else ""))
        self._draw_image()
        self._show_spectra()

    def _roi_changed(self, *_):
        self._roi = None if self.roi_mode.currentData() == "all" else self._roi
        self._draw_image()

    def _roi_typed(self):
        try:
            x0, x1, y0, y1 = (int(round(float(v))) for v in self.roi_edit.text().replace(",", " ").split())
        except ValueError:
            QMessageBox.warning(self, "Region", "Type four numbers: x0, x1, y0, y1 (px).")
            return
        self._set_rect(min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

    def _on_drag(self, press, release):
        if press.xdata is None or release.xdata is None:
            return
        x0, x1 = sorted((int(round(press.xdata)), int(round(release.xdata))))
        y0, y1 = sorted((int(round(press.ydata)), int(round(release.ydata))))
        self._set_rect(x0, x1, y0, y1)

    def _set_rect(self, x0, x1, y0, y1):
        self._roi = {"kind": "rect", "x0": x0, "x1": x1, "y0": y0, "y1": y1}
        self.roi_edit.setText(f"{x0}, {x1}, {y0}, {y1}")
        self.roi_mode.blockSignals(True)
        self.roi_mode.setCurrentIndex(self.roi_mode.findData("rect"))
        self.roi_mode.blockSignals(False)
        self._draw_image()
        self._show_spectra()

    def _roi_kwargs(self) -> dict:
        return {"use_roi": True, "roi_dict": self._roi} if self._roi else {}

    def _draw_image(self):
        self.img_ax.clear()
        if self._dataset_name:
            try:
                img = self.service.channel_image(self._dataset_name)
                self.img_ax.imshow(img, cmap="gray", origin="upper")
            except ValueError as exc:
                self.img_ax.set_title(str(exc), fontsize=8)
        if self._roi:
            r = self._roi
            self.img_ax.add_patch(Rectangle((r["x0"], r["y0"]), r["x1"] - r["x0"], r["y1"] - r["y0"],
                                            fill=False, ec="yellow", lw=1.5))
        self.img_ax.set_title("drag a rectangle to choose the region", fontsize=9)
        self.img_canvas.draw_idle()

    def _show_spectra(self):
        name = self._require_dataset()
        if not name:
            return
        try:
            sp = self.service.roi_spectra(name, self._roi)
        except ValueError as exc:
            QMessageBox.critical(self, "Spectra error", str(exc))
            return
        ax = self.spec_ax
        ax.clear()
        e = sp["energy"]
        ta, tb = sp["channels"]
        mk = "o" if len(e) < 30 else None  # few energies (or one): show the points
        ax.plot(e, sp[ta], label=ta, color="#60a5fa", marker=mk)
        ax.plot(e, sp[tb], label=tb, color="#f87171", marker=mk)
        ax2 = ax.twinx() if not hasattr(self, "_spec_ax2") else self._spec_ax2
        self._spec_ax2 = ax2
        ax2.clear()
        ax2.plot(e, sp["dichroism"], label=f"{tb} − {ta}", color="#a78bfa", marker=mk)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel(f"{tb} − {ta}")
        ax.set_xlabel("pair index" if sp["energy_source"] == "index" else "Energy")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=7, loc="upper left")
        ax2.legend(fontsize=7, loc="upper right")
        ax.set_title(f"{'region' if self._roi else 'whole image'}: {sp['n_pixels']} px")
        if sp.get("note"):
            self.status_label.setText(sp["note"])
        if len(e) == 1:  # one photon energy: values, not a spectrum
            ax.set_title(f"one photon energy ({float(e[0]):.2f} eV): {ta} {float(sp[ta][0]):.4g}, {tb} {float(sp[tb][0]):.4g}, "
                         f"{tb} − {ta} {float(sp['dichroism'][0]):+.4g}", fontsize=9)
            self.status_label.setText("This acquisition is at a single photon energy: spectra, background and the sum "
                                      "rule need an energy scan across the edge (L3 and L2).")
        self.spec_canvas.draw_idle()
        self.view_tabs.setCurrentIndex(1)

    # ------------------------------------------------------------------ moved from the PEEM panel
    def _bg_kwargs(self) -> dict:
        method = self.bg_method.currentData()
        kwargs = {
            "node": self.bg_node.currentData(),
            "method": method,
            "e0": self.bg_e0.value(),
            "e1": self.bg_e1.value(),
        }
        if method == "two_step":
            kwargs["post_e0"] = self.bg_post_e0.value()
            kwargs["post_e1"] = self.bg_post_e1.value()
        return {**kwargs, **self._roi_kwargs()}

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
            self.select_dataset(name)
            self.status_label.setText("Background subtracted and stored.")
        except Exception as exc:
            QMessageBox.critical(self, "BG apply error", str(exc))

    def _sumrule_kwargs(self) -> dict:
        return {
            "nh": self.sum_nh.value(),
            "l3_lo": self.sum_l3_lo.value(),
            "l3_hi": self.sum_l3_hi.value(),
            "l2_lo": self.sum_l2_lo.value(),
            "l2_hi": self.sum_l2_hi.value(),
            "r_lo": self.sum_r_lo.value(),
            "r_hi": self.sum_r_hi.value(),
            **self._roi_kwargs(),
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
            result = self.service.sumrule_apply(name, **self._sumrule_kwargs())
            stored = self.service.sumrule_preview(name, **self._sumrule_kwargs())
            self._plot_sumrule(stored)
            self._show_sumrule_text(stored)
            self.status_label.setText(
                f"Sum rule stored ({result['tag_plus']}/{result['tag_minus']}, "
                f"source={result['source_kind']})"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Sum rule apply error", str(exc))

    def _show_sumrule_text(self, result: dict):
        self.sum_results.setText(
            f"p={result['p']:.4g}±{result['p_std']:.4g}  "
            f"q={result['q']:.4g}±{result['q_std']:.4g}  "
            f"r={result['r']:.4g}±{result['r_std']:.4g}\n"
            f"m_orb={result['m_orb']:.4g}±{result['m_orb_std']:.4g}  "
            f"m_spin+dipole={result['m_spin_plus_dipole']:.4g}±"
            f"{result['m_spin_plus_dipole_std']:.4g}"
        )

    def _plot_bg(self, result: dict):
        self.bg_ax.clear()
        energy = result["energy"]
        self.bg_ax.plot(energy, result["spectrum"], label="spectrum", color="#60a5fa")
        self.bg_ax.plot(energy, result["bg"], label="bg", color="#f97316")
        self.bg_ax.plot(energy, result["subtracted"], label="subtracted", color="#34d399")
        self.bg_ax.set_xlabel("Energy")
        self.bg_ax.set_ylabel("Intensity")
        self.bg_ax.legend(fontsize=7)
        self.bg_ax.set_title(f"BG preview ({result['method']})")
        self.bg_canvas.draw()
        self.view_tabs.setCurrentIndex(2)

    def _plot_sumrule(self, result: dict):
        self.sum_ax.clear()
        energy = result["energy"]
        self.sum_ax.plot(energy, result["mu_plus"], label=result["tag_plus"], color="#60a5fa")
        self.sum_ax.plot(energy, result["mu_minus"], label=result["tag_minus"], color="#f87171")
        self.sum_ax.plot(energy, result["dichroism"], label="dichroism", color="#a78bfa")
        self.sum_ax.set_xlabel("Energy")
        self.sum_ax.set_ylabel("μ")
        self.sum_ax.legend(fontsize=7)
        self.sum_ax.set_title("XMCD sum rule")
        self.sum_canvas.draw()
        self.view_tabs.setCurrentIndex(3)
