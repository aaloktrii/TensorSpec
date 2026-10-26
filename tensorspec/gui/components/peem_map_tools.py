"""PEEM panel add-on: calculate (asymmetry map) and analyse (profiles, domain width, histogram, FFT,
ROI statistics, domain map, export, compare with a reference dataset) on the PEEM tab's own
corrected CP / CM channels.

Front end only; every step is a ``PeemService`` call (contract: gui/services/peem_steps/__init__.py).
The PEEM panel plugs it in with a few lines:

    tools = PeemMapTools(service, get_name=lambda: panel._dataset_name, send_to_sum_rule=...)
    left_layout.addWidget(tools.controls)           # sections 4-7
    view_tabs.addTab(tools.map_view, "Map")         # maps with ROI / profile overlays
    view_tabs.addTab(tools.plot_view, "Analysis")   # profiles, histogram, FFT (PDF export)
"""
from __future__ import annotations

import os
from typing import Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tensorspec.core.data_models import TensorData
from tensorspec.gui.components.data_viewer_panel import DataViewerPanel
from tensorspec.gui.components.peem_map_common import _coords_row, _dspin, _numbers, action_style, bold_titles
from tensorspec.plotting import peem_plots


def _opt_float(text: str) -> float | None:
    text = text.strip()
    return float(text) if text else None


class PeemMapTools:
    """Sections 4-7 of the PEEM panel and their two views (see the module docstring)."""

    def __init__(self, service, get_name: Callable[[], str | None], parent: QWidget | None = None,
                 send_to_sum_rule: Callable[[str], None] | None = None,
                 on_map_made: Callable[[], None] | None = None):
        self.service, self._get_name, self._parent = service, get_name, parent
        self._send_to_sum_rule = send_to_sum_rule
        self._on_map_made = on_map_made  # e.g. the panel refreshes its Frame "Show" list
        self._pair_tags: list[str] | None = None
        self._line: tuple | None = None
        self._loaded_name: str | None = None
        self._shown_name: str | None = None
        self._build_controls()
        self._build_map_view()
        self._build_plot_view()

    # ------------------------------------------------------------------ building
    def _build_controls(self):
        self.controls = QWidget()
        lay = QVBoxLayout(self.controls)
        lay.setContentsMargins(0, 0, 0, 0)

        asym = QGroupBox("4. Dichroism map")
        bold_titles(asym)
        af = QFormLayout(asym)
        self.map_register = QComboBox()
        self.map_register.addItem("Register polarizations (recommended)", True)
        self.map_register.addItem("No extra registration", False)
        af.addRow("Align:", self.map_register)
        row = QHBoxLayout()
        self.e_lo, self.e_hi = QLineEdit(), QLineEdit()
        for w, ph in ((self.e_lo, "from (all)"), (self.e_hi, "to (all)")):
            w.setPlaceholderText(ph)
            row.addWidget(w)
        af.addRow("Energy window (eV):", row)
        b = action_style(QPushButton("Make dichroism map"), "map")
        b.setToolTip("XMCD (CM−CP)/(CM+CP) for CP / CM, XLD (LV−LH)/(LV+LH) for LH / LV.\n"
                     "After 2. Stack Pairs (+ 3. Drift): the corrected pairs, optionally only the frames in the\n"
                     "energy window. Without Stack Pairs: all frames, each aligned by Register polarizations.")
        b.clicked.connect(self._asymmetry)
        af.addRow(b)
        self.btn_asymmetry = b
        self.show_combo = QComboBox()
        self.show_combo.activated.connect(lambda *_: self._select(*self.show_combo.currentData().split("|", 1)))
        af.addRow("Show:", self.show_combo)
        lay.addWidget(asym)

        an = QGroupBox("5. Image analysis")
        bold_titles(an)
        nf = QFormLayout(an)
        row = QHBoxLayout()
        b = QPushButton("Draw line")
        b.clicked.connect(lambda: self._start_tool("line", self._on_line, "orange"))
        row.addWidget(b)
        r2, self.line_edit = _coords_row("x0, y0, x1, y1 (px)", self._set_line)
        row.addLayout(r2, 1)
        nf.addRow("Line:", row)
        self.line_hw = QSpinBox()
        self.line_hw.setRange(0, 50)
        nf.addRow("Half width (px):", self.line_hw)
        row = QHBoxLayout()
        for label, fn in (("Add profile", self._add_profile), ("Clear profiles", self._clear_profiles)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        nf.addRow(row)
        self.prof_list = QListWidget()
        self.prof_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.prof_list.setMaximumHeight(110)
        self.prof_list.setToolTip("Tick = shown in the plot and on the map; select and Remove to delete")
        self.prof_list.itemChanged.connect(self._profile_toggled)
        nf.addRow("Profiles:", self.prof_list)
        row = QHBoxLayout()
        for label, fn in (("Remove selected", self._remove_profiles), ("Show all", lambda: self._show_all_profiles(True)),
                          ("Hide all", lambda: self._show_all_profiles(False))):
            b = QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        nf.addRow(row)
        self.prof_dock = QCheckBox("Profiles beside the map")
        self.prof_dock.setChecked(True)
        self.prof_dock.setToolTip("Each line's profile under the map (mostly horizontal line, on the map's x axis)\n"
                                  "or beside it (mostly vertical line, on the map's y axis); follows zoom")
        self.prof_dock.toggled.connect(lambda *_: self._dock_profiles())
        nf.addRow(self.prof_dock)
        self.prof_offset = _dspin(0, 1e6, 0.0, 0.01, 4)
        self.prof_offset.valueChanged.connect(lambda *_: self._draw_profiles())
        nf.addRow("Offset between profiles:", self.prof_offset)
        row = QHBoxLayout()
        for label, fn, tip in (("Histogram", self._histogram, "of the shown map, inside the ROI if set"),
                               ("FFT", self._fft, "2-D FFT and dominant spatial period"),
                               ("Domain width", self._domain_width, "area / domain-wall length, inside the ROI if set")):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        nf.addRow(row)
        lay.addWidget(an)

        rd = QGroupBox("6. ROIs and domains")
        bold_titles(rd)
        rf = QFormLayout(rd)
        row = QHBoxLayout()
        for label, kind in (("Draw rectangle", "rect"), ("Draw polygon", "polygon")):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, k=kind: self._start_tool(k, self._on_roi, "yellow"))
            row.addWidget(b)
        b = QPushButton("Clear")
        b.clicked.connect(lambda: self._set_roi(None))
        row.addWidget(b)
        rf.addRow("ROI:", row)
        r2, self.roi_edit = _coords_row("x0, x1, y0, y1 (px)", self._set_roi_typed)
        rf.addRow("Type ROI:", r2)
        self.thr = _dspin(0, 1, 0.15, 0.01, 3)
        rf.addRow("Domain threshold:", self.thr)
        row = QHBoxLayout()
        for label, fn in (("ROI statistics", self._roi_stats), ("Domain map", self._domain_map)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        rf.addRow(row)
        row = QHBoxLayout()
        b = QPushButton("Export shown map…")
        b.setToolTip("Values as a 32-bit TIFF and a colour PNG next to it (colours as shown)")
        b.clicked.connect(self._export)
        row.addWidget(b)
        b = QPushButton("More export options…")
        b.clicked.connect(self._export_more)
        row.addWidget(b)
        rf.addRow(row)
        lay.addWidget(rd)

        cg = QGroupBox("7. Compare with a reference")
        bold_titles(cg)
        cf = QFormLayout(cg)
        row = QHBoxLayout()
        self.cmp_ref = QComboBox()
        self.cmp_ref.setToolTip("Another loaded dataset with an asymmetry map, e.g. the same area above T_c\n"
                                "(XMLD = XLD(T) − XLD(T > T_c)), another azimuth or another field.")
        row.addWidget(self.cmp_ref, 1)
        b = QPushButton("↻")
        b.setToolTip("Update the list of datasets")
        b.setFixedWidth(30)
        b.clicked.connect(self._fill_compare_refs)
        row.addWidget(b)
        cf.addRow("Reference:", row)
        self.cmp_search = QComboBox()
        for label, key in (("Auto (search if the match is poor)", "auto"), ("Same view (shift + fine rotation)", "none"),
                           ("Full search (re-imaged field / FOV)", "full")):
            self.cmp_search.addItem(label, key)
        cf.addRow("Alignment:", self.cmp_search)
        self.cmp_rot = _dspin(-360, 360, 0.0, 1.0, 1)
        self.cmp_rot.setToolTip("Expected rotation of the reference field (deg), e.g. a different azimuth")
        cf.addRow("Expected rotation (deg):", self.cmp_rot)
        self.cmp_smooth = _dspin(0, 20, 0.0, 0.5, 1)
        self.cmp_smooth.setToolTip("Gaussian smoothing of a noisy reference (px)")
        cf.addRow("Smooth reference (px):", self.cmp_smooth)
        row = QHBoxLayout()
        b = QPushButton("Register + subtract")
        b.setToolTip("Reference map registered onto this one on the intensity images (shift, rotation, scale),\n"
                     "then this map − reference → Map view 'compare'.")
        b.clicked.connect(self._compare)
        row.addWidget(b)
        b = QPushButton("Figure (3 maps)…")
        b.setToolTip("Map | registered reference | difference on one grey scale, vector PDF")
        b.clicked.connect(self._compare_figure)
        row.addWidget(b)
        cf.addRow(row)
        lay.addWidget(cg)

        if self._send_to_sum_rule is not None:
            b = action_style(QPushButton("→ Sum Rule (corrected pairs)"), "handover")
            b.setToolTip("Open the Sum Rule window with this dataset: ROI spectra, background, sum rule\n"
                         "(the XMCD sum rule needs CP / CM; LH / LV give spectra and background)")
            b.clicked.connect(self._send)
            lay.addWidget(b)
            self.btn_sum_rule = b
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

    def _build_map_view(self):
        self.map_view = QWidget()
        v = QVBoxLayout(self.map_view)
        self.viewer = DataViewerPanel(compact=True)
        v.addWidget(self.viewer, 1)
        rng = QHBoxLayout()
        rng.addWidget(QLabel("Colour range:"))
        self.range_min, self.range_max = QDoubleSpinBox(), QDoubleSpinBox()
        for sp in (self.range_min, self.range_max):
            sp.setRange(-1e12, 1e12)
            sp.setDecimals(4)
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(self._apply_range)
        rng.addWidget(self.range_min, 1)
        rng.addWidget(QLabel("to"))
        rng.addWidget(self.range_max, 1)
        b = QPushButton("Auto")
        b.clicked.connect(self._auto_range)
        rng.addWidget(b)
        v.addLayout(rng)

    def _build_plot_view(self):
        self.plot_view = QWidget()
        v = QVBoxLayout(self.plot_view)
        h = QHBoxLayout()
        self.plot_title = QLabel("")
        h.addWidget(self.plot_title, 1)
        b = QPushButton("Export PDF…")
        b.setToolTip("Vector PDF with editable text (Illustrator); profiles also as a .csv next to it")
        b.clicked.connect(self._export_plot_pdf)
        h.addWidget(b)
        v.addLayout(h)
        self.pfig = Figure(figsize=(6, 4), layout="tight")
        self.pcanvas = FigureCanvas(self.pfig)
        v.addWidget(self.pcanvas, 1)
        self._plot_kind = None

    # ------------------------------------------------------------------ helpers
    def _name(self) -> str | None:
        name = self._get_name()
        if not name:
            self._err("Load PEEM data first.")
        return name

    def _err(self, msg: str):
        QMessageBox.warning(self._parent, "PEEM", msg)

    def _call(self, fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except (ValueError, KeyError) as exc:
            self._err(str(exc))
            return None

    def _shown(self) -> tuple[str, str]:
        view = self.viewer.primary_view()
        layer = view.combo_layer.currentText() if view is not None else "Intensity"
        if layer in ("", "Intensity"):
            return "stack", "data"
        return self.service.parse_layer(layer)

    def _roi(self) -> dict | None:
        name = self._get_name()
        return self.service.get_rois(name)["roi"] if name else None

    # ------------------------------------------------------------------ map view
    def set_pair(self, tags: list[str] | None):
        """Name the buttons after the dataset's pair (CP / CM or LH / LV, from Stack Pairs)."""
        self._pair_tags = list(tags[:2]) if tags and len(tags) >= 2 else None
        pair = " / ".join(tags[:2]) if tags and len(tags) >= 2 else "pairs"
        kind = {"CP / CM": "XMCD", "LH / LV": "XLD"}.get(pair, "dichroism")
        self.btn_asymmetry.setText(f"Make {kind} map ({pair})" if pair != "pairs" else "Make dichroism map")
        if hasattr(self, "btn_sum_rule"):
            self.btn_sum_rule.setText(f"→ Sum Rule (corrected {pair})")

    def refresh(self, select: tuple[str, str] | None = None):
        """Show the dataset's maps (after the asymmetry step); keeps the chosen layer."""
        name = self._get_name()
        prods = self.service.list_products(name) if name else []
        base = "stack" if "stack" in prods else ("nc" if "nc" in prods else None)  # nc: maps loaded from a .nc file
        if not name or base is None:
            self.clear()
            return
        if name != self._loaded_name or self.viewer.tensor_data is None:
            img = self.service.get_map(name, base, "data")
            ny, nx = img.shape
            td = TensorData(value=np.asarray(img, np.float32)[None], axes=[np.zeros(1), np.arange(ny, dtype=float),
                            np.arange(nx, dtype=float)], labels=["frame", "y", "x"], units=["", "px", "px"],
                            data_type="Experimental PEEM", metadata={"source": name})
            self.viewer.load_data(td)
            view = self.viewer.primary_view()
            view.combo_x.setCurrentIndex(2)
            view.combo_y.setCurrentIndex(1)
            view.combo_layer.currentIndexChanged.connect(self._on_layer_changed)
            view.canvas.mpl_connect("draw_event", lambda _e, v=view: (
                self._sync_range() if getattr(v, "fixed_limits", None) is None else None))
            self._loaded_name = name
        kind, var = select or (self._shown() if base == "stack" or name == self._shown_name else ("nc", "data"))
        self._shown_name = name
        self.viewer.set_layers(self.service.shown_layers(name), select=self.service.layer_for(name, kind, var),
                               show_data=False)  # the data is the stack's A+B image, listed as a layer
        self._on_layer_changed()
        self._fill_show(name)
        i = self.show_combo.findData(f"{kind}|{var}")
        if i >= 0:
            self.show_combo.setCurrentIndex(i)
        self._fill_compare_refs()
        self._fill_profile_list(self.service.list_profiles(name))
        self._dock_profiles()

    def clear(self):
        """No map for the current dataset (yet): empty the Map view and the Show list."""
        if self.viewer.tensor_data is not None:
            self.viewer._clear_tree()
            self.viewer.views.clear()
            self.viewer.tensor_data = None
        self._loaded_name, self._line = None, None
        self.show_combo.clear()

    def _fill_compare_refs(self):
        name = self._get_name()
        cur = self.cmp_ref.currentText()
        self.cmp_ref.clear()
        if name:
            self.cmp_ref.addItems(self.service.compare_candidates(name))
            if cur:
                self.cmp_ref.setCurrentText(cur)

    def _fill_show(self, name: str):
        """The same short list as the Map view's Layer menu (PeemService.shown_layers), in words."""
        self.show_combo.blockSignals(True)
        self.show_combo.clear()
        for layer in self.service.shown_layers(name):
            kind, var = self.service.parse_layer(layer)
            self.show_combo.addItem(self._show_label(kind, var, layer), f"{kind}|{var}")  # Qt turns tuples into lists
        self.show_combo.blockSignals(False)

    @staticmethod
    def _show_label(kind: str, var: str, layer: str) -> str:
        if kind == "stack":
            return layer.split(" · ", 1)[1].upper().replace("+", " + ") if var == "data" else \
                f"{var.split('_', 1)[-1].upper()} (mean)"
        if kind in ("pcd", "pld"):
            what = "XMCD" if kind == "pcd" else "XLD"
            return {"data": f"{what} map", "difference": f"{what} difference"}.get(var, f"{what} · {var}")
        if kind == "domain_map":
            return "domain map"
        return layer.replace(" · ", ": ")

    def _select(self, kind: str, var: str = "data"):
        self.refresh(select=(kind, var))

    def _on_layer_changed(self, *_):
        name = self._get_name()
        if not name or self.viewer.tensor_data is None:
            return
        kind, var = self._shown()
        try:
            hint = self.service.display_hint(name, kind, var)
        except ValueError:
            return
        self.viewer.configure_view(**hint, image_origin=True, aspect_equal=True, limits="auto")
        i = self.show_combo.findData(f"{kind}|{var}")  # Show follows the Layer menu
        if i >= 0 and i != self.show_combo.currentIndex():
            self.show_combo.blockSignals(True)
            self.show_combo.setCurrentIndex(i)
            self.show_combo.blockSignals(False)
        self._sync_range()
        self._update_overlays()
        self._dock_profiles()  # the drawn line's profile follows the shown map

    def _update_overlays(self):
        name = self._get_name()
        if not name or self.viewer.tensor_data is None:
            return
        ov = []
        roi = self._roi()
        if roi:
            ov.append({**roi, "color": "yellow"})
        for i, pr in enumerate(self.service.list_profiles(name)):
            if pr.get("visible", True):
                ov.append({"kind": "line", "a": pr["line"][0], "b": pr["line"][1], "color": peem_plots.profile_color(i)})
        if self._line:
            ov.append({"kind": "line", "a": self._line[0], "b": self._line[1], "color": "orange"})
        self.viewer.set_overlays(ov)

    def _sync_range(self):
        lim = self.viewer.color_limits()
        if lim is None:
            return
        step = max((lim[1] - lim[0]) / 50, 1e-6)
        for sp, val in ((self.range_min, lim[0]), (self.range_max, lim[1])):
            sp.blockSignals(True)
            sp.setSingleStep(step)
            sp.setValue(val)
            sp.blockSignals(False)

    def _apply_range(self, *_):
        lo, hi = self.range_min.value(), self.range_max.value()
        if hi > lo:
            self.viewer.configure_view(limits=(lo, hi))

    def _auto_range(self):
        self.viewer.configure_view(limits="auto")
        self._sync_range()

    def _start_tool(self, kind: str, cb, color: str):
        if self.viewer.tensor_data is None:
            self._err("Make the asymmetry map first (4.): drawing works on the Map view.")
            return
        self._show_map_tab()  # the tools draw on the Map view, not the Frame view
        self.viewer.start_tool(kind, cb, color)
        self.status.setText({"line": "Line: click the start, then the end point on the Map view.",
                             "rect": "Rectangle: press and drag on the Map view.",
                             "polygon": "Polygon: click the corners on the Map view; click the first corner "
                                        "again to close."}.get(kind, "Draw on the Map view."))

    def _show_map_tab(self):
        from PySide6.QtWidgets import QTabWidget

        w = self.map_view.parentWidget()
        while w is not None and not isinstance(w, QTabWidget):
            w = w.parentWidget()
        if w is not None:
            w.setCurrentWidget(self.map_view)

    # ------------------------------------------------------------------ 4. dichroism map
    def _asymmetry(self):
        """One button for the dichroism map. After Stack Pairs: the corrected pairs (energy window
        optional), the second polarization refined onto the first if Register is on. Without Stack
        Pairs: all frames by their polarization tags, every frame aligned if Register is on
        (PeemService.make_pcd_map / make_pld_map, then map_to_tools)."""
        name = self._name()
        if not name:
            return
        try:
            lo, hi = _opt_float(self.e_lo.text()), _opt_float(self.e_hi.text())
        except ValueError:
            self._err("Energy window: type numbers (eV) or leave empty for all.")
            return
        register = bool(self.map_register.currentData())
        if self.service.get_meta(name).get("processed_is_paired"):
            st = self._call(self.service.channel_stack, name, lo, hi, register)
            if st is None:
                return
            res = self._call(self.service.asymmetry, name)
            if res is None:
                return
            win = st.get("energy_window_eV")
            src = f"{st['n_pairs']} pairs{f' at {win[0]:.2f}–{win[1]:.2f} eV' if win else ''}"
        else:
            if lo is not None or hi is not None:
                self._err("An energy window needs 2. Stack Pairs first (each frame is then picked by its energy).")
                return
            kind = "pld" if self._pair_tags == ["LH", "LV"] else "pcd"
            made = self._call(getattr(self.service, f"make_{kind}_map"), name, register=register)
            if made is None:
                return
            res = self._call(self.service.map_to_tools, name, kind)
            if res is None:
                return
            src = f"all frames ({made['source']})"
        self.status.setText(f"{src}; {res['formula']}: mean {res.get('mean', float('nan')):+.4f}, "
                            f"std {res.get('std', float('nan')):.4f}")
        if self._on_map_made is not None:
            self._on_map_made()
        self.refresh(select=(res["kind"], "data"))
        self._show_map_tab()

    # ------------------------------------------------------------------ 7. compare
    def _compare(self):
        name = self._name()
        if not name:
            return
        ref = self.cmp_ref.currentText()
        if not ref:
            self._err("Load the reference dataset and make its asymmetry map (4.) first, then press ↻.")
            return
        self.status.setText(f"Registering {ref} onto {name} …")
        self.status.repaint()
        res = self._call(self.service.compare_maps, name, ref, search=self.cmp_search.currentData(),
                         rotation=self.cmp_rot.value(), smooth_reference=self.cmp_smooth.value())
        if res is None:
            self.status.setText("")
            return
        self.status.setText(
            f"{res['formula']}: overlap r {res['overlap_r']:.2f}, rotation {res['rotation_deg']:+.1f}°, "
            f"scale {res['scale']:.3f}, shift ({res['shift_dy_px']:+.0f}, {res['shift_dx_px']:+.0f}) px; "
            f"noise ratio {res['noise_ratio']:.2f}" + (f" — {res['warning']}" if res.get("warning") else ""))
        self.refresh(select=("compare", "data"))

    def _compare_figure(self):
        name = self._name()
        if not name:
            return
        if "compare" not in self.service.list_products(name):
            self._err("Run Register + subtract first.")
            return
        path, _ = QFileDialog.getSaveFileName(self._parent, "Save figure", f"{name}_compare.pdf",
                                              "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)")
        if path:
            out = self._call(self.service.export_compare_figure, name, path)
            if out:
                self.status.setText(f"Saved {os.path.basename(out)}")

    # ------------------------------------------------------------------ 5. analysis
    def _on_line(self, sh: dict):
        self._line = (tuple(sh["a"]), tuple(sh["b"]))
        (y1, x1), (y2, x2) = self._line
        self.line_edit.setText(f"{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}")
        self._update_overlays()
        self._dock_profiles()
        self.status.setText(f"Line ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f}) px: Add profile, Histogram, "
                            "FFT or Domain width use it.")

    def _set_line(self):
        try:
            x1, y1, x2, y2 = _numbers(self.line_edit.text(), 4)
        except ValueError as exc:
            self._err(str(exc))
            return
        self._on_line({"a": (y1, x1), "b": (y2, x2)})

    def _plot(self, kind: str, title: str):
        self._plot_kind = kind
        self.plot_title.setText(title)
        self.pfig.clear()
        return self.pfig.add_subplot()

    def _add_profile(self):
        name = self._name()
        if not name:
            return
        if not self._line:
            self._err("Draw or type a line first.")
            return
        kind, var = self._shown()
        if self._call(self.service.add_profile, name, *self._line, self.line_hw.value(), kind, var, 0) is not None:
            self._draw_profiles()
            self._update_overlays()

    def _draw_profiles(self):
        name = self._get_name()
        profiles = self.service.list_profiles(name) if name else []
        shown = sum(1 for pr in profiles if pr.get("visible", True))
        ax = self._plot("profiles", f"{shown} of {len(profiles)} line profile(s) shown")
        if profiles:
            peem_plots.line_profiles(ax, profiles, self.prof_offset.value())
        self.pcanvas.draw_idle()
        self._fill_profile_list(profiles)
        self._dock_profiles()

    def _dock_profiles(self):
        """Profiles of the visible lines (and the line being drawn) under or beside the map, by the
        line's orientation: |dx| >= |dy| -> under the map against image x, else beside it against y."""
        if self.viewer.tensor_data is None:
            return
        name = self._get_name()
        bottom, side = [], []
        if name and self.prof_dock.isChecked():
            lines = [(pr["line"], np.asarray(pr["values"], float), peem_plots.profile_color(i), "-")
                     for i, pr in enumerate(self.service.list_profiles(name)) if pr.get("visible", True)]
            if self._line:
                kind, var = self._shown()
                try:
                    cur = self.service.line_profile(name, kind, var, 0, *self._line, self.line_hw.value())
                    lines.append((self._line, np.asarray(cur["values"], float), "orange", "--"))
                except (ValueError, KeyError):
                    pass
            for (a, b), vals, color, ls in lines:
                (y0, x0), (y1, x1) = a, b
                t = np.linspace(0.0, 1.0, len(vals))
                if abs(x1 - x0) >= abs(y1 - y0):
                    bottom.append({"pos": x0 + (x1 - x0) * t, "values": vals, "color": color, "ls": ls})
                else:
                    side.append({"pos": y0 + (y1 - y0) * t, "values": vals, "color": color, "ls": ls})
        self.viewer.set_line_profiles(bottom, side)

    def _fill_profile_list(self, profiles: list[dict]):
        self.prof_list.blockSignals(True)
        self.prof_list.clear()
        for i, pr in enumerate(profiles):
            it = QListWidgetItem(peem_plots.profile_label(i, pr))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if pr.get("visible", True) else Qt.Unchecked)
            it.setForeground(QColor(peem_plots.profile_color(i)))
            self.prof_list.addItem(it)
        self.prof_list.blockSignals(False)

    def _profile_toggled(self, item):
        name = self._get_name()
        if name:
            self.service.set_profile_visible(name, self.prof_list.row(item), item.checkState() == Qt.Checked)
            self._draw_profiles()
            self._update_overlays()

    def _remove_profiles(self):
        name = self._name()
        rows = sorted((self.prof_list.row(it) for it in self.prof_list.selectedItems()), reverse=True)
        if not name or not rows:
            self._err("Select the profile(s) to remove in the list first.")
            return
        for r in rows:
            self.service.remove_profile(name, r)
        self._draw_profiles()
        self._update_overlays()

    def _show_all_profiles(self, visible: bool):
        name = self._get_name()
        if name:
            for i in range(len(self.service.list_profiles(name))):
                self.service.set_profile_visible(name, i, visible)
            self._draw_profiles()
            self._update_overlays()

    def _clear_profiles(self):
        name = self._name()
        if name:
            self.service.clear_profiles(name)
            self._line = None
            self._draw_profiles()
            self._update_overlays()

    def _histogram(self):
        name = self._name()
        if name:
            kind, var = self._shown()
            h = self._call(self.service.histogram, name, kind, var, 0, self._roi())
            if h is not None:
                peem_plots.histogram(self._plot("histogram", f"histogram of {kind} / {var}"), h)
                self.pcanvas.draw_idle()

    def _fft(self):
        name = self._name()
        if name:
            kind, var = self._shown()
            f = self._call(self.service.image_fft, name, kind, var, 0)
            if f is not None:
                self._plot_kind = "fft"
                self.plot_title.setText(f"FFT of {kind} / {var}")
                self.pfig.clear()
                peem_plots.image_fft(self.pfig, f)
                self.pcanvas.draw_idle()

    def _domain_width(self):
        name = self._name()
        if name:
            kind, var = self._shown()
            if kind == "stack":
                kind, var = "pcd" if "pcd" in self.service.list_products(name) else "pld", "data"
            w = self._call(self.service.domain_width, name, kind, var, self._roi())
            if w is not None:
                um = f" = {w['domain_width_um']:.3f} µm (period {w['period_um']:.3f} µm)" if "domain_width_um" in w else ""
                self.status.setText(f"Mean domain width {w['domain_width_px']:.1f} px{um} in "
                                    f"{'the ROI' if self._roi() else 'the whole map'} ({kind} / {var}).")

    # ------------------------------------------------------------------ 6. ROIs / domains
    def _set_roi(self, roi: dict | None):
        name = self._name()
        if name:
            self._call(self.service.set_roi, name, "roi", roi)
            self._update_overlays()

    def _on_roi(self, sh: dict):
        self._set_roi(sh)
        if sh.get("kind") == "rect":
            self.roi_edit.setText(f"{sh['x0']}, {sh['x1']}, {sh['y0']}, {sh['y1']}")
        self.status.setText("ROI set: ROI statistics, Domain map and Domain width use it.")

    def _set_roi_typed(self):
        try:
            x0, x1, y0, y1 = (int(round(v)) for v in _numbers(self.roi_edit.text(), 4))
        except ValueError as exc:
            self._err(str(exc))
            return
        self._set_roi({"kind": "rect", "x0": min(x0, x1), "x1": max(x0, x1), "y0": min(y0, y1), "y1": max(y0, y1)})

    def _roi_stats(self):
        name = self._name()
        if name:
            kind = "pcd" if "pcd" in self.service.list_products(name) else "pld"
            st = self._call(self.service.roi_statistics, name, kind, roi=self._roi())  # split at the ROI mean
            if st is not None:
                fr, avg = st["fractions"], st["average"]
                mean = avg.get("mean") if isinstance(avg, dict) else avg
                where = "the ROI" if self._roi() else "the whole map"
                self.status.setText(f"{kind} in {where}: mean {float(mean):+.4f}; up {fr['f_up']:.2f}, "
                                    f"down {fr['f_down']:.2f}, wall {fr['f_wall']:.2f} (split at the mean)")

    def _domain_map(self):
        name = self._name()
        if name:
            kind = "pcd" if "pcd" in self.service.list_products(name) else "pld"
            r = self._call(self.service.compute_domain_map, name, kind, roi=self._roi(), threshold=self.thr.value())
            if r is not None:
                self.status.setText("Domain map: " + ", ".join(f"{k} {float(r[k]):.2f}" for k in ("f_up", "f_down", "f_zero") if k in r))
                self.refresh(select=("domain_map", "data"))

    def _shown_look(self) -> dict:
        out = {}
        lim = self.viewer.color_limits()
        if lim is not None:
            out["vmin"], out["vmax"] = lim
        view = self.viewer.primary_view()
        if view is not None:
            from tensorspec.gui.components.data_viewer_panel import _mpl_cmap
            out["cmap"] = _mpl_cmap(view.combo_cmap.currentText(), view.chk_invert.isChecked())  # as shown
        return out

    def _export(self):
        name = self._name()
        if not name or self.viewer.tensor_data is None:
            return
        kind, var = self._shown()
        path, _ = QFileDialog.getSaveFileName(self._parent, "Export map", f"{name}_{kind}_{var}.tif", "TIFF (*.tif)")
        if path:
            files = self._call(self.service.export_map, name, kind, path, var, pictures=("png",), **self._shown_look())
            if files:
                self.status.setText("Saved " + ", ".join(os.path.basename(f) for f in files))

    def _export_more(self):
        from tensorspec.gui.components.peem_export_dialog import ExportMapDialog

        name = self._name()
        if not name or self.viewer.tensor_data is None:
            return
        kind, var = self._shown()
        roi = self._roi()
        box = None
        if roi and roi.get("kind") == "rect":
            box = (roi["x0"], roi["x1"], roi["y0"], roi["y1"])
        elif roi and roi.get("points"):
            xs, ys = [p[0] for p in roi["points"]], [p[1] for p in roi["points"]]
            box = (int(min(xs)), int(max(xs)) + 1, int(min(ys)), int(max(ys)) + 1)
        dlg = ExportMapDialog(self._parent, {**self.service.export_options(name, kind, var), **self._shown_look()}, box)
        if not dlg.exec():
            return
        path, _ = QFileDialog.getSaveFileName(self._parent, "Export map (file name without extension)", f"{name}_{kind}_{var}")
        if path:
            files = self._call(self.service.export_map, name, kind, path, var, **dlg.options())
            if files:
                self.status.setText("Saved " + ", ".join(os.path.basename(f) for f in files))

    def _export_plot_pdf(self):
        path, _ = QFileDialog.getSaveFileName(self._parent, "Export plot as PDF", os.path.expanduser("~/profile.pdf"), "PDF (*.pdf)")
        if not path:
            return
        path = path if path.lower().endswith(".pdf") else path + ".pdf"
        peem_plots.save_pdf(self.pfig, path)
        name = self._get_name()
        shown = [pr for pr in (self.service.list_profiles(name) if name else []) if pr.get("visible", True)]
        if self._plot_kind == "profiles" and shown:
            peem_plots.save_profiles_csv(shown, path[:-4] + ".csv")  # the profiles in the PDF
        self.status.setText(f"Saved {os.path.basename(path)}")

    # ------------------------------------------------------------------ to the Sum Rule window
    def _send(self):
        name = self._name()
        if name and self._send_to_sum_rule is not None:
            try:
                self._send_to_sum_rule(name)
            except ValueError as exc:
                self._err(str(exc))
