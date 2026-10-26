import weakref
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Global registry to track all active Data Viewer windows for crosshair syncing
GLOBAL_SYNC_REGISTRY = weakref.WeakSet()

import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
                               QComboBox, QPushButton, QCheckBox, QFrame, QMenu, QSpinBox, 
                               QFileDialog, QMessageBox, QDoubleSpinBox, QGridLayout, QMainWindow, QSplitter,
                               QInputDialog)
from PySide6.QtCore import Qt, QFileSystemWatcher, QTimer, Signal
from PySide6.QtGui import QCursor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

from tensorspec.core.data_models import TensorData
from tensorspec.gui.components.flow_layout import FlowLayout, group
from tensorspec.core.ml.ssl.reference import (
    FloorReference,
    build_roi_dict,
    integrate_xy_map,
    save_floor_reference,
)


def _default_floor_ref_dir() -> Path:
    """Default folder for saved floor-reference maps.

    Set TENSORSPEC_FLOOR_REF_DIR to use your own folder (e.g. a synced drive).
    """
    return Path(
        os.environ.get("TENSORSPEC_FLOOR_REF_DIR")
        or Path.home() / "TensorSpec" / "floor_references"
    ).expanduser()


def _mpl_cmap(name: str, invert: bool = False) -> str:
    """Menu names with white at low intensity. Matplotlib names pass through. invert: the reversed map."""
    name = {"grey": "gray_r", "blue": "Blues", "terrain": "terrain_r"}.get(name, name)
    if invert:
        name = name[:-2] if name.endswith("_r") else name + "_r"
    return name


def is_ml_label_layer(layer_name: str) -> bool:
    return (
        layer_name.startswith("domains_")
        or layer_name.startswith("Labels_")
        or layer_name == "Supervised Probabilities"
    )


def ml_label_spatial_map(layer_data: np.ndarray, value_shape: tuple) -> np.ndarray | None:
    """Reshape flat ML domain vectors (or prob cubes) to (nY, nX) for spatial maps."""
    if len(value_shape) != 4:
        return None
    nY, nX = value_shape[2], value_shape[3]
    arr = np.asarray(layer_data)
    if arr.ndim == 1 and arr.size == nY * nX:
        return arr.reshape(nY, nX)
    if arr.ndim == 3 and arr.shape[:2] == (nY, nX):
        return np.argmax(arr, axis=2)
    return None

def plane_layer(layer_data, shape: tuple, y_idx: int, x_idx: int) -> np.ndarray | None:
    """A 2-D layer that matches the displayed (y, x) dimensions of an N-D tensor, oriented
    (rows = y). Lets maps (e.g. an XMCD asymmetry over a frame x y x x stack) be shown as
    layers whatever the other axes are."""
    arr = np.asarray(layer_data)
    if arr.ndim != 2:
        return None
    ny, nx = shape[y_idx], shape[x_idx]
    if arr.shape == (ny, nx):
        return arr
    if arr.shape == (nx, ny) and nx != ny:
        return arr.T
    return None


def iter_norm_axis_choices(labels: list[str]) -> list[tuple[str, tuple[int, ...]]]:
    """Build Off / Along-dim / Plane-dim×dim choices from loaded axis labels."""
    choices: list[tuple[str, tuple[int, ...]]] = [("Off", ())]
    n = len(labels)
    for i, lab in enumerate(labels):
        choices.append((f"Along {lab}", (i,)))
    for i in range(n):
        for j in range(i + 1, n):
            choices.append((f"Plane: {labels[i]} × {labels[j]}", (i, j)))
    return choices


def normalize_over_axes(arr: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Display-only: for each fixed index outside *axes*, divide by max over *axes*.

    Example: axes=(Energy, Slit) → each dispersion picture (one per hv, φ, …) unit-max.
    Does not mutate *arr*.
    """
    out = np.array(arr, dtype=float, copy=True)
    if not axes or out.size == 0:
        return out
    for ax in axes:
        if ax < 0 or ax >= out.ndim:
            return out
    peaks = np.nanmax(np.abs(out), axis=axes, keepdims=True)
    bad = (peaks == 0) | ~np.isfinite(peaks)
    peaks = np.where(bad, 1.0, peaks)
    return out / peaks


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, layout='constrained')
        super().__init__(fig)
        from PySide6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

class SliceWidget(QFrame):
    """A modular 2D viewer that synchronizes with the Global N-Dimensional State."""
    def __init__(self, parent_panel, x_idx: int, y_idx: int, grid_row: int = 0, grid_col: int = 0):
        super().__init__(parent_panel)
        self.parent_panel = parent_panel
        self.tensor_data = parent_panel.tensor_data
        self.fixed_limits = None  # (vmin, vmax) typed by a host panel; None = automatic (contrast / Sym)
        
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(2)
        from PySide6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 150)
        
        self.x_idx = x_idx
        self.y_idx = y_idx
        
        # Track position in the puzzle grid
        self.grid_row = grid_row
        self.grid_col = grid_col
        
        self.slider_widgets = {}
        self.spin_widgets = {}
        
        self._init_ui()
        self._populate_dropdowns()

        # Register for cross-window syncing
        GLOBAL_SYNC_REGISTRY.add(self)
        self._is_syncing = False # Prevents infinite feedback loops
        
    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Top Controls (3 rows so mid-width panels do not clip) ---
        ctrl_layout_v = QVBoxLayout()

        # ROW 1: axes + window chrome
        row1 = QHBoxLayout()
        self.combo_x = QComboBox()
        self.combo_y = QComboBox()
        self.combo_layer = QComboBox()
        self.combo_x.currentIndexChanged.connect(self._on_dropdown_changed)
        self.combo_y.currentIndexChanged.connect(self._on_dropdown_changed)
        self.combo_layer.currentIndexChanged.connect(self.redraw)

        row1.addWidget(QLabel("X:"))
        row1.addWidget(self.combo_x)
        row1.addWidget(QLabel("Y:"))
        row1.addWidget(self.combo_y)
        row1.addWidget(QLabel("Layer:"))
        row1.addWidget(self.combo_layer)

        self.combo_aspect = QComboBox()
        self.combo_aspect.addItems(["Stretch (Auto)", "Square (equal)"])
        self.combo_aspect.currentIndexChanged.connect(self.update_aspect)
        row1.addWidget(self.combo_aspect)
        row1.addStretch(1)

        self.btn_toggle = QPushButton("Toggle UI")
        self.btn_toggle.setStyleSheet("font-weight: bold; background-color: #5bc0de; color: black;")
        self.btn_toggle.setToolTip("Hide/show Δ integration controls and dimension sliders.")
        self.btn_toggle.clicked.connect(self.toggle_ui_controls)
        row1.addWidget(self.btn_toggle)

        btn_close = QPushButton("✕")
        btn_close.setFixedWidth(25)
        btn_close.setStyleSheet("background-color: #d9534f; color: white;")
        btn_close.clicked.connect(self.close_widget)
        row1.addWidget(btn_close)
        self.btn_close = btn_close

        # ROW display: profiles / contrast / sync (was crowding row1)
        self.row_display = QWidget()
        row_disp = FlowLayout(self.row_display)  # wraps to a new line when narrow (no overlapping widgets)
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["magma", "viridis", "plasma", "inferno", "cividis", "hot", "cool", "gray", "twilight",
                                  "bwr", "RdBu_r", "grey", "blue", "terrain"])
        self.combo_cmap.setToolTip(
            "grey / blue / terrain: white = low intensity, dark end = high "
            "(MATLAB-style). bwr = RWB; use with Sym ±"
        )
        self.combo_cmap.setCurrentText("magma")
        self.combo_cmap.currentIndexChanged.connect(self.redraw)
        self.chk_invert = QCheckBox("Invert")
        self.chk_invert.setToolTip("Reverse the colour map (e.g. bwr: red for negative, blue for positive)")
        self.chk_invert.toggled.connect(self.redraw)

        self.btn_export = QPushButton("Save EPS/PDF")
        self.btn_export.setToolTip(
            "PDF or EPS for Illustrator: the map is one image object, "
            "axis letters stay editable text. Cursor lines are left out."
        )
        self.btn_export.clicked.connect(self.export_vector_figure)

        self.chk_profiles = QCheckBox("XY Profiles")
        self.chk_profiles.stateChanged.connect(self.toggle_profiles)

        self.combo_profile_mode = QComboBox()
        self.combo_profile_mode.addItems(["Raw (Sum)", "Mean", "Normalized to Max"])
        self.combo_profile_mode.currentIndexChanged.connect(self.redraw)

        self.sl_contrast = QSlider(Qt.Horizontal)
        self.sl_contrast.setRange(1, 100)
        self.sl_contrast.setValue(100)
        self.sl_contrast.setMinimumWidth(80)
        self.sl_contrast.setMaximumWidth(140)
        self.sl_contrast.setToolTip(
            "Scale color limits from data min toward max (100% = full range)."
        )
        self.sl_contrast.valueChanged.connect(self.redraw)

        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.1, 5.0)
        self.spin_gamma.setValue(1.0)
        self.spin_gamma.setSingleStep(0.1)
        self.spin_gamma.setDecimals(2)
        self.spin_gamma.setFixedWidth(60)
        self.spin_gamma.setToolTip("Gamma for intensity display (1 = linear).")
        self.spin_gamma.valueChanged.connect(self.redraw)

        self.combo_norm = QComboBox()
        self.combo_norm.setMinimumWidth(140)
        self.combo_norm.setToolTip(
            "Display-only max-normalize over chosen dims before the 2D cut "
            "(Along Energy = per spectrum; Plane Energy×Slit = per dispersion picture; "
            "Plane Slit×Deflect = per isoenergy map). Stored data unchanged. "
            "Each snapped panel has its own Norm."
        )
        self.combo_norm.currentIndexChanged.connect(self.redraw)

        self.chk_sym = QCheckBox("Sym ±")
        self.chk_sym.setToolTip("Symmetric colour limits c ± L (white = c with bwr); L = contrast % of the 99th percentile of |v - c|")
        self.chk_sym.toggled.connect(self.redraw)
        self.combo_center = QComboBox()
        self.combo_center.addItems(["0", "median"])
        self.combo_center.setToolTip("Centre of the symmetric limits")
        self.combo_center.currentIndexChanged.connect(self.redraw)
        self.chk_image_origin = QCheckBox("Top-left origin")
        self.chk_image_origin.setToolTip("Image convention: first row at the top (PEEM / camera images)")
        self.chk_image_origin.toggled.connect(self.redraw)

        self.chk_sync = QCheckBox("Sync")
        self.chk_sync.setChecked(False)
        self.chk_sync.setToolTip(
            "Opt-in crosshair lock with other open viewers that share the same Sync group "
            "and matching axis labels. Off by default — viewers stay independent."
        )
        self.spin_sync_group = QSpinBox()
        self.spin_sync_group.setRange(1, 99)
        self.spin_sync_group.setValue(1)
        self.spin_sync_group.setPrefix("grp ")
        self.spin_sync_group.setFixedWidth(70)
        self.spin_sync_group.setEnabled(False)
        self.spin_sync_group.setToolTip(
            "Only viewers with Sync ON and this same group number share crosshairs."
        )
        self.chk_sync.toggled.connect(self.spin_sync_group.setEnabled)
        self._sync_group_box = group(self.chk_sync, self.spin_sync_group)
        for widgets in ((QLabel("Colormap:"), self.combo_cmap, self.chk_invert), (self.btn_export,),
                        (self.chk_profiles, self.combo_profile_mode), (QLabel("Contrast %:"), self.sl_contrast),
                        (QLabel("γ:"), self.spin_gamma), (QLabel("Norm:"), self.combo_norm),
                        (self.chk_sym, self.combo_center), (self.chk_image_origin,)):
            row_disp.addWidget(group(*widgets))
        row_disp.addWidget(self._sync_group_box)
        if getattr(self.parent_panel, "_compact", False):  # embedded single view: no sync / close
            for w in (self._sync_group_box, self.btn_close):
                w.setVisible(False)

        # ROW 2 (collapsible via Toggle UI): Δ + ortho
        self.row2_widget = QWidget()
        row2 = FlowLayout(self.row2_widget)
        self.spin_dx = QSpinBox(); self.spin_dx.setRange(0, 100); self.spin_dx.setSuffix(" px")
        self.slider_dx = QSlider(Qt.Horizontal); self.slider_dx.setRange(0, 100); self.slider_dx.setFixedWidth(50)
        self.spin_dx.valueChanged.connect(self.slider_dx.setValue)
        self.slider_dx.valueChanged.connect(self.spin_dx.setValue)
        self.spin_dx.valueChanged.connect(self._on_delta_changed)

        self.spin_dy = QSpinBox(); self.spin_dy.setRange(0, 100); self.spin_dy.setSuffix(" px")
        self.slider_dy = QSlider(Qt.Horizontal); self.slider_dy.setRange(0, 100); self.slider_dy.setFixedWidth(50)
        self.spin_dy.valueChanged.connect(self.slider_dy.setValue)
        self.slider_dy.valueChanged.connect(self.spin_dy.setValue)
        self.spin_dy.valueChanged.connect(self._on_delta_changed)

        self.chk_ortho = QCheckBox("Extract Orthogonal:")
        self.chk_ortho.stateChanged.connect(self.toggle_profiles)
        self.combo_ortho = QComboBox()
        self.combo_ortho.currentIndexChanged.connect(self.redraw)
        for widgets in ((QLabel("ΔX:"), self.spin_dx, self.slider_dx), (QLabel("ΔY:"), self.spin_dy, self.slider_dy),
                        (self.chk_ortho, self.combo_ortho)):
            row2.addWidget(group(*widgets))

        ctrl_layout_v.addLayout(row1)
        ctrl_layout_v.addWidget(self.row_display)
        ctrl_layout_v.addWidget(self.row2_widget)
        layout.addLayout(ctrl_layout_v)

        # --- Canvas Setup (2x3 Grid) ---
        self.canvas = MplCanvas(self)
        self.gs = self.canvas.figure.add_gridspec(2, 3, width_ratios=[4, 1, 2], height_ratios=[4, 1])

        self.ax_main = self.canvas.figure.add_subplot(self.gs[0, 0])
        self.ax_prof_y = self.canvas.figure.add_subplot(self.gs[0, 1], sharey=self.ax_main)
        self.ax_prof_x = self.canvas.figure.add_subplot(self.gs[1, 0], sharex=self.ax_main)
        self.ax_prof_ortho = self.canvas.figure.add_subplot(self.gs[:, 2])

        self.im_main = self.ax_main.imshow(
            np.zeros((10, 10)), origin='lower', aspect='auto', cmap='magma', rasterized=True
        )
        self.line_x, = self.ax_prof_x.plot([], [], 'b-')
        self.line_y, = self.ax_prof_y.plot([], [], 'r-')
        self.line_ortho, = self.ax_prof_ortho.plot([], [], 'g-', linewidth=1.5)

        self.vline = self.ax_main.axvline(0, color='cyan', ls='--', alpha=0.7)
        self.hline = self.ax_main.axhline(0, color='cyan', ls='--', alpha=0.7)
        self.rect_window = patches.Rectangle((0,0), 0, 0, linewidth=1.5, edgecolor='cyan', facecolor='cyan', alpha=0.2, linestyle=':')
        self.ax_main.add_patch(self.rect_window)

        self.ax_prof_y.set_visible(False)
        self.ax_prof_x.set_visible(False)
        self.ax_prof_ortho.set_visible(False)
        # line profiles drawn by a host panel (set_line_profiles): under / beside the map
        self._line_profiles: dict[str, list] = {"bottom": [], "side": []}
        self._line_profile_artists: list = []
        self.canvas.mpl_connect("draw_event", self._align_docked_profiles)

        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_drag) # NEW: Listen for mouse movement
        layout.addWidget(self.canvas)
        layout.setStretchFactor(self.canvas, 1) # Force canvas to expand vertically!

        # --- Sliders Wrapper (Collapsible) ---
        self.sliders_widget = QWidget()
        self.sliders_layout = QVBoxLayout(self.sliders_widget)
        self.sliders_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.sliders_widget)

        # Automatically collapse GridSpec unused space on boot
        self.toggle_profiles()

    def _populate_dropdowns(self):
        self.combo_x.blockSignals(True)
        self.combo_y.blockSignals(True)
        self.combo_layer.blockSignals(True)
        
        self.combo_x.clear()
        self.combo_y.clear()
        self.combo_layer.clear()
        self.combo_layer.addItem("Intensity")
        layers = self.tensor_data.metadata.get('layers', {}) if hasattr(self.tensor_data, 'metadata') and self.tensor_data.metadata else {}
        for layer_name in layers:
            self.combo_layer.addItem(layer_name)

        for idx, (label, unit) in enumerate(zip(self.tensor_data.labels, self.tensor_data.units)):
            display = f"{label} ({unit})" if unit else label
            self.combo_x.addItem(display, userData=idx)
            self.combo_y.addItem(display, userData=idx)

        alts = {}
        if getattr(self.tensor_data, "metadata", None):
            alts = self.tensor_data.metadata.get("axis_alternates") or {}
        for text, spec in alts.items():
            dim = int(spec["dim"])
            self.combo_x.addItem(text, userData=dim)
            self.combo_y.addItem(text, userData=dim)

        self.combo_x.setCurrentIndex(self.x_idx)
        self.combo_y.setCurrentIndex(self.y_idx)
        self.combo_x.blockSignals(False)
        self.combo_y.blockSignals(False)
        self.combo_layer.blockSignals(False)
        self._refresh_norm_combo()
        self._update_ortho_combo()
        self._rebuild_sliders()

    def _refresh_norm_combo(self):
        """Fill Norm choices from current tensor axis labels; keep prior selection if possible."""
        prev = self.combo_norm.currentData()
        labels = list(self.tensor_data.labels) if self.tensor_data is not None else []
        self.combo_norm.blockSignals(True)
        self.combo_norm.clear()
        for text, axes in iter_norm_axis_choices(labels):
            self.combo_norm.addItem(text, userData=axes)
        restore = 0
        if prev is not None:
            for i in range(self.combo_norm.count()):
                if self.combo_norm.itemData(i) == prev:
                    restore = i
                    break
        self.combo_norm.setCurrentIndex(restore)
        self.combo_norm.blockSignals(False)

    def _selected_norm_axes(self) -> tuple[int, ...]:
        data = self.combo_norm.currentData()
        if not data:
            return ()
        return tuple(int(a) for a in data)

    def _apply_norm_nd(self, data: np.ndarray, layer_text: str) -> np.ndarray:
        axes = self._selected_norm_axes()
        if not axes or is_ml_label_layer(layer_text):
            return data
        if data.ndim != self.tensor_data.ndim:
            return data
        return normalize_over_axes(data, axes)

    def _update_ortho_combo(self):
        self.combo_ortho.blockSignals(True)
        self.combo_ortho.clear()
        for i in range(self.tensor_data.ndim):
            if i not in (self.x_idx, self.y_idx):
                label = self.tensor_data.labels[i]
                unit = self.tensor_data.units[i]
                display = f"{label} ({unit})" if unit else label
                self.combo_ortho.addItem(display, userData=i)
                
        has_rem_dims = self.combo_ortho.count() > 0
        self.chk_ortho.setEnabled(has_rem_dims)
        self.combo_ortho.setEnabled(has_rem_dims)
        if not has_rem_dims: self.chk_ortho.setChecked(False)
        self.combo_ortho.blockSignals(False)

    def _rebuild_sliders(self):
        while self.sliders_layout.count():
            item = self.sliders_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget(): sub.widget().deleteLater()
                    
        self.slider_widgets.clear()
        self.spin_widgets.clear()
        
        for i in range(self.tensor_data.ndim):
            if i in (self.x_idx, self.y_idx): continue
            
            row = QHBoxLayout()
            lbl = QLabel(f"{self.tensor_data.labels[i]}:")
            lbl.setMinimumWidth(80)
            ax_arr = self.tensor_data.axes[i]
            current_idx = self.parent_panel.global_coords[i]
            
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(min(ax_arr), max(ax_arr))
            spin.setValue(ax_arr[current_idx])
            spin.setSuffix(f" {self.tensor_data.units[i]}")
            spin.setKeyboardTracking(False)
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, len(ax_arr)-1)
            slider.setValue(current_idx)
            
            slider.valueChanged.connect(
                lambda val, dim=i: self.parent_panel.update_global_coord(
                    dim, val, cross_sync=self._want_cross_sync(),
                    sync_group=self._sync_group_id(),
                )
            )
            spin.valueChanged.connect(
                lambda val, dim=i, arr=ax_arr: self.parent_panel.update_global_coord(
                    dim, int((np.abs(arr - val)).argmin()),
                    cross_sync=self._want_cross_sync(),
                    sync_group=self._sync_group_id(),
                )
            )
            
            slider.setMinimumWidth(120)  # stays draggable / visible in a narrow window
            row.addWidget(lbl); row.addWidget(spin); row.addWidget(slider, 1)
            self.sliders_layout.addLayout(row)
            if len(ax_arr) > 1:
                # Play on its own line so it never squeezes the slider; "frame n / N" shows it running
                from tensorspec.gui.components.stack_player import StackPlayer
                play_row = QHBoxLayout()
                play_row.addSpacing(80)
                play_row.addWidget(StackPlayer(slider), 1)
                self.sliders_layout.addLayout(play_row)
            self.slider_widgets[i] = slider
            self.spin_widgets[i] = spin

    def _on_dropdown_changed(self):
        self.x_idx = self.combo_x.currentData()
        self.y_idx = self.combo_y.currentData()
        self._update_ortho_combo()
        self._rebuild_sliders()
        self.parent_panel.broadcast_redraw()

    def _want_cross_sync(self) -> bool:
        return bool(self.chk_sync.isChecked())

    def _sync_group_id(self) -> int | None:
        if not self.chk_sync.isChecked():
            return None
        return int(self.spin_sync_group.value())

    def _on_delta_changed(self):
        """Publish integration half-widths for this panel's X/Y dims to the dashboard."""
        self.parent_panel.update_halfwidth(self.x_idx, self.spin_dx.value(), broadcast=False)
        self.parent_panel.update_halfwidth(self.y_idx, self.spin_dy.value(), broadcast=True)

    def _sync_delta_spins_from_global(self):
        half = self.parent_panel.global_halfwidths
        self.spin_dx.blockSignals(True)
        self.slider_dx.blockSignals(True)
        self.spin_dy.blockSignals(True)
        self.slider_dy.blockSignals(True)
        self.spin_dx.setValue(int(half.get(self.x_idx, 0)))
        self.slider_dx.setValue(int(half.get(self.x_idx, 0)))
        self.spin_dy.setValue(int(half.get(self.y_idx, 0)))
        self.slider_dy.setValue(int(half.get(self.y_idx, 0)))
        self.spin_dx.blockSignals(False)
        self.slider_dx.blockSignals(False)
        self.spin_dy.blockSignals(False)
        self.slider_dy.blockSignals(False)

    def _integration_slices(self):
        """Build nd slices: full span on display axes, ±Δ window on hidden axes."""
        coords = self.parent_panel.global_coords
        half = self.parent_panel.global_halfwidths
        shape = self.tensor_data.value.shape
        slices = []
        for i in range(self.tensor_data.ndim):
            if i in (self.x_idx, self.y_idx):
                slices.append(slice(None))
                continue
            c = int(coords[i])
            hw = int(half.get(i, 0))
            n = int(shape[i])
            slices.append(slice(max(0, c - hw), min(n, c + hw + 1)))
        return slices

    def _reduce_hidden_axes(self, arr: np.ndarray, calc_mode: str) -> np.ndarray:
        """Collapse non-display axes after windowed slicing (Sum / Mean)."""
        reduce_axes = tuple(
            i for i in range(arr.ndim) if i not in (self.x_idx, self.y_idx)
        )
        if not reduce_axes:
            out = arr
        elif calc_mode == "Mean":
            out = np.mean(arr, axis=reduce_axes)
        else:
            out = np.sum(arr, axis=reduce_axes)
        out = np.asarray(out)
        if self.x_idx < self.y_idx:
            out = out.T
        return out

    def toggle_ui_controls(self):
        """Collapses or expands the integration controls and dimension sliders."""
        is_visible = self.row2_widget.isVisible()
        self.row2_widget.setVisible(not is_visible)
        self.sliders_widget.setVisible(not is_visible)

    def set_line_profiles(self, bottom: list[dict], side: list[dict]) -> None:
        """Profiles along drawn lines, docked to the map: ``bottom`` under it on the map's x axis
        ({'pos': x coords, 'values', 'color'}), ``side`` beside it on the map's y axis ({'pos': y
        coords, ...}). The axes are shared with the map, so the profiles follow zoom and pan."""
        for a in self._line_profile_artists:
            a.remove()
        self._line_profile_artists = []
        self._line_profiles = {"bottom": list(bottom or []), "side": list(side or [])}
        for pr in self._line_profiles["bottom"]:
            self._line_profile_artists += self.ax_prof_x.plot(pr["pos"], pr["values"], lw=1,
                                                              color=pr.get("color", "k"), ls=pr.get("ls", "-"))
        for pr in self._line_profiles["side"]:
            self._line_profile_artists += self.ax_prof_y.plot(pr["values"], pr["pos"], lw=1,
                                                              color=pr.get("color", "k"), ls=pr.get("ls", "-"))
        for ax, axis in ((self.ax_prof_x, "y"), (self.ax_prof_y, "x")):
            ax.relim()
            ax.autoscale(enable=True, axis=axis)
        # the side panel shares the map's y axis: its own y labels would sit on the map's edge
        self.ax_prof_y.tick_params(axis="y", labelleft=not self._line_profiles["side"])
        self.toggle_profiles()

    def _align_docked_profiles(self, _event=None):
        """With an equal-aspect map the map box is smaller than its grid cell; pin the docked profile
        axes to the map's drawn edges so each pixel column / row lines up with its profile point."""
        lp = self._line_profiles
        if not (lp["bottom"] or lp["side"]) or self.chk_profiles.isChecked():
            return
        fig = self.canvas.figure
        m = self.ax_main.get_position()
        renderer = getattr(_event, "renderer", None) or self.canvas.get_renderer()
        tb = self.ax_main.get_tightbbox(renderer).transformed(fig.transFigure.inverted())  # map + its labels
        moved = False
        for ax, show in ((self.ax_prof_x, lp["bottom"]), (self.ax_prof_y, lp["side"])):
            if not show or not ax.get_visible():
                continue
            ax.set_in_layout(False)  # placed here, not by the layout engine
            b = ax.get_position()
            if ax is self.ax_prof_x:   # same x extent as the map, under its x labels, room for own ticks
                h = max(0.08, min(0.2, tb.y0 - 0.08))
                want = [m.x0, max(0.06, tb.y0 - 0.015 - h), m.width, h]
            else:                      # same y extent as the map, right of it
                w = max(0.08, min(0.18, 0.97 - tb.x1 - 0.02))
                want = [tb.x1 + 0.015, m.y0, w, m.height]
            if max(abs(u - v) for u, v in zip(want, (b.x0, b.y0, b.width, b.height))) > 1e-3:
                ax.set_position(want)
                moved = True
        if moved:
            self.canvas.draw_idle()

    def toggle_profiles(self):
        """Toggles profile visibility and reclaims space for the main canvas. The under-map / beside-map
        axes also show when a host panel docked line profiles there (set_line_profiles)."""
        vis_xy = self.chk_profiles.isChecked()
        vis_ortho = self.chk_ortho.isChecked()
        lp = getattr(self, "_line_profiles", {"bottom": [], "side": []})
        vis_x = vis_xy or bool(lp["bottom"])
        vis_y = vis_xy or bool(lp["side"])
        for ax, docked in ((self.ax_prof_x, lp["bottom"]), (self.ax_prof_y, lp["side"])):
            if not docked or vis_xy:
                ax.set_in_layout(True)
        if (vis_x or vis_y) and not vis_xy and not vis_ortho:
            self.ax_prof_x.set_visible(vis_x)
            self.ax_prof_y.set_visible(vis_y)
            self.ax_prof_ortho.set_visible(False)
            if vis_x and vis_y:
                gs = self.canvas.figure.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
                self.ax_prof_x.set_subplotspec(gs[1, 0])
                self.ax_prof_y.set_subplotspec(gs[0, 1])
            elif vis_x:
                gs = self.canvas.figure.add_gridspec(2, 1, height_ratios=[4, 1])
                self.ax_prof_x.set_subplotspec(gs[1, 0])
            else:
                gs = self.canvas.figure.add_gridspec(1, 2, width_ratios=[4, 1])
                self.ax_prof_y.set_subplotspec(gs[0, 1])
            self.ax_main.set_subplotspec(gs[0, 0])
            self.gs = gs
            self.canvas.draw_idle()
            return

        self.ax_prof_x.set_visible(vis_xy)
        self.ax_prof_y.set_visible(vis_xy)
        self.ax_prof_ortho.set_visible(vis_ortho)
        
        # Rebuild GridSpec dynamically so constrained_layout works perfectly
        if vis_xy and vis_ortho:
            gs = self.canvas.figure.add_gridspec(2, 3, width_ratios=[4, 1, 2], height_ratios=[4, 1])
            self.ax_main.set_subplotspec(gs[0, 0])
            self.ax_prof_y.set_subplotspec(gs[0, 1])
            self.ax_prof_x.set_subplotspec(gs[1, 0])
            self.ax_prof_ortho.set_subplotspec(gs[:, 2])
        elif vis_xy:
            gs = self.canvas.figure.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
            self.ax_main.set_subplotspec(gs[0, 0])
            self.ax_prof_y.set_subplotspec(gs[0, 1])
            self.ax_prof_x.set_subplotspec(gs[1, 0])
        elif vis_ortho:
            gs = self.canvas.figure.add_gridspec(1, 2, width_ratios=[4, 2])
            self.ax_main.set_subplotspec(gs[0, 0])
            self.ax_prof_ortho.set_subplotspec(gs[0, 1])
        else:
            gs = self.canvas.figure.add_gridspec(1, 1)
            self.ax_main.set_subplotspec(gs[0, 0])
            
        self.gs = gs
        self.canvas.draw_idle()
        self.redraw()

    def sync_sliders_to_global(self):
        for i, slider in self.slider_widgets.items():
            slider.blockSignals(True)
            slider.setValue(self.parent_panel.global_coords[i])
            slider.blockSignals(False)
            
        for i, spin in self.spin_widgets.items():
            spin.blockSignals(True)
            val_idx = self.parent_panel.global_coords[i]
            spin.setValue(self.tensor_data.axes[i][val_idx])
            spin.blockSignals(False)

    def update_aspect(self):
        mode = "equal" if "Square" in self.combo_aspect.currentText() else "auto"
        self.ax_main.set_aspect(mode)
        self.canvas.draw_idle()

    def redraw(self):
        # --- NEW: Prevent 1D shape crash ---
        if self.x_idx == self.y_idx:
            return
        
        self.sync_sliders_to_global()
        self._sync_delta_spins_from_global()
        slices = self._integration_slices()
        calc_mode = self.combo_profile_mode.currentText()
                
        layer_text = self.combo_layer.currentText()
        layers = self.tensor_data.metadata.get('layers', {}) if self.tensor_data.metadata else {}
        plane = None
        if layer_text != "Intensity" and layer_text in layers:
            plane = plane_layer(layers[layer_text], self.tensor_data.value.shape, self.y_idx, self.x_idx)
        if plane is not None:
            sliced = np.asarray(plane, dtype=float)
        elif layer_text != "Intensity" and layer_text in layers:
            layer_data = layers[layer_text]
            spatial = ml_label_spatial_map(np.asarray(layer_data), self.tensor_data.value.shape)
            if spatial is not None and {self.x_idx, self.y_idx} <= {2, 3}:
                sliced = spatial
            else:
                data_source = np.asarray(layer_data)
                if data_source.ndim == self.tensor_data.ndim:
                    data_source = self._apply_norm_nd(data_source, layer_text)
                    sliced = self._reduce_hidden_axes(
                        data_source[tuple(slices)], calc_mode
                    )
                elif spatial is not None:
                    sliced = spatial
                else:
                    data_source = self._apply_norm_nd(
                        np.asarray(self.tensor_data.value), layer_text
                    )
                    sliced = self._reduce_hidden_axes(
                        data_source[tuple(slices)], calc_mode
                    )
        else:
            data_source = self._apply_norm_nd(
                np.asarray(self.tensor_data.value), layer_text
            )
            sliced = self._reduce_hidden_axes(
                data_source[tuple(slices)], calc_mode
            )
        
        x_arr, y_arr = self._axis_values(self.x_idx), self._axis_values(self.y_idx)
        dx_step = (x_arr[-1] - x_arr[0]) / max(1, len(x_arr) - 1) if len(x_arr) > 1 else 0.1
        dy_step = (y_arr[-1] - y_arr[0]) / max(1, len(y_arr) - 1) if len(y_arr) > 1 else 0.1
        extent = (x_arr[0] - dx_step/2, x_arr[-1] + dx_step/2, y_arr[0] - dy_step/2, y_arr[-1] + dy_step/2)
        
        self.im_main.set_data(sliced)
        self.im_main.set_extent(extent)
        if is_ml_label_layer(layer_text):
            import matplotlib.colors as mcolors

            self.im_main.set_norm(mcolors.Normalize(vmin=-0.5, vmax=19.5))
            try:
                import matplotlib as mpl
                self.im_main.set_cmap(mpl.colormaps["tab20"].with_extremes(under="#333333"))
            except Exception:
                self.im_main.set_cmap("tab20")
        else:
            import matplotlib.colors as mcolors

            finite = sliced[np.isfinite(sliced)]
            if finite.size == 0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin = float(np.nanmin(finite))
                vmax = float(np.nanmax(finite))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
            c_scale = max(0.01, float(self.sl_contrast.value()) / 100.0)
            vmax_adj = vmin + (vmax - vmin) * c_scale + 1e-8
            if self.chk_sym.isChecked() and finite.size:
                c = float(np.median(finite)) if self.combo_center.currentText() == "median" else 0.0
                half = float(np.percentile(np.abs(finite - c), 99)) * c_scale or 1.0  # robust to outliers
                vmin, vmax_adj = c - half, c + half
            if self.fixed_limits is not None:
                vmin, vmax_adj = self.fixed_limits
            self.shown_limits = (float(vmin), float(vmax_adj))
            gamma = float(self.spin_gamma.value())
            cmap_name = _mpl_cmap(self.combo_cmap.currentText(), self.chk_invert.isChecked())
            self.im_main.set_cmap(cmap_name)
            if abs(gamma - 1.0) < 1e-6:
                self.im_main.set_norm(mcolors.Normalize(vmin=vmin, vmax=vmax_adj))
            else:
                self.im_main.set_norm(
                    mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax_adj)
                )
        
        self.ax_main.set_xlabel(self.combo_x.currentText())
        self.ax_main.set_ylabel(self.combo_y.currentText())
        
        x_cross = self.parent_panel.global_coords[self.x_idx]
        y_cross = self.parent_panel.global_coords[self.y_idx]
        dx_px = int(self.parent_panel.global_halfwidths.get(self.x_idx, 0))
        dy_px = int(self.parent_panel.global_halfwidths.get(self.y_idx, 0))
        
        x1, x2 = max(0, x_cross - dx_px), min(sliced.shape[1], x_cross + dx_px + 1)
        y1, y2 = max(0, y_cross - dy_px), min(sliced.shape[0], y_cross + dy_px + 1)
        
        self.vline.set_xdata([x_arr[x_cross]])
        self.hline.set_ydata([y_arr[y_cross]])
        
        rect_x = x_arr[x1] - dx_step/2
        rect_y = y_arr[y1] - dy_step/2
        rect_w = (x2 - x1) * dx_step
        rect_h = (y2 - y1) * dy_step
        self.rect_window.set_bounds(rect_x, rect_y, rect_w, rect_h)

        if self.chk_profiles.isChecked():
            if calc_mode == "Mean":
                prof_x = np.mean(sliced[y1:y2, :], axis=0)
                prof_y = np.mean(sliced[:, x1:x2], axis=1)
            else:
                prof_x = np.sum(sliced[y1:y2, :], axis=0)
                prof_y = np.sum(sliced[:, x1:x2], axis=1)
                
            if calc_mode == "Normalized to Max":
                mx, my = np.nanmax(prof_x), np.nanmax(prof_y)
                if mx != 0: prof_x = prof_x / mx
                if my != 0: prof_y = prof_y / my
            
            self.line_x.set_data(x_arr, prof_x)
            self.line_y.set_data(prof_y, y_arr)
            self.ax_prof_x.relim()
            self.ax_prof_x.autoscale(enable=True, axis='y')
            self.ax_prof_y.relim()
            self.ax_prof_y.autoscale(enable=True, axis='x')
            
        if self.chk_ortho.isChecked() and self.combo_ortho.count() > 0:
            ortho_idx = self.combo_ortho.currentData()
            half = self.parent_panel.global_halfwidths
            coords = self.parent_panel.global_coords
            shape = self.tensor_data.value.shape
            slices_ortho = []
            for i in range(self.tensor_data.ndim):
                if i == ortho_idx:
                    slices_ortho.append(slice(None))
                elif i == self.x_idx:
                    slices_ortho.append(slice(x1, x2))
                elif i == self.y_idx:
                    slices_ortho.append(slice(y1, y2))
                else:
                    c = int(coords[i])
                    hw = int(half.get(i, 0))
                    n = int(shape[i])
                    slices_ortho.append(slice(max(0, c - hw), min(n, c + hw + 1)))
                    
            ortho_chunk = self.tensor_data.value[tuple(slices_ortho)]
            eval_axes = tuple(i for i in range(self.tensor_data.ndim) if i != ortho_idx)
            
            if calc_mode == "Mean":
                prof_ortho = np.mean(ortho_chunk, axis=eval_axes)
            else:
                prof_ortho = np.sum(ortho_chunk, axis=eval_axes)
                
            if calc_mode == "Normalized to Max":
                mo = np.nanmax(prof_ortho)
                if mo != 0: prof_ortho = prof_ortho / mo
            
            ortho_arr = self.tensor_data.axes[ortho_idx]
            self.line_ortho.set_data(ortho_arr, prof_ortho)
            self.ax_prof_ortho.set_xlabel(self.combo_ortho.currentText())
            self.ax_prof_ortho.set_ylabel(f"Intensity ({calc_mode})")
            self.ax_prof_ortho.relim(); self.ax_prof_ortho.autoscale_view()
        
        self.ax_main.set_xlim(extent[0], extent[1])
        if self.chk_image_origin.isChecked():
            self.ax_main.set_ylim(extent[3], extent[2])
        else:
            self.ax_main.set_ylim(extent[2], extent[3])
        self._last_sliced = sliced
        self._draw_overlays()
        self.canvas.draw_idle()

    

    # ------------------------------------------------------------- tools / overlays
    def _axis_values(self, dim: int) -> np.ndarray:
        """Coordinates drawn for ``dim``. A same-length alternate (kz vs hv) wins when selected."""
        alts = {}
        if getattr(self.tensor_data, "metadata", None):
            alts = self.tensor_data.metadata.get("axis_alternates") or {}
        for combo, idx in ((self.combo_x, self.x_idx), (self.combo_y, self.y_idx)):
            if idx != dim:
                continue
            spec = alts.get(combo.currentText())
            if spec and int(spec["dim"]) == dim:
                return np.asarray(spec["values"], float)
        return np.asarray(self.tensor_data.axes[dim], float)

    def _axis_to_index(self, val: float, dim: int) -> float:
        """Axis value -> fractional index along ``dim`` (axes assumed uniform)."""
        a = self._axis_values(dim)
        if a.size < 2 or a[-1] == a[0]:
            return 0.0
        return float((val - a[0]) / (a[-1] - a[0]) * (a.size - 1))

    def _index_to_axis(self, idx: float, dim: int) -> float:
        a = self._axis_values(dim)
        if a.size < 2:
            return float(a[0]) if a.size else 0.0
        return float(a[0] + idx * (a[-1] - a[0]) / (a.size - 1))

    def start_tool(self, kind: str, callback, color: str = "magenta", sticky: bool = False,
                   on_cancel=None) -> None:
        """Draw a shape on this view: kind 'rect' | 'polygon' | 'line'. ``callback(shape)``
        receives a dict in pixel-index coordinates of the displayed dims:
        rect {kind, x0, x1, y0, y1}; polygon {kind, points: [[x, y], ...]};
        line {kind, a: (y, x), b: (y, x)}. The crosshair is paused while the tool is active.

        line: press-drag-release with a rubber band (or click, click); Shift snaps to 0 / 45 / 90 deg;
        with ``sticky`` the tool stays on: drag again for a new line, drag an end to move it, Esc to stop
        (``on_cancel()`` is called then)."""
        from matplotlib.widgets import PolygonSelector, RectangleSelector

        self.cancel_tool()
        self._tool_cb = callback
        self._tool_on_cancel = on_cancel
        if kind == "rect":
            def done(press, release):
                xs = sorted((self._axis_to_index(press.xdata, self.x_idx), self._axis_to_index(release.xdata, self.x_idx)))
                ys = sorted((self._axis_to_index(press.ydata, self.y_idx), self._axis_to_index(release.ydata, self.y_idx)))
                self._finish_tool({"kind": "rect", "x0": int(round(xs[0])), "x1": int(round(xs[1])),
                                   "y0": int(round(ys[0])), "y1": int(round(ys[1]))})
            self._tool = RectangleSelector(self.ax_main, done, useblit=True, interactive=False,
                                           props=dict(ec=color, fill=False, lw=1.2))
        elif kind == "polygon":
            def done(verts):
                pts = [[self._axis_to_index(x, self.x_idx), self._axis_to_index(y, self.y_idx)] for x, y in verts]
                self._finish_tool({"kind": "polygon", "points": pts})
            self._tool = PolygonSelector(self.ax_main, done, useblit=True)
        elif kind == "line":
            self._start_line_tool(color, sticky)
        else:
            raise ValueError("tool must be 'rect', 'polygon' or 'line'")

    def _start_line_tool(self, color: str, sticky: bool) -> None:
        """Rubber-band line: press-drag-release (or click, click); Shift snaps the angle to 45 deg steps;
        sticky: stays on, a drag near an end moves that end, Esc stops."""
        from matplotlib.lines import Line2D

        st = {"a": None, "b": None, "grab": None, "press_px": None, "two_click": False}
        band = Line2D([], [], color=color, lw=1.4, marker="o", ms=4, zorder=20)
        self.ax_main.add_line(band)
        prev = getattr(self, "_line_ends", None)  # the last line, so its ends can be dragged
        if sticky and prev:
            st["a"], st["b"] = prev

        def to_shape(a, b):
            ya, xa = self._axis_to_index(a[1], self.y_idx), self._axis_to_index(a[0], self.x_idx)
            yb, xb = self._axis_to_index(b[1], self.y_idx), self._axis_to_index(b[0], self.x_idx)
            return {"kind": "line", "a": (ya, xa), "b": (yb, xb)}

        def snap(a, b, ev):
            mods = getattr(ev, "modifiers", None) or set()  # keys held during the mouse event
            gui = getattr(ev, "guiEvent", None)
            held = bool(gui is not None and hasattr(gui, "modifiers") and gui.modifiers() & Qt.ShiftModifier)
            if not held and "shift" not in mods and "shift" not in str(getattr(ev, "key", None) or ""):
                return b
            dx, dy = b[0] - a[0], b[1] - a[1]
            r = float(np.hypot(dx, dy))
            ang = np.round(np.arctan2(dy, dx) / (np.pi / 4)) * (np.pi / 4)
            return (a[0] + r * np.cos(ang), a[1] + r * np.sin(ang))

        def show(a, b):
            band.set_data([a[0], b[0]], [a[1], b[1]])
            self.canvas.draw_idle()

        def near(ev, pt):
            px = self.ax_main.transData.transform(pt)
            return float(np.hypot(px[0] - ev.x, px[1] - ev.y)) < 10

        def press(ev):
            if ev.inaxes is not self.ax_main or ev.xdata is None or getattr(ev, "button", 1) not in (1, None):
                return
            p = (ev.xdata, ev.ydata)
            if st["two_click"] or (st["grab"] is not None and st["a"] == st["b"]):  # second click of click-click
                st["b"], st["two_click"], st["grab"] = snap(st["a"], p, ev), False, None
                return finish()
            st["press_px"] = (getattr(ev, "x", None), getattr(ev, "y", None))
            if sticky and st["a"] is not None and st["b"] is not None and (near(ev, st["a"]) or near(ev, st["b"])):
                st["grab"] = "a" if near(ev, st["a"]) else "b"  # move this end
            else:
                st["a"], st["b"], st["grab"] = p, p, "b"
            show(st["a"], st["b"])

        def move(ev):
            if st["grab"] is None or ev.inaxes is not self.ax_main or ev.xdata is None:
                if st["two_click"] and ev.inaxes is self.ax_main and ev.xdata is not None:
                    show(st["a"], snap(st["a"], (ev.xdata, ev.ydata), ev))
                return
            other = st["b"] if st["grab"] == "a" else st["a"]
            st[st["grab"]] = snap(other, (ev.xdata, ev.ydata), ev)
            show(st["a"], st["b"])

        def release(ev):
            if st["grab"] is None:
                return
            if ev.inaxes is self.ax_main and ev.xdata is not None:  # end where released (snapped with Shift)
                other = st["b"] if st["grab"] == "a" else st["a"]
                st[st["grab"]] = snap(other, (ev.xdata, ev.ydata), ev)
            st["grab"] = None
            moved = st["press_px"] is not None and st["press_px"][0] is not None and ev.x is not None and \
                np.hypot(ev.x - st["press_px"][0], ev.y - st["press_px"][1]) > 3
            if not moved and st["a"] == st["b"]:
                st["two_click"] = True  # a click: the next click ends the line
                return
            finish()

        def finish():
            a, b = st["a"], st["b"]
            if a is None or b is None or a == b:
                return
            self._line_ends = (a, b)
            cb = getattr(self, "_tool_cb", None)
            if not sticky:
                self.cancel_tool()
            else:
                band.set_data([], [])
                self.canvas.draw_idle()
            if cb is not None:
                cb(to_shape(a, b))

        def key(ev):
            if ev.key == "escape":
                on_cancel = getattr(self, "_tool_on_cancel", None)
                self.cancel_tool()
                if on_cancel is not None:
                    on_cancel()

        cids = [self.canvas.mpl_connect(e, f) for e, f in (("button_press_event", press), ("motion_notify_event", move),
                                                          ("button_release_event", release), ("key_press_event", key))]
        self.canvas.setFocus()
        self._tool = ("cids", cids, band)

    def _finish_tool(self, shape: dict) -> None:
        cb = getattr(self, "_tool_cb", None)
        self.cancel_tool()
        if cb is not None:
            cb(shape)

    def cancel_tool(self) -> None:
        tool = getattr(self, "_tool", None)
        if isinstance(tool, tuple) and tool[0] == "cids":
            for cid in tool[1]:
                self.canvas.mpl_disconnect(cid)
            try:
                tool[2].remove()
            except (ValueError, NotImplementedError):
                pass
            self.canvas.draw_idle()
        elif isinstance(tool, tuple):
            self.canvas.mpl_disconnect(tool[1])
        elif tool is not None:
            try:
                tool.set_active(False)
                tool.disconnect_events()
            except Exception:
                pass
        self._tool = None
        self._tool_cb = None

    @property
    def tool_active(self) -> bool:
        return getattr(self, "_tool", None) is not None

    def _draw_overlays(self) -> None:
        """Shapes from parent_panel.overlays drawn on views showing their (y, x) dims."""
        for art in getattr(self, "_overlay_artists", []):
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        self._overlay_artists = []
        for ov in getattr(self.parent_panel, "overlays", []) or []:
            if tuple(ov.get("dims", (self.y_idx, self.x_idx))) != (self.y_idx, self.x_idx):
                continue
            col = ov.get("color", "yellow")
            kind = ov.get("kind")
            fx = lambda i: self._index_to_axis(i, self.x_idx)  # noqa: E731
            fy = lambda i: self._index_to_axis(i, self.y_idx)  # noqa: E731
            if kind == "rect":
                art = patches.Rectangle((fx(ov["x0"]), fy(ov["y0"])), fx(ov["x1"]) - fx(ov["x0"]),
                                        fy(ov["y1"]) - fy(ov["y0"]), fill=False, ec=col, lw=1.2)
                self.ax_main.add_patch(art)
            elif kind == "polygon":
                art = patches.Polygon([(fx(x), fy(y)) for x, y in ov["points"]], closed=True, fill=False, ec=col, lw=1.2)
                self.ax_main.add_patch(art)
            elif kind == "ellipse":
                art = patches.Ellipse((fx(ov["cx"]), fy(ov["cy"])), fx(ov["cx"] + ov["rx"]) - fx(ov["cx"] - ov["rx"]),
                                      fy(ov["cy"] + ov["ry"]) - fy(ov["cy"] - ov["ry"]), fill=False, ec=col, lw=1.2)
                self.ax_main.add_patch(art)
            elif kind == "line":
                (y1, x1), (y2, x2) = ov["a"], ov["b"]
                art, = self.ax_main.plot([fx(x1), fx(x2)], [fy(y1), fy(y2)], "-", color=col, lw=1.2)
            elif kind == "points":
                art, = self.ax_main.plot([fx(x) for y, x in ov["points"]], [fy(y) for y, x in ov["points"]],
                                         "x", color=col, ms=5)
            else:
                continue
            self._overlay_artists.append(art)

    def _on_click(self, event):
        if self.tool_active:
            return
        if event.inaxes != self.ax_main: return
        if event.xdata is None or event.ydata is None: return
        
        # --- NEW: Double-click to instantly spawn a new panel to the right ---
        if event.dblclick:
            dashboard = self.parentWidget()
            while dashboard and not hasattr(dashboard, 'spawn_view'):
                dashboard = dashboard.parentWidget()
            if dashboard:
                dashboard.spawn_view(self.x_idx, self.y_idx, self, "Right")
            return
            
        # 1. Map the raw mouse coordinates to the nearest valid index in the data arrays
        x_arr = self._axis_values(self.x_idx)
        y_arr = self._axis_values(self.y_idx)
        
        new_x_idx = int((np.abs(x_arr - event.xdata)).argmin())
        new_y_idx = int((np.abs(y_arr - event.ydata)).argmin())
        
        # 2. Update the global state manager (cross-dataset sync only if 🔗 Sync is on)
        want_sync = self._want_cross_sync()
        sync_group = self._sync_group_id()
        self.parent_panel.update_global_coord(
            self.x_idx, new_x_idx, broadcast=False,
            cross_sync=want_sync, sync_group=sync_group,
        )
        self.parent_panel.update_global_coord(
            self.y_idx, new_y_idx, broadcast=True,
            cross_sync=want_sync, sync_group=sync_group,
        )

        # 3. Broadcast physical crosshair coords to other Sync-enabled windows
        if want_sync and sync_group is not None and not self._is_syncing:
            x_label = self.combo_x.currentText()
            y_label = self.combo_y.currentText()
            
            for widget in list(GLOBAL_SYNC_REGISTRY):
                try:
                    if widget is self:
                        continue
                    if not getattr(widget, 'chk_sync', None) or not widget.chk_sync.isChecked():
                        continue
                    if widget._sync_group_id() != sync_group:
                        continue
                    widget.receive_sync(x_label, event.xdata, y_label, event.ydata)
                except RuntimeError:
                    # Clean up deleted C++ objects silently
                    GLOBAL_SYNC_REGISTRY.discard(widget)

    def receive_sync(self, source_x_label, source_x_val, source_y_label, source_y_val):
        """Receives crosshair coordinates from another window and updates if physical axes match."""
        if self._is_syncing: return
        self._is_syncing = True
        
        try:
            my_x_label = self.combo_x.currentText()
            my_y_label = self.combo_y.currentText()
            x_arr = self._axis_values(self.x_idx)
            y_arr = self._axis_values(self.y_idx)

            if my_x_label == source_x_label:
                new_x_idx = int((np.abs(x_arr - source_x_val)).argmin())
                self.parent_panel.update_global_coord(
                    self.x_idx, new_x_idx, broadcast=False, cross_sync=False
                )
            elif my_x_label == source_y_label:
                new_x_idx = int((np.abs(x_arr - source_y_val)).argmin())
                self.parent_panel.update_global_coord(
                    self.x_idx, new_x_idx, broadcast=False, cross_sync=False
                )

            if my_y_label == source_y_label:
                new_y_idx = int((np.abs(y_arr - source_y_val)).argmin())
                self.parent_panel.update_global_coord(
                    self.y_idx, new_y_idx, broadcast=True, cross_sync=False
                )
            elif my_y_label == source_x_label:
                new_y_idx = int((np.abs(y_arr - source_x_val)).argmin())
                self.parent_panel.update_global_coord(
                    self.y_idx, new_y_idx, broadcast=True, cross_sync=False
                )
            else:
                self.redraw()
        finally:
            self._is_syncing = False
    
    def _on_drag(self, event):
        """Allows the crosshair to update continuously while clicking and dragging."""
        if self.tool_active:
            return
        # event.button == 1 ensures this only triggers if the left mouse button is held down
        if event.inaxes == self.ax_main and event.button == 1:
            self._on_click(event)
    
    def export_profiles(self):
        # (Export logic remains perfectly intact)
        if not self.chk_profiles.isChecked():
            QMessageBox.warning(self, "Export Error", "Please enable '1D Profiles' (checkbox at top) before exporting.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export 1D Profiles", "spectra_profiles.csv", "CSV Files (*.csv);;Text Files (*.txt)")
        if not path: return

        try:
            calc_mode = self.combo_profile_mode.currentText()
            slices = self._integration_slices()
            sliced = self._reduce_hidden_axes(
                self.tensor_data.value[tuple(slices)], calc_mode
            )
                    
            x_arr, y_arr = self._axis_values(self.x_idx), self._axis_values(self.y_idx)
            x_cross = self.parent_panel.global_coords[self.x_idx]
            y_cross = self.parent_panel.global_coords[self.y_idx]
            dx_px = int(self.parent_panel.global_halfwidths.get(self.x_idx, 0))
            dy_px = int(self.parent_panel.global_halfwidths.get(self.y_idx, 0))
            
            x1, x2 = max(0, x_cross - dx_px), min(sliced.shape[1], x_cross + dx_px + 1)
            y1, y2 = max(0, y_cross - dy_px), min(sliced.shape[0], y_cross + dy_px + 1)
            
            if calc_mode == "Mean":
                prof_x = np.mean(sliced[y1:y2, :], axis=0)
                prof_y = np.mean(sliced[:, x1:x2], axis=1)
            else:
                prof_x = np.sum(sliced[y1:y2, :], axis=0)
                prof_y = np.sum(sliced[:, x1:x2], axis=1)
                
            if calc_mode == "Normalized to Max":
                mx, my = np.nanmax(prof_x), np.nanmax(prof_y)
                if mx != 0: prof_x = prof_x / mx
                if my != 0: prof_y = prof_y / my

            max_len = max(len(x_arr), len(y_arr))
            out_x_ax = np.pad(x_arr, (0, max_len - len(x_arr)), constant_values=np.nan)
            out_x_pr = np.pad(prof_x, (0, max_len - len(prof_x)), constant_values=np.nan)
            out_y_ax = np.pad(y_arr, (0, max_len - len(y_arr)), constant_values=np.nan)
            out_y_pr = np.pad(prof_y, (0, max_len - len(prof_y)), constant_values=np.nan)

            x_lbl = f"Horizontal_{self.combo_x.currentText()}"
            y_lbl = f"Vertical_{self.combo_y.currentText()}"
            header = f"{x_lbl},{x_lbl}_Intensity,{y_lbl},{y_lbl}_Intensity"

            data_matrix = np.column_stack((out_x_ax, out_x_pr, out_y_ax, out_y_pr))
            np.savetxt(path, data_matrix, delimiter=',', header=header, comments='', fmt='%g')
            QMessageBox.information(self, "Export Successful", f"Profiles safely exported to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{str(e)}")

    def export_vector_figure(self):
        path, selected = QFileDialog.getSaveFileName(
            self, "Save EPS/PDF", "arpes_figure.pdf",
            "PDF Files (*.pdf);;EPS Files (*.eps);;SVG Files (*.svg)",
        )
        if not path:
            return
        low = path.lower()
        if not low.endswith((".pdf", ".eps", ".svg")):
            if "EPS" in selected:
                path += ".eps"
            elif "SVG" in selected:
                path += ".svg"
            else:
                path += ".pdf"
        hidden = (self.vline, self.hline, self.rect_window)
        was = [a.get_visible() for a in hidden]
        try:
            import matplotlib as mpl
            # 42 = real font characters, not outlined paths, so Illustrator can edit the letters
            mpl.rcParams['pdf.fonttype'] = 42
            mpl.rcParams['ps.fonttype'] = 42
            mpl.rcParams['svg.fonttype'] = 'none'
            self.im_main.set_rasterized(True)
            for a in hidden:
                a.set_visible(False)
            self.canvas.figure.savefig(path, transparent=True, bbox_inches='tight', dpi=300)
            QMessageBox.information(self, "Export Successful", f"Figure saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export figure:\n{str(e)}")
        finally:
            for a, vis in zip(hidden, was):
                a.set_visible(vis)
            self.canvas.draw_idle()

    def contextMenuEvent(self, event):
        """Builds a dynamic right-click menu to handle snapping, de-snapping, and reattaching."""
        menu = QMenu(self)
        
        dashboard = self.parentWidget()
        while dashboard and not hasattr(dashboard, 'get_neighbor'):
            dashboard = dashboard.parentWidget()
            
        if not dashboard:
            return

        is_floating = hasattr(self, 'is_detached') and self.is_detached
        
        if is_floating:
            action_reattach = menu.addAction("🔗 Reattach to Main Grid")
            action_reattach.triggered.connect(lambda: dashboard.reattach_view(self))
        else:
            # NEW: Add options to spawn new connected panels
            spawn_menu = menu.addMenu("➕ Snap New Panel...")
            spawn_menu.addAction("Top").triggered.connect(lambda: dashboard.spawn_view(self.x_idx, self.y_idx, self, "Top"))
            spawn_menu.addAction("Bottom").triggered.connect(lambda: dashboard.spawn_view(self.x_idx, self.y_idx, self, "Bottom"))
            spawn_menu.addAction("Left").triggered.connect(lambda: dashboard.spawn_view(self.x_idx, self.y_idx, self, "Left"))
            spawn_menu.addAction("Right").triggered.connect(lambda: dashboard.spawn_view(self.x_idx, self.y_idx, self, "Right"))
            
            menu.addSeparator()

            # Existing directional de-snap options
            pos = dashboard.get_widget_position(self)
            if pos:
                row, col = pos[0], pos[1]
                neighbors = {
                    "Up": dashboard.get_neighbor(row, col, "up"),
                    "Bottom": dashboard.get_neighbor(row, col, "down"),
                    "Left": dashboard.get_neighbor(row, col, "left"),
                    "Right": dashboard.get_neighbor(row, col, "right")
                }
                
                has_neighbors = False
                for direction, neighbor in neighbors.items():
                    if neighbor:
                        action = menu.addAction(f"✂️ De-snap {direction} Connection")
                        action.triggered.connect(lambda checked, n=neighbor: dashboard.detach_view(n))
                        has_neighbors = True
                        
                if not has_neighbors:
                    action_none = menu.addAction("No connected neighbors to de-snap.")
                    action_none.setEnabled(False)

        menu.addSeparator()
        act_floor = menu.addAction("Save floor reference…")
        act_floor.setEnabled(self._is_yx_view())
        act_floor.triggered.connect(self._save_floor_reference)

        menu.exec(event.globalPos())

    def _axis_label_pair(self):
        td = self.parent_panel.tensor_data
        if td is None:
            return None, None
        try:
            return td.labels[self.y_idx], td.labels[self.x_idx]
        except (IndexError, TypeError):
            return None, None

    def _is_yx_view(self) -> bool:
        y_lab, x_lab = self._axis_label_pair()
        return y_lab == "Y" and x_lab == "X"

    def _reduce_mode_from_combo(self) -> str:
        text = self.combo_profile_mode.currentText().casefold()
        if "mean" in text:
            return "mean"
        return "sum"

    def _save_floor_reference(self):
        """Right-click shortcut: same dialog as the top-bar button."""
        self.parent_panel.save_condition_dialog()

    def close_widget(self):
        self.parent_panel.remove_view(self)


class DataViewerPanel(QWidget):
    """The unified Dashboard that synchronizes crosshairs across all spawned views."""
    
    # --- NEW: Global registry for Cross-Dataset Sync ---
    _active_instances = weakref.WeakSet()
    coordsChanged = Signal()  # cursor / slider position changed (e.g. the frame shown)

    def __init__(self, parent=None, compact: bool = False):
        super().__init__(parent)
        self.tensor_data = None
        self.overlays: list[dict] = []
        self._compact = bool(compact)
        self.views = []
        self.detached_windows = [] 
        
        # Central State Manager
        self.global_coords = {}
        self.global_halfwidths = {}
        
        # Register this panel to the global sync network
        DataViewerPanel._active_instances.add(self)

        self._floor_request_watcher: QFileSystemWatcher | None = None
        self._floor_request_dir: Path | None = None
        self._floor_request_seen: set[str] = set()
        
        self._init_ui()

    def enable_floor_request_watcher(self, request_dir: str | Path) -> None:
        """Watch ``request_dir`` for ``*.request`` files and save floor maps.

        Each request file is JSON: ``{"tag": "001", "reduce_mode": "sum"}``.
        Optional keys: ``out_dir``, ``source_id``. Writes
        ``<out_dir>/<source_stem>_floor_<tag>.npz`` and a sibling ``.done`` file.
        """
        path = Path(request_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._floor_request_dir = path
        self._floor_request_seen = {p.name for p in path.glob("*.request")}
        if self._floor_request_watcher is None:
            self._floor_request_watcher = QFileSystemWatcher(self)
            self._floor_request_watcher.directoryChanged.connect(
                self._on_floor_request_dir_changed
            )
        watched = self._floor_request_watcher.directories()
        if str(path) not in watched:
            self._floor_request_watcher.addPath(str(path))
        # Poll as well: some network /tmp layouts skip directoryChanged.
        if not hasattr(self, "_floor_request_timer"):
            self._floor_request_timer = QTimer(self)
            self._floor_request_timer.setInterval(500)
            self._floor_request_timer.timeout.connect(self._on_floor_request_dir_changed)
            self._floor_request_timer.start()

    def _on_floor_request_dir_changed(self, *_args) -> None:
        if self._floor_request_dir is None or self.tensor_data is None:
            return
        for request_path in sorted(self._floor_request_dir.glob("*.request")):
            if request_path.name in self._floor_request_seen:
                continue
            self._floor_request_seen.add(request_path.name)
            try:
                payload = json.loads(request_path.read_text(encoding="utf-8") or "{}")
            except Exception as exc:
                (request_path.with_suffix(".error")).write_text(str(exc), encoding="utf-8")
                continue
            tag = str(payload.get("tag") or request_path.stem).strip()
            if not tag:
                (request_path.with_suffix(".error")).write_text("missing tag", encoding="utf-8")
                continue
            try:
                saved = self.record_floor_reference(
                    tag=tag,
                    out_dir=payload.get("out_dir"),
                    source_id=payload.get("source_id"),
                    reduce_mode=payload.get("reduce_mode"),
                )
            except Exception as exc:
                (request_path.with_suffix(".error")).write_text(str(exc), encoding="utf-8")
                continue
            done = {
                "path": str(saved["path"]),
                "map_shape": list(saved["map_shape"]),
                "roi": saved["roi"],
            }
            request_path.with_suffix(".done").write_text(
                json.dumps(done, indent=2) + "\n", encoding="utf-8"
            )
            try:
                request_path.unlink()
            except OSError:
                pass

    def record_floor_reference(
        self,
        *,
        tag: str,
        out_dir: str | Path | None = None,
        source_id: str | None = None,
        reduce_mode: str | None = None,
    ) -> dict:
        """Save the current Y×X intensity map as a floor reference ``.npz``."""
        if self.tensor_data is None:
            raise ValueError("no data loaded")
        sid = (source_id or self.default_source_id() or "unknown").strip()
        if not sid:
            raise ValueError("source_id required")
        mode = (reduce_mode or self._default_reduce_mode()).lower()
        if mode not in {"sum", "mean"}:
            raise ValueError(f"reduce_mode must be sum or mean, got {mode!r}")
        stem = Path(sid).stem
        destination = Path(out_dir or _default_floor_ref_dir())
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"{stem}_floor_{tag}.npz"
        ref = self.build_floor_reference(
            y_label="Y",
            x_label="X",
            source_id=stem,
            reduce_mode=mode,
        )
        save_floor_reference(path, ref)
        return {
            "path": path,
            "map_shape": tuple(ref.map.shape),
            "roi": ref.roi,
            "tag": tag,
        }

    def _default_reduce_mode(self) -> str:
        for view in self._iter_slice_widgets():
            if view._is_yx_view():
                return view._reduce_mode_from_combo()
        return "sum"

    def _init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self._last_floor_dir = str(_default_floor_ref_dir())

        top_bar = QHBoxLayout()
        self.title_label = QLabel("<b>Dynamic Cross-Correlated Dashboard</b>")
        top_bar.addWidget(self.title_label)
        top_bar.addStretch()
        self.btn_save_condition = QPushButton("Save this condition…")
        self.btn_save_condition.setToolTip(
            "Save the current Y×X intensity map plus energy / deflector / "
            "slit centers and integration widths as a floor reference .npz"
        )
        self.btn_save_condition.setStyleSheet(
            "font-weight: bold; padding: 6px 12px; "
            "background-color: #0F6A8B; color: white;"
        )
        self.btn_save_condition.clicked.connect(self.save_condition_dialog)
        top_bar.addWidget(self.btn_save_condition)
        self.main_layout.addLayout(top_bar)
        if self._compact:  # embedded in another panel: no title / ARPES floor-reference button
            self.title_label.setVisible(False)
            self.btn_save_condition.setVisible(False)

        # Root splitter for vertical rows
        self.v_splitter = QSplitter(Qt.Vertical)
        self.v_splitter.setChildrenCollapsible(False)
        self.main_layout.addWidget(self.v_splitter)
        self.main_layout.setStretch(1, 1)

    def save_condition_dialog(self) -> None:
        """Ask for a tag and folder, then save the live Y×X floor reference."""
        if self.tensor_data is None:
            QMessageBox.warning(self, "Save this condition", "Load data first.")
            return
        if not any(view._is_yx_view() for view in self._iter_slice_widgets()):
            QMessageBox.warning(
                self,
                "Save this condition",
                "Need a panel showing Y × X. Set the axes to Y and X first.",
            )
            return

        default_sid = Path(self.default_source_id() or "scan").stem
        tag, ok = QInputDialog.getText(
            self,
            "Save this condition",
            "Short name for this cut (for example 001, near_EF, defl_mid):",
            text="",
        )
        if not ok:
            return
        tag = (tag or "").strip().replace(" ", "_")
        if not tag:
            QMessageBox.warning(self, "Save this condition", "Name required.")
            return

        folder = QFileDialog.getExistingDirectory(
            self,
            "Folder for floor references",
            self._last_floor_dir,
        )
        if not folder:
            return
        self._last_floor_dir = folder

        path = Path(folder) / f"{default_sid}_floor_{tag}.npz"
        if path.exists():
            answer = QMessageBox.question(
                self,
                "Save this condition",
                f"{path.name} already exists.\nOverwrite?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        try:
            ref = self.build_floor_reference(
                y_label="Y",
                x_label="X",
                source_id=default_sid,
                reduce_mode=self._default_reduce_mode(),
            )
            save_floor_reference(path, ref)
        except Exception as exc:
            QMessageBox.critical(self, "Save this condition", str(exc))
            return

        dims = ref.roi.get("dims", [])
        lines = [
            f"Saved {path}",
            f"map shape {tuple(ref.map.shape)}",
            f"reduce_mode {ref.roi.get('reduce_mode')}",
        ]
        for dim in dims:
            lines.append(
                f"{dim['label']}: center_index={dim['center_index']}  "
                f"halfwidth_px={dim['halfwidth_px']}  "
                f"[{dim['physical_lo']:.6g}, {dim['physical_hi']:.6g}]"
            )
        QMessageBox.information(self, "Save this condition", "\n".join(lines))
        
    def load_data(self, tensor_data: TensorData):
        self.tensor_data = tensor_data
        self.global_coords = {i: len(ax)//2 for i, ax in enumerate(self.tensor_data.axes)}
        self.global_halfwidths = {i: 0 for i in range(self.tensor_data.ndim)}
        
        self._clear_tree()
        from tensorspec.gui.components.stack_player import StackPlayer
        for old in self.views:  # discarded views must go, or their Play timers keep moving the frame
            for player in old.findChildren(StackPlayer):
                player.stop()
            old.deleteLater()
        self.views.clear()

        default_x = 1 if self.tensor_data.ndim > 1 else 0
        default_y = 0
        self.spawn_view(default_x, default_y)

    @staticmethod
    def _canonicalize_ssl_labels(labels: list[str]) -> list[str]:
        """Map loader aliases to Energy/Angle expected by reference APIs.

        Detector slit stays ``Angle``. Deflector names that contain ``slit``
        (for example ``Slit Defl.``) are left alone so a 5D cube keeps three
        independent hidden axes.
        """
        out: list[str] = []
        for lab in labels:
            key = lab.strip().casefold()
            if key == "energy" or key.startswith("energy "):
                out.append("Energy")
            elif key == "angle" or key.startswith("angle ") or "slit angle" in key:
                out.append("Angle")
            else:
                out.append(lab)
        return out

    def default_source_id(self) -> str:
        td = self.tensor_data
        if td is None:
            return ""
        md = td.metadata or {}
        for key in ("source_id", "filename", "filepath", "path", "source_path"):
            val = md.get(key)
            if val:
                return os.path.basename(str(val))
        return ""

    def build_floor_reference(
        self,
        *,
        y_label: str = "Y",
        x_label: str = "X",
        source_id: str,
        reduce_mode: str,
    ) -> FloorReference:
        if self.tensor_data is None:
            raise ValueError("no data loaded")
        labels = list(self.tensor_data.labels)
        if y_label not in labels:
            raise ValueError(f"missing display axis label: {y_label}")
        if x_label not in labels:
            raise ValueError(f"missing display axis label: {x_label}")

        canon = self._canonicalize_ssl_labels(labels)
        if "Energy" not in canon:
            raise ValueError("missing required axis label: Energy")
        if "Angle" not in canon:
            raise ValueError("missing required axis label: Angle")

        value = np.asarray(self.tensor_data.value)
        axes = list(self.tensor_data.axes)
        mode = reduce_mode.lower()
        m = integrate_xy_map(
            value,
            labels=canon,
            coords=dict(self.global_coords),
            halfwidths=dict(self.global_halfwidths),
            y_label=y_label,
            x_label=x_label,
            reduce_mode=mode,
        )
        roi = build_roi_dict(
            labels=canon,
            axes=axes,
            coords=dict(self.global_coords),
            halfwidths=dict(self.global_halfwidths),
            reduce_mode=mode,
            source_id=source_id,
            display_y_label=y_label,
            display_x_label=x_label,
            saved_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        y_ax = axes[labels.index(y_label)]
        x_ax = axes[labels.index(x_label)]
        return FloorReference(
            map=m.astype(np.float32),
            y_axis=y_ax,
            x_axis=x_ax,
            roi=roi,
        )

    def _clear_tree(self):
        """Safely detach all SliceWidgets and destroy intermediate splitters/spacers."""
        for w in self.views:
            w.setParent(None)
        while self.v_splitter.count():
            child = self.v_splitter.widget(0)
            child.setParent(None)
            if isinstance(child, QSplitter):
                child.deleteLater()

    def _rebuild_grid(self):
        """Rebuilds the grid geometry using synchronized QSplitters and spacers."""
        self._clear_tree()
        
        if not self.views:
            return
            
        max_row = max(w.grid_row for w in self.views)
        max_col = max(w.grid_col for w in self.views)
        
        for r in range(max_row + 1):
            h_splitter = QSplitter(Qt.Horizontal)
            
            # Check if this row actually has any widgets
            row_has_widgets = any(w.grid_row == r for w in self.views)
            if not row_has_widgets:
                continue
                
            for c in range(max_col + 1):
                w = next((v for v in self.views if v.grid_row == r and v.grid_col == c), None)
                if w:
                    h_splitter.addWidget(w)
                else:
                    # Blank space to maintain grid alignment
                    spacer = QWidget()
                    spacer.setStyleSheet("background-color: transparent;")
                    h_splitter.addWidget(spacer)
            
            # Distribute width evenly
            h_splitter.setChildrenCollapsible(False)
            h_splitter.setSizes([1000] * (max_col + 1))
            self.v_splitter.addWidget(h_splitter)

        # Distribute height evenly
        self.v_splitter.setChildrenCollapsible(False)
        self.v_splitter.setSizes([1000] * self.v_splitter.count())

    @staticmethod
    def compute_insert_slot(views, ref_row: int, ref_col: int, direction: str) -> tuple[int, int]:
        """Lego insert: shift occupants and return (row, col) for the new panel.

        ``views`` is a sequence of objects with mutable ``grid_row`` / ``grid_col``.
        """
        if direction == "Left":
            row, col = ref_row, ref_col
            for w in views:
                if w.grid_col >= col:
                    w.grid_col += 1
            return row, col
        if direction == "Right":
            row, col = ref_row, ref_col + 1
            for w in views:
                if w.grid_col >= col:
                    w.grid_col += 1
            return row, col
        if direction == "Top":
            row, col = ref_row, ref_col
            for w in views:
                if w.grid_row >= row:
                    w.grid_row += 1
            return row, col
        if direction == "Bottom":
            row, col = ref_row + 1, ref_col
            for w in views:
                if w.grid_row >= row:
                    w.grid_row += 1
            return row, col
        return ref_row, ref_col

    def spawn_view(self, x_idx: int, y_idx: int, ref_widget: SliceWidget = None, direction: str = None):
        """Creates a new SliceWidget and places it in the grid relative to ref_widget."""
        row, col = 0, 0
        
        if ref_widget and direction:
            row, col = self.compute_insert_slot(
                self.views, ref_widget.grid_row, ref_widget.grid_col, direction
            )

        new_widget = SliceWidget(self, x_idx, y_idx, row, col)
        self.views.append(new_widget)
        self._rebuild_grid()
        new_widget.redraw()

    def remove_view(self, widget: SliceWidget):
        if len(self.views) <= 1: return 
        if widget in self.views:
            self.views.remove(widget)
        self._rebuild_grid()
        widget.deleteLater()

    def detach_view(self, widget):
        """Rips a panel out of the grid and puts it in an independent floating window."""
        if len(self.views) <= 1:
            QMessageBox.warning(self, "Detach Error", "Cannot detach the last remaining panel.")
            return 
            
        if widget in self.views:
            self.views.remove(widget)
        
        widget.setParent(None)
        self._rebuild_grid()
        
        floating_window = QWidget()
        floating_window.setWindowTitle("Detached TensorSpec Panel")
        floating_window.resize(600, 500)
        floating_window.setWindowFlags(Qt.Window)
        layout = QVBoxLayout(floating_window)
        layout.addWidget(widget)
        widget.is_detached = True
        widget.floating_container = floating_window
        floating_window.show()
        self.detached_windows.append(floating_window)

    def reattach_view(self, widget):
        """Pulls a floating panel back into the grid."""
        if hasattr(widget, 'floating_container'):
            floating_window = widget.floating_container
            floating_window.layout().removeWidget(widget)
            widget.setParent(None)
            floating_window.close()
            if floating_window in self.detached_windows:
                self.detached_windows.remove(floating_window)
            widget.is_detached = False
            
            max_row = max((w.grid_row for w in self.views), default=-1)
            widget.grid_row = max_row + 1
            widget.grid_col = 0
            
            if widget not in self.views:
                self.views.append(widget)
            self._rebuild_grid()

    def _iter_slice_widgets(self):
        candidates = list(self.views)
        for win in self.detached_windows:
            try:
                candidates.extend(win.findChildren(SliceWidget))
            except RuntimeError:
                continue
        return candidates

    def accepts_sync_group(self, sync_group: int | None) -> bool:
        """True if any owned SliceWidget has Sync ON for this group id."""
        if sync_group is None:
            return False
        for view in self._iter_slice_widgets():
            chk = getattr(view, 'chk_sync', None)
            if chk is not None and chk.isChecked() and view._sync_group_id() == sync_group:
                return True
        return False

    def update_halfwidth(self, dim_idx: int, half_px: int, broadcast: bool = True):
        self.global_halfwidths[dim_idx] = max(0, int(half_px))
        if broadcast:
            self.broadcast_redraw()

    def update_global_coord(
        self,
        dim_idx: int,
        val_idx: int,
        broadcast=True,
        cross_sync=False,
        sync_group: int | None = None,
    ):
        """Update this panel's cursor index.

        cross_sync defaults False: cross-dataset linking must be opted in via
        🔗 Sync on the source (and peer) viewers with the same Sync group —
        never the silent default.
        """
        self.global_coords[dim_idx] = val_idx
        
        # Caller already opted in via cross_sync=True (from chk_sync on click/slider).
        # Peer must also have Sync ON with the same group id.
        if cross_sync and sync_group is not None and self.tensor_data is not None:
            phys_val = self.tensor_data.axes[dim_idx][val_idx]
            target_label = self.tensor_data.labels[dim_idx]
            target_unit = self.tensor_data.units[dim_idx]
            
            for peer in list(DataViewerPanel._active_instances):
                try:
                    if peer is self or peer.tensor_data is None:
                        continue
                    if not peer.accepts_sync_group(sync_group):
                        continue
                    for p_dim_idx, (p_label, p_unit) in enumerate(zip(peer.tensor_data.labels, peer.tensor_data.units)):
                        if p_label == target_label and p_unit == target_unit:
                            peer_axis = peer.tensor_data.axes[p_dim_idx]
                            nearest_idx = int(np.argmin(np.abs(peer_axis - phys_val)))
                            if peer.global_coords.get(p_dim_idx) != nearest_idx:
                                peer.update_global_coord(
                                    p_dim_idx, nearest_idx, broadcast=True, cross_sync=False
                                )
                            break 
                except RuntimeError:
                    DataViewerPanel._active_instances.discard(peer)
        
        if broadcast:
            self.broadcast_redraw()
        self.coordsChanged.emit()

    # ------------------------------------------------------------- host-panel API
    def set_layers(self, layers: dict, select: str | None = None, show_data: bool = True) -> None:
        """Replace the extra layers (name -> array; 2-D (y, x) maps work for any N-D data) and
        refresh the layer menus; ``select`` shows that layer ('Intensity' = the data). show_data=False
        leaves 'Intensity' out of the menu (when one of the layers already is the data)."""
        if self.tensor_data is None:
            return
        meta = dict(self.tensor_data.metadata or {})
        meta["layers"] = dict(layers)
        self.tensor_data.metadata = meta
        for view in self._iter_slice_widgets():
            current = select or view.combo_layer.currentText()
            view.combo_layer.blockSignals(True)
            view.combo_layer.clear()
            if show_data:
                view.combo_layer.addItem("Intensity")
            for name in layers:
                view.combo_layer.addItem(name)
            i = view.combo_layer.findText(current)
            view.combo_layer.setCurrentIndex(i if i >= 0 else 0)
            view.combo_layer.blockSignals(False)
            view.redraw()

    def primary_view(self) -> "SliceWidget | None":
        return self.views[0] if self.views else None

    def start_tool(self, kind: str, callback, color: str = "magenta", sticky: bool = False,
                   on_cancel=None) -> None:
        """Draw a shape on the first view (see SliceWidget.start_tool)."""
        view = self.primary_view()
        if view is None:
            raise ValueError("no data shown")
        view.start_tool(kind, callback, color, sticky=sticky, on_cancel=on_cancel)

    def cancel_tool(self) -> None:
        for view in self._iter_slice_widgets():
            view.cancel_tool()

    def set_line_profiles(self, bottom: list[dict], side: list[dict]) -> None:
        """Line profiles docked to the first view's map (SliceWidget.set_line_profiles)."""
        view = self.primary_view()
        if view is not None:
            view.set_line_profiles(bottom, side)

    def set_overlays(self, overlays: list[dict]) -> None:
        """Shapes to draw (pixel-index coords; optional 'dims': (y_idx, x_idx), 'color')."""
        self.overlays = list(overlays or [])
        for view in self._iter_slice_widgets():
            view._draw_overlays()
            view.canvas.draw_idle()

    def current_image(self) -> np.ndarray | None:
        """The 2-D array the first view shows (layer or intensity slice), rows = y."""
        view = self.primary_view()
        return None if view is None else getattr(view, "_last_sliced", None)

    def color_limits(self) -> tuple[float, float] | None:
        """(vmin, vmax) of the colour scale the first view shows now."""
        view = self.primary_view()
        return None if view is None else getattr(view, "shown_limits", None)

    def configure_view(self, cmap: str | None = None, symmetric: bool | None = None, center: str | None = None,
                       image_origin: bool | None = None, aspect_equal: bool | None = None,
                       limits: tuple[float, float] | str | None = None) -> None:
        """Set display options on every view (host panels, e.g. PEEM XMCD maps).
        limits: (vmin, vmax) fixes the colour range; "auto" goes back to the automatic one."""
        for v in self._iter_slice_widgets():
            if limits is not None:
                v.fixed_limits = None if limits == "auto" else (float(limits[0]), float(limits[1]))
            for w in (v.combo_cmap, v.chk_sym, v.combo_center, v.chk_image_origin, v.combo_aspect):
                w.blockSignals(True)
            if cmap is not None and v.combo_cmap.findText(cmap) >= 0:
                v.combo_cmap.setCurrentText(cmap)
            if symmetric is not None:
                v.chk_sym.setChecked(symmetric)
            if center is not None:
                v.combo_center.setCurrentText(center)
            if image_origin is not None:
                v.chk_image_origin.setChecked(image_origin)
            if aspect_equal is not None:
                v.combo_aspect.setCurrentIndex(1 if aspect_equal else 0)
            for w in (v.combo_cmap, v.chk_sym, v.combo_center, v.chk_image_origin, v.combo_aspect):
                w.blockSignals(False)
            v.update_aspect()
            v.redraw()

    def get_widget_position(self, widget):
        if widget not in self.views:
            return None
        return (widget.grid_row, widget.grid_col, 1, 1)

    def get_neighbor(self, row, col, direction):
        if direction == "up": r, c = row - 1, col
        elif direction == "down": r, c = row + 1, col
        elif direction == "left": r, c = row, col - 1
        elif direction == "right": r, c = row, col + 1
        else: return None
        
        for w in self.views:
            if w.grid_row == r and w.grid_col == c:
                return w
        return None

    def get_current_coords(self):
        return (self.global_coords.get(1, 0), self.global_coords.get(0, 0))

    def get_slider_values(self):
        return (0, 0, 0, 0)

    def get_dispersion_contrast(self):
        """Match Maestro4DViewer API for ML cluster/dendrogram contrast."""
        for view in self._iter_slice_widgets():
            sl = getattr(view, "sl_contrast", None)
            if sl is not None:
                return float(sl.value()) / 100.0
        return 1.0

    def sync_ml_layers(self, source_dict: dict):
        """Merge ML domain / label arrays into viewer metadata and refresh layer combos."""
        if self.tensor_data is None:
            return
        meta = dict(self.tensor_data.metadata or {})
        layers = dict(meta.get("layers") or {})
        for key, val in source_dict.items():
            if key.startswith(("Labels_", "domains_", "probs_")) or key == "Supervised Probabilities":
                layers[key] = val
        meta["layers"] = layers
        self.tensor_data.metadata = meta
        for view in self.views:
            current = view.combo_layer.currentText()
            view.combo_layer.blockSignals(True)
            view.combo_layer.clear()
            view.combo_layer.addItem("Intensity")
            for layer_name in layers:
                view.combo_layer.addItem(layer_name)
            restore = view.combo_layer.findText(current)
            view.combo_layer.setCurrentIndex(restore if restore >= 0 else 0)
            view.combo_layer.blockSignals(False)

    def focus_spatial_layer(self, layer_name: str, y_idx: int = 2, x_idx: int = 3):
        """Show an ML label layer on the Y vs X spatial map."""
        view = next((v for v in self.views if v.x_idx == x_idx and v.y_idx == y_idx), None)
        if view is None:
            self.spawn_view(x_idx, y_idx)
            view = next(v for v in self.views if v.x_idx == x_idx and v.y_idx == y_idx)
        if view.combo_layer.findText(layer_name) < 0:
            view.combo_layer.addItem(layer_name)
        view.combo_layer.setCurrentText(layer_name)
        view.redraw()

    def add_overlay_mode(self, mode_name):
        """Legacy hook — prefer sync_ml_layers() so combo entries have backing arrays."""
        for view in self.views:
            if hasattr(view, "combo_layer") and view.combo_layer.findText(mode_name) < 0:
                view.combo_layer.addItem(mode_name)

    def broadcast_redraw(self):
        for view in self.views:
            view.redraw()
        for win in self.detached_windows:
            try:
                for view in win.findChildren(SliceWidget):
                    view.redraw()
            except RuntimeError:
                continue
