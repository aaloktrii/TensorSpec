"""XMCD-imaging tab of the PEEM suite: load UE49 (BESSY II), UView (BNL ESM) or P3B (ALS) data, drift-correct, stack,
compute XMCD / XLD maps, analyse them and map domains, in the TensorSpec data viewer.

This file holds the assembly, shared helpers and the embedded-viewer adapter; each workflow
section is a mixin in gui/components/peem_tabs/."""
from __future__ import annotations

import json
import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from tensorspec.core.peem import stack_edits
from tensorspec.core.data_models import TensorData
from tensorspec.gui.components.data_viewer_panel import DataViewerPanel
from tensorspec.gui.services.peem_service import PeemService
from tensorspec.gui.components.peem_tabs.analysis_section import AnalysisSection
from tensorspec.gui.components.peem_tabs.cleanup_section import CleanupSection
from tensorspec.gui.components.peem_tabs.common import _dspin  # noqa: F401 - re-exported
from tensorspec.gui.components.peem_tabs.drift_section import DriftSection
from tensorspec.gui.components.peem_tabs.load_section import LoadSection
from tensorspec.gui.components.peem_tabs.roi_domain_section import RoiDomainSection
from tensorspec.gui.components.peem_tabs.stack_asym_section import StackAsymSection
from tensorspec.plotting import peem_plots

_DIVERGING = {"pcd", "pld", "pmld", "pmnld"}


class _Worker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def run(self):
        try:
            self.done.emit(self._fn(*self._args, **self._kwargs))
        except Exception as exc:  # surfaced in the panel
            self.failed.emit(f"{type(exc).__name__}: {exc}")




class PeemXmcdPanel(LoadSection, CleanupSection, DriftSection, StackAsymSection, AnalysisSection,
                 RoiDomainSection, QWidget):
    """Load any PEEM beamline format (UE49 / UView / P3B), compute pcd / pld / pmld / pmnld / domain_map, ROI statistics."""

    def __init__(self, service: PeemService | None = None, parent=None):
        super().__init__(parent)
        self.service = service or PeemService()
        self._name: str | None = None
        self._scan_paths: list[str] = []
        self._scan_pol: dict[str, str] = {}  # polarization set by the user, per item
        self._scan_detected: dict[str, tuple[str | None, str]] = {}  # detect_pol per item
        self._pending_rois: dict[str, dict | None] = {}  # regions set before any data is loaded
        self._worker: _Worker | None = None
        self._setup_ui()

    # ---------------------------------------------- regions / profiles (stored by the service)
    def _get_region(self, which: str) -> dict | None:
        if self._name:
            return self.service.get_rois(self._name)[which]
        return self._pending_rois.get(which)

    def _set_region(self, which: str, roi: dict | None) -> None:
        if self._name:
            self.service.set_roi(self._name, which, roi)
        else:
            self._pending_rois[which] = roi

    _roi = property(lambda self: self._get_region("roi"), lambda self, v: self._set_region("roi", v))
    _bg_roi = property(lambda self: self._get_region("bg"), lambda self, v: self._set_region("bg", v))
    _reg_roi = property(lambda self: self._get_region("drift_box"),  # drift-correction box (rect)
                        lambda self, v: self._set_region("drift_box", v))

    @property
    def _profiles(self) -> list[dict]:
        return self.service.list_profiles(self._name) if self._name else []

    @property
    def _last_profile(self) -> dict | None:
        profiles = self._profiles
        return profiles[-1] if profiles else None

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        controls = QWidget()
        cl = QVBoxLayout(controls)

        self._build_load(cl)
        self._build_cleanup(cl)
        self._build_drift(cl)
        self._build_stack_asym(cl)
        self._build_analysis(cl)
        self._build_roi_domain(cl)
        cl.addStretch(1)
        # every section can be hidden / shown with its tick box; 7 and 8 start hidden (2 and 6 are
        # built collapsible above)
        for i in range(cl.count()):
            box = cl.itemAt(i).widget()
            if isinstance(box, QGroupBox) and not box.isCheckable():
                hidden = box.title().startswith(("7.", "8."))
                if hidden:
                    box.setTitle(box.title() + " (tick to show)")
                self._make_collapsible(box, open_=not hidden)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(580)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # controls shrink to the pane width
        controls.setMinimumWidth(0)
        splitter.addWidget(scroll)

        right = QWidget()
        rl = QVBoxLayout(right)
        top = QHBoxLayout()
        top.addWidget(QLabel("Frames:"))
        self.view_combo = QComboBox()  # frame source shown as the viewer's Intensity
        self.view_combo.setToolTip("Stack shown as 'Intensity' in the viewer; maps (pcd, stack, domain_map, ...) "
                                   "are the viewer's Layer menu")
        self.view_combo.activated.connect(self._on_view_changed)  # also when the current entry is picked again
        top.addWidget(self.view_combo, 1)
        top.addWidget(QLabel("maps → viewer 'Layer' menu"))
        rl.addLayout(top)
        self.viewer = DataViewerPanel(compact=True)  # TensorSpec N-D data viewer
        rl.addWidget(self.viewer, 1)
        self.plot_box = QWidget()
        pb = QVBoxLayout(self.plot_box)
        pb.setContentsMargins(0, 0, 0, 0)
        ph = QHBoxLayout()
        self.plot_title = QLabel("")
        ph.addWidget(self.plot_title, 1)
        b = QPushButton("Export PDF…")
        b.setToolTip("Vector PDF with editable text (Adobe Illustrator); profiles also as a .csv next to it")
        b.clicked.connect(self._export_plot_pdf)
        ph.addWidget(b)
        b = QPushButton("Close plot")
        b.clicked.connect(lambda: self.plot_box.setVisible(False))
        ph.addWidget(b)
        pb.addLayout(ph)
        self.pfig = Figure(figsize=(6, 2.4))
        self.pcanvas = FigureCanvas(self.pfig)
        self.pcanvas.setMinimumHeight(230)
        pb.addWidget(self.pcanvas)
        self.plot_box.setVisible(False)
        rl.addWidget(self.plot_box)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumHeight(170)
        rl.addWidget(self.output)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([620, 760])

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _make_collapsible(box: QGroupBox, open_: bool = True) -> None:
        """Tick box in the title hides / shows the section's contents. Widgets that were hidden on
        purpose inside the section (e.g. collapsed sub-boxes) stay hidden when it is reopened."""
        box.setCheckable(True)
        kids = [w for w in box.findChildren(QWidget, options=Qt.FindDirectChildrenOnly)]
        hidden_on_purpose: set[int] = set()

        def toggle(on: bool):
            if on:
                for w in kids:
                    if id(w) not in hidden_on_purpose:
                        w.setVisible(True)
            else:
                hidden_on_purpose.clear()
                hidden_on_purpose.update(id(w) for w in kids if w.isHidden())
                for w in kids:
                    w.setVisible(False)
            box.setToolTip("" if on else "tick to show")

        box.toggled.connect(toggle)
        box.setChecked(open_)  # a new checkable box is checked, so False emits toggled(False) once

    # ------------------------------------------------------------ helpers (panel)
    def _log(self, text: str):
        self.output.appendPlainText(text)

    def _err(self, msg: str):
        self._log("ERROR: " + msg)
        if hasattr(self, "load_status") and self.btn_load.isEnabled():
            self.load_status.setText("Error: " + msg[:200])
        QMessageBox.warning(self, "PEEM", msg)

    def _run(self, fn, on_done, *args, **kwargs):
        if self._worker is not None and self._worker.isRunning():
            self._err("Busy: wait for the current operation to finish.")
            return
        self._worker = _Worker(fn, *args, **kwargs)
        self._worker.done.connect(on_done)
        self._worker.failed.connect(self._err)
        self.setCursor(Qt.BusyCursor)
        self._worker.finished.connect(lambda: self.unsetCursor())
        self._worker.start()

    def _require(self) -> str | None:
        if not self._name:
            self._err("Load data first.")
            return None
        return self._name

    def _crop(self):
        return self._roi if self.chk_crop.isChecked() else None

    def _show_result(self, res: dict, show_kind: str | None = None):
        self._log(json.dumps(res, indent=1, default=lambda o: round(float(o), 6)
                             if isinstance(o, (np.floating, float)) else str(o)))
        self._refresh_views(select=show_kind)
        if res.get("warnings"):
            QMessageBox.warning(self, "Check the result", "\n\n".join(res["warnings"]))

    def _export_plot_pdf(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export plot as PDF", os.path.expanduser("~/profile.pdf"), "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        peem_plots.save_pdf(self.pfig, path)
        msg = f"Saved {path}"
        if getattr(self, "_plot_kind", None) == "profiles" and getattr(self, "_profiles", None):
            csv = path[:-4] + ".csv"
            peem_plots.save_profiles_csv(self._profiles, csv)
            msg += f" and {os.path.basename(csv)}"
        self._log(msg)

    # ------------------------------------------------------------ ROI
    def _current_image(self) -> np.ndarray | None:
        return self.viewer.current_image() if self.viewer.tensor_data is not None else None

    def _start_tool(self, kind: str, cb, color: str, hint: str):
        if self._current_image() is None:
            self._err("Load data first (the image must be shown in the viewer).")
            return
        try:
            self.viewer.start_tool(kind, cb, color)
        except ValueError as exc:
            self._err(str(exc))
            return
        self._log(hint)














    # ------------------------------------------------------------ clean-up edits






















    # ------------------------------------------------------------ view (TensorSpec data viewer)
    _FRAME_KINDS = ("raw", "cleaned", "processed", "stacked")

    def _current_layer(self) -> str:
        view = self.viewer.primary_view()
        return view.combo_layer.currentText() if view is not None else "Intensity"

    def _view_kind(self) -> str:
        layer = self._current_layer()
        if layer in ("", "Intensity"):
            return self.view_combo.currentData() or "raw"
        return self.service.parse_layer(layer)[0]

    def _view_var(self) -> str:
        layer = self._current_layer()
        return "" if layer in ("", "Intensity") else self.service.parse_layer(layer)[1]

    def _frame(self) -> int:
        return int(self.viewer.global_coords.get(0, 0)) if self.viewer.tensor_data is not None else 0

    def _select(self, kind: str, var: str = "data"):
        self._refresh_views(select=kind, var=var)

    def _refresh_views(self, select: str | None = None, var: str = "data"):
        """Feed the viewer: the chosen frame stack as Intensity (frame x y x x) and every map
        variable as a 2-D layer. ``select`` = frame kind or map kind (+ ``var``) to show."""
        if not self._name:
            return
        srcs = [("raw frames", "raw")]
        if not stack_edits.is_empty(self.service.get_edits(self._name)):
            srcs.append(("cleaned frames (before drift)", "cleaned"))
        if self.service.get_drift(self._name, "processed") is not None:
            srcs.append(("drift-corrected frames", "processed"))
            srcs.append(("stacked images (per polarization / energy)", "stacked"))
        cur = self.view_combo.currentData()
        want = select if select in self._FRAME_KINDS else cur
        if want not in [k for _, k in srcs]:
            want = "raw"
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        for label, k in srcs:
            self.view_combo.addItem(label, k)
        self.view_combo.setCurrentIndex(self.view_combo.findData(want))
        self.view_combo.blockSignals(False)
        key = (self._name, want)
        layer_before = self._current_layer()
        if key != getattr(self, "_viewer_key", None) or getattr(self, "_frames_dirty", True):
            frame = self._frame()
            frames = self.service.view_frames(self._name, want)
            n, ny, nx = frames.shape
            md = self.service.xmcd_meta(self._name)
            td = TensorData(value=np.asarray(frames, dtype=np.float32),
                            axes=[np.arange(n, dtype=float), np.arange(ny, dtype=float), np.arange(nx, dtype=float)],
                            labels=["frame", "y", "x"], units=["", "px", "px"], data_type="Experimental PEEM",
                            metadata={"source": self._name, "um_per_px": md.get("um_per_px")})
            self.viewer.load_data(td)
            view = self.viewer.primary_view()
            view.combo_x.setCurrentIndex(2)  # x across, y down, frame on the slider
            view.combo_y.setCurrentIndex(1)
            view.combo_layer.currentIndexChanged.connect(self._on_layer_changed)
            self.viewer.configure_view(cmap="gray", symmetric=False, image_origin=True, aspect_equal=True)
            self.viewer.update_global_coord(0, min(frame, n - 1))
            self._viewer_key, self._frames_dirty = key, False
        if select in self._FRAME_KINDS:
            layer = "Intensity"
        elif select:
            layer = self.service.layer_name(select, var)
        else:
            layer = layer_before
        self.viewer.set_layers(self.service.map_layers(self._name), select=layer)
        if select or layer != layer_before:
            self._on_layer_changed()
        self._update_overlays()

    def _on_layer_changed(self, *_):
        """Signed maps: RWB with symmetric limits; intensity / means: gray (user can change it)."""
        if not self._name or self.viewer.tensor_data is None:
            return
        try:
            hint = self.service.display_hint(self._name, self._view_kind(), self._view_var(), self._frame())
        except ValueError:  # e.g. a layer of a map that is being replaced
            return
        self.viewer.configure_view(**hint)

    def _on_view_changed(self, *_):
        self._frames_dirty = True
        self._refresh_views(select=self.view_combo.currentData())

    def _refresh_view(self):
        self._update_overlays()

    def _update_overlays(self):
        ov = []
        if self._roi:
            ov.append({**self._roi, "color": "yellow"})
        if self._bg_roi:
            ov.append({**self._bg_roi, "color": "cyan"})
        box = self._drift_roi()
        if box:
            ov.append({**box, "color": "magenta"})
        for i, pr in enumerate(getattr(self, "_profiles", None) or []):
            ov.append({"kind": "line", "a": pr["line"][0], "b": pr["line"][1], "color": peem_plots.profile_color(i)})
        if getattr(self, "_line", None):
            ov.append({"kind": "line", "a": self._line[0], "b": self._line[1], "color": "orange"})
        if self._name:
            px = self.service.get_edits(self._name)["pixel_fixes"]
            if px:
                ov.append({"kind": "points", "points": px, "color": "red"})
        if self.viewer.tensor_data is not None:
            self.viewer.set_overlays(ov)

    # ------------------------------------------------------------ image analysis

