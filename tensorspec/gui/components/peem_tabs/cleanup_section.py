"""PEEM XMCD-tab section: 2. Stack clean-up before drift (non-destructive edits).

Mixin of ``PeemXmcdPanel`` (gui/components/peem_xmcd_panel.py); it uses the panel's shared state
(service, _name, ROIs, viewer, _log / _err / _run). Moved verbatim from the former ue49_panel.py."""
from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from tensorspec.core.peem import stack_edits

_DIVERGING = {"pcd", "pld", "pmld", "pmnld"}

from tensorspec.gui.components.peem_tabs.common import _dspin
from tensorspec.plotting import peem_plots


class CleanupSection:
    def _build_cleanup(self, cl):
        """Controls of this section, added to the control column ``cl``."""
        # ---------------------------------------------------------------- 2. Clean-up
        clean = QGroupBox("2. Stack clean-up (optional: tick to show)")
        clean.setCheckable(True)
        clean.setChecked(False)
        clean.setToolTip("Exclude / defocused frames, top-up, normalize / subtract a frame, disable pixels.\n"
                         "Raw data are never changed; saved edits apply even when this is collapsed.")
        clean_body = QWidget()
        cf = QFormLayout(clean_body)
        cf.setContentsMargins(0, 0, 0, 0)
        self.frames_edit = QLineEdit()
        self.frames_edit.setPlaceholderText("frames, e.g. 5, 12-14 (blank = current frame)")
        cf.addRow("Frames:", self.frames_edit)
        r = QHBoxLayout()
        for label, fn in (("Exclude", lambda: self._edit_exclude(True)), ("Include", lambda: self._edit_exclude(False)),
                          ("Include all", self._edit_include_all)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            r.addWidget(b)
        cf.addRow(r)
        r = QHBoxLayout()
        b = QPushButton("Find defocused")
        b.setToolTip("Igor RemoveDefocussedLayers: std of every frame in the box (or central half)")
        b.clicked.connect(self._edit_find_defocused)
        r.addWidget(b)
        self.defocus_rel = _dspin(0, 1, 0.8, 0.05, 2)
        self.defocus_rel.setPrefix("< ")
        self.defocus_rel.setSuffix(" × median std")
        r.addWidget(self.defocus_rel)
        b = QPushButton("Exclude below")
        b.setToolTip("Exclude the frames whose std in the box is below this fraction of the median")
        b.clicked.connect(self._edit_exclude_defocused)
        r.addWidget(b)
        cf.addRow(r)
        r = QHBoxLayout()
        self.topup_factor = _dspin(0.5, 2, 1.05, 0.01, 3)
        self.topup_factor.setPrefix("× ")
        r.addWidget(QLabel("Top-up (injection frames):"))
        r.addWidget(self.topup_factor)
        b = QPushButton("Apply")
        b.setToolTip("Igor TopUpCorrection: frames during injection x factor. Frames from the ring-current\n"
                     "jumps if the files have it (UE49 CurrentTable, ALS BEAM_CURRENT), else the Frames field")
        b.clicked.connect(self._edit_topup)
        r.addWidget(b)
        cf.addRow(r)
        r = QHBoxLayout()
        b = QPushButton("Normalize to frame")
        b.setToolTip("Igor NormalizeStackToThisLayer: every frame / that frame x its RMS (Frames field or current frame)")
        b.clicked.connect(lambda: self._edit_refop("normalize_to"))
        r.addWidget(b)
        b = QPushButton("Subtract frame")
        b.setToolTip("Igor SubtractThisLayerFromStack: every frame - that frame")
        b.clicked.connect(lambda: self._edit_refop("subtract"))
        r.addWidget(b)
        b = QPushButton("Undo")
        b.clicked.connect(self._edit_refop_clear)
        r.addWidget(b)
        cf.addRow(r)
        r = QHBoxLayout()
        self.btn_pixels = QPushButton("Disable pixels (click)")
        self.btn_pixels.setCheckable(True)
        self.btn_pixels.setToolTip("Igor DisablePixel: each clicked pixel -> mean of its diagonal neighbours at ±2 px")
        self.btn_pixels.toggled.connect(self._pixel_mode)
        r.addWidget(self.btn_pixels)
        b = QPushButton("Clear pixels")
        b.clicked.connect(self._edit_clear_pixels)
        r.addWidget(b)
        cf.addRow(r)
        r = QHBoxLayout()
        self.edits_label = QLabel("no clean-up edits")
        self.edits_label.setWordWrap(True)
        r.addWidget(self.edits_label, 1)
        b = QPushButton("Reset all")
        b.clicked.connect(self._edit_reset)
        r.addWidget(b)
        cf.addRow(r)
        QVBoxLayout(clean).addWidget(clean_body)
        clean_body.setVisible(False)
        clean.toggled.connect(clean_body.setVisible)  # collapsed by default; a checkable box would
        clean.toggled.connect(lambda _on: clean_body.setEnabled(True))  # otherwise disable its contents
        cl.addWidget(clean)

    def _edit_frames(self, single: bool = False) -> list[int] | None:
        """Frames from the Frames field (raw indices), or the current raw frame when blank."""
        n = self.service.xmcd_meta(self._name)["n_frames"]
        txt = self.frames_edit.text().strip()
        try:
            if txt:
                fr = stack_edits.parse_frames(txt, n)
            elif self._view_kind() == "raw":
                fr = [self._frame()]
            else:
                self._err("Type frame numbers, or show 'raw frames' and pick one with the slider.")
                return None
        except ValueError as exc:
            self._err(str(exc))
            return None
        if single and len(fr) != 1:
            self._err("Give exactly one frame.")
            return None
        return fr

    def _after_edit(self, res: dict):
        self._frames_dirty = True
        self.edits_label.setText(f"{res['summary']} · {res['frames_used']}/{res['frames_total']} frames used"
                                 + (f" · {res['note']}" if res.get("note") else ""))
        self._log(json.dumps({k: v for k, v in res.items() if k != "edits"}, default=str))
        self._refresh_views()

    def _edit_call(self, fn, *args, **kw):
        name = self._require()
        if not name:
            return
        try:
            self._after_edit(fn(name, *args, **kw))
        except ValueError as exc:
            self._err(str(exc))

    def _edit_exclude(self, exclude: bool):
        if self._require():
            fr = self._edit_frames()
            if fr is not None:
                self._edit_call(self.service.set_excluded, fr, exclude)

    def _edit_include_all(self):
        self._edit_call(self.service.include_all)

    def _edit_find_defocused(self):
        name = self._require()
        if not name:
            return
        r = self.service.frame_sharpness(name, self._reg_roi)
        sd, med = r["std"], r["median"]
        thr = self.defocus_rel.value() * med
        y0, y1, x0, x1 = r["box_y0y1x0x1"]
        ax = self._plot(f"focus check · std in box x {x0}-{x1}, y {y0}-{y1}")
        low = peem_plots.frame_sharpness(ax, sd, thr, self.defocus_rel.value())
        self.pfig.tight_layout()
        self.pcanvas.draw_idle()
        self._log(f"Defocused (below threshold): {low.tolist()}")

    def _edit_exclude_defocused(self):
        self._edit_call(self.service.exclude_defocused, self.defocus_rel.value(), self._reg_roi)

    def _edit_topup(self):
        name = self._require()
        if not name:
            return
        auto = self.service.injection_frames(name)
        fr = None if auto else self._edit_frames()
        if not auto and fr is None:
            return
        self._edit_call(self.service.set_topup, self.topup_factor.value(), fr)

    def _edit_refop(self, op: str):
        if self._require():
            fr = self._edit_frames(single=True)
            if fr is not None:
                self._edit_call(self.service.set_reference_frame_op, op, fr[0])

    def _edit_refop_clear(self):
        name = self._require()
        if name:
            self.service.set_reference_frame_op(name, "normalize_to", None)
            self._edit_call(self.service.set_reference_frame_op, "subtract", None)

    def _pixel_mode(self, on: bool):
        if on and not self._require():
            self.btn_pixels.setChecked(False)
            return
        view = self.viewer.primary_view()
        if view is not None:
            view.cancel_tool()
        if on and view is not None:
            cid = view.canvas.mpl_connect("button_press_event", lambda ev, v=view: self._on_pixel_click(v, ev))
            view._tool = ("cid", cid)  # pauses the crosshair while picking pixels
            self._log("Click pixels to disable; click the button again to stop.")

    def _on_pixel_click(self, view, ev):
        if ev.inaxes is not view.ax_main or ev.xdata is None:
            return
        y = int(round(view._axis_to_index(ev.ydata, view.y_idx)))
        x = int(round(view._axis_to_index(ev.xdata, view.x_idx)))
        self._edit_call(self.service.disable_pixels, [[y, x]], True)
        self._log(f"Disabled pixel (y {y}, x {x})")
        if self.btn_pixels.isChecked():  # the redraw replaced the view: keep picking
            self._pixel_mode(True)

    def _edit_clear_pixels(self):
        name = self._require()
        if name:
            self._edit_call(self.service.disable_pixels, self.service.get_edits(name)["pixel_fixes"], False)

    def _edit_reset(self):
        self._edit_call(self.service.reset_edits)
