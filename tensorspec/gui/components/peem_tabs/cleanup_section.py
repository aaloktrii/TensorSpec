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


class CleanupSection:
    def _build_cleanup(self, cl):
        """Controls of this section, added to the control column ``cl``."""
        # ---------------------------------------------------------------- 2. Clean-up
        clean = QGroupBox("2. Stack clean-up (optional: tick to show)")
        clean.setCheckable(True)
        clean.setChecked(False)
        clean.setToolTip("Exclude / include frames, normalize to or subtract a frame.\n"
                         "Raw data are never changed; saved edits apply even when this is collapsed.")
        clean_body = QWidget()
        cf = QFormLayout(clean_body)
        cf.setContentsMargins(0, 0, 0, 0)
        self.frames_edit = QLineEdit()
        self.frames_edit.setPlaceholderText("frames, e.g. 5, 12-14 (blank = current frame)")
        cf.addRow("Frames:", self.frames_edit)
        r = QHBoxLayout()
        for label, fn in (("Exclude", lambda: self._edit_exclude(True)), ("Include", lambda: self._edit_exclude(False))):
            b = QPushButton(label)
            b.clicked.connect(fn)
            r.addWidget(b)
        cf.addRow(r)
        r = QHBoxLayout()
        b = QPushButton("Normalize to frame")
        b.setToolTip("Every frame / that frame x its RMS (Frames field or current frame)")
        b.clicked.connect(lambda: self._edit_refop("normalize_to"))
        r.addWidget(b)
        b = QPushButton("Subtract frame")
        b.setToolTip("Every frame - that frame")
        b.clicked.connect(lambda: self._edit_refop("subtract"))
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

    def _edit_refop(self, op: str):
        if self._require():
            fr = self._edit_frames(single=True)
            if fr is not None:
                self._edit_call(self.service.set_reference_frame_op, op, fr[0])

    def _edit_reset(self):
        self._edit_call(self.service.reset_edits)
