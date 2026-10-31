"""Small widget helpers shared by the PEEM map tools (gui/components/peem_map_tools.py)."""
from __future__ import annotations

import os
import re

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import (QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QWidget)


def _dspin(lo, hi, val, step=0.01, dec=3):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    return s


def _numbers(text: str, n: int) -> list[float]:
    """n numbers typed as '10, 20, 300, 400' (commas, spaces or semicolons)."""
    vals = [v for v in re.split(r"[,;\s]+", text.strip()) if v]
    try:
        out = [float(v) for v in vals]
    except ValueError:
        raise ValueError(f"type {n} numbers, e.g. 10, 20, 300, 400") from None
    if len(out) != n:
        raise ValueError(f"type {n} numbers (got {len(out)})")
    return out


def _coords_row(placeholder: str, on_set) -> tuple[QHBoxLayout, QLineEdit]:
    """A coordinate field + 'Set' button (Return also sets)."""
    row = QHBoxLayout()
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    btn = QPushButton("Set")
    edit.returnPressed.connect(on_set)
    btn.clicked.connect(on_set)
    row.addWidget(edit, 1)
    row.addWidget(btn)
    return row, edit


# TensorSpec's main-button colours (see the other suites): white bold text on the suite colour; disabled = the
# same hue darker with grey text (QE generator pattern). Kept to the few main steps so the panel stays clean.
ACTION = {
    "load": ("#5cb85c", "#295329"),      # load / push (DFT Push bands)
    "map": ("#2b5c8f", "#132940"),       # primary compute (Workspace Browser Launch Data Viewer, DFT Calculate)
    "domains": ("#e67e22", "#68390f"),   # hand-over to analysis (DFT Prepare TB for ARPES, ARPES push)
    "handover": ("#17a2b8", "#0c5460"),  # data transfer (QE Fetch Wannier)
}


def action_style(button: QPushButton, kind: str) -> QPushButton:
    """Colour a step's main button as the other TensorSpec suites do (white bold text, 8 px padding)."""
    bg, off = ACTION[kind]
    button.setStyleSheet(f"QPushButton {{ background-color: {bg}; color: white; font-weight: bold; padding: 8px; }}"
                         f"QPushButton:disabled {{ background-color: {off}; color: #888888; }}")
    return button


def bold_titles(group) -> None:
    """Bold section title, as the QE / Wannier90 groups."""
    group.setStyleSheet("QGroupBox { font-weight: bold; }")


def _subtitle(text: str) -> QLabel:
    """A small heading inside a section (e.g. 'Flatten along the line', 'Profiles'), as the Workspace Browser's."""
    return QLabel(f"<b>{text}</b>")


# --- Export location: one folder for every PEEM save dialog (maps, profiles, Domains PDFs / CSV / .nc) ---
class _ExportDirHub(QObject):
    changed = Signal(str)


_hub: _ExportDirHub | None = None


def _export_hub() -> _ExportDirHub:
    global _hub
    if _hub is None:
        _hub = _ExportDirHub()
    return _hub


def _peem_settings() -> QSettings:
    return QSettings("TensorSpec", "PEEM")


def chosen_export_dir() -> str:
    """The folder picked under 'Export to' ('' = none picked, or it no longer exists)."""
    d = str(_peem_settings().value("export_dir", "") or "")
    return d if d and os.path.isdir(d) else ""


def export_dir() -> str:
    """Where save dialogs open: the 'Export to' folder, else next to the last loaded data, else home."""
    last = str(_peem_settings().value("last_dir", "") or "")
    return chosen_export_dir() or (last if last and os.path.isdir(last) else os.path.expanduser("~"))


def set_export_dir(folder: str) -> None:
    """Set (or clear with '') the export folder; every 'Export to' row shows the change."""
    folder = os.path.expanduser(folder.strip()) if folder else ""
    _peem_settings().setValue("export_dir", folder)
    _export_hub().changed.emit(folder)


def export_path(filename: str) -> str:
    """Default path for a save dialog: the file name in the export folder."""
    return os.path.join(export_dir(), filename)


def export_dir_row(parent: QWidget | None = None) -> QWidget:
    """'Export to' field + Browse…: type or pick a folder (empty = next to the loaded data)."""
    w = QWidget(parent)
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    edit = QLineEdit(chosen_export_dir())
    edit.setPlaceholderText("next to the loaded data")
    edit.setToolTip("Every PEEM export / save dialog opens in this folder. Leave empty to use the folder\n"
                    "the data was loaded from.")
    btn = QPushButton("Browse…")

    def typed():
        text = edit.text().strip()
        if text and not os.path.isdir(os.path.expanduser(text)):
            edit.setText(chosen_export_dir())  # not a folder: keep the old one
            return
        if text != chosen_export_dir():
            set_export_dir(text)

    def browse():
        folder = QFileDialog.getExistingDirectory(w, "Export to folder", export_dir())
        if folder:
            set_export_dir(folder)

    edit.editingFinished.connect(typed)
    btn.clicked.connect(browse)
    _export_hub().changed.connect(edit.setText)  # bound slot: dropped when the row is closed
    row.addWidget(edit, 1)
    row.addWidget(btn)
    w.edit, w.browse = edit, btn  # tests
    return w
