"""Small widget helpers shared by the PEEM map tools (gui/components/peem_map_tools.py)."""
from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton


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


# Lighter shades of the other suites' main-button colours (QE / Wannier90: Load W90 green, Run red,
# Fetch teal; ARPES: Push orange), with dark text. Kept to the few main steps so the panel stays clean.
ACTION = {
    "load": ("#a9dcab", "#1d3d1f", "#d6e6d6"),
    "map": ("#f2a9ae", "#5a1418", "#ead3d5"),
    "domains": ("#f7c896", "#5a3208", "#eadccd"),  # ARPES "Push" orange
    "handover": ("#a3dbe5", "#0c3f48", "#d2e5e8"),
}


def action_style(button: QPushButton, kind: str) -> QPushButton:
    """Colour a step's main button like the other suites (bold, light shade, greyed when disabled)."""
    bg, fg, off = ACTION[kind]
    button.setStyleSheet(f"QPushButton {{ background-color: {bg}; color: {fg}; font-weight: bold; padding: 5px; }}"
                         f"QPushButton:disabled {{ background-color: {off}; color: #999999; }}")  # as the DFT panel
    return button


def bold_titles(group) -> None:
    """Bold section title, as the QE / Wannier90 groups."""
    group.setStyleSheet("QGroupBox { font-weight: bold; }")


def _subtitle(text: str) -> QLabel:
    """A small heading inside a section (e.g. 'Flatten along the line', 'Profiles')."""
    lab = QLabel(text)
    lab.setStyleSheet("font-weight: bold; padding-top: 6px;")
    return lab


def tidy_forms(widget) -> None:
    """The TensorSpec form look of the DFT panel on every platform (macOS otherwise keeps fields at their
    size hint, right-aligns labels and clips combo texts): labels left, fields grow to the full width.
    Spacing stays the style's default, as in the other suites."""
    for form in widget.findChildren(QFormLayout):
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)
