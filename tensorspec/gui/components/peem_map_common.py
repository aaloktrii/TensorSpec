"""Small widget helpers shared by the PEEM map tools (gui/components/peem_map_tools.py)."""
from __future__ import annotations

import re

from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLineEdit, QPushButton


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
    btn.setFixedWidth(48)
    edit.returnPressed.connect(on_set)
    btn.clicked.connect(on_set)
    row.addWidget(edit, 1)
    row.addWidget(btn)
    return row, edit
