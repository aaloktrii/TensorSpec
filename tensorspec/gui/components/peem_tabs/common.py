"""Small widget helpers shared by the XPEEM panel sections."""
from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox


def _dspin(lo, hi, val, step=0.01, dec=3):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    return s
