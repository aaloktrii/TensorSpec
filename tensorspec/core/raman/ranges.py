"""Typed wavenumber ranges and peak labels (no Qt): the text a user enters for fit bands and for the
stacked figure, parsed the same way by any frontend.

Moved from gui/components/raman_tabs/stack_panel.py."""
from __future__ import annotations

import re
from typing import List, Tuple

_NUM = r"(\d+(?:\.\d+)?)"


def parse_windows(text: str) -> List[Tuple[float, float]]:
    """'1450-1650, 2550-2800' -> [(1450, 1650), (2550, 2800)]."""
    out = []
    for lo, hi in re.findall(_NUM + r"\s*[-–:]\s*" + _NUM, text):
        lo, hi = float(lo), float(hi)
        if hi > lo:
            out.append((lo, hi))
    return out


def parse_band_labels(text: str) -> List[Tuple[str, float]]:
    """'G 1583, 2D 2690' -> [('G', 1583.0), ('2D', 2690.0)]."""
    out = []
    for piece in re.split(r"[,;]", text):
        m = re.match(r"\s*(.+?)\s*@?\s*" + _NUM + r"\s*$", piece)
        if m:
            out.append((m.group(1), float(m.group(2))))
    return out
