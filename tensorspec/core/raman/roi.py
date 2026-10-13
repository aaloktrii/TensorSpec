"""Map areas (no Qt): the pixels of a map inside a rectangle, and their spectra integrated into one.

A rectangle is (x0, x1, y0, y1) in the map's current display coordinates (µm, after the
orientation/offset transform). A pixel belongs to it when its centre lies inside."""
from __future__ import annotations

from typing import Tuple

import numpy as np

AREA_MODES = ("sum", "mean")


def normalise_rect(x0, x1, y0, y1) -> Tuple[float, float, float, float]:
    return (float(min(x0, x1)), float(max(x0, x1)), float(min(y0, y1)), float(max(y0, y1)))


def rect_indices(mapfile, rect) -> np.ndarray:
    """Spectrum indices (rows of mapfile.data) whose pixel centre lies inside ``rect``."""
    x0, x1, y0, y1 = normalise_rect(*rect)
    cols = (mapfile.ux >= x0) & (mapfile.ux <= x1)
    rows = (mapfile.uy >= y0) & (mapfile.uy <= y1)
    idx = mapfile.index_map[np.ix_(rows, cols)].ravel()
    return np.sort(idx[idx >= 0])


def area_spectrum(mapfile, rect, mode: str = "sum"):
    """(wavenumber axis, integrated spectrum, spectrum indices) of the pixels inside ``rect``.
    mode 'sum' adds the spectra (total counts); 'mean' averages them."""
    if mode not in AREA_MODES:
        raise ValueError(f"mode must be one of {AREA_MODES}")
    idx = rect_indices(mapfile, rect)
    if idx.size == 0:
        return mapfile.axis, np.full(mapfile.axis.size, np.nan), idx
    block = mapfile.data[idx].astype(float)
    spec = np.nansum(block, axis=0) if mode == "sum" else np.nanmean(block, axis=0)
    return mapfile.axis, spec, idx
