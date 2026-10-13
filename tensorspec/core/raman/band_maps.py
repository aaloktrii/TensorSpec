"""Band-window maps (no Qt): the raw intensity of every spectrum integrated over a wavenumber window
(trapezoid rule, counts·cm⁻¹, no background subtracted), laid onto the map grid. The total-intensity
map (whole spectral range) is the default view of a freshly loaded map: it needs no settings."""
from __future__ import annotations

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


def window_integral(axis, data, lo: float, hi: float) -> np.ndarray:
    """∫ I dω over [lo, hi] for each row of ``data`` (n_spectra, n_axis); NaN if the window holds
    fewer than two axis points."""
    lo, hi = min(lo, hi), max(lo, hi)
    m = (axis >= lo) & (axis <= hi)
    if m.sum() < 2:
        return np.full(data.shape[0], np.nan)
    return _trapz(np.asarray(data, float)[:, m], axis[m], axis=1)


def window_integral_map(mapfile, lo: float, hi: float) -> np.ndarray:
    """window_integral on the map grid (NaN where no spectrum was measured)."""
    v = window_integral(mapfile.axis, mapfile.data, lo, hi)
    g = np.full(mapfile.shape, np.nan)
    ok = mapfile.index_map >= 0
    g[ok] = v[mapfile.index_map[ok]]
    return g


def total_intensity_map(mapfile) -> np.ndarray:
    """Raw counts integrated over each spectrum's whole range, on the map grid."""
    return window_integral_map(mapfile, float(np.min(mapfile.axis)), float(np.max(mapfile.axis)))

