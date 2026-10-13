"""domain_map: three-state magnetic domain segmentation of a dichroism map.

Classification follows the Spatial Domain Analysis (SDA) toolkit's value mode
(``xmcd_toolkit.stats.three_state_field``): m = A / P99(|A|) clipped to [-1, 1],
then +1 if m > t, -1 if m < -t, 0 otherwise. ``normalize="none"`` keeps the raw
asymmetry, so a real two-helicity A is thresholded in physical units.

When ``xmcd_toolkit`` is importable (installed, or its folder named by the
``TENSORSPEC_SDA_PATH`` environment variable) its despeckle, morphometry and
magnetization statistics are used. Otherwise a built-in scipy fallback computes
the same core quantities, and ``backend`` in the result says which ran.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
from scipy import ndimage

_FOUR = ndimage.generate_binary_structure(2, 1)


def load_sda():
    """Return the ``xmcd_toolkit`` module, or None if it cannot be imported."""
    path = os.environ.get("TENSORSPEC_SDA_PATH")
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
    try:
        import xmcd_toolkit  # type: ignore
    except Exception:
        return None
    return xmcd_toolkit


def normalize_asymmetry(a: np.ndarray, roi: np.ndarray, mode: str = "p99",
                        percentile: float = 99.0) -> tuple[np.ndarray, float]:
    if mode == "none":
        return np.asarray(a, dtype=float).copy(), 1.0
    if mode != "p99":
        raise ValueError("normalize must be 'p99' or 'none'")
    vals = np.abs(a[roi & np.isfinite(a)])
    scale = float(np.percentile(vals, percentile)) + 1e-12 if vals.size else 1.0
    return np.clip(a / scale, -1.0, 1.0), scale


def classify(m: np.ndarray, roi: np.ndarray, threshold: float) -> np.ndarray:
    mz = np.zeros(m.shape, dtype=np.int8)
    finite = roi & np.isfinite(m)
    mz[finite & (m > threshold)] = 1
    mz[finite & (m < -threshold)] = -1
    return mz


def despeckle(mz: np.ndarray, min_px: int) -> np.ndarray:
    """Up/down fragments smaller than ``min_px`` (4-connected) revert to 0."""
    if not min_px:
        return mz
    out = mz.copy()
    for state in (1, -1):
        lab, n = ndimage.label(mz == state, structure=_FOUR)
        if n:
            sizes = np.bincount(lab.ravel())
            small = sizes < min_px
            small[0] = False
            out[small[lab]] = 0
    return out


def _perimeter(mask: np.ndarray) -> int:
    """Number of pixel edges between mask and non-mask (4-neighbour)."""
    p = np.pad(mask, 1).astype(np.int8)
    return int(np.abs(np.diff(p, axis=0)).sum() + np.abs(np.diff(p, axis=1)).sum())


def state_morphometry(mz: np.ndarray, roi: np.ndarray, um_per_px: float | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for state, label in ((1, "up"), (-1, "down")):
        mask = (mz == state) & roi
        lab, n = ndimage.label(mask, structure=_FOUR)
        area = int(mask.sum())
        perim = _perimeter(mask)
        width_px = 2.0 * area / perim if perim else float("nan")
        out[f"n_domains_{label}"] = int(n)
        out[f"area_px_{label}"] = area
        out[f"width_px_{label}"] = width_px
        out[f"mean_domain_area_px_{label}"] = area / n if n else float("nan")
        if um_per_px:
            out[f"width_um_{label}"] = width_px * um_per_px
    out["period_px"] = out["width_px_up"] + out["width_px_down"]
    if um_per_px:
        out["period_um"] = out["period_px"] * um_per_px
    wall = _perimeter(mz == 1) + _perimeter(mz == -1)
    roi_area = int(roi.sum())
    out["wall_density_per_px"] = wall / 2.0 / roi_area if roi_area else float("nan")
    return out


def domain_map(
    asym: np.ndarray,
    roi: np.ndarray | None = None,
    threshold: float = 0.05,
    normalize: str = "p99",
    min_px: int = 12,
    flatten_sigma: float | None = None,
    offset: float = 0.0,
    um_per_px: float | None = None,
    use_sda: bool | None = None,
    smooth_sigma: float = 0.0,
) -> dict[str, Any]:
    """Segment ``asym`` into up (+1) / down (-1) / neutral (0) domains with statistics.

    ``use_sda``: None = use the SDA toolkit if importable, True = require it,
    False = built-in only.
    """
    a = np.asarray(asym, dtype=float) - float(offset or 0.0)
    if a.ndim != 2:
        raise ValueError("domain_map needs a 2-D map")
    roi = np.isfinite(a) if roi is None else (np.asarray(roi, dtype=bool) & np.isfinite(a))
    if roi.sum() < 10:
        raise ValueError("ROI has fewer than 10 finite pixels")
    if flatten_sigma:
        filled = np.where(roi, a, np.nanmean(a[roi]))
        a = a - ndimage.gaussian_filter(filled, float(flatten_sigma))

    if smooth_sigma:
        # NaN-aware Gaussian: per-pixel shot noise otherwise fragments domains into 1-2 px specks
        w = ndimage.gaussian_filter(roi.astype(float), float(smooth_sigma))
        num = ndimage.gaussian_filter(np.where(roi, a, 0.0), float(smooth_sigma))
        with np.errstate(invalid="ignore", divide="ignore"):
            a = np.where(roi, num / w, np.nan)

    sda = None if use_sda is False else load_sda()
    if use_sda and sda is None:
        raise ImportError("xmcd_toolkit not importable; install it or set TENSORSPEC_SDA_PATH")

    m, scale = normalize_asymmetry(a, roi, normalize)
    mz = classify(m, roi, threshold)
    extra: dict[str, Any] = {}
    if sda is not None:
        mz = np.asarray(sda.despeckle_mz(mz.astype(int), min_px=min_px), dtype=np.int8)
        try:
            from xmcd_toolkit.morphometry import per_state_morphometry  # type: ignore

            morph = per_state_morphometry(mz.astype(int), um_per_px=um_per_px, min_px=min_px)
            extra["sda_morphometry"] = {k: v for k, v in morph.items() if np.isscalar(v)}
        except Exception as exc:  # keep the built-in numbers if SDA's API changes
            extra["sda_error"] = f"{type(exc).__name__}: {exc}"
        backend = "xmcd_toolkit"
    else:
        mz = despeckle(mz, min_px)
        backend = "builtin"

    vals = a[roi]
    n = int(roi.sum())
    n_up = int(((mz == 1) & roi).sum())
    n_dn = int(((mz == -1) & roi).sum())
    stats: dict[str, Any] = {
        "backend": backend,
        "threshold": float(threshold),
        "normalize": normalize,
        "norm_scale": scale,
        "min_px": int(min_px),
        "smooth_sigma": float(smooth_sigma),
        "n_roi": n,
        "f_up": n_up / n,
        "f_down": n_dn / n,
        "f_neutral": (n - n_up - n_dn) / n,
        "mz_area": (n_up - n_dn) / n,
        "mean_asymmetry": float(np.mean(vals)),
        "mean_abs_asymmetry": float(np.mean(np.abs(vals))),
    }
    stats.update(state_morphometry(mz, roi, um_per_px))
    stats.update(extra)
    return {"mz": mz, "m": m, "roi": roi, "stats": stats}
