"""ROI statistics on dichroism maps (port of ROIAverage.ipf).

Masks are boolean ``(y, x)`` with True inside the ROI (Igor's M_ROIMask uses the
opposite sign). NaN pixels are ignored everywhere. The image is never modified.
"""
from __future__ import annotations

import numpy as np


def _roi_values(img: np.ndarray, mask: np.ndarray | None, bg: float = 0.0) -> np.ndarray:
    a = np.asarray(img, dtype=float)
    if mask is None:
        vals = a.ravel()
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != a.shape:
            raise ValueError(f"mask {mask.shape} does not match image {a.shape}")
        vals = a[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("no finite pixels inside the ROI")
    return vals - float(bg or 0.0)


def background_level(img: np.ndarray, bg_mask: np.ndarray) -> dict:
    """Mean / sd / N inside a background ROI (Igor 'Select Background ROI')."""
    vals = _roi_values(img, bg_mask)
    return {"bg": float(vals.mean()), "bg_sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "bg_n": int(vals.size)}


def subtract_background(img: np.ndarray, bg_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """New image = img - mean(background ROI)."""
    info = background_level(img, bg_mask)
    return np.asarray(img, dtype=float) - info["bg"], info


def roi_average(img: np.ndarray, mask: np.ndarray | None, bg: float = 0.0, band: float = 0.0) -> dict:
    """Mean of the real values in the ROI after subtracting ``bg``.

    Pixels with |value| < ``band`` (noise / domain-wall band) are excluded.
    """
    vals = _roi_values(img, mask, bg)
    n_all = int(vals.size)
    if band and band > 0:
        vals = vals[np.abs(vals) >= band]
    if vals.size == 0:
        raise ValueError("every ROI pixel fell inside the gray band")
    return {
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "n_used": int(vals.size),
        "n_roi": n_all,
        "bg": float(bg or 0.0),
        "band": float(band or 0.0),
    }


def updown_fractions(img: np.ndarray, mask: np.ndarray | None, threshold: float | None = None,
                     band: float = 0.0, bg: float = 0.0) -> dict:
    """Three-state area fractions f+, f-, f0 inside the ROI.

    up: v > t + band, down: v < t - band, wall: |v - t| <= band, where
    t = ``threshold`` or the ROI mean when None (Igor default).
    """
    vals = _roi_values(img, mask, bg)
    thr_source = "user"
    if threshold is None or not np.isfinite(threshold):
        threshold = float(vals.mean())
        thr_source = "roi_mean"
    band = float(band or 0.0)
    n = vals.size
    n_up = int((vals > threshold + band).sum())
    n_dn = int((vals < threshold - band).sum())
    n_wall = n - n_up - n_dn
    f_up, f_dn, f_wall = n_up / n, n_dn / n, n_wall / n
    return {
        "f_up": f_up,
        "f_down": f_dn,
        "f_wall": f_wall,
        "ratio": f_up / f_dn if n_dn else float("nan"),
        "mz_area": (n_up - n_dn) / n,
        "threshold": float(threshold),
        "threshold_source": thr_source,
        "band": band,
        "bg": float(bg or 0.0),
        "n": int(n),
        "n_up": n_up,
        "n_down": n_dn,
        "n_wall": n_wall,
    }
