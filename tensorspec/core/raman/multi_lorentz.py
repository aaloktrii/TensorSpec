"""Multi-Lorentzian fitting for the Spectra tab (no Qt): fit_lorentzians, suggest_peak_count.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Multi-Lorentzian fitting)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from tensorspec.core.raman.voigt_2d import _smooth, despike

# ---------------------------------------------------------------------------
# Multi-Lorentzian fitting (Spectra tab)
# ---------------------------------------------------------------------------

def _lorentz_sum(x, *p):
    """Sum of n Lorentzians (height, centre, FWHM) plus a linear baseline."""
    n = (len(p) - 2) // 3
    y = p[-2] * (x - x.mean()) + p[-1]
    for k in range(n):
        h, c, w = p[3 * k:3 * k + 3]
        y = y + h * (0.5 * w) ** 2 / ((x - c) ** 2 + (0.5 * w) ** 2)
    return y


def fit_lorentzians(x: np.ndarray, y: np.ndarray, n: int = 1) -> dict:
    """Fit n Lorentzians plus a linear baseline over the given range.

    Returns ok, per-peak centre / FWHM / height / area with 1-sigma errors,
    the FWHM of the summed band (what a single width means when a band has
    several components), R², and the fitted curve. Cosmic rays are removed
    first. Note: a Lorentzian describes the phonon line itself; at low
    spectral resolution the instrument adds Gaussian broadening, so widths
    from a 600 gr/mm spectrum are upper bounds.
    """
    out = {"ok": False, "n": n, "peaks": [], "band_fwhm": np.nan, "r2": np.nan,
           "curve": None, "note": ""}
    good = np.isfinite(y)
    x, y = x[good].astype(float), y[good].astype(float)
    if x.size < 3 * n + 6:
        out["note"] = "too few points in range"
        return out
    y, _ = despike(y)
    edge = max(3, x.size // 12)
    xe = np.r_[x[:edge], x[-edge:]]
    ye = np.r_[y[:edge], y[-edge:]]
    sl, ic = np.polyfit(xe - x.mean(), ye, 1)
    base = sl * (x - x.mean()) + ic
    r = _smooth(y - base, 9)
    span = x[-1] - x[0]
    step = float(np.median(np.abs(np.diff(x))))
    # Starting guesses: the n most prominent peaks, padded if there are fewer.
    pk, pr = find_peaks(r, prominence=max(1e-9, 0.05 * np.nanmax(r)))
    order = np.argsort(-pr["prominences"])[:n] if pk.size else []
    cens = sorted(float(x[pk[i]]) for i in order)
    top = float(x[int(np.argmax(r))])
    while len(cens) < n:
        cens.append(top + (len(cens) - (n - 1) / 2.0) * 0.15 * span / max(n, 1))
    cens = sorted(cens)
    hmax = float(np.nanmax(r))
    above = x[r >= 0.5 * hmax]
    w0 = max(3 * step, (above.max() - above.min()) / n if above.size > 1 else 0.1 * span)
    p0, lo, hi = [], [], []
    for c in cens:
        hc = max(float(np.interp(c, x, r)), 0.2 * hmax)
        p0 += [hc, c, w0]
        lo += [0.0, x[0], max(step, 0.5)]
        hi += [np.inf, x[-1], span]
    p0 += [sl, ic]
    lo += [-np.inf, -np.inf]
    hi += [np.inf, np.inf]
    p0 = [float(np.clip(v, a, b)) for v, a, b in zip(p0, lo, hi)]
    try:
        popt, pcov = curve_fit(_lorentz_sum, x, y, p0=p0, bounds=(lo, hi), maxfev=40000)
    except Exception as exc:
        out["note"] = "fit failed (%s)" % type(exc).__name__
        return out
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None)) if np.all(np.isfinite(pcov)) \
        else np.full(len(popt), np.nan)
    fit = _lorentz_sum(x, *popt)
    ss = float(np.sum((y - fit) ** 2))
    st = float(np.sum((y - y.mean()) ** 2)) or 1.0
    peaks = []
    for k in range(n):
        h, c, w = popt[3 * k:3 * k + 3]
        eh, ec, ew = perr[3 * k:3 * k + 3]
        peaks.append({"center": float(c), "fwhm": float(w), "height": float(h),
                      "area": float(np.pi * h * w / 2.0), "err_center": float(ec),
                      "err_fwhm": float(ew)})
    peaks.sort(key=lambda d: d["center"])
    # FWHM of the summed band, baseline removed, on a fine grid.
    xf = np.linspace(x[0], x[-1], 4000)
    band = _lorentz_sum(xf, *(list(popt[:-2]) + [0.0, 0.0]))
    if band.max() > 0:
        above = xf[band >= 0.5 * band.max()]
        out["band_fwhm"] = float(above.max() - above.min()) if above.size > 1 else np.nan
    npts, k = x.size, 3 * n + 2
    # Bayesian information criterion: rewards fit quality, penalises extra
    # peaks. Adding peaks always raises R²; BIC only falls if they earn it.
    out["bic"] = float(npts * np.log(max(ss, 1e-12) / npts) + k * np.log(npts))
    out.update(ok=True, peaks=peaks, r2=1.0 - ss / st, curve=(x, fit),
               baseline=(float(popt[-2]), float(popt[-1]), float(x.mean())),
               components=[(p["height"], p["center"], p["fwhm"]) for p in peaks])
    return out

def suggest_peak_count(x: np.ndarray, y: np.ndarray, resolution: float,
                       nmax: int = 4):
    """Smallest peak count the data justify.

    Two rules, because either alone misleads: a fit whose components are
    narrower than the spectral resolution is fitting noise or shape mismatch
    and is rejected; among the rest, the fewest peaks within 10 BIC units of
    the best (strong evidence) wins. A Lorentzian is not quite the shape of an
    instrument-broadened line, so BIC on its own keeps asking for more peaks.
    """
    fits = {n: fit_lorentzians(x, y, n) for n in range(1, nmax + 1)}
    valid = {n: f for n, f in fits.items() if f["ok"] and
             all(p["fwhm"] >= resolution for p in f["peaks"])}
    if not valid:
        return 1, fits
    best = min(f["bic"] for f in valid.values())
    return min(n for n, f in valid.items() if f["bic"] <= best + 10.0), fits
