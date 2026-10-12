"""2D-band analysis (no Qt): single Voigt + linear baseline per spectrum, despike, smoothing,
low-frequency asymmetry metrics, FWHM uncertainty; band windows; per-chunk fitting for
process pools.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (2D-band analysis)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.special import voigt_profile

#: np.trapezoid is NumPy >= 2.0; np.trapz is the 1.x spelling.
_trapz = getattr(np, "trapezoid", None) or np.trapz

try:
    from lmfit.models import LinearModel, VoigtModel
    HAVE_LMFIT = True
except Exception:                                    # pragma: no cover
    HAVE_LMFIT = False

# ---------------------------------------------------------------------------
# 2D-band analysis
# ---------------------------------------------------------------------------

def voigt_fwhm(sigma: float, gamma: float) -> float:
    """Olivero & Longbothum pseudo-analytic Voigt FWHM (accurate to ~0.02%).

    f_V = 0.5346 f_L + sqrt(0.2166 f_L^2 + f_G^2), with f_L = 2*gamma and
    f_G = 2*sigma*sqrt(2 ln2).
    """
    fl = 2.0 * abs(gamma)
    fg = 2.0 * abs(sigma) * np.sqrt(2.0 * np.log(2.0))
    return 0.5346 * fl + np.sqrt(0.2166 * fl * fl + fg * fg)


def _voigt_plus_line(x, amplitude, center, sigma, gamma, slope, intercept):
    return amplitude * voigt_profile(x - center, sigma, gamma) + slope * x + intercept


def despike(y: np.ndarray, sigma: float = 6.0, width: int = 5):
    """Replace cosmic-ray spikes with the local median.

    Returns (cleaned, n_removed). A spike is a point sitting more than
    ``sigma`` robust deviations above a running median. The 2D band spans
    hundreds of channels at these dispersions, so a 5-point median cannot
    touch it, while a cosmic ray is a handful of channels wide and would
    otherwise be fitted as an implausibly narrow, very high-SNR "band".
    """
    if y.size < width * 3:
        return y, 0
    med = median_filter(y, size=width, mode="nearest")
    resid = y - med
    mad = float(np.median(np.abs(resid - np.median(resid))))
    if mad <= 0:
        return y, 0
    bad = resid > sigma * 1.4826 * mad
    if not bad.any():
        return y, 0
    out = y.copy()
    out[bad] = med[bad]
    return out, int(bad.sum())


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Savitzky-Golay smoothing, used only for shape metrics - never for fitting."""
    w = int(window) | 1
    if w < 5 or y.size <= w:
        return y
    try:
        return savgol_filter(y, w, 3)
    except Exception:
        return y


def _initial_guess(x: np.ndarray, y: np.ndarray, smooth_window: int = 9) -> dict:
    ys = _smooth(y, smooth_window)
    n_edge = max(3, len(x) // 10)
    xe = np.concatenate([x[:n_edge], x[-n_edge:]])
    ye = np.concatenate([ys[:n_edge], ys[-n_edge:]])
    slope, intercept = np.polyfit(xe, ye, 1)
    yc = ys - (slope * x + intercept)

    i = int(np.nanargmax(yc))
    center = float(x[i])
    height = float(yc[i])
    if not np.isfinite(height) or height <= 0:
        height = float(np.nanmax(np.abs(yc))) or 1.0

    above = np.where(yc >= height / 2.0)[0]
    fw = abs(float(x[above[-1]] - x[above[0]])) if above.size >= 2 else 30.0
    fw = float(np.clip(fw, 2.0, max(4.0, x[-1] - x[0])))

    sigma = fw / 2.3548 * 0.7
    gamma = fw / 2.0 * 0.5
    peak = voigt_profile(0.0, sigma, gamma)
    amplitude = height / peak if peak > 0 else height * fw
    return dict(amplitude=amplitude, center=center, sigma=sigma, gamma=gamma,
                slope=float(slope), intercept=float(intercept), height=height)


def _asymmetry_metrics(x, y, fit, fwhm, cfg):
    """Low-frequency asymmetry of the 2D band; returns (hwhm, area, residual).

    All three are anchored at the maximum of the smoothed band, NOT at the
    fitted Voigt centre. That distinction matters: a single Voigt fitted to a
    band carrying a low-frequency shoulder drags its centroid into the
    shoulder, which re-balances the two lobes and hides the very asymmetry
    being measured. Anchoring at the maximum keeps the reference where the
    dominant component sits.

    hwhm     - (HWHM_low - HWHM_high) / (HWHM_low + HWHM_high), the classic
               peak-asymmetry ratio. Least sensitive to baseline and to the
               far wings.
    area     - normalised difference of the two lobe integrals out to
               asym_span x FWHM either side of the maximum.
    residual - what the single Voigt fails to describe on the low-frequency
               side, as a fraction of the fitted peak area.

    All are positive when there is excess weight on the low-frequency side
    (~2680 cm-1 for a band near 2700 cm-1), the ABC signature, and all are
    dimensionless.
    """
    nan3 = (np.nan, np.nan, np.nan)
    ys = _smooth(y, cfg["smooth_window"])
    base = float(np.percentile(ys, 5))
    yb = ys - base
    i = int(np.argmax(yb))
    peak = float(yb[i])
    if peak <= 0 or x.size < 10:
        return nan3
    xc = float(x[i])

    # --- HWHM ratio, by linear interpolation of the half-maximum crossings ---
    hw = np.nan
    half = peak / 2.0
    j = i
    while j > 0 and yb[j] > half:
        j -= 1
    lo_x = np.interp(half, [yb[j], yb[j + 1]], [x[j], x[j + 1]]) if j < i else np.nan
    j = i
    while j < x.size - 1 and yb[j] > half:
        j += 1
    hi_x = np.interp(half, [yb[j], yb[j - 1]], [x[j], x[j - 1]]) if j > i else np.nan
    left_w, right_w = xc - lo_x, hi_x - xc
    if np.isfinite(left_w) and np.isfinite(right_w) and (left_w + right_w) > 0:
        hw = float((left_w - right_w) / (left_w + right_w))

    # --- lobe areas about the maximum ---
    span = float(np.clip(cfg["asym_span"] * fwhm, 5.0, (x[-1] - x[0]) / 2.0))
    lmask = (x >= xc - span) & (x < xc)
    rmask = (x > xc) & (x <= xc + span)
    ar = np.nan
    if lmask.sum() >= 3 and rmask.sum() >= 3:
        yc = np.clip(yb, 0.0, None)
        a_l = float(_trapz(yc[lmask], x[lmask]))
        a_r = float(_trapz(yc[rmask], x[rmask]))
        if a_l + a_r > 0:
            ar = float((a_l - a_r) / (a_l + a_r))

    # --- unmodelled low-frequency shoulder ---
    rs = np.nan
    if fit is not None and lmask.sum() >= 3:
        resid = y - fit
        peak_area = float(_trapz(np.clip(fit - np.min(fit), 0, None), x))
        if peak_area > 0:
            shoulder = float(_trapz(np.clip(resid[lmask], 0, None), x[lmask]))
            rs = float(shoulder / peak_area)

    return hw, ar, rs


def fwhm_uncertainty(sigma: float, gamma: float, cov: np.ndarray) -> float:
    """1-sigma error on the Voigt FWHM, propagated from the fit covariance.

    Worth having, not decoration: the ABA/ABC boundary is only ~20 cm-1 from
    either threshold, so an FWHM carrying a +/-10 cm-1 error cannot support a
    stacking call no matter how confident the colour map looks.
    """
    if cov is None or not np.all(np.isfinite(cov)) or cov.shape != (2, 2):
        return np.nan
    h = 1e-6 * max(abs(sigma), abs(gamma), 1.0)
    d_s = (voigt_fwhm(sigma + h, gamma) - voigt_fwhm(sigma - h, gamma)) / (2 * h)
    d_g = (voigt_fwhm(sigma, gamma + h) - voigt_fwhm(sigma, gamma - h)) / (2 * h)
    var = (d_s * d_s * cov[0, 0] + d_g * d_g * cov[1, 1]
           + 2.0 * d_s * d_g * cov[0, 1])
    return float(np.sqrt(var)) if var > 0 else np.nan


def _empty_result() -> dict:
    return dict(ok=False, fwhm=np.nan, center=np.nan, height=np.nan, snr=np.nan,
                amplitude=np.nan, sigma=np.nan, gamma=np.nan,
                slope=np.nan, intercept=np.nan, asym_hwhm=np.nan,
                asym_area=np.nan, asym_residual=np.nan, spikes=0,
                fwhm_err=np.nan, note="", why="")


def fit_band(x: np.ndarray, y: np.ndarray, cfg: dict) -> dict:
    """Fit one Voigt + linear baseline to an already-cropped band region.

    Returns a dict carrying ``ok``, the FWHM, the fit parameters (so the curve
    can be rebuilt without re-fitting) and all three asymmetry metrics.
    """
    out = _empty_result()
    good = np.isfinite(y)
    if good.sum() < 10:
        out["note"] = "too few valid points"
        out["why"] = "too few valid points in the window"
        return out
    x = x[good].astype(float)
    y = y[good].astype(float)

    if np.all(y == 0):
        out["note"] = "empty spectrum (all zero)"
        out["why"] = "empty spectrum (all zero)"
        return out
    if np.ptp(y) == 0:
        out["note"] = "flat spectrum (no signal)"
        out["why"] = "flat spectrum (no signal)"
        return out

    if cfg.get("despike", True):
        y, n_spikes = despike(y, cfg.get("despike_sigma", 6.0))
        out["spikes"] = n_spikes

    g = _initial_guess(x, y, cfg["smooth_window"])
    span = float(x[-1] - x[0])
    params = None
    cov = None

    if cfg.get("engine", "scipy") == "lmfit" and HAVE_LMFIT:
        try:
            model = VoigtModel(prefix="v_") + LinearModel(prefix="b_")
            p = model.make_params(v_amplitude=g["amplitude"], v_center=g["center"],
                                  v_sigma=g["sigma"], b_slope=g["slope"],
                                  b_intercept=g["intercept"])
            p["v_center"].set(min=x[0], max=x[-1])
            p["v_sigma"].set(min=1e-3, max=span)
            p["v_amplitude"].set(min=0)
            # VoigtModel ties gamma to sigma by default; free it for real bands.
            p["v_gamma"].set(value=g["gamma"], vary=bool(cfg["free_gamma"]),
                             expr="" if cfg["free_gamma"] else None,
                             min=0, max=span)
            res = model.fit(y, p, x=x, nan_policy="omit")
            if res.covar is not None:
                names = list(res.var_names)
                if "v_sigma" in names and "v_gamma" in names:
                    i, j = names.index("v_sigma"), names.index("v_gamma")
                    cov = np.array([[res.covar[i, i], res.covar[i, j]],
                                    [res.covar[j, i], res.covar[j, j]]])
            params = dict(amplitude=res.params["v_amplitude"].value,
                          center=res.params["v_center"].value,
                          sigma=res.params["v_sigma"].value,
                          gamma=res.params["v_gamma"].value,
                          slope=res.params["b_slope"].value,
                          intercept=res.params["b_intercept"].value)
        except Exception:
            params = None

    if params is None:
        try:
            p0 = [g["amplitude"], g["center"], g["sigma"], g["gamma"],
                  g["slope"], g["intercept"]]
            lower = [0.0, x[0], 1e-3, 0.0, -np.inf, -np.inf]
            upper = [np.inf, x[-1], span, span, np.inf, np.inf]
            p0 = [float(np.clip(v, lo, hi)) for v, lo, hi in zip(p0, lower, upper)]
            if not cfg["free_gamma"]:
                # Tie gamma to sigma by fitting a reduced model.
                def _tied(xx, amp, cen, sig, sl, ic):
                    return _voigt_plus_line(xx, amp, cen, sig, sig, sl, ic)
                popt, pcov = curve_fit(_tied, x, y, p0=[p0[0], p0[1], p0[2], p0[4], p0[5]],
                                    bounds=([0, x[0], 1e-3, -np.inf, -np.inf],
                                            [np.inf, x[-1], span, np.inf, np.inf]),
                                    maxfev=20000)
                popt = [popt[0], popt[1], popt[2], popt[2], popt[3], popt[4]]
                cov = np.array([[pcov[2, 2], pcov[2, 2]],
                                [pcov[2, 2], pcov[2, 2]]])
            else:
                popt, pcov = curve_fit(_voigt_plus_line, x, y, p0=p0,
                                       bounds=(lower, upper), maxfev=20000)
                cov = pcov[np.ix_([2, 3], [2, 3])]
            params = dict(zip(["amplitude", "center", "sigma", "gamma",
                               "slope", "intercept"], map(float, popt)))
        except Exception as exc:
            out["note"] = "fit failed (%s)" % type(exc).__name__
            out["why"] = "the fit did not converge"
            return out

    fwhm = voigt_fwhm(params["sigma"], params["gamma"])
    fit = _voigt_plus_line(x, **params)
    baseline = params["slope"] * x + params["intercept"]
    height = float(np.max(fit - baseline))
    # Noise from the fit residual, not the band edges: a broad fit to pure
    # noise then scores an SNR near 1 instead of sneaking past the filter.
    noise = float(np.std(y - fit))
    snr = height / noise if noise > 0 else np.inf

    out.update(params)
    out["fwhm"] = float(fwhm)
    out["center"] = float(params["center"])
    out["height"] = height
    out["snr"] = float(snr)
    out["fwhm_err"] = fwhm_uncertainty(params["sigma"], params["gamma"], cov)

    if not np.isfinite(fwhm) or fwhm <= 1.0:
        out["note"] = "degenerate FWHM"
        out["why"] = "degenerate FWHM"
        return out
    if fwhm < cfg.get("min_fwhm", 5.0):
        # Narrower than the instrument can resolve, so it is an artefact
        # (usually a surviving cosmic ray), not a phonon band.
        out["note"] = ("FWHM %.1f below the %.1f cm-1 resolution limit"
                       % (fwhm, cfg.get("min_fwhm", 5.0)))
        out["why"] = ("narrower than the %.1f cm⁻¹ resolution limit — a spike, "
                      "not a band" % cfg.get("min_fwhm", 5.0))
        return out
    if fwhm > cfg["max_fwhm"]:
        out["note"] = "FWHM %.0f above max %.0f" % (fwhm, cfg["max_fwhm"])
        out["why"] = "wider than the %.0f cm⁻¹ limit" % cfg["max_fwhm"]
        return out
    if snr < cfg["min_snr"]:
        out["note"] = "SNR %.1f below %.1f" % (snr, cfg["min_snr"])
        out["why"] = "SNR below %.1f — no measurable band" % cfg["min_snr"]
        return out
    if not (x[0] < params["center"] < x[-1]):
        out["note"] = "centre pinned at the window edge"
        out["why"] = "fitted centre pinned at the window edge"
        return out

    hw, ar, rs = _asymmetry_metrics(x, y, fit, fwhm, cfg)
    out["asym_hwhm"], out["asym_area"], out["asym_residual"] = hw, ar, rs
    out["ok"] = True
    return out


def band_mask(wn: np.ndarray, cfg: dict) -> np.ndarray:
    return (wn >= cfg["band_min"]) & (wn <= cfg["band_max"])


def fit_2d_band(wn: np.ndarray, inten: np.ndarray, cfg: dict) -> dict:
    """Crop to the 2D-band window, then fit. Convenience wrapper."""
    m = band_mask(wn, cfg)
    if m.sum() < 10:
        out = _empty_result()
        out["note"] = "fewer than 10 points in %.0f-%.0f cm-1" % (
            cfg["band_min"], cfg["band_max"])
        out["why"] = "fewer than 10 points inside the band window"
        return out
    return fit_band(wn[m], inten[m], cfg)


def _fit_chunk(args):
    """Module-level so it can be shipped to a worker process."""
    x, block, cfg = args
    return [fit_band(x, block[i], cfg) for i in range(block.shape[0])]


def rebuild_fit(x: np.ndarray, res: dict) -> np.ndarray:
    return _voigt_plus_line(x, res["amplitude"], res["center"], res["sigma"],
                            res["gamma"], res["slope"], res["intercept"])


def asym_of(res: dict, metric: str) -> float:
    return res.get({"hwhm": "asym_hwhm", "area": "asym_area",
                    "residual": "asym_residual"}.get(metric, "asym_hwhm"), np.nan)
