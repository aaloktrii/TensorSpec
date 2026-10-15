"""Image analysis of SPEEM maps and frames: ports of the speem_eval_20230427_TPX3.ipf tools (no Qt).

====================================  ====================================================
Igor                                  here
====================================  ====================================================
LineProfile / getGaussProfile         ``line_profile``: npixels = round(length) + 1 points
(ImageLineProfile width=)             from A to B, bilinear, averaged over -hw..hw px
                                      perpendicular to the line
displayLines / stackLines             ``stack_line_profiles``: the profile of every frame ->
                                      2-D image (frame or energy x position)
getGaussProfile (CurveFit Gauss)      ``gauss_fit``: y0 + A exp(-((x - x0) / w)^2); FWHM =
                                      2 sqrt(ln 2) w (feature width)
getDiffGaussProfile                   ``diff_gauss_fit``: Gauss fit of the derivative (edge /
                                      domain-wall width, lateral resolution)
calcFrameHistogram                    ``histogram``: 255 bins, in the ROI; Igor's fixed 0.9-1.1
                                      range (ratio images) or the 0.5-99.5 % range
imageFFT                              ``image_fft``: largest power-of-2 window, mean removed,
                                      Hann, |FFT| (centred); ``radial_profile`` adds the
                                      azimuthal average and the dominant period
powerSpectrum                         ``power_spectrum``: 1-D, sin^2 window, zero-padded to
                                      2 calcPower2(n), |FFT|, 0-0.5 1/px
FFTFilter(cutoff)                     ``fft_filter``: gain 1 below the cutoff index, then
                                      1 / (i + 1 - cutoff); along x as Igor, or radial
Flatten / FlattenSequence             ``flatten_line``: linear fit of the line profile (or of
                                      the two end averages, 'points only') removed as a
                                      plane tilted along the line; offset optional
backgroundremoval                     ``background_poly``: polynomial fit of every row and
                                      every column (cell width), background = their mean
                                      minus its mean
bkgRemove / removeBackGround          ``remove_background_roi``: 2-D polynomial (order 3) fitted
(ImageRemoveBackground /P=3)          in the background ROI, subtracted, ROI mean restored
====================================  ====================================================

Coordinates are (y, x) array indices; Igor (p, q) = (x, y). NaN pixels (outside the field of
view of a map) are ignored where possible.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, optimize

from tensorspec.core.peem.igor_drift import calc_power2


# ------------------------------------------------------------------ line profiles
def _line_coords(a: tuple[float, float], b: tuple[float, float], halfwidth: int):
    (y1, x1), (y2, x2) = a, b
    length = float(np.hypot(y2 - y1, x2 - x1))
    n = int(round(length)) + 1
    t = np.linspace(0.0, 1.0, max(n, 2))
    ys, xs = y1 + (y2 - y1) * t, x1 + (x2 - x1) * t
    ny, nx = (-(x2 - x1) / length, (y2 - y1) / length) if length else (0.0, 0.0)  # unit normal (y, x)
    offs = np.arange(-int(halfwidth), int(halfwidth) + 1)
    return ys[None, :] + offs[:, None] * ny, xs[None, :] + offs[:, None] * nx, length, len(t)


def line_profile(img: np.ndarray, a: tuple[float, float], b: tuple[float, float], halfwidth: int = 0,
                 um_per_px: float | None = None) -> dict:
    """Profile from a = (y, x) to b, averaged over 2*halfwidth+1 parallel lines (NaN ignored)."""
    im = np.asarray(img, float)
    yy, xx, length, n = _line_coords(a, b, halfwidth)
    vals = ndimage.map_coordinates(np.nan_to_num(im), [yy, xx], order=1, mode="nearest")
    ok = ndimage.map_coordinates(np.isfinite(im).astype(float), [yy, xx], order=1, mode="nearest") > 0.999
    vals = np.where(ok, vals, np.nan)
    with np.errstate(invalid="ignore"):
        prof = np.nanmean(vals, axis=0)
    dist = np.linspace(0, length, n)
    return {"distance": dist * (um_per_px or 1.0), "unit": "µm" if um_per_px else "px",
            "values": prof, "length_px": length, "halfwidth": int(halfwidth)}


def stack_line_profiles(frames: np.ndarray, a, b, halfwidth: int = 0, um_per_px: float | None = None) -> dict:
    """displayLines / stackLines: profile of every frame -> (n_frames, n_points)."""
    rows = [line_profile(f, a, b, halfwidth, um_per_px) for f in np.asarray(frames)]
    return {"distance": rows[0]["distance"], "unit": rows[0]["unit"], "image": np.stack([r["values"] for r in rows])}


def _gauss(x, y0, a, x0, w):
    return y0 + a * np.exp(-((x - x0) / w) ** 2)


def gauss_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """Igor CurveFit Gauss: y0 + A exp(-((x - x0) / width)^2). FWHM = 2 sqrt(ln2) width."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 5:
        raise ValueError("profile has fewer than 5 finite points")
    y0 = float(np.median(np.r_[y[:3], y[-3:]]))
    i = int(np.argmax(np.abs(y - y0)))
    p0 = [y0, y[i] - y0, x[i], max((x[-1] - x[0]) / 10, 1e-9)]
    p, cov = optimize.curve_fit(_gauss, x, y, p0=p0, maxfev=10000)
    err = np.sqrt(np.clip(np.diag(cov), 0, None)) if np.all(np.isfinite(cov)) else np.full(4, np.nan)
    k = 2 * np.sqrt(np.log(2))
    return {"y0": p[0], "A": p[1], "x0": p[2], "width": abs(p[3]), "fwhm": k * abs(p[3]),
            "fwhm_err": k * err[3], "x0_err": err[2], "fit": _gauss(x, *p), "x": x}


def diff_gauss_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """getDiffGaussProfile: Gauss fit of d(profile)/dx -> edge / wall width (FWHM)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    d = np.gradient(y, x)
    out = gauss_fit(x, d)
    out["derivative"] = d
    return out


# ------------------------------------------------------------------ histogram
def histogram(img: np.ndarray, mask: np.ndarray | None = None, bins: int = 255,
              value_range: tuple[float, float] | None = None) -> dict:
    """calcFrameHistogram: 255 bins in the ROI (whole image if none). value_range None = 0.5-99.5 %;
    Igor uses a fixed (0.9, 1.1) for ratio images."""
    v = np.asarray(img, float)
    if mask is not None:
        v = v[np.asarray(mask, bool)]
    v = v[np.isfinite(v)]
    if not v.size:
        raise ValueError("no finite pixels in the ROI")
    lo, hi = value_range or tuple(np.percentile(v, [0.5, 99.5]))
    counts, edges = np.histogram(v, bins=int(bins), range=(lo, hi))
    centers = 0.5 * (edges[1:] + edges[:-1])
    return {"centers": centers, "counts": counts, "range": (float(lo), float(hi)), "n": int(v.size),
            "mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)),
            "outside_range": int(((v < lo) | (v > hi)).sum())}


# ------------------------------------------------------------------ Fourier tools
def image_fft(img: np.ndarray) -> dict:
    """imageFFT: centred calcPower2 window, mean removed, Hann, |FFT| (zero frequency centred).
    Frequencies in 1/px."""
    a = np.asarray(img, float)
    ny, nx = a.shape
    my, mx = calc_power2(ny), calc_power2(nx)
    r0, c0 = int(round(ny / 2 - my / 2)), int(round(nx / 2 - mx / 2))
    w = a[r0:r0 + my, c0:c0 + mx]
    fill = np.nanmean(w)
    w = np.where(np.isfinite(w), w, fill) - fill
    hann = np.sin(np.pi * np.arange(my) / my)[:, None] * np.sin(np.pi * np.arange(mx) / mx)[None, :]
    mag = np.abs(np.fft.fftshift(np.fft.fft2(w * hann)))
    return {"magnitude": mag, "fy": np.fft.fftshift(np.fft.fftfreq(my)), "fx": np.fft.fftshift(np.fft.fftfreq(mx)),
            "window": (r0, r0 + my, c0, c0 + mx)}


def radial_profile(fft: dict, um_per_px: float | None = None, f_min: float = 0.005) -> dict:
    """Azimuthal average of |FFT| vs spatial frequency; the peak above f_min gives the dominant
    period (e.g. the domain period). Addition to Igor's imageFFT."""
    mag = fft["magnitude"]
    fy, fx = np.meshgrid(fft["fy"], fft["fx"], indexing="ij")
    fr = np.hypot(fy, fx)
    step = min(abs(fft["fy"][1] - fft["fy"][0]), abs(fft["fx"][1] - fft["fx"][0]))
    idx = np.round(fr / step).astype(int)
    prof = ndimage.mean(mag ** 2, labels=idx, index=np.arange(idx.max() + 1))
    f = np.arange(idx.max() + 1) * step
    ok = (f >= f_min) & (f <= 0.5)
    k = int(np.argmax(np.where(ok, prof, -np.inf)))
    period_px = 1.0 / f[k] if f[k] > 0 else np.nan
    return {"f": f, "power": prof, "peak_f": float(f[k]), "period_px": float(period_px),
            "period_um": float(period_px * um_per_px) if um_per_px else None}


def power_spectrum(y: np.ndarray) -> dict:
    """powerSpectrum: mean removed, sin^2 window, zero-padded to 2 calcPower2(n), |FFT|, 0-0.5 1/px."""
    v = np.asarray(y, float)
    v = np.where(np.isfinite(v), v, np.nanmean(v))
    n = v.size
    m = calc_power2(n) * 2
    w = np.zeros(m)
    w[:n] = (v - v.mean()) * np.sin(np.pi * np.arange(n) / n) ** 2
    mag = np.abs(np.fft.rfft(w))
    f = np.fft.rfftfreq(m)
    k = int(np.argmax(np.where(f > 1.0 / n, mag, -np.inf)))
    return {"f": f, "magnitude": mag, "peak_f": float(f[k]), "period_px": float(1 / f[k]) if f[k] else np.nan}


def fft_filter(img: np.ndarray, cutoff: int, mode: str = "x") -> np.ndarray:
    """FFTFilter: gain 1 for frequency index i < cutoff, then 1 / (i + 1 - cutoff). mode 'x'
    filters along x only (Igor: index of the half spectrum along p), 'radial' uses the radial index."""
    a = np.asarray(img, float)
    nanmask = ~np.isfinite(a)
    fill = np.nanmean(a)
    a0 = np.where(nanmask, fill, a)
    ny, nx = a0.shape
    spec = np.fft.rfft2(a0)
    ix = np.arange(nx // 2 + 1)[None, :]
    if mode == "radial":
        iy = np.abs(np.fft.fftfreq(ny) * ny)[:, None]
        i = np.hypot(iy, ix)
    elif mode == "x":
        i = np.broadcast_to(ix, spec.shape).astype(float)
    else:
        raise ValueError("mode must be 'x' or 'radial'")
    gain = np.where(i < cutoff, 1.0, 1.0 / np.maximum(i + 1 - cutoff, 1.0))
    out = np.fft.irfft2(spec * gain, s=a0.shape)
    out[nanmask] = np.nan
    return out


# ------------------------------------------------------------------ flattening / background
def flatten_line(img: np.ndarray, a, b, halfwidth: int = 30, subtract_offset: bool = False,
                 points_only: bool = False) -> tuple[np.ndarray, dict]:
    """Flatten: fit a line to the profile A->B (or, 'points only', to the means of
    (halfwidth x halfwidth) boxes at A and B) and subtract the plane tilted along A->B, centred
    on the image centre; with subtract_offset the fitted level is removed too."""
    im = np.asarray(img, float)
    prof = line_profile(im, a, b, halfwidth)
    v = prof["values"]
    s = np.arange(v.size, dtype=float)  # one point per pixel along the line
    if points_only:
        h = max(1, halfwidth // 2)

        def box_mean(p):
            y, x = int(round(p[0])), int(round(p[1]))
            return float(np.nanmean(im[max(y - h, 0):y + h + 1, max(x - h, 0):x + h + 1]))
        first, last = box_mean(a), box_mean(b)
        v = first * (v.size - s) / v.size + last * s / v.size
    ok = np.isfinite(v)
    slope, c0 = np.polyfit(s[ok], v[ok], 1)
    (y1, x1), (y2, x2) = a, b
    length = float(np.hypot(y2 - y1, x2 - x1)) or 1.0
    uy, ux = (y2 - y1) / length, (x2 - x1) / length
    ny, nx = im.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    # level at the image centre (Igor: offset moved to np/2, nq/2)
    c_center = c0 + slope * ((ny / 2 - y1) * uy + (nx / 2 - x1) * ux)
    plane = slope * ((yy - ny / 2) * uy + (xx - nx / 2) * ux) + (c_center if subtract_offset else 0.0)
    return im - plane, {"slope_per_px": float(slope), "offset_center": float(c_center)}


def background_poly(img: np.ndarray, order: int = 2, width: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """backgroundremoval: polynomial (order) fit of every row and of every column, each averaged
    over a cell of `width` lines; background = mean of the two, minus its own mean. Returns
    (image - background, background)."""
    im = np.asarray(img, float)
    ny, nx = im.shape
    fill = np.where(np.isfinite(im), im, np.nanmean(im))
    sm_rows = ndimage.uniform_filter1d(fill, size=max(1, int(width)), axis=0, mode="nearest")
    sm_cols = ndimage.uniform_filter1d(fill, size=max(1, int(width)), axis=1, mode="nearest")
    x = np.arange(nx)
    y = np.arange(ny)
    bh = np.empty_like(fill)
    bv = np.empty_like(fill)
    vx = np.polynomial.polynomial.polyvander(x / nx, order)
    vy = np.polynomial.polynomial.polyvander(y / ny, order)
    coef_h = np.linalg.lstsq(vx, sm_rows.T, rcond=None)[0]  # one fit per row
    bh[:] = (vx @ coef_h).T
    coef_v = np.linalg.lstsq(vy, sm_cols, rcond=None)[0]  # one fit per column
    bv[:] = vy @ coef_v
    bg = (bh + bv) / 2
    bg -= bg.mean()
    return im - bg, bg


def remove_background_roi(img: np.ndarray, bg_mask: np.ndarray, order: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """bkgRemove (ImageRemoveBackground /P=3): 2-D polynomial of total degree `order` fitted to the
    pixels in the background ROI, subtracted; the ROI mean is added back (subtractOffset(-V_avg))."""
    im = np.asarray(img, float)
    m = np.asarray(bg_mask, bool) & np.isfinite(im)
    if m.sum() < (order + 1) * (order + 2):
        raise ValueError("background ROI has too few pixels for this order")
    ny, nx = im.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    yn, xn = yy / ny - 0.5, xx / nx - 0.5
    terms = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]
    a = np.stack([(xn[m] ** i) * (yn[m] ** j) for i, j in terms], axis=1)
    coef = np.linalg.lstsq(a, im[m], rcond=None)[0]
    bg = sum(c * (xn ** i) * (yn ** j) for c, (i, j) in zip(coef, terms))
    return im - bg + im[m].mean(), bg
