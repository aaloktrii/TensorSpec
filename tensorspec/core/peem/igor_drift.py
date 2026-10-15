"""Ports of the SPEEM (BESSY II UE49) Igor drift routines in ``speem_eval_20230427_TPX3.ipf``.

=============================  =========================================================
Igor                           here
=============================  =========================================================
autoImageCorrelation /         ``auto_image_correlation``: fixed window (half the width,
autoCalcStackCorrelation       full height shifted by +1/12 to avoid the MCP burn), mean
                               subtracted, optional filter, Hann window, FFT correlation,
                               whole-pixel peak. ``stack_auto_correlation`` applies it to
                               every frame against a reference frame.
driftCorrection                ``drift_correction_pair``: ONE shift between two images
(after autoCoaddXMCD)          (layer 0 and 1 = the co-added polarizations). Igor's
                               autoCoaddXMCD co-adds without shifting, so there is no
                               per-frame alignment.
marqueeDriftCorrection         ``marquee_drift_correction``: marquee correlation
(+ marqueeCorrelation)         (zero-padded to 2 x power of 2, Hann window), optional x N
                               interpolation for 1/N px steps, aligned frames are added to
                               the reference (running sum), up to 3 passes while the summed
                               |shift| > 5 px, frames named "pre" can be skipped.
marqueeImageRegistration       ``marquee_image_registration``: least-squares intensity
(Igor ImageRegistration)       registration inside the marquee mask: translation, plus ONE
                               of rotation / skew / isotropic scaling (mutually exclusive,
                               as in Igor).
=============================  =========================================================

Axes: Igor image waves are (p, q) = (x, y); arrays here are (y, x). Igor's ShiftStackLayer
does new[p] = old[p + pdrift]; the shifts returned here are in the ``registration.apply_shift``
convention (dy, dx) with new[y] = old[y - dy], i.e. dy = -qdrift, dx = -pdrift.

Not identical to Igor: ImageFilter kernels (find edges = 3x3 Laplacian, sharpen = identity +
Laplacian) and ImageInterpolate (bilinear zoom here) follow the Igor documentation, not its
source; ImageRegistration (Thevenaz-Unser pyramid) is replaced by scipy least squares with the
same parameter choices, and the residual is mean/contrast-normalized inside the mask
(ImageRegistration has no brightness term; CL and CR frames differ in brightness).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, optimize

_FIND_EDGES = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], float)
_SHARPEN = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], float)


def calc_power2(x: float) -> int:
    """Igor calcPower2: largest power of 2 below x (do p2*=2 while p2<x; return p2/2)."""
    p2 = 1
    while True:
        p2 *= 2
        if not p2 < x:
            break
    return max(1, p2 // 2)


def igor_filter(a: np.ndarray, filt: str | None) -> np.ndarray:
    """filterFlag 1 none, 2 'find_edges', 3 'sharpen'; other names go to registration filters."""
    if filt in (None, "none"):
        return a
    if filt == "find_edges":
        return ndimage.convolve(a, _FIND_EDGES, mode="nearest")
    if filt == "sharpen":
        return ndimage.convolve(a, _SHARPEN, mode="nearest")
    from tensorspec.core.peem.registration import _filtered
    return _filtered(a, filt)


def _peak(w0: np.ndarray, w1: np.ndarray) -> tuple[int, int]:
    """Igor: wc0 = conj(FFT w0) * FFT w1; IFFT; integer peak, wrapped. Returns (pdrift, qdrift)."""
    c = np.fft.ifft2(np.conj(np.fft.fft2(w0)) * np.fft.fft2(w1)).real
    mq, mp = c.shape  # rows = q (y), cols = p (x)
    iq, ip = np.unravel_index(int(np.argmax(c)), c.shape)
    qd = iq - mq if iq >= mq / 2 else iq
    pd = ip - mp if ip >= mp / 2 else ip
    return int(pd), int(qd)


def auto_image_correlation(img0: np.ndarray, img1: np.ndarray, filt: str | None = None) -> tuple[int, int]:
    """Igor autoImageCorrelation(img0, img1, filterFlag) -> (pdrift, qdrift) of img1 vs img0."""
    a0 = np.asarray(img0, float)
    a1 = np.asarray(img1, float)
    nq, n_p = a1.shape  # (y, x)
    mp = calc_power2(n_p / 2)
    mq = calc_power2(nq)
    mr = int(round(n_p / 2 - mp / 2))
    mc = int(round(nq / 2 - mq / 2 + nq / 12))  # +nq/12: MCP burn-in
    ip = np.clip(mr + np.arange(mp), 0, n_p - 1)  # Igor clamps out-of-range indices
    iq = np.clip(mc + np.arange(mq), 0, nq - 1)
    w0 = a0[np.ix_(iq, ip)]
    w1 = a1[np.ix_(iq, ip)]
    w0 = igor_filter(w0 - w0.mean(), filt)
    w1 = igor_filter(w1 - w1.mean(), filt)
    hann = np.sin(np.pi * np.arange(mq) / mq)[:, None] * np.sin(np.pi * np.arange(mp) / mp)[None, :]
    return _peak(w0 * hann, w1 * hann)


def _to_shift(pd: float, qd: float) -> tuple[float, float]:
    return -float(qd), -float(pd)


def igor_shift(img: np.ndarray, pd: float, qd: float) -> np.ndarray:
    """Igor ShiftStackLayer: new[p][q] = old[p + pd][q + qd] (indices clamped)."""
    from tensorspec.core.peem.registration import apply_shift
    return apply_shift(img, *_to_shift(pd, qd))


def stack_auto_correlation(frames: np.ndarray, ref_index: int = 0, filt: str | None = None):
    """autoImageCorrelation of every frame against frames[ref_index]; whole-pixel shifts.
    Returns (aligned, shifts (n, 2) as (dy, dx))."""
    st = np.asarray(frames, float)
    shifts = np.zeros((len(st), 2))
    out = np.empty_like(st)
    for i, f in enumerate(st):
        pd, qd = (0, 0) if i == ref_index else auto_image_correlation(st[ref_index], f, filt)
        shifts[i] = _to_shift(pd, qd)
        out[i] = igor_shift(f, pd, qd)
    return out, shifts


def drift_correction_pair(img_a: np.ndarray, img_b: np.ndarray, filt: str | None = "find_edges") -> tuple[float, float]:
    """Igor driftCorrection: one shift of layer 1 (B) relative to layer 0 (A); default filter
    'find edges' as in Igor. Returns (dy, dx) to apply to B."""
    pd, qd = auto_image_correlation(img_a, img_b, filt)
    return _to_shift(pd, qd)


def _marquee_fft(img: np.ndarray, shape: tuple[int, int], filt: str | None) -> np.ndarray:
    """Zero-padded, mean-subtracted, Hann-windowed marquee (Igor marqueeCorrelation) -> FFT."""
    from scipy import fft as sfft
    a = np.asarray(img, float)
    nq, n_p = a.shape
    mq, mp = shape
    mr, mc = int(round(mp / 2 - n_p / 2)), int(round(mq / 2 - nq / 2))
    hann = np.sin(np.pi * np.arange(nq) / nq)[:, None] * np.sin(np.pi * np.arange(n_p) / n_p)[None, :]
    w = np.zeros((mq, mp))
    w[mc:mc + nq, mr:mr + n_p] = (a - a.mean()) * hann
    return sfft.fft2(igor_filter(w, filt), workers=-1)


def _marquee_shape(nq: int, n_p: int) -> tuple[int, int]:
    return max(2 * calc_power2(nq), nq), max(2 * calc_power2(n_p), n_p)


def _peak_fft(f0: np.ndarray, f1: np.ndarray) -> tuple[int, int]:
    from scipy import fft as sfft
    c = sfft.ifft2(np.conj(f0) * f1, workers=-1).real
    mq, mp = c.shape
    iq, ip = np.unravel_index(int(np.argmax(c)), c.shape)
    return int(ip - mp if ip >= mp / 2 else ip), int(iq - mq if iq >= mq / 2 else iq)


def marquee_correlation(img0: np.ndarray, img1: np.ndarray, filt: str | None = None,
                        f0: np.ndarray | None = None) -> tuple[int, int]:
    """Igor marqueeCorrelation: zero-padded to 2*calcPower2(n), mean-subtracted, Hann over the
    marquee, whole-pixel peak. ``f0`` = precomputed FFT of img0 (same padding) to reuse."""
    shape = _marquee_shape(*np.shape(img1))
    if f0 is None:
        f0 = _marquee_fft(img0, shape, filt)
    return _peak_fft(f0, _marquee_fft(img1, shape, filt))


def _zoom(a: np.ndarray, f: float) -> np.ndarray:
    return ndimage.zoom(a, f, order=1, mode="nearest")


def marquee_drift_correction(
    frames: np.ndarray,
    box: tuple[int, int, int, int],
    ref_index: int = 0,
    filt: str | None = None,
    interp_factor: int = 1,
    skip: np.ndarray | None = None,
    max_passes: int = 3,
    tol_px: float = 5.0,
):
    """Igor marqueeDriftCorrection. box = (y0, y1, x0, x1) marquee in pixels.

    interp_factor > 1: the marquee is bilinearly upsampled for the correlation and the frame
    is shifted by the matching fraction (steps of 1/interp_factor px), as Igor's
    'Interpolate during drift correction' (default factor 4; Igor upsamples the whole frame,
    shifts by whole pixels and downsamples, which is a bilinear shift by pp/f). Each aligned marquee is added to
    the reference (running sum; with interpolation the upsampled reference is made once, as in
    Igor). Passes repeat while the summed |shift| > tol_px, at most max_passes.
    Returns (aligned, shifts (n, 2) as (dy, dx), n_passes).
    """
    st = np.asarray(frames, float).copy()
    n = len(st)
    y0, y1, x0, x1 = (int(v) for v in box)
    if y1 - y0 < 4 or x1 - x0 < 4:
        raise ValueError("marquee must be at least 4 x 4 px")
    skip = np.zeros(n, bool) if skip is None else np.asarray(skip, bool)
    f = max(1, int(interp_factor))
    ref = st[ref_index, y0:y1, x0:x1].copy()
    ref_int = _zoom(ref, f) if f > 1 else None
    # with interpolation Igor builds the upsampled reference once, so its FFT is reused
    f0_int = _marquee_fft(ref_int, _marquee_shape(*ref_int.shape), filt) if f > 1 else None
    pp_shift = np.zeros(n)
    qq_shift = np.zeros(n)
    passes = 0
    while True:
        total = 0.0
        for i in range(n):
            if skip[i]:
                continue
            m = st[i, y0:y1, x0:x1]
            if f > 1:
                pp, qq = marquee_correlation(ref_int, _zoom(m, f), filt, f0=f0_int)
                # Igor: bilinear upsample x f, whole-pixel shift, downsample = bilinear shift by pp/f
                st[i] = igor_shift(st[i], pp / f, qq / f)
                pp_shift[i] += pp / f
                qq_shift[i] += qq / f
            else:
                pp, qq = marquee_correlation(ref, m, filt)
                st[i] = igor_shift(st[i], pp, qq)
                pp_shift[i] += pp
                qq_shift[i] += qq
            ref += st[i, y0:y1, x0:x1]  # co-add to the reference
            total += abs(pp) + abs(qq)
        passes += 1
        if not (total > tol_px and passes < max_passes):
            break
    shifts = np.stack([-qq_shift, -pp_shift], axis=1)
    return st, shifts, passes


# --------------------------------------------------------------- marqueeImageRegistration
REGISTRATION_MODES = ("translation", "rotation", "skew", "isoscale")


def _matrix(mode: str, extra: float) -> np.ndarray:
    if mode == "rotation":
        c, s = np.cos(extra), np.sin(extra)
        return np.array([[c, -s], [s, c]])
    if mode == "skew":
        return np.array([[1.0, extra], [0.0, 1.0]])
    if mode == "isoscale":
        return np.eye(2) * (1.0 + extra)
    return np.eye(2)


def register_image(ref: np.ndarray, mov: np.ndarray, mask: np.ndarray, mode: str = "translation",
                   init: tuple[float, float] = (0.0, 0.0)) -> dict:
    """Least-squares registration of ``mov`` onto ``ref`` inside ``mask`` (Igor ImageRegistration
    with /TRNS and, optionally, /ROT or /SKEW or /ISOS). The model maps reference pixel r to
    mov pixel  M (r - c) + c + t ; returns t (translation), the extra parameter and the cost."""
    if mode not in REGISTRATION_MODES:
        raise ValueError(f"mode must be one of {REGISTRATION_MODES}")
    ref = np.asarray(ref, float)
    mov = np.asarray(mov, float)
    yy, xx = np.nonzero(mask)
    if yy.size < 16:
        raise ValueError("registration mask needs at least 16 px")
    c = np.array([(yy.min() + yy.max()) / 2, (xx.min() + xx.max()) / 2])
    r = np.stack([yy - c[0], xx - c[1]])
    target = ref[yy, xx]
    target = (target - target.mean()) / (target.std() or 1.0)

    def resid(v):
        t = v[:2]
        m = _matrix(mode, v[2] if v.size > 2 else 0.0)
        pts = m @ r + (c + t)[:, None]
        w = ndimage.map_coordinates(mov, pts, order=1, mode="nearest")
        # mean / contrast matched inside the mask: the two polarizations differ in brightness
        # (flux, magnetic contrast), which a plain intensity difference would fit as motion
        return (w - w.mean()) / (w.std() or 1.0) - target

    x0 = np.array(list(init) + ([0.0] if mode != "translation" else []), float)
    sol = optimize.least_squares(resid, x0, method="trf", diff_step=1e-3, x_scale="jac", max_nfev=200)
    v = sol.x
    return {"t": (float(v[0]), float(v[1])), "extra": float(v[2]) if v.size > 2 else 0.0,
            "center": (float(c[0]), float(c[1])), "cost": float(sol.cost), "success": bool(sol.success)}


def warp(img: np.ndarray, mode: str, t: tuple[float, float], extra: float, center: tuple[float, float]) -> np.ndarray:
    """Resample ``img`` with the registration model onto the reference grid (edge-clamped)."""
    m = _matrix(mode, extra)
    c = np.asarray(center)
    offset = c + np.asarray(t) - m @ c
    return ndimage.affine_transform(np.asarray(img, float), m, offset=offset, order=1, mode="nearest")


def marquee_image_registration(
    frames: np.ndarray,
    box: tuple[int, int, int, int],
    ref_index: int = 0,
    mode: str = "translation",
    skip: np.ndarray | None = None,
):
    """Igor marqueeImageRegistration: every frame registered to frames[ref_index] inside the
    marquee (refMask). A whole-pixel correlation of the marquee gives the starting translation.
    Returns (aligned, shifts (n, 2) as (dy, dx), params list)."""
    st = np.asarray(frames, float)
    n = len(st)
    y0, y1, x0, x1 = (int(v) for v in box)
    mask = np.zeros(st.shape[1:], bool)
    mask[y0:y1, x0:x1] = True
    skip = np.zeros(n, bool) if skip is None else np.asarray(skip, bool)
    ref = st[ref_index]
    out = st.copy()
    shifts = np.zeros((n, 2))
    params: list[dict] = []
    for i in range(n):
        if i == ref_index or skip[i]:
            params.append({"t": (0.0, 0.0), "extra": 0.0, "skipped": bool(skip[i])})
            continue
        pd, qd = marquee_correlation(ref[y0:y1, x0:x1], st[i, y0:y1, x0:x1])
        res = register_image(ref, st[i], mask, mode, init=(float(qd), float(pd)))
        out[i] = warp(st[i], mode, res["t"], res["extra"], res["center"])
        shifts[i] = (-res["t"][0], -res["t"][1])
        params.append(res)
    return out, shifts, params
