"""FFT cross-correlation registration (port of Igor autoImageCorrelation /
marqueeDriftCorrection / smartCoadd), extended to sub-pixel precision.

Convention: ``estimate_shift(ref, mov)`` returns (dy, dx) such that
``apply_shift(mov, dy, dx)`` overlays ``ref``.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _sine_window(ny: int, nx: int) -> np.ndarray:
    wy = np.sin(np.pi * (np.arange(ny) + 0.5) / ny)
    wx = np.sin(np.pi * (np.arange(nx) + 0.5) / nx)
    return wy[:, None] * wx[None, :]


def _crop(img: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    if box is None:
        return img
    y0, y1, x0, x1 = (int(v) for v in box)
    if y1 - y0 < 4 or x1 - x0 < 4:
        raise ValueError("registration box must be at least 4x4 pixels")
    return img[y0:y1, x0:x1]


def _prep(img: np.ndarray, filt: str | None) -> np.ndarray:
    a = np.nan_to_num(np.asarray(img, dtype=float))
    if filt == "edges":
        a = np.hypot(ndimage.sobel(a, 0), ndimage.sobel(a, 1))
    elif filt == "smooth":  # sigma 1 px: hides a detector's fixed pixel pattern (it would pin the shift at 0);
        a = ndimage.gaussian_filter(a, 1.0)  # wider blurs the peak and biases the sub-pixel fit
    elif filt == "smooth_edges":  # single noisy frames: smooth (sigma 2 px) before the gradient
        a = ndimage.gaussian_filter(a, 2.0)
        a = np.hypot(ndimage.sobel(a, 0), ndimage.sobel(a, 1))
    elif filt == "highpass":
        a = a - ndimage.gaussian_filter(a, 8)
    elif filt in ("find_edges", "sharpen"):  # Igor ImageFilter kernels
        from tensorspec.core.peem.igor_drift import igor_filter
        a = igor_filter(a, filt)
    elif filt not in (None, "none"):
        raise ValueError(f"unknown filter {filt!r}")
    a = a - a.mean()
    return a * _sine_window(*a.shape)


def _parabolic(cm: float, c0: float, cp: float) -> float:
    denom = cm - 2.0 * c0 + cp
    return 0.0 if denom == 0 else 0.5 * (cm - cp) / denom


def estimate_shift(
    ref: np.ndarray,
    mov: np.ndarray,
    box: tuple[int, int, int, int] | None = None,
    filt: str | None = None,
    subpixel: bool = True,
    max_shift: float | None = None,
) -> tuple[float, float]:
    """Translation of ``mov`` relative to ``ref`` from the windowed cross-correlation peak.

    ``max_shift`` (px) restricts the peak search to |dy|, |dx| <= max_shift. Between
    opposite polarizations the reversed domain pattern produces a false peak half a
    domain period away; bounding the search to plausible drift excludes it.
    """
    a = _prep(_crop(np.asarray(ref, dtype=float), box), filt)
    b = _prep(_crop(np.asarray(mov, dtype=float), box), filt)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    corr = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    ny, nx = corr.shape
    if max_shift is not None:
        wy = np.minimum(np.arange(ny), ny - np.arange(ny))
        wx = np.minimum(np.arange(nx), nx - np.arange(nx))
        allowed = (wy[:, None] <= max_shift) & (wx[None, :] <= max_shift)
        floor = float(corr[allowed].min())
        corr = np.where(allowed, corr, floor)
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    if max_shift is not None and (min(py, ny - py) >= max_shift or min(px, nx - px) >= max_shift):
        subpixel = False  # peak on the search boundary: no reliable parabola
    dy, dx = float(py), float(px)
    if subpixel:
        dy += _parabolic(corr[(py - 1) % ny, px], corr[py, px], corr[(py + 1) % ny, px])
        dx += _parabolic(corr[py, (px - 1) % nx], corr[py, px], corr[py, (px + 1) % nx])
    if dy > ny / 2:
        dy -= ny
    if dx > nx / 2:
        dx -= nx
    return dy, dx


def apply_shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Bilinear shift with edge replication (Igor clamps out-of-range indices)."""
    a = np.asarray(img, dtype=float)
    if dy == 0 and dx == 0:
        return a.copy()
    return ndimage.shift(a, (dy, dx), order=1, mode="nearest")


def register_stack(
    stack: np.ndarray,
    ref_index: int = 0,
    box: tuple[int, int, int, int] | None = None,
    filt: str | None = None,
    subpixel: bool = True,
    reference: np.ndarray | None = None,
    max_shift: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Align every frame of ``(n, y, x)`` to one reference. Returns (aligned, shifts (n, 2))."""
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError("stack must be (n, y, x)")
    if reference is None:
        if not 0 <= ref_index < stack.shape[0]:
            raise ValueError(f"ref_index {ref_index} out of range")
        reference = stack[ref_index]
    shifts = np.zeros((stack.shape[0], 2))
    out = np.empty_like(stack)
    for i, frame in enumerate(stack):
        dy, dx = estimate_shift(reference, frame, box=box, filt=filt, subpixel=subpixel, max_shift=max_shift)
        shifts[i] = dy, dx
        out[i] = apply_shift(frame, dy, dx)
    return out, shifts


def smart_coadd(
    stack: np.ndarray,
    register: bool = True,
    box: tuple[int, int, int, int] | None = None,
    filt: str | None = None,
    subpixel: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Drift-corrected mean of a stack; the reference is the running mean.

    Returns (mean image, shifts (n, 2)). Igor's smartCoadd uses integer shifts.
    """
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3 or stack.shape[0] == 0:
        raise ValueError("stack must be (n, y, x) with n >= 1")
    acc = stack[0].copy()
    shifts = np.zeros((stack.shape[0], 2))
    for k in range(1, stack.shape[0]):
        frame = stack[k]
        if register:
            dy, dx = estimate_shift(acc / k, frame, box=box, filt=filt, subpixel=subpixel)
            shifts[k] = dy, dx
            frame = apply_shift(frame, dy, dx)
        acc += frame
    return acc / stack.shape[0], shifts


# ------------------------------------------------------------------ ROI template tracking
def _filtered(img: np.ndarray, filt: str | None) -> np.ndarray:
    """Same filters as the cross-correlation path, without the apodization window."""
    a = np.nan_to_num(np.asarray(img, dtype=float))
    if filt == "edges":
        return np.hypot(ndimage.sobel(a, 0), ndimage.sobel(a, 1))
    if filt == "smooth":
        return ndimage.gaussian_filter(a, 1.0)
    if filt == "smooth_edges":
        a = ndimage.gaussian_filter(a, 2.0)
        return np.hypot(ndimage.sobel(a, 0), ndimage.sobel(a, 1))
    if filt == "highpass":
        return a - ndimage.gaussian_filter(a, 8)
    if filt in ("find_edges", "sharpen"):
        from tensorspec.core.peem.igor_drift import igor_filter
        return igor_filter(a, filt)
    if filt not in (None, "none"):
        raise ValueError(f"unknown filter {filt!r}")
    return a


def ncc_map(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation of ``template`` at every valid position in ``image``
    (FFT numerator, window sums from integral images; Lewis 1995)."""
    from scipy.signal import fftconvolve

    img = np.asarray(image, dtype=float)
    t = np.asarray(template, dtype=float)
    th, tw = t.shape
    if img.shape[0] < th or img.shape[1] < tw:
        raise ValueError("search window smaller than the template")
    t = t - t.mean()
    tnorm = float(np.sqrt((t * t).sum()))
    if tnorm <= np.finfo(float).eps:
        raise ValueError("ROI template has no contrast (zero variance)")
    num = fftconvolve(img, t[::-1, ::-1], mode="valid")

    def box_sum(a):
        c = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return c[th:, tw:] - c[:-th, tw:] - c[th:, :-tw] + c[:-th, :-tw]

    n = th * tw
    s1, s2 = box_sum(img), box_sum(img * img)
    var = np.maximum(s2 - s1 * s1 / n, 0.0)
    den = np.sqrt(var) * tnorm
    out = np.zeros_like(num)
    ok = den > np.finfo(float).eps * max(1.0, float(den.max()))
    out[ok] = num[ok] / den[ok]
    return out


def track_template(
    stack: np.ndarray,
    ref_index: int = 0,
    box: tuple[int, int, int, int] | None = None,
    search_radius: int = 20,
    filt: str | None = None,
    subpixel: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """ROI method: the reference frame's ``box`` is the template; in every frame it is found by
    NCC within ``search_radius`` px of where the previous frame put it, so slow drift far larger
    than the radius is followed. Returns (shifts (n, 2) as (dy, dx) for ``apply_shift``, peak NCC)."""
    st = np.asarray(stack, dtype=float)
    n, ny, nx = st.shape
    if not 0 <= ref_index < n:
        raise ValueError(f"ref_index {ref_index} out of range")
    y0, y1, x0, x1 = (int(v) for v in (box or (ny // 4, 3 * ny // 4, nx // 4, 3 * nx // 4)))
    if y1 - y0 < 8 or x1 - x0 < 8:
        raise ValueError("ROI box must be at least 8 x 8 px")
    r = int(search_radius)
    if r < 1:
        raise ValueError("search radius must be >= 1 px")
    template = _filtered(st[ref_index], filt)[y0:y1, x0:x1]
    shifts = np.zeros((n, 2))
    score = np.ones(n)
    order = list(range(ref_index + 1, n)) + list(range(ref_index - 1, -1, -1))
    for i in order:
        prev = shifts[i - 1] if i > ref_index else shifts[i + 1]
        # where the feature sits in frame i if it drifted like the previous frame
        cy, cx = y0 - int(round(prev[0])), x0 - int(round(prev[1]))
        wy0, wx0 = max(cy - r, 0), max(cx - r, 0)
        wy1, wx1 = min(cy + (y1 - y0) + r, ny), min(cx + (x1 - x0) + r, nx)
        win = _filtered(st[i], filt)[wy0:wy1, wx0:wx1]
        c = ncc_map(win, template)
        py, px = np.unravel_index(int(np.argmax(c)), c.shape)
        fy, fx = float(py), float(px)
        if subpixel:
            if 0 < py < c.shape[0] - 1:
                fy += _parabolic(c[py - 1, px], c[py, px], c[py + 1, px])
            if 0 < px < c.shape[1] - 1:
                fx += _parabolic(c[py, px - 1], c[py, px], c[py, px + 1])
        shifts[i] = y0 - (wy0 + fy), x0 - (wx0 + fx)
        score[i] = float(c[py, px])
    return shifts, score
