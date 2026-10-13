"""Detector-level corrections for UE49 SPEEM images.

Ports of ``speem_eval_20230427_TPX3.ipf`` (TPX3Correction, DarkCountsAtCorners,
subtractOffset, I0Normalization, binning_2x2) and ``circularcrop*.ipf``.

Arrays are numpy ``(y, x)``. Igor stores images as ``w[p][q]`` with p = x, so an
Igor wave equals the TIFF array transposed; the TPX3 assembly below runs in the
Igor frame and converts back. Unlike Igor, all arithmetic is float (no uint16
truncation between steps).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

TPX3_RAW = 512
TPX3_OUT = 514
TPX3_COEF_FALLBACK = 3.0
TPX3_IAVG_MIN = 25.0


def _expand_big_pixels(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two double-width pixel lines (len 512) -> four lines (4, 514)."""
    out = np.zeros((4, TPX3_OUT), dtype=float)
    for dst, src in ((slice(0, 256), slice(0, 256)), (slice(258, 514), slice(256, 512))):
        a0, b0 = a[src], b[src]
        out[0, dst] = a0
        out[1, dst] = (2.0 * a0 + b0) / 3.0
        out[2, dst] = (2.0 * b0 + a0) / 3.0
        out[3, dst] = b0
    return out


def tpx3_cross_coef(img: np.ndarray) -> float:
    """Intensity ratio of the double-size cross pixels to their neighbours."""
    w = np.asarray(img, dtype=float).T  # Igor frame
    iavg = 0.5 * (w[20:491, 253:255].mean() + w[20:491, 257:259].mean())
    icross = w[20:491, 255:257].mean()
    return float(icross / iavg) if iavg > TPX3_IAVG_MIN else TPX3_COEF_FALLBACK


def tpx3_assemble(img: np.ndarray, orientation: str = "antitranspose") -> np.ndarray:
    """512x512 quad-chip frame -> 514x514 with the chip cross re-gridded.

    ``orientation`` reproduces Igor's ``ImageRotate /V`` + ``/W`` (a reflection
    about one diagonal). "antitranspose" matches Igor output: it reproduces an
    Igor-processed UE49 co-add at r = 0.987 and puts the 514x514 dead-pixel map
    on the dark pixels. "transpose" / "none" are kept for other detector setups.
    """
    img = np.asarray(img, dtype=float)
    if img.shape != (TPX3_RAW, TPX3_RAW):
        raise ValueError(f"TPX3 frames must be 512x512, got {img.shape}")
    coef = tpx3_cross_coef(img)
    w = img.T
    out = np.zeros((TPX3_OUT, TPX3_OUT), dtype=float)
    out[0:255, 0:255] = w[0:255, 0:255]
    out[0:255, 259:514] = w[0:255, 257:512]
    out[259:514, 0:255] = w[257:512, 0:255]
    out[259:514, 259:514] = w[257:512, 257:512]
    out[255:259, :] = _expand_big_pixels(w[255, :], w[256, :]) / coef
    out[:, 255:259] = _expand_big_pixels(w[:, 255], w[:, 256]).T / coef
    out[255, 255:259] /= coef
    out[256, 255:259] = out[255, 255:259]
    out[258, 255:259] /= coef
    out[257, 255:259] = out[258, 255:259]
    if orientation == "transpose":
        out = out.T
    elif orientation == "antitranspose":
        out = out[::-1, ::-1].T
    elif orientation != "none":
        raise ValueError(f"unknown orientation {orientation!r}")
    return out.T  # back to numpy (y, x)


def correct_dead_pixels(img: np.ndarray, dead_map: np.ndarray) -> np.ndarray:
    """Igor rule: value 2 -> mean of the 4 diagonal neighbours at distance 2.

    0 keeps the pixel, 1 blends 50/50 (out = img*(2-d)/2 + d*diag/8).
    """
    img = np.asarray(img, dtype=float)
    d = np.asarray(dead_map, dtype=float)
    if d.shape != img.shape:
        raise ValueError(f"dead-pixel map {d.shape} does not match image {img.shape}")
    ny, nx = img.shape
    yy, xx = np.arange(ny), np.arange(nx)

    def at(dy: int, dx: int) -> np.ndarray:
        return img[np.ix_(np.clip(yy + dy, 0, ny - 1), np.clip(xx + dx, 0, nx - 1))]

    diag = at(-2, -2) + at(2, 2) + at(2, -2) + at(-2, 2)
    return img * (2.0 - d) * 0.5 + d * diag / 8.0


def flat_field(img: np.ndarray, norm_image: np.ndarray) -> np.ndarray:
    """img * mean(norm) / norm (division flat field, no dark term)."""
    norm = np.asarray(norm_image, dtype=float)
    if norm.shape != np.shape(img):
        raise ValueError(f"norm image {norm.shape} does not match image {np.shape(img)}")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.asarray(img, dtype=float) * norm.mean() / norm
    return np.where(np.isfinite(out), out, 0.0)


def tpx3_correct(
    img: np.ndarray,
    dead_map: np.ndarray | None = None,
    norm_image: np.ndarray | None = None,
    orientation: str = "antitranspose",
) -> np.ndarray:
    """Full LoadTPXTiff chain. Non-512 frames skip the cross re-gridding."""
    out = np.asarray(img, dtype=float)
    if out.shape == (TPX3_RAW, TPX3_RAW):
        out = tpx3_assemble(out, orientation=orientation)
    if dead_map is not None:
        out = correct_dead_pixels(out, dead_map)
    if norm_image is not None:
        out = flat_field(out, norm_image)
    return out


def load_detector_map(path: str | Path) -> np.ndarray:
    import tifffile

    return np.asarray(tifffile.imread(str(path)), dtype=float)


def dark_counts_at_corners(img: np.ndarray, size: int = 21) -> float:
    """Mean of the four ``size`` x ``size`` corner blocks."""
    a = np.asarray(img, dtype=float)
    s = int(size)
    blocks = (a[:s, :s], a[:s, -s:], a[-s:, -s:], a[-s:, :s])
    return float(np.mean([b.mean() for b in blocks]))


def subtract_offset(img: np.ndarray, offset: float) -> np.ndarray:
    """img - offset, clipped at 0, then +1 (Igor's divide-by-zero guard)."""
    return np.clip(np.asarray(img, dtype=float) - float(offset), 0.0, None) + 1.0


def i0_normalize(stack: np.ndarray, i0: np.ndarray) -> np.ndarray:
    """Per-frame stack / I0 * rms(I0), keeping counts in the original range."""
    stack = np.asarray(stack, dtype=float)
    i0 = np.asarray(i0, dtype=float).reshape(-1)
    if i0.shape[0] != stack.shape[0]:
        raise ValueError(f"I0 length {i0.shape[0]} != frames {stack.shape[0]}")
    if np.any(~np.isfinite(i0)) or np.any(i0 <= 0):
        raise ValueError("I0 must be finite and positive for every frame")
    rms = float(np.sqrt(np.mean(i0**2)))
    return stack / i0[:, None, None] * rms


def bin2x2(img: np.ndarray) -> np.ndarray:
    """2x2 sum binning (Igor binning_2x2); odd edges are dropped."""
    a = np.asarray(img, dtype=float)
    ny, nx = (a.shape[-2] // 2) * 2, (a.shape[-1] // 2) * 2
    a = a[..., :ny, :nx]
    return a[..., 0::2, 0::2] + a[..., 1::2, 0::2] + a[..., 0::2, 1::2] + a[..., 1::2, 1::2]


def circular_mask(ny: int, nx: int, cy: float | None = None, cx: float | None = None,
                  radius: float | None = None) -> np.ndarray:
    """Boolean disk (circularcrop*.ipf); defaults: image centre, radius min/4."""
    cy = ny / 2.0 if cy is None else float(cy)
    cx = nx / 2.0 if cx is None else float(cx)
    radius = min(ny, nx) / 4.0 if radius is None else float(radius)
    if radius <= 0:
        raise ValueError("radius must be positive")
    yy, xx = np.mgrid[0:ny, 0:nx]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def circular_crop(img: np.ndarray, cy=None, cx=None, radius=None, fill: float = np.nan) -> np.ndarray:
    """Copy of ``img`` (or each frame of a stack) with pixels outside the disk set to ``fill``."""
    a = np.array(img, dtype=float)
    mask = circular_mask(a.shape[-2], a.shape[-1], cy, cx, radius)
    a[..., ~mask] = fill
    return a


def auto_fov_circle(img: np.ndarray, dark_frac: float = 0.3) -> tuple[float, float, float]:
    """Illuminated field-of-view disk (cy, cx, radius), centred on the image.

    The radius is where the azimuthally averaged (heavily smoothed) intensity
    first falls below ``dark_frac`` x the central level; when the image is lit
    to its edge (UE49 10-25 um FOVs on the 514 px TPX3) the inscribed circle is
    returned. Bright sample features do not move the disk, unlike a threshold
    on the brightest pixels.
    """
    from scipy import ndimage

    a = np.nan_to_num(np.asarray(img, dtype=float))
    ny, nx = a.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    r_max = min(ny, nx) / 2.0
    sm = ndimage.gaussian_filter(a, max(2.0, min(ny, nx) / 50.0))
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(yy - cy, xx - cx)
    step = max(2.0, r_max / 40.0)
    edges = np.arange(0.0, r_max, step)
    prof = np.array([np.median(sm[(rr >= e) & (rr < e + step)]) for e in edges])
    centre = np.median(prof[: max(1, len(prof) // 4)])
    dark = np.flatnonzero(prof < dark_frac * centre)
    radius = float(edges[dark[0]]) if dark.size else r_max
    return float(cy), float(cx), radius
