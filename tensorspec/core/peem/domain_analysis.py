"""Spatial domain analysis of one dichroism map (no Qt).

The pipeline of the Spatial Domain Analysis (SDA) toolkit (``xmcd_toolkit.analyze_xmcd`` with
``three_state_field`` / ``despeckle_mz`` / ``per_state_morphometry``), on a map pushed from the PEEM
panel, step by step:

    a  normalize        raw values: 0.5-99.5 percentile clip to [0, 1]; "as shown": the colour
                        range (vmin, vmax) and gamma of the Map view
    b  denoise          TV-Chambolle (weight 0.20), median, or none
    c  local std        texture map (9 px window)
    d  flake (ROI)      the pushed ROI, automatic (Otsu on c, closing, fill holes, largest blob), or all
    e  background       Gaussian low-pass (sigma 25 px), the mean of a background region, or none
                        (zero = asymmetry 0)
    f  flattened        b - e (optionally FFT-filtered)
    g  asymmetry m      f / P99(|f| in the flake), clipped to [-1, 1]
    h  three states     m > m0 up, m < -m0 down, else neutral; fragments < min_px -> neutral
    i  up / down        two states: split at zero (background region / none) or Otsu (Gaussian)
    j  width map        local full width 2 * distance transform (nm when calibrated)
    k  skeleton         medial axis of each state (+1 up, -1 down)
    l  walls            the m = 0 contour inside the flake

Numbers: f_up, f_down, <m_z> = (A_up - A_down) / (A_up + A_down), three-state fractions, domain counts
and mean area, widths (2 A / P and skeleton mean / median per state), period, wall length and density,
stripe direction and alignment. Defaults are the toolkit's ``defaults.py``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

DEFAULTS: dict[str, Any] = {
    "denoise": "tv", "tv_weight": 0.20, "median_size": 3,
    "contrast_window": 9, "flake": "pushed", "min_flake_px": 800,
    "background": "gaussian", "flatten_sigma": 25.0,
    "fft_filter": False, "fft_cutoff": 20, "fft_mode": "radial",
    "m_threshold": 0.05, "min_px": 12, "connectivity": 1, "norm_percentile": 99.0,
}
STEPS = ("normalized", "denoised", "local_std", "flake", "background", "flattened", "asymmetry",
         "three_states", "up_down", "width_map", "skeleton", "walls")
STEP_LABELS = {
    "normalized": "(a) normalized", "denoised": "(b) denoised", "local_std": "(c) local std (texture)",
    "flake": "(d) flake (ROI)", "background": "(e) background", "flattened": "(f) flattened",
    "asymmetry": "(g) asymmetry m [-1, 1]", "three_states": "(h) three states",
    "up_down": "(i) up (+1) / down (-1)", "width_map": "(j) domain width map",
    "skeleton": "(k) domain skeleton", "walls": "(l) domain walls",
}
_DIAG = float(np.sqrt(2.0))


# ------------------------------------------------------------------ helpers
def _structure(connectivity: int) -> np.ndarray:
    return ndi.generate_binary_structure(2, 1 if int(connectivity) == 1 else 2)


def _remove_small(mask: np.ndarray, min_px: int, connectivity: int = 1) -> np.ndarray:
    """Keep connected components with at least min_px pixels (>=, as the toolkit)."""
    mask = np.asarray(mask, bool)
    if not min_px:
        return mask
    lbl, n = ndi.label(mask, structure=_structure(connectivity))
    if n < 1:
        return mask
    sizes = np.bincount(lbl.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lbl]


def _keep_largest(mask: np.ndarray) -> np.ndarray:
    lbl, n = ndi.label(mask)
    if n < 1:
        return mask
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    return lbl == int(sizes.argmax())


# ------------------------------------------------------------------ a-g
def normalize(a: np.ndarray, mode: str = "raw", clip: tuple[float, float] = (0.5, 99.5),
              vmin: float | None = None, vmax: float | None = None, gamma: float = 1.0) -> tuple[np.ndarray, float]:
    """(a) Map -> [0, 1] image and the image level of asymmetry 0.

    raw: robust percentile clip; shown: the Map view's colour range and gamma (as displayed)."""
    a = np.asarray(a, float)
    fin = np.isfinite(a)
    if not fin.any():
        raise ValueError("the map has no finite values")
    if mode == "shown":
        if vmin is None or vmax is None or not vmax > vmin:
            raise ValueError("'as shown' needs the colour range (vmin < vmax)")
        lo, hi = float(vmin), float(vmax)
    elif mode == "raw":
        lo, hi = (float(v) for v in np.percentile(a[fin], clip))
        if hi <= lo:
            lo, hi = float(a[fin].min()), float(a[fin].max())
    else:
        raise ValueError("mode must be 'raw' or 'shown'")
    span = hi - lo + 1e-12
    img = np.clip((np.where(fin, a, np.nanmedian(a[fin])) - lo) / span, 0.0, 1.0)
    zero = float((0.0 - lo) / span)  # not clipped: asymmetry 0 may lie outside the range
    g = float(gamma or 1.0)
    if abs(g - 1.0) > 1e-9:
        img, zero = img ** g, float(np.sign(zero) * abs(zero) ** g)
    return img, zero


def denoise(img: np.ndarray, method: str = "tv", tv_weight: float = 0.20, median_size: int = 3) -> np.ndarray:
    """(b) TV-Chambolle (edge-preserving), median, or none."""
    if method == "tv":
        from skimage.restoration import denoise_tv_chambolle

        return denoise_tv_chambolle(np.asarray(img, float), weight=float(tv_weight))
    if method == "median":
        return ndi.median_filter(np.asarray(img, float), size=int(median_size))
    if method == "none":
        return np.asarray(img, float)
    raise ValueError("denoise must be 'tv', 'median' or 'none'")


def local_std(img: np.ndarray, window: int = 9) -> np.ndarray:
    """(c) Local standard deviation (texture): domains keep it high, flat background low."""
    m = ndi.uniform_filter(img, size=int(window))
    sq = ndi.uniform_filter(img * img, size=int(window))
    return np.sqrt(np.maximum(sq - m * m, 0.0))


def edge_strips(a: np.ndarray, k: float = 2.0, max_frac: float = 0.2, margin: int = 5) -> np.ndarray:
    """Mask of the field without drift strips: rows / columns at an edge whose scatter is above k times the
    median row / column scatter (the area left uncovered by the drift shift, filled from the edge).
    At most max_frac of the field is cut on each side; margin px more are cut where a strip is found."""
    a = np.asarray(a, float)
    keep = np.isfinite(a)
    ny, nx = a.shape
    for axis, n in ((0, nx), (1, ny)):  # axis 0: scatter of each column; axis 1: of each row
        sd = np.nanstd(a, axis=axis)
        ref = float(np.nanmedian(sd))
        hi = sd > k * ref
        lim = int(max_frac * n)
        for side in (0, 1):
            idx = range(lim) if side == 0 else range(n - 1, n - 1 - lim, -1)
            cut = 0
            for i in idx:
                if not hi[i]:
                    break
                cut += 1
            if cut:
                cut = min(cut + margin, lim)
                sl = slice(0, cut) if side == 0 else slice(n - cut, n)
                if axis == 0:
                    keep[:, sl] = False
                else:
                    keep[sl, :] = False
    return keep


def auto_flake(texture: np.ndarray, min_px: int = 800, closing_px: int = 5,
               valid: np.ndarray | None = None) -> np.ndarray:
    """(d) Otsu threshold of the texture map, closing (disk), hole filling, small blobs removed,
    largest kept."""
    from skimage import morphology
    from skimage.filters import threshold_otsu

    valid = np.ones(texture.shape, bool) if valid is None else np.asarray(valid, bool)
    roi = (texture > threshold_otsu(texture[valid])) & valid
    roi = morphology.closing(roi, morphology.disk(int(closing_px)))
    roi = ndi.binary_fill_holes(roi)
    roi = _remove_small(roi, int(min_px))
    return _keep_largest(roi)


def flake_outline(mask: np.ndarray, tolerance: float = 1.0) -> list[list[float]]:
    """Largest outline of a mask as polygon points [[x, y], ...] (an ROI that can be saved and reused)."""
    from skimage import measure

    contours = measure.find_contours(np.pad(np.asarray(mask, float), 1), 0.5)
    if not contours:
        raise ValueError("empty mask: no outline")
    c = measure.approximate_polygon(max(contours, key=len), tolerance=float(tolerance)) - 1.0
    return [[float(x), float(y)] for y, x in c]


def _lowpass(field: np.ndarray, sigma: float) -> np.ndarray:
    """Large-sigma Gaussian on a downsampled copy (as the toolkit; exact below sigma 30)."""
    sigma = float(sigma)
    if sigma <= 30:
        return ndi.gaussian_filter(field, sigma=sigma)
    d = max(1, int(sigma // 8))
    small = ndi.gaussian_filter(field[::d, ::d], sigma=sigma / d)
    return ndi.zoom(small, (field.shape[0] / small.shape[0], field.shape[1] / small.shape[1]), order=1)


def background(den: np.ndarray, method: str = "gaussian", sigma: float = 25.0,
               bg_mask: np.ndarray | None = None, zero: float = 0.0) -> np.ndarray:
    """(e) Background estimate: Gaussian low-pass, a region's mean (the magnetic zero), or the
    image level of asymmetry 0 ('none')."""
    if method == "gaussian":
        return _lowpass(den, sigma)
    if method == "region":
        if bg_mask is None or not np.asarray(bg_mask, bool).any():
            raise ValueError("no background region: draw one in 6. ROI and background and push again")
        return np.full(den.shape, float(den[np.asarray(bg_mask, bool)].mean()))
    if method == "none":
        return np.full(den.shape, float(zero))
    raise ValueError("background must be 'gaussian', 'region' or 'none'")


def asymmetry(flat: np.ndarray, roi: np.ndarray, percentile: float = 99.0) -> tuple[np.ndarray, float]:
    """(g) m = flat / P99(|flat| in the flake), clipped to [-1, 1]: saturated domains near +-1."""
    scale = float(np.percentile(np.abs(flat[roi]), percentile)) + 1e-12
    return np.clip(flat / scale, -1.0, 1.0), scale


# ------------------------------------------------------------------ h-l
def three_states(m: np.ndarray, roi: np.ndarray, m_threshold: float = 0.05, min_px: int = 12) -> np.ndarray:
    """(h) +1 / 0 / -1 with a neutral band |m| <= m0; fragments below min_px revert to 0."""
    mz = np.zeros(m.shape, np.int8)
    mz[roi & (m > m_threshold)] = 1
    mz[roi & (m < -m_threshold)] = -1
    up = _remove_small(mz == 1, min_px)
    dn = _remove_small(mz == -1, min_px)
    out = np.zeros_like(mz)
    out[up], out[dn] = 1, -1
    return out


def _skeleton_length(skel: np.ndarray) -> float:
    s = skel.astype(bool)
    ortho = np.logical_and(s[:, :-1], s[:, 1:]).sum() + np.logical_and(s[:-1, :], s[1:, :]).sum()
    diag = np.logical_and(s[:-1, :-1], s[1:, 1:]).sum() + np.logical_and(s[:-1, 1:], s[1:, :-1]).sum()
    return float(ortho + _DIAG * diag) if s.sum() >= 2 else float(s.sum())


def state_width(mask: np.ndarray, min_px: int = 12) -> dict[str, Any]:
    """Distance transform on the skeleton of one state (toolkit ``domain_width``), plus 2 A / P."""
    from skimage import measure
    from skimage.morphology import skeletonize

    mask = _remove_small(np.asarray(mask, bool), min_px)
    out: dict[str, Any] = {"area_px": int(mask.sum())}
    if mask.sum() < 4:
        out.update(widths_px=np.array([]), dt=np.zeros(mask.shape), skeleton=np.zeros(mask.shape, bool),
                   skeleton_len_px=0.0, perimeter_px=0.0)
        return out
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - 1, 0), min(ys.max() + 2, mask.shape[0])
    x0, x1 = max(xs.min() - 1, 0), min(xs.max() + 2, mask.shape[1])
    sub = mask[y0:y1, x0:x1]
    dt = np.zeros(mask.shape)
    skel = np.zeros(mask.shape, bool)
    dt[y0:y1, x0:x1] = ndi.distance_transform_edt(sub)
    skel[y0:y1, x0:x1] = skeletonize(sub)
    out.update(widths_px=2.0 * dt[skel], dt=dt, skeleton=skel, skeleton_len_px=_skeleton_length(skel),
               perimeter_px=float(measure.perimeter(mask)))
    return out


def walls(m: np.ndarray, roi: np.ndarray, smooth: float = 1.0) -> tuple[np.ndarray, float]:
    """(l) The m = 0 contour (marching squares on lightly smoothed m) inside the flake; its length (px)."""
    from skimage import measure

    ms = ndi.gaussian_filter(np.asarray(m, float), float(smooth))
    edge = np.zeros(m.shape, bool)
    length = 0.0
    ny, nx = m.shape
    for c in measure.find_contours(ms, 0.0):
        r = np.clip(c[:, 0].astype(int), 0, ny - 1)
        cc = np.clip(c[:, 1].astype(int), 0, nx - 1)
        inside = roi[r, cc]
        edge[r[inside], cc[inside]] = True
        seg = np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1]))
        length += float(seg[inside[1:] & inside[:-1]].sum())
    return edge, length


def stripe_orientation(m: np.ndarray, roi: np.ndarray, period_px: float | None = None) -> dict[str, float]:
    """Stripe direction (deg counter-clockwise from +x as displayed, top-left origin, 0-180) and
    alignment (0 = isotropic labyrinth, 1 = perfect stripes) from the FFT power of the domain band
    (periods period/2 .. 2 period; the whole spectrum without a period)."""
    ys, xs = np.nonzero(roi)
    sub = np.where(roi, m, 0.0)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    w = np.outer(np.hanning(sub.shape[0]), np.hanning(sub.shape[1]))
    p = np.abs(np.fft.fftshift(np.fft.fft2((sub - sub.mean()) * w))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(sub.shape[0]))[:, None] * np.ones((1, sub.shape[1]))
    kx = np.fft.fftshift(np.fft.fftfreq(sub.shape[1]))[None, :] * np.ones((sub.shape[0], 1))
    k = np.hypot(kx, ky)
    band = k > 0
    if period_px and np.isfinite(period_px) and period_px > 2:
        band &= (k >= 0.5 / period_px) & (k <= 2.0 / period_px)
    phi = np.arctan2(-ky, kx)  # wave-vector angle as displayed (rows go down)
    wgt = p[band]
    z = np.sum(wgt * np.exp(2j * phi[band])) / max(wgt.sum(), 1e-300)
    stripes = (np.degrees(0.5 * np.angle(z)) + 90.0) % 180.0  # stripes run normal to the wave vector
    return {"stripe_angle_deg": float(stripes), "alignment": float(np.abs(z))}


# ------------------------------------------------------------------ pipeline
def analyze(a: np.ndarray, *, um_per_px: float | None = None, roi: np.ndarray | None = None,
            bg_mask: np.ndarray | None = None, normalize_mode: str = "raw", vmin: float | None = None,
            vmax: float | None = None, gamma: float = 1.0, **params) -> dict[str, Any]:
    """Run steps a-l on one map. ``params`` override DEFAULTS. Returns {"layers", "metrics", "areas"}."""
    p = {**DEFAULTS, **params}
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    a = np.asarray(a, float)
    if a.ndim != 2:
        raise ValueError("domain analysis needs a 2-D map")
    img, zero = normalize(a, normalize_mode, vmin=vmin, vmax=vmax, gamma=gamma)
    den = denoise(img, p["denoise"], p["tv_weight"], p["median_size"])
    tex = local_std(den, p["contrast_window"])

    if p["flake"] == "auto":
        flake = auto_flake(tex, p["min_flake_px"], valid=edge_strips(a))
    elif p["flake"] == "pushed" and roi is not None:
        flake = np.asarray(roi, bool)
    elif p["flake"] in ("pushed", "all"):
        flake = np.ones(a.shape, bool)
    else:
        raise ValueError("flake must be 'pushed', 'auto' or 'all'")
    flake = flake & np.isfinite(a)
    if flake.sum() < 50:
        raise ValueError("the flake / ROI has fewer than 50 pixels")

    bgr = background(den, p["background"], p["flatten_sigma"], bg_mask, zero)
    flat = den - bgr
    if p["fft_filter"]:
        from tensorspec.core.peem.image_analysis import fft_filter

        flat = fft_filter(flat, int(p["fft_cutoff"]), p["fft_mode"])
    m, scale = asymmetry(flat, flake, p["norm_percentile"])
    mz = three_states(m, flake, p["m_threshold"], p["min_px"])

    # (i) two states: the zero is known with a background region or none, else Otsu (toolkit)
    if p["background"] in ("region", "none"):
        split, split_from = 0.0, "zero"
    else:
        from skimage.filters import threshold_otsu

        split, split_from = float(threshold_otsu(flat[flake])), "otsu"
    up = flake & (flat > split)
    down = flake & (flat <= split)
    ud = np.zeros(a.shape, np.int8)
    ud[up], ud[down] = 1, -1

    st = _structure(p["connectivity"])
    up_c = _remove_small(up, p["min_px"], p["connectivity"])
    dn_c = _remove_small(down, p["min_px"], p["connectivity"])
    up_lbl, n_up = ndi.label(up_c, structure=st)
    dn_lbl, n_dn = ndi.label(dn_c, structure=st)
    up_areas = np.bincount(up_lbl.ravel())[1:] if n_up else np.array([], int)
    dn_areas = np.bincount(dn_lbl.ravel())[1:] if n_dn else np.array([], int)
    all_areas = np.concatenate([up_areas, dn_areas])

    wu, wd = state_width(mz == 1, p["min_px"]), state_width(mz == -1, p["min_px"])
    width_map = np.where(mz != 0, 2.0 * (wu["dt"] + wd["dt"]), np.nan)
    skel = np.zeros(a.shape, np.int8)
    skel[wu["skeleton"]], skel[wd["skeleton"]] = 1, -1
    edge, wall_len_contour = walls(m, flake)
    interior = ndi.binary_erosion(flake, iterations=1)
    wall_px = int(((ndi.binary_dilation(up) & down) & interior).sum())  # toolkit wall length

    s = um_per_px or None
    nm = (s * 1000.0) if s else None
    area_up, area_dn = int(up.sum()), int(down.sum())
    area = area_up + area_dn
    n = int(flake.sum())

    def w(v):
        return float(v * nm) if nm else float(v)

    def stat(arr, fn):
        return w(fn(arr)) if arr.size else float("nan")

    pa_up = 2.0 * wu["area_px"] / wu["perimeter_px"] if wu["perimeter_px"] else float("nan")
    pa_dn = 2.0 * wd["area_px"] / wd["perimeter_px"] if wd["perimeter_px"] else float("nan")
    med_up = float(np.median(wu["widths_px"])) if wu["widths_px"].size else float("nan")
    med_dn = float(np.median(wd["widths_px"])) if wd["widths_px"].size else float("nan")
    period_px = med_up + med_dn
    orient = stripe_orientation(m, flake, period_px if np.isfinite(period_px) else None)
    unit = "nm" if nm else "px"
    metrics: dict[str, Any] = {
        "um_per_px": s or float("nan"), "width_unit": unit,
        "flake_area_px": n, "flake_area_um2": n * s * s if s else float("nan"),
        "f_up": area_up / area if area else float("nan"),
        "f_down": area_dn / area if area else float("nan"),
        "mz": (area_up - area_dn) / area if area else float("nan"),
        "split": split_from,
        "p_up": float((mz[flake] == 1).mean()), "p_neutral": float((mz[flake] == 0).mean()),
        "p_down": float((mz[flake] == -1).mean()),
        "mz_three_state": float(mz[flake].mean()),
        "n_domains_up": int(n_up), "n_domains_down": int(n_dn),
        "mean_domain_area_px": float(all_areas.mean()) if all_areas.size else float("nan"),
        "mean_domain_area_um2": float(all_areas.mean() * s * s) if (s and all_areas.size) else float("nan"),
        f"width_up_{unit}": w(pa_up), f"width_down_{unit}": w(pa_dn),
        f"width_skeleton_mean_up_{unit}": stat(wu["widths_px"], np.mean),
        f"width_skeleton_mean_down_{unit}": stat(wd["widths_px"], np.mean),
        f"width_skeleton_median_up_{unit}": stat(wu["widths_px"], np.median),
        f"width_skeleton_median_down_{unit}": stat(wd["widths_px"], np.median),
        f"period_{unit}": w(period_px) if np.isfinite(period_px) else float("nan"),
        "wall_length_px": wall_px, "wall_length_um": wall_px * s if s else float("nan"),
        "wall_density_per_um": wall_px / (area * s * s) * s if (s and area) else float("nan"),
        "wall_contour_length_um": wall_len_contour * s if s else float("nan"),
        **orient,
        "norm_scale": scale,
    }
    layers = {
        "normalized": img, "denoised": den, "local_std": tex, "flake": flake.astype(np.uint8),
        "background": bgr, "flattened": flat, "asymmetry": np.where(flake, m, np.nan),
        "three_states": np.where(flake, mz, 0).astype(np.int8), "up_down": ud,
        "width_map": width_map * (nm or 1.0), "skeleton": skel, "walls": edge.astype(np.uint8),
    }
    sq = (s * s) if s else 1.0
    return {"layers": layers, "metrics": metrics,
            "areas": {"up": up_areas * sq, "down": dn_areas * sq, "unit": "um2" if s else "px2"}}
