"""Map geometry (no Qt): band heights without fitting, 1-D k-means, optical contrast,
map orientation/offset transforms and auto-alignment of a Raman map to the camera image.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (single spectra / map helpers)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from tensorspec.core.io.raman_loaders import MapFile

def band_height(wn: np.ndarray, D: np.ndarray, centre: float,
                half: float) -> np.ndarray:
    """Peak height above the flanking baseline, no fitting: the maximum
    within ±half of centre minus the median of the flanks 2–4 half-widths
    away. D is (n_spectra, n_axis)."""
    pk = (wn > centre - half) & (wn < centre + half)
    fl = (((wn > centre - 4 * half) & (wn < centre - 2 * half))
          | ((wn > centre + 2 * half) & (wn < centre + 4 * half)))
    if pk.sum() < 2 or fl.sum() < 4:
        return np.full(D.shape[0], np.nan)
    return np.nanmax(D[:, pk], axis=1) - np.nanmedian(D[:, fl], axis=1)


def kmeans_1d(v: np.ndarray, k: int, iters: int = 200) -> np.ndarray:
    """Sorted cluster centres for 1-D data (Lloyd's algorithm)."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    k = max(1, min(int(k), v.size))
    if v.size == 0:
        return np.array([])
    c = np.quantile(v, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        lab = np.argmin(np.abs(v[:, None] - c[None, :]), axis=1)
        nc = np.array([v[lab == j].mean() if (lab == j).any() else c[j]
                       for j in range(k)])
        if np.allclose(nc, c):
            break
        c = nc
    return np.sort(c)


def optical_contrast(mf: MapFile, opt: MapFile) -> Tuple[np.ndarray, float]:
    """Optical contrast (I - I_substrate) / I_substrate under each spectrum.

    The camera image is averaged over each Raman pixel's footprint rather
    than sampled at its centre, so a pixel is described by the same area the
    map step implies. The substrate level is the mode of the whole frame,
    which is bare substrate in nearly every flake image.
    """
    img = opt.image.astype(float)
    px = (opt.extent[1] - opt.extent[0]) / img.shape[1]
    img = gaussian_filter(median_filter(img, size=3), max(1.0, 0.15 / px))
    hist, edges = np.histogram(img[np.isfinite(img)], bins=256)
    hist = gaussian_filter(hist.astype(float), 2.0)
    j = int(np.argmax(hist))
    sub = 0.5 * (edges[j] + edges[j + 1])
    x0, x1, y0, y1 = opt.extent
    H, W = img.shape
    dx = float(np.median(np.diff(mf.ux))) if len(mf.ux) > 1 else 1.0
    dy = float(np.median(np.diff(mf.uy))) if len(mf.uy) > 1 else 1.0
    out = np.full(mf.n_points, np.nan)
    for i in range(mf.n_points):
        c0 = int((mf.x[i] - dx / 2 - x0) / (x1 - x0) * W)
        c1 = int((mf.x[i] + dx / 2 - x0) / (x1 - x0) * W)
        r0 = int((mf.y[i] - dy / 2 - y0) / (y1 - y0) * H)
        r1 = int((mf.y[i] + dy / 2 - y0) / (y1 - y0) * H)
        c0, c1 = max(c0, 0), min(max(c1, c0 + 1), W)
        r0, r1 = max(r0, 0), min(max(r1, r0 + 1), H)
        if c0 < W and r0 < H and c1 > c0 and r1 > r0:
            out[i] = (np.nanmean(img[r0:r1, c0:c1]) - sub) / sub
    return out, sub




INSTRUMENT_TRANSFORM = {"swap": False, "flip_x": True, "flip_y": False,
                        "dx": 0.0, "dy": 0.0}


def default_transform(mf: MapFile, cfg: dict) -> dict:
    """Placement used before any alignment: the instrument convention, or the
    header reading if the user prefers it."""
    if cfg.get("orientation_default", "instrument") == "headers":
        return header_transform(mf)
    return dict(INSTRUMENT_TRANSFORM)


def header_transform(mf: MapFile) -> dict:
    """The orientation the #AxisType headers imply, as an explicit transform."""
    return {"swap": bool(mf.swapped), "flip_x": False, "flip_y": False,
            "dx": 0.0, "dy": 0.0}


def describe_transform(t: dict) -> str:
    a, b = ("col1", "col0") if t["swap"] else ("col0", "col1")
    def term(sign, name, off):
        s = ("−" if sign else "") + name
        return s + (" %+.1f µm" % off if abs(off) > 1e-9 else "")
    return "X = %s,  Y = %s" % (term(t["flip_x"], a, t["dx"]),
                                term(t["flip_y"], b, t["dy"]))


def apply_map_transform(mf: MapFile, t: Optional[dict]) -> MapFile:
    """Rebuild X/Y from the raw coordinate columns.

    Only coordinates change; spectra stay in file order, so fits made before
    a transform stay valid and nothing needs refitting.
    """
    if t is None or mf.layout != "spectra" or mf.raw0 is None:
        return mf
    a, b = (mf.raw1, mf.raw0) if t["swap"] else (mf.raw0, mf.raw1)
    x = (-a if t["flip_x"] else a) + float(t["dx"])
    y = (-b if t["flip_y"] else b) + float(t["dy"])
    return MapFile(axis=mf.axis, x=x, y=y, data=mf.data, path=mf.path,
                   header=mf.header, layout="spectra", swapped=bool(t["swap"]),
                   raw0=mf.raw0, raw1=mf.raw1)


def band_presence(mf: MapFile, cfg: dict) -> Tuple[np.ndarray, str]:
    """A per-spectrum 'graphene is here' signal straight from the raw data.

    Peak height above the flanking baseline, using the G band when the file
    has one and the 2D band otherwise. Deliberately independent of whether a
    fit was accepted, so weak pixels still contribute to the alignment.
    """
    wn, D = mf.axis, mf.data
    use_g = int(((wn > 1560) & (wn < 1605)).sum()) >= 3
    if use_g:
        c, half, name = 1582.0, 22.0, "G band"
    else:
        c, half, name = 0.5 * (cfg["band_min"] + cfg["band_max"]), 45.0, "2D band"
    pk = (wn > c - half) & (wn < c + half)
    fl = ((wn > c - 4 * half) & (wn < c - 2 * half)) | ((wn > c + 2 * half) & (wn < c + 4 * half))
    if pk.sum() < 2 or fl.sum() < 4:
        return np.zeros(D.shape[0]), name
    return np.nanmax(D[:, pk], axis=1) - np.nanmedian(D[:, fl], axis=1), name


def auto_align(mf: MapFile, opt: MapFile, cfg: dict, search_um: float = 15.0,
               orientations=None, sign: Optional[int] = None):
    """Find the orientation and offset that put graphene where the camera sees it.

    Every combination of swap / flip-X / flip-Y is tried, each with a search
    over translations; the score is the correlation between the Raman band
    height and the (smoothed, dust-clipped) optical brightness at each
    spectrum's position. Absolute correlation is used because a flake can look
    brighter or darker than its substrate depending on oxide and lighting.

    Returns (candidates sorted best-first, signal name). Each candidate is
    (|r|, r, transform).
    """
    sig, name = band_presence(mf, cfg)
    if mf.raw0 is None or not np.isfinite(sig).any() or np.nanstd(sig) == 0:
        return [], name
    img = opt.image.astype(float)
    px = (opt.extent[1] - opt.extent[0]) / img.shape[1]
    img = gaussian_filter(median_filter(img, size=3), max(1.0, 0.25 / px))
    img = np.minimum(img, np.nanpercentile(img, 97))      # dust is not graphene
    x0, x1, y0, y1 = opt.extent
    H, W = img.shape
    sig = np.nan_to_num(sig)
    if sig.size > 4000:                                   # plenty for a correlation
        keep = np.linspace(0, sig.size - 1, 4000).astype(int)
    else:
        keep = np.arange(sig.size)
    s_ = sig[keep]

    def score(X, Y):
        ix = ((X - x0) / (x1 - x0) * W).astype(int)
        iy = ((Y - y0) / (y1 - y0) * H).astype(int)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        if ok.sum() < max(20, 0.5 * ok.size):
            return np.nan
        a, b = s_[ok], img[iy[ok], ix[ok]]
        if a.std() == 0 or b.std() == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    out = []
    r0, r1 = mf.raw0[keep], mf.raw1[keep]
    if orientations is None:
        orientations = [(sw, a, b) for sw in (False, True)
                        for a in (False, True) for b in (False, True)]
    for swap, fx, fy in orientations:
        if True:
            if True:
                a, b = (r1, r0) if swap else (r0, r1)
                A, B = (-a if fx else a), (-b if fy else b)
                best = (0.0, 0.0, 0.0, 0.0)
                grid = np.arange(-search_um, search_um + 1e-9, 1.0)
                for dx in grid:                            # coarse ...
                    for dy in grid:
                        r = score(A + dx, B + dy)
                        if np.isfinite(r) and (abs(r) if sign is None else sign * r) > best[0]:
                            best = ((abs(r) if sign is None else sign * r), r, dx, dy)
                fine = np.arange(-1.0, 1.0 + 1e-9, 0.25)
                _, _, cx, cy = best
                for dx in cx + fine:                       # ... then fine
                    for dy in cy + fine:
                        r = score(A + dx, B + dy)
                        if np.isfinite(r) and (abs(r) if sign is None else sign * r) > best[0]:
                            best = ((abs(r) if sign is None else sign * r), r, dx, dy)
                out.append((best[0], best[1],
                            {"swap": swap, "flip_x": fx, "flip_y": fy,
                             "dx": round(best[2], 2), "dy": round(best[3], 2)}))
    out.sort(key=lambda t: -t[0])
    return out, name
