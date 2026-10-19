"""Register one PEEM map onto another when the two are not just shifted (no Qt).

Two maps of the same area taken at a different temperature, azimuth, field of view or after a
refocus differ by rotation and scale as well as shift (UE49 June 2025: 1.8 deg between RT and
110 K, +-15 deg azimuths, FOV 10 vs 15 um = 1.6x and ~8.5 deg, 88.5 deg after a refocus). The
images used for the fit are intensity images (sum of the two channels: topography, no magnetic
sign), so maps of opposite or different dichroism still register.

    tr, r = align(ref_image, mov_image, ref_roi, mov_roi, scale=um_ref / um_mov, search="auto")
    moved = tr.warp(mov_map)          # mov_map on the reference grid (NaN outside)

Transform convention: reference pixel p = (y, x) takes the moving image's value at q = A p + t.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from tensorspec.core.peem.registration import _prep, estimate_shift


@dataclass
class Transform:
    A: np.ndarray = field(default_factory=lambda: np.eye(2))
    t: np.ndarray = field(default_factory=lambda: np.zeros(2))
    model: str = "shift"
    n_tiles: int = 0

    @classmethod
    def from_shift(cls, dy: float, dx: float) -> "Transform":
        return cls(np.eye(2), -np.array([dy, dx], float))

    def warp(self, img: np.ndarray, order: int = 1) -> np.ndarray:
        """``img`` resampled onto the reference grid; outside it NaN."""
        a = np.asarray(img, dtype=float)
        return ndimage.affine_transform(a, self.A, offset=self.t, order=order, mode="constant", cval=np.nan)

    def then(self, inner: "Transform") -> "Transform":
        """self maps grid G -> moving image; inner maps reference -> G. Composite reference -> moving."""
        return Transform(self.A @ inner.A, self.A @ inner.t + self.t, inner.model, inner.n_tiles)

    def describe(self, centre: tuple[float, float]) -> dict[str, float]:
        """Shift of ``centre``, rotation (deg) and scale of the moving image against the reference."""
        c = np.asarray(centre, float)
        dy, dx = c - (self.A @ c + self.t)
        u, _, vt = np.linalg.svd(self.A)
        r = u @ vt
        return {"shift_dy_px": float(dy), "shift_dx_px": float(dx),
                "rotation_deg": float(np.degrees(np.arctan2(r[1, 0], r[0, 0]))),
                "scale": float(np.sqrt(abs(np.linalg.det(self.A))))}

    def as_dict(self) -> dict:
        return {"A": self.A.tolist(), "t": self.t.tolist(), "model": self.model}


def _fill(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    ok = np.isfinite(a)
    return np.where(ok, a, np.nanmedian(a) if ok.any() else 0.0)


def _ncc(a: np.ndarray, b: np.ndarray, filt: str | None) -> float:
    a, b = _prep(a, filt), _prep(b, filt)
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def central_box(roi: np.ndarray, frac: float = 0.62) -> tuple[int, int, int, int]:
    """Square inside the field of view (fraction of its radius), (y0, y1, x0, x1)."""
    yy, xx = np.nonzero(roi)
    cy, cx = yy.mean(), xx.mean()
    r = np.sqrt(roi.sum() / np.pi) * frac
    return int(cy - r), int(cy + r), int(cx - r), int(cx + r)


def overlap(ref: np.ndarray, mov: np.ndarray, tr: Transform, box, filt: str | None = "smooth_edges") -> float:
    """Correlation of the reference and the warped moving image inside ``box`` (edge images by default,
    so values of different fits compare)."""
    y0, y1, x0, x1 = box
    return _ncc(_fill(ref)[y0:y1, x0:x1], _fill(tr.warp(_fill(mov)))[y0:y1, x0:x1], filt)


def _tiles(box, tile: int):
    y0, y1, x0, x1 = box
    ny, nx = max(1, round((y1 - y0) / tile)), max(1, round((x1 - x0) / tile))
    ys, xs = np.linspace(y0, y1, ny + 1).astype(int), np.linspace(x0, x1, nx + 1).astype(int)
    return [(ys[i], ys[i + 1], xs[j], xs[j + 1]) for i in range(ny) for j in range(nx)
            if ys[i + 1] - ys[i] >= 32 and xs[j + 1] - xs[j] >= 32]


def _lsq(p: np.ndarray, q: np.ndarray, model: str):
    if model == "affine":
        sol, *_ = np.linalg.lstsq(np.c_[p, np.ones(len(p))], q, rcond=None)
        return sol[:2].T, sol[2]
    n = len(p)
    M = np.zeros((2 * n, 4))
    M[0::2] = np.c_[p[:, 0], -p[:, 1], np.ones(n), np.zeros(n)]
    M[1::2] = np.c_[p[:, 1], p[:, 0], np.zeros(n), np.ones(n)]
    (a, b, ty, tx), *_ = np.linalg.lstsq(M, q.reshape(-1), rcond=None)
    return np.array([[a, -b], [b, a]]), np.array([ty, tx])


def fit_transform(ref: np.ndarray, mov: np.ndarray, box, filt: str | None = "highpass",
                  model: str = "similarity", max_shift: float | None = None, tile: int = 128,
                  min_r: float = 0.3) -> Transform:
    """Global shift in ``box``, then local shifts in tiles -> least-squares similarity (rotation +
    scale) or affine map. Tiles that correlate below ``min_r`` or deviate > 3 px (2.5 robust sigma)
    are dropped; fewer than 6 usable tiles keep the global shift."""
    ref, mov = _fill(ref), _fill(mov)
    g = Transform.from_shift(*estimate_shift(ref, mov, box=box, filt=filt, max_shift=max_shift))
    if model == "shift":
        return g
    moved = _fill(g.warp(mov))
    side = min(box[1] - box[0], box[3] - box[2])
    tile = int(min(max(tile, side // 5), max(48, side // 4)))  # ~4 x 4 to 5 x 5 tiles in the area
    lim = min(tile / 4, 25)
    pad = int(lim) + 2
    ny, nx = ref.shape
    p, q = [], []
    for b in _tiles(box, tile):
        y0, y1, x0, x1 = b
        d = estimate_shift(ref, moved, box=b, filt=filt, max_shift=lim)
        Y0, Y1, X0, X1 = max(y0 - pad, 0), min(y1 + pad, ny), max(x0 - pad, 0), min(x1 + pad, nx)
        local = _fill(Transform.from_shift(*d).warp(moved[Y0:Y1, X0:X1]))[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
        if _ncc(ref[y0:y1, x0:x1], local, filt) < min_r:
            continue
        c = np.array([(y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2])
        p.append(c)
        q.append(g.A @ (c - np.asarray(d)) + g.t)
    p, q = np.array(p), np.array(q)
    keep = np.ones(len(p), bool)
    for _ in range(3):
        if keep.sum() < 6:
            return g
        A, t = _lsq(p[keep], q[keep], model)
        res = np.hypot(*(q - (p @ A.T + t)).T)
        new = res < max(3.0, 2.5 * 1.4826 * np.median(res[keep]) + 1e-6)
        if (new == keep).all():
            break
        keep = new
    A, t = _lsq(p[keep], q[keep], model)
    return Transform(A, t, model, int(keep.sum()))


def _initial(shape_ref, shape_mov, scale: float, theta_deg: float) -> Transform:
    """Reference pixel -> moving pixel about the two image centres: A0 = R(theta) * scale."""
    th = np.radians(theta_deg)
    A0 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) * scale
    c_ref, c_mov = (np.array(shape_ref) - 1) / 2, (np.array(shape_mov) - 1) / 2
    return Transform(A0, c_mov - A0 @ c_ref, "initial")


def _bin(a: np.ndarray, k: int) -> np.ndarray:
    ny, nx = (np.array(a.shape) // k) * k
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.asarray(a, float)[:ny, :nx].reshape(ny // k, k, nx // k, k), axis=(1, 3))


def _try(ref, mov, box, rotations, scales, filters, model, max_shift):
    best = (None, -2.0)
    for th in rotations:
        for s in scales:
            t0 = _initial(ref.shape, mov.shape, s, th)
            pre = t0.warp(mov)
            for f in filters:
                try:
                    t1 = fit_transform(ref, pre, box, f, model, max_shift)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                tr = t0.then(t1)
                r = overlap(ref, mov, tr, box)
                for _ in range(4 if model != "shift" else 0):  # refit the residual until no gain
                    try:
                        t2 = tr.then(fit_transform(ref, tr.warp(mov), box, f, model, max_shift))
                    except (ValueError, np.linalg.LinAlgError):
                        break
                    r2 = overlap(ref, mov, t2, box)
                    if r2 <= r + 1e-3:
                        break
                    tr, r = t2, r2
                if r > best[1]:
                    best = (tr, r)
    return best


def align(ref: np.ndarray, mov: np.ndarray, ref_roi: np.ndarray | None = None, mov_roi: np.ndarray | None = None,
          scale: float = 1.0, rotation: float = 0.0, search: str = "auto", min_r: float = 0.5,
          max_shift: float = 120.0, bin_to: int = 600) -> tuple[Transform, float]:
    """Transform taking the reference grid to the moving image, and its overlap correlation.

    ``scale``: moving px per reference px expected from the pixel sizes (um_ref / um_mov);
    ``rotation``: expected rotation (deg), e.g. a PEEM / sample azimuth change.
    search: "none"   the given rotation / scale, shift + similarity fit only;
            "auto"   that first; if the overlap r < ``min_r`` also +-30 deg (3 deg steps) and then,
                     still poor, the full circle with scales 0.75-1.33 (refocus, FOV change);
            "full"   always the full search.
    Images larger than ``bin_to`` px are binned for the search, then refined at full resolution."""
    ref = np.where(ref_roi, ref, np.nan) if ref_roi is not None else np.asarray(ref, float)
    mov = np.where(mov_roi, mov, np.nan) if mov_roi is not None else np.asarray(mov, float)
    roi = np.isfinite(ref) if ref_roi is None else ref_roi
    k = max(1, int(round(max(ref.shape) / bin_to)))
    rb, mb = (_bin(ref, k), _bin(mov, k)) if k > 1 else (ref, mov)
    box_b = central_box(_bin(roi.astype(float), k) > 0.99 if k > 1 else roi)
    ms = max_shift / k

    tr, r = (None, -2.0)
    if search in ("none", "auto"):
        tr, r = _try(rb, mb, box_b, (rotation,), (scale,), ("highpass", "smooth_edges"), "similarity", ms)
    if search == "full" or (search == "auto" and r < min_r):
        t2, r2 = _try(rb, mb, box_b, tuple(rotation + d for d in np.arange(-30, 31, 3.0)), (scale,),
                      ("smooth_edges",), "shift", ms)
        if t2 is not None and r2 < min_r:
            t3, r3 = _try(rb, mb, box_b, tuple(np.arange(0, 360, 5.0)),
                          tuple(scale * f for f in (0.75, 0.83, 0.91, 1.0, 1.1, 1.21, 1.33)),
                          ("smooth_edges",), "shift", 2 * ms)
            if t3 is not None and r3 > r2:
                t2, r2 = t3, r3
        if t2 is not None:  # fine pass around the found rotation / scale
            d = t2.describe(((rb.shape[0] - 1) / 2, (rb.shape[1] - 1) / 2))
            t2, r2 = _try(rb, mb, box_b, tuple(d["rotation_deg"] + np.arange(-2.5, 2.6, 0.5)),
                          tuple(d["scale"] * np.array([0.95, 0.975, 1.0, 1.025, 1.05])),
                          ("highpass", "smooth_edges"), "similarity", 2 * ms)
            if t2 is not None and r2 > r:
                tr, r = t2, r2
    if tr is None:
        raise ValueError("the two images could not be registered")
    if k > 1:  # back to full pixels (binned pixel i covers k*i ... k*i+k-1), refine at full resolution
        c = np.full(2, (k - 1) / 2)
        t0 = Transform(tr.A, k * tr.t + c - tr.A @ c, tr.model, tr.n_tiles)
        box = central_box(roi)
        try:
            t1 = fit_transform(ref, t0.warp(mov), box, "highpass", "similarity", 20)
            t1 = t0.then(t1)
            r0, r1 = overlap(ref, mov, t0, box), overlap(ref, mov, t1, box)
            tr, r = (t1, r1) if r1 >= r0 else (t0, r0)
        except (ValueError, np.linalg.LinAlgError):
            tr, r = t0, overlap(ref, mov, t0, box)
    return tr, float(r)
