"""Stack clean-up before drift correction (ports of the SPEEM Igor stack tools), non-destructive.

The raw frames are never changed: the edits are a small dict kept with the dataset and
applied on the fly by ``apply_edits`` (order below). Igor equivalents in
speem_eval_20230427_TPX3.ipf:

  pixel_fixes   DisablePixel            listed pixels -> mean of the 4 diagonal neighbours at
                                        +-2 px (same rule as the dead-pixel map, value 2)
  topup         TopUpCorrection         frames taken during injection x factor (default 1.05);
                                        frames from the ring current jumps or a frame list
  normalize_to  NormalizeStackToThisLayer  every frame / that frame x RMS of that frame
  subtract      SubtractThisLayerFromStack  every frame - that frame
  flatten       FlattenSequence         every frame flattened along a line (image_analysis.flatten_line)
  bg_poly       backgroundremoval       row/column polynomial background of the first frame
                                        subtracted from every frame
  exclude       RemoveLayerFromStack /   frames dropped (with their polarization tags)
                RemoveDefocussedLayers  (defocused: std in the marquee below a threshold)

(``manualDrift`` acts on the drift-corrected frames and is handled by the service.)
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

EMPTY: dict[str, Any] = {"exclude": [], "pixel_fixes": [], "topup": None, "normalize_to": None, "subtract": None,
                         "flatten": None, "bg_poly": None}


def normalized(edits: dict | None) -> dict[str, Any]:
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in EMPTY.items()}
    for k, v in (edits or {}).items():
        if k in out:
            out[k] = v
    out["exclude"] = sorted({int(i) for i in out["exclude"] or []})
    out["pixel_fixes"] = [[int(y), int(x)] for y, x in (out["pixel_fixes"] or [])]
    return out


def is_empty(edits: dict | None) -> bool:
    e = normalized(edits)
    return not (e["exclude"] or e["pixel_fixes"] or e["topup"] or e["normalize_to"] is not None
                or e["subtract"] is not None or e["flatten"] or e["bg_poly"])


def parse_frames(text: str, n: int) -> list[int]:
    """'3, 7-9, 12' -> [3, 7, 8, 9, 12] (0-based frame indices, checked against n)."""
    out: list[int] = []
    for part in re.split(r"[,\s;]+", (text or "").strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            a, b = sorted((int(m.group(1)), int(m.group(2))))
            out += list(range(a, b + 1))
        elif part.isdigit():
            out.append(int(part))
        else:
            raise ValueError(f"cannot read frame list item {part!r} (use e.g. 3, 7-9)")
    bad = [i for i in out if not 0 <= i < n]
    if bad:
        raise ValueError(f"frames {bad} out of range 0..{n - 1}")
    return sorted(set(out))


def fix_pixels(img: np.ndarray, pixels: list[list[int]]) -> np.ndarray:
    """DisablePixel: replace each listed pixel by the mean of its diagonal neighbours at +-2 px."""
    a = np.asarray(img, float).copy()
    ny, nx = a.shape
    for y, x in pixels:
        nb = [a[yy, xx] for yy, xx in ((y - 2, x - 2), (y - 2, x + 2), (y + 2, x - 2), (y + 2, x + 2))
              if 0 <= yy < ny and 0 <= xx < nx]
        if nb:
            a[y, x] = float(np.mean(nb))
    return a


def frame_sharpness(frames: np.ndarray, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """RemoveDefocussedLayers metric: standard deviation of each frame inside the marquee."""
    st = np.asarray(frames)
    if box is not None:
        y0, y1, x0, x1 = (int(v) for v in box)
        st = st[:, y0:y1, x0:x1]
    return st.reshape(len(st), -1).std(axis=1).astype(float)


def injection_frames(current: np.ndarray, min_jump: float | None = None) -> list[int]:
    """Frames taken right after a top-up injection: ring current rises frame to frame by more
    than min_jump (default: 5 x the median |decay step|, at least 0.05 mA)."""
    c = np.asarray(current, float)
    if c.size < 2 or not np.isfinite(c).any():
        return []
    d = np.diff(c)
    if min_jump is None:
        dec = np.abs(d[d < 0])
        min_jump = max(0.05, 5 * float(np.median(dec)) if dec.size else 0.05)
    return [int(i + 1) for i in np.nonzero(d > min_jump)[0]]


def apply_edits(frames: np.ndarray, pol: list[str], edits: dict | None) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Edited copy of the stack. Returns (frames, pol, kept raw indices)."""
    e = normalized(edits)
    st = np.asarray(frames, dtype=np.float32)
    n = len(st)
    if is_empty(e):
        return st, list(pol), np.arange(n)
    st = st.astype(np.float64)
    if e["pixel_fixes"]:
        st = np.stack([fix_pixels(f, e["pixel_fixes"]) for f in st])
    tu = e["topup"]
    if tu:
        for i in tu.get("frames", []):
            if 0 <= int(i) < n:
                st[int(i)] *= float(tu.get("factor", 1.05))
    if e["normalize_to"] is not None:
        ref = st[int(e["normalize_to"])].copy()
        rms = float(np.sqrt(np.mean(ref ** 2)))
        with np.errstate(divide="ignore", invalid="ignore"):
            st = np.where(ref > 0, st / ref * rms, np.nan)
    if e["subtract"] is not None:
        st = st - st[int(e["subtract"])].copy()
    if e["flatten"]:
        from tensorspec.core.peem.image_analysis import flatten_line
        f = e["flatten"]
        st = np.stack([flatten_line(fr, tuple(f["a"]), tuple(f["b"]), int(f.get("halfwidth", 30)),
                                    bool(f.get("subtract_offset")), bool(f.get("points_only")))[0] for fr in st])
    if e["bg_poly"]:
        from tensorspec.core.peem.image_analysis import background_poly
        _, bg = background_poly(st[0], int(e["bg_poly"].get("order", 2)), int(e["bg_poly"].get("width", 5)))
        st = st - bg[None]
    keep = np.array([i for i in range(n) if i not in set(e["exclude"])], dtype=int)
    if keep.size == 0:
        raise ValueError("every frame is excluded")
    return st[keep].astype(np.float32), [pol[i] for i in keep], keep
