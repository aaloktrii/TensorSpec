"""Layer regions from optical contrast (no Qt): segment the map into terraces by the camera image's
contrast, turn each region's entry (a layer count, or an AFM step height to bare SiO2) into a layer
number, check the assignment, and save / load it.

Moved from gui/components/raman_tabs/layer_regions_dialog.py (segment, row_n, _n_by_spec,
update_status checks, save_regions, load_regions)."""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

import numpy as np

from tensorspec.core.raman.classification import terrace_edges
from tensorspec.core.raman.constants import LAYER_NM
from tensorspec.core.raman.map_geometry import kmeans_1d, optical_contrast


def segment(mf, optical, fits, cut: float, k: int, first_layer: int) -> dict:
    """Contrast of every spectrum against the substrate, k contrast levels (R1 = faintest =
    thinnest), the region of every spectrum (-1 = substrate), and a default entry per region:
    ("N", n) for regions that look like graphene terraces (contrast of the flake's sign and >= 20 %
    fitted 2D bands), counting up from ``first_layer``; ("N", 0) = unassigned otherwise.
    {"contrast", "levels" (None if too few pixels), "region_of", "entries"}."""
    contrast, _ = optical_contrast(mf, optical)
    region_of = np.full(mf.n_points, -1, dtype=int)
    on = np.isfinite(contrast) & (np.abs(contrast) >= cut)
    if on.sum() < k:
        return {"contrast": contrast, "levels": None, "region_of": region_of, "entries": []}
    c = kmeans_1d(contrast[on], k)
    c = c[np.argsort(np.abs(c))]          # R1 = faintest = thinnest
    idx = np.where(on)[0]
    region_of[idx] = np.argmin(np.abs(contrast[idx, None] - c[None, :]), axis=1)
    # Defaults: a region with almost no 2D-band fits, or contrast opposite
    # to the flake's, is not a graphene terrace and starts unassigned.
    fit_ok = np.array([bool(r.get("ok")) for r in fits])
    n_pos = int(((contrast >= cut) & fit_ok).sum())
    n_neg = int(((contrast <= -cut) & fit_ok).sum())
    sign = 1.0 if n_pos >= n_neg else -1.0
    nxt = int(first_layer)
    entries = []
    for j, cj in enumerate(c):
        m = region_of == j
        frac = float(fit_ok[m].mean()) if m.any() else 0.0
        if np.sign(cj) == sign and frac >= 0.2:
            entries.append(("N", float(min(9, nxt))))
            nxt += 1
        else:
            entries.append(("N", 0.0))
    return {"contrast": contrast, "levels": c, "region_of": region_of, "entries": entries}


def region_n(is_height: bool, value: float, sio2_offset: float) -> Tuple[Optional[float], int, str]:
    """(unrounded N or None, N used (0 = unassigned), note) for one region's entry: a layer count,
    or an AFM step height (nm) to bare SiO2, converted with the SiO2 offset."""
    if not is_height:
        n = int(round(value))
        return (float(n) if n > 0 else None), n, ""
    if value <= 0:
        return None, 0, ""
    raw = (value - sio2_offset) / LAYER_NM
    n = int(round(raw))
    note = ""
    if n < 1:
        note = "height is below the offset"
        n = 0
    elif abs(raw - n) > 0.25:
        note = "%.2f is not close to a whole number of layers" % raw
    return raw, n, note


def n_by_spectrum(region_of, n_per_region) -> np.ndarray:
    """Layer count of every spectrum from its region's N (0 = unassigned)."""
    n = np.zeros(len(region_of), dtype=int)
    for j, nj in enumerate(n_per_region):
        n[region_of == j] = max(nj, 0)
    return n


def check(mf, levels, region_of, rows, table_layers, exclude_edges: bool) -> Tuple[List[str], List[str], bool]:
    """Warnings and notes on an assignment. ``rows``: (raw, n, note) per region; ``table_layers``:
    the layer counts with reference widths. Returns (warnings, infos, anything assigned)."""
    warns, infos, assigned = [], [], []
    for j, (raw, n, note) in enumerate(rows):
        if n > 0:
            assigned.append((abs(levels[j]) if levels is not None else 0, j, n))
            if n not in table_layers and n not in (1, 2):
                warns.append("R%d: no reference widths for %dL in the table." % (j + 1, n))
        if note:
            warns.append("R%d: %s — check the SiO₂ offset or the line "
                         "profile." % (j + 1, note))
    assigned.sort()
    for (ca, ja, na), (cb, jb, nb) in zip(assigned[:-1], assigned[1:]):
        if nb < na:
            warns.append("R%d has stronger contrast than R%d but fewer layers "
                         "(%d vs %d) — check the values or which terrace is "
                         "which." % (jb + 1, ja + 1, nb, na))
        elif nb == na:
            infos.append("R%d and R%d are both %dL and will be treated as one "
                         "terrace." % (ja + 1, jb + 1, na))
    nb = n_by_spectrum(region_of, [n for _, n, _ in rows])
    g = np.zeros(mf.shape, dtype=int)
    ok = mf.index_map >= 0
    g[ok] = nb[mf.index_map[ok]]
    ne = int(terrace_edges(g).sum())
    infos.append("%d pixels assigned; %d step-edge pixels %s."
                 % (int((nb > 0).sum()), ne, "excluded" if exclude_edges else "kept"))
    return warns, infos, bool(assigned)


def save(path: str, mf, sio2_offset: float, cut: float, levels, entries, region_of) -> str:
    """entries: [(kind 'N' | 'height', value), ...] per region."""
    data = {"sio2_offset": sio2_offset, "cut": cut,
            "levels": [float(v) for v in (levels if levels is not None else [])],
            "regions": [{"entry": kind, "value": value} for kind, value in entries],
            "spectra": [[round(float(mf.raw0[i]), 4), round(float(mf.raw1[i]), 4),
                         int(region_of[i])] for i in range(mf.n_points)]}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return path


def load(path: str, mf) -> Optional[dict]:
    """{"region_of", "sio2_offset", "levels" (or None), "entries"}, or None when no saved position
    matches this map (saved for a different acquisition)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    key = {(round(float(a), 3), round(float(b), 3)): i
           for i, (a, b) in enumerate(zip(mf.raw0, mf.raw1))}
    region_of = np.full(mf.n_points, -1, dtype=int)
    hit = 0
    for a, b, r in data.get("spectra", []):
        i = key.get((round(a, 3), round(b, 3)))
        if i is not None:
            region_of[i] = int(r)
            hit += 1
    if hit == 0:
        return None
    lv = np.asarray(data.get("levels", []), dtype=float)
    return {"region_of": region_of, "sio2_offset": float(data.get("sio2_offset", 0.5)),
            "levels": lv if lv.size else None,
            "entries": [(r["entry"], float(r["value"])) for r in data.get("regions", [])]}
