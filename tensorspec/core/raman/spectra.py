"""Spectrum collections (no Qt): building the lists of spectra the suite works on — files, a folder of
spot files, every spectrum of a map, a map's median — as plain entries any frontend can show.

An entry is {"label", "wn", "y", "path", "x", "ypos"} (x / ypos: stage position in µm, NaN if unknown).

Moved from the Peak fitting and Stacked spectra tabs (add_files, add_folder, use_map, add_paths,
add_map_median)."""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

from tensorspec.core.io.raman_loaders import (_SPOT_RE, Spectrum, is_single_spectrum, parse_map_file,
                                              read_single_spectrum, read_spot_folder, short_label)


def entry(label, wn, y, path, x=np.nan, ypos=np.nan) -> dict:
    return {"label": label, "wn": np.asarray(wn, float), "y": np.asarray(y, float),
            "path": path, "x": x, "ypos": ypos}


def from_files(paths) -> Tuple[List[dict], List[Tuple[str, str]]]:
    """Single-spectrum files → (entries, [(path, error), ...] for files that could not be read)."""
    out, bad = [], []
    for p in paths:
        try:
            s = read_single_spectrum(p)
            out.append(entry(s.label, s.wn, s.inten, p))
        except Exception as exc:
            bad.append((p, str(exc)))
    return out, bad


def from_folder(folder) -> List[dict]:
    """Every two-column spectrum in a folder; stage X/Y from __X_..__Y_.. in the file name."""
    out = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if not name.lower().endswith((".txt", ".dat", ".csv")) or not is_single_spectrum(p):
            continue
        s = read_single_spectrum(p)
        m = _SPOT_RE.search(name)
        out.append(entry(short_label(p), s.wn, s.inten, p,
                         float(m.group(1)) if m else np.nan, float(m.group(2)) if m else np.nan))
    return out


def from_map(mf) -> List[dict]:
    """Every spectrum of a map, labelled with its index and position."""
    return [entry("spot %d (%.1f, %.1f)" % (i, mf.x[i], mf.y[i]), mf.axis, mf.data[i],
                  mf.path, float(mf.x[i]), float(mf.y[i])) for i in range(mf.n_points)]


def is_whole_map(entries, mf) -> bool:
    """True when ``entries`` are exactly the spots of map ``mf`` (from_map)."""
    return (mf is not None and len(entries) == mf.n_points
            and all(e["path"] == mf.path for e in entries))


def map_median(mf, label=None) -> Spectrum:
    return Spectrum(mf.axis, np.nanmedian(mf.data, 0),
                    label or "%s median" % short_label(mf.path), mf.path)


def map_pixel(mf, i: int) -> Spectrum:
    return Spectrum(mf.axis, mf.data[i].astype(float),
                    "%s (%.1f, %.1f)" % (short_label(mf.path), mf.x[i], mf.y[i]), mf.path)


def spectrum_from_path(p) -> Spectrum:
    """For the stacked figure: a single spectrum, or the median of a map file / spot folder."""
    if os.path.isdir(p):
        mf = read_spot_folder(p)
        return Spectrum(mf.axis, np.nanmedian(mf.data, 0), short_label(p) + " median", p)
    if is_single_spectrum(p):
        return read_single_spectrum(p)
    mf = parse_map_file(p)
    if mf.layout != "spectra":
        raise ValueError("not a spectrum")
    return Spectrum(mf.axis, np.nanmedian(mf.data, 0), short_label(p) + " median", p)
