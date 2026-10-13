"""Peak maps (no Qt): the step-1 fit model applied to every spectrum of a map, and the fitted
peaks laid onto the map grid — for each band the peak count, and for each peak its position,
FWHM, height and area.

Peaks are numbered per pixel from low to high wavenumber (tools.analyse_band sorts them), so with
an automatic peak count "peak 2" is the second peak in each pixel that has two or more; pixels
with fewer peaks, or whose band fit failed, are NaN (count 0)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from tensorspec.core.raman.fit_model import fit_spectrum
from tensorspec.core.raman.strain_doping import single_threaded_workers

QUANTITIES = ("pos", "fwhm", "height", "area")
UNITS = {"pos": "cm-1", "fwhm": "cm-1", "height": "counts", "area": "counts cm-1"}
CHUNK = 8


def _fit_chunk(args):
    axis, block, model = args
    return [fit_spectrum(axis, y, model) for y in block]


def fit_pixels(mapfile, model, progress=None, workers=None, parallel=True):
    """fit_spectrum of every spectrum of ``mapfile`` (index order). progress(done, total) is called
    as chunks finish; returning False cancels (the function then returns None)."""
    n = mapfile.n_points
    chunks = [(mapfile.axis, mapfile.data[i:i + CHUNK], model) for i in range(0, n, CHUNK)]
    out = [None] * len(chunks)
    done = 0
    if not parallel:
        for k, ch in enumerate(chunks):
            out[k] = _fit_chunk(ch)
            done += len(ch[1])
            if progress and progress(done, n) is False:
                return None
    else:
        with single_threaded_workers(), ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_fit_chunk, ch): k for k, ch in enumerate(chunks)}
            for f in as_completed(futs):
                out[futs[f]] = f.result()
                done += len(chunks[futs[f]][1])
                if progress and progress(done, n) is False:
                    ex.shutdown(cancel_futures=True)
                    return None
    return [r for part in out for r in part]


def peak_maps(mapfile, results, model) -> dict:
    """{key: (ny, nx) grid}. Keys: '<band>_n' (peak count, 0 = no fit) and
    '<band>_p<k>_<pos|fwhm|height|area>' for k = 1 .. the most peaks found in that band."""
    ny, nx = mapfile.shape
    ok_px = mapfile.index_map >= 0
    maps = {}
    for name, *_ in model.bands:
        bands = [r["bands"].get(name, {}) if r is not None else {} for r in results]
        counts = np.array([b.get("n", 0) if b.get("ok") else 0 for b in bands])
        g = np.zeros((ny, nx))
        g[~ok_px] = np.nan
        g[ok_px] = counts[mapfile.index_map[ok_px]]
        maps["%s_n" % name] = g
        for k in range(int(counts.max()) if counts.size else 0):
            for q in QUANTITIES:
                v = np.array([b["comps"][k][q] if b.get("ok") and len(b.get("comps") or []) > k
                              else np.nan for b in bands])
                g = np.full((ny, nx), np.nan)
                g[ok_px] = v[mapfile.index_map[ok_px]]
                maps["%s_p%d_%s" % (name, k + 1, q)] = g
    return maps
