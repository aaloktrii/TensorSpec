"""Files the Raman suite writes (no Qt), so every frontend writes the same ones.

Moved from the Peak fitting tab (fit_all's peak_fits.csv, _result_rows)."""
from __future__ import annotations

import csv
import os

import numpy as np


def _fmt(v, nd=3) -> str:
    return "" if v is None or not np.isfinite(v) else ("%.*f" % (nd, v))


def output_dir(path: str) -> str:
    """<folder>/output for a spectrum file or a folder of spectra (created)."""
    src = path.rstrip(os.sep)
    base = src if os.path.isdir(src) else os.path.dirname(src)
    out = os.path.join(base, "output")
    os.makedirs(out, exist_ok=True)
    return out


def write_peak_fits_csv(path: str, entries, results) -> str:
    """One row per fitted peak (or per band when nothing was fitted) of every spectrum."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["spectrum", "x", "y", "si_shift", "band", "n_peaks", "peak",
                    "centre", "centre_err", "fwhm", "fwhm_err", "height", "height_err",
                    "area", "area_err", "r2", "note"])
        for i, it in enumerate(entries):
            res = results[i]
            for name, b in res["bands"].items():
                rows = b.get("comps") or [None]
                for k, c in enumerate(rows):
                    w.writerow([it["label"], _fmt(it["x"]), _fmt(it["ypos"]),
                                _fmt(res["shift"]), name, b.get("n", 0), k + 1 if c else ""]
                               + (["%.3f" % c[q] for q in ("pos", "pos_err", "fwhm", "fwhm_err",
                                                            "height", "height_err", "area",
                                                            "area_err")] if c else [""] * 8)
                               + [_fmt(b.get("r2", np.nan)),
                                  b.get("why") or b.get("stopped", "")])
    return path


def peak_fit_rows(entries, results) -> list:
    """One row per fitted spectrum for /analysis/raman_peakfit: position, FWHM and height of each
    fitted peak per band."""
    rows = []
    for i, it in enumerate(entries):
        res = results.get(i)
        if res is None:
            continue
        row = {"spectrum": it["label"], "x": it["x"], "y": it["ypos"],
               "si_shift": res["shift"]}
        for name, b in res["bands"].items():
            row["%s_n_peaks" % name] = b.get("n", 0)
            for k, c in enumerate(b.get("comps") or []):
                for q in ("pos", "fwhm", "height"):
                    row["%s_%d_%s" % (name, k + 1, q)] = c[q]
        rows.append(row)
    return rows


def peak_counts(entries, results, names):
    """Peak count per band and spectrum (0 = no fit), and (x, y) when > 3 spectra have positions."""
    counts = [np.array([results[i]["bands"].get(name, {}).get("n", 0)
                        if results[i]["bands"].get(name, {}).get("ok") else 0
                        for i in range(len(entries))]) for name in names]
    has_xy = sum(np.isfinite(it["x"]) for it in entries) > 3
    xy = ((np.array([it["x"] for it in entries]), np.array([it["ypos"] for it in entries]))
          if has_xy else None)
    return counts, xy
