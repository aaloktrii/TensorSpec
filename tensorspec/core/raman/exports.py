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


def write_cells_csv(path: str, mf, cells, lo: float, hi: float, names) -> str:
    """Spectra of the compared map cells in [lo, hi] (one column each) to ``path``, and their
    n-Lorentzian fits to <path>_fits.csv. ``cells``: {idx, fit}; ``names``: one label per cell.
    Returns the fits path."""
    m = (mf.axis >= lo) & (mf.axis <= hi)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["raman_shift_cm-1"] + list(names))
        for j, xv in enumerate(mf.axis[m]):
            w.writerow(["%.3f" % xv] + ["%.3f" % mf.data[it["idx"]][m][j] for it in cells])
    fpath = path[:-4] + "_fits.csv"
    with open(fpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["spectrum", "x_um", "y_um", "range_lo", "range_hi", "n_peaks", "peak",
                    "centre_cm-1", "fwhm_cm-1", "fwhm_err_cm-1", "height", "area",
                    "band_fwhm_cm-1", "r2", "bic"])
        for it, name in zip(cells, names):
            fr = it.get("fit")
            if fr is None or not fr[1]["ok"]:
                continue
            (flo, fhi, n), r = fr
            for k, pk in enumerate(r["peaks"]):
                w.writerow([name, "%.3f" % mf.x[it["idx"]],
                            "%.3f" % mf.y[it["idx"]], flo, fhi, n, k + 1,
                            "%.2f" % pk["center"], "%.2f" % pk["fwhm"],
                            "%.2f" % pk["err_fwhm"], "%.2f" % pk["height"],
                            "%.1f" % pk["area"], "%.2f" % r["band_fwhm"],
                            "%.5f" % r["r2"], "%.1f" % r["bic"]])
    return fpath

