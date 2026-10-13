"""Raman batch runs (no Qt, no plotting): per-folder G/2D fits with CSV outputs, building the
1L/2L reference and predicting folders, and fitting every spectrum of a map with progress and
cancel callbacks (process pool for large maps).

Figures are not drawn here. Callers that want the original PDFs pass ``figures`` (the
plotting/backends/raman_matplotlib_engine module, or anything with fit_pdfs / plot_reference / plot_prediction).

Moved from Raman_Classifier: ``raman_peak_fit.py`` run(), ``graphene_reference.py`` build /
predict_folder / write_predictions, and the fitting loop of ``FitWorker`` in
``raman_map_classifier_8 (1).py``."""
from __future__ import annotations

import csv
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

from tensorspec.core.io.raman_loaders import SPECTRUM_EXT, MapFile, read_spectrum
from tensorspec.core.raman.classification import g_band_cfg
from tensorspec.core.raman.peak_fit import QUANTS, analyse_spectrum
from tensorspec.core.raman.reference import (MATCH, _with_log, _xy_from_name,
                                             predict_row, stats)
from tensorspec.core.raman.voigt_2d import _fit_chunk, band_mask, fit_band


def fit_folder(folder, a, figures=None):
    """Lorentzian G/2D fits of every spectrum in ``folder`` -> <out>/fits.csv, summary.csv
    (and summary.pdf, fits.pdf when ``figures`` is given). ``a`` carries ng, n2d, g_range,
    d_range, out. Returns (rows, out, summ, curves)."""
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(SPECTRUM_EXT))
    out = a.out or os.path.join(folder, "output")
    os.makedirs(out, exist_ok=True)
    rows, curves = [], []
    for f in files:
        try:
            wn, y = read_spectrum(os.path.join(folder, f))
        except Exception as exc:
            rows.append({"file": f, "ok": False, "note": "unreadable: %s" % exc})
            continue
        r, G, D = analyse_spectrum(wn, y, a.ng, a.n2d, a.g_range, a.d_range)
        rows.append({"file": f, **r})
        curves.append((f, G.get("curve"), D.get("curve"), r["ok"]))

    # ---- fits.csv
    keys = ["file", "ok", "note"]
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(out, "fits.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.6g" % v if isinstance(v, (float, np.floating)) else v)
                        for k, v in r.items()})

    # ---- summary.csv
    good = [r for r in rows if r.get("ok")]
    summ = []
    quants = list(QUANTS)
    if any(r.get("G_split") for r in good):
        quants += [("Gminus_pos", "G$^-$ position (cm$^{-1}$)"),
                   ("Gplus_pos", "G$^+$ position (cm$^{-1}$)")]
    for k in range(1, a.n2d + 1 if a.n2d > 1 else 1):
        quants += [("2D%d_pos" % k, "2D$_{%d}$ position (cm$^{-1}$)" % k),
                   ("2D%d_fwhm" % k, "2D$_{%d}$ FWHM (cm$^{-1}$)" % k)]
    for q, label in quants:
        v = np.array([r.get(q, np.nan) for r in good], float)
        v = v[np.isfinite(v)]
        summ.append((q, label, v))
    with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "mean", "std", "median", "n", "n_spectra", "n_failed"])
        for q, _, v in summ:
            w.writerow([q, "%.4g" % v.mean() if v.size else "", "%.4g" % v.std(ddof=1)
                        if v.size > 1 else "", "%.4g" % np.median(v) if v.size else "",
                        v.size, len(rows), len(rows) - len(good)])
    if figures is not None:
        figures.fit_pdfs(folder, rows, out, summ, curves)
    return rows, out, summ, curves


def build_reference(refs, out_json, figures=None):
    """refs: list of (label, folder). Fits each folder into
    <reference dir>/<folder name>/ and writes the JSON reference to ``out_json``
    (callers use reference.user_reference_path(workspace folder); the shipped
    reference.DEFAULT_REF is read-only)."""
    root = os.path.dirname(out_json)
    os.makedirs(root, exist_ok=True)
    args = SimpleNamespace(ng="auto", n2d=1, g_range=(1500, 1680),
                           d_range=(2560, 2810), out=None)
    data = {"created": time.strftime("%Y-%m-%d %H:%M"), "match_rule": MATCH,
            "datasets": []}
    for label, folder in refs:
        folder = os.path.expanduser(folder).rstrip(os.sep)
        name = os.path.basename(folder)
        args.out = os.path.join(root, "%s_%s" % (label, name.split("_532nm")[0]))
        rows, _, _, _ = fit_folder(folder, args, figures)
        good = _with_log([r for r in rows if r.get("ok")])
        lo, hi = MATCH[label]
        match = [r for r in good if lo <= r["2D_fwhm"] <= hi]
        pts = [{"file": r["file"], "2D_fwhm": r["2D_fwhm"], "I2D_IG": r["I2D_IG"],
                "2D_pos": r["2D_pos"], "match": bool(lo <= r["2D_fwhm"] <= hi)}
               for r in good]
        data["datasets"].append({
            "label": label, "name": name, "path": folder, "output": args.out,
            "n_spectra": len(rows), "n_fitted": len(good), "n_matching": len(match),
            "all": stats(good), "matching": stats(match), "points": pts})
        print("  %s %s: %d spectra, %d fitted, %d matching %s"
              % (label, name, len(rows), len(good), len(match), label))
    with open(out_json, "w") as fh:
        json.dump(data, fh, indent=1)
    if figures is not None:
        figures.plot_reference(data, os.path.join(root, "reference_summary.pdf"))
    print("reference ->", out_json)
    return data


def predict_folder(folder, ref, out=None, figures=None):
    folder = os.path.expanduser(folder).rstrip(os.sep)
    out = out or os.path.join(folder, "output")
    args = SimpleNamespace(ng="auto", n2d=1, g_range=(1500, 1680),
                           d_range=(2560, 2810), out=out)
    rows, _, _, _ = fit_folder(folder, args, figures)
    return write_predictions(rows, ref, out, os.path.basename(folder), figures)


def write_predictions(rows, ref, out, title, figures=None):
    os.makedirs(out, exist_ok=True)
    preds = []
    for r in rows:
        layer, poly, note, z1, z2 = predict_row(r, ref)
        x, y = r.get("x", np.nan), r.get("y", np.nan)
        if not np.isfinite(x):
            x, y = _xy_from_name(r.get("file", ""))
        preds.append({"file": r.get("file", ""), "x": x, "y": y, "layer": layer,
                      "polytype": poly, "z_1L": z1, "z_2L": z2,
                      "2D_fwhm": r.get("2D_fwhm", np.nan), "I2D_IG": r.get("I2D_IG", np.nan),
                      "2D_pos": r.get("2D_pos", np.nan), "G_pos": r.get("G_pos", np.nan),
                      "G_split": r.get("G_split", False), "note": note})
    with open(os.path.join(out, "predictions.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(preds[0]) if preds else ["file"])
        w.writeheader()
        for p in preds:
            w.writerow({k: ("%.5g" % v if isinstance(v, (float, np.floating)) else v)
                        for k, v in p.items()})
    if figures is not None:
        figures.plot_prediction(preds, ref, os.path.join(out, "prediction.pdf"), title)
    counts = {}
    for p in preds:
        key = p["layer"] + (" " + p["polytype"] if p["polytype"] not in ("", "—") else "")
        counts[key] = counts.get(key, 0) + 1
    print("%s: %s  ->  %s" % (title, ", ".join("%s %d" % kv for kv in sorted(counts.items())),
                              os.path.join(out, "predictions.csv")))
    return preds


# ---------------------------------------------------------------------------
# Map fitting (the GUI runs this off its main thread: gui/components/raman_panel.FitWorker)
# ---------------------------------------------------------------------------

class MapFitter:
    """Fits every spectrum of a map, across cores where it pays.

    progress(pct, msg) is called as spectra finish; cancel() stops at the next spectrum.
    run() returns {"twod": [...], "g": [...] or None}, or raises RuntimeError with the
    message the GUI shows ("Cancelled.", axis not covering the band, or a traceback)."""

    #: below this many spectra, process startup costs more than it saves
    PARALLEL_MIN = 1500
    CHUNK = 64

    def __init__(self, mapfile: MapFile, cfg: dict, progress=None):
        self.mapfile = mapfile
        self.cfg = dict(cfg)
        self._cancel = False
        self._progress = progress or (lambda pct, msg: None)

    def cancel(self):
        self._cancel = True

    def run(self):
        """Fit the 2D band, then the G band if the file contains one.

        Returns {"twod": [...], "g": [...] or None}.
        """
        try:
            mf = self.mapfile
            m = band_mask(mf.axis, self.cfg)
            if m.sum() < 10:
                raise RuntimeError(
                    "The axis of this file has fewer than 10 points between "
                    "%.0f and %.0f cm\u207b\u00b9."
                    % (self.cfg["band_min"], self.cfg["band_max"]))
            twod = self._fit_all(mf.axis[m], mf.data[:, m], self.cfg, "2D band")
            if twod is None:
                raise RuntimeError("Cancelled.")

            g = None
            gcfg = g_band_cfg(self.cfg)
            gm = band_mask(mf.axis, gcfg)
            if gm.sum() >= 10:
                g = self._fit_all(mf.axis[gm], mf.data[:, gm], gcfg, "G band")
                if g is None:
                    raise RuntimeError("Cancelled.")
            return {"twod": twod, "g": g}
        except RuntimeError:
            raise
        except Exception:
            raise RuntimeError(traceback.format_exc(limit=3))

    def _fit_all(self, x, block, cfg, label):
        n = block.shape[0]
        results = None
        if (cfg.get("parallel", True) and n >= self.PARALLEL_MIN
                and (os.cpu_count() or 1) > 1):
            results = self._run_parallel(x, block, n, cfg, label)
            if self._cancel:
                return None
        if results is None:
            results = self._run_serial(x, block, n, cfg, label)
        return results

    def _run_serial(self, x, block, n, cfg, label):
        results, step = [], max(1, n // 200)
        for i in range(n):
            if self._cancel:
                return None
            results.append(fit_band(x, block[i], cfg))
            if i % step == 0 or i == n - 1:
                self._progress(int(100 * (i + 1) / n),
                                   "Fitting %s: spectrum %d / %d" % (label, i + 1, n))
        return results

    def _run_parallel(self, x, block, n, cfg, label):
        """Returns None on failure so the caller can fall back to serial."""
        chunks = [(x, block[i:i + self.CHUNK], cfg)
                  for i in range(0, n, self.CHUNK)]
        try:
            results = []
            with ProcessPoolExecutor() as pool:
                for part in pool.map(_fit_chunk, chunks):
                    if self._cancel:
                        return []
                    results.extend(part)
                    self._progress(
                        int(100 * len(results) / n),
                        "Fitting %s: spectrum %d / %d (%d cores)"
                        % (label, len(results), n, os.cpu_count() or 1))
            return results if len(results) == n else None
        except Exception:
            return None
