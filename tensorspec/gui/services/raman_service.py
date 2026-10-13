"""Raman suite service (no Qt) — the backend API of the Raman suite.

One RamanService is one working session: it holds all state (step-1 spectra and their fits, the fit
model, the stacked-figure list, the loaded map, map area and map results, the analysis config) and
offers one method per user action, routing the science through RamanEngineRouter and keeping the
global workspace DataTree current (/raw on load, /analysis/raman_* after each step; /history is
appended by DataTreeBuilder). Long jobs take progress(done, total) -> False to cancel.

Any frontend (the Qt tabs in gui/components/raman_tabs/, a script, a notebook, a web app) drives the
suite through this class only and draws with plotting/; tests/test_raman_service_headless.py runs the
whole workflow through it without Qt.

The state and the non-Qt halves of MainWindow's methods (open_raman, _on_fits, rebuild_maps,
layer_grid, current/set_transform, _coverage_note, save/load_config, suggest_name) moved here from
Raman_Classifier ``raman_map_classifier_8 (1).py``; the dialogs and widget updates stay in
gui/components/raman_panel.py."""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np

from tensorspec.core.io import raman_loaders
from tensorspec.core.io.raman_loaders import MapFile
from tensorspec.core.raman import exports, presets, products, reference
from tensorspec.core.raman import spectra as spectra_lists
from tensorspec.core.raman.fit_model import FitModel
from tensorspec.core.raman.classification import (g_band_cfg, label_of, layer_from_ratio, pixel_ratio,
                                                  ratio_applicable)
from tensorspec.core.raman.constants import default_config
from tensorspec.core.raman.map_geometry import (apply_map_transform, auto_align, band_height,
                                                default_transform, describe_transform)
from tensorspec.core.raman.voigt_2d import asym_of
from tensorspec.core.raman_engine import RamanEngineRouter
from tensorspec.core.workspace import global_workspace


def workspace_name(path: str) -> str:
    """A workspace key for a loaded file or folder: 'raman_' + its name, safe characters only."""
    stem = os.path.splitext(os.path.basename(path.rstrip(os.sep)))[0]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.-")[:56] or "map"
    return "raman_" + stem


class RamanService:
    """Shared state of the Raman tabs (was MainWindow's attributes) and its non-GUI logic."""

    def __init__(self):
        self.engine = RamanEngineRouter()
        self.cfg = default_config()
        self.mapfile: Optional[MapFile] = None
        self.mapfile_raw: Optional[MapFile] = None
        self.layer_by_spec: Optional[np.ndarray] = None   # N per spectrum
        self.optical: Optional[MapFile] = None
        self.optical_image: Optional[np.ndarray] = None
        self.fits: List[dict] = []
        self.g_fits: Optional[List[dict]] = None
        self.ratio_maps: Dict[str, Optional[np.ndarray]] = {"height": None,
                                                            "area": None}
        self.ratio_ok: bool = False
        self.ratio_note: str = "no file loaded"
        self.fit_tally: Dict[str, int] = {}
        self.fwhm_map: Optional[np.ndarray] = None
        self.asym_maps: Dict[str, Optional[np.ndarray]] = {
            "hwhm": None, "area": None, "residual": None}
        self.name: Optional[str] = None                   # workspace key of the loaded map
        self.band_maps = {}                               # key -> (lo, hi, grid, mapfile it is on)
        # step 1
        self.spectra: List[dict] = []                     # entries of core/raman/spectra.py
        self.spectra_fits: Dict[int, dict] = {}           # entry index -> fit_spectrum result
        self.fit_model = FitModel()                       # the step-1 model every fit uses
        self.stack: List[dict] = []                       # {"spec": Spectrum, "color", "visible"}
        self.preset_windows: Dict[str, tuple] = {}        # extra band-integral windows of a preset
        self.strain_rows: Optional[list] = None
        # 2 · Map ▸ Compare cells
        self.cells: List[dict] = []                       # {"idx", "color", "visible", "fit"}
        self._band_written = None                         # band map shown as /analysis data
        self.clear_peak_maps()
        self.clear_area()

    # =========================================================================================
    # 1 · Spectra
    # =========================================================================================
    def set_fit_model(self, model: FitModel):
        """The fit model of 1 · Spectra; areas and pixel fits of 2 · Map use it too."""
        self.fit_model = model

    def preset(self, name: str) -> dict:
        """Apply a material preset (core/raman/presets.py): fit model bands + Si calibration and
        band-integral windows. Returns the preset, for the frontend to show its other settings."""
        p = presets.PRESETS[name]
        self.fit_model = FitModel(bands=p["bands"], background=self.fit_model.background,
                                  bg_opts=self.fit_model.bg_opts, si_calibrate=p["si_calibrate"],
                                  centre_tol=self.fit_model.centre_tol)
        self.preset_windows = dict(p.get("windows", {}))
        return p

    def add_spectra(self, entries) -> int:
        self.spectra.extend(entries)
        return len(entries)

    def add_spectrum(self, label, wn, y, path, x=np.nan, ypos=np.nan) -> int:
        return self.add_spectra([spectra_lists.entry(label, wn, y, path, x, ypos)])

    def add_spectra_files(self, paths):
        """Returns (number added, [(path, error), ...])."""
        entries, bad = spectra_lists.from_files(paths)
        return self.add_spectra(entries), bad

    def add_spectra_folder(self, folder) -> int:
        return self.add_spectra(spectra_lists.from_folder(folder))

    def add_map_spectra(self) -> int:
        """Every spectrum of the loaded map (0 if none is loaded)."""
        return self.add_spectra(spectra_lists.from_map(self.mapfile)) if self.mapfile else 0

    def clear_spectra(self):
        self.spectra, self.spectra_fits = [], {}

    def spectra_are_map(self) -> bool:
        """True when the step-1 list is exactly the loaded map's spectra."""
        return spectra_lists.is_whole_map(self.spectra, self.mapfile)

    def fit_spectrum(self, i: int) -> dict:
        it = self.spectra[i]
        res = self.engine.fit_spectrum(it["wn"], it["y"], self.fit_model)
        self.spectra_fits[i] = res
        return res

    def fit_all_spectra(self, progress=None) -> Optional[str]:
        """Fit every step-1 spectrum, write <folder>/output/peak_fits.csv (and /analysis/raman_peakfit
        when the list is the loaded map). Returns the CSV path, or None if cancelled."""
        n = len(self.spectra)
        for i in range(n):
            if progress and progress(i, n) is False:
                return None
            self.fit_spectrum(i)
        if progress:
            progress(n, n)
        path = exports.write_peak_fits_csv(
            os.path.join(exports.output_dir(self.spectra[0]["path"]), "peak_fits.csv"),
            self.spectra, self.spectra_fits)
        if self.spectra_are_map():
            self.write_table("raman_peakfit", exports.peak_fit_rows(self.spectra, self.spectra_fits),
                             "si_shift", {"background": self.fit_model.background})
        return path

    def peak_counts(self):
        """Peak count per band of the step-1 fits, and (x, y) of the spectra when known."""
        names = [b[0] for b in self.fit_model.bands]
        return names, *exports.peak_counts(self.spectra, self.spectra_fits, names)

    def strain_doping(self, ref, progress=None):
        """omega_G - omega_2D correlation rows of the step-1 spectra against the 1L/2L reference
        (None if cancelled)."""
        rows = self.engine.strain_doping_many([(e["wn"], e["y"]) for e in self.spectra], ref,
                                              progress=progress)
        if rows is not None:
            self.strain_rows = rows
        return rows

    def strain_doping_params(self, params: dict):
        """Grid parameters chosen for the correlation; stored with the map's results."""
        if self.strain_rows is not None and self.spectra_are_map():
            self.write_strain_doping(self.strain_rows, params)

    # -- 2 · Map ▸ Compare cells ----------------------------------------------------------------
    CELL_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
                   "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
    MAX_CELLS = 12

    def cells_add(self, idx: int) -> str:
        """Add a map cell (spectrum index). Returns '' or why it was not added."""
        if any(it["idx"] == idx for it in self.cells):
            return "already there"
        if len(self.cells) >= self.MAX_CELLS:
            return "The comparison holds at most %d spectra." % self.MAX_CELLS
        used = {it["color"] for it in self.cells}
        color = next((c for c in self.CELL_COLORS if c not in used), self.CELL_COLORS[0])
        self.cells.append({"idx": int(idx), "color": color, "visible": True, "fit": None})
        return ""

    def cells_toggle(self, idx: int) -> bool:
        """Remove the cell if it is in the comparison, else add it. True when it was added."""
        for k, it in enumerate(self.cells):
            if it["idx"] == idx:
                self.cells.pop(k)
                return False
        return self.cells_add(idx) == ""

    def cells_remove(self, rows):
        for r in sorted(rows, reverse=True):
            self.cells.pop(r)

    def cells_resolution(self) -> float:
        mf = self.mapfile
        return float((mf.spectral_resolution if mf is not None else None)
                     or self.cfg.get("min_fwhm", 5.0))

    def cell_fit(self, it, lo: float, hi: float, n: int):
        """n Lorentzians on one compared cell over [lo, hi]; kept with the cell."""
        mf = self.mapfile
        m = (mf.axis >= lo) & (mf.axis <= hi)
        it["fit"] = ((lo, hi, n), self.engine.fit_band("lorentz_n", mf.axis[m], mf.data[it["idx"]][m],
                                                      {"n": n}))

    def cells_fit(self, lo: float, hi: float, n: int):
        for it in self.cells:
            if it["visible"]:
                self.cell_fit(it, lo, hi, n)

    def cells_suggest(self, lo: float, hi: float):
        """Peak count the data justify for each visible cell, and the most common one:
        ([(cell, n), ...], best) or ([], None)."""
        mf = self.mapfile
        m = (mf.axis >= lo) & (mf.axis <= hi)
        res = self.cells_resolution()
        picks = [(it, self.engine.suggest_peak_count(mf.axis[m], mf.data[it["idx"]][m], res)[0])
                 for it in self.cells if it["visible"]]
        if not picks:
            return [], None
        counts = [n for _, n in picks]
        return picks, max(set(counts), key=counts.count)

    def cells_export(self, path: str, lo: float, hi: float, names) -> str:
        """Visible cells' spectra and fits to CSV (names: one label per visible cell)."""
        vis = [it for it in self.cells if it["visible"]]
        return exports.write_cells_csv(path, self.mapfile, vis, lo, hi, names)

    # -- stacked figure ------------------------------------------------------------------------
    STACK_COLORS = ["#000000", "#e8402a", "#1f5fa8", "#2a8c4a", "#8a5a00",
                    "#7b3fa0", "#17becf", "#7f7f7f"]

    def stack_add(self, spec):
        used = {it["color"] for it in self.stack}
        color = next((c for c in self.STACK_COLORS if c not in used), self.STACK_COLORS[0])
        self.stack.append({"spec": spec, "color": color, "visible": True})

    def stack_add_paths(self, paths):
        """Single spectra, or the median of map files / spot folders. Returns ['name: error', ...]."""
        bad = []
        for p in paths:
            try:
                self.stack_add(spectra_lists.spectrum_from_path(p))
            except Exception as exc:
                bad.append("%s: %s" % (os.path.basename(p), exc))
        return bad

    def stack_add_map_pixel(self, i: int):
        self.stack_add(spectra_lists.map_pixel(self.mapfile, i))

    def stack_add_map_median(self):
        self.stack_add(spectra_lists.map_median(self.mapfile))

    def stack_remove(self, rows):
        for r in sorted(rows, reverse=True):
            self.stack.pop(r)

    def stack_move(self, r: int, d: int) -> bool:
        t = r + d
        if not (0 <= t < len(self.stack)):
            return False
        self.stack[r], self.stack[t] = self.stack[t], self.stack[r]
        return True

    def stack_traces(self):
        return [{"wn": it["spec"].wn, "inten": it["spec"].inten,
                 "label": it["spec"].label, "color": it["color"]}
                for it in self.stack if it["visible"]]

    # =========================================================================================
    # 2 · Map  (and the state shared with 3 · Analysis)
    # =========================================================================================
    # -- naming --------------------------------------------------------------------------------
    def suggest_name(self, suffix: str) -> str:
        if not self.mapfile:
            return "map" + suffix
        base = os.path.splitext(self.mapfile.path)[0]
        return base + suffix

    # -- 1L / 2L reference files --------------------------------------------------------------
    @staticmethod
    def user_reference_path() -> str:
        """Where 'Build reference' writes: <workspace folder>/raman/reference/graphene_reference.json."""
        return reference.user_reference_path(global_workspace.project_dir)

    @staticmethod
    def preferred_reference() -> str:
        """The user-built reference if there is one, otherwise the shipped default."""
        return reference.preferred_reference(global_workspace.project_dir)

    # -- loading -------------------------------------------------------------------------------
    def load(self, path: str) -> dict:
        """Open any Raman file or spot folder. Returns {"kind": "spectrum" | "map" | "image",
        "obj": Spectrum / MapFile, "warnings": [...], "resolution": header cutoff or None}.
        A map becomes the current map (fits cleared, /raw written), a camera image the optical
        image; a single spectrum is only read (the frontend decides where it goes)."""
        kind, obj = raman_loaders.load_raman(path)
        out = {"kind": kind, "obj": obj, "warnings": [], "resolution": None}
        if kind == "image":
            self.set_optical(obj)
        elif kind == "map":
            if obj.axis.max() < self.cfg["band_min"] or obj.axis.min() > self.cfg["band_max"]:
                out["warnings"].append(
                    "The axis of this file spans %.0f–%.0f, which does not cover the "
                    "2D-band window %.0f–%.0f cm⁻¹.\n\nChange the window under "
                    "Analysis ▸ Fit parameters."
                    % (obj.axis.min(), obj.axis.max(), self.cfg["band_min"], self.cfg["band_max"]))
            out["resolution"] = self.set_map(obj)
        return out

    def set_map(self, mf: MapFile) -> Optional[float]:
        """Make ``mf`` the current map (fits cleared) and push it to the workspace (/raw).
        Returns the spectral resolution read from the header, if any."""
        # Honour the instrument's own resolution as the narrow-peak cutoff.
        res = mf.spectral_resolution
        if res and res > 0:
            self.cfg["min_fwhm"] = float(res)
        self.mapfile_raw = mf
        self.mapfile = apply_map_transform(mf, self.current_transform())
        self.layer_by_spec = np.zeros(mf.n_points, dtype=int)
        if self.cfg.get("layer_mode") == "map":
            self.cfg["layer_mode"] = "manual"          # a new map has no layer map
        self.ratio_ok, self.ratio_note = ratio_applicable(mf.axis, self.cfg)
        self.g_fits = None
        self.ratio_maps = {"height": None, "area": None}
        self.fits = []
        self.fwhm_map = None
        self.cells = []
        self.clear_area()
        self.band_maps = {}
        self._band_written = None
        self.clear_peak_maps()
        self.name = workspace_name(mf.path)
        global_workspace.push_spectroscopy_data(self.name, raman_loaders.map_to_tensor(mf))
        return res

    def load_optical(self, path: str) -> MapFile:
        """Read a camera image (or image-like map) and make it the optical image."""
        opt = raman_loaders.parse_map_file(path)
        self.set_optical(opt)
        return opt

    def set_optical(self, opt: MapFile):
        self.optical = opt
        self.optical_image = opt.greyscale()
        global_workspace.push_spectroscopy_data(workspace_name(opt.path) + "_optical",
                                                raman_loaders.map_to_tensor(opt))

    def coverage_note(self, opt: MapFile) -> str:
        """Flag a Raman map that falls outside the camera frame.

        Measured against the actual measurement positions, not the displayed
        pixel extents, which are padded by half a step and would make any
        coarse map look like it overflows. A small overhang is just a map
        drawn to the edge of the frame; a large one usually means the two
        coordinate columns were read the wrong way round.
        """
        mf = self.mapfile
        if mf is None or mf.layout != "spectra":
            return ""
        ol, orr, ob, ot = opt.extent
        over = max(ol - mf.x.min(), mf.x.max() - orr,
                   ob - mf.y.min(), mf.y.max() - ot)
        if over <= 1.0:
            return ""
        span = max(mf.x.max() - mf.x.min(), mf.y.max() - mf.y.min(), 1.0)
        if over > 0.1 * span:
            return ("  ⚠ the map sits %.1f µm outside the camera frame — check "
                    "that the X and Y columns were read the right way round"
                    % over)
        return "  (the map reaches %.1f µm past the camera frame edge)" % over

    # -- what a map view shows ----------------------------------------------------------------
    def height_map(self, which: str) -> np.ndarray:
        """Raw G or 2D peak height on the current grid (no fitting)."""
        mf = self.mapfile
        if which == "G height":
            v = band_height(mf.axis, mf.data, 1582.0, 22.0)
        else:
            c = 0.5 * (self.cfg["band_min"] + self.cfg["band_max"])
            v = band_height(mf.axis, mf.data, c, 45.0)
        g = np.full(mf.shape, np.nan)
        ok = mf.index_map >= 0
        g[ok] = v[mf.index_map[ok]]
        return g

    def quantity_map(self, quantity: str, window=None, peak_key=None):
        """The grid a map view shows: 'Total intensity', 'Band integral' (window = (name, lo, hi)),
        'Peak map' (peak_key of peak_grids), 'FWHM(2D)', 'G height', '2D height'.
        None when it needs a fit that has not been run."""
        mf = self.mapfile
        if quantity == "Total intensity":
            return self.band_integral_map("total", float(mf.axis.min()), float(mf.axis.max()))
        if quantity == "Band integral":
            return self.band_integral_map(*window)
        if quantity == "Peak map":
            return self.peak_map(peak_key) if peak_key in self.peak_grids else None
        if quantity == "FWHM(2D)":
            return self.fwhm_map
        return self.height_map(quantity)

    def pixel(self, row: int, col: int) -> Optional[dict]:
        """Everything known about one map pixel: its spectrum, the 2D-band fit (curve, centre,
        FWHM, asymmetry, SNR) and, with a G fit, I(2D)/I(G) and the layer number it implies.
        None if the pixel holds no spectrum."""
        mf = self.mapfile
        idx = mf.index_map[row, col]
        if idx < 0:
            return None
        wn, inten = mf.axis, mf.data[idx]
        lo, hi = self.cfg["band_min"], self.cfg["band_max"]
        res = self.fits[idx] if self.fits else None
        out = {"index": int(idx), "x": float(mf.x[idx]), "y": float(mf.y[idx]), "wn": wn,
               "inten": inten, "band": (lo, hi), "fit": res, "fit_curve": None, "center": None,
               "asym": None, "g_fit": None, "g_curve": None, "ratio": None, "layer": None}
        if res and np.isfinite(res.get("center", np.nan)):
            m = (wn >= lo) & (wn <= hi)
            if m.sum() > 3:
                out["fit_curve"] = (wn[m], self.engine.rebuild_fit(wn[m], res))
            out["center"] = res["center"]
            out["asym"] = asym_of(res, self.cfg["asym_metric"])
        gf = self.g_fits[idx] if self.g_fits else None
        out["g_fit"] = gf
        if gf is not None and res:
            r = pixel_ratio(res, gf, self.cfg.get("ratio_metric", "height"))
            n = layer_from_ratio(r, self.cfg)
            out["ratio"], out["layer"] = r, (label_of(n, "N/A") if n is not None else None)
            if gf.get("ok") and np.isfinite(gf.get("center", np.nan)):
                gcfg = g_band_cfg(self.cfg)
                gm = (wn >= gcfg["band_min"]) & (wn <= gcfg["band_max"])
                out["g_curve"] = (wn[gm], self.engine.rebuild_fit(wn[gm], gf))
        return out

    def auto_align(self, all_orientations: bool, sign=None) -> Optional[dict]:
        """Fit the map's placement to the optical image and apply it. Returns {"transform", "r",
        "signal", "verdict", "runner" (next-best candidate or None), "at_edge", "search_um",
        "all_orientations"}, or None when nothing matched."""
        t0 = self.current_transform()
        rng = float(self.cfg.get("align_search_um", 15.0))
        orient = None if all_orientations else [(t0["swap"], t0["flip_x"], t0["flip_y"])]
        cands, sig = auto_align(self.mapfile_raw, self.optical, self.cfg,
                                search_um=rng, orientations=orient, sign=sign)
        cands = [c for c in cands if c[0] > 0]
        if not cands:
            return None
        best = cands[0]
        t = best[2]
        self.apply_transform(t)
        out = {"transform": t, "r": best[1], "signal": sig, "runner": None, "search_um": rng,
               "at_edge": max(abs(t["dx"]), abs(t["dy"])) >= rng - 0.5,
               "all_orientations": all_orientations}
        if all_orientations:
            runner = next((c for c in cands[1:]
                           if (c[2]["swap"], c[2]["flip_x"], c[2]["flip_y"]) !=
                           (t["swap"], t["flip_x"], t["flip_y"])), None)
            margin = best[0] - (runner[0] if runner else 0.0)
            out["verdict"] = ("clear" if best[0] >= 0.5 and margin >= 0.2 else
                              "tentative — check it by eye, or against LabSpec's overlay")
            out["runner"] = (runner[2], runner[1]) if runner else None
        else:
            out["verdict"] = ("offset found" if best[0] >= 0.3 else "weak match — check it by eye")
        return out

    @staticmethod
    def describe_transform(t) -> str:
        return describe_transform(t)

    # -- fitting -------------------------------------------------------------------------------
    def map_fitter(self, progress=None):
        """A MapFitter (Voigt fits of every spectrum) for the current map; run it off the GUI thread."""
        return self.engine.map_fitter(self.mapfile, self.cfg, progress)

    def set_fits(self, payload) -> int:
        """Store the map fits, rebuild the maps and write /analysis/raman_2d_fit.
        Returns the number of spectra whose fit passed."""
        if isinstance(payload, dict):
            results, gres = payload["twod"], payload.get("g")
        else:                                   # older callers
            results, gres = payload, None
        self.fits = results
        self.g_fits = gres
        tally: Dict[str, int] = {}
        for res in results:
            if not res["ok"]:
                key = res.get("why") or "rejected"
                tally[key] = tally.get(key, 0) + 1
        self.fit_tally = tally
        ok = self.rebuild_maps()
        self.write_twod_fit()
        return ok

    def rebuild_maps(self) -> int:
        """Lay the per-spectrum fits onto the current grid. No refitting.

        Called after fitting and again whenever the orientation changes,
        since a new orientation is a new grid over the same spectra.
        """
        mf = self.mapfile
        if mf is None or not self.fits:
            self.fwhm_map = None
            return 0
        results, gres = self.fits, self.g_fits
        ny, nx = mf.shape
        self.fwhm_map = np.full((ny, nx), np.nan)
        self.asym_maps = {k: np.full((ny, nx), np.nan)
                          for k in ("hwhm", "area", "residual")}
        self.ratio_maps = {"height": None, "area": None}
        if gres is not None:
            self.ratio_maps = {m: np.full((ny, nx), np.nan) for m in ("height", "area")}
        ok = 0
        for r in range(ny):
            for c in range(nx):
                i = mf.index_map[r, c]
                if i < 0:
                    continue
                res = results[i]
                if res["ok"]:
                    ok += 1
                    self.fwhm_map[r, c] = res["fwhm"]
                    for k in self.asym_maps:
                        self.asym_maps[k][r, c] = asym_of(res, k)
                if gres is not None:
                    for m in ("height", "area"):
                        self.ratio_maps[m][r, c] = pixel_ratio(res, gres[i], m)
        return ok

    def layer_grid(self) -> np.ndarray:
        """The per-spectrum layer counts laid onto the current grid."""
        mf = self.mapfile
        g = np.zeros(mf.shape, dtype=int)
        if self.layer_by_spec is not None:
            ok = mf.index_map >= 0
            g[ok] = self.layer_by_spec[mf.index_map[ok]]
        return g

    # -- orientation ---------------------------------------------------------------------------
    def current_transform(self) -> Optional[dict]:
        if self.mapfile_raw is None:
            return None
        return self.cfg.get("map_transform") or default_transform(self.mapfile_raw, self.cfg)

    def apply_transform(self, t: Optional[dict]) -> bool:
        """Re-place the same spectra under a new orientation / offset. False if no map."""
        self.cfg["map_transform"] = t
        if self.mapfile_raw is None:
            return False
        self.mapfile = apply_map_transform(self.mapfile_raw, self.current_transform())
        self.clear_area()                       # the rectangle was drawn in the old placement
        self._band_written = None               # band maps are on the new grid now
        self._lay_peak_maps()                   # per-spectrum fits, laid onto the new grid
        self.rebuild_maps()
        self.write_twod_fit()
        return True

    # -- map area (2 · Map) --------------------------------------------------------------------
    AREA_MODE = "sum"

    # -- window-integral maps (2 · Map) -------------------------------------------------------
    def band_integral_map(self, name: str, lo: float, hi: float):
        """Raw counts integrated over [lo, hi] at every pixel of the current grid; kept, and written
        to /analysis/raman_band_maps."""
        key = "integral_%s_%g_%g" % (re.sub(r"[^A-Za-z0-9]+", "", name) or "window", lo, hi)
        hit = self.band_maps.get(key)
        if hit is not None and hit[3] is self.mapfile:
            grid = hit[2]
        else:
            grid = self.engine.window_integral_map(self.mapfile, lo, hi)
            self.band_maps[key] = (lo, hi, grid, self.mapfile)
        if self.name is not None and key != self._band_written:
            self._band_written = key
            ds = products.band_maps_dataset(self.mapfile, {k: v[:3] for k, v in self.band_maps.items()
                                                           if v[3] is self.mapfile}, key)
            global_workspace.write_analysis_data(self.name, "raman_band_maps", ds)
        return grid

    # -- peak maps: the step-1 model on every pixel (2 · Map) ------------------------------------
    def clear_peak_maps(self):
        self.peak_results = None                # fit_spectrum per spectrum index
        self.peak_model = None
        self.peak_grids = {}                    # key -> grid (peak_maps.peak_maps)
        self._peak_written = None

    def set_peak_results(self, results, model):
        self.peak_results, self.peak_model = results, model
        self._lay_peak_maps()

    def _lay_peak_maps(self):
        if self.peak_results is None or self.mapfile is None:
            return
        self.peak_grids = self.engine.peak_maps(self.mapfile, self.peak_results, self.peak_model)
        self._peak_written = None

    def peak_map(self, key: str):
        """One peak map; written to /analysis/raman_peak_maps (data = this one) when shown."""
        grid = self.peak_grids[key]
        if self.name is not None and key != self._peak_written:
            self._peak_written = key
            ds = products.peak_maps_dataset(self.mapfile, self.peak_grids, self.peak_model, key)
            global_workspace.write_analysis_data(self.name, "raman_peak_maps", ds)
        return grid

    def clear_area(self):
        self.area_rect = None
        self.area_indices = None
        self.area_axis = None
        self.area_spectrum = None
        self.area_fit = None

    def set_area(self, rect) -> int:
        """Sum the spectra of the pixels inside ``rect`` (x0, x1, y0, y1 in µm). Returns the
        number of pixels; 0 leaves no area selected."""
        if self.mapfile is None:
            return 0
        axis, spec, idx = self.engine.area_spectrum(self.mapfile, rect, self.AREA_MODE)
        if idx.size == 0:
            self.clear_area()
            return 0
        self.area_rect, self.area_indices = tuple(rect), idx
        self.area_axis, self.area_spectrum, self.area_fit = axis, spec, None
        return int(idx.size)

    def area_label(self) -> str:
        return "%s area (%d px)" % (raman_loaders.short_label(self.mapfile.path), len(self.area_indices))

    def add_area_to_spectra(self):
        """The area spectrum as a step-1 spectrum."""
        self.add_spectrum(self.area_label(), self.area_axis, self.area_spectrum, self.mapfile.path)

    def stack_add_area(self):
        self.stack_add(raman_loaders.Spectrum(self.area_axis, self.area_spectrum, self.area_label(),
                                              self.mapfile.path))

    def fit_area(self, model):
        """Fit the area spectrum with the step-1 model and write /analysis/raman_roi."""
        if self.area_spectrum is None:
            return None
        self.area_fit = self.engine.fit_spectrum(self.area_axis, self.area_spectrum, model)
        if self.name is not None:
            ds = products.roi_dataset(self.area_axis, self.area_spectrum, self.area_rect,
                                      self.area_indices, self.AREA_MODE, self.area_fit, model)
            global_workspace.write_analysis_data(self.name, "raman_roi", ds)
        return self.area_fit

    # -- settings ------------------------------------------------------------------------------
    def config_to_json(self) -> dict:
        out = dict(self.cfg)
        out["fwhm"] = {str(k): v for k, v in self.cfg["fwhm"].items()}
        out["colors"] = {str(k): v for k, v in self.cfg["colors"].items()}
        out["tol"] = {str(k): v for k, v in self.cfg["tol"].items()}
        out["ratio_cut"] = dict(self.cfg["ratio_cut"])
        out["map_transform"] = self.cfg.get("map_transform")
        return out

    @staticmethod
    def config_from_json(raw: dict) -> dict:
        cfg = default_config()
        cfg.update(raw)
        cfg["fwhm"] = {int(k): {kk: float(vv) for kk, vv in v.items()}
                       for k, v in raw.get("fwhm", {}).items()} or cfg["fwhm"]
        cfg["colors"] = {int(k): dict(v) for k, v in
                         raw.get("colors", {}).items()} or cfg["colors"]
        # Settings written by version 3 carry no tolerances; fall back to
        # half-gap there, which reproduces what that file meant.
        if raw.get("tol"):
            cfg["tol"] = {int(k): {kk: float(vv) for kk, vv in v.items()}
                          for k, v in raw["tol"].items()}
        else:
            cfg["tol"] = {L: {s: abs(cfg["fwhm"][L]["ABC"]
                                     - cfg["fwhm"][L]["ABA"]) / 2.0
                              for s in ("ABA", "ABC")} for L in cfg["fwhm"]}
        return cfg

    # -- workspace /analysis -------------------------------------------------------------------
    def write_twod_fit(self) -> bool:
        if self.name is None or self.fwhm_map is None:
            return False
        ds = products.twod_fit_dataset(self.mapfile, self.fwhm_map, self.asym_maps,
                                       self.ratio_maps, self.fit_tally, self.cfg)
        return global_workspace.write_analysis_data(self.name, "raman_2d_fit", ds)

    def write_classification(self, layers, calls, ncands) -> bool:
        if self.name is None or layers is None:
            return False
        ds = products.classification_dataset(self.mapfile, layers, calls, ncands, self.cfg)
        return global_workspace.write_analysis_data(self.name, "raman_classification", ds)

    def write_table(self, kind: str, rows: list, data_key: str, attrs: Optional[dict] = None) -> bool:
        """Per-spectrum results of the current map (prediction, peak fits, strain/doping)."""
        if self.name is None or not rows:
            return False
        ds = products.table_dataset(kind, rows, data_key, attrs)
        return global_workspace.write_analysis_data(self.name, kind, ds)

    def write_strain_doping(self, rows: list, params: dict) -> bool:
        pts = products.strain_doping_rows(
            rows, lambda g, d: self.engine.decompose_strain_doping(g, d, params))
        return self.write_table("raman_strain_doping", pts, "strain_pct", {"params": params})
