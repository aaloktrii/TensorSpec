"""Raman suite service (no Qt): holds the loaded map, the fits and the analysis config that the
Raman tabs share, routes analysis through RamanEngineRouter and keeps the global workspace
DataTree current (/raw on load, /analysis/raman_* after each step; /history is appended by
DataTreeBuilder).

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
from tensorspec.core.raman import products, reference
from tensorspec.core.raman.classification import pixel_ratio, ratio_applicable
from tensorspec.core.raman.constants import default_config
from tensorspec.core.raman.map_geometry import apply_map_transform, default_transform
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
        self._band_written = None                         # band map shown as /analysis data
        self.clear_peak_maps()
        self.clear_area()

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
        self.clear_area()
        self.band_maps = {}
        self._band_written = None
        self.clear_peak_maps()
        self.name = workspace_name(mf.path)
        global_workspace.push_spectroscopy_data(self.name, raman_loaders.map_to_tensor(mf))
        return res

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
