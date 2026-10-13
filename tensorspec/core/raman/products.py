"""xarray Datasets for the Raman outputs under ``/analysis/<kind>`` (no Qt).

As in core/peem/products.py, every Dataset has a ``data`` variable so ``pull_tensor_data`` and
the generic DataViewer can open it: map products put it on (y, x) of the current map grid, with
secondary maps as extra variables; per-spectrum tables put it on ``spot`` with the other columns
as extra variables. Scalars and settings go into attrs.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import xarray as xr

# /analysis/<kind> products of the Raman suite
RAMAN_ANALYSIS_KINDS = ("raman_2d_fit", "raman_classification", "raman_layer_prediction",
                        "raman_peakfit", "raman_strain_doping", "raman_roi", "raman_band_maps",
                        "raman_peak_maps")


def _grid_coords(mapfile) -> dict:
    return {"y": ("y", np.asarray(mapfile.uy, float), {"units": "um"}),
            "x": ("x", np.asarray(mapfile.ux, float), {"units": "um"})}


def _attr(v: Any):
    """A value HDF5/NetCDF attrs accept: scalars and strings as-is, the rest as JSON."""
    if isinstance(v, np.generic):
        return v.item()
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    return json.dumps(v, default=str)


def twod_fit_dataset(mapfile, fwhm_map, asym_maps: dict, ratio_maps: dict, fit_tally: dict,
                     cfg: dict) -> xr.Dataset:
    """FWHM(2D) map from the Voigt fits plus the asymmetry and I(2D)/I(G) maps."""
    coords = _grid_coords(mapfile)
    vars_ = {"data": (("y", "x"), np.asarray(fwhm_map, float),
                      {"long_name": "FWHM(2D)", "units": "cm-1"})}
    for k, m in (asym_maps or {}).items():
        if m is not None:
            vars_["asym_" + k] = (("y", "x"), np.asarray(m, float))
    for k, m in (ratio_maps or {}).items():
        if m is not None:
            vars_["ratio_2d_g_" + k] = (("y", "x"), np.asarray(m, float))
    attrs = {"kind": "raman_2d_fit", "model": "voigt",
             "band_min": float(cfg["band_min"]), "band_max": float(cfg["band_max"]),
             "engine": cfg.get("engine", "scipy"),
             "n_fitted": int(np.isfinite(fwhm_map).sum()),
             "rejected": _attr(fit_tally or {}),
             "map_transform": _attr(cfg.get("map_transform"))}
    return xr.Dataset(vars_, coords=coords, attrs=attrs)


def classification_dataset(mapfile, layers, calls, ncands, cfg: dict) -> xr.Dataset:
    """Layer number (data), stacking call and candidate count per map pixel."""
    calls = np.asarray([["" if c is None else str(c) for c in row] for row in calls])
    vars_ = {"data": (("y", "x"), np.asarray(layers, int), {"long_name": "layer number"}),
             "stacking": (("y", "x"), calls),
             "n_candidates": (("y", "x"), np.asarray(ncands, int))}
    attrs = {"kind": "raman_classification", "layer_mode": cfg.get("layer_mode", ""),
             "technique": _attr(cfg.get("technique", "")),
             "asym_metric": _attr(cfg.get("asym_metric", "")),
             "overlap_policy": _attr(cfg.get("overlap_policy", "")),
             "fwhm_table": _attr({str(k): v for k, v in cfg["fwhm"].items()}),
             "tolerance": _attr({str(k): v for k, v in cfg["tol"].items()})}
    return xr.Dataset(vars_, coords=_grid_coords(mapfile), attrs=attrs)


def table_dataset(kind: str, rows: list, data_key: str, attrs: dict | None = None) -> xr.Dataset:
    """Per-spectrum rows (list of dicts) as variables on ``spot``; ``data`` = rows[data_key]."""
    if kind not in RAMAN_ANALYSIS_KINDS:
        raise ValueError(f"kind must be one of {RAMAN_ANALYSIS_KINDS}")
    keys: list = []
    for r in rows:
        keys += [k for k in r if k not in keys and not isinstance(r[k], (list, tuple, dict))]
    vars_ = {}
    for k in keys:
        vals = [r.get(k) for r in rows]
        try:
            arr = np.array([np.nan if v is None or v == "" else float(v) for v in vals])
        except (TypeError, ValueError):
            arr = np.array(["" if v is None else str(v) for v in vals])
        vars_[k if k != "data" else "data_column"] = (("spot",), arr)
    vars_["data"] = vars_[data_key]
    out_attrs = {"kind": kind, "n_spots": len(rows)}
    out_attrs.update({k: _attr(v) for k, v in (attrs or {}).items()})
    return xr.Dataset(vars_, coords={"spot": np.arange(len(rows))}, attrs=out_attrs)


def roi_dataset(axis, spectrum, rect, indices, mode: str, fit=None, model=None) -> xr.Dataset:
    """Integrated spectrum of a map area (data on 'Raman shift'), the pixels it covers, and its
    fit with the step-1 model (per band: peak count and each peak's position / FWHM / height)."""
    vars_ = {"data": (("Raman shift",), np.asarray(spectrum, float),
                      {"long_name": "area spectrum (%s)" % mode}),
             "spectrum_index": (("pixel",), np.asarray(indices, int))}
    attrs = {"kind": "raman_roi", "rect_um": _attr(list(rect)), "mode": mode,
             "n_pixels": int(len(indices))}
    if model is not None:
        attrs["fit_model"] = _attr(model.to_dict())
    if fit is not None:
        attrs["si_shift"] = float(fit["shift"])
        attrs["fit"] = _attr({name: {"n": b.get("n", 0), "ok": bool(b.get("ok")),
                                     "peaks": [{q: float(c[q]) for q in ("pos", "fwhm", "height", "area")}
                                               for c in (b.get("comps") or [])]}
                              for name, b in fit["bands"].items()})
    return xr.Dataset(vars_, coords={"Raman shift": ("Raman shift", np.asarray(axis, float),
                                                     {"units": "cm-1"})}, attrs=attrs)


def band_maps_dataset(mapfile, maps: dict, latest: str) -> xr.Dataset:
    """Window-integral maps on (y, x): one variable per window, ``data`` = the latest one.
    ``maps``: {name: (lo, hi, grid)}."""
    vars_ = {}
    for name, (lo, hi, grid) in maps.items():
        vars_[name] = (("y", "x"), np.asarray(grid, float),
                       {"window_cm-1": [float(lo), float(hi)], "units": "counts cm-1",
                        "long_name": "raw counts integrated over %g-%g cm-1" % (lo, hi)})
    vars_["data"] = vars_[latest]
    return xr.Dataset(vars_, coords=_grid_coords(mapfile),
                      attrs={"kind": "raman_band_maps", "latest": latest,
                             "integration": "trapezoid, raw counts, no background subtracted"})


def peak_maps_dataset(mapfile, maps: dict, model, shown: str) -> xr.Dataset:
    """Peak maps of the step-1 model on (y, x), one variable per map; ``data`` = the one shown."""
    units = {"pos": "cm-1", "fwhm": "cm-1", "height": "counts", "area": "counts cm-1", "n": ""}
    vars_ = {k: (("y", "x"), np.asarray(g, float), {"units": units[k.rsplit("_", 1)[1]]})
             for k, g in maps.items()}
    vars_["data"] = vars_[shown]
    return xr.Dataset(vars_, coords=_grid_coords(mapfile),
                      attrs={"kind": "raman_peak_maps", "shown": shown,
                             "fit_model": _attr(model.to_dict()),
                             "peak_numbering": "per pixel, low to high wavenumber"})


def strain_doping_rows(rows: list, decompose) -> list:
    """One row per correlation point (as in strain_doping.write_csv), with strain / doping."""
    out = []
    for r in rows:
        base = {k: v for k, v in r.items() if k != "points"}
        for g, wg, wd in r.get("points") or []:
            e, n = decompose([wg], [wd])
            out.append({**base, "point_group": g, "w_G": wg, "w_2D": wd,
                        "strain_pct": float(e[0]), "doping_1e13": float(n[0])})
    return out
