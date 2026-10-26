"""PEEM service, step 12: spatial domain analysis of maps pushed from the Map view.

Mixin of ``PeemService``. A map (any shown layer: XMCD / XLD map, difference, ...) is pushed with its
ROI and, if wanted, the background region, either as raw values or as shown (the Map view's colour
range and gamma). Several maps of several datasets can be pushed; each is one node:

    /analysis/domains_<n>   data = the pushed map; roi / bg masks; after a run, the pipeline's steps
                            (core/peem/domain_analysis.STEPS) and the numbers in attrs

The math is core/peem/domain_analysis.py (the SDA toolkit's pipeline). ROI outlines can be found
automatically (auto_flake_roi) and saved / loaded as JSON to reuse on another map of the same field.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from tensorspec.core.peem import domain_analysis as da
from tensorspec.core.peem.products import _coords, _flat_attrs
from tensorspec.core.workspace import global_workspace
from tensorspec.gui.services.peem_steps.common import _mask, _record

PREFIX = "domains_"
TABLE_KEYS = ("f_up", "f_down", "mz", "p_up", "p_neutral", "p_down", "n_domains_up", "n_domains_down",
              "mean_domain_area_um2", "width_up", "width_down", "period", "wall_length_um",
              "wall_density_per_um", "stripe_angle_deg", "alignment", "flake_area_um2")


class PeemDomainAnalysisSteps:
    # ------------------------------------------------------------------ push / list
    def push_to_domains(self, name: str, kind: str, var: str = "data", mode: str = "raw",
                        vmin: float | None = None, vmax: float | None = None, gamma: float = 1.0,
                        use_roi: bool = True, use_bg: bool = False) -> dict[str, Any]:
        """Send a map to the domain analysis. mode 'raw': its values; 'shown': with the Map view's
        colour range (vmin, vmax) and gamma. The dataset's ROI and (use_bg) background region go along."""
        if mode not in ("raw", "shown"):
            raise ValueError("mode must be 'raw' or 'shown'")
        if mode == "shown" and (vmin is None or vmax is None or not vmax > vmin):
            raise ValueError("'as shown' needs the colour range of the Map view")
        a = np.asarray(self.get_map(name, kind, var), float)
        if a.ndim != 2:
            raise ValueError("only 2-D maps can be pushed")
        rois = self.get_rois(name)
        roi = rois["roi"] if use_roi else None
        bg = rois["bg"] if use_bg else None
        if use_bg and bg is None:
            raise ValueError("Background region is ticked but none is drawn")
        used = [int(k[len(PREFIX):]) for k in global_workspace.list_analysis_children(name)
                if k.startswith(PREFIX) and k[len(PREFIX):].isdigit()]
        label = f"{PREFIX}{max(used, default=0) + 1}"
        extra = {}
        if roi is not None:
            extra["roi"] = _mask(a.shape, roi).astype(np.uint8)
        if bg is not None:
            extra["bg"] = _mask(a.shape, bg).astype(np.uint8)
        attrs = {"source_kind": kind, "source_var": var, "mode": mode, "gamma": float(gamma or 1.0),
                 "vmin": vmin, "vmax": vmax, "analysed": 0, "um_per_px": self._um_per_px(name)}
        self._write_domains(name, label, a, extra, attrs)
        out = {"name": name, "label": label, "source": self.layer_for(name, kind, var), "mode": mode,
               "has_roi": roi is not None, "has_bg": bg is not None}
        _record(name, "push to domain analysis", out)
        return out

    def _write_domains(self, name: str, label: str, data: np.ndarray, extra: dict, attrs: dict) -> None:
        coords = _coords(data.shape, attrs.get("um_per_px"))
        variables = {"data": xr.DataArray(np.asarray(data), dims=("y", "x"), coords=coords)}
        for k, v in extra.items():
            variables[k] = xr.DataArray(np.asarray(v), dims=("y", "x"), coords=coords)
        if not global_workspace.write_analysis_data(name, label, xr.Dataset(variables, attrs=_flat_attrs(attrs))):
            raise ValueError(f"PEEM data '{name}' not found.")

    def _domains(self, name: str, label: str) -> xr.Dataset:
        ds = global_workspace.pull_analysis_data(name, label)
        if ds is None or not label.startswith(PREFIX):
            raise ValueError(f"no pushed map {label!r} in '{name}'")
        return ds

    def domain_inputs(self) -> list[dict[str, Any]]:
        """Every pushed map of every loaded dataset, in push order."""
        out = []
        for name in self.list_datasets():
            for label in sorted((k for k in global_workspace.list_analysis_children(name) if k.startswith(PREFIX)),
                                key=lambda k: int(k[len(PREFIX):]) if k[len(PREFIX):].isdigit() else 0):
                at = dict(self._domains(name, label).attrs)
                src = self.layer_name(at.get("source_kind", "?"), at.get("source_var", "data"))
                out.append({"name": name, "label": label, "source": src, "mode": at.get("mode", "raw"),
                            "analysed": bool(at.get("analysed", 0)), "title": f"{name} · {src} ({at.get('mode')})"})
        return out

    def remove_domain_input(self, name: str, label: str) -> None:
        self._domains(name, label)
        global_workspace.remove_analysis_data(name, label)

    # ------------------------------------------------------------------ run
    def run_domain_analysis(self, name: str, label: str, **params) -> dict[str, Any]:
        """Steps a-l on one pushed map (params: core/peem/domain_analysis.DEFAULTS; flake 'pushed' uses
        the pushed ROI, background 'region' the pushed background region). Returns the numbers."""
        ds = self._domains(name, label)
        at = dict(ds.attrs)
        a = np.asarray(ds["data"].values, float)
        roi = np.asarray(ds["roi"].values, bool) if "roi" in ds else None
        bg = np.asarray(ds["bg"].values, bool) if "bg" in ds else None
        um = at.get("um_per_px")
        res = da.analyze(a, um_per_px=um if um and np.isfinite(um) else None, roi=roi, bg_mask=bg,
                         normalize_mode=at.get("mode", "raw"), vmin=at.get("vmin"), vmax=at.get("vmax"),
                         gamma=at.get("gamma", 1.0), **params)
        extra = {k: ds[k].values for k in ("roi", "bg") if k in ds}
        extra.update(res["layers"])
        keep = {k: at[k] for k in ("source_kind", "source_var", "mode", "gamma", "vmin", "vmax", "um_per_px") if k in at}
        p = {**da.DEFAULTS, **params}
        attrs = {**keep, "analysed": 1, **{f"param_{k}": v for k, v in p.items()},
                 **{k: v for k, v in res["metrics"].items()},
                 "areas_unit": res["areas"]["unit"], "areas_up": [float(v) for v in res["areas"]["up"]],
                 "areas_down": [float(v) for v in res["areas"]["down"]]}
        self._write_domains(name, label, a, extra, attrs)
        out = {"name": name, "label": label, **res["metrics"]}
        _record(name, f"domain analysis {label}", {k: v for k, v in out.items() if np.isscalar(v)})
        return out

    def domain_result(self, name: str, label: str) -> dict[str, Any]:
        """The numbers of one analysed map ({} before a run)."""
        at = dict(self._domains(name, label).attrs)
        return {k: v for k, v in at.items() if np.isscalar(v) and not k.startswith("param_")} if at.get("analysed") else {}

    def domain_layer(self, name: str, label: str, step: str) -> np.ndarray:
        """One pipeline step of an analysed map (core/peem/domain_analysis.STEPS), or 'input'."""
        ds = self._domains(name, label)
        var = "data" if step == "input" else step
        if var not in ds:
            raise ValueError(f"run the domain analysis on {label} first" if step in da.STEPS else f"unknown step {step!r}")
        return np.asarray(ds[var].values)

    def domain_areas(self, name: str, label: str) -> dict[str, Any]:
        """Areas of the single domains (domain-size distribution): {"up", "down", "unit"}."""
        at = dict(self._domains(name, label).attrs)
        if not at.get("analysed"):
            raise ValueError(f"run the domain analysis on {label} first")
        return {"up": np.asarray(at.get("areas_up", []), float), "down": np.asarray(at.get("areas_down", []), float),
                "unit": at.get("areas_unit", "px2")}

    def domain_table(self, items: list[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
        """One row per analysed map: dataset, map, mode and the numbers (widths / period in nm or px)."""
        rows = []
        for it in self.domain_inputs():
            if items is not None and (it["name"], it["label"]) not in items:
                continue
            r = self.domain_result(it["name"], it["label"])
            if not r:
                continue
            unit = r.get("width_unit", "px")
            row = {"dataset": it["name"], "map": it["source"], "push": it["mode"], "label": it["label"]}
            for k in TABLE_KEYS:
                v = r.get(f"{k}_{unit}", r.get(k))
                row[f"{k} ({unit})" if k in ("width_up", "width_down", "period") else k] = v
            rows.append(row)
        return rows

    def export_domain_table(self, path: str | Path, items: list[tuple[str, str]] | None = None) -> str:
        rows = self.domain_table(items)
        if not rows:
            raise ValueError("no analysed maps to export")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return str(path)

    def export_domain_areas(self, path: str | Path, name: str, label: str) -> str:
        ar = self.domain_areas(name, label)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["state", f"area ({ar['unit']})"])
            w.writerows([("up", v) for v in ar["up"]] + [("down", v) for v in ar["down"]])
        return str(path)

    def export_domain_figure(self, path: str | Path, name: str, label: str, step: str = "three_states") -> str:
        """Editable PDF (TrueType text) of one step with axes in um, titled with the dataset and numbers."""
        from tensorspec.plotting import peem_plots

        at = dict(self._domains(name, label).attrs)
        img = self.domain_layer(name, label, step)
        um = at.get("um_per_px")
        title = f"{name} · {self.layer_name(at.get('source_kind', '?'), at.get('source_var', 'data'))}"
        ds = self._domains(name, label)
        flake = np.asarray(ds["flake"].values) if "flake" in ds else None
        peem_plots.save_domain_step_pdf(path, img, step, um if um and np.isfinite(um) else None, title,
                                        self.domain_result(name, label), flake)
        return str(path)

    # ------------------------------------------------------------------ ROI helpers
    def auto_flake_roi(self, name: str, kind: str, var: str = "data", tv_weight: float = 0.20,
                       min_px: int = 800) -> dict[str, Any]:
        """Find the flake on a map (denoise, local std, Otsu, closing, largest region) and set it as
        the dataset's ROI (a polygon, so it can be saved, edited and reused)."""
        a = np.asarray(self.get_map(name, kind, var), float)
        img, _ = da.normalize(a, "raw")
        mask = da.auto_flake(da.local_std(da.denoise(img, "tv", tv_weight)), min_px=min_px)
        roi = {"kind": "polygon", "points": da.flake_outline(mask)}
        self.set_roi(name, "roi", roi)
        return roi

    def save_roi(self, name: str, path: str | Path, which: str = "roi") -> str:
        """The ROI (or background region) as JSON, with the pixel size and map shape, to reuse later."""
        roi = self.get_rois(name)[which]
        if roi is None:
            raise ValueError(f"no {'ROI' if which == 'roi' else 'background region'} to save")
        Path(path).write_text(json.dumps({"tensorspec_roi": 1, "which": which, "roi": roi, "source": name,
                                          "um_per_px": self._um_per_px(name)}, indent=1))
        return str(path)

    def load_roi(self, name: str, path: str | Path, which: str = "roi") -> dict[str, Any]:
        """Set a saved ROI / background region on this dataset (pixel coordinates are kept as saved)."""
        d = json.loads(Path(path).read_text())
        roi = d.get("roi", d) if isinstance(d, dict) else None
        if not isinstance(roi, dict) or roi.get("kind") not in ("rect", "polygon", "ellipse"):
            raise ValueError("not a saved TensorSpec ROI (.json)")
        self.set_roi(name, which, roi)
        return roi
