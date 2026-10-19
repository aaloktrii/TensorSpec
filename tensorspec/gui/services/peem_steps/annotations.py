"""PEEM service, XPEEM-map steps: regions and profiles drawn (or typed) on a dataset.

Mixin of ``PeemService`` (gui/services/peem_service.py). The map ROI, background ROI, drift box and
the line profiles live in the dataset (``/analysis/annotations``, JSON attributes), not in a front
end, so any front end (Qt panel, script, web page) reads and changes the same ones and they stay
with the data.

    set_roi / get_rois       which = "roi" (statistics, blanking), "bg" (background), "drift_box"
    add_profile              profile of a line A->B on any shown image, stored with its line
    list_profiles / clear_profiles
"""
from __future__ import annotations

import json
from typing import Any

import xarray as xr

from tensorspec.core.workspace import global_workspace
from tensorspec.gui.services.peem_steps.common import _record

ROI_KINDS = ("roi", "bg", "drift_box")


class PeemAnnotationSteps:
    def _annotations(self, name: str) -> dict[str, Any]:
        ds = global_workspace.pull_analysis_data(name, "annotations")
        data = json.loads(ds.attrs.get("annotations", "{}")) if ds is not None else {}
        data.setdefault("rois", {})
        data.setdefault("profiles", [])
        return data

    def _save_annotations(self, name: str, data: dict[str, Any]) -> None:
        self._write(name, "annotations", xr.Dataset(attrs={"annotations": json.dumps(data)}))

    # ---------------------------------------------------------------- regions
    def get_rois(self, name: str) -> dict[str, dict | None]:
        """{"roi": ..., "bg": ..., "drift_box": ...}; None where nothing is set."""
        rois = self._annotations(name)["rois"]
        return {k: rois.get(k) for k in ROI_KINDS}

    def set_roi(self, name: str, which: str, roi: dict | None) -> dict[str, dict | None]:
        """Set (or clear with None) one region: rect {x0, x1, y0, y1}, polygon {points} or ellipse."""
        if which not in ROI_KINDS:
            raise ValueError(f"which must be one of {ROI_KINDS}")
        if roi is not None and roi.get("kind") not in ("rect", "polygon", "ellipse"):
            raise ValueError("roi kind must be rect, polygon or ellipse")
        data = self._annotations(name)
        data["rois"][which] = roi
        self._save_annotations(name, data)
        _record(name, f"set {which}", {"roi": json.dumps(roi)})
        return self.get_rois(name)

    # ---------------------------------------------------------------- profiles
    def add_profile(self, name: str, a, b, halfwidth: int = 0, kind: str = "raw", var: str = "data",
                    frame: int = 0) -> dict[str, Any]:
        """Profile A->B (points (y, x) in px) of the image the front end shows, added to the list."""
        pr = self.line_profile(name, kind, var, frame, a, b, halfwidth)
        (y1, x1), (y2, x2) = a, b
        data = self._annotations(name)
        n = len(data["profiles"]) + 1
        shown_var = var or "data"
        if kind == "nc" and shown_var == "data":  # a loaded .nc map: name it by its variable (e.g. xmld)
            shown_var = str(self.get_product_attrs(name, "nc").get("primary") or "data")
        entry = {"line": [[float(y1), float(x1)], [float(y2), float(x2)]], "halfwidth": int(halfwidth),
                 "kind": kind, "var": var or "data", "frame": int(frame),
                 "label": f"{n}: {kind}/{shown_var} ({x1:.0f},{y1:.0f})→({x2:.0f},{y2:.0f})",
                 "distance": [float(v) for v in pr["distance"]], "values": [float(v) for v in pr["values"]],
                 "unit": pr["unit"]}
        data["profiles"].append(entry)
        self._save_annotations(name, data)
        _record(name, "add profile", {k: entry[k] for k in ("label", "halfwidth", "kind", "var", "frame")})
        return entry

    def list_profiles(self, name: str) -> list[dict[str, Any]]:
        return self._annotations(name)["profiles"]

    def remove_profile(self, name: str, index: int) -> list[dict[str, Any]]:
        """Remove profile ``index`` (0-based, list order); returns the remaining profiles."""
        data = self._annotations(name)
        if not 0 <= int(index) < len(data["profiles"]):
            raise ValueError(f"no profile {index + 1}")
        gone = data["profiles"].pop(int(index))
        self._save_annotations(name, data)
        _record(name, "remove profile", {"label": gone.get("label", "")})
        return data["profiles"]

    def set_profile_visible(self, name: str, index: int, visible: bool) -> None:
        """Show / hide profile ``index`` in the profile plot and on the map (it stays in the list)."""
        data = self._annotations(name)
        if not 0 <= int(index) < len(data["profiles"]):
            raise ValueError(f"no profile {index + 1}")
        data["profiles"][int(index)]["visible"] = bool(visible)
        self._save_annotations(name, data)

    def clear_profiles(self, name: str) -> None:
        data = self._annotations(name)
        data["profiles"] = []
        self._save_annotations(name, data)
        _record(name, "clear profiles", {})
