"""PEEM service, step 11: compare a map with a reference dataset's map (e.g. XMLD = XLD(T) - XLD(T > T_c)).

Mixin of ``PeemService``. Both datasets need their asymmetry map (4. Stack -> asymmetry). The
reference's map is registered onto this dataset's grid on the intensity images (sum of the two
stacked channels; core/peem/map_align: shift, rotation, scale, with a rotation / scale search when
the field was re-imaged), then subtracted (dichroism.split_by_reference, register=False):

    /analysis/compare   data = map - registered reference, map, reference (registered),
                        attrs: reference dataset, kind, rotation / scale / shift, overlap r,
                        noise ratio, correlation, formula
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tensorspec.core.peem import dichroism, map_align, products
from tensorspec.core.workspace import global_workspace
from tensorspec.plotting import peem_plots
from tensorspec.gui.services.peem_steps.common import _record

ASYM_KINDS = ("pcd", "pld", "nc")  # nc: the primary map of a loaded .nc file
SEARCH = ("auto", "none", "full")


class PeemCompareSteps:
    def _asym_kind(self, name: str, kind: str | None = None) -> str:
        have = [k for k in ASYM_KINDS if k in self.list_products(name)]
        if kind:
            if kind not in have:
                raise ValueError(f"'{name}' has no {kind} map: run 4. asymmetry on it first.")
            return kind
        if not have:
            raise ValueError(f"'{name}' has no asymmetry map: run 4. asymmetry on it first (or load a .nc map).")
        return have[0]

    def _intensity(self, name: str, kind: str = "pcd") -> np.ndarray:
        """Image to register on: the sum of the stacked channels; for maps loaded without channels
        (.nc analysis outputs) the map itself (the fit uses edge images, so sign changes are fine)."""
        st = global_workspace.pull_analysis_data(name, "stack")
        if st is None:
            if kind == "nc":
                return np.asarray(self.get_map(name, "nc"), float)
            raise ValueError(f"'{name}' has no stacked images: run 4. asymmetry on it first.")
        means = [v for v in st.data_vars if str(v).startswith("mean_")]
        return np.sum([np.asarray(st[v].values, float) for v in means], axis=0)

    def compare_candidates(self, name: str) -> list[str]:
        """Other loaded datasets that have an asymmetry map (possible references)."""
        out = []
        for other in list(getattr(global_workspace, "_data", {})):
            if other == name or global_workspace.pull_tensor_data(other, "raw") is None:
                continue
            try:
                if any(k in self.list_products(other) for k in ASYM_KINDS):
                    out.append(other)
            except (ValueError, KeyError):
                continue
        return out

    def compare_maps(self, name: str, reference: str, kind: str | None = None, search: str = "auto",
                     rotation: float = 0.0, smooth_reference: float = 0.0) -> dict[str, Any]:
        """Register ``reference``'s asymmetry map onto ``name``'s and subtract it -> /analysis/compare.

        kind: 'pcd' or 'pld' (None: the one both have); search: 'auto' | 'none' | 'full' (see
        map_align.align); rotation: expected rotation of the reference field (deg, e.g. an azimuth
        change); smooth_reference: Gaussian sigma (px) for a noisy reference."""
        if search not in SEARCH:
            raise ValueError(f"search must be one of {SEARCH}")
        kind = self._asym_kind(name, kind)
        ref_have = [k for k in ASYM_KINDS if k in self.list_products(reference)]
        kind_ref = kind if kind in ref_have else self._asym_kind(reference)  # e.g. a .nc map vs a pld map
        m = np.asarray(self.get_map(name, kind), float)
        ref_map = np.asarray(self.get_map(reference, kind_ref), float)
        i_this, i_ref = self._intensity(name, kind), self._intensity(reference, kind_ref)
        if (kind == "nc") != (kind_ref == "nc"):  # one side has no channel image: register map on map
            i_this, i_ref = m, ref_map
        um_this, um_ref = self._um_per_px(name), self._um_per_px(reference)
        scale = um_this / um_ref if um_this and um_ref else 1.0  # reference px per px of this grid
        tr, r = map_align.align(i_this, i_ref, np.isfinite(m), np.isfinite(ref_map), scale=scale,
                                rotation=rotation, search=search)
        ref_reg = tr.warp(ref_map)
        ref_reg[~np.isfinite(m)] = np.nan
        sp = dichroism.split_by_reference(m, ref_reg, register=False, reference_smooth_sigma=smooth_reference)
        geo = tr.describe(((m.shape[0] - 1) / 2, (m.shape[1] - 1) / 2))
        formula = f"{kind}({name}) - {kind_ref}({reference}, registered)"
        attrs = {"reference": reference, "kind_compared": kind, "formula": formula, "search": search,
                 "overlap_r": r, "noise_ratio": sp["noise_ratio"], "correlation": sp["correlation"],
                 "reliable": bool(sp["reliable"]), "smooth_reference_px": float(smooth_reference),
                 "transform_A": np.ravel(tr.A).tolist(), "transform_t": list(tr.t), **geo}
        ds = products.map_dataset("compare", sp["pmld"], extra={"map": m, "reference": sp["pmnld"]},
                                  attrs=attrs, um_per_px=self._um_per_px(name),
                                  long_name=f"PEEM {kind} minus registered reference")
        self._write(name, "compare", ds)
        out = {"name": name, "step": "11. compare", "kind": kind, "reference": reference, "formula": formula,
               "overlap_r": r, "noise_ratio": sp["noise_ratio"], "correlation": sp["correlation"],
               "warning": sp["warning"], **geo, **self._map_summary(sp["pmld"])}
        _record(name, "compare", {k: v for k, v in out.items() if k != "name"})
        return out

    @staticmethod
    def compare_outputs(name: str) -> list[tuple[str, str]]:
        """[(label, variable)] of the compare product a front end can show."""
        return [("difference", "data"), ("map", "map"), ("reference (registered)", "reference")]

    def export_compare_figure(self, name: str, path: str | Path) -> str:
        """Map | registered reference | difference on one grey scale, vector PDF (or PNG / SVG by suffix)."""
        at = self.get_product_attrs(name, "compare")
        k = at.get("kind_compared", "map")
        panels = [(f"{k} {name}", self.get_map(name, "compare", "map")),
                  (f"{k} {at.get('reference', 'reference')} (registered)", self.get_map(name, "compare", "reference")),
                  ("difference", self.get_map(name, "compare", "data"))]
        return peem_plots.save_three_panel(panels, path, self._um_per_px(name))
