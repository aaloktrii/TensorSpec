"""PEEM service, XMCD-imaging steps: /analysis products, ROI statistics, SDA domain maps and the XMCD-tab metadata.

Mixin of ``PeemService`` (gui/services/peem_service.py). Moved verbatim from the former
gui/services/peem_xmcd_service.py."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tensorspec.core.peem import domains, products, roi_stats
from tensorspec.core.peem.products import ANALYSIS_KINDS
from tensorspec.core.peem.ue49 import detector
from tensorspec.core.workspace import global_workspace
from tensorspec.gui.services.peem_steps.common import _require_raw, _record, _mask


class PeemMapsDomainsSteps:
    @staticmethod
    def _map_summary(a: np.ndarray) -> dict[str, Any]:
        v = a[np.isfinite(a)]
        if v.size == 0:
            return {"n_finite": 0}
        return {"n_finite": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
                "p1": float(np.percentile(v, 1)), "p99": float(np.percentile(v, 99))}

    # ------------------------------------------------------------- products
    def list_products(self, name: str) -> list[str]:
        return [k for k in ANALYSIS_KINDS if global_workspace.pull_analysis_data(name, k) is not None]

    def get_map(self, name: str, kind: str, var: str = "data") -> np.ndarray:
        ds = global_workspace.pull_analysis_data(name, kind)
        if ds is None:
            raise ValueError(f"'{name}' has no /analysis/{kind}")
        if var not in ds:
            raise ValueError(f"/analysis/{kind} has no variable {var!r} (has {list(ds.data_vars)})")
        return np.asarray(ds[var].values)

    def get_product_attrs(self, name: str, kind: str) -> dict[str, Any]:
        ds = global_workspace.pull_analysis_data(name, kind)
        return {} if ds is None else products.stats_table(ds)

    def export_map_tiff(self, name: str, kind: str, path: str | Path, var: str = "data") -> str:
        import tifffile

        arr = self.get_map(name, kind, var).astype(np.float32)
        out = Path(path).expanduser()
        tifffile.imwrite(str(out), arr)
        return str(out)

    # ------------------------------------------------------------- ROI tools
    def roi_statistics(
        self,
        name: str,
        kind: str = "pcd",
        roi: dict | None = None,
        bg_roi: dict | None = None,
        band: float = 0.0,
        threshold: float | None = None,
        var: str = "data",
    ) -> dict[str, Any]:
        """ROIAverage.ipf: background ROI mean, ROI average outside the gray band, f+/f-/f0."""
        a = self.get_map(name, kind, var)
        bg = 0.0
        out: dict[str, Any] = {"name": name, "kind": kind}
        if bg_roi is not None:
            info = roi_stats.background_level(a, _mask(a.shape, bg_roi))
            bg = info["bg"]
            out.update(info)
        mask = _mask(a.shape, roi)
        out["average"] = roi_stats.roi_average(a, mask, bg=bg, band=band)
        out["fractions"] = roi_stats.updown_fractions(a, mask, threshold=threshold, band=band, bg=bg)
        return out

    @staticmethod
    def auto_fov_roi(image: np.ndarray, shrink: float = 0.9) -> dict:
        cy, cx, r = detector.auto_fov_circle(image)
        r *= shrink
        return {"kind": "ellipse", "cx": cx, "cy": cy, "rx": r, "ry": r}

    # ------------------------------------------------------------- domains
    def compute_domain_map(
        self,
        name: str,
        source_kind: str = "pcd",
        roi: dict | None = None,
        threshold: float = 0.05,
        normalize: str = "p99",
        min_px: int = 12,
        flatten_sigma: float | None = None,
        offset: float = 0.0,
        use_sda: bool | None = None,
        smooth_sigma: float = 0.0,
        bg_roi: dict | None = None,
    ) -> dict[str, Any]:
        """``bg_roi`` (a non-magnetic area) sets the zero of the asymmetry: its mean is
        subtracted before classification, on top of any explicit ``offset``."""
        if source_kind not in ("pcd", "pmld", "pld", "pmnld"):
            raise ValueError("domain_map source must be pcd, pld, pmld or pmnld")
        a = self.get_map(name, source_kind)
        upp = self._um_per_px(name)
        bg_offset = 0.0
        if bg_roi is not None:
            bg_offset = roi_stats.background_level(a, _mask(a.shape, bg_roi))["bg"]
        offset = float(offset or 0.0) + bg_offset
        res = domains.domain_map(a, roi=_mask(a.shape, roi), threshold=threshold, normalize=normalize,
                                 min_px=min_px, flatten_sigma=flatten_sigma, offset=offset,
                                 um_per_px=upp, use_sda=use_sda, smooth_sigma=smooth_sigma)
        ds = products.map_dataset(
            "domain_map", res["mz"].astype(np.int8),
            extra={"m": res["m"], "roi": res["roi"].astype(np.uint8)},
            attrs={**res["stats"], "source_kind": source_kind,
                   "flatten_sigma": flatten_sigma or 0.0, "offset": float(offset),
                   "bg_roi_offset": float(bg_offset)},
            um_per_px=upp, long_name="Three-state domain map (+1 up, -1 down, 0 neutral)",
        )
        self._write(name, "domain_map", ds)
        out = {"name": name, "kind": "domain_map", "offset": offset, "bg_roi_offset": bg_offset, **res["stats"]}
        _record(name, "domain map", out)
        return out

    # ------------------------------------------------------------- meta
    def xmcd_meta(self, name: str) -> dict[str, Any]:
        """PeemService.get_meta plus the XMCD-tab fields."""
        meta = self.get_meta(name)
        raw = _require_raw(name)
        md = raw.metadata or {}
        meta.update({
            "loader": md.get("loader"),
            "fov_um": md.get("fov_um"),
            "um_per_px": md.get("um_per_px"),
            "energy_range": self._energy_range(raw),
            "products": self.list_products(name),
        })
        return meta
