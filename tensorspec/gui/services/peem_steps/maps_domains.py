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

    def export_map_tiff(self, name: str, kind: str, path: str | Path, var: str = "data", png: bool = True) -> str:
        """32-bit float TIFF of the map values (for ImageJ / Igor / analysis) and, with png, a colour
        PNG next to it (<name>.png), scaled like the viewer, for looking at. Returns the TIFF path."""
        out = Path(path).expanduser()
        self.export_map(name, kind, out.with_suffix(""), var, pictures=("png",) if png else ())
        return str(out.with_suffix(".tif"))

    def save_results(self, name: str, folder: str | Path, pictures: tuple[str, ...] | list[str] = ("png",)) -> dict[str, Any]:
        """Everything computed for one dataset into <folder>/<name>/ (the frames are not copied):

            <kind>_<variable>.tif / .png   every map (stack, asymmetry, filtered maps, domain map ...)
            drift.csv                       per-frame drift (frame, polarization, dy, dx in px)
            history.txt                     every step and its parameters (/history)
            summary.json                    dataset summary and the scalar results of each map
        Then the dataset can be unloaded (unload) without losing the results."""
        import json

        out = Path(folder).expanduser() / name
        out.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        summary: dict[str, Any] = {"dataset": name, "maps": {}}
        try:
            summary["meta"] = {k: v for k, v in self.xmcd_meta(name).items()
                               if isinstance(v, (str, int, float, bool, list)) or v is None}
        except Exception as exc:  # summary is a convenience; maps are what matter
            summary["meta_error"] = str(exc)
        for kind in self.list_products(name):
            summary["maps"][kind] = {k: v for k, v in self.get_product_attrs(name, kind).items()
                                     if isinstance(v, (str, int, float, bool))}
            for var in self.map_variables(name, kind):
                if kind != "stack" and var.startswith("mean_"):
                    continue  # copies of the stacked images, saved once from /analysis/stack
                if np.ndim(self.get_map(name, kind, var)) == 2:
                    files += self.export_map(name, kind, out / f"{kind}_{var}", var, pictures=pictures)
        drift = self.get_drift(name, "processed")
        if drift is not None:
            pol = drift.get("pol") or [""] * len(drift["dy"])
            rows = ["frame,polarization,dy_px,dx_px"] + [f"{i},{p},{dy:.4f},{dx:.4f}" for i, (p, dy, dx)
                                                         in enumerate(zip(pol, drift["dy"], drift["dx"]))]
            (out / "drift.csv").write_text("\n".join(rows) + "\n")
            files.append(str(out / "drift.csv"))
        hist = global_workspace.get(name)["tree"]["history"]
        log = (hist.to_dataset() if hasattr(hist, "to_dataset") else hist.ds).attrs.get("log") or []
        (out / "history.txt").write_text("\n".join(map(str, log)) + "\n")
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        files += [str(out / "history.txt"), str(out / "summary.json")]
        _record(name, "save results", {"folder": str(out), "n_files": len(files)})
        return {"name": name, "folder": str(out), "files": files}

    def export_options(self, name: str, kind: str, var: str = "data") -> dict[str, Any]:
        """What an export dialog starts from: map size, pixel size, colour scale and label the viewer uses."""
        arr = self.get_map(name, kind, var)
        hint = self.display_hint(name, kind, var)
        from tensorspec.plotting import peem_plots

        vmin, vmax = peem_plots.color_limits(arr, hint)
        upp = self._um_per_px(name)
        return {"height_px": int(arr.shape[0]), "width_px": int(arr.shape[1]), "um_per_px": upp,
                "cmap": hint["cmap"], "vmin": vmin, "vmax": vmax,
                "label": str((self.get_product_attrs(name, kind).get("formula") or "") if var == "data" else ""),
                "title": f"{name}: {kind} / {var}"}

    def export_map(
        self,
        name: str,
        kind: str,
        path: str | Path,
        var: str = "data",
        *,
        values: bool = True,
        pictures: tuple[str, ...] | list[str] = ("png",),
        crop: tuple[int, int, int, int] | None = None,
        size_px: tuple[int, int] | None = None,
        dpi: int = 150,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        colorbar: bool = True,
        title: str | None = None,
        label: str | None = None,
        scalebar: bool = False,
        image_only: bool = False,
    ) -> list[str]:
        """Export one map. ``path`` without suffix; writes <path>.tif (32-bit float values, if values)
        and <path>.<fmt> for each picture format in png / pdf / svg. Returns the files written.

        crop        (x0, x1, y0, y1) in map px, applied to the values and the pictures
        size_px     (width, height) of the pictures in px; None = automatic
        cmap, vmin, vmax, title, label   None = what the viewer shows (export_options)
        other options: plotting/peem_plots.save_map_picture
        """
        import tifffile

        from tensorspec.plotting import peem_plots

        arr = self.get_map(name, kind, var).astype(np.float32)
        if crop is not None:
            x0, x1, y0, y1 = (int(round(v)) for v in crop)
            x0, x1 = sorted((max(0, x0), min(arr.shape[1], x1)))
            y0, y1 = sorted((max(0, y0), min(arr.shape[0], y1)))
            if x1 - x0 < 2 or y1 - y0 < 2:
                raise ValueError("the crop region is empty")
            arr = arr[y0:y1, x0:x1]
        bad = [f for f in pictures if f not in ("png", "pdf", "svg")]
        if bad:
            raise ValueError(f"picture formats are png, pdf, svg (got {bad})")
        if not values and not pictures:
            raise ValueError("choose at least one file to write")
        base = Path(path).expanduser()
        base = base.with_suffix("") if base.suffix.lower() in (".tif", ".tiff", ".png", ".pdf", ".svg") else base
        opts = self.export_options(name, kind, var)
        written = []
        if values:
            tifffile.imwrite(str(base.with_suffix(".tif")), arr)
            written.append(str(base.with_suffix(".tif")))
        for fmt in pictures:
            f = base.with_suffix(f".{fmt}")
            peem_plots.save_map_picture(
                arr, f, self.display_hint(name, kind, var), size_px=size_px, dpi=int(dpi), cmap=cmap,
                vmin=vmin, vmax=vmax, colorbar=colorbar, title=opts["title"] if title is None else title,
                label=opts["label"] if label is None else label, um_per_px=opts["um_per_px"],
                scalebar=scalebar, image_only=image_only)
            written.append(str(f))
        _record(name, "export map", {"kind": kind, "var": var, "files": written, "crop": crop,
                                     "size_px": size_px, "dpi": dpi, "cmap": cmap, "vmin": vmin, "vmax": vmax})
        return written

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
