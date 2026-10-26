"""PEEM service, XMCD-imaging steps: Viewer layers and image analysis of maps / frames (core/peem/image_analysis).

Mixin of ``PeemService`` (gui/services/peem_service.py). Moved verbatim from the former
gui/services/peem_xmcd_service.py."""
from __future__ import annotations

from typing import Any

import numpy as np

from tensorspec.core.peem.bg import resolve_energy
from tensorspec.core.peem import image_analysis
from tensorspec.core.workspace import global_workspace
from tensorspec.plotting import peem_plots
from tensorspec.gui.services.peem_steps.common import _require_raw, _working_frames, _record, _mask


class PeemImageAnalysisSteps:
    # ------------------------------------------------------------ image analysis (shown image)
    FRAME_VIEWS = ("raw", "cleaned", "processed", "stacked")

    def view_image(self, name: str, kind: str, var: str = "data", frame: int = 0) -> np.ndarray:
        """The image a view shows: a raw / cleaned / drift-corrected frame, or a map variable."""
        if kind == "cleaned":
            return self.get_working_frame(name, frame)
        if kind == "stacked":
            return np.asarray(self.stacked_frames(name)[0][int(frame)], dtype=float)
        if kind in ("raw", "processed"):
            return np.asarray(self.get_view_tensor(name, kind, int(frame)), dtype=float)
        return np.asarray(self.get_map(name, kind, var or "data"), dtype=float)

    def display_hint(self, name: str, kind: str, var: str = "data", frame: int = 0) -> dict[str, Any]:
        """Colour scale for the image a front end shows: {cmap, symmetric, center}
        (plotting/peem_plots.display_hint)."""
        if kind == "nc":  # loaded .nc maps: each variable keeps its type (signed / intensity / domain)
            t = self.map_type(name, kind, var)
            alias = {"signed": "pld", "domain": "domain_map"}.get(t, "stack")
            img = self.view_image(name, kind, var, frame) if t == "signed" else None
            return peem_plots.display_hint(alias, "data", img)
        signed = kind in peem_plots.SIGNED_KINDS and (var or "") in peem_plots.SIGNED_VARS
        img = self.view_image(name, kind, var, frame) if signed else None
        return peem_plots.display_hint(kind, var, img)

    @staticmethod
    def layer_name(kind: str, var: str = "data", channels: tuple[str, str] | None = None) -> str:
        """Viewer layer name of a map variable: 'pcd', 'pcd · difference', ...; the stack's own image
        (the two stacked images added) is 'stack · cp+cm' (channels from its data)."""
        if var in ("", "data"):
            return f"{kind} · {channels[0].lower()}+{channels[1].lower()}" if channels else kind
        return f"{kind} · {var}"

    @staticmethod
    def parse_layer(layer: str) -> tuple[str, str]:
        if " · " not in layer:
            return layer, "data"
        kind, var = layer.split(" · ", 1)
        return kind, "data" if "+" in var else var  # 'stack · cp+cm' is the stack's data

    def _channels(self, name: str, kind: str) -> tuple[str, str] | None:
        """(A, B) channel tags a sum map records (the stack), else None."""
        if kind != "stack":
            return None
        at = self.get_product_attrs(name, kind)
        return (str(at["channel_a"]), str(at["channel_b"])) if "channel_a" in at and "channel_b" in at else None

    def layer_for(self, name: str, kind: str, var: str = "data") -> str:
        """layer_name of a map of dataset ``name`` (the stack's sum named after its channels)."""
        return self.layer_name(kind, var, self._channels(name, kind))

    def map_layers(self, name: str) -> dict[str, np.ndarray]:
        """Every 2-D map variable of every product, keyed by viewer layer name."""
        out: dict[str, np.ndarray] = {}
        for kind in self.list_products(name):
            ds = global_workspace.pull_analysis_data(name, kind)
            for var in ds.data_vars:
                a = np.asarray(ds[var].values)
                if a.ndim == 2:
                    out[self.layer_for(name, kind, var)] = a.astype(float)
        return out

    # variables a viewer does not need to list: sums (= mean x frames), the asymmetry map's copies of the
    # stacked means, and the domain map's working arrays
    _HIDDEN_LAYERS = {"stack": ("sum_",), "pcd": ("mean_",), "pld": ("mean_",), "domain_map": ("m", "roi")}

    def shown_layers(self, name: str) -> dict[str, np.ndarray]:
        """map_layers without duplicates or working arrays: the stacked images (A+B, A, B), each map
        and its difference, the domain map, comparisons. What the Map view lists."""
        out = {}
        for layer, a in self.map_layers(name).items():
            kind, var = self.parse_layer(layer)
            hide = self._HIDDEN_LAYERS.get(kind, ())
            if var != "data" and any(var == h or (h.endswith("_") and var.startswith(h)) for h in hide):
                continue
            out[layer] = a
        return out

    def map_variables(self, name: str, kind: str) -> list[str]:
        ds = global_workspace.pull_analysis_data(name, kind)
        return [] if ds is None else [str(v) for v in ds.data_vars]

    def view_frames(self, name: str, kind: str) -> np.ndarray:
        if kind == "cleaned":
            return np.asarray(_working_frames(name)[0], dtype=float)
        if kind == "raw":
            return np.asarray(_require_raw(name).value, dtype=float)
        if kind == "processed":
            proc = global_workspace.pull_tensor_data(name, "processed")
            if proc is None:
                raise ValueError("no drift-corrected frames yet")
            return np.asarray(proc.value, dtype=float)
        if kind == "stacked":
            return np.asarray(self.stacked_frames(name)[0], dtype=float)
        raise ValueError("profile vs frame needs a frame view (raw, cleaned, drift-corrected or stacked)")

    def line_profile(self, name: str, kind: str, var: str, frame: int, a, b, halfwidth: int = 0) -> dict[str, Any]:
        """Igor LineProfile on the shown image (distance in µm when the pixel size is known)."""
        return image_analysis.line_profile(self.view_image(name, kind, var, frame), a, b, halfwidth, self._um_per_px(name))

    def stack_line_profiles(self, name: str, kind: str, a, b, halfwidth: int = 0) -> dict[str, Any]:
        """Igor displayLines / stackLines: profile of every frame of the shown stack."""
        out = image_analysis.stack_line_profiles(self.view_frames(name, kind), a, b, halfwidth, self._um_per_px(name))
        energy, src = resolve_energy(len(_require_raw(name).value), _require_raw(name).metadata or {})
        if kind == "raw" and src != "index":
            out["energy"] = np.asarray(energy, float)
        return out

    def histogram(self, name: str, kind: str, var: str, frame: int, roi: dict | None = None,
                  value_range: tuple[float, float] | None = None) -> dict[str, Any]:
        """Igor calcFrameHistogram (255 bins) of the shown image, inside the map ROI if given."""
        img = self.view_image(name, kind, var, frame)
        return image_analysis.histogram(img, _mask(img.shape, roi), 255, value_range)

    def image_fft(self, name: str, kind: str, var: str, frame: int) -> dict[str, Any]:
        """Igor imageFFT + azimuthal average -> dominant spatial period."""
        f = image_analysis.image_fft(self.view_image(name, kind, var, frame))
        f["radial"] = image_analysis.radial_profile(f, self._um_per_px(name))
        return f

    def _map_variant(self, name: str, kind: str, src_var: str, new_var: str, arr: np.ndarray, note: str) -> dict[str, Any]:
        ds = global_workspace.pull_analysis_data(name, kind)
        if ds is None:
            raise ValueError(f"no {kind} map")
        ds = ds.copy()
        ds[new_var] = (ds[src_var].dims, np.asarray(arr, dtype=float))
        ds[new_var].attrs["note"] = note
        self._write(name, kind, ds)
        out = {"name": name, "kind": kind, "variable": new_var, "from": src_var, "note": note,
               **self._map_summary(np.asarray(arr))}
        _record(name, "map variant", out)
        return out

    def _image_op(self, name: str, kind: str, var: str, frame: int, new_var: str, fn, note: str, stack_edit: dict | None):
        if kind in self.FRAME_VIEWS:  # frames: applied to the whole stack as a clean-up edit
            e = self.get_edits(name)
            e.update(stack_edit)
            out = self._set_edits(name, e)
            out["applied_to"] = "whole stack (clean-up edit; re-run the drift correction)"
            return out
        img = self.view_image(name, kind, var, frame)
        return self._map_variant(name, kind, var or "data", new_var, fn(img), note)

    def flatten(self, name: str, kind: str, var: str, frame: int, a, b, halfwidth: int = 30,
                subtract_offset: bool = False, points_only: bool = False) -> dict[str, Any]:
        """Igor Flatten (map -> variable 'flattened') / FlattenSequence (frames -> every frame)."""
        a, b = tuple(map(float, a)), tuple(map(float, b))
        return self._image_op(
            name, kind, var, frame, "flattened",
            lambda im: image_analysis.flatten_line(im, a, b, halfwidth, subtract_offset, points_only)[0],
            f"flattened along {a}->{b}, halfwidth {halfwidth}, offset {'removed' if subtract_offset else 'kept'}",
            {"flatten": {"a": list(a), "b": list(b), "halfwidth": int(halfwidth),
                         "subtract_offset": bool(subtract_offset), "points_only": bool(points_only)}})

    def background_poly(self, name: str, kind: str, var: str, frame: int, order: int = 2, width: int = 5) -> dict[str, Any]:
        """Igor backgroundremoval (row + column polynomials)."""
        return self._image_op(name, kind, var, frame, "bg_removed",
                              lambda im: image_analysis.background_poly(im, order, width)[0],
                              f"row/column polynomial background removed (order {order}, cell {width})",
                              {"bg_poly": {"order": int(order), "width": int(width)}})

    def background_roi(self, name: str, kind: str, var: str, frame: int, bg_roi: dict, order: int = 3) -> dict[str, Any]:
        """Igor bkgRemove: polynomial surface fitted in the background ROI (maps only)."""
        if kind in self.FRAME_VIEWS:
            raise ValueError("background from the BG ROI works on maps; for frames use the polynomial background")
        img = self.view_image(name, kind, var, frame)
        return self._map_variant(name, kind, var or "data", "bg_removed",
                                 image_analysis.remove_background_roi(img, _mask(img.shape, bg_roi), order)[0],
                                 f"order-{order} polynomial fitted in the BG ROI removed")

    def fft_filter(self, name: str, kind: str, var: str, frame: int, cutoff: int, mode: str = "x") -> dict[str, Any]:
        """Igor FFTFilter (maps -> variable 'fft_filtered')."""
        if kind in self.FRAME_VIEWS:
            raise ValueError("FFT filter works on maps (switch the view to a map)")
        img = self.view_image(name, kind, var, frame)
        return self._map_variant(name, kind, var or "data", "fft_filtered",
                                 image_analysis.fft_filter(img, int(cutoff), mode),
                                 f"FFT low-pass, cutoff {cutoff}, {mode}")

    def use_as_map_data(self, name: str, kind: str, var: str) -> dict[str, Any]:
        """Make a variable (e.g. 'flattened') the map 'data' used by ROI statistics and domain maps;
        the previous data are kept as 'data_original'."""
        ds = global_workspace.pull_analysis_data(name, kind)
        if ds is None or var not in ds:
            raise ValueError(f"no variable {var!r} in {kind}")
        if var == "data":
            return {"name": name, "kind": kind, "note": "already the map data"}
        ds = ds.copy()
        if "data_original" not in ds:
            ds["data_original"] = ds["data"].copy()
        ds["data"] = ds[var].copy()
        ds.attrs["data_from"] = var
        self._write(name, kind, ds)
        out = {"name": name, "kind": kind, "data_from": var, **self._map_summary(np.asarray(ds["data"].values))}
        _record(name, "use as map data", out)
        return out
