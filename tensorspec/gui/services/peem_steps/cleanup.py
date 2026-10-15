"""PEEM service, XMCD-imaging steps: Non-destructive stack clean-up before drift (core/peem/stack_edits; edits in /analysis/stack_edits).

Mixin of ``PeemService`` (gui/services/peem_service.py). Moved verbatim from the former
gui/services/peem_xmcd_service.py."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import xarray as xr

from tensorspec.core.peem import dichroism, stack_edits
from tensorspec.gui.services.peem_steps.common import _require_raw, _working_frames, _read_edits, _record, _box


class PeemCleanupSteps:
    # ------------------------------------------------------------ stack clean-up (before drift)
    def get_edits(self, name: str) -> dict[str, Any]:
        return _read_edits(name)

    def _set_edits(self, name: str, edits: dict[str, Any]) -> dict[str, Any]:
        """Edits live in /analysis/stack_edits (JSON attr); /raw is never written after loading."""
        e = stack_edits.normalized(edits)
        self._write(name, "stack_edits", xr.Dataset(attrs={"edits": json.dumps(e)}))
        out = self.edits_summary(name)
        _record(name, "stack edits", {**out, "edits": json.dumps(e)})
        return out

    def edits_summary(self, name: str) -> dict[str, Any]:
        e = self.get_edits(name)
        n = len(_require_raw(name).value)
        parts = []
        if e["exclude"]:
            parts.append(f"{len(e['exclude'])} frame(s) excluded")
        if e["topup"]:
            parts.append(f"top-up x{e['topup']['factor']:g} on {len(e['topup']['frames'])} frame(s)")
        if e["normalize_to"] is not None:
            parts.append(f"normalized to frame {e['normalize_to']}")
        if e["subtract"] is not None:
            parts.append(f"frame {e['subtract']} subtracted")
        if e["pixel_fixes"]:
            parts.append(f"{len(e['pixel_fixes'])} pixel(s) disabled")
        if e["flatten"]:
            parts.append("frames flattened along a line")
        if e["bg_poly"]:
            parts.append(f"polynomial background removed (order {e['bg_poly'].get('order')})")
        return {"name": name, "edits": e, "frames_used": n - len(e["exclude"]), "frames_total": n,
                "summary": "; ".join(parts) or "no clean-up edits",
                "note": "re-run the drift correction to apply" if parts else ""}

    def set_excluded(self, name: str, frames: list[int], exclude: bool = True) -> dict[str, Any]:
        """Igor RemoveLayerFromStack (non-destructive): exclude / include frames."""
        e = self.get_edits(name)
        cur = set(e["exclude"])
        cur = cur | set(map(int, frames)) if exclude else cur - set(map(int, frames))
        e["exclude"] = sorted(cur)
        return self._set_edits(name, e)

    def include_all(self, name: str) -> dict[str, Any]:
        e = self.get_edits(name)
        e["exclude"] = []
        return self._set_edits(name, e)

    def frame_sharpness(self, name: str, box_roi: dict | None = None) -> dict[str, Any]:
        """Igor RemoveDefocussedLayers metric: std of every raw frame in the marquee (or central half)."""
        raw = _require_raw(name)
        frames = np.asarray(raw.value)
        box = _box(box_roi) or dichroism._central_box(frames.shape)
        sd = stack_edits.frame_sharpness(frames, box)
        return {"std": sd, "median": float(np.median(sd)), "box_y0y1x0x1": [int(v) for v in box]}

    def exclude_defocused(self, name: str, rel_threshold: float = 0.8, box_roi: dict | None = None) -> dict[str, Any]:
        """Exclude frames whose std in the box is below rel_threshold x the median std."""
        r = self.frame_sharpness(name, box_roi)
        low = [int(i) for i in np.nonzero(r["std"] < rel_threshold * r["median"])[0]]
        out = self.set_excluded(name, low, True)
        out["defocused_frames"] = low
        return out

    def injection_frames(self, name: str) -> list[int]:
        """Frames right after a top-up injection, from the ring-current column (if the format has one)."""
        series = ((_require_raw(name).metadata or {}).get("beamline_table") or {}).get("series") or {}
        for key in ("CurrentTable", "current", "beam_current", "BEAM_CURRENT"):
            if key in series and len(series[key]) == len(_require_raw(name).value):
                return stack_edits.injection_frames(np.asarray(series[key], float))
        return []

    def set_topup(self, name: str, factor: float = 1.05, frames: list[int] | None = None) -> dict[str, Any]:
        """Igor TopUpCorrection: frames taken during injection x factor. frames None = detect from
        the ring current; factor 1 or an empty list removes the correction."""
        frames = self.injection_frames(name) if frames is None else [int(i) for i in frames]
        e = self.get_edits(name)
        e["topup"] = {"factor": float(factor), "frames": frames} if frames and float(factor) != 1.0 else None
        out = self._set_edits(name, e)
        out["topup_frames"] = frames
        return out

    def set_reference_frame_op(self, name: str, op: str, frame: int | None) -> dict[str, Any]:
        """Igor NormalizeStackToThisLayer (op='normalize_to') / SubtractThisLayerFromStack
        (op='subtract'); frame None removes it."""
        if op not in ("normalize_to", "subtract"):
            raise ValueError("op must be 'normalize_to' or 'subtract'")
        e = self.get_edits(name)
        e[op] = None if frame is None else int(frame)
        return self._set_edits(name, e)

    def disable_pixels(self, name: str, pixels: list[list[int]], add: bool = True) -> dict[str, Any]:
        """Igor DisablePixel: listed (y, x) pixels -> mean of the diagonal neighbours at +-2 px."""
        e = self.get_edits(name)
        cur = {tuple(p) for p in e["pixel_fixes"]}
        new = {(int(y), int(x)) for y, x in pixels}
        e["pixel_fixes"] = [list(p) for p in sorted(cur | new if add else cur - new)]
        return self._set_edits(name, e)

    def reset_edits(self, name: str) -> dict[str, Any]:
        return self._set_edits(name, {})

    def get_working_frame(self, name: str, index: int) -> np.ndarray:
        frames, _, _ = _working_frames(name)
        return np.asarray(frames[int(index)], dtype=float)
