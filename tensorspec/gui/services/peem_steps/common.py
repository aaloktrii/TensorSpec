"""Shared helpers of the PEEM-service step mixins (no Qt). Moved verbatim from the former
gui/services/peem_xmcd_service.py."""
from __future__ import annotations

import datetime
import json
from typing import Any

import numpy as np

from tensorspec.core.data_models import TensorData
from tensorspec.core.data_tree import DataTreeBuilder
from tensorspec.core.peem.roi import roi_to_mask
from tensorspec.core.peem import dichroism, stack_edits
from tensorspec.core.workspace import global_workspace

_POL_PAIRS = {"pcd": ("CP", "CM"), "pld": ("LH", "LV")}


def _require_raw(name: str) -> TensorData:
    tensor = global_workspace.pull_tensor_data(name, "raw")
    if tensor is None or tensor.data_type != "Experimental PEEM":
        raise ValueError(f"'{name}' is not PEEM data in the workspace.")
    return tensor


def _frames_for_pol(name: str, tag: str, node: str = "raw") -> tuple[np.ndarray, np.ndarray]:
    """Frames whose polarization is ``tag`` and their raw frame indices.

    node: "raw", "processed" (drift-corrected frame cube or paired cube), or
    "processed/<TAG>" (separated channel stack).
    """
    raw = _require_raw(name)
    pol = [str(p) for p in (raw.metadata or {}).get("pol", [])]
    node = node.strip("/")
    if node == "raw":  # stack clean-up edits (excluded frames, top-up, ...) applied on the fly
        frames, wpol, keep = _working_frames(name)
        idx = np.array([i for i, p in enumerate(wpol) if p == tag], dtype=int)
        if idx.size == 0:
            raise ValueError(f"no {tag} frames in '{name}' (tags: {sorted(set(wpol))})")
        return np.asarray(frames[idx], dtype=float), keep[idx]
    elif node == "processed":
        src = global_workspace.pull_tensor_data(name, "processed")
        if src is None:
            raise ValueError(f"'{name}' has no /processed data.")
        if src.value.ndim == 4:
            tags = [str(t) for t in (src.metadata or {}).get("channel_tags", [])]
            if tag not in tags:
                raise ValueError(f"paired cube has channels {tags}, not {tag}")
            return np.asarray(src.value[:, tags.index(tag)], dtype=float), np.array([], int)
    elif node.startswith("processed/"):
        src = global_workspace.pull_tensor_data(name, node)
        if src is None:
            raise ValueError(f"no data at /{node}")
        if node.split("/", 1)[1] != tag:
            raise ValueError(f"/{node} does not hold {tag} frames")
        return np.asarray(src.value, dtype=float), np.array([], int)
    else:
        raise ValueError("node must be 'raw', 'processed' or 'processed/<TAG>'")
    pol = [str(p) for p in (src.metadata or {}).get("pol", pol)]
    if src.value.shape[0] != len(pol):
        raise ValueError("frame count does not match the polarization tags")
    idx = np.array([i for i, p in enumerate(pol) if p == tag], dtype=int)
    if idx.size == 0:
        raise ValueError(f"no {tag} frames in '{name}' (tags: {sorted(set(pol))})")
    return np.asarray(src.value[idx], dtype=float), idx


def _working_frames(name: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Raw frames with the stack clean-up edits applied: (frames, pol tags, kept raw indices)."""
    raw = _require_raw(name)
    md = raw.metadata or {}
    return stack_edits.apply_edits(np.asarray(raw.value), [str(p) for p in md.get("pol", [])], _read_edits(name))


def _read_edits(name: str) -> dict[str, Any]:
    """Stack clean-up edits from /analysis/stack_edits (older sessions kept them in the /raw attrs)."""
    ds = global_workspace.pull_analysis_data(name, "stack_edits")
    if ds is not None and "edits" in ds.attrs:
        return stack_edits.normalized(json.loads(ds.attrs["edits"]))
    return stack_edits.normalized((_require_raw(name).metadata or {}).get("stack_edits"))


def _record(name: str, step: str, info: dict[str, Any]) -> None:
    """Append one PEEM-step line (step + its scalar / short-list parameters) to /history."""
    item = global_workspace.get(name)
    if not item or item.get("type") != "spectroscopy_tree":
        return
    params = {k: v for k, v in info.items()
              if k not in ("name", "step", "next") and v is not None
              and (isinstance(v, (str, int, float, bool)) or (isinstance(v, list) and len(v) <= 8))}
    DataTreeBuilder._append_history(
        item["tree"], f"[{datetime.datetime.now().time()}] PEEM {step}: {json.dumps(params, default=str)}")


def _mask(shape: tuple[int, int], roi: dict | None) -> np.ndarray | None:
    return None if roi is None else roi_to_mask(shape[0], shape[1], roi)


def _box(roi: dict | None) -> tuple[int, int, int, int] | None:
    """Registration box (y0, y1, x0, x1) from a rect ROI dict."""
    if roi is None:
        return None
    if roi.get("kind") != "rect":
        raise ValueError("registration ROI must be a rect")
    return int(roi["y0"]), int(roi["y1"]), int(roi["x0"]), int(roi["x1"])


def _drift_attrs(shape: tuple[int, ...], reg_roi: dict | None, a_shifts, b_shifts) -> dict[str, Any]:
    """Per-frame drift (channel A frames, then B) and the box it was measured in."""
    sh = np.concatenate([np.reshape(a_shifts, (-1, 2)), np.reshape(b_shifts, (-1, 2))])
    box = _box(reg_roi) or dichroism._central_box(shape)
    return {"drift_dy_px": [float(v) for v in sh[:, 0]], "drift_dx_px": [float(v) for v in sh[:, 1]],
            "drift_n_a": int(len(np.reshape(a_shifts, (-1, 2)))),
            "drift_box_y0y1x0x1": [int(v) for v in box],
            "drift_box_source": "drawn drift ROI" if reg_roi else "central half (default)"}


def _beamline_formula(name: str, convention: str) -> str:
    """pcd formula in the beamline's own channel names, e.g. (CR-CL)/(CR+CL) for UView."""
    a, b = ("CM", "CP") if dichroism.CONVENTIONS[convention] < 0 else ("CP", "CM")
    src = (_require_raw(name).metadata or {}).get("pol_source") or {}
    a, b = src.get(a, a), src.get(b, b)
    return f"({a}-{b})/({a}+{b})"
