"""Drift correction of a PEEM frame stack: one entry point over every drift method (no Qt).

``align_stack`` routes to the engine of the chosen method and returns the aligned frames and
the (dy, dx) shift of every frame (``registration.apply_shift`` convention). Both polarizations
are aligned together onto one reference frame. Moved verbatim from the XMCD-imaging service
(the PEEM service's drift_correct, gui/services/peem_steps/drift_stack.py), which keeps the data-tree reads / writes.

  central / marquee   registration.register_stack (sub-pixel cross-correlation)
  roi                 registration.track_template (NCC template tracking, sub-pixel)
  ncc_roi             engine.drift_correct (the original PEEM-panel drift, whole px)
  igor_*              igor_drift ports of the SPEEM procedures
"""
from __future__ import annotations

from typing import Any

import numpy as np

from tensorspec.core.data_models import TensorData
from tensorspec.core.peem import dichroism, igor_drift, registration

DRIFT_METHODS = {
    "central": "cross-correlation, central half (sub-pixel)",
    "marquee": "cross-correlation in the marquee (sub-pixel)",
    "roi": "ROI template tracking (NCC, sub-pixel)",
    "ncc_roi": "NCC ROI (original TensorSpec engine, whole px)",
    "igor_auto": "Igor autoImageCorrelation (fixed window, whole px)",
    "igor_pair": "Igor driftCorrection (co-add, one CL-CR shift, no per-frame alignment)",
    "igor_marquee": "Igor marqueeDriftCorrection (running reference, passes)",
    "igor_registration": "Igor marqueeImageRegistration (least squares in the marquee)",
}
NEEDS_BOX = ("marquee", "roi", "ncc_roi", "igor_marquee", "igor_registration")
PAIR_TAGS = {"xmcd": ("CP", "CM"), "xld": ("LH", "LV")}


def align_stack(
    frames: np.ndarray,
    pol: list[str],
    method: str,
    box: tuple[int, int, int, int] | None = None,
    *,
    filt: str | None = None,
    ref_index: int = 0,
    search_radius: int = 20,
    max_shift: float | None = None,
    interp_factor: int = 4,
    transform: str = "translation",
    skip: np.ndarray | None = None,
    pair: str = "xmcd",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any], tuple[int, int, int, int]]:
    """Align ``frames`` (n, ny, nx) with ``method``.

    box   (y0, y1, x0, x1) marquee / ROI; required for NEEDS_BOX methods, else the central half
    pol   polarization tag per frame (used by igor_pair)
    skip  per-frame bool, frames left out of the running reference (igor_marquee / registration)

    Returns (aligned, shifts (n, 2) as (dy, dx), NCC score per frame or None, method extras, box used).
    """
    if method not in DRIFT_METHODS:
        raise ValueError(f"method must be one of {sorted(DRIFT_METHODS)}")
    if method in NEEDS_BOX and box is None:
        raise ValueError("Draw the marquee / ROI box first (Draw box).")
    frames = np.asarray(frames, dtype=float)
    n = len(frames)
    if not 0 <= ref_index < n:
        raise ValueError(f"reference frame must be 0..{n - 1} (after excluded frames)")
    box = box if method in NEEDS_BOX else dichroism._central_box(frames.shape)
    score, extra = None, {}
    if method == "roi":
        shifts, score = registration.track_template(frames, ref_index, box, int(search_radius), filt)
        aligned = np.stack([registration.apply_shift(f, *d) for f, d in zip(frames, shifts)])
    elif method == "ncc_roi":  # the original PEEM-panel drift: core/peem/engine.drift_correct
        from tensorspec.core.peem.engine import drift_correct as engine_drift_correct
        y0, y1, x0, x1 = box
        src = TensorData(value=frames, axes=[np.arange(n, dtype=float), np.arange(frames.shape[1], dtype=float),
                                             np.arange(frames.shape[2], dtype=float)],
                         labels=["frame", "y", "x"], units=["", "px", "px"], data_type="PEEM", metadata={})
        res = engine_drift_correct(src, ref_index=ref_index, roi={"kind": "rect", "y0": y0, "y1": y1, "x0": x0, "x1": x1},
                                   search_radius=int(search_radius))
        aligned = np.asarray(res.value, dtype=float)
        shifts = np.array([[s["dy"], s["dx"]] for s in res.metadata["drift_shifts"]], dtype=float)
    elif method in ("central", "marquee"):
        aligned, shifts = registration.register_stack(frames, ref_index, box=box, filt=filt,
                                                     max_shift=max_shift or None)
    elif method == "igor_auto":
        aligned, shifts = igor_drift.stack_auto_correlation(frames, ref_index, filt)
    elif method == "igor_pair":
        ta, tb = PAIR_TAGS[pair]
        ia = [i for i, t in enumerate(pol) if t == ta]
        ib = [i for i, t in enumerate(pol) if t == tb]
        if not ia or not ib:
            raise ValueError(f"driftCorrection needs both {ta} and {tb} frames")
        # autoCoaddXMCD co-adds without shifting; driftCorrection then gives ONE shift of B onto A
        d = igor_drift.drift_correction_pair(frames[ia].mean(axis=0), frames[ib].mean(axis=0),
                                             filt if filt not in (None, "none") else "find_edges")
        shifts = np.zeros((n, 2))
        shifts[ib] = d
        aligned = np.stack([registration.apply_shift(f, *d) if i in set(ib) else f for i, f in enumerate(frames)])
        extra = {"pair_shift_dy_dx": [float(d[0]), float(d[1])],
                 "note": "no per-frame alignment (as Igor autoCoaddXMCD + driftCorrection): use for "
                         "stable acquisitions; for drifting stacks use a per-frame method"}
    elif method == "igor_marquee":
        aligned, shifts, passes = igor_drift.marquee_drift_correction(
            frames, box, ref_index, filt, int(interp_factor), skip)
        extra = {"passes": passes, "interp_factor": int(interp_factor)}
    else:
        aligned, shifts, params = igor_drift.marquee_image_registration(frames, box, ref_index, transform, skip)
        ex = [p.get("extra", 0.0) for p in params]
        extra = {"transform": transform,
                 "rotation_deg" if transform == "rotation" else f"{transform}_param":
                     [round(float(np.degrees(v) if transform == "rotation" else v), 4) for v in ex]}
    return aligned, np.asarray(shifts, dtype=float), score, extra, tuple(int(v) for v in box)
