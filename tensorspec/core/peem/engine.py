from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import numpy as np

from tensorspec.core.data_models import TensorData
from tensorspec.core.peem.roi import roi_to_mask

PairMode = Literal["auto", "CP_CM", "LH_LV"]

_VALID_MODES = frozenset({"auto", "CP_CM", "LH_LV"})

_PASSTHROUGH_KEYS = (
    "csv_attached",
    "I0",
    "source",
    "loader",
    "beamline_csv",
    "beamline_table",
)


def _shift_plane(plane: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a 2D plane using edge pixels to fill exposed borders."""
    ny, nx = plane.shape
    source_y = np.clip(np.arange(ny) - dy, 0, ny - 1)
    source_x = np.clip(np.arange(nx) - dx, 0, nx - 1)
    return plane[np.ix_(source_y, source_x)]


def _ncc_best_shift(
    ref_plane: np.ndarray,
    mov_plane: np.ndarray,
    mask: np.ndarray,
    search_radius: int,
) -> tuple[int, int]:
    """Return the integer shift that maximizes masked normalized correlation."""
    mask_y, mask_x = np.nonzero(mask)
    template = np.asarray(ref_plane[mask], dtype=float)
    template_centered = template - template.mean()
    template_norm = np.linalg.norm(template_centered)
    if not np.isfinite(template_norm) or template_norm <= np.finfo(float).eps:
        raise ValueError("reference ROI template must have non-zero variance")

    ny, nx = ref_plane.shape
    best_score = -np.inf
    best_shift: tuple[int, int] | None = None
    for dy in range(-search_radius, search_radius + 1):
        source_y = np.clip(mask_y - dy, 0, ny - 1)
        for dx in range(-search_radius, search_radius + 1):
            source_x = np.clip(mask_x - dx, 0, nx - 1)
            candidate = np.asarray(mov_plane[source_y, source_x], dtype=float)
            candidate_centered = candidate - candidate.mean()
            candidate_norm = np.linalg.norm(candidate_centered)
            if not np.isfinite(candidate_norm) or candidate_norm <= np.finfo(float).eps:
                continue
            score = float(
                np.dot(template_centered, candidate_centered)
                / (template_norm * candidate_norm)
            )
            shift_rank = (dx * dx + dy * dy, abs(dx), abs(dy))
            best_rank = (
                None
                if best_shift is None
                else (
                    best_shift[0] * best_shift[0] + best_shift[1] * best_shift[1],
                    abs(best_shift[0]),
                    abs(best_shift[1]),
                )
            )
            scores_tie = np.isclose(score, best_score, rtol=1e-12, atol=1e-12)
            if score > best_score and not scores_tie:
                best_score = score
                best_shift = (dx, dy)
            elif scores_tie and (best_rank is None or shift_rank < best_rank):
                best_score = max(best_score, score)
                best_shift = (dx, dy)

    if best_shift is None:
        raise ValueError("no valid NCC shift found in search range")
    return best_shift


def _require_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _drift_planes(
    tensor: TensorData,
    track_channel: int,
) -> tuple[np.ndarray, int, bool]:
    if tensor.ndim == 3 and tensor.labels == ["frame", "y", "x"]:
        if track_channel != 0:
            raise ValueError("track_channel must be 0 for raw PEEM tensors")
        return tensor.value, tensor.value.shape[0], False

    if tensor.ndim == 4 and tensor.labels == ["pair", "channel", "y", "x"]:
        n_channels = tensor.value.shape[1]
        if not 0 <= track_channel < n_channels:
            raise ValueError("track_channel out of range")
        return tensor.value[:, track_channel], tensor.value.shape[0], True

    raise ValueError(
        "Expected raw (frame, y, x) or paired (pair, channel, y, x) PEEM tensor"
    )


def drift_correct(
    tensor: TensorData,
    *,
    ref_index: int,
    roi: dict,
    search_radius: int,
    track_channel: int = 0,
) -> TensorData:
    """
    Correct integer translational drift using masked ROI NCC against a reference.

    Paired tensors estimate drift on one channel and apply each shift to all
    channels. Exposed borders use edge-clamp fill.
    """
    ref_index = _require_integer(ref_index, "ref_index")
    search_radius = _require_integer(search_radius, "search_radius")
    track_channel = _require_integer(track_channel, "track_channel")
    if not 1 <= search_radius <= 200:
        raise ValueError("search_radius must be between 1 and 200")

    tracking_planes, n_items, paired = _drift_planes(tensor, track_channel)
    if not 0 <= ref_index < n_items:
        raise ValueError("ref_index out of range")

    ny, nx = tracking_planes.shape[-2:]
    mask = roi_to_mask(ny, nx, roi)
    mask_y, mask_x = np.nonzero(mask)
    bbox_area = (
        (int(mask_y.max()) - int(mask_y.min()) + 1)
        * (int(mask_x.max()) - int(mask_x.min()) + 1)
    )
    if bbox_area < 9:
        raise ValueError("ROI bounding box area must be at least 9 pixels")

    ref_plane = tracking_planes[ref_index]
    # Validate template variance even when this is a one-item stack.
    ref_values = np.asarray(ref_plane[mask], dtype=float)
    ref_norm = np.linalg.norm(ref_values - ref_values.mean())
    if not np.isfinite(ref_norm) or ref_norm <= np.finfo(float).eps:
        raise ValueError("reference ROI template must have non-zero variance")

    shifts: list[tuple[int, int]] = []
    for index, moving_plane in enumerate(tracking_planes):
        if index == ref_index:
            shifts.append((0, 0))
        else:
            shifts.append(
                _ncc_best_shift(ref_plane, moving_plane, mask, search_radius)
            )

    corrected = np.empty_like(tensor.value)
    for index, (dx, dy) in enumerate(shifts):
        if paired:
            for channel in range(tensor.value.shape[1]):
                corrected[index, channel] = _shift_plane(
                    tensor.value[index, channel], dx, dy
                )
        else:
            corrected[index] = _shift_plane(tensor.value[index], dx, dy)

    metadata = deepcopy(tensor.metadata)
    metadata.update(
        {
            "drift_method": "ncc_roi",
            "drift_ref_index": ref_index,
            "drift_roi": deepcopy(roi),
            "drift_search_radius": search_radius,
            "drift_track_channel": track_channel,
            "drift_shifts": [
                {"index": index, "dx": dx, "dy": dy}
                for index, (dx, dy) in enumerate(shifts)
            ],
        }
    )
    return TensorData(
        value=corrected,
        axes=list(tensor.axes),
        labels=list(tensor.labels),
        units=list(tensor.units),
        data_type=tensor.data_type,
        metadata=metadata,
    )


def resolve_pair_mode(pol: list[str], mode: PairMode) -> tuple[str, list[str]]:
    """
    Returns (resolved_mode, channel_tags).
    auto: CP/CM-only → ("CP_CM", ["CP","CM"]); LH/LV-only → ("LH_LV", ["LH","LV"]);
    mixed or none → ValueError.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Unsupported pair mode: {mode!r}")

    if mode == "CP_CM":
        return "CP_CM", ["CP", "CM"]
    if mode == "LH_LV":
        return "LH_LV", ["LH", "LV"]

    has_cp_cm = any(tag in ("CP", "CM") for tag in pol)
    has_lh_lv = any(tag in ("LH", "LV") for tag in pol)

    if has_cp_cm and not has_lh_lv:
        return "CP_CM", ["CP", "CM"]
    if has_lh_lv and not has_cp_cm:
        return "LH_LV", ["LH", "LV"]
    raise ValueError("Cannot auto-resolve pair mode from pol tags")


def _validate_raw(tensor: TensorData) -> list[str]:
    if tensor.ndim != 3 or tensor.labels != ["frame", "y", "x"]:
        raise ValueError("Expected raw PEEM tensor with shape (frame, y, x)")

    pol = tensor.metadata.get("pol")
    if not isinstance(pol, list):
        raise ValueError("metadata['pol'] must be a list")

    n_frames = tensor.value.shape[0]
    if len(pol) < n_frames:
        raise ValueError("metadata['pol'] shorter than frame count")

    return list(pol[:n_frames])


def _frame_names(tensor: TensorData, n_frames: int) -> list[str | None]:
    names = tensor.metadata.get("frame_names")
    if not isinstance(names, list):
        return [None] * n_frames
    out: list[str | None] = []
    for i in range(n_frames):
        name = names[i] if i < len(names) else None
        out.append(name if isinstance(name, str) else None)
    return out


def _pair_indices(
    pol: list[str],
    channel_tags: list[str],
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    queues: dict[str, list[int]] = {tag: [] for tag in channel_tags}
    unpaired: list[dict[str, Any]] = []

    for idx, tag in enumerate(pol):
        if tag in queues:
            queues[tag].append(idx)
        else:
            unpaired.append({"frame_index": idx, "frame_name": None, "pol": tag})

    pairs: list[tuple[int, int]] = []
    tag_a, tag_b = channel_tags
    while queues[tag_a] and queues[tag_b]:
        pairs.append((queues[tag_a].pop(0), queues[tag_b].pop(0)))

    for tag in channel_tags:
        for idx in queues[tag]:
            unpaired.append({"frame_index": idx, "frame_name": None, "pol": tag})

    return pairs, unpaired


def pair_stack(tensor: TensorData, mode: PairMode) -> TensorData:
    """
    Input shape (frame,y,x) + metadata['pol'] (+ optional frame_names).
    Output shape (n_pairs, 2, y, x), data_type "Experimental PEEM (paired)".
    Metadata: pair_mode, channel_tags, pair_sources, unpaired, plus CSV/I0 passthrough.
    """
    pol = _validate_raw(tensor)
    pair_mode, channel_tags = resolve_pair_mode(pol, mode)
    n_frames, y_size, x_size = tensor.value.shape
    frame_names = _frame_names(tensor, n_frames)

    pairs, unpaired = _pair_indices(pol, channel_tags)
    if not pairs:
        raise ValueError("No pairs could be formed for the selected mode")

    for entry in unpaired:
        idx = entry["frame_index"]
        entry["frame_name"] = frame_names[idx]

    pair_sources: list[dict[str, Any]] = []
    stacked = np.empty((len(pairs), 2, y_size, x_size), dtype=float)
    for pair_idx, (idx_a, idx_b) in enumerate(pairs):
        stacked[pair_idx, 0] = tensor.value[idx_a]
        stacked[pair_idx, 1] = tensor.value[idx_b]
        pair_sources.append(
            {
                "pair": pair_idx,
                "channels": [
                    {
                        "tag": channel_tags[0],
                        "frame_index": idx_a,
                        "frame_name": frame_names[idx_a],
                    },
                    {
                        "tag": channel_tags[1],
                        "frame_index": idx_b,
                        "frame_name": frame_names[idx_b],
                    },
                ],
            }
        )

    metadata: dict[str, Any] = {
        "pair_mode": pair_mode,
        "channel_tags": list(channel_tags),
        "pair_sources": pair_sources,
        "unpaired": unpaired,
    }
    for key in _PASSTHROUGH_KEYS:
        if key in tensor.metadata:
            metadata[key] = tensor.metadata[key]

    return TensorData(
        value=stacked,
        axes=[np.arange(len(pairs)), np.arange(2), tensor.axes[1], tensor.axes[2]],
        labels=["pair", "channel", "y", "x"],
        units=["", "", tensor.units[1], tensor.units[2]],
        data_type="Experimental PEEM (paired)",
        metadata=metadata,
    )


_SEPARATE_PASSTHROUGH_KEYS = _PASSTHROUGH_KEYS + (
    "pair_mode",
    "drift_method",
    "drift_shifts",
)


def separate_pairs(tensor: TensorData) -> dict[str, TensorData]:
    if (
        tensor.data_type != "Experimental PEEM (paired)"
        or list(tensor.labels) != ["pair", "channel", "y", "x"]
        or tensor.value.ndim != 4
        or tensor.value.shape[1] != 2
    ):
        raise ValueError(
            "separate_pairs requires a (pair, channel=2, y, x) paired PEEM cube"
        )
    tags = list(tensor.metadata.get("channel_tags") or [])
    if len(tags) != 2 or any(not str(t).strip() or "/" in str(t) for t in tags):
        raise ValueError("channel_tags must be two single-segment names")
    tags = [str(t).strip() for t in tags]
    if tags[0] == tags[1]:
        raise ValueError("channel_tags must be distinct")

    n_pairs, _, y_size, x_size = tensor.value.shape
    out: dict[str, TensorData] = {}
    for ch, tag in enumerate(tags):
        meta: dict[str, Any] = {
            "channel_tag": tag,
            "separated_from": "paired",
        }
        for key in _SEPARATE_PASSTHROUGH_KEYS:
            if key in tensor.metadata:
                meta[key] = tensor.metadata[key]
        out[tag] = TensorData(
            value=np.asarray(tensor.value[:, ch], dtype=float),
            axes=[
                np.arange(n_pairs),
                np.asarray(tensor.axes[2]),
                np.asarray(tensor.axes[3]),
            ],
            labels=["frame", "y", "x"],
            units=["", tensor.units[2], tensor.units[3]],
            data_type=f"Experimental PEEM ({tag})",
            metadata=meta,
        )
    return out
