"""Direct PEEM core access for the Qt GUI (no HTTP)."""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tensorspec.core.data_models import TensorData
from tensorspec.core.io import peem_beamline_loaders
from tensorspec.core.io.peem_loaders import (
    find_beamline_csv,
    load_beamline_csv,
    load_tif_sequence,
    load_tif_stack,
)
from tensorspec.core.peem.engine import drift_correct, pair_stack, separate_pairs
from tensorspec.core.workspace import global_workspace
from tensorspec.gui.services.peem_steps.annotations import PeemAnnotationSteps
from tensorspec.gui.services.peem_steps.bg_sumrule import PeemBgSumruleSteps, _bg_meta_fields, _sumrule_meta_fields
from tensorspec.gui.services.peem_steps.stack_nodes import (_processed_pair_tensor, _processed_tensor, _require_tensor,
                                                             _separated_tensor)
from tensorspec.gui.services.peem_steps.cleanup import PeemCleanupSteps
from tensorspec.gui.services.peem_steps.drift_stack import PeemDriftStackSteps
from tensorspec.gui.services.peem_steps.image_analysis import PeemImageAnalysisSteps
from tensorspec.gui.services.peem_steps.load import PeemLoadSteps
from tensorspec.gui.services.peem_steps.maps_domains import PeemMapsDomainsSteps

MAX_PEEM_BYTES = 512 * 1024 * 1024
MAX_PEEM_FRAMES = 10_000
MAX_CSV_BYTES = 8 * 1024 * 1024
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _safe_label(name: str, fallback: str) -> str:
    cleaned = (name.strip() or fallback)[:64]
    if not SAFE_NAME.match(cleaned):
        raise ValueError(
            "Name must start with a letter or digit and contain only "
            "letters, digits, '.', '_' or '-'."
        )
    return cleaned


def _safe_upload_name(filename: str, fallback: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", Path(filename).name)[:120] or fallback


def _tif_paths(source_path: Path) -> list[Path]:
    if source_path.is_dir():
        candidates = sorted(source_path.glob("*.tif")) + sorted(
            source_path.glob("*.tiff")
        )
        seen: set[str] = set()
        paths: list[Path] = []
        for candidate in candidates:
            key = candidate.name.casefold()
            if key not in seen:
                seen.add(key)
                paths.append(candidate)
        return paths
    if source_path.suffix.lower() in (".tif", ".tiff"):
        return [source_path]
    raise ValueError("Path must be a TIF file or directory.")


def _float64_bytes_for_shape(shape: tuple[int, ...] | list[int]) -> int:
    dims = [int(size) for size in shape]
    if any(size < 0 for size in dims):
        raise ValueError(f"Invalid TIF shape {shape}")
    return math.prod(dims) * 8


def _enforce_peem_size(source_path: Path) -> None:
    paths = _tif_paths(source_path)
    if sum(path.stat().st_size for path in paths) > MAX_PEEM_BYTES:
        raise ValueError("PEEM TIF files exceed 512 MB.")

    n_frames = 0
    float64_bytes = 0
    import tifffile

    for path in paths:
        with tifffile.TiffFile(path) as tif:
            if not tif.pages:
                continue
            if len(tif.pages) == 1:
                shape = tuple(int(size) for size in tif.pages[0].shape)
                n_frames += shape[0] if len(shape) == 3 else 1
                float64_bytes += _float64_bytes_for_shape(shape)
            else:
                n_frames += len(tif.pages)
                float64_bytes += sum(
                    _float64_bytes_for_shape(page.shape) for page in tif.pages
                )

    if n_frames > MAX_PEEM_FRAMES:
        raise ValueError(f"PEEM data exceeds the {MAX_PEEM_FRAMES}-frame limit.")
    if float64_bytes > MAX_PEEM_BYTES:
        raise ValueError("PEEM float64 expansion exceeds the 512 MB limit.")


def _auto_csv_choice(
    directory: Path, preferred_stem: str
) -> tuple[Path | None, list[Path]]:
    candidates = find_beamline_csv(directory)
    if len(candidates) == 1:
        return candidates[0], candidates
    preferred = preferred_stem.casefold()
    matches = [
        path
        for path in candidates
        if path.stem.casefold() == preferred or preferred in path.stem.casefold()
    ]
    return (matches[0] if len(matches) == 1 else None), candidates


def _load_summary(
    name: str,
    tensor: TensorData,
    *,
    csv_prompt: bool = False,
    csv_candidates: list[str] | None = None,
) -> dict[str, Any]:
    metadata = tensor.metadata or {}
    pol = [str(value) for value in metadata.get("pol", [])]
    return {
        "name": name,
        "shape": [int(size) for size in tensor.value.shape],
        "n_frames": int(tensor.value.shape[0]),
        "data_type": tensor.data_type,
        "pol_summary": dict(Counter(pol)),
        "source": str(metadata.get("source", "")),
        "loader": str(metadata.get("loader", "")),
        "csv_attached": bool(metadata.get("csv_attached", False)),
        "I0_present": metadata.get("I0") is not None,
        "csv_prompt": csv_prompt,
        "csv_candidates": csv_candidates or [],
    }


class PeemService(PeemLoadSteps, PeemCleanupSteps, PeemDriftStackSteps, PeemImageAnalysisSteps,
                  PeemMapsDomainsSteps, PeemAnnotationSteps, PeemBgSumruleSteps):
    """Qt-facing PEEM operations via global_workspace.

    The one PEEM service of the PEEM suite: the stack / pair / drift / BG / sum-rule methods below,
    plus the XMCD-imaging steps (UE49 / UView / P3B load, clean-up, drift, stack, asymmetry, image
    analysis, ROIs, domains) inherited from the mixins in gui/services/peem_steps/."""

    def load_path(
        self,
        path: Path,
        name: str | None = None,
        csv_path: Path | str | None = None,
    ) -> dict[str, Any]:
        source_path = Path(path).expanduser().resolve()
        if not source_path.exists():
            raise ValueError(f"Path not found: {source_path}")
        if not _tif_paths(source_path) and peem_beamline_loaders.is_beamline_source(source_path):
            return self._load_beamline_path(source_path, name)

        _enforce_peem_size(source_path)
        load_directory = source_path if source_path.is_dir() else source_path.parent
        fallback = source_path.stem if source_path.is_file() else source_path.name

        explicit_csv: Path | None = None
        if csv_path is not None:
            explicit_csv = Path(csv_path).expanduser().resolve()
            if not explicit_csv.is_file():
                raise ValueError("CSV path must be an existing file.")
            if explicit_csv.suffix.lower() != ".csv":
                raise ValueError("Expected a .csv file.")
            if explicit_csv.stat().st_size > MAX_CSV_BYTES:
                raise ValueError("CSV exceeds the 8 MB limit.")

        auto_candidates: list[Path] = []
        loader_csv = explicit_csv
        if explicit_csv is None:
            preferred_stem = (
                source_path.name if source_path.is_dir() else source_path.stem
            )
            loader_csv, auto_candidates = _auto_csv_choice(
                load_directory, preferred_stem
            )
            if loader_csv is not None and loader_csv.stat().st_size > MAX_CSV_BYTES:
                raise ValueError("Auto-discovered CSV exceeds the 8 MB limit.")

        if source_path.is_dir():
            tensor = load_tif_sequence(source_path, csv_path=loader_csv)
        elif source_path.suffix.lower() in (".tif", ".tiff"):
            tensor = load_tif_stack(source_path, csv_path=loader_csv)
        else:
            raise ValueError("Path must be a TIF file or directory.")

        label = _safe_label(name, _safe_upload_name(fallback, "peem")) if name else _safe_label(
            _safe_upload_name(fallback, "peem"), "peem"
        )
        global_workspace.push_spectroscopy_data(label, tensor)
        attached = bool((tensor.metadata or {}).get("csv_attached", False))
        candidates = (
            []
            if attached or explicit_csv is not None
            else [str(p) for p in auto_candidates]
        )
        return _load_summary(
            label,
            tensor,
            csv_prompt=not attached and explicit_csv is None,
            csv_candidates=candidates,
        )

    def _load_beamline_path(self, source_path: Path, name: str | None) -> dict[str, Any]:
        """UE49 scan / UView (BNL) / P3B (ALS) data through core/io/peem_beamline_loaders. A single-
        polarization UView folder brings its partner (CL<->CR, LH<->LV) so pairing has both."""
        paths = [source_path]
        partner = peem_beamline_loaders.partner_folder(source_path)
        if partner is not None:
            paths.append(partner)
        tensor = peem_beamline_loaders.load_beamline(paths)
        if tensor.value.shape[0] > MAX_PEEM_FRAMES:
            raise ValueError(f"PEEM data exceeds the {MAX_PEEM_FRAMES}-frame limit.")
        fallback = source_path.stem if source_path.is_file() else source_path.name
        label = _safe_label(name, _safe_upload_name(fallback, "peem")) if name else _safe_label(
            _safe_upload_name(fallback, "peem"), "peem")
        global_workspace.push_spectroscopy_data(label, tensor)
        out = _load_summary(label, tensor)
        out["format"] = tensor.metadata.get("format")
        out["partner_added"] = partner.name if partner is not None else None
        return out

    def attach_csv(self, name: str, csv_path: Path | str) -> dict[str, Any]:
        _require_tensor(name)
        selected = Path(csv_path).expanduser().resolve()
        if not selected.is_file():
            raise ValueError("CSV path must be an existing file.")
        if selected.suffix.lower() != ".csv":
            raise ValueError("Expected a .csv file.")
        if selected.stat().st_size > MAX_CSV_BYTES:
            raise ValueError("CSV exceeds the 8 MB limit.")

        csv_meta = load_beamline_csv(selected)
        attrs = {
            "csv_attached": True,
            "beamline_csv": str(selected),
            "I0": csv_meta.get("I0"),
            "beamline_table": {
                "columns": csv_meta.get("columns"),
                "series": csv_meta.get("series"),
            },
            "beam_current": csv_meta.get("beam_current"),
        }
        if not global_workspace.merge_spectroscopy_raw_attrs(name, attrs):
            raise ValueError(f"PEEM data '{name}' not found.")
        return _load_summary(name, _require_tensor(name))

    def pair(self, name: str, mode: str) -> dict[str, Any]:
        tensor = _require_tensor(name)
        paired = pair_stack(tensor, mode)
        if not global_workspace.write_processed_data(name, paired):
            raise ValueError(f"PEEM data '{name}' not found.")
        metadata = paired.metadata or {}
        return {
            "name": name,
            "n_pairs": int(paired.value.shape[0]),
            "channel_tags": [str(v) for v in metadata.get("channel_tags", [])],
            "unpaired_count": len(metadata.get("unpaired", [])),
            "mode": str(metadata.get("pair_mode", mode)),
            "shape": [int(s) for s in paired.value.shape],
            "has_processed": True,
        }

    def separate(self, name: str) -> dict[str, Any]:
        _require_tensor(name)
        paired = _processed_pair_tensor(name)
        if paired is None:
            raise ValueError(
                "Separate requires a paired /processed cube. Run Stack Pairs first."
            )
        channels = separate_pairs(paired)
        for tag, td in channels.items():
            if not global_workspace.write_processed_child_data(name, tag, td):
                raise ValueError(f"PEEM data '{name}' not found.")
        tags = sorted(channels)
        sample = channels[tags[0]]
        return {
            "name": name,
            "channels": tags,
            "n_frames": int(sample.value.shape[0]),
            "shape": [int(s) for s in sample.value.shape],
            "has_separated": True,
        }


    def prepare_sum_rule(self, name: str, mode: str = "auto") -> dict[str, Any]:
        """Pair the CP / CM (or LH / LV) frames and separate the two channels: the input the
        background and sum-rule steps use. mode: auto / CP_CM / LH_LV.
        Returns {n_pairs, unpaired_count, channels}; ValueError when no pairs can be formed."""
        paired = self.pair(name, mode)
        sep = self.separate(name)
        return {"n_pairs": paired["n_pairs"], "unpaired_count": paired.get("unpaired_count", 0),
                "channels": sep["channels"]}
    def drift(
        self,
        name: str,
        ref_index: int,
        roi_dict: dict,
        search_radius: int,
        track_channel: int = 0,
        source: str = "raw",
    ) -> dict[str, Any]:
        if source == "raw":
            tensor = _require_tensor(name)
        else:
            tensor = _processed_tensor(name)
            if tensor is None:
                raise ValueError(f"PEEM data '{name}' has no processed data.")

        corrected = drift_correct(
            tensor,
            ref_index=ref_index,
            roi=roi_dict,
            search_radius=search_radius,
            track_channel=track_channel,
        )
        if not global_workspace.write_processed_data(name, corrected):
            raise ValueError(f"PEEM data '{name}' not found.")

        shifts = corrected.metadata.get("drift_shifts", [])
        return {
            "name": name,
            "source": source,
            "n_planes": int(corrected.value.shape[0]),
            "ref_index": ref_index,
            "search_radius": search_radius,
            "max_abs_dx": max((abs(int(item["dx"])) for item in shifts), default=0),
            "max_abs_dy": max((abs(int(item["dy"])) for item in shifts), default=0),
            "shape": [int(s) for s in corrected.value.shape],
            "has_drift": True,
            "drift_method": corrected.metadata.get("drift_method", "ncc_roi"),
        }


    def get_view_tensor(
        self,
        name: str,
        node: str = "raw",
        frame_index: int = 0,
        channel: int = 0,
    ) -> np.ndarray:
        node = node.strip("/")
        if node == "raw":
            tensor = _require_tensor(name)
            if not 0 <= frame_index < tensor.value.shape[0]:
                raise ValueError(f"Frame index {frame_index} is out of range.")
            return np.asarray(tensor.value[frame_index], dtype=float)
        if node == "processed":
            tensor = _processed_tensor(name)
            if tensor is None:
                raise ValueError(f"PEEM data '{name}' has no processed data.")
            if not 0 <= frame_index < tensor.value.shape[0]:
                kind = "Pair" if tensor.value.ndim == 4 else "Frame"
                raise ValueError(f"{kind} index {frame_index} is out of range.")
            if tensor.value.ndim == 4:
                if not 0 <= channel < tensor.value.shape[1]:
                    raise ValueError(f"Channel index {channel} is out of range.")
                return np.asarray(tensor.value[frame_index, channel], dtype=float)
            return np.asarray(tensor.value[frame_index], dtype=float)
        if node.startswith("processed/"):
            tensor, _tag = _separated_tensor(name, node)
            if not 0 <= frame_index < tensor.value.shape[0]:
                raise ValueError(f"Frame index {frame_index} is out of range.")
            return np.asarray(tensor.value[frame_index], dtype=float)
        raise ValueError("node must be 'raw', 'processed', or 'processed/<tag>'.")

    def get_meta(self, name: str) -> dict[str, Any]:
        tensor = _require_tensor(name)
        metadata = tensor.metadata or {}
        processed = _processed_tensor(name)
        processed_metadata = (processed.metadata or {}) if processed is not None else {}
        processed_is_pair = processed is not None and processed.value.ndim == 4
        processed_is_frame = processed is not None and processed.value.ndim == 3
        bg_meta = _bg_meta_fields(name)
        sumrule_meta = _sumrule_meta_fields(name)
        separated = global_workspace.list_processed_children(name)
        return {
            "name": name,
            "shape": [int(s) for s in tensor.value.shape],
            "labels": list(tensor.labels),
            "n_frames": int(tensor.value.shape[0]),
            "frame_names": [str(v) for v in metadata.get("frame_names", [])],
            "pol": [str(v) for v in metadata.get("pol", [])],
            "csv_attached": bool(metadata.get("csv_attached", False)),
            "I0_present": metadata.get("I0") is not None,
            "I0": metadata.get("I0"),
            "has_processed": processed is not None,
            "processed_shape": (
                [int(s) for s in processed.value.shape] if processed is not None else None
            ),
            "processed_is_paired": processed_is_pair,
            "n_processed_frames": (
                int(processed.value.shape[0]) if processed_is_frame else None
            ),
            "pair_mode": (
                str(processed_metadata.get("pair_mode"))
                if processed_metadata.get("pair_mode") is not None
                else None
            ),
            "n_pairs": int(processed.value.shape[0]) if processed_is_pair else None,
            "channel_tags": [
                str(v) for v in processed_metadata.get("channel_tags", [])
            ],
            "unpaired_count": len(processed_metadata.get("unpaired", [])),
            "has_drift": processed_metadata.get("drift_method") is not None,
            "drift_method": (
                str(processed_metadata["drift_method"])
                if processed_metadata.get("drift_method") is not None
                else None
            ),
            "processed_children": separated,
            "separated_channels": separated,
            "has_background": bg_meta["has_background"],
            "has_processed_bg": bg_meta["has_processed_bg"],
            "energy_source": bg_meta["energy_source"],
            "processed_bg_node": bg_meta["processed_bg_node"],
            "n_bg_frames": bg_meta["n_bg_frames"],
            "has_sumrule": sumrule_meta["has_sumrule"],
            "sumrule_i0_applied": sumrule_meta["sumrule_i0_applied"],
            "sumrule_tags": sumrule_meta["sumrule_tags"],
        }

    @staticmethod
    def default_drift_roi(ny: int, nx: int) -> dict:
        """Center 60% rectangle for drift tracking when UI has no ROI picker."""
        margin_y = max(1, int(ny * 0.2))
        margin_x = max(1, int(nx * 0.2))
        return {
            "kind": "rect",
            "x0": margin_x,
            "y0": margin_y,
            "x1": max(margin_x + 1, nx - margin_x),
            "y1": max(margin_y + 1, ny - margin_y),
        }
