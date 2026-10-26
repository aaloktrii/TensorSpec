"""Direct PEEM core access for the Qt GUI (no HTTP)."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import xarray as xr

from tensorspec.core.data_models import TensorData
from tensorspec.core.io import peem_beamline_loaders
from tensorspec.core.io.peem_loaders import (
    find_beamline_csv,
    load_beamline_csv,
    load_tif_sequence,
    load_tif_stack,
)
from tensorspec.core.peem.bg import (
    analysis_dataset as bg_analysis_dataset,
    apply_bg_to_stack,
    bg_child_name,
    ensemble_background,
    extract_spectrum,
    fit_background,
    is_bg_output_node,
    resolve_energy,
)
from tensorspec.core.peem import dichroism, products
from tensorspec.core.peem import domain_analysis as da
from tensorspec.core.peem.products import _coords, _flat_attrs
from tensorspec.core.peem.engine import drift_correct, pair_stack, separate_pairs
from tensorspec.core.peem.roi import roi_to_mask
from tensorspec.core.peem.sumrule import (
    analysis_dataset as sumrule_analysis_dataset,
    apply_i0,
    ensemble_sumrule,
    integrate_windows,
    moments,
    pick_source_kind,
)
from tensorspec.core.workspace import global_workspace
# XPEEM map tab: its steps are mixins (gui/services/peem_steps/); the stack / BG / sum-rule methods
# below are unchanged
from tensorspec.gui.services.peem_steps.annotations import PeemAnnotationSteps
from tensorspec.gui.services.peem_steps.cleanup import PeemCleanupSteps
from tensorspec.gui.services.peem_steps.drift_stack import PeemDriftStackSteps
from tensorspec.gui.services.peem_steps.image_analysis import PeemImageAnalysisSteps
from tensorspec.gui.services.peem_steps.load import PeemLoadSteps
from tensorspec.gui.services.peem_steps.maps_domains import PeemMapsDomainsSteps
from tensorspec.gui.services.peem_steps.channels import PeemChannelSteps
from tensorspec.gui.services.peem_steps.compare import PeemCompareSteps
from tensorspec.gui.services.peem_steps.common import _mask as _roi_mask, _record as _step_record

MAX_PEEM_BYTES = 512 * 1024 * 1024
MAX_PEEM_FRAMES = 10_000
MAX_CSV_BYTES = 8 * 1024 * 1024
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SUMRULE_PAIRS = (("CP", "CM"), ("LH", "LV"))


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


def _is_beamline_folder(path: Path) -> bool:
    try:
        peem_beamline_loaders.detect_format(path)
        return True
    except ValueError:
        return False


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


def _require_tensor(name: str) -> TensorData:
    tensor = global_workspace.pull_tensor_data(name, "raw")
    if tensor is None:
        raise ValueError(f"'{name}' is not spectroscopy data in the workspace.")
    if tensor.data_type != "Experimental PEEM":
        raise ValueError(f"'{name}' is not PEEM data.")
    return tensor


def _processed_tensor(name: str) -> TensorData | None:
    tensor = global_workspace.pull_tensor_data(name, "processed")
    if tensor is None:
        return None
    shape = tuple(int(size) for size in tensor.value.shape)
    valid_raw = (
        tensor.data_type == "Experimental PEEM"
        and tensor.labels == ["frame", "y", "x"]
        and tensor.value.ndim == 3
        and len(shape) == 3
        and all(size > 0 for size in shape)
    )
    valid_pair = (
        tensor.data_type == "Experimental PEEM (paired)"
        and tensor.labels == ["pair", "channel", "y", "x"]
        and tensor.value.ndim == 4
        and len(shape) == 4
        and shape[1] == 2
        and all(size > 0 for size in shape)
    )
    if not (valid_raw or valid_pair):
        raise ValueError(
            "Processed PEEM data must be a non-empty (frame, y, x) cube or "
            "(pair, channel=2, y, x) paired cube."
        )
    return tensor


def _processed_pair_tensor(name: str) -> TensorData | None:
    tensor = _processed_tensor(name)
    if tensor is None:
        return None
    if tensor.value.ndim != 4:
        raise ValueError(
            "Processed PEEM data must be a non-empty (pair, channel=2, y, x) paired cube."
        )
    return tensor


def _separated_tensor(name: str, node: str) -> tuple[TensorData, str]:
    rel = node.strip("/")
    if not rel.startswith("processed/") or rel.count("/") != 1:
        raise ValueError("Invalid separated node path.")
    tensor = global_workspace.pull_tensor_data(name, rel)
    if tensor is None:
        raise ValueError(f"No data at '{rel}'.")
    tag = rel.split("/", 1)[1]
    ok = (
        tensor.value.ndim == 3
        and list(tensor.labels) == ["frame", "y", "x"]
        and tensor.data_type.startswith("Experimental PEEM")
    )
    if not ok:
        raise ValueError(f"Invalid channel stack at '{rel}'.")
    return tensor, tag


def _default_ensemble_delta(energy: np.ndarray, energy_source: str) -> float:
    if energy_source == "index":
        return 1.0
    span = float(np.max(energy) - np.min(energy))
    return 0.05 * span if span > 0 else 1.0


def _pull_bg_source(
    name: str, node: str, channel: int
) -> tuple[np.ndarray, TensorData, TensorData]:
    raw = _require_tensor(name)
    node = node.strip("/")
    if is_bg_output_node(node):
        raise ValueError(
            f"node '{node}' is a background output; "
            "use raw, processed, or a non-bg separated channel."
        )
    if node == "raw":
        stack = np.asarray(raw.value, dtype=float)
        if stack.ndim != 3:
            raise ValueError("Raw PEEM data must be 3D.")
        return stack, raw, raw
    if node == "processed":
        processed = _processed_tensor(name)
        if processed is None:
            raise ValueError(f"PEEM data '{name}' has no processed data.")
        if processed.value.ndim == 4:
            if not 0 <= channel < processed.value.shape[1]:
                raise ValueError(f"Channel index {channel} is out of range.")
            stack = np.asarray(processed.value[:, channel], dtype=float)
        elif processed.value.ndim == 3:
            stack = np.asarray(processed.value, dtype=float)
        else:
            raise ValueError("Invalid processed PEEM shape.")
        return stack, processed, raw
    if node.startswith("processed/"):
        tensor, _tag = _separated_tensor(name, node)
        stack = np.asarray(tensor.value, dtype=float)
        return stack, tensor, raw
    raise ValueError("node must be 'raw', 'processed', or 'processed/<tag>'.")


def _bg_roi_mask(
    use_roi: bool, roi_dict: dict | None, ny: int, nx: int
) -> np.ndarray | None:
    if not use_roi:
        return None
    if roi_dict is None:
        raise ValueError("roi required when use_roi is true.")
    return roi_to_mask(ny, nx, roi_dict)


class _BgFitResult(NamedTuple):
    stack: np.ndarray
    source_tensor: TensorData
    energy: np.ndarray
    energy_source: str
    spectrum: np.ndarray
    fit: dict
    ensemble: dict
    delta: float
    roi_dict: dict | None


def _run_bg_fit(
    name: str,
    *,
    node: str = "raw",
    channel: int = 0,
    method: str = "linear",
    e0: float = 0.0,
    e1: float = 1.0,
    post_e0: float | None = None,
    post_e1: float | None = None,
    use_roi: bool = False,
    roi_dict: dict | None = None,
    ensemble_delta: float | None = None,
    ensemble_n: int = 21,
    seed: int | None = None,
) -> _BgFitResult:
    stack, source_tensor, raw = _pull_bg_source(name, node, channel)
    energy, energy_source = resolve_energy(stack.shape[0], raw.metadata or {})
    mask = _bg_roi_mask(use_roi, roi_dict, stack.shape[1], stack.shape[2])
    spectrum = extract_spectrum(stack, mask)
    fit = fit_background(
        method,
        energy,
        spectrum,
        e0=e0,
        e1=e1,
        post_e0=post_e0,
        post_e1=post_e1,
    )
    delta = ensemble_delta
    if delta is None:
        delta = _default_ensemble_delta(energy, energy_source)
    ensemble = ensemble_background(
        method,
        energy,
        spectrum,
        e0=e0,
        e1=e1,
        post_e0=post_e0,
        post_e1=post_e1,
        delta=delta,
        n=ensemble_n,
        seed=seed,
    )
    roi_out = roi_dict if use_roi and roi_dict else None
    return _BgFitResult(
        stack=stack,
        source_tensor=source_tensor,
        energy=energy,
        energy_source=energy_source,
        spectrum=spectrum,
        fit=fit,
        ensemble=ensemble,
        delta=delta,
        roi_dict=roi_out,
    )


def _bg_fit_to_dict(
    *,
    energy: np.ndarray,
    spectrum: np.ndarray,
    fit: dict,
    ensemble: dict,
    energy_source: str,
    e0: float,
    e1: float,
    post_e0: float | None = None,
    post_e1: float | None = None,
) -> dict[str, Any]:
    method = str(fit.get("method", "linear"))
    out: dict[str, Any] = {
        "energy": np.asarray(energy, dtype=float),
        "spectrum": np.asarray(spectrum, dtype=float),
        "bg": np.asarray(ensemble["bg_mean"], dtype=float),
        "subtracted": np.asarray(ensemble["subtracted_mean"], dtype=float),
        "method": method,
        "energy_source": energy_source,
        "e0": float(e0),
        "e1": float(e1),
        "ensemble_n_valid": int(ensemble["n_valid"]),
    }
    if method == "two_step":
        out.update(
            post_e0=float(post_e0 if post_e0 is not None else fit["post_e0"]),
            post_e1=float(post_e1 if post_e1 is not None else fit["post_e1"]),
            pre_slope=float(fit["pre_slope"]),
            pre_intercept=float(fit["pre_intercept"]),
            post_slope=float(fit["post_slope"]),
            post_intercept=float(fit["post_intercept"]),
        )
    else:
        out.update(
            slope=float(fit["slope"]),
            intercept=float(fit["intercept"]),
        )
    return out


def _bg_subtracted_tensor(
    source_tensor: TensorData,
    subtracted: np.ndarray,
    *,
    child_name: str,
    source_node: str,
    channel: int,
) -> TensorData:
    n, _y, _x = subtracted.shape
    if source_tensor.value.ndim == 4:
        y_axis = np.asarray(source_tensor.axes[2])
        x_axis = np.asarray(source_tensor.axes[3])
        y_unit = source_tensor.units[2]
        x_unit = source_tensor.units[3]
    else:
        y_axis = np.asarray(source_tensor.axes[1])
        x_axis = np.asarray(source_tensor.axes[2])
        y_unit = source_tensor.units[1]
        x_unit = source_tensor.units[2]

    meta = dict(source_tensor.metadata or {})
    meta.update(
        {
            "bg_subtracted": True,
            "bg_source_node": source_node,
            "bg_channel": int(channel),
            "analysis_node": "background",
            "bg_child": child_name,
        }
    )
    return TensorData(
        value=subtracted,
        axes=[np.arange(n), y_axis, x_axis],
        labels=["frame", "y", "x"],
        units=["", y_unit, x_unit],
        data_type="Experimental PEEM (bg subtracted)",
        metadata=meta,
    )


def _bg_meta_fields(name: str) -> dict:
    analysis = global_workspace.pull_analysis_data(name, "background")
    if analysis is None:
        return {
            "has_background": False,
            "has_processed_bg": False,
            "energy_source": None,
            "processed_bg_node": None,
            "n_bg_frames": None,
        }
    attrs = analysis.attrs or {}
    energy_source = attrs.get("energy_source")
    source_node = str(attrs.get("source_node", "raw"))
    try:
        child = bg_child_name(source_node)
    except ValueError:
        child = "bg"
    processed_bg_node = f"processed/{child}"
    children = global_workspace.list_processed_children(name)
    has_processed_bg = child in children
    n_bg_frames: int | None = None
    if has_processed_bg:
        bg_tensor = global_workspace.pull_tensor_data(name, processed_bg_node)
        if bg_tensor is not None and bg_tensor.value.ndim >= 1:
            n_bg_frames = int(bg_tensor.value.shape[0])
    return {
        "has_background": True,
        "has_processed_bg": has_processed_bg,
        "energy_source": str(energy_source) if energy_source is not None else None,
        "processed_bg_node": processed_bg_node,
        "n_bg_frames": n_bg_frames,
    }


def _sumrule_tags(name: str) -> tuple[str, str]:
    processed = _processed_tensor(name)
    if processed is None:
        raise ValueError(
            "Sum rule requires paired or separated CP/CM (or LH/LV) stacks."
        )
    channel_tags = processed.metadata.get("channel_tags", [])
    if len(channel_tags) >= 2:
        return (str(channel_tags[0]), str(channel_tags[1]))
    children = set(global_workspace.list_processed_children(name))
    for pair in _SUMRULE_PAIRS:
        if pair[0] in children and pair[1] in children:
            return pair
    if processed.value.ndim == 4:
        return ("CP", "CM")
    raise ValueError("Cannot resolve sum-rule channel pair (CP/CM or LH/LV).")


def _sumrule_available_nodes(name: str) -> list[str]:
    nodes = [
        f"processed/{tag}" for tag in global_workspace.list_processed_children(name)
    ]
    processed = _processed_tensor(name)
    if processed is not None and processed.value.ndim == 4:
        nodes.append("processed")
    return nodes


def _resolve_sumrule_stacks(
    name: str,
    tags: tuple[str, str],
    source_kind: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, str], str]:
    t0, t1 = tags
    if source_kind == "bg":
        plus_tensor, _ = _separated_tensor(name, f"processed/{t0}_bg")
        minus_tensor, _ = _separated_tensor(name, f"processed/{t1}_bg")
        plus_stack = np.asarray(plus_tensor.value, dtype=float)
        minus_stack = np.asarray(minus_tensor.value, dtype=float)
    elif source_kind == "separated":
        plus_tensor, _ = _separated_tensor(name, f"processed/{t0}")
        minus_tensor, _ = _separated_tensor(name, f"processed/{t1}")
        plus_stack = np.asarray(plus_tensor.value, dtype=float)
        minus_stack = np.asarray(minus_tensor.value, dtype=float)
    else:
        paired = _processed_pair_tensor(name)
        plus_stack = np.asarray(paired.value[:, 0], dtype=float)
        minus_stack = np.asarray(paired.value[:, 1], dtype=float)
    return plus_stack, minus_stack, tags, source_kind


def _sumrule_roi_mask(
    use_roi: bool, roi_dict: dict | None, ny: int, nx: int
) -> np.ndarray | None:
    if not use_roi:
        return None
    if roi_dict is None:
        raise ValueError("roi required when use_roi is true.")
    return roi_to_mask(ny, nx, roi_dict)


def _sumrule_i0(
    raw_metadata: dict, n_frames: int, spectrum: np.ndarray
) -> tuple[np.ndarray, bool]:
    i0 = (raw_metadata or {}).get("I0")
    if i0 is None:
        return apply_i0(spectrum, None)
    if isinstance(i0, list):
        if len(i0) != n_frames:
            return apply_i0(spectrum, None)
        return apply_i0(spectrum, np.asarray(i0, dtype=float))
    return apply_i0(spectrum, i0)


def _sumrule_bg_params(
    name: str, energy: np.ndarray, energy_source: str
) -> tuple[str, float | None, float | None, float | None, float | None, float, int]:
    analysis = global_workspace.pull_analysis_data(name, "background")
    if analysis is None:
        return "linear", None, None, None, None, 0.0, 1
    attrs = analysis.attrs or {}
    e0 = attrs.get("e0")
    e1 = attrs.get("e1")
    if e0 is None or e1 is None:
        return "linear", None, None, None, None, 0.0, 1
    bg_delta = attrs.get("ensemble_delta")
    if bg_delta is None:
        bg_delta = _default_ensemble_delta(energy, energy_source)
    else:
        bg_delta = float(bg_delta)
    bg_n = int(attrs.get("ensemble_n", 21))
    method = str(attrs.get("method", "linear"))
    post_e0 = attrs.get("post_e0")
    post_e1 = attrs.get("post_e1")
    post_e0_f = float(post_e0) if post_e0 is not None else None
    post_e1_f = float(post_e1) if post_e1 is not None else None
    return method, float(e0), float(e1), post_e0_f, post_e1_f, bg_delta, bg_n


class _SumruleResult(NamedTuple):
    energy: np.ndarray
    energy_source: str
    mu_plus: np.ndarray
    mu_minus: np.ndarray
    i0_applied: bool
    source_kind: str
    tags: tuple[str, str]
    integrals: dict[str, float]
    moment_vals: dict[str, float]
    ensemble: dict
    window_delta: float
    bg_e0: float | None
    bg_e1: float | None
    bg_method: str
    bg_post_e0: float | None
    bg_post_e1: float | None
    bg_delta: float
    bg_n: int
    roi_dict: dict | None
    nh: float
    l3_lo: float
    l3_hi: float
    l2_lo: float
    l2_hi: float
    r_lo: float
    r_hi: float


def _run_sumrule(
    name: str,
    *,
    nh: float = 1.0,
    l3_lo: float = 0.0,
    l3_hi: float = 1.0,
    l2_lo: float = 2.0,
    l2_hi: float = 3.0,
    r_lo: float = 0.0,
    r_hi: float = 4.0,
    use_roi: bool = False,
    roi_dict: dict | None = None,
    window_delta: float | None = None,
    window_n: int = 21,
    bg_delta: float | None = None,
    seed: int | None = None,
) -> _SumruleResult:
    raw = _require_tensor(name)
    tags = _sumrule_tags(name)
    source_kind = pick_source_kind(_sumrule_available_nodes(name), tags)

    plus_stack, minus_stack, tags, source_kind = _resolve_sumrule_stacks(
        name, tags, source_kind
    )
    if plus_stack.shape != minus_stack.shape:
        raise ValueError("Plus/minus stacks have mismatched shape.")
    if plus_stack.ndim != 3:
        raise ValueError("Sum-rule stacks must be 3D.")

    energy, energy_source = resolve_energy(plus_stack.shape[0], raw.metadata or {})
    mask = _sumrule_roi_mask(use_roi, roi_dict, plus_stack.shape[1], plus_stack.shape[2])
    mu_plus = extract_spectrum(plus_stack, mask)
    mu_minus = extract_spectrum(minus_stack, mask)

    mu_plus, i0_plus = _sumrule_i0(raw.metadata or {}, plus_stack.shape[0], mu_plus)
    mu_minus, i0_minus = _sumrule_i0(raw.metadata or {}, minus_stack.shape[0], mu_minus)
    i0_applied = i0_plus and i0_minus

    l3 = (l3_lo, l3_hi)
    l2 = (l2_lo, l2_hi)
    r_win = (r_lo, r_hi)
    if window_delta is None:
        window_delta = _default_ensemble_delta(energy, energy_source)

    bg_method, bg_e0, bg_e1, bg_post_e0, bg_post_e1, bg_delta_default, bg_n_default = (
        _sumrule_bg_params(name, energy, energy_source)
    )
    if source_kind == "bg":
        bg_e0, bg_e1, bg_post_e0, bg_post_e1 = None, None, None, None
    bg_delta_val = bg_delta if bg_delta is not None else bg_delta_default
    bg_n = bg_n_default if bg_e0 is not None else 1

    integrals = integrate_windows(energy, mu_plus, mu_minus, l3=l3, l2=l2, r_win=r_win)
    moment_vals = moments(integrals["p"], integrals["q"], integrals["r"], nh)
    ensemble = ensemble_sumrule(
        energy,
        mu_plus,
        mu_minus,
        l3=l3,
        l2=l2,
        r_win=r_win,
        nh=nh,
        window_delta=window_delta,
        window_n=window_n,
        bg_e0=bg_e0,
        bg_e1=bg_e1,
        bg_method=bg_method,
        bg_post_e0=bg_post_e0,
        bg_post_e1=bg_post_e1,
        bg_delta=bg_delta_val,
        bg_n=bg_n,
        seed=seed,
    )

    roi_out = roi_dict if use_roi and roi_dict else None
    return _SumruleResult(
        energy=energy,
        energy_source=energy_source,
        mu_plus=mu_plus,
        mu_minus=mu_minus,
        i0_applied=i0_applied,
        source_kind=source_kind,
        tags=tags,
        integrals=integrals,
        moment_vals=moment_vals,
        ensemble=ensemble,
        window_delta=window_delta,
        bg_e0=bg_e0,
        bg_e1=bg_e1,
        bg_method=bg_method,
        bg_post_e0=bg_post_e0,
        bg_post_e1=bg_post_e1,
        bg_delta=bg_delta_val,
        bg_n=bg_n,
        roi_dict=roi_out,
        nh=nh,
        l3_lo=l3[0],
        l3_hi=l3[1],
        l2_lo=l2[0],
        l2_hi=l2[1],
        r_lo=r_win[0],
        r_hi=r_win[1],
    )


def _sumrule_to_dict(result: _SumruleResult) -> dict[str, Any]:
    ens = result.ensemble
    d_mu = result.mu_plus - result.mu_minus
    return {
        "energy": np.asarray(result.energy, dtype=float),
        "mu_plus": np.asarray(result.mu_plus, dtype=float),
        "mu_minus": np.asarray(result.mu_minus, dtype=float),
        "dichroism": np.asarray(d_mu, dtype=float),
        "nh": float(result.nh),
        "l3_lo": float(result.l3_lo),
        "l3_hi": float(result.l3_hi),
        "l2_lo": float(result.l2_lo),
        "l2_hi": float(result.l2_hi),
        "r_lo": float(result.r_lo),
        "r_hi": float(result.r_hi),
        "p": float(ens["p_mean"]),
        "q": float(ens["q_mean"]),
        "r": float(ens["r_mean"]),
        "p_std": float(ens["p_std"]),
        "q_std": float(ens["q_std"]),
        "r_std": float(ens["r_std"]),
        "m_orb": float(ens["m_orb_mean"]),
        "m_orb_std": float(ens["m_orb_std"]),
        "m_spin_plus_dipole": float(ens["m_spin_plus_dipole_mean"]),
        "m_spin_plus_dipole_std": float(ens["m_spin_plus_dipole_std"]),
        "i0_applied": result.i0_applied,
        "source_kind": result.source_kind,
        "tag_plus": result.tags[0],
        "tag_minus": result.tags[1],
        "energy_source": result.energy_source,
        "ensemble_n_valid": int(ens["n_valid"]),
        "ensemble_n_valid_bg": int(ens.get("n_valid_bg", 0)),
    }


def _sumrule_meta_fields(name: str) -> dict:
    analysis = global_workspace.pull_analysis_data(name, "sumrule")
    if analysis is None:
        return {
            "has_sumrule": False,
            "sumrule_i0_applied": None,
            "sumrule_tags": [],
        }
    attrs = analysis.attrs or {}
    tags: list[str] = []
    tag_plus = attrs.get("tag_plus")
    tag_minus = attrs.get("tag_minus")
    if tag_plus is not None and tag_minus is not None:
        tags = [str(tag_plus), str(tag_minus)]
    return {
        "has_sumrule": True,
        "sumrule_i0_applied": bool(attrs.get("i0_applied", False)),
        "sumrule_tags": tags,
    }


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


# ------------------------------------------------------------------ spatial domain analysis
# Maps pushed from the Map view (raw values or as shown, with ROI and background region) -> one node
# each, /analysis/domains_<n>: the pushed map, roi / bg masks and, after a run, the steps of
# core/peem/domain_analysis (the SDA toolkit pipeline) with the numbers in attrs.
DOMAIN_PREFIX = "domains_"
DOMAIN_TABLE_KEYS = ("f_up", "f_down", "mz", "p_up", "p_neutral", "p_down", "n_domains_up", "n_domains_down",
              "mean_domain_area_um2", "width_up", "width_down", "period", "wall_length_um",
              "wall_density_per_um", "stripe_angle_deg", "alignment", "flake_area_um2")


class PeemDomainAnalysisSteps:
    # ------------------------------------------------------------------ push / list
    def push_to_domains(self, name: str, kind: str, var: str = "data", mode: str = "raw",
                        vmin: float | None = None, vmax: float | None = None, gamma: float = 1.0,
                        use_roi: bool = True, use_bg: bool = False, cmap: str | None = None) -> dict[str, Any]:
        """Send a map to the domain analysis. mode 'raw': its values; 'shown': with the Map view's
        colour range (vmin, vmax) and gamma. The dataset's ROI and (use_bg) background region go along.
        The look (vmin, vmax, gamma, cmap) is kept either way, so the pushed map is shown as it was."""
        if mode not in ("raw", "shown"):
            raise ValueError("mode must be 'raw' or 'shown'")
        if mode == "shown" and (vmin is None or vmax is None or not vmax > vmin):
            raise ValueError("'as shown' needs the colour range of the Map view")
        a = np.asarray(self.get_map(name, kind, var), float)
        if a.ndim != 2:
            raise ValueError("only 2-D maps can be pushed")
        rois = self.get_rois(name)
        roi = rois["roi"] if use_roi else None
        bg = rois["bg"] if use_bg else None
        if use_bg and bg is None:
            raise ValueError("Background region is ticked but none is drawn")
        used = [int(k[len(DOMAIN_PREFIX):]) for k in global_workspace.list_analysis_children(name)
                if k.startswith(DOMAIN_PREFIX) and k[len(DOMAIN_PREFIX):].isdigit()]
        label = f"{DOMAIN_PREFIX}{max(used, default=0) + 1}"
        extra = {}
        if roi is not None:
            extra["roi"] = _roi_mask(a.shape, roi).astype(np.uint8)
        if bg is not None:
            extra["bg"] = _roi_mask(a.shape, bg).astype(np.uint8)
        attrs = {"source_kind": kind, "source_var": var, "mode": mode, "gamma": float(gamma or 1.0),
                 "vmin": vmin, "vmax": vmax, "cmap": cmap, "analysed": 0, "um_per_px": self._um_per_px(name)}
        self._write_domains(name, label, a, extra, attrs)
        out = {"name": name, "label": label, "source": self.layer_for(name, kind, var), "mode": mode,
               "has_roi": roi is not None, "has_bg": bg is not None}
        _step_record(name, "push to domain analysis", out)
        return out

    def _write_domains(self, name: str, label: str, data: np.ndarray, extra: dict, attrs: dict) -> None:
        coords = _coords(data.shape, attrs.get("um_per_px"))
        variables = {"data": xr.DataArray(np.asarray(data), dims=("y", "x"), coords=coords)}
        for k, v in extra.items():
            variables[k] = xr.DataArray(np.asarray(v), dims=("y", "x"), coords=coords)
        if not global_workspace.write_analysis_data(name, label, xr.Dataset(variables, attrs=_flat_attrs(attrs))):
            raise ValueError(f"PEEM data '{name}' not found.")

    def _domains(self, name: str, label: str) -> xr.Dataset:
        ds = global_workspace.pull_analysis_data(name, label)
        if ds is None or not label.startswith(DOMAIN_PREFIX):
            raise ValueError(f"no pushed map {label!r} in '{name}'")
        return ds

    def domain_inputs(self) -> list[dict[str, Any]]:
        """Every pushed map of every loaded dataset, in push order."""
        out = []
        for name in self.list_datasets():
            for label in sorted((k for k in global_workspace.list_analysis_children(name) if k.startswith(DOMAIN_PREFIX)),
                                key=lambda k: int(k[len(DOMAIN_PREFIX):]) if k[len(DOMAIN_PREFIX):].isdigit() else 0):
                at = dict(self._domains(name, label).attrs)
                src = self.layer_name(at.get("source_kind", "?"), at.get("source_var", "data"))
                out.append({"name": name, "label": label, "source": src, "mode": at.get("mode", "raw"),
                            "analysed": bool(at.get("analysed", 0)), "title": f"{name} · {src} ({at.get('mode')})"})
        return out

    def remove_domain_input(self, name: str, label: str) -> None:
        self._domains(name, label)
        global_workspace.remove_analysis_data(name, label)

    # ------------------------------------------------------------------ run
    def run_domain_analysis(self, name: str, label: str, **params) -> dict[str, Any]:
        """Steps a-l on one pushed map (params: core/peem/domain_analysis.DEFAULTS; flake 'pushed' uses
        the pushed ROI, background 'region' the pushed background region). Returns the numbers."""
        ds = self._domains(name, label)
        at = dict(ds.attrs)
        a = np.asarray(ds["data"].values, float)
        roi = np.asarray(ds["roi"].values, bool) if "roi" in ds else None
        bg = np.asarray(ds["bg"].values, bool) if "bg" in ds else None
        um = at.get("um_per_px")
        res = da.analyze(a, um_per_px=um if um and np.isfinite(um) else None, roi=roi, bg_mask=bg,
                         normalize_mode=at.get("mode", "raw"), vmin=at.get("vmin"), vmax=at.get("vmax"),
                         gamma=at.get("gamma", 1.0), **params)
        extra = {k: ds[k].values for k in ("roi", "bg") if k in ds}
        extra.update(res["layers"])
        keep = {k: at[k] for k in ("source_kind", "source_var", "mode", "gamma", "vmin", "vmax", "cmap", "um_per_px")
                if k in at}
        p = {**da.DEFAULTS, **params}
        attrs = {**keep, "analysed": 1, **{f"param_{k}": v for k, v in p.items()},
                 **{k: v for k, v in res["metrics"].items()},
                 "areas_unit": res["areas"]["unit"], "areas_up": [float(v) for v in res["areas"]["up"]],
                 "areas_down": [float(v) for v in res["areas"]["down"]]}
        self._write_domains(name, label, a, extra, attrs)
        out = {"name": name, "label": label, **res["metrics"]}
        _step_record(name, f"domain analysis {label}", {k: v for k, v in out.items() if np.isscalar(v)})
        return out

    def domain_look(self, name: str, label: str) -> dict[str, Any]:
        """How the pushed map looked in the Map view: cmap, vmin, vmax, gamma, um_per_px, ROI / background masks."""
        ds = self._domains(name, label)
        at = dict(ds.attrs)
        um = at.get("um_per_px")
        return {"cmap": at.get("cmap"), "vmin": at.get("vmin"), "vmax": at.get("vmax"), "gamma": at.get("gamma", 1.0),
                "um_per_px": um if um and np.isfinite(um) else None,
                "roi": np.asarray(ds["roi"].values, bool) if "roi" in ds else None,
                "bg": np.asarray(ds["bg"].values, bool) if "bg" in ds else None}

    def domain_result(self, name: str, label: str) -> dict[str, Any]:
        """The numbers of one analysed map ({} before a run)."""
        at = dict(self._domains(name, label).attrs)
        return {k: v for k, v in at.items() if np.isscalar(v) and not k.startswith("param_")} if at.get("analysed") else {}

    def domain_layer(self, name: str, label: str, step: str) -> np.ndarray:
        """One pipeline step of an analysed map (core/peem/domain_analysis.STEPS), or 'input'."""
        ds = self._domains(name, label)
        var = "data" if step == "input" else step
        if var not in ds:
            raise ValueError(f"run the domain analysis on {label} first" if step in da.STEPS else f"unknown step {step!r}")
        return np.asarray(ds[var].values)

    def domain_areas(self, name: str, label: str) -> dict[str, Any]:
        """Areas of the single domains (domain-size distribution): {"up", "down", "unit"}."""
        at = dict(self._domains(name, label).attrs)
        if not at.get("analysed"):
            raise ValueError(f"run the domain analysis on {label} first")
        return {"up": np.asarray(at.get("areas_up", []), float), "down": np.asarray(at.get("areas_down", []), float),
                "unit": at.get("areas_unit", "px2")}

    def domain_table(self, items: list[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
        """One row per analysed map: dataset, map, mode and the numbers (widths / period in nm or px)."""
        rows = []
        for it in self.domain_inputs():
            if items is not None and (it["name"], it["label"]) not in items:
                continue
            r = self.domain_result(it["name"], it["label"])
            if not r:
                continue
            unit = r.get("width_unit", "px")
            row = {"dataset": it["name"], "map": it["source"], "push": it["mode"], "label": it["label"]}
            for k in DOMAIN_TABLE_KEYS:
                v = r.get(f"{k}_{unit}", r.get(k))
                row[f"{k} ({unit})" if k in ("width_up", "width_down", "period") else k] = v
            rows.append(row)
        return rows

    def export_domain_table(self, path: str | Path, items: list[tuple[str, str]] | None = None) -> str:
        rows = self.domain_table(items)
        if not rows:
            raise ValueError("no analysed maps to export")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return str(path)

    def export_domain_areas(self, path: str | Path, name: str, label: str) -> str:
        ar = self.domain_areas(name, label)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["state", f"area ({ar['unit']})"])
            w.writerows([("up", v) for v in ar["up"]] + [("down", v) for v in ar["down"]])
        return str(path)

    def export_domain_figure(self, path: str | Path, name: str, label: str, step: str = "three_states") -> str:
        """Editable PDF (TrueType text) of one step with axes in um, titled with the dataset and numbers."""
        from tensorspec.plotting import peem_plots

        at = dict(self._domains(name, label).attrs)
        img = self.domain_layer(name, label, step)
        um = at.get("um_per_px")
        title = f"{name} · {self.layer_name(at.get('source_kind', '?'), at.get('source_var', 'data'))}"
        ds = self._domains(name, label)
        flake = np.asarray(ds["flake"].values) if "flake" in ds else None
        peem_plots.save_domain_step_pdf(path, img, step, um if um and np.isfinite(um) else None, title,
                                        self.domain_result(name, label), flake, self.domain_look(name, label))
        return str(path)

    # ------------------------------------------------------------------ ROI helpers
    def auto_flake_roi(self, name: str, kind: str, var: str = "data", tv_weight: float = 0.20,
                       min_px: int = 800) -> dict[str, Any]:
        """Find the flake on a map (denoise, local std, Otsu, closing, largest region) and set it as
        the dataset's ROI (a polygon, so it can be saved, edited and reused)."""
        a = np.asarray(self.get_map(name, kind, var), float)
        img, _ = da.normalize(a, "raw")
        mask = da.auto_flake(da.local_std(da.denoise(img, "tv", tv_weight)), min_px=min_px,
                             valid=da.edge_strips(a))  # not the drift strips at the field's edges
        roi = {"kind": "polygon", "points": da.flake_outline(mask)}
        self.set_roi(name, "roi", roi)
        return roi

    def save_roi(self, name: str, path: str | Path, which: str = "roi") -> str:
        """The ROI (or background region) as JSON, with the pixel size and map shape, to reuse later."""
        roi = self.get_rois(name)[which]
        if roi is None:
            raise ValueError(f"no {'ROI' if which == 'roi' else 'background region'} to save")
        Path(path).write_text(json.dumps({"tensorspec_roi": 1, "which": which, "roi": roi, "source": name,
                                          "um_per_px": self._um_per_px(name)}, indent=1))
        return str(path)

    def load_roi(self, name: str, path: str | Path, which: str = "roi") -> dict[str, Any]:
        """Set a saved ROI / background region on this dataset (pixel coordinates are kept as saved)."""
        d = json.loads(Path(path).read_text())
        roi = d.get("roi", d) if isinstance(d, dict) else None
        if not isinstance(roi, dict) or roi.get("kind") not in ("rect", "polygon", "ellipse"):
            raise ValueError("not a saved TensorSpec ROI (.json)")
        self.set_roi(name, which, roi)
        return roi


class PeemService(PeemLoadSteps, PeemCleanupSteps, PeemDriftStackSteps, PeemImageAnalysisSteps,
                  PeemMapsDomainsSteps, PeemAnnotationSteps, PeemChannelSteps, PeemCompareSteps,
                  PeemDomainAnalysisSteps):
    """Qt-facing PEEM operations via global_workspace.

    The XPEEM map tab's steps (drift_correct, stack, asymmetry, maps, ROIs ...) and the Sum Rule
    window's ROI spectra come from the mixins in gui/services/peem_steps/ (contract:
    peem_steps/__init__.py)."""

    def load_path(
        self,
        path: Path,
        name: str | None = None,
        csv_path: Path | str | None = None,
    ) -> dict[str, Any]:
        source_path = Path(path).expanduser().resolve()
        if not source_path.exists():
            raise ValueError(f"Path not found: {source_path}")

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

    def load_beamline(
        self,
        paths: list[str | Path],
        name: str | None = None,
        *,
        pol: dict[str, str] | None = None,
        i0_normalize: bool | None = None,
        tpx3: bool = True,
        drop_bad_frames: bool = False,
        allow_mixed_fov: bool = False,
    ) -> dict[str, Any]:
        """UE49 / UView / P3B → workspace via ``peem_beamline_loaders.load_beamline``.

        Returns the same summary shape as ``load_path`` so the PEEM panel load-finished
        path works unchanged. Pair / drift / BG / sum rule stay manual on your panel.
        """
        if not paths:
            raise ValueError("Select at least one beamline file or folder.")
        resolved = [Path(p).expanduser().resolve() for p in paths]
        for p in resolved:
            if not p.exists():
                raise ValueError(f"Path not found: {p}")
            # Size gate for TIF stacks / folders only (.dat / .P3B / .txt skip it); a folder that is
            # a UE49 scan, UView or P3B acquisition is beamline data (loaded as float32), not a TIF stack
            if p.suffix.lower() in (".tif", ".tiff") or (p.is_dir() and not _is_beamline_folder(p)):
                _enforce_peem_size(p)

        # one loader for both PEEM tabs (peem_steps/load.py): UE49 detector maps, /history, automatic name
        loaded = PeemLoadSteps.load_beamline(
            self, resolved, name, pol=pol, i0_normalize=i0_normalize, tpx3=tpx3,
            drop_bad_frames=drop_bad_frames, allow_mixed_fov=allow_mixed_fov)
        tensor = _require_tensor(loaded["name"])
        summary = _load_summary(loaded["name"], tensor, csv_prompt=False, csv_candidates=[])
        summary.update(loaded)  # + format, beamline, pol_counts, fov_um, um_per_px, energy_range ...
        return summary

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

    def bg_preview(
        self,
        name: str,
        node: str = "raw",
        channel: int = 0,
        method: str = "linear",
        e0: float = 0.0,
        e1: float = 1.0,
        post_e0: float | None = None,
        post_e1: float | None = None,
        use_roi: bool = False,
        roi_dict: dict | None = None,
        ensemble_delta: float | None = None,
        ensemble_n: int = 21,
    ) -> dict[str, Any]:
        _require_tensor(name)
        result = _run_bg_fit(
            name,
            node=node,
            channel=channel,
            method=method,
            e0=e0,
            e1=e1,
            post_e0=post_e0,
            post_e1=post_e1,
            use_roi=use_roi,
            roi_dict=roi_dict,
            ensemble_delta=ensemble_delta,
            ensemble_n=ensemble_n,
        )
        return _bg_fit_to_dict(
            energy=result.energy,
            spectrum=result.spectrum,
            fit=result.fit,
            ensemble=result.ensemble,
            energy_source=result.energy_source,
            e0=e0,
            e1=e1,
            post_e0=post_e0,
            post_e1=post_e1,
        )

    def bg_apply(
        self,
        name: str,
        node: str = "raw",
        channel: int = 0,
        method: str = "linear",
        e0: float = 0.0,
        e1: float = 1.0,
        post_e0: float | None = None,
        post_e1: float | None = None,
        use_roi: bool = False,
        roi_dict: dict | None = None,
        ensemble_delta: float | None = None,
        ensemble_n: int = 21,
    ) -> dict[str, Any]:
        _require_tensor(name)
        result = _run_bg_fit(
            name,
            node=node,
            channel=channel,
            method=method,
            e0=e0,
            e1=e1,
            post_e0=post_e0,
            post_e1=post_e1,
            use_roi=use_roi,
            roi_dict=roi_dict,
            ensemble_delta=ensemble_delta,
            ensemble_n=ensemble_n,
        )
        subtracted = apply_bg_to_stack(result.stack, result.ensemble["bg_mean"])

        ds = bg_analysis_dataset(
            result.energy,
            result.spectrum,
            result.fit,
            result.ensemble,
            e0=e0,
            e1=e1,
            post_e0=post_e0,
            post_e1=post_e1,
            energy_source=result.energy_source,
            source_node=node,
            channel=channel,
            use_roi=use_roi,
            roi=result.roi_dict,
            ensemble_delta=result.delta,
            ensemble_n=ensemble_n,
        )
        if not global_workspace.write_analysis_data(name, "background", ds):
            raise ValueError(f"PEEM data '{name}' not found.")

        child = bg_child_name(node)
        bg_tensor = _bg_subtracted_tensor(
            result.source_tensor,
            subtracted,
            child_name=child,
            source_node=node,
            channel=channel,
        )
        if not global_workspace.write_processed_child_data(name, child, bg_tensor):
            raise ValueError(f"PEEM data '{name}' not found.")

        shape = [int(s) for s in subtracted.shape]
        return {
            "name": name,
            "processed_bg_node": f"processed/{child}",
            "n_frames": shape[0],
            "shape": shape,
            "energy_source": result.energy_source,
            "has_background": True,
            "has_processed_bg": True,
        }

    def sumrule_preview(
        self,
        name: str,
        nh: float = 1.0,
        l3_lo: float = 0.0,
        l3_hi: float = 1.0,
        l2_lo: float = 2.0,
        l2_hi: float = 3.0,
        r_lo: float = 0.0,
        r_hi: float = 4.0,
        use_roi: bool = False,
        roi_dict: dict | None = None,
        window_delta: float | None = None,
        window_n: int = 21,
        bg_delta: float | None = None,
    ) -> dict[str, Any]:
        _require_tensor(name)
        return _sumrule_to_dict(
            _run_sumrule(
                name,
                nh=nh,
                l3_lo=l3_lo,
                l3_hi=l3_hi,
                l2_lo=l2_lo,
                l2_hi=l2_hi,
                r_lo=r_lo,
                r_hi=r_hi,
                use_roi=use_roi,
                roi_dict=roi_dict,
                window_delta=window_delta,
                window_n=window_n,
                bg_delta=bg_delta,
            )
        )

    def sumrule_apply(
        self,
        name: str,
        nh: float = 1.0,
        l3_lo: float = 0.0,
        l3_hi: float = 1.0,
        l2_lo: float = 2.0,
        l2_hi: float = 3.0,
        r_lo: float = 0.0,
        r_hi: float = 4.0,
        use_roi: bool = False,
        roi_dict: dict | None = None,
        window_delta: float | None = None,
        window_n: int = 21,
        bg_delta: float | None = None,
    ) -> dict[str, Any]:
        _require_tensor(name)
        result = _run_sumrule(
            name,
            nh=nh,
            l3_lo=l3_lo,
            l3_hi=l3_hi,
            l2_lo=l2_lo,
            l2_hi=l2_hi,
            r_lo=r_lo,
            r_hi=r_hi,
            use_roi=use_roi,
            roi_dict=roi_dict,
            window_delta=window_delta,
            window_n=window_n,
            bg_delta=bg_delta,
        )
        ens = result.ensemble
        ds = sumrule_analysis_dataset(
            result.energy,
            result.mu_plus,
            result.mu_minus,
            integrals={
                "p": ens["p_mean"],
                "q": ens["q_mean"],
                "r": ens["r_mean"],
            },
            integral_stds={
                "p": ens["p_std"],
                "q": ens["q_std"],
                "r": ens["r_std"],
            },
            moment_vals={
                "m_orb": ens["m_orb_mean"],
                "m_spin_plus_dipole": ens["m_spin_plus_dipole_mean"],
            },
            moment_stds={
                "m_orb": ens["m_orb_std"],
                "m_spin_plus_dipole": ens["m_spin_plus_dipole_std"],
            },
            ensemble=ens,
            nh=nh,
            l3=(l3_lo, l3_hi),
            l2=(l2_lo, l2_hi),
            r_win=(r_lo, r_hi),
            i0_applied=result.i0_applied,
            source_kind=result.source_kind,
            tags=result.tags,
            window_delta=result.window_delta,
            window_n=window_n,
            bg_e0=result.bg_e0,
            bg_e1=result.bg_e1,
            bg_delta=result.bg_delta if result.bg_e0 is not None else None,
            bg_n=result.bg_n if result.bg_e0 is not None else None,
            use_roi=use_roi,
            roi=result.roi_dict,
        )
        ds.attrs["energy_source"] = result.energy_source
        if not global_workspace.write_analysis_data(name, "sumrule", ds):
            raise ValueError(f"PEEM data '{name}' not found.")

        return {
            "name": name,
            "i0_applied": result.i0_applied,
            "source_kind": result.source_kind,
            "tag_plus": result.tags[0],
            "tag_minus": result.tags[1],
            "energy_source": result.energy_source,
            "has_sumrule": True,
        }

    @staticmethod
    def _um_per_px(name: str) -> float | None:
        md = (_require_tensor(name).metadata or {})
        return md.get("um_per_px")

    @staticmethod
    def _pol_stacks(
        name: str,
        tags: tuple[str, str],
        source: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Return (stack_a, stack_b, source_used) for XMCD/XLD map making.

        source: ``separated`` (/processed/CP|CM), ``raw`` (pol tags on frames),
        ``processed`` (drift-corrected cube with pol tags), or ``auto``.
        """
        ta, tb = tags
        children = set(global_workspace.list_processed_children(name))
        if source == "auto":
            if ta in children and tb in children:
                source = "separated"
            else:
                raw = _require_tensor(name)
                if any(str(p) in (ta, tb) for p in (raw.metadata or {}).get("pol", [])):
                    source = "raw"
                else:
                    proc = _processed_tensor(name)
                    pmd = (proc.metadata or {}) if proc is not None else {}
                    if proc is not None and any(
                        str(p) in (ta, tb) for p in pmd.get("pol", [])
                    ):
                        source = "processed"
                    else:
                        raise ValueError(
                            f"Need {ta}/{tb}: run Separate, or load beamline data "
                            "with polarization tags."
                        )

        if source == "separated":
            a, _ = _separated_tensor(name, f"processed/{ta}")
            b, _ = _separated_tensor(name, f"processed/{tb}")
            return np.asarray(a.value, float), np.asarray(b.value, float), source

        if source in ("raw", "processed"):
            tensor = (
                _require_tensor(name)
                if source == "raw"
                else _processed_tensor(name)
            )
            if tensor is None:
                raise ValueError(f"No {source} data for '{name}'.")
            if tensor.value.ndim != 3:
                raise ValueError(
                    f"{source} must be a frame stack (n,y,x) with pol tags; "
                    "for paired 4-D cubes run Separate first."
                )
            pol = [str(p) for p in (tensor.metadata or {}).get("pol", [])]
            if len(pol) != tensor.value.shape[0]:
                raise ValueError(f"{source} has no per-frame pol tags.")
            ia = [i for i, p in enumerate(pol) if p == ta]
            ib = [i for i, p in enumerate(pol) if p == tb]
            if not ia or not ib:
                raise ValueError(f"{source} missing frames tagged {ta} and/or {tb}.")
            v = np.asarray(tensor.value, float)
            return v[ia], v[ib], source

        raise ValueError("source must be auto, separated, raw, or processed.")

    def make_pcd_map(
        self,
        name: str,
        *,
        source: str = "auto",
        register: bool = True,
        convention: str | None = None,
    ) -> dict[str, Any]:
        """Build /analysis/pcd from CP/CM stacks and return a short summary."""
        if convention is None:
            convention = (
                (_require_tensor(name).metadata or {}).get("default_convention")
                or dichroism.UNIVERSAL_CONVENTION
            )
        cp, cm, used = self._pol_stacks(name, ("CP", "CM"), source)
        res = dichroism.pcd_map(cp, cm, convention=convention, register=register)
        ds = products.map_dataset(
            "pcd",
            res["map"],
            extra={"mean_cp": res["mean_cp"], "mean_cm": res["mean_cm"]},
            attrs={
                "convention": convention,
                "formula": dichroism.XMCD_FORMULA,
                "n_cp": int(cp.shape[0]),
                "n_cm": int(cm.shape[0]),
                "registered": bool(register),
                "source_node": used,
            },
            um_per_px=self._um_per_px(name),
            long_name="PEEM circular dichroism asymmetry",
        )
        if not global_workspace.write_analysis_data(name, "pcd", ds):
            raise ValueError(f"PEEM data '{name}' not found.")
        a = res["map"]
        finite = a[np.isfinite(a)]
        return {
            "name": name,
            "kind": "pcd",
            "formula": dichroism.XMCD_FORMULA,
            "convention": convention,
            "source": used,
            "registered": bool(register),
            "shape": [int(s) for s in a.shape],
            "mean": float(np.nanmean(finite)) if finite.size else float("nan"),
            "display_node": "analysis/pcd",
        }

    def make_pld_map(
        self,
        name: str,
        *,
        source: str = "auto",
        register: bool = True,
    ) -> dict[str, Any]:
        """Build /analysis/pld from LH/LV stacks."""
        lh, lv, used = self._pol_stacks(name, ("LH", "LV"), source)
        res = dichroism.pld_map(lh, lv, register=register)
        ds = products.map_dataset(
            "pld",
            res["map"],
            extra={"mean_lh": res["mean_lh"], "mean_lv": res["mean_lv"]},
            attrs={
                "formula": dichroism.XLD_FORMULA,
                "n_lh": int(lh.shape[0]),
                "n_lv": int(lv.shape[0]),
                "registered": bool(register),
                "source_node": used,
            },
            um_per_px=self._um_per_px(name),
            long_name="PEEM linear dichroism asymmetry",
        )
        if not global_workspace.write_analysis_data(name, "pld", ds):
            raise ValueError(f"PEEM data '{name}' not found.")
        a = res["map"]
        finite = a[np.isfinite(a)]
        return {
            "name": name,
            "kind": "pld",
            "formula": dichroism.XLD_FORMULA,
            "source": used,
            "registered": bool(register),
            "shape": [int(s) for s in a.shape],
            "mean": float(np.nanmean(finite)) if finite.size else float("nan"),
            "display_node": "analysis/pld",
        }

    def export_analysis_map(
        self, name: str, kind: str, path: str | Path, *, png: bool = True
    ) -> str:
        """Write /analysis/<kind> ``data`` as float TIFF (+ optional colour PNG)."""
        import tifffile
        from matplotlib import pyplot as plt

        kind = kind.strip().strip("/")
        if kind.startswith("analysis/"):
            kind = kind.split("/", 1)[1]
        ds = global_workspace.pull_analysis_data(name, kind)
        if ds is None or "data" not in ds:
            raise ValueError(f"No /analysis/{kind} map on '{name}'.")
        arr = np.asarray(ds["data"].values, dtype=np.float32)
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(out, arr)
        if png:
            png_path = out.with_suffix(".png")
            finite = arr[np.isfinite(arr)]
            vmax = float(np.nanpercentile(np.abs(finite), 99)) if finite.size else 1.0
            if vmax <= 0:
                vmax = 1.0
            fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
            im = ax.imshow(arr, cmap="bwr", vmin=-vmax, vmax=vmax, origin="upper")
            ax.set_title(f"{name} / {kind}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.savefig(png_path, bbox_inches="tight")
            plt.close(fig)
        return str(out)

    def get_display_tensor(self, name: str, node: str = "raw") -> TensorData:
        """TensorData for DataViewer: raw / processed / processed/<tag> / analysis/<kind>."""
        node = (node or "raw").strip("/")
        if node in products.ANALYSIS_KINDS:
            node = f"analysis/{node}"
        if node.startswith("analysis/"):
            td = global_workspace.pull_tensor_data(name, node)
            if td is None:
                raise ValueError(f"No {node} on '{name}'.")
            return td
        if node == "raw":
            return _require_tensor(name)
        if node == "processed":
            td = _processed_tensor(name)
            if td is None:
                raise ValueError(f"PEEM data '{name}' has no processed data.")
            return td
        if node.startswith("processed/"):
            td, _ = _separated_tensor(name, node)
            return td
        raise ValueError(f"Unknown display node {node!r}.")

    def list_analysis_maps(self, name: str) -> list[str]:
        _require_tensor(name)
        return [
            k
            for k in products.ANALYSIS_KINDS
            if global_workspace.pull_analysis_data(name, k) is not None
        ]

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
            "analysis_maps": self.list_analysis_maps(name),
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
