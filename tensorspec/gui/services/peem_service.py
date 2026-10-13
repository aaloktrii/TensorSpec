"""Direct PEEM core access for the Qt GUI (no HTTP)."""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from tensorspec.core.data_models import TensorData
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


class PeemService:
    """Qt-facing PEEM operations via global_workspace."""

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
