"""PEEM service, Sum Rule tab: background subtraction and the XMCD sum rule (BG / sum-rule helpers and
the bg_preview / bg_apply / sumrule_preview / sumrule_apply methods).

Mixin of ``PeemService`` (gui/services/peem_service.py). Moved verbatim from gui/services/peem_service.py;
the math is in core/peem/bg.py and core/peem/sumrule.py."""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from tensorspec.core.data_models import TensorData
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
from tensorspec.gui.services.peem_steps.stack_nodes import (_processed_pair_tensor, _processed_tensor, _require_tensor,
                                                             _separated_tensor)


_SUMRULE_PAIRS = (("CP", "CM"), ("LH", "LV"))


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


class PeemBgSumruleSteps:
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
