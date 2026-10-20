"""Save an ARPES tensor as a MATLAB struct named from the tensor's own labels.

``data.value`` is the intensity, in the tensor's axis order. Each axis is a
1×N field whose name is that label (``Energy``, ``kx_Slit``, ``Photon_Energy``).
``data.info`` is the metadata.
"""
from __future__ import annotations

import re

import numpy as np
from scipy.io import savemat

from tensorspec.core.data_models import TensorData

_GREEK = {"Θ": "Theta", "Φ": "Phi", "θ": "theta", "φ": "phi"}


def is_arpes(tensor: TensorData) -> bool:
    kind = str(getattr(tensor, "data_type", "") or "").casefold()
    if "arpes" in kind or "maestro" in kind:
        return True
    blob = " ".join(str(lab) for lab in tensor.labels).casefold()
    return "energy" in blob and any(
        key in blob for key in ("slit", "defl", "kx", "ky", "kz", "theta", "phi", "photon", "angle")
    )


def matlab_field_name(label: str, used: set[str]) -> str:
    text = str(label)
    for src, dst in _GREEK.items():
        text = text.replace(src, dst)
    name = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    if not name or name[0].isdigit():
        name = "axis_" + name
    base = name
    n = 2
    while name in used:
        name = f"{base}_{n}"
        n += 1
    used.add(name)
    return name


def _info(metadata: dict) -> dict:
    raw = dict(metadata or {})
    raw.pop("axis_alternates", None)

    def conv(v):
        if v is None:
            return ""
        if isinstance(v, (bool, np.bool_)):
            return int(bool(v))
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            if not v:
                return np.zeros((0, 1))
            if all(isinstance(i, str) for i in v):
                return np.array(list(v), dtype=object)
            if all(
                isinstance(i, (int, float, np.integer, np.floating)) and not isinstance(i, bool)
                for i in v
            ):
                return np.asarray(v, float).reshape(1, -1)
            return np.array([str(i) for i in v], dtype=object)
        if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, np.ndarray):
            if v.dtype == object:
                return np.array([str(i) for i in v.ravel().tolist()], dtype=object)
            if v.ndim == 1:
                return np.asarray(v, float).reshape(1, -1)
            return np.asarray(v)
        return str(v)

    return {str(k): conv(v) for k, v in raw.items()}


def arpes_matlab_payload(tensor: TensorData) -> dict:
    if not is_arpes(tensor):
        raise ValueError("Save to MATLAB is for ARPES data.")
    value = np.asarray(tensor.value)
    if value.ndim != len(tensor.labels) or value.ndim != len(tensor.axes):
        raise ValueError("ARPES tensor labels do not match the intensity dimensions.")
    used = {"value", "info"}
    payload = {"value": np.ascontiguousarray(value), "info": _info(tensor.metadata)}
    for label, axis, n in zip(tensor.labels, tensor.axes, value.shape):
        coords = np.asarray(axis, float).ravel()
        if coords.size != n:
            raise ValueError(f"axis '{label}' has length {coords.size}, intensity has {n}.")
        payload[matlab_field_name(label, used)] = coords.reshape(1, -1)
    return payload


def save_arpes_matlab(tensor: TensorData, path: str) -> None:
    savemat(path, {"data": arpes_matlab_payload(tensor)}, do_compression=True)
