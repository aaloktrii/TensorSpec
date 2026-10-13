"""Helpers of the PEEM service that read a dataset's stack nodes (/raw, /processed, /processed/<channel>),
shared by several steps. Moved verbatim from gui/services/peem_service.py."""
from __future__ import annotations

from tensorspec.core.data_models import TensorData
from tensorspec.core.workspace import global_workspace


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
