from __future__ import annotations
from typing import Literal
import numpy as np

DeflPoolMode = Literal["mean", "mid"]


def aggregate_defl_embeddings(
    emb_defl: np.ndarray,
    *,
    mode: DeflPoolMode,
    mid_index: int | None = None,
) -> np.ndarray:
    arr = np.asarray(emb_defl, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"emb_defl must be (n_defl, D); got shape {arr.shape}")
    n_defl = arr.shape[0]
    if mode == "mean":
        return arr.mean(axis=0)
    if mode == "mid":
        idx = n_defl // 2 if mid_index is None else int(mid_index)
        if idx < 0 or idx >= n_defl:
            raise ValueError(f"mid_index {idx} out of range for n_defl={n_defl}")
        return arr[idx].copy()
    raise ValueError(f"unknown mode={mode!r}")
