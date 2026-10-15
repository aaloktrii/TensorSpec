"""xarray Datasets for the dichroism / domain outputs under ``/analysis/<kind>``.

Each Dataset has a ``data`` variable on (y, x) so ``pull_tensor_data`` and the
generic DataViewer can open it like any other node; secondary images ride along
as extra variables and scalars go into attrs (sum-rule nodes do the same).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

# /analysis/<kind> products of the XMCD-imaging workflow (stack: per-polarization images)
ANALYSIS_KINDS = ("stack", "pcd", "pld", "pmld", "pmnld", "domain_map")



def _coords(shape: tuple[int, int], um_per_px: float | None) -> dict:
    ny, nx = shape
    coords = {
        "y": ("y", np.arange(ny, dtype=float), {"units": "px"}),
        "x": ("x", np.arange(nx, dtype=float), {"units": "px"}),
    }
    if um_per_px:
        coords["y_um"] = ("y", np.arange(ny) * um_per_px, {"units": "um"})
        coords["x_um"] = ("x", np.arange(nx) * um_per_px, {"units": "um"})
    return coords


def _flat_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, (tuple, list)) and all(np.isscalar(x) for x in v):
            out[k] = list(v)
        elif isinstance(v, dict):
            out.update(_flat_attrs({f"{k}_{kk}": vv for kk, vv in v.items()}))
        elif isinstance(v, np.generic):
            out[k] = v.item()
        elif np.isscalar(v):
            out[k] = v
    return out


def map_dataset(
    kind: str,
    data: np.ndarray,
    *,
    extra: dict[str, np.ndarray] | None = None,
    attrs: dict[str, Any] | None = None,
    um_per_px: float | None = None,
    long_name: str | None = None,
) -> xr.Dataset:
    if kind not in ANALYSIS_KINDS:
        raise ValueError(f"kind must be one of {ANALYSIS_KINDS}")
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"{kind} map must be 2-D, got {data.shape}")
    coords = _coords(data.shape, um_per_px)
    variables = {
        "data": xr.DataArray(data, dims=("y", "x"), coords=coords, name="intensity",
                             attrs={"long_name": long_name or f"PEEM {kind}"}),
    }
    for name, arr in (extra or {}).items():
        arr = np.asarray(arr)
        if arr.shape == data.shape:
            variables[name] = xr.DataArray(arr, dims=("y", "x"), coords=coords)
    meta = {"kind": kind, "um_per_px": um_per_px}
    meta.update(attrs or {})
    return xr.Dataset(variables, attrs=_flat_attrs(meta))


def stats_table(ds: xr.Dataset) -> dict[str, Any]:
    """Scalar attrs of an analysis node, for display / CSV export."""
    return {k: v for k, v in ds.attrs.items() if np.isscalar(v)}
