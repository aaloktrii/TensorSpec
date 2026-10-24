"""
MATLAB ARPES Data Loader

Handles both legacy .mat files (v4–v7.2, via scipy.io.loadmat) and
HDF5-based .mat files (v7.3+, via h5py). Identifies the main data array
as the largest numerical variable and extracts coordinate axes when present.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class MatlabLoader:
    """Load MATLAB .mat files containing ARPES or generic N-D data."""

    def __init__(self, filepath: str, element_name: str | None = None):
        self.filepath = str(filepath)
        self.filename = os.path.basename(self.filepath)
        self.element_name = element_name

    def load(self) -> dict:
        if not self.filename.lower().endswith(".mat"):
            raise ValueError("Not a MATLAB file (.mat)")

        if _is_hdf5_mat(self.filepath):
            return self._load_hdf5()
        return self._load_scipy()

    @classmethod
    def list_elements(cls, filepath: str) -> dict[str, tuple[int, ...]]:
        """List all available data arrays in .mat file with their shapes.

        Returns:
            dict: {element_name: shape} for all numerical arrays
        """
        filename = os.path.basename(filepath)

        if _is_hdf5_mat(filepath):
            return cls._list_hdf5_elements(filepath)
        return cls._list_scipy_elements(filepath)

    @classmethod
    def _list_scipy_elements(cls, filepath: str) -> dict[str, tuple[int, ...]]:
        """List elements in legacy .mat files (v4–v7.2)."""
        from scipy.io import loadmat
        from scipy.io.matlab._mio5_params import mat_struct

        mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)
        elements = {}
        for k, v in mat.items():
            if k.startswith("_"):
                continue
            try:
                # Check if it's a MATLAB struct or numerical array
                if isinstance(v, mat_struct):
                    # MATLAB struct - include it
                    elements[k] = getattr(v, 'shape', ())
                else:
                    arr = np.asarray(v)
                    # Include numerical arrays
                    if arr.dtype.kind in ("f", "i", "u", "c", "V"):
                        elements[k] = arr.shape
            except Exception:
                pass
        return elements

    @classmethod
    def _list_hdf5_elements(cls, filepath: str) -> dict[str, tuple[int, ...]]:
        """List elements in HDF5-based .mat files (v7.3+)."""
        import h5py

        elements = {}
        with h5py.File(filepath, "r") as f:
            for k in f.keys():
                if k.startswith("#"):
                    continue
                try:
                    v = f[k]
                    if isinstance(v, h5py.Dataset):
                        arr = np.asarray(v)
                        # Include numerical arrays and structured types
                        if arr.dtype.kind in ("f", "i", "u", "c", "V"):
                            elements[k] = arr.shape
                except Exception:
                    pass
        return elements

    # ------------------------------------------------------------------
    # Legacy format (v4–v7.2)
    # ------------------------------------------------------------------

    def _load_scipy(self) -> dict:
        from scipy.io import loadmat
        from scipy.io.matlab._mio5_params import mat_struct

        mat = loadmat(self.filepath, squeeze_me=True, struct_as_record=False)
        variables = {k: v for k, v in mat.items() if not k.startswith("_")}

        # Load specific element if requested, otherwise load largest
        if self.element_name:
            if self.element_name not in variables:
                available = list(variables.keys())
                raise ValueError(f"Element '{self.element_name}' not found. Available: {available}")
            data_key = self.element_name
            data = variables[data_key]
            # If element is a mat_struct, extract 'value' field
            if isinstance(data, mat_struct):
                data = data.value
        else:
            data_key, data = _largest_array(variables)
            if data is None:
                raise ValueError(f"No numerical array found in {self.filename}")

        data = np.asarray(data, dtype=np.float64)
        # When element_name is specified, extract axes from the struct fields if present
        if self.element_name and isinstance(variables.get(self.element_name), mat_struct):
            struct = variables[self.element_name]
            # Convert struct fields to dict for axis extraction
            struct_dict = {name: getattr(struct, name) for name in struct._fieldnames}
            axes, labels, units = _extract_axes(struct_dict, data.shape, 'value')
        else:
            axes, labels, units = _extract_axes(variables, data.shape, data_key)

        mode = _guess_mode(variables, data.shape)
        print(f"[{self.filename}] Loaded via scipy (v4–v7.2), shape={data.shape}, element={data_key}")

        # Use element name if specified, otherwise use filename
        dataset_name = self.element_name if self.element_name else Path(self.filename).stem

        return {
            "name": dataset_name,
            "data": np.ascontiguousarray(data),
            "axes": axes,
            "labels": labels,
            "units": units,
            "mode": mode,
            "is_fixed": True,
            "facility": "MATLAB",
            "metadata": {k: _scalar_value(v) for k, v in variables.items()
                         if k != data_key and np.asarray(v).size == 1},
        }

    # ------------------------------------------------------------------
    # HDF5-based format (v7.3+)
    # ------------------------------------------------------------------

    def _load_hdf5(self) -> dict:
        import h5py

        with h5py.File(self.filepath, "r") as f:
            variables = {k: f[k] for k in f.keys() if not k.startswith("#")}

            # Load specific element if requested, otherwise load largest
            if self.element_name:
                if self.element_name not in variables:
                    available = list(variables.keys())
                    raise ValueError(f"Element '{self.element_name}' not found. Available: {available}")
                data_key = self.element_name
                data_ref = variables[data_key]
            else:
                data_key, data_ref = _largest_hdf5_dataset(variables)
                if data_ref is None:
                    raise ValueError(f"No numerical dataset found in {self.filename}")

            data = np.asarray(data_ref, dtype=np.float64)
            # MATLAB stores arrays column-major; transpose to C-order
            data = data.T if data.ndim > 1 else data

            scalar_meta = {}
            array_vars: dict[str, np.ndarray] = {}
            for k, v in variables.items():
                if k == data_key:
                    continue
                arr = np.asarray(v).squeeze()
                if arr.size == 1:
                    scalar_meta[k] = float(arr)
                elif arr.ndim == 1:
                    array_vars[k] = arr if arr.dtype.kind == "f" else arr.astype(float)

            axes, labels, units = _extract_axes(array_vars, data.shape, data_key)
            mode = _guess_mode_from_keys(set(variables.keys()), data.shape)
            print(f"[{self.filename}] Loaded via h5py (v7.3), shape={data.shape}, element={data_key}")

            # Use element name if specified, otherwise use filename
            dataset_name = self.element_name if self.element_name else Path(self.filename).stem

            return {
                "name": dataset_name,
                "data": np.ascontiguousarray(data),
                "axes": axes,
                "labels": labels,
                "units": units,
                "mode": mode,
                "is_fixed": True,
                "facility": "MATLAB",
                "metadata": scalar_meta,
            }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_hdf5_mat(filepath: str) -> bool:
    """Detect v7.3 .mat files by their HDF5 magic bytes."""
    try:
        with open(filepath, "rb") as fh:
            return fh.read(8) == b"\x89HDF\r\n\x1a\n"
    except OSError:
        return False


def _largest_array(variables: dict) -> tuple[str | None, np.ndarray | None]:
    from scipy.io.matlab._mio5_params import mat_struct

    best_key, best_arr, best_size = None, None, 0
    for k, v in variables.items():
        try:
            # Handle MATLAB structs by extracting 'value' field
            if isinstance(v, mat_struct) and hasattr(v, 'value'):
                arr = np.asarray(v.value)
            else:
                arr = np.asarray(v)
        except Exception:
            continue
        if arr.dtype.kind not in ("f", "i", "u", "c"):
            continue
        if arr.size > best_size:
            best_key, best_arr, best_size = k, arr, arr.size
    return best_key, best_arr


def _largest_hdf5_dataset(variables: dict):
    import h5py

    best_key, best_ref, best_size = None, None, 0
    for k, v in variables.items():
        if not isinstance(v, h5py.Dataset):
            continue
        arr = np.asarray(v)
        if arr.dtype.kind not in ("f", "i", "u", "c"):
            continue
        if arr.size > best_size:
            best_key, best_ref, best_size = k, v, arr.size
    return best_key, best_ref


_AXIS_KEYWORDS = {
    "energy": ("energy", "e_k", "ek", "ke", "be", "binding", "kinetic"),
    "angle":  ("angle", "theta", "phi", "kx", "ky", "k_x", "k_y",
               "momentum", "defl", "slit"),
    "kz":     ("kz", "k_z", "perp"),
    "x":      ("x", "scan_x", "posx", "pos_x"),
    "y":      ("y", "scan_y", "posy", "pos_y"),
    "time":   ("time", "delay", "pump_probe"),
    "temperature": ("temperature", "temp", "t_k"),
    "gate":   ("gate", "vg", "v_g", "voltage"),
}

_UNIT_MAP = {
    "energy": "eV",
    "angle":  "deg",
    "kz":     "Å⁻¹",
    "x":      "µm",
    "y":      "µm",
    "time":   "ps",
    "temperature": "K",
    "gate":   "V",
}


def _classify_key(key: str) -> str | None:
    label = key.casefold()
    for role, keywords in _AXIS_KEYWORDS.items():
        if any(kw in label for kw in keywords):
            return role
    return None


def _extract_axes(
    variables: dict,
    data_shape: tuple[int, ...],
    data_key: str | None,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Match 1-D arrays in *variables* to data_shape dimensions by length."""
    n_dims = len(data_shape)
    axes: list[np.ndarray | None] = [None] * n_dims
    labels: list[str] = [""] * n_dims
    units: list[str] = [""] * n_dims

    # Index candidate 1-D arrays by their length
    by_length: dict[int, list[tuple[str, np.ndarray]]] = {}
    for k, v in variables.items():
        if k == data_key:
            continue
        try:
            arr = np.asarray(v, dtype=float).squeeze()
        except Exception:
            continue
        if arr.ndim != 1 or arr.size < 2:
            continue
        by_length.setdefault(arr.size, []).append((k, arr))

    for dim_idx, dim_size in enumerate(data_shape):
        candidates = by_length.get(dim_size, [])
        if not candidates:
            axes[dim_idx] = np.arange(dim_size, dtype=float)
            labels[dim_idx] = f"Axis {dim_idx}"
            units[dim_idx] = "index"
            continue
        # Prefer a candidate whose key name classifies to a known role
        chosen_key, chosen_arr = candidates[0]
        for k, arr in candidates:
            if _classify_key(k) is not None:
                chosen_key, chosen_arr = k, arr
                break
        role = _classify_key(chosen_key)
        axes[dim_idx] = chosen_arr
        labels[dim_idx] = chosen_key if role is None else role.capitalize()
        units[dim_idx] = _UNIT_MAP.get(role, "a.u.") if role else "a.u."

    return [a if a is not None else np.arange(s) for a, s in zip(axes, data_shape)], labels, units


def _guess_mode(variables: dict, shape: tuple[int, ...]) -> str:
    keys = {k.casefold() for k in variables}
    return _guess_mode_from_keys(keys, shape)


def _guess_mode_from_keys(keys: set[str], shape: tuple[int, ...]) -> str:
    has_energy = any(kw in k for k in keys for kw in ("energy", "ek", "be", "ke"))
    has_angle  = any(kw in k for k in keys for kw in ("angle", "theta", "kx", "ky"))
    has_xy     = any(kw in k for k in keys for kw in ("scan_x", "scan_y", "posx", "posy"))

    if has_energy and has_angle and has_xy:
        return f"{len(shape)}D Spatial ARPES"
    if has_energy and has_angle:
        return "ARPES"
    if has_energy:
        return "Spectrum"
    return "Unknown"


def _scalar_value(v) -> float | str:
    try:
        arr = np.asarray(v).squeeze()
        if arr.size == 1:
            return float(arr)
    except Exception:
        pass
    return str(v)


__all__ = ["MatlabLoader"]
