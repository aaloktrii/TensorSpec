"""NetCDF (.nc) map files -> PEEM maps (no Qt).

Reads any NetCDF whose variables are 2-D images on one grid: the XPEEM batch outputs
(``tensorspec_peem/.../maps.nc``: pcd / pld + mean_<channel> + roi + domain_map), analysis
outputs (xld_110K, xmld, ... several maps per file) or any other file with 2-D variables. 3-D
variables (n, y, x) are kept as frame stacks.

Each 2-D variable gets a type, used for its default display:
  signed     asymmetry-type (pcd, pld, xmld, xld, xmcd, pmld, ... or values on both sides of 0)
             -> blue-white-red, symmetric
  intensity  channel means / sums / other positive images -> grey
  domain     a domain map (-1 / 0 / +1)
  mask       a region of interest (0 / 1, 'roi'): applied to the maps, not shown
Channel pairs are recognised from mean_<tag> variables: CP/CM, CL/CR (BNL; CL = CP, CR = CM),
LH/LV, A/B (L2 energies).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

SIGNED_NAME = re.compile(r"xmld|xmcd|xnld|xld|pmld|pmnld|pcd|pld|asym|dichro|diff|compare|ratio|(^|_)(ld|mcd)(_|$)", re.I)
INTENSITY_NAME = re.compile(r"^(mean|sum|intensity|counts|i0|stack)", re.I)
DOMAIN_NAME = re.compile(r"domain|^mz$", re.I)
MASK_NAMES = ("roi", "mask", "fov")
PAIRS = (("CP", "CM", "xmcd"), ("CL", "CR", "xmcd"), ("LH", "LV", "xld"), ("A", "B", "l2"))
UM_KEYS = ("um_per_px", "pixel_size_um", "um_per_pixel")


def var_type(name: str, arr: np.ndarray) -> str:
    """'mask', 'domain', 'intensity' or 'signed' (see the module docstring); the name decides
    first, then the values (both signs -> signed)."""
    a = np.asarray(arr)
    if a.dtype == bool or name.lower() in MASK_NAMES:
        return "mask"
    if DOMAIN_NAME.search(name):
        return "domain"
    if INTENSITY_NAME.search(name):
        return "intensity"
    if SIGNED_NAME.search(name):
        return "signed"
    fin = a[np.isfinite(a)] if a.dtype.kind == "f" else a.ravel()
    return "signed" if fin.size and float(np.min(fin)) < 0 < float(np.max(fin)) else "intensity"


def _um_per_px(ds) -> float | None:
    for k in UM_KEYS:
        v = ds.attrs.get(k)
        if v is not None and np.isfinite(float(v)) and float(v) > 0:
            return float(v)
    for c in ("x_um", "x"):
        if c in ds.coords and ds[c].size > 1 and str(ds[c].attrs.get("units", "")).lower() in ("um", "µm", "micron"):
            return float(abs(ds[c].values[1] - ds[c].values[0]))
    return None


def read_maps_nc(path: str | Path) -> dict[str, Any]:
    """{maps: {var: 2-D float}, types: {var: type}, stacks: {var: 3-D}, roi, um_per_px, attrs,
    pair: (tag_a, tag_b, kind) | None, primary: the variable to show first, source}."""
    import xarray as xr

    path = Path(path)
    with xr.open_dataset(path) as ds:
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in ds.attrs.items()}
        raw2d, stacks = {}, {}
        for v in ds.data_vars:
            a = np.asarray(ds[v].values)
            if a.ndim == 2:
                raw2d[str(v)] = a
            elif a.ndim == 3:
                stacks[str(v)] = a.astype(float)
        um = _um_per_px(ds)
    if not raw2d and not stacks:
        raise ValueError(f"{path.name}: no 2-D or 3-D image variables")
    shapes = {a.shape for a in raw2d.values()}
    if len(shapes) > 1:  # keep the most common grid
        common = max(shapes, key=lambda s: sum(a.shape == s for a in raw2d.values()))
        raw2d = {k: a for k, a in raw2d.items() if a.shape == common}
    types = {k: var_type(k, a) for k, a in raw2d.items()}
    roi = None
    for k, t in types.items():
        if t == "mask":
            roi = np.asarray(raw2d[k]) > 0
    maps = {}
    for k, a in raw2d.items():
        if types[k] == "mask":
            continue
        a = np.asarray(a, dtype=float)
        if roi is not None and types[k] != "domain":
            a = np.where(roi, a, np.nan)
        maps[k] = a
    types = {k: types[k] for k in maps}
    pair = None
    tags = {k[5:].upper(): k for k in maps if k.lower().startswith("mean_")}
    for a, b, kind in PAIRS:
        if a in tags and b in tags:
            pair = (tags[a], tags[b], kind)
            break
    order = ["xmld", "pmld", "pcd", "pld", "xld", "xmcd"]
    signed = [k for k in maps if types[k] == "signed"]
    primary = next((k for o in order for k in signed if k.lower().startswith(o)), None) \
        or (signed[0] if signed else None) or next(iter(maps), None)
    return {"maps": maps, "types": types, "stacks": stacks, "roi": roi, "um_per_px": um, "attrs": attrs,
            "pair": pair, "primary": primary, "source": str(path)}


def dataset_name(path: str | Path) -> str:
    """A readable dataset name: the file stem, or for generic file names (maps.nc) the folder;
    generic folders (rep1, merged, 20um ...) get the folder above as prefix:
    .../Sample_2_..._XLD/units/rep1/maps.nc -> 'Sample_2_..._XLD_rep1'."""
    p = Path(path)
    if p.stem.lower() not in ("maps", "data", "map"):
        return p.stem
    parts = [q.name for q in p.parents if q.name][:3]   # folder, its parent, grandparent
    name = parts[0] if parts else p.stem
    generic = re.compile(r"^(rep\d+|merged|ld_maps|map_l3|\d+um|units|[0-9]+)$", re.I)
    if generic.match(name):
        up = parts[2] if len(parts) > 2 and parts[1].lower() == "units" else (parts[1] if len(parts) > 1 else "")
        name = f"{up}_{name}" if up else name
    return name
