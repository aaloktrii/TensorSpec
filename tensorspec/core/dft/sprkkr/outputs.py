"""
Parsers for real Munich SPR-KKR 9.7 output files (`.spc` ARPES data, SCF log).

Pure numpy/xarray/re. Zero Qt (sandy_rule.md layer 1: core/ = math only).

Row-order fact (verified 2026-09-09 with a real NP=2 kkrspec9.7 run,
`tests/sprkkr/fixtures/_Cu_NP2_ARPES_data.spc`):
  - `.spc` ALWAYS has exactly 8 columns, even when NP>1. There is NO extra
    phi column -- phi is never written to the file.
  - Row nesting order is energy (outermost) -> phi (middle) -> theta
    (innermost, fastest). Total rows == NE*NT*NP.
  - Energy blocks are written from EMAXEV down to EMINEV (descending),
    matching kkrspec's internal binding-energy sweep direction, NOT the
    ascending EMINEV..EMAXEV order given in the .inp file.
  - Theta within a block is written ascending, matching the {a,b} spread
    used for SPEC_EL THETA.
  - Because phi values are never written, real phi angles are NOT
    recoverable from the .spc file alone. `parse_spc` accepts an optional
    `phi_range=(min, max)` (from the ArpesParams that produced the run) and
    linspaces NP points across it, mirroring how kkrspec itself expands
    `PHI={a,b}` for NP points (same convention verified for THETA). If not
    given, phi falls back to integer block indices 0..NP-1 and
    `ds.attrs["phi_is_index"] = True` documents the fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from xarray import DataTree

from tensorspec.core.data_models import TensorData
from tensorspec.core.data_tree import DataTreeBuilder

from .pointwise import AnglePoint

_HEADER_SCALAR_RE = re.compile(r"^\s*(NE|NT|NP|EFERMI|IREL)\s+([-+0-9.EeDd]+)\s*$")
_HASH_LINE_RE = re.compile(r"^#+\s*$")
_FLOAT_RE = re.compile(r"[-+]?\d*\.\d+[EeDd][-+]?\d+|[-+]?\d+\.\d*")


def _to_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _split_header_and_data(lines: List[str]) -> Tuple[List[str], List[str]]:
    """Header ends at the last all-`#` separator line; data is everything after."""
    sep_idx = None
    for i, line in enumerate(lines):
        if _HASH_LINE_RE.match(line.strip()):
            sep_idx = i
    if sep_idx is None:
        raise ValueError("No '#######' separator line found in .spc file; cannot locate data block.")
    return lines[:sep_idx], lines[sep_idx + 1 :]


def parse_spc(path: str, phi_range: Optional[Tuple[float, float]] = None) -> xr.Dataset:
    """Parse a Munich SPR-KKR `<DATASET>_ARPES_data.spc` file.

    Returns an `xr.Dataset` with dims ("energy", "theta", "phi") (phi size
    is always >= 1, even for NP=1), data vars I_tot, I_up, I_dn, pol,
    k_par, det, and attrs EF_Ry, NE, NT, NP, source_file.
    """
    p = Path(path)
    text = p.read_text()
    lines = text.splitlines()
    header_lines, data_lines = _split_header_and_data(lines)

    scalars: Dict[str, float] = {}
    for line in header_lines:
        m = _HEADER_SCALAR_RE.match(line)
        if m:
            key, val = m.group(1), m.group(2)
            scalars[key] = _to_float(val)

    for required in ("NE", "NT", "EFERMI"):
        if required not in scalars:
            raise ValueError(f"Could not find '{required}' in .spc header: {path}")

    ne = int(scalars["NE"])
    nt = int(scalars["NT"])
    np_ = int(scalars.get("NP", 1))
    ef_ry = scalars["EFERMI"]

    rows: List[List[float]] = []
    for line in data_lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 8:
            continue
        rows.append([_to_float(x) for x in parts[:8]])

    n_rows = len(rows)
    expected = ne * nt * np_
    if n_rows != expected:
        raise ValueError(
            f"{path}: found {n_rows} data rows, expected NE*NT*NP = {ne}*{nt}*{np_} = {expected}"
        )

    arr = np.asarray(rows, dtype=float)  # (n_rows, 8): theta, E, Itot, Iup, Idn, pol, kpar, det
    raw = arr.reshape(ne, np_, nt, 8)

    energy_raw = raw[:, 0, 0, 1]
    theta_raw = raw[0, 0, :, 0]

    e_order = np.argsort(energy_raw)
    t_order = np.argsort(theta_raw)

    raw = raw[e_order, :, :, :]
    raw = raw[:, :, t_order, :]

    energy = energy_raw[e_order]
    theta = theta_raw[t_order]

    phi_is_index = False
    if np_ <= 1:
        phi = np.array([phi_range[0] if phi_range else 0.0])
    elif phi_range is not None:
        phi = np.linspace(phi_range[0], phi_range[1], np_)
    else:
        phi = np.arange(np_, dtype=float)
        phi_is_index = True

    # raw: (energy, phi, theta, col) -> (energy, theta, phi, col)
    final = np.transpose(raw, (0, 2, 1, 3))

    def var(col: int) -> Tuple[Tuple[str, ...], np.ndarray]:
        return ("energy", "theta", "phi"), final[..., col]

    ds = xr.Dataset(
        data_vars={
            "I_tot": var(2),
            "I_up": var(3),
            "I_dn": var(4),
            "pol": var(5),
            "k_par": var(6),
            "det": var(7),
        },
        coords={
            "energy": ("energy", energy, {"units": "eV"}),
            "theta": ("theta", theta, {"units": "deg"}),
            "phi": ("phi", phi, {"units": "deg"}),
        },
        attrs={
            "EF_Ry": ef_ry,
            "NE": ne,
            "NT": nt,
            "NP": np_,
            "source_file": str(path),
            "phi_is_index": phi_is_index,
        },
    )
    return ds


def spc_to_tensor(ds: xr.Dataset, meta: Optional[Dict[str, Any]] = None) -> TensorData:
    """Convert a parsed `.spc` Dataset (I_tot) into the agnostic `TensorData` cube.

    Squeezes the phi axis away when NP==1 (2D Energy x Theta cube); keeps it
    for NP>1 (3D Energy x Theta x Phi cube).
    """
    meta = dict(meta or {})
    np_ = int(ds.attrs.get("NP", ds.sizes.get("phi", 1)))

    energy = ds["energy"].values
    theta = ds["theta"].values
    intensity = ds["I_tot"].values  # (energy, theta, phi)

    if np_ <= 1:
        value = intensity[:, :, 0]
        axes = [energy, theta]
        labels = ["Energy", "Theta"]
        units = ["eV", "deg"]
    else:
        phi = ds["phi"].values
        value = intensity
        axes = [energy, theta, phi]
        labels = ["Energy", "Theta", "Phi"]
        units = ["eV", "deg", "deg"]

    metadata: Dict[str, Any] = dict(ds.attrs)
    metadata.update(meta)

    return TensorData(
        value=value,
        axes=axes,
        labels=labels,
        units=units,
        data_type="Simulated ARPES (SPR-KKR)",
        metadata=metadata,
    )


def spc_to_datatree(ds: xr.Dataset, meta: Optional[Dict[str, Any]] = None) -> DataTree:
    """Build the `/simulated`-ready DataTree from a parsed `.spc` Dataset.

    `/raw` holds I_tot (via `DataTreeBuilder`, reused not re-implemented).
    `/processed` gets a Dataset carrying I_up, I_dn, pol, k_par (same coords).
    `/history` gets one provenance line.
    """
    meta = dict(meta or {})
    tensor = spc_to_tensor(ds, meta)
    name = meta.get("dataset") or meta.get("name") or "sprkkr_arpes"

    tree = DataTreeBuilder.build_from_tensor(name, tensor)

    processed = xr.Dataset(
        data_vars={
            "I_up": ds["I_up"],
            "I_dn": ds["I_dn"],
            "pol": ds["pol"],
            "k_par": ds["k_par"],
        },
        coords=ds.coords,
        attrs=dict(ds.attrs),
    )
    tree["processed"] = processed

    source = ds.attrs.get("source_file", "<unknown>")
    DataTreeBuilder._append_history(
        tree, f"SPR-KKR kkrspec ARPES parsed from {source}"
    )
    return tree


_ITER_RE = re.compile(
    r"^\s*(\d+)\s+ERR\s+([-+0-9.EeDd]+)\s+[-+0-9.EeDd]+\s+EF\s+([-+0-9.EeDd]+)",
)
_ETOT_RE = re.compile(r"ETOT\s+([-+0-9.EeDd]+)")
_CONVERGED_RE = re.compile(r"SCF\s*-\s*cycle converged")


@dataclass
class ScfStatus:
    iterations: int
    converged: bool
    ef_ry: Optional[float]
    etot_ry: Optional[float]
    last_err: Optional[float]
    history: List[Tuple[int, float, float]] = field(default_factory=list)


def parse_scf_log(path: str) -> ScfStatus:
    """Parse a Munich SPR-KKR SCF log (`<DATASET>_SCF.out` or its tail)."""
    text = Path(path).read_text()

    history: List[Tuple[int, float, float]] = []
    for line in text.splitlines():
        m = _ITER_RE.match(line)
        if m:
            it = int(m.group(1))
            err = _to_float(m.group(2))
            ef = _to_float(m.group(3))
            history.append((it, err, ef))

    converged = bool(_CONVERGED_RE.search(text))

    etot_match = list(_ETOT_RE.finditer(text))
    etot_ry = _to_float(etot_match[-1].group(1)) if etot_match else None

    if history:
        iterations, last_err, ef_ry = history[-1]
    else:
        iterations, last_err, ef_ry = 0, None, None

    return ScfStatus(
        iterations=iterations,
        converged=converged,
        ef_ry=ef_ry,
        etot_ry=etot_ry,
        last_err=last_err,
        history=history,
    )


def stitch_spc(datasets: List[xr.Dataset]) -> xr.Dataset:
    """Concatenate energy-chunked `.spc` Datasets (from fan-out) into one.

    Sorts by energy and drops any duplicate boundary energy point.
    """
    if not datasets:
        raise ValueError("stitch_spc: empty dataset list")
    if len(datasets) == 1:
        return datasets[0]

    merged = xr.concat(datasets, dim="energy")
    merged = merged.sortby("energy")
    merged = merged.drop_duplicates("energy", keep="first")

    attrs = dict(datasets[0].attrs)
    attrs["NE"] = merged.sizes["energy"]
    merged.attrs.update(attrs)
    return merged


def stitch_points(
    datasets: List[xr.Dataset],
    points: List[AnglePoint],
    deflector_deg: float = 0.0,
) -> xr.Dataset:
    """Stitch N single-point (theta=1, phi=1) `.spc` datasets into one lab-axis dataset.

    Each input dataset corresponds to one AnglePoint; its theta and phi coordinates
    are replaced with the lab slit angle and deflector angle respectively.
    Lab momenta (k_par, k_defl) are computed from the energy-independent ratio
    of lab to SPR-KKR k values.

    Args:
        datasets: List of single-point xr.Dataset, each with theta size=1, phi size=1
        points: Corresponding AnglePoint list, len(points) == len(datasets)
        deflector_deg: Fixed deflector angle for the phi axis (all points)

    Returns:
        Merged xr.Dataset with theta axis = lab slit angles, phi axis = [deflector_deg],
        and data vars k_par (signed lab slit k), k_defl (signed lab deflector k),
        k_par_sprkkr (original SPR-KKR k_par), plus all original I_tot, I_up, I_dn, pol, det.
    """
    if len(datasets) != len(points):
        raise ValueError(
            f"stitch_points: len(datasets)={len(datasets)} != len(points)={len(points)}"
        )

    # Validate and transform each dataset
    processed = []
    for ds, p in zip(datasets, points):
        # Assert single-point structure
        if ds.sizes.get("theta", 0) != 1:
            raise ValueError(f"stitch_points: dataset theta size != 1 (got {ds.sizes.get('theta')})")
        if ds.sizes.get("phi", 0) != 1:
            raise ValueError(f"stitch_points: dataset phi size != 1 (got {ds.sizes.get('phi')})")

        # Replace theta with lab slit angle, phi with deflector angle
        ds = ds.assign_coords(theta=[p.slit_deg], phi=[float(deflector_deg)])

        # Preserve SPR-KKR k_par before overwriting
        ds = ds.rename({"k_par": "k_par_sprkkr"})

        processed.append(ds)

    # Concat along theta (slit angle) and sort
    merged = xr.concat(processed, dim="theta").sortby("theta")

    # Compute lab momenta from energy-independent ratios
    # ratio = lab_k / sprkkr_k; both proportional to sin(angle), so ratio is const per point
    n_points = len(points)
    ratio_slit = np.zeros(n_points)
    ratio_defl = np.zeros(n_points)
    for i, p in enumerate(points):
        if p.k_par > 0.0:
            ratio_slit[i] = p.k_slit / p.k_par
            ratio_defl[i] = p.k_defl / p.k_par
        else:
            ratio_slit[i] = 0.0
            ratio_defl[i] = 0.0

    # Broadcast ratios along energy and phi axes, then apply
    # merged has dims (energy, theta, phi); ratios indexed by theta
    # Carry the slit-angle coordinate so xarray aligns by value, not by position
    # (merged was sortby("theta"); `points` may arrive in any order).
    slit_coord = [float(p.slit_deg) for p in points]
    ratio_slit_bcast = xr.DataArray(ratio_slit, dims="theta", coords={"theta": slit_coord})
    ratio_defl_bcast = xr.DataArray(ratio_defl, dims="theta", coords={"theta": slit_coord})

    merged["k_par"] = merged["k_par_sprkkr"] * ratio_slit_bcast
    merged["k_defl"] = merged["k_par_sprkkr"] * ratio_defl_bcast

    # Update attributes
    attrs = dict(datasets[0].attrs)
    attrs["NT"] = n_points
    attrs["NP"] = 1
    attrs["pointwise"] = True
    attrs["deflector_deg"] = deflector_deg
    attrs["phi_is_index"] = False
    merged.attrs.update(attrs)

    return merged


def write_points_json(
    path: str,
    points: List[AnglePoint],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Write AnglePoint list and metadata to JSON file.

    File format: {"points": [asdict of each AnglePoint], "meta": metadata dict}
    """
    data = {
        "points": [asdict(p) for p in points],
        "meta": meta or {},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_points_json(path: str) -> Tuple[List[AnglePoint], Dict[str, Any]]:
    """Read AnglePoint list and metadata from JSON file.

    Returns:
        (points list, metadata dict)
    """
    with open(path, "r") as f:
        data = json.load(f)
    points = [AnglePoint(**d) for d in data.get("points", [])]
    meta = data.get("meta", {})
    return points, meta
