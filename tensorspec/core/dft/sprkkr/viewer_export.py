"""Export SPR-KKR ARPES results to TensorSpec's native simulated-ARPES .npz.

The Data Viewer / ARPES suite "Load ARPES Data" opens .npz through
``tensorspec.core.io.simulated_loader.SimulatedARPESLoader``, which expects:

    intensity : (kx, ky, E)
    kx        : (nkx,)   1/A   -- slit axis
    ky        : (nky,)   1/A   -- deflection axis
    E         : (ne,)    eV    -- relative to E_F
    metadata  : dict (pickled 0-d object array)

kkrspec gives us I(energy, theta, phi). For a point-wise run the ``theta`` axis is
already the LAB slit angle and the sidecar ``pointwise_points.json`` carries the
signed lab momenta (k_slit per point, one k_defl for the cut), so kx/ky are exact.
For a legacy fixed-PHI run the cut goes through Gamma: kx = k_par at the energy
closest to E_F, ky = [0.0].

``stack_viewer_cubes`` stitches several single-deflector cubes (one per k_defl)
into one (kx, ky, E) Fermi-map cube along ky.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

VIEWER_SUFFIX = "_viewer.npz"


def _points_from_json(points_json: Optional[str]):
    if not points_json or not os.path.isfile(points_json):
        return None, {}
    with open(points_json) as fh:
        d = json.load(fh)
    pts = sorted(d.get("points", []), key=lambda p: p["slit_deg"])
    return pts, d.get("meta", {}) or {}


def arrays_to_viewer_npz(
    intensity_etp: np.ndarray,
    energy: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    out_path: str,
    *,
    points_json: Optional[str] = None,
    k_par_e0: Optional[np.ndarray] = None,
    metadata: Optional[Dict] = None,
) -> str:
    """Write a viewer .npz from raw kkrspec arrays.

    intensity_etp : (energy, theta, phi) as saved by sprkkr_e2e in pot_arpes.npz
    theta         : lab slit angles (pointwise) or SPR-KKR THETA (legacy), deg
    points_json   : pointwise sidecar -> exact kx (k_slit) and ky (k_defl)
    k_par_e0      : legacy fallback, |k_par| per theta at E~E_F (signed by theta)
    """
    I = np.asarray(intensity_etp, dtype=float)
    if I.ndim == 2:
        I = I[:, :, np.newaxis]
    energy = np.asarray(energy, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    meta: Dict = dict(metadata or {})

    pts, side_meta = _points_from_json(points_json)
    if pts is not None and len(pts) == theta.size:
        kx = np.array([p["k_slit"] for p in pts], dtype=float)
        ky = np.array([float(pts[0]["k_defl"])], dtype=float)
        meta.setdefault("pointwise", True)
        meta.setdefault("deflector_deg", side_meta.get("deflector_deg"))
        meta.setdefault("k_defl_1perA", float(ky[0]))
        meta.setdefault("theta_lab_deg", theta.tolist())
    else:
        if k_par_e0 is None:
            raise ValueError("legacy run: need k_par_e0 (|k_par| per theta at E~E_F)")
        kx = np.sign(theta) * np.abs(np.asarray(k_par_e0, dtype=float))
        ky = np.zeros(1, dtype=float)
        meta.setdefault("pointwise", False)
        meta.setdefault("note", "fixed-PHI cut through Gamma; ky set to 0")
        meta.setdefault("theta_sprkkr_deg", theta.tolist())
        meta.setdefault("phi_sprkkr_deg", phi.tolist())

    # (E, theta, phi) -> (kx, ky, E)
    cube = np.transpose(I, (1, 2, 0))
    if cube.shape[1] != ky.size:
        # NP>1 legacy grid: keep the phi axis as an index axis, viewer still loads it
        ky = np.arange(cube.shape[1], dtype=float)
        meta["note"] = meta.get("note", "") + " | ky is a phi index, not 1/A"
    meta.setdefault("engine", "SPR-KKR kkrspec (one-step)")
    meta.setdefault("axes", "intensity(kx,ky,E); kx=k_slit, ky=k_defl [1/A], E rel. E_F [eV]")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez_compressed(out_path, intensity=cube, kx=kx, ky=ky, E=energy, metadata=meta)
    return out_path


def run_dir_to_viewer_npz(run_dir: str, out_path: Optional[str] = None) -> str:
    """Convert a finished sprkkr_e2e run dir (pot_arpes.npz [+ arpes/pointwise_points.json])."""
    import glob

    cands = [p for p in glob.glob(os.path.join(run_dir, "*_arpes.npz")) if not p.endswith(VIEWER_SUFFIX)]
    if not cands:
        raise FileNotFoundError(f"no *_arpes.npz in {run_dir}")
    src = cands[0]
    d = np.load(src, allow_pickle=True)
    pj = os.path.join(run_dir, "arpes", "pointwise_points.json")
    if not os.path.isfile(pj):
        pj = os.path.join(run_dir, "pointwise_points.json")
    k_par_e0 = None
    if not os.path.isfile(pj):
        # legacy: need |k_par|(theta) at E~E_F -> read one .spc if present
        spc = glob.glob(os.path.join(run_dir, "**", "*_ARPES_data.spc"), recursive=True)
        if spc:
            from .outputs import parse_spc

            ds = parse_spc(spc[0])
            ie = int(np.argmin(np.abs(ds["energy"].values)))
            k_par_e0 = np.abs(ds["k_par"].isel(energy=ie, phi=0).values)
    out = out_path or src.replace(".npz", VIEWER_SUFFIX)
    meta = {"source_npz": os.path.abspath(src)}
    return arrays_to_viewer_npz(
        d["intensity"], d["energy"], d["theta"], d["phi"], out,
        points_json=pj if os.path.isfile(pj) else None, k_par_e0=k_par_e0, metadata=meta,
    )


def stack_viewer_cubes(paths: Sequence[str], out_path: str, metadata: Optional[Dict] = None) -> str:
    """Stack single-deflector viewer cubes (each ky of size 1) into one (kx, ky, E) cube.

    Sorted by ky. All inputs must share kx and E (checked to 1e-6)."""
    if not paths:
        raise ValueError("no cubes to stack")
    loaded = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        loaded.append((float(np.asarray(d["ky"]).ravel()[0]), d["intensity"], d["kx"], d["E"], p))
    loaded.sort(key=lambda t: t[0])
    ky = np.array([t[0] for t in loaded], dtype=float)
    kx0, E0 = loaded[0][2], loaded[0][3]
    for _, I, kx, E, p in loaded:
        if kx.shape != kx0.shape or not np.allclose(kx, kx0, atol=1e-6):
            raise ValueError(f"kx axis differs in {p}")
        if E.shape != E0.shape or not np.allclose(E, E0, atol=1e-6):
            raise ValueError(f"E axis differs in {p}")
        if I.shape[1] != 1:
            raise ValueError(f"{p}: expected a single-deflector cube, got ky size {I.shape[1]}")
    cube = np.concatenate([t[1] for t in loaded], axis=1)  # (kx, ky, E)
    meta = {
        "engine": "SPR-KKR kkrspec (one-step), point-wise deflector map",
        "axes": "intensity(kx,ky,E); kx=k_slit, ky=k_defl [1/A], E rel. E_F [eV]",
        "n_deflectors": int(ky.size),
        "sources": [t[4] for t in loaded],
    }
    meta.update(metadata or {})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez_compressed(out_path, intensity=cube, kx=kx0, ky=ky, E=E0, metadata=meta)
    return out_path
