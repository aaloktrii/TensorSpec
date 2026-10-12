#!/usr/bin/env python3
"""Export a point-wise SPR-KKR cut (or a stacked map) to MATLAB .mat.

Layout written (matches Sandy's MATLAB convention):

    data.value : intensity, size [n_angle x n_energy]   (one cut)
                 or [n_angle x n_defl x n_energy]       (stacked map)
    data.x     : lab slit angle, deg   ROW vector 1 x n_angle
    data.y     : E - E_F, eV           ROW vector 1 x n_energy
    data.z     : deflector angle, deg  ROW vector 1 x n_defl  (stacked map only)
    data.info  : struct of metadata (geometry, hv, pol, k_defl, paths, ...)

Momentum axes are NOT written -- k_slit / k_defl are recoverable from the
angles (k = 0.512316*sqrt(hv - WF + E) * sin(angle)); info keeps hv_eV,
ework_eV and the deflector angle so the conversion can be redone in MATLAB.

Usage:
  python scripts/sprkkr/to_matlab.py <run_dir> [more_run_dirs ...] [--out FILE.mat]
  python scripts/sprkkr/to_matlab.py <run_dir>/pot_arpes_viewer.npz --out cut.mat
  python scripts/sprkkr/to_matlab.py <config_dir>/vte2_cuts_*.npz --out map.mat

With several inputs and no --out, one .mat is written next to each input.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _points_sidecar(run_dir):
    for p in (os.path.join(run_dir, "arpes", "pointwise_points.json"),
              os.path.join(run_dir, "pointwise_points.json")):
        if os.path.isfile(p):
            with open(p) as fh:
                return json.load(fh)
    return None


def _clean(v):
    """numpy/py scalars -> things scipy.io.savemat is happy with."""
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=float) if v and isinstance(v[0], (int, float)) else [str(x) for x in v]
    if isinstance(v, bool):
        return int(v)
    if v is None:
        return ""
    return v


def load_source(path):
    """path = run dir | *_viewer.npz | stacked *.npz  ->  (value, x_deg, y_eV, k_slit, k_defl, info)."""
    info = {}
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "*_viewer.npz")))
        if not cands:
            cands = sorted(glob.glob(os.path.join(path, "vte2_*.npz")))
        if not cands:
            raise FileNotFoundError(f"no *_viewer.npz / stacked .npz in {path}")
        npz_path, run_dir = cands[0], path
    else:
        npz_path, run_dir = path, os.path.dirname(path)

    d = np.load(npz_path, allow_pickle=True)
    I = np.asarray(d["intensity"], dtype=float)   # (kx, ky, E)
    kx = np.asarray(d["kx"], dtype=float)
    ky = np.asarray(d["ky"], dtype=float)
    E = np.asarray(d["E"], dtype=float)
    meta = d["metadata"].item() if "metadata" in d else {}

    side = _points_sidecar(run_dir)
    if side:
        pts = sorted(side.get("points", []), key=lambda p: p["slit_deg"])
        x_deg = np.array([p["slit_deg"] for p in pts], dtype=float)
        info.update({f"sidecar_{k}": _clean(v) for k, v in (side.get("meta") or {}).items()})
    else:
        # stacked map / no sidecar: k_slit = k(E_F) * sin(theta_slit) -> invert exactly
        k0 = _k_from_meta(meta)
        x_deg = np.degrees(np.arcsin(np.clip(kx / k0, -1.0, 1.0)))
        info["angle_axis"] = "reconstructed from k_slit via asin(k/k0)"
    if x_deg.size != kx.size:
        raise ValueError(f"angle axis {x_deg.size} != k axis {kx.size} in {npz_path}")

    k0 = _k_from_meta(meta)
    if I.shape[1] == 1:                       # single cut -> [angle x energy]
        value = I[:, 0, :]
        k_defl = float(ky[0])
        defl_deg = float((meta or {}).get("deflector_deg",
                         np.degrees(np.arcsin(np.clip(ky[0] / k0, -1.0, 1.0)))))
    else:                                     # stacked -> [angle x defl x energy]
        value = I
        k_defl = ky
        defl_deg = np.degrees(np.arcsin(np.clip(ky / k0, -1.0, 1.0)))

    for k, v in (meta or {}).items():
        info[str(k)] = _clean(v)
    # Config dirs are named ..._th<THETA>_ph<PHI> (beam polar / azimuth). The e2e
    # metadata records theta_ph_deg but not phi_ph_deg, so recover it from the
    # path rather than patching a script a multi-day loop is re-running.
    m = re.search(r"_th(\d+(?:\.\d+)?)_ph(\d+(?:\.\d+)?)", os.path.abspath(npz_path))
    if m:
        info.setdefault("theta_ph_deg", float(m.group(1)))
        info["phi_ph_deg"] = float(m.group(2))
        info["beam_vs_slit"] = "parallel (beam in slit plane)" if float(m.group(2)) == 0.0 \
            else "perpendicular (beam 90 deg from slit plane; sample also rotated 90 deg)"
    info["source_npz"] = os.path.abspath(npz_path)
    info["axes"] = ("value[angle,energy]" if I.shape[1] == 1 else "value[angle,defl,energy]")
    info["x_units"] = "deg (lab slit angle)"
    info["y_units"] = "eV (E - E_F)"
    info["k_units"] = "1/Angstrom (info only; axes are angles)"
    info["engine"] = info.get("engine", "SPR-KKR kkrspec (one-step)")
    return value, x_deg, E, defl_deg, k_defl, info


def _k_from_meta(meta):
    hv = float(meta.get("hv_eV", 84.0))
    wf = float(meta.get("ework_eV", 4.5))
    return 0.512316 * np.sqrt(max(hv - wf, 0.1))


def write_mat(out_path, value, x_deg, y_eV, defl_deg, k_defl, info):
    """Axes are written as MATLAB ROW vectors (1 x N)."""
    from scipy.io import savemat

    data = {
        "value": value,
        "x": np.asarray(x_deg, dtype=float).reshape(1, -1),
        "y": np.asarray(y_eV, dtype=float).reshape(1, -1),
        "info": {k: v for k, v in info.items()},
    }
    if np.ndim(defl_deg) == 0:
        data["info"]["deflector_deg"] = float(defl_deg)
        data["info"]["k_defl"] = float(k_defl)
    else:
        data["z"] = np.asarray(defl_deg, dtype=float).reshape(1, -1)
        data["info"]["k_defl"] = np.asarray(k_defl, dtype=float).reshape(1, -1)
        data["info"]["z_units"] = "deg (deflector angle)"
    savemat(out_path, {"data": data}, do_compression=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="run dir(s), *_viewer.npz, or stacked .npz")
    ap.add_argument("--out", default=None, help="output .mat (only with a single source)")
    args = ap.parse_args()

    if args.out and len(args.sources) > 1:
        sys.exit("--out takes a single source")

    for src in args.sources:
        value, x, y, dd, kd, info = load_source(src)
        if args.out:
            out = args.out
        else:
            base = src.rstrip("/")
            out = (base[:-4] + ".mat") if base.endswith(".npz") else os.path.join(base, "cut.mat")
        write_mat(out, value, x, y, dd, kd, info)
        shape = "x".join(str(s) for s in np.shape(value))
        zmsg = "" if np.ndim(dd) == 0 else f"  z {np.min(dd):.2f}..{np.max(dd):.2f} deg"
        print(f"[mat] {out}  value[{shape}]  x {x.min():.2f}..{x.max():.2f} deg  "
              f"y {y.min():.2f}..{y.max():.2f} eV{zmsg}")


if __name__ == "__main__":
    main()
