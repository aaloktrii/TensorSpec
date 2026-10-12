#!/usr/bin/env python3
"""Convert chinook_remote_runner cubes to the TensorSpec Data-Viewer contract.

The runner writes cube(theta, phi, E) with theta/phi in DEGREES. The viewer
(tensorspec/core/io/simulated_loader.SimulatedARPESLoader) wants

    intensity : (kx, ky, E)    kx = k_slit, ky = k_defl, both 1/A
    kx, ky, E : 1-D axes
    metadata  : dict

which for a single-deflector cut is the same array with the axes relabelled:
kx = k sin(theta_slit), ky = [k sin(deflector)], k = 0.512316*sqrt(hv - W).

Produces, per configuration, five *_viewer.npz plus one stacked cube along ky,
exactly matching what scripts/sprkkr/to_viewer_npz.py gives for SPR-KKR, so the
two engines' outputs load and plot identically.

  python scripts/chinook/to_viewer_npz_chinook.py \
      --root scratch/chinook_matrix/cubes \
      --out-dir scratch/chinook_matrix
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

K_PER_SQRT_EV = 0.512316

CONFIGS = {
    "th55_ph0":  dict(incidence=55.0, azimuth=90.0, slit=0.0,
                      note="beam 55 deg, in the slit plane (as measured)"),
    "th20_ph0":  dict(incidence=20.0, azimuth=90.0, slit=0.0,
                      note="beam 20 deg, in the slit plane"),
    "th55_ph90": dict(incidence=55.0, azimuth=0.0, slit=90.0,
                      note="beam 55 deg, perpendicular to the slit plane"),
    "th20_ph90": dict(incidence=20.0, azimuth=0.0, slit=90.0,
                      note="beam 20 deg, perpendicular to the slit plane"),
}
DEFL = [("m10p3", -10.3), ("m5", -5.0), ("0", 0.0), ("5", 5.0), ("10p3", 10.3)]


def convert(path, cfg, defl_deg, hv, workf, v0, out_path):
    z = np.load(path, allow_pickle=True)
    cube = np.asarray(z["cube"], dtype=float)          # (theta, phi, E) == (kx, ky, E)
    theta = np.asarray(z["theta"], dtype=float)
    E = np.asarray(z["energy"], dtype=float)

    k = K_PER_SQRT_EV * np.sqrt(max(hv - workf, 0.1))
    kx = k * np.sin(np.radians(theta))
    ky = np.array([k * np.sin(np.radians(defl_deg))], dtype=float)
    if cube.shape[1] != 1:
        raise ValueError(f"{path}: expected one deflector, got nphi={cube.shape[1]}")

    p = CONFIGS[cfg]
    meta = {
        "engine": "Chinook one-step + GrizzlyME",
        "axes": "intensity(kx,ky,E); kx=k_slit, ky=k_defl [1/A], E rel. E_F [eV]",
        "config": cfg,
        "hv_eV": float(hv),
        "ework_eV": float(workf),
        "inner_potential_eV": float(v0),
        "polarization": "Linear Horizontal (p)",
        "incidence_angle_deg": p["incidence"],
        "manip_azimuth_deg": p["azimuth"],
        "slit_angle_deg": p["slit"],
        "deflector_deg": float(defl_deg),
        "k_defl_1perA": float(ky[0]),
        "theta_lab_deg": theta.tolist(),
        "hkl": [-2, 0, 1],
        "fermi_edge_applied": True,     # unlike the SPR-KKR cubes
        "fresnel": False,
        "wall_s": float(z["wall_s"]) if "wall_s" in z else None,
        "device": str(z["device"]) if "device" in z else None,
        "source_npz": os.path.abspath(path),
        "note": p["note"],
    }
    np.savez_compressed(out_path, intensity=cube, kx=kx, ky=ky, E=E, metadata=meta)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="scratch/chinook_matrix/cubes")
    ap.add_argument("--out-dir", default="scratch/chinook_matrix")
    ap.add_argument("--hv", type=float, default=84.0)
    ap.add_argument("--workf", type=float, default=4.5)
    ap.add_argument("--v0", type=float, default=12.0)
    args = ap.parse_args()

    from tensorspec.core.dft.sprkkr.viewer_export import stack_viewer_cubes

    for cfg in CONFIGS:
        outdir = os.path.join(args.out_dir, f"cuts_vte2_{cfg}_chinook")
        os.makedirs(outdir, exist_ok=True)
        cubes = []
        for tag, dd in DEFL:
            src = os.path.join(args.root, f"cube_{cfg}_defl_{tag}.npz")
            if not os.path.isfile(src):
                raise FileNotFoundError(src)
            dst = os.path.join(outdir, f"{cfg}_defl_{tag}_viewer.npz")
            cubes.append(convert(src, cfg, dd, args.hv, args.workf, args.v0, dst))
            print(f"  {dst}")
        stacked = os.path.join(outdir, f"vte2_cuts_{cfg}_chinook.npz")
        stack_viewer_cubes(cubes, stacked, metadata={
            "engine": "Chinook one-step + GrizzlyME",
            "config": cfg,
            "deflectors_deg": [d for _, d in DEFL],
            "fermi_edge_applied": True,
        })
        d = np.load(stacked, allow_pickle=True)
        print(f"{stacked}  intensity{d['intensity'].shape} "
              f"kx[{d['kx'].min():.3f},{d['kx'].max():.3f}] "
              f"ky={np.round(d['ky'], 3).tolist()} "
              f"E[{d['E'].min():.2f},{d['E'].max():.2f}]")


if __name__ == "__main__":
    main()
