#!/usr/bin/env python3
"""Plot a point-wise SPR-KKR ARPES cut (sprkkr_e2e.py --pointwise output).

Reads <run>/pot_arpes.npz  (intensity dims = (energy, theta, phi); theta = LAB slit angle)
and   <run>/arpes/pointwise_points.json  (per-point k_slit / k_defl at the reference energy)
and draws E vs lab slit angle and E vs k_slit.

Usage:
  python scripts/sprkkr/plot_pointwise.py scratch/sprkkr_e2e/pointwise_vte2
  python scripts/sprkkr/plot_pointwise.py <run_dir> --vmax 0.6 --cmap inferno --out cut.png
"""
import argparse
import glob
import json
import os

import numpy as np


def load_run(run_dir: str):
    npz = glob.glob(os.path.join(run_dir, "*_arpes.npz"))
    if not npz:
        raise FileNotFoundError(f"no *_arpes.npz under {run_dir}")
    d = np.load(npz[0])
    I = d["intensity"]  # (energy, theta, phi)
    if I.ndim == 3:
        I = I[:, :, 0]
    E = d["energy"]
    theta = d["theta"]  # lab slit angle, deg
    pj = os.path.join(run_dir, "arpes", "pointwise_points.json")
    if not os.path.isfile(pj):
        pj = os.path.join(run_dir, "pointwise_points.json")
    k_slit = None
    k_defl = None
    if os.path.isfile(pj):
        pts = json.load(open(pj))["points"]
        pts = sorted(pts, key=lambda p: p["slit_deg"])
        k_slit = np.array([p["k_slit"] for p in pts])
        k_defl = float(pts[0]["k_defl"])
    return I, E, theta, k_slit, k_defl, npz[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--vmax", type=float, default=0.7, help="colour saturation, fraction of max")
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--out", default=None, help="png path (default <run_dir>/pointwise_cut.png)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    I, E, theta, k_slit, k_defl, src = load_run(args.run_dir)
    In = I / I.max()
    print(f"[plot] {src}: I(E,theta)={I.shape} E={E.min():.2f}..{E.max():.2f} eV "
          f"theta={theta.min():.1f}..{theta.max():.1f} deg"
          + (f" k_slit={k_slit.min():.3f}..{k_slit.max():.3f} 1/A k_defl={k_defl:+.3f}" if k_slit is not None else ""))

    ncol = 2 if k_slit is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6 * ncol, 5), squeeze=False)
    ax = axes[0, 0]
    ax.pcolormesh(theta, E, In, shading="auto", cmap=args.cmap, vmin=0, vmax=args.vmax)
    ax.set_xlabel("lab slit angle (deg)")
    ax.set_ylabel(r"$E - E_F$ (eV)")
    ax.set_title("point-wise SPR-KKR cut")
    if k_slit is not None:
        ax = axes[0, 1]
        ax.pcolormesh(k_slit, E, In, shading="auto", cmap=args.cmap, vmin=0, vmax=args.vmax)
        ax.set_xlabel(r"$k_{slit}$ ($\mathrm{\AA}^{-1}$)")
        ax.set_ylabel(r"$E - E_F$ (eV)")
        ax.set_title(rf"$k_{{defl}}$ = {k_defl:+.3f} $\mathrm{{\AA}}^{{-1}}$ (offset from $\Gamma$)")
    plt.tight_layout()
    out = args.out or os.path.join(args.run_dir, "pointwise_cut.png")
    plt.savefig(out, dpi=130)
    print(f"[plot] saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
