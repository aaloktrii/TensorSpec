#!/usr/bin/env python3
"""4 configs x 5 deflectors dispersion grid for VTe2 (-201), Chinook/GrizzlyME.

Deliberately the same layout, axes and colour rules as
scripts/sprkkr/plot_grid_4x5.py, so the two figures can be read side by side:
rows = beam/sample configuration, columns = deflector, ONE colour scale per row
(99.5th percentile of that configuration), row label carries the total intensity.

  python scripts/chinook/plot_grid_4x5_chinook.py \
      --root scratch/chinook_matrix/cubes \
      --out  scratch/chinook_matrix/vte2_grid_4configs_5deflectors_chinook.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
# Vector export: keep text as TEXT (editable in Illustrator), not outlines.
# The 20 intensity panels are image data and stay raster inside the PDF/SVG;
# every axis, tick, label, title and frame remains an editable vector object.
matplotlib.rcParams["pdf.fonttype"] = 42      # TrueType, not Type-3
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"  # <text> elements, not <path>
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

K_PER_SQRT_EV = 0.512316

CONFIGS = [
    ("th55_ph0",  "A\nbeam 55$^\\circ$\n$\\parallel$ slit\n(as measured)"),
    ("th20_ph0",  "B\nbeam 20$^\\circ$\n$\\parallel$ slit"),
    ("th55_ph90", "C\nbeam 55$^\\circ$\n$\\perp$ slit\n(rot 90$^\\circ$)"),
    ("th20_ph90", "D\nbeam 20$^\\circ$\n$\\perp$ slit\n(rot 90$^\\circ$)"),
]
DEFL = [("m10p3", -10.3), ("m5", -5.0), ("0", 0.0), ("5", 5.0), ("10p3", 10.3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="scratch/chinook_matrix/cubes")
    ap.add_argument("--out", default="scratch/chinook_matrix/"
                                     "vte2_grid_4configs_5deflectors_chinook.png")
    ap.add_argument("--hv", type=float, default=84.0)
    ap.add_argument("--workf", type=float, default=4.5)
    ap.add_argument("--gamma", type=float, default=0.4,
                    help="PowerNorm exponent for the colour map. <1 lifts the dim panels "
                         "while keeping ONE monotonic mapping per row, so panels stay "
                         "comparable. 1.0 = plain linear.")
    ap.add_argument("--per-panel", action="store_true",
                    help="scale each panel to its own 99.5th pct instead of one scale "
                         "per row; use when one deflector dominates the row")
    args = ap.parse_args()

    k = K_PER_SQRT_EV * np.sqrt(args.hv - args.workf)

    data, kx, E, walls = {}, None, None, []
    for tag, _ in CONFIGS:
        cols = []
        for dtag, _dd in DEFL:
            p = os.path.join(args.root, f"cube_{tag}_defl_{dtag}.npz")
            z = np.load(p, allow_pickle=True)
            cols.append(np.asarray(z["cube"], float)[:, 0, :])   # (theta, E)
            walls.append(float(z["wall_s"]))
            if kx is None:
                kx = k * np.sin(np.radians(np.asarray(z["theta"], float)))
                E = np.asarray(z["energy"], float)
        data[tag] = np.stack(cols, axis=1)                        # (theta, defl, E)

    fig, axes = plt.subplots(4, 5, figsize=(18.0, 13.0), sharex=True, sharey=True)
    ext = [kx.min(), kx.max(), E.min(), E.max()]

    totals = {}
    for r, (tag, label) in enumerate(CONFIGS):
        I = data[tag]
        totals[tag] = I.sum()
        vmax = np.percentile(I, 99.5)
        for c in range(5):
            ax = axes[r, c]
            vm = np.percentile(I[:, c, :], 99.5) if args.per_panel else vmax
            ax.imshow(I[:, c, :].T, origin="lower", aspect="auto", extent=ext,
                      cmap="magma",
                      norm=PowerNorm(args.gamma, vmin=0.0, vmax=max(vm, 1e-30)))
            ax.axhline(0.0, color="w", lw=0.6, alpha=0.45)
            ax.axvline(0.0, color="w", lw=0.6, alpha=0.3, ls=":")
            if r == 0:
                ax.set_title("deflector %+.1f$^\\circ$\n$k_\\perp$ = %+.3f $\\AA^{-1}$"
                             % (DEFL[c][1], k * np.sin(np.radians(DEFL[c][1]))),
                             fontsize=10.5)
            if c == 0:
                ax.set_ylabel("$E - E_F$  (eV)", fontsize=10)
            if r == 3:
                ax.set_xlabel("$k_{slit}$  ($\\AA^{-1}$)", fontsize=10)
            ax.tick_params(labelsize=8.5)
        axes[r, 0].text(-0.60, 0.5, label + "\n$\\Sigma I$=%.3e" % totals[tag],
                        transform=axes[r, 0].transAxes, va="center", ha="center",
                        fontsize=10, linespacing=1.45)

    ref = totals["th55_ph0"]
    sub = "  |  ".join("%s = %.2f$\\times$" % (lab.split("\n")[0], totals[t] / ref)
                       for t, lab in CONFIGS)
    fig.suptitle("VTe$_2$ ($\\overline{2}$01)  Chinook one-step + GrizzlyME on a V100  --  "
                 "h$\\nu$ = 84 eV, LH, $V_0$ = 12 eV\n"
                 "100 slit angles x 200 energies per cut, %.0f s mean per cut, %.1f min for all 20"
                 "        relative total intensity (A = 1):  %s%s"
                 % (np.mean(walls), sum(walls) / 60.0, sub,
                    ("\nEACH PANEL scaled to its own 99.5th percentile - panels are NOT "
                     "comparable to each other" if args.per_panel
                     else "\ncolour: one scale per row, 99.5th pct, PowerNorm gamma=%.2f "
                          "(monotonic - panels within a row remain comparable)" % args.gamma)),
                 fontsize=12.0, y=0.985)
    fig.tight_layout(rect=[0.085, 0, 1.0, 0.945])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)

    print("\n=== total intensity ===")
    for t, _ in CONFIGS:
        print("%-12s %.4e   %.3f x A" % (t, totals[t], totals[t] / ref))
    print("\n=== per-deflector totals ===")
    print("%-12s %s" % ("", "  ".join("%9.1f" % d for _, d in DEFL)))
    for t, _ in CONFIGS:
        print("%-12s %s" % (t, "  ".join("%9.3e" % data[t][:, c, :].sum()
                                         for c in range(5))))
    print("\nwall: mean %.1f s, min %.1f, max %.1f, total %.1f min"
          % (np.mean(walls), min(walls), max(walls), sum(walls) / 60.0))


if __name__ == "__main__":
    main()
