#!/usr/bin/env python3
"""4 configs x 5 deflectors dispersion grid for VTe2 (-201), SPR-KKR kkrspec point-wise.

  python scripts/sprkkr/plot_grid_4x5.py [--root scratch/sprkkr_e2e] [--out FILE.png]

Reads the four stacked cubes cuts_vte2_th{55,20}_ph{0,90}/vte2_cuts_*.npz
(each intensity(k_slit, k_defl, E)) and lays them out as rows = beam/sample
configuration, columns = deflector. One colour scale per ROW (99.5th pct), so
panels are comparable across deflectors within a configuration but not between
configurations -- the row label carries the total intensity for that.
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

_ap = argparse.ArgumentParser()
_ap.add_argument("--root", default="scratch/sprkkr_e2e")
_ap.add_argument("--out", default="scratch/sprkkr_e2e/vte2_grid_4configs_5deflectors.png")
_ap.add_argument("--gamma", type=float, default=0.4,
                 help="PowerNorm exponent; <1 lifts dim panels, one monotonic mapping per row")
_args = _ap.parse_args()
ROOT = _args.root
CONFIGS = [
    ("cuts_vte2_th55_ph0",  "A\nbeam 55$^\\circ$\n$\\parallel$ slit\n(as measured)"),
    ("cuts_vte2_th20_ph0",  "B\nbeam 20$^\\circ$\n$\\parallel$ slit"),
    ("cuts_vte2_th55_ph90", "C\nbeam 55$^\\circ$\n$\\perp$ slit\n(rot 90$^\\circ$)"),
    ("cuts_vte2_th20_ph90", "D\nbeam 20$^\\circ$\n$\\perp$ slit\n(rot 90$^\\circ$)"),
]
DEFL_DEG = [-10.3, -5.0, 0.0, 5.0, 10.3]

cubes, kx, E, ky = {}, None, None, None
for d, _ in CONFIGS:
    p = os.path.join(ROOT, d, "vte2_cuts_%s.npz" % d.replace("cuts_vte2_", ""))
    z = np.load(p, allow_pickle=True)
    cubes[d] = np.asarray(z["intensity"], float)   # (k_slit, k_defl, E)
    kx, E, ky = z["kx"], z["E"], z["ky"]

fig, axes = plt.subplots(4, 5, figsize=(18.0, 13.0), sharex=True, sharey=True)
ext = [kx.min(), kx.max(), E.min(), E.max()]

totals = {}
for r, (d, label) in enumerate(CONFIGS):
    I = cubes[d]
    totals[d] = I.sum()
    vmax = np.percentile(I, 99.5)          # one scale per ROW
    for c in range(5):
        ax = axes[r, c]
        ax.imshow(I[:, c, :].T, origin="lower", aspect="auto", extent=ext,
                  cmap="magma", norm=PowerNorm(_args.gamma, vmin=0.0, vmax=vmax))
        ax.axhline(0.0, color="w", lw=0.6, alpha=0.45)
        ax.axvline(0.0, color="w", lw=0.6, alpha=0.3, ls=":")
        if r == 0:
            ax.set_title("deflector %+.1f$^\\circ$\n$k_\\perp$ = %+.3f $\\AA^{-1}$"
                         % (DEFL_DEG[c], ky[c]), fontsize=10.5)
        if c == 0:
            ax.set_ylabel("$E - E_F$  (eV)", fontsize=10)
        if r == 3:
            ax.set_xlabel("$k_{slit}$  ($\\AA^{-1}$)", fontsize=10)
        ax.tick_params(labelsize=8.5)
    axes[r, 0].text(-0.60, 0.5, label + "\n$\\Sigma I$=%.3e" % totals[d],
                    transform=axes[r, 0].transAxes, va="center", ha="center",
                    fontsize=10, linespacing=1.45)

ref = totals["cuts_vte2_th55_ph0"]
sub = "  |  ".join("%s = %.2f$\\times$" % (lab.split("\n")[0], totals[d] / ref) for d, lab in CONFIGS)
fig.suptitle("VTe$_2$ ($\\overline{2}$01)  SPR-KKR one-step ARPES  --  h$\\nu$ = 84 eV, LH, "
             "point-wise deflector cuts (100 slit angles x 200 energies each)\n"
             "relative total intensity (A = 1):  " + sub
             + "\ncolour: one scale per row, 99.5th pct, PowerNorm gamma=%.2f" % _args.gamma,
             fontsize=12.5, y=0.985)
fig.tight_layout(rect=[0.085, 0, 1.0, 0.955])
out = _args.out
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print("wrote", out)

print("\n=== total intensity ===")
for d, lab in CONFIGS:
    print("%-22s %.4e   %.3f x A" % (d, totals[d], totals[d] / ref))
print("\n=== per-deflector totals (rows = config) ===")
print("%-22s %s" % ("", "  ".join("%9.1f" % x for x in DEFL_DEG)))
for d, lab in CONFIGS:
    print("%-22s %s" % (d, "  ".join("%9.3e" % cubes[d][:, c, :].sum() for c in range(5))))
