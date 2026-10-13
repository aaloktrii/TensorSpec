"""omega_G - omega_2D correlation (strain / doping decomposition, Lee et al. 2012) for Raman maps,
split by layer number and substrate, with peak counting. Method: core/raman/strain_doping.py.

Run:
    python scripts/raman/strain_doping.py FOLDER [FOLDER ...] [--out DIR]
writes <folder>/output/strain_doping.{csv,pdf}, and strain_doping_combined.{csv,pdf} for several
folders. Run it as a script file (not from stdin): the worker processes re-import __main__.

CLI of Raman_Classifier ``strain_doping.py``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tensorspec.core.raman import reference as gref  # noqa: E402
from tensorspec.core.raman.strain_doping import analyse_folder, summarise, write_csv  # noqa: E402
from tensorspec.plotting.backends.raman_matplotlib_engine import plot_correlation  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", help="folder for the combined plot (default: first folder's output)")
    ap.add_argument("--json", default=gref.DEFAULT_REF)
    a = ap.parse_args()
    ref = gref.load(a.json)
    every = []
    for f in a.folders:
        f = os.path.expanduser(f).rstrip(os.sep)
        rows = analyse_folder(f, ref)
        out = os.path.join(f, "output")
        os.makedirs(out, exist_ok=True)
        write_csv(rows, os.path.join(out, "strain_doping.csv"))
        plot_correlation(rows, os.path.join(out, "strain_doping.pdf"), os.path.basename(f))
        every += rows
        print("%s -> %s" % (os.path.basename(f), out))
        for (lay, sub), cnt in sorted(summarise(rows).items()):
            print("   %s on %-5s n=%3d   (G peaks, 2D peaks): %s" % (
                lay, sub, sum(cnt.values()),
                ", ".join("%s:%d" % (k, v) for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]))))
    if len(a.folders) > 1:
        out = os.path.expanduser(a.out) if a.out else os.path.join(
            os.path.expanduser(a.folders[0]).rstrip(os.sep), "output")
        os.makedirs(out, exist_ok=True)
        plot_correlation(every, os.path.join(out, "strain_doping_combined.pdf"), "combined")
        write_csv(every, os.path.join(out, "strain_doping_combined.csv"))
        print("combined ->", out)


if __name__ == "__main__":
    main()
