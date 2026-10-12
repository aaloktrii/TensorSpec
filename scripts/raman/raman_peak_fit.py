"""Lorentzian G/2D fits for every spectrum in a dataset folder, with per-dataset statistics and plots.

Everything goes into <dataset>/output/ (or --out):
    fits.csv      one row per spectrum
    summary.csv   mean, standard deviation, median and N of every quantity,
                  over the spectra whose fits passed
    summary.pdf   histograms (mean ± SD marked) and 2D FWHM vs I(2D)/I(G)
    fits.pdf      every spectrum with its fit, 12 per page

Run (manually, on as many dataset folders as you like):
    python scripts/raman/raman_peak_fit.py ~/Downloads/Graphene/007_02 --n2d 4

CLI of Raman_Classifier ``raman_peak_fit.py``; the fitting is core/raman/peak_fit.py + batch.py,
the PDFs plotting/raman_figures.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tensorspec.core.raman import batch  # noqa: E402
from tensorspec.plotting import raman_figures  # noqa: E402


def run(folder, a):
    rows, out, summ, _ = batch.fit_folder(folder, a, figures=raman_figures)
    good = [r for r in rows if r.get("ok")]
    name = os.path.basename(os.path.normpath(folder))
    print("\n%s: %d spectra, %d fitted  ->  %s" % (name, len(rows), len(good), out))
    for q, _, v in summ:
        if v.size:
            print("   %-10s %10.4g ± %-8.3g (n=%d)" % (q, v.mean(), v.std(ddof=1) if v.size > 1
                                                       else 0, v.size))
    return rows, out


def main():
    ap = argparse.ArgumentParser(description="Lorentzian G/2D fits per dataset folder")
    ap.add_argument("datasets", nargs="+", help="folders of two-column spectra")
    ap.add_argument("--ng", default="auto", choices=["auto", "1", "2"],
                    help="G as 1 Lorentzian, 2 (G-, G+), or auto (default: 2 only "
                         "when clearly split)")
    ap.add_argument("--n2d", type=int, default=1, help="Lorentzians for 2D (default 1; 4 for AB bilayer)")
    ap.add_argument("--g-range", type=float, nargs=2, default=(1500, 1680))
    ap.add_argument("--d-range", type=float, nargs=2, default=(2560, 2810))
    ap.add_argument("--out", help="output folder (default <dataset>/output)")
    a = ap.parse_args()
    a.ng = a.ng if a.ng == "auto" else int(a.ng)
    for d in a.datasets:
        run(os.path.expanduser(d), a)


if __name__ == "__main__":
    main()
