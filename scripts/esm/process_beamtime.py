"""Process an NSLS-II ESM XPEEM beamtime folder (Elmitec UView .dat).

    python scripts/esm/process_beamtime.py "<data>/BNL/<proposal>" [--plan] [--redo]

Outputs go to <date>_<sample>/tensorspec_xpeem/ (clusters/, ld_splits/, spectra/,
leem/, index.csv); the beamtime folder gets tensorspec_xpeem_index.csv and
tensorspec_xpeem_catalog.csv. Existing cluster outputs are reused unless --redo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tensorspec.core.peem.esm import batch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--convention", default="cp_minus_cm", choices=["cp_minus_cm", "cm_minus_cp"],
                    help="cp_minus_cm: pcd = (CL-CR)/(CL+CR)")
    a = ap.parse_args()
    root = Path(a.root)
    if a.plan:
        cat = batch.catalog(root)
        groups = batch.plan(cat)
        print(f"{len(cat['acqs'])} acquisitions, {len(cat['leem'])} LEEM images, {len(groups)} clusters")
        for g in groups:
            print(f"  {g.gid:55s} units={len(g.units)} unpaired={[a.pol for a in g.unpaired]}")
        for x in cat["acqs"]:
            if x.kind == "XAS":
                print("  XAS", x.sample, x.t_label, Path(x.path).name, x.n_files, "frames")
        return 0
    from tensorspec.plotting.ue49_preview import Preview

    batch.run(root, batch.Settings(convention=a.convention), preview=Preview(),
              log=lambda m: print(m, flush=True), skip_existing=not a.redo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
