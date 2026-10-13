"""1L / 2L reference values from labelled Raman datasets, and layer-number / stacking predictions.

Build the reference (fits every spectrum, stores the statistics and plots):
    python scripts/raman/graphene_reference.py build --ref 1L ~/Downloads/ML_hBN_RuCl3_532nm_map1_2__X_-2372 \
                                                     --ref 2L ~/Downloads/BL_hBN_RuCl3_532nm_map1_7__X_-2367
Predict a new dataset (folder of spectra; writes <folder>/output/predictions.csv and prediction.pdf):
    python scripts/raman/graphene_reference.py predict ~/Downloads/new_dataset

'build' writes to TensorSpec_Workspace/raman/reference/ (the workspace folder of the current
directory, git-ignored user data). 'predict' uses that reference when it exists, otherwise the
shipped read-only default tensorspec/core/raman/data/graphene_reference.json.
CLI of Raman_Classifier ``graphene_reference.py``; the method is described in core/raman/reference.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tensorspec.core.io.raman_loaders import SPECTRUM_EXT  # noqa: E402
from tensorspec.core.raman.reference import preferred_reference, user_reference_path  # noqa: E402
from tensorspec.core.raman_engine import RamanEngineRouter  # noqa: E402
from tensorspec.plotting.backends import raman_matplotlib_engine  # noqa: E402

#: same folder as core/workspace.py's project_dir, without starting the workspace manager
WORKSPACE = Path.cwd() / "TensorSpec_Workspace"


def main():
    ap = argparse.ArgumentParser(description="graphene 1L/2L reference and prediction")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ref", nargs=2, action="append", metavar=("LABEL", "FOLDER"),
                   required=True, help="LABEL is 1L or 2L")
    b.add_argument("--json", default=user_reference_path(WORKSPACE))
    p = sub.add_parser("predict")
    p.add_argument("folders", nargs="+")
    p.add_argument("--json", default=preferred_reference(WORKSPACE))
    a = ap.parse_args()
    engine = RamanEngineRouter()
    if a.cmd == "build":
        engine.build_reference([(lab, f) for lab, f in a.ref], a.json, figures=raman_matplotlib_engine)
    else:
        if not os.path.exists(a.json):
            raise SystemExit("No reference at %s - run 'build' first." % a.json)
        ref = engine.load_reference(a.json)
        bad = 0
        for f in a.folders:
            d = os.path.expanduser(f)
            if not os.path.isdir(d):
                print("skipped %s: no such folder" % f)
                bad += 1
                continue
            if not any(n.lower().endswith(SPECTRUM_EXT) for n in os.listdir(d)):
                print("skipped %s: no .txt/.dat/.csv spectra in it" % f)
                bad += 1
                continue
            engine.predict_folder(d, ref, figures=raman_matplotlib_engine)
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
