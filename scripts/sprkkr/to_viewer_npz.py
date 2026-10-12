#!/usr/bin/env python3
"""Export SPR-KKR runs to TensorSpec Data-Viewer .npz (intensity(kx,ky,E), kx, ky, E, metadata).

One run dir  -> <run>/pot_arpes_viewer.npz
  python scripts/sprkkr/to_viewer_npz.py scratch/sprkkr_e2e/pointwise_vte2_100x200

Fermi map    -> stack every defl_*/ cube under a parent dir along ky
  python scripts/sprkkr/to_viewer_npz.py scratch/sprkkr_e2e/fmap_vte2 --stack --out scratch/sprkkr_e2e/fmap_vte2/vte2_fermi_map.npz

Load in the GUI: ARPES suite -> Load ARPES Data -> pick the *_viewer.npz / *_fermi_map.npz.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tensorspec.core.dft.sprkkr.viewer_export import (  # noqa: E402
    VIEWER_SUFFIX,
    run_dir_to_viewer_npz,
    stack_viewer_cubes,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="run dir, or parent dir of defl_*/ run dirs with --stack")
    ap.add_argument("--stack", action="store_true", help="stack defl_*/ cubes into one Fermi-map cube")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.stack:
        out = run_dir_to_viewer_npz(args.path, args.out)
        import numpy as np
        d = np.load(out, allow_pickle=True)
        print(f"[viewer] {out}  intensity{d['intensity'].shape} kx[{d['kx'].min():.3f},{d['kx'].max():.3f}] "
              f"ky={np.round(d['ky'],3).tolist()} E[{d['E'].min():.2f},{d['E'].max():.2f}]")
        return

    run_dirs = sorted(p for p in glob.glob(os.path.join(args.path, "*")) if os.path.isdir(p)
                      and glob.glob(os.path.join(p, "*_arpes.npz")))
    if not run_dirs:
        sys.exit(f"no run dirs with *_arpes.npz under {args.path}")
    cubes = []
    for rd in run_dirs:
        existing = [p for p in glob.glob(os.path.join(rd, f"*{VIEWER_SUFFIX}"))]
        cubes.append(existing[0] if existing else run_dir_to_viewer_npz(rd))
        print(f"[viewer] cube: {cubes[-1]}")
    out = args.out or os.path.join(args.path, "fermi_map_viewer.npz")
    stack_viewer_cubes(cubes, out)
    import numpy as np
    d = np.load(out, allow_pickle=True)
    print(f"[viewer] STACKED {out}  intensity{d['intensity'].shape} "
          f"kx[{d['kx'].min():.3f},{d['kx'].max():.3f}] ky={np.round(d['ky'],3).tolist()} "
          f"E[{d['E'].min():.2f},{d['E'].max():.2f}]")


if __name__ == "__main__":
    main()
