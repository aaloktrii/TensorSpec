#!/usr/bin/env python3
"""Cu(001) SPR-KKR end-to-end smoke test: SCF -> converged pot -> ARPES.

Standalone CLI. Zero Qt (imports only tensorspec.core.dft.sprkkr and
tensorspec.gui.services.cluster_utils.load_clusters/find_cluster_by_name,
neither of which touch Qt at module load or in the functions used here).

Run from repo root:
    TensorSpec_env/bin/python scripts/sprkkr/cu001_e2e.py --target local \
        --bin-dir /path/to/sprkkr/bin --nproc 2 --small --out /tmp/e2e_test

As root, mpirun refuses to run without --allow-run-as-root, so for nproc>1
locally pass e.g.:
    --mpi-prefix "mpirun --allow-run-as-root --oversubscribe -np {n}"
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# repo root = parents[2] of this script (scripts/sprkkr/cu001_e2e.py)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from tensorspec.core.dft.sprkkr import (  # noqa: E402
    ArpesParams,
    EtaModel,
    LocalLauncher,
    RemoteLauncher,
    ScfParams,
    Vault,
    format_eta,
    load_settings,
    pot_key,
    run_arpes,
    run_scf,
)
from tensorspec.core.compute import cluster_paths  # noqa: E402
from tensorspec.gui.services.cluster_utils import find_cluster_by_name  # noqa: E402


def _tail(path, n=30) -> str:
    p = Path(path)
    if not p.exists():
        return f"<no log at {p}>"
    lines = p.read_text(errors="ignore").splitlines()
    return "\n".join(lines[-n:])


def build_structure():
    from pymatgen.core import Lattice, Structure

    s = Structure(
        Lattice.cubic(3.615),
        ["Cu"] * 4,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )
    return s.get_primitive_structure()


def make_scf_poll_cb():
    last = [0.0]

    def cb(status):
        now = time.time()
        if now - last[0] < 2.0:
            return
        last[0] = now
        print(
            f"[scf] iter={status.iterations} err={status.last_err} EF={status.ef_ry}",
            flush=True,
        )

    return cb


def make_arpes_progress_cb():
    last_bucket = [-1]

    def cb(frac):
        bucket = int(frac * 100) // 5
        if bucket == last_bucket[0]:
            return
        last_bucket[0] = bucket
        print(f"[arpes] {bucket * 5}%", flush=True)

    return cb


def parse_args():
    ap = argparse.ArgumentParser(description="Cu(001) SPR-KKR SCF+ARPES e2e smoke test")
    ap.add_argument("--target", default="local", help="'local' or a cluster name from ~/.tensorspec_clusters.json")
    ap.add_argument("--bin-dir", default=None, help="local SPR-KKR bin dir (default: settings.bin_dir)")
    ap.add_argument("--nproc", type=int, default=None, help="default: settings.nproc")
    ap.add_argument("--mode", default="auto", choices=["auto", "mpi", "chunks"])
    ap.add_argument("--small", action="store_true", help="ARPES NE=2 NT=3 quick check")
    ap.add_argument("--full", action="store_true", help="ARPES NE=71 NT=21 (default sizes, explicit)")
    ap.add_argument("--hv", type=float, default=21.2)
    ap.add_argument("--pot", default=None, help="reuse this converged pot; skip SCF")
    ap.add_argument("--out", default=None, help="default scratch/sprkkr_e2e/<ts>")
    ap.add_argument("--skip-arpes", action="store_true")
    ap.add_argument("--mpi-prefix", default=None, help='e.g. "mpirun --allow-run-as-root --oversubscribe -np {n}"')
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")

    settings = load_settings()
    bin_dir = args.bin_dir or settings.bin_dir
    nproc = args.nproc if args.nproc is not None else settings.nproc

    out_dir = Path(args.out) if args.out else (REPO_ROOT / "scratch" / "sprkkr_e2e" / ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_remote = args.target != "local"
    cluster = None
    if is_remote:
        cluster = find_cluster_by_name(args.target)
        if cluster is None:
            print(f"ERROR: no cluster named {args.target!r} in ~/.tensorspec_clusters.json", file=sys.stderr)
            return 1
        launcher = RemoteLauncher(cluster)
    else:
        mpi_prefix = args.mpi_prefix.format(n=nproc) if args.mpi_prefix else None
        launcher = LocalLauncher(bin_dir, mpi_prefix=mpi_prefix)

    structure = build_structure()

    scf_workdir = out_dir / "scf"
    arpes_workdir = out_dir / "arpes"

    remote_scf_workdir = f"{cluster_paths.job_dir(cluster, 'sprkkr')}/e2e_{ts}/scf" if is_remote else None
    remote_arpes_workdir = f"{cluster_paths.job_dir(cluster, 'sprkkr')}/e2e_{ts}/arpes" if is_remote else None

    current_log = None
    try:
        scf_params = ScfParams()  # ne=30, as usual, regardless of --small/--full

        if args.pot:
            print(f"[scf] skipped, reusing pot: {args.pot}")
            pot_path = args.pot
        else:
            current_log = scf_workdir / f"{scf_params.dataset}_SCF.out"
            scf_result = run_scf(
                structure,
                scf_params,
                workdir=scf_workdir,
                launcher=launcher,
                nproc=nproc,
                poll_cb=make_scf_poll_cb(),
                remote_workdir=remote_scf_workdir,
            )
            pot_path = scf_result.pot_path
            print(
                f"[scf] done wall={scf_result.wall_s:.1f}s EF={scf_result.status.ef_ry} "
                f"converged={scf_result.status.converged} pot={pot_path}"
            )

            vault = Vault(Path(settings.vault_root))
            key = pot_key(structure, scf_params)
            vault_name = f"Cu_e2e_{ts}"
            vault.register(
                key=key,
                name=vault_name,
                pot_path=pot_path,
                cluster=args.target if is_remote else None,
                meta={"structure": "Cu fcc a=3.615", "target": args.target},
            )
            print(f"[vault] registered {vault_name} key={key}")

        if args.skip_arpes:
            print("[arpes] skipped (--skip-arpes)")
            return 0

        if args.small:
            ne, nt = 2, 3
        else:
            ne, nt = 71, 21  # --full defaults (also the plain default)

        arpes_params = ArpesParams(hv_eV=args.hv, ne=ne, nt=nt)

        current_log = arpes_workdir / f"{arpes_params.dataset}_ARPES.out"
        result = run_arpes(
            pot_path,
            arpes_params,
            workdir=arpes_workdir,
            launcher=launcher,
            nproc=nproc,
            mode=args.mode,
            progress_cb=make_arpes_progress_cb(),
            remote_workdir=remote_arpes_workdir,
        )

        ds = result.dataset
        dims = {k: v for k, v in ds.sizes.items()}
        i_tot_max = float(ds["I_tot"].max())
        print(
            f"[arpes] done wall={result.wall_s:.1f}s dims={dims} I_tot_max={i_tot_max:.6g}"
        )

        npz_path = out_dir / "cu001_arpes.npz"
        np.savez(
            npz_path,
            intensity=ds["I_tot"].values,
            energy=ds["energy"].values,
            theta=ds["theta"].values,
            phi=ds["phi"].values,
        )
        print(f"[out] saved {npz_path}")

        for spc in result.spc_paths:
            dest = out_dir / Path(spc).name
            if str(Path(spc).resolve()) != str(dest.resolve()):
                shutil.copy2(spc, dest)
            print(f"[out] spc -> {dest}")

        eta_store = Path("~/.tensorspec_sprkkr_eta.json").expanduser()
        eta_model = EtaModel(store_path=str(eta_store))
        t_point = eta_model.calibrate(arpes_params, nproc, result.wall_s, host=args.target)
        print(f"[eta] calibrated t_point_s={t_point:.4f} (host={args.target}) eta_full={format_eta(eta_model.estimate_seconds(ArpesParams(), nproc))}")

        return 0

    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        if current_log is not None:
            print(f"--- log tail ({current_log}) ---", file=sys.stderr)
            print(_tail(current_log, 30), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
