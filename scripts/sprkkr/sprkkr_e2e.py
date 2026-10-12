#!/usr/bin/env python3
"""Generic SPR-KKR SCF -> ARPES end-to-end runner, driven from a CIF.

Generalizes cu001_e2e.py (which stays untouched as the Cu-specific smoke
test) to any structure: builds a (primitive by default) structure from a
CIF, runs SCF, works out the surface geometry from the resulting .pot
(layer stack, IQ_AT_SURF, and the raw-ABAS-frame equivalent of the
requested Miller index -- design doc §0 / tensorspec.core.dft.sprkkr.geometry),
then runs ARPES with CRYS_VECS + that raw-frame index so the surface really
is the one requested in the CIF's conventional frame.

Run from repo root:
    TensorSpec_env/bin/python scripts/sprkkr/sprkkr_e2e.py --cif structure.cif \\
        --hkl 0 0 1 --target local --bin-dir /path/to/sprkkr/bin --nproc 2 \\
        --small --out /tmp/e2e_test

As root, mpirun refuses to run without --allow-run-as-root, so for nproc>1
locally pass e.g.:
    --mpi-prefix "mpirun --allow-run-as-root --oversubscribe -np {n}"

``--dry-run`` builds SCF+ARPES inputs (does NOT launch any binary), prints
the layer stack / IQ_AT_SURF / hkl_abas / NQ, and exits 0 -- use it to sanity
check a slow/large material before committing to a real SCF run.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

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
    angle_points,
    build_arpes_inputs,
    build_scf_inputs,
    format_eta,
    load_settings,
    pot_key,
    resolve_surface_geometry,
    run_arpes,
    run_scf,
)
from tensorspec.core.dft.sprkkr.geometry import (  # noqa: E402
    ANGSTROM_PER_BOHR,
    atoms_per_plane_max,
    hkl_to_abas_frame,
    layer_stack,
    parse_pot_geometry,
    pick_surface_site,
)
from tensorspec.core.compute import cluster_paths  # noqa: E402
from tensorspec.gui.services.cluster_utils import find_cluster_by_name  # noqa: E402


def _tail(path, n=30) -> str:
    p = Path(path)
    if not p.exists():
        return f"<no log at {p}>"
    lines = p.read_text(errors="ignore").splitlines()
    return "\n".join(lines[-n:])


def build_structure(cif_path: str, primitive: bool):
    from pymatgen.core import Structure

    s = Structure.from_file(cif_path)
    return s.get_primitive_structure() if primitive else s


def resolve_hkl_abas(cif_lattice_matrix, pot_geom, hkl):
    """hkl given in the CIF's OWN cell frame -> hkl in the raw ABAS frame of the
    .pot SPR-KKR will read (written with CRYS_VECS so SPR-KKR uses ABAS as-is).

    Frame of reference = the lattice exactly as read from the CIF (what the
    user means by h k l), NOT a pymatgen re-standardized cell (axis setting
    may differ -> wrong plane). No silent fallback: a non-integer result is
    an error (would build the wrong surface).
    """
    abas_cart = pot_geom.abas * pot_geom.alat_bohr * ANGSTROM_PER_BOHR
    hkl_abas = hkl_to_abas_frame(cif_lattice_matrix, hkl, abas_cart)
    return hkl_abas, True


def print_layer_stack(pot_geom, hkl_abas) -> None:
    planes = layer_stack(pot_geom, hkl_abas)
    print(f"[geom] NQ={len(pot_geom.sites)} hkl_abas={hkl_abas} planes={len(planes)}")
    for i, pl in enumerate(planes):
        species = ",".join(sorted(set(pl.species)))
        print(f"[geom]   plane {i}: proj={pl.proj_alat:.4f} n={len(pl.iqs)} iqs={pl.iqs} species=[{species}]")


def make_scf_poll_cb(heartbeat_s: float = 120.0):
    """Print only when the SCF status actually changes.

    The previous version printed every 2 s regardless, which buries a long
    first iteration (core states + BZ integration emit nothing) under hundreds
    of identical "iter=0" lines. A heartbeat still fires every `heartbeat_s`
    so a live-but-quiet job is distinguishable from a hung one.
    """
    last_state = [None]
    last_print = [0.0]
    t0 = [time.time()]

    def cb(status):
        now = time.time()
        state = (status.iterations, status.last_err, status.ef_ry)
        changed = state != last_state[0]
        if not changed and now - last_print[0] < heartbeat_s:
            return
        last_state[0] = state
        last_print[0] = now
        mins = (now - t0[0]) / 60.0
        tag = "" if changed else "  (no change)"
        print(
            f"[scf] t={mins:5.1f}m iter={status.iterations} "
            f"err={status.last_err} EF={status.ef_ry}{tag}",
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
    ap = argparse.ArgumentParser(description="Generic CIF -> SPR-KKR SCF+ARPES e2e runner")
    ap.add_argument("--cif", default=None, help="input CIF (required unless --pot given and --scf-only omitted)")
    ap.add_argument("--hkl", type=float, nargs=3, default=(0.0, 0.0, 1.0), metavar=("H", "K", "L"),
                     help="Miller index, conventional CIF frame (default 0 0 1)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--primitive", dest="primitive", action="store_true", default=True)
    grp.add_argument("--no-primitive", dest="primitive", action="store_false")
    ap.add_argument("--nonmag", action="store_true")
    ap.add_argument("--nl", type=int, default=3)
    ap.add_argument("--nktab", type=int, default=250)
    ap.add_argument("--iq-surf", default="auto", help="'auto' (pick_surface_site) or an integer IQ")
    ap.add_argument("--hv", type=float, default=21.2)
    ap.add_argument("--pol", default="P", choices=["P", "S", "C+", "C-"])
    ap.add_argument("--theta-ph", type=float, default=45.0, help="light incidence polar angle from surface normal (deg)")
    ap.add_argument("--phi-ph", type=float, default=0.0, help="light incidence azimuth (deg)")
    ap.add_argument("--ework", type=float, default=4.5, help="work function eV")
    ap.add_argument("--imv-fin", type=float, default=2.0, help="final-state imaginary potential eV")
    ap.add_argument("--theta", type=float, nargs=3, default=(-20.0, 20.0, 21), metavar=("MIN", "MAX", "N"))
    ap.add_argument("--phi", type=float, nargs=3, default=(0.0, 0.0, 1), metavar=("MIN", "MAX", "N"))
    ap.add_argument("--slit", type=float, default=0.0, help="analyzer slit orientation in lab, deg")
    ap.add_argument("--deflector", type=float, default=0.0, help="deflector angle, deg")
    ap.add_argument("--manip-theta", type=float, default=0.0)
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--azimuth", type=float, default=0.0)
    ap.add_argument("--phi-offset", type=float, default=0.0, help="UNCALIBRATED SPR-KKR PHI zero offset")
    ap.add_argument("--ref-energy", type=float, default=0.0, help="eV rel. E_F where (Theta,Phi) are evaluated")
    ap.add_argument("--pointwise", action="store_true", help="one kkrspec job per slit angle")
    ap.add_argument("--erange", type=float, nargs=3, default=(-6.0, 1.0, 71), metavar=("MIN", "MAX", "N"))
    ap.add_argument("--n-layer", type=int, default=50)
    ap.add_argument("--nlat-g-vec", type=int, default=57)
    ap.add_argument("--target", default="local")
    ap.add_argument("--bin-dir", default=None)
    ap.add_argument("--nproc", type=int, default=None)
    ap.add_argument("--mode", default="auto", choices=["auto", "mpi", "chunks"])
    ap.add_argument("--small", action="store_true", help="theta=3pts erange=2pts (quick check)")
    ap.add_argument("--pot", default=None, help="reuse this converged pot; skip SCF")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scf-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                     help="build SCF+ARPES inputs only, no binary launch; print geometry and exit 0")
    ap.add_argument("--mpi-prefix", default=None)
    ap.add_argument(
        "--raw", action="append", default=[], metavar="SECTION.KEY=VALUE",
        help="inject an extra raw .inp keyword not in ArpesParams, e.g. "
             "--raw SPEC_EL.TYP=3 --raw SPEC_EL.BETA1=-15.0 (repeatable; "
             "2026-09-12 design-doc probe, see docs/superpowers/specs/"
             "2026-09-11-sprkkr-full-geometry-design.md §7)",
    )
    return ap.parse_args()


def _parse_raw_overrides(raw_args) -> dict:
    """"SECTION.KEY=VALUE" list -> {SECTION: {KEY: value}}. VALUE is parsed as
    float when possible, {a,b,c} as a list of floats, else kept as a string
    (ase2sprkkr's own setattr validates/rejects it either way)."""
    out: dict = {}
    for item in raw_args:
        path, _, value_str = item.partition("=")
        section, _, key = path.partition(".")
        if not section or not key or not value_str:
            raise ValueError(f"--raw expects SECTION.KEY=VALUE, got: {item!r}")
        value_str = value_str.strip()
        if value_str.startswith("{") and value_str.endswith("}"):
            value = [float(v) for v in value_str[1:-1].split(",")]
        else:
            try:
                value = float(value_str)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                value = value_str
        out.setdefault(section, {})[key] = value
    return out


def main() -> int:
    args = parse_args()
    extra_raw = _parse_raw_overrides(args.raw)
    ts = time.strftime("%Y%m%d_%H%M%S")

    if args.pot is None and args.cif is None:
        print("ERROR: need --cif (or --pot to skip SCF)", file=sys.stderr)
        return 1

    settings = load_settings()
    bin_dir = args.bin_dir or settings.bin_dir
    nproc = args.nproc if args.nproc is not None else settings.nproc

    out_dir = Path(args.out) if args.out else (REPO_ROOT / "scratch" / "sprkkr_e2e" / ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    hkl_conv = tuple(int(round(v)) for v in args.hkl)
    theta_lo, theta_hi, theta_n = args.theta
    phi_lo, phi_hi, phi_n = args.phi
    e_lo, e_hi, e_n = args.erange
    if args.small:
        theta_n, e_n = 3, 2

    is_remote = args.target != "local"
    launcher = None
    cluster = None
    # Resolve the cluster even for --dry-run: it is a local read of
    # ~/.tensorspec_clusters.json (no network, no ssh), and remote_scf_workdir /
    # remote_arpes_workdir below dereference `cluster` whenever is_remote, so
    # leaving it None made "--dry-run --target <cluster>" crash in
    # cluster_paths.job_dir. Only the launcher needs the live connection.
    if is_remote:
        cluster = find_cluster_by_name(args.target)
        if cluster is None:
            print(f"ERROR: no cluster named {args.target!r} in ~/.tensorspec_clusters.json", file=sys.stderr)
            return 1
    if not args.dry_run:
        if is_remote:
            launcher = RemoteLauncher(cluster)
        else:
            mpi_prefix = args.mpi_prefix.format(n=nproc) if args.mpi_prefix else None
            launcher = LocalLauncher(bin_dir, mpi_prefix=mpi_prefix)

    structure = build_structure(args.cif, args.primitive) if args.cif else None
    # hkl frame = the CIF cell exactly as written (user's frame), even if we run the primitive cell.
    from pymatgen.core import Structure as _Structure  # noqa: E402
    cif_lattice = _Structure.from_file(args.cif).lattice.matrix if args.cif else None
    formula = structure.composition.reduced_formula if structure is not None else "pot"

    scf_workdir = out_dir / "scf"
    arpes_workdir = out_dir / "arpes"
    remote_scf_workdir = f"{cluster_paths.job_dir(cluster, 'sprkkr')}/e2e_{ts}/scf" if is_remote else None
    remote_arpes_workdir = f"{cluster_paths.job_dir(cluster, 'sprkkr')}/e2e_{ts}/arpes" if is_remote else None

    current_log = None
    try:
        scf_params = ScfParams(nl=args.nl, nktab=args.nktab, nonmag=args.nonmag, dataset=formula)

        pot_path: str
        if args.pot:
            print(f"[scf] skipped, reusing pot: {args.pot}")
            pot_path = args.pot
        elif args.dry_run:
            scf_inputs = build_scf_inputs(structure, scf_params, scf_workdir)
            print(f"[scf] dry-run: wrote {scf_inputs.inp_path}, {scf_inputs.pot_path} (no launch)")
            pot_path = str(scf_inputs.pot_path)
        else:
            current_log = scf_workdir / f"{scf_params.dataset}_SCF.out"
            scf_result = run_scf(
                structure, scf_params, workdir=scf_workdir, launcher=launcher, nproc=nproc,
                poll_cb=make_scf_poll_cb(), remote_workdir=remote_scf_workdir,
            )
            pot_path = scf_result.pot_path
            print(
                f"[scf] done wall={scf_result.wall_s:.1f}s EF={scf_result.status.ef_ry} "
                f"converged={scf_result.status.converged} pot={pot_path}"
            )
            vault = Vault(Path(settings.vault_root))
            key = pot_key(structure, scf_params)
            vault_name = f"{formula}_{ts}"
            vault.register(
                key=key, name=vault_name, pot_path=pot_path,
                cluster=args.target if is_remote else None,
                meta={"formula": formula, "target": args.target, "hkl_conv": hkl_conv},
            )
            print(f"[vault] registered {vault_name} key={key}")

        # --- geometry: layer stack, IQ_AT_SURF, hkl_abas (core, shared w/ GUI B3) ---
        pot_geom = parse_pot_geometry(pot_path)
        iq_arg = None if args.iq_surf == "auto" else int(args.iq_surf)
        geom = resolve_surface_geometry(pot_path, hkl_conv, cif_lattice_matrix=cif_lattice, iq_at_surf=iq_arg)
        hkl_abas = geom.hkl_abas
        iq_at_surf = geom.iq_at_surf
        n_max = geom.atoms_per_plane_max
        print_layer_stack(pot_geom, hkl_abas)
        print(f"[geom] IQ_AT_SURF={iq_at_surf} hkl_abas={hkl_abas} hkl_frame={'abas' if cif_lattice is not None else 'abas(fallback=conv)'} atoms_per_plane_max={n_max}")

        if args.scf_only:
            print("[arpes] skipped (--scf-only)")
            return 0

        arpes_params = ArpesParams(
            hv_eV=args.hv, pol_p=args.pol,
            theta_ph=args.theta_ph, phi_ph=args.phi_ph,
            ework_eV=args.ework, imv_fin_eV=args.imv_fin,
            theta_e=(theta_lo, theta_hi), nt=int(theta_n),
            phi_e=(phi_lo, phi_hi), np_=int(phi_n),
            e_min_eV=e_lo, e_max_eV=e_hi, ne=int(e_n),
            hkl=hkl_abas, hkl_frame="abas", iq_at_surf=iq_at_surf,
            n_layer=args.n_layer, nlat_g_vec=args.nlat_g_vec,
            nl=args.nl, nktab=args.nktab, dataset=f"{formula}_arpes",
        )

        pts = None
        if args.pointwise:
            pts = angle_points(
                arpes_params.theta_e, arpes_params.nt,
                hv_eV=arpes_params.hv_eV, work_function_eV=arpes_params.ework_eV,
                deflector_deg=args.deflector, slit_rot_deg=args.slit,
                manip_theta_deg=args.manip_theta, manip_azimuth_deg=args.azimuth,
                manip_tilt_deg=args.tilt, ref_energy_eV=args.ref_energy,
                phi_offset_deg=args.phi_offset,
            )
            for p in pts:
                print(f"[pt] i={p.index:02d} slit={p.slit_deg:+.2f} Theta={p.theta_e_deg:+.2f} Phi={p.phi_e_deg:+.2f} k_par={p.k_par:.4f} k_slit={p.k_slit:+.4f}")
            crosses_gamma = any(abs(p.k_par) < 0.01 for p in pts)
            min_k_par = min(p.k_par for p in pts) if pts else 0.0
            max_k_par = max(p.k_par for p in pts) if pts else 0.0
            print(f"[pt] N={len(pts)} min|k_par|={min_k_par:.4f} max|k_par|={max_k_par:.4f} crosses_gamma={crosses_gamma}")

        if args.dry_run:
            if args.pointwise:
                inp_count = 0
                for p in pts:
                    sub_params = replace(arpes_params,
                                        theta_e=(p.theta_e_deg, p.theta_e_deg), nt=1,
                                        phi_e=(p.phi_e_deg, p.phi_e_deg), np_=1,
                                        dataset=f"{arpes_params.dataset}_p{p.index:04d}")
                    sub_workdir = arpes_workdir / f"{arpes_params.dataset}_p{p.index:04d}"
                    arpes_inputs = build_arpes_inputs(pot_path, sub_params, sub_workdir, extra_raw=extra_raw)
                    inp_count += 1
                print(f"[arpes] dry-run: wrote {inp_count} .inp under {arpes_workdir}")
            else:
                arpes_inputs = build_arpes_inputs(pot_path, arpes_params, arpes_workdir, extra_raw=extra_raw)
                print(f"[arpes] dry-run: wrote {arpes_inputs.inp_path} (no launch)")
            if extra_raw:
                print(f"[arpes] --raw applied: {extra_raw}")
            print(f"[cost] atoms_per_plane_max={n_max} n_points={arpes_params.n_points}")
            print(f"[dry-run] OK NQ={len(pot_geom.sites)} IQ_AT_SURF={iq_at_surf} hkl_abas={hkl_abas}")
            return 0

        current_log = arpes_workdir / f"{arpes_params.dataset}_ARPES.out"
        result = run_arpes(
            pot_path, arpes_params, workdir=arpes_workdir, launcher=launcher, nproc=nproc,
            mode=args.mode, progress_cb=make_arpes_progress_cb(), remote_workdir=remote_arpes_workdir,
            extra_raw=extra_raw,
            angle_points=pts if args.pointwise else None, deflector_deg=args.deflector,
        )

        ds = result.dataset
        dims = {k: v for k, v in ds.sizes.items()}
        i_tot_max = float(ds["I_tot"].max())
        print(f"[arpes] done wall={result.wall_s:.1f}s dims={dims} I_tot_max={i_tot_max:.6g}")

        npz_path = out_dir / f"{formula}_arpes.npz"
        np.savez(
            npz_path,
            intensity=ds["I_tot"].values, energy=ds["energy"].values,
            theta=ds["theta"].values, phi=ds["phi"].values,
        )
        print(f"[out] saved {npz_path}")

        # Data-Viewer-loadable twin: intensity(kx,ky,E) + kx/ky/E/metadata
        # (SimulatedARPESLoader contract). Pointwise -> exact lab k from the sidecar.
        try:
            from tensorspec.core.dft.sprkkr.viewer_export import arrays_to_viewer_npz
            viewer_path = out_dir / f"{formula}_arpes_viewer.npz"
            sidecar = arpes_workdir / "pointwise_points.json"
            k_par_e0 = None
            if not (args.pointwise and sidecar.is_file()):
                ie = int(np.argmin(np.abs(ds["energy"].values)))
                k_par_e0 = np.abs(ds["k_par"].isel(energy=ie, phi=0).values)
            arrays_to_viewer_npz(
                ds["I_tot"].values, ds["energy"].values, ds["theta"].values, ds["phi"].values,
                str(viewer_path),
                points_json=str(sidecar) if (args.pointwise and sidecar.is_file()) else None,
                k_par_e0=k_par_e0,
                metadata={
                    "hv_eV": arpes_params.hv_eV, "pol": arpes_params.pol_p,
                    "theta_ph_deg": arpes_params.theta_ph, "ework_eV": arpes_params.ework_eV,
                    "hkl": list(arpes_params.hkl), "iq_at_surf": arpes_params.iq_at_surf,
                    "slit_deg": args.slit, "deflector_deg": args.deflector,
                    "manip_theta_deg": args.manip_theta, "manip_tilt_deg": args.tilt,
                    "manip_azimuth_deg": args.azimuth, "phi_offset_deg": args.phi_offset,
                    "pointwise": bool(args.pointwise), "pot": str(pot_path),
                    "fermi_edge_applied": False,
                },
            )
            print(f"[out] viewer npz -> {viewer_path}  (ARPES suite: Load ARPES Data)")
        except Exception as exc:  # never lose the run over an export nicety
            print(f"[out] viewer npz export failed: {exc}")

        for spc in result.spc_paths:
            dest = out_dir / Path(spc).name
            if str(Path(spc).resolve()) != str(dest.resolve()):
                shutil.copy2(spc, dest)
            print(f"[out] spc -> {dest}")

        eta_store = Path("~/.tensorspec_sprkkr_eta.json").expanduser()
        eta_model = EtaModel(store_path=str(eta_store))
        t_point = eta_model.calibrate(arpes_params, nproc, result.wall_s, host=args.target, material=formula)
        print(
            f"[eta] calibrated t_point_s={t_point:.4f} (host={args.target} material={formula}) "
            f"eta_full={format_eta(eta_model.estimate_seconds(ArpesParams(), nproc, host=args.target, material=formula))}"
        )
        print(f"[cost] atoms_per_plane_max={n_max}")

        return 0

    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        if current_log is not None:
            print(f"--- log tail ({current_log}) ---", file=sys.stderr)
            print(_tail(current_log, 30), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
