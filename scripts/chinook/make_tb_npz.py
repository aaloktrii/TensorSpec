#!/usr/bin/env python3
"""Build tb_data.npz (+ arpes_physics.json) for chinook_remote_runner.py from a
Wannier90 hr.dat + a structure file, without going through the GUI.

Why this exists: the GUI builds tb_data.npz in memory and SFTPs it to the cluster.
For a large third-party Wannier90 export (VTe2 CDW: 324 spinor WFs x 1265 R-vectors
= 6.2 GB of text) it is far better to build the npz once, next to the hr.dat, and
copy only that: at hop_tol = 1e-4 the surviving model is ~1.7M hops, about 38 MB.

    python scripts/chinook/make_tb_npz.py \
        --hr   /path/to/wannier90_hr.dat \
        --cif  /path/to/structure.cif \
        --out  /path/to/tb_data.npz

The Fermi energy is read from FERMI_ENERGY.txt beside the hr.dat unless --fermi is
given; it is folded into the on-site terms, so the eigenvalues of the resulting model
are already relative to E_F and `e_fermi` in the npz is 0.

Run it with the environment that has pymatgen (on a mac GUI install:
TensorSpec_env/bin/python).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def read_fermi(work_dir):
    p = os.path.join(work_dir, "FERMI_ENERGY.txt")
    if not os.path.isfile(p):
        return None
    # e.g. "     the Fermi energy is    12.4202 ev"
    for tok in open(p).read().split():
        try:
            return float(tok)
        except ValueError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr", required=True, help="wannier90_hr.dat (the real one; it is streamed)")
    ap.add_argument("--cif", required=True, help="structure file pymatgen can read (.cif, POSCAR, ...)")
    ap.add_argument("--out", default="tb_data.npz")
    ap.add_argument("--hop-tol", type=float, default=1e-4,
                    help="drop |t| below this (eV). Default 1e-4 = the ARPES default.")
    ap.add_argument("--fermi", type=float, default=None,
                    help="E_F in eV (QE scale). Default: FERMI_ENERGY.txt beside the hr.dat.")
    ap.add_argument("--use-soc", action="store_true",
                    help="only for a NON-spinor hr.dat; a spinor one is already doubled.")
    args = ap.parse_args()

    from pymatgen.core import Structure

    from tensorspec.core.dft.chinook_tb import ChinookTightBindingEngine

    work_dir = os.path.dirname(os.path.abspath(args.hr))
    ef = args.fermi if args.fermi is not None else read_fermi(work_dir)
    if ef is None:
        raise SystemExit("no Fermi energy: pass --fermi or put FERMI_ENERGY.txt beside the hr.dat")

    eng = ChinookTightBindingEngine()
    eng.crystal_structure = Structure.from_file(args.cif)
    print(f"structure : {eng.crystal_structure.composition.reduced_formula}, "
          f"{eng.crystal_structure.num_sites} sites")
    print(f"E_F (QE)  : {ef} eV")
    print(f"hop_tol   : {args.hop_tol:g} eV")

    tb_dict, basis_args = eng.export_wannier_dictionary(
        args.hr, use_soc=args.use_soc, onsite_e=0.0, hop_tol=args.hop_tol, qe_fermi=ef
    )

    h_list = tb_dict["list"]
    indices = np.array([[h[0], h[1], h[2], h[3], h[4]] for h in h_list], dtype=np.float64)
    values = np.array([h[5] for h in h_list], dtype=np.complex128)

    pos, Z, orbs = basis_args["pos"], basis_args["Z"], basis_args["orbs"]
    basis_list = []
    for i in range(len(pos)):
        lab = orbs[i][0] if isinstance(orbs[i], (list, tuple)) else orbs[i]
        zi = Z[i] if not isinstance(Z, dict) else Z[i]
        basis_list.append({"pos": [float(x) for x in pos[i]], "label": str(lab),
                           "spin": 1.0, "Z": int(zi)})

    a_mat = np.asarray(tb_dict["a"], dtype=float)
    b_matrix = np.asarray(eng.crystal_structure.lattice.reciprocal_lattice.matrix, dtype=float)

    np.savez_compressed(
        args.out,
        indices=indices, values=values, basis_list=basis_list,
        a_mat=a_mat, b_matrix=b_matrix,
        e_fermi=0.0,              # E_F already folded into the on-site terms
        onsite_e=0.0,
        fermi_energy_qe=float(ef),
        h_includes_onsite=True,
    )
    print(f"hoppings  : {indices.shape[0]:,}")
    print(f"basis     : {len(basis_list)} orbitals, source={basis_args.get('basis_source')}, "
          f"spinors={basis_args.get('wannier_spinors')}")
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
