#!/usr/bin/env python3
"""Add a hand-imposed spin splitting  H -> H + (Delta/2) n.sigma  to a Wannier spinor
tb_data.npz, on every orbital's own (up, down) pair at R = 0.

This is the "spin part only" Zeeman/exchange term of the VTe2 magnetic handoff (§7).
It is NOT self-consistent and omits the direct mu_B B.L coupling; orbital
repolarisation enters only through the SOC already in H. Say so in every output.

Basis contract (checked, not assumed): 324 spinor WFs, interleaved
(2m = orbital m up, 2m+1 = orbital m down), spin quantised along the QE Cartesian z,
and the npz Cartesian frame equal to QE's CELL_PARAMETERS.

    python add_zeeman.py --tb tb_data.npz --physics arpes_physics_th55_ph0.json \
        --delta-mev 20 --dir normal --out tb_B20_normal.npz
    --dir: normal | slit (+theta) | defl (+phi), from --physics via TensorSpec geometry;
           lab:x,y,z  field fixed in the Chinook lab frame (e.g. lab:0,0,-1 = vertical down),
                      rotated into the crystal with the cut's azimuth + hkl;
           or x,y,z (crystal Cartesian, any normalisation)
    --flip: reverse the field (n -> -n)

Rows are APPENDED (the original rows are untouched); the H builders accumulate
duplicate (i, j, R) entries, so this adds the term exactly. Both (i, j) and (j, i)
are written, like the Wannier hoppings already in the file.
"""
import argparse

import numpy as np

def geometry_axes(physics_file, b_matrix):
    """Surface normal, +slit (+theta) and +deflector (+phi) in the TB Cartesian frame,
    taken from TensorSpec's own build_k_bulk_mesh for this physics JSON (hkl, manip
    azimuth, slit angle), so the field follows exactly the geometry the cubes use."""
    import importlib.util
    import json
    import os
    import sys

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(root, "tensorspec/core/arpes/one_step/chinook_arpes_kmesh.py")
    spec = importlib.util.spec_from_file_location("chinook_arpes_kmesh", path)
    km = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("chinook_arpes_kmesh", km)
    spec.loader.exec_module(km)
    p = json.load(open(physics_file))

    def k_bulk(th, ph):
        k, *_ = km.build_k_bulk_mesh(
            {"X": [th, th, 1], "Y": [ph, ph, 1], "E": [0, 0, 1]},
            hv=84.0, work_function=4.5, inner_potential=12.0,
            slit_angle=p.get("slit_angle", 0.0), manip_theta=p.get("manip_theta", 0.0),
            manip_azimuth=p.get("manip_azimuth", 0.0), manip_tilt=p.get("manip_tilt", 0.0),
            incidence_angle=p.get("incidence_angle", 55.0), polarization=p.get("polarization", "Linear Horizontal"),
            hkl=tuple(p["hkl"]), B_matrix=b_matrix, fresnel_enabled=False)
        return k[:, 0]

    normal, ysurf = km.get_hkl_surface_frame(tuple(p["hkl"]), b_matrix, azimuthal_ref=np.array([0.0, 1.0, 0.0]))
    axes = {"normal": normal}
    # lab -> crystal, exactly as TensorSpec rotates the light vector A (manip + hkl frame)
    r_hkl = np.column_stack((np.cross(ysurf, normal), ysurf, normal))
    axes["_lab_to_bulk"] = lambda v: r_hkl @ km.sample_to_bulk_frame(
        np.asarray(v, float), p.get("manip_theta", 0.0), p.get("manip_azimuth", 0.0), p.get("manip_tilt", 0.0))
    for name, (a, b) in {"slit": ((1, 0), (-1, 0)), "defl": ((0, 1), (0, -1))}.items():
        v = k_bulk(*a) - k_bulk(*b)
        v -= (v @ normal) * normal
        axes[name] = v / np.linalg.norm(v)
    return axes


def field_direction(name, physics_file, b_matrix):
    if name in ("normal", "slit", "defl") or name.startswith("lab:"):
        if not physics_file:
            raise SystemExit(f"--dir {name} needs --physics (the cube's arpes_physics JSON)")
        named = geometry_axes(physics_file, np.asarray(b_matrix, float))
        if name.startswith("lab:"):
            # field fixed in the LAB (Chinook lab frame: normal +y, beam in x-y, slit 0 on x)
            v = named["_lab_to_bulk"]([float(x) for x in name[4:].split(",")])
            return v / np.linalg.norm(v)
        return named[name]
    v = np.array([float(x) for x in name.split(",")])
    return v / np.linalg.norm(v)


def check_basis(basis_list):
    labels = [b["label"] for b in basis_list]
    pos = np.array([b["pos"] for b in basis_list], float)
    n = len(labels)
    if n % 2 or labels[0::2] != labels[1::2]:
        raise SystemExit("basis is not interleaved spinor pairs (labels 2m != 2m+1)")
    dpos = np.abs(pos[0::2] - pos[1::2]).max()
    if dpos > 0.05:
        raise SystemExit(f"spinor partners sit {dpos:.3f} Å apart — not the same orbital?")
    return n


MU_B = 5.7883818060e-5  # eV / T
# Wannier label projection -> real-harmonic polynomial (Wannier90 / chinook order)
_REAL_HARM = {
    1: {"x": "x", "y": "y", "z": "z"},
    2: {"xy": "x*y", "yz": "y*z", "xz": "x*z", "XY": "(x**2-y**2)/2",
        "ZR": "(3*z**2-(x**2+y**2+z**2))/(2*sqrt(3))"},
}


def angular_momentum(l, projs):
    """(Lx, Ly, Lz) in the real-harmonic basis ``projs`` (hbar = 1), built by applying
    L = -i r x grad to the polynomials, so no Ylm phase convention can slip in."""
    import sympy as sp

    x, y, z = sp.symbols("x y z")
    fs = [sp.sympify(_REAL_HARM[l][p], locals={"x": x, "y": y, "z": z}) for p in projs]
    ops = [lambda f: -sp.I * (y * sp.diff(f, z) - z * sp.diff(f, y)),
           lambda f: -sp.I * (z * sp.diff(f, x) - x * sp.diff(f, z)),
           lambda f: -sp.I * (x * sp.diff(f, y) - y * sp.diff(f, x))]
    monos = sorted({m for f in fs for m in sp.Poly(sp.expand(f), x, y, z).monoms()} |
                   {m for f in fs for op in ops for m in sp.Poly(sp.expand(op(f)) + x ** l, x, y, z).monoms()})

    def vec(f):
        P = sp.Poly(sp.expand(f), x, y, z)
        return np.array([complex(P.coeff_monomial(m)) for m in monos])

    F = np.array([vec(f) for f in fs]).T                       # monomial x basis
    out = []
    for op in ops:
        G = np.array([vec(op(f)) for f in fs]).T
        c, res, *_ = np.linalg.lstsq(F, G, rcond=None)
        if np.abs(F @ c - G).max() > 1e-10:
            raise RuntimeError(f"L does not close on l={l} basis {projs}")
        out.append(c)                                          # equal-norm basis => matrix
    return out


def orbital_zeeman_rows(basis_list, bvec_ev):
    """mu_B B.L on each atom's p and d shells, identical for both spin components."""
    labels = [b["label"] for b in basis_list]
    pos = np.array([b["pos"] for b in basis_list], float)
    # The basis is listed atom by atom (s, p z/x/y, d x5). Group by that order, not by
    # Wannier centre: on some V sites the p/d centres sit up to ~0.5 Å off the atom
    # (hybridised WFs), so L from the labels is only approximate there.
    site = -1
    shells = {}
    for m in range(len(labels) // 2):                          # spatial orbital m -> index 2m (+s)
        lab = labels[2 * m]
        l = int(lab[1])
        if l == 0:
            site += 1
            continue
        key = (site, lab[:2])
        shells.setdefault(key, []).append((m, lab[2:]))
    rows, vals = [], []
    for (_, nl), members in shells.items():
        l = int(nl[1])
        if sorted(p for _, p in members) != sorted(_REAL_HARM[l]):
            raise SystemExit(f"incomplete shell {nl}: {[p for _, p in members]}")
        L = angular_momentum(l, [p for _, p in members])
        HL = sum(bvec_ev[a] * L[a] for a in range(3))
        for a, (ma, _) in enumerate(members):
            for b, (mb, _) in enumerate(members):
                if abs(HL[a, b]) < 1e-15:
                    continue
                for s in range(2):
                    rows.append([2 * ma + s, 2 * mb + s, 0.0, 0.0, 0.0])
                    vals.append(HL[a, b])
    return rows, vals, len(shells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--b-tesla", type=float, help="physical field: mu_B B.(L + 2S), g_s = 2")
    g.add_argument("--delta-mev", type=float, help="legacy: spin-only splitting Delta, meV")
    ap.add_argument("--no-orbital", action="store_true", help="with --b-tesla: drop mu_B B.L")
    ap.add_argument("--dir", default="normal")
    ap.add_argument("--physics", default=None,
                    help="arpes_physics JSON of the cut; defines normal / +slit / +defl")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.tb, allow_pickle=True)
    n = check_basis(d["basis_list"])
    nhat = field_direction(args.dir, args.physics, d["b_matrix"]) * (-1 if args.flip else 1)
    if args.b_tesla is not None:
        half = MU_B * args.b_tesla                               # 2 mu_B B S = mu_B B sigma
        orbital = not args.no_orbital
    else:
        half = 0.5 * args.delta_mev * 1e-3                       # eV
        orbital = False
    # (Delta/2) n.sigma in the (up, down) basis
    blk = half * np.array([[nhat[2], nhat[0] - 1j * nhat[1]],
                           [nhat[0] + 1j * nhat[1], -nhat[2]]])

    rows, vals = [], []
    for m in range(n // 2):
        for s in range(2):
            for t in range(2):
                rows.append([2 * m + s, 2 * m + t, 0.0, 0.0, 0.0])
                vals.append(blk[s, t])
    # H builders read the upper triangle; entries for (j, i) are the conjugates and are
    # already included above because the block is Hermitian (blk[t,s] = conj blk[s,t]).
    n_shells = 0
    if orbital:
        r2, v2, n_shells = orbital_zeeman_rows(d["basis_list"], MU_B * args.b_tesla * nhat)
        rows += r2
        vals += v2
    rows = np.array(rows, float)
    vals = np.array(vals, complex)

    out = {k: d[k] for k in d.files}
    out["indices"] = np.vstack([d["indices"], rows])
    out["values"] = np.concatenate([d["values"], vals])
    out["zeeman_spin_half_splitting_ev"] = float(half)
    out["zeeman_b_tesla"] = float(args.b_tesla) if args.b_tesla is not None else np.nan
    out["zeeman_orbital"] = bool(orbital)
    out["zeeman_nhat_cart"] = nhat
    out["zeeman_note"] = ("hand-imposed on-site Zeeman: mu_B B.sigma"
                          + (" + mu_B B.L (p,d shells, L from real-harmonic labels; Wannier "
                             "functions are not pure atomic orbitals)" if orbital else " only")
                          + "; not self-consistent (no exchange enhancement)")
    np.savez_compressed(args.out, **out)
    what = f"B={args.b_tesla} T ({'spin+orbital' if orbital else 'spin only'})" if args.b_tesla is not None \
        else f"Delta={args.delta_mev} meV (spin only)"
    print(f"{what}  n={np.round(nhat, 4)} ({args.dir}{', flipped' if args.flip else ''})"
          f"  +{len(vals)} rows, {n_shells} p/d shells  -> {args.out}")


if __name__ == "__main__":
    main()
