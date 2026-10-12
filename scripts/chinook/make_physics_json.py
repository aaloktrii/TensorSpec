#!/usr/bin/env python3
"""Write arpes_physics.json for the four VTe2 (-201) beam/sample configurations.

These mirror, as closely as Chinook's parameterisation allows, the SPR-KKR
point-wise deflector runs in scratch/sprkkr_e2e/cuts_vte2_th{55,20}_ph{0,90}.

GEOMETRY MAPPING (verified against the real lattice, not assumed)
----------------------------------------------------------------
For hkl = (-2,0,1) with this Wannier cell, get_hkl_surface_frame puts
  Y_surf = the 3.54 A short axis (crystal b, Cartesian y)
  X_surf = the long in-plane direction (2pi/|R| = 0.336 1/A)
At manip_azimuth = 0 / slit_angle = 0 the slit lands on X_surf -- the LONG axis,
which is not what the SPR-KKR runs used. Both pairings below rotate it by 90 deg
onto the short axis, matching SPR-KKR's slit || x, and differ only in where the
beam sits relative to the slit:

  SPR-KKR                      Chinook
  SPEC_PH THETA=55 PHI=0   ->  incidence_angle=55, manip_azimuth=90, slit_angle=0
  SPEC_PH THETA=20 PHI=0   ->  incidence_angle=20, manip_azimuth=90, slit_angle=0
  SPEC_PH THETA=55 PHI=90  ->  incidence_angle=55, manip_azimuth=0,  slit_angle=90
  SPEC_PH THETA=20 PHI=90  ->  incidence_angle=20, manip_azimuth=0,  slit_angle=90

Chinook has no beam-azimuth parameter (the incidence plane is hard-wired to the
lab x-y plane), which is exactly why the sample and slit must be rotated together
instead. It has no deflector parameter either: the deflector axis IS k_bounds['Y'],
so a fixed deflector d is the collapsed range --phi_min d --phi_max d --nphi 1.

PHYSICS CHOICES (all deliberate; see the notes in each entry)
------------------------------------------------------------
inner_potential = 12.0 eV   Chinook needs a V0; SPR-KKR has none (its final state
                            is a real LEED state). 12 eV is the value already used
                            in this repo's GrizzlyME benchmarks.
mfp = 8.7 A                 matches SPR-KKR's IMV_FIN_EV = 2.0 eV:
                            lambda = hbar*v / (2*ImV) with k = 4.568 1/A at 84 eV
                            -> 2 * 3.81 * 4.568 / 4.0 = 8.70 A
se_width = 0.05 eV          matches SPR-KKR's IMV_INI_EV = 0.05 (hole lifetime)
fresnel_enabled = False     kkrspec has no dielectric input, so it does not refract
                            the light; leaving Fresnel on would be an extra effect
                            SPR-KKR cannot have
kz_halfwidth = 0.0          SPR-KKR's kz smearing already comes from IMV_FIN; adding
                            Chinook's kz integration would double-count it (and cost 7x)
res_E, res_k small          SPR-KKR cubes carry no instrumental resolution
temperature = 10 K          NOTE: Chinook applies a Fermi function, SPR-KKR does not.
                            The two therefore differ above ~E_F. Everything below E_F
                            is unaffected. Set --no-fermi to push T up and suppress it.

Usage:
    python scripts/chinook/make_physics_json.py --out-dir physics/
    python scripts/chinook/make_physics_json.py --out-dir physics/ --v0 15 --no-fermi
"""
import argparse
import json
import os

CONFIGS = [
    # tag,          incidence, azimuth, slit, matches
    ("th55_ph0",  55.0, 90.0,  0.0, "SPR-KKR SPEC_PH THETA=55 PHI=0  (beam || slit, as measured)"),
    ("th20_ph0",  20.0, 90.0,  0.0, "SPR-KKR SPEC_PH THETA=20 PHI=0  (beam || slit)"),
    ("th55_ph90", 55.0,  0.0, 90.0, "SPR-KKR SPEC_PH THETA=55 PHI=90 (beam perp. slit)"),
    ("th20_ph90", 20.0,  0.0, 90.0, "SPR-KKR SPEC_PH THETA=20 PHI=90 (beam perp. slit)"),
]


def physics(incidence, azimuth, slit, *, hv, wf, v0, temperature, mfp,
            se_width, res_e, res_k, note):
    return {
        "hv": hv,
        "work_function": wf,
        "inner_potential": v0,
        "temperature": temperature,
        "incidence_angle": incidence,
        "polarization": "Linear Horizontal",   # = p-pol, A in the plane of incidence
        "lin_pol_angle": 45.0,                 # unused for Linear Horizontal
        "matrix_element_mode": "Full Matrix Elements",
        "manip_theta": 0.0,
        "manip_azimuth": azimuth,
        "manip_tilt": 0.0,
        "hkl": [-2, 0, 1],
        "slit_angle": slit,
        "se_width": se_width,
        "res_E": res_e,
        "res_k": res_k,
        "rad_type": "slater",
        "mfp": mfp,
        "kz_halfwidth": 0.0,
        "kz_npoints": 1,
        "fresnel_enabled": False,
        "optical_n": 1.0,
        "optical_k": 0.0,
        "include_photon_momentum": False,
        "_note": note,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="physics")
    ap.add_argument("--hv", type=float, default=84.0)
    ap.add_argument("--workf", type=float, default=4.5)
    ap.add_argument("--v0", type=float, default=12.0)
    ap.add_argument("--mfp", type=float, default=8.7)
    ap.add_argument("--se-width", type=float, default=0.05)
    ap.add_argument("--res-e", type=float, default=0.005)
    ap.add_argument("--res-k", type=float, default=0.005)
    ap.add_argument("--temperature", type=float, default=10.0)
    ap.add_argument("--no-fermi", action="store_true",
                    help="raise T so the Fermi function is flat, matching SPR-KKR")
    args = ap.parse_args()

    T = 30000.0 if args.no_fermi else args.temperature
    os.makedirs(args.out_dir, exist_ok=True)
    for tag, inc, azi, slit, note in CONFIGS:
        p = physics(inc, azi, slit, hv=args.hv, wf=args.workf, v0=args.v0,
                    temperature=T, mfp=args.mfp, se_width=args.se_width,
                    res_e=args.res_e, res_k=args.res_k, note=note)
        path = os.path.join(args.out_dir, f"arpes_physics_{tag}.json")
        with open(path, "w") as fh:
            json.dump(p, fh, indent=2)
        print(f"{path}: incidence={inc:g} azimuth={azi:g} slit={slit:g} V0={args.v0:g} T={T:g}")


if __name__ == "__main__":
    main()
