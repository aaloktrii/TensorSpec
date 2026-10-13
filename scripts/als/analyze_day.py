"""Analyse one ALS PEEM-3 (BL 11.0.1) day folder, e.g. ``Jun2021/210601``.

    python scripts/als/analyze_day.py "<data>/ALS/<beamtime>/<day>" [--convention cp_minus_cm]

Each acquisition (a folder of ``#NNNN.P3B`` frames, or a single loose frame) is
classified from its frame metadata:

* XMCD   one photon energy, both EPU polarizations (+0.9 / -0.9)
* XAS    an energy sweep (>= 10 energies) at one polarization
* image  anything else (exported only)

XMCD: frames are I0-normalized, registered (joint registration + edge refinement,
the UE49 pipeline) and pcd = (I(+0.9) - I(-0.9)) / (I(+0.9) + I(-0.9)) is formed;
acquisitions at the same stage position, FOV and energy are merged into a cluster.
XAS: ROI-mean / I0 per frame; an XMCD spectrum is formed from a + and a - sweep
when both exist (flagged when they were taken at different positions / FOVs).
Outputs go to ``<day>/tensorspec_xpeem/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tensorspec.core.io import p3b_loaders as p3b  # noqa: E402
from tensorspec.core.io.p3b_loaders import i0_reliable, um_per_px  # noqa: E402
from tensorspec.core.peem.esm.batch import _domain_zeroed  # noqa: E402
from tensorspec.core.peem.ue49 import batch as ub  # noqa: E402

FE = {"pre": (695.0, 704.0), "L3": (704.0, 715.0), "L2": (716.0, 728.0), "post": (735.0, 745.0)}


def acquisitions(day: Path) -> dict[str, list[Path]]:
    acqs: dict[str, list[Path]] = {}
    for d in sorted(p for p in day.iterdir() if p.is_dir() and p.name != "tensorspec_xpeem"):
        fs = p3b.frame_files(d)
        if fs:
            acqs[d.name] = fs
    for f in sorted(day.glob("*#*.P3B")):  # loose single frames
        acqs.setdefault(f.name.split("#")[0], []).append(f)
    return acqs


def dat_summary(day: Path, name: str, frame: Path | None = None) -> dict:
    f = day / f"{name}.dat"
    if not f.exists() and frame is not None:
        # 2021-06-26: folders OF_AL_003 / 012 hold frames and .dat named IF_AL_003 / 012
        f = day / f"{frame.name.split('#')[0]}.dat"
    if not f.exists():
        return {}
    cols = p3b.read_dat(f)
    out = {}
    for key in ("Cryostat Temperature", "Sample Temperature Goal", "Beam Current", "Pressure IG111", "Pressure IG112"):
        v = cols.get(key)
        if v is not None and np.isfinite(v).any():
            out[key.lower().replace(" ", "_")] = float(np.nanmean(v))
    return out


def classify(metas: list[dict]) -> str:
    energies = {round(m.get("BEAMLINE_ENERGY", np.nan), 1) for m in metas}
    pols = {np.sign(m.get("EPU_POLARIZATION", 0)) for m in metas}
    if len(energies) == 1 and pols >= {1.0, -1.0}:
        return "XMCD"
    if len(energies) >= 10:
        return "XAS"
    return "image"


def load(files: list[Path], i0_norm: bool = True):
    frames, metas = [], []
    for f in files:
        img, m = p3b.read_p3b(f)
        frames.append(img)
        metas.append(m)
    stack = np.stack(frames).astype(np.float64)
    i0 = np.array([m.get("I0", np.nan) for m in metas], dtype=float)
    ok, why = i0_reliable(metas)
    used = bool(i0_norm and ok)
    if used:
        stack = stack / i0[:, None, None] * float(np.sqrt(np.mean(i0**2)))
    return stack, metas, {"i0_normalized": used, "i0_check": why}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("day")
    ap.add_argument("--use-i0", action="store_true",
                    help="normalize XMCD frames by I0. Off by default: on 2021-06-26 I0 passed the variation "
                         "check but did not track the image intensity (within-helicity correlation -0.7..+0.6) "
                         "and flipped the sign of the measured asymmetry")
    ap.add_argument("--convention", default="cp_minus_cm", choices=["cp_minus_cm", "cm_minus_cp"],
                    help="cp_minus_cm: pcd = (I(+) - I(-))/(I(+) + I(-))")
    a = ap.parse_args()
    day = Path(a.day)
    out = day / "tensorspec_xpeem"
    out.mkdir(exist_ok=True)
    from tensorspec.plotting.ue49_preview import Preview

    preview = Preview()
    ust = ub.Settings(dead_map=None, norm_image=None, convention=a.convention)
    acqs = acquisitions(day)
    rows, xmcd_units, xas = [], [], []
    for name, files in acqs.items():
        metas = [p3b.read_p3b(f, with_image=False)[1] for f in files]
        kind = classify(metas)
        m0 = metas[0]
        info = {"acq": name, "kind": kind, "n_frames": len(files), "fov_um": m0.get("FOV"),
                "energy_eV": round(float(np.median([m["BEAMLINE_ENERGY"] for m in metas])), 2),
                "energy_min": round(min(m["BEAMLINE_ENERGY"] for m in metas), 2),
                "energy_max": round(max(m["BEAMLINE_ENERGY"] for m in metas), 2),
                "pol": sorted({m.get("EPU_POLARIZATION") for m in metas}), "x": round(m0.get("X", np.nan), 1),
                "y": round(m0.get("Y", np.nan), 1), "magnet_current": m0.get("MAGNET_CURRENT"),
                "exposure_ms": m0.get("EXPOSURETIME"), "time_start": metas[0].get("timestamp"),
                "time_end": metas[-1].get("timestamp"), **dat_summary(day, name, files[0])}
        print(f"{name}: {kind} ({len(files)} frames)", flush=True)
        if kind == "XMCD":
            stack, metas, i0info = load(files, i0_norm=a.use_i0)
            if not a.use_i0:
                i0info["i0_check"] = "not used for XMCD (default); " + i0info["i0_check"]
            info.update(i0info)
            pol = np.array([np.sign(m["EPU_POLARIZATION"]) for m in metas])
            res = ub._dichroism_map(stack[pol > 0], stack[pol < 0], "XMCD", ust)
            upp = um_per_px(m0)
            dom, offset = _domain_zeroed(res, ust, upp)
            uo = out / "xmcd" / name
            attrs = {**info, "channels": ("plus", "minus"), "convention": "(I+ - I-)/(I+ + I-)"
                     if a.convention == "cp_minus_cm" else "(I- - I+)/(I- + I+)", "domain_map_offset_removed": offset,
                     "refine_shift": list(res["refine_shift"]), "pol": json.dumps(info["pol"])}
            ub._write_maps(uo, res, dom, attrs, upp)
            stats = {**info, "map": ub._summary(res["map"]), "domain": dom["stats"] if dom else None,
                     "frame_shifts_px": res["frame_shifts"], "domain_offset_removed": offset}
            ub._json(uo / "summary.json", stats)
            preview(uo / "preview.png", res, dom, f"{name} | {info['energy_eV']} eV | "
                    f"T_cryo {info.get('cryostat_temperature', float('nan')):.0f} K | FOV {info['fov_um']} um")
            xmcd_units.append((info, res, stack, pol))
            info.update({f"pcd_{k}": v for k, v in ub._summary(res["map"]).items()})
        elif kind == "XAS":
            xas.append((info, files))
        else:
            for f in files:
                img, m = p3b.read_p3b(f)
                io = out / "images"
                io.mkdir(parents=True, exist_ok=True)
                import tifffile
                tifffile.imwrite(io / f"{f.stem.replace('#', '_')}.tif", img, compression="zlib")
                preview.image(io / f"{f.stem.replace('#', '_')}.png", img,
                              f"{f.stem} | {m.get('BEAMLINE_ENERGY', 0):.2f} eV pol {m.get('EPU_POLARIZATION')} "
                              f"| FOV {m.get('FOV')} um")
        rows.append(info)

    # clusters: XMCD at the same position / FOV / energy
    groups = defaultdict(list)
    for info, res, stack, pol in xmcd_units:
        groups[(round(info["x"]), round(info["y"]), info["fov_um"], info["energy_eV"])].append((info, res, stack, pol))
    for key, members in groups.items():
        cid = "pos_x{:+d}_y{:+d}_fov{}_{}eV".format(key[0], key[1], key[2], key[3])
        maps = [r["map"] for _, r, _, _ in members]
        pair_r = [ub._corr(x, y) for i, x in enumerate(maps) for y in maps[i + 1:]]
        crow = {"acq": cid, "kind": "XMCD cluster", "members": ";".join(m[0]["acq"] for m in members),
                "repeat_corr": float(np.nanmean(pair_r)) if pair_r else None}
        if len(members) > 1:
            plus = np.concatenate([s[p > 0] for _, _, s, p in members])  # noqa: E501
            minus = np.concatenate([s[p < 0] for _, _, s, p in members])
            res = ub._dichroism_map(plus, minus, "XMCD", ust)
            upp = um_per_px({"FOV": key[2], "DIMX": plus.shape[2], "DIMY": plus.shape[1]})
            dom, offset = _domain_zeroed(res, ust, upp)
            co = out / "xmcd_clusters" / cid
            ub._write_maps(co, res, dom, {"members": crow["members"], "convention": a.convention,
                                          "domain_map_offset_removed": offset}, upp)
            preview(co / "preview.png", res, dom, f"{cid} merged: {crow['members']}")
            crow.update({f"pcd_{k}": v for k, v in ub._summary(res["map"]).items()})
            ub._json(co / "cluster.json", {**crow, "pairwise_corr": pair_r, "domain": dom["stats"] if dom else None})
        rows.append(crow)

    # XAS: ROI mean / I0 per frame
    spectra = {}
    for info, files in xas:
        mask = None
        e, mu, metas = [], [], []
        for f in files:
            img, m = p3b.read_p3b(f)
            if mask is None:
                mask = ub._fov_mask(img, 0.7)
            e.append(m["BEAMLINE_ENERGY"])
            mu.append(float(img[mask].mean()))
            metas.append(m)
        ok, why = i0_reliable(metas)
        info.update({"i0_normalized": ok, "i0_check": why})
        if ok:
            mu = [v / m["I0"] for v, m in zip(mu, metas)]
        e, mu = np.array(e), np.array(mu)
        o = np.argsort(e)
        e, mu = e[o], mu[o]
        norm = ub.normalize_xas(e, mu, FE["pre"])
        so = out / "xas" / info["acq"]
        so.mkdir(parents=True, exist_ok=True)
        ub._write_csv(so / "spectrum.csv", {"energy_eV": e, "mu": mu, "mu_norm": norm})
        l3 = (e >= FE["L3"][0]) & (e <= FE["L3"][1])
        info["L3_peak_eV"] = float(e[l3][np.argmax(norm[l3])]) if l3.any() else None
        info["edge_jump_raw"] = float(mu[e >= e.max() - 0.05 * np.ptp(e)].mean() - mu[e <= e.min() + 0.05 * np.ptp(e)].mean())
        preview.spectra(so / "spectra.png", {f"pol {info['pol']}": (e, norm)}, f"{info['acq']} (ROI 70 % of FOV{', /I0' if ok else ', no I0'})")
        spectra[info["acq"]] = (info, e, norm)
    plus = [v for v in spectra.values() if all(p > 0 for p in v[0]["pol"])]
    minus = [v for v in spectra.values() if all(p < 0 for p in v[0]["pol"])]
    if plus and minus:
        # widest-range sweep of each sign
        pi, pe, pn = max(plus, key=lambda v: np.ptp(v[1]))
        mi, me, mn = max(minus, key=lambda v: np.ptp(v[1]))
        grid, _ = ub.merge_spectra([(pe, pn), (me, mn)])

        def peak_norm(e, n):
            # pre-edge -> 0, own L3 peak -> 1: the edge jumps are a few % of the
            # background (6-28 counts on ~270), so edge-jump normalization is unstable
            y = np.interp(grid, e, n)
            pre = (grid >= FE["pre"][0]) & (grid <= FE["pre"][1])
            y = y - (y[pre].mean() if pre.any() else y[:5].mean())
            l3 = (grid >= FE["L3"][0]) & (grid <= FE["L3"][1])
            return y / y[l3].max()

        npl, nmi = peak_norm(pe, pn), peak_norm(me, mn)
        xm = npl - nmi
        l3 = (grid >= FE["L3"][0]) & (grid <= FE["L3"][1])
        l2 = (grid >= FE["L2"][0]) & (grid <= FE["L2"][1])
        v3, v2 = xm[l3][np.argmax(np.abs(xm[l3]))], xm[l2][np.argmax(np.abs(xm[l2]))]
        same_place = (round(pi["x"]) == round(mi["x"]) and round(pi["y"]) == round(mi["y"]) and pi["fov_um"] == mi["fov_um"])
        so = out / "xas" / "xmcd_spectrum"
        so.mkdir(parents=True, exist_ok=True)
        ub._write_csv(so / "xmcd_spectrum.csv", {"energy_eV": grid, "mu_plus": npl, "mu_minus": nmi, "xmcd": xm})
        note = {"plus": pi["acq"], "minus": mi["acq"], "L3_xmcd_pct_of_L3_peak": float(100 * v3),
                "L2_xmcd_pct_of_L3_peak": float(100 * v2), "signs_opposite": bool(np.sign(v3) != np.sign(v2)),
                "same_position_and_fov": bool(same_place),
                "caveat": "" if same_place else "the + and - sweeps were taken at different stage positions / FOVs; "
                                                "the spectra average different areas"}
        ub._json(so / "summary.json", note)
        preview.spectra(so / "spectra.png", {"mu+": (grid, npl), "mu-": (grid, nmi), "XMCD": (grid, xm)},
                        f"Fe L2,3 XMCD from sweeps ({'same area' if same_place else 'DIFFERENT areas'})")
        rows.append({"acq": "xmcd_spectrum", "kind": "XAS XMCD", **note})
    ub.write_csv_rows(out / "index.csv", rows)
    print(f"done: {len(rows)} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
