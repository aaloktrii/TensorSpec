"""Whole-beamtime processing of NSLS-II ESM PEEM data (no Qt, no plotting).

Folder layout (one BNL ESM beamtime, June 2026)::

    <date>_<sample>/Sample_<n>/<RT|110K>/XPEEM_<FOV>um_<apertures>_<E>eV_<edge>_<CL|CR|LH|LV>_<az>deg[_<rep>]/*.dat
    <date>_<sample>/Sample_<n>/<T>/XAS_<FOV>um_<apertures>_<E0>eV_<E1>eV_<step>eV-step_<edge>_<pol>/*.dat
    <date>_<sample>/Sample_<n>/<T>/*LEEM*.dat                    single LEEM images

Conditions come from the folder names (sample, temperature label, FOV, photon
energy, polarization, sample azimuth, repeat) and from the UView metadata of
every frame (sample temperature, COL/DPS/MCH/PCH pressures, timestamps).

* unit     CL+CR (XMCD -> pcd) or LH+LV (XLD -> pld) at one condition and repeat
* cluster  all repeats of one condition (e.g. "0deg" and "0deg_2"), merged
* LD split pld at low T minus pld at RT for the same sample / FOV / azimuth / energy
           (pmnld = RT reference, pmld = low T - RT), when both exist
* XAS      ROI spectrum; energies from the folder name (E0 + step * index)
* LEEM     single images exported as TIFF + preview

Sign convention: pcd = (CL - CR) / (CL + CR). CL is mapped to the internal "CP"
channel and CR to "CM"; flip ``convention`` if your helicity naming differs.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tensorspec.core.io import uview_loaders as uview
from tensorspec.core.peem.ue49 import batch as ub
from tensorspec.core.peem import dichroism

OUT_DIRNAME = "tensorspec_xpeem"
_XPEEM = re.compile(r"^XPEEM_(?P<fov>[\d.]+)um_(?P<ap>.+?)_(?P<e>[\d.]+)eV_(?P<edge>\w+?)_(?P<pol>CL|CR|LH|LV)"
                    r"_(?P<az>[+-]?\d+)deg(?:_(?P<rep>\d+))?$")
_XAS = re.compile(r"^XAS_(?P<fov>[\d.]+)um_(?P<ap>.+?)_(?P<e0>[\d.]+)eV_(?P<e1>[\d.]+)eV_(?P<step>[\d.]+)eV-step"
                  r"_(?P<edge>\w+?)_(?P<pol>CL|CR|LH|LV)$")
PAIRS = {"XMCD": ("CL", "CR"), "XLD": ("LH", "LV")}


@dataclass
class Acq:
    kind: str  # "XPEEM" | "XAS"
    path: str
    date: str
    sample: str
    t_label: str
    fov_um: float
    edge: str
    pol: str
    energy_eV: float
    azimuth_deg: int = 0
    repeat: int = 1
    e0: float | None = None
    e1: float | None = None
    step: float | None = None
    n_files: int = 0

    @property
    def condition(self) -> tuple:
        return (self.date, self.sample, self.t_label, self.fov_um, self.energy_eV, self.azimuth_deg)


def catalog(root: str | Path) -> dict[str, list]:
    root = Path(root)
    acqs: list[Acq] = []
    leem: list[Path] = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        rel = d.relative_to(root).parts
        if len(rel) < 4 or OUT_DIRNAME in rel:
            continue
        date, sample, t_label, name = rel[0], rel[1], rel[2], rel[-1]
        files = sorted(d.glob("*.dat"))
        if not files:
            continue
        m = _XPEEM.match(name)
        if m:
            acqs.append(Acq("XPEEM", str(d), date, sample, t_label, float(m["fov"]), m["edge"], m["pol"],
                            float(m["e"]), int(m["az"]), int(m["rep"] or 1), n_files=len(files)))
            continue
        m = _XAS.match(name)
        if m:
            acqs.append(Acq("XAS", str(d), date, sample, t_label, float(m["fov"]), m["edge"], m["pol"],
                            float(m["e0"]), e0=float(m["e0"]), e1=float(m["e1"]), step=float(m["step"]),
                            n_files=len(files)))
    for f in sorted(root.rglob("*.dat")):
        rel = f.relative_to(root).parts
        if len(rel) == 4 and OUT_DIRNAME not in rel:  # <date>/<Sample>/<T>/<file>.dat
            leem.append(f)
    return {"acqs": acqs, "leem": leem}


@dataclass
class Group:
    gid: str
    kind: str  # "XMCD" | "XLD"
    condition: tuple
    units: list[tuple[Acq, Acq]] = field(default_factory=list)  # (CL|LH, CR|LV) per repeat
    unpaired: list[Acq] = field(default_factory=list)


def plan(cat: dict[str, list]) -> list[Group]:
    groups: dict[tuple, Group] = {}
    by_key: dict[tuple, dict[str, Acq]] = {}
    for a in cat["acqs"]:
        if a.kind != "XPEEM":
            continue
        kind = "XMCD" if a.pol in PAIRS["XMCD"] else "XLD"
        by_key.setdefault((kind, a.condition, a.repeat), {})[a.pol] = a
    for (kind, cond, rep), pols in sorted(by_key.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        date, sample, t, fov, e, az = cond
        gid = f"{sample}_{t}_{fov:g}um_{e:g}eV_{az:+d}deg_{kind}".replace("+0deg", "0deg")
        g = groups.setdefault((kind, cond), Group(gid, kind, cond))
        a_pol, b_pol = PAIRS[kind]
        if a_pol in pols and b_pol in pols:
            g.units.append((pols[a_pol], pols[b_pol]))
        else:
            g.unpaired.extend(pols.values())
    return list(groups.values())


def load_acq(a: Acq) -> dict[str, Any]:
    files = sorted(Path(a.path).glob("*.dat"))
    frames, metas, bad = [], [], []
    for f in files:
        try:
            img, m = uview.read_uview(f)
        except Exception as exc:  # corrupt / truncated frame
            bad.append(f"{f.name}: {type(exc).__name__}")
            continue
        frames.append(img)
        metas.append(m)
    if not frames:
        raise ValueError(f"no readable frames in {a.path}")
    if len({f.shape for f in frames}) != 1:
        raise ValueError(f"mixed image sizes in {a.path}")
    return {"stack": np.stack(frames), "metas": metas, "bad": bad}


def summarize_meta(metas: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("sample_temp_K", "pressure_COL_torr", "pressure_DPS_torr", "pressure_MCH_torr", "pressure_PCH_torr",
                "start_voltage_V", "v_ccs"):
        v = np.array([m.get(key, np.nan) for m in metas], dtype=float)
        if np.isfinite(v).any():
            out[f"{key}_mean"] = float(np.nanmean(v))
            out[f"{key}_min"] = float(np.nanmin(v))
            out[f"{key}_max"] = float(np.nanmax(v))
    ts = [m.get("timestamp") for m in metas if m.get("timestamp")]
    if ts:
        out["time_start"], out["time_end"] = min(ts), max(ts)
    fovs = {m.get("fov_um") for m in metas if m.get("fov_um")}
    out["fov_um_meta"] = sorted(fovs)
    out["um_per_px"] = uview.um_per_px(metas[0])
    return out


@dataclass
class Settings:
    convention: str = "cp_minus_cm"  # (CL - CR)/(CL + CR)
    fov_shrink: float = 0.9
    domain_threshold: float = 0.15
    domain_smooth: float = 1.5
    domain_min_px: int = 12


def _ue_settings(st: Settings) -> ub.Settings:
    return ub.Settings(dead_map=None, norm_image=None, convention=st.convention, fov_shrink=st.fov_shrink,
                       domain_threshold=st.domain_threshold, domain_smooth=st.domain_smooth,
                       domain_min_px=st.domain_min_px)


def _out_root(root: Path, date: str) -> Path:
    return root / date / OUT_DIRNAME


def process_group(g: Group, root: Path, st: Settings, preview=None) -> dict[str, Any]:
    date, sample, t, fov, e, az = g.condition
    out = _out_root(root, date) / "clusters" / g.gid
    ust = _ue_settings(st)
    row: dict[str, Any] = {"type": "cluster", "id": g.gid, "kind": g.kind, "date": date, "sample": sample,
                           "T_label": t, "fov_um": fov, "energy_eV": e, "azimuth_deg": az,
                           "n_units": len(g.units), "unpaired": ";".join(Path(a.path).name for a in g.unpaired),
                           "path": str(out.relative_to(root))}
    if not g.units:
        row["status"] = "no complete CL/CR or LH/LV pair"
        return row
    fa, fb, unit_maps, unit_rows, meta_all = [], [], [], [], []
    upp = None
    for a_acq, b_acq in g.units:
        la, lb = load_acq(a_acq), load_acq(b_acq)
        meta_all += la["metas"] + lb["metas"]
        upp = upp or uview.um_per_px(la["metas"][0])
        res = ub._dichroism_map(la["stack"], lb["stack"], g.kind if g.kind == "XMCD" else "XLD_pol", ust)
        dom, offset = _domain_zeroed(res, ust, upp)
        uid = f"rep{a_acq.repeat}"
        tags = PAIRS[g.kind]
        attrs = {"unit": uid, "kind": g.kind, "channels": tags, "n_a": len(la["stack"]), "n_b": len(lb["stack"]),
                 "convention": "(CL-CR)/(CL+CR)" if st.convention == "cp_minus_cm" else "(CR-CL)/(CR+CL)",
                 "formula": "(LH-LV)/(LH+LV)" if g.kind == "XLD" else None,
                 "refine_shift": list(res["refine_shift"]), "sample": sample, "T_label": t,
                 "energy_eV": e, "azimuth_deg": az, "folders": [Path(a_acq.path).name, Path(b_acq.path).name],
                 "bad_frames": la["bad"] + lb["bad"],
                 "domain_map_offset_removed": offset,
                 "domain_map_note": ("pcd median over the ROI subtracted before classification: the CL/CR flux "
                                     "differ (no I0 in UView files; RT maps show a +0.04 offset), which also "
                                     "removes any net magnetization of the field") if g.kind == "XMCD" else "",
                 **{k: v for k, v in summarize_meta(la["metas"] + lb["metas"]).items() if not isinstance(v, list)}}
        uo = out / "units" / uid
        ub._write_maps(uo, res, dom, attrs, upp)
        ub._json(uo / "summary.json", {**attrs, "map": ub._summary(res["map"]),
                                       "domain": dom["stats"] if dom else None})
        if preview:
            preview(uo / "preview.png", res, dom,
                    f"{sample} {t} | {fov:g} um {e:g} eV az {az:+d} | {g.kind} {uid} | T={attrs.get('sample_temp_K_mean', float('nan')):.0f} K")
        unit_maps.append(res["map"])
        unit_rows.append({"unit": uid, **{f"map_{k}": v for k, v in ub._summary(res["map"]).items()},
                          "f_up": dom["stats"]["f_up"] if dom else "", "f_down": dom["stats"]["f_down"] if dom else ""})
        fa.append(la["stack"])
        fb.append(lb["stack"])
        del la, lb
    merged = None
    if len(g.units) > 1:
        merged = ub._dichroism_map(np.concatenate(fa), np.concatenate(fb), g.kind if g.kind == "XMCD" else "XLD_pol", ust)
        mdom, _ = _domain_zeroed(merged, ust, upp)
        ub._write_maps(out / "merged", merged, mdom, {"cluster": g.gid, "kind": g.kind, "channels": PAIRS[g.kind],
                                                      "units": [r["unit"] for r in unit_rows]}, upp)
        if preview:
            preview(out / "merged" / "preview.png", merged, mdom, f"{g.gid} merged ({len(g.units)} repeats)")
        row.update({f"merged_{k}": v for k, v in ub._summary(merged["map"]).items()})
    pair_r = [ub._corr(x, y) for i, x in enumerate(unit_maps) for y in unit_maps[i + 1:]]
    meta = summarize_meta(meta_all)
    ub._json(out / "cluster.json", {"cluster": g.gid, "kind": g.kind, "condition": dict(zip(
        ("date", "sample", "T_label", "fov_um", "energy_eV", "azimuth_deg"), g.condition)),
        "units": unit_rows, "repeat_pairwise_corr": pair_r, "metadata": meta,
        "unpaired": [Path(a.path).name for a in g.unpaired]})
    row.update({"status": "ok", "repeat_mean_corr": float(np.nanmean(pair_r)) if pair_r else None,
                "T_K_measured": meta.get("sample_temp_K_mean"), "um_per_px": upp,
                **{k: v for k, v in meta.items() if k.startswith("pressure") and k.endswith("_mean")},
                "time_start": meta.get("time_start"),
                **{f"unit_{k}": v for k, v in (unit_rows[0].items() if len(unit_rows) == 1 else [])}})
    return row


def _domain_zeroed(res: dict[str, Any], ust, upp):
    if res["kind"] != "pcd":
        return None, None
    v = res["map"][res["mask"] & np.isfinite(res["map"])]
    offset = float(np.median(v)) if v.size else 0.0
    shifted = {**res, "map": res["map"] - offset}
    return ub._domain(shifted, ust, upp), offset


def best_map(root: Path, row: dict[str, Any], var: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(map, mean_a) of a processed cluster: merged if present, else the first unit."""
    import xarray as xr

    cdir = root / row["path"]
    nc = cdir / "merged" / "maps.nc"
    if not nc.exists():
        units = sorted((cdir / "units").glob("*/maps.nc"))
        if not units:
            return None
        nc = units[0]
    with xr.open_dataset(nc) as ds:
        mean_var = [v for v in ds.data_vars if v.startswith("mean_")][0]
        return np.asarray(ds[var].values), np.asarray(ds[mean_var].values)


def ld_temperature_splits(rows: list[dict[str, Any]], root: Path, st: Settings, preview=None) -> list[dict[str, Any]]:
    """Low-T minus RT for every cluster with an RT partner (same sample/FOV/E/azimuth/kind).

    XLD: pmnld = pld(RT) (natural LD), pmld = pld(low T) - pld(RT).
    XMCD: pcd_corrected = pcd(low T) - pcd(RT): RT has no magnetic order, so its pcd is
    the helicity-dependent illumination / flux pattern (the BNL files carry no I0).
    """
    import tifffile
    import xarray as xr

    out_rows = []
    for kind in ("XLD", "XMCD"):
        out_rows += _splits_for_kind(kind, rows, root, preview, tifffile, xr)
    return out_rows


def _splits_for_kind(kind, rows, root, preview, tifffile, xr) -> list[dict[str, Any]]:
    out_rows = []
    var = "pld" if kind == "XLD" else "pcd"
    names = (("pld_lowT", "pmnld", "pmld") if kind == "XLD" else ("pcd_lowT", "pcd_RT", "pcd_corrected"))
    xld = [r for r in rows if r.get("kind") == kind and r.get("status") == "ok"]
    for low in (r for r in xld if r["T_label"].upper() != "RT"):
        ref = next((r for r in xld if r["T_label"].upper() == "RT" and r["sample"] == low["sample"]
                    and r["fov_um"] == low["fov_um"] and r["energy_eV"] == low["energy_eV"]
                    and r["azimuth_deg"] == low["azimuth_deg"]), None)
        if ref is None:
            continue
        a, a_int = best_map(root, low, var)
        b, b_int = best_map(root, ref, var)
        sid = f"{low['id']}_vs_RT"
        out = _out_root(root, low["date"]) / "rt_splits" / sid
        row = {"type": "ld_split", "id": sid, "date": low["date"], "sample": low["sample"],
               "low": low["id"], "reference": ref["id"], "path": str(out.relative_to(root))}
        try:
            res = dichroism.split_by_reference(a, b, align_image=a_int, reference_align_image=b_int)
        except ValueError as exc:
            row["status"] = f"error: {exc}"
            out_rows.append(row)
            continue
        out.mkdir(parents=True, exist_ok=True)
        maps = dict(zip(names, (a, res["pmnld"], res["pmld"])))
        xr.Dataset({k: (("y", "x"), v.astype(np.float32)) for k, v in maps.items()},
                   attrs={names[2]: f"{var}(low T) - {var}(RT), RT registered on intensity",
                          names[1]: f"{var}(RT) registered", "low": low["id"], "reference": ref["id"],
                          "reference_shift_dy": res["reference_shift"][0], "reference_shift_dx": res["reference_shift"][1],
                          "noise_ratio": res["noise_ratio"], "reference_correlation": res["correlation"],
                          "reliable": int(res["reliable"])}).to_netcdf(
            out / "maps.nc", encoding={k: {"zlib": True} for k in maps})
        for k, v in maps.items():
            tifffile.imwrite(out / f"{k}.tif", v.astype(np.float32), compression="zlib")
        if preview:
            preview.ld(out / "preview.png", maps, f"{sid}: {names[2]} = low T - RT; noise ratio "
                                                  f"{res['noise_ratio']:.2f}, r(lowT,RT)={res['correlation']:.2f}")
        row.update({"status": "ok", "reference_shift": list(res["reference_shift"]), "noise_ratio": res["noise_ratio"],
                    "reference_correlation": res["correlation"], "reliable": res["reliable"],
                    **{f"{names[2]}_{k}": v for k, v in ub._summary(res["pmld"]).items()}})
        ub._json(out / "summary.json", row)
        out_rows.append(row)
    return out_rows


def process_xas(a: Acq, root: Path, preview=None) -> dict[str, Any]:
    sid = f"{a.sample}_{a.t_label}_{Path(a.path).name}"
    out = _out_root(root, a.date) / "spectra" / sid
    out.mkdir(parents=True, exist_ok=True)
    la = load_acq(a)
    stack = la["stack"]
    n = len(stack)
    energy = a.e0 + a.step * np.arange(n)
    n_expected = int(round((a.e1 - a.e0) / a.step)) + 1
    mask = ub._fov_mask(stack.mean(axis=0), 0.7)
    mu = stack[:, mask].mean(axis=1)
    edge = "Fe" if "Fe" in a.edge else "Cr"
    pre = ub.EDGE_WINDOWS[edge][0]
    norm = ub.normalize_xas(energy, mu, pre)
    temps = np.array([m.get("sample_temp_K", np.nan) for m in la["metas"]])
    ub._write_csv(out / "spectrum.csv", {"energy_eV_assumed": energy, "mu": mu, "mu_norm": norm, "sample_temp_K": temps})
    note = ("" if n == n_expected else
            f"{n} frames but {a.e0:g}-{a.e1:g} eV in {a.step:g} eV steps is {n_expected} points; energies assumed "
            f"E0 + step*index (last = {energy[-1]:.2f} eV). Photon energy is not stored in the UView metadata.")
    summary = {"job": sid, "folder": Path(a.path).name, "pol": a.pol, "n_frames": n, "n_expected": n_expected,
               "energy_note": note, "max_eV": float(energy[np.argmax(norm)]), "min_eV": float(energy[np.argmin(norm)]),
               "roi": "central disk, 70 % of FOV (mostly substrate on these images)",
               "shape_note": ("intensity dips at the edge (inverted XAS): the ROI is dominated by non-Cr area, so this "
                              "likely tracks incident-flux dips; use a flake ROI for Cr XAS")
               if norm[np.argmin(norm)] < -0.5 else "", **summarize_meta(la["metas"]),
               "bad_frames": la["bad"]}
    ub._json(out / "summary.json", summary)
    if preview:
        preview.spectra(out / "spectra.png", {f"{a.pol} {a.t_label}": (energy, norm)},
                        f"{a.sample} {a.t_label} XAS ({n} frames){' - energy axis assumed' if note else ''}")
    return {"type": "spectra:XAS", "id": sid, "date": a.date, "sample": a.sample, "T_label": a.t_label,
            "status": "ok", "max_eV": summary["max_eV"], "min_eV": summary["min_eV"], "energy_note": note,
            "shape_note": summary["shape_note"],
            "T_K_measured": summary.get("sample_temp_K_mean"), "path": str(out.relative_to(root))}


def export_leem(files: list[Path], root: Path, preview=None) -> list[dict[str, Any]]:
    import tifffile

    rows = []
    for f in files:
        rel = f.relative_to(root).parts
        out = _out_root(root, rel[0]) / "leem" / f"{rel[1]}_{rel[2]}"
        out.mkdir(parents=True, exist_ok=True)
        try:
            img, m = uview.read_uview(f)
        except Exception as exc:
            rows.append({"type": "leem", "id": f.name, "date": rel[0], "status": f"error: {exc}"})
            continue
        tifffile.imwrite(out / f"{f.stem}.tif", img.astype(np.float32), compression="zlib")
        ub._json(out / f"{f.stem}.json", m)
        if preview:
            preview.image(out / f"{f.stem}.png", img, f"{rel[1]} {rel[2]} {f.stem} | FOV {m.get('fov_um')} um | "
                                                          f"SV {m.get('start_voltage_V', float('nan')):.2f} V")
        rows.append({"type": "leem", "id": f.stem, "date": rel[0], "sample": rel[1], "T_label": rel[2], "status": "ok",
                     "fov_um": m.get("fov_um"), "start_voltage_V": m.get("start_voltage_V"),
                     "T_K_measured": m.get("sample_temp_K"), "path": str((out / f"{f.stem}.tif").relative_to(root))})
    return rows


def run(root: str | Path, st: Settings | None = None, preview=None, log=print, skip_existing: bool = True) -> dict[str, Any]:
    root = Path(root)
    st = st or Settings()
    cat = catalog(root)
    groups = plan(cat)
    ub.write_csv_rows(root / "tensorspec_xpeem_catalog.csv", [asdict(a) for a in cat["acqs"]])
    rows: list[dict[str, Any]] = []
    for i, g in enumerate(groups, 1):
        target = _out_root(root, g.condition[0]) / "clusters" / g.gid
        saved = target / "index_row.json"
        if skip_existing and saved.exists():
            rows.append(json.loads(saved.read_text()))
            log(f"[{i}/{len(groups)}] {g.gid}: existing")
            continue
        try:
            row = process_group(g, root, st, preview)
        except Exception as exc:
            row = {"type": "cluster", "id": g.gid, "date": g.condition[0], "status": f"error: {type(exc).__name__}: {exc}"}
        if row.get("status") == "ok":
            ub._json(saved, row)
        rows.append(row)
        log(f"[{i}/{len(groups)}] {g.gid}: {row.get('status')}")
    rows += ld_temperature_splits(rows, root, st, preview)
    for a in (a for a in cat["acqs"] if a.kind == "XAS"):
        try:
            rows.append(process_xas(a, root, preview))
        except Exception as exc:
            rows.append({"type": "spectra:XAS", "id": Path(a.path).name, "date": a.date, "status": f"error: {exc}"})
    rows += export_leem(cat["leem"], root, preview)
    ub.write_csv_rows(root / "tensorspec_xpeem_index.csv", rows)
    for date in sorted({r["date"] for r in rows if r.get("date")}):
        d = _out_root(root, date)
        if d.is_dir():
            ub.write_csv_rows(d / "index.csv", [r for r in rows if r.get("date") == date])
    log(f"done: {sum(1 for r in rows if str(r.get('status')).startswith('ok'))}/{len(rows)} ok")
    return {"rows": rows, "groups": groups, "catalog": cat}
