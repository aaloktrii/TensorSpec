"""Whole-beamtime processing for UE49 data with a transcribed logbook (no Qt, no plotting).

Inputs (next to the day folders)::

    logbook_sessions.csv     session -> sample, day folders, edge, default energies
    logbook_conditions.csv   one row per logbook block: session, X|Mono, first..last scan,
                             T, CCS field, pulse before, PEEM rotation, energy, notes

Scan numbers restart between samples, so a scan is identified by (session, number).

Planning
--------
* **unit**     one acquisition of an X series: a pos/neg/pos (XMCD) or A/B/A (L2 XLD)
               triplet, or an LH/LV pair; detected from the image-name keys.
* **cluster**  all units of one logbook block, i.e. consecutive measurements in the
               same condition (no T / field / pulse change in between). Units are
               processed one by one and also merged (joint registration of all frames).
* **spectra**  Mono_S energy stacks: single XAS, CP/CM pairs (XMCD spectrum, sum rule,
               L3 pcd map), LH/LV pairs (XLD spectrum, pld / pmld / pmnld maps), and
               merges of repeated stacks taken with the same polarization and condition.

X scans with no logbook block are grouped automatically (same T within 3 K, FOV,
energy, consecutive) and marked ``condition_source = auto``.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tensorspec.core.io import ue49_loaders as loaders
from tensorspec.core.peem import dichroism, domains, registration
from tensorspec.core.peem.ue49 import detector

OUT_DIRNAME = "tensorspec_peem"
_NUM = re.compile(r"(\d+)$")
# Cr d3 (Cr3+): 7 holes. Spin sum rule at the Cr L edge needs a jj-mixing
# correction (~2x); values are reported raw.
NH_DEFAULT = {"Cr": 7.0, "Fe": 3.4}
EDGE_WINDOWS = {  # (pre-edge, L3, L2) in eV
    "Cr": ((570.0, 572.5), (572.5, 581.5), (581.5, 592.0)),
    "Fe": ((700.0, 704.0), (704.0, 714.0), (714.0, 730.0)),
}


# ----------------------------------------------------------------------------- data model
@dataclass
class Scan:
    session: str
    day: str
    number: int
    kind: str  # "X" | "Mono"
    path: str
    n: int
    key: str
    pol: str
    energy: float
    e_min: float
    e_max: float
    temperature: float
    fov: str
    n_tif: int

    @property
    def label(self) -> str:
        return f"{'X' if self.kind == 'X' else 'Mono_S'}_{self.number:04d}"


@dataclass
class Condition:
    session: str
    kind: str
    first: int
    last: int
    T_K_log: str = ""
    ccs_mA: str = ""
    pulse_before: str = ""
    rotation_deg: str = ""
    energy_eV: str = ""
    notes: str = ""
    page: str = ""
    transcription_uncertain: str = ""
    source: str = "logbook"

    @property
    def cid(self) -> str:
        return f"{self.session}_{self.kind}_{self.first:04d}-{self.last:04d}"


@dataclass
class Unit:
    uid: str
    kind: str  # "XMCD" | "XLD_L2" | "XLD_pol"
    scans: list[Scan]


@dataclass
class Cluster:
    cid: str
    session: str
    sample: str
    day: str
    condition: Condition
    units: list[Unit] = field(default_factory=list)
    unassigned: list[Scan] = field(default_factory=list)


@dataclass
class SpectraJob:
    sid: str
    session: str
    sample: str
    day: str
    kind: str  # "XAS" | "XMCD_pair" | "XLD_pair" | "merge"
    scans: list[Scan]
    conditions: list[Condition]


# ----------------------------------------------------------------------------- inputs
def read_sessions(path: str | Path) -> dict[str, dict[str, Any]]:
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            row["folders"] = [f.strip() for f in row["folders"].split(";") if f.strip()]
            lo, _, hi = (row.get("scan_range") or "0-100000").partition("-")
            row["scan_lo"], row["scan_hi"] = int(lo), int(hi or lo)
            out[row["session"].strip()] = row
    return out


def read_conditions(path: str | Path) -> list[Condition]:
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get("session"):
                continue
            rows.append(Condition(
                session=row["session"].strip(), kind=row["kind"].strip(),
                first=int(row["first"]), last=int(row["last"]),
                **{k: (row.get(k) or "").strip() for k in (
                    "T_K_log", "ccs_mA", "pulse_before", "rotation_deg", "energy_eV",
                    "notes", "page", "transcription_uncertain")},
            ))
    return rows


def _num(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def catalog(root: str | Path, sessions: dict[str, dict[str, Any]]) -> dict[tuple[str, str, int], Scan]:
    """Read every scan table of every session folder (tables only, no images)."""
    root = Path(root)
    out: dict[tuple[str, str, int], Scan] = {}
    for sid, sess in sessions.items():
        for day in sess["folders"]:
            ddir = root / day
            if not ddir.is_dir():
                continue
            for txt in sorted(ddir.glob("*.txt")):
                stem = txt.stem
                if not (stem.upper().startswith("X_") or stem.upper().startswith("MONO_S_")):
                    continue
                folder = txt.with_suffix("")
                if not folder.is_dir():
                    continue
                m = _NUM.search(stem)
                if not m or not sess["scan_lo"] <= int(m.group(1)) <= sess["scan_hi"]:
                    continue  # day folders are shared between samples; numbers decide
                cols = loaders.read_scan_table(txt)
                names = [n for n in cols["ImageNameTable"] if n]
                if not names:
                    continue
                settings = loaders.read_beamline_settings(folder / "BeamlineSettings.txt")
                pol, _ = loaders.resolve_pol(names[0], loaders.pol_from_settings(settings))
                e = np.array([_num(v) for v in cols.get("EnergyTable", [])])
                t = np.array([_num(v) for v in cols.get("TemperatureTable", [])])
                fov = sorted({v for v in cols.get("FOVTable", []) if v and _num(v) == _num(v)})
                key = re.sub(r"[_\-]*\d*$", "", names[0])
                kind = "X" if stem.upper().startswith("X_") else "Mono"
                scan = Scan(
                    session=sid, day=day, number=int(m.group(1)), kind=kind, path=str(txt), n=len(names),
                    key=key, pol=pol,
                    energy=float(np.nanmedian(e)) if e.size else float("nan"),
                    e_min=float(np.nanmin(e)) if e.size else float("nan"),
                    e_max=float(np.nanmax(e)) if e.size else float("nan"),
                    temperature=float(np.nanmean(t)) if np.isfinite(t).any() else float("nan"),
                    fov=";".join(fov), n_tif=len(list(folder.glob("*.tif"))),
                )
                out[(sid, kind, scan.number)] = scan
    return out


# ----------------------------------------------------------------------------- planning
def _unit_kind(scans: list[Scan]) -> str | None:
    pols = {s.pol for s in scans}
    keys = [s.key for s in scans]
    if pols == {"CP", "CM"}:
        return "XMCD"
    if pols == {"LH", "LV"}:
        return "XLD_pol"
    if len(set(keys)) == 2 and all(k[-2:] in ("_A", "_B") for k in keys):
        return "XLD_L2"
    return None


def split_units(scans: list[Scan]) -> tuple[list[Unit], list[Scan]]:
    """Consecutive X scans -> triplets (a, b, a) or pairs (a, b); leftovers are returned."""
    scans = sorted(scans, key=lambda s: s.number)
    units, left, i = [], [], 0
    while i < len(scans):
        tri = scans[i:i + 3]
        if (len(tri) == 3 and tri[0].key == tri[2].key != tri[1].key
                and tri[2].number - tri[0].number == 2 and _unit_kind(tri)):
            units.append(Unit(f"X_{tri[0].number:04d}-{tri[2].number:04d}", _unit_kind(tri), tri))
            i += 3
            continue
        pair = scans[i:i + 2]
        if len(pair) == 2 and pair[0].key != pair[1].key and pair[1].number - pair[0].number == 1 and _unit_kind(pair):
            units.append(Unit(f"X_{pair[0].number:04d}-{pair[1].number:04d}", _unit_kind(pair), pair))
            i += 2
            continue
        left.append(scans[i])
        i += 1
    return units, left


def _auto_blocks(scans: list[Scan]) -> list[list[Scan]]:
    blocks: list[list[Scan]] = []
    for s in sorted(scans, key=lambda s: s.number):
        if blocks:
            p = blocks[-1][-1]
            same = (s.number - p.number == 1 and s.fov == p.fov and abs(s.energy - p.energy) < 0.6
                    and (abs(s.temperature - p.temperature) < 3 or not np.isfinite(s.temperature)))
            if same:
                blocks[-1].append(s)
                continue
        blocks.append([s])
    return blocks


def plan(root: str | Path, sessions: dict, conditions: list[Condition],
         cat: dict[tuple[str, str, int], Scan] | None = None) -> dict[str, Any]:
    cat = cat if cat is not None else catalog(root, sessions)
    clusters: list[Cluster] = []
    used: set[tuple[str, str, int]] = set()
    for cond in (c for c in conditions if c.kind == "X"):
        scans = [cat[(cond.session, "X", n)] for n in range(cond.first, cond.last + 1)
                 if (cond.session, "X", n) in cat]
        if not scans:
            continue
        used.update((s.session, "X", s.number) for s in scans)
        units, left = split_units(scans)
        clusters.append(Cluster(cond.cid, cond.session, sessions[cond.session]["sample"], scans[0].day,
                                cond, units, left))
    leftovers = [s for k, s in cat.items() if k[1] == "X" and k not in used]
    by_session: dict[str, list[Scan]] = {}
    for s in leftovers:
        by_session.setdefault(s.session, []).append(s)
    for sid, scans in by_session.items():
        for block in _auto_blocks(scans):
            cond = Condition(sid, "X", block[0].number, block[-1].number,
                             T_K_log=f"{np.nanmean([b.temperature for b in block]):.0f}",
                             energy_eV=f"{block[0].energy:.2f}", notes="not in logbook transcription",
                             source="auto")
            units, left = split_units(block)
            clusters.append(Cluster(cond.cid + "_auto", sid, sessions[sid]["sample"], block[0].day, cond, units, left))

    mono_cond = {(c.session, c.first): c for c in conditions if c.kind == "Mono"}
    monos = sorted((s for k, s in cat.items() if k[1] == "Mono"), key=lambda s: (s.session, s.number))
    jobs: list[SpectraJob] = []

    def cond_of(s: Scan) -> Condition:
        return mono_cond.get((s.session, s.number)) or Condition(
            s.session, "Mono", s.number, s.number, T_K_log=f"{s.temperature:.0f}", source="auto")

    for s in monos:
        jobs.append(SpectraJob(f"{s.session}_{s.label}", s.session, sessions[s.session]["sample"], s.day,
                               "XAS", [s], [cond_of(s)]))

    def same_condition(a: Scan, b: Scan) -> bool:
        ca, cb = cond_of(a), cond_of(b)
        t_ok = not (np.isfinite(a.temperature) and np.isfinite(b.temperature)) or abs(a.temperature - b.temperature) < 5
        return (a.session == b.session and a.fov == b.fov and t_ok
                and ca.rotation_deg == cb.rotation_deg and ca.ccs_mA == cb.ccs_mA
                and not cb.pulse_before)

    # pairs: each CP (LH) stack with the closest CM (LV) stack in the same condition,
    # preferring the same energy range (short/long scans); a stack is paired once
    comp = {"CP": "CM", "LH": "LV"}
    taken: set[tuple[str, int]] = set()
    cands = []
    for a in monos:
        if a.pol not in comp:
            continue
        for b in monos:
            if b.session != a.session or b.pol != comp[a.pol] or abs(b.number - a.number) > 12:
                continue
            lo, hi = sorted((a, b), key=lambda s: s.number)
            if not same_condition(lo, hi):
                continue
            overlap = min(a.e_max, b.e_max) - max(a.e_min, b.e_min)
            if overlap < 3.0:
                continue
            mismatch = abs(a.e_min - b.e_min) + abs(a.e_max - b.e_max)
            cands.append((mismatch > 2.0, abs(b.number - a.number), a, b))
    for _, _, a, b in sorted(cands, key=lambda c: (c[0], c[1], c[2].number)):
        if (a.session, a.number) in taken or (b.session, b.number) in taken:
            continue
        taken.update({(a.session, a.number), (b.session, b.number)})
        lo, hi = sorted((a, b), key=lambda s: s.number)
        kind = "XMCD_pair" if a.pol in ("CP", "CM") else "XLD_pair"
        jobs.append(SpectraJob(f"{a.session}_{kind}_{lo.number:04d}-{hi.number:04d}", a.session,
                               sessions[a.session]["sample"], lo.day, kind, [lo, hi], [cond_of(lo), cond_of(hi)]))
    # merges: runs of same-polarization stacks in the same condition
    run: list[Scan] = []
    for s in monos + [None]:  # type: ignore[list-item]
        if run and (s is None or s.pol != run[-1].pol or s.number - run[-1].number > 2
                    or not same_condition(run[-1], s) or abs(s.e_min - run[-1].e_min) > 1.0
                    or abs(s.e_max - run[-1].e_max) > 1.0):
            if len(run) > 1:
                jobs.append(SpectraJob(f"{run[0].session}_merge_{run[0].pol}_{run[0].number:04d}-{run[-1].number:04d}",
                                       run[0].session, sessions[run[0].session]["sample"], run[0].day, "merge",
                                       list(run), [cond_of(r) for r in run]))
            run = []
        if s is not None:
            run.append(s)
    return {"catalog": cat, "clusters": clusters, "spectra": jobs}


def validate(plan_: dict[str, Any], conditions: list[Condition]) -> list[dict[str, Any]]:
    """Logbook vs data: missing scans, temperature disagreement, incomplete image folders."""
    cat = plan_["catalog"]
    issues = []
    for c in conditions:
        for n in range(c.first, c.last + 1):
            s = cat.get((c.session, c.kind, n))
            if s is None:
                issues.append({"session": c.session, "scan": f"{c.kind}_{n:04d}", "issue": "in logbook, not on disk",
                               "block": c.cid})
                continue
            t_log = _num(c.T_K_log)
            if np.isfinite(t_log) and np.isfinite(s.temperature) and abs(t_log - s.temperature) > 6:
                issues.append({"session": c.session, "scan": s.label,
                               "issue": f"T logbook {t_log:g} K vs measured {s.temperature:.1f} K", "block": c.cid})
    for s in cat.values():
        if s.n_tif < s.n:
            issues.append({"session": s.session, "scan": s.label,
                           "issue": f"{s.n - s.n_tif} of {s.n} images missing", "block": ""})
    for cl in plan_["clusters"]:
        for s in cl.unassigned:
            issues.append({"session": s.session, "scan": s.label, "issue": "not part of a complete unit",
                           "block": cl.cid})
    return issues


# ----------------------------------------------------------------------------- compute
def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return float("nan")
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den else float("nan")


def _summary(a: np.ndarray) -> dict[str, float]:
    v = a[np.isfinite(a)]
    if not v.size:
        return {"n_finite": 0}
    return {"n_finite": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
            "p1": float(np.percentile(v, 1)), "p99": float(np.percentile(v, 99))}


@dataclass
class Settings:
    dead_map: str | None
    norm_image: str | None
    convention: str = "ue49"  # (neg-pos)/(neg+pos) = (CR-CL)/(CR+CL), same for every beamline
    fov_shrink: float = 0.9
    domain_threshold: float = 0.15
    domain_smooth: float = 1.5
    domain_min_px: int = 12


def _fov_mask(img: np.ndarray, shrink: float) -> np.ndarray:
    cy, cx, r = detector.auto_fov_circle(img)
    return detector.circular_mask(img.shape[0], img.shape[1], cy, cx, r * shrink)


def _load(scans: list[Scan], st: Settings):
    return loaders.load_scans([s.path for s in scans], dead_map_path=st.dead_map,
                              norm_image_path=st.norm_image, skip_missing=True, allow_mixed_fov=True)


def _channel_frames(td, unit: Unit) -> tuple[np.ndarray, np.ndarray, dict]:
    md = td.metadata
    scan_of = np.array(md["scan_of_frame"])
    stems = {Path(s.path).stem for s in unit.scans}
    sel = np.isin(scan_of, list(stems))
    names = np.array(md["frame_names"])
    pol = np.array(md["pol"])
    if unit.kind == "XMCD":
        a, b, tags = sel & (pol == "CP"), sel & (pol == "CM"), ("CP", "CM")
    elif unit.kind == "XLD_pol":
        a, b, tags = sel & (pol == "LH"), sel & (pol == "LV"), ("LH", "LV")
    else:
        key = np.array([re.sub(r"[_\-]*\d*$", "", n) for n in names])
        ka = sorted({k for k in key[sel] if k.endswith("_A")})
        kb = sorted({k for k in key[sel] if k.endswith("_B")})
        a, b, tags = sel & np.isin(key, ka), sel & np.isin(key, kb), ("A", "B")
    energy = np.asarray(md["beamline_table"]["series"].get("energy", [np.nan] * len(names)), dtype=float)
    info = {"tags": tags, "n_a": int(a.sum()), "n_b": int(b.sum()),
            "energy_a": float(np.nanmedian(energy[a])) if a.any() else float("nan"),
            "energy_b": float(np.nanmedian(energy[b])) if b.any() else float("nan")}
    return td.value[a], td.value[b], info


def _sign_label(kind: str, st: Settings) -> str:
    if kind == "XMCD":
        return st.convention
    return dichroism.XLD_FORMULA if kind == "XLD_pol" else "(A-B)/(A+B)"  # XLD_L2: energies A / B


def _dichroism_map(frames_a: np.ndarray, frames_b: np.ndarray, kind: str, st: Settings) -> dict[str, Any]:
    j = dichroism.joint_means(frames_a, frames_b, register=True)
    if kind == "XMCD":
        sign = dichroism.CONVENTIONS[st.convention]
        amap = sign * dichroism.asymmetry(j["mean_a"], j["mean_b"])
        name = "pcd"
    elif kind == "XLD_pol":  # A = LH, B = LV -> universal (LV-LH)/(LV+LH)
        amap = dichroism.xld_asymmetry(j["mean_a"], j["mean_b"])
        name = "pld"
    else:  # XLD_L2: one polarization, energies A / B
        amap = dichroism.asymmetry(j["mean_a"], j["mean_b"])
        name = "pld"
    mask = _fov_mask(j["mean_a"], st.fov_shrink)
    amap = np.where(mask, amap, np.nan)
    return {"kind": name, "map": amap, "mean_a": j["mean_a"], "mean_b": j["mean_b"], "mask": mask,
            "frame_shifts": j["frame_shifts"], "refine_shift": j["refine_shift"]}


def contrast_snr(amap: np.ndarray, sigma: float = 3.0) -> float:
    """Large-scale contrast / expected white noise after the same smoothing.

    Informational only. It is NOT a test for magnetic contrast: on the June 2025
    data, units above T_c (276 K, 120 K "no XMCD contrast") score 7-9 because
    non-magnetic structure (flake edges, helicity-dependent illumination) is not
    white noise, while a 100 K unit with real domains scores 6. Use the logbook
    remarks for that judgement.
    """
    noise = dichroism._noise(amap)
    if not np.isfinite(noise) or noise <= 0:
        return float("nan")
    sm = dichroism.nan_gaussian(amap, sigma)
    v = sm[np.isfinite(sm)]
    return float(v.std() / (noise / (2.0 * np.sqrt(np.pi) * sigma))) if v.size else float("nan")


def _domain(res: dict[str, Any], st: Settings, um_per_px: float | None) -> dict[str, Any] | None:
    if res["kind"] != "pcd":
        return None
    try:
        dom = domains.domain_map(res["map"], roi=res["mask"], threshold=st.domain_threshold,
                                 smooth_sigma=st.domain_smooth, min_px=st.domain_min_px,
                                 um_per_px=um_per_px, use_sda=None)
    except ValueError:
        return None
    dom["stats"]["contrast_snr"] = contrast_snr(res["map"])
    return dom


def _write_maps(out: Path, res: dict[str, Any], dom: dict[str, Any] | None, attrs: dict[str, Any],
                um_per_px: float | None) -> None:
    import tifffile
    import xarray as xr

    out.mkdir(parents=True, exist_ok=True)
    ny, nx = res["map"].shape
    coords = {"y": np.arange(ny), "x": np.arange(nx)}
    tags = attrs.get("channels", ("a", "b"))
    data = {
        res["kind"]: (("y", "x"), res["map"].astype(np.float32)),
        f"mean_{tags[0]}": (("y", "x"), res["mean_a"].astype(np.float32)),
        f"mean_{tags[1]}": (("y", "x"), res["mean_b"].astype(np.float32)),
        "roi": (("y", "x"), res["mask"].astype(np.uint8)),
    }
    if dom is not None:
        data["domain_map"] = (("y", "x"), dom["mz"].astype(np.int8))
    flat = {k: (json.dumps(v) if isinstance(v, (list, tuple, dict)) else int(v) if isinstance(v, (bool, np.bool_)) else v)
            for k, v in attrs.items() if v is not None and not (isinstance(v, float) and not np.isfinite(v))}
    if um_per_px:
        flat["um_per_px"] = um_per_px
    ds = xr.Dataset(data, coords=coords, attrs=flat)
    enc = {k: {"zlib": True, "complevel": 4} for k in data}
    ds.to_netcdf(out / "maps.nc", encoding=enc)
    tifffile.imwrite(out / f"{res['kind']}.tif", res["map"].astype(np.float32), compression="zlib")
    if dom is not None:
        tifffile.imwrite(out / "domain_map.tif", dom["mz"].astype(np.int8), compression="zlib")


def _json(path: Path, obj: Any) -> None:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)
    path.write_text(json.dumps(obj, indent=1, default=default))


def process_cluster(cl: Cluster, root: Path, st: Settings, preview=None) -> dict[str, Any]:
    """Per-unit maps + merged cluster map. Returns the index row."""
    out = root / cl.day / OUT_DIRNAME / "clusters" / cl.cid
    cond = asdict(cl.condition)
    row: dict[str, Any] = {"type": "cluster", "id": cl.cid, "session": cl.session, "sample": cl.sample,
                           "day": cl.day, "n_units": len(cl.units), "path": str(out.relative_to(root)),
                           **{f"log_{k}": v for k, v in cond.items() if k not in ("session",)}}
    if not cl.units:
        row["status"] = "no complete units"
        return row
    scans = [s for u in cl.units for s in u.scans]
    td = _load(scans, st)
    md = td.metadata
    upp = md.get("um_per_px")
    unit_maps, unit_rows = [], []
    for u in cl.units:
        a, b, info = _channel_frames(td, u)
        if not len(a) or not len(b):
            unit_rows.append({"unit": u.uid, "status": "missing a channel"})
            continue
        res = _dichroism_map(a, b, u.kind, st)
        dom = _domain(res, st, upp)
        t = float(np.nanmean([s.temperature for s in u.scans]))
        attrs = {"unit": u.uid, "kind": u.kind, "channels": info["tags"], "n_a": info["n_a"], "n_b": info["n_b"],
                 "energy_a_eV": info["energy_a"], "energy_b_eV": info["energy_b"], "T_K": t,
                 "convention": _sign_label(u.kind, st),
                 "refine_shift": list(res["refine_shift"]), "cluster": cl.cid, "sample": cl.sample,
                 "domain_map_note": ("no background ROI: offset not removed; segmentation is only meaningful "
                                     "where the logbook reports magnetic contrast" if dom else ""),
                 "logbook_notes": cl.condition.notes}
        uo = out / "units" / u.uid
        _write_maps(uo, res, dom, attrs, upp)
        stats = {**attrs, "map": _summary(res["map"]), "domain": dom["stats"] if dom else None,
                 "frame_shifts_px": res["frame_shifts"]}
        _json(uo / "summary.json", stats)
        if preview:
            preview(uo / "preview.png", res, dom, f"{cl.sample} | {u.uid} | {u.kind} | {t:.0f} K | {_cond_text(cl.condition)}")
        unit_maps.append(res["map"])
        unit_rows.append({"unit": u.uid, "kind": u.kind, "T_K": t, **{f"map_{k}": v for k, v in _summary(res["map"]).items()},
                          "f_up": dom["stats"]["f_up"] if dom else "", "f_down": dom["stats"]["f_down"] if dom else "",
                          "contrast_snr": dom["stats"]["contrast_snr"] if dom else contrast_snr(res["map"])})
    # merged cluster map (all frames of all units, joint registration)
    kinds = {u.kind for u in cl.units}
    merged = None
    if len(kinds) == 1 and len(cl.units) > 1:
        kind = kinds.pop()
        fa, fb = [], []
        for u in cl.units:
            a, b, info = _channel_frames(td, u)
            fa.append(a)
            fb.append(b)
        merged = _dichroism_map(np.concatenate(fa), np.concatenate(fb), kind, st)
        mdom = _domain(merged, st, upp)
        attrs = {"cluster": cl.cid, "kind": kind, "channels": info["tags"], "units": [u.uid for u in cl.units],
                 "n_a": int(sum(len(x) for x in fa)), "n_b": int(sum(len(x) for x in fb)),
                 "convention": _sign_label(kind, st), "sample": cl.sample}
        _write_maps(out / "merged", merged, mdom, attrs, upp)
        if preview:
            preview(out / "merged" / "preview.png", merged, mdom,
                    f"{cl.sample} | {cl.cid} merged ({len(cl.units)} units) | {_cond_text(cl.condition)}")
        row.update({f"merged_{k}": v for k, v in _summary(merged["map"]).items()})
        row["merged_contrast_snr"] = contrast_snr(merged["map"])
        if mdom:
            row.update({"merged_f_up": mdom["stats"]["f_up"], "merged_f_down": mdom["stats"]["f_down"]})
    pair_r = [_corr(x, y) for i, x in enumerate(unit_maps) for y in unit_maps[i + 1:]]
    repeat = {"pairwise_corr": pair_r, "mean_corr": float(np.nanmean(pair_r)) if pair_r else None}
    _json(out / "cluster.json", {"cluster": cl.cid, "sample": cl.sample, "condition": cond,
                                 "units": unit_rows, "repeatability": repeat,
                                 "unassigned_scans": [s.label for s in cl.unassigned],
                                 "missing_frames": md.get("missing_frames", []),
                                 "temperature_K_measured": float(np.nanmean([s.temperature for s in scans])),
                                 "fov_um": md.get("fov_values_um"), "um_per_px": upp})
    row.update({"status": "ok", "repeat_mean_corr": repeat["mean_corr"],
                "T_K_measured": float(np.nanmean([s.temperature for s in scans])),
                "fov_um": ";".join(f"{v:g}" for v in md.get("fov_values_um", [])),
                "missing_frames": len(md.get("missing_frames", []))})
    return row


def _cond_text(c: Condition) -> str:
    bits = []
    if c.ccs_mA:
        bits.append(f"CCS {c.ccs_mA} mA")
    if c.pulse_before:
        bits.append(c.pulse_before)
    if c.rotation_deg and c.rotation_deg != "0":
        bits.append(f"rot {c.rotation_deg} deg")
    if c.notes:
        bits.append(c.notes)
    return "; ".join(bits) or ("auto group" if c.source == "auto" else "no field/pulse noted")


def _spectrum(td, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    stack = td.value
    if mask is None:
        mask = _fov_mask(stack.mean(axis=0), 0.7)
    e = np.asarray(td.metadata["beamline_table"]["series"]["energy"], dtype=float)
    mu = stack[:, mask].mean(axis=1)
    order = np.argsort(e)
    return e[order], mu[order]


def normalize_xas(e: np.ndarray, mu: np.ndarray, pre: tuple[float, float] | None = None) -> np.ndarray:
    """Pre-edge mean -> 0, post-edge mean (last 5 % of range) -> 1."""
    pre_m = (e >= pre[0]) & (e <= pre[1]) if pre else e <= e.min() + 0.05 * np.ptp(e)
    post_m = e >= e.max() - 0.05 * np.ptp(e)
    if not pre_m.any():
        pre_m = e <= e.min() + 0.05 * np.ptp(e)
    p0, p1 = mu[pre_m].mean(), mu[post_m].mean()
    return (mu - p0) / (p1 - p0) if p1 != p0 else mu - p0


def merge_spectra(spectra: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Igor MergeScans, averaged: interpolate onto the common range at the finest step."""
    lo = max(e.min() for e, _ in spectra)
    hi = min(e.max() for e, _ in spectra)
    if hi <= lo:
        raise ValueError("spectra do not overlap")
    step = min(np.median(np.diff(e)) for e, _ in spectra)
    grid = np.arange(lo, hi + step / 2, step)
    return grid, np.mean([np.interp(grid, e, m) for e, m in spectra], axis=0)


def _edge(sample_edge: str) -> str:
    return "Fe" if "Fe" in sample_edge else "Cr"


def process_spectra(job: SpectraJob, root: Path, st: Settings, sessions: dict, preview=None) -> dict[str, Any]:
    out = root / job.day / OUT_DIRNAME / "spectra" / job.sid
    out.mkdir(parents=True, exist_ok=True)
    sess = sessions[job.session]
    edge = _edge(sess["edge"])
    pre, l3, l2 = EDGE_WINDOWS[edge]
    row: dict[str, Any] = {"type": f"spectra:{job.kind}", "id": job.sid, "session": job.session, "sample": job.sample,
                           "day": job.day, "path": str(out.relative_to(root)),
                           "scans": ";".join(s.label for s in job.scans),
                           "pols": ";".join(s.pol for s in job.scans),
                           "T_K_measured": float(np.nanmean([s.temperature for s in job.scans])),
                           "log_notes": " | ".join(c.notes for c in job.conditions if c.notes),
                           "log_rotation_deg": job.conditions[0].rotation_deg,
                           "log_ccs_mA": job.conditions[0].ccs_mA}
    tds = [_load([s], st) for s in job.scans]
    # one common ROI (first stack's central disk) so paired spectra see the same area
    mask = _fov_mask(tds[0].value.mean(axis=0), 0.7)
    specs = [_spectrum(td, mask if td.value.shape[1:] == mask.shape else None) for td in tds]
    lines = {}
    for s, (e, mu) in zip(job.scans, specs):
        lines[f"{s.label}_{s.pol}"] = (e, mu)
    summary: dict[str, Any] = {"job": job.sid, "kind": job.kind, "sample": job.sample, "edge": edge,
                               "conditions": [asdict(c) for c in job.conditions],
                               "scans": [asdict(s) for s in job.scans], "roi": "central disk, 70 % of FOV"}
    if job.kind in ("XAS", "merge"):
        if job.kind == "merge":
            e, mu = merge_spectra(specs)
            lines["merged"] = (e, mu)
        else:
            e, mu = specs[0]
        norm = normalize_xas(e, mu, pre)
        _write_csv(out / "spectrum.csv", {"energy_eV": e, "mu": mu, "mu_norm": norm})
        summary["peak_eV"] = float(e[np.argmax(norm)])
        row["peak_eV"] = summary["peak_eV"]
    elif job.kind == "XMCD_pair":
        (e1, m1), (e2, m2) = specs
        grid, _ = merge_spectra(specs)
        mp = np.interp(grid, e1, m1) if job.scans[0].pol == "CP" else np.interp(grid, e2, m2)
        mm = np.interp(grid, e2, m2) if job.scans[0].pol == "CP" else np.interp(grid, e1, m1)
        # each stack: pre-edge mean -> 0, own L3 peak -> 1 (the stacks are taken minutes
        # apart with different backgrounds and often almost no edge jump, so edge-jump
        # normalization is unstable; June 2025 S1 0257/0258 post-edge 1.5 vs 0.4)
        def peak_norm(m):
            pm = (grid >= pre[0]) & (grid <= pre[1])
            m0 = m - (m[pm].mean() if pm.any() else m[:5].mean())
            l3m = (grid >= l3[0]) & (grid <= l3[1])
            return m0 / m0[l3m].max()
        mu_p, mu_m = peak_norm(mp), peak_norm(mm)
        xmcd = mu_p - mu_m
        _write_csv(out / "xmcd_spectrum.csv", {"energy_eV": grid, "mu_plus": mu_p, "mu_minus": mu_m,
                                               "xmcd_plus_minus": xmcd, "mu_plus_raw": mp, "mu_minus_raw": mm})
        lines = {"mu+ (CP)": (grid, mu_p), "mu- (CM)": (grid, mu_m), "XMCD mu+ - mu-": (grid, xmcd)}
        l3m = (grid >= l3[0]) & (grid <= l3[1])
        l2m = (grid >= l2[0]) & (grid <= l2[1])
        summary["xmcd_L3_percent_of_peak"] = float(100 * xmcd[l3m][np.argmax(np.abs(xmcd[l3m]))])
        summary["xmcd_L2_percent_of_peak"] = float(100 * xmcd[l2m][np.argmax(np.abs(xmcd[l2m]))])
        same_sign = np.sign(summary["xmcd_L3_percent_of_peak"]) == np.sign(summary["xmcd_L2_percent_of_peak"])
        summary["xmcd_quality"] = ("SUSPECT: L3 and L2 dichroism have the same sign; stack-to-stack intensity "
                                   "drift dominates (real XMCD reverses between L3 and L2)" if same_sign
                                   else "L3/L2 signs opposite as expected")
        row["xmcd_quality"] = "suspect" if same_sign else "ok"
        summary["sumrule"] = None
        summary["sumrule_note"] = ("not computed in batch: the stack backgrounds and edge jumps are not reliable "
                                   "enough for unattended sum rules; use the PEEM suite's stack / BG / sum-rule tab with a "
                                   "hand-set two-step background")
        row.update({"xmcd_L3_pct": summary["xmcd_L3_percent_of_peak"], "xmcd_L2_pct": summary["xmcd_L2_percent_of_peak"]})
        # L3 dichroism map from the stacks (frames within +-0.3 eV of the L3 XMCD maximum)
        e_l3 = float(grid[(grid >= l3[0]) & (grid <= l3[1])][np.argmax(np.abs(xmcd[(grid >= l3[0]) & (grid <= l3[1])]))])
        summary["map_energy_eV"] = e_l3
        _stack_map(tds, job, e_l3, 0.3, "XMCD", st, out, summary, preview)
    elif job.kind == "XLD_pair":
        (e1, m1), (e2, m2) = specs
        grid, _ = merge_spectra(specs)
        lh = np.interp(grid, *specs[0]) if job.scans[0].pol == "LH" else np.interp(grid, *specs[1])
        lv = np.interp(grid, *specs[1]) if job.scans[0].pol == "LH" else np.interp(grid, *specs[0])
        nlh, nlv = normalize_xas(grid, lh, pre), normalize_xas(grid, lv, pre)
        _write_csv(out / "xld_spectrum.csv", {"energy_eV": grid, "LH_norm": nlh, "LV_norm": nlv, "xld_LV_minus_LH": nlv - nlh})
        lines = {"LH": (grid, nlh), "LV": (grid, nlv), "XLD LV-LH": (grid, nlv - nlh)}
        ea, eb = _num(sess.get("xld_e_a_eV")), _num(sess.get("xld_e_b_eV"))
        if np.isfinite(ea) and np.isfinite(eb) and edge == "Cr":
            _ld_maps(tds, job, ea, eb, st, out, summary, preview)
    _json(out / "summary.json", summary)
    if preview:
        preview.spectra(out / "spectra.png", lines, f"{job.sample} | {job.sid} | {row['T_K_measured']:.0f} K")
    row["status"] = "ok"
    return row


def _write_csv(path: Path, cols: dict[str, np.ndarray]) -> None:
    keys = list(cols)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for vals in zip(*(cols[k] for k in keys)):
            w.writerow([f"{v:.6g}" for v in vals])


def _frames_near(td, e0: float, hw: float) -> np.ndarray:
    e = np.asarray(td.metadata["beamline_table"]["series"]["energy"], dtype=float)
    return td.value[dichroism.nearest_frames(e, e0, hw)]


def _stack_map(tds, job, e0, hw, kind, st, out, summary, preview) -> None:
    if tds[0].value.shape[1:] != tds[1].value.shape[1:]:
        summary["map_error"] = "image sizes differ"
        return
    a = _frames_near(tds[0], e0, hw)
    b = _frames_near(tds[1], e0, hw)
    if job.scans[0].pol in ("CM", "LV"):
        a, b = b, a
    res = _dichroism_map(a, b, kind, st)
    upp = tds[0].metadata.get("um_per_px")
    dom = _domain(res, st, upp)
    _write_maps(out / "map_L3", res, dom, {"energy_eV": e0, "half_width_eV": hw, "kind": kind,
                                          "channels": ("CP", "CM"), "convention": st.convention}, upp)
    summary["map_L3"] = _summary(res["map"])
    if preview:
        preview(out / "map_L3" / "preview.png", res, dom, f"{job.sid} pcd from energy stacks at {e0:.2f} eV")


def _ld_maps(tds, job, ea, eb, st, out, summary, preview) -> None:
    if tds[0].value.shape[1:] != tds[1].value.shape[1:]:
        summary["map_error"] = "image sizes differ"
        return
    lh_td, lv_td = (tds[0], tds[1]) if job.scans[0].pol == "LH" else (tds[1], tds[0])
    e_lh = np.asarray(lh_td.metadata["beamline_table"]["series"]["energy"], dtype=float)
    e_lv = np.asarray(lv_td.metadata["beamline_table"]["series"]["energy"], dtype=float)
    res = dichroism.pmld_pmnld_maps(lh_td.value, lv_td.value, e_lh, e_lv, ea, eb, half_width=0.15, register=True)
    mask = _fov_mask(lh_td.value.mean(axis=0), st.fov_shrink)
    maps = {k: np.where(mask, res[k], np.nan) for k in ("pld_a", "pld_b", "pmld", "pmnld")}
    import tifffile
    import xarray as xr

    (out / "ld_maps").mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset({k: (("y", "x"), v.astype(np.float32)) for k, v in maps.items()},
                    attrs={"e_a_eV": ea, "e_b_eV": eb, "half_width_eV": 0.15,
                           "pld": dichroism.XLD_FORMULA, "pmld": "[pld(E_A)-pld(E_B)]/2",
                           "pmnld": "[pld(E_A)+pld(E_B)]/2"})
    ds.to_netcdf(out / "ld_maps" / "maps.nc", encoding={k: {"zlib": True} for k in maps})
    for k, v in maps.items():
        tifffile.imwrite(out / "ld_maps" / f"{k}.tif", v.astype(np.float32), compression="zlib")
    summary["ld_maps"] = {k: _summary(v) for k, v in maps.items()}
    if preview:
        preview.ld(out / "ld_maps" / "preview.png", maps, f"{job.sid} LD at {ea}/{eb} eV")


# ----------------------------------------------------------------------------- driver
def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def run(root: str | Path, settings: Settings, *, days: Iterable[str] | None = None, skip_existing: bool = True,
        preview=None, log=print) -> dict[str, Any]:
    root = Path(root)
    sessions = read_sessions(root / "logbook_sessions.csv")
    conditions = read_conditions(root / "logbook_conditions.csv")
    p = plan(root, sessions, conditions)
    days = set(days) if days else None
    issues = validate(p, conditions)
    write_csv_rows(root / "tensorspec_peem_logbook_check.csv", issues)
    write_csv_rows(root / "tensorspec_peem_catalog.csv", [asdict(s) for s in sorted(
        p["catalog"].values(), key=lambda s: (s.session, s.kind, s.number))])
    rows: list[dict[str, Any]] = []
    todo = [("cluster", c) for c in p["clusters"]] + [("spectra", j) for j in p["spectra"]]
    for i, (typ, job) in enumerate(todo, 1):
        if days and job.day not in days:
            continue
        jid = job.cid if typ == "cluster" else job.sid
        target = root / job.day / OUT_DIRNAME / ("clusters" if typ == "cluster" else "spectra") / jid
        marker = target / ("cluster.json" if typ == "cluster" else "summary.json")
        if skip_existing and marker.exists():
            saved = target / "index_row.json"
            if saved.exists():
                rows.append({**json.loads(saved.read_text()), "status": "ok (existing)"})
            else:
                rows.append({"type": typ, "id": jid, "day": job.day, "status": "existing",
                             "path": str(target.relative_to(root))})
            continue
        try:
            row = (process_cluster(job, root, settings, preview) if typ == "cluster"
                   else process_spectra(job, root, settings, sessions, preview))
        except Exception as exc:  # keep going; record the failure
            row = {"type": typ, "id": jid, "day": job.day, "status": f"error: {type(exc).__name__}: {exc}"}
        rows.append(row)
        if str(row.get("status", "")).startswith("ok"):
            _json(target / "index_row.json", row)
        log(f"[{i}/{len(todo)}] {jid}: {row.get('status')}")
    write_csv_rows(root / "tensorspec_peem_index.csv", rows)
    for day in sorted({r["day"] for r in rows}):
        d = root / day / OUT_DIRNAME
        if d.is_dir():
            write_csv_rows(d / "index.csv", [r for r in rows if r["day"] == day])
    return {"rows": rows, "issues": issues, "plan": p}
