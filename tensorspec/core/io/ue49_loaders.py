"""UE49 SPEEM scan loader (port of Igor ``autoFileLoad`` / ``LoadTPXTiff``).

On-disk layout written by the UE49 acquisition::

    <day>/X_0002.txt                 tab table, one row per image
    <day>/X_0002/Cr_pos_0000.tif     images named by ImageNameTable
    <day>/X_0002/BeamlineSettings.txt  NameTable / ValueTable (PolTable, ...)

``X_*`` scans are image series at fixed energy; ``Mono_S_*`` scans are energy
stacks whose image names are the photon energies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tensorspec.core.data_models import TensorData
from tensorspec.core.io.peem_loaders import infer_pol_from_name, package_stack
from tensorspec.core.peem.ue49 import detector

# BeamlineSettings PolTable: 0 lin.hor, 1 lin.vert, 2 circular+, 3 circular-
POL_TABLE = {0: "LH", 1: "LV", 2: "CP", 3: "CM"}
UE49_POL_ALIASES = (("pos", "CP"), ("neg", "CM"), ("lh", "LH"), ("lv", "LV"))  # Cr_pos_0000, ...
MAX_UE49_FRAMES = 10_000


def read_scan_table(txt_path: str | Path) -> dict[str, list[str]]:
    """Tab-delimited scan table -> {column: [str, ...]}."""
    lines = [ln for ln in Path(txt_path).read_text(errors="replace").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty scan table: {txt_path}")
    header = [h.strip() for h in lines[0].split("\t")]
    cols: dict[str, list[str]] = {h: [] for h in header}
    for ln in lines[1:]:
        cells = ln.split("\t")
        for i, h in enumerate(header):
            cols[h].append(cells[i].strip() if i < len(cells) else "")
    if "ImageNameTable" not in cols:
        raise ValueError(f"{txt_path} has no ImageNameTable column")
    return cols


def read_beamline_settings(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.is_file():
        return {}
    out: dict[str, str] = {}
    for ln in p.read_text(errors="replace").splitlines()[1:]:
        parts = ln.split("\t")
        if len(parts) >= 2:
            out[parts[0].strip()] = parts[1].strip()
    return out


def pol_from_settings(settings: dict[str, str]) -> str | None:
    for key, value in settings.items():
        if key.startswith("PolTable"):
            try:
                return POL_TABLE.get(int(float(value)))
            except ValueError:
                return None
    return None


def pol_from_name(name: str) -> str:
    """UE49 file names: pos / neg / lh / lv tokens, then CP / CM / LH / LV (peem_loaders)."""
    return infer_pol_from_name(name, UE49_POL_ALIASES)


def resolve_pol(name: str, scan_pol: str | None) -> tuple[str, str]:
    """(tag, source). The filename token wins: on the June 2025 beamtime the
    BeamlineSettings PolTable lagged the helicity switch by one scan
    (X_0003 'Cr_neg' recorded circular+). PolTable is used only for names
    without a token (e.g. Mono_S 'Img_0000')."""
    from_name = pol_from_name(name)
    if from_name != "unknown":
        return from_name, "filename"
    if scan_pol:
        return scan_pol, "BeamlineSettings"
    return "unknown", "none"


def _float_column(cols: dict[str, list[str]], name: str) -> np.ndarray | None:
    if name not in cols:
        return None
    try:
        return np.asarray([float(v) if v else np.nan for v in cols[name]], dtype=float)
    except ValueError:
        return None


def flag_bad_frames(i0: np.ndarray | None, bstab: np.ndarray | None = None,
                    bstab_limit: float = 0.05) -> np.ndarray:
    """Igor ``RemoveBadDataPoints``: I0 spikes (> 5x mean jump on both sides) or
    beam instability sqrt(HStab^2+VStab^2) > ``bstab_limit``. Returns bool mask."""
    n = len(i0) if i0 is not None else (len(bstab) if bstab is not None else 0)
    bad = np.zeros(n, dtype=bool)
    if i0 is not None and n >= 3:
        jumps = np.abs(np.diff(i0))
        p2p = 5.0 * np.nanmean(jumps)
        spike = (jumps[:-1] > p2p) & (jumps[1:] > p2p)
        bad[1:-1] |= spike
    if bstab is not None:
        bad |= np.abs(np.nan_to_num(bstab)) > bstab_limit
    return bad


def load_scan(
    txt_path: str | Path,
    *,
    dead_map: np.ndarray | None = None,
    norm_image: np.ndarray | None = None,
    tpx3: bool = True,
    name_filter: str | None = None,
    skip_missing: bool = False,
    pol: str | None = None,
) -> dict[str, Any]:
    """Load one scan. ``pol`` (CP / CM / LH / LV, set by the user) tags every frame and overrides
    the image-name tokens and the PolTable. Returns dict(frames (n,y,x), names, pol, table, settings, missing).

    ``skip_missing`` drops rows whose TIFF is absent (listed in ``missing``)
    instead of raising.
    """
    txt_path = Path(txt_path)
    cols = read_scan_table(txt_path)
    folder = txt_path.with_suffix("")
    if not folder.is_dir():
        raise ValueError(f"image folder not found next to {txt_path.name}: {folder}")
    settings = read_beamline_settings(folder / "BeamlineSettings.txt")
    scan_pol = pol_from_settings(settings)

    import re
    import tifffile

    keep = [
        i for i, n in enumerate(cols["ImageNameTable"])
        if n and (not name_filter or re.search(name_filter, n))
    ]
    if not keep:
        raise ValueError(f"no images selected in {txt_path.name}")
    if len(keep) > MAX_UE49_FRAMES:
        raise ValueError(f"{len(keep)} frames exceeds limit {MAX_UE49_FRAMES}")

    frames, names, missing, kept = [], [], [], []
    for i in keep:
        name = cols["ImageNameTable"][i]
        tif = folder / f"{name}.tif"
        if not tif.is_file():
            if skip_missing:
                missing.append(name)
                continue
            raise ValueError(f"missing image {tif}")
        try:
            img = np.asarray(tifffile.imread(str(tif)), dtype=float)
        except Exception as exc:  # e.g. June 2025 X_0117/Cr_neg_0039.tif is a text file
            if skip_missing:
                missing.append(f"{name} (unreadable: {type(exc).__name__})")
                continue
            raise ValueError(f"unreadable image {tif}: {exc}") from exc
        kept.append(i)
        if tpx3:
            img = detector.tpx3_correct(img, dead_map, norm_image)
        frames.append(img)
        names.append(name)
    if not frames:
        raise ValueError(f"no images found for {txt_path.name}")
    keep = kept
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"inconsistent image shapes in {txt_path.name}: {sorted(shapes)}")

    table = {k: [v[i] for i in keep] for k, v in cols.items()}
    resolved = [(pol, "user") for _ in names] if pol else [resolve_pol(n, scan_pol) for n in names]
    pol = [r[0] for r in resolved]
    conflicts = sorted({n for n, (tag, src) in zip(names, resolved)
                        if src == "filename" and scan_pol and tag != scan_pol})
    return {
        "frames": np.stack(frames),
        "names": names,
        "pol": pol,
        "table": table,
        "settings": settings,
        "settings_pol": scan_pol,
        "missing": missing,
        "pol_source": sorted({r[1] for r in resolved}),
        "pol_conflicts": conflicts,
        "scan": txt_path.stem,
        "source": str(txt_path),
    }


def load_scans(
    txt_paths: Iterable[str | Path],
    *,
    dead_map_path: str | Path | None = None,
    norm_image_path: str | Path | None = None,
    tpx3: bool = True,
    i0_normalize: bool = True,
    drop_bad_frames: bool = False,
    name_filter: str | None = None,
    allow_mixed_fov: bool = False,
    skip_missing: bool = False,
    pol: dict[str, str] | None = None,
) -> TensorData:
    """Load and concatenate UE49 scans into a PEEM ``(frame, y, x)`` TensorData.

    Metadata matches the generic PEEM loaders (frame_names, pol, I0,
    beamline_table) so pair / drift / background / sum-rule work unchanged.
    The energy column is exposed as ``beamline_table.series['energy']``.
    """
    paths = [Path(p) for p in txt_paths]
    if not paths:
        raise ValueError("no scan tables given")
    dead = detector.load_detector_map(dead_map_path) if dead_map_path else None
    norm = detector.load_detector_map(norm_image_path) if norm_image_path else None
    pol_set = {str(Path(k)): v for k, v in (pol or {}).items()}
    scans = [load_scan(p, dead_map=dead, norm_image=norm, tpx3=tpx3, name_filter=name_filter,
                       skip_missing=skip_missing, pol=pol_set.get(str(p))) for p in paths]
    if len({s["frames"].shape[1:] for s in scans}) != 1:
        raise ValueError("scans have different image sizes")

    stack = np.concatenate([s["frames"] for s in scans])
    names = [n for s in scans for n in s["names"]]
    pol = [p for s in scans for p in s["pol"]]
    scan_of = [s["scan"] for s in scans for _ in s["names"]]
    columns = list(dict.fromkeys(k for s in scans for k in s["table"]))
    table = {k: [v for s in scans for v in s["table"].get(k, [""] * len(s["names"]))] for k in columns}
    series = {k: _float_column(table, k) for k in columns}
    series = {k: v.tolist() for k, v in series.items() if v is not None and np.isfinite(v).any()}

    energy = _float_column(table, "EnergyTable")
    if energy is not None:
        series["energy"] = energy.tolist()
    i0 = _float_column(table, "I0Table")
    hs, vs = _float_column(table, "HStabTable"), _float_column(table, "VStabTable")
    bstab = np.sqrt(hs**2 + vs**2) if hs is not None and vs is not None else None

    keep = np.ones(len(names), dtype=bool)
    if drop_bad_frames:
        keep = ~flag_bad_frames(i0, bstab)
        if not keep.any():
            raise ValueError("every frame was flagged as bad")

    i0_applied = False
    if i0_normalize and i0 is not None and np.all(np.isfinite(i0)) and np.all(i0 > 0):
        stack = detector.i0_normalize(stack, i0)
        i0_applied = True

    idx = np.flatnonzero(keep)
    stack = stack[idx]
    names = [names[i] for i in idx]
    pol = [pol[i] for i in idx]
    scan_of = [scan_of[i] for i in idx]
    series = {k: [v[i] for i in idx] for k, v in series.items()}

    fov = _float_column(table, "FOVTable")
    fov_values = sorted({float(v) for v in fov[idx] if np.isfinite(v)}) if fov is not None else []
    if len(fov_values) > 1 and not allow_mixed_fov:
        per_scan = {s["scan"]: sorted(set(s["table"].get("FOVTable", []))) for s in scans}
        raise ValueError(
            f"scans have different fields of view {per_scan} (um); pixels do not "
            "correspond. Load matching scans, or pass allow_mixed_fov=True."
        )
    fov_um = fov_values[0] if len(fov_values) == 1 else None  # Igor: FOV = -1 on mismatch
    n_frames, ny, nx = stack.shape
    um_per_px = fov_um / (nx - 1) if fov_um else None
    temp = _float_column(table, "TemperatureTable")

    metadata: dict[str, Any] = {
        "frame_names": names,
        "pol": pol,
        "scan_of_frame": scan_of,
        "scans": [s["scan"] for s in scans],
        "pair_id": None,
        "source": ";".join(s["source"] for s in scans),
        "loader": "ue49_scan",
        "beamline": "BESSY II UE49 SPEEM",
        "csv_attached": True,
        "beamline_csv": None,
        "I0": [i0[i] for i in idx] if i0 is not None else None,
        "i0_normalized": i0_applied,
        "tpx3_corrected": bool(tpx3),
        "dead_pixel_map": str(dead_map_path) if dead_map_path else None,
        "norm_image": str(norm_image_path) if norm_image_path else None,
        "dropped_frames": int((~keep).sum()),
        "missing_frames": [f"{s['scan']}: {n}" for s in scans for n in s["missing"]],
        "fov_um": fov_um,
        "um_per_px": um_per_px,
        "temperature_K": float(np.nanmean(temp)) if temp is not None and np.isfinite(temp).any() else None,
        "fov_values_um": fov_values,
        "pol_by_scan": {s["scan"]: sorted(set(s["pol"])) for s in scans},
        "settings_pol_by_scan": {s["scan"]: s["settings_pol"] for s in scans},
        "pol_conflicts": [f"{s['scan']}: {n}" for s in scans for n in s["pol_conflicts"][:3]],
        "beamline_settings": scans[0]["settings"],
        "beamline_table": {"columns": list(series), "series": series},
    }
    tensor = package_stack(stack, names, source=metadata["source"], loader=metadata["loader"], pol=pol,
                           dtype=np.float32)
    tensor.metadata.update(metadata)
    return tensor


def find_scans(day_dir: str | Path, prefix: str | None = None) -> list[Path]:
    """Scan tables in a beamtime day folder that have a matching image folder."""
    day = Path(day_dir)
    out = []
    for txt in sorted(day.glob("*.txt")):
        if prefix and not txt.stem.startswith(prefix):
            continue
        if txt.with_suffix("").is_dir():
            out.append(txt)
    return out
