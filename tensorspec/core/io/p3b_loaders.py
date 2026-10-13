"""ALS PEEM-3 (beamline 11.0.1) files.

``.P3B``: a flat sequence of tagged records, little-endian, after a 4-byte lead int::

    int32 name_len, name, int32 type, int32 count, value

types: 1 bool, 2 int16, 3 int32, 4 float32, 5 float64, 6 int64, 7 string (count = bytes),
12 uint16, 13 uint32, 14 uint64. ``DATA`` (type 12) holds DIMY x DIMX uint16 pixels.
After DATA come TIME (text), EXPOSURETIME (ms), FLATFIELD / BACKGROUND flags and
beamline values: BEAMLINE_ENERGY (eV), EPU_POLARIZATION (+-0.9 circular), I0,
BEAM_CURRENT (mA), FOV (um), MAGNET_CURRENT, SAMPLE_GOAL (V), X / Y / Z stage, ...

An acquisition folder ``<name>/<name>#NNNN.P3B`` holds one frame per file; a
``Result.P3B`` / ``Result.TIF`` is the beamline viewer's processed image (cropped,
offset ~32768) and is not used. ``<name>.dat`` next to the folder logs one row per
frame (cryostat temperature, pressures, lens voltages; column sets vary).

``load_p3b_folders`` builds a PEEM dataset; the polarization of every frame is
its EPU_POLARIZATION (+0.9 -> "CP", -0.9 -> "CM"), so ``pcd`` with convention
the default convention ``ue49`` gives (I(-0.9) - I(+0.9))/(I(-0.9) + I(+0.9)).
"""
from __future__ import annotations

import datetime as _dt
import struct
from pathlib import Path
from typing import Any

import numpy as np

from tensorspec.core.data_models import TensorData
from tensorspec.core.io.peem_loaders import package_stack

_SIZES = {1: 1, 2: 2, 3: 4, 4: 4, 5: 8, 6: 8, 12: 2, 13: 4, 14: 8}
_FMT = {1: "?", 2: "h", 3: "i", 4: "f", 5: "d", 6: "q", 12: "H", 13: "I", 14: "Q"}


def read_p3b(path: str | Path, with_image: bool = True) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return (image (DIMY, DIMX) float32 or None, metadata)."""
    b = Path(path).read_bytes()
    p = 4
    meta: dict[str, Any] = {}
    img = None
    while p < len(b) - 8:
        nl = struct.unpack_from("<i", b, p)[0]
        if not 0 < nl <= 64:
            break
        p += 4
        name = b[p:p + nl].decode("latin1")
        p += nl
        typ, cnt = struct.unpack_from("<ii", b, p)
        p += 8
        if typ == 7:
            meta[name] = b[p:p + cnt].decode("latin1")
            p += cnt
            continue
        if typ not in _SIZES or cnt < 0:
            meta.setdefault("_unparsed", name)
            break
        n = _SIZES[typ] * cnt
        if name == "DATA":
            if with_image:
                img = np.frombuffer(b, "<" + _FMT[typ], count=cnt, offset=p).astype(np.float32)
            p += n
            continue
        v = struct.unpack_from(f"<{cnt}{_FMT[typ]}", b, p)
        p += n
        meta[name] = v[0] if cnt == 1 else list(v)
    if "DIMX" not in meta or "DIMY" not in meta:
        raise ValueError(f"{path}: not a P3B image (no DIMX/DIMY)")
    if img is not None:
        img = img.reshape(meta["DIMY"], meta["DIMX"])
    t = meta.get("TIME")
    if t:
        try:
            meta["timestamp"] = _dt.datetime.strptime(t.strip(), "%a %b %d %H:%M:%S %Y").isoformat()
        except ValueError:
            meta["timestamp"] = None
    return img, meta


def read_dat(path: str | Path) -> dict[str, np.ndarray]:
    """Per-frame log -> {column: values}. Columns vary between acquisitions."""
    lines = Path(path).read_text(encoding="latin1").splitlines()
    if not lines:
        return {}
    hdr = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        cells = ln.split("\t")
        if not ln or ln[0] not in "-0123456789" or len(cells) < len(hdr) - 3:
            continue
        vals = []
        for x in cells[:len(hdr)]:
            try:
                vals.append(float(x))
            except ValueError:
                vals.append(np.nan)
        rows.append(vals + [np.nan] * (len(hdr) - len(vals)))
    a = np.array(rows, dtype=float) if rows else np.empty((0, len(hdr)))
    return {h: a[:, i] for i, h in enumerate(hdr)}


def _is_raw_frame(path: Path) -> bool:
    try:
        return read_p3b(path, with_image=False)[1].get("EPU_POLARIZATION") is not None
    except Exception:  # noqa: BLE001 - not a readable frame
        return False


def frame_files(acq: str | Path) -> list[Path]:
    acq = Path(acq)
    if acq.is_dir():
        numbered = sorted(acq.glob("*#*.P3B"))
        if numbered:
            return numbered
        # renamed frames: every .P3B that is a raw frame (has EPU_POLARIZATION; the viewer's
        # processed Result.P3B has none)
        return [f for f in sorted(acq.glob("*.[Pp]3[Bb]")) if _is_raw_frame(f)]
    return [acq] if acq.suffix.upper() == ".P3B" else []


def um_per_px(meta: dict[str, Any]) -> float | None:
    """FOV spans the longer image side."""
    fov = meta.get("FOV")
    return float(fov) / max(meta["DIMX"], meta["DIMY"]) if fov else None


def i0_reliable(metas: list[dict[str, Any]]) -> tuple[bool, str]:
    """I0 is used only when it behaves like a flux monitor: all positive, coefficient of
    variation <= 0.2. On 2021-06-01 it jumped randomly between 0 and 0.12 while the image
    intensity stayed constant to 1 %, and normalizing by it created fake offsets."""
    i0 = np.array([m.get("I0", np.nan) for m in metas], dtype=float)
    if not np.all(np.isfinite(i0)) or np.any(i0 <= 0):
        return False, "I0 missing or non-positive"
    cv = float(np.std(i0) / np.mean(i0))
    return (cv <= 0.2), f"I0 coefficient of variation {cv:.2f}"


def load_p3b_folders(paths: list[str | Path], i0_normalize: bool = False,
                     allow_mixed_fov: bool = False) -> TensorData:
    """Acquisition folders (or single ``.P3B`` frames) -> PEEM ``(frame, y, x)`` TensorData.

    I0 normalization is off by default: on 2021-06-26 I0 passed the variation check but
    did not track the image intensity and flipped the sign of the asymmetry. When asked
    for, it is applied only if ``i0_reliable`` passes (the reason is in the metadata).
    """
    if not paths:
        raise ValueError("no P3B folders given")
    frames, metas, names, scan_of, pol = [], [], [], [], []
    pol_by_scan: dict[str, list[str]] = {}
    for p in map(Path, paths):
        files = frame_files(p)
        if not files:
            raise ValueError(f"{p}: no .P3B frames")
        scan = p.name if p.is_dir() else p.name.split("#")[0]
        tags = []
        for f in files:
            img, m = read_p3b(f)
            e = m.get("EPU_POLARIZATION")
            if e is None or e == 0:
                raise ValueError(f"{f.name}: EPU_POLARIZATION is {e}; only circular (+/-) frames are supported")
            tag = "CP" if e > 0 else "CM"
            frames.append(img)
            metas.append(m)
            names.append(f.name)
            scan_of.append(scan)
            pol.append(tag)
            tags.append(tag)
        pol_by_scan[scan] = sorted(set(tags))
    if len({f.shape for f in frames}) != 1:
        raise ValueError("acquisitions have different image sizes")
    fovs = {round(float(m["FOV"]), 3) for m in metas if m.get("FOV")}
    if len(fovs) > 1 and not allow_mixed_fov:
        raise ValueError(f"acquisitions have different fields of view {sorted(fovs)} (um); "
                         "pixels do not correspond. Load matching acquisitions.")
    stack = np.stack(frames)
    i0 = np.array([m.get("I0", np.nan) for m in metas], dtype=float)
    ok, why = i0_reliable(metas)
    used = bool(i0_normalize and ok)
    if used:
        stack = stack / i0[:, None, None] * float(np.sqrt(np.mean(i0 ** 2)))
    n, ny, nx = stack.shape
    energy = [m.get("BEAMLINE_ENERGY") for m in metas]
    metadata: dict[str, Any] = {
        "frame_names": names,
        "pol": pol,
        "pol_source": {"CP": "I(+0.9)", "CM": "I(-0.9)"},
        "scan_of_frame": scan_of,
        "scans": list(pol_by_scan),
        "pair_id": None,
        "source": ";".join(str(p) for p in paths),
        "loader": "p3b",
        "beamline": "ALS 11.0.1 PEEM-3",
        "I0": i0.tolist(),
        "i0_normalized": used,
        "i0_check": why,
        "dropped_frames": 0,
        "missing_frames": [],
        "fov_um": next(iter(fovs)) if len(fovs) == 1 else None,
        "um_per_px": um_per_px(metas[0]) if len(fovs) == 1 else None,
        "fov_values_um": sorted(fovs),
        "temperature_K": None,
        "timestamps": [m.get("timestamp") for m in metas],
        "pol_by_scan": pol_by_scan,
        "pol_conflicts": [],
        "beamline_table": {"columns": ["energy", "current"],
                           "series": {"energy": energy, "current": [m.get("BEAM_CURRENT") for m in metas]}},
    }
    tensor = package_stack(stack, names, source=metadata["source"], loader=metadata["loader"], pol=pol,
                           dtype=np.float32)
    tensor.metadata.update(metadata)
    return tensor
