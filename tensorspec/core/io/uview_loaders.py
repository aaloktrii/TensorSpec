"""Elmitec UView image files (``UKSOFT2001`` header, e.g. NSLS-II ESM PEEM/LEEM).

Layout used here (verified on the June 2026 BNL files, header version 9):

* file header, 104 bytes: id[20], short size, short version, short bits/pixel,
  short camera bits, short MCP diameter (unbinned px), char h/v binning,
  ..., short width @40, short height @42, short n images @44, short recipe size @46
* image header @104: short size (288), short version, ..., int64 FILETIME @112
* tagged metadata ("LEEM data"): module records ``name<unit digit>\\0`` + float32,
  gauges ``name\\0unit\\0`` + float32, FOV ``\\x6e<text>\\0`` + float32
* pixel data: little-endian uint16, the last width*height*2 bytes of the file

The tag stream has variable-length records that are not fully documented, so
values are looked up by name (robust) instead of walking every record.

``load_uview_folders`` builds a PEEM dataset from acquisition folders (one
polarization each, from the ``_CL_`` / ``_CR_`` / ``_LH_`` / ``_LV_`` name
token). CL is stored as the internal "CP" channel and CR as "CM", so ``pcd``
with the default convention ``ue49`` gives (CR - CL)/(CR + CL).
"""
from __future__ import annotations

import datetime as _dt
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

from tensorspec.core.data_models import TensorData
from tensorspec.core.io.peem_loaders import match_pol_token, package_stack

MAGIC = b"UKSOFT2001"
GAUGES = ("COL", "DPS", "MCH", "PCH")
_UNIT_DIGIT = {"0": "", "1": "V", "2": "mA", "3": "A", "4": "C", "5": "K", "6": "mV", "7": "pA", "8": "nA", "9": "uA"}


def _header(b: bytes) -> dict[str, Any]:
    if not b.startswith(MAGIC):
        raise ValueError("not a UView file (missing UKSOFT2001 id)")
    size, version, bpp = struct.unpack_from("<3h", b, 20)
    mcp_diam = struct.unpack_from("<h", b, 28)[0]
    hbin, vbin = b[30], b[31]
    width, height, n_images, recipe = struct.unpack_from("<4h", b, 40)
    if bpp != 16:
        raise ValueError(f"only 16-bit UView images are supported (got {bpp})")
    return {"version": version, "width": width, "height": height, "n_images": n_images,
            "mcp_diameter_px": mcp_diam, "binning": (hbin, vbin)}


def _module(blk: bytes, name: str) -> tuple[float, str] | None:
    """Module record ``name<unit digit>\\0`` + float32 -> (value, unit)."""
    m = re.search(re.escape(name.encode()) + rb"([0-9])\x00", blk)
    if not m or m.end() + 4 > len(blk):
        return None
    return float(struct.unpack_from("<f", blk, m.end())[0]), _UNIT_DIGIT.get(m.group(1).decode(), "")


def _gauge(blk: bytes, name: str) -> float | None:
    m = re.search(re.escape(name.encode()) + rb"[0-9]?\x00([A-Za-z]{1,6})\x00", blk)
    if not m or m.end() + 4 > len(blk):
        return None
    return float(struct.unpack_from("<f", blk, m.end())[0])


def read_uview(path: str | Path, with_image: bool = True) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return (image (h, w) float32 or None, metadata)."""
    b = Path(path).read_bytes()
    h = _header(b)
    n = h["width"] * h["height"] * 2
    if len(b) < 104 + n:
        raise ValueError(f"{path}: file too short for a {h['width']}x{h['height']} image")
    blk = b[104:len(b) - n]
    meta: dict[str, Any] = dict(h)
    ft = struct.unpack_from("<q", b, 112)[0]
    try:
        meta["timestamp"] = (_dt.datetime(1601, 1, 1) + _dt.timedelta(microseconds=ft / 10)).isoformat()
    except (OverflowError, ValueError):
        meta["timestamp"] = None
    t = _module(blk, "Sample Temp.")
    if t is not None:
        val, unit = t
        meta["sample_temp_K"] = val + 273.15 if unit == "C" else val
        meta["sample_temp_raw"] = f"{val:g} {unit}"
    sv = _module(blk, "Start Voltage")
    if sv is not None:
        meta["start_voltage_V"] = sv[0]
    for g in GAUGES:
        v = _gauge(blk, g)
        if v is not None:
            meta[f"pressure_{g}_torr"] = v
    fov = re.search(rb"\x6e([^\x00]{1,16})\x00", blk[blk.find(b"FOV") - 40 if b"FOV" in blk else 0:])
    m = re.search(rb"([0-9.]+)\s*\xb5m\x00", blk)
    if m:
        meta["fov_um"] = float(m.group(1))
    elif fov:
        meta["fov_text"] = fov.group(1).decode("latin1")
    for name in ("V CCS", "LakeShore", "Objective"):
        v = _module(blk, name)
        if v is not None:
            meta[name.replace(" ", "_").lower()] = v[0]
    img = None
    if with_image:
        img = np.frombuffer(b[len(b) - n:], dtype="<u2").reshape(h["height"], h["width"]).astype(np.float32)
    return img, meta


def um_per_px(meta: dict[str, Any]) -> float | None:
    """FOV is the channel-plate diameter; MCP diameter is given in unbinned pixels."""
    fov = meta.get("fov_um")
    diam = meta.get("mcp_diameter_px")
    hb = (meta.get("binning") or (1, 1))[0] or 1
    if fov and diam and diam > 0:
        return float(fov) / (diam / hb)
    return None


POL_ALIAS = {"CL": "CP", "CR": "CM", "LH": "LH", "LV": "LV"}
UVIEW_POL_ALIASES = tuple((k.lower(), v) for k, v in POL_ALIAS.items())  # ..._Cr2p_CL_0deg
_FOV = re.compile(r"(?:^|_)([\d.]+)um(?:_|$)")


def pol_from_name(name: str) -> str | None:
    """Beamline token CL / CR / LH / LV of a folder name (whole word; peem_loaders.match_pol_token)."""
    hit = match_pol_token(name, UVIEW_POL_ALIASES)
    return hit[0].upper() if hit else None


def _frame_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.dat"))
    return [path] if path.suffix.lower() == ".dat" else []


def load_uview_folders(paths: list[str | Path], allow_mixed_fov: bool = False,
                       pol: dict[str, str] | None = None) -> TensorData:
    """Acquisition folders (or single ``.dat`` files) -> PEEM ``(frame, y, x)`` TensorData.

    UView files do not record the polarization, so it comes from ``pol`` ({path: CL / CR / LH /
    LV}, set by the user) or else from the CL / CR / LH / LV token of the folder name."""
    pol_set = {str(Path(k)): v.upper() for k, v in (pol or {}).items()}
    if not paths:
        raise ValueError("no UView folders given")
    frames, pol, names, scan_of, metas, bad = [], [], [], [], [], []
    pol_by_scan: dict[str, list[str]] = {}
    fov_by_scan: dict[str, float | None] = {}
    for p in map(Path, paths):
        files = _frame_files(p)
        if not files:
            raise ValueError(f"{p}: no .dat frames")
        tok = pol_set.get(str(p)) or pol_from_name(p.name if p.is_dir() else p.stem) or pol_from_name(p.parent.name)
        if tok not in POL_ALIAS:
            raise ValueError(f"{p.name}: the files do not record the polarization and the name has no CL/CR/LH/LV; "
                             "set its polarization in the load list")
        scan = p.name if p.is_dir() else p.stem
        n0 = len(frames)
        for f in files:
            try:
                img, m = read_uview(f)
            except Exception as exc:  # truncated / corrupt frame
                bad.append(f"{scan}/{f.name}: {type(exc).__name__}")
                continue
            frames.append(img)
            metas.append(m)
            names.append(f.name)
            pol.append(POL_ALIAS[tok])
            scan_of.append(scan)
        if len(frames) == n0:
            raise ValueError(f"{p}: no readable frames")
        pol_by_scan[scan] = [tok]
        fov = metas[-1].get("fov_um")
        if fov is None:
            m = _FOV.search(scan)
            fov = float(m.group(1)) if m else None
        fov_by_scan[scan] = fov
    if len({f.shape for f in frames}) != 1:
        raise ValueError("acquisitions have different image sizes")
    fovs = {v for v in fov_by_scan.values() if v is not None}
    if len(fovs) > 1 and not allow_mixed_fov:
        raise ValueError(f"acquisitions have different fields of view {fov_by_scan} (um); "
                         "pixels do not correspond. Load matching acquisitions.")
    stack = np.stack(frames)
    n, ny, nx = stack.shape
    fov = next(iter(fovs)) if len(fovs) == 1 else None
    upp = um_per_px({**metas[0], "fov_um": fov}) if fov else None
    temps = np.array([m.get("sample_temp_K", np.nan) for m in metas], dtype=float)
    metadata: dict[str, Any] = {
        "frame_names": names,
        "pol": pol,
        "pol_source": {v: k for k, v in POL_ALIAS.items() if k != v},
        "scan_of_frame": scan_of,
        "scans": list(pol_by_scan),
        "pair_id": None,
        "source": ";".join(str(p) for p in paths),
        "loader": "uview",
        "beamline": "Elmitec UView (NSLS-II ESM)",
        "i0_normalized": False,
        "dropped_frames": 0,
        "missing_frames": bad,
        "fov_um": fov,
        "um_per_px": upp,
        "fov_values_um": sorted(fovs),
        "temperature_K": float(np.nanmean(temps)) if np.isfinite(temps).any() else None,
        "timestamps": [m.get("timestamp") for m in metas],
        "pol_by_scan": pol_by_scan,
        "pol_conflicts": [],
    }
    tensor = package_stack(stack, names, source=metadata["source"], loader=metadata["loader"], pol=pol,
                           dtype=np.float32)
    tensor.metadata.update(metadata)
    return tensor
