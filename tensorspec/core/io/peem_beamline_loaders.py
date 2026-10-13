"""One PEEM loader for the three beamline formats (no Qt).

``load_beamline(paths)`` detects the format from the paths / file contents and
returns the same PEEM ``(frame, y, x)`` TensorData for all of them, with one
polarization tag per frame in ``metadata["pol"]`` ("CP", "CM", "LH", "LV"):

=========  ===========================================  ==========================================
format     what to pass                                 polarization from
=========  ===========================================  ==========================================
ue49       ``X_0002.txt`` scan tables (or the image     file name pos / neg / _lh / _lv
           folder next to one)                          (BeamlineSettings PolTable as fallback)
uview      acquisition folders of UView ``.dat``        not in the files: ``pol`` (user) or the
           frames (``UKSOFT2001``), or single ``.dat``  folder name _CL_ / _CR_ / _LH_ / _LV_
p3b        acquisition folders of ``.P3B`` frames       EPU_POLARIZATION of every frame
                                                        (+0.9 -> CP, -0.9 -> CM)
=========  ===========================================  ==========================================

Names are not needed: the format is detected from the files (UView by its header, P3B by its
records, UE49 by the scan table next to its image folder), and ``pol`` ({path: CP / CM / LH /
LV}) sets the polarization of an item whose data do not record it (``detect_pol`` tells which).

Beamline-specific corrections run only where they exist: TPX3 cross / dead pixel /
flat field for UE49, I0 for UE49 (on by default) and ALS (off by default; used only
when it passes ``p3b_loaders.i0_reliable``). UView files carry no I0.

``metadata["default_convention"]`` is the same pcd sign for every beamline, the UE49
convention (neg-pos)/(neg+pos) = (CR-CL)/(CR+CL) = (CM-CP)/(CM+CP) (UE49 pos, BNL CL and
ALS +0.9 are CP); XLD is (LV-LH)/(LV+LH) (core/peem/dichroism).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tensorspec.core.data_models import TensorData
from tensorspec.core.io import p3b_loaders, ue49_loaders, uview_loaders

FORMATS = ("ue49", "uview", "p3b")
DEFAULT_CONVENTION = {"ue49": "ue49", "uview": "ue49", "p3b": "ue49"}  # UE49 sign everywhere


def detect_format(path: str | Path) -> tuple[str, Path]:
    """Return (format, normalized path) for one path given by the user."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"not found: {p}")
    if p.is_file():
        if p.suffix.lower() == ".txt" and p.with_suffix("").is_dir():
            return "ue49", p
        if p.suffix.upper() == ".P3B":
            return "p3b", p
        if p.suffix.lower() == ".dat" and _is_uview(p):
            return "uview", p
        raise ValueError(f"{p.name}: not a UE49 scan table, UView .dat or .P3B frame")
    txt = Path(str(p) + ".txt")
    if txt.is_file():
        return "ue49", txt  # UE49 image folder -> its scan table
    if p3b_loaders.frame_files(p):
        return "p3b", p
    dats = sorted(p.glob("*.dat"))
    if dats and _is_uview(dats[0]):
        return "uview", p
    raise ValueError(f"{p.name}: no UE49 scan table, UView .dat or .P3B frames found")


def partner_folder(path: str | Path) -> Path | None:
    """For a single-polarization UView folder (..._CL_...) the opposite one next to it
    (CL<->CR, LH<->LV), if it exists."""
    p = Path(path)
    for a, b in (("_CL_", "_CR_"), ("_CR_", "_CL_"), ("_LH_", "_LV_"), ("_LV_", "_LH_")):
        if a in p.name:
            cand = p.with_name(p.name.replace(a, b, 1))
            return cand if cand.is_dir() else None
    return None


POL_TAGS = ("CP", "CM", "LH", "LV")
_UVIEW_LABEL = {"CP": "CL", "CM": "CR", "LH": "LH", "LV": "LV"}


def detect_pol(path: str | Path) -> tuple[str | None, str]:
    """(tag CP / CM / LH / LV, or "per frame", or None; where it comes from) for one item, without
    loading images. None = the data do not tell: the user sets it (``load_beamline(pol=...)``)."""
    fmt, p = detect_format(path)
    if fmt == "p3b":
        return "per frame", "EPU polarization recorded in every frame"
    if fmt == "uview":
        tok = uview_loaders.pol_from_name(p.name if p.is_dir() else p.stem) or uview_loaders.pol_from_name(p.parent.name)
        return (uview_loaders.POL_ALIAS[tok], f"folder name ({tok})") if tok else (None, "not recorded in UView files")
    names = [n for n in ue49_loaders.read_scan_table(p).get("ImageNameTable", []) if n]
    tags = {ue49_loaders.pol_from_name(n) for n in names} - {"unknown"}
    if len(tags) == 1:
        return tags.pop(), "image names in the scan table"
    scan_pol = ue49_loaders.pol_from_settings(ue49_loaders.read_beamline_settings(p.with_suffix("") / "BeamlineSettings.txt"))
    if scan_pol and not tags:
        return scan_pol, "BeamlineSettings PolTable (can lag the helicity switch by one scan)"
    return None, "mixed / no polarization in the scan" if tags else "no polarization in the scan"


def is_beamline_source(path: str | Path) -> bool:
    try:
        detect_format(path)
        return True
    except ValueError:
        return False


def _is_uview(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(len(uview_loaders.MAGIC)) == uview_loaders.MAGIC


def load_beamline(
    paths: list[str | Path],
    *,
    i0_normalize: bool | None = None,
    allow_mixed_fov: bool = False,
    dead_map_path: str | Path | None = None,
    norm_image_path: str | Path | None = None,
    tpx3: bool = True,
    drop_bad_frames: bool = False,
    pol: dict[str, str] | None = None,
) -> TensorData:
    """Load UE49 / UView / P3B acquisitions (all of one format) into one PEEM dataset.

    i0_normalize: None = the beamline default (UE49 on, ALS off, UView has no I0).
    dead_map_path / norm_image_path / tpx3 / drop_bad_frames apply to UE49 only.
    pol: {path: CP / CM / LH / LV} for items whose data do not record the polarization (UView
    always, UE49 scans without tokens); ALS frames carry it, so it is not accepted there.
    """
    if not paths:
        raise ValueError("no files or folders given")
    found = [detect_format(p) for p in paths]
    fmts = {f for f, _ in found}
    if len(fmts) > 1:
        raise ValueError(f"mixed formats {sorted(fmts)}: load one beamline at a time")
    fmt = fmts.pop()
    norm = list(dict.fromkeys(p for _, p in found))
    pol_set: dict[str, str] = {}
    for given, (_, q) in zip(paths, found):
        tag = (pol or {}).get(str(given)) or (pol or {}).get(str(q))
        if tag:
            if tag not in POL_TAGS:
                raise ValueError(f"polarization must be one of {POL_TAGS}, got {tag!r}")
            pol_set[str(q)] = tag
    if pol_set and fmt == "p3b":
        raise ValueError("ALS .P3B frames record their polarization (EPU); it cannot be set by hand")
    if fmt == "ue49":
        tensor = ue49_loaders.load_scans(
            norm, dead_map_path=dead_map_path, norm_image_path=norm_image_path, tpx3=tpx3,
            i0_normalize=True if i0_normalize is None else i0_normalize,
            drop_bad_frames=drop_bad_frames, allow_mixed_fov=allow_mixed_fov, pol=pol_set)
    elif fmt == "uview":
        tensor = uview_loaders.load_uview_folders(
            norm, allow_mixed_fov=allow_mixed_fov, pol={k: _UVIEW_LABEL[v] for k, v in pol_set.items()})
    else:
        tensor = p3b_loaders.load_p3b_folders(norm, i0_normalize=bool(i0_normalize),
                                              allow_mixed_fov=allow_mixed_fov)
    md: dict[str, Any] = tensor.metadata
    md["format"] = fmt
    md["default_convention"] = DEFAULT_CONVENTION[fmt]
    return tensor
