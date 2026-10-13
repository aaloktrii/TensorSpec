"""One XPEEM loader for the three beamline formats (no Qt).

``load_xpeem(paths)`` detects the format from the paths / file contents and
returns the same PEEM ``(frame, y, x)`` TensorData for all of them, with one
polarization tag per frame in ``metadata["pol"]`` ("CP", "CM", "LH", "LV"):

=========  ===========================================  ==========================================
format     what to pass                                 polarization from
=========  ===========================================  ==========================================
ue49       ``X_0002.txt`` scan tables (or the image     file name pos / neg / _lh / _lv
           folder next to one)                          (BeamlineSettings PolTable as fallback)
uview      acquisition folders of UView ``.dat``        folder name _CL_ / _CR_ / _LH_ / _LV_
           frames (``UKSOFT2001``), or single ``.dat``  (CL -> CP, CR -> CM)
p3b        acquisition folders of ``*#*.P3B`` frames    EPU_POLARIZATION of every frame
                                                        (+0.9 -> CP, -0.9 -> CM)
=========  ===========================================  ==========================================

Beamline-specific corrections run only where they exist: TPX3 cross / dead pixel /
flat field for UE49, I0 for UE49 (on by default) and ALS (off by default; used only
when it passes ``p3b_loaders.i0_reliable``). UView files carry no I0.

``metadata["default_convention"]`` is the pcd sign each beamline's own pipeline uses
(UE49: (CM-CP)/(CM+CP); BNL: (CL-CR)/(CL+CR); ALS: (I+0.9 - I-0.9)/sum), so maps
stay comparable with earlier results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tensorspec.core.data_models import TensorData
from tensorspec.core.io import p3b_loaders, ue49_loaders, uview_loaders

FORMATS = ("ue49", "uview", "p3b")
DEFAULT_CONVENTION = {"ue49": "ue49", "uview": "cp_minus_cm", "p3b": "cp_minus_cm"}


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


def is_xpeem_source(path: str | Path) -> bool:
    try:
        detect_format(path)
        return True
    except ValueError:
        return False


def _is_uview(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(len(uview_loaders.MAGIC)) == uview_loaders.MAGIC


def load_xpeem(
    paths: list[str | Path],
    *,
    i0_normalize: bool | None = None,
    allow_mixed_fov: bool = False,
    dead_map_path: str | Path | None = None,
    norm_image_path: str | Path | None = None,
    tpx3: bool = True,
    drop_bad_frames: bool = False,
) -> TensorData:
    """Load UE49 / UView / P3B acquisitions (all of one format) into one PEEM dataset.

    i0_normalize: None = the beamline default (UE49 on, ALS off, UView has no I0).
    dead_map_path / norm_image_path / tpx3 / drop_bad_frames apply to UE49 only.
    """
    if not paths:
        raise ValueError("no files or folders given")
    found = [detect_format(p) for p in paths]
    fmts = {f for f, _ in found}
    if len(fmts) > 1:
        raise ValueError(f"mixed formats {sorted(fmts)}: load one beamline at a time")
    fmt = fmts.pop()
    norm = list(dict.fromkeys(p for _, p in found))
    if fmt == "ue49":
        tensor = ue49_loaders.load_scans(
            norm, dead_map_path=dead_map_path, norm_image_path=norm_image_path, tpx3=tpx3,
            i0_normalize=True if i0_normalize is None else i0_normalize,
            drop_bad_frames=drop_bad_frames, allow_mixed_fov=allow_mixed_fov)
    elif fmt == "uview":
        tensor = uview_loaders.load_uview_folders(norm, allow_mixed_fov=allow_mixed_fov)
    else:
        tensor = p3b_loaders.load_p3b_folders(norm, i0_normalize=bool(i0_normalize),
                                              allow_mixed_fov=allow_mixed_fov)
    md: dict[str, Any] = tensor.metadata
    md["format"] = fmt
    md["default_convention"] = DEFAULT_CONVENTION[fmt]
    return tensor
