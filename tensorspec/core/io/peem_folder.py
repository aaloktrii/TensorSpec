"""PEEM "Load Folder": recognize what a chosen folder holds and how to load it (no Qt).

    inspect_folder(path)   {"kind", "format", "paths", "ignored", "summary"}:
        beamline   the folder is one acquisition: UE49 scan image folder (its .txt next to it),
                   Elmitec UView .dat (BNL ESM) or ALS PEEM-3 .P3B  -> peem_beamline_loaders
        tif        .tif / .tiff frames                               -> peem_loaders (TIF sequence)
        images     .png / .jpg / .jpeg frames                        -> load_image_folder
        container  a folder of acquisitions (sub-folders, scan tables)  -> pick, then beamline
    Other files (.rtf, .ipf, .csv, notes ...) are not frames; they are listed in "ignored".
    load_image_folder(path)  sorted PNG / JPEG frames -> PEEM TensorData (like the TIF sequence)
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tensorspec.core.data_models import TensorData
from tensorspec.core.io import peem_beamline_loaders
from tensorspec.core.io.peem_loaders import _auto_attach_csv, package_stack

TIF_EXT = (".tif", ".tiff")
IMAGE_EXT = (".png", ".jpg", ".jpeg")


def _natural(p: Path):
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", p.name)]


def _files(folder: Path, exts: tuple[str, ...]) -> list[Path]:
    return sorted((f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in exts
                   and not f.name.startswith(".")), key=_natural)


def inspect_folder(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.is_dir():
        raise ValueError(f"not a folder: {p}")
    files = [f for f in p.iterdir() if f.is_file() and not f.name.startswith(".")]
    out: dict[str, Any] = {"path": str(p), "format": None, "paths": [str(p)]}
    try:
        fmt, _ = peem_beamline_loaders.detect_format(p)
        out.update(kind="beamline", format=fmt)
    except ValueError:
        tifs, imgs = _files(p, TIF_EXT), _files(p, IMAGE_EXT)
        if tifs:
            out.update(kind="tif", format="tif", n_frames=len(tifs))
        elif imgs:
            out.update(kind="images", format=imgs[0].suffix.lower().lstrip("."), n_frames=len(imgs))
        else:
            found = peem_beamline_loaders.find_acquisitions(p)
            if not found:
                exts = Counter(f.suffix.lower() or "(no extension)" for f in files)
                raise ValueError(f"{p.name}: no PEEM data found (.tif / .tiff, .png / .jpg, UE49 .txt scan, "
                                 f"UView .dat, ALS .P3B). Files here: "
                                 + (", ".join(f"{n} {e}" for e, n in exts.most_common()) or "none"))
            out.update(kind="container", paths=[str(q) for _, q in found], formats=[f for f, _ in found])
    used = set()
    if out["kind"] == "tif":
        used = set(TIF_EXT)
    elif out["kind"] == "images":
        used = {"." + out["format"]} | ({".jpeg", ".jpg"} if out["format"] in ("jpg", "jpeg") else set())
    elif out["kind"] in ("beamline", "container"):  # frames, scan tables and their .dat logs
        used = {".dat", ".p3b", ".tif", ".tiff", ".txt"}
    ignored = Counter(f.suffix.lower() or "(no extension)" for f in files if f.suffix.lower() not in used)
    if out["kind"] in ("tif", "images"):
        ignored.pop(".csv", None)  # a beamline CSV next to TIF / image frames is attached automatically
    out["ignored"] = dict(ignored)
    return out


def load_image_folder(path: str | Path) -> TensorData:
    """PNG / JPEG frames in natural name order -> (n, y, x) PEEM stack (grayscale; colour images are
    converted to luminance). A beamline CSV in the folder is attached as for TIF sequences."""
    from PIL import Image

    p = Path(path).expanduser()
    paths = _files(p, IMAGE_EXT)
    if not paths:
        raise ValueError(f"No PNG / JPEG files found in {p}")
    frames, ref = [], None
    for f in paths:
        with Image.open(f) as im:
            mode = im.mode
            a = np.asarray(im.convert("F") if mode not in ("I;16", "I;16B", "I") else im, dtype=np.float32)
        if ref is None:
            ref = a.shape
        elif a.shape != ref:
            raise ValueError(f"image sizes differ in {p.name}: {f.name} {a.shape} vs {ref}")
        frames.append(a)
    return package_stack(np.stack(frames), [f.stem for f in paths], source=str(p), loader="image_sequence",
                         csv_meta=_auto_attach_csv(p, preferred_stem=p.name), dtype=np.float32)
