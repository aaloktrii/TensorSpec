"""Raman readers (no Qt): LabRAM/LabSpec map exports (.txt, spectra or camera-image layout),
single-spot two-column spectra and LabSpec .h5 spectra, one-file-per-spot map folders
(stage X/Y from ``__X_<x>__Y_<y>__`` in each filename), and load_raman() -> TensorData for the
workspace DataTree (/raw).

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (file parsing, single spectra and
spot folders) and ``raman_peak_fit.py`` (read_spectrum)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tensorspec.core.data_models import TensorData

# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def _to_float(token: str) -> float:
    """Tolerate European decimal commas as well as plain floats."""
    token = token.strip()
    try:
        return float(token)
    except ValueError:
        return float(token.replace(",", "."))


def _split(line: str) -> List[str]:
    parts = line.split("\t")
    if len([p for p in parts if p.strip()]) < 3:
        parts = line.split()
    return parts


def parse_header_meta(header: List[str]) -> dict:
    """Pull the LabSpec-style ``#Key=<tab>value`` metadata out of the header."""
    meta = {}
    for line in header:
        body = line.lstrip("#").strip()
        if "=" not in body:
            continue
        k, v = body.split("=", 1)
        meta[k.strip()] = v.strip()
    return meta


def axis_roles(header: List[str]) -> Dict[int, str]:
    """Read the ``#AxisType[n]`` declarations.

    These files list axes FASTEST-VARYING FIRST after AxisType[0] (the
    intensity channel):

        AxisType[1]  the axis running along row 0
        AxisType[2]  the fastest-varying leading column
        AxisType[3]  the next slower leading column, and so on

    A Raman map declares Intens / Spectr / X / Y, so the spectral axis runs
    along the row, the *second* leading column is X and the *first* is Y. A
    camera file declares '' / X / Y: X runs along the row and the single
    leading column is Y. Both readings agree, which is what makes the
    convention trustworthy.
    """
    roles = {}
    for line in header:
        body = line.lstrip("#").strip()
        if not body.startswith("AxisType[") or "]" not in body:
            continue
        try:
            idx = int(body[len("AxisType["):body.index("]")])
        except ValueError:
            continue
        val = body.split("=", 1)[1].strip() if "=" in body else ""
        roles[idx] = val
    return roles


@dataclass
class MapFile:
    """A parsed export, in either of the two layouts these files use.

    layout "spectra"  one spectrum per row: two leading coordinate columns
                      followed by the intensities, with the wavenumbers on
                      row 0. This is the Raman map.

    layout "image"    a plain 2D image: row 0 carries the fast spatial axis
                      and every later row starts with its own slow-axis
                      coordinate followed by one value per column. This is
                      the camera file, and it is NOT the same layout as the
                      Raman map despite what the spec assumes.
    """

    axis: np.ndarray               # (n_axis,) wavenumbers, or X positions
    x: np.ndarray                  # (n_points,) X in micrometres
    y: np.ndarray                  # (n_points,) Y in micrometres
    data: np.ndarray               # (n_points, n_axis) intensities
    path: str = ""
    header: List[str] = field(default_factory=list)
    layout: str = "spectra"
    image: Optional[np.ndarray] = None      # (ny, nx), image layout only
    swapped: bool = False                   # X/Y taken from the headers
    raw0: Optional[np.ndarray] = None       # coordinate column 0, as in the file
    raw1: Optional[np.ndarray] = None       # coordinate column 1, as in the file

    ux: np.ndarray = field(init=False)
    uy: np.ndarray = field(init=False)
    index_map: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.layout == "image":
            self.ux = np.asarray(self.axis, dtype=float)
            self.uy = np.asarray(self.y, dtype=float)
            self.index_map = np.zeros((self.uy.size, self.ux.size), dtype=int)
            return
        xr = np.round(self.x, 4)
        yr = np.round(self.y, 4)
        self.ux = np.unique(xr)
        self.uy = np.unique(yr)
        idx = np.full((len(self.uy), len(self.ux)), -1, dtype=int)
        cols = np.searchsorted(self.ux, xr)
        rows = np.searchsorted(self.uy, yr)
        for i, (r, c) in enumerate(zip(rows, cols)):
            idx[r, c] = i
        self.index_map = idx

    # -- convenience --------------------------------------------------------
    @property
    def meta(self) -> dict:
        return parse_header_meta(self.header)

    @property
    def n_points(self) -> int:
        return 0 if self.data is None else int(self.data.shape[0])

    @property
    def shape(self) -> Tuple[int, int]:
        return self.index_map.shape

    @property
    def is_optical(self) -> bool:
        return self.layout == "image" or any("camera" in h.lower()
                                             for h in self.header)

    @property
    def spectral_resolution(self) -> Optional[float]:
        for k, v in self.meta.items():
            if k.lower().startswith("spectral res"):
                try:
                    return float(v.replace(",", "."))
                except ValueError:
                    return None
        return None

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """(left, right, bottom, top) in micrometres, pixel-edge aligned."""
        dx = float(np.median(np.diff(self.ux))) if len(self.ux) > 1 else 1.0
        dy = float(np.median(np.diff(self.uy))) if len(self.uy) > 1 else 1.0
        return (float(self.ux[0] - dx / 2), float(self.ux[-1] + dx / 2),
                float(self.uy[0] - dy / 2), float(self.uy[-1] + dy / 2))

    def spectrum(self, row: int, col: int) -> Optional[np.ndarray]:
        if self.layout != "spectra":
            return None
        i = self.index_map[row, col]
        return None if i < 0 else self.data[i]

    def greyscale(self) -> np.ndarray:
        """A (ny, nx) image: the camera frame, or the mean over channels."""
        if self.layout == "image":
            return self.image
        vals = np.nanmean(self.data, axis=1)
        img = np.full(self.shape, np.nan)
        ny, nx = self.shape
        for r in range(ny):
            for c in range(nx):
                i = self.index_map[r, c]
                if i >= 0:
                    img[r, c] = vals[i]
        return img


def _read_lines(path: str) -> Tuple[List[str], List[str]]:
    header, body = [], []
    with open(path, "r", encoding="ISO-8859-1", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            (header if line.lstrip().startswith("#") else body).append(
                line.strip() if line.lstrip().startswith("#") else line)
    return header, body


def parse_map_file(path: str) -> MapFile:
    """Parse either export layout, detected from the axis row.

    The number of blank fields before the first number on the axis row is the
    number of leading coordinate columns each data row carries: two for a
    Raman map (X and Y per spectrum), one for a camera image (the slow axis
    coordinate per row). Guessing wrong silently transposes the whole map, so
    this is measured rather than assumed.
    """
    header, body = _read_lines(path)
    if len(body) < 2:
        raise ValueError("File contains an axis row but no data rows.")

    tabbed = "\t" in body[0]
    def fields(line):
        return line.split("\t") if tabbed else line.split()

    axis_parts = fields(body[0])
    axis_vals = [t for t in axis_parts if t.strip()]
    axis = np.array([_to_float(t) for t in axis_vals], dtype=float)
    if axis.size < 1:
        raise ValueError("Could not read an axis from the first data row.")

    # How many coordinate columns precede the values on each data row? Two
    # for a Raman map (X and Y per spectrum), one for a camera image. Count
    # the blank leading fields where the file is tab-delimited, but confirm
    # it against the width difference between a data row and the axis row,
    # which also works for whitespace-delimited exports where consecutive
    # blank columns collapse and cannot be counted.
    n_blank = 0
    while n_blank < len(axis_parts) and not axis_parts[n_blank].strip():
        n_blank += 1
    n_lead = None
    for line in body[1:]:
        width = len([t for t in fields(line) if t.strip()])
        if width > axis.size:
            n_lead = width - axis.size
            break
    if n_lead not in (1, 2):
        n_lead = n_blank if n_blank in (1, 2) else 2

    roles = axis_roles(header)

    if n_lead == 1:
        return _parse_image(path, header, body, axis, roles)
    return _parse_spectra(path, header, body, axis, roles)


def _parse_spectra(path, header, body, axis, roles) -> MapFile:
    n_axis = axis.size
    xs, ys, vals = [], [], []
    for parts in (_split(line) for line in body[1:]):
        if len(parts) < 3:
            continue
        try:
            a = _to_float(parts[0])
            b = _to_float(parts[1])
        except ValueError:
            continue
        inten = []
        for t in parts[2:]:
            if not t.strip():
                continue
            try:
                inten.append(_to_float(t))
            except ValueError:
                inten.append(np.nan)
        if inten:
            xs.append(a)
            ys.append(b)
            vals.append(inten)
    if not vals:
        raise ValueError("No data rows could be parsed.")

    width = min(n_axis, min(len(v) for v in vals))
    axis = axis[:width]
    data = np.array([v[:width] for v in vals], dtype=float)
    col0 = np.array(xs)
    col1 = np.array(ys)

    # Which leading column is X? The headers list axes fastest-first, so
    # AxisType[2] names the fast column (col1) and AxisType[3] the slow one
    # (col0). Absent headers, fall back to the spec's col0=X, col1=Y.
    def _axis_kind(label: str) -> str:
        """'X'/'XA'/'X_ABS' etc. all mean the X axis; same for Y. Matched by
        prefix, not exact equality, so an instrument-specific suffix (this
        file's 'XA'/'YA' for absolute stage position, as opposed to the
        earlier file's plain 'X'/'Y') is still recognised instead of quietly
        falling through to a guessed default."""
        u = label.strip().upper()
        if u.startswith("X"):
            return "X"
        if u.startswith("Y"):
            return "Y"
        return ""

    role2, role3 = _axis_kind(roles.get(2, "")), _axis_kind(roles.get(3, ""))
    swapped = False
    if role2 == "X" and role3 == "Y":
        x, y = col1, col0
        swapped = True
    elif role2 == "Y" and role3 == "X":
        x, y = col0, col1
    elif role2 or role3:
        # One is recognisable but they do not form a clean X/Y pair - safer
        # to say so than to guess.
        raise ValueError(
            "Cannot determine the X/Y column order: AxisType[2]=%r, "
            "AxisType[3]=%r. Expected one to start with X and the other "
            "with Y." % (roles.get(2, ""), roles.get(3, "")))
    else:
        # No AxisType header at all - only place we fall back to the
        # spec's stated order, and it is worth knowing that happened.
        x, y = col0, col1

    if axis.size > 1 and axis[0] > axis[-1]:
        order = np.argsort(axis)
        axis = axis[order]
        data = data[:, order]

    return MapFile(axis=axis, x=x, y=y, data=data, path=path, header=header,
                   raw0=col0.copy(), raw1=col1.copy(),
                   layout="spectra", swapped=swapped)


def _parse_image(path, header, body, axis, roles) -> MapFile:
    """Row 0 is the fast spatial axis; each later row is <coord> then values."""
    n_cols = axis.size
    ys = np.empty(len(body) - 1, dtype=float)
    img = np.empty((len(body) - 1, n_cols), dtype=np.float32)
    keep = 0
    for line in body[1:]:
        v = np.fromstring(line, sep="\t")
        if v.size < 2:
            v = np.fromstring(line, sep=" ")
        if v.size < 2:
            continue
        ys[keep] = v[0]
        row = v[1:]
        if row.size < n_cols:
            img[keep, :row.size] = row
            img[keep, row.size:] = np.nan
        else:
            img[keep] = row[:n_cols]
        keep += 1
    if keep == 0:
        raise ValueError("No image rows could be parsed.")
    ys = ys[:keep]
    img = img[:keep]

    xs = axis
    if xs.size > 1 and xs[0] > xs[-1]:
        order = np.argsort(xs)
        xs = xs[order]
        img = img[:, order]
    if ys.size > 1 and ys[0] > ys[-1]:          # camera rows run top-to-bottom
        ys = ys[::-1]
        img = img[::-1]

    # The row axis is X and the leading column Y unless the headers say
    # otherwise; transpose rather than mislabel if they do.
    if roles.get(1, "").upper() == "Y" and roles.get(2, "").upper() == "X":
        xs, ys, img = ys, xs, img.T

    return MapFile(axis=xs, x=xs, y=ys, data=np.empty((0, 0)), path=path,
                   header=header, layout="image", image=img)


# ---------------------------------------------------------------------------
# Single spectra and one-file-per-spot map folders
# ---------------------------------------------------------------------------

@dataclass
class Spectrum:
    """One spectrum that is not part of a map: a single-spot LabSpec export,
    a LabSpec .h5 spectrum, or a spectrum taken out of a loaded map."""
    wn: np.ndarray                 # ascending wavenumbers
    inten: np.ndarray
    label: str
    path: str = ""
    meta: dict = field(default_factory=dict)


def is_single_spectrum(path: str) -> bool:
    """True for a two-column (wavenumber, intensity) text export. A map's
    first data row is the wavenumber axis and has far more than two fields,
    so this never matches one."""
    if path.lower().endswith(".h5"):
        return True
    try:
        _, body = _read_lines(path)
    except OSError:
        return False
    rows = body[:50]
    return len(body) > 10 and all(
        len([t for t in _split(r) if t.strip()]) == 2 for r in rows)


def read_single_spectrum(path: str) -> Spectrum:
    if path.lower().endswith(".h5"):
        try:
            import h5py
        except ImportError:
            raise ValueError("Reading .h5 files needs h5py:  pip install h5py")
        with h5py.File(path, "r") as h:
            if "Datas/Data1" not in h:
                raise ValueError("No Datas/Data1 in this file; only LabSpec "
                                 ".h5 exports are supported.")
            ds = h["Datas/Data1"]
            if ds.ndim != 1:
                raise ValueError("This .h5 holds a %d-D dataset; only single "
                                 "spectra are supported." % ds.ndim)
            wn = np.asarray(ds.attrs["Axis1"], dtype=float)
            inten = ds[:].astype(float)
            meta = {}
            if "Datas/DataInfo1" in h:
                for r in h["Datas/DataInfo1"][:]:
                    meta[r[0].decode(errors="replace")] = r[2].decode(errors="replace")
    else:
        header, body = _read_lines(path)
        a = np.array([[_to_float(t) for t in _split(r) if t.strip()][:2]
                      for r in body], dtype=float)
        wn, inten = a[:, 0], a[:, 1]
        meta = parse_header_meta(header)
    o = np.argsort(wn)
    return Spectrum(wn=wn[o], inten=inten[o], label=short_label(path),
                    path=path, meta=meta)


def short_label(path: str) -> str:
    """A compact default label: the first three '_' parts of the file or
    folder name, with the parent folder added when that is very short
    (e.g. '007_01/5')."""
    path = path.rstrip(os.sep)
    stem = os.path.splitext(os.path.basename(path))[0]
    lab = "_".join(stem.split("_")[:3])[:28]
    if len(lab) <= 4:
        lab = os.path.basename(os.path.dirname(path)) + "/" + lab
    return lab


_SPOT_RE = re.compile(r"__X_([-+\d.]+)__Y_([-+\d.]+)__")


def read_spot_folder(folder: str) -> MapFile:
    """A map exported as one two-column file per spot, the stage position in
    each filename (``..._<n>__X_<x>__Y_<y>__Time_<t>.txt``).

    raw0 holds −X so that the instrument default (X = −col0, Y = col1)
    places the spots exactly at their filename coordinates.
    """
    xs, ys, rows, axis = [], [], [], None
    for name in sorted(os.listdir(folder)):
        m = _SPOT_RE.search(name)
        if not m or not name.lower().endswith((".txt", ".dat", ".csv")):
            continue
        a = np.loadtxt(os.path.join(folder, name), comments="#")
        a = a[np.argsort(a[:, 0])]
        if axis is None:
            axis = a[:, 0]
        elif a.shape[0] != axis.size or not np.allclose(a[:, 0], axis, atol=1e-3):
            raise ValueError("%s has a different wavenumber axis from the "
                             "other spots." % name)
        xs.append(float(m.group(1)))
        ys.append(float(m.group(2)))
        rows.append(a[:, 1])
    if not rows:
        raise ValueError("No files named like ..._X_<x>__Y_<y>__... in this folder.")
    x, y = np.array(xs), np.array(ys)
    return MapFile(axis=axis, x=x, y=y, data=np.array(rows), path=folder,
                   header=[], raw0=-x, raw1=y.copy(), layout="spectra")


SPECTRUM_EXT = (".txt", ".dat", ".csv")


def read_spectrum(path):
    rows = []
    with open(path, encoding="ISO-8859-1") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.replace(",", ".").split()
            if len(p) == 2:
                rows.append((float(p[0]), float(p[1])))
    a = np.array(rows)
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1]


# ---------------------------------------------------------------------------
# TensorSpec workspace entry
# ---------------------------------------------------------------------------

def map_to_tensor(mf: MapFile) -> TensorData:
    """A parsed map (or camera image) as TensorData for the workspace /raw node.

    Spectra layout: intensity on the file's own (Y, X) grid, NaN where no spot was measured,
    in the coordinates as read (the GUI's orientation/offset transform is a display choice
    and stays in the analysis config). Image layout: the (Y, X) camera frame."""
    meta = {"source_path": mf.path, "layout": mf.layout, "swapped_xy": bool(mf.swapped)}
    meta.update({"header_" + k: v for k, v in mf.meta.items()})
    if mf.spectral_resolution:
        meta["spectral_resolution"] = float(mf.spectral_resolution)
    if mf.layout == "image":
        return TensorData(value=np.asarray(mf.image, dtype=float), axes=[mf.uy, mf.ux],
                          labels=["Y", "X"], units=["µm", "µm"],
                          data_type="Raman optical image", metadata=meta)
    ny, nx = mf.shape
    cube = np.full((ny, nx, mf.axis.size), np.nan)
    ok = mf.index_map >= 0
    cube[ok] = mf.data[mf.index_map[ok]]
    return TensorData(value=cube, axes=[mf.uy, mf.ux, mf.axis],
                      labels=["Y", "X", "Raman shift"], units=["µm", "µm", "cm⁻¹"],
                      data_type="Raman map", metadata=meta)


def spectrum_to_tensor(sp: Spectrum) -> TensorData:
    """A single spectrum as 1-D TensorData."""
    meta = {"source_path": sp.path, "label": sp.label}
    meta.update({"header_" + str(k): str(v) for k, v in sp.meta.items()})
    return TensorData(value=np.asarray(sp.inten, dtype=float), axes=[sp.wn],
                      labels=["Raman shift"], units=["cm⁻¹"],
                      data_type="Raman spectrum", metadata=meta)


def load_raman(path: str):
    """Read any supported Raman file or spot folder.

    Returns (kind, obj): ("spectrum", Spectrum) for a single spectrum, ("map", MapFile) for a
    map or spot folder, ("image", MapFile) for a camera image."""
    if not os.path.isdir(path) and is_single_spectrum(path):
        return "spectrum", read_single_spectrum(path)
    mf = read_spot_folder(path) if os.path.isdir(path) else parse_map_file(path)
    return ("image" if mf.layout == "image" else "map"), mf


def to_tensor(kind: str, obj) -> TensorData:
    """TensorData for the result of load_raman()."""
    return spectrum_to_tensor(obj) if kind == "spectrum" else map_to_tensor(obj)

