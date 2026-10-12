"""Surface/Miller-index geometry helpers for SPR-KKR ARPES setup. Zero Qt.

Design doc §0 facts this module encodes:
- SPR-KKR's default `MILLER_HKL` frame is the *conventional* cell (Fortran
  `CREATE_3D_SURFACE` in spec_inputs.f rebuilds cubic-like axes for
  F/I/B/C-centered Bravais types before interpreting the indices).
- The `CRYS_VECS` keyword (note trailing S) in the TASK section switches
  interpretation to the raw ABAS frame (primitive vectors exactly as
  written in the .pot LATTICE `A(1..3)` lines) instead.

`hkl_to_abas_frame` converts a Miller index given in the conventional frame
to its raw-ABAS-frame equivalent so a caller can emit `CRYS_VECS` + the
converted index and get the identical physical surface. For a centered
Bravais lattice a literal per-component round can legitimately fail to
land on integers at scale 1 (classic case: cubic FCC conventional (0,0,1)
is Miller-forbidden by the face-centering parity rule and only clears at
(0,0,2) -> raw ABAS (1,1,0)); this function searches integer scales
1..max_scale, then reduces the result by its gcd, before giving up.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

BOHR_PER_ANGSTROM = 1.8897261254535
ANGSTROM_PER_BOHR = 1.0 / BOHR_PER_ANGSTROM


# ---------------------------------------------------------------------------
# Miller-index frame conversion
# ---------------------------------------------------------------------------


def hkl_to_abas_frame(
    conv_lattice_matrix,
    hkl: Tuple[float, float, float],
    abas_matrix,
    tol: float = 1e-3,
    max_scale: int = 8,
) -> Tuple[int, int, int]:
    """Conventional-frame Miller index -> raw-ABAS-frame Miller index.

    ``conv_lattice_matrix`` and ``abas_matrix`` must both be 3x3 with rows =
    direct lattice vectors, in the SAME absolute length units (both
    cartesian Angstrom, or both cartesian Bohr) -- the transform is
    scale-invariant only when the two share one absolute unit. A raw
    ``PotGeometry.abas`` is dimensionless (units of ALAT); multiply by
    ``alat_bohr`` (and convert Bohr<->Angstrom to match a pymatgen
    ``conv_lattice_matrix``) before calling this.

    ``G = hkl @ inv(conv).T`` is the (2*pi-free) reciprocal-space plane
    normal in cartesian coordinates; ``abas @ G`` projects it onto the raw
    direct lattice vectors. Exact (integer at scale 1) whenever the two
    bases are related by an integer/unimodular change of basis; otherwise
    only integer at some smallest scale k, tried 1..max_scale.
    """
    conv = np.asarray(conv_lattice_matrix, dtype=float)
    abas = np.asarray(abas_matrix, dtype=float)
    h = np.asarray(hkl, dtype=float)
    if conv.shape != (3, 3) or abas.shape != (3, 3) or h.shape != (3,):
        raise ValueError("conv_lattice_matrix/abas_matrix must be 3x3, hkl must have length 3")

    g_vec = h @ np.linalg.inv(conv).T
    raw = abas @ g_vec

    for k in range(1, max_scale + 1):
        scaled = raw * k
        rounded = np.round(scaled)
        if np.max(np.abs(scaled - rounded)) < tol:
            ints = rounded.astype(int)
            nz = [abs(int(x)) for x in ints if x != 0]
            if nz:
                d = nz[0]
                for x in nz[1:]:
                    d = gcd(d, x)
                if d > 1:
                    ints = ints // d
            return (int(ints[0]), int(ints[1]), int(ints[2]))

    raise ValueError(
        f"hkl {tuple(hkl)} has no integer ABAS-frame equivalent within "
        f"tol={tol} up to scale {max_scale}; raw={raw.tolist()}"
    )


# ---------------------------------------------------------------------------
# .pot parsing
# ---------------------------------------------------------------------------


@dataclass
class PotGeometry:
    alat_bohr: float
    abas: np.ndarray  # (3,3), rows = A(1..3), dimensionless (units of ALAT)
    sites: List[Tuple[int, np.ndarray, str]] = field(default_factory=list)
    # each site: (iq, xyz_cart (3,), units of ALAT, TYPES TXT string)


@dataclass
class Plane:
    proj_alat: float
    iqs: List[int] = field(default_factory=list)
    species: List[str] = field(default_factory=list)


_ASTERISK_RE = re.compile(r"^\*+$")
_A_VEC_RE = re.compile(r"^A\((\d)\)$")


def _num(tok: str) -> float:
    return float(tok.replace("D", "E").replace("d", "e"))


def _split_sections(text: str) -> Dict[str, List[str]]:
    """Split a .pot file into {SECTION_TITLE: [content lines]}.

    SPR-KKR .pot layout is a run of ``***...`` delimiter lines, each
    followed by a title line (e.g. ``LATTICE``) then content lines, then
    the next delimiter. Section titles used here (LATTICE, SITES,
    OCCUPATION, TYPES) are single tokens in both the plain-decimal
    (ase2sprkkr-written) and fixed-width scientific (kkrscf-written)
    variants seen in practice.
    """
    sections: Dict[str, List[str]] = {}
    current = None
    prev_was_asterisk = False
    for line in text.splitlines():
        stripped = line.strip()
        if _ASTERISK_RE.match(stripped):
            prev_was_asterisk = True
            continue
        if prev_was_asterisk:
            current = stripped
            sections.setdefault(current, [])
            prev_was_asterisk = False
            continue
        if current is not None:
            sections[current].append(line)
    return sections


def parse_pot_geometry(pot_path: PathLike) -> PotGeometry:
    """Parse LATTICE (ALAT, A(1..3)), SITES (CARTESIAN T/F + QBAS), OCCUPATION
    (IQ->ITOQ) and TYPES (IT->TXT) out of a SPR-KKR .pot file.

    Returns cartesian site positions in units of ALAT regardless of whether
    the file used CARTESIAN T (positions already cartesian) or CARTESIAN F
    (positions are fractional coordinates along A(1)/A(2)/A(3), converted
    here via ``frac @ abas``).
    """
    text = Path(pot_path).read_text(errors="ignore")
    sections = _split_sections(text)

    lattice_lines = sections.get("LATTICE", [])
    alat = None
    abas_rows: Dict[int, List[float]] = {}
    for line in lattice_lines:
        toks = line.split()
        if not toks:
            continue
        if toks[0] == "ALAT" and len(toks) >= 2:
            alat = _num(toks[1])
            continue
        m = _A_VEC_RE.match(toks[0])
        if m and len(toks) >= 4:
            abas_rows[int(m.group(1))] = [_num(t) for t in toks[1:4]]
    if alat is None or set(abas_rows) != {1, 2, 3}:
        raise ValueError(f"{pot_path}: could not parse LATTICE ALAT/A(1..3)")
    abas = np.array([abas_rows[1], abas_rows[2], abas_rows[3]], dtype=float)

    sites_lines = sections.get("SITES", [])
    cartesian = True
    raw_sites: List[Tuple[int, np.ndarray]] = []
    for line in sites_lines:
        toks = line.split()
        if not toks:
            continue
        if toks[0] == "CARTESIAN":
            cartesian = len(toks) >= 2 and toks[1].upper().startswith("T")
            continue
        if toks[0] in ("BASSCALE", "IQ"):
            continue
        try:
            iq = int(toks[0])
            xyz = np.array([_num(t) for t in toks[1:4]], dtype=float)
        except (ValueError, IndexError):
            continue
        raw_sites.append((iq, xyz))
    if not raw_sites:
        raise ValueError(f"{pot_path}: no SITES rows parsed")

    occ_lines = sections.get("OCCUPATION", [])
    itoq_of: Dict[int, int] = {}
    for line in occ_lines:
        toks = line.split()
        if not toks or toks[0] == "IQ":
            continue
        try:
            iq = int(toks[0])
            itoq = int(toks[4])
        except (ValueError, IndexError):
            continue
        itoq_of[iq] = itoq

    types_lines = sections.get("TYPES", [])
    txt_of: Dict[int, str] = {}
    for line in types_lines:
        toks = line.split()
        if not toks or toks[0] == "IT":
            continue
        try:
            it = int(toks[0])
            txt = toks[1]
        except (ValueError, IndexError):
            continue
        # ZT (atomic number) = 3rd column; empty spheres have ZT=0. SPR-KKR
        # renames types in pot_new (e.g. 'Vc' -> 'Vc_4'), so never rely on TXT alone.
        try:
            zt = int(float(toks[2]))
        except (ValueError, IndexError):
            zt = -1
        if zt == 0 and not txt.startswith("Vc"):
            txt = "Vc_" + txt
        txt_of[it] = txt

    sites: List[Tuple[int, np.ndarray, str]] = []
    for iq, xyz in raw_sites:
        if not cartesian:
            xyz = xyz[0] * abas[0] + xyz[1] * abas[1] + xyz[2] * abas[2]
        itoq = itoq_of.get(iq)
        txt = txt_of.get(itoq, "?") if itoq is not None else "?"
        sites.append((iq, xyz, txt))
    sites.sort(key=lambda s: s[0])

    return PotGeometry(alat_bohr=alat, abas=abas, sites=sites)


# ---------------------------------------------------------------------------
# Layer stacking along a surface normal
# ---------------------------------------------------------------------------


def _normal_hat(abas: np.ndarray, hkl_abas: Tuple[int, int, int]) -> np.ndarray:
    h = np.asarray(hkl_abas, dtype=float)
    g_vec = h @ np.linalg.inv(abas).T  # same construction as hkl_to_abas_frame's G
    norm = np.linalg.norm(g_vec)
    if norm < 1e-12:
        raise ValueError(f"hkl_abas {tuple(hkl_abas)} gives a zero normal vector")
    return g_vec / norm


def layer_stack(pot_geom: PotGeometry, hkl_abas: Tuple[int, int, int], tol: float = 0.02) -> List[Plane]:
    """Group sites into atomic planes along the ``hkl_abas`` surface normal.

    Sites are projected onto the unit normal (ascending order = bottom to
    top of the cell along +hkl_abas); consecutive sites are merged into one
    plane while the running projection stays within ``tol`` (alat units) of
    the group's (updated) mean projection.
    """
    n_hat = _normal_hat(pot_geom.abas, hkl_abas)

    projs = sorted(
        ((iq, float(np.dot(xyz, n_hat)), txt) for iq, xyz, txt in pot_geom.sites),
        key=lambda t: t[1],
    )

    planes: List[Plane] = []
    for iq, p, txt in projs:
        if planes and abs(p - planes[-1].proj_alat) < tol:
            plane = planes[-1]
            plane.iqs.append(iq)
            plane.species.append(txt)
            n = len(plane.iqs)
            plane.proj_alat = (plane.proj_alat * (n - 1) + p) / n
        else:
            planes.append(Plane(proj_alat=p, iqs=[iq], species=[txt]))
    return planes


def pick_surface_site(
    pot_geom: PotGeometry,
    hkl_abas: Tuple[int, int, int],
    exclude: Tuple[str, ...] = ("Vc",),
    tol: float = 0.02,
) -> int:
    """IQ of the topmost (max projection along ``hkl_abas``) site whose TYPES
    TXT is not in ``exclude`` (empty spheres excluded by default)."""
    planes = layer_stack(pot_geom, hkl_abas, tol=tol)
    for plane in reversed(planes):
        for iq, txt in zip(plane.iqs, plane.species):
            # prefix match: pot_new renames 'Vc' -> 'Vc_4', 'Te' -> 'Te_2', ...
            if not any(txt == ex or txt.startswith(ex + "_") for ex in exclude):
                return iq
    raise ValueError(f"no site outside {exclude} found along hkl_abas={hkl_abas}")


def atoms_per_plane_max(pot_geom: PotGeometry, hkl_abas: Tuple[int, int, int], tol: float = 0.02) -> int:
    """Cost hint: size (site count) of the densest atomic plane."""
    planes = layer_stack(pot_geom, hkl_abas, tol=tol)
    return max((len(p.iqs) for p in planes), default=0)
