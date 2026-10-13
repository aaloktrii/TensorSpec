"""Detect QE Fermi energy beside a Wannier/QE work directory."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple


def _parse_fermi_line(line: str) -> Optional[float]:
    if "the Fermi energy is" in line:
        try:
            return float(line.split("is")[1].split("ev")[0].strip())
        except (IndexError, ValueError):
            return None
    return None


def _parse_fermi_energy_txt(path: str) -> Optional[float]:
    try:
        with open(path, "r") as f:
            for line in f:
                parsed = _parse_fermi_line(line)
                if parsed is not None:
                    return parsed
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    return float(stripped.split()[0])
                except ValueError:
                    continue
    except OSError:
        return None
    return None


def _parse_qe_out_fermi(path: str) -> Optional[float]:
    try:
        with open(path, "r") as f:
            for line in f:
                parsed = _parse_fermi_line(line)
                if parsed is not None:
                    return parsed
    except OSError:
        return None
    return None


def detect_qe_fermi_eV(work_dir: str) -> Tuple[float, str]:
    """
    Auto-detect QE Fermi (eV) from a Wannier/QE folder.

    Order: nscf.out → scf.out → FERMI_ENERGY.txt → (0.0, \"none\").
    """
    if not work_dir:
        return 0.0, "none"

    nscf = os.path.join(work_dir, "nscf.out")
    if os.path.isfile(nscf):
        val = _parse_qe_out_fermi(nscf)
        if val is not None:
            return float(val), "nscf.out"

    scf = os.path.join(work_dir, "scf.out")
    if os.path.isfile(scf):
        val = _parse_qe_out_fermi(scf)
        if val is not None:
            return float(val), "scf.out"

    fermi_txt = os.path.join(work_dir, "FERMI_ENERGY.txt")
    if os.path.isfile(fermi_txt):
        val = _parse_fermi_energy_txt(fermi_txt)
        if val is not None:
            return float(val), "FERMI_ENERGY.txt"

    return 0.0, "none"


def _parse_w90_labelinfo(path: str) -> List[Tuple[str, int]]:
    """Return [(label, 1-based band.dat row index), ...]."""
    out: List[Tuple[str, int]] = []
    try:
        with open(path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        out.append((parts[0], int(parts[1])))
                    except ValueError:
                        continue
    except OSError:
        return []
    return out


def _parse_w90_band_blocks(path: str) -> List[List[float]]:
    """One list of energies per Wannier band (absolute eV)."""
    blocks: List[List[float]] = []
    ys: List[float] = []
    try:
        with open(path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        ys.append(float(parts[1]))
                    except ValueError:
                        continue
                elif not parts and ys:
                    blocks.append(ys)
                    ys = []
    except OSError:
        return []
    if ys:
        blocks.append(ys)
    return blocks


def detect_w90_touch_fermi_eV(
    work_dir: str,
    gap_max: float = 0.15,
    near_qe_window: float = 2.0,
) -> Tuple[Optional[float], str]:
    """
    Midgap of a near-touch at a high-sym node from wannier90_band.dat.

    Coarse/entangled W90 often puts a Dirac/semimetal touch ~0.5 eV off the
    QE scf Fermi. Prefer that midgap so plotted E=0 sits on the crossing.

    Ignores tiny gaps far from QE EF (deep accidental degeneracies at Γ/A).

    Returns (None, \"none\") if no usable touch found.
    """
    if not work_dir:
        return None, "none"
    labelinfo = os.path.join(work_dir, "wannier90_band.labelinfo.dat")
    band_dat = os.path.join(work_dir, "wannier90_band.dat")
    if not (os.path.isfile(labelinfo) and os.path.isfile(band_dat)):
        return None, "none"

    labels = _parse_w90_labelinfo(labelinfo)
    blocks = _parse_w90_band_blocks(band_dat)
    if not labels or not blocks:
        return None, "none"
    n_pts = min(len(b) for b in blocks)
    if n_pts < 1:
        return None, "none"

    qe, qe_src = detect_qe_fermi_eV(work_dir)
    use_qe_window = qe_src != "none"

    best: Optional[Tuple[float, float, str]] = None  # gap, mid, label
    for lab, idx1 in labels:
        i = idx1 - 1
        if i < 0 or i >= n_pts:
            continue
        evs = sorted(float(b[i]) for b in blocks)
        for j in range(len(evs) - 1):
            gap = evs[j + 1] - evs[j]
            if gap > gap_max:
                continue
            mid = 0.5 * (evs[j] + evs[j + 1])
            # Skip deep/high accidental degeneracies far from QE EF.
            if use_qe_window and abs(mid - qe) > near_qe_window:
                continue
            if best is None or gap < best[0]:
                best = (gap, mid, lab)

    if best is None:
        return None, "none"
    return float(best[1]), f"W90 midgap@{best[2]}"


def detect_plot_fermi_eV(
    work_dir: str,
    touch_offset_min: float = 0.05,
) -> Tuple[float, str]:
    """
    Fermi for W90 band plots / H folding.

    Prefer W90 near-touch midgap when it differs from QE EF by more than
    ``touch_offset_min`` (Dirac / semimetal case). Else QE EF.
    """
    qe, qe_src = detect_qe_fermi_eV(work_dir)
    touch, touch_src = detect_w90_touch_fermi_eV(work_dir)
    if touch is not None and abs(touch - qe) >= touch_offset_min:
        return float(touch), f"{touch_src} (QE {qe_src} was {qe:.4f})"
    return qe, qe_src
