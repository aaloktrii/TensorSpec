"""Wannier90 MDRS / use_ws_distance remapping (material-agnostic).

Matches Wannier90 ``use_ws_distance=.true.`` Fourier interpolation:

* Prefer ``seedname_wsvec.dat`` when present (exact W90 shifts).
* Else compute the same search as ``ws_distance.F90``:
  ``R_in = R_cart + τ_j − τ_i``, minimise over supercell translates
  ``T = (n N1, m N2, ℓ N3)`` with ``n ∈ [-L-1, L+1]``, ``L = ws_search_size``.

Phase uses ``(R+T)`` only; amplitude divided by image degeneracy.
See Wannier90 ``notes_interpolations`` / ``ws_distance.F90``.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_DEFAULT_WS_SEARCH_SIZE = 2
_DEFAULT_WS_DISTANCE_TOL = 1.0e-5  # Angstrom


def parse_win_ws_settings(win_path: str) -> dict:
    """Read use_ws_distance / mp_grid / search / tol / unit_cell from wannier90.win."""
    out = {
        "use_ws_distance": True,  # W90 v3 default
        "mp_grid": None,
        "ws_search_size": _DEFAULT_WS_SEARCH_SIZE,
        "ws_distance_tol": _DEFAULT_WS_DISTANCE_TOL,
        "unit_cell_cart": None,  # 3x3 Angstrom, rows = a1,a2,a3
    }
    if not win_path or not os.path.isfile(win_path):
        return out

    section = None
    cell_rows: List[List[float]] = []
    with open(win_path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.split("!")[0].split("#")[0].strip()
            if not line:
                continue
            low = line.lower().replace(" ", "")
            if line.lower().startswith("begin unit_cell_cart"):
                section = "cell"
                cell_rows = []
                continue
            if line.lower().startswith("end unit_cell_cart"):
                if len(cell_rows) == 3:
                    out["unit_cell_cart"] = np.asarray(cell_rows, dtype=float)
                section = None
                continue
            if section == "cell":
                # skip optional 'ang' / 'bohr' unit line
                if line.lower() in ("ang", "angstrom", "bohr"):
                    continue
                nums = [float(x) for x in line.split()[:3]]
                if len(nums) == 3:
                    cell_rows.append(nums)
                continue
            if low.startswith("use_ws_distance"):
                val = line.split("=")[-1].strip().strip(".").lower()
                out["use_ws_distance"] = val in ("true", "t", ".true.")
            elif low.startswith("mp_grid"):
                nums = re.findall(r"-?\d+", line.split("=", 1)[-1])
                if len(nums) >= 3:
                    out["mp_grid"] = (int(nums[0]), int(nums[1]), int(nums[2]))
            elif low.startswith("ws_search_size"):
                nums = re.findall(r"-?\d+", line.split("=", 1)[-1])
                if nums:
                    out["ws_search_size"] = max(int(x) for x in nums)
            elif low.startswith("ws_distance_tol"):
                try:
                    out["ws_distance_tol"] = float(line.split("=")[-1].strip())
                except ValueError:
                    pass
    return out


def resolve_wsvec_path(work_dir: str, w90_filepath: str) -> Optional[str]:
    """Locate seedname_wsvec.dat beside hr.dat."""
    base = os.path.basename(w90_filepath)
    seed = base.replace("_hr.dat", "").replace(".dat", "")
    if seed.endswith("_hr"):
        seed = seed[: -len("_hr")]
    candidates = [
        os.path.join(work_dir, f"{seed}_wsvec.dat"),
        os.path.join(work_dir, "wannier90_wsvec.dat"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def parse_wsvec_dat(wsvec_path: str) -> Dict[Tuple[int, int, int, int, int], List[np.ndarray]]:
    """Parse wannier90_wsvec.dat → {(Rx,Ry,Rz,i,j): [T,...]} with 1-based i,j."""
    table: Dict[Tuple[int, int, int, int, int], List[np.ndarray]] = {}
    with open(wsvec_path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("##")]
    idx = 0
    while idx < len(lines):
        parts = lines[idx].split()
        if len(parts) < 5:
            idx += 1
            continue
        Rx, Ry, Rz, iw, jw = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        idx += 1
        if idx >= len(lines):
            break
        ndeg = int(lines[idx].split()[0])
        idx += 1
        shifts: List[np.ndarray] = []
        for _ in range(ndeg):
            if idx >= len(lines):
                break
            sp = lines[idx].split()
            shifts.append(np.array([int(sp[0]), int(sp[1]), int(sp[2])], dtype=float))
            idx += 1
        if not shifts:
            shifts = [np.zeros(3, dtype=float)]
        table[(Rx, Ry, Rz, iw, jw)] = shifts
    return table


def ws_minimal_shifts(
    tau_i: np.ndarray,
    tau_j: np.ndarray,
    R_frac: np.ndarray,
    A: np.ndarray,
    mp_grid: Sequence[int],
    *,
    search_size: int = _DEFAULT_WS_SEARCH_SIZE,
    tol: float = _DEFAULT_WS_DISTANCE_TOL,
) -> Tuple[List[np.ndarray], int]:
    """Mirror Wannier90 ``R_wz_sc`` for ``R_in = R_cart + τ_j − τ_i``.

    Search range is ``[-L-1, L+1]`` with ``L = search_size`` (W90 convention).
    """
    N1, N2, N3 = (int(mp_grid[0]), int(mp_grid[1]), int(mp_grid[2]))
    R_frac = np.asarray(R_frac, dtype=float).reshape(3)
    A = np.asarray(A, dtype=float).reshape(3, 3)
    tau_i = np.asarray(tau_i, dtype=float).reshape(3)
    tau_j = np.asarray(tau_j, dtype=float).reshape(3)
    # W90: R_in = -τ_i + (R_cart + τ_j)
    R_in = R_frac @ A + tau_j - tau_i

    L = int(search_size)
    # First pass: absolute shortest image
    dist2_min = None
    shift0 = np.zeros(3, dtype=float)
    for n in range(-L - 1, L + 2):
        for m in range(-L - 1, L + 2):
            for ell in range(-L - 1, L + 2):
                T = np.array([n * N1, m * N2, ell * N3], dtype=float)
                d2 = float(np.sum((R_in + T @ A) ** 2))
                if dist2_min is None or d2 < dist2_min:
                    dist2_min = d2
                    shift0 = T

    if dist2_min is None:
        return [np.zeros(3, dtype=float)], 1
    if dist2_min < tol * tol:
        return [shift0.copy()], 1

    # Second pass: all images at the same distance (WS boundary degeneracy)
    R_bz = R_in + shift0 @ A
    dist_ref = float(np.sqrt(np.sum(R_bz**2)))
    best: List[np.ndarray] = []
    for n in range(-L - 1, L + 2):
        for m in range(-L - 1, L + 2):
            for ell in range(-L - 1, L + 2):
                T_extra = np.array([n * N1, m * N2, ell * N3], dtype=float)
                # W90 second loop: displace from R_bz, total shift = shift0 + T_extra
                R = R_bz + T_extra @ A
                if abs(float(np.linalg.norm(R)) - dist_ref) < tol:
                    best.append(shift0 + T_extra)

    if not best:
        best = [shift0]
    # Deduplicate
    uniq: List[np.ndarray] = []
    for T in best:
        if not any(np.allclose(T, U) for U in uniq):
            uniq.append(T)
    return uniq, len(uniq)


def expand_hoppings_ws(
    R_frac: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    t_ij: np.ndarray,
    centres: Sequence[np.ndarray],
    A: np.ndarray,
    mp_grid: Sequence[int],
    *,
    search_size: int = _DEFAULT_WS_SEARCH_SIZE,
    tol: float = _DEFAULT_WS_DISTANCE_TOL,
    wsvec_table: Optional[Dict[Tuple[int, int, int, int, int], List[np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand hoppings with WS supercell images; divide t by image degeneracy."""
    centres = [np.asarray(c, dtype=float).reshape(3) for c in centres]
    n_orb = len(centres)
    if n_orb == 0:
        raise ValueError("WS remap needs Wannier centres.")

    R_list: List[np.ndarray] = []
    i_list: List[int] = []
    j_list: List[int] = []
    t_list: List[complex] = []

    R_frac = np.asarray(R_frac, dtype=float)
    i_idx = np.asarray(i_idx, dtype=np.int64)
    j_idx = np.asarray(j_idx, dtype=np.int64)
    t_ij = np.asarray(t_ij, dtype=np.complex128)

    for n in range(t_ij.shape[0]):
        ii = int(i_idx[n]) % n_orb
        jj = int(j_idx[n]) % n_orb
        R0 = R_frac[n]
        shifts = None
        if wsvec_table is not None:
            key = (
                int(round(R0[0])),
                int(round(R0[1])),
                int(round(R0[2])),
                ii + 1,
                jj + 1,
            )
            shifts = wsvec_table.get(key)
        if shifts is None:
            shifts, ndeg = ws_minimal_shifts(
                centres[ii],
                centres[jj],
                R0,
                A,
                mp_grid,
                search_size=search_size,
                tol=tol,
            )
        else:
            ndeg = len(shifts)
        t_scaled = t_ij[n] / float(max(ndeg, 1))
        for T in shifts:
            R_list.append(R0 + np.asarray(T, dtype=float))
            i_list.append(int(i_idx[n]))
            j_list.append(int(j_idx[n]))
            t_list.append(t_scaled)

    return (
        np.asarray(R_list, dtype=float),
        np.asarray(i_list, dtype=np.int64),
        np.asarray(j_list, dtype=np.int64),
        np.asarray(t_list, dtype=np.complex128),
    )
