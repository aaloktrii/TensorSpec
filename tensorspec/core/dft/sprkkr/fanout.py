"""Fan-out helpers: split one big ARPES job into many small ones.

No Qt. No ase2sprkkr, no pymatgen imports (sandy_rule.md layer 1: core =
math only). Pure dataclasses.replace() + numpy.

Chunk math (energy fan-out)
----------------------------
SPR-KKR itself writes NE points as ``np.linspace(EMINEV, EMAXEV, NE)``
inclusive of both ends. To split into K chunks that ``outputs.stitch_spc``
can concatenate cleanly we must NOT repeat a boundary energy between two
chunks (stitch_spc's dedupe is a safety net, not something to rely on
here) and every chunk must itself be a valid inclusive linspace.

The trick: take the FULL axis ``full = np.linspace(e_min, e_max, ne)``,
cut its *index* range into K contiguous, non-overlapping, near-equal
pieces (sizes differ by at most 1 point), and for each piece read its own
``(e_min_i, e_max_i, ne_i)`` straight off ``full`` (``full[start]``,
``full[end-1]``, ``end-start``). Because each piece is a contiguous slice
of an already-uniform grid, re-running ``np.linspace(e_min_i, e_max_i,
ne_i)`` reproduces that slice to within a couple of ULP (float64 rounding
in linspace's internal step multiply differs a hair from slicing — not
bit-identical, but agrees to ~1e-13 relative, far tighter than anything
SPR-KKR or downstream stitching cares about). Concatenating the K
recomputed chunks therefore reproduces the full axis to the same
tolerance, with each energy appearing exactly once (no shared boundary
point, unlike a "K+1 dividers, endpoints repeated" split).

K is clipped so no chunk has fewer than ``MIN_CHUNK_PTS`` points.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .params import ArpesParams

MIN_CHUNK_PTS = 2


def full_energy_axis(params: ArpesParams) -> np.ndarray:
    """The exact energy axis SPR-KKR will write for this job."""
    return np.linspace(params.e_min_eV, params.e_max_eV, params.ne)


def _chunk_sizes(ne: int, k: int) -> List[int]:
    """K near-equal, contiguous chunk sizes summing to ne (first `rem` get +1)."""
    base, rem = divmod(ne, k)
    return [base + 1 if i < rem else base for i in range(k)]


def _clip_k(ne: int, k: int) -> int:
    """Largest k' <= k such that every chunk still gets >= MIN_CHUNK_PTS points."""
    k = max(1, int(k))
    max_k = max(1, ne // MIN_CHUNK_PTS)
    return min(k, max_k)


def split_energy(params: ArpesParams, k: int) -> List[ArpesParams]:
    """Split params' energy range into up to k contiguous, non-overlapping chunks.

    Each returned ArpesParams is a `dataclasses.replace()` of params with
    only e_min_eV, e_max_eV, ne and dataset changed. k=1 (or any k that
    clips down to 1) returns `[params]` unchanged (same dataset).
    """
    k_eff = _clip_k(params.ne, k)
    if k_eff <= 1:
        return [params]

    full = full_energy_axis(params)
    sizes = _chunk_sizes(params.ne, k_eff)

    chunks: List[ArpesParams] = []
    start = 0
    for i, size in enumerate(sizes):
        end = start + size
        e_lo = float(full[start])
        e_hi = float(full[end - 1])
        chunks.append(
            dataclasses.replace(
                params,
                e_min_eV=e_lo,
                e_max_eV=e_hi,
                ne=size,
                dataset=f"{params.dataset}_e{i:02d}",
            )
        )
        start = end
    return chunks


def _sanitize_hv(hv: float) -> str:
    """`21.2` -> `21p2`, `-3.5` -> `m3p5` -- safe in a filename/dataset name."""
    s = f"{hv:.1f}"
    return s.replace("-", "m").replace(".", "p")


def split_hv(params: ArpesParams, hv_list: List[float]) -> List[ArpesParams]:
    """One job per photon energy (kz scan). Independent, no stitching needed."""
    return [
        dataclasses.replace(
            params,
            hv_eV=float(hv),
            dataset=f"{params.dataset}_hv{_sanitize_hv(float(hv))}",
        )
        for hv in hv_list
    ]


@dataclass
class FanoutPlan:
    jobs: List[Tuple[ArpesParams, str, int]]  # (params, subdir_name, nproc)
    mode: str  # actual mode used: "mpi" or "chunks"


def plan_jobs(
    params: ArpesParams,
    nproc_total: int,
    mode: str = "auto",
    mpi_available: bool = True,
) -> FanoutPlan:
    """Decide how to spend nproc_total cores on this ArpesParams job.

    - "mpi": one job, all nproc_total cores (native NT x NP grid, no split).
    - "chunks": nproc_total energy chunks (via split_energy), 1 core each.
    - "auto": mpi if mpi_available else chunks.
    """
    if mode not in ("auto", "mpi", "chunks"):
        raise ValueError(f"mode must be auto/mpi/chunks, got {mode!r}")

    nproc_total = max(1, int(nproc_total))

    actual = mode
    if mode == "auto":
        actual = "mpi" if mpi_available else "chunks"

    if actual == "mpi":
        jobs = [(params, params.dataset, nproc_total)]
    else:
        chunks = split_energy(params, nproc_total)
        jobs = [(c, c.dataset, 1) for c in chunks]

    return FanoutPlan(jobs=jobs, mode=actual)
