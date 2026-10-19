"""Photon-energy scan helpers for Chinook/Grizzly ARPES stacking."""
from __future__ import annotations

import numpy as np


def build_hv_list(start: float, finish: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if finish < start:
        raise ValueError(f"finish ({finish}) < start ({start})")
    hv = np.arange(float(start), float(finish) + float(step) / 2.0, float(step))
    if hv.size == 0:
        raise ValueError("hv list empty")
    return hv.astype(float)


def hv_list_from_cli_args(args) -> list[float]:
    if (
        getattr(args, "hv_start", None) is not None
        and getattr(args, "hv_finish", None) is not None
        and getattr(args, "hv_step", None) is not None
    ):
        return build_hv_list(
            args.hv_start, args.hv_finish, args.hv_step
        ).tolist()
    return [float(args.hv)]


def resolve_photon_energies(
    *,
    mode: str,
    single: float | None = None,
    start: float | None = None,
    finish: float | None = None,
    step: float | None = None,
) -> list[float]:
    if mode == "single":
        if single is None:
            raise ValueError("single hv required")
        return [float(single)]
    if mode == "range":
        return build_hv_list(start, finish, step).tolist()
    raise ValueError(f"unknown hv mode: {mode}")


def stack_hv_cubes(
    cubes: list[np.ndarray], hv_list: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(cubes) == 0:
        raise ValueError("no cubes to stack")
    if len(cubes) != len(hv_list):
        raise ValueError("cubes and hv_list length mismatch")
    stacked = np.stack([np.asarray(c) for c in cubes], axis=0)
    return stacked, np.asarray(hv_list, dtype=float)


def tensor_from_stacked_sim(
    stacked: np.ndarray,
    hv: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    energy: np.ndarray,
    metadata: dict | None = None,
):
    from tensorspec.core.data_models import TensorData

    # A stack whose in-plane axes are already momenta (metadata["stack_axes"] == "k", e.g. a photon-energy scan computed
    # with a fixed k_slit range at every hv) is labelled in 1/A; the default is the angle-space stack of the runner.
    meta = dict(metadata or {})
    if str(meta.get("stack_axes", "angle")).lower() == "k":
        labels = ["Photon Energy", "Energy", "kx (Slit)", "ky (Deflect)"]
        units = ["eV", "eV", "1/A", "1/A"]
    else:
        labels = ["Photon Energy", "Energy", "Θ (Slit)", "Φ (Deflect)"]
        units = ["eV", "eV", "deg", "deg"]
    return TensorData(
        value=np.transpose(stacked, (0, 3, 1, 2)),
        axes=[hv, energy, theta, phi],
        labels=labels,
        units=units,
        data_type="Simulated ARPES Matrix Elements",
        metadata=meta,
    )


def is_dispersion_axes(theta: np.ndarray, phi: np.ndarray, tol: float = 1e-9) -> bool:
    def _deg(a: np.ndarray) -> bool:
        a = np.asarray(a)
        return a.size <= 1 or float(np.ptp(a)) < tol

    return _deg(theta) or _deg(phi)
