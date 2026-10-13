"""Synthetic UE49-like data with known ground truth (tests and demos)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def stripe_domains(ny: int = 128, nx: int = 128, period_px: float = 16.0,
                   up_fraction: float = 0.5, angle_deg: float = 0.0) -> np.ndarray:
    """+1 / -1 stripe pattern; up domains are ``up_fraction`` of each period."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    t = np.deg2rad(angle_deg)
    s = (xx * np.cos(t) + yy * np.sin(t)) % period_px
    return np.where(s < up_fraction * period_px, 1, -1).astype(np.int8)


def helicity_stacks(mz: np.ndarray, a0: float = 0.05, n_cp: int = 4, n_cm: int = 4,
                    base: float = 1000.0, noise: float = 0.0, shift_cm: tuple[float, float] = (0.0, 0.0),
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """CP / CM stacks with (CP - CM)/(CP + CM) = a0 * mz.

    ``shift_cm`` displaces the CM images (dy, dx) to mimic helicity drift.
    """
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:mz.shape[0], 0:mz.shape[1]]
    topo = base * (1.0 + 0.2 * np.sin(xx / 9.0) * np.cos(yy / 13.0))
    # a few sharp, bright particles: the non-magnetic features real
    # registration locks onto (stronger than the few-percent XMCD contrast)
    for k in range(6):
        cy, cx = rng.uniform(0.1, 0.9, 2) * mz.shape
        topo += base * 0.6 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.5**2))
    cp = topo * (1.0 + a0 * mz)
    cm = topo * (1.0 - a0 * mz)
    if shift_cm != (0.0, 0.0):
        cm = ndimage.shift(cm, shift_cm, order=1, mode="nearest")

    def stack(img, n):
        out = np.repeat(img[None], n, axis=0)
        if noise:
            out = out + rng.normal(0.0, noise, out.shape)
        return out

    return stack(cp, n_cp), stack(cm, n_cm)


def write_ue49_scan(day: Path, scan: str, frames: np.ndarray, names: list[str], *,
                    energy: np.ndarray | None = None, i0: np.ndarray | None = None,
                    fov: float = 10.0, pol_table: int | None = None,
                    hstab: np.ndarray | None = None) -> Path:
    """Write ``<day>/<scan>.txt`` + ``<day>/<scan>/*.tif`` (+ BeamlineSettings.txt)."""
    import tifffile

    day = Path(day)
    folder = day / scan
    folder.mkdir(parents=True, exist_ok=True)
    n = len(names)
    energy = np.full(n, 576.6) if energy is None else np.asarray(energy, dtype=float)
    i0 = np.full(n, 7.8e-10) if i0 is None else np.asarray(i0, dtype=float)
    hstab = np.zeros(n) if hstab is None else np.asarray(hstab, dtype=float)
    header = ["EnergyTable", "ImageNameTable", "I0Table", "ErrorTable", "FOVTable", "HStabTable", "VStabTable"]
    rows = ["\t".join(header)]
    for i, name in enumerate(names):
        rows.append("\t".join([f"{energy[i]:.5f}", name, f"{i0[i]:.6e}", "No Error", f"{fov:g}",
                               f"{hstab[i]:g}", "0"]))
        tifffile.imwrite(folder / f"{name}.tif", np.asarray(frames[i], dtype=np.float32))
    (day / f"{scan}.txt").write_text("\n".join(rows) + "\n")
    if pol_table is not None:
        (folder / "BeamlineSettings.txt").write_text(
            "NameTable\tValueTable\nIdOn\t1\n"
            f"PolTable (0: lin.hor, 1: lin.vert., 2: circular+, 3: circular-)\t{pol_table}\n"
        )
    return day / f"{scan}.txt"
