"""Preview PNGs for the UE49 batch (matplotlib Agg; called by core.peem.ue49.batch)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _div(ax, img, title, plt):
    v = img[np.isfinite(img)]
    c = float(np.median(v)) if v.size else 0.0
    lim = float(np.percentile(np.abs(v - c), 99)) if v.size else 1.0
    im = ax.imshow(img, cmap="RdBu_r", vmin=c - lim, vmax=c + lim)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _gray(ax, img, title, plt):
    v = img[np.isfinite(img)]
    lo, hi = (np.percentile(v, [1, 99.5]) if v.size else (0, 1))
    im = ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


class Preview:
    """Callable used by ``batch.run``: ``preview(path, res, dom, title)`` plus
    ``.spectra(path, lines, title)`` and ``.ld(path, maps, title)``."""

    dpi = 70

    def __call__(self, path: Path, res: dict, dom: dict | None, title: str) -> None:
        plt = _plt()
        n = 3 if dom is not None else 2
        fig, ax = plt.subplots(1, n, figsize=(4.6 * n, 4.4))
        _gray(ax[0], res["mean_a"], "mean (channel A)", plt)
        _div(ax[1], res["map"], res["kind"], plt)
        if dom is not None:
            ax[2].imshow(np.where(res["mask"], dom["mz"], np.nan), cmap="bwr", vmin=-1, vmax=1)
            s = dom["stats"]
            ax[2].set_title(f"domain_map f+={s['f_up']:.2f} f-={s['f_down']:.2f} (no BG ROI)", fontsize=9)
            ax[2].axis("off")
        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)

    def spectra(self, path: Path, lines: dict, title: str) -> None:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for label, (e, y) in lines.items():
            ax.plot(e, y, lw=1.1, label=label)
        ax.set_xlabel("Photon energy (eV)")
        ax.set_ylabel("Intensity (arb. u.)")
        ax.legend(fontsize=8, frameon=False)
        ax.set_title(title, fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)

    def image(self, path: Path, img: np.ndarray, title: str) -> None:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        _gray(ax, img, title, plt)
        ax.title.set_fontsize(7)
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)

    def ld(self, path: Path, maps: dict, title: str) -> None:
        plt = _plt()
        fig, ax = plt.subplots(1, len(maps), figsize=(4.4 * len(maps), 4.2))
        for a, (k, v) in zip(np.atleast_1d(ax), maps.items()):
            _div(a, v, k, plt)
        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)


def session_overview(root: Path, session: str, rows: list[dict], out_path: Path, ncols: int = 6) -> int:
    """Contact sheet of each cluster's merged map (single unit when there is no merge),
    in scan order, titled with the logbook condition. Returns the number of panels."""
    import xarray as xr

    plt = _plt()
    panels = []
    for r in rows:
        if r.get("session") != session or r.get("type") != "cluster" or r.get("status") != "ok":
            continue
        cdir = Path(root) / r["path"]
        nc = cdir / "merged" / "maps.nc"
        if not nc.exists():
            units = sorted((cdir / "units").glob("*/maps.nc"))
            if not units:
                continue
            nc = units[0]
        with xr.open_dataset(nc) as ds:
            var = "pcd" if "pcd" in ds else "pld"
            img = ds[var].values
        bits = [r["id"].split("_", 1)[1], f"{float(r.get('T_K_measured') or 'nan'):.0f} K"]
        for k in ("log_ccs_mA", "log_pulse_before", "log_rotation_deg"):
            v = r.get(k)
            if v and not (k == "log_rotation_deg" and v == "0"):
                bits.append(f"CCS {v} mA" if k == "log_ccs_mA" else f"rot {v}" if k == "log_rotation_deg" else v)
        panels.append((img, var, " | ".join(bits)))
    if not panels:
        return 0
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.3 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (img, var, title) in zip(np.atleast_1d(axes).ravel(), panels):
        v = img[np.isfinite(img)]
        c = float(np.median(v)) if v.size else 0.0
        lim = float(np.percentile(np.abs(v - c), 99)) if v.size else 1.0
        ax.imshow(img, cmap="RdBu_r", vmin=c - lim, vmax=c + lim)
        ax.set_title(title, fontsize=6.5)
    fig.suptitle(f"{session}: merged cluster maps in scan order (each panel scaled to its own median ± p99)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=60)
    plt.close(fig)
    return len(panels)
