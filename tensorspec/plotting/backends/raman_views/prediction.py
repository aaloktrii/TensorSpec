"""Tab 5 · Layer prediction renderer: the 1L/2L reference with the predicted spectra, and their
positions when the spectra have X/Y.

Moved from gui/components/raman_tabs/predict_panel.py (redraw)."""
from __future__ import annotations

import numpy as np

from tensorspec.plotting.backends.raman_matplotlib_engine import REF_COLORS, draw_reference, orient_axes


def draw_prediction(fig, ref, preds, title: str, cfg: dict):
    has_xy = sum(np.isfinite(p["x"]) for p in preds) > 3
    axs = np.atleast_1d(fig.subplots(1, 2 if has_xy else 1))
    ax = axs[0]
    draw_reference(ax, ref, matching_only=not preds)
    for lab, c in REF_COLORS.items():
        if lab == "no fit":                 # failed fits carry no usable values
            continue
        sel = [p for p in preds if p["layer"] == lab and np.isfinite(p["2D_fwhm"])]
        if sel:
            ax.scatter([p["2D_fwhm"] for p in sel], [p["I2D_IG"] for p in sel], s=18,
                       marker="x", lw=0.8, color="k" if lab in ("1L", "2L") else c,
                       label="%s: %s (%d)" % (title, lab, len(sel)))
    ax.legend(fontsize=7, frameon=False)
    if has_xy:
        ax = axs[1]
        for lab, c in REF_COLORS.items():
            sel = [p for p in preds if p["layer"] == lab]
            if sel:
                ax.scatter([p["x"] for p in sel], [p["y"] for p in sel], s=40,
                           marker="s", color=c, lw=0, label="%s (%d)" % (lab, len(sel)))
        ax.set_aspect("equal")
        ax.set_xlabel("X  [µm]")
        ax.set_ylabel("Y  [µm]")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
        orient_axes(ax, cfg)
