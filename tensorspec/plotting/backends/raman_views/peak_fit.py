"""Tab 6 · Peak fitting renderers: one spectrum's band fits (data, background, fit, components,
normalised residuals) and the peak count per band (map or histogram).

Moved from gui/components/raman_tabs/peak_fit_panel.py (draw_fit, draw_summary, the correlation
dialog's figure, COMP_COLORS)."""
from __future__ import annotations

import matplotlib
import numpy as np

from tensorspec.plotting.backends.raman_matplotlib_engine import draw_correlation, orient_axes

COMP_COLORS = ["#1f5fa8", "#e8402a", "#2a8c4a", "#8a5a00", "#7b3fa0", "#17becf"]


def draw_band_fits(fig, bands):
    """``bands``: [(name, result of tools.analyse_band), ...], one column per band."""
    gs = fig.add_gridspec(2, len(bands), height_ratios=[3, 1], hspace=0.05)
    for j, (name, b) in enumerate(bands):
        ax = fig.add_subplot(gs[0, j])
        axr = fig.add_subplot(gs[1, j], sharex=ax)
        if "x" not in b:
            ax.text(0.5, 0.5, "%s: %s" % (name, b.get("why", "")), transform=ax.transAxes,
                    ha="center", color="#999")
            continue
        x, raw, bg = b["x"], b["raw"], b["bg"]
        ax.plot(x, raw, color="#999", lw=0.7, label="data")
        ax.plot(x, bg, color="k", lw=0.6, ls="--", label="background")
        if b.get("ok"):
            base = bg + b["baseline"]
            ax.plot(x, base + b["curve"][2] - b["baseline"], color="#d81b60", lw=1.0,
                    label="fit")
            for k, comp in enumerate(b["components"]):
                ax.plot(x, base + comp, color=COMP_COLORS[k % 6], lw=0.7)
            axr.plot(x, (b["curve"][1] - b["curve"][2]) / (b.get("noise") or 1),
                     color="#555", lw=0.6)
        ax.set_title("%s: %d peak%s%s" % (name, b.get("n", 0), "" if b.get("n") == 1 else "s",
                                          "" if b.get("ok") else " (failed)"), fontsize=9)
        ax.tick_params(labelbottom=False, labelsize=7)
        ax.set_yticks([])
        axr.axhline(0, color="k", lw=0.4)
        axr.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=8)
        axr.tick_params(labelsize=7)
        if j == 0:
            axr.set_ylabel("resid./σ", fontsize=7)
            ax.legend(fontsize=7, frameon=False)


def draw_peak_counts(fig, names, counts, xy=None, cfg=None):
    """Peak count per band: a map when ``xy`` = (x, y) of the spectra is given, otherwise a
    histogram. ``counts``: one array per band name (0 = no fit)."""
    axs = np.atleast_1d(fig.subplots(1, len(names)))
    for ax, name, n in zip(axs, names, counts):
        if xy is not None:
            x, y = xy
            cmap = matplotlib.colors.ListedColormap(["#eeeeee"] + COMP_COLORS)
            sc = ax.scatter(x, y, c=n, cmap=cmap, vmin=-0.5, vmax=6.5, s=40, marker="s", lw=0)
            ax.set_aspect("equal")
            orient_axes(ax, cfg)
            cb = fig.colorbar(sc, ax=ax, ticks=range(0, 7), shrink=0.8)
            cb.set_label("peaks (0 = no fit)", fontsize=7)
        else:
            ax.hist(n, bins=np.arange(-0.5, 7.5, 1), color="#777")
            ax.set_xlabel("peaks")
        ax.set_title("%s peak count" % name, fontsize=9)


def draw_correlation_figure(fig, rows, params):
    """omega_G - omega_2D correlation of strain_doping rows over the Lee et al. grid."""
    ax = fig.add_subplot(111)
    draw_correlation(ax, rows, params)


def draw_spectra_overlay(fig, items, bands, max_legend=12):
    """All loaded spectra on one axes (raw counts), the fit bands shaded and named.
    ``items``: [{label, wn, y}]; ``bands``: the fit model's (name, lo, hi, ...) rows."""
    import matplotlib.transforms as mtransforms
    ax = fig.add_subplot(111)
    many = len(items) > max_legend
    for k, it in enumerate(items):
        ax.plot(it["wn"], it["y"], lw=0.4 if many else 0.8,
                color="#555" if many else COMP_COLORS[k % len(COMP_COLORS)],
                alpha=0.35 if many else 1.0, label=None if many else it["label"])
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for name, lo, hi, *_ in bands:
        ax.axvspan(lo, hi, color="#ffd54f", alpha=0.25, lw=0)
        ax.text(0.5 * (lo + hi), 1.0, name, transform=trans, ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title("%d spectr%s" % (len(items), "um" if len(items) == 1 else "a"), fontsize=9, pad=14)
    if not many and items:
        ax.legend(fontsize=7, frameon=False)
