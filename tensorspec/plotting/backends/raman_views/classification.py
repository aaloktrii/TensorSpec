"""Tab 2 · Classification renderers: the stacking-order class map, the threshold histograms, and the
figures of its two dialogs (thickness from I(2D)/I(G); layer regions from optical contrast).

Moved from gui/components/raman_tabs/classification_panel.py (refresh_map, show_histograms),
thickness_dialog.py (redraw) and layer_regions_dialog.py (_build_figure, refresh_map, REGION_COLORS)."""
from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Patch

from tensorspec.core.raman.classification import label_of, windows_for
from tensorspec.plotting.backends.raman_matplotlib_engine import (class_rgba, draw_map_grid,
                                                                  draw_optical_underlay, frame_axes,
                                                                  na_color, orient_axes)


def draw_class_map(ax, mf, layers, calls, cfg: dict, optical=None, optical_image=None,
                   show_optical=False, show_grid=False):
    rgba = class_rgba(layers, calls, cfg)
    ax.imshow(rgba, origin="lower", extent=mf.extent,
              interpolation="nearest", aspect="equal", zorder=1)

    if optical is not None and show_optical and optical_image is not None:
        draw_optical_underlay(ax, optical, optical_image, cfg["optical_alpha"])
    if show_grid:
        draw_map_grid(ax, mf)

    present = set()
    ny, nx = layers.shape
    for r in range(ny):
        for c in range(nx):
            present.add((int(layers[r, c]), calls[r, c]))
    handles = []
    for lay, call in sorted(present, key=lambda t: (t[0] < 0, t[0], str(t[1]))):
        if call is None or (lay < 0 and call != "N/A"):
            continue
        col = (na_color(lay, cfg) if call == "N/A"
               else cfg["colors"][lay][call])
        handles.append(Patch(facecolor=col, edgecolor="#555",
                             label=label_of(lay, call)))
    handles.append(Patch(facecolor=cfg["unclassified_color"],
                         edgecolor="#555", label="Unclassified"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=True, title="Class")
    ax.set_xlabel("X  [µm]")
    ax.set_ylabel("Y  [µm]")
    ax.set_title("Stacking-order classification")
    ax.set_anchor("C")
    frame_axes(ax, mf, cfg, show_optical and optical is not None)
    orient_axes(ax, cfg)


def draw_threshold_histograms(fig, fw, am, cfg: dict, shown):
    """FWHM(2D) histogram with the ABA/ABC windows of the ``shown`` layers, and the asymmetry
    histogram with its ABC cut."""
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.hist(fw, bins=max(10, min(60, fw.size // 8)),
             color="#607d8b", edgecolor="white", linewidth=0.4)
    top = ax1.get_ylim()[1]
    for layer in shown:
        (a0, a1), (c0, c1) = windows_for(layer, cfg)
        for (lo, hi), col, key in (((a0, a1), "#1e88e5", "ABA"),
                                   ((c0, c1), "#e53935", "ABC")):
            if hi < fw.min() - 5 or lo > fw.max() + 5:
                continue
            ax1.axvspan(lo, hi, color=col, alpha=0.16, lw=0)
            ax1.axvline(cfg["fwhm"][layer][key], color=col, lw=1.0, ls="--")
        if len(shown) == 1:
            ax1.text(cfg["fwhm"][layer]["ABA"], top * 0.96, " ABA",
                     color="#1e88e5", fontsize=8, va="top")
            ax1.text(cfg["fwhm"][layer]["ABC"], top * 0.96, " ABC",
                     color="#e53935", fontsize=8, va="top")
    ax1.set_xlabel("FWHM(2D)  [cm$^{-1}$]")
    ax1.set_ylabel("pixels")
    ax1.set_title("Width, with the ABA (blue) and ABC (red) windows"
                  if len(shown) == 1 else
                  "Width, with every layer's ABA/ABC windows", fontsize=9)

    if am.size:
        ax2.hist(am, bins=max(10, min(60, am.size // 8)),
                 color="#8d6e63", edgecolor="white", linewidth=0.4)
        ax2.axvline(cfg["asym_threshold"], color="#d81b60", lw=1.4, ls="--",
                    label="threshold %.3f" % cfg["asym_threshold"])
        ax2.legend(fontsize=8)
    ax2.set_xlabel("asymmetry (%s)" % cfg["asym_metric"])
    ax2.set_ylabel("pixels")
    ax2.set_title("Shape asymmetry, with the ABC cut", fontsize=10)


# -- Thickness from I(2D)/I(G) dialog --------------------------------------------------------------

def draw_thickness(fig, mf, rm, lm, counts: dict, c: dict, cfg: dict, layer_order, fin,
                   optical=None, optical_image=None):
    """Ratio map, thickness map and ratio histogram with the cutoffs. ``c`` is the dialog's local
    config (cutoffs, metric), ``cfg`` the suite's (orientation)."""
    a1 = fig.add_subplot(131)
    a2 = fig.add_subplot(132)
    a3 = fig.add_subplot(133)

    for ax in (a1, a2):
        if optical is not None and optical_image is not None:
            ax.imshow(np.ma.masked_invalid(optical_image),
                      origin="lower", extent=optical.extent, cmap="gray",
                      aspect="equal", zorder=0)
    im = a1.imshow(np.ma.masked_invalid(rm), origin="lower", extent=mf.extent,
                   cmap="magma", aspect="equal", interpolation="nearest",
                   alpha=0.85, zorder=1)
    fig.colorbar(im, ax=a1, fraction=0.046, pad=0.03).set_label(
        "I(2D)/I(G) (%s)" % c["ratio_metric"])
    a1.set_title("I(2D)/I(G)", fontsize=10)

    cols = [na_color(n, c) for n in layer_order]
    cmap = mcolors.ListedColormap(cols)
    a2.imshow(np.ma.masked_invalid(lm), origin="lower", extent=mf.extent,
              cmap=cmap, vmin=-0.5, vmax=3.5, aspect="equal",
              interpolation="nearest", alpha=0.9, zorder=1)
    a2.legend(handles=[Patch(facecolor=cols[i], edgecolor="#555",
                             label="%s  (%d)" % (label_of(n, "N/A"), counts[n]))
                       for i, n in enumerate(layer_order)],
              fontsize=8, loc="upper left", framealpha=0.9)
    a2.set_title("Thickness from the ratio", fontsize=10)
    for ax in (a1, a2):
        ax.set_xlim(mf.extent[0], mf.extent[1])
        ax.set_ylim(mf.extent[2], mf.extent[3])
        orient_axes(ax, cfg)
        ax.set_xlabel("X [µm]")
    a1.set_ylabel("Y [µm]")

    if fin.size:
        a3.hist(fin, bins=max(8, min(50, fin.size // 3)), color="#607d8b",
                edgecolor="white", linewidth=0.4)
        for key, lab in (("1_2", "1|2"), ("2_3", "2|3"), ("3_thick", "3|≥4")):
            v = c["ratio_cut"][key]
            a3.axvline(v, color="#d81b60", ls="--", lw=1.1)
            a3.text(v, a3.get_ylim()[1] * 0.96, " " + lab, color="#d81b60",
                    fontsize=8, va="top")
    a3.set_xlabel("I(2D)/I(G)")
    a3.set_ylabel("pixels")
    a3.set_title("Distribution, with the cutoffs", fontsize=10)


# -- Layer regions dialog --------------------------------------------------------------------------

REGION_COLORS = ["#1e88e5", "#fb8c00", "#8e24aa", "#43a047", "#e53935", "#00acc1"]


def build_regions_figure(fig, mf, cfg: dict, optical=None, optical_image=None):
    """Region map axes (camera image + an RGBA overlay on the map grid) and the contrast
    histogram axes. Returns (ax, axh, overlay)."""
    gsp = fig.add_gridspec(3, 1)
    ax = fig.add_subplot(gsp[:2, 0])
    axh = fig.add_subplot(gsp[2, 0])
    if optical is not None and optical_image is not None:
        ax.imshow(np.ma.masked_invalid(optical_image),
                  origin="lower", extent=optical.extent, cmap="gray",
                  aspect="equal", zorder=0)
    ny, nx = mf.shape
    overlay = ax.imshow(np.zeros((ny, nx, 4)), origin="lower",
                        extent=mf.extent, interpolation="nearest",
                        aspect="equal", zorder=1)
    left, right, bottom, top = mf.extent
    pad = 0.15 * max(right - left, top - bottom)
    ax.set_xlim(left - pad, right + pad)
    ax.set_ylim(bottom - pad, top + pad)
    ax.set_xlabel("X [µm]")
    ax.set_ylabel("Y [µm]")
    ax.set_title("Regions (click one to select its row)", fontsize=10)
    orient_axes(ax, cfg)
    return ax, axh, overlay


def region_rgba(mf, region_of, n_regions: int, selected: int, edges=None):
    """Overlay colours: one per region (the selected one more opaque), terrace edges white."""
    ny, nx = mf.shape
    rgba = np.zeros((ny, nx, 4))
    reg = np.full((ny, nx), -1, dtype=int)
    ok = mf.index_map >= 0
    reg[ok] = region_of[mf.index_map[ok]]
    for j in range(n_regions):
        a = 0.85 if j == selected else 0.5
        rgba[reg == j] = mcolors.to_rgba(REGION_COLORS[j % len(REGION_COLORS)], a)
    if edges is not None:
        rgba[edges] = (1.0, 1.0, 1.0, 0.85)
    return rgba


def draw_region_labels(ax, mf, region_of, n_regions: int, old_texts):
    """'R1', 'R2', … at each region's median position; removes ``old_texts``, returns the new."""
    for t in old_texts:
        t.remove()
    texts = []
    for j in range(n_regions):
        m = region_of == j
        if m.any():
            texts.append(ax.text(
                float(np.median(mf.x[m])), float(np.median(mf.y[m])), "R%d" % (j + 1),
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc=REGION_COLORS[j % 6],
                          ec="none", alpha=0.9)))
    return texts


def draw_contrast_histogram(axh, contrast, cut: float, levels):
    """Optical-contrast histogram with the ±cut band, the region levels and their boundaries."""
    axh.clear()
    if contrast is not None:
        fin = contrast[np.isfinite(contrast)]
        if fin.size:
            axh.hist(fin, bins=max(10, min(60, fin.size // 3)),
                     color="#90a4ae", edgecolor="white", linewidth=0.4)
        for v in (-cut, cut):
            axh.axvline(v, color="#555", ls=":", lw=1)
        if levels is not None:
            top = axh.get_ylim()[1]
            for j, c in enumerate(levels):
                axh.axvline(c, color=REGION_COLORS[j % 6], lw=1.6)
                axh.text(c, top * 0.92, " R%d" % (j + 1),
                         color=REGION_COLORS[j % 6], fontsize=9,
                         fontweight="bold")
            srt = np.sort(levels)
            for a, b in zip(srt[:-1], srt[1:]):
                axh.axvline(0.5 * (a + b), color="#555", ls="--", lw=0.8)
        axh.set_xlabel(r"optical contrast  $(I - I_{\mathrm{SiO_2}})\,/\,I_{\mathrm{SiO_2}}$")
        axh.set_ylabel("pixels")
