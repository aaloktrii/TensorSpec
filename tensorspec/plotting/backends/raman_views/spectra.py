"""Tab 3 · Spectra renderers: the cell picker (FWHM(2D) / classification / layer map with the
selected cells outlined) and the stacked spectra of the selected cells with their Lorentzian fits.

Moved from gui/components/raman_tabs/spectra_panel.py (refresh_mini, redraw)."""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import Patch

from tensorspec.core.raman.classification import label_of
from tensorspec.plotting.backends.raman_matplotlib_engine import (class_rgba, n_color, na_color,
                                                                  orient_axes, side_colorbar)


def draw_cell_picker(fig, ax, mf, view: str, cfg: dict, items, optical=None, optical_image=None,
                     with_opt=False, alpha=1.0, class_layers=None, class_calls=None,
                     layer_grid=None, fwhm_map=None):
    """Small map to pick cells from. ``items``: the selected cells ({idx, color, visible});
    ``class_layers/calls``: tab 2's classification (or None); ``layer_grid``: N per pixel (or None)."""
    if with_opt:
        ax.imshow(np.ma.masked_invalid(optical_image), origin="lower",
                  extent=optical.extent, cmap="gray", aspect="equal", zorder=0)
    handles, note, title = [], "", view

    if view == "Classification":
        if class_layers is None or class_layers.shape != mf.shape:
            note = "No classification yet —\nrun 'Classify map' in tab 2"
        else:
            rgba = class_rgba(class_layers, class_calls, cfg)
            rgba[..., 3] *= alpha
            ax.imshow(rgba, origin="lower", extent=mf.extent,
                      interpolation="nearest", aspect="equal", zorder=1)
            present = sorted({(int(class_layers[r, c]), class_calls[r, c])
                              for r in range(mf.shape[0]) for c in range(mf.shape[1])},
                             key=lambda t: (t[0] < 0, t[0], str(t[1])))
            for lay, call in present:
                if call is None or (lay < 0 and call != "N/A"):
                    continue
                col = (na_color(lay, cfg) if call == "N/A"
                       else cfg["colors"][lay][call])
                handles.append(Patch(facecolor=col, edgecolor="#555",
                                     label=label_of(lay, call)))
            handles.append(Patch(facecolor=cfg["unclassified_color"],
                                 edgecolor="#555", label="Unclassified"))
            mode = {"manual": "Manual %dL" % int(cfg.get("manual_layer", 3)),
                    "auto": "Auto, I(2D)/I(G)",
                    "map": "layer regions"}.get(cfg.get("layer_mode"), "")
            title = "Classification (%s)" % mode
    elif view == "Layer map (N)":
        g = layer_grid
        if g is None or not (g > 0).any():
            note = "No layer map yet —\ndefine layer regions in tab 2"
        else:
            rgba = np.zeros(g.shape + (4,))
            for n in sorted(set(g[g > 0].tolist())):
                rgba[g == n] = mcolors.to_rgba(n_color(n, cfg), alpha)
                handles.append(Patch(facecolor=n_color(n, cfg), edgecolor="#555",
                                     label="%dL" % n))
            ax.imshow(rgba, origin="lower", extent=mf.extent,
                      interpolation="nearest", aspect="equal", zorder=1)
            title = "Layer map (N per region)"
    if view == "FWHM(2D)" or note:
        data = (fwhm_map if fwhm_map is not None
                else mf.greyscale())
        im = ax.imshow(np.ma.masked_invalid(data), origin="lower", extent=mf.extent,
                       cmap=cfg.get("cmap", "viridis"), interpolation="nearest",
                       aspect="equal", alpha=alpha, zorder=1)
        if not note:
            cb = side_colorbar(fig, ax, im, size="5%", pad=0.05)
            cb.set_label("FWHM(2D) [cm$^{-1}$]", fontsize=7)
            cb.ax.tick_params(labelsize=7)
            title = "FWHM(2D)"
    if note:
        ax.text(0.5, 0.5, note, transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#b71c1c", zorder=6,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#b71c1c", alpha=0.9))

    left, right, bottom, top = mf.extent
    ny, nx = mf.shape
    dx, dy = (right - left) / nx, (top - bottom) / ny
    for xv in np.linspace(left, right, nx + 1):
        ax.axvline(xv, color="w", lw=0.3, alpha=0.5, zorder=2)
    for yv in np.linspace(bottom, top, ny + 1):
        ax.axhline(yv, color="w", lw=0.3, alpha=0.5, zorder=2)
    for it in items:
        r, c = np.argwhere(mf.index_map == it["idx"])[0]
        ax.add_patch(mpatches.Rectangle((left + c * dx, bottom + r * dy), dx, dy,
                                        fill=False, edgecolor=it["color"],
                                        lw=2.6 if it["visible"] else 1.0,
                                        ls="-" if it["visible"] else ":", zorder=4))
    pad = 0.08 * max(right - left, top - bottom) if with_opt else 0.0
    ax.set_xlim(left - pad, right + pad)
    ax.set_ylim(bottom - pad, top + pad)
    ax.set_title(title + " — click cells", fontsize=9)
    ax.tick_params(labelsize=7)
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="upper center",
                  bbox_to_anchor=(0.5, -0.08), ncol=min(4, len(handles)),
                  frameon=False, handlelength=1.2, columnspacing=0.8)
    orient_axes(ax, cfg)


def draw_no_selection(ax):
    ax.text(0.5, 0.5, "Select grid cells on the left to compare their spectra",
            ha="center", va="center", color="#999", transform=ax.transAxes)
    ax.set_axis_off()


def draw_stacked_selection(ax, mf, vis, names, lo: float, hi: float, n_peaks: int, norm: bool,
                           offset_frac: float, show_fit: bool, show_comp: bool,
                           resolution: float):
    """Spectra of the visible selected cells (``vis``, labelled ``names``), offset and optionally
    normalised, with their n-Lorentzian fits for this (lo, hi, n_peaks) and the FWHM per trace."""
    m = (mf.axis >= lo) & (mf.axis <= hi)
    x = mf.axis[m]
    scaled = []
    for it in vis:
        y = mf.data[it["idx"]][m]
        ymin = float(np.nanpercentile(y, 2))
        ymax = float(np.nanmax(y))
        k = (ymax - ymin) if norm and ymax > ymin else 1.0
        scaled.append((it, y, ymin if norm else 0.0, k))
    span = max(((np.nanmax(y) - (b if norm else np.nanmin(y))) / k) for _, y, b, k in scaled)
    step = offset_frac * span
    for j, (it, y, b, k) in enumerate(scaled):
        off = j * step
        ax.plot(x, (y - b) / k + off, color=it["color"], lw=0.9, label=names[j])
        fr = it.get("fit")
        txt = ""
        if fr is not None and fr[0] == (lo, hi, n_peaks) and fr[1]["ok"]:
            res = fr[1]
            fx, fy = res["curve"]
            if show_fit:
                ax.plot(fx, (fy - b) / k + off, color="k", lw=1.1, ls="--")
            if show_comp and len(res["peaks"]) > 1:
                sl, ic, xm = res["baseline"]
                basel = sl * (fx - xm) + ic
                for (h, c, w) in res["components"]:
                    comp = h * (w / 2) ** 2 / ((fx - c) ** 2 + (w / 2) ** 2)
                    ax.fill_between(fx, (basel - b) / k + off, (basel + comp - b) / k + off,
                                    color=it["color"], alpha=0.18, lw=0)
            txt = "FWHM " + " / ".join("%.1f" % p["fwhm"] for p in res["peaks"])
            if len(res["peaks"]) > 1:
                txt += "  (band %.1f)" % res["band_fwhm"]
        ax.text(x[-1], float(np.nanmedian((y[-8:] - b) / k)) + off, "  " + txt,
                color=it["color"], fontsize=8, va="bottom", ha="right")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Raman shift  [cm$^{-1}$]")
    ax.set_ylabel("Intensity (normalised, stacked)" if norm else "Intensity [counts], stacked")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
    ax.set_title("%d spectra · %d Lorentzian peak%s · resolution %.1f cm$^{-1}$"
                 % (len(vis), n_peaks, "" if n_peaks == 1 else "s",
                    resolution), fontsize=9)
