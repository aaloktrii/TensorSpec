"""Tab 1 · Map Explorer renderers: the quantity map (FWHM(2D) / G height / 2D height) with optical
underlay, pixel grid and selection, and the clicked pixel's spectrum with its fits.

Moved from gui/components/raman_tabs/map_explorer_panel.py (refresh_map, _draw_optical,
_draw_grid, _draw_selection, _apply_view, show_spectrum)."""
from __future__ import annotations

import matplotlib.patches as mpatches
import numpy as np

from tensorspec.plotting.backends.raman_matplotlib_engine import (draw_map_grid, draw_optical_underlay,
                                                                  frame_axes, orient_axes, side_colorbar)


def draw_selection(ax, mf, row: int, col: int):
    left, right, bottom, top = mf.extent
    ny, nx = mf.shape
    dx = (right - left) / nx
    dy = (top - bottom) / ny
    rect = mpatches.Rectangle(
        (left + col * dx, bottom + row * dy), dx, dy,
        fill=False, edgecolor="red", lw=2.0, zorder=4)
    ax.add_patch(rect)


def draw_quantity_map(fig, ax, mf, data, quantity: str, fit_tally: dict, cfg: dict,
                      optical=None, optical_image=None, show_optical=False,
                      show_grid=False, sel=None, area=None, cbar_label=None, title=None):
    """Map of ``data`` (FWHM(2D) map, raw band height or band-window integral) on the map grid.
    ``area``: (x0, x1, y0, y1) of the selected area, drawn as a dashed outline.
    ``cbar_label`` / ``title`` override the defaults for quantities other than FWHM / height."""
    fwhm = quantity == "FWHM(2D)"
    finite = np.isfinite(data)
    if finite.any():
        vmin = float(np.nanpercentile(data[finite], 2))
        vmax = float(np.nanpercentile(data[finite], 98))
        if vmin == vmax:
            vmin, vmax = vmin - 1, vmax + 1
    else:
        vmin, vmax = 0, 1

    im = ax.imshow(np.ma.masked_invalid(data), origin="lower",
                   extent=mf.extent, cmap=cfg["cmap"],
                   vmin=vmin, vmax=vmax, interpolation="nearest",
                   aspect="equal", zorder=1)
    cb = side_colorbar(fig, ax, im)
    cb.set_label(cbar_label or ("FWHM(2D)  [cm$^{-1}$]" if fwhm else "%s  [counts]" % quantity))

    if fwhm and not finite.any():
        # An all-NaN map is otherwise indistinguishable from a map that
        # simply has not been fitted yet, so spell it out.
        tally = fit_tally or {}
        lines = ["No spectrum passed the quality filters",
                 "(%d pixels, none fitted)" % mf.n_points]
        for why, n in sorted(tally.items(), key=lambda kv: -kv[1])[:4]:
            lines.append("%d × %s" % (n, why))
        lines.append("Adjust under Analysis > Fit parameters")
        ax.text(0.5, 0.5, "\n".join(lines), transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#b71c1c",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.6", fc="white",
                          ec="#b71c1c", alpha=0.92))

    if optical is not None and show_optical and optical_image is not None:
        draw_optical_underlay(ax, optical, optical_image, cfg["optical_alpha"])
    if show_grid:
        draw_map_grid(ax, mf)
    if sel is not None:
        draw_selection(ax, mf, *sel)
    if area is not None:
        draw_area(ax, area)

    ax.set_xlabel("X  [µm]")
    ax.set_ylabel("Y  [µm]")
    ax.set_title("%s   (%d × %d pixels)"
                 % (title or ("FWHM of the 2D band" if fwhm else quantity + " (raw)"),
                    mf.shape[1], mf.shape[0]))
    # An equal-aspect image in a wide canvas is anchored right by default
    # once the colorbar has taken its slice, stranding the map in the
    # corner with empty figure beside it.
    ax.set_anchor("C")
    frame_axes(ax, mf, cfg, show_optical and optical is not None)
    orient_axes(ax, cfg)


def draw_no_spectrum(ax):
    ax.text(0.5, 0.5, "No spectrum at this pixel",
            ha="center", va="center", color="#999")
    ax.set_axis_off()


def draw_pixel_spectrum(ax, wn, inten, band, fit_curve, center, row: int, col: int):
    """Raw spectrum of a pixel, the 2D band window, the Voigt fit (fit_curve = (x, y) or None)
    and its centre (None when there is no fit)."""
    ax.plot(wn, inten, lw=0.8, color="#333", label="raw spectrum")
    lo, hi = band
    ax.axvspan(lo, hi, color="#ffd54f", alpha=0.30, label="2D band region")
    if center is not None:
        if fit_curve is not None:
            ax.plot(fit_curve[0], fit_curve[1], lw=1.4, color="#d81b60",
                    label="Voigt + baseline")
        ax.axvline(center, color="#1e88e5", lw=0.8, ls="--")
    ax.set_xlabel("Raman shift  [cm$^{-1}$]")
    ax.set_ylabel("Intensity  [a.u.]")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=3,
              borderpad=0.3, handlelength=1.4)
    ax.set_title("Pixel (row %d, col %d)" % (row, col), fontsize=9)


def set_spectrum_view(ax, wn, inten, lo: float, hi: float, fit_y_to_view: bool):
    """Re-limit the spectrum axes without refitting anything."""
    ax.set_xlim(lo, hi)
    if fit_y_to_view:
        sel = (wn >= lo) & (wn <= hi) & np.isfinite(inten)
        if sel.sum() >= 2:
            y0, y1 = float(np.min(inten[sel])), float(np.max(inten[sel]))
            pad = 0.08 * (y1 - y0) if y1 > y0 else max(1.0, abs(y1) * 0.1)
            ax.set_ylim(y0 - pad, y1 + pad)
    else:
        good = np.isfinite(inten)
        if good.any():
            y0, y1 = float(np.min(inten[good])), float(np.max(inten[good]))
            pad = 0.08 * (y1 - y0) if y1 > y0 else 1.0
            ax.set_ylim(y0 - pad, y1 + pad)


def draw_g_fit(ax, x, y):
    ax.plot(x, y, lw=1.4, color="#1e88e5",
            label="G fit")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=4,
              borderpad=0.3, handlelength=1.4)


# -- map areas (2 · Map ▸ area spectrum) ------------------------------------------------------------

def draw_area(ax, rect):
    """Dashed outline of a selected area (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = rect
    ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    edgecolor="#ff1744", lw=1.6, ls="--", zorder=5))


def make_area_selector(ax, on_select, rect=None, min_span=0.0):
    """Interactive rectangle on a map axes: drag to draw one, drag inside it to move it, drag its
    corner / edge handles to resize it. on_select(x0, x1, y0, y1) (data units) after each change.
    ``rect`` shows an existing area; a plain click (smaller than ``min_span``) leaves it as it was.
    Keep the returned object referenced, or it stops responding; remove it with
    remove_area_selector()."""
    from matplotlib.widgets import RectangleSelector
    state = {"rect": tuple(rect) if rect is not None else None}

    def done(press, release):
        x0, x1, y0, y1 = (float(v) for v in sel.extents)
        state["rect"] = (x0, x1, y0, y1)
        on_select(x0, x1, y0, y1)

    sel = RectangleSelector(ax, done, useblit=False, button=[1], interactive=True,
                            drag_from_anywhere=True, minspanx=min_span, minspany=min_span,
                            spancoords="data",
                            props=dict(edgecolor="#ff1744", fill=False, ls="--", lw=1.4),
                            handle_props=dict(markersize=6, markeredgecolor="#ff1744",
                                              markerfacecolor="white"))
    if rect is not None:
        sel.extents = tuple(rect)
        sel.set_visible(True)

    def restore(event):
        # a click too small to be a rectangle hides the selector; show the current area again
        if state["rect"] is not None and not sel._selection_artist.get_visible():
            sel.extents = state["rect"]
            sel.set_visible(True)
            ax.figure.canvas.draw_idle()
    sel._restore_cid = ax.figure.canvas.mpl_connect("button_release_event", restore)
    return sel


def remove_area_selector(sel):
    """Disconnect a selector made by make_area_selector."""
    if sel is None:
        return
    sel.disconnect_events()
    sel.canvas.mpl_disconnect(sel._restore_cid)


def draw_area_spectrum(ax, wn, spec, n_pixels: int, band, mode: str = "sum"):
    """The integrated spectrum of a map area, with the 2D band window."""
    ax.plot(wn, spec, lw=0.8, color="#333", label="area spectrum (%s)" % mode)
    lo, hi = band
    ax.axvspan(lo, hi, color="#ffd54f", alpha=0.30, label="2D band region")
    ax.set_xlabel("Raman shift  [cm$^{-1}$]")
    ax.set_ylabel("Intensity  [counts]")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=2,
              borderpad=0.3, handlelength=1.4)
    ax.set_title("Area: %d pixels, spectra %s" % (n_pixels, "summed" if mode == "sum" else "averaged"),
                 fontsize=9)

