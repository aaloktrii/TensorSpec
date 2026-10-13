"""Raman figures (matplotlib only; no Qt): the Raman suite's figure style, map-axes helpers,
class / layer colours, the stacked-spectra publication figure, the per-dataset fit PDFs, the
1L/2L reference and prediction plots, and the omega_G - omega_2D strain/doping grid and
correlation plot.

Moved from Raman_Classifier: ``raman_map_classifier_8 (1).py`` (rcParams, side_colorbar,
frame_axes, orient_axes, layer colours, draw_stack), ``raman_peak_fit.py`` (STYLE, styled,
subplots, summary.pdf / fits.pdf), ``graphene_reference.py`` (COLORS as REF_COLORS, draw_reference,
plot_reference, plot_prediction) and ``strain_doping.py`` (GROUP_STYLE, draw_grid,
draw_correlation, plot_correlation).

The original GUI set its rcParams globally at import. Inside TensorSpec that would restyle
every other suite, so they are kept in GUI_RC and applied with raman_style() around Raman
drawing and saving only."""
from __future__ import annotations

import functools
import logging as _logging
import os
from typing import List, Tuple

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.ticker
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tensorspec.core.raman.classification import LAYER_THICK
from tensorspec.core.raman.strain_doping import PARAMS, PHYS_2D, PHYS_G, _point

_logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)

GUI_RC = {
    # Vector exports keep text as real, editable text rather than outlines, so
    # labels can be changed in Inkscape or Illustrator before going to Overleaf.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    # Arial first (on every Mac, and what journals usually ask for), then
    # Helvetica; Liberation Sans is Arial's metric twin on Linux. The maths text
    # (cm⁻¹, µm superscripts) must use the same family, or every label containing
    # it would still carry matplotlib's own font into Illustrator.
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
}


def raman_style():
    """Context in which the Raman suite draws and saves its figures (GUI_RC)."""
    return matplotlib.rc_context(GUI_RC)


def side_colorbar(fig, ax, mappable, size="4%", pad=0.08):
    """A colour bar glued to the map itself. fig.colorbar(ax=...) places it
    beside the axes' full box, which for an equal-aspect map in a wide panel
    is far from the map."""
    cax = make_axes_locatable(ax).append_axes("right", size=size, pad=pad)
    return fig.colorbar(mappable, cax=cax)


def frame_axes(ax, mf, cfg: dict, with_optical: bool):
    """Frame a map view on the Raman grid. With the camera image underneath
    the axes would otherwise stretch to the whole camera frame and shrink the
    map to a patch in the middle; a small margin keeps the flake's
    surroundings visible."""
    if mf is None:
        return
    left, right, bottom, top = mf.extent
    if with_optical and not cfg.get("zoom_to_map", True):
        return
    pad = 0.08 * max(right - left, top - bottom) if with_optical else 0.0
    ax.set_xlim(left - pad, right + pad)
    ax.set_ylim(bottom - pad, top + pad)


def orient_axes(ax, cfg: dict):
    """Draw +Y downward when asked, matching LabSpec's display. Display only:
    data coordinates and mouse clicks are unaffected."""
    down = bool(cfg.get("y_down", True))
    if down != ax.yaxis_inverted():
        ax.invert_yaxis()


def draw_placeholder(ax, text=None):
    """Grey centred message on an empty axes (no map / no classification yet)."""
    if text is not None:
        ax.text(0.5, 0.5, text, ha="center", va="center", color="#999")
    ax.set_axis_off()


def draw_figure_message(fig, text):
    """Grey centred message on an otherwise empty figure."""
    fig.text(0.5, 0.5, text, ha="center", va="center", color="#999")


def draw_optical_underlay(ax, optical, image, alpha):
    """The camera image under a map view (map and classification tabs)."""
    ax.imshow(np.ma.masked_invalid(image), origin="lower", extent=optical.extent,
              cmap="gray", alpha=alpha,
              interpolation="bilinear", aspect="equal", zorder=2)


def draw_map_grid(ax, mf):
    """White pixel-grid lines over a map view."""
    left, right, bottom, top = mf.extent
    ny, nx = mf.shape
    for xv in np.linspace(left, right, nx + 1):
        ax.axvline(xv, color="w", lw=0.4, alpha=0.6, zorder=3)
    for yv in np.linspace(bottom, top, ny + 1):
        ax.axhline(yv, color="w", lw=0.4, alpha=0.6, zorder=3)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)


# Neighbouring layer counts get strongly contrasting colours.
N_COLORS = {3: "#1e88e5", 4: "#fb8c00", 5: "#43a047", 6: "#8e24aa",
            7: "#e53935", 8: "#00acc1", 9: "#6d4c41"}


def n_color(n: int, cfg: dict) -> str:
    """One colour per layer count, for views of the layer map itself."""
    if n in (1, 2) or n == LAYER_THICK:
        return na_color(n, cfg)
    return N_COLORS.get(int(n), "#9e9e9e")


def na_color(layer: int, cfg: dict) -> str:
    rc = cfg.get("ratio_colors", {})
    key = "thick" if layer == LAYER_THICK else str(layer)
    return rc.get(key, "#607d8b")


def class_rgba(layers: np.ndarray, calls: np.ndarray, cfg: dict) -> np.ndarray:
    ny, nx = layers.shape
    rgba = np.zeros((ny, nx, 4), dtype=float)
    unc = np.array(mcolors.to_rgba(cfg["unclassified_color"]))
    unc[3] = float(cfg.get("unclassified_alpha", 1.0))
    for r in range(ny):
        for c in range(nx):
            lay, call = int(layers[r, c]), calls[r, c]
            if call == "N/A":
                rgba[r, c] = mcolors.to_rgba(na_color(lay, cfg))
            elif lay < 0 or call is None:
                rgba[r, c] = unc
            else:
                hexcol = cfg["colors"].get(lay, {}).get(call, "#000000")
                rgba[r, c] = mcolors.to_rgba(hexcol)
    return rgba


def _tick_step(span: float) -> float:
    for s in (10, 25, 50, 100, 200, 500, 1000):
        if span / s <= 3:
            return s
    return 1000.0


def draw_stack(fig, traces: List[dict], windows: List[Tuple[float, float]],
               step: float = 1.15, normalise: bool = True,
               trace_labels: bool = False,
               band_labels: List[Tuple[str, float]] = (), title: str = ""):
    """Stacked spectra in one closed frame, the x axis broken between
    spectral windows with // marks, no y ticks. Every window has the same
    cm⁻¹ per centimetre. Raw data only: each trace is shifted by its 5th
    percentile inside the windows and scaled to its maximum there (or, with
    normalise off, all traces by one common factor)."""
    fig.clear()
    if not traces or not windows:
        fig.text(0.5, 0.5, "No spectra" if windows else "No ranges",
                 ha="center", va="center", color="#999", fontsize=8)
        return
    spans = [hi - lo for lo, hi in windows]
    axs = np.atleast_1d(fig.subplots(
        1, len(windows), sharey=True,
        gridspec_kw={"width_ratios": spans, "wspace": 0.04}))

    def in_windows(wn):
        m = np.zeros(wn.shape, bool)
        for lo, hi in windows:
            m |= (wn >= lo) & (wn <= hi)
        return m

    scale = []
    for t in traces:
        m = in_windows(t["wn"])
        if m.sum() < 2:
            scale.append(None)
            continue
        base = float(np.nanpercentile(t["inten"][m], 5))
        scale.append((base, float(np.nanmax(t["inten"][m])) - base))
    common = max([s[1] for s in scale if s] or [1.0]) or 1.0

    n = len(traces)
    lo0, hi0 = windows[0]
    for i, (t, sc) in enumerate(zip(traces, scale)):
        if sc is None:
            continue
        off = (n - 1 - i) * step               # first trace at the top
        y = (t["inten"] - sc[0]) / ((sc[1] or 1.0) if normalise else common) + off
        for ax, (lo, hi) in zip(axs, windows):
            s = (t["wn"] >= lo) & (t["wn"] <= hi)
            ax.plot(t["wn"][s], y[s], color=t["color"], lw=0.6)
        if trace_labels:
            left = (t["wn"] >= lo0) & (t["wn"] <= lo0 + 0.3 * (hi0 - lo0))
            ly = float(np.nanmax(y[left])) + 0.06 if left.any() else off + 0.25
            axs[0].text(lo0 + 0.02 * (hi0 - lo0), ly, t["label"], color=t["color"],
                        fontsize=6.5, style="italic", va="bottom")

    top = (n - 1) * step + 1.06
    ymax = top + (0.3 if band_labels else 0.08) + (0.32 if title else 0.0)
    axs[0].set_ylim(-0.12, ymax)
    for text, pos in band_labels:
        for ax, (lo, hi) in zip(axs, windows):
            if lo <= pos <= hi:
                ax.text(pos, top, text, ha="center", va="bottom", fontsize=8)
    if title:
        axs[0].text(lo0 + 0.02 * (hi0 - lo0), ymax - 0.04, title, style="italic",
                    fontsize=8, va="top")

    for k, (ax, (lo, hi)) in enumerate(zip(axs, windows)):
        ax.set_xlim(lo, hi)
        st = _tick_step(hi - lo)
        last = hi if k == len(windows) - 1 else hi - st / 4   # no label on a break
        ax.set_xticks(np.arange(np.ceil(lo / st) * st, last + 1e-9, st))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(st / 2))
        ax.tick_params(axis="x", direction="out", labelsize=7, width=0.6,
                       length=3, top=False)
        ax.tick_params(axis="x", which="minor", direction="out", width=0.5,
                       length=1.6, top=False)
        ax.tick_params(axis="y", left=False, labelleft=False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        if k > 0:
            ax.spines["left"].set_visible(False)
        if k < len(windows) - 1:
            ax.spines["right"].set_visible(False)
    for a, b in zip(axs[:-1], axs[1:]):                # // marks on each break
        for ax, xb in ((a, 1), (b, 0)):
            ax.plot([xb, xb], [0, 1], transform=ax.transAxes, ls="none",
                    marker=[(-1, -1.6), (1, 1.6)], markersize=5,
                    markeredgewidth=0.6, color="k", clip_on=False)
    axs[0].set_ylabel("Intensity (arb. u.)", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.12)
    # One x label centred under all windows: an empty frameless axes over
    # the same area carries it, so it sits just below the tick labels.
    l = axs[0].get_position().x0
    r = axs[-1].get_position().x1
    b, t = axs[0].get_position().y0, axs[0].get_position().y1
    big = fig.add_axes([l, b, r - l, t - b], frameon=False, zorder=-1)
    big.set_xticks([])
    big.set_yticks([])
    big.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=8, labelpad=13)


# Applied only while this module draws, so importing it (e.g. from the GUI)
# leaves the caller's matplotlib backend and settings alone.
STYLE = {"font.family": "sans-serif",
         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
         "font.size": 8, "pdf.fonttype": 42, "axes.linewidth": 0.6}



def stack_figure(width_in: float, height_in: float, traces: List[dict],
                 windows: List[Tuple[float, float]], **kw) -> Figure:
    """An off-screen (Agg) figure of draw_stack, drawn in the Raman figure style."""
    fig = Figure(figsize=(width_in, height_in))
    FigureCanvasAgg(fig)
    with raman_style():
        draw_stack(fig, traces, windows, **kw)
    return fig


def styled(fn):
    @functools.wraps(fn)
    def wrap(*a, **k):
        with matplotlib.rc_context(STYLE):
            return fn(*a, **k)
    return wrap


def subplots(*a, figsize=None, **k):
    """A pyplot-free figure: no GUI window, no global figure registry."""
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig, fig.subplots(*a, **k)


@styled
def fit_pdfs(folder, rows, out, summ, curves):
    """summary.pdf (histograms, mean ± SD; 2D FWHM vs I(2D)/I(G)) and fits.pdf (every
    spectrum with its fit, 12 per page) for core.raman.batch.fit_folder."""
    good = [r for r in rows if r.get("ok")]
    # ---- summary.pdf
    name = os.path.basename(os.path.normpath(folder))
    nrow = int(np.ceil((len(summ) + 1) / 3))
    fig, axs = subplots(nrow, 3, figsize=(17.8 / 2.54, 5 * nrow / 2.54))
    for ax, (q, label, v) in zip(axs.flat, summ):
        if v.size:
            ax.hist(v, bins=min(30, max(5, v.size // 3)), color="#888", edgecolor="none")
            m, sd = v.mean(), v.std(ddof=1) if v.size > 1 else 0
            ax.axvline(m, color="k", lw=0.8)
            ax.axvspan(m - sd, m + sd, color="k", alpha=0.08, lw=0)
            ax.set_title("%.4g ± %.2g  (n=%d)" % (m, sd, v.size), fontsize=7)
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    for ax in axs.flat[len(summ):]:
        ax.axis("off")
    ax = axs.flat[len(summ)]
    ax.axis("on")
    x = np.array([r["2D_fwhm"] for r in good], float)
    y = np.array([r["I2D_IG"] for r in good], float)
    ax.scatter(x, y, s=4, color="k", lw=0)
    ax.set_xlabel("2D FWHM (cm$^{-1}$)")
    ax.set_ylabel("I(2D)/I(G)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("%s  —  %d spectra, %d fitted" % (name, len(rows), len(good)), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "summary.pdf"))

    # ---- fits.pdf
    with PdfPages(os.path.join(out, "fits.pdf")) as pdf:
        per = 12
        for start in range(0, len(curves), per):
            fig, axs = subplots(per // 2, 4, figsize=(17.8 / 2.54, 24 / 2.54),
                                    gridspec_kw={"width_ratios": [2, 3, 2, 3]})
            for k in range(per):
                aG, aD = axs[k // 2, 2 * (k % 2)], axs[k // 2, 2 * (k % 2) + 1]
                if start + k >= len(curves):
                    aG.axis("off"); aD.axis("off")
                    continue
                f, cg, cd, ok = curves[start + k]
                for ax, c in ((aG, cg), (aD, cd)):
                    if c is not None:
                        ax.plot(c[0], c[1], color="#999", lw=0.5)
                        ax.plot(c[0], c[2], color="#e8402a" if ok else "#1f5fa8", lw=0.7)
                    ax.set_yticks([])
                    ax.tick_params(labelsize=5, length=2)
                aG.set_title(f[:38] + ("" if ok else "  [rejected]"), fontsize=5, loc="left")
            fig.tight_layout(h_pad=0.3, w_pad=0.2)
            pdf.savefig(fig)


REF_COLORS = {"1L": "#1f5fa8", "2L": "#e8402a", ">=3L": "#222222",
          "2L twisted?": "#f2a900", "1L or twisted 2L": "#7fb2e5",
          "unassigned": "#bbbbbb", "no fit": "#eeeeee"}


# ---------------------------------------------------------------------------
# plots (vector PDF)
# ---------------------------------------------------------------------------

def draw_reference(ax, ref, matching_only=False):
    for d in ref["datasets"]:
        c = REF_COLORS[d["label"]]
        pts = d["points"]
        for m in (True, False):
            if matching_only and not m:
                continue
            sel = [p for p in pts if p["match"] == m]
            ax.scatter([p["2D_fwhm"] for p in sel], [p["I2D_IG"] for p in sel], s=6,
                       facecolors=c if m else "none", edgecolors=c, lw=0.4,
                       label="%s ref (%s, %s)" % (d["label"], d["name"].split("_532nm")[0],
                                                  "matching" if m else "other spots"))
    ax.set_xlabel("2D FWHM (cm$^{-1}$)")
    ax.set_ylabel("I(2D)/I(G)")
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)


@styled
def plot_reference(ref, path):
    fig, axs = subplots(1, 2, figsize=(17.8 / 2.54, 7 / 2.54))
    draw_reference(axs[0], ref)
    axs[0].legend(fontsize=5, frameon=False)
    ax = axs[1]
    for d in ref["datasets"]:
        v = [p["2D_fwhm"] for p in d["points"]]
        ax.hist(v, bins=np.arange(10, 90, 2), histtype="step", color=REF_COLORS[d["label"]],
                label="%s (%s)" % (d["label"], d["name"].split("_532nm")[0]))
    ax.set_xlabel("2D FWHM (cm$^{-1}$)")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path)


@styled
def plot_prediction(preds, ref, path, title):
    has_xy = sum(np.isfinite(p["x"]) for p in preds) > 3
    fig, axs = subplots(1, 2 if has_xy else 1, figsize=(17.8 / 2.54, 7.5 / 2.54),
                            squeeze=False)
    ax = axs[0, 0]
    draw_reference(ax, ref, matching_only=True)
    for lab, c in REF_COLORS.items():
        if lab == "no fit":                     # failed fits carry no usable values
            continue
        sel = [p for p in preds if p["layer"] == lab and np.isfinite(p["2D_fwhm"])]
        if sel:
            ax.scatter([p["2D_fwhm"] for p in sel], [p["I2D_IG"] for p in sel], s=14,
                       marker="x", color=c if lab not in ("1L", "2L") else "k", lw=0.7,
                       label="%s: %s (%d)" % (title[:20], lab, len(sel)))
    ax.legend(fontsize=5, frameon=False)
    if has_xy:
        ax = axs[0, 1]
        for lab, c in REF_COLORS.items():
            sel = [p for p in preds if p["layer"] == lab]
            if sel:
                ax.scatter([p["x"] for p in sel], [p["y"] for p in sel], s=18, marker="s",
                           color=c, lw=0, label="%s (%d)" % (lab, len(sel)))
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.legend(fontsize=5, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(path)


GROUP_STYLE = {                       # (colour, marker)
    "1L hBN": ("#2a8c4a", "o"), "1L RuCl3": ("#8a5a00", "o"),
    "2L hBN": ("#7b3fa0", "D"),
    "2L RuCl3 G-,2D-": ("#d62728", "o"), "2L RuCl3 G+,2D+": ("#1f3fa8", "s"),
    "2L RuCl3": ("#e8402a", "D"),
}


def draw_grid(ax, p=PARAMS, eps_ticks=(-0.1, -0.2, -0.3), n_ticks=(-0.5, -1, -1.5, -2),
              xlim=None, ylim=None):
    """Strain axis (n = 0) and doping axis (ε = 0) as solid lines, iso-strain
    lines dashed (parallel to the doping axis), iso-doping lines dotted
    (parallel to the strain axis), as in Lee et al. 2012."""
    big = 5.0
    e_ext, n_ext = (min(eps_ticks) - 0.1, 0.2), (min(n_ticks) - 1.0, 1.0)
    x, y = zip(_point(p, e_ext[0], 0), _point(p, e_ext[1], 0))
    ax.plot(x, y, color="k", lw=0.8, zorder=1)
    x, y = zip(_point(p, 0, n_ext[0]), _point(p, 0, n_ext[1]))
    ax.plot(x, y, color="k", lw=0.8, zorder=1)
    for e in eps_ticks:
        x, y = zip(_point(p, e, 0), _point(p, e, -big))
        ax.plot(x, y, color="#999", lw=0.6, ls=(0, (4, 3)), zorder=0)
        tx, ty = _point(p, e, 0)
        ax.text(tx - 0.6, ty + 0.8, "%g" % e, fontsize=7, ha="right")
    for n in n_ticks:
        x, y = zip(_point(p, -big / 10, n), _point(p, big / 10, n))
        ax.plot(x, y, color="#999", lw=0.6, ls=(0, (1, 2)), zorder=0)
    ax.plot([p["G0"]], [p["D0"]], marker="s", ms=7, mfc="#4caf50", mec="k", mew=0.6,
            zorder=3)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    # doping labels where the iso-doping lines leave the bottom of the frame
    y0 = ax.get_ylim()[0]
    for n in n_ticks:
        # point on the line ε-parameterised: solve ω2D(ε, n) = y0 + small margin
        e = (y0 + 1.5 - p["D0"] - p["s_doping"] * p["b_G"] * n) / (p["s_strain"] * p["a_G"])
        tx, ty = _point(p, e, n)
        if ax.get_xlim()[0] < tx < ax.get_xlim()[1]:
            ax.text(tx, ty, "%g" % n, fontsize=7, ha="center", va="bottom")
    ax.text(0.98, 0.02, "n (10$^{13}$ cm$^{-2}$)", transform=ax.transAxes, fontsize=7,
            ha="right", va="bottom")
    ax.text(0.98, 0.98, "ε (%)", transform=ax.transAxes, fontsize=8, ha="right", va="top")


def draw_correlation(ax, rows, p=PARAMS):
    """Scatter the correlation points of `rows` by group over the grid."""
    groups = {}
    for r in rows:
        for g, wg, wd in r["points"]:
            # positions outside the physical windows are failed fits
            if PHYS_G[0] < wg < PHYS_G[1] and PHYS_2D[0] < wd < PHYS_2D[1]:
                groups.setdefault(g, []).append((wg, wd))
    for g in sorted(groups, key=lambda k: list(GROUP_STYLE).index(k) if k in GROUP_STYLE else 99):
        c, m = GROUP_STYLE.get(g, ("#777", "o"))
        a = np.array(groups[g])
        ax.scatter(a[:, 0], a[:, 1], s=9, color=c, marker=m, lw=0, alpha=0.85, zorder=2,
                   label="%s (%d)" % (g, len(a)))
    allp = np.array([pt for v in groups.values() for pt in v]) if groups else np.zeros((0, 2))
    # limits from the bulk of the data (1st-99th percentile) and the origin
    if allp.size:
        lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0)
    else:
        lo, hi = np.array([p["G0"], p["D0"]]), np.array([p["G0"], p["D0"]])
    xlim = (min(p["G0"], lo[0]) - 2, max(p["G0"] + 26, hi[0]) + 2)
    ylim = (min(p["D0"], lo[1]) - 4, max(p["D0"] + 58, hi[1]) + 4)
    draw_grid(ax, p, xlim=xlim, ylim=ylim)
    ax.set_xlabel("ω$_G$ (cm$^{-1}$)")
    ax.set_ylabel("ω$_{2D}$ (cm$^{-1}$)")
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")


@styled
def plot_correlation(rows, path, title="", p=PARAMS):
    fig, ax = subplots(1, 1, figsize=(8.7 / 2.54, 8.0 / 2.54))
    draw_correlation(ax, rows, p)
    if title:
        ax.set_title(title, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
