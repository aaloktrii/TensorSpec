"""Matplotlib drawing for the PEEM panel map tools (no Qt).

Each function draws one result of the PEEM service onto a matplotlib Axes / Figure the panel owns
(gui/components/peem_map_tools.py keeps the canvas, titles and redraws).
"""
from __future__ import annotations

import numpy as np


SIGNED_KINDS = {"pcd", "pld", "pmld", "pmnld", "domain_map", "compare"}
# PEEM colour convention (every signed map, domain map and export): positive blue, negative red, white at
# the centre; up (+) domains blue, down (-) red. Change it here only.
SIGNED_CMAP = "bwr_r"
SIGNED_VARS = ("", "data", "difference", "flattened", "bg_removed", "fft_filtered", "data_original",
               "map", "reference", "offset_removed")


def display_hint(kind: str, var: str, image: np.ndarray | None = None) -> dict:
    """Colour scale for a shown image: signed maps (asymmetries, domain map) diverging blue-white-red
    with symmetric limits, everything else gray. The centre is 0, or the median when the map
    carries an offset larger than its spread (e.g. data without I0: flux offset between the two
    polarizations). Moved from the XPEEM-map panel (_on_layer_changed)."""
    signed = kind in SIGNED_KINDS and (var or "") in SIGNED_VARS
    center = "0"
    if signed and image is not None and kind != "domain_map":
        fin = np.asarray(image, float)
        fin = fin[np.isfinite(fin)]
        if fin.size and abs(float(np.median(fin))) > float(np.std(fin)):
            center = "median"
    return {"cmap": SIGNED_CMAP if signed else "gray", "symmetric": signed, "center": center}


def color_limits(image: np.ndarray, hint: dict) -> tuple[float, float]:
    """Limits the viewer uses by default: signed maps c +/- the 99th percentile of |v - c|, the
    rest the 1st-99th percentile."""
    fin = np.asarray(image, float)
    fin = fin[np.isfinite(fin)]
    if not fin.size:
        return 0.0, 1.0
    if hint.get("symmetric"):
        c = float(np.median(fin)) if hint.get("center") == "median" else 0.0
        half = float(np.percentile(np.abs(fin - c), 99)) or 1.0
        return c - half, c + half
    lo, hi = (float(v) for v in np.percentile(fin, (1, 99)))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def save_map_png(image: np.ndarray, path, hint: dict, title: str = "", label: str = "") -> None:
    """Colour picture of a map with its colour bar, scaled like the viewer shows it (a float TIFF of
    asymmetry values around +/-0.1 looks blank in ordinary image viewers)."""
    save_map_picture(image, path, hint, title=title, label=label)


def _nice_length(target: float) -> float:
    """1, 2 or 5 x 10^n just below ``target``."""
    if target <= 0:
        return 0.0
    e = np.floor(np.log10(target))
    return max(m for m in (1, 2, 5) if m * 10 ** e <= target) * 10 ** e


def save_map_picture(
    image: np.ndarray,
    path,
    hint: dict,
    *,
    size_px: tuple[int, int] | None = None,
    dpi: int = 150,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    title: str = "",
    label: str = "",
    um_per_px: float | None = None,
    scalebar: bool = False,
    image_only: bool = False,
) -> None:
    """Picture of a map (PNG / PDF / SVG from the file suffix).

    size_px      (width, height) of the saved picture in pixels; None = 6 x 5.2 in at ``dpi`` with the
                 margins trimmed. With image_only the picture is exactly this size (map only, stretched
                 to fill it). PDF / SVG: the page is size_px / dpi inches.
    cmap, vmin, vmax   colour map and limits; None = the viewer's (``hint`` / color_limits)
    scalebar     a micrometre bar in the lower left (needs um_per_px)
    PDF / SVG text stays editable (TrueType), as in save_pdf.
    """
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    img = np.asarray(image, float)
    lo, hi = color_limits(img, hint)
    vmin, vmax = (lo if vmin is None else float(vmin)), (hi if vmax is None else float(vmax))
    if vmax <= vmin:
        raise ValueError("the colour maximum must be above the minimum")
    figsize = (size_px[0] / dpi, size_px[1] / dpi) if size_px else (6, 5.2)
    fig = Figure(figsize=figsize, dpi=dpi, layout="none" if image_only or not size_px else "constrained")
    FigureCanvasAgg(fig)
    if image_only:
        ax = fig.add_axes((0, 0, 1, 1))
        aspect = "auto"
    else:
        ax = fig.add_subplot()
        aspect = "equal"
    im = ax.imshow(img, cmap=cmap or hint.get("cmap", "gray"), vmin=vmin, vmax=vmax, aspect=aspect,
                   interpolation="nearest")
    ax.axis("off")
    if not image_only:
        if title:
            ax.set_title(title, fontsize=9)
        if colorbar:
            fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    if scalebar and um_per_px:
        ny, nx = img.shape
        um = _nice_length(nx * um_per_px / 5)
        if um > 0:
            px = um / um_per_px
            x0, y0 = nx * 0.05, ny * 0.93
            ax.plot([x0, x0 + px], [y0, y0], "-", color="black", lw=3, solid_capstyle="butt")
            ax.text(x0 + px / 2, y0 - ny * 0.02, f"{um:g} µm", ha="center", va="bottom", fontsize=9)
    kw = {} if (size_px or image_only) else {"bbox_inches": "tight"}
    with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        fig.savefig(path, dpi=dpi, **kw)


def line_profile(ax, pr: dict) -> None:
    """Profile A->B (PeemService.line_profile)."""
    ax.plot(pr["distance"], pr["values"], "k-", lw=1)
    ax.set_xlabel(f"distance ({pr['unit']})")


PROFILE_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2",
                  "#7f7f7f", "#bcbd22")


def profile_color(i: int) -> str:
    return PROFILE_COLORS[i % len(PROFILE_COLORS)]


def profile_label(i: int, pr: dict) -> str:
    """'<n>: <kind>/<var> (x0,y0)→(x1,y1)' numbered by the current list position."""
    import re

    return f"{i + 1}: " + re.sub(r"^\d+:\s*", "", str(pr.get("label", "")))


def line_profiles(ax, profiles: list[dict], offset: float = 0.0) -> None:
    """The visible profiles in one plot (each shifted up by its list position x ``offset``), coloured
    like their lines on the map; legend in one column to the right of the axes, so the plot keeps
    its width however many profiles there are."""
    shown = [(i, pr) for i, pr in enumerate(profiles) if pr.get("visible", True)]
    for k, (i, pr) in enumerate(shown):
        ax.plot(pr["distance"], np.asarray(pr["values"], float) + k * offset, "-", lw=1,
                color=profile_color(i), label=profile_label(i, pr))
    if shown:
        ax.set_xlabel(f"distance ({shown[-1][1]['unit']})")
        ax.legend(frameon=False, fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0.0)


def save_profiles_csv(profiles: list[dict], path) -> None:
    """distance / value columns of every profile side by side (blank where a profile is shorter)."""
    import csv

    cols = []
    for i, pr in enumerate(profiles):
        cols += [(f"distance_{i + 1}_{pr['unit']}", list(pr["distance"])), (f"value_{i + 1}", list(pr["values"]))]
    n = max(len(c) for _, c in cols)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([h for h, _ in cols])
        for k in range(n):
            w.writerow([f"{c[k]:.6g}" if k < len(c) else "" for _, c in cols])


def save_pdf(fig, path) -> None:
    """Vector PDF with editable text (TrueType fonts), readable in Adobe Illustrator."""
    import matplotlib

    with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        fig.savefig(path, format="pdf", bbox_inches="tight", transparent=True)


def profile_fit(ax, pr: dict, res: dict, edge: bool) -> None:
    """Gaussian peak fit, or for an edge the derivative + its Gaussian fit on a twin axis."""
    if edge:
        ax2 = ax.twinx()
        ax2.plot(pr["distance"], res["derivative"], color="0.6", lw=0.8)
        ax2.plot(res["x"], res["fit"], "r-", lw=1.2)
        ax2.set_ylabel("d/dx", color="0.5")
    else:
        ax.plot(res["x"], res["fit"], "r-", lw=1.2)


def stack_line_profiles(ax, st: dict) -> None:
    """Profile vs frame (or energy) image (PeemService.stack_line_profiles)."""
    im = st["image"]
    y = st.get("energy")
    ext = [st["distance"][0], st["distance"][-1], len(im) - 0.5, -0.5]
    if y is not None:
        ext = [st["distance"][0], st["distance"][-1], float(y[-1]), float(y[0])]
    fin = im[np.isfinite(im)]
    lo, hi = np.percentile(fin, [1, 99]) if fin.size else (0, 1)
    ax.imshow(im, aspect="auto", extent=ext, cmap="gray", vmin=lo, vmax=hi)
    ax.set_xlabel(f"distance ({st['unit']})")
    ax.set_ylabel("energy (eV)" if y is not None else "frame")


def histogram(ax, h: dict) -> None:
    """Pixel histogram (PeemService.histogram)."""
    ax.bar(h["centers"], h["counts"], width=h["centers"][1] - h["centers"][0], color="0.3")
    ax.set_xlabel("value")
    ax.set_ylabel("pixels")


def image_fft(fig, f: dict) -> None:
    """2-D FFT magnitude and its azimuthal power (PeemService.image_fft); clears ``fig``."""
    rad = f["radial"]
    fig.clear()
    a1 = fig.add_subplot(121)
    ext = [f["fx"][0], f["fx"][-1], f["fy"][-1], f["fy"][0]]
    a1.imshow(np.log10(f["magnitude"] + 1e-12), cmap="gray", extent=ext)
    a1.set_xlabel("fx (1/px)")
    a1.set_ylabel("fy (1/px)")
    a2 = fig.add_subplot(122)
    a2.semilogy(rad["f"][1:], rad["power"][1:], "k-", lw=1)
    a2.axvline(rad["peak_f"], color="r", ls="--", lw=0.8)
    a2.set_xlabel("spatial frequency (1/px)")
    a2.set_ylabel("power (azimuthal mean)")


def power_spectrum(ax, ps: dict) -> None:
    """Power spectrum of a line profile (core.peem.image_analysis.power_spectrum)."""
    ax.plot(ps["f"], ps["magnitude"], "k-", lw=1)
    ax.axvline(ps["peak_f"], color="r", ls="--", lw=0.8)
    ax.set_xlabel("spatial frequency (1/px)")


def drift(ax, d: dict) -> None:
    """Per-frame drift shifts (PeemService.get_drift), polarization changes marked, NCC on a twin axis."""
    f = np.arange(len(d["dy"]))
    ax.plot(f, d["dy"], "o-", ms=3, label="dy")
    ax.plot(f, d["dx"], "s-", ms=3, label="dx")
    pol = d.get("pol")
    if pol:  # frame order of the stack: mark where the polarization changes
        for i in range(1, len(pol)):
            if pol[i] != pol[i - 1]:
                ax.axvline(i - 0.5, color="0.6", ls="--", lw=0.8)
        starts = [0] + [i for i in range(1, len(pol)) if pol[i] != pol[i - 1]]
        for i in starts:
            ax.annotate(pol[i], (i, 1.0), xycoords=("data", "axes fraction"), fontsize=8,
                             va="top", color="0.4")
        ax.set_xlabel("frame")
    else:
        ax.axvline(d["n_a"] - 0.5, color="0.6", ls="--", lw=0.8)
        ax.set_xlabel("frame (polarization A, then B)")
    ax.set_ylabel("shift applied (px)")
    if d.get("score") is not None:
        ax2 = ax.twinx()
        ax2.plot(f, d["score"], color="0.5", lw=0.8, label="NCC")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("template match (NCC)", color="0.5")
    ax.legend(frameon=False, loc="lower right")


def save_three_panel(panels: list[tuple[str, np.ndarray]], path, um_per_px: float | None = None,
                     scale_bar_um: float = 2.0) -> str:
    """Three maps side by side on ONE signed colour scale (SIGNED_CMAP; each centred on its median: offsets
    without I0 are not compared), unlabelled scale bar on the last panel; vector PDF with editable text."""
    import matplotlib
    from matplotlib.figure import Figure

    matplotlib.rcParams["pdf.fonttype"] = 42
    imgs = [(t, np.asarray(a, float)) for t, a in panels]
    lim = max(float(np.nanpercentile(np.abs(a - np.nanmedian(a)), 99.5)) for _, a in imgs) or 1.0
    fig = Figure(figsize=(3.4 * len(imgs), 3.7), layout="tight")
    axes = fig.subplots(1, len(imgs))
    for ax, (t, a) in zip(np.atleast_1d(axes), imgs):
        ax.imshow(a - np.nanmedian(a), cmap=SIGNED_CMAP, vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_axis_off()
        ax.set_title(t, fontsize=8)
    if um_per_px:
        a = imgs[-1][1]
        n = scale_bar_um / um_per_px
        np.atleast_1d(axes)[-1].plot([a.shape[1] * 0.95 - n, a.shape[1] * 0.95], [a.shape[0] * 0.95] * 2,
                                     color="black", lw=3, solid_capstyle="butt")
    fig.savefig(str(path))
    return str(path)


# ------------------------------------------------------------------ spatial domain analysis
UP_COLOR, DOWN_COLOR = "#0000ff", "#ff0000"  # up (+) blue, down (-) red: the ends of SIGNED_CMAP


def domain_step(ax, img: np.ndarray, step: str, um_per_px: float | None = None, title: str = ""):
    """One step of the domain analysis (core/peem/domain_analysis.STEPS) with axes in um (px without a
    pixel size). Returns the image artist (for a colour bar) or None for categorical steps."""
    from matplotlib.colors import ListedColormap

    a = np.asarray(img, float)
    ny, nx = a.shape
    s = um_per_px or 1.0
    ext = [0, nx * s, ny * s, 0]
    kw = {"extent": ext, "interpolation": "nearest"}
    im = None
    if step in ("three_states",):
        ax.imshow(np.where(a == 0, np.nan, a), cmap=ListedColormap([DOWN_COLOR, UP_COLOR]), vmin=-1, vmax=1, **kw)
    elif step in ("up_down", "skeleton"):
        shown = np.where(a == 0, np.nan, a)
        if step == "skeleton":
            ax.set_facecolor("white")
        ax.imshow(shown, cmap=ListedColormap([DOWN_COLOR, UP_COLOR]), vmin=-1, vmax=1, **kw)
    elif step in ("flake", "walls"):
        ax.imshow(a, cmap="gray", vmin=0, vmax=1, **kw)
    elif step in ("asymmetry", "flattened"):
        lim = float(np.nanpercentile(np.abs(a), 99)) or 1.0
        im = ax.imshow(a, cmap=SIGNED_CMAP, vmin=-lim, vmax=lim, **kw)
    elif step == "width_map":
        im = ax.imshow(a, cmap="viridis", **kw)
    elif step == "local_std":
        im = ax.imshow(a, cmap="magma", **kw)
    else:
        im = ax.imshow(a, cmap="gray", **kw)
    unit = "µm" if um_per_px else "px"
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    if title:
        ax.set_title(title, fontsize=8)
    return im


def zoom_to_mask(ax, mask: np.ndarray, um_per_px: float | None = None, margin_um: float = 1.0) -> None:
    """Limit the axes to a mask's bounding box plus a margin (top-left origin kept)."""
    ys, xs = np.nonzero(np.asarray(mask, bool))
    if not ys.size:
        return
    s = um_per_px or 1.0
    pad = margin_um if um_per_px else 20
    ny, nx = np.shape(mask)
    ax.set_xlim(max(xs.min() * s - pad, 0), min((xs.max() + 1) * s + pad, nx * s))
    ax.set_ylim(min((ys.max() + 1) * s + pad, ny * s), max(ys.min() * s - pad, 0))


def pushed_map(ax, img: np.ndarray, look: dict, title: str = ""):
    """A pushed map exactly as it looked in the Map view (colour map incl. invert, colour range, gamma),
    axes in um, with the ROI (yellow) and background region (cyan) outlines. Returns the image artist."""
    from matplotlib import colors as mcolors

    a = np.asarray(img, float)
    um = look.get("um_per_px")
    s = um or 1.0
    ny, nx = a.shape
    vmin, vmax = look.get("vmin"), look.get("vmax")
    if vmin is None or vmax is None:
        fin = a[np.isfinite(a)]
        vmin, vmax = (float(np.percentile(fin, 1)), float(np.percentile(fin, 99))) if fin.size else (0.0, 1.0)
    g = float(look.get("gamma") or 1.0)
    norm = mcolors.Normalize(vmin, vmax) if abs(g - 1.0) < 1e-6 else mcolors.PowerNorm(g, vmin, vmax)
    im = ax.imshow(a, cmap=look.get("cmap") or "gray", norm=norm, extent=[0, nx * s, ny * s, 0],
                   interpolation="nearest")
    xs, ys = (np.arange(nx) + 0.5) * s, (np.arange(ny) + 0.5) * s
    for key, color in (("roi", "yellow"), ("bg", "cyan")):
        m = look.get(key)
        if m is not None and np.any(m):
            ax.contour(xs, ys, np.asarray(m, float), [0.5], colors=color, linewidths=0.8)
    unit = "µm" if um else "px"
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    if title:
        ax.set_title(title, fontsize=8)
    return im


def domain_numbers_text(m: dict) -> str:
    """Short caption of the main numbers of one analysed map."""
    if not m:
        return ""
    u = m.get("width_unit", "px")
    if "n_px" not in m:  # results from before the pixel counts
        return (f"<m_z> {m.get('mz', np.nan):+.3f}\n"
                f"width up / down {m.get(f'width_up_{u}', np.nan):.0f} / {m.get(f'width_down_{u}', np.nan):.0f} {u}   "
                f"period {m.get(f'period_{u}', np.nan):.0f} {u}")
    return (f"{m['n_px']} px: up {m['n_up_px']}, down {m['n_down_px']}, <m_z> {m.get('mz', np.nan):+.3f}   |   "
            f"m0 = {m['m0']:g}: up {m['n_up_px_m0']}, neutral {m['n_neutral_px_m0']}, down {m['n_down_px_m0']}, "
            f"<m_z> {m.get('mz_three_state', np.nan):+.3f}\n"
            f"width up / down {m.get(f'width_up_{u}', np.nan):.0f} / {m.get(f'width_down_{u}', np.nan):.0f} {u}   "
            f"period {m.get(f'period_{u}', np.nan):.0f} {u}")


def domain_sizes(ax, areas: dict) -> None:
    """Domain-size distribution (areas of single up / down domains), log-spaced bins."""
    up, dn = np.asarray(areas["up"], float), np.asarray(areas["down"], float)
    allv = np.concatenate([up, dn])
    allv = allv[allv > 0]
    if not allv.size:
        return
    bins = np.logspace(np.log10(allv.min()), np.log10(allv.max()) + 1e-9, 24)
    ax.hist(up, bins=bins, histtype="step", color=UP_COLOR, label=f"up ({up.size})")
    ax.hist(dn, bins=bins, histtype="step", color=DOWN_COLOR, label=f"down ({dn.size})")
    ax.set_xscale("log")
    unit = "µm²" if areas.get("unit") == "um2" else "px²"
    ax.set_xlabel(f"domain area ({unit})")
    ax.set_ylabel("domains")
    ax.legend(frameon=False, fontsize=7)


def size_fit(ax, fits: dict, quantity: str) -> None:
    """Width or period of both states on one axes: density histograms (outlines, common bins), log-normal (solid) and
    Gaussian (dashed) fits, up blue and down red; AIC / KS p and delta AIC per state in the corner."""
    unit = fits["unit"]
    ax.set_title(f"domain {quantity}: log-normal vs Gaussian", fontsize=8)
    ax.set_xlabel(f"domain {quantity} ({unit})")
    ax.set_ylabel("probability density")
    vals = [np.asarray(fits[quantity][st].get("samples", []), float) for st in ("up", "down")]
    allv = np.concatenate(vals)
    if allv.size < 5:
        ax.text(0.5, 0.5, "not enough samples\n(need >= 5 per state)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="0.4")
        return
    bins = np.linspace(allv.min(), allv.max(), 25)
    txt = []
    for st, col, v in (("up", UP_COLOR, vals[0]), ("down", DOWN_COLOR, vals[1])):
        f = fits[quantity][st]
        if v.size < 5:
            txt.append(f"{st}: too few samples")
            continue
        ax.hist(v, bins=bins, density=True, histtype="step", color=col, lw=0.8, alpha=0.7,
                label=f"{st} ({v.size})")  # outlines: up and down stay readable where they overlap
        if f["lognormal"]:
            ax.plot(f["lognormal"]["pdf_x"], f["lognormal"]["pdf_y"], "-", color=col, lw=1.3, label=f"{st} log-normal")
        if f["gaussian"]:
            ax.plot(f["gaussian"]["pdf_x"], f["gaussian"]["pdf_y"], "--", color=col, lw=1.3, label=f"{st} Gaussian")
        if f["preferred"]:
            txt.append(f"{st}: AIC log-normal {f['lognormal']['aic']:.1f}, Gaussian {f['gaussian']['aic']:.1f}; "
                       f"KS p {f['lognormal']['ks_p']:.3f} / {f['gaussian']['ks_p']:.3f}\n"
                       f"     ΔAIC {f['delta_aic']:+.1f} → {'log-normal' if f['preferred'] == 'lognormal' else 'Gaussian'}")
    ax.text(0.98, 0.98, "\n".join(txt), transform=ax.transAxes, fontsize=5.5, va="top", ha="right",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.legend(frameon=False, fontsize=5.5, loc="center right")


def size_fits(axes, fits: dict) -> None:
    """Two axes: widths, then periods (up and down on each)."""
    for ax, q in zip(axes, ("width", "period")):
        size_fit(ax, fits, q)


def size_fits_text(fits: dict) -> str:
    """One line: delta AIC and the favoured model per quantity and state."""
    parts = []
    for q in ("width", "period"):
        for state in ("up", "down"):
            f = fits[q][state]
            parts.append(f"{q} {state}: " + (f"ΔAIC {f['delta_aic']:+.1f} ({'log-normal' if f['preferred'] == 'lognormal' else 'Gaussian'})"
                                           if f["preferred"] else "too few samples"))
    return "Log-normal vs Gaussian (ΔAIC = AIC Gaussian − AIC log-normal; > 0 favours log-normal, |Δ| > 10 decisive): " \
        + "; ".join(parts)


def save_size_fits_pdf(path, fits: dict, title: str = "") -> None:
    """Editable PDF (TrueType text) of the log-normal vs Gaussian comparison (widths | periods, up blue and
    down red on each), 17.8 cm wide."""
    import matplotlib
    from matplotlib.figure import Figure

    with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                                "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                                "font.size": 7, "axes.linewidth": 0.6}):
        fig = Figure(figsize=(7.0, 3.0), layout="constrained")
        size_fits(fig.subplots(1, 2), fits)
        if title:
            fig.suptitle(title, fontsize=8)
        fig.savefig(path)


def save_domain_step_pdf(path, img: np.ndarray, step: str, um_per_px: float | None, title: str,
                         numbers: dict | None = None, flake: np.ndarray | None = None,
                         look: dict | None = None) -> None:
    """Editable PDF (TrueType text, axes in um) of one domain-analysis step, 8.7 cm wide."""
    import matplotlib
    from matplotlib.figure import Figure

    with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                                "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                                "font.size": 7, "axes.linewidth": 0.6}):
        fig = Figure(figsize=(3.42, 3.2), layout="constrained")
        ax = fig.add_subplot(111)
        if step == "input" and look is not None:
            im = pushed_map(ax, img, look, title)
        else:
            im = domain_step(ax, img, step, um_per_px, title)
            if flake is not None:
                zoom_to_mask(ax, flake, um_per_px)
        if im is not None:
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).outline.set_linewidth(0.5)
        if numbers:
            ax.set_title((title + "\n" if title else "") + domain_numbers_text(numbers), fontsize=6)
        fig.savefig(path)


def save_domain_sizes_pdf(path, areas: dict, title: str = "") -> None:
    import matplotlib
    from matplotlib.figure import Figure

    with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7}):
        fig = Figure(figsize=(3.42, 2.4), layout="constrained")
        ax = fig.add_subplot(111)
        domain_sizes(ax, areas)
        if title:
            ax.set_title(title, fontsize=7)
        fig.savefig(path)
