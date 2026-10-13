"""Matplotlib drawing for the PEEM suite's XMCD-imaging tab (no Qt).

Each function draws one result of the PEEM service onto a matplotlib Axes / Figure the panel owns
(gui/components/peem_xmcd_panel.py keeps the canvas, titles and redraws). Moved verbatim from the
section mixins in gui/components/peem_tabs/ (analysis, drift).
"""
from __future__ import annotations

import numpy as np


SIGNED_KINDS = {"pcd", "pld", "pmld", "pmnld", "domain_map"}
SIGNED_VARS = ("", "data", "difference", "flattened", "bg_removed", "fft_filtered", "data_original")


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
    return {"cmap": "bwr" if signed else "gray", "symmetric": signed, "center": center}


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


def line_profiles(ax, profiles: list[dict], offset: float = 0.0) -> None:
    """Several profiles in one plot (profile i shifted up by i x ``offset``), coloured like their lines."""
    for i, pr in enumerate(profiles):
        ax.plot(pr["distance"], np.asarray(pr["values"], float) + i * offset, "-", lw=1,
                color=profile_color(i), label=pr.get("label", f"{i + 1}"))
    if profiles:
        ax.set_xlabel(f"distance ({profiles[-1]['unit']})")
    if len(profiles) > 1:
        ax.legend(frameon=False, fontsize=7, ncol=min(len(profiles), 5))


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
