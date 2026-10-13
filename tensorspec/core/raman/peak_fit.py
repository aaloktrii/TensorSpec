"""Lorentzian fits of the graphene G and 2D bands (no Qt).

For each spectrum, G and 2D are fitted separately as a sum of Lorentzians on
a linear baseline (scipy least squares). Reported per band: position, FWHM,
height (intensity) and area, each with its 1-sigma fit error, and R^2.
I(2D)/I(G) is given from heights and from areas. A split G band (G- and G+,
e.g. layer-asymmetrically doped bilayer on RuCl3) is detected automatically
(ng='auto'): two Lorentzians held near 1582 and 1600 cm-1 are used only when
they fit clearly better. For a multi-component band the position is that of
the summed fit's maximum, FWHM its width at half maximum, height its maximum
and area the sum of the components; the components are listed too.

Moved from Raman_Classifier ``raman_peak_fit.py`` (fitting part); the per-folder run is in
core/raman/batch.py, its PDFs in plotting/backends/raman_matplotlib_engine.py, the CLI in scripts/raman/."""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

MIN_R2 = 0.80



def lorentz_sum(x, xc, *p):
    """p = (m, c, x0_1, w_1, h_1, x0_2, ...): linear baseline about xc plus
    Lorentzians of centre x0, FWHM w and height h."""
    y = p[0] * (x - xc) + p[1]
    for k in range(2, len(p), 3):
        x0, w, h = p[k:k + 3]
        y = y + h * (w / 2) ** 2 / ((x - x0) ** 2 + (w / 2) ** 2)
    return y


def fit_band(wn, y, lo, hi, centres=None, halfwin=None, wmax=200.0):
    """Fit Lorentzians on a linear baseline in [lo, hi].

    centres None: one Lorentzian started at the band maximum, free within
    the window. Otherwise one Lorentzian per starting centre; with halfwin
    each centre is held within ±halfwin of its start, so a component cannot
    wander onto its neighbour. Returns a result dict (ok False on failure).
    """
    s = (wn >= lo) & (wn <= hi)
    x, v = wn[s], y[s]
    out = {"ok": False}
    n = 1 if centres is None else len(centres)
    if x.size < 10 + 3 * n:
        out["why"] = "band not covered"
        return out
    xc = 0.5 * (lo + hi)
    base = float(np.median(np.r_[v[:10], v[-10:]]))
    peak = float(x[np.argmax(v)])
    h0 = max(float(v.max()) - base, 1e-9)
    starts = [peak] if centres is None else list(centres)
    p0, lb, ub = [0.0, base], [-np.inf, -np.inf], [np.inf, np.inf]
    for c0 in starts:
        c0 = float(np.clip(c0, lo + 1, hi - 1))
        p0 += [c0, 25.0 if n == 1 else 12.0, h0 / n]
        lb += [max(lo, c0 - halfwin) if halfwin else lo, 2.0, 0.0]
        ub += [min(hi, c0 + halfwin) if halfwin else hi, wmax, np.inf]
    f = lambda xx, *p: lorentz_sum(xx, xc, *p)
    try:
        p, cov = curve_fit(f, x, v, p0=p0, bounds=(lb, ub), maxfev=20000)
    except Exception as exc:
        out["why"] = "fit failed: %s" % str(exc)[:60]
        return out
    err = np.sqrt(np.clip(np.diag(cov), 0, np.inf)) if np.all(np.isfinite(cov)) \
        else np.full(len(p), np.nan)
    resid = v - f(x, *p)
    rss = float(np.sum(resid ** 2))
    r2 = 1 - rss / np.sum((v - v.mean()) ** 2)
    comps = []
    for k in range(n):
        x0, w, h = p[2 + 3 * k:5 + 3 * k]
        ex0, ew, eh = err[2 + 3 * k:5 + 3 * k]
        comps.append(dict(pos=x0, fwhm=w, height=h, area=np.pi * h * w / 2,
                          pos_err=ex0, fwhm_err=ew, height_err=eh,
                          area_err=np.pi / 2 * np.hypot(h * ew, w * eh)))
    comps.sort(key=lambda c: c["pos"])
    if n == 1:
        band = dict(comps[0])
    else:
        # envelope of the components without baseline, on a fine grid
        g = np.linspace(lo, hi, 4000)
        env = lorentz_sum(g, xc, 0.0, 0.0, *p[2:])
        hmax = env.max()
        above = g[env >= hmax / 2]
        band = dict(pos=float(g[np.argmax(env)]),
                    fwhm=float(above.max() - above.min()), height=float(hmax),
                    area=float(sum(c["area"] for c in comps)),
                    pos_err=np.nan, fwhm_err=np.nan, height_err=np.nan,
                    area_err=float(np.sqrt(sum(c["area_err"] ** 2 for c in comps))))
    band.update(ok=bool(r2 >= MIN_R2), r2=float(r2), comps=comps,
                bic=float(x.size * np.log(max(rss, 1e-12) / x.size) + len(p) * np.log(x.size)),
                why="" if r2 >= MIN_R2 else "R2 %.2f < %.2f" % (r2, MIN_R2),
                curve=(x, v, f(x, *p)))
    return band


#: G- / G+ starting centres for a split G band (layer-asymmetric doping,
#: e.g. bilayer graphene on RuCl3), each held within ±GSPLIT_HALFWIN.
GSPLIT_CENTRES = (1582.0, 1600.0)
GSPLIT_HALFWIN = 10.0
#: The split model must lower the BIC by this much, and its weaker component
#: carry at least GSPLIT_MIN_FRAC of the stronger one's height, to be used.
GSPLIT_MIN_DBIC = 10.0
GSPLIT_MIN_FRAC = 0.10
#: ...and the two components must be this far apart (cm-1), neither resting
#: on its ±GSPLIT_HALFWIN limit; otherwise it is one band, not a split.
GSPLIT_MIN_SEP = 8.0


def fit_g(wn, y, lo, hi, ng="auto"):
    """G band as one Lorentzian, or as G- + G+ when ng=2 or when ng='auto'
    and the split model is clearly better (see GSPLIT_*)."""
    one = fit_band(wn, y, lo, hi)
    if ng == 1:
        return one
    two = fit_band(wn, y, lo, hi, GSPLIT_CENTRES, GSPLIT_HALFWIN, wmax=40.0)
    if ng == 2 or not one.get("comps"):
        return two
    if not two.get("comps"):
        return one
    hs = sorted(c["height"] for c in two["comps"])
    c1, c2 = two["comps"]
    on_bound = any(abs(c["pos"] - (c0 + sgn * GSPLIT_HALFWIN)) < 0.5
                   for c, c0 in zip((c1, c2), GSPLIT_CENTRES) for sgn in (-1, 1))
    better = (two["bic"] < one["bic"] - GSPLIT_MIN_DBIC
              and hs[0] >= GSPLIT_MIN_FRAC * hs[1]
              and c2["pos"] - c1["pos"] >= GSPLIT_MIN_SEP and not on_bound)
    return two if better else one


def analyse_spectrum(wn, y, ng="auto", n2d=1, g_range=(1500, 1680),
                     d_range=(2560, 2810)):
    """Fit G and 2D of one spectrum. Returns (row dict, G result, 2D result)."""
    G = fit_g(wn, y, *g_range, ng=ng)
    D = fit_band(wn, y, *d_range) if n2d == 1 else fit_band(
        wn, y, *d_range, centres=list(2690 + np.linspace(-40, 30, n2d)), halfwin=25)
    r = {}
    for tag, b in (("G", G), ("2D", D)):
        for q in ("pos", "fwhm", "height", "area"):
            r["%s_%s" % (tag, q)] = b.get(q, np.nan)
            r["%s_%s_err" % (tag, q)] = b.get(q + "_err", np.nan)
        r["%s_r2" % tag] = b.get("r2", np.nan)
    comps = G.get("comps") or []
    r["G_split"] = len(comps) == 2
    for k, name in enumerate(("Gminus", "Gplus")):
        for q in ("pos", "fwhm", "height"):
            r["%s_%s" % (name, q)] = comps[k][q] if len(comps) == 2 else np.nan
    for k, c in enumerate(D.get("comps") or [] if n2d > 1 else []):
        for q in ("pos", "fwhm", "height", "area"):
            r["2D%d_%s" % (k + 1, q)] = c[q]
    r["I2D_IG"], r["I2D_IG_err"] = ratio(r["2D_height"], r["2D_height_err"],
                                         r["G_height"], r["G_height_err"])
    r["A2D_AG"], r["A2D_AG_err"] = ratio(r["2D_area"], r["2D_area_err"],
                                         r["G_area"], r["G_area_err"])
    r["ok"] = bool(G.get("ok") and D.get("ok"))
    r["note"] = "; ".join(x for x in ("G " + G.get("why", "") if G.get("why") else "",
                                      "2D " + D.get("why", "") if D.get("why") else "") if x)
    return r, G, D


def ratio(a, ea, b, eb):
    r = a / b if b else np.nan
    return r, abs(r) * np.hypot(ea / a if a else np.nan, eb / b if b else np.nan)


QUANTS = [("G_pos", "G position (cm$^{-1}$)"), ("G_fwhm", "G FWHM (cm$^{-1}$)"),
          ("G_height", "G intensity (counts)"), ("2D_pos", "2D position (cm$^{-1}$)"),
          ("2D_fwhm", "2D FWHM (cm$^{-1}$)"), ("2D_height", "2D intensity (counts)"),
          ("I2D_IG", "I(2D)/I(G), heights"), ("A2D_AG", "A(2D)/A(G), areas")]
