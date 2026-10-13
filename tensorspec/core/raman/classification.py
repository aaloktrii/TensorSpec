"""Layer number and ABA/ABC stacking classification (no Qt): I(2D)/I(G) thickness rules,
FWHM(2D) windows, per-pixel and per-map classification, terrace edges, class labels.

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Classification)."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from tensorspec.core.raman.constants import DEFAULT_TOL

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

#: Sentinel layer number for "four or more layers, ratio cannot say which".
LAYER_THICK = 99


def g_band_cfg(cfg: dict) -> dict:
    """Fit settings for the G band: same quality rules, its own window.

    Ceiling on width is lower than for the 2D band because a G band much
    wider than ~80 cm-1 is not a G band.
    """
    g = dict(cfg)
    g["band_min"] = cfg.get("g_band_min", 1500.0)
    g["band_max"] = cfg.get("g_band_max", 1680.0)
    g["max_fwhm"] = 80.0
    return g


def ratio_applicable(axis: np.ndarray, cfg: dict) -> Tuple[bool, str]:
    """Can this file support thickness from I(2D)/I(G)?

    Decided by spectral coverage, not by the grating named in the header:
    what matters is whether both bands were actually recorded. A high-
    resolution (1800 gr/mm) acquisition centred on the 2D band spans only a
    few hundred cm-1 and never reaches the G band.
    """
    if axis is None or axis.size < 10:
        return False, "no spectral axis"
    lo, hi = float(axis.min()), float(axis.max())
    g_ok = int(((axis >= cfg.get("g_band_min", 1500.0)) &
                (axis <= cfg.get("g_band_max", 1680.0))).sum()) >= 10
    d_ok = int(((axis >= cfg["band_min"]) & (axis <= cfg["band_max"])).sum()) >= 10
    if g_ok and d_ok:
        return True, ("G and 2D bands both inside this file's range "
                      "(%.0f–%.0f cm⁻¹)" % (lo, hi))
    missing = " and ".join(n for n, ok in (("G band", g_ok), ("2D band", d_ok))
                           if not ok)
    return False, ("this file spans %.0f–%.0f cm⁻¹ and contains no %s"
                   % (lo, hi, missing))


def layer_from_ratio(ratio: float, cfg: dict) -> Optional[int]:
    """1, 2, 3 or LAYER_THICK from I(2D)/I(G); None when there is no ratio.

    The ratio falls steeply from monolayer to trilayer and then flattens
    toward the bulk value, so above three layers it stops resolving N at all.
    That is why the last class is ">=4L" rather than 4L, 5L, ...
    """
    if ratio is None or not np.isfinite(ratio) or ratio <= 0:
        return None
    c = cfg["ratio_cut"]
    if ratio > c["1_2"]:
        return 1
    if ratio > c["2_3"]:
        return 2
    if ratio > c["3_thick"]:
        return 3
    return LAYER_THICK


def pixel_ratio(res_2d: Optional[dict], res_g: Optional[dict], metric: str) -> float:
    """I(2D)/I(G) from two accepted fits. Height = fitted peak height above
    baseline; area = the Voigt amplitude, which is the integrated area."""
    if not res_2d or not res_g or not res_2d.get("ok") or not res_g.get("ok"):
        return np.nan
    key = "amplitude" if metric == "area" else "height"
    a, b = res_2d.get(key, np.nan), res_g.get(key, np.nan)
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0:
        return np.nan
    return float(a / b)


def boundary_for(layer: int, cfg: dict) -> float:
    """Midpoint of the two reference widths. Still reported, but it is no
    longer what decides a pixel — see windows_for()."""
    d = cfg["fwhm"][layer]
    return 0.5 * (d["ABA"] + d["ABC"])


def windows_for(layer: int, cfg: dict) -> Tuple[Tuple[float, float],
                                                Tuple[float, float]]:
    """Acceptance windows ((ABA_lo, ABA_hi), (ABC_lo, ABC_hi)) in cm-1.

    A bare midpoint boundary forces every pixel to one side or the other, so a
    width of 20 or 200 cm-1 still comes back labelled with total confidence.
    Requiring the width to land within a tolerance of a tabulated value
    instead means a pixel that matches neither reference is reported as
    matching neither.
    """
    d = cfg["fwhm"][layer]
    t = cfg.get("tol", {}).get(layer, {"ABA": DEFAULT_TOL, "ABC": DEFAULT_TOL})
    ta = abs(float(t.get("ABA", DEFAULT_TOL)))
    tc = abs(float(t.get("ABC", DEFAULT_TOL)))
    return ((d["ABA"] - ta, d["ABA"] + ta), (d["ABC"] - tc, d["ABC"] + tc))


def width_call(fwhm: float, layer: int, cfg: dict) -> Tuple[Optional[str], str]:
    """Stacking from the 2D width alone. Returns (call, reason).

    Three outcomes rather than two:
      - inside exactly one window            -> that stacking
      - inside neither                       -> None, "outside both windows"
      - inside both (the windows overlap)    -> settled by overlap_policy

    Set each tolerance to half the gap between the two reference values and
    the windows meet exactly at the midpoint, which reproduces the old
    boundary rule everywhere except far from the table, where it now abstains.
    """
    if not np.isfinite(fwhm):
        return None, "no fit"
    (a0, a1), (c0, c1) = windows_for(layer, cfg)
    in_a = a0 <= fwhm <= a1
    in_c = c0 <= fwhm <= c1
    if in_a and not in_c:
        return "ABA", ""
    if in_c and not in_a:
        return "ABC", ""
    if in_a and in_c:
        if cfg.get("overlap_policy", "nearest") == "unclassified":
            return None, "in the ABA/ABC window overlap"
        d = cfg["fwhm"][layer]
        # Equidistant goes to ABC, matching the published rule that the
        # midpoint itself counts as the upper class.
        nearer = "ABA" if abs(fwhm - d["ABA"]) < abs(fwhm - d["ABC"]) else "ABC"
        return nearer, ""
    return None, "%.1f outside both %dL windows (%.1f-%.1f, %.1f-%.1f)" % (
        fwhm, layer, a0, a1, c0, c1)


def candidate_layers(fwhm: float, cfg: dict) -> List[int]:
    """Every layer with a window this width could belong to.

    The union of the ABA and ABC windows, not the span between them: with a
    tolerance narrower than half the gap there is a genuine dead zone in the
    middle that belongs to no layer.
    """
    if not np.isfinite(fwhm):
        return []
    out = []
    for layer in sorted(cfg["fwhm"]):
        (a0, a1), (c0, c1) = windows_for(layer, cfg)
        if (a0 <= fwhm <= a1) or (c0 <= fwhm <= c1):
            out.append(layer)
    return out


def auto_layer(fwhm: float, cfg: dict) -> Tuple[Optional[int], int]:
    """Assign a layer number from the 2D-band width alone.

    Returns (layer, n_candidates). Still under-determined, and widening the
    tolerances widens the overlap: the windows already overlap heavily at zero
    tolerance, so n_candidates is the number to watch. The tie-break follows
    cfg["auto_ambiguity"]: strict refuses to guess, nearest takes the closest
    reference value, lowest takes the smallest layer that fits.

    Where the layer count is known from AFM, optical contrast or a step edge,
    Manual mode remains the sounder choice.
    """
    cands = candidate_layers(fwhm, cfg)
    n = len(cands)
    if n == 0:
        return None, 0
    if n == 1:
        return cands[0], 1
    mode = cfg.get("auto_ambiguity", "nearest")
    if mode == "strict":
        return None, n
    if mode == "lowest":
        return cands[0], n

    def nearest_ref(layer):
        d = cfg["fwhm"][layer]
        return min(abs(fwhm - d["ABA"]), abs(fwhm - d["ABC"]))
    return min(cands, key=nearest_ref), n


def classify_pixel(fwhm: float, asym: float, cfg: dict, ratio: float = np.nan,
                   n_here: int = 0) -> Tuple[Optional[int], Optional[str], int, str]:
    """Return (layer, 'ABA'|'ABC'|'N/A'|None, n_candidate_layers, reason).

    'N/A' means the layer count is known but stacking is not defined for it:
    ABA versus ABC only exists from three layers up, and ">=4L" from the
    ratio carries no specific N to look up in the width table.
    """
    if not np.isfinite(fwhm):
        return None, None, 0, "no fit"

    if cfg["layer_mode"] == "manual":
        layer = int(cfg["manual_layer"])
        ncand = len(candidate_layers(fwhm, cfg))
        if layer not in cfg["fwhm"]:
            return None, None, ncand, "layer not in table"
    elif cfg["layer_mode"] == "map":
        ncand = 0
        if n_here <= 0:
            return None, None, 0, "no layer count assigned in the layer map"
        if n_here in (1, 2):
            return int(n_here), "N/A", 0, ""
        layer = int(n_here)
        if layer not in cfg["fwhm"]:
            return layer, None, 0, "%dL has no row in the width table" % layer
    else:
        ncand = 0
        n = layer_from_ratio(ratio, cfg)
        if n is None:
            return None, None, 0, "no I(2D)/I(G) (G band not fitted)"
        if n in (1, 2):
            return n, "N/A", 0, ""
        if n == LAYER_THICK:
            if cfg.get("thick_policy", "leave") != "assume":
                return LAYER_THICK, "N/A", 0, ""
            layer = int(cfg.get("thick_assume", 5))
            if layer not in cfg["fwhm"]:
                return LAYER_THICK, "N/A", 0, "assumed N not in table"
        else:
            layer = 3

    call_w, why_w = width_call(fwhm, layer, cfg)
    call_a = None
    if np.isfinite(asym):
        call_a = "ABC" if asym >= cfg["asym_threshold"] else "ABA"

    technique = cfg["technique"]
    if technique == "fwhm":
        return layer, call_w, ncand, why_w
    if technique == "asym":
        return layer, call_a, ncand, "" if call_a else "no asymmetry value"
    # "both": AND logic - the two techniques must agree
    if call_w is None:
        return layer, None, ncand, why_w
    if call_a is None:
        return layer, None, ncand, "no asymmetry value"
    if call_w != call_a:
        return layer, None, ncand, "width says %s, asymmetry says %s" % (call_w, call_a)
    return layer, call_w, ncand, ""


def terrace_edges(n_map: np.ndarray) -> np.ndarray:
    """Pixels with a 4-neighbour of a *different* assigned layer count.

    At a thickness step the laser spot straddles two terraces and records a
    sum of two 2D bands fitted as one. Its layer count is half wrong by
    construction, and when the two terraces differ enough in width the
    broadened sum can also be misread as ABC.
    Only steps between assigned layers count: at the flake's outer edge the
    spot's other half is bare substrate, which adds no 2D band at all.
    """
    edge = np.zeros(n_map.shape, dtype=bool)
    a = n_map
    for sl_a, sl_b in (((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                       ((slice(None), slice(1, None)), (slice(None), slice(None, -1)))):
        x, y = a[sl_a], a[sl_b]
        diff = (x > 0) & (y > 0) & (x != y)
        edge[sl_a] |= diff
        edge[sl_b] |= diff
    return edge


def classify_map(fwhm_map: np.ndarray, asym_map: np.ndarray, cfg: dict,
                 ratio_map: Optional[np.ndarray] = None,
                 n_map: Optional[np.ndarray] = None):
    ny, nx = fwhm_map.shape
    if ratio_map is None:
        ratio_map = np.full((ny, nx), np.nan)
    if n_map is None:
        n_map = np.zeros((ny, nx), dtype=int)
    edge = (terrace_edges(n_map) if cfg["layer_mode"] == "map"
            and cfg.get("exclude_edges", True) else np.zeros((ny, nx), dtype=bool))
    layers = np.full((ny, nx), -1, dtype=int)
    ncands = np.zeros((ny, nx), dtype=int)
    calls = np.empty((ny, nx), dtype=object)
    reasons = np.empty((ny, nx), dtype=object)
    for r in range(ny):
        for c in range(nx):
            if edge[r, c] and np.isfinite(fwhm_map[r, c]):
                lay, call, nc, why = (int(n_map[r, c]), None, 0,
                                      "on a terrace edge (spot spans two thicknesses)")
            else:
                lay, call, nc, why = classify_pixel(fwhm_map[r, c], asym_map[r, c],
                                                    cfg, ratio_map[r, c],
                                                    int(n_map[r, c]))
            layers[r, c] = -1 if lay is None else lay
            calls[r, c] = call
            ncands[r, c] = nc
            reasons[r, c] = why
    return layers, calls, ncands, reasons


def label_of(layer: int, call: Optional[str]) -> str:
    if layer == LAYER_THICK:
        return "≥4L (thick)"
    if layer < 0 or call is None:
        return "Unclassified"
    if call == "N/A":
        return "%dL" % layer
    return "%dL-%s" % (layer, call)
