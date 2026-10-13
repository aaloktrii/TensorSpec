"""1L / 2L reference values from labelled Raman datasets, and layer-number /
stacking predictions for new data against them (no Qt).

Building the reference and predicting a folder are in core/raman/batch.py
(build_reference, predict_folder); CLI: scripts/raman/graphene_reference.py.

Reference statistics are kept twice for every dataset:
  all       every spectrum whose G and 2D fits passed
  matching  only the spectra whose fitted 2D width agrees with the dataset's
            label (1L: FWHM < 35 cm-1; 2L: 42-65 cm-1)
A map named after one layer number can contain other thicknesses (the ML and
BL maps here both contain a monolayer strip and a bilayer region), so
predictions use the 'matching' statistics.

Prediction, per spectrum whose fits passed:
  - distance to each reference in units of its standard deviation, over
    2D FWHM and ln I(2D)/I(G); the nearer reference gives 1L or 2L
  - 2D FWHM above the 2L reference (mean + 3 SD): >=3L, with ABA/ABC from
    the Bouhafs et al. (Carbon 2021) 2D-width table
  - 2D FWHM between the 1L and 2L references (beyond 3 SD of both):
    '2L twisted?' - a single, intermediate-width 2D band is typical of
    twisted bilayers, which these references do not contain
  - 1L-like but 2D blue-shifted by more than 3 SD: '1L or twisted 2L'
  - more than MAX_Z SD from both: unassigned
Stacking for 2L inside the reference: 'AB (Bernal)', since the 2L reference
is Bernal bilayer. These are statistical calls against two references from
one flake; check them against optical contrast or AFM.

Moved from Raman_Classifier ``graphene_reference.py`` (statistics and prediction);
its plots are in plotting/backends/raman_matplotlib_engine.py.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
#: Shipped default reference (read-only, part of the package).
DEFAULT_REF = os.path.join(HERE, "data", "graphene_reference.json")
#: Where references built by the user go, under the TensorSpec workspace folder
#: (core/workspace.py project_dir, git-ignored user data), with their per-dataset fit outputs.
USER_REF_SUBDIR = os.path.join("raman", "reference")
REF_NAME = "graphene_reference.json"


def user_reference_path(project_dir):
    """JSON path of a user-built reference inside ``project_dir`` (the workspace folder)."""
    return os.path.join(str(project_dir), USER_REF_SUBDIR, REF_NAME)


def preferred_reference(project_dir):
    """The user-built reference if one exists, otherwise the shipped default."""
    p = user_reference_path(project_dir)
    return p if os.path.exists(p) else DEFAULT_REF

MATCH = {"1L": (0.0, 35.0), "2L": (42.0, 65.0)}       # fitted 2D FWHM, cm-1
FEATURES = ["G_pos", "G_fwhm", "G_height", "2D_pos", "2D_fwhm", "2D_height",
            "I2D_IG", "A2D_AG", "log_I2D_IG", "Gminus_pos", "Gplus_pos"]
MAX_Z = 4.0
#: Bouhafs et al., Carbon 177 (2021) 282, Fig. 2c: 2D FWHM by layer, cm-1.
BOUHAFS = {3: (45, 65), 4: (50, 70), 5: (55, 75), 6: (58, 80), 7: (60, 85),
           8: (63, 88), 9: (65, 90)}



# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def stats(rows):
    out = {}
    for q in FEATURES:
        v = np.array([r.get(q, np.nan) for r in rows], float)
        v = v[np.isfinite(v)]
        out[q] = {"mean": float(v.mean()) if v.size else None,
                  "std": float(v.std(ddof=1)) if v.size > 1 else None,
                  "median": float(np.median(v)) if v.size else None, "n": int(v.size)}
    return out


def _with_log(rows):
    for r in rows:
        x = r.get("I2D_IG", np.nan)
        r["log_I2D_IG"] = float(np.log(x)) if np.isfinite(x) and x > 0 else np.nan
    return rows


def _xy_from_name(name):
    m = re.search(r"__X_([-+\d.]+)__Y_([-+\d.]+)__", name)
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)





def load(path=DEFAULT_REF):
    with open(path) as fh:
        return json.load(fh)


def ref_stats(ref, label, which="matching"):
    for d in ref["datasets"]:
        if d["label"] == label:
            return d[which]
    raise KeyError("no %s reference" % label)


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------

def _z(r, s):
    terms = []
    for q, x in (("2D_fwhm", r["2D_fwhm"]), ("log_I2D_IG", r.get("log_I2D_IG"))):
        m, sd = s[q]["mean"], s[q]["std"]
        if m is None or not sd or x is None or not np.isfinite(x):
            continue
        terms.append(((x - m) / sd) ** 2)
    return float(np.sqrt(sum(terms))) if terms else np.inf


def bouhafs(fwhm, tol=10.0):
    """(N, 'ABA'|'ABC') whose tabulated width is nearest, or (None, None)."""
    best = None
    for n, (aba, abc) in BOUHAFS.items():
        for poly, w in (("ABA", aba), ("ABC", abc)):
            d = abs(fwhm - w)
            if d <= tol and (best is None or d < best[0]):
                best = (d, n, poly)
    return (best[1], best[2]) if best else (None, None)


def predict_row(r, ref):
    """(layer, polytype, note, z_1L, z_2L) for one analyse_spectrum row."""
    if not r.get("ok"):
        return "no fit", "", r.get("note", ""), np.nan, np.nan
    _with_log([r])
    s1, s2 = ref_stats(ref, "1L"), ref_stats(ref, "2L")
    z1, z2 = _z(r, s1), _z(r, s2)
    f = r["2D_fwhm"]
    f1 = s1["2D_fwhm"]["mean"] + 3 * s1["2D_fwhm"]["std"]
    f2lo = s2["2D_fwhm"]["mean"] - 3 * s2["2D_fwhm"]["std"]
    f2hi = s2["2D_fwhm"]["mean"] + 3 * s2["2D_fwhm"]["std"]
    if f > f2hi:
        n, poly = bouhafs(f)
        return (">=3L", ("%dL %s" % (n, poly)) if n else "",
                "2D FWHM %.1f above the 2L reference" % f, z1, z2)
    if f1 < f < f2lo:
        return ("2L twisted?", "",
                "2D FWHM %.1f between the 1L and 2L references" % f, z1, z2)
    if min(z1, z2) > MAX_Z:
        return "unassigned", "", "more than %g SD from both references" % MAX_Z, z1, z2
    if z1 <= z2:
        p1 = s1["2D_pos"]
        if p1["std"] and r["2D_pos"] > p1["mean"] + 3 * p1["std"]:
            return ("1L or twisted 2L", "",
                    "1L-like but 2D blue-shifted to %.1f" % r["2D_pos"], z1, z2)
        return "1L", "—", "", z1, z2
    return "2L", "AB (Bernal)", "", z1, z2
