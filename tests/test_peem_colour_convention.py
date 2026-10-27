"""PEEM colour convention, everywhere: positive blue, negative red (plotting/peem_plots.SIGNED_CMAP);
up (+) domains blue, down (-) red."""
import re
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

from tensorspec.plotting import peem_plots

ROOT = Path(__file__).resolve().parents[1] / "tensorspec"
PEEM_FILES = [ROOT / "plotting/peem_plots.py", ROOT / "plotting/ue49_preview.py",
              *sorted((ROOT / "gui/components").glob("peem_*.py")), ROOT / "gui/services/peem_service.py",
              *sorted((ROOT / "gui/services/peem_steps").glob("*.py"))]


def test_signed_maps_are_blue_positive():
    cm = colormaps[peem_plots.SIGNED_CMAP]
    r, g, b, _ = cm(1.0)
    assert b > 0.9 and r < 0.1  # top of the scale (positive) blue
    r, g, b, _ = cm(0.0)
    assert r > 0.9 and b < 0.1  # bottom (negative) red
    for kind in sorted(peem_plots.SIGNED_KINDS):
        assert peem_plots.display_hint(kind, "data")["cmap"] == peem_plots.SIGNED_CMAP, kind


def test_up_domains_blue_down_red():
    ax = Figure().add_subplot()
    peem_plots.domain_step(ax, np.array([[1, -1]]), "three_states")
    rgba = ax.images[0].to_rgba(ax.images[0].get_array())
    assert rgba[0, 0][2] > 0.9 and rgba[0, 1][0] > 0.9  # +1 blue, -1 red


def test_no_red_positive_map_named_in_peem_code():
    """Signed maps take SIGNED_CMAP; no PEEM file names a red-positive diverging map directly."""
    bad = re.compile(r'cmap\s*=\s*"(bwr|RdBu_r|seismic|coolwarm)"|\("bwr"|"RdBu_r"')
    hits = [f"{p.name}:{i}" for p in PEEM_FILES for i, line in enumerate(p.read_text().splitlines(), 1)
            if bad.search(line) and "findText" not in line]
    assert not hits, hits
