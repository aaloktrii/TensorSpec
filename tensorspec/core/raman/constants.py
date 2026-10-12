"""Raman constants and the default analysis config (no Qt): Bouhafs FWHM(2D) table,
acceptance tolerances, class colours, default_config().

Moved from Raman_Classifier ``raman_map_classifier_8 (1).py`` (Constants and default configuration)."""
from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Constants and default configuration
# ---------------------------------------------------------------------------

APP_NAME = "Raman Map Classifier 8"

#: Bouhafs et al., Carbon 177 (2021) 282-290, Fig. 2c - FWHM(2D) in cm-1.
BOUHAFS_FWHM: Dict[int, Dict[str, float]] = {
    3: {"ABA": 45.0, "ABC": 65.0},
    4: {"ABA": 50.0, "ABC": 70.0},
    5: {"ABA": 55.0, "ABC": 75.0},
    6: {"ABA": 58.0, "ABC": 80.0},
    7: {"ABA": 60.0, "ABC": 85.0},
    8: {"ABA": 63.0, "ABC": 88.0},
    9: {"ABA": 65.0, "ABC": 90.0},
}

#: Half-width of the acceptance window around each reference FWHM, in cm-1.
#: A measured width has to land within DEFAULT_TOL of a tabulated value to be
#: called that stacking at all. Ten is a starting point, not a constant.
DEFAULT_TOL = 10.0

DEFAULT_COLORS: Dict[int, Dict[str, str]] = {
    3: {"ABA": "#0000ff", "ABC": "#008000"},   # blue    / green
    4: {"ABA": "#00ffff", "ABC": "#00ff00"},   # cyan    / lime
    5: {"ABA": "#ffa500", "ABC": "#ff0000"},   # orange  / red
    6: {"ABA": "#800080", "ABC": "#ffc0cb"},   # purple  / pink
    7: {"ABA": "#a52a2a", "ABC": "#ffff00"},   # brown   / yellow
    8: {"ABA": "#808080", "ABC": "#ffffff"},   # grey    / white
    9: {"ABA": "#000000", "ABC": "#ff00ff"},   # black   / magenta
}

LAYERS = sorted(BOUHAFS_FWHM.keys())


def default_config() -> dict:
    """Every tunable lives here so edits survive a re-run without reloading."""
    return {
        # Per-layer FWHM(2D) thresholds, acceptance windows and colours
        "fwhm": {k: dict(v) for k, v in BOUHAFS_FWHM.items()},
        "tol": {k: {"ABA": DEFAULT_TOL, "ABC": DEFAULT_TOL} for k in BOUHAFS_FWHM},
        "overlap_policy": "nearest",   # "nearest" | "unclassified"
        "colors": {k: dict(v) for k, v in DEFAULT_COLORS.items()},
        "unclassified_color": "#e8e8e8",
        "unclassified_alpha": 0.35,   # let the optical image read through
        # Fitting
        "band_min": 2600.0,        # cm-1, crop before Voigt fit
        "band_max": 2800.0,        # cm-1
        "engine": "scipy",         # "scipy" (~12x faster) | "lmfit"
        "free_gamma": True,        # let Lorentzian width float independently
        "min_snr": 5.0,            # fitted peak height / residual noise
        "max_fwhm": 120.0,         # cm-1, reject wider "peaks" as junk fits
        "min_fwhm": 5.0,           # cm-1, set from the instrument resolution
        "despike": True,           # remove cosmic rays before fitting
        "despike_sigma": 6.0,      # MAD multiples flagged as a spike
        "smooth_window": 9,        # Savitzky-Golay points for shape metrics
        "parallel": True,          # fit across CPU cores for large maps
        # Asymmetry
        "asym_metric": "hwhm",     # "hwhm" | "area" | "residual"
        "asym_threshold": 0.15,
        "asym_span": 1.5,          # window half-width, in units of the FWHM
        # Classification
        "layer_mode": "auto",      # "auto" | "manual"
        "manual_layer": 3,
        "technique": "both",       # "fwhm" | "asym" | "both"
        # Auto mode = thickness from I(2D)/I(G). Needs both bands in the
        # spectral window, i.e. a wide (600 gr/mm) acquisition.
        "g_band_min": 1500.0,
        "g_band_max": 1680.0,
        "ratio_metric": "height",      # "height" | "area"
        # Descending cutoffs on I(2D)/I(G): above 1_2 -> 1L, above 2_3 -> 2L,
        # above 3_thick -> 3L, otherwise >=4L. Placeholders from the typical
        # literature values (1L ~4, 2L ~2, 3L ~1.1, thick ~0.3); calibrate.
        "ratio_cut": {"1_2": 3.0, "2_3": 1.5, "3_thick": 0.8},
        "thick_policy": "leave",       # "leave" | "assume"
        "thick_assume": 5,             # N used for >=4L pixels if "assume"
        # How the map's two coordinate columns become X/Y in the camera
        # frame. None = take it from the #AxisType headers. The headers have
        # proved unreliable on this instrument, so Auto-align exists.
        "map_transform": None,
        # "instrument": X = −col0, Y = col1 — verified on this LabRAM against
        # LabSpec's own overlay (004_01_8) and independently by Auto-align
        # (R5G_7TRY2). "headers": follow the #AxisType labels, which disagree.
        "orientation_default": "instrument",
        "align_all_orientations": False,     # Auto-align: lock orientation, search offset
        "align_search_um": 15.0,
        "y_down": True,
        "zoom_to_map": True,                 # frame maps on the Raman grid, not the camera                      # draw maps as LabSpec does: +Y downward
        # Layer map: a layer count per region, from AFM / optical contrast.
        "exclude_edges": True,         # drop pixels straddling a thickness step
        "contrast_levels": 2,
        "contrast_cut": 0.03,          # |optical contrast| below this = substrate
        "sio2_offset": 0.5,            # nm, AFM tip/water-layer offset on SiO2
        "ratio_colors": {"1": "#26a69a", "2": "#9575cd", "3": "#ffb300",
                         "thick": "#546e7a"},
        # (restrict_range / range_margin retired: the tolerance windows in
        #  "tol" now decide what counts as in range)
        # Display
        "cmap": "viridis",
        "optical_alpha": 0.5,
    }
