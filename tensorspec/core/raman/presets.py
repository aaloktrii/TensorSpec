"""Material presets (no Qt). The suite starts neutral (no bands, no Si calibration, full-range
figures) so it works for any material; a preset fills in one material's settings in one go."""
from __future__ import annotations

GRAPHENE = {
    # 1 · Spectra ▸ Peak fitting: bands (name, from, to, peaks or "auto", max) and Si calibration
    "bands": [("G", 1500.0, 1680.0, "auto", 3), ("2D", 2560.0, 2810.0, "auto", 4)],
    "si_calibrate": True,
    # 2 · Map ▸ Band integral: windows offered besides the bands above (cm-1)
    "windows": {"D": (1300.0, 1400.0), "Si": (500.0, 540.0)},
    # 1 · Spectra ▸ Stacked spectra: x-axis windows and peak labels
    "stack_windows": "1450-1650, 2550-2800",
    "stack_labels": "G 1583, 2D 2690",
}

PRESETS = {"Graphene": GRAPHENE}
