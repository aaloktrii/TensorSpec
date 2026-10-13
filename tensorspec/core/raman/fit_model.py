"""The step-1 fit model (no Qt): Si calibration, background and the Lorentzian bands (window, fixed
or automatic peak count), defined once in 1 · Spectra ▸ Peak fitting and applied unchanged to any
spectrum — a file, a map pixel or a map area (2 · Map).

fit_spectrum() is the per-spectrum calculation that the Peak fitting tab ran itself
(``fit_item`` in Raman_Classifier ``raman_map_classifier_8 (1).py``, Tab 6)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from tensorspec.core.raman import tools

#: (name, from cm-1, to cm-1, peaks: int or "auto", max peaks for "auto")
Band = Tuple[str, float, float, object, int]

DEFAULT_BANDS: List[Band] = [("G", 1500.0, 1680.0, "auto", 3), ("2D", 2560.0, 2810.0, "auto", 4)]


@dataclass
class FitModel:
    bands: List[Band] = field(default_factory=lambda: list(DEFAULT_BANDS))
    background: str = "linear"                      # one of tools.BACKGROUNDS
    bg_opts: dict = field(default_factory=lambda: {"order": 3, "lam": 1e5, "p": 0.01})
    si_calibrate: bool = True                       # shift the axis so Si sits at tools.SI_REF

    def to_dict(self) -> dict:
        return {"bands": [list(b) for b in self.bands], "background": self.background,
                "bg_opts": dict(self.bg_opts), "si_calibrate": bool(self.si_calibrate)}

    @classmethod
    def from_dict(cls, d: dict) -> "FitModel":
        return cls(bands=[tuple(b) for b in d.get("bands", DEFAULT_BANDS)],
                   background=d.get("background", "linear"),
                   bg_opts=dict(d.get("bg_opts", {})),
                   si_calibrate=bool(d.get("si_calibrate", True)))


def fit_spectrum(wn, y, model: FitModel) -> dict:
    """Fit every band of ``model`` on one spectrum.

    Returns {"shift": Si shift applied (cm-1), "si": Si fit or None, "noise": noise level,
    "bands": {name: tools.analyse_band result}}."""
    wn = np.asarray(wn, float)
    y = np.asarray(y, float)
    shift, si = tools.si_shift(wn, y) if model.si_calibrate else (0.0, None)
    w = wn + shift
    nz = tools.noise_level(w, y)
    res = {"shift": shift, "si": si, "noise": nz, "bands": {}}
    for name, lo, hi, n, nmax in model.bands:
        res["bands"][name] = tools.analyse_band(w, y, lo, hi, n=n, nmax=nmax,
                                                bg=model.background,
                                                bg_opts=model.bg_opts, noise=nz)
    return res
