"""The step-1 fit model (no Qt): Si calibration, background and the bands — each a wavenumber
window with its peaks found automatically (count found from the data, or a given count) or placed by
hand (each held within ± centre_tol), and a line shape (Lorentzian, Gaussian, pseudo-Voigt) — defined
once in 1 · Spectra ▸ Peak fitting and applied unchanged to any spectrum: a file, a map pixel or a map
area (2 · Map).

fit_spectrum() is the per-spectrum calculation that the Peak fitting tab ran itself
(``fit_item`` in Raman_Classifier ``raman_map_classifier_8 (1).py``, Tab 6)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from tensorspec.core.raman import tools

#: (name, from cm-1, to cm-1, peaks, max peaks for "auto", shape, manual centres)
#: peaks: "auto" (count found from the data, up to max), an int (that many, found automatically)
#: or "manual" (one peak at each of the manual centres, held within ± FitModel.centre_tol).
Band = Tuple[str, float, float, object, int, str, Tuple[float, ...]]

PEAK_MODES = ("auto", "manual")              # besides a fixed int


def band(name, lo, hi, n="auto", nmax=4, shape="lorentzian", centres=()) -> Band:
    """A band row; 5-field rows (older settings, presets) get a Lorentzian shape and no centres."""
    return (str(name), float(lo), float(hi), n, int(nmax), shape, tuple(float(c) for c in centres))


def as_band(row) -> Band:
    return band(*row)

#: Neutral start: no bands until the user (or a preset, core/raman/presets.py) defines them.
DEFAULT_BANDS: List[Band] = []


@dataclass
class FitModel:
    bands: List[Band] = field(default_factory=lambda: list(DEFAULT_BANDS))
    background: str = "linear"                      # one of tools.BACKGROUNDS
    bg_opts: dict = field(default_factory=lambda: {"order": 3, "lam": 1e5, "p": 0.01})
    si_calibrate: bool = False                      # shift the axis so Si sits at tools.SI_REF
    centre_tol: float = 10.0                        # cm-1: how far a manual peak may move

    def __post_init__(self):
        self.bands = [as_band(b) for b in self.bands]

    def to_dict(self) -> dict:
        return {"bands": [list(b[:6]) + [list(b[6])] for b in self.bands],
                "background": self.background, "bg_opts": dict(self.bg_opts),
                "si_calibrate": bool(self.si_calibrate), "centre_tol": float(self.centre_tol)}

    @classmethod
    def from_dict(cls, d: dict) -> "FitModel":
        return cls(bands=[as_band(b) for b in d.get("bands", DEFAULT_BANDS)],
                   background=d.get("background", "linear"),
                   bg_opts=dict(d.get("bg_opts", {})),
                   si_calibrate=bool(d.get("si_calibrate", False)),
                   centre_tol=float(d.get("centre_tol", 10.0)))


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
    for name, lo, hi, n, nmax, shape, centres in model.bands:
        if n == "manual":
            inside = [c for c in centres if lo < c < hi]
            if not inside:
                res["bands"][name] = {"ok": False, "n": 0, "comps": [], "tried": {},
                                      "why": "no peaks placed in this band"}
                continue
            res["bands"][name] = tools.analyse_band(
                w, y, lo, hi, n=len(inside), bg=model.background, bg_opts=model.bg_opts,
                noise=nz, starts=[(c, 10.0, 0.0) for c in inside], shape=shape,
                centre_tol=model.centre_tol)
        else:
            res["bands"][name] = tools.analyse_band(w, y, lo, hi, n=n, nmax=nmax,
                                                    bg=model.background,
                                                    bg_opts=model.bg_opts, noise=nz, shape=shape)
    return res
