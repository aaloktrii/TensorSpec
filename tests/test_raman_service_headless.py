"""The Raman backend contract: the whole workflow driven through RamanService alone — no Qt, no
widgets — exactly as any frontend (the Qt tabs, a script, a notebook, a web app) would drive it.

Runs on synthetic spectra; set TENSORSPEC_RAMAN_MAP to a one-file-per-spot map folder to also run it
on real data."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

WN = np.linspace(100.0, 3200.0, 3100)


def lorentz(x, x0, w, h):
    return h * (w / 2) ** 2 / ((x - x0) ** 2 + (w / 2) ** 2)


def graphene(split_g=False, w_2d=30.0, seed=0):
    rng = np.random.default_rng(seed)
    g = (lorentz(WN, 1585, 12, 600) + lorentz(WN, 1603, 12, 500)) if split_g else lorentz(WN, 1584, 14, 800)
    si = lorentz(WN, 520.7, 4, 3000)
    return 100 + si + g + lorentz(WN, 2690, w_2d, 1500) + rng.normal(0, 5, WN.size)


@pytest.fixture
def spot_folder(tmp_path):
    """A 4 x 3 map, one file per spot: the right half has a split G."""
    d = tmp_path / "map"
    d.mkdir()
    for j in range(3):
        for i in range(4):
            np.savetxt(d / f"m_{i}_{j}__X_{2.0 * i:.1f}__Y_{2.0 * j:.1f}__T.txt",
                       np.c_[WN, graphene(split_g=i >= 2, seed=10 * j + i)], delimiter="\t")
    return d


def test_no_qt_is_imported_by_the_backend():
    before = {m for m in sys.modules if m.startswith(("PySide6", "PyQt"))}
    import tensorspec.gui.services.raman_service  # noqa: F401
    after = {m for m in sys.modules if m.startswith(("PySide6", "PyQt"))}
    assert after == before


def test_step1_spectra(spot_folder):
    from tensorspec.gui.services.raman_service import RamanService
    svc = RamanService()
    assert svc.add_spectra_folder(str(spot_folder)) == 12
    assert svc.fit_model.bands == []                                  # universal start
    svc.preset("Graphene")
    assert [b[0] for b in svc.fit_model.bands] == ["G", "2D"] and svc.fit_model.si_calibrate
    res = svc.fit_spectrum(0)
    assert res["bands"]["G"]["n"] == 1 and abs(res["shift"]) < 0.5      # Si already at 520.7
    ticks = []
    path = svc.fit_all_spectra(lambda k, n: ticks.append(k) is None)
    assert os.path.exists(path) and ticks[-1] == 12
    names, counts, xy = svc.peak_counts()
    g = dict(zip(names, counts))["G"]
    split = np.array([e["x"] >= 4.0 for e in svc.spectra])
    assert (g[split] == 2).all() and (g[~split] == 1).all()
    assert xy is not None
    # cancelling stops the batch
    assert svc.fit_all_spectra(lambda k, n: k < 3) is None
    # stacked figure list
    svc.stack_add_paths([str(spot_folder)])
    from tensorspec.core.io.raman_loaders import Spectrum
    svc.stack_add(Spectrum(WN, graphene(), "synthetic", ""))
    assert [t["label"] for t in svc.stack_traces()][1] == "synthetic"
    assert svc.stack_move(1, -1) and svc.stack_traces()[0]["label"] == "synthetic"
