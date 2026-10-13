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


def test_step2_map(spot_folder):
    from tensorspec.gui.services.raman_service import RamanService
    svc = RamanService()
    info = svc.load(str(spot_folder))
    assert info["kind"] == "map" and svc.mapfile.shape == (3, 4) and not svc.fits     # nothing fitted
    total = svc.quantity_map("Total intensity")
    assert total.shape == (3, 4) and np.isfinite(total).all()
    assert svc.quantity_map("FWHM(2D)") is None                                   # needs the 2D fit
    g = svc.quantity_map("Band integral", ("G", 1500.0, 1680.0))
    assert np.isfinite(g).all()
    px = svc.pixel(0, 0)
    assert px["fit"] is None and px["inten"].shape == WN.shape
    # area: the two right-hand columns of the middle row (display coordinates)
    mf = svc.mapfile
    cols = mf.ux[2:]
    n = svc.set_area((cols.min() - 0.1, cols.max() + 0.1, mf.uy[1] - 0.1, mf.uy[1] + 0.1))
    assert n == 2 and np.allclose(svc.area_spectrum, mf.data[svc.area_indices].sum(0))
    svc.preset("Graphene")
    res = svc.fit_area(svc.fit_model)
    assert res["bands"]["G"]["n"] == 2                                            # split G
    svc.add_area_to_spectra()
    assert svc.spectra[-1]["label"].endswith("area (2 px)")
    # fit every pixel -> peak maps
    svc.set_peak_results(svc.engine.fit_pixels(mf, svc.fit_model, parallel=False), svc.fit_model)
    counts = svc.quantity_map("Peak map", peak_key="G_n")
    assert set(np.unique(counts)) == {1.0, 2.0}
    # 2D-band Voigt fit of every pixel (what 'Fit 2D band on all pixels' runs)
    assert svc.set_fits(svc.map_fitter().run()) == 12
    assert np.isfinite(svc.quantity_map("FWHM(2D)")).all()
    assert svc.pixel(1, 1)["center"] is not None
    # compare cells
    for idx in (0, 5):
        assert svc.cells_add(idx) == ""
    svc.cells_fit(2550.0, 2850.0, 1)
    assert all(c["fit"][1]["ok"] for c in svc.cells)
    picks, best = svc.cells_suggest(2550.0, 2850.0)
    assert best == 1 and len(picks) == 2


def test_step3_analysis(spot_folder, tmp_path):
    from tensorspec.gui.services.raman_service import RamanService
    svc = RamanService()
    svc.load(str(spot_folder))
    with pytest.raises(ValueError):
        svc.classify()                                            # needs the 2D-band fit first
    svc.set_fits(svc.map_fitter().run())
    svc.cfg["layer_mode"] = "auto"
    notes = svc.classify()
    assert svc.classification["layers"].shape == (3, 4)
    assert "Pixels: 12" in svc.classification_report()
    assert os.path.exists(svc.export_classification_csv(str(tmp_path / "class.csv")))
    fw, am, shown = svc.histogram_data()
    assert fw.size == 12 and shown
    pv = svc.thickness_preview(dict(svc.cfg))
    assert pv["valid"] and sum(pv["counts"].values()) == pv["with_ratio"]
    # reference + prediction
    from tensorspec.core.raman.reference import DEFAULT_REF
    svc.set_reference(DEFAULT_REF)
    preds, title, out = svc.predict_map()
    assert len(preds) == 12 and os.path.exists(os.path.join(out, "predictions.csv"))
    # strain / doping of the step-1 spectra
    svc.add_map_spectra()
    rows = svc.strain_doping(svc.reference)
    assert len(rows) == 12
    # settings round trip
    path = svc.save_config(str(tmp_path / "s.json"))
    assert svc.read_config(path)["band_min"] == svc.cfg["band_min"]
    assert isinstance(notes, list)


@pytest.mark.skipif(not os.environ.get("TENSORSPEC_RAMAN_MAP"), reason="set TENSORSPEC_RAMAN_MAP")
def test_real_map_through_the_service():
    from tensorspec.gui.services.raman_service import RamanService
    svc = RamanService()
    info = svc.load(os.path.expanduser(os.environ["TENSORSPEC_RAMAN_MAP"]))
    assert info["kind"] == "map"
    svc.preset("Graphene")
    mf = svc.mapfile
    l, r, b, t = mf.extent
    assert svc.set_area((l, r, b, t)) == mf.n_points                # the whole map
    assert svc.fit_area(svc.fit_model)["bands"]["G"]["ok"]
