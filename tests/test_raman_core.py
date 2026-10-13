"""Raman suite core on synthetic spectra: readers, Voigt / Lorentzian fits, peak counting,
the engine router, /analysis products and the workspace service.

Real-data check (optional): set TENSORSPEC_RAMAN_MAP to a one-file-per-spot map folder."""
from __future__ import annotations

import os

import numpy as np
import pytest

from tensorspec.core.io import raman_loaders
from tensorspec.core.raman import products
from tensorspec.core.raman.constants import default_config
from tensorspec.core.raman_engine import RamanEngineRouter

WN = np.linspace(1200.0, 2900.0, 1700)


def lorentz(x, x0, w, h):
    return h * (w / 2) ** 2 / ((x - x0) ** 2 + (w / 2) ** 2)


def graphene(x0_2d=2690.0, w_2d=30.0, split_g=False, seed=0):
    rng = np.random.default_rng(seed)
    g = (lorentz(WN, 1585, 12, 600) + lorentz(WN, 1603, 12, 500)) if split_g else lorentz(WN, 1584, 14, 800)
    return 100 + g + lorentz(WN, x0_2d, w_2d, 1500) + rng.normal(0, 5, WN.size)


def write_spot_folder(folder, n=(3, 2)):
    folder.mkdir()
    for j in range(n[1]):
        for i in range(n[0]):
            y = graphene(w_2d=28 + 4 * i, seed=10 * j + i)
            np.savetxt(folder / f"map_{i}_{j}__X_{2.0 * i:.1f}__Y_{2.0 * j:.1f}__Time_0.txt",
                       np.c_[WN, y], delimiter="\t")
    return folder


def test_spot_folder_to_tensor(tmp_path):
    kind, mf = raman_loaders.load_raman(str(write_spot_folder(tmp_path / "m")))
    assert kind == "map" and mf.shape == (2, 3) and mf.n_points == 6
    td = raman_loaders.to_tensor(kind, mf)
    assert td.value.shape == (2, 3, WN.size)
    assert td.labels == ["Y", "X", "Raman shift"]
    np.testing.assert_allclose(td.value[1, 2], mf.data[mf.index_map[1, 2]])


def test_single_spectrum(tmp_path):
    p = tmp_path / "spot.txt"
    np.savetxt(p, np.c_[WN, graphene()], delimiter="\t")
    kind, sp = raman_loaders.load_raman(str(p))
    assert kind == "spectrum" and sp.wn.size == WN.size
    assert raman_loaders.to_tensor(kind, sp).value.ndim == 1


def test_router_models():
    eng = RamanEngineRouter()
    cfg = default_config()
    y = graphene(w_2d=30.0)
    m = (WN >= cfg["band_min"]) & (WN <= cfg["band_max"])
    voigt = eng.fit_band("voigt", WN[m], y[m], cfg)
    assert voigt["ok"] and abs(voigt["fwhm"] - 30.0) < 2.0
    lor = eng.fit_band("lorentz_n", WN[m], y[m], {"n": 1})
    assert lor["ok"] and abs(lor["peaks"][0]["fwhm"] - 30.0) < 2.0
    row, G, D = eng.fit_band("g_2d", WN, graphene(split_g=True))
    assert row["G_split"] and abs(row["Gplus_pos"] - 1603) < 1.5
    band = eng.fit_band("lorentz_auto", WN, graphene(split_g=True),
                        dict(lo=1500, hi=1680, n="auto", nmax=3))
    assert band["n"] == 2
    with pytest.raises(ValueError):
        eng.fit_band("nope", WN, y)


def test_products_and_service(tmp_path):
    from tensorspec.core.workspace import global_workspace
    from tensorspec.gui.services.raman_service import RamanService

    svc = RamanService()
    _, mf = raman_loaders.load_raman(str(write_spot_folder(tmp_path / "svc")))
    svc.set_map(mf)
    payload = svc.map_fitter().run()
    assert svc.set_fits(payload) == 6
    tree = global_workspace._data[svc.name]["tree"]
    assert "raman_2d_fit" in tree["analysis"].children
    fwhm = tree["analysis"]["raman_2d_fit"].to_dataset()["data"].values
    assert np.isfinite(fwhm).all()
    ds = products.table_dataset("raman_peakfit", [{"a": 1.0, "s": "x"}, {"a": 2.0, "s": "y"}], "a")
    assert list(ds["data"].values) == [1.0, 2.0]
    assert svc.config_from_json(svc.config_to_json())["fwhm"] == svc.cfg["fwhm"]


@pytest.mark.skipif(not os.environ.get("TENSORSPEC_RAMAN_MAP"), reason="set TENSORSPEC_RAMAN_MAP")
def test_real_map_folder():
    kind, mf = raman_loaders.load_raman(os.path.expanduser(os.environ["TENSORSPEC_RAMAN_MAP"]))
    assert kind == "map" and mf.n_points > 0
    eng = RamanEngineRouter()
    row, _, _ = eng.fit_band("g_2d", mf.axis, mf.data[0].astype(float))
    assert "2D_fwhm" in row


def test_reference_build_goes_to_workspace(tmp_path):
    from tensorspec.core.raman import reference
    project = tmp_path / "TensorSpec_Workspace"
    assert reference.preferred_reference(project) == reference.DEFAULT_REF
    shipped = open(reference.DEFAULT_REF, "rb").read()
    one = write_spot_folder(tmp_path / "one")
    two = write_spot_folder(tmp_path / "two")
    out = reference.user_reference_path(project)
    data = RamanEngineRouter().build_reference([("1L", str(one)), ("2L", str(two))], out)
    assert os.path.exists(out) and len(data["datasets"]) == 2
    assert reference.preferred_reference(project) == out
    assert open(reference.DEFAULT_REF, "rb").read() == shipped       # shipped default untouched


def test_fit_model_matches_band_by_band_fit():
    from tensorspec.core.raman import tools
    from tensorspec.core.raman.fit_model import FitModel, fit_spectrum
    y = graphene(split_g=True)
    model = FitModel(bands=[("G", 1500.0, 1680.0, "auto", 3), ("2D", 2560.0, 2810.0, 1, 4)],
                     background="linear", si_calibrate=False)
    res = fit_spectrum(WN, y, model)
    nz = tools.noise_level(WN, y)
    g = tools.analyse_band(WN, y, 1500.0, 1680.0, n="auto", nmax=3, bg="linear",
                           bg_opts=model.bg_opts, noise=nz)
    assert res["bands"]["G"]["n"] == g["n"] == 2
    assert res["bands"]["2D"]["n"] == 1
    assert FitModel.from_dict(model.to_dict()) == model


def test_area_spectrum_sums_pixels_inside_rectangle(tmp_path):
    from tensorspec.core.raman import roi
    _, mf = raman_loaders.load_raman(str(write_spot_folder(tmp_path / "area")))   # x 0,2,4  y 0,2
    axis, spec, idx = roi.area_spectrum(mf, (1.0, 4.5, -1.0, 1.0), "sum")       # x 2,4 at y 0
    assert sorted(mf.x[idx]) == [2.0, 4.0] and set(mf.y[idx]) == {0.0}
    np.testing.assert_allclose(spec, mf.data[idx].sum(0))
    _, mean, _ = roi.area_spectrum(mf, (4.5, 1.0, 1.0, -1.0), "mean")           # corners in any order
    np.testing.assert_allclose(mean, mf.data[idx].mean(0))
    _, empty, none = roi.area_spectrum(mf, (10, 11, 10, 11))
    assert none.size == 0 and np.isnan(empty).all()


def test_window_integral_is_raw_trapezoid(tmp_path):
    from tensorspec.core.raman import band_maps
    axis = np.linspace(0.0, 100.0, 101)
    data = np.vstack([np.full(101, 3.0), axis])                      # constant 3; I = ω
    v = band_maps.window_integral(axis, data, 20.0, 60.0)
    np.testing.assert_allclose(v, [3.0 * 40.0, (60.0 ** 2 - 20.0 ** 2) / 2])
    np.testing.assert_allclose(band_maps.window_integral(axis, data, 60.0, 20.0), v)   # order-free
    assert np.isnan(band_maps.window_integral(axis, data, 200.0, 300.0)).all()
    _, mf = raman_loaders.load_raman(str(write_spot_folder(tmp_path / "bm")))
    g = band_maps.window_integral_map(mf, 2560.0, 2810.0)
    assert g.shape == mf.shape and np.isfinite(g).all()
    i = mf.index_map[1, 2]
    np.testing.assert_allclose(g[1, 2], band_maps.window_integral(mf.axis, mf.data[i:i + 1], 2560, 2810)[0])


def test_neutral_defaults_and_graphene_preset():
    from tensorspec.core.raman.fit_model import FitModel
    from tensorspec.core.raman.presets import PRESETS
    m = FitModel()
    assert m.bands == [] and m.si_calibrate is False          # universal start: nothing assumed
    g = PRESETS["Graphene"]
    assert [b[0] for b in g["bands"]] == ["G", "2D"] and g["si_calibrate"] is True
