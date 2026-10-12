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
