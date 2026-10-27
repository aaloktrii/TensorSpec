"""Opening NetCDF map files as PEEM datasets (core/io/peem_maps_nc, PeemService.load_maps_nc)."""
import numpy as np
import xarray as xr
from scipy import ndimage

from tensorspec.core.io import peem_maps_nc


def _pattern(n=192, seed=0):
    rng = np.random.default_rng(seed)
    img = ndimage.gaussian_filter(rng.normal(size=(n, n)), 2)
    img[40:120, 30:160] += 1.0
    return img


def _write_analysis(path, xmld, um=0.02):
    xr.Dataset({"xld_110K": (("y", "x"), (0.5 * xmld).astype(np.float32)),
                "xmld": (("y", "x"), xmld.astype(np.float32))},
               attrs={"um_per_px": um}).to_netcdf(path)


def _write_batch(path, n=192):
    lh = 100 + 10 * _pattern(n, 1)
    lv = lh * (1 + 0.02 * np.sign(_pattern(n, 2)))
    yy, xx = np.mgrid[:n, :n]
    roi = ((yy - n / 2) ** 2 + (xx - n / 2) ** 2 < (0.45 * n) ** 2).astype(np.uint8)
    pld_old = (lh - lv) / (lh + lv)  # batch files written before the one-sign rule
    xr.Dataset({"pld": (("y", "x"), pld_old.astype(np.float32)), "mean_LH": (("y", "x"), lh.astype(np.float32)),
                "mean_LV": (("y", "x"), lv.astype(np.float32)), "roi": (("y", "x"), roi)},
               attrs={"um_per_px": 0.0195, "formula": "(LH-LV)/(LH+LV)", "kind": "XLD"}).to_netcdf(path)
    return lh, lv, roi


def test_reader_types_and_names(tmp_path):
    f = tmp_path / "Sample_XMLD.nc"
    _write_analysis(f, _pattern() - 0.2)
    r = peem_maps_nc.read_maps_nc(f)
    assert r["primary"] == "xmld" and r["types"] == {"xld_110K": "signed", "xmld": "signed"}
    assert r["um_per_px"] == 0.02 and r["pair"] is None
    d = tmp_path / "clusters" / "S2_XLD" / "units" / "rep1"
    d.mkdir(parents=True)
    _write_batch(d / "maps.nc")
    r = peem_maps_nc.read_maps_nc(d / "maps.nc")
    assert r["types"]["mean_LH"] == "intensity" and "roi" not in r["maps"] and r["pair"][2] == "xld"
    assert peem_maps_nc.dataset_name(d / "maps.nc") == "S2_XLD_rep1"


def test_service_loads_batch_file_with_sign_and_roi(tmp_path):
    from tensorspec.gui.services.peem_service import PeemService

    s = PeemService()
    f = tmp_path / "batch.nc"
    lh, lv, roi = _write_batch(f)
    out = s.load_maps_nc(f, "nc_batch")
    n = out["name"]
    assert set(s.list_products(n)) >= {"stack", "pld", "nc"} and n in s.list_datasets()
    pld = s.get_map(n, "pld")
    ok = roi.astype(bool)
    assert np.allclose(pld[ok], ((lv - lh) / (lv + lh))[ok], atol=1e-5)  # suite sign (LV-LH)/(LV+LH)
    assert np.isnan(pld[~ok]).all() and "pld" in out["sign_flips"]
    assert s.display_hint(n, "nc", "mean_LH")["cmap"] == "gray"
    assert s.display_hint(n, "nc", "data")["cmap"] == "bwr_r"
    assert [v for _, v in s.nc_outputs(n)][0] == "data"
    st = s.roi_statistics(n, "nc", roi={"kind": "rect", "x0": 60, "x1": 130, "y0": 60, "y1": 130}, threshold=0.0)
    assert set(st["fractions"]) >= {"f_up", "f_down"}
    assert s.histogram(n, "nc", "data", 0)["counts"].sum() > 0


def test_compare_two_loaded_files(tmp_path):
    from tensorspec.gui.services.peem_service import PeemService

    s = PeemService()
    base = _pattern(256, 3)
    shifted = ndimage.shift(base, (4, -6), order=1, mode="nearest")
    _write_analysis(tmp_path / "low.nc", base)
    _write_analysis(tmp_path / "ref.nc", shifted)
    a = s.load_maps_nc(tmp_path / "low.nc", "nc_low")["name"]
    b = s.load_maps_nc(tmp_path / "ref.nc", "nc_ref")["name"]
    assert b in s.compare_candidates(a)
    res = s.compare_maps(a, b, search="none")
    assert res["kind"] == "nc" and res["overlap_r"] > 0.9
    assert abs(res["shift_dy_px"] - (-4)) < 1 and abs(res["shift_dx_px"] - 6) < 1


def test_profiles_remove_and_hide(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from tensorspec.gui.services.peem_service import PeemService
    from tensorspec.plotting import peem_plots

    s = PeemService()
    f = tmp_path / "prof.nc"
    _write_analysis(f, _pattern() - 0.2)
    n = s.load_maps_nc(f, "nc_prof")["name"]
    for a, b in (((20, 20), (150, 150)),) * 3 + (((30, 100), (160, 100)),):
        s.add_profile(n, a, b, kind="nc", var="data")
    assert s.list_profiles(n)[0]["label"].startswith("1: nc/xmld")
    s.remove_profile(n, 2)
    s.remove_profile(n, 1)
    s.set_profile_visible(n, 0, False)
    prof = s.list_profiles(n)
    assert len(prof) == 2 and prof[0]["visible"] is False
    ax = Figure().add_subplot()
    peem_plots.line_profiles(ax, prof)
    assert len(ax.lines) == 1 and ax.get_legend().get_texts()[0].get_text().startswith("2: ")
    assert ax.get_legend()._ncols == 1
