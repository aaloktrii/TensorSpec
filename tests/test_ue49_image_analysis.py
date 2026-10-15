"""Ports of the SPEEM Igor image-analysis tools (core/peem/ue49/image_analysis.py)."""
import numpy as np
import pytest

from tensorspec.core.peem import image_analysis as A

YY, XX = np.mgrid[0:200, 0:200].astype(float)


def test_line_profile_length_and_gauss_width():
    img = 5 + 10 * np.exp(-((XX - 100) / 6.0) ** 2)
    pr = A.line_profile(img, (100, 60), (100, 140), halfwidth=3, um_per_px=0.02)
    assert len(pr["values"]) == 81 and pr["unit"] == "µm"  # Igor npixels = round(length) + 1
    g = A.gauss_fit(pr["distance"], pr["values"])
    assert g["x0"] == pytest.approx(0.8, abs=1e-3) and g["width"] == pytest.approx(0.12, abs=1e-3)


def test_edge_fit_and_profile_ignores_nan():
    wall = np.tanh((XX - 100) / 4.0)
    wall[:, :20] = np.nan
    pr = A.line_profile(wall, (50, 70), (50, 130))
    d = A.diff_gauss_fit(pr["distance"], pr["values"])
    assert d["x0"] == pytest.approx(30, abs=0.2) and 6 < d["fwhm"] < 9


def test_fft_period_power_spectrum_and_filter():
    st = np.sin(2 * np.pi * (XX * 0.6 + YY * 0.8) / 16)
    assert A.radial_profile(A.image_fft(st))["period_px"] == pytest.approx(16, rel=0.05)
    ps = A.power_spectrum(A.line_profile(st, (100, 0), (100, 199))["values"])
    assert ps["period_px"] == pytest.approx(16 / 0.6, rel=0.06)
    noisy = st + np.random.default_rng(1).normal(0, 1, st.shape)
    assert np.std(A.fft_filter(noisy, 20, "radial") - st) < 0.4


def test_flatten_and_backgrounds():
    tilt = 3 + 0.05 * XX
    fl, c = A.flatten_line(tilt, (100, 10), (100, 190), halfwidth=5, subtract_offset=True)
    assert c["slope_per_px"] == pytest.approx(0.05, rel=1e-3) and np.abs(fl).max() < 1e-6
    bgimg = 100 + 0.002 * (XX - 100) ** 2 + 0.001 * (YY - 80) ** 2
    assert np.std(A.background_poly(bgimg, 2, 5)[0]) < 0.05
    m = np.zeros_like(bgimg, bool)
    m[::7, ::7] = True
    out, _ = A.remove_background_roi(bgimg, m, 3)
    assert np.std(out) < 1e-6 and out.mean() == pytest.approx(bgimg[m].mean())


def test_histogram_range_and_roi():
    img = np.linspace(0.8, 1.2, 40000).reshape(200, 200)
    h = A.histogram(img, None, 255, (0.9, 1.1))
    assert len(h["counts"]) == 255 and h["outside_range"] > 0
    mask = np.zeros_like(img, bool)
    mask[:10] = True
    assert A.histogram(img, mask)["n"] == 2000
