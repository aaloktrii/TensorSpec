"""Spatial domain analysis (core/peem/domain_analysis.py): the SDA toolkit's pipeline and numbers."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from tensorspec.core.peem import domain_analysis as da


def _labyrinth(n=160, seed=4):
    rng = np.random.default_rng(seed)
    f = ndi.gaussian_filter(rng.normal(size=(n, n)), 6)
    return np.where(f > 0, 0.03, -0.03) + rng.normal(0, 0.006, (n, n))


def _stripes(n=200, period=20, angle_deg=0.0, offset=0.0):
    yy, xx = np.mgrid[:n, :n]
    t = np.radians(angle_deg)
    # stripes run along angle_deg as displayed (rows go down): the wave vector is normal to them
    u = -xx * np.sin(t) - yy * np.cos(t)
    return np.where(np.sin(2 * np.pi * u / period) > 0, 0.02, -0.02) + offset


def test_matches_the_sda_toolkit():
    """Reference numbers from xmcd_toolkit 2026-10 (analyze_xmcd + three_state_field + despeckle_mz +
    domain_widths) on the same synthetic labyrinth, full frame, Gaussian background."""
    a = _labyrinth()
    r = da.analyze(a, um_per_px=0.02, flake="all")
    m = r["metrics"]
    assert m["f_up"] == pytest.approx(0.5535546875)
    assert m["mz"] == pytest.approx(0.107109375)
    assert (m["n_domains_up"], m["n_domains_down"], m["wall_length_px"]) == (7, 7, 1284)
    assert m["width_up_nm"] == pytest.approx(316.732602, abs=1e-5)
    assert m["width_down_nm"] == pytest.approx(249.644753, abs=1e-5)
    assert m["width_skeleton_mean_up_nm"] == pytest.approx(334.593974, abs=1e-5)


def test_stripe_width_period_and_direction():
    for angle in (0.0, 45.0, 90.0):
        r = da.analyze(_stripes(angle_deg=angle), um_per_px=None, flake="all", background="none",
                       denoise="none")
        m = r["metrics"]
        assert m["f_up"] == pytest.approx(0.5, abs=0.03)
        assert m["period_px"] == pytest.approx(20, abs=3)
        assert abs((m["stripe_angle_deg"] - angle + 90) % 180 - 90) < 3, (angle, m["stripe_angle_deg"])
        assert m["alignment"] > 0.9


def test_background_region_sets_the_zero():
    """An offset on the whole map biases a split at the map's own zero; a background region removes it."""
    a = _stripes(offset=0.025)  # larger than the +-0.02 contrast: every pixel reads "up" against 0
    bg = np.zeros(a.shape, bool)
    a[:, :40] = 0.025  # non-magnetic strip at the offset level
    bg[:, :40] = True
    roi = ~bg
    biased = da.analyze(a, roi=roi, background="none", denoise="none")["metrics"]
    fixed = da.analyze(a, roi=roi, bg_mask=bg, background="region", denoise="none")["metrics"]
    assert biased["f_up"] > 0.95
    assert fixed["f_up"] == pytest.approx(0.5, abs=0.03) and fixed["split"] == "zero"
    with pytest.raises(ValueError, match="background region"):
        da.analyze(a, background="region")


def test_as_shown_uses_the_colour_range_and_gamma():
    a = np.linspace(-1, 1, 100).reshape(10, 10)
    img, zero = da.normalize(a, "shown", vmin=-0.5, vmax=0.5, gamma=2.0)
    assert img.min() == 0 and img.max() == 1 and zero == pytest.approx(0.25)
    raw, rzero = da.normalize(a, "raw")
    assert rzero == pytest.approx(0.5, abs=0.01)


def test_auto_flake_and_outline():
    a = np.random.default_rng(0).normal(0, 0.002, (200, 200))
    a[50:150, 40:170] += _stripes(n=200)[50:150, 40:170]
    r = da.analyze(a, flake="auto")
    flake = r["layers"]["flake"].astype(bool)
    assert 0.8 < flake[50:150, 40:170].mean() and flake[:30].mean() < 0.05
    pts = da.flake_outline(flake)
    from tensorspec.core.peem.roi import roi_to_mask

    back = roi_to_mask(200, 200, {"kind": "polygon", "points": pts})
    assert (back == flake).mean() > 0.98
    assert set(r["layers"]) == set(da.STEPS)


def test_auto_flake_ignores_a_drift_strip_at_the_edge():
    """The area left uncovered by the drift shift (a noisy strip at an edge) is not taken as the flake."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.002, (200, 200))
    a[60:140, 70:170] += _stripes(n=200)[60:140, 70:170]
    a[:, :25] = rng.normal(0, 0.05, (200, 25))  # drift strip on the left
    keep = da.edge_strips(a)
    assert not keep[:, :25].any() and keep[:, 40:].all()
    flake = da.analyze(a, flake="auto")["layers"]["flake"].astype(bool)
    assert not flake[:, :30].any() and flake[60:140, 70:170].mean() > 0.8
