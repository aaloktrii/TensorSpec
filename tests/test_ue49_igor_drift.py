"""Ports of the SPEEM Igor drift routines and the stack clean-up edits."""
import numpy as np
import pytest
from scipy import ndimage

from tensorspec.core.peem import igor_drift as G
from tensorspec.core.peem import stack_edits as E
from tensorspec.core.peem.registration import apply_shift

RNG = np.random.default_rng(3)
BASE = 1000 + 300 * ndimage.gaussian_filter(RNG.normal(size=(192, 192)), 3)
TRUE = np.array([[0, 0], [3, -2], [5.25, -4.5], [7, -6]])
STACK = np.stack([apply_shift(BASE, -a, -b) for a, b in TRUE]) + RNG.normal(0, 2, (4, 192, 192))
BOX = (40, 150, 40, 150)


def test_calc_power2_matches_igor():
    assert G.calc_power2(257) == 256 and G.calc_power2(512) == 256 and G.calc_power2(514) == 512


def test_auto_image_correlation_whole_pixels():
    _, sh = G.stack_auto_correlation(STACK, 0)
    assert np.abs(sh - np.round(TRUE)).max() <= 1.0 and np.all(sh == np.round(sh))


@pytest.mark.parametrize("f, tol", [(1, 1.0), (4, 0.01)])
def test_marquee_drift_correction(f, tol):
    _, sh, passes = G.marquee_drift_correction(STACK, BOX, interp_factor=f)
    assert np.abs(sh - TRUE).max() <= tol and 1 <= passes <= 3


@pytest.mark.parametrize("mode", G.REGISTRATION_MODES)
def test_registration_modes_with_brightness_difference(mode):
    st = STACK.copy()
    st[2:] *= 0.9  # second polarization darker
    _, sh, _ = G.marquee_image_registration(st, BOX, 0, mode)
    assert np.abs(sh - TRUE).max() < 0.1


def test_registration_finds_rotation():
    rot = ndimage.rotate(BASE, 1.5, reshape=False, order=1, mode="nearest")
    _, sh, prm = G.marquee_image_registration(np.stack([BASE, apply_shift(rot, -2, 3)]), BOX, 0, "rotation")
    assert np.degrees(prm[1]["extra"]) == pytest.approx(1.5, abs=0.05)


def test_parse_frames_and_edits():
    assert E.parse_frames("3, 5-7; 9", 10) == [3, 5, 6, 7, 9]
    with pytest.raises(ValueError):
        E.parse_frames("12", 10)
    fr = np.ones((5, 8, 8), np.float32) * np.arange(1, 6)[:, None, None]
    fr[0, 4, 4] = 99
    out, pol, keep = E.apply_edits(fr, list("AABBB"), {"exclude": [1], "pixel_fixes": [[4, 4]],
                                                    "topup": {"factor": 2.0, "frames": [3]}})
    assert keep.tolist() == [0, 2, 3, 4] and pol == list("ABBB")
    assert out[0, 4, 4] == pytest.approx(1.0) and out[2, 0, 0] == pytest.approx(8.0)
    sub, _, _ = E.apply_edits(fr, list("AABBB"), {"subtract": 0})
    assert np.allclose(sub[0], 0)


def test_injection_frames_from_ring_current():
    cur = np.array([300.0, 299.9, 299.8, 300.4, 300.3, 300.2])
    assert E.injection_frames(cur) == [3]
