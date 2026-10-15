import numpy as np
import pytest

from tensorspec.core.peem.ue49 import detector as det


def test_tpx3_assemble_shape_and_uniform_field():
    img = np.full((512, 512), 100.0)
    img[:, 255:257] = 300.0  # double-size pixels collect ~3x
    img[255:257, :] = 300.0
    out = det.tpx3_assemble(img)
    assert out.shape == (514, 514)
    # Igor samples the neighbour band over rows 20..490, which crosses the
    # horizontal chip cross, so the coefficient is slightly below 3
    coef = det.tpx3_cross_coef(img)
    assert coef == pytest.approx(3.0, rel=0.02)
    # after dividing by coef the cross matches the panels
    assert np.allclose(out[100, 255:259], 100.0, rtol=0.02)
    assert np.allclose(out[255:259, 100], 100.0, rtol=0.02)
    assert np.allclose(out[:255, :255], 100.0)


def test_tpx3_orientation_is_a_diagonal_reflection():
    img = np.zeros((512, 512))
    img[10, 400] = 50.0  # top-right chip; column 400 moves to 402 past the cross
    assert det.tpx3_assemble(img, orientation="none")[10, 402] == 50.0
    assert det.tpx3_assemble(img, orientation="transpose")[402, 10] == 50.0
    assert det.tpx3_assemble(img, orientation="antitranspose")[513 - 402, 513 - 10] == 50.0


def test_tpx3_coef_fallback_on_dark_frames():
    assert det.tpx3_cross_coef(np.ones((512, 512))) == det.TPX3_COEF_FALLBACK


def test_dead_pixel_uses_diagonal_neighbours():
    img = np.arange(100, dtype=float).reshape(10, 10)
    dead = np.zeros_like(img)
    dead[5, 5] = 2
    out = det.correct_dead_pixels(img, dead)
    expect = (img[3, 3] + img[7, 7] + img[7, 3] + img[3, 7]) / 4
    assert out[5, 5] == pytest.approx(expect)
    mask = dead == 0
    assert np.array_equal(out[mask], img[mask])


def test_flat_field_divides_by_normalized_flat():
    norm = np.array([[1.0, 3.0], [1.0, 3.0]])
    out = det.flat_field(np.array([[2.0, 6.0], [2.0, 6.0]]), norm)
    assert np.allclose(out, 4.0)


def test_dark_and_offset():
    img = np.full((60, 60), 5.0)
    assert det.dark_counts_at_corners(img) == 5.0
    assert np.allclose(det.subtract_offset(np.array([3.0, 10.0]), 5.0), [1.0, 6.0])


def test_i0_normalize_and_validation():
    stack = np.stack([np.full((2, 2), 2.0), np.full((2, 2), 4.0)])
    out = det.i0_normalize(stack, np.array([1.0, 2.0]))
    assert np.allclose(out[0], out[1])
    with pytest.raises(ValueError):
        det.i0_normalize(stack, np.array([1.0, 0.0]))


def test_bin_and_circle():
    assert det.bin2x2(np.ones((5, 4))).tolist() == [[4.0, 4.0], [4.0, 4.0]]
    m = det.circular_mask(21, 21, 10, 10, 5)
    assert m[10, 10] and not m[0, 0] and m.sum() == pytest.approx(np.pi * 25, rel=0.1)
    c = det.circular_crop(np.ones((21, 21)), 10, 10, 5)
    assert np.isnan(c[0, 0]) and c[10, 10] == 1.0
    lit = np.where(det.circular_mask(200, 200, 99.5, 99.5, 70), 100.0, 1.0)
    cy, cx, r = det.auto_fov_circle(lit)
    assert (cy, cx) == pytest.approx((99.5, 99.5)) and r == pytest.approx(70, abs=6)
    # fully lit frame with a bright flake: inscribed circle, not the flake
    full = np.full((200, 200), 50.0)
    full[20:60, 150:190] = 500.0
    assert det.auto_fov_circle(full)[2] == pytest.approx(100.0)
