import unittest

import numpy as np

from tensorspec.core.data_models import TensorData
from tensorspec.core.peem.engine import drift_correct


def _translated(plane, dx, dy):
    out = np.zeros_like(plane)
    y_src_lo = max(0, -dy)
    y_src_hi = min(plane.shape[0], plane.shape[0] - dy)
    x_src_lo = max(0, -dx)
    x_src_hi = min(plane.shape[1], plane.shape[1] - dx)
    out[
        y_src_lo + dy : y_src_hi + dy,
        x_src_lo + dx : x_src_hi + dx,
    ] = plane[y_src_lo:y_src_hi, x_src_lo:x_src_hi]
    return out


def _feature(ny=32, nx=32):
    plane = np.zeros((ny, nx), dtype=float)
    plane[12:16, 11:14] = 2.0
    plane[17:20, 16:21] = 0.7
    plane[10, 19] = 3.0
    return plane


def _raw_stack(shifts=((0, 0), (3, -2), (-1, 4))):
    base = _feature()
    value = np.stack([_translated(base, dx, dy) for dx, dy in shifts])
    return TensorData(
        value=value,
        axes=[np.arange(len(shifts)), np.arange(32), np.arange(32)],
        labels=["frame", "y", "x"],
        units=["", "px", "px"],
        data_type="Experimental PEEM",
        metadata={
            "pol": ["unknown"] * len(shifts),
            "source": "test",
            "loader": "tif_sequence",
            "csv_attached": True,
            "I0": [1.0] * len(shifts),
        },
    )


ROI = {"kind": "rect", "x0": 7, "y0": 7, "x1": 25, "y1": 25}


class TestDriftCorrect(unittest.TestCase):
    def test_stationary_frame_prefers_zero_when_ncc_scores_tie(self):
        yy, xx = np.mgrid[0:32, 0:32]
        repeated_pattern = ((xx + yy) % 2).astype(float)
        tensor = TensorData(
            value=np.stack([repeated_pattern, repeated_pattern]),
            axes=[np.arange(2), np.arange(32), np.arange(32)],
            labels=["frame", "y", "x"],
            units=["", "px", "px"],
            data_type="Experimental PEEM",
            metadata={"pol": ["unknown", "unknown"]},
        )

        out = drift_correct(
            tensor,
            ref_index=0,
            roi={"kind": "rect", "x0": 4, "y0": 4, "x1": 27, "y1": 27},
            search_radius=2,
        )

        self.assertEqual(out.metadata["drift_shifts"][1], {"index": 1, "dx": 0, "dy": 0})

    def test_recovers_known_integer_shifts_and_metadata(self):
        out = drift_correct(
            _raw_stack(),
            ref_index=0,
            roi=ROI,
            search_radius=8,
        )

        self.assertEqual(
            [(item["dx"], item["dy"]) for item in out.metadata["drift_shifts"]],
            [(0, 0), (-3, 2), (1, -4)],
        )
        self.assertEqual(out.metadata["drift_method"], "ncc_roi")
        self.assertEqual(out.metadata["drift_ref_index"], 0)
        self.assertEqual(out.metadata["drift_roi"], ROI)
        self.assertEqual(out.metadata["drift_search_radius"], 8)
        self.assertEqual(out.metadata["drift_track_channel"], 0)
        self.assertTrue(out.metadata["csv_attached"])
        self.assertEqual(out.metadata["I0"], [1.0, 1.0, 1.0])
        self.assertEqual(out.labels, ["frame", "y", "x"])
        self.assertEqual(out.value.shape, (3, 32, 32))

    def test_paired_estimates_track_channel_and_shifts_both_channels(self):
        base = _feature()
        other = base * 0.25 + 5.0
        cube = np.stack(
            [
                np.stack([base, other]),
                np.stack([_translated(base, 3, 0), _translated(other, 3, 0)]),
            ]
        )
        tensor = TensorData(
            value=cube,
            axes=[np.arange(2), np.arange(2), np.arange(32), np.arange(32)],
            labels=["pair", "channel", "y", "x"],
            units=["", "", "px", "px"],
            data_type="Experimental PEEM (paired)",
            metadata={"channel_tags": ["CP", "CM"], "pair_mode": "CP_CM"},
        )

        out = drift_correct(
            tensor,
            ref_index=0,
            roi=ROI,
            search_radius=6,
            track_channel=0,
        )

        self.assertEqual(out.metadata["drift_shifts"][1], {"index": 1, "dx": -3, "dy": 0})
        np.testing.assert_array_equal(out.value[1, 0, :, :-3], base[:, :-3])
        np.testing.assert_array_equal(out.value[1, 1, :, :-3], other[:, :-3])
        self.assertEqual(out.metadata["channel_tags"], ["CP", "CM"])
        self.assertEqual(out.metadata["pair_mode"], "CP_CM")

    def test_shift_uses_edge_clamp_instead_of_wraparound(self):
        tensor = _raw_stack(shifts=((0, 0), (3, 0)))

        out = drift_correct(tensor, ref_index=0, roi=ROI, search_radius=5)

        self.assertEqual(out.metadata["drift_shifts"][1]["dx"], -3)
        expected_edge = np.repeat(tensor.value[1, :, -1:], 3, axis=1)
        np.testing.assert_array_equal(out.value[1, :, -3:], expected_edge)

    def test_rejects_invalid_reference_index(self):
        with self.assertRaises(ValueError):
            drift_correct(_raw_stack(), ref_index=99, roi=ROI, search_radius=3)

    def test_rejects_search_radius_outside_cap(self):
        for radius in (0, 201):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                drift_correct(_raw_stack(), ref_index=0, roi=ROI, search_radius=radius)

    def test_rejects_roi_with_tiny_bounding_box(self):
        with self.assertRaises(ValueError):
            drift_correct(
                _raw_stack(),
                ref_index=0,
                roi={"kind": "rect", "x0": 4, "y0": 4, "x1": 5, "y1": 5},
                search_radius=3,
            )

    def test_rejects_zero_variance_reference_template(self):
        tensor = _raw_stack()
        with self.assertRaises(ValueError):
            drift_correct(
                tensor,
                ref_index=0,
                roi={"kind": "rect", "x0": 0, "y0": 0, "x1": 4, "y1": 4},
                search_radius=3,
            )

    def test_rejects_wrong_shape_and_track_channel(self):
        wrong_shape = TensorData(
            value=np.zeros((2, 3, 4)),
            axes=[np.arange(2), np.arange(3), np.arange(4)],
            labels=["pair", "y", "x"],
            units=["", "px", "px"],
            data_type="Experimental PEEM",
        )
        with self.assertRaises(ValueError):
            drift_correct(wrong_shape, ref_index=0, roi=ROI, search_radius=3)

        with self.assertRaises(ValueError):
            drift_correct(_raw_stack(), ref_index=0, roi=ROI, search_radius=3, track_channel=1)


def test_fft_ncc_scores_match_shift_by_shift():
    """The FFT score map equals the masked NCC of each shift (edge-clamped), so the fast search
    picks the same shift as trying every shift one by one."""
    from tensorspec.core.peem.engine import _ncc_best_shift, _ncc_score_map

    rng = np.random.default_rng(3)
    base = rng.random((70, 80)) * 1000 + 5000  # large offset: checks the variance stays accurate
    ref, mov = base[10:60, 10:70], base[13:63, 8:68]  # moved by 3 rows, -2 columns: correct with dx=-2, dy=3
    for mask in (np.zeros(ref.shape, bool), None):
        if mask is None:
            yy, xx = np.mgrid[: ref.shape[0], : ref.shape[1]]
            mask = ((yy - 25) / 15) ** 2 + ((xx - 30) / 20) ** 2 < 1  # ellipse
        else:
            mask[12:38, 15:45] = True
        r = 6
        scores = _ncc_score_map(ref, mov, mask, r)
        my, mx = np.nonzero(mask)
        t = ref[mask] - ref[mask].mean()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                c = mov[np.clip(my - dy, 0, 49), np.clip(mx - dx, 0, 59)]
                c = c - c.mean()
                exact = t @ c / (np.linalg.norm(t) * np.linalg.norm(c))
                assert abs(scores[r - dy, r - dx] - exact) < 1e-9
        assert _ncc_best_shift(ref, mov, mask, r) == (-2, 3)


def _sample_with_detector_pattern(shifts, seed=0, n=128):
    """A smooth sample translated by (dx, dy) per frame, times a fixed per-pixel detector gain (a TPX3
    without its flat field): the gain does not move with the sample."""
    from scipy import ndimage
    rng = np.random.default_rng(seed)
    sample = ndimage.gaussian_filter(rng.normal(size=(n, n)), 4)
    sample = sample / sample.std() * 40 + 200  # contrast like the UE49 frames (std ~40 on ~230)
    gain = 1 + 0.15 * rng.normal(size=(n, n))
    frames = [np.roll(sample, (dy, dx), (0, 1)) * gain + rng.normal(0, 2, (n, n)) for dx, dy in shifts]
    return np.stack(frames)


def test_smoothing_finds_drift_under_a_fixed_detector_pattern():
    """Unsmoothed NCC locks on the fixed pixel pattern (shift 0); smooth_px=2 finds the sample's drift."""
    shifts = [(0, 0), (4, -3), (-5, 2)]
    v = _sample_with_detector_pattern(shifts)
    td = TensorData(value=v, axes=[np.arange(3), np.arange(128), np.arange(128)], labels=["frame", "y", "x"],
                    units=["", "px", "px"], data_type="Experimental PEEM", metadata={"pol": ["CP"] * 3})
    roi = {"kind": "rect", "x0": 24, "y0": 24, "x1": 104, "y1": 104}
    plain = drift_correct(td, ref_index=0, roi=roi, search_radius=8).metadata["drift_shifts"]
    smooth = drift_correct(td, ref_index=0, roi=roi, search_radius=8, smooth_px=2.0).metadata["drift_shifts"]
    assert [(s["dx"], s["dy"]) for s in plain] == [(0, 0)] * 3
    assert [(s["dx"], s["dy"]) for s in smooth] == [(-dx, -dy) for dx, dy in shifts]


def test_per_channel_drift_tracks_cp_and_cm_frames_on_their_own():
    """UE49: the CP and CM frames of a pair come from different scans and drift differently; with
    per_channel each frame gets its own shift (the default applies the CP shift to its CM partner)."""
    cp_shifts, cm_shifts = [(0, 0), (2, 1), (3, 2)], [(-4, 3), (-3, 3), (-1, 4)]
    cp, cm = _sample_with_detector_pattern(cp_shifts), _sample_with_detector_pattern(cm_shifts)
    td = TensorData(value=np.stack([cp, cm], 1), axes=[np.arange(3), np.arange(2), np.arange(128), np.arange(128)],
                    labels=["pair", "channel", "y", "x"], units=["", "", "px", "px"],
                    data_type="Experimental PEEM (paired)", metadata={"channel_tags": ["CP", "CM"]})
    roi = {"kind": "rect", "x0": 24, "y0": 24, "x1": 104, "y1": 104}
    out = drift_correct(td, ref_index=0, roi=roi, search_radius=8, smooth_px=2.0, per_channel=True)
    got = [[(c["dx"], c["dy"]) for c in per] for per in out.metadata["drift_channel_shifts"]]
    assert got == [[(-a, -b), (-c, -d)] for (a, b), (c, d) in zip(cp_shifts, cm_shifts)]
    assert [(s["dx"], s["dy"]) for s in out.metadata["drift_shifts"]] == [(-a, -b) for a, b in cp_shifts]
    same = drift_correct(td, ref_index=0, roi=roi, search_radius=8, smooth_px=2.0)
    assert "drift_channel_shifts" not in same.metadata  # default unchanged: one shift per pair
