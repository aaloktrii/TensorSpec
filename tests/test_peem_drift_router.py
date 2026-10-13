"""core/peem/drift.align_stack: one entry point over every drift method (no Qt, no workspace)."""
import numpy as np
import pytest

from tensorspec.core.peem import drift, registration, synthetic


@pytest.mark.parametrize("method", sorted(drift.DRIFT_METHODS))
def test_align_stack_every_method_recovers_a_known_shift(method):
    mz = synthetic.stripe_domains(64, 64, period_px=16)
    cp, _ = synthetic.helicity_stacks(mz, a0=0.05, n_cp=1, n_cm=1, noise=0.0)
    base = cp[0]
    moved = registration.apply_shift(base, 0.0, 2.0)  # same contrast, drifted by +2 px in x
    frames = np.stack([base, base, moved, moved])
    pol = ["CP", "CP", "CM", "CM"]
    box = (12, 52, 12, 52) if method in drift.NEEDS_BOX else None
    aligned, shifts, score, extra, used = drift.align_stack(frames, pol, method, box, search_radius=6)
    assert aligned.shape == frames.shape and shifts.shape == (4, 2)
    assert used == (box or used)
    # frames 2-3 drifted by +2 px in x; every method shifts them back by -2 px
    assert np.allclose(shifts[2:, 1], -2.0, atol=0.6), shifts


def test_align_stack_needs_a_box_for_box_methods():
    with pytest.raises(ValueError, match="Draw the marquee"):
        drift.align_stack(np.zeros((2, 8, 8)), ["CP", "CM"], "marquee", None)


def test_stack_means_refines_b_onto_a_across_reversed_contrast():
    mz = synthetic.stripe_domains(64, 64, period_px=16)
    cp, cm = synthetic.helicity_stacks(mz, a0=0.05, n_cp=2, n_cm=2, noise=0.0, shift_cm=(0.0, 1.0))
    from tensorspec.core.peem import dichroism
    mean_a, mean_b, refine = dichroism.stack_means(cp, cm, (16, 48, 16, 48))
    assert mean_a.shape == mean_b.shape == (64, 64)
    assert refine[1] == pytest.approx(-1.0, abs=0.4)


def test_average_by_energy_pairs_separate_sweeps_and_drops_lonely_bins():
    from tensorspec.core.peem.engine import average_by_energy
    frames = np.arange(6, dtype=float)[:, None, None] * np.ones((6, 2, 2))
    pol = ["CP", "CP", "CP", "CM", "CM", "CM"]
    energy = np.array([700.0, 701.0, 702.0, 700.05, 701.05, 705.0])  # CM sweep slightly offset; 702 / 705 unpaired
    stack, tags, points, counts, width = average_by_energy(frames, pol, energy, ("CP", "CM"), energy_bin_eV=1.0)
    assert width == 1.0 and tags == ["CP", "CM", "CP", "CM"]
    assert points == pytest.approx([700.025, 701.025]) and counts == [[1, 1], [1, 1]]
    assert stack[:, 0, 0].tolist() == [0.0, 3.0, 1.0, 4.0]
    even = np.array([700.0, 701.0, 702.0, 700.05, 701.05, 702.05])  # automatic width = the median step
    assert average_by_energy(frames, pol, even, ("CP", "CM"))[4] == pytest.approx(1.0)
    one, _, pts, cnt, w = average_by_energy(frames, pol, None, ("CP", "CM"))  # no energy axis: one bin
    assert len(one) == 2 and pts == [0.0] and cnt == [[3, 3]] and w is None
    with pytest.raises(ValueError, match="no energy has both"):
        average_by_energy(frames[:3], pol[:3], energy[:3], ("CP", "CM"))
