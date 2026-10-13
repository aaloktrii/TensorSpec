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
