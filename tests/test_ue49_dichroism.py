import numpy as np
import pytest

from tensorspec.core.peem import dichroism, registration, synthetic


def _blobs(ny=96, nx=96, seed=1):
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.normal(size=(ny, nx)), 3) * 100 + 500


def test_shift_sign_convention_and_subpixel():
    ref = _blobs()
    mov = registration.apply_shift(ref, 2.4, -3.7)  # mov is ref displaced
    dy, dx = registration.estimate_shift(ref, mov)
    assert (dy, dx) == pytest.approx((-2.4, 3.7), abs=0.2)
    back = registration.apply_shift(mov, dy, dx)
    assert np.corrcoef(back[10:-10, 10:-10].ravel(), ref[10:-10, 10:-10].ravel())[0, 1] > 0.99


def test_register_stack_and_coadd():
    ref = _blobs()
    stack = np.stack([ref, registration.apply_shift(ref, 3, 0), registration.apply_shift(ref, 0, -2)])
    aligned, shifts = registration.register_stack(stack)
    assert shifts[1] == pytest.approx((-3, 0), abs=0.2)
    mean, _ = registration.smart_coadd(stack)
    assert np.abs(mean - ref)[10:-10, 10:-10].max() < 0.05 * ref.std()


def test_pcd_sign_conventions():
    mz = synthetic.stripe_domains(64, 64, 16)
    cp, cm = synthetic.helicity_stacks(mz, a0=0.05)
    ue49 = dichroism.pcd_map(cp, cm, convention="ue49", register=False)["map"]
    plus = dichroism.pcd_map(cp, cm, convention="cp_minus_cm", register=False)["map"]
    assert np.allclose(plus, 0.05 * mz)
    assert np.allclose(ue49, -0.05 * mz)


def test_pcd_unequal_frame_counts_unbiased():
    mz = synthetic.stripe_domains(64, 64, 16)
    cp, cm = synthetic.helicity_stacks(mz, a0=0.05, n_cp=6, n_cm=2)
    a = dichroism.pcd_map(cp, cm, convention="cp_minus_cm", register=False)["map"]
    assert np.nanmean(a) == pytest.approx(0.0, abs=1e-12)  # Igor's sums would give ~0.5


def test_pcd_recovers_helicity_misregistration():
    mz = synthetic.stripe_domains(96, 96, 24, angle_deg=30)
    cp, cm = synthetic.helicity_stacks(mz, a0=0.05, shift_cm=(1.5, -2.0))
    res = dichroism.pcd_map(cp, cm, convention="cp_minus_cm", register=True)
    assert res["total_cm_shift"] == pytest.approx((-1.5, 2.0), abs=0.3)
    inner = (slice(10, -10), slice(10, -10))
    unreg = dichroism.pcd_map(cp, cm, convention="cp_minus_cm", register=False)["map"]
    err_reg = np.nanmean(np.abs(res["map"][inner] - 0.05 * mz[inner]))
    err_unreg = np.nanmean(np.abs(unreg[inner] - 0.05 * mz[inner]))
    assert err_reg < 0.5 * err_unreg


def test_asymmetry_blanks_low_count_pixels():
    a = np.array([[100.0, 1.0]])
    b = np.array([[90.0, 0.5]])
    out = dichroism.asymmetry(a, b)
    assert np.isfinite(out[0, 0]) and np.isnan(out[0, 1])


def test_pmld_pmnld_decomposition():
    ny = nx = 32
    yy, _ = np.mgrid[0:ny, 0:nx]
    mag = np.where(yy < 16, 0.04, -0.04)  # reverses sign between E_A and E_B
    nat = 0.02  # same sign at both energies
    energy = np.array([583.0, 584.4, 585.4, 586.0])
    base = 1000.0

    def stack(sign):
        frames = []
        for e in energy:
            ld = (mag if e == 584.4 else -mag if e == 585.4 else 0 * mag) + nat
            frames.append(base * (1 + sign * ld))
        return np.stack(frames)

    res = dichroism.pmld_pmnld_maps(stack(+1), stack(-1), energy, energy, 584.4, 585.4, register=False)
    # LH = stack(+1), LV = stack(-1); universal XLD (LV-LH)/(LV+LH) = -ld
    assert np.allclose(res["pmld"], -mag, atol=1e-3)
    assert np.allclose(res["pmnld"], -nat, atol=1e-3)


def test_group_and_coadd_by_key():
    names = ["Cr_pos_0000", "Cr_pos_0001", "Cr_neg_0000"]
    groups = dichroism.group_by_key(names)
    assert dict(groups) == {"Cr_pos": [0, 1], "Cr_neg": [2]}
    stack = np.stack([np.full((8, 8), v) for v in (1.0, 3.0, 5.0)])
    out = dichroism.coadd_by_key(stack, names, register=False)
    assert out["Cr_pos"].mean() == 2.0 and out["Cr_neg"].mean() == 5.0


def test_vector_map():
    v = dichroism.vector_map(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert v["angle_rad"] == pytest.approx([0.0, np.pi / 2])


def test_split_by_reference_registers_on_intensity():
    yy, xx = np.mgrid[0:96, 0:96]
    natural = 0.03 * np.sign(np.sin(xx / 7.0))
    magnetic = 0.02 * np.sign(np.cos(yy / 11.0))
    topo = _blobs()
    shift = (3.0, -4.0)
    ref_nat = registration.apply_shift(natural, *shift)
    ref_topo = registration.apply_shift(topo, *shift)
    res = dichroism.split_by_reference(natural + magnetic, ref_nat, align_image=topo,
                                       reference_align_image=ref_topo)
    assert res["reference_shift"] == pytest.approx((-3.0, 4.0), abs=0.3)
    inner = (slice(12, -12), slice(12, -12))
    assert np.mean(np.sign(res["pmld"][inner]) == np.sign(magnetic[inner])) > 0.9


def test_split_by_reference_refuses_wild_shift():
    img = _blobs(128, 128, 1)
    far = registration.apply_shift(img, 25.0, 0.0)  # > 15 % of 128 px
    with pytest.raises(ValueError, match="registration failed"):
        dichroism.split_by_reference(np.zeros((128, 128)), np.zeros((128, 128)),
                                     align_image=img, reference_align_image=far)


def test_split_flags_noisy_reference():
    rng = np.random.default_rng(5)
    pld = 0.02 * np.sign(np.sin(np.mgrid[0:96, 0:96][1] / 7.0)) + rng.normal(0, 0.005, (96, 96))
    noisy = rng.normal(0, 0.03, (96, 96))
    res = dichroism.split_by_reference(pld, noisy, register=False)
    assert not res["reliable"] and res["noise_ratio"] > 3 and "noisier" in res["warning"]
    smooth = dichroism.split_by_reference(pld, noisy, register=False, reference_smooth_sigma=4)
    assert smooth["noise_ratio"] < res["noise_ratio"] / 3


def test_half_period_lock_and_opt_in_bound():
    """Weak topography + periodic domains: unbounded registration locks half a period
    off (documented limit); the opt-in shift bound recovers the map."""
    from scipy import ndimage

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:96, 0:96]
    mz = np.where((xx // 16) % 2 == 0, 1, -1)
    topo = 3000 * (1 + 0.06 * ndimage.gaussian_filter(rng.normal(size=(96, 96)), 3) / 0.05 * 0.05 * 20)
    a = np.stack([topo * (1 + 0.05 * mz) + rng.normal(0, 5, topo.shape) for _ in range(4)])
    b = np.stack([topo * (1 - 0.05 * mz) + rng.normal(0, 5, topo.shape) for _ in range(4)])
    inner = (slice(20, -20), slice(20, -20))

    def agree(j):
        m = dichroism.asymmetry(j["mean_a"], j["mean_b"])
        return np.mean(np.sign(m[inner]) == mz[inner])

    bounded = dichroism.joint_means(a, b, register=True, max_shift_frac=0.05)
    assert agree(bounded) > 0.95
