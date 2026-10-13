import numpy as np
import pytest

from tensorspec.core.peem import domains, roi_stats, synthetic


def test_roi_average_band_and_background():
    img = np.array([[0.11, -0.09, 0.01, 0.10]])
    out = roi_stats.roi_average(img, None, bg=0.01, band=0.05)
    assert out["n_roi"] == 4 and out["n_used"] == 3  # 0.01-0.01=0 dropped
    assert out["mean"] == pytest.approx((0.10 - 0.10 + 0.09) / 3)


def test_background_roi_and_nan_ignored():
    img = np.array([[1.0, 1.2, np.nan], [5.0, 5.0, 5.0]])
    bg = np.array([[True, True, True], [False, False, False]])
    info = roi_stats.background_level(img, bg)
    assert info["bg"] == pytest.approx(1.1) and info["bg_n"] == 2
    new, _ = roi_stats.subtract_background(img, bg)
    assert new[1, 0] == pytest.approx(3.9)


def test_updown_fractions_three_state():
    img = np.array([0.2, 0.2, -0.2, 0.0, 0.01])
    out = roi_stats.updown_fractions(img, None, threshold=0.0, band=0.05)
    assert (out["n_up"], out["n_down"], out["n_wall"]) == (2, 1, 2)
    assert out["f_up"] + out["f_down"] + out["f_wall"] == pytest.approx(1.0)
    assert out["ratio"] == pytest.approx(2.0)
    auto = roi_stats.updown_fractions(img, None)
    assert auto["threshold_source"] == "roi_mean"


def test_domain_map_recovers_stripes():
    mz_true = synthetic.stripe_domains(128, 128, period_px=16, up_fraction=0.5)
    rng = np.random.default_rng(0)
    a = 0.05 * mz_true + rng.normal(0, 0.005, mz_true.shape)
    res = domains.domain_map(a, threshold=0.2, min_px=4, um_per_px=0.02, use_sda=False)
    s = res["stats"]
    assert s["backend"] == "builtin"
    assert np.mean(res["mz"] == mz_true) > 0.97
    assert s["f_up"] == pytest.approx(0.5, abs=0.03)
    assert s["width_px_up"] == pytest.approx(8, rel=0.1)
    assert s["period_px"] == pytest.approx(16, rel=0.1)
    assert s["period_um"] == pytest.approx(0.32, rel=0.1)


def test_domain_map_raw_threshold_and_roi():
    a = np.full((40, 40), 0.03)
    a[:, 20:] = -0.03
    roi = np.zeros_like(a, dtype=bool)
    roi[5:35, 5:35] = True
    res = domains.domain_map(a, roi=roi, threshold=0.01, normalize="none", min_px=0, use_sda=False)
    assert res["stats"]["n_roi"] == 900
    assert res["stats"]["f_up"] == pytest.approx(0.5)
    assert (res["mz"][~roi] == 0).all()
    weak = domains.domain_map(a, threshold=0.05, normalize="none", min_px=0, use_sda=False)
    assert weak["stats"]["f_neutral"] == 1.0


def test_despeckle_removes_small_fragments():
    mz = np.zeros((20, 20), dtype=np.int8)
    mz[2, 2] = 1
    mz[10:16, 10:16] = -1
    out = domains.despeckle(mz, min_px=5)
    assert out[2, 2] == 0 and (out[10:16, 10:16] == -1).all()


def test_smoothing_reduces_speckle():
    mz_true = synthetic.stripe_domains(96, 96, period_px=24)
    rng = np.random.default_rng(1)
    a = 0.02 * mz_true + rng.normal(0, 0.02, mz_true.shape)
    raw = domains.domain_map(a, threshold=0.1, min_px=0, use_sda=False)
    smooth = domains.domain_map(a, threshold=0.1, min_px=0, use_sda=False, smooth_sigma=2.0)
    assert smooth["stats"]["n_domains_up"] < raw["stats"]["n_domains_up"]
    assert np.mean(smooth["mz"] == mz_true) > np.mean(raw["mz"] == mz_true)


def test_sda_required_but_missing(monkeypatch):
    monkeypatch.setattr(domains, "load_sda", lambda: None)
    with pytest.raises(ImportError):
        domains.domain_map(np.ones((10, 10)), use_sda=True)


def test_contrast_snr_white_noise_vs_synthetic_domains():
    # informational metric only; see batch.contrast_snr for why it is not a significance test
    from tensorspec.core.peem.ue49 import batch

    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.02, (160, 160))
    domains_ = 0.02 * synthetic.stripe_domains(160, 160, period_px=40) + noise
    assert batch.contrast_snr(noise) < 2.0
    assert batch.contrast_snr(domains_) > 6.0
