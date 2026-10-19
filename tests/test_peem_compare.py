"""PEEM step 11 (compare with a reference): map_align recovers rotation / scale / shift, and the
service subtracts a registered reference map without Qt."""
import numpy as np
from scipy import ndimage

from tensorspec.core.peem import map_align


def _texture(n=320, seed=0):
    rng = np.random.default_rng(seed)
    img = ndimage.gaussian_filter(rng.normal(size=(n, n)), 3)
    img[90:200, 60:250] += 1.5   # flake-like blocks: topography both images share
    img[140:300, 180:230] += 1.0
    return img


def _moved(img, rot_deg, scale, shift):
    """Moving image whose pixel q shows the reference at p = R(-rot) (q - c)/scale + c - shift."""
    tr = map_align._initial(img.shape, img.shape, scale, rot_deg)
    inv = np.linalg.inv(tr.A)
    c = (np.array(img.shape) - 1) / 2
    return ndimage.affine_transform(img, inv, offset=c - inv @ c - np.asarray(shift), order=1, mode="nearest")


def test_align_recovers_rotation_scale_shift():
    ref = _texture()
    mov = _moved(ref, 6.0, 1.08, (7.0, -5.0))
    roi = np.ones(ref.shape, bool)
    tr, r = map_align.align(ref, mov, roi, roi, search="auto")
    d = tr.describe(((ref.shape[0] - 1) / 2, (ref.shape[1] - 1) / 2))
    assert r > 0.8
    assert abs(d["rotation_deg"] - 6.0) < 0.5
    assert abs(d["scale"] - 1.08) < 0.01
    back = tr.warp(mov)
    ok = np.isfinite(back)
    ok[:30], ok[-30:], ok[:, :30], ok[:, -30:] = False, False, False, False
    assert np.corrcoef(back[ok], ref[ok])[0, 1] > 0.95


def test_align_finds_large_rotation_with_full_search():
    ref = _texture(seed=1)
    mov = _moved(ref, 88.0, 1.0, (0.0, 0.0))
    roi = np.ones(ref.shape, bool)
    tr, r = map_align.align(ref, mov, roi, roi, search="full")
    assert abs(tr.describe((159.5, 159.5))["rotation_deg"] - 88.0) < 1.0 and r > 0.7


def test_service_compare_maps(tmp_path):
    from tensorspec.core.peem import synthetic
    from tensorspec.gui.services.peem_service import PeemService

    mz = synthetic.stripe_domains(96, 96, period_px=24)
    s = PeemService()
    names = []
    for tag, a0, shift in (("low", 0.05, (0.0, 0.0)), ("ref", 0.0, (2.0, -3.0))):
        cp, cm = synthetic.helicity_stacks(np.roll(mz, (int(shift[0]), int(shift[1])), (0, 1)), a0=a0,
                                           n_cp=2, n_cm=2, noise=1.0, shift_cm=(0.0, 0.0))
        d = tmp_path / tag
        d.mkdir()
        pos = synthetic.write_ue49_scan(d, "X_0002", cp, [f"Cr_pos_000{i}" for i in range(2)])
        neg = synthetic.write_ue49_scan(d, "X_0003", cm, [f"Cr_neg_000{i}" for i in range(2)])
        n = s.load_beamline([pos, neg], f"cmp_{tag}", tpx3=False)["name"]
        s.drift_correct(n, {"kind": "rect", "x0": 12, "x1": 84, "y0": 12, "y1": 84}, method="marquee")
        s.stack(n)
        s.asymmetry(n)
        names.append(n)
    low, ref = names
    assert ref in s.compare_candidates(low)
    res = s.compare_maps(low, ref, search="none")
    assert res["kind"] == "pcd" and "compare" in s.list_products(low)
    diff = s.get_map(low, "compare", "data")
    m = s.get_map(low, "compare", "map")
    rr = s.get_map(low, "compare", "reference")
    ok = np.isfinite(diff) & np.isfinite(m) & np.isfinite(rr)
    assert np.allclose(diff[ok], (m - rr)[ok])
    out = s.export_compare_figure(low, tmp_path / "cmp.pdf")
    assert (tmp_path / "cmp.pdf").exists() and out.endswith("cmp.pdf")
