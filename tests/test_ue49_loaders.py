import numpy as np
import pytest

from tensorspec.core.peem.bg import resolve_energy
from tensorspec.core.io import ue49_loaders as loaders
from tensorspec.core.peem import synthetic


def _frames(n, val=100.0, shape=(16, 16)):
    return np.full((n,) + shape, val)


def test_load_xmcd_scans_filename_beats_lagging_poltable(tmp_path):
    # mirrors June 2025 X_0002..X_0004: PolTable lags the helicity switch
    p2 = synthetic.write_ue49_scan(tmp_path, "X_0002", _frames(2), ["Cr_pos_0000", "Cr_pos_0001"], pol_table=2)
    p3 = synthetic.write_ue49_scan(tmp_path, "X_0003", _frames(3), [f"Cr_neg_000{i}" for i in range(3)], pol_table=2)
    td = loaders.load_scans([p2, p3], tpx3=False)
    assert td.value.shape == (5, 16, 16)
    assert td.labels == ["frame", "y", "x"]
    assert td.metadata["pol"] == ["CP", "CP", "CM", "CM", "CM"]
    assert td.metadata["pol_conflicts"]  # X_0003 names disagree with PolTable 2
    assert td.metadata["fov_um"] == 10.0
    assert td.metadata["um_per_px"] == pytest.approx(10.0 / 15)


def test_poltable_used_when_names_have_no_token(tmp_path):
    energy = np.linspace(570, 590, 4)
    p = synthetic.write_ue49_scan(tmp_path, "Mono_S_0001", _frames(4), [f"Img_000{i}" for i in range(4)],
                                  energy=energy, pol_table=1)
    td = loaders.load_scans([p], tpx3=False)
    assert set(td.metadata["pol"]) == {"LV"}
    e, src = resolve_energy(4, td.metadata)
    assert src == "csv" and np.allclose(e, energy)


def test_i0_normalization(tmp_path):
    frames = np.stack([np.full((8, 8), 100.0), np.full((8, 8), 200.0)])
    p = synthetic.write_ue49_scan(tmp_path, "X_0001", frames, ["a_pos_0000", "a_pos_0001"],
                                  i0=np.array([1e-9, 2e-9]))
    td = loaders.load_scans([p], tpx3=False)
    assert td.metadata["i0_normalized"]
    assert np.allclose(td.value[0], td.value[1])
    raw = loaders.load_scans([p], tpx3=False, i0_normalize=False)
    assert raw.value[1].mean() == 200.0


def test_mixed_fov_refused(tmp_path):
    a = synthetic.write_ue49_scan(tmp_path, "Mono_S_0199", _frames(2), ["Img_0000", "Img_0001"], fov=15, pol_table=1)
    b = synthetic.write_ue49_scan(tmp_path, "Mono_S_0201", _frames(2), ["Img_0000", "Img_0001"], fov=10, pol_table=0)
    with pytest.raises(ValueError, match="fields of view"):
        loaders.load_scans([a, b], tpx3=False)
    td = loaders.load_scans([a, b], tpx3=False, allow_mixed_fov=True)
    assert td.metadata["fov_um"] is None


def test_drop_bad_frames(tmp_path):
    # Igor's spike rule is |jump| > 5x mean jump on both sides, so it needs a
    # series long enough that one spike does not dominate the mean
    n = 30
    i0 = np.ones(n) * 1e-9
    i0[10] = 5e-9  # spike
    hstab = np.zeros(n)
    hstab[20] = 0.2  # unstable beam
    names = [f"x_pos_{i:04d}" for i in range(n)]
    p = synthetic.write_ue49_scan(tmp_path, "X_0009", _frames(n), names, i0=i0, hstab=hstab)
    assert np.flatnonzero(loaders.flag_bad_frames(i0, hstab)).tolist() == [10, 20]
    td = loaders.load_scans([p], tpx3=False, drop_bad_frames=True)
    assert td.metadata["dropped_frames"] == 2 and td.value.shape[0] == n - 2


def test_tpx3_applied_to_512_frames(tmp_path):
    p = synthetic.write_ue49_scan(tmp_path, "X_0010", np.ones((1, 512, 512)), ["q_pos_0000"])
    td = loaders.load_scans([p])
    assert td.value.shape == (1, 514, 514) and td.metadata["tpx3_corrected"]


def test_missing_folder_and_find_scans(tmp_path):
    (tmp_path / "orphan.txt").write_text("ImageNameTable\nx\n")
    with pytest.raises(ValueError, match="image folder"):
        loaders.load_scan(tmp_path / "orphan.txt", tpx3=False)
    synthetic.write_ue49_scan(tmp_path, "X_0001", _frames(1), ["a_pos_0000"])
    assert [p.name for p in loaders.find_scans(tmp_path)] == ["X_0001.txt"]


def test_skip_missing_images(tmp_path):
    p = synthetic.write_ue49_scan(tmp_path, "X_0115", _frames(4), [f"Cr_pos_000{i}" for i in range(4)])
    (tmp_path / "X_0115" / "Cr_pos_0002.tif").unlink()
    with pytest.raises(ValueError, match="missing image"):
        loaders.load_scans([p], tpx3=False)
    td = loaders.load_scans([p], tpx3=False, skip_missing=True)
    assert td.value.shape[0] == 3 and td.metadata["missing_frames"] == ["X_0115: Cr_pos_0002"]


def test_unreadable_tif_is_skipped(tmp_path):
    p = synthetic.write_ue49_scan(tmp_path, "X_0117", _frames(3), [f"Cr_neg_000{i}" for i in range(3)])
    (tmp_path / "X_0117" / "Cr_neg_0002.tif").write_text("NameTable\tValueTable\n")
    with pytest.raises(ValueError, match="unreadable"):
        loaders.load_scans([p], tpx3=False)
    td = loaders.load_scans([p], tpx3=False, skip_missing=True)
    assert td.value.shape[0] == 2 and "unreadable" in td.metadata["missing_frames"][0]
