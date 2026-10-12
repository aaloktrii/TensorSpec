"""Gate: chunked wannier90_hr.dat reader == whole-file np.loadtxt. No chinook, no Qt.

The production file is 324 orbitals x 1265 R-vectors = 1.33e8 rows (6.2 GB), which
np.loadtxt cannot hold. These tests use tiny synthetic files where the whole-file
answer can still be computed, and check the streamed reader reproduces it exactly --
including across chunk boundaries that fall inside an R-block.
"""
import numpy as np
import pytest

from tensorspec.core.dft.chinook_tb import _load_hr_rows


def write_hr(path, num_wann, nrpts, rows, deg=None):
    """rows: (nrpts*num_wann^2, 7) array in wannier90 order."""
    deg = np.ones(nrpts, dtype=int) if deg is None else np.asarray(deg)
    lines = ["written by test\n", f"{num_wann}\n", f"{nrpts}\n"]
    for i in range(0, nrpts, 15):
        lines.append("".join(f"{d:5d}" for d in deg[i : i + 15]) + "\n")
    for r in rows:
        lines.append(
            "%5d%5d%5d%5d%5d%12.6f%12.6f\n"
            % (r[0], r[1], r[2], r[3], r[4], r[5], r[6])
        )
    path.write_text("".join(lines))
    return 3 + int(np.ceil(nrpts / 15.0))


def make_rows(num_wann, nrpts, seed=0, big_frac=0.05):
    """Mostly-negligible rows with a few real ones, like a real untruncated hr.dat."""
    rng = np.random.default_rng(seed)
    n_block = num_wann * num_wann
    Rs = [(rx, 0, 0) for rx in range(-(nrpts // 2), -(nrpts // 2) + nrpts)]
    rows = []
    for b, R in enumerate(Rs):
        for j in range(1, num_wann + 1):
            for i in range(1, num_wann + 1):
                if rng.random() < big_frac:
                    re, im = rng.uniform(-1, 1), rng.uniform(-1, 1)
                else:
                    re, im = rng.uniform(-1e-6, 1e-6), rng.uniform(-1e-6, 1e-6)
                rows.append([R[0], R[1], R[2], i, j, re, im])
    assert len(rows) == nrpts * n_block
    return np.array(rows, dtype=float)


def reference(path, data_start, num_wann, hop_tol):
    """What the old whole-file code produced: np.loadtxt everything, r_idx by position."""
    rows = np.loadtxt(path, skiprows=data_start)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    n_block = num_wann * num_wann
    r_idx = np.arange(rows.shape[0], dtype=np.int64) // n_block
    sel = (np.abs(rows[:, 5]) > hop_tol) | (np.abs(rows[:, 6]) > hop_tol)
    sel |= (
        (rows[:, 0] == 0)
        & (rows[:, 1] == 0)
        & (rows[:, 2] == 0)
        & (rows[:, 3] == rows[:, 4])
    )
    return rows[sel], r_idx[sel]


@pytest.mark.parametrize("chunk", [7, 100, 1000, 10**6])
def test_chunked_matches_wholefile(tmp_path, chunk):
    num_wann, nrpts, tol = 4, 5, 1e-4
    rows = make_rows(num_wann, nrpts)
    p = tmp_path / "wannier90_hr.dat"
    data_start = write_hr(p, num_wann, nrpts, rows)

    hop, r_idx, n_total = _load_hr_rows(
        str(p), data_start, num_wann, tol, chunk_rows=chunk
    )
    ref_hop, ref_r = reference(str(p), data_start, num_wann, tol)

    assert n_total == rows.shape[0]
    np.testing.assert_allclose(hop, ref_hop, rtol=0, atol=0)
    np.testing.assert_array_equal(r_idx, ref_r)


def test_chunk_boundary_inside_a_block_keeps_r_idx_correct(tmp_path):
    """chunk_rows=7 with block size 16 splits every block: r_idx must not restart."""
    num_wann, nrpts = 4, 5
    rows = make_rows(num_wann, nrpts, seed=3, big_frac=1.0)  # keep everything
    p = tmp_path / "wannier90_hr.dat"
    data_start = write_hr(p, num_wann, nrpts, rows)

    hop, r_idx, n_total = _load_hr_rows(str(p), data_start, num_wann, 1e-4, chunk_rows=7)
    assert n_total == hop.shape[0] == nrpts * num_wann * num_wann
    # every row kept -> r_idx is exactly the positional block index
    np.testing.assert_array_equal(
        r_idx, np.arange(hop.shape[0], dtype=np.int64) // (num_wann * num_wann)
    )
    # and r_idx must agree with the R_x written in the file
    for b in range(nrpts):
        assert len(set(hop[r_idx == b, 0])) == 1


def test_tiny_onsite_rows_are_never_dropped(tmp_path):
    """On-site rows are shifted by -E_F downstream, so they must survive the prefilter."""
    num_wann, nrpts = 3, 3
    rows = make_rows(num_wann, nrpts, seed=1, big_frac=0.0)  # everything negligible
    # R = 0 block is index 1 (R_x = -1, 0, 1)
    p = tmp_path / "wannier90_hr.dat"
    data_start = write_hr(p, num_wann, nrpts, rows)

    hop, r_idx, _ = _load_hr_rows(str(p), data_start, num_wann, 1e-4, chunk_rows=5)
    assert hop.shape[0] == num_wann  # exactly the on-site diagonal, nothing else
    assert np.all(hop[:, 0:3] == 0)
    assert np.all(hop[:, 3] == hop[:, 4])
    np.testing.assert_array_equal(np.unique(r_idx), [1])


def test_degeneracy_weights_gt_one_do_not_break_the_superset(tmp_path):
    """|t_ij| = |t_raw| / weight <= |t_raw|, so a raw-magnitude prefilter is safe."""
    num_wann, nrpts = 3, 3
    rows = make_rows(num_wann, nrpts, seed=5, big_frac=0.3)
    p = tmp_path / "wannier90_hr.dat"
    data_start = write_hr(p, num_wann, nrpts, rows, deg=[2, 1, 4])

    tol = 1e-4
    hop, r_idx, _ = _load_hr_rows(str(p), data_start, num_wann, tol, chunk_rows=11)
    deg = np.array([2.0, 1.0, 4.0])
    t_kept = (hop[:, 5] + 1j * hop[:, 6]) / deg[r_idx]

    # every row the downstream filter would keep is present in the streamed set
    n_block = num_wann * num_wann
    all_r = np.arange(rows.shape[0], dtype=np.int64) // n_block
    t_all = (rows[:, 5] + 1j * rows[:, 6]) / deg[all_r]
    assert int(np.count_nonzero(np.abs(t_all) > tol)) == int(
        np.count_nonzero(np.abs(t_kept) > tol)
    )


def test_empty_after_filter_is_reported_not_silently_empty(tmp_path):
    num_wann, nrpts = 2, 1  # single R = (0,0,0) block
    rows = np.array(
        [[0, 0, 0, i, j, 1e-9, 1e-9] for j in (1, 2) for i in (1, 2)], dtype=float
    )
    p = tmp_path / "wannier90_hr.dat"
    data_start = write_hr(p, num_wann, nrpts, rows)
    hop, r_idx, n_total = _load_hr_rows(str(p), data_start, num_wann, 1e-4)
    assert n_total == 4
    assert hop.shape[0] == 2  # the two on-site rows only
