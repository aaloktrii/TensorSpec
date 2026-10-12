"""Gate: viewer_export.py -> SimulatedARPESLoader contract. No binary, no Qt."""
import json

import numpy as np
import pytest

from tensorspec.core.dft.sprkkr.pointwise import angle_points
from tensorspec.core.dft.sprkkr.outputs import write_points_json
from tensorspec.core.dft.sprkkr.viewer_export import arrays_to_viewer_npz, stack_viewer_cubes
from tensorspec.core.io.simulated_loader import SimulatedARPESLoader


def _fake_run(tmp_path, deflector, nt=5, ne=7):
    pts = angle_points((-10.0, 10.0), nt, hv_eV=84.0, work_function_eV=4.5, deflector_deg=deflector)
    pj = tmp_path / f"points_{deflector}.json"
    write_points_json(str(pj), pts, meta={"deflector_deg": deflector})
    E = np.linspace(-1.0, 0.1, ne)
    theta = np.array([p.slit_deg for p in pts])
    I = np.random.default_rng(0).random((ne, nt, 1))
    return I, E, theta, np.array([deflector]), str(pj), pts


def test_pointwise_viewer_npz_loads_with_lab_axes(tmp_path):
    I, E, theta, phi, pj, pts = _fake_run(tmp_path, -10.3)
    out = arrays_to_viewer_npz(I, E, theta, phi, str(tmp_path / "v.npz"), points_json=pj)
    d = np.load(out, allow_pickle=True)
    assert d["intensity"].shape == (5, 1, 7)  # (kx, ky, E)
    np.testing.assert_allclose(d["kx"], [p.k_slit for p in pts])
    np.testing.assert_allclose(d["ky"], [pts[0].k_defl])
    td = SimulatedARPESLoader.load(out)
    assert td.value.shape == (7, 5, 1)  # loader -> (E, kx, ky)
    assert td.labels[1].startswith("kx")
    np.testing.assert_allclose(td.value[:, :, 0], I[:, :, 0])
    assert td.metadata["pointwise"] is True


def test_legacy_viewer_npz_uses_kpar_and_ky_zero(tmp_path):
    E = np.linspace(-1.0, 0.1, 4)
    theta = np.array([-15.0, 0.0, 15.0])
    I = np.ones((4, 3, 1))
    out = arrays_to_viewer_npz(I, E, theta, np.array([-10.3]), str(tmp_path / "l.npz"),
                               k_par_e0=np.array([1.18, 0.0, 1.18]))
    d = np.load(out, allow_pickle=True)
    np.testing.assert_allclose(d["kx"], [-1.18, 0.0, 1.18])
    np.testing.assert_allclose(d["ky"], [0.0])
    assert SimulatedARPESLoader.load(out).value.shape == (4, 3, 1)


def test_stack_cubes_sorted_by_ky(tmp_path):
    paths = []
    for dfl in (5.0, -10.3, 0.0):
        I, E, theta, phi, pj, _ = _fake_run(tmp_path, dfl)
        paths.append(arrays_to_viewer_npz(I, E, theta, phi, str(tmp_path / f"c{dfl}.npz"), points_json=pj))
    out = stack_viewer_cubes(paths, str(tmp_path / "map.npz"))
    d = np.load(out, allow_pickle=True)
    assert d["intensity"].shape == (5, 3, 7)
    assert np.all(np.diff(d["ky"]) > 0)
    td = SimulatedARPESLoader.load(out)
    assert td.value.shape == (7, 5, 3)
    assert td.metadata["n_deflectors"] == 3


def test_stack_rejects_mismatched_kx(tmp_path):
    I, E, theta, phi, pj, _ = _fake_run(tmp_path, 0.0)
    a = arrays_to_viewer_npz(I, E, theta, phi, str(tmp_path / "a.npz"), points_json=pj)
    I2, E2, theta2, phi2, pj2, _ = _fake_run(tmp_path, 3.0, nt=6)
    b = arrays_to_viewer_npz(I2, E2, theta2, phi2, str(tmp_path / "b.npz"), points_json=pj2)
    with pytest.raises(ValueError):
        stack_viewer_cubes([a, b], str(tmp_path / "bad.npz"))
