"""Gate: Wannier90 use_ws_distance MDRS remap (material-agnostic)."""
from __future__ import annotations

import numpy as np
import pytest

from tensorspec.core.dft.w90_ws_distance import (
    expand_hoppings_ws,
    parse_win_ws_settings,
    ws_minimal_shifts,
)


def test_parse_win_ws_settings(tmp_path):
    win = tmp_path / "wannier90.win"
    win.write_text(
        "use_ws_distance = true\n"
        "mp_grid = 4 4 1\n"
        "ws_search_size = 2\n"
        "ws_distance_tol = 1.0e-5\n"
    )
    cfg = parse_win_ws_settings(str(win))
    assert cfg["use_ws_distance"] is True
    assert cfg["mp_grid"] == (4, 4, 1)
    assert cfg["ws_search_size"] == 2


def test_parse_win_ws_false(tmp_path):
    win = tmp_path / "wannier90.win"
    win.write_text("use_ws_distance = false\nmp_grid = 6 6 6\n")
    cfg = parse_win_ws_settings(str(win))
    assert cfg["use_ws_distance"] is False
    assert cfg["mp_grid"] == (6, 6, 6)


def test_ws_shift_identity_when_centres_on_origin():
    """Centres at 0 → T=0 is always minimal (no graphene-specific logic)."""
    A = np.eye(3) * 2.0
    tau = np.zeros(3)
    shifts, ndeg = ws_minimal_shifts(tau, tau, np.array([1.0, 0.0, 0.0]), A, (4, 4, 4))
    assert ndeg >= 1
    assert any(np.allclose(s, 0.0) for s in shifts)


def test_expand_divides_by_degeneracy():
    A = np.eye(3)
    centres = [np.zeros(3), np.array([0.5, 0.0, 0.0])]
    R = np.array([[0.0, 0.0, 0.0]])
    i = np.array([0])
    j = np.array([1])
    t = np.array([2.0 + 0.0j])
    R2, i2, j2, t2 = expand_hoppings_ws(R, i, j, t, centres, A, (2, 2, 2))
    assert len(t2) >= 1
    # Sum of weights over images recovers original |t| scale for FT
    assert pytest.approx(abs(t[0]), rel=1e-9) == abs(np.sum(t2))


@pytest.mark.parametrize("label", ["graphene_scratch"])
def test_graphene_K_matches_w90_band_dat_when_present(label):
    """Integration: if scratch graphene W90 run exists, blue≡red at K."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    drive = root / "scratch" / "graphene_dft_drive"
    hr = drive / "wannier90_hr.dat"
    band = drive / "wannier90_band.dat"
    cif = root / "tensorspec" / "cif_file" / "graphene.cif"
    if not (hr.is_file() and band.is_file() and cif.is_file()):
        pytest.skip("graphene W90 scratch not present")

    from pymatgen.core import Structure
    from tensorspec.core.dft.chinook_tb import ChinookTightBindingEngine
    from tensorspec.core.dft.qe_fermi import detect_qe_fermi_eV

    # Bust any stale in-process assumptions via fresh engine
    struct = Structure.from_file(str(cif))
    ef, _ = detect_qe_fermi_eV(str(drive))
    eng = ChinookTightBindingEngine()
    eng.crystal_structure = struct
    B = struct.lattice.reciprocal_lattice.matrix
    kpts = np.array(
        [
            np.dot([0.0, 0.0, 0.0], B),
            np.dot([0.5, 0.0, 0.0], B),
            np.dot([1.0 / 3.0, 1.0 / 3.0, 0.0], B),
        ]
    )
    evals, _, _ = eng.solve_bands(
        kpts,
        w90_filepath=str(hr),
        onsite_e=0.0,
        need_eigenvectors=False,
        qe_fermi=ef,
    )

    def w90_at(idx: int):
        bands = []
        cur = []
        with open(band) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    cur.append(float(parts[1]))
                elif not parts and cur:
                    bands.append(np.array(cur))
                    cur = []
        if cur:
            bands.append(np.array(cur))
        return np.sort([b[idx] - ef for b in bands])

    # labelinfo: G=1→0, M=101→100, K=159→158
    for name, i, wi, tol in (
        ("G", 0, 0, 1e-3),
        ("M", 1, 100, 1e-3),
        ("K", 2, 158, 5e-2),  # WS should bring K within tens of meV
    ):
        diff = np.max(np.abs(np.sort(evals[i]) - w90_at(wi)))
        assert diff < tol, f"{name}: max|blue-w90|={diff:.4f} eV (tol={tol})"
