import math
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from tensorspec.core.bernevig_twist import (
    classify_bravais,
    hexagonal_pair,
    q_lattice_label,
    square_pair,
)


def _hex_ab(a=2.46):
    return np.array([[a, 0.0], [-a / 2, a * math.sqrt(3) / 2]])


def test_bravais_and_q_labels():
    assert classify_bravais(_hex_ab()) == "hexagonal"
    assert classify_bravais(np.array([[3.0, 0.0], [0.0, 3.0]])) == "square"
    assert classify_bravais(np.array([[3.0, 0.0], [0.0, 5.0]])) == "rectangular"
    assert classify_bravais(np.array([[3.0, 0.0], [1.0, 4.0]])) == "oblique"
    assert q_lattice_label("hexagonal", "Γ") == "triangular"
    assert q_lattice_label("hexagonal", "K") == "honeycomb"
    assert q_lattice_label("hexagonal", "M") == "kagome"
    assert q_lattice_label("square", "X") == "nested square"
    assert q_lattice_label("square", "Γ") == "square"
    assert q_lattice_label("rectangular", "HSP") == "rectangular"
    with pytest.raises(ValueError):
        q_lattice_label("hexagonal", "X")


def test_hex_and_square_pairs_match_paper_angles():
    th, S = hexagonal_pair(5, 4)
    assert th == pytest.approx(7.340993016630217)
    assert abs(int(round(np.linalg.det(S)))) == 61
    assert S.tolist() == [[5, -4], [4, 9]]
    th98, S98 = hexagonal_pair(9, 8)
    assert th98 == pytest.approx(3.8902381690076835)
    assert abs(int(round(np.linalg.det(S98)))) == 217
    ths, Ss = square_pair(2, 1)
    assert ths == pytest.approx(53.13010235415598)
    assert abs(int(round(np.linalg.det(Ss)))) == 5
    assert Ss.tolist() == [[2, 1], [-1, 2]]


def test_snap_hex_paper_angles_and_rotation_residual():
    from tensorspec.core.bernevig_twist import snap_commensurate

    ab = _hex_ab()
    hit = snap_commensurate(ab, 7.34, max_cells=217)
    assert hit["bravais"] == "hexagonal"
    assert hit["n_cells"] == 61
    assert hit["theta_used"] == pytest.approx(7.340993016630217)
    assert hit["residual"] < 1e-8
    moire = hit["S"] @ ab
    th = math.radians(hit["theta_used"])
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]])
    coeffs = (moire @ R) @ np.linalg.inv(ab)
    assert np.max(np.abs(coeffs - np.round(coeffs))) < 1e-8

    hit98 = snap_commensurate(ab, 3.89, max_cells=217)
    assert hit98["n_cells"] == 217
    assert hit98["theta_used"] == pytest.approx(3.8902381690076835)


def test_snap_square_pair():
    from tensorspec.core.bernevig_twist import snap_commensurate

    ab = np.array([[3.0, 0.0], [0.0, 3.0]])
    hit = snap_commensurate(ab, 53.13, max_cells=30)
    assert hit["bravais"] == "square"
    assert hit["n_cells"] == 5
    assert hit["theta_used"] == pytest.approx(53.13010235415598)
    assert hit["residual"] < 1e-8


def test_snap_hex_negative_theta():
    from tensorspec.core.bernevig_twist import _residual, snap_commensurate

    ab = _hex_ab()
    hit = snap_commensurate(ab, -7.34, max_cells=217)
    assert hit["theta_used"] == pytest.approx(-7.340993016630217)
    assert hit["n_cells"] == 61
    assert hit["residual"] < 1e-8
    moire = hit["S"].astype(float) @ ab
    assert _residual(ab, moire, hit["theta_used"]) < 1e-8


def test_snap_hex_no_pair_raises():
    from tensorspec.core.bernevig_twist import snap_commensurate

    with pytest.raises(ValueError, match="commensurate"):
        snap_commensurate(_hex_ab(), 7.34, max_cells=1)


def test_rect_inexact_angle_raises():
    from tensorspec.core.bernevig_twist import snap_commensurate

    ab = np.array([[3.0, 0.0], [0.0, 5.0]])
    with pytest.raises(ValueError, match="commensurate"):
        snap_commensurate(ab, 10.0, max_cells=20)


def _graphene(a=2.46):
    lat = Lattice.hexagonal(a, 25.0)
    return Structure(lat, ["C", "C"], [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]])


def test_paper_angle_fills_61_cells_per_layer():
    from tensorspec.core.bernevig_twist import build_bernevig_bilayer

    g = _graphene()
    struct, info = build_bernevig_bilayer(
        g, g, theta_deg=7.34, stacking="AA",
        z_bottom=0.0, z_top=3.4, vacuum_ang=20.0,
        max_cells=217, valley="M",
    )
    assert info["status"] == "commensurate"
    assert info["n_cells"] == 61
    assert info["q_lattice"] == "kagome"
    assert info["theta_used"] == pytest.approx(7.340993016630217)
    assert len(struct) == 244
    tags = struct.site_properties["layer_tag"]
    assert sum(t.endswith("_L1") for t in tags) == 122
    assert sum(t.endswith("_L2") for t in tags) == 122
    # unstrained monolayer edge length times |S row|
    ab = g.lattice.matrix[:2, :2]
    moire = info["S"].astype(float) @ ab
    assert struct.lattice.a == pytest.approx(float(np.linalg.norm(moire[0])), abs=1e-6)
    assert struct.lattice.c == pytest.approx(3.4 + 20.0, abs=2.5)
    tags = struct.site_properties["layer_tag"]
    coords = np.array([site.coords for site in struct])
    c = coords[[i for i, t in enumerate(tags) if t.endswith("_L1")], :2]
    dmin = min(
        float(np.linalg.norm(c[i] - c[j]))
        for i in range(len(c)) for j in range(i + 1, len(c))
        if np.linalg.norm(c[i] - c[j]) < 0.6 * 2.46
    )
    assert dmin == pytest.approx(2.46 / math.sqrt(3), abs=0.02)


def test_aa_zero_stacks_on_same_xy():
    from tensorspec.core.bernevig_twist import build_bernevig_bilayer

    g = _graphene()
    struct, info = build_bernevig_bilayer(
        g, g, 0.0, "AA", 0.0, 3.4, 20.0, valley="K",
    )
    assert info["status"] == "perfect_alignment"
    assert len(struct) == 4
    tags = struct.site_properties["layer_tag"]
    xy = struct.cart_coords[:, :2]
    bot = xy[[i for i, t in enumerate(tags) if t.endswith("_L1")]]
    top = xy[[i for i, t in enumerate(tags) if t.endswith("_L2")]]
    for p in bot:
        assert np.min(np.linalg.norm(top - p, axis=1)) < 1e-6


def test_ab_zero_is_180_about_origin():
    from tensorspec.core.bernevig_twist import build_bernevig_bilayer

    g = _graphene()
    struct, _ = build_bernevig_bilayer(g, g, 0.0, "AB", 0.0, 3.4, 20.0, valley="K")
    tags = struct.site_properties["layer_tag"]
    xy = struct.cart_coords[:, :2]
    bot = xy[[i for i, t in enumerate(tags) if t.endswith("_L1")]]
    top = xy[[i for i, t in enumerate(tags) if t.endswith("_L2")]]
    ab = struct.lattice.matrix[:2, :2]
    inv = np.linalg.inv(ab)
    for p in top:
        deltas = (p.reshape(1, 2) + bot) @ inv
        err = np.min(np.abs(deltas - np.round(deltas)))
        assert err < 1e-6


def test_hetero_and_vacuum_raise():
    from tensorspec.core.bernevig_twist import build_bernevig_bilayer

    g = _graphene(2.46)
    with pytest.raises(ValueError, match="homobilayer"):
        build_bernevig_bilayer(g, _graphene(2.50), 7.34, "AA", 0.0, 3.4, 20.0)
    with pytest.raises(ValueError, match="15"):
        build_bernevig_bilayer(g, g, 7.34, "AA", 0.0, 3.4, 15.0)


def test_slightly_nonideal_hex_raises_not_strained():
    from tensorspec.core.bernevig_twist import (
        build_bernevig_bilayer,
        classify_bravais,
        snap_commensurate,
    )

    lat = Lattice.from_parameters(2.46, 2.46, 25.0, 90.0, 90.0, 120.5)
    ab = lat.matrix[:2, :2]
    assert classify_bravais(ab) == "hexagonal"
    with pytest.raises(ValueError, match="commensurate") as exc:
        snap_commensurate(ab, 7.34, max_cells=217)
    assert "residual" in str(exc.value)
    s = Structure(lat, ["C", "C"], [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]])
    with pytest.raises(ValueError, match="commensurate"):
        build_bernevig_bilayer(s, s, 7.34, "AA", 0.0, 3.4, 20.0)
    g = _graphene()
    struct, _ = build_bernevig_bilayer(g, g, 7.34, "AA", 0.0, 3.4, 20.0)
    assert len(struct) == 244


def test_tile_layer_unstrained_rejects_noninteger_ratio():
    from tensorspec.core.bernevig_twist import tile_layer_unstrained

    layer = np.array([[3.0, 0.0], [0.0, 3.0]])
    moire = np.array([[6.0, 0.0], [0.0, 6.001]])
    with pytest.raises(ValueError, match="commensurate"):
        tile_layer_unstrained(moire, layer, ["C"], [[0.0, 0.0, 0.0]], ["C_L1"])


def test_search_integer_S_fast_and_bounded_at_max_cells_2000():
    import time

    from tensorspec.core.bernevig_twist import snap_commensurate

    ab = np.array([[3.0, 0.0], [0.0, 5.0]])
    t0 = time.time()
    with pytest.raises(ValueError, match="commensurate"):
        snap_commensurate(ab, 10.0, max_cells=2000)
    hit = snap_commensurate(ab, 180.0, max_cells=2000)
    assert hit["n_cells"] == 1
    assert hit["residual"] < 1e-6
    assert time.time() - t0 < 5.0


def test_search_integer_S_finds_rect_commensurate_cell():
    from tensorspec.core.bernevig_twist import snap_commensurate

    # 3x4 oblique-free rectangle: rotating by 180 deg is the only exact rotation
    ab = np.array([[3.0, 0.0], [0.0, 4.0]])
    hit = snap_commensurate(ab, 180.0, max_cells=50)
    assert hit["n_cells"] == 1


def test_per_layer_atom_count_mismatch_raises(monkeypatch):
    import tensorspec.core.bernevig_twist as bt

    g = _graphene()
    real = bt.tile_layer_unstrained
    calls = {"n": 0}

    def fake(*args, **kwargs):
        s, c, t = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            return s[:-1], c[:-1], t[:-1]
        return s, c, t

    monkeypatch.setattr(bt, "tile_layer_unstrained", fake)
    with pytest.raises(ValueError, match="Layer 1"):
        bt.build_bernevig_bilayer(g, g, 7.34, "AA", 0.0, 3.4, 20.0)
