from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from tensorspec.core.crystallography import CrystalEngine


def _mono(a: float, species=("C", "C")) -> Structure:
    lat = Lattice.hexagonal(a, 25.0)
    return Structure(lat, list(species), [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]])


def _layer(struct, z=0.0, twist=0.0, sc_x=1, sc_y=1):
    return {
        "struct": struct,
        "sc_x": sc_x,
        "sc_y": sc_y,
        "z_shift": z,
        "twist": twist,
    }


def test_isotropic_gr_on_hbn():
    assert abs(CrystalEngine.isotropic_match_strain_percent(2.46, 2.5) - 1.62601626) < 1e-3


def test_classify_empty():
    assert CrystalEngine.classify_stack_for_dft([]) == "empty"


def test_classify_aligned_n_layers():
    g = _mono(2.46)
    layers = [_layer(g, 0.0), _layer(g, 3.4), _layer(g, 6.8)]
    assert CrystalEngine.classify_stack_for_dft(layers) == "aligned"


def test_classify_twist_two_layers():
    layers = [_layer(_mono(2.46), 0.0, 0.0), _layer(_mono(2.50), 3.4, 30.0)]
    assert CrystalEngine.classify_stack_for_dft(layers) == "twist"


def test_classify_reject_three_with_twist():
    g = _mono(2.46)
    layers = [_layer(g, 0.0, 0.0), _layer(g, 3.4, 10.0), _layer(g, 6.8, 0.0)]
    assert CrystalEngine.classify_stack_for_dft(layers) == "reject_multitwist"


def test_suggest_ref_prefers_closer_lattice():
    # identical twins -> either ok; mismatched pair -> closer match wins
    layers = [_layer(_mono(2.46), 0.0), _layer(_mono(2.50), 3.4)]
    idx, strains = CrystalEngine.suggest_reference_layer(layers)
    assert len(strains) == 2
    assert idx in (0, 1)
    assert strains[idx] == min(strains)


def test_aligned_bilayer_not_dummy_and_vacuum():
    g = _mono(2.46)
    layers = [_layer(g, 0.0), _layer(g, 3.4)]
    vacuum = 20.0
    s = CrystalEngine.build_dft_aligned_stack(layers, vacuum_ang=vacuum, ref_idx=0)
    assert s.lattice.a < 100.0
    assert s.lattice.a != 500.0
    thickness = 3.4  # |z1-z0| from spinboxes (same placement)
    # c must be roughly thickness + vacuum (allow builder offset conventions ±2 Å)
    assert abs(s.lattice.c - (thickness + vacuum)) < 2.5
    assert "layer_tag" in s.site_properties
    assert len(s) == 4


def test_aligned_preserves_inplane_bonds():
    """Center+wrap must not tear honeycomb bonds (regression)."""
    g = _mono(2.46)
    bn = _mono(2.50, ("B", "N"))
    layers = [_layer(g, 0.0), _layer(bn, 3.4)]
    s = CrystalEngine.build_dft_aligned_stack(layers, vacuum_ang=20.0, ref_idx=1)
    coords = np.array([site.coords for site in s])
    syms = [site.specie.symbol for site in s]
    c = coords[[i for i, x in enumerate(syms) if x == "C"]]
    b = coords[[i for i, x in enumerate(syms) if x == "B"]]
    n = coords[[i for i, x in enumerate(syms) if x == "N"]]
    cc = float(np.linalg.norm(c[0, :2] - c[1, :2]))
    bn_len = float(np.linalg.norm(b[0, :2] - n[0, :2]))
    # a/√3 for hexagonal honeycomb
    assert abs(cc - 2.50 / np.sqrt(3)) < 0.02
    assert abs(bn_len - 2.50 / np.sqrt(3)) < 0.02


def test_aligned_mismatch_flag():
    layers = [_layer(_mono(2.46), 0.0), _layer(_mono(2.50), 3.4)]
    assert CrystalEngine.aligned_needs_strain_dialog(layers, ref_idx=0) is True
    same = [_layer(_mono(2.46), 0.0), _layer(_mono(2.46), 3.4)]
    assert CrystalEngine.aligned_needs_strain_dialog(same, ref_idx=0) is False


def test_twist_two_layer_cell_not_dummy():
    g = _mono(2.46)
    layers = [_layer(g, 0.0, 0.0), _layer(g, 3.4, 7.34)]
    struct, info = CrystalEngine.build_dft_twist_stack(layers, vacuum_ang=20.0, ref_idx=0)
    assert struct.lattice.a < 499.0
    assert info["status"] == "commensurate"
    assert info["n_cells"] == 61
    assert "layer_tag" in struct.site_properties


def test_twist_requires_two_layers():
    layers = [_layer(_mono(2.46), 0.0, 5.0)]
    try:
        CrystalEngine.build_dft_twist_stack(layers, 20.0, 0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_twist_rejects_sc_not_1x1():
    """Twist rebuild must not SC-expand layers; moiré tiling already fills the cell."""
    g = _mono(2.46)
    layers = [
        _layer(g, 0.0, 0.0, sc_x=2, sc_y=1),
        _layer(g, 3.4, 21.5, sc_x=1, sc_y=1),
    ]
    try:
        CrystalEngine.build_dft_twist_stack(layers, 20.0, 0)
        assert False, "expected ValueError"
    except ValueError as e:
        msg = str(e).lower()
        assert "1×1" in str(e) or "1x1" in msg
        assert "sc" in msg or "supercell" in msg


def test_twist_commensurate_fills_moire_cell():
    g = _mono(2.46)
    layers = [_layer(g, 0.0, 0.0), _layer(g, 3.4, 7.34)]
    struct, info = CrystalEngine.build_dft_twist_stack(layers, vacuum_ang=20.0, ref_idx=0)
    assert info["status"] == "commensurate"
    assert len(struct) == 244
    tags = struct.site_properties["layer_tag"]
    n_l1 = sum(1 for t in tags if t.endswith("_L1"))
    n_l2 = sum(1 for t in tags if t.endswith("_L2"))
    assert n_l1 == n_l2


def test_commensurate_lattice_is_moire_matrix():
    g = _mono(2.46)
    layers = [_layer(g, 0.0, 0.0), _layer(g, 3.4, 7.34)]
    monolayer_ab = np.asarray(g.lattice.matrix[:2, :2], dtype=float)
    struct, info = CrystalEngine.build_dft_twist_stack(layers, vacuum_ang=20.0, ref_idx=0)
    assert info["status"] == "commensurate"
    S = np.asarray(info["S"], dtype=float)
    moire_a = float(np.linalg.norm((S @ monolayer_ab)[0]))
    assert abs(struct.lattice.a - moire_a) < 0.02


def test_commensurate_build_matches_expected_atom_count():
    g = _mono(2.46)
    layers = [_layer(g, 0.0, 0.0), _layer(g, 3.4, 7.34)]
    struct, info = CrystalEngine.build_dft_twist_stack(layers, vacuum_ang=20.0, ref_idx=0)
    assert info["status"] == "commensurate"
    assert len(struct) == 244


def test_incommensurate_uses_ref_lattice():
    g = _mono(2.46)
    bn = _mono(2.50, ("B", "N"))
    layers = [_layer(g, 0.0, 0.0), _layer(bn, 3.4, 30.0)]
    with pytest.raises(ValueError, match="homobilayer"):
        CrystalEngine.build_dft_twist_stack(layers, vacuum_ang=20.0, ref_idx=0)


def test_gr_hbn_30deg_isotropic_not_frobenius_stretch():
    """Gr/hBN 30°: isotropic stretch ~1.6%, Frobenius+twist score ~50% — not the same."""
    g = _mono(2.46)
    bn = _mono(2.50, ("B", "N"))
    layers = [_layer(g, 0.0, 0.0), _layer(bn, 3.4, 30.0)]
    _, strains = CrystalEngine.suggest_reference_layer(layers)
    iso = CrystalEngine.isotropic_match_strain_percent(2.46, 2.50)
    assert abs(iso - 1.626) < 0.02
    assert min(strains) > 40.0
    assert iso < 5.0


def test_twist_commensurate_empty_tile_raises(monkeypatch):
    import tensorspec.core.bernevig_twist as bt

    g = _mono(2.46)
    layers = [_layer(g, 0.0, 0.0), _layer(g, 3.4, 7.34)]

    def _empty(*args, **kwargs):
        return [], [], []

    monkeypatch.setattr(bt, "tile_layer_unstrained", _empty)
    with pytest.raises(ValueError, match="atoms"):
        CrystalEngine.build_dft_twist_stack(layers, 20.0, 0)



def _crystal_suite_stub(rows_twist=(0.0, 0.0), stacking="AA", max_cells=217, builder="plain"):
    from types import SimpleNamespace as NS
    from unittest import mock

    pytest.importorskip("PySide6")
    from tensorspec.gui.suites import crystal_suite as cs

    g = CrystalEngine.generate_template_structure("Graphene")

    class Spin:
        def __init__(self, v):
            self.v = v

        def value(self):
            return self.v

        def setValue(self, v):
            self.v = v

    rows = []
    for i, t in enumerate(rows_twist):
        row = NS(spin_twist=Spin(t), lbl_name=NS(text=lambda: "g"))
        row.get_layer_dict = (
            lambda r=row, z=(0.0, 3.4)[i]: {
                "struct": g, "sc_x": 1, "sc_y": 1, "z_shift": z,
                "twist": r.spin_twist.v,
            }
        )
        rows.append(row)
    w = NS(
        stack_layer_rows=rows,
        combo_stack_builder=NS(currentData=lambda: builder),
        spin_bernevig_theta=Spin(7.34),
        spin_stack_vacuum=Spin(20.0),
        spin_bernevig_max_cells=Spin(max_cells),
        combo_bernevig_stacking=NS(currentText=lambda: stacking),
        combo_bernevig_valley=NS(currentText=lambda: "K"),
        lbl_moire=NS(setText=lambda t: None),
        refresh_render=mock.Mock(),
        _refresh_bernevig_valleys=lambda layers: None,
    )
    S = cs.CrystalViewerSuite
    w._bernevig_opted_in = lambda: S._bernevig_opted_in(w)
    w._try_bernevig_build = lambda: S._try_bernevig_build(w)
    return cs, S, w


def test_gui_zero_twist_aa_push_stays_aligned_despite_default_spin():
    from unittest import mock

    cs, S, w = _crystal_suite_stub()
    pushed = {}
    with mock.patch.object(
        CrystalEngine, "build_dft_twist_stack", side_effect=AssertionError("twist path")
    ), mock.patch.object(
        cs.global_workspace, "push_crystal_structure", lambda n, s: pushed.setdefault("s", s)
    ), mock.patch.object(
        cs.QInputDialog, "getText", return_value=("x", True)
    ), mock.patch.object(cs, "QMessageBox"):
        S.push_current_to_workspace(w)
    assert len(pushed["s"]) == 4
    assert not S._bernevig_opted_in(w)


def test_gui_build_twisted_writes_row_twists_and_push_follows():
    from unittest import mock

    cs, S, w = _crystal_suite_stub(builder="bernevig")
    ok, err = S._try_bernevig_build(w)
    assert ok, err
    assert [r.spin_twist.v for r in w.stack_layer_rows] == [-3.67, 3.67]
    assert S._bernevig_opted_in(w)
    pushed = {}
    with mock.patch.object(
        cs.global_workspace, "push_crystal_structure", lambda n, s: pushed.setdefault("s", s)
    ), mock.patch.object(
        cs.QInputDialog, "getText", return_value=("x", True)
    ), mock.patch.object(cs, "QMessageBox"):
        S.push_current_to_workspace(w)
    assert len(pushed["s"]) == 244


def test_gui_bernevig_build_failure_is_nonmodal_result():
    from unittest import mock

    cs, S, w = _crystal_suite_stub(rows_twist=(0.0, 5.0), max_cells=1, builder="bernevig")
    ok, err = S._try_bernevig_build(w)
    assert not ok and "commensurate" in err
    assert S._bernevig_opted_in(w)


class _Val:
    def __init__(self, v):
        self.v = v

    def value(self):
        return self.v


def _moire_stub(twist2, builder="plain"):
    from unittest import mock

    cs, S, w = _crystal_suite_stub(rows_twist=(0.0, twist2), builder=builder)
    for row, z in zip(w.stack_layer_rows, (0.0, 3.4)):
        row.spin_z = _Val(z)
    labels = []
    w.lbl_moire = mock.Mock(setText=labels.append)
    w.renderer = mock.Mock(spec=["draw_moire_envelope", "plotter"])
    w.handle_draw_stack = mock.Mock()
    w.handle_bernevig_twist = mock.Mock()
    return S, w, labels


def test_moire_button_plain_reports_period_and_draws_outline():
    S, w, labels = _moire_stub(5.0)
    S.handle_moire(w)
    w.handle_bernevig_twist.assert_not_called()
    w.handle_draw_stack.assert_called_once()
    assert "Incommensurate" in labels[-1] and "28.20" in labels[-1]
    m, z_lo, z_hi = w.renderer.draw_moire_envelope.call_args.args
    assert np.isclose(np.linalg.norm(m[0]), 28.20, atol=0.01)
    assert (z_lo, z_hi) == (-1.5, 4.9)


def test_moire_button_plain_commensurate_counts_cells():
    S, w, labels = _moire_stub(21.787)
    S.handle_moire(w)
    assert "Commensurate" in labels[-1] and "7 cells" in labels[-1]


def test_moire_button_plain_zero_twist_no_crash_no_outline():
    S, w, labels = _moire_stub(0.0)
    S.handle_moire(w)
    assert "No Moiré" in labels[-1]
    w.renderer.draw_moire_envelope.assert_not_called()


def test_moire_button_bernevig_runs_builder():
    S, w, labels = _moire_stub(5.0, builder="bernevig")
    S.handle_moire(w)
    w.handle_bernevig_twist.assert_called_once()
    w.renderer.draw_moire_envelope.assert_not_called()
