"""Gate C / P5 GUI smoke tests: ARPESPanel B3 (SPR-KKR) settings + trigger wiring.

Qt only (no real kkrspec run): ARPESRunnerThread.start is monkeypatched so
trigger_simulation never actually launches a job; we just check the kwargs
it would have started with.
"""
from pathlib import Path

import pytest

from tensorspec.gui.components import arpes_panel as ap
from tensorspec.core.workspace import global_workspace


def _make_panel(qapp):
    return ap.ARPESPanel()


def test_b3_engine_shows_sprkkr_group(qapp):
    panel = _make_panel(qapp)
    idx = panel.engine_dropdown.findData("B3")
    assert idx >= 0
    panel.engine_dropdown.setCurrentIndex(idx)

    assert not panel.sprkkr_group.isHidden()


def test_update_kkr_eta_sets_label(qapp):
    panel = _make_panel(qapp)
    idx = panel.engine_dropdown.findData("B3")
    panel.engine_dropdown.setCurrentIndex(idx)

    panel._update_kkr_eta()

    assert "ETA" in panel.lbl_kkr_eta.text()


def test_trigger_simulation_b3_local_builds_expected_kwargs(qapp, monkeypatch, tmp_path):
    panel = _make_panel(qapp)
    idx = panel.engine_dropdown.findData("B3")
    panel.engine_dropdown.setCurrentIndex(idx)

    # Force local compute target (index 0 is "Local only" already, but be explicit).
    local_idx = panel.target_dropdown.findData(None)
    if local_idx < 0:
        local_idx = 0
    panel.target_dropdown.setCurrentIndex(local_idx)
    assert not panel._is_remote_target()

    # Any real file works as a fake ".pot_new" for this smoke test (existence check only).
    fake_pot = Path(__file__).resolve()
    panel.sprkkr_pot_edit.setText(str(fake_pot))
    panel.sprkkr_bin_edit.setText(str(tmp_path))

    recorded = {}

    def _fake_start(self):
        recorded["kwargs"] = dict(self.experiment_kwargs)
        recorded["model_choice"] = self.model_choice

    monkeypatch.setattr(ap.ARPESRunnerThread, "start", _fake_start)
    # Default grid (100x100x100) trips the new long-run ETA gate; say Yes.
    monkeypatch.setattr(ap.QMessageBox, "question", staticmethod(lambda *a, **k: ap.QMessageBox.Yes))
    # Auto + no lattice -> ABAS fallback warns once.
    monkeypatch.setattr(ap.QMessageBox, "warning", staticmethod(lambda *a, **k: None))

    panel.trigger_simulation()

    assert recorded.get("model_choice") == "B3"
    kwargs = recorded["kwargs"]
    assert kwargs["pot_path"] == str(fake_pot)
    assert "k_bounds" in kwargs and "X" in kwargs["k_bounds"] and "Y" in kwargs["k_bounds"]
    assert kwargs["n_layer"] == panel.spin_n_layer.value()
    assert "launcher" in kwargs
    assert kwargs["temperature_K"] == panel.temperature_spin.value()
    # No lattice available -> falls back to treating hkl as already-ABAS.
    assert kwargs["hkl_frame"] == "abas"
    assert kwargs["iq_at_surf"] is None  # chk_iq_auto defaults to checked


def test_trigger_simulation_b3_force_abas_skips_cif_even_with_vault(qapp, monkeypatch, tmp_path):
    """Explicit ABAS frame must not convert even when vault has cif_lattice."""
    panel = _make_panel(qapp)
    idx = panel.engine_dropdown.findData("B3")
    panel.engine_dropdown.setCurrentIndex(idx)

    local_idx = panel.target_dropdown.findData(None)
    if local_idx < 0:
        local_idx = 0
    panel.target_dropdown.setCurrentIndex(local_idx)

    fake_pot = Path(__file__).resolve()
    panel.sprkkr_pot_edit.setText(str(fake_pot))
    panel.sprkkr_bin_edit.setText(str(tmp_path))

    lattice = [[14.175, 0.0, -4.977], [0.0, 3.541, 0.0], [0.0, 0.0, 9.058]]
    global_workspace.push_remote_run(
        name="VTe2_abas_force_vault",
        cluster_name="local",
        engine="SPRKKR",
        remote_path=str(fake_pot),
        meta={"cif_lattice": lattice},
    )
    panel.vault_combo.addItem("VTe2_abas_force_vault")
    panel.vault_combo.setCurrentText("VTe2_abas_force_vault")

    abas_idx = panel.combo_hkl_frame.findData("abas")
    assert abas_idx >= 0
    panel.combo_hkl_frame.setCurrentIndex(abas_idx)
    panel.spin_h.setValue(-2)
    panel.spin_k.setValue(0)
    panel.spin_l.setValue(1)

    recorded = {}

    def _fake_start(self):
        recorded["kwargs"] = dict(self.experiment_kwargs)

    monkeypatch.setattr(ap.ARPESRunnerThread, "start", _fake_start)
    monkeypatch.setattr(ap.QMessageBox, "question", staticmethod(lambda *a, **k: ap.QMessageBox.Yes))
    monkeypatch.setattr(
        ap.QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError("ABAS must not warn"))),
    )

    try:
        panel.trigger_simulation()
    finally:
        global_workspace.remove("VTe2_abas_force_vault")

    kwargs = recorded["kwargs"]
    assert kwargs["hkl_frame"] == "abas"
    assert "cif_lattice" not in kwargs
    assert kwargs["hkl"] == (-2, 0, 1)


def test_trigger_simulation_b3_uses_vault_cif_lattice(qapp, monkeypatch, tmp_path):
    """A vault with a stored cif_lattice -> hkl_frame='cif' + cif_lattice passthrough;
    pinning IQ_AT_SURF via the spinbox overrides auto-pick."""
    panel = _make_panel(qapp)
    idx = panel.engine_dropdown.findData("B3")
    panel.engine_dropdown.setCurrentIndex(idx)

    local_idx = panel.target_dropdown.findData(None)
    if local_idx < 0:
        local_idx = 0
    panel.target_dropdown.setCurrentIndex(local_idx)

    fake_pot = Path(__file__).resolve()
    panel.sprkkr_pot_edit.setText(str(fake_pot))
    panel.sprkkr_bin_edit.setText(str(tmp_path))

    lattice = [[14.175, 0.0, -4.977], [0.0, 3.541, 0.0], [0.0, 0.0, 9.058]]
    global_workspace.push_remote_run(
        name="VTe2_test_vault", cluster_name="local", engine="SPRKKR",
        remote_path=str(fake_pot), meta={"cif_lattice": lattice},
    )
    panel.vault_combo.addItem("VTe2_test_vault")
    panel.vault_combo.setCurrentText("VTe2_test_vault")
    # Auto (default) uses vault cif_lattice -> cif frame.
    assert panel.combo_hkl_frame.currentData() == "auto"

    panel.chk_iq_auto.setChecked(False)
    panel.spin_iq_surf.setValue(7)

    recorded = {}

    def _fake_start(self):
        recorded["kwargs"] = dict(self.experiment_kwargs)

    monkeypatch.setattr(ap.ARPESRunnerThread, "start", _fake_start)
    monkeypatch.setattr(ap.QMessageBox, "question", staticmethod(lambda *a, **k: ap.QMessageBox.Yes))
    monkeypatch.setattr(ap.QMessageBox, "warning", staticmethod(lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("should not warn: vault has a cif_lattice")
    )))

    try:
        panel.trigger_simulation()
    finally:
        global_workspace.remove("VTe2_test_vault")

    kwargs = recorded["kwargs"]
    assert kwargs["hkl_frame"] == "cif"
    assert kwargs["cif_lattice"] == lattice
    assert kwargs["iq_at_surf"] == 7


def test_lookup_cif_lattice_uses_crystal_not_band_combo(qapp):
    """Regression: ws_combo lists bands; CIF lattice must come from crystal_structure."""
    from pymatgen.core import Lattice, Structure

    panel = _make_panel(qapp)
    lat = Lattice.from_parameters(14.175, 3.541, 9.058, 90, 110.55, 90)
    struct = Structure(lat, ["V", "Te"], [[0, 0, 0], [0.3, 0.3, 0.3]])
    global_workspace.push_crystal_structure("VTe2_cif", struct)
    # Band combo selected name that is NOT the crystal name.
    panel.ws_combo.clear()
    panel.ws_combo.addItem("some_band_push")
    panel.ws_combo.setCurrentText("some_band_push")
    try:
        matrix = panel._lookup_cif_lattice()
        assert matrix is not None
        assert len(matrix) == 3 and len(matrix[0]) == 3
        frame, got = panel._resolve_sprkkr_hkl_frame("cif")
        assert frame == "cif"
        assert got == matrix
    finally:
        global_workspace.remove("VTe2_cif")


def test_update_kkr_eta_ky_degenerate_matches_single_step(qapp):
    """min==max on ky (with ky_steps=10) must ETA the same as ky_steps=1."""
    panel = _make_panel(qapp)
    idx = panel.engine_dropdown.findData("B3")
    panel.engine_dropdown.setCurrentIndex(idx)

    panel.spin_ky_min.setValue(0.0)
    panel.spin_ky_max.setValue(0.0)
    panel.spin_ky_steps.setValue(10)
    panel._update_kkr_eta()
    degenerate_text = panel.lbl_kkr_eta.text()

    panel.spin_ky_steps.setValue(1)
    panel._update_kkr_eta()
    single_step_text = panel.lbl_kkr_eta.text()

    assert degenerate_text == single_step_text
