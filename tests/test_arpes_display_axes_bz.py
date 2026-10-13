"""BZ overlay is k-space only — gated behind Momentum display mode."""
from __future__ import annotations

import numpy as np


def test_bz_checkbox_disabled_in_angle_mode(qapp):
    from tensorspec.gui.components.arpes_panel import ARPESPanel

    panel = ARPESPanel()
    assert panel.combo_display_axes.currentData() == "angle"
    assert not panel.chk_overlay_bz.isEnabled()

    panel.combo_display_axes.setCurrentIndex(
        panel.combo_display_axes.findData("momentum")
    )
    assert panel.chk_overlay_bz.isEnabled()

    panel.chk_overlay_bz.setChecked(True)
    assert panel.chk_overlay_bz.isChecked()

    panel.combo_display_axes.setCurrentIndex(
        panel.combo_display_axes.findData("angle")
    )
    assert not panel.chk_overlay_bz.isEnabled()
    assert not panel.chk_overlay_bz.isChecked()


def test_plot_axis_arrays_momentum_uses_sin_map(qapp):
    from tensorspec.gui.components.arpes_panel import ARPESPanel

    panel = ARPESPanel()
    panel.sim_kx = np.array([-22.0, 0.0, 22.0])
    panel.sim_ky = np.array([-10.0, 0.0, 10.0])
    panel.photon_energy_spin.setValue(90.0)
    panel.work_function_spin.setValue(4.5)

    th, ph, use_ang = panel._plot_axis_arrays()
    assert use_ang is True
    assert np.allclose(th, panel.sim_kx)

    panel.combo_display_axes.setCurrentIndex(
        panel.combo_display_axes.findData("momentum")
    )
    kx, ky, use_ang = panel._plot_axis_arrays()
    assert use_ang is False
    e_kin = 90.0 - 4.5
    k_r = 0.512316 * np.sqrt(e_kin)
    assert np.allclose(kx, k_r * np.sin(np.radians(panel.sim_kx)))
    assert np.allclose(ky, k_r * np.sin(np.radians(panel.sim_ky)))


def test_draw_bz_noop_in_angle_mode(qapp, monkeypatch):
    from tensorspec.gui.components.arpes_panel import ARPESPanel

    panel = ARPESPanel()
    called = {"n": 0}

    def _boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("BZ draw must not run in angle mode")

    monkeypatch.setattr(
        "tensorspec.core.crystallography.CrystalEngine.calculate_brillouin_zone",
        _boom,
    )
    panel._draw_bz_overlay()
    assert called["n"] == 0
