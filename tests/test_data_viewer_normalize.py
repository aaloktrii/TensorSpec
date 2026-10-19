"""Axis-choice Norm on DataViewer (Along / Plane; per snapped panel)."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from tensorspec.core.data_models import TensorData  # noqa: E402
from tensorspec.core.io.simulated_loader import SimulatedARPESLoader  # noqa: E402
from tensorspec.gui.components.data_viewer_panel import (  # noqa: E402
    DataViewerPanel,
    iter_norm_axis_choices,
    normalize_over_axes,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


LABELS = ["Photon Energy", "Energy", "Θ (Slit)", "Φ (Deflect)"]


def test_iter_norm_axis_choices_covers_along_and_planes():
    choices = iter_norm_axis_choices(LABELS)
    texts = [t for t, _ in choices]
    assert texts[0] == "Off"
    assert "Along Energy" in texts
    assert "Along Θ (Slit)" in texts
    assert "Plane: Energy × Θ (Slit)" in texts
    assert "Plane: Θ (Slit) × Φ (Deflect)" in texts
    assert len(choices) == 1 + 4 + 6  # Off + Along×4 + Plane×C(4,2)


def test_normalize_over_axes_along_energy():
    # (hv, E): hv=0 weak, hv=1 strong — Along Energy normalizes each spectrum
    arr = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    out = normalize_over_axes(arr, (1,))
    assert np.allclose(out[0], [1 / 3, 2 / 3, 1.0])
    assert np.allclose(out[1], [1 / 3, 2 / 3, 1.0])
    assert arr[1, 2] == 30.0


def test_normalize_over_axes_plane_energy_slit():
    # (hv, E, θ): scale whole E×θ block per hv
    arr = np.zeros((2, 2, 2))
    arr[0] = 1.0
    arr[1] = 10.0
    out = normalize_over_axes(arr, (1, 2))
    assert np.allclose(out[0], 1.0)
    assert np.allclose(out[1], 1.0)
    assert arr[1, 0, 0] == 10.0


def test_kz_is_an_axis_option_and_sets_the_extent(app, tmp_path):
    p = tmp_path / "hv_k.npz"
    np.savez(
        p,
        intensity=np.ones((3, 4, 1, 5)),
        hv=np.array([40.0, 84.0, 120.0]),
        kx=np.linspace(-1.0, 1.0, 4),
        ky=np.array([0.0]),
        E=np.linspace(-2.0, 0.1, 5),
        metadata={"stack_axes": "k", "kz_normal_emission_1perA": [2.0, 4.0, 6.0]},
    )
    panel = DataViewerPanel()
    panel.load_data(SimulatedARPESLoader.load(str(p)))
    view = panel.primary_view()
    ik = view.combo_x.findText("kz (1/A)")
    assert ik >= 0
    assert view.combo_y.findText("kz (1/A)") == ik
    view.combo_x.setCurrentIndex(ik)
    view.combo_y.setCurrentIndex(1)  # Energy
    app.processEvents()
    view.redraw()
    extent = view.im_main.get_extent()
    assert extent[0] < 2.0 < 6.0 < extent[1]
    assert view.ax_main.get_xlabel() == "kz (1/A)"


def test_norm_combo_per_panel_and_along_energy(app):
    hv = np.array([20.0, 40.0, 60.0])
    E = np.linspace(-1, 0, 5)
    kx = np.linspace(-0.5, 0.5, 4)
    ky = np.linspace(-0.5, 0.5, 3)
    value = np.zeros((3, 5, 4, 3))
    for i in range(3):
        value[i] = (i + 1) * 10.0
    td = TensorData(
        value=value,
        axes=[hv, E, kx, ky],
        labels=LABELS,
        units=["eV", "eV", "1/A", "1/A"],
        data_type="Simulated ARPES Matrix Elements",
        metadata={},
    )
    panel = DataViewerPanel()
    panel.load_data(td)
    v0 = panel.primary_view()
    assert v0.combo_norm.findText("Along Energy") >= 0
    assert v0.combo_norm.findText("Plane: Energy × Θ (Slit)") >= 0
    assert v0.combo_norm.findText("Plane: Θ (Slit) × Φ (Deflect)") >= 0

    # Energy vs Photon Energy
    v0.combo_x.setCurrentIndex(0)
    v0.combo_y.setCurrentIndex(1)
    app.processEvents()
    v0.combo_norm.setCurrentIndex(0)  # Off
    v0.redraw()
    assert np.nanmax(v0._last_sliced) == pytest.approx(30.0)

    i_along = v0.combo_norm.findText("Along Energy")
    v0.combo_norm.setCurrentIndex(i_along)
    v0.redraw()
    # each hv column is a spectrum along Energy → unit max
    assert np.nanmax(v0._last_sliced, axis=0) == pytest.approx([1.0, 1.0, 1.0])
    assert panel.tensor_data.value[2, 0, 0, 0] == 30.0

    panel.spawn_view(0, 1, v0, "Right")
    v1 = panel.views[-1]
    assert v1.combo_norm.currentText() == "Off"
    v1.combo_norm.setCurrentIndex(i_along)
    v0.combo_norm.setCurrentIndex(0)
    v0.redraw()
    v1.redraw()
    assert np.nanmax(v0._last_sliced) == pytest.approx(30.0)
    assert np.nanmax(v1._last_sliced, axis=0) == pytest.approx([1.0, 1.0, 1.0])
    panel.close()
