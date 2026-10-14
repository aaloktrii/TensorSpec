"""Generic DataViewerPanel features used by the PEEM XMCD tab (off by default for ARPES / ML)."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from tensorspec.core.data_models import TensorData  # noqa: E402
from tensorspec.gui.components.data_viewer_panel import DataViewerPanel, plane_layer  # noqa: E402


@pytest.fixture(autouse=True)
def _private_load_settings(tmp_path, monkeypatch):
    """The load list remembers its folder in QSettings: keep tests out of the user's settings."""
    from PySide6.QtCore import QSettings
    from tensorspec.gui.components.peem_tabs.load_section import LoadSection
    store = QSettings(str(tmp_path / "load_settings.ini"), QSettings.IniFormat)
    monkeypatch.setattr(LoadSection, "_settings", staticmethod(lambda: store))



@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _stack(n=6, ny=40, nx=50):
    v = np.arange(n * ny * nx, dtype=float).reshape(n, ny, nx)
    return TensorData(value=v, axes=[np.arange(n, dtype=float), np.arange(ny, dtype=float), np.arange(nx, dtype=float)],
                      labels=["frame", "y", "x"], units=["", "px", "px"], data_type="Experimental PEEM", metadata={})


def _panel(app):
    p = DataViewerPanel(compact=True)
    p.load_data(_stack())
    view = p.primary_view()
    view.combo_x.setCurrentIndex(2)
    view.combo_y.setCurrentIndex(1)
    app.processEvents()
    return p, view


def test_defaults_unchanged(app):
    p = DataViewerPanel()
    p.load_data(_stack())
    v = p.primary_view()
    assert v.combo_cmap.currentText() == "magma" and not v.chk_sym.isChecked() and not v.chk_image_origin.isChecked()
    assert p.btn_save_condition.isVisibleTo(p)


def test_plane_layer_any_ndim():
    assert plane_layer(np.ones((40, 50)), (6, 40, 50), 1, 2).shape == (40, 50)
    assert plane_layer(np.ones((50, 40)), (6, 40, 50), 1, 2).shape == (40, 50)
    assert plane_layer(np.ones((3, 3)), (6, 40, 50), 1, 2) is None


def test_map_layer_symmetric_and_origin(app):
    p, v = _panel(app)
    m = np.linspace(-0.1, 0.3, 40 * 50).reshape(40, 50)
    p.set_layers({"pcd": m}, select="pcd")
    assert np.allclose(p.current_image(), m)
    p.configure_view(cmap="bwr", symmetric=True, center="0", image_origin=True, aspect_equal=True)
    lo, hi = v.im_main.get_clim()
    assert lo == pytest.approx(-0.296, abs=0.002) and hi == pytest.approx(0.296, abs=0.002)  # 99th percentile
    y0, y1 = v.ax_main.get_ylim()
    assert y0 > y1  # top-left origin
    p.configure_view(center="median")
    lo, hi = v.im_main.get_clim()
    assert (lo + hi) / 2 == pytest.approx(np.median(m), abs=1e-9)
    assert not p.btn_save_condition.isVisibleTo(p)


def test_frame_player_and_signal(app):
    p, v = _panel(app)
    seen = []
    p.coordsChanged.connect(lambda: seen.append(p.global_coords[0]))
    slider = v.slider_widgets[0]
    from tensorspec.gui.components.stack_player import StackPlayer
    players = [pl for pl in v.sliders_widget.findChildren(StackPlayer) if pl.slider is slider]
    assert len(players) == 1  # one Play per hidden axis (older ones are pending deletion)
    slider.setValue(0)  # the viewer starts on the middle frame
    players[0].toggle()
    players[0]._step()
    players[0]._step()
    players[0].stop()
    assert p.global_coords[0] == 2 and seen and seen[-1] == 2


def test_tools_and_overlays(app):
    p, v = _panel(app)
    got = []
    p.start_tool("line", got.append)
    assert v.tool_active
    from types import SimpleNamespace as NS
    v._tool and v.canvas.callbacks.process("button_press_event", NS(inaxes=v.ax_main, xdata=5.0, ydata=10.0, name="button_press_event"))
    v.canvas.callbacks.process("button_press_event", NS(inaxes=v.ax_main, xdata=25.0, ydata=30.0, name="button_press_event"))
    assert got and got[0]["kind"] == "line" and got[0]["a"] == pytest.approx((10, 5)) and got[0]["b"] == pytest.approx((30, 25))
    assert not v.tool_active
    p.set_overlays([{"kind": "rect", "x0": 2, "x1": 10, "y0": 3, "y1": 9, "color": "magenta"},
                    {"kind": "line", "a": (1, 1), "b": (20, 30)},
                    {"kind": "rect", "x0": 0, "x1": 1, "y0": 0, "y1": 1, "dims": (0, 2)}])  # other dims: skipped
    assert len(v._overlay_artists) == 2


def test_xpeem_map_colour_range_is_live_and_used_by_the_export(tmp_path, monkeypatch):
    """Typed min / max fix the viewer's colour scale at once; Auto and a new layer go back to the
    automatic range; the one-click export saves the range that is shown."""
    import numpy as np
    from PySide6.QtWidgets import QApplication, QFileDialog
    from tensorspec.core.peem import synthetic
    from tensorspec.gui.components.peem_xmcd_panel import PeemXmcdPanel

    app = QApplication.instance() or QApplication([])
    mz = synthetic.stripe_domains(64, 64, period_px=16)
    cp, cm = synthetic.helicity_stacks(mz, a0=0.05, n_cp=2, n_cm=2, noise=0.0)
    pos = synthetic.write_ue49_scan(tmp_path, "X_0002", cp, ["Cr_pos_0000", "Cr_pos_0001"])
    neg = synthetic.write_ue49_scan(tmp_path, "X_0003", cm, ["Cr_neg_0000", "Cr_neg_0001"])
    p = PeemXmcdPanel()
    p._run = lambda fn, done, *a, **k: done(fn(*a, **k))
    p.chk_tpx3.setChecked(False)
    p._add_paths([str(pos), str(neg)])
    p._load()
    n = p._name
    p.service.drift_correct(n, method="central")
    p.service.stack(n)
    p.service.asymmetry(n)
    p._refresh_views(select="pcd")
    app.processEvents()
    view = p.viewer.primary_view()
    auto = p.viewer.color_limits()
    assert p.range_state.text() == "auto" and np.allclose((p.range_min.value(), p.range_max.value()), auto, atol=1e-4)
    p.range_min.setValue(-0.2)
    p.range_max.setValue(0.3)
    assert (view.im_main.norm.vmin, view.im_main.norm.vmax) == (-0.2, 0.3) and p.range_state.text() == "fixed"
    p.range_max.setValue(-0.5)  # max below min: ignored, said so
    assert view.fixed_limits == (-0.2, 0.3) and "above" in p.range_state.text()
    p.range_max.setValue(0.3)
    calls = []
    real = p.service.export_map
    monkeypatch.setattr(p.service, "export_map", lambda *a, **k: (calls.append(k), real(*a, **k))[1])
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "m.tif"), "")))
    p._export()
    assert (calls[-1]["vmin"], calls[-1]["vmax"], calls[-1]["cmap"]) == (-0.2, 0.3, "bwr")
    assert (tmp_path / "m.png").exists()
    p._auto_range()
    assert view.fixed_limits is None and p.range_state.text() == "auto"
    p.range_min.setValue(-0.25)
    p._select("stack", "mean_cp")
    app.processEvents()
    assert view.fixed_limits is None and p.range_state.text() == "auto"
    p.service.unload(n)
