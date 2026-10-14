"""Generic DataViewerPanel features (cmap / Sym± / draw / Play; off by default for ARPES / ML)."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from tensorspec.core.data_models import TensorData  # noqa: E402
from tensorspec.gui.components.data_viewer_panel import DataViewerPanel, plane_layer  # noqa: E402


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


# XPEEM-map colour-range export test lives with PR #6 (needs peem_xmcd_panel).
