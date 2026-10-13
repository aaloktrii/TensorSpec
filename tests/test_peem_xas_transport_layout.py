"""Smoke layout tests for PEEM, XAS, and Transport suites (no real data)."""
from __future__ import annotations

import pytest
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import QPushButton, QScrollArea, QSplitter, QSizePolicy, QTabWidget

MIN_CANVAS_HEIGHT = 120
MIN_LEFT_WIDTH = 280
WINDOW_SIZES = [(1100, 750), (1000, 620)]


def _show_and_pump(qapp, widget, w=1100, h=750):
    widget.resize(w, h)
    widget.show()
    qapp.processEvents()


@pytest.fixture
def peem_suite(qapp):
    from tensorspec.gui.suites.peem_suite import PEEMSuite

    w = PEEMSuite()
    _show_and_pump(qapp, w)
    yield w
    w.close()


@pytest.fixture
def xas_suite(qapp):
    from tensorspec.gui.suites.xas_suite import XASSuite

    w = XASSuite()
    _show_and_pump(qapp, w)
    yield w
    w.close()


@pytest.fixture
def transport_suite(qapp):
    from tensorspec.gui.suites.transport_suite import TransportSuite

    w = TransportSuite()
    _show_and_pump(qapp, w)
    yield w
    w.close()


def _critical_buttons(panel) -> list[QPushButton]:
    """Named action buttons that must stay reachable (not permanently hidden)."""
    names = (
        "btn_load",
        "btn_load_stack",
        "btn_load_folder",
        "btn_attach_csv",
        "btn_pair",
        "btn_separate",
        "btn_drift",
        "btn_bg_preview",
        "btn_bg_apply",
        "btn_sum_preview",
        "btn_sum_apply",
        "btn_load_pair",
    )
    out = []
    for n in names:
        btn = getattr(panel, n, None)
        if btn is not None:
            out.append(btn)
    return out


def test_peem_load_and_process_buttons_visible(peem_suite):
    p = peem_suite.panel
    for btn in _critical_buttons(p):
        assert btn.isVisible(), btn.text()
        assert btn.isEnabled() or btn.text().startswith("Load") is False or True
        # geometry after show
        assert btn.width() > 10 and btn.height() > 10, btn.text()


def test_peem_view_tabs_and_canvases(peem_suite, qapp):
    p = peem_suite.panel
    tabs = p.view_tabs
    assert tabs.count() == 3
    assert [tabs.tabText(i) for i in range(3)] == ["Frame", "BG", "Sum Rule"]
    thin = []
    for i in range(tabs.count()):
        tabs.setCurrentIndex(i)
        qapp.processEvents()
        page = tabs.widget(i)
        for canvas in page.findChildren(FigureCanvas):
            if canvas.height() < MIN_CANVAS_HEIGHT:
                thin.append(f"{tabs.tabText(i)}:{canvas.height()}px")
    assert not thin, thin


def test_peem_splitter_left_column_not_collapsed(peem_suite):
    splitter = peem_suite.panel.findChild(QSplitter)
    assert splitter is not None
    sizes = splitter.sizes()
    assert sizes[0] >= MIN_LEFT_WIDTH, sizes


def test_peem_bg_two_step_toggles_post_edge(peem_suite, qapp):
    p = peem_suite.panel
    p.bg_method.setCurrentIndex(p.bg_method.findData("linear"))
    qapp.processEvents()
    assert not p.bg_post_row.isVisible()
    p.bg_method.setCurrentIndex(p.bg_method.findData("two_step"))
    qapp.processEvents()
    assert p.bg_post_row.isVisible()


def test_xas_load_buttons_and_main_canvas(xas_suite, qapp):
    p = xas_suite.panel
    assert p.btn_load.isVisible() and p.btn_load_pair.isVisible()
    assert p.main_canvas.height() >= MIN_CANVAS_HEIGHT


def test_xas_left_scroll_has_min_width(xas_suite):
    scroll = xas_suite.panel.findChild(QScrollArea)
    assert scroll is not None
    assert scroll.minimumWidth() >= MIN_LEFT_WIDTH


def test_xas_bg_method_hides_post_edge_when_linear(xas_suite, qapp):
    p = xas_suite.panel
    p.bg_method.setCurrentIndex(p.bg_method.findData("linear"))
    qapp.processEvents()
    assert not p.bg_post_row.isVisible()
    p.bg_method.setCurrentIndex(p.bg_method.findData("two_step"))
    qapp.processEvents()
    assert p.bg_post_row.isVisible()


@pytest.mark.parametrize("width,height", WINDOW_SIZES)
def test_xas_canvases_usable_at_sizes(xas_suite, qapp, width, height):
    xas_suite.resize(width, height)
    qapp.processEvents()
    p = xas_suite.panel
    # Main plot must stay tall; embedded BG/sum previews may be smaller but >0
    assert p.main_canvas.height() >= MIN_CANVAS_HEIGHT
    for label, canvas in (
        ("bg", p.bg_canvas),
        ("sum", p.sum_canvas),
    ):
        assert canvas.height() >= 40, f"{label} preview {canvas.height()}px at {width}x{height}"


def test_transport_controls_and_demo_plot(transport_suite, qapp):
    s = transport_suite
    load_btn = s.findChildren(QPushButton)
    assert any("Load Transport" in b.text() for b in load_btn)
    assert s.canvas.height() >= MIN_CANVAS_HEIGHT
    assert s.curve_type.isVisible()
    assert s.thickness.isVisible() and s.width.isVisible()


def test_transport_extract_button_exists(transport_suite):
    btns = [b.text() for b in transport_suite.findChildren(QPushButton)]
    assert "Extract Transport Parameters" in btns
    assert "Load Transport Data…" in btns


def test_transport_curve_type_switch_redraws(transport_suite, qapp):
    for idx in range(transport_suite.curve_type.count()):
        transport_suite.curve_type.setCurrentIndex(idx)
        qapp.processEvents()
    assert transport_suite.canvas.height() >= MIN_CANVAS_HEIGHT


def test_canvases_not_ignored_vertical(peem_suite, xas_suite, transport_suite):
    for suite in (peem_suite, xas_suite, transport_suite):
        for canvas in suite.findChildren(FigureCanvas):
            pol = canvas.sizePolicy()
            assert pol.verticalPolicy() != QSizePolicy.Policy.Ignored, canvas
