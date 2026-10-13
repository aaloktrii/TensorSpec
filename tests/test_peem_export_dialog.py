"""XPEEM map export dialog: it only collects options for PeemService.export_map."""
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from tensorspec.gui.components.peem_tabs.export_dialog import ExportMapDialog

OPTS = {"height_px": 512, "width_px": 1024, "um_per_px": 0.02, "cmap": "bwr", "vmin": -0.1, "vmax": 0.1,
        "label": "(CM-CP)/(CM+CP)", "title": "d: pcd / data"}


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_defaults_are_the_map_size_and_viewer_look(app):
    d = ExportMapDialog(None, OPTS)
    o = d.options()
    assert o["size_px"] == (1024, 512) and o["crop"] is None and o["values"] and o["pictures"] == ("png",)
    assert o["vmin"] is None and o["cmap"] == "bwr" and o["label"] == "(CM-CP)/(CM+CP)"
    assert not d.rb_roi.isEnabled()


def test_region_aspect_and_typed_limits(app):
    d = ExportMapDialog(None, OPTS, roi_box=(100, 300, 50, 150))
    d.rb_roi.setChecked(True)
    assert d.options()["crop"] == (100, 300, 50, 150) and (d.w_spin.value(), d.h_spin.value()) == (200, 100)
    d.w_spin.setValue(800)                       # aspect kept: 2:1
    assert d.h_spin.value() == 400
    d.rb_typed.setChecked(True)
    d.crop_edit.setText("0, 64, 0, 32")
    d._region_changed()
    assert d.options()["crop"] == (0, 64, 0, 32) and d.options()["size_px"] == (64, 32)
    d.chk_auto_lim.setChecked(False)
    d.vmin.setValue(-0.05)
    d.vmax.setValue(0.05)
    d.chk_pdf.setChecked(True)
    o = d.options()
    assert (o["vmin"], o["vmax"]) == (-0.05, 0.05) and o["pictures"] == ("png", "pdf")
    d.chk_auto_size.setChecked(True)
    assert d.options()["size_px"] is None
