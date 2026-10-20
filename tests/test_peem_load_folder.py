"""PEEM panel "Load Folder": any PEEM folder in one step (core/io/peem_folder.py)."""
import time

import numpy as np
import pytest
import tifffile
from PIL import Image
from test_als_p3b import write_p3b
from test_esm_uview_loader import _acq as uview_acq

from tensorspec.core.io import peem_folder
from tensorspec.core.workspace import global_workspace


def _frames(d, ext, n=4):
    d.mkdir()
    for i, pol in enumerate(["CP", "CM"] * (n // 2)):
        a = (np.random.default_rng(i).random((12, 16)) * 200 + 20).astype(np.uint16 if ext in (".tif", ".tiff") else np.uint8)
        f = d / f"img_{pol}_{i:02d}{ext}"
        tifffile.imwrite(f, a) if ext in (".tif", ".tiff") else Image.fromarray(a).save(f)
    (d / "notes.rtf").write_text("{\\rtf1 notes}")
    (d / "macro.ipf").write_text("#pragma rtGlobals=1")
    (d / "readme.txt").write_text("not a scan table")
    return d


def _p3b(root, name="S2_IF_AL_011"):
    d = root / f"bl11_{name}"
    d.mkdir(parents=True)
    for i, pol in enumerate([0.9, 0.9, -0.9, -0.9]):
        write_p3b(d / f"bl11_{name}#{i:04d}.P3B", np.full((8, 10), 1100 if pol > 0 else 900, np.uint16), pol=pol)
    return d


@pytest.mark.parametrize("ext, kind", [(".tif", "tif"), (".tiff", "tif"), (".png", "images"), (".jpg", "images"),
                                       (".jpeg", "images")])
def test_inspect_frames_and_skip_other_files(tmp_path, ext, kind):
    info = peem_folder.inspect_folder(_frames(tmp_path / "d", ext))
    assert info["kind"] == kind
    assert info["ignored"] == {".rtf": 1, ".ipf": 1, ".txt": 1}


def test_inspect_beamline_and_container(tmp_path):
    p3b = _p3b(tmp_path / "day")
    assert peem_folder.inspect_folder(p3b)["kind"] == "beamline" and peem_folder.inspect_folder(p3b)["format"] == "p3b"
    cl = uview_acq(tmp_path / "day", "CL", 1100)
    assert peem_folder.inspect_folder(cl)["format"] == "uview"
    info = peem_folder.inspect_folder(tmp_path / "day")
    assert info["kind"] == "container" and sorted(info["formats"]) == ["p3b", "uview"]
    with pytest.raises(ValueError, match="no PEEM data found"):
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "notes.rtf").write_text("x")
        peem_folder.inspect_folder(empty)


def test_image_folder_loads_as_a_stack(tmp_path):
    t = peem_folder.load_image_folder(_frames(tmp_path / "pngs", ".png"))
    assert t.value.shape == (4, 12, 16) and sorted(t.metadata["pol"]) == ["CM", "CM", "CP", "CP"]
    assert t.metadata["frame_names"] == sorted(t.metadata["frame_names"])  # name order, as the TIF sequence


def _wait(qapp, p):
    t0 = time.time()
    while p._load_thread is not None and p._load_thread.isRunning() and time.time() - t0 < 60:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()


def test_panel_load_folder_one_step(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog, QMessageBox
    from tensorspec.gui.components.peem_panel import PeemPanel

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warnings.append(a[2])))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: warnings.append(a[2])))
    p = PeemPanel()
    assert not hasattr(p, "btn_load_beamline")  # Load Folder covers beamline folders
    p._exec_dialog = lambda dlg: QDialog.Accepted  # acquisition list: load all (pre-selected)
    folders = {"png": _frames(tmp_path / "pngs", ".png"), "tif": _frames(tmp_path / "tifs", ".tif"),
               "p3b": _p3b(tmp_path / "day", "S2_IF_AL_011")}
    _p3b(tmp_path / "day", "S2_IF_AL_013")
    try:
        for key, folder in folders.items():  # one folder at a time
            p.load_folders([str(folder)])
            _wait(qapp, p)
            assert not warnings, warnings
            assert "| 4 frames" in p.status_label.text(), (key, p.status_label.text())
            if key in ("png", "tif"):
                assert "skipped: 1 .rtf, 1 .ipf, 1 .txt" in p.status_label.text()
        assert p._dataset_name == "bl11_S2_IF_AL_011"
        # a folder of P3B acquisitions: one dataset each by default
        p.load_folders([str(tmp_path / "day")])
        _wait(qapp, p)
        assert "Loaded 2 datasets" in p.status_label.text(), p.status_label.text()
        # or combined, named after the parent folder
        p._exec_dialog = lambda dlg: (getattr(p, "_acq_mode", None) and p._acq_mode.setCurrentIndex(1), QDialog.Accepted)[1]
        p.load_folders([str(tmp_path / "day")])
        _wait(qapp, p)
        assert p._dataset_name == "day" and "| 8 frames" in p.status_label.text(), p.status_label.text()
        assert not warnings, warnings
    finally:
        for n in ("pngs", "tifs", "bl11_S2_IF_AL_011", "bl11_S2_IF_AL_013", "day"):
            global_workspace.remove(n)


def test_panel_loads_several_folders_at_once(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog, QMessageBox
    from tensorspec.gui.components.peem_panel import PeemPanel

    errors = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: errors.append(a[2])))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: errors.append(a[2])))
    picked = [_frames(tmp_path / "pngs2", ".png"), _frames(tmp_path / "tifs2", ".tif"),
              _p3b(tmp_path / "acq", "S2_IF_AL_017"), tmp_path / "empty"]
    (tmp_path / "empty").mkdir()
    p = PeemPanel()
    p._exec_dialog = lambda dlg: QDialog.Accepted
    p._pick_folders = lambda: [str(f) for f in picked]
    try:
        p._load_folder()
        _wait(qapp, p)
        assert len(errors) == 1 and "no PEEM data found" in errors[0]  # the empty folder is reported, rest load
        assert "Loaded 3 datasets" in p.status_label.text(), p.status_label.text()
        for n in ("pngs2", "tifs2", "bl11_S2_IF_AL_017"):
            assert global_workspace.pull_tensor_data(n, "raw") is not None
            assert p.dataset_combo.findText(n) >= 0
    finally:
        for n in ("pngs2", "tifs2", "bl11_S2_IF_AL_017"):
            global_workspace.remove(n)


def test_frame_viewer_pauses_after_reload(qapp):
    """Reloading the viewer while playing must not leave a hidden player moving the frame."""
    import shiboken6
    from PySide6.QtWidgets import QApplication
    from tensorspec.core.data_models import TensorData
    from tensorspec.gui.components.data_viewer_panel import DataViewerPanel
    from tensorspec.gui.components.stack_player import StackPlayer

    td = TensorData(np.random.rand(10, 8, 8).astype("f4"), [np.arange(10.0), np.arange(8.0), np.arange(8.0)],
                    ["frame", "y", "x"], ["", "px", "px"], "image")
    v = DataViewerPanel(compact=True)
    v.load_data(td)
    v.findChildren(StackPlayer)[0].toggle()
    v.load_data(td)
    for _ in range(5):
        qapp.processEvents()
        time.sleep(0.05)
    live = [w for w in QApplication.allWidgets() if isinstance(w, StackPlayer) and shiboken6.isValid(w)]
    assert not any(w.playing for w in live)


def test_combined_dataset_named_after_all_folders():
    from tensorspec.gui.components.peem_panel import PeemPanel

    name = PeemPanel._group_name(["/d/XPEEM_20um_579eV_Cr2p_LH_-15deg", "/d/XPEEM_20um_579eV_Cr2p_LV_-15deg"])
    assert name == "XPEEM_20um_579eV_Cr2p_LH_LV_-15deg"
    assert PeemPanel._group_name(["/d/Cr2p_CL_0deg", "/d/Cr2p_CR_0deg_2"]) == "Cr2p_CL_0deg_CR_0deg_2"
