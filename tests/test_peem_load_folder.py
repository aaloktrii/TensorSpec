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


def test_panel_load_folder_one_step(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox
    from tensorspec.gui.components.peem_panel import PeemPanel

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warnings.append(a[2])))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: warnings.append(a[2])))
    p = PeemPanel()
    p._exec_dialog = lambda dlg: QDialog.Accepted  # folder of acquisitions: load all (pre-selected)
    folders = {"png": _frames(tmp_path / "pngs", ".png"), "tif": _frames(tmp_path / "tifs", ".tif"),
               "p3b": _p3b(tmp_path / "day", "S2_IF_AL_011"), "container": tmp_path / "day"}
    _p3b(tmp_path / "day", "S2_IF_AL_013")
    expected = {"png": 4, "tif": 4, "p3b": 4, "container": 8}
    try:
        for key, folder in folders.items():
            p.name_edit.clear()
            monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, f=folder, **k: str(f)))
            p._load_folder()
            t0 = time.time()
            while p._load_thread is not None and p._load_thread.isRunning() and time.time() - t0 < 60:
                qapp.processEvents()
                time.sleep(0.02)
            qapp.processEvents()
            assert not warnings, warnings
            assert f"| {expected[key]} frames" in p.status_label.text(), (key, p.status_label.text())
            if key in ("png", "tif"):
                assert "skipped: 1 .rtf, 1 .ipf, 1 .txt" in p.status_label.text()
        assert p._dataset_name == "day"  # a folder of acquisitions is one dataset named after it
    finally:
        for n in ("pngs", "tifs", "bl11_S2_IF_AL_011", "day"):
            global_workspace.remove(n)
