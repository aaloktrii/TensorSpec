"""One loader for UE49 / UView / P3B (core/io/peem_beamline_loaders.py) and XMCD through the service."""
import os
import numpy as np
import pytest
from test_als_p3b import write_p3b
from test_esm_uview_loader import _acq as uview_acq

from tensorspec.core.io import peem_beamline_loaders
from tensorspec.core.peem import synthetic


@pytest.fixture(autouse=True)
def _private_load_settings(tmp_path, monkeypatch):
    """The load list remembers its folder in QSettings: keep tests out of the user's settings."""
    from PySide6.QtCore import QSettings
    from tensorspec.gui.components.peem_tabs.load_section import LoadSection
    store = QSettings(str(tmp_path / "load_settings.ini"), QSettings.IniFormat)
    monkeypatch.setattr(LoadSection, "_settings", staticmethod(lambda: store))



def _ue49(tmp_path):
    frames = np.stack([np.full((16, 16), v, np.float32) for v in (1100, 1100, 900, 900)])
    names = ["Cr_pos_0001.tif", "Cr_pos_0002.tif", "Cr_neg_0001.tif", "Cr_neg_0002.tif"]
    return synthetic.write_ue49_scan(tmp_path, "X_0002", frames, names)


def _p3b(tmp_path, n=2, shape=(16, 16)):
    d = tmp_path / "S2_IF_AL_011"
    d.mkdir()
    for i in range(2 * n):
        pol = 0.9 if i < n else -0.9
        write_p3b(d / f"S2_IF_AL_011#{i:04d}.P3B", np.full(shape, 1100 if pol > 0 else 900, np.uint16), pol=pol)
    return d


def test_detect_each_format(tmp_path):
    txt = _ue49(tmp_path)
    assert peem_beamline_loaders.detect_format(txt) == ("ue49", txt)
    assert peem_beamline_loaders.detect_format(txt.with_suffix("")) == ("ue49", txt)  # image folder -> table
    cl = uview_acq(tmp_path, "CL", 1100)
    assert peem_beamline_loaders.detect_format(cl)[0] == "uview"
    assert peem_beamline_loaders.detect_format(next(cl.glob("*.dat")))[0] == "uview"
    als = _p3b(tmp_path)
    assert peem_beamline_loaders.detect_format(als)[0] == "p3b"
    (tmp_path / "junk").mkdir()
    with pytest.raises(ValueError, match="no UE49"):
        peem_beamline_loaders.detect_format(tmp_path / "junk")


def test_mixed_formats_refused(tmp_path):
    with pytest.raises(ValueError, match="mixed formats"):
        peem_beamline_loaders.load_beamline([_ue49(tmp_path), _p3b(tmp_path)])


def test_p3b_stack_pol_and_i0_default_off(tmp_path):
    t = peem_beamline_loaders.load_beamline([_p3b(tmp_path)])
    md = t.metadata
    assert t.value.shape == (4, 16, 16)
    assert md["pol"] == ["CP", "CP", "CM", "CM"]
    assert md["format"] == "p3b" and md["default_convention"] == "ue49"
    assert md["i0_normalized"] is False and md["um_per_px"] == pytest.approx(20.76 / 16)


@pytest.mark.parametrize("make, expected", [
    ("ue49", -0.1),   # UE49 sign (neg-pos)/(neg+pos) everywhere: pos (1100), neg (900)
    ("uview", -0.1),  # (CR-CL)/(CR+CL): CL = pos
    ("p3b", -0.1),    # (I-0.9 - I+0.9)/sum: +0.9 = pos
])
def test_service_xmcd_uses_beamline_sign(tmp_path, make, expected):
    from tensorspec.gui.services.peem_service import PeemService
    if make == "ue49":
        paths = [_ue49(tmp_path)]
    elif make == "uview":
        paths = [uview_acq(tmp_path, "CL", 1100), uview_acq(tmp_path, "CR", 900)]
    else:
        paths = [_p3b(tmp_path)]
    svc = PeemService()
    res = svc.load_beamline(paths, name=f"t_{make}", i0_normalize=False)
    assert res["format"] == make
    svc.compute_pcd(f"t_{make}")  # drift correction on, sign from the beamline
    assert np.nanmean(svc.get_map(f"t_{make}", "pcd")) == pytest.approx(expected, abs=1e-6)


def test_drift_roi_stored_with_map(tmp_path):
    from tensorspec.gui.services.peem_service import PeemService
    svc = PeemService()
    svc.load_beamline([_p3b(tmp_path, shape=(64, 64))], name="t_drift")
    r = svc.compute_pcd("t_drift")
    assert r["drift_box"] == "central half (default)" and r["drift_box_y0y1x0x1"] == [16, 48, 16, 48]
    roi = {"kind": "rect", "x0": 4, "x1": 40, "y0": 8, "y1": 60}
    r = svc.compute_pcd("t_drift", reg_roi=roi)
    assert r["drift_box"] == "drawn drift ROI" and r["drift_box_y0y1x0x1"] == [8, 60, 4, 40]
    d = svc.get_drift("t_drift")
    assert len(d["dy"]) == 4 and d["n_a"] == 2
    svc.compute_pcd("t_drift", register=False)
    assert svc.get_drift("t_drift") is None


def test_step_by_step_drift_stack_asymmetry(tmp_path):
    from tensorspec.gui.services.peem_service import PeemService
    svc = PeemService()
    svc.load_beamline([uview_acq(tmp_path, "CL", 1100), uview_acq(tmp_path, "CR", 900)], name="t_steps")
    with pytest.raises(ValueError, match="drift correct"):
        svc.stack("t_steps")
    with pytest.raises(ValueError, match="stack"):
        svc.asymmetry("t_steps")
    r1 = svc.drift_correct("t_steps")
    assert r1["frames"].startswith("6 (CP 3, CM 3)") and svc.get_drift("t_steps", "processed") is not None
    r2 = svc.stack("t_steps")
    assert r2["images"].startswith("CP: 3 frames -> 1 image")
    assert svc.list_products("t_steps") == ["stack"]
    r3 = svc.asymmetry("t_steps")
    assert r3["formula"] == "(CM-CP)/(CM+CP)"
    assert np.nanmean(svc.get_map("t_steps", "pcd")) == pytest.approx(-0.1, abs=1e-6)
    assert np.nanmean(svc.get_map("t_steps", "pcd", "difference")) == pytest.approx(-200.0, abs=1e-3)
    assert svc.get_drift("t_steps", "pcd")["n_a"] == 3


@pytest.mark.parametrize("method, filt", [("central", None), ("marquee", None), ("roi", None), ("roi", "smooth_edges")])
def test_drift_methods_recover_known_drift(tmp_path, method, filt):
    from scipy import ndimage
    from tensorspec.core.peem import registration as R
    from tensorspec.gui.services.peem_service import PeemService
    rng = np.random.default_rng(1)
    base = 1000 + 200 * ndimage.gaussian_filter(rng.normal(size=(160, 160)), 3)
    true = np.array([[0.6 * i, -0.4 * i] for i in range(8)])  # 4 CL then 4 CR frames, steady drift
    d = tmp_path / "acq"
    for tag, idx in (("CL", range(4)), ("CR", range(4, 8))):
        f = tmp_path / f"XPEEM_20um_x_579eV_Cr2p_{tag}_0deg"
        f.mkdir()
        for i in idx:
            from test_esm_uview_loader import _write_uview
            _write_uview(f / f"{i:06d}.dat", np.clip(R.apply_shift(base, -true[i, 0], -true[i, 1]), 0, 65535).astype(np.uint16))
    svc = PeemService()
    svc.load_beamline(sorted(tmp_path.glob("XPEEM_*")), name=f"t_{method}")
    roi = {"kind": "rect", "x0": 40, "x1": 120, "y0": 40, "y1": 120}
    r = svc.drift_correct(f"t_{method}", None if method == "central" else roi, method=method, filt=filt,
                          search_radius=4)
    assert "8 (CP 4, CM 4) aligned together to frame 0" in r["frames"]
    dr = svc.get_drift(f"t_{method}", "processed")
    assert np.abs(dr["dy"] - true[:, 0]).max() < 0.3 and np.abs(dr["dx"] - true[:, 1]).max() < 0.3
    if method == "roi":
        assert dr["score"].min() > 0.9


def test_drift_box_required_for_marquee_and_roi(tmp_path):
    from tensorspec.gui.services.peem_service import PeemService
    svc = PeemService()
    svc.load_beamline([uview_acq(tmp_path, "CL", 1100), uview_acq(tmp_path, "CR", 900)], name="t_nobox")
    with pytest.raises(ValueError, match="Draw the marquee"):
        svc.drift_correct("t_nobox", method="roi")


def test_hand_over_to_peem_stack_bins_energy(tmp_path):
    """CP and CM sweeps on different energy points pair up in energy bins for BG / sum rule."""
    from tensorspec.core.data_models import TensorData
    from tensorspec.core.workspace import global_workspace
    from tensorspec.gui.services.peem_service import PeemService
    e_cp = np.linspace(570, 580, 21)
    e_cm = np.linspace(570.03, 580.03, 21)  # slightly different grid
    frames = np.concatenate([np.full((21, 16, 16), 1100.0), np.full((21, 16, 16), 900.0)]).astype(np.float32)
    md = {"pol": ["CP"] * 21 + ["CM"] * 21, "beamline_table": {"columns": ["energy"],
          "series": {"energy": list(e_cp) + list(e_cm)}}, "format": "ue49", "default_convention": "ue49"}
    t = TensorData(value=frames, axes=[np.arange(42.0), np.arange(16.0), np.arange(16.0)],
                   labels=["frame", "y", "x"], units=["", "px", "px"], data_type="Experimental PEEM", metadata=md)
    global_workspace.push_spectroscopy_data("t_scan", t)
    svc = PeemService()
    with pytest.raises(ValueError, match="drift correction"):
        svc.to_peem_stack("t_scan")
    svc.drift_correct("t_scan")
    r = svc.to_peem_stack("t_scan")
    assert r["name"] == "t_scan_xmcd" and r["n_energies"] == 21 and r["separated"] == ["CM", "CP"]
    assert r["energy_range_eV"][0] == pytest.approx(570.015, abs=0.01)
    bg = svc.bg_apply("t_scan_xmcd", node="processed/CP", method="linear", e0=570, e1=572)
    assert bg["energy_source"] == "csv"


def test_beamline_tokens_go_through_the_original_parser():
    from tensorspec.core.io.peem_loaders import infer_pol_from_name
    from tensorspec.core.io import ue49_loaders, uview_loaders
    assert infer_pol_from_name("Cr_pos_0000") == "unknown"  # original behaviour unchanged
    assert ue49_loaders.pol_from_name("Cr_pos_0000") == "CP"
    assert ue49_loaders.pol_from_name("position_scan_LV") == "LV"  # whole words: 'pos' != 'position'
    assert uview_loaders.pol_from_name("XPEEM_20um_2CA_1AN_579eV_Cr2p_CR_0deg") == "CR"  # not 'Cr2p'


def test_ncc_roi_drift_uses_the_original_engine(tmp_path):
    from scipy import ndimage
    from tensorspec.core.data_models import TensorData
    from tensorspec.core.peem.registration import apply_shift
    from tensorspec.core.workspace import global_workspace
    from tensorspec.gui.services.peem_service import PeemService
    base = 1000 + 300 * ndimage.gaussian_filter(np.random.default_rng(2).normal(size=(96, 96)), 3)
    true = np.array([[i, -i] for i in range(4)], float)
    fr = np.stack([apply_shift(base, -a, -b) for a, b in true]).astype(np.float32)
    global_workspace.push_spectroscopy_data("t_ncc", TensorData(
        value=fr, axes=[np.arange(4.0), np.arange(96.0), np.arange(96.0)], labels=["frame", "y", "x"],
        units=["", "px", "px"], data_type="Experimental PEEM", metadata={"pol": ["CP", "CP", "CM", "CM"]}))
    svc = PeemService()
    r = svc.drift_correct("t_ncc", {"kind": "rect", "x0": 20, "x1": 76, "y0": 20, "y1": 76},
                          method="ncc_roi", search_radius=6)
    assert r["method"].startswith("NCC ROI (original")
    d = svc.get_drift("t_ncc", "processed")
    assert np.allclose(np.c_[d["dy"], d["dx"]], true)


def test_loading_does_not_need_names(tmp_path):
    """Renamed acquisitions load: format from the file contents, polarization from the data
    (UE49 scan table, ALS EPU) or set by the user (UView files do not record it)."""
    import shutil
    # BNL UView: folders without CL / CR in the name
    a, b = tmp_path / "first", tmp_path / "second"
    shutil.move(uview_acq(tmp_path, "CL", 1100), a)
    shutil.move(uview_acq(tmp_path, "CR", 900), b)
    assert peem_beamline_loaders.detect_format(a)[0] == "uview"
    assert peem_beamline_loaders.detect_pol(a)[0] is None
    with pytest.raises(ValueError, match="set its polarization"):
        peem_beamline_loaders.load_beamline([a, b])
    t = peem_beamline_loaders.load_beamline([a, b], pol={str(a): "CP", str(b): "CM"})
    assert set(t.metadata["pol"]) == {"CP", "CM"} and t.metadata["pol"][0] == "CP"
    # ALS: frames renamed without '#NNNN'; a .P3B that is not a raw frame is skipped
    d = _p3b(tmp_path)
    for i, f in enumerate(sorted(d.glob("*.P3B"))):
        f.rename(d / f"frame_{i}.P3B")
    (d / "processed.P3B").write_bytes(b"not a raw frame")
    t = peem_beamline_loaders.load_beamline([d])
    assert t.metadata["pol"] == ["CP", "CP", "CM", "CM"]
    assert peem_beamline_loaders.detect_pol(d)[0] == "per frame"
    with pytest.raises(ValueError, match="cannot be set by hand"):
        peem_beamline_loaders.load_beamline([d], pol={str(d): "CM"})
    # UE49: scan table + image folder renamed together; tokens are in the table's image names
    (tmp_path / "u").mkdir()
    txt = _ue49(tmp_path / "u")
    new = txt.with_name("my_scan.txt")
    txt.rename(new)
    txt.with_suffix("").rename(new.with_suffix(""))
    assert peem_beamline_loaders.detect_pol(new) == (None, "mixed / no polarization in the scan")  # pos + neg
    t = peem_beamline_loaders.load_beamline([new], tpx3=False)
    assert t.metadata["pol"] == ["CP", "CP", "CM", "CM"]


def test_load_list_sets_polarization_of_unnamed_folders(qapp, tmp_path, monkeypatch):
    import shutil
    from PySide6.QtWidgets import QMessageBox
    from tensorspec.core.workspace import global_workspace
    from tensorspec.gui.components.peem_xmcd_panel import PeemXmcdPanel
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: errors.append(a[2])))
    a, b = tmp_path / "first", tmp_path / "second"
    shutil.move(uview_acq(tmp_path, "CL", 1100), a)
    shutil.move(uview_acq(tmp_path, "CR", 900), b)
    p = PeemXmcdPanel()
    p._run = lambda fn, done, *args, **kw: done(fn(*args, **kw))  # synchronous for the test
    p._add_paths([str(a), str(b)])
    assert "? — set the polarization" in p.scan_list.item(0).text()
    assert "Set the polarization of: first, second" in p.load_status.text()
    p._load()
    assert errors and "Set the polarization" in errors[-1] and p._name is None
    for row, tag in ((0, "CP"), (1, "CM")):
        p.scan_list.setCurrentRow(row)
        p.pol_combo.setCurrentIndex(p.pol_combo.findData(tag))
        p._set_item_pol()
    assert p.scan_list.item(1).text().endswith("· CM (set)")
    p._load()
    try:
        assert p._name and global_workspace.pull_tensor_data(p._name, "raw").metadata["pol"].count("CP") > 0
    finally:
        if p._name:
            global_workspace.remove(p._name)


@pytest.mark.parametrize("tags, expected", [(("CL", "CR"), -0.1), (("LH", "LV"), -0.1)])
def test_universal_asymmetry_step_by_step(tmp_path, tags, expected):
    """XMCD = (CM-CP)/(CM+CP), XLD = (LV-LH)/(LV+LH) for every data source (first tag 1100, second 900)."""
    from tensorspec.gui.services.peem_service import PeemService
    svc = PeemService()
    name = f"t_univ_{tags[0]}"
    svc.load_beamline([uview_acq(tmp_path, tags[0], 1100), uview_acq(tmp_path, tags[1], 900)], name=name)
    svc.drift_correct(name, method="central")
    svc.stack(name, pair="xmcd" if tags[0] == "CL" else "xld")
    out = svc.asymmetry(name)
    kind = "pcd" if tags[0] == "CL" else "pld"
    assert out["formula"] == ("(CM-CP)/(CM+CP)" if kind == "pcd" else "(LV-LH)/(LV+LH)")
    assert np.nanmean(svc.get_map(name, kind)) == pytest.approx(expected, abs=1e-6)


def test_find_acquisitions_in_a_folder_of_acquisitions(tmp_path):
    root = tmp_path / "sample"
    (root / "RT").mkdir(parents=True)
    cl, cr = uview_acq(root / "RT", "CL", 1100), uview_acq(root / "RT", "CR", 900)
    loose = next(cl.glob("*.dat"))
    (root / "RT" / "single.dat").write_bytes(loose.read_bytes())  # a loose single-image file
    day = root / "day"
    day.mkdir()
    txt = _ue49(day)
    als = _p3b(root)
    (root / "tensorspec_peem").mkdir()  # outputs are skipped
    found = {str(p.relative_to(root)): fmt for fmt, p in peem_beamline_loaders.find_acquisitions(root)}
    assert found == {"RT/" + cl.name: "uview", "RT/" + cr.name: "uview", "RT/single.dat": "uview",
                     "day/" + txt.name: "ue49", als.name: "p3b"}
    assert peem_beamline_loaders.is_container(root / "RT") and not peem_beamline_loaders.is_container(cl)


def test_add_several_at_once_checklist_and_drop(qapp, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QListWidget
    from tensorspec.gui.components.peem_xmcd_panel import PeemXmcdPanel
    cl, cr = uview_acq(tmp_path, "CL", 1100), uview_acq(tmp_path, "CR", 900)
    als = _p3b(tmp_path)
    p = PeemXmcdPanel()

    def tick_uview(dlg):  # all start selected; the user ⌘-clicks the P3B folder off and clicks Add
        lst = dlg.findChild(QListWidget)
        assert all(lst.item(i).isSelected() for i in range(lst.count()))
        for i in range(lst.count()):
            if not lst.item(i).text().startswith("[uview]"):
                lst.item(i).setSelected(False)
        ok = dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)
        assert ok.text() == "Add selected (2)"
        return QDialog.Accepted

    p._exec_dialog = tick_uview
    p._add_any([str(tmp_path)])  # "Add folders…" on the parent folder
    assert sorted(p._scan_paths) == sorted([str(cl), str(cr)])
    p._clear_scans()
    p._drop_filter._callback([str(cl), str(cr)])  # two folders dropped from Finder
    assert p._scan_paths == [str(cl), str(cr)]
    p._add_any([str(als)])  # an acquisition itself is added directly (no checklist)
    assert p._scan_paths[-1] == str(als)



def test_add_folders_dialog_takes_several_selected_folders(qapp, tmp_path):
    from PySide6.QtCore import QItemSelectionModel
    import os
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QListView
    from tensorspec.gui.components.peem_xmcd_panel import PeemXmcdPanel
    cl, cr = uview_acq(tmp_path, "CL", 1100), uview_acq(tmp_path, "CR", 900)
    p = PeemXmcdPanel()
    p._start_dir = lambda: str(tmp_path)  # the dialog opens in the folder that holds both acquisitions

    def select_two(dlg):  # the user cmd-clicks both folders and presses Choose
        view = dlg.findChild(QListView, "listView")
        model = view.model()
        for _ in range(200):  # the file-system model fills in the background
            QApplication.processEvents()
            root = view.rootIndex()
            rows = [model.index(r, 0, root) for r in range(model.rowCount(root))]
            names = {i.data() for i in rows}
            if {cl.name, cr.name} <= names:
                break
        sel = view.selectionModel()
        for idx in rows:
            if idx.data() in (cl.name, cr.name):
                sel.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
        dlg.findChild(QDialogButtonBox).accepted.emit()
        return dlg.result()

    p._exec_dialog = select_two
    chosen = p._pick_folders()
    assert sorted(map(os.path.realpath, chosen)) == sorted(map(os.path.realpath, [str(cl), str(cr)]))


def test_remove_selected_items_from_the_load_list(qapp, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from tensorspec.gui.components.peem_xmcd_panel import PeemXmcdPanel
    cl, cr = uview_acq(tmp_path, "CL", 1100), uview_acq(tmp_path, "CR", 900)
    lh = uview_acq(tmp_path, "LH", 1000)
    p = PeemXmcdPanel()
    p._add_paths([str(cl), str(cr), str(lh)])
    p._scan_pol[str(cr)] = "CM"
    p.scan_list.item(1).setSelected(True)  # remove CR only
    p._remove_selected()
    assert p._scan_paths == [str(cl), str(lh)] and p.scan_list.count() == 2 and str(cr) not in p._scan_pol
    assert p.scan_list.item(1).toolTip() == str(lh)  # list rows stay matched to their paths
    p.show()
    p.scan_list.setFocus()
    p.scan_list.clearSelection()
    p.scan_list.item(0).setSelected(True)
    QTest.keyClick(p.scan_list, Qt.Key_Delete)  # the Delete key does the same
    qapp.processEvents()
    assert p._scan_paths == [str(lh)]
    errors = []
    p._err = errors.append
    p.scan_list.clearSelection()
    p._remove_selected()
    assert errors and p._scan_paths == [str(lh)]


def test_add_dialogs_open_in_the_folder_used_last_time(qapp, tmp_path, monkeypatch):
    """Add files / Add folders start where the last item came from, also after Clear all and in a
    new panel (a restart): the folder is kept in the settings."""
    from tensorspec.gui.components.peem_xmcd_panel import PeemXmcdPanel
    day = tmp_path / "beamtime" / "110K"
    day.mkdir(parents=True)
    cl = uview_acq(day, "CL", 1100)
    p = PeemXmcdPanel()
    assert p._start_dir() == os.path.expanduser("~")  # nothing remembered yet
    p._add_paths([str(cl)])
    assert p._start_dir() == str(day)
    p._clear_scans()
    assert p._start_dir() == str(day)
    assert PeemXmcdPanel()._start_dir() == str(day)  # a new window (restart) too
