"""End-to-end PEEM service (XMCD-imaging steps) on synthetic UE49 scans, plus an offscreen suite smoke test."""
import numpy as np
import pytest

from tensorspec.core.peem import synthetic
from tensorspec.core.workspace import global_workspace
from tensorspec.gui.services.peem_service import PeemService


@pytest.fixture
def xmcd_dataset(tmp_path):
    mz = synthetic.stripe_domains(64, 64, period_px=16)
    cp, cm = synthetic.helicity_stacks(mz, a0=0.05, n_cp=3, n_cm=3, noise=2.0, shift_cm=(0.0, 1.0))
    pos = synthetic.write_ue49_scan(tmp_path, "X_0002", cp, [f"Cr_pos_000{i}" for i in range(3)])
    neg = synthetic.write_ue49_scan(tmp_path, "X_0003", cm, [f"Cr_neg_000{i}" for i in range(3)])
    svc = PeemService()
    res = svc.load_ue49([pos, neg], name="synth_xmcd", tpx3=False)
    yield svc, res, mz
    global_workspace.remove("synth_xmcd")


def test_load_pcd_stats_domain_map(xmcd_dataset):
    svc, res, mz = xmcd_dataset
    assert res["pol_counts"] == {"CM": 3, "CP": 3}
    pcd = svc.compute_pcd("synth_xmcd", convention="cp_minus_cm")
    # 64x64 image, 32x32 central registration box: ~0.3 px estimator bias is expected here
    assert pcd["cm_shift_dx"] == pytest.approx(-1.0, abs=0.4)
    a = svc.get_map("synth_xmcd", "pcd")
    inner = (slice(4, -4), slice(4, -4))
    assert np.nanmean(np.sign(a[inner]) == mz[inner]) > 0.9

    stats = svc.roi_statistics("synth_xmcd", "pcd", roi={"kind": "rect", "x0": 4, "y0": 4, "x1": 60, "y1": 60},
                               threshold=0.0)
    assert stats["fractions"]["f_up"] == pytest.approx(0.5, abs=0.05)

    dm = svc.compute_domain_map("synth_xmcd", threshold=0.2, min_px=4, smooth_sigma=1.0, use_sda=False)
    assert dm["f_up"] == pytest.approx(0.5, abs=0.06)
    assert set(svc.list_products("synth_xmcd")) == {"pcd", "domain_map"}
    # /analysis nodes are ordinary DataTree nodes the DataViewer can pull
    td = global_workspace.pull_tensor_data("synth_xmcd", "analysis/domain_map")
    assert td.value.shape == (64, 64) and td.labels == ["y", "x"]
    meta = svc.xmcd_meta("synth_xmcd")
    assert meta["products"] == ["pcd", "domain_map"] and meta["loader"] == "ue49_scan"


def test_existing_peem_ops_still_work_on_ue49_data(xmcd_dataset):
    svc, _, _ = xmcd_dataset
    paired = svc.pair("synth_xmcd", "CP_CM")
    assert paired["n_pairs"] == 3
    out = svc.compute_pcd("synth_xmcd", node="processed")
    assert out["n_cp"] == 3


def test_pmld_service(tmp_path):
    ny = nx = 32
    yy, _ = np.mgrid[0:ny, 0:nx]
    mag = np.where(yy < 16, 0.04, -0.04)
    energy = np.array([583.0, 584.4, 585.4, 586.0])

    def frames(sign):
        return np.stack([1000 * (1 + sign * ((mag if e == 584.4 else -mag if e == 585.4 else 0 * mag) + 0.02))
                         for e in energy])

    lh = synthetic.write_ue49_scan(tmp_path, "Mono_S_0226", frames(+1), [f"Img_000{i}" for i in range(4)],
                                   energy=energy, pol_table=0)
    lv = synthetic.write_ue49_scan(tmp_path, "Mono_S_0227", frames(-1), [f"Img_000{i}" for i in range(4)],
                                   energy=energy, pol_table=1)
    svc = PeemService()
    svc.load_ue49([lh, lv], name="synth_xmld", tpx3=False)
    try:
        out = svc.compute_pmld("synth_xmld", 584.4, 585.4, register=False)
        assert out["kinds"] == ["pld", "pmld", "pmnld"]
        # LH = 1000(1 + ld), LV = 1000(1 - ld); universal XLD (LV-LH)/(LV+LH) = -ld
        assert np.allclose(svc.get_map("synth_xmld", "pmld"), -mag, atol=1e-3)
        assert np.allclose(svc.get_map("synth_xmld", "pmnld"), -0.02, atol=1e-3)
        with pytest.raises(ValueError, match="no CP frames"):
            svc.compute_pcd("synth_xmld")
    finally:
        global_workspace.remove("synth_xmld")


def test_peem_suite_tabs_share_one_service(qapp):
    """The original PEEM panel stays the first tab, unchanged; the XPEEM map is added next to it."""
    from tensorspec.gui.suites.peem_suite import PEEMSuite

    suite = PEEMSuite()
    assert [suite.tabs.tabText(i) for i in range(suite.tabs.count())] == ["PEEM", "XPEEM map"]
    assert suite.tabs.currentWidget() is suite.panel
    assert suite.xmcd_panel.service is suite.panel.service is suite.service  # one PEEM service
    assert hasattr(suite.panel, "btn_pair") and hasattr(suite.panel, "btn_drift")  # original flow kept
    suite.close()


def test_domain_map_bg_roi_sets_zero(xmcd_dataset):
    svc, _, _ = xmcd_dataset
    svc.compute_pcd("synth_xmcd", convention="cp_minus_cm")
    from tensorspec.core.peem import products
    a = svc.get_map("synth_xmcd", "pcd") + 0.04  # helicity-dependent offset
    global_workspace.write_analysis_data("synth_xmcd", "pcd", products.map_dataset("pcd", a))
    biased = svc.compute_domain_map("synth_xmcd", threshold=0.2, normalize="p99", use_sda=False)
    # a stripe pair as "background" has zero mean magnetization
    fixed = svc.compute_domain_map("synth_xmcd", threshold=0.2, use_sda=False,
                                   bg_roi={"kind": "rect", "x0": 4, "y0": 4, "x1": 60, "y1": 60})
    assert biased["f_down"] < 0.1  # the offset pushes down domains into the neutral band
    assert fixed["f_up"] == pytest.approx(0.5, abs=0.08)
    assert fixed["f_down"] == pytest.approx(0.5, abs=0.08)
    assert fixed["bg_roi_offset"] == pytest.approx(0.04, abs=0.01)


def test_steps_logged_in_history_and_edits_kept_out_of_raw(xmcd_dataset):
    svc, _, _ = xmcd_dataset
    svc.set_excluded("synth_xmcd", [1])
    assert "stack_edits" not in (global_workspace.pull_tensor_data("synth_xmcd", "raw").metadata or {})
    assert svc.get_edits("synth_xmcd")["exclude"] == [1]
    svc.drift_correct("synth_xmcd", method="central")
    svc.stack("synth_xmcd")
    svc.asymmetry("synth_xmcd", convention="cp_minus_cm")
    assert "stack_edits" not in svc.list_products("synth_xmcd")  # not a viewer layer
    log = global_workspace.get("synth_xmcd")["tree"]["history"].attrs["log"]
    steps = [line.split("] ", 1)[1].split(":", 1)[0] for line in log if "] PEEM " in line]
    assert steps == ["PEEM load", "PEEM stack edits", "PEEM drift correct", "PEEM stack", "PEEM asymmetry"]
    assert '"method": "cross-correlation, central half (sub-pixel)"' in next(l for l in log if "PEEM drift" in l)


def test_stack_step_shows_stacked_images_and_sums(qapp, xmcd_dataset):
    svc, _, _ = xmcd_dataset
    from tensorspec.gui.components.peem_xmcd_panel import PeemXmcdPanel
    p = PeemXmcdPanel(service=svc)
    p._run = lambda fn, done, *args, **kw: done(fn(*args, **kw))  # synchronous for the test
    p._name = "synth_xmcd"
    p._refresh_views(select="raw")
    svc.drift_correct("synth_xmcd")
    p._stack_step()
    st = global_workspace.pull_analysis_data("synth_xmcd", "stack")
    labels = [p.stack_out.itemText(i) for i in range(p.stack_out.count())]
    assert labels[0] == "Show: CP stacked (mean, 3 frames)" and "CP + CM" in labels[2]
    assert [p.stack_out.itemData(i) for i in range(p.stack_out.count())] == \
        [v for _, v in svc.stack_outputs("synth_xmcd")] == ["mean_cp", "mean_cm", "data", "sum_cp", "sum_cm"]
    for i in range(p.stack_out.count()):
        p.stack_out.setCurrentIndex(i)
        var = p.stack_out.currentData()
        assert p._view_kind() == "stack" and (p._view_var() or "data") == var
        assert np.allclose(p._current_image(), st[var].values, equal_nan=True)
    assert np.allclose(st["sum_cp"].values, 3 * st["mean_cp"].values)
    assert np.allclose(st["data"].values, st["mean_cp"].values + st["mean_cm"].values)


def test_stacked_frames_view_and_send_to_xas_tab(qapp, xmcd_dataset):
    svc, _, _ = xmcd_dataset
    from tensorspec.gui.suites.peem_suite import PEEMSuite
    suite = PEEMSuite()
    p = suite.xmcd_panel
    suite.service = p.service = suite.panel.service = svc
    p._run = lambda fn, done, *args, **kw: done(fn(*args, **kw))
    p._name = "synth_xmcd"
    svc.drift_correct("synth_xmcd")
    p._refresh_views(select="stacked")
    assert p.view_combo.currentData() == "stacked"
    stack, tags, *_ = svc.stacked_frames("synth_xmcd")
    assert tags == ["CP", "CM"] and stack.shape[0] == 2  # one energy: CP image, CM image
    assert np.allclose(p.viewer.tensor_data.value, stack)
    assert not p._send_row.isHidden() and p.btn_to_xas.text() == "→ PEEM tab"
    p.pair_combo.setCurrentIndex(p.pair_combo.findData("xmcd"))
    for src, dataset, n in (("stacked", "synth_xmcd_xmcd", 2), ("processed", "synth_xmcd_xmcd_frames", 6)):
        p.send_source.setCurrentIndex(p.send_source.findData(src))
        p._send_to_xas()
        assert suite.tabs.currentWidget() is suite.panel  # the PEEM tab now shows the pushed images
        assert suite.panel._dataset_name == dataset
        assert global_workspace.pull_tensor_data(dataset, "raw").value.shape[0] == n
        assert global_workspace.pull_tensor_data(dataset, "processed/CP") is not None  # paired + separated
        global_workspace.remove(dataset)
    suite.close()


def test_numbers_can_be_typed_for_line_box_and_rois(qapp, xmcd_dataset):
    svc, _, _ = xmcd_dataset
    from PySide6.QtWidgets import QAbstractSpinBox
    from tensorspec.gui.components.peem_xmcd_panel import PeemXmcdPanel
    p = PeemXmcdPanel(service=svc)
    p._name = "synth_xmcd"
    p._refresh_views(select="raw")
    p.line_coords.setText("4, 5, 60, 50")
    p._set_line_coords()
    assert p._line == ((5.0, 4.0), (50.0, 60.0))
    p.box_coords.setText("40 40 8 8")  # any order / separator
    p._set_box_coords()
    assert p._reg_roi == {"kind": "rect", "x0": 8, "x1": 40, "y0": 8, "y1": 40}
    p.roi_coords.setText("1, 2, 30, 31")
    p._set_rect_roi("roi")
    p.bg_coords.setText("32, 32, 60, 60")
    p._set_rect_roi("bg")
    assert p._roi["x1"] == 30 and p._bg_roi["y0"] == 32
    # number fields hold only numbers (units are in the labels)
    own = [s for s in p.findChildren(QAbstractSpinBox) if hasattr(s, "prefix") and not p.viewer.isAncestorOf(s)]
    assert own and all(not s.prefix() and not s.suffix() for s in own)


def test_several_profiles_clear_and_illustrator_pdf(qapp, xmcd_dataset, tmp_path, monkeypatch):
    svc, _, _ = xmcd_dataset
    from PySide6.QtWidgets import QFileDialog
    from tensorspec.gui.components.peem_xmcd_panel import PeemXmcdPanel
    p = PeemXmcdPanel(service=svc)
    p._name = "synth_xmcd"
    p._refresh_views(select="raw")
    for coords in ("4, 10, 60, 10", "4, 30, 60, 30", "4, 50, 60, 50"):
        p.line_coords.setText(coords)
        p._set_line_coords()
        p._an_profile()
    assert len(p._profiles) == 3 and len(p.pfig.axes[0].lines) == 3
    assert sum(ov["kind"] == "line" for ov in p.viewer.overlays) >= 3  # each profile's line on the image
    p.profile_offset.setValue(100.0)  # stacked with an offset
    ys = [ln.get_ydata() for ln in p.pfig.axes[0].lines]
    assert np.allclose(np.asarray(ys[1]) - np.asarray(p._profiles[1]["values"]), 100.0)
    pdf = tmp_path / "profiles.pdf"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(pdf), "")))
    p._export_plot_pdf()
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF") and b"/FontFile2" in data  # TrueType embedded: text stays editable
    assert (tmp_path / "profiles.csv").read_text().startswith("distance_1_")
    p._an_clear_profiles()
    assert p._profiles == [] and not any(ov.get("color") == "#1f77b4" for ov in p.viewer.overlays)


def test_regions_and_profiles_live_in_the_back_end(qapp, xmcd_dataset):
    """Any front end sees the same regions / profiles: they are stored in the dataset by the service."""
    svc, _, _ = xmcd_dataset
    box = {"kind": "rect", "x0": 8, "x1": 40, "y0": 8, "y1": 40}
    svc.set_roi("synth_xmcd", "drift_box", box)  # e.g. set from a script
    pr = svc.add_profile("synth_xmcd", (10, 4), (10, 60), halfwidth=2)
    assert svc.get_rois("synth_xmcd")["drift_box"] == box and svc.get_rois("synth_xmcd")["roi"] is None
    assert len(pr["values"]) > 10 and svc.list_profiles("synth_xmcd")[0]["line"] == [[10.0, 4.0], [10.0, 60.0]]
    assert global_workspace.pull_analysis_data("synth_xmcd", "annotations") is not None  # with the data
    assert "annotations" not in svc.list_products("synth_xmcd")  # not a map layer
    from tensorspec.gui.components.peem_xmcd_panel import PeemXmcdPanel
    p = PeemXmcdPanel(service=svc)
    p._name = "synth_xmcd"
    assert p._reg_roi == box and len(p._profiles) == 1  # the Qt panel shows what the service holds
    p._roi = {"kind": "rect", "x0": 1, "x1": 9, "y0": 1, "y1": 9}  # drawn in the panel ...
    assert svc.get_rois("synth_xmcd")["roi"]["x1"] == 9  # ... is stored by the service
    p._an_clear_profiles()
    assert svc.list_profiles("synth_xmcd") == []
    log = global_workspace.get("synth_xmcd")["tree"]["history"].attrs["log"]
    assert any("PEEM set drift_box" in line for line in log) and any("PEEM add profile" in line for line in log)


def test_display_hint_is_a_back_end_call(qapp, xmcd_dataset):
    svc, _, _ = xmcd_dataset
    from tensorspec.plotting import peem_plots
    svc.compute_pcd("synth_xmcd", convention="cp_minus_cm")
    assert svc.display_hint("synth_xmcd", "raw", "", 0) == {"cmap": "gray", "symmetric": False, "center": "0"}
    assert svc.display_hint("synth_xmcd", "pcd") == {"cmap": "bwr", "symmetric": True, "center": "0"}
    offset = np.full((8, 8), 0.05) + np.random.default_rng(0).normal(0, 0.01, (8, 8))  # map with a flux offset
    assert peem_plots.display_hint("pcd", "data", offset)["center"] == "median"
    from tensorspec.gui.components.peem_xmcd_panel import PeemXmcdPanel
    p = PeemXmcdPanel(service=svc)
    p._name = "synth_xmcd"
    p._refresh_views(select="pcd")  # the panel applies the service's choice
    assert p.viewer.primary_view().combo_cmap.currentText() == "bwr"



def test_show_lists_come_from_the_back_end(xmcd_dataset):
    svc, _, _ = xmcd_dataset
    assert svc.stack_outputs("synth_xmcd") == [] and svc.asymmetry_outputs("synth_xmcd") == []
    svc.drift_correct("synth_xmcd")
    svc.stack("synth_xmcd")
    svc.asymmetry("synth_xmcd")
    assert svc.asymmetry_outputs("synth_xmcd") == [
        ("asymmetry", "data"), ("difference", "difference"),
        ("first stacked image (CP)", "mean_cp"), ("second stacked image (CM)", "mean_cm")]
    assert svc.asymmetry_outputs("synth_xmcd", "xld") == []  # no XLD map


def test_asymmetry_show_list_filled_by_the_service(qapp, xmcd_dataset):
    svc, _, _ = xmcd_dataset
    from tensorspec.gui.components.peem_xmcd_panel import PeemXmcdPanel
    p = PeemXmcdPanel(service=svc)
    p._run = lambda fn, done, *args, **kw: done(fn(*args, **kw))
    p._name = "synth_xmcd"
    p._refresh_views(select="raw")
    assert p.asym_out.currentData() is None  # placeholder before the step
    svc.drift_correct("synth_xmcd")
    p._stack_step()
    p._asym_step()
    items = [(p.asym_out.itemText(i), p.asym_out.itemData(i)) for i in range(p.asym_out.count())]
    assert items[0] == ("Show: asymmetry", "data") and items[-1][1] == "mean_cm"
    p.asym_out.setCurrentIndex(1)
    assert p._view_kind() == "pcd" and p._view_var() == "difference"


def test_xpeem_images_go_through_the_original_background_step(tmp_path):
    """An energy scan from the XPEEM map, sent to the PEEM tab, runs the original BG step."""
    energy = np.linspace(570.0, 595.0, 26)
    edge = 1.0 + 0.8 * np.exp(-0.5 * ((energy - 577.0) / 1.0) ** 2) + 0.4 * np.exp(-0.5 * ((energy - 586.0) / 1.2) ** 2)
    rng = np.random.default_rng(1)
    frames = lambda s: np.stack([np.full((16, 16), 1000 * e * (1 + s)) + rng.normal(0, 5, (16, 16)) for e in edge])
    pos = synthetic.write_ue49_scan(tmp_path, "Mono_S_0001", frames(0.02), [f"Cr_pos_{i:04d}" for i in range(26)], energy=energy)
    neg = synthetic.write_ue49_scan(tmp_path, "Mono_S_0002", frames(-0.02), [f"Cr_neg_{i:04d}" for i in range(26)], energy=energy)
    svc = PeemService()
    svc.load_ue49([pos, neg], name="synth_scan", tpx3=False)
    try:
        svc.drift_correct("synth_scan", method="central")
        n = svc.to_peem_stack("synth_scan", "xmcd")["name"]
        bg = svc.bg_preview(n, node="processed/CP", method="linear", e0=570.0, e1=573.0)
        assert len(bg["bg"]) == 26 and np.all(np.isfinite(bg["bg"]))
        global_workspace.remove(n)
    finally:
        global_workspace.remove("synth_scan")


def test_export_map_writes_float_tiff_and_a_colour_png(tmp_path):
    """The TIFF keeps the asymmetry values (they look blank in ordinary viewers); the PNG next to it
    is the coloured picture, scaled like the viewer."""
    import tifffile
    from matplotlib.image import imread

    from tensorspec.core.peem import synthetic
    from tensorspec.gui.services.peem_service import PeemService

    mz = synthetic.stripe_domains(64, 64, period_px=16)
    cp, cm = synthetic.helicity_stacks(mz, a0=0.05, n_cp=2, n_cm=2, noise=0.0)
    pos = synthetic.write_ue49_scan(tmp_path, "X_0002", cp, ["Cr_pos_0000", "Cr_pos_0001"])
    neg = synthetic.write_ue49_scan(tmp_path, "X_0003", cm, ["Cr_neg_0000", "Cr_neg_0001"])
    s = PeemService()
    n = s.load_beamline([pos, neg], "export", tpx3=False)["name"]
    s.drift_correct(n, method="central"); s.stack(n); s.asymmetry(n)
    tif = s.export_map_tiff(n, "pcd", tmp_path / "map.tif")
    vals = tifffile.imread(tif)
    assert vals.dtype == np.float32 and np.nanmax(np.abs(vals)) < 0.2
    rgb = imread(tmp_path / "map.png")[..., :3]
    assert rgb.std(axis=-1).max() > 0.2  # coloured (red / blue), not gray or blank
    assert not (tmp_path / "raw.png").exists()
    s.export_map_tiff(n, "stack", tmp_path / "raw.tif", "data", png=False)
    assert (tmp_path / "raw.tif").exists() and not (tmp_path / "raw.png").exists()


def test_export_map_with_options_crop_size_formats_and_limits(tmp_path):
    import tifffile
    from matplotlib.image import imread

    from tensorspec.core.peem import synthetic
    from tensorspec.gui.services.peem_service import PeemService

    mz = synthetic.stripe_domains(64, 64, period_px=16)
    cp, cm = synthetic.helicity_stacks(mz, a0=0.05, n_cp=2, n_cm=2, noise=0.0)
    pos = synthetic.write_ue49_scan(tmp_path, "X_0002", cp, ["Cr_pos_0000", "Cr_pos_0001"])
    neg = synthetic.write_ue49_scan(tmp_path, "X_0003", cm, ["Cr_neg_0000", "Cr_neg_0001"])
    s = PeemService()
    n = s.load_beamline([pos, neg], "opts", tpx3=False)["name"]
    s.drift_correct(n, method="central"); s.stack(n); s.asymmetry(n)
    o = s.export_options(n, "pcd")
    assert (o["width_px"], o["height_px"]) == (64, 64) and o["cmap"] == "bwr" and o["vmin"] < 0 < o["vmax"]
    files = s.export_map(n, "pcd", tmp_path / "cut", crop=(10, 50, 4, 24), pictures=("png", "pdf", "svg"),
                         size_px=(400, 200), dpi=100, cmap="RdBu_r", vmin=-0.1, vmax=0.1, scalebar=True)
    assert [f.rsplit(".", 1)[1] for f in files] == ["tif", "png", "pdf", "svg"]
    assert tifffile.imread(tmp_path / "cut.tif").shape == (20, 40)       # crop: y 4-24, x 10-50
    assert imread(tmp_path / "cut.png").shape[:2] == (200, 400)          # exactly the asked size
    s.export_map(n, "pcd", tmp_path / "bare.png", values=False, size_px=(128, 128), image_only=True)
    assert not (tmp_path / "bare.tif").exists() and imread(tmp_path / "bare.png").shape[:2] == (128, 128)
    with pytest.raises(ValueError, match="crop region is empty"):
        s.export_map(n, "pcd", tmp_path / "x", crop=(5, 5, 0, 10))
    with pytest.raises(ValueError, match="maximum must be above"):
        s.export_map(n, "pcd", tmp_path / "x", vmin=0.1, vmax=-0.1)


def test_stack_sum_layer_is_named_after_its_channels(tmp_path):
    """The stack's own image (CP + CM) shows as 'stack · cp+cm' in the viewer, from the stack's data."""
    from tensorspec.core.peem import synthetic
    from tensorspec.gui.services.peem_service import PeemService

    mz = synthetic.stripe_domains(32, 32, period_px=8)
    cp, cm = synthetic.helicity_stacks(mz, a0=0.05, n_cp=2, n_cm=2, noise=0.0)
    pos = synthetic.write_ue49_scan(tmp_path, "X_0002", cp, ["Cr_pos_0000", "Cr_pos_0001"])
    neg = synthetic.write_ue49_scan(tmp_path, "X_0003", cm, ["Cr_neg_0000", "Cr_neg_0001"])
    s = PeemService()
    n = s.load_beamline([pos, neg], "layers", tpx3=False)["name"]
    s.drift_correct(n, method="central"); s.stack(n); s.asymmetry(n)
    layers = s.map_layers(n)
    assert "stack · cp+cm" in layers and "stack" not in layers and "pcd" in layers
    assert np.array_equal(layers["stack · cp+cm"], s.get_map(n, "stack", "data"))
    assert s.layer_for(n, "stack") == "stack · cp+cm" and s.parse_layer("stack · cp+cm") == ("stack", "data")
    assert s.layer_for(n, "stack", "sum_cp") == "stack · sum_cp" and s.layer_for(n, "pcd") == "pcd"
    s.unload(n)
