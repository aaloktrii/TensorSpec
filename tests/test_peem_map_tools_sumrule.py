"""PEEM panel calculate / analyse sections (peem_map_tools.py) and the Sum Rule window (XAS suite,
From PEEM tab: peem_sumrule_panel.py), on the PEEM panel's own load -> Stack Pairs flow."""
import gc

import numpy as np
import pytest

from tensorspec.core.peem import synthetic
from tensorspec.core.workspace import global_workspace
from tensorspec.gui.services.peem_service import PeemService

E = np.linspace(570.0, 595.0, 26)
EDGE = 1 + 0.8 * np.exp(-0.5 * ((E - 577) / 1.0) ** 2) + 0.4 * np.exp(-0.5 * ((E - 586) / 1.2) ** 2)


def _images(tmp_path, s, name="img"):
    mz = synthetic.stripe_domains(64, 64, period_px=16)
    cp, cm = synthetic.helicity_stacks(mz, a0=0.05, n_cp=3, n_cm=3, noise=1.0)
    pos = synthetic.write_ue49_scan(tmp_path, "X_0002", cp, [f"Cr_pos_000{i}" for i in range(3)])
    neg = synthetic.write_ue49_scan(tmp_path, "X_0003", cm, [f"Cr_neg_000{i}" for i in range(3)])
    s.load_beamline([pos, neg], name, tpx3=False)
    s.pair(name, "CP_CM")
    return name


def _scan(tmp_path, s, name="scan"):
    rng = np.random.default_rng(1)
    frames = lambda f: np.stack([np.full((16, 16), 1000 * e * (1 + f)) + rng.normal(0, 5, (16, 16)) for e in EDGE])
    a = synthetic.write_ue49_scan(tmp_path, "Mono_S_0001", frames(0.02), [f"Cr_pos_{i:04d}" for i in range(26)], energy=E)
    b = synthetic.write_ue49_scan(tmp_path, "Mono_S_0002", frames(-0.02), [f"Cr_neg_{i:04d}" for i in range(26)], energy=E)
    s.load_beamline([a, b], name, tpx3=False)
    s.pair(name, "CP_CM")
    return name


@pytest.fixture(autouse=True)
def _clean_workspace():
    yield
    for name in list(getattr(global_workspace, "_data", {})):
        if name.startswith(("img", "scan")):  # incl. scan_mix
            global_workspace.remove(name)


def test_channel_stack_energy_window_and_asymmetry(tmp_path):
    s = PeemService()
    n = _scan(tmp_path, s)
    st = s.channel_stack(n, 575.0, 579.0)
    assert st["n_pairs"] == 5 and st["energy_window_eV"] == [575.0, 579.0]
    assert s.asymmetry(n)["formula"] == "(CM-CP)/(CM+CP)"
    assert np.nanmean(s.get_map(n, "pcd")) == pytest.approx(-0.02, abs=0.003)  # CM 2 % below CP
    with pytest.raises(ValueError, match="no CP or CM frame"):
        s.channel_stack(n, 600.0, 610.0)
    with pytest.raises(ValueError, match="Stack Pairs"):
        s.load_path  # noqa: B018  (the panel's own TIF loader is untouched)
        s.channel_stack("missing")


def test_to_sum_rule_gives_background_and_sum_rule_an_energy_axis(tmp_path):
    s = PeemService()
    n = _scan(tmp_path, s)
    sr = s.to_sum_rule(n)
    assert sr["name"] == "scan_sumrule" and sr["energy_range_eV"] == [570.0, 595.0] and sr["channels"] == ["CM", "CP"]
    assert s.sum_rule_datasets() == ["scan_sumrule"]
    spec = s.roi_spectra(sr["name"], {"kind": "rect", "x0": 2, "x1": 14, "y0": 2, "y1": 14})
    assert spec["energy"][0] == 570.0 and spec["energy"][-1] == 595.0 and spec["n_pixels"] == 13 * 13  # ROI edges inclusive
    bg = s.bg_preview(sr["name"], node="processed/CP", method="linear", e0=570.0, e1=573.0, use_roi=True,
                      roi_dict={"kind": "rect", "x0": 2, "x1": 14, "y0": 2, "y1": 14})
    assert len(bg["energy"]) == 26 and bg["energy"][0] == 570.0
    res = s.sumrule_preview(sr["name"], nh=3, l3_lo=574, l3_hi=582, l2_lo=582, l2_hi=592, r_lo=570, r_hi=595)
    assert np.isfinite(res["m_orb"]) and np.isfinite(res["m_spin_plus_dipole"])
    assert global_workspace.pull_tensor_data(n, "processed").value.ndim == 4  # the original is unchanged


def test_domain_width_of_stripes(tmp_path):
    s = PeemService()
    n = _images(tmp_path, s)
    s.channel_stack(n)
    s.asymmetry(n)
    w = s.domain_width(n)
    assert 8 <= w["domain_width_px"] <= 8 * 1.4  # stripes of period 16 px: width 8 (straight stripes read ~1/3 high)


def test_peem_panel_sections_and_sum_rule_window(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
    from tensorspec.gui.suites.peem_suite import PEEMSuite
    from tensorspec.gui.suites.xas_suite import XASSuite

    errors = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: errors.append(a[2])))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: errors.append(a[2])))
    suite = PEEMSuite()
    p, t = suite.panel, suite.panel.map_tools
    assert [p.view_tabs.tabText(i) for i in range(p.view_tabs.count())] == ["Frame", "Map", "Analysis"]
    assert not hasattr(p, "bg_method") and not hasattr(p, "sum_nh")  # moved to the Sum Rule window
    p._dataset_name = _images(tmp_path, p.service)
    p._refresh_meta()
    t._asymmetry()
    assert t._shown() == ("pcd", "data") and t.viewer.primary_view().combo_cmap.currentText() == "bwr"
    t._on_line({"a": (32, 4), "b": (32, 60)})
    t._add_profile()
    t._histogram()
    assert len(p.service.list_profiles("img")) == 1 and t.plot_title.text().startswith("histogram")
    t._on_roi({"kind": "rect", "x0": 4, "x1": 60, "y0": 4, "y1": 60})
    t._roi_stats()
    assert "f_up 0.5" in t.status.text() or "f_up 0.4" in t.status.text()
    t._push()  # -> Analysis / Domains
    assert t.analysis_view.currentWidget() is t.domain_view and t.domain_view.items.count() == 1
    t.domain_view._run(all_items=True)
    assert t.domain_view.table.rowCount() == 1 and "width up / down" in t.domain_view.caption.text()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "m.tif"), "")))
    t._select("pcd")
    t._export()
    assert (tmp_path / "m.tif").exists() and (tmp_path / "m.png").exists()

    p._dataset_name = _scan(tmp_path, p.service)
    p._refresh_meta()
    assert t.viewer.tensor_data is None  # no map yet for the new dataset
    t._send()
    gc.collect()
    windows = [w for w in QApplication.allWidgets() if isinstance(w, XASSuite)]
    assert len(windows) == 1 and windows[0].tabs.currentWidget() is windows[0].peem_panel
    sp = windows[0].peem_panel
    assert sp._dataset_name == "scan_sumrule"
    sp._set_rect(2, 14, 2, 14)
    assert sp.spec_ax.get_xlabel() == "Energy"
    sp.bg_node.setCurrentIndex(sp.bg_node.findData("processed/CP"))
    sp.bg_e0.setValue(570)
    sp.bg_e1.setValue(573)
    sp._bg_preview()
    for spin, v in ((sp.sum_l3_lo, 574), (sp.sum_l3_hi, 582), (sp.sum_l2_lo, 582), (sp.sum_l2_hi, 592),
                    (sp.sum_r_lo, 570), (sp.sum_r_hi, 595), (sp.sum_nh, 3)):
        spin.setValue(v)
    sp._sumrule_preview()
    assert sp.sum_results.text().startswith("p=") and not errors, errors
    t._send()  # a second push reuses the open window
    assert len([w for w in QApplication.allWidgets() if isinstance(w, XASSuite)]) == 1
    windows[0].close()
    suite.close()


def test_mismatched_sweeps_are_regrouped_by_energy(tmp_path):
    """CP swept in 1 eV steps, CM in 1.25 eV steps over the same range (sweeps of different length):
    Stack Pairs pairs them in order (energies apart); the hand-over and the spectra regroup by each
    frame's own energy, so CP and CM points match."""
    s = PeemService()
    shape = lambda e: 1 + 0.8 * np.exp(-0.5 * ((e - 582.0) / 4.0) ** 2)
    ecp, ecm = np.arange(570.0, 595.01, 1.0), np.arange(570.0, 595.01, 1.25)
    a = synthetic.write_ue49_scan(tmp_path, "Mono_S_0001", np.stack([np.full((8, 8), 1000 * shape(e) * 1.02) for e in ecp]),
                                  [f"Cr_pos_{i:04d}" for i in range(len(ecp))], energy=ecp)
    b = synthetic.write_ue49_scan(tmp_path, "Mono_S_0002", np.stack([np.full((8, 8), 1000 * shape(e) * 0.98) for e in ecm]),
                                  [f"Cr_neg_{i:04d}" for i in range(len(ecm))], energy=ecm)
    s.load_beamline([a, b], "scan_mix", tpx3=False)
    s.pair("scan_mix", "CP_CM")
    sr = s.to_sum_rule("scan_mix")
    assert "different photon energies" in sr["note"]
    sp = s.roi_spectra(sr["name"])
    xas, xmcd = np.array(sp["xas"]), np.array(sp["dichroism"])
    r = xmcd / xas  # (CM - CP) / ((CP + CM) / 2) = -0.04 for CM 2 % below CP
    assert np.mean(r) == pytest.approx(-0.04, abs=0.005)
    assert np.abs(r + 0.04).max() < 0.025  # bins of 1.25 eV on a 4 eV-wide edge; pairing in order is up to 5 eV off


def test_one_dichroism_map_button_with_and_without_stack_pairs(qapp, tmp_path, monkeypatch):
    """Section 4: without Stack Pairs the map is made from all frames (make_pcd_map, every frame
    registered) and handed to the tools; after Stack Pairs from the corrected pairs, with an energy window."""
    from PySide6.QtWidgets import QMessageBox
    from tensorspec.gui.components.peem_panel import PeemPanel

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: errors.append(a[2])))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: errors.append(a[2])))
    p = PeemPanel()
    t = p.map_tools
    mz = synthetic.stripe_domains(64, 64, period_px=16)
    cp, cm = synthetic.helicity_stacks(mz, a0=0.05, n_cp=3, n_cm=3, noise=1.0)
    pos = synthetic.write_ue49_scan(tmp_path, "X_0002", cp, [f"Cr_pos_000{i}" for i in range(3)])
    neg = synthetic.write_ue49_scan(tmp_path, "X_0003", cm, [f"Cr_neg_000{i}" for i in range(3)])
    p.service.load_beamline([pos, neg], "img", tpx3=False)  # no Stack Pairs
    p._dataset_name = "img"
    p._refresh_meta()
    assert t.btn_asymmetry.text() == "Make XMCD map (CP / CM)"
    t.e_lo.setText("570")
    t._asymmetry()
    assert errors and "Stack Pairs" in errors[-1]  # an energy window needs the pairs
    t.e_lo.clear()
    t._asymmetry()
    assert len(errors) == 1, errors
    assert t._shown() == ("pcd", "data") and "all frames" in t.status.text()
    pcd = global_workspace.pull_analysis_data("img", "pcd")
    assert "difference" in pcd.data_vars and pcd.attrs["formula"] == "(CM-CP)/(CM+CP)"
    assert p.view_tabs.currentWidget() is t.map_view
    p.service.pair("img", "CP_CM")
    p._refresh_meta()
    t._asymmetry()
    assert "3 pairs" in t.status.text() and t._shown() == ("pcd", "data")

def test_drift_uses_the_pairs_after_stack_pairs(qapp, tmp_path, monkeypatch):
    """After Stack Pairs, Drift works on the pairs (Source: Processed), so Separate still works;
    if Drift ran on Raw anyway, Separate says how to recover."""
    from PySide6.QtWidgets import QMessageBox
    from tensorspec.gui.components.peem_panel import PeemPanel

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: errors.append(a[2])))
    p = PeemPanel()
    p._dataset_name = _images(tmp_path, p.service)
    p._refresh_meta()
    assert p.drift_source.currentData() == "processed"
    p._apply_drift()
    p._separate()
    assert not errors and "Separated" in p.status_label.text()
    p.drift_source.setCurrentIndex(p.drift_source.findData("raw"))
    p._apply_drift()
    p._separate()
    assert errors and "Stack Pairs again" in errors[-1]


def test_draw_buttons_bring_up_the_map_view(qapp, tmp_path):
    """Draw line / rectangle work on the Map view: the button switches to it from the Frame tab."""
    from tensorspec.gui.components.peem_panel import PeemPanel

    p = PeemPanel()
    p._dataset_name = _images(tmp_path, p.service)
    p._refresh_meta()
    p.map_tools._asymmetry()
    p.view_tabs.setCurrentIndex(0)  # Frame
    p.map_tools._start_tool("line", p.map_tools._on_line, "orange")
    assert p.view_tabs.currentWidget() is p.map_tools.map_view
    assert p.map_tools.viewer.primary_view().tool_active and "click the start" in p.map_tools.status.text()
    p.map_tools.viewer.primary_view()._finish_tool({"kind": "line", "a": (10, 5), "b": (10, 50)})
    assert p.map_tools.line_edit.text() == "5, 10, 50, 10" and "Line (5, 10)" in p.map_tools.status.text()


def test_map_view_lists_only_useful_layers(qapp, tmp_path):
    """No sums, no copies of the stacked means under the map, no working arrays, no 'Intensity'."""
    from tensorspec.gui.components.peem_panel import PeemPanel

    p = PeemPanel()
    p._dataset_name = _images(tmp_path, p.service)
    p._refresh_meta()
    assert p.map_tools.btn_asymmetry.text() == "Make XMCD map (CP / CM)"
    p.map_tools._asymmetry()
    p.map_tools._push()  # a pushed map is not a Map layer
    v = p.map_tools.viewer.primary_view()
    assert [v.combo_layer.itemText(i) for i in range(v.combo_layer.count())] == [
        "stack · cp+cm", "stack · mean_cp", "stack · mean_cm", "pcd", "pcd · difference"]


def test_push_raw_or_as_shown_with_roi_and_background(qapp, tmp_path, monkeypatch):
    """Section 6 -> Analysis / Domains: both push modes, the ROI (auto outline, saved and reloaded) and the
    background region (only while ticked) go along; ROI statistics split at the background mean."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    from tensorspec.gui.components.peem_panel import PeemPanel

    errors = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: errors.append(a[2])))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: errors.append(a[2])))
    p = PeemPanel()
    t, s = p.map_tools, p.service
    p._dataset_name = _images(tmp_path, s)
    p._refresh_meta()
    t._asymmetry()
    t._on_roi({"kind": "rect", "x0": 4, "x1": 59, "y0": 4, "y1": 59})
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "r.json"), "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(tmp_path / "r.json"), "")))
    t._save_outline()
    t._set_roi(None)
    t._load_outline()
    assert s.get_rois("img")["roi"]["x1"] == 59
    t.use_bg.setChecked(True)
    t._roi_stats()
    assert errors and "none is drawn" in errors[-1]
    t._on_bg({"kind": "rect", "x0": 0, "x1": 3, "y0": 0, "y1": 63})
    t._roi_stats()
    assert "zero = background mean" in t.status.text()
    t.push_mode.setCurrentIndex(1)  # as shown
    t._push()
    t.use_bg.setChecked(False)
    t.push_mode.setCurrentIndex(0)
    t._push()
    ins = s.domain_inputs()
    assert [i["mode"] for i in ins] == ["shown", "raw"]
    dv = t.domain_view
    dv.background.setCurrentIndex(dv.background.findData("region"))
    dv.items.clearSelection()
    dv.items.item(0).setSelected(True)
    dv._run(all_items=False)  # shown + its background region
    assert s.domain_result("img", ins[0]["label"])["split"] == "zero"
    dv.items.clearSelection()
    dv.items.item(1).setSelected(True)
    n = len(errors)
    dv._run(all_items=False)  # raw, pushed without a background region
    assert len(errors) == n + 1 and "background region" in errors[-1]
    dv.step.setCurrentIndex(dv.step.findData("width_map"))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "d.pdf"), "")))
    dv.items.clearSelection()
    dv.items.item(0).setSelected(True)
    dv._export_pdf()
    assert (tmp_path / "d.pdf").stat().st_size > 1000


def test_domains_tab_hides_the_left_column_and_shows_the_pushed_look(qapp, tmp_path):
    from tensorspec.gui.components.peem_panel import PeemPanel

    p = PeemPanel()
    t = p.map_tools
    p._dataset_name = _images(tmp_path, p.service)
    p._refresh_meta()
    t._asymmetry()
    t.viewer.primary_view().chk_invert.setChecked(True)
    t._push()
    assert p.view_tabs.currentWidget() is t.analysis_view and not p._controls_scroll.isVisibleTo(p)
    look = p.service.domain_look("img", "domains_1")
    assert look["cmap"] == "bwr_r" and look["vmin"] < 0 < look["vmax"]
    t.analysis_view.setCurrentWidget(t.plot_view)
    assert p._controls_scroll.isVisibleTo(p)
