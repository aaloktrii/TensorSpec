"""Batch planning and processing on a synthetic two-sample beamtime."""
import csv

import numpy as np
import pytest

from tensorspec.core.peem import synthetic
from tensorspec.core.peem.ue49 import batch


def _frames(n, val=100.0, shape=(32, 32)):
    return np.full((n,) + shape, val)


@pytest.fixture
def beamtime(tmp_path):
    root = tmp_path / "June"
    d1, d2 = root / "20250601", root / "20250602"
    mz = synthetic.stripe_domains(32, 32, period_px=8)
    cp, cm = synthetic.helicity_stacks(mz, a0=0.05, n_cp=2, n_cm=4)
    # sample A (S1): two XMCD triplets in one block, then a pulse, one more triplet
    n = 10
    for _ in range(3):
        synthetic.write_ue49_scan(d1, f"X_{n:04d}", cp, ["Cr_pos_0000", "Cr_pos_0001"]); n += 1
        synthetic.write_ue49_scan(d1, f"X_{n:04d}", cm, [f"Cr_neg_000{i}" for i in range(4)]); n += 1
        synthetic.write_ue49_scan(d1, f"X_{n:04d}", cp, ["Cr_pos_0000", "Cr_pos_0001"]); n += 1
    e = np.linspace(570, 595, 12)
    synthetic.write_ue49_scan(d1, "Mono_S_0020", _frames(12), [f"Img_{i:04d}" for i in range(12)], energy=e, pol_table=0)
    synthetic.write_ue49_scan(d1, "Mono_S_0021", _frames(12), [f"Img_{i:04d}" for i in range(12)], energy=e, pol_table=1)
    synthetic.write_ue49_scan(d1, "Mono_S_0022", _frames(12), [f"Img_{i:04d}" for i in range(12)], energy=e, pol_table=1)
    # sample B (S2) shares day 2 and restarts numbering at 1
    synthetic.write_ue49_scan(d2, "X_0001", cp, ["Fe_pos_0000", "Fe_pos_0001"])
    synthetic.write_ue49_scan(d2, "X_0002", cm, [f"Fe_neg_000{i}" for i in range(4)])
    synthetic.write_ue49_scan(d2, "X_0003", cp, ["Fe_pos_0000", "Fe_pos_0001"])
    with open(root / "logbook_sessions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "sample", "folders", "scan_range", "edge", "xmcd_energy_eV", "xld_e_a_eV", "xld_e_b_eV"])
        w.writerow(["S1", "CrSBr", "20250601", "10-30", "Cr L2;3", "576.3", "584.2", "585.3"])
        w.writerow(["S2", "FGT", "20250602", "0-9", "Fe L3", "707.7", "", ""])
    with open(root / "logbook_conditions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "kind", "first", "last", "T_K_log", "ccs_mA", "pulse_before", "rotation_deg",
                    "energy_eV", "notes", "page", "transcription_uncertain"])
        w.writerow(["S1", "X", 10, 15, "100", "0", "", "0", "576.6", "", "1", ""])
        w.writerow(["S1", "X", 16, 18, "100", "-40", "+2000 mA pulse", "0", "576.6", "", "1", ""])
        w.writerow(["S1", "X", 19, 21, "100", "0", "", "0", "576.6", "not measured", "1", ""])
    return root


def test_plan_units_clusters_sessions(beamtime):
    sessions = batch.read_sessions(beamtime / "logbook_sessions.csv")
    conds = batch.read_conditions(beamtime / "logbook_conditions.csv")
    p = batch.plan(beamtime, sessions, conds)
    assert len(p["catalog"]) == 15  # S1: 9 X + 3 Mono; S2: 3 X (numbers restart, no clash)
    ids = {c.cid: c for c in p["clusters"]}
    assert [u.uid for u in ids["S1_X_0010-0015"].units] == ["X_0010-0012", "X_0013-0015"]
    assert ids["S1_X_0016-0018"].condition.pulse_before == "+2000 mA pulse"
    assert ids["S2_X_0001-0003_auto"].units[0].kind == "XMCD"  # not in logbook -> auto
    kinds = sorted(j.kind for j in p["spectra"])
    assert kinds.count("XLD_pair") == 1 and kinds.count("merge") == 1 and kinds.count("XAS") == 3
    issues = batch.validate(p, conds)
    assert any(i["issue"] == "in logbook, not on disk" and i["scan"] == "X_0019" for i in issues)


def test_run_writes_outputs(beamtime):
    st = batch.Settings(dead_map=None, norm_image=None)
    res = batch.run(beamtime, st, preview=None, log=lambda *_: None)
    status = {r["id"]: r["status"] for r in res["rows"]}
    assert status["S1_X_0010-0015"] == "ok"
    out = beamtime / "20250601" / batch.OUT_DIRNAME / "clusters" / "S1_X_0010-0015"
    assert (out / "units" / "X_0010-0012" / "maps.nc").exists()
    assert (out / "merged" / "pcd.tif").exists()
    import json
    meta = json.loads((out / "cluster.json").read_text())
    assert meta["repeatability"]["mean_corr"] > 0.9  # identical synthetic repeats
    assert (beamtime / "20250602" / batch.OUT_DIRNAME / "index.csv").exists()
    assert (beamtime / "tensorspec_xpeem_logbook_check.csv").exists()
    # second run skips existing outputs
    again = batch.run(beamtime, st, preview=None, log=lambda *_: None)
    assert {r["status"] for r in again["rows"]} == {"ok (existing)"}
