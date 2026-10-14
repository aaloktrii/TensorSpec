"""The PEEM back end on its own: every public service call is in the documented contract, and the
whole workflow runs through the service without Qt (as a script or another front end would)."""
import inspect
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from tensorspec.gui.services import peem_steps
from tensorspec.gui.services.peem_service import PeemService


def test_every_public_service_call_is_in_the_contract():
    doc = peem_steps.__doc__
    public = [n for n, f in inspect.getmembers(PeemService) if not n.startswith("_") and callable(f)]
    missing = [n for n in public if not re.search(rf"\b{n}\b", doc)]
    assert not missing, f"add to the contract in peem_steps/__init__.py: {missing}"


WORKFLOW = textwrap.dedent('''
    import sys
    import numpy as np
    from tensorspec.core.peem import synthetic
    from tensorspec.gui.services.peem_service import PeemService

    tmp = sys.argv[1]
    mz = synthetic.stripe_domains(64, 64, period_px=16)
    cp, cm = synthetic.helicity_stacks(mz, a0=0.05, n_cp=3, n_cm=3, noise=2.0, shift_cm=(0.0, 1.0))
    pos = synthetic.write_ue49_scan(tmp, "X_0002", cp, [f"Cr_pos_000{i}" for i in range(3)])
    neg = synthetic.write_ue49_scan(tmp, "X_0003", cm, [f"Cr_neg_000{i}" for i in range(3)])
    s = PeemService()
    n = s.load_beamline([pos, neg], "w", tpx3=False)["name"]
    s.set_excluded(n, [5], True)                                   # 2. clean
    s.set_roi(n, "drift_box", {"kind": "rect", "x0": 8, "x1": 56, "y0": 8, "y1": 56})
    s.drift_correct(n, s.get_rois(n)["drift_box"], method="marquee")  # 3. drift
    s.stack(n)                                                     # 4. stack
    assert [v for _, v in s.stack_outputs(n)][:2] == ["mean_cp", "mean_cm"]
    s.asymmetry(n)                                                 # 5. asymmetry
    assert s.asymmetry_outputs(n)[0] == ("asymmetry", "data")
    assert s.display_hint(n, "pcd")["cmap"] == "bwr"               # 6. view
    s.add_profile(n, (32, 4), (32, 60), halfwidth=2, kind="pcd")   # 7. profiles
    assert len(s.list_profiles(n)) == 1
    s.set_roi(n, "roi", {"kind": "rect", "x0": 4, "x1": 60, "y0": 4, "y1": 60})
    st = s.roi_statistics(n, "pcd", roi=s.get_rois(n)["roi"], threshold=0.0)   # 8. maps
    assert set(st["fractions"]) >= {"f_up", "f_down"}
    dm = s.compute_domain_map(n, threshold=0.2, use_sda=False)
    h = s.to_peem_stack(n, "xmcd")                                 # 9. to the PEEM tab
    assert h["separated"] and s.get_meta(h["name"])["name"] == h["name"]
    print("QT_LOADED", any(m.startswith("PySide6") for m in sys.modules), "f_up", round(dm["f_up"], 2))
''')


def test_whole_workflow_runs_without_qt(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "workflow.py"
    script.write_text(WORKFLOW)
    res = subprocess.run([sys.executable, str(script), str(tmp_path)], capture_output=True, text=True,
                         cwd=root, env={"PYTHONPATH": str(root), "PATH": "/usr/bin:/bin"}, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    line = next(ln for ln in res.stdout.splitlines() if ln.startswith("QT_LOADED"))
    assert line.split()[1] == "False", "the PEEM back end imported Qt"
    assert 0.3 < float(line.split()[3]) < 0.7  # stripe domains: about half up
