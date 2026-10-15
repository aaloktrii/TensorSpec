"""PEEM panel map path: make_pcd / make_pld on PeemService (no XPEEM tab)."""
from __future__ import annotations

import numpy as np

from tensorspec.core.data_models import TensorData
from tensorspec.core.peem import synthetic
from tensorspec.core.workspace import global_workspace
from tensorspec.gui.services.peem_service import PeemService


def _push_helicity(name: str) -> str:
    if global_workspace.get(name) is not None:
        global_workspace.remove(name)
    mz = synthetic.stripe_domains(48, 48, period_px=12)
    cp, cm = synthetic.helicity_stacks(mz, a0=0.08, n_cp=3, n_cm=3, noise=0.0)
    frames = np.concatenate([cp, cm], axis=0).astype(np.float32)
    pol = ["CP"] * len(cp) + ["CM"] * len(cm)
    td = TensorData(
        value=frames,
        axes=[np.arange(frames.shape[0], dtype=float), np.arange(48.0), np.arange(48.0)],
        labels=["frame", "y", "x"],
        units=["", "px", "px"],
        data_type="Experimental PEEM",
        metadata={"pol": pol, "default_convention": "ue49", "scans": [name]},
    )
    global_workspace.push_spectroscopy_data(name, td)
    return name


def test_make_pcd_from_raw_pol_tags():
    name = _push_helicity("t_pcd_raw")
    svc = PeemService()
    out = svc.make_pcd_map(name, source="raw", register=False)
    assert out["kind"] == "pcd" and out["display_node"] == "analysis/pcd"
    assert "pcd" in svc.list_analysis_maps(name)
    td = svc.get_display_tensor(name, "analysis/pcd")
    assert td.value.ndim == 2 and td.value.shape == (48, 48)
    assert np.isfinite(td.value).any()
    global_workspace.remove(name)


def test_make_pcd_after_separate():
    name = _push_helicity("t_pcd_sep")
    svc = PeemService()
    svc.pair(name, "CP_CM")
    svc.separate(name)
    out = svc.make_pcd_map(name, source="auto", register=False)
    assert out["source"] == "separated"
    td = svc.get_display_tensor(name, "analysis/pcd")
    assert td.value.shape == (48, 48)
    assert np.nanstd(td.value) > 1e-4  # domain contrast present (mean may ~0)
    global_workspace.remove(name)
