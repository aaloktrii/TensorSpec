"""Gate C: KKRWrapper.run_simulation Fermi-Dirac cutoff (temperature_K / apply_fermi).

No real kkrscf/kkrspec run (FakeLauncher fakes those, reused from
test_workflow.py); ase2sprkkr + pymatgen ARE used for real to build the
seed .pot (via the shared ``cu_pot`` fixture).
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ASE2SPRKKR = importlib.util.find_spec("ase2sprkkr") is not None
PYMATGEN = importlib.util.find_spec("pymatgen") is not None

pytestmark = pytest.mark.skipif(
    not (ASE2SPRKKR and PYMATGEN), reason="ase2sprkkr and/or pymatgen not installed"
)

from tensorspec.core.dft.sprkkr.params import ArpesParams
from tensorspec.core.arpes.one_step.kkr_wrapper import KKRWrapper
from tests.sprkkr.test_workflow import FakeLauncher, _touch_bin, cu_pot, make_fake_spc  # noqa: F401


def test_fermi_cutoff_kills_above_ef_keeps_below(tmp_path, cu_pot):
    bin_dir = tmp_path / "bin"
    _touch_bin(bin_dir, "kkrspec9.7")

    # Fine energy grid, symmetric around EF, so +/-0.3 eV falls cleanly
    # inside "far above" / "far below" bands.
    fake_params = ArpesParams(
        e_min_eV=-1.0, e_max_eV=1.0, ne=9, nt=3, np_=1, dataset="arpes",
    )
    launcher = FakeLauncher(bin_dir=bin_dir, spc_by_dataset={"arpes": fake_params})

    base_kwargs = {
        "pot_path": str(cu_pot),
        "k_bounds": {"X": [-5.0, 5.0, 3], "Y": [0.0, 0.0, 1]},
        "e_range": [-1.0, 1.0, 9],
        "launcher": launcher,
        "nproc": 1,
        "mode": "mpi",
    }

    wrapper = KKRWrapper()

    out_cold = wrapper.run_simulation(
        {}, dict(base_kwargs, workdir=str(tmp_path / "run_cold"), temperature_K=10.0)
    )
    out_raw = wrapper.run_simulation(
        {}, dict(base_kwargs, workdir=str(tmp_path / "run_raw"), apply_fermi=False)
    )

    energy = out_cold["energy"]  # eV, rel. EF
    above = energy > 0.3
    below = energy < -0.3
    assert above.any() and below.any()

    # intensity_broadened layout: (theta, phi, energy)
    cold = out_cold["intensity_broadened"]
    raw = out_raw["intensity_broadened"]

    assert np.allclose(cold[..., above], 0.0, atol=1e-6)
    assert np.allclose(cold[..., below], raw[..., below])

    # tensor.value carries the same cutoff (energy-first convention) + metadata.
    tensor = out_cold["tensor"]
    assert tensor.metadata["fermi_cutoff_K"] == 10.0
    assert np.allclose(tensor.value[above, ...], 0.0, atol=1e-6)

    # Raw dataset (untouched) must NOT be cut.
    assert not np.allclose(out_cold["dataset"]["I_tot"].values[above, ...], 0.0, atol=1e-6)
