import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

FIXTURES = Path(__file__).parent / "fixtures"

# Import outputs.py by file path: the sibling P1 worker owns
# tensorspec/core/dft/sprkkr/__init__.py and may not have landed yet.
_OUTPUTS_PATH = (
    Path(__file__).parent.parent.parent
    / "tensorspec"
    / "core"
    / "dft"
    / "sprkkr"
    / "outputs.py"
)


def _load_outputs_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tensorspec.core.dft.sprkkr.outputs", _OUTPUTS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


outputs = _load_outputs_module()
parse_spc = outputs.parse_spc
spc_to_tensor = outputs.spc_to_tensor
spc_to_datatree = outputs.spc_to_datatree
parse_scf_log = outputs.parse_scf_log
stitch_spc = outputs.stitch_spc


SPC_NP1 = FIXTURES / "_Cu_ARPES_ARPES_data.spc"
SPC_NP2 = FIXTURES / "_Cu_NP2_ARPES_data.spc"
SCF_TAIL = FIXTURES / "Cu_SCF.out.tail"


class TestParseSpcNP1:
    def test_shape(self):
        ds = parse_spc(SPC_NP1)
        assert ds["I_tot"].shape == (8, 9, 1)
        assert ds.sizes["energy"] == 8
        assert ds.sizes["theta"] == 9
        assert ds.sizes["phi"] == 1

    def test_ef(self):
        ds = parse_spc(SPC_NP1)
        assert ds.attrs["EF_Ry"] == pytest.approx(0.65979)
        assert ds.attrs["NE"] == 8
        assert ds.attrs["NT"] == 9
        assert ds.attrs["NP"] == 1

    def test_energy_ascending(self):
        ds = parse_spc(SPC_NP1)
        energy = ds["energy"].values
        assert np.all(np.diff(energy) > 0)
        assert energy[0] == pytest.approx(-6.0)
        assert energy[-1] == pytest.approx(1.0)

    def test_theta_range(self):
        ds = parse_spc(SPC_NP1)
        theta = ds["theta"].values
        assert theta[0] == pytest.approx(-20.0)
        assert theta[-1] == pytest.approx(20.0)
        assert np.all(np.diff(theta) > 0)

    def test_intensity_nonnegative(self):
        ds = parse_spc(SPC_NP1)
        assert np.all(ds["I_tot"].values >= 0)

    def test_kpar_sign_follows_theta(self):
        ds = parse_spc(SPC_NP1)
        theta = ds["theta"].values
        kpar_at_zero_energy_row = ds["k_par"].isel(energy=0, phi=0).values
        # k_par should carry the same sign pattern as theta (negative theta -> negative k_par)
        for th, kp in zip(theta, kpar_at_zero_energy_row):
            if th < 0:
                assert kp <= 0
            elif th > 0:
                assert kp >= 0
            else:
                assert kp == pytest.approx(0.0)


class TestParseSpcNP2:
    """NP=2 real-run fixture (kkrspec9.7, verified 2026-09-09).

    Fact: .spc still has 8 columns for NP>1 -- no phi column is ever
    written. Row order is energy (outer) -> phi (middle) -> theta (inner).
    Real phi angles are not recoverable from the file; we pass phi_range
    explicitly here (as the GUI layer would, from the ArpesParams that
    produced the run).
    """

    def test_shape_with_phi_range(self):
        ds = parse_spc(SPC_NP2, phi_range=(0.0, 20.0))
        assert ds["I_tot"].shape == (2, 3, 2)
        assert ds.attrs["NE"] == 2
        assert ds.attrs["NT"] == 3
        assert ds.attrs["NP"] == 2
        assert ds.attrs["phi_is_index"] is False
        assert ds["phi"].values[0] == pytest.approx(0.0)
        assert ds["phi"].values[1] == pytest.approx(20.0)

    def test_shape_without_phi_range_falls_back_to_index(self):
        ds = parse_spc(SPC_NP2)
        assert ds["I_tot"].shape == (2, 3, 2)
        assert ds.attrs["phi_is_index"] is True
        assert list(ds["phi"].values) == [0.0, 1.0]

    def test_energy_ascending_and_theta_range(self):
        ds = parse_spc(SPC_NP2, phi_range=(0.0, 20.0))
        energy = ds["energy"].values
        assert np.all(np.diff(energy) > 0)
        assert energy[0] == pytest.approx(-2.0)
        assert energy[1] == pytest.approx(-1.0)
        theta = ds["theta"].values
        assert list(theta) == pytest.approx([-10.0, 0.0, 10.0])

    def test_values_row_count_matches_ne_nt_np(self):
        ds = parse_spc(SPC_NP2, phi_range=(0.0, 20.0))
        assert ds["I_tot"].size == 2 * 3 * 2


class TestSpcToTensor:
    def test_np1_squeezes_phi(self):
        ds = parse_spc(SPC_NP1)
        tensor = spc_to_tensor(ds, meta={"dataset": "Cu_test"})
        assert tensor.ndim == 2
        assert tensor.labels == ["Energy", "Theta"]
        assert tensor.units == ["eV", "deg"]
        assert tensor.value.shape == (8, 9)
        assert tensor.data_type == "Simulated ARPES (SPR-KKR)"
        assert tensor.metadata["dataset"] == "Cu_test"
        assert tensor.metadata["EF_Ry"] == pytest.approx(0.65979)

    def test_np2_keeps_phi(self):
        ds = parse_spc(SPC_NP2, phi_range=(0.0, 20.0))
        tensor = spc_to_tensor(ds)
        assert tensor.ndim == 3
        assert tensor.labels == ["Energy", "Theta", "Phi"]
        assert tensor.value.shape == (2, 3, 2)


class TestSpcToDatatree:
    def test_build_from_tensor_and_processed(self):
        ds = parse_spc(SPC_NP1)
        tree = spc_to_datatree(ds, meta={"dataset": "Cu_test"})

        raw_ds = tree["raw"].to_dataset() if hasattr(tree["raw"], "to_dataset") else tree["raw"].ds
        assert "data" in raw_ds
        assert raw_ds["data"].shape == (8, 9)

        processed_ds = (
            tree["processed"].to_dataset()
            if hasattr(tree["processed"], "to_dataset")
            else tree["processed"].ds
        )
        assert "I_up" in processed_ds
        assert "I_dn" in processed_ds
        assert "pol" in processed_ds
        assert "k_par" in processed_ds

        history_ds = (
            tree["history"].to_dataset()
            if hasattr(tree["history"], "to_dataset")
            else tree["history"].ds
        )
        log = history_ds.attrs.get("log", [])
        assert any("SPR-KKR kkrspec ARPES parsed from" in line for line in log)


class TestParseScfLog:
    def test_converged_status(self):
        status = parse_scf_log(SCF_TAIL)
        assert status.converged is True
        assert status.iterations == 12
        assert status.ef_ry == pytest.approx(0.65979)
        assert status.etot_ry == pytest.approx(-3304.998, abs=1e-2)
        assert status.last_err == pytest.approx(0.948e-05, rel=1e-3)
        assert len(status.history) == 1
        assert status.history[0][0] == 12


class TestStitchSpc:
    def test_stitch_split_equals_full(self):
        full = parse_spc(SPC_NP1)

        first_half = full.isel(energy=slice(0, 4))
        second_half = full.isel(energy=slice(4, 8))

        stitched = stitch_spc([first_half, second_half])

        assert stitched.sizes["energy"] == 8
        xr.testing.assert_allclose(
            stitched["I_tot"].sortby("energy"), full["I_tot"].sortby("energy")
        )
        assert list(stitched["energy"].values) == list(full["energy"].values)

    def test_stitch_single_dataset_passthrough(self):
        full = parse_spc(SPC_NP1)
        stitched = stitch_spc([full])
        assert stitched is full
