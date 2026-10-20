import numpy as np
from scipy.io import loadmat

from tensorspec.core.data_models import TensorData
from tensorspec.core.io.matlab_arpes import arpes_matlab_payload, matlab_field_name, save_arpes_matlab


def test_field_name_keeps_the_tensor_label():
    used = {"value", "info"}
    assert matlab_field_name("Energy", used) == "Energy"
    assert matlab_field_name("kx (Slit)", used) == "kx_Slit"
    assert matlab_field_name("Θ (Slit)", used) == "Theta_Slit"


def test_payload_uses_labels_and_value():
    td = TensorData(
        value=np.zeros((2, 3, 4)),
        axes=[np.array([-1.0, 0.1]), np.linspace(-1, 1, 3), np.linspace(-0.5, 0.5, 4)],
        labels=["Energy", "kx (Slit)", "ky (Deflection)"],
        units=["eV", "1/A", "1/A"],
        data_type="Simulated ARPES",
        metadata={"note": "slit horizontal"},
    )
    payload = arpes_matlab_payload(td)
    assert payload["value"].shape == (2, 3, 4)
    assert payload["Energy"].shape == (1, 2)
    assert payload["kx_Slit"].shape == (1, 3)
    assert payload["ky_Deflection"].shape == (1, 4)
    assert "x" not in payload
    assert payload["info"]["note"] == "slit horizontal"


def test_roundtrip_mat(tmp_path):
    td = TensorData(
        value=np.arange(6, dtype=np.float32).reshape(2, 3),
        axes=[np.array([20.0, 40.0]), np.array([-1.0, 0.0, 1.0])],
        labels=["Photon Energy", "kx (Slit)"],
        units=["eV", "1/A"],
        data_type="ARPES",
        metadata={},
    )
    path = tmp_path / "cut.mat"
    save_arpes_matlab(td, str(path))
    data = loadmat(path, squeeze_me=False, struct_as_record=False)["data"][0, 0]
    assert data.value.shape == (2, 3)
    assert data.Photon_Energy.shape == (1, 2)
    assert data.kx_Slit.shape == (1, 3)
