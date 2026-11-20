"""load_arpes_matlab on tiny synthetic files: the writer's labelled struct (v7), Sandy's
value/x/y/info/para struct (v7), and the same struct as MATLAB v7.3 writes it (HDF5, dimensions
reversed). No real beamline data needed."""
import numpy as np
import pytest
from scipy.io import savemat

from tensorspec.core.data_models import TensorData
from tensorspec.core.io.matlab_arpes import (
    arpes_metadata_number,
    load_arpes_matlab,
    save_arpes_matlab,
)

K = np.linspace(-0.2, 0.2, 5)
E = np.linspace(-0.3, 0.1, 7)
CUT = np.outer(np.arange(5.0), np.ones(7)) + np.arange(7.0) / 10.0   # value[k, E]


def test_reads_back_what_the_writer_wrote(tmp_path):
    td = TensorData(value=CUT.T.copy(), axes=[E, K], labels=["Energy", "kx (Slit)"],
                    units=["eV", "1/A"], data_type="Experimental ARPES", metadata={"hv": 82.0})
    path = tmp_path / "written.mat"
    save_arpes_matlab(td, str(path))
    back = load_arpes_matlab(path)
    assert back.labels == ["Energy", "kx_Slit"]
    assert back.units == ["eV", "1/A"]
    np.testing.assert_allclose(back.value, CUT.T)
    np.testing.assert_allclose(back.axes[1], K)
    assert back.metadata["info"]["hv"] == 82.0
    assert back.metadata["matlab_variable"] == "data"


def _maestro_struct():
    return {"value": CUT, "x": K.reshape(1, -1), "y": E.reshape(1, -1),
            "info": {"Temperature": {"Sample": 13.06}, "Beamline": {"Eresolution": 0.005}},
            "para": {"k_step": 856.0, "latt_const": 3.14}}


def test_maestro_struct_v7(tmp_path):
    path = tmp_path / "cut.mat"
    savemat(str(path), {"MAESTRO_cut_k_sp_cut": _maestro_struct(), "other_cut": _maestro_struct(),
                        "note": "x"})
    with pytest.raises(ValueError):
        load_arpes_matlab(path)           # two structs: var is required
    with pytest.raises(KeyError):
        load_arpes_matlab(path, "missing")
    back = load_arpes_matlab(path, "MAESTRO_cut_k_sp_cut")
    assert back.value.shape == (5, 7)
    assert back.labels == ["kx", "Energy"] and back.units == ["1/A", "eV"]
    np.testing.assert_allclose(back.axes[0], K)
    np.testing.assert_allclose(back.axes[1], E)
    assert arpes_metadata_number(back.metadata, "info", "Temperature", "Sample") == 13.06
    assert arpes_metadata_number(back.metadata, "info", "Beamline", "Eresolution") == 0.005
    assert arpes_metadata_number(back.metadata, "info", "Nope", default=-1) == -1
    # single struct in the file: var may be omitted
    savemat(str(path), {"only": _maestro_struct()})
    assert load_arpes_matlab(path).value.shape == (5, 7)


def test_angle_cut_is_labelled_as_slit_angle(tmp_path):
    struct = _maestro_struct()
    struct["x"] = np.linspace(-15, 15, 5).reshape(1, -1)
    struct.pop("para")
    path = tmp_path / "angle.mat"
    savemat(str(path), {"MAESTRO_cut": struct})
    back = load_arpes_matlab(path)
    assert back.labels[0] == "Slit Angle" and back.units[0] == "deg"


def test_maestro_struct_v73(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "cut73.mat"
    with h5py.File(path, "w", userblock_size=512) as f:
        g = f.create_group("scan_k_sp")
        g.attrs["MATLAB_class"] = np.bytes_(b"struct")
        # MATLAB stores value(k, E) column-major; h5py sees it as (E, k)
        d = g.create_dataset("value", data=CUT.T)
        d.attrs["MATLAB_class"] = np.bytes_(b"double")
        g.create_dataset("x", data=K.reshape(-1, 1)).attrs["MATLAB_class"] = np.bytes_(b"double")
        g.create_dataset("y", data=E.reshape(-1, 1)).attrs["MATLAB_class"] = np.bytes_(b"double")
        info = g.create_group("info")
        info.attrs["MATLAB_class"] = np.bytes_(b"struct")
        t = info.create_group("Temperature")
        t.create_dataset("Sample", data=np.array([[13.06]])).attrs["MATLAB_class"] = np.bytes_(b"double")
        name = info.create_dataset("Name", data=np.array([[ord(c)] for c in "MAESTRO"], dtype=np.uint16))
        name.attrs["MATLAB_class"] = np.bytes_(b"char")
    with open(path, "r+b") as fh:
        fh.write(b"MATLAB 7.3 MAT-file, synthetic test header")
    back = load_arpes_matlab(path)
    assert back.value.shape == (5, 7)
    np.testing.assert_allclose(back.value, CUT)
    assert back.labels == ["kx", "Energy"]
    assert back.metadata["info"]["Name"] == "MAESTRO"
    assert arpes_metadata_number(back.metadata, "info", "Temperature", "Sample") == 13.06
