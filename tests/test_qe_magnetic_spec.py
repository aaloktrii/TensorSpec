"""MagneticSpec + QE input emission for collinear / AFM / SOC."""
from __future__ import annotations

from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from tensorspec.core.dft.magnetic_spec import MagneticSpec, afm_split_spec, collinear_spec
from tensorspec.core.dft.qe_generator import QEInputGenerator


def _fe_dimer():
    lat = Lattice.cubic(4.0)
    return Structure(lat, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_system_flags_collinear_and_soc():
    off = MagneticSpec.off()
    assert off.system_flags() == ""
    col = MagneticSpec(
        mode="collinear",
        starting_magnetization={"Fe": 1.0},
        tot_magnetization=2.0,
    )
    flags = col.system_flags(["Fe"])
    assert "nspin = 2" in flags
    assert "starting_magnetization(1) = 1.0" in flags
    assert "tot_magnetization = 2.0" in flags
    soc = MagneticSpec(mode="soc", starting_magnetization={"Fe": (0, 0, 1)})
    sflags = soc.system_flags(["Fe"])
    assert "noncolin" in sflags and "lspinorb" in sflags
    assert soc.spinors_wannier


def test_afm_split_alternate():
    struct = _fe_dimer()
    mag = afm_split_spec(struct, ["Fe"])
    assert mag.mode == "afm"
    assert mag.site_labels == ["Fe_up", "Fe_dn"]
    assert mag.starting_magnetization["Fe_up"] == 1.0
    assert mag.starting_magnetization["Fe_dn"] == -1.0


def test_scf_emits_afm_types(tmp_path, monkeypatch):
    struct = _fe_dimer()
    gen = QEInputGenerator(struct)
    # Fake a scalar Fe UPF in app pseudo dir
    pseudo = tmp_path / "pseudo_src"
    pseudo.mkdir()
    upf = pseudo / "Fe.us.pbe.upf"
    upf.write_text("fake")
    gen.app_pseudo_dir = str(pseudo)

    mag = afm_split_spec(struct, ["Fe"])
    out = tmp_path / "run"
    path = gen.write_scf_input(str(out), magnetic=mag)
    text = Path(path).read_text()
    assert "nspin = 2" in text
    assert "nosym = .true." in text
    assert "tot_magnetization = 0.0" in text
    assert "mixing_beta = 0.3" in text
    assert "Fe_up" in text and "Fe_dn" in text
    assert "starting_magnetization(1)" in text
    assert "starting_magnetization(2)" in text
    assert "ntyp = 2" in text


def test_scf_soc_backcompat_use_soc(tmp_path):
    lat = Lattice.cubic(3.0)
    struct = Structure(lat, ["C"], [[0, 0, 0]])
    gen = QEInputGenerator(struct)
    pseudo = tmp_path / "pseudo_src"
    pseudo.mkdir()
    (pseudo / "C.fr.pbe.upf").write_text("fake")
    gen.app_pseudo_dir = str(pseudo)
    out = tmp_path / "run"
    path = gen.write_scf_input(str(out), use_soc=True)
    text = Path(path).read_text()
    assert "noncolin" in text and "lspinorb" in text


def test_wannier_spinors_only_for_soc(tmp_path):
    struct = _fe_dimer()
    gen = QEInputGenerator(struct)
    out = tmp_path / "run"
    out.mkdir()
    mag = collinear_spec(struct, {"Fe": 1.0})
    win = gen.write_wannier90_input(str(out), magnetic=mag)
    assert "spinors" not in Path(win).read_text()
    # SOC needs FR pseudo only for species copy on scf; win alone doesn't copy
    soc = MagneticSpec(mode="soc")
    win2 = gen.write_wannier90_input(str(out), magnetic=soc, num_wann=8)
    assert "spinors = true" in Path(win2).read_text()
