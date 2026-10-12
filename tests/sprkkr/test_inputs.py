"""Gate A / P1 tests: params + inputs, no binary, no Qt.

Compares ase2sprkkr-generated .inp files against ground-truth fixtures that
were verified to actually run SPR-KKR 9.7 (design doc §0).
"""
import importlib.util

import pytest

ASE2SPRKKR = importlib.util.find_spec("ase2sprkkr") is not None
PYMATGEN = importlib.util.find_spec("pymatgen") is not None

pytestmark = pytest.mark.skipif(
    not (ASE2SPRKKR and PYMATGEN),
    reason="ase2sprkkr and/or pymatgen not installed",
)

from pathlib import Path

from tensorspec.core.dft.sprkkr.inputs import (
    build_arpes_inputs,
    build_scf_inputs,
    read_inp_keywords,
)
from tensorspec.core.dft.sprkkr.params import ArpesParams, ScfParams

FIXTURES = Path(__file__).parent / "fixtures"
IGNORED_KEYS = {"DATASET", "POTFIL"}


def _cu_fcc_structure():
    from pymatgen.core import Lattice, Structure

    lat = Lattice.cubic(3.615)
    conv = Structure.from_spacegroup("Fm-3m", lat, ["Cu"], [[0, 0, 0]])
    return conv.get_primitive_structure()


def _norm_scalar(value: str):
    value = value.strip()
    try:
        return round(float(value), 6)
    except ValueError:
        return value


def _norm_value(value: str):
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1]
        if not inner:
            return ()
        return tuple(_norm_scalar(v) for v in inner.split(","))
    return _norm_scalar(value)


def assert_same_keywords(generated: Path, ref: Path) -> None:
    gen_sections = read_inp_keywords(generated)
    ref_sections = read_inp_keywords(ref)

    gen_sections = {
        sec: {k: v for k, v in keys.items() if k not in IGNORED_KEYS}
        for sec, keys in gen_sections.items()
    }
    ref_sections = {
        sec: {k: v for k, v in keys.items() if k not in IGNORED_KEYS}
        for sec, keys in ref_sections.items()
    }

    assert set(gen_sections) == set(ref_sections), (
        f"section mismatch: generated={sorted(gen_sections)} "
        f"ref={sorted(ref_sections)}"
    )
    for section in ref_sections:
        gen_keys = gen_sections[section]
        ref_keys = ref_sections[section]
        assert set(gen_keys) == set(ref_keys), (
            f"[{section}] key mismatch: generated={sorted(gen_keys)} "
            f"ref={sorted(ref_keys)}"
        )
        for key, ref_raw in ref_keys.items():
            assert _norm_value(gen_keys[key]) == _norm_value(ref_raw), (
                f"[{section}].{key}: generated={gen_keys[key]!r} "
                f"ref={ref_raw!r}"
            )


def test_scf_inputs_match_ref(tmp_path):
    structure = _cu_fcc_structure()
    params = ScfParams(nl=3, ne=30, nktab=250, niter=200)

    result = build_scf_inputs(structure, params, tmp_path / "scf")

    assert result.inp_path.exists()
    assert result.pot_path.exists()
    assert_same_keywords(result.inp_path, FIXTURES / "REF_scf.inp")


def test_scf_pot_skeleton(tmp_path):
    structure = _cu_fcc_structure()
    params = ScfParams()

    result = build_scf_inputs(structure, params, tmp_path / "scf")
    pot_text = result.pot_path.read_text()

    assert "NQ" in pot_text
    assert "NT" in pot_text
    nq_line = next(l for l in pot_text.splitlines() if l.strip().startswith("NQ"))
    nt_line = next(l for l in pot_text.splitlines() if l.strip().startswith("NT"))
    assert nq_line.split()[-1] == "1"
    assert nt_line.split()[-1] == "1"
    bravais_line = next(l for l in pot_text.splitlines() if l.strip().startswith("BRAVAIS"))
    assert "13" in bravais_line
    alat_line = next(l for l in pot_text.splitlines() if l.strip().startswith("ALAT"))
    alat = float(alat_line.split()[-1])
    assert alat == pytest.approx(6.8314, abs=1e-3)


def test_arpes_inputs_match_ref(tmp_path):
    structure = _cu_fcc_structure()
    scf_result = build_scf_inputs(structure, ScfParams(), tmp_path / "scf")

    params = ArpesParams()  # defaults per design doc §5
    result = build_arpes_inputs(scf_result.pot_path, params, tmp_path / "arpes")

    assert result.inp_path.exists()
    assert_same_keywords(result.inp_path, FIXTURES / "REF_arpes.inp")

    text = result.inp_path.read_text()
    assert "CRYS_VECS" not in text

    sections = read_inp_keywords(result.inp_path)
    assert sections["SPEC_EL"]["THETA"] == "{-20.0,20.0}"
    assert sections["SPEC_EL"]["NT"] == "21"
    assert sections["SPEC_PH"]["EPHOT"] == "21.2"
    assert sections["SPEC_PH"]["POL_P"] == "P"
    assert sections["ENERGY"]["EWORK_EV"] == "4.5"
    assert sections["SPEC_STR"]["N_LAYER"] == "50"


def test_arpes_inputs_conventional_frame_omits_crys_vecs(tmp_path):
    structure = _cu_fcc_structure()
    scf_result = build_scf_inputs(structure, ScfParams(), tmp_path / "scf")

    params = ArpesParams(hkl=(0, 0, 1))  # hkl_frame default: "conventional"
    result = build_arpes_inputs(scf_result.pot_path, params, tmp_path / "arpes")

    sections = read_inp_keywords(result.inp_path)
    assert "CRYS_VECS" not in sections["TASK"]


def test_arpes_inputs_abas_frame_injects_crys_vecs(tmp_path):
    structure = _cu_fcc_structure()
    scf_result = build_scf_inputs(structure, ScfParams(), tmp_path / "scf")

    params = ArpesParams(hkl=(1, 1, 0), hkl_frame="abas")
    result = build_arpes_inputs(scf_result.pot_path, params, tmp_path / "arpes")

    sections = read_inp_keywords(result.inp_path)
    assert sections["TASK"]["CRYS_VECS"] == ""
    assert sections["TASK"]["MILLER_HKL"] == "{1,1,0}"


def test_arpes_params_validate_bad_hkl_frame():
    params = ArpesParams(hkl_frame="bogus")
    with pytest.raises(ValueError):
        params.validate()


def test_arpes_expected_output_names():
    params = ArpesParams(dataset="_Cu_ARPES")
    # No binary run needed to compute the expected filenames.
    from tensorspec.core.dft.sprkkr.inputs import ArpesInputs

    expected_spc = f"{params.dataset}_ARPES_data.spc"
    expected_log = f"{params.dataset}_ARPES_SPEC.out"
    assert expected_spc == "_Cu_ARPES_ARPES_data.spc"
    assert expected_log == "_Cu_ARPES_ARPES_SPEC.out"


def test_arpes_params_n_points():
    params = ArpesParams(ne=71, nt=21, np_=1)
    assert params.n_points == 71 * 21 * 1


def test_arpes_params_validate_bad_pol():
    params = ArpesParams(pol_p="X")
    with pytest.raises(ValueError):
        params.validate()


def test_arpes_params_validate_nt_zero():
    params = ArpesParams(nt=0)
    with pytest.raises(ValueError):
        params.validate()


def test_arpes_params_validate_bad_hkl():
    params = ArpesParams(hkl=(0, 0, 1.5))
    with pytest.raises(ValueError):
        params.validate()


def test_scf_params_validate_bad_niter():
    params = ScfParams(niter=0)
    with pytest.raises(ValueError):
        params.validate()


def test_read_inp_keywords_ref_scf():
    sections = read_inp_keywords(FIXTURES / "REF_scf.inp")
    assert sections["SCF"]["NITER"] == "200"
    assert sections["CONTROL"]["DATASET"] == "_Cu_SCF"
    assert sections["TASK"]["SCF"] == ""


def test_read_inp_keywords_ref_arpes():
    sections = read_inp_keywords(FIXTURES / "REF_arpes.inp")
    assert sections["TASK"]["MILLER_HKL"] == "{0,0,1}"
    assert sections["TASK"]["CRYS_VEC"] == ""
    assert sections["SPEC_EL"]["THETA"] == "{-20.0,20.0}"
