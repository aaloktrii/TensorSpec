import json
import os
import subprocess
import sys

from pymatgen.core import Lattice, Structure

from tensorspec.core.dft.sync_after_relax import sync_after_relax

FIXTURE_CRYSTAL = """
ATOMIC_POSITIONS (crystal)
C  0.333333  0.666667  0.410000
C  0.666667  0.333333  0.410000
End final coordinates
"""


def _hex_carbon_template():
    lat = Lattice.hexagonal(2.5, 23.4)
    return Structure(
        lat,
        ["C", "C"],
        [[1 / 3, 2 / 3, 0.4], [2 / 3, 1 / 3, 0.4]],
    )


def _write_sidecars(out_dir, *, ecutwfc=40, ecutrho=160, vdw=True):
    template = _hex_carbon_template()
    template.to(filename=os.path.join(out_dir, "structure_template.cif"), fmt="cif")
    meta = {
        "ecutwfc": ecutwfc,
        "ecutrho": ecutrho,
        "kmesh": [4, 4, 1],
        "nbnd": 12,
        "use_soc": False,
        "use_gpu": False,
        "vdw_dft_d3": vdw,
        "mlwf": False,
        "calculation": "relax",
    }
    with open(os.path.join(out_dir, "tensorspec_relax_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with open(os.path.join(out_dir, "relax.out"), "w", encoding="utf-8") as f:
        f.write(FIXTURE_CRYSTAL)
    return template


def _stub_pseudo_dir(tmp_path):
    pseudo = tmp_path / "pseudo_src"
    pseudo.mkdir()
    (pseudo / "C.pbe.z_4.upf").write_text("stub")
    return str(pseudo)


def test_sync_after_relax_rewrites_inputs(tmp_path, monkeypatch):
    template = _write_sidecars(tmp_path, ecutrho=160)
    pseudo_dir = _stub_pseudo_dir(tmp_path)

    sync_after_relax(
        str(tmp_path),
        template_cif="structure_template.cif",
        pseudo_dir=pseudo_dir,
    )

    relaxed_path = tmp_path / "relaxed_structure.cif"
    assert relaxed_path.is_file()
    relaxed = Structure.from_file(relaxed_path)
    assert abs(relaxed[0].frac_coords[2] - 0.41) < 1e-5
    assert relaxed.lattice.a == template.lattice.a

    scf_text = (tmp_path / "scf.in").read_text()
    assert "0.410000" in scf_text
    assert "0.400000" not in scf_text
    assert "ecutrho = 160" in scf_text
    assert "vdw_corr = 'dft-d3'" in scf_text

    nscf_text = (tmp_path / "nscf.in").read_text()
    assert "0.410000" in nscf_text
    assert "nbnd = 12" in nscf_text

    win_text = (tmp_path / "wannier90.win").read_text()
    assert "0.410000" in win_text

    assert (tmp_path / "pw2wan.in").is_file()


def test_sync_after_relax_cli(tmp_path, monkeypatch):
    _write_sidecars(tmp_path)
    pseudo_dir = _stub_pseudo_dir(tmp_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".")
    env["TENSORSPEC_PSEUDO_DIR"] = pseudo_dir

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorspec.core.dft.sync_after_relax",
            "--out-dir",
            str(tmp_path),
            "--template-cif",
            "structure_template.cif",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "relaxed_structure.cif").is_file()
    assert "0.410000" in (tmp_path / "scf.in").read_text()


def test_sync_after_relax_missing_relax_out(tmp_path):
    _write_sidecars(tmp_path)
    (tmp_path / "relax.out").unlink()
    pseudo_dir = _stub_pseudo_dir(tmp_path)

    try:
        sync_after_relax(
            str(tmp_path),
            template_cif="structure_template.cif",
            pseudo_dir=pseudo_dir,
        )
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised
