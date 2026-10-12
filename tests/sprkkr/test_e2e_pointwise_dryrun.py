"""E2E dry-run tests for pointwise deflector cuts."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ASE2SPRKKR = importlib.util.find_spec("ase2sprkkr") is not None
PYMATGEN = importlib.util.find_spec("pymatgen") is not None

pytestmark = pytest.mark.skipif(
    not (ASE2SPRKKR and PYMATGEN), reason="ase2sprkkr and/or pymatgen not installed"
)

from tensorspec.core.dft.sprkkr.inputs import read_inp_keywords

FIXTURES = Path(__file__).parent / "fixtures"
VTE2_POT = FIXTURES / "VTe2_prim.pot"
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dry_run_pointwise_writes_n_inp(tmp_path):
    """Pointwise dry-run should write N .inp files for N slit angles."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sprkkr" / "sprkkr_e2e.py"),
        "--pot", str(VTE2_POT),
        "--hkl", "0", "1", "-1",
        "--iq-surf", "2",
        "--hv", "84",
        "--theta", "-15", "15", "5",
        "--deflector", "-10.3",
        "--slit", "0",
        "--pointwise",
        "--dry-run",
        "--out", str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0, f"stderr: {result.stderr}"

    arpes_dir = tmp_path / "arpes"
    inp_files = [f for f in arpes_dir.glob("**/*.inp") if "_p" in f.parent.name]
    assert len(inp_files) == 5, f"Expected 5 .inp files, got {len(inp_files)}: {inp_files}"


def test_dry_run_pointwise_inp_theta_phi_lines(tmp_path):
    """Each pointwise .inp should have NT=1, NP=1, scalar THETA/PHI."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sprkkr" / "sprkkr_e2e.py"),
        "--pot", str(VTE2_POT),
        "--hkl", "0", "1", "-1",
        "--iq-surf", "2",
        "--hv", "84",
        "--theta", "-15", "15", "5",
        "--deflector", "-10.3",
        "--slit", "0",
        "--pointwise",
        "--dry-run",
        "--out", str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0

    arpes_dir = tmp_path / "arpes"
    inp_files = sorted([f for f in arpes_dir.glob("**/*.inp") if "_p" in f.parent.name])
    assert len(inp_files) == 5

    for inp_path in inp_files:
        kw = read_inp_keywords(str(inp_path))
        assert kw["SPEC_EL"]["NT"] == "1", f"NT not 1 in {inp_path}"
        assert kw["SPEC_EL"]["NP"] == "1", f"NP not 1 in {inp_path}"
        # THETA/PHI should be bare scalars (not lists like {-15, 15})
        theta_str = kw["SPEC_EL"]["THETA"]
        phi_str = kw["SPEC_EL"]["PHI"]
        assert not theta_str.startswith("{"), f"THETA is a set in {inp_path}: {theta_str}"
        assert not phi_str.startswith("{"), f"PHI is a set in {inp_path}: {phi_str}"


def test_dry_run_pointwise_prints_every_point(tmp_path):
    """Pointwise dry-run should print one [pt] line per point."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sprkkr" / "sprkkr_e2e.py"),
        "--pot", str(VTE2_POT),
        "--hkl", "0", "1", "-1",
        "--iq-surf", "2",
        "--hv", "84",
        "--theta", "-15", "15", "5",
        "--deflector", "-10.3",
        "--slit", "0",
        "--pointwise",
        "--dry-run",
        "--out", str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0

    stdout = result.stdout
    pt_lines = [line for line in stdout.split("\n") if "[pt] i=" in line]
    assert len(pt_lines) == 5, f"Expected 5 [pt] lines, got {len(pt_lines)}"

    summary_lines = [line for line in stdout.split("\n") if "[pt] N=" in line and "crosses_gamma" in line]
    assert len(summary_lines) == 1, f"Expected 1 summary line, got {len(summary_lines)}"
    assert "crosses_gamma=False" in summary_lines[0]


def test_dry_run_non_pointwise_unchanged(tmp_path):
    """Non-pointwise dry-run should still write exactly one .inp with NT=5."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sprkkr" / "sprkkr_e2e.py"),
        "--pot", str(VTE2_POT),
        "--hkl", "0", "1", "-1",
        "--iq-surf", "2",
        "--hv", "84",
        "--theta", "-15", "15", "5",
        "--deflector", "-10.3",
        "--slit", "0",
        "--dry-run",
        "--out", str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0

    arpes_dir = tmp_path / "arpes"
    inp_files = [f for f in arpes_dir.glob("*.inp")]
    assert len(inp_files) == 1, f"Expected 1 .inp file, got {len(inp_files)}"

    kw = read_inp_keywords(str(inp_files[0]))
    assert kw["SPEC_EL"]["NT"] == "5", f"NT should be 5 (non-pointwise), got {kw['SPEC_EL']['NT']}"

    # Check that pointwise [pt] lines do NOT appear
    stdout = result.stdout
    pt_lines = [line for line in stdout.split("\n") if "[pt] i=" in line]
    assert len(pt_lines) == 0, "Non-pointwise should not print [pt] lines"
