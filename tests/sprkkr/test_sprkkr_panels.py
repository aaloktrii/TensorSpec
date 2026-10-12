"""Gate C / P5 GUI tests: SPRKKRDftPanel / SPRKKRArpesPanel (qapp fixture).

ScfRunnerThread.start is monkeypatched to call run() synchronously (same
thread -> Qt signal/slot connections fire directly, no event loop needed).
A fake `kkrscf9.7` bash script stands in for the real binary: it cats a real
SCF-log fixture to stdout (captured by LocalLauncher into <dataset>_SCF.out)
and touches <dataset>.pot_new, matching what workflow.run_scf expects.
"""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ASE2SPRKKR = importlib.util.find_spec("ase2sprkkr") is not None
PYMATGEN = importlib.util.find_spec("pymatgen") is not None

pytestmark = pytest.mark.skipif(
    not (ASE2SPRKKR and PYMATGEN), reason="ase2sprkkr and/or pymatgen not installed"
)

from tensorspec.gui.components import sprkkr_panels

FIXTURES = Path(__file__).parent / "fixtures"
SCF_LOG_TAIL = FIXTURES / "Cu_SCF.out.tail"


def _cu_structure():
    from pymatgen.core import Lattice, Structure

    lat = Lattice.cubic(3.615)
    return Structure.from_spacegroup("Fm-3m", lat, ["Cu"], [[0, 0, 0]])


def _make_fake_kkrscf(bin_dir: Path) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "kkrscf9.7"
    script.write_text(f"#!/bin/bash\ncat '{SCF_LOG_TAIL}'\ntouch scf.pot_new\n")
    script.chmod(0o755)
    return script


class TestSPRKKRDftPanel:
    def test_run_scf_converges_and_registers_vault(self, qapp, tmp_path, monkeypatch):
        # Run the QThread synchronously in-thread so signals fire immediately.
        monkeypatch.setattr(sprkkr_panels.ScfRunnerThread, "start", lambda self: self.run())
        # QMessageBox.exec() is modal and blocks forever under offscreen w/
        # nobody to click it -- no-op it out for the headless test.
        monkeypatch.setattr(sprkkr_panels.QMessageBox, "information", staticmethod(lambda *a, **k: None))
        monkeypatch.setattr(sprkkr_panels.QMessageBox, "critical", staticmethod(lambda *a, **k: None))
        monkeypatch.setattr(sprkkr_panels.QMessageBox, "warning", staticmethod(lambda *a, **k: None))

        vault_root = tmp_path / "vault"
        monkeypatch.setenv("SPRKKR_VAULT", str(vault_root))

        bin_dir = tmp_path / "bin"
        _make_fake_kkrscf(bin_dir)

        engine = SimpleNamespace(crystal_structure=_cu_structure())
        panel = sprkkr_panels.SPRKKRDftPanel(engine=engine)
        panel.edit_bin_dir.setText(str(bin_dir))
        panel.spin_nproc.setValue(1)

        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            panel.run_scf()
        finally:
            os.chdir(cwd)

        status = panel.lbl_scf_status.text().lower()
        assert "converged=true" in status
        assert panel.txt_scf_tail.toPlainText().strip() != ""

        from tensorspec.core.dft.sprkkr import Vault

        vault = Vault(vault_root)
        entries = vault.list()
        assert len(entries) == 1
        assert Path(entries[0].pot_path).exists()

    def test_missing_binary_warns_no_crash(self, qapp, tmp_path, monkeypatch):
        monkeypatch.setattr(sprkkr_panels.ScfRunnerThread, "start", lambda self: self.run())
        monkeypatch.setattr(
            sprkkr_panels.QMessageBox, "warning", staticmethod(lambda *a, **k: None)
        )

        engine = SimpleNamespace(crystal_structure=_cu_structure())
        panel = sprkkr_panels.SPRKKRDftPanel(engine=engine)
        panel.edit_bin_dir.setText(str(tmp_path / "no_such_bin_dir"))

        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            panel.run_scf()  # must not raise
        finally:
            os.chdir(cwd)


class TestSPRKKRArpesPanel:
    def test_constructs_and_refreshes(self, qapp):
        panel = sprkkr_panels.SPRKKRArpesPanel()
        panel.refresh_vaults()
        assert panel.combo_vault.count() >= 1
