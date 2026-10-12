import os
import time
from pathlib import Path

import pytest

from tensorspec.core.dft.sprkkr.jobs import (
    JobSpec,
    LocalLauncher,
    RemoteLauncher,
    build_nohup_command,
    build_sbatch_script,
    resolve_binary,
)
from tensorspec.core.dft.sprkkr.params import ArpesParams
from tensorspec.core.dft.sprkkr.progress import (
    EtaModel,
    arpes_fraction_done,
    arpes_rows_done,
    format_eta,
    scf_progress,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# resolve_binary
# ---------------------------------------------------------------------------

def _touch(path: Path) -> None:
    path.write_text("#!/bin/bash\n")
    path.chmod(0o755)


class TestResolveBinary:
    def test_picks_mpi_when_nproc_gt_1(self, tmp_path):
        _touch(tmp_path / "kkrscf9.7")
        _touch(tmp_path / "kkrscf9.7MPI")
        found = resolve_binary("scf", nproc=4, bin_dir=tmp_path)
        assert found == str(tmp_path / "kkrscf9.7MPI")

    def test_serial_fallback_when_no_mpi_binary(self, tmp_path):
        _touch(tmp_path / "kkrspec9.7")
        found = resolve_binary("arpes", nproc=8, bin_dir=tmp_path)
        assert found == str(tmp_path / "kkrspec9.7")

    def test_serial_when_nproc_1(self, tmp_path):
        _touch(tmp_path / "kkrscf9.7")
        _touch(tmp_path / "kkrscf9.7MPI")
        found = resolve_binary("scf", nproc=1, bin_dir=tmp_path)
        assert found == str(tmp_path / "kkrscf9.7")

    def test_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_binary("scf", nproc=1, bin_dir=tmp_path)

    def test_unknown_kind_raises(self, tmp_path):
        with pytest.raises(ValueError):
            resolve_binary("bogus", nproc=1, bin_dir=tmp_path)


# ---------------------------------------------------------------------------
# LocalLauncher
# ---------------------------------------------------------------------------

class TestLocalLauncher:
    def test_launch_runs_and_logs(self, tmp_path):
        binary = tmp_path / "fake_kkr.sh"
        binary.write_text("#!/bin/bash\necho hi $1\nsleep 0.2\n")
        binary.chmod(0o755)

        workdir = tmp_path / "run"
        workdir.mkdir()
        job = JobSpec(
            kind="scf", workdir=str(workdir), inp_name="scf.inp",
            binary=str(binary), nproc=1, log_name="job.log",
        )
        launcher = LocalLauncher(bin_dir=tmp_path)
        handle = launcher.launch(job)

        assert handle.poll() is None  # still running (sleep 0.2)
        rc = handle.wait(timeout=5)
        assert rc == 0
        assert handle.poll() == 0
        assert handle.log_path == workdir / "job.log"
        assert "hi scf.inp" in handle.log_path.read_text()

    def test_kill(self, tmp_path):
        binary = tmp_path / "slow.sh"
        binary.write_text("#!/bin/bash\nsleep 30\n")
        binary.chmod(0o755)

        workdir = tmp_path / "run"
        workdir.mkdir()
        job = JobSpec(
            kind="scf", workdir=str(workdir), inp_name="scf.inp",
            binary=str(binary), log_name="job.log",
        )
        launcher = LocalLauncher(bin_dir=tmp_path)
        handle = launcher.launch(job)
        assert handle.poll() is None

        handle.kill()
        rc = handle.wait(timeout=5)
        assert rc is not None
        assert rc != 0  # killed, not a clean exit


# ---------------------------------------------------------------------------
# RemoteLauncher — pure command-string builders (no real ssh)
# ---------------------------------------------------------------------------

class TestRemoteCommandBuilders:
    def _cluster(self, **extra):
        c = {"host": "cluster.example.edu", "user": "alice"}
        c.update(extra)
        return c

    def test_nohup_command_contains_cd_binary_and_log(self):
        cluster = self._cluster()
        job = JobSpec(
            kind="arpes", workdir="/scratch/alice/run1", inp_name="arpes.inp",
            binary="/opt/sprkkr/bin/kkrspec9.7", nproc=1, log_name="arpes.log",
        )
        cmd = build_nohup_command(cluster, job)
        assert "nohup bash -c" in cmd
        assert "cd /scratch/alice/run1" in cmd
        assert "/opt/sprkkr/bin/kkrspec9.7 arpes.inp" in cmd
        assert "> /scratch/alice/run1/arpes.log 2>&1" in cmd
        assert "echo $!" in cmd

    def test_sbatch_script_has_sbatch_directives(self):
        cluster = self._cluster(mode="SLURM", slurm={"account": "m1234"})
        job = JobSpec(
            kind="arpes", workdir="/scratch/alice/run1", inp_name="arpes.inp",
            binary="/opt/sprkkr/bin/kkrspec9.7MPI", nproc=4, mpi=True,
            log_name="arpes.log",
        )
        script = build_sbatch_script(cluster, job)
        assert "#SBATCH -A m1234" in script
        assert "#!/bin/bash" in script
        assert "cd /scratch/alice/run1" in script
        assert "srun -n 4" in script
        assert "kkrspec9.7MPI arpes.inp" in script

    def test_launch_uses_fake_ssh_nohup_path(self):
        calls = []

        class FakeStream:
            def __init__(self, data: bytes = b""):
                self._data = data

            def read(self):
                return self._data

        class FakeSSH:
            def exec_command(self, cmd):
                calls.append(cmd)
                if "sbatch" in cmd:
                    return None, FakeStream(b"Submitted batch job 555\n"), FakeStream(b"")
                if cmd.startswith("mkdir"):
                    return None, FakeStream(b""), FakeStream(b"")
                return None, FakeStream(b"12345\n"), FakeStream(b"")

            def open_sftp(self):
                raise AssertionError("sftp not needed for nohup launch")

        cluster = self._cluster()
        job = JobSpec(
            kind="scf", workdir="/scratch/alice/run1", inp_name="scf.inp",
            binary="/opt/sprkkr/bin/kkrscf9.7", log_name="scf.log",
        )
        launcher = RemoteLauncher(cluster, ssh_factory=FakeSSH)
        handle = launcher.launch(job)

        assert handle.pid == "12345"
        assert handle.jobid is None
        assert any("nohup bash -c" in c for c in calls)

    def test_launch_slurm_path_parses_jobid(self):
        class FakeStream:
            def __init__(self, data: bytes = b""):
                self._data = data

            def read(self):
                return self._data

        class FakeSFTPFile:
            def __init__(self):
                self.written = ""

            def write(self, content):
                self.written += content

            def close(self):
                pass

        class FakeSFTP:
            def __init__(self):
                self.last_file = None

            def file(self, path, mode):
                self.last_file = FakeSFTPFile()
                return self.last_file

            def close(self):
                pass

        class FakeSSH:
            def __init__(self):
                self.sftp = FakeSFTP()

            def exec_command(self, cmd):
                if "sbatch" in cmd:
                    return None, FakeStream(b"Submitted batch job 555\n"), FakeStream(b"")
                return None, FakeStream(b""), FakeStream(b"")

            def open_sftp(self):
                return self.sftp

        cluster = self._cluster(mode="SLURM", slurm={"account": "m1234"})
        job = JobSpec(
            kind="arpes", workdir="/scratch/alice/run1", inp_name="arpes.inp",
            binary="/opt/sprkkr/bin/kkrspec9.7", nproc=1, log_name="arpes.log",
        )
        launcher = RemoteLauncher(cluster, ssh_factory=FakeSSH)
        handle = launcher.launch(job)

        assert handle.jobid == "555"
        assert handle.pid is None
        assert "#SBATCH -A m1234" in launcher._ssh.sftp.last_file.written


# ---------------------------------------------------------------------------
# progress.py
# ---------------------------------------------------------------------------

class TestArpesProgress:
    def test_rows_done_on_fixture(self):
        assert arpes_rows_done(FIXTURES / "_Cu_ARPES_ARPES_data.spc") == 72  # NE=8 x NT=9

    def test_fraction_done_full(self):
        params = ArpesParams(ne=8, nt=9, np_=1)
        assert arpes_fraction_done(FIXTURES / "_Cu_ARPES_ARPES_data.spc", params) == 1.0

    def test_fraction_done_with_int_expected(self):
        assert arpes_fraction_done(FIXTURES / "_Cu_ARPES_ARPES_data.spc", 144) == pytest.approx(0.5)

    def test_missing_file_is_zero(self, tmp_path):
        assert arpes_rows_done(tmp_path / "nope.spc") == 0
        assert arpes_fraction_done(tmp_path / "nope.spc", 72) == 0.0


class TestScfProgress:
    def test_converged_from_fixture(self):
        status = scf_progress(FIXTURES / "Cu_SCF.out.tail")
        assert status.converged is True
        assert status.iterations == 12
        assert status.ef_ry == pytest.approx(0.65979)

    def test_missing_file_returns_empty_status(self, tmp_path):
        status = scf_progress(tmp_path / "nope.out")
        assert status.converged is False
        assert status.iterations == 0
        assert status.ef_ry is None


class TestEtaModel:
    def test_estimate_seconds(self):
        model = EtaModel(t_point_s=4.0)
        params = ArpesParams(ne=71, nt=21, np_=1)
        assert model.estimate_seconds(params, nproc=1) == pytest.approx(4.0 * 71 * 21)
        assert model.estimate_seconds(params, nproc=4) == pytest.approx(4.0 * 71 * 21 / 4)

    def test_calibrate_roundtrip_no_disk(self):
        model = EtaModel(t_point_s=4.0)
        params = ArpesParams(ne=10, nt=10, np_=1)  # n_points = 100
        t_point = model.calibrate(params, nproc=2, wall_s=200.0)
        assert t_point == pytest.approx(200.0 * 2 / 100)
        assert model.t_point_s == pytest.approx(4.0)

    def test_calibrate_persists_and_load_roundtrip(self, tmp_path):
        store = tmp_path / "eta.json"
        model = EtaModel(t_point_s=4.0, store_path=str(store))
        params = ArpesParams(ne=10, nt=10, np_=1)
        model.calibrate(params, nproc=2, wall_s=200.0, host="mac")
        assert store.exists()

        fresh = EtaModel(t_point_s=1.0, store_path=str(store))
        loaded = fresh.load(host="mac")
        assert loaded == pytest.approx(4.0)
        assert fresh.load(host="no-such-host") == pytest.approx(4.0)  # unchanged

    def test_format_eta(self):
        assert format_eta(9) == "9s"
        assert format_eta(65) == "1m 5s"
        assert format_eta(3661) == "1h 1m"


# ---------------------------------------------------------------------------
# Smoke test: real Cu ARPES NE=2 NT=3 NP=1 via LocalLauncher.
# Run once with SPRKKR_SMOKE=1 to prove end-to-end wiring.
# ---------------------------------------------------------------------------

SMOKE_BIN = Path("/home/claude/sprkkr_bench/bin/kkrspec9.7")
SMOKE_POT = Path("/home/claude/sprkkr_bench/cu_run/scf/Cu.pot_new")


@pytest.mark.skipif(
    not (SMOKE_BIN.exists() and os.environ.get("SPRKKR_SMOKE") == "1"),
    reason="real kkrspec9.7 binary + SPRKKR_SMOKE=1 required",
)
def test_smoke_real_arpes_run(tmp_path):
    from tensorspec.core.dft.sprkkr.inputs import build_arpes_inputs
    from tensorspec.core.dft.sprkkr.outputs import parse_spc
    from tensorspec.core.dft.sprkkr.params import ArpesParams

    params = ArpesParams(ne=2, nt=3, np_=1, dataset="_SmokeARPES")
    out_dir = tmp_path / "arpes_smoke"
    arpes_inputs = build_arpes_inputs(pot_path=str(SMOKE_POT), params=params, out_dir=out_dir)

    job = JobSpec(
        kind="arpes",
        workdir=str(out_dir),
        inp_name=arpes_inputs.inp_path.name,
        binary=str(SMOKE_BIN),
        nproc=1,
        log_name="smoke_run.log",
    )
    launcher = LocalLauncher(bin_dir=SMOKE_BIN.parent)

    start = time.time()
    handle = launcher.launch(job)
    rc = handle.wait(timeout=600)
    wall_s = time.time() - start
    print(f"\nSPRKKR_SMOKE wall time: {wall_s:.1f}s (returncode={rc})")

    assert rc == 0, handle.log_path.read_text()[-4000:]

    spc_path = out_dir / arpes_inputs.expected_spc
    assert spc_path.exists()
    ds = parse_spc(spc_path)
    assert ds["I_tot"].shape == (2, 3, 1)
