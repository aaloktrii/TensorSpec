"""Launch SPR-KKR binaries: local (subprocess) or remote (paramiko ssh/sbatch).

Zero Qt (sandy_rule.md layer 1). paramiko is imported lazily inside methods,
never at module load, so this module imports fine on a machine without it.

Binaries take the .inp file as an ARG, not stdin (design doc §0):
    kkrscf9.7 scf.inp > scf.out
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tensorspec.core.compute import cluster_paths

PathLike = Union[str, Path]

_KIND_TO_STEM: Dict[str, str] = {"scf": "kkrscf", "arpes": "kkrspec"}


@dataclass
class JobSpec:
    """One SPR-KKR run: which binary, on what input, in which directory."""

    kind: str  # "scf" | "arpes"
    workdir: str
    inp_name: str
    binary: str
    nproc: int = 1
    mpi: bool = False
    env: Dict[str, str] = field(default_factory=dict)
    log_name: str = "job.log"
    expected_outputs: List[str] = field(default_factory=list)


def resolve_binary(kind: str, nproc: int, bin_dir: PathLike, suffix: str = "9.7") -> str:
    """Pick the SPR-KKR binary path for ``kind`` in ``bin_dir``.

    ``kind`` -> stem: scf -> kkrscf, arpes -> kkrspec. When ``nproc`` > 1 and
    ``<stem><suffix>MPI`` exists, return that; otherwise fall back to the
    serial ``<stem><suffix>``. Raises FileNotFoundError if neither exists.
    """
    stem = _KIND_TO_STEM.get(kind)
    if stem is None:
        raise ValueError(f"Unknown job kind {kind!r}; expected one of {sorted(_KIND_TO_STEM)}")

    bin_dir = Path(bin_dir)
    serial_name = f"{stem}{suffix}"
    mpi_name = f"{serial_name}MPI"

    if nproc > 1:
        mpi_path = bin_dir / mpi_name
        if mpi_path.exists():
            return str(mpi_path)

    serial_path = bin_dir / serial_name
    if serial_path.exists():
        return str(serial_path)

    raise FileNotFoundError(
        f"No SPR-KKR binary for kind={kind!r} in {bin_dir} "
        f"(tried {mpi_name if nproc > 1 else '(mpi skipped, nproc=1)'}, {serial_name})"
    )


# ---------------------------------------------------------------------------
# Local launcher
# ---------------------------------------------------------------------------


@dataclass
class LocalHandle:
    proc: "subprocess.Popen"
    log_path: Path

    @property
    def pid(self) -> int:
        return self.proc.pid

    def poll(self) -> Optional[int]:
        return self.proc.poll()

    def wait(self, timeout: Optional[float] = None) -> int:
        return self.proc.wait(timeout=timeout)

    def kill(self) -> None:
        # start_new_session=True put the child in its own process group, so
        # this also kills any mpirun-spawned rank children.
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self.proc.kill()
            except Exception:
                pass


class LocalLauncher:
    """Launch SPR-KKR binaries as local subprocesses."""

    def __init__(self, bin_dir: PathLike, mpi_prefix: Optional[str] = None):
        self.bin_dir = Path(bin_dir)
        self.mpi_prefix = mpi_prefix

    def launch(self, job: JobSpec) -> LocalHandle:
        binary = str(job.binary)
        use_mpi = job.nproc > 1 and binary.endswith("MPI")

        cmd: List[str] = []
        if use_mpi:
            prefix = (
                self.mpi_prefix
                if self.mpi_prefix is not None
                else cluster_paths.mpi_launch_prefix(None, job.nproc)
            )
            cmd.extend(prefix.split())
        cmd.append(binary)
        cmd.append(job.inp_name)

        workdir = Path(job.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        log_path = workdir / job.log_name

        env = os.environ.copy()
        env.update(job.env or {})

        log_file = open(log_path, "wb")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(workdir),
                env=env,
                start_new_session=True,
            )
        finally:
            # The child dup'd the fd; the parent's copy can close now.
            log_file.close()

        return LocalHandle(proc=proc, log_path=log_path)


# ---------------------------------------------------------------------------
# Remote launcher — pure command/script builders (testable without ssh)
# ---------------------------------------------------------------------------


def build_remote_run_line(cluster: Dict[str, Any], job: JobSpec) -> str:
    if job.mpi and job.nproc > 1:
        prefix = cluster_paths.mpi_launch_prefix(cluster, job.nproc)
        return f"{prefix}{job.binary} {job.inp_name}"
    return f"{job.binary} {job.inp_name}"


def build_nohup_command(cluster: Dict[str, Any], job: JobSpec) -> str:
    """`nohup bash -c 'cd dir && exports && binary inp > log 2>&1 & echo $!'`."""
    run_line = build_remote_run_line(cluster, job)
    log_path = f"{job.workdir}/{job.log_name}"

    exports = " && ".join(
        part
        for part in (
            cluster_paths.shell_export_tmp(cluster, one_line=True),
            cluster_paths.shell_thread_limits(one_line=True),
            # Remote SPR-KKR builds may link against a conda QE env's libs.
            cluster_paths.qe_env_exports(cluster, one_line=True),
        )
        if part
    )
    env_exports = " && ".join(f'export {k}="{v}"' for k, v in (job.env or {}).items())

    parts = [f"cd {job.workdir}"]
    if exports:
        parts.append(exports)
    if env_exports:
        parts.append(env_exports)
    parts.append(run_line)

    # The WHOLE chain runs in a background subshell with ALL fds redirected.
    # If only the binary were redirected, the backgrounded `cd && ... &&` subshell
    # would still hold the ssh channel's stdout/stderr and the ssh exec would
    # block until the job finished (seen on a remote Linux host: wall=0.0s, no live logs).
    chain = " && ".join(parts)
    inner = f"( {chain} ) > {log_path} 2>&1 < /dev/null & echo $!"
    return f"nohup bash -c '{inner}'"


def build_sbatch_script(cluster: Dict[str, Any], job: JobSpec) -> str:
    """SLURM batch script text for ``job`` (uses real slurm_sbatch_header)."""
    header = cluster_paths.slurm_sbatch_header(
        cluster, mpi_ranks=job.nproc, job_name=f"ts_sprkkr_{job.kind}"
    )
    run_line = build_remote_run_line(cluster, job)
    log_path = f"{job.workdir}/{job.log_name}"

    lines = [header, "", f"cd {job.workdir}", cluster_paths.shell_export_tmp(cluster)]
    lines.append(cluster_paths.shell_thread_limits())
    lines.append(cluster_paths.qe_env_exports(cluster))
    for k, v in (job.env or {}).items():
        lines.append(f'export {k}="{v}"')
    lines.append(f"{run_line} > {log_path} 2>&1")
    return "\n".join(part for part in lines if part is not None) + "\n"


@dataclass
class RemoteHandle:
    launcher: "RemoteLauncher"
    job: JobSpec
    pid: Optional[str]
    jobid: Optional[str]
    log_path: str

    def is_running(self) -> bool:
        ssh = self.launcher.connect()
        if self.jobid:
            out, _ = self.launcher._run(ssh, f"squeue -j {self.jobid} -h")
            return bool(out.strip())
        if self.pid:
            out, _ = self.launcher._run(ssh, f"kill -0 {self.pid} 2>/dev/null && echo alive || echo dead")
            return "alive" in out
        return False

    def tail(self, n: int = 50) -> str:
        ssh = self.launcher.connect()
        out, _ = self.launcher._run(ssh, f"tail -n {n} {self.log_path}")
        return out

    def fetch(self, remote_names: List[str], local_dir: PathLike) -> List[Path]:
        ssh = self.launcher.connect()
        sftp = ssh.open_sftp()
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        fetched = []
        try:
            for name in remote_names:
                remote_path = f"{self.job.workdir}/{name}"
                local_path = local_dir / Path(name).name
                sftp.get(remote_path, str(local_path))
                fetched.append(local_path)
        finally:
            sftp.close()
        return fetched


class RemoteLauncher:
    """Launch SPR-KKR binaries on a remote cluster via ssh (paramiko).

    ``ssh_factory`` lets tests inject a fake ssh client (anything with
    ``exec_command`` and ``open_sftp``) instead of a real paramiko
    connection. Default (None) connects for real via
    ``cluster_paths.ssh_connect`` -- paramiko is imported lazily here, not
    at module load (sandy_rule.md).
    """

    def __init__(self, cluster: Dict[str, Any], ssh_factory=None):
        self.cluster = cluster
        self._ssh_factory = ssh_factory
        self._ssh = None

    def connect(self):
        if self._ssh is not None:
            return self._ssh
        if self._ssh_factory is not None:
            self._ssh = self._ssh_factory()
            return self._ssh

        import paramiko

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        cluster_paths.ssh_connect(client, self.cluster)
        self._ssh = client
        return self._ssh

    @staticmethod
    def _run(ssh, cmd: str) -> Tuple[str, str]:
        _stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read() if hasattr(stdout, "read") else b""
        err = stderr.read() if hasattr(stderr, "read") else b""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        return out, err

    def upload(self, local_paths: List[PathLike], remote_dir: str) -> None:
        ssh = self.connect()
        self._run(ssh, f'mkdir -p "{remote_dir}"')
        sftp = ssh.open_sftp()
        try:
            for lp in local_paths:
                lp = Path(lp)
                sftp.put(str(lp), f"{remote_dir}/{lp.name}")
        finally:
            sftp.close()

    def download(self, remote_paths: List[str], local_dir: PathLike) -> List[Path]:
        """sftp-get each of ``remote_paths`` into ``local_dir`` (created if needed).

        Used to pull a cluster-only file (e.g. a converged pot named by a
        remote path) down before a local step (ase2sprkkr) needs to read it.
        """
        ssh = self.connect()
        sftp = ssh.open_sftp()
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        fetched = []
        try:
            for rp in remote_paths:
                local_path = local_dir / Path(rp).name
                sftp.get(str(rp), str(local_path))
                fetched.append(local_path)
        finally:
            sftp.close()
        return fetched

    def _write_remote_file(self, ssh, remote_path: str, content: str) -> None:
        sftp = ssh.open_sftp()
        try:
            f = sftp.file(remote_path, "w")
            try:
                f.write(content)
            finally:
                f.close()
        finally:
            sftp.close()

    def launch(self, job: JobSpec) -> RemoteHandle:
        ssh = self.connect()

        if cluster_paths.is_slurm(self.cluster):
            script = build_sbatch_script(self.cluster, job)
            remote_script = f"{job.workdir}/job.sbatch"
            self._run(ssh, f'mkdir -p "{job.workdir}"')
            self._write_remote_file(ssh, remote_script, script)
            out, _err = self._run(ssh, f"cd {job.workdir} && sbatch job.sbatch")
            m = re.search(r"(\d+)", out)
            jobid = m.group(1) if m else None
            return RemoteHandle(
                launcher=self, job=job, pid=None, jobid=jobid,
                log_path=f"{job.workdir}/{job.log_name}",
            )

        cmd = build_nohup_command(self.cluster, job)
        self._run(ssh, f'mkdir -p "{job.workdir}"')
        out, _err = self._run(ssh, cmd)
        pid = out.strip().splitlines()[-1].strip() if out.strip() else None
        return RemoteHandle(
            launcher=self, job=job, pid=pid, jobid=None,
            log_path=f"{job.workdir}/{job.log_name}",
        )

    def close(self) -> None:
        if self._ssh is not None:
            try:
                self._ssh.close()
            except Exception:
                pass
            self._ssh = None
