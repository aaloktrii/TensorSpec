"""Unit tests for remote cluster path resolution."""

import os

import pytest

from tensorspec.core.compute.cluster_paths import (
    adapt_pipeline_mpi_launcher,
    build_qe_slurm_batch,
    heavy_root,
    is_arpes_fetch_candidate,
    is_slurm,
    job_dir,
    load_private_key,
    mpi_launch_prefix,
    python_bin,
    repo_root,
    slurm_account,
    slurm_qos,
    slurm_walltime,
    sprkkr_binary,
    tmp_dir,
    uses_sshproxy,
)


def test_lbl_defaults_when_paths_omitted():
    c = {"user": "alice", "host": "h", "name": "n"}
    assert heavy_root(c) == "/mnt/data/alice/tensorspec_heavy"
    assert tmp_dir(c) == "/mnt/data/alice/tmp"
    assert repo_root(c) == "/home/alice/TensorSpec"
    assert python_bin(c) == "/home/alice/TensorSpec/TensorSpec_env/bin/python"
    assert job_dir(c, "chinook") == "/mnt/data/alice/tensorspec_heavy/chinook_gui_run"


def test_custom_paths_block():
    c = {
        "user": "alice",
        "paths": {
            "heavy_root": "/scratch/ts",
            "tmp_dir": "/scratch/tmp",
            "repo_root": "/projects/ts",
            "python": "/projects/ts/.venv/bin/python",
            "sprkkr_bin": "/opt/sprkkr/bin",
        },
    }
    assert heavy_root(c) == "/scratch/ts"
    assert tmp_dir(c) == "/scratch/tmp"
    assert python_bin(c) == "/projects/ts/.venv/bin/python"
    assert job_dir(c, "qe") == "/scratch/ts/qe_gui_run"
    assert sprkkr_binary(c, "kkrspec9.7") == "/opt/sprkkr/bin/kkrspec9.7"


def test_legacy_top_level_keys():
    c = {
        "user": "bob",
        "heavy_root": "/data/heavy",
        "python": "/usr/bin/python3",
    }
    assert heavy_root(c) == "/data/heavy"
    assert python_bin(c) == "/usr/bin/python3"


def test_slurm_helpers_and_batch():
    c = {
        "user": "alice",
        "mode": "SLURM",
        "paths": {
            "heavy_root": "/scratch/ts",
            "qe_module": "espresso/7.5-libxc-7.0.0-cpu",
            "slurm": {
                "account": "m9999",
                "qos": "regular_0",
                "constraint": "cpu",
                "walltime": "06:00:00",
            },
        },
    }
    assert is_slurm(c)
    assert slurm_account(c) == "m9999"
    assert slurm_qos(c) == "regular_0"
    assert slurm_walltime(c) == "06:00:00"
    assert mpi_launch_prefix(c, 128).startswith("srun -n 128")
    batch = build_qe_slurm_batch(
        c, remote_dir="/scratch/ts/qe_gui_run", script_name="run_pipeline.sh", mpi_ranks=128
    )
    assert "#SBATCH -A m9999" in batch
    assert "module load espresso/7.5-libxc-7.0.0-cpu" in batch
    assert "bash run_pipeline.sh" in batch


def test_adapt_pipeline_mpi_launcher_slurm_vs_daemon():
    script = (
        "mpirun -np 128 pw.x -in scf.in\n"
        "mpirun -np 36 pw2wannier90.x -in pw2wan.in\n"
    )
    slurm = {"mode": "SLURM", "paths": {"slurm": {"account": "als"}}}
    adapted = adapt_pipeline_mpi_launcher(script, slurm)
    assert "srun -n 128 --cpu-bind=cores pw.x" in adapted
    assert "srun -n 36 --cpu-bind=cores pw2wannier90.x" in adapted
    assert "mpirun" not in adapted

    daemon = {"mode": "Daemon"}
    back = adapt_pipeline_mpi_launcher(adapted, daemon)
    assert "mpirun -np 128 pw.x" in back
    assert "mpirun -np 36 pw2wannier90.x" in back
    assert "--oversubscribe" not in back
    assert "--use-hwthread-cpus" not in back
    assert "srun" not in back

    # Legacy oversubscribe scripts still rewrite cleanly
    legacy = (
        "mpirun --use-hwthread-cpus --oversubscribe -np 16 pw.x -in scf.in\n"
    )
    cleaned = adapt_pipeline_mpi_launcher(legacy, daemon)
    assert cleaned.strip() == "mpirun -np 16 pw.x -in scf.in"

    local = adapt_pipeline_mpi_launcher(adapted, None)
    assert "mpirun -np" in local
    assert "srun" not in local


def test_mpi_launch_prefix_daemon_is_plain():
    assert mpi_launch_prefix({"mode": "Daemon"}, 8) == "mpirun -np 8 "
    assert "--oversubscribe" not in mpi_launch_prefix(None, 4)


def test_load_private_key_sshproxy_rsa_pem():
    # Point TENSORSPEC_TEST_SSHPROXY_KEY at a real sshproxy key to exercise this.
    key_path = os.path.expanduser(os.environ.get("TENSORSPEC_TEST_SSHPROXY_KEY", ""))
    if not key_path or not os.path.isfile(key_path):
        pytest.skip("TENSORSPEC_TEST_SSHPROXY_KEY not set / key not present")
    pkey = load_private_key(key_path)
    assert pkey is not None


def test_uses_sshproxy_and_arpes_fetch_filter():
    # Opt-in only: no host-name detection.
    assert not uses_sshproxy({"host": "login.hpc.example.org", "user": "u"})
    assert uses_sshproxy({"host": "gpu.example.edu", "auth": "sshproxy"})
    assert not uses_sshproxy({"host": "gpu.example.edu", "mode": "Daemon"})
    assert not uses_sshproxy({"host": "login.hpc.example.org", "auth": "password"})

    assert is_arpes_fetch_candidate("wannier90_hr.dat")
    assert is_arpes_fetch_candidate("scf.out")
    assert is_arpes_fetch_candidate("relax.out")
    assert is_arpes_fetch_candidate("relaxed_structure.cif")
    assert is_arpes_fetch_candidate("structure_template.cif")
    assert is_arpes_fetch_candidate("tensorspec_relax_meta.json")
    assert is_arpes_fetch_candidate("sys.out.full")
    assert is_arpes_fetch_candidate("scf.in")
    assert is_arpes_fetch_candidate("FERMI_ENERGY.txt")
    assert not is_arpes_fetch_candidate("wannier90.mmn")
    assert not is_arpes_fetch_candidate("wannier90.amn")
    assert not is_arpes_fetch_candidate("wannier90.eig")
    assert not is_arpes_fetch_candidate("wannier90.chk")
