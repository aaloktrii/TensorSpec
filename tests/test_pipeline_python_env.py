"""Pipeline PYTHON/PYTHONPATH exports for sync_after_relax."""

from tensorspec.gui.components.qe_generator_panel import pipeline_python_env


def test_pipeline_python_env_local_exports():
    bash, ps = pipeline_python_env(None)
    assert 'export PYTHON="' in bash
    assert "export PYTHONPATH=" in bash
    assert "tensorspec" in bash.lower() or "TensorSpec" in bash
    assert "$env:PYTHON=" in ps
    assert "$env:PYTHONPATH=" in ps


def test_pipeline_python_env_cluster_uses_cluster_paths():
    cluster = {
        "user": "user",
        "host": "gpu.example.org",
        "paths": {
            "repo_root": "/opt/tensorspec/TensorSpec",
            "python": "/opt/tensorspec/TensorSpec/TensorSpec_env/bin/python",
        },
    }
    bash, _ = pipeline_python_env(cluster)
    assert '/opt/tensorspec/TensorSpec/TensorSpec_env/bin/python' in bash
    assert 'PYTHONPATH="/opt/tensorspec/TensorSpec:' in bash
