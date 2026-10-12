"""End-to-end tests for the SSL probe CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tensorspec.core.ml.ssl.cli import main
from tensorspec.core.ml.ssl.dino import DinoModel
from tensorspec.core.ml.ssl.models.vit2d import build_vit2d
from tensorspec.core.ml.ssl.reference import (
    FloorReference,
    build_roi_dict,
    save_floor_reference,
)
from tensorspec.core.ml.ssl.shards import ShardWriter, write_manifest
from tensorspec.core.ml.ssl.spec import (
    AugmentSpec,
    DinoSpec,
    ModelSpec,
    OptimSpec,
    RunConfig,
    to_jsonable,
)


SOURCE_ID = "synthetic_probe.h5"


def _tiny_ckpt(tmp_path: Path) -> Path:
    cfg = RunConfig(
        augment=AugmentSpec(
            arm="A1", n_global=2, n_local=0, global_size=32, local_size=32
        ),
        model=ModelSpec(name="vit_ti", img_size=32, patch_size=8, in_chans=1),
        dino=DinoSpec(out_dim=64, hidden_dim=64, bottleneck_dim=32),
        optim=OptimSpec(epochs=1, batch_size=2, lr=1e-3),
        seed=0,
    )
    student = build_vit2d(cfg.model)
    teacher = build_vit2d(cfg.model)
    model = DinoModel(student, teacher, cfg.dino)
    path = tmp_path / "last.pt"
    torch.save(
        {
            "step": 1,
            "epoch": 0,
            "model": model.state_dict(),
            "optimizer": {},
            "scaler": {},
            "config": to_jsonable(cfg),
            "seed": 0,
            "git_commit": "test",
        },
        path,
    )
    return path


def _mini_probe_shards(data_dir: Path) -> None:
    """2x2 grid, 4 samples @ 32x32 float16; two intensity domains."""
    writer = ShardWriter(str(data_dir), target_bytes=10_000_000)
    coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for y, x in coords:
        # left column low, right column high — matches reference map
        level = 0.2 if x == 0 else 0.9
        img = np.full((32, 32), level, dtype=np.float32)
        writer.add(img, {"source_id": SOURCE_ID, "index": {"y": y, "x": x}})
    # distractor source (must be filtered out)
    writer.add(
        np.full((32, 32), 0.5, dtype=np.float32),
        {"source_id": "other.h5", "index": {"y": 0, "x": 0}},
    )
    partial = writer.close()
    manifest = {
        **partial,
        "preprocess": {"sample": {"mode": "disp2d", "index_roles": ["y", "x"]}},
        "sources": [
            {
                "id": SOURCE_ID,
                "size": 1,
                "sha256_head": "0" * 64,
                "sha256_tail": "0" * 64,
                "kind": "xy_fine_4d",
                "shape": [2, 2, 32, 32],
                "detector": {},
                "dead_pixel": {},
                "calibration": {
                    "deg_per_raw_px": 0.048,
                    "axis_source": "test",
                },
            },
            {
                "id": "other.h5",
                "size": 1,
                "sha256_head": "1" * 64,
                "sha256_tail": "1" * 64,
                "kind": "xy_fine_4d",
                "shape": [1, 1, 32, 32],
                "detector": {},
                "dead_pixel": {},
                "calibration": {
                    "deg_per_raw_px": 0.048,
                    "axis_source": "test",
                },
            },
        ],
    }
    write_manifest(str(data_dir / "manifest.json"), manifest)


def _mini_reference(path: Path) -> None:
    # left low, right high — same domain structure as shard intensities
    map_ = np.array([[1.0, 10.0], [1.0, 10.0]], dtype=np.float32)
    roi = build_roi_dict(
        labels=["Y", "X", "Energy", "Angle"],
        axes=[
            np.arange(2.0),
            np.arange(2.0),
            np.linspace(0, 1, 8),
            np.linspace(-1, 1, 6),
        ],
        coords={0: 0, 1: 0, 2: 3, 3: 2},
        halfwidths={0: 0, 1: 0, 2: 1, 3: 1},
        reduce_mode="sum",
        source_id=SOURCE_ID,
        display_y_label="Y",
        display_x_label="X",
        saved_utc="2026-09-07T12:00:00Z",
    )
    save_floor_reference(
        path,
        FloorReference(
            map=map_,
            y_axis=np.arange(2.0),
            x_axis=np.arange(2.0),
            roi=roi,
        ),
    )


def test_cli_probe_writes_metrics_and_figures(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _mini_probe_shards(data)
    ckpt = _tiny_ckpt(tmp_path)
    ref_path = tmp_path / "floor.npz"
    _mini_reference(ref_path)
    out = tmp_path / "probe_out"

    result = main(
        [
            "probe",
            "--ckpt",
            str(ckpt),
            "--data",
            str(data),
            "--reference",
            str(ref_path),
            "--out",
            str(out),
            "--source-id",
            SOURCE_ID,
            "--k",
            "2",
            "--seed",
            "0",
            "--batch-size",
            "2",
        ]
    )

    assert result == 0
    metrics_path = out / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["source_id"] == SOURCE_ID
    assert metrics["n_samples"] == 4
    assert metrics["k"] == 2
    assert metrics["use_teacher"] is True
    assert metrics["embed"] == "cls"
    assert metrics["l2_normalize"] is False
    for key in ("ari", "nmi", "iou", "contiguity"):
        assert key in metrics
    assert (out / "embeddings.npy").exists()
    emb = np.load(out / "embeddings.npy")
    assert emb.shape[0] == 4
    for name in ("fig_ref.png", "fig_ssl.png", "fig_overlay.png"):
        assert (out / name).exists()
        assert (out / name).stat().st_size > 0


def test_cli_probe_embed_patch_mean(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _mini_probe_shards(data)
    ckpt = _tiny_ckpt(tmp_path)
    ref_path = tmp_path / "floor.npz"
    _mini_reference(ref_path)
    out = tmp_path / "probe_out"

    result = main(
        [
            "probe",
            "--ckpt",
            str(ckpt),
            "--data",
            str(data),
            "--reference",
            str(ref_path),
            "--out",
            str(out),
            "--source-id",
            SOURCE_ID,
            "--embed",
            "patch_mean",
            "--k",
            "2",
            "--seed",
            "0",
            "--batch-size",
            "2",
        ]
    )

    assert result == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["embed"] == "patch_mean"
    assert metrics["l2_normalize"] is True
    assert metrics["roi_mode"] == "full"
    emb = np.load(out / "embeddings.npy")
    assert emb.shape == (4, 192)
