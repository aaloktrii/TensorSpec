from pathlib import Path

import numpy as np
import torch

from tensorspec.core.ml.ssl.dino import DinoModel
from tensorspec.core.ml.ssl.models.vit2d import build_vit2d
from tensorspec.core.ml.ssl.probe import (
    ProbeConfig,
    agreement_metrics,
    cluster_embeddings,
    extract_cls_embeddings,
    extract_patch_mean_embeddings,
    extract_patch_mean_from_backbone,
    filter_manifest_indices,
    labels_to_grid,
    load_dino_for_probe,
    probe,
    probe_precomputed_embeddings,
    reference_to_binary,
    spatial_contiguity,
)
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

PROBE_SOURCE_ID = "synthetic_probe.h5"


def _tiny_ckpt(tmp_path):
    cfg = RunConfig(
        augment=AugmentSpec(arm="A1", n_global=2, n_local=0, global_size=32, local_size=32),
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
    return path, cfg


def test_filter_manifest_indices_by_source():
    manifest = {
        "samples": [
            {"source_id": "a.h5", "shard": 0, "offset": 0, "index": {"y": 0, "x": 0}},
            {"source_id": "b.h5", "shard": 0, "offset": 1, "index": {"y": 0, "x": 1}},
            {"source_id": "a.h5", "shard": 0, "offset": 2, "index": {"y": 1, "x": 0}},
        ]
    }
    assert filter_manifest_indices(manifest, "a.h5") == [0, 2]
    assert filter_manifest_indices(manifest, "missing.h5") == []


def _mini_probe_shards(data_dir: Path) -> None:
    """2x2 grid, 4 samples @ 32x32 float16; two intensity domains."""
    writer = ShardWriter(str(data_dir), target_bytes=10_000_000)
    coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for y, x in coords:
        level = 0.2 if x == 0 else 0.9
        img = np.full((32, 32), level, dtype=np.float32)
        writer.add(img, {"source_id": PROBE_SOURCE_ID, "index": {"y": y, "x": x}})
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
                "id": PROBE_SOURCE_ID,
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
        source_id=PROBE_SOURCE_ID,
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


def test_probe_patch_mean_records_embed(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _mini_probe_shards(data)
    ckpt, _ = _tiny_ckpt(tmp_path)
    ref_path = tmp_path / "floor.npz"
    _mini_reference(ref_path)
    out = tmp_path / "probe_out"

    metrics = probe(
        ckpt=ckpt,
        data_dir=data,
        reference=ref_path,
        out_dir=out,
        config=ProbeConfig(
            source_id=PROBE_SOURCE_ID,
            embed="patch_mean",
            k=2,
            seed=0,
            batch_size=2,
        ),
    )

    assert metrics["embed"] == "patch_mean"
    assert metrics["l2_normalize"] is True
    assert metrics["roi_mode"] == "full"
    emb = np.load(out / "embeddings.npy")
    assert emb.shape == (4, 192)


def test_probe_rejects_k_lt_2(tmp_path):
    import pytest

    with pytest.raises(ValueError, match=r"k>=2"):
        probe(
            ckpt=tmp_path / "missing.pt",
            data_dir=tmp_path,
            reference=tmp_path / "missing.npz",
            out_dir=tmp_path / "out",
            config=ProbeConfig(source_id="x", k=1),
        )


def test_probe_precomputed_writes_metrics(tmp_path):
    import json

    ny, nx, d = 4, 4, 8
    emb = np.zeros((ny * nx, d), dtype=np.float32)
    provenances = []
    i = 0
    for y in range(ny):
        for x in range(nx):
            emb[i, 0] = 0.0 if x < 2 else 10.0
            provenances.append({"index": {"y": y, "x": x}})
            i += 1
    ref_map = np.zeros((ny, nx), dtype=np.float32)
    ref_map[:, 2:] = 1.0
    ref = FloorReference(
        map=ref_map,
        y_axis=np.arange(ny, dtype=np.float64),
        x_axis=np.arange(nx, dtype=np.float64),
        roi={"source_id": "toy", "saved_utc": "t"},
    )
    out = tmp_path / "out"
    cfg = ProbeConfig(k=2, pca_dim=0, seed=0, source_id="toy")
    metrics = probe_precomputed_embeddings(
        embeddings=emb,
        provenances=provenances,
        reference=ref,
        out_dir=out,
        config=cfg,
        meta={"arm": "unit"},
    )
    assert (out / "embeddings.npy").exists()
    assert (out / "metrics.json").exists()
    assert (out / "fig_overlay.png").exists()
    assert metrics["ari"] > 0.5
    assert json.loads((out / "metrics.json").read_text())["arm"] == "unit"


def test_agreement_metrics_multiclass_ari():
    # 3 SSL domains vs binary ref: ARI defined; IoU is nan
    pred = np.array([[0, 0, 1], [0, 2, 1], [2, 2, 1]])
    ref = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
    m = agreement_metrics(pred, ref)
    assert m["ari"] == m["ari"]  # not nan
    assert m["nmi"] == m["nmi"]
    assert m["iou"] != m["iou"]  # nan
    assert 0.0 <= m["contiguity"] <= 1.0


def test_extract_cls_shape(tmp_path):
    path, cfg = _tiny_ckpt(tmp_path)
    model, loaded_cfg = load_dino_for_probe(path, device=torch.device("cpu"))
    images = np.random.randn(5, 32, 32).astype(np.float32)
    emb = extract_cls_embeddings(
        model, images, batch_size=2, use_teacher=True, device=torch.device("cpu")
    )
    assert emb.shape == (5, model.teacher.embed_dim)


def test_extract_patch_mean_shape_and_l2(tmp_path):
    path, cfg = _tiny_ckpt(tmp_path)
    model, _ = load_dino_for_probe(path, device=torch.device("cpu"))
    images = np.random.randn(5, 32, 32).astype(np.float32)
    emb = extract_patch_mean_embeddings(
        model,
        images,
        batch_size=2,
        use_teacher=True,
        device=torch.device("cpu"),
        l2_normalize=True,
    )
    assert emb.shape == (5, model.teacher.embed_dim)
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_extract_patch_mean_no_l2_differs(tmp_path):
    path, _ = _tiny_ckpt(tmp_path)
    model, _ = load_dino_for_probe(path, device=torch.device("cpu"))
    images = np.random.randn(3, 32, 32).astype(np.float32)
    a = extract_patch_mean_embeddings(
        model, images, batch_size=3, use_teacher=True,
        device=torch.device("cpu"), l2_normalize=False,
    )
    b = extract_patch_mean_embeddings(
        model, images, batch_size=3, use_teacher=True,
        device=torch.device("cpu"), l2_normalize=True,
    )
    assert a.shape == b.shape
    assert not np.allclose(a, b)


def test_extract_patch_mean_from_backbone_matches_wrapper():
    vit = build_vit2d(ModelSpec(name="vit_ti", img_size=128, patch_size=16, in_chans=1))

    class Wrap:
        def __init__(self, b):
            self.teacher = b
            self.student = b

        def eval(self):
            self.teacher.eval()

    images = np.random.randn(4, 128, 128).astype(np.float32)
    device = torch.device("cpu")
    a = extract_patch_mean_from_backbone(vit, images, batch_size=2, device=device)
    b = extract_patch_mean_embeddings(
        Wrap(vit), images, batch_size=2, use_teacher=True, device=device
    )
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_cluster_two_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, size=(40, 8))
    b = rng.normal(5, 0.1, size=(40, 8))
    emb = np.vstack([a, b]).astype(np.float32)
    labels = cluster_embeddings(emb, ProbeConfig(source_id="x", k=2, pca_dim=4, seed=0))
    assert set(labels.tolist()) == {0, 1}
    assert labels[:40].mean() != labels[40:].mean()  # separated


def test_cluster_full_embedding_skips_pca():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.1, size=(30, 16))
    b = rng.normal(3, 0.1, size=(30, 16))
    emb = np.vstack([a, b]).astype(np.float32)
    labels = cluster_embeddings(emb, ProbeConfig(source_id="x", k=2, pca_dim=0, seed=0))
    assert set(labels.tolist()) == {0, 1}
    assert labels[:30].mean() != labels[30:].mean()



def test_labels_to_grid_fills_yx():
    assigns = np.array([0, 1, 0, 1])
    prov = [
        {"index": {"y": 0, "x": 0}},
        {"index": {"y": 0, "x": 1}},
        {"index": {"y": 1, "x": 0}},
        {"index": {"y": 1, "x": 1}},
    ]
    grid = labels_to_grid(assigns, prov, ny=2, nx=2)
    np.testing.assert_array_equal(grid, [[0, 1], [0, 1]])


def test_labels_to_grid_rejects_hole():
    import pytest
    with pytest.raises(ValueError, match="incomplete"):
        labels_to_grid(
            np.array([0]),
            [{"index": {"y": 0, "x": 0}}],
            ny=2,
            nx=2,
        )


def test_agreement_permutation_invariant():
    ref = np.array([[0, 0], [1, 1]])
    pred = np.array([[1, 1], [0, 0]])  # swapped labels
    m = agreement_metrics(pred, ref)
    assert m["ari"] == 1.0
    assert m["nmi"] == 1.0
    assert m["iou"] == 1.0


def test_contiguity_perfect_blocks():
    lab = np.array([[0, 0], [1, 1]])
    assert spatial_contiguity(lab) == 1.0


def test_reference_to_binary_two_level_map():
    low = np.full((3, 4), 1.0, dtype=np.float32)
    high = np.full((3, 4), 10.0, dtype=np.float32)
    map_ = np.hstack([low, high])
    labels = reference_to_binary(map_, seed=0)
    assert labels.shape == map_.shape
    assert set(np.unique(labels).tolist()) == {0, 1}
    left = np.unique(labels[:, :4])
    right = np.unique(labels[:, 4:])
    assert left.size == 1 and right.size == 1
    assert left[0] != right[0]


def test_roi_slices_and_mask():
    from tensorspec.core.ml.ssl.probe import (
        apply_roi_mask,
        roi_slices_from_reference,
    )

    energy = np.linspace(-2.0, 0.5, 128)
    slit = np.linspace(-5.0, 5.0, 128)
    roi = {
        "dims": [
            {
                "label": "Energy",
                "physical_lo": float(energy[40]),
                "physical_hi": float(energy[45]),
            },
            {
                "label": "Angle",
                "physical_lo": float(slit[10]),
                "physical_hi": float(slit[20]),
            },
        ]
    }
    e_sl, s_sl = roi_slices_from_reference(roi, energy, slit)
    assert e_sl.start <= 40 and e_sl.stop >= 45
    assert s_sl.start <= 10 and s_sl.stop >= 20
    images = np.ones((3, 128, 128), dtype=np.float32)
    images[:, e_sl, s_sl] = 2.0
    masked = apply_roi_mask(images, e_sl, s_sl, renormalize=True)
    assert float(masked[:, 0, 0].sum()) == 0.0
    ey = (e_sl.start + e_sl.stop) // 2
    sx = (s_sl.start + s_sl.stop) // 2
    assert float(masked[:, ey, sx].min()) > 0.0
