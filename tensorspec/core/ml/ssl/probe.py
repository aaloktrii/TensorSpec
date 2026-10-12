from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tensorspec.core.ml.ssl.dino import DinoModel
from tensorspec.core.ml.ssl.models.vit2d import build_vit2d
from tensorspec.core.ml.ssl.reference import load_floor_reference
from tensorspec.core.ml.ssl.shards import ShardDataset, load_manifest
from tensorspec.core.ml.ssl.spec import run_config_from_dict


@dataclass(frozen=True)
class ProbeConfig:
    source_id: str
    k: int = 2
    pca_dim: int = 50
    seed: int = 0
    batch_size: int = 64
    use_teacher: bool = True
    roi_mode: str = "full"
    embed: str = "cls"
    l2_normalize: bool = True


def filter_manifest_indices(manifest: dict, source_id: str) -> list[int]:
    return [
        i
        for i, sample in enumerate(manifest["samples"])
        if sample.get("source_id") == source_id
    ]


def roi_slices_from_reference(
    roi: dict,
    energy_axis: np.ndarray,
    slit_axis: np.ndarray,
) -> tuple[slice, slice]:
    """Map floor-reference Energy/Angle physical window onto resampled (E, slit) grid."""
    energy_axis = np.asarray(energy_axis, dtype=np.float64)
    slit_axis = np.asarray(slit_axis, dtype=np.float64)
    dims = {d["label"]: d for d in roi.get("dims", [])}
    if "Energy" not in dims:
        raise ValueError("ROI missing Energy dim")
    angle = dims.get("Angle") or dims.get("Slit")
    if angle is None:
        raise ValueError("ROI missing Angle/Slit dim")
    e = dims["Energy"]

    def _bounds(axis: np.ndarray, lo: float, hi: float) -> slice:
        a_lo, a_hi = float(np.min(axis)), float(np.max(axis))
        lo_c = min(max(float(lo), a_lo), a_hi)
        hi_c = min(max(float(hi), a_lo), a_hi)
        if hi_c < lo_c:
            lo_c, hi_c = hi_c, lo_c
        i0 = int(np.searchsorted(axis, lo_c, side="left"))
        i1 = int(np.searchsorted(axis, hi_c, side="right"))
        i0 = max(0, min(i0, axis.size - 1))
        i1 = max(i0 + 1, min(i1, axis.size))
        return slice(i0, i1)

    return _bounds(energy_axis, e["physical_lo"], e["physical_hi"]), _bounds(
        slit_axis, angle["physical_lo"], angle["physical_hi"]
    )


def apply_roi_mask(
    images: np.ndarray,
    energy_slice: slice,
    slit_slice: slice,
    *,
    renormalize: bool = True,
) -> np.ndarray:
    """Zero outside ROI; optional per-sample min-max on the kept window."""
    x = np.asarray(images, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("images must be (N,H,W) with H=energy, W=slit")
    out = np.zeros_like(x)
    patch = x[:, energy_slice, slit_slice]
    if patch.size == 0:
        raise ValueError("ROI slice is empty")
    if renormalize:
        flat = patch.reshape(patch.shape[0], -1)
        lo = flat.min(axis=1, keepdims=True)
        hi = flat.max(axis=1, keepdims=True)
        scale = hi - lo
        flat_n = np.where(scale > 1e-6, (flat - lo) / np.maximum(scale, 1e-6), 1.0)
        patch = flat_n.reshape(patch.shape)
    out[:, energy_slice, slit_slice] = patch
    return out


def load_disp2d_axes(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    return np.asarray(payload["energy_axis"]), np.asarray(payload["slit_axis"])


def load_mae_encoder_for_probe(ckpt, *, device):
    """Load the ViT2D encoder stored under the checkpoint key ``encoder``."""
    path = Path(ckpt)
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "encoder" not in payload:
        raise ValueError("checkpoint is missing encoder")
    cfg = run_config_from_dict(payload["config"])
    encoder = build_vit2d(cfg.model)
    encoder.load_state_dict(payload["encoder"])
    encoder.to(device)
    encoder.eval()
    return encoder


def load_dino_for_probe(ckpt, *, device):
    path = Path(ckpt)
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = run_config_from_dict(payload["config"])
    student = build_vit2d(cfg.model)
    teacher = build_vit2d(cfg.model)
    model = DinoModel(student, teacher, cfg.dino)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def extract_cls_embeddings(model, images, *, batch_size, use_teacher, device):
    model.eval()
    backbone = model.teacher if use_teacher else model.student
    outs = []
    x_all = torch.from_numpy(np.asarray(images, dtype=np.float32))
    if x_all.ndim != 3:
        raise ValueError("images must be (N,H,W)")
    for i in range(0, len(x_all), batch_size):
        batch = x_all[i : i + batch_size].unsqueeze(1).to(device)  # N,1,H,W
        cls, _ = backbone.forward_features(batch)
        outs.append(cls.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def extract_patch_mean_from_backbone(
    backbone, images, *, batch_size, device, l2_normalize=True
):
    backbone.eval()
    outs = []
    x_all = torch.from_numpy(np.asarray(images, dtype=np.float32))
    if x_all.ndim != 3:
        raise ValueError("images must be (N,H,W)")
    for i in range(0, len(x_all), batch_size):
        batch = x_all[i : i + batch_size].unsqueeze(1).to(device)
        _, patches = backbone.forward_features(batch)
        vec = patches.float().mean(dim=1)
        if l2_normalize:
            vec = torch.nn.functional.normalize(vec, dim=-1)
        outs.append(vec.cpu().numpy())
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def extract_patch_mean_embeddings(
    model, images, *, batch_size, use_teacher, device, l2_normalize=True
):
    model.eval()
    backbone = model.teacher if use_teacher else model.student
    return extract_patch_mean_from_backbone(
        backbone,
        images,
        batch_size=batch_size,
        device=device,
        l2_normalize=l2_normalize,
    )


def cluster_embeddings(emb, cfg: ProbeConfig):
    """K-means on embeddings. ``pca_dim <= 0`` skips PCA (full embedding space)."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    emb = np.asarray(emb)
    n = emb.shape[0]
    # pca_dim <= 0: use full D-dimensional embedding (no PCA / no whitening).
    if cfg.pca_dim <= 0:
        z = emb
    else:
        dim = min(cfg.pca_dim, n - 1, emb.shape[1])
        if dim < 1:
            raise ValueError("not enough samples for PCA")
        z = PCA(n_components=dim, random_state=cfg.seed).fit_transform(emb)
    return KMeans(n_clusters=cfg.k, random_state=cfg.seed, n_init=10).fit_predict(z).astype(
        np.int32
    )


def labels_to_grid(assignments, provenances, *, ny, nx):
    if len(assignments) != len(provenances):
        raise ValueError("assignments and provenances length mismatch")
    grid = np.full((ny, nx), -1, dtype=np.int32)
    for lab, prov in zip(assignments, provenances, strict=True):
        y = int(prov["index"]["y"])
        x = int(prov["index"]["x"])
        grid[y, x] = int(lab)
    if (grid < 0).any():
        raise ValueError("incomplete spatial coverage for source grid")
    return grid


def reference_to_binary(map_, *, seed=0):
    from sklearn.cluster import KMeans

    flat = np.asarray(map_, dtype=np.float64).reshape(-1, 1)
    km = KMeans(n_clusters=2, random_state=seed, n_init=10)
    return km.fit_predict(flat).reshape(map_.shape).astype(np.int32)


def _mean_iou(pred: np.ndarray, ref: np.ndarray) -> float:
    classes = np.unique(np.concatenate([pred.ravel(), ref.ravel()]))
    ious: list[float] = []
    for c in classes:
        p_mask = pred == c
        r_mask = ref == c
        inter = float((p_mask & r_mask).sum())
        union = float((p_mask | r_mask).sum())
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def _best_permutation_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    pred = np.asarray(pred)
    ref = np.asarray(ref)
    if pred.shape != ref.shape:
        raise ValueError("pred and ref shape mismatch")

    best: dict[str, float] | None = None
    best_pred = pred
    for flip in (False, True):
        p = (1 - pred) if flip else pred
        iou = _mean_iou(p, ref)
        cand = {
            "ari": float(adjusted_rand_score(ref.ravel(), p.ravel())),
            "nmi": float(normalized_mutual_info_score(ref.ravel(), p.ravel())),
            "iou": float(iou),
        }
        if best is None or cand["iou"] > best["iou"]:
            best = cand
            best_pred = p
    assert best is not None
    best["contiguity"] = float(spatial_contiguity(best_pred))
    return best


def agreement_metrics(pred, ref) -> dict[str, float]:
    """Partition agreement. ARI/NMI work for any label sets; IoU uses flip match only when both binary."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    pred_arr = np.asarray(pred)
    ref_arr = np.asarray(ref)
    if pred_arr.shape != ref_arr.shape:
        raise ValueError("pred and ref shape mismatch")
    pred_labels = set(np.unique(pred_arr).tolist())
    ref_labels = set(np.unique(ref_arr).tolist())
    binary = pred_labels.issubset({0, 1}) and ref_labels.issubset({0, 1})
    if binary:
        return _best_permutation_metrics(pred_arr, ref_arr)
    return {
        "ari": float(adjusted_rand_score(ref_arr.ravel(), pred_arr.ravel())),
        "nmi": float(normalized_mutual_info_score(ref_arr.ravel(), pred_arr.ravel())),
        "iou": float("nan"),
        "contiguity": float(spatial_contiguity(pred_arr)),
    }


def spatial_contiguity(labels) -> float:
    """Fraction of pixels whose label equals the 8-neighbour majority (ties count).

    Out-of-bounds slots in the 3x3 neighbourhood are filled by edge-replicate
    padding: each missing neighbour takes the center pixel's label. Without this,
    corner/edge pixels on small grids (e.g. 2x2 block tests) see too many
    cross-boundary votes and contiguity collapses toward 0 even for perfect blocks.
    """
    lab = np.asarray(labels)
    if lab.ndim != 2:
        raise ValueError("labels must be 2D")
    if lab.size == 0:
        return 0.0

    ny, nx = lab.shape
    matches = 0
    for y in range(ny):
        for x in range(nx):
            center = int(lab[y, x])
            neigh: list[int] = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < ny and 0 <= xx < nx:
                        neigh.append(int(lab[yy, xx]))
                    else:
                        neigh.append(center)
            counts: dict[int, int] = {}
            for v in neigh:
                counts[v] = counts.get(v, 0) + 1
            max_count = max(counts.values())
            modes = [v for v, c in counts.items() if c == max_count]
            if center in modes:
                matches += 1
    return float(matches) / float(ny * nx)


def _save_overlay_pngs(out: Path, ref_map, ssl_map, ref_lab) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_map = np.asarray(ref_map)
    ssl_map = np.asarray(ssl_map)
    ref_lab = np.asarray(ref_lab)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(ref_map, origin="lower", aspect="equal")
    ax.set_title("reference intensity")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / "fig_ref.png", dpi=120)
    plt.close(fig)

    n_clusters = int(np.unique(ssl_map).size)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(
        ssl_map,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap="tab10",
        vmin=0,
        vmax=max(n_clusters - 1, 1),
    )
    ax.set_title(f"SSL cluster map (k={n_clusters})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=range(n_clusters))
    fig.tight_layout()
    fig.savefig(out / "fig_ssl.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(ref_lab, origin="lower", aspect="equal", interpolation="nearest")
    axes[0].set_title("ref binary")
    axes[1].imshow(
        ssl_map,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap="tab10",
        vmin=0,
        vmax=max(n_clusters - 1, 1),
    )
    axes[1].set_title(f"SSL k={n_clusters}")
    # Pixel disagrees if SSL label set vs binary ref is not informative; show SSL only side-by-side
    axes[2].imshow(ref_map, origin="lower", aspect="equal")
    axes[2].set_title("ref intensity")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "fig_overlay.png", dpi=120)
    plt.close(fig)


def probe(
    *,
    ckpt,
    data_dir,
    reference,
    out_dir,
    config: ProbeConfig,
    axes_path: str | Path | None = None,
) -> dict:
    if config.k < 2:
        raise ValueError(f"probe requires k>=2 (got k={config.k})")
    if config.roi_mode not in ("full", "mask"):
        raise ValueError(f"unknown roi_mode={config.roi_mode!r}")
    if config.embed not in ("cls", "patch_mean"):
        raise ValueError(f"unknown embed={config.embed!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = load_floor_reference(reference)
    manifest = load_manifest(str(Path(data_dir) / "manifest.json"))
    idxs = filter_manifest_indices(manifest, config.source_id)
    if not idxs:
        raise ValueError(f"no samples for source_id={config.source_id!r}")
    dataset = ShardDataset(str(data_dir))
    images = []
    provenances = []
    for i in idxs:
        sample, prov = dataset[i]
        images.append(np.asarray(sample, dtype=np.float32))
        provenances.append(prov)
    images = np.stack(images, axis=0)
    roi_meta: dict = {"roi_mode": config.roi_mode}
    if config.roi_mode == "mask":
        if axes_path is None:
            raise ValueError("axes_path required when roi_mode='mask'")
        energy_axis, slit_axis = load_disp2d_axes(axes_path)
        e_sl, s_sl = roi_slices_from_reference(ref.roi, energy_axis, slit_axis)
        images = apply_roi_mask(images, e_sl, s_sl, renormalize=True)
        roi_meta.update(
            {
                "energy_slice": [e_sl.start, e_sl.stop],
                "slit_slice": [s_sl.start, s_sl.stop],
                "axes_path": str(axes_path),
            }
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    header = torch.load(ckpt, map_location="cpu", weights_only=False)
    mae_ckpt = (
        isinstance(header, dict) and "encoder" in header and "model" not in header
    )
    if mae_ckpt:
        if config.embed != "patch_mean":
            raise ValueError("MAE probe requires embed=patch_mean")
        encoder = load_mae_encoder_for_probe(ckpt, device=device)
        emb = extract_patch_mean_from_backbone(
            encoder,
            images,
            batch_size=config.batch_size,
            device=device,
            l2_normalize=config.l2_normalize,
        )
    else:
        model, _run_cfg = load_dino_for_probe(ckpt, device=device)
        if config.embed == "cls":
            emb = extract_cls_embeddings(
                model,
                images,
                batch_size=config.batch_size,
                use_teacher=config.use_teacher,
                device=device,
            )
        else:
            emb = extract_patch_mean_embeddings(
                model,
                images,
                batch_size=config.batch_size,
                use_teacher=config.use_teacher,
                device=device,
                l2_normalize=config.l2_normalize,
            )
    np.save(out / "embeddings.npy", emb)
    assigns = cluster_embeddings(emb, config)
    ny, nx = ref.map.shape
    ssl_map = labels_to_grid(assigns, provenances, ny=ny, nx=nx)
    ref_lab = reference_to_binary(ref.map, seed=config.seed)
    metrics = agreement_metrics(ssl_map, ref_lab)
    metrics.update(
        {
            "source_id": config.source_id,
            "k": config.k,
            "pca_dim": config.pca_dim,
            "seed": config.seed,
            "use_teacher": config.use_teacher,
            "embed": config.embed,
            "l2_normalize": bool(config.l2_normalize)
            if config.embed == "patch_mean"
            else False,
            "n_samples": int(len(idxs)),
            "ckpt": str(ckpt),
            "reference_roi_source_id": ref.roi.get("source_id"),
            "reference_saved_utc": ref.roi.get("saved_utc"),
            **roi_meta,
        }
    )
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    _save_overlay_pngs(out, ref.map, ssl_map, ref_lab)
    return metrics


def probe_precomputed_embeddings(
    *,
    embeddings,
    provenances,
    reference,
    out_dir,
    config: ProbeConfig,
    meta: dict | None = None,
) -> dict:
    """Cluster + metrics + overlay from precomputed XY embeddings (no ckpt/shards)."""
    if config.k < 2:
        raise ValueError(f"probe requires k>=2 (got k={config.k})")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError("embeddings must be (N, D)")
    if len(provenances) != emb.shape[0]:
        raise ValueError("provenances length mismatch")
    ref = reference if hasattr(reference, "map") else load_floor_reference(reference)
    np.save(out / "embeddings.npy", emb)
    assigns = cluster_embeddings(emb, config)
    ny, nx = ref.map.shape
    ssl_map = labels_to_grid(assigns, provenances, ny=ny, nx=nx)
    ref_lab = reference_to_binary(ref.map, seed=config.seed)
    metrics = agreement_metrics(ssl_map, ref_lab)
    metrics.update(
        {
            "source_id": config.source_id,
            "k": config.k,
            "pca_dim": config.pca_dim,
            "seed": config.seed,
            "embed": "patch_mean",
            "n_samples": int(emb.shape[0]),
            **(meta or {}),
        }
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _save_overlay_pngs(out, ref.map, ssl_map, ref_lab)
    return metrics
