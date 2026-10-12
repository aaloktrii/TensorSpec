"""Masked autoencoder for disp2d Energy×slit frames."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from tensorspec.core.ml.ssl.models.vit2d import Block, ViT2D, build_vit2d
from tensorspec.core.ml.ssl.shards import ShardDataset
from tensorspec.core.ml.ssl.spec import MaeSpec, RunConfig, to_jsonable


def strip_block_mask(batch: int, grid: int, rng) -> np.ndarray:
    """Boolean mask shaped ``(batch, grid * grid)``. True marks a hidden patch.

    A coin flip below 0.5 hides energy rows ``grid[start:start+4, :]``.
    Otherwise it hides slit columns ``grid[:, start:start+4]``.
    ``start`` is drawn from ``rng.integers(0, 5)``.
    """
    if grid < 4:
        raise ValueError(f"grid {grid} cannot hold a 4-patch strip")
    masks = np.zeros((batch, grid, grid), dtype=bool)
    for i in range(batch):
        energy = float(rng.random()) < 0.5
        start = int(rng.integers(0, 5))
        if energy:
            masks[i, start : start + 4, :] = True
        else:
            masks[i, :, start : start + 4] = True
    return masks.reshape(batch, grid * grid)


def health_indices(n: int, seed: int = 0) -> np.ndarray:
    """Sorted ``n // 100`` shard indices from a numpy Generator."""
    size = int(n) // 100
    if size < 1:
        raise ValueError(f"need at least 100 samples for health slice (got {n})")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(int(n), size=size, replace=False)
    return np.sort(np.asarray(chosen, dtype=np.int64))


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Pack ``(B,H,W)`` or ``(B,C,H,W)`` into ``(B, N, C*P*P)``."""
    if images.ndim == 3:
        images = images.unsqueeze(1)
    if images.ndim != 4:
        raise ValueError("images must be (B,H,W) or (B,C,H,W)")
    batch, channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError("image size must divide patch_size")
    grid_h = height // patch_size
    grid_w = width // patch_size
    packed = images.reshape(batch, channels, grid_h, patch_size, grid_w, patch_size)
    packed = packed.permute(0, 2, 4, 3, 5, 1)
    return packed.reshape(batch, grid_h * grid_w, patch_size * patch_size * channels)


def mae_reconstruction_loss(
    pred: torch.Tensor,
    images: torch.Tensor,
    hidden: torch.Tensor,
    valid: torch.Tensor,
    patch_size: int = 16,
) -> torch.Tensor:
    """MSE on hidden patches after per-patch mean/std normalization.

    Invalid samples stay in the batch and contribute nothing. A batch with no
    valid hidden patches returns a zero that still tracks ``pred``'s device.
    """
    target = patchify(images, patch_size)
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
    mean = target.mean(dim=-1, keepdim=True)
    var = ((target - mean) ** 2).mean(dim=-1, keepdim=True)
    normalized = (target - mean) / (torch.sqrt(var) + 1e-6)
    per_patch = ((pred - normalized) ** 2).mean(dim=-1)
    weight = hidden.to(dtype=per_patch.dtype) * valid.to(dtype=per_patch.dtype).unsqueeze(-1)
    denom = weight.sum()
    if float(denom.detach()) == 0.0:
        return pred.sum() * 0.0
    return (per_patch * weight).sum() / denom


def mae_learning_rate(
    step: int,
    steps_per_epoch: int,
    warmup_epochs: float,
    total_steps: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """Linear warmup, then cosine from ``base_lr`` down to ``min_lr``."""
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    if warmup_steps and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


class MaeDecoder(nn.Module):
    """Lightweight decoder. Mask tokens live in decoder width, not encoder width."""

    def __init__(
        self,
        *,
        encoder_dim: int,
        decoder_dim: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        num_patches: int,
        mlp_ratio: int = 4,
        in_chans: int = 1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        self.blocks = nn.ModuleList(
            Block(decoder_dim, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_size * patch_size * in_chans)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        visible_tokens: torch.Tensor,
        vis_idx: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.proj(visible_tokens)
        batch, n_patches = hidden.shape
        full = self.mask_token.expand(batch, n_patches, -1).clone()
        index = vis_idx.unsqueeze(-1).expand(-1, -1, projected.shape[-1])
        # Autocast runs the projection in fp16 while the mask token stays fp32.
        full.scatter_(1, index, projected.to(dtype=full.dtype))
        full = full + self.pos_embed
        for block in self.blocks:
            full = block(full)
        return self.head(self.norm(full))


class MaeModel(nn.Module):
    def __init__(self, encoder: ViT2D, decoder: MaeDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        pixels = images.unsqueeze(1) if images.ndim == 3 else images
        tokens, vis_idx = self.encoder.forward_visible(pixels, hidden)
        return self.decoder(tokens, vis_idx, hidden)


def build_mae(config: RunConfig) -> MaeModel:
    encoder = build_vit2d(config.model)
    spec: MaeSpec = config.mae
    decoder = MaeDecoder(
        encoder_dim=encoder.embed_dim,
        decoder_dim=spec.decoder_dim,
        depth=spec.decoder_depth,
        num_heads=spec.decoder_heads,
        patch_size=encoder.patch_size,
        num_patches=encoder.num_patches,
        mlp_ratio=spec.mlp_ratio,
        in_chans=config.model.in_chans,
    )
    return MaeModel(encoder, decoder)


class MaeShardDataset(Dataset):
    """One full frame plus a strip mask. Invalid frames become zeros."""

    def __init__(self, base: ShardDataset, seed: int, patch_size: int) -> None:
        self.base = base
        self.seed = int(seed)
        self.patch_size = int(patch_size)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int):
        sample, _meta = self.base[index]
        image = np.asarray(sample, dtype=np.float32)
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]
        valid = bool(image.size and np.isfinite(image).all() and np.any(image))
        grid = int(image.shape[0]) // self.patch_size
        if not valid:
            side = grid * self.patch_size
            image = np.zeros((side, side), dtype=np.float32)
        rng = np.random.default_rng(self.seed + self.epoch * len(self.base) + int(index))
        hidden = strip_block_mask(1, grid, rng)[0]
        return (
            torch.from_numpy(np.ascontiguousarray(image)),
            torch.from_numpy(np.ascontiguousarray(hidden)),
            bool(valid),
        )


def _collate_mae(batch):
    images, hidden, valid = zip(*batch)
    return (
        torch.stack(images, dim=0),
        torch.stack(hidden, dim=0),
        torch.tensor(valid, dtype=torch.bool),
    )


def _unwrap(model: nn.Module) -> MaeModel:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _save_mae_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    epoch: int,
    config: RunConfig,
) -> None:
    raw = _unwrap(model)
    payload = {
        "objective": "mae",
        "step": int(step),
        "epoch": int(epoch),
        "encoder": raw.encoder.state_dict(),
        "decoder": raw.decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": to_jsonable(config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _append_jsonl(path: Path, row: dict, *, is_main: bool) -> None:
    if not is_main:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


@torch.no_grad()
def _health_mse(
    model: nn.Module,
    dataset: MaeShardDataset,
    indices: np.ndarray,
    device: torch.device,
    patch_size: int,
    batch_size: int,
) -> float:
    raw = _unwrap(model)
    was_training = raw.training
    raw.eval()
    total = 0.0
    weight = 0.0
    for start in range(0, len(indices), batch_size):
        chunk = [dataset[int(i)] for i in indices[start : start + batch_size]]
        images, hidden, valid = _collate_mae(chunk)
        images = images.to(device)
        hidden = hidden.to(device)
        valid = valid.to(device)
        pred = raw(images, hidden)
        loss = mae_reconstruction_loss(pred, images, hidden, valid, patch_size=patch_size)
        n_valid = float(valid.sum().item())
        total += float(loss.item()) * n_valid
        weight += n_valid
    if was_training:
        raw.train()
    if weight == 0.0:
        return float("inf")
    return total / weight


def train_mae(
    config: RunConfig,
    data_dir: str,
    out_dir: str,
    *,
    resume: str | None = None,
) -> dict[str, Any]:
    """Train the disp2d masked autoencoder. DINO multi-crop is not used."""
    if config.objective != "mae":
        raise ValueError(f"train_mae requires objective 'mae' (got {config.objective!r})")
    if config.pretrained:
        raise ValueError("MAE trains from scratch; pretrained init is refused")

    from tensorspec.core.ml.ssl.train import (
        _data_loader_workers,
        _device_and_ddp,
        _remaining_batches,
        _resume_position,
    )

    device, distributed, rank, world_size, initialized_here = _device_and_ddp()
    is_main = rank == 0
    output = Path(out_dir)
    checkpoints = output / "checkpoints"
    metrics_path = output / "metrics.jsonl"
    try:
        if is_main:
            checkpoints.mkdir(parents=True, exist_ok=True)
            (output / "run_config.json").write_text(
                json.dumps(to_jsonable(config), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if resume is None:
                metrics_path.write_text("", encoding="utf-8")
        if distributed:
            dist.barrier()

        seed = config.seed + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        base = ShardDataset(data_dir)
        dataset = MaeShardDataset(base, config.seed, config.model.patch_size)
        if is_main and not (output / "health_indices.npy").exists():
            np.save(output / "health_indices.npy", health_indices(len(base), seed=0))
        if distributed:
            dist.barrier()
        heldout = np.load(output / "health_indices.npy")

        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=config.seed,
            )
            if distributed
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=config.optim.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=_data_loader_workers(
                config.num_workers,
                resume=resume is not None,
                distributed=distributed,
            ),
            drop_last=False,
            collate_fn=_collate_mae,
        )
        if not len(loader):
            raise ValueError("training dataset is empty")
        steps_per_epoch = len(loader)
        total_steps = (
            config.max_steps
            if config.max_steps is not None
            else config.optim.epochs * steps_per_epoch
        )

        model = build_mae(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay_start,
        )
        amp_enabled = bool(config.optim.use_amp and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        step = 0
        if resume is not None:
            payload = torch.load(resume, map_location=device, weights_only=False)
            if not isinstance(payload, dict) or "encoder" not in payload:
                raise ValueError("checkpoint is missing encoder")
            raw = _unwrap(model)
            raw.encoder.load_state_dict(payload["encoder"])
            raw.decoder.load_state_dict(payload["decoder"])
            optimizer.load_state_dict(payload["optimizer"])
            scaler.load_state_dict(payload["scaler"])
            step = int(payload["step"])

        if distributed:
            ddp_kwargs: dict[str, Any] = {}
            if device.type == "cuda":
                ddp_kwargs = {"device_ids": [device.index], "output_device": device.index}
            # cls_token and the iBOT mask token stay on ViT2D for forward_features
            # but the MAE loss never reads them.
            ddp_kwargs["find_unused_parameters"] = True
            model = DistributedDataParallel(model, **ddp_kwargs)

        start_epoch, completed_batches = _resume_position(step, steps_per_epoch)
        last_loss: float | None = None
        skipped = 0
        epoch1_health: float | None = None
        probe_allowed = True
        stop = False

        for epoch in range(start_epoch, config.optim.epochs):
            dataset.set_epoch(epoch)
            if sampler is not None:
                sampler.set_epoch(epoch)
            batches = loader
            if epoch == start_epoch and completed_batches:
                batches = _remaining_batches(loader, completed_batches)
            for images, hidden, valid in batches:
                if step >= total_steps:
                    stop = True
                    break
                images = images.to(device, non_blocking=True)
                hidden = hidden.to(device, non_blocking=True)
                valid = valid.to(device)
                lr = mae_learning_rate(
                    step,
                    steps_per_epoch,
                    config.optim.warmup_epochs,
                    total_steps,
                    config.optim.lr,
                    config.mae.min_lr,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                    group["weight_decay"] = config.optim.weight_decay_start
                optimizer.zero_grad(set_to_none=True)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if amp_enabled
                    else nullcontext()
                )
                with autocast:
                    pred = model(images, hidden)
                    loss = mae_reconstruction_loss(
                        pred,
                        images,
                        hidden,
                        valid,
                        patch_size=config.model.patch_size,
                    )
                if not torch.isfinite(loss.detach()):
                    skipped += 1
                    if distributed:
                        zero = sum(parameter.sum() * 0.0 for parameter in model.parameters())
                        zero.backward()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if config.optim.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    last_loss = float(loss.detach().item())
                step += 1
                if is_main and config.log_every and step % config.log_every == 0:
                    _append_jsonl(
                        metrics_path,
                        {
                            "step": step,
                            "epoch": epoch,
                            "loss": last_loss,
                            "lr": lr,
                            "skipped": skipped,
                        },
                        is_main=True,
                    )
                if config.ckpt_every and step % config.ckpt_every == 0 and is_main:
                    _save_mae_checkpoint(
                        checkpoints / f"step_{step:06d}.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=step,
                        epoch=epoch,
                        config=config,
                    )
            if stop or step >= total_steps:
                break
            if is_main:
                health = _health_mse(
                    model,
                    dataset,
                    heldout,
                    device,
                    config.model.patch_size,
                    config.optim.batch_size,
                )
                finished_epoch = epoch + 1
                _append_jsonl(
                    metrics_path,
                    {"epoch": finished_epoch, "health_mse": health, "step": step},
                    is_main=True,
                )
                if finished_epoch == 1:
                    epoch1_health = health
                if (
                    finished_epoch == 10
                    and epoch1_health is not None
                    and not (health < epoch1_health)
                ):
                    probe_allowed = False
                    stop = True
            if distributed:
                flag = torch.tensor(
                    [0 if probe_allowed else 1], dtype=torch.int32, device=device
                )
                dist.broadcast(flag, src=0)
                if int(flag.item()) == 1:
                    probe_allowed = False
                    stop = True
            if stop:
                break

        finished_epoch, _ = _resume_position(step, steps_per_epoch)
        if is_main:
            _save_mae_checkpoint(
                checkpoints / "last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                epoch=finished_epoch,
                config=config,
            )
            (output / "probe_allowed.txt").write_text(
                "yes\n" if probe_allowed else "no\n", encoding="utf-8"
            )
        return {
            "steps": step,
            "epoch": finished_epoch,
            "last_loss": last_loss,
            "probe_allowed": probe_allowed,
            "skipped": skipped,
        }
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()
            os.environ.pop("WORLD_SIZE", None)
