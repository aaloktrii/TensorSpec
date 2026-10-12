"""3D masked autoencoder on one deflector × energy × slit cube per XY.

The encoder starts from a finished 2D image MAE. The 2D patch filter is
copied along the deflector tubelet and divided by the tubelet depth, and
the 2D positional table is repeated once per deflector patch. Tokens from
different deflector angles sit in one sequence.
"""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from tensorspec.core.ml.ssl.mae import health_indices, mae_learning_rate
from tensorspec.core.ml.ssl.models.vit2d import Block
from tensorspec.core.ml.ssl.shards import ShardDataset
from tensorspec.core.ml.ssl.train import _device_and_ddp


def volume_block_mask(
    batch: int,
    grid: tuple[int, int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Hide one solid half-slab along deflector, energy, or slit.

    ``grid`` is ``(defl, energy, slit)`` patch counts. Returns bool
    ``[batch, N]`` in the same order as ``patchify3d`` (defl, then energy,
    then slit). Every row hides the same number of patches.
    """
    gd, gh, gw = grid
    axis = int(rng.integers(0, 3))
    sizes = (gd, gh, gw)
    hide = sizes[axis] // 2
    if hide < 1 or hide >= sizes[axis]:
        raise ValueError(f"grid {grid} cannot hide a half-slab on axis {axis}")
    start = int(rng.integers(0, sizes[axis] - hide + 1))
    span = np.zeros(sizes[axis], dtype=bool)
    span[start : start + hide] = True
    planes = [np.ones(gd, dtype=bool), np.ones(gh, dtype=bool), np.ones(gw, dtype=bool)]
    planes[axis] = span
    hidden = planes[0][:, None, None] & planes[1][None, :, None] & planes[2][None, None, :]
    flat = np.broadcast_to(hidden.reshape(1, -1), (batch, gd * gh * gw)).copy()
    return flat


def patchify3d(
    volumes: torch.Tensor,
    tubelet: tuple[int, int, int],
) -> torch.Tensor:
    """``(B, D, H, W)`` → ``(B, N, tubelet volume)``."""
    if volumes.ndim == 5:
        volumes = volumes[:, 0]
    batch, depth, height, width = volumes.shape
    td, ph, pw = tubelet
    if depth % td or height % ph or width % pw:
        raise ValueError("volume does not divide the tubelet")
    gd, gh, gw = depth // td, height // ph, width // pw
    patches = volumes.reshape(batch, gd, td, gh, ph, gw, pw)
    patches = patches.permute(0, 1, 3, 5, 2, 4, 6)
    return patches.reshape(batch, gd * gh * gw, td * ph * pw)


def mae3d_reconstruction_loss(
    prediction: torch.Tensor,
    target_patches: torch.Tensor,
    hidden: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Per-patch normalized MSE on hidden patches of valid cubes."""
    mean = target_patches.mean(dim=-1, keepdim=True)
    std = target_patches.std(dim=-1, keepdim=True)
    target = (target_patches - mean) / (std + 1e-6)
    per_patch = (prediction - target).pow(2).mean(dim=-1)
    selected = per_patch * hidden.to(dtype=per_patch.dtype)
    per_sample = selected.sum(dim=-1) / hidden.to(dtype=per_patch.dtype).sum(dim=-1).clamp_min(1)
    weights = valid.to(dtype=per_sample.dtype)
    denom = weights.sum().clamp_min(1)
    return (per_sample * weights).sum() / denom


class ViT3D(nn.Module):
    """Tubelet ViT. No CLS token: the probe reads the patch-token mean."""

    def __init__(
        self,
        *,
        volume_size: tuple[int, int, int] = (16, 128, 128),
        tubelet_size: tuple[int, int, int] = (4, 16, 16),
        in_chans: int = 1,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        if any(size % patch for size, patch in zip(volume_size, tubelet_size)):
            raise ValueError("volume_size must divide tubelet_size")
        self.volume_size = volume_size
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim
        self.grid = tuple(size // patch for size, patch in zip(volume_size, tubelet_size))
        self.num_patches = int(math.prod(self.grid))
        td, ph, pw = tubelet_size
        self.patch_embed = nn.Conv3d(
            in_chans, embed_dim, kernel_size=(td, ph, pw), stride=(td, ph, pw)
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.blocks = nn.ModuleList(
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_visible(
        self, volumes: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if volumes.ndim == 4:
            volumes = volumes.unsqueeze(1)
        if hidden.dtype != torch.bool or hidden.shape != (volumes.shape[0], self.num_patches):
            raise ValueError("hidden must be bool [B, N]")
        tokens = self.patch_embed(volumes).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        visible = ~hidden
        counts = visible.sum(dim=1)
        if int(counts.min()) < 1:
            raise ValueError("encoder received no visible patches")
        if not torch.all(counts == counts[0]):
            raise ValueError("visible count must match across the batch")
        n_vis = int(counts[0].item())
        vis_idx = visible.nonzero(as_tuple=False)[:, 1].view(volumes.shape[0], n_vis)
        gathered = torch.gather(
            tokens, 1, vis_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        )
        for block in self.blocks:
            gathered = block(gathered)
        return self.norm(gathered), vis_idx

    def forward_features(self, volumes: torch.Tensor) -> torch.Tensor:
        """All patch tokens, unmasked. ``(B, N, D)``."""
        if volumes.ndim == 4:
            volumes = volumes.unsqueeze(1)
        tokens = self.patch_embed(volumes).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


class Mae3DDecoder(nn.Module):
    def __init__(
        self,
        *,
        num_patches: int,
        encoder_dim: int,
        decoder_dim: int,
        depth: int,
        num_heads: int,
        tubelet_size: tuple[int, int, int],
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        td, ph, pw = tubelet_size
        self.proj = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        self.blocks = nn.ModuleList(
            Block(decoder_dim, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, td * ph * pw)
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
        full.scatter_(1, index, projected.to(dtype=full.dtype))
        full = full + self.pos_embed
        for block in self.blocks:
            full = block(full)
        return self.head(self.norm(full))


class Mae3DModel(nn.Module):
    def __init__(self, encoder: ViT3D, decoder: Mae3DDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, volumes: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        tokens, vis_idx = self.encoder.forward_visible(volumes, hidden)
        return self.decoder(tokens, vis_idx, hidden)


def build_mae3d(
    *,
    embed_dim: int = 384,
    depth: int = 12,
    num_heads: int = 6,
    decoder_dim: int = 192,
    decoder_depth: int = 4,
    decoder_heads: int = 6,
    volume_size: tuple[int, int, int] = (16, 128, 128),
    tubelet_size: tuple[int, int, int] = (4, 16, 16),
) -> Mae3DModel:
    encoder = ViT3D(
        volume_size=volume_size,
        tubelet_size=tubelet_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
    )
    decoder = Mae3DDecoder(
        num_patches=encoder.num_patches,
        encoder_dim=embed_dim,
        decoder_dim=decoder_dim,
        depth=decoder_depth,
        num_heads=decoder_heads,
        tubelet_size=tubelet_size,
    )
    return Mae3DModel(encoder, decoder)


def _copy_matching(module: nn.Module, state: dict) -> None:
    own = module.state_dict()
    transferred = {
        key: value
        for key, value in state.items()
        if key in own and own[key].shape == value.shape
    }
    own.update(transferred)
    module.load_state_dict(own)


def inflate_from_mae2d(model: Mae3DModel, checkpoint: dict) -> None:
    """Copy 2D MAE transformer weights and spread the patch filter along deflector."""
    encoder_state = checkpoint["encoder"]
    _copy_matching(model.encoder, encoder_state)
    weight = encoder_state["patch_embed.weight"]
    depth = model.encoder.tubelet_size[0]
    inflated = weight.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth
    model.encoder.patch_embed.weight.data.copy_(inflated)
    if "patch_embed.bias" in encoder_state:
        model.encoder.patch_embed.bias.data.copy_(encoder_state["patch_embed.bias"])
    spatial = encoder_state["pos_embed"][:, 1:, :]
    repeats = model.encoder.grid[0]
    model.encoder.pos_embed.data.copy_(spatial.repeat(1, repeats, 1))
    if "decoder" in checkpoint:
        decoder_state = checkpoint["decoder"]
        _copy_matching(model.decoder, decoder_state)
        if "pos_embed" in decoder_state and decoder_state["pos_embed"].shape[-1] == model.decoder.pos_embed.shape[-1]:
            model.decoder.pos_embed.data.copy_(
                decoder_state["pos_embed"].repeat(1, repeats, 1)
            )


class _CubeDataset(Dataset):
    def __init__(self, root: str | Path) -> None:
        self.base = ShardDataset(root)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        array, provenance = self.base[index]
        valid = bool(np.isfinite(array).all() and float(np.max(array)) > float(np.min(array)))
        if not valid:
            array = np.zeros_like(array)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
        return tensor, valid, index


def _collate(batch):
    volumes, valid, indices = zip(*batch)
    return (
        torch.stack(volumes, 0),
        torch.tensor(valid, dtype=torch.bool),
        torch.tensor(indices, dtype=torch.long),
    )


def train_mae3d(
    *,
    data_dir: str | Path,
    out_dir: str | Path,
    init_checkpoint: str | Path,
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    min_lr: float = 1e-6,
    weight_decay: float = 0.05,
    warmup_epochs: int = 5,
    grad_clip: float = 3.0,
    seed: int = 0,
    workers: int = 4,
    log_every: int = 20,
    ckpt_every: int = 1000,
    amp: bool = True,
) -> dict:
    device, distributed, rank, world_size, _initialized = _device_and_ddp()
    is_main = rank == 0
    out = Path(out_dir)
    init_path = str(init_checkpoint)
    if is_main:
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    torch.manual_seed(seed + rank)
    model = build_mae3d().to(device)
    checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
    if "encoder" not in checkpoint:
        raise ValueError("2D checkpoint has no encoder key")
    inflate_from_mae2d(model, checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    amp_enabled = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if distributed:
        ddp_kwargs: dict = {}
        if device.type == "cuda":
            ddp_kwargs = {"device_ids": [device.index], "output_device": device.index}
        model = DistributedDataParallel(model, **ddp_kwargs)
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    grid = raw.encoder.grid
    tubelet = raw.encoder.tubelet_size
    dataset = _CubeDataset(data_dir)
    health_path = out / "health_indices.npy"
    if is_main and not health_path.exists():
        np.save(health_path, health_indices(len(dataset), seed))
    if distributed:
        dist.barrier()
    held = np.load(health_path)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        collate_fn=_collate,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    steps_per_epoch = len(loader)
    total_steps = epochs * steps_per_epoch
    metrics_path = out / "metrics.jsonl"
    step = 0
    health_mse: dict[int, float] = {}
    probe_allowed = True
    epoch = -1
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for volumes, valid, indices in loader:
            volumes = volumes.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            masks = [
                volume_block_mask(
                    1,
                    grid,
                    np.random.default_rng(seed + epoch * len(dataset) + int(index)),
                )[0]
                for index in indices.tolist()
            ]
            hidden = torch.from_numpy(np.stack(masks)).to(device)
            rate = mae_learning_rate(
                step, steps_per_epoch, warmup_epochs, total_steps, lr, min_lr
            )
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with autocast:
                prediction = model(volumes, hidden)
                loss = mae3d_reconstruction_loss(
                    prediction, patchify3d(volumes, tubelet), hidden, valid
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if is_main and step % log_every == 0:
                record = {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": float(loss.detach().cpu()),
                    "lr": rate,
                    "skipped": int((~valid).sum().cpu()),
                }
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
            if is_main and step % ckpt_every == 0:
                _save(out / "checkpoints" / f"step{step}.pt", raw, optimizer, scaler, step, epoch, init_path)
        if distributed:
            dist.barrier()
        if is_main:
            mse = _health_mse(raw, dataset, held, device, tubelet)
            health_mse[epoch + 1] = mse
            with metrics_path.open("a") as handle:
                handle.write(
                    json.dumps({"epoch": epoch + 1, "health_mse": mse, "step": step}) + "\n"
                )
            if epoch + 1 in (1, 10):
                _save(
                    out / "checkpoints" / f"epoch{epoch + 1}.pt",
                    raw, optimizer, scaler, step, epoch, init_path,
                )
        if epoch + 1 == 10 and 1 in health_mse and not (health_mse[10] < health_mse[1]):
            probe_allowed = False
        if distributed:
            flag = torch.zeros(1, dtype=torch.int32, device=device)
            if is_main and not probe_allowed:
                flag[0] = 1
            dist.broadcast(flag, src=0)
            dist.barrier()
            if int(flag.item()) == 1:
                probe_allowed = False
                break
        elif not probe_allowed:
            break
    if is_main:
        _save(out / "checkpoints" / "last.pt", raw, optimizer, scaler, step, epoch, init_path)
        (out / "probe_allowed.txt").write_text("yes\n" if probe_allowed else "no\n")
    return {"steps": step, "probe_allowed": probe_allowed, "health_mse": health_mse}


def _health_mse(model, dataset, held, device, tubelet) -> float:
    """Fixed energy-slab fill on the held-out cubes. Rank 0 only."""
    model.eval()
    losses = []
    hide = model.encoder.grid[1] // 2
    mask = np.zeros(model.encoder.grid, dtype=bool)
    mask[:, :hide, :] = True
    hidden = torch.from_numpy(mask.reshape(1, -1)).to(device)
    with torch.no_grad():
        for index in held.tolist():
            volume, valid, _ = dataset[int(index)]
            if not valid:
                continue
            batch = volume.unsqueeze(0).to(device)
            prediction = model(batch, hidden)
            target = patchify3d(batch, tubelet)
            losses.append(
                float(
                    mae3d_reconstruction_loss(
                        prediction,
                        target,
                        hidden,
                        torch.ones(1, dtype=torch.bool, device=device),
                    ).cpu()
                )
            )
    model.train()
    if not losses:
        return float("nan")
    return float(np.mean(losses))


def _save(path, model, optimizer, scaler, step, epoch, init_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "objective": "mae3d",
            "step": step,
            "epoch": epoch + 1,
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "init_checkpoint": init_path,
            "volume_size": model.encoder.volume_size,
            "tubelet_size": model.encoder.tubelet_size,
            "grid": model.encoder.grid,
        },
        path,
    )
