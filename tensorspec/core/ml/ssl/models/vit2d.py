"""ViT-2D backbone for Stage-1 DINO (no projection head)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tensorspec.core.ml.ssl.spec import ModelSpec

_MODEL_CFG: dict[str, dict[str, int]] = {
    "vit_ti": {"embed_dim": 192, "num_heads": 3, "depth": 12, "mlp_ratio": 4},
    "vit_s": {"embed_dim": 384, "num_heads": 6, "depth": 12, "mlp_ratio": 4},
}


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim {dim} must divide num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT2D(nn.Module):
    """Vision Transformer for 2D inputs; returns CLS and patch tokens only."""

    def __init__(
        self,
        *,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must divide patch_size {patch_size}")

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        grid = img_size // patch_size
        self.num_patches = grid * grid

        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.blocks = nn.ModuleList(
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                std = math.sqrt(2.0 / fan_out)
                nn.init.trunc_normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cls_token[B,D], patch_tokens[B,N,D]) before any DINO head."""
        b = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        if mask is not None:
            if mask.shape != x.shape[:2] or mask.dtype != torch.bool:
                raise ValueError("mask must be bool tensor shaped [B, N]")
            mask_tokens = self.mask_token.expand(b, x.shape[1], -1)
            x = torch.where(mask.unsqueeze(-1), mask_tokens, x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]

    def forward_visible(
        self, x: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode visible patches only.

        ``hidden`` is bool ``[B, N]`` with True on hidden patches. Every row
        must hide the same count. Returns ``(tokens [B, V, D], vis_idx [B, V])``.
        """
        if hidden.dtype != torch.bool or hidden.shape != (x.shape[0], self.num_patches):
            raise ValueError("hidden must be bool [B, N]")
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, 1:, :]
        visible = ~hidden
        counts = visible.sum(dim=1)
        if int(counts.min()) < 1:
            raise ValueError("encoder received no visible patches")
        if not torch.all(counts == counts[0]):
            raise ValueError("visible count must match across the batch")
        n_vis = int(counts[0].item())
        vis_idx = visible.nonzero(as_tuple=False)[:, 1].view(x.shape[0], n_vis)
        gathered = torch.gather(
            tokens, 1, vis_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        )
        for block in self.blocks:
            gathered = block(gathered)
        return self.norm(gathered), vis_idx


def build_vit2d(spec: ModelSpec) -> ViT2D:
    cfg = _MODEL_CFG[spec.name]
    return ViT2D(
        img_size=spec.img_size,
        patch_size=spec.patch_size,
        in_chans=spec.in_chans,
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
    )
