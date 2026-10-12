"""Adapt DINOv2 ViT-S/14 pretrained weights into our ViT2D backbone."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from tensorspec.core.ml.ssl.models.vit2d import ViT2D

# Keys that exist in DINOv2 but not in our ViT2D (heads, register tokens, etc.)
_SKIP_PREFIXES = ("head.", "dino_head.")


def _adapt_patch_embed(
    w_rgb: torch.Tensor, bias: torch.Tensor, out_chans: int, out_patch: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """RGB [D,3,P,P] -> mean over channels -> interpolate to out_patch."""
    w = w_rgb.mean(dim=1, keepdim=True)
    if out_chans != 1:
        w = w.expand(-1, out_chans, -1, -1).contiguous()
    w = F.interpolate(w, size=(out_patch, out_patch), mode="bilinear", align_corners=False)
    return w, bias


_DINOV2_VITS14_NUM_REGISTERS = 4


def _spatial_pos_tokens(pos: torch.Tensor, num_registers: int = 0) -> torch.Tensor:
    """Return spatial grid tokens, skipping CLS and optional register tokens."""
    rest = pos[:, 1:]
    n_rest = rest.shape[1]
    g0 = int(n_rest**0.5)
    if g0 * g0 == n_rest:
        return rest

    candidates = (num_registers,) if num_registers else (_DINOV2_VITS14_NUM_REGISTERS, 1, 8)
    for n_reg in candidates:
        if n_reg >= n_rest:
            continue
        n_spat = n_rest - n_reg
        g = int(n_spat**0.5)
        if g * g == n_spat:
            return rest[:, n_reg:]
    raise ValueError(
        f"pos_embed has {n_rest} non-CLS tokens; cannot infer spatial grid"
    )


def _adapt_pos_embed(
    pos: torch.Tensor, num_patches: int, *, num_registers: int = 0
) -> torch.Tensor:
    """Interpolate spatial pos_embed from DINOv2 grid to target patch count."""
    cls = pos[:, :1]
    spat = _spatial_pos_tokens(pos, num_registers)
    n0 = spat.shape[1]
    g0 = int(n0**0.5)
    d = spat.shape[-1]
    spat = spat.reshape(1, g0, g0, d).permute(0, 3, 1, 2)
    g1 = int(num_patches**0.5)
    spat = F.interpolate(spat, size=(g1, g1), mode="bicubic", align_corners=False)
    spat = spat.permute(0, 2, 3, 1).reshape(1, g1 * g1, d)
    return torch.cat([cls, spat], dim=1)


def _get_patch_embed_keys(sd: dict[str, torch.Tensor]) -> tuple[str, str] | None:
    if "patch_embed.proj.weight" in sd:
        return "patch_embed.proj.weight", "patch_embed.proj.bias"
    if "patch_embed.weight" in sd:
        return "patch_embed.weight", "patch_embed.bias"
    return None


def adapt_dinov2_state_to_vit2d(
    dinov2_sd: dict[str, torch.Tensor], vit: ViT2D
) -> dict[str, torch.Tensor]:
    """Map DINOv2 state dict keys/shapes to match *vit* backbone."""
    adapted: dict[str, torch.Tensor] = {}
    target_keys = set(vit.state_dict().keys())

    pe_keys = _get_patch_embed_keys(dinov2_sd)
    if pe_keys is not None:
        w_key, b_key = pe_keys
        w, b = _adapt_patch_embed(
            dinov2_sd[w_key],
            dinov2_sd[b_key],
            vit.patch_embed.in_channels,
            vit.patch_size,
        )
        adapted["patch_embed.weight"] = w
        adapted["patch_embed.bias"] = b

    if "cls_token" in dinov2_sd:
        adapted["cls_token"] = dinov2_sd["cls_token"]

    if "pos_embed" in dinov2_sd:
        adapted["pos_embed"] = _adapt_pos_embed(dinov2_sd["pos_embed"], vit.num_patches)

    for key, val in dinov2_sd.items():
        if key.startswith(_SKIP_PREFIXES):
            continue
        if key.startswith("patch_embed.") or key in ("cls_token", "pos_embed"):
            continue
        if key in target_keys:
            adapted[key] = val

    return adapted


def load_dinov2_vits14_state_dict(**kwargs: Any) -> dict[str, torch.Tensor]:
    """Download DINOv2 ViT-S/14 from torch.hub (requires network)."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True, **kwargs)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def load_pretrained_into_vit2d(vit: ViT2D, *, source: str = "dinov2_vits14") -> dict[str, int]:
    """Load and adapt pretrained weights into *vit* backbone (strict=False)."""
    if source != "dinov2_vits14":
        raise ValueError(f"unsupported pretrained source: {source!r}")

    dinov2_sd = load_dinov2_vits14_state_dict()
    adapted = adapt_dinov2_state_to_vit2d(dinov2_sd, vit)
    target_sd = vit.state_dict()
    overlap = {k: v for k, v in adapted.items() if k in target_sd and target_sd[k].shape == v.shape}
    vit.load_state_dict(overlap, strict=False)
    return {"loaded": len(overlap), "skipped": len(target_sd) - len(overlap)}


def load_dinov2_vit2d_for_probe(
    *,
    device: str | torch.device = "cpu",
    img_size: int = 128,
    patch_size: int = 16,
) -> ViT2D:
    """Build vit_s 1-ch ViT2D, load DINOv2 weights, eval mode on *device*."""
    from tensorspec.core.ml.ssl.spec import ModelSpec
    from tensorspec.core.ml.ssl.models.vit2d import build_vit2d

    vit = build_vit2d(
        ModelSpec(name="vit_s", img_size=img_size, patch_size=patch_size, in_chans=1)
    )
    load_pretrained_into_vit2d(vit, source="dinov2_vits14")
    vit.to(device)
    vit.eval()
    return vit
