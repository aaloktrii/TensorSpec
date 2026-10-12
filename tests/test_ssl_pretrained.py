import torch

from tensorspec.core.ml.ssl.models.vit2d import build_vit2d
from tensorspec.core.ml.ssl.pretrained import (
    _adapt_pos_embed,
    adapt_dinov2_state_to_vit2d,
    load_pretrained_into_vit2d,
)
from tensorspec.core.ml.ssl.spec import ModelSpec


def _fake_dinov2_vits14_state(*, patch=14, grid=37, dim=384, depth=12):
    """Minimal DINOv2-like keys/shapes for offline tests."""
    sd = {}
    sd["patch_embed.proj.weight"] = torch.randn(dim, 3, patch, patch)
    sd["patch_embed.proj.bias"] = torch.randn(dim)
    sd["cls_token"] = torch.randn(1, 1, dim)
    sd["pos_embed"] = torch.randn(1, 1 + grid * grid, dim)
    for i in range(depth):
        sd[f"blocks.{i}.norm1.weight"] = torch.ones(dim)
        sd[f"blocks.{i}.norm1.bias"] = torch.zeros(dim)
        sd[f"blocks.{i}.attn.qkv.weight"] = torch.randn(dim * 3, dim)
        sd[f"blocks.{i}.attn.qkv.bias"] = torch.randn(dim * 3)
        sd[f"blocks.{i}.attn.proj.weight"] = torch.randn(dim, dim)
        sd[f"blocks.{i}.attn.proj.bias"] = torch.randn(dim)
        sd[f"blocks.{i}.norm2.weight"] = torch.ones(dim)
        sd[f"blocks.{i}.norm2.bias"] = torch.zeros(dim)
        sd[f"blocks.{i}.mlp.fc1.weight"] = torch.randn(dim * 4, dim)
        sd[f"blocks.{i}.mlp.fc1.bias"] = torch.randn(dim * 4)
        sd[f"blocks.{i}.mlp.fc2.weight"] = torch.randn(dim, dim * 4)
        sd[f"blocks.{i}.mlp.fc2.bias"] = torch.randn(dim)
    sd["norm.weight"] = torch.ones(dim)
    sd["norm.bias"] = torch.zeros(dim)
    return sd


def test_adapt_pos_embed_skips_register_tokens():
    dim = 384
    grid = 37
    n_reg = 4
    pos = torch.randn(1, 1 + n_reg + grid * grid, dim)
    vit = build_vit2d(ModelSpec(name="vit_s", img_size=128, patch_size=16, in_chans=1))
    adapted = _adapt_pos_embed(pos, vit.num_patches)
    assert adapted.shape == (1, 1 + vit.num_patches, dim)


def test_adapt_shapes_match_vit_s_128():
    vit = build_vit2d(ModelSpec(name="vit_s", img_size=128, patch_size=16, in_chans=1))
    adapted = adapt_dinov2_state_to_vit2d(_fake_dinov2_vits14_state(), vit)
    target = vit.state_dict()
    assert adapted["patch_embed.weight"].shape == target["patch_embed.weight"].shape
    assert adapted["pos_embed"].shape == target["pos_embed"].shape
    assert adapted["blocks.0.attn.qkv.weight"].shape == target["blocks.0.attn.qkv.weight"].shape


def test_load_changes_backbone_from_random(monkeypatch):
    vit = build_vit2d(ModelSpec(name="vit_s", img_size=128, patch_size=16, in_chans=1))
    before = vit.patch_embed.weight.detach().clone()
    import tensorspec.core.ml.ssl.pretrained as pret

    monkeypatch.setattr(pret, "load_dinov2_vits14_state_dict", lambda **kw: _fake_dinov2_vits14_state())
    stats = load_pretrained_into_vit2d(vit, source="dinov2_vits14")
    assert stats["loaded"] > 0
    assert not torch.allclose(vit.patch_embed.weight, before)
    x = torch.randn(2, 1, 128, 128)
    cls, patches = vit.forward_features(x)
    assert cls.shape == (2, 384)
    assert patches.shape == (2, 64, 384)


def test_load_dinov2_vit2d_for_probe_shape(monkeypatch):
    import tensorspec.core.ml.ssl.pretrained as pret

    def fake_load(v, *, source="dinov2_vits14"):
        assert source == "dinov2_vits14"
        return {"loaded": 1, "skipped": 0, "missing": 0, "unexpected": 0}

    monkeypatch.setattr(pret, "load_pretrained_into_vit2d", fake_load)
    out = pret.load_dinov2_vit2d_for_probe(device="cpu")
    assert out.img_size == 128
    x = torch.zeros(2, 1, 128, 128)
    cls, patches = out.forward_features(x)
    assert cls.shape == (2, 384)
    assert patches.shape[0] == 2
