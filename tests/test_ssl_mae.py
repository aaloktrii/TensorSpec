import numpy as np
import pytest
import torch

from tensorspec.core.ml.ssl.mae import (
    MaeDecoder,
    MaeModel,
    health_indices,
    mae_reconstruction_loss,
    patchify,
    strip_block_mask,
)
from tensorspec.core.ml.ssl.models.vit2d import ViT2D, build_vit2d
from tensorspec.core.ml.ssl.probe import (
    extract_patch_mean_from_backbone,
    load_mae_encoder_for_probe,
)
from tensorspec.core.ml.ssl.spec import ModelSpec, RunConfig, run_config_from_dict, to_jsonable


class _FakeRng:
    def __init__(self, coin: float, start: int) -> None:
        self._coin = coin
        self._start = start

    def random(self) -> float:
        return self._coin

    def integers(self, low: int, high: int) -> int:
        assert low == 0 and high == 5
        return self._start


def test_energy_strip_is_four_consecutive_rows():
    mask = strip_block_mask(1, 8, _FakeRng(0.1, 1)).reshape(8, 8)
    assert mask[1:5].all()
    assert not mask[0].any()
    assert not mask[5:].any()
    assert int(mask.sum()) == 32


def test_slit_strip_is_four_consecutive_columns():
    mask = strip_block_mask(1, 8, _FakeRng(0.9, 2)).reshape(8, 8)
    assert mask[:, 2:6].all()
    assert not mask[:, :2].any()
    assert not mask[:, 6:].any()
    assert int(mask.sum()) == 32


def test_strip_block_mask_repeats_for_same_seed():
    first = strip_block_mask(3, 8, np.random.default_rng(0))
    second = strip_block_mask(3, 8, np.random.default_rng(0))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, 64)
    assert first.dtype == bool


def test_loss_ignores_visible_patches_and_invalid_samples():
    images = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    hidden = torch.zeros(1, 4, dtype=torch.bool)
    hidden[0, 0] = True
    pred = torch.zeros(1, 4, 4)
    valid = torch.tensor([True])
    loss_a = mae_reconstruction_loss(pred, images, hidden, valid, patch_size=2)
    pred_visible = pred.clone()
    pred_visible[0, 1:] = 99.0
    loss_b = mae_reconstruction_loss(pred_visible, images, hidden, valid, patch_size=2)
    assert torch.isfinite(loss_a)
    assert torch.allclose(loss_a, loss_b)
    loss_invalid = mae_reconstruction_loss(
        pred, images, hidden, torch.tensor([False]), patch_size=2
    )
    assert loss_invalid.item() == 0.0


def test_loss_uses_per_patch_normalized_target():
    images = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    target = patchify(images, 2)
    mean = target.mean(dim=-1, keepdim=True)
    var = ((target - mean) ** 2).mean(dim=-1, keepdim=True)
    normalized = (target - mean) / (torch.sqrt(var) + 1e-6)
    hidden = torch.ones(1, 4, dtype=torch.bool)
    valid = torch.tensor([True])
    loss = mae_reconstruction_loss(normalized, images, hidden, valid, patch_size=2)
    assert loss.item() < 1e-5
    raw = mae_reconstruction_loss(target, images, hidden, valid, patch_size=2)
    assert raw.item() > 1e-3


def test_forward_visible_returns_one_token_per_visible_patch():
    model = ViT2D(img_size=128, patch_size=16, embed_dim=32, depth=2, num_heads=4)
    images = torch.randn(2, 1, 128, 128)
    hidden = strip_block_mask(2, 8, _FakeRng(0.1, 0))
    hidden_t = torch.from_numpy(hidden)
    tokens, vis_idx = model.forward_visible(images, hidden_t)
    assert tokens.shape == (2, 32, 32)
    assert vis_idx.shape == (2, 32)
    cls, patches = model.forward_features(images)
    assert cls.shape == (2, 32)
    assert patches.shape == (2, 64, 32)


def test_smoke_step_lowers_finite_loss():
    torch.manual_seed(0)
    encoder = ViT2D(img_size=128, patch_size=16, embed_dim=32, depth=2, num_heads=4)
    decoder = MaeDecoder(
        encoder_dim=32,
        decoder_dim=16,
        depth=1,
        num_heads=4,
        patch_size=16,
        num_patches=64,
    )
    model = MaeModel(encoder, decoder)
    images = torch.zeros(2, 128, 128)
    images[:, 40:56, :] = 1.0
    hidden = torch.from_numpy(strip_block_mask(2, 8, np.random.default_rng(1)))
    valid = torch.tensor([True, True])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = mae_reconstruction_loss(
        model(images, hidden), images, hidden, valid, patch_size=16
    )
    assert torch.isfinite(start)
    for _ in range(5):
        opt.zero_grad()
        loss = mae_reconstruction_loss(
            model(images, hidden), images, hidden, valid, patch_size=16
        )
        loss.backward()
        opt.step()
    end = mae_reconstruction_loss(
        model(images, hidden), images, hidden, valid, patch_size=16
    )
    assert torch.isfinite(end)
    assert end.item() < start.item()


def test_decoder_scatter_accepts_autocast_half_projection():
    decoder = MaeDecoder(
        encoder_dim=32,
        decoder_dim=16,
        depth=1,
        num_heads=4,
        patch_size=16,
        num_patches=64,
    )
    visible = torch.randn(2, 32, 32)
    vis_idx = torch.arange(32).view(1, 32).expand(2, 32).contiguous()
    hidden = torch.zeros(2, 64, dtype=torch.bool)
    hidden[:, 32:] = True
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        pred = decoder(visible, vis_idx, hidden)
    assert pred.shape == (2, 64, 256)
    assert torch.isfinite(pred.float()).all()


def test_health_indices_are_sorted_one_percent():
    first = health_indices(250, seed=0)
    second = health_indices(250, seed=0)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2,)
    assert np.all(first[1:] >= first[:-1])
    assert first.min() >= 0 and first.max() < 250


def test_missing_objective_stays_dino():
    cfg = run_config_from_dict({"model": {}, "optim": {}})
    assert cfg.objective == "dino"


def test_mae_checkpoint_patch_mean_and_missing_key(tmp_path):
    cfg = RunConfig(objective="mae")
    cfg.model = ModelSpec(name="vit_s", img_size=128, patch_size=16, in_chans=1)
    encoder = build_vit2d(cfg.model)
    good = tmp_path / "last.pt"
    torch.save({"encoder": encoder.state_dict(), "config": to_jsonable(cfg)}, good)
    loaded = load_mae_encoder_for_probe(good, device=torch.device("cpu"))
    images = np.random.default_rng(0).random((2, 128, 128)).astype(np.float32)
    emb = extract_patch_mean_from_backbone(
        loaded, images, batch_size=2, device=torch.device("cpu")
    )
    assert emb.shape == (2, 384)

    bad = tmp_path / "bad.pt"
    torch.save({"config": to_jsonable(cfg)}, bad)
    with pytest.raises(ValueError, match="encoder"):
        load_mae_encoder_for_probe(bad, device=torch.device("cpu"))
