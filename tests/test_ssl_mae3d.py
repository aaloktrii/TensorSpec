"""3D MAE: slab mask, 2D-weight inflation, one cube forward."""

import numpy as np
import torch

from tensorspec.core.ml.ssl.mae import MaeDecoder
from tensorspec.core.ml.ssl.mae3d import (
    build_mae3d,
    inflate_from_mae2d,
    mae3d_reconstruction_loss,
    patchify3d,
    volume_block_mask,
)
from tensorspec.core.ml.ssl.models.vit2d import ViT2D


def test_volume_block_mask_hides_one_half_slab():
    mask = volume_block_mask(2, (4, 8, 8), np.random.default_rng(0))
    assert mask.shape == (2, 256)
    assert mask.dtype == np.bool_
    assert mask.sum(axis=1).tolist() == [128, 128]
    cube = mask[0].reshape(4, 8, 8)
    slabs = [
        cube.all(axis=(1, 2)).sum() == 2 and (~cube).all(axis=(1, 2)).sum() == 2,
        cube.all(axis=(0, 2)).sum() == 4 and (~cube).all(axis=(0, 2)).sum() == 4,
        cube.all(axis=(0, 1)).sum() == 4 and (~cube).all(axis=(0, 1)).sum() == 4,
    ]
    assert sum(slabs) == 1


def test_inflate_spreads_2d_filter_along_deflector():
    encoder = ViT2D(img_size=16, patch_size=8, embed_dim=32, depth=1, num_heads=4)
    decoder = MaeDecoder(
        encoder_dim=32, decoder_dim=16, depth=1, num_heads=4, num_patches=4, patch_size=8
    )
    model = build_mae3d(
        embed_dim=32,
        depth=1,
        num_heads=4,
        decoder_dim=16,
        decoder_depth=1,
        decoder_heads=4,
        volume_size=(8, 16, 16),
        tubelet_size=(4, 8, 8),
    )
    inflate_from_mae2d(
        model,
        {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
    )
    weight = encoder.patch_embed.weight.detach()
    got = model.encoder.patch_embed.weight.detach()
    assert got.shape == (32, 1, 4, 8, 8)
    assert torch.allclose(got[:, :, 0], weight / 4)
    assert torch.allclose(got[:, :, 3], weight / 4)
    spatial = encoder.pos_embed.detach()[:, 1:, :]
    assert torch.allclose(model.encoder.pos_embed.detach()[:, :4], spatial)
    assert torch.allclose(model.encoder.pos_embed.detach()[:, 4:], spatial)
    assert torch.allclose(
        model.encoder.blocks[0].norm1.weight.detach(),
        encoder.blocks[0].norm1.weight.detach(),
    )


def test_loss_ignores_visible_patches():
    prediction = torch.zeros(2, 4, 8)
    target = torch.zeros(2, 4, 8)
    target[:, 1] = torch.arange(8, dtype=torch.float32)
    hidden = torch.tensor([[False, True, False, False], [False, True, False, False]])
    valid = torch.tensor([True, False])
    loss = mae3d_reconstruction_loss(prediction, target, hidden, valid)
    assert loss.ndim == 0
    assert float(loss) > 0
    visible_only = mae3d_reconstruction_loss(
        prediction, target, torch.zeros_like(hidden), valid
    )
    assert float(visible_only) == 0.0


def test_forward_predicts_patch_voxels():
    model = build_mae3d(
        embed_dim=32,
        depth=1,
        num_heads=4,
        decoder_dim=16,
        decoder_depth=1,
        decoder_heads=4,
        volume_size=(8, 16, 16),
        tubelet_size=(4, 8, 8),
    )
    volumes = torch.rand(2, 8, 16, 16)
    hidden = torch.from_numpy(volume_block_mask(2, (2, 2, 2), np.random.default_rng(1)))
    prediction = model(volumes, hidden)
    assert prediction.shape == patchify3d(volumes, (4, 8, 8)).shape
    loss = mae3d_reconstruction_loss(
        prediction,
        patchify3d(volumes, (4, 8, 8)),
        hidden,
        torch.ones(2, dtype=torch.bool),
    )
    loss.backward()
    assert model.encoder.patch_embed.weight.grad is not None
