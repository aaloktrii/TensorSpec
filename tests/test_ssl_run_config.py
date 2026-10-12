import json

from tensorspec.core.ml.ssl.spec import RunConfig, run_config_from_dict, to_jsonable


def test_run_config_defaults_and_round_trip(tmp_path):
    cfg = RunConfig(seed=17, max_steps=50)
    cfg.augment.arm = "A0"
    cfg.model.name = "vit_ti"
    cfg.dino.gram_enabled = False
    payload = to_jsonable(cfg)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = run_config_from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert loaded.seed == 17
    assert loaded.max_steps == 50
    assert loaded.augment.arm == "A0"
    assert loaded.model.name == "vit_ti"
    assert loaded.dino.out_dim == 4096
    assert loaded.dino.gram_enabled is False


def test_run_config_pretrained_roundtrip():
    cfg = RunConfig(pretrained="dinov2_vits14")
    payload = to_jsonable(cfg)
    assert payload["pretrained"] == "dinov2_vits14"
    loaded = run_config_from_dict(payload)
    assert loaded.pretrained == "dinov2_vits14"


def test_run_config_rejects_unknown_arm():
    import pytest

    with pytest.raises((ValueError, KeyError, TypeError)):
        run_config_from_dict({"augment": {"arm": "A9"}, "model": {}, "dino": {}, "optim": {}})
