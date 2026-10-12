"""SSL preprocess config dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

AxisRole = Literal["energy", "slit", "defl", "x", "y", "other"]
SampleModeName = Literal["fermi3d", "disp2d"]
AugmentArm = Literal["A0", "A1"]
ModelName = Literal["vit_ti", "vit_s"]
ObjectiveName = Literal["dino", "mae"]

_VALID_AUGMENT_ARMS = frozenset({"A0", "A1"})
_VALID_MODEL_NAMES = frozenset({"vit_ti", "vit_s"})


@dataclass
class TrimSpec:
    ranges: dict[str, tuple[float, float]]  # role -> (lo, hi)
    source_kind: str
    note: str = ""


@dataclass
class NormSpec:
    clip_percentiles: tuple[float, float] = (1.0, 99.0)
    scope: Literal["per_sample", "per_file"] = "per_sample"
    dead_pixel_sigma: float = 6.0
    subsample_points: int = 512


@dataclass
class ResampleSpec:
    energy_size: int = 224
    slit_size: int = 224
    defl_size: int = 16  # used for fermi3d
    deg_per_raw_px: float | None = None  # None -> lookup table


@dataclass
class SampleSpec:
    mode: SampleModeName
    index_roles: tuple[str, ...]  # e.g. ("y", "x") or ("x",) or ()


@dataclass
class AugmentSpec:
    arm: AugmentArm = "A1"
    n_global: int = 2
    n_local: int = 6
    global_size: int = 128
    local_size: int = 96
    global_crop_frac: float = 0.875
    local_crop_frac: float = 0.4
    poisson_scale: float = 1000.0
    gain_jitter: float = 0.05
    energy_shift_px: float = 2.0
    dead_pixel_prob: float = 0.01
    smooth_sigma: float = 0.5
    mirror_prob: float = 0.5
    rrc_scale_min: float = 0.2
    rrc_scale_max: float = 1.0


@dataclass
class ModelSpec:
    name: ModelName = "vit_s"
    img_size: int = 128
    patch_size: int = 16
    in_chans: int = 1


@dataclass
class DinoSpec:
    out_dim: int = 4096
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    student_temp: float = 0.1
    teacher_temp_start: float = 0.04
    teacher_temp_end: float = 0.07
    teacher_temp_warmup_epochs: float = 30.0
    center_momentum: float = 0.9
    momentum_teacher: float = 0.996
    ibot_mask_ratio: float = 0.3
    ibot_weight: float = 1.0
    koleo_weight: float = 0.1
    gram_enabled: bool = False
    gram_weight: float = 0.0


@dataclass
class MaeSpec:
    decoder_dim: int = 192
    decoder_depth: int = 4
    decoder_heads: int = 6
    mlp_ratio: int = 4
    min_lr: float = 1e-6


@dataclass
class OptimSpec:
    lr: float = 0.0005
    weight_decay_start: float = 0.04
    weight_decay_end: float = 0.4
    warmup_epochs: float = 10.0
    epochs: int = 100
    batch_size: int = 64
    grad_clip: float = 3.0
    use_amp: bool = True


@dataclass
class RunConfig:
    augment: AugmentSpec = field(default_factory=AugmentSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    dino: DinoSpec = field(default_factory=DinoSpec)
    optim: OptimSpec = field(default_factory=OptimSpec)
    objective: ObjectiveName = "dino"
    mae: MaeSpec = field(default_factory=MaeSpec)
    seed: int = 0
    num_workers: int = 2
    log_every: int = 20
    ckpt_every: int = 1000
    max_steps: int | None = None
    pretrained: str | None = None


@dataclass
class PreprocessConfig:
    """Config fragment stored in shard manifest (Plan A)."""

    trim: TrimSpec
    norm: NormSpec
    resample: ResampleSpec
    sample: SampleSpec
    seed: int = 0


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    return value


def to_jsonable(obj) -> dict:
    return _jsonable_value(asdict(obj))


def _augment_spec_from_dict(d: dict) -> AugmentSpec:
    arm = d.get("arm", "A1")
    if arm not in _VALID_AUGMENT_ARMS:
        raise ValueError(f"invalid augment arm: {arm!r}")
    return AugmentSpec(
        arm=arm,
        n_global=d.get("n_global", 2),
        n_local=d.get("n_local", 6),
        global_size=d.get("global_size", 128),
        local_size=d.get("local_size", 96),
        global_crop_frac=d.get("global_crop_frac", 0.875),
        local_crop_frac=d.get("local_crop_frac", 0.4),
        poisson_scale=d.get("poisson_scale", 1000.0),
        gain_jitter=d.get("gain_jitter", 0.05),
        energy_shift_px=d.get("energy_shift_px", 2.0),
        dead_pixel_prob=d.get("dead_pixel_prob", 0.01),
        smooth_sigma=d.get("smooth_sigma", 0.5),
        mirror_prob=d.get("mirror_prob", 0.5),
        rrc_scale_min=d.get("rrc_scale_min", 0.2),
        rrc_scale_max=d.get("rrc_scale_max", 1.0),
    )


def _model_spec_from_dict(d: dict) -> ModelSpec:
    name = d.get("name", "vit_s")
    if name not in _VALID_MODEL_NAMES:
        raise ValueError(f"invalid model name: {name!r}")
    return ModelSpec(
        name=name,
        img_size=d.get("img_size", 128),
        patch_size=d.get("patch_size", 16),
        in_chans=d.get("in_chans", 1),
    )


def _dino_spec_from_dict(d: dict) -> DinoSpec:
    return DinoSpec(
        out_dim=d.get("out_dim", 4096),
        hidden_dim=d.get("hidden_dim", 2048),
        bottleneck_dim=d.get("bottleneck_dim", 256),
        student_temp=d.get("student_temp", 0.1),
        teacher_temp_start=d.get("teacher_temp_start", 0.04),
        teacher_temp_end=d.get("teacher_temp_end", 0.07),
        teacher_temp_warmup_epochs=d.get("teacher_temp_warmup_epochs", 30.0),
        center_momentum=d.get("center_momentum", 0.9),
        momentum_teacher=d.get("momentum_teacher", 0.996),
        ibot_mask_ratio=d.get("ibot_mask_ratio", 0.3),
        ibot_weight=d.get("ibot_weight", 1.0),
        koleo_weight=d.get("koleo_weight", 0.1),
        gram_enabled=d.get("gram_enabled", False),
        gram_weight=d.get("gram_weight", 0.0),
    )


def _mae_spec_from_dict(d: dict) -> MaeSpec:
    return MaeSpec(
        decoder_dim=d.get("decoder_dim", 192),
        decoder_depth=d.get("decoder_depth", 4),
        decoder_heads=d.get("decoder_heads", 6),
        mlp_ratio=d.get("mlp_ratio", 4),
        min_lr=d.get("min_lr", 1e-6),
    )


def _optim_spec_from_dict(d: dict) -> OptimSpec:
    return OptimSpec(
        lr=d.get("lr", 0.0005),
        weight_decay_start=d.get("weight_decay_start", 0.04),
        weight_decay_end=d.get("weight_decay_end", 0.4),
        warmup_epochs=d.get("warmup_epochs", 10.0),
        epochs=d.get("epochs", 100),
        batch_size=d.get("batch_size", 64),
        grad_clip=d.get("grad_clip", 3.0),
        use_amp=d.get("use_amp", True),
    )


def run_config_from_dict(d: dict) -> RunConfig:
    augment_d = d.get("augment", {})
    model_d = d.get("model", {})
    dino_d = d.get("dino", {})
    optim_d = d.get("optim", {})
    objective = d.get("objective", "dino")
    if objective not in ("dino", "mae"):
        raise ValueError(f"invalid objective: {objective!r}")
    return RunConfig(
        augment=_augment_spec_from_dict(augment_d),
        model=_model_spec_from_dict(model_d),
        dino=_dino_spec_from_dict(dino_d),
        optim=_optim_spec_from_dict(optim_d),
        objective=objective,
        mae=_mae_spec_from_dict(d.get("mae") or {}),
        seed=d.get("seed", 0),
        num_workers=d.get("num_workers", 2),
        log_every=d.get("log_every", 20),
        ckpt_every=d.get("ckpt_every", 1000),
        max_steps=d.get("max_steps"),
        pretrained=d.get("pretrained"),
    )


def preprocess_config_from_dict(d: dict) -> PreprocessConfig:
    trim_d = d["trim"]
    norm_d = d["norm"]
    resample_d = d["resample"]
    sample_d = d["sample"]

    trim = TrimSpec(
        ranges={role: tuple(bounds) for role, bounds in trim_d["ranges"].items()},
        source_kind=trim_d["source_kind"],
        note=trim_d.get("note", ""),
    )
    norm = NormSpec(
        clip_percentiles=tuple(norm_d.get("clip_percentiles", (1.0, 99.0))),
        scope=norm_d.get("scope", "per_sample"),
        dead_pixel_sigma=norm_d.get("dead_pixel_sigma", 6.0),
        subsample_points=norm_d.get("subsample_points", 512),
    )
    resample = ResampleSpec(
        energy_size=resample_d.get("energy_size", 224),
        slit_size=resample_d.get("slit_size", 224),
        defl_size=resample_d.get("defl_size", 16),
        deg_per_raw_px=resample_d.get("deg_per_raw_px"),
    )
    sample = SampleSpec(
        mode=sample_d["mode"],
        index_roles=tuple(sample_d["index_roles"]),
    )
    return PreprocessConfig(
        trim=trim,
        norm=norm,
        resample=resample,
        sample=sample,
        seed=d.get("seed", 0),
    )
