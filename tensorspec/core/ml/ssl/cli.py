"""Command-line entry point for SSL dataset preprocessing."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

from tensorspec.core.ml.ssl.preprocess import preprocess_file
from tensorspec.core.ml.ssl.probe import ProbeConfig, probe
from tensorspec.core.ml.ssl.spec import (
    NormSpec,
    PreprocessConfig,
    ResampleSpec,
    RunConfig,
    SampleSpec,
    TrimSpec,
    preprocess_config_from_dict,
    run_config_from_dict,
)
from tensorspec.core.ml.ssl.train import train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tensorspec.core.ml.ssl.cli")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    preprocess = subparsers.add_parser(
        "preprocess", help="convert one Maestro file into SSL shards"
    )
    preprocess.add_argument("--input", required=True)
    preprocess.add_argument("--out", required=True)
    preprocess.add_argument(
        "--mode", choices=["disp2d", "fermi3d"], required=True
    )
    preprocess.add_argument(
        "--config", help="JSON PreprocessConfig; defaults used if omitted"
    )
    preprocess.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing shard dataset in --out",
    )
    preprocess.add_argument(
        "--append",
        action="store_true",
        help="append samples into an existing dataset in --out",
    )
    preprocess.add_argument(
        "--source-id",
        help="manifest source id (default: input basename)",
    )
    train_parser = subparsers.add_parser(
        "train", help="run stage-1 DINO training from shard dataset"
    )
    train_parser.add_argument(
        "--config", required=True, help="JSON RunConfig path"
    )
    train_parser.add_argument(
        "--data", required=True, help="shard dataset directory"
    )
    train_parser.add_argument(
        "--out", required=True, help="training output directory"
    )
    train_parser.add_argument(
        "--resume",
        help="checkpoint path to resume from",
    )
    train_parser.add_argument(
        "--pretrained",
        help="override RunConfig.pretrained (e.g. dinov2_vits14)",
    )
    probe_parser = subparsers.add_parser(
        "probe", help="floor metrics from ckpt + reference map"
    )
    probe_parser.add_argument("--ckpt", required=True)
    probe_parser.add_argument("--data", required=True)
    probe_parser.add_argument("--reference", required=True)
    probe_parser.add_argument("--out", required=True)
    probe_parser.add_argument("--source-id", required=True)
    probe_parser.add_argument("--k", type=int, default=2)
    probe_parser.add_argument("--seed", type=int, default=0)
    probe_parser.add_argument("--batch-size", type=int, default=64)
    probe_parser.add_argument(
        "--student",
        action="store_true",
        help="use student CLS instead of teacher",
    )
    probe_parser.add_argument(
        "--roi-mask",
        action="store_true",
        help="zero outside Energy/Angle ROI from reference; requires --axes",
    )
    probe_parser.add_argument(
        "--axes",
        help="npz with energy_axis/slit_axis for ROI mapping (roi-mask mode)",
    )
    probe_parser.add_argument(
        "--embed",
        choices=["cls", "patch_mean"],
        default="cls",
        help="CLS token or mean of all patch tokens",
    )
    probe_parser.add_argument(
        "--pca-dim",
        type=int,
        default=50,
        help="PCA dims before k-means; <=0 skips PCA (full embedding)",
    )
    return parser


def _run_config(path: str) -> RunConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return run_config_from_dict(payload)


def _config(path: str | None, mode: str) -> PreprocessConfig:
    if path is None:
        config = PreprocessConfig(
            trim=TrimSpec(ranges={}, source_kind=""),
            norm=NormSpec(),
            resample=ResampleSpec(),
            sample=SampleSpec(mode=mode, index_roles=()),
        )
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = preprocess_config_from_dict(payload)
    return replace(config, sample=replace(config.sample, mode=mode))


def _progress_printer(label: str):
    started = time.monotonic()
    last_print = 0.0

    def progress(current: int, total: int) -> None:
        nonlocal last_print
        now = time.monotonic()
        done = current >= total
        if not done and now - last_print < 5.0 and current % 500 != 0:
            return
        last_print = now
        elapsed = now - started
        rate = current / elapsed if elapsed > 0 else 0.0
        print(
            f"{label}: {current}/{total} ({100.0 * current / max(total, 1):.1f}%) "
            f"{rate:.1f} samp/s",
            file=sys.stderr,
            flush=True,
        )

    return progress


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "preprocess":
        if args.overwrite and args.append:
            raise SystemExit("error: --overwrite and --append are mutually exclusive")
        label = Path(args.input).name
        manifest = preprocess_file(
            args.input,
            args.out,
            _config(args.config, args.mode),
            source_id=args.source_id,
            overwrite=args.overwrite,
            append=args.append,
            progress=_progress_printer(label),
        )
        print(
            f"done {label}: total_samples={manifest['total_samples']} "
            f"sources={len(manifest['sources'])}",
            file=sys.stderr,
            flush=True,
        )
        return 0
    if args.cmd == "train":
        cfg = _run_config(args.config)
        if args.pretrained:
            cfg = replace(cfg, pretrained=args.pretrained)
        summary = train(
            cfg,
            args.data,
            args.out,
            resume=args.resume,
        )
        print(
            f"done train: steps={summary['steps']} epoch={summary['epoch']} "
            f"last_loss={summary['last_loss']}",
            file=sys.stderr,
            flush=True,
        )
        return 0
    if args.cmd == "probe":
        config = ProbeConfig(
            source_id=args.source_id,
            k=args.k,
            seed=args.seed,
            batch_size=args.batch_size,
            use_teacher=not args.student,
            roi_mode="mask" if args.roi_mask else "full",
            embed=args.embed,
            pca_dim=args.pca_dim,
        )
        metrics = probe(
            ckpt=args.ckpt,
            data_dir=args.data,
            reference=args.reference,
            out_dir=args.out,
            config=config,
            axes_path=args.axes,
        )
        iou = metrics["iou"]
        iou_s = f"{iou:.4f}" if iou == iou else "nan"
        print(
            f"done probe: n_samples={metrics['n_samples']} "
            f"embed={metrics.get('embed')} "
            f"roi_mode={metrics.get('roi_mode')} "
            f"ari={metrics['ari']:.4f} nmi={metrics['nmi']:.4f} "
            f"iou={iou_s} contiguity={metrics['contiguity']:.4f}",
            file=sys.stderr,
            flush=True,
        )
        return 0
    raise AssertionError(f"unhandled command {args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
