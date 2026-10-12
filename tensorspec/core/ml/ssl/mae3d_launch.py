"""torchrun entry: ``python -m tensorspec.core.ml.ssl.mae3d_launch``."""

from __future__ import annotations

import argparse

from tensorspec.core.ml.ssl.mae3d import train_mae3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 00742 3D MAE")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train_mae3d(
        data_dir=args.data,
        out_dir=args.out,
        init_checkpoint=args.init,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
