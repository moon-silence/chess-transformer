"""Train one model.

    python scripts/train.py --vocab 512
    python scripts/train.py --vocab 1024 --max_steps 2000 --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from chesslm.train import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--device", default=None)
    for f in fields(TrainConfig):
        ap.add_argument(f"--{f.name}", type=type(f.default), default=None)
    args = ap.parse_args()

    overrides = {f.name: getattr(args, f.name) for f in fields(TrainConfig)}
    cfg = TrainConfig.from_yaml(args.config, **overrides)
    train(cfg, f"{args.data_dir}/v{args.vocab}", f"{args.runs_dir}/v{args.vocab}", args.device)


if __name__ == "__main__":
    main()
