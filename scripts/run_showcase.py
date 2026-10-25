"""Run frozen subject-isolated experiments; outputs always use a fresh directory."""

import argparse
from pathlib import Path

import yaml

from ppg_hr.showcase_experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/showcase.yaml"))
    parser.add_argument("--splits", type=Path, default=Path("splits/loso_v1.json"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds) or any(x < 0 or x >= 15 for x in args.folds):
        parser.error("folds must be unique indices from 0 to 14")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(
        run_experiment(
            args.cache, config, args.splits, args.output_root, args.folds, device=args.device
        )
    )


if __name__ == "__main__":
    main()
