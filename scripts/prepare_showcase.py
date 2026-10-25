"""Prepare a new local engineering cache, preserving strict audit artifacts."""

import argparse
from pathlib import Path

import yaml

from ppg_hr.showcase_data import prepare_engineering


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--config", type=Path, default=Path("configs/showcase.yaml"))
    parser.add_argument("--trust-local-pickle", action="store_true", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(
        prepare_engineering(
            args.data_root, args.cache_root, config, trust_local_pickle=args.trust_local_pickle
        )
    )


if __name__ == "__main__":
    main()
