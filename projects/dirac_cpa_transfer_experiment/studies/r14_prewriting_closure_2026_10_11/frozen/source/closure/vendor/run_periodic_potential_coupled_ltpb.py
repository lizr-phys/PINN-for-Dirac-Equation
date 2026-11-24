from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .periodic_potential_config import PERIODIC_POTENTIAL_MODELS, PERIODIC_POTENTIAL_SEEDS, PeriodicPotentialConfig
from .train_periodic_potential_coupled_ltpb import ROOT, RAW, load_formal_config, train_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--model", choices=PERIODIC_POTENTIAL_MODELS, action="append")
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args()
    if args.mode == "smoke":
        cfg = replace(PeriodicPotentialConfig(), adam_steps=20, test_x_points=64, test_t_points=21,
                      history_every=5, log_every=20)
        base = ROOT/"results/raw/periodic_potential_coupled_ltpb_smoke"
        seeds = args.seed or (0,)
    else:
        cfg = load_formal_config()
        base = RAW/"learned"
        seeds = args.seed or PERIODIC_POTENTIAL_SEEDS
    for seed in seeds:
        for model in args.model or PERIODIC_POTENTIAL_MODELS:
            train_run(cfg, model, seed, base/model/f"seed_{seed}", formal=args.mode == "full")


if __name__ == "__main__":
    main()
