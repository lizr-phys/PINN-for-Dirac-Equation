from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .aggregate import aggregate_results
from .config import CASES, MODELS, PRINCIPAL_MODELS, ExperimentConfig, RunSpec
from .train import train_one_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_directory(results_root: Path, spec: RunSpec) -> Path:
    base = results_root / "raw" / spec.experiment / f"case_{spec.case}" / spec.model
    if spec.experiment == "sensitivity":
        label = f"{spec.lambda_struct:g}".replace(".", "p")
        base = base / f"lambda_{label}"
    return base / f"seed_{spec.seed}"


def build_specs(mode: str, cfg: ExperimentConfig) -> list[RunSpec]:
    if mode == "smoke":
        return [
            RunSpec("smoke", "M", model, 0, 1.0, 200) for model in MODELS
        ]
    if mode == "main":
        specs = [
            RunSpec("main", case, model, seed, 1.0, cfg.adam_steps)
            for case in CASES
            for model in PRINCIPAL_MODELS
            for seed in (0, 1, 2)
        ]
        specs.extend(
            RunSpec("main", case, "two_branch", 0, 1.0, cfg.adam_steps)
            for case in CASES
        )
        return specs
    if mode == "sensitivity":
        return [
            RunSpec("sensitivity", "M", "sp_cpa_conservative", 0, value, cfg.adam_steps)
            for value in (0.1, 10.0)
        ]
    if mode == "all":
        return build_specs("smoke", cfg) + build_specs("main", cfg) + build_specs(
            "sensitivity", cfg
        )
    if mode == "aggregate":
        return []
    raise ValueError(mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("smoke", "main", "sensitivity", "all", "aggregate"), required=True
    )
    parser.add_argument("--only-case", choices=tuple(CASES))
    parser.add_argument("--only-model", choices=MODELS)
    parser.add_argument("--only-seed", type=int)
    parser.add_argument("--only-lambda", type=float)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=ExperimentConfig.cpu_threads)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cfg = ExperimentConfig(cpu_threads=args.cpu_threads)
    specs = build_specs(args.mode, cfg)
    if args.only_case is not None:
        specs = [spec for spec in specs if spec.case == args.only_case]
    if args.only_model is not None:
        specs = [spec for spec in specs if spec.model == args.only_model]
    if args.only_seed is not None:
        specs = [spec for spec in specs if spec.seed == args.only_seed]
    if args.only_lambda is not None:
        specs = [
            spec for spec in specs if abs(spec.lambda_struct - args.only_lambda) < 1.0e-12
        ]
    results_root = PROJECT_ROOT / "results"
    for spec in specs:
        train_one_run(
            cfg,
            spec,
            run_directory(results_root, spec),
            force=args.force,
        )
    if not args.no_aggregate:
        formal, aggregated = aggregate_results(PROJECT_ROOT)
        print(
            f"Aggregated {len(formal)} formal rows and {len(aggregated)} case/model groups.",
            flush=True,
        )


if __name__ == "__main__":
    main()
