from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .config import CASES, ExperimentConfig, RunSpec
from .train import train_one_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
PILOT_EXPERIMENT = "learned_two_phase_pilot"
SMOKE_EXPERIMENT = "learned_two_phase_pilot_smoke"
PILOT_MODEL = "ltpb_cpa"
PILOT_SEEDS = (0, 1, 2)


def run_directory(spec: RunSpec) -> Path:
    return (
        RESULTS_ROOT
        / "raw"
        / spec.experiment
        / f"case_{spec.case}"
        / spec.model
        / f"seed_{spec.seed}"
    )


def build_specs(mode: str, cfg: ExperimentConfig, smoke_steps: int) -> list[RunSpec]:
    if mode == "smoke":
        return [RunSpec(SMOKE_EXPERIMENT, "M", PILOT_MODEL, 0, 1.0, smoke_steps)]
    if mode == "full":
        return [
            RunSpec(PILOT_EXPERIMENT, case, PILOT_MODEL, seed, 1.0, cfg.adam_steps)
            for case in CASES
            for seed in PILOT_SEEDS
        ]
    if mode == "aggregate":
        return []
    raise ValueError(mode)


def aggregate_pilot() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    raw_root = RESULTS_ROOT / "raw" / PILOT_EXPERIMENT
    for metrics_path in sorted(raw_root.rglob("metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        if row.get("experiment") != PILOT_EXPERIMENT:
            continue
        row["metrics_path"] = str(metrics_path.resolve())
        rows.append(row)
    runs = pd.DataFrame(rows)
    expected = {(case, seed) for case in CASES for seed in PILOT_SEEDS}
    found = (
        set(zip(runs["case"], runs["seed"], strict=False))
        if not runs.empty
        else set()
    )
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise RuntimeError(
            f"Pilot aggregation requires exactly six formal runs; missing={missing}, extra={extra}"
        )
    runs = runs.sort_values(["case", "seed"]).reset_index(drop=True)
    runs.to_csv(RESULTS_ROOT / "learned_two_phase_pilot_runs.csv", index=False)

    excluded = {"seed", "eta", "lambda_struct", "steps"}
    numeric_columns = [
        column
        for column in runs.select_dtypes(include="number").columns
        if column not in excluded
    ]
    grouped = runs.groupby(["case", "model"], sort=True)
    mean = grouped[numeric_columns].mean().add_suffix("_mean")
    std = grouped[numeric_columns].std(ddof=0).fillna(0.0).add_suffix("_std")
    count = grouped.size().rename("n_seeds")
    aggregate = pd.concat((count, mean, std), axis=1).reset_index()
    aggregate.to_csv(
        RESULTS_ROOT / "learned_two_phase_pilot_summary.csv", index=False
    )

    baseline_path = RESULTS_ROOT / "summary.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing reusable baseline summary: {baseline_path}")
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[
        (baseline["experiment"] == "main")
        & baseline["model"].isin(
            ["vanilla", "sp_cpa", "sp_cpa_conservative", "two_branch"]
        )
    ].copy()
    comparison = pd.concat((baseline, runs), ignore_index=True, sort=False)
    comparison = comparison.sort_values(["case", "model", "seed"]).reset_index(drop=True)
    comparison.to_csv(
        RESULTS_ROOT / "learned_two_phase_pilot_comparison.csv", index=False
    )
    return runs, aggregate, comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full", "aggregate"), required=True)
    parser.add_argument("--only-case", choices=tuple(CASES))
    parser.add_argument("--only-seed", type=int, choices=PILOT_SEEDS)
    parser.add_argument("--smoke-steps", type=int, default=20)
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
    specs = build_specs(args.mode, cfg, args.smoke_steps)
    if args.only_case is not None:
        specs = [spec for spec in specs if spec.case == args.only_case]
    if args.only_seed is not None:
        specs = [spec for spec in specs if spec.seed == args.only_seed]
    for spec in specs:
        train_one_run(cfg, spec, run_directory(spec), force=args.force)
    if args.mode == "aggregate" or (
        args.mode == "full" and not args.no_aggregate
    ):
        runs, aggregate, comparison = aggregate_pilot()
        print(
            f"Pilot aggregation: {len(runs)} runs, {len(aggregate)} groups, "
            f"{len(comparison)} comparison rows.",
            flush=True,
        )


if __name__ == "__main__":
    main()
