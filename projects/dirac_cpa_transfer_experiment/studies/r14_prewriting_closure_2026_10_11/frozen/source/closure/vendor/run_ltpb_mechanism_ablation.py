from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch

from .config import ExperimentConfig, RunSpec, write_json
from .ltpb_mechanism_decision import (
    ABLATION_MODELS,
    ABLATION_SEEDS,
    compute_mechanism_decision,
)
from .train import train_one_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
EXPERIMENT = "ltpb_mechanism_ablation"
SMOKE_EXPERIMENT = "ltpb_mechanism_ablation_smoke"
CASE = "M"


def ablation_config(cpu_threads: int) -> ExperimentConfig:
    return ExperimentConfig(
        x_min=0.0,
        x_max=2.0 * math.pi,
        cpu_threads=cpu_threads,
    )


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
        return [
            RunSpec(SMOKE_EXPERIMENT, CASE, model, 0, 1.0, smoke_steps)
            for model in ABLATION_MODELS
        ]
    if mode == "full":
        return [
            RunSpec(EXPERIMENT, CASE, model, seed, 1.0, cfg.adam_steps)
            for model in ABLATION_MODELS
            for seed in ABLATION_SEEDS
        ]
    if mode == "aggregate":
        return []
    raise ValueError(mode)


def _load_formal_runs() -> pd.DataFrame:
    rows: list[dict] = []
    raw_root = RESULTS_ROOT / "raw" / EXPERIMENT
    for metrics_path in sorted(raw_root.rglob("metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        if row.get("experiment") != EXPERIMENT:
            continue
        row["metrics_path"] = str(metrics_path.resolve())
        rows.append(row)
    runs = pd.DataFrame(rows)
    expected = {(model, seed) for model in ABLATION_MODELS for seed in ABLATION_SEEDS}
    if len(runs) != len(expected):
        raise RuntimeError(
            "Mechanism aggregation requires exactly 15 formal rows; "
            f"found={len(runs)}"
        )
    duplicated = runs.duplicated(subset=["model", "seed"], keep=False)
    if bool(duplicated.any()):
        duplicates = runs.loc[duplicated, ["model", "seed"]].to_dict("records")
        raise RuntimeError(f"Duplicate formal model/seed rows: {duplicates}")
    found = (
        set(zip(runs["model"], runs["seed"], strict=False))
        if not runs.empty
        else set()
    )
    if found != expected:
        raise RuntimeError(
            "Mechanism aggregation requires exactly 15 formal runs; "
            f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
        )
    if set(runs["case"]) != {CASE}:
        raise RuntimeError(f"Formal mechanism runs must all be Case {CASE}")
    if set(runs["steps"].astype(int)) != {5000}:
        raise RuntimeError("Formal mechanism runs must all have 5000 steps")
    return runs.sort_values(["model", "seed"]).reset_index(drop=True)


def aggregate_ablation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    runs = _load_formal_runs()
    runs_path = RESULTS_ROOT / "ltpb_mechanism_ablation_runs.csv"
    runs.to_csv(runs_path, index=False)

    excluded = {"seed", "eta", "lambda_struct", "steps"}
    numeric_columns = [
        column
        for column in runs.select_dtypes(include="number").columns
        if column not in excluded
    ]
    grouped = runs.groupby(["case", "model"], sort=False)
    mean = grouped[numeric_columns].mean().add_suffix("_mean")
    std = grouped[numeric_columns].std(ddof=0).fillna(0.0).add_suffix("_std")
    count = grouped.size().rename("n_seeds")
    summary = pd.concat((count, mean, std), axis=1).reset_index()
    order = {model: index for index, model in enumerate(ABLATION_MODELS)}
    summary["_order"] = summary["model"].map(order)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    summary_path = RESULTS_ROOT / "ltpb_mechanism_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    comparison_rows: list[dict[str, float | str | int]] = []
    ltpb_row = summary[summary["model"] == "ltpb_cpa"].iloc[0]
    for _, row in summary.iterrows():
        model = str(row["model"])
        for column in numeric_columns:
            value = float(row[f"{column}_mean"])
            reference = float(ltpb_row[f"{column}_mean"])
            comparison_rows.append(
                {
                    "case": CASE,
                    "model": model,
                    "metric": column,
                    "n_seeds": int(row["n_seeds"]),
                    "mean": value,
                    "population_sd": float(row[f"{column}_std"]),
                    "ratio_to_ltpb": (
                        value / reference
                        if reference != 0.0
                        else (0.0 if value == 0.0 else float("inf"))
                    ),
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(
        RESULTS_ROOT / "ltpb_mechanism_ablation_comparisons.csv", index=False
    )

    decision = compute_mechanism_decision(runs, summary)
    write_json(RESULTS_ROOT / "ltpb_mechanism_ablation_decision.json", decision)
    return runs, summary, comparisons, decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full", "aggregate"), required=True)
    parser.add_argument("--only-model", choices=ABLATION_MODELS)
    parser.add_argument("--only-seed", type=int, choices=ABLATION_SEEDS)
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
    cfg = ablation_config(args.cpu_threads)
    specs = build_specs(args.mode, cfg, args.smoke_steps)
    if args.only_model is not None:
        specs = [spec for spec in specs if spec.model == args.only_model]
    if args.only_seed is not None:
        specs = [spec for spec in specs if spec.seed == args.only_seed]
    for spec in specs:
        train_one_run(cfg, spec, run_directory(spec), force=args.force)
    if args.mode == "aggregate" or (args.mode == "full" and not args.no_aggregate):
        runs, summary, comparisons, decision = aggregate_ablation()
        print(
            f"Mechanism aggregation: {len(runs)} runs, {len(summary)} models, "
            f"{len(comparisons)} comparison rows, "
            f"strong_support={decision['strong_mechanism_support']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
