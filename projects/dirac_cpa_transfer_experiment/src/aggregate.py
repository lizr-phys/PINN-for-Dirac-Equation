from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


IDENTITY_COLUMNS = [
    "experiment",
    "case",
    "model",
    "seed",
    "lambda_struct",
    "steps",
]


def aggregate_results(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_root = project_root / "results" / "raw"
    rows: list[dict] = []
    for metrics_path in sorted(raw_root.rglob("metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["metrics_path"] = str(metrics_path.resolve())
        rows.append(row)
    all_runs = pd.DataFrame(rows)
    results_root = project_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    if all_runs.empty:
        all_runs.to_csv(results_root / "summary.csv", index=False)
        return all_runs, pd.DataFrame()

    smoke = all_runs[all_runs["experiment"] == "smoke"].copy()
    smoke.to_csv(results_root / "smoke_summary.csv", index=False)
    # Keep the original experiment's official summaries isolated from later
    # extension studies stored under results/raw/.  Each extension owns a
    # dedicated aggregator and must not silently enter summary.csv.
    formal = all_runs[all_runs["experiment"].isin(("main", "sensitivity"))].copy()

    main_lambda_one = formal[
        (formal["experiment"] == "main")
        & (formal["case"] == "M")
        & (formal["model"] == "sp_cpa_conservative")
        & (formal["seed"] == 0)
        & (formal["lambda_struct"] == 1.0)
    ]
    has_sensitivity_one = not formal[
        (formal["experiment"] == "sensitivity")
        & (formal["lambda_struct"] == 1.0)
    ].empty
    if not main_lambda_one.empty and not has_sensitivity_one:
        reused = main_lambda_one.copy()
        reused["experiment"] = "sensitivity"
        reused["metrics_path"] = reused["metrics_path"] + " (reused lambda=1 main run)"
        formal = pd.concat((formal, reused), ignore_index=True)

    formal = formal.sort_values(
        ["experiment", "case", "model", "lambda_struct", "seed"]
    ).reset_index(drop=True)
    formal.to_csv(results_root / "summary.csv", index=False)

    main = formal[formal["experiment"] == "main"].copy()
    numeric_metrics = [
        column
        for column in main.select_dtypes(include="number").columns
        if column not in {"seed", "eta", "lambda_struct", "steps"}
    ]
    if main.empty:
        aggregated = pd.DataFrame()
    else:
        grouped = main.groupby(["case", "model"], sort=True)
        mean = grouped[numeric_metrics].mean().add_suffix("_mean")
        std = grouped[numeric_metrics].std(ddof=0).fillna(0.0).add_suffix("_std")
        count = grouped.size().rename("n_seeds")
        aggregated = pd.concat((count, mean, std), axis=1).reset_index()
    aggregated.to_csv(results_root / "summary_aggregated.csv", index=False)
    return formal, aggregated
