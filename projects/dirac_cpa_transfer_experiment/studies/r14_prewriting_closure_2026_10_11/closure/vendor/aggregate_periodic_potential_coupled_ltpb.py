from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .periodic_potential_config import PERIODIC_POTENTIAL_MODELS, PERIODIC_POTENTIAL_SEEDS
from .periodic_potential_coupled_ltpb_decision import evaluate_periodic_potential_decision
from .train_periodic_potential_coupled_ltpb import RAW, write_json


def aggregate() -> dict:
    rows = []
    for model in PERIODIC_POTENTIAL_MODELS:
        for seed in PERIODIC_POTENTIAL_SEEDS:
            path = RAW/"learned"/model/f"seed_{seed}"/"metrics.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["steps"] != 5000 or not record["formal"]:
                raise ValueError(f"Nonformal/incomplete run: {path}")
            rows.append(record)
    runs = pd.DataFrame(rows)
    runs.to_csv(RAW/"runs.csv", index=False)
    numeric = runs.select_dtypes(include=[np.number]).columns.difference(["seed", "steps"])
    summary = []
    for model in PERIODIC_POTENTIAL_MODELS:
        data = runs[runs.model == model]
        row = {"model": model, "n_seeds": 5}
        for metric in numeric:
            v = data[metric].dropna().to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(v)) if len(v) else np.nan
            row[f"{metric}_population_sd"] = float(np.std(v, ddof=0)) if len(v) else np.nan
        summary.append(row)
    pd.DataFrame(summary).to_csv(RAW/"summary.csv", index=False)
    decision = evaluate_periodic_potential_decision(runs)
    write_json(RAW/"decision.json", decision)
    pd.DataFrame(decision["paired_seed_comparisons"]).to_csv(RAW/"paired_comparisons.csv", index=False)
    pd.DataFrame([{"metric": k, **v} for k, v in decision["mean_ratios_and_paired_uncertainty"].items()]).to_csv(RAW/"comparison_summary.csv", index=False)
    return decision


if __name__ == "__main__":
    print(json.dumps(aggregate(), indent=2))
