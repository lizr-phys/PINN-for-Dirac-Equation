from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .freeze_r7_phase_release import RAW, write_json
from .periodic_potential_coupled_ltpb_decision import absolute_success
from .r7_decision import decide, CORE


def stage_runs(stage):
    paths = sorted((RAW/stage).glob("*/*/seed_*/metrics.json"))
    return pd.DataFrame([json.loads(path.read_text(encoding="utf-8")) for path in paths])


def summary_table(runs):
    rows = []
    identifiers = ["stage", "model", "training_mode"]
    numeric = runs.select_dtypes(include="number").columns.difference(["seed"])
    for group, frame in runs.groupby(identifiers):
        row = dict(zip(identifiers, group))
        row["n_seeds"] = len(frame)
        for key in numeric:
            values = frame[key].to_numpy(dtype=float)
            row[key+"_mean"] = float(np.mean(values))
            row[key+"_population_sd"] = float(np.std(values, ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate():
    combined = []
    for stage in ("short", "main", "long"):
        runs = stage_runs(stage)
        if runs.empty:
            continue
        runs.to_csv(RAW/f"{stage}_runs.csv", index=False)
        summary_table(runs).to_csv(RAW/f"{stage}_summary.csv", index=False)
        combined.append(runs)
    if not combined:
        raise ValueError("No completed experimental runs")
    pd.concat(combined, ignore_index=True).to_csv(RAW/"all_runs.csv", index=False)
    main = stage_runs("main")
    if len(main) == 30:
        result = decide(main)
        write_json(RAW/"decision.json", result)
        paired_rows = []
        for comparison, values in result["comparisons"].items():
            paired_rows += [{"comparison": comparison, **row} for row in values["per_seed"]]
        for model, values in result["causal_minus_ordinary"].items():
            paired_rows += [{"comparison": f"causal_minus_ordinary/{model}", **row} for row in values["per_seed"]]
        pd.DataFrame(paired_rows).to_csv(RAW/"paired_comparisons.csv", index=False)
        effects = []
        for comparison, values in result["comparisons"].items():
            for key, row in values["summary"].items():
                effects.append(dict(comparison=comparison, metric=key, **row))
        for model, values in result["causal_minus_ordinary"].items():
            for key, row in values["summary"].items():
                effects.append(dict(comparison=f"causal_minus_ordinary/{model}", metric=key, **row))
        pd.DataFrame(effects).to_csv(RAW/"paired_summary.csv", index=False)
        interaction = []
        for key, value in result["representation_training_interaction"].items():
            interaction += [dict(metric=key, seed=s, interaction=entry) for s, entry in enumerate(value["per_seed"])]
        pd.DataFrame(interaction).to_csv(RAW/"factor_interactions.csv", index=False)
        checks = []
        for path in sorted((RAW/"main").glob("*/*/seed_*/metrics.json")):
            base = json.loads(path.read_text(encoding="utf-8"))
            for step in (1000, 2500, 5000):
                target = path if step == 5000 else path.parent/"checkpoints"/f"step_{step}"/"metrics.json"
                metrics = json.loads(target.read_text(encoding="utf-8"))
                checks.append(dict(stage="main", model=base["model"], training_mode=base["training_mode"],
                                   seed=base["seed"], step=step, **{key: metrics[key] for key in CORE+("fidelity_mean",)}))
        pd.DataFrame(checks).to_csv(RAW/"checkpoint_metrics.csv", index=False)
        print(result["classification"], flush=True)
    long = stage_runs("long")
    if len(long) == 15:
        absolute = {model: absolute_success(frame) for model, frame in long.groupby("model")}
        write_json(RAW/"long_decision.json", dict(absolute_results=absolute,
                   release_absolute_success=absolute["phase_release_ltpb"]["pass"],
                   is_time_extrapolation=False))
    return main


if __name__ == "__main__":
    aggregate()
