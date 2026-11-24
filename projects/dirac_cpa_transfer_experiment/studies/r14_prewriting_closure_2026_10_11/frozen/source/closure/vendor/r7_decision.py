from __future__ import annotations

import numpy as np
import pandas as pd

from .periodic_potential_coupled_ltpb_decision import absolute_success
from .r7_config import MODELS, TRAINING_MODES

CORE = ("spinor_rel_l2_gauge_aligned", "current_rel_l2")


def paired_comparison(target, control, *, target_success=False):
    target, control = target.sort_values("seed").set_index("seed"), control.sort_values("seed").set_index("seed")
    rows, summaries = [], {}
    for seed in range(5):
        row = {"seed": seed}
        wins = []
        for key in CORE:
            a, b = float(target.loc[seed, key]), float(control.loc[seed, key])
            row[key+"_difference"] = a-b
            row[key+"_ratio"] = a/b if b else None
            wins.append(a < b)
        row["joint_win"] = all(wins)
        rows.append(row)
    lower_means = []
    for key in CORE:
        a, b = target[key].to_numpy(), control[key].to_numpy()
        summaries[key] = dict(target_mean=float(a.mean()), control_mean=float(b.mean()),
                              ratio_of_means=float(a.mean()/b.mean()) if b.mean() else None,
                              paired_difference_mean=float((a-b).mean()),
                              paired_difference_population_sd=float((a-b).std(ddof=0)))
        lower_means.append(a.mean() < b.mean())
    win_count = sum(r["joint_win"] for r in rows)
    return dict(comparative_advantage=bool(target_success and all(lower_means) and win_count >= 4),
                target_absolute_success=target_success, paired_joint_win_count=win_count,
                per_seed=rows, summary=summaries)


def decide(runs: pd.DataFrame):
    inventory = {(m, mode, s) for m in MODELS for mode in TRAINING_MODES for s in range(5)}
    if len(runs) != 30 or set(zip(runs.model, runs.training_mode, runs.seed)) != inventory:
        raise ValueError("Main decision requires exactly all 30 preregistered runs")
    absolute, comparisons, training_effects = {}, {}, {}
    for mode in TRAINING_MODES:
        for model in MODELS:
            absolute[f"{model}/{mode}"] = absolute_success(runs[(runs.model == model) & (runs.training_mode == mode)])
        release = runs[(runs.model == "phase_release_ltpb") & (runs.training_mode == mode)]
        for control in ("original_ltpb", "complex"):
            comparisons[f"release_vs_{control}/{mode}"] = paired_comparison(
                release, runs[(runs.model == control) & (runs.training_mode == mode)],
                target_success=absolute[f"phase_release_ltpb/{mode}"]["pass"])
    for model in MODELS:
        training_effects[model] = paired_comparison(
            runs[(runs.model == model) & (runs.training_mode == "causal")],
            runs[(runs.model == model) & (runs.training_mode == "ordinary")])
    interaction = {}
    indexed = runs.set_index(["model", "training_mode", "seed"])
    for key in CORE:
        values = []
        for seed in range(5):
            release_effect = indexed.loc[("phase_release_ltpb", "causal", seed), key]-indexed.loc[("phase_release_ltpb", "ordinary", seed), key]
            original_effect = indexed.loc[("original_ltpb", "causal", seed), key]-indexed.loc[("original_ltpb", "ordinary", seed), key]
            values.append(float(release_effect-original_effect))
        interaction[key] = dict(per_seed=values, mean=float(np.mean(values)), population_sd=float(np.std(values, ddof=0)),
                                definition="(release_causal-original_causal)-(release_ordinary-original_ordinary)")
    successful = [mode for mode in TRAINING_MODES if absolute[f"phase_release_ltpb/{mode}"]["pass"]]
    chosen = "ordinary" if "ordinary" in successful else ("causal" if "causal" in successful else None)
    if successful:
        classification = "RELEASE_ABSOLUTE_SUCCESS"
    elif any(absolute[f"complex/{mode}"]["pass"] for mode in TRAINING_MODES):
        classification = "RELEASE_FAILURE_WITH_SUCCESSFUL_COMPLEX_CONTROL"
    else:
        classification = "INCONCLUSIVE_FULL_DYNAMICS"
    return dict(classification=classification, absolute_results=absolute, comparisons=comparisons,
                causal_minus_ordinary=training_effects, representation_training_interaction=interaction,
                long_training_mode_selected=chosen, long_training_triggered_by_main=chosen is not None,
                uncertainty="population SD (ddof=0); five paired seeds; no population-significance assertion",
                local_structure_repair_is_not_full_dynamics_success=True)
