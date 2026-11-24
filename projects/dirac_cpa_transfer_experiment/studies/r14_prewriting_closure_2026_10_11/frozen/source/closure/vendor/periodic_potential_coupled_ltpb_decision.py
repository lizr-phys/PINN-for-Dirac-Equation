from __future__ import annotations

import numpy as np
import pandas as pd

from .periodic_potential_config import PERIODIC_POTENTIAL_MODELS

TARGET = "coupled_shared_spectral_ltpb"
CONTROL = "coupled_shared_spectral_complex"
THRESHOLDS = {
    "spinor_rel_l2_gauge_aligned": ("<=", 0.10),
    "current_rel_l2": ("<=", 0.15),
    "fidelity_mean": (">=", 0.98),
    "modal_population_rel_l2": ("<=", 0.10),
    "max_norm_drift": ("<=", 0.02),
}


def absolute_success(frame: pd.DataFrame) -> dict:
    rows = frame.sort_values("seed")
    if rows.seed.tolist() != [0, 1, 2, 3, 4]:
        raise ValueError("Exactly five distinct registered seeds are required")
    checks, per_seed = {}, np.ones(5, dtype=bool)
    for metric, (operator, boundary) in THRESHOLDS.items():
        values = rows[metric].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite primary metric: {metric}")
        mean = float(np.mean(values))
        test = values <= boundary if operator == "<=" else values >= boundary
        mean_pass = mean <= boundary if operator == "<=" else mean >= boundary
        checks[metric] = {"mean": mean, "population_sd": float(np.std(values, ddof=0)),
                          "operator": operator, "threshold": boundary, "pass": bool(mean_pass)}
        per_seed &= test
    return {"pass": all(x["pass"] for x in checks.values()) and int(per_seed.sum()) >= 4,
            "mean_checks": checks, "joint_absolute_pass_count": int(per_seed.sum()),
            "per_seed_joint_pass": [{"seed": i, "pass": bool(p)} for i, p in enumerate(per_seed)]}


def evaluate_periodic_potential_decision(runs: pd.DataFrame) -> dict:
    expected = {(m, s) for m in PERIODIC_POTENTIAL_MODELS for s in range(5)}
    if len(runs) != 15 or set(zip(runs.model, runs.seed)) != expected:
        raise ValueError("Decision requires the complete 15-run inventory")
    target = runs[runs.model == TARGET].sort_values("seed").set_index("seed")
    control = runs[runs.model == CONTROL].sort_values("seed").set_index("seed")
    absolute = {m: absolute_success(runs[runs.model == m]) for m in PERIODIC_POTENTIAL_MODELS}
    paired, ratios = [], {}
    for seed in range(5):
        entry = {"seed": seed}
        wins = []
        for metric in ("spinor_rel_l2_gauge_aligned", "current_rel_l2"):
            a, b = float(target.loc[seed, metric]), float(control.loc[seed, metric])
            entry[f"{metric}_ltpb_minus_complex"] = a-b
            entry[f"{metric}_ltpb_over_complex"] = a/b if b else None
            wins.append(a < b)
        entry["joint_win"] = all(wins)
        paired.append(entry)
    mean_wins = []
    for metric in ("spinor_rel_l2_gauge_aligned", "current_rel_l2"):
        a, b = float(target[metric].mean()), float(control[metric].mean())
        differences = target[metric].to_numpy()-control[metric].to_numpy()
        ratios[metric] = {"ltpb_mean": a, "complex_mean": b,
                          "ltpb_over_complex": a/b if b else None,
                          "complex_over_ltpb": b/a if a else None,
                          "paired_difference_mean": float(np.mean(differences)),
                          "paired_difference_population_sd": float(np.std(differences, ddof=0))}
        mean_wins.append(a < b)
    wins = sum(p["joint_win"] for p in paired)
    transfer = absolute[TARGET]["pass"]
    advantage = transfer and all(mean_wins) and wins >= 4
    if advantage:
        classification = "COMPARATIVE_ADVANTAGE"
    elif transfer:
        classification = "TRANSFER_SUCCESS"
    elif absolute[CONTROL]["pass"]:
        classification = "REPRESENTATION_FAILURE"
    else:
        classification = "INCONCLUSIVE"
    return {"classification": classification, "transfer_success": transfer,
            "comparative_advantage": advantage, "absolute_results": absolute,
            "paired_joint_win_count": wins, "paired_seed_comparisons": paired,
            "mean_ratios_and_paired_uncertainty": ratios,
            "uncertainty_definition": "Population SD across the five paired seed differences; no significance claim",
            "rescue_models_trained": False}
