from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .multimode_config import MODE_KEYS


BASELINE_MODELS = (
    "vanilla",
    "sp_cpa",
    "component_polar",
    "multi_branch_complex",
)
TARGET_MODEL = "multi_ltpb"


def _finite_mean(values: pd.Series) -> float:
    array = values.to_numpy(dtype=np.float64)
    return float(np.mean(array)) if np.all(np.isfinite(array)) else math.inf


def evaluate_preregistered_decision(runs: pd.DataFrame) -> dict[str, Any]:
    """Apply the frozen Multi-LTPB success rule without post-hoc choices."""

    required = {
        "model",
        "seed",
        "spinor_rel_l2_gauge_aligned",
        "current_rel_l2",
        "fidelity_mean",
    }
    missing = sorted(required.difference(runs.columns))
    if missing:
        raise ValueError(f"Decision input is missing columns: {missing}")
    if len(runs) != 25:
        raise ValueError(f"Expected exactly 25 learned runs, found {len(runs)}")
    expected_pairs = {(model, seed) for model in (*BASELINE_MODELS, TARGET_MODEL) for seed in range(5)}
    actual_pairs = {(str(row.model), int(row.seed)) for row in runs.itertuples()}
    if actual_pairs != expected_pairs:
        missing_pairs = sorted(expected_pairs - actual_pairs)
        extra_pairs = sorted(actual_pairs - expected_pairs)
        raise ValueError(f"Run inventory mismatch; missing={missing_pairs}, extra={extra_pairs}")

    means = runs.groupby("model", sort=False).mean(numeric_only=True)
    target = means.loc[TARGET_MODEL]
    best_spinor_model = min(
        BASELINE_MODELS,
        key=lambda model: float(means.loc[model, "spinor_rel_l2_gauge_aligned"]),
    )
    best_current_model = min(
        BASELINE_MODELS,
        key=lambda model: float(means.loc[model, "current_rel_l2"]),
    )
    target_spinor = float(target["spinor_rel_l2_gauge_aligned"])
    target_current = float(target["current_rel_l2"])
    target_fidelity = float(target["fidelity_mean"])
    best_spinor = float(means.loc[best_spinor_model, "spinor_rel_l2_gauge_aligned"])
    best_current = float(means.loc[best_current_model, "current_rel_l2"])

    absolute = {
        "spinor_mean_le_0p05": {
            "value": target_spinor,
            "threshold": 0.05,
            "operator": "<=",
            "pass": target_spinor <= 0.05,
        },
        "current_mean_le_0p10": {
            "value": target_current,
            "threshold": 0.10,
            "operator": "<=",
            "pass": target_current <= 0.10,
        },
        "fidelity_mean_ge_0p99": {
            "value": target_fidelity,
            "threshold": 0.99,
            "operator": ">=",
            "pass": target_fidelity >= 0.99,
        },
    }
    advantages = {
        "spinor": {
            "best_baseline": best_spinor_model,
            "best_baseline_mean": best_spinor,
            "multi_ltpb_mean": target_spinor,
            "improvement_factor": best_spinor / target_spinor if target_spinor > 0 else math.inf,
            "required_factor": 10.0,
            "pass": target_spinor <= 0.1 * best_spinor,
        },
        "current": {
            "best_baseline": best_current_model,
            "best_baseline_mean": best_current,
            "multi_ltpb_mean": target_current,
            "improvement_factor": best_current / target_current if target_current > 0 else math.inf,
            "required_factor": 10.0,
            "pass": target_current <= 0.1 * best_current,
        },
    }

    paired: list[dict[str, Any]] = []
    for seed in range(5):
        seed_rows = runs[runs["seed"] == seed].set_index("model")
        target_row = seed_rows.loc[TARGET_MODEL]
        seed_best_spinor_model = min(
            BASELINE_MODELS,
            key=lambda model: float(seed_rows.loc[model, "spinor_rel_l2_gauge_aligned"]),
        )
        seed_best_current_model = min(
            BASELINE_MODELS,
            key=lambda model: float(seed_rows.loc[model, "current_rel_l2"]),
        )
        seed_best_spinor = float(seed_rows.loc[seed_best_spinor_model, "spinor_rel_l2_gauge_aligned"])
        seed_best_current = float(seed_rows.loc[seed_best_current_model, "current_rel_l2"])
        target_seed_spinor = float(target_row["spinor_rel_l2_gauge_aligned"])
        target_seed_current = float(target_row["current_rel_l2"])
        spinor_pass = target_seed_spinor <= 0.1 * seed_best_spinor
        current_pass = target_seed_current <= 0.1 * seed_best_current
        paired.append(
            {
                "seed": seed,
                "spinor_best_baseline": seed_best_spinor_model,
                "spinor_best_baseline_value": seed_best_spinor,
                "spinor_multi_ltpb_value": target_seed_spinor,
                "spinor_improvement_factor": seed_best_spinor / target_seed_spinor if target_seed_spinor > 0 else math.inf,
                "current_best_baseline": seed_best_current_model,
                "current_best_baseline_value": seed_best_current,
                "current_multi_ltpb_value": target_seed_current,
                "current_improvement_factor": seed_best_current / target_seed_current if target_seed_current > 0 else math.inf,
                "spinor_pass": bool(spinor_pass),
                "current_pass": bool(current_pass),
                "joint_pass": bool(spinor_pass and current_pass),
            }
        )
    paired_pass_count = sum(int(item["joint_pass"]) for item in paired)

    slope_channels: list[dict[str, Any]] = []
    target_rows = runs[runs["model"] == TARGET_MODEL].sort_values("seed")
    for mode in MODE_KEYS:
        error_column = f"{mode}_phase_slope_relative_error"
        valid_column = f"{mode}_phase_valid_fraction"
        if error_column not in target_rows or valid_column not in target_rows:
            mean_error = math.inf
            valid_all = False
            per_seed_errors = [math.inf] * 5
            per_seed_valid = [0.0] * 5
        else:
            per_seed_errors = target_rows[error_column].astype(float).tolist()
            per_seed_valid = target_rows[valid_column].astype(float).tolist()
            valid_all = all(
                math.isfinite(error) and valid >= 0.90
                for error, valid in zip(per_seed_errors, per_seed_valid)
            )
            mean_error = _finite_mean(target_rows[error_column]) if valid_all else math.inf
        slope_channels.append(
            {
                "mode": mode,
                "per_seed_absolute_relative_errors": per_seed_errors,
                "per_seed_valid_fractions": per_seed_valid,
                "mean_absolute_relative_error": mean_error,
                "threshold": 0.01,
                "operator": "<",
                "pass": bool(valid_all and mean_error < 0.01),
            }
        )
    slope_pass_count = sum(int(item["pass"]) for item in slope_channels)

    component_passes = [item["pass"] for item in absolute.values()] + [
        advantages["spinor"]["pass"],
        advantages["current"]["pass"],
        paired_pass_count >= 4,
        slope_pass_count >= 3,
    ]
    return {
        "decision_name": "preregistered_multimode_ltpb_success",
        "population_sd_ddof": 0,
        "absolute_thresholds": absolute,
        "mean_baseline_advantage": advantages,
        "paired_seed_results": paired,
        "paired_joint_pass_count": paired_pass_count,
        "paired_joint_pass_required": 4,
        "paired_seed_rule_pass": paired_pass_count >= 4,
        "phase_slope_channels": slope_channels,
        "phase_slope_pass_count": slope_pass_count,
        "phase_slope_pass_required": 3,
        "phase_slope_rule_pass": slope_pass_count >= 3,
        "overall_pass": bool(all(component_passes)),
    }
