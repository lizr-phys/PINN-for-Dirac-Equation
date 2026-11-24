from __future__ import annotations

"""Pre-registered decision logic for the LTPB mechanism ablation."""

from typing import Any

import numpy as np
import pandas as pd


ABLATION_MODELS = (
    "ltpb_cpa",
    "branch_complex",
    "component_polar",
    "rotated_ltpb_15",
    "rotated_ltpb_30",
)
ABLATION_SEEDS = (0, 1, 2)
CORE_METRICS = (
    "spinor_rel_l2",
    "current_rel_l2",
    "relative_phase_rmse_rad",
)

DOMINANCE_RATIO_MAX = 0.2
ROTATION_STEP_RATIO_MIN = 1.25
ROTATION_PAIRED_SEEDS_MIN = 2
ROTATION_30_RATIO_MIN = 3.0
PROXIMITY_ERROR_RATIO_MAX = 2.0
PROXIMITY_FIDELITY_GAP_MAX = 0.01


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0 if numerator <= 0.0 else float("inf")
    return numerator / denominator


def _mean(summary: pd.DataFrame, model: str, metric: str) -> float:
    row = summary[summary["model"] == model]
    if len(row) != 1:
        raise RuntimeError(f"Expected one summary row for {model}, found {len(row)}")
    value = float(row.iloc[0][f"{metric}_mean"])
    if not np.isfinite(value):
        raise RuntimeError(f"Non-finite summary value for {model}/{metric}: {value}")
    return value


def _seed_values(runs: pd.DataFrame, model: str, metric: str) -> np.ndarray:
    rows = runs[runs["model"] == model].sort_values("seed")
    if tuple(rows["seed"].astype(int)) != ABLATION_SEEDS:
        raise RuntimeError(f"Expected seeds {ABLATION_SEEDS} for {model}")
    values = rows[metric].to_numpy(float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"Non-finite run values for {model}/{metric}")
    return values


def compute_mechanism_decision(
    runs: pd.DataFrame, summary: pd.DataFrame
) -> dict[str, Any]:
    """Apply the fixed thresholds without inspecting or selecting individual runs."""

    dominance: dict[str, Any] = {}
    rotation_trend: dict[str, Any] = {}
    rotation_30: dict[str, Any] = {}
    for metric in CORE_METRICS:
        ltpb = _mean(summary, "ltpb_cpa", metric)
        branch_complex = _mean(summary, "branch_complex", metric)
        component_polar = _mean(summary, "component_polar", metric)
        rotated_15 = _mean(summary, "rotated_ltpb_15", metric)
        rotated_30 = _mean(summary, "rotated_ltpb_30", metric)

        dominance_branch_ratio = _ratio(ltpb, branch_complex)
        dominance_component_ratio = _ratio(ltpb, component_polar)
        dominance[metric] = {
            "ltpb_over_branch_complex": dominance_branch_ratio,
            "ltpb_over_component_polar": dominance_component_ratio,
            "passes": dominance_branch_ratio <= DOMINANCE_RATIO_MAX
            and dominance_component_ratio <= DOMINANCE_RATIO_MAX,
        }

        ratio_15_over_0 = _ratio(rotated_15, ltpb)
        ratio_30_over_15 = _ratio(rotated_30, rotated_15)
        seed_0 = _seed_values(runs, "ltpb_cpa", metric)
        seed_15 = _seed_values(runs, "rotated_ltpb_15", metric)
        seed_30 = _seed_values(runs, "rotated_ltpb_30", metric)
        paired_increases_0_to_15 = int(np.sum(seed_15 > seed_0))
        paired_increases_15_to_30 = int(np.sum(seed_30 > seed_15))
        rotation_trend[metric] = {
            "rot15_over_ltpb": ratio_15_over_0,
            "rot30_over_rot15": ratio_30_over_15,
            "paired_seed_increases_0_to_15": paired_increases_0_to_15,
            "paired_seed_increases_15_to_30": paired_increases_15_to_30,
            "passes": ratio_15_over_0 >= ROTATION_STEP_RATIO_MIN
            and ratio_30_over_15 >= ROTATION_STEP_RATIO_MIN
            and paired_increases_0_to_15 >= ROTATION_PAIRED_SEEDS_MIN
            and paired_increases_15_to_30 >= ROTATION_PAIRED_SEEDS_MIN,
        }

        ratio_30_over_0 = _ratio(rotated_30, ltpb)
        rotation_30[metric] = {
            "rot30_over_ltpb": ratio_30_over_0,
            "passes": ratio_30_over_0 >= ROTATION_30_RATIO_MIN,
        }

    dominance_passes = all(item["passes"] for item in dominance.values())
    rotation_trend_count = sum(bool(item["passes"]) for item in rotation_trend.values())
    rotation_30_count = sum(bool(item["passes"]) for item in rotation_30.values())

    proximity: dict[str, Any] = {}
    ltpb_fidelity = _mean(summary, "ltpb_cpa", "fidelity_mean")
    for model in ("branch_complex", "component_polar"):
        error_ratios = {
            metric: _ratio(_mean(summary, model, metric), _mean(summary, "ltpb_cpa", metric))
            for metric in CORE_METRICS
        }
        fidelity_gap = abs(_mean(summary, model, "fidelity_mean") - ltpb_fidelity)
        proximity[model] = {
            "error_ratios_over_ltpb": error_ratios,
            "fidelity_absolute_gap": fidelity_gap,
            "passes": all(
                value <= PROXIMITY_ERROR_RATIO_MAX for value in error_ratios.values()
            )
            and fidelity_gap <= PROXIMITY_FIDELITY_GAP_MAX,
        }

    strong_support = (
        dominance_passes
        and rotation_trend_count >= 2
        and rotation_30_count >= 2
    )
    return {
        "thresholds": {
            "dominance_ratio_max": DOMINANCE_RATIO_MAX,
            "rotation_step_ratio_min": ROTATION_STEP_RATIO_MIN,
            "rotation_paired_seeds_min": ROTATION_PAIRED_SEEDS_MIN,
            "rotation_30_ratio_min": ROTATION_30_RATIO_MIN,
            "proximity_error_ratio_max": PROXIMITY_ERROR_RATIO_MAX,
            "proximity_fidelity_gap_max": PROXIMITY_FIDELITY_GAP_MAX,
        },
        "core_metrics": list(CORE_METRICS),
        "ltpb_dominance": dominance,
        "ltpb_dominance_passes": dominance_passes,
        "rotation_trend": rotation_trend,
        "rotation_trend_passing_metric_count": rotation_trend_count,
        "rotation_trend_passes": rotation_trend_count >= 2,
        "rotation_30_degradation": rotation_30,
        "rotation_30_passing_metric_count": rotation_30_count,
        "rotation_30_passes": rotation_30_count >= 2,
        "proximity": proximity,
        "strong_mechanism_support": strong_support,
    }

