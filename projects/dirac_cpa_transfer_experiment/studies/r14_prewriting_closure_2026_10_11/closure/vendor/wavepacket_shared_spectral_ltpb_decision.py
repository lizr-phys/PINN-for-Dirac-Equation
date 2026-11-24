from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


BASELINE_MODELS = (
    "physical_vanilla",
    "physical_sp_cpa",
    "physical_component_polar",
    "shared_spectral_complex",
)
TARGET_MODEL = "shared_spectral_ltpb"
ALL_MODELS = (*BASELINE_MODELS, TARGET_MODEL)


def _mean(frame: pd.DataFrame, column: str) -> float:
    values = frame[column].to_numpy(dtype=np.float64)
    return float(np.mean(values)) if np.all(np.isfinite(values)) else math.inf


def _fidelity_mean(frame: pd.DataFrame) -> float:
    values = frame["fidelity_mean"].to_numpy(dtype=np.float64)
    return float(np.mean(values)) if np.all(np.isfinite(values)) else -math.inf


def evaluate_wavepacket_preregistered_decision(runs: pd.DataFrame) -> dict[str, Any]:
    """Apply the frozen shared-spectral wavepacket success rule."""

    required = {
        "model",
        "seed",
        "spinor_rel_l2_gauge_aligned",
        "current_rel_l2",
        "fidelity_mean",
        "training_phase_slope_weighted_relative_error",
        "training_phase_min_valid_fraction",
        "heldout_phase_slope_weighted_relative_error",
        "heldout_phase_min_valid_fraction",
        "branch_plus_center_rmse",
        "branch_minus_center_rmse",
        "branch_plus_velocity_relative_error",
        "branch_minus_velocity_relative_error",
        "branch_plus_velocity",
        "branch_minus_velocity",
        "branch_plus_width_rel_l2",
        "branch_minus_width_rel_l2",
        "branch_plus_final_center",
        "branch_minus_final_center",
        "branch_plus_final_width_change",
        "branch_minus_final_width_change",
        "branch_plus_exact_final_width_change",
        "branch_minus_exact_final_width_change",
        "branch_minimum_norm",
    }
    missing = sorted(required.difference(runs.columns))
    if missing:
        raise ValueError(f"Decision input is missing columns: {missing}")
    expected = {(model, seed) for model in ALL_MODELS for seed in range(5)}
    actual = {(str(row.model), int(row.seed)) for row in runs.itertuples()}
    if len(runs) != 25 or actual != expected:
        raise ValueError(
            "Formal run inventory mismatch; "
            f"count={len(runs)}, missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )

    means = runs.groupby("model", sort=False).mean(numeric_only=True)
    target_rows = runs[runs.model == TARGET_MODEL].sort_values("seed")
    best_spinor_model = min(
        BASELINE_MODELS,
        key=lambda model: _mean(
            runs[runs.model == model], "spinor_rel_l2_gauge_aligned"
        ),
    )
    best_current_model = min(
        BASELINE_MODELS,
        key=lambda model: _mean(runs[runs.model == model], "current_rel_l2"),
    )
    target_spinor = _mean(target_rows, "spinor_rel_l2_gauge_aligned")
    target_current = _mean(target_rows, "current_rel_l2")
    target_fidelity = _fidelity_mean(target_rows)
    best_spinor = _mean(
        runs[runs.model == best_spinor_model], "spinor_rel_l2_gauge_aligned"
    )
    best_current = _mean(
        runs[runs.model == best_current_model], "current_rel_l2"
    )

    absolute = {
        "spinor_mean_le_0p05": {
            "value": target_spinor, "threshold": 0.05, "operator": "<=",
            "pass": bool(target_spinor <= 0.05),
        },
        "current_mean_le_0p10": {
            "value": target_current, "threshold": 0.10, "operator": "<=",
            "pass": bool(target_current <= 0.10),
        },
        "fidelity_mean_ge_0p99": {
            "value": target_fidelity, "threshold": 0.99, "operator": ">=",
            "pass": bool(target_fidelity >= 0.99),
        },
    }
    advantage = {
        "spinor": {
            "best_baseline": best_spinor_model,
            "best_baseline_mean": best_spinor,
            "target_mean": target_spinor,
            "improvement_factor": best_spinor / target_spinor if target_spinor > 0 else math.inf,
            "required_factor": 10.0,
            "pass": bool(target_spinor <= 0.1 * best_spinor),
        },
        "current": {
            "best_baseline": best_current_model,
            "best_baseline_mean": best_current,
            "target_mean": target_current,
            "improvement_factor": best_current / target_current if target_current > 0 else math.inf,
            "required_factor": 10.0,
            "pass": bool(target_current <= 0.1 * best_current),
        },
    }

    paired: list[dict[str, Any]] = []
    for seed in range(5):
        seed_rows = runs[runs.seed == seed].set_index("model")
        target_row = seed_rows.loc[TARGET_MODEL]
        spinor_model = min(
            BASELINE_MODELS,
            key=lambda model: float(seed_rows.loc[model, "spinor_rel_l2_gauge_aligned"]),
        )
        current_model = min(
            BASELINE_MODELS,
            key=lambda model: float(seed_rows.loc[model, "current_rel_l2"]),
        )
        seed_spinor = float(target_row["spinor_rel_l2_gauge_aligned"])
        seed_current = float(target_row["current_rel_l2"])
        baseline_spinor = float(seed_rows.loc[spinor_model, "spinor_rel_l2_gauge_aligned"])
        baseline_current = float(seed_rows.loc[current_model, "current_rel_l2"])
        spinor_pass = seed_spinor <= 0.1 * baseline_spinor
        current_pass = seed_current <= 0.1 * baseline_current
        paired.append(
            {
                "seed": seed,
                "spinor_best_baseline": spinor_model,
                "spinor_best_baseline_value": baseline_spinor,
                "spinor_target_value": seed_spinor,
                "spinor_improvement_factor": baseline_spinor / seed_spinor if seed_spinor > 0 else math.inf,
                "current_best_baseline": current_model,
                "current_best_baseline_value": baseline_current,
                "current_target_value": seed_current,
                "current_improvement_factor": baseline_current / seed_current if seed_current > 0 else math.inf,
                "spinor_pass": bool(spinor_pass),
                "current_pass": bool(current_pass),
                "joint_pass": bool(spinor_pass and current_pass),
            }
        )
    paired_count = sum(int(row["joint_pass"]) for row in paired)

    train_slope = _mean(target_rows, "training_phase_slope_weighted_relative_error")
    train_valid = float(target_rows["training_phase_min_valid_fraction"].min())
    train_slope_pass = math.isfinite(train_slope) and train_slope < 0.01 and train_valid >= 0.90
    heldout_slope = _mean(target_rows, "heldout_phase_slope_weighted_relative_error")
    heldout_valid = float(target_rows["heldout_phase_min_valid_fraction"].min())
    heldout_slope_pass = (
        math.isfinite(heldout_slope) and heldout_slope < 0.02 and heldout_valid >= 0.90
    )

    branch_values = {
        key: _mean(target_rows, key)
        for key in (
            "branch_plus_center_rmse",
            "branch_minus_center_rmse",
            "branch_plus_velocity_relative_error",
            "branch_minus_velocity_relative_error",
            "branch_plus_velocity",
            "branch_minus_velocity",
            "branch_plus_width_rel_l2",
            "branch_minus_width_rel_l2",
            "branch_plus_final_center",
            "branch_minus_final_center",
            "branch_plus_final_width_change",
            "branch_minus_final_width_change",
            "branch_plus_exact_final_width_change",
            "branch_minus_exact_final_width_change",
        )
    }
    branch_minimum_norm = float(target_rows["branch_minimum_norm"].min())
    branch_all_finite = bool(
        np.all(
            np.isfinite(
                target_rows[
                    [
                        *branch_values.keys(),
                        "branch_minimum_norm",
                    ]
                ].to_numpy(dtype=np.float64)
            )
        )
    )
    branch_checks = {
        "finite_all_seeds": branch_all_finite,
        "identifiable_all_seeds": branch_all_finite
        and branch_minimum_norm >= 1.0e-8,
        "velocity_signs": branch_values["branch_plus_velocity"] > 0.0
        and branch_values["branch_minus_velocity"] < 0.0,
        "plus_velocity_relative_error_lt_0p10": branch_values[
            "branch_plus_velocity_relative_error"
        ] < 0.10,
        "minus_velocity_relative_error_lt_0p10": branch_values[
            "branch_minus_velocity_relative_error"
        ] < 0.10,
        "plus_center_rmse_le_0p25": branch_values["branch_plus_center_rmse"] <= 0.25,
        "minus_center_rmse_le_0p25": branch_values["branch_minus_center_rmse"] <= 0.25,
        "plus_width_rel_l2_lt_0p10": branch_values["branch_plus_width_rel_l2"] < 0.10,
        "minus_width_rel_l2_lt_0p10": branch_values["branch_minus_width_rel_l2"] < 0.10,
        "terminal_centers_split": branch_values["branch_plus_final_center"] > 0.0
        and branch_values["branch_minus_final_center"] < 0.0,
        "plus_width_change_direction": branch_values["branch_plus_final_width_change"]
        * branch_values["branch_plus_exact_final_width_change"] > 0.0,
        "minus_width_change_direction": branch_values["branch_minus_final_width_change"]
        * branch_values["branch_minus_exact_final_width_change"] > 0.0,
    }
    branch_pass = bool(all(branch_checks.values()))

    components = [
        *(entry["pass"] for entry in absolute.values()),
        advantage["spinor"]["pass"],
        advantage["current"]["pass"],
        paired_count >= 4,
        train_slope_pass,
        heldout_slope_pass,
        branch_pass,
    ]
    overall_pass = bool(all(components))
    return {
        "decision_name": "preregistered_wavepacket_shared_spectral_ltpb_success",
        "population_sd_ddof": 0,
        "absolute_thresholds": absolute,
        "mean_baseline_advantage": advantage,
        "paired_seed_results": paired,
        "paired_joint_pass_count": paired_count,
        "paired_joint_pass_required": 4,
        "paired_seed_rule_pass": bool(paired_count >= 4),
        "training_phase_slope": {
            "five_seed_mean_weighted_relative_error": train_slope,
            "maximum_unidentifiable_fraction": 1.0 - train_valid,
            "threshold": 0.01,
            "operator": "<",
            "pass": bool(train_slope_pass),
        },
        "heldout_midpoint_phase_slope": {
            "five_seed_mean_weighted_relative_error": heldout_slope,
            "maximum_unidentifiable_fraction": 1.0 - heldout_valid,
            "threshold": 0.02,
            "operator": "<",
            "pass": bool(heldout_slope_pass),
        },
        "wavepacket_branch_dynamics": {
            "five_seed_mean_values": branch_values,
            "minimum_branch_norm_across_target_runs": branch_minimum_norm,
            "checks": {key: bool(value) for key, value in branch_checks.items()},
            "pass": branch_pass,
        },
        "continuous_spectral_interpolation_claim_allowed": overall_pass,
        "overall_pass": overall_pass,
    }


__all__ = ["evaluate_wavepacket_preregistered_decision"]
