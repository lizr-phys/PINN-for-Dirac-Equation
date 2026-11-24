from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .wavepacket_shared_spectral_ltpb_decision import (
    BASELINE_MODELS,
    TARGET_MODEL,
    evaluate_wavepacket_preregistered_decision,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "results" / "raw" / "wavepacket_shared_spectral_ltpb" / "learned"
DEFAULT_RESULTS = ROOT / "results"
MODELS = (*BASELINE_MODELS, TARGET_MODEL)
SEEDS = tuple(range(5))
CORE_METRICS = (
    "spinor_rel_l2_gauge_aligned",
    "spinor_rel_l2_raw",
    "final_spinor_rel_l2_gauge_aligned",
    "density_rel_l2",
    "current_rel_l2",
    "fidelity_mean",
    "spectral_coefficient_rel_l2_gauge_aligned",
    "spectral_occupation_rel_l2",
    "training_phase_slope_weighted_relative_error",
    "training_phase_min_valid_fraction",
    "heldout_phase_slope_weighted_relative_error",
    "heldout_phase_min_valid_fraction",
    "heldout_spectral_pde_residual_rms",
    "heldout_amplitude_drift_weighted_rel_rms",
    "dense_physical_pde_residual_rms",
    "spectral_training_pde_residual_rms",
    "max_norm_drift",
    "branch_plus_center_rmse",
    "branch_minus_center_rmse",
    "branch_plus_velocity",
    "branch_minus_velocity",
    "branch_plus_exact_velocity",
    "branch_minus_exact_velocity",
    "branch_plus_velocity_relative_error",
    "branch_minus_velocity_relative_error",
    "branch_plus_width_rel_l2",
    "branch_minus_width_rel_l2",
    "branch_plus_final_center",
    "branch_minus_final_center",
    "branch_plus_final_width_change",
    "branch_minus_final_width_change",
    "branch_plus_exact_final_width_change",
    "branch_minus_exact_final_width_change",
    "branch_minimum_norm",
    "modal_amplitude_drift_weighted_rel_rms",
    "final_loss_pde",
    "training_seconds",
    "peak_rss_mb",
    "peak_rss_increment_mb",
    "parameter_count",
)


def collect_wavepacket_runs(raw_root: Path = DEFAULT_RAW) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for seed in SEEDS:
            path = raw_root / model / f"seed_{seed}" / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing formal metrics: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            missing = [metric for metric in CORE_METRICS if metric not in payload]
            if missing:
                raise KeyError(f"{path} missing required metrics: {missing}")
            row: dict[str, Any] = {"model": model, "seed": seed}
            for metric in CORE_METRICS:
                value = payload[metric]
                if value is None:
                    row[metric] = np.nan
                elif isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"Non-numeric metric {metric} in {path}")
                else:
                    row[metric] = float(value)
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["model", "seed"], kind="stable")
    frame = frame.reset_index(drop=True)
    if len(frame) != 25:
        raise AssertionError(f"Expected 25 formal rows, got {len(frame)}")
    return frame


def summarize_wavepacket_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        group = runs[runs.model == model]
        if len(group) != 5:
            raise ValueError(f"Expected five runs for {model}, found {len(group)}")
        row: dict[str, Any] = {"model": model, "n_seeds": 5}
        for metric in CORE_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"{metric}_n"] = int(len(finite))
            row[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else np.nan
            row[f"{metric}_std"] = float(np.std(finite, ddof=0)) if len(finite) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_wavepacket_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    means = runs.groupby("model", sort=False).mean(numeric_only=True)
    rows: list[dict[str, Any]] = []
    for metric, direction in (
        ("spinor_rel_l2_gauge_aligned", "lower"),
        ("density_rel_l2", "lower"),
        ("current_rel_l2", "lower"),
        ("fidelity_mean", "higher"),
        ("training_seconds", "lower"),
        ("peak_rss_increment_mb", "lower"),
    ):
        baseline = means.loc[list(BASELINE_MODELS), metric].dropna()
        best_model = baseline.idxmin() if direction == "lower" else baseline.idxmax()
        best_value = float(baseline.loc[best_model])
        target = float(means.loc[TARGET_MODEL, metric])
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "best_baseline": best_model,
                "best_baseline_mean": best_value,
                "target_mean": target,
                "best_baseline_over_target": best_value / target if target != 0 else np.inf,
                "target_over_best_baseline": target / best_value if best_value != 0 else np.inf,
            }
        )
    return pd.DataFrame(rows)


def write_wavepacket_outputs(
    raw_root: Path = DEFAULT_RAW, results_root: Path = DEFAULT_RESULTS
) -> dict[str, Path]:
    runs = collect_wavepacket_runs(raw_root)
    summary = summarize_wavepacket_runs(runs)
    comparisons = build_wavepacket_comparisons(runs)
    decision = evaluate_wavepacket_preregistered_decision(runs)
    paths = {
        "runs": results_root / "wavepacket_shared_spectral_ltpb_runs.csv",
        "summary": results_root / "wavepacket_shared_spectral_ltpb_summary.csv",
        "comparisons": results_root / "wavepacket_shared_spectral_ltpb_comparisons.csv",
        "decision": results_root / "wavepacket_shared_spectral_ltpb_decision.json",
    }
    results_root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(paths["runs"], index=False)
    summary.to_csv(paths["summary"], index=False)
    comparisons.to_csv(paths["comparisons"], index=False)
    paths["decision"].write_text(
        json.dumps(decision, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate shared-spectral wavepacket runs")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    for key, path in write_wavepacket_outputs(args.raw_root, args.results_root).items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
