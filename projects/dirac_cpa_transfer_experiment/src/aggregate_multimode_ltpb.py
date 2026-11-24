from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .multimode_config import MODE_KEYS, MULTIMODE_MODELS, MULTIMODE_SEEDS
from .multimode_ltpb_decision import evaluate_preregistered_decision


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "results" / "raw" / "multimode_ltpb" / "learned"
DEFAULT_RESULTS = ROOT / "results"

CORE_METRICS = (
    "spinor_rel_l2_gauge_aligned",
    "spinor_rel_l2_raw",
    "current_rel_l2",
    "fidelity_mean",
    "density_rel_l2",
    "mode_coefficient_rel_l2_gauge_aligned",
    "mode_occupation_rmse_abs",
    "mode_occupation_rmse_rel",
    "dense_pde_residual_rms",
    "max_norm_drift",
    "final_loss_pde",
    "training_seconds",
    "parameter_count",
    "spinor_component_relative_phase_rmse_rad",
    "spinor_component_relative_phase_mask_fraction",
    "relative_phase_k2_branches_circular_rmse_rad",
    "relative_phase_k5_branches_circular_rmse_rad",
    "relative_phase_positive_cross_k_circular_rmse_rad",
    "relative_phase_negative_cross_k_circular_rmse_rad",
    "relative_phase_pairs_circular_rmse_rad",
    "phase_velocity_fluctuation_weighted_rms",
    "phase_curvature_weighted_rms",
    "modal_occupation_sum_defect_mean_abs",
)


def metric_columns() -> tuple[str, ...]:
    per_mode: list[str] = []
    for mode in MODE_KEYS:
        per_mode.extend(
            (
                f"{mode}_coefficient_rel_l2",
                f"{mode}_occupation_rmse_abs",
                f"{mode}_phase_slope",
                f"{mode}_phase_slope_target",
                f"{mode}_phase_slope_relative_error",
                f"{mode}_phase_slope_r2",
                f"{mode}_phase_valid_fraction",
                f"{mode}_phase_circular_rmse_rad",
                f"{mode}_phase_velocity_fluctuation_rms",
                f"{mode}_phase_curvature_weighted_rms",
                f"{mode}_amplitude_drift_rel_rms",
                f"internal_{mode}_amplitude_drift_rel_rms",
                f"internal_{mode}_phase_slope",
                f"internal_{mode}_phase_slope_r2",
            )
        )
    return CORE_METRICS + tuple(per_mode) + ("internal_phase_curvature_weighted_rms",)


def collect_runs(raw_root: Path = DEFAULT_RAW) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    allowed = metric_columns()
    for model in MULTIMODE_MODELS:
        for seed in MULTIMODE_SEEDS:
            path = raw_root / model / f"seed_{seed}" / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing formal metrics: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            row: dict[str, Any] = {"model": model, "seed": seed}
            missing = [key for key in CORE_METRICS if key not in payload]
            if missing:
                raise KeyError(f"{path} missing required metrics {missing}")
            for key in allowed:
                if key in payload:
                    value = payload[key]
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise TypeError(f"Metric {key} in {path} is not numeric")
                    row[key] = float(value)
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["model", "seed"], kind="stable").reset_index(drop=True)
    if len(frame) != 25:
        raise AssertionError(f"Expected 25 rows, got {len(frame)}")
    return frame


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in runs.columns if column not in {"model", "seed"}]
    rows: list[dict[str, Any]] = []
    for model in MULTIMODE_MODELS:
        group = runs[runs["model"] == model]
        if len(group) != 5:
            raise ValueError(f"Expected five runs for {model}, found {len(group)}")
        row: dict[str, Any] = {"model": model, "n_seeds": 5}
        for metric in numeric:
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def build_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = ("spinor_rel_l2_gauge_aligned", "current_rel_l2", "fidelity_mean")
    means = runs.groupby("model")[list(metrics)].mean()
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        minimize = metric != "fidelity_mean"
        baselines = means.loc[[model for model in MULTIMODE_MODELS if model != "multi_ltpb"], metric]
        best_model = baselines.idxmin() if minimize else baselines.idxmax()
        best_value = float(baselines.loc[best_model])
        target = float(means.loc["multi_ltpb", metric])
        rows.append(
            {
                "metric": metric,
                "direction": "lower" if minimize else "higher",
                "best_baseline": best_model,
                "best_baseline_mean": best_value,
                "multi_ltpb_mean": target,
                "ratio_best_baseline_over_multi_ltpb": best_value / target if target != 0 else np.inf,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(raw_root: Path = DEFAULT_RAW, results_root: Path = DEFAULT_RESULTS) -> dict[str, Path]:
    runs = collect_runs(raw_root)
    summary = summarize_runs(runs)
    comparisons = build_comparisons(runs)
    decision = evaluate_preregistered_decision(runs)
    paths = {
        "runs": results_root / "multimode_ltpb_runs.csv",
        "summary": results_root / "multimode_ltpb_summary.csv",
        "comparisons": results_root / "multimode_ltpb_comparisons.csv",
        "decision": results_root / "multimode_ltpb_decision.json",
    }
    results_root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(paths["runs"], index=False)
    summary.to_csv(paths["summary"], index=False)
    comparisons.to_csv(paths["comparisons"], index=False)
    paths["decision"].write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate formal Multi-LTPB runs")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    for name, path in write_outputs(args.raw_root, args.results_root).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
