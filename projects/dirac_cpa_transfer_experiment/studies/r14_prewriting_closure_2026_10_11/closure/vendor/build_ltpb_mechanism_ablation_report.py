from __future__ import annotations

"""Validate and build the LTPB mechanism-ablation report.

The report is intentionally generated only after all 15 formal runs, aggregate
tables, the pre-registered decision file, and all figure exports are complete.
Run it as:

    python -m src.build_ltpb_mechanism_ablation_report
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ltpb_mechanism_decision import (
    ABLATION_MODELS,
    ABLATION_SEEDS,
    CORE_METRICS,
    compute_mechanism_decision,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
RAW_ROOT = RESULTS_ROOT / "raw" / "ltpb_mechanism_ablation"
FIGURE_ROOT = RESULTS_ROOT / "figures" / "ltpb_mechanism_ablation"
RUNS_PATH = RESULTS_ROOT / "ltpb_mechanism_ablation_runs.csv"
SUMMARY_PATH = RESULTS_ROOT / "ltpb_mechanism_ablation_summary.csv"
COMPARISONS_PATH = RESULTS_ROOT / "ltpb_mechanism_ablation_comparisons.csv"
DECISION_PATH = RESULTS_ROOT / "ltpb_mechanism_ablation_decision.json"
MANIFEST_PATH = FIGURE_ROOT / "figure_manifest.json"
PROTECTED_BASELINE_PATH = RESULTS_ROOT / "ltpb_mechanism_ablation_protected_baseline.json"
AUDIT_PATH = RESULTS_ROOT / "ltpb_mechanism_ablation_audit.json"
REPORT_PATH = PROJECT_ROOT / "LTPB_MECHANISM_ABLATION_REPORT.md"

EXPERIMENT = "ltpb_mechanism_ablation"
CASE = "M"
ETA = 1.0
RUN_FILES = (
    "config.json",
    "metrics.json",
    "loss_history.csv",
    "time_series.csv",
    "dense_grid.npz",
    "model_state.pt",
)
EXPECTED_FIGURES = (
    "01_core_error_comparison",
    "02_per_seed_summary",
    "03_current_x0",
    "04_relative_phase_x0",
    "05_rotation_degradation",
    "06_amplitude_phase_complexity",
    "07_loss_histories",
)

MODEL_LABELS = {
    "ltpb_cpa": "LTPB-CPA",
    "branch_complex": "Branch-Complex",
    "component_polar": "Component-Polar",
    "rotated_ltpb_15": "Rotated-LTPB 15°",
    "rotated_ltpb_30": "Rotated-LTPB 30°",
}

REQUESTED_METRICS = (
    "spinor_rel_l2",
    "final_spinor_rel_l2",
    "rho_rel_l2",
    "current_rel_l2",
    "fidelity_mean",
    "relative_phase_rmse_rad",
    "branch_coefficient_rel_l2",
    "branch_occupation_rmse",
    "parameter_count",
    "training_seconds",
    "final_loss_pde",
)

INTERNAL_METRICS = (
    "internal_amplitude_drift_rel_rms",
    "internal_exact_amplitude_drift_rel_rms",
    "internal_phase_velocity_fluctuation_weighted_rms",
    "internal_exact_phase_velocity_fluctuation_weighted_rms",
    "internal_phase_curvature_weighted_rms",
    "internal_exact_phase_curvature_weighted_rms",
    "internal_channel_1_phase_slope_mean",
    "internal_exact_channel_1_phase_slope_mean",
    "internal_channel_2_phase_slope_mean",
    "internal_exact_channel_2_phase_slope_mean",
)

REQUIRED_METRICS = REQUESTED_METRICS + INTERNAL_METRICS + (
    "fidelity_min",
    "fidelity_final",
    "relative_phase_mae_rad",
    "branch_plus_coefficient_abs_rms",
    "negative_branch_coefficient_abs_rms",
    "branch_plus_occupation_rmse",
    "branch_minus_occupation_rmse",
    "final_loss_total",
    "final_loss_continuity",
    "final_loss_norm",
    "final_loss_energy",
    "final_loss_structure",
    "internal_channel_coefficient_rel_l2",
    "internal_amplitude_rel_l2",
    "internal_phase_velocity_fluctuation_over_energy",
    "internal_exact_phase_velocity_fluctuation_over_energy",
    "internal_phase_curvature_over_energy_squared",
    "internal_exact_phase_curvature_over_energy_squared",
    "internal_initial_coefficient_min_abs",
    "internal_basis_orthogonality_max_abs",
    "internal_reconstruction_max_abs",
    "internal_channel_1_near_zero_fraction",
    "internal_channel_2_near_zero_fraction",
    "internal_channel_1_phase_valid_link_fraction",
    "internal_channel_2_phase_valid_link_fraction",
)

TIME_SERIES_COLUMNS = (
    "t",
    "current_x0_predicted",
    "current_x0_exact",
    "relative_phase_x0_predicted",
    "relative_phase_x0_exact",
    "fidelity",
    "occupation_plus_predicted",
    "occupation_plus_exact",
    "occupation_minus_predicted",
    "occupation_minus_exact",
    "internal_channel_1_amplitude_spatial_mean",
    "internal_exact_channel_1_amplitude_spatial_mean",
    "internal_channel_1_phase_spatial_weighted_mean",
    "internal_exact_channel_1_phase_spatial_weighted_mean",
    "internal_channel_1_phase_velocity_spatial_weighted_mean",
    "internal_exact_channel_1_phase_velocity_spatial_weighted_mean",
    "internal_channel_2_amplitude_spatial_mean",
    "internal_exact_channel_2_amplitude_spatial_mean",
    "internal_channel_2_phase_spatial_weighted_mean",
    "internal_exact_channel_2_phase_spatial_weighted_mean",
    "internal_channel_2_phase_velocity_spatial_weighted_mean",
    "internal_exact_channel_2_phase_velocity_spatial_weighted_mean",
)

DENSE_GRID_KEYS = (
    "x",
    "t",
    "psi_predicted",
    "psi_exact",
    "rho_predicted",
    "rho_exact",
    "current_predicted",
    "current_exact",
    "relative_phase_predicted",
    "relative_phase_exact",
    "internal_channel_coefficient_predicted",
    "internal_channel_coefficient_exact",
    "internal_channel_amplitude_predicted",
    "internal_channel_amplitude_exact",
    "internal_channel_phase_unwrapped_predicted",
    "internal_channel_phase_unwrapped_exact",
    "internal_channel_phase_velocity_predicted",
    "internal_channel_phase_velocity_exact",
    "internal_channel_phase_curvature_predicted",
    "internal_channel_phase_curvature_exact",
    "internal_channel_basis",
    "internal_channel_initial_coefficients",
)

METRIC_LABELS = {
    "spinor_rel_l2": "全时空 spinor relative L2",
    "final_spinor_rel_l2": "final-time relative L2",
    "rho_rel_l2": "density relative L2",
    "current_rel_l2": "current relative L2",
    "fidelity_mean": "mean fidelity",
    "relative_phase_rmse_rad": "分量 relative-phase RMSE",
    "branch_coefficient_rel_l2": "正负能分支系数 relative L2",
    "branch_occupation_rmse": "正负能分支占据 RMSE",
    "parameter_count": "参数量",
    "training_seconds": "训练时间 (s)",
    "final_loss_pde": "最终 PDE loss",
    "internal_amplitude_drift_rel_rms": "内部通道振幅漂移",
    "internal_exact_amplitude_drift_rel_rms": "解析固有振幅漂移",
    "internal_phase_velocity_fluctuation_weighted_rms": "内部相位斜率波动",
    "internal_exact_phase_velocity_fluctuation_weighted_rms": "解析固有相位斜率波动",
    "internal_phase_curvature_weighted_rms": "内部加权相位曲率",
    "internal_exact_phase_curvature_weighted_rms": "解析固有加权相位曲率",
    "internal_channel_1_phase_slope_mean": "通道 1 平均相位斜率",
    "internal_exact_channel_1_phase_slope_mean": "通道 1 解析平均相位斜率",
    "internal_channel_2_phase_slope_mean": "通道 2 平均相位斜率",
    "internal_exact_channel_2_phase_slope_mean": "通道 2 解析平均相位斜率",
}


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Required file is empty: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def finite_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite: {value!r}")
    return result


def assert_close(
    actual: Any,
    expected: float,
    label: str,
    *,
    rel_tol: float = 1.0e-10,
    abs_tol: float = 1.0e-12,
) -> None:
    try:
        raw_value = float(actual)
    except (TypeError, ValueError):
        raw_value = float("nan")
    if math.isnan(expected) and math.isnan(raw_value):
        # Parameterization-specific diagnostics are intentionally absent for
        # models where they are not defined.  Matching NaNs reproduce that
        # structural missingness and are not a failed numeric check.
        return
    value = finite_number(actual, label)
    if not math.isclose(value, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        raise RuntimeError(f"{label}: expected {expected!r}, found {value!r}")


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing columns in {source}: {missing}")


def path_inside(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes its root: {relative}") from exc
    return path


def validate_figure_file(path: Path) -> None:
    require_file(path)
    if path.stat().st_size < 100:
        raise RuntimeError(f"Figure file is implausibly small: {path}")
    prefix = path.read_bytes()[:1024].lstrip()
    if path.suffix.lower() == ".png" and not prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"Invalid PNG signature: {path}")
    if path.suffix.lower() == ".pdf" and not prefix.startswith(b"%PDF-"):
        raise RuntimeError(f"Invalid PDF signature: {path}")
    if path.suffix.lower() == ".svg":
        if b"<svg" not in prefix:
            raise RuntimeError(f"Invalid SVG content: {path}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "<text" not in text:
            raise RuntimeError(f"SVG does not retain editable text nodes: {path}")


def expected_run_dir(model: str, seed: int) -> Path:
    return RAW_ROOT / "case_M" / model / f"seed_{seed}"


def expected_initial_coefficients(
    model: str, config: dict[str, Any]
) -> np.ndarray:
    eig = config["eigensystem"]
    u_plus = np.asarray(eig["u_plus"], dtype=float)
    u_minus = np.asarray(eig["u_minus"], dtype=float)
    norm = math.sqrt(4.0 * math.pi)
    energy_coefficients = np.asarray((1.0 / norm, 1.0 / norm))
    if model in {"ltpb_cpa", "branch_complex"}:
        return energy_coefficients
    if model == "component_polar":
        return (u_plus + u_minus) / norm
    angle = 15.0 if model == "rotated_ltpb_15" else 30.0
    radians = math.radians(angle)
    rotation_transpose = np.asarray(
        (
            (math.cos(radians), math.sin(radians)),
            (-math.sin(radians), math.cos(radians)),
        )
    )
    return rotation_transpose @ energy_coefficients


def validate_config(
    config: dict[str, Any], model: str, seed: int, source: Path
) -> None:
    run_spec = config.get("run_spec")
    experiment_config = config.get("experiment_config")
    sampling = config.get("sampling")
    parameterization = config.get("two_channel_parameterization")
    if not all(
        isinstance(item, dict)
        for item in (run_spec, experiment_config, sampling, parameterization)
    ):
        raise RuntimeError(f"Incomplete config sections in {source}")

    expected_run = {
        "experiment": EXPERIMENT,
        "case": CASE,
        "eta": ETA,
        "model": model,
        "seed": seed,
        "steps": 5000,
        "lambda_struct": 1.0,
    }
    for key, expected in expected_run.items():
        actual = run_spec.get(key)
        if isinstance(expected, float):
            assert_close(actual, expected, f"{source}:{key}")
        elif actual != expected:
            raise RuntimeError(
                f"{source}:{key}: expected {expected!r}, found {actual!r}"
            )

    fixed_config = {
        "x_min": 0.0,
        "x_max": 2.0 * math.pi,
        "t_max": 5.0,
        "mass": 5.0,
        "wave_number": 2.0,
        "hidden_layers": 4,
        "hidden_width": 64,
        "collocation_points": 1024,
        "adam_steps": 5000,
        "learning_rate": 1.0e-3,
        "history_every": 10,
        "gradient_diagnostic_every": 250,
        "test_x_points": 128,
        "test_t_points": 201,
        "dtype": "float64",
    }
    for key, expected in fixed_config.items():
        actual = experiment_config.get(key)
        if isinstance(expected, float):
            assert_close(actual, expected, f"{source}:experiment_config.{key}")
        elif actual != expected:
            raise RuntimeError(
                f"{source}:experiment_config.{key}: "
                f"expected {expected!r}, found {actual!r}"
            )

    if config.get("model_initialization_seed") != seed:
        raise RuntimeError(f"Wrong model initialization seed in {source}")
    if sampling.get("collocation_generator_seed") != 100_000 + seed:
        raise RuntimeError(f"Wrong collocation generator seed in {source}")
    if sampling.get("structure_generator_seed") != 200_000 + seed:
        raise RuntimeError(f"Wrong structure generator seed in {source}")
    if sampling.get("collocation_resampled_each_step") is not True:
        raise RuntimeError(f"Collocation is not resampled each step in {source}")
    if config.get("initialization") != "Xavier normal for every linear weight; zero biases":
        raise RuntimeError(f"Unexpected initialization protocol in {source}")

    if parameterization.get("uses_exact_energy_or_time_carrier_in_forward") is not False:
        raise RuntimeError(f"Exact-energy leakage flag is not false in {source}")
    aligned_expected = model in {"ltpb_cpa", "branch_complex"}
    if parameterization.get("energy_basis_aligned") is not aligned_expected:
        raise RuntimeError(f"Wrong energy-basis alignment flag in {source}")
    angle_expected: float | None = {
        "ltpb_cpa": 0.0,
        "branch_complex": 0.0,
        "component_polar": None,
        "rotated_ltpb_15": 15.0,
        "rotated_ltpb_30": 30.0,
    }[model]
    angle_actual = parameterization.get("rotation_degrees")
    if angle_expected is None:
        if angle_actual is not None:
            raise RuntimeError(f"Component-Polar rotation metadata should be null in {source}")
    else:
        assert_close(angle_actual, angle_expected, f"{source}:rotation_degrees")

    basis = np.asarray(parameterization.get("basis_matrix_columns"), dtype=float)
    if basis.shape != (2, 2):
        raise RuntimeError(f"Invalid basis matrix shape in {source}: {basis.shape}")
    if np.max(np.abs(basis.T @ basis - np.eye(2))) >= 1.0e-12:
        raise RuntimeError(f"Non-orthogonal channel basis in {source}")
    initial = np.asarray(parameterization.get("initial_coefficients"), dtype=float)
    expected_initial = expected_initial_coefficients(model, config)
    if initial.shape != (2,) or np.max(np.abs(initial - expected_initial)) >= 1.0e-12:
        raise RuntimeError(
            f"Incorrect transformed initial coefficients in {source}: "
            f"expected={expected_initial.tolist()}, found={initial.tolist()}"
        )
    if np.min(np.abs(initial)) <= 1.0e-12:
        raise RuntimeError(f"An internal channel starts unoccupied in {source}")
    digest = config.get("trainable_initial_state_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"Missing trainable initialization hash in {source}")


def validate_metrics(
    metrics: dict[str, Any], model: str, seed: int, source: Path
) -> None:
    expected_metadata = {
        "experiment": EXPERIMENT,
        "case": CASE,
        "model": model,
        "seed": seed,
        "steps": 5000,
        "parameter_count": 12_996,
    }
    for key, expected in expected_metadata.items():
        if metrics.get(key) != expected:
            raise RuntimeError(
                f"{source}:{key}: expected {expected!r}, found {metrics.get(key)!r}"
            )
    assert_close(metrics.get("eta"), ETA, f"{source}:eta")
    for metric in REQUIRED_METRICS:
        finite_number(metrics.get(metric), f"{source}:{metric}")
    if finite_number(metrics["training_seconds"], f"{source}:training_seconds") <= 0.0:
        raise RuntimeError(f"Non-positive training time in {source}")
    for key in (
        "final_loss_continuity",
        "final_loss_norm",
        "final_loss_energy",
        "final_loss_structure",
    ):
        assert_close(metrics[key], 0.0, f"{source}:{key}", abs_tol=1.0e-15)
    assert_close(
        metrics["final_loss_total"],
        finite_number(metrics["final_loss_pde"], f"{source}:final_loss_pde"),
        f"{source}:PDE-only total loss",
    )
    if metrics["internal_initial_coefficient_min_abs"] <= 0.0:
        raise RuntimeError(f"Unidentifiable initial internal phase in {source}")
    if metrics["internal_basis_orthogonality_max_abs"] >= 1.0e-12:
        raise RuntimeError(f"Internal basis failed orthogonality in {source}")
    if metrics["internal_reconstruction_max_abs"] >= 1.0e-12:
        raise RuntimeError(f"Internal channel reconstruction failed in {source}")
    for channel in (1, 2):
        valid_fraction = metrics[f"internal_channel_{channel}_phase_valid_link_fraction"]
        if not 0.0 <= valid_fraction <= 1.0:
            raise RuntimeError(f"Invalid phase-link coverage in {source}")
        near_zero = metrics[f"internal_channel_{channel}_near_zero_fraction"]
        if not 0.0 <= near_zero <= 1.0:
            raise RuntimeError(f"Invalid near-zero fraction in {source}")


def validate_loss_history(path: Path) -> None:
    history = pd.read_csv(require_file(path))
    require_columns(
        history,
        (
            "step",
            "total",
            "pde",
            "continuity",
            "norm",
            "energy",
            "structure",
            "elapsed_seconds",
        ),
        path,
    )
    if len(history) != 501:
        raise RuntimeError(f"Expected 501 loss-history rows in {path}, found {len(history)}")
    if int(history["step"].iloc[0]) != 1 or int(history["step"].iloc[-1]) != 5000:
        raise RuntimeError(f"Loss history does not span steps 1..5000 in {path}")
    required_columns = [
        "step",
        "total",
        "pde",
        "continuity",
        "norm",
        "energy",
        "structure",
        "elapsed_seconds",
    ]
    required = history.loc[:, required_columns].to_numpy(float)
    if not np.all(np.isfinite(required)):
        raise RuntimeError(f"Non-finite required loss history in {path}")
    if np.max(np.abs(history["total"].to_numpy(float) - history["pde"].to_numpy(float))) > 1.0e-14:
        raise RuntimeError(f"Total loss differs from PDE loss in {path}")
    for key in ("continuity", "norm", "energy", "structure"):
        if np.max(np.abs(history[key].to_numpy(float))) > 1.0e-15:
            raise RuntimeError(f"Non-PDE loss {key} is nonzero in {path}")


def validate_time_series(path: Path) -> None:
    frame = pd.read_csv(require_file(path))
    require_columns(frame, TIME_SERIES_COLUMNS, path)
    if len(frame) != 201:
        raise RuntimeError(f"Expected 201 time rows in {path}, found {len(frame)}")
    required = frame[list(TIME_SERIES_COLUMNS)].to_numpy(float)
    if not np.all(np.isfinite(required)):
        raise RuntimeError(f"Non-finite time-series diagnostic in {path}")
    assert_close(frame["t"].iloc[0], 0.0, f"{path}:t[0]")
    assert_close(frame["t"].iloc[-1], 5.0, f"{path}:t[-1]")


def validate_dense_grid(path: Path) -> None:
    require_file(path)
    with np.load(path) as dense:
        missing = sorted(set(DENSE_GRID_KEYS) - set(dense.files))
        if missing:
            raise RuntimeError(f"Missing dense-grid arrays in {path}: {missing}")
        expected_shapes = {
            "x": (128,),
            "t": (201,),
            "psi_predicted": (201, 128, 2),
            "psi_exact": (201, 128, 2),
            "rho_predicted": (201, 128),
            "rho_exact": (201, 128),
            "current_predicted": (201, 128),
            "current_exact": (201, 128),
            "relative_phase_predicted": (201, 128),
            "relative_phase_exact": (201, 128),
            "internal_channel_coefficient_predicted": (201, 128, 2),
            "internal_channel_coefficient_exact": (201, 128, 2),
            "internal_channel_amplitude_predicted": (201, 128, 2),
            "internal_channel_amplitude_exact": (201, 128, 2),
            "internal_channel_phase_unwrapped_predicted": (201, 128, 2),
            "internal_channel_phase_unwrapped_exact": (201, 128, 2),
            "internal_channel_phase_velocity_predicted": (201, 128, 2),
            "internal_channel_phase_velocity_exact": (201, 128, 2),
            "internal_channel_phase_curvature_predicted": (201, 128, 2),
            "internal_channel_phase_curvature_exact": (201, 128, 2),
            "internal_channel_basis": (2, 2),
            "internal_channel_initial_coefficients": (2,),
        }
        for key, shape in expected_shapes.items():
            if dense[key].shape != shape:
                raise RuntimeError(
                    f"Wrong dense-grid shape for {key} in {path}: "
                    f"expected={shape}, found={dense[key].shape}"
                )
        if dense["psi_predicted"].dtype != np.complex128:
            raise RuntimeError(f"Predicted spinor is not complex128 in {path}")
        if dense["psi_exact"].dtype != np.complex128:
            raise RuntimeError(f"Exact spinor is not complex128 in {path}")
        expected_x = 2.0 * math.pi * np.arange(128, dtype=float) / 128.0
        expected_t = np.linspace(0.0, 5.0, 201)
        if not np.allclose(dense["x"], expected_x, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"Dense x grid is not endpoint-free [0,2pi) in {path}")
        if not np.allclose(dense["t"], expected_t, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"Dense t grid is not the fixed 201-point [0,5] grid in {path}")
        for key in DENSE_GRID_KEYS:
            values = dense[key]
            if np.iscomplexobj(values):
                finite = np.isfinite(values.real) & np.isfinite(values.imag)
            else:
                finite = np.isfinite(values)
            if not np.all(finite):
                raise RuntimeError(f"Non-finite dense-grid array {key} in {path}")


def validate_run(model: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = expected_run_dir(model, seed)
    for name in RUN_FILES:
        require_file(run_dir / name)
    config = read_json(run_dir / "config.json")
    metrics = read_json(run_dir / "metrics.json")
    validate_config(config, model, seed, run_dir / "config.json")
    validate_metrics(metrics, model, seed, run_dir / "metrics.json")
    validate_loss_history(run_dir / "loss_history.csv")
    validate_time_series(run_dir / "time_series.csv")
    validate_dense_grid(run_dir / "dense_grid.npz")
    if (run_dir / "model_state.pt").stat().st_size < 1000:
        raise RuntimeError(f"Checkpoint is implausibly small: {run_dir / 'model_state.pt'}")
    return config, metrics


def validate_nested(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(f"JSON structure mismatch at {label}")
        for key in expected:
            validate_nested(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"JSON list mismatch at {label}")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            validate_nested(actual_item, expected_item, f"{label}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise RuntimeError(
                f"JSON value mismatch at {label}: expected={expected!r}, found={actual!r}"
            )
        return
    assert_close(actual, float(expected), label)


def load_and_validate_tables() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[tuple[str, int], dict[str, Any]],
]:
    runs = pd.read_csv(require_file(RUNS_PATH))
    require_columns(
        runs,
        ("experiment", "case", "model", "seed", "steps") + REQUIRED_METRICS,
        RUNS_PATH,
    )
    expected_pairs = {
        (model, seed) for model in ABLATION_MODELS for seed in ABLATION_SEEDS
    }
    found_pairs = set(zip(runs["model"], runs["seed"].astype(int), strict=False))
    if len(runs) != 15 or found_pairs != expected_pairs:
        raise RuntimeError(
            f"Runs table must contain exactly 15 planned runs; "
            f"missing={sorted(expected_pairs - found_pairs)}, "
            f"extra={sorted(found_pairs - expected_pairs)}"
        )

    actual_metric_paths = {
        path.resolve() for path in RAW_ROOT.rglob("metrics.json")
    }
    expected_metric_paths = {
        (expected_run_dir(model, seed) / "metrics.json").resolve()
        for model, seed in expected_pairs
    }
    if actual_metric_paths != expected_metric_paths:
        raise RuntimeError(
            "Formal raw directory does not contain exactly the planned metric files; "
            f"missing={sorted(map(str, expected_metric_paths - actual_metric_paths))}, "
            f"extra={sorted(map(str, actual_metric_paths - expected_metric_paths))}"
        )

    configs: dict[tuple[str, int], dict[str, Any]] = {}
    metric_payloads: dict[tuple[str, int], dict[str, Any]] = {}
    for model in ABLATION_MODELS:
        for seed in ABLATION_SEEDS:
            config, metrics = validate_run(model, seed)
            configs[(model, seed)] = config
            metric_payloads[(model, seed)] = metrics
            row = runs[
                (runs["model"] == model) & (runs["seed"].astype(int) == seed)
            ].iloc[0]
            for key in ("experiment", "case", "model", "seed", "steps"):
                if row[key] != metrics[key]:
                    raise RuntimeError(
                        f"Runs CSV disagrees with metrics JSON for {model}/seed {seed}/{key}"
                    )
            for key in REQUIRED_METRICS:
                assert_close(
                    row[key],
                    finite_number(metrics[key], f"{model}/{seed}/{key}"),
                    f"runs CSV {model}/seed {seed}/{key}",
                )

    for seed in ABLATION_SEEDS:
        hashes = {
            configs[(model, seed)]["trainable_initial_state_sha256"]
            for model in ABLATION_MODELS
        }
        if len(hashes) != 1:
            raise RuntimeError(
                f"Models do not share the same initialized trainable state at seed {seed}"
            )

    summary = pd.read_csv(require_file(SUMMARY_PATH))
    if len(summary) != len(ABLATION_MODELS) or tuple(summary["model"]) != ABLATION_MODELS:
        raise RuntimeError("Summary rows are incomplete or not in the declared model order")
    if set(summary["case"]) != {CASE} or set(summary["n_seeds"].astype(int)) != {3}:
        raise RuntimeError("Summary must contain n=3 Case M groups")

    excluded = {"seed", "eta", "lambda_struct", "steps"}
    numeric_columns = [
        column
        for column in runs.select_dtypes(include="number").columns
        if column not in excluded
    ]
    for model in ABLATION_MODELS:
        model_runs = runs[runs["model"] == model]
        summary_row = summary[summary["model"] == model].iloc[0]
        for metric in numeric_columns:
            raw_std = model_runs[metric].std(ddof=0)
            expected_std = 0.0 if pd.isna(raw_std) else float(raw_std)
            assert_close(
                summary_row[f"{metric}_mean"],
                float(model_runs[metric].mean()),
                f"summary {model}/{metric} mean",
            )
            assert_close(
                summary_row[f"{metric}_std"],
                expected_std,
                f"summary {model}/{metric} population SD",
            )

    comparisons = pd.read_csv(require_file(COMPARISONS_PATH))
    required_comparison_rows = len(ABLATION_MODELS) * len(numeric_columns)
    if len(comparisons) != required_comparison_rows:
        raise RuntimeError(
            f"Expected {required_comparison_rows} comparison rows, "
            f"found {len(comparisons)}"
        )
    ltpb_summary = summary[summary["model"] == "ltpb_cpa"].iloc[0]
    for model in ABLATION_MODELS:
        model_summary = summary[summary["model"] == model].iloc[0]
        for metric in numeric_columns:
            rows = comparisons[
                (comparisons["model"] == model) & (comparisons["metric"] == metric)
            ]
            if len(rows) != 1:
                raise RuntimeError(f"Missing unique comparison row for {model}/{metric}")
            row = rows.iloc[0]
            mean_value = float(model_summary[f"{metric}_mean"])
            reference = float(ltpb_summary[f"{metric}_mean"])
            expected_ratio = (
                mean_value / reference
                if reference != 0.0
                else (0.0 if mean_value == 0.0 else float("inf"))
            )
            assert_close(row["mean"], mean_value, f"comparison {model}/{metric} mean")
            assert_close(
                row["population_sd"],
                float(model_summary[f"{metric}_std"]),
                f"comparison {model}/{metric} SD",
            )
            if math.isnan(expected_ratio):
                if not math.isnan(float(row["ratio_to_ltpb"])):
                    raise RuntimeError(
                        f"Expected NaN comparison ratio for {model}/{metric}"
                    )
            elif math.isfinite(expected_ratio):
                assert_close(
                    row["ratio_to_ltpb"],
                    expected_ratio,
                    f"comparison {model}/{metric} ratio",
                )
            elif not math.isinf(float(row["ratio_to_ltpb"])):
                raise RuntimeError(f"Expected infinite comparison ratio for {model}/{metric}")

    decision = read_json(DECISION_PATH)
    recomputed = compute_mechanism_decision(runs, summary)
    validate_nested(decision, recomputed, "decision")
    return runs, summary, comparisons, decision, configs


def load_and_validate_manifest() -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    if manifest.get("experiment") != EXPERIMENT:
        raise RuntimeError("Figure manifest experiment identity is incorrect")
    format_contract = manifest.get("format_contract")
    statistics = manifest.get("statistics")
    if not isinstance(format_contract, dict) or not isinstance(statistics, dict):
        raise RuntimeError("Figure manifest lacks format/statistics contracts")
    if int(format_contract.get("png_dpi", 0)) != 600:
        raise RuntimeError("Figure manifest does not declare 600 dpi PNG exports")
    assert_close(
        format_contract.get("figure_width_mm"),
        183.0,
        "figure manifest width",
    )
    if statistics.get("spread") != "population standard deviation (ddof=0)":
        raise RuntimeError("Figure manifest does not declare population SD")
    figures = manifest.get("figures")
    if not isinstance(figures, dict) or tuple(figures) != EXPECTED_FIGURES:
        raise RuntimeError(
            "Figure manifest must contain the seven contracted groups in order; "
            f"found={tuple(figures) if isinstance(figures, dict) else type(figures)}"
        )
    required_concepts = {
        "core": False,
        "seed": False,
        "current": False,
        "phase": False,
        "rotation": False,
        "complexity": False,
    }
    for key, entry in figures.items():
        if not isinstance(entry, dict):
            raise RuntimeError(f"Invalid manifest entry for {key}")
        files = entry.get("files")
        if not isinstance(files, dict) or set(("png", "pdf", "svg")) - set(files):
            raise RuntimeError(f"Figure {key} lacks PNG/PDF/SVG exports")
        for extension in ("png", "pdf", "svg"):
            path = path_inside(FIGURE_ROOT, str(files[extension]), f"{key}/{extension}")
            if path.suffix.lower() != f".{extension}":
                raise RuntimeError(f"Wrong extension for {key}/{extension}: {path}")
            if path.stem != key:
                raise RuntimeError(f"Figure basename does not match key {key}: {path}")
            validate_figure_file(path)
        source_data = entry.get("source_data")
        if not isinstance(source_data, list) or not source_data:
            raise RuntimeError(f"Figure {key} lacks source-data links")
        for relative in source_data:
            source_path = path_inside(FIGURE_ROOT, str(relative), f"{key}/source_data")
            require_file(source_path)
            if source_path.suffix.lower() == ".csv":
                frame = pd.read_csv(source_path)
                if frame.empty:
                    raise RuntimeError(f"Empty figure source data: {source_path}")
                numeric = frame.select_dtypes(include="number")
                if not numeric.empty and not np.all(
                    np.isfinite(numeric.to_numpy(float))
                ):
                    raise RuntimeError(f"Non-finite figure source data: {source_path}")
        if not str(entry.get("conclusion", "")).strip():
            raise RuntimeError(f"Figure {key} lacks a declared conclusion")
        if not str(entry.get("question", "")).strip():
            raise RuntimeError(f"Figure {key} lacks a declared research question")
        if not str(entry.get("title", "")).strip():
            raise RuntimeError(f"Figure {key} lacks a report title")
        if not str(entry.get("statistics", "")).strip():
            raise RuntimeError(f"Figure {key} lacks statistics metadata")
        searchable = " ".join(
            (
                key,
                str(entry.get("title", "")),
                str(entry.get("conclusion", "")),
            )
        ).lower()
        tokens = {
            "core": ("core", "metric", "error"),
            "seed": ("seed", "individual"),
            "current": ("current",),
            "phase": ("relative phase", "relative_phase"),
            "rotation": ("rotation", "angle"),
            "complexity": ("complexity", "curvature", "amplitude"),
        }
        for concept, choices in tokens.items():
            if any(choice in searchable for choice in choices):
                required_concepts[concept] = True
    missing_concepts = [key for key, present in required_concepts.items() if not present]
    if missing_concepts:
        raise RuntimeError(
            f"Figure manifest does not document required concepts: {missing_concepts}"
        )
    return manifest


def fmt(value: Any, digits: int = 4) -> str:
    number = float(value)
    if not math.isfinite(number):
        return str(number)
    absolute = abs(number)
    if absolute == 0.0:
        return "0"
    if absolute >= 1.0e4 or absolute < 1.0e-3:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def pass_text(value: Any) -> str:
    return "PASS" if bool(value) else "FAIL"


def escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(escape_cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(escape_cell(item) for item in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def relative_link(path: Path, label: str) -> str:
    relative = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    return f"[{label}]({relative})"


def mean_sd(summary: pd.DataFrame, model: str, metric: str) -> str:
    row = summary[summary["model"] == model].iloc[0]
    return (
        f"{fmt(row[f'{metric}_mean'])} ± "
        f"{fmt(row[f'{metric}_std'], digits=3)}"
    )


def build_per_seed_accuracy_table(runs: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    for model in ABLATION_MODELS:
        subset = runs[runs["model"] == model].sort_values("seed")
        for _, row in subset.iterrows():
            rows.append(
                [
                    MODEL_LABELS[model],
                    str(int(row["seed"])),
                    fmt(row["spinor_rel_l2"]),
                    fmt(row["final_spinor_rel_l2"]),
                    fmt(row["rho_rel_l2"]),
                    fmt(row["current_rel_l2"]),
                    fmt(row["fidelity_mean"], 6),
                    fmt(row["relative_phase_rmse_rad"]),
                    fmt(row["branch_coefficient_rel_l2"]),
                    fmt(row["branch_occupation_rmse"]),
                    str(int(row["parameter_count"])),
                    fmt(row["training_seconds"], 2),
                    fmt(row["final_loss_pde"]),
                ]
            )
    return md_table(
        [
            "模型",
            "seed",
            "spinor L2",
            "final L2",
            "rho L2",
            "current L2",
            "fidelity",
            "relative-phase RMSE",
            "branch coeff. L2",
            "branch occ. RMSE",
            "params",
            "time (s)",
            "final PDE",
        ],
        rows,
    )


def build_per_seed_internal_table(runs: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    for model in ABLATION_MODELS:
        subset = runs[runs["model"] == model].sort_values("seed")
        for _, row in subset.iterrows():
            rows.append(
                [
                    MODEL_LABELS[model],
                    str(int(row["seed"])),
                    fmt(row["internal_amplitude_drift_rel_rms"]),
                    fmt(row["internal_exact_amplitude_drift_rel_rms"]),
                    fmt(row["internal_phase_velocity_fluctuation_weighted_rms"]),
                    fmt(row["internal_exact_phase_velocity_fluctuation_weighted_rms"]),
                    fmt(row["internal_phase_curvature_weighted_rms"]),
                    fmt(row["internal_exact_phase_curvature_weighted_rms"]),
                    fmt(row["internal_channel_1_phase_slope_mean"]),
                    fmt(row["internal_exact_channel_1_phase_slope_mean"]),
                    fmt(row["internal_channel_2_phase_slope_mean"]),
                    fmt(row["internal_exact_channel_2_phase_slope_mean"]),
                ]
            )
    return md_table(
        [
            "模型",
            "seed",
            "amp drift pred.",
            "amp drift exact",
            "slope fluct. pred.",
            "slope fluct. exact",
            "curvature pred.",
            "curvature exact",
            "slope 1 pred.",
            "slope 1 exact",
            "slope 2 pred.",
            "slope 2 exact",
        ],
        rows,
    )


def build_aggregate_accuracy_table(summary: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    for model in ABLATION_MODELS:
        rows.append(
            [
                MODEL_LABELS[model],
                "3",
                mean_sd(summary, model, "spinor_rel_l2"),
                mean_sd(summary, model, "final_spinor_rel_l2"),
                mean_sd(summary, model, "rho_rel_l2"),
                mean_sd(summary, model, "current_rel_l2"),
                mean_sd(summary, model, "fidelity_mean"),
                mean_sd(summary, model, "relative_phase_rmse_rad"),
                mean_sd(summary, model, "branch_coefficient_rel_l2"),
                mean_sd(summary, model, "branch_occupation_rmse"),
                mean_sd(summary, model, "training_seconds"),
                mean_sd(summary, model, "final_loss_pde"),
            ]
        )
    return md_table(
        [
            "模型",
            "n",
            "spinor L2",
            "final L2",
            "rho L2",
            "current L2",
            "fidelity",
            "relative-phase RMSE",
            "branch coeff. L2",
            "branch occ. RMSE",
            "time (s)",
            "final PDE",
        ],
        rows,
    )


def build_aggregate_internal_table(summary: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    for model in ABLATION_MODELS:
        rows.append(
            [
                MODEL_LABELS[model],
                mean_sd(summary, model, "internal_amplitude_drift_rel_rms"),
                mean_sd(summary, model, "internal_exact_amplitude_drift_rel_rms"),
                mean_sd(
                    summary,
                    model,
                    "internal_phase_velocity_fluctuation_weighted_rms",
                ),
                mean_sd(
                    summary,
                    model,
                    "internal_exact_phase_velocity_fluctuation_weighted_rms",
                ),
                mean_sd(summary, model, "internal_phase_curvature_weighted_rms"),
                mean_sd(summary, model, "internal_exact_phase_curvature_weighted_rms"),
                mean_sd(summary, model, "internal_channel_1_phase_slope_mean"),
                mean_sd(summary, model, "internal_channel_2_phase_slope_mean"),
            ]
        )
    return md_table(
        [
            "模型",
            "amp drift pred.",
            "amp drift exact",
            "slope fluct. pred.",
            "slope fluct. exact",
            "curvature pred.",
            "curvature exact",
            "mean slope ch.1",
            "mean slope ch.2",
        ],
        rows,
    )


def build_decision_tables(decision: dict[str, Any]) -> str:
    dominance_rows: list[list[str]] = []
    trend_rows: list[list[str]] = []
    rotation_30_rows: list[list[str]] = []
    for metric in CORE_METRICS:
        label = METRIC_LABELS[metric]
        dominance = decision["ltpb_dominance"][metric]
        dominance_rows.append(
            [
                label,
                fmt(dominance["ltpb_over_branch_complex"]),
                fmt(dominance["ltpb_over_component_polar"]),
                pass_text(dominance["passes"]),
            ]
        )
        trend = decision["rotation_trend"][metric]
        trend_rows.append(
            [
                label,
                fmt(trend["rot15_over_ltpb"]),
                fmt(trend["rot30_over_rot15"]),
                str(trend["paired_seed_increases_0_to_15"]),
                str(trend["paired_seed_increases_15_to_30"]),
                pass_text(trend["passes"]),
            ]
        )
        rotation_30 = decision["rotation_30_degradation"][metric]
        rotation_30_rows.append(
            [
                label,
                fmt(rotation_30["rot30_over_ltpb"]),
                pass_text(rotation_30["passes"]),
            ]
        )

    thresholds = decision["thresholds"]
    return "\n\n".join(
        (
            "### 预判据 1：LTPB 同时优于 Branch-Complex 与 Component-Polar\n\n"
            f"每项要求两个 LTPB/control 误差比均不超过 "
            f"{fmt(thresholds['dominance_ratio_max'])}。\n\n"
            + md_table(
                [
                    "核心指标",
                    "LTPB / Branch-Complex",
                    "LTPB / Component-Polar",
                    "结果",
                ],
                dominance_rows,
            )
            + f"\n\n整体：**{pass_text(decision['ltpb_dominance_passes'])}**。",
            "### 预判据 2：旋转后至少两项核心误差随角度明显增大\n\n"
            f"实现为每一角度步均至少 {fmt(thresholds['rotation_step_ratio_min'])} 倍，"
            f"且两个步长分别至少有 "
            f"{thresholds['rotation_paired_seeds_min']}/3 个 paired seeds 增大。\n\n"
            + md_table(
                [
                    "核心指标",
                    "15° / 0°",
                    "30° / 15°",
                    "0→15 增大 seeds",
                    "15→30 增大 seeds",
                    "结果",
                ],
                trend_rows,
            )
            + f"\n\n通过指标数：{decision['rotation_trend_passing_metric_count']}/3；"
            f"整体：**{pass_text(decision['rotation_trend_passes'])}**。",
            "### 预判据 3：30° 至少两项达到 LTPB 的 3 倍\n\n"
            + md_table(
                ["核心指标", "30° / LTPB", "结果"],
                rotation_30_rows,
            )
            + f"\n\n通过指标数：{decision['rotation_30_passing_metric_count']}/3；"
            f"整体：**{pass_text(decision['rotation_30_passes'])}**。",
        )
    )


def proximity_text(decision: dict[str, Any]) -> str:
    parts: list[str] = []
    branch = decision["proximity"]["branch_complex"]
    component = decision["proximity"]["component_polar"]
    branch_ratios = branch["error_ratios_over_ltpb"]
    component_ratios = component["error_ratios_over_ltpb"]
    parts.append(
        "Branch-Complex 接近性判定为 "
        f"**{pass_text(branch['passes'])}**：其 spinor/current/relative-phase "
        f"误差相对 LTPB 的倍数为 "
        f"{fmt(branch_ratios['spinor_rel_l2'])}、"
        f"{fmt(branch_ratios['current_rel_l2'])}、"
        f"{fmt(branch_ratios['relative_phase_rmse_rad'])}，"
        f"fidelity 绝对差为 {fmt(branch['fidelity_absolute_gap'])}。"
    )
    parts.append(
        "Component-Polar 接近性判定为 "
        f"**{pass_text(component['passes'])}**：其三个误差倍数为 "
        f"{fmt(component_ratios['spinor_rel_l2'])}、"
        f"{fmt(component_ratios['current_rel_l2'])}、"
        f"{fmt(component_ratios['relative_phase_rmse_rad'])}，"
        f"fidelity 绝对差为 {fmt(component['fidelity_absolute_gap'])}。"
    )
    if branch["passes"]:
        parts.append(
            "Branch-Complex 若接近 LTPB，说明在本预算内已知能量本征基底可能是主要贡献，"
            "不能把全部提升归因于显式极坐标相位。"
        )
    if component["passes"]:
        parts.append(
            "Component-Polar 若接近 LTPB，只能说明当前原始分量基下的双极坐标足以优化；"
            "由于该基底仅相当于 -10.9007° 旋转，不能推广到任意失配基底。"
        )
    if not branch["passes"] and not component["passes"]:
        parts.append(
            "两个控制均未接近 LTPB，结合旋转趋势才能进一步判断是否支持"
            "本征能量支与独立相位的一一对应。"
        )
    return "\n\n".join(parts)


def failure_text(decision: dict[str, Any]) -> str:
    failures: list[str] = []
    if not decision["ltpb_dominance_passes"]:
        failed = [
            METRIC_LABELS[metric]
            for metric, item in decision["ltpb_dominance"].items()
            if not item["passes"]
        ]
        failures.append(
            "LTPB 对两个控制的 0.2 倍优势未在以下指标同时成立："
            + "、".join(failed)
            + "。"
        )
    if not decision["rotation_trend_passes"]:
        failed = [
            METRIC_LABELS[metric]
            for metric, item in decision["rotation_trend"].items()
            if not item["passes"]
        ]
        failures.append(
            "严格逐角度退化条件不足两项；未通过的指标为：" + "、".join(failed) + "。"
        )
    if not decision["rotation_30_passes"]:
        failed = [
            METRIC_LABELS[metric]
            for metric, item in decision["rotation_30_degradation"].items()
            if not item["passes"]
        ]
        failures.append(
            "30° 至少三倍退化条件不足两项；未通过的指标为：" + "、".join(failed) + "。"
        )
    if not failures:
        failures.append(
            "三个强机制子判据均通过；这里的“失败情况”主要指个别 seed 或次要指标可能仍有"
            "非单调波动，不能被总体 PASS 掩盖，逐 seed 表保留了这些差异。"
        )
    failures.append(
        "smoke test 仅用于排除 NaN 和流水线错误；正式配置在查看结果前固定为 "
        "5000 steps、Adam 1e-3 和每步 1024 个配点。正式目录对应预先列出的 15 个"
        "模型—seed 组合，本报告未按结果改变学习率、步数或挑选 seed，也未为有利结果"
        "选择性重跑。"
    )
    return "\n\n".join(f"- {item}" for item in failures)


def current_phase_ranking(summary: pd.DataFrame) -> str:
    paragraphs: list[str] = []
    for metric in ("current_rel_l2", "relative_phase_rmse_rad"):
        ordered = sorted(
            (
                (
                    MODEL_LABELS[model],
                    float(
                        summary[summary["model"] == model].iloc[0][f"{metric}_mean"]
                    ),
                )
                for model in ABLATION_MODELS
            ),
            key=lambda item: item[1],
        )
        paragraphs.append(
            f"{METRIC_LABELS[metric]} 从低到高为："
            + " < ".join(f"{label} ({fmt(value)})" for label, value in ordered)
            + "。"
        )
    return "\n\n".join(paragraphs)


def build_raw_links() -> str:
    rows: list[list[str]] = []
    for model in ABLATION_MODELS:
        for seed in ABLATION_SEEDS:
            run_dir = expected_run_dir(model, seed)
            links = "；".join(
                relative_link(run_dir / name, name) for name in RUN_FILES
            )
            rows.append((MODEL_LABELS[model], str(seed), links))
    return md_table(["模型", "seed", "逐 run 原始文件"], [list(row) for row in rows])


def build_figure_section(manifest: dict[str, Any]) -> str:
    sections: list[str] = []
    for index, (key, entry) in enumerate(manifest["figures"].items(), start=1):
        files = entry["files"]
        png = path_inside(FIGURE_ROOT, str(files["png"]), f"{key}/png")
        pdf = path_inside(FIGURE_ROOT, str(files["pdf"]), f"{key}/pdf")
        svg = path_inside(FIGURE_ROOT, str(files["svg"]), f"{key}/svg")
        source_links = "；".join(
            relative_link(
                path_inside(FIGURE_ROOT, str(relative), f"{key}/source"),
                Path(str(relative)).name,
            )
            for relative in entry["source_data"]
        )
        title = str(entry.get("title", key.replace("_", " ")))
        note = str(entry.get("note", "")).strip()
        note_text = f"\n\n说明：{note}" if note else ""
        sections.append(
            f"### 图 {index}. {title}\n\n"
            f"![{title}]({png.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()})\n\n"
            f"导出：{relative_link(png, 'PNG')}；{relative_link(pdf, 'PDF')}；"
            f"{relative_link(svg, 'SVG')}。源数据：{source_links}。\n\n"
            f"研究问题：{entry['question']}\n\n"
            f"图结论：{entry['conclusion']}\n\n"
            f"统计说明：{entry['statistics']}{note_text}"
        )
    return "\n\n".join(sections)


def build_report(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    decision: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    strong = bool(decision["strong_mechanism_support"])
    verdict = (
        "通过预注册强机制支持判据"
        if strong
        else "未通过预注册强机制支持判据"
    )
    verdict_explanation = (
        "在这一自由、固定单动量 Case M 和固定优化预算内，结果支持"
        "“本征能量支与独立相位一一对应”是性能来源之一。"
        if strong
        else
        "因此本实验不能把 LTPB 的优势强归因于“本征能量支与独立相位一一对应”；"
        "下面按失败子判据保留更弱、条件化的解释。"
    )

    phi_degrees = 0.5 * math.degrees(math.atan2(2.0, 5.0))
    accuracy_table = build_per_seed_accuracy_table(runs)
    internal_table = build_per_seed_internal_table(runs)
    aggregate_accuracy = build_aggregate_accuracy_table(summary)
    aggregate_internal = build_aggregate_internal_table(summary)
    decisions = build_decision_tables(decision)
    proximity = proximity_text(decision)
    failures = failure_text(decision)
    rankings = current_phase_ranking(summary)
    raw_links = build_raw_links()
    figures = build_figure_section(manifest)
    ltpb_slope_1 = mean_sd(
        summary, "ltpb_cpa", "internal_channel_1_phase_slope_mean"
    )
    ltpb_slope_2 = mean_sd(
        summary, "ltpb_cpa", "internal_channel_2_phase_slope_mean"
    )
    spectral_energy = math.sqrt(29.0)

    return f"""# LTPB-CPA 机制归因消融实验报告

## 摘要与结论

本实验在自由一维 Dirac 方程的单动量、正负能等权混合 Case M 上，对五个四输出模型完成了 seeds 0、1、2 共 15 个正式 5000-step run。所有模型共享完全相同的 PDE 强残差、硬初值、4×64 tanh MLP、float64/complex128、Adam 1e-3、每步重新采样 1024 个配点以及逐 seed 配点序列。

**自动判定：{verdict}。** {verdict_explanation}

这一结论只涉及当前自由、固定 k=2、m=5、T=5 的受控问题和给定优化预算。它不是通用 Dirac 求解器结论，不涉及“世界首次”，也不否定其他表示在不同优化器、预算、波包或外势下的可能性。

数据入口：{relative_link(RUNS_PATH, '15-run CSV')}；{relative_link(SUMMARY_PATH, 'mean 与 population SD')}；{relative_link(COMPARISONS_PATH, '相对 LTPB 比较表')}；{relative_link(DECISION_PATH, '预判据 JSON')}；{relative_link(MANIFEST_PATH, '图 manifest')}；{relative_link(PROTECTED_BASELINE_PATH, '旧结果逐文件保护基线')}；{relative_link(AUDIT_PATH, '最终机器审计')}。

正式指标出现前固定的运行集与判据见 {relative_link(PROJECT_ROOT / 'LTPB_MECHANISM_ABLATION_PREREGISTRATION.md', '控制记录')}，图证据链见 {relative_link(PROJECT_ROOT / 'LTPB_MECHANISM_ABLATION_FIGURE_CONTRACT.md', 'figure contract')}。

## 1. 已有工作与本次研究问题

原实验报告 {relative_link(PROJECT_ROOT / 'EXPERIMENT_REPORT.md', 'EXPERIMENT_REPORT.md')} 说明公共单相位 SP-CPA 在纯正能支 Case P 中有效，但在 Case M 中其有限预算优势明显退化；连续性、总范数和总能量也不能充分识别正负能相对相位错误。随后 {relative_link(PROJECT_ROOT / 'LEARNED_TWO_PHASE_PILOT_REPORT.md', 'LEARNED_TWO_PHASE_PILOT_REPORT.md')} 表明 LTPB-CPA 能在不给定精确时间载波的条件下学习相反的能支相位并恢复 Case M。

然而 LTPB 同时改变了两件事：使用已知正负能本征基底，并为每个通道设置独立实振幅和相位。本次消融检验性能主要来自：

1. 已知能量本征基底；
2. 一般双通道极坐标表示；
3. 本征能量支与独立相位的一一对齐。

## 2. 物理问题

自然单位下求解

    i ∂t ψ = (-i σx ∂x + m σz) ψ,

其中 x∈[0,2π]、t∈[0,5]、m=5、k=2、η=1。初值为

    ψ(x,0) = exp(ikx) (u+ + u-) / sqrt(4π).

自由符号 Hk=kσx+mσz 的归一化本征旋量 u+、u- 属于已知物理算符先验。精确能量 E=sqrt(m²+k²) 只用于训练后的解析评价；训练目标中没有解析标签、相位标签或精确载波。

## 3. 五个比较模型

五个模型都由同一个 3→64→64→64→64→4 tanh MLP 产生四个实输出，并用 τ=t/T 硬满足初值。

### 3.1 LTPB-CPA

    ψhat = exp(ikx) [a+ u+ exp(iS+) + a- u- exp(iS-)]

其中 a±=a±,0+τr±，S±=τq±。它使用能量本征基底和显式独立相位，但 forward 不使用 E。

### 3.2 Branch-Complex

    ψhat = exp(ikx) (c+ u+ + c- u-)

其中 c±=c±,0+τ(δc±,R+iδc±,I)。它保留相同能量本征基底，却不显式分离振幅和相位。因此 LTPB 与 Branch-Complex 的差异主要检验显式极坐标参数化，而不是基底。

### 3.3 Component-Polar

    ψhat = exp(ikx) [b1 exp(iS1), b2 exp(iS2)]ᵀ

其中 bj=bj,0+τrj，Sj=τqj。其初始 bj,0 由 (u++u-)/sqrt(4π) 的两个实分量精确给出，允许有符号振幅；它使用双极坐标但不显式使用能量支基底。

### 3.4 Rotated-LTPB 15° 与 30°

令

    v1 = cosθ u+ + sinθ u-,
    v2 = -sinθ u+ + cosθ u-,

并使用

    ψhat = exp(ikx) [b1 v1 exp(iS1) + b2 v2 exp(iS2)].

初始能量基系数 α0=(1,1)ᵀ/sqrt(4π) 通过正交变换 β0=Rθᵀα0 精确计算。15°、30° 两个角度的通道初始振幅均非零。

### 3.5 Component-Polar 是旋转族的特例

在当前确定性实规范下 U=(u+,u-)=Rφ，其中

    φ = 0.5 atan2(k,m) = {phi_degrees:.10f}°.

因此原始分量基 I=U R(-φ)，Component-Polar 数学上等价于 θ=-{phi_degrees:.4f}° 的 Rotated-LTPB，而不是与旋转族完全独立的第五种机制。若 Component-Polar 接近 LTPB，只能说明当前轻度失配的原始分量基足够，不能推出任意双分量极坐标都足够。

## 4. 公平性、随机源与无能量泄漏

- 五个模型均为四个实输出、12,996 个可训练参数；没有通过增加容量获得优势。
- 旧 pilot 使用 `[-π,π]`，本任务明确指定 `[0,2π]`。两者描述同一周期环但配点坐标和 `[sin x, cos x, t/T]` 输入逐点不同，因此没有复用旧 LTPB 指标；五个模型的 15 个 run 均在新域上重新训练。
- 训练配置为 float64/complex128、4×64 tanh、Adam 1e-3、5000 steps、每步 1024 个重新采样配点。
- 对每个 seed，五模型的 trainable initialization SHA-256 完全相同。
- 模型初始化使用 seed；配点使用独立的 100000+seed Generator。因此相同 seed 的五模型获得相同配点序列，且配点随机源不受模型构造消耗的全局 RNG 影响。
- 每个正式 run 只有 PDE loss；loss history 中 total=PDE，continuity/norm/energy/structure 均为 0。
- 五模型配置均声明并经测试验证 forward 不读取已存储 E，也不含 exp(∓iEt) 载波。配置快照记录谱本征值仅用于可复现性；解析解、能量目标和通道精确复杂度只在训练完成后计算。
- 自动测试另行验证硬初值、输出 dtype/shape、参数量、基变换正交性、精确初始系数、无 E 泄漏、解析表示和机器精度 PDE residual。
- 项目路径中的 `.git` 缺少有效仓库元数据，`git status` 返回 `not a git repository`，因此不能把 Git 当作保护证据。正式写入新目录前已对 268 个既有 raw、figure、summary、audit 与两份正式报告保存逐文件 SHA-256；最终独立 audit 会重新比较路径集合、大小和内容哈希。

首次正式启动使用前台 worker，三个 seed 均在首个 LTPB run 尚未完成时因宿主会话中断；当时没有任何 `metrics.json`、评价结果或 completed run。随后三者均以完全相同的固定代码、初始化 seed 和配点 seed 从 step 1 确定性重启，未使用 `--force`、未观察可供筛选的正式指标，也没有选择性保留或重跑。完整 worker 日志与恢复记录保存在 `results/logs/ltpb_mechanism_ablation/`，只有重启后完整完成的 15 个 run 进入统计。

统计均使用三个独立 seeds 的算术均值和 population SD（ddof=0）。n=3 只支持描述性稳定性判断，不作总体显著性检验。

## 5. 每个 seed 的完整结果

### 5.1 场精度、能支精度与计算量

{accuracy_table}

### 5.2 内部通道诊断

{internal_table}

内部相位诊断不是直接读取可能受 (b,S)→(-b,S+π) 影响的 raw S，而是从物理复通道系数构造振幅加权相位链接。近零振幅位置不贡献相位统计，并单独保存有效覆盖率。正负能分支系数误差仍统一投影到 u± 上，未被模型自身旋转通道替代。

## 6. 均值与 population SD

### 6.1 主要误差

{aggregate_accuracy}

### 6.2 预测内部复杂度与解析固有复杂度

{aggregate_internal}

LTPB 的两个能量对齐通道学得平均相位斜率分别为 `{ltpb_slope_1}` 与 `{ltpb_slope_2}`，训练后评价目标为 `-E={-spectral_energy:.10f}` 与 `+E={spectral_energy:.10f}`。这里的 E 只在评价阶段用于解释斜率，未进入 forward、loss 或优化器。

振幅漂移、相位斜率波动和加权相位曲率在旋转基底中并不以零为解析目标：一个旋转通道同时含 exp(-iEt) 与 exp(+iEt)，其解析振幅和瞬时相位速度本来就随时间变化。因此上表将 predicted 与 exact intrinsic complexity 并列；predicted 较大不自动等于误差，predicted 与 exact 的失配才反映学习不足。对能量对齐的 LTPB/Branch-Complex，解析通道则为恒振幅、线性相位，固有斜率波动和曲率应接近数值零。

## 7. 预注册机制判据

{decisions}

### 7.1 强机制总判定

**{verdict}：{pass_text(strong)}。** {verdict_explanation}

### 7.2 Branch-Complex 与 Component-Polar 的接近性解释

{proximity}

## 8. 概率流与旋量相对相位

Case M 的概率流来自正负能支干涉，是本征支相对相位的敏感量；旋量分量 relative phase 则直接检查两分量相位关系，而不会被密度正确所替代。

{rankings}

x=0 的完整 current 与 relative-phase 时间曲线保存在每个 run 的 time_series.csv，并在图中同时显示逐 seed、均值和 population SD。判定仍以全时空 current L2 和带掩膜的 circular relative-phase RMSE 为核心，不凭训练 loss 或单条曲线下结论。

## 9. 失败子判据、未调参与如实解释

{failures}

若 Branch-Complex 接近 LTPB，最直接解释是已知能量基底已把 PDE 近似解耦，显式相位并非主要来源。若 Component-Polar 接近 LTPB，则当前轻度旋转的原始分量极坐标可能已足够。若旋转不随角度退化，则不能声称能量支对齐关键。无论总判定为何，本报告都不把 density、norm 或单独 final PDE loss 的改善替代 spinor、current 和 relative phase。

为避免概念混淆，本实验只作如下分层归因：

- **表达能力**：自动测试用各模型自身通道精确重建解析解并得到机器精度 Dirac residual；因此误差差异不能写成控制模型“数学上不能表示”。
- **载波预条件**：LTPB 把目标化为近恒振幅和近线性独立相位；旋转/分量基的解析目标本身需要振幅调制与非线性相位，Branch-Complex 则仍需直接拟合快速复振荡。
- **优化难度**：所有排名只说明固定 5000-step Adam、固定容量和固定随机流下的可达精度，不说明无限预算极限。
- **守恒量盲点**：本轮故意只用 PDE loss；已有实验关于连续性、总范数和总能量不能唯一识别能支相对相位的结论是背景证据，不是本轮新增守恒消融。
- **已知能支基底先验**：u± 来自已知 Hk，LTPB 不是无先验地发现能支；即使最终支持对齐机制，也必须把这项物理先验与“未硬编码 E 时间载波”同时陈述。

## 10. 局限性与不能推出的结论

1. 这里只研究自由、周期、单一 k=2、固定 m=5、有限 T=5 的平面波混合态，不能推广为波包、外势、散射、非周期边界或长时间 Dirac 求解器。
2. LTPB 和 Branch-Complex 显式使用已知 Hk 的 u±；即使无 E 时间载波，也不是从原始数据无先验发现能支。
3. 这不是完整析因设计：缺少例如 Component-Complex、Rotated-Complex 以及不同角度与不同 m/k 的全交叉组合，不能无条件分离所有基底—参数化交互。
4. Component-Polar 只对应 -10.9007° 的轻度旋转；当前 m/k 使标准分量基天然接近能量基。
5. 三个 seeds 只能估计当前初始化和配点随机性的有限波动，不能给出可靠总体显著性或失败概率。
6. 5000-step Adam 比较反映表达、载波预条件和优化难度的联合结果；一次消融不能证明某表示在无限优化预算下不可达。
7. 内部通道相位在振幅接近零时不可辨识。虽然本实验选择的解析初始系数非零并采用振幅权重与覆盖率，但学习轨迹仍可能接近零。
8. 不使用总连续性、总范数或总能量损失是为了隔离表示机制；本实验不评价这些约束在其他问题上的实用价值。
9. 本报告不声称世界首次，也不声称公共单相位在数学上不能表示混合态。
10. 三个 seed worker 并行运行并共享 CPU；各模型的训练时间仅作资源记录，不用于声称某种参数化具有更高计算效率。

## 11. 原始数据与可复现链接

### 11.1 十五个正式 run

{raw_links}

每个 run 均包含 config.json、metrics.json、loss_history.csv、time_series.csv、dense_grid.npz 和 model_state.pt。聚合与判定文件见报告摘要的数据入口。

### 11.2 图、三格式导出与源数据

{figures}

## 12. 复现命令

在项目根目录执行：

    python -m src.run_ltpb_mechanism_ablation --mode smoke --smoke-steps 20
    python -m src.run_ltpb_mechanism_ablation --mode full
    python -m src.run_ltpb_mechanism_ablation --mode aggregate
    python -m src.plot_ltpb_mechanism_ablation
    python -m src.build_ltpb_mechanism_ablation_report

正式并行运行时可以按 model/seed 拆分并使用 --no-aggregate，但最终 aggregate、plot 与本报告构建必须在全部 15 个 run 完成后执行。报告构建器会重新核对逐 run 配置、六件套、聚合均值与 population SD、比较比值、预判据 JSON、七个固定图组及其非空有限源数据；独立 audit 还会从原始工件重算图源统计并核验格式与旧结果保护。任一必需项缺失或不一致都会使相应阶段失败。
"""


def main() -> None:
    require_file(PROTECTED_BASELINE_PATH)
    runs, summary, _comparisons, decision, _configs = load_and_validate_tables()
    manifest = load_and_validate_manifest()
    report = build_report(runs, summary, decision, manifest)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(
        f"Generated {REPORT_PATH} from {len(runs)} validated formal runs; "
        f"strong_support={decision['strong_mechanism_support']}"
    )


if __name__ == "__main__":
    main()
