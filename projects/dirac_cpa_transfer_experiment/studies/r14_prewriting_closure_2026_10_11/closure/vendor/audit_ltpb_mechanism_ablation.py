from __future__ import annotations

"""Audit the immutable outputs of the formal LTPB mechanism ablation.

This module never trains, aggregates, plots, or rewrites an existing result.  It
only reads the declared artifacts and always writes one machine-readable audit
record.  A failed check produces a non-zero process exit status.
"""

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ltpb_mechanism_decision import (  # noqa: E402
    ABLATION_MODELS,
    ABLATION_SEEDS,
    CORE_METRICS,
    ROTATION_30_RATIO_MIN,
    compute_mechanism_decision,
)


RESULTS_ROOT = PROJECT_ROOT / "results"
RAW_ROOT = RESULTS_ROOT / "raw" / "ltpb_mechanism_ablation"
FIGURE_ROOT = RESULTS_ROOT / "figures" / "ltpb_mechanism_ablation"
AUDIT_PATH = RESULTS_ROOT / "ltpb_mechanism_ablation_audit.json"
PROTECTED_BASELINE_PATH = (
    RESULTS_ROOT / "ltpb_mechanism_ablation_protected_baseline.json"
)
PROTECTED_BASELINE_AGGREGATE_SHA256 = (
    "d88d4987c6ffe6b198ebf9b376d328b0a547493a52fa047b6e60655716dd40ab"
)

EXPERIMENT = "ltpb_mechanism_ablation"
CASE = "M"
ETA = 1.0
EXPECTED_PARAMETER_COUNT = 12_996
EXPECTED_RUN_COUNT = 15

RUN_FILES = (
    "config.json",
    "metrics.json",
    "loss_history.csv",
    "time_series.csv",
    "dense_grid.npz",
    "model_state.pt",
)

AGGREGATE_PATHS = {
    "runs": RESULTS_ROOT / "ltpb_mechanism_ablation_runs.csv",
    "summary": RESULTS_ROOT / "ltpb_mechanism_ablation_summary.csv",
    "comparisons": RESULTS_ROOT / "ltpb_mechanism_ablation_comparisons.csv",
}
DECISION_PATH = RESULTS_ROOT / "ltpb_mechanism_ablation_decision.json"

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
    "rotated_ltpb_15": r"Rotated-LTPB $15^\circ$",
    "rotated_ltpb_30": r"Rotated-LTPB $30^\circ$",
}
CORE_METRIC_LABELS = {
    "spinor_rel_l2": "Spinor relative L2",
    "current_rel_l2": "Current relative L2",
    "relative_phase_rmse_rad": "Component relative-phase RMSE (rad)",
}
PLOT_FLOOR = 1.0e-12
SUPPORT_METRICS = (
    "final_spinor_rel_l2",
    "rho_rel_l2",
    "infidelity_mean",
    "branch_coefficient_rel_l2",
    "branch_occupation_rmse",
    "final_loss_pde",
)
SUPPORT_METRIC_LABELS = {
    "final_spinor_rel_l2": "Final-time spinor L2",
    "rho_rel_l2": "Density relative L2",
    "infidelity_mean": "Mean infidelity (1-F)",
    "branch_coefficient_rel_l2": "Energy-branch coefficient L2",
    "branch_occupation_rmse": "Energy-branch occupation RMSE",
    "final_loss_pde": "Final PDE loss",
}
COMPLEXITY_METRICS = (
    (
        "internal_amplitude_drift_rel_rms",
        "internal_exact_amplitude_drift_rel_rms",
    ),
    (
        "internal_phase_velocity_fluctuation_over_energy",
        "internal_exact_phase_velocity_fluctuation_over_energy",
    ),
    (
        "internal_phase_curvature_over_energy_squared",
        "internal_exact_phase_curvature_over_energy_squared",
    ),
)
COMPLEXITY_METRIC_LABELS = {
    "internal_amplitude_drift_rel_rms": "Amplitude drift (relative RMS)",
    "internal_phase_velocity_fluctuation_over_energy": (
        r"Phase-velocity fluctuation / $E$"
    ),
    "internal_phase_curvature_over_energy_squared": (
        r"Weighted phase curvature / $E^2$"
    ),
}

SOURCE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "01_core_error_comparison": (
        "model", "model_label", "seed", "metric", "metric_label", "value",
        "group_mean", "population_sd", "group_ratio_to_ltpb",
    ),
    "02_per_seed_summary": (
        "model", "model_label", "seed", "metric", "metric_label", "value",
        "ltpb_same_seed_value", "paired_ratio_with_plot_floor", "group_mean",
        "population_sd", "ltpb_group_mean", "group_ratio_with_plot_floor",
        "plot_floor",
    ),
    "03_current_x0": (
        "model", "model_label", "seed", "t", "predicted_raw", "exact_raw",
        "predicted_display", "exact_display", "circular_error",
        "seed_display_branch_shift_rad", "group_mean_display",
        "population_sd_display",
    ),
    "04_relative_phase_x0": (
        "model", "model_label", "seed", "t", "predicted_raw", "exact_raw",
        "predicted_display", "exact_display", "circular_error",
        "seed_display_branch_shift_rad", "group_mean_display",
        "population_sd_display",
    ),
    "05_rotation_degradation": (
        "metric", "metric_label", "model", "angle_degrees", "role", "seed",
        "value", "group_mean", "population_sd", "ltpb_same_seed_value",
        "paired_ratio_to_ltpb", "group_ratio_to_ltpb",
        "three_x_ltpb_mean_threshold",
    ),
    "06_amplitude_phase_complexity": (
        "model", "model_label", "seed", "metric", "exact_metric",
        "metric_label", "predicted", "exact_representation",
        "predicted_group_mean", "predicted_population_sd", "exact_group_mean",
        "exact_population_sd",
    ),
    "07_loss_histories": (
        "model", "model_label", "seed", "step", "pde_loss", "group_mean",
        "population_sd",
    ),
}

SOURCE_ROWS = {
    "01_core_error_comparison": 45,
    "02_per_seed_summary": 90,
    "03_current_x0": 3015,
    "04_relative_phase_x0": 3015,
    "05_rotation_degradation": 36,
    "06_amplitude_phase_complexity": 45,
    "07_loss_histories": 7515,
}

# These are intentionally literals, rather than values captured at audit time.
# Directory digests are computed by _directory_sha256 exactly as documented
# below.  Changing any byte or relative name in a protected baseline fails the
# audit, even if a newer audit would otherwise consider the artifact valid.
BASELINE_SHA256: dict[str, tuple[Path, str, str]] = {
    "raw/main": (
        RESULTS_ROOT / "raw" / "main",
        "directory",
        "9B9808481529CC320949F3D095B81EACA7FDCD1D7D089FDE96FAF7A68BFCCEF7",
    ),
    "raw/sensitivity": (
        RESULTS_ROOT / "raw" / "sensitivity",
        "directory",
        "F64F0E0BA4327A51C864D2884AF960414CF516F3E191BA1ECF96080648BE8D2E",
    ),
    "raw/learned_two_phase_pilot": (
        RESULTS_ROOT / "raw" / "learned_two_phase_pilot",
        "directory",
        "9541B41DDB0C956230A2621D3BF8EC6727F2B81392E9A97C311029CE72455090",
    ),
    "figures/learned_two_phase_pilot": (
        RESULTS_ROOT / "figures" / "learned_two_phase_pilot",
        "directory",
        "DB20697EDE956EA571887B3EBB0D1B1FDBD7DC3EC456B59B27978B20D9697917",
    ),
    "EXPERIMENT_REPORT.md": (
        PROJECT_ROOT / "EXPERIMENT_REPORT.md",
        "file",
        "EBEA2E8E066CDB64E49DA33D26ED456C46E1E75ED3AAD56CD95BCA7DC0DC454B",
    ),
    "LEARNED_TWO_PHASE_PILOT_REPORT.md": (
        PROJECT_ROOT / "LEARNED_TWO_PHASE_PILOT_REPORT.md",
        "file",
        "FFBD3ADB4C50697623DEA90E40F6559278AF2EE5785430CC24009A6690EBB1A4",
    ),
    "summary.csv": (
        RESULTS_ROOT / "summary.csv",
        "file",
        "668C8F59586753D8EE08C352424C80E824C9628820A78069F7A433D00497841D",
    ),
    "summary_aggregated.csv": (
        RESULTS_ROOT / "summary_aggregated.csv",
        "file",
        "AF386F88F8052E27733D15D5A91A288AD0106E2E04F6CAF7F3E6D5A257195F91",
    ),
    "learned_two_phase_pilot_runs.csv": (
        RESULTS_ROOT / "learned_two_phase_pilot_runs.csv",
        "file",
        "33D07AD90FE7E6FEA0643BCECA4AA640E638A710D9C8C757EF53540F94C944BF",
    ),
    "learned_two_phase_pilot_summary.csv": (
        RESULTS_ROOT / "learned_two_phase_pilot_summary.csv",
        "file",
        "BE5F925DBFDDF63F2A911214B9C9231089D42E8EB32F93A8BB76C59286E72FD2",
    ),
    "learned_two_phase_pilot_comparison.csv": (
        RESULTS_ROOT / "learned_two_phase_pilot_comparison.csv",
        "file",
        "FCFF9D97BD7CE16A544BC93285B2C4333C751ECABD430787CA65F2D87911A7D9",
    ),
}

FIXED_CONFIG: dict[str, Any] = {
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
    "test_x_points": 128,
    "test_t_points": 201,
    "history_every": 10,
    "gradient_diagnostic_every": 250,
    "log_every": 250,
    "struct_time_slices": 4,
    "struct_x_points": 64,
    "length": 2.0 * math.pi,
    "reference_density": 1.0 / (2.0 * math.pi),
    "spectral_energy": math.sqrt(29.0),
    "cpu_threads": 2,
    "dtype": "float64",
}

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

REQUIRED_METRICS = (
    "training_seconds",
    "spinor_rel_l2",
    "final_spinor_rel_l2",
    "rho_rel_l2",
    "current_rel_l2",
    "fidelity_mean",
    "fidelity_min",
    "fidelity_final",
    "relative_phase_mae_rad",
    "relative_phase_rmse_rad",
    "branch_coefficient_rel_l2",
    "branch_occupation_rmse",
    "branch_plus_coefficient_abs_rms",
    "negative_branch_coefficient_abs_rms",
    "branch_plus_occupation_rmse",
    "branch_minus_occupation_rmse",
    "final_loss_total",
    "final_loss_pde",
    "final_loss_continuity",
    "final_loss_norm",
    "final_loss_energy",
    "final_loss_structure",
    *INTERNAL_METRICS,
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

DENSE_ARRAYS: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "x": ((128,), np.dtype(np.float64)),
    "t": ((201,), np.dtype(np.float64)),
    "psi_predicted": ((201, 128, 2), np.dtype(np.complex128)),
    "psi_exact": ((201, 128, 2), np.dtype(np.complex128)),
    "rho_predicted": ((201, 128), np.dtype(np.float64)),
    "rho_exact": ((201, 128), np.dtype(np.float64)),
    "current_predicted": ((201, 128), np.dtype(np.float64)),
    "current_exact": ((201, 128), np.dtype(np.float64)),
    "continuity": ((201, 128), np.dtype(np.float64)),
    "energy_density": ((201, 128), np.dtype(np.float64)),
    "relative_phase_predicted": ((201, 128), np.dtype(np.float64)),
    "relative_phase_exact": ((201, 128), np.dtype(np.float64)),
    "internal_channel_coefficient_predicted": (
        (201, 128, 2),
        np.dtype(np.complex128),
    ),
    "internal_channel_coefficient_exact": (
        (201, 128, 2),
        np.dtype(np.complex128),
    ),
    "internal_channel_amplitude_predicted": ((201, 128, 2), np.dtype(np.float64)),
    "internal_channel_amplitude_exact": ((201, 128, 2), np.dtype(np.float64)),
    "internal_channel_phase_unwrapped_predicted": (
        (201, 128, 2),
        np.dtype(np.float64),
    ),
    "internal_channel_phase_unwrapped_exact": (
        (201, 128, 2),
        np.dtype(np.float64),
    ),
    "internal_channel_phase_velocity_predicted": (
        (201, 128, 2),
        np.dtype(np.float64),
    ),
    "internal_channel_phase_velocity_exact": (
        (201, 128, 2),
        np.dtype(np.float64),
    ),
    "internal_channel_phase_curvature_predicted": (
        (201, 128, 2),
        np.dtype(np.float64),
    ),
    "internal_channel_phase_curvature_exact": (
        (201, 128, 2),
        np.dtype(np.float64),
    ),
    "internal_channel_basis": ((2, 2), np.dtype(np.float64)),
    "internal_channel_initial_coefficients": ((2,), np.dtype(np.float64)),
}


class Audit:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not bool(condition):
            self.failures.append(message)

    def capture(self, label: str, operation: Callable[[], Any]) -> Any:
        try:
            return operation()
        except Exception as exc:  # audit must continue and write its JSON
            self.failures.append(f"{label}: {type(exc).__name__}: {exc}")
            return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ) and bool(np.isfinite(value))


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return _finite(actual) and math.isclose(
            float(actual), expected, rel_tol=1.0e-12, abs_tol=1.0e-12
        )
    return actual == expected


def _close(actual: Any, expected: Any, *, atol: float = 1.0e-12) -> bool:
    if (
        isinstance(actual, (int, float, np.integer, np.floating))
        and not isinstance(actual, (bool, np.bool_))
        and isinstance(expected, (int, float, np.integer, np.floating))
        and not isinstance(expected, (bool, np.bool_))
        and bool(np.isnan(actual))
        and bool(np.isnan(expected))
    ):
        # Some diagnostics are structurally undefined outside their native
        # parameterization.  Pandas writes these aggregate cells as NaN; two
        # matching NaNs mean the missingness itself has been reproduced.
        return True
    return _finite(actual) and _finite(expected) and bool(
        np.isclose(float(actual), float(expected), rtol=1.0e-10, atol=atol)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _directory_sha256(root: Path) -> tuple[str, int]:
    """Hash sorted POSIX relative names as UTF-8 + NUL + exact file bytes."""

    if not root.is_dir():
        raise FileNotFoundError(f"missing directory: {root}")
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest().upper(), len(files)


def _run_dir(model: str, seed: int) -> Path:
    return RAW_ROOT / "case_M" / model / f"seed_{seed}"


def _check_inventory(audit: Audit) -> dict[str, Any]:
    expected_dirs = {_run_dir(model, seed) for model in ABLATION_MODELS for seed in ABLATION_SEEDS}
    actual_run_dirs = (
        {path.parent for path in RAW_ROOT.rglob("config.json")}
        if RAW_ROOT.is_dir()
        else set()
    )
    audit.require(
        actual_run_dirs == expected_dirs,
        "formal raw inventory is not exactly five models x three seeds: "
        f"missing={sorted(map(str, expected_dirs - actual_run_dirs))}, "
        f"extra={sorted(map(str, actual_run_dirs - expected_dirs))}",
    )
    expected_files = {
        run_dir / name for run_dir in expected_dirs for name in RUN_FILES
    }
    actual_files = (
        {path for path in RAW_ROOT.rglob("*") if path.is_file()}
        if RAW_ROOT.is_dir()
        else set()
    )
    audit.require(
        actual_files <= expected_files,
        "formal raw tree contains files outside the 15 declared six-piece runs: "
        f"{sorted(map(str, actual_files - expected_files))}",
    )
    complete = 0
    exact = 0
    per_run: dict[str, Any] = {}
    for run_dir in sorted(expected_dirs, key=str):
        files = {path.name for path in run_dir.iterdir() if path.is_file()} if run_dir.is_dir() else set()
        nonempty = {
            name: (run_dir / name).is_file() and (run_dir / name).stat().st_size > 0
            for name in RUN_FILES
        }
        is_complete = all(nonempty.values())
        is_exact = files == set(RUN_FILES)
        complete += int(is_complete)
        exact += int(is_exact)
        key = run_dir.relative_to(RAW_ROOT).as_posix()
        per_run[key] = {
            "required_nonempty": nonempty,
            "files_exactly_six_piece_set": is_exact,
            "extra_files": sorted(files - set(RUN_FILES)),
        }
        audit.require(is_complete, f"{run_dir}: incomplete six-piece artifact set")
        audit.require(is_exact, f"{run_dir}: files are not exactly the declared six-piece set")
    return {
        "passed": actual_run_dirs == expected_dirs
        and actual_files == expected_files
        and complete == EXPECTED_RUN_COUNT
        and exact == EXPECTED_RUN_COUNT,
        "expected_runs": EXPECTED_RUN_COUNT,
        "found_run_directories": len(actual_run_dirs),
        "complete_six_piece_runs": complete,
        "exact_six_piece_runs": exact,
        "runs": per_run,
    }


def _check_config(config: dict[str, Any], model: str, seed: int) -> dict[str, Any]:
    exp = config.get("experiment_config", {})
    run = config.get("run_spec", {})
    sampling = config.get("sampling", {})
    parameterization = config.get("two_channel_parameterization", {})
    checks = {
        **{f"experiment_config.{key}": _same(exp.get(key), expected) for key, expected in FIXED_CONFIG.items()},
        "run.experiment": run.get("experiment") == EXPERIMENT,
        "run.case": run.get("case") == CASE,
        "run.eta": _same(run.get("eta"), ETA),
        "run.model": run.get("model") == model,
        "run.seed": run.get("seed") == seed,
        "run.steps": run.get("steps") == 5000,
        "run.lambda_struct": _same(run.get("lambda_struct"), 1.0),
        "run.name": run.get("run_name")
        == f"{EXPERIMENT}__case_M__{model}__seed_{seed}__lambda_1",
        "model_initialization_seed": config.get("model_initialization_seed") == seed,
        "initialization_protocol": config.get("initialization")
        == "Xavier normal for every linear weight; zero biases",
        "collocation_seed": sampling.get("collocation_generator_seed") == 100_000 + seed,
        "structure_seed": sampling.get("structure_generator_seed") == 200_000 + seed,
        "collocation_resampling": sampling.get("collocation_resampled_each_step") is True,
        "periodic_endpoint_excluded": sampling.get("periodic_spatial_quadrature_endpoint") is False,
        "no_exact_energy_carrier": parameterization.get(
            "uses_exact_energy_or_time_carrier_in_forward"
        )
        is False,
        "initialization_hash": isinstance(config.get("trainable_initial_state_sha256"), str)
        and bool(re.fullmatch(r"[0-9a-fA-F]{64}", config["trainable_initial_state_sha256"])),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _check_metrics(metrics: dict[str, Any], model: str, seed: int) -> dict[str, Any]:
    missing = sorted(set(REQUIRED_METRICS) - set(metrics))
    finite = not missing and all(_finite(metrics[key]) for key in REQUIRED_METRICS)
    identity = {
        "experiment": metrics.get("experiment") == EXPERIMENT,
        "case": metrics.get("case") == CASE,
        "eta": _same(metrics.get("eta"), ETA),
        "model": metrics.get("model") == model,
        "seed": metrics.get("seed") == seed,
        "steps": metrics.get("steps") == 5000,
        "parameter_count": metrics.get("parameter_count") == EXPECTED_PARAMETER_COUNT,
    }
    pde_only = False
    if not missing:
        pde_only = _close(metrics["final_loss_total"], metrics["final_loss_pde"], atol=1.0e-14) and all(
            _close(metrics[key], 0.0, atol=1.0e-15)
            for key in (
                "final_loss_continuity",
                "final_loss_norm",
                "final_loss_energy",
                "final_loss_structure",
            )
        )
    internal_semantics = False
    if not missing:
        internal_semantics = (
            metrics["internal_initial_coefficient_min_abs"] > 0.0
            and metrics["internal_basis_orthogonality_max_abs"] < 1.0e-12
            and metrics["internal_reconstruction_max_abs"] < 1.0e-12
            and all(
                0.0 <= metrics[f"internal_channel_{channel}_near_zero_fraction"] <= 1.0
                and 0.0
                <= metrics[f"internal_channel_{channel}_phase_valid_link_fraction"]
                <= 1.0
                for channel in (1, 2)
            )
        )
    return {
        "passed": not missing and finite and all(identity.values()) and pde_only and internal_semantics,
        "missing_required_metrics": missing,
        "required_metrics_finite": finite,
        "identity": identity,
        "pde_only": pde_only,
        "internal_diagnostic_semantics": internal_semantics,
    }


def _check_loss_history(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    columns = ("step", "total", "pde", "continuity", "norm", "energy", "structure", "elapsed_seconds")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        return {"passed": False, "rows": len(frame), "missing_columns": missing}
    expected_steps = np.asarray([1, *range(10, 5001, 10)], dtype=np.int64)
    schedule = len(frame) == 501 and np.array_equal(frame["step"].to_numpy(), expected_steps)
    values = frame[list(columns)].to_numpy(dtype=np.float64)
    finite = bool(np.isfinite(values).all())
    pde_only = bool(
        np.allclose(frame["total"], frame["pde"], rtol=0.0, atol=1.0e-14)
        and np.allclose(
            frame[["continuity", "norm", "energy", "structure"]],
            0.0,
            rtol=0.0,
            atol=1.0e-15,
        )
    )
    return {
        "passed": schedule and finite and pde_only,
        "rows": len(frame),
        "schedule_is_step_1_then_every_10_through_5000": schedule,
        "required_values_finite": finite,
        "pde_only": pde_only,
    }


def _check_time_series(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    missing = sorted(set(TIME_SERIES_COLUMNS) - set(frame.columns))
    finite = False
    time_grid = False
    if not missing:
        finite = bool(np.isfinite(frame[list(TIME_SERIES_COLUMNS)].to_numpy(float)).all())
        time_grid = len(frame) == 201 and bool(
            np.allclose(frame["t"], np.linspace(0.0, 5.0, 201), rtol=0.0, atol=1.0e-12)
        )
    return {
        "passed": not missing and finite and time_grid,
        "rows": len(frame),
        "missing_internal_or_physical_columns": missing,
        "required_values_finite": finite,
        "time_grid_201": time_grid,
    }


def _check_dense(path: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    internal_semantics = False
    with np.load(path, allow_pickle=False) as dense:
        for name, (shape, dtype) in DENSE_ARRAYS.items():
            if name not in dense.files:
                checks[name] = {"present": False}
                continue
            array = dense[name]
            finite = bool(
                np.isfinite(array.real).all() and np.isfinite(array.imag).all()
                if np.iscomplexobj(array)
                else np.isfinite(array).all()
            )
            checks[name] = {
                "present": True,
                "shape": list(array.shape),
                "shape_ok": array.shape == shape,
                "dtype": str(array.dtype),
                "dtype_ok": array.dtype == dtype,
                "finite": finite,
            }
        grid = (
            "x" in dense.files
            and "t" in dense.files
            and np.allclose(dense["x"], 2.0 * math.pi * np.arange(128) / 128, rtol=0.0, atol=1.0e-12)
            and np.allclose(dense["t"], np.linspace(0.0, 5.0, 201), rtol=0.0, atol=1.0e-12)
        )
        required_internal = {
            "psi_predicted",
            "psi_exact",
            "internal_channel_coefficient_predicted",
            "internal_channel_coefficient_exact",
            "internal_channel_basis",
            "internal_channel_initial_coefficients",
        }
        if required_internal <= set(dense.files):
            basis = dense["internal_channel_basis"]
            initial = dense["internal_channel_initial_coefficients"]
            internal_semantics = bool(
                np.max(np.abs(basis.T @ basis - np.eye(2))) < 1.0e-12
                and np.min(np.abs(initial)) > 0.0
                and np.allclose(
                    dense["internal_channel_coefficient_predicted"][0],
                    initial[None, :],
                    rtol=0.0,
                    atol=2.0e-14,
                )
                and np.allclose(
                    dense["internal_channel_coefficient_exact"][0],
                    initial[None, :],
                    rtol=0.0,
                    atol=2.0e-14,
                )
                and np.allclose(
                    dense["psi_predicted"][0],
                    dense["psi_exact"][0],
                    rtol=0.0,
                    atol=2.0e-14,
                )
            )
    passed = bool(grid) and internal_semantics and all(
        item.get("present", False)
        and item.get("shape_ok", False)
        and item.get("dtype_ok", False)
        and item.get("finite", False)
        for item in checks.values()
    )
    return {
        "passed": passed,
        "coordinate_grids": bool(grid),
        "internal_basis_initial_and_reconstruction_semantics": internal_semantics,
        "arrays": checks,
    }


def _check_checkpoint(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise TypeError("checkpoint state_dict is missing or invalid")
    trainable = {name: tensor for name, tensor in state.items() if name.startswith("mlp.")}
    parameter_count = sum(int(tensor.numel()) for tensor in trainable.values())
    dtype_ok = bool(trainable) and all(tensor.dtype == torch.float64 for tensor in trainable.values())
    finite = bool(trainable) and all(bool(torch.isfinite(tensor).all()) for tensor in trainable.values())
    metadata = (
        checkpoint.get("experiment_config") == config.get("experiment_config")
        and checkpoint.get("run_spec") == config.get("run_spec")
    )
    return {
        "passed": parameter_count == EXPECTED_PARAMETER_COUNT and dtype_ok and finite and metadata,
        "parameter_count": parameter_count,
        "parameter_count_ok": parameter_count == EXPECTED_PARAMETER_COUNT,
        "trainable_tensors_float64": dtype_ok,
        "trainable_tensors_finite": finite,
        "checkpoint_metadata_matches_config": metadata,
    }


def _check_runs(audit: Audit) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    configs: dict[tuple[str, int], dict[str, Any]] = {}
    metrics_payloads: dict[tuple[str, int], dict[str, Any]] = {}
    for model in ABLATION_MODELS:
        for seed in ABLATION_SEEDS:
            run_dir = _run_dir(model, seed)
            result: dict[str, Any] = {"model": model, "seed": seed, "directory": str(run_dir)}
            try:
                config = _read_json(run_dir / "config.json")
                metrics = _read_json(run_dir / "metrics.json")
                configs[(model, seed)] = config
                metrics_payloads[(model, seed)] = metrics
                result["config"] = _check_config(config, model, seed)
                result["metrics"] = _check_metrics(metrics, model, seed)
                result["loss_history"] = _check_loss_history(run_dir / "loss_history.csv")
                result["time_series"] = _check_time_series(run_dir / "time_series.csv")
                result["dense_grid"] = _check_dense(run_dir / "dense_grid.npz")
                result["checkpoint"] = _check_checkpoint(run_dir / "model_state.pt", config)
                result["passed"] = all(
                    bool(result[name].get("passed"))
                    for name in ("config", "metrics", "loss_history", "time_series", "dense_grid", "checkpoint")
                )
                audit.require(result["passed"], f"{run_dir}: one or more run-integrity checks failed")
            except Exception as exc:
                result.update({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
                audit.failures.append(f"{run_dir}: {type(exc).__name__}: {exc}")
            results.append(result)

    paired: dict[str, Any] = {}
    for seed in ABLATION_SEEDS:
        available = all((model, seed) in configs for model in ABLATION_MODELS)
        hashes = {
            configs[(model, seed)].get("trainable_initial_state_sha256")
            for model in ABLATION_MODELS
            if (model, seed) in configs
        }
        collocation = {
            configs[(model, seed)].get("sampling", {}).get("collocation_generator_seed")
            for model in ABLATION_MODELS
            if (model, seed) in configs
        }
        passed = available and len(hashes) == 1 and collocation == {100_000 + seed}
        paired[str(seed)] = {
            "passed": passed,
            "all_five_configs_available": available,
            "initialization_hashes": sorted(str(value) for value in hashes),
            "collocation_seeds": sorted(str(value) for value in collocation),
        }
        audit.require(passed, f"seed {seed}: initialization hash or collocation stream is not matched across all five models")
    experiment_configs = {
        json.dumps(config.get("experiment_config", {}), sort_keys=True)
        for config in configs.values()
    }
    all_configs_available = len(configs) == EXPECTED_RUN_COUNT
    fixed_across_runs = all_configs_available and len(experiment_configs) == 1
    audit.require(
        fixed_across_runs,
        "all 15 runs do not share one identical fixed experiment_config",
    )
    results.append({"paired_rng_checks": paired, "passed": all(item["passed"] for item in paired.values())})
    results.append(
        {
            "fixed_experiment_config_across_all_runs": fixed_across_runs,
            "available_config_count": len(configs),
            "distinct_experiment_config_count": len(experiment_configs),
            "passed": fixed_across_runs,
        }
    )
    return results, configs, metrics_payloads


def _validate_nested(actual: Any, expected: Any, path: str = "decision") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"structure mismatch at {path}")
        for key in expected:
            _validate_nested(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"list mismatch at {path}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _validate_nested(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise ValueError(f"value mismatch at {path}: {actual!r} != {expected!r}")
        return
    if not _close(actual, expected):
        raise ValueError(f"numeric mismatch at {path}: {actual!r} != {expected!r}")


def _check_aggregates(metrics_payloads: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    for path in (*AGGREGATE_PATHS.values(), DECISION_PATH):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty aggregate artifact: {path}")
    runs = pd.read_csv(AGGREGATE_PATHS["runs"])
    expected_pairs = {(model, seed) for model in ABLATION_MODELS for seed in ABLATION_SEEDS}
    found_pairs = set(zip(runs.get("model", []), pd.to_numeric(runs.get("seed", []), errors="coerce"), strict=False))
    if len(runs) != 15 or found_pairs != expected_pairs:
        raise ValueError("runs CSV is not exactly the declared 15 model/seed pairs")
    if set(runs.get("experiment", ())) != {EXPERIMENT}:
        raise ValueError("runs CSV contains a non-ablation experiment")
    if set(runs.get("case", ())) != {CASE}:
        raise ValueError("runs CSV contains a case other than Case M")
    if set(pd.to_numeric(runs.get("steps", ()), errors="coerce")) != {5000}:
        raise ValueError("runs CSV contains a training budget other than 5000 steps")
    if set(pd.to_numeric(runs.get("parameter_count", ()), errors="coerce")) != {
        EXPECTED_PARAMETER_COUNT
    }:
        raise ValueError("runs CSV contains a parameter count other than 12,996")
    if len(metrics_payloads) != 15:
        raise ValueError("not all 15 metrics JSON payloads were available for aggregate verification")
    for model, seed in expected_pairs:
        row = runs[(runs["model"] == model) & (runs["seed"].astype(int) == seed)]
        if len(row) != 1:
            raise ValueError(f"runs CSV lacks unique row for {model}/seed {seed}")
        payload = metrics_payloads[(model, seed)]
        for key, expected in payload.items():
            if key not in row.columns or not _finite(expected):
                continue
            if not _close(row.iloc[0][key], expected):
                raise ValueError(f"runs CSV differs from metrics JSON for {model}/seed {seed}/{key}")

    excluded = {"seed", "eta", "lambda_struct", "steps"}
    numeric = [name for name in runs.select_dtypes(include="number").columns if name not in excluded]
    summary = pd.read_csv(AGGREGATE_PATHS["summary"])
    if len(summary) != 5 or tuple(summary.get("model", ())) != tuple(ABLATION_MODELS):
        raise ValueError("summary CSV is not exactly five models in declared order")
    for model in ABLATION_MODELS:
        source = runs[runs["model"] == model]
        row = summary[summary["model"] == model]
        if len(row) != 1 or int(row.iloc[0].get("n_seeds", -1)) != 3:
            raise ValueError(f"summary lacks one n=3 row for {model}")
        for metric in numeric:
            raw_std = source[metric].std(ddof=0)
            expected_std = 0.0 if pd.isna(raw_std) else float(raw_std)
            for suffix, expected in (
                ("mean", source[metric].mean()),
                # aggregate_ablation deliberately applies fillna(0.0) to
                # population SD while leaving an all-missing mean as NaN.
                ("std", expected_std),
            ):
                column = f"{metric}_{suffix}"
                if column not in summary or not _close(row.iloc[0][column], expected):
                    raise ValueError(f"summary cannot be reproduced for {model}/{column}")

    comparisons = pd.read_csv(AGGREGATE_PATHS["comparisons"])
    if len(comparisons) != len(ABLATION_MODELS) * len(numeric):
        raise ValueError("comparisons CSV has the wrong row count")
    reference = summary[summary["model"] == "ltpb_cpa"].iloc[0]
    for model in ABLATION_MODELS:
        source = summary[summary["model"] == model].iloc[0]
        for metric in numeric:
            row = comparisons[(comparisons["model"] == model) & (comparisons["metric"] == metric)]
            if len(row) != 1:
                raise ValueError(f"comparisons lacks unique row for {model}/{metric}")
            if str(row.iloc[0].get("case")) != CASE or int(
                row.iloc[0].get("n_seeds", -1)
            ) != 3:
                raise ValueError(f"comparison metadata is wrong for {model}/{metric}")
            mean = float(source[f"{metric}_mean"])
            sd = float(source[f"{metric}_std"])
            ref = float(reference[f"{metric}_mean"])
            ratio = mean / ref if ref != 0.0 else (0.0 if mean == 0.0 else float("inf"))
            actual_ratio = float(row.iloc[0]["ratio_to_ltpb"])
            if not _close(row.iloc[0]["mean"], mean) or not _close(row.iloc[0]["population_sd"], sd):
                raise ValueError(f"comparison mean/SD is not reproducible for {model}/{metric}")
            if math.isnan(ratio):
                if not math.isnan(actual_ratio):
                    raise ValueError(
                        f"comparison ratio should be NaN for {model}/{metric}"
                    )
            elif math.isfinite(ratio):
                if not _close(actual_ratio, ratio):
                    raise ValueError(f"comparison ratio is not reproducible for {model}/{metric}")
            elif not math.isinf(actual_ratio):
                raise ValueError(f"comparison ratio should be infinite for {model}/{metric}")

    decision = _read_json(DECISION_PATH)
    recomputed = compute_mechanism_decision(runs, summary)
    _validate_nested(decision, recomputed)
    return {
        "passed": True,
        "csv_files": {key: str(path) for key, path in AGGREGATE_PATHS.items()},
        "runs": len(runs),
        "summary_rows": len(summary),
        "comparison_rows": len(comparisons),
        "decision_recomputed_exactly": True,
        "strong_mechanism_support": decision.get("strong_mechanism_support"),
    }


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def _source_runs() -> pd.DataFrame:
    frame = pd.read_csv(AGGREGATE_PATHS["runs"])
    return frame.sort_values(["model", "seed"]).reset_index(drop=True)


def _run_values(runs: pd.DataFrame, model: str, metric: str) -> np.ndarray:
    subset = runs[runs["model"] == model].sort_values("seed")
    if tuple(subset["seed"].astype(int)) != ABLATION_SEEDS:
        raise ValueError(f"source recomputation lacks seeds for {model}/{metric}")
    if metric == "infidelity_mean":
        return np.maximum(0.0, 1.0 - subset["fidelity_mean"].to_numpy(float))
    return subset[metric].to_numpy(float)


def _compare_source_values(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    keys: tuple[str, ...],
    numeric: tuple[str, ...],
    text: tuple[str, ...] = (),
) -> None:
    left = actual.sort_values(list(keys), kind="stable").reset_index(drop=True)
    right = expected.sort_values(list(keys), kind="stable").reset_index(drop=True)
    if len(left) != len(right):
        raise ValueError(f"source evidence row mismatch: {len(left)} != {len(right)}")
    for key in keys:
        if pd.api.types.is_numeric_dtype(right[key]):
            if not np.allclose(
                left[key].to_numpy(float), right[key].to_numpy(float),
                rtol=0.0, atol=1.0e-11,
            ):
                raise ValueError(f"source evidence key mismatch: {key}")
        elif not np.array_equal(left[key].astype(str), right[key].astype(str)):
            raise ValueError(f"source evidence key mismatch: {key}")
    for column in numeric:
        if not np.allclose(
            left[column].to_numpy(float), right[column].to_numpy(float),
            rtol=5.0e-9, atol=2.0e-11,
        ):
            delta = float(
                np.max(np.abs(left[column].to_numpy(float) - right[column].to_numpy(float)))
            )
            raise ValueError(f"source evidence cannot be recomputed: {column}, max delta={delta}")
    for column in text:
        if not np.array_equal(left[column].astype(str), right[column].astype(str)):
            raise ValueError(f"source evidence text mismatch: {column}")


def _expected_metric_source(name: str, runs: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    rows: list[dict[str, Any]] = []
    if name == "01_core_error_comparison":
        for metric in CORE_METRICS:
            reference_mean = float(np.mean(_run_values(runs, "ltpb_cpa", metric)))
            for model in ABLATION_MODELS:
                values = _run_values(runs, model, metric)
                mean, sd = float(np.mean(values)), float(np.std(values, ddof=0))
                for seed, value in zip(ABLATION_SEEDS, values, strict=True):
                    rows.append({
                        "model": model, "model_label": MODEL_LABELS[model],
                        "seed": seed, "metric": metric,
                        "metric_label": CORE_METRIC_LABELS[metric],
                        "value": value, "group_mean": mean, "population_sd": sd,
                        "group_ratio_to_ltpb": mean / max(reference_mean, PLOT_FLOOR),
                    })
        return pd.DataFrame(rows), ("model", "seed", "metric"), (
            "value", "group_mean", "population_sd", "group_ratio_to_ltpb",
        ), ("model_label", "metric_label")

    if name == "02_per_seed_summary":
        for metric in SUPPORT_METRICS:
            reference = _run_values(runs, "ltpb_cpa", metric)
            reference_mean = float(np.mean(reference))
            for model in ABLATION_MODELS:
                values = _run_values(runs, model, metric)
                mean, sd = float(np.mean(values)), float(np.std(values, ddof=0))
                for seed, value, same_seed in zip(ABLATION_SEEDS, values, reference, strict=True):
                    rows.append({
                        "model": model, "model_label": MODEL_LABELS[model],
                        "seed": seed, "metric": metric,
                        "metric_label": SUPPORT_METRIC_LABELS[metric],
                        "value": value, "ltpb_same_seed_value": same_seed,
                        "paired_ratio_with_plot_floor": max(value, PLOT_FLOOR) / max(same_seed, PLOT_FLOOR),
                        "group_mean": mean, "population_sd": sd,
                        "ltpb_group_mean": reference_mean,
                        "group_ratio_with_plot_floor": max(mean, PLOT_FLOOR) / max(reference_mean, PLOT_FLOOR),
                        "plot_floor": PLOT_FLOOR,
                    })
        return pd.DataFrame(rows), ("model", "seed", "metric"), (
            "value", "ltpb_same_seed_value", "paired_ratio_with_plot_floor",
            "group_mean", "population_sd", "ltpb_group_mean",
            "group_ratio_with_plot_floor", "plot_floor",
        ), ("model_label", "metric_label")

    if name == "05_rotation_degradation":
        component_angle = -0.5 * math.degrees(math.atan2(2.0, 5.0))
        models = (
            ("ltpb_cpa", 0.0, "preregistered_rotation_path"),
            ("rotated_ltpb_15", 15.0, "preregistered_rotation_path"),
            ("rotated_ltpb_30", 30.0, "preregistered_rotation_path"),
            ("component_polar", component_angle, "component_basis_special_case"),
        )
        for metric in CORE_METRICS:
            reference = _run_values(runs, "ltpb_cpa", metric)
            reference_mean = float(np.mean(reference))
            threshold = ROTATION_30_RATIO_MIN * reference_mean
            for model, angle, role in models:
                values = _run_values(runs, model, metric)
                mean, sd = float(np.mean(values)), float(np.std(values, ddof=0))
                for seed, value, same_seed in zip(ABLATION_SEEDS, values, reference, strict=True):
                    rows.append({
                        "metric": metric, "metric_label": CORE_METRIC_LABELS[metric],
                        "model": model, "seed": seed,
                        "angle_degrees": angle, "role": role, "value": value,
                        "group_mean": mean, "population_sd": sd,
                        "ltpb_same_seed_value": same_seed,
                        "paired_ratio_to_ltpb": value / max(same_seed, PLOT_FLOOR),
                        "group_ratio_to_ltpb": mean / max(reference_mean, PLOT_FLOOR),
                        "three_x_ltpb_mean_threshold": threshold,
                    })
        return pd.DataFrame(rows), ("metric", "model", "seed"), (
            "angle_degrees", "value", "group_mean", "population_sd",
            "ltpb_same_seed_value", "paired_ratio_to_ltpb",
            "group_ratio_to_ltpb", "three_x_ltpb_mean_threshold",
        ), ("metric_label", "role")

    if name == "06_amplitude_phase_complexity":
        for metric, exact_metric in COMPLEXITY_METRICS:
            for model in ABLATION_MODELS:
                predicted = _run_values(runs, model, metric)
                exact = _run_values(runs, model, exact_metric)
                predicted_mean, predicted_sd = float(np.mean(predicted)), float(np.std(predicted, ddof=0))
                exact_mean, exact_sd = float(np.mean(exact)), float(np.std(exact, ddof=0))
                for seed, value, exact_value in zip(ABLATION_SEEDS, predicted, exact, strict=True):
                    rows.append({
                        "model": model, "model_label": MODEL_LABELS[model],
                        "seed": seed, "metric": metric,
                        "exact_metric": exact_metric, "predicted": value,
                        "metric_label": COMPLEXITY_METRIC_LABELS[metric],
                        "exact_representation": exact_value,
                        "predicted_group_mean": predicted_mean,
                        "predicted_population_sd": predicted_sd,
                        "exact_group_mean": exact_mean,
                        "exact_population_sd": exact_sd,
                    })
        return pd.DataFrame(rows), ("model", "seed", "metric"), (
            "predicted", "exact_representation", "predicted_group_mean",
            "predicted_population_sd", "exact_group_mean", "exact_population_sd",
        ), ("model_label", "exact_metric", "metric_label")
    raise ValueError(name)


def _expected_time_source(name: str) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    phase = name == "04_relative_phase_x0"
    predicted_column = "relative_phase_x0_predicted" if phase else "current_x0_predicted"
    exact_column = "relative_phase_x0_exact" if phase else "current_x0_exact"
    rows: list[dict[str, Any]] = []
    for model in ABLATION_MODELS:
        frames = [pd.read_csv(_run_dir(model, seed) / "time_series.csv") for seed in ABLATION_SEEDS]
        t = frames[0]["t"].to_numpy(float)
        predicted = np.stack([frame[predicted_column].to_numpy(float) for frame in frames])
        exact = np.stack([frame[exact_column].to_numpy(float) for frame in frames])
        if not all(np.allclose(frame["t"], t, rtol=0.0, atol=1.0e-12) for frame in frames):
            raise ValueError(f"misaligned time grids for {model}")
        display_predicted = predicted.copy()
        display_exact = exact.copy()
        circular_error = np.zeros_like(predicted)
        shifts = np.zeros(3)
        if phase:
            for index in range(3):
                display_exact[index] = np.unwrap(exact[index])
                circular_error[index] = np.angle(np.exp(1j * (predicted[index] - exact[index])))
                display_predicted[index] = display_exact[index] + np.unwrap(circular_error[index])
            reference = display_predicted[0]
            for index in (1, 2):
                winding = int(np.rint(np.median(reference - display_predicted[index]) / (2.0 * math.pi)))
                shifts[index] = 2.0 * math.pi * winding
                display_predicted[index] += shifts[index]
        mean = np.mean(display_predicted, axis=0)
        sd = np.std(display_predicted, axis=0, ddof=0)
        for index, seed in enumerate(ABLATION_SEEDS):
            for time_index, time in enumerate(t):
                rows.append({
                    "model": model, "model_label": MODEL_LABELS[model],
                    "seed": seed, "t": time,
                    "predicted_raw": predicted[index, time_index],
                    "exact_raw": exact[index, time_index],
                    "predicted_display": display_predicted[index, time_index],
                    "exact_display": display_exact[index, time_index],
                    "circular_error": circular_error[index, time_index],
                    "seed_display_branch_shift_rad": shifts[index],
                    "group_mean_display": mean[time_index],
                    "population_sd_display": sd[time_index],
                })
    return pd.DataFrame(rows), ("model", "seed", "t"), (
        "predicted_raw", "exact_raw", "predicted_display", "exact_display",
        "circular_error", "seed_display_branch_shift_rad", "group_mean_display",
        "population_sd_display",
    ), ("model_label",)


def _expected_loss_source() -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    rows: list[dict[str, Any]] = []
    for model in ABLATION_MODELS:
        frames = [pd.read_csv(_run_dir(model, seed) / "loss_history.csv") for seed in ABLATION_SEEDS]
        steps = frames[0]["step"].to_numpy(int)
        losses = np.stack([frame["pde"].to_numpy(float) for frame in frames])
        if not all(np.array_equal(frame["step"].to_numpy(int), steps) for frame in frames):
            raise ValueError(f"misaligned loss steps for {model}")
        mean, sd = np.mean(losses, axis=0), np.std(losses, axis=0, ddof=0)
        for index, seed in enumerate(ABLATION_SEEDS):
            for step_index, step in enumerate(steps):
                rows.append({
                    "model": model, "model_label": MODEL_LABELS[model],
                    "seed": seed, "step": step,
                    "pde_loss": losses[index, step_index],
                    "group_mean": mean[step_index], "population_sd": sd[step_index],
                })
    return pd.DataFrame(rows), ("model", "seed", "step"), (
        "pde_loss", "group_mean", "population_sd",
    ), ("model_label",)


def _check_source_csv(name: str, path: Path) -> dict[str, Any]:
    if path.name != f"{name}.csv":
        raise ValueError(f"fixed source filename mismatch for {name}: {path.name}")
    frame = pd.read_csv(path)
    schema = SOURCE_SCHEMAS[name]
    if tuple(frame.columns) != schema:
        raise ValueError(f"fixed source schema mismatch for {name}: {tuple(frame.columns)}")
    if len(frame) != SOURCE_ROWS[name]:
        raise ValueError(f"fixed source row count mismatch for {name}: {len(frame)}")
    if name in {"01_core_error_comparison", "02_per_seed_summary", "05_rotation_degradation", "06_amplitude_phase_complexity"}:
        expected, keys, numeric, text = _expected_metric_source(name, _source_runs())
    elif name in {"03_current_x0", "04_relative_phase_x0"}:
        expected, keys, numeric, text = _expected_time_source(name)
    else:
        expected, keys, numeric, text = _expected_loss_source()
    if bool(frame.duplicated(list(keys)).any()):
        raise ValueError(f"duplicate source evidence keys for {name}")
    numeric_frame = frame.select_dtypes(include="number")
    if numeric_frame.empty or not np.isfinite(numeric_frame.to_numpy(float)).all():
        raise ValueError(f"non-finite numeric source evidence for {name}")
    _compare_source_values(frame, expected, keys, numeric, text)
    return {
        "passed": True,
        "path": str(path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "fixed_schema": True,
        "unique_keys": list(keys),
        "all_numeric_values_finite": True,
        "raw_values_and_population_statistics_recomputed": True,
    }


def _check_figure_file(path: Path, extension: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size < 100:
        return {"passed": False, "exists_nontrivial": False}
    prefix = path.read_bytes()[:1024].lstrip()
    signature = {
        "png": prefix.startswith(b"\x89PNG\r\n\x1a\n"),
        "pdf": prefix.startswith(b"%PDF-"),
        "svg": b"<svg" in prefix,
    }[extension]
    editable_text = True
    details: dict[str, Any] = {}
    if extension == "png":
        from PIL import Image

        expected_width = 183.0 / 25.4 * 600.0
        with Image.open(path) as raster:
            raster.load()
            width, height = raster.size
            dpi = raster.info.get("dpi")
        dpi_ok = (
            isinstance(dpi, tuple)
            and len(dpi) >= 2
            and all(abs(float(value) - 600.0) <= 5.0 for value in dpi[:2])
        )
        width_ok = abs(width - expected_width) <= max(5.0, expected_width * 0.005)
        signature = signature and dpi_ok and width_ok and height > 100
        details.update(
            {
                "pixel_width": width,
                "pixel_height": height,
                "expected_183mm_at_600dpi_width": expected_width,
                "width_matches": width_ok,
                "dpi": list(dpi) if isinstance(dpi, tuple) else dpi,
                "dpi_approximately_600": dpi_ok,
            }
        )
    if extension == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        extracted = "\n".join((page.extract_text() or "") for page in reader.pages)
        editable_text = len(extracted.strip()) >= 20
        details.update(
            {
                "pages": len(reader.pages),
                "extractable_text_characters": len(extracted.strip()),
                "editable_or_extractable_text": editable_text,
            }
        )
    if extension == "svg":
        text = path.read_text(encoding="utf-8", errors="strict")
        editable_text = re.search(r"<text(?:\s|>)", text) is not None
    result = {
        "passed": signature and editable_text,
        "exists_nontrivial": True,
        "signature_ok": signature,
        "editable_svg_text": editable_text,
        "bytes": path.stat().st_size,
    }
    result.update(details)
    return result


def _check_figures() -> dict[str, Any]:
    manifest_path = FIGURE_ROOT / "figure_manifest.json"
    manifest = _read_json(manifest_path)
    figures = manifest.get("figures")
    if not isinstance(figures, dict) or tuple(figures) != EXPECTED_FIGURES:
        raise ValueError(
            f"manifest figure groups must be exactly {EXPECTED_FIGURES}, found={tuple(figures or ())}"
        )
    results: dict[str, Any] = {}
    linked_sources: set[Path] = set()
    for name in EXPECTED_FIGURES:
        entry = figures[name]
        if not isinstance(entry, dict):
            raise TypeError(f"invalid manifest entry for {name}")
        exports: dict[str, Any] = {}
        files = entry.get("files")
        if not isinstance(files, dict):
            raise TypeError(f"manifest files object missing for {name}")
        for extension in ("png", "pdf", "svg"):
            relative = files.get(extension)
            if not isinstance(relative, str):
                raise ValueError(f"manifest {name} lacks {extension} export")
            path = _inside(FIGURE_ROOT, relative)
            if path.suffix.lower() != f".{extension}":
                raise ValueError(f"manifest extension mismatch for {name}/{extension}")
            exports[extension] = _check_figure_file(path, extension)
        sources: list[dict[str, Any]] = []
        source_entries = entry.get("source_data")
        if not isinstance(source_entries, list) or not source_entries:
            raise ValueError(f"figure {name} has no source-data CSV")
        for relative in source_entries:
            path = _inside(FIGURE_ROOT, str(relative))
            if path.suffix.lower() != ".csv" or not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"figure {name} source data is not a nonempty CSV: {path}")
            linked_sources.add(path.resolve())
            sources.append(_check_source_csv(name, path))
        passed = all(item["passed"] for item in exports.values()) and bool(sources)
        if not passed:
            raise ValueError(f"figure {name} failed export/editable-text checks")
        results[name] = {"passed": passed, "exports": exports, "source_data": sources}

    actual_sources = {path.resolve() for path in (FIGURE_ROOT / "source_data").glob("*.csv")}
    if len(linked_sources) != 7 or actual_sources != linked_sources:
        raise ValueError(
            "figure source-data inventory must be exactly seven linked CSV files: "
            f"linked={len(linked_sources)}, actual={len(actual_sources)}"
        )
    exported_stems = {
        path.stem
        for path in FIGURE_ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".svg"}
    }
    if exported_stems != set(EXPECTED_FIGURES):
        raise ValueError(f"figure export groups mismatch: {sorted(exported_stems)}")
    return {
        "passed": True,
        "groups": results,
        "group_count": len(results),
        "export_count": len(results) * 3,
        "source_csv_count": len(linked_sources),
        "manifest": str(manifest_path),
    }


def _selected_protected_paths(selection: dict[str, Any]) -> set[str]:
    excluded = tuple(str(item) for item in selection.get("excluded_prefixes", ()))

    def included(relative: str) -> bool:
        return not any(relative.startswith(prefix) for prefix in excluded)

    selected: set[str] = set()
    roots = selection.get("recursive_roots")
    if not isinstance(roots, list):
        raise TypeError("protected baseline selection.recursive_roots must be a list")
    for relative_root in roots:
        root = _inside(PROJECT_ROOT, str(relative_root))
        if not root.is_dir():
            raise FileNotFoundError(f"protected recursive root is missing: {root}")
        for path in root.rglob("*"):
            if path.is_file():
                relative = path.relative_to(PROJECT_ROOT).as_posix()
                if included(relative):
                    selected.add(relative)

    if selection.get("top_level_results_files") is True:
        excluded_top = str(selection.get("excluded_top_level_prefix", ""))
        for path in RESULTS_ROOT.iterdir():
            if not path.is_file():
                continue
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if included(relative) and not relative.startswith(excluded_top):
                selected.add(relative)

    reports = selection.get("protected_reports")
    if not isinstance(reports, list):
        raise TypeError("protected baseline selection.protected_reports must be a list")
    for relative in reports:
        canonical = _inside(PROJECT_ROOT, str(relative)).relative_to(PROJECT_ROOT).as_posix()
        selected.add(canonical)
    return selected


def _check_baselines(audit: Audit) -> dict[str, Any]:
    """Strictly compare the frozen 268-file manifest and hard-coded aggregate."""

    try:
        baseline = _read_json(PROTECTED_BASELINE_PATH)
        if baseline.get("schema_version") != 1:
            raise ValueError("protected baseline schema_version must be 1")
        files = baseline.get("files")
        selection = baseline.get("selection")
        if not isinstance(files, dict) or not isinstance(selection, dict):
            raise TypeError("protected baseline must contain files and selection objects")
        declared = set(files)
        selected = _selected_protected_paths(selection)
        if selected != declared:
            raise ValueError(
                "protected path set changed: "
                f"missing={sorted(declared - selected)}, extra={sorted(selected - declared)}"
            )
        if baseline.get("file_count") != len(declared) or len(declared) != 268:
            raise ValueError(
                f"protected baseline file count is not the frozen 268: {len(declared)}"
            )

        per_file: dict[str, Any] = {}
        aggregate = hashlib.sha256()
        for relative in sorted(declared):
            metadata = files[relative]
            if not isinstance(metadata, dict):
                raise TypeError(f"invalid protected-file metadata: {relative}")
            path = _inside(PROJECT_ROOT, relative)
            if not path.is_file():
                raise FileNotFoundError(f"protected file is missing: {relative}")
            expected_size = metadata.get("size_bytes")
            expected_sha = str(metadata.get("sha256", "")).lower()
            actual_size = path.stat().st_size
            actual_sha = _file_sha256(path).lower()
            passed = actual_size == expected_size and actual_sha == expected_sha
            per_file[relative] = {
                "passed": passed,
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
            }
            if not passed:
                raise ValueError(
                    f"protected file changed: {relative}: "
                    f"size {actual_size}/{expected_size}, sha256 {actual_sha}/{expected_sha}"
                )
            aggregate.update(relative.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(bytes.fromhex(actual_sha))

        actual_aggregate = aggregate.hexdigest()
        manifest_aggregate = str(baseline.get("aggregate_sha256", "")).lower()
        if actual_aggregate != manifest_aggregate:
            raise ValueError(
                "protected per-file aggregate disagrees with its manifest: "
                f"{actual_aggregate} != {manifest_aggregate}"
            )
        if actual_aggregate != PROTECTED_BASELINE_AGGREGATE_SHA256:
            raise ValueError(
                "protected per-file aggregate disagrees with the hard-coded baseline: "
                f"{actual_aggregate} != {PROTECTED_BASELINE_AGGREGATE_SHA256}"
            )

        legacy_records = baseline.get("legacy_pre_turn_aggregate_records", {})
        legacy_directory_constants_match = all(
            str(legacy_records.get(f"results/{label}" if label.startswith(("raw/", "figures/")) else label, "")).upper()
            == expected
            for label, (_path, kind, expected) in BASELINE_SHA256.items()
            if kind == "directory"
        )
        legacy_file_constants_match = all(
            _file_sha256(path) == expected
            for _label, (path, kind, expected) in BASELINE_SHA256.items()
            if kind == "file"
        )
        legacy_constants_match = (
            legacy_directory_constants_match and legacy_file_constants_match
        )
        audit.require(
            legacy_constants_match,
            "protected baseline does not preserve the four supplied legacy directory SHA-256 records",
        )
        return {
            "passed": legacy_constants_match,
            "manifest": str(PROTECTED_BASELINE_PATH),
            "file_count": len(per_file),
            "path_set_exact": True,
            "sizes_and_sha256_exact": True,
            "aggregate_algorithm": baseline.get("aggregate_algorithm"),
            "manifest_aggregate_sha256": manifest_aggregate,
            "hard_coded_aggregate_sha256": PROTECTED_BASELINE_AGGREGATE_SHA256,
            "aggregate_exact": True,
            "legacy_directory_sha256_records_preserved": legacy_directory_constants_match,
            "legacy_file_sha256_constants_match": legacy_file_constants_match,
            "files": per_file,
        }
    except Exception as exc:
        audit.failures.append(f"protected legacy baseline: {type(exc).__name__}: {exc}")
        return {
            "passed": False,
            "manifest": str(PROTECTED_BASELINE_PATH),
            "hard_coded_aggregate_sha256": PROTECTED_BASELINE_AGGREGATE_SHA256,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _write_audit(payload: dict[str, Any]) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = AUDIT_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(AUDIT_PATH)


def main() -> None:
    audit = Audit()
    inventory = audit.capture("formal run inventory", lambda: _check_inventory(audit))
    run_checks, _configs, metrics_payloads = _check_runs(audit)
    aggregates = audit.capture(
        "aggregate tables and decision",
        lambda: _check_aggregates(metrics_payloads),
    )
    figures = audit.capture("figure exports and source data", _check_figures)
    baselines = _check_baselines(audit)

    payload = {
        "status": "PASS" if not audit.failures else "FAIL",
        "failures": audit.failures,
        "counts": {
            "expected_formal_runs": EXPECTED_RUN_COUNT,
            "checked_formal_runs": sum("model" in item for item in run_checks),
            "expected_six_piece_files": EXPECTED_RUN_COUNT * len(RUN_FILES),
            "expected_aggregate_csv_files": 3,
            "expected_figure_groups": 7,
            "expected_figure_exports": 21,
            "expected_figure_source_csv_files": 7,
            "protected_legacy_files": 268,
            "supplied_legacy_sha256_records": len(BASELINE_SHA256),
        },
        "formal_inventory": inventory,
        "run_checks": run_checks,
        "aggregate_checks": aggregates,
        "figure_checks": figures,
        "legacy_baseline_protection": baselines,
        "scope": {
            "training_started": False,
            "audit_output": str(AUDIT_PATH),
            "experiment": EXPERIMENT,
        },
    }
    _write_audit(payload)
    print(json.dumps(payload["counts"], indent=2))
    print(f"status={payload['status']}")
    print(f"audit={AUDIT_PATH}")
    if audit.failures:
        raise SystemExit(
            f"LTPB mechanism-ablation audit failed with {len(audit.failures)} issue(s); "
            f"see {AUDIT_PATH}"
        )


if __name__ == "__main__":
    main()
