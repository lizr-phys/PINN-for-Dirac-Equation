from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
PILOT_ROOT = RESULTS_ROOT / "raw" / "learned_two_phase_pilot"
FIGURE_ROOT = RESULTS_ROOT / "figures" / "learned_two_phase_pilot"
AUDIT_PATH = RESULTS_ROOT / "learned_two_phase_pilot_audit.json"

EXPERIMENT = "learned_two_phase_pilot"
MODEL = "ltpb_cpa"
CASES = {"P": 0.0, "M": 1.0}
SEEDS = (0, 1, 2)
EXPECTED_PARAMETER_COUNT = 12_996
EXPECTED_SPINOR_SHAPE = (201, 128, 2)
EXPECTED_SCALAR_FIELD_SHAPE = (201, 128)

REQUIRED_RUN_FILES = (
    "config.json",
    "metrics.json",
    "loss_history.csv",
    "time_series.csv",
    "dense_grid.npz",
    "model_state.pt",
)

COMMON_METRICS = {
    "training_seconds",
    "spinor_rel_l2",
    "final_spinor_rel_l2",
    "rho_rel_l2",
    "current_rel_l2",
    "continuity_rms",
    "continuity_normalized_rms",
    "max_norm_drift",
    "max_abs_energy_drift",
    "fidelity_mean",
    "fidelity_min",
    "fidelity_final",
    "relative_phase_mae_rad",
    "relative_phase_rmse_rad",
    "relative_phase_mask_fraction",
    "branch_coefficient_rel_l2",
    "branch_occupation_rmse",
    "branch_plus_coefficient_abs_rms",
    "negative_branch_coefficient_abs_rms",
    "branch_plus_occupation_rmse",
    "branch_minus_occupation_rmse",
    "target_mode_occupation_defect_mean",
    "final_loss_total",
    "final_loss_pde",
    "final_loss_continuity",
    "final_loss_norm",
    "final_loss_energy",
    "final_loss_structure",
}

BRANCH_METRICS = {
    "phase_plus_rmse_rad",
    "phase_minus_rmse_rad",
    "phase_plus_circular_rmse_rad",
    "phase_minus_circular_rmse_rad",
    "branch_relative_phase_rmse_rad",
    "branch_relative_phase_circular_rmse_rad",
    "phase_plus_slope",
    "phase_plus_slope_target",
    "phase_plus_slope_abs_error",
    "phase_plus_fit_intercept",
    "phase_plus_fit_r2",
    "phase_minus_slope",
    "phase_minus_slope_target",
    "phase_minus_slope_abs_error",
    "phase_minus_fit_intercept",
    "phase_minus_fit_r2",
    "amplitude_plus_drift_abs_rms",
    "amplitude_plus_drift_abs_max",
    "amplitude_plus_drift_rel_rms",
    "amplitude_plus_drift_rel_max",
    "amplitude_minus_drift_abs_rms",
    "amplitude_minus_drift_abs_max",
    "amplitude_minus_drift_rel_rms",
    "amplitude_minus_drift_rel_max",
}

LOSS_COLUMNS = ("total", "pde", "continuity", "norm", "energy", "structure")

TIME_SERIES_COLUMNS = {
    "t",
    "norm_predicted",
    "norm_exact",
    "energy_predicted",
    "energy_exact",
    "continuity_rms",
    "current_x0_predicted",
    "current_x0_exact",
    "relative_phase_x0_predicted",
    "relative_phase_x0_exact",
    "fidelity",
    "c_plus_predicted_real",
    "c_plus_predicted_imag",
    "c_plus_exact_real",
    "c_plus_exact_imag",
    "c_minus_predicted_real",
    "c_minus_predicted_imag",
    "c_minus_exact_real",
    "c_minus_exact_imag",
    "occupation_plus_predicted",
    "occupation_plus_exact",
    "occupation_minus_predicted",
    "occupation_minus_exact",
    "phase_plus_spatial_mean",
    "phase_plus_exact",
    "phase_minus_spatial_mean",
    "phase_minus_exact",
    "branch_relative_phase_spatial_mean",
    "branch_relative_phase_exact",
    "amplitude_plus_spatial_mean",
    "amplitude_plus_spatial_std",
    "amplitude_plus_exact",
    "amplitude_minus_spatial_mean",
    "amplitude_minus_spatial_std",
    "amplitude_minus_exact",
}

BASE_DENSE_FIELDS = {
    "x",
    "t",
    "psi_predicted",
    "psi_exact",
    "rho_predicted",
    "rho_exact",
    "current_predicted",
    "current_exact",
    "continuity",
    "energy_density",
    "relative_phase_predicted",
    "relative_phase_exact",
}

BRANCH_DENSE_FIELDS = {
    "amplitude_plus",
    "amplitude_minus",
    "amplitude_plus_exact",
    "amplitude_minus_exact",
    "phase_plus",
    "phase_minus",
    "phase_plus_exact",
    "phase_minus_exact",
    "branch_relative_phase",
    "branch_relative_phase_exact",
}

EXPECTED_FIGURES = (
    "01_main_metric_comparison",
    "02_learned_branch_phases",
    "03_case_M_current_x0",
    "04_case_M_relative_phase_x0",
    "05_amplitude_and_occupations",
    "06_loss_histories",
)


def _run_dir(case: str, seed: int) -> Path:
    return PILOT_ROOT / f"case_{case}" / MODEL / f"seed_{seed}"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object, found {type(payload).__name__}")
    return payload


def _write_audit(payload: dict[str, Any]) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(value))
    )


def _close(value: Any, expected: float, *, atol: float = 1.0e-12) -> bool:
    return _is_finite_number(value) and _is_finite_number(expected) and bool(
        np.isclose(float(value), expected, rtol=0.0, atol=atol)
    )


def _config_checks(config: dict[str, Any], case: str, seed: int) -> dict[str, bool]:
    exp = config.get("experiment_config", {})
    run = config.get("run_spec", {})
    sampling = config.get("sampling", {})
    expected_eta = CASES[case]
    expected_name = f"{EXPERIMENT}__case_{case}__{MODEL}__seed_{seed}__lambda_1"
    return {
        "mass": _close(exp.get("mass"), 5.0),
        "wave_number": _close(exp.get("wave_number"), 2.0),
        "t_max": _close(exp.get("t_max"), 5.0),
        "x_min": _close(exp.get("x_min"), -math.pi),
        "x_max": _close(exp.get("x_max"), math.pi),
        "dtype": exp.get("dtype") == "float64",
        "hidden_layers": exp.get("hidden_layers") == 4,
        "hidden_width": exp.get("hidden_width") == 64,
        "collocation_points": exp.get("collocation_points") == 1024,
        "adam_steps": exp.get("adam_steps") == 5000,
        "learning_rate": _close(exp.get("learning_rate"), 1.0e-3),
        "test_t_points": exp.get("test_t_points") == 201,
        "test_x_points": exp.get("test_x_points") == 128,
        "experiment": run.get("experiment") == EXPERIMENT,
        "case": run.get("case") == case,
        "eta": _close(run.get("eta"), expected_eta),
        "model": run.get("model") == MODEL,
        "seed": run.get("seed") == seed,
        "steps": run.get("steps") == 5000,
        "lambda_struct": _close(run.get("lambda_struct"), 1.0),
        "run_name": run.get("run_name") == expected_name,
        "initialization": config.get("initialization")
        == "Xavier normal for every linear weight; zero biases",
        "collocation_generator_seed": sampling.get("collocation_generator_seed")
        == 100_000 + seed,
        "structure_generator_seed": sampling.get("structure_generator_seed")
        == 200_000 + seed,
        "collocation_resampled_each_step": sampling.get(
            "collocation_resampled_each_step"
        )
        is True,
        "periodic_endpoint_excluded": sampling.get(
            "periodic_spatial_quadrature_endpoint"
        )
        is False,
    }


def _reference_sampling_checks(
    config: dict[str, Any], case: str, seed: int
) -> dict[str, bool]:
    reference_path = (
        RESULTS_ROOT
        / "raw"
        / "main"
        / f"case_{case}"
        / "vanilla"
        / f"seed_{seed}"
        / "config.json"
    )
    if not reference_path.exists():
        return {
            "reference_config_present": False,
            "collocation_seed_matches_main": False,
            "structure_seed_matches_main": False,
            "initialization_matches_main": False,
        }
    try:
        reference = _load_json(reference_path)
    except (OSError, ValueError, TypeError):
        return {
            "reference_config_present": False,
            "collocation_seed_matches_main": False,
            "structure_seed_matches_main": False,
            "initialization_matches_main": False,
        }
    sampling = config.get("sampling", {})
    reference_sampling = reference.get("sampling", {})
    return {
        "reference_config_present": True,
        "collocation_seed_matches_main": sampling.get("collocation_generator_seed")
        == reference_sampling.get("collocation_generator_seed"),
        "structure_seed_matches_main": sampling.get("structure_generator_seed")
        == reference_sampling.get("structure_generator_seed"),
        "initialization_matches_main": config.get("initialization")
        == reference.get("initialization"),
    }


def _check_metrics(
    path: Path, case: str, seed: int, failures: list[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        metrics = _load_json(path)
    except (OSError, ValueError, TypeError) as exc:
        failures.append(f"{path}: cannot read metrics: {exc}")
        return {"readable": False}

    missing = sorted((COMMON_METRICS | BRANCH_METRICS) - metrics.keys())
    identity = {
        "experiment": metrics.get("experiment") == EXPERIMENT,
        "case": metrics.get("case") == case,
        "eta": _close(metrics.get("eta"), CASES[case]),
        "model": metrics.get("model") == MODEL,
        "seed": metrics.get("seed") == seed,
        "steps": metrics.get("steps") == 5000,
        "lambda_struct": _close(metrics.get("lambda_struct"), 1.0),
        "parameter_count": metrics.get("parameter_count") == EXPECTED_PARAMETER_COUNT,
    }
    permitted_null = (
        {"amplitude_minus_drift_rel_rms", "amplitude_minus_drift_rel_max"}
        if case == "P"
        else set()
    )
    numeric_keys = (COMMON_METRICS | BRANCH_METRICS) - permitted_null
    non_finite = sorted(
        key
        for key in numeric_keys
        if key in metrics and not _is_finite_number(metrics[key])
    )
    null_semantics = {
        key: metrics.get(key) is None for key in sorted(permitted_null)
    }
    phase_targets = {
        "phase_plus_slope_target": _close(
            metrics.get("phase_plus_slope_target"), -math.sqrt(29.0)
        ),
        "phase_minus_slope_target": _close(
            metrics.get("phase_minus_slope_target"), math.sqrt(29.0)
        ),
    }
    pde_only = {
        "total_equals_pde": _close(
            metrics.get("final_loss_total"),
            metrics.get("final_loss_pde"),
            atol=1.0e-11,
        ),
        "continuity_zero": _close(metrics.get("final_loss_continuity"), 0.0),
        "norm_zero": _close(metrics.get("final_loss_norm"), 0.0),
        "energy_zero": _close(metrics.get("final_loss_energy"), 0.0),
        "structure_zero": _close(metrics.get("final_loss_structure"), 0.0),
    }
    result.update(
        {
            "readable": True,
            "missing_required_keys": missing,
            "identity": identity,
            "non_finite_required_metrics": non_finite,
            "case_P_null_semantics": null_semantics,
            "phase_slope_targets": phase_targets,
            "pde_only": pde_only,
        }
    )
    if missing:
        failures.append(f"{path}: missing required metric keys {missing}")
    if not all(identity.values()):
        failures.append(f"{path}: identity/config metric mismatch {identity}")
    if non_finite:
        failures.append(f"{path}: non-finite required metrics {non_finite}")
    if null_semantics and not all(null_semantics.values()):
        failures.append(f"{path}: invalid zero-branch relative-drift values {null_semantics}")
    if not all(phase_targets.values()):
        failures.append(f"{path}: exact phase-slope evaluation targets mismatch {phase_targets}")
    if not all(pde_only.values()):
        failures.append(f"{path}: final losses are not PDE-only {pde_only}")
    return result


def _check_loss_history(path: Path, failures: list[str]) -> dict[str, Any]:
    try:
        history = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        failures.append(f"{path}: cannot read loss history: {exc}")
        return {"readable": False}

    missing = sorted(({"step", "elapsed_seconds", *LOSS_COLUMNS}) - set(history.columns))
    result: dict[str, Any] = {
        "readable": True,
        "rows": int(len(history)),
        "missing_columns": missing,
    }
    if missing:
        failures.append(f"{path}: missing loss columns {missing}")
        return result

    expected_steps = np.asarray([1, *range(10, 5001, 10)], dtype=np.int64)
    observed_steps = history["step"].to_numpy()
    schedule_matches = observed_steps.shape == expected_steps.shape and bool(
        np.array_equal(observed_steps, expected_steps)
    )
    values = history[[*LOSS_COLUMNS, "elapsed_seconds"]].to_numpy(dtype=np.float64)
    finite = bool(np.isfinite(values).all())
    pde_only = bool(
        np.allclose(history["total"], history["pde"], rtol=0.0, atol=1.0e-11)
        and np.allclose(
            history[["continuity", "norm", "energy", "structure"]],
            0.0,
            rtol=0.0,
            atol=1.0e-14,
        )
    )
    gradient_cosine_valid = True
    if "gradient_cosine" in history:
        diagnostic = history["gradient_cosine"].dropna().to_numpy(dtype=np.float64)
        gradient_cosine_valid = bool(np.isfinite(diagnostic).all())
    result.update(
        {
            "schedule_matches_5000_steps": schedule_matches,
            "all_loss_values_finite": finite,
            "pde_only": pde_only,
            "gradient_cosine_non_null_values_finite": gradient_cosine_valid,
        }
    )
    if not schedule_matches:
        failures.append(f"{path}: history schedule is not step 1 plus every 10 through 5000")
    if not finite:
        failures.append(f"{path}: loss history contains non-finite loss/time values")
    if not pde_only:
        failures.append(f"{path}: loss history is not PDE-only")
    if not gradient_cosine_valid:
        failures.append(f"{path}: gradient_cosine contains infinity")
    return result


def _check_time_series(path: Path, failures: list[str]) -> dict[str, Any]:
    try:
        series = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        failures.append(f"{path}: cannot read time series: {exc}")
        return {"readable": False}
    missing = sorted(TIME_SERIES_COLUMNS - set(series.columns))
    finite = False
    grid_matches = False
    if not missing and len(series) == 201:
        values = series[sorted(TIME_SERIES_COLUMNS)].to_numpy(dtype=np.float64)
        finite = bool(np.isfinite(values).all())
        grid_matches = bool(
            np.allclose(series["t"].to_numpy(), np.linspace(0.0, 5.0, 201))
        )
    result = {
        "readable": True,
        "rows": int(len(series)),
        "missing_columns": missing,
        "all_required_values_finite": finite,
        "time_grid_matches": grid_matches,
    }
    if len(series) != 201:
        failures.append(f"{path}: expected 201 rows, found {len(series)}")
    if missing:
        failures.append(f"{path}: missing time-series columns {missing}")
    elif not finite:
        failures.append(f"{path}: required time-series values are not all finite")
    if not grid_matches:
        failures.append(f"{path}: time grid is not linspace(0, 5, 201)")
    return result


def _array_is_finite(array: np.ndarray) -> bool:
    return bool(np.isfinite(array.real).all() and np.isfinite(array.imag).all())


def _check_dense_grid(
    path: Path, case: str, failures: list[str]
) -> dict[str, Any]:
    expected_fields = BASE_DENSE_FIELDS | BRANCH_DENSE_FIELDS
    try:
        with np.load(path, allow_pickle=False) as data:
            fields = set(data.files)
            missing = sorted(expected_fields - fields)
            result: dict[str, Any] = {
                "readable": True,
                "missing_fields": missing,
                "field_count": len(fields),
            }
            if missing:
                failures.append(f"{path}: missing dense-grid fields {missing}")
                return result

            spinor_checks = {
                name: {
                    "shape": list(data[name].shape),
                    "shape_ok": data[name].shape == EXPECTED_SPINOR_SHAPE,
                    "dtype": str(data[name].dtype),
                    "dtype_ok": data[name].dtype == np.dtype(np.complex128),
                    "finite": _array_is_finite(data[name]),
                }
                for name in ("psi_predicted", "psi_exact")
            }
            scalar_names = sorted(
                (BASE_DENSE_FIELDS | BRANCH_DENSE_FIELDS)
                - {"x", "t", "psi_predicted", "psi_exact"}
            )
            scalar_checks = {
                name: {
                    "shape_ok": data[name].shape == EXPECTED_SCALAR_FIELD_SHAPE,
                    "dtype_ok": data[name].dtype == np.dtype(np.float64),
                    "finite": _array_is_finite(data[name]),
                }
                for name in scalar_names
            }
            coordinate_checks = {
                "x_shape_dtype": data["x"].shape == (128,)
                and data["x"].dtype == np.dtype(np.float64),
                "t_shape_dtype": data["t"].shape == (201,)
                and data["t"].dtype == np.dtype(np.float64),
                "x_finite": _array_is_finite(data["x"]),
                "t_finite": _array_is_finite(data["t"]),
                "t_grid": bool(np.allclose(data["t"], np.linspace(0.0, 5.0, 201))),
                "x_periodic_grid": bool(
                    np.allclose(
                        data["x"],
                        -math.pi + 2.0 * math.pi * np.arange(128) / 128,
                    )
                ),
            }
            amplitude_plus_initial = 1.0 / math.sqrt(
                2.0 * math.pi * (1.0 + CASES[case] ** 2)
            )
            amplitude_minus_initial = CASES[case] * amplitude_plus_initial
            hard_initial_condition = {
                "psi": bool(
                    np.allclose(
                        data["psi_predicted"][0],
                        data["psi_exact"][0],
                        rtol=0.0,
                        atol=2.0e-14,
                    )
                ),
                "phase_plus": bool(np.all(data["phase_plus"][0] == 0.0)),
                "phase_minus": bool(np.all(data["phase_minus"][0] == 0.0)),
                "amplitude_plus": bool(
                    np.allclose(
                        data["amplitude_plus"][0],
                        amplitude_plus_initial,
                        rtol=0.0,
                        atol=2.0e-14,
                    )
                ),
                "amplitude_minus": bool(
                    np.allclose(
                        data["amplitude_minus"][0],
                        amplitude_minus_initial,
                        rtol=0.0,
                        atol=2.0e-14,
                    )
                ),
            }
            result.update(
                {
                    "spinor_checks": spinor_checks,
                    "scalar_field_checks": scalar_checks,
                    "coordinate_checks": coordinate_checks,
                    "hard_initial_condition": hard_initial_condition,
                }
            )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        failures.append(f"{path}: cannot validate dense grid: {exc}")
        return {"readable": False, "error": str(exc)}

    if not all(all(item.values()) for item in spinor_checks.values()):
        failures.append(f"{path}: spinor shape/dtype/finiteness check failed")
    if not all(all(item.values()) for item in scalar_checks.values()):
        failures.append(f"{path}: scalar dense-field shape/dtype/finiteness check failed")
    if not all(coordinate_checks.values()):
        failures.append(f"{path}: coordinate grid check failed {coordinate_checks}")
    if not all(hard_initial_condition.values()):
        failures.append(f"{path}: hard initial condition check failed {hard_initial_condition}")
    return result


def _check_checkpoint(
    path: Path, config: dict[str, Any], failures: list[str]
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["state_dict"]
        learned_tensors = {
            name: tensor
            for name, tensor in state.items()
            if name not in {"u_plus", "u_minus"}
        }
        parameter_count = sum(tensor.numel() for tensor in learned_tensors.values())
        all_float64 = all(tensor.dtype == torch.float64 for tensor in state.values())
        all_finite = all(bool(torch.isfinite(tensor).all()) for tensor in state.values())
        config_matches = checkpoint.get("experiment_config") == config.get(
            "experiment_config"
        )
        run_matches = checkpoint.get("run_spec") == config.get("run_spec")
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
        failures.append(f"{path}: cannot validate checkpoint: {exc}")
        return {"readable": False, "error": str(exc)}

    result = {
        "readable": True,
        "parameter_count": parameter_count,
        "parameter_count_ok": parameter_count == EXPECTED_PARAMETER_COUNT,
        "all_state_tensors_float64": all_float64,
        "all_state_tensors_finite": all_finite,
        "experiment_config_matches": config_matches,
        "run_spec_matches": run_matches,
    }
    if not all(
        (
            result["parameter_count_ok"],
            all_float64,
            all_finite,
            config_matches,
            run_matches,
        )
    ):
        failures.append(f"{path}: checkpoint mismatch {result}")
    return result


def _expected_main_files() -> set[str]:
    files: set[str] = set()
    for case in CASES:
        for model in ("vanilla", "sp_cpa", "sp_cpa_conservative"):
            for seed in SEEDS:
                prefix = Path(f"case_{case}") / model / f"seed_{seed}"
                files.update(str(prefix / name) for name in REQUIRED_RUN_FILES)
        prefix = Path(f"case_{case}") / "two_branch" / "seed_0"
        files.update(str(prefix / name) for name in REQUIRED_RUN_FILES)
    return files


def _expected_sensitivity_files() -> set[str]:
    files: set[str] = {
        str(
            Path("case_M")
            / "sp_cpa_conservative"
            / "lambda_1"
            / "REUSED_MAIN_RUN.json"
        )
    }
    for label in ("lambda_0p1", "lambda_10"):
        prefix = Path("case_M") / "sp_cpa_conservative" / label / "seed_0"
        files.update(str(prefix / name) for name in REQUIRED_RUN_FILES)
    return files


def _relative_file_inventory(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


def _run_original_reproducibility_audit() -> dict[str, Any]:
    """Run the original audit in memory while suppressing its JSON write."""
    from . import audit_results

    captured: dict[str, Any] = {}
    original_writer = audit_results.write_json

    def capture_writer(path: Path, payload: dict[str, Any]) -> None:
        captured["path"] = str(path)
        captured["payload"] = payload

    audit_results.write_json = capture_writer
    output = io.StringIO()
    raised: str | None = None
    try:
        with contextlib.redirect_stdout(output):
            audit_results.main()
    except SystemExit as exc:
        raised = str(exc)
    finally:
        audit_results.write_json = original_writer
    payload = captured.get("payload", {})
    return {
        "passed": raised is None and payload.get("status") == "PASS",
        "status": payload.get("status"),
        "failures": payload.get("failures", []),
        "raised": raised,
        "stdout": output.getvalue().strip(),
        "write_suppressed": True,
    }


def _check_baseline_protection(failures: list[str]) -> dict[str, Any]:
    main_root = RESULTS_ROOT / "raw" / "main"
    sensitivity_root = RESULTS_ROOT / "raw" / "sensitivity"
    expected_main = _expected_main_files()
    expected_sensitivity = _expected_sensitivity_files()
    actual_main = _relative_file_inventory(main_root)
    actual_sensitivity = _relative_file_inventory(sensitivity_root)
    main_missing = sorted(expected_main - actual_main)
    main_extra = sorted(actual_main - expected_main)
    sensitivity_missing = sorted(expected_sensitivity - actual_sensitivity)
    sensitivity_extra = sorted(actual_sensitivity - expected_sensitivity)

    if main_missing or main_extra:
        failures.append(
            "results/raw/main inventory changed: "
            f"missing={main_missing}, extra={main_extra}"
        )
    if sensitivity_missing or sensitivity_extra:
        failures.append(
            "results/raw/sensitivity inventory changed: "
            f"missing={sensitivity_missing}, extra={sensitivity_extra}"
        )

    try:
        original_audit = _run_original_reproducibility_audit()
    except Exception as exc:  # keep the pilot audit report writable on audit failure
        original_audit = {
            "passed": False,
            "status": None,
            "failures": [],
            "raised": f"{type(exc).__name__}: {exc}",
            "stdout": "",
            "write_suppressed": True,
        }
    if not original_audit["passed"]:
        failures.append(f"original reproducibility audit failed: {original_audit}")

    return {
        "main_file_count": len(actual_main),
        "main_expected_file_count": len(expected_main),
        "main_missing": main_missing,
        "main_extra": main_extra,
        "sensitivity_file_count": len(actual_sensitivity),
        "sensitivity_expected_file_count": len(expected_sensitivity),
        "sensitivity_missing": sensitivity_missing,
        "sensitivity_extra": sensitivity_extra,
        "original_reproducibility_audit": original_audit,
    }


def _check_pilot_inventory(failures: list[str]) -> dict[str, Any]:
    expected_seed_dirs = {
        str(Path(f"case_{case}") / MODEL / f"seed_{seed}")
        for case in CASES
        for seed in SEEDS
    }
    actual_seed_dirs = (
        {
            str(path.relative_to(PILOT_ROOT))
            for path in PILOT_ROOT.rglob("seed_*")
            if path.is_dir()
        }
        if PILOT_ROOT.exists()
        else set()
    )
    missing_seed_dirs = sorted(expected_seed_dirs - actual_seed_dirs)
    extra_seed_dirs = sorted(actual_seed_dirs - expected_seed_dirs)
    expected_files = {
        str(Path(seed_dir) / name)
        for seed_dir in expected_seed_dirs
        for name in REQUIRED_RUN_FILES
    }
    actual_files = _relative_file_inventory(PILOT_ROOT)
    missing_files = sorted(expected_files - actual_files)
    extra_files = sorted(actual_files - expected_files)
    if missing_seed_dirs or extra_seed_dirs or missing_files or extra_files:
        failures.append(
            "formal pilot run set is not exactly P/M x seeds 0,1,2: "
            f"missing_dirs={missing_seed_dirs}, extra_dirs={extra_seed_dirs}, "
            f"missing_files={missing_files}, extra_files={extra_files}"
        )
    return {
        "expected_run_count": 6,
        "actual_run_count": len(actual_seed_dirs),
        "missing_seed_directories": missing_seed_dirs,
        "extra_seed_directories": extra_seed_dirs,
        "expected_file_count": len(expected_files),
        "actual_file_count": len(actual_files),
        "missing_files": missing_files,
        "extra_files": extra_files,
    }


def _check_summary_tables(failures: list[str]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    specifications = (
        ("learned_two_phase_pilot_runs.csv", 6),
        ("learned_two_phase_pilot_summary.csv", 2),
        ("learned_two_phase_pilot_comparison.csv", 26),
    )
    for name, expected_rows in specifications:
        path = RESULTS_ROOT / name
        check: dict[str, Any] = {
            "name": name,
            "expected_rows": expected_rows,
            "present": path.exists() and path.stat().st_size > 0,
        }
        if not check["present"]:
            failures.append(f"{path}: missing or empty summary table")
            checks.append(check)
            continue
        try:
            table = pd.read_csv(path)
            check["rows"] = len(table)
            check["row_count_ok"] = len(table) == expected_rows
            numeric = table.select_dtypes(include="number")
            check["non_finite_numeric_cells"] = int(
                (~np.isfinite(numeric.to_numpy(dtype=np.float64))).sum()
            )
            if name == "learned_two_phase_pilot_runs.csv":
                identities = set(
                    zip(table.get("case", []), table.get("seed", []), strict=False)
                )
                check["six_run_identities_ok"] = identities == {
                    (case, seed) for case in CASES for seed in SEEDS
                }
                check["model_ok"] = "model" in table and bool(table["model"].eq(MODEL).all())
                check["experiment_ok"] = "experiment" in table and bool(
                    table["experiment"].eq(EXPERIMENT).all()
                )
            elif name == "learned_two_phase_pilot_summary.csv":
                check["three_seeds_per_case"] = "n_seeds" in table and bool(
                    table["n_seeds"].eq(3).all()
                )
        except (OSError, ValueError, TypeError, pd.errors.ParserError) as exc:
            check["read_error"] = str(exc)
        required_bools = [value for value in check.values() if isinstance(value, bool)]
        if not required_bools or not all(required_bools):
            failures.append(f"{path}: summary-table check failed {check}")
        checks.append(check)
    return checks


def _check_figures(failures: list[str]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    exported_stems = (
        {
            path.stem
            for path in FIGURE_ROOT.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".svg"}
        }
        if FIGURE_ROOT.exists()
        else set()
    )
    missing_groups = sorted(set(EXPECTED_FIGURES) - exported_stems)
    extra_groups = sorted(exported_stems - set(EXPECTED_FIGURES))
    if missing_groups or extra_groups:
        failures.append(
            f"figure groups mismatch: missing={missing_groups}, extra={extra_groups}"
        )

    for name in EXPECTED_FIGURES:
        check: dict[str, Any] = {"name": name}
        for extension in ("png", "pdf", "svg"):
            path = FIGURE_ROOT / f"{name}.{extension}"
            check[extension] = path.exists() and path.stat().st_size > 0
        svg_path = FIGURE_ROOT / f"{name}.svg"
        try:
            svg_text_nodes = (
                svg_path.read_text(encoding="utf-8").count("<text")
                if check["svg"]
                else 0
            )
        except (OSError, UnicodeError):
            svg_text_nodes = 0
        check["svg_text_nodes"] = svg_text_nodes
        check["editable_svg_text"] = svg_text_nodes > 0
        if not all(check[extension] for extension in ("png", "pdf", "svg")) or not check[
            "editable_svg_text"
        ]:
            failures.append(f"figure {name}: export/editable-text check failed {check}")
        checks.append(check)
    return checks


def main() -> None:
    failures: list[str] = []
    inventory = _check_pilot_inventory(failures)
    run_checks: list[dict[str, Any]] = []

    for case in CASES:
        for seed in SEEDS:
            run_dir = _run_dir(case, seed)
            artifact_checks = {
                name: (run_dir / name).exists() and (run_dir / name).stat().st_size > 0
                for name in REQUIRED_RUN_FILES
            }
            run_result: dict[str, Any] = {
                "case": case,
                "seed": seed,
                "directory": str(run_dir),
                "artifacts": artifact_checks,
            }
            missing = [name for name, present in artifact_checks.items() if not present]
            if missing:
                failures.append(f"{run_dir}: missing or empty required artifacts {missing}")
                run_checks.append(run_result)
                continue

            try:
                config = _load_json(run_dir / "config.json")
            except (OSError, ValueError, TypeError) as exc:
                failures.append(f"{run_dir / 'config.json'}: cannot read config: {exc}")
                run_checks.append(run_result)
                continue

            config_checks = _config_checks(config, case, seed)
            reference_checks = _reference_sampling_checks(config, case, seed)
            run_result["config_checks"] = config_checks
            run_result["baseline_random_source_checks"] = reference_checks
            if not all(config_checks.values()):
                failures.append(f"{run_dir}: formal config mismatch {config_checks}")
            if not all(reference_checks.values()):
                failures.append(
                    f"{run_dir}: random source/initialization differs from main {reference_checks}"
                )

            run_result["metrics"] = _check_metrics(
                run_dir / "metrics.json", case, seed, failures
            )
            run_result["loss_history"] = _check_loss_history(
                run_dir / "loss_history.csv", failures
            )
            run_result["time_series"] = _check_time_series(
                run_dir / "time_series.csv", failures
            )
            run_result["dense_grid"] = _check_dense_grid(
                run_dir / "dense_grid.npz", case, failures
            )
            run_result["model_state"] = _check_checkpoint(
                run_dir / "model_state.pt", config, failures
            )
            run_checks.append(run_result)

    summary_checks = _check_summary_tables(failures)
    figure_checks = _check_figures(failures)
    baseline_protection = _check_baseline_protection(failures)

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "counts": {
            "expected_formal_runs": 6,
            "checked_formal_runs": len(run_checks),
            "complete_six_piece_runs": sum(
                all(run["artifacts"].values()) for run in run_checks
            ),
            "expected_figure_groups": len(EXPECTED_FIGURES),
            "expected_figure_exports": len(EXPECTED_FIGURES) * 3,
        },
        "pilot_inventory": inventory,
        "run_checks": run_checks,
        "summary_checks": summary_checks,
        "figure_checks": figure_checks,
        "baseline_protection": baseline_protection,
        "notes": {
            "loss_finiteness": (
                "All six actual loss columns and elapsed_seconds must be finite. "
                "gradient_cosine may be blank because LTPB-CPA is trained with PDE loss only."
            ),
            "baseline_audit": (
                "The original audit was executed with its write function replaced by an "
                "in-memory capture, so this audit does not rewrite baseline artifacts."
            ),
        },
    }
    _write_audit(payload)
    print(json.dumps(payload["counts"], indent=2))
    print(f"status={payload['status']}")
    print(f"audit={AUDIT_PATH}")
    if failures:
        raise SystemExit(
            f"Learned two-phase pilot audit failed with {len(failures)} issue(s); "
            f"see {AUDIT_PATH}"
        )


if __name__ == "__main__":
    main()
