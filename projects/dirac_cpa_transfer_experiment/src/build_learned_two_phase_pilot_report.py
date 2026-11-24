from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
RAW_ROOT = RESULTS_ROOT / "raw" / "learned_two_phase_pilot"
FIGURE_ROOT = RESULTS_ROOT / "figures" / "learned_two_phase_pilot"
PILOT_SUMMARY_PATH = RESULTS_ROOT / "learned_two_phase_pilot_summary.csv"
BASELINE_SUMMARY_PATH = RESULTS_ROOT / "summary.csv"
REPORT_PATH = PROJECT_ROOT / "LEARNED_TWO_PHASE_PILOT_REPORT.md"

CASES = {"P": 0.0, "M": 1.0}
SEEDS = (0, 1, 2)
MODEL = "ltpb_cpa"
EXPERIMENT = "learned_two_phase_pilot"

RUN_FILES = (
    "config.json",
    "metrics.json",
    "loss_history.csv",
    "time_series.csv",
    "dense_grid.npz",
    "model_state.pt",
)

COMMON_METRICS = (
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
    "final_loss_pde",
    "final_loss_continuity",
    "final_loss_norm",
    "final_loss_energy",
    "final_loss_structure",
    "final_loss_total",
    "parameter_count",
    "training_seconds",
)

LTPB_METRICS = (
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
)

TIME_SERIES_COLUMNS = (
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
    "continuity",
    "energy_density",
    "relative_phase_predicted",
    "relative_phase_exact",
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
)

FIGURE_BASENAMES = (
    "01_main_metric_comparison",
    "02_learned_branch_phases",
    "03_case_M_current_x0",
    "04_case_M_relative_phase_x0",
    "05_amplitude_and_occupations",
    "06_loss_histories",
)

FIGURE_SOURCE_FILES = (
    "01_main_metric_comparison.csv",
    "02_learned_branch_phases.csv",
    "02_learned_branch_phase_slopes.csv",
    "03_case_M_current_x0.csv",
    "04_case_M_relative_phase_x0.csv",
    "05_amplitude_and_occupations.csv",
    "06_loss_histories.csv",
)

FIGURE_SOURCE_BY_FIGURE = {
    "01_main_metric_comparison": ("01_main_metric_comparison.csv",),
    "02_learned_branch_phases": (
        "02_learned_branch_phases.csv",
        "02_learned_branch_phase_slopes.csv",
    ),
    "03_case_M_current_x0": ("03_case_M_current_x0.csv",),
    "04_case_M_relative_phase_x0": ("04_case_M_relative_phase_x0.csv",),
    "05_amplitude_and_occupations": ("05_amplitude_and_occupations.csv",),
    "06_loss_histories": ("06_loss_histories.csv",),
}

LABELS = {
    "vanilla": "Vanilla",
    "sp_cpa": "SP-CPA",
    "sp_cpa_conservative": "SP-CPA-Conservative",
    "ltpb_cpa": "LTPB-CPA",
    "two_branch": "Two-Branch oracle",
}

# These operational cutoffs are deliberately encoded independently of the run values.
# They turn the qualitative words in the experiment request into reproducible rules.
P_NOT_WORSE_SPINOR_RATIO = 1.25
P_NOT_WORSE_FIDELITY_DROP = 0.01
M_MATERIAL_ERROR_RATIO = 0.75
M_MATERIAL_FIDELITY_GAIN = 0.05
CARRIER_RELATIVE_SLOPE_ERROR = 0.10
CARRIER_MIN_R2 = 0.95
ORACLE_PROXIMITY_ERROR_RATIO = 3.0
ORACLE_PROXIMITY_FIDELITY_GAP = 0.05


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Required file is empty: {path}")
    return path


def validate_figure_file(path: Path) -> None:
    require_file(path)
    if path.stat().st_size < 100:
        raise RuntimeError(f"Figure file is implausibly small: {path}")
    prefix = path.read_bytes()[:512].lstrip()
    if path.suffix == ".png" and not prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"Invalid PNG signature: {path}")
    if path.suffix == ".pdf" and not prefix.startswith(b"%PDF-"):
        raise RuntimeError(f"Invalid PDF signature: {path}")
    if path.suffix == ".svg" and b"<svg" not in prefix:
        raise RuntimeError(f"Invalid SVG content: {path}")


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing columns in {source}: {missing}")


def read_json(path: Path) -> dict[str, Any]:
    require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return payload


def assert_close(actual: Any, expected: float, label: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric: {actual!r}") from exc
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise RuntimeError(f"{label}: expected {expected!r}, found {actual!r}")


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, found {actual!r}")


def finite_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite: {value!r}")
    return result


def validate_fixed_config(
    config: dict[str, Any], case: str, seed: int, experiment: str, model: str
) -> None:
    try:
        cfg = config["experiment_config"]
        run_spec = config["run_spec"]
        sampling = config["sampling"]
    except KeyError as exc:
        raise RuntimeError(f"Incomplete run configuration: missing {exc.args[0]}") from exc

    exact_config = {
        "mass": 5.0,
        "wave_number": 2.0,
        "t_max": 5.0,
        "hidden_layers": 4,
        "hidden_width": 64,
        "collocation_points": 1024,
        "learning_rate": 1.0e-3,
        "adam_steps": 5000,
        "test_t_points": 201,
        "test_x_points": 128,
    }
    for key, expected in exact_config.items():
        assert_close(cfg.get(key), expected, f"experiment_config.{key}")
    assert_equal(cfg.get("dtype"), "float64", "experiment_config.dtype")

    assert_equal(run_spec.get("case"), case, "run_spec.case")
    assert_close(run_spec.get("eta"), CASES[case], "run_spec.eta")
    assert_equal(run_spec.get("experiment"), experiment, "run_spec.experiment")
    assert_equal(run_spec.get("model"), model, "run_spec.model")
    assert_equal(run_spec.get("seed"), seed, "run_spec.seed")
    assert_equal(run_spec.get("steps"), 5000, "run_spec.steps")

    assert_equal(
        sampling.get("collocation_resampled_each_step"),
        True,
        "sampling.collocation_resampled_each_step",
    )
    assert_equal(
        sampling.get("collocation_generator_seed"),
        100000 + seed,
        "sampling.collocation_generator_seed",
    )
    assert_equal(
        sampling.get("structure_generator_seed"),
        200000 + seed,
        "sampling.structure_generator_seed",
    )
    assert_equal(
        config.get("initialization"),
        "Xavier normal for every linear weight; zero biases",
        "initialization",
    )


def validate_pilot_run(case: str, seed: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
    run_dir = RAW_ROOT / f"case_{case}" / MODEL / f"seed_{seed}"
    for filename in RUN_FILES:
        require_file(run_dir / filename)

    config = read_json(run_dir / "config.json")
    metrics = read_json(run_dir / "metrics.json")
    validate_fixed_config(config, case, seed, EXPERIMENT, MODEL)

    identity = {
        "experiment": EXPERIMENT,
        "case": case,
        "eta": CASES[case],
        "model": MODEL,
        "seed": seed,
        "steps": 5000,
    }
    for key, expected in identity.items():
        if isinstance(expected, float):
            assert_close(metrics.get(key), expected, f"{run_dir.name}: metrics.{key}")
        else:
            assert_equal(metrics.get(key), expected, f"{run_dir.name}: metrics.{key}")

    for key in COMMON_METRICS + LTPB_METRICS:
        if key not in metrics:
            raise RuntimeError(f"Missing metric {key!r} in {run_dir / 'metrics.json'}")
        value = metrics[key]
        if key in {"amplitude_minus_drift_rel_rms", "amplitude_minus_drift_rel_max"}:
            if case == "P":
                if value is not None:
                    raise RuntimeError(
                        f"{key} must be null for Case P because a_minus,0=0: {run_dir}"
                    )
                continue
            if value is None:
                raise RuntimeError(f"{key} must be finite for Case M: {run_dir}")
        finite_number(value, f"{run_dir.name}: metrics.{key}")

    assert_equal(int(metrics["parameter_count"]), 12996, "LTPB parameter_count")
    assert_close(metrics["phase_plus_slope_target"], -math.sqrt(29.0), "S+ slope target")
    assert_close(metrics["phase_minus_slope_target"], math.sqrt(29.0), "S- slope target")
    for key in (
        "final_loss_continuity",
        "final_loss_norm",
        "final_loss_energy",
        "final_loss_structure",
    ):
        assert_close(metrics[key], 0.0, f"PDE-only {key}")
    assert_close(metrics["final_loss_total"], metrics["final_loss_pde"], "PDE-only total loss")

    loss_path = run_dir / "loss_history.csv"
    loss = pd.read_csv(loss_path)
    require_columns(
        loss,
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
        loss_path,
    )
    if len(loss) != 501 or int(loss.iloc[-1]["step"]) != 5000:
        raise RuntimeError(f"Incomplete 5000-step loss history in {loss_path}")
    if not np.isfinite(loss[["total", "pde", "elapsed_seconds"]].to_numpy(float)).all():
        raise RuntimeError(f"Non-finite loss values in {loss_path}")
    if not np.allclose(loss["total"], loss["pde"], rtol=1.0e-12, atol=1.0e-12):
        raise RuntimeError(f"LTPB run is not PDE-only in {loss_path}")

    series_path = run_dir / "time_series.csv"
    series = pd.read_csv(series_path)
    require_columns(series, TIME_SERIES_COLUMNS, series_path)
    if len(series) != 201:
        raise RuntimeError(f"Expected 201 time rows in {series_path}, found {len(series)}")
    if not np.isfinite(series[list(TIME_SERIES_COLUMNS)].to_numpy(float)).all():
        raise RuntimeError(f"Non-finite time-series values in {series_path}")
    assert_close(series.iloc[0]["t"], 0.0, f"{series_path}: first t")
    assert_close(series.iloc[-1]["t"], 5.0, f"{series_path}: final t")
    if not np.allclose(
        series.iloc[0][["phase_plus_spatial_mean", "phase_minus_spatial_mean"]],
        0.0,
        rtol=0.0,
        atol=1.0e-13,
    ):
        raise RuntimeError(f"Hard phase initial condition is not exact in {series_path}")

    dense_path = run_dir / "dense_grid.npz"
    with np.load(dense_path) as dense:
        missing = sorted(set(DENSE_GRID_KEYS) - set(dense.files))
        if missing:
            raise RuntimeError(f"Missing dense-grid arrays in {dense_path}: {missing}")
        expected_shapes = {
            "x": (128,),
            "t": (201,),
            "psi_predicted": (201, 128, 2),
            "psi_exact": (201, 128, 2),
        }
        for key in DENSE_GRID_KEYS:
            if key in expected_shapes:
                expected_shape = expected_shapes[key]
            elif key not in {"x", "t", "psi_predicted", "psi_exact"}:
                expected_shape = (201, 128)
            else:
                continue
            if dense[key].shape != expected_shape:
                raise RuntimeError(
                    f"Bad shape for {key} in {dense_path}: "
                    f"expected {expected_shape}, found {dense[key].shape}"
                )
            if not np.isfinite(dense[key]).all():
                raise RuntimeError(f"Non-finite dense-grid values in {dense_path}:{key}")
        if not np.iscomplexobj(dense["psi_predicted"]):
            raise RuntimeError(f"Predicted spinor is not complex in {dense_path}")

    return metrics, config, run_dir


def load_and_validate_pilot_runs() -> tuple[pd.DataFrame, dict[tuple[str, int], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    configs: dict[tuple[str, int], dict[str, Any]] = {}
    for case in CASES:
        for seed in SEEDS:
            metrics, config, run_dir = validate_pilot_run(case, seed)
            row = dict(metrics)
            row["run_dir"] = str(run_dir)
            rows.append(row)
            configs[(case, seed)] = config

    discovered = {
        path.parent.resolve()
        for path in RAW_ROOT.rglob("metrics.json")
        if path.is_file()
    }
    expected = {
        (RAW_ROOT / f"case_{case}" / MODEL / f"seed_{seed}").resolve()
        for case in CASES
        for seed in SEEDS
    }
    if discovered != expected:
        raise RuntimeError(
            "Pilot raw directory must contain exactly the six formal metrics files; "
            f"missing={sorted(map(str, expected - discovered))}, "
            f"extra={sorted(map(str, discovered - expected))}"
        )
    return pd.DataFrame(rows).sort_values(["case", "seed"]).reset_index(drop=True), configs


def load_and_validate_pilot_summary(runs: pd.DataFrame) -> pd.DataFrame:
    require_file(PILOT_SUMMARY_PATH)
    summary = pd.read_csv(PILOT_SUMMARY_PATH)
    require_columns(summary, ("case", "model", "n_seeds"), PILOT_SUMMARY_PATH)
    expected_groups = {(case, MODEL) for case in CASES}
    found_groups = set(zip(summary["case"], summary["model"], strict=False))
    if found_groups != expected_groups or len(summary) != 2:
        raise RuntimeError(
            f"Pilot summary must contain exactly {expected_groups}; found {found_groups}"
        )

    for case in CASES:
        row = summary[(summary["case"] == case) & (summary["model"] == MODEL)].iloc[0]
        assert_equal(int(row["n_seeds"]), 3, f"pilot summary {case} n_seeds")
        group = runs[runs["case"] == case]
        for metric in COMMON_METRICS + LTPB_METRICS:
            mean_column = f"{metric}_mean"
            std_column = f"{metric}_std"
            require_columns(summary, (mean_column, std_column), PILOT_SUMMARY_PATH)
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            actual_mean = row[mean_column]
            actual_std = row[std_column]
            if len(finite) == 0:
                if not pd.isna(actual_mean):
                    raise RuntimeError(
                        f"Expected NaN {mean_column} for Case {case}, found {actual_mean}"
                    )
                assert_close(actual_std, 0.0, f"pilot summary {case} {std_column}")
                continue
            expected_mean = float(np.mean(finite))
            expected_std = float(np.std(finite, ddof=0))
            assert_close(actual_mean, expected_mean, f"pilot summary {case} {mean_column}")
            assert_close(actual_std, expected_std, f"pilot summary {case} {std_column}")
    return summary.sort_values("case").reset_index(drop=True)


def validate_baseline_run_artifacts(case: str, model: str, seed: int, row: pd.Series) -> dict[str, Any]:
    run_dir = RESULTS_ROOT / "raw" / "main" / f"case_{case}" / model / f"seed_{seed}"
    for filename in RUN_FILES:
        require_file(run_dir / filename)
    config = read_json(run_dir / "config.json")
    metrics = read_json(run_dir / "metrics.json")
    validate_fixed_config(config, case, seed, "main", model)
    for metric in (
        "spinor_rel_l2",
        "final_spinor_rel_l2",
        "rho_rel_l2",
        "current_rel_l2",
        "fidelity_mean",
        "relative_phase_rmse_rad",
        "branch_coefficient_rel_l2",
        "branch_occupation_rmse",
        "final_loss_pde",
        "max_norm_drift",
    ):
        assert_close(row[metric], metrics[metric], f"baseline summary {case}/{model}/{seed}/{metric}")
    expected_parameters = 13061 if model in {"sp_cpa", "sp_cpa_conservative"} else 12996
    assert_equal(int(row["parameter_count"]), expected_parameters, f"{model} parameter_count")
    return config


def load_and_validate_baselines(
    pilot_configs: dict[tuple[str, int], dict[str, Any]]
) -> pd.DataFrame:
    require_file(BASELINE_SUMMARY_PATH)
    summary = pd.read_csv(BASELINE_SUMMARY_PATH)
    required = (
        "experiment",
        "case",
        "model",
        "seed",
        "eta",
        "steps",
        "parameter_count",
        "spinor_rel_l2",
        "final_spinor_rel_l2",
        "rho_rel_l2",
        "current_rel_l2",
        "fidelity_mean",
        "relative_phase_rmse_rad",
        "branch_coefficient_rel_l2",
        "branch_occupation_rmse",
        "final_loss_pde",
        "max_norm_drift",
    )
    require_columns(summary, required, BASELINE_SUMMARY_PATH)
    main = summary[summary["experiment"] == "main"].copy()

    expected: set[tuple[str, str, int]] = set()
    for case in CASES:
        for model in ("vanilla", "sp_cpa", "sp_cpa_conservative"):
            expected.update((case, model, seed) for seed in SEEDS)
        expected.add((case, "two_branch", 0))
    found = set(zip(main["case"], main["model"], main["seed"].astype(int), strict=False))
    if found != expected or len(main) != 20:
        raise RuntimeError(
            "Baseline summary must contain exactly 20 main rows; "
            f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
        )
    if main.duplicated(["case", "model", "seed"]).any():
        raise RuntimeError("Duplicate main rows in baseline summary")

    for _, row in main.iterrows():
        case = str(row["case"])
        model = str(row["model"])
        seed = int(row["seed"])
        assert_close(row["eta"], CASES[case], f"baseline {case}/{model}/{seed} eta")
        assert_equal(int(row["steps"]), 5000, f"baseline {case}/{model}/{seed} steps")
        baseline_config = validate_baseline_run_artifacts(case, model, seed, row)

        # The SP-CPA comparison is the primary fairness check. CPU thread count and
        # logging cadence are deliberately excluded because they do not alter samples.
        if model == "sp_cpa":
            pilot_config = pilot_configs[(case, seed)]
            for section, keys in {
                "experiment_config": (
                    "x_min",
                    "x_max",
                    "t_max",
                    "mass",
                    "wave_number",
                    "hidden_layers",
                    "hidden_width",
                    "dtype",
                    "learning_rate",
                    "adam_steps",
                    "collocation_points",
                    "test_t_points",
                    "test_x_points",
                ),
                "sampling": (
                    "collocation_generator_seed",
                    "collocation_resampled_each_step",
                    "structure_generator_seed",
                    "periodic_spatial_quadrature_endpoint",
                ),
            }.items():
                for key in keys:
                    assert_equal(
                        pilot_config[section][key],
                        baseline_config[section][key],
                        f"fairness mismatch {case}/seed_{seed}/{section}.{key}",
                    )
            assert_equal(
                pilot_config["initialization"],
                baseline_config["initialization"],
                f"fairness mismatch {case}/seed_{seed}/initialization",
            )
    return main.sort_values(["case", "model", "seed"]).reset_index(drop=True)


def load_and_validate_figures() -> dict[str, Any]:
    manifest_path = require_file(FIGURE_ROOT / "figure_manifest.json")
    manifest = read_json(manifest_path)
    if "figure_contract" not in manifest or "figures" not in manifest:
        raise RuntimeError(
            f"Figure manifest must contain figure_contract and figures: {manifest_path}"
        )
    figures = manifest["figures"]
    if not isinstance(figures, dict) or set(figures) != set(FIGURE_BASENAMES):
        raise RuntimeError(
            "Figure manifest must describe exactly the six pilot figures; "
            f"found={sorted(figures) if isinstance(figures, dict) else type(figures)}"
        )
    for basename in FIGURE_BASENAMES:
        entry = figures[basename]
        if not isinstance(entry, dict):
            raise RuntimeError(f"Invalid manifest entry for {basename}")
        files = entry.get("files")
        if not isinstance(files, dict) or set(files) != {"png", "pdf", "svg"}:
            raise RuntimeError(f"Manifest files are incomplete for {basename}")
        for extension in ("png", "pdf", "svg"):
            figure_path = FIGURE_ROOT / f"{basename}.{extension}"
            validate_figure_file(figure_path)
            expected = figure_path.resolve()
            declared = Path(str(files[extension]))
            if not declared.is_absolute():
                declared = (FIGURE_ROOT / declared).resolve()
            else:
                declared = declared.resolve()
            if declared != expected:
                raise RuntimeError(
                    f"Manifest points to the wrong {extension} for {basename}: {declared}"
                )
        declared_sources = entry.get("source_data")
        if not isinstance(declared_sources, list):
            raise RuntimeError(f"Manifest source_data must be a list for {basename}")
        declared_source_names = {Path(str(path)).name for path in declared_sources}
        expected_source_names = set(FIGURE_SOURCE_BY_FIGURE[basename])
        if declared_source_names != expected_source_names:
            raise RuntimeError(
                f"Manifest source_data is incomplete for {basename}: "
                f"expected={sorted(expected_source_names)}, "
                f"found={sorted(declared_source_names)}"
            )
        if not str(entry.get("conclusion", "")).strip():
            raise RuntimeError(f"Manifest has no conclusion for {basename}")
        if "statistics" not in entry:
            raise RuntimeError(f"Manifest has no statistics field for {basename}")

    source_root = FIGURE_ROOT / "source_data"
    for filename in FIGURE_SOURCE_FILES:
        source_path = require_file(source_root / filename)
        source = pd.read_csv(source_path)
        if source.empty:
            raise RuntimeError(f"Figure source data is empty: {source_path}")
    return manifest


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    if number == 0.0:
        return "0"
    if abs(number) < 1.0e-3 or abs(number) >= 1.0e3:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def relative_link(path: Path, label: str) -> str:
    relative = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    return f"[{label}]({relative})"


def baseline_stat(baselines: pd.DataFrame, case: str, model: str, metric: str) -> tuple[float, float, int]:
    values = baselines[
        (baselines["case"] == case) & (baselines["model"] == model)
    ][metric].to_numpy(float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise RuntimeError(f"Cannot compute baseline statistic {case}/{model}/{metric}")
    return float(np.mean(values)), float(np.std(values, ddof=0)), len(values)


def pilot_stat(summary: pd.DataFrame, case: str, metric: str) -> tuple[float, float, int]:
    row = summary[(summary["case"] == case) & (summary["model"] == MODEL)]
    if len(row) != 1:
        raise RuntimeError(f"Missing unique pilot summary row for Case {case}")
    record = row.iloc[0]
    return (
        finite_number(record[f"{metric}_mean"], f"Case {case} {metric}_mean"),
        finite_number(record[f"{metric}_std"], f"Case {case} {metric}_std"),
        int(record["n_seeds"]),
    )


def stat(
    pilot_summary: pd.DataFrame,
    baselines: pd.DataFrame,
    case: str,
    model: str,
    metric: str,
) -> tuple[float, float, int]:
    if model == MODEL:
        return pilot_stat(pilot_summary, case, metric)
    return baseline_stat(baselines, case, model, metric)


def mean_pm(value: tuple[float, float, int]) -> str:
    mean, sd, n = value
    if n == 1:
        return f"{fmt(mean)}（n=1）"
    return f"{fmt(mean)} ± {fmt(sd)}"


def improvement(reference: float, candidate: float) -> float:
    if reference <= 0.0:
        raise RuntimeError("Relative improvement requires a positive reference value")
    return (reference - candidate) / reference


def ratio(candidate: float, reference: float) -> float:
    if reference <= 0.0:
        raise RuntimeError("Error ratio requires a positive reference value")
    return candidate / reference


def build_per_run_tables(runs: pd.DataFrame) -> str:
    core_rows: list[list[str]] = []
    phase_rows: list[list[str]] = []
    branch_rows: list[list[str]] = []
    training_rows: list[list[str]] = []
    for _, row in runs.sort_values(["case", "seed"]).iterrows():
        case = str(row["case"])
        seed = str(int(row["seed"]))
        core_rows.append(
            [
                case,
                seed,
                fmt(row["spinor_rel_l2"]),
                fmt(row["final_spinor_rel_l2"]),
                fmt(row["rho_rel_l2"]),
                fmt(row["current_rel_l2"]),
                "/".join(
                    fmt(row[key], 6)
                    for key in ("fidelity_mean", "fidelity_min", "fidelity_final")
                ),
                f"{fmt(row['relative_phase_mae_rad'])}/{fmt(row['relative_phase_rmse_rad'])}",
            ]
        )
        minus_suffix = "†" if case == "P" else ""
        phase_rows.append(
            [
                case,
                seed,
                f"{fmt(row['phase_plus_rmse_rad'])}/{fmt(row['phase_plus_circular_rmse_rad'])}",
                f"{fmt(row['phase_minus_rmse_rad'])}/{fmt(row['phase_minus_circular_rmse_rad'])}{minus_suffix}",
                f"{fmt(row['phase_plus_slope'])} / {fmt(row['phase_plus_slope_target'])}",
                f"{fmt(row['phase_minus_slope'])} / {fmt(row['phase_minus_slope_target'])}{minus_suffix}",
                f"{fmt(row['phase_plus_fit_r2'])}/{fmt(row['phase_minus_fit_r2'])}{minus_suffix}",
                f"{fmt(row['branch_relative_phase_rmse_rad'])}/{fmt(row['branch_relative_phase_circular_rmse_rad'])}{minus_suffix}",
            ]
        )
        minus_relative = (
            "N/A（a-,0=0）"
            if case == "P"
            else fmt(row["amplitude_minus_drift_rel_rms"])
        )
        branch_rows.append(
            [
                case,
                seed,
                f"{fmt(row['amplitude_plus_drift_abs_rms'])}/{fmt(row['amplitude_plus_drift_rel_rms'])}",
                f"{fmt(row['amplitude_minus_drift_abs_rms'])}/{minus_relative}",
                f"{fmt(row['branch_plus_coefficient_abs_rms'])}/{fmt(row['negative_branch_coefficient_abs_rms'])}",
                f"{fmt(row['branch_plus_occupation_rmse'])}/{fmt(row['branch_minus_occupation_rmse'])}",
                fmt(row["branch_coefficient_rel_l2"]),
                fmt(row["branch_occupation_rmse"]),
            ]
        )
        training_rows.append(
            [
                case,
                seed,
                fmt(row["continuity_normalized_rms"]),
                fmt(row["max_norm_drift"]),
                fmt(row["max_abs_energy_drift"]),
                fmt(row["target_mode_occupation_defect_mean"]),
                fmt(row["final_loss_pde"]),
                str(int(row["parameter_count"])),
                fmt(row["training_seconds"], 1),
            ]
        )

    return "\n\n".join(
        (
            "**核心精度指标**\n\n"
            + md_table(
                [
                    "Case",
                    "seed",
                    "spinor L2",
                    "final L2",
                    "rho L2",
                    "j L2",
                    "F mean/min/final",
                    "旋量相对相位 MAE/RMSE",
                ],
                core_rows,
            ),
            "**学习到的相位诊断**（RMSE 均为 raw/circular，斜率为 fit/target）\n\n"
            + md_table(
                [
                    "Case",
                    "seed",
                    "S+ RMSE",
                    "S- RMSE",
                    "dS+/dt",
                    "dS-/dt",
                    "R2 +/−",
                    "支相对相位 RMSE",
                ],
                phase_rows,
            )
            + "\n\n† Case P 的精确态没有负能支（`a_{-,0}=0`），故 `S_-` 及其斜率不受 PDE 解唯一决定；数值仅为网络输出，不能解释为学会或未学会 `+E`。",
            "**振幅、能支系数与占据**（振幅漂移为 absolute RMS/relative RMS）\n\n"
            + md_table(
                [
                    "Case",
                    "seed",
                    "a+ 漂移",
                    "a- 漂移",
                    "c+/c- abs RMS",
                    "n+/n- RMSE",
                    "branch coeff. L2",
                    "branch occ. RMSE",
                ],
                branch_rows,
            ),
            "**PDE、守恒诊断与运行信息**\n\n"
            + md_table(
                [
                    "Case",
                    "seed",
                    "C normalized RMS",
                    "max norm drift",
                    "max abs E drift",
                    "mode occ. defect",
                    "final PDE",
                    "params",
                    "seconds",
                ],
                training_rows,
            )
            + "\n\nLTPB-CPA 的训练目标只有 PDE；上表守恒量仅作训练后诊断。`final total=final PDE`，四个结构损失字段均严格为 0。",
        )
    )


def build_aggregate_table(pilot_summary: pd.DataFrame, baselines: pd.DataFrame) -> str:
    metrics = (
        ("spinor_rel_l2", "spinor L2"),
        ("final_spinor_rel_l2", "final L2"),
        ("rho_rel_l2", "rho L2"),
        ("current_rel_l2", "j L2"),
        ("fidelity_mean", "F mean"),
        ("relative_phase_rmse_rad", "旋量相对相位 RMSE"),
        ("branch_coefficient_rel_l2", "branch coeff. L2"),
        ("branch_occupation_rmse", "branch occ. RMSE"),
    )
    rows: list[list[str]] = []
    for case in CASES:
        for model in ("vanilla", "sp_cpa", "sp_cpa_conservative", MODEL, "two_branch"):
            values = [stat(pilot_summary, baselines, case, model, metric) for metric, _ in metrics]
            rows.append(
                [case, LABELS[model], str(values[0][2])] + [mean_pm(value) for value in values]
            )
    return md_table(
        ["Case", "Model", "n"] + [label for _, label in metrics],
        rows,
    )


def build_phase_aggregate_table(pilot_summary: pd.DataFrame) -> str:
    metrics = (
        ("phase_plus_rmse_rad", "S+ raw RMSE"),
        ("phase_minus_rmse_rad", "S- raw RMSE"),
        ("phase_plus_slope", "dS+/dt"),
        ("phase_minus_slope", "dS-/dt"),
        ("phase_plus_fit_r2", "R2+"),
        ("phase_minus_fit_r2", "R2-"),
        ("amplitude_plus_drift_rel_rms", "a+ rel drift"),
        ("amplitude_minus_drift_rel_rms", "a- rel drift"),
    )
    rows: list[list[str]] = []
    for case in CASES:
        row = [case]
        for metric, _ in metrics:
            if case == "P" and metric in {
                "phase_minus_rmse_rad",
                "phase_minus_slope",
                "phase_minus_fit_r2",
                "amplitude_minus_drift_rel_rms",
            }:
                row.append("不可辨识†")
            else:
                row.append(mean_pm(pilot_stat(pilot_summary, case, metric)))
        rows.append(row)
    return (
        md_table(["Case"] + [label for _, label in metrics], rows)
        + "\n\n目标斜率为 `dS_+/dt=-E=-5.3851648071`、`dS_-/dt=+E=+5.3851648071`。"
        "† Case P 没有负能支，因此不把 `S_-` 数值纳入载波学习判定。"
    )


def build_run_links(runs: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    for _, row in runs.sort_values(["case", "seed"]).iterrows():
        run_dir = Path(str(row["run_dir"]))
        links = " / ".join(
            relative_link(run_dir / filename, label)
            for filename, label in (
                ("config.json", "config"),
                ("metrics.json", "metrics"),
                ("loss_history.csv", "loss"),
                ("time_series.csv", "time series"),
                ("dense_grid.npz", "dense grid"),
                ("model_state.pt", "state"),
            )
        )
        rows.append([str(row["case"]), str(int(row["seed"])), links])
    return md_table(["Case", "seed", "逐 run 文件"], rows)


def build_figure_section() -> str:
    captions = {
        "01_main_metric_comparison": "主精度指标：LTPB-CPA 与原正式 baselines 的均值及 population SD。",
        "02_learned_branch_phases": "两个学习相位及拟合斜率；Case P 的负支不参与判定。",
        "03_case_M_current_x0": "Case M 在 x=0 的概率流时间曲线。",
        "04_case_M_relative_phase_x0": "Case M 在 x=0 的旋量分量相对相位时间曲线。",
        "05_amplitude_and_occupations": "实振幅漂移和正负能支占据。",
        "06_loss_histories": "PDE-only 训练轨迹；训练损失不替代场精度诊断。",
    }
    blocks: list[str] = []
    source_root = FIGURE_ROOT / "source_data"
    for index, basename in enumerate(FIGURE_BASENAMES, start=1):
        png = relative_link(FIGURE_ROOT / f"{basename}.png", "PNG")
        pdf = relative_link(FIGURE_ROOT / f"{basename}.pdf", "PDF")
        svg = relative_link(FIGURE_ROOT / f"{basename}.svg", "SVG")
        sources = ", ".join(
            relative_link(source_root / filename, filename)
            for filename in FIGURE_SOURCE_BY_FIGURE[basename]
        )
        image_path = (FIGURE_ROOT / f"{basename}.png").resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        blocks.append(
            f"### 图 {index}. {captions[basename]}\n\n"
            f"![{captions[basename]}]({image_path})\n\n"
            f"文件：{png} / {pdf} / {svg}；作图源数据：{sources}。"
        )
    return "\n\n".join(blocks)


def decision_payload(pilot_summary: pd.DataFrame, baselines: pd.DataFrame) -> dict[str, Any]:
    p_ltpb_spinor = pilot_stat(pilot_summary, "P", "spinor_rel_l2")[0]
    p_sp_spinor = baseline_stat(baselines, "P", "sp_cpa", "spinor_rel_l2")[0]
    p_ltpb_fidelity = pilot_stat(pilot_summary, "P", "fidelity_mean")[0]
    p_sp_fidelity = baseline_stat(baselines, "P", "sp_cpa", "fidelity_mean")[0]
    p_spinor_ratio = ratio(p_ltpb_spinor, p_sp_spinor)
    p_fidelity_delta = p_ltpb_fidelity - p_sp_fidelity
    p_pass = (
        p_spinor_ratio <= P_NOT_WORSE_SPINOR_RATIO
        and p_fidelity_delta >= -P_NOT_WORSE_FIDELITY_DROP
    )

    m_error_metrics = ("spinor_rel_l2", "current_rel_l2", "relative_phase_rmse_rad")
    m_ratios: dict[str, float] = {}
    for metric in m_error_metrics:
        ltpb = pilot_stat(pilot_summary, "M", metric)[0]
        sp = baseline_stat(baselines, "M", "sp_cpa", metric)[0]
        m_ratios[metric] = ratio(ltpb, sp)
    m_ltpb_fidelity = pilot_stat(pilot_summary, "M", "fidelity_mean")[0]
    m_sp_fidelity = baseline_stat(baselines, "M", "sp_cpa", "fidelity_mean")[0]
    m_fidelity_gain = m_ltpb_fidelity - m_sp_fidelity
    m_directional = all(value < 1.0 for value in m_ratios.values()) and m_fidelity_gain > 0.0
    m_material = (
        all(value <= M_MATERIAL_ERROR_RATIO for value in m_ratios.values())
        and m_fidelity_gain >= M_MATERIAL_FIDELITY_GAIN
    )

    energy = math.sqrt(29.0)
    plus_slope = pilot_stat(pilot_summary, "M", "phase_plus_slope")[0]
    minus_slope = pilot_stat(pilot_summary, "M", "phase_minus_slope")[0]
    plus_r2 = pilot_stat(pilot_summary, "M", "phase_plus_fit_r2")[0]
    minus_r2 = pilot_stat(pilot_summary, "M", "phase_minus_fit_r2")[0]
    plus_relative_error = abs(plus_slope + energy) / energy
    minus_relative_error = abs(minus_slope - energy) / energy
    carrier_pass = (
        plus_relative_error <= CARRIER_RELATIVE_SLOPE_ERROR
        and minus_relative_error <= CARRIER_RELATIVE_SLOPE_ERROR
        and plus_r2 >= CARRIER_MIN_R2
        and minus_r2 >= CARRIER_MIN_R2
    )

    oracle_ratios: dict[str, float] = {}
    for metric in m_error_metrics:
        ltpb = pilot_stat(pilot_summary, "M", metric)[0]
        oracle = baseline_stat(baselines, "M", "two_branch", metric)[0]
        oracle_ratios[metric] = ratio(ltpb, oracle)
    oracle_fidelity = baseline_stat(baselines, "M", "two_branch", "fidelity_mean")[0]
    oracle_close = (
        all(value <= ORACLE_PROXIMITY_ERROR_RATIO for value in oracle_ratios.values())
        and abs(oracle_fidelity - m_ltpb_fidelity) <= ORACLE_PROXIMITY_FIDELITY_GAP
    )

    rho_ratio = ratio(
        pilot_stat(pilot_summary, "M", "rho_rel_l2")[0],
        baseline_stat(baselines, "M", "sp_cpa", "rho_rel_l2")[0],
    )
    norm_ratio = ratio(
        pilot_stat(pilot_summary, "M", "max_norm_drift")[0],
        baseline_stat(baselines, "M", "sp_cpa", "max_norm_drift")[0],
    )
    only_density_or_norm = (rho_ratio <= M_MATERIAL_ERROR_RATIO or norm_ratio <= M_MATERIAL_ERROR_RATIO) and not m_directional

    if p_pass and m_material and carrier_pass:
        verdict = "支持"
        verdict_text = (
            "按预先编码的数值规则，本试验支持“双能支可学习相位能在固定预算下解决公共单相位载波失配”这一初步假设。"
        )
    elif m_material and carrier_pass and not p_pass:
        verdict = "Case M 支持、总体判据未通过"
        verdict_text = (
            "Case M 的核心误差和两个载波学习达到规则，但 Case P 的“不明显差于 SP-CPA”门槛未通过；"
            "因此支持混合支机制解释，不支持把当前实现判为全面成功。"
        )
    elif m_directional:
        verdict = "部分支持"
        verdict_text = (
            "LTPB-CPA 在 Case M 的全部核心指标方向一致地优于 SP-CPA，但没有同时达到预设的效应量、"
            "Case P 兼容性和载波斜率门槛；证据为部分支持。"
        )
    elif only_density_or_norm:
        verdict = "不支持：只改善密度或范数"
        verdict_text = (
            "LTPB-CPA 只在密度或范数诊断上达到改善门槛，而旋量、概率流、旋量相对相位和 fidelity "
            "没有一致改善，故按核心判据判为未解决。"
        )
    else:
        verdict = "不支持"
        verdict_text = (
            "LTPB-CPA 未使 Case M 的旋量、概率流、旋量相对相位和 fidelity 一致改善，故本轮不支持核心假设。"
        )

    return {
        "verdict": verdict,
        "verdict_text": verdict_text,
        "p_pass": p_pass,
        "p_spinor_ratio": p_spinor_ratio,
        "p_fidelity_delta": p_fidelity_delta,
        "m_ratios": m_ratios,
        "m_fidelity_gain": m_fidelity_gain,
        "m_directional": m_directional,
        "m_material": m_material,
        "carrier_pass": carrier_pass,
        "plus_slope": plus_slope,
        "minus_slope": minus_slope,
        "plus_relative_error": plus_relative_error,
        "minus_relative_error": minus_relative_error,
        "plus_r2": plus_r2,
        "minus_r2": minus_r2,
        "oracle_close": oracle_close,
        "oracle_ratios": oracle_ratios,
        "oracle_fidelity_gap": abs(oracle_fidelity - m_ltpb_fidelity),
        "rho_ratio": rho_ratio,
        "norm_ratio": norm_ratio,
    }


def build_failure_analysis(
    decision: dict[str, Any], pilot_summary: pd.DataFrame, baselines: pd.DataFrame
) -> str:
    observations: list[str] = []
    if decision["carrier_pass"]:
        observations.append(
            "Case M 两个空间平均相位斜率及线性度达到载波门槛；若核心场指标仍不足，失败不能简单归为频率没有学到。"
        )
    else:
        observations.append(
            "Case M 至少一个相位斜率或线性拟合 R2 未达到载波门槛，直接证据指向相位学习/长期时间优化尚未完成。"
        )

    plus_drift = pilot_stat(pilot_summary, "M", "amplitude_plus_drift_rel_rms")[0]
    minus_drift = pilot_stat(pilot_summary, "M", "amplitude_minus_drift_rel_rms")[0]
    occupation = pilot_stat(pilot_summary, "M", "branch_occupation_rmse")[0]
    if max(plus_drift, minus_drift) > 0.25 or occupation > 0.10:
        observations.append(
            f"Case M 的平均相对振幅漂移为 `a+={fmt(plus_drift)}`、`a-={fmt(minus_drift)}`，"
            f"branch occupation RMSE 为 `{fmt(occupation)}`；实振幅通道发生了不可忽略的补偿/漂移。"
        )
    else:
        observations.append(
            f"Case M 的平均相对振幅漂移为 `a+={fmt(plus_drift)}`、`a-={fmt(minus_drift)}`，"
            f"branch occupation RMSE 为 `{fmt(occupation)}`，未显示大幅占据重排。"
        )

    ltpb_pde = pilot_stat(pilot_summary, "M", "final_loss_pde")[0]
    sp_pde = baseline_stat(baselines, "M", "sp_cpa", "final_loss_pde")[0]
    if ltpb_pde >= sp_pde:
        observations.append(
            f"最终 PDE loss 的均值 `{fmt(ltpb_pde)}` 不低于 SP-CPA 的 `{fmt(sp_pde)}`，"
            "与固定 5000-step Adam 下的优化困难一致；该比较本身不构成因果证明。"
        )
    else:
        observations.append(
            f"最终 PDE loss 的均值 `{fmt(ltpb_pde)}` 低于 SP-CPA 的 `{fmt(sp_pde)}`；"
            "若物理指标仍差，说明不能用训练损失单独判定成功。"
        )
    observations.append(
        "本轮没有依据结果调学习率、步数、损失权重或网络宽度，也不做事后择优重跑，因此无法把失败排除为可调参问题。"
    )
    return "\n".join(f"- {observation}" for observation in observations)


def build_report(
    runs: pd.DataFrame,
    pilot_summary: pd.DataFrame,
    baselines: pd.DataFrame,
    manifest: dict[str, Any],
) -> str:
    del manifest  # Completeness is validated above; report links use the fixed contract.
    decision = decision_payload(pilot_summary, baselines)
    energy = math.sqrt(29.0)

    p_v = baseline_stat(baselines, "P", "vanilla", "spinor_rel_l2")[0]
    p_sp = baseline_stat(baselines, "P", "sp_cpa", "spinor_rel_l2")[0]
    m_v = baseline_stat(baselines, "M", "vanilla", "spinor_rel_l2")[0]
    m_sp = baseline_stat(baselines, "M", "sp_cpa", "spinor_rel_l2")[0]
    m_cons = baseline_stat(baselines, "M", "sp_cpa_conservative", "spinor_rel_l2")[0]
    m_oracle = baseline_stat(baselines, "M", "two_branch", "spinor_rel_l2")[0]

    p_gain = improvement(p_v, p_sp)
    m_gain = improvement(m_v, m_sp)
    ltpb_m_spinor = pilot_stat(pilot_summary, "M", "spinor_rel_l2")[0]
    ltpb_m_current = pilot_stat(pilot_summary, "M", "current_rel_l2")[0]
    ltpb_m_phase = pilot_stat(pilot_summary, "M", "relative_phase_rmse_rad")[0]
    ltpb_m_fidelity = pilot_stat(pilot_summary, "M", "fidelity_mean")[0]
    sp_m_current = baseline_stat(baselines, "M", "sp_cpa", "current_rel_l2")[0]
    sp_m_phase = baseline_stat(baselines, "M", "sp_cpa", "relative_phase_rmse_rad")[0]
    sp_m_fidelity = baseline_stat(baselines, "M", "sp_cpa", "fidelity_mean")[0]

    oracle_description = (
        "达到本报告定义的“接近 oracle”门槛"
        if decision["oracle_close"]
        else "未达到本报告定义的“接近 oracle”门槛"
    )
    p_description = "通过" if decision["p_pass"] else "未通过"
    carrier_description = "通过" if decision["carrier_pass"] else "未通过"

    aggregate_table = build_aggregate_table(pilot_summary, baselines)
    phase_aggregate_table = build_phase_aggregate_table(pilot_summary)
    per_run_tables = build_per_run_tables(runs)
    run_links = build_run_links(runs)
    figure_section = build_figure_section()
    failure_analysis = build_failure_analysis(decision, pilot_summary, baselines)

    report = rf"""# 可学习正负能支双相位 Dirac-PINN 初步数值实验

## 摘要

本报告在原公共单相位 CPA-to-Dirac 实验的固定配置上，新增 Learned Two-Phase Branch CPA（LTPB-CPA）。正式数据严格包括 Case P/M × seeds `{{0,1,2}}` 六个 5000-step run；脚本在逐 run 文件、聚合表、原 baseline 表或六组三格式图不全时拒绝生成报告。结论不是按训练 loss 或单个 seed 给出，而由第 4.2 节的固定数值规则产生。

**自动判定：{decision['verdict']}。** {decision['verdict_text']}

数据入口：{relative_link(PILOT_SUMMARY_PATH, 'LTPB 聚合 CSV')}；{relative_link(BASELINE_SUMMARY_PATH, '原正式 summary.csv')}；{relative_link(FIGURE_ROOT / 'figure_manifest.json', '新图 manifest')}。原报告 {relative_link(PROJECT_ROOT / 'EXPERIMENT_REPORT.md', 'EXPERIMENT_REPORT.md')} 未被覆盖。

## 1. 已有实验结论

原实验已经区分了数学表达能力和有限预算优化效果：

1. Case P 中，SP-CPA 的三 seed 平均 spinor L2 从 Vanilla 的 `{fmt(p_v)}` 降为 `{fmt(p_sp)}`，改善 `{100.0 * p_gain:.1f}%`，说明单一正能载波的预条件很有效。
2. Case M 中，同一改善只从 `{fmt(m_v)}` 到 `{fmt(m_sp)}`，即 `{100.0 * m_gain:.1f}%`；公共单相位仍有完整表示能力，但不能同时慢化 `exp(-iEt)` 与 `exp(+iEt)`，固定预算优势明显退化。
3. Case M 的 SP-CPA-Conservative 平均 spinor L2 为 `{fmt(m_cons)}`。连续性、总范数和总能量只约束总量，对正交能支的相对相位存在盲方向；因此守恒改善不等于概率流、相对相位或旋量恢复。
4. Two-Branch oracle 的 Case M spinor L2 为 `{fmt(m_oracle)}`，说明给出正确双载波后问题可被高精度优化。但它把精确 `E` 和 `exp(∓iEt)` 写入 forward，只能诊断机制，不能作为新方法。
5. 所以原结论是**载波预条件与优化优势退化**，不是“公共单相位在数学上无法表示混合态”。

## 2. LTPB-CPA 的理论动机和公式

已知自由 Dirac 符号

\[
H_k=k\sigma_x+m\sigma_z
\]

给出正交能支基底 `u_+、u_-`。LTPB-CPA 写作

\[
\widehat\psi(x,t)=e^{{ikx}}\left[
a_+(x,t)u_+e^{{iS_+(x,t)}}+a_-(x,t)u_-e^{{iS_-(x,t)}}
\right].
\]

4×64 `tanh` MLP 的输入为 `[sin(x),cos(x),t/T]`，输出四个实函数 `(r_+,r_-,q_+,q_-)`。令 `tau=t/T`，

\[
a_+=a_{{+,0}}+\tau r_+,
\qquad a_-=a_{{-,0}}+\tau r_-,
\]

\[
S_+=\tau q_+,
\qquad S_-=\tau q_-,
\]

\[
a_{{+,0}}=\frac{{1}}{{\sqrt{{2\pi(1+\eta^2)}}}},
\qquad
a_{{-,0}}=\frac{{\eta}}{{\sqrt{{2\pi(1+\eta^2)}}}}.
\]

因此 `t=0` 的原旋量初值严格满足。振幅为可取正负的实数，防止快速时间相位藏进任意复振幅；同时仍存在离散的 `(a,S)→(-a,S+pi)` 等价，需要 raw 与 circular phase error 一起检查。

训练只最小化强形式 Dirac PDE 残差

\[
L=L_{{PDE}}=\operatorname{{mean}}|i\psi_t+i\sigma_x\psi_x-m\sigma_z\psi|^2.
\]

forward 使用从已知 `H_k` 计算的 `u_±`，但不使用精确能量 `{energy:.10f}`、目标斜率或解析时间载波。复用的配置快照为可复现性记录了谱本征值；该数值不进入 LTPB 的 forward、loss 或 optimizer，其计算用途仅在训练后评价 phase slope target、解析解和误差。

## 3. 与 Two-Branch oracle 的关键区别

| 项目 | LTPB-CPA | Two-Branch oracle |
|---|---|---|
| 能支基底 | 从已知 `H_k` 计算 `u_±` | 同样使用 `u_±` |
| 时间频率 | `S_±` 由 PDE 残差学习 | forward 硬编码精确 `∓E` |
| 时间载波 | `exp(iS_±)` | 硬编码 `exp(∓iEt)` |
| 振幅 | 两个有符号实函数 | 两个复修正系数 |
| 初值 | 硬满足 | 硬满足 |
| 角色 | 待检验的新参数化 | 机制诊断上界，不参与公平排名 |

LTPB-CPA 即使接近 oracle，也仍依赖已知能支基底这一强物理先验。它检验的是“不给频率时，PDE 能否在该基底中学出两相位”，不是无先验发现正负能支。

## 4. 实验配置、公平性与数值判定规则

### 4.1 固定配置

| 项目 | 值 |
|---|---:|
| `m,k,T` | `5,2,5` |
| dtype | `float64/complex128` |
| 输入 | `[sin(x),cos(x),t/T]` |
| MLP | `4 × 64 tanh`，共享 trunk，输出 4 实数 |
| 初始化 | 所有线性层 Xavier normal，bias=0 |
| 配点 | 每步重新采样 1024 点 |
| optimizer | Adam，`lr=1e-3` |
| steps | `5000` |
| seeds | `{{0,1,2}}` |
| test grid | `201 × 128 (t × x)` |
| loss | 仅 `L_PDE` |
| 参数量 | LTPB-CPA `12,996`；Vanilla/Two-Branch `12,996`；SP-CPA `13,061` |

生成脚本逐项核对六个 config，并逐 seed 与 SP-CPA 核对区域、`m,k,T`、dtype、网络宽深、学习率、步数、配点数、测试网格、Xavier 初始化以及 collocation/structure 随机源。CPU 线程数和日志频率不进入算法公平性判断。LTPB 没有增加容量，且每个 seed 使用与旧实验相同的配点随机序列。

### 4.2 固定数值规则

为了不事后选择“成功”口径，本脚本使用如下可复现阈值：

1. Case P“不明显差”：`spinor_L2(LTPB)/spinor_L2(SP) <= {P_NOT_WORSE_SPINOR_RATIO:.2f}`，且 mean fidelity 降幅不超过 `{P_NOT_WORSE_FIDELITY_DROP:.2f}`。
2. Case M“实质改善”：spinor L2、current L2、旋量分量 relative-phase RMSE 三项都至少降低 `{100.0 * (1.0 - M_MATERIAL_ERROR_RATIO):.0f}%`，且 mean fidelity 至少提高 `{M_MATERIAL_FIDELITY_GAIN:.2f}`。
3. “学到 ±E 载波”：Case M 两个平均 slope 的相对误差均不超过 `{100.0 * CARRIER_RELATIVE_SLOPE_ERROR:.0f}%`，且两个线性拟合 mean R2 均不低于 `{CARRIER_MIN_R2:.2f}`。Case P 的 `S_-` 不参与该判定。
4. “接近 oracle”只作描述：三项 Case M 核心误差均不超过 oracle 的 `{ORACLE_PROXIMITY_ERROR_RATIO:.0f}` 倍，且 fidelity 绝对差不超过 `{ORACLE_PROXIMITY_FIDELITY_GAP:.2f}`。
5. 所有 `±` 均为三 seed 的 **population SD (`ddof=0`)**。Two-Branch 只有预注册 seed 0，明确标作 `n=1`，不伪装成不确定性估计。

## 5. 六个正式 run 的完整结果

{per_run_tables}

### 5.1 可复核文件

{run_links}

## 6. 均值、population SD 与 baseline 比较

{aggregate_table}

### 6.1 LTPB 相位与振幅聚合

{phase_aggregate_table}

## 7. 学到的两个相位斜率

Case M 的空间平均拟合得到 `dS_+/dt={fmt(decision['plus_slope'])}`、`dS_-/dt={fmt(decision['minus_slope'])}`，目标分别为 `{-energy:.10f}` 与 `{energy:.10f}`；相对误差为 `{100.0 * decision['plus_relative_error']:.2f}%`、`{100.0 * decision['minus_relative_error']:.2f}%`，mean R2 为 `{fmt(decision['plus_r2'])}`、`{fmt(decision['minus_r2'])}`。按第 4.2 节，载波学习门槛**{carrier_description}**。

Case P 只把 `S_+` 视为可辨识量。由于 `eta=0` 时负支系数恒为零，任意 `S_-` 都不改变 `psi_hat`；报告保留其原始数值供调试，却不拿它与 `+E` 的接近程度支持或反驳模型。

## 8. Case M：概率流与旋量相对相位

与 SP-CPA 相比，LTPB-CPA 的三 seed 平均：

- spinor L2：`{fmt(m_sp)} -> {fmt(ltpb_m_spinor)}`，误差比 `{fmt(decision['m_ratios']['spinor_rel_l2'])}`；
- current L2：`{fmt(sp_m_current)} -> {fmt(ltpb_m_current)}`，误差比 `{fmt(decision['m_ratios']['current_rel_l2'])}`；
- 旋量分量 relative-phase RMSE：`{fmt(sp_m_phase)} -> {fmt(ltpb_m_phase)}` rad，误差比 `{fmt(decision['m_ratios']['relative_phase_rmse_rad'])}`；
- mean fidelity：`{fmt(sp_m_fidelity, 6)} -> {fmt(ltpb_m_fidelity, 6)}`，绝对变化 `{decision['m_fidelity_gain']:+.4f}`。

这里的“旋量分量 relative phase”来自 `arg(conj(psi_1) psi_2)`，不同于模型内部 `S_- - S_+`；二者均已分别诊断。时间曲线见图 3 和图 4，不能由训练 loss 或密度曲线替代。

## 9. 是否支持双能支相位方案

**结论：{decision['verdict']}。** {decision['verdict_text']}

- Case P 兼容性门槛**{p_description}**：spinor error ratio `{fmt(decision['p_spinor_ratio'])}`，fidelity 变化 `{decision['p_fidelity_delta']:+.4f}`。
- Case M 三个核心 error ratios 为 spinor `{fmt(decision['m_ratios']['spinor_rel_l2'])}`、current `{fmt(decision['m_ratios']['current_rel_l2'])}`、旋量相对相位 `{fmt(decision['m_ratios']['relative_phase_rmse_rad'])}`；fidelity 变化 `{decision['m_fidelity_gain']:+.4f}`。
- 两个 Case M 相位 slope 的载波门槛**{carrier_description}**。
- 相对 Two-Branch oracle，三个核心误差倍数为 `{fmt(decision['oracle_ratios']['spinor_rel_l2'])}`、`{fmt(decision['oracle_ratios']['current_rel_l2'])}`、`{fmt(decision['oracle_ratios']['relative_phase_rmse_rad'])}`，fidelity 绝对差 `{fmt(decision['oracle_fidelity_gap'])}`；因此{oracle_description}。

补充数值诊断（无论门槛是否全部通过均保留）：

{failure_analysis}

### 9.1 五类概念必须分开

- **表达能力**：公共单相位乘任意复旋量振幅仍能表示混合态；LTPB 的结果不能反证这一点。LTPB 的实振幅限制反而缩小了表示类，但解析双相位目标仍在该类中。
- **载波预条件**：LTPB 把正负能支放进不同相位通道；只有 slope 和场指标同时改善，才能说这一预条件被 PDE 学会并发挥作用。
- **优化难度**：5000-step Adam 的结果包含长时域 PINN 优化、相位回归、实振幅补偿和随机初始化效应；训练 loss 不能单独定位原因。
- **守恒量盲点**：连续性、总范数和总能量无法唯一识别能支相对相位。LTPB 本轮没训练这些损失，训练后守恒指标仅作诊断。
- **已知 `u_±` 先验**：LTPB 不泄露时间频率，却显式使用自由 Dirac 符号的能支基底；这是重要的结构先验，不能与无先验 PINN 直接等同。

## 10. 当前实验不能推出的结论

1. 不能推出公共单相位在数学上不能表示正负能支混合；现有结果只涉及固定网络和优化预算。
2. 不能推出 LTPB 在多 `k` 波包、外势、非周期边界、更长时间或更大质量下仍有效。
3. 不能推出 PDE 从原始旋量中“自动发现”了能支，因为 `u_±` 已由已知 `H_k` 提供。
4. 不能把 Case P 的 `S_-` 当作可辨识参数，也不能把其偶然接近 `+Et` 当成证据。
5. 不能由小 density/norm error 或小 PDE training loss 单独推出概率流、相对相位和旋量正确。
6. 不能把单个 Two-Branch oracle seed 与三 seed learned models 做统计优越性检验。
7. 不能从本轮判断实振幅限制是最优选择；外势耦合时复分支包络可能是必要的。

## 11. 扩展到多 k 波包和光滑外势的下一步

1. **多 k 分解**：平面波的固定 `u_±(k)` 要扩展成动量依赖投影器或谱通道；必须控制离散 Fourier 分辨率、不同 `k` 的色散相位和波包群速度，而不能给每个训练样本泄露解析 `E(k)` 载波。
2. **光滑外势与分支耦合**：`V(x)` 下动量不守恒，局部能支基底随 `x` 变化并产生基底导数/非绝热耦合。需要决定采用全局谱投影、局部符号投影还是可学习 projector，并把相应连接项纳入 PDE 残差。
3. **规范与可辨识性**：对每个 `k` 或局部基底固定本征矢规范；处理 `(a,S)` 的符号-相位离散冗余、相位 unwrap 和支振幅过零，否则 slope 指标会失真。
4. **实振幅限制**：外势散射会改变支间幅度并可能产生包络相位。下一轮应比较实振幅、受控慢复包络和显式 gauge fixing，防止快速频率再次藏回复振幅。
5. **优化与时间推进**：在不调参追逐有利结果的前提下，预注册比较 causal sampling、time marching、分段时间窗和匹配参数量的双 Fourier-time baseline，区分表示增益与长时 PINN 优化增益。
6. **物理诊断**：继续同时报告 spinor、current、旋量相对相位、能支系数/占据、fidelity 和守恒量；含势时还需反射/透射、正负频转换和局部能量交换指标。

## 12. 图表与可编辑源

{figure_section}

---

本报告由 `python -m src.build_learned_two_phase_pilot_report` 从六个正式 run、两个汇总 CSV 和图 manifest 确定性生成。生成器会验证聚合数值与逐 run metrics 一致，并在任何要求的数据或图缺失时失败；它不启动训练，也不修改原 `EXPERIMENT_REPORT.md`。
"""
    return report


def main() -> None:
    runs, configs = load_and_validate_pilot_runs()
    pilot_summary = load_and_validate_pilot_summary(runs)
    baselines = load_and_validate_baselines(configs)
    manifest = load_and_validate_figures()
    report = build_report(runs, pilot_summary, baselines, manifest)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
