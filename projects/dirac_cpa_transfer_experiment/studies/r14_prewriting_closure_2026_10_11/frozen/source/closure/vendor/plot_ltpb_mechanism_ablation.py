from __future__ import annotations

"""Publication figures for the formal LTPB mechanism-ablation experiment.

The module reads only the isolated 15-run experiment and writes only to
``results/figures/ltpb_mechanism_ablation``.  Every figure has an independent
source-data CSV and is exported as a 600 dpi PNG plus editable-text PDF/SVG.
Training is never started by this module.
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .ltpb_mechanism_decision import (
    ABLATION_MODELS,
    ABLATION_SEEDS,
    CORE_METRICS,
    ROTATION_30_RATIO_MIN,
    compute_mechanism_decision,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
RUNS_PATH = RESULTS_ROOT / "ltpb_mechanism_ablation_runs.csv"
SUMMARY_PATH = RESULTS_ROOT / "ltpb_mechanism_ablation_summary.csv"
DECISION_PATH = RESULTS_ROOT / "ltpb_mechanism_ablation_decision.json"
RAW_ROOT = RESULTS_ROOT / "raw" / "ltpb_mechanism_ablation"
FIGURE_ROOT = RESULTS_ROOT / "figures" / "ltpb_mechanism_ablation"
SOURCE_ROOT = FIGURE_ROOT / "source_data"

EXPERIMENT = "ltpb_mechanism_ablation"
CASE = "M"
FIGURE_WIDTH_MM = 183.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4
RASTER_DPI = 600
PLOT_FLOOR = 1.0e-12

MODEL_LABELS = {
    "ltpb_cpa": "LTPB-CPA",
    "branch_complex": "Branch-Complex",
    "component_polar": "Component-Polar",
    "rotated_ltpb_15": r"Rotated-LTPB $15^\circ$",
    "rotated_ltpb_30": r"Rotated-LTPB $30^\circ$",
}
MODEL_TICK_LABELS = {
    "ltpb_cpa": "LTPB-CPA",
    "branch_complex": "Branch-\nComplex",
    "component_polar": "Component-\nPolar",
    "rotated_ltpb_15": "Rotated\n15 deg",
    "rotated_ltpb_30": "Rotated\n30 deg",
}
MODEL_COLORS = {
    "ltpb_cpa": "#2F65A7",
    "branch_complex": "#4F99A4",
    "component_polar": "#D3843F",
    "rotated_ltpb_15": "#A17AB5",
    "rotated_ltpb_30": "#6E4C8B",
}
MODEL_MARKERS = {
    "ltpb_cpa": "o",
    "branch_complex": "s",
    "component_polar": "^",
    "rotated_ltpb_15": "D",
    "rotated_ltpb_30": "P",
}
SEED_COLORS = {0: "#4C78A8", 1: "#F28E2B", 2: "#59A14F"}
EXACT_COLOR = "#202020"
GRID_COLOR = "#D9D9D9"

CORE_METRIC_LABELS = {
    "spinor_rel_l2": "Spinor relative L2",
    "current_rel_l2": "Current relative L2",
    "relative_phase_rmse_rad": "Component relative-phase RMSE (rad)",
}

SUPPORT_METRICS = (
    ("final_spinor_rel_l2", "Final-time spinor L2"),
    ("rho_rel_l2", "Density relative L2"),
    ("infidelity_mean", "Mean infidelity (1-F)"),
    ("branch_coefficient_rel_l2", "Energy-branch coefficient L2"),
    ("branch_occupation_rmse", "Energy-branch occupation RMSE"),
    ("final_loss_pde", "Final PDE loss"),
)

COMPLEXITY_METRICS = (
    (
        "internal_amplitude_drift_rel_rms",
        "internal_exact_amplitude_drift_rel_rms",
        "Amplitude drift (relative RMS)",
    ),
    (
        "internal_phase_velocity_fluctuation_over_energy",
        "internal_exact_phase_velocity_fluctuation_over_energy",
        r"Phase-velocity fluctuation / $E$",
    ),
    (
        "internal_phase_curvature_over_energy_squared",
        "internal_exact_phase_curvature_over_energy_squared",
        r"Weighted phase curvature / $E^2$",
    ),
)

EXPECTED_CONFIG: dict[str, Any] = {
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
    "dtype": "float64",
}

TIME_SERIES_COLUMNS = (
    "t",
    "current_x0_predicted",
    "current_x0_exact",
    "relative_phase_x0_predicted",
    "relative_phase_x0_exact",
    "internal_channel_1_amplitude_spatial_mean",
    "internal_exact_channel_1_amplitude_spatial_mean",
    "internal_channel_2_amplitude_spatial_mean",
    "internal_exact_channel_2_amplitude_spatial_mean",
)
LOSS_COLUMNS = ("step", "total", "pde")
DENSE_FIELDS = (
    "x",
    "t",
    "current_predicted",
    "current_exact",
    "relative_phase_predicted",
    "relative_phase_exact",
    "internal_channel_amplitude_predicted",
    "internal_channel_amplitude_exact",
)

REQUIRED_RUN_METRICS = {
    *CORE_METRICS,
    "final_spinor_rel_l2",
    "rho_rel_l2",
    "fidelity_mean",
    "branch_coefficient_rel_l2",
    "branch_occupation_rmse",
    "final_loss_pde",
    "final_loss_total",
    "parameter_count",
    "training_seconds",
    *[item[0] for item in COMPLEXITY_METRICS],
    *[item[1] for item in COMPLEXITY_METRICS],
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.titlesize": 7.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 6.0,
        "axes.linewidth": 0.75,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    }
)


RunKey = tuple[str, int]


@dataclass(frozen=True)
class AblationInputs:
    runs: pd.DataFrame
    summary: pd.DataFrame
    decision: dict[str, Any]
    time_series: dict[RunKey, pd.DataFrame]
    loss_history: dict[RunKey, pd.DataFrame]
    dense_grid: dict[RunKey, dict[str, np.ndarray]]
    configs: dict[RunKey, dict[str, Any]]


def _run_dir(model: str, seed: int) -> Path:
    return RAW_ROOT / "case_M" / model / f"seed_{seed}"


def _read_json(path: Path, context: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {context}: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in {context}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {context}: {path}")
    return payload


def _read_csv(path: Path, columns: Iterable[str], context: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {context}: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        raise RuntimeError(f"Could not parse {context}: {path}: {error}") from error
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{context} is missing columns {missing}: {path}")
    if frame.empty:
        raise RuntimeError(f"{context} contains no rows: {path}")
    return frame


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(
                float(actual), expected, rel_tol=1.0e-12, abs_tol=1.0e-12
            )
        except (TypeError, ValueError):
            return False
    return actual == expected


def _assert_finite(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    names = list(columns)
    try:
        values = frame[names].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Non-numeric required values in {context}: {error}") from error
    if not np.all(np.isfinite(values)):
        bad = np.argwhere(~np.isfinite(values))
        raise RuntimeError(f"NaN or infinite required values in {context}: {bad[:10].tolist()}")


def _validate_runs(frame: pd.DataFrame) -> pd.DataFrame:
    identity = {"experiment", "case", "model", "seed", "steps"}
    missing = sorted((identity | REQUIRED_RUN_METRICS) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Run table is missing columns {missing}: {RUNS_PATH}")
    if len(frame) != len(ABLATION_MODELS) * len(ABLATION_SEEDS):
        raise RuntimeError(f"Expected exactly 15 formal runs, found {len(frame)}")
    pairs = list(zip(frame["model"], frame["seed"].astype(int), strict=False))
    expected = {(model, seed) for model in ABLATION_MODELS for seed in ABLATION_SEEDS}
    if set(pairs) != expected or len(set(pairs)) != len(pairs):
        raise RuntimeError(
            "Formal run identities do not equal five models x three seeds: "
            f"found={sorted(pairs)}"
        )
    if set(frame["experiment"].astype(str)) != {EXPERIMENT}:
        raise RuntimeError("Run table contains a non-ablation experiment")
    if set(frame["case"].astype(str)) != {CASE}:
        raise RuntimeError("Run table contains a case other than Case M")
    if set(frame["steps"].astype(int)) != {5000}:
        raise RuntimeError("Every formal run must have 5000 steps")
    if set(frame["parameter_count"].astype(int)) != {12_996}:
        raise RuntimeError("Every ablation model must have exactly 12,996 parameters")
    _assert_finite(frame, REQUIRED_RUN_METRICS, "formal run table")
    if (frame[list(CORE_METRICS)].to_numpy(float) < 0.0).any():
        raise RuntimeError("Core error metrics must be non-negative")
    fidelity = frame["fidelity_mean"].to_numpy(float)
    if ((fidelity < -1.0e-10) | (fidelity > 1.0 + 1.0e-8)).any():
        raise RuntimeError("Mean fidelity lies outside the numerical [0,1] range")
    if not np.allclose(
        frame["final_loss_total"].to_numpy(float),
        frame["final_loss_pde"].to_numpy(float),
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise RuntimeError("Formal ablation run table is not PDE-only")
    order = {model: index for index, model in enumerate(ABLATION_MODELS)}
    result = frame.copy()
    result["_model_order"] = result["model"].map(order)
    return (
        result.sort_values(["_model_order", "seed"])
        .drop(columns="_model_order")
        .reset_index(drop=True)
    )


def _validate_summary(frame: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    required_metrics = REQUIRED_RUN_METRICS - {"parameter_count", "training_seconds"}
    required = {"case", "model", "n_seeds"}
    for metric in required_metrics:
        required.update((f"{metric}_mean", f"{metric}_std"))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Summary table is missing columns {missing}: {SUMMARY_PATH}")
    if len(frame) != len(ABLATION_MODELS) or set(frame["model"]) != set(ABLATION_MODELS):
        raise RuntimeError("Summary must contain exactly one row for each of the five models")
    if set(frame["case"].astype(str)) != {CASE} or not frame["n_seeds"].eq(3).all():
        raise RuntimeError("Every summary row must be Case M with n_seeds=3")
    for model in ABLATION_MODELS:
        row = frame[frame["model"] == model].iloc[0]
        subset = runs[runs["model"] == model]
        for metric in required_metrics:
            values = subset[metric].to_numpy(float)
            expected_mean = float(np.mean(values))
            expected_std = float(np.std(values, ddof=0))
            actual_mean = float(row[f"{metric}_mean"])
            actual_std = float(row[f"{metric}_std"])
            if not (
                np.isclose(actual_mean, expected_mean, rtol=1.0e-9, atol=1.0e-12)
                and np.isclose(actual_std, expected_std, rtol=1.0e-9, atol=1.0e-12)
            ):
                raise RuntimeError(
                    f"Summary does not reproduce population statistics for {model}/{metric}"
                )
    order = {model: index for index, model in enumerate(ABLATION_MODELS)}
    result = frame.copy()
    result["_model_order"] = result["model"].map(order)
    return result.sort_values("_model_order").drop(columns="_model_order").reset_index(drop=True)


def _nested_close(actual: Any, expected: Any, path: str = "decision") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(f"Decision structure mismatch at {path}")
        for key, value in expected.items():
            _nested_close(actual[key], value, f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"Decision list mismatch at {path}")
        for index, value in enumerate(expected):
            _nested_close(actual[index], value, f"{path}[{index}]")
        return
    if isinstance(expected, (float, np.floating)):
        try:
            matches = np.isclose(float(actual), float(expected), rtol=1.0e-10, atol=1.0e-12)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise RuntimeError(f"Decision value mismatch at {path}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise RuntimeError(f"Decision value mismatch at {path}: {actual!r} != {expected!r}")


def _validate_config(path: Path, model: str, seed: int) -> dict[str, Any]:
    payload = _read_json(path, "run config")
    experiment_config = payload.get("experiment_config")
    run_spec = payload.get("run_spec")
    sampling = payload.get("sampling")
    parameterization = payload.get("two_channel_parameterization")
    if not all(
        isinstance(item, dict)
        for item in (experiment_config, run_spec, sampling, parameterization)
    ):
        raise RuntimeError(f"Malformed formal config: {path}")
    mismatches: list[str] = []
    for name, expected in EXPECTED_CONFIG.items():
        if not _same_value(experiment_config.get(name), expected):
            mismatches.append(f"experiment_config.{name}")
    expected_spec = {
        "experiment": EXPERIMENT,
        "case": CASE,
        "eta": 1.0,
        "model": model,
        "seed": seed,
        "steps": 5000,
        "lambda_struct": 1.0,
    }
    for name, expected in expected_spec.items():
        if not _same_value(run_spec.get(name), expected):
            mismatches.append(f"run_spec.{name}")
    if payload.get("initialization") != "Xavier normal for every linear weight; zero biases":
        mismatches.append("initialization")
    if payload.get("model_initialization_seed") != seed:
        mismatches.append("model_initialization_seed")
    if sampling.get("collocation_generator_seed") != 100_000 + seed:
        mismatches.append("sampling.collocation_generator_seed")
    if sampling.get("structure_generator_seed") != 200_000 + seed:
        mismatches.append("sampling.structure_generator_seed")
    if sampling.get("collocation_resampled_each_step") is not True:
        mismatches.append("sampling.collocation_resampled_each_step")
    if parameterization.get("uses_exact_energy_or_time_carrier_in_forward") is not False:
        mismatches.append("two_channel_parameterization.energy_leakage_flag")
    if not isinstance(payload.get("trainable_initial_state_sha256"), str):
        mismatches.append("trainable_initial_state_sha256")
    expected_rotation = {
        "ltpb_cpa": 0.0,
        "branch_complex": 0.0,
        "component_polar": None,
        "rotated_ltpb_15": 15.0,
        "rotated_ltpb_30": 30.0,
    }[model]
    if not _same_value(parameterization.get("rotation_degrees"), expected_rotation):
        mismatches.append("two_channel_parameterization.rotation_degrees")
    if mismatches:
        raise RuntimeError(f"Formal-run fairness mismatch in {path}: {mismatches}")
    return payload


def _load_dense(path: Path, context: str) -> dict[str, np.ndarray]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {context}: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set(DENSE_FIELDS) - set(archive.files))
            if missing:
                raise RuntimeError(f"{context} is missing arrays {missing}: {path}")
            arrays = {name: np.asarray(archive[name]).copy() for name in DENSE_FIELDS}
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read {context}: {path}: {error}") from error
    expected_shapes = {
        "x": (128,),
        "t": (201,),
        "current_predicted": (201, 128),
        "current_exact": (201, 128),
        "relative_phase_predicted": (201, 128),
        "relative_phase_exact": (201, 128),
        "internal_channel_amplitude_predicted": (201, 128, 2),
        "internal_channel_amplitude_exact": (201, 128, 2),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise RuntimeError(
                f"Unexpected dense-grid shape for {context}/{name}: "
                f"{arrays[name].shape}, expected {expected}"
            )
        if not np.all(np.isfinite(arrays[name])):
            raise RuntimeError(f"Non-finite dense-grid values in {context}/{name}")
    x_expected = 2.0 * math.pi * np.arange(128, dtype=np.float64) / 128.0
    if not np.allclose(arrays["x"], x_expected, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"Dense spatial grid is not endpoint-free [0,2pi): {path}")
    if not np.allclose(arrays["t"], np.linspace(0.0, 5.0, 201), atol=1.0e-12):
        raise RuntimeError(f"Dense temporal grid is not [0,5] with 201 points: {path}")
    return arrays


def load_inputs() -> AblationInputs:
    runs = _validate_runs(_read_csv(RUNS_PATH, (), "15-run table"))
    summary = _validate_summary(_read_csv(SUMMARY_PATH, (), "summary table"), runs)
    decision = _read_json(DECISION_PATH, "mechanism decision")
    expected_decision = compute_mechanism_decision(runs, summary)
    _nested_close(decision, expected_decision)

    time_series: dict[RunKey, pd.DataFrame] = {}
    loss_history: dict[RunKey, pd.DataFrame] = {}
    dense_grid: dict[RunKey, dict[str, np.ndarray]] = {}
    configs: dict[RunKey, dict[str, Any]] = {}
    for model in ABLATION_MODELS:
        for seed in ABLATION_SEEDS:
            key = (model, seed)
            run_dir = _run_dir(model, seed)
            configs[key] = _validate_config(run_dir / "config.json", model, seed)
            series = _read_csv(
                run_dir / "time_series.csv", TIME_SERIES_COLUMNS, "time series"
            )
            history = _read_csv(run_dir / "loss_history.csv", LOSS_COLUMNS, "loss history")
            dense = _load_dense(run_dir / "dense_grid.npz", "dense grid")

            _assert_finite(series, TIME_SERIES_COLUMNS, f"time series {key}")
            _assert_finite(history, LOSS_COLUMNS, f"loss history {key}")
            t = series["t"].to_numpy(float)
            if len(series) != 201 or not np.allclose(t, dense["t"], atol=1.0e-12):
                raise RuntimeError(f"Time-series/dense-grid mismatch for {key}")
            steps = history["step"].to_numpy(int)
            if len(history) != 501 or steps[0] != 1 or steps[-1] != 5000:
                raise RuntimeError(f"Loss history must contain 501 records through step 5000: {key}")
            if not np.all(np.diff(steps) > 0):
                raise RuntimeError(f"Loss steps are not strictly increasing: {key}")
            if not np.allclose(
                history["total"].to_numpy(float),
                history["pde"].to_numpy(float),
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise RuntimeError(f"Loss history is not PDE-only: {key}")

            index_x0 = int(np.argmin(np.abs(dense["x"])))
            if not np.allclose(
                series["current_x0_predicted"].to_numpy(float),
                dense["current_predicted"][:, index_x0],
                atol=1.0e-11,
            ):
                raise RuntimeError(f"Current x=0 time series does not match dense grid: {key}")
            phase_disagreement = np.angle(
                np.exp(
                    1j
                    * (
                        series["relative_phase_x0_predicted"].to_numpy(float)
                        - dense["relative_phase_predicted"][:, index_x0]
                    )
                )
            )
            if np.max(np.abs(phase_disagreement)) > 1.0e-11:
                raise RuntimeError(f"Relative-phase x=0 series does not match dense grid: {key}")
            for channel in (1, 2):
                dense_mean = np.mean(
                    dense["internal_channel_amplitude_predicted"][:, :, channel - 1], axis=1
                )
                column = f"internal_channel_{channel}_amplitude_spatial_mean"
                if not np.allclose(series[column].to_numpy(float), dense_mean, atol=1.0e-11):
                    raise RuntimeError(f"Internal amplitude series does not match dense grid: {key}")

            time_series[key] = series
            loss_history[key] = history
            dense_grid[key] = dense

    for seed in ABLATION_SEEDS:
        hashes = {
            configs[(model, seed)]["trainable_initial_state_sha256"]
            for model in ABLATION_MODELS
        }
        collocation_seeds = {
            configs[(model, seed)]["sampling"]["collocation_generator_seed"]
            for model in ABLATION_MODELS
        }
        if len(hashes) != 1 or collocation_seeds != {100_000 + seed}:
            raise RuntimeError(
                f"Same-seed models do not share initialization/collocation streams: seed={seed}"
            )

    return AblationInputs(
        runs=runs,
        summary=summary,
        decision=decision,
        time_series=time_series,
        loss_history=loss_history,
        dense_grid=dense_grid,
        configs=configs,
    )


def _figure_relative(path: Path) -> str:
    return path.resolve().relative_to(FIGURE_ROOT.resolve()).as_posix()


def _write_source(frame: pd.DataFrame, filename: str) -> Path:
    if frame.empty:
        raise RuntimeError(f"Refusing to write empty source data: {filename}")
    numeric = frame.select_dtypes(include=[np.number])
    if not numeric.empty and not np.all(np.isfinite(numeric.to_numpy(float))):
        raise RuntimeError(f"Refusing to write non-finite source data: {filename}")
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    path = SOURCE_ROOT / filename
    frame.to_csv(path, index=False, float_format="%.12g")
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Source-data export failed: {path}")
    return path


def _save_figure(fig: plt.Figure, basename: str) -> dict[str, str]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for extension in ("png", "pdf", "svg"):
        path = FIGURE_ROOT / f"{basename}.{extension}"
        kwargs: dict[str, Any] = {}
        if extension == "png":
            kwargs["dpi"] = RASTER_DPI
        fig.savefig(path, **kwargs)
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Figure export failed: {path}")
        files[extension] = _figure_relative(path)
    plt.close(fig)
    svg_path = FIGURE_ROOT / f"{basename}.svg"
    if "<text" not in svg_path.read_text(encoding="utf-8"):
        raise RuntimeError(f"SVG text is not editable: {svg_path}")
    if not (FIGURE_ROOT / f"{basename}.pdf").read_bytes().startswith(b"%PDF"):
        raise RuntimeError(f"Invalid PDF export: {basename}")
    if not (FIGURE_ROOT / f"{basename}.png").read_bytes().startswith(b"\x89PNG"):
        raise RuntimeError(f"Invalid PNG export: {basename}")
    return files


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=20,
    )


def _style_axis(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.45, alpha=0.8)
    ax.tick_params(width=0.7, length=2.5)


def _model_values(runs: pd.DataFrame, model: str, metric: str) -> np.ndarray:
    subset = runs[runs["model"] == model].sort_values("seed")
    values = subset[metric].to_numpy(float)
    if tuple(subset["seed"].astype(int)) != ABLATION_SEEDS:
        raise RuntimeError(f"Missing ordered seeds for {model}/{metric}")
    return values


def _metric_value(row: pd.Series, metric: str) -> float:
    if metric == "infidelity_mean":
        return max(0.0, 1.0 - float(row["fidelity_mean"]))
    return float(row[metric])


def _log_errorbar(
    ax: plt.Axes,
    x: float,
    mean: float,
    std: float,
    *,
    color: str,
) -> None:
    plotted_mean = max(mean, PLOT_FLOOR)
    lower_raw = mean - std
    lower = plotted_mean if lower_raw <= PLOT_FLOOR else lower_raw
    upper = max(mean + std, plotted_mean)
    ax.errorbar(
        x,
        plotted_mean,
        yerr=np.array([[plotted_mean - lower], [upper - plotted_mean]]),
        fmt="_",
        markersize=11,
        markeredgewidth=1.35,
        color=color,
        capsize=2.0,
        linewidth=0.9,
        zorder=6,
    )


def _aligned_stack(
    frames: Iterable[pd.DataFrame], x_column: str, y_column: str
) -> tuple[np.ndarray, np.ndarray]:
    items = list(frames)
    if not items:
        raise RuntimeError("Cannot align an empty collection of time series")
    x = items[0][x_column].to_numpy(float)
    values: list[np.ndarray] = []
    for frame in items:
        candidate_x = frame[x_column].to_numpy(float)
        if not np.allclose(candidate_x, x, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"Misaligned {x_column} grids")
        values.append(frame[y_column].to_numpy(float))
    return x, np.stack(values, axis=0)


def _circularly_aligned_unwrap(
    predicted_wrapped: np.ndarray, exact_wrapped: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exact_unwrapped = np.unwrap(exact_wrapped)
    circular_error = np.angle(np.exp(1j * (predicted_wrapped - exact_wrapped)))
    continuous_error = np.unwrap(circular_error)
    predicted_aligned = exact_unwrapped + continuous_error
    return predicted_aligned, exact_unwrapped, circular_error


def plot_core_error_comparison(inputs: AblationInputs) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for metric in CORE_METRICS:
        for model in ABLATION_MODELS:
            values = _model_values(inputs.runs, model, metric)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=0))
            ltpb_mean = float(np.mean(_model_values(inputs.runs, "ltpb_cpa", metric)))
            for seed, value in zip(ABLATION_SEEDS, values, strict=True):
                rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "seed": seed,
                        "metric": metric,
                        "metric_label": CORE_METRIC_LABELS[metric],
                        "value": value,
                        "group_mean": mean,
                        "population_sd": std,
                        "group_ratio_to_ltpb": mean / max(ltpb_mean, PLOT_FLOOR),
                    }
                )
    source = pd.DataFrame(rows)
    source_path = _write_source(source, "01_core_error_comparison.csv")

    fig, axes = plt.subplots(
        1, 3, figsize=(FIGURE_WIDTH_IN, 2.65), layout="constrained"
    )
    for panel, (ax, metric) in enumerate(zip(axes, CORE_METRICS, strict=True)):
        for model_index, model in enumerate(ABLATION_MODELS):
            values = _model_values(inputs.runs, model, metric)
            offsets = np.linspace(-0.07, 0.07, len(values))
            ax.scatter(
                model_index + offsets,
                np.maximum(values, PLOT_FLOOR),
                s=18,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.88,
                zorder=5,
            )
            _log_errorbar(
                ax,
                float(model_index),
                float(np.mean(values)),
                float(np.std(values, ddof=0)),
                color=MODEL_COLORS[model],
            )
        ax.set_yscale("log")
        ax.set_xticks(range(len(ABLATION_MODELS)))
        ax.set_xticklabels(
            [MODEL_TICK_LABELS[model] for model in ABLATION_MODELS],
            rotation=23,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_title(CORE_METRIC_LABELS[metric])
        ax.set_ylabel("Error")
        _style_axis(ax, grid_axis="y")
        _add_panel_label(ax, chr(ord("a") + panel))
    fig.suptitle(
        "Case M core errors: independent seeds and mean +/- population SD",
        fontsize=8.0,
    )
    files = _save_figure(fig, "01_core_error_comparison")
    return {
        "title": "三项核心误差比较",
        "question": "能量对齐的 LTPB 是否在 spinor、current 和相对相位三项核心误差上同时优于匹配容量的控制模型？",
        "conclusion": _decision_conclusion_zh(inputs.decision),
        "statistics": "三个独立 seeds；中心为算术均值，离散为 population SD（ddof=0）；不作推断性显著性检验。",
        "note": "对数轴上若 mean-SD 不为正，则不绘制无定义的下侧误差臂；源 CSV 仍保存未截断的 population SD。",
        "source_data": [_figure_relative(source_path)],
        "files": files,
    }


def _support_source(inputs: AblationInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, label in SUPPORT_METRICS:
        ltpb_rows = inputs.runs[inputs.runs["model"] == "ltpb_cpa"].set_index("seed")
        ltpb_values = np.array(
            [_metric_value(ltpb_rows.loc[seed], metric) for seed in ABLATION_SEEDS],
            dtype=float,
        )
        ltpb_mean = float(np.mean(ltpb_values))
        for model in ABLATION_MODELS:
            subset = inputs.runs[inputs.runs["model"] == model].sort_values("seed")
            values = np.array([_metric_value(row, metric) for _, row in subset.iterrows()])
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=0))
            for seed, value, reference in zip(
                ABLATION_SEEDS, values, ltpb_values, strict=True
            ):
                rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "seed": seed,
                        "metric": metric,
                        "metric_label": label,
                        "value": value,
                        "ltpb_same_seed_value": reference,
                        "paired_ratio_with_plot_floor": max(value, PLOT_FLOOR)
                        / max(reference, PLOT_FLOOR),
                        "group_mean": mean,
                        "population_sd": std,
                        "ltpb_group_mean": ltpb_mean,
                        "group_ratio_with_plot_floor": max(mean, PLOT_FLOOR)
                        / max(ltpb_mean, PLOT_FLOOR),
                        "plot_floor": PLOT_FLOOR,
                    }
                )
    return pd.DataFrame(rows)


def plot_per_seed_summary(inputs: AblationInputs) -> dict[str, Any]:
    source = _support_source(inputs)
    source_path = _write_source(source, "02_per_seed_summary.csv")
    metrics = [metric for metric, _ in SUPPORT_METRICS]
    metric_labels = [
        "Final spinor",
        "Density",
        "1-F",
        "Branch coeff.",
        "Occupation",
        "PDE loss",
    ]
    seed_rows = [(model, seed) for model in ABLATION_MODELS for seed in ABLATION_SEEDS]
    seed_matrix = np.empty((len(seed_rows), len(metrics)), dtype=float)
    for row_index, (model, seed) in enumerate(seed_rows):
        for column_index, metric in enumerate(metrics):
            row = source[
                (source["model"] == model)
                & (source["seed"] == seed)
                & (source["metric"] == metric)
            ].iloc[0]
            seed_matrix[row_index, column_index] = math.log10(
                float(row["paired_ratio_with_plot_floor"])
            )
    group_matrix = np.empty((len(ABLATION_MODELS), len(metrics)), dtype=float)
    annotations: list[list[str]] = []
    for row_index, model in enumerate(ABLATION_MODELS):
        annotation_row: list[str] = []
        for column_index, metric in enumerate(metrics):
            row = source[
                (source["model"] == model) & (source["metric"] == metric)
            ].iloc[0]
            group_matrix[row_index, column_index] = math.log10(
                float(row["group_ratio_with_plot_floor"])
            )
            annotation_row.append(
                f"{float(row['group_mean']):.1e}\n+/-{float(row['population_sd']):.1e}"
            )
        annotations.append(annotation_row)
    limit = max(
        1.0,
        float(np.max(np.abs(seed_matrix))),
        float(np.max(np.abs(group_matrix))),
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(FIGURE_WIDTH_IN, 5.65),
        gridspec_kw={"height_ratios": [2.35, 1.0]},
        layout="constrained",
    )
    image_seed = axes[0].imshow(
        seed_matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    axes[0].set_yticks(range(len(seed_rows)))
    axes[0].set_yticklabels(
        [f"{MODEL_TICK_LABELS[model].replace(chr(10), ' ')} | s{seed}" for model, seed in seed_rows]
    )
    axes[0].set_xticks(range(len(metrics)))
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_title("Paired seed errors normalized to LTPB at the same seed")
    _add_panel_label(axes[0], "a")

    axes[1].imshow(
        group_matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    axes[1].set_yticks(range(len(ABLATION_MODELS)))
    axes[1].set_yticklabels([MODEL_TICK_LABELS[model].replace("\n", " ") for model in ABLATION_MODELS])
    axes[1].set_xticks(range(len(metrics)))
    axes[1].set_xticklabels(metric_labels)
    axes[1].set_title("Arithmetic mean +/- population SD (cell text); color is mean/LTPB")
    for row_index in range(len(ABLATION_MODELS)):
        for column_index in range(len(metrics)):
            text_color = "white" if abs(group_matrix[row_index, column_index]) > 0.58 * limit else "#222222"
            axes[1].text(
                column_index,
                row_index,
                annotations[row_index][column_index],
                ha="center",
                va="center",
                fontsize=4.8,
                color=text_color,
            )
    _add_panel_label(axes[1], "b")
    colorbar = fig.colorbar(image_seed, ax=axes, shrink=0.82, pad=0.015)
    colorbar.set_label(r"$\log_{10}$(error / LTPB reference)")
    files = _save_figure(fig, "02_per_seed_summary")
    return {
        "title": "逐 seed 与辅助指标汇总",
        "question": "模型差异是否在每个 seed 以及场、能支和最终 PDE 辅助指标上都可见？",
        "conclusion": "图中保留全部 seed 值；汇总只报告算术均值与 population SD，不声称统计显著性。",
        "statistics": "同 seed 比值按预先固定的随机流配对；模型汇总 n=3，离散为 population SD。",
        "source_data": [_figure_relative(source_path)],
        "files": files,
    }


def _time_series_source(
    inputs: AblationInputs,
    predicted_column: str,
    exact_column: str,
    *,
    align_circular_phase: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in ABLATION_MODELS:
        frames = [inputs.time_series[(model, seed)] for seed in ABLATION_SEEDS]
        t, predicted_stack = _aligned_stack(frames, "t", predicted_column)
        _, exact_stack = _aligned_stack(frames, "t", exact_column)
        if not np.allclose(exact_stack, exact_stack[0], atol=1.0e-11):
            raise RuntimeError(f"Exact time series differs across seeds for {model}/{exact_column}")
        transformed_predicted = np.empty_like(predicted_stack)
        transformed_exact = np.empty_like(exact_stack)
        circular_errors = np.zeros_like(predicted_stack)
        seed_display_shifts = np.zeros(len(ABLATION_SEEDS), dtype=float)
        for seed_index in range(len(ABLATION_SEEDS)):
            if align_circular_phase:
                aligned, exact_unwrapped, circular_error = _circularly_aligned_unwrap(
                    predicted_stack[seed_index], exact_stack[seed_index]
                )
                transformed_predicted[seed_index] = aligned
                transformed_exact[seed_index] = exact_unwrapped
                circular_errors[seed_index] = circular_error
            else:
                transformed_predicted[seed_index] = predicted_stack[seed_index]
                transformed_exact[seed_index] = exact_stack[seed_index]
        if align_circular_phase:
            # Independent unwraps are defined only up to a constant integer
            # multiple of 2*pi.  Align that harmless display branch across
            # seeds before taking a linear mean/SD; the circular errors and
            # quantitative RMSE are unchanged.
            reference = transformed_predicted[0]
            for seed_index in range(1, len(ABLATION_SEEDS)):
                winding = int(
                    np.rint(
                        np.median(reference - transformed_predicted[seed_index])
                        / (2.0 * math.pi)
                    )
                )
                shift = 2.0 * math.pi * winding
                transformed_predicted[seed_index] += shift
                seed_display_shifts[seed_index] = shift
        mean = np.mean(transformed_predicted, axis=0)
        std = np.std(transformed_predicted, axis=0, ddof=0)
        for seed_index, seed in enumerate(ABLATION_SEEDS):
            for time_index, time_value in enumerate(t):
                rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "seed": seed,
                        "t": time_value,
                        "predicted_raw": predicted_stack[seed_index, time_index],
                        "exact_raw": exact_stack[seed_index, time_index],
                        "predicted_display": transformed_predicted[seed_index, time_index],
                        "exact_display": transformed_exact[seed_index, time_index],
                        "circular_error": circular_errors[seed_index, time_index],
                        "seed_display_branch_shift_rad": seed_display_shifts[
                            seed_index
                        ],
                        "group_mean_display": mean[time_index],
                        "population_sd_display": std[time_index],
                    }
                )
    return pd.DataFrame(rows)


def _plot_five_model_time_panels(
    source: pd.DataFrame,
    *,
    basename: str,
    ylabel: str,
    title: str,
) -> dict[str, str]:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIGURE_WIDTH_IN, 4.15),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    flat = axes.ravel()
    for panel, model in enumerate(ABLATION_MODELS):
        ax = flat[panel]
        subset = source[source["model"] == model]
        for seed in ABLATION_SEEDS:
            seed_frame = subset[subset["seed"] == seed].sort_values("t")
            ax.plot(
                seed_frame["t"],
                seed_frame["predicted_display"],
                color=SEED_COLORS[seed],
                alpha=0.38,
                linewidth=0.75,
            )
        summary = subset[subset["seed"] == ABLATION_SEEDS[0]].sort_values("t")
        t = summary["t"].to_numpy(float)
        mean = summary["group_mean_display"].to_numpy(float)
        std = summary["population_sd_display"].to_numpy(float)
        exact = summary["exact_display"].to_numpy(float)
        ax.fill_between(
            t,
            mean - std,
            mean + std,
            color=MODEL_COLORS[model],
            alpha=0.17,
            linewidth=0.0,
            zorder=2,
        )
        ax.plot(t, mean, color=MODEL_COLORS[model], linewidth=1.45, zorder=4)
        ax.plot(t, exact, color=EXACT_COLOR, linestyle="--", linewidth=1.0, zorder=5)
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("t")
        ax.set_ylabel(ylabel)
        _style_axis(ax)
        _add_panel_label(ax, chr(ord("a") + panel))
    legend_ax = flat[-1]
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], color=MODEL_COLORS["ltpb_cpa"], linewidth=1.45, label="model mean"),
        Line2D([0], [0], color=MODEL_COLORS["ltpb_cpa"], linewidth=5.0, alpha=0.17, label="population SD"),
        Line2D([0], [0], color=EXACT_COLOR, linestyle="--", label="analytic target"),
    ]
    handles.extend(
        Line2D([0], [0], color=SEED_COLORS[seed], linewidth=0.9, label=f"seed {seed}")
        for seed in ABLATION_SEEDS
    )
    legend_ax.legend(handles=handles, loc="center", ncol=1)
    legend_ax.text(
        0.5,
        0.18,
        "No inferential test (n=3)",
        ha="center",
        va="center",
        fontsize=6.0,
        color="#555555",
    )
    fig.suptitle(title, fontsize=8.0)
    return _save_figure(fig, basename)


def plot_current_x0(inputs: AblationInputs) -> dict[str, Any]:
    source = _time_series_source(
        inputs,
        "current_x0_predicted",
        "current_x0_exact",
        align_circular_phase=False,
    )
    source_path = _write_source(source, "03_current_x0.csv")
    files = _plot_five_model_time_panels(
        source,
        basename="03_current_x0",
        ylabel=r"$j(0,t)$",
        title="Case M probability current at x=0",
    )
    return {
        "title": "Case M 在 x=0 的概率流",
        "question": "各表示是否恢复 Case M 的快速概率流振荡？",
        "conclusion": "图中并列解析轨迹、三个 seed 和模型均值；不以标量训练 loss 代替 current 精度。",
        "statistics": "三个独立 seeds；粗线为算术均值，阴影为 population SD。",
        "source_data": [_figure_relative(source_path)],
        "files": files,
    }


def plot_relative_phase_x0(inputs: AblationInputs) -> dict[str, Any]:
    source = _time_series_source(
        inputs,
        "relative_phase_x0_predicted",
        "relative_phase_x0_exact",
        align_circular_phase=True,
    )
    source_path = _write_source(source, "04_relative_phase_x0.csv")
    files = _plot_five_model_time_panels(
        source,
        basename="04_relative_phase_x0",
        ylabel="Aligned unwrapped phase (rad)",
        title="Case M spinor-component relative phase at x=0",
    )
    return {
        "title": "Case M 在 x=0 的旋量分量相对相位",
        "question": "各表示是否恢复旋量两分量的相对相位时间轨迹？",
        "conclusion": "预测先按圆周差对齐解析分支再沿时间展开；原始角度、圆周误差和跨 seed 的 2π 显示平移均保存在源数据。",
        "statistics": "三个独立 seeds；粗线为算术均值，阴影为 population SD；不作假设检验。",
        "source_data": [_figure_relative(source_path)],
        "files": files,
    }


def _rotation_source(inputs: AblationInputs) -> pd.DataFrame:
    model_angles = (("ltpb_cpa", 0.0), ("rotated_ltpb_15", 15.0), ("rotated_ltpb_30", 30.0))
    component_angle = -0.5 * math.degrees(math.atan2(2.0, 5.0))
    rows: list[dict[str, Any]] = []
    for metric in CORE_METRICS:
        ltpb_values = _model_values(inputs.runs, "ltpb_cpa", metric)
        ltpb_mean = float(np.mean(ltpb_values))
        threshold = ROTATION_30_RATIO_MIN * ltpb_mean
        for model, angle in model_angles:
            values = _model_values(inputs.runs, model, metric)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=0))
            for seed, value, reference in zip(
                ABLATION_SEEDS, values, ltpb_values, strict=True
            ):
                rows.append(
                    {
                        "metric": metric,
                        "metric_label": CORE_METRIC_LABELS[metric],
                        "model": model,
                        "angle_degrees": angle,
                        "role": "preregistered_rotation_path",
                        "seed": seed,
                        "value": value,
                        "group_mean": mean,
                        "population_sd": std,
                        "ltpb_same_seed_value": reference,
                        "paired_ratio_to_ltpb": value / max(reference, PLOT_FLOOR),
                        "group_ratio_to_ltpb": mean / max(ltpb_mean, PLOT_FLOOR),
                        "three_x_ltpb_mean_threshold": threshold,
                    }
                )
        component_values = _model_values(inputs.runs, "component_polar", metric)
        component_mean = float(np.mean(component_values))
        component_std = float(np.std(component_values, ddof=0))
        for seed, value, reference in zip(
            ABLATION_SEEDS, component_values, ltpb_values, strict=True
        ):
            rows.append(
                {
                    "metric": metric,
                    "metric_label": CORE_METRIC_LABELS[metric],
                    "model": "component_polar",
                    "angle_degrees": component_angle,
                    "role": "component_basis_special_case",
                    "seed": seed,
                    "value": value,
                    "group_mean": component_mean,
                    "population_sd": component_std,
                    "ltpb_same_seed_value": reference,
                    "paired_ratio_to_ltpb": value / max(reference, PLOT_FLOOR),
                    "group_ratio_to_ltpb": component_mean / max(ltpb_mean, PLOT_FLOOR),
                    "three_x_ltpb_mean_threshold": threshold,
                }
            )
    return pd.DataFrame(rows)


def plot_rotation_degradation(inputs: AblationInputs) -> dict[str, Any]:
    source = _rotation_source(inputs)
    source_path = _write_source(source, "05_rotation_degradation.csv")
    fig, axes = plt.subplots(
        1, 3, figsize=(FIGURE_WIDTH_IN, 2.65), layout="constrained"
    )
    angles = np.array([0.0, 15.0, 30.0])
    component_angle = -0.5 * math.degrees(math.atan2(2.0, 5.0))
    for panel, (ax, metric) in enumerate(zip(axes, CORE_METRICS, strict=True)):
        subset = source[source["metric"] == metric]
        rotation_subset = subset[
            subset["role"] == "preregistered_rotation_path"
        ]
        for seed in ABLATION_SEEDS:
            seed_frame = rotation_subset[
                rotation_subset["seed"] == seed
            ].sort_values("angle_degrees")
            ax.plot(
                seed_frame["angle_degrees"],
                np.maximum(seed_frame["value"], PLOT_FLOOR),
                color=SEED_COLORS[seed],
                marker="o",
                markersize=3.0,
                linewidth=0.8,
                alpha=0.7,
                label=f"seed {seed}" if panel == 0 else None,
            )
        summary = rotation_subset[
            rotation_subset["seed"] == ABLATION_SEEDS[0]
        ].sort_values("angle_degrees")
        means = summary["group_mean"].to_numpy(float)
        stds = summary["population_sd"].to_numpy(float)
        ax.plot(
            angles,
            np.maximum(means, PLOT_FLOOR),
            color="#111111",
            linewidth=1.4,
            marker="s",
            markersize=3.5,
            label="mean" if panel == 0 else None,
            zorder=6,
        )
        for angle, mean, std in zip(angles, means, stds, strict=True):
            _log_errorbar(ax, float(angle), float(mean), float(std), color="#111111")
        threshold = float(summary["three_x_ltpb_mean_threshold"].iloc[0])
        ax.axhline(
            max(threshold, PLOT_FLOOR),
            color="#B94A48",
            linestyle="--",
            linewidth=1.0,
            label="3x LTPB mean" if panel == 0 else None,
        )
        for angle, mean in zip(angles[1:], means[1:], strict=True):
            ratio = mean / max(means[0], PLOT_FLOOR)
            ax.annotate(
                f"{ratio:.2g}x",
                (angle, max(mean, PLOT_FLOOR)),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=5.5,
                color="#333333",
            )
        component = subset[
            subset["role"] == "component_basis_special_case"
        ].sort_values("seed")
        component_values = component["value"].to_numpy(float)
        component_offsets = np.linspace(-0.45, 0.45, len(component_values))
        ax.scatter(
            component_angle + component_offsets,
            np.maximum(component_values, PLOT_FLOOR),
            s=18,
            marker="^",
            facecolor="white",
            edgecolor=MODEL_COLORS["component_polar"],
            linewidth=0.8,
            zorder=7,
            label="Component-Polar" if panel == 0 else None,
        )
        _log_errorbar(
            ax,
            component_angle,
            float(np.mean(component_values)),
            float(np.std(component_values, ddof=0)),
            color=MODEL_COLORS["component_polar"],
        )
        ax.set_yscale("log")
        ax.set_xticks(np.concatenate(([component_angle], angles)))
        ax.set_xticklabels(("C\n-10.9", "0", "15", "30"))
        ax.set_xlabel("Basis rotation angle (deg)")
        ax.set_ylabel("Error")
        ax.set_title(CORE_METRIC_LABELS[metric])
        _style_axis(ax, grid_axis="y")
        _add_panel_label(ax, chr(ord("a") + panel))
    axes[0].legend(loc="best")
    fig.suptitle("Paired-seed degradation away from the energy eigenbasis", fontsize=8.0)
    files = _save_figure(fig, "05_rotation_degradation")
    passing = int(inputs.decision["rotation_trend_passing_metric_count"])
    passing_30 = int(inputs.decision["rotation_30_passing_metric_count"])
    return {
        "title": "误差随基底旋转角的退化",
        "question": "匹配的双极坐标通道偏离能量本征基后，三项核心误差是否按预判据随角度退化？",
        "conclusion": f"通过严格逐角度趋势的指标为 {passing}/3；30° 达到至少 3 倍的指标为 {passing_30}/3。Component-Polar 仅作为精确有效角 -10.9007° 的特例显示，不进入 0/15/30 的预判路径。",
        "statistics": "同 seed 轨迹作描述性配对；中心为算术均值，离散为 population SD；不作推断性检验。",
        "note": "对数轴上若 mean-SD 不为正，则不绘制无定义的下侧误差臂；源 CSV 保留完整 population SD。",
        "source_data": [_figure_relative(source_path)],
        "files": files,
    }


def _complexity_source(inputs: AblationInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for predicted_metric, exact_metric, label in COMPLEXITY_METRICS:
        for model in ABLATION_MODELS:
            subset = inputs.runs[inputs.runs["model"] == model].sort_values("seed")
            predicted = subset[predicted_metric].to_numpy(float)
            exact = subset[exact_metric].to_numpy(float)
            predicted_mean = float(np.mean(predicted))
            predicted_std = float(np.std(predicted, ddof=0))
            exact_mean = float(np.mean(exact))
            exact_std = float(np.std(exact, ddof=0))
            for seed, predicted_value, exact_value in zip(
                ABLATION_SEEDS, predicted, exact, strict=True
            ):
                rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "seed": seed,
                        "metric": predicted_metric,
                        "exact_metric": exact_metric,
                        "metric_label": label,
                        "predicted": predicted_value,
                        "exact_representation": exact_value,
                        "predicted_group_mean": predicted_mean,
                        "predicted_population_sd": predicted_std,
                        "exact_group_mean": exact_mean,
                        "exact_population_sd": exact_std,
                    }
                )
    return pd.DataFrame(rows)


def plot_amplitude_phase_complexity(inputs: AblationInputs) -> dict[str, Any]:
    source = _complexity_source(inputs)
    source_path = _write_source(source, "06_amplitude_phase_complexity.csv")
    fig, axes = plt.subplots(
        1, 3, figsize=(FIGURE_WIDTH_IN, 2.75), layout="constrained"
    )
    for panel, (ax, (predicted_metric, _, label)) in enumerate(
        zip(axes, COMPLEXITY_METRICS, strict=True)
    ):
        metric_source = source[source["metric"] == predicted_metric]
        for model_index, model in enumerate(ABLATION_MODELS):
            subset = metric_source[metric_source["model"] == model].sort_values("seed")
            predicted = subset["predicted"].to_numpy(float)
            offsets = np.linspace(-0.07, 0.07, len(predicted))
            ax.scatter(
                model_index + offsets - 0.06,
                predicted,
                s=15,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                alpha=0.8,
                edgecolor="white",
                linewidth=0.3,
                zorder=5,
            )
            mean = float(subset["predicted_group_mean"].iloc[0])
            std = float(subset["predicted_population_sd"].iloc[0])
            ax.errorbar(
                model_index - 0.06,
                mean,
                yerr=std,
                fmt="_",
                markersize=9,
                markeredgewidth=1.2,
                capsize=2.0,
                linewidth=0.8,
                color=MODEL_COLORS[model],
                zorder=6,
            )
            exact = float(subset["exact_group_mean"].iloc[0])
            ax.scatter(
                model_index + 0.13,
                exact,
                s=20,
                marker="D",
                facecolor="white",
                edgecolor=EXACT_COLOR,
                linewidth=0.8,
                zorder=7,
            )
        ax.set_xticks(range(len(ABLATION_MODELS)))
        ax.set_xticklabels(
            [MODEL_TICK_LABELS[model] for model in ABLATION_MODELS],
            rotation=23,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
        _style_axis(ax, grid_axis="y")
        _add_panel_label(ax, chr(ord("a") + panel))
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="#666666", label="predicted seeds"),
        Line2D([0], [0], marker="_", linestyle="none", color="#333333", markersize=9, label="predicted mean +/- SD"),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=EXACT_COLOR,
            label="analytic representation",
        ),
    ]
    axes[0].legend(handles=handles, loc="best")
    fig.suptitle(
        "Learned channel burden versus the analytic solution in the same basis",
        fontsize=8.0,
    )
    files = _save_figure(fig, "06_amplitude_phase_complexity")
    return {
        "title": "振幅漂移与内部相位复杂度",
        "question": "模型退化是否伴随失配基底对解析解本身引入的振幅和相位复杂度？",
        "conclusion": "预测复杂度与同一模型基底中的解析固有复杂度并列显示；旋转基底所必需的解析调制不被误判为训练误差。",
        "statistics": "预测值来自三个独立 seeds，报告算术均值与 population SD；解析表示值是训练后确定性参照。",
        "source_data": [_figure_relative(source_path)],
        "files": files,
    }


def _loss_source(inputs: AblationInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in ABLATION_MODELS:
        frames = [inputs.loss_history[(model, seed)] for seed in ABLATION_SEEDS]
        steps, losses = _aligned_stack(frames, "step", "pde")
        mean = np.mean(losses, axis=0)
        std = np.std(losses, axis=0, ddof=0)
        for seed_index, seed in enumerate(ABLATION_SEEDS):
            for index, step in enumerate(steps):
                rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "seed": seed,
                        "step": int(step),
                        "pde_loss": losses[seed_index, index],
                        "group_mean": mean[index],
                        "population_sd": std[index],
                    }
                )
    return pd.DataFrame(rows)


def plot_loss_histories(inputs: AblationInputs) -> dict[str, Any]:
    source = _loss_source(inputs)
    source_path = _write_source(source, "07_loss_histories.csv")
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIGURE_WIDTH_IN, 4.15),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    flat = axes.ravel()
    for panel, model in enumerate(ABLATION_MODELS):
        ax = flat[panel]
        subset = source[source["model"] == model]
        for seed in ABLATION_SEEDS:
            seed_frame = subset[subset["seed"] == seed].sort_values("step")
            ax.plot(
                seed_frame["step"],
                np.maximum(seed_frame["pde_loss"], PLOT_FLOOR),
                color=SEED_COLORS[seed],
                linewidth=0.7,
                alpha=0.45,
            )
        summary = subset[subset["seed"] == ABLATION_SEEDS[0]].sort_values("step")
        step = summary["step"].to_numpy(float)
        raw_mean = summary["group_mean"].to_numpy(float)
        mean = np.maximum(raw_mean, PLOT_FLOOR)
        std = summary["population_sd"].to_numpy(float)
        lower_raw = raw_mean - std
        # A symmetric mean +/- SD interval can cross zero even though every
        # loss is positive.  Sending such values to the log-axis floor creates
        # a visually dominant vertical stripe.  Leave the lower arm blank at
        # those steps instead; the three seed traces retain the full evidence.
        lower = np.where(
            lower_raw > 0.0,
            np.maximum(lower_raw, PLOT_FLOOR),
            np.nan,
        )
        upper = np.maximum(raw_mean + std, PLOT_FLOOR)
        ax.fill_between(step, lower, upper, color=MODEL_COLORS[model], alpha=0.17, linewidth=0.0)
        ax.plot(step, mean, color=MODEL_COLORS[model], linewidth=1.4)
        ax.set_yscale("log")
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Adam step")
        ax.set_ylabel("PDE loss")
        _style_axis(ax, grid_axis="y")
        _add_panel_label(ax, chr(ord("a") + panel))
    legend_ax = flat[-1]
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], color=SEED_COLORS[seed], linewidth=0.9, label=f"seed {seed}")
        for seed in ABLATION_SEEDS
    ]
    handles.extend(
        [
            Line2D([0], [0], color=MODEL_COLORS["ltpb_cpa"], linewidth=1.5, label="model mean"),
            Line2D([0], [0], color=MODEL_COLORS["ltpb_cpa"], linewidth=5.0, alpha=0.17, label="population SD"),
        ]
    )
    legend_ax.legend(handles=handles, loc="center")
    legend_ax.text(
        0.5,
        0.17,
        "Optimization context only;\nnot a success criterion",
        ha="center",
        va="center",
        fontsize=6.0,
        color="#555555",
    )
    fig.suptitle("PDE-only optimization histories", fontsize=8.0)
    files = _save_figure(fig, "07_loss_histories")
    return {
        "title": "PDE-only 训练轨迹",
        "question": "不同表示的最终精度排序与其 PDE-only 优化轨迹有何对应？",
        "conclusion": "loss history 只提供优化背景，不替代 spinor、current 或相对相位精度判定。",
        "statistics": "三个独立 seeds；粗线为算术均值，阴影为 population SD；当 mean-SD 不为正时不在对数轴上绘制该阴影下臂；不作推断性检验。",
        "source_data": [_figure_relative(source_path)],
        "files": files,
    }


def _decision_conclusion(decision: dict[str, Any]) -> str:
    if bool(decision.get("strong_mechanism_support")):
        return (
            "All pre-registered dominance, rotation-trend, and 30-degree degradation "
            "rules pass, supporting the energy-branch/independent-phase alignment mechanism "
            "within this fixed single-k experiment."
        )
    return (
        "The complete pre-registered strong-mechanism rule does not pass; the figure "
        "reports every seed and component rule without post-hoc threshold changes."
    )


def _decision_conclusion_zh(decision: dict[str, Any]) -> str:
    if bool(decision.get("strong_mechanism_support")):
        return (
            "预先固定的 LTPB 优势、逐角度退化和 30° 三倍退化三组规则全部通过；"
            "在当前单 k 固定预算内支持能量支—独立相位对齐机制。"
        )
    return (
        "完整的预先固定强机制规则未通过；图中保留每个 seed 与各子判据，"
        "未在观察结果后改变阈值。"
    )


def _write_manifest(figures: dict[str, dict[str, Any]], inputs: AblationInputs) -> Path:
    payload = {
        "experiment": EXPERIMENT,
        "problem": (
            "在四输出容量、初始化、配点随机流和 PDE-only 预算完全匹配时，区分"
            "能量本征旋量基、一般双通道极坐标以及能量支—独立相位对齐各自的贡献。"
        ),
        "conclusion": _decision_conclusion_zh(inputs.decision),
        "statistics": {
            "independent_seeds": list(ABLATION_SEEDS),
            "n_per_model": 3,
            "center": "arithmetic mean",
            "spread": "population standard deviation (ddof=0)",
            "paired_display": "same-seed model/angle trajectories use matched RNG streams",
            "inferential_tests": "none",
        },
        "data_files": {
            "runs": str(RUNS_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "summary": str(SUMMARY_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "decision": str(DECISION_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "raw_runs": str(RAW_ROOT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "format_contract": {
            "backend": "Python/matplotlib only",
            "figure_width_mm": FIGURE_WIDTH_MM,
            "png_dpi": RASTER_DPI,
            "exports": ["PNG", "PDF", "SVG"],
            "editable_text": "svg.fonttype=none; pdf.fonttype=42",
            "background": "white",
        },
        "figures": figures,
    }
    path = FIGURE_ROOT / "figure_manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate all 15 formal runs, aggregate tables, time series, loss histories, and dense grids without exporting figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = load_inputs()
    if args.check_only:
        print(
            "Formal mechanism-ablation plotting inputs are complete and consistent: "
            "15 runs, five models, three seeds, matched RNG streams."
        )
        return

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    figures: dict[str, dict[str, Any]] = {}
    figures["01_core_error_comparison"] = plot_core_error_comparison(inputs)
    figures["02_per_seed_summary"] = plot_per_seed_summary(inputs)
    figures["03_current_x0"] = plot_current_x0(inputs)
    figures["04_relative_phase_x0"] = plot_relative_phase_x0(inputs)
    figures["05_rotation_degradation"] = plot_rotation_degradation(inputs)
    figures["06_amplitude_phase_complexity"] = plot_amplitude_phase_complexity(inputs)
    figures["07_loss_histories"] = plot_loss_histories(inputs)
    manifest_path = _write_manifest(figures, inputs)
    print(
        f"Generated {len(figures)} figure groups in {FIGURE_ROOT}\n"
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
