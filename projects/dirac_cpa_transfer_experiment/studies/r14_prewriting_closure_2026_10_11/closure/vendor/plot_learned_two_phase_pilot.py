from __future__ import annotations

"""Publication figures for the formal Learned Two-Phase Branch CPA pilot.

The script is deliberately independent of ``plot_results.py`` and writes only to
``results/figures/learned_two_phase_pilot``.  It validates the six formal LTPB
runs and the reused SP-CPA / Two-Branch baselines before exporting any figure.
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
COMPARISON_PATH = RESULTS_ROOT / "learned_two_phase_pilot_comparison.csv"
PILOT_RAW_ROOT = RESULTS_ROOT / "raw" / "learned_two_phase_pilot"
MAIN_RAW_ROOT = RESULTS_ROOT / "raw" / "main"
FIGURE_ROOT = RESULTS_ROOT / "figures" / "learned_two_phase_pilot"
SOURCE_ROOT = FIGURE_ROOT / "source_data"

FIGURE_WIDTH_MM = 183.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4
RASTER_DPI = 600
PLOT_FLOOR = 1.0e-12
CASES = ("P", "M")
LEARNED_SEEDS = (0, 1, 2)

METHODS = ("sp_cpa", "ltpb_cpa", "two_branch")
METHOD_LABELS = {
    "sp_cpa": "SP-CPA",
    "ltpb_cpa": "LTPB-CPA",
    "two_branch": "Two-Branch oracle",
}
METHOD_COLORS = {
    "sp_cpa": "#5D78A6",
    "ltpb_cpa": "#C86448",
    "two_branch": "#2F8A7B",
}
METHOD_MARKERS = {
    "sp_cpa": "o",
    "ltpb_cpa": "s",
    "two_branch": "D",
}
BRANCH_COLORS = {"plus": "#3E70B2", "minus": "#C05B42"}
EXACT_COLOR = "#202020"
GRID_COLOR = "#D9D9D9"

METRIC_SPECS = (
    ("spinor_rel_l2", "Spinor rel. L2"),
    ("rho_rel_l2", r"Density $\rho$ rel. L2"),
    ("current_rel_l2", r"Current $j$ rel. L2"),
    ("infidelity_mean", "Mean infidelity\n$1-F$"),
    ("relative_phase_rmse_rad", "Relative-phase\nRMSE (rad)"),
)

EXPECTED_CONFIG: dict[str, Any] = {
    "x_min": -math.pi,
    "x_max": math.pi,
    "mass": 5.0,
    "wave_number": 2.0,
    "t_max": 5.0,
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

BASE_TIME_COLUMNS = (
    "t",
    "current_x0_predicted",
    "current_x0_exact",
    "relative_phase_x0_predicted",
    "relative_phase_x0_exact",
    "occupation_plus_predicted",
    "occupation_plus_exact",
    "occupation_minus_predicted",
    "occupation_minus_exact",
)
LTPB_TIME_COLUMNS = BASE_TIME_COLUMNS + (
    "phase_plus_spatial_mean",
    "phase_plus_exact",
    "phase_minus_spatial_mean",
    "phase_minus_exact",
    "amplitude_plus_spatial_mean",
    "amplitude_plus_spatial_std",
    "amplitude_plus_exact",
    "amplitude_minus_spatial_mean",
    "amplitude_minus_spatial_std",
    "amplitude_minus_exact",
)
LOSS_COLUMNS = ("step", "total", "pde")


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


RunKey = tuple[str, str, int]


@dataclass(frozen=True)
class PilotInputs:
    comparison: pd.DataFrame
    time_series: dict[RunKey, pd.DataFrame]
    loss_history: dict[RunKey, pd.DataFrame]
    metrics: dict[RunKey, dict[str, Any]]


def _run_dir(case: str, model: str, seed: int) -> Path:
    if model == "ltpb_cpa":
        return PILOT_RAW_ROOT / f"case_{case}" / model / f"seed_{seed}"
    return MAIN_RAW_ROOT / f"case_{case}" / model / f"seed_{seed}"


def _required_run_keys() -> list[RunKey]:
    keys: list[RunKey] = []
    for case in CASES:
        keys.extend((case, "sp_cpa", seed) for seed in LEARNED_SEEDS)
        keys.extend((case, "ltpb_cpa", seed) for seed in LEARNED_SEEDS)
        keys.append((case, "two_branch", 0))
    return keys


def _figure_relative(path: Path) -> str:
    """Return a manifest path relative to the figure-manifest directory."""
    return path.resolve().relative_to(FIGURE_ROOT.resolve()).as_posix()


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing {context}: {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in {context}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected an object in {context}: {path}")
    return payload


def _read_csv(path: Path, columns: Iterable[str], context: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {context}: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Empty {context}: {path}")
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
            return math.isclose(float(actual), expected, rel_tol=1.0e-12, abs_tol=1.0e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _validate_config(path: Path, key: RunKey) -> None:
    case, model, seed = key
    payload = _read_json(path, "run config")
    experiment_config = payload.get("experiment_config")
    run_spec = payload.get("run_spec")
    sampling = payload.get("sampling")
    if not isinstance(experiment_config, dict) or not isinstance(run_spec, dict):
        raise RuntimeError(f"Malformed run config (missing experiment_config/run_spec): {path}")
    mismatches = []
    for name, expected in EXPECTED_CONFIG.items():
        actual = experiment_config.get(name)
        if not _same_value(actual, expected):
            mismatches.append(f"{name}={actual!r} (expected {expected!r})")
    expected_experiment = "learned_two_phase_pilot" if model == "ltpb_cpa" else "main"
    expected_spec = {
        "experiment": expected_experiment,
        "case": case,
        "model": model,
        "seed": seed,
        "lambda_struct": 1.0,
        "steps": 5000,
    }
    for name, expected in expected_spec.items():
        actual = run_spec.get(name)
        if not _same_value(actual, expected):
            mismatches.append(f"run_spec.{name}={actual!r} (expected {expected!r})")
    if payload.get("initialization") != "Xavier normal for every linear weight; zero biases":
        mismatches.append("initialization does not match the declared Xavier/zero-bias protocol")
    if not isinstance(sampling, dict):
        mismatches.append("sampling block is missing")
    else:
        if sampling.get("collocation_resampled_each_step") is not True:
            mismatches.append("sampling.collocation_resampled_each_step is not true")
        expected_collocation_seed = 100000 + seed
        if sampling.get("collocation_generator_seed") != expected_collocation_seed:
            mismatches.append(
                "sampling.collocation_generator_seed="
                f"{sampling.get('collocation_generator_seed')!r} "
                f"(expected {expected_collocation_seed})"
            )
    if mismatches:
        joined = "; ".join(mismatches)
        raise RuntimeError(f"Formal-run fairness check failed for {path}: {joined}")


def _validate_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "experiment",
        "case",
        "model",
        "seed",
        "steps",
        "spinor_rel_l2",
        "rho_rel_l2",
        "current_rel_l2",
        "fidelity_mean",
        "relative_phase_rmse_rad",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Comparison table is missing columns {missing}: {COMPARISON_PATH}")

    relevant = frame[frame["model"].isin(METHODS)].copy()
    expected_by_model = {
        "sp_cpa": {(case, seed) for case in CASES for seed in LEARNED_SEEDS},
        "ltpb_cpa": {(case, seed) for case in CASES for seed in LEARNED_SEEDS},
        "two_branch": {(case, 0) for case in CASES},
    }
    errors: list[str] = []
    for model, expected in expected_by_model.items():
        subset = relevant[relevant["model"] == model]
        actual_pairs = list(zip(subset["case"], subset["seed"].astype(int), strict=False))
        actual = set(actual_pairs)
        if actual != expected or len(actual_pairs) != len(expected):
            errors.append(
                f"{model}: expected {sorted(expected)}, found {sorted(actual_pairs)}"
            )
        expected_experiment = "learned_two_phase_pilot" if model == "ltpb_cpa" else "main"
        experiments = set(subset["experiment"].astype(str))
        if experiments != {expected_experiment}:
            errors.append(
                f"{model}: experiment values {sorted(experiments)}; expected {expected_experiment}"
            )
        if not subset.empty and not np.all(subset["steps"].to_numpy(float) == 5000.0):
            errors.append(f"{model}: not every comparison row has steps=5000")
    if errors:
        raise RuntimeError("Incomplete or inconsistent formal comparison table:\n- " + "\n- ".join(errors))

    numeric_metrics = [spec[0] for spec in METRIC_SPECS if spec[0] != "infidelity_mean"]
    numeric_metrics.append("fidelity_mean")
    values = relevant[numeric_metrics].to_numpy(float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Formal comparison contains NaN or infinite primary metrics")
    non_fidelity = relevant[[name for name in numeric_metrics if name != "fidelity_mean"]]
    if (non_fidelity.to_numpy(float) < 0.0).any():
        raise RuntimeError("Formal comparison contains a negative error metric")
    fidelity = relevant["fidelity_mean"].to_numpy(float)
    if ((fidelity < -1.0e-10) | (fidelity > 1.0 + 1.0e-8)).any():
        raise RuntimeError("Formal comparison contains fidelity outside the numerical [0, 1] range")
    return relevant.sort_values(["case", "model", "seed"]).reset_index(drop=True)


def load_inputs() -> PilotInputs:
    if not COMPARISON_PATH.exists():
        raise FileNotFoundError(
            "Formal pilot comparison is missing: "
            f"{COMPARISON_PATH}. Finish all six runs, then run "
            "`python -m src.run_learned_two_phase_pilot --mode aggregate`."
        )
    comparison = _validate_comparison(pd.read_csv(COMPARISON_PATH))

    missing_artifacts: list[Path] = []
    for key in _required_run_keys():
        case, model, seed = key
        run_dir = _run_dir(case, model, seed)
        required_names = ["config.json", "metrics.json", "time_series.csv", "loss_history.csv"]
        if model == "ltpb_cpa":
            required_names.extend(["dense_grid.npz", "model_state.pt"])
        missing_artifacts.extend(
            path
            for name in required_names
            if not (path := run_dir / name).exists() or path.stat().st_size == 0
        )
    if missing_artifacts:
        preview = "\n".join(f"- {path}" for path in missing_artifacts[:20])
        suffix = "" if len(missing_artifacts) <= 20 else f"\n- ... and {len(missing_artifacts) - 20} more"
        raise FileNotFoundError(
            f"Formal plotting inputs are incomplete ({len(missing_artifacts)} missing artifacts):\n"
            f"{preview}{suffix}"
        )

    time_series: dict[RunKey, pd.DataFrame] = {}
    loss_history: dict[RunKey, pd.DataFrame] = {}
    metrics: dict[RunKey, dict[str, Any]] = {}
    for key in _required_run_keys():
        case, model, seed = key
        run_dir = _run_dir(case, model, seed)
        _validate_config(run_dir / "config.json", key)
        needed_time_columns = LTPB_TIME_COLUMNS if model == "ltpb_cpa" else BASE_TIME_COLUMNS
        time_frame = _read_csv(run_dir / "time_series.csv", needed_time_columns, "time series")
        loss_frame = _read_csv(run_dir / "loss_history.csv", LOSS_COLUMNS, "loss history")
        metrics_payload = _read_json(run_dir / "metrics.json", "metrics")

        t = time_frame["t"].to_numpy(float)
        if len(t) != EXPECTED_CONFIG["test_t_points"] or not np.all(np.diff(t) > 0.0):
            raise RuntimeError(f"Unexpected or non-monotone test-time grid: {run_dir / 'time_series.csv'}")
        if not (math.isclose(t[0], 0.0, abs_tol=1.0e-12) and math.isclose(t[-1], 5.0, abs_tol=1.0e-12)):
            raise RuntimeError(f"Time grid does not span [0, 5]: {run_dir / 'time_series.csv'}")
        if int(loss_frame["step"].iloc[-1]) != 5000:
            raise RuntimeError(f"Loss history does not reach step 5000: {run_dir / 'loss_history.csv'}")
        numeric_time = time_frame[list(needed_time_columns)].to_numpy(float)
        numeric_loss = loss_frame[list(LOSS_COLUMNS)].to_numpy(float)
        if not np.all(np.isfinite(numeric_time)) or not np.all(np.isfinite(numeric_loss)):
            raise RuntimeError(f"NaN or infinite values in formal run: {run_dir}")
        expected_experiment = "learned_two_phase_pilot" if model == "ltpb_cpa" else "main"
        metadata_expected = {
            "experiment": expected_experiment,
            "case": case,
            "model": model,
            "seed": seed,
            "steps": 5000,
        }
        bad_metadata = {
            name: (metrics_payload.get(name), expected)
            for name, expected in metadata_expected.items()
            if not _same_value(metrics_payload.get(name), expected)
        }
        if bad_metadata:
            raise RuntimeError(f"Metrics metadata mismatch in {run_dir / 'metrics.json'}: {bad_metadata}")
        time_series[key] = time_frame
        loss_history[key] = loss_frame
        metrics[key] = metrics_payload

    return PilotInputs(
        comparison=comparison,
        time_series=time_series,
        loss_history=loss_history,
        metrics=metrics,
    )


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def _style_axis(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.45, alpha=0.8)
    ax.tick_params(width=0.7, length=2.5)


def _write_source(frame: pd.DataFrame, filename: str) -> Path:
    if frame.empty:
        raise RuntimeError(f"Refusing to export empty source data: {filename}")
    numeric = frame.select_dtypes(include=[np.number])
    if not numeric.empty and not np.all(np.isfinite(numeric.to_numpy(float))):
        raise RuntimeError(f"Refusing to export non-finite source data: {filename}")
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    path = SOURCE_ROOT / filename
    frame.to_csv(path, index=False, float_format="%.12g")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Source-data export failed: {path}")
    return path


def _save_figure(fig: plt.Figure, basename: str) -> dict[str, str]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for extension in ("png", "pdf", "svg"):
        path = FIGURE_ROOT / f"{basename}.{extension}"
        kwargs: dict[str, Any] = {}
        if extension == "png":
            kwargs["dpi"] = RASTER_DPI
        fig.savefig(path, **kwargs)
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Figure export failed: {path}")
        paths[extension] = _figure_relative(path)
    plt.close(fig)

    svg_path = FIGURE_ROOT / f"{basename}.svg"
    if "<text" not in svg_path.read_text(encoding="utf-8"):
        raise RuntimeError(f"SVG text was outlined instead of remaining editable: {svg_path}")
    if not (FIGURE_ROOT / f"{basename}.pdf").read_bytes().startswith(b"%PDF"):
        raise RuntimeError(f"Invalid PDF export: {basename}.pdf")
    if not (FIGURE_ROOT / f"{basename}.png").read_bytes().startswith(b"\x89PNG"):
        raise RuntimeError(f"Invalid PNG export: {basename}.png")
    return paths


def _aligned_stack(
    frames: Iterable[pd.DataFrame], x_column: str, y_column: str
) -> tuple[np.ndarray, np.ndarray]:
    frame_list = list(frames)
    if not frame_list:
        raise RuntimeError(f"No series available for {y_column}")
    x_reference = frame_list[0][x_column].to_numpy(float)
    rows = []
    for frame in frame_list:
        x = frame[x_column].to_numpy(float)
        if x.shape != x_reference.shape or not np.allclose(x, x_reference, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"Cannot aggregate misaligned grids for {y_column}")
        rows.append(frame[y_column].to_numpy(float))
    return x_reference, np.stack(rows, axis=0)


def _plot_mean_band(
    ax: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    label: str | None,
    log_safe: bool = False,
    linewidth: float = 1.45,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=0)
    if log_safe:
        plotted_mean = np.maximum(mean, PLOT_FLOOR)
        raw_lower = mean - std
        lower = np.maximum(raw_lower, PLOT_FLOOR)
        upper = np.maximum(mean + std, PLOT_FLOOR)
        band_where = raw_lower > PLOT_FLOOR
    else:
        plotted_mean = mean
        lower = mean - std
        upper = mean + std
        band_where = None
    ax.fill_between(
        x,
        lower,
        upper,
        where=band_where,
        color=color,
        alpha=0.18,
        linewidth=0.0,
        zorder=1,
    )
    ax.plot(x, plotted_mean, color=color, linewidth=linewidth, label=label, zorder=3)
    return mean, std


def plot_main_metric_comparison(inputs: PilotInputs) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for _, row in inputs.comparison.iterrows():
        model = str(row["model"])
        for metric, metric_label in METRIC_SPECS:
            if metric == "infidelity_mean":
                value = max(0.0, 1.0 - float(row["fidelity_mean"]))
            else:
                value = float(row[metric])
            rows.append(
                {
                    "case": str(row["case"]),
                    "model": model,
                    "method_label": METHOD_LABELS[model],
                    "seed": int(row["seed"]),
                    "metric": metric,
                    "metric_label": metric_label,
                    "value": value,
                    "plot_value": max(value, PLOT_FLOOR),
                }
            )
    source = pd.DataFrame(rows)
    grouped = source.groupby(["case", "model", "metric"], sort=False)["value"]
    source["n_seeds"] = grouped.transform("size").astype(int)
    source["group_mean"] = grouped.transform("mean")
    source["population_sd"] = grouped.transform(lambda series: series.std(ddof=0))
    source_path = _write_source(source, "01_main_metric_comparison.csv")

    fig, axes = plt.subplots(
        2,
        len(METRIC_SPECS),
        figsize=(FIGURE_WIDTH_IN, 4.15),
        squeeze=False,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.14, top=0.91, wspace=0.48, hspace=0.28)
    letters = iter("abcdefghij")
    for row_index, case in enumerate(CASES):
        for column_index, (metric, label) in enumerate(METRIC_SPECS):
            ax = axes[row_index, column_index]
            for method_index, model in enumerate(METHODS):
                subset = source[
                    (source["case"] == case)
                    & (source["model"] == model)
                    & (source["metric"] == metric)
                ].sort_values("seed")
                values = subset["value"].to_numpy(float)
                plotted = np.maximum(values, PLOT_FLOOR)
                offsets = np.linspace(-0.055, 0.055, len(values)) if len(values) > 1 else np.array([0.0])
                ax.scatter(
                    method_index + offsets,
                    plotted,
                    s=15,
                    color=METHOD_COLORS[model],
                    marker=METHOD_MARKERS[model],
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.85,
                    zorder=4,
                )
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=0))
                mean_plot = max(mean, PLOT_FLOOR)
                if len(values) > 1:
                    raw_lower = mean - std
                    # A negative additive lower whisker is undefined on a log
                    # axis; omit that side instead of drawing it to an arbitrary
                    # floor many decades below the observations.
                    lower = mean_plot if raw_lower <= PLOT_FLOOR else raw_lower
                    upper = max(mean + std, PLOT_FLOOR)
                    yerr = np.array([[mean_plot - lower], [upper - mean_plot]])
                    ax.errorbar(
                        method_index,
                        mean_plot,
                        yerr=yerr,
                        fmt="_",
                        markersize=10,
                        markeredgewidth=1.35,
                        color=METHOD_COLORS[model],
                        capsize=2.0,
                        linewidth=0.9,
                        zorder=5,
                    )
            ax.set_yscale("log")
            ax.set_xlim(-0.38, len(METHODS) - 0.62)
            ax.set_xticks(range(len(METHODS)))
            if row_index == 1:
                ax.set_xticklabels(
                    ["SP-CPA", "LTPB-CPA", "Oracle\n(n=1)"],
                    rotation=22,
                    ha="right",
                    rotation_mode="anchor",
                )
            else:
                ax.set_xticklabels([])
            if row_index == 0:
                ax.set_title(label, pad=4.0)
            if column_index == 0:
                ax.set_ylabel(f"Error (Case {case})")
            _style_axis(ax, grid_axis="y")
            _add_panel_label(ax, next(letters))
    fig.text(
        0.5,
        0.025,
        "Dots are independent seeds; crossbars show mean +/- population SD (n=3). "
        "Non-positive lower whiskers are omitted on the log axis. Oracle: n=1.",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color="#555555",
    )
    files = _save_figure(fig, "01_main_metric_comparison")
    return {
        "files": files,
        "source_data": [_figure_relative(source_path)],
        "conclusion": "Compares SP-CPA and LTPB-CPA under the matched budget in Cases P and M, with the oracle as a single-run diagnostic reference.",
        "statistics": "Independent seeds n=3; center=mean; spread=population SD; oracle n=1; no hypothesis test.",
    }


def _phase_source(inputs: PilotInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectory_rows: list[dict[str, Any]] = []
    slope_rows: list[dict[str, Any]] = []
    for case in CASES:
        for seed in LEARNED_SEEDS:
            key = (case, "ltpb_cpa", seed)
            frame = inputs.time_series[key]
            metrics = inputs.metrics[key]
            t = frame["t"].to_numpy(float)
            for branch in ("plus", "minus"):
                predicted = frame[f"phase_{branch}_spatial_mean"].to_numpy(float)
                exact = frame[f"phase_{branch}_exact"].to_numpy(float)
                slope, intercept = np.polyfit(t, predicted, deg=1)
                fitted = slope * t + intercept
                residual_sum = float(np.sum((predicted - fitted) ** 2))
                total_sum = float(np.sum((predicted - np.mean(predicted)) ** 2))
                r_squared = 1.0 if total_sum == 0.0 and residual_sum == 0.0 else (
                    0.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum
                )
                target = float(np.polyfit(t, exact, deg=1)[0])
                stored_slope = float(metrics[f"phase_{branch}_slope"])
                stored_target = float(metrics[f"phase_{branch}_slope_target"])
                if not (
                    math.isclose(float(slope), stored_slope, rel_tol=1.0e-10, abs_tol=1.0e-10)
                    and math.isclose(target, stored_target, rel_tol=1.0e-10, abs_tol=1.0e-10)
                ):
                    raise RuntimeError(f"Stored and recomputed phase slopes disagree for {key}, branch={branch}")
                slope_rows.append(
                    {
                        "case": case,
                        "seed": seed,
                        "branch": branch,
                        "slope": float(slope),
                        "target_slope": target,
                        "slope_abs_error": abs(float(slope) - target),
                        "intercept": float(intercept),
                        "fit_r_squared": r_squared,
                    }
                )
                trajectory_rows.extend(
                    {
                        "case": case,
                        "seed": seed,
                        "t": float(time),
                        "branch": branch,
                        "phase_spatial_mean": float(value),
                        "phase_exact": float(target_value),
                    }
                    for time, value, target_value in zip(t, predicted, exact, strict=True)
                )
    return pd.DataFrame(trajectory_rows), pd.DataFrame(slope_rows)


def plot_learned_branch_phases(inputs: PilotInputs) -> dict[str, Any]:
    source, slopes = _phase_source(inputs)
    source_path = _write_source(source, "02_learned_branch_phases.csv")
    slopes_path = _write_source(slopes, "02_learned_branch_phase_slopes.csv")

    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_IN, 4.35), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.12, top=0.86, wspace=0.24, hspace=0.31)
    letters = iter("abcd")
    for row_index, case in enumerate(CASES):
        for column_index, branch in enumerate(("plus", "minus")):
            ax = axes[row_index, column_index]
            subset = source[(source["case"] == case) & (source["branch"] == branch)]
            seed_frames = [subset[subset["seed"] == seed].sort_values("t") for seed in LEARNED_SEEDS]
            t, phase = _aligned_stack(seed_frames, "t", "phase_spatial_mean")
            _, exact_stack = _aligned_stack(seed_frames, "t", "phase_exact")
            if not np.allclose(exact_stack, exact_stack[0], rtol=0.0, atol=1.0e-12):
                raise RuntimeError(f"Exact phase trajectories disagree across seeds for Case {case}, {branch}")
            for seed_index, values in enumerate(phase):
                ax.plot(t, values, color=BRANCH_COLORS[branch], alpha=0.28, linewidth=0.65)
            _plot_mean_band(
                ax,
                t,
                phase,
                color=BRANCH_COLORS[branch],
                label="LTPB mean",
                linewidth=1.5,
            )
            ax.plot(t, exact_stack[0], color=EXACT_COLOR, linestyle="--", linewidth=1.25, zorder=4)

            slope_subset = slopes[(slopes["case"] == case) & (slopes["branch"] == branch)]
            slope_values = slope_subset["slope"].to_numpy(float)
            target = float(slope_subset["target_slope"].iloc[0])
            ax.text(
                0.03,
                0.95,
                f"fit slope = {np.mean(slope_values):+.3f} +/- {np.std(slope_values, ddof=0):.3f}\n"
                f"target = {target:+.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.0,
                color="#444444",
            )
            # Descriptive text avoids backend/font-specific math glyph clipping
            # while keeping the vector titles editable.
            branch_title = "positive-branch phase" if branch == "plus" else "negative-branch phase"
            ax.set_title(f"Case {case}: {branch_title}", pad=5.0)
            if column_index == 0:
                ax.set_ylabel("Spatial-mean phase (rad)")
            if row_index == 1:
                ax.set_xlabel("Time t")
            _style_axis(ax)
            _add_panel_label(ax, next(letters))
            if case == "P" and branch == "minus":
                ax.text(
                    0.98,
                    0.05,
                    r"$a_{-,0}=0$",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=5.5,
                    color="#666666",
                )
    handles = [
        Line2D([0], [0], color=EXACT_COLOR, linestyle="--", linewidth=1.25, label=r"Exact $\mp Et$"),
        Line2D([0], [0], color="#777777", linewidth=0.7, alpha=0.5, label="Individual seeds"),
        Patch(facecolor="#999999", alpha=0.18, edgecolor="none", label="Mean +/- population SD"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3)
    files = _save_figure(fig, "02_learned_branch_phases")
    return {
        "files": files,
        "source_data": [_figure_relative(source_path), _figure_relative(slopes_path)],
        "conclusion": "Tests whether LTPB learns the two opposite temporal phase slopes from the PDE rather than receiving +/-E as carriers.",
        "statistics": "Three independent seeds; line=center mean; band=population SD; slopes are OLS fits over the full 201-point time grid.",
        "note": "The negative branch is initially unoccupied in Case P, so its phase is not physically identifiable at t=0.",
    }


def _method_keys(case: str, model: str) -> list[RunKey]:
    seeds = LEARNED_SEEDS if model != "two_branch" else (0,)
    return [(case, model, seed) for seed in seeds]


def _time_series_source(
    inputs: PilotInputs,
    *,
    case: str,
    predicted_column: str,
    exact_column: str,
    predicted_output: str,
    exact_output: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model in METHODS:
        for key in _method_keys(case, model):
            frame = inputs.time_series[key]
            piece = pd.DataFrame(
                {
                    "case": case,
                    "model": model,
                    "method_label": METHOD_LABELS[model],
                    "seed": key[2],
                    "t": frame["t"].to_numpy(float),
                    predicted_output: frame[predicted_column].to_numpy(float),
                    exact_output: frame[exact_column].to_numpy(float),
                }
            )
            rows.append(piece)
    return pd.concat(rows, ignore_index=True)


def plot_case_m_current(inputs: PilotInputs) -> dict[str, Any]:
    source = _time_series_source(
        inputs,
        case="M",
        predicted_column="current_x0_predicted",
        exact_column="current_x0_exact",
        predicted_output="current_predicted",
        exact_output="current_exact",
    )
    source_path = _write_source(source, "03_case_M_current_x0.csv")

    all_frames = [
        source[(source["model"] == model) & (source["seed"] == key[2])].sort_values("t")
        for model in METHODS
        for key in _method_keys("M", model)
    ]
    _, exact_stack = _aligned_stack(all_frames, "t", "current_exact")
    if not np.allclose(exact_stack, exact_stack[0], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Case M exact-current trajectories disagree across reused runs")

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 2.75))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.2, top=0.82)
    reference = source[(source["model"] == "sp_cpa") & (source["seed"] == 0)].sort_values("t")
    ax.plot(
        reference["t"],
        reference["current_exact"],
        color=EXACT_COLOR,
        linewidth=1.55,
        label="Exact",
        zorder=5,
    )
    for model in ("sp_cpa", "ltpb_cpa"):
        frames = [
            source[(source["model"] == model) & (source["seed"] == seed)].sort_values("t")
            for seed in LEARNED_SEEDS
        ]
        t, values = _aligned_stack(frames, "t", "current_predicted")
        _plot_mean_band(
            ax,
            t,
            values,
            color=METHOD_COLORS[model],
            label=f"{METHOD_LABELS[model]} mean +/- SD",
        )
    oracle = source[source["model"] == "two_branch"].sort_values("t")
    ax.plot(
        oracle["t"],
        oracle["current_predicted"],
        color=METHOD_COLORS["two_branch"],
        linestyle=(0, (4, 2)),
        linewidth=1.35,
        label="Two-Branch oracle (n=1)",
        zorder=6,
    )
    ax.set_xlabel("Time t")
    ax.set_ylabel(r"Probability current $j(0,t)$")
    _style_axis(ax)
    _add_panel_label(ax, "a")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4)
    files = _save_figure(fig, "03_case_M_current_x0")
    return {
        "files": files,
        "source_data": [_figure_relative(source_path)],
        "conclusion": "Directly tests recovery of the Case M current oscillation, which is sensitive to positive/negative-branch interference.",
        "statistics": "SP-CPA and LTPB-CPA: independent seeds n=3, line=mean, band=population SD; oracle n=1.",
    }


def _circularly_aligned_unwrap(
    predicted_wrapped: np.ndarray, exact_wrapped: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unwrap through the circular residual, preserving the exact phase branch."""
    exact_unwrapped = np.unwrap(exact_wrapped)
    residual_wrapped = np.angle(np.exp(1j * (predicted_wrapped - exact_wrapped)))
    residual_unwrapped = np.unwrap(residual_wrapped)
    return exact_unwrapped + residual_unwrapped, exact_unwrapped, residual_unwrapped


def plot_case_m_relative_phase(inputs: PilotInputs) -> dict[str, Any]:
    rows: list[pd.DataFrame] = []
    for model in METHODS:
        for key in _method_keys("M", model):
            frame = inputs.time_series[key]
            predicted_wrapped = frame["relative_phase_x0_predicted"].to_numpy(float)
            exact_wrapped = frame["relative_phase_x0_exact"].to_numpy(float)
            predicted_unwrapped, exact_unwrapped, residual_unwrapped = _circularly_aligned_unwrap(
                predicted_wrapped, exact_wrapped
            )
            rows.append(
                pd.DataFrame(
                    {
                        "case": "M",
                        "model": model,
                        "method_label": METHOD_LABELS[model],
                        "seed": key[2],
                        "t": frame["t"].to_numpy(float),
                        "phase_predicted_wrapped": predicted_wrapped,
                        "phase_exact_wrapped": exact_wrapped,
                        "phase_predicted_aligned_unwrapped": predicted_unwrapped,
                        "phase_exact_unwrapped": exact_unwrapped,
                        "circular_residual_unwrapped": residual_unwrapped,
                    }
                )
            )
    source = pd.concat(rows, ignore_index=True)
    source_path = _write_source(source, "04_case_M_relative_phase_x0.csv")

    all_frames = [
        source[(source["model"] == model) & (source["seed"] == key[2])].sort_values("t")
        for model in METHODS
        for key in _method_keys("M", model)
    ]
    _, exact_stack = _aligned_stack(all_frames, "t", "phase_exact_unwrapped")
    if not np.allclose(exact_stack, exact_stack[0], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Case M exact relative-phase trajectories disagree across reused runs")

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 2.75))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.2, top=0.82)
    reference = source[(source["model"] == "sp_cpa") & (source["seed"] == 0)].sort_values("t")
    ax.plot(
        reference["t"],
        reference["phase_exact_unwrapped"],
        color=EXACT_COLOR,
        linewidth=1.55,
        label="Exact",
        zorder=5,
    )
    for model in ("sp_cpa", "ltpb_cpa"):
        frames = [
            source[(source["model"] == model) & (source["seed"] == seed)].sort_values("t")
            for seed in LEARNED_SEEDS
        ]
        t, values = _aligned_stack(frames, "t", "phase_predicted_aligned_unwrapped")
        _plot_mean_band(
            ax,
            t,
            values,
            color=METHOD_COLORS[model],
            label=f"{METHOD_LABELS[model]} mean +/- SD",
        )
    oracle = source[source["model"] == "two_branch"].sort_values("t")
    ax.plot(
        oracle["t"],
        oracle["phase_predicted_aligned_unwrapped"],
        color=METHOD_COLORS["two_branch"],
        linestyle=(0, (4, 2)),
        linewidth=1.35,
        label="Two-Branch oracle (n=1)",
        zorder=6,
    )
    ax.set_xlabel("Time t")
    ax.set_ylabel(r"Unwrapped $\arg(\psi_1^*\psi_2)$ at $x=0$ (rad)")
    _style_axis(ax)
    _add_panel_label(ax, "a")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4)
    ax.text(
        0.99,
        0.04,
        "Aligned via the circular residual before unwrapping",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="#666666",
    )
    files = _save_figure(fig, "04_case_M_relative_phase_x0")
    return {
        "files": files,
        "source_data": [_figure_relative(source_path)],
        "conclusion": "Checks whether the component relative phase is recovered rather than hidden by density or conservation diagnostics.",
        "statistics": "SP-CPA and LTPB-CPA: independent seeds n=3, line=mean, band=population SD; oracle n=1. Wrapped phases are aligned by circular residual and then unwrapped per seed.",
    }


def _amplitude_occupation_source(inputs: PilotInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in CASES:
        for seed in LEARNED_SEEDS:
            frame = inputs.time_series[(case, "ltpb_cpa", seed)]
            for branch in ("plus", "minus"):
                amplitude = frame[f"amplitude_{branch}_spatial_mean"].to_numpy(float)
                amplitude_std = frame[f"amplitude_{branch}_spatial_std"].to_numpy(float)
                amplitude_exact = frame[f"amplitude_{branch}_exact"].to_numpy(float)
                occupation = frame[f"occupation_{branch}_predicted"].to_numpy(float)
                occupation_exact = frame[f"occupation_{branch}_exact"].to_numpy(float)
                rows.extend(
                    {
                        "case": case,
                        "seed": seed,
                        "t": float(time),
                        "branch": branch,
                        "amplitude_spatial_mean": float(amp),
                        "amplitude_spatial_sd": float(amp_std),
                        "amplitude_exact": float(amp_exact),
                        "amplitude_drift": float(amp - amp_exact),
                        "occupation_predicted": float(occ),
                        "occupation_exact": float(occ_exact),
                    }
                    for time, amp, amp_std, amp_exact, occ, occ_exact in zip(
                        frame["t"].to_numpy(float),
                        amplitude,
                        amplitude_std,
                        amplitude_exact,
                        occupation,
                        occupation_exact,
                        strict=True,
                    )
                )
    return pd.DataFrame(rows)


def plot_amplitude_and_occupations(inputs: PilotInputs) -> dict[str, Any]:
    source = _amplitude_occupation_source(inputs)
    source_path = _write_source(source, "05_amplitude_and_occupations.csv")

    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_IN, 4.45), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.86, wspace=0.24, hspace=0.28)
    for column_index, case in enumerate(CASES):
        amplitude_ax = axes[0, column_index]
        occupation_ax = axes[1, column_index]
        for branch in ("plus", "minus"):
            subset = source[(source["case"] == case) & (source["branch"] == branch)]
            seed_frames = [subset[subset["seed"] == seed].sort_values("t") for seed in LEARNED_SEEDS]
            t, amplitude_drift = _aligned_stack(seed_frames, "t", "amplitude_drift")
            _plot_mean_band(
                amplitude_ax,
                t,
                amplitude_drift,
                color=BRANCH_COLORS[branch],
                label=None,
            )
            _, occupation = _aligned_stack(seed_frames, "t", "occupation_predicted")
            _, occupation_exact = _aligned_stack(seed_frames, "t", "occupation_exact")
            if not np.allclose(occupation_exact, occupation_exact[0], rtol=0.0, atol=1.0e-12):
                raise RuntimeError(
                    f"Exact occupations disagree across LTPB seeds for Case {case}, branch={branch}"
                )
            occupation_mean, _ = _plot_mean_band(
                occupation_ax,
                t,
                occupation,
                color=BRANCH_COLORS[branch],
                label=None,
            )
            marker = "o" if branch == "plus" else "s"
            marker_start = 0 if branch == "plus" else 12
            occupation_ax.plot(
                t,
                occupation_mean,
                linestyle="none",
                marker=marker,
                markevery=(marker_start, 25),
                markersize=1.8,
                markerfacecolor="white",
                markeredgecolor=BRANCH_COLORS[branch],
                markeredgewidth=0.55,
                zorder=4,
            )
            occupation_ax.plot(
                t,
                occupation_exact[0],
                color=BRANCH_COLORS[branch],
                linestyle="--",
                linewidth=1.0,
                alpha=0.9,
            )
        amplitude_ax.axhline(0.0, color=EXACT_COLOR, linestyle="--", linewidth=0.8)
        amplitude_ax.set_title(f"Case {case}")
        amplitude_ax.set_ylabel("Amplitude drift" if column_index == 0 else "")
        occupation_ax.set_ylabel("Branch occupation" if column_index == 0 else "")
        occupation_ax.set_xlabel("Time t")
        occupation_ax.set_ylim(-0.05, 1.05)
        for ax in (amplitude_ax, occupation_ax):
            _style_axis(ax)
        _add_panel_label(amplitude_ax, chr(ord("a") + column_index))
        _add_panel_label(occupation_ax, chr(ord("c") + column_index))
    handles = [
        Line2D(
            [0],
            [0],
            color=BRANCH_COLORS["plus"],
            linewidth=1.5,
            marker="o",
            markersize=2.5,
            markerfacecolor="white",
            label=r"$+$ branch",
        ),
        Line2D(
            [0],
            [0],
            color=BRANCH_COLORS["minus"],
            linewidth=1.5,
            marker="s",
            markersize=2.5,
            markerfacecolor="white",
            label=r"$-$ branch",
        ),
        Patch(facecolor="#888888", alpha=0.18, edgecolor="none", label="Mean +/- population SD"),
        Line2D([0], [0], color=EXACT_COLOR, linestyle="--", linewidth=1.0, label="Exact / zero drift"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4)
    files = _save_figure(fig, "05_amplitude_and_occupations")
    return {
        "files": files,
        "source_data": [_figure_relative(source_path)],
        "conclusion": "Separates phase learning from compensation by real-amplitude drift and checks the inferred positive/negative branch occupations.",
        "statistics": "Three independent LTPB seeds; line=mean of spatial-mean amplitudes or occupations; band=population SD across seeds. Per-run spatial SD is retained in source data.",
    }


def _loss_source(inputs: PilotInputs) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for case in CASES:
        for model in METHODS:
            for key in _method_keys(case, model):
                frame = inputs.loss_history[key]
                rows.append(
                    pd.DataFrame(
                        {
                            "case": case,
                            "model": model,
                            "method_label": METHOD_LABELS[model],
                            "seed": key[2],
                            "step": frame["step"].to_numpy(int),
                            "total_loss": frame["total"].to_numpy(float),
                            "pde_loss": frame["pde"].to_numpy(float),
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


def plot_loss_histories(inputs: PilotInputs) -> dict[str, Any]:
    source = _loss_source(inputs)
    source_path = _write_source(source, "06_loss_histories.csv")

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN, 2.85), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.78, wspace=0.16)
    for case_index, case in enumerate(CASES):
        ax = axes[case_index]
        for model in ("sp_cpa", "ltpb_cpa"):
            frames = [
                source[
                    (source["case"] == case)
                    & (source["model"] == model)
                    & (source["seed"] == seed)
                ].sort_values("step")
                for seed in LEARNED_SEEDS
            ]
            step, values = _aligned_stack(frames, "step", "pde_loss")
            _plot_mean_band(
                ax,
                step,
                values,
                color=METHOD_COLORS[model],
                label=f"{METHOD_LABELS[model]} mean +/- SD",
                log_safe=True,
            )
        oracle = source[(source["case"] == case) & (source["model"] == "two_branch")].sort_values("step")
        ax.semilogy(
            oracle["step"],
            np.maximum(oracle["pde_loss"].to_numpy(float), PLOT_FLOOR),
            color=METHOD_COLORS["two_branch"],
            linestyle=(0, (4, 2)),
            linewidth=1.25,
            label="Two-Branch oracle (n=1)",
        )
        ax.set_yscale("log")
        ax.set_title(f"Case {case}")
        ax.set_xlabel("Adam step")
        if case_index == 0:
            ax.set_ylabel("Dirac PDE loss")
        _style_axis(ax)
        _add_panel_label(ax, chr(ord("a") + case_index))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3)
    files = _save_figure(fig, "06_loss_histories")
    return {
        "files": files,
        "source_data": [_figure_relative(source_path)],
        "conclusion": "Documents optimization behavior under the common 5000-step PDE-only budget without treating training loss as the success criterion.",
        "statistics": "SP-CPA and LTPB-CPA: independent seeds n=3, line=mean, band=population SD where mean-SD remains positive on the log axis; oracle n=1.",
    }


def _write_manifest(figures: dict[str, dict[str, Any]]) -> Path:
    payload = {
        "figure_contract": {
            "core_conclusion": "At matched budget, test whether LTPB learns the +/-E phase slopes and jointly restores Case M spinor, current, relative phase, and fidelity.",
            "evidence_chain": [
                "Dense-grid primary metrics in Cases P and M",
                "Learned branch-phase trajectories and fitted slopes",
                "Case M probability current and component relative phase",
                "Amplitude drift and branch occupations",
                "PDE optimization histories",
            ],
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib only",
            "final_width_mm": FIGURE_WIDTH_MM,
            "text_size_pt": "5-7 (8 pt bold panel labels)",
            "editable_text": "svg.fonttype=none; pdf.fonttype=42",
            "exports": ["PNG 600 dpi", "PDF vector", "SVG vector"],
        },
        "figures": figures,
    }
    path = FIGURE_ROOT / "figure_manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the comparison table, artifacts, configs, and grids without exporting figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = load_inputs()
    if args.check_only:
        print("Formal learned-two-phase plotting inputs are complete and consistent.")
        return

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    figures: dict[str, dict[str, Any]] = {}
    figures["01_main_metric_comparison"] = plot_main_metric_comparison(inputs)
    figures["02_learned_branch_phases"] = plot_learned_branch_phases(inputs)
    figures["03_case_M_current_x0"] = plot_case_m_current(inputs)
    figures["04_case_M_relative_phase_x0"] = plot_case_m_relative_phase(inputs)
    figures["05_amplitude_and_occupations"] = plot_amplitude_and_occupations(inputs)
    figures["06_loss_histories"] = plot_loss_histories(inputs)
    manifest_path = _write_manifest(figures)
    print(
        f"Generated {len(figures)} learned-two-phase figure groups in {FIGURE_ROOT}\n"
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
