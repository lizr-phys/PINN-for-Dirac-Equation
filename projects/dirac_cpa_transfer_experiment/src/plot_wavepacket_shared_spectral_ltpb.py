from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .wavepacket_config import MODEL_LABELS, WAVEPACKET_MODELS, WAVEPACKET_SEEDS


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = RESULTS / "raw" / "wavepacket_shared_spectral_ltpb" / "learned"
FIGURES = RESULTS / "figures" / "wavepacket_shared_spectral_ltpb"
RUNS_CSV = RESULTS / "wavepacket_shared_spectral_ltpb_runs.csv"
REPRESENTATIVE_MODEL = "shared_spectral_ltpb"
REPRESENTATIVE_SEED = 0

FIGURE_IDS = (
    "primary_error_comparison",
    "per_seed_results",
    "wavepacket_density_spacetime",
    "wavepacket_current_spacetime",
    "branch_center_width",
    "phase_slope_train_midpoint",
    "spectral_coefficient_errors",
    "training_curves",
    "accuracy_time_memory",
)

EXPECTED_SOURCE_ROWS = {
    "primary_error_comparison": 100,
    "per_seed_results": 75,
    "wavepacket_density_spacetime": 201 * 256,
    "wavepacket_current_spacetime": 201 * 256,
    "branch_center_width": 5 * 201 * 2,
    "phase_slope_train_midpoint": 5 * (33 + 32) * 2,
    "spectral_coefficient_errors": 5 * 201 * 33 * 2,
    "training_curves": 25 * 501,
    "accuracy_time_memory": 25,
}

SOURCE_SCHEMAS = {
    "primary_error_comparison": (
        "model",
        "seed",
        "metric",
        "value",
        "display_value",
    ),
    "per_seed_results": ("model", "seed", "metric", "value"),
    "wavepacket_density_spacetime": (
        "model",
        "seed",
        "t",
        "x",
        "exact",
        "predicted",
        "absolute_error",
    ),
    "wavepacket_current_spacetime": (
        "model",
        "seed",
        "t",
        "x",
        "exact",
        "predicted",
        "signed_error",
        "absolute_error",
    ),
    "branch_center_width": (
        "model",
        "seed",
        "t",
        "branch",
        "center_predicted",
        "center_exact",
        "width_predicted",
        "width_exact",
    ),
    "phase_slope_train_midpoint": (
        "model",
        "seed",
        "set",
        "k",
        "branch",
        "slope_predicted",
        "slope_exact",
        "relative_error",
        "valid_fraction",
    ),
    "spectral_coefficient_errors": (
        "model",
        "seed",
        "t",
        "k",
        "branch",
        "coefficient_abs_error",
        "coefficient_relative_error",
        "occupation_predicted",
        "occupation_exact",
        "occupation_abs_error",
        "occupation_relative_error",
        "initial_occupation",
    ),
    "training_curves": (
        "model",
        "seed",
        "step",
        "pde_loss",
        "elapsed_seconds",
    ),
    "accuracy_time_memory": (
        "model",
        "seed",
        "spinor_error",
        "current_error",
        "training_seconds",
        "peak_rss_mb",
        "peak_rss_increment_mb",
        "parameter_count",
    ),
}

COLORS = {
    "physical_vanilla": "#5B7894",
    "physical_sp_cpa": "#D39144",
    "physical_component_polar": "#5E9178",
    "shared_spectral_complex": "#8B6F9B",
    "shared_spectral_ltpb": "#C95D55",
    "exact": "#202124",
    "train": "#C95D55",
    "midpoint": "#6D77A8",
}
BRANCH_COLORS = {"plus": "#C95D55", "minus": "#4E79A7"}
BRANCH_LABELS = {"plus": "Positive-energy branch", "minus": "Negative-energy branch"}
DISPLAY_FLOOR = 1.0e-15


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.35,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _panel_labels(axes: Iterable[plt.Axes]) -> None:
    for label, axis in zip("abcdefghijklmnopqrstuvwxyz", axes):
        axis.text(
            -0.12,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
            ha="right",
        )


def _save(
    fig: plt.Figure,
    name: str,
    source: pd.DataFrame,
    figure_dir: Path,
    *,
    inputs: Iterable[Path],
    raster_allowed: bool = False,
) -> dict[str, object]:
    expected_columns = list(SOURCE_SCHEMAS[name])
    if source.columns.tolist() != expected_columns:
        raise ValueError(
            f"{name} source columns differ from contract: {source.columns.tolist()}"
        )
    if len(source) != EXPECTED_SOURCE_ROWS[name]:
        raise ValueError(
            f"{name} expected {EXPECTED_SOURCE_ROWS[name]} rows, received {len(source)}"
        )
    numeric = source.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if numeric.size and not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} source data contain non-finite numeric values")

    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir = figure_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / f"{name}.csv"
    source.to_csv(source_path, index=False, float_format="%.17g")

    outputs: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    for suffix in ("png", "svg", "pdf"):
        path = figure_dir / f"{name}.{suffix}"
        kwargs: dict[str, object] = {
            "facecolor": "white",
            "metadata": {"Creator": "Python/matplotlib formal wavepacket plotter"},
        }
        if suffix == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs[suffix] = _manifest_path(path)
        output_hashes[suffix] = _sha256(path)
    plt.close(fig)

    unique_inputs = sorted({Path(path).resolve() for path in inputs}, key=str)
    input_records = [
        {"path": _manifest_path(path), "sha256": _sha256(path)}
        for path in unique_inputs
    ]
    return {
        "id": name,
        "source_data": _manifest_path(source_path),
        "source_rows": int(len(source)),
        "source_columns": expected_columns,
        "source_sha256": _sha256(source_path),
        "input_files": input_records,
        "outputs": outputs,
        "output_sha256": output_hashes,
        "raster_allowed": bool(raster_allowed),
        "population_sd_ddof": 0,
    }


def _run_dir(raw_root: Path, model: str, seed: int) -> Path:
    return raw_root / model / f"seed_{seed}"


def _require_formal_inputs(results_root: Path, raw_root: Path) -> Path:
    runs_path = results_root / RUNS_CSV.name
    if not runs_path.is_file():
        raise FileNotFoundError(
            f"Formal aggregate does not exist; plotting is intentionally blocked: {runs_path}"
        )
    required = {
        "config.json",
        "metrics.json",
        "loss_history.csv",
        "time_series.csv",
        "dense_grid.npz",
        "model_state.pt",
    }
    for model in WAVEPACKET_MODELS:
        for seed in WAVEPACKET_SEEDS:
            run_dir = _run_dir(raw_root, model, seed)
            if not run_dir.is_dir():
                raise FileNotFoundError(f"Missing formal run: {run_dir}")
            observed = {path.name for path in run_dir.iterdir()}
            if observed != required:
                raise RuntimeError(
                    f"Formal run inventory mismatch at {run_dir}: {sorted(observed)}"
                )
    return runs_path


def _load_runs(path: Path) -> pd.DataFrame:
    runs = pd.read_csv(path)
    expected_pairs = {
        (model, seed) for model in WAVEPACKET_MODELS for seed in WAVEPACKET_SEEDS
    }
    observed_pairs = set(zip(runs["model"], runs["seed"], strict=True))
    if len(runs) != 25 or observed_pairs != expected_pairs:
        raise ValueError("Formal aggregate must contain exactly five models x five seeds")
    return runs.sort_values(["model", "seed"], kind="stable").reset_index(drop=True)


def _load_dense(raw_root: Path, model: str, seed: int) -> tuple[dict[str, np.ndarray], Path]:
    path = _run_dir(raw_root, model, seed) / "dense_grid.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    return arrays, path


def _target_time_series(raw_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for seed in WAVEPACKET_SEEDS:
        path = _run_dir(raw_root, REPRESENTATIVE_MODEL, seed) / "time_series.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        frames.append(frame)
        paths.append(path)
    return pd.concat(frames, ignore_index=True), paths


def _all_history(raw_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for model in WAVEPACKET_MODELS:
        for seed in WAVEPACKET_SEEDS:
            path = _run_dir(raw_root, model, seed) / "loss_history.csv"
            frame = pd.read_csv(path)
            frame.insert(0, "seed", seed)
            frame.insert(0, "model", model)
            frames.append(frame)
            paths.append(path)
    return pd.concat(frames, ignore_index=True), paths


def figure_primary(
    runs: pd.DataFrame,
    figure_dir: Path,
    runs_path: Path,
) -> dict[str, object]:
    records: list[pd.DataFrame] = []
    for metric, source_column, complement in (
        ("spinor_relative_l2", "spinor_rel_l2_gauge_aligned", False),
        ("density_relative_l2", "density_rel_l2", False),
        ("current_relative_l2", "current_rel_l2", False),
        ("infidelity", "fidelity_mean", True),
    ):
        values = 1.0 - runs[source_column] if complement else runs[source_column]
        records.append(
            pd.DataFrame(
                {
                    "model": runs["model"],
                    "seed": runs["seed"].astype(int),
                    "metric": metric,
                    "value": values.astype(float),
                    "display_value": np.maximum(values.astype(float), DISPLAY_FLOOR),
                }
            )
        )
    source = pd.concat(records, ignore_index=True)[list(SOURCE_SCHEMAS["primary_error_comparison"])]

    fig, axes = plt.subplots(1, 4, figsize=(7.35, 2.65))
    titles = {
        "spinor_relative_l2": "Spinor relative L2",
        "density_relative_l2": "Density relative L2",
        "current_relative_l2": "Current relative L2",
        "infidelity": r"Infidelity $1-\overline{F}$",
    }
    positions = np.arange(len(WAVEPACKET_MODELS))
    for axis, metric in zip(axes, titles, strict=True):
        for index, model in enumerate(WAVEPACKET_MODELS):
            values = source[(source.model == model) & (source.metric == metric)][
                "value"
            ].to_numpy(float)
            shown = np.maximum(values, DISPLAY_FLOOR)
            center_raw = float(np.mean(values))
            spread = float(np.std(values, ddof=0))
            center = max(center_raw, DISPLAY_FLOOR)
            lower = max(center_raw - spread, DISPLAY_FLOOR)
            upper = max(center_raw + spread, DISPLAY_FLOOR)
            axis.errorbar(
                index,
                center,
                yerr=np.asarray([[center - lower], [upper - center]]),
                fmt="o",
                ms=5.0,
                capsize=2.2,
                color=COLORS[model],
                zorder=4,
            )
            jitter = np.linspace(-0.075, 0.075, len(shown))
            axis.scatter(
                index + jitter,
                shown,
                s=9,
                color=COLORS[model],
                alpha=0.45,
                linewidths=0,
                zorder=3,
            )
        axis.set_yscale("log")
        axis.set_title(titles[metric])
        axis.set_xticks(
            positions,
            [MODEL_LABELS[model] for model in WAVEPACKET_MODELS],
            rotation=42,
            ha="right",
        )
        axis.grid(axis="y", which="both", alpha=0.16)
    axes[0].set_ylabel("Error (lower is better)")
    axes[-1].text(
        0.98,
        0.02,
        "display floor $10^{-15}$",
        transform=axes[-1].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#666666",
    )
    _panel_labels(axes)
    fig.tight_layout(w_pad=0.65)
    return _save(fig, "primary_error_comparison", source, figure_dir, inputs=[runs_path])


def figure_per_seed(
    runs: pd.DataFrame,
    figure_dir: Path,
    runs_path: Path,
) -> dict[str, object]:
    frames: list[pd.DataFrame] = []
    for metric, column in (
        ("spinor_relative_l2", "spinor_rel_l2_gauge_aligned"),
        ("current_relative_l2", "current_rel_l2"),
        ("fidelity", "fidelity_mean"),
    ):
        frames.append(
            pd.DataFrame(
                {
                    "model": runs.model,
                    "seed": runs.seed.astype(int),
                    "metric": metric,
                    "value": runs[column].astype(float),
                }
            )
        )
    source = pd.concat(frames, ignore_index=True)[list(SOURCE_SCHEMAS["per_seed_results"])]

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.55), sharex=True)
    specs = (
        ("spinor_relative_l2", "Spinor relative L2", True),
        ("current_relative_l2", "Current relative L2", True),
        ("fidelity", "Mean fidelity", False),
    )
    for axis, (metric, title, logarithmic) in zip(axes, specs, strict=True):
        for model in WAVEPACKET_MODELS:
            subset = source[(source.model == model) & (source.metric == metric)].sort_values(
                "seed"
            )
            axis.plot(
                subset.seed,
                np.maximum(subset.value, DISPLAY_FLOOR),
                marker="o",
                markersize=3.2,
                color=COLORS[model],
                label=MODEL_LABELS[model],
            )
        if logarithmic:
            axis.set_yscale("log")
        else:
            axis.set_ylim(-0.03, 1.03)
        axis.set_title(title)
        axis.set_xlabel("Paired seed")
        axis.set_xticks(WAVEPACKET_SEEDS)
        axis.grid(axis="y", which="both", alpha=0.16)
    axes[0].set_ylabel("Metric value")
    axes[-1].legend(loc="lower center", bbox_to_anchor=(-0.72, -0.48), ncol=3)
    _panel_labels(axes)
    fig.tight_layout(w_pad=0.8)
    fig.subplots_adjust(bottom=0.31)
    return _save(fig, "per_seed_results", source, figure_dir, inputs=[runs_path])


def _representative_field_source(
    raw_root: Path,
    quantity: str,
) -> tuple[pd.DataFrame, Path]:
    dense, path = _load_dense(raw_root, REPRESENTATIVE_MODEL, REPRESENTATIVE_SEED)
    x = np.asarray(dense["x"], dtype=np.float64)
    times = np.asarray(dense["t"], dtype=np.float64)
    predicted = np.asarray(dense[f"{quantity}_predicted"], dtype=np.float64)
    exact = np.asarray(dense[f"{quantity}_exact"], dtype=np.float64)
    if predicted.shape != (201, 256) or exact.shape != (201, 256):
        raise ValueError(f"Unexpected {quantity} dense shape: {predicted.shape}/{exact.shape}")
    t_mesh, x_mesh = np.meshgrid(times, x, indexing="ij")
    payload: dict[str, object] = {
        "model": REPRESENTATIVE_MODEL,
        "seed": REPRESENTATIVE_SEED,
        "t": t_mesh.ravel(),
        "x": x_mesh.ravel(),
        "exact": exact.ravel(),
        "predicted": predicted.ravel(),
    }
    difference = predicted - exact
    if quantity == "current":
        payload["signed_error"] = difference.ravel()
    payload["absolute_error"] = np.abs(difference).ravel()
    name = (
        "wavepacket_density_spacetime"
        if quantity == "rho"
        else "wavepacket_current_spacetime"
    )
    return pd.DataFrame(payload)[list(SOURCE_SCHEMAS[name])], path


def figure_density_spacetime(raw_root: Path, figure_dir: Path) -> dict[str, object]:
    source, dense_path = _representative_field_source(raw_root, "rho")
    exact = source.exact.to_numpy().reshape(201, 256)
    predicted = source.predicted.to_numpy().reshape(201, 256)
    error = source.absolute_error.to_numpy().reshape(201, 256)
    extent = [source.x.min(), source.x.max(), source.t.min(), source.t.max()]
    shared_max = float(max(np.max(exact), np.max(predicted)))
    error_max = float(np.max(error))

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.65), constrained_layout=True)
    images = [
        axes[0].imshow(
            exact,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="cividis",
            vmin=0.0,
            vmax=shared_max,
            rasterized=True,
        ),
        axes[1].imshow(
            predicted,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="cividis",
            vmin=0.0,
            vmax=shared_max,
            rasterized=True,
        ),
        axes[2].imshow(
            error,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=max(error_max, np.finfo(float).eps),
            rasterized=True,
        ),
    ]
    for axis, title in zip(axes, ("Exact density", "Shared-Spectral-LTPB", "Absolute error"), strict=True):
        axis.set_title(title)
        axis.set_xlabel("Position x")
    axes[0].set_ylabel("Time t")
    fig.colorbar(images[1], ax=axes[:2], shrink=0.86, label=r"Density $\rho$")
    fig.colorbar(images[2], ax=axes[2], shrink=0.86, label=r"$|\widehat{\rho}-\rho|$")
    _panel_labels(axes)
    return _save(
        fig,
        "wavepacket_density_spacetime",
        source,
        figure_dir,
        inputs=[dense_path],
        raster_allowed=True,
    )


def figure_current_spacetime(raw_root: Path, figure_dir: Path) -> dict[str, object]:
    source, dense_path = _representative_field_source(raw_root, "current")
    exact = source.exact.to_numpy().reshape(201, 256)
    predicted = source.predicted.to_numpy().reshape(201, 256)
    difference = source.signed_error.to_numpy().reshape(201, 256)
    extent = [source.x.min(), source.x.max(), source.t.min(), source.t.max()]
    shared_limit = float(max(np.max(np.abs(exact)), np.max(np.abs(predicted))))
    error_limit = float(np.max(np.abs(difference)))

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.65), constrained_layout=True)
    images = [
        axes[0].imshow(
            exact,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            vmin=-shared_limit,
            vmax=shared_limit,
            rasterized=True,
        ),
        axes[1].imshow(
            predicted,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            vmin=-shared_limit,
            vmax=shared_limit,
            rasterized=True,
        ),
        axes[2].imshow(
            difference,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="PuOr_r",
            vmin=-max(error_limit, np.finfo(float).eps),
            vmax=max(error_limit, np.finfo(float).eps),
            rasterized=True,
        ),
    ]
    for axis, title in zip(axes, ("Exact current", "Shared-Spectral-LTPB", "Signed error"), strict=True):
        axis.set_title(title)
        axis.set_xlabel("Position x")
    axes[0].set_ylabel("Time t")
    fig.colorbar(images[1], ax=axes[:2], shrink=0.86, label="Probability current j")
    fig.colorbar(images[2], ax=axes[2], shrink=0.86, label=r"$\widehat{j}-j$")
    _panel_labels(axes)
    return _save(
        fig,
        "wavepacket_current_spacetime",
        source,
        figure_dir,
        inputs=[dense_path],
        raster_allowed=True,
    )


def figure_branch_center_width(raw_root: Path, figure_dir: Path) -> dict[str, object]:
    times, paths = _target_time_series(raw_root)
    frames: list[pd.DataFrame] = []
    for branch in ("plus", "minus"):
        frame = pd.DataFrame(
            {
                "model": REPRESENTATIVE_MODEL,
                "seed": times.seed.astype(int),
                "t": times.t.astype(float),
                "branch": branch,
                "center_predicted": times[f"branch_{branch}_center"].astype(float),
                "center_exact": times[f"branch_{branch}_center_exact"].astype(float),
                "width_predicted": times[f"branch_{branch}_width"].astype(float),
                "width_exact": times[f"branch_{branch}_width_exact"].astype(float),
            }
        )
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)[list(SOURCE_SCHEMAS["branch_center_width"])]

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.8), sharex=True)
    for branch in ("plus", "minus"):
        subset = source[source.branch == branch]
        exact = subset[subset.seed == REPRESENTATIVE_SEED].sort_values("t")
        color = BRANCH_COLORS[branch]
        for axis, quantity in zip(axes, ("center", "width"), strict=True):
            pivot = subset.pivot(index="t", columns="seed", values=f"{quantity}_predicted")
            mean = pivot.mean(axis=1)
            spread = pivot.std(axis=1, ddof=0)
            axis.fill_between(
                pivot.index,
                mean - spread,
                mean + spread,
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            axis.plot(
                pivot.index,
                mean,
                color=color,
                label=f"{BRANCH_LABELS[branch]} prediction",
            )
            axis.plot(
                exact.t,
                exact[f"{quantity}_exact"],
                color=color,
                linestyle="--",
                linewidth=1.0,
                label=f"{BRANCH_LABELS[branch]} exact",
            )
    axes[0].axhline(0.0, color="#999999", linewidth=0.6, zorder=0)
    axes[0].set_title("Branch probability centers")
    axes[1].set_title("Branch circular widths")
    axes[0].set_ylabel("Unwrapped center")
    axes[1].set_ylabel("Circular standard deviation")
    for axis in axes:
        axis.set_xlabel("Time t")
        axis.grid(alpha=0.16)
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.10, -0.26), ncol=2)
    _panel_labels(axes)
    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(bottom=0.29)
    return _save(fig, "branch_center_width", source, figure_dir, inputs=paths)


def _phase_source(raw_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for seed in WAVEPACKET_SEEDS:
        dense, path = _load_dense(raw_root, REPRESENTATIVE_MODEL, seed)
        paths.append(path)
        for set_name, prefix, k_key in (
            ("training", "", "training_wave_numbers"),
            ("midpoint", "heldout_", "heldout_wave_numbers"),
        ):
            k = np.asarray(dense[k_key], dtype=np.float64)
            slopes = np.asarray(dense[f"{prefix}phase_slopes"], dtype=np.float64)
            targets = np.asarray(dense[f"{prefix}phase_slope_targets"], dtype=np.float64)
            errors = np.asarray(
                dense[f"{prefix}phase_slope_relative_errors"], dtype=np.float64
            )
            valid = np.asarray(dense[f"{prefix}phase_valid_fractions"], dtype=np.float64)
            if slopes.shape != (len(k), 2):
                raise ValueError(f"Unexpected {set_name} slope shape for seed {seed}")
            for branch_index, branch in enumerate(("plus", "minus")):
                frames.append(
                    pd.DataFrame(
                        {
                            "model": REPRESENTATIVE_MODEL,
                            "seed": seed,
                            "set": set_name,
                            "k": k,
                            "branch": branch,
                            "slope_predicted": slopes[:, branch_index],
                            "slope_exact": targets[:, branch_index],
                            "relative_error": errors[:, branch_index],
                            "valid_fraction": valid[:, branch_index],
                        }
                    )
                )
    source = pd.concat(frames, ignore_index=True)
    return source[list(SOURCE_SCHEMAS["phase_slope_train_midpoint"])], paths


def figure_phase_slope(raw_root: Path, figure_dir: Path) -> dict[str, object]:
    source, paths = _phase_source(raw_root)
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 4.55), sharex="col")
    for column, branch in enumerate(("plus", "minus")):
        branch_source = source[source.branch == branch]
        exact = branch_source[branch_source.seed == REPRESENTATIVE_SEED].sort_values("k")
        exact = exact.drop_duplicates("k")
        axes[0, column].plot(
            exact.k,
            exact.slope_exact,
            color=COLORS["exact"],
            linestyle="--",
            linewidth=1.4,
            label="Exact target",
            zorder=2,
        )
        for set_name, marker in (("training", "o"), ("midpoint", "^")):
            subset = branch_source[branch_source["set"] == set_name]
            set_color = COLORS["train" if set_name == "training" else set_name]
            stats = subset.groupby("k", sort=True).slope_predicted.agg(
                mean="mean", std=lambda values: np.std(values, ddof=0)
            )
            axes[0, column].errorbar(
                stats.index,
                stats["mean"],
                yerr=stats["std"],
                fmt=marker,
                markersize=2.8,
                capsize=1.2,
                linestyle="none",
                color=set_color,
                alpha=0.86,
                label="Training modes" if set_name == "training" else "Held-out midpoints",
                zorder=3,
            )
            error_stats = subset.groupby("k", sort=True).relative_error.agg(
                mean="mean", std=lambda values: np.std(values, ddof=0)
            )
            axes[1, column].errorbar(
                error_stats.index,
                np.maximum(error_stats["mean"], DISPLAY_FLOOR),
                yerr=error_stats["std"],
                fmt=marker,
                markersize=2.8,
                capsize=1.2,
                linestyle="none",
                color=set_color,
                alpha=0.86,
            )
        axes[0, column].set_title(BRANCH_LABELS[branch])
        axes[0, column].grid(alpha=0.15)
        axes[1, column].set_yscale("log")
        axes[1, column].axhline(0.01, color="#777777", linestyle="--", linewidth=0.8)
        axes[1, column].axhline(0.02, color="#999999", linestyle=":", linewidth=0.8)
        axes[1, column].set_xlabel("Wave number k")
        axes[1, column].grid(axis="y", which="both", alpha=0.15)
    axes[0, 0].set_ylabel("Phase slope")
    axes[1, 0].set_ylabel("Absolute relative error")
    axes[0, 1].legend(loc="best")
    axes[1, 1].text(
        0.98,
        0.05,
        "dashed: 1% train\ndotted: 2% midpoint",
        transform=axes[1, 1].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#555555",
    )
    _panel_labels(axes.flat)
    fig.tight_layout(h_pad=0.7, w_pad=0.9)
    return _save(fig, "phase_slope_train_midpoint", source, figure_dir, inputs=paths)


def _spectral_error_source(raw_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for seed in WAVEPACKET_SEEDS:
        dense, path = _load_dense(raw_root, REPRESENTATIVE_MODEL, seed)
        paths.append(path)
        times = np.asarray(dense["t"], dtype=np.float64)
        k = np.asarray(dense["training_wave_numbers"], dtype=np.float64)
        predicted = np.asarray(
            dense["branch_coefficients_predicted_gauge_aligned"], dtype=np.complex128
        )
        exact = np.asarray(dense["branch_coefficients_exact"], dtype=np.complex128)
        occupation_predicted = np.asarray(
            dense["branch_occupations_predicted"], dtype=np.float64
        )
        occupation_exact = np.asarray(
            dense["branch_occupations_exact"], dtype=np.float64
        )
        coefficient_abs_error = np.abs(predicted - exact)
        coefficient_relative_error = coefficient_abs_error / np.maximum(
            np.abs(exact), 1.0e-30
        )
        occupation_abs_error = np.abs(occupation_predicted - occupation_exact)
        occupation_relative_error = occupation_abs_error / np.maximum(
            occupation_exact, 1.0e-30
        )
        initial_occupation = occupation_exact[0]
        t_index, k_index, branch_index = np.indices(predicted.shape)
        branch_names = np.asarray(("plus", "minus"), dtype=object)[branch_index.ravel()]
        frames.append(
            pd.DataFrame(
                {
                    "model": REPRESENTATIVE_MODEL,
                    "seed": seed,
                    "t": times[t_index.ravel()],
                    "k": k[k_index.ravel()],
                    "branch": branch_names,
                    "coefficient_abs_error": coefficient_abs_error.ravel(),
                    "coefficient_relative_error": coefficient_relative_error.ravel(),
                    "occupation_predicted": occupation_predicted.ravel(),
                    "occupation_exact": occupation_exact.ravel(),
                    "occupation_abs_error": occupation_abs_error.ravel(),
                    "occupation_relative_error": occupation_relative_error.ravel(),
                    "initial_occupation": initial_occupation[
                        k_index.ravel(), branch_index.ravel()
                    ],
                }
            )
        )
    source = pd.concat(frames, ignore_index=True)
    return source[list(SOURCE_SCHEMAS["spectral_coefficient_errors"])], paths


def figure_spectral_errors(raw_root: Path, figure_dir: Path) -> dict[str, object]:
    source, paths = _spectral_error_source(raw_root)
    representative = source[source.seed == REPRESENTATIVE_SEED]
    times = np.sort(representative.t.unique())
    wave_numbers = np.sort(representative.k.unique())
    heatmaps: dict[str, np.ndarray] = {}
    for branch in ("plus", "minus"):
        heatmaps[branch] = (
            representative[representative.branch == branch]
            .pivot(index="t", columns="k", values="coefficient_abs_error")
            .reindex(index=times, columns=wave_numbers)
            .to_numpy(float)
        )
    shared_max = max(float(np.max(value)) for value in heatmaps.values())

    fig, axes = plt.subplots(2, 2, figsize=(7.35, 4.7), constrained_layout=True)
    images = []
    for axis, branch in zip(axes[0], ("plus", "minus"), strict=True):
        image = axis.imshow(
            heatmaps[branch],
            origin="lower",
            aspect="auto",
            extent=[wave_numbers.min(), wave_numbers.max(), times.min(), times.max()],
            cmap="magma",
            vmin=0.0,
            vmax=max(shared_max, np.finfo(float).eps),
            rasterized=True,
        )
        images.append(image)
        axis.set_title(f"{BRANCH_LABELS[branch]} coefficient error")
        axis.set_xlabel("Wave number k")
        axis.set_ylabel("Time t")
    fig.colorbar(images[-1], ax=axes[0, :], shrink=0.84, label=r"$|\widehat{c}-c|$")

    per_seed_k = (
        source.assign(square=lambda frame: frame.coefficient_relative_error**2)
        .groupby(["seed", "branch", "k"], as_index=False)
        .square.mean()
    )
    per_seed_k["time_rms_relative_error"] = np.sqrt(per_seed_k.square)
    for branch in ("plus", "minus"):
        subset = per_seed_k[per_seed_k.branch == branch]
        stats = subset.groupby("k", sort=True).time_rms_relative_error.agg(
            mean="mean", std=lambda values: np.std(values, ddof=0)
        )
        axes[1, 0].plot(
            stats.index,
            np.maximum(stats["mean"], DISPLAY_FLOOR),
            color=BRANCH_COLORS[branch],
            label=BRANCH_LABELS[branch],
        )
        axes[1, 0].fill_between(
            stats.index,
            np.maximum(stats["mean"] - stats["std"], DISPLAY_FLOOR),
            np.maximum(stats["mean"] + stats["std"], DISPLAY_FLOOR),
            color=BRANCH_COLORS[branch],
            alpha=0.15,
            linewidth=0,
        )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Time-RMS coefficient error by k")
    axes[1, 0].set_xlabel("Wave number k")
    axes[1, 0].set_ylabel("Relative error")
    axes[1, 0].grid(axis="y", which="both", alpha=0.15)
    axes[1, 0].legend()

    weighted_records: list[dict[str, object]] = []
    for (seed, branch), subset in source.groupby(["seed", "branch"], sort=False):
        numerator = float(
            np.sum(subset.initial_occupation * subset.occupation_abs_error**2)
        )
        denominator = float(
            np.sum(subset.initial_occupation * subset.occupation_exact**2)
        )
        weighted_records.append(
            {
                "seed": int(seed),
                "branch": str(branch),
                "value": math.sqrt(numerator / max(denominator, 1.0e-30)),
            }
        )
    weighted = pd.DataFrame(weighted_records)
    for index, branch in enumerate(("plus", "minus")):
        values = weighted[weighted.branch == branch].value.to_numpy(float)
        axes[1, 1].scatter(
            index + np.linspace(-0.06, 0.06, len(values)),
            np.maximum(values, DISPLAY_FLOOR),
            s=13,
            color=BRANCH_COLORS[branch],
            alpha=0.5,
        )
        axes[1, 1].errorbar(
            index,
            max(float(np.mean(values)), DISPLAY_FLOOR),
            yerr=float(np.std(values, ddof=0)),
            fmt="D",
            markersize=5,
            capsize=2.2,
            color=BRANCH_COLORS[branch],
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks((0, 1), ("Positive", "Negative"))
    axes[1, 1].set_title("Occupation-weighted error")
    axes[1, 1].set_ylabel("Weighted relative RMS")
    axes[1, 1].grid(axis="y", which="both", alpha=0.15)
    _panel_labels(axes.flat)
    return _save(
        fig,
        "spectral_coefficient_errors",
        source,
        figure_dir,
        inputs=paths,
        raster_allowed=True,
    )


def figure_training(raw_root: Path, figure_dir: Path) -> dict[str, object]:
    history, paths = _all_history(raw_root)
    source = history[
        ["model", "seed", "step", "pde", "elapsed_seconds"]
    ].rename(columns={"pde": "pde_loss"})
    source = source[list(SOURCE_SCHEMAS["training_curves"])]

    fig, axis = plt.subplots(figsize=(5.35, 3.15))
    for model in WAVEPACKET_MODELS:
        subset = source[source.model == model]
        for _, run in subset.groupby("seed"):
            axis.plot(
                run.step,
                np.maximum(run.pde_loss, DISPLAY_FLOOR),
                color=COLORS[model],
                alpha=0.11,
                linewidth=0.55,
            )
        pivot = subset.pivot(index="step", columns="seed", values="pde_loss")
        center = pivot.mean(axis=1)
        spread = pivot.std(axis=1, ddof=0)
        axis.plot(
            pivot.index,
            np.maximum(center, DISPLAY_FLOOR),
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )
        axis.fill_between(
            pivot.index,
            np.maximum(center - spread, DISPLAY_FLOOR),
            np.maximum(center + spread, DISPLAY_FLOOR),
            color=COLORS[model],
            alpha=0.10,
            linewidth=0,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Adam step")
    axis.set_ylabel("Strong Dirac PDE loss")
    axis.grid(which="both", alpha=0.15)
    axis.legend(ncol=2)
    fig.tight_layout()
    return _save(fig, "training_curves", source, figure_dir, inputs=paths)


def figure_accuracy_time_memory(
    runs: pd.DataFrame,
    figure_dir: Path,
    runs_path: Path,
) -> dict[str, object]:
    source = runs[
        [
            "model",
            "seed",
            "spinor_rel_l2_gauge_aligned",
            "current_rel_l2",
            "training_seconds",
            "peak_rss_mb",
            "peak_rss_increment_mb",
            "parameter_count",
        ]
    ].rename(
        columns={
            "spinor_rel_l2_gauge_aligned": "spinor_error",
            "current_rel_l2": "current_error",
        }
    )
    source = source[list(SOURCE_SCHEMAS["accuracy_time_memory"])]

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.0), sharey=True)
    for model in WAVEPACKET_MODELS:
        subset = source[source.model == model]
        label = f"{MODEL_LABELS[model]} ({int(round(subset.parameter_count.mean())):,} params)"
        for axis, x_column in zip(
            axes, ("training_seconds", "peak_rss_increment_mb"), strict=True
        ):
            axis.scatter(
                subset[x_column],
                np.maximum(subset.spinor_error, DISPLAY_FLOOR),
                s=19,
                color=COLORS[model],
                alpha=0.48,
                linewidths=0,
            )
            axis.scatter(
                subset[x_column].mean(),
                max(float(subset.spinor_error.mean()), DISPLAY_FLOOR),
                s=52,
                marker="D",
                color=COLORS[model],
                edgecolor="white",
                linewidth=0.6,
                label=label if axis is axes[1] else None,
                zorder=4,
            )
    axes[0].set_xlabel("Training time (s)")
    axes[1].set_xlabel("Peak RSS increment (MiB)")
    axes[0].set_ylabel("Spinor relative L2")
    axes[0].set_title("Accuracy versus wall time")
    axes[1].set_title("Accuracy versus training memory")
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(which="both", alpha=0.15)
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.08, -0.27), ncol=2)
    _panel_labels(axes)
    fig.tight_layout(w_pad=0.9)
    fig.subplots_adjust(bottom=0.32)
    return _save(fig, "accuracy_time_memory", source, figure_dir, inputs=[runs_path])


def generate_all(
    results_root: Path = RESULTS,
    raw_root: Path = RAW,
    figure_dir: Path = FIGURES,
) -> list[dict[str, object]]:
    """Generate the nine formal figures; refuses incomplete or smoke inputs."""

    _style()
    runs_path = _require_formal_inputs(results_root, raw_root)
    runs = _load_runs(runs_path)
    manifest = [
        figure_primary(runs, figure_dir, runs_path),
        figure_per_seed(runs, figure_dir, runs_path),
        figure_density_spacetime(raw_root, figure_dir),
        figure_current_spacetime(raw_root, figure_dir),
        figure_branch_center_width(raw_root, figure_dir),
        figure_phase_slope(raw_root, figure_dir),
        figure_spectral_errors(raw_root, figure_dir),
        figure_training(raw_root, figure_dir),
        figure_accuracy_time_memory(runs, figure_dir, runs_path),
    ]
    if tuple(item["id"] for item in manifest) != FIGURE_IDS:
        raise AssertionError("Figure manifest order differs from the frozen contract")
    payload = {
        "experiment": "wavepacket_shared_spectral_ltpb",
        "backend": "Python/matplotlib",
        "representative_model": REPRESENTATIVE_MODEL,
        "representative_seed": REPRESENTATIVE_SEED,
        "population_sd_ddof": 0,
        "figure_count": len(manifest),
        "figures": manifest,
    }
    manifest_path = figure_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the formal shared-spectral Gaussian-wavepacket experiment"
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    for item in generate_all(args.results_root, args.raw_root, args.figure_dir):
        print(f"{item['id']}: {item['source_rows']} source rows")


if __name__ == "__main__":
    main()
