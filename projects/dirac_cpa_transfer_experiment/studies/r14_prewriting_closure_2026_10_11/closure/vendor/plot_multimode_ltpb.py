from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .multimode_config import MODE_KEYS, MODEL_LABELS, MULTIMODE_MODELS, MULTIMODE_SEEDS


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = RESULTS / "raw" / "multimode_ltpb" / "learned"
FIGURES = RESULTS / "figures" / "multimode_ltpb"

COLORS = {
    "vanilla": "#4C78A8",
    "sp_cpa": "#F58518",
    "component_polar": "#54A24B",
    "multi_branch_complex": "#B279A2",
    "multi_ltpb": "#E45756",
    "exact": "#222222",
}
MODE_LABELS = {
    "k2_plus": r"$k=2,+$",
    "k2_minus": r"$k=2,-$",
    "k5_plus": r"$k=5,+$",
    "k5_minus": r"$k=5,-$",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, name: str, source: pd.DataFrame, figure_dir: Path) -> dict[str, object]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_path = figure_dir / f"{name}_source_data.csv"
    source.to_csv(source_path, index=False)
    outputs: dict[str, str] = {}
    for suffix in ("png", "svg", "pdf"):
        path = figure_dir / f"{name}.{suffix}"
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(path, **kwargs)
        outputs[suffix] = path.relative_to(ROOT).as_posix()
    plt.close(fig)
    return {
        "id": name,
        "source_data": source_path.relative_to(ROOT).as_posix(),
        "source_rows": int(len(source)),
        "outputs": outputs,
    }


def _runs(results_root: Path) -> pd.DataFrame:
    return pd.read_csv(results_root / "multimode_ltpb_runs.csv")


def _all_time_series(raw_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model in MULTIMODE_MODELS:
        for seed in MULTIMODE_SEEDS:
            frame = pd.read_csv(raw_root / model / f"seed_{seed}" / "time_series.csv")
            frame.insert(0, "seed", seed)
            frame.insert(0, "model", model)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _all_history(raw_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model in MULTIMODE_MODELS:
        for seed in MULTIMODE_SEEDS:
            frame = pd.read_csv(raw_root / model / f"seed_{seed}" / "loss_history.csv")
            frame.insert(0, "seed", seed)
            frame.insert(0, "model", model)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def figure_primary(runs: pd.DataFrame, figure_dir: Path) -> dict[str, object]:
    source = runs.melt(
        id_vars=["model", "seed"],
        value_vars=["spinor_rel_l2_gauge_aligned", "current_rel_l2", "fidelity_mean"],
        var_name="metric",
        value_name="value",
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    specs = (
        ("spinor_rel_l2_gauge_aligned", "Spinor relative L2", False),
        ("current_rel_l2", "Current relative L2", False),
        ("fidelity_mean", r"Infidelity $1-\overline{F}$", True),
    )
    x = np.arange(len(MULTIMODE_MODELS))
    for ax, (metric, title, complement) in zip(axes, specs):
        values_for_limits: list[float] = []
        for index, model in enumerate(MULTIMODE_MODELS):
            values = source[(source.model == model) & (source.metric == metric)].value.to_numpy(float)
            shown = 1.0 - values if complement else values
            shown = np.maximum(shown, 1.0e-16)
            values_for_limits.extend(shown.tolist())
            ax.errorbar(
                index,
                np.mean(shown),
                yerr=np.std(shown, ddof=0),
                fmt="o",
                ms=5.3,
                capsize=2.5,
                color=COLORS[model],
                zorder=3,
            )
            jitter = np.linspace(-0.08, 0.08, len(shown))
            ax.scatter(index + jitter, shown, s=10, color=COLORS[model], alpha=0.5, linewidths=0)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(x, [MODEL_LABELS[m] for m in MULTIMODE_MODELS], rotation=35, ha="right")
        ax.grid(axis="y", which="both", alpha=0.18)
    axes[0].set_ylabel("Error (lower is better)")
    fig.tight_layout(w_pad=1.0)
    return _save(fig, "primary_error_comparison", source, figure_dir)


def figure_per_seed(runs: pd.DataFrame, figure_dir: Path) -> dict[str, object]:
    source = runs.melt(
        id_vars=["model", "seed"],
        value_vars=["spinor_rel_l2_gauge_aligned", "current_rel_l2"],
        var_name="metric",
        value_name="value",
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharex=True)
    for ax, metric, title in zip(
        axes,
        ("spinor_rel_l2_gauge_aligned", "current_rel_l2"),
        ("Spinor relative L2 by paired seed", "Current relative L2 by paired seed"),
    ):
        for model in MULTIMODE_MODELS:
            subset = source[(source.model == model) & (source.metric == metric)].sort_values("seed")
            ax.plot(subset.seed, subset.value, marker="o", ms=3.5, color=COLORS[model], label=MODEL_LABELS[model])
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Paired seed")
        ax.set_xticks(MULTIMODE_SEEDS)
        ax.grid(axis="y", which="both", alpha=0.18)
    axes[0].set_ylabel("Relative L2 error")
    axes[1].legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout(w_pad=1.0)
    return _save(fig, "per_seed_results", source, figure_dir)


def figure_density_current(times: pd.DataFrame, figure_dir: Path) -> dict[str, object]:
    frames: list[pd.DataFrame] = []
    for quantity, pred, exact in (
        ("density", "rho_x0_predicted", "rho_x0_exact"),
        ("current", "current_x0_predicted", "current_x0_exact"),
    ):
        frame = times[["model", "seed", "t", pred, exact]].rename(columns={pred: "predicted", exact: "exact"})
        frame.insert(3, "quantity", quantity)
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.5), sharex=True)
    for ax, quantity, ylabel in zip(axes, ("density", "current"), (r"$\rho(0,t)$", r"$j(0,t)$")):
        subset = source[source.quantity == quantity]
        exact = subset[subset.model == MULTIMODE_MODELS[0]].sort_values(["seed", "t"]).drop_duplicates("t")
        ax.plot(exact.t, exact.exact, color=COLORS["exact"], lw=1.8, label="Exact", zorder=5)
        for model in MULTIMODE_MODELS:
            model_data = subset[subset.model == model]
            for _, seed_data in model_data.groupby("seed"):
                ax.plot(seed_data.t, seed_data.predicted, color=COLORS[model], alpha=0.10, lw=0.65)
            mean = model_data.groupby("t", as_index=False).predicted.mean()
            ax.plot(mean.t, mean.predicted, color=COLORS[model], lw=1.1, label=MODEL_LABELS[model])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.16)
    axes[0].legend(frameon=False, ncol=3, loc="upper center")
    axes[-1].set_xlabel("Time")
    fig.tight_layout(h_pad=0.5)
    return _save(fig, "representative_density_current", source, figure_dir)


def figure_phases(times: pd.DataFrame, figure_dir: Path) -> dict[str, object]:
    selected = times[times.model == "multi_ltpb"]
    frames: list[pd.DataFrame] = []
    for mode in MODE_KEYS:
        frame = selected[["seed", "t", f"{mode}_phase", f"{mode}_phase_exact", f"{mode}_phase_valid"]].copy()
        frame.columns = ["seed", "t", "phase_predicted", "phase_exact", "valid"]
        frame.insert(1, "mode", mode)
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    for ax, mode in zip(axes.flat, MODE_KEYS):
        subset = source[source["mode"] == mode]
        exact = subset[subset.seed == 0]
        ax.plot(exact.t, exact.phase_exact, color=COLORS["exact"], ls="--", lw=1.6, label="Exact")
        for _, seed_data in subset.groupby("seed"):
            ax.plot(seed_data.t, seed_data.phase_predicted, color=COLORS["multi_ltpb"], alpha=0.22, lw=0.75)
        pivot = subset.pivot(index="t", columns="seed", values="phase_predicted")
        ax.plot(pivot.index, pivot.mean(axis=1), color=COLORS["multi_ltpb"], lw=1.6, label="Multi-LTPB mean")
        ax.set_title(MODE_LABELS[mode])
        ax.grid(alpha=0.16)
    axes[0, 0].set_ylabel("Unwrapped projected phase")
    axes[1, 0].set_ylabel("Unwrapped projected phase")
    axes[1, 0].set_xlabel("Time")
    axes[1, 1].set_xlabel("Time")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    return _save(fig, "learned_four_phases", source, figure_dir)


def figure_modal_errors(runs: pd.DataFrame, figure_dir: Path) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for row in runs.itertuples(index=False):
        for mode in MODE_KEYS:
            records.extend(
                (
                    {"model": row.model, "seed": row.seed, "mode": mode, "metric": "coefficient_rel_l2", "value": getattr(row, f"{mode}_coefficient_rel_l2")},
                    {"model": row.model, "seed": row.seed, "mode": mode, "metric": "occupation_rmse_abs", "value": getattr(row, f"{mode}_occupation_rmse_abs")},
                )
            )
    source = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    width = 0.15
    x = np.arange(4)
    for ax, metric, title in zip(axes, ("coefficient_rel_l2", "occupation_rmse_abs"), ("Modal coefficient error", "Modal occupation RMSE")):
        for model_index, model in enumerate(MULTIMODE_MODELS):
            subset = source[(source.model == model) & (source.metric == metric)]
            stats = subset.groupby("mode", sort=False).value.agg(["mean", lambda v: np.std(v, ddof=0)]).reindex(MODE_KEYS)
            xpos = x + (model_index - 2) * width
            ax.errorbar(xpos, stats["mean"], yerr=stats["<lambda_0>"], fmt="o", ms=3.8, capsize=2, color=COLORS[model], label=MODEL_LABELS[model])
            for mode_index, mode in enumerate(MODE_KEYS):
                points = subset[subset["mode"] == mode].value.to_numpy(float)
                ax.scatter(np.full(len(points), xpos[mode_index]), points, s=6, color=COLORS[model], alpha=0.3, linewidths=0)
        ax.set_yscale("log")
        ax.set_xticks(x, [MODE_LABELS[m] for m in MODE_KEYS])
        ax.set_title(title)
        ax.grid(axis="y", which="both", alpha=0.18)
    axes[0].set_ylabel("Error (lower is better)")
    axes[1].legend(frameon=False, fontsize=6.4)
    fig.tight_layout(w_pad=0.8)
    return _save(fig, "modal_coefficient_errors", source, figure_dir)


def figure_training(history: pd.DataFrame, figure_dir: Path) -> dict[str, object]:
    source = history[["model", "seed", "step", "pde"]].rename(columns={"pde": "pde_loss"})
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for model in MULTIMODE_MODELS:
        subset = source[source.model == model]
        for _, run in subset.groupby("seed"):
            ax.plot(run.step, np.maximum(run.pde_loss, 1e-30), color=COLORS[model], alpha=0.12, lw=0.6)
        pivot = subset.pivot(index="step", columns="seed", values="pde_loss")
        center = np.exp(np.mean(np.log(np.maximum(pivot, 1e-30)), axis=1))
        ax.plot(pivot.index, center, color=COLORS[model], label=MODEL_LABELS[model])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Adam step")
    ax.set_ylabel("PDE loss")
    ax.grid(which="both", alpha=0.16)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return _save(fig, "training_curves", source, figure_dir)


def figure_accuracy_time(runs: pd.DataFrame, figure_dir: Path) -> dict[str, object]:
    source = runs[["model", "seed", "training_seconds", "spinor_rel_l2_gauge_aligned"]].rename(columns={"spinor_rel_l2_gauge_aligned": "spinor_error"})
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for model in MULTIMODE_MODELS:
        subset = source[source.model == model]
        ax.scatter(subset.training_seconds, subset.spinor_error, s=22, color=COLORS[model], alpha=0.52)
        ax.scatter(subset.training_seconds.mean(), subset.spinor_error.mean(), s=72, marker="D", color=COLORS[model], edgecolor="white", linewidth=0.7, label=MODEL_LABELS[model])
    ax.set_yscale("log")
    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("Spinor relative L2")
    ax.grid(which="both", alpha=0.16)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return _save(fig, "accuracy_time_tradeoff", source, figure_dir)


def generate_all(results_root: Path = RESULTS, raw_root: Path = RAW, figure_dir: Path = FIGURES) -> list[dict[str, object]]:
    _style()
    runs = _runs(results_root)
    times = _all_time_series(raw_root)
    history = _all_history(raw_root)
    manifest = [
        figure_primary(runs, figure_dir),
        figure_per_seed(runs, figure_dir),
        figure_density_current(times, figure_dir),
        figure_phases(times, figure_dir),
        figure_modal_errors(runs, figure_dir),
        figure_training(history, figure_dir),
        figure_accuracy_time(runs, figure_dir),
    ]
    (figure_dir / "figure_manifest.json").write_text(
        json.dumps({"figures": manifest}, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the formal Multi-LTPB experiment")
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    for item in generate_all(args.results_root, args.raw_root, args.figure_dir):
        print(f"{item['id']}: {item['source_rows']} source rows")


if __name__ == "__main__":
    main()
