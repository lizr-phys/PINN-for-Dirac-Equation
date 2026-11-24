from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from .config import ExperimentConfig


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False
plt.rcParams["savefig.facecolor"] = "white"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
FIGURE_ROOT = RESULTS_ROOT / "figures"
SOURCE_ROOT = RESULTS_ROOT / "raw" / "figure_source_data"
CFG = ExperimentConfig()

METHODS = ["vanilla", "sp_cpa", "sp_cpa_conservative", "two_branch"]
METHOD_LABELS = {
    "vanilla": "Vanilla",
    "sp_cpa": "SP-CPA",
    "sp_cpa_conservative": "SP-CPA-Cons.",
    "two_branch": "Two-Branch oracle",
}
METHOD_COLORS = {
    "vanilla": "#484878",
    "sp_cpa": "#7884B4",
    "sp_cpa_conservative": "#B64342",
    "two_branch": "#42949E",
}
METHOD_MARKERS = {
    "vanilla": "o",
    "sp_cpa": "s",
    "sp_cpa_conservative": "D",
    "two_branch": "^",
}


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_figure(fig: plt.Figure, name: str) -> list[str]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for extension, dpi in (("svg", 300), ("pdf", 300), ("png", 300)):
        path = FIGURE_ROOT / f"{name}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def main_run_dir(case: str, model: str, seed: int) -> Path:
    return RESULTS_ROOT / "raw" / "main" / f"case_{case}" / model / f"seed_{seed}"


def sensitivity_run_dir(lambda_struct: float) -> Path:
    label = f"{lambda_struct:g}".replace(".", "p")
    return (
        RESULTS_ROOT
        / "raw"
        / "sensitivity"
        / "case_M"
        / "sp_cpa_conservative"
        / f"lambda_{label}"
        / "seed_0"
    )


def require_inputs() -> pd.DataFrame:
    summary_path = RESULTS_ROOT / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Run aggregation before plotting: results/summary.csv is missing")
    summary = pd.read_csv(summary_path)
    main = summary[summary["experiment"] == "main"]
    expected = 20
    if len(main) != expected:
        raise RuntimeError(f"Expected {expected} main runs, found {len(main)}")
    return summary


def plot_main_comparison(summary: pd.DataFrame) -> dict:
    main = summary[summary["experiment"] == "main"].copy()
    main["infidelity_mean"] = np.maximum(1.0 - main["fidelity_mean"], 1.0e-16)
    metrics = [
        ("spinor_rel_l2", "Spinor relative L2"),
        ("rho_rel_l2", "Density relative L2"),
        ("current_rel_l2", "Current relative L2"),
        ("infidelity_mean", "Mean infidelity (1 - F)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.4), constrained_layout=True)
    rng = np.random.default_rng(20260830)
    letters = iter("abcdefgh")
    for row_index, case in enumerate(("P", "M")):
        for column_index, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_index, column_index]
            for method_index, method in enumerate(METHODS):
                values = main[(main["case"] == case) & (main["model"] == method)][
                    metric
                ].to_numpy(float)
                n = len(values)
                jitter = np.zeros(n) if n == 1 else rng.uniform(-0.08, 0.08, n)
                ax.scatter(
                    np.full(n, method_index) + jitter,
                    values,
                    s=13,
                    marker=METHOD_MARKERS[method],
                    color=METHOD_COLORS[method],
                    alpha=0.75,
                    edgecolor="white",
                    linewidth=0.3,
                    zorder=3,
                )
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=0))
                lower = min(std, 0.9 * mean)
                ax.errorbar(
                    method_index,
                    mean,
                    yerr=np.array([[lower], [std]]),
                    fmt="_",
                    markersize=10,
                    markeredgewidth=1.5,
                    color=METHOD_COLORS[method],
                    capsize=2.5,
                    linewidth=1.0,
                    zorder=4,
                )
            ax.set_yscale("log")
            ax.grid(axis="y", which="both", color="#D8D8D8", linewidth=0.5, alpha=0.7)
            ax.set_xticks(range(len(METHODS)))
            if row_index == 1:
                ax.set_xticklabels(
                    [METHOD_LABELS[method] for method in METHODS], rotation=25, ha="right"
                )
            else:
                ax.set_xticklabels([])
            ax.set_ylabel(ylabel if column_index == 0 else "")
            ax.set_title(("Case P: positive branch" if case == "P" else "Case M: mixed branches") if column_index == 0 else ylabel)
            if column_index == 0:
                ax.text(
                    0.01,
                    0.03,
                    "n=3 for learned models; n=1 oracle",
                    transform=ax.transAxes,
                    fontsize=6,
                    color="#606060",
                )
            add_panel_label(ax, next(letters))
    fig.suptitle("Dense-grid errors: points are seeds; crossbars show mean +/- population SD", fontsize=9)
    return {
        "files": save_figure(fig, "01_main_metric_comparison"),
        "source": str((RESULTS_ROOT / "summary.csv").resolve()),
        "conclusion": "Compares Case P and Case M accuracy under the fixed training budget.",
    }


def _collect_case_m_time_series() -> pd.DataFrame:
    pieces = []
    for method in METHODS:
        frame = pd.read_csv(main_run_dir("M", method, 0) / "time_series.csv")
        frame.insert(0, "model", method)
        pieces.append(frame)
    combined = pd.concat(pieces, ignore_index=True)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(SOURCE_ROOT / "case_M_seed0_time_series.csv", index=False)
    return combined


def plot_current(time_series: pd.DataFrame) -> dict:
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    reference = time_series[time_series["model"] == METHODS[0]]
    ax.plot(
        reference["t"],
        reference["current_x0_exact"],
        color="#272727",
        linewidth=2.0,
        label=r"Exact ($\omega=2E$)",
        zorder=5,
    )
    for method in METHODS:
        frame = time_series[time_series["model"] == method]
        ax.plot(
            frame["t"],
            frame["current_x0_predicted"],
            color=METHOD_COLORS[method],
            linewidth=1.05,
            alpha=0.9,
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("t")
    ax.set_ylabel(r"Probability current $j(0,t)$")
    ax.grid(color="#D8D8D8", linewidth=0.5)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    add_panel_label(ax, "a")
    return {
        "files": save_figure(fig, "02_case_M_current_x0"),
        "source": str((SOURCE_ROOT / "case_M_seed0_time_series.csv").resolve()),
        "conclusion": "Tests whether each model recovers the mixed-branch 2E current oscillation.",
    }


def plot_relative_phase(time_series: pd.DataFrame) -> dict:
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    reference = time_series[time_series["model"] == METHODS[0]]
    exact = np.unwrap(reference["relative_phase_x0_exact"].to_numpy())
    ax.plot(reference["t"], exact, color="#272727", linewidth=2.0, label="Exact", zorder=5)
    for method in METHODS:
        frame = time_series[time_series["model"] == method]
        phase = np.unwrap(frame["relative_phase_x0_predicted"].to_numpy())
        ax.plot(
            frame["t"],
            phase,
            color=METHOD_COLORS[method],
            linewidth=1.05,
            alpha=0.9,
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("t")
    ax.set_ylabel(r"Unwrapped $\arg(\psi_1^*\psi_2)$ at $x=0$ (rad)")
    ax.grid(color="#D8D8D8", linewidth=0.5)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    add_panel_label(ax, "a")
    return {
        "files": save_figure(fig, "03_case_M_relative_phase_x0"),
        "source": str((SOURCE_ROOT / "case_M_seed0_time_series.csv").resolve()),
        "conclusion": "Directly checks the spinor component relative phase missed by conservation laws.",
    }


def plot_error_maps() -> dict:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    x = t = None
    for method in METHODS:
        with np.load(main_run_dir("M", method, 0) / "dense_grid.npz") as data:
            x = data["x"]
            t = data["t"]
            arrays[method] = {
                "rho": np.abs(data["rho_predicted"] - data["rho_exact"]),
                "current": np.abs(data["current_predicted"] - data["current_exact"]),
                "psi1": np.abs(data["psi_predicted"][:, :, 0] - data["psi_exact"][:, :, 0]),
                "psi2": np.abs(data["psi_predicted"][:, :, 1] - data["psi_exact"][:, :, 1]),
            }
    assert x is not None and t is not None
    metric_rows = [
        ("rho", r"$|\Delta\rho|$"),
        ("current", r"$|\Delta j|$"),
        ("psi1", r"$|\Delta\psi_1|$"),
        ("psi2", r"$|\Delta\psi_2|$"),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.0), constrained_layout=True, sharex=True, sharey=True)
    letters = iter("abcdefghijklmnop")
    for row_index, (metric, label) in enumerate(metric_rows):
        vmax = max(float(np.max(arrays[method][metric])) for method in METHODS)
        vmin = max(vmax * 1.0e-5, 1.0e-12)
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10.0))
        image = None
        for column_index, method in enumerate(METHODS):
            ax = axes[row_index, column_index]
            image = ax.imshow(
                arrays[method][metric],
                origin="lower",
                aspect="auto",
                extent=(x[0], x[-1] + CFG.length / CFG.test_x_points, t[0], t[-1]),
                cmap="magma",
                norm=norm,
                interpolation="nearest",
            )
            if row_index == 0:
                ax.set_title(METHOD_LABELS[method])
            if column_index == 0:
                ax.set_ylabel(f"{label}\nt")
            if row_index == len(metric_rows) - 1:
                ax.set_xlabel("x")
            add_panel_label(ax, next(letters))
        assert image is not None
        colorbar = fig.colorbar(image, ax=axes[row_index, :], fraction=0.015, pad=0.01)
        colorbar.ax.tick_params(labelsize=6)
    return {
        "files": save_figure(fig, "04_case_M_spacetime_error_maps"),
        "source": "Per-run dense_grid.npz files under results/raw/main/case_M",
        "conclusion": "Locates density, current, and component errors over the full x-t domain using shared per-metric scales.",
    }


def plot_conservation(time_series: pd.DataFrame) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    quantities = [
        ("continuity_rms", r"Continuity RMS / $(E\rho_0)$", lambda frame: frame["continuity_rms"] / (CFG.energy * CFG.reference_density)),
        ("norm", r"$|N(t)-1|$", lambda frame: np.abs(frame["norm_predicted"] - 1.0)),
        ("energy", r"$|E_D(t)-E_D(0)|$", lambda frame: np.abs(frame["energy_predicted"] - frame["energy_exact"])),
    ]
    for index, (_, ylabel, transform) in enumerate(quantities):
        ax = axes[index]
        for method in METHODS:
            frame = time_series[time_series["model"] == method]
            values = np.maximum(np.asarray(transform(frame), dtype=float), 1.0e-14)
            ax.semilogy(
                frame["t"],
                values,
                color=METHOD_COLORS[method],
                linewidth=1.1,
                label=METHOD_LABELS[method],
            )
        ax.set_xlabel("t")
        ax.set_ylabel(ylabel)
        ax.grid(which="both", color="#D8D8D8", linewidth=0.5)
        add_panel_label(ax, chr(ord("a") + index))
    axes[0].legend(ncol=2, fontsize=6)
    return {
        "files": save_figure(fig, "05_case_M_conservation_diagnostics"),
        "source": str((SOURCE_ROOT / "case_M_seed0_time_series.csv").resolve()),
        "conclusion": "Separates conservation quality from wavefunction and current accuracy.",
    }


def plot_losses() -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7), constrained_layout=True)
    source_rows = []
    for row_index, case in enumerate(("P", "M")):
        ax_total = axes[row_index, 0]
        for method in METHODS:
            history = pd.read_csv(main_run_dir(case, method, 0) / "loss_history.csv")
            source = history.copy()
            source.insert(0, "model", method)
            source.insert(0, "case", case)
            source_rows.append(source)
            ax_total.semilogy(
                history["step"],
                history["total"],
                color=METHOD_COLORS[method],
                linewidth=1.1,
                label=METHOD_LABELS[method],
            )
        ax_total.set_title(f"Case {case}: total loss (seed 0)")
        ax_total.set_xlabel("Adam step")
        ax_total.set_ylabel("Loss")
        ax_total.grid(which="both", color="#D8D8D8", linewidth=0.5)
        add_panel_label(ax_total, "a" if row_index == 0 else "c")

        ax_components = axes[row_index, 1]
        history = pd.read_csv(
            main_run_dir(case, "sp_cpa_conservative", 0) / "loss_history.csv"
        )
        component_styles = {
            "pde": ("PDE", "#484878", "-"),
            "continuity": ("Continuity", "#B64342", "--"),
            "norm": ("Norm", "#42949E", "-."),
            "energy": ("Energy", "#9A4D8E", ":"),
        }
        for column, (label, color, linestyle) in component_styles.items():
            ax_components.semilogy(
                history["step"],
                np.maximum(history[column], 1.0e-16),
                color=color,
                linestyle=linestyle,
                linewidth=1.1,
                label=label,
            )
        ax_components.set_title(f"Case {case}: conservative components")
        ax_components.set_xlabel("Adam step")
        ax_components.set_ylabel("Normalized loss")
        ax_components.grid(which="both", color="#D8D8D8", linewidth=0.5)
        add_panel_label(ax_components, "b" if row_index == 0 else "d")
    axes[0, 0].legend(ncol=2, fontsize=6)
    axes[0, 1].legend(ncol=2, fontsize=6)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    pd.concat(source_rows, ignore_index=True).to_csv(
        SOURCE_ROOT / "seed0_loss_histories.csv", index=False
    )
    return {
        "files": save_figure(fig, "06_training_loss_curves"),
        "source": str((SOURCE_ROOT / "seed0_loss_histories.csv").resolve()),
        "conclusion": "Shows optimization trajectories and the conservative loss decomposition without treating loss as the accuracy metric.",
    }


def plot_sensitivity(summary: pd.DataFrame) -> dict:
    sensitivity = summary[summary["experiment"] == "sensitivity"].copy()
    sensitivity = sensitivity.sort_values("lambda_struct")
    sensitivity["infidelity_mean"] = np.maximum(1.0 - sensitivity["fidelity_mean"], 1.0e-16)
    sensitivity.to_csv(SOURCE_ROOT / "lambda_sensitivity_metrics.csv", index=False)
    metrics = [
        ("spinor_rel_l2", "Spinor rel. L2"),
        ("current_rel_l2", "Current rel. L2"),
        ("infidelity_mean", "Mean infidelity"),
        ("continuity_normalized_rms", "Normalized continuity RMS"),
        ("max_norm_drift", "Max norm drift"),
        ("max_abs_energy_drift", "Max abs. energy drift"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), constrained_layout=True)
    for index, (metric, ylabel) in enumerate(metrics):
        ax = axes.flat[index]
        ax.plot(
            sensitivity["lambda_struct"],
            np.maximum(sensitivity[metric], 1.0e-16),
            color=METHOD_COLORS["sp_cpa_conservative"],
            marker="o",
            linewidth=1.4,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([0.1, 1.0, 10.0], labels=["0.1", "1", "10"])
        ax.set_xlabel(r"$\lambda_{struct}$")
        ax.set_ylabel(ylabel)
        ax.grid(which="both", color="#D8D8D8", linewidth=0.5)
        add_panel_label(ax, chr(ord("a") + index))
    fig.suptitle("Case M sensitivity (fixed seed 0; lambda=1 reuses the main run)", fontsize=9)
    return {
        "files": save_figure(fig, "07_lambda_struct_sensitivity"),
        "source": str((SOURCE_ROOT / "lambda_sensitivity_metrics.csv").resolve()),
        "conclusion": "Checks whether the conservation-versus-wavefunction trade-off is robust to the predeclared structure weight range.",
    }


def plot_gradient_cosines() -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True, sharey=True)
    source_rows = []
    conflict_summary = []
    for case_index, case in enumerate(("P", "M")):
        series = []
        for seed in (0, 1, 2):
            history = pd.read_csv(
                main_run_dir(case, "sp_cpa_conservative", seed) / "loss_history.csv"
            )
            diagnostic = history[np.isfinite(history["gradient_cosine"])][
                ["step", "gradient_cosine"]
            ].copy()
            diagnostic["case"] = case
            diagnostic["seed"] = seed
            source_rows.append(diagnostic)
            series.append(diagnostic.set_index("step")["gradient_cosine"])
        aligned = pd.concat(series, axis=1)
        aligned.columns = [0, 1, 2]
        mean = aligned.mean(axis=1)
        std = aligned.std(axis=1, ddof=0)
        axes[case_index].plot(mean.index, mean, color="#B64342", linewidth=1.5)
        axes[case_index].fill_between(
            mean.index.to_numpy(float),
            (mean - std).to_numpy(float),
            (mean + std).to_numpy(float),
            color="#E9A6A1",
            alpha=0.45,
            linewidth=0,
        )
        axes[case_index].axhline(0.0, color="#606060", linestyle="--", linewidth=0.8)
        axes[case_index].set_xlabel("Adam step")
        axes[case_index].set_title(f"Case {case}")
        axes[case_index].grid(color="#D8D8D8", linewidth=0.5)
        add_panel_label(axes[case_index], chr(ord("a") + case_index))
        all_values = aligned.to_numpy().ravel()
        all_values = all_values[np.isfinite(all_values)]
        conflict_summary.append(
            {
                "case": case,
                "diagnostic_points": int(all_values.size),
                "gradient_conflict_fraction": float(np.mean(all_values < 0.0)),
                "gradient_cosine_mean": float(np.mean(all_values)),
            }
        )
    axes[0].set_ylabel("cos(gradient PDE, gradient structure)")
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    pd.concat(source_rows, ignore_index=True).to_csv(
        SOURCE_ROOT / "gradient_cosine_diagnostics.csv", index=False
    )
    pd.DataFrame(conflict_summary).to_csv(
        RESULTS_ROOT / "gradient_conflict_summary.csv", index=False
    )
    return {
        "files": save_figure(fig, "08_gradient_cosine_similarity"),
        "source": str((SOURCE_ROOT / "gradient_cosine_diagnostics.csv").resolve()),
        "conclusion": "Quantifies alignment or conflict between PDE and conservation gradients over training.",
    }


def main() -> None:
    summary = require_inputs()
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    manifest["01_main_metric_comparison"] = plot_main_comparison(summary)
    time_series = _collect_case_m_time_series()
    manifest["02_case_M_current_x0"] = plot_current(time_series)
    manifest["03_case_M_relative_phase_x0"] = plot_relative_phase(time_series)
    manifest["04_case_M_spacetime_error_maps"] = plot_error_maps()
    manifest["05_case_M_conservation_diagnostics"] = plot_conservation(time_series)
    manifest["06_training_loss_curves"] = plot_losses()
    manifest["07_lambda_struct_sensitivity"] = plot_sensitivity(summary)
    manifest["08_gradient_cosine_similarity"] = plot_gradient_cosines()
    with (FIGURE_ROOT / "figure_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"Generated {len(manifest)} figure groups in {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
