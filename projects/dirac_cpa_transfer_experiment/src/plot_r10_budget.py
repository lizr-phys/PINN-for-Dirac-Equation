from __future__ import annotations
import json
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from .r10_config import RAW, FIGURES, SEEDS, run_directory, source_directory
from .r9_config import THRESHOLDS
from .aggregate_r10_budget import STAGES
from .freeze_r7_phase_release import write_json, sha256

COLORS = dict(zip(SEEDS, ("#396F94", "#AB673A", "#766696", "#458978", "#AC5271")))
STAGE_COLORS = ("#9B6843", "#396F94")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def save(fig, name, sources, *, seeds, claim):
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES/"source_data").mkdir(exist_ok=True)
    source_paths = []
    for suffix, table in sources.items():
        path = FIGURES/"source_data"/f"{name}_{suffix}.csv"
        table.to_csv(path, index=False)
        source_paths.append(path.relative_to(FIGURES).as_posix())
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    undrawn = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            tolerance = 16*np.finfo(float).eps*max(abs(low), abs(high), abs(high-low), np.finfo(float).tiny)
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if tick.get_loc() < low-tolerance or tick.get_loc() > high+tolerance:
                    undrawn.update((id(tick.label1), id(tick.label2)))
    outside = []
    for item in fig.findobj(Text):
        if id(item) not in undrawn and item.get_visible() and item.get_text().strip():
            box = item.get_window_extent(renderer)
            if box.x0 < -2 or box.y0 < -2 or box.x1 > width+2 or box.y1 > height+2:
                outside.append(item.get_text())
    exports = {}
    for suffix in ("png", "svg", "pdf"):
        path = FIGURES/f"{name}.{suffix}"
        fig.savefig(path, dpi=300)
        exports[suffix] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    plt.close(fig)
    return dict(figure=name, claim=claim, seeds=list(seeds), source_csv=source_paths,
        exports=exports, bounding_box_pass=not outside, text_outside_canvas=outside,
        svg_editable_text="<text" in (FIGURES/f"{name}.svg").read_text(encoding="utf-8"),
        statistics="paired ancestries, n=5; population SD ddof=0" if len(seeds) == 5 else "fixed seed300 dynamics",
        backend="Python/matplotlib", undrawn_out_of_view_tick_labels_excluded=len(undrawn))


def letters(axes):
    for letter, ax in zip(string.ascii_lowercase, np.asarray(axes).ravel()):
        ax.text(-.055, 1.035, letter, transform=ax.transAxes, weight="bold", fontsize=11)


def diagnostic_table(stage, seed, filename):
    directory = RAW/"r9_source_diagnostics"/f"seed_{seed}" if stage == STAGES[0] else run_directory(seed)
    frame = pd.read_csv(directory/filename)
    frame["stage"], frame["seed"] = stage, seed
    return frame


def make_figures():
    frame = pd.read_csv(RAW/"per_seed_metrics.csv")
    qa = []
    titles = ["Spinor relative L2", "Current relative L2", "Mean fidelity",
              "Modal population relative L2", "Maximum norm drift"]
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.8), layout="constrained")
    for ax, key, title in zip(axes.ravel(), THRESHOLDS, titles):
        for seed in SEEDS:
            values = frame[frame.seed == seed].set_index("stage").loc[list(STAGES), key].to_numpy()
            ax.plot([0, 1], values, "o-", color=COLORS[seed], lw=1, ms=4, label=str(seed))
        for index, stage in enumerate(STAGES):
            v = frame[frame.stage == stage][key]
            ax.errorbar(index+.065, v.mean(), yerr=v.std(ddof=0), fmt="_", color="black", capsize=4)
        ax.axhline(THRESHOLDS[key], ls="--", color="black", lw=.8)
        ax.set(xticks=[0, 1], xticklabels=["R9 20000", "R10 26000"], title=title, xlim=(-.18, 1.2))
        ax.grid(axis="y", alpha=.15)
    axes[1, 2].axis("off")
    axes[1, 2].legend(*axes[0, 0].get_legend_handles_labels(), title="Same training ancestry", loc="upper left")
    axes[1, 2].text(.05, .35, "All five paired ancestries\nBlack: mean +/- population SD\nDashed: fixed absolute criterion\nExtra coupled updates: 6000\nCarrier unchanged; total budgets differ", va="top")
    letters(axes.ravel()[:5])
    qa.append(save(fig, "paired_primary_metrics", {"values": frame[["stage", "seed"]+list(THRESHOLDS)]},
        seeds=SEEDS, claim="All five fixed endpoint pairs; conditional continuation and absolute criteria are separate."))

    balances = pd.concat([diagnostic_table(stage, seed, "norm_balance.csv") for stage in STAGES for seed in SEEDS], ignore_index=True)
    terms = pd.concat([diagnostic_table(stage, seed, "residual_decomposition.csv") for stage in STAGES for seed in SEEDS], ignore_index=True)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), layout="constrained")
    for ax, seed in zip(axes.ravel(), SEEDS):
        for stage, linestyle in zip(STAGES, ("--", "-")):
            part = balances[(balances.stage == stage)&(balances.seed == seed)]
            ax.plot(part.t, part.norm_change_from_initial, ls=linestyle, color=COLORS[seed], label=stage)
            ax.plot(part.t, part.residual_projected_integral, ls=":", lw=.9, color="black", alpha=.7)
        ax.axhline(.02, ls="--", color="gray", lw=.7)
        ax.axhline(-.02, ls="--", color="gray", lw=.7)
        ax.set(title=f"Seed {seed}", xlabel="t", ylabel="N(t) - N(0)")
    axes[1, 2].axis("off")
    axes[1, 2].legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left")
    axes[1, 2].text(.04, .58, "Dotted black: integral of 2 Im <psi,R>\nGray lines: +/- 0.02\n4001-point raw predictions\nNo prediction normalization\nNumerical quadrature; no continuous certificate", va="top", fontsize=8.5)
    letters(axes.ravel()[:5])
    qa.append(save(fig, "norm_balance_all_seeds", {"balance": balances}, seeds=SEEDS,
        claim="Residual projection reconstructs norm change; finite-grid integration is a numerical diagnostic."))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for stage, linestyle in zip(STAGES, ("--", "-")):
        for seed in SEEDS:
            part = terms[(terms.stage == stage)&(terms.seed == seed)]
            axes[0, 0].plot(part.t, part.envelope_tangent_norm_rate, ls=linestyle, color=COLORS[seed], lw=1)
            axes[0, 1].plot(part.t, part.carrier_defect_norm_rate, ls=linestyle, color=COLORS[seed], lw=.7)
            axes[1, 0].plot(part.t, part.coupling_global_norm_rate, ls=linestyle, color=COLORS[seed], lw=.7)
            balance = balances[(balances.stage == stage)&(balances.seed == seed)]
            axes[1, 1].plot(balance.t, balance.numerical_integration_discrepancy, ls=linestyle, color=COLORS[seed], lw=1, label=str(seed) if stage == STAGES[1] else None)
    for ax, title, label in zip(axes.ravel(),
        ("Envelope contribution", "Carrier defect direct contribution", "Hermitian coupling contribution", "Norm balance quadrature discrepancy"),
        ("dN/dt", "2 Im <psi,Gd>", "-2 Im <psi,Vpsi>", "N(t)-N(0)-integral")):
        ax.set(title=title, xlabel="t", ylabel=label)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[1, 1].legend(ncol=3, title="Seed; dashed R9, solid R10", fontsize=7.5)
    letters(axes)
    qa.append(save(fig, "residual_decomposition", {"decomposition": terms, "balance": balances}, seeds=SEEDS,
        claim="Unitary carrier defects and Hermitian coupling have no direct global norm source; no pretraining causal ablation."))

    with np.load(source_directory(300)/"dense_grid.npz") as old, np.load(run_directory(300)/"dense_grid.npz") as new:
        t, x = new["t"], new["x"]
        density = [new["density_reference"], old["density_predicted"], new["density_predicted"]]
        current = [new["current_reference"], old["current_predicted"], new["current_predicted"]]
        fig, axes = plt.subplots(3, 3, figsize=(11, 8), layout="constrained")
        for row, fields in enumerate((density, current)):
            low, high = min(v.min() for v in fields), max(v.max() for v in fields)
            for column, (field, title) in enumerate(zip(fields, ("Reference", "R9 A: 20000", "R10 A: 26000"))):
                image = axes[row, column].pcolormesh(x, t, field, shading="auto", vmin=low, vmax=high,
                    cmap="viridis" if row == 0 else "RdBu_r", rasterized=True)
                axes[row, column].set(xlabel="x", ylabel="t", title=title+(": density" if row == 0 else ": current"))
            fig.colorbar(image, ax=axes[row].tolist(), shrink=.82)
        series = []
        for stage, color, getter in zip(STAGES, STAGE_COLORS, (source_directory, run_directory)):
            part = pd.read_csv(getter(300)/"time_series.csv")
            part["stage"] = stage
            series.append(part)
            for ax, key in zip(axes[2], ("spinor_aligned_error", "norm_predicted", "coherence_relative_error")):
                ax.plot(part.t, part[key], label=stage, color=color)
        for ax, title in zip(axes[2], ("Spinor error: one constant phase", "Unnormalized total norm", "Reference-active coherence error")):
            ax.set(xlabel="t", title=title)
        axes[2, 1].axhline(1, color="black", ls=":", lw=.8)
        axes[2, 0].legend()
        letters(axes)
        ti, xi = np.meshgrid(t, x, indexing="ij")
        grid = pd.DataFrame(dict(t=ti.ravel(), x=xi.ravel(),
            density_reference=density[0].ravel(), density_r9=density[1].ravel(), density_r10=density[2].ravel(),
            current_reference=current[0].ravel(), current_r9=current[1].ravel(), current_r10=current[2].ravel()))
    qa.append(save(fig, "dynamics_seed300", {"physical_grid": grid, "series": pd.concat(series, ignore_index=True)},
        seeds=(300,), claim="Fixed seed300 physical fields and errors; reference and both stages use common scales."))

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 5.8), layout="constrained")
    branch_sources = []
    for stage, linestyle, getter in zip(STAGES, ("--", "-"), (source_directory, run_directory)):
        for seed in SEEDS:
            part = pd.read_csv(getter(seed)/"time_series.csv")
            part["stage"], part["seed"] = stage, seed
            branch_sources.append(part)
            for row, branch in enumerate(("plus", "minus")):
                for column, quantity in enumerate(("center", "width")):
                    error = part[f"{branch}_{quantity}_predicted"]-part[f"{branch}_{quantity}_reference"]
                    axes[row, column].plot(part.t, error, ls=linestyle, color=COLORS[seed], lw=1,
                        label=str(seed) if row == column == 0 and stage == STAGES[1] else None)
                    axes[row, column].set(xlabel="t", ylabel="Prediction - reference",
                        title=f"{branch} branch {quantity} trajectory error")
    axes[0, 0].legend(ncol=3, title="Seed; dashed R9, solid R10", fontsize=7.5)
    letters(axes)
    qa.append(save(fig, "branch_trajectories", {"series": pd.concat(branch_sources, ignore_index=True)},
        seeds=SEEDS, claim="Both free-branch center and width trajectories relative to the same coupled reference; all seeds."))

    costs = pd.read_csv(RAW/"cost_and_residual_work.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    xpos = np.arange(5)
    bottom = np.zeros(5)
    for key, color, label in (("inherited_carrier_training_seconds", "#8B8B8B", "Inherited carrier"),
        ("inherited_coupled_training_seconds", "#AE8E75", "Inherited coupled"),
        ("additional_training_seconds", "#396F94", "New coupled")):
        values = costs[key].to_numpy()/60
        axes[0, 0].bar(xpos, values, bottom=bottom, color=color, label=label)
        bottom += values
    axes[0, 0].set(xticks=xpos, xticklabels=SEEDS, ylabel="Measured training time (min)", title="Inherited and additional training cost")
    axes[0, 0].legend(fontsize=7.5)
    axes[0, 1].bar([0, 1, 2], [107.328, 24.768, 132.096], color=["#AE8E75", "#396F94", "#666666"])
    axes[0, 1].set(xticks=[0, 1, 2], xticklabels=["Inherited", "Additional", "Cumulative"],
        ylabel="Spinor residual points (million/run)", title="Work: 20000 + 6000 optimizer steps")
    for seed in SEEDS:
        part = frame[frame.seed == seed].set_index("stage").loc[list(STAGES)]
        axes[1, 0].plot(part.residual_integral_fine, part.max_norm_drift, "o-", color=COLORS[seed], label=str(seed))
        axes[1, 1].plot([0, 1], part.full_physical_residual_integral, "o-", color=COLORS[seed])
    axes[1, 0].axhline(.02, color="black", ls="--", lw=.8)
    axes[1, 0].set(xlabel="Numerical integral ||R|| dt", ylabel="Maximum norm drift", title="Endpoint changes; no causal regression")
    axes[1, 0].legend(ncol=3, fontsize=7.5)
    axes[1, 1].set(xticks=[0, 1], xticklabels=["R9 20000", "R10 26000"],
        ylabel="Numerical integral ||R_full|| dt", title="Full residual includes signed exterior modes")
    letters(axes)
    qa.append(save(fig, "budget_and_residuals", {"costs": costs,
        "diagnostics": frame[["stage", "seed", "max_norm_drift", "residual_integral_fine", "full_physical_residual_integral"]]},
        seeds=SEEDS, claim="Total optimizer steps and residual work differ; inherited and new measured costs are reported separately."))
    result = dict(status="PASS" if all(q["bounding_box_pass"] and q["svg_editable_text"] for q in qa) else "FAIL",
        figures=qa, visual_inspection_pending=True, figure_count=len(qa), dynamics_seed=300,
        paired_ancestries=list(SEEDS), population_sd_ddof=0)
    write_json(FIGURES/"FIGURE_QA.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(make_figures(), indent=2))
