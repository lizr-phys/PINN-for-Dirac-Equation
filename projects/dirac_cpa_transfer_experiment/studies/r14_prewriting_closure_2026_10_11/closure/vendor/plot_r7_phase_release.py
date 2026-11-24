from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .freeze_r7_phase_release import ROOT, RAW, sha256, write_json
from .r7_config import EXPERIMENT, MODELS, TRAINING_MODES

FIGURES = ROOT/"results/figures"/EXPERIMENT
COLORS = dict(complex="#757575", original_ltpb="#4C78A8", phase_release_ltpb="#D58F39")
LABELS = dict(complex="Complex", original_ltpb="Original", phase_release_ltpb="Release")
GROUPS = [(m, mode) for mode in TRAINING_MODES for m in MODELS]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                    "axes.spines.top": False, "axes.spines.right": False,
                    "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42})


def read_dense(model, mode, stage="main"):
    return np.load(RAW/stage/model/mode/"seed_0/dense_grid.npz")


def legend_handles():
    return [Line2D([], [], color=COLORS[m], label=LABELS[m]) for m in MODELS]


def finish(fig, name, sources, caption, *, seed_scope):
    FIGURES.mkdir(parents=True, exist_ok=True)
    source_root = FIGURES/"source_data"
    source_root.mkdir(exist_ok=True)
    source_paths = []
    for suffix, data in sources.items():
        path = source_root/f"{name}_{suffix}.csv"
        data.to_csv(path, index=False)
        source_paths.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha256(path), rows=len(data)))
    panel = 0
    for axis in fig.axes:
        if axis.axison and axis.get_label() != "<colorbar>":
            axis.text(-.13, 1.025, chr(ord("a")+panel), transform=axis.transAxes, fontweight="bold", fontsize=10,
                      ha="left", va="bottom")
            panel += 1
    fig.canvas.draw()
    exports = []
    for extension in ("png", "svg", "pdf"):
        path = FIGURES/f"{name}.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        exports.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha256(path)))
    plt.close(fig)
    return dict(figure_id=name, caption=caption, seed_scope=seed_scope,
                archetype="quantitative grid", source_data=source_paths, exports=exports,
                statistics="all paired seeds; mean +/- population SD (ddof=0); no significance inference")


def primary(runs, *, name="all_seed_primary_metrics"):
    fields = [("spinor_rel_l2_gauge_aligned", "Spinor relative L2", .1),
              ("current_rel_l2", "Current relative L2", .15),
              ("fidelity_mean", "Mean fidelity", .98),
              ("modal_population_rel_l2", "Modal population relative L2", .1),
              ("max_norm_drift", "Maximum norm drift", .02)]
    groups = GROUPS if len(runs) == 30 else [(m, runs.training_mode.iloc[0]) for m in MODELS]
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.0), layout="constrained")
    for ax, (key, label, threshold) in zip(axes.flat, fields):
        for index, (model, mode) in enumerate(groups):
            rows = runs[(runs.model == model)&(runs.training_mode == mode)].sort_values("seed")
            values = rows[key].to_numpy()
            offsets = np.linspace(-.14, .14, len(values))
            ax.scatter(index+offsets, values, color=COLORS[model], marker="o" if mode == "ordinary" else "^", s=16, alpha=.75)
            ax.errorbar(index, values.mean(), yerr=values.std(ddof=0), color="#222222", fmt="_", capsize=3, lw=1)
        ax.axhline(threshold, color="#B45252", ls=":", lw=1)
        if key != "fidelity_mean":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.03)
        ax.set_title(label)
        ax.set_xticks(range(len(groups)), [LABELS[m]+"\n"+mode for m, mode in groups], rotation=25, fontsize=7)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, .9, "All 5 paired seeds per configuration\nBlack: mean ± population SD\nDotted red: absolute criterion\nMain endpoint fixed at step 5000", va="top")
    return finish(fig, name, {"runs": runs}, "All seeds and absolute accuracy conditions; relative improvement alone is not success.", seed_scope="all five seeds")


def checkpoints(table):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    for ax, key, label in zip(axes, ("spinor_rel_l2_gauge_aligned", "current_rel_l2", "fidelity_mean"),
                              ("Spinor relative L2", "Current relative L2", "Mean fidelity")):
        for model, mode in GROUPS:
            part = table[(table.model == model)&(table.training_mode == mode)]
            for seed, rows in part.groupby("seed"):
                ax.plot(rows.step, rows[key], color=COLORS[model], ls="-" if mode == "ordinary" else "--", alpha=.17, lw=.8)
            summary = part.groupby("step")[key].agg(["mean", lambda x: x.std(ddof=0)])
            ax.errorbar(summary.index, summary["mean"], yerr=summary.iloc[:, 1], color=COLORS[model],
                        ls="-" if mode == "ordinary" else "--", marker="o", ms=3)
        if key != "fidelity_mean": ax.set_yscale("log")
        ax.set(xlabel="Fixed checkpoint step", ylabel=label, xticks=[1000, 2500, 5000])
    fig.legend(handles=legend_handles()+[Line2D([], [], color="black", ls="--", label="causal (dashed)")], loc="outside lower center", ncol=4)
    return finish(fig, "checkpoint_accuracy", {"checkpoints": table}, "Checkpoint trajectories include every seed; no checkpoint selected by reference accuracy.", seed_scope="all five seeds")


def short_dynamics(short):
    fig, axes = plt.subplots(2, 3, figsize=(10.3, 6), layout="constrained")
    for ax, key, title in zip(axes[0], ("initial_sideband_time_derivative_rel_l2",
                          "initial_sideband_imaginary_tangent_parameter_sensitivity", "spinor_rel_l2_gauge_aligned"),
                         ("Initial sideband derivative error", "Imaginary tangent sensitivity", "Short-time spinor error")):
        for i, model in enumerate(MODELS):
            values = short[short.model == model][key].to_numpy()
            ax.scatter(i+np.linspace(-.1, .1, len(values)), values, color=COLORS[model], s=22)
            ax.errorbar(i, values.mean(), yerr=values.std(ddof=0), color="black", fmt="_", capsize=3)
        ax.set_xticks(range(3), [LABELS[m] for m in MODELS])
        ax.set_title(title)
        if key != "initial_sideband_imaginary_tangent_parameter_sensitivity": ax.set_yscale("log")
    source = []
    for model in MODELS:
        path = RAW/"short"/model/"ordinary/seed_0/early_time_series.csv"
        rows = pd.read_csv(path)
        rows["model"] = model
        source.append(rows)
        part = rows[(rows.sideband == 1)&(rows.branch == 0)]
        for ax, key in zip(axes[1], ("predicted_real", "predicted_imag", "predicted_occupation")):
            ax.plot(part.t, part[key], color=COLORS[model], label=LABELS[model])
    ref = source[0]
    ref = ref[(ref.sideband == 1)&(ref.branch == 0)]
    for ax, key, label in zip(axes[1], ("reference_real", "reference_imag", "reference_occupation"),
                               ("Upper sideband + real", "Upper sideband + imaginary", "Upper sideband + population")):
        ax.plot(ref.t, ref[key], color="black", ls=":", label="Reference")
        ax.set(xlabel="t (gate remains t/5)", title=label)
        ax.set_xscale("symlog", linthresh=1e-4)
    axes[1, -1].legend(fontsize=7)
    return finish(fig, "initial_tangent_and_short_dynamics", {"metrics": short, "early_seed0": pd.concat(source)},
                  "Local tangent capability, learned derivative accuracy, and early sideband dynamics are distinct checks.", seed_scope="all three short seeds; dynamics fixed seed 0")


def main_tangent(runs):
    fields = [("initial_sideband_time_derivative_rel_l2", "Learned initial derivative error", "log"),
              ("initial_sideband_imaginary_tangent_parameter_sensitivity", "Imaginary tangent sensitivity", "symlog"),
              ("initial_output_tangent_jacobian_min_singular_value", "Minimum output-coordinate singular value", "symlog"),
              ("last_r_output_gradient", "Last-step amplitude-output gradient norm", "log"),
              ("last_b_output_gradient", "Last-step phase/imaginary-output gradient norm", "log")]
    rows = runs.copy()
    for index, row in rows.iterrows():
        history = pd.read_csv(RAW/"main"/row.model/row.training_mode/f"seed_{row.seed}/loss_history.csv")
        rows.loc[index, "last_r_output_gradient"] = history.output_r_gradient_norm.iloc[-1]
        rows.loc[index, "last_b_output_gradient"] = history.output_b_gradient_norm.iloc[-1]
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.0), layout="constrained")
    for ax, (key, title, scale) in zip(axes.flat, fields):
        for i, (model, mode) in enumerate(GROUPS):
            values = rows[(rows.model == model)&(rows.training_mode == mode)][key].to_numpy()
            ax.scatter(i+np.linspace(-.13, .13, 5), values, color=COLORS[model], s=17,
                       marker="o" if mode == "ordinary" else "^")
            ax.errorbar(i, values.mean(), yerr=values.std(ddof=0), color="black", fmt="_", capsize=3)
        ax.set_xticks(range(6), [LABELS[m]+"\n"+mode for m, mode in GROUPS], rotation=25, fontsize=7)
        ax.set_title(title)
        ax.set_yscale(scale, **({"linthresh": 1e-6} if scale == "symlog" else {}))
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, .9, "All five main seeds\nNonzero sensitivity is not accuracy\nRank two may still be ill-conditioned\nGradients: recorded step-5000 batch", va="top")
    return finish(fig, "main_initial_tangent", {"metrics_and_gradients": rows},
                  "Initial direction availability, trained derivative error and output-gradient scales remain separate diagnostics.", seed_scope="all five main seeds")


def modal_heatmaps():
    fig, axes = plt.subplots(6, 3, figsize=(10.5, 12), layout="constrained")
    source = []
    for row, (model, mode) in enumerate(GROUPS):
        data = read_dense(model, mode)
        exact, predicted = data["modal_occupations_exact"], data["modal_occupations_predicted"]
        extent = [data["t"][0], data["t"][-1], data["mode_offsets"][0]-.5, data["mode_offsets"][-1]+.5]
        for col, values in enumerate((exact, predicted, np.abs(predicted-exact))):
            image = axes[row, col].imshow(values.T, origin="lower", aspect="auto", extent=extent,
                                           cmap="viridis" if col < 2 else "magma", vmin=0, vmax=1,
                                           interpolation="nearest")
            if col < 2: population_image = image
            if np.max(values) > 1:
                axes[row, col].text(.02, .96, f"max={np.max(values):.5g}; color clipped >1", transform=axes[row, col].transAxes,
                                   va="top", fontsize=7, color="#8B2222", bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
            axes[row, col].set_ylabel(LABELS[model]+"\n"+mode+"\nj")
            if row == 0: axes[row, col].set_title(("Reference", "Prediction", "Absolute error")[col])
            if row == 5: axes[row, col].set_xlabel("t")
        for n, t in enumerate(data["t"]):
            for index, j in enumerate(data["mode_offsets"]):
                source.append(dict(model=model, training_mode=mode, seed=0, t=t, j=j,
                                   reference=exact[n, index], predicted=predicted[n, index],
                                   absolute_error=abs(predicted[n, index]-exact[n, index])))
    fig.colorbar(population_image, ax=axes[:, :2], shrink=.55, label="Population (common 0–1 scale)")
    fig.colorbar(image, ax=axes[:, -1], shrink=.55, label="Absolute error (common 0–1 scale)")
    return finish(fig, "modal_occupation_heatmaps", {"grid": pd.DataFrame(source)},
                  "Common 0–1 scale shows whether sidebands arise; values beyond scale are explicitly annotated, not hidden; no best-seed selection.", seed_scope="fixed seed 0")


def trajectories():
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.0), layout="constrained")
    rows = []
    for mode_index, mode in enumerate(TRAINING_MODES):
        for model in MODELS:
            data = read_dense(model, mode)
            for column, offset in enumerate((0, -1, 1)):
                index = np.flatnonzero(data["mode_offsets"] == offset)[0]
                predicted, exact = data["modal_occupations_predicted"][:, index], data["modal_occupations_exact"][:, index]
                axes[mode_index, column].plot(data["t"], predicted, color=COLORS[model], label=LABELS[model])
                if model == MODELS[0]: axes[mode_index, column].plot(data["t"], exact, color="black", ls=":", label="Reference")
                rows += [dict(model=model, training_mode=mode, seed=0, t=t, j=offset, predicted=p, reference=e)
                         for t, p, e in zip(data["t"], predicted, exact)]
                axes[mode_index, column].set(xlabel="t", title=f"{mode}: j={offset:+d}")
        axes[mode_index, 0].set_ylabel("Modal population")
    axes[0, -1].legend(fontsize=7)
    return finish(fig, "sideband_trajectories", {"trajectories": pd.DataFrame(rows)},
                  "Center and both first sidebands are checked against the same reference.", seed_scope="fixed seed 0")


def coherence_current(runs):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6), layout="constrained")
    for ax, key, title in zip(axes[0], ("active_coherence_rel_l2", "active_branch_phase_coverage"),
                               ("Active-channel coherence error", "Predicted phase coverage")):
        for i, (model, mode) in enumerate(GROUPS):
            values = runs[(runs.model == model)&(runs.training_mode == mode)][key].to_numpy()
            ax.scatter(i+np.linspace(-.13, .13, 5), values, color=COLORS[model], s=17)
            ax.errorbar(i, values.mean(), yerr=values.std(ddof=0), fmt="_", color="black", capsize=3)
        ax.set_xticks(range(6), [LABELS[m]+"\n"+mode for m, mode in GROUPS], rotation=25, fontsize=7)
        ax.set_title(title)
        if key == "active_coherence_rel_l2": ax.set_yscale("log")
    rows = []
    for ax, mode in zip(axes[1], TRAINING_MODES):
        for model in MODELS:
            data = read_dense(model, mode)
            index = int(np.argmin(np.abs(data["x"])))
            predicted, exact = data["current_predicted"][:, index], data["current_exact"][:, index]
            ax.plot(data["t"], predicted, color=COLORS[model], label=LABELS[model])
            if model == MODELS[0]: ax.plot(data["t"], exact, color="black", ls=":", label="Reference")
            rows += [dict(model=model, training_mode=mode, seed=0, x=data["x"][index], t=t, predicted=p, reference=e)
                     for t, p, e in zip(data["t"], predicted, exact)]
        ax.set(xlabel="t", ylabel="Current density at x=0", title=mode+": x=0 only")
    axes[1, -1].legend(fontsize=7)
    return finish(fig, "coherence_current", {"metrics": runs, "current_seed0": pd.DataFrame(rows)},
                  "Coherence retains penalties for collapsed predicted channels; x=0 current is not a substitute for the full space-time current error.", seed_scope="all five seeds; current fixed seed 0")


def physical_fields():
    fig, axes = plt.subplots(4, 3, figsize=(10.7, 9.1), layout="constrained")
    sources = []
    for mode_index, mode in enumerate(TRAINING_MODES):
        data = read_dense("phase_release_ltpb", mode)
        xx, tt = np.meshgrid(data["x"], data["t"])
        extent = [data["x"][0], data["x"][-1], data["t"][0], data["t"][-1]]
        for field_index, field in enumerate(("rho", "current")):
            row = 2*mode_index+field_index
            exact, predicted = data[field+"_exact"], data[field+"_predicted"]
            if field == "current":
                peak = max(np.max(abs(exact)), np.max(abs(predicted)))
                vmin, vmax, cmap = -peak, peak, "RdBu_r"
            else:
                vmin, vmax, cmap = 0, max(np.max(exact), np.max(predicted)), "viridis"
            errors = abs(predicted-exact)
            for column, values in enumerate((exact, predicted, errors)):
                im = axes[row, column].imshow(values, extent=extent, origin="lower", aspect="auto", interpolation="nearest",
                    cmap=cmap if column < 2 else "magma", vmin=vmin if column < 2 else 0, vmax=vmax if column < 2 else errors.max())
                if column < 2: common_image = im
                axes[row, column].set(ylabel=f"{mode}: {field}\nt", xlabel="x")
                if row == 0: axes[row, column].set_title(("Reference", "Release prediction", "Absolute error")[column])
            fig.colorbar(common_image, ax=axes[row, :2], shrink=.85, label=field)
            fig.colorbar(im, ax=axes[row, 2], shrink=.85, label="Absolute error")
            sources.append(pd.DataFrame(dict(model="phase_release_ltpb", training_mode=mode, seed=0, field=field,
                           x=xx.ravel(), t=tt.ravel(), reference=exact.ravel(), predicted=predicted.ravel(), absolute_error=errors.ravel())))
    return finish(fig, "density_current_fields", {"physical_grids": pd.concat(sources)},
                  "Full physical fields reveal errors missed by a single-position trace; reference/prediction share scales, error has an explicit separate scale.", seed_scope="fixed seed 0; release ordinary and causal")


def residual_weights(runs):
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 8.4), layout="constrained")
    histories, all_histories, integrals, segments = [], [], [], []
    for column, model in enumerate(MODELS):
        history = pd.read_csv(RAW/"main"/model/"causal/seed_0/loss_history.csv")
        history["model"] = model
        histories.append(history)
        for mode in TRAINING_MODES:
            for seed in range(5):
                full_history = pd.read_csv(RAW/"main"/model/mode/f"seed_{seed}/loss_history.csv")
                full_history["model"], full_history["training_mode"], full_history["seed"] = model, mode, seed
                all_histories.append(full_history)
        for i in range(8):
            color = plt.cm.viridis(i/7)
            axes[0, column].plot(history.step, history[f"segment_{i+1}_weight"], color=color, lw=.8, label=f"I{i+1}")
            axes[1, column].semilogy(history.step, history[f"segment_{i+1}_unweighted_pde"], color=color, lw=.8)
        axes[0, column].set(title=LABELS[model]+": causal weights", ylim=(-.02, 1.02))
        axes[1, column].set(title="Unweighted segment residual", xlabel="Step")
        for mode in TRAINING_MODES:
            table = pd.read_csv(RAW/"main"/model/mode/"seed_0/residual_integral.csv")
            table["model"], table["training_mode"] = model, mode
            integrals.append(table)
            axes[2, column].plot(table.t, table.error_bound, color=COLORS[model], ls="-" if mode == "ordinary" else "--", label=mode+" bound")
            axes[2, column].plot(table.t, table.raw_spinor_error_norm, color="black", ls="-" if mode == "ordinary" else "--", lw=.9, label=mode+" raw error")
            for seed in range(5):
                segment = pd.read_csv(RAW/"main"/model/mode/f"seed_{seed}/residual_by_segment_mode.csv")
                segment["model"], segment["training_mode"], segment["seed"] = model, mode, seed
                segments.append(segment)
        axes[2, column].set(xlabel="t", ylabel="Unweighted integral / raw error")
    axes[0, -1].legend(ncol=4, fontsize=6)
    axes[2, -1].legend(fontsize=6)
    return finish(fig, "residual_weights", {"causal_history_seed0": pd.concat(histories),
                  "all_seed_history": pd.concat(all_histories),
                  "integral_seed0": pd.concat(integrals), "dense_segment_mode_all_seeds": pd.concat(segments),
                  "quadrature_metrics": runs}, "Small causal objective is not a residual certificate; numerical quadrature is separately refined.", seed_scope="all-seed dense segment source; trajectories fixed seed 0")


def accuracy_cost(runs):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.3), layout="constrained")
    for model, mode in GROUPS:
        rows = runs[(runs.model == model)&(runs.training_mode == mode)]
        marker = "o" if mode == "ordinary" else "^"
        axes[0, 0].scatter(rows.training_seconds, rows.spinor_rel_l2_gauge_aligned, color=COLORS[model], marker=marker, s=25)
        axes[0, 1].scatter(rows.peak_rss_mb, rows.current_rel_l2, color=COLORS[model], marker=marker, s=25)
    axes[0, 0].set(xlabel="Training wall time (s)", ylabel="Spinor relative L2", yscale="log")
    axes[0, 1].set(xlabel="Peak training RSS (MB)", ylabel="Current relative L2", yscale="log")
    histories = []
    for model, mode in GROUPS:
        part = []
        for seed in range(5):
            history = pd.read_csv(RAW/"main"/model/mode/f"seed_{seed}/loss_history.csv")
            history["model"], history["training_mode"], history["seed"] = model, mode, seed
            histories.append(history)
            part.append(history)
        summary = pd.concat(part).groupby("step")[["objective", "unweighted_pde"]].mean()
        for ax, key in zip(axes[1], ("objective", "unweighted_pde")):
            for history in part:
                ax.semilogy(history.step, history[key], color=COLORS[model], ls="-" if mode == "ordinary" else "--", alpha=.12, lw=.6)
            ax.semilogy(summary.index, summary[key], color=COLORS[model], ls="-" if mode == "ordinary" else "--")
            ax.set(xlabel="Step", ylabel=key.replace("_", " "))
    fig.legend(handles=legend_handles()+[Line2D([], [], color="black", ls="--", label="causal (dashed/triangle)")], loc="outside lower center", ncol=4)
    return finish(fig, "accuracy_cost", {"runs": runs, "all_seed_history": pd.concat(histories)},
                  "All seeds share 12932 parameters and 2.08M training residual points; actual wall time and RSS are measured, not assumed equal.", seed_scope="all five seeds")


def interactions():
    source = pd.read_csv(RAW/"factor_interactions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), layout="constrained")
    for ax, (metric, rows) in zip(axes, source.groupby("metric", sort=False)):
        ax.scatter(rows.seed, rows.interaction, color="#6B5B95", s=25)
        ax.axhline(0, color="gray", ls=":")
        ax.errorbar(5.0, rows.interaction.mean(), yerr=rows.interaction.std(ddof=0), color="black", fmt="o", capsize=4)
        ax.set_xticks(range(6), ["0", "1", "2", "3", "4", "Mean±SD"])
        ax.set(ylabel="Difference-in-differences", xlabel="Paired seed", title="Spinor" if metric.startswith("spinor") else "Current")
    return finish(fig, "factor_interactions", {"interactions": source, "paired_effects": pd.read_csv(RAW/"paired_summary.csv")},
                  "Representation-by-training interaction is a descriptive paired difference, not a significance test.", seed_scope="all five paired seeds")


def main():
    runs = pd.read_csv(RAW/"main_runs.csv")
    manifest = [primary(runs), checkpoints(pd.read_csv(RAW/"checkpoint_metrics.csv")),
                short_dynamics(pd.read_csv(RAW/"short_runs.csv")), main_tangent(runs), modal_heatmaps(), trajectories(),
                coherence_current(runs), physical_fields(), residual_weights(runs), accuracy_cost(runs), interactions()]
    if (RAW/"long_runs.csv").exists():
        manifest.append(primary(pd.read_csv(RAW/"long_runs.csv"), name="long_time_validation"))
    write_json(FIGURES/"figure_manifest.json", dict(figures=manifest, statistics="population SD; all seeds",
               fixed_single_run_seed=0, backend="Python/matplotlib"))
    print(f"Exported {len(manifest)} figures with PNG/SVG/PDF and source CSV", flush=True)


if __name__ == "__main__":
    main()
