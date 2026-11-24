from __future__ import annotations
import json
import math
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from .r9_config import RAW, FIGURES, MODELS, SEEDS, run_directory, THRESHOLDS
from .freeze_r7_phase_release import write_json, sha256

LABELS = {MODELS[0]: "A: learned carrier", MODELS[1]: "B: Complex", MODELS[2]: "C: double Complex"}
COLORS = {MODELS[0]: "#376C91", MODELS[1]: "#A96D3D", MODELS[2]: "#776697", "Reference": "#252525"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 9, "legend.fontsize": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def save(fig, name, sources, *, seeds, claim):
    FIGURES.mkdir(parents=True, exist_ok=True)
    source_dir = FIGURES/"source_data"
    source_dir.mkdir(exist_ok=True)
    source_paths = []
    for suffix, data in sources.items():
        path = source_dir/f"{name}_{suffix}.csv"
        data.to_csv(path, index=False)
        source_paths.append(path)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    # Matplotlib retains tick-label artists beyond the view limits even though
    # Axis.draw does not render them. Their boxes must not count as clipping.
    undrawn_tick_labels = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            tolerance = 1e-12*max(1., abs(low), abs(high))
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if tick.get_loc() < low-tolerance or tick.get_loc() > high+tolerance:
                    undrawn_tick_labels.update((id(tick.label1), id(tick.label2)))
    outside = []
    for item in fig.findobj(Text):
        if id(item) not in undrawn_tick_labels and item.get_visible() and item.get_text().strip():
            box = item.get_window_extent(renderer)
            if box.x0 < -2 or box.y0 < -2 or box.x1 > width+2 or box.y1 > height+2:
                outside.append(item.get_text())
    files = []
    for suffix in ("png", "svg", "pdf"):
        path = FIGURES/f"{name}.{suffix}"
        fig.savefig(path, dpi=300)
        files.append(path)
    plt.close(fig)
    return dict(figure=name, claim=claim, seeds=list(seeds), source_csv=[str(p.relative_to(FIGURES)) for p in source_paths],
                exports={p.suffix[1:]: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in files},
                text_outside_canvas=outside, bounding_box_pass=not outside, backend="Python/matplotlib",
                undrawn_out_of_view_tick_labels_excluded=len(undrawn_tick_labels),
                statistics=("all n=5 paired seeds; mean and population SD, ddof=0; no significance test" if len(seeds) == 5
                            else ("fixed seed300 example; no seed uncertainty estimated" if seeds else "deterministic reference calibration; no seed uncertainty estimated")),
                svg_editable_text="<text" in files[1].read_text(encoding="utf-8"))


def panel_letters(axes):
    for letter, ax in zip(string.ascii_lowercase, np.asarray(axes).ravel()):
        ax.text(-.07, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)


def seed_points(ax, frame, key, *, log=False):
    for index, name in enumerate(MODELS):
        values = frame[frame.model == name].sort_values("seed")[key].to_numpy()
        ax.scatter(index+np.linspace(-.14, .14, 5), values, c=COLORS[name], s=26, zorder=3)
        ax.errorbar(index, values.mean(), yerr=values.std(ddof=0), fmt="_", capsize=5, color="black", lw=1)
    ax.set_xticks(range(3), ["A", "B", "C"])
    if log and (frame[key] > 0).all():
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=.16)


def make_figures():
    frame = pd.read_csv(RAW/"per_seed_metrics.csv")
    qa = []
    reference = json.loads((RAW/"reference_validation.json").read_text(encoding="utf-8"))
    calibration = pd.DataFrame(reference["comparisons"])
    free = np.load(RAW/"diagnostic_ignoring_potential.npz")
    coupled = np.load(RAW/"reference_test_grid.npz")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), layout="constrained")
    labels = [f"[{r[0]}, {r[1]}]" for r in calibration.smaller]
    axes[0].plot(labels, calibration.full_complex_zero_padded_relative_difference, "o-", color=COLORS[MODELS[0]])
    axes[0].axhline(1e-6, ls="--", color="black", lw=.9)
    axes[0].set(ylabel="Zero-padded trajectory relative L2", yscale="log", title="Reference convergence")
    for column, label in (("lower_eight_layers_max_total_occupation", "Lower 8 layers"),
                          ("upper_eight_layers_max_total_occupation", "Upper 8 layers")):
        axes[1].plot(labels, calibration[column], "o-", label=label)
    axes[1].axhline(1e-8, ls="--", color="black", lw=.9)
    axes[1].set(ylabel="Maximum total boundary occupation", yscale="log", title="Boundary blocks")
    axes[1].legend()
    difference = np.linalg.norm(coupled["state"]-free["state"], axis=(1, 2))
    axes[2].plot(coupled["t"], difference, color=COLORS[MODELS[0]])
    axes[2].set(xlabel="t", ylabel="Raw spinor difference norm", title="Effect of V = 0.2 cos(x)")
    panel_letters(axes)
    qa.append(save(fig, "reference_calibration", {"truncation": calibration,
        "free_difference": pd.DataFrame(dict(t=coupled["t"], raw_difference_norm=difference))},
        seeds=[], claim="The prescribed second pair converges; the potential measurably changes the trajectory."))

    keys = list(THRESHOLDS)
    titles = ["Spinor relative L2", "Current relative L2", "Mean fidelity", "Modal population relative L2", "Maximum norm drift"]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.3), layout="constrained")
    for ax, key, title in zip(axes.ravel(), keys, titles):
        seed_points(ax, frame, key)
        ax.axhline(THRESHOLDS[key], color="black", ls="--", lw=.9)
        ax.set_title(title)
    axes.ravel()[5].axis("off")
    axes.ravel()[5].text(.0, .9, "A: learned carrier + complex envelope\nB: one Complex network\nC: two independent Complex networks\n\nAll seeds 300-304 shown\nBlack: mean +/- population SD\nDashed: absolute criterion\nFinal checkpoint: step 20000", va="top", fontsize=9)
    panel_letters(axes.ravel()[:5])
    qa.append(save(fig, "all_seed_primary_metrics", {"all_seeds": frame[["model", "seed"]+keys]},
        seeds=SEEDS, claim="Absolute success is assessed with all five metrics and all five seeds."))

    paired = pd.read_csv(RAW/"paired_comparisons.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.5), layout="constrained")
    for row, key in enumerate(keys[:2]):
        for column, variable in enumerate(("paired_difference", "actual_ratio")):
            ax = axes[row, column]
            for index, (target, base) in enumerate(((MODELS[0], MODELS[1]), (MODELS[0], MODELS[2]), (MODELS[2], MODELS[1]))):
                values = paired[(paired.target == target)&(paired.baseline == base)&(paired.metric == key)].sort_values("seed")[variable]
                ax.scatter(index+np.linspace(-.14, .14, 5), values, color=COLORS[target], s=28)
                ax.errorbar(index, values.mean(), yerr=values.std(ddof=0), fmt="_", capsize=5, color="black")
            ax.axhline(0 if column == 0 else 1, ls="--", color="black", lw=.8)
            ax.set_xticks(range(3), ["A - B" if column == 0 else "A / B", "A - C" if column == 0 else "A / C", "C - B" if column == 0 else "C / B"])
            ax.set_title(("Spinor" if row == 0 else "Current")+": "+("paired difference" if column == 0 else "actual error ratio"))
    panel_letters(axes)
    qa.append(save(fig, "paired_comparisons", {"all_pairs": paired}, seeds=SEEDS,
                   claim="Observed paired differences and ratios assess advantage separately from absolute success."))

    dense = {name: np.load(run_directory(name, 300)/"dense_grid.npz") for name in MODELS}
    first = dense[MODELS[0]]
    labels = ["Reference"]+[LABELS[name] for name in MODELS]
    fig, axes = plt.subplots(2, 4, figsize=(11, 6.1), layout="constrained")
    physical_rows = []
    for row, key in enumerate(("density", "current")):
        arrays = [first[f"{key}_reference"]]+[dense[name][f"{key}_predicted"] for name in MODELS]
        scale = max(float(abs(array).max()) for array in arrays)
        for col, (array, label) in enumerate(zip(arrays, labels)):
            im = axes[row, col].imshow(array, origin="lower", aspect="auto", extent=[first["x"][0], first["x"][-1], 0, 5],
                 cmap="viridis" if row == 0 else "RdBu_r", vmin=0 if row == 0 else -scale, vmax=scale, rasterized=True)
            axes[row, col].set(xlabel="x", ylabel="t" if col == 0 else "", title=label+ (" density" if row == 0 else " current"))
            ti, xi = np.indices(array.shape)
            physical_rows.append(pd.DataFrame(dict(quantity=key, model=label, seed=300, t=first["t"][ti.ravel()],
                                                    x=first["x"][xi.ravel()], value=array.ravel())))
        fig.colorbar(im, ax=axes[row].tolist(), shrink=.85, label=key)
    panel_letters(axes)
    qa.append(save(fig, "dynamics_seed300", {"full_grid": pd.concat(physical_rows, ignore_index=True)}, seeds=[300],
                   claim="Fixed seed300 density and current reveal whether the whole wavepacket dynamics is recovered."))

    fig, axes = plt.subplots(3, 4, figsize=(10.8, 7.7), layout="constrained")
    modal_rows = []
    for row, quantity in enumerate(("total", "plus", "minus")):
        arrays = [first["modal_occupations_reference"] if row == 0 else first["branch_occupations_reference"][..., row-1]]
        arrays += [dense[name]["modal_occupations_predicted"] if row == 0 else dense[name]["branch_occupations_predicted"][..., row-1] for name in MODELS]
        vmax = max(float(a.max()) for a in arrays)
        for col, (array, label) in enumerate(zip(arrays, labels)):
            # Shared color scale; no per-model normalization.
            im = axes[row, col].imshow(array, origin="lower", aspect="auto", extent=[first["n"][0]-.5, first["n"][-1]+.5, 0, 5],
                                       cmap="viridis", vmin=0, vmax=vmax, rasterized=True)
            axes[row, col].axvline(-.5, color="white", ls="--", lw=.7)
            axes[row, col].axvline(32.5, color="white", ls="--", lw=.7)
            axes[row, col].set(xlabel="Fourier index n", ylabel="t" if col == 0 else "", title=label+": "+quantity)
            ti, ni = np.indices(array.shape)
            modal_rows.append(pd.DataFrame(dict(quantity=quantity, model=label, seed=300, t=first["t"][ti.ravel()],
                                                n=first["n"][ni.ravel()], occupation=array.ravel())))
        fig.colorbar(im, ax=axes[row].tolist(), shrink=.8, label="Occupation")
    panel_letters(axes)
    qa.append(save(fig, "modal_branch_seed300", {"full_grid": pd.concat(modal_rows, ignore_index=True)}, seeds=[300],
        claim="Modal and free-branch occupations compare the original broad spectrum and generated support outside n=0,...,32."))

    series_rows = []
    for name in MODELS:
        for seed in SEEDS:
            part = pd.read_csv(run_directory(name, seed)/"time_series.csv")
            series_rows.append(part.assign(model=name, seed=seed))
    series = pd.concat(series_rows, ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.9, 5.7), layout="constrained")
    for row, branch in enumerate(("plus", "minus")):
        for col, quantity in enumerate(("center", "width")):
            ax = axes[row, col]
            reference_series = series[(series.model == MODELS[0])&(series.seed == 300)]
            ax.plot(reference_series.t, reference_series[f"{branch}_{quantity}_reference"], color="black", lw=2, label="Reference")
            for name in MODELS:
                for seed in SEEDS:
                    part = series[(series.model == name)&(series.seed == seed)]
                    ax.plot(part.t, part[f"{branch}_{quantity}_predicted"], color=COLORS[name],
                            lw=1.4 if seed == 300 else .7, alpha=.9 if seed == 300 else .5,
                            label=LABELS[name] if seed == 300 else None)
            ax.set(xlabel="t", ylabel=quantity+" (length)", title=branch+" branch "+quantity)
    axes[0, 0].legend(fontsize=7)
    panel_letters(axes)
    qa.append(save(fig, "branch_center_width", {"all_seed_trajectories": series}, seeds=SEEDS,
                   claim="All five seeds' branch centers and widths are compared to the potential-driven reference."))

    diagnostic_keys = ["active_coherence_rel_l2", "branch_complex_coefficient_rel_l2", "residual_integral_fine",
        "exterior_residual_leakage_max_l2_norm", "predicted_max_occupation_outside_initial_support", "fine_unweighted_pde_loss"]
    diagnostic_titles = ["Active coherence relative L2", "Complex branch coefficient relative L2", "Numerical residual integral",
        "Maximum exterior residual norm", "Occupation outside initial support", "Fine unweighted PDE loss"]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), layout="constrained")
    for ax, key, title in zip(axes.ravel(), diagnostic_keys, diagnostic_titles):
        seed_points(ax, frame, key, log=key not in diagnostic_keys[:2])
        ax.set_title(title)
        if key == "predicted_max_occupation_outside_initial_support":
            ax.axhline(frame.reference_max_occupation_outside_initial_support.iloc[0], color="black", ls="--", lw=.8)
    panel_letters(axes)
    qa.append(save(fig, "diagnostics_all_seeds", {"all_seeds": frame[["model", "seed"]+diagnostic_keys]}, seeds=SEEDS,
                   claim="Coherence, complex coefficients, residual integration, and exterior leakage test distinct failure modes."))

    carrier = frame[frame.model == MODELS[0]]
    fig, axes = plt.subplots(1, 3, figsize=(9.1, 3.2), layout="constrained")
    for ax, key, title in zip(axes, ("carrier_free_pde_residual_rms", "carrier_phase_slope_mean_relative_error", "carrier_phase_slope_max_relative_error"),
        ("Unit-IVP carrier defect RMS", "Mean relative phase-slope error", "Maximum relative phase-slope error")):
        ax.scatter(carrier.seed, carrier[key], color=COLORS[MODELS[0]])
        ax.set(xlabel="Seed", ylabel=title)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.set_xticks(list(SEEDS))
    panel_letters(axes)
    qa.append(save(fig, "carrier_defects_all_seeds", {"all_seeds": carrier[["model", "seed", "carrier_free_pde_residual_rms",
            "carrier_phase_slope_mean_relative_error", "carrier_phase_slope_max_relative_error"]]}, seeds=SEEDS,
                   claim="The frozen carrier is learned and approximate; exact free rates are evaluation-only targets."))

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.3), layout="constrained")
    for ax, key, title in zip(axes, ("training_seconds", "evaluation_seconds", "peak_rss_mib"),
                             ("Training wall time (s)", "Evaluation wall time (s)", "Peak RSS (MiB)")):
        seed_points(ax, frame, key)
        ax.set_title(title)
    panel_letters(axes)
    qa.append(save(fig, "training_costs", {"all_costs": pd.read_csv(RAW/"cost_and_residual_work.csv")}, seeds=SEEDS,
                   claim="Equal optimizer steps have different parameter, residual, time and memory costs."))
    write_json(FIGURES/"FIGURE_QA.json", dict(status="PASS" if all(q["bounding_box_pass"] and q["svg_editable_text"] for q in qa) else "FAIL",
        visual_inspection_pending=True, figures=qa, source_data_complete=True, all_statistics_include_five_seeds=True,
        fixed_dynamic_seed=300, image_adjustments="shared scales; no smoothing or selective omission; heatmaps rasterized in vector files"))
    (FIGURES/"FIGURE_QA.md").write_text("# R9 FIGURE_QA\n\nPython/matplotlib; all statistics use five seeds and population SD. Dynamics use seed300.\n\n"+
        "\n".join(f"- {q['figure']}: bounding boxes {'PASS' if q['bounding_box_pass'] else 'FAIL'}; editable SVG {'PASS' if q['svg_editable_text'] else 'FAIL'}; source CSV complete." for q in qa)+
        "\n\nVisual inspection must be recorded before final delivery.\n", encoding="utf-8")
    return qa


if __name__ == "__main__":
    print(json.dumps(make_figures(), indent=2))
