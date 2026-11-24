from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .periodic_potential_config import PERIODIC_POTENTIAL_MODELS
from .periodic_potential_coupled_ltpb_decision import TARGET, CONTROL
from .train_periodic_potential_coupled_ltpb import ROOT, RAW, write_json

FIGURES = ROOT/"results/figures/periodic_potential_coupled_ltpb"
COLORS = {"physical_vanilla": "#777777", CONTROL: "#4477AA", TARGET: "#CC6677"}
LABELS = {"physical_vanilla": "Physical Vanilla", CONTROL: "Spectral Complex", TARGET: "Spectral LTPB"}
FIGURE_IDS = (
    "calibration_modal_transfer", "all_seed_primary_metrics", "modal_occupation_heatmaps_seed0",
    "center_sideband_trajectories_seed0", "energy_branch_occupations_seed0",
    "density_current_spacetime_seed0", "active_channel_diagnostics_seed0",
    "training_efficiency_comparison",
)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7,
                     "legend.frameon": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white"})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_seed0(model=TARGET):
    with np.load(RAW/"learned"/model/"seed_0/dense_grid.npz") as data:
        return {key: data[key] for key in data.files}


def finish(fig, identifier, sources, *, seed_scope, narrative):
    FIGURES.mkdir(parents=True, exist_ok=True)
    data_dir = FIGURES/"source_data"
    data_dir.mkdir(exist_ok=True)
    source_info = []
    for suffix, table in sources.items():
        name = f"{identifier}{'_'+suffix if suffix else ''}.csv"
        path = data_dir/name
        table.to_csv(path, index=False)
        source_info.append({"file": f"source_data/{name}", "rows": len(table),
                            "columns": list(table.columns), "sha256": sha(path)})
    outputs = {}
    for extension in ("png", "svg", "pdf"):
        path = FIGURES/f"{identifier}.{extension}"
        fig.savefig(path, dpi=400, bbox_inches="tight")
        outputs[extension] = sha(path)
    plt.close(fig)
    return {"id": identifier, "source_data": source_info, "output_sha256": outputs,
            "seed_scope": seed_scope, "description": narrative}


def calibration_plot():
    source = pd.read_csv(RAW/"calibration/calibration_mode_occupation_source.csv")
    summary = pd.read_csv(RAW/"calibration/calibration_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.2), layout="constrained")
    for row, horizon in enumerate((5.0, 8.0)):
        ax = axes[row, 0]
        for strength, color in zip((0.1, 0.2, 0.5), ("#4477AA", "#228833", "#CC6677")):
            center = source[(source.t_max == horizon) & (source.potential_strength == strength) & (source.ladder_index == 0)]
            ax.plot(center.t, 1-center.modal_occupation, color=color, label=f"V0 = {strength:g}")
        ax.axhspan(0.1, 0.6, color="#CCCCCC", alpha=.25)
        ax.axhline(.1, color="black", lw=.5, ls=":")
        ax.axhline(.6, color="black", lw=.5, ls=":")
        ax.set(xlabel="Time", ylabel="Outside-center occupation", ylim=(0, 1.02), title=f"{'ac'[row]}   T = {horizon:g}: transfer")
        if row == 0:
            ax.legend(loc="upper left")
        ax = axes[row, 1]
        subset = summary[summary.t_max == horizon]
        for metric, label, marker in (
            ("truncation_comparison__padded_full_trajectory_spinor_relative_l2", "J4 / J6 full spinor", "o"),
            ("large_ladder_observables__max_outer_layer_total_occupation", "J6 boundary occupation", "s"),
        ):
            ax.plot(subset.potential_strength, subset[metric], marker=marker, ms=4, label=label)
        ax.axhline(1e-6, color="#4477AA", lw=.7, ls="--", label="Spinor gate 1e-6")
        ax.axhline(1e-8, color="#CC6677", lw=.7, ls=":", label="Boundary gate 1e-8")
        ax.set(xlabel="Potential V0", ylabel="Calibration diagnostic", yscale="log", title=f"{'bd'[row]}   T = {horizon:g}: original gates")
        if row == 0:
            ax.legend(fontsize=6, loc="lower right")
    selection = json.loads((RAW/"formal_selection.json").read_text(encoding="utf-8"))
    fig.suptitle(f"Approved revision: V0 = 0.2, T = 5, J6/J8 = {selection['truncation_relative_l2']:.2e}", fontsize=10)
    amended = pd.DataFrame([{**selection["selected"], "comparison_small_J": 6, "comparison_large_J": 8,
                             "full_spinor_relative_l2": selection["truncation_relative_l2"],
                             "convergence_threshold": 1e-6}])
    return finish(fig, FIGURE_IDS[0], {"trajectories": source, "gates": summary, "approved_revision": amended}, seed_scope="reference only", narrative="Original failed gates and authorized truncation revision remain visible")


def primary_plot(runs):
    quantities = (("spinor_rel_l2_gauge_aligned", "Spinor relative L2", .1),
                  ("current_rel_l2", "Current relative L2", .15),
                  ("modal_population_rel_l2", "Modal-population relative L2", .1),
                  ("density_rel_l2", "Density relative L2", None),
                  ("one_minus_fidelity", "1 - mean fidelity", .02),
                  ("max_norm_drift", "Maximum norm drift", .02))
    data = runs.copy()
    data["one_minus_fidelity"] = 1-data.fidelity_mean
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.8), layout="constrained")
    for panel, (metric, label, boundary) in enumerate(quantities):
        ax = axes.flat[panel]
        norm_panel = metric == "max_norm_drift"
        for index, model in enumerate(PERIODIC_POTENTIAL_MODELS):
            values = data[data.model == model].sort_values("seed")[metric].to_numpy()
            ax.scatter(index+np.linspace(-.12, .12, 5), np.maximum(values, 1e-15), c=COLORS[model], s=18, zorder=3)
            mean, sd = np.mean(values), np.std(values, ddof=0)
            error = sd if norm_panel else [[min(sd, max(mean-1e-15, 0))], [sd]]
            ax.errorbar(index, mean, yerr=error, color="black", fmt="_", capsize=4, zorder=4)
        if boundary is not None:
            ax.axhline(boundary, color="black", linestyle=":", lw=.7)
        if norm_panel:
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_ylim(-.004, 2)
            label = "Max. norm drift (symlog)"
        else:
            ax.set_yscale("log")
        ax.set(xticks=range(3), xticklabels=("Physical", "Complex", "LTPB"), title=f"{chr(97+panel)}   {label}")
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Five paired seeds: points; mean and population SD: black bars", fontsize=10)
    return finish(fig, FIGURE_IDS[1], {"": data[["model", "seed"]+[q[0] for q in quantities]]}, seed_scope="all five seeds", narrative="All absolute criteria and seed variation")


def mode_heatmaps(d):
    truth, pred = d["modal_occupations_exact"], d["modal_occupations_predicted"]
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.9), layout="constrained")
    for n, (values, title) in enumerate(((truth, "Reference"), (pred, "LTPB"), (np.abs(pred-truth), "Absolute error"))):
        upper = max(truth.max(), pred.max()) if n < 2 else np.abs(pred-truth).max()
        im = axes[n].pcolormesh(d["t"], d["mode_offsets"], values.T, shading="nearest", vmin=0, vmax=max(upper, 1e-15), cmap="viridis", rasterized=True)
        axes[n].set(title=f"{chr(97+n)}   {title}", xlabel="Time", ylabel="Mode offset j")
        fig.colorbar(im, ax=axes[n], shrink=.85, label="Occupation")
    ti, ji = np.meshgrid(d["t"], d["mode_offsets"], indexing="ij")
    source = pd.DataFrame({"t": ti.ravel(), "j": ji.ravel(), "reference": truth.ravel(), "ltpb_seed0": pred.ravel(), "absolute_error": np.abs(pred-truth).ravel()})
    return finish(fig, FIGURE_IDS[2], {"": source}, seed_scope="seed 0", narrative="Full ladder modal transfer with a shared reference/prediction scale")


def sidebands():
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.8), layout="constrained")
    sources = []
    for model in PERIODIC_POTENTIAL_MODELS:
        d = load_seed0(model)
        for panel, offset in enumerate((0, -1, 1)):
            idx = int(np.flatnonzero(d["mode_offsets"] == offset)[0])
            truth, pred = d["modal_occupations_exact"][:, idx], d["modal_occupations_predicted"][:, idx]
            axes[panel].plot(d["t"], pred, color=COLORS[model], lw=1.1, label=LABELS[model])
            if model == PERIODIC_POTENTIAL_MODELS[0]:
                axes[panel].plot(d["t"], truth, color="black", lw=1.2, ls="--", label="Reference")
            axes[panel].set(title=f"{chr(97+panel)}   j = {offset:+d}", xlabel="Time", ylabel="Modal occupation")
            sources.append(pd.DataFrame({"model": model, "seed": 0, "j": offset, "t": d["t"], "reference": truth, "predicted": pred}))
    axes[0].legend(fontsize=6, loc="best")
    return finish(fig, FIGURE_IDS[3], {"": pd.concat(sources)}, seed_scope="seed 0", narrative="Center and directly coupled sidebands across all three models")


def branches(d):
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.7), layout="constrained")
    sources = []
    for branch in range(2):
        truth, pred = d["branch_occupations_exact"][:, :, branch], d["branch_occupations_predicted"][:, :, branch]
        mask = d["reference_active_branch_mask"][:, :, branch]
        for col, (values, title) in enumerate(((truth, "Reference"), (pred, "LTPB"), (mask, "Reference active mask"))):
            ax = axes[branch, col]
            im = ax.pcolormesh(d["t"], d["mode_offsets"], values.T, shading="nearest", cmap="Greys" if col == 2 else "viridis",
                               vmin=0, vmax=1 if col == 2 else max(truth.max(), pred.max(), 1e-15), rasterized=True)
            ax.set(title=f"{chr(97+branch*3+col)}   {'+' if branch == 0 else '-'} : {title}", xlabel="Time", ylabel="Mode offset j")
            fig.colorbar(im, ax=ax, shrink=.8)
        ti, ji = np.meshgrid(d["t"], d["mode_offsets"], indexing="ij")
        sources.append(pd.DataFrame({"branch": branch, "t": ti.ravel(), "j": ji.ravel(), "reference": truth.ravel(),
                                     "ltpb_seed0": pred.ravel(), "reference_active_mask": mask.ravel().astype(int)}))
    return finish(fig, FIGURE_IDS[4], {"": pd.concat(sources)}, seed_scope="seed 0", narrative="Free-basis occupations and physical phase-validity regions")


def physical_fields(d):
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.7), layout="constrained")
    sources = []
    for row, field in enumerate(("rho", "current")):
        truth, pred = d[f"{field}_exact"], d[f"{field}_predicted"]
        bound = max(np.max(np.abs(truth)), np.max(np.abs(pred)), 1e-15)
        for col, (values, title) in enumerate(((truth, "Reference"), (pred, "LTPB"), (np.abs(pred-truth), "Absolute error"))):
            ax = axes[row, col]
            signed = field == "current" and col < 2
            im = ax.pcolormesh(d["x"], d["t"], values, shading="nearest", cmap="RdBu_r" if signed else "viridis",
                               vmin=-bound if signed else 0, vmax=bound if col < 2 else max(values.max(), 1e-15), rasterized=True)
            ax.set(title=f"{chr(97+row*3+col)}   {field}: {title}", xlabel="Position x", ylabel="Time")
            fig.colorbar(im, ax=ax, shrink=.8, format="%.2g")
        ti, xi = np.meshgrid(d["t"], d["x"], indexing="ij")
        sources.append(pd.DataFrame({"field": field, "t": ti.ravel(), "x": xi.ravel(), "reference": truth.ravel(),
                                     "ltpb_seed0": pred.ravel(), "absolute_error": np.abs(pred-truth).ravel()}))
    return finish(fig, FIGURE_IDS[5], {"": pd.concat(sources)}, seed_scope="seed 0", narrative="Density and current use full fields and shared scales")


def active_diagnostics(d):
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 4.9), layout="constrained")
    sources = []
    center = int(np.flatnonzero(d["mode_offsets"] == 0)[0])
    for offset, color in ((-1, "#4477AA"), (1, "#CC6677")):
        idx = int(np.flatnonzero(d["mode_offsets"] == offset)[0])
        for branch, ls in ((0, "-"), (1, ":")):
            a, b = 2*idx+branch, 2*center+branch
            mask = d["reference_active_pair_mask"][:, a, b]
            exact, pred = d["channel_coherence_exact"][:, a, b], d["channel_coherence_predicted"][:, a, b]
            amplitudes = np.sqrt(d["branch_occupations_predicted"][:, idx, branch])
            exact_amplitudes = np.sqrt(d["branch_occupations_exact"][:, idx, branch])
            label = f"j={offset:+d}, {'+' if branch == 0 else '-'}"
            axes[0, 0].plot(d["t"], exact_amplitudes, color=color, ls=ls, alpha=.4)
            axes[0, 0].plot(d["t"], amplitudes, color=color, ls=ls, label=label)
            axes[0, 1].plot(d["t"], np.where(mask, np.abs(pred-exact), np.nan), color=color, ls=ls, label=label)
            predicted_valid = d["predicted_phase_defined_mask"].reshape(len(d["t"]), -1)
            phase_defined = mask & predicted_valid[:, a] & predicted_valid[:, b]
            phase_error = np.where(phase_defined, np.angle(pred*np.conj(exact)), np.nan)
            axes[1, 0].plot(d["t"], phase_error, color=color, ls=ls)
            sources.append(pd.DataFrame({"t": d["t"], "j": offset, "branch": branch,
                                         "amplitude_reference": exact_amplitudes, "amplitude_ltpb": amplitudes,
                                         "coherence_real_reference": exact.real, "coherence_imag_reference": exact.imag,
                                         "coherence_real_ltpb": pred.real, "coherence_imag_ltpb": pred.imag,
                                         "pair_valid_mask": mask.astype(int), "phase_defined_mask": phase_defined.astype(int),
                                         "relative_phase_error_rad": phase_error}))
    axes[0, 0].set(title="a   Sideband amplitudes", xlabel="Time", ylabel="Coefficient magnitude")
    axes[0, 0].legend(fontsize=6, title="Pale: reference; dark: LTPB", title_fontsize=6)
    axes[0, 1].set(title="b   Active sideband-center coherence", xlabel="Time", ylabel="Absolute complex error")
    axes[1, 0].set(title="c   Active relative-phase error", xlabel="Time", ylabel="Wrapped phase error (rad)", ylim=(-np.pi, np.pi))
    tangent = []
    for index, model in enumerate((CONTROL, TARGET)):
        data = load_seed0(model)
        mask = data["initial_first_sideband_mask"]
        pred = data["initial_time_derivative_predicted"][mask]
        exact = data["initial_time_derivative_required_by_pde"][mask]
        imaginary = np.linalg.norm(pred.imag)
        required = np.linalg.norm(exact.imag)
        axes[1, 1].bar(index, imaginary, color=COLORS[model], width=.5)
        axes[1, 1].scatter(index, required, color="black", marker="_", s=150, zorder=5)
        tangent.append({"model": model, "seed": 0, "predicted_imaginary_derivative_norm": imaginary,
                        "required_imaginary_derivative_norm": required})
    axes[1, 1].set(title="d   Initial sideband tangent", ylabel="Imaginary derivative norm", xticks=[0, 1], xticklabels=["Complex", "LTPB"])
    axes[1, 1].text(.46, .72, "Black marks: PDE-required", transform=axes[1, 1].transAxes, va="top", fontsize=6)
    return finish(fig, FIGURE_IDS[6], {"channels": pd.concat(sources), "tangent": pd.DataFrame(tangent)}, seed_scope="seed 0", narrative="Only valid pairs are used; missing phases remain masked and tangent constraint is explicit")


def efficiency(runs):
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.0), layout="constrained")
    sources = []
    for model in PERIODIC_POTENTIAL_MODELS:
        for seed in range(5):
            h = pd.read_csv(RAW/"learned"/model/f"seed_{seed}/loss_history.csv")
            axes[0, 0].plot(h.step, h.pde, color=COLORS[model], alpha=.45, lw=.6,
                            label=LABELS[model] if seed == 0 else None)
            sources.append(h.assign(model=model, seed=seed))
    axes[0, 0].set(yscale="log", xlabel="Adam step", ylabel="PDE loss", title="a   All 15 training curves")
    axes[0, 0].legend(fontsize=6, loc="upper right")
    for ax, metric, title, label in ((axes[0, 1], "training_seconds", "b   Training time", "Seconds"),
                                    (axes[1, 0], "peak_rss_mb", "c   Peak training RSS", "MiB")):
        for index, model in enumerate(PERIODIC_POTENTIAL_MODELS):
            y = runs[runs.model == model].sort_values("seed")[metric].to_numpy()
            ax.scatter(index+np.linspace(-.12, .12, 5), y, color=COLORS[model], s=20)
            ax.errorbar(index, np.mean(y), yerr=np.std(y, ddof=0), fmt="_", color="black", capsize=4)
        ax.set(title=title, ylabel=label, xticks=range(3), xticklabels=["Physical", "Complex", "LTPB"])
    ax = axes[1, 1]
    for model in PERIODIC_POTENTIAL_MODELS:
        subset = runs[runs.model == model]
        ax.scatter(subset.training_seconds, subset.spinor_rel_l2_gauge_aligned, c=COLORS[model], s=22,
                   label=f"{LABELS[model]} ({int(subset.parameter_count.iloc[0])} params)")
    ax.set(yscale="log", xlabel="Training seconds", ylabel="Spinor relative L2", title="d   Accuracy / time / parameter budget")
    ax.legend(fontsize=5.5, loc="best")
    return finish(fig, FIGURE_IDS[7], {"loss": pd.concat(sources), "resources": runs[["model", "seed", "training_seconds", "peak_rss_mb", "peak_rss_increment_mb", "parameter_count", "spinor_rel_l2_gauge_aligned"]]},
                  seed_scope="all five seeds", narrative="All runs have equal step budgets; resource variation is measured, not inferred")


def generate_all():
    runs = pd.read_csv(RAW/"runs.csv")
    d = load_seed0()
    entries = [calibration_plot(), primary_plot(runs), mode_heatmaps(d), sidebands(), branches(d),
               physical_fields(d), active_diagnostics(d), efficiency(runs)]
    write_json(FIGURES/"figure_manifest.json", {"backend": "Python/matplotlib", "script_sha256": sha(Path(__file__)),
               "figures": entries, "all_statistical_panels_show_five_seeds": True,
               "spacetime_seed": 0, "sd_definition": "population SD, ddof=0"})
    return entries


if __name__ == "__main__":
    generate_all()
    print(f"Generated {len(FIGURE_IDS)} figures at {FIGURES}")
