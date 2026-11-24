from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .r10_config import RAW, SEEDS, EXPERIMENT, run_directory, source_directory
from .r9_config import ARTIFACTS, THRESHOLDS
from .r9_decision import absolute_success
from .freeze_r7_phase_release import write_json

STAGES = ("R9 step20000", "R10 step26000")


def paired_statistics(frame):
    before = frame[frame.stage == STAGES[0]].set_index("seed")
    after = frame[frame.stage == STAGES[1]].set_index("seed")
    metrics = tuple(THRESHOLDS)+("density_rel_l2", "active_coherence_rel_l2",
        "branch_population_rel_l2", "branch_complex_coefficient_rel_l2",
        "residual_integral_fine", "full_physical_residual_integral")
    rows, summary = [], []
    for key in metrics:
        delta = (after[key]-before[key]).loc[list(SEEDS)]
        ratio = (after[key]/before[key]).loc[list(SEEDS)]
        for seed in SEEDS:
            rows.append(dict(seed=seed, metric=key, before=float(before.loc[seed, key]),
                after=float(after.loc[seed, key]), paired_difference=float(delta[seed]),
                actual_ratio=float(ratio[seed])))
        summary.append(dict(metric=key, n=5, paired_difference_mean=float(delta.mean()),
            paired_difference_population_sd=float(delta.std(ddof=0)),
            ratio_of_means=float(after[key].mean()/before[key].mean()),
            mean_of_paired_ratios=float(ratio.mean()),
            favorable_pair_count=int((delta > 0 if key == "fidelity_mean" else delta < 0).sum())))
    return pd.DataFrame(rows), pd.DataFrame(summary)


def aggregate():
    rows, artifacts = [], []
    for stage, getter in zip(STAGES, (source_directory, run_directory)):
        for seed in SEEDS:
            directory = getter(seed)
            if not all((directory/name).is_file() for name in ARTIFACTS+("recovery_state.pt",)):
                raise RuntimeError("Missing paired fixed endpoint: "+str(directory))
            row = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
            row["stage"] = stage
            if stage == STAGES[0]:
                row.update(json.loads((RAW/"r9_source_diagnostics"/f"seed_{seed}"/"diagnostics.json").read_text(encoding="utf-8")))
            rows.append(row)
            for name in ARTIFACTS+("recovery_state.pt",):
                artifacts.append(dict(stage=stage, seed=seed, artifact=name, path=str(directory/name)))
    frame = pd.DataFrame(rows)
    frame.to_csv(RAW/"per_seed_metrics.csv", index=False)
    pd.DataFrame(artifacts).to_csv(RAW/"run_artifact_index.csv", index=False)
    keys = [k for k in frame.select_dtypes(include=[np.number]).columns if k != "seed"]
    summary = []
    for stage, part in frame.groupby("stage", sort=False):
        for key in keys:
            summary.append(dict(stage=stage, metric=key, n=len(part), n_valid=int(part[key].notna().sum()),
                applicable=bool(part[key].notna().any()),
                mean=float(part[key].mean()), population_sd=float(part[key].std(ddof=0)),
                minimum=float(part[key].min()), maximum=float(part[key].max())))
    pd.DataFrame(summary).to_csv(RAW/"summary_metrics.csv", index=False)
    pairs, changes = paired_statistics(frame)
    pairs.to_csv(RAW/"paired_changes.csv", index=False)
    changes.to_csv(RAW/"paired_change_summary.csv", index=False)
    decisions = {stage: absolute_success(frame[frame.stage == stage]) for stage in STAGES}
    decision = dict(experiment=EXPERIMENT,
        status="CONDITIONAL_BUDGET_RECOVERY" if decisions[STAGES[1]]["pass_check"] else "FIXED_ADDITIONAL_BUDGET_INSUFFICIENT",
        absolute_success=decisions, thresholds=THRESHOLDS, population_sd_ddof=0,
        inherited_optimizer_steps=20000, additional_optimizer_steps=6000, fixed_final_step=26000,
        all_five_ancestries_included=True, fresh_independent_seed_replication=False,
        r9_original_classification_unchanged=True, total_budget_matched_comparison_to_r9_C=False,
        pretraining_causal_effect_isolated=False, rescue_structure_added=False,
        scope="same five R9 training ancestries; known finite Fourier Gaussian spectrum and fixed weak periodic potential")
    write_json(RAW/"decision.json", decision)
    costs = []
    for seed in SEEDS:
        row = frame[(frame.stage == STAGES[1])&(frame.seed == seed)].iloc[0]
        names = ("active_parameters", "frozen_parameters", "total_parameters",
            "inherited_optimizer_steps", "additional_optimizer_steps", "steps", "carrier_steps", "coupled_steps",
            "inherited_carrier_training_seconds", "inherited_coupled_training_seconds", "inherited_training_seconds",
            "additional_training_seconds", "cumulative_training_seconds", "evaluation_seconds",
            "run_elapsed_seconds_this_session",
            "training_peak_rss_mib", "peak_rss_mib", "additional_coupled_spinor_residual_points",
            "additional_complex_scalar_residual_evaluations", "cumulative_training_spinor_residual_points",
            "evaluation_coupled_spinor_residual_points", "evaluation_carrier_unit_spinor_residual_points",
            "norm_diagnostic_additional_forward_spinor_points", "norm_diagnostic_additional_residual_calls",
            "cpu_affinity_mask")
        costs.append(dict(seed=seed, **{k: row[k] for k in names},
            additional_training_residual_batches=6000,
            inherited_training_residual_batches=20000,
            cumulative_training_residual_batches=26000,
            cumulative_complex_scalar_residual_evaluations=2*int(row["cumulative_training_spinor_residual_points"])))
    pd.DataFrame(costs).to_csv(RAW/"cost_and_residual_work.csv", index=False)
    return frame, decision


if __name__ == "__main__":
    print(json.dumps(aggregate()[1], indent=2))
