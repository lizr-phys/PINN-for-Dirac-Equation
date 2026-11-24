from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .r9_config import RAW, MODELS, SEEDS, ARTIFACTS, run_directory, THRESHOLDS
from .r9_decision import absolute_success, comparisons
from .freeze_r7_phase_release import write_json


def aggregate():
    records, artifact_rows = [], []
    for model in MODELS:
        for seed in SEEDS:
            directory = run_directory(model, seed)
            if not all((directory/p).is_file() for p in ARTIFACTS+("recovery_state.pt",)):
                raise RuntimeError(f"Missing formal evidence: {directory}")
            records.append(json.loads((directory/"metrics.json").read_text(encoding="utf-8")))
            for filename in ARTIFACTS+("recovery_state.pt", "execution_events.jsonl", "residual_by_mode.csv", "residual_integral.csv",
                                       "full_physical_residual.npz", "full_physical_residual_archival_record.json"):
                artifact_rows.append(dict(model=model, seed=seed, artifact=filename,
                                         path=(directory/filename).relative_to(RAW.parents[2]).as_posix()))
    frame = pd.DataFrame(records)
    frame.to_csv(RAW/"per_seed_metrics.csv", index=False)
    pd.DataFrame(artifact_rows).to_csv(RAW/"run_artifact_index.csv", index=False)
    numeric = [key for key in frame.select_dtypes(include=[np.number]).columns if key != "seed"]
    rows = []
    for model, part in frame.groupby("model", sort=False):
        for key in numeric:
            rows.append(dict(model=model, metric=key, n=len(part), mean=float(part[key].mean()),
                             n_valid=int(part[key].notna().sum()), applicable=bool(part[key].notna().any()),
                             population_sd=float(part[key].std(ddof=0)), minimum=float(part[key].min()),
                             maximum=float(part[key].max())))
    pd.DataFrame(rows).to_csv(RAW/"summary_metrics.csv", index=False)
    paired, summary = comparisons(frame)
    paired.to_csv(RAW/"paired_comparisons.csv", index=False)
    summary.to_csv(RAW/"comparison_summary.csv", index=False)
    decisions = {model: absolute_success(frame[frame.model == model]) for model in MODELS}
    advantages = {}
    for base in MODELS[1:]:
        a = frame[frame.model == MODELS[0]].set_index("seed")
        b = frame[frame.model == base].set_index("seed")
        advantages[base] = dict(
            spinor_ratio_of_means=float(a.spinor_rel_l2_gauge_aligned.mean()/b.spinor_rel_l2_gauge_aligned.mean()),
            current_ratio_of_means=float(a.current_rel_l2.mean()/b.current_rel_l2.mean()),
            both_mean_errors_lower=bool(a.spinor_rel_l2_gauge_aligned.mean() < b.spinor_rel_l2_gauge_aligned.mean()
                                       and a.current_rel_l2.mean() < b.current_rel_l2.mean()),
            paired_joint_spinor_current_wins=int(((a.spinor_rel_l2_gauge_aligned < b.spinor_rel_l2_gauge_aligned)
                                                   &(a.current_rel_l2 < b.current_rel_l2)).sum()),
            additional_advantage_threshold_used=False)
    decision = dict(status="FULL_DYNAMICS_SUCCESS" if decisions[MODELS[0]]["pass_check"] else "FULL_DYNAMICS_NOT_ESTABLISHED",
                    experiment="r9_wavepacket_periodic_potential", absolute_success=decisions,
                    A_advantages=advantages, thresholds=THRESHOLDS, population_sd_ddof=0,
                    final_checkpoint_only=True, optimization_steps_per_run=20000,
                    pretraining_causal_effect_isolated=False, rescue_models_added=False,
                    capacity_match_relative_difference=130/25734,
                    scope="known finite Fourier Gaussian initial spectrum in this weak periodic potential only")
    write_json(RAW/"decision.json", decision)
    cost_columns = ["model", "seed", "active_parameters", "frozen_parameters", "total_parameters", "carrier_steps",
        "coupled_steps", "carrier_training_seconds", "coupled_training_seconds", "training_seconds", "evaluation_seconds",
        "training_peak_rss_mib", "peak_rss_mib", "training_residual_batches", "training_carrier_unit_spinor_residual_points",
        "training_coupled_spinor_residual_points", "training_complex_scalar_residual_evaluations",
        "evaluation_coupled_spinor_residual_points", "evaluation_carrier_unit_spinor_residual_points", "cpu_affinity_mask"]
    frame[cost_columns].to_csv(RAW/"cost_and_residual_work.csv", index=False)
    return frame, decision


if __name__ == "__main__":
    print(json.dumps(aggregate()[1], indent=2))
