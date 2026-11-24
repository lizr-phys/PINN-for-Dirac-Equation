from __future__ import annotations
import numpy as np
import pandas as pd
from .r11_config import GROUPS, SEEDS, THRESHOLDS, EXPERIMENT

def absolute_success(frame):
    complete = len(frame) == 5 and sorted(frame.seed.tolist()) == list(SEEDS)
    means = {key: float(frame[key].mean()) for key in THRESHOLDS}
    checks = {key:means[key] >= bound if key == "fidelity_mean" else means[key] <= bound
              for key, bound in THRESHOLDS.items()}
    joint = np.ones(len(frame), dtype=bool)
    for key, bound in THRESHOLDS.items():
        joint &= frame[key].to_numpy() >= bound if key == "fidelity_mean" else frame[key].to_numpy() <= bound
    return dict(complete=complete, means=means, mean_checks=checks,
        joint_pass_count=int(joint.sum()), joint_pass_seeds=frame.seed.to_numpy()[joint].tolist(),
        pass_check=bool(complete and all(checks.values()) and joint.sum() >= 4))

def decision(frame):
    records = {endpoint:{g:absolute_success(frame[(frame.endpoint == endpoint) & (frame.model == g)])
               for g in GROUPS} for endpoint in ("final", "coupled_20000")}
    return dict(experiment=EXPERIMENT,
        status="INDEPENDENT_REPLICATION_SUCCESS" if records["final"][GROUPS[0]]["pass_check"]
            else "INDEPENDENT_REPLICATION_NOT_ESTABLISHED",
        absolute_success=records, primary_groups=list(GROUPS[:4]), thresholds=THRESHOLDS,
        population_sd_ddof=0, fresh_seed_replication=True, fixed_checkpoint_only=True,
        analytic_diagnostic_excluded_from_learned_success=True,
        pretraining_vs_freezing_single_factor_isolated=False,
        selective_reruns=False, additional_advantage_threshold=False,
        scope="known finite Fourier Gaussian initial spectrum, this fixed weak periodic potential")

def comparisons(frame):
    rows, summary = [], []
    keys = tuple(THRESHOLDS)+("density_rel_l2", "active_coherence_rel_l2",
        "branch_complex_coefficient_rel_l2", "residual_integral_fine")
    pairs = ((0, 1), (0, 2), (0, 3), (4, 0), (2, 1), (3, 1))
    for endpoint in ("final", "coupled_20000"):
        data = frame[frame.endpoint == endpoint]
        for ai, bi in pairs:
            a, b = (data[data.model == GROUPS[i]].set_index("seed").loc[list(SEEDS)] for i in (ai, bi))
            for key in keys:
                delta, ratio = a[key]-b[key], a[key]/b[key]
                for seed in SEEDS:
                    rows.append(dict(endpoint=endpoint, target=GROUPS[ai], baseline=GROUPS[bi],
                        metric=key, seed=seed, paired_difference=float(delta[seed]), actual_ratio=float(ratio[seed])))
                summary.append(dict(endpoint=endpoint, target=GROUPS[ai], baseline=GROUPS[bi], metric=key,
                    n=5, paired_difference_mean=float(delta.mean()),
                    paired_difference_population_sd=float(delta.std(ddof=0)),
                    ratio_of_means=float(a[key].mean()/b[key].mean()), mean_of_paired_ratios=float(ratio.mean()),
                    favorable_pair_count=int((delta > 0 if key == "fidelity_mean" else delta < 0).sum())))
    return pd.DataFrame(rows), pd.DataFrame(summary)
