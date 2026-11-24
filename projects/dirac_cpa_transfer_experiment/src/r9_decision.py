from __future__ import annotations
import numpy as np
import pandas as pd
from .r9_config import THRESHOLDS, SEEDS, MODELS


def absolute_success(frame):
    complete = len(frame) == 5 and sorted(frame.seed.tolist()) == list(SEEDS)
    means = {key: float(frame[key].mean()) for key in THRESHOLDS}
    mean_checks = {key: (means[key] >= bound if key == "fidelity_mean" else means[key] <= bound)
                   for key, bound in THRESHOLDS.items()}
    joint = np.ones(len(frame), dtype=bool)
    for key, bound in THRESHOLDS.items():
        joint &= frame[key].to_numpy() >= bound if key == "fidelity_mean" else frame[key].to_numpy() <= bound
    return dict(complete=complete, mean_checks=mean_checks, means=means, joint_pass_count=int(joint.sum()),
                joint_pass_seeds=frame.seed.to_numpy()[joint].tolist(),
                pass_check=bool(complete and all(mean_checks.values()) and joint.sum() >= 4))


def comparisons(frame):
    rows, summary = [], []
    metrics = tuple(THRESHOLDS)+("density_rel_l2", "active_coherence_rel_l2", "branch_complex_coefficient_rel_l2")
    for target, base in ((MODELS[0], MODELS[1]), (MODELS[0], MODELS[2]), (MODELS[2], MODELS[1])):
        a = frame[frame.model == target].set_index("seed")
        b = frame[frame.model == base].set_index("seed")
        for key in metrics:
            delta = (a[key]-b[key]).loc[list(SEEDS)]
            ratio = (a[key]/b[key]).loc[list(SEEDS)]
            for seed in SEEDS:
                rows.append(dict(target=target, baseline=base, metric=key, seed=seed,
                                 paired_difference=float(delta[seed]), actual_ratio=float(ratio[seed])))
            summary.append(dict(target=target, baseline=base, metric=key, n=5,
                paired_difference_mean=float(delta.mean()), paired_difference_population_sd=float(delta.std(ddof=0)),
                mean_of_paired_ratios=float(ratio.mean()), ratio_of_means=float(a[key].mean()/b[key].mean()),
                favorable_pair_count=int((delta > 0 if key == "fidelity_mean" else delta < 0).sum())))
    return pd.DataFrame(rows), pd.DataFrame(summary)
