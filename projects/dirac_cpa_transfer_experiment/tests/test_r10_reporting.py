import numpy as np
import pandas as pd
import pytest
from src.aggregate_r10_budget import paired_statistics, STAGES
from src.r9_config import THRESHOLDS
from src.r10_config import SEEDS


def test_paired_summary_uses_population_sd_and_distinguishes_two_ratios():
    # Known unequal baseline magnitudes make ratio-of-means != mean-of-ratios.
    before, after = np.array([1., 2., 3., 4., 5.]), np.array([2., 1., 4., 3., 8.])
    extra = ("density_rel_l2", "active_coherence_rel_l2", "branch_population_rel_l2",
             "branch_complex_coefficient_rel_l2", "residual_integral_fine", "full_physical_residual_integral")
    rows = []
    for stage, values in zip(STAGES, (before, after)):
        for seed, value in zip(SEEDS, values):
            rows.append(dict(stage=stage, seed=seed, **{k: value for k in tuple(THRESHOLDS)+extra}))
    pairs, summary = paired_statistics(pd.DataFrame(rows))
    item = summary[summary.metric == "max_norm_drift"].iloc[0]
    assert item.paired_difference_mean == pytest.approx(.6)
    assert item.paired_difference_population_sd == pytest.approx(1.4966629547095767)
    assert item.ratio_of_means == pytest.approx(1.2)
    assert item.mean_of_paired_ratios == pytest.approx(1.2366666666666666)
    assert item.favorable_pair_count == 2
    fidelity = summary[summary.metric == "fidelity_mean"].iloc[0]
    assert fidelity.favorable_pair_count == 3
    assert pairs[pairs.metric == "max_norm_drift"].paired_difference.tolist() == [1., -1., 1., -1., 3.]
