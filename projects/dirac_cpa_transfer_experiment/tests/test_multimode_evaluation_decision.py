from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from src.multimode_config import MODE_KEYS, MULTIMODE_MODELS, MultimodeConfig
from src.multimode_evaluate import compute_multimode_evaluation
from src.multimode_ltpb_decision import evaluate_preregistered_decision
from src.multimode_physics import multimode_exact_solution


class GaugeShiftedExact(torch.nn.Module):
    def __init__(self, cfg: MultimodeConfig, *, constant: float, rate: float = 0.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.constant = constant
        self.rate = rate
        self.register_buffer("anchor", torch.zeros((), dtype=torch.float64))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        exact = multimode_exact_solution(
            x,
            t,
            mass=self.cfg.mass,
            wave_numbers=self.cfg.wave_numbers,
            length=self.cfg.length,
        )
        return exact * torch.exp(1j * (self.constant + self.rate * t))


def _evaluate(model: torch.nn.Module, cfg: MultimodeConfig) -> dict[str, float | int | str]:
    metrics, _, _ = compute_multimode_evaluation(
        model,
        cfg,
        model_name="test",
        seed=0,
        parameter_count=0,
        training_seconds=0.0,
        final_pde_loss=0.0,
        derivative_batch_size=512,
    )
    return metrics


def test_one_constant_global_gauge_is_removed_but_a_wrong_common_frequency_is_not() -> None:
    cfg = replace(MultimodeConfig(), test_x_points=32, test_t_points=41)
    constant = _evaluate(GaugeShiftedExact(cfg, constant=0.73), cfg)
    assert float(constant["spinor_rel_l2_gauge_aligned"]) < 2.0e-12
    assert float(constant["spinor_rel_l2_raw"]) > 0.5
    for key in MODE_KEYS:
        assert float(constant[f"{key}_phase_slope_relative_error"]) < 2.0e-12

    rate = 0.25
    drifting = _evaluate(GaugeShiftedExact(cfg, constant=-0.42, rate=rate), cfg)
    assert float(drifting["spinor_rel_l2_gauge_aligned"]) > 0.2
    for key in MODE_KEYS:
        target = float(drifting[f"{key}_phase_slope_target"])
        observed = float(drifting[f"{key}_phase_slope"])
        assert abs((observed - target) - rate) < 2.0e-10


def _synthetic_runs(*, phase_errors: tuple[float, float, float, float]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model in MULTIMODE_MODELS:
        for seed in range(5):
            if model == "multi_ltpb":
                row: dict[str, float | int | str] = {
                    "model": model,
                    "seed": seed,
                    "spinor_rel_l2_gauge_aligned": 0.05,
                    "current_rel_l2": 0.10,
                    "fidelity_mean": 0.99,
                }
                for key, error in zip(MODE_KEYS, phase_errors, strict=True):
                    row[f"{key}_phase_slope_relative_error"] = error
                    row[f"{key}_phase_valid_fraction"] = 0.90
            else:
                row = {
                    "model": model,
                    "seed": seed,
                    "spinor_rel_l2_gauge_aligned": 0.50,
                    "current_rel_l2": 1.00,
                    "fidelity_mean": 0.50,
                }
                for key in MODE_KEYS:
                    row[f"{key}_phase_slope_relative_error"] = 1.0
                    row[f"{key}_phase_valid_fraction"] = 1.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_preregistered_decision_uses_frozen_inclusive_and_strict_boundaries() -> None:
    decision = evaluate_preregistered_decision(
        _synthetic_runs(phase_errors=(0.009, 0.009, 0.009, 0.02))
    )
    assert decision["overall_pass"] is True
    assert decision["paired_joint_pass_count"] == 5
    assert decision["phase_slope_pass_count"] == 3

    strict_failure = evaluate_preregistered_decision(
        _synthetic_runs(phase_errors=(0.01, 0.01, 0.01, 0.009))
    )
    assert strict_failure["phase_slope_pass_count"] == 1
    assert strict_failure["overall_pass"] is False


def test_preregistered_decision_requires_four_joint_paired_seeds() -> None:
    runs = _synthetic_runs(phase_errors=(0.0, 0.0, 0.0, 0.0))
    # Make exactly one seed fail the paired current advantage while preserving
    # the target mean absolute thresholds.
    index = (runs.model == "multi_ltpb") & (runs.seed == 4)
    runs.loc[index, "current_rel_l2"] = 0.11
    runs.loc[(runs.model != "multi_ltpb") & (runs.seed == 4), "current_rel_l2"] = 1.00
    decision = evaluate_preregistered_decision(runs)
    assert decision["paired_joint_pass_count"] == 4
    assert decision["paired_seed_rule_pass"] is True
