from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from src.wavepacket_config import WavepacketConfig
from src.wavepacket_evaluate import (
    _branch_diagnostics,
    _circular_center_width,
    _heldout_diagnostics,
    _phase_diagnostics,
    compute_wavepacket_evaluation,
)
from src.wavepacket_models import SharedSpectralModel, SpectralOracle
from src.wavepacket_physics import (
    compute_spectral_basis,
    exact_spectral_state,
    gaussian_spectral_envelope,
    project_energy_branches,
    reconstruct_physical_grid,
    reference_energies,
    training_wave_numbers,
)
from src.wavepacket_shared_spectral_ltpb_decision import (
    ALL_MODELS,
    TARGET_MODEL,
    evaluate_wavepacket_preregistered_decision,
)


torch.set_default_dtype(torch.float64)
CFG = WavepacketConfig()


def _evaluate(
    model: torch.nn.Module,
    cfg: WavepacketConfig,
    *,
    model_name: str,
) -> tuple[dict[str, float | int | str | None], dict[str, np.ndarray]]:
    metrics, _, dense = compute_wavepacket_evaluation(
        model,
        cfg,
        model_name=model_name,
        seed=0,
        parameter_count=0,
        training_seconds=0.0,
        final_pde_loss=0.0,
        peak_rss_mb=0.0,
        peak_rss_increment_mb=0.0,
        derivative_batch_size=4096,
    )
    return metrics, dense


class GaugeShiftedOracle(torch.nn.Module):
    """Exact spectral state multiplied by one constant and/or common rate."""

    def __init__(
        self,
        cfg: WavepacketConfig,
        *,
        constant: float,
        rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.constant = float(constant)
        self.rate = float(rate)
        self.register_buffer("anchor", torch.zeros((), dtype=torch.float64))

    def forward(self, wave_numbers: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        exact = exact_spectral_state(wave_numbers, t, self.cfg)
        time = t.reshape(-1, 1)
        return exact * torch.exp(1j * (self.constant + self.rate * time))


class SmoothRateShiftedShared(SharedSpectralModel):
    """Test double whose output phase shift is visible only after projection."""

    def __init__(self, cfg: WavepacketConfig, *, rate_scale: float) -> None:
        super().__init__(cfg, initialization_seed=987_654)
        self.rate_scale = float(rate_scale)

    def forward(self, wave_numbers: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        k = wave_numbers.reshape(-1, 1)
        time = t.reshape(-1, 1)
        exact = exact_spectral_state(k, time, self.cfg)
        rate = self.rate_scale * (1.0 + k / self.cfg.wave_number_max)
        return exact * torch.exp(1j * rate * time)


def test_spectral_oracle_has_machine_precision_metrics_and_reference_branches() -> None:
    metrics, dense = _evaluate(SpectralOracle(CFG), CFG, model_name="spectral_oracle")

    for key in (
        "spinor_rel_l2_gauge_aligned",
        "density_rel_l2",
        "current_rel_l2",
        "spectral_coefficient_rel_l2_gauge_aligned",
        "spectral_occupation_rel_l2",
        "training_phase_slope_weighted_relative_error",
        "modal_amplitude_drift_weighted_rel_rms",
        "dense_physical_pde_residual_rms",
        "parseval_dense_rms_absolute_difference",
        "max_norm_drift",
    ):
        assert float(metrics[key]) < 2.0e-12, (key, metrics[key])
    assert abs(float(metrics["fidelity_mean"]) - 1.0) < 2.0e-14
    assert float(metrics["training_phase_min_valid_fraction"]) == 1.0
    assert metrics["heldout_phase_slope_weighted_relative_error"] is None

    assert abs(float(metrics["branch_plus_norm_mean"]) - 0.5) < 3.0e-13
    assert abs(float(metrics["branch_minus_norm_mean"]) - 0.5) < 3.0e-13
    assert abs(float(metrics["branch_plus_exact_velocity"]) - 0.3673318155) < 2.0e-9
    assert abs(float(metrics["branch_minus_exact_velocity"]) + 0.3673318155) < 2.0e-9
    assert abs(float(metrics["branch_plus_exact_final_center"]) - 1.8366638585) < 2.0e-9
    assert abs(float(metrics["branch_minus_exact_final_center"]) + 1.8366638585) < 2.0e-9
    assert abs(float(dense["branch_plus_width_exact"][0]) - 1.0053040292) < 2.0e-9
    assert abs(float(dense["branch_plus_width_exact"][-1]) - 1.0816978693) < 2.0e-9
    assert np.allclose(
        dense["branch_plus_width_exact"],
        dense["branch_minus_width_exact"],
        rtol=0.0,
        atol=2.0e-12,
    )


def test_one_constant_gauge_is_removed_but_a_common_wrong_frequency_is_not() -> None:
    cfg = replace(CFG, test_x_points=64, test_t_points=81)
    constant, _ = _evaluate(
        GaugeShiftedOracle(cfg, constant=0.73),
        cfg,
        model_name="spectral_oracle",
    )
    assert float(constant["spinor_rel_l2_gauge_aligned"]) < 3.0e-12
    assert float(constant["spinor_rel_l2_raw"]) > 0.5
    assert float(constant["training_phase_slope_weighted_relative_error"]) < 2.0e-12

    rate = 0.25
    drifting, dense = _evaluate(
        GaugeShiftedOracle(cfg, constant=-0.42, rate=rate),
        cfg,
        model_name="spectral_oracle",
    )
    assert float(drifting["spinor_rel_l2_gauge_aligned"]) > 0.2
    observed = dense["phase_slopes"]
    target = dense["phase_slope_targets"]
    assert np.max(np.abs((observed - target) - rate)) < 3.0e-11
    assert float(drifting["training_phase_slope_weighted_relative_error"]) > 0.04


def test_training_and_heldout_phases_use_projected_coefficients_and_initial_weights() -> None:
    times = np.linspace(0.0, CFG.t_max, 101)
    k = torch.tensor([1.0, 2.5])
    basis = compute_spectral_basis(k, CFG.mass)
    energies = np.asarray(reference_energies(k.numpy(), CFG.mass))
    targets = np.stack((-energies, energies), axis=1)
    initial = np.array([0.9, 0.1])
    rate_errors = np.array([[0.01, -0.02], [1.2, -0.8]])
    slopes = targets + rate_errors
    coefficients = initial[None, :, None] * np.exp(
        1j * times[:, None, None] * slopes[None, :, :]
    )
    exact_coefficients = initial[None, :, None] * np.exp(
        1j * times[:, None, None] * targets[None, :, :]
    )

    total_spectral = (
        torch.as_tensor(coefficients[:, :, 0:1], dtype=torch.complex128)
        * basis.u_plus.to(torch.complex128)[None]
        + torch.as_tensor(coefficients[:, :, 1:2], dtype=torch.complex128)
        * basis.u_minus.to(torch.complex128)[None]
    )
    projected = project_energy_branches(total_spectral, k, CFG.mass).numpy()
    diagnostics, fields = _phase_diagnostics(
        projected,
        exact_coefficients,
        times,
        targets,
        initial,
    )
    expected = float(
        np.sum(
            np.repeat(initial[:, None] ** 2, 2, axis=1)
            * np.abs(rate_errors)
            / np.abs(targets)
        )
        / np.sum(np.repeat(initial[:, None] ** 2, 2, axis=1))
    )
    assert np.max(np.abs(fields["phase_slopes"] - slopes)) < 2.0e-12
    assert abs(diagnostics["weighted_relative_error"] - expected) < 2.0e-13
    assert diagnostics["weighted_relative_error"] < float(
        np.mean(np.abs(rate_errors) / np.abs(targets))
    )

    shared = SmoothRateShiftedShared(CFG, rate_scale=0.01)
    heldout, heldout_dense = _heldout_diagnostics(
        shared,
        CFG,
        torch.linspace(0.0, CFG.t_max, 101),
        0.0,
        derivative_batch_size=4096,
    )
    midpoint = heldout_dense["heldout_wave_numbers"]
    midpoint_energy = np.asarray(reference_energies(midpoint, CFG.mass))
    midpoint_initial = (
        gaussian_spectral_envelope(torch.as_tensor(midpoint), CFG).numpy()
        / math.sqrt(2.0)
    )
    midpoint_rate = 0.01 * (1.0 + midpoint / CFG.wave_number_max)
    expected_heldout = float(
        np.sum(midpoint_initial**2 * midpoint_rate / midpoint_energy)
        / np.sum(midpoint_initial**2)
    )
    assert abs(
        float(heldout["heldout_phase_slope_weighted_relative_error"])
        - expected_heldout
    ) < 3.0e-12
    assert float(heldout["heldout_phase_min_valid_fraction"]) == 1.0
    assert heldout_dense["heldout_phase_slopes"].shape == (32, 2)


def test_circular_center_and_width_match_the_exact_branch_reference() -> None:
    x = CFG.x_min + CFG.length * torch.arange(CFG.test_x_points) / CFG.test_x_points
    times = torch.linspace(0.0, CFG.t_max, CFG.test_t_points)
    k = training_wave_numbers(CFG)
    t_mesh, k_mesh = torch.meshgrid(times, k, indexing="ij")
    spectral = exact_spectral_state(
        k_mesh.reshape(-1, 1), t_mesh.reshape(-1, 1), CFG
    ).reshape(CFG.test_t_points, CFG.mode_count, 2)
    coefficients = project_energy_branches(spectral, k, CFG.mass).numpy()
    branch_metrics, branch_fields = _branch_diagnostics(
        coefficients,
        coefficients,
        k,
        x,
        times.numpy(),
        CFG,
    )

    assert abs(branch_metrics["branch_plus_exact_velocity"] - 0.3673318155) < 2.0e-9
    assert abs(branch_metrics["branch_minus_exact_velocity"] + 0.3673318155) < 2.0e-9
    assert abs(branch_fields["branch_plus_center_exact"][0]) < 2.0e-12
    assert abs(branch_fields["branch_plus_center_exact"][-1] - 1.8366638585) < 2.0e-9
    assert abs(branch_fields["branch_minus_center_exact"][-1] + 1.8366638585) < 2.0e-9
    assert abs(branch_fields["branch_plus_width_exact"][0] - 1.0053040292) < 2.0e-9
    assert abs(branch_fields["branch_plus_width_exact"][-1] - 1.0816978693) < 2.0e-9

    translated_center = 0.48 * CFG.length
    width = 0.7
    wrapped_distance = (
        (x.numpy() - translated_center + 0.5 * CFG.length) % CFG.length
        - 0.5 * CFG.length
    )
    density = np.exp(-0.5 * (wrapped_distance / width) ** 2)[None]
    center, recovered_width, _ = _circular_center_width(
        density, x.numpy(), CFG.length, CFG.x_min
    )
    assert abs(center[0] - translated_center) < 3.0e-8
    assert abs(recovered_width[0] - width) < 3.0e-8


def _synthetic_runs() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model in ALL_MODELS:
        for seed in range(5):
            target = model == TARGET_MODEL
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "spinor_rel_l2_gauge_aligned": 0.05 if target else 0.50,
                    "current_rel_l2": 0.10 if target else 1.00,
                    "fidelity_mean": 0.99 if target else 0.50,
                    "training_phase_slope_weighted_relative_error": 0.009 if target else 1.0,
                    "training_phase_min_valid_fraction": 0.90 if target else 1.0,
                    "heldout_phase_slope_weighted_relative_error": 0.019 if target else math.nan,
                    "heldout_phase_min_valid_fraction": 0.90 if target else math.nan,
                    "branch_plus_center_rmse": 0.25 if target else 1.0,
                    "branch_minus_center_rmse": 0.25 if target else 1.0,
                    "branch_plus_velocity_relative_error": 0.099 if target else 1.0,
                    "branch_minus_velocity_relative_error": 0.099 if target else 1.0,
                    "branch_plus_velocity": 0.36 if target else 0.0,
                    "branch_minus_velocity": -0.36 if target else 0.0,
                    "branch_plus_width_rel_l2": 0.099 if target else 1.0,
                    "branch_minus_width_rel_l2": 0.099 if target else 1.0,
                    "branch_plus_final_center": 1.8 if target else 0.0,
                    "branch_minus_final_center": -1.8 if target else 0.0,
                    "branch_plus_final_width_change": 0.07 if target else 0.0,
                    "branch_minus_final_width_change": 0.07 if target else 0.0,
                    "branch_plus_exact_final_width_change": 0.07,
                    "branch_minus_exact_final_width_change": 0.07,
                    "branch_minimum_norm": 0.50 if target else 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_decision_boundaries_are_frozen_and_midpoint_or_branch_failure_blocks_pass() -> None:
    passing = evaluate_wavepacket_preregistered_decision(_synthetic_runs())
    assert passing["overall_pass"] is True
    assert passing["paired_joint_pass_count"] == 5
    assert passing["absolute_thresholds"]["spinor_mean_le_0p05"]["pass"] is True
    assert passing["absolute_thresholds"]["current_mean_le_0p10"]["pass"] is True
    assert passing["absolute_thresholds"]["fidelity_mean_ge_0p99"]["pass"] is True
    assert passing["wavepacket_branch_dynamics"]["checks"]["plus_center_rmse_le_0p25"] is True

    training_boundary = _synthetic_runs()
    training_boundary.loc[
        training_boundary.model == TARGET_MODEL,
        "training_phase_slope_weighted_relative_error",
    ] = 0.01
    training_failure = evaluate_wavepacket_preregistered_decision(training_boundary)
    assert training_failure["training_phase_slope"]["pass"] is False
    assert training_failure["overall_pass"] is False

    midpoint_boundary = _synthetic_runs()
    midpoint_boundary.loc[
        midpoint_boundary.model == TARGET_MODEL,
        "heldout_phase_slope_weighted_relative_error",
    ] = 0.02
    midpoint_failure = evaluate_wavepacket_preregistered_decision(midpoint_boundary)
    assert midpoint_failure["heldout_midpoint_phase_slope"]["pass"] is False
    assert midpoint_failure["continuous_spectral_interpolation_claim_allowed"] is False
    assert midpoint_failure["overall_pass"] is False

    branch_boundary = _synthetic_runs()
    branch_boundary.loc[
        branch_boundary.model == TARGET_MODEL,
        "branch_plus_velocity_relative_error",
    ] = 0.10
    branch_failure = evaluate_wavepacket_preregistered_decision(branch_boundary)
    assert branch_failure["wavepacket_branch_dynamics"]["pass"] is False
    assert branch_failure["overall_pass"] is False

    vanished_branch = _synthetic_runs()
    vanished_branch.loc[
        (vanished_branch.model == TARGET_MODEL) & (vanished_branch.seed == 4),
        "branch_minimum_norm",
    ] = 0.0
    norm_failure = evaluate_wavepacket_preregistered_decision(vanished_branch)
    assert norm_failure["wavepacket_branch_dynamics"]["checks"][
        "identifiable_all_seeds"
    ] is False
    assert norm_failure["overall_pass"] is False
