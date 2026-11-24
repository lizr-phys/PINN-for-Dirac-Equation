from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, RunSpec, write_json
from .models import DiracModel, LearnedTwoPhaseBranchCPA, TwoChannelDiracModel
from .physics import (
    analytic_total_energy,
    compute_eigensystem,
    continuity_from_derivatives,
    density_current,
    energy_density,
    exact_solution,
    spinor_derivatives,
)


def _relative_l2(predicted: np.ndarray, exact: np.ndarray) -> float:
    denominator = np.linalg.norm(exact.ravel())
    return float(np.linalg.norm((predicted - exact).ravel()) / denominator)


def _circular_difference(predicted: np.ndarray, exact: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (predicted - exact)))


def _linear_fit(t: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    """Return ordinary-least-squares slope, intercept, and R^2."""
    slope, intercept = np.polyfit(t, values, deg=1)
    fitted = slope * t + intercept
    residual_sum = float(np.sum((values - fitted) ** 2))
    total_sum = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = 1.0 if total_sum == 0.0 and residual_sum == 0.0 else (
        0.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    )
    return float(slope), float(intercept), float(r_squared)


def _weighted_mean(values: np.ndarray, weights: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    numerator = np.sum(weights * values, axis=axis)
    denominator = np.sum(weights, axis=axis)
    return numerator / np.maximum(denominator, 1.0e-30)


def _internal_channel_diagnostics(
    coefficients: np.ndarray,
    t_values: np.ndarray,
    initial_coefficients: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Return gauge-robust amplitude/phase-complexity diagnostics.

    ``coefficients`` has shape ``(n_t, n_x, 2)``.  Phases are unwrapped along
    time at fixed x.  All phase-complexity summaries are weighted by channel
    occupation, suppressing the undefined phase of a coefficient near zero.
    """

    if coefficients.ndim != 3 or coefficients.shape[2] != 2:
        raise ValueError(f"Expected channel coefficient shape (n_t,n_x,2), got {coefficients.shape}")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Internal channel coefficients contain non-finite values")
    amplitude = np.abs(coefficients)
    occupation_weight = amplitude**2
    phase_unwrapped = np.unwrap(np.angle(coefficients), axis=0)
    edge_order = 2 if len(t_values) >= 3 else 1
    phase_velocity = np.gradient(
        phase_unwrapped, t_values, axis=0, edge_order=edge_order
    )
    phase_curvature = np.gradient(
        phase_velocity, t_values, axis=0, edge_order=edge_order
    )
    time_steps = np.diff(t_values).reshape(-1, 1, 1)
    phase_links = np.angle(np.conj(coefficients[:-1]) * coefficients[1:])
    link_velocity = phase_links / time_steps
    link_weight = amplitude[:-1] * amplitude[1:]
    link_midpoints = 0.5 * (t_values[:-1] + t_values[1:])
    curvature_time_steps = np.diff(link_midpoints).reshape(-1, 1, 1)
    link_curvature = np.diff(link_velocity, axis=0) / curvature_time_steps

    initial_amplitude = np.abs(initial_coefficients).reshape(1, 1, 2)
    amplitude_drift = amplitude - initial_amplitude
    amplitude_drift_denominator = float(np.sum(initial_amplitude**2) * amplitude.shape[0] * amplitude.shape[1])
    combined_amplitude_drift = float(
        np.sqrt(
            np.sum(amplitude_drift**2)
            / max(amplitude_drift_denominator, 1.0e-30)
        )
    )

    metrics: dict[str, float] = {
        "amplitude_drift_rel_rms": combined_amplitude_drift,
    }
    time_payload: dict[str, np.ndarray] = {}
    all_velocity_residuals: list[np.ndarray] = []
    all_curvatures: list[np.ndarray] = []
    all_velocity_weights: list[np.ndarray] = []
    all_curvature_weights: list[np.ndarray] = []
    slope_variances: list[float] = []
    slope_weights: list[float] = []

    for channel in range(2):
        channel_number = channel + 1
        weights = occupation_weight[:, :, channel]
        phases = phase_unwrapped[:, :, channel]
        velocities = phase_velocity[:, :, channel]
        link_velocities = link_velocity[:, :, channel]
        channel_link_weight = link_weight[:, :, channel]
        channel_link_curvature = link_curvature[:, :, channel]
        initial = float(initial_amplitude[0, 0, channel])
        phase_amplitude_threshold = max(1.0e-12, 1.0e-6 * abs(initial))
        valid_link = (
            (amplitude[:-1, :, channel] > phase_amplitude_threshold)
            & (amplitude[1:, :, channel] > phase_amplitude_threshold)
            & (np.abs(phase_links[:, :, channel]) < 0.9 * math.pi)
        )
        velocity_weights = np.where(valid_link, channel_link_weight, 0.0)
        valid_curvature = valid_link[:-1] & valid_link[1:]
        curvature_weights = np.where(
            valid_curvature,
            np.sqrt(velocity_weights[:-1] * velocity_weights[1:]),
            0.0,
        )
        spatial_weights = np.sum(velocity_weights, axis=0)
        slopes = np.sum(velocity_weights * link_velocities, axis=0) / np.maximum(
            spatial_weights, 1.0e-30
        )
        slope_mean = float(
            np.sum(spatial_weights * slopes) / max(float(np.sum(spatial_weights)), 1.0e-30)
        )
        slope_std = float(
            np.sqrt(
                np.sum(spatial_weights * (slopes - slope_mean) ** 2)
                / max(float(np.sum(spatial_weights)), 1.0e-30)
            )
        )
        velocity_fluctuation = float(
            np.sqrt(
                np.sum(velocity_weights * (link_velocities - slope_mean) ** 2)
                / max(float(np.sum(velocity_weights)), 1.0e-30)
            )
        )
        curvature_rms = float(
            np.sqrt(
                np.sum(curvature_weights * channel_link_curvature**2)
                / max(float(np.sum(curvature_weights)), 1.0e-30)
            )
        )
        amplitude_drift_rel = float(
            np.sqrt(np.mean(amplitude_drift[:, :, channel] ** 2))
            / max(abs(initial), 1.0e-30)
        )

        prefix = f"channel_{channel_number}"
        metrics.update(
            {
                f"{prefix}_amplitude_drift_rel_rms": amplitude_drift_rel,
                f"{prefix}_phase_slope_mean": slope_mean,
                f"{prefix}_phase_slope_spatial_std": slope_std,
                f"{prefix}_phase_velocity_fluctuation_weighted_rms": velocity_fluctuation,
                f"{prefix}_phase_curvature_weighted_rms": curvature_rms,
                f"{prefix}_phase_valid_link_fraction": float(np.mean(valid_link)),
                f"{prefix}_phase_valid_curvature_fraction": float(
                    np.mean(valid_curvature)
                ),
                f"{prefix}_near_zero_fraction": float(
                    np.mean(amplitude[:, :, channel] <= phase_amplitude_threshold)
                ),
                f"{prefix}_max_abs_phase_link": float(
                    np.max(np.abs(phase_links[:, :, channel]))
                ),
            }
        )
        time_payload.update(
            {
                f"{prefix}_amplitude_spatial_mean": np.mean(amplitude[:, :, channel], axis=1),
                f"{prefix}_phase_spatial_weighted_mean": _weighted_mean(
                    phases, weights, axis=1
                ),
                f"{prefix}_phase_velocity_spatial_weighted_mean": _weighted_mean(
                    velocities, weights, axis=1
                ),
            }
        )
        all_velocity_residuals.append(link_velocities - slope_mean)
        all_curvatures.append(channel_link_curvature)
        all_velocity_weights.append(velocity_weights)
        all_curvature_weights.append(curvature_weights)
        slope_variances.append(slope_std**2)
        slope_weights.append(float(np.sum(spatial_weights)))

    stacked_velocity_weights = np.stack(all_velocity_weights, axis=2)
    stacked_curvature_weights = np.stack(all_curvature_weights, axis=2)
    stacked_velocity_residuals = np.stack(all_velocity_residuals, axis=2)
    stacked_curvatures = np.stack(all_curvatures, axis=2)
    metrics.update(
        {
            "phase_velocity_fluctuation_weighted_rms": float(
                np.sqrt(
                    np.sum(stacked_velocity_weights * stacked_velocity_residuals**2)
                    / max(float(np.sum(stacked_velocity_weights)), 1.0e-30)
                )
            ),
            "phase_curvature_weighted_rms": float(
                np.sqrt(
                    np.sum(stacked_curvature_weights * stacked_curvatures**2)
                    / max(float(np.sum(stacked_curvature_weights)), 1.0e-30)
                )
            ),
            "phase_slope_spatial_std_weighted": float(
                np.sqrt(
                    np.sum(np.asarray(slope_weights) * np.asarray(slope_variances))
                    / max(float(np.sum(slope_weights)), 1.0e-30)
                )
            ),
        }
    )
    fields = {
        "coefficient": coefficients,
        "amplitude": amplitude,
        "phase_unwrapped": phase_unwrapped,
        "phase_velocity": phase_velocity,
        "phase_curvature": phase_curvature,
        **time_payload,
    }
    return metrics, fields


def _batched_derivatives(
    model: DiracModel,
    x_flat: torch.Tensor,
    t_flat: torch.Tensor,
    cfg: ExperimentConfig,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    continuity_parts: list[np.ndarray] = []
    energy_parts: list[np.ndarray] = []
    for start in range(0, x_flat.shape[0], batch_size):
        stop = min(start + batch_size, x_flat.shape[0])
        x = x_flat[start:stop].detach().clone().requires_grad_(True)
        t = t_flat[start:stop].detach().clone().requires_grad_(True)
        psi = model(x, t)
        psi_x, psi_t = spinor_derivatives(psi, x, t, create_graph=False)
        continuity = continuity_from_derivatives(psi, psi_x, psi_t)
        e_density = energy_density(psi, psi_x, mass=cfg.mass)
        continuity_parts.append(continuity.detach().cpu().numpy())
        energy_parts.append(e_density.detach().cpu().numpy())
    return np.concatenate(continuity_parts), np.concatenate(energy_parts)


def evaluate_model(
    model: DiracModel,
    cfg: ExperimentConfig,
    spec: RunSpec,
    run_dir: Path,
    *,
    parameter_count: int,
    training_seconds: float,
    final_losses: dict[str, float],
) -> dict[str, float | str | int]:
    model.eval()
    device = next(model.parameters()).device
    x_values = cfg.x_min + cfg.length * torch.arange(
        cfg.test_x_points, device=device, dtype=torch.float64
    ) / cfg.test_x_points
    t_values = torch.linspace(
        0.0, cfg.t_max, cfg.test_t_points, device=device, dtype=torch.float64
    )
    t_mesh, x_mesh = torch.meshgrid(t_values, x_values, indexing="ij")
    x_flat = x_mesh.reshape(-1, 1)
    t_flat = t_mesh.reshape(-1, 1)
    with torch.no_grad():
        predicted = model(x_flat, t_flat)
        exact = exact_solution(
            x_flat,
            t_flat,
            mass=cfg.mass,
            wave_number=cfg.wave_number,
            eta=spec.eta,
            eigensystem=model.eigensystem,
        )
        rho_predicted, current_predicted = density_current(predicted)
        rho_exact, current_exact = density_current(exact)
        branch_fields = (
            model.branch_fields(x_flat, t_flat)
            if isinstance(model, LearnedTwoPhaseBranchCPA)
            else None
        )
        internal_coefficients_flat = (
            model.channel_coefficients(x_flat, t_flat)
            if isinstance(model, TwoChannelDiracModel)
            else None
        )
        internal_basis = (
            model.channel_basis_matrix.detach().cpu().numpy()
            if isinstance(model, TwoChannelDiracModel)
            else None
        )
        internal_initial = (
            model.initial_channel_coefficients.detach().cpu().numpy()
            if isinstance(model, TwoChannelDiracModel)
            else None
        )

    continuity_flat, energy_density_flat = _batched_derivatives(
        model, x_flat, t_flat, cfg
    )
    shape = (cfg.test_t_points, cfg.test_x_points)
    predicted_np = predicted.detach().cpu().numpy().reshape(*shape, 2)
    exact_np = exact.detach().cpu().numpy().reshape(*shape, 2)
    rho_predicted_np = rho_predicted.detach().cpu().numpy().reshape(shape)
    rho_exact_np = rho_exact.detach().cpu().numpy().reshape(shape)
    current_predicted_np = current_predicted.detach().cpu().numpy().reshape(shape)
    current_exact_np = current_exact.detach().cpu().numpy().reshape(shape)
    continuity_np = continuity_flat.reshape(shape)
    energy_density_np = energy_density_flat.reshape(shape)
    x_np = x_values.detach().cpu().numpy()
    t_np = t_values.detach().cpu().numpy()

    norms = cfg.length * rho_predicted_np.mean(axis=1)
    exact_norms = cfg.length * rho_exact_np.mean(axis=1)
    energies = cfg.length * energy_density_np.mean(axis=1)
    exact_energy = analytic_total_energy(cfg.energy, spec.eta)
    continuity_rms_t = np.sqrt(np.mean(continuity_np**2, axis=1))

    overlaps = cfg.length * np.mean(
        np.sum(np.conj(exact_np) * predicted_np, axis=2), axis=1
    )
    pred_norms_for_fidelity = cfg.length * np.mean(
        np.sum(np.abs(predicted_np) ** 2, axis=2), axis=1
    )
    exact_norms_for_fidelity = cfg.length * np.mean(
        np.sum(np.abs(exact_np) ** 2, axis=2), axis=1
    )
    fidelity_t = np.abs(overlaps) ** 2 / (
        pred_norms_for_fidelity * exact_norms_for_fidelity + 1.0e-30
    )

    relative_phase_predicted = np.angle(np.conj(predicted_np[:, :, 0]) * predicted_np[:, :, 1])
    relative_phase_exact = np.angle(np.conj(exact_np[:, :, 0]) * exact_np[:, :, 1])
    exact_product_magnitude = np.abs(np.conj(exact_np[:, :, 0]) * exact_np[:, :, 1])
    mask_threshold = 1.0e-8 * float(np.max(exact_product_magnitude))
    phase_mask = exact_product_magnitude > mask_threshold
    phase_difference = _circular_difference(relative_phase_predicted, relative_phase_exact)
    phase_selected = phase_difference[phase_mask]

    eig = compute_eigensystem(cfg.mass, cfg.wave_number)
    u_plus = eig.u_plus.detach().cpu().numpy()
    u_minus = eig.u_minus.detach().cpu().numpy()
    demodulated = np.exp(-1j * cfg.wave_number * x_np)[None, :, None] * predicted_np
    coefficient_plus = math.sqrt(cfg.length) * np.mean(
        np.sum(np.conj(u_plus)[None, None, :] * demodulated, axis=2), axis=1
    )
    coefficient_minus = math.sqrt(cfg.length) * np.mean(
        np.sum(np.conj(u_minus)[None, None, :] * demodulated, axis=2), axis=1
    )
    coefficient_plus_exact = np.exp(-1j * cfg.energy * t_np) / math.sqrt(
        1.0 + spec.eta**2
    )
    coefficient_minus_exact = spec.eta * np.exp(1j * cfg.energy * t_np) / math.sqrt(
        1.0 + spec.eta**2
    )
    coefficient_predicted = np.column_stack((coefficient_plus, coefficient_minus))
    coefficient_exact = np.column_stack(
        (coefficient_plus_exact, coefficient_minus_exact)
    )
    occupation_predicted = np.abs(coefficient_predicted) ** 2
    occupation_exact = np.abs(coefficient_exact) ** 2

    internal_predicted: np.ndarray | None = None
    internal_exact: np.ndarray | None = None
    internal_predicted_metrics: dict[str, float] = {}
    internal_exact_metrics: dict[str, float] = {}
    internal_predicted_fields: dict[str, np.ndarray] = {}
    internal_exact_fields: dict[str, np.ndarray] = {}
    if internal_coefficients_flat is not None:
        if internal_basis is None or internal_initial is None:
            raise RuntimeError("Two-channel model did not expose basis and initial coefficients")
        internal_predicted = (
            internal_coefficients_flat.detach().cpu().numpy().reshape(*shape, 2)
        )
        demodulated_exact = (
            np.exp(-1j * cfg.wave_number * x_np)[None, :, None] * exact_np
        )
        internal_exact = np.einsum(
            "txa,aj->txj", demodulated_exact, np.conj(internal_basis)
        )
        internal_predicted_metrics, internal_predicted_fields = (
            _internal_channel_diagnostics(
                internal_predicted, t_np, internal_initial
            )
        )
        internal_exact_metrics, internal_exact_fields = _internal_channel_diagnostics(
            internal_exact, t_np, internal_initial
        )

    final_spinor_l2 = _relative_l2(predicted_np[-1], exact_np[-1])
    metrics: dict[str, float | str | int] = {
        "experiment": spec.experiment,
        "case": spec.case,
        "eta": spec.eta,
        "model": spec.model,
        "seed": spec.seed,
        "lambda_struct": spec.lambda_struct,
        "steps": spec.steps,
        "parameter_count": parameter_count,
        "training_seconds": training_seconds,
        "spinor_rel_l2": _relative_l2(predicted_np, exact_np),
        "final_spinor_rel_l2": final_spinor_l2,
        "rho_rel_l2": _relative_l2(rho_predicted_np, rho_exact_np),
        "current_rel_l2": _relative_l2(current_predicted_np, current_exact_np),
        "continuity_rms": float(np.sqrt(np.mean(continuity_np**2))),
        "continuity_normalized_rms": float(
            np.sqrt(np.mean(continuity_np**2))
            / (cfg.energy * cfg.reference_density)
        ),
        "max_norm_drift": float(np.max(np.abs(norms - 1.0))),
        "max_abs_energy_drift": float(np.max(np.abs(energies - exact_energy))),
        "fidelity_mean": float(np.mean(fidelity_t)),
        "fidelity_min": float(np.min(fidelity_t)),
        "fidelity_final": float(fidelity_t[-1]),
        "relative_phase_mae_rad": float(np.mean(np.abs(phase_selected))),
        "relative_phase_rmse_rad": float(np.sqrt(np.mean(phase_selected**2))),
        "relative_phase_mask_fraction": float(np.mean(phase_mask)),
        "branch_coefficient_rel_l2": _relative_l2(
            coefficient_predicted, coefficient_exact
        ),
        "branch_occupation_rmse": float(
            np.sqrt(np.mean((occupation_predicted - occupation_exact) ** 2))
        ),
        "branch_plus_coefficient_abs_rms": float(
            np.sqrt(np.mean(np.abs(coefficient_plus - coefficient_plus_exact) ** 2))
        ),
        "negative_branch_coefficient_abs_rms": float(
            np.sqrt(np.mean(np.abs(coefficient_minus - coefficient_minus_exact) ** 2))
        ),
        "branch_plus_occupation_rmse": float(
            np.sqrt(np.mean((occupation_predicted[:, 0] - occupation_exact[:, 0]) ** 2))
        ),
        "branch_minus_occupation_rmse": float(
            np.sqrt(np.mean((occupation_predicted[:, 1] - occupation_exact[:, 1]) ** 2))
        ),
        "target_mode_occupation_defect_mean": float(
            np.mean(np.abs(norms - np.sum(occupation_predicted, axis=1)))
        ),
    }
    metrics.update({f"final_loss_{key}": value for key, value in final_losses.items()})
    if internal_predicted is not None and internal_exact is not None:
        metrics.update(
            {
                f"internal_{key}": value
                for key, value in internal_predicted_metrics.items()
            }
        )
        metrics.update(
            {
                f"internal_exact_{key}": value
                for key, value in internal_exact_metrics.items()
            }
        )
        metrics.update(
            {
                "internal_phase_velocity_fluctuation_over_energy": internal_predicted_metrics[
                    "phase_velocity_fluctuation_weighted_rms"
                ]
                / cfg.energy,
                "internal_exact_phase_velocity_fluctuation_over_energy": internal_exact_metrics[
                    "phase_velocity_fluctuation_weighted_rms"
                ]
                / cfg.energy,
                "internal_phase_curvature_over_energy_squared": internal_predicted_metrics[
                    "phase_curvature_weighted_rms"
                ]
                / (cfg.energy**2),
                "internal_exact_phase_curvature_over_energy_squared": internal_exact_metrics[
                    "phase_curvature_weighted_rms"
                ]
                / (cfg.energy**2),
            }
        )
        reconstructed = np.einsum(
            "txj,aj->txa", internal_predicted, internal_basis
        )
        demodulated_predicted = (
            np.exp(-1j * cfg.wave_number * x_np)[None, :, None] * predicted_np
        )
        gram = np.conj(internal_basis).T @ internal_basis
        metrics.update(
            {
                "internal_channel_coefficient_rel_l2": _relative_l2(
                    internal_predicted, internal_exact
                ),
                "internal_amplitude_rel_l2": _relative_l2(
                    np.abs(internal_predicted), np.abs(internal_exact)
                ),
                "internal_reconstruction_max_abs": float(
                    np.max(np.abs(reconstructed - demodulated_predicted))
                ),
                "internal_basis_orthogonality_max_abs": float(
                    np.max(np.abs(gram - np.eye(2)))
                ),
                "internal_initial_coefficient_min_abs": float(
                    np.min(np.abs(internal_initial))
                ),
            }
        )

    index_x0 = int(np.argmin(np.abs(x_np)))
    time_series_payload: dict[str, np.ndarray | float] = {
            "t": t_np,
            "norm_predicted": norms,
            "norm_exact": exact_norms,
            "energy_predicted": energies,
            "energy_exact": exact_energy,
            "continuity_rms": continuity_rms_t,
            "current_x0_predicted": current_predicted_np[:, index_x0],
            "current_x0_exact": current_exact_np[:, index_x0],
            "relative_phase_x0_predicted": relative_phase_predicted[:, index_x0],
            "relative_phase_x0_exact": relative_phase_exact[:, index_x0],
            "fidelity": fidelity_t,
            "c_plus_predicted_real": coefficient_plus.real,
            "c_plus_predicted_imag": coefficient_plus.imag,
            "c_plus_exact_real": coefficient_plus_exact.real,
            "c_plus_exact_imag": coefficient_plus_exact.imag,
            "c_minus_predicted_real": coefficient_minus.real,
            "c_minus_predicted_imag": coefficient_minus.imag,
            "c_minus_exact_real": coefficient_minus_exact.real,
            "c_minus_exact_imag": coefficient_minus_exact.imag,
            "occupation_plus_predicted": occupation_predicted[:, 0],
            "occupation_plus_exact": occupation_exact[:, 0],
            "occupation_minus_predicted": occupation_predicted[:, 1],
            "occupation_minus_exact": occupation_exact[:, 1],
    }

    dense_grid_payload: dict[str, np.ndarray] = {
        "x": x_np,
        "t": t_np,
        "psi_predicted": predicted_np,
        "psi_exact": exact_np,
        "rho_predicted": rho_predicted_np,
        "rho_exact": rho_exact_np,
        "current_predicted": current_predicted_np,
        "current_exact": current_exact_np,
        "continuity": continuity_np,
        "energy_density": energy_density_np,
        "relative_phase_predicted": relative_phase_predicted,
        "relative_phase_exact": relative_phase_exact,
    }

    if internal_predicted is not None and internal_exact is not None:
        for channel in (1, 2):
            for quantity in (
                "amplitude_spatial_mean",
                "phase_spatial_weighted_mean",
                "phase_velocity_spatial_weighted_mean",
            ):
                name = f"channel_{channel}_{quantity}"
                time_series_payload[f"internal_{name}"] = internal_predicted_fields[name]
                time_series_payload[f"internal_exact_{name}"] = internal_exact_fields[name]
        dense_grid_payload.update(
            {
                "internal_channel_coefficient_predicted": internal_predicted,
                "internal_channel_coefficient_exact": internal_exact,
                "internal_channel_amplitude_predicted": internal_predicted_fields["amplitude"],
                "internal_channel_amplitude_exact": internal_exact_fields["amplitude"],
                "internal_channel_phase_unwrapped_predicted": internal_predicted_fields[
                    "phase_unwrapped"
                ],
                "internal_channel_phase_unwrapped_exact": internal_exact_fields[
                    "phase_unwrapped"
                ],
                "internal_channel_phase_velocity_predicted": internal_predicted_fields[
                    "phase_velocity"
                ],
                "internal_channel_phase_velocity_exact": internal_exact_fields[
                    "phase_velocity"
                ],
                "internal_channel_phase_curvature_predicted": internal_predicted_fields[
                    "phase_curvature"
                ],
                "internal_channel_phase_curvature_exact": internal_exact_fields[
                    "phase_curvature"
                ],
                "internal_channel_basis": internal_basis,
                "internal_channel_initial_coefficients": internal_initial,
            }
        )

    if branch_fields is not None:
        amplitude_plus, amplitude_minus, phase_plus, phase_minus = (
            field.detach().cpu().numpy().reshape(shape) for field in branch_fields
        )
        amplitude_plus_initial, amplitude_minus_initial = model.initial_branch_amplitudes
        phase_plus_exact = -cfg.energy * t_np[:, None] * np.ones((1, cfg.test_x_points))
        phase_minus_exact = cfg.energy * t_np[:, None] * np.ones((1, cfg.test_x_points))
        amplitude_plus_exact = np.full(shape, amplitude_plus_initial)
        amplitude_minus_exact = np.full(shape, amplitude_minus_initial)

        phase_plus_mean = np.mean(phase_plus, axis=1)
        phase_minus_mean = np.mean(phase_minus, axis=1)
        amplitude_plus_mean = np.mean(amplitude_plus, axis=1)
        amplitude_minus_mean = np.mean(amplitude_minus, axis=1)
        phase_plus_slope, phase_plus_intercept, phase_plus_r2 = _linear_fit(
            t_np, phase_plus_mean
        )
        phase_minus_slope, phase_minus_intercept, phase_minus_r2 = _linear_fit(
            t_np, phase_minus_mean
        )

        amplitude_plus_drift = amplitude_plus - amplitude_plus_initial
        amplitude_minus_drift = amplitude_minus - amplitude_minus_initial
        phase_plus_difference = phase_plus - phase_plus_exact
        phase_minus_difference = phase_minus - phase_minus_exact
        branch_relative_phase = phase_minus - phase_plus
        branch_relative_phase_exact = phase_minus_exact - phase_plus_exact

        metrics.update(
            {
                "phase_plus_rmse_rad": float(np.sqrt(np.mean(phase_plus_difference**2))),
                "phase_minus_rmse_rad": float(np.sqrt(np.mean(phase_minus_difference**2))),
                "phase_plus_circular_rmse_rad": float(
                    np.sqrt(np.mean(_circular_difference(phase_plus, phase_plus_exact) ** 2))
                ),
                "phase_minus_circular_rmse_rad": float(
                    np.sqrt(np.mean(_circular_difference(phase_minus, phase_minus_exact) ** 2))
                ),
                "branch_relative_phase_rmse_rad": float(
                    np.sqrt(np.mean((branch_relative_phase - branch_relative_phase_exact) ** 2))
                ),
                "branch_relative_phase_circular_rmse_rad": float(
                    np.sqrt(
                        np.mean(
                            _circular_difference(
                                branch_relative_phase, branch_relative_phase_exact
                            )
                            ** 2
                        )
                    )
                ),
                "phase_plus_slope": phase_plus_slope,
                "phase_plus_slope_target": -cfg.energy,
                "phase_plus_slope_abs_error": abs(phase_plus_slope + cfg.energy),
                "phase_plus_fit_intercept": phase_plus_intercept,
                "phase_plus_fit_r2": phase_plus_r2,
                "phase_minus_slope": phase_minus_slope,
                "phase_minus_slope_target": cfg.energy,
                "phase_minus_slope_abs_error": abs(phase_minus_slope - cfg.energy),
                "phase_minus_fit_intercept": phase_minus_intercept,
                "phase_minus_fit_r2": phase_minus_r2,
                "amplitude_plus_drift_abs_rms": float(
                    np.sqrt(np.mean(amplitude_plus_drift**2))
                ),
                "amplitude_plus_drift_abs_max": float(
                    np.max(np.abs(amplitude_plus_drift))
                ),
                "amplitude_plus_drift_rel_rms": float(
                    np.sqrt(np.mean(amplitude_plus_drift**2))
                    / abs(amplitude_plus_initial)
                ),
                "amplitude_plus_drift_rel_max": float(
                    np.max(np.abs(amplitude_plus_drift)) / abs(amplitude_plus_initial)
                ),
                "amplitude_minus_drift_abs_rms": float(
                    np.sqrt(np.mean(amplitude_minus_drift**2))
                ),
                "amplitude_minus_drift_abs_max": float(
                    np.max(np.abs(amplitude_minus_drift))
                ),
                "amplitude_minus_drift_rel_rms": (
                    None
                    if amplitude_minus_initial == 0.0
                    else float(
                        np.sqrt(np.mean(amplitude_minus_drift**2))
                        / abs(amplitude_minus_initial)
                    )
                ),
                "amplitude_minus_drift_rel_max": (
                    None
                    if amplitude_minus_initial == 0.0
                    else float(
                        np.max(np.abs(amplitude_minus_drift))
                        / abs(amplitude_minus_initial)
                    )
                ),
            }
        )

        time_series_payload.update(
            {
                "phase_plus_spatial_mean": phase_plus_mean,
                "phase_plus_exact": -cfg.energy * t_np,
                "phase_minus_spatial_mean": phase_minus_mean,
                "phase_minus_exact": cfg.energy * t_np,
                "branch_relative_phase_spatial_mean": phase_minus_mean - phase_plus_mean,
                "branch_relative_phase_exact": 2.0 * cfg.energy * t_np,
                "amplitude_plus_spatial_mean": amplitude_plus_mean,
                "amplitude_plus_spatial_std": np.std(amplitude_plus, axis=1, ddof=0),
                "amplitude_plus_exact": amplitude_plus_initial,
                "amplitude_minus_spatial_mean": amplitude_minus_mean,
                "amplitude_minus_spatial_std": np.std(amplitude_minus, axis=1, ddof=0),
                "amplitude_minus_exact": amplitude_minus_initial,
            }
        )
        dense_grid_payload.update(
            {
                "amplitude_plus": amplitude_plus,
                "amplitude_minus": amplitude_minus,
                "amplitude_plus_exact": amplitude_plus_exact,
                "amplitude_minus_exact": amplitude_minus_exact,
                "phase_plus": phase_plus,
                "phase_minus": phase_minus,
                "phase_plus_exact": phase_plus_exact,
                "phase_minus_exact": phase_minus_exact,
                "branch_relative_phase": branch_relative_phase,
                "branch_relative_phase_exact": branch_relative_phase_exact,
            }
        )

    time_series = pd.DataFrame(time_series_payload)
    time_series.to_csv(run_dir / "time_series.csv", index=False)
    np.savez_compressed(run_dir / "dense_grid.npz", **dense_grid_payload)
    write_json(run_dir / "metrics.json", metrics)
    return metrics
