from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import write_json
from .multimode_config import MODE_KEYS, MultimodeConfig
from .multimode_physics import (
    compute_multimode_basis,
    multimode_exact_solution,
    project_modal_coefficients,
    reference_energies,
)
from .physics import (
    continuity_from_derivatives,
    density_current,
    dirac_residual,
    spinor_derivatives,
)


MetricValue = float | int | str


def _relative_l2(predicted: np.ndarray, exact: np.ndarray) -> float:
    denominator = float(np.linalg.norm(exact.ravel()))
    if denominator <= 0.0:
        raise ValueError("Cannot form a relative L2 error against a zero reference")
    return float(np.linalg.norm((predicted - exact).ravel()) / denominator)


def _circular_difference(predicted: np.ndarray, exact: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (predicted - exact)))


def _weighted_linear_fit(
    times: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    """Weighted slope, intercept and weighted R-squared.

    The phase fits use modal occupation as a weight.  This makes a phase close
    to a coefficient zero contribute little without silently deleting a whole
    seed.  Identifiability is reported separately by ``phase_valid_fraction``.
    """

    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (times.ndim == values.ndim == weights.ndim == 1):
        raise ValueError("Weighted linear fit expects one-dimensional arrays")
    if not (len(times) == len(values) == len(weights)):
        raise ValueError("Weighted linear fit arrays must have equal length")
    weights = np.maximum(weights, 0.0)
    weight_sum = float(np.sum(weights))
    if len(times) < 2 or weight_sum <= 1.0e-30:
        return 0.0, float(values[0]) if len(values) else 0.0, 0.0
    design = np.column_stack((times, np.ones_like(times)))
    root_weight = np.sqrt(weights)
    slope, intercept = np.linalg.lstsq(
        design * root_weight[:, None], values * root_weight, rcond=None
    )[0]
    fitted = slope * times + intercept
    mean = float(np.sum(weights * values) / weight_sum)
    residual_sum = float(np.sum(weights * (values - fitted) ** 2))
    total_sum = float(np.sum(weights * (values - mean) ** 2))
    if total_sum <= 1.0e-30:
        r_squared = 1.0 if residual_sum <= 1.0e-30 else 0.0
    else:
        r_squared = 1.0 - residual_sum / total_sum
    return float(slope), float(intercept), float(r_squared)


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return torch.device("cpu") if buffer is None else buffer.device


def _batched_pde_diagnostics(
    model: nn.Module,
    x_flat: torch.Tensor,
    t_flat: torch.Tensor,
    *,
    mass: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    residual_parts: list[np.ndarray] = []
    continuity_parts: list[np.ndarray] = []
    for start in range(0, x_flat.shape[0], batch_size):
        stop = min(start + batch_size, x_flat.shape[0])
        x = x_flat[start:stop].detach().clone().requires_grad_(True)
        t = t_flat[start:stop].detach().clone().requires_grad_(True)
        psi = model(x, t)
        psi_x, psi_t = spinor_derivatives(psi, x, t, create_graph=False)
        residual = dirac_residual(psi, psi_x, psi_t, mass=mass)
        continuity = continuity_from_derivatives(psi, psi_x, psi_t)
        residual_parts.append(residual.detach().cpu().numpy())
        continuity_parts.append(continuity.detach().cpu().numpy())
    return np.concatenate(residual_parts), np.concatenate(continuity_parts)


def _global_gauge_align(
    predicted: np.ndarray, exact: np.ndarray
) -> tuple[np.ndarray, float]:
    """Remove exactly one constant phase from the full spacetime prediction."""

    overlap = np.vdot(exact.ravel(), predicted.ravel())
    phase = 0.0 if abs(overlap) <= 1.0e-30 else float(np.angle(overlap))
    return predicted * np.exp(-1j * phase), phase


def _phase_diagnostics(
    coefficients: np.ndarray,
    exact_coefficients: np.ndarray,
    times: np.ndarray,
    targets: np.ndarray,
    initial_coefficient: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    metrics: dict[str, float] = {}
    fields: dict[str, np.ndarray] = {}
    velocity_residuals: list[np.ndarray] = []
    velocity_weights: list[np.ndarray] = []
    curvatures: list[np.ndarray] = []
    curvature_weights: list[np.ndarray] = []

    for index, key in enumerate(MODE_KEYS):
        coefficient = coefficients[:, index]
        exact = exact_coefficients[:, index]
        amplitude = np.abs(coefficient)
        occupation_weight = amplitude**2
        valid = amplitude > (1.0e-6 * abs(initial_coefficient))
        phase = np.unwrap(np.angle(coefficient))
        exact_phase = np.unwrap(np.angle(exact))
        slope, intercept, r_squared = _weighted_linear_fit(
            times, phase, occupation_weight
        )
        target = float(targets[index])
        phase_error = _circular_difference(np.angle(coefficient), np.angle(exact))
        edge_order = 2 if len(times) >= 3 else 1
        velocity = np.gradient(phase, times, edge_order=edge_order)
        curvature = np.gradient(velocity, times, edge_order=edge_order)
        velocity_residuals.append(velocity - slope)
        velocity_weights.append(occupation_weight)
        curvatures.append(curvature)
        curvature_weights.append(occupation_weight)

        metrics.update(
            {
                f"{key}_phase_slope": slope,
                f"{key}_phase_slope_target": target,
                f"{key}_phase_slope_relative_error": abs(slope - target)
                / abs(target),
                f"{key}_phase_slope_r2": r_squared,
                f"{key}_phase_fit_intercept": intercept,
                f"{key}_phase_valid_fraction": float(np.mean(valid)),
                f"{key}_phase_circular_rmse_rad": float(
                    np.sqrt(np.mean(phase_error**2))
                ),
                f"{key}_phase_velocity_fluctuation_rms": float(
                    np.sqrt(
                        np.sum(occupation_weight * (velocity - slope) ** 2)
                        / max(float(np.sum(occupation_weight)), 1.0e-30)
                    )
                ),
                f"{key}_phase_curvature_weighted_rms": float(
                    np.sqrt(
                        np.sum(occupation_weight * curvature**2)
                        / max(float(np.sum(occupation_weight)), 1.0e-30)
                    )
                ),
            }
        )
        fields[f"{key}_phase_unwrapped"] = phase
        fields[f"{key}_phase_exact_unwrapped"] = exact_phase
        fields[f"{key}_phase_valid"] = valid
        fields[f"{key}_phase_velocity"] = velocity
        fields[f"{key}_phase_curvature"] = curvature
        fields[f"{key}_phase_circular_error"] = phase_error

    all_velocity_residuals = np.stack(velocity_residuals, axis=1)
    all_velocity_weights = np.stack(velocity_weights, axis=1)
    all_curvatures = np.stack(curvatures, axis=1)
    all_curvature_weights = np.stack(curvature_weights, axis=1)
    metrics["phase_velocity_fluctuation_weighted_rms"] = float(
        np.sqrt(
            np.sum(all_velocity_weights * all_velocity_residuals**2)
            / max(float(np.sum(all_velocity_weights)), 1.0e-30)
        )
    )
    metrics["phase_curvature_weighted_rms"] = float(
        np.sqrt(
            np.sum(all_curvature_weights * all_curvatures**2)
            / max(float(np.sum(all_curvature_weights)), 1.0e-30)
        )
    )
    return metrics, fields


def _relative_phase_diagnostics(
    coefficients: np.ndarray,
    exact_coefficients: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    # Each tuple is (left, right), with phase arg(conj(left) * right).
    pairs = {
        "relative_phase_k2_branches": (0, 1),
        "relative_phase_k5_branches": (2, 3),
        "relative_phase_positive_cross_k": (0, 2),
        "relative_phase_negative_cross_k": (1, 3),
    }
    metrics: dict[str, float] = {}
    fields: dict[str, np.ndarray] = {}
    all_errors: list[np.ndarray] = []
    for name, (left, right) in pairs.items():
        predicted = np.angle(np.conj(coefficients[:, left]) * coefficients[:, right])
        exact = np.angle(
            np.conj(exact_coefficients[:, left]) * exact_coefficients[:, right]
        )
        difference = _circular_difference(predicted, exact)
        metrics[f"{name}_circular_rmse_rad"] = float(
            np.sqrt(np.mean(difference**2))
        )
        fields[f"{name}_predicted"] = predicted
        fields[f"{name}_exact"] = exact
        fields[f"{name}_circular_error"] = difference
        fields[f"{name}_predicted_unwrapped"] = np.unwrap(predicted)
        fields[f"{name}_exact_unwrapped"] = np.unwrap(exact)
        all_errors.append(difference)
    metrics["relative_phase_pairs_circular_rmse_rad"] = float(
        np.sqrt(np.mean(np.stack(all_errors, axis=1) ** 2))
    )
    return metrics, fields


def _internal_ltpb_diagnostics(
    model: nn.Module,
    x_flat: torch.Tensor,
    t_flat: torch.Tensor,
    shape: tuple[int, int],
    times: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray]]:
    if getattr(model, "model_name", "") != "multi_ltpb" or not hasattr(
        model, "mode_fields"
    ):
        return {}, {}, {}
    with torch.no_grad():
        amplitude_flat, phase_flat = model.mode_fields(x_flat, t_flat)
    amplitudes = amplitude_flat.detach().cpu().numpy().reshape(*shape, 4)
    phases = phase_flat.detach().cpu().numpy().reshape(*shape, 4)
    weights = amplitudes**2
    denominator = np.maximum(np.sum(weights, axis=1), 1.0e-30)
    phase_means = np.sum(weights * phases, axis=1) / denominator
    amplitude_means = np.mean(amplitudes, axis=1)
    amplitude_stds = np.std(amplitudes, axis=1, ddof=0)
    metrics: dict[str, float] = {}
    time_fields: dict[str, np.ndarray] = {}
    initial = float(model.cfg.initial_mode_coefficient)
    edge_order = 2 if len(times) >= 3 else 1
    all_curvature_numerators = 0.0
    all_curvature_denominators = 0.0
    for index, key in enumerate(MODE_KEYS):
        slope, intercept, r_squared = _weighted_linear_fit(
            times, phase_means[:, index], np.mean(weights[:, :, index], axis=1)
        )
        curvature = np.gradient(
            np.gradient(phase_means[:, index], times, edge_order=edge_order),
            times,
            edge_order=edge_order,
        )
        time_weight = np.mean(weights[:, :, index], axis=1)
        all_curvature_numerators += float(np.sum(time_weight * curvature**2))
        all_curvature_denominators += float(np.sum(time_weight))
        drift = amplitudes[:, :, index] - initial
        metrics.update(
            {
                f"internal_{key}_amplitude_drift_rel_rms": float(
                    np.sqrt(np.mean(drift**2)) / abs(initial)
                ),
                f"internal_{key}_phase_slope": slope,
                f"internal_{key}_phase_slope_r2": r_squared,
                f"internal_{key}_phase_fit_intercept": intercept,
            }
        )
        time_fields[f"internal_{key}_amplitude_spatial_mean"] = amplitude_means[
            :, index
        ]
        time_fields[f"internal_{key}_amplitude_spatial_std"] = amplitude_stds[
            :, index
        ]
        time_fields[f"internal_{key}_phase_spatial_weighted_mean"] = phase_means[
            :, index
        ]
    metrics["internal_phase_curvature_weighted_rms"] = float(
        math.sqrt(
            all_curvature_numerators / max(all_curvature_denominators, 1.0e-30)
        )
    )
    return metrics, time_fields, {
        "internal_mode_amplitudes": amplitudes,
        "internal_mode_phases": phases,
    }


def compute_multimode_evaluation(
    model: nn.Module,
    cfg: MultimodeConfig,
    *,
    model_name: str,
    seed: int,
    parameter_count: int,
    training_seconds: float,
    final_pde_loss: float,
    steps: int | None = None,
    experiment: str = "multimode_ltpb",
    derivative_batch_size: int = 4096,
) -> tuple[dict[str, MetricValue], pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate one learned model or the deterministic diagnostic oracle.

    Exact energies are first computed in this post-training routine.  A single
    constant phase, optimized over the entire spacetime grid, is removed before
    spinor and modal-coefficient comparisons.  It is deliberately *not* fitted
    independently at each time or for each mode.
    """

    model.eval()
    device = _model_device(model)
    x_values = cfg.x_min + cfg.length * torch.arange(
        cfg.test_x_points, device=device, dtype=torch.float64
    ) / cfg.test_x_points
    t_values = torch.linspace(
        0.0, cfg.t_max, cfg.test_t_points, device=device, dtype=torch.float64
    )
    t_mesh, x_mesh = torch.meshgrid(t_values, x_values, indexing="ij")
    x_flat = x_mesh.reshape(-1, 1)
    t_flat = t_mesh.reshape(-1, 1)
    shape = (cfg.test_t_points, cfg.test_x_points)

    # This is the only point where analytic energies enter the learned-run
    # pipeline: reference solution generation and post-training diagnostics.
    energies = reference_energies(cfg.mass, cfg.wave_numbers)
    targets = np.asarray(
        (-energies[0], energies[0], -energies[1], energies[1]),
        dtype=np.float64,
    )
    with torch.no_grad():
        predicted_torch = model(x_flat, t_flat)
        exact_torch = multimode_exact_solution(
            x_flat,
            t_flat,
            mass=cfg.mass,
            wave_numbers=cfg.wave_numbers,
            length=cfg.length,
        )

    predicted_raw = predicted_torch.detach().cpu().numpy().reshape(*shape, 2)
    exact = exact_torch.detach().cpu().numpy().reshape(*shape, 2)
    predicted, global_phase = _global_gauge_align(predicted_raw, exact)
    aligned_torch = torch.as_tensor(
        predicted.reshape(-1, 2), device=device, dtype=torch.complex128
    )

    rho_predicted_t, current_predicted_t = density_current(aligned_torch)
    rho_exact_t, current_exact_t = density_current(exact_torch)
    rho_predicted = rho_predicted_t.detach().cpu().numpy().reshape(shape)
    rho_exact = rho_exact_t.detach().cpu().numpy().reshape(shape)
    current_predicted = current_predicted_t.detach().cpu().numpy().reshape(shape)
    current_exact = current_exact_t.detach().cpu().numpy().reshape(shape)

    residual_flat, continuity_flat = _batched_pde_diagnostics(
        model,
        x_flat,
        t_flat,
        mass=cfg.mass,
        batch_size=derivative_batch_size,
    )
    residual = residual_flat.reshape(*shape, 2)
    continuity = continuity_flat.reshape(shape)

    basis = compute_multimode_basis(
        cfg.mass, cfg.wave_numbers, device=device, dtype=torch.float64
    )
    coefficients_t = project_modal_coefficients(
        aligned_torch.reshape(*shape, 2), x_values, basis
    )
    exact_coefficients_t = project_modal_coefficients(
        exact_torch.reshape(*shape, 2), x_values, basis
    )
    coefficients = coefficients_t.detach().cpu().numpy()
    exact_coefficients = exact_coefficients_t.detach().cpu().numpy()
    occupations = cfg.length * np.abs(coefficients) ** 2
    exact_occupations = cfg.length * np.abs(exact_coefficients) ** 2

    x_np = x_values.detach().cpu().numpy()
    t_np = t_values.detach().cpu().numpy()
    norms = cfg.length * np.mean(rho_predicted, axis=1)
    exact_norms = cfg.length * np.mean(rho_exact, axis=1)
    overlap_t = cfg.length * np.mean(
        np.sum(np.conj(exact) * predicted, axis=2), axis=1
    )
    predicted_norm_t = cfg.length * np.mean(np.sum(np.abs(predicted) ** 2, axis=2), axis=1)
    exact_norm_t = cfg.length * np.mean(np.sum(np.abs(exact) ** 2, axis=2), axis=1)
    fidelity = np.clip(
        np.abs(overlap_t) ** 2
        / np.maximum(predicted_norm_t * exact_norm_t, 1.0e-30),
        0.0,
        1.0,
    )

    component_phase_predicted = np.angle(
        np.conj(predicted[:, :, 0]) * predicted[:, :, 1]
    )
    component_phase_exact = np.angle(np.conj(exact[:, :, 0]) * exact[:, :, 1])
    component_product_exact = np.abs(np.conj(exact[:, :, 0]) * exact[:, :, 1])
    phase_mask = component_product_exact > 1.0e-8 * float(
        np.max(component_product_exact)
    )
    component_phase_error = _circular_difference(
        component_phase_predicted, component_phase_exact
    )

    phase_metrics, phase_fields = _phase_diagnostics(
        coefficients,
        exact_coefficients,
        t_np,
        targets,
        cfg.initial_mode_coefficient,
    )
    pair_metrics, pair_fields = _relative_phase_diagnostics(
        coefficients, exact_coefficients
    )
    internal_metrics, internal_time, internal_dense = _internal_ltpb_diagnostics(
        model, x_flat, t_flat, shape, t_np
    )

    coefficient_denominator = np.maximum(np.abs(exact_coefficients), 1.0e-30)
    occupation_scale = float(np.sqrt(np.mean(exact_occupations**2)))
    metrics: dict[str, MetricValue] = {
        "experiment": experiment,
        "model": model_name,
        "seed": int(seed),
        "steps": cfg.adam_steps if steps is None else int(steps),
        "parameter_count": int(parameter_count),
        "training_seconds": float(training_seconds),
        "final_loss_pde": float(final_pde_loss),
        "global_gauge_phase_rad": global_phase,
        "spinor_rel_l2_gauge_aligned": _relative_l2(predicted, exact),
        "spinor_rel_l2_raw": _relative_l2(predicted_raw, exact),
        "final_spinor_rel_l2_gauge_aligned": _relative_l2(predicted[-1], exact[-1]),
        "density_rel_l2": _relative_l2(rho_predicted, rho_exact),
        "rho_rel_l2": _relative_l2(rho_predicted, rho_exact),
        "current_rel_l2": _relative_l2(current_predicted, current_exact),
        "fidelity_mean": float(np.mean(fidelity)),
        "fidelity_min": float(np.min(fidelity)),
        "fidelity_final": float(fidelity[-1]),
        "spinor_component_relative_phase_rmse_rad": float(
            np.sqrt(np.mean(component_phase_error[phase_mask] ** 2))
        ),
        "spinor_component_relative_phase_mask_fraction": float(np.mean(phase_mask)),
        "mode_coefficient_rel_l2_gauge_aligned": _relative_l2(
            coefficients, exact_coefficients
        ),
        "mode_occupation_rmse_abs": float(
            np.sqrt(np.mean((occupations - exact_occupations) ** 2))
        ),
        "mode_occupation_rmse_rel": float(
            np.sqrt(np.mean((occupations - exact_occupations) ** 2))
            / max(occupation_scale, 1.0e-30)
        ),
        "modal_occupation_sum_defect_mean_abs": float(
            np.mean(np.abs(np.sum(occupations, axis=1) - norms))
        ),
        "dense_pde_residual_rms": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
        "dense_pde_residual_max_abs": float(np.max(np.abs(residual))),
        "dense_continuity_rms": float(np.sqrt(np.mean(continuity**2))),
        "max_norm_drift": float(np.max(np.abs(norms - 1.0))),
        "reference_energy_k2": float(energies[0]),
        "reference_energy_k5": float(energies[1]),
    }
    for index, key in enumerate(MODE_KEYS):
        coefficient_error = coefficients[:, index] - exact_coefficients[:, index]
        occupation_error = occupations[:, index] - exact_occupations[:, index]
        metrics.update(
            {
                f"{key}_coefficient_rel_l2": float(
                    np.linalg.norm(coefficient_error)
                    / max(float(np.linalg.norm(exact_coefficients[:, index])), 1.0e-30)
                ),
                f"{key}_coefficient_relative_rms": float(
                    np.sqrt(np.mean(np.abs(coefficient_error / coefficient_denominator[:, index]) ** 2))
                ),
                f"{key}_occupation_rmse_abs": float(
                    np.sqrt(np.mean(occupation_error**2))
                ),
                f"{key}_occupation_rmse_rel": float(
                    np.sqrt(np.mean(occupation_error**2))
                    / max(
                        float(np.sqrt(np.mean(exact_occupations[:, index] ** 2))),
                        1.0e-30,
                    )
                ),
                f"{key}_amplitude_drift_rel_rms": float(
                    np.sqrt(
                        np.mean(
                            (
                                np.abs(coefficients[:, index])
                                - cfg.initial_mode_coefficient
                            )
                            ** 2
                        )
                    )
                    / abs(cfg.initial_mode_coefficient)
                ),
            }
        )
    metrics.update(phase_metrics)
    metrics.update(pair_metrics)
    metrics.update(internal_metrics)

    index_x0 = int(np.argmin(np.abs(x_np)))
    time_payload: dict[str, np.ndarray] = {
        "t": t_np,
        "norm_predicted": norms,
        "norm_exact": exact_norms,
        "norm_drift": norms - 1.0,
        "rho_x0_predicted": rho_predicted[:, index_x0],
        "rho_x0_exact": rho_exact[:, index_x0],
        "current_x0_predicted": current_predicted[:, index_x0],
        "current_x0_exact": current_exact[:, index_x0],
        "fidelity": fidelity,
        "dense_pde_residual_rms_t": np.sqrt(np.mean(np.abs(residual) ** 2, axis=(1, 2))),
        "continuity_rms_t": np.sqrt(np.mean(continuity**2, axis=1)),
        "spinor_component_relative_phase_x0_predicted": component_phase_predicted[:, index_x0],
        "spinor_component_relative_phase_x0_exact": component_phase_exact[:, index_x0],
        "spinor_component_relative_phase_x0_circular_error": component_phase_error[:, index_x0],
    }
    for index, key in enumerate(MODE_KEYS):
        time_payload.update(
            {
                f"{key}_coefficient_real": coefficients[:, index].real,
                f"{key}_coefficient_imag": coefficients[:, index].imag,
                f"{key}_coefficient_exact_real": exact_coefficients[:, index].real,
                f"{key}_coefficient_exact_imag": exact_coefficients[:, index].imag,
                f"{key}_occupation": occupations[:, index],
                f"{key}_occupation_exact": exact_occupations[:, index],
                f"{key}_phase": phase_fields[f"{key}_phase_unwrapped"],
                f"{key}_phase_exact": phase_fields[
                    f"{key}_phase_exact_unwrapped"
                ],
                f"{key}_phase_valid": phase_fields[f"{key}_phase_valid"],
                f"{key}_phase_circular_error": phase_fields[
                    f"{key}_phase_circular_error"
                ],
                f"{key}_phase_velocity": phase_fields[f"{key}_phase_velocity"],
            }
        )
    time_payload.update(pair_fields)
    time_payload.update(internal_time)
    time_series = pd.DataFrame(time_payload)

    dense_grid: dict[str, np.ndarray] = {
        "x": x_np,
        "t": t_np,
        "psi_predicted_raw": predicted_raw,
        "psi_predicted_gauge_aligned": predicted,
        "psi_exact": exact,
        "rho_predicted": rho_predicted,
        "rho_exact": rho_exact,
        "current_predicted": current_predicted,
        "current_exact": current_exact,
        "dirac_residual": residual,
        "continuity_residual": continuity,
        "spinor_component_relative_phase_predicted": component_phase_predicted,
        "spinor_component_relative_phase_exact": component_phase_exact,
        "spinor_component_relative_phase_circular_error": component_phase_error,
        "spinor_component_relative_phase_mask": phase_mask,
        "mode_coefficients_predicted_gauge_aligned": coefficients,
        "mode_coefficients_exact": exact_coefficients,
        "mode_occupations_predicted": occupations,
        "mode_occupations_exact": exact_occupations,
        "mode_phase_slopes_target": targets,
        "reference_energies": energies,
        "global_gauge_phase_rad": np.asarray(global_phase, dtype=np.float64),
    }
    dense_grid.update(internal_dense)
    return metrics, time_series, dense_grid


def evaluate_multimode_model(
    model: nn.Module,
    cfg: MultimodeConfig,
    spec: Any,
    run_dir: Path,
    *,
    parameter_count: int,
    training_seconds: float,
    final_pde_loss: float,
    derivative_batch_size: int = 4096,
) -> dict[str, MetricValue]:
    """Compute, persist, and return the complete evaluation for one run."""

    metrics, time_series, dense_grid = compute_multimode_evaluation(
        model,
        cfg,
        model_name=str(spec.model),
        seed=int(spec.seed),
        parameter_count=parameter_count,
        training_seconds=training_seconds,
        final_pde_loss=final_pde_loss,
        steps=int(spec.steps),
        experiment=str(spec.experiment),
        derivative_batch_size=derivative_batch_size,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    time_series.to_csv(run_dir / "time_series.csv", index=False)
    np.savez_compressed(run_dir / "dense_grid.npz", **dense_grid)
    write_json(run_dir / "metrics.json", metrics)
    return metrics


__all__ = ["compute_multimode_evaluation", "evaluate_multimode_model"]
