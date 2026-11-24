from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .physics import density_current, dirac_residual, spinor_derivatives
from .wavepacket_config import WavepacketConfig
from .wavepacket_models import SharedSpectralModel
from .wavepacket_physics import (
    compute_spectral_basis,
    exact_spectral_state,
    gaussian_spectral_envelope,
    project_energy_branches,
    reconstruct_physical_grid,
    reference_energies,
    spatial_fourier_coefficients,
    spectral_dirac_residual,
    training_wave_numbers,
)


MetricValue = float | int | str | None


def _relative_l2(predicted: np.ndarray, exact: np.ndarray) -> float:
    denominator = float(np.linalg.norm(exact.ravel()))
    if denominator <= 0.0:
        raise ValueError("Cannot form relative L2 against a zero reference")
    return float(np.linalg.norm((predicted - exact).ravel()) / denominator)


def _global_gauge_align(
    predicted: np.ndarray, exact: np.ndarray
) -> tuple[np.ndarray, float]:
    overlap = np.vdot(exact.ravel(), predicted.ravel())
    phase = 0.0 if abs(overlap) <= 1.0e-30 else float(np.angle(overlap))
    return predicted * np.exp(-1j * phase), phase


def _weighted_linear_fit(
    times: np.ndarray, values: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float]:
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if not (times.ndim == values.ndim == weights.ndim == 1):
        raise ValueError("Weighted fit inputs must be one-dimensional")
    if not (len(times) == len(values) == len(weights)):
        raise ValueError("Weighted fit inputs must have equal lengths")
    weight_sum = float(np.sum(weights))
    if len(times) < 2 or weight_sum <= 1.0e-30:
        return math.nan, math.nan, math.nan
    design = np.column_stack((times, np.ones_like(times)))
    root_weight = np.sqrt(weights)
    slope, intercept = np.linalg.lstsq(
        design * root_weight[:, None], values * root_weight, rcond=None
    )[0]
    fitted = slope * times + intercept
    mean = float(np.sum(weights * values) / weight_sum)
    residual_sum = float(np.sum(weights * (values - fitted) ** 2))
    total_sum = float(np.sum(weights * (values - mean) ** 2))
    r_squared = 1.0 if total_sum <= 1.0e-30 else 1.0 - residual_sum / total_sum
    return float(slope), float(intercept), float(r_squared)


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return torch.device("cpu") if buffer is None else buffer.device


def _batched_model(
    model: nn.Module,
    first: torch.Tensor,
    second: torch.Tensor,
    batch_size: int = 4096,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(first), batch_size):
            stop = min(start + batch_size, len(first))
            pieces.append(model(first[start:stop], second[start:stop]).detach())
    return torch.cat(pieces, dim=0)


def _complex_time_derivative(
    state: torch.Tensor, time: torch.Tensor, *, create_graph: bool
) -> torch.Tensor:
    components: list[torch.Tensor] = []
    for index in range(2):
        real = torch.autograd.grad(
            state[:, index].real,
            time,
            grad_outputs=torch.ones_like(state[:, index].real),
            create_graph=create_graph,
            retain_graph=True,
        )[0].reshape(-1)
        imag = torch.autograd.grad(
            state[:, index].imag,
            time,
            grad_outputs=torch.ones_like(state[:, index].imag),
            create_graph=create_graph,
            retain_graph=True,
        )[0].reshape(-1)
        components.append(torch.complex(real, imag))
    return torch.stack(components, dim=1)


def _physical_residual_grid(
    model: nn.Module,
    x_flat: torch.Tensor,
    t_flat: torch.Tensor,
    *,
    mass: float,
    batch_size: int,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for start in range(0, len(x_flat), batch_size):
        stop = min(start + batch_size, len(x_flat))
        x = x_flat[start:stop].detach().clone().requires_grad_(True)
        t = t_flat[start:stop].detach().clone().requires_grad_(True)
        psi = model(x, t)
        psi_x, psi_t = spinor_derivatives(psi, x, t, create_graph=False)
        pieces.append(dirac_residual(psi, psi_x, psi_t, mass=mass).detach())
    return torch.cat(pieces, dim=0)


def _spectral_residual_grid(
    model: nn.Module,
    wave_numbers: torch.Tensor,
    times: torch.Tensor,
    *,
    mass: float,
    batch_size: int,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for start in range(0, len(wave_numbers), batch_size):
        stop = min(start + batch_size, len(wave_numbers))
        k = wave_numbers[start:stop].detach().clone()
        t = times[start:stop].detach().clone().requires_grad_(True)
        state = model(k, t)
        state_t = _complex_time_derivative(state, t, create_graph=False)
        pieces.append(spectral_dirac_residual(state, state_t, k.reshape(-1), mass).detach())
    return torch.cat(pieces, dim=0)


def _phase_diagnostics(
    coefficients: np.ndarray,
    exact_coefficients: np.ndarray,
    times: np.ndarray,
    targets: np.ndarray,
    initial_amplitudes: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Fit every physical Fourier-energy coefficient, then occupation-weight."""

    if coefficients.shape != exact_coefficients.shape:
        raise ValueError("Predicted/exact coefficient shapes differ")
    if coefficients.ndim != 3 or coefficients.shape[-1] != 2:
        raise ValueError("Coefficient arrays must have shape (time,k,branch)")
    n_k = coefficients.shape[1]
    slopes = np.empty((n_k, 2), dtype=np.float64)
    intercepts = np.empty_like(slopes)
    r_squared = np.empty_like(slopes)
    valid_fraction = np.empty_like(slopes)
    relative_error = np.empty_like(slopes)
    circular_rmse = np.empty_like(slopes)
    phase = np.unwrap(np.angle(coefficients), axis=0)
    exact_phase = np.unwrap(np.angle(exact_coefficients), axis=0)
    for index in range(n_k):
        for branch in range(2):
            amplitude = np.abs(coefficients[:, index, branch])
            valid = amplitude > 1.0e-6 * max(initial_amplitudes[index], 1.0e-30)
            valid_fraction[index, branch] = float(np.mean(valid))
            slope, intercept, r2 = _weighted_linear_fit(
                times, phase[:, index, branch], amplitude**2
            )
            slopes[index, branch] = slope
            intercepts[index, branch] = intercept
            r_squared[index, branch] = r2
            target = targets[index, branch]
            relative_error[index, branch] = abs(slope - target) / abs(target)
            phase_error = np.angle(
                np.exp(
                    1j
                    * (
                        np.angle(coefficients[:, index, branch])
                        - np.angle(exact_coefficients[:, index, branch])
                    )
                )
            )
            circular_rmse[index, branch] = float(np.sqrt(np.mean(phase_error**2)))
    initial_weights = np.repeat(initial_amplitudes[:, None] ** 2, 2, axis=1)
    identifiable = bool(np.all(valid_fraction >= 0.90))
    weighted_error = float(
        np.sum(initial_weights * relative_error) / np.sum(initial_weights)
    )
    if not identifiable or not np.all(np.isfinite(relative_error)):
        weighted_error = math.inf
    edge_order = 2 if n_k >= 3 else 1
    slope_first_derivative = np.gradient(slopes, axis=0, edge_order=edge_order)
    slope_second_derivative = np.gradient(
        slope_first_derivative, axis=0, edge_order=edge_order
    )
    error_adjacent_jump_rms = float(
        np.sqrt(np.mean(np.diff(relative_error, axis=0) ** 2))
    ) if n_k > 1 else 0.0
    return (
        {
            "weighted_relative_error": weighted_error,
            "min_valid_fraction": float(np.min(valid_fraction)),
            "identifiable_fraction": float(np.mean(valid_fraction >= 0.90)),
            "relative_error_adjacent_jump_rms": error_adjacent_jump_rms,
            "slope_curvature_rms": float(np.sqrt(np.mean(slope_second_derivative**2))),
        },
        {
            "phase_slopes": slopes,
            "phase_slope_targets": targets,
            "phase_slope_relative_errors": relative_error,
            "phase_fit_intercepts": intercepts,
            "phase_fit_r_squared": r_squared,
            "phase_valid_fractions": valid_fraction,
            "phase_circular_rmse": circular_rmse,
            "phases_unwrapped": phase,
            "phases_exact_unwrapped": exact_phase,
            "phase_slope_curvature": slope_second_derivative,
        },
    )


def _circular_center_width(
    density: np.ndarray, x: np.ndarray, length: float, x_min: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(density, dtype=np.float64)
    norms = length * np.mean(weights, axis=1)
    angle_x = 2.0 * math.pi * (x - x_min) / length
    numerator = np.mean(weights * np.exp(1j * angle_x[None, :]), axis=1)
    z = numerator / np.maximum(np.mean(weights, axis=1), 1.0e-30)
    angle = np.unwrap(np.mod(np.angle(z), 2.0 * math.pi))
    centers = x_min + length * angle / (2.0 * math.pi)
    resultant = np.clip(np.abs(z), 1.0e-15, 1.0)
    widths = length * np.sqrt(np.maximum(-2.0 * np.log(resultant), 0.0)) / (
        2.0 * math.pi
    )
    return centers, widths, norms


def _branch_diagnostics(
    coefficients: np.ndarray,
    exact_coefficients: np.ndarray,
    wave_numbers: torch.Tensor,
    x: torch.Tensor,
    times: np.ndarray,
    cfg: WavepacketConfig,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    basis = compute_spectral_basis(wave_numbers, cfg.mass)
    predicted_t = torch.as_tensor(coefficients, dtype=torch.complex128)
    exact_t = torch.as_tensor(exact_coefficients, dtype=torch.complex128)
    plus_spectral = predicted_t[:, :, 0:1] * basis.u_plus.to(torch.complex128)[None]
    minus_spectral = predicted_t[:, :, 1:2] * basis.u_minus.to(torch.complex128)[None]
    exact_plus_spectral = exact_t[:, :, 0:1] * basis.u_plus.to(torch.complex128)[None]
    exact_minus_spectral = exact_t[:, :, 1:2] * basis.u_minus.to(torch.complex128)[None]
    fields = {
        "plus": reconstruct_physical_grid(
            plus_spectral, wave_numbers, x, length=cfg.length
        ).numpy(),
        "minus": reconstruct_physical_grid(
            minus_spectral, wave_numbers, x, length=cfg.length
        ).numpy(),
        "plus_exact": reconstruct_physical_grid(
            exact_plus_spectral, wave_numbers, x, length=cfg.length
        ).numpy(),
        "minus_exact": reconstruct_physical_grid(
            exact_minus_spectral, wave_numbers, x, length=cfg.length
        ).numpy(),
    }
    diagnostics: dict[str, np.ndarray] = {}
    metrics: dict[str, float] = {}
    minimum_norm = math.inf
    for branch in ("plus", "minus"):
        predicted_density = np.sum(np.abs(fields[branch]) ** 2, axis=2)
        exact_density = np.sum(np.abs(fields[f"{branch}_exact"]) ** 2, axis=2)
        center, width, norm = _circular_center_width(
            predicted_density, x.numpy(), cfg.length, cfg.x_min
        )
        exact_center, exact_width, exact_norm = _circular_center_width(
            exact_density, x.numpy(), cfg.length, cfg.x_min
        )
        minimum_norm = min(minimum_norm, float(np.min(norm)))
        velocity = float(np.polyfit(times, center, 1)[0])
        exact_velocity = float(np.polyfit(times, exact_center, 1)[0])
        metrics.update(
            {
                f"branch_{branch}_center_rmse": float(
                    np.sqrt(np.mean((center - exact_center) ** 2))
                ),
                f"branch_{branch}_velocity": velocity,
                f"branch_{branch}_exact_velocity": exact_velocity,
                f"branch_{branch}_velocity_relative_error": abs(
                    velocity - exact_velocity
                ) / abs(exact_velocity),
                f"branch_{branch}_width_rel_l2": _relative_l2(width, exact_width),
                f"branch_{branch}_final_center": float(center[-1]),
                f"branch_{branch}_exact_final_center": float(exact_center[-1]),
                f"branch_{branch}_final_width_change": float(width[-1] - width[0]),
                f"branch_{branch}_exact_final_width_change": float(
                    exact_width[-1] - exact_width[0]
                ),
                f"branch_{branch}_norm_mean": float(np.mean(norm)),
            }
        )
        diagnostics.update(
            {
                f"branch_{branch}_density": predicted_density,
                f"branch_{branch}_density_exact": exact_density,
                f"branch_{branch}_center": center,
                f"branch_{branch}_center_exact": exact_center,
                f"branch_{branch}_width": width,
                f"branch_{branch}_width_exact": exact_width,
                f"branch_{branch}_norm": norm,
                f"branch_{branch}_norm_exact": exact_norm,
            }
        )
    metrics["branch_minimum_norm"] = float(minimum_norm)
    return metrics, diagnostics


def _heldout_diagnostics(
    model: nn.Module,
    cfg: WavepacketConfig,
    times: torch.Tensor,
    global_phase: float,
    *,
    derivative_batch_size: int,
) -> tuple[dict[str, float | None], dict[str, np.ndarray]]:
    if not isinstance(model, SharedSpectralModel):
        return (
            {
                "heldout_phase_slope_weighted_relative_error": None,
                "heldout_phase_min_valid_fraction": None,
                "heldout_spectral_pde_residual_rms": None,
                "heldout_amplitude_drift_weighted_rel_rms": None,
                "heldout_phase_relative_error_adjacent_jump_rms": None,
                "heldout_phase_slope_curvature_rms": None,
            },
            {},
        )
    device = _model_device(model)
    spacing = cfg.wave_number_spacing
    midpoint = torch.arange(
        cfg.n_min,
        cfg.n_max,
        dtype=torch.float64,
        device=device,
    ) * spacing + 0.5 * spacing
    t_mesh, k_mesh = torch.meshgrid(times.to(device), midpoint, indexing="ij")
    k_flat = k_mesh.reshape(-1, 1)
    t_flat = t_mesh.reshape(-1, 1)
    predicted_state = _batched_model(model, k_flat, t_flat)
    exact_state = exact_spectral_state(k_flat, t_flat, cfg)
    predicted_state = predicted_state * torch.exp(
        torch.tensor(-1j * global_phase, dtype=torch.complex128, device=device)
    )
    shape = (len(times), len(midpoint), 2)
    predicted_spectral = predicted_state.reshape(shape)
    exact_spectral = exact_state.reshape(shape)
    coefficients = project_energy_branches(
        predicted_spectral, midpoint, cfg.mass
    ).detach().cpu().numpy()
    exact_coefficients = project_energy_branches(
        exact_spectral, midpoint, cfg.mass
    ).detach().cpu().numpy()
    energies = np.asarray(reference_energies(midpoint.cpu().numpy(), cfg.mass))
    targets = np.stack((-energies, energies), axis=1)
    initial = (
        gaussian_spectral_envelope(midpoint, cfg).detach().cpu().numpy()
        / math.sqrt(2.0)
    )
    phase_metrics, phase_fields = _phase_diagnostics(
        coefficients,
        exact_coefficients,
        times.cpu().numpy(),
        targets,
        initial,
    )
    residual = _spectral_residual_grid(
        model,
        k_flat,
        t_flat,
        mass=cfg.mass,
        batch_size=derivative_batch_size,
    ).reshape(shape)
    denominator = len(times) * float(np.sum(np.repeat(initial[:, None] ** 2, 2, axis=1)))
    drift = float(
        np.sqrt(
            np.sum((np.abs(coefficients) - initial[None, :, None]) ** 2)
            / max(denominator, 1.0e-30)
        )
    )
    metrics: dict[str, float | None] = {
        "heldout_phase_slope_weighted_relative_error": phase_metrics[
            "weighted_relative_error"
        ],
        "heldout_phase_min_valid_fraction": phase_metrics["min_valid_fraction"],
        "heldout_spectral_pde_residual_rms": float(
            torch.sqrt(torch.mean(torch.abs(residual) ** 2)).cpu()
        ),
        "heldout_amplitude_drift_weighted_rel_rms": drift,
        "heldout_phase_relative_error_adjacent_jump_rms": phase_metrics[
            "relative_error_adjacent_jump_rms"
        ],
        "heldout_phase_slope_curvature_rms": phase_metrics[
            "slope_curvature_rms"
        ],
    }
    dense = {
        "heldout_wave_numbers": midpoint.cpu().numpy(),
        "heldout_spectral_state_predicted_gauge_aligned": predicted_spectral.cpu().numpy(),
        "heldout_spectral_state_exact": exact_spectral.cpu().numpy(),
        "heldout_branch_coefficients_predicted_gauge_aligned": coefficients,
        "heldout_branch_coefficients_exact": exact_coefficients,
        "heldout_dirac_residual": residual.cpu().numpy(),
        **{f"heldout_{key}": value for key, value in phase_fields.items()},
    }
    return metrics, dense


def compute_wavepacket_evaluation(
    model: nn.Module,
    cfg: WavepacketConfig,
    *,
    model_name: str,
    seed: int,
    parameter_count: int,
    training_seconds: float,
    final_pde_loss: float,
    peak_rss_mb: float,
    peak_rss_increment_mb: float,
    steps: int | None = None,
    experiment: str = "wavepacket_shared_spectral_ltpb",
    derivative_batch_size: int = 2048,
) -> tuple[dict[str, MetricValue], pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate a learned model or isolated spectral oracle on frozen grids."""

    model.eval()
    device = _model_device(model)
    x = cfg.x_min + cfg.length * torch.arange(
        cfg.test_x_points, dtype=torch.float64, device=device
    ) / cfg.test_x_points
    times = torch.linspace(
        0.0, cfg.t_max, cfg.test_t_points, dtype=torch.float64, device=device
    )
    wave_numbers = training_wave_numbers(cfg, device=device)
    t_mesh_k, k_mesh = torch.meshgrid(times, wave_numbers, indexing="ij")
    spectral_shape = (cfg.test_t_points, cfg.mode_count, 2)
    exact_spectral = exact_spectral_state(
        k_mesh.reshape(-1, 1), t_mesh_k.reshape(-1, 1), cfg
    ).reshape(spectral_shape)
    exact_physical = reconstruct_physical_grid(
        exact_spectral, wave_numbers, x, length=cfg.length
    )

    is_spectral = isinstance(model, SharedSpectralModel) or model_name == "spectral_oracle"
    if is_spectral:
        predicted_spectral_raw = _batched_model(
            model, k_mesh.reshape(-1, 1), t_mesh_k.reshape(-1, 1)
        ).reshape(spectral_shape)
        predicted_physical_raw = reconstruct_physical_grid(
            predicted_spectral_raw, wave_numbers, x, length=cfg.length
        )
    else:
        t_mesh_x, x_mesh = torch.meshgrid(times, x, indexing="ij")
        predicted_physical_raw = _batched_model(
            model, x_mesh.reshape(-1, 1), t_mesh_x.reshape(-1, 1)
        ).reshape(cfg.test_t_points, cfg.test_x_points, 2)

    predicted_raw_np = predicted_physical_raw.cpu().numpy()
    exact_np = exact_physical.cpu().numpy()
    predicted_np, global_phase = _global_gauge_align(predicted_raw_np, exact_np)
    predicted_physical = torch.as_tensor(
        predicted_np, dtype=torch.complex128, device=device
    )
    predicted_spectral = spatial_fourier_coefficients(
        predicted_physical, x, wave_numbers, length=cfg.length
    )
    aligned_exact_spectral = spatial_fourier_coefficients(
        exact_physical, x, wave_numbers, length=cfg.length
    )
    coefficients = project_energy_branches(
        predicted_spectral, wave_numbers, cfg.mass
    ).cpu().numpy()
    exact_coefficients = project_energy_branches(
        aligned_exact_spectral, wave_numbers, cfg.mass
    ).cpu().numpy()

    rho_pred_t, current_pred_t = density_current(predicted_physical.reshape(-1, 2))
    rho_exact_t, current_exact_t = density_current(exact_physical.reshape(-1, 2))
    physical_shape = (cfg.test_t_points, cfg.test_x_points)
    rho_pred = rho_pred_t.cpu().numpy().reshape(physical_shape)
    rho_exact = rho_exact_t.cpu().numpy().reshape(physical_shape)
    current_pred = current_pred_t.cpu().numpy().reshape(physical_shape)
    current_exact = current_exact_t.cpu().numpy().reshape(physical_shape)

    overlap_t = cfg.length * np.mean(
        np.sum(np.conj(exact_np) * predicted_np, axis=2), axis=1
    )
    predicted_norm_t = cfg.length * np.mean(
        np.sum(np.abs(predicted_np) ** 2, axis=2), axis=1
    )
    exact_norm_t = cfg.length * np.mean(
        np.sum(np.abs(exact_np) ** 2, axis=2), axis=1
    )
    fidelity = np.clip(
        np.abs(overlap_t) ** 2
        / np.maximum(predicted_norm_t * exact_norm_t, 1.0e-30),
        0.0,
        1.0,
    )

    if is_spectral:
        spectral_residual = _spectral_residual_grid(
            model,
            k_mesh.reshape(-1, 1),
            t_mesh_k.reshape(-1, 1),
            mass=cfg.mass,
            batch_size=derivative_batch_size,
        ).reshape(spectral_shape)
        physical_residual = reconstruct_physical_grid(
            spectral_residual, wave_numbers, x, length=cfg.length
        )
        spectral_parseval_rms = float(
            torch.sqrt(
                torch.mean(torch.sum(torch.abs(spectral_residual) ** 2, dim=1))
                / cfg.length
            ).cpu()
        )
    else:
        t_mesh_x, x_mesh = torch.meshgrid(times, x, indexing="ij")
        physical_residual = _physical_residual_grid(
            model,
            x_mesh.reshape(-1, 1),
            t_mesh_x.reshape(-1, 1),
            mass=cfg.mass,
            batch_size=derivative_batch_size,
        ).reshape(cfg.test_t_points, cfg.test_x_points, 2)
        projected_residual = spatial_fourier_coefficients(
            physical_residual, x, wave_numbers, length=cfg.length
        )
        spectral_residual = projected_residual
        spectral_parseval_rms = float(
            torch.sqrt(torch.mean(torch.abs(physical_residual) ** 2)).cpu()
        )
    dense_physical_pde_rms = float(
        torch.sqrt(torch.mean(torch.abs(physical_residual) ** 2)).cpu()
    )

    energies = np.asarray(reference_energies(wave_numbers.cpu().numpy(), cfg.mass))
    targets = np.stack((-energies, energies), axis=1)
    initial = (
        gaussian_spectral_envelope(wave_numbers, cfg).cpu().numpy() / math.sqrt(2.0)
    )
    phase_metrics, phase_fields = _phase_diagnostics(
        coefficients,
        exact_coefficients,
        times.cpu().numpy(),
        targets,
        initial,
    )
    initial_weight_total = cfg.test_t_points * float(
        np.sum(np.repeat(initial[:, None] ** 2, 2, axis=1))
    )
    amplitude_drift = float(
        np.sqrt(
            np.sum((np.abs(coefficients) - initial[None, :, None]) ** 2)
            / max(initial_weight_total, 1.0e-30)
        )
    )
    occupations = np.abs(coefficients) ** 2
    exact_occupations = np.abs(exact_coefficients) ** 2
    branch_metrics, branch_fields = _branch_diagnostics(
        coefficients,
        exact_coefficients,
        wave_numbers.cpu(),
        x.cpu(),
        times.cpu().numpy(),
        cfg,
    )
    heldout_metrics, heldout_fields = _heldout_diagnostics(
        model,
        cfg,
        times.cpu(),
        global_phase,
        derivative_batch_size=derivative_batch_size,
    )

    norms = cfg.length * np.mean(rho_pred, axis=1)
    exact_norms = cfg.length * np.mean(rho_exact, axis=1)
    coefficient_error_by_k = np.sqrt(
        np.mean(np.abs(coefficients - exact_coefficients) ** 2, axis=(0, 2))
    ) / np.maximum(
        np.sqrt(np.mean(np.abs(exact_coefficients) ** 2, axis=(0, 2))), 1.0e-30
    )
    occupation_error_by_k = np.sqrt(
        np.mean((occupations - exact_occupations) ** 2, axis=(0, 2))
    ) / np.maximum(
        np.sqrt(np.mean(exact_occupations**2, axis=(0, 2))), 1.0e-30
    )
    metrics: dict[str, MetricValue] = {
        "experiment": experiment,
        "model": model_name,
        "seed": int(seed),
        "steps": cfg.adam_steps if steps is None else int(steps),
        "parameter_count": int(parameter_count),
        "training_seconds": float(training_seconds),
        "peak_rss_mb": float(peak_rss_mb),
        "peak_rss_increment_mb": float(peak_rss_increment_mb),
        "final_loss_pde": float(final_pde_loss),
        "global_gauge_phase_rad": global_phase,
        "spinor_rel_l2_gauge_aligned": _relative_l2(predicted_np, exact_np),
        "spinor_rel_l2_raw": _relative_l2(predicted_raw_np, exact_np),
        "final_spinor_rel_l2_gauge_aligned": _relative_l2(
            predicted_np[-1], exact_np[-1]
        ),
        "density_rel_l2": _relative_l2(rho_pred, rho_exact),
        "current_rel_l2": _relative_l2(current_pred, current_exact),
        "fidelity_mean": float(np.mean(fidelity)),
        "fidelity_min": float(np.min(fidelity)),
        "fidelity_final": float(fidelity[-1]),
        "spectral_coefficient_rel_l2_gauge_aligned": _relative_l2(
            coefficients, exact_coefficients
        ),
        "spectral_occupation_rel_l2": _relative_l2(
            occupations, exact_occupations
        ),
        "training_phase_slope_weighted_relative_error": phase_metrics[
            "weighted_relative_error"
        ],
        "training_phase_min_valid_fraction": phase_metrics["min_valid_fraction"],
        "training_phase_identifiable_fraction": phase_metrics[
            "identifiable_fraction"
        ],
        "training_phase_relative_error_adjacent_jump_rms": phase_metrics[
            "relative_error_adjacent_jump_rms"
        ],
        "training_phase_slope_curvature_rms": phase_metrics[
            "slope_curvature_rms"
        ],
        "modal_amplitude_drift_weighted_rel_rms": amplitude_drift,
        "dense_physical_pde_residual_rms": dense_physical_pde_rms,
        "spectral_training_pde_residual_rms": spectral_parseval_rms,
        "parseval_dense_rms_absolute_difference": abs(
            dense_physical_pde_rms - spectral_parseval_rms
        ) if is_spectral else None,
        "max_norm_drift": float(np.max(np.abs(norms - 1.0))),
        "initial_condition_max_abs_error": float(
            np.max(np.abs(predicted_raw_np[0] - exact_np[0]))
        ),
    }
    metrics.update(branch_metrics)
    metrics.update(heldout_metrics)

    index_x0 = int(np.argmin(np.abs(x.cpu().numpy())))
    time_series_payload: dict[str, np.ndarray] = {
        "t": times.cpu().numpy(),
        "norm_predicted": norms,
        "norm_exact": exact_norms,
        "norm_drift": norms - 1.0,
        "fidelity": fidelity,
        "rho_x0_predicted": rho_pred[:, index_x0],
        "rho_x0_exact": rho_exact[:, index_x0],
        "current_x0_predicted": current_pred[:, index_x0],
        "current_x0_exact": current_exact[:, index_x0],
        "dense_physical_pde_residual_rms_t": torch.sqrt(
            torch.mean(torch.abs(physical_residual) ** 2, dim=(1, 2))
        ).cpu().numpy(),
    }
    for branch in ("plus", "minus"):
        for quantity in ("center", "width", "norm"):
            time_series_payload[f"branch_{branch}_{quantity}"] = branch_fields[
                f"branch_{branch}_{quantity}"
            ]
            time_series_payload[f"branch_{branch}_{quantity}_exact"] = branch_fields[
                f"branch_{branch}_{quantity}_exact"
            ]
    time_series = pd.DataFrame(time_series_payload)

    dense: dict[str, np.ndarray] = {
        "x": x.cpu().numpy(),
        "t": times.cpu().numpy(),
        "training_wave_numbers": wave_numbers.cpu().numpy(),
        "psi_predicted_raw": predicted_raw_np,
        "psi_predicted_gauge_aligned": predicted_np,
        "psi_exact": exact_np,
        "rho_predicted": rho_pred,
        "rho_exact": rho_exact,
        "current_predicted": current_pred,
        "current_exact": current_exact,
        "dirac_residual_physical": physical_residual.cpu().numpy(),
        "dirac_residual_spectral_or_inband_projection": spectral_residual.cpu().numpy(),
        "spectral_state_predicted_gauge_aligned": predicted_spectral.cpu().numpy(),
        "spectral_state_exact": aligned_exact_spectral.cpu().numpy(),
        "branch_coefficients_predicted_gauge_aligned": coefficients,
        "branch_coefficients_exact": exact_coefficients,
        "branch_occupations_predicted": occupations,
        "branch_occupations_exact": exact_occupations,
        "spectral_coefficient_relative_error_by_k": coefficient_error_by_k,
        "spectral_occupation_relative_error_by_k": occupation_error_by_k,
        "global_gauge_phase_rad": np.asarray(global_phase),
        **phase_fields,
        **branch_fields,
        **heldout_fields,
    }
    return metrics, time_series, dense


__all__ = [
    "compute_wavepacket_evaluation",
    "_global_gauge_align",
    "_phase_diagnostics",
    "_circular_center_width",
]
