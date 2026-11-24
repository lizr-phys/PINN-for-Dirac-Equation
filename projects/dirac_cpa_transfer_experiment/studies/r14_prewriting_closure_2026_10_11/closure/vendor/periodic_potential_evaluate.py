from __future__ import annotations

"""Post-training evaluation for the coupled periodic-potential experiment.

The learned models are deliberately treated as black boxes here.  Reference
evolution and the free-energy basis are constructed only inside this
post-training module.  In particular, no reference eigenvalue or time carrier
is exposed to a learned forward pass, loss, optimizer, or checkpoint.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .physics import density_current, sigma_x_apply, sigma_z_apply, spinor_derivatives
from .wavepacket_physics import (
    compute_spectral_basis,
    reconstruct_physical_grid,
    spatial_fourier_coefficients,
)


MetricValue = float | int | str | bool | None


def _cfg_value(cfg: Any, *names: str) -> Any:
    for name in names:
        if hasattr(cfg, name):
            value = getattr(cfg, name)
            return value() if callable(value) else value
    raise AttributeError(f"Configuration is missing all of {names}")


def _wave_numbers(cfg: Any) -> np.ndarray:
    if hasattr(cfg, "wave_numbers"):
        values = getattr(cfg, "wave_numbers")
        values = values() if callable(values) else values
        return np.asarray(values, dtype=np.float64).reshape(-1)
    k0 = float(_cfg_value(cfg, "k0"))
    q = float(_cfg_value(cfg, "coupling_wave_number", "q"))
    truncation = int(_cfg_value(cfg, "J", "j_max", "truncation_j"))
    return k0 + q * np.arange(-truncation, truncation + 1, dtype=np.float64)


def _potential_strength(cfg: Any) -> float:
    return float(_cfg_value(cfg, "v0", "V0", "potential_strength", "selected_v0"))


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return torch.device("cpu") if buffer is None else buffer.device


def _relative_l2(predicted: np.ndarray, exact: np.ndarray) -> float:
    denominator = float(np.linalg.norm(np.asarray(exact).ravel()))
    if denominator <= 0.0:
        raise ValueError("Cannot form relative L2 against a zero reference")
    return float(np.linalg.norm((np.asarray(predicted) - np.asarray(exact)).ravel()) / denominator)


def _global_gauge_align(
    predicted: np.ndarray, exact: np.ndarray
) -> tuple[np.ndarray, float]:
    """Remove exactly one spacetime-independent phase, never per mode/time."""

    overlap = np.vdot(np.asarray(exact).ravel(), np.asarray(predicted).ravel())
    phase = 0.0 if abs(overlap) <= 1.0e-30 else float(np.angle(overlap))
    return np.asarray(predicted) * np.exp(-1j * phase), phase


def _batched_model(
    model: nn.Module,
    first: torch.Tensor,
    second: torch.Tensor,
    *,
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
    pieces: list[torch.Tensor] = []
    for component in range(2):
        real = torch.autograd.grad(
            state[:, component].real,
            time,
            grad_outputs=torch.ones_like(state[:, component].real),
            create_graph=create_graph,
            retain_graph=True,
        )[0].reshape(-1)
        imag = torch.autograd.grad(
            state[:, component].imag,
            time,
            grad_outputs=torch.ones_like(state[:, component].imag),
            create_graph=create_graph,
            retain_graph=True,
        )[0].reshape(-1)
        pieces.append(torch.complex(real, imag))
    return torch.stack(pieces, dim=1)


def _reference_spectral_state(
    cfg: Any, times: np.ndarray, wave_numbers: np.ndarray
) -> np.ndarray:
    """High-accuracy block-Hamiltonian reference used only after training."""

    mass = float(_cfg_value(cfg, "mass"))
    strength = _potential_strength(cfg)
    mode_count = len(wave_numbers)
    hamiltonian = np.zeros((2 * mode_count, 2 * mode_count), dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    for mode, wave_number in enumerate(wave_numbers):
        block = np.asarray(
            [[mass, wave_number], [wave_number, -mass]], dtype=np.complex128
        )
        slc = slice(2 * mode, 2 * mode + 2)
        hamiltonian[slc, slc] = block
        if mode + 1 < mode_count:
            next_slc = slice(2 * (mode + 1), 2 * (mode + 2))
            hamiltonian[slc, next_slc] = 0.5 * strength * identity
            hamiltonian[next_slc, slc] = 0.5 * strength * identity
    if not np.allclose(hamiltonian, hamiltonian.conj().T, rtol=0.0, atol=1.0e-14):
        raise AssertionError("Reference Hamiltonian is not Hermitian")
    basis = compute_spectral_basis(
        torch.as_tensor(wave_numbers, dtype=torch.float64), mass
    )
    central = int(np.argmin(np.abs(wave_numbers - float(_cfg_value(cfg, "k0")))))
    initial = np.zeros(2 * mode_count, dtype=np.complex128)
    initial[2 * central : 2 * central + 2] = (
        basis.u_plus[central].numpy() + basis.u_minus[central].numpy()
    ) / math.sqrt(2.0)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    coordinates = eigenvectors.conj().T @ initial
    evolved = np.einsum(
        "ab,tb,b->ta",
        eigenvectors,
        np.exp(-1j * np.asarray(times)[:, None] * eigenvalues[None, :]),
        coordinates,
        optimize=True,
    )
    return evolved.reshape(len(times), mode_count, 2)


def _project_branches(spectral: np.ndarray, wave_numbers: np.ndarray, mass: float) -> np.ndarray:
    basis = compute_spectral_basis(
        torch.as_tensor(wave_numbers, dtype=torch.float64), mass
    )
    state = torch.as_tensor(spectral, dtype=torch.complex128)
    plus = torch.einsum(
        "kc,...kc->...k", basis.u_plus.to(torch.complex128), state
    )
    minus = torch.einsum(
        "kc,...kc->...k", basis.u_minus.to(torch.complex128), state
    )
    return torch.stack((plus, minus), dim=-1).numpy()


def _branch_phase_and_coherence(
    predicted: np.ndarray,
    exact: np.ndarray,
    *,
    active_threshold: float,
) -> tuple[dict[str, float | None], dict[str, np.ndarray]]:
    """Gauge-invariant active-channel coherence and explicitly masked phases.

    A channel is physically phase-identifiable solely when its *reference*
    occupation reaches the preregistered threshold.  Raw coherences remain
    well-defined if a prediction collapses to zero and therefore do not hide
    that failure.  The optional phase-only RMSE additionally records prediction
    coverage rather than inventing a phase for a zero predicted coefficient.
    """

    predicted_occupation = np.abs(predicted) ** 2
    exact_occupation = np.abs(exact) ** 2
    reference_active = exact_occupation >= float(active_threshold)
    predicted_phase_defined = predicted_occupation >= 1.0e-12
    phase_mask = reference_active & predicted_phase_defined
    phase_error = np.angle(np.exp(1j * (np.angle(predicted) - np.angle(exact))))
    phase_error_masked = np.where(phase_mask, phase_error, np.nan)
    active_count = int(np.count_nonzero(reference_active))
    phase_count = int(np.count_nonzero(phase_mask))
    phase_rmse = (
        float(np.sqrt(np.sum(phase_error[phase_mask]**2) / phase_count))
        if phase_count
        else None
    )

    predicted_flat = predicted.reshape(predicted.shape[0], -1)
    exact_flat = exact.reshape(exact.shape[0], -1)
    active_flat = reference_active.reshape(reference_active.shape[0], -1)
    predicted_coherence = predicted_flat[:, :, None] * np.conj(
        predicted_flat[:, None, :]
    )
    exact_coherence = exact_flat[:, :, None] * np.conj(exact_flat[:, None, :])
    pair_mask = active_flat[:, :, None] & active_flat[:, None, :]
    diagonal = np.eye(predicted_flat.shape[1], dtype=bool)[None, :, :]
    pair_mask &= ~diagonal
    coherence_difference = predicted_coherence - exact_coherence
    denominator = float(np.sum(np.abs(exact_coherence[pair_mask]) ** 2))
    active_coherence_rel_l2 = (
        float(np.sqrt(np.sum(np.abs(coherence_difference[pair_mask]) ** 2) / denominator))
        if denominator > 0.0
        else None
    )
    return (
        {
            "active_branch_phase_circular_rmse": phase_rmse,
            "active_branch_phase_coverage": (
                float(phase_count / active_count) if active_count else None
            ),
            "active_coherence_rel_l2": active_coherence_rel_l2,
            "active_reference_channel_sample_count": float(active_count),
            "active_reference_pair_sample_count": float(np.count_nonzero(pair_mask)),
        },
        {
            "reference_active_branch_mask": reference_active,
            "predicted_phase_defined_mask": predicted_phase_defined,
            "branch_phase_evaluation_mask": phase_mask,
            "branch_phase_error_rad_masked_nan": phase_error_masked,
            "reference_active_pair_mask": pair_mask,
            "channel_coherence_predicted": predicted_coherence,
            "channel_coherence_exact": exact_coherence,
            "channel_coherence_abs_error": np.abs(coherence_difference),
        },
    )


def _coupled_residual_from_state(
    state: torch.Tensor,
    state_t: torch.Tensor,
    wave_numbers: torch.Tensor,
    *,
    mass: float,
    potential_strength: float,
) -> torch.Tensor:
    """Residual on the retained block-tridiagonal ladder."""

    h_state = wave_numbers[None, :, None] * torch.stack(
        (state[:, :, 1], state[:, :, 0]), dim=2
    ) + mass * torch.stack((state[:, :, 0], -state[:, :, 1]), dim=2)
    neighbor = torch.zeros_like(state)
    neighbor[:, 1:] += state[:, :-1]
    neighbor[:, :-1] += state[:, 1:]
    return 1j * state_t - h_state - 0.5 * potential_strength * neighbor


def _spectral_model_residual_grid(
    model: nn.Module,
    wave_numbers: torch.Tensor,
    times: torch.Tensor,
    *,
    mass: float,
    potential_strength: float,
    batch_times: int = 32,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    mode_count = len(wave_numbers)
    for start in range(0, len(times), batch_times):
        time_batch = times[start : start + batch_times]
        t_mesh, k_mesh = torch.meshgrid(time_batch, wave_numbers, indexing="ij")
        k_flat = k_mesh.reshape(-1, 1)
        t_flat = t_mesh.reshape(-1, 1).detach().clone().requires_grad_(True)
        state_flat = model(k_flat, t_flat)
        state_t_flat = _complex_time_derivative(state_flat, t_flat, create_graph=False)
        state = state_flat.reshape(len(time_batch), mode_count, 2)
        state_t = state_t_flat.reshape(len(time_batch), mode_count, 2)
        pieces.append(
            _coupled_residual_from_state(
                state,
                state_t,
                wave_numbers,
                mass=mass,
                potential_strength=potential_strength,
            ).detach()
        )
    return torch.cat(pieces, dim=0)


def _physical_model_residual_grid(
    model: nn.Module,
    x_flat: torch.Tensor,
    t_flat: torch.Tensor,
    *,
    mass: float,
    potential_strength: float,
    q: float,
    batch_size: int,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for start in range(0, len(x_flat), batch_size):
        stop = min(start + batch_size, len(x_flat))
        x = x_flat[start:stop].detach().clone().requires_grad_(True)
        t = t_flat[start:stop].detach().clone().requires_grad_(True)
        psi = model(x, t)
        psi_x, psi_t = spinor_derivatives(psi, x, t, create_graph=False)
        residual = (
            1j * psi_t
            + 1j * sigma_x_apply(psi_x)
            - mass * sigma_z_apply(psi)
            - potential_strength * torch.cos(q * x) * psi
        )
        pieces.append(residual.detach())
    return torch.cat(pieces, dim=0)


def _initial_tangent_diagnostics(model: nn.Module, cfg: Any) -> tuple[dict, dict]:
    """Test the tangent at zero-occupied first sidebands; no training occurs."""
    from .periodic_potential_physics import apply_coupled_hamiltonian_numpy, central_initial_state

    k = torch.tensor(_wave_numbers(cfg), dtype=torch.float64).reshape(-1, 1)
    t = torch.zeros_like(k, requires_grad=True)
    state = model(k, t)
    derivative = _complex_time_derivative(state, t, create_graph=True)
    exact = -1j*apply_coupled_hamiltonian_numpy(
        central_initial_state(cfg), k.detach().numpy().ravel(), float(cfg.mass), _potential_strength(cfg)
    )
    offsets = np.rint((k.detach().numpy().ravel()-float(cfg.k0)) / float(_cfg_value(cfg, "coupling_wave_number", "q")))
    sidebands = np.abs(offsets) == 1
    target = exact[sidebands]
    predicted = derivative.detach().numpy()[sidebands]
    parameters = tuple(model.parameters())
    if parameters:
        gradients = torch.autograd.grad(derivative[sidebands].imag.sum(), parameters, allow_unused=True)
        sensitivity = math.sqrt(sum(float(torch.sum(g*g)) for g in gradients if g is not None))
    else:
        sensitivity = 0.0
    return (
        {"initial_sideband_time_derivative_rel_l2": _relative_l2(predicted, target),
         "initial_sideband_imaginary_tangent_parameter_sensitivity": sensitivity,
         "initial_sideband_required_derivative_norm": float(np.linalg.norm(target))},
        {"initial_time_derivative_predicted": derivative.detach().numpy(),
         "initial_time_derivative_required_by_pde": exact,
         "initial_first_sideband_mask": sidebands},
    )


def compute_periodic_potential_evaluation(
    model: nn.Module,
    cfg: Any,
    *,
    model_name: str,
    seed: int,
    parameter_count: int,
    training_seconds: float,
    final_pde_loss: float,
    peak_rss_mb: float,
    peak_rss_increment_mb: float,
    reference_spectral: np.ndarray | torch.Tensor | None = None,
    steps: int | None = None,
    experiment: str = "periodic_potential_coupled_ltpb",
    derivative_batch_size: int = 2048,
) -> tuple[dict[str, MetricValue], pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate one formal learned run on the frozen dense grid."""

    if model_name not in {
        "physical_vanilla",
        "coupled_shared_spectral_complex",
        "coupled_shared_spectral_ltpb",
    }:
        raise ValueError(f"Unknown learned model: {model_name}")
    model.eval()
    device = _model_device(model)
    length = float(_cfg_value(cfg, "length"))
    x_min = float(_cfg_value(cfg, "x_min"))
    t_max = float(_cfg_value(cfg, "t_max"))
    mass = float(_cfg_value(cfg, "mass"))
    q = float(_cfg_value(cfg, "coupling_wave_number", "q"))
    strength = _potential_strength(cfg)
    n_x = int(_cfg_value(cfg, "test_x_points"))
    n_t = int(_cfg_value(cfg, "test_t_points"))
    active_threshold = float(
        getattr(cfg, "active_occupation_threshold", 1.0e-6)
    )
    wave_numbers_np = _wave_numbers(cfg)
    mode_count = len(wave_numbers_np)
    x = x_min + length * torch.arange(n_x, dtype=torch.float64, device=device) / n_x
    times = torch.linspace(0.0, t_max, n_t, dtype=torch.float64, device=device)
    wave_numbers = torch.as_tensor(wave_numbers_np, dtype=torch.float64, device=device)

    if reference_spectral is None:
        exact_spectral_np = _reference_spectral_state(
            cfg, times.detach().cpu().numpy(), wave_numbers_np
        )
    else:
        exact_spectral_np = np.asarray(
            reference_spectral.detach().cpu().numpy()
            if isinstance(reference_spectral, torch.Tensor)
            else reference_spectral,
            dtype=np.complex128,
        )
    expected_shape = (n_t, mode_count, 2)
    if exact_spectral_np.shape != expected_shape:
        raise ValueError(
            f"reference_spectral has shape {exact_spectral_np.shape}, expected {expected_shape}"
        )
    exact_spectral = torch.as_tensor(
        exact_spectral_np, dtype=torch.complex128, device=device
    )
    exact_physical = reconstruct_physical_grid(
        exact_spectral, wave_numbers, x, length=length
    )

    is_spectral = model_name.startswith("coupled_shared_spectral_")
    if is_spectral:
        t_mesh_k, k_mesh = torch.meshgrid(times, wave_numbers, indexing="ij")
        predicted_spectral_raw = _batched_model(
            model, k_mesh.reshape(-1, 1), t_mesh_k.reshape(-1, 1)
        ).reshape(expected_shape)
        predicted_physical_raw = reconstruct_physical_grid(
            predicted_spectral_raw, wave_numbers, x, length=length
        )
    else:
        t_mesh_x, x_mesh = torch.meshgrid(times, x, indexing="ij")
        predicted_physical_raw = _batched_model(
            model, x_mesh.reshape(-1, 1), t_mesh_x.reshape(-1, 1)
        ).reshape(n_t, n_x, 2)
        predicted_spectral_raw = spatial_fourier_coefficients(
            predicted_physical_raw, x, wave_numbers, length=length
        )

    predicted_raw_np = predicted_physical_raw.detach().cpu().numpy()
    exact_np = exact_physical.detach().cpu().numpy()
    predicted_np, global_phase = _global_gauge_align(predicted_raw_np, exact_np)
    gauge_factor = np.exp(-1j * global_phase)
    predicted_spectral_np = (
        predicted_spectral_raw.detach().cpu().numpy() * gauge_factor
    )
    predicted_physical = torch.as_tensor(
        predicted_np, dtype=torch.complex128, device=device
    )

    rho_pred_t, current_pred_t = density_current(predicted_physical.reshape(-1, 2))
    rho_exact_t, current_exact_t = density_current(exact_physical.reshape(-1, 2))
    rho_pred = rho_pred_t.reshape(n_t, n_x).cpu().numpy()
    rho_exact = rho_exact_t.reshape(n_t, n_x).cpu().numpy()
    current_pred = current_pred_t.reshape(n_t, n_x).cpu().numpy()
    current_exact = current_exact_t.reshape(n_t, n_x).cpu().numpy()
    predicted_norm = length * np.mean(rho_pred, axis=1)
    exact_norm = length * np.mean(rho_exact, axis=1)
    overlap_t = length * np.mean(
        np.sum(np.conj(exact_np) * predicted_np, axis=2), axis=1
    )
    fidelity = np.clip(
        np.abs(overlap_t) ** 2
        / np.maximum(predicted_norm * exact_norm, 1.0e-30),
        0.0,
        1.0,
    )

    predicted_coefficients = _project_branches(
        predicted_spectral_np, wave_numbers_np, mass
    )
    exact_coefficients = _project_branches(exact_spectral_np, wave_numbers_np, mass)
    predicted_modal_occupations = np.sum(np.abs(predicted_spectral_np) ** 2, axis=2)
    exact_modal_occupations = np.sum(np.abs(exact_spectral_np) ** 2, axis=2)
    predicted_branch_occupations = np.abs(predicted_coefficients) ** 2
    exact_branch_occupations = np.abs(exact_coefficients) ** 2
    phase_metrics, phase_dense = _branch_phase_and_coherence(
        predicted_coefficients,
        exact_coefficients,
        active_threshold=active_threshold,
    )

    offsets = np.rint(
        (wave_numbers_np - float(_cfg_value(cfg, "k0"))) / q
    ).astype(np.int64)
    selected_metrics: dict[str, float] = {}
    selected_indices: dict[str, int] = {}
    for label, offset in (("center", 0), ("lower_sideband", -1), ("upper_sideband", 1)):
        matches = np.flatnonzero(offsets == offset)
        if len(matches) != 1:
            raise ValueError(f"Expected one {label} mode at offset {offset}")
        index = int(matches[0])
        selected_indices[label] = index
        pred = predicted_modal_occupations[:, index]
        exact = exact_modal_occupations[:, index]
        selected_metrics[f"{label}_occupation_rel_l2"] = _relative_l2(pred, exact)
        selected_metrics[f"{label}_occupation_rmse"] = float(
            np.sqrt(np.mean((pred - exact) ** 2))
        )
        selected_metrics[f"{label}_occupation_max_abs_error"] = float(
            np.max(np.abs(pred - exact))
        )

    retained_norm = np.sum(predicted_modal_occupations, axis=1)
    out_of_ladder_fraction = np.maximum(predicted_norm - retained_norm, 0.0) / np.maximum(
        predicted_norm, 1.0e-30
    )

    if is_spectral:
        spectral_residual = _spectral_model_residual_grid(
            model,
            wave_numbers,
            times,
            mass=mass,
            potential_strength=strength,
        )
        retained_spectral_residual_np = spectral_residual.cpu().numpy()
        # Add the two exterior residuals generated by multiplying the edge
        # modes by cos(qx); this makes the dense physical Parseval diagnostic
        # honest even though the trained ODE residual is truncated.
        extended = torch.zeros(
            (n_t, mode_count + 2, 2), dtype=torch.complex128, device=device
        )
        extended[:, 1:-1] = spectral_residual
        extended[:, 0] = -0.5 * strength * predicted_spectral_raw[:, 0]
        extended[:, -1] = -0.5 * strength * predicted_spectral_raw[:, -1]
        extended_wave_numbers = torch.cat(
            (wave_numbers[:1] - q, wave_numbers, wave_numbers[-1:] + q)
        )
        physical_residual = reconstruct_physical_grid(
            extended, extended_wave_numbers, x, length=length
        )
        full_spectral_parseval_rms = float(
            torch.sqrt(torch.mean(torch.sum(torch.abs(extended) ** 2, dim=1)) / length)
            .cpu()
        )
        tangent_metrics, tangent_dense = _initial_tangent_diagnostics(model, cfg)
    else:
        t_mesh_x, x_mesh = torch.meshgrid(times, x, indexing="ij")
        physical_residual = _physical_model_residual_grid(
            model,
            x_mesh.reshape(-1, 1),
            t_mesh_x.reshape(-1, 1),
            mass=mass,
            potential_strength=strength,
            q=q,
            batch_size=derivative_batch_size,
        ).reshape(n_t, n_x, 2)
        retained_projection = spatial_fourier_coefficients(
            physical_residual, x, wave_numbers, length=length
        )
        retained_spectral_residual_np = retained_projection.cpu().numpy()
        full_spectral_parseval_rms = math.nan
        tangent_metrics, tangent_dense = {}, {}
    dense_physical_pde_rms = float(
        torch.sqrt(torch.mean(torch.abs(physical_residual) ** 2)).cpu()
    )
    retained_spectral_pde_rms = float(
        np.sqrt(np.mean(np.sum(np.abs(retained_spectral_residual_np) ** 2, axis=1)) / length)
    )

    metrics: dict[str, MetricValue] = {
        "experiment": experiment,
        "model": model_name,
        "seed": int(seed),
        "steps": int(steps if steps is not None else _cfg_value(cfg, "adam_steps")),
        "parameter_count": int(parameter_count),
        "training_seconds": float(training_seconds),
        "peak_rss_mb": float(peak_rss_mb),
        "peak_rss_increment_mb": float(peak_rss_increment_mb),
        "final_loss_pde": float(final_pde_loss),
        "global_gauge_phase_rad": global_phase,
        "spinor_rel_l2_gauge_aligned": _relative_l2(predicted_np, exact_np),
        "spinor_rel_l2_raw": _relative_l2(predicted_raw_np, exact_np),
        "final_spinor_rel_l2_gauge_aligned": _relative_l2(predicted_np[-1], exact_np[-1]),
        "retained_spectral_spinor_rel_l2_gauge_aligned": _relative_l2(
            predicted_spectral_np, exact_spectral_np
        ),
        "density_rel_l2": _relative_l2(rho_pred, rho_exact),
        "current_rel_l2": _relative_l2(current_pred, current_exact),
        "fidelity_mean": float(np.mean(fidelity)),
        "fidelity_min": float(np.min(fidelity)),
        "fidelity_final": float(fidelity[-1]),
        "modal_population_rel_l2": _relative_l2(
            predicted_modal_occupations, exact_modal_occupations
        ),
        "branch_population_rel_l2": _relative_l2(
            predicted_branch_occupations, exact_branch_occupations
        ),
        "max_norm_drift": float(np.max(np.abs(predicted_norm - 1.0))),
        "max_out_of_ladder_norm_fraction": float(np.max(out_of_ladder_fraction)),
        "mean_out_of_ladder_norm_fraction": float(np.mean(out_of_ladder_fraction)),
        "dense_physical_pde_residual_rms": dense_physical_pde_rms,
        "retained_coupled_spectral_pde_residual_rms": retained_spectral_pde_rms,
        "full_extended_spectral_parseval_rms": (
            full_spectral_parseval_rms if is_spectral else None
        ),
        "parseval_dense_rms_absolute_difference": (
            abs(dense_physical_pde_rms - full_spectral_parseval_rms)
            if is_spectral
            else None
        ),
        "initial_condition_max_abs_error": float(
            np.max(np.abs(predicted_raw_np[0] - exact_np[0]))
        ),
        "active_occupation_threshold": active_threshold,
        **phase_metrics,
        **selected_metrics,
        **tangent_metrics,
    }

    time_series_payload: dict[str, np.ndarray] = {
        "t": times.cpu().numpy(),
        "norm_predicted": predicted_norm,
        "norm_exact": exact_norm,
        "norm_drift": predicted_norm - 1.0,
        "fidelity": fidelity,
        "out_of_ladder_norm_fraction": out_of_ladder_fraction,
        "dense_physical_pde_residual_rms_t": torch.sqrt(
            torch.mean(torch.abs(physical_residual) ** 2, dim=(1, 2))
        ).cpu().numpy(),
    }
    for mode, offset in enumerate(offsets):
        prefix = f"mode_{int(offset):+d}"
        time_series_payload[f"{prefix}_occupation_predicted"] = predicted_modal_occupations[:, mode]
        time_series_payload[f"{prefix}_occupation_exact"] = exact_modal_occupations[:, mode]
        for branch, branch_name in enumerate(("plus", "minus")):
            time_series_payload[
                f"{prefix}_{branch_name}_occupation_predicted"
            ] = predicted_branch_occupations[:, mode, branch]
            time_series_payload[
                f"{prefix}_{branch_name}_occupation_exact"
            ] = exact_branch_occupations[:, mode, branch]
            time_series_payload[
                f"{prefix}_{branch_name}_reference_phase_active"
            ] = phase_dense["reference_active_branch_mask"][:, mode, branch].astype(np.int8)
    time_series = pd.DataFrame(time_series_payload)

    dense: dict[str, np.ndarray] = {
        "x": x.cpu().numpy(),
        "t": times.cpu().numpy(),
        "wave_numbers": wave_numbers_np,
        "mode_offsets": offsets,
        "psi_predicted_raw": predicted_raw_np,
        "psi_predicted_gauge_aligned": predicted_np,
        "psi_exact": exact_np,
        "rho_predicted": rho_pred,
        "rho_exact": rho_exact,
        "current_predicted": current_pred,
        "current_exact": current_exact,
        "spectral_state_predicted_raw": predicted_spectral_raw.cpu().numpy(),
        "spectral_state_predicted_gauge_aligned": predicted_spectral_np,
        "spectral_state_exact": exact_spectral_np,
        "modal_occupations_predicted": predicted_modal_occupations,
        "modal_occupations_exact": exact_modal_occupations,
        "branch_coefficients_predicted_gauge_aligned": predicted_coefficients,
        "branch_coefficients_exact": exact_coefficients,
        "branch_occupations_predicted": predicted_branch_occupations,
        "branch_occupations_exact": exact_branch_occupations,
        "dirac_residual_physical": physical_residual.cpu().numpy(),
        "retained_coupled_spectral_residual_or_projection": retained_spectral_residual_np,
        "global_gauge_phase_rad": np.asarray(global_phase),
        "selected_mode_indices_center_lower_upper": np.asarray(
            [
                selected_indices["center"],
                selected_indices["lower_sideband"],
                selected_indices["upper_sideband"],
            ],
            dtype=np.int64,
        ),
        **phase_dense,
        **tangent_dense,
    }
    if hasattr(model, "branch_fields"):
        with torch.no_grad():
            a, s = model.branch_fields(k_mesh.reshape(-1, 1), t_mesh_k.reshape(-1, 1))
        dense["internal_real_amplitudes"] = a.reshape(n_t, mode_count, 2).numpy()
        dense["internal_phases"] = s.reshape(n_t, mode_count, 2).numpy()
    return metrics, time_series, dense


__all__ = [
    "compute_periodic_potential_evaluation",
    "_global_gauge_align",
    "_relative_l2",
    "_reference_spectral_state",
    "_branch_phase_and_coherence",
    "_coupled_residual_from_state",
]
