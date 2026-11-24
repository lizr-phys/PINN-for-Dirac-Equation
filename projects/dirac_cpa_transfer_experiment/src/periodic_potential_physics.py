from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .periodic_potential_config import PeriodicPotentialConfig


@dataclass(frozen=True)
class ReferenceSolution:
    """Machine-precision solution of the finite coupled momentum ladder."""

    times: np.ndarray  # (n_t,)
    ladder_indices: np.ndarray  # (n_k,)
    wave_numbers: np.ndarray  # (n_k,)
    state: np.ndarray  # (n_t,n_k,2), complex128
    time_derivative: np.ndarray  # (n_t,n_k,2), complex128
    eigenvalues: np.ndarray  # (2*n_k,)
    eigenvectors: np.ndarray  # (2*n_k,2*n_k)
    initial_state: np.ndarray  # (n_k,2)
    mass: float
    potential_strength: float
    coupling_wave_number: float
    ladder_half_width: int

    @property
    def modal_occupation(self) -> np.ndarray:
        return np.sum(np.abs(self.state) ** 2, axis=-1)

    @property
    def total_norm(self) -> np.ndarray:
        return np.sum(self.modal_occupation, axis=-1)

    @property
    def branch_coefficients(self) -> np.ndarray:
        return project_energy_branches_numpy(
            self.state, self.wave_numbers, self.mass
        )

    @property
    def branch_occupation(self) -> np.ndarray:
        return np.abs(self.branch_coefficients) ** 2


def ladder_indices(half_width: int) -> np.ndarray:
    if int(half_width) != half_width or half_width < 1:
        raise ValueError("half_width must be a positive integer")
    return np.arange(-int(half_width), int(half_width) + 1, dtype=np.int64)


def ladder_wave_numbers(
    cfg: PeriodicPotentialConfig,
    *,
    half_width: int | None = None,
) -> np.ndarray:
    width = cfg.ladder_half_width if half_width is None else int(half_width)
    return cfg.k0 + cfg.coupling_wave_number * ladder_indices(width)


def dirac_symbol_numpy(wave_number: float, mass: float) -> np.ndarray:
    """Return ``k sigma_x + m sigma_z`` as a real symmetric matrix."""

    return np.asarray(
        [[float(mass), float(wave_number)], [float(wave_number), -float(mass)]],
        dtype=np.float64,
    )


def energy_basis_numpy(
    wave_numbers: np.ndarray | list[float] | tuple[float, ...],
    mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous real free-energy basis, without returning eigenvalues."""

    k = np.asarray(wave_numbers, dtype=np.float64).reshape(-1)
    half_angle = 0.5 * np.arctan2(k, float(mass))
    cosine = np.cos(half_angle)
    sine = np.sin(half_angle)
    u_plus = np.stack((cosine, sine), axis=-1)
    u_minus = np.stack((-sine, cosine), axis=-1)
    return u_plus, u_minus


def build_coupled_hamiltonian(
    cfg: PeriodicPotentialConfig,
    *,
    half_width: int | None = None,
    potential_strength: float | None = None,
) -> np.ndarray:
    """Construct the block-tridiagonal Hermitian spectral Hamiltonian.

    The off-diagonal blocks are ``V0 I_2 / 2``.  No analytic free dispersion
    is used; the diagonal blocks are the original non-diagonal Dirac symbols.
    """

    width = cfg.ladder_half_width if half_width is None else int(half_width)
    v0 = cfg.potential_strength if potential_strength is None else float(
        potential_strength
    )
    if v0 < 0.0:
        raise ValueError("potential_strength cannot be negative")
    wave_numbers = ladder_wave_numbers(cfg, half_width=width)
    mode_count = wave_numbers.size
    hamiltonian = np.zeros((2 * mode_count, 2 * mode_count), dtype=np.complex128)
    coupling = 0.5 * v0 * np.eye(2, dtype=np.complex128)
    for mode, wave_number in enumerate(wave_numbers):
        row = slice(2 * mode, 2 * mode + 2)
        hamiltonian[row, row] = dirac_symbol_numpy(wave_number, cfg.mass)
        if mode + 1 < mode_count:
            neighbor = slice(2 * (mode + 1), 2 * (mode + 1) + 2)
            hamiltonian[row, neighbor] = coupling
            hamiltonian[neighbor, row] = coupling
    return hamiltonian


def central_initial_state(
    cfg: PeriodicPotentialConfig,
    *,
    half_width: int | None = None,
) -> np.ndarray:
    """Initial state with only ``j=0`` and equal free +/- branches occupied."""

    width = cfg.ladder_half_width if half_width is None else int(half_width)
    state = np.zeros((2 * width + 1, 2), dtype=np.complex128)
    u_plus, u_minus = energy_basis_numpy(np.asarray([cfg.k0]), cfg.mass)
    state[width] = (u_plus[0] + u_minus[0]) / math.sqrt(2.0)
    return state


def apply_coupled_hamiltonian_numpy(
    state: np.ndarray,
    wave_numbers: np.ndarray,
    mass: float,
    potential_strength: float,
) -> np.ndarray:
    """Apply the full coupled spinor equation to ``(...,n_k,2)`` states."""

    values = np.asarray(state, dtype=np.complex128)
    k = np.asarray(wave_numbers, dtype=np.float64).reshape(-1)
    if values.shape[-2:] != (k.size, 2):
        raise ValueError("state must end with (number of modes, 2)")
    result = np.empty_like(values)
    result[..., :, 0] = float(mass) * values[..., :, 0] + k * values[..., :, 1]
    result[..., :, 1] = k * values[..., :, 0] - float(mass) * values[..., :, 1]
    coupling = 0.5 * float(potential_strength)
    result[..., 1:, :] += coupling * values[..., :-1, :]
    result[..., :-1, :] += coupling * values[..., 1:, :]
    return result


def apply_coupled_hamiltonian_torch(
    state: torch.Tensor,
    wave_numbers: torch.Tensor,
    mass: float,
    potential_strength: float,
) -> torch.Tensor:
    """Torch counterpart used by learned strong-residual code.

    This operates on the original coupled spinor equation and contains no
    analytic energy, target phase or reference trajectory.
    """

    k = wave_numbers.reshape(-1)
    if state.shape[-2:] != (k.numel(), 2):
        raise ValueError("state must end with (number of modes, 2)")
    result = torch.empty_like(state)
    result[..., :, 0] = float(mass) * state[..., :, 0] + k * state[..., :, 1]
    result[..., :, 1] = k * state[..., :, 0] - float(mass) * state[..., :, 1]
    coupling = 0.5 * float(potential_strength)
    result[..., 1:, :] = result[..., 1:, :] + coupling * state[..., :-1, :]
    result[..., :-1, :] = result[..., :-1, :] + coupling * state[..., 1:, :]
    return result


def coupled_spectral_residual(
    state: torch.Tensor,
    time_derivative: torch.Tensor,
    wave_numbers: torch.Tensor,
    mass: float,
    potential_strength: float,
) -> torch.Tensor:
    if time_derivative.shape != state.shape:
        raise ValueError("state and time_derivative must have identical shapes")
    return 1j * time_derivative - apply_coupled_hamiltonian_torch(
        state, wave_numbers, mass, potential_strength
    )


def solve_reference_eigendecomposition(
    cfg: PeriodicPotentialConfig,
    times: np.ndarray | list[float] | tuple[float, ...],
    *,
    half_width: int | None = None,
    potential_strength: float | None = None,
) -> ReferenceSolution:
    """Solve the finite ladder by one Hermitian eigendecomposition."""

    width = cfg.ladder_half_width if half_width is None else int(half_width)
    v0 = cfg.potential_strength if potential_strength is None else float(
        potential_strength
    )
    time = np.asarray(times, dtype=np.float64).reshape(-1)
    if time.size == 0 or not np.all(np.isfinite(time)):
        raise ValueError("times must contain finite values")
    hamiltonian = build_coupled_hamiltonian(
        cfg, half_width=width, potential_strength=v0
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    initial = central_initial_state(cfg, half_width=width)
    initial_flat = initial.reshape(-1)
    eigen_coefficients = eigenvectors.conj().T @ initial_flat
    carriers = np.exp(-1j * time[:, None] * eigenvalues[None, :])
    evolved_eigen = carriers * eigen_coefficients[None, :]
    state_flat = evolved_eigen @ eigenvectors.T
    derivative_flat = (
        -1j * eigenvalues[None, :] * evolved_eigen
    ) @ eigenvectors.T
    mode_count = 2 * width + 1
    return ReferenceSolution(
        times=time,
        ladder_indices=ladder_indices(width),
        wave_numbers=ladder_wave_numbers(cfg, half_width=width),
        state=np.asarray(state_flat.reshape(-1, mode_count, 2), np.complex128),
        time_derivative=np.asarray(
            derivative_flat.reshape(-1, mode_count, 2), np.complex128
        ),
        eigenvalues=np.asarray(eigenvalues, np.float64),
        eigenvectors=np.asarray(eigenvectors, np.complex128),
        initial_state=initial,
        mass=cfg.mass,
        potential_strength=v0,
        coupling_wave_number=cfg.coupling_wave_number,
        ladder_half_width=width,
    )


def propagate_reference_state(
    state: np.ndarray,
    delta_t: float,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Apply the stored exact unitary propagator to one flattened state."""

    values = np.asarray(state, dtype=np.complex128)
    original_shape = values.shape
    flat = values.reshape(-1)
    if flat.size != eigenvalues.size:
        raise ValueError("state size and eigensystem size do not match")
    coefficients = eigenvectors.conj().T @ flat
    evolved = eigenvectors @ (np.exp(-1j * float(delta_t) * eigenvalues) * coefficients)
    return evolved.reshape(original_shape)


def project_energy_branches_numpy(
    spectral_state: np.ndarray,
    wave_numbers: np.ndarray,
    mass: float,
) -> np.ndarray:
    values = np.asarray(spectral_state, dtype=np.complex128)
    k = np.asarray(wave_numbers, dtype=np.float64).reshape(-1)
    if values.shape[-2:] != (k.size, 2):
        raise ValueError("spectral_state must end with (number of modes, 2)")
    u_plus, u_minus = energy_basis_numpy(k, mass)
    positive = np.einsum("kc,...kc->...k", u_plus, values)
    negative = np.einsum("kc,...kc->...k", u_minus, values)
    return np.stack((positive, negative), axis=-1)


def reconstruct_physical_numpy(
    spectral_state: np.ndarray,
    wave_numbers: np.ndarray,
    positions: np.ndarray,
    *,
    length: float,
) -> np.ndarray:
    values = np.asarray(spectral_state, dtype=np.complex128)
    k = np.asarray(wave_numbers, dtype=np.float64).reshape(-1)
    x = np.asarray(positions, dtype=np.float64).reshape(-1)
    if values.shape[-2:] != (k.size, 2):
        raise ValueError("spectral_state must end with (number of modes, 2)")
    spatial = np.exp(1j * x[:, None] * k[None, :])
    return np.einsum("xk,...kc->...xc", spatial, values) / math.sqrt(length)


def density_and_current_numpy(spinor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(spinor, dtype=np.complex128)
    if values.shape[-1] != 2:
        raise ValueError("spinor must have final dimension two")
    density = np.sum(np.abs(values) ** 2, axis=-1)
    current = 2.0 * np.real(np.conj(values[..., 0]) * values[..., 1])
    return density, current


def relative_l2(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    align_global_phase: bool = False,
) -> float:
    predicted = np.asarray(prediction, dtype=np.complex128)
    exact = np.asarray(reference, dtype=np.complex128)
    if predicted.shape != exact.shape:
        raise ValueError("prediction and reference must have identical shapes")
    denominator = float(np.linalg.norm(exact.reshape(-1)))
    if denominator == 0.0:
        raise ValueError("reference norm cannot be zero")
    if align_global_phase:
        overlap = np.vdot(predicted.reshape(-1), exact.reshape(-1))
        if abs(overlap) > 0.0:
            predicted = predicted * np.exp(1j * np.angle(overlap))
    return float(np.linalg.norm((predicted - exact).reshape(-1)) / denominator)


def embed_small_ladder(
    small_state: np.ndarray,
    *,
    small_half_width: int,
    large_half_width: int,
) -> np.ndarray:
    if large_half_width < small_half_width:
        raise ValueError("large_half_width cannot be smaller than small_half_width")
    values = np.asarray(small_state, dtype=np.complex128)
    if values.shape[-2:] != (2 * small_half_width + 1, 2):
        raise ValueError("small_state does not match small_half_width")
    result = np.zeros(
        values.shape[:-2] + (2 * large_half_width + 1, 2),
        dtype=np.complex128,
    )
    offset = large_half_width - small_half_width
    result[..., offset : offset + 2 * small_half_width + 1, :] = values
    return result


def compare_reference_truncations(
    small: ReferenceSolution,
    large: ReferenceSolution,
) -> dict[str, float]:
    """Return auditable J-small/J-large complex-spinor comparisons.

    Metrics are reported both after zero-padding the small ladder and on the
    common mode slice.  Each has full-trajectory, maximum-over-time, final and
    one-global-constant-phase-aligned variants.  No per-time or per-mode phase
    alignment is performed.
    """

    if not np.array_equal(small.times, large.times):
        raise ValueError("reference time grids must match")
    if large.ladder_half_width < small.ladder_half_width:
        raise ValueError("large reference must have the wider ladder")
    padded = embed_small_ladder(
        small.state,
        small_half_width=small.ladder_half_width,
        large_half_width=large.ladder_half_width,
    )
    offset = large.ladder_half_width - small.ladder_half_width
    common_large = large.state[
        :, offset : offset + 2 * small.ladder_half_width + 1, :
    ]

    def time_errors(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
        numerator = np.linalg.norm(
            (prediction - reference).reshape(prediction.shape[0], -1), axis=1
        )
        denominator = np.linalg.norm(
            reference.reshape(reference.shape[0], -1), axis=1
        )
        return numerator / np.maximum(denominator, 1.0e-300)

    padded_time = time_errors(padded, large.state)
    common_time = time_errors(small.state, common_large)
    small_population = np.sum(np.abs(small.state) ** 2, axis=-1)
    large_population = np.sum(np.abs(common_large) ** 2, axis=-1)

    return {
        "padded_full_trajectory_spinor_relative_l2": relative_l2(
            padded, large.state
        ),
        "padded_full_trajectory_global_phase_aligned_relative_l2": relative_l2(
            padded, large.state, align_global_phase=True
        ),
        "padded_max_time_spinor_relative_l2": float(np.max(padded_time)),
        "padded_final_spinor_relative_l2": float(padded_time[-1]),
        "common_full_trajectory_spinor_relative_l2": relative_l2(
            small.state, common_large
        ),
        "common_full_trajectory_global_phase_aligned_relative_l2": relative_l2(
            small.state, common_large, align_global_phase=True
        ),
        "common_max_time_spinor_relative_l2": float(np.max(common_time)),
        "common_final_spinor_relative_l2": float(common_time[-1]),
        "common_modal_population_relative_l2": relative_l2(
            small_population, large_population
        ),
        "common_modal_population_max_absolute_error": float(
            np.max(np.abs(small_population - large_population))
        ),
    }


def reference_solution_checks(solution: ReferenceSolution) -> dict[str, float]:
    hamiltonian = build_hamiltonian_from_solution(solution)
    hermiticity_error = float(
        np.max(np.abs(hamiltonian - hamiltonian.conj().T))
    )
    norm_drift = float(np.max(np.abs(solution.total_norm - solution.total_norm[0])))
    recovered = propagate_reference_state(
        solution.state[-1],
        -float(solution.times[-1] - solution.times[0]),
        solution.eigenvalues,
        solution.eigenvectors,
    )
    time_reversal_error = relative_l2(recovered, solution.state[0])
    residual = 1j * solution.time_derivative - apply_coupled_hamiltonian_numpy(
        solution.state,
        solution.wave_numbers,
        solution.mass,
        solution.potential_strength,
    )
    residual_rms = float(np.sqrt(np.mean(np.abs(residual) ** 2)))
    return {
        "hermiticity_max_abs": hermiticity_error,
        "norm_initial": float(solution.total_norm[0]),
        "norm_max_drift": norm_drift,
        "time_reversal_relative_l2": time_reversal_error,
        "spectral_pde_residual_rms": residual_rms,
        "initial_state_relative_l2": relative_l2(
            solution.state[0], solution.initial_state
        ),
    }


def build_hamiltonian_from_solution(solution: ReferenceSolution) -> np.ndarray:
    mode_count = solution.wave_numbers.size
    hamiltonian = np.zeros((2 * mode_count, 2 * mode_count), dtype=np.complex128)
    coupling = 0.5 * solution.potential_strength * np.eye(2)
    for mode, wave_number in enumerate(solution.wave_numbers):
        current = slice(2 * mode, 2 * mode + 2)
        hamiltonian[current, current] = dirac_symbol_numpy(
            wave_number, solution.mass
        )
        if mode + 1 < mode_count:
            neighbor = slice(2 * (mode + 1), 2 * (mode + 1) + 2)
            hamiltonian[current, neighbor] = coupling
            hamiltonian[neighbor, current] = coupling
    return hamiltonian


def calibration_observables(
    solution: ReferenceSolution,
) -> dict[str, float]:
    populations = solution.modal_occupation
    center = solution.ladder_half_width
    outside_center = 1.0 - populations[:, center]
    outer_total = populations[:, 0] + populations[:, -1]
    outer_individual = populations[:, (0, -1)]
    return {
        "max_total_occupation_outside_center": float(np.max(outside_center)),
        "time_of_max_total_occupation_outside_center": float(
            solution.times[int(np.argmax(outside_center))]
        ),
        "max_outer_layer_total_occupation": float(np.max(outer_total)),
        "max_outer_single_mode_occupation": float(np.max(outer_individual)),
        "max_left_outer_mode_occupation": float(np.max(populations[:, 0])),
        "max_right_outer_mode_occupation": float(np.max(populations[:, -1])),
    }


def reference_solution_metadata(solution: ReferenceSolution) -> dict[str, Any]:
    return {
        "mass": solution.mass,
        "potential_strength": solution.potential_strength,
        "coupling_wave_number": solution.coupling_wave_number,
        "ladder_half_width": solution.ladder_half_width,
        "mode_count": int(solution.wave_numbers.size),
        "time_start": float(solution.times[0]),
        "time_end": float(solution.times[-1]),
        "time_points": int(solution.times.size),
        "wave_numbers": solution.wave_numbers.tolist(),
    }
