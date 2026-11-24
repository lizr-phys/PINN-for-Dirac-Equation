from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .wavepacket_config import WavepacketConfig


@dataclass(frozen=True)
class SpectralBasis:
    """Wave numbers and a deterministic, continuous real eigenspinor gauge.

    Only eigenvectors are retained.  In particular, this object exposes no
    analytic time rate that a learned forward could reuse as a carrier.
    """

    wave_numbers: torch.Tensor  # (n_k,)
    u_plus: torch.Tensor  # (n_k, 2)
    u_minus: torch.Tensor  # (n_k, 2)


def _flat_real(values: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if values.is_complex():
        raise TypeError(f"{name} must be real")
    return values.reshape(-1)


def _paired_flat(
    first: torch.Tensor,
    second: torch.Tensor,
    first_name: str,
    second_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_flat = _flat_real(first, first_name)
    second_flat = _flat_real(second, second_name)
    if first_flat.numel() == 1 and second_flat.numel() != 1:
        first_flat = first_flat.expand_as(second_flat)
    elif second_flat.numel() == 1 and first_flat.numel() != 1:
        second_flat = second_flat.expand_as(first_flat)
    elif first_flat.numel() != second_flat.numel():
        raise ValueError(
            f"{first_name} and {second_name} must have equal sizes or be scalar"
        )
    return first_flat, second_flat


def training_wave_numbers(
    cfg: WavepacketConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    return torch.as_tensor(cfg.wave_numbers, device=device, dtype=dtype)


def compute_spectral_basis(
    wave_numbers: torch.Tensor,
    mass: float,
) -> SpectralBasis:
    """Return a continuous real gauge without constructing eigenvalues.

    For ``theta=atan2(k,m)`` the columns
    ``u_plus=(cos(theta/2),sin(theta/2))`` and
    ``u_minus=(-sin(theta/2),cos(theta/2))`` diagonalize the known Dirac
    symbol.  This analytic gauge is deterministic and continuous on the
    preregistered interval k>=0, including k=0, while exposing no dispersion
    values to learned code.
    """

    k = _flat_real(wave_numbers, "wave_numbers")
    mass_tensor = torch.full_like(k, float(mass))
    half_angle = 0.5 * torch.atan2(k, mass_tensor)
    cosine = torch.cos(half_angle)
    sine = torch.sin(half_angle)
    u_plus = torch.stack((cosine, sine), dim=1)
    u_minus = torch.stack((-sine, cosine), dim=1)
    return SpectralBasis(k, u_plus, u_minus)


def gaussian_normalization(
    cfg: WavepacketConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    grid = training_wave_numbers(cfg, device=device, dtype=dtype)
    raw = torch.exp(
        -((grid - cfg.k0) ** 2)
        / (4.0 * cfg.sigma_k**2)
    )
    return torch.rsqrt(torch.sum(raw**2))


def gaussian_spectral_envelope(
    wave_numbers: torch.Tensor,
    cfg: WavepacketConfig,
) -> torch.Tensor:
    """Evaluate the known initial Gaussian using the training-grid constant.

    The same constant is used at held-out midpoints; the midpoint set is not
    renormalized, because those points are interpolation probes rather than a
    second initial state on a different quadrature grid.
    """

    k = _flat_real(wave_numbers, "wave_numbers")
    constant = gaussian_normalization(cfg, device=k.device, dtype=k.dtype)
    return constant * torch.exp(
        -((k - cfg.k0) ** 2)
        / (4.0 * cfg.sigma_k**2)
    )


def spectral_initial_state(
    wave_numbers: torch.Tensor,
    cfg: WavepacketConfig,
) -> torch.Tensor:
    """Return orthonormal-Fourier coefficients at t=0, shape ``(n_k, 2)``."""

    k = _flat_real(wave_numbers, "wave_numbers")
    basis = compute_spectral_basis(k, cfg.mass)
    coefficient = gaussian_spectral_envelope(k, cfg) / math.sqrt(2.0)
    return coefficient[:, None].to(torch.complex128) * (
        basis.u_plus + basis.u_minus
    ).to(torch.complex128)


def reference_energies(
    wave_numbers: torch.Tensor | np.ndarray,
    mass: float,
) -> torch.Tensor | np.ndarray:
    """Analytic dispersion for reference generation and post-training use."""

    if isinstance(wave_numbers, torch.Tensor):
        return torch.sqrt(wave_numbers**2 + mass**2)
    array = np.asarray(wave_numbers, dtype=np.float64)
    return np.sqrt(array**2 + mass**2)


def exact_spectral_state(
    wave_numbers: torch.Tensor,
    t: torch.Tensor,
    cfg: WavepacketConfig,
) -> torch.Tensor:
    """Reference spectral state at paired ``(k,t)`` samples."""

    k, time = _paired_flat(wave_numbers, t, "wave_numbers", "t")
    basis = compute_spectral_basis(k, cfg.mass)
    energies = reference_energies(k, cfg.mass)
    assert isinstance(energies, torch.Tensor)
    coefficient = gaussian_spectral_envelope(k, cfg) / math.sqrt(2.0)
    positive = torch.exp(-1j * energies * time)[:, None] * basis.u_plus.to(
        torch.complex128
    )
    negative = torch.exp(+1j * energies * time)[:, None] * basis.u_minus.to(
        torch.complex128
    )
    return coefficient[:, None].to(torch.complex128) * (positive + negative)


def wavepacket_initial_solution(
    x: torch.Tensor,
    cfg: WavepacketConfig,
) -> torch.Tensor:
    """Known physical initial state without constructing analytic time rates."""

    positions = _flat_real(x, "x")
    k = training_wave_numbers(cfg, device=positions.device, dtype=positions.dtype)
    coefficients = spectral_initial_state(k, cfg)
    spatial = torch.exp(1j * positions[:, None] * k[None, :])
    return torch.einsum("xk,kc->xc", spatial, coefficients) / math.sqrt(
        cfg.length
    )


def wavepacket_exact_solution(
    x: torch.Tensor,
    t: torch.Tensor,
    cfg: WavepacketConfig,
) -> torch.Tensor:
    """Reference physical solution at paired ``(x,t)`` samples."""

    positions, time = _paired_flat(x, t, "x", "t")
    k = training_wave_numbers(cfg, device=positions.device, dtype=positions.dtype)
    basis = compute_spectral_basis(k, cfg.mass)
    energies = reference_energies(k, cfg.mass)
    assert isinstance(energies, torch.Tensor)
    envelope = gaussian_spectral_envelope(k, cfg) / math.sqrt(2.0)

    positive = (
        torch.exp(-1j * time[:, None] * energies[None, :])[:, :, None]
        * basis.u_plus.to(torch.complex128)[None, :, :]
    )
    negative = (
        torch.exp(+1j * time[:, None] * energies[None, :])[:, :, None]
        * basis.u_minus.to(torch.complex128)[None, :, :]
    )
    spectral = envelope.to(torch.complex128)[None, :, None] * (
        positive + negative
    )
    spatial = torch.exp(1j * positions[:, None] * k[None, :])
    return torch.einsum("xk,xkc->xc", spatial, spectral) / math.sqrt(
        cfg.length
    )


def reconstruct_physical_grid(
    spectral_state: torch.Tensor,
    wave_numbers: torch.Tensor,
    x: torch.Tensor,
    *,
    length: float,
) -> torch.Tensor:
    """Inverse orthonormal Fourier sum on an arbitrary physical grid.

    ``spectral_state`` may have shape ``(n_k,2)`` or ``(...,n_k,2)``.  The
    returned array has shape ``(n_x,2)`` or ``(...,n_x,2)`` respectively.
    """

    k = _flat_real(wave_numbers, "wave_numbers")
    positions = _flat_real(x, "x")
    if spectral_state.shape[-2:] != (k.numel(), 2):
        raise ValueError(
            "spectral_state must end with (number of wave numbers, 2)"
        )
    spatial = torch.exp(1j * positions[:, None] * k[None, :])
    return torch.einsum("xk,...kc->...xc", spatial, spectral_state) / math.sqrt(
        length
    )


def spatial_fourier_coefficients(
    psi: torch.Tensor,
    x: torch.Tensor,
    wave_numbers: torch.Tensor,
    *,
    length: float,
) -> torch.Tensor:
    """Project an endpoint-free uniform physical grid onto orthonormal modes."""

    positions = _flat_real(x, "x")
    k = _flat_real(wave_numbers, "wave_numbers")
    if psi.shape[-2:] != (positions.numel(), 2):
        raise ValueError("psi must end with (number of x points, 2)")
    spatial_conjugate = torch.exp(-1j * positions[:, None] * k[None, :])
    return math.sqrt(length) * torch.einsum(
        "xk,...xc->...kc", spatial_conjugate, psi
    ) / positions.numel()


def project_energy_branches(
    spectral_state: torch.Tensor,
    wave_numbers: torch.Tensor,
    mass: float,
) -> torch.Tensor:
    """Project total spectral spinors onto physical ``(+,-)`` coefficients."""

    k = _flat_real(wave_numbers, "wave_numbers")
    if spectral_state.shape[-2:] != (k.numel(), 2):
        raise ValueError(
            "spectral_state must end with (number of wave numbers, 2)"
        )
    basis = compute_spectral_basis(k, mass)
    positive = torch.einsum(
        "kc,...kc->...k", basis.u_plus.to(torch.complex128), spectral_state
    )
    negative = torch.einsum(
        "kc,...kc->...k", basis.u_minus.to(torch.complex128), spectral_state
    )
    return torch.stack((positive, negative), dim=-1)


def spectral_hamiltonian_apply(
    spectral_state: torch.Tensor,
    wave_numbers: torch.Tensor,
    mass: float,
) -> torch.Tensor:
    k = _flat_real(wave_numbers, "wave_numbers")
    if spectral_state.shape != (k.numel(), 2):
        raise ValueError("paired spectral state must have shape (n_samples, 2)")
    sigma_x_state = torch.stack(
        (spectral_state[:, 1], spectral_state[:, 0]), dim=1
    )
    sigma_z_state = torch.stack(
        (spectral_state[:, 0], -spectral_state[:, 1]), dim=1
    )
    return k[:, None] * sigma_x_state + mass * sigma_z_state


def spectral_dirac_residual(
    spectral_state: torch.Tensor,
    spectral_time_derivative: torch.Tensor,
    wave_numbers: torch.Tensor,
    mass: float,
) -> torch.Tensor:
    """Strong residual ``i d_t psi_tilde - (k sigma_x+m sigma_z)psi_tilde``."""

    if spectral_time_derivative.shape != spectral_state.shape:
        raise ValueError("state and time derivative shapes must match")
    return 1j * spectral_time_derivative - spectral_hamiltonian_apply(
        spectral_state, wave_numbers, mass
    )
