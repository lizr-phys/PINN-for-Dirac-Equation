from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .physics import compute_eigensystem


@dataclass(frozen=True)
class MultimodeBasis:
    """Known spatial momenta and deterministic real Dirac eigenspinors.

    Spectral energies are deliberately not stored in this object so a learned
    model cannot obtain an analytic time carrier through its physical basis.
    """

    wave_numbers: torch.Tensor  # (2,)
    u_plus: torch.Tensor  # (2 momenta, 2 spinor components)
    u_minus: torch.Tensor  # (2 momenta, 2 spinor components)


def compute_multimode_basis(
    mass: float,
    wave_numbers: tuple[float, float] = (2.0, 5.0),
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> MultimodeBasis:
    systems = [
        compute_eigensystem(mass, wave_number, device=device, dtype=dtype)
        for wave_number in wave_numbers
    ]
    return MultimodeBasis(
        wave_numbers=torch.tensor(wave_numbers, device=device, dtype=dtype),
        u_plus=torch.stack([system.u_plus for system in systems], dim=0),
        u_minus=torch.stack([system.u_minus for system in systems], dim=0),
    )


def reference_energies(
    mass: float, wave_numbers: tuple[float, float] = (2.0, 5.0)
) -> np.ndarray:
    """Analytic energies for reference generation and post-training evaluation."""

    return np.sqrt(np.asarray(wave_numbers, dtype=np.float64) ** 2 + mass**2)


def multimode_exact_solution(
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    mass: float = 5.0,
    wave_numbers: tuple[float, float] = (2.0, 5.0),
    length: float = 2.0 * math.pi,
) -> torch.Tensor:
    basis = compute_multimode_basis(
        mass, wave_numbers, device=x.device, dtype=x.dtype
    )
    energies = torch.sqrt(basis.wave_numbers**2 + mass**2)
    coefficient = 1.0 / (2.0 * math.sqrt(length))
    result = torch.zeros((x.shape[0], 2), device=x.device, dtype=torch.complex128)
    for index in range(2):
        spatial = torch.exp(1j * basis.wave_numbers[index] * x)
        positive = (
            basis.u_plus[index].to(torch.complex128).view(1, 2)
            * torch.exp(-1j * energies[index] * t)
        )
        negative = (
            basis.u_minus[index].to(torch.complex128).view(1, 2)
            * torch.exp(+1j * energies[index] * t)
        )
        result = result + coefficient * spatial * (positive + negative)
    return result


def multimode_initial_solution(
    x: torch.Tensor,
    *,
    mass: float = 5.0,
    wave_numbers: tuple[float, float] = (2.0, 5.0),
    length: float = 2.0 * math.pi,
) -> torch.Tensor:
    # Build the known initial condition directly from k and eigenspinors.  Do
    # not call the analytic time-dependent solution here: learned forwards use
    # this function and therefore must never compute or receive exact E_l.
    basis = compute_multimode_basis(
        mass, wave_numbers, device=x.device, dtype=x.dtype
    )
    coefficient = 1.0 / (2.0 * math.sqrt(length))
    result = torch.zeros((x.shape[0], 2), device=x.device, dtype=torch.complex128)
    for index in range(2):
        spatial = torch.exp(1j * basis.wave_numbers[index] * x)
        result = result + coefficient * spatial * (
            basis.u_plus[index].to(torch.complex128).view(1, 2)
            + basis.u_minus[index].to(torch.complex128).view(1, 2)
        )
    return result


def modal_basis_functions(
    x: torch.Tensor, basis: MultimodeBasis
) -> torch.Tensor:
    """Return ordered basis functions with shape ``(n, 4, 2)``.

    Order: ``k1+``, ``k1-``, ``k2+``, ``k2-``.
    """

    functions: list[torch.Tensor] = []
    for index in range(2):
        spatial = torch.exp(1j * basis.wave_numbers[index] * x)
        functions.extend(
            (
                spatial * basis.u_plus[index].to(torch.complex128).view(1, 2),
                spatial * basis.u_minus[index].to(torch.complex128).view(1, 2),
            )
        )
    return torch.stack(functions, dim=1)


def project_modal_coefficients(
    psi: torch.Tensor, x: torch.Tensor, basis: MultimodeBasis
) -> torch.Tensor:
    """Project a periodic grid field onto the four normalized modal channels.

    ``psi`` is ``(..., n_x, 2)`` and ``x`` contains the endpoint-free uniform
    spatial grid.  The discrete mean equals ``L^-1 integral`` for these integer
    Fourier modes.
    """

    functions = modal_basis_functions(x.reshape(-1, 1), basis)
    return torch.mean(
        torch.sum(torch.conj(functions) * psi.unsqueeze(-2), dim=-1), dim=-2
    )
