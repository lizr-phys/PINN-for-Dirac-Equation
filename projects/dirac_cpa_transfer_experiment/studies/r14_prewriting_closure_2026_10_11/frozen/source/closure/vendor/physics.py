from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Eigensystem:
    energy: float
    u_plus: torch.Tensor
    u_minus: torch.Tensor
    cross_current: float


def compute_eigensystem(
    mass: float,
    wave_number: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Eigensystem:
    """Return a deterministic real gauge for the +/- eigenvectors of H_k."""
    h_k = torch.tensor(
        [[mass, wave_number], [wave_number, -mass]], device=device, dtype=dtype
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(h_k)
    u_minus = eigenvectors[:, 0]
    u_plus = eigenvectors[:, 1]
    if float(u_plus[0]) < 0.0:
        u_plus = -u_plus
    sigma_x_u_minus = torch.stack((u_minus[1], u_minus[0]))
    cross = torch.dot(u_plus, sigma_x_u_minus)
    if float(cross) < 0.0:
        u_minus = -u_minus
        cross = -cross
    energy = float((eigenvalues[1] - eigenvalues[0]) / 2.0)
    return Eigensystem(energy, u_plus, u_minus, float(cross))


def complex_from_four(values: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            torch.complex(values[:, 0], values[:, 1]),
            torch.complex(values[:, 2], values[:, 3]),
        ),
        dim=1,
    )


def exact_solution(
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    mass: float,
    wave_number: float,
    eta: float,
    eigensystem: Eigensystem | None = None,
) -> torch.Tensor:
    eig = eigensystem or compute_eigensystem(
        mass, wave_number, device=x.device, dtype=x.dtype
    )
    u_plus = eig.u_plus.to(device=x.device, dtype=x.dtype).to(torch.complex128)
    u_minus = eig.u_minus.to(device=x.device, dtype=x.dtype).to(torch.complex128)
    norm = math.sqrt(2.0 * math.pi * (1.0 + eta**2))
    spatial = torch.exp(1j * wave_number * x)
    positive = torch.exp(-1j * eig.energy * t) * u_plus.view(1, 2)
    negative = eta * torch.exp(1j * eig.energy * t) * u_minus.view(1, 2)
    return spatial * (positive + negative) / norm


def initial_solution(
    x: torch.Tensor,
    *,
    mass: float,
    wave_number: float,
    eta: float,
    eigensystem: Eigensystem | None = None,
) -> torch.Tensor:
    return exact_solution(
        x,
        torch.zeros_like(x),
        mass=mass,
        wave_number=wave_number,
        eta=eta,
        eigensystem=eigensystem,
    )


def scalar_derivative(
    y: torch.Tensor,
    x: torch.Tensor,
    *,
    create_graph: bool,
    retain_graph: bool = True,
) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=retain_graph,
    )[0]


def complex_derivative(
    z: torch.Tensor, x: torch.Tensor, *, create_graph: bool
) -> torch.Tensor:
    real = scalar_derivative(z.real, x, create_graph=create_graph)
    imag = scalar_derivative(z.imag, x, create_graph=create_graph)
    return torch.complex(real, imag).squeeze(-1)


def spinor_derivatives(
    psi: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    psi_x = torch.stack(
        [complex_derivative(psi[:, index], x, create_graph=create_graph) for index in range(2)],
        dim=1,
    )
    psi_t = torch.stack(
        [complex_derivative(psi[:, index], t, create_graph=create_graph) for index in range(2)],
        dim=1,
    )
    return psi_x, psi_t


def spinor_spatial_derivative(
    psi: torch.Tensor, x: torch.Tensor, *, create_graph: bool
) -> torch.Tensor:
    return torch.stack(
        [complex_derivative(psi[:, index], x, create_graph=create_graph) for index in range(2)],
        dim=1,
    )


def sigma_x_apply(psi: torch.Tensor) -> torch.Tensor:
    return torch.stack((psi[:, 1], psi[:, 0]), dim=1)


def sigma_z_apply(psi: torch.Tensor) -> torch.Tensor:
    return torch.stack((psi[:, 0], -psi[:, 1]), dim=1)


def dirac_residual(
    psi: torch.Tensor,
    psi_x: torch.Tensor,
    psi_t: torch.Tensor,
    *,
    mass: float,
) -> torch.Tensor:
    return 1j * psi_t + 1j * sigma_x_apply(psi_x) - mass * sigma_z_apply(psi)


def density_current(psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rho = torch.sum(torch.abs(psi) ** 2, dim=1)
    current = torch.sum(torch.conj(psi) * sigma_x_apply(psi), dim=1).real
    return rho, current


def continuity_from_derivatives(
    psi: torch.Tensor, psi_x: torch.Tensor, psi_t: torch.Tensor
) -> torch.Tensor:
    rho_t = 2.0 * torch.sum(torch.conj(psi) * psi_t, dim=1).real
    current_x = 2.0 * torch.sum(
        torch.conj(psi_x) * sigma_x_apply(psi), dim=1
    ).real
    return rho_t + current_x


def energy_density(
    psi: torch.Tensor, psi_x: torch.Tensor, *, mass: float
) -> torch.Tensor:
    h_psi = -1j * sigma_x_apply(psi_x) + mass * sigma_z_apply(psi)
    return torch.sum(torch.conj(psi) * h_psi, dim=1).real


def analytic_total_energy(energy: float, eta: float) -> float:
    return energy * (1.0 - eta**2) / (1.0 + eta**2)


def analytic_current(
    t: np.ndarray,
    *,
    mass: float,
    wave_number: float,
    eta: float,
    cross_current: float,
) -> np.ndarray:
    energy = math.sqrt(mass**2 + wave_number**2)
    norm_sq = 1.0 / (2.0 * math.pi * (1.0 + eta**2))
    diagonal = (1.0 - eta**2) * wave_number / energy
    oscillatory = 2.0 * eta * cross_current * np.cos(2.0 * energy * t)
    return norm_sq * (diagonal + oscillatory)
