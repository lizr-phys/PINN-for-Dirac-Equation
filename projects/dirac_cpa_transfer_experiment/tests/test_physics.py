from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.config import ExperimentConfig
from src.models import build_model
from src.physics import (
    analytic_current,
    analytic_total_energy,
    compute_eigensystem,
    continuity_from_derivatives,
    density_current,
    dirac_residual,
    energy_density,
    exact_solution,
    spinor_derivatives,
)


torch.set_default_dtype(torch.float64)
CFG = ExperimentConfig()


def periodic_grid(n: int) -> torch.Tensor:
    return CFG.x_min + CFG.length * torch.arange(n, dtype=torch.float64).view(-1, 1) / n


def test_eigenspinors_are_orthonormal_and_satisfy_eigenproblem() -> None:
    eig = compute_eigensystem(CFG.mass, CFG.wave_number)
    h_k = torch.tensor(
        [[CFG.mass, CFG.wave_number], [CFG.wave_number, -CFG.mass]],
        dtype=torch.float64,
    )
    gram = torch.stack((eig.u_plus, eig.u_minus), dim=1).T @ torch.stack(
        (eig.u_plus, eig.u_minus), dim=1
    )
    plus_error = torch.linalg.norm(h_k @ eig.u_plus - eig.energy * eig.u_plus)
    minus_error = torch.linalg.norm(h_k @ eig.u_minus + eig.energy * eig.u_minus)
    assert torch.max(torch.abs(gram - torch.eye(2))).item() < 1.0e-12
    assert plus_error.item() < 1.0e-10
    assert minus_error.item() < 1.0e-10


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_exact_solution_autodiff_dirac_residual_is_machine_precision(eta: float) -> None:
    generator = torch.Generator().manual_seed(1234)
    x = (CFG.x_min + CFG.length * torch.rand((96, 1), generator=generator)).requires_grad_(True)
    t = (CFG.t_max * torch.rand((96, 1), generator=generator)).requires_grad_(True)
    psi = exact_solution(
        x, t, mass=CFG.mass, wave_number=CFG.wave_number, eta=eta
    )
    psi_x, psi_t = spinor_derivatives(psi, x, t, create_graph=True)
    residual = dirac_residual(psi, psi_x, psi_t, mass=CFG.mass)
    assert torch.max(torch.abs(residual)).item() < 1.0e-10


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_exact_norm_and_total_energy_are_constant(eta: float) -> None:
    n_x, n_t = 256, 9
    x = periodic_grid(n_x).repeat(n_t, 1).requires_grad_(True)
    times = torch.linspace(0.0, CFG.t_max, n_t).view(-1, 1)
    t = times.repeat_interleave(n_x, dim=0).requires_grad_(True)
    psi = exact_solution(
        x, t, mass=CFG.mass, wave_number=CFG.wave_number, eta=eta
    )
    psi_x, _ = spinor_derivatives(psi, x, t, create_graph=True)
    rho, _ = density_current(psi)
    e_density = energy_density(psi, psi_x, mass=CFG.mass)
    norms = CFG.length * rho.reshape(n_t, n_x).mean(dim=1)
    energies = CFG.length * e_density.reshape(n_t, n_x).mean(dim=1)
    target_energy = analytic_total_energy(CFG.energy, eta)
    assert torch.max(torch.abs(norms - 1.0)).item() < 1.0e-12
    assert torch.max(torch.abs(energies - target_energy)).item() < 1.0e-10


def test_case_m_density_constant_and_current_has_two_e_frequency() -> None:
    eig = compute_eigensystem(CFG.mass, CFG.wave_number)
    t = torch.linspace(0.0, CFG.t_max, 4097).view(-1, 1)
    x = torch.zeros_like(t)
    psi = exact_solution(
        x, t, mass=CFG.mass, wave_number=CFG.wave_number, eta=1.0, eigensystem=eig
    )
    rho, current = density_current(psi)
    current_expected = analytic_current(
        t[:, 0].numpy(),
        mass=CFG.mass,
        wave_number=CFG.wave_number,
        eta=1.0,
        cross_current=eig.cross_current,
    )
    carrier = np.cos(2.0 * CFG.energy * t[:, 0].numpy())
    correlation = np.corrcoef(current.detach().numpy(), carrier)[0, 1]
    assert torch.max(torch.abs(rho - 1.0 / (2.0 * math.pi))).item() < 1.0e-12
    assert np.max(np.abs(current.detach().numpy() - current_expected)) < 1.0e-12
    assert np.ptp(current_expected) > 0.25
    assert correlation > 0.999999999


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_exact_continuity_residual_is_machine_precision(eta: float) -> None:
    generator = torch.Generator().manual_seed(987)
    x = (CFG.x_min + CFG.length * torch.rand((128, 1), generator=generator)).requires_grad_(True)
    t = (CFG.t_max * torch.rand((128, 1), generator=generator)).requires_grad_(True)
    psi = exact_solution(
        x, t, mass=CFG.mass, wave_number=CFG.wave_number, eta=eta
    )
    psi_x, psi_t = spinor_derivatives(psi, x, t, create_graph=True)
    continuity = continuity_from_derivatives(psi, psi_x, psi_t)
    assert torch.max(torch.abs(continuity)).item() < 1.0e-10


@pytest.mark.parametrize("model_name", ["vanilla", "sp_cpa", "sp_cpa_conservative", "two_branch"])
@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_models_hard_code_initial_condition_and_periodicity(
    model_name: str, eta: float
) -> None:
    torch.manual_seed(5)
    model = build_model(model_name, CFG, eta).to(dtype=torch.float64)
    x = periodic_grid(33)
    t0 = torch.zeros_like(x)
    with torch.no_grad():
        predicted = model(x, t0)
        exact = exact_solution(
            x, t0, mass=CFG.mass, wave_number=CFG.wave_number, eta=eta
        )
        t = torch.linspace(0.0, CFG.t_max, 17).view(-1, 1)
        left = model(torch.full_like(t, CFG.x_min), t)
        right = model(torch.full_like(t, CFG.x_max), t)
    assert torch.max(torch.abs(predicted - exact)).item() < 1.0e-12
    assert torch.max(torch.abs(left - right)).item() < 1.0e-10
