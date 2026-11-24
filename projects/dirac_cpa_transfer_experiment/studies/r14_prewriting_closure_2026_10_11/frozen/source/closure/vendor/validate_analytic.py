from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, write_json
from .physics import (
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


def main() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = ExperimentConfig()
    eig = compute_eigensystem(cfg.mass, cfg.wave_number)
    h_k = torch.tensor(
        [[cfg.mass, cfg.wave_number], [cfg.wave_number, -cfg.mass]],
        dtype=torch.float64,
    )
    eigen_error = max(
        torch.linalg.norm(h_k @ eig.u_plus - eig.energy * eig.u_plus).item(),
        torch.linalg.norm(h_k @ eig.u_minus + eig.energy * eig.u_minus).item(),
    )
    orthogonality_error = torch.max(
        torch.abs(
            torch.stack((eig.u_plus, eig.u_minus), dim=1).T
            @ torch.stack((eig.u_plus, eig.u_minus), dim=1)
            - torch.eye(2)
        )
    ).item()

    n_x, n_t = 64, 101
    x_values = cfg.x_min + cfg.length * torch.arange(n_x).view(-1, 1) / n_x
    t_values = torch.linspace(0.0, cfg.t_max, n_t).view(-1, 1)
    x = x_values.repeat(n_t, 1).requires_grad_(True)
    t = t_values.repeat_interleave(n_x, dim=0).requires_grad_(True)
    exact = exact_solution(
        x,
        t,
        mass=cfg.mass,
        wave_number=cfg.wave_number,
        eta=1.0,
        eigensystem=eig,
    )
    exact_x, exact_t = spinor_derivatives(exact, x, t, create_graph=True)
    residual = dirac_residual(exact, exact_x, exact_t, mass=cfg.mass)
    continuity = continuity_from_derivatives(exact, exact_x, exact_t)
    rho, current = density_current(exact)
    e_density = energy_density(exact, exact_x, mass=cfg.mass)
    norms = cfg.length * rho.reshape(n_t, n_x).mean(dim=1)
    energies = cfg.length * e_density.reshape(n_t, n_x).mean(dim=1)

    delta = 0.7 * torch.sin(0.9 * t)
    norm = math.sqrt(4.0 * math.pi)
    wrong = torch.exp(1j * cfg.wave_number * x) * (
        eig.u_plus.to(torch.complex128).view(1, 2) * torch.exp(-1j * eig.energy * t)
        + eig.u_minus.to(torch.complex128).view(1, 2)
        * torch.exp(1j * eig.energy * t + 1j * delta)
    ) / norm
    wrong_x, wrong_t = spinor_derivatives(wrong, x, t, create_graph=True)
    wrong_residual = dirac_residual(wrong, wrong_x, wrong_t, mass=cfg.mass)
    wrong_continuity = continuity_from_derivatives(wrong, wrong_x, wrong_t)
    wrong_rho, wrong_current = density_current(wrong)
    wrong_energy_density = energy_density(wrong, wrong_x, mass=cfg.mass)
    wrong_norms = cfg.length * wrong_rho.reshape(n_t, n_x).mean(dim=1)
    wrong_energies = cfg.length * wrong_energy_density.reshape(n_t, n_x).mean(dim=1)

    current_t = current.reshape(n_t, n_x)[:, 0].detach().numpy()
    expected_current = analytic_current(
        t_values[:, 0].numpy(),
        mass=cfg.mass,
        wave_number=cfg.wave_number,
        eta=1.0,
        cross_current=eig.cross_current,
    )
    payload = {
        "mass": cfg.mass,
        "wave_number": cfg.wave_number,
        "T": cfg.t_max,
        "energy": cfg.energy,
        "two_E_angular_frequency": 2.0 * cfg.energy,
        "oscillation_period": math.pi / cfg.energy,
        "u_plus": eig.u_plus.tolist(),
        "u_minus": eig.u_minus.tolist(),
        "cross_current": eig.cross_current,
        "orthogonality_error": orthogonality_error,
        "eigen_equation_max_error": eigen_error,
        "exact_pde_residual_max_abs": torch.max(torch.abs(residual)).item(),
        "exact_continuity_max_abs": torch.max(torch.abs(continuity)).item(),
        "exact_norm_max_drift": torch.max(torch.abs(norms - 1.0)).item(),
        "exact_energy_max_drift": torch.max(
            torch.abs(energies - analytic_total_energy(cfg.energy, 1.0))
        ).item(),
        "case_m_density_max_deviation": torch.max(
            torch.abs(rho - cfg.reference_density)
        ).item(),
        "case_m_current_peak_to_peak": float(np.ptp(current_t)),
        "case_m_current_analytic_max_error": float(
            np.max(np.abs(current_t - expected_current))
        ),
        "wrong_phase_demo": {
            "delta_definition": "0.7 sin(0.9 t) on the negative-energy branch",
            "continuity_max_abs": torch.max(torch.abs(wrong_continuity)).item(),
            "norm_max_drift": torch.max(torch.abs(wrong_norms - 1.0)).item(),
            "energy_max_abs_drift": torch.max(torch.abs(wrong_energies)).item(),
            "pde_residual_rms": torch.sqrt(torch.mean(torch.abs(wrong_residual) ** 2)).item(),
            "current_relative_l2": (
                torch.linalg.norm(wrong_current - current) / torch.linalg.norm(current)
            ).item(),
            "spinor_relative_l2": (
                torch.linalg.norm(wrong - exact) / torch.linalg.norm(exact)
            ).item(),
        },
    }
    out = Path(__file__).resolve().parents[1] / "results" / "raw" / "analytic_validation.json"
    write_json(out, payload)
    print(out)
    for key, value in payload.items():
        if not isinstance(value, (dict, list)):
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
