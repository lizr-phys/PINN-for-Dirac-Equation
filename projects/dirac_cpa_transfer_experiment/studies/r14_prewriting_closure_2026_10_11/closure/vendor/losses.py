from __future__ import annotations

import math

import torch

from .config import ExperimentConfig
from .models import DiracModel
from .physics import (
    analytic_total_energy,
    continuity_from_derivatives,
    density_current,
    dirac_residual,
    energy_density,
    spinor_derivatives,
    spinor_spatial_derivative,
)


class LossComputer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        model: DiracModel,
        *,
        conservative: bool,
        lambda_struct: float,
        device: torch.device,
        collocation_generator: torch.Generator,
        structure_generator: torch.Generator,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.conservative = conservative
        self.lambda_struct = lambda_struct
        self.device = device
        self.collocation_generator = collocation_generator
        self.structure_generator = structure_generator
        # PDE-only ablations must not even consult the analytic spectral energy
        # during training.  The quantity is needed only by the legacy
        # conservative objective.
        self.energy_initial = (
            analytic_total_energy(cfg.energy, model.eta) if conservative else None
        )
        self.x_quadrature = (
            cfg.x_min
            + cfg.length
            * torch.arange(cfg.struct_x_points, device=device, dtype=torch.float64).view(-1, 1)
            / cfg.struct_x_points
        )

    def _sample_collocation(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.cfg.x_min + self.cfg.length * torch.rand(
            (self.cfg.collocation_points, 1),
            generator=self.collocation_generator,
            device=self.device,
            dtype=torch.float64,
        )
        t = self.cfg.t_max * torch.rand(
            (self.cfg.collocation_points, 1),
            generator=self.collocation_generator,
            device=self.device,
            dtype=torch.float64,
        )
        return x.requires_grad_(True), t.requires_grad_(True)

    def _global_structure_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.energy_initial is None:
            raise RuntimeError("Global structure losses require conservative=True")
        times = self.cfg.t_max * torch.rand(
            (self.cfg.struct_time_slices, 1),
            generator=self.structure_generator,
            device=self.device,
            dtype=torch.float64,
        )
        x = self.x_quadrature.repeat(self.cfg.struct_time_slices, 1).clone()
        t = times.repeat_interleave(self.cfg.struct_x_points, dim=0)
        x.requires_grad_(True)
        t.requires_grad_(True)
        psi = self.model(x, t)
        psi_x = spinor_spatial_derivative(psi, x, create_graph=True)
        rho, _ = density_current(psi)
        e_density = energy_density(psi, psi_x, mass=self.cfg.mass)
        rho = rho.reshape(self.cfg.struct_time_slices, self.cfg.struct_x_points)
        e_density = e_density.reshape(
            self.cfg.struct_time_slices, self.cfg.struct_x_points
        )
        norms = self.cfg.length * rho.mean(dim=1)
        energies = self.cfg.length * e_density.mean(dim=1)
        norm_loss = torch.mean((norms - 1.0) ** 2)
        energy_loss = torch.mean(
            ((energies - self.energy_initial) / self.cfg.energy) ** 2
        )
        return norm_loss, energy_loss

    def __call__(self) -> dict[str, torch.Tensor]:
        x, t = self._sample_collocation()
        psi = self.model(x, t)
        psi_x, psi_t = spinor_derivatives(psi, x, t, create_graph=True)
        residual = dirac_residual(psi, psi_x, psi_t, mass=self.cfg.mass)
        pde = torch.mean(torch.abs(residual) ** 2)
        zero = torch.zeros((), dtype=torch.float64, device=self.device)
        if not self.conservative:
            return {
                "total": pde,
                "pde": pde,
                "continuity": zero,
                "norm": zero,
                "energy": zero,
                "structure": zero,
            }
        continuity = continuity_from_derivatives(psi, psi_x, psi_t)
        continuity_scale = self.cfg.energy * self.cfg.reference_density
        continuity_loss = torch.mean((continuity / continuity_scale) ** 2)
        norm_loss, energy_loss = self._global_structure_losses()
        structure = continuity_loss + norm_loss + energy_loss
        total = pde + self.lambda_struct * structure
        return {
            "total": total,
            "pde": pde,
            "continuity": continuity_loss,
            "norm": norm_loss,
            "energy": energy_loss,
            "structure": structure,
        }


def gradient_cosine(
    pde_loss: torch.Tensor, structure_loss: torch.Tensor, model: torch.nn.Module
) -> float:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    pde_gradients = torch.autograd.grad(
        pde_loss, parameters, retain_graph=True, allow_unused=True
    )
    structure_gradients = torch.autograd.grad(
        structure_loss, parameters, retain_graph=True, allow_unused=True
    )
    dot = torch.zeros((), dtype=torch.float64, device=pde_loss.device)
    pde_norm_sq = torch.zeros_like(dot)
    structure_norm_sq = torch.zeros_like(dot)
    for pde_gradient, structure_gradient in zip(pde_gradients, structure_gradients):
        if pde_gradient is not None:
            pde_norm_sq = pde_norm_sq + torch.sum(pde_gradient**2)
        if structure_gradient is not None:
            structure_norm_sq = structure_norm_sq + torch.sum(structure_gradient**2)
        if pde_gradient is not None and structure_gradient is not None:
            dot = dot + torch.sum(pde_gradient * structure_gradient)
    denominator = torch.sqrt(pde_norm_sq * structure_norm_sq)
    if float(denominator.detach().cpu()) == 0.0:
        return math.nan
    return float((dot / denominator).detach().cpu())
