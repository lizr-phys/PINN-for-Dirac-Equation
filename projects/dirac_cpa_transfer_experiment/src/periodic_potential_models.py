from __future__ import annotations

"""Matched-capacity models for the coupled momentum-ladder experiment.

This module contains only the initial state, the known free-symbol spinor
basis, and trainable maps.  Potential-induced channel coupling belongs to the
PDE loss, not to any model forward.
"""

import hashlib
import math
from typing import Any

import torch
from torch import nn

from .wavepacket_models import PairedTanhMLP
from .wavepacket_physics import compute_spectral_basis


PERIODIC_POTENTIAL_MODELS = (
    "physical_vanilla",
    "coupled_shared_spectral_complex",
    "coupled_shared_spectral_ltpb",
)


def _paired_columns(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_column = first.reshape(-1, 1)
    second_column = second.reshape(-1, 1)
    if first_column.shape[0] == 1 and second_column.shape[0] != 1:
        first_column = first_column.expand_as(second_column)
    elif second_column.shape[0] == 1 and first_column.shape[0] != 1:
        second_column = second_column.expand_as(first_column)
    elif first_column.shape[0] != second_column.shape[0]:
        raise ValueError("paired inputs must have equal sizes or be scalar")
    return first_column, second_column


def _complex_spinor(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Cartesian spinor output must have four real columns")
    return torch.stack(
        (
            torch.complex(values[:, 0], values[:, 1]),
            torch.complex(values[:, 2], values[:, 3]),
        ),
        dim=1,
    )


def _length(cfg: Any) -> float:
    if hasattr(cfg, "length"):
        return float(cfg.length)
    return float(cfg.x_max) - float(cfg.x_min)


def _coupling_wave_number(cfg: Any) -> float:
    if hasattr(cfg, "coupling_wave_number"):
        return float(cfg.coupling_wave_number)
    return float(cfg.q)


def _ladder_limits(cfg: Any) -> tuple[float, float]:
    if hasattr(cfg, "wave_number_min") and hasattr(cfg, "wave_number_max"):
        return float(cfg.wave_number_min), float(cfg.wave_number_max)
    truncation = int(
        getattr(cfg, "truncation_j", getattr(cfg, "ladder_half_width", 0))
    )
    if truncation <= 0:
        raise ValueError("configuration must expose a nonzero ladder half-width")
    center = float(cfg.k0)
    spacing = _coupling_wave_number(cfg)
    endpoints = (center - truncation * spacing, center + truncation * spacing)
    return min(endpoints), max(endpoints)


def central_initial_spinor(
    cfg: Any,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the normalized two-component spinor occupying the center mode."""

    center = torch.as_tensor([float(cfg.k0)], device=device, dtype=dtype)
    basis = compute_spectral_basis(center, float(cfg.mass))
    return (
        (basis.u_plus[0] + basis.u_minus[0]).to(torch.complex128)
        / math.sqrt(2.0)
    )


class PhysicalPotentialVanilla(nn.Module):
    """Physical-space Cartesian spinor with a multiplicative hard time gate."""

    model_name = "physical_vanilla"
    output_dim = 4

    def __init__(self, cfg: Any, initialization_seed: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.initialization_seed = int(initialization_seed)
        self.mlp = PairedTanhMLP(
            3,
            self.output_dim,
            int(cfg.hidden_width),
            int(cfg.hidden_layers),
            initialization_seed=self.initialization_seed,
        )

    def features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        positions, time = _paired_columns(x, t)
        angle = 2.0 * math.pi * (positions - float(self.cfg.x_min)) / _length(
            self.cfg
        )
        return torch.cat(
            (torch.sin(angle), torch.cos(angle), time / float(self.cfg.t_max)),
            dim=1,
        )

    def initial(self, x: torch.Tensor) -> torch.Tensor:
        positions = x.reshape(-1)
        spinor = central_initial_spinor(
            self.cfg, device=positions.device, dtype=positions.dtype
        )
        spatial = torch.exp(1j * float(self.cfg.k0) * positions) / math.sqrt(
            _length(self.cfg)
        )
        return spatial[:, None] * spinor[None, :]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        positions, time = _paired_columns(x, t)
        correction = _complex_spinor(self.mlp(self.features(positions, time)))
        return self.initial(positions) + (time / float(self.cfg.t_max)) * correction


class CoupledSharedSpectralModel(nn.Module):
    """Base class for one four-output MLP shared by every ladder channel."""

    model_name = "coupled_shared_spectral_base"
    output_dim = 4

    def __init__(self, cfg: Any, initialization_seed: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.initialization_seed = int(initialization_seed)
        self.mlp = PairedTanhMLP(
            2,
            self.output_dim,
            int(cfg.hidden_width),
            int(cfg.hidden_layers),
            initialization_seed=self.initialization_seed,
        )

    def features(
        self, wave_numbers: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        k, time = _paired_columns(wave_numbers, t)
        lower, upper = _ladder_limits(self.cfg)
        if not upper > lower:
            raise ValueError("momentum ladder must span more than one channel")
        normalized_k = 2.0 * (k - lower) / (upper - lower) - 1.0
        return torch.cat((normalized_k, time / float(self.cfg.t_max)), dim=1)

    def initial_branch_coefficients(self, wave_numbers: torch.Tensor) -> torch.Tensor:
        k = wave_numbers.reshape(-1)
        spacing = abs(_coupling_wave_number(self.cfg))
        tolerance = max(1.0, spacing) * 1.0e-12
        occupied = torch.isclose(
            k,
            torch.full_like(k, float(self.cfg.k0)),
            rtol=0.0,
            atol=tolerance,
        )
        amplitude = occupied.to(dtype=k.dtype) / math.sqrt(2.0)
        return amplitude[:, None].expand(-1, 2)

    def branch_coefficients(
        self, wave_numbers: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, wave_numbers: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        k, time = _paired_columns(wave_numbers, t)
        basis = compute_spectral_basis(k.reshape(-1), float(self.cfg.mass))
        coefficients = self.branch_coefficients(k, time)
        return (
            coefficients[:, 0:1] * basis.u_plus.to(torch.complex128)
            + coefficients[:, 1:2] * basis.u_minus.to(torch.complex128)
        )


class CoupledSharedSpectralComplex(CoupledSharedSpectralModel):
    model_name = "coupled_shared_spectral_complex"

    def branch_coefficients(
        self, wave_numbers: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        k, time = _paired_columns(wave_numbers, t)
        output = self.mlp(self.features(k, time))
        correction = torch.complex(output[:, :2], output[:, 2:])
        return self.initial_branch_coefficients(k).to(torch.complex128) + (
            time / float(self.cfg.t_max)
        ) * correction


class CoupledSharedSpectralLTPB(CoupledSharedSpectralModel):
    model_name = "coupled_shared_spectral_ltpb"

    def branch_fields(
        self, wave_numbers: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k, time = _paired_columns(wave_numbers, t)
        output = self.mlp(self.features(k, time))
        tau = time / float(self.cfg.t_max)
        amplitudes = self.initial_branch_coefficients(k) + tau * output[:, :2]
        phases = tau * output[:, 2:]
        return amplitudes, phases

    def branch_coefficients(
        self, wave_numbers: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        amplitudes, phases = self.branch_fields(wave_numbers, t)
        return torch.complex(amplitudes, torch.zeros_like(amplitudes)) * torch.exp(
            1j * phases
        )


def build_periodic_potential_model(
    model_name: str,
    cfg: Any,
    initialization_seed: int,
) -> nn.Module:
    constructors: dict[str, type[nn.Module]] = {
        "physical_vanilla": PhysicalPotentialVanilla,
        "coupled_shared_spectral_complex": CoupledSharedSpectralComplex,
        "coupled_shared_spectral_ltpb": CoupledSharedSpectralLTPB,
    }
    if model_name not in PERIODIC_POTENTIAL_MODELS:
        raise ValueError(f"Unknown periodic-potential model: {model_name}")
    return constructors[model_name](cfg, initialization_seed)  # type: ignore[call-arg]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def trainable_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


__all__ = [
    "PERIODIC_POTENTIAL_MODELS",
    "PhysicalPotentialVanilla",
    "CoupledSharedSpectralModel",
    "CoupledSharedSpectralComplex",
    "CoupledSharedSpectralLTPB",
    "central_initial_spinor",
    "build_periodic_potential_model",
    "count_trainable_parameters",
    "trainable_state_sha256",
]
