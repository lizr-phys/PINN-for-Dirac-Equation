from __future__ import annotations

import hashlib
import math

import torch
from torch import nn

from .wavepacket_config import WAVEPACKET_MODELS, WavepacketConfig
from .wavepacket_physics import (
    compute_spectral_basis,
    gaussian_spectral_envelope,
    wavepacket_initial_solution,
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
    return torch.stack(
        (
            torch.complex(values[:, 0], values[:, 1]),
            torch.complex(values[:, 2], values[:, 3]),
        ),
        dim=1,
    )


class PairedTanhMLP(nn.Module):
    """A float64 MLP initialized from an explicit model-independent stream."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int,
        hidden_layers: int,
        *,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        linears: list[nn.Linear] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            linear = nn.Linear(current_dim, width, dtype=torch.float64)
            linears.append(linear)
            layers.extend((linear, nn.Tanh()))
            current_dim = width
        output = nn.Linear(current_dim, output_dim, dtype=torch.float64)
        linears.append(output)
        layers.append(output)
        self.net = nn.Sequential(*layers)

        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        with torch.no_grad():
            for linear in linears:
                fan_in, fan_out = linear.in_features, linear.out_features
                scale = math.sqrt(2.0 / (fan_in + fan_out))
                sampled = torch.randn(
                    linear.weight.shape,
                    generator=generator,
                    device="cpu",
                    dtype=torch.float64,
                )
                linear.weight.copy_(scale * sampled)
                linear.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class PhysicalWavepacketModel(nn.Module):
    model_name = "physical_base"
    output_dim = 0

    def __init__(
        self,
        cfg: WavepacketConfig,
        *,
        output_dim: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.output_dim = output_dim
        self.initialization_seed = int(initialization_seed)
        self.mlp = PairedTanhMLP(
            3,
            output_dim,
            cfg.hidden_width,
            cfg.hidden_layers,
            initialization_seed=initialization_seed,
        )

    def features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        positions, time = _paired_columns(x, t)
        periodic_angle = (
            2.0 * math.pi * (positions - self.cfg.x_min) / self.cfg.length
        )
        return torch.cat(
            (torch.sin(periodic_angle), torch.cos(periodic_angle), time / self.cfg.t_max),
            dim=1,
        )

    def initial(self, x: torch.Tensor) -> torch.Tensor:
        return wavepacket_initial_solution(x, self.cfg)


class PhysicalVanilla(PhysicalWavepacketModel):
    model_name = "physical_vanilla"

    def __init__(self, cfg: WavepacketConfig, initialization_seed: int) -> None:
        super().__init__(
            cfg, output_dim=4, initialization_seed=initialization_seed
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        positions, time = _paired_columns(x, t)
        correction = _complex_spinor(self.mlp(self.features(positions, time)))
        return self.initial(positions) + (time / self.cfg.t_max) * correction


class PhysicalSinglePhaseCPA(PhysicalWavepacketModel):
    model_name = "physical_sp_cpa"

    def __init__(self, cfg: WavepacketConfig, initialization_seed: int) -> None:
        super().__init__(
            cfg, output_dim=5, initialization_seed=initialization_seed
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        positions, time = _paired_columns(x, t)
        output = self.mlp(self.features(positions, time))
        amplitude = self.initial(positions) + (time / self.cfg.t_max) * _complex_spinor(
            output[:, :4]
        )
        common_phase = (time / self.cfg.t_max) * output[:, 4:5]
        return amplitude * torch.exp(1j * common_phase)


class PhysicalComponentPolar(PhysicalWavepacketModel):
    model_name = "physical_component_polar"

    def __init__(self, cfg: WavepacketConfig, initialization_seed: int) -> None:
        super().__init__(
            cfg, output_dim=4, initialization_seed=initialization_seed
        )

    def component_fields(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions, time = _paired_columns(x, t)
        output = self.mlp(self.features(positions, time))
        initial = self.initial(positions)
        initial_amplitude = torch.abs(initial)
        initial_unit_phase = torch.where(
            initial_amplitude > 1.0e-14,
            initial / torch.clamp_min(initial_amplitude, 1.0e-14),
            torch.ones_like(initial),
        )
        tau = time / self.cfg.t_max
        amplitudes = initial_amplitude + tau * output[:, :2]
        phases = tau * output[:, 2:]
        return amplitudes, phases, initial_unit_phase

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        amplitudes, phases, initial_unit_phase = self.component_fields(x, t)
        return (
            torch.complex(amplitudes, torch.zeros_like(amplitudes))
            * initial_unit_phase
            * torch.exp(1j * phases)
        )


class SharedSpectralModel(nn.Module):
    """Base class for a single shared network evaluated at arbitrary k."""

    model_name = "shared_spectral_base"
    output_dim = 4

    def __init__(self, cfg: WavepacketConfig, initialization_seed: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.initialization_seed = int(initialization_seed)
        self.mlp = PairedTanhMLP(
            2,
            self.output_dim,
            cfg.hidden_width,
            cfg.hidden_layers,
            initialization_seed=initialization_seed,
        )

    def features(
        self,
        wave_numbers: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        k, time = _paired_columns(wave_numbers, t)
        span = self.cfg.wave_number_max - self.cfg.wave_number_min
        if span <= 0.0:
            raise ValueError("shared spectral input requires a nonzero k interval")
        normalized_k = 2.0 * (k - self.cfg.wave_number_min) / span - 1.0
        return torch.cat((normalized_k, time / self.cfg.t_max), dim=1)

    def initial_branch_coefficients(self, wave_numbers: torch.Tensor) -> torch.Tensor:
        k = wave_numbers.reshape(-1)
        amplitude = gaussian_spectral_envelope(k, self.cfg) / math.sqrt(2.0)
        return amplitude[:, None].expand(-1, 2)

    def branch_coefficients(
        self,
        wave_numbers: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, wave_numbers: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        k, time = _paired_columns(wave_numbers, t)
        basis = compute_spectral_basis(k.reshape(-1), self.cfg.mass)
        coefficients = self.branch_coefficients(k, time)
        return (
            coefficients[:, 0:1] * basis.u_plus.to(torch.complex128)
            + coefficients[:, 1:2] * basis.u_minus.to(torch.complex128)
        )


class SharedSpectralComplex(SharedSpectralModel):
    model_name = "shared_spectral_complex"

    def branch_coefficients(
        self,
        wave_numbers: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        k, time = _paired_columns(wave_numbers, t)
        output = self.mlp(self.features(k, time))
        correction = torch.complex(output[:, :2], output[:, 2:])
        return self.initial_branch_coefficients(k).to(torch.complex128) + (
            time / self.cfg.t_max
        ) * correction


class SharedSpectralLTPB(SharedSpectralModel):
    model_name = "shared_spectral_ltpb"

    def branch_fields(
        self,
        wave_numbers: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k, time = _paired_columns(wave_numbers, t)
        output = self.mlp(self.features(k, time))
        tau = time / self.cfg.t_max
        amplitudes = self.initial_branch_coefficients(k) + tau * output[:, :2]
        phases = tau * output[:, 2:]
        return amplitudes, phases

    def branch_coefficients(
        self,
        wave_numbers: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        amplitudes, phases = self.branch_fields(wave_numbers, t)
        return torch.complex(amplitudes, torch.zeros_like(amplitudes)) * torch.exp(
            1j * phases
        )


class SpectralOracle(nn.Module):
    """Exact spectral reference; excluded from every learned comparison."""

    model_name = "spectral_oracle"

    def __init__(self, cfg: WavepacketConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("device_anchor", torch.zeros((), dtype=torch.float64))

    def forward(self, wave_numbers: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        from .wavepacket_physics import exact_spectral_state

        return exact_spectral_state(wave_numbers, t, self.cfg)


def build_wavepacket_model(
    model_name: str,
    cfg: WavepacketConfig,
    initialization_seed: int,
) -> nn.Module:
    constructors: dict[str, type[nn.Module]] = {
        "physical_vanilla": PhysicalVanilla,
        "physical_sp_cpa": PhysicalSinglePhaseCPA,
        "physical_component_polar": PhysicalComponentPolar,
        "shared_spectral_complex": SharedSpectralComplex,
        "shared_spectral_ltpb": SharedSpectralLTPB,
    }
    if model_name not in WAVEPACKET_MODELS:
        raise ValueError(f"Unknown wavepacket model: {model_name}")
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
        if parameter.requires_grad:
            array = parameter.detach().cpu().contiguous().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()
