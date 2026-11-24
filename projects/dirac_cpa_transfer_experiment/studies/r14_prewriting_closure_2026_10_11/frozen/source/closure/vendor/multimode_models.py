from __future__ import annotations

import hashlib
import math

import torch
from torch import nn

from .multimode_config import MultimodeConfig
from .multimode_physics import (
    MultimodeBasis,
    compute_multimode_basis,
    multimode_exact_solution,
    multimode_initial_solution,
)


def _complex_spinor(values: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            torch.complex(values[:, 0], values[:, 1]),
            torch.complex(values[:, 2], values[:, 3]),
        ),
        dim=1,
    )


class PairedTanhMLP(nn.Module):
    """Tanh MLP initialized from a model-independent local random stream."""

    def __init__(
        self,
        output_dim: int,
        width: int,
        hidden_layers: int,
        *,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = 3
        linears: list[nn.Linear] = []
        for _ in range(hidden_layers):
            linear = nn.Linear(input_dim, width)
            linears.append(linear)
            layers.extend((linear, nn.Tanh()))
            input_dim = width
        output = nn.Linear(width, output_dim)
        linears.append(output)
        layers.append(output)
        self.net = nn.Sequential(*layers)

        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        with torch.no_grad():
            for linear in linears:
                fan_in, fan_out = linear.in_features, linear.out_features
                scale = math.sqrt(2.0 / (fan_in + fan_out))
                values = torch.randn(
                    linear.weight.shape,
                    generator=generator,
                    dtype=linear.weight.dtype,
                )
                linear.weight.copy_(scale * values)
                linear.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class MultimodeDiracModel(nn.Module):
    model_name = "base"
    output_dim = 0

    def __init__(
        self,
        cfg: MultimodeConfig,
        *,
        output_dim: int,
        width: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        basis = compute_multimode_basis(cfg.mass, cfg.wave_numbers)
        self.register_buffer("wave_numbers", basis.wave_numbers.clone())
        self.register_buffer("u_plus", basis.u_plus.clone())
        self.register_buffer("u_minus", basis.u_minus.clone())
        self.mlp = PairedTanhMLP(
            output_dim,
            width,
            cfg.hidden_layers,
            initialization_seed=initialization_seed,
        )
        self.output_dim = output_dim
        self.width = width
        self.initialization_seed = initialization_seed

    @property
    def basis(self) -> MultimodeBasis:
        return MultimodeBasis(self.wave_numbers, self.u_plus, self.u_minus)

    @property
    def initial_modal_coefficients(self) -> torch.Tensor:
        return torch.full(
            (4,),
            self.cfg.initial_mode_coefficient,
            dtype=torch.float64,
            device=self.wave_numbers.device,
        )

    def features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.sin(x), torch.cos(x), t / self.cfg.t_max), dim=1)

    def initial(self, x: torch.Tensor) -> torch.Tensor:
        return multimode_initial_solution(
            x,
            mass=self.cfg.mass,
            wave_numbers=self.cfg.wave_numbers,
            length=self.cfg.length,
        )

    def compose_modes(
        self, x: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        result = torch.zeros((x.shape[0], 2), dtype=torch.complex128, device=x.device)
        channel = 0
        for index in range(2):
            spatial = torch.exp(1j * self.wave_numbers[index] * x)
            result = result + spatial * (
                coefficients[:, channel : channel + 1]
                * self.u_plus[index].to(torch.complex128).view(1, 2)
                + coefficients[:, channel + 1 : channel + 2]
                * self.u_minus[index].to(torch.complex128).view(1, 2)
            )
            channel += 2
        return result


class MultimodeVanilla(MultimodeDiracModel):
    model_name = "vanilla"

    def __init__(self, cfg: MultimodeConfig, initialization_seed: int) -> None:
        super().__init__(
            cfg, output_dim=4, width=cfg.baseline_width,
            initialization_seed=initialization_seed,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        correction = _complex_spinor(self.mlp(self.features(x, t)))
        return self.initial(x) + (t / self.cfg.t_max) * correction


class MultimodeSinglePhaseCPA(MultimodeDiracModel):
    model_name = "sp_cpa"

    def __init__(self, cfg: MultimodeConfig, initialization_seed: int) -> None:
        super().__init__(
            cfg, output_dim=5, width=cfg.baseline_width,
            initialization_seed=initialization_seed,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        output = self.mlp(self.features(x, t))
        amplitude = self.initial(x) + (t / self.cfg.t_max) * _complex_spinor(output[:, :4])
        common_phase = (t / self.cfg.t_max) * output[:, 4:5]
        return amplitude * torch.exp(1j * common_phase)


class MultimodeComponentPolar(MultimodeDiracModel):
    model_name = "component_polar"

    def __init__(self, cfg: MultimodeConfig, initialization_seed: int) -> None:
        super().__init__(
            cfg, output_dim=4, width=cfg.baseline_width,
            initialization_seed=initialization_seed,
        )

    def component_fields(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.mlp(self.features(x, t))
        initial = self.initial(x)
        initial_amplitude = torch.abs(initial)
        if bool(torch.any(initial_amplitude <= 1.0e-12)):
            raise FloatingPointError("ComponentPolar initial component is not identifiable")
        initial_unit_phase = initial / initial_amplitude
        tau = t / self.cfg.t_max
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


class MultiChannelModel(MultimodeDiracModel):
    def mode_coefficients(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.compose_modes(x, self.mode_coefficients(x, t))


class MultiBranchComplex(MultiChannelModel):
    model_name = "multi_branch_complex"

    def __init__(self, cfg: MultimodeConfig, initialization_seed: int) -> None:
        super().__init__(
            cfg, output_dim=8, width=cfg.multichannel_width,
            initialization_seed=initialization_seed,
        )

    def mode_coefficients(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        output = self.mlp(self.features(x, t))
        correction = torch.complex(output[:, :4], output[:, 4:])
        return self.initial_modal_coefficients.to(torch.complex128).view(1, 4) + (
            t / self.cfg.t_max
        ) * correction


class MultiLTPB(MultiChannelModel):
    model_name = "multi_ltpb"

    def __init__(self, cfg: MultimodeConfig, initialization_seed: int) -> None:
        super().__init__(
            cfg, output_dim=8, width=cfg.multichannel_width,
            initialization_seed=initialization_seed,
        )

    def mode_fields(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.mlp(self.features(x, t))
        tau = t / self.cfg.t_max
        amplitudes = self.initial_modal_coefficients.view(1, 4) + tau * output[:, :4]
        phases = tau * output[:, 4:]
        return amplitudes, phases

    def mode_coefficients(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        amplitudes, phases = self.mode_fields(x, t)
        return torch.complex(amplitudes, torch.zeros_like(amplitudes)) * torch.exp(
            1j * phases
        )


class FourCarrierOracle(nn.Module):
    model_name = "oracle"

    def __init__(self, cfg: MultimodeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("device_anchor", torch.zeros((), dtype=torch.float64))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return multimode_exact_solution(
            x,
            t,
            mass=self.cfg.mass,
            wave_numbers=self.cfg.wave_numbers,
            length=self.cfg.length,
        )


def build_multimode_model(
    model_name: str, cfg: MultimodeConfig, initialization_seed: int
) -> MultimodeDiracModel:
    constructors = {
        "vanilla": MultimodeVanilla,
        "sp_cpa": MultimodeSinglePhaseCPA,
        "component_polar": MultimodeComponentPolar,
        "multi_branch_complex": MultiBranchComplex,
        "multi_ltpb": MultiLTPB,
    }
    try:
        constructor = constructors[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown multimode model: {model_name}") from exc
    return constructor(cfg, initialization_seed)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


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
