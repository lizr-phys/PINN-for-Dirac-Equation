from __future__ import annotations

import math

import torch
from torch import nn

from .config import ExperimentConfig
from .physics import Eigensystem, complex_from_four, compute_eigensystem, initial_solution


class TanhMLP(nn.Module):
    def __init__(self, output_dim: int, width: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = 3
        for _ in range(hidden_layers):
            linear = nn.Linear(input_dim, width)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.Tanh()))
            input_dim = width
        output = nn.Linear(width, output_dim)
        nn.init.xavier_normal_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class DiracModel(nn.Module):
    model_name = "base"

    def __init__(self, cfg: ExperimentConfig, eta: float, output_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.eta = eta
        eig = compute_eigensystem(cfg.mass, cfg.wave_number)
        self.energy = eig.energy
        self.cross_current = eig.cross_current
        self.register_buffer("u_plus", eig.u_plus.clone())
        self.register_buffer("u_minus", eig.u_minus.clone())
        self.mlp = TanhMLP(output_dim, cfg.hidden_width, cfg.hidden_layers)

    @property
    def eigensystem(self) -> Eigensystem:
        sigma_x_u_minus = torch.stack((self.u_minus[1], self.u_minus[0]))
        return Eigensystem(
            self.energy,
            self.u_plus,
            self.u_minus,
            self.cross_current,
        )

    def features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.sin(x), torch.cos(x), t / self.cfg.t_max), dim=1)

    def initial(self, x: torch.Tensor) -> torch.Tensor:
        return initial_solution(
            x,
            mass=self.cfg.mass,
            wave_number=self.cfg.wave_number,
            eta=self.eta,
            eigensystem=self.eigensystem,
        )


class TwoChannelDiracModel(DiracModel):
    """Base class for matched four-output two-channel ablation models.

    Subclasses expose the internal coefficients and orthonormal channel basis so
    the mechanism diagnostics can be computed without reverse engineering the
    forward pass.  Analytic energies and time carriers are deliberately absent
    from this interface.
    """

    @property
    def channel_basis_matrix(self) -> torch.Tensor:
        """Return a real ``(2, 2)`` matrix whose columns are channel vectors."""
        raise NotImplementedError

    @property
    def initial_channel_coefficients(self) -> torch.Tensor:
        """Return the real hard-initial coefficients in ``channel_basis_matrix``."""
        raise NotImplementedError

    def channel_coefficients(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return the two complex coefficients after removing ``exp(i k x)``."""
        raise NotImplementedError

    def compose_channels(
        self, x: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        basis = self.channel_basis_matrix.to(
            device=coefficients.device, dtype=torch.complex128
        )
        demodulated = coefficients @ basis.T
        return torch.exp(1j * self.cfg.wave_number * x) * demodulated

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.compose_channels(x, self.channel_coefficients(x, t))


class VanillaDiracPINN(DiracModel):
    model_name = "vanilla"

    def __init__(self, cfg: ExperimentConfig, eta: float) -> None:
        super().__init__(cfg, eta, output_dim=4)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        correction = complex_from_four(self.mlp(self.features(x, t)))
        return self.initial(x) + (t / self.cfg.t_max) * correction


class SinglePhaseCPA(DiracModel):
    model_name = "sp_cpa"

    def __init__(self, cfg: ExperimentConfig, eta: float) -> None:
        super().__init__(cfg, eta, output_dim=5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        output = self.mlp(self.features(x, t))
        amplitude_correction = complex_from_four(output[:, :4])
        norm = math.sqrt(2.0 * math.pi * (1.0 + self.eta**2))
        a0 = (self.u_plus + self.eta * self.u_minus).to(torch.complex128) / norm
        amplitude = a0.view(1, 2) + (t / self.cfg.t_max) * amplitude_correction
        phase = self.cfg.wave_number * x + (t / self.cfg.t_max) * output[:, 4:5]
        return amplitude * torch.exp(1j * phase)


class TwoBranchPINN(DiracModel):
    model_name = "two_branch"

    def __init__(self, cfg: ExperimentConfig, eta: float) -> None:
        super().__init__(cfg, eta, output_dim=4)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        correction = complex_from_four(self.mlp(self.features(x, t)))
        tau = t / self.cfg.t_max
        norm = math.sqrt(2.0 * math.pi * (1.0 + self.eta**2))
        b_plus = 1.0 / norm + tau[:, 0] * correction[:, 0]
        b_minus = self.eta / norm + tau[:, 0] * correction[:, 1]
        carrier_plus = b_plus[:, None] * self.u_plus.to(torch.complex128)[None, :] * torch.exp(
            -1j * self.energy * t
        )
        carrier_minus = b_minus[:, None] * self.u_minus.to(torch.complex128)[None, :] * torch.exp(
            1j * self.energy * t
        )
        return torch.exp(1j * self.cfg.wave_number * x) * (carrier_plus + carrier_minus)


class LearnedTwoPhaseBranchCPA(TwoChannelDiracModel):
    """Positive/negative-energy branch CPA with learned real amplitudes and phases.

    The eigenspinors are computed from the known free Dirac symbol H_k.  In
    contrast to ``TwoBranchPINN``, this forward pass never uses the spectral
    energy or an analytic time carrier: both branch phases are MLP outputs.
    """

    model_name = "ltpb_cpa"

    def __init__(self, cfg: ExperimentConfig, eta: float) -> None:
        super().__init__(cfg, eta, output_dim=4)

    @property
    def initial_branch_amplitudes(self) -> tuple[float, float]:
        norm = math.sqrt(2.0 * math.pi * (1.0 + self.eta**2))
        return 1.0 / norm, self.eta / norm

    @property
    def channel_basis_matrix(self) -> torch.Tensor:
        return torch.stack((self.u_plus, self.u_minus), dim=1)

    @property
    def initial_channel_coefficients(self) -> torch.Tensor:
        return torch.tensor(
            self.initial_branch_amplitudes,
            dtype=torch.float64,
            device=self.u_plus.device,
        )

    def branch_fields(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``a_plus, a_minus, S_plus, S_minus`` as real 1-D tensors."""
        output = self.mlp(self.features(x, t))
        tau = (t / self.cfg.t_max)[:, 0]
        amplitude_plus_initial, amplitude_minus_initial = self.initial_branch_amplitudes
        amplitude_plus = amplitude_plus_initial + tau * output[:, 0]
        amplitude_minus = amplitude_minus_initial + tau * output[:, 1]
        phase_plus = tau * output[:, 2]
        phase_minus = tau * output[:, 3]
        return amplitude_plus, amplitude_minus, phase_plus, phase_minus

    def channel_coefficients(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        amplitude_plus, amplitude_minus, phase_plus, phase_minus = self.branch_fields(x, t)
        return torch.stack(
            (
                torch.complex(amplitude_plus, torch.zeros_like(amplitude_plus))
                * torch.exp(1j * phase_plus),
                torch.complex(amplitude_minus, torch.zeros_like(amplitude_minus))
                * torch.exp(1j * phase_minus),
            ),
            dim=1,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.compose_channels(x, self.channel_coefficients(x, t))


class BranchComplexCPA(TwoChannelDiracModel):
    """Energy-eigenbasis channels with unrestricted complex corrections."""

    model_name = "branch_complex"

    def __init__(self, cfg: ExperimentConfig, eta: float) -> None:
        super().__init__(cfg, eta, output_dim=4)

    @property
    def channel_basis_matrix(self) -> torch.Tensor:
        return torch.stack((self.u_plus, self.u_minus), dim=1)

    @property
    def initial_channel_coefficients(self) -> torch.Tensor:
        norm = math.sqrt(2.0 * math.pi * (1.0 + self.eta**2))
        return torch.tensor(
            (1.0 / norm, self.eta / norm),
            dtype=torch.float64,
            device=self.u_plus.device,
        )

    def channel_coefficients(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        output = self.mlp(self.features(x, t))
        # Pair the first two output rows with the two real/amplitude channels
        # and the last two with the imaginary/phase channels.  This makes the
        # paired-seed initialization correspondence identical to the polar
        # ablations, while retaining four statistically identical outputs.
        correction = torch.complex(output[:, :2], output[:, 2:])
        initial = self.initial_channel_coefficients.to(torch.complex128)[None, :]
        return initial + (t / self.cfg.t_max) * correction


class ComponentPolarCPA(TwoChannelDiracModel):
    """Independent real amplitude/phase pairs in the raw spinor basis."""

    model_name = "component_polar"

    def __init__(self, cfg: ExperimentConfig, eta: float) -> None:
        super().__init__(cfg, eta, output_dim=4)

    @property
    def channel_basis_matrix(self) -> torch.Tensor:
        return torch.eye(2, dtype=torch.float64, device=self.u_plus.device)

    @property
    def initial_channel_coefficients(self) -> torch.Tensor:
        norm = math.sqrt(2.0 * math.pi * (1.0 + self.eta**2))
        return (self.u_plus + self.eta * self.u_minus) / norm

    def polar_fields(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.mlp(self.features(x, t))
        tau = t / self.cfg.t_max
        amplitudes = self.initial_channel_coefficients[None, :] + tau * output[:, :2]
        phases = tau * output[:, 2:]
        return amplitudes, phases

    def channel_coefficients(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        amplitudes, phases = self.polar_fields(x, t)
        return torch.complex(amplitudes, torch.zeros_like(amplitudes)) * torch.exp(
            1j * phases
        )


class RotatedLearnedTwoPhaseBranchCPA(TwoChannelDiracModel):
    """LTPB parameterization in an exactly rotated eigenspinor basis."""

    def __init__(self, cfg: ExperimentConfig, eta: float, angle_degrees: float) -> None:
        super().__init__(cfg, eta, output_dim=4)
        self.angle_degrees = float(angle_degrees)

    @property
    def channel_basis_matrix(self) -> torch.Tensor:
        angle = math.radians(self.angle_degrees)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        v1 = cosine * self.u_plus + sine * self.u_minus
        v2 = -sine * self.u_plus + cosine * self.u_minus
        return torch.stack((v1, v2), dim=1)

    @property
    def initial_channel_coefficients(self) -> torch.Tensor:
        norm = math.sqrt(2.0 * math.pi * (1.0 + self.eta**2))
        energy_coefficients = torch.tensor(
            (1.0 / norm, self.eta / norm),
            dtype=torch.float64,
            device=self.u_plus.device,
        )
        angle = math.radians(self.angle_degrees)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        energy_to_rotated = torch.tensor(
            ((cosine, sine), (-sine, cosine)),
            dtype=torch.float64,
            device=self.u_plus.device,
        )
        return energy_to_rotated @ energy_coefficients

    def polar_fields(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.mlp(self.features(x, t))
        tau = t / self.cfg.t_max
        amplitudes = self.initial_channel_coefficients[None, :] + tau * output[:, :2]
        phases = tau * output[:, 2:]
        return amplitudes, phases

    def channel_coefficients(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        amplitudes, phases = self.polar_fields(x, t)
        return torch.complex(amplitudes, torch.zeros_like(amplitudes)) * torch.exp(
            1j * phases
        )


class RotatedLTPB15(RotatedLearnedTwoPhaseBranchCPA):
    model_name = "rotated_ltpb_15"

    def __init__(self, cfg: ExperimentConfig, eta: float) -> None:
        super().__init__(cfg, eta, angle_degrees=15.0)


class RotatedLTPB30(RotatedLearnedTwoPhaseBranchCPA):
    model_name = "rotated_ltpb_30"

    def __init__(self, cfg: ExperimentConfig, eta: float) -> None:
        super().__init__(cfg, eta, angle_degrees=30.0)


def build_model(model_name: str, cfg: ExperimentConfig, eta: float) -> DiracModel:
    if model_name == "vanilla":
        return VanillaDiracPINN(cfg, eta)
    if model_name in {"sp_cpa", "sp_cpa_conservative"}:
        return SinglePhaseCPA(cfg, eta)
    if model_name == "two_branch":
        return TwoBranchPINN(cfg, eta)
    if model_name == "ltpb_cpa":
        return LearnedTwoPhaseBranchCPA(cfg, eta)
    if model_name == "branch_complex":
        return BranchComplexCPA(cfg, eta)
    if model_name == "component_polar":
        return ComponentPolarCPA(cfg, eta)
    if model_name == "rotated_ltpb_15":
        return RotatedLTPB15(cfg, eta)
    if model_name == "rotated_ltpb_30":
        return RotatedLTPB30(cfg, eta)
    raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
