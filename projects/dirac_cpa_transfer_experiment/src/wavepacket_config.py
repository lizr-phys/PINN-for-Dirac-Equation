from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


WAVEPACKET_EXPERIMENT = "wavepacket_shared_spectral_ltpb"
WAVEPACKET_MODELS = (
    "physical_vanilla",
    "physical_sp_cpa",
    "physical_component_polar",
    "shared_spectral_complex",
    "shared_spectral_ltpb",
)
WAVEPACKET_BASELINES = WAVEPACKET_MODELS[:-1]
WAVEPACKET_SEEDS = (0, 1, 2, 3, 4)

MODEL_LABELS = {
    "physical_vanilla": "Physical-Vanilla",
    "physical_sp_cpa": "Physical-SP-CPA",
    "physical_component_polar": "Physical-ComponentPolar",
    "shared_spectral_complex": "Shared-Spectral-Complex",
    "shared_spectral_ltpb": "Shared-Spectral-LTPB",
    "spectral_oracle": "Spectral oracle",
}

INITIALIZATION_SEED_OFFSET = 200_000
COLLOCATION_SEED_OFFSET = 300_000


def paired_initialization_seed(seed: int) -> int:
    """Return the model-independent initialization stream for a paired seed."""

    return INITIALIZATION_SEED_OFFSET + int(seed)


def paired_collocation_seed(seed: int) -> int:
    """Return the model-independent collocation/time stream for a paired seed."""

    return COLLOCATION_SEED_OFFSET + int(seed)


@dataclass(frozen=True)
class WavepacketConfig:
    """Frozen free-Dirac Gaussian-wavepacket experiment configuration.

    The configuration deliberately contains only quantities known from the
    equation and initial state.  Analytic time rates belong exclusively to the
    reference/evaluation layer and are not represented here.
    """

    x_min: float = -8.0 * math.pi
    x_max: float = 8.0 * math.pi
    t_max: float = 5.0
    mass: float = 5.0
    n_min: int = 0
    n_max: int = 32
    k0: float = 2.0
    sigma_k: float = 0.5
    hidden_layers: int = 4
    hidden_width: int = 64
    physical_collocation_points: int = 1024
    spectral_time_points: int = 32
    adam_steps: int = 5000
    learning_rate: float = 1.0e-3
    test_x_points: int = 256
    test_t_points: int = 201
    history_every: int = 10
    log_every: int = 250
    dtype: str = "float64"
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        if not self.x_max > self.x_min:
            raise ValueError("x_max must be larger than x_min")
        if self.n_min < 0 or self.n_max < self.n_min:
            raise ValueError("mode indices must define a nonempty nonnegative range")
        if self.sigma_k <= 0.0:
            raise ValueError("sigma_k must be positive")
        if self.t_max <= 0.0:
            raise ValueError("t_max must be positive")
        if self.hidden_layers <= 0 or self.hidden_width <= 0:
            raise ValueError("network depth and width must be positive")
        if self.spectral_time_points <= 0:
            raise ValueError("spectral_time_points must be positive")

    @property
    def length(self) -> float:
        return self.x_max - self.x_min

    @property
    def wave_number_spacing(self) -> float:
        return 2.0 * math.pi / self.length

    @property
    def mode_count(self) -> int:
        return self.n_max - self.n_min + 1

    @property
    def wave_numbers(self) -> tuple[float, ...]:
        spacing = self.wave_number_spacing
        return tuple(
            spacing * index
            for index in range(self.n_min, self.n_max + 1)
        )

    @property
    def wave_number_min(self) -> float:
        return self.wave_numbers[0]

    @property
    def wave_number_max(self) -> float:
        return self.wave_numbers[-1]

    @property
    def spectral_collocation_points(self) -> int:
        return self.spectral_time_points * self.mode_count

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "length": self.length,
                "wave_number_spacing": self.wave_number_spacing,
                "mode_count": self.mode_count,
                "wave_numbers": list(self.wave_numbers),
                "spectral_collocation_points": self.spectral_collocation_points,
            }
        )
        return data


@dataclass(frozen=True)
class WavepacketRunSpec:
    model: str
    seed: int
    steps: int = 5000
    experiment: str = WAVEPACKET_EXPERIMENT

    @property
    def run_name(self) -> str:
        return f"{self.experiment}__{self.model}__seed_{self.seed}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"run_name": self.run_name}
