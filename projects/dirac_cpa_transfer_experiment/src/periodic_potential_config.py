from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


PERIODIC_POTENTIAL_EXPERIMENT = "periodic_potential_coupled_ltpb"
PERIODIC_POTENTIAL_MODELS = (
    "physical_vanilla",
    "coupled_shared_spectral_complex",
    "coupled_shared_spectral_ltpb",
)
PERIODIC_POTENTIAL_SEEDS = (0, 1, 2, 3, 4)

INITIALIZATION_SEED_OFFSET = 500_000
TIME_COLLOCATION_SEED_OFFSET = 600_000
SPACE_COLLOCATION_SEED_OFFSET = 700_000


def paired_initialization_seed(seed: int) -> int:
    return INITIALIZATION_SEED_OFFSET + int(seed)


def paired_time_collocation_seed(seed: int) -> int:
    return TIME_COLLOCATION_SEED_OFFSET + int(seed)


def paired_space_collocation_seed(seed: int) -> int:
    return SPACE_COLLOCATION_SEED_OFFSET + int(seed)


@dataclass(frozen=True)
class PeriodicPotentialConfig:
    """Configuration known from the equation and the initial condition.

    ``potential_strength``, ``t_max`` and ``ladder_half_width`` are populated
    with the intended formal values only after reference-only calibration has
    selected and frozen them.  No analytic dispersion or reference trajectory
    is stored here.
    """

    x_min: float = -8.0 * math.pi
    x_max: float = 8.0 * math.pi
    t_max: float = 5.0
    mass: float = 5.0
    k0: float = 2.0
    coupling_wave_number: float = 1.0
    potential_strength: float = 0.2
    ladder_half_width: int = 6
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
        if self.t_max <= 0.0:
            raise ValueError("t_max must be positive")
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.coupling_wave_number <= 0.0:
            raise ValueError("coupling_wave_number must be positive")
        if self.potential_strength < 0.0:
            raise ValueError("potential_strength cannot be negative")
        if self.ladder_half_width < 1:
            raise ValueError("ladder_half_width must be at least one")
        if self.hidden_layers <= 0 or self.hidden_width <= 0:
            raise ValueError("network depth and width must be positive")
        if self.physical_collocation_points <= 0:
            raise ValueError("physical_collocation_points must be positive")
        if self.spectral_time_points <= 0:
            raise ValueError("spectral_time_points must be positive")

        ratio = self.coupling_wave_number / self.fundamental_wave_number
        if abs(ratio - round(ratio)) > 1.0e-12:
            raise ValueError(
                "coupling wave number must be commensurate with the periodic grid"
            )
        center_ratio = self.k0 / self.fundamental_wave_number
        if abs(center_ratio - round(center_ratio)) > 1.0e-12:
            raise ValueError("k0 must be commensurate with the periodic grid")

    @property
    def length(self) -> float:
        return self.x_max - self.x_min

    @property
    def fundamental_wave_number(self) -> float:
        return 2.0 * math.pi / self.length

    @property
    def mode_count(self) -> int:
        return 2 * self.ladder_half_width + 1

    @property
    def ladder_indices(self) -> tuple[int, ...]:
        half_width = self.ladder_half_width
        return tuple(range(-half_width, half_width + 1))

    @property
    def wave_numbers(self) -> tuple[float, ...]:
        return tuple(
            self.k0 + index * self.coupling_wave_number
            for index in self.ladder_indices
        )

    @property
    def spectral_collocation_points(self) -> int:
        return self.spectral_time_points * self.mode_count

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "length": self.length,
                "fundamental_wave_number": self.fundamental_wave_number,
                "mode_count": self.mode_count,
                "ladder_indices": list(self.ladder_indices),
                "wave_numbers": list(self.wave_numbers),
                "spectral_collocation_points": self.spectral_collocation_points,
            }
        )
        return data


@dataclass(frozen=True)
class ReferenceCalibrationConfig:
    """Reference-only calibration protocol, independent of learned models."""

    candidate_potential_strengths: tuple[float, ...] = (0.1, 0.2, 0.5)
    primary_t_max: float = 5.0
    fallback_t_max: float = 8.0
    small_half_width: int = 4
    large_half_width: int = 6
    time_points: int = 4001
    sideband_occupation_min: float = 0.10
    sideband_occupation_max: float = 0.60
    outer_mode_occupation_max: float = 1.0e-8
    spinor_truncation_relative_l2_max: float = 1.0e-6
    hermiticity_tolerance: float = 2.0e-15
    norm_drift_tolerance: float = 2.0e-13
    time_reversal_tolerance: float = 2.0e-13

    def __post_init__(self) -> None:
        if not self.candidate_potential_strengths:
            raise ValueError("at least one candidate potential is required")
        if any(value <= 0.0 for value in self.candidate_potential_strengths):
            raise ValueError("candidate potential strengths must be positive")
        if tuple(sorted(self.candidate_potential_strengths)) != (
            self.candidate_potential_strengths
        ):
            raise ValueError("candidate potential strengths must be increasing")
        if self.primary_t_max <= 0.0 or self.fallback_t_max <= self.primary_t_max:
            raise ValueError("fallback time must exceed the positive primary time")
        if not 1 <= self.small_half_width < self.large_half_width:
            raise ValueError("calibration half widths must satisfy 1 <= small < large")
        if self.time_points < 2:
            raise ValueError("time_points must be at least two")
        if not 0.0 <= self.sideband_occupation_min < self.sideband_occupation_max:
            raise ValueError("invalid sideband occupation interval")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
