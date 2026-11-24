from __future__ import annotations

from dataclasses import dataclass, replace

from .periodic_potential_config import PeriodicPotentialConfig

EXPERIMENT = "coupled_ltpb_phase_release_r7"
MODELS = ("complex", "original_ltpb", "phase_release_ltpb")
TRAINING_MODES = ("ordinary", "causal")
SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class R7Config(PeriodicPotentialConfig):
    # t_max is the feature / hard-gate scale. The short experiment keeps it 5.
    training_horizon: float = 5.0
    time_bins: int = 8
    points_per_bin: int = 4
    causal_epsilons: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    epsilon_steps: int = 1000
    checkpoint_steps: tuple[int, ...] = (1000, 2500, 5000)
    recovery_every: int = 250
    residual_time_points: int = 4001
    early_time_points: int = 201
    active_occupation_threshold: float = 1.0e-6

    def __post_init__(self):
        super().__post_init__()
        if self.training_horizon <= 0 or self.training_horizon > self.t_max:
            raise ValueError("training horizon must be positive and not exceed gate scale")
        if self.time_bins * self.points_per_bin != self.spectral_time_points:
            raise ValueError("Each batch must have 8 bins x 4 times")
        if self.residual_time_points < 3 or self.residual_time_points % 2 != 1:
            raise ValueError("Nested residual quadrature needs an odd number of points")

    @property
    def loss_scale(self):
        return self.mass**2 / (2.0 * self.length)

    def evaluation_config(self):
        return replace(self, t_max=self.training_horizon)


def stage_config(stage: str, *, long_half_width: int | None = None):
    if stage == "main":
        return R7Config()
    if stage == "short":
        return R7Config(training_horizon=0.5, adam_steps=1000, checkpoint_steps=(1000,))
    if stage == "long":
        if long_half_width is None:
            raise ValueError("Long-time ladder needs reference-only selection")
        return R7Config(t_max=8.0, training_horizon=8.0, ladder_half_width=long_half_width)
    if stage == "smoke":
        return R7Config(adam_steps=20, test_x_points=64, test_t_points=17,
                        checkpoint_steps=(20,), residual_time_points=41, early_time_points=21)
    raise ValueError(stage)
