from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

EXPERIMENT = "r9_wavepacket_periodic_potential"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw" / EXPERIMENT
DOCS = ROOT / "docs/experiments" / EXPERIMENT
FIGURES = ROOT / "results/figures" / EXPERIMENT
LOGS = ROOT / "results/logs" / EXPERIMENT
MODELS = ("A_carrier_complex", "B_complex", "C_double_complex")
SEEDS = (300, 301, 302, 303, 304)
TRUNCATIONS = ((-32, 64), (-48, 80), (-64, 96), (-80, 112))
THRESHOLDS = {"spinor_rel_l2_gauge_aligned": .10, "current_rel_l2": .15,
              "fidelity_mean": .98, "modal_population_rel_l2": .10, "max_norm_drift": .02}
ARTIFACTS = ("config.json", "metrics.json", "loss_history.csv", "time_series.csv",
             "dense_grid.npz", "model_state.pt")


@dataclass(frozen=True)
class R9Config:
    n_min: int = -32
    n_max: int = 64
    mass: float = 5.
    t_max: float = 5.
    potential_strength: float = .2
    coupling_wave_number: float = 1.
    k0: float = 2.
    sigma_k: float = .5
    x_min: float = -8*math.pi
    x_max: float = 8*math.pi
    hidden_layers: int = 4
    hidden_width: int = 64
    adam_steps: int = 20000
    carrier_steps: int = 6000
    learning_rate: float = .001
    spectral_time_points: int = 32
    test_x_points: int = 1024
    test_t_points: int = 201
    residual_time_points: int = 4001
    history_every: int = 25
    log_every: int = 500
    recovery_every: int = 500
    active_occupation_threshold: float = 1e-6

    def __post_init__(self):
        if self.n_min > 0 or self.n_max < 32 or self.n_max+self.n_min != 32:
            raise ValueError("Symmetric truncation must contain the unchanged R5 support")
        if self.mode_count < 17:
            raise ValueError("Truncation too small for shift-eight coupling")

    @property
    def length(self):
        return self.x_max-self.x_min

    @property
    def mode_count(self):
        return self.n_max-self.n_min+1

    @property
    def indices(self):
        return tuple(range(self.n_min, self.n_max+1))

    @property
    def wave_numbers(self):
        return tuple(n/8 for n in self.indices)

    @property
    def ladder_half_width(self):
        # R8's unchanged carrier feature scale is half of the retained k span.
        return (self.n_max-self.n_min)/16

    @property
    def spectral_collocation_points(self):
        return self.spectral_time_points*self.mode_count

    def as_dict(self):
        return asdict(self) | dict(length=self.length, mode_count=self.mode_count,
                                  wave_numbers=list(self.wave_numbers), coupling_index_shift=8)


def run_directory(model, seed, *, smoke=False):
    return RAW/("smoke" if smoke else "formal")/model/f"seed_{seed}"
