from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


MULTIMODE_EXPERIMENT = "multimode_ltpb"
MULTIMODE_MODELS = (
    "vanilla",
    "sp_cpa",
    "component_polar",
    "multi_branch_complex",
    "multi_ltpb",
)
MULTIMODE_SEEDS = (0, 1, 2, 3, 4)
MODE_KEYS = ("k2_plus", "k2_minus", "k5_plus", "k5_minus")
MODEL_LABELS = {
    "vanilla": "Vanilla",
    "sp_cpa": "SP-CPA",
    "component_polar": "ComponentPolar",
    "multi_branch_complex": "MultiBranchComplex",
    "multi_ltpb": "Multi-LTPB",
    "oracle": "Four-carrier oracle",
}


@dataclass(frozen=True)
class MultimodeConfig:
    x_min: float = 0.0
    x_max: float = 2.0 * math.pi
    t_max: float = 5.0
    mass: float = 5.0
    wave_numbers: tuple[float, float] = (2.0, 5.0)
    hidden_layers: int = 4
    baseline_width: int = 64
    multichannel_width: int = 63
    collocation_points: int = 1024
    adam_steps: int = 5000
    learning_rate: float = 1.0e-3
    test_x_points: int = 128
    test_t_points: int = 201
    history_every: int = 10
    log_every: int = 250
    dtype: str = "float64"
    cpu_threads: int = 1

    @property
    def length(self) -> float:
        return self.x_max - self.x_min

    @property
    def initial_mode_coefficient(self) -> float:
        return 1.0 / (2.0 * math.sqrt(self.length))

    def width_for_model(self, model: str) -> int:
        if model in {"multi_branch_complex", "multi_ltpb"}:
            return self.multichannel_width
        return self.baseline_width

    def as_dict(self) -> dict[str, Any]:
        # Exact spectral energies are intentionally absent.  They are generated
        # only by reference/evaluation code after optimization.
        data = asdict(self)
        data["length"] = self.length
        data["initial_mode_coefficient"] = self.initial_mode_coefficient
        return data


@dataclass(frozen=True)
class MultimodeRunSpec:
    model: str
    seed: int
    steps: int = 5000
    experiment: str = MULTIMODE_EXPERIMENT

    @property
    def run_name(self) -> str:
        return f"{self.experiment}__{self.model}__seed_{self.seed}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"run_name": self.run_name}
