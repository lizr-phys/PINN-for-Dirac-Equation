from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CASES = {"P": 0.0, "M": 1.0}
MODELS = ("vanilla", "sp_cpa", "sp_cpa_conservative", "two_branch")
PRINCIPAL_MODELS = ("vanilla", "sp_cpa", "sp_cpa_conservative")


@dataclass(frozen=True)
class ExperimentConfig:
    x_min: float = -math.pi
    x_max: float = math.pi
    t_max: float = 5.0
    mass: float = 5.0
    wave_number: float = 2.0
    hidden_layers: int = 4
    hidden_width: int = 64
    collocation_points: int = 1024
    adam_steps: int = 5000
    learning_rate: float = 1.0e-3
    struct_time_slices: int = 4
    struct_x_points: int = 64
    test_x_points: int = 128
    test_t_points: int = 201
    gradient_diagnostic_every: int = 250
    history_every: int = 10
    log_every: int = 250
    dtype: str = "float64"
    cpu_threads: int = 8

    @property
    def length(self) -> float:
        return self.x_max - self.x_min

    @property
    def energy(self) -> float:
        return math.sqrt(self.mass**2 + self.wave_number**2)

    @property
    def reference_density(self) -> float:
        return 1.0 / self.length

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "length": self.length,
                "spectral_energy": self.energy,
                "reference_density": self.reference_density,
            }
        )
        return data


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    case: str
    model: str
    seed: int
    lambda_struct: float = 1.0
    steps: int = 5000

    @property
    def eta(self) -> float:
        return CASES[self.case]

    @property
    def run_name(self) -> str:
        lam = f"{self.lambda_struct:g}".replace(".", "p")
        return f"{self.experiment}__case_{self.case}__{self.model}__seed_{self.seed}__lambda_{lam}"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["eta"] = self.eta
        data["run_name"] = self.run_name
        return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)

