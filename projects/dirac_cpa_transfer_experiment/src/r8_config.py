from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from .periodic_potential_config import PeriodicPotentialConfig

EXPERIMENT = "r8_coupling_recovery"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw" / EXPERIMENT
DOCS = ROOT / "docs/experiments" / EXPERIMENT
FIGURES = ROOT / "results/figures" / EXPERIMENT
LOGS = ROOT / "results/logs" / EXPERIMENT
CANDIDATES = ("A_carrier_complex", "B_free_correction", "C_release_windows", "D_carrier_windows")
CONTROLS = ("complex", "phase_release")
EXPLORATION_SEEDS = (0, 1, 2)
VALIDATION_SEEDS = (100, 101, 102, 103, 104)
EXTENSION_SEEDS = (200, 201, 202)
THRESHOLDS = {"spinor_rel_l2_gauge_aligned": 0.10, "current_rel_l2": 0.15,
              "fidelity_mean": 0.98, "modal_population_rel_l2": 0.10, "max_norm_drift": 0.02}


@dataclass(frozen=True)
class R8Config(PeriodicPotentialConfig):
    adam_steps: int = 10000
    carrier_steps: int = 3000
    windows: int = 10
    residual_time_points: int = 4001
    active_occupation_threshold: float = 1e-6
    history_every: int = 25
    log_every: int = 500
    recovery_every: int = 250
    case: str = "main"

    @property
    def training_horizon(self):
        return self.t_max

    @property
    def time_bins(self):
        return 10

    def as_dict(self):
        return super().as_dict()


def config_for(stage, case="main", half_width=None):
    if stage == "smoke":
        cfg = R8Config(adam_steps=40, carrier_steps=10, residual_time_points=101,
                       test_x_points=32, test_t_points=21, history_every=1, log_every=10)
    elif stage in ("exploration", "rescue"):
        cfg = R8Config()
    elif stage in ("validation", "ablation", "extension"):
        cfg = R8Config(adam_steps=20000, carrier_steps=6000)
    else:
        raise ValueError(stage)
    cases = {"main": {}, "T8": {"t_max": 8.0}, "V01": {"potential_strength": .1},
             "V04": {"potential_strength": .4},
             "free_packet": {"potential_strength": 0.0, "coupling_wave_number": .125,
                             "ladder_half_width": 16}}
    if case not in cases:
        raise ValueError(case)
    cfg = replace(cfg, case=case, **cases[case])
    if half_width is not None:
        cfg = replace(cfg, ladder_half_width=half_width)
    return cfg


def has_carrier(model):
    return model in ("A_carrier_complex", "B_free_correction", "D_carrier_windows",
                     "A_identity_carrier", "D_identity_carrier", "B_static_free")


def windowed(model):
    return model in ("C_release_windows", "D_carrier_windows", "window_complex", "D_identity_carrier")


def allocation(cfg, model):
    pre = cfg.carrier_steps if has_carrier(model) else 0
    count = cfg.windows if windowed(model) else 1
    if (cfg.adam_steps-pre) % count:
        raise ValueError("Window budgets must divide exactly")
    return pre, count, (cfg.adam_steps-pre)//count


def run_directory(stage, model, seed, *, case="main", balanced=False):
    return RAW/stage/case/(model+("_balanced" if balanced else ""))/f"seed_{seed}"
