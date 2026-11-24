from __future__ import annotations

import hashlib

import torch

from .periodic_potential_config import paired_time_collocation_seed


class StratifiedTimeSampler:
    """Explicit, recoverable, model-independent time stream, ordered by bin."""

    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.generator = torch.Generator().manual_seed(paired_time_collocation_seed(seed))
        self.trace = []
        self.k = torch.tensor(cfg.wave_numbers, dtype=torch.float64).reshape(-1, 1)

    @property
    def batch_index(self):
        return len(self.trace)

    def sample(self):
        cfg = self.cfg
        random = torch.rand((cfg.time_bins, cfg.points_per_bin),
                            generator=self.generator, dtype=torch.float64)
        random = torch.sort(random, dim=1).values
        bins = torch.arange(cfg.time_bins, dtype=torch.float64)[:, None]
        times = ((bins + random) * (cfg.training_horizon / cfg.time_bins)).reshape(-1, 1)
        self.trace.append(times.detach().clone())
        return (self.k.repeat(cfg.spectral_time_points, 1),
                times.repeat_interleave(cfg.mode_count, dim=0).detach().requires_grad_(True))

    def trace_tensor(self):
        return torch.stack(self.trace) if self.trace else torch.empty((0, 32, 1), dtype=torch.float64)

    def digest(self):
        return hashlib.sha256(self.trace_tensor().numpy().tobytes()).hexdigest()

    def state_dict(self):
        return {"generator_state": self.generator.get_state(), "time_trace": self.trace_tensor()}

    def load_state_dict(self, state):
        self.generator.set_state(state["generator_state"])
        self.trace = list(state["time_trace"].unbind(0))
