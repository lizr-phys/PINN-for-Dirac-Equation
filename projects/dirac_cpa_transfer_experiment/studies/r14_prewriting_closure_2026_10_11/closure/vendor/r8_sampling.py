from __future__ import annotations
import hashlib
import torch


class TimeSampler:
    """Paired normalized uniforms; physical-time mappings differ by window."""
    def __init__(self, cfg, seed, *, carrier=False):
        self.cfg = cfg
        self.generator = torch.Generator().manual_seed((910000 if carrier else 920000)+seed)
        self.trace = []
        self.normalized_trace = []
        self.k = torch.tensor(cfg.wave_numbers, dtype=torch.float64).reshape(-1, 1)

    def sample(self, start=0., duration=None):
        u = torch.sort(torch.rand((32, 1), generator=self.generator, dtype=torch.float64), dim=0).values
        times = start+(self.cfg.t_max if duration is None else duration)*u
        self.trace.append(times.clone())
        self.normalized_trace.append(u.clone())
        return self.k.repeat(32, 1), times.repeat_interleave(self.cfg.mode_count, dim=0).requires_grad_()

    def state_dict(self):
        return dict(generator_state=self.generator.get_state(),
                    trace=torch.stack(self.trace) if self.trace else torch.empty((0, 32, 1), dtype=torch.float64),
                    normalized_trace=torch.stack(self.normalized_trace) if self.normalized_trace else torch.empty((0, 32, 1), dtype=torch.float64))

    def load_state_dict(self, state):
        self.generator.set_state(state["generator_state"])
        self.trace, self.normalized_trace = list(state["trace"].unbind()), list(state["normalized_trace"].unbind())

    def digest(self, normalized=False):
        values = self.normalized_trace if normalized else self.trace
        return hashlib.sha256(torch.stack(values).numpy().tobytes()).hexdigest() if values else hashlib.sha256(b"").hexdigest()
