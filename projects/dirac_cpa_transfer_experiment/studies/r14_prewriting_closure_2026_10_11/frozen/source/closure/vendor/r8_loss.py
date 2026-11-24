from __future__ import annotations

import torch
from .physics import complex_derivative
from .periodic_potential_physics import coupled_spectral_residual
from .r8_models import basis


def derivative(values, t, create_graph=True):
    return torch.stack([complex_derivative(values[:, s], t, create_graph=create_graph)
                        for s in range(values.shape[1])], dim=1)


def coupled_residual(model, cfg, k, t, create_graph=True):
    nk = cfg.mode_count
    if len(t) % nk:
        raise ValueError("Complete ladder required")
    nt = len(t)//nk
    expected = torch.tensor(cfg.wave_numbers, dtype=torch.float64, device=k.device)
    if not torch.equal(k.reshape(nt, nk), expected.expand(nt, -1)):
        raise ValueError("Ordered complete ladder required")
    tm = t.reshape(nt, nk)
    if not torch.equal(tm.detach(), tm[:, :1].detach().expand_as(tm)):
        raise ValueError("Neighbors must be evaluated at the same time")
    state = model(k, t)
    dt = derivative(state, t, create_graph)
    return coupled_spectral_residual(state.reshape(nt, nk, 2), dt.reshape(nt, nk, 2),
                                    expected, cfg.mass, cfg.potential_strength)


def free_carrier_residual(carrier, cfg, k, t, create_graph=True):
    p = carrier(k, t)
    up, um = basis(k, cfg.mass)
    # Separate nonzero unit spinor IVPs, not a cancelling +/- superposition.
    state = torch.stack((p[:, :1]*up, p[:, 1:]*um), dim=1)
    dt = torch.stack([derivative(state[:, branch], t, create_graph) for branch in range(2)], dim=1)
    h = k.reshape(-1, 1, 1)*state.flip(-1)
    h = h+cfg.mass*torch.stack((state[..., 0], -state[..., 1]), dim=-1)
    return 1j*dt-h


class ResidualBalance:
    """One preregistered rescue, not used by the primary six configurations."""
    def __init__(self, nk):
        self.ema = torch.ones(nk, dtype=torch.float64)

    def weights(self, power):
        with torch.no_grad():
            self.ema = .99*self.ema+.01*power.detach().mean(0)
            raw = 1/(self.ema+1e-8)
            raw = torch.clamp(raw/raw.mean(), .1, 10)
            return raw/raw.mean()


def training_loss(model, cfg, k, t, balance=None):
    residual = coupled_residual(model, cfg, k, t)
    power = residual.abs().square().mean(-1)/cfg.length
    weights = torch.ones(cfg.mode_count, dtype=torch.float64) if balance is None else balance.weights(power)
    return (power*weights[None]).sum(-1).mean(), power.mean(0), weights
