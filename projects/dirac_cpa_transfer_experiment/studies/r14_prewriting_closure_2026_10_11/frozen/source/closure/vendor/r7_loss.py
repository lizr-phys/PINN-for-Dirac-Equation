from __future__ import annotations

import torch

from .physics import complex_derivative
from .periodic_potential_physics import coupled_spectral_residual


def epsilon_for_step(cfg, step):
    return cfg.causal_epsilons[min((step - 1) // cfg.epsilon_steps, len(cfg.causal_epsilons)-1)]


def causal_weights(segment_losses, loss_scale, epsilon):
    previous = torch.cat((torch.zeros_like(segment_losses[:1]),
                          torch.cumsum(segment_losses.detach() / loss_scale, dim=0)[:-1]))
    return torch.exp(-float(epsilon) * previous)


def coupled_residual(model, cfg, k, t, *, create_graph=True):
    n_t = len(t) // cfg.mode_count
    ks = k.reshape(n_t, cfg.mode_count)
    times = t.reshape(n_t, cfg.mode_count)
    expected = torch.tensor(cfg.wave_numbers, dtype=torch.float64, device=k.device)
    if not torch.equal(ks, expected.expand(n_t, -1)):
        raise ValueError("Ordered complete momentum ladder required")
    if not torch.equal(times.detach(), times[:, :1].detach().expand_as(times)):
        raise ValueError("Neighbors must share time")
    psi = model(k, t)
    derivative = torch.stack([complex_derivative(psi[:, s], t, create_graph=create_graph)
                              for s in range(2)], dim=1)
    return coupled_spectral_residual(psi.reshape(n_t, cfg.mode_count, 2),
                                    derivative.reshape(n_t, cfg.mode_count, 2),
                                    expected, cfg.mass, cfg.potential_strength)


def training_loss(model, cfg, k, t, training_mode, step):
    residual = coupled_residual(model, cfg, k, t)
    power_t = torch.sum(torch.abs(residual)**2, dim=1).mean(dim=1) / cfg.length
    segments = power_t.reshape(cfg.time_bins, cfg.points_per_bin).mean(dim=1)
    if training_mode == "ordinary":
        weights = torch.ones_like(segments)
    elif training_mode == "causal":
        weights = causal_weights(segments, cfg.loss_scale, epsilon_for_step(cfg, step))
    else:
        raise ValueError(training_mode)
    return (weights * segments).mean(), segments, weights
