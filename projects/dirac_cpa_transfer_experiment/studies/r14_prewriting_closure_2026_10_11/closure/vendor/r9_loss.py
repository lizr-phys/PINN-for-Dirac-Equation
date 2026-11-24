from __future__ import annotations
import torch
from .r8_loss import derivative
from .r9_time_derivative import spinor_value_and_time_derivative, unit_carrier_spinor_residual as free_carrier_residual


def hamiltonian_apply(state, cfg):
    k = torch.tensor(cfg.wave_numbers, dtype=torch.float64, device=state.device)
    result = k[..., None]*state.flip(-1)
    result = result+cfg.mass*torch.stack((state[..., 0], -state[..., 1]), -1)
    # q=1, dk=1/8: eight Fourier indices, not one index.
    shift = 8
    neighbors = torch.zeros_like(state)
    neighbors[..., shift:, :] = state[..., :-shift, :]
    neighbors[..., :-shift, :] = neighbors[..., :-shift, :]+state[..., shift:, :]
    return result+cfg.potential_strength/2*neighbors


def coupled_residual(model, cfg, k, t, create_graph=True):
    nk = cfg.mode_count
    if len(t) % nk:
        raise ValueError("Full retained Fourier grid required")
    nt = len(t)//nk
    expected = torch.tensor(cfg.wave_numbers, dtype=torch.float64, device=k.device)
    if not torch.equal(k.reshape(nt, nk), expected.expand(nt, -1)):
        raise ValueError("Ordered full Fourier grid required")
    times = t.detach().reshape(nt, nk)
    if not torch.equal(times, times[:, :1].expand_as(times)):
        raise ValueError("All coupled neighbors must have the same physical time")
    # The learned formula is unchanged; exact tanh chain rules retain p_t and its defect.
    if hasattr(model, "networks"):
        state, dt = spinor_value_and_time_derivative(model, cfg, k, t)
    else:
        state = model(k, t)
        dt = derivative(state, t, create_graph)
    return 1j*dt.reshape(nt, nk, 2)-hamiltonian_apply(state.reshape(nt, nk, 2), cfg)


def training_loss(model, cfg, k, t):
    residual = coupled_residual(model, cfg, k, t)
    # Unweighted Parseval loss, unchanged R8 normalization.
    power = residual.abs().square().mean(-1)/cfg.length
    return power.sum(-1).mean(), power.detach().mean(0)
