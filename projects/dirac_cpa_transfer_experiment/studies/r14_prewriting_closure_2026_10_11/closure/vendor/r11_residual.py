from __future__ import annotations
import torch
from .r8_models import paired, basis
from .r9_loss import hamiltonian_apply
from .r9_time_derivative import (network_value_and_time_derivative,
    carrier_value_and_time_derivative as learned_carrier_value_and_time_derivative)

def carrier_value_and_time_derivative(carrier, cfg, k, t):
    if getattr(carrier, "analytic_prior", False):
        return carrier.value_and_time_derivative(k, t)
    return learned_carrier_value_and_time_derivative(carrier, cfg, k, t)

def spinor_value_and_time_derivative(model, cfg, k, t):
    k, t = paired(k, t)
    output, output_t = network_value_and_time_derivative(model.networks[0], cfg, k, t)
    if len(model.networks) == 2:
        second, second_t = network_value_and_time_derivative(model.networks[1], cfg, k, t)
        output, output_t = (output+second)/2**.5, (output_t+second_t)/2**.5
    z, zt = torch.complex(output[:, :2], output[:, 2:]), torch.complex(output_t[:, :2], output_t[:, 2:])
    envelope = model.initial_coeff[model.lookup(k)]+t/cfg.t_max*z
    envelope_t = z/cfg.t_max+t/cfg.t_max*zt
    if model.carrier is not None:
        p, pt = carrier_value_and_time_derivative(model.carrier, cfg, k, t)
        coefficient, coefficient_t = p*envelope, pt*envelope+p*envelope_t
    else:
        coefficient, coefficient_t = envelope, envelope_t
    up, um = basis(k, cfg.mass)
    return (coefficient[:, :1]*up+coefficient[:, 1:]*um,
            coefficient_t[:, :1]*up+coefficient_t[:, 1:]*um)

def free_carrier_residual(carrier, cfg, k, t):
    k, t = paired(k, t)
    p, pt = carrier_value_and_time_derivative(carrier, cfg, k, t)
    up, um = basis(k, cfg.mass)
    state = torch.stack((p[:, :1]*up, p[:, 1:]*um), 1)
    dt = torch.stack((pt[:, :1]*up, pt[:, 1:]*um), 1)
    applied = k.reshape(-1, 1, 1)*state.flip(-1)
    applied = applied+cfg.mass*torch.stack((state[..., 0], -state[..., 1]), -1)
    return 1j*dt-applied

def coupled_residual(model, cfg, k, t):
    nk = cfg.mode_count
    if len(t) % nk:
        raise ValueError("Complete retained Cartesian mode grid required")
    nt = len(t)//nk
    expected = torch.tensor(cfg.wave_numbers, dtype=torch.float64, device=k.device)
    if not torch.equal(k.reshape(nt, nk), expected.expand(nt, -1)):
        raise ValueError("Ordered full Fourier grid required")
    times = t.detach().reshape(nt, nk)
    if not torch.equal(times, times[:, :1].expand_as(times)):
        raise ValueError("Coupled neighbors must have identical time")
    state, dt = spinor_value_and_time_derivative(model, cfg, k, t)
    return 1j*dt.reshape(nt, nk, 2)-hamiltonian_apply(state.reshape(nt, nk, 2), cfg)

def training_loss(model, cfg, k, t):
    residual = coupled_residual(model, cfg, k, t)
    power = residual.abs().square().mean(-1)/cfg.length
    return power.sum(-1).mean(), power.detach().mean(0)
