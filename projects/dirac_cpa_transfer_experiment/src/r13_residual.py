from __future__ import annotations

import torch
from .r8_models import paired, basis
from .r9_time_derivative import network_value_and_time_derivative
from .r12_hamiltonian import hamiltonian_apply
from .r13_models import total_norm_parameterization, validate_cartesian

def carrier_value_and_time_derivative(carrier, cfg, k, t):
    return carrier.value_and_time_derivative(k, t)

def spinor_value_and_time_derivative(model, cfg, k, t):
    k, t = paired(k, t)
    output, tangent = network_value_and_time_derivative(model.networks[0], cfg, k, t)
    z = torch.complex(output[:, :2], output[:, 2:])
    zt = torch.complex(tangent[:, :2], tangent[:, 2:])
    raw = model.initial_coeff[model.lookup(k)]+t/cfg.t_max*z
    raw_t = z/cfg.t_max+t/cfg.t_max*zt
    if model.envelope_kind == 'hard_total_norm':
        validate_cartesian(cfg, k, t)
        raw, raw_t = total_norm_parameterization(raw, raw_t, mode_count=cfg.mode_count)
    p, pt = carrier_value_and_time_derivative(model.carrier, cfg, k, t)
    coefficient, coefficient_t = p*raw, pt*raw+p*raw_t
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
    applied = applied+cfg.mass*torch.stack((state[...,0], -state[...,1]), -1)
    return 1j*dt-applied

def coupled_residual(model, cfg, k, t):
    nt = validate_cartesian(cfg, k, t)
    state, tangent = spinor_value_and_time_derivative(model, cfg, k, t)
    return 1j*tangent.reshape(nt, cfg.mode_count, 2)-hamiltonian_apply(state.reshape(nt, cfg.mode_count, 2), cfg)

def training_loss(model, cfg, k, t):
    nt = validate_cartesian(cfg, k, t)
    residual = coupled_residual(model, cfg, k, t)
    power = residual.abs().square().mean(-1)/cfg.length
    pde_loss = power.sum(-1).mean()
    if model.envelope_kind == 'norm_regularized':
        state = model(k, t).reshape(nt, cfg.mode_count, 2)
        norm_penalty = (state.abs().square().sum((1,2))-1).square().mean()
        loss = pde_loss+cfg.norm_regularization_weight*norm_penalty
    else:
        norm_penalty = pde_loss.new_zeros(())
        loss = pde_loss
    return loss, power.detach().mean(0), dict(pde_loss=float(pde_loss.detach()), norm_penalty=float(norm_penalty.detach()))
