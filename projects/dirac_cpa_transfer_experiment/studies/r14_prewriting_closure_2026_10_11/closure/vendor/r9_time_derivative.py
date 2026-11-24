from __future__ import annotations

"""Exact chain-rule time derivatives of the unchanged tanh MLPs.

This avoids four separate reverse sweeps for two complex outputs. All tangent
operations retain the parameter graph; no temporal or reference approximation.
"""
import torch
from torch import nn
from .r8_models import features, paired, basis


def network_value_and_time_derivative(network, cfg, k, t):
    value = features(cfg, k, t)
    tangent = torch.cat((torch.zeros_like(t), torch.ones_like(t)/cfg.t_max), 1)
    for layer in network.net:
        if isinstance(layer, nn.Linear):
            value = layer(value)
            tangent = torch.nn.functional.linear(tangent, layer.weight)
        elif isinstance(layer, nn.Tanh):
            value = torch.tanh(value)
            tangent = (1-value.square())*tangent
        else:
            raise TypeError("Only the registered Linear/Tanh architecture is allowed")
    return value, tangent


def carrier_value_and_time_derivative(carrier, cfg, k, t):
    k, t = paired(k, t)
    q, qt = network_value_and_time_derivative(carrier.mlp, cfg, k, t)
    p = torch.exp(1j*(t/cfg.t_max)*q)
    pt = 1j*(q/cfg.t_max+(t/cfg.t_max)*qt)*p
    return p, pt


def spinor_value_and_time_derivative(model, cfg, k, t):
    k, t = paired(k, t)
    output, output_t = network_value_and_time_derivative(model.networks[0], cfg, k, t)
    if len(model.networks) == 2:
        second, second_t = network_value_and_time_derivative(model.networks[1], cfg, k, t)
        output, output_t = (output+second)/2**.5, (output_t+second_t)/2**.5
    z = torch.complex(output[:, :2], output[:, 2:])
    zt = torch.complex(output_t[:, :2], output_t[:, 2:])
    c = model.initial_coeff[model.lookup(k)]+t/cfg.t_max*z
    ct = z/cfg.t_max+t/cfg.t_max*zt
    if model.carrier is not None:
        p, pt = carrier_value_and_time_derivative(model.carrier, cfg, k, t)
        c, ct = p*c, pt*c+p*ct
    up, um = basis(k, cfg.mass)
    return c[:, :1]*up+c[:, 1:]*um, ct[:, :1]*up+ct[:, 1:]*um


def unit_carrier_spinor_residual(carrier, cfg, k, t, create_graph=True):
    k, t = paired(k, t)
    p, pt = carrier_value_and_time_derivative(carrier, cfg, k, t)
    up, um = basis(k, cfg.mass)
    state = torch.stack((p[:, :1]*up, p[:, 1:]*um), 1)
    dt = torch.stack((pt[:, :1]*up, pt[:, 1:]*um), 1)
    applied = k.reshape(-1, 1, 1)*state.flip(-1)
    applied = applied+cfg.mass*torch.stack((state[..., 0], -state[..., 1]), -1)
    return 1j*dt-applied
