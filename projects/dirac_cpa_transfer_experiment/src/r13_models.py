from __future__ import annotations

import torch
from torch import nn
from .r8_models import paired, features
from .r9_models import R9Model, tensor_digest, parameter_counts
from .wavepacket_models import PairedTanhMLP
from .r9_time_derivative import network_value_and_time_derivative
from .r13_config import CARRIER_KINDS, ENVELOPE_KINDS

class LearnedRecoveryCarrier(nn.Module):
    """Learned free phase. Exact eigenfrequencies are absent from this module."""
    analytic_prior = False

    def __init__(self, cfg, kind, seed):
        super().__init__()
        if kind not in CARRIER_KINDS:
            raise ValueError(kind)
        self.cfg, self.kind = cfg, kind
        self.phase_scale = cfg.mass*cfg.t_max if kind == 'scaled_phase' else 1.
        original = PairedTanhMLP(2, 2, cfg.hidden_width, cfg.hidden_layers, initialization_seed=900000+seed)
        if kind == 'static_frequency':
            self.mlp = PairedTanhMLP(1, 2, cfg.hidden_width, cfg.hidden_layers, initialization_seed=900000+seed)
            with torch.no_grad():
                for i, layer in enumerate(self.mlp.net):
                    if isinstance(layer, nn.Linear):
                        layer.weight.copy_(original.net[i].weight[:, :1] if i == 0 else original.net[i].weight)
                        layer.bias.copy_(original.net[i].bias)
                self.mlp.net[-1].weight.div_(-cfg.t_max)
                self.mlp.net[-1].bias.div_(-cfg.t_max)
        else:
            self.mlp = original
            if kind == 'scaled_phase':
                with torch.no_grad():
                    self.mlp.net[-1].weight.div_(self.phase_scale)
                    self.mlp.net[-1].bias.div_(self.phase_scale)

    def phase(self, k, t):
        k, t = paired(k, t)
        if self.kind == 'static_frequency':
            raw_rate = self.mlp((k-self.cfg.k0)/self.cfg.ladder_half_width)
            return -t*raw_rate
        return (t/self.cfg.t_max)*self.phase_scale*self.mlp(features(self.cfg, k, t))

    def forward(self, k, t):
        return torch.exp(1j*self.phase(k, t))

    def value_and_time_derivative(self, k, t):
        k, t = paired(k, t)
        if self.kind == 'static_frequency':
            raw_rate = self.mlp((k-self.cfg.k0)/self.cfg.ladder_half_width)
            phase, slope = -t*raw_rate, -raw_rate
        else:
            raw, raw_t = network_value_and_time_derivative(self.mlp, self.cfg, k, t)
            tau = t/self.cfg.t_max
            phase = tau*self.phase_scale*raw
            slope = self.phase_scale*(raw/self.cfg.t_max+tau*raw_t)
        p = torch.exp(1j*phase)
        return p, 1j*slope*p

    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

def validate_cartesian(cfg, k, t):
    k, t = paired(k, t)
    if len(t) % cfg.mode_count:
        raise ValueError('Full retained mode grid required')
    nt = len(t)//cfg.mode_count
    expected = torch.as_tensor(cfg.wave_numbers, dtype=torch.float64, device=k.device)
    if not torch.equal(k.detach().reshape(nt, cfg.mode_count), expected.expand(nt, -1)):
        raise ValueError('Ordered full retained modes required')
    times = t.detach().reshape(nt, cfg.mode_count)
    if not torch.equal(times, times[:, :1].expand_as(times)):
        raise ValueError('One common time per full modal block required')
    return nt

def total_norm_parameterization(raw, raw_t=None, *, mode_count):
    """Radial projection on the entire complex spectrum; no modal normalization."""
    shaped = raw.reshape(-1, mode_count, 2)
    square = shaped.abs().square().sum((1,2), keepdim=True)
    if not bool(torch.isfinite(square).all()) or bool((square <= 0).any()):
        raise FloatingPointError('Undefined total-norm parameterization at zero/nonfinite raw spectrum')
    radius = torch.sqrt(square)
    projected = shaped/radius
    if raw_t is None:
        return projected.reshape_as(raw)
    tangent = raw_t.reshape_as(shaped)
    radius_t = (shaped.conj()*tangent).real.sum((1,2), keepdim=True)/radius
    projected_t = tangent/radius-shaped*radius_t/radius.square()
    return projected.reshape_as(raw), projected_t.reshape_as(raw_t)

class R13Model(R9Model):
    def __init__(self, cfg, carrier_kind, envelope_kind, seed):
        if carrier_kind not in CARRIER_KINDS+('exact',) or envelope_kind not in ENVELOPE_KINDS:
            raise ValueError((carrier_kind, envelope_kind))
        super().__init__(cfg, 'B_complex', seed)
        self.carrier_kind, self.envelope_kind = carrier_kind, envelope_kind
        if carrier_kind == 'exact':
            from .r11_exact_carrier_diagnostic import ExactCarrierDiagnostic
            self.carrier = ExactCarrierDiagnostic(cfg)
        else:
            self.carrier = LearnedRecoveryCarrier(cfg, carrier_kind, seed)
        self.model_name = f'{carrier_kind}__{envelope_kind}'

    def branch_coefficients(self, k, t):
        k, t = paired(k, t)
        output = self.networks[0](features(self.cfg, k, t))
        z = torch.complex(output[:, :2], output[:, 2:])
        raw = self.initial_coeff[self.lookup(k)]+t/self.cfg.t_max*z
        if self.envelope_kind == 'hard_total_norm':
            validate_cartesian(self.cfg, k, t)
            raw = total_norm_parameterization(raw, mode_count=self.cfg.mode_count)
        return self.carrier(k, t)*raw

def hashes(model):
    return dict(envelopes=[tensor_digest(n.parameters()) for n in model.networks],
        carrier=None if model.carrier_kind == 'exact' else tensor_digest(model.carrier.parameters()))

def coupled_parameters(model):
    return tuple(model.networks.parameters())
