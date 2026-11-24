from __future__ import annotations

"""Learned representation only: no reference evolution or exact frequency."""

import math
import torch
from torch import nn
from .wavepacket_models import PairedTanhMLP
from .r8_config import has_carrier, windowed


def paired(k, t):
    k, t = k.reshape(-1, 1), t.reshape(-1, 1)
    if k.numel() == 1:
        k = k.expand_as(t)
    elif t.numel() == 1:
        t = t.expand_as(k)
    if k.shape != t.shape:
        raise ValueError("Paired k,t required")
    return k, t


def basis(k, mass):
    angle = torch.atan2(k.reshape(-1), torch.full_like(k.reshape(-1), float(mass)))/2
    return torch.stack((torch.cos(angle), torch.sin(angle)), -1), torch.stack((-torch.sin(angle), torch.cos(angle)), -1)


def features(cfg, k, t):
    scale = cfg.ladder_half_width*cfg.coupling_wave_number
    return torch.cat(((k-cfg.k0)/scale, t/cfg.t_max), dim=1)


def initial_coefficients(cfg):
    k = torch.tensor(cfg.wave_numbers, dtype=torch.float64)
    if cfg.case == "free_packet":
        g = torch.exp(-(k-cfg.k0)**2/(4*.5**2))
        g = g/torch.linalg.vector_norm(g)
    else:
        g = (k == cfg.k0).to(torch.float64)
    return g[:, None].expand(-1, 2).to(torch.complex128)/math.sqrt(2)


def compose(coeff, k, mass):
    up, um = basis(k, mass)
    return coeff[:, :1]*up + coeff[:, 1:]*um


class LearnedCarrier(nn.Module):
    """Unit-modulus free carriers learned from two separate unit spinor IVPs."""
    def __init__(self, cfg, seed):
        super().__init__()
        self.cfg = cfg
        self.mlp = PairedTanhMLP(2, 2, cfg.hidden_width, cfg.hidden_layers,
                                initialization_seed=900000+seed)

    def forward(self, k, t):
        k, t = paired(k, t)
        return torch.exp(1j*(t/self.cfg.t_max)*self.mlp(features(self.cfg, k, t)))

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self


class LocalField(nn.Module):
    def __init__(self, cfg, seed, kind, initial, start=0., duration=None):
        super().__init__()
        self.cfg, self.kind, self.start = cfg, kind, float(start)
        self.duration = float(duration or cfg.t_max)
        self.mlp = PairedTanhMLP(2, 4, cfg.hidden_width, cfg.hidden_layers,
                                initialization_seed=800000+seed)
        self.register_buffer("initial_coeff", initial.detach().clone().to(torch.complex128))
        # Exact support at the predicted handoff, not a threshold switch.
        self.register_buffer("initial_occupied", self.initial_coeff.abs() != 0)
        self.register_buffer("initial_modulus", self.initial_coeff.abs())
        self.register_buffer("initial_argument", torch.angle(self.initial_coeff))

    def lookup(self, k):
        indices = torch.round((k.reshape(-1)-self.cfg.wave_numbers[0])/self.cfg.coupling_wave_number).long()
        if torch.any(indices < 0) or torch.any(indices >= self.cfg.mode_count):
            raise ValueError("Local hard initial state is defined on the registered grid")
        return indices

    def branch_coefficients(self, k, t, carrier=None):
        k, t = paired(k, t)
        local = (t-self.start)/self.duration
        # Window inputs use local normalized time; k normalization is unchanged.
        output = self.mlp(torch.cat((features(self.cfg, k, t)[:, :1], local), dim=1))
        index = self.lookup(k)
        c0 = self.initial_coeff[index]
        delta = torch.complex(output[:, :2], output[:, 2:])
        if self.kind == "complex":
            return c0+local*delta
        if self.kind in ("carrier", "carrier_identity"):
            if carrier is None:
                raise ValueError("Learned frozen carrier required")
            p = carrier(k, t) if self.kind == "carrier" else torch.ones_like(c0)
            # No detach/no_grad: frozen parameters retain the full time derivative.
            p0 = carrier(k, torch.full_like(t, self.start)) if self.kind == "carrier" else torch.ones_like(c0)
            return (p/p0)*(c0+local*delta)
        if self.kind == "release":
            amp = self.initial_modulus[index]+local*output[:, :2]
            phase_delta = torch.where(self.initial_occupied[index], local*output[:, 2:], output[:, 2:])
            phase = self.initial_argument[index]+phase_delta
            return amp*torch.exp(1j*phase)
        if self.kind in ("correction", "static_correction"):
            free = carrier(k, t)*c0 if self.kind == "correction" else c0
            return free, self.cfg.potential_strength*(t-self.start)*delta
        raise ValueError(self.kind)

    def forward(self, k, t, carrier=None):
        k, t = paired(k, t)
        result = self.branch_coefficients(k, t, carrier)
        if self.kind in ("correction", "static_correction"):
            free, chi = result
            # chi is a Cartesian two-component complex spinor, not branch coefficients.
            return compose(free, k, self.cfg.mass)+chi
        return compose(result, k, self.cfg.mass)


class R8Model(nn.Module):
    def __init__(self, cfg, name, seed):
        super().__init__()
        self.cfg, self.model_name, self.seed = cfg, name, seed
        self.carrier = LearnedCarrier(cfg, seed) if has_carrier(name) else None
        kinds = {"complex": "complex", "phase_release": "release",
                 "A_carrier_complex": "carrier", "B_free_correction": "correction",
                 "C_release_windows": "release", "D_carrier_windows": "carrier",
                 "window_complex": "complex", "A_identity_carrier": "carrier_identity",
                 "D_identity_carrier": "carrier_identity", "B_static_free": "static_correction"}
        self.kind = kinds[name]
        self.fields = nn.ModuleList([LocalField(cfg, seed, self.kind, initial_coefficients(cfg),
                                               duration=cfg.t_max/cfg.windows if windowed(name) else cfg.t_max)])

    def add_window(self):
        previous = self.fields[-1]
        end = previous.start+previous.duration
        k = torch.tensor(self.cfg.wave_numbers, dtype=torch.float64).reshape(-1, 1)
        with torch.no_grad():
            state = previous(k, torch.full_like(k, end), self.carrier)
            up, um = basis(k, self.cfg.mass)
            coeff = torch.stack(((state*up).sum(-1), (state*um).sum(-1)), -1)
        for p in previous.parameters():
            p.requires_grad_(False)
        self.fields.append(LocalField(self.cfg, self.seed+10000*len(self.fields), self.kind, coeff,
                                      start=end, duration=self.cfg.t_max/self.cfg.windows))
        return coeff

    @property
    def active_field(self):
        return self.fields[-1]

    def forward(self, k, t):
        k, t = paired(k, t)
        if len(self.fields) == 1:
            return self.fields[0](k, t, self.carrier)
        # t only chooses a piece; its un-detached value still differentiates the field.
        idx = torch.clamp(torch.floor(t.detach().reshape(-1)/(self.cfg.t_max/self.cfg.windows)).long(),
                          0, len(self.fields)-1)
        result = torch.zeros((len(t), 2), dtype=torch.complex128, device=t.device)
        for window, field in enumerate(self.fields):
            mask = idx == window
            if mask.any():
                result[mask] = field(k[mask], t[mask], self.carrier)
        return result


def parameter_counts(model):
    return {"parameter_count": sum(p.numel() for p in model.parameters()),
            "active_trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "carrier_parameter_count": 0 if model.carrier is None else sum(p.numel() for p in model.carrier.parameters()),
            "envelope_parameter_count_per_window": sum(p.numel() for p in model.fields[0].parameters()),
            "stored_window_network_count": len(model.fields)}
