from __future__ import annotations

"""R8 A is retained; only the registered Fourier grid and hard initial data change."""
import math
import hashlib
import torch
from torch import nn
from .wavepacket_models import PairedTanhMLP
from .r8_models import LearnedCarrier, basis, paired, features, compose


def original_gaussian():
    k = torch.arange(33, dtype=torch.float64)/8
    raw = torch.exp(-(k-2)**2/(4*.5**2))
    # This is the R5 normalization on exactly n=0,...,32, once only.
    return torch.rsqrt(torch.sum(raw**2))*raw


def initial_coefficients(cfg):
    g = torch.zeros(cfg.mode_count, dtype=torch.float64)
    g[-cfg.n_min:33-cfg.n_min] = original_gaussian()
    return g[:, None].expand(-1, 2).to(torch.complex128)/math.sqrt(2)


def tensor_digest(values):
    digest = hashlib.sha256()
    for value in values:
        a = value.detach().cpu().contiguous().numpy()
        digest.update(str(a.shape).encode())
        digest.update(str(a.dtype).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


class R9Model(nn.Module):
    def __init__(self, cfg, name, seed):
        super().__init__()
        if name not in ("A_carrier_complex", "B_complex", "C_double_complex"):
            raise ValueError(name)
        self.cfg, self.model_name, self.seed = cfg, name, seed
        self.carrier = LearnedCarrier(cfg, seed) if name == "A_carrier_complex" else None
        self.networks = nn.ModuleList([PairedTanhMLP(2, 4, cfg.hidden_width, cfg.hidden_layers,
                                     initialization_seed=800000+seed)])
        if name == "C_double_complex":
            self.networks.append(PairedTanhMLP(2, 4, cfg.hidden_width, cfg.hidden_layers,
                                              initialization_seed=830000+seed))
        self.register_buffer("initial_coeff", initial_coefficients(cfg))

    def lookup(self, k):
        n = torch.round(8*k.reshape(-1)).long()
        if torch.any(n < self.cfg.n_min) or torch.any(n > self.cfg.n_max):
            raise ValueError("Only registered integer Fourier indices are allowed")
        if not torch.allclose(k.reshape(-1), n.to(k.dtype)/8, atol=1e-14, rtol=0):
            raise ValueError("Non-Fourier wave number")
        return n-self.cfg.n_min

    def branch_coefficients(self, k, t):
        k, t = paired(k, t)
        output = self.networks[0](features(self.cfg, k, t))
        if len(self.networks) == 2:
            output = (output+self.networks[1](features(self.cfg, k, t)))/math.sqrt(2)
        z = torch.complex(output[:, :2], output[:, 2:])
        c = self.initial_coeff[self.lookup(k)]+(t/self.cfg.t_max)*z
        # Exactly R8 A: p(c0+tau*z). No detach or no_grad on the carrier.
        return c if self.carrier is None else self.carrier(k, t)*c

    def forward(self, k, t):
        k, t = paired(k, t)
        return compose(self.branch_coefficients(k, t), k, self.cfg.mass)


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total_parameters=total, active_parameters=active, frozen_parameters=total-active,
                carrier_parameters=0 if model.carrier is None else sum(p.numel() for p in model.carrier.parameters()),
                complex_network_parameters=sum(p.numel() for p in model.networks.parameters()))
