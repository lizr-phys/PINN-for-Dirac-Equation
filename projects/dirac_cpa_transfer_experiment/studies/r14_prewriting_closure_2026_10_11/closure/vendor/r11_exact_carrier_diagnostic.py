from __future__ import annotations
import torch
from torch import nn
from .r8_models import paired

class ExactCarrierDiagnostic(nn.Module):
    """Explicit analytic-prior exception, instantiated only for R11 group E."""
    analytic_prior = True
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def value_and_time_derivative(self, k, t):
        k, t = paired(k, t)
        energy = torch.sqrt(k.square()+self.cfg.mass**2)
        rate = torch.cat((-energy, energy), 1)
        p = torch.exp(1j*t*rate)
        return p, 1j*rate*p

    def forward(self, k, t):
        return self.value_and_time_derivative(k, t)[0]

    def freeze(self):
        return self
