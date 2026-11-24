from __future__ import annotations
from .r9_models import R9Model, parameter_counts, tensor_digest
from .r11_config import GROUPS

class R11Model(R9Model):
    """Unchanged R9 shapes and hard initial state; fresh seed initialization."""
    def __init__(self, cfg, group, seed):
        if group not in GROUPS:
            raise ValueError(group)
        parent = "C_double_complex" if group == GROUPS[3] else (
            "B_complex" if group == GROUPS[4] else "A_carrier_complex")
        super().__init__(cfg, parent, seed)
        self.model_name = group
        if group == GROUPS[1]:
            self.carrier.freeze()
        elif group == GROUPS[4]:
            from .r11_exact_carrier_diagnostic import ExactCarrierDiagnostic
            self.carrier = ExactCarrierDiagnostic(cfg)

def coupled_parameters(model):
    if model.model_name == GROUPS[2]:
        return tuple(model.parameters())
    return tuple(model.networks.parameters())

def hashes(model):
    return dict(envelopes=[tensor_digest(n.parameters()) for n in model.networks],
        carrier=None if model.carrier is None or getattr(model.carrier, "analytic_prior", False)
        else tensor_digest(model.carrier.parameters()))
