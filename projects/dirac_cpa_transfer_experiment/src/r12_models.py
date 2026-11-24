from __future__ import annotations
from .r12_config import GROUPS
from .r11_models import R11Model, hashes, parameter_counts, coupled_parameters

class R12Model(R11Model):
    """R11 A/C/D/E with condition-specific operator, initial basis and time scaling."""
    def __init__(self,cfg,group,seed):
        if group not in GROUPS:
            raise ValueError(group)
        super().__init__(cfg,group,seed)
