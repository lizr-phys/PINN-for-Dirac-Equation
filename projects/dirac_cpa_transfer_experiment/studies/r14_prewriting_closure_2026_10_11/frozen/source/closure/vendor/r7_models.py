from __future__ import annotations

import torch

from .periodic_potential_models import (
    CoupledSharedSpectralComplex, CoupledSharedSpectralLTPB, _paired_columns,
)
from .r7_config import MODELS


class R7Complex(CoupledSharedSpectralComplex):
    model_name = "complex"


class R7OriginalLTPB(CoupledSharedSpectralLTPB):
    model_name = "original_ltpb"


class R7PhaseReleaseLTPB(CoupledSharedSpectralLTPB):
    """Release only the phase of channels empty in the known initial state.

    The support mask never depends on a prediction or a future occupation.
    All occupied channels retain the original hard phase gate exactly.
    """
    model_name = "phase_release_ltpb"

    def branch_fields(self, wave_numbers, t):
        k, time = _paired_columns(wave_numbers, t)
        output = self.mlp(self.features(k, time))
        tau = time / float(self.cfg.t_max)
        initial = self.initial_branch_coefficients(k)
        occupied_initially = initial != 0.0
        amplitude = initial + tau * output[:, :2]
        phase = torch.where(occupied_initially, tau * output[:, 2:], output[:, 2:])
        return amplitude, phase


def build_model(name, cfg, initialization_seed):
    constructors = dict(zip(MODELS, (R7Complex, R7OriginalLTPB, R7PhaseReleaseLTPB)))
    if name not in constructors:
        raise ValueError(name)
    return constructors[name](cfg, initialization_seed)
