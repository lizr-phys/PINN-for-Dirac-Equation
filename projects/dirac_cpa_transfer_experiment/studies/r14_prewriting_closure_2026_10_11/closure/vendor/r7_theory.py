from __future__ import annotations

import numpy as np
import torch

from .freeze_r7_phase_release import write_json
from .periodic_potential_evaluate import _initial_tangent_diagnostics
from .periodic_potential_models import count_trainable_parameters, trainable_state_sha256
from .periodic_potential_physics import central_initial_state
from .r7_config import R7Config, MODELS
from .r7_models import build_model
from .r7_evaluate import early_diagnostics


def validate_theory(destination):
    torch.set_num_threads(1)
    cfg = R7Config()
    rows, gates = [], []
    for seed in range(3):
        hashes = []
        for name in MODELS:
            model = build_model(name, cfg, 500000+seed)
            k = torch.tensor(cfg.wave_numbers, dtype=torch.float64)
            with torch.no_grad():
                initial = model(k, torch.zeros_like(k)).numpy()
            initial_error = float(np.max(np.abs(initial-central_initial_state(cfg))))
            tangent, arrays = _initial_tangent_diagnostics(model, cfg)
            coordinate, _, _ = early_diagnostics(model, cfg)
            hashes.append(trainable_state_sha256(model))
            original = name == "original_ltpb"
            sensitivity = tangent["initial_sideband_imaginary_tangent_parameter_sensitivity"]
            gates.append(initial_error < 1e-14 and count_trainable_parameters(model) == 12932)
            gates.append(sensitivity == 0.0 if original else sensitivity > 0.0)
            rows.append(dict(seed=seed, model=name, initial_condition_max_abs_error=initial_error,
                             parameter_count=12932, **tangent, **coordinate))
        gates.append(len(set(hashes)) == 1)
    result = dict(status="PASS" if all(gates) else "FAIL", diagnostic_not_training=True,
                  initial_weights_tested_seeds=[0, 1, 2], records=rows,
                  interpretation="Release permits the complex initial tangent when r is nonzero; it does not certify learned dynamics or avoid r=0 degeneracy.")
    write_json(destination/"theory_validation.json", result)
    return result


if __name__ == "__main__":
    from .freeze_r7_phase_release import RAW
    print(validate_theory(RAW)["status"])
