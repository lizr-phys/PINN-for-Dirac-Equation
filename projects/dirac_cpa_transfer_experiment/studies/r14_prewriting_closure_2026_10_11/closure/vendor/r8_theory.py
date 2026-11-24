from __future__ import annotations
import numpy as np
import torch
from .r8_config import R8Config, CANDIDATES, CONTROLS, RAW
from .r8_models import R8Model
from .r8_loss import derivative
from .r8_reference import initial_numpy
from .freeze_r7_phase_release import write_json


def validate_theory():
    torch.set_num_threads(1)
    cfg = R8Config()
    k = torch.tensor(cfg.wave_numbers, dtype=torch.float64).reshape(-1, 1)
    rows = []
    for name in CANDIDATES+CONTROLS:
        model = R8Model(cfg, name, 0)
        if model.carrier is not None:
            model.carrier.freeze()
        t = torch.zeros_like(k, requires_grad=True)
        state = model(k, t)
        dt = derivative(state, t)
        grads = torch.autograd.grad(dt[[5, 7]].imag.sum(), tuple(model.fields[0].parameters()), allow_unused=True)
        sensitivity = np.sqrt(sum(float((g*g).sum()) for g in grads if g is not None))
        row = dict(model=name, hard_initial_max_abs=float(np.max(abs(state.detach().numpy()-initial_numpy(cfg)))),
                   empty_sideband_imaginary_tangent_sensitivity=float(sensitivity),
                   derivative_finite=bool(torch.isfinite(dt).all()), carrier_time_graph_retained=model.carrier is not None)
        row["pass_check"] = row["hard_initial_max_abs"] < 1e-15 and sensitivity > 0 and row["derivative_finite"]
        rows.append(row)
    result = dict(status="PASS" if all(r["pass_check"] for r in rows) else "FAIL", checks=rows,
                  scope="Implementation and available local tangent, not complete coupled dynamics",
                  derivative_supervision_used=False)
    write_json(RAW/"theory_validation.json", result)
    return result


if __name__ == "__main__":
    print(validate_theory())
