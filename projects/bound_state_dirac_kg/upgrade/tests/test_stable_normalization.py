"""Regression for overflow-induced zero waves and exact numerical scale invariance."""
import json
import numpy as np
import pytest
import torch
from bound_state_dirac_kg.upgrade.model import StructuredPINN
from bound_state_dirac_kg.upgrade.physics import ROOT,config
from bound_state_dirac_kg.upgrade.training_core import checked_representation,NumericalTrialError

@pytest.mark.parametrize("kind",["dirac","kg"])
@pytest.mark.parametrize("state",[0,1,2])
def test_global_scaling_preserves_wave_residual_and_parameter_derivative(kind,state):
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.manual_seed(17)
    cfg={**config(),"hidden_width":6,"hidden_layers":1,"quadrature_points":64}
    old=StructuredPINN(kind,{**cfg,"version":6},cfg["cases"]["deep_wide"],state,initial_fraction=.61)
    stable=StructuredPINN(kind,{**cfg,"version":7},cfg["cases"]["deep_wide"],state,initial_fraction=.61)
    stable.load_state_dict(old.state_dict())
    x=torch.linspace(-8,8,41)[:,None].requires_grad_(True)
    y0,y1=old(x),stable(x)
    torch.testing.assert_close(y0,y1,rtol=1e-11,atol=2e-14)
    torch.testing.assert_close(old.residual(x),stable.residual(x),rtol=1e-10,atol=2e-13)
    f0=torch.sum(y0*y0*torch.cos(x));f1=torch.sum(y1*y1*torch.cos(x))
    g0=torch.autograd.grad(f0,list(old.parameters()))
    g1=torch.autograd.grad(f1,list(stable.parameters()))
    for a,b in zip(g0,g1):torch.testing.assert_close(a,b,rtol=1e-9,atol=2e-12)
    checked_representation(stable)

def test_rejects_actual_v6_overflow_zero_wave_regression():
    out=ROOT/"results/stable_formal/deep_wide/dirac/state_2/derivative/seed_0"
    if not (out/"model.pt").exists():pytest.skip("Archived v6 failure is not present")
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    meta=json.loads((out/"config.json").read_text(encoding="utf-8"))
    old=StructuredPINN("dirac",meta["config"],meta["case"],2)
    old.load_state_dict(torch.load(out/"model.pt",weights_only=True))
    assert torch.isinf(old.normalizer())
    assert torch.sum(old(old.q).detach()**2)==0
    with pytest.raises(NumericalTrialError,match="quadrature norm"):
        checked_representation(old)
