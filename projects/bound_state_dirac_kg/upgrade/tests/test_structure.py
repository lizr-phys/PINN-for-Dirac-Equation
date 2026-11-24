import ast
from pathlib import Path
import numpy as np
import pytest
import torch
from bound_state_dirac_kg.upgrade.physics import config
from bound_state_dirac_kg.upgrade.model import StructuredPINN,grad,gauss_grid

@pytest.fixture(autouse=True)
def double_precision():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)

@pytest.mark.parametrize("kind",["dirac","kg"])
@pytest.mark.parametrize("state",[0,1,2])
def test_hard_constraints_random_network(kind,state):
    cfg=config();case=cfg["cases"]["deep_wide"]
    torch.manual_seed(170+state)
    model=StructuredPINN(kind,cfg,case,state,initial_fraction=.6)
    x=torch.linspace(-case["half_length"],case["half_length"],1001)[:,None].requires_grad_(True)
    y=model(x)
    torch.testing.assert_close(y[:,0],(-1)**state*torch.flip(y[:,0],[0]),rtol=1e-12,atol=1e-14)
    if kind=="dirac":
        torch.testing.assert_close(y[:,1],(-1)**(state+1)*torch.flip(y[:,1],[0]),rtol=1e-12,atol=1e-14)
        assert model.residual(x)[:,0].abs().max()<1e-13
    q,w=gauss_grid(512,case["half_length"])
    nq=2*torch.sum(w*model(q)**2)
    assert abs(float(nq.detach())-1)<1e-10
    assert float(model.boundary_residual().abs().max().detach())<1e-12
    xb=torch.tensor([[case["half_length"]]],requires_grad=True)
    rb=model.residual(xb)
    tail=model(xb)[:,0].abs()
    assert float((rb.abs()/tail).max().detach())<5e-12

def test_dirac_sign_and_second_order_elimination():
    cfg=config();case=cfg["cases"]["base"]
    model=StructuredPINN("dirac",cfg,case,initial_fraction=.65)
    x=torch.linspace(-7,7,81)[:,None].requires_grad_(True)
    y=model(x);u=y[:,:1]
    A=model.energy-model.potential(x)+model.mass
    vp=grad(model.potential(x),x)
    second=grad(grad(u,x),x)+vp/A*grad(u,x)+((model.energy-model.potential(x))**2-model.mass**2)*u
    torch.testing.assert_close(second,-A*model.residual(x)[:,1:2],rtol=2e-11,atol=1e-13)

def test_training_has_no_reference_labels():
    source=(Path(__file__).parents[1]/"train.py").read_text(encoding="utf-8")
    tree=ast.parse(source)
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert "reference" not in imports
    assert "reference_on" not in source
    assert "reference_energy" not in source
    assert "shooting" not in source

def test_whole_positive_gap_and_trainable_nodes():
    cfg=config();case=cfg["cases"]["deep_wide"]
    for fraction in [.1,.95]:
        model=StructuredPINN("kg",cfg,case,state=2,initial_fraction=fraction)
        assert abs(float(model.energy.detach())-fraction)<1e-14
        assert 0<float(model.node.detach())<case["half_length"]/2
        x=torch.tensor([[.4]],requires_grad=True)
        loss=model(x).square().sum()
        loss.backward()
        assert abs(float(model.raw_node.grad))>1e-8
