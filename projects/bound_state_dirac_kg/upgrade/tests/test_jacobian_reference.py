import numpy as np
import pytest
import torch
from torch.nn.utils import parameters_to_vector,vector_to_parameters
from bound_state_dirac_kg.upgrade.model import StructuredPINN,StructuredLoss
from bound_state_dirac_kg.upgrade.optimizer import physics_vector,residual_jacobian
from bound_state_dirac_kg.upgrade.physics import ROOT,config,reference_on,potential

@pytest.mark.parametrize("kind",["dirac","kg"])
def test_residual_jacobian_against_directional_difference(kind):
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.manual_seed(9)
    cfg={**config(),"hidden_width":6,"hidden_layers":1,"collocation_points":12,"quadrature_points":24}
    model=StructuredPINN(kind,cfg,cfg["cases"]["base"],initial_fraction=.61)
    loss=StructuredLoss(model,cfg)
    params=list(model.parameters())
    theta=parameters_to_vector(params).detach().clone()
    direction=torch.randn_like(theta);direction/=torch.linalg.norm(direction)
    vector=physics_vector(loss,"derivative")
    J=residual_jacobian(vector,params,batch_size=4)
    h=1e-5
    vector_to_parameters(theta+h*direction,params)
    plus=physics_vector(loss,"derivative").detach()
    vector_to_parameters(theta-h*direction,params)
    minus=physics_vector(loss,"derivative").detach()
    vector_to_parameters(theta,params)
    torch.testing.assert_close(J@direction,(plus-minus)/(2*h),rtol=2e-5,atol=1e-8)

def test_saved_reference_layout_and_kg_charge_orthogonality():
    directory=ROOT/"results"/"reference"/"deep_wide"
    if not (directory/"summary.json").exists(): pytest.skip("Deep reference not generated yet")
    cfg=config();case=cfg["cases"]["deep_wide"]
    x=np.linspace(-case["half_length"],case["half_length"],16001)
    p0,e0=reference_on(directory,"kg",0,x)
    p2,e2=reference_on(directory,"kg",2,x)
    ordinary=np.trapezoid(p0[:,0]*p2[:,0],x)
    weighted=np.trapezoid((e0+e2-2*potential(x,case))*p0[:,0]*p2[:,0],x)
    assert abs(ordinary)>1e-3  # same parity; ordinary L2 orthogonality is wrong here
    assert abs(weighted)<1e-9
    for kind in ["dirac","kg"]:
        for state in [0,1,2]:
            data=np.load(directory/f"{kind}_state_{state}.npz")
            y,e=reference_on(directory,kind,state,data["x"])
            np.testing.assert_allclose(y,data["y"],atol=2e-14)
            assert e==float(data["energy"])
