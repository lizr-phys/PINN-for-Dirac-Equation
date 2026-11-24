import numpy as np
import pytest
import torch
from scattering_dirac_kg.mlp import MLP


@pytest.mark.parametrize("kind", ["dirac", "kg"])
def test_zero_field_is_penalized_by_incident_source(kind):
    torch.set_default_dtype(torch.float64)
    model=MLP(kind,1.2,width=4)
    for parameter in model.parameters():
        with torch.no_grad():
            parameter.zero_()
    error,left,right=model.boundary()
    assert np.allclose(error.detach().numpy(),[-1,0],atol=1e-15)
    assert float(torch.sum(abs(error)**2).detach())==1.
