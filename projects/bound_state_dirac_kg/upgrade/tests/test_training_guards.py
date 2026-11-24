import pytest
import torch
from torch import nn
from bound_state_dirac_kg.upgrade.training_core import checked_energy,NumericalTrialError
from bound_state_dirac_kg.upgrade.optimizer import polish

class Trial(nn.Module):
    mass=1.
    def __init__(self,value):
        super().__init__()
        self.raw_energy=nn.Parameter(torch.tensor(value,dtype=torch.float64))
    @property
    def energy(self):return torch.sigmoid(self.raw_energy)

@pytest.mark.parametrize("value",[1000.,-1000.,float("nan")])
def test_guard_rejects_unrepresented_gap_energy(value):
    with pytest.raises(NumericalTrialError):checked_energy(Trial(value))

def test_gauss_newton_rejects_trial_and_restores_before_retry():
    model=Trial(0.)
    failed=[]
    def vector():
        if float(model.raw_energy.detach())>.1:
            failed.append(float(model.raw_energy.detach()))
            raise NumericalTrialError("Synthetic invalid trial")
        return (model.raw_energy-.2).reshape(1)
    result=polish(model,vector,max_steps=1)
    assert failed
    assert torch.isfinite(model.raw_energy)
    assert float(model.raw_energy.detach())<=.1
    assert result["accepted_steps"]==1
    assert float(torch.sum(vector().detach()**2))<.04
