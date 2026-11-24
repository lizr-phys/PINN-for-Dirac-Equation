import importlib
import numpy as np
import pytest
from scattering_dirac_kg.model import ScatteringNetwork
from scattering_dirac_kg.physics import CASE, current, free_matrix


@pytest.mark.parametrize("kind", ["dirac", "kg"])
def test_physics_only_training_free_problem_and_save_roundtrip(kind, tmp_path, monkeypatch):
    module = importlib.import_module("scattering_dirac_kg.train")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    directory = module.train(kind, 1.2, 2, "free_test", features=12,
                            collocation=80, refinements=0, case={**CASE, "depth":0.})
    net = ScatteringNetwork.load(directory/"model.npz")
    r, t = net.amplitudes(1.2)
    assert abs(r)<1e-10 and abs(t-1)<1e-10
    x = np.linspace(-16,16,1001)
    state, residual = net.state_and_residual(x, 1.2)
    reference = np.array([free_matrix(kind,1.2,xx)[:,0] for xx in x])
    assert np.max(abs(state-reference))<1e-10
    assert np.max(abs(residual))<1e-10
    assert np.max(abs(current(kind,state)-1))<1e-10
