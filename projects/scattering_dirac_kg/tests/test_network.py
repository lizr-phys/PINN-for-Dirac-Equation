import ast
from pathlib import Path
import numpy as np
import pytest
import torch
from scattering_dirac_kg.model import ScatteringNetwork
from scattering_dirac_kg.physics import ROOT, free_matrix, current
from scattering_dirac_kg.train import train, training_energies


def complex_grad(y, x):
    return torch.autograd.grad(y.real.sum(), x, create_graph=True)[0]+1j*torch.autograd.grad(y.imag.sum(), x, create_graph=True)[0]


@pytest.mark.parametrize("kind", ["dirac", "kg"])
@pytest.mark.parametrize("energy", [1.02, 2.])
def test_hard_incident_and_outgoing_channels_for_arbitrary_coefficients(kind, energy):
    net = ScatteringNetwork(kind, features=12)
    rng = np.random.default_rng(5)
    net.coefficients = rng.normal(size=net.coefficients.shape)+1j*rng.normal(size=net.coefficients.shape)
    y, _ = net.state_and_residual(np.array([-16., 16.]), energy)
    left = np.linalg.solve(free_matrix(kind, energy, -16.), y[0])
    right = np.linalg.solve(free_matrix(kind, energy, 16.), y[1])
    r, t = net.amplitudes(energy)
    assert abs(left[0]-1)<1e-12 and abs(right[1])<1e-12
    assert abs(left[1]-r)<1e-12 and abs(right[0]-t)<1e-12


@pytest.mark.parametrize("kind", ["dirac", "kg"])
def test_analytic_network_derivatives_against_independent_autodiff(kind):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    net = ScatteringNetwork(kind, features=8)
    rng = np.random.default_rng(2)
    net.coefficients = rng.normal(size=net.coefficients.shape)+1j*rng.normal(size=net.coefficients.shape)
    x = torch.linspace(-15.3, 15.1, 93, requires_grad=True)
    y = net.torch_field(x, 1.2)
    dy = complex_grad(y, x)
    ddy = complex_grad(dy, x)
    expected = net.field(x.detach().numpy(), 1.2)
    for true, ad in zip(expected, [y, dy, ddy]):
        assert np.allclose(true, ad.detach().numpy(), atol=2e-12, rtol=2e-12)


def test_dirac_reconstruction_original_equations_not_real_bound_state_rule():
    net = ScatteringNetwork("dirac", features=8)
    state, residual = net.state_and_residual(np.linspace(-16, 16, 41), 1.2)
    assert np.max(abs(residual[:, 1]))<1e-15
    assert np.ptp(current("dirac", state))>1e-2  # unit incident is not the interacting solution
    assert np.max(abs(state.imag))>.1


def test_training_modules_have_no_reference_dependency():
    for filename in ("model.py", "train.py"):
        parsed = ast.parse((ROOT/filename).read_text(encoding="utf-8"))
        imports = [getattr(n, "module", "") or "" for n in ast.walk(parsed) if isinstance(n, ast.ImportFrom)]
        assert not any("reference" in name or "evaluate" in name for name in imports)
    assert len(training_energies(25))==25
    assert abs(training_energies(25)[0]-1.01)<1e-14
    assert abs(training_energies(25)[-1]-2.)<1e-14
