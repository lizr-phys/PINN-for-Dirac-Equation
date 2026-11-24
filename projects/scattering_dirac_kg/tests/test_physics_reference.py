import numpy as np
import pytest
from scattering_dirac_kg.physics import CASE, free_matrix, current, ode_matrix, relative_l2
from scattering_dirac_kg.reference import jost, boundary_value, square_jost, square_analytic


@pytest.mark.parametrize("kind", ["dirac", "kg"])
@pytest.mark.parametrize("energy", [1.02, 1.2, 1.5, 2.])
def test_free_channels_and_global_transmission_phase(kind, energy):
    mat = free_matrix(kind, energy, .37)
    assert np.allclose(current(kind, mat.T), [1., -1.], atol=2e-15)
    sol = jost(kind, energy, potential_fn=lambda x: 0.)
    assert abs(sol.r)<3e-11
    assert abs(sol.t-1)<3e-11
    x = np.linspace(-16, 16, 301)
    true = np.array([free_matrix(kind, energy, xx)[:, 0] for xx in x])
    assert relative_l2(x, sol.value(x), true)<3e-11


@pytest.mark.parametrize("kind", ["dirac", "kg"])
@pytest.mark.parametrize("energy", [1.02, 1.2, 1.5, 2.])
def test_square_well_complex_amplitude_and_flux(kind, energy):
    r, t = square_analytic(kind, energy)
    sol = square_jost(kind, energy)
    assert abs(sol.r-r)<3e-11
    assert abs(sol.t-t)<3e-11
    assert abs(abs(r)**2+abs(t)**2-1)<3e-15


@pytest.mark.parametrize("kind", ["dirac", "kg"])
def test_jost_bvp_independent_wave_and_amplitudes(kind):
    a, b = jost(kind, 1.02), boundary_value(kind, 1.02, tol=2e-10)
    x = np.linspace(-16, 16, 1201)
    assert abs(a.r-b.r)<1e-9 and abs(a.t-b.t)<1e-9
    assert relative_l2(x, a.value(x), b.value(x))<1e-9
    assert np.ptp(current(kind, a.value(x)))<1e-10


def test_dirac_ode_complex_sign_and_mass_terms():
    e, v = 1.2, -.4
    mat = ode_matrix("dirac", e, v)
    assert mat[0, 1] == 1j*(e-v+1)
    assert mat[1, 0] == 1j*(e-v-1)
    assert np.allclose(mat@mat, -((e-v)**2-1)*np.eye(2))


def test_models_are_not_assumed_identical():
    assert abs(jost("dirac", 1.02).R-jost("kg", 1.02).R)>1e-3


def test_no_energy_eigenvalue_training_or_closed_channel():
    with pytest.raises(ValueError):
        free_matrix("dirac", .9, 0.)
