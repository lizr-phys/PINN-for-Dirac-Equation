import mpmath as mp
import pytest
from scattering_dirac_kg.precision_reference import taylor_jost, mp_amplitude
from scattering_dirac_kg.reference import jost


@pytest.mark.parametrize("kind", ["dirac", "kg"])
def test_arbitrary_precision_free_phase_and_flux(kind):
    r = taylor_jost(kind, 1.2, length=4., degree=32, digits=60, depth=0.)
    with mp.workdps(60):
        assert abs(mp_amplitude(r, "r"))<mp.mpf('1e-30')
        assert abs(mp_amplitude(r, "t")-1)<mp.mpf('1e-30')


@pytest.mark.parametrize("kind", ["dirac", "kg"])
def test_arbitrary_precision_independent_gaussian_amplitude(kind):
    r = taylor_jost(kind, 1.02, length=20., step=.5, degree=28, digits=50)
    ref = jost(kind, 1.02, length=20.)
    assert abs(complex(mp_amplitude(r, "r"))-ref.r)<2e-12
    assert abs(complex(mp_amplitude(r, "t"))-ref.t)<2e-12
