import numpy as np
import pytest
import torch
from bound_state_dirac_kg.physics import rhs,potential,config
from bound_state_dirac_kg.reference import fourier_matrices,spectrum,bvp,shoot
from bound_state_dirac_kg.model import BoundPINN


def test_dirac_real_complex_equivalence():
    x=np.linspace(-3,3,31); rng=np.random.default_rng(73)
    y=rng.normal(size=(2,len(x))); e=.67
    dy=rhs("dirac",x,y,e)
    psi=y[0]+0j,1j*y[1]
    hpsi=np.array([dy[1]+(1+potential(x))*psi[0],-1j*dy[0]+(potential(x)-1)*psi[1]])
    np.testing.assert_allclose(hpsi,e*np.array(psi),atol=1e-14)


def test_fourier_derivatives_and_free_dispersion():
    n=65;L=10;x,d,d2=fourier_matrices(n,L);k=3*np.pi/L
    np.testing.assert_allclose(d@np.sin(k*x),k*np.cos(k*x),atol=2e-13)
    np.testing.assert_allclose(d2@np.sin(k*x),-k*k*np.sin(k*x),atol=3e-13)
    h=np.block([[np.eye(n),d],[-d,-np.eye(n)]])
    e=np.linalg.eigvalsh(h)
    freq=2*np.pi*np.fft.fftfreq(n,d=2*L/n)
    expected=np.sort(np.r_[-np.sqrt(1+freq**2),np.sqrt(1+freq**2)])
    np.testing.assert_allclose(e,expected,atol=3e-13)
    assert np.sum(abs(e-1)<1e-10)==1 # only one zero-momentum positive mode


def test_kg_quadratic_pencil():
    n=65;L=20;x,d,d2=fourier_matrices(n,L)
    xx,es,vs=spectrum("kg",n,L)
    assert len(es)>0
    for i,e in enumerate(es):
        phi=vs[:n,i]
        r=d2@phi+((e-potential(x))**2-1)*phi
        assert np.linalg.norm(r)/np.linalg.norm(phi)<1e-10
        np.testing.assert_allclose(vs[n:,i],e*phi,atol=1e-11)


@pytest.mark.parametrize("kind",["dirac","kg"])
def test_autodiff_free_stationary_solution(kind):
    torch.set_default_dtype(torch.float64)
    class Free(BoundPINN):
        @property
        def energy(self): return torch.tensor(np.sqrt(1+.3**2))
        def potential(self,x): return torch.zeros_like(x)
        def forward(self,x):
            u=torch.cos(.3*x)
            if self.kind=="kg": return u
            return torch.cat([u,.3/(self.energy+1)*torch.sin(.3*x)],dim=1)
    model=Free(kind,config())
    x=torch.linspace(-5,5,41)[:,None].requires_grad_(True)
    assert model.residual(x).abs().max()<1e-14


@pytest.mark.parametrize("kind",["dirac","kg"])
def test_bvp_against_shooting(kind):
    sol=bvp(kind,L=16,n=121,tol=1e-9)
    e,_=shoot(kind,20,(.60,.70),rtol=1e-10)
    assert abs(sol.p[0]-e)<2e-9


def test_vector_and_scalar_coupling_differ():
    x=np.array([0.]);e=.65;v=potential(x)
    vector=(e-v)**2-1
    scalar=e*e-(1+v)**2
    assert abs(float((vector-scalar)[0]))>.1


def test_nodeless_parity_and_positive_kg_charge():
    torch.set_default_dtype(torch.float64);torch.manual_seed(17)
    model=BoundPINN("kg",config());x=torch.linspace(-20,20,101)[:,None]
    y=model(x)
    torch.testing.assert_close(y,torch.flip(y,[0]))
    assert bool(torch.all(y>0))
    assert bool(torch.all(2*(model.energy-model.potential(x))*y*y>0))


@pytest.mark.parametrize("kind",["dirac","kg"])
def test_saved_reference_roundtrip(kind):
    from bound_state_dirac_kg.physics import ROOT
    from bound_state_dirac_kg.evaluate import reference_on
    path=ROOT/"results"/"reference"/f"{kind}.npz"
    if not path.exists(): pytest.skip("Reference has not been generated")
    ref=np.load(path)
    y,e=reference_on(kind,ref["x"])
    np.testing.assert_allclose(y,ref["y"],atol=1e-14)
    assert e==ref["energy"]
