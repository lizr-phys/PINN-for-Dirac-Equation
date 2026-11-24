"""Complex scattering networks with hard channel boundary conditions.

The main network is a random-hidden-layer radial-basis neural network. Its
complex output weights are trained on the differential equation only. Analytic
network derivatives are used for the linear solve and checked against PyTorch
automatic differentiation in tests. This is a PIELM-type PINN, not a claim about
all-weight-trained deep PINNs.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from .physics import CASE, potential, potential_prime, free_data


def multiply_carrier(envelope, first, second, x, signed_k):
    carrier = np.exp(1j*signed_k*x)
    return (envelope*carrier, (first+1j*signed_k*envelope)*carrier,
            (second+2j*signed_k*first-signed_k**2*envelope)*carrier)


def lifts(x, length):
    x = np.asarray(x)
    s = (x+length)/(2*length)
    ds = 1/(2*length)
    h = 3*s*s-2*s**3
    dh = (6*s-6*s*s)*ds
    ddh = (6-12*s)*ds*ds
    gate = 16*(s*s-2*s**3+s**4)
    dgate = 16*(2*s-6*s*s+4*s**3)*ds
    ddgate = 16*(2-12*s+12*s*s)*ds*ds
    hl = 2*length*(s**3-2*s*s+s)
    dhl = 3*s*s-4*s+1
    ddhl = (6*s-4)*ds
    hr = 2*length*(s**3-s*s)
    dhr = 3*s*s-2*s
    ddhr = (6*s-2)*ds
    return (1-h, -dh, -ddh), (h, dh, ddh), (gate, dgate, ddgate), (hl, dhl, ddhl), (hr, dhr, ddhr)


class ScatteringNetwork:
    def __init__(self, kind, seed=0, features=64, case=None, degree=0,
                 width_min=0.8, width_max=2.8):
        if kind not in ("dirac", "kg"):
            raise ValueError(kind)
        self.kind, self.seed, self.features, self.degree = kind, seed, features, degree
        self.case = CASE.copy() if case is None else case.copy()
        rng = np.random.default_rng(seed)
        length = self.case["half_length"]
        spacing = 2*length/features
        self.centers = np.linspace(-length+spacing/2, length-spacing/2, features)
        self.centers += rng.uniform(-.35, .35, features)*spacing
        self.widths = rng.uniform(width_min, width_max, features)
        self.coefficients = np.zeros((2+2*features, degree+1), complex)

    @property
    def trainable_parameters(self):
        return int(2*self.coefficients.size)

    @property
    def frozen_parameters(self):
        return 2*self.features  # shared centers and widths for both carriers

    def energy_features(self, energy):
        k = np.sqrt(np.asarray(energy)**2-self.case["mass"]**2)
        kmin, kmax = np.sqrt(1.01**2-1), np.sqrt(2.**2-1)
        z = 2*(k-kmin)/(kmax-kmin)-1
        return chebvander(np.atleast_1d(z), self.degree)

    def basis(self, x, energy):
        """Return base and neural columns with exact first/second derivatives."""
        x = np.atleast_1d(x).astype(float)
        length = self.case["half_length"]
        k, _, _ = free_data(self.kind, energy, self.case["mass"])
        left, right, gate, hl, hr = lifts(x, length)
        inc = tuple((1j*k)**order*np.exp(1j*k*x) for order in range(3))
        rr = multiply_carrier(*left, x, -k)
        tau = multiply_carrier(*right, x, k)
        # With Dirac elimination the boundary derivative involves the actual
        # A at the endpoint, while the spinor itself is the free-channel basis.
        if self.kind == "dirac":
            delta_left = -k*potential(-length, self.case)/(energy+self.case["mass"])
            delta_right = -k*potential(length, self.case)/(energy+self.case["mass"])
            inc = tuple(inc[j]+1j*delta_left*np.exp(-1j*k*length)*hl[j]
                        +1j*delta_right*np.exp(1j*k*length)*hr[j] for j in range(3))
            rr = tuple(rr[j]-1j*delta_left*np.exp(1j*k*length)*hl[j] for j in range(3))
            tau = tuple(tau[j]+1j*delta_right*np.exp(1j*k*length)*hr[j] for j in range(3))
        dx = x[:, None]-self.centers[None, :]
        w2 = self.widths[None, :]**2
        activation = np.exp(-0.5*dx*dx/w2)
        da = -dx/w2*activation
        dda = (dx*dx/(w2*w2)-1/w2)*activation
        p = gate[0][:, None]*activation
        dp = gate[1][:, None]*activation+gate[0][:, None]*da
        ddp = gate[2][:, None]*activation+2*gate[1][:, None]*da+gate[0][:, None]*dda
        plus = multiply_carrier(p, dp, ddp, x[:, None], k)
        minus = multiply_carrier(p, dp, ddp, x[:, None], -k)
        matrix = tuple(np.column_stack([rr[j], tau[j], plus[j], minus[j]]) for j in range(3))
        return inc, matrix

    def residual_basis(self, x, energy):
        base, basis = self.basis(x, energy)
        v, vp = potential(x, self.case), potential_prime(x, self.case)
        a, b = energy-v+self.case["mass"], energy-v-self.case["mass"]
        p = vp/a if self.kind == "dirac" else np.zeros_like(v)
        rhs = base[2]+p*base[1]+a*b*base[0]
        mat = basis[2]+p[:, None]*basis[1]+(a*b)[:, None]*basis[0]
        # Dirac eliminated equation is divided by A, giving the magnitude of
        # the nontrivial original first-order component residual.
        scale = a if self.kind == "dirac" else np.ones_like(a)
        return rhs/scale, mat/scale[:, None]

    def field(self, x, energy):
        base, basis = self.basis(x, energy)
        coefficients = self.coefficients @ self.energy_features(energy)[0]
        _, amplitude, _ = free_data(self.kind, energy, self.case["mass"])
        return tuple(amplitude*(base[j]+basis[j]@coefficients) for j in range(3))

    def state_and_residual(self, x, energy):
        f, df, ddf = self.field(x, energy)
        v, vp = potential(x, self.case), potential_prime(x, self.case)
        a, b = energy-v+self.case["mass"], energy-v-self.case["mass"]
        if self.kind == "dirac":
            g = -1j*df/a
            dg = -1j*(ddf/a+df*vp/(a*a))
            state = np.column_stack([f, g])
            residual = np.column_stack([-1j*dg-b*f, -1j*df-a*g])
        else:
            state = np.column_stack([f, df])
            residual = (ddf+a*b*f)[:, None]
        return state, residual

    def amplitudes(self, energy):
        coeff = self.coefficients @ self.energy_features(energy)[0]
        return complex(coeff[0]), complex(1+coeff[1])

    def save(self, path):
        np.savez_compressed(path, coefficients=self.coefficients, centers=self.centers,
                            widths=self.widths, kind=self.kind, seed=self.seed,
                            features=self.features, degree=self.degree,
                            case=np.array([self.case[k] for k in ("mass", "depth", "width", "half_length")]))

    @classmethod
    def load(cls, path):
        data = np.load(path)
        case = dict(zip(("mass", "depth", "width", "half_length"), data["case"].tolist()))
        net = cls(str(data["kind"]), int(data["seed"]), int(data["features"]), case, int(data["degree"]))
        net.coefficients, net.centers, net.widths = data["coefficients"], data["centers"], data["widths"]
        return net

    def torch_field(self, x, energy):
        """Independent differentiable implementation used for AD audit."""
        import torch
        length = self.case["half_length"]
        k, amplitude, _ = free_data(self.kind, energy, self.case["mass"])
        coeff = torch.tensor(self.coefficients @ self.energy_features(energy)[0], dtype=torch.complex128)
        s = (x+length)/(2*length)
        h = 3*s*s-2*s**3
        gate = 16*s*s*(1-s)**2
        plus, minus = torch.exp(1j*k*x), torch.exp(-1j*k*x)
        y = plus+coeff[0]*(1-h)*minus+coeff[1]*h*plus
        if self.kind == "dirac":
            dl = -k*potential(-length, self.case)/(energy+self.case["mass"])
            dr = -k*potential(length, self.case)/(energy+self.case["mass"])
            hl, hr = 2*length*(s**3-2*s*s+s), 2*length*(s**3-s*s)
            y = y+1j*dl*np.exp(-1j*k*length)*hl+1j*dr*np.exp(1j*k*length)*hr
            y = y-coeff[0]*1j*dl*np.exp(1j*k*length)*hl+coeff[1]*1j*dr*np.exp(1j*k*length)*hr
        centers = torch.tensor(self.centers)
        widths = torch.tensor(self.widths)
        activation = torch.exp(-0.5*((x[:, None]-centers)/widths)**2)
        y = y+gate*(plus*(activation@coeff[2:2+self.features])
                    +minus*(activation@coeff[2+self.features:]))
        return amplitude*y

