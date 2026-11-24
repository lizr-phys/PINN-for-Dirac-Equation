"""Physics-only ansatz, integral normalization and exact finite-domain Robin BC.

The derivative variant uses v=-u'/(E-V+m), which is an exact relation for
the chosen real Dirac components, not an approximate nonrelativistic rule.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

def gauss_grid(n, length):
    z, w = np.polynomial.legendre.leggauss(n)
    return torch.tensor((z+1)*length/2)[:, None], torch.tensor(w*length/2)[:, None]

class StructuredPINN(nn.Module):
    def __init__(self, kind, cfg, case, state=0, variant="derivative", initial_fraction=0.5):
        super().__init__()
        if state not in (0,1,2):
            raise ValueError("Only explicit 0/1/2-node ansatzes are implemented.")
        if variant not in ("derivative", "independent"):
            raise ValueError(variant)
        self.kind, self.state, self.variant = kind, state, variant
        self.shape_parameterization = cfg.get("shape_parameterization","compressed_feature")
        self.feature_count = cfg.get("feature_count",1)
        self.mass, self.depth, self.width = case["mass"], case["depth"], case["width"]
        self.L = case["half_length"]
        f = torch.tensor(initial_fraction)
        self.raw_energy = nn.Parameter(torch.logit(f))
        if state == 2:
            # Nodes are trainable within the domain; no reference node locations.
            initial_node = self.width*float(torch.empty(()).uniform_(0.5,1.5))
            self.raw_node = nn.Parameter(torch.logit(torch.tensor(initial_node/(self.L/2))))
        outs = 2 if kind == "dirac" and variant == "independent" else 1
        widths = [self.feature_count]+[cfg["hidden_width"]]*cfg["hidden_layers"]+[outs]
        layers = []
        for i, (a,b) in enumerate(zip(widths[:-1],widths[1:])):
            layer = nn.Linear(a,b)
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            if i == len(widths)-2:
                with torch.no_grad():
                    layer.weight.mul_(0.1)
            layers.append(layer)
            if i < len(widths)-2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        q,qw = gauss_grid(cfg["quadrature_points"],self.L)
        self.register_buffer("q",q)
        self.register_buffer("qw",qw)

    @property
    def energy(self):
        return self.mass*torch.sigmoid(self.raw_energy)

    @property
    def kappa(self):
        return torch.sqrt(self.mass**2-self.energy**2)

    @property
    def node(self):
        return self.L/2*torch.sigmoid(self.raw_node)

    def potential(self, x):
        return -self.depth*torch.exp(-x*x/(2*self.width**2))

    def feature(self, x):
        t = x*x/(x*x+self.width**2)
        if self.feature_count == 3:
            return torch.cat([2*t-1,2*torch.exp(-x*x/(2*self.width**2))-1,
                              2*torch.log1p(x*x/self.width**2)/np.log1p(self.L**2/self.width**2)-1],dim=1)
        if self.shape_parameterization == "compact_gate":
            return 2*t-1
        tL = self.L**2/(self.L**2+self.width**2)
        z = t/tL
        return 2*(1-(1-z)**3)-1  # first and second derivatives vanish at +-L

    def node_factor(self, x):
        if self.state == 0:
            return torch.ones_like(x)
        if self.state == 1:
            return torch.tanh(x/self.width)
        return (1-x*x/self.node**2)/(1+x*x/self.width**2)

    def node_log_derivative_at_L(self):
        if self.state == 0:
            return torch.zeros_like(self.energy)
        if self.state == 1:
            p = torch.tanh(torch.tensor(self.L/self.width))
            return (1-p*p)/(self.width*p)
        return 2*self.L/(self.L**2-self.node**2)-2*self.L/(self.width**2+self.L**2)

    def node_log_second_derivative_at_L(self):
        if self.state == 0:
            return torch.zeros_like(self.energy)
        if self.state == 1:
            p = torch.tanh(torch.tensor(self.L/self.width))
            return -(1-p*p)*(1+1/(p*p))/self.width**2
        return -2*(self.L**2+self.node**2)/(self.L**2-self.node**2)**2-2*(self.width**2-self.L**2)/(self.width**2+self.L**2)**2

    def envelope(self, x):
        vL = -self.depth*np.exp(-self.L**2/(2*self.width**2))
        kbc = self.kappa
        if self.kind == "dirac":
            kbc = kbc*(self.energy-vL+self.mass)/(self.energy+self.mass)
            A = self.energy-vL+self.mass
            B = self.energy-vL-self.mass
            vpL = -self.L/self.width**2*vL
            target_curvature = vpL/A*kbc-A*B-kbc*kbc
        else:
            target_curvature = self.mass**2-(self.energy-vL)**2-self.kappa**2
        sL = self.L/np.sqrt(self.L**2+self.width**2)
        s2L = self.width**2/(self.L**2+self.width**2)**1.5
        d1 = kbc+self.node_log_derivative_at_L()-self.kappa*sL
        d2 = self.L*(self.node_log_second_derivative_at_L()-self.kappa*s2L-target_curvature)
        c4 = (d2-d1)/2
        c2 = (3*d1-d2)/2
        s = torch.sqrt(x*x+self.width**2)-self.width
        return torch.exp(-self.kappa*s-c2*x*x/(2*self.L)-c4*x**4/(4*self.L**3))

    def raw_wave(self, x):
        if not x.requires_grad and self.kind == "dirac" and self.variant == "derivative":
            x = x.detach().requires_grad_(True)
        raw = self.net(self.feature(x))
        env = self.envelope(x)
        log_shape = raw[:, :1]
        if self.shape_parameterization == "compact_gate":
            log_shape = log_shape*(1-x*x/self.L**2)**3
        u = env*self.node_factor(x)*torch.exp(log_shape)
        if self.kind == "kg":
            return u
        if self.variant == "derivative":
            v = -grad(u,x)/(self.energy-self.potential(x)+self.mass)
        else:
            xL = torch.tensor([[self.L]])
            uL_without_env = self.node_factor(xL)
            if self.shape_parameterization != "compact_gate":
                uL_without_env = uL_without_env*torch.exp(self.net(self.feature(xL))[:, :1])
            p = torch.tanh(x/self.width) if self.state % 2 == 0 else torch.ones_like(x)
            pL = np.tanh(self.L/self.width) if self.state % 2 == 0 else 1.0
            endpoint = self.kappa/(self.energy+self.mass)*uL_without_env/pL
            v = env*p*(endpoint+(1-x*x/self.L**2)*raw[:, 1:2])
        return torch.cat([u,v],dim=1)

    def normalizer(self):
        yq = self.raw_wave(self.q.detach().requires_grad_(True))
        return torch.sqrt(2*torch.sum(self.qw*yq*yq))

    def forward(self, x, scale=None):
        return self.raw_wave(x)/(self.normalizer() if scale is None else scale)

    def residual(self, x, scale=None):
        y = self(x,scale)
        ev = self.energy-self.potential(x)
        if self.kind == "dirac":
            return torch.cat([grad(y[:, :1],x)+(ev+self.mass)*y[:, 1:2],
                              grad(y[:, 1:2],x)-(ev-self.mass)*y[:, :1]],dim=1)
        return grad(grad(y,x),x)+(ev*ev-self.mass**2)*y

    def boundary_residual(self, scale=None):
        x = torch.tensor([[self.L]],requires_grad=True)
        y = self(x,scale)
        if self.kind == "dirac":
            r = y[:, 1:2]-self.kappa/(self.energy+self.mass)*y[:, :1]
        else:
            r = grad(y,x)+self.kappa*y
        # Relative to the actual upper-component tail amplitude, not its envelope.
        return r/y[:, :1]

class StructuredLoss:
    def __init__(self, model, cfg, lower_state=None):
        self.model = model
        self.x,self.w = gauss_grid(cfg["collocation_points"],model.L)
        self.lower_state=lower_state
        self.ortho_weight=cfg.get("orthogonality_weight",1.0)

    def __call__(self):
        scale = self.model.normalizer()
        x = self.x.detach().requires_grad_(True)
        r = self.model.residual(x,scale)
        pde = 2*torch.sum(self.w*r*r)
        ortho=torch.zeros_like(pde)
        if self.lower_state is not None:
            ly,le=self.lower_state
            yq=self.model.raw_wave(self.model.q.detach().requires_grad_(True))/scale
            if self.model.kind=="dirac":
                overlap=2*torch.sum(self.model.qw*yq*ly)
            else:
                v=self.model.potential(self.model.q)
                overlap=2*torch.sum(self.model.qw*(self.model.energy+le-2*v)*yq*ly)
                charge=2*torch.sum(self.model.qw*2*(self.model.energy-v)*yq*yq)
                lower_charge=2*torch.sum(self.model.qw*2*(le-v)*ly*ly)
                overlap=overlap/torch.sqrt(charge*lower_charge)
            ortho=overlap**2
        return pde+self.ortho_weight*ortho, {"pde":pde,"orthogonality":ortho,
                     "normalization":torch.zeros_like(pde),"boundary":torch.zeros_like(pde),"norm":torch.ones_like(pde)}
