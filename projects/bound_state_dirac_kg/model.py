"""Physics-only neural eigenfunction; no reference solver is imported here."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn


def grad(y,x):
    return torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True)[0]


class BoundPINN(nn.Module):
    def __init__(self,kind,cfg):
        super().__init__()
        self.kind=kind
        self.mass=cfg["mass"]; self.depth=cfg["depth"]; self.width=cfg["width"]
        self.L=cfg["half_length"]
        self.elo,self.ehi=cfg["energy_interval"]
        self.raw_energy=nn.Parameter(torch.empty(()).uniform_(-1,1))
        widths=[1]+[cfg["hidden_width"]]*cfg["hidden_layers"]+[2 if kind=="dirac" else 1]
        layers=[]
        for i,(a,b) in enumerate(zip(widths[:-1],widths[1:])):
            linear=nn.Linear(a,b)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            if i==len(widths)-2:
                with torch.no_grad(): linear.weight.mul_(0.1)
            layers.append(linear)
            if i<len(widths)-2: layers.append(nn.Tanh())
        self.net=nn.Sequential(*layers)

    @property
    def energy(self):
        return self.elo+(self.ehi-self.elo)*torch.sigmoid(self.raw_energy)

    @property
    def kappa(self):
        return torch.sqrt(self.mass**2-self.energy**2)

    def envelope(self,x):
        return torch.exp(-self.kappa*(torch.sqrt(x*x+self.width**2)-self.width))

    def forward(self,x):
        t=x*x/(x*x+self.width**2)
        raw=self.net(2*t-1)
        env=self.envelope(x)
        u=env*torch.exp(raw[:,:1])
        if self.kind=="kg": return u
        v=env*torch.tanh(x/self.width)*raw[:,1:2]
        return torch.cat([u,v],dim=1)

    def potential(self,x):
        return -self.depth*torch.exp(-x*x/(2*self.width**2))

    def residual(self,x):
        y=self(x); e=self.energy; vpot=self.potential(x)
        if self.kind=="dirac":
            return torch.cat([grad(y[:,:1],x)+(e-vpot+self.mass)*y[:,1:2],
                              grad(y[:,1:2],x)-(e-vpot-self.mass)*y[:,:1]],dim=1)
        return grad(grad(y,x),x)+((e-vpot)**2-self.mass**2)*y

    def boundary_residual(self):
        x=torch.tensor([[self.L]],requires_grad=True)
        y=self(x)
        if self.kind=="dirac":
            r=y[:,1:2]-self.kappa/(self.energy+self.mass)*y[:,:1]
        else:
            r=grad(y,x)+self.kappa*y
        # Scale out the known tiny tail to enforce its shape, not only its size.
        return r/self.envelope(x)


def gauss_grid(n,L):
    x,w=np.polynomial.legendre.leggauss(n)
    return torch.tensor((x+1)*L/2)[:,None],torch.tensor(w*L/2)[:,None]


class Loss:
    def __init__(self,model,cfg):
        self.model=model
        self.x,self.w=gauss_grid(cfg["collocation_points"],model.L)
        self.q,self.qw=gauss_grid(cfg["quadrature_points"],model.L)

    def __call__(self):
        x=self.x.detach().requires_grad_(True)
        r=self.model.residual(x)
        pde=2*torch.sum(self.w*r*r)
        y=self.model(self.q)
        norm=2*torch.sum(self.qw*y*y)
        normalization=(norm-1)**2
        bc=torch.sum(self.model.boundary_residual()**2)
        total=pde+10*normalization+0.01*bc
        return total,dict(pde=pde,normalization=normalization,boundary=bc,norm=norm)
