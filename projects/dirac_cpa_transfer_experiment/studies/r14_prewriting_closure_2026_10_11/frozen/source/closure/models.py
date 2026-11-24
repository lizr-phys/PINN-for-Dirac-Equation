from __future__ import annotations
import math
import torch
from torch import nn
from .config import initial_coefficients
from .vendor.r13_models import LearnedRecoveryCarrier,total_norm_parameterization,validate_cartesian
from .vendor.r9_models import R9Model
from .vendor.wavepacket_models import PairedTanhMLP
from .vendor.r8_models import paired,features,basis
from .vendor.r12_hamiltonian import hamiltonian_apply

class IdentityCarrier(nn.Module):
    analytic_prior=False
    def forward(self,k,t):return self.value_and_time_derivative(k,t)[0]
    def value_and_time_derivative(self,k,t):
        k,t=paired(k,t);p=torch.ones((len(t),2),dtype=torch.complex128,device=t.device)
        return p,torch.zeros_like(p)
    def freeze(self):return self

class StaticCarrier(LearnedRecoveryCarrier):
    """Coalesce identical k inputs for the autonomous carrier; exact shared graph."""
    def __init__(self,cfg,seed):super().__init__(cfg,'static_frequency',seed)
    def rates(self,k):
        if len(k)%self.cfg.mode_count==0:
            nk=self.cfg.mode_count
            blocks=k.reshape(-1,nk,1)
            if torch.equal(blocks,blocks[:1].expand_as(blocks)):
                omega=self.mlp((blocks[0]-self.cfg.k0)/self.cfg.ladder_half_width)
                return omega.repeat(len(blocks),1)
        return self.mlp((k-self.cfg.k0)/self.cfg.ladder_half_width)
    def forward(self,k,t):return self.value_and_time_derivative(k,t)[0]
    def value_and_time_derivative(self,k,t):
        k,t=paired(k,t);omega=self.rates(k);p=torch.exp(-1j*t*omega)
        return p,-1j*omega*p

class Sine(nn.Module):
    def __init__(self,omega):super().__init__();self.omega=float(omega)
    def forward(self,x):return torch.sin(self.omega*x)

class OscillatoryMLP(nn.Module):
    def __init__(self,cfg,family,options,seed):
        super().__init__();self.family=family
        gen=torch.Generator().manual_seed(800000+seed)
        feature_gen=torch.Generator().manual_seed(840000+seed)
        layers=[];nin=2
        if family=='fourier':
            self.register_buffer('B',torch.randn((32,2),generator=feature_gen,dtype=torch.float64)*options['scale'])
            nin=64
        for i in range(cfg.hidden_layers):
            linear=nn.Linear(nin,cfg.hidden_width,dtype=torch.float64)
            with torch.no_grad():
                if family=='siren':
                    bound=1/nin if i==0 else math.sqrt(6/nin)/options['omega']
                    linear.weight.uniform_(-bound,bound,generator=gen);linear.bias.uniform_(-bound,bound,generator=gen)
                else:
                    linear.weight.copy_(torch.randn(linear.weight.shape,generator=gen,dtype=torch.float64)*math.sqrt(2/(nin+cfg.hidden_width)));linear.bias.zero_()
            layers.extend((linear,Sine(options['omega']) if family=='siren' else nn.Tanh()));nin=cfg.hidden_width
        out=nn.Linear(nin,4,dtype=torch.float64)
        with torch.no_grad():
            if family=='siren':
                bound=math.sqrt(6/nin)/options['omega'];out.weight.uniform_(-bound,bound,generator=gen);out.bias.uniform_(-bound,bound,generator=gen)
            else:out.weight.copy_(torch.randn(out.weight.shape,generator=gen,dtype=torch.float64)*math.sqrt(2/(nin+4)));out.bias.zero_()
        self.net=nn.Sequential(*layers,out)
    def map_features(self,value,tangent=None):
        if self.family!='fourier':return value if tangent is None else (value,tangent)
        phase=2*math.pi*value@self.B.T
        mapped=torch.cat((torch.sin(phase),torch.cos(phase)),1)
        if tangent is None:return mapped
        dt=2*math.pi*tangent@self.B.T
        return mapped,torch.cat((torch.cos(phase)*dt,-torch.sin(phase)*dt),1)
    def forward(self,value):return self.net(self.map_features(value))

def network_value_and_time_derivative(network,cfg,k,t):
    value=features(cfg,k,t)
    tangent=torch.cat((torch.zeros_like(t),torch.ones_like(t)/cfg.t_max),1)
    if isinstance(network,OscillatoryMLP):value,tangent=network.map_features(value,tangent)
    for layer in network.net:
        if isinstance(layer,nn.Linear):value,tangent=layer(value),torch.nn.functional.linear(tangent,layer.weight)
        elif isinstance(layer,nn.Tanh):value=torch.tanh(value);tangent=(1-value.square())*tangent
        elif isinstance(layer,Sine):
            phase=layer.omega*value;tangent=layer.omega*torch.cos(phase)*tangent;value=torch.sin(phase)
        else:raise TypeError(layer)
    return value,tangent

class Model(R9Model):
    def __init__(self,cfg,group,seed,options=None):
        super().__init__(cfg,'B_complex',seed)
        self.group=group;self.envelope_kind='pde_only' if group.endswith('_pde') else 'hard_total_norm'
        self.initial_coeff.copy_(initial_coefficients(cfg))
        if group in ('original_pde','original_hard'):
            self.carrier_kind='original';self.carrier=LearnedRecoveryCarrier(cfg,'original',seed)
        elif group in ('static_pde','learned_hard','joint_hard'):
            self.carrier_kind='static_frequency';self.carrier=StaticCarrier(cfg,seed)
        elif group=='analytic_hard':
            from .vendor.r11_exact_carrier_diagnostic import ExactCarrierDiagnostic
            self.carrier_kind='exact';self.carrier=ExactCarrierDiagnostic(cfg)
        elif group in ('no_carrier_hard','siren','fourier'):
            self.carrier_kind='none';self.carrier=IdentityCarrier()
            if group in ('siren','fourier'):self.networks[0]=OscillatoryMLP(cfg,group,options,seed)
        else:raise ValueError(group)
        self.model_name=group
    def branch_coefficients(self,k,t):
        k,t=paired(k,t);z=self.networks[0](features(self.cfg,k,t))
        raw=self.initial_coeff[self.lookup(k)]+t/self.cfg.t_max*torch.complex(z[:,:2],z[:,2:])
        if self.envelope_kind=='hard_total_norm':
            validate_cartesian(self.cfg,k,t);raw=total_norm_parameterization(raw,mode_count=self.cfg.mode_count)
        return self.carrier(k,t)*raw

def spinor_value_and_time_derivative(model,cfg,k,t):
    k,t=paired(k,t)
    z,zt=network_value_and_time_derivative(model.networks[0],cfg,k,t)
    z=torch.complex(z[:,:2],z[:,2:]);zt=torch.complex(zt[:,:2],zt[:,2:])
    raw=model.initial_coeff[model.lookup(k)]+t/cfg.t_max*z;raw_t=z/cfg.t_max+t/cfg.t_max*zt
    if model.envelope_kind=='hard_total_norm':raw,raw_t=total_norm_parameterization(raw,raw_t,mode_count=cfg.mode_count)
    p,pt=model.carrier.value_and_time_derivative(k,t);c,ct=p*raw,pt*raw+p*raw_t
    up,um=basis(k,cfg.mass)
    return c[:,:1]*up+c[:,1:]*um,ct[:,:1]*up+ct[:,1:]*um

def residual(model,cfg,k,t):
    nt=validate_cartesian(cfg,k,t);s,st=spinor_value_and_time_derivative(model,cfg,k,t)
    return 1j*st.reshape(nt,cfg.mode_count,2)-hamiltonian_apply(s.reshape(nt,cfg.mode_count,2),cfg)

def training_loss(model,cfg,k,t):
    r=residual(model,cfg,k,t);return r.abs().square().mean(-1).sum(-1).mean()/cfg.length

def free_loss(carrier,cfg,k,t):
    from .vendor.r13_residual import free_carrier_residual
    return free_carrier_residual(carrier,cfg,k,t).abs().square().mean()
