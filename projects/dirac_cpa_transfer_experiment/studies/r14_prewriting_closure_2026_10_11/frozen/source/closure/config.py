from __future__ import annotations
from dataclasses import dataclass
import json,math
import numpy as np
import torch
from plan import BASE,CASES
from .vendor.r13_config import R13Config

@dataclass(frozen=True)
class Config(R13Config):
    @property
    def ladder_half_width(self):return 10.

def configuration(case,bounds=None):
    if case not in CASES:raise ValueError(case)
    if bounds is None:
        v=json.loads((BASE/'references/calibration.json').read_text(encoding='utf-8'))
        if v['status']!='PASS':raise RuntimeError('Reference not calibrated')
        bounds=v['selected_common_truncation']
    return Config(case=case,n_min=int(bounds[0]),n_max=int(bounds[1]),mass=10. if case=='m10' else 5.,
        t_max=10. if case=='T10' else 5.,potential_strength=.4 if case=='V04' else .2,
        potential_harmonics=((1,.2/math.sqrt(2)),(2,.2/math.sqrt(2))) if case=='two_harmonics' else ((1,.4 if case=='V04' else .2),),
        residual_time_points=8001 if case=='T10' else 4001)

def initial_coefficients(cfg):
    from .vendor.r9_models import initial_coefficients as original
    coeff=original(cfg)
    if cfg.case=='chirped':
        k=torch.tensor(cfg.wave_numbers,dtype=torch.float64)
        phase=torch.exp(.5j*(k-2).square())
        coeff=coeff*phase[:,None]
        coeff[:,1]*=torch.exp(torch.tensor(1j*math.pi/3,dtype=torch.complex128))
    return coeff

def independent_initial(cfg):
    k=np.arange(33)/8
    g=np.exp(-(k-2)**2/(4*.5**2));g=g/np.sqrt(np.sum(g*g))
    coeff=np.zeros((cfg.mode_count,2),dtype=np.complex128)
    coeff[-cfg.n_min:33-cfg.n_min,:]=g[:,None]/np.sqrt(2)
    wave=np.asarray(cfg.wave_numbers)
    if cfg.case=='chirped':
        coeff*=np.exp(.5j*(wave-2)**2)[:,None];coeff[:,1]*=np.exp(1j*np.pi/3)
    a=np.arctan2(wave,cfg.mass)/2
    plus=np.stack((np.cos(a),np.sin(a)),-1);minus=np.stack((-np.sin(a),np.cos(a)),-1)
    return coeff[:,:1]*plus+coeff[:,1:]*minus
