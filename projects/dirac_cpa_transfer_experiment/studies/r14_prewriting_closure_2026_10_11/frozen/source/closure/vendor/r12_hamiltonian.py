from __future__ import annotations
import numpy as np
import torch

def potential_apply(state, cfg):
    result = torch.zeros_like(state) if torch.is_tensor(state) else np.zeros_like(state)
    for shift, coefficient in cfg.coupling_terms:
        result[:, shift:] += coefficient*state[:, :-shift]
        result[:, :-shift] += coefficient*state[:, shift:]
    return result

def hamiltonian_apply(state, cfg):
    if torch.is_tensor(state):
        k = torch.tensor(cfg.wave_numbers,dtype=torch.float64,device=state.device)[None,:,None]
        free = k*state.flip(-1)+cfg.mass*torch.stack((state[...,0],-state[...,1]),-1)
    else:
        k = np.asarray(cfg.wave_numbers)[None,:,None]
        free = k*state[...,::-1]+cfg.mass*np.stack((state[...,0],-state[...,1]),-1)
    return free+potential_apply(state,cfg)

def build_hamiltonian(cfg):
    h = np.zeros((2*cfg.mode_count,2*cfg.mode_count),dtype=np.complex128)
    for j,k in enumerate(cfg.wave_numbers):
        h[2*j:2*j+2,2*j:2*j+2] = [[cfg.mass,k],[k,-cfg.mass]]
        for shift,coefficient in cfg.coupling_terms:
            if j+shift < cfg.mode_count:
                h[2*j:2*j+2,2*(j+shift):2*(j+shift)+2] = coefficient*np.eye(2)
                h[2*(j+shift):2*(j+shift)+2,2*j:2*j+2] = coefficient*np.eye(2)
    return h

def exterior_residual(state,cfg):
    """Zero extension: R_out = -V psi, retaining overlapping harmonic sources."""
    width=cfg.max_coupling_shift
    expanded=np.zeros((len(state),cfg.mode_count+2*width,2),dtype=np.complex128)
    expanded[:,width:width+cfg.mode_count]=state
    source=potential_apply(expanded,cfg)
    outside=np.r_[np.arange(width),np.arange(width+cfg.mode_count,len(expanded[0]))]
    ns=np.r_[np.arange(cfg.n_min-width,cfg.n_min),np.arange(cfg.n_max+1,cfg.n_max+width+1)]
    return -source[:,outside],source[:,outside],ns
