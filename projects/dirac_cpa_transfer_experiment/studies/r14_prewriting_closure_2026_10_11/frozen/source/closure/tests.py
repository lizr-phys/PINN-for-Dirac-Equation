"""Scientific equivalence, complex derivatives, gradients and exact resume tests."""
from __future__ import annotations
import json,math,time
from pathlib import Path
import numpy as np
import torch
from plan import BASE,write,TUNING,sha
from .config import configuration,independent_initial,initial_coefficients
from .models import Model,StaticCarrier,spinor_value_and_time_derivative,training_loss,free_loss
from .vendor.r13_models import R13Model,LearnedRecoveryCarrier,total_norm_parameterization
from .vendor.r13_residual import training_loss as old_loss
from .vendor.r8_models import compose
from .vendor.r12_hamiltonian import hamiltonian_apply
from .train import run

def test_derivative(group,options=None):
    cfg=configuration('chirped',(0,32));model=Model(cfg,group,999001,options)
    ks=torch.tensor(cfg.wave_numbers).reshape(-1,1);times=torch.tensor([.31,.79],requires_grad=True)
    k=ks.repeat(2,1);t=times.repeat_interleave(cfg.mode_count).reshape(-1,1)
    actual,dt=spinor_value_and_time_derivative(model,cfg,k,t)
    def forward(shared):return model(k,shared.repeat_interleave(cfg.mode_count).reshape(-1,1))
    value,reference=torch.autograd.functional.jvp(forward,times,torch.ones_like(times),create_graph=True)
    val_error=float(abs(actual-value).max());der_error=float(abs(dt-reference).max())
    manual=1j*dt.reshape(2,cfg.mode_count,2)-hamiltonian_apply(actual.reshape(2,cfg.mode_count,2),cfg)
    independent=1j*reference.reshape(2,cfg.mode_count,2)-hamiltonian_apply(value.reshape(2,cfg.mode_count,2),cfg)
    lm=manual.abs().square().mean(-1).sum(-1).mean()/cfg.length;lr=independent.abs().square().mean(-1).sum(-1).mean()/cfg.length
    params=tuple(model.parameters());gm=torch.autograd.grad(lm,params,allow_unused=True);gr=torch.autograd.grad(lr,params,allow_unused=True)
    grad_error=max(float(abs(a-b).max())/max(float(abs(b).max()),1.) for a,b in zip(gm,gr) if a is not None)
    assert val_error<2e-13 and der_error<2e-12 and grad_error<1e-11,(group,val_error,der_error,grad_error)
    norm=np.sum(abs(value.detach().numpy().reshape(2,cfg.mode_count,2))**2,(1,2))
    assert abs(norm-1).max()<2e-14
    initial=forward(torch.zeros(2));init=np.tile(independent_initial(cfg),(2,1))
    assert float(abs(initial.detach()-torch.from_numpy(init)).max())<2e-14
    return dict(group=group,forward_max=val_error,time_derivative_max=der_error,parameter_gradient_scaled_max=grad_error,all_active_gradients_checked=True)

def old_equivalence():
    cfg=configuration('m5T5',(0,32));ks=torch.tensor(cfg.wave_numbers).reshape(-1,1).repeat(2,1)
    ts=torch.tensor([.4,.7]).repeat_interleave(cfg.mode_count).reshape(-1,1);records=[]
    for group,kind,envelope in (('original_pde','original','pde_only'),('static_pde','static_frequency','pde_only'),('original_hard','original','hard_total_norm'),('learned_hard','static_frequency','hard_total_norm'),('analytic_hard','exact','hard_total_norm')):
        new=Model(cfg,group,999002);old=R13Model(cfg,kind,envelope,999002)
        assert torch.equal(new.initial_coeff,old.initial_coeff)
        value=float(abs(new(ks,ts)-old(ks,ts)).max());lm=training_loss(new,cfg,ks,ts);lo=old_loss(old,cfg,ks,ts)[0]
        gn=torch.autograd.grad(lm,tuple(new.parameters()));go=torch.autograd.grad(lo,tuple(old.parameters()))
        grad=max(float(abs(a-b).max())/max(1.,float(abs(b).max())) for a,b in zip(gn,go))
        assert value<2e-14 and abs(float(lm-lo))<2e-13 and grad<2e-12
        records.append(dict(group=group,value_max=value,objective_difference=abs(float(lm-lo)),gradient_scaled_max=grad))
    return records

def exact_recovery():
    case='m5T5';root='engineering/recovery';budget=8
    jobs=[]
    for label in ('continuous','resumed'):
        jobs.append(dict(type='coupled',stage='engineering',case=case,group='joint_hard',seed=999003,budget=budget,
            snapshots=[budget],bounds=[0,32],directory=root+'/'+label))
    run(jobs[0],engineering=True,do_evaluation=False)
    run(jobs[1],engineering=True,stop_after=3,do_evaluation=False);run(jobs[1],engineering=True,do_evaluation=False)
    a=torch.load(BASE/jobs[0]['directory']/'recovery_state.pt',weights_only=False);b=torch.load(BASE/jobs[1]['directory']/'recovery_state.pt',weights_only=False)
    def equal(x,y):
        if isinstance(x,torch.Tensor):return torch.equal(x,y)
        if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
        if isinstance(x,(tuple,list)):return len(x)==len(y) and all(equal(u,v) for u,v in zip(x,y))
        if isinstance(x,np.ndarray):return np.array_equal(x,y)
        return x==y
    keys=['state_dict','optimizer','sampler','python_rng','numpy_rng','torch_rng']
    matches={k:equal(a[k],b[k]) for k in keys};assert all(matches.values()),matches
    return dict(bitwise_equal=matches,interrupted_at_step=3,endpoint=8,new_initialization_after_resume=False)

def main():
    start=time.perf_counter();torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    rows=[test_derivative(g,TUNING[g][0] if g in TUNING else None) for g in ('learned_hard','original_hard','analytic_hard','joint_hard','no_carrier_hard','siren','fourier')]
    old=old_equivalence();recovery=exact_recovery()
    try:total_norm_parameterization(torch.zeros((33,2),dtype=torch.complex128),mode_count=33)
    except FloatingPointError:zero_failure=True
    else:zero_failure=False
    assert zero_failure
    result=dict(status='PASS',derivative_and_gradient_checks=rows,R13_equivalence=old,exact_recovery=recovery,
        zero_norm_denominator_preserved_failure=zero_failure,seconds=time.perf_counter()-start)
    write(BASE/'engineering/tests_and_recovery.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
