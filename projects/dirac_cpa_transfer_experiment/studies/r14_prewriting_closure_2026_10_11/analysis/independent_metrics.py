"""Independent metric implementation using FFT quadrature, no training evaluator."""
from __future__ import annotations
import math
import numpy as np
from scipy.fft import ifft

def grid_checks(pred,ref,n,length=16*math.pi):
    """Independent spatial-grid and initial-data checks, without coherence recomputation."""
    n=np.asarray(n,dtype=int);phase=float(np.angle(np.vdot(ref.ravel(),pred.ravel())))
    aligned=pred*np.exp(-1j*phase);norm=np.sum(np.abs(pred)**2,axis=(1,2));observed={}
    for nx in (1024,2048):
        fields=[]
        for s in (aligned,ref):
            padded=np.zeros((len(s),nx,2),dtype=np.complex128)
            padded[:,n%nx]=s*np.exp(-1j*math.pi*n)[None,:,None]
            fields.append(np.fft.ifft(padded,axis=1)*nx/math.sqrt(length))
        p,r=fields;rho_p,rho_r=np.sum(np.abs(p)**2,axis=2),np.sum(np.abs(r)**2,axis=2)
        j_p,j_r=2*np.real(p[...,0].conj()*p[...,1]),2*np.real(r[...,0].conj()*r[...,1])
        observed[nx]=(float(np.linalg.norm(rho_p-rho_r)/max(np.linalg.norm(rho_r),1e-300)),
            float(np.linalg.norm(j_p-j_r)/max(np.linalg.norm(j_r),1e-300)),
            float(np.max(np.abs(length*rho_p.mean(axis=1)-norm))))
    return dict(initial_condition_max_abs_error=float(np.max(np.abs(pred[0]-ref[0]))),
        Parseval_max_error=observed[1024][2],
        density_1024_2048_difference=abs(observed[1024][0]-observed[2048][0]),
        current_1024_2048_difference=abs(observed[1024][1]-observed[2048][1]))

def compute(pred,ref,n,mass,length=16*math.pi,nx=1024):
    n=np.asarray(n,dtype=int);phase=float(np.angle(np.vdot(ref.ravel(),pred.ravel())));aligned=pred*np.exp(-1j*phase)
    def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
    def physical(spec):
        padded=np.zeros((len(spec),nx,2),dtype=np.complex128)
        padded[:,n%nx]=spec
        # Grid begins at x=-L/2. Constant Fourier factors must be included.
        padded[:,n%nx]*=np.exp(-1j*math.pi*n)[None,:,None]
        return ifft(padded,axis=1,norm='ortho',workers=1)*math.sqrt(nx/length)
    p,r=physical(aligned),physical(ref)
    rho_p,rho_r=np.sum(abs(p)**2,-1),np.sum(abs(r)**2,-1)
    j_p,j_r=2*(p[...,0].conj()*p[...,1]).real,2*(r[...,0].conj()*r[...,1]).real
    npred,nref=np.sum(abs(pred)**2,(1,2)),np.sum(abs(ref)**2,(1,2))
    fidelity=abs(np.sum(ref.conj()*pred,(1,2)))**2/np.maximum(npred*nref,1e-300)
    angle=np.arctan2(n/8,mass)/2;up=np.stack((np.cos(angle),np.sin(angle)),-1);um=np.stack((-np.sin(angle),np.cos(angle)),-1)
    def branches(spec):return np.stack((np.sum(spec*up[None],-1),np.sum(spec*um[None],-1)),-1)
    cp,cr=branches(aligned),branches(ref);pop_p,pop_r=abs(cp)**2,abs(cr)**2
    active=pop_r>=1e-6;defined=active & (pop_p>=1e-12);phase_error=np.angle(cp*cr.conj())
    numerator=denominator=0.;pairs=0
    for a,b,mask in zip(cp,cr,active):
        a,b,mask=a.ravel(),b.ravel(),mask.ravel();ids=np.flatnonzero(mask)
        aa,bb=a[ids],b[ids];ap=aa[:,None]*aa.conj()[None];bp=bb[:,None]*bb.conj()[None]
        m=~np.eye(len(ids),dtype=bool);numerator+=np.sum(abs(ap[m]-bp[m])**2);denominator+=np.sum(abs(bp[m])**2);pairs+=int(m.sum())
    return dict(spinor_rel_l2_raw=rel(pred,ref),spinor_rel_l2_gauge_aligned=rel(aligned,ref),current_rel_l2=rel(j_p,j_r),
        density_rel_l2=rel(rho_p,rho_r),fidelity_mean=float(fidelity.mean()),fidelity_min=float(fidelity.min()),
        modal_population_rel_l2=rel(np.sum(abs(pred)**2,-1),np.sum(abs(ref)**2,-1)),
        branch_population_rel_l2=rel(pop_p,pop_r),max_norm_drift=float(abs(npred-1).max()),
        active_coherence_rel_l2=float(np.sqrt(numerator/max(denominator,1e-300))),active_coherence_pair_count=pairs,
        phase_defined_reference_active_coverage=float(defined.sum()/max(active.sum(),1)),
        active_phase_circular_rmse=float(np.sqrt(np.mean(phase_error[defined]**2))) if defined.any() else None,
        global_spacetime_constant_phase=phase)

def self_test():
    import torch
    from closure.config import configuration
    from closure.vendor.r9_evaluate import field_metrics
    cfg=configuration('m5T5',(0,32));rng=np.random.default_rng(666666)
    ref=rng.normal(size=(3,33,2))+1j*rng.normal(size=(3,33,2));ref/=np.linalg.norm(ref,axis=(1,2))[:,None,None]
    pred=ref+.03*(rng.normal(size=ref.shape)+1j*rng.normal(size=ref.shape));t=np.array([0.,1.,5.])
    a=compute(pred,ref,np.array(cfg.indices),cfg.mass);b,_,_=field_metrics(pred,ref,cfg,t)
    errors={k:abs(a[k]-b[k]) for k in a if a[k] is not None}
    assert max(errors.values())<2e-12,errors
    assert compute(ref*np.exp(.9j),ref,np.array(cfg.indices),cfg.mass)['spinor_rel_l2_gauge_aligned']<2e-14
    # Norm-one incorrect dynamics must still yield large spinor/current errors.
    wrong=np.roll(ref,3,axis=1);c=compute(wrong,ref,np.array(cfg.indices),cfg.mass)
    assert c['max_norm_drift']<2e-14 and c['spinor_rel_l2_gauge_aligned']>.1
    return dict(status='PASS',independent_FFT_vs_direct_quadrature=errors,constant_phase_only_verified=True,structural_norm_counterexample_verified=True)

if __name__=='__main__':
    from plan import BASE,write
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):r=self_test()
    write(BASE/'engineering/independent_metric_tests.json',r);print(r)
